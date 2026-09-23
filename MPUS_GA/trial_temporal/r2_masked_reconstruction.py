"""Original R2 classification with a bounded, training-only masked objective."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .anatomical_prior import region_weight_matrix_torch
from .masked_multiscale import SCALE_SLICES, pack_windows, coordinated_mask, masked_reconstruction_loss


@dataclass(frozen=True)
class R2MaskConfig:
    source_weight: float = .05
    target_weight: float = .05
    auxiliary_batch_size: int = 4
    max_encoder_gradient_ratio: float = .20
    time_mask_probability: float = .15
    spatial_chunk_windows: int = 256

    def validate(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('R2 mask configuration must be finite')
        if min(self.source_weight, self.target_weight) < 0 or not 0 <= self.max_encoder_gradient_ratio <= 1:
            raise ValueError('Invalid reconstruction weights or gradient bound')
        if min(self.auxiliary_batch_size, self.spatial_chunk_windows) < 1 or not 0 <= self.time_mask_probability < 1:
            raise ValueError('Invalid mask/chunk settings')


def bounded_auxiliary_gradients(base, auxiliary, ratio):
    """Project conflicting encoder gradients, then bound their Euclidean norm.

    This protects only the first-order R2 loss direction before AdamW; it does
    not guarantee downstream accuracy or monotonic loss after an optimizer step.
    """
    zero = base[0].new_zeros(())
    base2 = sum((g.square().sum() for g in base), zero)
    aux2 = sum((g.square().sum() for g in auxiliary), zero)
    dot = sum(((a * b).sum() for a, b in zip(base, auxiliary)), zero)
    if not torch.isfinite(base2 + aux2 + dot):
        raise FloatingPointError('Non-finite reconstruction gradients')
    coefficient = dot.clamp_max(0.) / base2.clamp_min(1e-20)
    projected = [a - coefficient * b for b, a in zip(base, auxiliary)]
    norm = sum((g.square().sum() for g in projected), zero).sqrt()
    factor = (ratio * base2.sqrt() / norm.clamp_min(1e-20)).clamp(max=1.)
    applied = [g * factor for g in projected]
    after_dot = sum(((a * b).sum() for a, b in zip(base, applied)), zero)
    applied_norm = norm * factor
    return applied, {'base_encoder_gradient_norm': float(base2.sqrt()),
                     'raw_auxiliary_gradient_norm': float(aux2.sqrt()),
                     'applied_auxiliary_gradient_norm': float(applied_norm),
                     'applied_gradient_ratio': float(applied_norm / base2.sqrt().clamp_min(1e-20)),
                     'gradient_dot_before': float(dot), 'gradient_dot_after': float(after_dot),
                     'conflict_projected': bool(dot < 0)}


class R2MaskedReconstruction(nn.Module):
    def __init__(self, d_model, heads, config=R2MaskConfig()):
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer('regions', region_weight_matrix_torch())
        self.mask_value = nn.Parameter(torch.zeros(62, 5))
        self.slot_identity = nn.Parameter(torch.randn(7, d_model) * .02)
        self.group_token = nn.Parameter(torch.randn(1, 1, d_model) * .02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, .1,
                                            batch_first=True, norm_first=True, activation='gelu')
        self.local_scale_context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.decoder = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, 310))
        self.steps = self.source_active = self.target_active = self.gradient_active = self.conflicts = 0
        self.target_seen = set()
        self.predictions = []

    def reconstruction(self, model, x, masks, hidden=None):
        values, valid = pack_windows(x, masks)
        if hidden is None:
            hidden = coordinated_mask(valid, self.regions, self.config.time_mask_probability)
        corrupted = torch.where(hidden, self.mask_value, values)
        b, groups = valid.shape[:2]
        features = []
        for scale, slots in SCALE_SLICES.items():
            width = slots.stop - slots.start
            length = masks[scale].shape[1]
            windows = corrupted[:, :, slots].reshape(b, groups * width, 62, 5)[:, :length]
            sequence = model.encode_window_sequence(windows, masks[scale], scale)
            sequence = F.pad(sequence, (0, 0, 0, groups * width - length))
            features.append(sequence.reshape(b, groups, width, -1))
        tokens = torch.cat(features, 2) + self.slot_identity
        tokens = tokens.reshape(b * groups, 7, -1)
        tokens = torch.cat((self.group_token.expand(b * groups, -1, -1), tokens), 1)
        local_valid = torch.cat((torch.ones(b * groups, 1, device=valid.device, dtype=torch.bool),
                                 valid.reshape(b * groups, 7)), 1)
        context = self.local_scale_context(tokens, src_key_padding_mask=~local_valid).reshape(b, groups, 8, -1)
        global_token = context[:, :, :1].expand(-1, -1, 7, -1)
        prediction = self.decoder(torch.cat((context[:, :, 1:], global_token), -1)).reshape_as(values)
        loss, by_scale = masked_reconstruction_loss(prediction, values, hidden)
        return loss, {'by_scale': by_scale, 'masked_fraction': float(hidden.sum() / (valid.sum() * 310)),
                      'trial_count': b}, prediction

    def accumulate(self, model, source_batches, target_batch, device):
        if any(k in target_batch for k in ('y', 'target_y', 'target_labels', 'labels', 'target_label')):
            raise RuntimeError('R2 masked reconstruction must not receive target truth')
        if len(source_batches) != 1:
            raise ValueError('R2+MSMR v1 uses one source dataset per direction')
        shared = list(model.spatial.parameters()) + list(model.temporal.parameters())
        own = list(self.parameters())
        params = shared + own
        summed = [None] * len(params)
        losses, details = {}, {}
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        # Additional masks/dropout do not consume the clean R2 branch's RNG stream.
        with torch.random.fork_rng(devices=cuda_devices):
            for name, batch, weight in [('source', source_batches[0], self.config.source_weight),
                                         ('target', target_batch, self.config.target_weight)]:
                n = min(self.config.auxiliary_batch_size, next(iter(batch['x'].values())).shape[0])
                x = {s: v[:n].to(device) for s, v in batch['x'].items()}
                masks = {s: v[:n].to(device) for s, v in batch['mask'].items()}
                if weight == 0:
                    losses[name], details[name] = 0., {'trial_count': 0}
                    continue
                loss, info, _ = self.reconstruction(model, x, masks)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite reconstruction loss')
                gradients = torch.autograd.grad(weight * loss, params, allow_unused=True)
                for i, gradient in enumerate(gradients):
                    if gradient is not None:
                        summed[i] = gradient.detach() if summed[i] is None else summed[i] + gradient.detach()
                losses[name], details[name] = float(loss.detach()), info
                if name == 'target':
                    self.target_seen.update(zip(*(batch[k][:n].tolist() for k in ('subject_id', 'session_id', 'trial_id'))))
        active = [i for i in range(len(shared)) if summed[i] is not None]
        if active:
            base = [shared[i].grad.detach() if shared[i].grad is not None else torch.zeros_like(shared[i]) for i in active]
            applied, diagnostics = bounded_auxiliary_gradients(base, [summed[i] for i in active], self.config.max_encoder_gradient_ratio)
            for i, gradient in zip(active, applied):
                shared[i].grad = gradient if shared[i].grad is None else shared[i].grad + gradient
        else:
            diagnostics = {'applied_auxiliary_gradient_norm': 0., 'applied_gradient_ratio': 0., 'conflict_projected': False}
        for i, parameter in enumerate(own, len(shared)):
            if summed[i] is not None:
                parameter.grad = summed[i] if parameter.grad is None else parameter.grad + summed[i]
        self.steps += 1
        self.source_active += losses['source'] > 0
        self.target_active += losses['target'] > 0
        self.gradient_active += diagnostics['applied_auxiliary_gradient_norm'] > 0
        self.conflicts += diagnostics['conflict_projected']
        return {'source_reconstruction': losses['source'], 'target_reconstruction': losses['target'],
                'added_loss': self.config.source_weight * losses['source'] + self.config.target_weight * losses['target'],
                'source_mask': details['source'], 'target_mask': details['target'], **diagnostics}

    def state(self):
        return {'config': asdict(self.config), 'steps': self.steps,
                'source_reconstruction_active_steps': self.source_active,
                'target_reconstruction_active_steps': self.target_active,
                'encoder_gradient_active_steps': self.gradient_active, 'conflict_projection_steps': self.conflicts,
                'unique_target_trials_reconstructed': len(self.target_seen),
                'scope': 'training_only; original_R2_predictions; no_pseudo_label_admission',
                'gradient_rule': 'project_negative_R2_dot_then_cap_encoder_norm; no guarantee after AdamW'}


def attach_reconstruction(model, heads, config=R2MaskConfig()):
    # Keep the base model initialization and clean-training random state intact.
    with torch.random.fork_rng(devices=[]):
        model.r2_mask_reconstruction = R2MaskedReconstruction(model.d_model, heads, config).to(next(model.parameters()).device)
    model.activation_checkpointing = True
    model.spatial_chunk_windows = config.spatial_chunk_windows
    return model.r2_mask_reconstruction
