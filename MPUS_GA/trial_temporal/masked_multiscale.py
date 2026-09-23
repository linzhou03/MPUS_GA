"""MSMR v1: aligned local scale interaction and masked DE reconstruction.

This is an independent experimental model, not a modification of R2's banks.
Every four-second interval contains up to seven windows; partial tails survive.
All overlapping views hide the same region/band cells to prevent direct copies.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .anatomical_prior import region_weight_matrix_torch

SCALE_SLICES = {'1s': slice(0, 4), '2s': slice(4, 6), '4s': slice(6, 7)}


@dataclass(frozen=True)
class MSMRConfig:
    d_model: int = 64
    heads: int = 4
    temporal_layers: int = 2
    dropout: float = .20
    source_batch_size: int = 8
    target_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    source_balance_alpha: float = .4
    label_smoothing: float = .1
    source_reconstruction_weight: float = .1
    target_reconstruction_weight: float = .1
    time_mask_probability: float = .15
    cuda_memory_budget_gib: float = 8.

    def validate(self):
        if any(not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError('MSMR configuration must be finite')
        if self.d_model < 8 or self.d_model % self.heads or self.d_model % 2:
            raise ValueError('d_model must be even and divisible by heads')
        if min(self.source_batch_size, self.target_batch_size, self.temporal_layers) < 1:
            raise ValueError('Batch sizes and layer count must be positive')
        if not 0 <= self.dropout < 1 or not 0 <= self.time_mask_probability < 1:
            raise ValueError('Invalid dropout or mask probability')
        if min(self.learning_rate, self.source_reconstruction_weight,
               self.target_reconstruction_weight, self.cuda_memory_budget_gib) <= 0:
            raise ValueError('All three training objectives must be enabled')


def pack_windows(x, masks):
    """[B,T_s,62,5] -> [B,ceil(T_1/4),7,62,5], with no temporal resizing."""
    if set(x) != set(SCALE_SLICES) or set(masks) != set(SCALE_SLICES):
        raise ValueError('MSMR requires exactly the aligned 1s/2s/4s scales')
    batch, length = masks['1s'].shape
    groups = (length + 3) // 4
    counts = {s: m.sum(1) for s, m in masks.items()}
    if not torch.equal(counts['1s'] // 2, counts['2s']) or not torch.equal(counts['1s'] // 4, counts['4s']):
        raise ValueError('Scale lengths are inconsistent with non-overlapping windows')
    blocks, valid = [], []
    for scale, slots in SCALE_SLICES.items():
        width = slots.stop - slots.start
        values, mask = x[scale], masks[scale]
        if values.shape != (*mask.shape, 62, 5) or mask.dtype != torch.bool:
            raise ValueError('Invalid feature or validity-mask shape')
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < counts[scale][:, None]
        if not torch.equal(mask, expected) or (counts[scale] < 1).any():
            raise ValueError('Valid windows must form a nonempty contiguous prefix')
        pad = groups * width - values.shape[1]
        if pad < 0:
            raise ValueError('Padded scale length exceeds aligned group length')
        blocks.append(F.pad(values, (0, 0, 0, 0, 0, pad)).reshape(batch, groups, width, 62, 5))
        valid.append(F.pad(mask, (0, pad)).reshape(batch, groups, width))
    return torch.cat(blocks, 2), torch.cat(valid, 2)


def coordinated_mask(valid, region_weights, time_probability=.15, generator=None):
    """Mask two regions plus one frequency band per group, jointly across scales.

    Some complete time groups are additionally hidden. Group zero always has
    observed cells, including for a single-group trial. Nothing uses target y.
    """
    b, g, _ = valid.shape
    kwargs = {'device': valid.device, 'generator': generator}
    regions = region_weights.shape[0]
    order = torch.rand(b, g, regions, **kwargs).argsort(-1)
    selected_regions = torch.zeros(b, g, regions, dtype=torch.bool, device=valid.device)
    selected_regions.scatter_(-1, order[..., :2], True)
    channel_region = region_weights.argmax(0)
    channel_hidden = selected_regions[..., channel_region]
    band = torch.randint(5, (b, g, 1), **kwargs)
    band_hidden = torch.zeros(b, g, 5, dtype=torch.bool, device=valid.device).scatter_(-1, band, True)
    hidden = channel_hidden[..., None] | band_hidden[..., None, :]
    time_hidden = torch.rand(b, g, **kwargs) < time_probability
    time_hidden[:, 0] = False
    hidden |= time_hidden[..., None, None]
    return hidden[:, :, None].expand(-1, -1, 7, -1, -1) & valid[..., None, None]


def encoder_layer(d, heads, dropout):
    return nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation='gelu',
                                      batch_first=True, norm_first=True)


class RegionWindowEncoder(nn.Module):
    """Explicit electrode order plus attention between eight fixed EEG regions."""
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.register_buffer('regions', region_weight_matrix_torch())
        self.raw = nn.Linear(620, d)
        self.region_projection = nn.Linear(10, d)
        self.region_identity = nn.Parameter(torch.randn(8, d) * .02)
        self.region_attention = nn.TransformerEncoder(encoder_layer(d, heads, dropout), 1,
                                                     enable_nested_tensor=False)
        self.fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, values, hidden):
        # Values have already been corrupted before any normalization or pooling.
        cells = torch.cat((values, hidden.to(values.dtype)), -1)
        raw = self.raw(cells.flatten(-2))
        regional = torch.einsum('rc,ncf->nrf', self.regions, cells)
        tokens = self.region_projection(regional) + self.region_identity
        regional = self.region_attention(tokens).mean(1)
        return self.fusion(torch.cat((raw, regional), -1))


class MaskedMultiScaleModel(nn.Module):
    def __init__(self, config=MSMRConfig()):
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.spatial = RegionWindowEncoder(d, config.heads, config.dropout)
        self.mask_value = nn.Parameter(torch.zeros(62, 5))
        self.slot_identity = nn.Parameter(torch.randn(7, d) * .02)
        self.local_cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        self.local = nn.TransformerEncoder(encoder_layer(d, config.heads, config.dropout), 1,
                                          enable_nested_tensor=False)
        self.temporal = nn.TransformerEncoder(encoder_layer(d, config.heads, config.dropout),
                                             config.temporal_layers, enable_nested_tensor=False)
        self.pool_score = nn.Linear(d, 1)
        self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.classifier = nn.Linear(d, 3)
        self.decoder = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 310))

    def encode_packed(self, values, valid, hidden):
        b, g, slots = valid.shape
        d = self.config.d_model
        corrupted = torch.where(hidden, self.mask_value, values)
        spatial = values.new_zeros(b, g, slots, d)
        spatial[valid] = self.spatial(corrupted[valid], hidden[valid])
        tokens = (spatial + self.slot_identity).reshape(b * g, slots, d)
        tokens = torch.cat((self.local_cls.expand(b * g, -1, -1), tokens), 1)
        local_valid = torch.cat((torch.ones(b * g, 1, dtype=torch.bool, device=valid.device),
                                 valid.reshape(b * g, slots)), 1)
        local = self.local(tokens, src_key_padding_mask=~local_valid).reshape(b, g, 8, d)
        group_valid = valid.any(-1)
        positions = torch.arange(g, device=values.device, dtype=values.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, d, 2, device=values.device, dtype=values.dtype)
                              * (-math.log(10000.) / d))
        pe = values.new_zeros(g, d)
        pe[:, 0::2], pe[:, 1::2] = (positions * frequency).sin(), (positions * frequency).cos()
        sequence = self.temporal(local[:, :, 0] + pe, src_key_padding_mask=~group_valid)
        sequence = sequence.masked_fill(~group_valid[..., None], 0.)
        weight = self.pool_score(sequence).squeeze(-1).masked_fill(~group_valid, -torch.inf).softmax(1)
        mean = sequence.sum(1) / group_valid.sum(1, keepdim=True)
        feature = self.pool(torch.cat(((sequence * weight[..., None]).sum(1), mean), -1))
        return feature, local[:, :, 1:], sequence

    def forward(self, x, masks):
        values, valid = pack_windows(x, masks)
        feature, _, _ = self.encode_packed(values, valid, torch.zeros_like(values, dtype=torch.bool))
        return {'logits': self.classifier(feature), 'features': feature}

    def reconstruct(self, x, masks, generator=None):
        values, valid = pack_windows(x, masks)
        hidden = coordinated_mask(valid, self.spatial.regions, self.config.time_mask_probability, generator)
        _, local, global_sequence = self.encode_packed(values, valid, hidden)
        context = global_sequence[:, :, None].expand_as(local)
        prediction = self.decoder(torch.cat((local, context), -1)).reshape_as(values)
        loss, per_scale = masked_reconstruction_loss(prediction, values, hidden)
        return loss, {'by_scale': per_scale, 'masked_fraction': float(hidden.sum() / (valid.sum() * 310)),
                      'masked_cells': int(hidden.sum())}


def masked_reconstruction_loss(prediction, target, hidden):
    """Average masked cells within each trial/scale, then scales, then trials."""
    error = F.smooth_l1_loss(prediction, target.detach(), reduction='none')
    by_trial, diagnostics = [], {}
    for scale, slots in SCALE_SLICES.items():
        mask = hidden[:, :, slots]
        count = mask.flatten(1).sum(1)
        if (count == 0).any():
            raise ValueError('Each trial/scale needs masked supervision')
        values = (error[:, :, slots] * mask).flatten(1).sum(1) / count
        by_trial.append(values)
        diagnostics[scale] = float(values.detach().mean())
    return torch.stack(by_trial, 1).mean(), diagnostics
