"""R4: one network, all-trial soft labels, source semantic compactness and alignment.

Source statistics use source truth only. Target records are keyed by unique trial;
no target truth, EMA network, fixed adaptation delay, or class-prior matching.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .subgroup_alignment import trial_keys

R4_VARIANTS = ('full', 'source_only', 'no_recovery', 'no_balance', 'no_compact', 'no_align')


@dataclass(frozen=True)
class FullPseudoConfig:
    variant: str = 'full'
    refresh_interval: int = 25
    pseudo_weight: float = .3
    compact_weight: float = .05
    align_weight: float = .1
    source_momentum: float = .9
    anchor_mix: float = .25
    anchor_temperature: float = .2
    max_class_weight: float = 3.
    compact_radius: float = .25
    separation_margin: float = 1.


def attach_semantic_projection(model):
    """Keep independent scale tokens; the classifier trains this task projection."""
    width = model.classifier.in_features
    layer = nn.Linear(width, width)
    nn.init.eye_(layer.weight)
    nn.init.zeros_(layer.bias)
    model.r4_semantic_projection = nn.Sequential(layer, nn.LayerNorm(width)).to(
        next(model.parameters()).device)


class FullPseudoAlignment:
    def __init__(self, model, config=FullPseudoConfig()):
        if config.variant not in R4_VARIANTS:
            raise ValueError('Unknown R4 variant')
        self.config = config
        device = next(model.parameters()).device
        self.centers = torch.zeros(len(model.scales), 3, model.classifier.in_features, device=device)
        self.ready = torch.zeros(len(model.scales), 3, dtype=torch.bool, device=device)
        self.confusion = torch.zeros(len(model.scales), 3, 3, device=device)
        self.rows = {}
        self.last_refresh = 0
        self.history = []

    @torch.no_grad()
    def observe_source(self, outputs, batches):
        z = F.normalize(torch.cat([o['scale_embeddings'].detach() for o in outputs]), dim=-1)
        logits = torch.cat([o['scale_logits'].detach() for o in outputs])
        y = torch.cat([b['y'].to(z.device) for b in batches])
        keys = [key for d, b in enumerate(batches) for key in trial_keys(b, d)]
        indices = list({key: i for i, key in enumerate(keys)}.values())
        z, logits, y = z[indices], logits[indices], y[indices]
        for s in range(z.shape[1]):
            matrix = torch.bincount(y * 3 + logits[:, s].argmax(-1), minlength=9).reshape(3, 3)
            self.confusion[s].lerp_(matrix.float(), 1 - self.config.source_momentum)
            for c in range(3):
                if (y == c).any():
                    center = z[y == c, s].mean(0)
                    if self.ready[s, c]:
                        self.centers[s, c].lerp_(center, 1 - self.config.source_momentum)
                    else:
                        self.centers[s, c].copy_(center)
                        self.ready[s, c] = True

    def competence(self):
        diagonal = self.confusion.diagonal(dim1=1, dim2=2)
        precision = diagonal / self.confusion.sum(1).clamp_min(1e-8)
        recall = diagonal / self.confusion.sum(2).clamp_min(1e-8)
        return 2 * precision * recall / (precision + recall).clamp_min(1e-8)

    @torch.no_grad()
    def distribution(self, output):
        probability = output['scale_logits'].softmax(-1)
        if self.config.variant == 'no_recovery':
            return probability.mean(1), probability.mean(1)
        reliability = self.competence().clamp_min(.1)
        q = (probability * reliability[None]).sum(1) / reliability.sum(0)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-8)
        anchors = F.normalize(self.centers, dim=-1)
        similarity = torch.einsum('nsd,scd->nsc', F.normalize(output['scale_embeddings'], dim=-1), anchors)
        anchor = (similarity / self.config.anchor_temperature).softmax(-1)
        anchor = (anchor * reliability[None]).sum(1) / reliability.sum(0)
        anchor = anchor / anchor.sum(-1, keepdim=True).clamp_min(1e-8)
        # A missing source class must never bias every target away from that class.
        if self.ready.all():
            mixed = q.pow(1 - self.config.anchor_mix) * anchor.clamp_min(1e-8).pow(self.config.anchor_mix)
            q = mixed / mixed.sum(-1, keepdim=True)
        return q, anchor

    @torch.no_grad()
    def refresh(self, model, loader, iteration):
        previous_mode = model.training
        model.eval()
        fresh = {}
        device = self.centers.device
        try:
            for batch in loader:
                if 'y' in batch:
                    raise RuntimeError('R4 target evidence must be unlabeled')
                x = {k: v.to(device) for k, v in batch['x'].items()}
                mask = {k: v.to(device) for k, v in batch['mask'].items()}
                output = model(x, mask, compute_domain=False)
                q, anchor = self.distribution(output)
                # A second weak view of the same network supplies perturbation evidence.
                perturbed = {k: v + .01 * torch.randn_like(v) * mask[k][..., None, None] for k, v in x.items()}
                q2, _ = self.distribution(model(perturbed, mask, compute_domain=False))
                stable = (1 - .5 * (q - q2).abs().sum(-1)).clamp(0, 1)
                certainty = (1 + (q * q.clamp_min(1e-8).log()).sum(-1) / math.log(3)).clamp(0, 1)
                margin = q.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
                quality = (q * self.competence().mean(0)[None]).sum(-1)
                anchor_support = (q * anchor).sum(-1)
                weight = certainty * margin * stable * quality * anchor_support
                for i, key in enumerate(trial_keys(batch, 0)):
                    old = self.rows.get(key)
                    temporal = 1. if old is None else max(0., 1 - .5 * float((q[i].cpu() - old['q']).abs().sum()))
                    fresh[key] = {'q': q[i].cpu(), 'reliability': float(weight[i]) * temporal,
                                  'label': int(q[i].argmax()), 'iteration': iteration}
            if len(fresh) != len(loader.dataset):
                raise RuntimeError('R4 evidence must cover every distinct target trial')
            self.rows = fresh
            self.last_refresh = iteration
            self.history.append(self.summary())
        finally:
            model.train(previous_mode)

    def class_weights(self):
        # Keep absolute reliability in the loss; normalization must not cancel it.
        mass = self.centers.new_zeros(3)
        for row in self.rows.values():
            mass[row['label']] += row['reliability']
        if self.config.variant == 'no_balance' or not (mass > 0).any():
            return torch.ones_like(mass)
        return (mass.mean() / mass.clamp_min(1e-6)).sqrt().clamp(.5, self.config.max_class_weight)

    def loss(self, source_outputs, source_batches, target_output, target_batch):
        if 'y' in target_batch:
            raise RuntimeError('R4 target batch must be unlabeled')
        z_s = F.normalize(torch.cat([o['scale_embeddings'] for o in source_outputs]), dim=-1)
        y_s = torch.cat([b['y'].to(z_s.device) for b in source_batches])
        z_t = F.normalize(target_output['scale_embeddings'], dim=-1)
        rows = [self.rows[k] for k in trial_keys(target_batch, 0)]
        q = torch.stack([r['q'] for r in rows]).to(z_t).detach()
        r = z_t.new_tensor([row['reliability'] for row in rows])
        b = self.class_weights()[q.argmax(-1)]
        ce = -(q * target_output['logits'].log_softmax(-1)).sum(-1)
        ce_scale = -(q[:, None] * target_output['scale_logits'].log_softmax(-1)).sum(-1).mean(-1)
        pseudo = (r * b * (ce + .3 * ce_scale)).mean()
        zero = z_t.sum() * 0
        compact_terms, align_terms = [], []
        current_centers = []
        for c in range(3):
            source = z_s[y_s == c]
            if len(source) == 0:
                continue
            center = source.mean(0)
            current_centers.append(center)
            if len(source) > 1:
                compact_terms.append(F.relu((source - center).square().sum(-1) - self.config.compact_radius).mean())
            # Only equal coarse labels align. Uncertain samples stay soft in classification.
            w = r * q[:, c] * (q.argmax(-1) == c)
            mass = w.sum()
            if mass.detach() > 1e-6:
                mean_t = (z_t * w[:, None, None]).sum(0) / mass
                var_t = ((z_t - mean_t).square() * w[:, None, None]).sum(0) / mass
                var_s = (source - center).square().mean(0)
                discrepancy = (center - mean_t).square().sum(-1).mean() + (var_s - var_t).square().sum(-1).mean()
                # Absolute support prevents a single unreliable target gaining full force.
                align_terms.append(discrepancy * (mass / len(z_t)).detach())
        for i, center in enumerate(current_centers):
            for other in current_centers[i + 1:]:
                compact_terms.append(F.relu(self.config.separation_margin - (center - other).square().sum(-1)).mean())
        compact = torch.stack(compact_terms).mean() if compact_terms else zero
        align = torch.stack(align_terms).mean() if align_terms else zero
        variant = self.config.variant
        total = zero
        if variant != 'source_only':
            total = self.config.pseudo_weight * pseudo
            if variant != 'no_compact':
                total = total + self.config.compact_weight * compact
            if variant != 'no_align':
                total = total + self.config.align_weight * align
        return total, {'pseudo': float(pseudo.detach()), 'compact': float(compact.detach()),
                       'align': float(align.detach()), 'weighted_loss': float(total.detach()),
                       'mean_reliability': float(r.mean()), 'class_weights': self.class_weights().tolist(),
                       **self.summary()}

    def summary(self):
        counts, mass = [0] * 3, [0.] * 3
        for row in self.rows.values():
            counts[row['label']] += 1
            mass[row['label']] += row['reliability']
        return {'refresh_iteration': self.last_refresh, 'unique_target_trials': len(self.rows),
                'pseudo_count_by_class': counts, 'reliable_mass_by_class': mass}

    def state(self):
        return {'config': asdict(self.config), 'teacher': False, 'adaptation_warmup': 0,
                'scope': 'current_target_subject_all_unique_trials; no_target_truth',
                'source_competence_scope': 'online_source_training_confusion_proxy',
                'history': self.history, 'source_competence': self.competence().tolist(),
                'pseudo_labels': [{'trial_key': list(k), **{name: value.tolist() if torch.is_tensor(value) else value
                                  for name, value in row.items()}} for k, row in sorted(self.rows.items())]}
