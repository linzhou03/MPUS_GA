"""Prototype-free feature alignment inspired by CRCo (CVPR 2023), not a reproduction.

CRCo's p R p' relationship becomes detached sample-pair supervision of feature
similarities. R comes from source-label-conditioned prediction statistics, never
classifier weights, feature centroids, clustering or target truth.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

REFERENCE = 'https://github.com/zhyx12/CRCo'
REFERENCE_COMMIT = 'd4611eb0bdb8c10094df149fe3b7a2b2a544dd3f'


@dataclass(frozen=True)
class RelationConfig:
    alignment_weight: float = .2
    instance_weight: float = .1
    temperature: float = .2
    neighbor_temperature: float = .2
    neighbors_per_class: int = 3
    inter_class_weight: float = .25
    source_momentum: float = .95
    warmup: int = 300
    ramp_end: int = 600
    channel_dropout: float = .1
    noise_std: float = .02
    negative_floor: float = .1


def trial_ids(batch):
    return list(zip(*(batch[k].detach().cpu().tolist()
                      for k in ('subject_id', 'session_id', 'trial_id'))))


def unique_rows(keys, device):
    seen, rows = set(), []
    for index, key in enumerate(keys):
        if key not in seen:
            seen.add(key); rows.append(index)
    return torch.tensor(rows, dtype=torch.long, device=device)


def encode_view(model, batch, rows, device, config):
    """Same electrodes are masked across all times/scales of a trial."""
    keep = None
    pooled = []
    for key in model.scale_keys:
        x = batch['x'][key].to(device)[rows]
        mask = batch['mask'][key].to(device)[rows]
        if keep is None:
            keep = (torch.rand(len(rows), 1, x.shape[-2], 1, device=device) >= config.channel_dropout)
            # A valid view always retains at least one channel.
            keep[:, :, 0] = True
        view = (x * keep + config.noise_std * torch.randn_like(x) * keep) * mask[:, :, None, None]
        sequence = model.encode_window_sequence(view, mask, key)
        pooled.append(model.temporal[key].pool_sequence(sequence, mask))
    return model.scale_output_norm(torch.stack(pooled, 1) + model.scale_embedding[None])


@torch.no_grad()
def sample_targets(similarity, probability, labels, relation, config):
    """Distribute each class's CRS mass to its nearest individual source trials.

    No averaging of source feature vectors. Different target trials can match
    different modes of a source class. Missing classes receive no invented keys.
    """
    class_mass = probability.detach() @ relation.detach()
    present = torch.stack([(labels == c).any() for c in range(3)])
    class_mass = class_mass * present
    class_mass = class_mass / class_mass.sum(-1, keepdim=True).clamp_min(1e-8)
    weights = torch.zeros_like(similarity)
    for c in range(3):
        members = (labels == c).nonzero().flatten()
        if not len(members):
            continue
        scores = similarity[:, members].detach()
        chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, :min(config.neighbors_per_class, len(members))]
        mass = (scores.gather(1, chosen) / config.neighbor_temperature).softmax(-1) * class_mass[:, c, None]
        weights.scatter_add_(1, members[chosen], mass)
    return weights


def cross_domain_loss(source_z, source_y, target_z, probability, relation, reliability, config):
    source = F.normalize(source_z.detach(), dim=-1)
    target = F.normalize(target_z, dim=-1)
    losses = []
    present = torch.stack([(source_y == c).any() for c in range(3)])
    for s in range(target.shape[1]):
        cosine = target[:, s] @ source[:, s].T
        desired = sample_targets(cosine, probability[:, s], source_y, relation[s], config)
        per_trial = -(desired * (cosine / config.temperature).log_softmax(-1)).sum(-1)
        # Do not renormalize by sum(reliability); ambiguous batches exert less force.
        support = probability[:, s, present].sum(-1).detach()
        losses.append((per_trial * reliability * support).mean())
    return torch.stack(losses).mean()


def instance_loss(weak, strong, probability, relation, config):
    if len(weak) < 2:
        return (weak.sum() + strong.sum()) * 0.
    # CR similarity reduces false-negative pressure for semantically related trials.
    affinity = torch.einsum('isc,scd,jsd->ij', probability.detach(), relation.detach(), probability.detach()) / probability.shape[1]
    negative_weight = (1 - affinity).clamp(min=config.negative_floor, max=1.)
    negative_weight.fill_diagonal_(1.)
    log_weight = negative_weight.log().detach()
    a, b = F.normalize(weak, dim=-1), F.normalize(strong, dim=-1)
    losses = []
    for s in range(a.shape[1]):
        for query, key in ((a[:, s], b[:, s].detach()), (b[:, s], a[:, s].detach())):
            logits = query @ key.T / config.temperature
            losses.append(((logits + log_weight).logsumexp(-1) - logits.diag()).mean())
    return torch.stack(losses).mean()


class RelationAlignment:
    def __init__(self, config, device, scales=3):
        self.config = config
        self.profile = torch.eye(3, device=device).repeat(scales, 1, 1)
        self.source_counts = torch.zeros(3, device=device, dtype=torch.long)
        self.initialized = torch.zeros(3, device=device, dtype=torch.bool)
        self.last_record = {}
        self.active_steps = 0

    @torch.no_grad()
    def observe_source(self, probabilities, labels, iteration):
        if iteration > self.config.warmup:
            return
        for c in range(3):
            selected = labels == c
            if selected.any():
                value = probabilities[selected].detach().mean(0)
                m = self.config.source_momentum if self.initialized[c] else 0.
                self.profile[:, c] = m * self.profile[:, c] + (1-m) * value
                self.initialized[c] = True
                self.source_counts[c] += selected.sum()

    @torch.no_grad()
    def relation(self):
        profile = F.normalize(self.profile, dim=-1)
        affinity = profile @ profile.transpose(-1, -2)
        eye = torch.eye(3, device=profile.device)[None]
        valid = self.initialized[:, None] & self.initialized[None, :]
        return eye + self.config.inter_class_weight * affinity * (1-eye) * valid

    def loss(self, model, outputs, batches, target_output, target_batch, iteration, device):
        if 'y' in target_batch:
            raise RuntimeError('Target truth must not enter feature alignment')
        c = self.config
        ramp = min(1., max(0., (iteration-c.warmup)/(c.ramp_end-c.warmup)))
        source_keys = [(d, *key) for d, batch in enumerate(batches) for key in trial_ids(batch)]
        si = unique_rows(source_keys, device)
        ti = unique_rows(trial_ids(target_batch), device)
        zs = torch.cat([o['scale_embeddings'] for o in outputs])[si]
        ps = torch.cat([o['scale_logits'].detach().softmax(-1) for o in outputs])[si]
        ys = torch.cat([b['y'].to(device) for b in batches])[si]
        self.observe_source(ps, ys, iteration)
        zero = target_output['scale_embeddings'].sum() * 0.
        record = dict(active=ramp>0, ramp=ramp, alignment=0., instance=0., added_loss=0.,
            unique_source=len(si), unique_target=len(ti), mean_reliability=0.)
        if ramp <= 0:
            self.last_record = record
            return zero
        zt = target_output['scale_embeddings'][ti]
        q = target_output['calibrated_scale_logits'][ti].detach().softmax(-1)
        avg = q.mean(1)
        certainty = (1 + (avg * avg.clamp_min(1e-8).log()).sum(-1)/math.log(3)).clamp(0,1)
        jsd = (q * (q.clamp_min(1e-8).log()-avg[:, None].clamp_min(1e-8).log())).sum(-1).mean(-1)
        reliability = (certainty * torch.exp(-jsd)).detach()
        relation = self.relation()
        alignment = cross_domain_loss(zs, ys, zt, q, relation, reliability, c)
        strong = encode_view(model, target_batch, ti, device, c)
        instance = instance_loss(zt, strong, q, relation, c)
        added = ramp * (c.alignment_weight * alignment + c.instance_weight * instance)
        if not torch.isfinite(added):
            raise FloatingPointError('Non-finite class relationship alignment')
        self.active_steps += 1
        record.update(alignment=float(alignment.detach()), instance=float(instance.detach()),
            added_loss=float(added.detach()), mean_reliability=float(reliability.mean()),
            relation=relation.cpu().tolist())
        self.last_record = record
        return added

    def state(self):
        return dict(config=asdict(self.config), relation=self.relation().cpu().tolist(),
            source_profile=self.profile.cpu().tolist(), source_observation_counts=self.source_counts.cpu().tolist(),
            active_steps=self.active_steps, feature_prototypes=False, target_truth_used=False,
            teacher=False, source_relation_frozen_after=self.config.warmup,
            reference=REFERENCE, reference_commit=REFERENCE_COMMIT,
            scope='CRCo-inspired source-access feature-pair adaptation; not original SFUDA reproduction')
