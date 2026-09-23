"""MUSE evidence and target-only supervision accounting; probabilities stay unchanged."""
from dataclasses import asdict, dataclass, replace
import math
import torch
from torch import nn
import torch.nn.functional as F

MUSE_VARIANTS = ('r2', 'hard', 'partial', 'starvation', 'gated', 'full')


@dataclass(frozen=True)
class MuseConfig:
    enabled: bool = True
    hard_confidence: float = .80
    hard_jsd: float = .08
    hard_margin: float = .20
    hard_min_votes: int = 2
    partial_top2_mass: float = .90
    partial_jsd: float = .15
    partial_min_top2_votes: int = 2
    tracker_ema: float = .95
    partial_effective_weight: float = .5
    starvation_lambda: float = 1.
    starvation_weight_max: float = 2.
    subgroup_gamma: float = 2.
    min_hard_support: float = 16.
    hard_support_full: float = 64.
    ot_enabled: bool = True
    ot_null_cost: float = .35
    ot_match_cost_threshold: float = .35
    ot_mass_threshold: float = .05
    ot_sinkhorn_epsilon: float = .05
    ot_sinkhorn_iters: int = 30
    lambda_hard: float = 1.
    lambda_partial: float = .5
    lambda_subgroup: float = .1
    partial_enabled: bool = True
    starvation_enabled: bool = True
    alignment_enabled: bool = True
    eps: float = 1e-8

    def validate(self):
        if any(not math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('MUSE config must be finite')
        if not 0 <= self.tracker_ema < 1 or not 0 < self.eps < 1:
            raise ValueError('Invalid MUSE EMA/epsilon')
        if min(self.hard_min_votes, self.partial_min_top2_votes, self.ot_sinkhorn_iters) < 1:
            raise ValueError('Invalid MUSE count')
        if self.ot_sinkhorn_epsilon <= 0 or not 0 <= self.min_hard_support <= self.hard_support_full or self.hard_support_full <= 0:
            raise ValueError('Invalid MUSE support/OT configuration')
        for name in ('hard_confidence','hard_margin','partial_top2_mass','partial_effective_weight'):
            if not 0 <= getattr(self, name) <= 1: raise ValueError('Invalid '+name)
        if min(self.hard_jsd, self.partial_jsd, self.subgroup_gamma, self.starvation_lambda,
               self.lambda_hard, self.lambda_partial, self.lambda_subgroup,
               self.ot_null_cost, self.ot_match_cost_threshold, self.ot_mass_threshold) < 0 or self.starvation_weight_max < 1:
            raise ValueError('MUSE weights and thresholds must be nonnegative')


def muse_config(variant='full', **overrides):
    if variant not in MUSE_VARIANTS: raise ValueError('Unknown MUSE variant')
    config = MuseConfig(**overrides)
    config = replace(config, enabled=config.enabled and variant != 'r2',
                     partial_enabled=config.partial_enabled and variant != 'hard',
                     starvation_enabled=config.starvation_enabled and variant in ('starvation','gated','full'),
                     alignment_enabled=config.alignment_enabled and variant in ('gated','full'),
                     ot_enabled=config.ot_enabled and variant == 'full')
    config.validate()
    return config


def muse_ramp(iteration, warmup=300, ramp_end=600):
    return min(max((iteration - warmup) / max(ramp_end - warmup, 1), 0.), 1.)


def assert_scale_trial_ids(batch):
    ids = batch.get('trial_key_by_scale')
    if ids is None or tuple(ids) != tuple(batch['x']):
        raise ValueError('MUSE requires actual trial keys for every scale')
    reference = torch.stack([batch[k] for k in ('subject_id','session_id','trial_id')], dim=-1)
    for scale, value in ids.items():
        if not torch.equal(value.cpu(), reference.cpu()):
            raise ValueError(f'Trial IDs do not align at scale {scale}')
        if len(batch['x'][scale]) != len(reference):
            raise ValueError('Scale batch sizes differ')


def reject_target_truth(batch):
    if any(k in batch for k in ('y', 'target_y', 'target_label', 'target_labels')):
        raise RuntimeError('MUSE target adaptation must be unlabeled')


@torch.no_grad()
def multiscale_evidence(probability, config):
    if probability.ndim != 3 or probability.shape[-1] < 2:
        raise ValueError('Expected independent probabilities [B,S,C], C>=2')
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError('Invalid teacher probabilities')
    if not torch.allclose(probability.sum(-1), torch.ones_like(probability[..., 0]), atol=1e-5):
        raise ValueError('Teacher probabilities must sum to one')
    p = probability.detach()
    q = p.mean(1)
    top, candidate = q.topk(2, dim=-1)
    votes = F.one_hot(p.argmax(-1), p.shape[-1]).sum(1).max(-1).values
    jsd = (p * (p.clamp_min(config.eps).log() - q[:,None].clamp_min(config.eps).log())).sum(-1).mean(-1)
    sets = p.topk(2, dim=-1).indices.sort(-1).values
    set_votes = (sets[:,:,None] == sets[:,None,:]).all(-1).sum(-1).max(-1).values
    hard = ((top[:,0] >= config.hard_confidence) & (jsd <= config.hard_jsd)
            & (top[:,0] - top[:,1] >= config.hard_margin) & (votes >= config.hard_min_votes))
    partial = (~hard & (top.sum(-1) >= config.partial_top2_mass) & (jsd <= config.partial_jsd)
               & (set_votes >= config.partial_min_top2_votes))
    if not config.partial_enabled: partial.zero_()
    return {'q':q, 'confidence':top[:,0], 'margin':top[:,0]-top[:,1], 'jsd':jsd,
            'votes':votes, 'top2_votes':set_votes, 'hard_label':q.argmax(-1), 'candidate_set':candidate,
            'hard_mask':hard, 'partial_mask':partial, 'unsupervised_mask':~(hard|partial)}


class TargetSupervisionTracker(nn.Module):
    def __init__(self, classes, config):
        super().__init__()
        self.config = config
        for name in ('M','E','H','P'):
            self.register_buffer(name, torch.zeros(classes))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, evidence):
        q = evidence['q']; classes = q.shape[-1]
        hard = torch.bincount(evidence['hard_label'][evidence['hard_mask']], minlength=classes).to(q)
        partial = torch.zeros_like(hard)
        sets = evidence['candidate_set'][evidence['partial_mask']]
        if sets.numel(): partial.scatter_add_(0, sets.flatten(), torch.full_like(sets.flatten(), .5, dtype=q.dtype))
        values = {'M':q.sum(0), 'H':hard, 'P':partial,
                  'E':hard + self.config.partial_effective_weight * partial}
        for name, value in values.items():
            getattr(self,name).lerp_(value.to(self.M), 1-self.config.tracker_ema)
        self.updates.add_(1)
        return {'hard_count':hard.tolist(), 'partial_count':partial.tolist(),
                'partial_effective_count':(self.config.partial_effective_weight * partial).tolist()}

    def coverage(self):
        return (self.E / (self.M + self.config.eps)).clamp(0,1)

    def starvation(self):
        return 1-self.coverage()

    def class_weights(self, ramp=1.):
        strength = self.config.starvation_lambda * ramp if self.config.starvation_enabled else 0.
        return (1+strength*self.starvation()).clamp(1,self.config.starvation_weight_max).detach()

    def gate(self, iteration, warmup=300, ramp_end=600):
        if not self.config.alignment_enabled or iteration < ramp_end:
            return torch.zeros_like(self.H)
        support = torch.where(self.H < self.config.min_hard_support, 0., (self.H/self.config.hard_support_full).clamp(max=1.))
        return (muse_ramp(iteration,warmup,ramp_end)*support*self.coverage().pow(self.config.subgroup_gamma)).detach()

    def summary(self, iteration, warmup=300, ramp_end=600):
        ramp=muse_ramp(iteration,warmup,ramp_end)
        return {'soft_mass':self.M.tolist(),'effective_supervision':self.E.tolist(),
                'hard_ema':self.H.tolist(),'partial_ema':self.P.tolist(),
                'hard_coverage':(self.H/(self.M+self.config.eps)).clamp(0,1).tolist(),
                'partial_coverage':(self.config.partial_effective_weight*self.P/(self.M+self.config.eps)).clamp(0,1).tolist(),
                'coverage':self.coverage().tolist(),'starvation':self.starvation().tolist(),
                'class_weight':self.class_weights(ramp).tolist(),
                'alignment_gate':self.gate(iteration,warmup,ramp_end).tolist()}


def target_supervision_loss(logits, evidence, weights, config):
    zero = logits.sum()*0
    hard=evidence['hard_mask'];partial=evidence['partial_mask'];labels=evidence['hard_label']
    if hard.any():
        values=F.cross_entropy(logits[hard].flatten(0,1),labels[hard,None].expand(-1,logits.shape[1]).flatten(),reduction='none')
        hard_loss=(values.reshape(-1,logits.shape[1]).mean(-1)*weights[labels[hard]]).mean()
    else: hard_loss=zero
    if partial.any():
        sets=evidence['candidate_set'][partial]
        mass=logits[partial].softmax(-1).gather(-1,sets[:,None].expand(-1,logits.shape[1],-1)).sum(-1)
        partial_loss=(-(mass+config.eps).log().mean(-1)*weights[sets].mean(-1)).mean()
    else: partial_loss=zero
    return hard_loss,partial_loss
