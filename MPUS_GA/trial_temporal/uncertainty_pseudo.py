"""Prototype-free target k-NN voting and entropy-weighted positive CE.

Component adaptation of Litrico et al., CVPR 2023, not the full SFUDA method.
Probability history is an arithmetic mean of unreﬁned full-bank refreshes;
there is no momentum teacher, negative learning, or contrastive objective.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
import math

import torch
import torch.nn.functional as F

from .neighbor_soft import passive_inference
from .pcdiag import cpu, identities

REFERENCE_COMMIT = '1cf5184f3d20fb016206bf5b3fe8397367699053'


@dataclass(frozen=True)
class UncertaintyConfig:
    neighbors: int = 10
    history_refreshes: int = 3
    refresh_interval: int = 50
    weight: float = 1.
    activation_checkpointing: bool = True
    spatial_chunk_windows: int = 256

    def __post_init__(self):
        if min(self.neighbors, self.history_refreshes, self.refresh_interval) < 1 or self.weight < 0:
            raise ValueError('Invalid uncertainty pseudo-label configuration')


@torch.no_grad()
def entropy_weights(probability):
    p = probability.detach().float()
    entropy = -(p * p.clamp_min(1e-8).log()).sum(-1) / math.log(p.shape[-1])
    return (-entropy.clamp(0., 1.)).exp()


def weighted_ce(logits, probability):
    """Batch mean, NOT sum(weights)-normalized; gradients only enter logits."""
    p = probability.detach()
    return (entropy_weights(p) * F.cross_entropy(logits, p.argmax(-1), reduction='none')).mean()


@torch.no_grad()
def collect_evidence(model, loader, device, context):
    from .train import _batch_to_device
    values = {}
    with passive_inference(model, device):
        for batch in loader:
            if 'y' in batch:
                raise RuntimeError('Target truth must not enter the feature bank')
            x, mask = _batch_to_device(batch, device)
            out = model(x, mask, **context)
            row = dict(ids=identities(batch), embeddings=out['scale_embeddings'],
                       raw_probability=out['calibrated_scale_logits'].softmax(-1).mean(1),
                       fused=out['probability'], scale_logits=out['calibrated_scale_logits'])
            for key, value in row.items():
                values.setdefault(key, []).append(cpu(value))
    if not values:
        raise ValueError('Empty target evidence')
    return {key: torch.cat(value) for key, value in values.items()}


class TrialBank:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=config.history_refreshes)
        self.history_iterations = deque(maxlen=config.history_refreshes)
        self.ids = self.features = self.probability = None
        self.iteration = None

    @torch.no_grad()
    def update(self, evidence, iteration):
        ids, z, p = (cpu(evidence[k]) for k in ('ids', 'embeddings', 'raw_probability'))
        if len(ids) < 2 or len({tuple(x) for x in ids.tolist()}) != len(ids):
            raise ValueError('Bank requires unique target trials and at least two trials')
        if not torch.isfinite(z).all() or not torch.isfinite(p).all():
            raise FloatingPointError('Non-finite bank evidence')
        if p.ndim != 2 or len(p) != len(ids) or len(z) != len(ids):
            raise ValueError('Inconsistent bank shapes')
        if torch.any(p < 0) or not torch.allclose(p.sum(-1), torch.ones(len(p)), atol=1e-5):
            raise ValueError('Expected probability distributions')
        if self.ids is not None and not torch.equal(ids, self.ids):
            raise ValueError('Target trial order changed during history accumulation')
        if self.iteration is not None and iteration <= self.iteration:
            raise ValueError('Bank refresh iterations must increase')
        self.ids, self.features = ids.clone(), F.normalize(z.float(), dim=-1)
        self.history.append(p.clone())
        self.history_iterations.append(int(iteration))
        self.probability = torch.stack(tuple(self.history)).mean(0)
        self.iteration = int(iteration)

    @torch.no_grad()
    def query(self, features, ids):
        if self.ids is None:
            raise RuntimeError('Bank has not been initialized')
        device = features.device
        query = F.normalize(features.detach().float(), dim=-1)
        same = (ids.detach().cpu()[:, None] == self.ids[None]).all(-1)
        if not torch.all(same.sum(-1) == 1):
            raise ValueError('Every query identity must occur exactly once in target bank')
        similarity = torch.einsum('isd,jsd->ij', query, self.features.to(device)) / query.shape[1]
        similarity.masked_fill_(same.to(device), -torch.inf)
        k = min(self.config.neighbors, len(self.ids) - 1)
        indices = similarity.argsort(dim=1, descending=True, stable=True)[:, :k]
        probability = self.probability.to(device)[indices].mean(1)
        return probability, indices

    def state(self):
        return dict(ids=self.ids, features=self.features, probability=self.probability,
                    history=list(self.history), history_iterations=list(self.history_iterations),
                    iteration=self.iteration)


class UncertaintyPseudo:
    def __init__(self, config, evidence_loader, device):
        self.config, self.evidence_loader, self.device = config, evidence_loader, device
        self.bank = TrialBank(config)
        self.refreshes = []
        self.active_steps = 0
        self.last_probability = None
        self.last_batch = None
        self.last_record = dict(active=False, target_ce=0., added_loss=0.)

    def prepare(self, model, iteration, active, context):
        self.last_probability = self.last_batch = None
        self.last_record = dict(active=bool(active), target_ce=0., added_loss=0.)
        if active and (self.bank.iteration is None or iteration - self.bank.iteration >= self.config.refresh_interval):
            evidence = collect_evidence(model, self.evidence_loader, self.device, context)
            self.bank.update(evidence, iteration)
            self.refreshes.append(int(iteration))

    @torch.no_grad()
    def refine(self, output, batch, original, iteration, active, confidence_threshold, jsd_threshold, minimum_votes):
        if 'y' in batch:
            raise RuntimeError('Target truth entered pseudo-label generation')
        if not active:
            return original
        q, indices = self.bank.query(output['scale_embeddings'], identities(batch))
        labels, confidence = q.argmax(-1), q.max(-1).values
        # Keep R2's three checks for DOMAIN alignment only, now against the
        # refined label/confidence. Target CE never uses this hard mask.
        votes = (output['calibrated_scale_logits'].argmax(-1) == labels[:, None]).sum(-1)
        domain_mask = (confidence >= confidence_threshold) & (votes >= minimum_votes) & (original.js_divergence <= jsd_threshold)
        self.last_probability = q
        weights = entropy_weights(q)
        self.active_steps += 1
        self.last_batch = dict(iteration=int(iteration), ids=cpu(identities(batch)),
            raw_probability=cpu(original.probability), refined_probability=cpu(q),
            neighbor_ids=self.bank.ids[indices.cpu()], weights=cpu(weights), domain_mask=cpu(domain_mask),
            bank_iteration=self.bank.iteration, history_iterations=list(self.bank.history_iterations))
        self.last_record.update(bank_iteration=self.bank.iteration, history_count=len(self.bank.history),
            neighbors=indices.shape[1], mean_weight=float(weights.mean()), min_weight=float(weights.min()),
            mean_confidence=float(confidence.mean()), changed_label_fraction=float((labels != original.pseudo_label).float().mean()),
            target_ce_coverage=1., domain_coverage=float(domain_mask.float().mean()),
            pseudo_class_count=torch.bincount(labels, minlength=q.shape[-1]).tolist())
        return replace(original, probability=q, pseudo_label=labels, confidence=confidence,
                       valid_mask=domain_mask, vote_count=votes)

    def loss(self, logits, ramp):
        if self.last_probability is None:
            return logits.sum() * 0.
        loss = weighted_ce(logits, self.last_probability)
        added = self.config.weight * ramp * loss
        self.last_record.update(target_ce=float(loss.detach()), coefficient=self.config.weight * ramp,
                                added_loss=float(added.detach()))
        return added

    def metadata(self):
        return dict(config=asdict(self.config), reference_commit=REFERENCE_COMMIT,
                    active_steps=self.active_steps, refresh_iterations=self.refreshes,
                    generator='self-excluded target kNN; arithmetic probability history',
                    weight_formula='exp(-H(q)/log(C))', target_ce_hard_mask=False,
                    target_truth_used=False, prototype_learning=False)
