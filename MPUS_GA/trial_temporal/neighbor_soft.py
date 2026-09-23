"""Trial-neighbour pseudo labels and SoftMatch-style weighted target CE.

Mechanisms adapted from AdaContrast (CVPR 2022) and SoftMatch (ICLR 2023).
This is a component adaptation, not a reproduction of either full framework.
All inputs to this module are unlabeled; truth joins live in the offline runner.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .pcdiag import cpu, identities, restore_rng, rng_state

VARIANTS = ('full', 'r2', 'raw_ce', 'raw_soft', 'neighbor_ce')


@dataclass(frozen=True)
class NeighborConfig:
    variant: str = 'full'
    neighbors: int = 5
    refresh_interval: int = 50
    weight: float = 1.0
    momentum: float = .9  # EMA is updated per full refresh, not per minibatch.
    n_sigma: float = 2.0
    variance_floor: float = 1e-6

    def __post_init__(self):
        if self.variant not in VARIANTS or self.neighbors < 1 or self.refresh_interval < 1:
            raise ValueError('Invalid neighbor configuration')
        if not 0 <= self.momentum < 1 or self.weight < 0 or self.n_sigma <= 0:
            raise ValueError('Invalid weight configuration')

    @property
    def refined(self):
        return self.variant in ('full', 'neighbor_ce')

    @property
    def weighted(self):
        return self.variant in ('full', 'raw_soft')


@contextmanager
def passive_inference(model, device):
    """Extra bank forwards cannot consume training RNG or change module modes."""
    state = rng_state(device)
    modes = [(module, module.training) for module in model.modules()]
    grl = getattr(model, 'grl', None)
    alpha = getattr(grl, 'alpha', None)
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        if alpha is not None:
            grl.alpha = alpha
        restore_rng(state, device)


@torch.no_grad()
def collect_bank(model, loader, device, context):
    from .train import _batch_to_device
    keys, features, probabilities = [], [], []
    with passive_inference(model, device):
        for batch in loader:
            if 'y' in batch:
                raise RuntimeError('Neighbor memory must never receive target labels')
            x, mask = _batch_to_device(batch, device)
            out = model(x, mask, **context)
            keys.append(identities(batch))
            features.append(out['scale_embeddings'].detach().cpu())
            probabilities.append(out['probability'].detach().cpu())
    if not keys:
        raise ValueError('Empty target trial bank')
    bank = dict(ids=torch.cat(keys), embeddings=torch.cat(features), raw_probability=torch.cat(probabilities))
    unique = {tuple(row) for row in bank['ids'].tolist()}
    if len(unique) != len(bank['ids']):
        raise ValueError('Each target trial must appear exactly once in the bank')
    if not torch.isfinite(bank['embeddings']).all() or not torch.isfinite(bank['raw_probability']).all():
        raise FloatingPointError('Non-finite target evidence')
    return bank


@torch.no_grad()
def refine_bank(bank, neighbors):
    """Equal-scale cosine similarity; exclude the query trial before soft voting."""
    z = F.normalize(bank['embeddings'].float(), dim=-1)
    n = len(z)
    if n < 2:
        raise ValueError('Neighbor voting needs at least two distinct target trials')
    if len({tuple(row) for row in bank['ids'].tolist()}) != n:
        raise ValueError('Duplicate trial identities')
    similarity = torch.einsum('isd,jsd->ij', z, z) / z.shape[1]
    similarity.fill_diagonal_(-float('inf'))
    # Stable order resolves distance ties by fixed trial order on both servers.
    indices = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :min(neighbors, n - 1)]
    q = bank['raw_probability'][indices].mean(1)
    return q, indices


class SoftWeight:
    def __init__(self, config):
        self.config, self.mean, self.variance = config, None, None

    @torch.no_grad()
    def update(self, probability):
        confidence = probability.max(-1).values.float()
        mean = confidence.mean()
        variance = confidence.var(unbiased=len(confidence) > 1).clamp_min(self.config.variance_floor)
        # Seed statistics from a full target scan, avoiding a long artificial transient.
        if self.mean is None:
            self.mean, self.variance = mean, variance
        else:
            m = self.config.momentum
            self.mean = m * self.mean + (1 - m) * mean
            self.variance = (m * self.variance + (1 - m) * variance).clamp_min(self.config.variance_floor)
        return self.weights(confidence)

    def weights(self, confidence):
        denominator = 2 * self.variance.clamp_min(self.config.variance_floor) / self.config.n_sigma ** 2
        return torch.exp(-((confidence - self.mean).clamp(max=0).square() / denominator)).detach()

    def state(self):
        return {'mean': None if self.mean is None else float(self.mean),
                'variance': None if self.variance is None else float(self.variance)}


class NeighborLearning:
    def __init__(self, config, loader, device):
        if config.variant == 'r2':
            raise ValueError('Original R2 must not instantiate an auxiliary learner')
        self.config, self.loader, self.device = config, loader, device
        self.weighting = SoftWeight(config)
        self.bank = None
        self.last_refresh = None
        self.snapshots, self.events = [], []

    @torch.no_grad()
    def refresh(self, model, iteration, context):
        bank = collect_bank(model, self.loader, self.device, context)
        refined, neighbors = refine_bank(bank, self.config.neighbors)
        selected = refined if self.config.refined else bank['raw_probability']
        weight = self.weighting.update(selected) if self.config.weighted else torch.ones(len(selected))
        bank.update(refined_probability=refined, selected_probability=selected,
                    labels=selected.argmax(-1), weights=weight, neighbors=neighbors,
                    iteration=iteration, statistics=self.weighting.state())
        self.bank, self.last_refresh = bank, iteration
        self.index = {tuple(row): i for i, row in enumerate(bank['ids'].tolist())}
        self.snapshots.append(cpu(bank))

    def loss(self, model, output, batch, iteration, ramp, context):
        if 'y' in batch:
            raise RuntimeError('Target CE must not receive target truth')
        coefficient = self.config.weight * ramp
        if coefficient <= 0:
            return output['logits'].sum() * 0., dict(active=False, ce=0., added_loss=0., coefficient=0.)
        if self.bank is None or iteration - self.last_refresh >= self.config.refresh_interval:
            self.refresh(model, iteration, context)
        ids = identities(batch)
        rows = torch.tensor([self.index[tuple(key)] for key in ids.tolist()], dtype=torch.long)
        labels = self.bank['labels'][rows].to(output['logits'].device)
        weights = self.bank['weights'][rows].to(output['logits'])
        # Divide by B, not by sum(weights): uncertain batches exert less force.
        loss = (F.cross_entropy(output['logits'], labels, reduction='none') * weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite target classification loss')
        self.events.append(dict(iteration=iteration, ids=ids, labels=labels.detach().cpu(),
                                raw_labels=self.bank['raw_probability'][rows].argmax(-1),
                                weights=weights.detach().cpu(), coefficient=coefficient,
                                bank_iteration=self.last_refresh))
        counts = torch.bincount(labels, minlength=3)
        masses = [float(weights[labels == c].sum()) for c in range(3)]
        record = dict(active=True, ce=float(loss.detach()), added_loss=float(loss.detach()) * coefficient,
                      coefficient=coefficient, mean_weight=float(weights.mean()),
                      labels_by_class=counts.tolist(), weight_mass_by_class=masses,
                      changed_fraction=float((labels.cpu() != self.bank['raw_probability'][rows].argmax(-1)).float().mean()),
                      bank_iteration=self.last_refresh, **self.weighting.state())
        return coefficient * loss, record

    def state(self):
        return dict(config=asdict(self.config), snapshots=self.snapshots, events=self.events,
                    statistics=self.weighting.state(), target_truth_used=False,
                    scope='AdaContrast neighbor voting + SoftMatch weighting adaptation; no extra teacher or clustering')
