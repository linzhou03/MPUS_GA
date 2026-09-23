"""Source-supervised dual heads and matched-budget classifier discrepancy steps."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('r2', 'proto_off', 'dual_source', 'dual_mcd')


@dataclass(frozen=True)
class BoundaryConfig:
    variant: str
    source_weight: float = .3
    discrepancy_weight: float = 1.
    warmup: int = 300
    ramp_end: int = 600
    temperature: float = 1.

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(self.variant)

    def ramp(self, iteration):
        return min(1., max(0., (iteration-self.warmup)/(self.ramp_end-self.warmup)))


class DualHeads(nn.Module):
    def __init__(self, scales, dimension, config):
        super().__init__()
        self.config = config
        self.head1 = nn.Linear(scales * dimension, 3)
        self.head2 = nn.Linear(scales * dimension, 3)
        self.last_record = {}

    def forward(self, z):
        # Normalize each scale without learned parameters; preserve all scales.
        x = F.layer_norm(z, (z.shape[-1],)).flatten(1)
        return self.head1(x), self.head2(x)

    def probability(self, z):
        a, b = self(z)
        return ((a / self.config.temperature).softmax(-1) +
                (b / self.config.temperature).softmax(-1)) / 2

    def source_loss(self, embeddings, labels):
        a, b = self(embeddings)
        # Equal weight to classes actually present; never impose target priors.
        def balanced_ce(logits):
            losses = F.cross_entropy(logits, labels, reduction='none', label_smoothing=.1)
            return torch.stack([losses[labels == c].mean() for c in range(3) if (labels == c).any()]).mean()
        return (balanced_ce(a) + balanced_ce(b)) / 2

    def discrepancy(self, embeddings):
        a, b = self(embeddings)
        return (a.softmax(-1) - b.softmax(-1)).abs().mean()


def attach_heads(model, config, device):
    if config.variant not in ('dual_source', 'dual_mcd'):
        return
    # Adding classifiers must not shift subsequent encoder/dropout RNG streams.
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        model.boundary_heads = DualHeads(len(model.scale_keys), model.scale_embedding.shape[-1], config).to(device)


def encode(model, batch, device):
    pooled = []
    for key in model.scale_keys:
        x, mask = batch['x'][key].to(device), batch['mask'][key].to(device)
        sequence = model.encode_window_sequence(x, mask, key)
        pooled.append(model.temporal[key].pool_sequence(sequence, mask))
    return model.scale_output_norm(torch.stack(pooled, 1) + model.scale_embedding[None])


@contextmanager
def trainable_only(model, keep):
    saved = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, was_trainable in saved:
            p.requires_grad_(was_trainable and id(p) in keep)
        yield
    finally:
        for p, value in saved:
            p.requires_grad_(value)


def extra_steps(model, sources, target, optimizer, device, iteration, gradient_clip,
                encode_fn=encode):
    """B updates ONLY heads; C updates ONLY encoder. Source control has lambda=0.

    Both dual variants execute the same steps and forwards after source warmup.
    No hard target pseudolabel or target true label enters these objectives.
    """
    if 'y' in target:
        raise RuntimeError('Target labels must not enter boundary adaptation')
    heads = model.boundary_heads
    config = heads.config
    ramp = config.ramp(iteration)
    record = dict(active=iteration > config.warmup, discrepancy_coefficient=0.,
                  head_source=0., head_discrepancy=0., encoder_source=0., encoder_discrepancy=0.)
    if not record['active']:
        return record
    coefficient = config.discrepancy_weight * ramp if config.variant == 'dual_mcd' else 0.
    record['discrepancy_coefficient'] = coefficient
    head_ids = {id(p) for p in heads.parameters()}
    encoder_ids = {id(p) for p in model.parameters()} - head_ids
    labels = torch.cat([batch['y'].to(device) for batch in sources])
    devices = [device.index or 0] if device.type == 'cuda' else []
    # Extra forwards use isolated random streams, keeping the next base A step
    # and target/source batch identities matched across both dual variants.
    with torch.random.fork_rng(devices=devices):
        with torch.no_grad():
            source_z = torch.cat([encode_fn(model, b, device) for b in sources])
            target_z = encode_fn(model, target, device)
        optimizer.zero_grad(set_to_none=True)
        with trainable_only(model, head_ids):
            source_loss = heads.source_loss(source_z, labels)
            discrepancy = heads.discrepancy(target_z)
            (source_loss - coefficient * discrepancy).backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), gradient_clip)
            optimizer.step()
        record.update(head_source=float(source_loss.detach()), head_discrepancy=float(discrepancy.detach()))
        optimizer.zero_grad(set_to_none=True)
        with trainable_only(model, encoder_ids):
            source_z = torch.cat([encode_fn(model, b, device) for b in sources])
            target_z = encode_fn(model, target, device)
            source_loss = heads.source_loss(source_z, labels)
            discrepancy = heads.discrepancy(target_z)
            (config.source_weight * source_loss + coefficient * discrepancy).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], gradient_clip)
            optimizer.step()
        record.update(encoder_source=float(source_loss.detach()), encoder_discrepancy=float(discrepancy.detach()))
        optimizer.zero_grad(set_to_none=True)
    return record
