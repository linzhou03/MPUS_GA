"""Trial-level CBST: class quantiles, probability/threshold voting, selected CE.

Equations 6/8 and Algorithm 2 of Zou et al., ECCV 2018. Independent adaptation
to EEG trials, without segmentation spatial priors or prototype learning.
"""
from dataclasses import asdict, dataclass, replace
import math

import torch
import torch.nn.functional as F
from .pcdiag import cpu, identities
from .uncertainty_pseudo import collect_evidence

REFERENCE_COMMIT = '24439a54d1a9e5f29b8347603092b4ea3b9e9071'


@dataclass(frozen=True)
class CBSTConfig:
    initial_portion: float = .20
    portion_increment: float = .05
    maximum_portion: float = .50
    refresh_interval: int = 50
    weight: float = 1.
    activation_checkpointing: bool = False
    spatial_chunk_windows: int = 0
    selection_mode: str = 'strict_quota'
    target_loss_mode: str = 'sample_mean'
    positive_min_votes: int = 0
    positive_negative_margin: float = 0.

    def __post_init__(self):
        if not 0 < self.initial_portion <= self.maximum_portion <= 1:
            raise ValueError('Invalid CBST selection portions')
        if self.refresh_interval < 1 or self.portion_increment < 0 or self.weight < 0:
            raise ValueError('Invalid CBST schedule')
        if (self.selection_mode, self.target_loss_mode) not in (
            ('strict_quota', 'sample_mean'), ('independent', 'class_mean'),
            ('positive_gate', 'sample_mean')):
            raise ValueError('Unsupported CBST selection/loss combination')
        if self.selection_mode == 'positive_gate':
            if self.positive_min_votes < 2 or self.positive_negative_margin < 0:
                raise ValueError('Invalid positive gate')
        elif self.positive_min_votes or self.positive_negative_margin:
            raise ValueError('Positive gate requires positive_gate selection mode')

    def portion(self, round_index):
        return round(min(self.maximum_portion, self.initial_portion + self.portion_increment * round_index), 10)


@torch.no_grad()
def select(probability, portion, mode='strict_quota', scale_logits=None,
           positive_min_votes=0, positive_negative_margin=0.):
    p = probability.detach().float()
    if p.ndim != 2 or len(p) == 0 or not 0 < portion <= 1:
        raise ValueError('Expected a nonempty probability table and valid portion')
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones(len(p),device=p.device), atol=1e-5):
        raise ValueError('Invalid probability distribution')
    raw_confidence, raw_label = p.max(-1)
    num_classes = p.shape[1]
    thresholds = p.new_ones(num_classes)
    support = torch.bincount(raw_label, minlength=num_classes)
    eligible = torch.ones(len(p), dtype=torch.bool, device=p.device)
    if mode == 'positive_gate':
        if num_classes != 3 or scale_logits is None or scale_logits.ndim != 3 or (
                scale_logits.shape[0] != len(p) or scale_logits.shape[2] != num_classes or
                not 2 <= positive_min_votes <= scale_logits.shape[1] or
                positive_negative_margin < 0):
            raise ValueError('Positive gate requires three classes and valid scale logits')
        votes = (scale_logits.detach().to(p.device).argmax(-1) == 0).sum(-1)
        positive = raw_label == 0
        eligible[positive] = ((votes >= positive_min_votes) &
                              ((p[:, 0] - p[:, 2]) >= positive_negative_margin))[positive]
    eligible_support = torch.bincount(raw_label[eligible], minlength=num_classes)
    ranks = torch.zeros_like(support)
    accepted = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    if mode not in ('strict_quota', 'independent', 'positive_gate'):
        raise ValueError('Unknown CBST selection mode')
    min_support = int(eligible_support.min().item())
    if mode == 'independent' or min_support > 0:
        shared_quota = max(1, int(math.floor(min_support * portion + 1e-9))) if mode != 'independent' else None
        for c in range(num_classes):
            if eligible_support[c] == 0:
                continue
            quota = shared_quota if shared_quota is not None else max(1, int(math.floor(int(eligible_support[c]) * portion + 1e-9)))
            candidates = torch.where(eligible & (raw_label == c))[0]
            cand_conf = raw_confidence[candidates]
            sorted_order = cand_conf.sort(descending=True, stable=True).indices
            top_quota = candidates[sorted_order[:quota]]
            accepted[top_quota] = True
            ranks[c] = quota
            thresholds[c] = cand_conf[sorted_order[quota - 1]]
    ratio = p / thresholds[None].clamp_min(1e-8)
    score = p.gather(1, raw_label[:, None]).squeeze(1)
    return dict(raw_probability=p, raw_label=raw_label, thresholds=thresholds,
        raw_class_support=support, eligible_class_support=eligible_support,
        positive_gate_eligible=eligible, class_rank=ranks, ratio=ratio,
        probability=ratio/ratio.sum(-1,keepdim=True), pseudo_label=raw_label,
        confidence=raw_confidence, score=score, accepted=accepted,
        accepted_class_count=torch.bincount(raw_label[accepted],minlength=num_classes))


def selected_ce(logits, label, accepted, mode='sample_mean'):
    accepted = accepted.detach().bool()
    if not accepted.any():
        return logits.sum() * 0.
    if mode == 'sample_mean':
        return F.cross_entropy(logits[accepted], label.detach()[accepted])
    if mode != 'class_mean':
        raise ValueError('Unknown target loss mode')
    labels = label.detach()
    return torch.stack([
        F.cross_entropy(logits[mask], labels[mask])
        for c in range(logits.shape[-1])
        if (mask := accepted & (labels == c)).any()
    ]).mean()


class CBSTController:
    def __init__(self, config, evidence_loader, device):
        self.config, self.evidence_loader, self.device = config, evidence_loader, device
        self.ids = self.table = self.iteration = None
        self.rounds = []
        self.active_steps = 0
        self.last_batch = None
        self.last_record = dict(active=False, target_ce=0., added_loss=0., coverage=0.)

    def update(self, evidence, iteration):
        ids=cpu(evidence['ids'])
        if len({tuple(v) for v in ids.tolist()}) != len(ids):raise ValueError('Duplicate trial IDs')
        if self.ids is not None and not torch.equal(ids,self.ids):raise ValueError('Target trial order changed')
        if self.iteration is not None and iteration <= self.iteration:raise ValueError('Non-increasing refresh')
        portion=self.config.portion(len(self.rounds))
        self.table=cpu(select(evidence['raw_probability'],portion,self.config.selection_mode,
            evidence.get('scale_logits'),self.config.positive_min_votes,
            self.config.positive_negative_margin))
        self.ids,self.iteration=ids,int(iteration)
        self.rounds.append(dict(iteration=self.iteration,portion=portion,ids=ids,**cpu(self.table)))

    def prepare(self, model, iteration, active, context):
        self.last_batch = None
        self.last_record = dict(active=bool(active), target_ce=0., added_loss=0., coverage=0.)
        if active and (self.iteration is None or iteration-self.iteration >= self.config.refresh_interval):
            self.update(collect_evidence(model,self.evidence_loader,self.device,context),iteration)

    def lookup(self, ids, device):
        if self.table is None:raise RuntimeError('CBST labels not initialized')
        matches=(ids.cpu()[:,None] == self.ids[None]).all(-1)
        if not (matches.sum(-1)==1).all():raise ValueError('Unknown target trial identity')
        idx=matches.long().argmax(-1)
        return {k:v[idx].to(device) for k,v in self.table.items()
                if k in ('probability','pseudo_label','confidence','accepted','raw_probability','ratio','score')}

    @torch.no_grad()
    def refine(self, output, batch, original, iteration, active, confidence_threshold, jsd_threshold, minimum_votes):
        if 'y' in batch:raise RuntimeError('Target labels entered CBST')
        if not active:return original
        ids=identities(batch)
        row=self.lookup(ids,output['logits'].device)
        self.active_steps+=1
        self.last_batch=dict(iteration=int(iteration),ids=ids,refresh_iteration=self.iteration,**cpu(row))
        self.last_record.update(coverage=float(row['accepted'].float().mean()),refresh_iteration=self.iteration,
            portion=self.rounds[-1]['portion'],thresholds=self.table['thresholds'].tolist(),
            raw_class_support=self.table['raw_class_support'].tolist(),
            eligible_class_support=self.table['eligible_class_support'].tolist(),
            full_accepted_class_count=self.table['accepted_class_count'].tolist(),
            batch_accepted_class_count=torch.bincount(row['pseudo_label'][row['accepted']],minlength=3).tolist(),
            changed_label_fraction=float((row['pseudo_label'] != original.pseudo_label).float().mean()))
        votes=(output['calibrated_scale_logits'].argmax(-1)==row['pseudo_label'][:,None]).sum(-1)
        # CBST is the sole hard selector for CE and target domain balancing.
        # Old 0.6/0.9, scale agreement and JSD do not veto its selected trials.
        return replace(original,probability=row['probability'],pseudo_label=row['pseudo_label'],
            confidence=row['confidence'],valid_mask=row['accepted'],vote_count=votes)

    def loss(self, logits, ramp):
        if self.last_batch is None:return logits.sum()*0.
        row=self.last_batch
        loss=selected_ce(logits,row['pseudo_label'].to(logits.device),row['accepted'].to(logits.device),self.config.target_loss_mode)
        added=self.config.weight*ramp*loss
        self.last_record.update(target_ce=float(loss.detach()),added_loss=float(added.detach()),coefficient=self.config.weight*ramp)
        return added

    def state(self):
        return dict(ids=self.ids,table=cpu(self.table),iteration=self.iteration,rounds=cpu(self.rounds))

    def metadata(self):
        return dict(config=asdict(self.config),reference_commit=REFERENCE_COMMIT,active_steps=self.active_steps,
            refresh_iterations=[r['iteration'] for r in self.rounds],generator='CBST probability / class threshold',
            selection=self.config.selection_mode,target_truth_used_for_training=False,
            prototype_learning=False,knn=False,entropy_weights=False,teacher=False)
