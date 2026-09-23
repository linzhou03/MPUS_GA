from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from MPUS_GA.trial_temporal.care_pseudo_label import (
    CareConfig, CareController, care_config, candidate_evidence, CARE_VARIANTS,
)
from MPUS_GA.trial_temporal.multiscale_coteaching import CoTeachingConfig


def geometry(labels):
    p = F.one_hot(labels, 3).float() * .55 + .15
    p = p[:, None].repeat(1, 3, 1)
    f = F.one_hot(labels, 3).float()[:, None].repeat(1, 3, 1)
    proto = torch.eye(3)[None, :, None].repeat(3, 1, 2, 1)
    support = torch.ones(3, 3, 2, dtype=torch.long)
    return p, f, proto, support


def test_dual_evidence_disagreement_and_absolute_floor():
    p, f, proto, support = geometry(torch.tensor([0, 1, 2]))
    assert candidate_evidence(p, f, proto, support, CareConfig())['eligible'].all()
    assert not candidate_evidence(p, f.roll(1, -1), proto, support, CareConfig())['eligible'].any()
    assert not candidate_evidence(p * 0 + 1/3, f, proto, support, CareConfig())['eligible'].any()
    support[:, 2] = 0
    assert not candidate_evidence(p, f, proto, support, CareConfig())['eligible'].any()


def test_no_geometry_ablation_preserves_classifier_floor():
    p, f, proto, support = geometry(torch.tensor([1]))
    assert candidate_evidence(p, -f, proto, support * 0, care_config('no_geometry'))['eligible'].all()
    assert not candidate_evidence(p * 0 + 1/3, f, proto, support, care_config('no_geometry'))['eligible'].any()


def row(i, label=1, anchor=False):
    return {'label': label, 'q': [.15, .7, .15], 'score': .8, 'eligible': not anchor,
            'anchor': anchor, 'anchor_label': label, 'seen': i}


def test_unique_trial_stability_and_adaptive_selection():
    full = CareController(care_config(), CoTeachingConfig())
    fixed = CareController(care_config('fixed_selection'), CoTeachingConfig())
    for c in (full, fixed):
        for iteration in (50, 100):
            for i in range(10):
                for _ in range(4): c.cache[(i,)] = row(iteration)
            c.refresh(iteration)
            if iteration == 50: assert not c.selected
    assert len(full.selected) == 7 and len(fixed.selected) == 2
    assert full.latest['candidate_unique'] == [0., 10., 0.]
    assert not any(r['label'] != 1 for r in full.selected.values())
    full.refresh(150)  # No newly observed trials: cannot invent stability.
    assert not full.selected
    full.refresh(400)
    assert not full.cache


def test_anchor_relative_gap_does_not_use_absolute_class_count():
    c = CareController(care_config(), CoTeachingConfig())
    for iteration in (50, 100):
        c.cache = {(i,): row(iteration, anchor=i >= 10) for i in range(100)}
        c.refresh(iteration)
    assert .8 < c.coverage[1] < 1
    assert .2 < c.fractions[1] < .3
    assert len(c.selected) == 3


def fake_alignment(n=6):
    labels = torch.arange(n) % 3
    p, f, proto, support = geometry(labels)
    keys = [(0, 1, 1, i) for i in range(n)]
    a = SimpleNamespace(
        pending=([], None, None, None, keys, f.requires_grad_(), p.requires_grad_(), 350),
        bank=SimpleNamespace(prototypes=torch.stack((proto, proto)).requires_grad_(),
                             support=torch.stack((support, support))),
        training_evidence={'target_valid': torch.zeros(n, 3, dtype=torch.bool),
                           'target_labels': labels, 'target_confidence': torch.full((n,), .7)},
        evidence=SimpleNamespace(published={}))
    batch = {'subject_id': torch.ones(n).long(), 'session_id': torch.ones(n).long(),
             'trial_id': torch.arange(n)}
    return a, batch


def test_expansion_gradients_and_no_mutation_of_r2():
    a, batch = fake_alignment()
    c = CareController(care_config(), CoTeachingConfig())
    proto = a.bank.prototypes.detach().clone()
    logits = torch.randn(6, 3, 3, requires_grad=True)
    for iteration in (350, 400):
        a.pending = (*a.pending[:-1], iteration)
        value, record = c.loss(a, {'scale_logits': logits}, batch, iteration)
    assert value > 0 and sum(record['accepted_by_class']) > 0
    value.backward()
    assert logits.grad.abs().sum() > 0
    assert a.pending[5].grad is None and a.pending[6].grad is None
    assert a.bank.prototypes.grad is None
    torch.testing.assert_close(proto, a.bank.prototypes)
    assert not a.training_evidence['target_valid'].any()
    assert a.evidence.published == {}


def test_live_r2_selection_is_never_rescued_and_warmup_zero():
    a, batch = fake_alignment()
    a.evidence.published = {key: {'selected': True, 'last_seen': 300} for key in a.pending[4]}
    c = CareController(care_config(), CoTeachingConfig())
    for iteration in (250, 300, 350):
        a.pending = (*a.pending[:-1], iteration)
        loss, rec = c.loss(a, {'scale_logits': torch.randn(6, 3, 3, requires_grad=True)}, batch, iteration)
        assert loss == 0 and not any(rec['accepted_by_class'])


def test_target_truth_and_misaligned_scales_rejected():
    a, batch = fake_alignment()
    c = CareController(care_config(), CoTeachingConfig())
    with pytest.raises(RuntimeError, match='target truth'):
        c.loss(a, {'scale_logits': torch.randn(6, 3, 3)}, {**batch, 'y': torch.zeros(6)}, 350)
    with pytest.raises(ValueError, match='trial identities'):
        c.loss(a, {'scale_logits': torch.randn(6, 3, 3)},
               {**batch, 'trial_key_by_scale': {1.: torch.zeros(6, 3).long()}}, 350)


def test_disabled_exact_r2_update_and_no_added_parameters():
    from MPUS_GA.tests.test_muse_integration import fixture, with_ids, batch, step
    from MPUS_GA.trial_temporal.train import care_experiment, coteaching_experiment
    torch.manual_seed(43); left = fixture(enabled=False)
    torch.manual_seed(43); right = fixture(enabled=False)
    source, target = with_ids(batch(True, 6)), with_ids(batch(False, 6))
    for iteration in (1, 300, 350, 600, 650):
        rng = torch.get_rng_state()
        expected = step(left, source, target, iteration, coteaching_experiment('A'))
        torch.set_rng_state(rng)
        actual = step(right, source, target, iteration, care_experiment('A', 'r2'))
        assert expected == actual
        for k, v in left[0].state_dict().items():
            torch.testing.assert_close(v, right[0].state_dict()[k], rtol=0, atol=0)
        for k, v in left[1].teacher.state_dict().items():
            torch.testing.assert_close(v, right[1].teacher.state_dict()[k], rtol=0, atol=0)


def test_suite_and_all_training_commands():
    from MPUS_GA.scripts.run_care_suite import training_command, DEFAULT_VARIANTS
    from MPUS_GA.scripts.run_r3_suite import build_plan, split_plan, plan_summary
    from MPUS_GA.trial_temporal.train import build_parser, validate_args, care_experiment, subgroup_config
    plan = build_plan('care_test', Path('/tmp'), DEFAULT_VARIANTS, [43, 42], 43)
    queues = split_plan(plan, ['0', '1'])
    assert plan_summary(plan) == {'direction_jobs': 30, 'direction_seed_combinations': 36, 'folds': 612}
    assert [plan_summary(q)['folds'] for q in queues.values()] == [312, 300]
    for item in plan:
        cmd = training_command('python', item, '/tmp/data', 'all', CoTeachingConfig())
        args = build_parser().parse_args(cmd[4:])
        validate_args(args, care_experiment(item['direction'], item['variant']))
        assert args.source_batch_size == args.target_batch_size == 8
        assert args.cuda_memory_budget_gib == 8
        assert asdict(subgroup_config(args)) == asdict(CoTeachingConfig())


def test_anchor_ce_control_only_supervises_original_anchors():
    a, batch = fake_alignment()
    a.training_evidence['target_valid'][0] = True
    c = CareController(care_config('anchor_ce'), CoTeachingConfig())
    value, rec = c.loss(a, {'scale_logits': torch.randn(6, 3, 3, requires_grad=True)}, batch, 350)
    assert value > 0 and rec['accepted_by_class'] == [1, 0, 0]
    assert not c.selected and not c.cache
