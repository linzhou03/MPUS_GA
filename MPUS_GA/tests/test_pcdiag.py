from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from MPUS_GA.tests.test_msmr import dataset
from MPUS_GA.tests.test_r2_masked import batches, parts
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.data import PreparedMultiSource
from MPUS_GA.trial_temporal.pcdiag import (Audit, cpu, frozen_profile, oracle_policy, rng_state, restore_rng)
from MPUS_GA.trial_temporal.pcdiag_observe import (observe_snapshot, metric_rows, rejected_rows, subject_summary)
from MPUS_GA.trial_temporal.train_pcdiag import arguments
from MPUS_GA.scripts.run_r2_pcdiag_suite import make_plan, command


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def settings(tmp_path, direction='C'):
    args, spec = arguments(direction, tmp_path, tmp_path, subjects='1', seeds=(42,), device='cpu')
    args.d_model, args.dim_feedforward, args.dropout = 32, 64, .3
    args.spatial_layers = args.temporal_layers = args.fusion_layers = 1
    args.source_batch_size = args.target_batch_size = 6
    source = dataset(6)
    prepared = PreparedMultiSource(spec.source_domains, (source,), source.stats)
    return args, spec, prepared


def assert_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_tree(left[k], right[k])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree(a, b)
    else:
        assert left == right


def test_profile_is_historical_and_queue_counts(tmp_path):
    reference = json.loads((Path(__file__).parents[1] / 'trial_temporal/pcdiag_frozen.json').read_text())
    for d in 'ABCDEF':
        spec, args = frozen_profile(d)
        for k, v in reference['historical_optimization'].items():
            if k in args:
                assert args[k] == v
        assert spec.prototype_weight == .1 and spec.use_source_prototype_memory
        assert not spec.use_subgroup_alignment and not spec.use_multiscale_coteaching
    baseline, replay, oracle, summary = make_plan('r2_original_pcdiag_test', tmp_path)
    assert summary['baseline_folds'] == 306 and summary['new_oracle_tails'] == 30
    assert [x['direction'] for x in baseline['0'] if x['stage'] == 'train'] == list('CEA')
    assert [x['direction'] for x in baseline['1'] if x['stage'] == 'train'] == list('DFB')
    assert sum(x['folds'] for q in oracle.values() for x in q if x['stage'] == 'train') == 30
    assert sum(x['folds'] for q in replay.values() for x in q) == 10
    assert '--mode' in command(sys.executable, baseline['0'][0], tmp_path)


def test_logging_is_exactly_passive_with_dropout_all_three_stages(tmp_path):
    args, spec, prepared = settings(tmp_path)
    torch.manual_seed(91)
    left = train.build_fold_model(args, spec, prepared, torch.device('cpu'))
    right = deepcopy(left)
    ls, rs = parts(left), parts(right)
    source, target = batches()
    audit = Audit(tmp_path / 'audit', nodes=())
    audit.runtime = {'args': args}
    for iteration in (1, 350, 650):
        state = rng_state(torch.device('cpu'))
        a = train.train_step(left, [source], target, ls[0], ls[1], torch.device('cpu'), iteration, spec,
                             torch.ones(1, 3) / 3, ls[2], .1, 5., 300, 600, .6, source_prototype_memory=ls[3])
        expected_rng = rng_state(torch.device('cpu'))
        restore_rng(state, torch.device('cpu'))
        b = train.train_step(right, [source], target, rs[0], rs[1], torch.device('cpu'), iteration, spec,
                             torch.ones(1, 3) / 3, rs[2], .1, 5., 300, 600, .6, source_prototype_memory=rs[3], diagnostic=audit)
        assert_tree(rng_state(torch.device('cpu')), expected_rng)
        assert_tree(a, b)
        assert_tree(left.state_dict(), right.state_dict())
        assert_tree(vars(ls[2]), vars(rs[2])); assert_tree(vars(ls[3]), vars(rs[3]))
        assert 'y' not in audit.pending[-1] and 'true_label' not in audit.pending[-1]


def test_oracle_same_counts_and_low_confidence_reaches_both_consumers():
    ref = {'pseudo_label': torch.tensor([0, 1, 2, 0, 1, 2]),
           'valid_mask': torch.tensor([True, True, False, False, False, False]),
           'confidence': torch.tensor([.8, .9, .45, .55, .58, .52]),
           'effective_weight': torch.tensor([.5, .75, 0, 0, 0, 0]),
           'target_ids': torch.tensor([[1, 1, i] for i in range(6)])}
    truth = torch.tensor([2, 1, 2, 0, 1, 2])
    p = oracle_policy(ref, truth, 'P', 301)
    c = oracle_policy(ref, truth, 'C', 301)
    pc = oracle_policy(ref, truth, 'PC', 301)
    assert torch.equal(p[1], ref['valid_mask']) and torch.equal(p[2], ref['effective_weight'])
    assert torch.equal(c[3], pc[3]) and int(c[3].sum()) == 4
    assert torch.equal(pc[0][pc[1]], truth[pc[1]])
    embedding = torch.randn(6, 3, 32, requires_grad=True)
    probability = torch.tensor([[.45, .30, .25]]).expand(6, -1)
    from MPUS_GA.trial_temporal.losses import class_conditional_prototype_alignment_loss
    loss, coverage = class_conditional_prototype_alignment_loss([torch.randn(6, 3, 32)], [truth], embedding,
                    probability, torch.ones(1, 3, 3) / 3, .6, c[3],
                    pseudo_label_override=c[0], effective_weight_override=c[2])
    loss.backward()
    assert loss > 0 and embedding.grad[c[3]].abs().sum() > 0 and coverage == pytest.approx(4 / 6)
    bank = parts(train.build_fold_model(*settings(Path('/tmp'))[:3], torch.device('cpu')))[2]
    bank.update_target(embedding.detach(), probability, .6, c[3], pseudo_label_override=c[0], effective_weight_override=c[2])
    assert bank.target_initialized.all()


def test_denominators_rejected_buckets_and_subject_seed_aggregation():
    observation = {'pseudo_label': torch.tensor([0, 0, 2]), 'valid_mask': torch.tensor([True, False, False]),
                   'confidence': torch.tensor([.9, .45, .8]), 'step': 250,
                   'fused_probability': torch.eye(3), 'raw_scale_probability': torch.eye(3)[:, None].expand(-1, 3, -1),
                   'calibrated_scale_probability': torch.eye(3)[:, None].expand(-1, 3, -1),
                   'passes_confidence': torch.tensor([True, False, True]),
                   'passes_votes': torch.tensor([True, True, False]), 'passes_jsd': torch.tensor([True, True, True])}
    truth = [0, 1, 2]
    metrics = metric_rows(observation, truth)
    assert metrics[0]['P'] == 1 and metrics[1]['P'] is None and metrics[1]['A'] is None
    assert all(m['eligible_correct_coverage'] == 0 for m in metrics)
    rejected = rejected_rows(observation, truth, ['happy', 'neutral', 'sad'])
    assert sum(r['count'] for r in rejected if r['grouping'] == 'true_class') == 2
    assert subject_summary([(1, 0.), (1, 0.), (1, 0.), (2, 1.)])['mean'] == .5


def test_full_fold_checkpoint_offline_and_replay(tmp_path, monkeypatch):
    args, spec, prepared = settings(tmp_path / 'baseline')
    args.adaptation_warmup_iterations, args.adaptation_ramp_end = 2, 4
    target = dataset(45)
    monkeypatch.setattr(train, 'prepare_target', lambda *a: target)
    monkeypatch.setattr(train, 'FIXED_UDA_PROTOCOL', replace(train.FIXED_UDA_PROTOCOL, training_iterations=6))
    directory = args.result_root / 'C/seed_42_subject_01.audit'
    args._diagnostic = Audit(directory, nodes=(0, 2, 4, 6), resume_step=2)
    train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
    last = torch.load(directory / 'step_0006.pt', weights_only=False)
    saved_rng = rng_state(torch.device('cpu'))
    original_state = deepcopy(last['objects'])
    observed = observe_snapshot(last, target, torch.device('cpu'), torch.load(directory / 'formal_final_context.pt', weights_only=False))
    assert_tree(last['objects'], original_state)
    assert len(observed['target_ids']) == 45 and not observed['teacher_present']
    result = json.loads(directory.with_suffix('.json').read_text())
    assert float((observed['fused_probability'].argmax(-1) == target.labels).float().mean()) == pytest.approx(result['evaluation']['fused']['accuracy'])
    args.result_root = tmp_path / 'replay'
    args._diagnostic = Audit(args.result_root / 'C/seed_42_subject_01.audit', 'replay', directory, nodes=(0, 2, 4, 6), resume_step=2)
    train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
    replay = torch.load(args._diagnostic.directory / 'step_0006.pt', weights_only=False)
    assert_tree(last['model'], replay['model'])
    assert_tree(last['objects'], replay['objects'])
    assert_tree(last['rng'], replay['rng'])
    assert args._diagnostic.start_iteration == 2
    from MPUS_GA.trial_temporal import pcdiag_observe as observer
    mapping = {tuple(k): (int(target.labels[i]), ('happy', 'neutral', 'sad')[int(target.labels[i])]) for i, k in enumerate(target.keys)}
    monkeypatch.setattr(observer, 'load_truth', lambda *a: mapping)
    monkeypatch.setattr(observer, 'prepare_target', lambda *a: target)
    report = observer.observe_fold(directory.with_suffix('.json'), tmp_path, torch.device('cpu'))
    assert len(report['nodes']) == 4 and len(report['train_exposures']) == 1
    assert observer.aggregate(directory.parent.parent) == 1
    for mode in ('P', 'C', 'PC'):
        args.result_root = tmp_path / mode
        args._diagnostic = Audit(args.result_root / 'C/seed_42_subject_01.audit', mode, directory, nodes=(0, 2, 4, 6), resume_step=2)
        train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
        result = json.loads((args.result_root / 'C/seed_42_subject_01.json').read_text())
        assert result['oracle'] and result['diagnostic']['target_labels_used_for_training']


def test_original_recovery_code_matches_logged_training(tmp_path, monkeypatch):
    """Check the recovered R2 train/model implementation, not only the new log switch."""
    import tarfile
    import types
    root = Path(__file__).parents[1]
    archive = root / 'code_backups/care_20260911/r2_851ccb2.tar.gz'
    with tarfile.open(archive) as tar:
        modules = {}
        for name in ('model', 'train'):
            qualified = 'MPUS_GA.trial_temporal._pcdiag_legacy_' + name
            module = types.ModuleType(qualified)
            module.__file__ = str(root / 'trial_temporal' / (name + '.py'))
            module.__package__ = 'MPUS_GA.trial_temporal'
            monkeypatch.setitem(sys.modules, qualified, module)
            exec(compile(tar.extractfile('trial_temporal/' + name + '.py').read(), module.__file__, 'exec'), module.__dict__)
            modules[name] = module
    legacy = modules['train']
    legacy.MultiScaleMultiSourceDANN = modules['model'].MultiScaleMultiSourceDANN
    args, spec, prepared = settings(tmp_path / 'original')
    args.adaptation_warmup_iterations, args.adaptation_ramp_end = 2, 4
    target = dataset(45)
    for mod in (train, legacy):
        monkeypatch.setattr(mod, 'prepare_target', lambda *a: target)
        monkeypatch.setattr(mod, 'FIXED_UDA_PROTOCOL', replace(train.FIXED_UDA_PROTOCOL, training_iterations=6))
    seen = {}
    actual = legacy.train_step
    def capture(*pos, **kw):
        record = actual(*pos, **kw)
        seen['model'] = pos[0]
        seen['rng'] = rng_state(torch.device('cpu'))
        return record
    legacy.train_step = capture
    legacy.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
    args.result_root = tmp_path / 'logged'
    directory = args.result_root / 'C/seed_42_subject_01.audit'
    args._diagnostic = Audit(directory, nodes=(0, 2, 4, 6), resume_step=2)
    train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
    new = torch.load(directory / 'step_0006.pt', weights_only=False)
    assert_tree(seen['model'].state_dict(), new['model'])
    assert_tree(seen['rng'], new['rng'])
    old_result = json.loads((tmp_path / 'original/C/seed_42_subject_01.json').read_text())
    new_result = json.loads((tmp_path / 'logged/C/seed_42_subject_01.json').read_text())
    assert_tree(old_result['evaluation'], new_result['evaluation'])
