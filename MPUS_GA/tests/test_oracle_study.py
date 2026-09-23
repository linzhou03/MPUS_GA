from dataclasses import replace
import json

import pytest
import torch

from MPUS_GA.tests.test_msmr import dataset
from MPUS_GA.tests.test_pcdiag import settings, assert_tree
from MPUS_GA.trial_temporal import train, pcdiag_observe as observer
from MPUS_GA.trial_temporal.oracle_study import StudyAudit, truth_coverage_policy
from MPUS_GA.trial_temporal import oracle_study_report as reporting
from MPUS_GA.trial_temporal.train_pcdiag import fold_paths
from MPUS_GA.scripts.run_r2_oracle_suite import make_plan, command


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def test_single_gpu_predeclared_dependencies(tmp_path):
    queues, summary = make_plan('r2_oracle_test', tmp_path)
    assert list(queues) == ['0'] and len(queues['0']) == 85
    assert summary['reported_conditions'] == 70 and summary['training_steps'] == 64000
    assert [i['mode'] for i in queues['0'][:5]] == ['reference', 'R2', 'P', 'C', 'PC']
    assert sum(i['mode'].endswith('_truth') for i in queues['0']) == 10
    assert all(i['direction'] == 'E' for i in queues['0'] if i['mode'].endswith('_truth'))
    assert '--subject' in command('python', queues['0'][0], tmp_path)


def test_truth_coverage_can_reach_misclassified_neutral_without_duplicates():
    ref = {'pseudo_label': torch.tensor([0, 0, 2, 0, 2]),
           'valid_mask': torch.tensor([True, False, False, False, False]),
           'confidence': torch.tensor([.9, .9, .8, .9, .7]),
           'probability': torch.tensor([[.9, .05, .05], [.9, .1, 0], [.1, .1, .8], [.9, .1, 0], [.1, .2, .7]]),
           'effective_weight': torch.tensor([.75, .75, .5, .75, .25]),
           'target_ids': torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 1], [1, 1, 3]])}
    truth = torch.tensor([1, 1, 1, 1, 2])
    c = truth_coverage_policy(ref, truth, 'C_truth', 301)
    pc = truth_coverage_policy(ref, truth, 'PC_truth', 301)
    assert int(c[3].sum()) == 2 and torch.equal(c[3], pc[3])
    assert torch.equal(pc[0][pc[1]], truth[pc[1]])
    assert c[0][0] == 0 and pc[0][0] == 1
    assert (c[2][c[3]] == .25).all()


def test_exposure_zero_precision_denominator_is_na():
    truth = {(1, 1, 1): (1, 'neutral'), (1, 1, 2): (2, 'sad')}
    record = {'step': 301, 'adaptation_active': True, 'target_ids': torch.tensor([[1, 1, 1], [1, 1, 2]]),
              'pseudo_label': torch.tensor([0, 2]), 'valid_mask': torch.tensor([False, True]),
              'added': torch.tensor([False, True]), 'effective_weight': torch.tensor([0., .25])}
    for k in ('domain_used', 'prototype_used', 'memory_used'):
        record[k] = record['valid_mask']
    rows = reporting.summarize_exposures([record], truth)
    assert rows[1]['P'] is None and rows[1]['R_exposure'] == 0
    assert rows[2]['correct_effective_mass'] == .25 and rows[2]['unique_correct_coverage'] == 1


def test_complete_E_reference_exact_replay_all_oracles_and_paired_audit(tmp_path, monkeypatch):
    from copy import deepcopy
    base_args, spec, prepared = settings(tmp_path / 'reference', 'E')
    target = dataset(80)
    monkeypatch.setattr(train, 'prepare_target', lambda *a: target)
    monkeypatch.setattr(train, 'FIXED_UDA_PROTOCOL', replace(train.FIXED_UDA_PROTOCOL, training_iterations=6))
    truth = {tuple(k): (int(target.labels[i]), ('happy', 'neutral', 'sad')[int(target.labels[i])]) for i, k in enumerate(target.keys)}
    monkeypatch.setattr(observer, 'prepare_target', lambda *a: target)
    monkeypatch.setattr(observer, 'load_truth', lambda *a: truth)
    monkeypatch.setattr(reporting, 'load_truth', lambda *a: truth)
    reference = fold_paths(tmp_path / 'reference', 'E', 42, 1)[1]
    for mode in ('reference', 'R2', 'P', 'C', 'PC', 'C_truth', 'PC_truth'):
        args = deepcopy(base_args)
        args.result_root = tmp_path / mode
        args.adaptation_warmup_iterations, args.adaptation_ramp_end = 2, 4
        result, directory = fold_paths(args.result_root, 'E', 42, 1)
        args._diagnostic = StudyAudit(directory, mode, None if mode == 'reference' else reference,
                                      nodes=(0, 2, 4, 6), resume_step=2)
        train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
        if mode == 'reference':
            cp = torch.load(reference / 'step_0002.pt', weights_only=False)
            assert 'optimizer' in cp and 'scheduler' in cp
            continue
        observer.observe_fold(result, tmp_path, torch.device('cpu'))
        reporting.write_fold_exposures(result, tmp_path)
    control = fold_paths(tmp_path / 'R2', 'E', 42, 1)[1]
    assert (control / 'exact_replay_passed.json').exists()
    assert_tree(torch.load(control / 'step_0006.pt', weights_only=False)['model'],
                torch.load(reference / 'step_0006.pt', weights_only=False)['model'])
    assert reporting.report_study(tmp_path) == 6
    data = json.loads((tmp_path / 'paired_results.json').read_text())
    assert data['completed_conditions'] == 6
    assert all(c['subjects'] in (0, 1) for c in data['comparisons'])
    with pytest.raises(RuntimeError, match='Incomplete conditions'):
        reporting.report_study(tmp_path, require_complete=True)


def test_exact_gate_does_not_silently_relax_tolerance(tmp_path):
    audit = StudyAudit(tmp_path, 'R2', tmp_path / 'reference')
    with pytest.raises(AssertionError, match='Exact R2 replay failed'):
        audit.compare_evidence(311, 'probability', torch.tensor([.5]), torch.tensor([.50000006]))
    assert (tmp_path / 'gate_failure.json').exists()
