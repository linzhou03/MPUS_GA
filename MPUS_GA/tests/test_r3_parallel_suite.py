"""Behavioral checks for concurrent R3 relays and safe reuse of existing folds."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from MPUS_GA.scripts.run_r3_suite import (
    SuiteStopped, build_plan, plan_summary, run_parallel_plans, scheduler_only_change, split_plan,
)
from MPUS_GA.trial_temporal.multiscale_coteaching import CoTeachingConfig
from MPUS_GA.trial_temporal.style_augmentation import STYLE_VARIANTS
from MPUS_GA.trial_temporal.train import build_parser, r3_experiment, r3_config, run_fold, subgroup_config


def test_each_gpu_gets_half_without_duplicate_or_missing_experiments(tmp_path):
    plan = build_plan('r3_test', tmp_path, STYLE_VARIANTS, [43, 42], 43)
    queues = split_plan(plan, ['0', '1'])
    for gpu, directions, folds in [('0', 'ACE', 468), ('1', 'BDF', 450)]:
        assert plan_summary(queues[gpu]) == {
            'direction_jobs': 24, 'direction_seed_combinations': 27, 'folds': folds}
        assert [item['direction'] for item in queues[gpu]] == list(directions) * 8
    keys = [(item['variant'], item['direction']) for queue in queues.values() for item in queue]
    assert len(keys) == len(set(keys)) == 48
    assert split_plan(plan, ['1']) == {'1': plan}


def test_parallel_queues_progress_independently_and_never_overlap_on_same_gpu(tmp_path):
    plan = build_plan('r3_test', tmp_path, STYLE_VARIANTS, [43, 42], 43)
    queues = split_plan(plan, ['0', '1'])
    calls, active, logs = [], {}, []
    now, previous_exit = [0.0], {}

    def launch(command, **kwargs):
        gpu = kwargs['env']['CUDA_VISIBLE_DEVICES'][-1]
        assert gpu not in active
        if gpu in previous_exit:
            assert now[0] - previous_exit[gpu] == 1.0
        item = (command[command.index('--r3-variant') + 1], command[command.index('--experiment') + 1])
        child = Mock(pid=len(calls) + 100, ticks=2 if gpu == '0' else 1)
        child.poll.side_effect = lambda: 0 if child.ticks == 0 else None
        active[gpu] = child
        calls.append((gpu, *item))
        logs.append(kwargs['stdout'])
        return child

    def pause(seconds):
        assert len(calls) < 100
        now[0] += seconds
        for gpu, child in list(active.items()):
            child.ticks -= 1
            if child.ticks == 0:
                del active[gpu]
                previous_exit[gpu] = now[0]

    assert run_parallel_plans(queues, 'python', tmp_path, '/tmp/data', 'all', CoTeachingConfig(),
                              {'0': 'GPU-0', '1': 'GPU-1'}, launch, pause, lambda: now[0]) == 0
    assert calls[:3] == [('0', 'full', 'A'), ('1', 'full', 'B'), ('1', 'full', 'D')]
    for gpu in queues:
        assert [(v, d) for g, v, d in calls if g == gpu] == [
            (item['variant'], item['direction']) for item in queues[gpu]]
    assert len(calls) == 48 and all(log.closed for log in logs)


@pytest.mark.parametrize('cancel', [False, True])
def test_failure_or_stop_terminates_both_children_without_submitting_more(tmp_path, cancel):
    plan = build_plan('r3_test', tmp_path, ('full',), [43, 42], 43)
    children, logs = [], []

    def launch(command, **kwargs):
        child = Mock(pid=len(children) + 100)
        child.poll.return_value = 7 if not cancel and not children else None
        children.append(child)
        logs.append(kwargs['stdout'])
        return child

    def pause(_):
        raise SuiteStopped()

    arguments = (split_plan(plan, ['0', '1']), 'python', tmp_path, '/tmp/data', 'all',
                 CoTeachingConfig(), {'0': 'GPU-0', '1': 'GPU-1'}, launch, pause)
    if cancel:
        with pytest.raises(SuiteStopped):
            run_parallel_plans(*arguments)
    else:
        assert run_parallel_plans(*arguments) == 7
    assert len(children) == 2
    assert children[1].terminate.call_count == 1
    assert children[0].terminate.call_count == int(cancel)
    assert all(log.closed for log in logs)


def test_resume_allows_only_scheduler_changes():
    old = {'method': 'r3', 'plan': [{'variant': 'full', 'seeds': [43, 42]}],
           'data_identity': [['seed_v/data.npz', 128, 100]], 'physical_gpu': '0', 'gpu_uuid': 'GPU-0',
           'code_package': '/release/v1/MPUS_GA',
           'code_sha256': {'trial_temporal/train.py': 'same', 'scripts/run_r3_suite.py': 'old'}}
    new = deepcopy(old)
    del new['physical_gpu'], new['gpu_uuid']
    new.update(physical_gpus=['0', '1'], gpu_uuids={'0': 'GPU-0', '1': 'GPU-1'},
               gpu_queues={'0': ['A', 'C', 'E'], '1': ['B', 'D', 'F']}, code_package='/release/v2/MPUS_GA')
    new['code_sha256']['scripts/run_r3_suite.py'] = 'new'
    assert scheduler_only_change(old, new)
    for key, value in [('method', 'r2'), ('plan', []), ('data_identity', []),
                       ('code_sha256', {'trial_temporal/train.py': 'changed'}), ('extra_training_flag', True)]:
        changed = deepcopy(new)
        changed[key] = value
        assert not scheduler_only_change(old, changed)


def test_completed_json_fold_resumes_but_real_config_changes_are_rejected(tmp_path):
    args = build_parser().parse_args(['--experiment', 'A', '--method', 'r3', '--result-root', str(tmp_path)])
    spec = r3_experiment('A', 'full')
    directory = tmp_path / 'A'
    directory.mkdir()
    path = directory / 'seed_43_subject_01.json'
    saved = {'experiment_spec': asdict(spec), 'subgroup_alignment': {'config': asdict(subgroup_config(args))},
             'style_augmentation': {'config': asdict(r3_config(args))}}
    path.write_text(json.dumps(saved))
    original = path.read_bytes()
    # None prepared/device will fail if resume accidentally starts training.
    run_fold(args, spec, None, 43, 1, None)
    assert path.read_bytes() == original
    for key in ['experiment_spec', 'subgroup_alignment', 'style_augmentation']:
        changed = deepcopy(saved)
        changed[key] = {}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='Existing result'):
            run_fold(args, spec, None, 43, 1, None)
