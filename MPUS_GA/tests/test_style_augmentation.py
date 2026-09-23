"""R3 gradient, information-boundary, unique-support and relay regressions."""
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from MPUS_GA.tests.test_multiscale_coteaching import model, batch
from MPUS_GA.tests.test_training import _small_model
from MPUS_GA.trial_temporal.multiscale_coteaching import ClassConditionalCoTeaching, CoTeachingConfig
from MPUS_GA.trial_temporal.style_augmentation import (
    StyleConfig, STYLE_VARIANTS, style_config, TargetStyleBank, R3StyleController,
    ReliabilityGatedStyleAugmentor, coverage_weights, domain_assignments, matched_variance_loss,
)
from MPUS_GA.trial_temporal.train import (
    train_step, r3_experiment, coteaching_experiment, PrototypeBank, SourceMultiPrototypeMemory,
    build_parser, validate_args, subgroup_config, r3_config,
)
from MPUS_GA.scripts.run_r3_suite import build_plan, run_plan, training_command, SuiteStopped
from MPUS_GA.scripts.stop_owned_training import select_suite, select_gpu, descendants


def fixture(config=None):
    torch.manual_seed(43)
    student = model()
    alignment = ClassConditionalCoTeaching(student, CoTeachingConfig(k=1), torch.device('cpu'))
    controller = R3StyleController(student, alignment, config or StyleConfig())
    labels = torch.arange(3).repeat_interleave(6)
    centers = F.one_hot(labels, 32).float()[:, None].repeat(1, 3, 1) * 4
    noise = torch.randn(18, 3, 32) * .02
    source = centers + noise
    target = centers + noise * 4 + .1
    keys = [(0, 1, 1, i) for i in range(18)]
    valid = torch.ones(18, 3, dtype=torch.bool)
    probability = (F.one_hot(labels, 3).float() * .85 + .05)[:, None].repeat(1, 3, 1)
    for domain, features in enumerate((source, target)):
        alignment.bank.observe(domain, keys, features, labels, torch.ones(18)*.9, valid, 600)
    alignment.bank.refresh(600)
    alignment.readiness.fill_(1)
    alignment.evidence.raw = {key: (probability[i], target[i], 600) for i, key in enumerate(keys)}
    alignment.evidence.published = {key: {'selected': True, 'label': int(labels[i]), 'confidence': .9,
                                            'scale_valid': valid[i], 'last_seen': 600} for i, key in enumerate(keys)}
    alignment.evidence.last_refresh = 600
    alignment.training_evidence = {'source_keys': keys, 'source_features': source, 'source_labels': labels,
                                   'target_keys': keys, 'target_features': target, 'target_labels': labels,
                                   'target_confidence': torch.ones(18)*.9, 'target_valid': valid}
    controller.after_step(alignment, 600)
    return student, alignment, controller, source, target, labels, keys


def outputs(student, source, target):
    source = source.clone().requires_grad_(); target = target.clone().requires_grad_()
    return [{'scale_embeddings': source, 'scale_logits': student.classifier(source)}], {'scale_embeddings': target}


def test_transfer_identity_gradient_clamps_and_invalid_statistics():
    cfg = StyleConfig()
    augment = ReliabilityGatedStyleAugmentor(cfg)
    z = torch.zeros(2, 8, requires_grad=True)
    value = augment.transfer(z, torch.zeros(8), torch.ones(8), torch.full((8,), 100.), torch.ones(8)*100, 5.)
    torch.testing.assert_close(value, torch.full_like(value, .125))
    value.sum().backward(); torch.testing.assert_close(z.grad, torch.ones_like(z))
    invalid = augment.transfer(z, torch.zeros(8), torch.ones(8), torch.full((8,), float('nan')), torch.ones(8), .2)
    torch.testing.assert_close(invalid, z)
    small = augment.transfer(torch.ones(8)*.1, torch.zeros(8), torch.ones(8), torch.zeros(8), torch.ones(8)*100, .2)
    torch.testing.assert_close(small, torch.ones(8)*.12)


def test_global_style_works_without_any_pseudo_label():
    student, alignment, style, source, target, labels, keys = fixture()
    alignment.evidence.published.clear()
    alignment.evidence.last_refresh = 650
    style.after_step(alignment, 650)
    out_s, out_t = outputs(student, source, target)
    loss, record = style.loss(student, out_s, out_t, alignment, 650)
    assert record['style_mode_ratio'] == [1., 0., 0.]
    assert record['accepted_count_per_class'] == [0, 0, 0]
    assert record['coverage_weights'] == [1., 1., 1.]
    assert loss > 0 and record['displacement'] > 0
    loss.backward(); assert out_s[0]['scale_embeddings'].grad.abs().sum() > 0


def test_hierarchy_falls_back_without_changing_logits():
    student, alignment, style, source, target, labels, keys = fixture()
    out_s, out_t = outputs(student, source, target)
    before = out_s[0]['scale_logits'].clone()
    _, full = style.loss(student, out_s, out_t, alignment, 600)
    assert full['style_mode_ratio'] == [0., 0., 1.]
    style.bank.pair_reliability.zero_()
    _, class_record = style.loss(student, out_s, out_t, alignment, 600)
    assert class_record['style_mode_ratio'] == [0., 1., 0.]
    style.bank.target_class_count.fill_(2)
    _, global_record = style.loss(student, out_s, out_t, alignment, 600)
    assert global_record['style_mode_ratio'] == [1., 0., 0.]
    torch.testing.assert_close(out_s[0]['scale_logits'], before, rtol=0, atol=0)


def test_warmup_loss_zero_at_and_before_300():
    student, alignment, style, source, target, *_ = fixture()
    out_s, out_t = outputs(student, source, target)
    for step in (1, 20, 299, 300):
        loss, record = style.loss(student, out_s, out_t, alignment, step)
        assert loss == 0 and record['active_style_samples'] == 0
    assert style.loss(student, out_s, out_t, alignment, 301)[0] > 0


def test_subgroup_variance_has_source_and_target_gradients():
    student, alignment, style, source, target, *_ = fixture()
    cfg = replace(style.config, loss_cls=0, loss_sem=0)
    style.config = cfg
    out_s, out_t = outputs(student, source, target)
    loss, record = style.loss(student, out_s, out_t, alignment, 600)
    assert record['variance_pairs'] == 9 and record['variance_student_rows'] == 108
    assert loss > 0
    loss.backward()
    assert out_s[0]['scale_embeddings'].grad.abs().sum() > 0
    assert out_t['scale_embeddings'].grad.abs().sum() > 0
    assert all(value.grad is None for value in style.bank.buffers())
    style.bank.pair_reliability.zero_()
    out_s, out_t = outputs(student, source, target)
    assert style.loss(student, out_s, out_t, alignment, 600)[1]['variance_pairs'] == 0


def test_variance_cache_deduplicates_and_needs_current_participation():
    student, alignment, style, source, target, labels, keys = fixture()
    valid = torch.ones(18, 3, dtype=torch.bool)
    slots = [domain_assignments(alignment.bank, d, x, labels, valid) for d, x in enumerate((source, target))]
    current = [{'keys': keys, 'features': x.clone().requires_grad_(), 'labels': labels, 'valid': valid, 'slots': slots[d]}
               for d, x in enumerate((source, target))]
    expected = matched_variance_loss(style.bank, style.snapshots, current, style.config)[0]
    doubled = [{name: value + value if name == 'keys' else torch.cat((value, value)) for name, value in row.items()} for row in current]
    actual = matched_variance_loss(style.bank, style.snapshots, doubled, style.config)[0]
    torch.testing.assert_close(expected, actual)
    current[1]['keys'] = [(9, *key[1:]) for key in keys]
    loss, pairs, _ = matched_variance_loss(style.bank, style.snapshots, current, style.config)
    assert pairs == 0 and loss == 0


def test_unique_support_and_subgroup_statistics_reset():
    student, alignment, style, source, target, labels, keys = fixture()
    assert style.bank.target_class_count.tolist() == [[6, 6, 6]] * 3
    before = style.bank.target_subgroup_mu.clone()
    for _ in range(5):
        style.after_step(alignment, 600)
    assert style.bank.target_class_count.tolist() == [[6, 6, 6]] * 3
    changed = target + .2
    alignment.evidence.raw = {key: (row[0], changed[i], 650) for i, (key, row) in enumerate(alignment.evidence.raw.items())}
    alignment.evidence.last_refresh = 650
    style.after_step(alignment, 650)
    torch.testing.assert_close(style.bank.target_subgroup_mu, before + .2, atol=1e-6, rtol=1e-6)
    assert all(not b.requires_grad for b in style.bank.buffers())


def test_style_checkpoint_roundtrip_restores_next_loss():
    student, alignment, style, source, target, *_ = fixture()
    stream = io.BytesIO(); torch.save(style.state_dict(), stream); stream.seek(0)
    restored = R3StyleController(student, alignment, style.config)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    left, right = outputs(student, source, target), outputs(student, source, target)
    expected = style.loss(student, *left, alignment, 600)
    actual = restored.loss(student, *right, alignment, 600)
    torch.testing.assert_close(expected[0], actual[0], atol=0, rtol=0)
    assert expected[1] == actual[1]


def test_coverage_and_zero_perturbation_control():
    cfg = StyleConfig()
    weights = coverage_weights(torch.tensor([20, 10, 1]), cfg)
    assert weights[2] > weights[1] > weights[0]
    torch.testing.assert_close(weights.mean(), torch.tensor(1.))
    student, alignment, style, source, target, *_ = fixture(style_config('ce_control'))
    out_s, out_t = outputs(student, source, target)
    _, record = style.loss(student, out_s, out_t, alignment, 600)
    assert record['displacement'] == 0 and record['semantic'] == 0
    assert record['style_cls'] > 0 and record['style_gate_per_scale'] == [0., 0., 0.]


def step_fixture(variant):
    student = model()
    device = torch.device('cpu')
    alignment = ClassConditionalCoTeaching(student, CoTeachingConfig(), device)
    style = R3StyleController(student, alignment, style_config(variant))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    bank = PrototypeBank(1, 3, 3, 32, .9, .15, .1, device)
    memory = SourceMultiPrototypeMemory(3, 3, 4, 32, .9, device)
    return student, alignment, style, optimizer, scheduler, bank, memory


def test_disabled_full_optimizer_step_equals_coteaching():
    torch.manual_seed(43)
    original = step_fixture('baseline')
    torch.manual_seed(43)
    new = step_fixture('baseline')
    source, target = batch(True, 6), batch(False, 6)
    def run(items, style_on):
        student, alignment, style, opt, sched, bank, memory = items
        return train_step(student, [source], target, opt, sched, torch.device('cpu'), 600,
                          r3_experiment('A', 'baseline') if style_on else coteaching_experiment('A'),
                          torch.ones(1, 3)/3, bank, .1, 5., 300, 600, .6,
                          source_prototype_memory=memory, subgroup_alignment=alignment,
                          style_augmentation=style if style_on else None)
    state = torch.get_rng_state(); left = run(original, False)
    torch.set_rng_state(state); right = run(new, True)
    assert left['total'] == right['total']
    for a, b in zip(original[0].parameters(), new[0].parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_target_truth_rejected_and_source_memory_freezes():
    torch.manual_seed(43)
    student, alignment, style, opt, sched, bank, memory = step_fixture('full')
    source, target = batch(True, 6), batch(False, 6)
    def step(t, iteration):
        return train_step(student, [source], t, opt, sched, torch.device('cpu'), iteration,
                          r3_experiment('A'), torch.ones(1, 3)/3, bank, .1, 5., 300, 600, .6,
                          source_prototype_memory=memory, subgroup_alignment=alignment, style_augmentation=style)
    try:
        step({**target, 'y': source['y']}, 300)
    except RuntimeError as error:
        assert 'labels' in str(error)
    else:
        raise AssertionError('Target labels accepted')
    calls = []
    update = memory.update_source
    def observed(features, labels):
        calls.append(labels.clone()); update(features, labels)
    with patch.object(memory, 'update_source', side_effect=observed):
        step(target, 300)
        frozen = memory.memory.clone()
        step(target, 301)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], source['y'])
    torch.testing.assert_close(memory.memory, frozen, atol=0, rtol=0)


def test_all_variants_and_six_direction_cli():
    plan = build_plan('r3_test', Path('/tmp/output'), STYLE_VARIANTS, [43, 42], 43)
    assert len(plan) == 48 and sum(p['folds'] for p in plan) == 918
    assert sum(len(p['seeds']) for p in plan) == 54
    for i, variant in enumerate(STYLE_VARIANTS):
        jobs = plan[i*6:(i+1)*6]
        assert [p['direction'] for p in jobs] == list('ABCDEF')
        assert all(p['seeds'] == ([43, 42] if variant == 'full' else [43]) for p in jobs)
        for job in jobs:
            command = training_command('python', job, '/tmp/data', 'all', CoTeachingConfig())
            args = build_parser().parse_args(command[4:]); spec = r3_experiment(job['direction'], variant)
            validate_args(args, spec)
            assert subgroup_config(args).confidence == .6 and r3_config(args) == style_config(variant)
            assert spec.prototype_weight == coteaching_experiment(job['direction']).prototype_weight


def test_dimension_96_and_subgroup_k_are_inferred_from_existing_model():
    student = _small_model(scales=(1., 2., 4.), d_model=96, num_domains=2,
                           use_temporal_msad=True, use_source_prototype_memory=True)
    alignment = ClassConditionalCoTeaching(student, CoTeachingConfig(k=2), torch.device('cpu'))
    style = R3StyleController(student, alignment, StyleConfig())
    source = batch(True, 3)
    result = student(source['x'], source['mask'], compute_domain=False)
    assert result['scale_embeddings'].shape == (3, 3, 96)
    assert style.bank.target_subgroup_mu.shape == (3, 3, 2, 96)
    torch.testing.assert_close(result['scale_logits'], student.classifier(result['scale_embeddings']))


def test_invalid_style_parameters_rejected():
    for changes in ({'max_strength': .3}, {'class_min_count': 1}, {'momentum': 1.},
                    {'eps': 0.}, {'loss_var': float('nan')}, {'ratio_min': 3.}, {'max_level': 3}):
        try:
            replace(StyleConfig(), **changes).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(changes)


def test_relay_failure_stops_later_directions_and_variants():
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp)/'MPUS_GA'
        plan = build_plan('r3_test', package, ('full', 'baseline'), [43, 42], 43)
        calls = []
        def launch(command, **kwargs):
            direction = command[command.index('--experiment')+1]
            calls.append(direction)
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-only-one'
            child = Mock()
            child.wait.return_value = 7 if direction == 'C' else 0
            child.poll.return_value = child.wait.return_value
            return child
        assert run_plan(plan, 'python', package, '/tmp/data', 'all', CoTeachingConfig(), 'GPU-only-one', launch) == 7
        assert calls == list('ABC')


def test_relay_cancellation_terminates_child_without_next_job():
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp)/'MPUS_GA'
        plan = build_plan('r3_test', package, ('full',), [43, 42], 43)
        child = Mock(); child.wait.side_effect = [SuiteStopped(), 143]; child.poll.return_value = None
        launch = Mock(return_value=child)
        try:
            run_plan(plan, 'python', package, '/tmp/data', 'all', CoTeachingConfig(), 'GPU-one', launch)
        except SuiteStopped:
            pass
        else:
            raise AssertionError('Cancellation swallowed')
        assert launch.call_count == 1 and child.terminate.call_count == 1


def test_stop_command_exact_suite_and_owned_gpu_selection():
    table = {
        10: {'ppid': 1, 'args': ['python', '-m', 'MPUS_GA.scripts.run_r2_coteaching_suite', '--worker', '--run-name', 'r2_coteaching_test']},
        11: {'ppid': 10, 'args': ['python', '-m', 'MPUS_GA.trial_temporal.train']},
        12: {'ppid': 1, 'args': ['python', '-m', 'MPUS_GA.scripts.run_r2_coteaching_suite', '--run-name', 'r2_coteaching_test_other']},
        20: {'ppid': 1, 'args': ['python', 'other_training.py']},
        30: {'ppid': 1, 'args': ['python', 't.py']},
        31: {'ppid': 30, 'args': ['python', '-c', 'spawn_main()', '--multiprocessing-fork']},
    }
    assert descendants(select_suite(table, 'r2_coteaching_test'), table) == {10, 11}
    assert select_gpu(table, 'GPU-1', {11: {'GPU-0'}, 20: {'GPU-1'}, 999: {'GPU-1'}}) == {20}
    assert select_gpu(table, 'GPU-1', {11: {'GPU-0'}, 31: {'GPU-1'}}) == {30, 31}
    assert 30 not in select_gpu(table, 'GPU-1', {30: {'GPU-0'}, 31: {'GPU-1'}})


if __name__ == '__main__':
    torch.set_num_threads(2)
    tests = [value for name, value in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test(); print('PASS', test.__name__, flush=True)
    print(len(tests), 'tests passed')
