from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from MPUS_GA.trial_temporal import train_msmr as train
from MPUS_GA.trial_temporal.data import (ScaleArrays, MultiScaleTrialDataset,
                                       PreparedMultiSource, collate_multiscale)
from MPUS_GA.trial_temporal.masked_multiscale import (
    MSMRConfig, MaskedMultiScaleModel, pack_windows, coordinated_mask, masked_reconstruction_loss)
from MPUS_GA.scripts import run_msmr_suite as suite
from MPUS_GA.scripts.stop_owned_training import select_suite


def dataset(trials=6):
    arrays, stats = {}, {}
    rng = np.random.default_rng(123)
    for scale in (1., 2., 4.):
        counts = [(8 if i % 2 == 0 else 7) // int(scale) for i in range(trials)]
        arrays[scale] = ScaleArrays(
            features=rng.normal(size=(sum(counts), 62, 5)).astype(np.float32),
            labels=np.repeat(np.arange(trials) % 3, counts), subjects=np.ones(sum(counts), np.int64),
            sessions=np.ones(sum(counts), np.int64), trials=np.repeat(np.arange(trials), counts),
            window_ids=np.concatenate([np.arange(n) for n in counts]))
        stats[scale] = (np.zeros((62, 5), np.float32), np.ones((62, 5), np.float32))
    return MultiScaleTrialDataset(arrays, stats, domain_id=0)


def batch(labeled=True):
    data = dataset()
    return collate_multiscale([data._item(i, include_label=labeled) for i in range(2)])


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_real_time_slots_and_partial_tail():
    b = batch()
    for scale, x in b['x'].items():
        x[:] = torch.arange(x.shape[1])[None, :, None, None] + int(scale[0]) * 100
    values, valid = pack_windows(b['x'], b['mask'])
    assert values[0, 0, :, 0, 0].tolist() == [100, 101, 102, 103, 200, 201, 400]
    assert values[0, 1, :, 0, 0].tolist() == [104, 105, 106, 107, 202, 203, 401]
    assert valid[1, 1].tolist() == [True, True, True, False, True, False, False]
    b['mask']['2s'][0, -1] = False
    with pytest.raises(ValueError, match='lengths'):
        pack_windows(b['x'], b['mask'])


def test_mask_is_shared_by_overlapping_scales_and_hidden_values_cannot_leak():
    model = MaskedMultiScaleModel(replace(MSMRConfig(), d_model=16, dropout=0.)).eval()
    b = batch()
    values, valid = pack_windows(b['x'], b['mask'])
    hidden = coordinated_mask(valid, model.spatial.regions, .9, torch.Generator().manual_seed(4))
    for trial in range(2):
        for group in range(2):
            masks = hidden[trial, group][valid[trial, group]]
            assert torch.equal(masks, masks[:1].expand_as(masks))
    assert (valid[..., None, None] & ~hidden).flatten(1).any(1).all()
    changed = values.clone()
    changed[hidden] += 10000
    with torch.no_grad():
        before = model.encode_packed(values, valid, hidden)
        after = model.encode_packed(changed, valid, hidden)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_padding_cannot_change_logits():
    model = MaskedMultiScaleModel(replace(MSMRConfig(), d_model=16, dropout=0.)).eval()
    d = dataset()
    together = collate_multiscale([d[0], d[1]])
    alone = collate_multiscale([d[1]])
    with torch.no_grad():
        a = model(together['x'], together['mask'])['logits'][1]
        b = model(alone['x'], alone['mask'])['logits'][0]
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


def test_reconstruction_equal_trial_and_scale_weighting():
    target = torch.zeros(2, 4, 7, 62, 5)
    predicted = torch.ones_like(target)
    predicted[1] = 3
    hidden = torch.ones_like(target, dtype=torch.bool)
    hidden[0, 1:] = False
    loss, by_scale = masked_reconstruction_loss(predicted, target, hidden)
    assert loss.item() == pytest.approx(1.5)
    assert all(v == pytest.approx(1.5) for v in by_scale.values())


def test_target_reconstruction_reaches_encoder_without_classifier_or_labels():
    model = MaskedMultiScaleModel(replace(MSMRConfig(), d_model=16, dropout=0.))
    b = batch(False)
    loss, info = model.reconstruct(b['x'], b['mask'])
    loss.backward()
    assert loss.item() > 0 and info['masked_cells'] > 0
    assert model.spatial.raw.weight.grad.abs().sum() > 0
    assert model.temporal.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad is None


def test_production_training_step_and_target_label_guard():
    config = replace(MSMRConfig(), d_model=16, dropout=0.)
    model = MaskedMultiScaleModel(config)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    before = model.classifier.weight.detach().clone()
    record = train.train_step(model, opt, batch(), batch(False), torch.device('cpu'), config)
    assert min(record[k] for k in ('classification', 'source_reconstruction', 'target_reconstruction')) > 0
    assert not torch.equal(before, model.classifier.weight)
    with pytest.raises(ValueError, match='must not receive labels'):
        train.train_step(model, opt, batch(), batch(), torch.device('cpu'), config)


def test_fold_saves_final_weights_metrics_predictions_and_resumes(tmp_path, monkeypatch):
    source, target = dataset(6), dataset(45)
    prepared = PreparedMultiSource(('seed_vii',), (source,), source.stats)
    monkeypatch.setattr(train, 'prepare_target', lambda *a: target)
    monkeypatch.setattr(train, 'FIXED_UDA_PROTOCOL', replace(train.FIXED_UDA_PROTOCOL, training_iterations=3))
    calls = []
    original_eval = train.evaluate
    def evaluate(*args):
        calls.append(len(args[1].dataset))
        return original_eval(*args)
    monkeypatch.setattr(train, 'evaluate', evaluate)
    args = SimpleNamespace(experiment='A', result_root=tmp_path, data_dir=tmp_path, device='cpu')
    config = replace(MSMRConfig(), d_model=16, dropout=0., source_batch_size=2, target_batch_size=2)
    train.run_fold(args, prepared, 1, 42, config, 'synthetic-test-only')
    path = tmp_path / 'A/seed_42_subject_01.json'
    result = json.loads(path.read_text())
    assert calls == [45, 6]
    assert result['activity']['target_reconstruction_steps'] == 3
    assert len(result['predictions']) == 45
    assert result['protocol']['target_evaluations'] == 1
    checkpoint = torch.load(path.with_suffix('.pt'), weights_only=True)
    restored = MaskedMultiScaleModel(config)
    restored.load_state_dict(checkpoint['model'])
    evaluation, _ = original_eval(restored, train.loader(target, 2), torch.device('cpu'))
    assert evaluation == result['evaluation']
    stamp = path.stat().st_mtime_ns
    train.run_fold(args, prepared, 1, 42, config, 'synthetic-test-only')
    assert path.stat().st_mtime_ns == stamp
    assert (tmp_path / 'A/summary_seed_42.csv').exists()
    with pytest.raises(ValueError, match='different identity'):
        train.run_fold(args, prepared, 1, 42, config, 'changed')


def test_six_main_jobs_three_seeds_two_relays():
    plan, queues, summary = suite.main_plan('msmr_test', Path('/tmp/test'), ['0', '1'])
    assert summary['direction_jobs'] == 6
    assert summary['direction_seed_combinations'] == 18
    assert summary['folds'] == 306
    assert summary['switch_interval_seconds'] == 1
    assert [p['direction'] for p in queues['0']] == list('ACE')
    assert [p['direction'] for p in queues['1']] == list('BDF')
    assert summary['gpu_queues']['0']['folds'] == 156
    assert summary['gpu_queues']['1']['folds'] == 150
    for p in plan:
        command = suite.training_command('python', p, '/data', 'all', MSMRConfig())
        assert p['seeds'] == [42, 43, 44] and p['variant'] == 'full'
        assert command[3] == 'MPUS_GA.trial_temporal.train_msmr'


def test_parent_holds_lock_across_worker_launch_and_blocks_duplicate(tmp_path, monkeypatch):
    data = tmp_path / 'data/seed_v/window_1s'
    data.mkdir(parents=True)
    (data / 'example.npz').touch()
    monkeypatch.setattr(suite, 'resolve_gpu_uuid', lambda g: 'GPU-' + g)
    monkeypatch.setattr(suite, '_code_hashes', lambda p: {'test': 'same'})
    descriptors = []
    def spawn(*a, **kw):
        descriptors.append(os.dup(kw['pass_fds'][0]))
        return SimpleNamespace(pid=1234)
    monkeypatch.setattr(suite.subprocess, 'Popen', spawn)
    monkeypatch.setattr(suite.sys, 'argv', ['run_msmr_suite', '--run-name', 'msmr_test',
                                         '--data-dir', str(tmp_path / 'data'), '--output-root', str(tmp_path)])
    try:
        suite.main()
        with pytest.raises(SystemExit, match='already running'):
            suite.main()
        assert len(descriptors) == 1
    finally:
        for fd in descriptors:
            os.close(fd)


def test_stopper_recognizes_only_exact_new_training_run():
    def process(root):
        return {'args': ['python', '-m', 'MPUS_GA.trial_temporal.train_msmr', '--result-root', root]}
    table = {1: process('/x/results_msmr_test_full'), 2: process('/x/results_unrelated')}
    assert select_suite(table, 'msmr_test_full') == {1}
