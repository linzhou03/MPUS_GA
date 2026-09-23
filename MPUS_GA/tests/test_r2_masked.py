from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from MPUS_GA.tests.test_multiscale_coteaching import model as small_model
from MPUS_GA.tests.test_msmr import dataset
from MPUS_GA.trial_temporal.data import collate_multiscale, PreparedMultiSource
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.masked_multiscale import pack_windows, coordinated_mask
from MPUS_GA.trial_temporal.r2_masked_reconstruction import (
    R2MaskConfig, attach_reconstruction, bounded_auxiliary_gradients)
from MPUS_GA.scripts.run_r2_msmr_suite import main_plan, training_command


@pytest.fixture(autouse=True)
def runtime():
    torch.set_num_threads(2)


def batches():
    data = dataset(6)
    return (collate_multiscale([data._item(i, True) for i in range(6)]),
            collate_multiscale([data._item(i, False) for i in range(6)]))


def parts(m):
    optimizer = torch.optim.AdamW(m.parameters(), lr=.0005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    bank = train.PrototypeBank(1, 3, 3, 32, .9, .15, .1, torch.device('cpu'))
    memory = train.SourceMultiPrototypeMemory(3, 3, 4, 32, .9, torch.device('cpu'))
    return optimizer, scheduler, bank, memory


def step(m, state, source, target, iteration):
    optimizer, scheduler, bank, memory = state
    return train.train_step(m, [source], target, optimizer, scheduler, torch.device('cpu'), iteration,
                            train.r2_masked_experiment('A'), torch.ones(1, 3) / 3, bank,
                            .1, 5., 300, 600, .6, source_prototype_memory=memory)


def test_base_profile_is_original_r2_not_peer_or_subgroup():
    for direction in 'ABCDEF':
        expected = asdict(train.EXPERIMENTS['A_R2'])
        current = asdict(train.r2_masked_experiment(direction))
        identity = {'name', 'description', 'transfer_direction', 'ablation', 'source_domains',
                    'target_dataset', 'target_subject_count', 'target_trials'}
        assert {k:v for k,v in current.items() if k not in identity} == {k:v for k,v in expected.items() if k not in identity}
        assert not current['use_subgroup_alignment'] and not current['use_multiscale_coteaching']
        assert current['prototype_weight'] == .1


def test_checkpointed_chunking_preserves_outputs_and_gradients_without_dropout():
    left = small_model()
    right = deepcopy(left)
    right.activation_checkpointing = True
    right.spatial_chunk_windows = 5
    source, _ = batches()
    a = left(source['x'], source['mask'])
    b = right(source['x'], source['mask'])
    for key in ('logits', 'scale_logits', 'scale_embeddings'):
        torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=1e-5)
    a['logits'].square().sum().backward()
    b['logits'].square().sum().backward()
    for p, q in zip(left.parameters(), right.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)


def test_attach_preserves_base_weights_rng_and_predictions():
    m = small_model().eval()
    source, _ = batches()
    before = {k:v.clone() for k,v in m.state_dict().items()}
    expected = m(source['x'], source['mask'])['probability']
    rng = torch.get_rng_state()
    attach_reconstruction(m, 4)
    assert torch.equal(rng, torch.get_rng_state())
    actual = m(source['x'], source['mask'])['probability']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, value in before.items():
        torch.testing.assert_close(m.state_dict()[key], value, atol=0, rtol=0)


def test_zero_auxiliary_retains_original_r2_updates():
    left = small_model()
    right = deepcopy(left)
    attach_reconstruction(right, 4, replace(R2MaskConfig(), source_weight=0., target_weight=0.))
    ls, rs = parts(left), parts(right)
    source, target = batches()
    for iteration in (1, 350, 650):
        a = step(left, ls, source, target, iteration)
        b = step(right, rs, source, target, iteration)
        assert a['total'] == pytest.approx(b['total'], abs=1e-6)
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], atol=2e-6, rtol=1e-5)


def test_conflict_projection_and_norm_bound():
    base = [torch.tensor([1., 0.]), torch.tensor([0., 2.])]
    aux = [torch.tensor([-4., 3.]), torch.tensor([4., -2.])]
    result, diagnostics = bounded_auxiliary_gradients(base, aux, .2)
    assert diagnostics['conflict_projected']
    assert diagnostics['gradient_dot_after'] >= -1e-6
    assert diagnostics['applied_gradient_ratio'] <= .200001
    assert sum(x.square().sum() for x in result) > 0
    zero, _ = bounded_auxiliary_gradients([torch.zeros(2)], [torch.ones(2)], .2)
    assert zero[0].abs().sum() == 0


def test_hidden_cross_scale_cells_do_not_leak_and_encoder_receives_gradient():
    m = small_model().eval()
    head = attach_reconstruction(m, 4)
    head.eval()
    source, _ = batches()
    packed, valid = pack_windows(source['x'], source['mask'])
    hidden = coordinated_mask(valid, head.regions, .15)
    x = deepcopy(source['x'])
    for scale, start, width in [('1s',0,4),('2s',4,2),('4s',6,1)]:
        mask = hidden[:, :, start:start+width].reshape(len(valid), -1, 62, 5)[:, :x[scale].shape[1]]
        x[scale][mask] += 10000
    loss, _, expected = head.reconstruction(m, source['x'], source['mask'], hidden)
    _, _, actual = head.reconstruction(m, x, source['mask'], hidden)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    loss.backward()
    assert m.spatial.input_projection.weight.grad.abs().sum() > 0
    assert m.temporal['1s'].transformer.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.classifier.parameters())


def test_actual_three_stage_training_updates_nonzero_auxiliary_gradients():
    m = small_model()
    head = attach_reconstruction(m, 4)
    state = parts(m)
    source, target = batches()
    for iteration in (1, 350, 650):
        record = step(m, state, source, target, iteration)['r2_masked_reconstruction']
        assert record['source_reconstruction'] > 0 and record['target_reconstruction'] > 0
        assert record['applied_auxiliary_gradient_norm'] > 0
        assert record['applied_gradient_ratio'] <= .200001
    assert head.state()['encoder_gradient_active_steps'] == 3
    with pytest.raises(RuntimeError, match='target truth'):
        head.accumulate(m, [source], source, torch.device('cpu'))


def test_six_main_directions_with_original_r2_batch_and_learning_rate():
    plan, queues, summary = main_plan('r2_msmr_test', Path('/tmp/test'), ['0', '1'])
    assert summary['folds'] == 306 and summary['direction_seed_combinations'] == 18
    assert [p['direction'] for p in queues['0']] == list('ACE')
    assert [p['direction'] for p in queues['1']] == list('BDF')
    for item in plan:
        command = training_command('python', item, '/tmp/data', 'all', R2MaskConfig())
        args = train.build_parser().parse_args(command[4:])
        train.validate_args(args, train.r2_masked_experiment(item['direction']))
        assert args.method == 'r2_msmr' and args.random_seeds == [42,43,44]
        assert (args.source_batch_size, args.target_batch_size, args.learning_rate) == (24,16,.0005)
        assert args.d_model == 96 and args.cuda_memory_budget_gib == 8


def test_fold_checkpoint_contains_model_and_exact_inference_context(tmp_path, monkeypatch):
    source, target = dataset(6), dataset(45)
    prepared = PreparedMultiSource(('seed_vii',), (source,), source.stats)
    monkeypatch.setattr(train, 'prepare_target', lambda *a: target)
    monkeypatch.setattr(train, 'FIXED_UDA_PROTOCOL', replace(train.FIXED_UDA_PROTOCOL, training_iterations=3))
    args = train.build_parser().parse_args(['--method','r2_msmr','--experiment','A',
        '--result-root',str(tmp_path),'--data-dir',str(tmp_path),'--device','cpu',
        '--source-batch-size','6','--target-batch-size','6','--d-model','32',
        '--spatial-layers','1','--temporal-layers','1','--fusion-layers','1',
        '--dim-feedforward','64','--dropout','0'])
    spec = train.r2_masked_experiment('A')
    train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
    path = tmp_path/'A/seed_42_subject_01.json'
    result = json.loads(path.read_text())
    checkpoint = torch.load(path.with_suffix('.pt'), weights_only=True)
    assert len(result['predictions']) == 45
    assert result['r2_masked_reconstruction']['encoder_gradient_active_steps'] == 3
    assert result['protocol']['target_evaluations'] == 1
    # Capture the exact constructor used by run_fold for the reload check.
    saved_constructor = train.MultiScaleMultiSourceDANN
    seen = []
    def capture(**kwargs):
        seen.append(kwargs)
        return saved_constructor(**kwargs)
    monkeypatch.setattr(train, 'MultiScaleMultiSourceDANN', capture)
    args.result_root = tmp_path/'capture'
    train.run_fold(args, spec, prepared, 42, 1, torch.device('cpu'))
    restored = saved_constructor(**seen[0])
    attach_reconstruction(restored, 4)
    restored.load_state_dict(checkpoint['model'])
    restored.eval()
    batch = collate_multiscale([target[0]])
    with torch.no_grad():
        output = restored(batch['x'], batch['mask'], **checkpoint['inference_context'])['probability'][0]
    torch.testing.assert_close(output, torch.tensor(result['predictions'][0]['probability']), atol=2e-6, rtol=1e-5)
