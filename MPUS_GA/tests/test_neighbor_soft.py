from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from MPUS_GA.trial_temporal.neighbor_soft import (NeighborConfig, NeighborLearning, SoftWeight,
                                                collect_bank, refine_bank)
from MPUS_GA.scripts.run_neighbor_soft_suite import make_plan
from MPUS_GA.trial_temporal.pcdiag import rng_state, restore_rng
from MPUS_GA.trial_temporal import train
from MPUS_GA.tests.test_pcdiag import settings, assert_tree
from MPUS_GA.tests.test_r2_masked import batches, parts


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def bank():
    return dict(ids=torch.tensor([[1, 1, i] for i in range(4)]),
        embeddings=torch.tensor([[[1., 0.]], [[.99, .01]], [[0., 1.]], [[.01, .99]]]).repeat(1, 3, 1),
        raw_probability=torch.tensor([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9], [.05, .8, .15]]))


def test_neighbors_exclude_self_and_can_change_class():
    value = bank()
    q, indices = refine_bank(value, 1)
    assert indices.flatten().tolist() == [1, 0, 3, 2]
    assert q.argmax(-1).tolist() == [1, 0, 1, 2]
    assert torch.equal(value['raw_probability'], bank()['raw_probability'])
    value['ids'][1] = value['ids'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        refine_bank(value, 1)


def test_weight_finite_constant_batch_and_downweights_uncertainty():
    hook = SoftWeight(NeighborConfig())
    weights = hook.update(torch.tensor([[.4, .3, .3], [.9, .05, .05], [.7, .2, .1]]))
    assert 0 <= weights[0] < weights[2] <= weights[1] == 1
    constant = SoftWeight(NeighborConfig()).update(torch.full((1, 3), 1 / 3))
    assert torch.isfinite(constant).all() and constant.item() == 1


def test_plan_counts_order_seeds_and_gpu_constraint(tmp_path):
    csu, c = make_plan('csu', 'neighbor_soft_test', tmp_path)
    xju, x = make_plan('xju', 'neighbor_soft_test', tmp_path)
    assert [phase for phase, _ in csu] == ['full', 'r2']
    assert [phase for phase, _ in xju] == ['raw_ce', 'raw_soft', 'neighbor_ce']
    assert c['direction_seed_jobs'] == 24 and c['folds'] == 408
    assert x['direction_seed_jobs'] == 9 and x['folds'] == 156
    for phase, queues in xju:
        assert list(queues) == ['1']
        assert [i['direction'] for i in queues['1']] == list('ABC')
        assert all(i['seeds'] == [42] for i in queues['1'])
    for phase, queues in csu:
        assert [i['direction'] for i in queues['0']] == list('ACE')
        assert [i['direction'] for i in queues['1']] == list('BDF')
        assert all(i['seeds'] == ([42, 43, 44] if phase == 'full' else [42]) for q in queues.values() for i in q)


def test_bank_preserves_model_modes_rng_and_excludes_labels(tmp_path):
    args, spec, prepared = settings(tmp_path)
    model = train.build_fold_model(args, spec, prepared, torch.device('cpu')).train()
    _, target = batches()
    state = rng_state(torch.device('cpu'))
    model_before = deepcopy(model.state_dict())
    memory = collect_bank(model, [target], torch.device('cpu'), {'compute_domain': False})
    assert_tree(rng_state(torch.device('cpu')), state)
    assert_tree(model.state_dict(), model_before)
    assert model.training and len(memory['ids']) == len(target['trial_id'])
    with pytest.raises(RuntimeError, match='labels'):
        collect_bank(model, [{**target, 'y': torch.zeros(len(target['trial_id']))}], torch.device('cpu'), {})


@pytest.mark.parametrize('variant', ['full', 'raw_ce', 'raw_soft', 'neighbor_ce'])
def test_integration_warmup_is_exact_then_auxiliary_reaches_classifier(tmp_path, variant):
    args, spec, prepared = settings(tmp_path)
    left = train.build_fold_model(args, spec, prepared, torch.device('cpu'))
    right = deepcopy(left)
    ls, rs = parts(left), parts(right)
    source, target = batches()
    learner = NeighborLearning(NeighborConfig(variant), [target], torch.device('cpu'))
    state = rng_state(torch.device('cpu'))
    a = train.train_step(left, [source], target, ls[0], ls[1], torch.device('cpu'), 1, spec,
                        torch.ones(1, 3) / 3, ls[2], .1, 5., 300, 600, .6, source_prototype_memory=ls[3])
    expected = rng_state(torch.device('cpu'))
    restore_rng(state, torch.device('cpu'))
    b = train.train_step(right, [source], target, rs[0], rs[1], torch.device('cpu'), 1, spec,
                        torch.ones(1, 3) / 3, rs[2], .1, 5., 300, 600, .6,
                        source_prototype_memory=rs[3], neighbor_learning=learner)
    assert b.pop('neighbor_learning')['active'] is False
    assert_tree(a, b)
    assert_tree(left.state_dict(), right.state_dict())
    assert_tree(rng_state(torch.device('cpu')), expected)
    before = right.classifier.weight.detach().clone()
    record = train.train_step(right, [source], target, rs[0], rs[1], torch.device('cpu'), 650, spec,
                        torch.ones(1, 3) / 3, rs[2], .1, 5., 300, 600, .6,
                        source_prototype_memory=rs[3], neighbor_learning=learner)
    assert record['neighbor_learning']['added_loss'] > 0
    assert not torch.equal(before, right.classifier.weight)
    assert len(learner.events) == 1 and len(learner.snapshots) == 1
    assert not learner.bank['weights'].requires_grad


def test_target_loss_uses_batch_denominator():
    learner = NeighborLearning(NeighborConfig('full'), [], torch.device('cpu'))
    learner.bank = dict(labels=torch.tensor([1, 2]), weights=torch.tensor([.01, .01]),
                        raw_probability=torch.tensor([[.1, .8, .1], [.1, .1, .8]]))
    learner.index = {(1, 1, 1): 0, (1, 1, 2): 1}
    learner.last_refresh = 650
    output = {'logits': torch.zeros(2, 3, requires_grad=True)}
    batch = dict(subject_id=torch.tensor([1, 1]), session_id=torch.tensor([1, 1]), trial_id=torch.tensor([1, 2]))
    loss, record = learner.loss(None, output, batch, 650, 1., {})
    torch.testing.assert_close(loss, torch.log(torch.tensor(3.)) * .01)
    loss.backward()
    assert output['logits'].grad.abs().sum() > 0
