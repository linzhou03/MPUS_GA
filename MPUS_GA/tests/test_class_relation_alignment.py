from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))

import pytest
import torch
from MPUS_GA.trial_temporal.class_relation_alignment import (
    RelationConfig,RelationAlignment,cross_domain_loss,instance_loss,sample_targets,unique_rows)
from MPUS_GA.scripts.run_relation_suite import plan


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def test_feature_gradient_not_probability_or_source_key_gradient():
    torch.manual_seed(2)
    zs=torch.randn(9,3,8,requires_grad=True)
    zt=torch.randn(4,3,8,requires_grad=True)
    logits=torch.randn(4,3,3,requires_grad=True)
    q=logits.softmax(-1)
    loss=cross_domain_loss(zs,torch.arange(9)%3,zt,q,torch.eye(3).repeat(3,1,1),torch.ones(4),RelationConfig())
    loss.backward()
    assert zt.grad.abs().sum()>0 and zs.grad is None and logits.grad is None


def test_relation_uses_source_only_and_freezes_after_warmup():
    learner=RelationAlignment(RelationConfig(),torch.device('cpu'))
    p=torch.eye(3)[:,None].repeat(1,3,1)
    learner.observe_source(p,torch.arange(3),300)
    expected=learner.relation().clone()
    torch.testing.assert_close(expected,torch.eye(3).repeat(3,1,1))
    learner.observe_source(torch.ones_like(p)/3,torch.arange(3),301)
    torch.testing.assert_close(learner.relation(),expected)
    with pytest.raises(RuntimeError,match='truth'):
        learner.loss(None,[],[],{},dict(y=torch.zeros(1)),1,torch.device('cpu'))


def test_local_source_matching_handles_missing_class_and_has_unit_mass():
    c=RelationConfig(neighbors_per_class=1)
    similarity=torch.tensor([[.9,.1,.3],[.2,.8,.4]])
    labels=torch.tensor([0,0,1])
    probability=torch.tensor([[.8,.1,.1],[.2,.7,.1]])
    weights=sample_targets(similarity,probability,labels,torch.eye(3),c)
    torch.testing.assert_close(weights.sum(-1),torch.ones(2))
    assert weights[0,1]==0 and weights[1,0]==0
    assert weights[0,0]>weights[0,2] and weights[1,2]>weights[1,1]


def test_instance_loss_matches_views_and_singletons_are_safe():
    z=torch.eye(4)[:,None].repeat(1,3,1).requires_grad_()
    q=torch.ones(4,3,3)/3;r=torch.eye(3).repeat(3,1,1);c=RelationConfig()
    matched=instance_loss(z,z.clone(),q,r,c)
    mismatched=instance_loss(z,z.flip(0),q,r,c)
    assert matched<mismatched
    matched.backward();assert torch.isfinite(z.grad).all()
    assert instance_loss(z[:1],z[:1],q[:1],r,c).item()==0
    assert unique_rows([(1,1,1),(1,1,1),(1,1,2)],torch.device('cpu')).tolist()==[0,2]


def test_exact_main_only_six_direction_plan():
    queues,summary=plan('crco_feature_test','/tmp')
    assert summary['groups']==1 and summary['jobs']==6 and summary['folds']==102
    assert [x['direction'] for x in queues['0']]==list('ACE')
    assert [x['direction'] for x in queues['1']]==list('BDF')
    assert all(x['seeds']==[42] and x['variant']=='crco_feature' for q in queues.values() for x in q)
