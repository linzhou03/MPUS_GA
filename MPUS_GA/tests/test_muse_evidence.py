from copy import deepcopy
from dataclasses import replace
import io
import numpy as np
import pytest
import torch

from MPUS_GA.trial_temporal.multiscale_evidence import (
    MuseConfig, TargetSupervisionTracker, multiscale_evidence, target_supervision_loss,
    assert_scale_trial_ids, reject_target_truth,
)
from MPUS_GA.trial_temporal.data import ScaleArrays, MultiScaleTrialDataset, collate_multiscale, UnlabeledMultiScaleView


def evidence(rows, config=MuseConfig()):
    p=torch.tensor(rows,dtype=torch.float32)
    if p.ndim==2:p=p[:,None].repeat(1,3,1)
    return multiscale_evidence(p,config)


def test_masks_hard_priority_and_raw_mean():
    p=torch.tensor([[.9,.05,.05],[.48,.04,.48],[1/3]*3])[:,None].repeat(1,3,1)
    before=p.clone();e=multiscale_evidence(p,MuseConfig())
    assert e['hard_mask'].tolist()==[True,False,False]
    assert e['partial_mask'].tolist()==[False,True,False]
    assert torch.all(e['hard_mask'].int()+e['partial_mask'].int()+e['unsupervised_mask'].int()==1)
    torch.testing.assert_close(e['q'],p.mean(1),atol=0,rtol=0)
    torch.testing.assert_close(p,before,atol=0,rtol=0)


def test_partial_sets_ignore_order():
    e=evidence([[[.55,.01,.44],[.44,.01,.55],[.52,.02,.46]]])
    assert e['partial_mask'].item() and e['top2_votes'].item()==3
    assert set(e['candidate_set'][0].tolist())=={0,2}


def test_partial_loss_exact_and_gradient():
    e=evidence([[.48,.04,.48]])
    p=torch.tensor([[[.2,.3,.5]]*3]);logits=p.log().requires_grad_()
    hard,partial=target_supervision_loss(logits,e,torch.ones(3),MuseConfig())
    torch.testing.assert_close(partial,-torch.log(torch.tensor(.7)+1e-8))
    assert hard==0;partial.backward();assert logits.grad.abs().sum()>0


def test_tracker_exact_counts():
    tracker=TargetSupervisionTracker(3,MuseConfig())
    tracker.update(evidence([[.9,.05,.05],[.48,.04,.48]]))
    torch.testing.assert_close(tracker.H,torch.tensor([.05,0,0]))
    torch.testing.assert_close(tracker.P,torch.tensor([.025,0,.025]))
    torch.testing.assert_close(tracker.E,torch.tensor([.0625,0,.0125]))


def test_starvation_increases_then_falls():
    t=TargetSupervisionTracker(3,MuseConfig())
    for _ in range(40):t.update(evidence([[.95,.025,.025]]*16))
    low=t.starvation()[0].item()
    for _ in range(40):t.update(evidence([[.65,.2,.15]]*16))
    high=t.starvation()[0].item();assert high>low
    for _ in range(40):t.update(evidence([[.95,.025,.025]]*16))
    assert t.starvation()[0]<high


def test_empty_extreme_batches_finite():
    t=TargetSupervisionTracker(3,MuseConfig())
    for p in (torch.empty(0,3,3),torch.eye(3)[:,None].repeat(1,3,1),torch.ones(16,3,3)/3):
        t.update(multiscale_evidence(p,MuseConfig()))
        for value in (t.M,t.E,t.H,t.coverage(),t.starvation(),t.class_weights(),t.gate(650)):
            assert torch.isfinite(value).all()


def test_support_and_starvation_gate_and_schedule():
    t=TargetSupervisionTracker(3,MuseConfig())
    t.H.copy_(torch.tensor([15.,32.,32.]));t.M.fill_(40);t.E.copy_(torch.tensor([40.,30.,10.]))
    assert t.gate(650)[0]==0 and t.gate(650)[1]>t.gate(650)[2]
    assert not t.gate(599).any()


def test_tracker_checkpoint_roundtrip():
    t=TargetSupervisionTracker(3,MuseConfig());t.update(evidence([[.95,.025,.025]]*16))
    stream=io.BytesIO();torch.save(t.state_dict(),stream);stream.seek(0)
    restored=TargetSupervisionTracker(3,MuseConfig());restored.load_state_dict(torch.load(stream,weights_only=True))
    for k,v in t.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)


def arrays(reverse=False,bad=False):
    a=ScaleArrays(np.arange(4*62*5,dtype=np.float32).reshape(4,62,5),np.array([0,0,1,1]),
                  np.ones(4,dtype=np.int64),np.ones(4,dtype=np.int64),np.array([1,1,2,2]),np.array([0,1,0,1]))
    if reverse:a=ScaleArrays(**{k:getattr(a,k)[::-1].copy() for k in a.__dataclass_fields__})
    if bad:a=replace(a,trials=a.trials+10)
    return a


def test_loader_reorders_actual_trial_keys_and_rejects_missing_ids():
    stats={s:(np.zeros((62,5)),np.ones((62,5))) for s in (1.,2.,4.)}
    dataset=MultiScaleTrialDataset({1.:arrays(),2.:arrays(True),4.:arrays()},stats,0)
    b=collate_multiscale([UnlabeledMultiScaleView(dataset)[i] for i in range(2)])
    assert_scale_trial_ids(b);assert 'y' not in b
    torch.testing.assert_close(b['x']['1s'],b['x']['2s'])
    broken=deepcopy(b);broken['trial_key_by_scale']['2s'][0,-1]+=1
    with pytest.raises(ValueError,match='Trial IDs'):assert_scale_trial_ids(broken)
    with pytest.raises(ValueError,match='Trial keys'):
        MultiScaleTrialDataset({1.:arrays(),2.:arrays(bad=True),4.:arrays()},stats,0)


@pytest.mark.parametrize('key',['y','target_y','target_label','target_labels'])
def test_target_truth_rejected(key):
    with pytest.raises(RuntimeError,match='unlabeled'):reject_target_truth({key:torch.tensor([1])})
