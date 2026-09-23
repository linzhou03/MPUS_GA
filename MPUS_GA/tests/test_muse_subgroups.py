import torch
import torch.nn.functional as F
from MPUS_GA.tests.test_muse_integration import fixture, with_ids, batch


def subgroup_fixture():
    parts=fixture('gated');muse=parts[1].muse;bank=parts[1].bank
    source_batch=with_ids(batch(True,18));target_batch=with_ids(batch(False,18))
    labels=source_batch['y'];base=F.one_hot(labels,32).float()[:,None].repeat(1,3,1)
    source=base.clone().requires_grad_();target=(base+F.one_hot(torch.full_like(labels,3),32).float()[:,None]*.1).requires_grad_()
    probability=(F.one_hot(labels,3).float()*.94+.02)[:,None].repeat(1,3,1)
    muse.observe(probability,target_batch,650)
    muse.tracker.H.copy_(torch.tensor([4.,0.,4.]));muse.tracker.M.fill_(8);muse.tracker.E.fill_(8)
    keys=[(0,1,1,i) for i in range(18)];valid=torch.ones(18,3).bool()
    for d,z in enumerate((source,target)):bank.observe(d,keys,z,labels,torch.ones(18),valid,600)
    bank.refresh(600);muse.after_bank_refresh(bank)
    return parts,source,target,source_batch,target_batch


def test_gate_applies_before_normalization_and_both_domains_receive_gradients():
    parts,source,target,sb,tb=subgroup_fixture();muse=parts[1].muse
    loss,record=muse.subgroup_loss(parts[1],[{'scale_embeddings':source}],[sb],{'scale_embeddings':target},tb)
    assert loss>0 and record['gated_subgroup_loss_per_class'][1]==0
    loss.backward();assert source.grad.abs().sum()>0 and target.grad.abs().sum()>0
    assert source.grad[sb['y']==1].abs().sum()==0


def test_no_match_means_zero_and_no_forced_subgroup_fallback():
    parts,source,target,sb,tb=subgroup_fixture();parts[1].bank.matches.zero_()
    loss,_=parts[1].muse.subgroup_loss(parts[1],[{'scale_embeddings':source}],[sb],{'scale_embeddings':target},tb)
    assert loss==0


def test_starved_class_retains_conservative_centroid_fallback():
    parts,*_=subgroup_fixture();alignment=parts[1];alignment.readiness.fill_(1)
    strength=alignment.centroid_strength(650)
    assert torch.all(strength[:,1]==1) and torch.all(strength[:,0]<1)
    alignment.muse.iteration=599
    assert torch.all(alignment.centroid_strength(599)==1)
