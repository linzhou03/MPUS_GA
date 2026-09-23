from dataclasses import dataclass
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[2]))
from MPUS_GA.trial_temporal.uncertainty_pseudo import (
    TrialBank, UncertaintyConfig, UncertaintyPseudo, entropy_weights, weighted_ce)
from MPUS_GA.scripts.run_uncertainty_suite import plan, command
from MPUS_GA.scripts.stop_owned_training import select_suite


def evidence():
    return dict(ids=torch.tensor([[1, 1, i] for i in range(4)]),
        embeddings=torch.tensor([[[1., 0.]], [[.99, .01]], [[0., 1.]], [[.01, .99]]]).repeat(1, 3, 1),
        raw_probability=torch.tensor([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9], [.05, .8, .15]]))


def test_temporal_average_self_exclusion_and_no_recursive_feedback():
    bank=TrialBank(UncertaintyConfig(neighbors=1,history_refreshes=2))
    a=evidence();bank.update(a,301)
    q,idx=bank.query(a['embeddings'],a['ids'])
    assert idx.flatten().tolist()==[1,0,3,2]
    assert q.argmax(-1).tolist()==[1,0,1,2]
    b=evidence();b['raw_probability']=b['raw_probability'].roll(1,-1)
    bank.update(b,351)
    q,_=bank.query(a['embeddings'],a['ids'])
    torch.testing.assert_close(q,((a['raw_probability']+b['raw_probability'])/2)[idx].squeeze(1))
    bank.update(b,401)
    torch.testing.assert_close(bank.probability,b['raw_probability'])
    assert list(bank.history_iterations)==[351,401]
    b['ids'][1]=b['ids'][0]
    with pytest.raises(ValueError,match='unique'):bank.update(b,451)
    with pytest.raises(ValueError,match='identity'):bank.query(a['embeddings'],a['ids']+10)


def test_paper_entropy_formula_batch_mean_and_gradient_boundary():
    q=torch.tensor([[1.,0.,0.],[1/3,1/3,1/3]],requires_grad=True)
    w=entropy_weights(q)
    torch.testing.assert_close(w,torch.tensor([1.,math.exp(-1)]))
    logits=torch.tensor([[.1,.8,.1],[.6,.2,.2]],requires_grad=True)
    loss=weighted_ce(logits,q)
    expected=(w*F.cross_entropy(logits,q.argmax(-1),reduction='none')).sum()/2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert q.grad is None and logits.grad.abs().sum()>0
    assert logits.grad[1].abs().sum()>0  # Uniform uncertainty is downweighted, not rejected.


@dataclass(frozen=True)
class Consensus:
    probability: torch.Tensor
    pseudo_label: torch.Tensor
    confidence: torch.Tensor
    valid_mask: torch.Tensor
    vote_count: torch.Tensor
    js_divergence: torch.Tensor


def test_replaces_consensus_even_when_domain_filter_rejects_all():
    e=evidence();p=e['raw_probability'];n=len(p)
    learner=UncertaintyPseudo(UncertaintyConfig(neighbors=1),[],torch.device('cpu'))
    learner.bank.update(e,301)
    original=Consensus(p,p.argmax(-1),p.max(-1).values,torch.zeros(n,dtype=torch.bool),
                       torch.zeros(n,dtype=torch.long),torch.ones(n))
    batch=dict(subject_id=e['ids'][:,0],session_id=e['ids'][:,1],trial_id=e['ids'][:,2])
    output=dict(scale_embeddings=e['embeddings'],calibrated_scale_logits=p.log()[:,None].repeat(1,3,1))
    assert learner.refine(output,batch,original,300,False,.6,.15,2) is original
    new=learner.refine(output,batch,original,301,True,.6,.15,2)
    assert new.pseudo_label.tolist()==[1,0,1,2]
    assert not new.valid_mask.any()
    logits=torch.zeros(n,3,requires_grad=True)
    loss=learner.loss(logits,1.)
    loss.backward()
    assert loss>0 and (logits.grad.abs().sum(-1)>0).all()
    assert learner.active_steps==1 and learner.last_record['target_ce_coverage']==1.


def test_exact_requested_plan_and_orphan_stop(tmp_path):
    queues,summary=plan('uncertainty_knn_test',tmp_path)
    assert summary['jobs']==6 and summary['folds']==102 and summary['seeds']==[43]
    assert summary['gpu_directions']=={'0':list('ACE'),'1':list('BDF')}
    assert summary['switch_interval_seconds']==1
    for queue in queues.values():
        for item in queue:
            assert item['seeds']==[43] and 'seed_43' in item['log_dir']
            args=command('python',item,tmp_path)
            assert args[args.index('--seed')+1]=='43'
    processes={100:dict(args=['python','-m','MPUS_GA.trial_temporal.train_uncertainty_pseudo',
        '--result-root',str(tmp_path/'results_uncertainty_knn_test')],ppid=1)}
    assert select_suite(processes,'uncertainty_knn_test')=={100}
