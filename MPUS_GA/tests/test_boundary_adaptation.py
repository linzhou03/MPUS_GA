from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest
import torch
from torch import nn
from MPUS_GA.trial_temporal.boundary_adaptation import BoundaryConfig, attach_heads, extra_steps
from MPUS_GA.scripts.run_boundary_suite import make_plan


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_keys = ['a','b','c']
        self.scale_embedding = nn.Parameter(torch.randn(3,4))
        self.encoder = nn.Linear(4,12)


def encode(model,batch,device):
    return model.encoder(batch['x']).reshape(-1,3,4) + model.scale_embedding


@pytest.mark.parametrize('variant',['dual_source','dual_mcd'])
def test_stages_freeze_parameters_and_preserve_rng(variant):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=Toy(); device=torch.device('cpu')
    state=torch.get_rng_state().clone()
    attach_heads(model,BoundaryConfig(variant),device)
    assert torch.equal(state,torch.get_rng_state())
    source={'x':torch.randn(9,4),'y':torch.arange(9)%3}
    target={'x':torch.randn(7,4)}
    changes=[]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    step=optimizer.step
    def tracked():
        before={k:p.detach().clone() for k,p in model.named_parameters()}
        step()
        changes.append({k for k,p in model.named_parameters() if not torch.equal(before[k],p)})
    optimizer.step=tracked
    state=torch.get_rng_state().clone()
    assert not extra_steps(model,[source],target,optimizer,device,300,5,encode)['active']
    r=extra_steps(model,[source],target,optimizer,device,600,5,encode)
    assert torch.equal(state,torch.get_rng_state())
    assert len(changes)==2 and all(changes)
    assert all(k.startswith('boundary_heads.') for k in changes[0])
    assert all(not k.startswith('boundary_heads.') for k in changes[1])
    assert all(p.requires_grad for p in model.parameters())
    assert r['discrepancy_coefficient']==(1 if variant=='dual_mcd' else 0)
    with pytest.raises(RuntimeError,match='labels'):
        extra_steps(model,[source],dict(target,y=torch.zeros(7)),optimizer,device,600,5,encode)


def test_complete_grid():
    queues,summary=make_plan('boundary_test','/tmp')
    assert summary['jobs']==48 and summary['folds']==816
    for gpu,seed in [('0',42),('1',43)]:
        rows=queues[gpu]
        assert len(rows)==24 and sum(r['folds'] for r in rows)==408
        assert {(r['variant'],r['direction'],r['seeds'][0]) for r in rows} == {
            (v,d,seed) for v in ('r2','proto_off','dual_source','dual_mcd') for d in 'ABCDEF'}
