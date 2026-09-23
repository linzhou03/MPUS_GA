from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))
import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.cbst import CBSTController,CBSTConfig
from MPUS_GA.trial_temporal.train_cbst import settings,passive_evaluation
from MPUS_GA.trial_temporal.data import PreparedMultiSource
from MPUS_GA.tests.test_msmr import dataset
from MPUS_GA.tests.test_r2_masked import batches


def test_real_train_step_has_selected_ce_without_prototypes(tmp_path):
    torch.set_num_threads(2)
    args,spec=settings('A',tmp_path,tmp_path,'1',43,'cpu')
    for mode in ('fixed_final','test_best'):
        a,s=settings('A',tmp_path,tmp_path,'1',43,'cpu',mode)
        assert not s.use_prototypes and not s.use_source_prototype_memory and not s.use_subgroup_alignment
        assert not hasattr(a,'_uncertainty_config')
    args.d_model=32;args.dim_feedforward=64;args.dropout=0.
    args.spatial_layers=args.temporal_layers=args.fusion_layers=1
    data=dataset(6);prepared=PreparedMultiSource(spec.source_domains,(data,),data.stats)
    model=train.build_fold_model(args,spec,prepared,torch.device('cpu'))
    source,target=batches();model.cbst=CBSTController(CBSTConfig(),[target],torch.device('cpu'))
    opt=torch.optim.AdamW(model.parameters(),lr=.0005);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    before=model.spatial.input_projection.weight.detach().clone()
    records=[]
    for step in (300,350):
        records.append(train.train_step(model,[source],target,opt,sch,torch.device('cpu'),step,spec,
            torch.ones(1,3)/3,None,.1,5.,300,600,.6))
    assert records[0]['cbst']['added_loss']==0
    assert records[1]['cbst']['target_ce']>0 and records[1]['prototype']==0
    assert model.cbst.active_steps==1 and model.cbst.last_batch['accepted'].any()
    assert not torch.equal(before,model.spatial.input_projection.weight.detach())


def test_passive_evaluation_restores_rng_and_model_mode():
    model=torch.nn.Linear(2,2).train()
    def evaluator(model,loader,device):
        model.eval();return torch.rand(3)
    before=torch.get_rng_state().clone()
    result=passive_evaluation(evaluator)(model,None,torch.device('cpu'))
    assert torch.equal(before,torch.get_rng_state()) and model.training and len(result)==3
