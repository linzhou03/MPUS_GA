from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))

import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.train_uncertainty_pseudo import settings
from MPUS_GA.trial_temporal.uncertainty_pseudo import UncertaintyPseudo,UncertaintyConfig,collect_evidence
from MPUS_GA.trial_temporal.data import PreparedMultiSource
from MPUS_GA.trial_temporal.pcdiag import rng_state,assert_replay_state
from MPUS_GA.tests.test_msmr import dataset
from MPUS_GA.tests.test_r2_masked import batches


def test_real_encoder_ce_gradient_and_passive_bank(tmp_path):
    torch.set_num_threads(2)
    for direction in 'ABCDEF':
        args,spec=settings(direction,tmp_path,tmp_path,'1',43,'cpu')
        assert not any((spec.use_prototypes,spec.use_source_prototype_memory,spec.use_subgroup_alignment,spec.use_multiscale_coteaching))
        assert spec.domain_weight==.2 and spec.prototype_weight==0
    args.d_model=32;args.dim_feedforward=64;args.dropout=.1
    args.spatial_layers=args.temporal_layers=args.fusion_layers=1
    data=dataset(6);prepared=PreparedMultiSource(spec.source_domains,(data,),data.stats)
    model=train.build_fold_model(args,spec,prepared,torch.device('cpu')).train()
    source,target=batches()
    state=rng_state(torch.device('cpu'));before=deepcopy(model.state_dict())
    evidence=collect_evidence(model,[target],torch.device('cpu'),{'compute_domain':False})
    assert_replay_state(rng_state(torch.device('cpu')),state)
    assert_replay_state(model.state_dict(),before)
    assert model.training
    learner=UncertaintyPseudo(UncertaintyConfig(neighbors=2),[target],torch.device('cpu'))
    learner.bank.update(evidence,301)
    out=model(target['x'],target['mask'])
    original=train.independent_scale_consensus(out['calibrated_scale_logits'],.6,.15,2)
    refined=learner.refine(out,target,original,301,True,.6,.15,2)
    learner.loss(out['logits'],1.).backward()
    assert model.spatial.input_projection.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.classifier.parameters())
    assert not refined.probability.requires_grad
