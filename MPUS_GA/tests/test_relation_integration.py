from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))

import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.train_relation_alignment import settings
from MPUS_GA.trial_temporal.class_relation_alignment import RelationAlignment,RelationConfig
from MPUS_GA.trial_temporal.data import PreparedMultiSource
from MPUS_GA.tests.test_msmr import dataset
from MPUS_GA.tests.test_r2_masked import batches


def test_profile_and_auxiliary_gradient_reaches_real_encoder_without_classifier(tmp_path):
    torch.set_num_threads(2)
    for d in 'ABCDEF':
        args,spec=settings(d,tmp_path,tmp_path,'1',42,'cpu')
        assert not spec.use_prototypes and not spec.use_source_prototype_memory
        assert not spec.use_subgroup_alignment and not spec.use_multiscale_coteaching
        assert spec.domain_weight==spec.prototype_weight==0
    args.d_model=32;args.dim_feedforward=64;args.dropout=0.
    args.spatial_layers=args.temporal_layers=args.fusion_layers=1
    source_data=dataset(6)
    prepared=PreparedMultiSource(spec.source_domains,(source_data,),source_data.stats)
    model=train.build_fold_model(args,spec,prepared,torch.device('cpu'))
    source,target=batches()
    source_out=model(source['x'],source['mask'])
    target_out=model(target['x'],target['mask'])
    learner=RelationAlignment(RelationConfig(),torch.device('cpu'))
    warm=learner.loss(model,[source_out],[source],target_out,target,300,torch.device('cpu'))
    assert warm.item()==0
    loss=learner.loss(model,[source_out],[source],target_out,target,350,torch.device('cpu'))
    assert loss.item()>0
    loss.backward()
    assert model.spatial.input_projection.weight.grad.abs().sum()>0
    assert model.temporal['1s'].transformer.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.classifier.parameters())
    assert learner.last_record['alignment']>0 and learner.last_record['instance']>0
