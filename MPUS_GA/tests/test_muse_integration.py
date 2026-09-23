from copy import deepcopy
from dataclasses import replace
import io
import torch

from MPUS_GA.tests.test_multiscale_coteaching import model, batch
from MPUS_GA.trial_temporal.train import (train_step, coteaching_experiment, muse_experiment,
    PrototypeBank, SourceMultiPrototypeMemory, build_parser, validate_args, muse_args_config)
from MPUS_GA.trial_temporal.multiscale_coteaching import ClassConditionalCoTeaching, CoTeachingConfig
from MPUS_GA.trial_temporal.multiscale_evidence import muse_config
from MPUS_GA.trial_temporal.muse_alignment import MuseController


def with_ids(b):
    key=torch.stack([b[n] for n in ('subject_id','session_id','trial_id')],-1)
    b['trial_key_by_scale']={s:key.clone() for s in b['x']}
    return b


def fixture(variant='starvation', enabled=True):
    student=model()
    cfg=CoTeachingConfig()
    alignment=ClassConditionalCoTeaching(student,cfg,torch.device('cpu'))
    config=muse_config(variant,enabled=enabled,min_hard_support=2.,hard_support_full=8.)
    if config.enabled:alignment.muse=MuseController(student,config,cfg)
    optimizer=torch.optim.AdamW(student.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    bank=PrototypeBank(1,3,3,32,.9,.15,.1,torch.device('cpu'))
    memory=SourceMultiPrototypeMemory(3,3,4,32,.9,torch.device('cpu'))
    return student,alignment,optimizer,scheduler,bank,memory


def step(parts,source,target,iteration,spec=None):
    m,a,o,s,b,memory=parts
    return train_step(m,[source],target,o,s,torch.device('cpu'),iteration,
                      spec or muse_experiment('A'),torch.ones(1,3)/3,b,.1,5.,300,600,.6,
                      source_prototype_memory=memory,subgroup_alignment=a)


def test_disabled_exact_r2_outputs_parameters_and_teacher():
    torch.manual_seed(43);left=fixture(enabled=False)
    torch.manual_seed(43);right=fixture(enabled=False)
    source,target=with_ids(batch(True,6)),with_ids(batch(False,6))
    for iteration in (1,20,299,300,301,350,599,600,650):
        rng=torch.get_rng_state()
        expected=step(left,source,target,iteration,coteaching_experiment('A'))
        torch.set_rng_state(rng)
        actual=step(right,source,target,iteration,muse_experiment('A','r2'))
        assert expected==actual
        for k,v in left[0].state_dict().items():torch.testing.assert_close(v,right[0].state_dict()[k],atol=0,rtol=0)
        for k,v in left[1].teacher.state_dict().items():torch.testing.assert_close(v,right[1].teacher.state_dict()[k],atol=0,rtol=0)


def test_parser_defaults_and_launcher_overrides():
    args=build_parser().parse_args(['--method','muse','--experiment','A','--result-root','/tmp/muse_test'])
    validate_args(args,muse_experiment('A'))
    assert muse_args_config(args).min_hard_support==16
    assert args.adaptation_warmup_iterations==300 and args.adaptation_ramp_end==600
    assert not muse_config('r2').enabled


def test_controller_checkpoint_restores_training_state():
    parts=fixture();source,target=with_ids(batch(True,6)),with_ids(batch(False,6))
    step(parts,source,target,1)
    controller=parts[1].muse
    stream=io.BytesIO();torch.save(controller.state_dict(),stream);stream.seek(0)
    restored=MuseController(parts[0],controller.config,parts[1].config)
    restored.load_state_dict(torch.load(stream,weights_only=True))
    for k,v in controller.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)
