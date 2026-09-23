from dataclasses import replace
from pathlib import Path
import torch
import io
from MPUS_GA.trial_temporal.multiscale_evidence import MuseConfig
from MPUS_GA.trial_temporal.rejectable_ot import rejectable_transport
from MPUS_GA.tests.test_muse_subgroups import subgroup_fixture
from MPUS_GA.scripts.run_muse_suite import training_command, DEFAULT_VARIANTS
from MPUS_GA.scripts.run_r3_suite import build_plan, split_plan
from MPUS_GA.trial_temporal.multiscale_coteaching import CoTeachingConfig
from MPUS_GA.trial_temporal.train import build_parser,validate_args,muse_experiment,muse_args_config
from MPUS_GA.trial_temporal.muse_alignment import MuseController


def test_near_subgroups_match_and_far_subgroup_goes_to_null():
    cost=torch.tensor([[.01,1.7],[1.6,1.8]])
    r=rejectable_transport(cost,MuseConfig())
    assert r['accepted'].tolist()==[[True,False],[False,False]]
    assert r['plan'][1,-1]>0 and r['plan'][-1,1]>0
    assert not r['real_mass'][1].any() and not r['real_mass'][:,1].any()
    assert torch.isfinite(r['plan']).all()


def test_null_cost_configuration_and_empty_rectangular_inputs():
    for shape in ((0,0),(0,3),(2,0),(2,4)):
        result=rejectable_transport(torch.ones(shape),MuseConfig())
        assert result['plan'].shape==(shape[0]+1,shape[1]+1)
        assert torch.isfinite(result['plan']).all() and not result['accepted'].any()
    a=rejectable_transport(torch.tensor([[.2]]),MuseConfig(ot_null_cost=.01))
    b=rejectable_transport(torch.tensor([[.2]]),MuseConfig(ot_null_cost=.8))
    assert a['plan'][0,-1]>b['plan'][0,-1]


def test_ot_reuses_bank_and_gradients_and_checkpoint():
    parts,source,target,sb,tb=subgroup_fixture();a=parts[1];muse=a.muse
    original=a.bank.prototypes.clone()
    muse.config=replace(muse.config,ot_enabled=True)
    muse.after_bank_refresh(a.bank)
    torch.testing.assert_close(a.bank.prototypes,original,atol=0,rtol=0)
    loss,record=muse.subgroup_loss(a,[{'scale_embeddings':source}],[sb],{'scale_embeddings':target},tb)
    assert loss>0 and a.bank.muse_transport.sum()>0
    loss.backward();assert source.grad.abs().sum()>0 and target.grad.abs().sum()>0
    assert muse.get_extra_state()['ot_plans']
    before=a.bank.muse_transport.clone()
    stream=io.BytesIO();torch.save(muse.state_dict(),stream);stream.seek(0)
    restored=MuseController(parts[0],muse.config,a.config)
    a.bank.muse_transport.zero_()
    restored.load_checkpoint(torch.load(stream,weights_only=True),a.bank)
    torch.testing.assert_close(before,a.bank.muse_transport,atol=0,rtol=0)
    torch.testing.assert_close(muse.class_gate(),restored.class_gate(),atol=0,rtol=0)
    a.bank.matches.zero_();a.bank.muse_transport.zero_()
    loss,_=muse.subgroup_loss(a,[{'scale_embeddings':source}],[sb],{'scale_embeddings':target},tb)
    assert loss==0


def test_launch_plan_and_all_ablation_commands():
    plan=build_plan('muse_test',Path('/tmp'),DEFAULT_VARIANTS,[43,42],43)
    queues=split_plan(plan,['0','1'])
    assert sum(i['folds'] for i in plan)==714
    assert sum(len(i['seeds']) for i in plan)==42
    assert {i['direction'] for i in queues['0']}==set('ACE')
    assert {i['direction'] for i in queues['1']}==set('BDF')
    for item in plan:
        args=build_parser().parse_args(training_command('python',item,Path('/tmp'),'all',CoTeachingConfig())[4:])
        validate_args(args,muse_experiment(item['direction'],item['variant']))
        assert muse_args_config(args).min_hard_support==2 and muse_args_config(args).hard_support_full==8
