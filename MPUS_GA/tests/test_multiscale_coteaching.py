"""Behavioral tests: exact R2 fallback, peer isolation, rank/stability and gradients."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch, Mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from MPUS_GA.tests.test_training import _small_model
from MPUS_GA.trial_temporal.multiscale_coteaching import (
    CoTeachingConfig, ClassConditionalCoTeaching, PeriodicPeerEvidence,
    peer_probabilities, voting_consensus,
)
from MPUS_GA.trial_temporal.losses import class_conditional_prototype_alignment_loss
from MPUS_GA.trial_temporal.subgroup_alignment import SubgroupConfig, teacher_evidence
from MPUS_GA.trial_temporal.train import (
    EXPERIMENTS, PrototypeBank, build_parser, subgroup_config, coteaching_experiment,
    train_step, validate_args, SourceMultiPrototypeMemory,
)
from MPUS_GA.scripts.run_r2_subgroup_suite import command, GPU_EXPERIMENTS, gpu_experiments
from MPUS_GA.scripts import run_r2_subgroup_suite as suite


def model():
    return _small_model(scales=(1., 2., 4.), num_domains=2, pyramid_gate_mode='sample_class',
        multiview_fusion_mode='class_query_low_rank', use_multiview_uncertainty=True,
        use_sign_aware_pyramid_guard=True, multiview_source_anchor_mix=.5,
        pyramid_gate_shrinkage=.5, pyramid_gate_ceiling=.25,
        use_temporal_msad=True, use_source_prototype_memory=True, multiview_low_rank=8)


def test_peer_target_excludes_own_scale_and_weak_scale_does_not_veto():
    p=torch.tensor([[[.05,.90,.05],[.10,.80,.10],[.4,.2,.4]]])
    old=teacher_evidence(p.log(), SubgroupConfig())[2]
    assert not old.any()
    peers,agree,_=peer_probabilities(p,torch.ones(3,3))
    assert agree[0,2] and peers[0,2,1]>.8
    changed=p.clone(); changed[0,2]=torch.tensor([.99,.005,.005])
    torch.testing.assert_close(peers[:,2],peer_probabilities(changed,torch.ones(3,3))[0][:,2])
    labels,prob,_,valid=voting_consensus(p,torch.ones(3,3))
    assert valid[0] and labels[0]==1 and prob[0,1]>.8


def observations(labels):
    p=F.one_hot(labels,3).float()*.85+.05
    p=p[:,None].repeat(1,3,1)
    features=torch.randn(len(labels),3,32)
    keys=[(0,1,1,i) for i in range(len(labels))]
    return keys,p,features


def test_class_ranking_preserves_minority_and_never_invents_missing_class():
    config=CoTeachingConfig()
    e=PeriodicPeerEvidence(3,3,config)
    keys,p,f=observations(torch.tensor([0]*20+[1]*6))
    # Repeated batch draws do not create new distinct trials or stable refreshes.
    for i in range(1,51):e.observe_target(keys,p,f,i)
    e.refresh(50)
    assert len(e.raw)==26 and not any(r['selected'] for r in e.published.values())
    e.observe_target(keys,p,f,100); e.refresh(100)
    stages=e.history[-1]['stages_by_class']
    assert stages['selected']==[10,3,0]
    assert all(not value.requires_grad for value in (e.reliability,p))
    before=deepcopy(e.published)
    e.observe_target(keys,p.flip(-1),f,101)
    assert not e.refresh(101)
    assert [r['label'] for r in e.published.values()]==[r['label'] for r in before.values()]


def test_temporal_switch_and_stale_trials_revoke_evidence():
    e=PeriodicPeerEvidence(3,3,CoTeachingConfig(keep_fraction=1.))
    keys,p,f=observations(torch.zeros(6).long())
    for i in (50,100):e.observe_target(keys,p,f,i);e.refresh(i)
    assert all(r['selected'] for r in e.published.values())
    p=p.roll(1,-1)
    e.observe_target(keys,p,f,150);e.refresh(150)
    assert not any(r['selected'] for r in e.published.values())
    e.refresh(200)  # Cached predictions without a new observation cannot become stable.
    assert not any(r['selected'] for r in e.published.values())
    e.refresh(400)
    assert not e.raw and not e.published and not e.previous


def test_source_reliability_tracks_class_accuracy_without_target_labels():
    e=PeriodicPeerEvidence(3,3,CoTeachingConfig())
    y=torch.arange(3).repeat(4)
    p=F.one_hot(y,3).float()*.85+.05
    logits=torch.stack((p,p.roll(1,-1),p),1).log()
    e.observe_source(logits,y);e.refresh(50)
    assert (e.reliability[0]>e.reliability[1]).all()
    torch.testing.assert_close(e.reliability[0],torch.ones(3))


def test_centroid_fallback_preserves_value_gradient_and_does_not_renormalize():
    torch.manual_seed(43)
    source=torch.randn(6,3,8,requires_grad=True)
    target=torch.randn(6,3,8,requires_grad=True)
    labels=torch.arange(3).repeat(2)
    prob=F.one_hot(labels,3).float()*.9+.1/3
    weights=torch.ones(1,3,3)/3
    def loss(strength=None):
        return class_conditional_prototype_alignment_loss([source],[labels],target,prob,weights,.6,
                                                         scale_class_strength=strength)[0]
    baseline=loss();original_grad=torch.autograd.grad(baseline,(source,target),retain_graph=True)
    fallback=loss(torch.ones(3,3));new_grad=torch.autograd.grad(fallback,(source,target),retain_graph=True)
    torch.testing.assert_close(baseline,fallback,atol=0,rtol=0)
    for left,right in zip(original_grad,new_grad):torch.testing.assert_close(left,right,atol=0,rtol=0)
    torch.testing.assert_close(loss(torch.full((3,3),.5)),baseline*.5)
    assert loss(torch.zeros(3,3))==0
    masked=torch.ones(3,3);masked[:,0]=0
    residual=loss(masked)
    sg,tg=torch.autograd.grad(residual,(source,target))
    assert sg[labels==0].abs().sum()==0 and tg[labels==0].abs().sum()==0
    assert sg[labels!=0].abs().sum()>0


def batch(labeled,n=3):
    b={'x':{k:torch.randn(n,3,62,5) for k in ('1s','2s','4s')},
       'mask':{k:torch.ones(n,3).bool() for k in ('1s','2s','4s')},
       'subject_id':torch.ones(n).long(),'session_id':torch.ones(n).long(),
       'trial_id':torch.arange(n),'domain_id':torch.full((n,),0 if labeled else 1)}
    if labeled:b['y']=torch.arange(n)%3
    return b


def test_full_training_step_matches_r2_when_no_evidence():
    torch.manual_seed(43)
    original=model();new=deepcopy(original)
    sb,tb=batch(True),batch(False)
    device=torch.device('cpu')
    bank=PrototypeBank(1,3,3,32,.9,.15,.1,device)
    memory=SourceMultiPrototypeMemory(3,3,4,32,.9,device)
    memory.update_source(torch.randn(12,3,32),torch.arange(3).repeat(4))
    alignment=ClassConditionalCoTeaching(new,CoTeachingConfig(),device)
    def step(m,spec,b,memory,controller=None):
        optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        return train_step(m,[sb],tb,optimizer,scheduler,device,600,spec,torch.tensor([[.2,.2,.6]]),
                          b,.1,5.,300,600,.0,source_prototype_memory=memory,subgroup_alignment=controller)
    expected=step(original,EXPERIMENTS['A_R2'],deepcopy(bank),deepcopy(memory))
    actual=step(new,coteaching_experiment('A'),deepcopy(bank),deepcopy(memory),alignment)
    assert expected['prototype']>0
    assert actual['peer_teaching']==0 and actual['subgroup_contrast']==0
    assert abs(actual['total']-expected['total'])<1e-6
    for left,right in zip(original.parameters(),new.parameters()):torch.testing.assert_close(left,right,atol=1e-7,rtol=1e-6)
    assert all(p.grad is None for p in alignment.teacher.parameters())


def test_periodic_peer_and_subgroup_losses_both_backpropagate():
    torch.manual_seed(43)
    m=model();c=ClassConditionalCoTeaching(m,CoTeachingConfig(),torch.device('cpu'))
    sb,tb=batch(True,18),batch(False,18)
    labels=torch.arange(18)%3
    features=F.one_hot(labels,32).float()[:,None].repeat(1,3,1)
    probs=(F.one_hot(labels,3).float()*.85+.05)[:,None].repeat(1,3,1)
    def predict(b):return features,probs.log()
    with patch.object(c,'_predict',side_effect=predict):
        for i in (50,100,300,350,400,450,600):
            student_source=torch.randn(18,3,32,requires_grad=True)
            student_target=torch.randn(18,3,32,requires_grad=True)
            student_logits=torch.randn(18,3,3,requires_grad=True)
            contrast,record=c.loss([{'scale_embeddings':student_source}],[sb],
                                  {'scale_embeddings':student_target,'scale_logits':student_logits},tb,i)
            if i==350:assert contrast==0 # Uses the old published readiness, before this refresh.
            if i==450:
                assert contrast>0 and c.teaching_loss>0 and record['positive_pairs']>0
                (contrast+c.teaching_loss).backward()
                assert student_source.grad.abs().sum()>0 and student_target.grad.abs().sum()>0
                assert student_logits.grad.abs().sum()>0
                assert c.bank.prototypes.grad is None
            c.after_step(m)
        assert (c.readiness==1).all()
        assert (c.centroid_strength(600)==0).all()
        json.dumps(c.state(),allow_nan=False)
        # A new epoch with no reliable evidence restores the original alignment.
        c.bank.support.zero_();c.bank.matches.zero_();c.evidence.published={}
        c._refresh_readiness(650)
        assert (c.centroid_strength(650)==1).all()


def test_no_target_labels_and_zero_contrast_weight_keeps_fallback():
    c=object.__new__(ClassConditionalCoTeaching)
    try:c.loss([],[],{}, {'y':torch.tensor([0])},1)
    except RuntimeError:pass
    else:raise AssertionError('Target labels accepted')
    c=ClassConditionalCoTeaching(model(),CoTeachingConfig(weight=0),torch.device('cpu'))
    c.readiness.fill_(1)
    assert (c.centroid_strength(1000)==1).all()


def test_six_direction_config_and_seed_order_preserve_r2_architecture():
    base=asdict(EXPERIMENTS['A_R2'])
    changed={'name','description','transfer_direction','ablation','source_domains','target_dataset',
             'target_subject_count','target_trials','use_subgroup_alignment','use_multiscale_coteaching'}
    folds=0
    for d in 'ABCDEF':
        spec=coteaching_experiment(d)
        assert all(v==base[k] for k,v in asdict(spec).items() if k not in changed)
        cmd=command('python',d,'/tmp/data','/tmp/r2_coteaching_tests',[43,42],'all',CoTeachingConfig(),'r2_coteaching')
        args=build_parser().parse_args(cmd[4:]);validate_args(args,spec)
        assert args.random_seeds==[43,42] and subgroup_config(args)==CoTeachingConfig()
        folds+=2*len(args.target_subjects)
    assert folds==204 and sorted(sum(GPU_EXPERIMENTS,()))==list('ABCDEF')
    assert gpu_experiments(['0']) == (tuple('ABCDEF'),)
    assert subgroup_config(build_parser().parse_args(['--experiment','A','--method','r2_coteaching'])).confidence==.6
    assert subgroup_config(build_parser().parse_args(['--experiment','A','--method','r2_subgroup'])).confidence==.9


def test_single_gpu_worker_runs_exactly_abcdef_and_stops_after_failure():
    with tempfile.TemporaryDirectory() as directory:
        package=Path(directory)/'MPUS_GA'
        for domain in ('seed_iv','seed_v','seed_vii'):
            p=package/'data_processed'/domain/'window_1s'/'subject_01_session_1.npz'
            p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        for reverse,failure in ((False,None),(False,'C'),(True,None),(True,'C')):
            calls=[]
            gpu='1' if reverse else '0'
            order='FEDCBA' if reverse else 'ABCDEF'
            run=f'r2_coteaching_scheduler_{reverse}_{failure}'
            def launch(cmd,**kwargs):
                direction=cmd[cmd.index('--experiment')+1]
                assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='GPU-test-'+gpu
                assert cmd[cmd.index('--random-seeds')+1:cmd.index('--random-seeds')+3]==['43','42']
                calls.append(direction)
                return 7 if direction==failure else 0
            with patch.object(suite,'__file__',str(package/'scripts'/'suite.py')), \
                 patch.object(suite,'resolve_gpu_uuid',return_value='GPU-test-'+gpu), \
                 patch.object(suite.subprocess,'call',side_effect=launch), \
                 patch.object(sys,'argv',['suite','--worker','--run-name',run,'--gpus',gpu]+(['--reverse'] if reverse else [])):
                try:suite.main(method='r2_coteaching',config_type=CoTeachingConfig,default_gpus=['0'])
                except SystemExit as error:assert failure and error.code==1
                else:assert failure is None
            assert calls==list(order[:order.index(failure)+1] if failure else order)
            manifest=json.loads((package/('results_'+run)/'suite_manifest.json').read_text())
            assert manifest['gpu_mapping']=={gpu:list(order)}


def test_reverse_flag_reaches_detached_worker():
    with tempfile.TemporaryDirectory() as directory:
        package=Path(directory)/'MPUS_GA'
        for domain in ('seed_iv','seed_v','seed_vii'):
            p=package/'data_processed'/domain/'window_1s'/'subject_01_session_1.npz'
            p.parent.mkdir(parents=True,exist_ok=True);p.touch()
        with patch.object(suite,'__file__',str(package/'scripts'/'suite.py')), \
             patch.object(suite,'resolve_gpu_uuid',return_value='GPU-test-one'), \
             patch.object(suite.subprocess,'Popen',return_value=Mock(pid=12345)) as popen, \
             patch.object(sys,'argv',['suite','--run-name','r2_coteaching_reverse_dispatch',
                                      '--gpus','1','--reverse','--random-seeds','43','42']):
            suite.main(method='r2_coteaching',config_type=CoTeachingConfig,
                       module='MPUS_GA.scripts.run_r2_coteaching_suite')
        cmd=popen.call_args.args[0]
        assert '--worker' in cmd and '--reverse' in cmd
        assert cmd[cmd.index('--gpus')+1]=='1'
        assert cmd[cmd.index('--random-seeds')+1:cmd.index('--random-seeds')+3]==['43','42']
        assert popen.call_args.kwargs['start_new_session']


if __name__=='__main__':
    torch.set_num_threads(2)
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:test();print('PASS',test.__name__,flush=True)
    print(len(tests),'tests passed',flush=True)
