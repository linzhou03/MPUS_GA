from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))
import torch
from MPUS_GA.trial_temporal.cbst import CBSTConfig,CBSTController,select,selected_ce
from MPUS_GA.scripts.run_cbst_suite import plan,command
from MPUS_GA.scripts.run_cbst_balanced_suite import plan as balanced_plan, command as balanced_command
from MPUS_GA.scripts.run_cbst_positive_gate_suite import plan as gate_plan, command as gate_command


def test_class_threshold_and_normalized_reassignment():
    p=torch.tensor([[.9,.05,.05],[.8,.1,.1],[.51,.48,.01],[.39,.6,.01],[.3,.5,.2],[.2,.2,.6]])
    out=select(p,.5)
    torch.testing.assert_close(out['thresholds'],torch.tensor([.9,.6,.6]))
    assert out['raw_class_support'].tolist()==[3,2,1]
    assert out['class_rank'].tolist()==[1,1,1]
    assert out['accepted'].tolist()==[True,False,False,True,False,True]
    assert out['accepted_class_count'].tolist()==[1,1,1]


def test_small_class_and_missing_class_do_not_use_global_threshold():
    p=torch.tensor([[.36,.34,.30],[.33,.35,.32],[.37,.33,.30]])
    out=select(p,.2)
    torch.testing.assert_close(out['thresholds'],torch.tensor([1.,1.,1.]))
    assert out['accepted'].tolist()==[False,False,False]
    assert out['class_rank'].tolist()==[0,0,0]
    assert not out['accepted'].any()


def test_selected_ce_ignores_rejected_and_empty_has_zero_gradient():
    logits=torch.zeros(3,3,requires_grad=True);label=torch.tensor([0,1,2]);mask=torch.tensor([True,False,True])
    loss=selected_ce(logits,label,mask);loss.backward()
    assert logits.grad[1].abs().sum()==0 and logits.grad[0].abs().sum()>0
    logits.grad=None
    zero=selected_ce(logits,label,torch.zeros(3,dtype=torch.bool));zero.backward()
    assert zero.item()==0 and logits.grad.abs().sum()==0


def test_independent_acceptance_keeps_present_classes_when_one_is_missing():
    p=torch.tensor([[.90,.05,.05],[.80,.10,.10],[.70,.20,.10],
                    [.10,.85,.05],[.15,.75,.10],[.20,.70,.10]])
    strict=select(p,.5)
    independent=select(p,.5,'independent')
    assert strict['accepted_class_count'].tolist()==[0,0,0]
    assert independent['accepted_class_count'].tolist()==[1,1,0]
    assert independent['accepted'].tolist()==[True,False,False,True,False,False]


def test_positive_gate_filters_before_shared_quota_without_changing_target_loss():
    p=torch.tensor([[.65,.20,.15],[.90,.05,.05],[.42,.17,.41],
                    [.10,.80,.10],[.10,.70,.20],[.10,.10,.80],[.20,.10,.70]])
    scales=torch.tensor([
        [[3.,0.,0.],[3.,0.,0.],[3.,0.,0.]],
        [[3.,0.,0.],[3.,0.,0.],[0.,0.,3.]],
        [[3.,0.,0.],[3.,0.,0.],[3.,0.,0.]],
        [[0.,3.,0.]]*3,[[0.,3.,0.]]*3,
        [[0.,0.,3.]]*3,[[0.,0.,3.]]*3])
    out=select(p,.5,'positive_gate',scales,3,.10)
    assert out['raw_class_support'].tolist()==[3,2,2]
    assert out['eligible_class_support'].tolist()==[1,2,2]
    assert out['accepted'].tolist()==[True,False,False,True,False,True,False]
    assert out['accepted_class_count'].tolist()==[1,1,1]
    assert select(p,.5)['accepted'][1]
    assert CBSTConfig(selection_mode='positive_gate',positive_min_votes=3,
                      positive_negative_margin=.10).target_loss_mode=='sample_mean'


def test_positive_gate_empty_class_vetoes_all_target_ce():
    p=torch.tensor([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8]])
    scales=torch.tensor([[[3.,0.,0.],[3.,0.,0.],[0.,0.,3.]],
                         [[0.,3.,0.]]*3,[[0.,0.,3.]]*3])
    out=select(p,.5,'positive_gate',scales,3,.10)
    assert out['eligible_class_support'].tolist()==[0,1,1]
    assert out['accepted_class_count'].tolist()==[0,0,0]


def test_class_mean_ce_gives_each_present_class_equal_weight():
    logits=torch.tensor([[3.,0.,0.],[2.,0.,0.],[0.,3.,0.]],requires_grad=True)
    label=torch.tensor([0,0,1]);mask=torch.tensor([True,True,True])
    individual=torch.nn.functional.cross_entropy(logits,label,reduction='none')
    expected=(individual[:2].mean()+individual[2])/2
    actual=selected_ce(logits,label,mask,'class_mean')
    torch.testing.assert_close(actual,expected)
    assert not torch.isclose(actual,selected_ce(logits,label,mask))
    actual.backward()
    assert logits.grad.abs().sum()>0


def test_frozen_round_lookup_and_growth():
    c=CBSTController(CBSTConfig(),[],torch.device('cpu'))
    ids=torch.tensor([[1,1,0],[1,1,1]])
    p=torch.tensor([[.8,.1,.1],[.2,.6,.2]])
    c.update(dict(ids=ids,raw_probability=p),301)
    queried=c.lookup(ids[[1,1,0]],torch.device('cpu'))
    assert queried['pseudo_label'].tolist()==[1,1,0]
    assert c.config.portion(0)==.2 and c.config.portion(6)==.5 and c.config.portion(13)==.5
    c.update(dict(ids=ids,raw_probability=p),351)
    assert c.rounds[-1]['portion']==.25
    assert c.metadata()['knn'] is False and c.metadata()['teacher'] is False


def test_two_protocols_same_six_directions_one_seed(tmp_path):
    for host,selection in [('xju','fixed_final'),('csu','test_best')]:
        queues,summary=plan(host,'cbst_test',tmp_path)
        assert summary['selection']==selection and summary['folds']==102 and summary['jobs']==6
        assert summary['evaluation_interval']==1
        assert summary['gpu_directions']=={'0':list('ACE'),'1':list('BDF')}
        for queue in queues.values():
            for item in queue:
                args=command('python',item,tmp_path)
                assert item['seeds']==[43] and args[args.index('--selection')+1]==selection


def test_balanced_suite_is_paired_across_six_directions(tmp_path):
    queues,summary=balanced_plan('r3_balanced_test',tmp_path)
    assert summary['folds']==204 and summary['jobs']==12
    assert summary['seeds']==[43] and summary['selection']=='post300_bal_best'
    for queue in queues.values():
        for item in queue:
            args=balanced_command('python',item,tmp_path)
            assert args[args.index('--variant')+1]==item['variant']
            assert args[args.index('--selection')+1]=='post300_bal_best'


def test_positive_gate_suite_has_three_concurrent_workers(tmp_path):
    queues,summary=gate_plan('r3_posgate_test',tmp_path)
    assert summary['jobs']==3 and summary['folds']==56
    assert summary['phases']==[
        {'physical_gpu':'0','concurrent_directions':['B','E']},
        {'physical_gpu':'0','concurrent_directions':['C'],
         'starts_after':'B and E both complete'}]
    assert set(queues)=={'gpu0_B','gpu0_E','gpu0_C'}
    assert all(items[0]['physical_gpu']=='0' for items in queues.values())
    for items in queues.values():
        item=items[0]
        args=gate_command('python',item,tmp_path)
        assert args[args.index('--variant')+1]=='positive_gate'
        assert args[args.index('--selection')+1]=='post300_bal_best'
