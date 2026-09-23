from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[2]))
import torch
from MPUS_GA.trial_temporal.cbst import CBSTConfig,CBSTController,select,selected_ce
from MPUS_GA.scripts.run_cbst_suite import plan,command


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
