"""Prototype-free R2 + CBST; fixed-final or explicitly test-selected reporting."""
import argparse
from dataclasses import asdict, replace
from functools import wraps
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from . import train
from .cbst import CBSTConfig
from .data import prepare_sources
from .neighbor_soft import passive_inference
from .oracle_study import configure_determinism
from .pcdiag import cpu, identities, save_json, save_pt
from .train_pcdiag import arguments, fold_paths
from .train_boundary_study import context, metrics, model_hash
from .uncertainty_pseudo import collect_evidence


def cbst_config(variant):
    if variant == 'strict_quota':
        return CBSTConfig()
    if variant == 'independent':
        return CBSTConfig(selection_mode='independent', target_loss_mode='class_mean')
    raise ValueError(f'Unknown CBST variant: {variant}')


def settings(direction,data_dir,result_root,subjects='all',seed=43,device='cuda:0',selection='fixed_final',variant='strict_quota'):
    args,spec=arguments(direction,data_dir,result_root,subjects,(seed,),device)
    spec=replace(spec,ablation=f'cbst_{variant}_{selection}',
        description=f'{direction}: prototype-free R2 with trial CBST; {variant}; {selection}',
        use_prototypes=False,use_source_prototype_memory=False,prototype_weight=0.,use_source_multiview_anchor=False)
    assert not spec.use_subgroup_alignment and not spec.use_multiscale_coteaching
    args._cbst_config=cbst_config(variant)
    args.source_balance_alpha = 1.0
    if selection == 'post300_bal_best':
        args._target_selection_metric = 'balanced_accuracy'
        args._target_selection_min_iteration = args.adaptation_warmup_iterations + 1
    # Both servers evaluate each update so logs expose current/best metrics.
    # Recorder.finish restores fixed-final formal reporting on xju.
    args.evaluation_protocol=train.EVALUATION_PROTOCOL_CAGA_TARGET_BEST
    args.target_eval_interval=1
    return args,spec


def passive_evaluation(fn):
    @wraps(fn)
    def wrapped(model,loader,device,*args,**kwargs):
        with passive_inference(model,device):return fn(model,loader,device,*args,**kwargs)
    return wrapped


def install_passive_evaluation():
    for name in ('evaluate_trials','collect_unlabeled_target_evidence'):
        fn=getattr(train,name)
        if not getattr(fn,'_cbst_passive',False):
            wrapped=passive_evaluation(fn);wrapped._cbst_passive=True;setattr(train,name,wrapped)


class Recorder:
    start_iteration=0

    def __init__(self,path,selection):
        self.path,self.selection=path,selection
        self.steps=0;self.snapshots={};self.batches=[];self.batch_hash=hashlib.sha256()

    def bind(self,runtime):
        self.runtime={k:runtime[k] for k in ('model','args','spec','device','prepared','seed','subject',
            'target_evidence_loader','prototype_bank','source_prototype_memory','target_prior_estimator')}
        assert runtime['prototype_bank'] is None and runtime['source_prototype_memory'] is None
        assert runtime['subgroup_alignment'] is None and runtime['neighbor_learning'] is None
        assert not any(hasattr(runtime['model'],n) for n in ('uncertainty_pseudo','relation_alignment','boundary_heads'))
        self.initial_hash=model_hash(runtime['model'])

    def collect(self,step,evidence=None):
        r=self.runtime
        row=collect_evidence(r['model'],r['target_evidence_loader'],r['device'],context(r,step,evidence))
        learner=r['model'].cbst
        if learner.table is not None:
            selected=learner.lookup(row['ids'],torch.device('cpu'))
            row.update(cbst=selected,thresholds=cpu(learner.table['thresholds']),refresh_iteration=learner.iteration)
        return row

    def ready(self):self.snapshots[0]=self.collect(0)

    def capture(self,iteration,sources,target,output,consensus,active,source_labels,threshold):
        if 'y' in target:raise RuntimeError('Target labels entered training')
        for b in [*sources,target]:self.batch_hash.update(identities(b).numpy().tobytes())
        if active:
            last=self.runtime['model'].cbst.last_batch
            assert torch.equal(cpu(consensus.valid_mask),last['accepted'])
            assert torch.equal(cpu(consensus.pseudo_label),last['pseudo_label'])
            self.batches.append(cpu(last))
        self.steps+=1
        return consensus,{}

    def after_step(self,iteration,record):
        if iteration in (1,301) or iteration%50==0:
            print('CBST '+json.dumps(dict(iteration=iteration,**record['cbst'])),flush=True)
        if iteration in (300,500,750,1000):self.snapshots[iteration]=self.collect(iteration)

    def checkpoint(self,iteration,evidence,evaluation=None):
        r=self.runtime
        return dict(model=cpu(r['model'].state_dict()),context=cpu(context(r,1000,evidence)),
            source_stats=cpu(r['prepared'].stats),spec=asdict(r['spec']),
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(r['args']).items() if not k.startswith('_')},
            cbst=r['model'].cbst.metadata(),cbst_state=r['model'].cbst.state(),
            seed=r['seed'],subject=r['subject'],iteration=iteration,evaluation=evaluation,
            target_test_labels_used_for_selection=self.selection!='fixed_final')

    def selected_checkpoint(self,iteration,evaluation,evidence):
        if self.selection=='fixed_final':return
        save_pt(self.path.with_suffix('.best.pt'),self.checkpoint(iteration,evidence,evaluation))
        save_pt(self.path.with_suffix('.best.pseudo.pt'),dict(iteration=iteration,prediction=self.collect(1000,evidence)))

    def finish(self,result,final_evidence):
        r=self.runtime;learner=r['model'].cbst
        if self.selection=='fixed_final':
            last=result['target_evaluation_trace'][-1]
            result['evaluation']=last['evaluation']
            result['protocol']['selected_iteration']=train.FIXED_UDA_PROTOCOL.training_iterations
            result['protocol']['checkpoint_selection']=train.FIXED_UDA_PROTOCOL.checkpoint_selection
        final=self.collect(1000,final_evidence)
        save_pt(self.path.with_suffix('.pseudo.pt'),dict(final=final,snapshots=self.snapshots,
            training_batches=self.batches,cbst_state=learner.state(),target_truth_used_for_pseudo_labels=False))
        suffix='.pt' if self.selection=='fixed_final' else '.last.pt'
        save_pt(self.path.with_suffix(suffix),self.checkpoint(train.FIXED_UDA_PROTOCOL.training_iterations,final_evidence,
            result['target_evaluation_trace'][-1]['evaluation']))
        if self.selection!='fixed_final':self.path.with_suffix('.best.pt').replace(self.path.with_suffix('.pt'))
        result['method']='r2_proto_off_cbst'
        result['cbst']=dict(**learner.metadata(),observed_steps=self.steps,optimizer_steps=self.steps,
            initial_model_sha256=self.initial_hash,batch_sequence_sha256=self.batch_hash.hexdigest())
        result['reporting']=dict(protocol=self.selection,selection_metric=('fused.balanced_accuracy' if self.selection=='post300_bal_best' else 'fused.accuracy'),tie_break='earliest',
            target_test_labels_used_for_selection=self.selection!='fixed_final',
            primary_output='test_selected_fused' if self.selection!='fixed_final' else 'fused',
            selected_iteration=result['protocol']['selected_iteration'],evaluation_interval=1,
            logged_best_is_diagnostic=self.selection=='fixed_final',
            note=('Target-test BalAcc selected from steps 301..1000' if self.selection=='post300_bal_best'
                  else 'Target-test-selected diagnostic' if self.selection=='test_best' else 'Fixed final step 1000'))


def offline_report(path,data_dir):
    from .pcdiag_observe import load_truth
    row=json.loads(path.read_text())
    saved=torch.load(path.with_suffix('.pseudo.pt'),map_location='cpu',weights_only=False)
    truth_map=load_truth(data_dir,row['experiment_spec']['target_dataset'],row['target_subject'])
    def truth(p):return torch.tensor([truth_map[tuple(v)][0] for v in p['ids'].tolist()])
    final=saved['final'];primary_prediction=final
    if row['reporting']['protocol']!='fixed_final':
        best=torch.load(path.with_suffix('.best.pseudo.pt'),map_location='cpu',weights_only=False)
        assert best['iteration']==row['protocol']['selected_iteration']
        primary_prediction=best['prediction']
    primary=metrics(primary_prediction['fused'],truth(primary_prediction))
    if primary['confusion_matrix']!=row['evaluation']['fused']['confusion_matrix']:
        raise RuntimeError('Saved selected predictions disagree with formal evaluation')
    rounds=[]
    for rnd in saved['cbst_state']['rounds']:
        y=truth(rnd);pred=rnd['pseudo_label'];mask=rnd['accepted'];classes=[]
        for c,name in enumerate(train.CLASS_NAMES):
            selected=mask&(pred==c);correct=selected&(y==c);n=int(selected.sum());t=int((y==c).sum())
            classes.append(dict(class_name=name,threshold=float(rnd['thresholds'][c]),
                predicted=int((pred==c).sum()),accepted=n,correct_accepted=int(correct.sum()),
                precision=float(correct.sum()/n) if n else None,
                correct_coverage=float(correct.sum()/t) if t else None))
        rounds.append(dict(iteration=rnd['iteration'],portion=rnd['portion'],classes=classes))
    save_json(path.with_suffix('.offline.json'),dict(primary=primary,last_step=metrics(final['fused'],truth(final)),
        reporting=row['reporting'],rounds=rounds,scope='Truth joined after training; never used to set CBST thresholds'))


def complete(path,selection,variant='strict_quota'):
    if not path.exists():return False
    row=json.loads(path.read_text());s=row.get('cbst',{});p=row['protocol']
    if s.get('config')!=asdict(cbst_config(variant)) or s.get('observed_steps')!=1000 or s.get('active_steps')!=700:
        raise RuntimeError('Incomplete or conflicting CBST result '+str(path))
    if row['reporting']['protocol']!=selection or row['final_prototype_bank'] is not None or row['final_source_prototype_memory'] is not None:
        raise RuntimeError('Wrong CBST protocol')
    if selection=='fixed_final':
        assert p['selected_iteration']==1000 and p['target_evaluations']==1000 and p['target_eval_interval']==1
    else:
        assert 1<=p['selected_iteration']<=1000 and p['target_evaluations']==1000 and p['target_eval_interval']==1
        if selection=='post300_bal_best':
            assert 301<=p['selected_iteration']<=1000
            assert row['reporting']['selection_metric']=='fused.balanced_accuracy'
    for ext in ('.pt','.pseudo.pt')+ (('.last.pt','.best.pseudo.pt') if selection!='fixed_final' else ()):
        if not path.with_suffix(ext).exists():raise RuntimeError('Missing artifact '+str(path.with_suffix(ext)))
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction',choices=list('ABCDEF'),required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--result-root',type=Path,required=True)
    p.add_argument('--selection',choices=['fixed_final','test_best','post300_bal_best'],required=True)
    p.add_argument('--variant',choices=['strict_quota','independent'],default='strict_quota')
    p.add_argument('--seed',type=int,default=43,choices=[43]);p.add_argument('--subjects',default='all')
    cli=p.parse_args();configure_determinism();torch.set_num_threads(4);install_passive_evaluation()
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose one physical GPU per worker')
    args,spec=settings(cli.direction,cli.data_dir,cli.result_root,cli.subjects,cli.seed,selection=cli.selection,variant=cli.variant)
    save_json(cli.result_root/cli.direction/'frozen_config.json',dict(spec=asdict(spec),config=asdict(cbst_config(cli.variant)),
        selection=cli.selection,variant=cli.variant,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')}))
    prepared=None
    for subject in args.target_subjects:
        path,_=fold_paths(cli.result_root,cli.direction,cli.seed,subject)
        if not complete(path,cli.selection,cli.variant):
            if prepared is None:prepared=prepare_sources(args.data_dir,spec.source_domains,spec.scales)
            args._diagnostic=Recorder(path,cli.selection)
            print(f'START CBST/{cli.variant}/{cli.selection}/{cli.direction}/seed{cli.seed}/subject{subject:02d}',flush=True)
            train.run_fold(args,spec,prepared,cli.seed,subject,device)
            del args._diagnostic;gc.collect();torch.cuda.empty_cache()
        if not path.with_suffix('.offline.json').exists():offline_report(path,args.data_dir)
        print('COMPLETE '+str(path),flush=True);time.sleep(1.)


if __name__=='__main__':main()
