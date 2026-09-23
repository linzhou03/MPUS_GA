"""One prototype-free CRCo-inspired main model, fixed final-1000 UDA protocol."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from . import train
from .class_relation_alignment import RelationConfig
from .data import prepare_sources
from .oracle_study import configure_determinism
from .pcdiag import cpu, identities, save_json, save_pt
from .train_pcdiag import arguments, fold_paths
from .train_boundary_study import collect, context, metrics, audit_prediction, model_hash


def settings(direction, data_dir, result_root, subjects='all', seed=42, device='cuda:0'):
    args, spec = arguments(direction, data_dir, result_root, subjects, (seed,), device)
    spec = replace(spec, ablation='crco_feature_alignment',
        description=spec.description+'; prototype-free CRCo-inspired feature alignment',
        use_prototypes=False, use_source_prototype_memory=False, prototype_weight=0.,
        use_source_multiview_anchor=False, domain_weight=0.)
    args._relation_config = RelationConfig()
    return args, spec


class Recorder:
    start_iteration = 0

    def __init__(self, path):
        self.path = path
        self.snapshots = {}
        self.steps = 0
        self.batch_hash = hashlib.sha256()

    def bind(self, runtime):
        self.runtime = {k:runtime[k] for k in ('model','args','spec','device','prepared','seed','subject',
            'target_evidence_loader','prototype_bank','source_prototype_memory','target_prior_estimator')}
        assert runtime['prototype_bank'] is None and runtime['source_prototype_memory'] is None
        assert runtime['spec'].domain_weight == 0
        assert not hasattr(runtime['model'], 'boundary_heads')
        self.initial_hash = model_hash(runtime['model'])

    def ready(self):
        self.snapshots[0] = collect(self.runtime,0)

    def capture(self, iteration, sources, target, output, consensus, active, source_labels, threshold):
        if 'y' in target: raise RuntimeError('Target labels entered training')
        for b in [*sources,target]:self.batch_hash.update(identities(b).numpy().tobytes())
        self.steps += 1
        return consensus, {}

    def after_step(self, iteration, record):
        value = dict(self.runtime['model'].relation_alignment.last_record)
        record['relation_alignment'] = value
        if iteration == 1 or iteration == 301 or iteration % 100 == 0:
            print('RELATION_ALIGNMENT '+json.dumps(dict(iteration=iteration,**value)),flush=True)
        if iteration in (300,500,750,1000):
            self.snapshots[iteration] = collect(self.runtime,iteration)

    def finish(self, result, final_evidence):
        r=self.runtime
        config=asdict(r['args']._relation_config)
        final=collect(r,1000,final_evidence)
        save_pt(self.path.with_suffix('.pseudo.pt'),dict(snapshots=self.snapshots,final=final,config=config,target_truth_used=False))
        state=r['model'].relation_alignment.state()
        save_pt(self.path.with_suffix('.pt'),dict(model=cpu(r['model'].state_dict()),
            context=cpu(context(r,1000,final_evidence)),source_stats=cpu(r['prepared'].stats),
            spec=asdict(r['spec']),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(r['args']).items() if not k.startswith('_')},
            relation_alignment=state,seed=r['seed'],subject=r['subject'],iteration=result['protocol']['selected_iteration']))
        result['relation_alignment']=dict(**state,observed_steps=self.steps,optimizer_steps=self.steps,
            initial_model_sha256=self.initial_hash,batch_sequence_sha256=self.batch_hash.hexdigest(),
            primary_output='fused',target_hard_ce=False,old_domain_loss_weight=0.)


def offline_report(path,data_dir):
    from .pcdiag_observe import load_truth
    row=json.loads(path.read_text())
    saved=torch.load(path.with_suffix('.pseudo.pt'),map_location='cpu',weights_only=False)
    truth_map=load_truth(data_dir,row['experiment_spec']['target_dataset'],row['target_subject'])
    def truth(p):return torch.tensor([truth_map[tuple(key)][0] for key in p['ids'].tolist()])
    final=saved['final'];y=truth(final)
    primary=metrics(final['fused'],y)
    if primary['confusion_matrix']!=row['evaluation']['fused']['confusion_matrix']:
        raise RuntimeError('Saved final predictions differ from formal evaluation')
    save_json(path.with_suffix('.offline.json'),dict(scope='Post-training truth join only',primary_output='fused',
        primary=primary,final=audit_prediction(final,y),
        nodes={str(k):audit_prediction(v,truth(v)) for k,v in saved['snapshots'].items()}))


def complete(path):
    if not path.exists():return False
    row=json.loads(path.read_text())
    state=row.get('relation_alignment',{})
    if state.get('config')!=asdict(RelationConfig()) or state.get('observed_steps')!=1000 or state.get('active_steps')!=700:
        raise RuntimeError('Conflicting/incomplete formal result '+str(path))
    if row['protocol']['selected_iteration']!=1000 or row['final_prototype_bank'] is not None or row['final_source_prototype_memory'] is not None:
        raise RuntimeError('Wrong training protocol '+str(path))
    for suffix in ('.pt','.pseudo.pt'):
        if not path.with_suffix(suffix).exists():raise RuntimeError('Missing checkpoint '+str(path))
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction',choices=list('ABCDEF'),required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--result-root',type=Path,required=True)
    p.add_argument('--subjects',default='all')
    p.add_argument('--seed',type=int,default=42,choices=[42])
    cli=p.parse_args()
    configure_determinism();torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose one physical GPU by UUID')
    args,spec=settings(cli.direction,cli.data_dir,cli.result_root,cli.subjects,cli.seed)
    save_json(cli.result_root/cli.direction/'frozen_config.json',dict(spec=asdict(spec),config=asdict(RelationConfig()),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')}))
    prepared=None
    for subject in args.target_subjects:
        path,_=fold_paths(cli.result_root,cli.direction,cli.seed,subject)
        if not complete(path):
            if prepared is None:prepared=prepare_sources(args.data_dir,spec.source_domains,spec.scales)
            args._diagnostic=Recorder(path)
            print(f'START crco_feature/{cli.direction}/seed{cli.seed}/subject{subject:02d}',flush=True)
            train.run_fold(args,spec,prepared,cli.seed,subject,device)
            del args._diagnostic
            gc.collect();torch.cuda.empty_cache()
        if not path.with_suffix('.offline.json').exists():offline_report(path,args.data_dir)
        print('COMPLETE '+str(path),flush=True)
        time.sleep(1.)


if __name__=='__main__':main()
