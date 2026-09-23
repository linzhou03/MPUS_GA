"""One prototype-free kNN/entropy-weighted CE main model, seed 43."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from . import train
from .data import prepare_sources
from .oracle_study import configure_determinism
from .pcdiag import cpu, identities, save_json, save_pt
from .train_pcdiag import arguments, fold_paths
from .train_boundary_study import context, metrics, model_hash
from .uncertainty_pseudo import UncertaintyConfig, collect_evidence, entropy_weights


def settings(direction, data_dir, result_root, subjects='all', seed=43, device='cuda:0'):
    args, spec = arguments(direction, data_dir, result_root, subjects, (seed,), device)
    spec = replace(spec, ablation='uncertainty_knn',
        description=spec.description+'; no prototypes; historical kNN pseudo-labels and entropy-weighted target CE',
        use_prototypes=False, use_source_prototype_memory=False, prototype_weight=0.,
        use_source_multiview_anchor=False)
    assert not spec.use_subgroup_alignment and not spec.use_multiscale_coteaching
    args._uncertainty_config = UncertaintyConfig()
    return args, spec


class Recorder:
    start_iteration = 0

    def __init__(self, path):
        self.path, self.snapshots, self.batches = path, {}, []
        self.steps = 0
        self.batch_hash = hashlib.sha256()

    def bind(self, runtime):
        self.runtime = {k:runtime[k] for k in ('model','args','spec','device','prepared','seed','subject',
            'target_evidence_loader','prototype_bank','source_prototype_memory','target_prior_estimator')}
        assert runtime['prototype_bank'] is None and runtime['source_prototype_memory'] is None
        assert runtime['spec'].domain_weight == .2
        assert runtime['subgroup_alignment'] is None and runtime['neighbor_learning'] is None
        assert not hasattr(runtime['model'], 'boundary_heads') and not hasattr(runtime['model'], 'relation_alignment')
        self.initial_hash = model_hash(runtime['model'])

    def collect(self, step, evidence=None):
        r = self.runtime
        row = collect_evidence(r['model'], r['target_evidence_loader'], r['device'], context(r, step, evidence))
        learner = r['model'].uncertainty_pseudo
        if learner.bank.ids is not None:
            q, idx = learner.bank.query(row['embeddings'], row['ids'])
            row.update(refined=q, weights=entropy_weights(q), neighbor_ids=learner.bank.ids[idx],
                       bank_iteration=learner.bank.iteration, history_iterations=list(learner.bank.history_iterations))
        return row

    def ready(self):
        self.snapshots[0] = self.collect(0)

    def capture(self, iteration, sources, target, output, consensus, active, source_labels, threshold):
        if 'y' in target:raise RuntimeError('Target labels entered training')
        for b in [*sources,target]:self.batch_hash.update(identities(b).numpy().tobytes())
        learner=self.runtime['model'].uncertainty_pseudo
        if active:
            assert torch.equal(consensus.probability,learner.last_probability)
            self.batches.append(cpu(learner.last_batch))
        self.steps += 1
        return consensus, {}

    def after_step(self, iteration, record):
        value = dict(self.runtime['model'].uncertainty_pseudo.last_record)
        record['uncertainty_pseudo'] = value
        if iteration in (1,301) or iteration % 100 == 0:
            print('UNCERTAINTY_KNN '+json.dumps(dict(iteration=iteration,**value)),flush=True)
        if iteration in (300,500,750,1000):self.snapshots[iteration] = self.collect(iteration)

    def finish(self, result, final_evidence):
        r=self.runtime
        learner=r['model'].uncertainty_pseudo
        final=self.collect(1000,final_evidence)
        save_pt(self.path.with_suffix('.pseudo.pt'),dict(snapshots=self.snapshots,final=final,
            training_batches=self.batches,bank=cpu(learner.bank.state()),config=asdict(learner.config),target_truth_used=False))
        save_pt(self.path.with_suffix('.pt'),dict(model=cpu(r['model'].state_dict()),
            context=cpu(context(r,1000,final_evidence)),source_stats=cpu(r['prepared'].stats),
            spec=asdict(r['spec']),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(r['args']).items() if not k.startswith('_')},
            uncertainty_pseudo=learner.metadata(),bank=cpu(learner.bank.state()),
            seed=r['seed'],subject=r['subject'],iteration=result['protocol']['selected_iteration']))
        result['uncertainty_pseudo']=dict(**learner.metadata(),observed_steps=self.steps,optimizer_steps=self.steps,
            initial_model_sha256=self.initial_hash,batch_sequence_sha256=self.batch_hash.hexdigest(),
            primary_output='fused',old_domain_loss_weight=r['spec'].domain_weight,
            domain_selector='R2 confidence, scale votes against refined label, original scale JSD; CE ignores mask',
            final_refined_scope='final features queried against last training bank; not primary inference')


def offline_report(path,data_dir):
    # This is deliberately called AFTER the formal result has been committed.
    from .pcdiag_observe import load_truth
    row=json.loads(path.read_text())
    saved=torch.load(path.with_suffix('.pseudo.pt'),map_location='cpu',weights_only=False)
    truth_map=load_truth(data_dir,row['experiment_spec']['target_dataset'],row['target_subject'])
    def truth(p):return torch.tensor([truth_map[tuple(key)][0] for key in p['ids'].tolist()])
    def audit(p):
        y=truth(p)
        out=dict(raw=metrics(p['raw_probability'],y))
        if 'refined' in p:
            q,w=p['refined'],p['weights'];correct=q.argmax(-1)==y
            out.update(refined=metrics(q,y),weighted_precision=float((w*correct).sum()/w.sum()),
                       mean_weight=float(w.mean()),min_weight=float(w.min()),classes=[])
            for c,name in enumerate(train.CLASS_NAMES):
                selected=q.argmax(-1)==c;right=selected&(y==c)
                out['classes'].append(dict(class_name=name,pseudo_count=int(selected.sum()),
                    correct_count=int(right.sum()),true_count=int((y==c).sum()),
                    precision=float(right.sum()/selected.sum()) if selected.any() else None,
                    correct_coverage=float(right.sum()/(y==c).sum()) if (y==c).any() else None,
                    weight_mass=float(w[selected].sum())))
        return out
    final=saved['final'];primary=metrics(final['fused'],truth(final))
    if primary['confusion_matrix']!=row['evaluation']['fused']['confusion_matrix']:
        raise RuntimeError('Saved final predictions differ from formal evaluation')
    save_json(path.with_suffix('.offline.json'),dict(scope='Post-training truth join only',primary_output='fused',
        primary=primary,final=audit(final),nodes={str(k):audit(v) for k,v in saved['snapshots'].items()},
        target_ce_selection='all adaptation trials, positive entropy weights; no hard threshold'))


def complete(path):
    if not path.exists():return False
    row=json.loads(path.read_text());state=row.get('uncertainty_pseudo',{})
    if state.get('config')!=asdict(UncertaintyConfig()) or state.get('observed_steps')!=1000 or state.get('active_steps')!=700:
        raise RuntimeError('Conflicting/incomplete formal result '+str(path))
    if row['protocol']['selected_iteration']!=1000 or row['final_prototype_bank'] is not None or row['final_source_prototype_memory'] is not None:
        raise RuntimeError('Wrong training protocol '+str(path))
    for suffix in ('.pt','.pseudo.pt'):
        if not path.with_suffix(suffix).exists():raise RuntimeError('Missing checkpoint '+str(path))
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction',choices=list('ABCDEF'),required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--result-root',type=Path,required=True)
    p.add_argument('--subjects',default='all');p.add_argument('--seed',type=int,default=43,choices=[43])
    cli=p.parse_args()
    configure_determinism();torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose one physical GPU by UUID')
    args,spec=settings(cli.direction,cli.data_dir,cli.result_root,cli.subjects,cli.seed)
    save_json(cli.result_root/cli.direction/'frozen_config.json',dict(spec=asdict(spec),config=asdict(UncertaintyConfig()),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')}))
    prepared=None
    for subject in args.target_subjects:
        path,_=fold_paths(cli.result_root,cli.direction,cli.seed,subject)
        if not complete(path):
            if prepared is None:prepared=prepare_sources(args.data_dir,spec.source_domains,spec.scales)
            args._diagnostic=Recorder(path)
            print(f'START uncertainty_knn/{cli.direction}/seed{cli.seed}/subject{subject:02d}',flush=True)
            train.run_fold(args,spec,prepared,cli.seed,subject,device)
            del args._diagnostic
            gc.collect();torch.cuda.empty_cache()
        if not path.with_suffix('.offline.json').exists():offline_report(path,args.data_dir)
        print('COMPLETE '+str(path),flush=True)
        time.sleep(1.)


if __name__=='__main__':main()
