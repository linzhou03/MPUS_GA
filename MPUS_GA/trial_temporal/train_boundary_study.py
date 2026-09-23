"""Paired original R2 / ProtoOff / dual-source / MCD experiments, fixed 1000 steps."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from . import train
from .boundary_adaptation import BoundaryConfig, VARIANTS
from .data import prepare_sources
from .oracle_study import configure_determinism
from .pcdiag import cpu, identities, save_json, save_pt
from .train_pcdiag import arguments, fold_paths

NODES = (0, 300, 500, 750, 1000)


def context(runtime, step, evidence=None):
    args, spec = runtime['args'], runtime['spec']
    bank, memory = runtime['prototype_bank'], runtime['source_prototype_memory']
    ramp = train._adaptation_ramp(step, args.adaptation_warmup_iterations, args.adaptation_ramp_end)
    extra = {} if evidence is None else dict(boundary_mean_probability=evidence.mean_probability,
                                            boundary_hard_frequency=evidence.hard_frequency)
    bias = None
    if ramp > 0:
        bias = runtime['target_prior_estimator'].common_bias_adjustments(
            source_excess_strength=args.common_bias_strength * ramp,
            source_relative_tolerance=args.common_bias_relative_tolerance,
            boundary_strength=args.boundary_bias_strength * ramp,
            boundary_ratio_tolerance=args.boundary_bias_ratio_tolerance,
            maximum_adjustment=args.common_bias_max_adjustment, **extra)['combined']
    return dict(compute_domain=False,
        scale_class_reliability=bank.scale_class_reliability() if bank is not None else None,
        multiview_source_anchor=bank.source_relation_weights().mean(0) if bank is not None else None,
        class_logit_adjustment=bias,
        pyramid_gate_ramp=ramp if spec.use_pyramid_gate_warmup else 1.,
        pyramid_bias_risk=(-bias/args.common_bias_max_adjustment).clamp(0,1) if bias is not None else None,
        source_prototype_memory=memory.memory if memory is not None else None,
        source_prototype_initialized=memory.initialized if memory is not None else None)


@torch.inference_mode()
def collect(runtime, step, evidence=None):
    model, args, device = runtime['model'], runtime['args'], runtime['device']
    was_training = model.training
    devices = [device.index or 0] if device.type == 'cuda' else []
    values = {}
    with torch.random.fork_rng(devices=devices):
        model.eval()
        try:
            ctx = context(runtime, step, evidence)
            for batch in runtime['target_evidence_loader']:
                if 'y' in batch:
                    raise RuntimeError('Target evidence contains labels')
                x, mask = train._batch_to_device(batch, device)
                output = model(x, mask, **ctx)
                con = train.independent_scale_consensus(output['calibrated_scale_logits'],
                    args.pseudo_confidence_threshold, args.consensus_jsd_threshold, args.consensus_minimum_votes)
                row = dict(ids=identities(batch), fused=output['probability'],
                           consensus=con.probability, consensus_accepted=con.valid_mask,
                           scale_logits=output['calibrated_scale_logits'])
                heads = getattr(model, 'boundary_heads', None)
                if heads is not None:
                    a,b = heads(output['scale_embeddings'])
                    p1,p2 = a.softmax(-1),b.softmax(-1)
                    q = (p1+p2)/2
                    row.update(head1=p1, head2=p2, generated=q,
                        disagreement=(p1-p2).abs().sum(-1)/2,
                        accepted=(q.max(-1).values >= args.pseudo_confidence_threshold) & (p1.argmax(-1)==p2.argmax(-1)))
                else:
                    row.update(generated=con.probability, accepted=con.valid_mask)
                for k,v in row.items():
                    values.setdefault(k,[]).append(cpu(v))
        finally:
            model.train(was_training)
    return {k:torch.cat(v) for k,v in values.items()}


def model_hash(model, heads=False):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        if key.startswith('boundary_heads.') != heads:
            continue
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Recorder:
    start_iteration = 0

    def __init__(self, result, config):
        self.result, self.config = result, config
        self.snapshots = {}
        self.batch_hash = hashlib.sha256()
        self.steps = 0
        self.extra_active = 0

    def bind(self, runtime):
        self.runtime = {k:runtime[k] for k in ('model','args','spec','device','prepared','seed','subject',
            'target_evidence_loader','prototype_bank','source_prototype_memory','target_prior_estimator')}
        self.initial_hash = model_hash(runtime['model'])
        self.head_hash = model_hash(runtime['model'], heads=True)

    def ready(self):
        self.snapshots[0] = collect(self.runtime, 0)

    def capture(self, iteration, sources, target, output, consensus, active, source_labels, threshold):
        if 'y' in target:
            raise RuntimeError('Target labels in training')
        for batch in [*sources,target]:
            self.batch_hash.update(identities(batch).numpy().tobytes())
        self.steps += 1
        return consensus, {}

    def after_step(self, iteration, record):
        heads = getattr(self.runtime['model'], 'boundary_heads', None)
        if heads is not None:
            self.extra_active += int(heads.last_record['active'])
            record['boundary'] = dict(heads.last_record)
            if iteration == 1 or iteration % 100 == 0 or iteration == 301:
                print('BOUNDARY ' + json.dumps(dict(iteration=iteration, **heads.last_record)), flush=True)
        if iteration in NODES:
            self.snapshots[iteration] = collect(self.runtime, iteration)

    def finish(self, result, final_evidence):
        r = self.runtime
        final = collect(r, 1000, final_evidence)
        save_pt(self.result.with_suffix('.pseudo.pt'), dict(snapshots=self.snapshots, final=final,
            target_truth_used=False, config=asdict(self.config)))
        save_pt(self.result.with_suffix('.pt'), dict(model=cpu(r['model'].state_dict()),
            context=cpu(context(r,1000,final_evidence)), source_stats=cpu(r['prepared'].stats),
            spec=asdict(r['spec']), args={k:str(v) if isinstance(v,Path) else v for k,v in vars(r['args']).items() if not k.startswith('_')},
            config=asdict(self.config), seed=r['seed'], subject=r['subject'], iteration=1000))
        result['boundary_study'] = dict(config=asdict(self.config), backbone_initial_sha256=self.initial_hash,
            heads_initial_sha256=self.head_hash, batch_sequence_sha256=self.batch_hash.hexdigest(),
            observed_steps=self.steps, extra_stages_each=self.extra_active,
            optimizer_steps=self.steps+2*self.extra_active, target_hard_ce=False,
            primary_output='dual_mean' if self.config.variant.startswith('dual_') else 'fused',
            generator='dual_mean' if self.config.variant.startswith('dual_') else 'independent_scale_consensus',
            target_truth_used=False, numerics='deterministic; TF32 disabled',
            caveat='Original R2 domain loss retained in every group; dual_source refers to NEW head objectives only')


def metrics(probability, truth):
    p=probability.argmax(-1)
    cm=torch.bincount(truth*3+p,minlength=9).reshape(3,3).double()
    tp=cm.diag(); rec=tp/cm.sum(1).clamp_min(1)
    f1=2*tp/(cm.sum(0)+cm.sum(1)).clamp_min(1)
    return dict(accuracy=float(tp.sum()/cm.sum()), balanced_accuracy=float(rec.mean()), macro_f1=float(f1.mean()),
        confusion_matrix=cm.int().tolist(), per_class_recall=dict(zip(train.CLASS_NAMES,rec.tolist())))


def audit_prediction(prediction, truth):
    q=prediction['generated']; yhat=q.argmax(-1); confidence=q.max(-1).values
    classes=[]
    for c,name in enumerate(train.CLASS_NAMES):
        predicted=yhat==c; accepted=predicted & prediction['accepted']; correct=accepted & (truth==c)
        n,k,t=int(accepted.sum()),int(correct.sum()),int((truth==c).sum())
        classes.append(dict(class_name=name,predicted=int(predicted.sum()),accepted=n,correct_accepted=k,
            true_count=t,precision=k/n if n else None,correct_coverage=k/t if t else None))
    order=confidence.argsort(descending=True,stable=True)
    coverage=[]
    for fraction in (.25,.5,.75,1.):
        selected=order[:max(1,int(len(truth)*fraction))]
        correct=yhat[selected]==truth[selected]
        coverage.append(dict(requested_fraction=fraction,selected=len(selected),precision=float(correct.float().mean()),
            correct_by_class={name:int(((truth[selected]==c)&correct).sum()) for c,name in enumerate(train.CLASS_NAMES)}))
    return dict(generator=metrics(q,truth),classes=classes,confidence_ranked_equal_coverage=coverage,
                selector_note='diagnostic only; baseline original consensus mask; dual agreement and confidence>=0.6')


def offline_report(result_path,data_dir):
    # Called only after the completed training result is committed.
    from .pcdiag_observe import load_truth
    result=json.loads(result_path.read_text())
    evidence=torch.load(result_path.with_suffix('.pseudo.pt'),map_location='cpu',weights_only=False)
    truth_map=load_truth(data_dir,result['experiment_spec']['target_dataset'],result['target_subject'])
    def truth(pred):
        return torch.tensor([truth_map[tuple(key)][0] for key in pred['ids'].tolist()])
    final=evidence['final']; y=truth(final)
    fused=metrics(final['fused'],y)
    if fused['confusion_matrix'] != result['evaluation']['fused']['confusion_matrix']:
        raise RuntimeError('Final saved fused predictions do not replay formal evaluation')
    primary=final['generated'] if result['boundary_study']['primary_output']=='dual_mean' else final['fused']
    doc=dict(scope='post-training target-truth audit; never used for updates or selection',
        primary_output=result['boundary_study']['primary_output'],primary=metrics(primary,y),fused=fused,
        final=audit_prediction(final,y),nodes={str(k):audit_prediction(v,truth(v)) for k,v in evidence['snapshots'].items()})
    save_json(result_path.with_suffix('.offline.json'),doc)


def complete(path,config):
    if not path.exists():
        return False
    row=json.loads(path.read_text())
    if row.get('boundary_study',{}).get('config') != asdict(config):
        raise RuntimeError('Conflicting existing fold: '+str(path))
    for ext in ('.pt','.pseudo.pt'):
        if not path.with_suffix(ext).exists():
            raise RuntimeError('Missing required artifact: '+str(path.with_suffix(ext)))
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction',choices=list('ABCDEF'),required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--subjects',default='all')
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--result-root',type=Path,required=True)
    cli=p.parse_args()
    configure_determinism(); torch.set_num_threads(4)
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one physical GPU by UUID')
    config=BoundaryConfig(cli.variant)
    args,spec=arguments(cli.direction,cli.data_dir,cli.result_root,cli.subjects,cli.seeds)
    if cli.variant!='r2':
        spec=replace(spec,ablation='boundary_'+cli.variant,description=spec.description+'; '+cli.variant,
            use_prototypes=False,use_source_prototype_memory=False,prototype_weight=0.)
    if cli.variant.startswith('dual_'):
        args._boundary_config=config
    save_json(cli.result_root/cli.direction/f'config_seed_{cli.seeds[0]}.json',dict(config=asdict(config),
        spec=asdict(spec),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')},
        torch=torch.__version__,cuda=torch.version.cuda))
    prepared=None
    for seed in cli.seeds:
        for subject in args.target_subjects:
            result,_=fold_paths(cli.result_root,cli.direction,seed,subject)
            if not complete(result,config):
                if prepared is None:
                    prepared=prepare_sources(args.data_dir,spec.source_domains,spec.scales)
                args._diagnostic=Recorder(result,config)
                print(f'START {cli.variant}/{cli.direction}/seed{seed}/subject{subject:02d}',flush=True)
                train.run_fold(args,spec,prepared,seed,subject,device)
                del args._diagnostic
                gc.collect(); torch.cuda.empty_cache()
            if not result.with_suffix('.offline.json').exists():
                offline_report(result,args.data_dir)
            print('COMPLETE '+str(result),flush=True)
            time.sleep(1.)


if __name__=='__main__':
    main()
