"""Explicit target-test-selected reporting; every one of 1000 updates is eligible."""
from dataclasses import asdict, replace
from functools import wraps
import json

import torch
from . import train
from . import train_uncertainty_pseudo as base
from .neighbor_soft import passive_inference
from .pcdiag import cpu, save_json, save_pt
from .train_boundary_study import context, metrics
from .uncertainty_pseudo import UncertaintyConfig


def best_config():
    return replace(UncertaintyConfig(), activation_checkpointing=False, spatial_chunk_windows=0)


def settings(*args, **kwargs):
    options,spec=base_settings(*args,**kwargs)
    options.evaluation_protocol=train.EVALUATION_PROTOCOL_CAGA_TARGET_BEST
    options.target_eval_interval=1
    options._uncertainty_config=best_config()
    return options,replace(spec,ablation='uncertainty_knn_test_best',
        description=spec.description+'; target TEST accuracy selects checkpoint from every update')


def passive_evaluation(fn):
    @wraps(fn)
    def wrapped(model,loader,device,*args,**kwargs):
        # Iterating an eval DataLoader consumes RNG even with shuffle=False.
        # Preserve training RNG/modes so evaluations do not change optimization.
        with passive_inference(model,device):
            return fn(model,loader,device,*args,**kwargs)
    return wrapped


class Recorder(base.Recorder):
    def selected_checkpoint(self, iteration, evaluation, evidence):
        r=self.runtime; learner=r['model'].uncertainty_pseudo
        packet=dict(model=cpu(r['model'].state_dict()), context=cpu(context(r,1000,evidence)),
            source_stats=cpu(r['prepared'].stats),spec=asdict(r['spec']),
            uncertainty_pseudo=learner.metadata(),bank=cpu(learner.bank.state()),
            seed=r['seed'],subject=r['subject'],iteration=iteration,evaluation=evaluation,
            target_test_labels_used_for_selection=True,selection_metric='fused.accuracy',tie_break='earliest')
        save_pt(self.path.with_suffix('.best.pt'),packet)
        # Inference uses the same full inference context as evaluate_trials,
        # including when the selected training update is inside source warmup.
        prediction=self.collect(1000,evidence)
        save_pt(self.path.with_suffix('.best.pseudo.pt'),dict(iteration=iteration,prediction=prediction))

    def finish(self,result,final_evidence):
        # Keep the final-step model/diagnostics too; never label it as best.
        selected=result['protocol']['selected_iteration']
        result['protocol']['selected_iteration']=train.FIXED_UDA_PROTOCOL.training_iterations
        super().finish(result,final_evidence)
        result['protocol']['selected_iteration']=selected
        self.path.with_suffix('.pt').replace(self.path.with_suffix('.last.pt'))
        self.path.with_suffix('.best.pt').replace(self.path.with_suffix('.pt'))
        result['reporting']=dict(protocol='target_test_selected',target_test_labels_used_for_selection=True,
            selection_metric='fused.accuracy',tie_break='earliest',evaluation_interval=1,
            candidates=result['protocol']['target_evaluations'],selected_iteration=selected,
            note='Test-selected diagnostic result, not a held-out fixed-final UDA estimate')
        result['uncertainty_pseudo']['primary_output']='test_selected_fused'


def offline_report(path,data_dir):
    from .pcdiag_observe import load_truth
    row=json.loads(path.read_text())
    saved=torch.load(path.with_suffix('.best.pseudo.pt'),map_location='cpu',weights_only=False)
    final=torch.load(path.with_suffix('.pseudo.pt'),map_location='cpu',weights_only=False)['final']
    truth_map=load_truth(data_dir,row['experiment_spec']['target_dataset'],row['target_subject'])
    def measure(p):
        y=torch.tensor([truth_map[tuple(key)][0] for key in p['ids'].tolist()])
        return metrics(p['fused'],y)
    primary=measure(saved['prediction'])
    assert saved['iteration']==row['protocol']['selected_iteration']
    if primary['confusion_matrix']!=row['evaluation']['fused']['confusion_matrix']:
        raise RuntimeError('Selected checkpoint predictions differ from reported best evaluation')
    save_json(path.with_suffix('.offline.json'),dict(primary_output='test_selected_fused',
        primary=primary,last_step=measure(final),reporting=row['reporting']))


def complete(path):
    if not path.exists():return False
    row=json.loads(path.read_text());state=row.get('uncertainty_pseudo',{});p=row['protocol']
    if state.get('config')!=asdict(best_config()) or state.get('observed_steps')!=1000 or state.get('active_steps')!=700:
        raise RuntimeError('Wrong/incomplete test-selected result '+str(path))
    if not 1<=p['selected_iteration']<=1000 or p['target_evaluations']!=1000 or p['target_eval_interval']!=1:
        raise RuntimeError('Expected all 1000 checkpoints to be evaluated')
    if row['final_prototype_bank'] is not None or row['final_source_prototype_memory'] is not None:
        raise RuntimeError('Prototypes must be disabled')
    for suffix in ('.pt','.last.pt','.pseudo.pt','.best.pseudo.pt'):
        if not path.with_suffix(suffix).exists():raise RuntimeError('Missing artifact '+str(path))
    return True


base_settings=base.settings


def install():
    base.settings=settings
    base.Recorder=Recorder
    base.offline_report=offline_report
    base.complete=complete
    base.UncertaintyConfig=best_config
    train.evaluate_trials=passive_evaluation(train.evaluate_trials)
    train.collect_unlabeled_target_evidence=passive_evaluation(train.collect_unlabeled_target_evidence)


def main():
    install()
    base.main()


if __name__=='__main__':main()
