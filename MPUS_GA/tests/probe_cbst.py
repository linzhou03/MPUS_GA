"""Real-data three-update smoke test, isolated from formal results."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.train_cbst import settings,Recorder,offline_report,install_passive_evaluation
from MPUS_GA.trial_temporal.data import prepare_sources
from MPUS_GA.trial_temporal.oracle_study import configure_determinism


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--selection',choices=['fixed_final','test_best','post300_bal_best'],required=True)
    p.add_argument('--variant',choices=['strict_quota','independent'],default='strict_quota')
    a=p.parse_args()
    assert 'probe' in str(a.output_root)
    configure_determinism();torch.set_num_threads(4);install_passive_evaluation()
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    args,spec=settings('B',a.data_dir,a.output_root,'1',43,selection=a.selection,variant=a.variant)
    if a.selection=='post300_bal_best':args._target_selection_min_iteration=2
    path=a.output_root/'B'/'seed_43_subject_01.json';args._diagnostic=Recorder(path,a.selection)
    original=train.train_step;records=[]
    def step(*pos,**kw):
        pos=list(pos);pos[6]=(300,350,1000)[pos[6]-1]
        row=original(*pos,**kw);records.append(row['cbst'])
        print('PROBE',pos[6],row['cbst'],flush=True)
        return row
    train.train_step=step
    train.FIXED_UDA_PROTOCOL=replace(train.FIXED_UDA_PROTOCOL,training_iterations=3)
    prepared=prepare_sources(a.data_dir,spec.source_domains,spec.scales)
    train.run_fold(args,spec,prepared,43,1,device)
    offline_report(path,a.data_dir)
    row=json.loads(path.read_text());trace=row['target_evaluation_trace']
    expected=(max(trace[1:],key=lambda x:x['evaluation']['fused']['balanced_accuracy'])
              if a.selection=='post300_bal_best' else
              max(trace,key=lambda x:x['evaluation']['fused']['accuracy'])
              if a.selection=='test_best' else trace[-1])
    assert len(trace)==3
    assert row['protocol']['selected_iteration']==expected['iteration']
    best=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
    assert best['iteration']==expected['iteration']
    assert row['cbst']['active_steps']==2 and row['final_prototype_bank'] is None and row['final_source_prototype_memory'] is None
    assert records[0]['added_loss']==0
    assert all((r['target_ce']>0)==(sum(r['batch_accepted_class_count'])>0) for r in records[1:])
    assert records[-1]['portion']==.25
    print('PROBE PASS',a.variant,a.selection,'actual selected checkpoint',best['iteration'],'peak MiB',torch.cuda.max_memory_allocated()/1024**2,flush=True)


if __name__=='__main__':main()
