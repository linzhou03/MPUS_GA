"""Three real-data updates plus final artifact/replay check, NOT a formal run."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.data import prepare_sources
from MPUS_GA.trial_temporal.oracle_study import configure_determinism
from MPUS_GA.trial_temporal.train_uncertainty_pseudo import settings,Recorder,offline_report
from MPUS_GA.trial_temporal.train_pcdiag import fold_paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--direction',choices=list('ABCDEF'),default='B')
    cli=p.parse_args()
    if 'probe' not in str(cli.output_root):raise ValueError('Use an isolated probe output directory')
    configure_determinism();torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    args,spec=settings(cli.direction,cli.data_dir,cli.output_root,'1')
    path,_=fold_paths(cli.output_root,cli.direction,43,1)
    args._diagnostic=Recorder(path)
    original=train.train_step;records=[]
    def step(*pos,**kw):
        pos=list(pos);pos[6]=(300,350,1000)[pos[6]-1]
        record=original(*pos,**kw)
        value=dict(record['uncertainty_pseudo'])
        value['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        records.append(value)
        print('PROBE '+json.dumps(value),flush=True)
        return record
    train.train_step=step
    train.FIXED_UDA_PROTOCOL=replace(train.FIXED_UDA_PROTOCOL,training_iterations=3)
    prepared=prepare_sources(args.data_dir,spec.source_domains,spec.scales)
    train.run_fold(args,spec,prepared,43,1,device)
    offline_report(path,args.data_dir)
    row=json.loads(path.read_text())
    assert row['uncertainty_pseudo']['observed_steps']==3 and row['uncertainty_pseudo']['active_steps']==2
    assert row['final_prototype_bank'] is None and row['final_source_prototype_memory'] is None
    assert records[0]['added_loss']==0 and all(r['target_ce']>0 and r['added_loss']>0 for r in records[1:])
    assert records[-1]['history_count']==2 and records[-1]['target_ce_coverage']==1.
    print('PROBE PASS '+cli.direction+'; three updates only, no formal score',flush=True)


if __name__=='__main__':main()
