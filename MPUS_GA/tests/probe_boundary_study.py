"""Three-update real-data smoke including final save and offline replay; NOT a study result."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.data import prepare_sources
from MPUS_GA.trial_temporal.boundary_adaptation import BoundaryConfig
from MPUS_GA.trial_temporal.train_boundary_study import Recorder, offline_report
from MPUS_GA.trial_temporal.train_pcdiag import arguments, fold_paths
from MPUS_GA.trial_temporal.oracle_study import configure_determinism


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--variant',required=True)
    cli=p.parse_args()
    if 'probe' not in str(cli.output_root):raise ValueError('Use an isolated probe output directory')
    configure_determinism();torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    args,spec=arguments('A',cli.data_dir,cli.output_root,'1',(42,))
    config=BoundaryConfig(cli.variant)
    if cli.variant!='r2':
        spec=replace(spec,use_prototypes=False,use_source_prototype_memory=False,prototype_weight=0.)
    if cli.variant.startswith('dual_'):args._boundary_config=config
    path,_=fold_paths(cli.output_root,'A',42,1)
    args._diagnostic=Recorder(path,config)
    original=train.train_step
    stages=(1,350,1000)
    def step(*pos,**kw):
        pos=list(pos);pos[6]=stages[pos[6]-1]
        result=original(*pos,**kw)
        if cli.variant.startswith('dual_'):
            print('PROBE '+json.dumps(pos[0].boundary_heads.last_record),flush=True)
        return result
    train.train_step=step
    train.FIXED_UDA_PROTOCOL=replace(train.FIXED_UDA_PROTOCOL,training_iterations=3)
    prepared=prepare_sources(args.data_dir,spec.source_domains,spec.scales)
    train.run_fold(args,spec,prepared,42,1,device)
    offline_report(path,args.data_dir)
    row=json.loads(path.read_text())
    assert row['protocol']['selected_iteration']==3 and row['boundary_study']['observed_steps']==3
    print('PROBE PASS '+cli.variant+' (three updates; final artifacts and offline replay)',flush=True)


if __name__=='__main__':main()
