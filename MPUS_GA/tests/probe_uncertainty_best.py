"""Three-update real-data check of per-update test selection and artifacts."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from MPUS_GA.trial_temporal import train
from MPUS_GA.trial_temporal.train_uncertainty_best import install,settings,Recorder,offline_report
from MPUS_GA.trial_temporal.data import prepare_sources
from MPUS_GA.trial_temporal.oracle_study import configure_determinism


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    assert 'probe' in str(a.output_root)
    configure_determinism();torch.set_num_threads(4);install()
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    args,spec=settings('B',a.data_dir,a.output_root,'1',43)
    assert not args._uncertainty_config.activation_checkpointing and args._uncertainty_config.spatial_chunk_windows==0
    path=a.output_root/'B'/'seed_43_subject_01.json';args._diagnostic=Recorder(path)
    original=train.train_step
    def step(*pos,**kw):
        pos=list(pos);pos[6]=(300,350,1000)[pos[6]-1]
        row=original(*pos,**kw)
        print('PROBE',pos[6],row['uncertainty_pseudo'],flush=True)
        return row
    train.train_step=step
    train.FIXED_UDA_PROTOCOL=replace(train.FIXED_UDA_PROTOCOL,training_iterations=3)
    prepared=prepare_sources(a.data_dir,spec.source_domains,spec.scales)
    train.run_fold(args,spec,prepared,43,1,device)
    offline_report(path,a.data_dir)
    row=json.loads(path.read_text());trace=row['target_evaluation_trace']
    expected=max(trace,key=lambda x:x['evaluation']['fused']['accuracy'])
    assert len(trace)==3 and row['protocol']['selected_iteration']==expected['iteration']
    best=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
    last=torch.load(path.with_suffix('.last.pt'),map_location='cpu',weights_only=False)
    assert best['iteration']==expected['iteration'] and last['iteration']==3
    if best['iteration']!=3:
        assert any(not torch.equal(v,last['model'][k]) for k,v in best['model'].items())
    assert row['uncertainty_pseudo']['active_steps']==2
    print('PROBE PASS: 3 evaluations; selected',best['iteration'],'; separate best/last checkpoints; prediction replay matched; peak MiB',torch.cuda.max_memory_allocated()/1024**2,flush=True)


if __name__=='__main__':main()
