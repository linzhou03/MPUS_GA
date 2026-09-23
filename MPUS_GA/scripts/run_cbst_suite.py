"""Six directions, one main model, seed 43; two independent GPU relays."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
from statistics import mean
import sys

COUNTS=dict(A=16,B=20,C=16,D=15,E=20,F=15)


def plan(host,run,output):
    selection={"xju":"fixed_final","csu":"test_best"}[host]
    root=Path(output)/('results_'+run)
    queues={}
    for gpu,directions in (('0','ACE'),('1','BDF')):
        queues[gpu]=[dict(variant='cbst',selection=selection,direction=d,seeds=[43],folds=COUNTS[d],
            result_root=str(root),log_dir=str(Path(output)/'logs'/run/'main'/'seed_43')) for d in directions]
    assert sorted(j['direction'] for q in queues.values() for j in q)==list('ABCDEF')
    summary=dict(host=host,selection=selection,evaluation_interval=1,groups=1,directions=list('ABCDEF'),seeds=[43],jobs=6,folds=102,iterations=1000,
        gpu_directions={g:[j['direction'] for j in q] for g,q in queues.items()},
        gpu_folds={g:sum(j['folds'] for j in q) for g,q in queues.items()},switch_interval_seconds=1)
    return queues,summary


def command(python,item,data_dir,subjects=None,config=None):
    return [python,'-u','-m','MPUS_GA.trial_temporal.train_cbst','--direction',item['direction'],
        '--data-dir',str(data_dir),'--result-root',item['result_root'],'--seed','43','--subjects','all','--selection',item['selection']]


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def completed(root):
    counts={d:0 for d in COUNTS}
    for d,n in COUNTS.items():
        for subject in range(1,n+1):
            p=root/d/f'seed_43_subject_{subject:02d}.json'
            if all(p.with_suffix(ext).exists() for ext in ('.json','.pt','.pseudo.pt','.offline.json')):
                counts[d]+=1
    return counts


def report(root):
    rows=[]
    selection=None
    for d,n in COUNTS.items():
        values=[];last_values=[]
        for subject in range(1,n+1):
            p=root/d/f'seed_43_subject_{subject:02d}.offline.json'
            if p.exists():
                result=json.loads(p.read_text());values.append(result['primary']);last_values.append(result['last_step'])
                selection=result['reporting']['protocol']
        rows.append(dict(direction=d,completed=len(values),expected=n,complete=len(values)==n,
            metrics={k:mean(x[k] for x in values) for k in ('accuracy','balanced_accuracy','macro_f1')} if values else None,
            last_step={k:mean(x[k] for x in last_values) for k in ('accuracy','balanced_accuracy','macro_f1')} if last_values else None))
    return dict(selection=selection,primary_output='test_selected_fused' if selection=='test_best' else 'fused',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True)
    p.add_argument('--host',choices=['xju','csu'],required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args()
    if not a.run_name.startswith('cbst_') or Path(a.run_name).name!=a.run_name:p.error('Use cbst_<unique name>')
    queues,summary=plan(a.host,a.run_name,a.output_root)
    selection=summary["selection"]
    root=a.output_root/('results_'+a.run_name)
    if a.dry_run:print(json.dumps(dict(summary=summary,queues=queues),indent=2));return
    if a.report:print(json.dumps(report(root),indent=2));return
    if a.status:
        state=json.loads((root/'suite_state.json').read_text()) if (root/'suite_state.json').exists() else None
        if state and state.get('status')=='running':
            try:os.kill(state['pid'],0)
            except ProcessLookupError:state={**state,'status':'not_alive'}
        print(json.dumps(dict(summary=summary,completed=completed(root),state=state),indent=2));return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.suite.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('ALREADY RUNNING');return
        from .run_r3_suite import run_parallel_plans,SuiteStopped
        from .run_six_direction_final_suite import resolve_gpu_uuid,_code_hashes
        from .run_neighbor_soft_suite import environment
        from ..trial_temporal.cbst import CBSTConfig,REFERENCE_COMMIT
        from ..trial_temporal.train_cbst import complete
        from dataclasses import asdict
        package=Path(__file__).resolve().parents[1]
        uuids={g:resolve_gpu_uuid(g) for g in queues}
        if len(set(uuids.values()))!=2:raise RuntimeError('Two distinct physical GPUs required')
        data_files=[]
        for ds in ('seed_iv','seed_v','seed_vii'):
            files=sorted((a.data_dir/ds).glob('window_*/*.npz'))
            if not files:raise RuntimeError('Missing '+ds)
            data_files+=files
        manifest=dict(summary=summary,queues=queues,gpu_uuids=uuids,environment=environment(),
            code_package=str(package),code_sha256=_code_hashes(package),config=asdict(CBSTConfig()),
            frozen_profile_sha256=hashlib.sha256((package/'trial_temporal/pcdiag_frozen.json').read_bytes()).hexdigest(),
            reference_commit=REFERENCE_COMMIT,
            data_identity=[[str(f),f.stat().st_size,f.stat().st_mtime_ns] for f in data_files])
        path=root/'suite_manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Code/config/data manifest changed')
        if not path.exists() and list(root.glob('*/seed_*_subject_*.json')):raise RuntimeError('Refusing unmanifested results')
        save(path,manifest);save(root/'experiment_list.json',dict(summary=summary,queues=queues))
        def stop(signum,frame):raise SuiteStopped()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        save(root/'suite_state.json',dict(status='running',pid=os.getpid(),started=datetime.now().isoformat()))
        print('EXPERIMENT PLAN '+json.dumps(summary),flush=True)
        try:
            status=run_parallel_plans(queues,sys.executable,package,a.data_dir,'all',None,uuids,build_command=command)
            if status:raise RuntimeError('Training task failed, both relays stopped: '+str(status))
            for d,n in COUNTS.items():
                for subject in range(1,n+1):
                    if not complete(root/d/f'seed_43_subject_{subject:02d}.json',selection):raise RuntimeError('Missing formal fold')
            if completed(root)!=COUNTS:raise RuntimeError('Missing offline/checkpoint artifacts')
        except SuiteStopped:
            save(root/'suite_state.json',dict(status='stopped',completed=completed(root)));raise SystemExit(143)
        except BaseException as exc:
            save(root/'suite_state.json',dict(status='failed',error=str(exc),completed=completed(root)));raise
        save(root/'complete.json',dict(summary=summary,completed=completed(root)))
        save(root/'primary_summary.json',report(root))
        save(root/'suite_state.json',dict(status='complete',completed=completed(root)))
        print('ALL SIX DIRECTIONS COMPLETE',flush=True)


if __name__=='__main__':main()
