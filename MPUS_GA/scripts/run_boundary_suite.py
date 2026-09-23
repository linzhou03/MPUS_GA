"""Complete six-direction/two-seed prototype-free boundary study on two GPU relays."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

COUNTS = dict(A=16,B=20,C=16,D=15,E=20,F=15)
VARIANTS = ('r2','proto_off','dual_source','dual_mcd')
SEEDS = (42,43)


def make_plan(run_name, output_root, variants=VARIANTS):
    root=Path(output_root)/f'results_{run_name}'
    logs=Path(output_root)/'logs'/run_name
    queues={str(i):[] for i in range(2)}
    for gpu,seed in zip(queues,SEEDS):
        for d in COUNTS:
            for variant in variants:
                queues[gpu].append(dict(variant=variant,direction=d,seeds=[seed],folds=COUNTS[d],
                    result_root=str(root/variant),log_dir=str(logs/variant/f'seed_{seed}')))
    keys=[(j['variant'],j['direction'],j['seeds'][0]) for q in queues.values() for j in q]
    if len(keys)!=len(set(keys)) or len(keys)!=len(variants)*12:
        raise RuntimeError('Missing or duplicated study jobs')
    summary=dict(variants=list(variants),directions=list(COUNTS),seeds=list(SEEDS),
        groups=len(variants),jobs=len(keys),folds=len(variants)*204,folds_per_group=204,
        gpu_jobs={g:len(q) for g,q in queues.items()},gpu_folds={g:sum(j['folds'] for j in q) for g,q in queues.items()},
        switch_interval_seconds=1,iterations=1000,
        order='Each GPU: direction A..F; within each direction r2/proto_off/dual_source/dual_mcd',
        optimizer_steps=dict(r2=1000,proto_off=1000,dual_source=2400,dual_mcd=2400))
    return queues,summary


def command(python,item,data_dir,subjects=None,config=None):
    return [python,'-u','-m','MPUS_GA.trial_temporal.train_boundary_study',
        '--direction',item['direction'],'--variant',item['variant'],'--seeds',*map(str,item['seeds']),
        '--data-dir',str(data_dir),'--result-root',item['result_root'],'--subjects','all']


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def completed(root,variants):
    counts={v:0 for v in variants}
    for v in variants:
        for d,n in COUNTS.items():
            for s in SEEDS:
                for subject in range(1,n+1):
                    p=root/v/d/f'seed_{s}_subject_{subject:02d}.json'
                    if all(p.with_suffix(ext).exists() for ext in ('.json','.pt','.pseudo.pt','.offline.json')):
                        counts[v]+=1
    return counts


def verify_finished(root,variants):
    counts=completed(root,variants)
    if any(n!=204 for n in counts.values()):
        raise RuntimeError('Expected 204 complete folds per group: '+str(counts))
    for d,n in COUNTS.items():
        for s in SEEDS:
            for subject in range(1,n+1):
                records={}
                for v in variants:
                    p=root/v/d/f'seed_{s}_subject_{subject:02d}.json'
                    r=json.loads(p.read_text())
                    if r['protocol']['selected_iteration']!=1000 or r['boundary_study']['observed_steps']!=1000:
                        raise RuntimeError('Incomplete training protocol')
                    if r['boundary_study']['config']['variant']!=v:
                        raise RuntimeError('Wrong condition in '+str(p))
                    if v!='r2' and (r['final_prototype_bank'] is not None or r['final_source_prototype_memory'] is not None):
                        raise RuntimeError('Unexpected prototype state')
                    records[v]=r['boundary_study']
                for key in ('backbone_initial_sha256','heads_initial_sha256','batch_sequence_sha256','optimizer_steps'):
                    if records['dual_source'][key]!=records['dual_mcd'][key]:
                        raise RuntimeError(f'Paired dual-group control mismatch: {d}/{s}/{subject}/{key}')
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if not a.run_name.startswith('boundary_') or Path(a.run_name).name!=a.run_name:
        p.error('Use boundary_<unique name>')
    if len(a.variants)!=len(set(a.variants)) or not {'proto_off','dual_source','dual_mcd'}<=set(a.variants):
        p.error('Study requires proto_off/dual_source/dual_mcd exactly once, optionally r2')
    queues,summary=make_plan(a.run_name,a.output_root,a.variants)
    root=a.output_root/f'results_{a.run_name}';logs=a.output_root/'logs'/a.run_name
    if a.dry_run:
        print(json.dumps(dict(summary=summary,queues=queues),indent=2));return
    if a.status:
        state=json.loads((root/'suite_state.json').read_text()) if (root/'suite_state.json').exists() else None
        print(json.dumps(dict(summary=summary,completed=completed(root,a.variants),state=state),indent=2));return
    root.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    with (root/'.suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('ALREADY RUNNING: no second queue started');return
        from .run_r3_suite import run_parallel_plans,SuiteStopped
        from .run_six_direction_final_suite import resolve_gpu_uuid,_code_hashes
        from .run_neighbor_soft_suite import environment
        from ..trial_temporal.boundary_adaptation import BoundaryConfig
        from dataclasses import asdict
        package=Path(__file__).resolve().parents[1]
        uuids={g:resolve_gpu_uuid(g) for g in queues}
        if len(set(uuids.values()))!=2:
            raise RuntimeError('GPU UUIDs must differ')
        files=[]
        for ds in ('seed_iv','seed_v','seed_vii'):
            ds_files=sorted((a.data_dir/ds).glob('window_*/*.npz'))
            if not ds_files:raise RuntimeError('Missing dataset '+ds)
            files+=ds_files
        manifest=dict(summary=summary,queues=queues,environment=environment(),gpu_uuids=uuids,
            code_sha256=_code_hashes(package),code_package=str(package),
            frozen_profile_sha256=hashlib.sha256((package/'trial_temporal/pcdiag_frozen.json').read_bytes()).hexdigest(),
            config={v:asdict(BoundaryConfig(v)) for v in a.variants},
            data_identity=[[str(f),f.stat().st_size,f.stat().st_mtime_ns] for f in files])
        path=root/'suite_manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise RuntimeError('Manifest mismatch; code/data/config changed')
        if not path.exists() and list(root.glob('*/*/seed_*_subject_*.json')):
            raise RuntimeError('Refusing unmanifested result reuse')
        save(path,manifest)
        save(root/'experiment_list.json',dict(summary=summary,queues=queues))
        def stop(signum,frame):raise SuiteStopped()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        save(root/'suite_state.json',dict(status='running',pid=os.getpid(),started=datetime.now().isoformat()))
        print('FULL EXPERIMENT PLAN '+json.dumps(summary),flush=True)
        try:
            status=run_parallel_plans(queues,sys.executable,package,a.data_dir,'all',None,uuids,build_command=command)
            if status:raise RuntimeError('A training task failed; both relays stopped, exit='+str(status))
            counts=verify_finished(root,a.variants)
        except SuiteStopped:
            save(root/'suite_state.json',dict(status='stopped',pid=os.getpid(),completed=completed(root,a.variants)))
            raise SystemExit(143)
        except BaseException as exc:
            save(root/'suite_state.json',dict(status='failed',error=str(exc),completed=completed(root,a.variants)))
            raise
        save(root/'complete.json',dict(summary=summary,completed=counts,paired_controls_verified=True))
        from .report_boundary_study import summarize
        save(root/'primary_summary.json',summarize(root))
        save(root/'suite_state.json',dict(status='complete',completed=counts))
        print('ALL STUDY GROUPS COMPLETE',flush=True)


if __name__=='__main__':main()
