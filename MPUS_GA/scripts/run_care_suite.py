"""R2 + CARE: two GPU relays, six directions, two full seeds and ablation seed 43."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from MPUS_GA.trial_temporal.care_pseudo_label import care_config, CARE_VARIANTS
from MPUS_GA.trial_temporal.multiscale_coteaching import CoTeachingConfig
from .run_r2_subgroup_suite import command as r2_command
from .run_r3_suite import build_plan, split_plan, plan_summary, run_parallel_plans, SuiteStopped
from .run_six_direction_final_suite import _code_hashes, resolve_gpu_uuid

MODULE='MPUS_GA.scripts.run_care_suite'
DEFAULT_VARIANTS=('full','r2','anchor_ce','no_geometry','fixed_selection')


SOURCE_BATCH = 8
TARGET_BATCH = 8
CUDA_BUDGET_GIB = 8.


def training_command(python,item,data_dir,subjects,config):
    command = r2_command(python,item['direction'],data_dir,item['result_root'],item['seeds'],subjects,
                         config,method='care')
    command[command.index('--source-batch-size')+1] = str(SOURCE_BATCH)
    command[command.index('--target-batch-size')+1] = str(TARGET_BATCH)
    return command + ['--care-variant',item['variant'], '--cuda-memory-budget-gib',str(CUDA_BUDGET_GIB)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs=2,default=['0','1'])
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=CARE_VARIANTS,default=list(DEFAULT_VARIANTS))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if len(set(args.gpus))!=2 or any(not g.isdigit() for g in args.gpus):parser.error('Provide two distinct GPUs')
    if not args.run_name.startswith('care_') or Path(args.run_name).name!=args.run_name:parser.error('Use a new care_ run name')
    if len(set(args.variants))!=len(args.variants):parser.error('Variants must be unique')
    package=Path(__file__).resolve().parents[1]
    plan=build_plan(args.run_name,args.output_root,args.variants,[43,42],43)
    queues=split_plan(plan,args.gpus)
    summary={'configurations':len(args.variants),**plan_summary(plan),
             'gpu_queues':{g:{'directions':list(dict.fromkeys(i['direction'] for i in q)),**plan_summary(q)} for g,q in queues.items()},
             'switch_interval_seconds':1., 'source_batch_size':SOURCE_BATCH,
             'target_batch_size':TARGET_BATCH, 'cuda_allocator_budget_gib':CUDA_BUDGET_GIB}
    if args.dry_run:
        print(json.dumps({'summary':summary,'commands':[training_command(sys.executable,p,args.data_dir,'all',CoTeachingConfig()) for p in plan]},indent=2))
        return
    uuids={g:resolve_gpu_uuid(g) for g in args.gpus}
    files=sorted(args.data_dir.glob('seed_*/window_*/*.npz'))
    if not files:raise SystemExit('Processed data missing')
    root=args.output_root/f'results_{args.run_name}';log_dir=args.output_root/'logs'/args.run_name
    root.mkdir(parents=True,exist_ok=True);log_dir.mkdir(parents=True,exist_ok=True)
    manifest={'method':'care','summary':summary,'plan':plan,'gpu_uuids':uuids,
              'care_configs':{v:asdict(care_config(v)) for v in args.variants},
              'r2_config':asdict(CoTeachingConfig()),'code_sha256':_code_hashes(package),
              'data_identity':[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files],
              'python':sys.executable,'protocol':'fixed_final_1000; no_target_truth_selection'}
    serialized=json.dumps(manifest,sort_keys=True,indent=2)+'\n';path=root/'suite_manifest.json'
    if path.exists() and path.read_text()!=serialized:raise SystemExit('Code/config/data changed; choose a new run name')
    if not args.worker:
        with (root/'.suite.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit('This CARE batch is already running')
        with (log_dir/'suite.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,*sys.argv[1:],'--worker'],
                                   cwd=package.parent,env=dict(os.environ,PYTHONPATH=str(package.parent)),
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'worker_pid':child.pid,'suite_log':str(log_dir/'suite.log'),**summary},indent=2))
        return
    with (root/'.suite.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This CARE batch is already running')
        if not path.exists() and any(list(Path(i['result_root']).glob('*/seed_*_subject_*.json')) for i in plan):
            raise SystemExit('Refusing unmanifested results')
        path.write_text(serialized)
        for item in plan:
            variant_root=Path(item['result_root']);variant_root.mkdir(parents=True,exist_ok=True)
            (variant_root/'suite_manifest.json').write_text(serialized)
        def stop(signum,frame):raise SuiteStopped()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        print('SUITE START '+json.dumps(summary),flush=True)
        try:
            code=run_parallel_plans(queues,sys.executable,package,args.data_dir,'all',CoTeachingConfig(),uuids,
                                    build_command=training_command)
        except SuiteStopped:
            print('SUITE STOPPED; subsequent tasks cancelled',flush=True);raise SystemExit(143)
        print('SUITE END exit='+str(code),flush=True)
        raise SystemExit(code)


if __name__=='__main__':main()
