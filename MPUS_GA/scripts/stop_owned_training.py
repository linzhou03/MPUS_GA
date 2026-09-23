"""Preview or stop only your exact training suite / processes on one physical GPU.

Linux-only. Default is read-only. --execute is an explicit command for the user
to run when ready; deployment and validation must never pass it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_table():
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            args = [part.decode(errors='replace') for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
            result[int(directory.name)] = {'ppid': int(fields[1]), 'start': int(fields[19]), 'args': args}
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return result


def argument(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def descendants(roots, processes):
    selected = set(roots)
    while True:
        added = {pid for pid, row in processes.items() if row['ppid'] in selected} - selected
        if not added:
            return selected
        selected.update(added)


def select_suite(processes, run_name):
    roots = set()
    for pid, row in processes.items():
        args = row['args']
        module = argument(args, '-m') or ''
        if module.startswith('MPUS_GA.scripts.run_') and module.endswith('_suite') and argument(args, '--run-name') == run_name:
            roots.add(pid)
        result_root = argument(args, '--result-root')
        if module in ('MPUS_GA.trial_temporal.train', 'MPUS_GA.trial_temporal.train_msmr') and result_root and Path(result_root).name == 'results_' + run_name:
            roots.add(pid)
        if module in ('MPUS_GA.trial_temporal.train_pcdiag', 'MPUS_GA.trial_temporal.pcdiag_observe',
                      'MPUS_GA.trial_temporal.train_oracle_study',
                      'MPUS_GA.trial_temporal.train_uncertainty_pseudo',
                      'MPUS_GA.trial_temporal.train_uncertainty_best',
                      'MPUS_GA.trial_temporal.train_cbst',
                      'MPUS_GA.trial_temporal.train_neighbor_soft') and result_root:
            if any(p.name == 'results_' + run_name for p in (Path(result_root), *Path(result_root).parents)):
                roots.add(pid)
    return roots


def gpu_processes(gpu):
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if not uuid.startswith('GPU-') or '\n' in uuid:
        raise RuntimeError('Expected exactly one physical GPU')
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    mapping = {}
    for line in output.splitlines():
        device, pid = (part.strip() for part in line.split(',', 1))
        if pid.isdigit():
            mapping.setdefault(int(pid), set()).add(device)
    return uuid, mapping


def select_gpu(processes, uuid, mapping):
    roots = {pid for pid, devices in mapping.items() if uuid in devices and pid in processes}
    if any(mapping[pid] - {uuid} for pid in roots):
        raise RuntimeError('An owned process spans multiple GPUs; refusing a single-GPU stop')
    # Include a same-GPU MPUS supervisor so it cannot restart the terminated child.
    # Never stop a shared supervisor that also owns a process on another GPU.
    for pid in list(roots):
        child_args = processes[pid]['args']
        parent = processes[pid]['ppid']
        while parent in processes:
            args = processes[parent]['args']
            module = argument(args, '-m') or ''
            is_suite = module.startswith('MPUS_GA.scripts.run_') and module.endswith('_suite')
            is_spawn_parent = (parent == processes[pid]['ppid'] and '--multiprocessing-fork' in child_args
                               and len(args) > 1 and Path(args[0]).name.startswith('python') and args[1].endswith('.py'))
            if is_suite or is_spawn_parent:
                family = descendants({parent}, processes)
                if not any(devices - {uuid} for member, devices in mapping.items() if member in family):
                    roots.add(parent)
            parent = processes[parent]['ppid']
    return roots


def same_process(pid, original):
    current = process_table().get(pid)
    return current is not None and current['start'] == original['start']


def stop(roots, original):
    frozen = []
    try:
        for pid in roots:
            if same_process(pid, original[pid]):
                os.kill(pid, signal.SIGSTOP); frozen.append(pid)
        current = process_table()
        targets = descendants(set(frozen), current)
        identities = {pid: current[pid] for pid in targets if pid in current}
        for pid, row in identities.items():
            if same_process(pid, row):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    finally:
        for pid in frozen:
            if same_process(pid, original[pid]):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
    deadline = time.monotonic() + 10
    remaining = identities
    while remaining and time.monotonic() < deadline:
        remaining = {pid: row for pid, row in remaining.items() if same_process(pid, row)}
        if remaining:
            time.sleep(.2)
    for pid, row in remaining.items():
        if same_process(pid, row):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    print(json.dumps({'termination_sent_to': sorted(identities), 'forced_kill': sorted(remaining)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--suite-run-name')
    select.add_argument('--gpu', type=int)
    parser.add_argument('--execute', action='store_true', help='Actually send signals; omit for read-only preview')
    args = parser.parse_args()
    if not Path('/proc/self/stat').exists():
        parser.error('This helper must run on the Linux training server')
    processes = process_table()
    if args.suite_run_name:
        if Path(args.suite_run_name).name != args.suite_run_name:
            parser.error('Provide the exact run name, without a path')
        roots = select_suite(processes, args.suite_run_name)
    else:
        uuid, mapping = gpu_processes(args.gpu)
        roots = select_gpu(processes, uuid, mapping)
    if os.getpid() in descendants(roots, processes):
        raise SystemExit('Refusing to stop the command executing this helper')
    selected = descendants(roots, processes)
    print(json.dumps({'mode': 'execute' if args.execute else 'preview', 'uid': os.getuid(),
                      'roots': sorted(roots), 'processes': [{'pid': pid, **processes[pid]} for pid in sorted(selected)]}, indent=2))
    if args.execute and roots:
        stop(roots, processes)
    elif not roots:
        print('No matching processes owned by this user')


if __name__ == '__main__':
    main()
