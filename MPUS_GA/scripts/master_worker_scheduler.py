#!/usr/bin/env python3
"""
Master-Worker 调度器：总队列 + 双卡并行接力

设计：
1. 所有任务放入统一队列（JSON 文件）
2. Master 进程管理队列和文件锁
3. 两个 Worker 进程（GPU 0 和 GPU 1）循环领取任务
4. 任务完成后自动领取下一个，直到队列清空

Created: 2026-10-01
"""

import json
import time
import fcntl
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class MasterScheduler:
    """总队列控制器"""

    def __init__(self, task_queue_file, lock_file, state_file):
        self.task_queue_file = Path(task_queue_file)
        self.lock_file = Path(lock_file)
        self.state_file = Path(state_file)

        # 初始化状态
        if not self.state_file.exists():
            self._save_state({
                'completed': [],
                'failed': [],
                'in_progress': {},
            })

    def _save_state(self, state):
        """保存状态到文件"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """加载状态"""
        with open(self.state_file, 'r') as f:
            return json.load(f)

    def _load_queue(self):
        """加载任务队列"""
        if not self.task_queue_file.exists():
            return []
        with open(self.task_queue_file, 'r') as f:
            return json.load(f)

    def _save_queue(self, queue):
        """保存任务队列"""
        with open(self.task_queue_file, 'w') as f:
            json.dump(queue, f, indent=2)

    def get_next_task(self, worker_id):
        """Worker 请求下一个任务"""
        with open(self.lock_file, 'a') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                queue = self._load_queue()
                state = self._load_state()

                if len(queue) == 0:
                    return None

                # 取出第一个任务
                task = queue.pop(0)
                self._save_queue(queue)

                # 标记为进行中
                state['in_progress'][task] = {
                    'worker_id': worker_id,
                    'start_time': datetime.now().isoformat(),
                }
                self._save_state(state)

                self._log(f"Worker {worker_id} 领取任务: {task}")
                return task

            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def report_completion(self, worker_id, task, success):
        """Worker 报告任务完成"""
        with open(self.lock_file, 'a') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._load_state()

                # 从进行中移除
                if task in state['in_progress']:
                    del state['in_progress'][task]

                # 添加到完成或失败列表
                if success:
                    state['completed'].append({
                        'task': task,
                        'worker_id': worker_id,
                        'time': datetime.now().isoformat(),
                    })
                    self._log(f"✓ Worker {worker_id} 完成: {task}")
                else:
                    state['failed'].append({
                        'task': task,
                        'worker_id': worker_id,
                        'time': datetime.now().isoformat(),
                    })
                    self._log(f"✗ Worker {worker_id} 失败: {task}")

                self._save_state(state)

            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _log(self, message):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        sys.stdout.flush()


class WorkerScheduler:
    """Worker 控制器（GPU 专属）"""

    def __init__(self, worker_id, gpu_id, master, config):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.master = master
        self.config = config

    def run(self):
        """循环领取任务直到队列空"""
        self._log("启动 Worker")

        while True:
            # 领取任务
            task = self.master.get_next_task(self.worker_id)
            if task is None:
                self._log("队列已空，退出")
                break

            # 解析任务：direction_config
            parts = task.rsplit('_', 1)
            if len(parts) == 2:
                direction, config_name = parts
            else:
                self._log(f"无法解析任务: {task}")
                self.master.report_completion(self.worker_id, task, False)
                continue

            # 运行训练
            success = self.run_training(direction, config_name)

            # 报告结果
            self.master.report_completion(self.worker_id, task, success)

            # 休息 1 秒避免文件锁冲突
            time.sleep(1)

        self._log("Worker 退出")

    def run_training(self, direction, config_name):
        """调用训练脚本"""
        # 构建命令
        cmd = [
            '/home/gzw/miniforge3/envs/BCI/bin/python',
            '-m', 'trial_temporal.train_confusion_fix',
            '--experiment', direction,
            '--config', config_name,
            '--random-seed', '43',
            '--target-subjects', 'all',
            '--data-root', self.config['data_root'],
            '--result-root', f"{self.config['result_root']}/{config_name}",
            '--device', 'cuda:0',
        ]

        # 日志文件
        log_dir = Path(self.config['log_root']) / config_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{direction}.log"

        self._log(f"开始训练: {direction} / {config_name}")
        self._log(f"日志: {log_file}")

        # 运行
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        try:
            with open(log_file, 'w') as f:
                ret = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=self.config['project_root'],
                )
            success = (ret.returncode == 0)
        except Exception as e:
            self._log(f"训练异常: {e}")
            success = False

        return success

    def _log(self, message):
        """写入日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [Worker {self.worker_id}] {message}"
        print(log_message)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='Master-Worker 调度器')
    parser.add_argument('--mode', choices=['master', 'worker'], required=True)
    parser.add_argument('--worker-id', type=int, help='Worker ID (0 or 1)')
    parser.add_argument('--gpu-id', type=int, help='GPU ID')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 创建 Master
    master = MasterScheduler(
        task_queue_file=config['task_queue_file'],
        lock_file=config['lock_file'],
        state_file=config['state_file'],
    )

    if args.mode == 'worker':
        # Worker 模式
        if args.worker_id is None or args.gpu_id is None:
            print("Worker 模式需要 --worker-id 和 --gpu-id")
            sys.exit(1)

        worker = WorkerScheduler(
            worker_id=args.worker_id,
            gpu_id=args.gpu_id,
            master=master,
            config=config,
        )
        worker.run()

    else:
        # Master 模式（仅用于监控）
        print("Master 模式：监控队列状态")
        while True:
            state = master._load_state()
            queue = master._load_queue()

            print(f"\n{'='*60}")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"队列剩余: {len(queue)}")
            print(f"进行中: {len(state['in_progress'])}")
            print(f"已完成: {len(state['completed'])}")
            print(f"失败: {len(state['failed'])}")

            if len(queue) == 0 and len(state['in_progress']) == 0:
                print("\n所有任务完成！")
                break

            time.sleep(60)  # 每分钟检查一次


if __name__ == '__main__':
    main()
