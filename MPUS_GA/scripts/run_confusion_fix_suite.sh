#!/bin/bash
#
# 混淆矩阵改进实验：ACE 三方向双卡并行调度脚本
#
# 使用方法：
#   bash scripts/run_confusion_fix_suite.sh start   # 启动实验
#   bash scripts/run_confusion_fix_suite.sh status  # 查看状态
#   bash scripts/run_confusion_fix_suite.sh stop    # 停止实验
#
# Created: 2026-10-01

set -e

BATCH_NAME="confusion_fix_lds_ace_20261001_v1"
PROJECT_ROOT="/home/gzw/projects/MPUS_GA"
DATA_ROOT="/home/gzw/projects/MPUS_GA/data_processed_ivfront_lds_a1c1q2r1p01_20260929_v1"
RESULT_ROOT="${PROJECT_ROOT}/results_${BATCH_NAME}"
LOG_ROOT="${PROJECT_ROOT}/logs/${BATCH_NAME}"

# 临时文件
TASK_QUEUE_FILE="/tmp/task_queue_${BATCH_NAME}.json"
LOCK_FILE="/tmp/${BATCH_NAME}.lock"
STATE_FILE="/tmp/${BATCH_NAME}_state.json"
CONFIG_FILE="/tmp/${BATCH_NAME}_config.json"

# Worker PID 文件
WORKER0_PID_FILE="/tmp/${BATCH_NAME}_worker0.pid"
WORKER1_PID_FILE="/tmp/${BATCH_NAME}_worker1.pid"

# 创建配置文件
create_config() {
    cat > "${CONFIG_FILE}" <<EOF
{
  "project_root": "${PROJECT_ROOT}",
  "data_root": "${DATA_ROOT}",
  "result_root": "${RESULT_ROOT}",
  "log_root": "${LOG_ROOT}",
  "task_queue_file": "${TASK_QUEUE_FILE}",
  "lock_file": "${LOCK_FILE}",
  "state_file": "${STATE_FILE}"
}
EOF
    echo "配置文件已创建: ${CONFIG_FILE}"
}

# 创建任务队列
create_task_queue() {
    cat > "${TASK_QUEUE_FILE}" <<EOF
[
  "A_baseline", "C_baseline", "E_baseline",
  "A_asy", "C_asy", "E_asy",
  "A_cal", "C_cal", "E_cal",
  "A_cbst", "C_cbst", "E_cbst",
  "A_asy_cal", "C_asy_cal", "E_asy_cal",
  "A_asy_cbst", "C_asy_cbst", "E_asy_cbst",
  "A_cal_cbst", "C_cal_cbst", "E_cal_cbst",
  "A_full", "C_full", "E_full"
]
EOF
    echo "任务队列已创建: ${TASK_QUEUE_FILE}"
    echo "总任务数: 24"
}

# 启动实验
start_experiment() {
    echo "=============================="
    echo "启动混淆矩阵改进实验"
    echo "=============================="
    echo "批次名称: ${BATCH_NAME}"
    echo "数据路径: ${DATA_ROOT}"
    echo "结果路径: ${RESULT_ROOT}"
    echo "日志路径: ${LOG_ROOT}"
    echo ""

    # 检查数据是否存在
    if [ ! -d "${DATA_ROOT}" ]; then
        echo "错误: 数据目录不存在: ${DATA_ROOT}"
        exit 1
    fi

    # 创建必要的目录
    mkdir -p "${RESULT_ROOT}"
    mkdir -p "${LOG_ROOT}"

    # 创建配置和队列
    create_config
    create_task_queue

    # 清空旧的状态文件
    rm -f "${STATE_FILE}"
    rm -f "${LOCK_FILE}"

    # 启动 Worker 0（GPU 0）
    echo ""
    echo "启动 Worker 0 (GPU 0)..."
    nohup /home/gzw/miniforge3/envs/BCI/bin/python \
        scripts/master_worker_scheduler.py \
        --mode worker \
        --worker-id 0 \
        --gpu-id 0 \
        --config "${CONFIG_FILE}" \
        > "${LOG_ROOT}/suite_gpu0.log" 2>&1 &

    WORKER0_PID=$!
    echo ${WORKER0_PID} > "${WORKER0_PID_FILE}"
    echo "Worker 0 PID: ${WORKER0_PID}"

    # 启动 Worker 1（GPU 1）
    echo ""
    echo "启动 Worker 1 (GPU 1)..."
    nohup /home/gzw/miniforge3/envs/BCI/bin/python \
        scripts/master_worker_scheduler.py \
        --mode worker \
        --worker-id 1 \
        --gpu-id 1 \
        --config "${CONFIG_FILE}" \
        > "${LOG_ROOT}/suite_gpu1.log" 2>&1 &

    WORKER1_PID=$!
    echo ${WORKER1_PID} > "${WORKER1_PID_FILE}"
    echo "Worker 1 PID: ${WORKER1_PID}"

    echo ""
    echo "=============================="
    echo "实验启动成功！"
    echo "=============================="
    echo ""
    echo "查看状态: bash scripts/run_confusion_fix_suite.sh status"
    echo "查看 GPU 0 日志: tail -f ${LOG_ROOT}/suite_gpu0.log"
    echo "查看 GPU 1 日志: tail -f ${LOG_ROOT}/suite_gpu1.log"
    echo ""
}

# 查看状态
show_status() {
    echo "=============================="
    echo "实验状态"
    echo "=============================="

    # 检查 Worker 是否运行
    if [ -f "${WORKER0_PID_FILE}" ]; then
        WORKER0_PID=$(cat "${WORKER0_PID_FILE}")
        if ps -p ${WORKER0_PID} > /dev/null 2>&1; then
            echo "Worker 0 (PID ${WORKER0_PID}): 运行中"
        else
            echo "Worker 0 (PID ${WORKER0_PID}): 已停止"
        fi
    else
        echo "Worker 0: 未启动"
    fi

    if [ -f "${WORKER1_PID_FILE}" ]; then
        WORKER1_PID=$(cat "${WORKER1_PID_FILE}")
        if ps -p ${WORKER1_PID} > /dev/null 2>&1; then
            echo "Worker 1 (PID ${WORKER1_PID}): 运行中"
        else
            echo "Worker 1 (PID ${WORKER1_PID}): 已停止"
        fi
    else
        echo "Worker 1: 未启动"
    fi

    echo ""

    # 显示队列状态
    if [ -f "${STATE_FILE}" ]; then
        echo "队列状态:"
        python3 << EOF
import json
with open('${STATE_FILE}', 'r') as f:
    state = json.load(f)
with open('${TASK_QUEUE_FILE}', 'r') as f:
    queue = json.load(f)

print(f"  队列剩余: {len(queue)}")
print(f"  进行中: {len(state['in_progress'])}")
print(f"  已完成: {len(state['completed'])}")
print(f"  失败: {len(state['failed'])}")

if state['in_progress']:
    print("\n当前任务:")
    for task, info in state['in_progress'].items():
        print(f"  {task} (Worker {info['worker_id']})")

if state['failed']:
    print("\n失败任务:")
    for item in state['failed']:
        print(f"  {item['task']} (Worker {item['worker_id']})")
EOF
    else
        echo "状态文件不存在"
    fi

    echo ""
    echo "=============================="
}

# 停止实验
stop_experiment() {
    echo "=============================="
    echo "停止实验"
    echo "=============================="

    # 停止 Worker 0
    if [ -f "${WORKER0_PID_FILE}" ]; then
        WORKER0_PID=$(cat "${WORKER0_PID_FILE}")
        if ps -p ${WORKER0_PID} > /dev/null 2>&1; then
            echo "停止 Worker 0 (PID ${WORKER0_PID})..."
            kill ${WORKER0_PID}
        fi
        rm -f "${WORKER0_PID_FILE}"
    fi

    # 停止 Worker 1
    if [ -f "${WORKER1_PID_FILE}" ]; then
        WORKER1_PID=$(cat "${WORKER1_PID_FILE}")
        if ps -p ${WORKER1_PID} > /dev/null 2>&1; then
            echo "停止 Worker 1 (PID ${WORKER1_PID})..."
            kill ${WORKER1_PID}
        fi
        rm -f "${WORKER1_PID_FILE}"
    fi

    echo ""
    echo "实验已停止"
    echo "=============================="
}

# 主函数
main() {
    case "${1:-}" in
        start)
            start_experiment
            ;;
        status)
            show_status
            ;;
        stop)
            stop_experiment
            ;;
        *)
            echo "使用方法: $0 {start|status|stop}"
            exit 1
            ;;
    esac
}

main "$@"
