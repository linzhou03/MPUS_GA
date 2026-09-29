#!/bin/bash
#
# 混淆矩阵改进实验 - 启动前检查脚本
#
# 在 CSU 上执行此脚本，确保所有前置条件满足
#
# 使用方法：bash scripts/pre_launch_check.sh

set -e

echo "======================================"
echo "混淆矩阵改进实验 - 启动前检查"
echo "======================================"
echo ""

PASSED=0
FAILED=0

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED=$((FAILED + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. 检查数据目录
echo "1. 检查数据..."
DATA_ROOT="/home/gzw/projects/MPUS_GA/data_processed_ivfront_lds_a1c1q2r1p01_20260929_v1"
if [ -d "${DATA_ROOT}" ]; then
    NPZ_COUNT=$(find "${DATA_ROOT}" -name "*.npz" | wc -l)
    if [ ${NPZ_COUNT} -eq 519 ]; then
        check_pass "数据目录存在，包含 519 个 NPZ 文件"
    else
        check_warn "数据目录存在，但 NPZ 文件数量为 ${NPZ_COUNT}（预期 519）"
    fi
else
    check_fail "数据目录不存在: ${DATA_ROOT}"
fi

# 2. 检查 Python 环境
echo ""
echo "2. 检查 Python 环境..."
PYTHON_BIN="/home/gzw/miniforge3/envs/BCI/bin/python"
if [ -f "${PYTHON_BIN}" ]; then
    PYTHON_VERSION=$(${PYTHON_BIN} --version 2>&1)
    check_pass "Python 环境: ${PYTHON_VERSION}"
else
    check_fail "Python 解释器不存在: ${PYTHON_BIN}"
fi

# 3. 检查 GPU
echo ""
echo "3. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ ${GPU_COUNT} -ge 2 ]; then
        check_pass "GPU 数量: ${GPU_COUNT}"
        nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | while read line; do
            echo "    ${line}"
        done
    else
        check_warn "GPU 数量为 ${GPU_COUNT}（预期至少 2）"
    fi
else
    check_fail "nvidia-smi 不可用"
fi

# 4. 检查核心模块文件
echo ""
echo "4. 检查核心模块..."
MODULES=(
    "trial_temporal/losses/asymmetric_confusion.py"
    "trial_temporal/calibration/multiscale_balance.py"
    "trial_temporal/pseudo_label/confusion_aware_cbst.py"
    "trial_temporal/train_confusion_fix.py"
)

for module in "${MODULES[@]}"; do
    if [ -f "${module}" ]; then
        check_pass "${module}"
    else
        check_fail "${module} 不存在"
    fi
done

# 5. 检查调度脚本
echo ""
echo "5. 检查调度脚本..."
SCRIPTS=(
    "scripts/master_worker_scheduler.py"
    "scripts/run_confusion_fix_suite.sh"
    "scripts/summarize_confusion_fix.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "${script}" ]; then
        if [ -x "${script}" ]; then
            check_pass "${script} (可执行)"
        else
            check_warn "${script} (无执行权限)"
        fi
    else
        check_fail "${script} 不存在"
    fi
done

# 6. 运行单元测试
echo ""
echo "6. 运行单元测试..."

echo "  测试非对称损失..."
if ${PYTHON_BIN} -m trial_temporal.losses.asymmetric_confusion > /tmp/test_asy.log 2>&1; then
    check_pass "非对称损失模块测试通过"
else
    check_fail "非对称损失模块测试失败（查看 /tmp/test_asy.log）"
fi

echo "  测试偏置校准..."
if ${PYTHON_BIN} -m trial_temporal.calibration.multiscale_balance > /tmp/test_cal.log 2>&1; then
    check_pass "偏置校准模块测试通过"
else
    check_fail "偏置校准模块测试失败（查看 /tmp/test_cal.log）"
fi

echo "  测试 CA-CBST..."
if ${PYTHON_BIN} -m trial_temporal.pseudo_label.confusion_aware_cbst > /tmp/test_cbst.log 2>&1; then
    check_pass "CA-CBST 模块测试通过"
else
    check_fail "CA-CBST 模块测试失败（查看 /tmp/test_cbst.log）"
fi

# 7. 检查磁盘空间
echo ""
echo "7. 检查磁盘空间..."
AVAILABLE=$(df -h /home/gzw/projects/MPUS_GA | awk 'NR==2 {print $4}')
check_pass "可用空间: ${AVAILABLE}"

# 8. 检查是否有其他实验正在运行
echo ""
echo "8. 检查进程冲突..."
EXISTING_WORKERS=$(ps aux | grep "master_worker_scheduler.py" | grep -v grep | wc -l)
if [ ${EXISTING_WORKERS} -gt 0 ]; then
    check_warn "已有 ${EXISTING_WORKERS} 个 worker 进程在运行"
    ps aux | grep "master_worker_scheduler.py" | grep -v grep
else
    check_pass "无冲突进程"
fi

# 总结
echo ""
echo "======================================"
echo "检查完成"
echo "======================================"
echo -e "${GREEN}通过: ${PASSED}${NC}"
echo -e "${RED}失败: ${FAILED}${NC}"

if [ ${FAILED} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 所有检查通过，可以启动实验${NC}"
    echo ""
    echo "下一步："
    echo "  bash scripts/run_confusion_fix_suite.sh start"
    exit 0
else
    echo ""
    echo -e "${RED}✗ 有 ${FAILED} 项检查失败，请修复后再启动${NC}"
    exit 1
fi
