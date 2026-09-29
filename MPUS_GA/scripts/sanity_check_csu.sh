#!/bin/bash
# Quick sanity check for confusion fix experiment on CSU

echo "=== Pre-launch Sanity Check ==="

# 1. Check required modules exist
echo "1. Checking module files..."
REQUIRED_FILES=(
    "MPUS_GA/trial_temporal/train_cbst_confusion_fix.py"
    "MPUS_GA/trial_temporal/losses/asymmetric_confusion.py"
    "MPUS_GA/trial_temporal/calibration/multiscale_balance.py"
    "MPUS_GA/trial_temporal/pseudo_label/confusion_aware_cbst.py"
    "MPUS_GA/scripts/master_worker_scheduler.py"
    "MPUS_GA/scripts/run_confusion_fix_suite.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done

# 2. Check data directory
echo
echo "2. Checking data directory..."
if [ -d "MPUS_GA/data_processed" ]; then
    echo "  ✓ data_processed exists"
    echo "  Files: $(find MPUS_GA/data_processed -name "*.npz" | wc -l) .npz files"
else
    echo "  ✗ data_processed not found"
    exit 1
fi

# 3. Check Python import
echo
echo "3. Testing Python imports..."
python3 << 'ENDPY'
try:
    from MPUS_GA.trial_temporal.losses.asymmetric_confusion import AsymmetricConfusionLoss
    from MPUS_GA.trial_temporal.calibration.multiscale_balance import MultiscaleBalanceCalibrator
    from MPUS_GA.trial_temporal.pseudo_label.confusion_aware_cbst import ConfusionAwareCBST
    print("  ✓ All modules import successfully")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    exit(1)
ENDPY

# 4. Check GPU availability
echo
echo "4. Checking GPU availability..."
python3 << 'ENDPY'
import torch
if torch.cuda.is_available():
    print(f"  ✓ {torch.cuda.device_count()} GPUs available")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  ✗ No GPU available")
    exit(1)
ENDPY

echo
echo "=== All checks passed! ==="
echo "Ready to launch experiments."
