# Confusion Fix Experiment - Quick Start Guide

## 📋 Overview
This experiment validates confusion matrix improvement modules for R4 CBST pipeline.

**Branch**: `r4_confusion_fix_20261001`

**Key Components**:
- AsymmetricConfusionLoss: penalizes specific confusion patterns
- MultiscaleBalanceCalibrator: calibrates scale-specific biases
- ConfusionAwareCBST: integrates reliability states into pseudo-label selection

## 🚀 Quick Launch on CSU

### 1. Deploy to CSU
```bash
# On local machine
cd /Users/gongzhenwei/Desktop/gongzhenwei/projects/CAGA-SGA/MPUS_GA
./scripts/deploy_confusion_fix_csu.sh
```

### 2. SSH to CSU and verify
```bash
ssh csu
cd /data/linzhou/MPUS_GA
git branch  # should show r4_confusion_fix_20261001
./scripts/sanity_check_csu.sh  # run pre-launch checks
```

### 3. Launch scheduler (automatic GPU assignment)
```bash
nohup python scripts/master_worker_scheduler.py > scheduler.log 2>&1 &
tail -f scheduler.log
```

The scheduler will:
- Monitor 4 GPUs (3,4,5,6)
- Launch jobs when GPUs are free
- Retry failed jobs after 3 minutes
- Process all 8 configs × 6 directions = 48 runs

### 4. Monitor progress
```bash
# Watch scheduler log
tail -f scheduler.log

# Check specific job logs
ls -lth analysis/confusion_fix_results_*/direction_*/run_*.log

# Quick summary
python scripts/summarize_confusion_fix.py
```

## 📊 Expected Outputs

Each run produces:
```
analysis/confusion_fix_results_{timestamp}/
├── direction_A/
│   ├── {config}_seed{s}/
│   │   ├── checkpoints/post300_bal_best.pth
│   │   ├── final_report.json
│   │   ├── loss_curves_{config}.png
│   │   └── selection_*.json
│   └── run_{config}_seed{s}.log
├── ... (B through F)
└── aggregate_results.csv
```

## 🔧 Manual Single Run (for debugging)
```bash
# Example: baseline config, direction A, seed 42
python trial_temporal/train_cbst_confusion_fix.py \
    --ablation_config baseline \
    --direction A \
    --seed 42 \
    --output_base analysis/confusion_fix_test
```

## 📈 Analysis
```bash
# Generate summary tables and plots
python scripts/summarize_confusion_fix.py \
    --result_dir analysis/confusion_fix_results_20261001_0230
```

## ⚙️ Ablation Configs
1. **baseline**: R4 vanilla CBST
2. **asy**: + AsymmetricConfusionLoss
3. **cal**: + MultiscaleBalanceCalibrator
4. **cbst**: + ConfusionAwareCBST
5. **asy_cal**: asymmetric + calibration
6. **asy_cbst**: asymmetric + confusion-aware CBST
7. **cal_cbst**: calibration + confusion-aware CBST
8. **full**: all three modules

## 🎯 Success Criteria
- **A方向正向召回提升**: from ~50% to >60%
- **B方向正向召回提升**: from ~47% to >55%
- **C/D/E减少负向→中性混淆**: 提高负向召回5-10个百分点
- **F方向中性召回提升**: from 17% to >30%
- **整体BalAcc提升**: 2-5个百分点

## 🐛 Troubleshooting
```bash
# GPU memory issue
# → scheduler auto-retries after 3min, or manually reduce batch size

# Import error
./scripts/sanity_check_csu.sh  # re-run checks

# Stuck jobs
ps aux | grep train_cbst_confusion_fix
kill <pid>  # scheduler will retry

# Check specific direction performance
grep "Direction.*final metrics" analysis/confusion_fix_results_*/direction_*/run_*.log
```
