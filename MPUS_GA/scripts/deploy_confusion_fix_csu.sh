#!/bin/bash
# Deploy confusion fix experiment to CSU servers

set -e

PROJECT_ROOT="/data/linzhou/MPUS_GA"
BRANCH="r4_confusion_fix_20261001"

echo "=== Confusion Fix Deployment to CSU ==="
echo "Branch: $BRANCH"
echo "Project root: $PROJECT_ROOT"
echo

# Push branch to remote
echo "1. Pushing branch to remote..."
git push mpusga "$BRANCH"

echo
echo "2. SSH to CSU and pull changes..."
ssh csu << 'ENDSSH'
cd /data/linzhou/MPUS_GA
git fetch origin
git checkout r4_confusion_fix_20261001
git pull origin r4_confusion_fix_20261001

echo "Branch updated successfully"
git log --oneline -3
ENDSSH

echo
echo "3. Launch scheduler on CSU..."
ssh csu << 'ENDSSH2'
cd /data/linzhou/MPUS_GA

# Run sanity check
echo "Running pre-launch checks..."
./scripts/sanity_check_csu.sh
if [ $? -ne 0 ]; then
    echo "Sanity check failed. Aborting."
    exit 1
fi

# Launch scheduler with nohup
echo
echo "Launching scheduler..."
nohup python scripts/master_worker_scheduler.py > scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler started with PID: $SCHEDULER_PID"

# Wait a bit and check if it's running
sleep 2
if ps -p $SCHEDULER_PID > /dev/null; then
    echo "✓ Scheduler is running"
    echo "Log file: scheduler.log"
    echo
    echo "First few log lines:"
    head -20 scheduler.log
else
    echo "✗ Scheduler failed to start"
    cat scheduler.log
    exit 1
fi
ENDSSH2

echo
echo "=== Deployment Complete ==="
echo "Scheduler is running on CSU"
echo
echo "Monitor with:"
echo "  ssh csu 'tail -f /data/linzhou/MPUS_GA/scheduler.log'"
echo
echo "Check progress:"
echo "  ssh csu 'cd /data/linzhou/MPUS_GA && python scripts/summarize_confusion_fix.py'"
