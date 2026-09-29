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
git push origin "$BRANCH"

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
echo "3. Ready to launch experiments on CSU"
echo
echo "Next steps:"
echo "  ssh csu"
echo "  cd /data/linzhou/MPUS_GA"
echo "  nohup python scripts/master_worker_scheduler.py > scheduler.log 2>&1 &"
echo "  tail -f scheduler.log"
