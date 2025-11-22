#!/bin/bash
################################################################################
# Backup Daemon for MaskFeat Training
################################################################################
# Purpose: Continuously monitors training and creates periodic backups
# Usage: ./backup_daemon.sh <job_id> [backup_interval]
#
# Features:
# - Monitors SLURM job status
# - Creates backups every N seconds (default: 1800 = 30 minutes)
# - Creates final backup when training completes
# - Handles consecutive backup failures gracefully
#
# Typical usage:
#   nohup ./backup_daemon.sh 123456 > backup_daemon.log 2>&1 &
#
# Author: Musicamatics
# Course: Introduction to Machine Learning (HKU, Fall 2025)
################################################################################

# Parse arguments
JOB_ID=$1
BACKUP_INTERVAL=${2:-1800}  # Default: 30 minutes

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id> [backup_interval_seconds]"
    echo "Example: ./backup_daemon.sh 123456 1800"
    echo ""
    echo "Default backup interval: 1800 seconds (30 minutes)"
    exit 1
fi

# Configuration
LOG_FILE=~/maskfeat_in1k_${JOB_ID}.log
CONSECUTIVE_FAILURES=0
MAX_FAILURES=3

echo "=========================================="
echo "Backup Daemon Started"
echo "Time: $(date)"
echo "Job ID: $JOB_ID"
echo "Log File: $LOG_FILE"
echo "Backup Interval: ${BACKUP_INTERVAL}s ($((BACKUP_INTERVAL / 60)) minutes)"
echo "=========================================="

################################################################################
# Function: Check if training is still running
################################################################################
is_training_running() {
    # Check if SLURM job exists
    if squeue -j $JOB_ID &>/dev/null; then
        # Also verify log file is being updated (within last 10 minutes)
        if [ -f "$LOG_FILE" ]; then
            local log_age=$(( $(date +%s) - $(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0) ))
            if [ $log_age -lt 600 ]; then
                return 0  # Training is running
            fi
        fi
    fi
    return 1  # Training stopped or stalled
}

################################################################################
# Main monitoring loop
################################################################################
while is_training_running; do
    echo ""
    echo "[$(date)] Training is active - creating backup..."
    
    # Run backup script
    if ~/maskfeat_project/scripts/backup/backup_results.sh; then
        echo "[$(date)] ✓ Backup successful"
        CONSECUTIVE_FAILURES=0
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        echo "[$(date)] ✗ Backup failed (consecutive failures: $CONSECUTIVE_FAILURES)"
        
        # Alert if too many failures
        if [ $CONSECUTIVE_FAILURES -ge $MAX_FAILURES ]; then
            echo "[$(date)] ⚠️  WARNING: $MAX_FAILURES consecutive backup failures!"
            echo "[$(date)] Possible issues:"
            echo "  - Disk quota exceeded (check: df -h ~)"
            echo "  - Permission issues"
            echo "  - Source files locked/in use"
            echo "  - Network storage issues"
        fi
    fi
    
    # Wait before next backup
    echo "[$(date)] Next backup in $((BACKUP_INTERVAL / 60)) minutes..."
    sleep $BACKUP_INTERVAL
done

################################################################################
# Training completed or stopped - create final backup
################################################################################
echo ""
echo "=========================================="
echo "[$(date)] Training completed or stopped"
echo "Creating final backup..."

if ~/maskfeat_project/scripts/backup/backup_results.sh; then
    echo "[$(date)] ✓ Final backup successful"
else
    echo "[$(date)] ✗ Final backup failed"
    echo "You may want to manually backup:"
    echo "  ~/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune/"
fi

echo ""
echo "[$(date)] Backup daemon stopped"
echo "Total backups created: Check ~/maskfeat_backup/"
echo "=========================================="

exit 0

