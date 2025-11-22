#!/bin/bash
################################################################################
# Email Monitoring Script for MaskFeat Training
################################################################################
# Purpose: Monitors SLURM training jobs and sends periodic email updates
# Usage: ./email_monitor_training.sh <job_id> [email_address]
# 
# Features:
# - Sends startup notification
# - Hourly progress updates with training/validation metrics
# - Completion notification when training finishes
# - Detects premature stops and sends alerts
#
# Author: Musicamatics
# Course: Introduction to Machine Learning (HKU, Fall 2025)
################################################################################

# Parse command line arguments
JOB_ID=$1
EMAIL_ADDRESS=${2:-"your-email@example.com"}  # Default or provide your email

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id> [email_address]"
    echo "Example: ./email_monitor_training.sh 123456 user@example.com"
    echo ""
    echo "If email_address is not provided, will use: your-email@example.com"
    exit 1
fi

# Configuration
LOG_FILE=~/maskfeat_in1k_${JOB_ID}.log
CHECK_INTERVAL=3600  # 1 hour in seconds
LAST_EPOCH=0
START_TIME=$(date +%s)

echo "=========================================="
echo "Starting email monitor for job $JOB_ID"
echo "Email notifications: $EMAIL_ADDRESS"
echo "Monitoring log: $LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s (1 hour)"
echo "=========================================="

################################################################################
# Function: Check if training is still running
################################################################################
is_training_running() {
    # Check if Slurm job is in RUNNING or PENDING state
    if squeue -j $JOB_ID -h 2>/dev/null | grep -q "RUNNING\|PENDING"; then
        return 0
    fi
    
    # Check if log file is being updated (modified in last 5 minutes)
    if [ -f "$LOG_FILE" ]; then
        local file_age=$(($(date +%s) - $(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)))
        if [ $file_age -lt 300 ]; then
            return 0
        fi
    fi
    
    return 1  # Training stopped
}

################################################################################
# Send startup notification
################################################################################
cat << EMAIL | mail -s "🚀 MaskFeat Training Started (Job $JOB_ID)" "$EMAIL_ADDRESS"
Training job started!

Job ID: $JOB_ID
Start Time: $(date)
Log File: $LOG_FILE

Configuration:
- Dataset: ImageNet-1K (1000 classes)
- Epochs: 100
- Model: Vision Transformer Base (ViT-B)
- GPUs: 4× NVIDIA RTX 4090

You will receive hourly progress updates.

Monitor logs live with:
  tail -f $LOG_FILE
EMAIL

echo "[$(date)] Sent startup notification"

################################################################################
# Main monitoring loop
################################################################################
while true; do
    # Check if training is still running
    if ! is_training_running; then
        echo "[$(date)] Training stopped - checking completion status..."
        
        # Extract final epoch and logs
        FINAL_EPOCH=$(grep "train_epoch" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '"epoch":\s*"\K[0-9]+' || echo "0")
        FINAL_LOG=$(tail -50 "$LOG_FILE" 2>/dev/null)
        FINAL_VAL=$(grep "val_epoch" "$LOG_FILE" 2>/dev/null | tail -1)
        
        if [ "$FINAL_EPOCH" -ge 100 ]; then
            # Training completed successfully
            DURATION=$(($(date +%s) - START_TIME))
            HOURS=$((DURATION / 3600))
            MINUTES=$(((DURATION % 3600) / 60))
            
            cat << EMAIL | mail -s "✅ MaskFeat Training Completed! (Job $JOB_ID)" "$EMAIL_ADDRESS"
Training completed successfully!

Job ID: $JOB_ID
Final Epoch: $FINAL_EPOCH/100
Total Duration: ${HOURS}h ${MINUTES}m
Completed at: $(date)

Final Validation Results:
$FINAL_VAL

Checkpoints saved in:
~/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune/checkpoints/

Last 50 lines of log:
$FINAL_LOG

Next steps:
- Evaluate model performance
- Extract training curves
- Compare with paper results
EMAIL
            echo "[$(date)] ✓ Sent completion notification (SUCCESS)"
        else
            # Training stopped prematurely
            cat << EMAIL | mail -s "❌ MaskFeat Training Stopped Early (Job $JOB_ID)" "$EMAIL_ADDRESS"
WARNING: Training stopped before completion!

Job ID: $JOB_ID
Last Completed Epoch: $FINAL_EPOCH/100
Stopped at: $(date)

Please investigate:
- Error log: ~/maskfeat_in1k_${JOB_ID}.err
- Main log: $LOG_FILE
- SLURM job status: scontrol show job $JOB_ID

Last 50 lines of log:
$FINAL_LOG

Common issues:
- Out of memory (OOM)
- Disk quota exceeded
- Job time limit reached
- Node failure
EMAIL
            echo "[$(date)] ✗ Sent alert notification (PREMATURE STOP)"
        fi
        break
    fi
    
    # Check for progress updates
    if [ -f "$LOG_FILE" ]; then
        CURRENT_EPOCH=$(grep "train_epoch" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '"epoch":\s*"\K[0-9]+' || echo "")
        
        # Send update if epoch changed
        if [ ! -z "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" != "$LAST_EPOCH" ]; then
            LAST_EPOCH=$CURRENT_EPOCH
            ELAPSED=$(($(date +%s) - START_TIME))
            HOURS=$((ELAPSED / 3600))
            MINUTES=$(((ELAPSED % 3600) / 60))
            
            # Extract recent logs
            RECENT_TRAIN_EPOCH=$(grep "train_epoch" "$LOG_FILE" 2>/dev/null | tail -1)
            RECENT_TRAIN_ITER=$(grep "train_iter_" "$LOG_FILE" 2>/dev/null | tail -5)
            RECENT_VAL=$(grep "val_epoch" "$LOG_FILE" 2>/dev/null | tail -1)
            
            cat << EMAIL | mail -s "📊 MaskFeat Progress: Epoch $CURRENT_EPOCH/100 (Job $JOB_ID)" "$EMAIL_ADDRESS"
Training Progress Update

Job ID: $JOB_ID
Current Epoch: $CURRENT_EPOCH/100
Elapsed Time: ${HOURS}h ${MINUTES}m

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Epoch Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$RECENT_TRAIN_EPOCH

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recent Training Iterations (last 5):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$RECENT_TRAIN_ITER

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Latest Validation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$RECENT_VAL

NOTE: Use ETA from "train_iter_" logs (not "train_epoch" ETA).
The epoch-level ETA accumulates data loading time and is inaccurate.

Timestamp: $(date)
EMAIL
            echo "[$(date)] Sent progress update for epoch $CURRENT_EPOCH"
        fi
    fi
    
    # Wait before next check
    sleep $CHECK_INTERVAL
done

echo "[$(date)] Email monitoring stopped"
echo "=========================================="

