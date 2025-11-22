#!/bin/bash
################################################################################
# Email Monitoring Script for MaskFeat Training (Batch Size 512)
################################################################################
# Purpose: Monitors SLURM training jobs and sends periodic email updates
#          This version is for batch size 512 experiments with more frequent updates
# Usage: ./email_monitor_training_bs512.sh <job_id> [email_address]
# 
# Features:
# - Sends startup notification
# - Updates every 30 minutes (faster than standard version)
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
    echo "Example: ./email_monitor_training_bs512.sh 123456 user@example.com"
    echo ""
    echo "If email_address is not provided, will use: your-email@example.com"
    exit 1
fi

# Configuration
LOG_FILE=~/maskfeat_in1k_bs512_${JOB_ID}.log
CHECK_INTERVAL=1800  # 30 minutes (faster updates for BS512 experiments)
LAST_EPOCH=0
START_TIME=$(date +%s)

echo "=========================================="
echo "Starting email monitor for job $JOB_ID (Batch Size 512)"
echo "Email notifications: $EMAIL_ADDRESS"
echo "Monitoring log: $LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s (30 minutes)"
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
cat << EMAIL | mail -s "🚀 MaskFeat Training Started - BS512 (Job $JOB_ID)" "$EMAIL_ADDRESS"
Training job started!

Job ID: $JOB_ID
Start Time: $(date)
Log File: $LOG_FILE

Configuration:
- Dataset: ImageNet-1K (1000 classes)
- Batch Size: 512 (128 per GPU × 4 GPUs)
- Learning Rate: 0.004 (scaled)
- Warmup: 5 epochs
- Total Epochs: 100
- Mixed Precision: FP16
- Model: Vision Transformer Base (ViT-B)
- GPUs: 4× NVIDIA RTX 4090

Expected Results:
- Training Time: ~16 hours
- Target Accuracy: ~82-83% (vs paper's 84% with BS2048)

You will receive updates every 30 minutes.

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
        
        # Extract accuracy from final validation
        FINAL_ACC=$(echo "$FINAL_VAL" | grep -oP '"top1_err":\s*\K[0-9.]+' | awk '{print 100-$1}')
        
        if [ "$FINAL_EPOCH" -ge 100 ]; then
            # Training completed successfully
            DURATION=$(($(date +%s) - START_TIME))
            HOURS=$((DURATION / 3600))
            MINUTES=$(((DURATION % 3600) / 60))
            
            cat << EMAIL | mail -s "✅ MaskFeat BS512 Training Completed! (Job $JOB_ID)" "$EMAIL_ADDRESS"
Training completed successfully!

Job ID: $JOB_ID
Configuration: Batch Size 512, Warmup 5 epochs
Final Epoch: $FINAL_EPOCH/100
Total Duration: ${HOURS}h ${MINUTES}m
Completed at: $(date)

Final Validation Results:
$FINAL_VAL

Final Top-1 Accuracy: ${FINAL_ACC}% (estimated)

Checkpoints saved in:
~/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune/checkpoints/

Comparison:
- Your result (BS512): ${FINAL_ACC}%
- Paper result (BS2048): 84.0%
- Gap: Explained by batch size difference

Next steps:
- Compare with baseline (BS128) results
- Analyze learning rate schedule impact
- Extract training curves for presentation
EMAIL
            echo "[$(date)] ✓ Sent completion notification (SUCCESS)"
        else
            # Training stopped prematurely
            cat << EMAIL | mail -s "❌ MaskFeat BS512 Training Stopped Early (Job $JOB_ID)" "$EMAIL_ADDRESS"
WARNING: Training stopped before completion!

Job ID: $JOB_ID
Configuration: Batch Size 512
Last Completed Epoch: $FINAL_EPOCH/100
Stopped at: $(date)

Please investigate:
- Error log: ~/maskfeat_in1k_bs512_${JOB_ID}.err
- Main log: $LOG_FILE
- SLURM job status: scontrol show job $JOB_ID

Common issues with BS512:
- Out of memory (check FP16 is enabled)
- GPU memory fragmentation
- Disk quota exceeded
- Job time limit reached

Last 50 lines of log:
$FINAL_LOG
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
            
            # Extract GPU memory usage
            GPU_MEM=$(echo "$RECENT_TRAIN_EPOCH" | grep -oP '"gpu_mem":\s*"\K[^"]+')
            
            cat << EMAIL | mail -s "📊 MaskFeat BS512: Epoch $CURRENT_EPOCH/100 (Job $JOB_ID)" "$EMAIL_ADDRESS"
Training Progress Update (Batch Size 512)

Job ID: $JOB_ID
Current Epoch: $CURRENT_EPOCH/100
Elapsed Time: ${HOURS}h ${MINUTES}m
GPU Memory: $GPU_MEM per GPU

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

Progress: [$(printf '=%.0s' $(seq 1 $((CURRENT_EPOCH/2))))>$(printf ' %.0s' $(seq 1 $((50-CURRENT_EPOCH/2))))] $CURRENT_EPOCH%

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

