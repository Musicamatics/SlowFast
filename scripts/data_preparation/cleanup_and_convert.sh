#!/bin/bash
# ImageNet-1K Cleanup and Streaming Conversion
# This script runs in background and survives SSH disconnection

set -e

LOG_FILE=~/maskfeat_project/conversion_log_$(date +%Y%m%d_%H%M%S).log

echo "================================================" | tee -a $LOG_FILE
echo "ImageNet-1K Cleanup and Streaming Conversion" | tee -a $LOG_FILE
echo "Started at: $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "================================================" | tee -a $LOG_FILE

# Step 1: Clean up HuggingFace cache
echo "" | tee -a $LOG_FILE
echo "Step 1: Cleaning HuggingFace cache..." | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

CACHE_DIR=~/.cache/huggingface/hub/datasets--ILSVRC--imagenet-1k
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" | cut -f1)
    echo "Found cache directory: $CACHE_SIZE" | tee -a $LOG_FILE
    echo "Deleting cache..." | tee -a $LOG_FILE
    rm -rf "$CACHE_DIR"
    echo "✅ Cache deleted!" | tee -a $LOG_FILE
else
    echo "Cache directory not found (already deleted?)" | tee -a $LOG_FILE
fi

# Also clean datasets cache
DATASETS_CACHE=~/.cache/huggingface/datasets
if [ -d "$DATASETS_CACHE" ]; then
    echo "Cleaning datasets cache..." | tee -a $LOG_FILE
    rm -rf "$DATASETS_CACHE"/*
    echo "✅ Datasets cache cleaned!" | tee -a $LOG_FILE
fi

# Check space
echo "" | tee -a $LOG_FILE
echo "Disk space after cleanup:" | tee -a $LOG_FILE
df -h ~ | tee -a $LOG_FILE

# Step 2: Run streaming conversion
echo "" | tee -a $LOG_FILE
echo "Step 2: Starting streaming conversion..." | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE
echo "This will take 2-4 hours" | tee -a $LOG_FILE
echo "Conversion start time: $(date)" | tee -a $LOG_FILE

cd ~/maskfeat_project

# Run Python streaming conversion
python3 convert_imagenet1k_streaming.py 2>&1 | tee -a $LOG_FILE

CONVERSION_EXIT=${PIPESTATUS[0]}

if [ $CONVERSION_EXIT -ne 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "❌ Conversion failed with exit code: $CONVERSION_EXIT" | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "✅ Conversion completed at: $(date)" | tee -a $LOG_FILE

# Step 3: Verify conversion
echo "" | tee -a $LOG_FILE
echo "Step 3: Verifying conversion..." | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

CONVERTED_DIR=~/maskfeat_project/data/imagenet-1k-converted

if [ ! -d "$CONVERTED_DIR" ]; then
    echo "❌ Converted directory not found!" | tee -a $LOG_FILE
    exit 1
fi

TRAIN_CLASSES=$(ls $CONVERTED_DIR/train/ 2>/dev/null | wc -l)
VAL_CLASSES=$(ls $CONVERTED_DIR/validation/ 2>/dev/null | wc -l)

echo "Train classes: $TRAIN_CLASSES" | tee -a $LOG_FILE
echo "Validation classes: $VAL_CLASSES" | tee -a $LOG_FILE

if [ "$TRAIN_CLASSES" -ne 1000 ] || [ "$VAL_CLASSES" -ne 1000 ]; then
    echo "❌ Verification failed!" | tee -a $LOG_FILE
    echo "Expected 1000 classes each, got train=$TRAIN_CLASSES, val=$VAL_CLASSES" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Verification passed!" | tee -a $LOG_FILE

# Step 4: Submit training job
echo "" | tee -a $LOG_FILE
echo "Step 4: Submitting training job..." | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Restore PYTHONPATH for training
export PYTHONPATH=~/detectron2_mock:~/maskfeat_project/SlowFast:$PYTHONPATH

cd ~/maskfeat_project
JOB_OUTPUT=$(sbatch run_imagenet1k_finetune.sbatch 2>&1)
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job!" | tee -a $LOG_FILE
    echo "Output: $JOB_OUTPUT" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Job submitted successfully!" | tee -a $LOG_FILE
echo "Job ID: $JOB_ID" | tee -a $LOG_FILE

# Step 5: Start monitoring
echo "" | tee -a $LOG_FILE
echo "Step 5: Starting monitoring..." | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

nohup ~/maskfeat_project/email_monitor_in1k.sh $JOB_ID > ~/maskfeat_project/email_monitor_in1k_${JOB_ID}.log 2>&1 &
EMAIL_PID=$!

echo "✅ Email monitor started (PID: $EMAIL_PID)" | tee -a $LOG_FILE

# Final summary
echo "" | tee -a $LOG_FILE
echo "================================================" | tee -a $LOG_FILE
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY!" | tee -a $LOG_FILE
echo "================================================" | tee -a $LOG_FILE
echo "Completion time: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Training Job ID: $JOB_ID" | tee -a $LOG_FILE
echo "Training Log: ~/maskfeat_in1k_${JOB_ID}.log" | tee -a $LOG_FILE
echo "Email Monitor: ~/maskfeat_project/email_monitor_in1k_${JOB_ID}.log" | tee -a $LOG_FILE
echo "Conversion Log: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Monitor training:" | tee -a $LOG_FILE
echo "  tail -f ~/maskfeat_in1k_${JOB_ID}.log" | tee -a $LOG_FILE
echo "================================================" | tee -a $LOG_FILE
