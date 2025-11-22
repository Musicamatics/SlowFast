#!/bin/bash
# ImageNet-1K Complete Pipeline Script
# This script will:
# 1. Wait for download to complete
# 2. Convert dataset
# 3. Verify conversion
# 4. Submit training job
# 5. Start monitoring and backup

set -e  # Exit on any error

echo "========================================"
echo "ImageNet-1K Complete Pipeline"
echo "Started at: $(date)"
echo "========================================"

# Configuration
DATA_DIR=~/maskfeat_project/data/imagenet-1k/data
CONVERTED_DIR=~/maskfeat_project/data/imagenet-1k-converted
EXPECTED_PARQUET_FILES=334
EXPECTED_CLASSES=1000

# Step 0: Wait for download to complete
echo ""
echo "Step 0: Checking if download is complete..."
echo "----------------------------------------"

while true; do
    if [ -d "$DATA_DIR" ]; then
        CURRENT_FILES=$(ls $DATA_DIR/*.parquet 2>/dev/null | wc -l)
        echo "Current parquet files: $CURRENT_FILES / $EXPECTED_PARQUET_FILES"
        
        if [ "$CURRENT_FILES" -ge "$EXPECTED_PARQUET_FILES" ]; then
            echo "✅ Download appears complete!"
            sleep 5  # Wait a bit more to ensure files are fully written
            break
        else
            echo "⏳ Still downloading... (checking again in 30 seconds)"
            sleep 30
        fi
    else
        echo "❌ Data directory not found: $DATA_DIR"
        exit 1
    fi
done

# Step 1: Convert dataset
echo ""
echo "Step 1: Converting ImageNet-1K dataset..."
echo "----------------------------------------"
echo "This will take 2-4 hours for 1.3M images"
echo "Start time: $(date)"

cd ~/maskfeat_project

# IMPORTANT: Temporarily unset PYTHONPATH for conversion
# (datasets library conflicts with SlowFast)
ORIGINAL_PYTHONPATH="$PYTHONPATH"
unset PYTHONPATH

# Remove SlowFast from Python path temporarily
python << 'EOF'
import sys
import os

# Remove SlowFast paths
sys.path = [p for p in sys.path if 'SlowFast' not in p]

# Now run conversion
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm

print("Loading ImageNet-1K from parquet...")
dataset = load_dataset("parquet", data_dir="./data/imagenet-1k/data")

output_dir = Path("./data/imagenet-1k-converted")
if output_dir.exists():
    print("Output directory exists, cleaning up...")
    import shutil
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

def save_split(split_name, split_data):
    split_dir = output_dir / split_name
    split_dir.mkdir(exist_ok=True)
    
    print(f"\nConverting {split_name} split ({len(split_data):,} images)...")
    for idx, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
        label = sample['label']
        image = sample['image']
        
        # Create class directory
        class_dir = split_dir / f"class_{label:03d}"
        class_dir.mkdir(exist_ok=True)
        
        # Save image as JPEG
        image_path = class_dir / f"{idx:06d}.JPEG"
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(image_path, quality=95, format='JPEG')

# Convert both splits
save_split("train", dataset['train'])
save_split("validation", dataset['validation'])

print("\n✅ Conversion complete!")
EOF

CONVERSION_EXIT=$?

# Restore PYTHONPATH
export PYTHONPATH="$ORIGINAL_PYTHONPATH"

if [ $CONVERSION_EXIT -ne 0 ]; then
    echo "❌ Conversion failed!"
    exit 1
fi

echo "✅ Conversion completed at: $(date)"

# Step 2: Verify conversion
echo ""
echo "Step 2: Verifying conversion..."
echo "----------------------------------------"

if [ ! -d "$CONVERTED_DIR" ]; then
    echo "❌ Converted directory not found: $CONVERTED_DIR"
    exit 1
fi

TRAIN_CLASSES=$(ls $CONVERTED_DIR/train/ 2>/dev/null | wc -l)
VAL_CLASSES=$(ls $CONVERTED_DIR/validation/ 2>/dev/null | wc -l)

echo "Train classes: $TRAIN_CLASSES"
echo "Validation classes: $VAL_CLASSES"

if [ "$TRAIN_CLASSES" -ne "$EXPECTED_CLASSES" ]; then
    echo "❌ ERROR: Expected $EXPECTED_CLASSES train classes, found $TRAIN_CLASSES"
    exit 1
fi

if [ "$VAL_CLASSES" -ne "$EXPECTED_CLASSES" ]; then
    echo "❌ ERROR: Expected $EXPECTED_CLASSES validation classes, found $VAL_CLASSES"
    exit 1
fi

# Check sample class has images
SAMPLE_CLASS=$(ls $CONVERTED_DIR/train/ | head -1)
SAMPLE_IMAGES=$(ls $CONVERTED_DIR/train/$SAMPLE_CLASS/*.JPEG 2>/dev/null | wc -l)
echo "Sample class ($SAMPLE_CLASS) has $SAMPLE_IMAGES images"

if [ "$SAMPLE_IMAGES" -eq 0 ]; then
    echo "❌ ERROR: Sample class has no images!"
    exit 1
fi

echo "✅ Verification passed!"

# Step 3: Restore environment for training
echo ""
echo "Step 3: Setting up environment for training..."
echo "----------------------------------------"

# Set PYTHONPATH for training (includes detectron2 mock)
export PYTHONPATH=~/detectron2_mock:~/maskfeat_project/SlowFast:$PYTHONPATH

echo "PYTHONPATH restored: $PYTHONPATH"

# Step 4: Submit training job
echo ""
echo "Step 4: Submitting training job..."
echo "----------------------------------------"

cd ~/maskfeat_project
JOB_OUTPUT=$(sbatch run_imagenet1k_finetune.sbatch)
JOB_ID=$(echo $JOB_OUTPUT | grep -oP '\d+')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job!"
    exit 1
fi

echo "✅ Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo "Job details:"
squeue -j $JOB_ID

# Wait a moment for job to start
sleep 5

# Step 5: Start email monitoring
echo ""
echo "Step 5: Starting email monitoring..."
echo "----------------------------------------"

nohup ~/maskfeat_project/email_monitor_in1k.sh $JOB_ID > ~/maskfeat_project/email_monitor_in1k_${JOB_ID}.log 2>&1 &
EMAIL_PID=$!
echo "✅ Email monitor started (PID: $EMAIL_PID)"
echo "Email log: ~/maskfeat_project/email_monitor_in1k_${JOB_ID}.log"

# Step 6: Backup daemon is started by the sbatch script automatically
echo ""
echo "Step 6: Backup daemon..."
echo "----------------------------------------"
echo "✅ Backup daemon will be started automatically by training job"
echo "Backups will run every 30 minutes"
echo "Backup location: ~/maskfeat_backup_in1k/"

# Final summary
echo ""
echo "========================================"
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================"
echo "Job ID: $JOB_ID"
echo "Started at: $(date)"
echo ""
echo "📊 Training Status:"
echo "  Log file: ~/maskfeat_in1k_${JOB_ID}.log"
echo "  Error log: ~/maskfeat_in1k_${JOB_ID}.err"
echo ""
echo "📧 Email Monitoring:"
echo "  Status: Active (PID: $EMAIL_PID)"
echo "  Log: ~/maskfeat_project/email_monitor_in1k_${JOB_ID}.log"
echo ""
echo "💾 Backup:"
echo "  Location: ~/maskfeat_backup_in1k/"
echo "  Frequency: Every 30 minutes"
echo ""
echo "🔍 Monitor progress:"
echo "  tail -f ~/maskfeat_in1k_${JOB_ID}.log"
echo "  squeue -j $JOB_ID"
echo ""
echo "⏱️ Expected completion: ~15-20 hours"
echo "🎯 Target accuracy: ~84.0%"
echo "========================================"
