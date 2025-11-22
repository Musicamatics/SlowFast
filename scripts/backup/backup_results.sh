#!/bin/bash
################################################################################
# Backup Script for MaskFeat Training Results
################################################################################
# Purpose: Creates timestamped backups of training checkpoints and logs
# Usage: ./backup_results.sh [backup_dir] [source_dir]
#
# Features:
# - Backs up checkpoints, logs, and configuration files
# - Creates timestamped backup directories
# - Keeps only recent backups (deletes old ones to save space)
# - Can be run manually or via backup_daemon.sh
#
# Author: Musicamatics
# Course: Introduction to Machine Learning (HKU, Fall 2025)
################################################################################

# Default directories (can be overridden by command line arguments)
BACKUP_DIR="${1:-$HOME/maskfeat_backup}"
SOURCE_DIR="${2:-$HOME/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune}"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_SUBDIR="$BACKUP_DIR/backup_$TIMESTAMP"
KEEP_RECENT=3  # Keep only the 3 most recent backups

echo "=========================================="
echo "MaskFeat Backup Script"
echo "Time: $(date)"
echo "=========================================="

################################################################################
# Check if source directory exists
################################################################################
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR"
    echo "Training might not have started yet."
    exit 1
fi

################################################################################
# Create backup directory
################################################################################
echo "Creating backup directory: $BACKUP_SUBDIR"
mkdir -p "$BACKUP_SUBDIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create backup directory"
    exit 1
fi

################################################################################
# Backup checkpoints (if they exist)
################################################################################
if [ -d "$SOURCE_DIR/checkpoints" ]; then
    CHECKPOINT_COUNT=$(ls -1 "$SOURCE_DIR/checkpoints"/*.pyth 2>/dev/null | wc -l)
    
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "Backing up $CHECKPOINT_COUNT checkpoint(s)..."
        mkdir -p "$BACKUP_SUBDIR/checkpoints"
        
        # Copy all checkpoint files
        cp -v "$SOURCE_DIR/checkpoints"/*.pyth "$BACKUP_SUBDIR/checkpoints/" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "✓ Checkpoints backed up successfully"
        else
            echo "✗ Warning: Some checkpoints may not have been backed up"
        fi
    else
        echo "⚠ No checkpoints found to backup"
    fi
else
    echo "⚠ Checkpoint directory not found: $SOURCE_DIR/checkpoints"
fi

################################################################################
# Backup logs
################################################################################
echo "Backing up logs..."
if [ -f "$SOURCE_DIR/stdout.log" ]; then
    cp -v "$SOURCE_DIR/stdout.log" "$BACKUP_SUBDIR/" 2>/dev/null
    echo "✓ Training log backed up"
else
    echo "⚠ Training log not found"
fi

# Backup SLURM logs from home directory (if they exist)
SLURM_LOGS=$(ls -1 ~/maskfeat_in1k*.log ~/maskfeat_in1k*.err 2>/dev/null)
if [ ! -z "$SLURM_LOGS" ]; then
    echo "Backing up SLURM logs..."
    for log in $SLURM_LOGS; do
        cp -v "$log" "$BACKUP_SUBDIR/" 2>/dev/null
    done
    echo "✓ SLURM logs backed up"
fi

################################################################################
# Backup configuration files
################################################################################
echo "Backing up configuration files..."
CONFIG_DIR="$HOME/maskfeat_project/SlowFast/configs"
if [ -d "$CONFIG_DIR" ]; then
    mkdir -p "$BACKUP_SUBDIR/configs"
    cp -rv "$CONFIG_DIR/masked_ssl" "$BACKUP_SUBDIR/configs/" 2>/dev/null
    echo "✓ Configuration files backed up"
fi

################################################################################
# Create backup summary
################################################################################
cat > "$BACKUP_SUBDIR/backup_info.txt" << EOF
MaskFeat Training Backup
========================
Backup Time: $(date)
Source: $SOURCE_DIR
Backup: $BACKUP_SUBDIR

Contents:
$(du -sh "$BACKUP_SUBDIR" 2>/dev/null)

Files backed up:
$(find "$BACKUP_SUBDIR" -type f | wc -l) files

Checkpoint files:
$(ls -1 "$BACKUP_SUBDIR/checkpoints"/*.pyth 2>/dev/null | wc -l) checkpoints
EOF

echo "✓ Backup summary created"

################################################################################
# Clean up old backups (keep only recent ones)
################################################################################
echo ""
echo "Cleaning up old backups (keeping $KEEP_RECENT most recent)..."
BACKUP_COUNT=$(ls -1d "$BACKUP_DIR"/backup_* 2>/dev/null | wc -l)

if [ $BACKUP_COUNT -gt $KEEP_RECENT ]; then
    BACKUPS_TO_DELETE=$((BACKUP_COUNT - KEEP_RECENT))
    echo "Found $BACKUP_COUNT backups, will delete $BACKUPS_TO_DELETE oldest"
    
    # Delete oldest backups, but keep logs
    ls -1dt "$BACKUP_DIR"/backup_* | tail -n +$((KEEP_RECENT + 1)) | while read old_backup; do
        echo "Processing old backup: $old_backup"
        
        # Keep logs but delete large checkpoint files
        if [ -d "$old_backup/checkpoints" ]; then
            echo "  Removing checkpoints from: $old_backup"
            rm -rf "$old_backup/checkpoints"
        fi
        
        # If backup is now empty except logs, we can optionally remove it entirely
        # Uncomment the line below to completely remove old backups
        # rm -rf "$old_backup"
    done
    
    echo "✓ Cleanup completed"
else
    echo "Only $BACKUP_COUNT backups exist, no cleanup needed"
fi

################################################################################
# Display backup summary
################################################################################
echo ""
echo "=========================================="
echo "Backup completed successfully!"
echo "Location: $BACKUP_SUBDIR"
echo "Size: $(du -sh "$BACKUP_SUBDIR" 2>/dev/null | cut -f1)"
echo ""
echo "To restore checkpoints:"
echo "  cp $BACKUP_SUBDIR/checkpoints/*.pyth \\"
echo "     $SOURCE_DIR/checkpoints/"
echo "=========================================="

exit 0

