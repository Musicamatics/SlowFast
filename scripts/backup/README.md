# Backup Scripts

## Purpose

These scripts create periodic backups of training checkpoints, logs, and configuration files during long MaskFeat training runs. This protects against data loss from:
- Disk quota exceeded
- Job failures
- Accidental file deletion
- Storage corruption

## Scripts

### `backup_results.sh`
Creates a single timestamped backup of training outputs.
- Can be run manually or called by daemon
- Includes checkpoints, logs, and configs
- Smart cleanup: keeps recent backups, deletes old checkpoint files

### `backup_daemon.sh`
Continuously monitors training and creates periodic backups.
- Runs in background during training
- Creates backups every 30 minutes (configurable)
- Automatically stops when training completes
- Creates final backup after training

## Usage

### Manual Backup

```bash
# Create a one-time backup
./backup_results.sh

# Backup to custom location
./backup_results.sh /path/to/backup/dir /path/to/source
```

### Automated Backups with Daemon

```bash
# Start backup daemon (replace JOB_ID with your SLURM job ID)
nohup ./backup_daemon.sh 123456 > backup_daemon.log 2>&1 &

# With custom backup interval (seconds)
nohup ./backup_daemon.sh 123456 3600 > backup_daemon.log 2>&1 &  # Every hour
```

### Integration with SLURM Job

Add to your SLURM batch script:

```bash
#!/bin/bash
#SBATCH --job-name=maskfeat
#SBATCH --time=60:00:00

# Get job ID
JOB_ID=$SLURM_JOB_ID

# Start backup daemon in background
nohup ~/maskfeat_project/scripts/backup/backup_daemon.sh $JOB_ID \
  > ~/backup_daemon_${JOB_ID}.log 2>&1 &

# Your training command here
python tools/run_net.py --cfg configs/...
```

## Backup Structure

```
~/maskfeat_backup/
├── backup_20251122_140530/
│   ├── checkpoints/
│   │   ├── ssl_eval_checkpoint_epoch_00010.pyth  (983MB)
│   │   ├── ssl_eval_checkpoint_epoch_00020.pyth  (983MB)
│   │   └── ...
│   ├── stdout.log                    # Training log
│   ├── maskfeat_in1k_123456.log     # SLURM stdout
│   ├── maskfeat_in1k_123456.err     # SLURM stderr
│   ├── configs/                      # Configuration files
│   └── backup_info.txt              # Backup metadata
├── backup_20251122_150530/          # Next backup (30 min later)
└── ...
```

## Smart Cleanup

To save disk space, the backup script:
1. **Keeps** the 3 most recent backups fully (with checkpoints)
2. **Deletes** checkpoint files from older backups (but keeps logs)
3. **Preserves** all log files and configurations

Example after cleanup:
```
backup_20251122_160530/  (10GB)  ← Most recent (FULL)
backup_20251122_150530/  (10GB)  ← Recent (FULL)
backup_20251122_140530/  (10GB)  ← Recent (FULL)
backup_20251122_130530/  (50MB)  ← Older (logs only, checkpoints deleted)
backup_20251122_120530/  (50MB)  ← Older (logs only, checkpoints deleted)
```

## Configuration

### Backup Daemon Settings

```bash
# Backup interval (default: 30 minutes)
BACKUP_INTERVAL=1800  # seconds

# Maximum consecutive failures before alert
MAX_FAILURES=3
```

### Backup Script Settings

```bash
# Number of recent backups to keep fully
KEEP_RECENT=3

# Default backup location
BACKUP_DIR="$HOME/maskfeat_backup"

# Default source directory
SOURCE_DIR="$HOME/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune"
```

## Monitoring Backups

### Check backup daemon status

```bash
# View daemon logs
tail -f backup_daemon.log

# Check if daemon is running
ps aux | grep backup_daemon
```

### List all backups

```bash
# List backups with sizes
ls -lh ~/maskfeat_backup/

# Find backups with checkpoints
find ~/maskfeat_backup/ -name "*.pyth" -exec dirname {} \; | sort -u

# Total backup size
du -sh ~/maskfeat_backup/
```

## Restoring from Backup

### Restore checkpoints

```bash
# Restore specific epoch
cp ~/maskfeat_backup/backup_20251122_140530/checkpoints/ssl_eval_checkpoint_epoch_00050.pyth \
   ~/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune/checkpoints/

# Restore all checkpoints from a backup
cp -r ~/maskfeat_backup/backup_20251122_140530/checkpoints/* \
   ~/maskfeat_project/SlowFast/output/maskfeat_in1k_finetune/checkpoints/
```

### Resume training from backup

```bash
# The training script will auto-resume if checkpoints are present
# Just ensure TRAIN.AUTO_RESUME: True in your config

sbatch run_imagenet1k_finetune.sbatch
```

## Disk Space Management

### Check available space

```bash
# Check home directory quota
df -h ~

# Check backup directory size
du -sh ~/maskfeat_backup/
```

### Manual cleanup

```bash
# Delete very old backups (keeps logs only in recent 3)
# This is done automatically, but you can run manually:
cd ~/maskfeat_backup
ls -dt backup_* | tail -n +4 | xargs -I {} rm -rf {}/checkpoints
```

## Troubleshooting

### Backup fails with "Permission denied"

```bash
# Check directory permissions
ls -ld ~/maskfeat_backup/
chmod 755 ~/maskfeat_backup/
```

### "Disk quota exceeded"

```bash
# Check disk usage
df -h ~

# Free up space by removing old checkpoints
rm -rf ~/maskfeat_backup/backup_*/checkpoints/

# Or clean up data directories
rm -rf ~/maskfeat_project/data/imagenet-*-converted/
```

### Consecutive backup failures

Check the daemon log for details:
```bash
tail -20 backup_daemon.log
```

Common causes:
- Disk quota exceeded
- Training files locked/in use
- NFS storage issues
- Permission problems

## Best Practices

1. **Start backup daemon at the same time as training**
2. **Check daemon logs** periodically to ensure backups are working
3. **Monitor disk usage** - checkpoints can be large (983MB each)
4. **Keep at least one full backup** before deleting old ones
5. **Test restoration** before relying on backups for critical work

## Author

Created by Musicamatics for MaskFeat reproduction project.

