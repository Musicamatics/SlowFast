# Email Monitoring Scripts

## Purpose

These scripts monitor long-running MaskFeat training jobs and send email notifications with progress updates. This is especially useful for multi-day training runs where you want to track progress remotely.

## Scripts

### `email_monitor_training.sh`
Standard monitoring script for batch size 128/256 experiments.
- **Update frequency**: Every 1 hour
- **Use case**: Standard training runs

### `email_monitor_training_bs512.sh`
Monitoring script for batch size 512 experiments (faster convergence).
- **Update frequency**: Every 30 minutes
- **Use case**: Batch size 512 experiments where epochs complete faster

## Usage

### Basic Usage

```bash
# Start monitoring (replace JOB_ID with your SLURM job ID)
./email_monitor_training.sh 123456 your-email@example.com
```

### Run in Background (Recommended)

```bash
# Start in background with nohup
nohup ./email_monitor_training.sh 123456 your-email@example.com \
  > monitor.log 2>&1 &
  
# Check monitoring status
tail -f monitor.log
```

### Integration with SLURM Job

Add to your SLURM batch script:

```bash
#!/bin/bash
#SBATCH --job-name=maskfeat
#SBATCH --time=60:00:00

# Get job ID
JOB_ID=$SLURM_JOB_ID

# Start email monitoring in background
nohup ~/maskfeat_project/scripts/monitoring/email_monitor_training.sh \
  $JOB_ID your-email@example.com \
  > ~/email_monitor_${JOB_ID}.log 2>&1 &

# Your training command here
python tools/run_net.py --cfg configs/...
```

## Email Notifications

You will receive emails for:

1. **Startup**: Training job started confirmation
2. **Progress**: Hourly (or half-hourly) updates with:
   - Current epoch
   - Training/validation metrics
   - Recent iteration logs
   - ETA
3. **Completion**: Final results when training completes
4. **Alerts**: If training stops prematurely

## Configuration

Edit the script to customize:

```bash
# Email address (or pass as second argument)
EMAIL_ADDRESS="your-email@example.com"

# Check interval (seconds)
CHECK_INTERVAL=3600  # 1 hour

# Log file location
LOG_FILE=~/maskfeat_in1k_${JOB_ID}.log
```

## Requirements

- Mail command (`mail` or `mailx`) configured on your system
- Access to SLURM queue (`squeue` command)
- Training logs written to `~/maskfeat_in1k_<job_id>.log`

## Troubleshooting

### Not Receiving Emails

1. **Check mail configuration**:
   ```bash
   echo "Test" | mail -s "Test" your-email@example.com
   ```

2. **Check spam folder**: Automated emails may be filtered

3. **Verify monitoring is running**:
   ```bash
   ps aux | grep email_monitor
   ```

### Script Stops Early

- Check monitoring log file: `tail -f monitor.log`
- Ensure SLURM job is visible: `squeue -j <job_id>`
- Verify log file permissions and location

## Example Email Output

### Progress Email

```
Subject: 📊 MaskFeat Progress: Epoch 50/100 (Job 123456)

Training Progress Update

Job ID: 123456
Current Epoch: 50/100
Elapsed Time: 8h 15m

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Epoch Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"_type": "train_epoch", "epoch": "50/100", "loss": 3.874, 
 "lr": 0.00217, "top1_err": 27.45, "gpu_mem": "14.39G"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Latest Validation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"_type": "val_epoch", "epoch": "50/100", "top1_err": 30.30, 
 "top5_err": 10.03}

Timestamp: Sat Nov 22 14:30:00 HKT 2025
```

## Author

Created by Musicamatics for MaskFeat reproduction project.

