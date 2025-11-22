import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import signal

# CRITICAL FIX: Remove PYTHONPATH temporarily to avoid SlowFast conflicts
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# Remove SlowFast from sys.path if present
sys.path = [p for p in sys.path if 'SlowFast' not in p]

# Now safe to import HuggingFace datasets
from datasets import load_dataset

# ============================================================================
# GRACEFUL SHUTDOWN HANDLER
# ============================================================================
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        logger.info("\n" + "="*70)
        logger.info("🛑 SHUTDOWN REQUESTED - Finishing current image...")
        logger.info("   Press Ctrl+C again to force quit.")
        logger.info("="*70)
    else:
        logger.warning("\n⚠️  FORCE QUIT!")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# LOGGING SETUP - Logs to both console and file
# ============================================================================
log_file = "conversion.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("Starting ImageNet-1K conversion with RESUME capability")
logger.info("Press Ctrl+C to stop gracefully")
logger.info("="*70)

# ============================================================================
# Load dataset WITHOUT streaming (faster for skipping)
# ============================================================================
logger.info("Loading ImageNet-1k from parquet...")
logger.info("This loads metadata into memory (~2-5GB RAM) but enables fast skipping")
dataset = load_dataset("parquet", data_dir="./data/imagenet-1k/data")

output_dir = Path("./data/imagenet-1k-converted")
logger.info(f"Output directory: {output_dir}")

if not output_dir.exists():
    output_dir.mkdir(exist_ok=True)
    logger.info("Created new output directory")
else:
    logger.info("Output directory already exists - will resume from existing files")

# ============================================================================
# Build existing files set ONCE (fast lookup)
# ============================================================================
def get_existing_files(split_dir):
    """Build a set of existing files for O(1) lookup"""
    if not split_dir.exists():
        return set()
    logger.info(f"Scanning existing files in {split_dir.name}...")
    existing = set(split_dir.glob("*/[0-9]*.JPEG"))
    logger.info(f"Found {len(existing):,} existing files")
    return existing

# ============================================================================
# Process images with resume capability
# ============================================================================
def save_split_with_resume(split_name, split_data):
    """Save split with resume capability - skips existing files"""
    global shutdown_requested
    
    split_dir = output_dir / split_name
    split_dir.mkdir(exist_ok=True)
    
    # Get existing files set (FAST O(1) lookup)
    existing_files = get_existing_files(split_dir)
    
    logger.info(f"\nConverting {split_name} split ({len(split_data):,} images)...")
    logger.info(f"Already converted: {len(existing_files):,} images")
    logger.info(f"Remaining to convert: {len(split_data) - len(existing_files):,} images")
    
    skipped = 0
    saved = 0
    errors = 0
    
    # Use tqdm but disable it in log files
    for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}", 
                                       disable=None, ncols=100)):
        # Check for shutdown request
        if shutdown_requested:
            logger.info(f"\n🛑 Stopping {split_name} conversion gracefully...")
            break
        
        label = sample['label']
        image = sample['image']
        
        # Create class directory
        class_dir = split_dir / f"class_{label:03d}"
        class_dir.mkdir(exist_ok=True)
        
        # Save image as JPEG
        image_path = class_dir / f"{idx:06d}.JPEG"
        
        # Skip if file already exists (fast O(1) lookup with set)
        if image_path in existing_files:
            skipped += 1
            continue
        
        # Log when we start saving (first new image after skips)
        if saved == 0 and skipped > 0:
            logger.info(f"✅ Skipped {skipped:,} existing files, now saving new images starting at index {idx:,}")
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        try:
            # Save with quality=85 (saves 30-40% space vs quality=95)
            image.save(image_path, quality=85, format='JPEG', optimize=True)
            saved += 1
            
            # Log progress every 10,000 saved images
            if saved % 10000 == 0:
                logger.info(f"Progress: {saved:,} new images saved | {skipped:,} skipped | {errors:,} errors")
        except Exception as e:
            logger.error(f"Error saving {image_path}: {e}")
            errors += 1
    
    logger.info(f"\n{split_name} conversion complete:")
    logger.info(f"  - Skipped (already existed): {skipped:,}")
    logger.info(f"  - Newly saved: {saved:,}")
    logger.info(f"  - Errors: {errors:,}")
    logger.info(f"  - Total completed: {len(existing_files) + saved:,}/{len(split_data):,}")
    
    if shutdown_requested:
        logger.info(f"⚠️  {split_name} conversion INTERRUPTED - safe to resume later")
        return False
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================
try:
    logger.info("\n" + "="*70)
    logger.info("STARTING CONVERSIONS")
    logger.info("="*70)
    
    # Process train split
    logger.info("\n📊 Processing TRAIN split...")
    train_success = save_split_with_resume("train", dataset['train'])
    
    # Only process validation if train completed and no shutdown
    if train_success and not shutdown_requested:
        logger.info("\n📊 Processing VALIDATION split...")
        save_split_with_resume("val", dataset['validation'])
    
    if not shutdown_requested:
        logger.info("\n" + "="*70)
        logger.info("✅ CONVERSION COMPLETE!")
        logger.info("="*70)
    else:
        logger.info("\n" + "="*70)
        logger.info("⚠️  CONVERSION STOPPED GRACEFULLY")
        logger.info("="*70)
        logger.info("All completed images have been saved.")
        logger.info("Re-run this script to continue from where you left off.")
    
    logger.info(f"\nDataset structure:")
    logger.info(f"  {output_dir}/")
    logger.info(f"    train/ - training images in 1000 classes")
    logger.info(f"    val/   - validation images in 1000 classes")
    logger.info(f"\nReady for training with SlowFast!")

except KeyboardInterrupt:
    logger.info("\n" + "="*70)
    logger.info("⚠️  INTERRUPTED BY USER")
    logger.info("="*70)
    logger.info("Conversion stopped. All completed images are saved.")
    logger.info("Re-run this script to resume.")
    sys.exit(0)

except Exception as e:
    logger.error(f"\n❌ ERROR during conversion: {e}", exc_info=True)
    logger.info("\nYou can re-run this script to resume from where it stopped.")
    sys.exit(1)