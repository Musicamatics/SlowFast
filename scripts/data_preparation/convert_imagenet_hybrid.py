import sys
import os
import logging
import gc
import psutil
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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
        logger.info("🛑 SHUTDOWN REQUESTED - Waiting for workers to finish...")
        logger.info("   Press Ctrl+C again to force quit.")
        logger.info("="*70)
    else:
        logger.warning("\n⚠️  FORCE QUIT!")
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# LOGGING SETUP
# ============================================================================
log_file = "conversion_hybrid.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY MONITORING (Lightweight, non-blocking)
# ============================================================================
def get_memory_usage_mb():
    """
    Get current process memory usage in MB.
    Lightweight operation (~1ms), safe to call frequently.
    """
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0


def log_memory_checkpoint(label):
    """
    Log memory at key checkpoints.
    Called only a few times per script run, NOT per-image.
    Zero bottleneck impact.
    """
    mem_mb = get_memory_usage_mb()
    mem_gb = mem_mb / 1024
    logger.info(f"[MEMORY] {label}: {mem_gb:.2f} GB")


logger.info("="*70)
logger.info("ImageNet-1K Conversion (HYBRID: Parallel + Resume + Safe)")
logger.info("Press Ctrl+C to stop gracefully")
logger.info("="*70)

# Memory checkpoint at startup
log_memory_checkpoint("Startup")


# ============================================================================
# Load dataset WITHOUT streaming
# ============================================================================
logger.info("Loading ImageNet-1k from parquet...")
dataset = load_dataset("parquet", data_dir="./data/imagenet-1k/data")

output_dir = Path("./data/imagenet-1k-converted")
logger.info(f"Output directory: {output_dir}")

if not output_dir.exists():
    output_dir.mkdir(exist_ok=True)
    logger.info("Created new output directory")

log_memory_checkpoint("After dataset load")


# ============================================================================
# BUILD SET OF EXISTING FILES (Fast O(1) lookup)
# ============================================================================
def get_existing_indices(split_dir):
    """
    Scan disk and return a set of indices that already exist.
    Much faster than checking each index during iteration.

    Note: This operation is only done ONCE per split, not per-image.
    Very minimal performance impact.
    """
    if not split_dir.exists():
        logger.info(f"{split_dir.name} directory does not exist yet")
        return set()

    logger.info(f"Scanning existing files in {split_dir.name}...")
    existing_indices = set()

    # Glob all JPEG files and extract index from filename
    for jpeg_path in split_dir.glob("*/[0-9]*.JPEG"):
        try:
            # Extract index from filename: "000123.JPEG" -> 123
            idx = int(jpeg_path.stem)
            existing_indices.add(idx)
        except (ValueError, AttributeError):
            continue

    logger.info(f"Found {len(existing_indices):,} existing files")
    return existing_indices


# ============================================================================
# Worker function for multiprocessing (save ONE image)
# ============================================================================
def _save_one_sample(args):
    """
    Worker function for multiprocessing Pool.
    Expected `args` tuple: (sample_idx, sample_dict, split_dir)

    Memory management:
      - Image object is scoped to function
      - Automatically freed when function returns
      - No lingering PIL Image references
      - try/finally ensures cleanup even on error
    """
    idx, sample, split_dir = args

    # Check for global shutdown
    if shutdown_requested:
        return None

    label = sample['label']
    image = sample['image']

    try:
        # class folder
        class_dir = split_dir / f"class_{label:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # file name
        image_path = class_dir / f"{idx:06d}.JPEG"

        # convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # save JPEG (quality=85 for 30-40% space savings vs quality=95)
        image.save(image_path, quality=85, format="JPEG", optimize=True)
        return idx

    except Exception as e:
        logger.error(f"Error saving image {idx}: {e}")
        return None

    finally:
        # Explicit cleanup (ensures memory freed even on error)
        try:
            del image
            del sample
        except:
            pass


# ============================================================================
# Process split with PARALLEL + RESUME + MEMORY SAFETY
# ============================================================================
def save_split_parallel_resumable(split_name, split_data, n_jobs=None):
    """
    Save split using multiprocessing with memory safety.

    KEY OPTIMIZATIONS:
      - Pre-compute indices to process (avoids sequential checking)
      - Pass only needed indices to worker pool
      - Workers process in parallel without existence checks
      - Explicit garbage collection after split completes
      - Memory monitoring at key checkpoints (non-blocking)
    """
    global shutdown_requested

    split_dir = output_dir / split_name
    split_dir.mkdir(exist_ok=True)

    # STEP 1: Get existing indices (ONE scan, not per-image)
    existing_indices = get_existing_indices(split_dir)

    # STEP 2: Pre-compute which indices need processing
    logger.info(f"\nCalculating indices to process for {split_name}...")
    indices_to_process = [
        idx for idx in range(len(split_data))
        if idx not in existing_indices
    ]

    logger.info(f"Processing {split_name} split:")
    logger.info(f"  Total images: {len(split_data):,}")
    logger.info(f"  Already exist: {len(existing_indices):,}")
    logger.info(f"  Need to save: {len(indices_to_process):,}")

    if len(indices_to_process) == 0:
        logger.info(f"✅ {split_name} split is already complete!")
        return True

    if n_jobs is None:
        n_jobs = cpu_count()

    logger.info(f"Starting multiprocessing with {n_jobs} workers...")
    log_memory_checkpoint(f"Before {split_name} processing")

    # STEP 3: Prepare arguments for only the needed indices
    args_iter = (
        (idx, split_data[idx], split_dir)
        for idx in indices_to_process
    )

    # STEP 4: Process with Pool (parallel workers)
    saved = 0
    errors = 0
    progress_interval = 50000  # Log every 50k images (not per-image)

    with Pool(processes=n_jobs) as pool:
        for result in tqdm(
            pool.imap_unordered(_save_one_sample, args_iter, chunksize=10),
            total=len(indices_to_process),
            desc=f"Saving {split_name}",
            unit="img",
            ncols=100
        ):
            if shutdown_requested:
                logger.info(f"\n🛑 Stopping {split_name} conversion gracefully...")
                pool.terminate()
                pool.join()
                break

            if result is not None:
                saved += 1
                # Log progress only every 50k images (not per-image!)
                if (saved + len(existing_indices)) % progress_interval == 0:
                    logger.info(f"Progress: {saved + len(existing_indices):,} total saved")
            else:
                errors += 1

    logger.info(f"\n{split_name} conversion complete:")
    logger.info(f"  - Previously existed: {len(existing_indices):,}")
    logger.info(f"  - Newly saved: {saved:,}")
    logger.info(f"  - Errors: {errors:,}")
    logger.info(f"  - Total completed: {len(existing_indices) + saved:,}/{len(split_data):,}")

    # SAFEGUARD 1: Explicit garbage collection after split
    # This is fast (~50-200ms) and only called 2x per script run
    # Zero bottleneck impact on 60+ minute operation
    logger.info(f"Running garbage collection for {split_name}...")
    gc.collect()
    log_memory_checkpoint(f"After {split_name} cleanup")

    if shutdown_requested:
        logger.info(f"⚠️  {split_name} conversion INTERRUPTED")
        return False

    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        logger.info("\n" + "="*70)
        logger.info("STARTING CONVERSIONS (Hybrid + Memory Safe)")
        logger.info("="*70)

        n_workers = os.cpu_count()
        logger.info(f"Using {n_workers} CPU cores for parallelization")

        # Process train split
        logger.info("\n📊 Processing TRAIN split...")
        train_success = save_split_parallel_resumable(
            "train", 
            dataset['train'], 
            n_jobs=n_workers
        )

        # Only process validation if train completed and no shutdown
        if train_success and not shutdown_requested:
            logger.info("\n📊 Processing VALIDATION split...")
            save_split_parallel_resumable(
                "val", 
                dataset['validation'], 
                n_jobs=n_workers
            )

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

        log_memory_checkpoint("Final")

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
