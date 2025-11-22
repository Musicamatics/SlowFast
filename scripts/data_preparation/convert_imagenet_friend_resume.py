import sys
import os

# CRITICAL FIX: Remove PYTHONPATH temporarily to avoid SlowFast conflicts
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# Remove SlowFast from sys.path if present
sys.path = [p for p in sys.path if 'SlowFast' not in p]

# Now safe to import HuggingFace datasets
# ══════════════════════════════════════════════════════════════════════════════
#  ImageNet-1k parquet → folder structure (multicore version WITH RESUME)
# ══════════════════════════════════════════════════════════════════════════════

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion_friend_resume.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load the dataset (metadata-only, low memory)
# ──────────────────────────────────────────────────────────────────────────────
logger.info("Loading ImageNet-1k from parquet...")
dataset = load_dataset("parquet", data_dir="./data/imagenet-1k/data")
logger.info("Checkpoint! Dataset loaded.")

output_dir = Path("./data/imagenet-1k-converted")
output_dir.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Get existing indices (for resume capability)
# ──────────────────────────────────────────────────────────────────────────────
def get_existing_indices(split_dir):
    """
    Scan disk and return set of indices that already exist.
    Used for resume capability.
    """
    if not split_dir.exists():
        logger.info(f"{split_dir.name} directory does not exist yet")
        return set()

    logger.info(f"Scanning existing files in {split_dir.name}...")
    existing_indices = set()

    for jpeg_path in split_dir.glob("*/[0-9]*.JPEG"):
        try:
            idx = int(jpeg_path.stem)
            existing_indices.add(idx)
        except (ValueError, AttributeError):
            continue

    logger.info(f"Found {len(existing_indices):,} existing files")
    return existing_indices


# ──────────────────────────────────────────────────────────────────────────────
# Worker initializer: Reload dataset once per worker process
# This ensures each worker has its own lazy-loaded dataset instance
# ──────────────────────────────────────────────────────────────────────────────
def init_worker():
    global global_dataset
    global_dataset = load_dataset("parquet", data_dir="./data/imagenet-1k/data")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helper that writes **one** sample to disk (with error handling)
# ──────────────────────────────────────────────────────────────────────────────
def save_one_sample(args):
    """
    Expected `args` tuple:
        (sample_idx, split_name, split_dir)

    Load sample in worker (in init_worker's dataset).
    """
    idx, split_name, split_dir = args

    try:
        sample = global_dataset[split_name][idx]  # Load here, in the worker
        label = sample["label"]
        image = sample["image"]

        # class folder
        class_dir = split_dir / f"class_{label:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # file name
        image_path = class_dir / f"{idx:06d}.JPEG"

        # convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # save JPEG (quality=95)
        image.save(image_path, quality=95, format="JPEG")
        return (idx, True, None)  # (index, success, error_msg)

    except Exception as e:
        logger.error(f"Error saving image {idx}: {e}")
        return (idx, False, str(e))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Parallel conversion of a single split (with resume capability)
# ──────────────────────────────────────────────────────────────────────────────
def save_split_parallel(split_name: str, n_jobs: int | None = None):
    """
    Save split with resume capability.
    Only processes images that don't already exist.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(exist_ok=True)

    # GET EXISTING INDICES (KEY ADDITION FOR RESUME)
    existing_indices = get_existing_indices(split_dir)

    split_len = len(dataset[split_name])  # Metadata only
    logger.info(f"\nConverting {split_name} split:")
    logger.info(f"  Total images: {split_len:,}")
    logger.info(f"  Already exist: {len(existing_indices):,}")
    logger.info(f"  Need to process: {split_len - len(existing_indices):,}")

    # PRE-COMPUTE INDICES TO PROCESS (KEY ADDITION FOR RESUME)
    indices_to_process = [
        idx for idx in range(split_len)
        if idx not in existing_indices
    ]

    if len(indices_to_process) == 0:
        logger.info(f"✅ {split_name} split is already complete!")
        return

    print(f"\nConverting {split_name} split ({len(indices_to_process):,} remaining images) "
          f"on {n_jobs or cpu_count()} cores...")

    # Prepare lightweight arguments for only needed indices
    args_iter = (
        (idx, split_name, split_dir)
        for idx in indices_to_process
    )

    # Use Pool with initializer
    saved = 0
    errors = 0

    with Pool(processes=n_jobs, initializer=init_worker) as pool:
        for result in tqdm(pool.imap_unordered(save_one_sample, args_iter),
                          total=len(indices_to_process),
                          desc=f"Saving {split_name}",
                          unit="img"):
            idx, success, error_msg = result
            if success:
                saved += 1
            else:
                errors += 1
                if errors <= 20:  # Only log first 20 errors
                    logger.error(f"Failed to save image {idx}: {error_msg}")

    logger.info(f"\n{split_name} conversion complete:")
    logger.info(f"  - Previously existed: {len(existing_indices):,}")
    logger.info(f"  - Newly saved: {saved:,}")
    logger.info(f"  - Errors: {errors:,}")
    logger.info(f"  - Total completed: {len(existing_indices) + saved:,}/{split_len:,}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Run both splits
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":          # <-- important for Windows / macOS
    n_workers = os.cpu_count()      # or set a fixed number, e.g. 8

    logger.info(f"Starting conversion with {n_workers} workers")

    save_split_parallel("train", n_jobs=n_workers)
    save_split_parallel("validation", n_jobs=n_workers)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Final report
    # ──────────────────────────────────────────────────────────────────────────
    print("\nConversion complete!")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    train/ - {len(dataset['train']):,} images in 1000 classes")
    print(f"    validation/ - {len(dataset['validation']):,} images in 1000 classes")
    print(f"\nTotal size: ~8-10 GB (after conversion)")
    print(f"Ready for training with SlowFast!")

    logger.info("✅ Conversion process complete!")
