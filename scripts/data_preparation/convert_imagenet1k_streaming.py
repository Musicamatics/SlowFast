#!/usr/bin/env python3
"""
ImageNet-1K Streaming Conversion Script
Converts parquet files directly without caching to avoid disk quota issues
"""

import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq

print("="*60)
print("ImageNet-1K STREAMING Conversion")
print("No caching - Direct parquet processing")
print("="*60)

# Configuration
DATA_DIR = Path("./data/imagenet-1k/data")
OUTPUT_DIR = Path("./data/imagenet-1k-converted")

if OUTPUT_DIR.exists():
    print(f"Removing existing output directory...")
    import shutil
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True)

def process_parquet_file(parquet_file, split_dir):
    """Process a single parquet file and save images"""
    table = pq.read_table(parquet_file)
    
    for row_idx in range(len(table)):
        # Get data
        label = table['label'][row_idx].as_py()
        image_data = table['image'][row_idx].as_py()
        
        # Create class directory
        class_dir = split_dir / f"class_{label:03d}"
        class_dir.mkdir(exist_ok=True)
        
        # Open and save image
        image = Image.open(io.BytesIO(image_data['bytes']))
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save with unique filename
        image_filename = f"{parquet_file.stem}_{row_idx:06d}.JPEG"
        image_path = class_dir / image_filename
        image.save(image_path, quality=95, format='JPEG')

def convert_split(split_name):
    """Convert a dataset split"""
    print(f"\n{'='*60}")
    print(f"Converting {split_name} split")
    print(f"{'='*60}")
    
    # Find all parquet files for this split
    parquet_files = sorted(DATA_DIR.glob(f"{split_name}-*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found for {split_name}")
        return False
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Create split directory
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(exist_ok=True)
    
    # Process each parquet file
    for parquet_file in tqdm(parquet_files, desc=f"Processing {split_name}"):
        try:
            process_parquet_file(parquet_file, split_dir)
        except Exception as e:
            print(f"\n❌ Error processing {parquet_file}: {e}")
            return False
    
    # Verify
    num_classes = len(list(split_dir.iterdir()))
    print(f"✅ {split_name}: {num_classes} classes created")
    
    return True

# Convert both splits
import io

if not convert_split("train"):
    print("❌ Failed to convert train split")
    sys.exit(1)

if not convert_split("validation"):
    print("❌ Failed to convert validation split")
    sys.exit(1)

print("\n" + "="*60)
print("✅ CONVERSION COMPLETE!")
print("="*60)

# Final verification
train_classes = len(list((OUTPUT_DIR / "train").iterdir()))
val_classes = len(list((OUTPUT_DIR / "validation").iterdir()))

print(f"Train classes: {train_classes}")
print(f"Validation classes: {val_classes}")

if train_classes == 1000 and val_classes == 1000:
    print("✅ Verification passed!")
    sys.exit(0)
else:
    print("❌ Verification failed!")
    sys.exit(1)