import sys
import os

# CRITICAL FIX: Remove PYTHONPATH temporarily to avoid SlowFast conflicts
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# Remove SlowFast from sys.path if present
sys.path = [p for p in sys.path if 'SlowFast' not in p]

# Now safe to import HuggingFace datasets
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm

print("Loading ImageNet-1k from parquet...")
dataset = load_dataset("parquet", data_dir="./data/imagenet-1k/data")
print("checkpoint!!!!!")
output_dir = Path("./data/imagenet-1k-converted")
print(output_dir)
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
save_split("val", dataset['validation'])

print("\n✅ Conversion complete!")
print(f"\nDataset structure:")
print(f"  {output_dir}/")
print(f"    train/ - {len(dataset['train']):,} images in 1000 classes")
print(f"    val/   - {len(dataset['validation']):,} images in 1000 classes")
print(f"\nTotal size: ~8-10 GB (after conversion)")
print(f"Ready for training with SlowFast!")
