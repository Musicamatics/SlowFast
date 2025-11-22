import pyarrow.parquet as pq
import os

print("Checking ImageNet-100 dataset files...")

data_dir = "./data/imagenet-100/data"

# Check training files
train_files = [f for f in os.listdir(data_dir) if f.startswith('train-')]
val_files = [f for f in os.listdir(data_dir) if f.startswith('validation-')]

print(f"\n✅ Found {len(train_files)} training parquet files")
print(f"✅ Found {len(val_files)} validation parquet files")

# Read first parquet file to check structure
first_train = os.path.join(data_dir, sorted(train_files)[0])
table = pq.read_table(first_train)

print(f"\n📊 Dataset Schema:")
print(f"  Columns: {table.column_names}")
print(f"  Rows in first file: {len(table):,}")

# Check first record
first_row = table.to_pandas().iloc[0]
print(f"\n📝 First sample:")
print(f"  Label: {first_row['label']}")
print(f"  Image size: {len(first_row['image']['bytes'])} bytes")

print("\n✅ Dataset verification complete!")
print(f"Total training files: {len(train_files)}")
print(f"Total validation files: {len(val_files)}")
