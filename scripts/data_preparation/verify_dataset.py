from datasets import load_dataset

# Load the dataset from local directory
dataset = load_dataset("parquet", data_dir="./data/imagenet-100/data")

print("Dataset loaded successfully!")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Features: {dataset['train'].features}")
print(f"\nFirst sample keys: {dataset['train'][0].keys()}")
