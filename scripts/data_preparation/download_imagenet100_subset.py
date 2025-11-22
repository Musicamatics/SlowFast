from huggingface_hub import snapshot_download
import os

# Get token from environment variable
# Set your token first: export HF_TOKEN="your_token_here"
# Or use: huggingface-cli login
token = os.environ.get("HF_TOKEN", None)

if token is None:
    print("WARNING: No HuggingFace token found!")
    print("Please either:")
    print("  1. Run: huggingface-cli login")
    print("  2. Or set: export HF_TOKEN='your_token_here'")
    print("\nAttempting download without token (may work for public datasets)...")

print("Starting ImageNet-100 download (8.4 GB)...")
print("This is a 100-class subset of ImageNet-1K")

# Download ImageNet-100 subset
dataset_path = snapshot_download(
    repo_id="clane9/imagenet-100",
    repo_type="dataset",
    local_dir="./data/imagenet-100",
    token=token
)

print(f"Successfully downloaded to: {dataset_path}")
print("Dataset size: ~8.4 GB")