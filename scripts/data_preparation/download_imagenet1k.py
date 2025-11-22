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
    print("\nAttempting download without token (may fail for private datasets)...")

# Download with token from environment
dataset_path = snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    local_dir="./data/imagenet-1k",
    token=token
)
print(f"Downloaded to: {dataset_path}")