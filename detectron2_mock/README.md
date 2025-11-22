# Detectron2 Mock

## Why This Exists

The original MaskFeat codebase (based on Facebook Research's SlowFast) imports `detectron2` for certain vision detection features. However, we encountered significant issues:

1. **Unmaintained**: Meta AI's `detectron2` is no longer actively maintained
2. **Build Issues**: Fails to build from source with modern CUDA (11.8, 12.1)
3. **Compatibility**: Has conflicts with PyTorch 2.0+ and modern dependencies
4. **Actually Not Needed**: For image-only fine-tuning (our use case), detectron2 features are never actually called

## The Solution

This mock provides minimal stubs to satisfy `import detectron2` statements without requiring the full (broken) detectron2 installation. It allows the MaskFeat image fine-tuning pipeline to run successfully.

## What's Included

```
detectron2_mock/
└── detectron2/
    ├── __init__.py           # Empty init
    ├── config/
    │   └── __init__.py       # Empty config init
    ├── layers/
    │   └── __init__.py       # Empty layers init
    └── model_zoo.py          # Minimal stub
```

## Usage

Add this directory to your `PYTHONPATH` before running training:

```bash
export PYTHONPATH=/path/to/detectron2_mock:$PYTHONPATH
```

Or in your SLURM batch script:

```bash
export PYTHONPATH=~/detectron2_mock:~/maskfeat_project/SlowFast:$PYTHONPATH
```

## Verification

Test that the mock is working:

```bash
python -c "import detectron2; print('✅ detectron2 mock working!')"
```

## Limitations

⚠️ **This mock ONLY works for image-based MaskFeat fine-tuning.**

The following will **NOT** work with this mock:
- Video-based training (requires actual Mask R-CNN features)
- Object detection tasks
- Instance segmentation
- Any code path that actually uses detectron2 features (not just imports)

For our ImageNet fine-tuning reproduction, these limitations don't matter since we only use image classification.

## Why Not Just Fix Detectron2?

We tried! Here's what didn't work:

1. **Build from source**:
   ```bash
   git clone https://github.com/facebookresearch/detectron2.git
   python -m pip install -e detectron2
   # ❌ Fails with CUDA version mismatches
   ```

2. **Pre-built wheels**:
   ```bash
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/...
   # ❌ No wheels available for CUDA 11.8/12.1 + PyTorch 2.0
   ```

3. **Conda**:
   ```bash
   conda install -c conda-forge detectron2
   # ❌ Package conflicts with other dependencies
   ```

Creating this mock was faster and more reliable than debugging detectron2 installation issues.

## Contributing

If you find that this mock is insufficient for your use case, you have two options:

1. **Extend the mock**: Add more stub methods/classes as needed
2. **Install real detectron2**: If you need actual detection features, you'll need to properly install detectron2 (good luck!)

## License

## Author

Created by Musicamatics for MaskFeat reproduction project.

## License

This mock implementation is provided as-is for educational and research purposes.
Original SlowFast and detectron2 are under Apache 2.0 license.

