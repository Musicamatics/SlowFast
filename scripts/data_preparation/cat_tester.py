#!/usr/bin/env python3
"""
Check if cat_all_gather is available in pytorchvideo.layers.distributed
"""

try:
    import pytorchvideo
    print(f"pytorchvideo version: {pytorchvideo.__version__}")
    from pytorchvideo.layers.distributed import cat_all_gather
    print("✓ SUCCESS: cat_all_gather is available in pytorchvideo.layers.distributed")
    print(f"Function: {cat_all_gather}")
except ImportError as e:
    print(f"✗ FAILED: Cannot import cat_all_gather")
    print(f"Error: {e}")
    
    # Try to list what IS available in the module
    try:
        import pytorchvideo.layers.distributed as distributed_module
        print("\nAvailable functions/classes in pytorchvideo.layers.distributed:")
        available = [name for name in dir(distributed_module) if not name.startswith('_')]
        for name in available:
            print(f"  - {name}")
    except Exception as e2:
        print(f"Could not inspect module: {e2}")
except Exception as e:
    print(f"✗ UNEXPECTED ERROR: {e}")
