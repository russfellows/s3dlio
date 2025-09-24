#!/usr/bin/env python3
"""Simple test to check what functions are available in s3dlio."""

import s3dlio
import tempfile
import os

print("Available functions in s3dlio:")
attrs = sorted([name for name in dir(s3dlio) if not name.startswith('_')])
for attr in attrs:
    print(f"  - {attr}")

# Test the key functions we implemented
print(f"\ncreate_dataset available: {'create_dataset' in attrs}")
print(f"create_async_loader available: {'create_async_loader' in attrs}")
print(f"list available: {'list' in attrs}")
print(f"stat available: {'stat' in attrs}")
print(f"get available: {'get' in attrs}")

# Test if we can create a simple dataset
if 'create_dataset' in attrs:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
        tmp.write(b"test data")
        tmp_path = tmp.name
    
    try:
        dataset = s3dlio.create_dataset(f"file://{tmp_path}")
        print(f"✓ create_dataset works! Type: {type(dataset)}")
    except Exception as e:
        print(f"✗ create_dataset failed: {e}")
    finally:
        os.unlink(tmp_path)

# Test PyTorch import
try:
    from s3dlio.torch import S3IterableDataset
    print("✓ S3IterableDataset import works")
except Exception as e:
    print(f"✗ S3IterableDataset import failed: {e}")

print("\nTorch module contents:")
try:
    import s3dlio.torch as torch_mod
    torch_attrs = [name for name in dir(torch_mod) if not name.startswith('_')]
    for attr in torch_attrs:
        print(f"  - {attr}")
except Exception as e:
    print(f"Failed to import torch module: {e}")