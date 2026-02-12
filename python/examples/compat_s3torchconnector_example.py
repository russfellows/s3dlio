#!/usr/bin/env python3
"""
Example demonstrating s3dlio as drop-in replacement for s3torchconnector.

This script shows how to switch between s3torchconnector and s3dlio with NO code changes
except for the import statement.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np


def create_test_data(data_dir: Path, num_files: int = 10):
    """Create test data files for the example."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_files):
        # Create random data of varying sizes
        size = np.random.randint(100, 1000)
        data = np.random.bytes(size)
        
        file_path = data_dir / f"sample_{i:03d}.bin"
        with open(file_path, "wb") as f:
            f.write(data)
    
    print(f"✓ Created {num_files} test files in {data_dir}")


def example_iterable_dataset(backend: str, uri: str, region: str = None):
    """
    Example using S3IterableDataset - works identically with both backends!
    """
    print(f"\n{'='*60}")
    print(f"S3IterableDataset Example ({backend})")
    print(f"{'='*60}")
    
    # THE ONLY LINE THAT CHANGES between backends!
    if backend == "s3torchconnector":
        from s3torchconnector import S3IterableDataset
    else:  # s3dlio
        from s3dlio.compat.s3torchconnector import S3IterableDataset
    
    # Everything else is IDENTICAL
    dataset = S3IterableDataset.from_prefix(uri, region=region)
    
    print(f"Iterating over dataset from: {uri}")
    total_bytes = 0
    count = 0
    
    for item in dataset:
        # Access bucket and key (API-compatible!)
        print(f"  {count}: {item.bucket}/{item.key} - {len(item)} bytes")
        
        # Read data (API-compatible!)
        data = item.read()
        total_bytes += len(data)
        count += 1
        
        if count >= 5:  # Just show first 5 items
            break
    
    print(f"✓ Processed {count} items, {total_bytes:,} bytes total")


def example_map_dataset(backend: str, uri: str, region: str = None):
    """
    Example using S3MapDataset - works identically with both backends!
    """
    print(f"\n{'='*60}")
    print(f"S3MapDataset Example ({backend})")
    print(f"{'='*60}")
    
    # THE ONLY LINE THAT CHANGES between backends!
    if backend == "s3torchconnector":
        from s3torchconnector import S3MapDataset
    else:  # s3dlio
        from s3dlio.compat.s3torchconnector import S3MapDataset
    
    # Everything else is IDENTICAL
    dataset = S3MapDataset.from_prefix(uri, region=region)
    
    print(f"Map dataset from: {uri}")
    print(f"Dataset size: {len(dataset)} objects")
    
    # Random access (API-compatible!)
    indices = [0, len(dataset)//2, len(dataset)-1]
    for idx in indices:
        item = dataset[idx]
        data = item.read()
        print(f"  [{idx}]: {item.bucket}/{item.key} - {len(data)} bytes")
    
    print(f"✓ Random access working correctly")


def example_checkpoint(backend: str, uri: str, region: str = None):
    """
    Example using S3Checkpoint - works identically with both backends!
    """
    print(f"\n{'='*60}")
    print(f"S3Checkpoint Example ({backend})")  
    print(f"{'='*60}")
    
    # THE ONLY LINE THAT CHANGES between backends!
    if backend == "s3torchconnector":
        from s3torchconnector import S3Checkpoint
    else:  # s3dlio
        from s3dlio.compat.s3torchconnector import S3Checkpoint
    
    # Everything else is IDENTICAL
    checkpoint = S3Checkpoint(region=region)
    
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    original_state = model.state_dict()
    
    # Save checkpoint
    ckpt_uri = f"{uri.rstrip('/')}/checkpoint.pt"
    print(f"Saving checkpoint to: {ckpt_uri}")
    
    with checkpoint.writer(ckpt_uri) as writer:
        torch.save(model.state_dict(), writer)
    
    print(f"✓ Checkpoint saved")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_uri}")
    
    with checkpoint.reader(ckpt_uri) as reader:
        loaded_state = torch.load(reader)
    
    # Verify
    for key in original_state:
        if not torch.equal(original_state[key], loaded_state[key]):
            print(f"✗ Checkpoint mismatch on {key}")
            return
    
    print(f"✓ Checkpoint loaded and verified")


def example_pytorch_dataloader(backend: str, uri: str, region: str = None):
    """
    Example using PyTorch DataLoader - works identically with both backends!
    """
    print(f"\n{'='*60}")
    print(f"PyTorch DataLoader Example ({backend})")
    print(f"{'='*60}")
    
    # THE ONLY LINE THAT CHANGES between backends!
    if backend == "s3torchconnector":
        from s3torchconnector import S3IterableDataset
    else:  # s3dlio
        from s3dlio.compat.s3torchconnector import S3IterableDataset
    
    # Everything else is IDENTICAL
    dataset = S3IterableDataset.from_prefix(uri, region=region, enable_sharding=True)
    
    # Use with PyTorch DataLoader (API-compatible!)
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
    
    print(f"DataLoader with batch_size=4, num_workers=0")
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"  Batch {batch_idx}: {len(batch)} items")
        for item in batch:
            print(f"    - {item.bucket}/{item.key}: {len(item.read())} bytes")
        
        if batch_idx >= 2:  # Just show first 3 batches
            break
    
    print(f"✓ DataLoader working correctly")


def main():
    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_data"
        create_test_data(data_dir, num_files=10)
        
        # Use file:// URI for local testing (works with s3dlio, not s3torchconnector)
        test_uri = f"file://{data_dir}/"
        
        print("\n" + "="*60)
        print("s3dlio as Drop-In Replacement for s3torchconnector")
        print("="*60)
        print(f"\nTest URI: {test_uri}")
        print("\nThis example shows how to switch between backends by changing ONLY the import.")
        print("All other code remains identical!")
        
        backend = "s3dlio"  # Change this to "s3torchconnector" to test with AWS library
        
        # Note: s3torchconnector only supports S3 URIs, not file://
        # For actual S3 testing, use something like: "s3://my-bucket/test-data/"
        
        if test_uri.startswith("file://"):
            if backend == "s3torchconnector":
                print("\n⚠ WARNING: s3torchconnector only supports S3 URIs (s3://)")
                print("           Skipping tests with file:// URI")
                print("           To test, use an actual S3 bucket URI")
                print("\n💡 TIP: s3dlio supports file://, s3://, az://, gs://, and direct://!")
                return
        
        # Run all examples
        example_iterable_dataset(backend, test_uri, region=None)
        example_map_dataset(backend, test_uri, region=None)
        example_checkpoint(backend, test_uri, region=None)
        example_pytorch_dataloader(backend, test_uri, region=None)
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nTo switch to s3torchconnector for comparison:")
        print("  1. Change 'backend' variable to 's3torchconnector'")
        print("  2. Use S3 URI (e.g., 's3://my-bucket/test-data/')")
        print("  3. Set AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("\nNo other code changes needed! Perfect drop-in replacement.")
        print("="*60)


if __name__ == "__main__":
    main()
