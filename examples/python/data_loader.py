#!/usr/bin/env python3
"""
s3dlio Data Loader Example

Demonstrates using s3dlio as a data loader for ML/AI workloads.
Shows patterns for efficiently loading datasets from object storage.

Key concepts:
- Listing and filtering objects by pattern
- Batch loading with parallel I/O
- Shuffled iteration over datasets
- Memory-efficient streaming

Usage:
    python examples/python/data_loader.py
    python examples/python/data_loader.py s3://mybucket/dataset/
"""

import os
import sys
import tempfile
import time
import random

try:
    import s3dlio
except ImportError:
    print("ERROR: s3dlio module not found.")
    print("Build and install with: ./build_pyo3.sh && ./install_pyo3_wheel.sh")
    sys.exit(1)


class SimpleDataLoader:
    """
    A simple data loader that uses s3dlio for storage I/O.
    
    This demonstrates how to build a PyTorch/TensorFlow-style data loader
    on top of s3dlio's parallel I/O capabilities.
    """
    
    def __init__(self, uri_prefix: str, pattern: str = None, 
                 batch_size: int = 32, shuffle: bool = True,
                 prefetch: int = 2, max_in_flight: int = 16):
        """
        Initialize the data loader.
        
        Args:
            uri_prefix: Base URI for the dataset (e.g., "s3://bucket/data/")
            pattern: Optional regex pattern to filter files
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples each epoch
            prefetch: Number of batches to prefetch (not implemented in this simple version)
            max_in_flight: Parallel I/O operations for loading
        """
        self.uri_prefix = uri_prefix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_in_flight = max_in_flight
        
        # List all objects matching the pattern
        print(f"Scanning dataset at {uri_prefix}...")
        all_objects = s3dlio.list(uri_prefix, recursive=True)
        
        # Filter by pattern if provided
        if pattern:
            import re
            regex = re.compile(pattern)
            self.object_uris = [uri for uri in all_objects if regex.search(uri)]
        else:
            self.object_uris = all_objects
        
        print(f"Found {len(self.object_uris)} samples")
        self._indices = list(range(len(self.object_uris)))
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return (len(self.object_uris) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        # Shuffle indices at the start of each epoch
        if self.shuffle:
            random.shuffle(self._indices)
        
        # Yield batches
        for batch_start in range(0, len(self._indices), self.batch_size):
            batch_indices = self._indices[batch_start:batch_start + self.batch_size]
            batch_uris = [self.object_uris[i] for i in batch_indices]
            
            # Load batch in parallel
            results = s3dlio.get_many(batch_uris, max_in_flight=self.max_in_flight)
            
            # Return list of (uri, data) tuples
            yield [(uri, bytes(data)) for uri, data in results]
    
    def get_sample(self, index: int) -> tuple:
        """Get a single sample by index."""
        uri = self.object_uris[index]
        data = s3dlio.get(uri)
        return uri, bytes(data)


def create_synthetic_dataset(base_uri: str, num_samples: int = 100, 
                            sample_size: int = 4096) -> list:
    """Create a synthetic dataset for testing."""
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    # Create samples using parallel PUT
    s3dlio.put(
        prefix=base_uri,
        num=num_samples,
        template="sample_{}.bin",
        size=sample_size,
        object_type="random",
        max_in_flight=32,
    )
    
    # Return list of created URIs
    return s3dlio.list(base_uri, recursive=False)


def main():
    print("=" * 60)
    print("s3dlio Data Loader Example")
    print("=" * 60)
    
    # Determine base URI from command line or use local filesystem
    if len(sys.argv) > 1:
        base_uri = sys.argv[1]
        if not base_uri.endswith('/'):
            base_uri += '/'
        create_data = False
    else:
        temp_dir = tempfile.mkdtemp(prefix="s3dlio_dataloader_example_")
        base_uri = f"file://{temp_dir}/"
        print(f"\nUsing local filesystem: {base_uri}")
        create_data = True
    
    print(f"Dataset URI: {base_uri}")
    
    try:
        # =====================================================================
        # Create synthetic dataset if needed
        # =====================================================================
        if create_data:
            sample_uris = create_synthetic_dataset(
                base_uri, 
                num_samples=100, 
                sample_size=4096
            )
            print(f"Created {len(sample_uris)} samples")
        
        # =====================================================================
        # Example 1: Basic data loader usage
        # =====================================================================
        print("\n" + "=" * 60)
        print("📚 Example 1: Basic Data Loader")
        print("=" * 60)
        
        loader = SimpleDataLoader(
            uri_prefix=base_uri,
            batch_size=10,
            shuffle=True,
            max_in_flight=8,
        )
        
        print(f"Dataset size: {len(loader.object_uris)} samples")
        print(f"Batch size: {loader.batch_size}")
        print(f"Batches per epoch: {len(loader)}")
        
        # Iterate through one epoch
        print("\nIterating through epoch 1...")
        epoch_bytes = 0
        batch_count = 0
        start_time = time.time()
        
        for batch in loader:
            batch_bytes = sum(len(data) for _, data in batch)
            epoch_bytes += batch_bytes
            batch_count += 1
            if batch_count % 5 == 0:
                print(f"  Processed batch {batch_count}/{len(loader)}")
        
        elapsed = time.time() - start_time
        print(f"\nEpoch 1 complete:")
        print(f"  Batches: {batch_count}")
        print(f"  Total bytes: {epoch_bytes:,}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {epoch_bytes / elapsed / 1024 / 1024:.2f} MB/s")
        
        # =====================================================================
        # Example 2: Multiple epochs with timing
        # =====================================================================
        print("\n" + "=" * 60)
        print("📚 Example 2: Multiple Epochs")
        print("=" * 60)
        
        num_epochs = 3
        epoch_times = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            samples_processed = 0
            
            for batch in loader:
                samples_processed += len(batch)
                # Simulate some processing time
                # In real usage, you'd process the data here
            
            elapsed = time.time() - start_time
            epoch_times.append(elapsed)
            print(f"  Epoch {epoch + 1}: {elapsed:.3f}s ({samples_processed} samples)")
        
        avg_time = sum(epoch_times) / len(epoch_times)
        print(f"\nAverage epoch time: {avg_time:.3f}s")
        
        # =====================================================================
        # Example 3: Filtered data loading
        # =====================================================================
        print("\n" + "=" * 60)
        print("📚 Example 3: Filtered Data Loading")
        print("=" * 60)
        
        # Load only samples with indices 0-49
        filtered_loader = SimpleDataLoader(
            uri_prefix=base_uri,
            pattern=r"sample_[0-4]\d\.bin",  # Matches sample_00.bin to sample_49.bin
            batch_size=10,
            shuffle=False,
        )
        
        print(f"Filtered dataset size: {len(filtered_loader.object_uris)} samples")
        
        # =====================================================================
        # Example 4: Single sample access
        # =====================================================================
        print("\n" + "=" * 60)
        print("📚 Example 4: Random Access")
        print("=" * 60)
        
        # Get specific samples by index
        indices_to_fetch = [0, 10, 50, 99]
        print(f"Fetching samples at indices: {indices_to_fetch}")
        
        for idx in indices_to_fetch:
            if idx < len(loader.object_uris):
                uri, data = loader.get_sample(idx)
                print(f"  [{idx}] {os.path.basename(uri)}: {len(data)} bytes")
        
        # =====================================================================
        # Cleanup
        # =====================================================================
        if create_data:
            print("\n🧹 Cleaning up...")
            s3dlio.delete(base_uri, recursive=True)
            print("   Done")
        
        print("\n" + "=" * 60)
        print("✅ Data loader examples completed!")
        print("=" * 60)
        print("\nThis example demonstrates:")
        print("  - Building a data loader on top of s3dlio")
        print("  - Parallel batch loading with get_many()")
        print("  - Shuffled iteration for training")
        print("  - Pattern-based filtering")
        print("  - Random access to individual samples")
        
    finally:
        # Clean up temp directory if we created one
        if base_uri.startswith("file://") and "s3dlio_dataloader_example_" in base_uri:
            import shutil
            temp_dir = base_uri.replace("file://", "").rstrip('/')
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
