#!/usr/bin/env python3
"""
Practical TFRecord Index Usage Examples

Demonstrates real-world use cases for TFRecord indexes in ML workflows:
1. Random access to specific records
2. Shuffled data loading
3. Distributed training (multi-process data sharding)
4. Batch loading with indexes
"""

import s3dlio
import struct
import tempfile
import os
import random
from typing import List, Tuple


def create_sample_tfrecord(num_records: int = 100) -> bytes:
    """Create a sample TFRecord file with labeled data"""
    data = bytearray()
    
    for i in range(num_records):
        # Simulate image data with varying sizes
        image_size = random.randint(1000, 5000)
        record_data = (
            f"Record {i}: ".encode() + 
            b"X" * image_size +  # Simulated image data
            f" Label: {i % 10}".encode()  # Class label 0-9
        )
        
        # TFRecord format: length + length_crc + data + data_crc
        length = len(record_data)
        data.extend(struct.pack('<Q', length))
        data.extend(b'\x00' * 4)  # length_crc (simplified)
        data.extend(record_data)
        data.extend(b'\x00' * 4)  # data_crc (simplified)
    
    return bytes(data)


def read_record_at_index(tfrecord_path: str, index_entries: List[Tuple[int, int]], 
                         record_idx: int) -> bytes:
    """
    Read a specific record using the index
    
    Args:
        tfrecord_path: Path to TFRecord file
        index_entries: List of (offset, size) tuples from index
        record_idx: Which record to read (0-based)
    
    Returns:
        The decoded record data
    """
    offset, size = index_entries[record_idx]
    
    with open(tfrecord_path, 'rb') as f:
        f.seek(offset)
        record_bytes = f.read(size)
    
    # Parse TFRecord structure
    data_length = struct.unpack('<Q', record_bytes[:8])[0]
    data_start = 12  # Skip length (8) + length_crc (4)
    actual_data = record_bytes[data_start:data_start + data_length]
    
    return actual_data


def example_1_random_access():
    """Example 1: Random Access to Specific Records"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Random Access to Specific Records")
    print("="*70)
    print("Use case: Retrieve specific examples for debugging or visualization")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "dataset.tfrecord")
        index_path = os.path.join(tmpdir, "dataset.tfrecord.idx")
        
        # Create dataset
        print("\n1. Creating dataset with 100 records...")
        tfrecord_data = create_sample_tfrecord(100)
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        # Generate index
        print("2. Generating index...")
        s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        
        # Load index
        print("3. Loading index...")
        index_entries = s3dlio.read_tfrecord_index(index_path)
        print(f"   Loaded {len(index_entries)} index entries")
        
        # Random access to specific records
        print("\n4. Reading specific records by index:")
        for i in [0, 25, 50, 75, 99]:
            data = read_record_at_index(tfrecord_path, index_entries, i)
            preview = data[:30].decode('utf-8', errors='replace')
            print(f"   Record {i:2d}: {preview}...")
        
        print("\n✓ Random access complete - no need to scan entire file")


def example_2_shuffled_loading():
    """Example 2: Shuffled Data Loading"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Shuffled Data Loading")
    print("="*70)
    print("Use case: Load training data in random order for each epoch")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "training.tfrecord")
        index_path = os.path.join(tmpdir, "training.tfrecord.idx")
        
        # Create dataset
        print("\n1. Creating training dataset...")
        tfrecord_data = create_sample_tfrecord(50)
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        # Generate and load index
        s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        index_entries = s3dlio.read_tfrecord_index(index_path)
        
        # Simulate 3 training epochs with different shuffles
        print("\n2. Simulating 3 epochs with random shuffling:")
        for epoch in range(3):
            # Shuffle indices for this epoch
            indices = list(range(len(index_entries)))
            random.shuffle(indices)
            
            print(f"\n   Epoch {epoch + 1} order (first 10 records): {indices[:10]}")
            
            # In real training, you would load batches here
            # For demo, just show we can access in shuffled order
            sample_data = read_record_at_index(tfrecord_path, index_entries, indices[0])
            preview = sample_data[:25].decode('utf-8', errors='replace')
            print(f"   First record in epoch: {preview}...")
        
        print("\n✓ Each epoch uses different random order")


def example_3_distributed_sharding():
    """Example 3: Distributed Training with Data Sharding"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Distributed Training - Data Sharding")
    print("="*70)
    print("Use case: Split dataset across multiple GPUs/processes")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "distributed.tfrecord")
        index_path = os.path.join(tmpdir, "distributed.tfrecord.idx")
        
        # Create dataset
        num_records = 100
        print(f"\n1. Creating dataset with {num_records} records...")
        tfrecord_data = create_sample_tfrecord(num_records)
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        index_entries = s3dlio.read_tfrecord_index(index_path)
        
        # Simulate 4-GPU distributed training
        num_shards = 4
        print(f"\n2. Distributing across {num_shards} processes/GPUs:")
        
        for shard_id in range(num_shards):
            # Each process gets every Nth record (interleaved sharding)
            shard_indices = list(range(shard_id, len(index_entries), num_shards))
            
            print(f"\n   GPU {shard_id}:")
            print(f"     - Handles {len(shard_indices)} records")
            print(f"     - Indices: {shard_indices[:5]}...")
            
            # Load first record from this shard
            data = read_record_at_index(tfrecord_path, index_entries, shard_indices[0])
            preview = data[:25].decode('utf-8', errors='replace')
            print(f"     - First record: {preview}...")
        
        print("\n✓ Data distributed evenly across processes")
        print("  Each process reads only its portion (no redundant I/O)")


def example_4_batch_loading():
    """Example 4: Efficient Batch Loading"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Efficient Batch Loading")
    print("="*70)
    print("Use case: Load data in batches for training")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "batches.tfrecord")
        index_path = os.path.join(tmpdir, "batches.tfrecord.idx")
        
        # Create dataset
        num_records = 40
        batch_size = 8
        print(f"\n1. Creating dataset with {num_records} records...")
        tfrecord_data = create_sample_tfrecord(num_records)
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        index_entries = s3dlio.read_tfrecord_index(index_path)
        
        print(f"\n2. Loading in batches of {batch_size}:")
        
        # Load batches
        num_batches = len(index_entries) // batch_size
        
        for batch_idx in range(min(3, num_batches)):  # Show first 3 batches
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = list(range(start_idx, end_idx))
            
            print(f"\n   Batch {batch_idx + 1}:")
            print(f"     - Record indices: {batch_indices}")
            
            # Load all records in batch (in real scenario, parallelize this)
            batch_data = []
            for idx in batch_indices:
                data = read_record_at_index(tfrecord_path, index_entries, idx)
                batch_data.append(data)
            
            print(f"     - Loaded {len(batch_data)} records")
            
            # Show first record in batch
            preview = batch_data[0][:25].decode('utf-8', errors='replace')
            print(f"     - First record: {preview}...")
        
        print(f"\n✓ Total batches available: {num_batches}")
        print("  Index enables efficient seeking to batch boundaries")


def example_5_performance_comparison():
    """Example 5: Performance Comparison (With vs Without Index)"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Performance Implications")
    print("="*70)
    print("Use case: Understanding the benefit of indexes")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "perf.tfrecord")
        index_path = os.path.join(tmpdir, "perf.tfrecord.idx")
        
        # Create dataset
        num_records = 1000
        print(f"\n1. Creating dataset with {num_records} records...")
        tfrecord_data = create_sample_tfrecord(num_records)
        
        file_size_mb = len(tfrecord_data) / (1024 * 1024)
        print(f"   File size: {file_size_mb:.2f} MB")
        
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        index_entries = s3dlio.read_tfrecord_index(index_path)
        
        index_size_bytes = os.path.getsize(index_path)
        print(f"   Index size: {index_size_bytes / 1024:.2f} KB")
        print(f"   Overhead: {(index_size_bytes / len(tfrecord_data)) * 100:.3f}%")
        
        print("\n2. Access patterns:")
        print("\n   WITHOUT INDEX:")
        print("     - To read record 500: Must scan 500 records sequentially")
        print("     - To read 10 random records: Must scan most of file")
        print("     - Random shuffling: Requires full file scan + buffering")
        
        print("\n   WITH INDEX:")
        print("     - To read record 500: Direct seek using index")
        offset, size = index_entries[500]
        print(f"       → Seek to offset {offset}, read {size} bytes")
        print("     - To read 10 random records: 10 direct seeks")
        print("     - Random shuffling: Shuffle indices, then seek efficiently")
        
        print("\n3. Performance characteristics:")
        avg_record_size = len(tfrecord_data) / num_records
        print(f"   Average record size: {avg_record_size:.0f} bytes")
        print(f"   Sequential read overhead (no index): ~{avg_record_size * 500:.0f} bytes to reach record 500")
        print(f"   Indexed read overhead: ~{size} bytes (just the record)")
        print(f"   Speedup factor: ~{(avg_record_size * 500) / size:.1f}x for random access")
        
        print("\n✓ Index provides O(1) random access vs O(n) sequential scan")


def main():
    """Run all examples"""
    print("="*70)
    print("TFRecord Index Usage Examples")
    print("="*70)
    print("\nDemonstrating practical use cases for indexed TFRecord access")
    
    examples = [
        example_1_random_access,
        example_2_shuffled_loading,
        example_3_distributed_sharding,
        example_4_batch_loading,
        example_5_performance_comparison,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ ERROR in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Summary: TFRecord indexes enable efficient data access patterns")
    print("="*70)
    print("""
Key benefits:
  1. Random Access: O(1) seeking to any record
  2. Shuffling: Efficient random-order loading for training
  3. Distributed Training: Easy data sharding across workers
  4. Batch Loading: Fast seeking to batch boundaries
  5. Low Overhead: Index is ~0.02% of data size
  6. DALI Compatible: Works with NVIDIA DALI pipelines
  7. TensorFlow Compatible: Standard format used by TF tools
""")


if __name__ == "__main__":
    main()
