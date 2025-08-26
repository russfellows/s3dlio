#!/usr/bin/env python3
"""
Simple Python validation test for s3dlio v0.7.0
Tests the core functionality that should work with the current API
"""

import tempfile
import os
import numpy as np
import s3dlio

def test_basic_checkpoint_functionality():
    """Test basic checkpoint functionality with the working API"""
    print("=== Testing Basic v0.7.0 Checkpoint Functionality ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        print(f"Using URI: {base_uri}")
        
        # Test data
        test_data = np.random.rand(100, 100).astype(np.float32)
        print(f"Test data: {test_data.shape}, {test_data.nbytes} bytes")
        
        # Create checkpoint store 
        store = s3dlio.PyCheckpointStore(base_uri, strategy='binary', multipart_threshold=1024*1024)
        writer = store.writer(world_size=1, rank=0)
        
        # Test save with correct API
        print("Testing save_distributed_shard...")
        shard_meta = writer.save_distributed_shard(
            100,    # step
            1,      # epoch
            "test", # framework
            test_data.tobytes()  # data
        )
        
        print(f"✓ Saved: {shard_meta['size']} bytes")
        print(f"✓ Key: {shard_meta['key']}")
        
        # Test streaming interface
        print("Testing streaming interface...")
        stream = writer.get_distributed_shard_stream(200, 2, "test")  # step, epoch, framework
        
        # Write in chunks
        chunk_size = 1000
        test_bytes = test_data.tobytes()
        for i in range(0, len(test_bytes), chunk_size):
            chunk = test_bytes[i:i+chunk_size]
            stream.write_chunk(chunk)
        
        stream_meta = stream.finalize()
        print(f"✓ Streamed: {stream_meta['size']} bytes")
        print(f"✓ Stream key: {stream_meta['key']}")
        
        # Verify sizes match
        assert shard_meta['size'] == stream_meta['size'], "Sizes should match"
        
        print("✅ Basic checkpoint functionality test passed")

def test_framework_specific_patterns():
    """Test framework-specific usage patterns"""
    print("\n=== Testing Framework-Specific Patterns ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        
        frameworks = ["jax", "pytorch", "tensorflow", "numpy"]
        
        store = s3dlio.PyCheckpointStore(base_uri, strategy='binary', multipart_threshold=1024*1024)
        writer = store.writer(world_size=1, rank=0)
        
        for i, framework in enumerate(frameworks):
            # Different data for each framework
            data = np.random.rand(50, 50).astype(np.float32) * (i + 1)
            
            print(f"Testing {framework} framework...")
            
            # Save with framework-specific step/epoch
            meta = writer.save_distributed_shard(
                i * 100,     # step
                i,           # epoch
                framework,   # framework
                data.tobytes()  # data
            )
            
            print(f"✓ {framework}: {meta['size']} bytes saved to {meta['key']}")
            
            # Test streaming too
            stream = writer.get_distributed_shard_stream(
                (i * 100) + 50,   # step
                i + 1,            # epoch
                framework         # framework
            )
            
            stream.write_chunk(data.tobytes())
            stream_meta = stream.finalize()
            
            print(f"✓ {framework} stream: {stream_meta['size']} bytes")
        
        print("✅ Framework-specific patterns test passed")

def test_multirank_simulation():
    """Test multi-rank checkpoint simulation"""
    print("\n=== Testing Multi-Rank Simulation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        
        world_size = 4
        
        for rank in range(world_size):
            print(f"Simulating rank {rank}/{world_size-1}...")
            
            store = s3dlio.PyCheckpointStore(base_uri, strategy='binary', multipart_threshold=1024*1024)
            writer = store.writer(world_size=world_size, rank=rank)
            
            # Rank-specific data
            rank_data = np.random.rand(25, 25).astype(np.float32) * (rank + 1)
            
            meta = writer.save_distributed_shard(
                1000,         # step
                10,           # epoch
                "distributed", # framework
                rank_data.tobytes()  # data
            )
            
            print(f"✓ Rank {rank}: {meta['size']} bytes saved to {meta['key']}")
        
        print("✅ Multi-rank simulation test passed")

def test_v070_features():
    """Test v0.7.0 specific features"""
    print("\n=== Testing v0.7.0 Specific Features ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        
        # Test enhanced multipart threshold
        store = s3dlio.PyCheckpointStore(
            base_uri, 
            strategy='binary',
            multipart_threshold=256*1024  # 256KB threshold
        )
        writer = store.writer(world_size=1, rank=0)
        
        # Large data to trigger multipart if implemented
        large_data = np.random.rand(1000, 1000).astype(np.float32)  # ~4MB
        print(f"Large data test: {large_data.nbytes/1024/1024:.1f}MB")
        
        meta = writer.save_distributed_shard(
            2000,
            20,
            "large_test",
            large_data.tobytes()
        )
        
        print(f"✓ Large data saved: {meta['size']} bytes")
        
        # Test various step/epoch combinations (v0.7.0 enhancement)
        test_configs = [
            (0, 0, "initial"),
            (1000, 1, "mid_training"),
            (10000, 100, "final"),
        ]
        
        for step, epoch, tag in test_configs:
            test_data = np.random.rand(10, 10).astype(np.float32)
            
            meta = writer.save_distributed_shard(
                step,
                epoch,
                f"config_{tag}",
                test_data.tobytes()
            )
            
            print(f"✓ Config {tag} (step={step}, epoch={epoch}): {meta['size']} bytes")
        
        print("✅ v0.7.0 features test passed")

def main():
    """Run all simple Python tests for s3dlio v0.7.0"""
    print("🚀 Starting Simple s3dlio v0.7.0 Python Validation")
    print("Testing core functionality with working API patterns")
    print("=" * 60)
    
    try:
        test_basic_checkpoint_functionality()
        test_framework_specific_patterns()
        test_multirank_simulation()
        test_v070_features()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Basic checkpoint functionality: PASSED")
        print("✅ Framework-specific patterns: PASSED")
        print("✅ Multi-rank simulation: PASSED")
        print("✅ v0.7.0 features: PASSED")
        print("\ns3dlio v0.7.0 Python API is working correctly! 🚀")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
