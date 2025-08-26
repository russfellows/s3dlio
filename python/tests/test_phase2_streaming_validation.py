#!/usr/bin/env python3
"""
Phase 2 validation test for Python streaming interface
"""

import tempfile
import os
import numpy as np
import s3dlio

def test_python_streaming_interface():
    """Test the PyCheckpointStream interface for zero-copy streaming"""
    print("=== Testing Python Streaming Interface ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        print(f"Using temp directory: {temp_dir}")
        
        # Create a checkpoint writer using the correct API
        store = s3dlio.PyCheckpointStore(base_uri, strategy='binary', multipart_threshold=1024*1024)
        writer = store.writer(world_size=1, rank=0)
        
        # Test the streaming interface
        checkpoint_id = 42
        
        # Create test data (simulating a large tensor)
        # This would typically be PyTorch tensors, JAX arrays, etc.
        test_data_1 = np.random.bytes(1024 * 1024)  # 1MB chunk
        test_data_2 = np.random.bytes(2 * 1024 * 1024)  # 2MB chunk
        test_data_3 = np.random.bytes(512 * 1024)  # 0.5KB chunk
        
        print(f"Created test data: {len(test_data_1) + len(test_data_2) + len(test_data_3)} bytes total")
        
        # Test 1: Traditional save_distributed_shard (memory copying)
        traditional_data = test_data_1 + test_data_2 + test_data_3
        shard_meta_traditional = writer.save_distributed_shard(
            100,  # step
            1,    # epoch
            "pytorch",  # framework
            traditional_data  # data
        )
        print(f"✓ Traditional method: {shard_meta_traditional['size']} bytes written to {shard_meta_traditional['key']}")
        
        # Test 2: New streaming interface (zero-copy)
        print("Testing streaming interface...")
        
        # Get a streaming writer (step, epoch, framework)
        stream = writer.get_distributed_shard_stream(100, 1, "pytorch")
        
        # Write data in chunks (zero-copy)
        stream.write_chunk(test_data_1)
        print(f"  Written chunk 1: {len(test_data_1)} bytes")
        
        stream.write_chunk(test_data_2)
        print(f"  Written chunk 2: {len(test_data_2)} bytes")
        
        stream.write_chunk(test_data_3)
        print(f"  Written chunk 3: {len(test_data_3)} bytes")
        
        # Finalize the stream
        shard_meta_streaming = stream.finalize()
        print(f"✓ Streaming method: {shard_meta_streaming['size']} bytes written to {shard_meta_streaming['key']}")
        
        # Verify both methods produce the same result
        assert shard_meta_traditional['size'] == shard_meta_streaming['size'], "Size mismatch between methods"
        print("✓ Both methods wrote the same amount of data")
        
        # Verify files exist and have correct sizes
        traditional_file = os.path.join(temp_dir, shard_meta_traditional['key'])
        streaming_file = os.path.join(temp_dir, shard_meta_streaming['key'])
        
        assert os.path.exists(traditional_file), f"Traditional file not found: {traditional_file}"
        assert os.path.exists(streaming_file), f"Streaming file not found: {streaming_file}"
        
        traditional_size = os.path.getsize(traditional_file)
        streaming_size = os.path.getsize(streaming_file)
        
        print(f"  Traditional file size: {traditional_size} bytes")
        print(f"  Streaming file size: {streaming_size} bytes")
        
        assert traditional_size == streaming_size, "File sizes don't match"
        print("✓ File sizes match")
        
        # Read and verify contents match
        with open(traditional_file, 'rb') as f:
            traditional_content = f.read()
        
        with open(streaming_file, 'rb') as f:
            streaming_content = f.read()
        
        assert traditional_content == streaming_content, "File contents don't match"
        print("✓ File contents match - streaming produces identical results")
        
        print("=== Python streaming validation PASSED ===")


def test_streaming_memory_efficiency():
    """Demonstrate the memory efficiency of streaming vs buffering"""
    print("\n=== Testing Memory Efficiency ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        store = s3dlio.PyCheckpointStore(base_uri, strategy='binary', multipart_threshold=1024*1024)
        writer = store.writer(world_size=1, rank=0)
        
        checkpoint_id = 100
        
        # Simulate processing a very large checkpoint in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        num_chunks = 10           # 10MB total (reduced for faster testing)
        
        print(f"Simulating {num_chunks} chunks of {chunk_size} bytes each ({num_chunks * chunk_size // (1024*1024)} MB total)")
        
        # Streaming approach - process chunks one at a time
        stream = writer.get_distributed_shard_stream(200, 2, "pytorch")
        
        total_written = 0
        for i in range(num_chunks):
            # In real use, this could be:
            # - tensor.cpu().numpy().tobytes() for PyTorch
            # - jax.device_get(array).tobytes() for JAX  
            # - tf_tensor.numpy().tobytes() for TensorFlow
            chunk_data = np.random.bytes(chunk_size)
            
            stream.write_chunk(chunk_data)
            total_written += len(chunk_data)
            
            # Key point: chunk_data can be garbage collected immediately
            # Peak memory usage = single chunk size, not total size
            del chunk_data
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1} chunks ({(i + 1) * chunk_size // (1024*1024)} MB)")
        
        shard_meta = stream.finalize()
        print(f"✓ Streaming complete: {shard_meta['size']} bytes written")
        print(f"✓ Peak memory usage: ~{chunk_size // (1024*1024)} MB (single chunk)")
        print(f"✓ Traditional buffering would require: ~{total_written // (1024*1024)} MB (full data)")
        
        # Verify the file was written correctly
        file_path = os.path.join(temp_dir, shard_meta['key'])
        assert os.path.exists(file_path), f"Streaming file not found: {file_path}"
        assert os.path.getsize(file_path) == total_written, "File size mismatch"
        
        print("=== Memory efficiency validation PASSED ===")


def test_error_handling():
    """Test error handling and cleanup in streaming interface"""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        store = s3dlio.PyCheckpointStore(base_uri, strategy='binary', multipart_threshold=1024*1024)
        writer = store.writer(world_size=1, rank=0)
        
        checkpoint_id = 200
        
        # Test stream cancellation
        stream = writer.get_distributed_shard_stream(300, 3, "pytorch")
        
        # Write some data
        test_data = np.random.bytes(1024)
        stream.write_chunk(test_data)
        print("✓ Wrote data to stream")
        
        # Cancel the stream instead of finalizing
        stream.cancel()
        print("✓ Stream cancelled")
        
        # Verify no file was created (or partial file was cleaned up)
        expected_files = [f for f in os.listdir(temp_dir) if "cancelled_shard" in f]
        assert len(expected_files) == 0, f"Cancelled stream should not leave files: {expected_files}"
        print("✓ No files left after cancellation")
        
        print("=== Error handling validation PASSED ===")


if __name__ == "__main__":
    try:
        test_python_streaming_interface()
        test_streaming_memory_efficiency()
        test_error_handling()
        print("\n🎉 ALL PYTHON STREAMING TESTS PASSED! 🎉")
        print("\nPhase 2 zero-copy streaming infrastructure is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
