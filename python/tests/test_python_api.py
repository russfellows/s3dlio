#!/usr/bin/env python3
"""
Test script for the new DataGenMode configuration in the Python API.
"""

import sys
import time
sys.path.insert(0, 'python')

try:
    from s3dlio import put
    print("✓ Successfully imported s3dlio.put")
except ImportError as e:
    print(f"✗ Failed to import s3dlio: {e}")
    sys.exit(1)

def test_data_gen_modes():
    """Test different data generation modes"""
    
    print("\n=== Testing DataGenMode Configuration ===")
    
    # Test default mode (should be Streaming)
    print("\n1. Testing default mode (Streaming):")
    try:
        result = put(
            "s3://test-bucket/test-object-default-{}.bin",  # template
            num=1,  # number of objects
            template="default-{}",  # template for naming
            size=1024 * 1024,  # 1MB
            object_type="binary",
            dedup_factor=1,
            compress_factor=1
        )
        print(f"   ✓ Default mode test completed: {result}")
    except Exception as e:
        print(f"   ✗ Default mode test failed: {e}")
    
    # Test explicit Streaming mode
    print("\n2. Testing explicit Streaming mode:")
    try:
        result = put(
            "s3://test-bucket/test-object-streaming-{}.bin",
            num=1,
            template="streaming-{}",
            size=1024 * 1024,  # 1MB
            object_type="binary",
            dedup_factor=1,
            compress_factor=1,
            data_gen_mode="streaming",
            chunk_size=8192
        )
        print(f"   ✓ Streaming mode test completed: {result}")
    except Exception as e:
        print(f"   ✗ Streaming mode test failed: {e}")
    
    # Test SinglePass mode
    print("\n3. Testing SinglePass mode:")
    try:
        result = put(
            "s3://test-bucket/test-object-singlepass-{}.bin",
            num=1,
            template="singlepass-{}",
            size=1024 * 1024,  # 1MB
            object_type="binary",
            dedup_factor=1,
            compress_factor=1,
            data_gen_mode="singlepass",
            chunk_size=4096
        )
        print(f"   ✓ SinglePass mode test completed: {result}")
    except Exception as e:
        print(f"   ✗ SinglePass mode test failed: {e}")
    
    # Test invalid mode
    print("\n4. Testing invalid mode (should fail gracefully):")
    try:
        result = put(
            "s3://test-bucket/test-object-invalid-{}.bin",
            num=1,
            template="invalid-{}",
            size=1024,
            object_type="binary",
            dedup_factor=1,
            compress_factor=1,
            data_gen_mode="invalidmode"
        )
        print(f"   ✗ Invalid mode test unexpectedly succeeded: {result}")
    except Exception as e:
        print(f"   ✓ Invalid mode correctly failed: {e}")

def test_different_object_types():
    """Test with different object types"""
    
    print("\n=== Testing Different Object Types with Streaming ===")
    
    object_types = ["binary", "hdf5", "npz", "tfrecord"]
    
    for obj_type in object_types:
        print(f"\n   Testing {obj_type} with Streaming mode:")
        try:
            result = put(
                f"s3://test-bucket/test-{obj_type}-streaming-{{}}.bin",
                num=1,
                template=f"{obj_type}-{{}}",
                size=512 * 1024,  # 512KB
                object_type=obj_type,
                dedup_factor=2,
                compress_factor=2,
                data_gen_mode="streaming",
                chunk_size=4096
            )
            print(f"     ✓ {obj_type} test completed: {result}")
        except Exception as e:
            print(f"     ✗ {obj_type} test failed: {e}")

def test_performance_comparison():
    """Test performance difference between modes"""
    
    print("\n=== Performance Comparison Test ===")
    
    size = 4 * 1024 * 1024  # 4MB - good size for seeing streaming benefits
    
    # Test Streaming mode
    print("\n   Testing Streaming mode performance:")
    start_time = time.time()
    try:
        result = put(
            "s3://test-bucket/perf-streaming-{}.bin",
            num=1,
            template="perf-streaming-{}",
            size=size,
            object_type="binary",
            dedup_factor=1,
            compress_factor=1,
            data_gen_mode="streaming",
            chunk_size=8192
        )
        streaming_time = time.time() - start_time
        print(f"     ✓ Streaming completed in {streaming_time:.3f}s: {result}")
    except Exception as e:
        print(f"     ✗ Streaming test failed: {e}")
        streaming_time = None
    
    # Test SinglePass mode  
    print("\n   Testing SinglePass mode performance:")
    start_time = time.time()
    try:
        result = put(
            "s3://test-bucket/perf-singlepass-{}.bin",
            num=1,
            template="perf-singlepass-{}",
            size=size,
            object_type="binary", 
            dedup_factor=1,
            compress_factor=1,
            data_gen_mode="singlepass",
            chunk_size=8192
        )
        singlepass_time = time.time() - start_time
        print(f"     ✓ SinglePass completed in {singlepass_time:.3f}s: {result}")
    except Exception as e:
        print(f"     ✗ SinglePass test failed: {e}")
        singlepass_time = None
    
    # Compare performance
    if streaming_time and singlepass_time:
        ratio = singlepass_time / streaming_time
        faster_mode = "Streaming" if streaming_time < singlepass_time else "SinglePass"
        print(f"\n   📊 Performance Summary:")
        print(f"     Streaming: {streaming_time:.3f}s")
        print(f"     SinglePass: {singlepass_time:.3f}s")
        print(f"     {faster_mode} is {abs(ratio - 1) * 100:.1f}% faster")

if __name__ == "__main__":
    print("🚀 Starting s3dlio Python API test...")
    
    test_data_gen_modes()
    test_different_object_types() 
    test_performance_comparison()
    
    print("\n🎉 Test completed!")