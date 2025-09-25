#!/usr/bin/env python3
"""
Test the Python API with real S3 backend using different data generation modes
"""

import s3dlio
import time

def test_streaming_mode():
    """Test streaming mode via Python API"""
    print("Testing streaming mode via Python API...")
    start_time = time.time()
    
    try:
        result = s3dlio.put(
            prefix="s3://test-python-api/python-streaming-{}.bin",
            num=3,
            template="test-{}-of-{}",
            size=4194304,  # 4MB
            data_gen_mode="streaming",
            chunk_size=65536
        )
        elapsed = time.time() - start_time
        print(f"Streaming mode completed in {elapsed:.2f} seconds")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Streaming mode failed: {e}")
        return False

def test_single_pass_mode():
    """Test single-pass mode via Python API"""
    print("\nTesting single-pass mode via Python API...")
    start_time = time.time()
    
    try:
        result = s3dlio.put(
            prefix="s3://test-python-api/python-singlepass-{}.bin",
            num=3,
            template="test-{}-of-{}",
            size=4194304,  # 4MB
            data_gen_mode="single-pass",
            chunk_size=65536
        )
        elapsed = time.time() - start_time
        print(f"Single-pass mode completed in {elapsed:.2f} seconds")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Single-pass mode failed: {e}")
        return False

def test_default_mode():
    """Test default mode (should be streaming) via Python API"""
    print("\nTesting default mode (should be streaming) via Python API...")
    start_time = time.time()
    
    try:
        result = s3dlio.put(
            prefix="s3://test-python-api/python-default-{}.bin",
            num=3,
            template="test-{}-of-{}",
            size=4194304  # 4MB - using defaults for data_gen_mode and chunk_size
        )
        elapsed = time.time() - start_time
        print(f"Default mode completed in {elapsed:.2f} seconds")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Default mode failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing s3dlio Python API with real S3 backend...")
    print("=" * 60)
    
    streaming_ok = test_streaming_mode()
    single_pass_ok = test_single_pass_mode()
    default_ok = test_default_mode()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Streaming mode: {'✓ PASS' if streaming_ok else '✗ FAIL'}")
    print(f"Single-pass mode: {'✓ PASS' if single_pass_ok else '✗ FAIL'}")
    print(f"Default mode: {'✓ PASS' if default_ok else '✗ FAIL'}")
    
    if all([streaming_ok, single_pass_ok, default_ok]):
        print("\n🎉 All tests passed! Data generation modes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")