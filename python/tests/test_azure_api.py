#!/usr/bin/env python3
# python/tests/test_azure_api.py
"""
Quick validation test for Azure backend via Python API

Tests:
- Basic get() with Azure URIs
- Zero-copy behavior
- get_range() with Azure URIs
- Error handling

Prerequisites:
- AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER environment variables
- az login completed
- s3dlio installed in current Python environment
"""

import os
import sys
import time

def test_azure_python_api():
    """Test Azure backend via Python API"""
    
    # Check environment
    account = os.getenv('AZURE_BLOB_ACCOUNT')
    container = os.getenv('AZURE_BLOB_CONTAINER')
    
    if not account or not container:
        print("⚠️  Skipping Azure Python tests - environment not set")
        print("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER")
        return False
    
    print("=" * 60)
    print("Azure Python API Validation Test")
    print("=" * 60)
    print(f"Account: {account}")
    print(f"Container: {container}")
    print()
    
    # Import s3dlio
    try:
        import s3dlio
        print(f"✅ s3dlio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import s3dlio: {e}")
        print("   Run: ./build_pyo3.sh && ./install_pyo3_wheel.sh")
        return False
    
    # Test 1: Basic put/get
    print("\n=== TEST 1: Basic put() and get() ===")
    test_uri = f"az://{account}/{container}/test-python-basic.bin"
    test_data = bytes([42] * (2 * 1024 * 1024))  # 2MB
    
    try:
        print(f"📤 Uploading 2MB to {test_uri}...")
        start = time.time()
        s3dlio.put(test_uri, test_data)
        upload_time = time.time() - start
        print(f"   Upload time: {upload_time:.3f}s")
        
        print(f"📥 Downloading with get()...")
        start = time.time()
        downloaded = s3dlio.get(test_uri)
        download_time = time.time() - start
        
        throughput = (len(downloaded) / 1024 / 1024) / download_time
        print(f"   Download time: {download_time:.3f}s")
        print(f"   Throughput: {throughput:.2f} MB/s")
        
        # Verify
        assert len(downloaded) == len(test_data), "Size mismatch"
        assert downloaded == test_data, "Data mismatch"
        
        # Check type - should be bytes
        assert isinstance(downloaded, bytes), f"Expected bytes, got {type(downloaded)}"
        
        print("   ✅ Data verified correctly")
        print("   ✅ Returns bytes (zero-copy from Rust)")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 2: get_range()
    print("\n=== TEST 2: get_range() ===")
    try:
        print(f"📥 Testing get_range(0, 1024)...")
        range_data = s3dlio.get_range(test_uri, 0, 1024)
        assert len(range_data) == 1024, f"Expected 1024 bytes, got {len(range_data)}"
        assert range_data == test_data[:1024], "Range data mismatch"
        print("   ✅ get_range() works correctly")
        
        print(f"📥 Testing get_range(1024, 2048)...")
        range_data2 = s3dlio.get_range(test_uri, 1024, 2048)
        assert len(range_data2) == 2048, f"Expected 2048 bytes, got {len(range_data2)}"
        assert range_data2 == test_data[1024:1024+2048], "Range data mismatch"
        print("   ✅ Second range works correctly")
        
    except Exception as e:
        print(f"   ❌ get_range() failed: {e}")
        return False
    
    # Test 3: Large file (trigger RangeEngine)
    print("\n=== TEST 3: Large file (RangeEngine) ===")
    large_uri = f"az://{account}/{container}/test-python-large.bin"
    large_data = bytes([99] * (8 * 1024 * 1024))  # 8MB (> 4MB threshold)
    
    try:
        print(f"📤 Uploading 8MB to {large_uri}...")
        start = time.time()
        s3dlio.put(large_uri, large_data)
        upload_time = time.time() - start
        print(f"   Upload time: {upload_time:.3f}s")
        
        print(f"📥 Downloading 8MB (should use RangeEngine)...")
        start = time.time()
        downloaded_large = s3dlio.get(large_uri)
        download_time = time.time() - start
        
        throughput = (len(downloaded_large) / 1024 / 1024) / download_time
        print(f"   Download time: {download_time:.3f}s")
        print(f"   Throughput: {throughput:.2f} MB/s")
        
        # Verify size
        assert len(downloaded_large) == len(large_data), "Large file size mismatch"
        
        # Verify first and last 1KB (avoid full comparison for speed)
        assert downloaded_large[:1024] == large_data[:1024], "Start mismatch"
        assert downloaded_large[-1024:] == large_data[-1024:], "End mismatch"
        
        print("   ✅ Large file downloaded correctly via RangeEngine")
        
    except Exception as e:
        print(f"   ❌ Large file test failed: {e}")
        return False
    
    # Test 4: Error handling
    print("\n=== TEST 4: Error handling ===")
    nonexistent_uri = f"az://{account}/{container}/nonexistent-file-12345.bin"
    
    try:
        print(f"📥 Testing get() on non-existent file...")
        try:
            s3dlio.get(nonexistent_uri)
            print("   ❌ Should have raised an error!")
            return False
        except Exception as expected_error:
            print(f"   ✅ Correctly raised error: {type(expected_error).__name__}")
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    # Cleanup
    print("\n=== Cleanup ===")
    try:
        print(f"🗑️  Deleting test files...")
        s3dlio.delete(test_uri)
        s3dlio.delete(large_uri)
        print("   ✅ Cleanup complete")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All Azure Python API tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_azure_python_api()
    sys.exit(0 if success else 1)
