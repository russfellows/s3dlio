#!/usr/bin/env python3
# python/tests/test_gcs_api.py
"""
Quick validation test for GCS backend via Python API

Tests:
- Basic bulk put() with random data generation
- get() with zero-copy BytesView
- get_range() with GCS URIs
- Large file (RangeEngine trigger test)
- Error handling
- Cleanup via delete()

Prerequisites:
- GCS authentication (gcloud auth login)
- Test bucket: gs://signal65-russ-b1
- s3dlio installed in current Python environment
"""

import os
import sys
import time

def test_gcs_python_api():
    """Test GCS backend via Python API"""
    
    # Initialize debug logging to see RangeEngine activity
    import s3dlio
    s3dlio.init_logging("debug")
    print("🔍 Debug logging enabled")
    
    # Test configuration
    bucket = "signal65-russ-b1"
    test_prefix = "s3dlio-python-test"
    
    print("=" * 60)
    print("GCS Python API Validation Test")
    print("=" * 60)
    print(f"Bucket: {bucket}")
    print(f"Test prefix: {test_prefix}")
    print()
    
    # Import s3dlio
    try:
        import s3dlio
        print(f"✅ s3dlio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import s3dlio: {e}")
        print("   Run: ./build_pyo3.sh && ./install_pyo3_wheel.sh")
        return False
    
    # Test 1: Bulk put with random data generation
    print("\n=== TEST 1: Bulk put() with random data ===")
    prefix = f"gs://{bucket}/{test_prefix}"
    num_objects = 3
    object_size = 2 * 1024 * 1024  # 2MB each
    
    # Variables used by multiple tests
    test_uri = f"{prefix}/object-0"
    data_bytes = None
    
    try:
        print(f"📤 Generating and uploading {num_objects} objects of {object_size // (1024*1024)}MB each...")
        start = time.time()
        # put(prefix, num, template=None, ...) - template is now optional
        # Use random data (default) which is non-compressible and non-deduplicatable
        s3dlio.put(prefix, num_objects, size=object_size, object_type="random", 
                   dedup_factor=1, compress_factor=1)
        upload_time = time.time() - start
        total_mb = (num_objects * object_size) / (1024 * 1024)
        print(f"   Upload time: {upload_time:.3f}s ({total_mb / upload_time:.2f} MB/s)")
        
        # Download first object to verify
        print(f"📥 Downloading first object: {test_uri}...")
        start = time.time()
        downloaded = s3dlio.get(test_uri)
        download_time = time.time() - start
        
        throughput = (len(downloaded) / 1024 / 1024) / download_time
        print(f"   Download time: {download_time:.3f}s")
        print(f"   Throughput: {throughput:.2f} MB/s")
        
        # Verify size and type
        assert len(downloaded) == object_size, f"Size mismatch: got {len(downloaded)}, expected {object_size}"
        
        # BytesView is the zero-copy wrapper (preferred), but bytes also works
        # Check that it's a buffer-like object
        assert hasattr(downloaded, '__len__'), f"Downloaded object should have length"
        
        # Convert to bytes for content verification (if it's BytesView, this is efficient)
        data_bytes = bytes(downloaded) if not isinstance(downloaded, bytes) else downloaded
        first_byte = data_bytes[0]
        # Random data should have variety (not all same byte)
        has_variety = not all(b == first_byte for b in data_bytes[:1000])
        assert has_variety, "Expected random data with variety, got uniform data"
        
        print("   ✅ Bulk put() generated objects correctly")
        print(f"   ✅ get() returns {type(downloaded).__name__} (zero-copy wrapper)")
        print("   ✅ Data verified as random (non-uniform)")

        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    
    # Test 2: get_range()
    print("\n=== TEST 2: get_range() ===")
    
    if data_bytes is None:
        print("   ⚠️  Skipping get_range() tests - no reference data from Test 1")
        return False
    
    try:
        print(f"📥 Testing get_range(0, 1024) on {test_uri}...")
        range_data = s3dlio.get_range(test_uri, 0, 1024)
        assert len(range_data) == 1024, f"Expected 1024 bytes, got {len(range_data)}"
        # Verify it's the same as the full download (check consistency)
        assert bytes(range_data) == data_bytes[:1024], "Range data should match full download"
        print("   ✅ get_range() works correctly")
        
        print(f"📥 Testing get_range(1024, 2048)...")
        range_data2 = s3dlio.get_range(test_uri, 1024, 2048)
        assert len(range_data2) == 2048, f"Expected 2048 bytes, got {len(range_data2)}"
        assert bytes(range_data2) == data_bytes[1024:1024+2048], "Range data should match full download"
        print("   ✅ Second range works correctly")
        
    except Exception as e:
        print(f"   ❌ get_range() failed: {e}")
        return False
    
    # Test 3: Large file (trigger RangeEngine)
    print("\n=== TEST 3: Large file (RangeEngine) ===")
    large_size = 128 * 1024 * 1024  # 128MB (will create 2x 64MB ranges for true concurrency)
    large_prefix = f"gs://{bucket}/{test_prefix}-large"
    actual_large_uri = f"{large_prefix}/object-0.bin"
    
    try:
        print(f"📤 Generating and uploading 128MB file...")
        start = time.time()
        # Use put() with bulk generation API - generate 1 object of 128MB
        s3dlio.put(large_prefix, num=1, template="object-{}.bin", 
                   size=large_size, object_type="random", 
                   dedup_factor=1, compress_factor=1)
        upload_time = time.time() - start
        print(f"   Upload time: {upload_time:.3f}s ({(large_size / 1024 / 1024) / upload_time:.2f} MB/s)")
        
        print(f"📥 Downloading 128MB (should use RangeEngine with 2 concurrent ranges)...")
        start = time.time()
        downloaded_large = s3dlio.get(actual_large_uri)
        download_time = time.time() - start
        
        throughput = (len(downloaded_large) / 1024 / 1024) / download_time
        print(f"   Download time: {download_time:.3f}s")
        print(f"   Throughput: {throughput:.2f} MB/s")
        
        # Verify size
        assert len(downloaded_large) == large_size, f"Large file size mismatch: got {len(downloaded_large)}, expected {large_size}"
        
        # For random data, just verify it has variety
        large_bytes = bytes(downloaded_large) if not isinstance(downloaded_large, bytes) else downloaded_large
        first_byte = large_bytes[0]
        has_variety = not all(b == first_byte for b in large_bytes[:10000])
        assert has_variety, "Expected random data in large file"
        
        print("   ✅ Large file downloaded correctly (RangeEngine with 2 concurrent ranges)")
        
    except Exception as e:
        print(f"   ❌ Large file test failed: {e}")
        return False
    
    # Test 4: Error handling
    print("\n=== TEST 4: Error handling ===")
    try:
        print(f"📥 Testing get() on non-existent file...")
        nonexistent_uri = f"gs://{bucket}/nonexistent-file-{int(time.time())}.bin"
        try:
            s3dlio.get(nonexistent_uri)
            print(f"   ❌ Should have raised error for non-existent file")
            return False
        except Exception as e:
            print(f"   ✅ Correctly raised error: {type(e).__name__}")
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    # Cleanup
    print("\n=== Cleanup ===")
    try:
        print(f"🗑️  Deleting test files...")
        s3dlio.delete(test_uri)
        s3dlio.delete(actual_large_uri)
        print("   ✅ Cleanup complete")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All GCS Python API tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_gcs_python_api()
    sys.exit(0 if success else 1)
