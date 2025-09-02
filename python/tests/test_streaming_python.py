#!/usr/bin/env python3
"""
Quick test to verify our current streaming API works correctly.

This is a simplified version that tests the core functionality without
getting into the advanced O_DIRECT + compression challenges.
"""

import tempfile
import os
import sys
from pathlib import Path

import s3dlio

def test_basic_streaming():
    """Test basic streaming functionality that we know works."""
    print("🧪 Testing basic streaming API...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "basic_streaming_test.bin"
        test_uri = f"file://{test_file}"
        
        # Test basic streaming without compression
        options = s3dlio.PyWriterOptions()
        writer = s3dlio.create_filesystem_writer(test_uri, options)
        
        # Write test data
        chunk1 = b"Hello streaming world! " * 100  # 2300 bytes
        chunk2 = b"Second chunk of data. " * 50     # 1100 bytes
        
        writer.write_owned_bytes(chunk1)
        writer.write_owned_bytes(chunk2)
        
        # Check stats
        bytes_written = writer.bytes_written()
        expected_bytes = len(chunk1) + len(chunk2)
        assert bytes_written == expected_bytes
        
        # Finalize
        stats = writer.finalize()
        print(f"  ✅ Written: {bytes_written} bytes, Stats: {stats}")
        
        # Verify file
        assert test_file.exists()
        file_size = test_file.stat().st_size
        assert file_size == expected_bytes
        print(f"  ✅ File size correct: {file_size} bytes")
        
    print("✅ Basic streaming test passed!")
    return True

def test_compression_streaming():
    """Test streaming with compression."""
    print("🧪 Testing compression streaming...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "compression_test"
        test_uri = f"file://{test_file}"
        
        # Test with compression
        options = s3dlio.PyWriterOptions()
        options.with_compression("zstd", 3)
        writer = s3dlio.create_filesystem_writer(test_uri, options)
        
        # Write highly compressible data
        data = b"A" * 8192  # Very compressible
        writer.write_owned_bytes(data)
        
        # Finalize
        stats = writer.finalize()
        print(f"  ✅ Compression stats: {stats}")
        
        # Check compressed file (should have .zst extension)
        compressed_file = Path(str(test_file) + ".zst")
        assert compressed_file.exists()
        
        compressed_size = compressed_file.stat().st_size
        print(f"  ✅ Original: {len(data)} bytes → Compressed: {compressed_size} bytes")
        
        # Verify significant compression
        compression_ratio = len(data) / compressed_size
        print(f"  ✅ Compression ratio: {compression_ratio:.1f}x")
        assert compression_ratio > 10  # Should compress very well
        
    print("✅ Compression streaming test passed!")
    return True

def main():
    """Run basic streaming tests to verify API works."""
    print("=" * 50)
    print("🧪 S3DLIO Basic Streaming Verification")
    print("=" * 50)
    
    tests = [
        test_basic_streaming,
        test_compression_streaming,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Basic streaming API is working correctly!")
        return True
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total} passed)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
