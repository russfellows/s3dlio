#!/usr/bin/env python3
"""
Test TFRecord index generation from Python

Demonstrates the new NVIDIA DALI-compatible TFRecord indexing functionality.
"""

import s3dlio
import struct
import tempfile
import os


def create_test_tfrecord(records):
    """Create a simple TFRecord file with test data"""
    data = bytearray()
    
    for record_data in records:
        # Length (u64 little-endian)
        length = len(record_data)
        data.extend(struct.pack('<Q', length))
        
        # Length CRC (dummy for test)
        data.extend(b'\x00' * 4)
        
        # Data payload
        data.extend(record_data)
        
        # Data CRC (dummy for test)
        data.extend(b'\x00' * 4)
    
    return bytes(data)


def test_index_tfrecord_bytes():
    """Test index_tfrecord_bytes function"""
    print("\n=== Test 1: index_tfrecord_bytes ===")
    
    # Create test data
    records = [b"first record", b"second record", b"third record"]
    tfrecord_data = create_test_tfrecord(records)
    
    # Generate index text
    index_text = s3dlio.index_tfrecord_bytes(tfrecord_data)
    
    print(f"TFRecord size: {len(tfrecord_data)} bytes")
    print(f"Index text:\n{index_text}")
    
    # Verify format
    lines = index_text.strip().split('\n')
    assert len(lines) == len(records), f"Expected {len(records)} index lines, got {len(lines)}"
    
    for i, line in enumerate(lines):
        parts = line.split(' ')
        assert len(parts) == 2, f"Line {i}: Expected 'offset size', got '{line}'"
        offset = int(parts[0])
        size = int(parts[1])
        print(f"Record {i}: offset={offset}, size={size}")
    
    print("✓ Test passed: index_tfrecord_bytes works correctly")


def test_get_tfrecord_index_entries():
    """Test get_tfrecord_index_entries function"""
    print("\n=== Test 2: get_tfrecord_index_entries ===")
    
    # Create test data
    records = [b"data1", b"data2", b"data3", b"data4"]
    tfrecord_data = create_test_tfrecord(records)
    
    # Get index entries
    entries = s3dlio.get_tfrecord_index_entries(tfrecord_data)
    
    print(f"Number of entries: {len(entries)}")
    assert len(entries) == len(records), f"Expected {len(records)} entries, got {len(entries)}"
    
    for i, (offset, size) in enumerate(entries):
        print(f"Entry {i}: offset={offset}, size={size}")
        assert isinstance(offset, int), "Offset must be int"
        assert isinstance(size, int), "Size must be int"
        assert offset >= 0, "Offset must be non-negative"
        assert size > 0, "Size must be positive"
    
    print("✓ Test passed: get_tfrecord_index_entries works correctly")


def test_create_tfrecord_index():
    """Test create_tfrecord_index function with file I/O"""
    print("\n=== Test 3: create_tfrecord_index ===")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "test.tfrecord")
        index_path = os.path.join(tmpdir, "test.tfrecord.idx")
        
        # Create test TFRecord file
        records = [
            b"sample record one",
            b"sample record two",
            b"sample record three with more data",
        ]
        tfrecord_data = create_test_tfrecord(records)
        
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        print(f"Created TFRecord file: {tfrecord_path}")
        print(f"File size: {len(tfrecord_data)} bytes")
        
        # Generate index
        num_records = s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        
        print(f"Generated index for {num_records} records")
        assert num_records == len(records), f"Expected {len(records)} records, got {num_records}"
        
        # Verify index file exists and has correct format
        assert os.path.exists(index_path), "Index file was not created"
        
        with open(index_path, 'r') as f:
            index_content = f.read()
        
        print(f"\nIndex file content:\n{index_content}")
        
        lines = index_content.strip().split('\n')
        assert len(lines) == len(records), f"Index has wrong number of lines"
        
        # Verify DALI-compatible format: "{offset} {size}\n"
        for line in lines:
            parts = line.split(' ')
            assert len(parts) == 2, f"Invalid format: expected 'offset size', got '{line}'"
            int(parts[0])  # Should not raise
            int(parts[1])  # Should not raise
        
        print("✓ Test passed: create_tfrecord_index works correctly")


def test_create_tfrecord_index_default_path():
    """Test create_tfrecord_index with default index path"""
    print("\n=== Test 4: create_tfrecord_index (default path) ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "mydata.tfrecord")
        
        # Create test TFRecord file
        records = [b"record1", b"record2"]
        tfrecord_data = create_test_tfrecord(records)
        
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        # Generate index without specifying output path (should default to input + ".idx")
        num_records = s3dlio.create_tfrecord_index(tfrecord_path)
        
        # Check that default path was created
        expected_index_path = tfrecord_path + ".idx"
        assert os.path.exists(expected_index_path), f"Index file not found at {expected_index_path}"
        
        print(f"Generated index at default path: {expected_index_path}")
        print(f"Indexed {num_records} records")
        
        print("✓ Test passed: default index path works correctly")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TFRecord Index Python API Tests")
    print("NVIDIA DALI Compatible Format: '{offset} {size}\\n'")
    print("=" * 60)
    
    try:
        test_index_tfrecord_bytes()
        test_get_tfrecord_index_entries()
        test_create_tfrecord_index()
        test_create_tfrecord_index_default_path()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe TFRecord indexing API is working correctly!")
        print("These index files are compatible with:")
        print("  - NVIDIA DALI fn.readers.tfrecord(index_path=...)")
        print("  - TensorFlow tooling")
        print("  - Python ML workflows")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
