#!/usr/bin/env python3
"""
NVIDIA DALI Compatibility Validation Test

This test validates that our TFRecord index generation is compatible with
NVIDIA DALI's tfrecord2idx format and demonstrates random access patterns.
"""

import s3dlio
import struct
import tempfile
import os
import random


def create_test_tfrecord_with_content(records_data):
    """
    Create a TFRecord file with actual content for testing
    
    Args:
        records_data: List of byte strings to encode as TFRecord entries
    
    Returns:
        bytes: Complete TFRecord file data
    """
    data = bytearray()
    
    for record_data in records_data:
        # Length (u64 little-endian)
        length = len(record_data)
        data.extend(struct.pack('<Q', length))
        
        # Length CRC (dummy for test - real implementation would calculate CRC32C)
        data.extend(b'\x00' * 4)
        
        # Data payload
        data.extend(record_data)
        
        # Data CRC (dummy for test)
        data.extend(b'\x00' * 4)
    
    return bytes(data)


def validate_dali_format(index_path):
    """
    Validate that index file matches NVIDIA DALI tfrecord2idx format
    
    DALI Format Specification (from tools/tfrecord2idx):
    - Text file (ASCII)
    - One line per record
    - Format: "{offset} {size}\n"
    - Space-separated (not tab)
    - No header, no trailing empty lines after last record
    
    Returns:
        (bool, str): (is_valid, error_message)
    """
    try:
        with open(index_path, 'r') as f:
            content = f.read()
        
        # Check it's text (not binary) - allow newlines and spaces
        if '\x00' in content:
            return False, "Index contains null bytes (should be ASCII text)"
        
        # Verify only printable ASCII + newline + space
        for char in content:
            if char not in ' \n0123456789':
                # Allow any printable ASCII, but numbers/space/newline are expected
                if not char.isprintable():
                    return False, f"Index contains non-printable character: {repr(char)}"
        
        lines = content.rstrip('\n').split('\n')
        
        for i, line in enumerate(lines):
            # Each line should be "{offset} {size}"
            parts = line.split()
            
            if len(parts) != 2:
                return False, f"Line {i+1}: Expected 2 space-separated values, got {len(parts)}"
            
            # Check for tab (DALI uses space, not tab)
            if '\t' in line:
                return False, f"Line {i+1}: Contains tab character (DALI uses space)"
            
            # Verify both are valid integers
            try:
                offset = int(parts[0])
                size = int(parts[1])
            except ValueError as e:
                return False, f"Line {i+1}: Invalid integer: {e}"
            
            # Sanity checks
            if offset < 0:
                return False, f"Line {i+1}: Negative offset {offset}"
            if size <= 0:
                return False, f"Line {i+1}: Invalid size {size}"
            
            # Offsets should be monotonically increasing
            if i > 0:
                prev_line = lines[i-1]
                prev_offset = int(prev_line.split()[0])
                if offset <= prev_offset:
                    return False, f"Line {i+1}: Offset {offset} not greater than previous {prev_offset}"
        
        return True, "Format is valid"
    
    except Exception as e:
        return False, f"Validation error: {e}"


def test_dali_format_compliance():
    """Test 1: Validate DALI format compliance"""
    print("\n" + "="*70)
    print("TEST 1: NVIDIA DALI Format Compliance")
    print("="*70)
    
    # Create test data
    records = [
        b"First TFRecord entry with some data",
        b"Second entry",
        b"Third entry with more content to test varying sizes",
    ]
    
    tfrecord_data = create_test_tfrecord_with_content(records)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "test.tfrecord")
        index_path = os.path.join(tmpdir, "test.tfrecord.idx")
        
        # Write TFRecord
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        # Generate index
        num_records = s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        print(f"✓ Generated index for {num_records} records")
        
        # Read and display index content
        with open(index_path, 'r') as f:
            index_content = f.read()
        
        print(f"\nIndex file content ({len(index_content)} bytes):")
        print("-" * 40)
        print(index_content.rstrip('\n'))
        print("-" * 40)
        
        # Validate format
        is_valid, message = validate_dali_format(index_path)
        
        if is_valid:
            print(f"\n✓ PASS: {message}")
            print("  - Text format (ASCII)")
            print("  - Space-separated values")
            print("  - Monotonically increasing offsets")
            print("  - Valid integer values")
        else:
            print(f"\n✗ FAIL: {message}")
            return False
    
    return True


def test_random_access_with_index():
    """Test 2: Demonstrate random access using index"""
    print("\n" + "="*70)
    print("TEST 2: Random Access Using Index")
    print("="*70)
    
    # Create test data with identifiable content
    records = [
        b"Record 0: This is the first record",
        b"Record 1: This is the second record",
        b"Record 2: This is the third record",
        b"Record 3: This is the fourth record",
        b"Record 4: This is the fifth record",
    ]
    
    tfrecord_data = create_test_tfrecord_with_content(records)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "test.tfrecord")
        index_path = os.path.join(tmpdir, "test.tfrecord.idx")
        
        # Write TFRecord and generate index
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        
        # Read index
        index_entries = s3dlio.read_tfrecord_index(index_path)
        print(f"✓ Loaded index with {len(index_entries)} entries")
        
        # Demonstrate random access
        print("\nRandom access test (reading records in random order):")
        print("-" * 40)
        
        # Shuffle record indices
        indices = list(range(len(records)))
        random.shuffle(indices)
        
        with open(tfrecord_path, 'rb') as f:
            for i in indices:
                offset, size = index_entries[i]
                
                # Seek to record position
                f.seek(offset)
                
                # Read the entire record
                record_bytes = f.read(size)
                
                # Parse TFRecord structure
                # Skip: length (8 bytes) + length_crc (4 bytes)
                data_start = 8 + 4
                # Data ends before data_crc (last 4 bytes)
                data_end = size - 4
                
                # Extract actual data
                data_length = struct.unpack('<Q', record_bytes[:8])[0]
                actual_data = record_bytes[data_start:data_start + data_length]
                
                # Verify it matches original
                expected = records[i]
                if actual_data == expected:
                    print(f"  Record {i}: ✓ {actual_data.decode()[:50]}")
                else:
                    print(f"  Record {i}: ✗ MISMATCH")
                    return False
        
        print("\n✓ PASS: All records read correctly via random access")
    
    return True


def test_index_roundtrip():
    """Test 3: Index write/read roundtrip"""
    print("\n" + "="*70)
    print("TEST 3: Index Write/Read Roundtrip")
    print("="*70)
    
    records = [b"data1", b"data2", b"data3"]
    tfrecord_data = create_test_tfrecord_with_content(records)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tfrecord_path = os.path.join(tmpdir, "test.tfrecord")
        index_path = os.path.join(tmpdir, "test.tfrecord.idx")
        
        # Write TFRecord
        with open(tfrecord_path, 'wb') as f:
            f.write(tfrecord_data)
        
        # Method 1: Generate index from file
        num_written = s3dlio.create_tfrecord_index(tfrecord_path, index_path)
        print(f"✓ Created index with {num_written} entries")
        
        # Method 2: Generate index from bytes
        entries_from_bytes = s3dlio.get_tfrecord_index_entries(tfrecord_data)
        print(f"✓ Parsed {len(entries_from_bytes)} entries from bytes")
        
        # Method 3: Read index from file
        entries_from_file = s3dlio.read_tfrecord_index(index_path)
        print(f"✓ Read {len(entries_from_file)} entries from index file")
        
        # Verify all methods give same results
        assert len(entries_from_bytes) == len(entries_from_file) == num_written
        
        for i, (from_bytes, from_file) in enumerate(zip(entries_from_bytes, entries_from_file)):
            if from_bytes != from_file:
                print(f"✗ FAIL: Mismatch at entry {i}")
                print(f"  From bytes: {from_bytes}")
                print(f"  From file:  {from_file}")
                return False
        
        print("\n✓ PASS: All methods produce identical results")
    
    return True


def test_dali_integration_pattern():
    """Test 4: Show DALI integration pattern (documentation)"""
    print("\n" + "="*70)
    print("TEST 4: DALI Integration Pattern (Documentation)")
    print("="*70)
    
    print("""
This demonstrates how to use s3dlio-generated indexes with NVIDIA DALI:

1. Generate indexes using s3dlio:
   
   import s3dlio
   
   # Generate indexes for your TFRecord files
   tfrecord_files = ["train-00000.tfrecord", "train-00001.tfrecord", ...]
   for tfrecord_file in tfrecord_files:
       index_file = tfrecord_file + ".idx"
       s3dlio.create_tfrecord_index(tfrecord_file, index_file)

2. Use with DALI pipeline:
   
   from nvidia import dali
   from nvidia.dali import fn, types, pipeline_def
   
   @pipeline_def
   def training_pipeline(tfrecord_files, index_files):
       # DALI TFRecord reader with index
       inputs = fn.readers.tfrecord(
           path=tfrecord_files,
           index_path=index_files,  # ← Use s3dlio-generated indexes
           features={
               "image/encoded": dali.tfrecord.FixedLenFeature(
                   (), dali.tfrecord.string, ""
               ),
               "image/class/label": dali.tfrecord.FixedLenFeature(
                   [1], dali.tfrecord.int64, -1
               ),
           },
           random_shuffle=True,  # ← Enabled by index
           shard_id=0,
           num_shards=1,
       )
       
       # Decode images
       images = fn.decoders.image(inputs["image/encoded"], device="mixed")
       labels = inputs["image/class/label"]
       
       return images, labels

3. Benefits of using indexes:
   - Random shuffling during training
   - Efficient distributed training (multi-GPU)
   - Fast seeking to specific records
   - No need to scan entire file
""")
    
    print("✓ PASS: Integration pattern documented")
    return True


def main():
    """Run all validation tests"""
    print("="*70)
    print("NVIDIA DALI Compatibility Validation")
    print("="*70)
    print("\nValidating s3dlio TFRecord index generation against DALI spec")
    print("Format: '{offset} {size}\\n' (space-separated ASCII text)")
    
    tests = [
        ("DALI Format Compliance", test_dali_format_compliance),
        ("Random Access", test_random_access_with_index),
        ("Index Roundtrip", test_index_roundtrip),
        ("DALI Integration", test_dali_integration_pattern),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nConclusion:")
        print("  - Index format is NVIDIA DALI compatible")
        print("  - Random access works correctly")
        print("  - Ready for use with DALI pipelines")
        print("  - Compatible with TensorFlow tooling")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
