#!/usr/bin/env python3
"""
Test script to verify data generation methods in s3dlio Python API.
Tests: default (random), explicit random, and explicit prand methods.
"""

import s3dlio
import time
import os
import shutil
from pathlib import Path

# Test configuration
TEST_DIR = "/tmp/s3dlio_test_data_gen"
NUM_OBJECTS = 10
OBJECT_SIZE = 1024 * 1024  # 1 MB
MAX_IN_FLIGHT = 4

def setup_test_dir():
    """Create clean test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"✓ Created test directory: {TEST_DIR}")

def cleanup_test_dir():
    """Remove test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    print(f"✓ Cleaned up test directory")

def test_default_method():
    """Test PUT with default data generation method (should be 'random')."""
    print("\n" + "="*70)
    print("TEST 1: Default data generation method (random)")
    print("="*70)
    
    prefix = f"file://{TEST_DIR}/default/"
    
    start = time.time()
    s3dlio.put(
        prefix=prefix,
        num=NUM_OBJECTS,
        template="object-{}.dat",
        max_in_flight=MAX_IN_FLIGHT,
        size=OBJECT_SIZE,
        object_type="raw",
        dedup_factor=1,
        compress_factor=1
        # Note: NOT specifying data_gen_algorithm - should default to "random"
    )
    elapsed = time.time() - start
    
    # Verify files were created
    files = list(Path(f"{TEST_DIR}/default").glob("*.dat"))
    print(f"✓ Created {len(files)} objects in {elapsed:.3f}s")
    print(f"✓ Throughput: {(NUM_OBJECTS * OBJECT_SIZE / 1024**2) / elapsed:.2f} MB/s")
    
    assert len(files) == NUM_OBJECTS, f"Expected {NUM_OBJECTS} files, got {len(files)}"
    print("✓ Test PASSED")

def test_explicit_random():
    """Test PUT with explicit 'random' data generation method."""
    print("\n" + "="*70)
    print("TEST 2: Explicit 'random' data generation method")
    print("="*70)
    
    prefix = f"file://{TEST_DIR}/random/"
    
    start = time.time()
    s3dlio.put(
        prefix=prefix,
        num=NUM_OBJECTS,
        template="object-{}.dat",
        max_in_flight=MAX_IN_FLIGHT,
        size=OBJECT_SIZE,
        object_type="raw",
        dedup_factor=1,
        compress_factor=1,
        data_gen_algorithm="random"  # Explicitly specify random
    )
    elapsed = time.time() - start
    
    # Verify files were created
    files = list(Path(f"{TEST_DIR}/random").glob("*.dat"))
    print(f"✓ Created {len(files)} objects in {elapsed:.3f}s")
    print(f"✓ Throughput: {(NUM_OBJECTS * OBJECT_SIZE / 1024**2) / elapsed:.2f} MB/s")
    
    assert len(files) == NUM_OBJECTS, f"Expected {NUM_OBJECTS} files, got {len(files)}"
    print("✓ Test PASSED")

def test_explicit_prand():
    """Test PUT with explicit 'prand' data generation method."""
    print("\n" + "="*70)
    print("TEST 3: Explicit 'prand' data generation method")
    print("="*70)
    
    prefix = f"file://{TEST_DIR}/prand/"
    
    start = time.time()
    s3dlio.put(
        prefix=prefix,
        num=NUM_OBJECTS,
        template="object-{}.dat",
        max_in_flight=MAX_IN_FLIGHT,
        size=OBJECT_SIZE,
        object_type="raw",
        dedup_factor=1,
        compress_factor=1,
        data_gen_algorithm="prand"  # Explicitly specify prand
    )
    elapsed = time.time() - start
    
    # Verify files were created
    files = list(Path(f"{TEST_DIR}/prand").glob("*.dat"))
    print(f"✓ Created {len(files)} objects in {elapsed:.3f}s")
    print(f"✓ Throughput: {(NUM_OBJECTS * OBJECT_SIZE / 1024**2) / elapsed:.2f} MB/s")
    
    assert len(files) == NUM_OBJECTS, f"Expected {NUM_OBJECTS} files, got {len(files)}"
    print("✓ Test PASSED")

def test_prand_with_streaming():
    """Test PUT with prand + streaming mode."""
    print("\n" + "="*70)
    print("TEST 4: 'prand' with 'streaming' mode")
    print("="*70)
    
    prefix = f"file://{TEST_DIR}/prand_streaming/"
    
    start = time.time()
    s3dlio.put(
        prefix=prefix,
        num=NUM_OBJECTS,
        template="object-{}.dat",
        max_in_flight=MAX_IN_FLIGHT,
        size=OBJECT_SIZE,
        object_type="raw",
        dedup_factor=1,
        compress_factor=1,
        data_gen_algorithm="prand",
        data_gen_mode="streaming"  # Combine prand with streaming
    )
    elapsed = time.time() - start
    
    # Verify files were created
    files = list(Path(f"{TEST_DIR}/prand_streaming").glob("*.dat"))
    print(f"✓ Created {len(files)} objects in {elapsed:.3f}s")
    print(f"✓ Throughput: {(NUM_OBJECTS * OBJECT_SIZE / 1024**2) / elapsed:.2f} MB/s")
    
    assert len(files) == NUM_OBJECTS, f"Expected {NUM_OBJECTS} files, got {len(files)}"
    print("✓ Test PASSED")

def main():
    """Run all tests."""
    print("="*70)
    print("s3dlio Data Generation Methods Test Suite")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Objects per test: {NUM_OBJECTS}")
    print(f"  - Object size: {OBJECT_SIZE / 1024**2:.1f} MB")
    print(f"  - Max in-flight: {MAX_IN_FLIGHT}")
    print(f"  - Test directory: {TEST_DIR}")
    
    try:
        setup_test_dir()
        
        # Run all tests
        test_default_method()
        test_explicit_random()
        test_explicit_prand()
        test_prand_with_streaming()
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nSummary:")
        print("  ✓ Default method (random) works")
        print("  ✓ Explicit 'random' method works")
        print("  ✓ Explicit 'prand' method works")
        print("  ✓ Combined 'prand' + 'streaming' mode works")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_test_dir()
    
    return 0

if __name__ == "__main__":
    exit(main())
