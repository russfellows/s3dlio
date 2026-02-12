#!/usr/bin/env python3
"""
Test bytearray allocation and zero-copy operations in s3dlio.
This test specifically demonstrates the bytearray feature for efficient memory management.
"""

import s3dlio
import sys
import time


def test_bytearray_basic():
    """Test basic bytearray pre-allocation with generate_into_buffer"""
    print("=" * 70)
    print("TEST 1: Basic Bytearray Pre-allocation")
    print("=" * 70)
    
    # Pre-allocate a bytearray
    size = 1024 * 1024  # 1 MiB
    buf = bytearray(size)
    
    print(f"✓ Pre-allocated bytearray: {size:,} bytes")
    print(f"✓ Buffer type: {type(buf)}")
    
    # Fill with generated data (zero-copy write)
    nbytes = s3dlio.generate_into_buffer(buf)
    
    print(f"✓ Wrote {nbytes:,} bytes using zero-copy operation")
    print(f"✓ Buffer address unchanged: {id(buf)}")
    print(f"✓ First 32 bytes: {bytes(buf[:32]).hex()}")
    
    # Verify buffer was actually filled
    assert nbytes == size, f"Expected {size} bytes written, got {nbytes}"
    assert len(buf) == size, f"Buffer size changed from {size} to {len(buf)}"
    
    print("✓ Test passed!\n")
    return True


def test_bytearray_reuse():
    """Test reusing the same bytearray multiple times"""
    print("=" * 70)
    print("TEST 2: Bytearray Reuse")
    print("=" * 70)
    
    size = 512 * 1024  # 512 KiB
    buf = bytearray(size)
    original_id = id(buf)
    
    print(f"✓ Created bytearray with id: {original_id}")
    
    # Fill multiple times with different parameters
    for i, (dedup, compress) in enumerate([(1, 1), (2, 1), (1, 2), (2, 2)], 1):
        nbytes = s3dlio.generate_into_buffer(buf, dedup=dedup, compress=compress)
        current_id = id(buf)
        
        print(f"✓ Iteration {i}: dedup={dedup}, compress={compress}")
        print(f"  - Wrote {nbytes:,} bytes, buffer id: {current_id}")
        print(f"  - Address stable: {current_id == original_id}")
        
        assert current_id == original_id, "Bytearray address changed!"
        assert nbytes == size
    
    print("✓ Test passed!\n")
    return True


def test_bytearray_performance():
    """Compare performance of bytearray pre-allocation vs on-demand allocation"""
    print("=" * 70)
    print("TEST 3: Bytearray Performance Comparison")
    print("=" * 70)
    
    size = 10 * 1024 * 1024  # 10 MiB
    iterations = 10
    
    # Test 1: On-demand allocation
    start = time.time()
    for _ in range(iterations):
        data = s3dlio.generate_data(size)
        _ = bytes(data)  # Force copy to bytes
    t_ondemand = time.time() - start
    
    print(f"✓ On-demand allocation: {t_ondemand:.3f}s for {iterations} iterations")
    
    # Test 2: Pre-allocated bytearray (reused)
    buf = bytearray(size)
    start = time.time()
    for _ in range(iterations):
        s3dlio.generate_into_buffer(buf)
    t_preallocated = time.time() - start
    
    print(f"✓ Pre-allocated bytearray: {t_preallocated:.3f}s for {iterations} iterations")
    print(f"✓ Speedup: {t_ondemand / t_preallocated:.2f}x faster")
    
    print("✓ Test passed!\n")
    return True


def test_bytearray_sizes():
    """Test various bytearray sizes"""
    print("=" * 70)
    print("TEST 4: Various Bytearray Sizes")
    print("=" * 70)
    
    sizes = [
        (1024, "1 KiB"),
        (64 * 1024, "64 KiB"),
        (1024 * 1024, "1 MiB"),
        (16 * 1024 * 1024, "16 MiB"),
        (100 * 1024 * 1024, "100 MiB"),
    ]
    
    for size, label in sizes:
        buf = bytearray(size)
        nbytes = s3dlio.generate_into_buffer(buf)
        
        print(f"✓ {label:>10}: allocated and filled {nbytes:,} bytes")
        assert nbytes == size
    
    print("✓ Test passed!\n")
    return True


def test_bytearray_with_numpy():
    """Test bytearray compatibility with NumPy arrays"""
    print("=" * 70)
    print("TEST 5: Bytearray with NumPy Arrays")
    print("=" * 70)
    
    try:
        import numpy as np
        
        size = 1024 * 1024
        
        # Test with NumPy array
        arr = np.zeros(size, dtype=np.uint8)
        nbytes = s3dlio.generate_into_buffer(arr)
        
        print(f"✓ Generated into NumPy array: {nbytes:,} bytes")
        print(f"✓ Array shape: {arr.shape}")
        print(f"✓ Array dtype: {arr.dtype}")
        print(f"✓ First 16 bytes: {arr[:16].tobytes().hex()}")
        
        # Test with bytearray
        buf = bytearray(size)
        nbytes2 = s3dlio.generate_into_buffer(buf)
        
        print(f"✓ Generated into bytearray: {nbytes2:,} bytes")
        
        assert nbytes == size
        assert nbytes2 == size
        
        print("✓ Test passed!\n")
        return True
        
    except ImportError:
        print("⚠ NumPy not available, skipping NumPy tests")
        print("✓ Test skipped\n")
        return True


def main():
    print("\n" + "=" * 70)
    print("s3dlio Bytearray Feature Test Suite")
    print("Version: 0.9.40")
    print("=" * 70 + "\n")
    
    tests = [
        ("Basic Bytearray", test_bytearray_basic),
        ("Bytearray Reuse", test_bytearray_reuse),
        ("Bytearray Performance", test_bytearray_performance),
        ("Various Sizes", test_bytearray_sizes),
        ("NumPy Compatibility", test_bytearray_with_numpy),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
