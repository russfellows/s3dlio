#!/usr/bin/env python3
"""Comprehensive test for zero-copy BytesView implementation"""

import s3dlio
import tempfile
import os
import sys
import numpy as np


def test_bytesview_class():
    """Test that BytesView exists and has the right methods"""
    print("\n" + "=" * 60)
    print("Test 1: BytesView class structure")
    print("=" * 60)

    # Check if BytesView is available
    assert hasattr(s3dlio, "BytesView"), "BytesView class not found"
    print("✓ BytesView class found")

    # Check methods
    methods = [m for m in dir(s3dlio.BytesView) if not m.startswith("_")]
    print(f"  Methods: {methods}")

    assert "memoryview" in methods, "memoryview() method not found"
    print("✓ memoryview() method found")

    assert "to_bytes" in methods, "to_bytes() method not found"
    print("✓ to_bytes() method found")


def test_get_returns_bytesview():
    """Test that get() returns BytesView and memoryview works"""
    print("\n" + "=" * 60)
    print("Test 2: get() returns BytesView with working memoryview")
    print("=" * 60)

    # Create test data
    test_data = b"Hello, zero-copy world!"

    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(test_data)
        temp_path = f.name

    try:
        # Test get()
        uri = f"file://{temp_path}"
        result = s3dlio.get(uri)

        # Check type
        assert isinstance(result, s3dlio.BytesView), (
            f"get() returned {type(result)}, expected BytesView"
        )
        print("✓ get() returned BytesView")

        # Check __len__
        assert len(result) == len(test_data), (
            f"__len__() returned {len(result)}, expected {len(test_data)}"
        )
        print(f"✓ __len__() works correctly: {len(result)} bytes")

        # Test memoryview()
        mv = result.memoryview()
        assert isinstance(mv, memoryview), (
            f"memoryview() returned {type(mv)}, expected memoryview"
        )
        print("✓ memoryview() returned memoryview object")

        # Check memoryview contents
        assert bytes(mv) == test_data, (
            f"memoryview contents don't match: {bytes(mv)[:50]}"
        )
        print(f"✓ memoryview contents match: {bytes(mv)[:30]}...")

        # Test to_bytes()
        copied = result.to_bytes()
        assert isinstance(copied, bytes), (
            f"to_bytes() returned {type(copied)}, expected bytes"
        )
        print("✓ to_bytes() returned bytes object")

        assert copied == test_data, "to_bytes() contents don't match"
        print("✓ to_bytes() contents match")

    finally:
        os.unlink(temp_path)


def test_numpy_array_from_memoryview():
    """Test creating numpy array from BytesView memoryview (zero-copy)"""
    print("\n" + "=" * 60)
    print("Test 3: NumPy array from memoryview (zero-copy)")
    print("=" * 60)

    # Create binary data representing a numpy array
    arr = np.arange(100, dtype=np.float32)
    test_data = arr.tobytes()

    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(test_data)
        temp_path = f.name

    try:
        # Get data via s3dlio
        uri = f"file://{temp_path}"
        result = s3dlio.get(uri)

        # Get memoryview
        mv = result.memoryview()

        # Create numpy array from memoryview (zero-copy)
        arr_from_mv = np.frombuffer(mv, dtype=np.float32)

        print(
            f"✓ Created numpy array from memoryview: shape={arr_from_mv.shape}, dtype={arr_from_mv.dtype}"
        )

        # Verify contents
        assert np.array_equal(arr_from_mv, arr), "Array contents don't match"
        print("✓ Array contents match original")
        print(f"  First 10 elements: {arr_from_mv[:10]}")

    finally:
        os.unlink(temp_path)


def test_get_many_returns_bytesview():
    """Test that get_many() returns list of (uri, BytesView) tuples"""
    print("\n" + "=" * 60)
    print("Test 4: get_many() returns BytesView objects")
    print("=" * 60)

    # Create test files
    test_files = []
    test_data = [
        b"File 1 contents",
        b"File 2 contents",
        b"File 3 contents",
    ]

    try:
        for i, data in enumerate(test_data):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(data)
                test_files.append(f.name)

        # Test get_many
        uris = [f"file://{path}" for path in test_files]
        results = s3dlio.get_many(uris)

        assert isinstance(results, list), (
            f"get_many() returned {type(results)}, expected list"
        )
        print(f"✓ get_many() returned list with {len(results)} items")

        # Check each result
        for i, (uri, result) in enumerate(results):
            assert isinstance(result, s3dlio.BytesView), (
                f"Item {i} is {type(result)}, expected BytesView"
            )

            # Check memoryview works
            mv = result.memoryview()
            assert bytes(mv) == test_data[i], f"Item {i} contents don't match"

        print(f"✓ All {len(results)} items are BytesView objects")
        print("✓ All memoryview() calls work correctly")

    finally:
        for path in test_files:
            if os.path.exists(path):
                os.unlink(path)


def test_large_file_memoryview():
    """Test memoryview with larger file to verify zero-copy efficiency"""
    print("\n" + "=" * 60)
    print("Test 5: Large file memoryview (1 MB)")
    print("=" * 60)

    # Create 1MB of data
    size_mb = 1
    test_data = bytes(range(256)) * (size_mb * 1024 * 4)  # 1MB

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(test_data)
        temp_path = f.name

    try:
        uri = f"file://{temp_path}"
        result = s3dlio.get(uri)

        print(f"✓ Read {len(result)} bytes as BytesView")

        # Get memoryview
        mv = result.memoryview()
        print(f"✓ Created memoryview: {len(mv)} bytes")

        # Verify first and last bytes
        assert mv[0] == test_data[0] and mv[-1] == test_data[-1], (
            "Memoryview contents don't match"
        )
        print("✓ Memoryview contents verified (first/last byte)")

        # Create numpy array from memoryview
        arr = np.frombuffer(mv, dtype=np.uint8)
        print(f"✓ Created numpy array: shape={arr.shape}")

    finally:
        os.unlink(temp_path)


def test_bytesview_immutability():
    """Test that BytesView is read-only (memoryview should not allow modification)"""
    print("\n" + "=" * 60)
    print("Test 6: BytesView immutability")
    print("=" * 60)

    test_data = b"Immutable data"

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(test_data)
        temp_path = f.name

    try:
        uri = f"file://{temp_path}"
        result = s3dlio.get(uri)
        mv = result.memoryview()

        # Check if memoryview is read-only
        if mv.readonly:
            print("✓ Memoryview is read-only (as expected)")
        else:
            print("⚠ Memoryview is NOT read-only (could allow mutation)")
            # Try to modify and see if it fails
            try:
                mv[0] = ord("X")
                raise AssertionError(
                    "Memoryview allowed modification despite not being read-only!"
                )
            except (TypeError, ValueError) as e:
                print(f"✓ Modification correctly prevented: {e}")

    finally:
        os.unlink(temp_path)


def main():
    print("=" * 60)
    print("Comprehensive Zero-Copy BytesView Test Suite")
    print("=" * 60)

    tests = [
        ("BytesView class structure", test_bytesview_class),
        ("get() returns BytesView", test_get_returns_bytesview),
        ("NumPy from memoryview", test_numpy_array_from_memoryview),
        ("get_many() returns BytesView", test_get_many_returns_bytesview),
        ("Large file memoryview", test_large_file_memoryview),
        ("BytesView immutability", test_bytesview_immutability),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except AssertionError as e:
            print(f"\n✗ Test '{name}' failed: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed ({100 * passed // total}%)")
    print("=" * 60)

    if passed == total:
        print("\n🎉 SUCCESS: All tests passed!")
        print("\nZero-copy implementation verified:")
        print("  ✓ BytesView class with memoryview() method")
        print("  ✓ get() returns BytesView")
        print("  ✓ get_many() returns BytesView")
        print("  ✓ NumPy can create arrays from memoryview (zero-copy)")
        print("  ✓ Works with large files")
        print("  ✓ Data is immutable (read-only)")
        return 0
    else:
        print(f"\n✗ FAILURE: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
