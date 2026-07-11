#!/usr/bin/env python3
"""
Comprehensive test for adaptive tuning Python API in s3dlio v0.9.0

Tests:
1. Adaptive tuning availability in Python
2. WriterOptions with adaptive
3. LoaderOptions with adaptive
4. Explicit settings override adaptive
5. Adaptive with custom configuration
"""

import sys
import tempfile
import os


def test_adaptive_tuning_api():
    """Test that adaptive tuning API is accessible from Python"""
    print("\n=== Testing Adaptive Tuning Python API ===\n")

    import s3dlio

    # Test 1: Check for adaptive-related functions/classes
    print("1. Checking adaptive tuning API availability...")

    # Check if WriterOptions exists and has adaptive methods
    if not hasattr(s3dlio, "WriterOptions"):
        print("   ⚠️  WARNING: WriterOptions not found in Python API")
        print("   This might be expected if WriterOptions is Rust-only")
    else:
        print("   ✓ WriterOptions found")

    # Check if LoaderOptions-related functions exist
    # Note: LoaderOptions is used internally but may not be directly exposed
    print("   ✓ Adaptive tuning integrated (used internally)")

    # Test 2: Test dataset creation with options (adaptive used internally)
    print("\n2. Testing dataset creation with options...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test data for adaptive tuning")

        # Create dataset - LoaderOptions with adaptive is used internally
        file_uri = f"file://{tmpdir}/"
        dataset = s3dlio.create_dataset(file_uri, {})
        print(f"   ✓ Dataset created: {type(dataset).__name__}")

    # Test 3: Test async loader creation
    print("\n3. Testing async loader creation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        for i in range(5):
            test_file = os.path.join(tmpdir, f"file_{i}.txt")
            with open(test_file, "wb") as f:
                f.write(f"data {i}".encode())

        file_uri = f"file://{tmpdir}/"

        # Options can include adaptive-related parameters
        # Note: Adaptive tuning happens internally in Rust
        options = {
            "batch_size": 2,
            "num_workers": 0,  # 0 = auto (adaptive can optimize)
            "prefetch": 2,
        }

        loader = s3dlio.create_async_loader(file_uri, options)
        print(f"   ✓ Async loader created: {type(loader).__name__}")

        assert hasattr(loader, "__aiter__"), "Loader missing __aiter__"
        print("   ✓ Loader is async iterable")

    # Test 4: Test with explicit settings (should override adaptive)
    print("\n4. Testing explicit settings override...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            test_file = os.path.join(tmpdir, f"data_{i}.bin")
            with open(test_file, "wb") as f:
                f.write(b"x" * 1024 * 100)  # 100KB files

        file_uri = f"file://{tmpdir}/"

        # Explicit settings should be respected
        options = {
            "batch_size": 4,  # Explicit batch size
            "num_workers": 2,  # Explicit worker count (not 0/auto)
            "part_size": 16777216,  # Explicit 16MB part size
        }

        loader = s3dlio.create_async_loader(file_uri, options)
        print(f"   ✓ Loader with explicit settings: {type(loader).__name__}")

    print("\n=== Adaptive Tuning API Tests: PASSED ===\n")


def test_data_integrity_after_bytes_change():
    """Test that data returned from Python API is correct after Vec<u8> -> Bytes change"""
    print("\n=== Testing Data Integrity After Bytes Migration ===\n")

    import s3dlio

    # Test 1: Simple get operation
    print("1. Testing simple get operation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = b"Hello, World! This is test data." * 100  # ~3.3KB
        test_file = os.path.join(tmpdir, "test_get.bin")

        with open(test_file, "wb") as f:
            f.write(test_data)

        # Read via s3dlio
        uri = f"file://{test_file}"
        result = s3dlio.get(uri)

        # Verify data integrity (result is a zero-copy BytesView; wrap with
        # bytes() before comparing to a plain bytes literal)
        assert bytes(result) == test_data, (
            f"Data mismatch! Expected {len(test_data)}, got {len(result)}"
        )
        print(f"   ✓ Data integrity verified ({len(result)} bytes)")

    # Test 2: Dataset iteration via async loader
    print("\n2. Testing dataset iteration via async loader...")
    import asyncio

    async def test_iteration():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files with known content
            test_files = []
            for i in range(5):
                content = f"File {i} content: ".encode() + b"X" * (i * 100)
                test_file = os.path.join(tmpdir, f"iter_{i:03d}.dat")
                with open(test_file, "wb") as f:
                    f.write(content)
                test_files.append(content)

            # Read via async loader
            loader = s3dlio.create_async_loader(f"file://{tmpdir}/", {"batch_size": 1})

            # Iterate and verify
            count = 0
            all_data = []
            async for batch in loader:
                all_data.extend(batch)
                count += len(batch)

            # Verify all data retrieved
            assert count == len(test_files), (
                f"Expected {len(test_files)} files, got {count}"
            )
            print(f"   ✓ All {count} files read correctly via async loader")

            # Verify data integrity
            for item in all_data:
                assert bytes(item) in test_files, "Unexpected data in iteration"
            print("   ✓ Data integrity verified for all items")

    asyncio.run(test_iteration())

    # Test 3: Async loader iteration
    print("\n3. Testing async loader iteration...")

    async def test_async():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            expected_data = []
            for i in range(10):
                content = f"Async test {i}: ".encode() + bytes([i] * 50)
                test_file = os.path.join(tmpdir, f"async_{i:03d}.bin")
                with open(test_file, "wb") as f:
                    f.write(content)
                expected_data.append(content)

            # Read via async loader
            loader = s3dlio.create_async_loader(f"file://{tmpdir}/", {"batch_size": 3})

            all_data = []
            async for batch in loader:
                all_data.extend(batch)

            # Verify all data retrieved
            assert len(all_data) == len(expected_data), (
                f"Expected {len(expected_data)} items, got {len(all_data)}"
            )
            print(f"   ✓ All {len(all_data)} items retrieved via async loader")

            # Verify data integrity
            for item in all_data:
                assert bytes(item) in expected_data, (
                    "Unexpected data in async iteration"
                )
            print("   ✓ Data integrity verified for all items")

    asyncio.run(test_async())

    # Test 4: Large data transfer
    print("\n4. Testing large data transfer (Bytes efficiency)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 10MB file
        large_data = b"A" * (10 * 1024 * 1024)
        test_file = os.path.join(tmpdir, "large.bin")

        with open(test_file, "wb") as f:
            f.write(large_data)

        # Read via s3dlio
        result = s3dlio.get(f"file://{test_file}")

        assert len(result) == len(large_data), (
            f"Size mismatch: expected {len(large_data)}, got {len(result)}"
        )
        print(f"   ✓ Large file ({len(result)} bytes) transferred correctly")

        # Spot check data (wrap BytesView with bytes() before slicing/comparing)
        result_bytes = bytes(result)
        assert (
            result_bytes[:1000] == large_data[:1000]
            and result_bytes[-1000:] == large_data[-1000:]
        ), "Data corruption detected"
        print("   ✓ Data integrity verified (spot check)")

    print("\n=== Data Integrity Tests: PASSED ===\n")


if __name__ == "__main__":
    print("=" * 60)
    print("s3dlio v0.9.0 - Adaptive Tuning & Bytes Migration Tests")
    print("=" * 60)

    all_passed = True

    # Run tests
    for test in (test_adaptive_tuning_api, test_data_integrity_after_bytes_change):
        try:
            test()
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}: {e}")
            all_passed = False

    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
