#!/usr/bin/env python3

"""
Test script to verify the s3dlio bug fix and new generic API functionality.

This tests:
1. The specific bug fix: PyS3AsyncDataLoader -> create_async_loader
2. The new generic API with multiple URI schemes
3. All major API improvements
"""

import asyncio
import sys
import tempfile
import os

import pytest

import s3dlio


# Test generic dataset creation and async loader functionality
@pytest.mark.asyncio
async def test_generic_api():
    """Test the new generic create_async_loader functionality"""
    print("\n=== Testing Generic API ===")

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"Hello, this is test data!\nLine 2\nLine 3")
        tmp_path = tmp.name

    try:
        # Test 1: file:// URI scheme
        file_uri = f"file://{tmp_path}"
        print(f"Testing file URI: {file_uri}")

        # Test create_async_loader (this was the source of the bug!)
        loader = s3dlio.create_async_loader(file_uri)
        assert loader is not None, (
            "create_async_loader failed to return a loader for file:// URI"
        )
        print("✓ create_async_loader works with file:// URI")

        # Test create_dataset
        dataset = s3dlio.create_dataset(file_uri)
        assert dataset is not None, (
            "create_dataset failed to return a dataset for file:// URI"
        )
        print("✓ create_dataset works with file:// URI")

        # Test convenience functions
        # NOTE: list/stat/get are synchronous functions in the current API
        # (not coroutines) - the original test incorrectly awaited them,
        # which raised TypeError and was masked by the return-False anti-pattern.
        # Test list function
        parent_dir = f"file://{os.path.dirname(tmp_path)}"
        files = s3dlio.list(parent_dir)
        assert len(files) > 0, f"list function returned no files for {parent_dir}"
        print(f"✓ list function works, found {len(files)} files")

        # Test stat function
        stat_result = s3dlio.stat(file_uri)
        assert stat_result.get("size") is not None, (
            "stat function did not return a size"
        )
        print(f"✓ stat function works, size: {stat_result.get('size', 'unknown')}")

        # Test get function
        data = s3dlio.get(file_uri)
        assert len(data) > 0, "get function returned no data"
        print(f"✓ get function works, retrieved {len(data)} bytes")

    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def test_pytorch_integration():
    """Test that the PyTorch wrapper works with the fixed API"""
    print("\n=== Testing PyTorch Integration ===")

    try:
        # This should not fail anymore - the bug was here!
        # NOTE: the old API name was S3DataLoader; the current equivalent
        # exposed by s3dlio.torch is S3IterableDataset.
        from s3dlio.torch import S3IterableDataset
    except ImportError as e:
        pytest.skip(reason=f"torch not available: {e}")

    print("✓ Successfully imported S3IterableDataset from torch module")

    # Test creating with file URI (should use the fixed create_async_loader)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"test data")
        tmp_path = tmp.name

    try:
        # This was the line that failed before the fix!
        # NOTE: the bare constructor requires a `loader_opts` kwarg with no
        # default; the supported entry point is the from_prefix() classmethod,
        # which fills in sane defaults.
        loader = S3IterableDataset.from_prefix(f"file://{tmp_path}")
        assert loader is not None, "S3IterableDataset creation returned None"
        print("✓ S3IterableDataset creation works with fixed API")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def test_compatibility_shims():
    """Test that compatibility wrappers still work"""
    print("\n=== Testing Compatibility Shims ===")

    # Test that old API names still exist (with deprecation warnings)
    assert hasattr(s3dlio, "PyS3Dataset"), "PyS3Dataset compatibility shim missing"
    print("✓ PyS3Dataset compatibility shim exists")

    assert hasattr(s3dlio, "PyS3AsyncDataLoader"), (
        "PyS3AsyncDataLoader compatibility shim missing"
    )
    print("✓ PyS3AsyncDataLoader compatibility shim exists")


async def main():
    """Main test function"""
    print("S3DLIO Bug Fix and API Enhancement Test")
    print("=" * 50)

    tests = [
        ("Generic API", test_generic_api, True),
        ("PyTorch Integration", test_pytorch_integration, False),
        ("Compatibility Shims", test_compatibility_shims, False),
    ]

    passed = 0
    total = len(tests)

    for name, test, is_async in tests:
        try:
            if is_async:
                await test()
            else:
                test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} failed: {e}")
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n=== Test Results ===")

    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("✓ Bug fix successful!")
        print("✓ Generic API enhancement successful!")
        return 0
    else:
        print(f"✗ Some tests failed ({passed}/{total})")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
