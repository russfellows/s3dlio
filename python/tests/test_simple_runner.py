#!/usr/bin/env python3
"""
Simple test runner to verify s3dlio functionality works.
This bypasses linter issues and focuses on actual functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_basic_functionality():
    """Test basic s3dlio functionality."""
    print("=== Basic Functionality Tests ===")

    import s3dlio

    print("✓ s3dlio import successful")

    # Check available functions
    assert hasattr(s3dlio, "create_dataset"), "create_dataset not available"
    print("✓ create_dataset available")
    assert hasattr(s3dlio, "create_async_loader"), "create_async_loader not available"
    print("✓ create_async_loader available")


def test_file_uri_functionality():
    """Test file:// URI functionality."""
    print("\n=== File URI Tests ===")

    import s3dlio

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test_file.txt"
        test_data = b"Hello, s3dlio file system test!"
        test_file.write_bytes(test_data)

        file_uri = f"file://{test_file}"
        print(f"Testing URI: {file_uri}")

        # Test dataset creation
        s3dlio.create_dataset(file_uri)
        print("✓ create_dataset with file:// URI successful")

        # Test async loader creation
        s3dlio.create_async_loader(file_uri)
        print("✓ create_async_loader with file:// URI successful")

        # Test directory URI
        dir_uri = f"file://{tmpdir}"
        print(f"Testing directory URI: {dir_uri}")

        s3dlio.create_dataset(dir_uri)
        print("✓ create_dataset with directory URI successful")


def test_error_handling():
    """Test error handling for invalid URIs."""
    print("\n=== Error Handling Tests ===")

    import s3dlio

    # Test unsupported scheme
    try:
        s3dlio.create_dataset("ftp://example.com/path")
        raise AssertionError("Unsupported scheme should have failed")
    except AssertionError:
        raise
    except Exception as e:
        print(f"✓ Unsupported scheme properly rejected: {type(e).__name__}")

    # Test malformed URI
    try:
        s3dlio.create_dataset("not-a-uri")
        raise AssertionError("Malformed URI should have failed")
    except AssertionError:
        raise
    except Exception as e:
        print(f"✓ Malformed URI properly rejected: {type(e).__name__}")

    # Test nonexistent file
    try:
        s3dlio.create_dataset("file:///nonexistent/path/file.txt")
        raise AssertionError("Nonexistent file should have failed")
    except AssertionError:
        raise
    except Exception as e:
        print(f"✓ Nonexistent file properly rejected: {type(e).__name__}")


def test_options_functionality():
    """Test options parsing."""
    print("\n=== Options Tests ===")

    import s3dlio

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test_options.txt"
        test_file.write_bytes(b"Test data for options")

        # Test with valid options
        options = {"batch_size": 32, "shuffle": True, "num_workers": 2}

        s3dlio.create_dataset(f"file://{test_file}", options)
        print("✓ Options parsing successful")

        # Test with invalid options type
        try:
            s3dlio.create_dataset(f"file://{test_file}", ["invalid", "options"])
            raise AssertionError("Invalid options type should have failed")
        except AssertionError:
            raise
        except Exception as e:
            print(f"✓ Invalid options type properly rejected: {type(e).__name__}")


def test_torch_integration():
    """Test PyTorch integration if available."""
    print("\n=== PyTorch Integration Tests ===")

    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip(reason="PyTorch not available in this environment")

    from s3dlio.torch import S3IterableDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "torch_test.txt"
        test_file.write_bytes(b"PyTorch integration test data")

        # Test S3IterableDataset creation
        dataset = S3IterableDataset(f"file://{test_file}", loader_opts={})
        print("✓ S3IterableDataset creation successful")

        # Test iteration
        items = list(dataset)
        assert len(items) > 0, "Dataset iteration returned no items"
        print(f"✓ Dataset iteration successful, got {len(items)} items")


def test_compatibility_shims():
    """Test backward compatibility wrappers."""
    print("\n=== Compatibility Tests ===")

    import s3dlio

    # Test PyS3Dataset compatibility
    if hasattr(s3dlio, "PyS3Dataset"):
        print("✓ PyS3Dataset compatibility wrapper available")
    else:
        print("- PyS3Dataset not available (may be expected)")

    # Test PyS3AsyncDataLoader compatibility
    if hasattr(s3dlio, "PyS3AsyncDataLoader"):
        print("✓ PyS3AsyncDataLoader compatibility wrapper available")
    else:
        print("- PyS3AsyncDataLoader not available (may be expected)")

    # Test new classes
    assert hasattr(s3dlio, "PyDataset"), "PyDataset class not available"
    print("✓ PyDataset class available")

    assert hasattr(s3dlio, "PyBytesAsyncDataLoader"), (
        "PyBytesAsyncDataLoader class not available"
    )
    print("✓ PyBytesAsyncDataLoader class available")


def main():
    """Run all tests."""
    print("S3DLIO Comprehensive Test Suite")
    print("=" * 50)

    tests = [
        test_basic_functionality,
        test_file_uri_functionality,
        test_error_handling,
        test_options_functionality,
        test_torch_integration,
        test_compatibility_shims,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
        except pytest.skip.Exception as e:
            total -= 1
            print(f"- {test.__name__} skipped: {e}")
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")

    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
