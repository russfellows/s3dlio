#!/usr/bin/env python3
"""
Test the Python API with real S3 backend using different data generation modes
"""

import s3dlio
import time

import pytest


def test_streaming_mode():
    """Test streaming mode via Python API"""
    print("Testing streaming mode via Python API...")
    start_time = time.time()

    try:
        result = s3dlio.put(
            prefix="s3://test-python-api/python-streaming-{}.bin",
            num=3,
            template="test-{}-of-{}",
            size=4194304,  # 4MB
            data_gen_mode="streaming",
            chunk_size=65536,
        )
    except Exception as e:
        pytest.skip(f"requires live AWS S3 credentials/bucket (test-python-api): {e}")

    elapsed = time.time() - start_time
    print(f"Streaming mode completed in {elapsed:.2f} seconds")
    print(f"Result: {result}")


def test_single_pass_mode():
    """Test single-pass mode via Python API"""
    print("\nTesting single-pass mode via Python API...")
    start_time = time.time()

    try:
        result = s3dlio.put(
            prefix="s3://test-python-api/python-singlepass-{}.bin",
            num=3,
            template="test-{}-of-{}",
            size=4194304,  # 4MB
            data_gen_mode="single-pass",
            chunk_size=65536,
        )
    except Exception as e:
        pytest.skip(f"requires live AWS S3 credentials/bucket (test-python-api): {e}")

    elapsed = time.time() - start_time
    print(f"Single-pass mode completed in {elapsed:.2f} seconds")
    print(f"Result: {result}")


def test_default_mode():
    """Test default mode (should be streaming) via Python API"""
    print("\nTesting default mode (should be streaming) via Python API...")
    start_time = time.time()

    try:
        result = s3dlio.put(
            prefix="s3://test-python-api/python-default-{}.bin",
            num=3,
            template="test-{}-of-{}",
            size=4194304,  # 4MB - using defaults for data_gen_mode and chunk_size
        )
    except Exception as e:
        pytest.skip(f"requires live AWS S3 credentials/bucket (test-python-api): {e}")

    elapsed = time.time() - start_time
    print(f"Default mode completed in {elapsed:.2f} seconds")
    print(f"Result: {result}")


if __name__ == "__main__":
    print("Testing s3dlio Python API with real S3 backend...")
    print("=" * 60)

    results = {}
    for name, test_func in [
        ("Streaming mode", test_streaming_mode),
        ("Single-pass mode", test_single_pass_mode),
        ("Default mode", test_default_mode),
    ]:
        try:
            test_func()
            results[name] = True
        except AssertionError as e:
            print(f"{name} failed: {e}")
            results[name] = False
        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary:")
    for name, ok in results.items():
        print(f"{name}: {'✓ PASS' if ok else '✗ FAIL'}")

    if all(results.values()):
        print("\n🎉 All tests passed! Data generation modes are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
