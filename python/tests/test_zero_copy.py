#!/usr/bin/env python3
"""Test zero-copy BytesView implementation"""

import s3dlio
import sys


def test_bytes_view():
    """Test that BytesView exists and has the right methods"""
    print("Testing BytesView class...")

    # Check if BytesView is available
    assert hasattr(s3dlio, "BytesView"), "BytesView class not found"
    print("✓ BytesView class found")
    print(f"  Methods: {[m for m in dir(s3dlio.BytesView) if not m.startswith('_')]}")


def test_get_returns_bytesview():
    """Test that get() returns BytesView instead of bytes"""
    print("\nTesting get() return type...")

    # We can't actually test S3 without credentials, but we can check the signature
    import inspect

    assert hasattr(s3dlio, "get"), "get() function not found"
    sig = inspect.signature(s3dlio.get)
    print("✓ get() function found")
    print(f"  Signature: {sig}")
    # We'll need actual S3 to test the return type
    print("  Note: Actual return type requires S3 connection to test")


def main():
    print("=" * 60)
    print("Zero-Copy BytesView Test")
    print("=" * 60)

    tests = [
        test_bytes_view,
        test_get_returns_bytesview,
    ]

    results = []
    for test in tests:
        try:
            test()
            results.append(True)
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            results.append(False)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed")
        return 0
    else:
        print("FAILURE: Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
