#!/usr/bin/env python3
"""
Test v0.8.19 Python API fixes across all backends
Tests:
1. Universal list() function with all backends
2. Universal stat() function with all backends
3. Regex pattern support in list()
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import s3dlio

# --- Which live endpoints are configured -------------------------------------------------
# A test that needs a live cloud endpoint SKIPS (never fails) when that endpoint is not
# configured in the environment, and ASSERTS what its skip text promises when it is.
# Credentials always come from whatever the environment provides (including ADC / profiles /
# `az login`); only the variables below decide whether the test runs.
#
#   S3     S3DLIO_TEST_BUCKET = a bucket holding hello.txt (AWS_ENDPOINT_URL optionally
#          points at an S3-compatible server)
#   GCS    S3DLIO_TEST_GCS_BUCKET = a bucket holding test.txt
#   Azure  AZURE_BLOB_ACCOUNT = the storage account, and S3DLIO_TEST_AZURE_CONTAINER = a
#          container holding test.txt
AZURE_ACCOUNT = os.environ.get("AZURE_BLOB_ACCOUNT", "")
AZURE_CONTAINER = os.environ.get("S3DLIO_TEST_AZURE_CONTAINER", "")
S3_BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "")
GCS_BUCKET = os.environ.get("S3DLIO_TEST_GCS_BUCKET", "")


def _s3_skip_reason():
    if not S3_BUCKET:
        return (
            "live S3 not configured: set S3DLIO_TEST_BUCKET to a bucket holding hello.txt "
            "(credentials and endpoint come from the environment)"
        )
    return None


def _gcs_skip_reason():
    if not GCS_BUCKET:
        return (
            "live GCS not configured: set S3DLIO_TEST_GCS_BUCKET to a bucket holding "
            "test.txt (credentials come from the environment, e.g. ADC)"
        )
    return None


def _azure_skip_reason():
    missing = [
        n
        for n, v in (
            ("AZURE_BLOB_ACCOUNT", AZURE_ACCOUNT),
            ("S3DLIO_TEST_AZURE_CONTAINER", AZURE_CONTAINER),
        )
        if not v
    ]
    if missing:
        return (
            "live Azure not configured: set AZURE_BLOB_ACCOUNT (storage account) and "
            "S3DLIO_TEST_AZURE_CONTAINER (a container holding test.txt); credentials come "
            "from the environment (AZURE_* variables or `az login`); missing: "
            + ", ".join(missing)
        )
    return None


def _skip_unless_configured(reason):
    if reason:
        pytest.skip(reason)


def _check_live_backend(label, base_uri, object_name):
    """list() sees ``object_name`` (full URI), the .txt pattern filter keeps it, stat()
    reports a positive size: exactly what the skip text of the live tests promises."""
    expected = f"{base_uri}{object_name}"

    print(f"\n[{label}] list() - baseline")
    keys = s3dlio.list(base_uri)
    print(f"  Found {len(keys)} objects")
    assert expected in keys, f"{expected} not in list({base_uri!r}): {keys}"

    print(f"\n[{label}] list() with pattern (.*\\.txt$)")
    txt_keys = s3dlio.list(base_uri, pattern=r".*\.txt$")
    print(f"  Filtered to {len(txt_keys)} .txt files")
    for key in txt_keys:
        print(f"    - {key}")
    assert expected in txt_keys, f"{expected} not in pattern list: {txt_keys}"
    assert all(k.endswith(".txt") for k in txt_keys), txt_keys

    print(f"\n[{label}] stat() - {object_name}")
    stat = s3dlio.stat(expected)
    print(f"  Size: {stat['size']}")
    print(f"  ETag: {stat['etag']}")
    assert stat["size"] > 0, f"{expected}: stat size {stat['size']}"


def test_s3():
    """Test S3 backend"""
    _skip_unless_configured(_s3_skip_reason())
    print("=" * 60)
    print("S3 Backend Testing")
    print("=" * 60)
    _check_live_backend("S3", f"s3://{S3_BUCKET}/", "hello.txt")
    print("\n✅ S3 tests PASSED\n")


def test_gcs():
    """Test GCS backend"""
    _skip_unless_configured(_gcs_skip_reason())
    print("=" * 60)
    print("GCS Backend Testing")
    print("=" * 60)
    _check_live_backend("GCS", f"gs://{GCS_BUCKET}/", "test.txt")
    print("\n✅ GCS tests PASSED\n")


def test_azure():
    """Test Azure backend"""
    _skip_unless_configured(_azure_skip_reason())
    print("=" * 60)
    print("Azure Backend Testing")
    print("=" * 60)
    _check_live_backend("Azure", f"az://{AZURE_ACCOUNT}/{AZURE_CONTAINER}/", "test.txt")
    print("\n✅ Azure tests PASSED\n")


def test_file(tmp_path):
    """Test File backend (creates its own data under tmp_path)"""
    print("=" * 60)
    print("File Backend Testing")
    print("=" * 60)

    test_dir = tmp_path / "file-test"
    test_dir.mkdir()
    for i in range(3):
        (test_dir / f"file{i}.txt").write_text(f"Test file {i}")
    (test_dir / "data.json").write_text('{"test": true}')
    base = f"file://{test_dir}/"

    # Test list
    print(f"\n[File] list() - {test_dir}")
    keys = s3dlio.list(base)
    print(f"  Found {len(keys)} objects")
    assert sorted(keys) == sorted(
        f"file://{test_dir}/{n}"
        for n in ("file0.txt", "file1.txt", "file2.txt", "data.json")
    ), keys

    # Test list with pattern
    print("\n[File] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list(base, pattern=r".*\.txt$")
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    assert sorted(keys) == sorted(f"file://{test_dir}/file{i}.txt" for i in range(3)), (
        keys
    )

    # Test stat
    print(f"\n[File] stat() - {test_dir}/file0.txt")
    stat = s3dlio.stat(f"file://{test_dir}/file0.txt")
    print(f"  Size: {stat['size']}")
    print(f"  Storage Class: {stat['storage_class']}")
    assert stat["size"] == len("Test file 0")

    print("\n✅ File tests PASSED\n")


def test_directio(tmp_path):
    """Test DirectIO backend (creates its own data under tmp_path)"""
    print("=" * 60)
    print("DirectIO Backend Testing")
    print("=" * 60)

    test_dir = str(tmp_path / "directio-test")
    os.makedirs(test_dir)
    payload = b"directio test file one\n"
    with open(f"{test_dir}/testfile1.txt", "wb") as f:
        f.write(payload)
    with open(f"{test_dir}/testfile2.txt", "wb") as f:
        f.write(b"directio test file two\n")
    with open(f"{test_dir}/data.json", "wb") as f:
        f.write(b'{"test": true}')

    # Test list
    print(f"\n[DirectIO] list() - {test_dir}")
    keys = s3dlio.list(f"direct://{test_dir}/")
    print(f"  Found {len(keys)} objects")
    assert len(keys) == 3, keys

    # Test list with pattern
    print("\n[DirectIO] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list(f"direct://{test_dir}/", pattern=r".*\.txt$")
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    assert len(keys) == 2, keys

    # Test stat
    print(f"\n[DirectIO] stat() - {test_dir}/testfile1.txt")
    stat = s3dlio.stat(f"direct://{test_dir}/testfile1.txt")
    print(f"  Size: {stat['size']}")
    print(f"  Storage Class: {stat['storage_class']}")
    assert stat["size"] == len(payload)

    print("\n✅ DirectIO tests PASSED\n")


def main():
    """Run all backend tests"""
    print("\n" + "=" * 60)
    print("Python API v0.8.19 - Universal Backend Testing")
    print("=" * 60)
    print()

    failed = []
    skipped = []

    with tempfile.TemporaryDirectory(prefix="s3dlio-v0819-") as scratch:
        # Test each backend
        for test_func, name in [
            (test_s3, "S3"),
            (test_gcs, "GCS"),
            (test_azure, "Azure"),
            (test_file, "File"),
            (lambda: test_directio(Path(scratch)), "DirectIO"),
        ]:
            try:
                test_func()
            except pytest.skip.Exception as e:
                print(f"\n⏭️  {name} tests SKIPPED: {e}\n")
                skipped.append(name)
            except Exception as e:
                print(f"\n❌ {name} tests FAILED: {e}\n")
                import traceback

                traceback.print_exc()
                failed.append(name)

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if skipped:
        print(f"\nSkipped (endpoint not configured): {', '.join(skipped)}")

    if not failed:
        print("\n✅ ALL CONFIGURED BACKEND TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {len(failed)} backend(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
