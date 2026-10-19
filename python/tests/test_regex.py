"""Live-S3 list()/download() recursion test.

Needs a live S3 endpoint.  It SKIPS (never fails) unless S3DLIO_TEST_BUCKET is set:

  S3DLIO_TEST_BUCKET   an existing, writable bucket; the test writes under a unique prefix
                       and deletes it afterwards.  Credentials and the endpoint come from
                       whatever the environment provides (AWS_* variables, profiles,
                       AWS_ENDPOINT_URL for an S3-compatible server).
"""

import contextlib
import os
import shutil
import tempfile
import time
import uuid

import pytest
import s3dlio

# --- Configuration ---
BUCKET_NAME = os.environ.get("S3DLIO_TEST_BUCKET", "")
TEST_PREFIX = f"test-recursive-py-{uuid.uuid4().hex[:12]}/"
S3_URI_PREFIX = f"s3://{BUCKET_NAME}/{TEST_PREFIX}"


def _skip_reason():
    if not BUCKET_NAME:
        return (
            "live S3 not configured: set S3DLIO_TEST_BUCKET to an existing writable bucket "
            "(credentials and endpoint come from the environment)"
        )
    return None


def _assert_listing(listing, expected, what):
    """list() returns full s3://bucket/key URIs (the documented contract): compare exactly."""
    want = sorted(f"s3://{BUCKET_NAME}/{TEST_PREFIX}{name}" for name in expected)
    assert sorted(listing) == want, f"{what}: got {sorted(listing)}, expected {want}"


def setup_test_files():
    """Create a test file structure in S3 with unique filenames."""
    print("--- Setting up test files in S3... ---")

    src_dir = tempfile.mkdtemp(prefix="s3dlio-regex-src-")
    try:
        # Create three unique local files to upload
        for i in (1, 2, 3):
            with open(os.path.join(src_dir, f"dummy{i}.txt"), "w") as f:
                f.write(f"test{i}")

        # Upload each unique file to its destination
        s3dlio.upload([os.path.join(src_dir, "dummy1.txt")], f"{S3_URI_PREFIX}")
        s3dlio.upload([os.path.join(src_dir, "dummy2.txt")], f"{S3_URI_PREFIX}data/")
        s3dlio.upload(
            [os.path.join(src_dir, "dummy3.txt")], f"{S3_URI_PREFIX}data/logs/"
        )
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)

    print("Setup complete.")
    time.sleep(2)  # Allow time for S3 to be consistent


def cleanup():
    """Best-effort delete of the test objects, then assert nothing remains."""
    print("\n--- Cleaning up test files... ---")
    with contextlib.suppress(Exception):  # best-effort; the check below is the verdict
        s3dlio.delete(S3_URI_PREFIX, recursive=True)

    remaining_files = s3dlio.list(S3_URI_PREFIX, recursive=True)
    assert len(remaining_files) == 0, (
        f"Cleanup failed, objects still remain in S3: {remaining_files}"
    )

    print("✅ Cleanup successful!")


@pytest.fixture(scope="module")
def s3_test_tree():
    """Populate the unique test prefix once for the module; remove it afterwards."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    try:
        setup_test_files()
        yield S3_URI_PREFIX
    finally:
        cleanup()


def test_list(s3_test_tree):
    """Test the list function with and without recursion."""
    print("\n--- Testing s3dlio.list() ---")

    # Test non-recursive list
    non_recursive_list = s3dlio.list(S3_URI_PREFIX, recursive=False)
    print(
        f"Non-recursive list found {len(non_recursive_list)} items: {non_recursive_list}"
    )
    assert len(non_recursive_list) == 2, "Non-recursive list should find 2 items"
    _assert_listing(non_recursive_list, ["dummy1.txt", "data/"], "non-recursive list")

    # Test recursive list
    recursive_list = s3dlio.list(S3_URI_PREFIX, recursive=True)
    print(f"Recursive list found {len(recursive_list)} items: {recursive_list}")
    assert len(recursive_list) == 3, "Recursive list should find 3 files"
    _assert_listing(
        recursive_list,
        ["dummy1.txt", "data/dummy2.txt", "data/logs/dummy3.txt"],
        "recursive list",
    )
    print("✅ List tests passed!")


def test_download(s3_test_tree, tmp_path):
    """Test the download function."""
    print("\n--- Testing s3dlio.download() ---")

    # Test non-recursive download
    nonrec_dir = tmp_path / "py_download_nonrecursive"
    nonrec_dir.mkdir()
    s3dlio.download(S3_URI_PREFIX, str(nonrec_dir), recursive=False)
    downloaded_files = os.listdir(nonrec_dir)
    print(f"Non-recursive download got: {downloaded_files}")
    assert len(downloaded_files) == 1
    assert "dummy1.txt" in downloaded_files  # Check for dummy1.txt

    # Test recursive download
    rec_dir = tmp_path / "py_download_recursive"
    rec_dir.mkdir()
    s3dlio.download(S3_URI_PREFIX, str(rec_dir), recursive=True)
    # Issue #284: each object keeps its key path relative to the listing base.
    downloaded_files_rec = sorted(
        os.path.relpath(os.path.join(root, name), rec_dir).replace(os.sep, "/")
        for root, _dirs, names in os.walk(rec_dir)
        for name in names
    )
    print(f"Recursive download got: {downloaded_files_rec}")
    assert downloaded_files_rec == [
        "data/dummy2.txt",
        "data/logs/dummy3.txt",
        "dummy1.txt",
    ]
    print("✅ Download tests passed!")


if __name__ == "__main__":
    # Run under pytest so the fixture skips cleanly when no live S3 is configured.
    raise SystemExit(pytest.main([__file__, "-s", "-rs"]))
