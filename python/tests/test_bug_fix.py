"""
Regression test for the PyTorch wrapper calling a non-existent PyS3AsyncDataLoader.

The bug was in s3dlio/torch.py: it called PyS3AsyncDataLoader, which did not exist. The fix
changed it to use create_async_loader, which supports file://, s3://, az://, gs:// and
direct:// URIs through the generic API.

Nothing here touches the filesystem at import time. The data file lives in pytest's
tmp_path by default; set S3DLIO_TEST_DIR to a directory to place it on a specific file system
(for example a network mount) instead.
"""

import os
import shutil

import pytest

pytest.importorskip("torch", reason="s3dlio.torch needs PyTorch")


@pytest.fixture
def data_file(tmp_path):
    """A small file in S3DLIO_TEST_DIR (when set) or in pytest's tmp_path, removed afterwards."""
    base = os.environ.get("S3DLIO_TEST_DIR")
    if not base:
        path = tmp_path / "test_data.txt"
        path.write_bytes(b"Hello, this is test data for the bug fix!\nLine 2\nLine 3\n")
        yield path
        return

    if not os.path.isdir(base):
        pytest.skip(f"S3DLIO_TEST_DIR={base!r} is not an existing directory")
    test_dir = os.path.join(base, f"s3dlio_bug_test_{os.getpid()}")
    os.makedirs(test_dir)
    path = os.path.join(test_dir, "test_data.txt")
    try:
        with open(path, "wb") as f:
            f.write(b"Hello, this is test data for the bug fix!\nLine 2\nLine 3\n")
        yield path
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_s3_iterable_dataset_uses_create_async_loader(data_file):
    """Building and iterating S3IterableDataset over a file:// URI works (no PyS3AsyncDataLoader)."""
    from s3dlio.torch import S3IterableDataset

    file_uri = f"file://{data_file}"

    with pytest.warns(DeprecationWarning):
        dataset = S3IterableDataset(file_uri, loader_opts={"batch_size": 1})

    first_item = next(iter(dataset))
    assert len(first_item) > 0
