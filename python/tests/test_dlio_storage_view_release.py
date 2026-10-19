"""A failed multipart ``put_data`` must not leave buffer exports behind (stage G).

``put_data`` reads the caller's ``io.BytesIO`` in place (``getbuffer()``) and slices it with a
derived ``memoryview``. A failed upload leaves ``put_data``'s frame alive in the exception's
traceback, and with it the derived view; the caller's ``with io.BytesIO(...)`` block then failed
in ``close()`` with ``BufferError: Existing exports of data`` and that error REPLACED the real
upload error. ``put_data`` must release the views it created before the exception leaves.

``dlio_benchmark`` is not a dependency of this repo; its import surface is faked the same way as
in ``test_s3dlio_storage_multipart_write_failure.py``.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_s3dlio_storage_multipart_write_failure import (
    ENV_VARS,
    _install_fake_dlio_benchmark,
)

_install_fake_dlio_benchmark()

import s3dlio
from s3dlio.integrations.dlio import s3_torch_storage, s3dlio_storage

PART = 1024 * 1024


@pytest.fixture(autouse=True)
def _multipart_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("S3DLIO_MULTIPART_THRESHOLD_MB", "0")  # always multipart
    monkeypatch.setenv("S3DLIO_MULTIPART_PART_SIZE_MB", "1")


class FailingWriter:
    """Keeps no reference to what it is given (unlike a MagicMock, which records call args)."""

    def __init__(self, fail_on_write):
        self.fail_on_write = fail_on_write
        self.writes = 0
        self.aborted = False
        self.closed = False

    def write(self, chunk):
        self.writes += 1
        if self.writes == self.fail_on_write:
            # The traceback keeps THIS frame alive: do not let it keep the slice (and its buffer
            # export) too. A real writer fails in native code and holds no Python frame.
            del chunk
            raise RuntimeError("simulated upload failure")

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


def _storage(cls):
    store = object.__new__(cls)
    store.prefix = "s3://bucket"
    return store


STORAGES = [
    pytest.param(s3dlio_storage.S3dlioStorage, id="s3dlio_storage"),
    pytest.param(s3_torch_storage.S3PyTorchConnectorStorage, id="s3_torch_storage"),
]


@pytest.mark.parametrize("storage_cls", STORAGES)
def test_failed_multipart_put_inside_with_bytesio_raises_the_real_error(
    monkeypatch, storage_cls
):
    writer = FailingWriter(fail_on_write=2)
    monkeypatch.setattr(
        s3dlio.MultipartUploadWriter, "from_uri", staticmethod(lambda *a, **k: writer)
    )
    store = _storage(storage_cls)
    with (
        pytest.raises(RuntimeError, match="simulated upload failure"),
        io.BytesIO(b"x" * (4 * PART)) as bio,
    ):
        store.put_data("key", bio)
    assert writer.aborted and not writer.closed


@pytest.mark.parametrize("storage_cls", STORAGES)
def test_the_bytesio_is_closeable_again_after_a_failed_put(monkeypatch, storage_cls):
    writer = FailingWriter(fail_on_write=3)
    monkeypatch.setattr(
        s3dlio.MultipartUploadWriter, "from_uri", staticmethod(lambda *a, **k: writer)
    )
    store = _storage(storage_cls)
    bio = io.BytesIO(b"y" * (4 * PART))
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        store.put_data("key", bio)
    bio.close()  # BufferError ("Existing exports of data") if a view survived


@pytest.mark.parametrize("storage_cls", STORAGES)
def test_a_successful_multipart_put_releases_the_bytesio(monkeypatch, storage_cls):
    writer = FailingWriter(fail_on_write=0)
    monkeypatch.setattr(
        s3dlio.MultipartUploadWriter, "from_uri", staticmethod(lambda *a, **k: writer)
    )
    store = _storage(storage_cls)
    with io.BytesIO(b"z" * (3 * PART)) as bio:
        store.put_data("key", bio)
    assert writer.closed and writer.writes == 3
