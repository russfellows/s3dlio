#!/usr/bin/env python3
"""
RED-then-GREEN regression test for s3dlio issue #153 bug 3.7 (D6).

Bug: S3dlioStorage.put_data()'s multipart branch had no try/finally around
the write loop. If writer.write() raised partway through (e.g. part 3 of
5), the exception propagated via the outer `except Exception as e: print(...);
raise`, but the writer itself was never explicitly aborted -- cleanup relied
entirely on abort_on_drop=True firing whenever the Rust-side MultipartUploadSink
eventually got dropped, which depends on CPython refcounting/GC timing, not a
guarantee. No explicit, deterministic cleanup happened at the point of failure.

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 6, bug D6): wrap the write loop in try/finally (here, try/except since
success falls through to close() and re-raise) and call writer.abort()
explicitly on any exception, immediately and deterministically, before
re-raising.

dlio_benchmark is not a dependency of this repo, so this test fakes just
enough of its import surface via sys.modules, matching the pattern used
throughout this test suite.
"""

import os
import sys
import types
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _install_fake_dlio_benchmark():
    if "dlio_benchmark" in sys.modules:
        return

    dlio_benchmark = types.ModuleType("dlio_benchmark")

    common = types.ModuleType("dlio_benchmark.common")
    constants = types.ModuleType("dlio_benchmark.common.constants")
    constants.MODULE_STORAGE = "storage"
    enumerations = types.ModuleType("dlio_benchmark.common.enumerations")

    class NamespaceType:
        FLAT = "flat"

    class MetadataType:
        FILE = "file"
        DIRECTORY = "directory"

    enumerations.NamespaceType = NamespaceType
    enumerations.MetadataType = MetadataType

    storage = types.ModuleType("dlio_benchmark.storage")
    storage_handler = types.ModuleType("dlio_benchmark.storage.storage_handler")

    class Namespace:
        def __init__(self, name, ns_type):
            self.name = name
            self.ns_type = ns_type

    class DataStorage:
        def __init__(self, framework=None):
            self.framework = framework
            self._args = types.SimpleNamespace()

    storage_handler.DataStorage = DataStorage
    storage_handler.Namespace = Namespace

    s3_storage_mod = types.ModuleType("dlio_benchmark.storage.s3_storage")

    class S3Storage(DataStorage):
        pass

    s3_storage_mod.S3Storage = S3Storage

    utils = types.ModuleType("dlio_benchmark.utils")
    utility = types.ModuleType("dlio_benchmark.utils.utility")

    class _NullProfileDecorator:
        def __call__(self, fn):
            return fn

    class Profile:
        def __init__(self, module):
            self.module = module
            self.log = _NullProfileDecorator()
            self.log_init = _NullProfileDecorator()

    utility.Profile = Profile

    sys.modules["dlio_benchmark"] = dlio_benchmark
    sys.modules["dlio_benchmark.common"] = common
    sys.modules["dlio_benchmark.common.constants"] = constants
    sys.modules["dlio_benchmark.common.enumerations"] = enumerations
    sys.modules["dlio_benchmark.storage"] = storage
    sys.modules["dlio_benchmark.storage.storage_handler"] = storage_handler
    sys.modules["dlio_benchmark.storage.s3_storage"] = s3_storage_mod
    sys.modules["dlio_benchmark.utils"] = utils
    sys.modules["dlio_benchmark.utils.utility"] = utility


_install_fake_dlio_benchmark()

import s3dlio  # noqa: E402  (real module; put_bytes()/MultipartUploadWriter are mocked per-test)
from s3dlio.integrations.dlio import s3dlio_storage  # noqa: E402

ENV_VARS = (
    "S3DLIO_MULTIPART_THRESHOLD_MB",
    "S3DLIO_MULTIPART_PART_SIZE_MB",
    "S3DLIO_MULTIPART_MAX_IN_FLIGHT",
    "S3DLIO_DISABLE_MULTIPART",
)


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {name: os.environ.get(name) for name in ENV_VARS}
    for name in ENV_VARS:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _make_storage():
    store = object.__new__(s3dlio_storage.S3dlioStorage)
    store.prefix = "s3://bucket"
    return store


class TestMultipartWriteFailureAborts:
    def test_write_failure_on_a_middle_part_calls_abort(self):
        """5 parts' worth of data, writer.write() raises on the 3rd call
        (part 3 of 5) -- writer.abort() must be called explicitly and
        deterministically, and writer.close() must NOT be called."""
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"  # always multipart
        os.environ["S3DLIO_MULTIPART_PART_SIZE_MB"] = "1"
        part_size = 1 * 1024 * 1024
        content = b"x" * (part_size * 5)  # exactly 5 parts

        mock_writer = mock.MagicMock()
        call_count = {"n": 0}

        def write_side_effect(chunk):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("simulated transient upload failure on part 3")

        mock_writer.write.side_effect = write_side_effect

        store = _make_storage()
        with (
            mock.patch.object(s3dlio, "put_bytes"),
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            mock_writer_cls.from_uri.return_value = mock_writer
            with pytest.raises(
                RuntimeError, match="simulated transient upload failure"
            ):
                store.put_data("key", content)

        assert call_count["n"] == 3, (
            "write() should have been called exactly 3 times (stopped at the failure)"
        )
        mock_writer.abort.assert_called_once()
        mock_writer.close.assert_not_called()

    def test_successful_multipart_write_does_not_call_abort(self):
        """Sanity check: the fix must not call abort() on the happy path."""
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        os.environ["S3DLIO_MULTIPART_PART_SIZE_MB"] = "1"
        part_size = 1 * 1024 * 1024
        content = b"x" * (part_size * 2)

        mock_writer = mock.MagicMock()
        store = _make_storage()
        with (
            mock.patch.object(s3dlio, "put_bytes"),
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            mock_writer_cls.from_uri.return_value = mock_writer
            store.put_data("key", content)

        mock_writer.close.assert_called_once()
        mock_writer.abort.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
