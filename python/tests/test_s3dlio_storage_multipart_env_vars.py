#!/usr/bin/env python3
"""
RED-then-GREEN regression test for s3dlio issue #153 bug 3.1 (B10) --
the mlcommons/storage#715 seed bug.

Bug: S3dlioStorage.put_data() hardcoded its single-PUT-vs-multipart size
threshold (32 MiB), part size (32 MiB), and max-in-flight part count (8)
as module-level constants with no way to override them without a code
change. mlperf-storage benchmark sweeps and different backends (MinIO vs
AWS S3 vs on-prem appliances) need different thresholds.

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 4, bug B10; see python/s3dlio/integrations/dlio/_multipart_config.py
for the full rationale): read S3DLIO_MULTIPART_THRESHOLD_MB (reusing the
same env var name already documented for DLIO's own obj_store_lib.py),
S3DLIO_MULTIPART_PART_SIZE_MB, S3DLIO_MULTIPART_MAX_IN_FLIGHT, and an
explicit S3DLIO_DISABLE_MULTIPART switch.

dlio_benchmark is not a dependency of this repo (only DLIO_local_changes
has it), so this test fakes just enough of its import surface via
sys.modules before importing s3dlio_storage.py, matching the pattern in
test_dlio_storage_get_data_range.py. s3dlio.put_bytes and
s3dlio.MultipartUploadWriter are mocked directly so this test never
touches the network.
"""

import os
import sys
import types
import pytest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def _install_fake_dlio_benchmark():
    """Register minimal fake dlio_benchmark.* submodules in sys.modules
    so `import dlio_benchmark.common.constants` etc. succeeds without the
    real package installed. Returns nothing; idempotent."""
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

    # Registered even though this file's tests only need
    # storage_handler.DataStorage -- sys.modules is process-global and
    # test_dlio_storage_get_data_range.py's s3_torch_storage import needs
    # this submodule too. Whichever test file's installer runs first
    # "wins" the idempotent `if 'dlio_benchmark' in sys.modules: return`
    # guard, so every installer must register the full set every other
    # DLIO test file needs, regardless of pytest collection order.
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
from s3dlio.integrations.dlio import s3_torch_storage  # noqa: E402


def _make_storage():
    store = object.__new__(s3dlio_storage.S3dlioStorage)
    store.prefix = "s3://bucket"
    return store


def _make_torch_storage():
    """S3PyTorchConnectorStorage.put_data() passes `id` straight through
    (no self._make_uri call, unlike S3dlioStorage) -- no attributes need
    to be set up beyond bypassing __init__."""
    return object.__new__(s3_torch_storage.S3PyTorchConnectorStorage)


ENV_VARS = (
    "S3DLIO_MULTIPART_THRESHOLD_MB",
    "S3DLIO_MULTIPART_PART_SIZE_MB",
    "S3DLIO_MULTIPART_MAX_IN_FLIGHT",
    "S3DLIO_DISABLE_MULTIPART",
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Every test starts with none of the multipart env vars set, and
    they're restored afterward regardless of what a test sets."""
    saved = {name: os.environ.get(name) for name in ENV_VARS}
    for name in ENV_VARS:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class TestMultipartEnvVarWiring:
    def test_default_threshold_small_object_uses_put_bytes(self):
        """No env vars set -- a tiny object stays on the single-PUT path,
        exactly like before B10 (regression guard on the default)."""
        store = _make_storage()
        with (
            mock.patch.object(s3dlio, "put_bytes") as mock_put_bytes,
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            store.put_data("key", b"tiny-object")

        mock_put_bytes.assert_called_once_with("s3://bucket/key", b"tiny-object")
        mock_writer_cls.from_uri.assert_not_called()

    def test_threshold_zero_forces_multipart_even_for_tiny_object(self):
        """S3DLIO_MULTIPART_THRESHOLD_MB=0 means 'always use multipart'
        (matches the documented obj_store_lib.py contract). Pre-fix, the
        hardcoded 32 MiB threshold ignored this env var entirely, so a
        tiny object always took the single-PUT path regardless."""
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        store = _make_storage()
        mock_writer = mock.MagicMock()
        with (
            mock.patch.object(s3dlio, "put_bytes") as mock_put_bytes,
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            mock_writer_cls.from_uri.return_value = mock_writer
            store.put_data("key", b"tiny-object")

        mock_put_bytes.assert_not_called()
        mock_writer_cls.from_uri.assert_called_once()
        mock_writer.close.assert_called_once()

    def test_disable_switch_forces_put_bytes_even_when_threshold_is_zero(self):
        """S3DLIO_DISABLE_MULTIPART overrides everything else, including
        a threshold of 0 that would otherwise force multipart."""
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        os.environ["S3DLIO_DISABLE_MULTIPART"] = "true"
        store = _make_storage()
        with (
            mock.patch.object(s3dlio, "put_bytes") as mock_put_bytes,
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            store.put_data("key", b"tiny-object")

        mock_put_bytes.assert_called_once_with("s3://bucket/key", b"tiny-object")
        mock_writer_cls.from_uri.assert_not_called()

    def test_part_size_and_max_in_flight_env_vars_are_passed_through(self):
        """When the multipart path is taken, S3DLIO_MULTIPART_PART_SIZE_MB
        and S3DLIO_MULTIPART_MAX_IN_FLIGHT must reach
        MultipartUploadWriter.from_uri() as part_size/max_in_flight,
        not the hardcoded 32 MiB / 8 defaults."""
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        os.environ["S3DLIO_MULTIPART_PART_SIZE_MB"] = "5"
        os.environ["S3DLIO_MULTIPART_MAX_IN_FLIGHT"] = "3"
        store = _make_storage()
        mock_writer = mock.MagicMock()
        with (
            mock.patch.object(s3dlio, "put_bytes"),
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            mock_writer_cls.from_uri.return_value = mock_writer
            store.put_data("key", b"tiny-object")

        _, kwargs = mock_writer_cls.from_uri.call_args
        assert kwargs["part_size"] == 5 * 1024 * 1024
        assert kwargs["max_in_flight"] == 3


class TestTorchStorageMultipartEnvVarWiring:
    """Same bug (audit #153 bug 3.2 / B6), same fix, second file:
    s3_torch_storage.py::S3PyTorchConnectorStorage.put_data(). Prior to
    B6 this method had NO multipart path at all -- every write, of any
    size, went through a single s3dlio.put_bytes() call, hitting the S3
    5 GiB single-PUT limit for large checkpoint/dataset objects."""

    def test_default_threshold_small_object_uses_put_bytes(self):
        store = _make_torch_storage()
        with (
            mock.patch.object(s3dlio, "put_bytes") as mock_put_bytes,
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            store.put_data("s3://bucket/key", b"tiny-object")

        mock_put_bytes.assert_called_once_with("s3://bucket/key", b"tiny-object")
        mock_writer_cls.from_uri.assert_not_called()

    def test_threshold_zero_forces_multipart_even_for_tiny_object(self):
        """Pre-B6, there was no threshold check of any kind -- put_data()
        always called put_bytes() regardless of this env var."""
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        store = _make_torch_storage()
        mock_writer = mock.MagicMock()
        with (
            mock.patch.object(s3dlio, "put_bytes") as mock_put_bytes,
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            mock_writer_cls.from_uri.return_value = mock_writer
            store.put_data("s3://bucket/key", b"tiny-object")

        mock_put_bytes.assert_not_called()
        mock_writer_cls.from_uri.assert_called_once()
        mock_writer.close.assert_called_once()

    def test_disable_switch_forces_put_bytes_even_when_threshold_is_zero(self):
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        os.environ["S3DLIO_DISABLE_MULTIPART"] = "true"
        store = _make_torch_storage()
        with (
            mock.patch.object(s3dlio, "put_bytes") as mock_put_bytes,
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            store.put_data("s3://bucket/key", b"tiny-object")

        mock_put_bytes.assert_called_once_with("s3://bucket/key", b"tiny-object")
        mock_writer_cls.from_uri.assert_not_called()

    def test_part_size_and_max_in_flight_env_vars_are_passed_through(self):
        os.environ["S3DLIO_MULTIPART_THRESHOLD_MB"] = "0"
        os.environ["S3DLIO_MULTIPART_PART_SIZE_MB"] = "5"
        os.environ["S3DLIO_MULTIPART_MAX_IN_FLIGHT"] = "3"
        store = _make_torch_storage()
        mock_writer = mock.MagicMock()
        with (
            mock.patch.object(s3dlio, "put_bytes"),
            mock.patch.object(s3dlio, "MultipartUploadWriter") as mock_writer_cls,
        ):
            mock_writer_cls.from_uri.return_value = mock_writer
            store.put_data("s3://bucket/key", b"tiny-object")

        _, kwargs = mock_writer_cls.from_uri.call_args
        assert kwargs["part_size"] == 5 * 1024 * 1024
        assert kwargs["max_in_flight"] == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
