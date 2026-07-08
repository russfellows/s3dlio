#!/usr/bin/env python3
"""
RED-then-GREEN regression test for s3dlio issue #153 sub-bug 3.6
(audit finding f4).

Bug: S3dlioStorage.get_data() and S3PyTorchConnectorStorage.get_data()
both guarded the get_range() path with
`if offset is not None and length is not None:`. Since the docstring
documents offset and length as INDEPENDENTLY optional, a caller passing
offset-only (e.g. get_data(id, offset=1024), expecting bytes from 1024
to EOF) hit the else branch instead and got the ENTIRE object back from
byte 0 -- silently wrong data, no error. Same bug for length-only.

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 4, bug B5): route through get_range() whenever offset is given
(length may be None, meaning "to end", which matches s3dlio.get_range's
own contract) OR whenever length is given without offset (defaults
offset=0). Only call the full-object get() when NEITHER is given.

dlio_benchmark is not a dependency of this repo (only DLIO_local_changes
has it), so this test fakes just enough of its import surface
(dlio_benchmark.common.constants/enumerations, storage.storage_handler,
utils.utility) via sys.modules before importing s3dlio_storage.py --
matching how a real DLIO installation's plugin loader would provide
those symbols. The real s3dlio module's get()/get_range() are mocked
directly (H4-style Python mock harness) so this test never touches the
network.
"""

import os
import sys
import types
import pytest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


def _install_fake_dlio_benchmark():
    """Register minimal fake dlio_benchmark.* submodules in sys.modules
    so `import dlio_benchmark.common.constants` etc. succeeds without the
    real package installed. Returns nothing; idempotent."""
    if 'dlio_benchmark' in sys.modules:
        return

    dlio_benchmark = types.ModuleType('dlio_benchmark')

    common = types.ModuleType('dlio_benchmark.common')
    constants = types.ModuleType('dlio_benchmark.common.constants')
    constants.MODULE_STORAGE = 'storage'
    enumerations = types.ModuleType('dlio_benchmark.common.enumerations')

    class NamespaceType:
        FLAT = 'flat'

    class MetadataType:
        FILE = 'file'
        DIRECTORY = 'directory'

    enumerations.NamespaceType = NamespaceType
    enumerations.MetadataType = MetadataType

    storage = types.ModuleType('dlio_benchmark.storage')
    storage_handler = types.ModuleType('dlio_benchmark.storage.storage_handler')

    class Namespace:
        def __init__(self, name, ns_type):
            self.name = name
            self.ns_type = ns_type

    class DataStorage:
        """Fake base class -- real DataStorage does argparse-backed config
        wiring via self._args; tests set attributes directly instead of
        going through __init__, so this stub only needs to exist as a
        valid base class for `class S3dlioStorage(DataStorage)`."""

        def __init__(self, framework=None):
            self.framework = framework
            self._args = types.SimpleNamespace()

    storage_handler.DataStorage = DataStorage
    storage_handler.Namespace = Namespace

    s3_storage_mod = types.ModuleType('dlio_benchmark.storage.s3_storage')

    class S3Storage(DataStorage):
        """Fake base class for S3PyTorchConnectorStorage."""
        pass

    s3_storage_mod.S3Storage = S3Storage

    utils = types.ModuleType('dlio_benchmark.utils')
    utility = types.ModuleType('dlio_benchmark.utils.utility')

    class _NullProfileDecorator:
        """Fake dlp.log / dlp.log_init -- a plain passthrough decorator."""

        def __call__(self, fn):
            return fn

    class Profile:
        def __init__(self, module):
            self.module = module
            self.log = _NullProfileDecorator()
            self.log_init = _NullProfileDecorator()

    utility.Profile = Profile

    sys.modules['dlio_benchmark'] = dlio_benchmark
    sys.modules['dlio_benchmark.common'] = common
    sys.modules['dlio_benchmark.common.constants'] = constants
    sys.modules['dlio_benchmark.common.enumerations'] = enumerations
    sys.modules['dlio_benchmark.storage'] = storage
    sys.modules['dlio_benchmark.storage.storage_handler'] = storage_handler
    sys.modules['dlio_benchmark.storage.s3_storage'] = s3_storage_mod
    sys.modules['dlio_benchmark.utils'] = utils
    sys.modules['dlio_benchmark.utils.utility'] = utility


_install_fake_dlio_benchmark()

import s3dlio  # noqa: E402  (real module; get()/get_range() are mocked per-test)
from s3dlio.integrations.dlio import s3dlio_storage  # noqa: E402
from s3dlio.integrations.dlio import s3_torch_storage  # noqa: E402


def _make_storage():
    """Build an S3dlioStorage instance without going through __init__
    (which wires up MPI/multi-endpoint config this test doesn't need) --
    get_data() only touches self.prefix via self._make_uri()."""
    store = object.__new__(s3dlio_storage.S3dlioStorage)
    store.prefix = "s3://bucket"
    return store


def _make_torch_storage():
    """S3PyTorchConnectorStorage.get_data() doesn't call self._make_uri
    at all -- it passes `id` straight through to s3dlio.get_range/get
    -- so no attributes need to be set up beyond bypassing __init__."""
    return object.__new__(s3_torch_storage.S3PyTorchConnectorStorage)


class TestGetDataRangeGuard:
    def test_offset_only_calls_get_range_not_full_get(self):
        """offset given, length omitted (None) -- must call get_range()
        with offset and length=None (read to end), NOT the full get()."""
        store = _make_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get_range.return_value = b'tail-bytes'
            result = store.get_data('key', offset=1024)

        mock_get.assert_not_called()
        mock_get_range.assert_called_once_with(
            "s3://bucket/key", offset=1024, length=None
        )
        assert result == b'tail-bytes'

    def test_length_only_calls_get_range_from_zero(self):
        """length given, offset omitted (None) -- must call get_range()
        with offset=0 and the given length, NOT the full get()."""
        store = _make_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get_range.return_value = b'head-bytes'
            result = store.get_data('key', length=512)

        mock_get.assert_not_called()
        mock_get_range.assert_called_once_with(
            "s3://bucket/key", offset=0, length=512
        )
        assert result == b'head-bytes'

    def test_both_given_calls_get_range_with_both(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get_range.return_value = b'range-bytes'
            result = store.get_data('key', offset=100, length=50)

        mock_get.assert_not_called()
        mock_get_range.assert_called_once_with(
            "s3://bucket/key", offset=100, length=50
        )
        assert result == b'range-bytes'

    def test_neither_given_calls_full_get(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get.return_value = b'full-object-bytes'
            result = store.get_data('key')

        mock_get_range.assert_not_called()
        mock_get.assert_called_once_with("s3://bucket/key")
        assert result == b'full-object-bytes'


class TestTorchStorageGetDataRangeGuard:
    """Same bug (audit f4), same fix, second file:
    s3_torch_storage.py::S3PyTorchConnectorStorage.get_data()."""

    def test_offset_only_calls_get_range_not_full_get(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get_range.return_value = b'tail-bytes'
            result = store.get_data('s3://bucket/key', offset=1024)

        mock_get.assert_not_called()
        mock_get_range.assert_called_once_with(
            's3://bucket/key', offset=1024, length=None
        )
        assert result == b'tail-bytes'

    def test_length_only_calls_get_range_from_zero(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get_range.return_value = b'head-bytes'
            result = store.get_data('s3://bucket/key', length=512)

        mock_get.assert_not_called()
        mock_get_range.assert_called_once_with(
            's3://bucket/key', offset=0, length=512
        )
        assert result == b'head-bytes'

    def test_neither_given_calls_full_get(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, 'get_range') as mock_get_range, \
             mock.patch.object(s3dlio, 'get') as mock_get:
            mock_get.return_value = b'full-object-bytes'
            result = store.get_data('s3://bucket/key')

        mock_get_range.assert_not_called()
        mock_get.assert_called_once_with('s3://bucket/key')
        assert result == b'full-object-bytes'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
