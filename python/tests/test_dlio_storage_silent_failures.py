#!/usr/bin/env python3
"""
RED-then-GREEN regression tests for s3dlio issue #153 bugs 3.3/3.4/3.5
(Phase C, C1/C2/C3) -- silent exception-swallowing in the DLIO storage
integrations.

Bug (C1, bug 3.3): walk_node() caught any listing exception (including
PermissionDenied / network errors / anything) and returned `[]` --
indistinguishable from "this directory is legitimately empty". A caller
iterating an empty-looking dataset would silently train on zero samples
instead of seeing the real error.

Bug (C2, bug 3.4): create_node(exist_ok=True) caught ANY exception from
s3dlio.mkdir and returned True, not just the "already exists" case. An
auth failure, a network error, or (for cloud backends, where mkdir is
frequently unimplemented) a "not implemented" error was all silently
treated as "the directory already exists, success" -- masking real
failures behind a boolean True.

Bug (C3, bug 3.5): delete_node() caught any exception from s3dlio.delete
and returned False, indistinguishable from "delete failed because the
object doesn't exist" vs. "delete failed because of a real error"
(auth, network, etc).

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 5): narrow each catch to only the specific benign case (already-
exists for create_node, not-found for delete_node) and propagate
everything else; walk_node no longer swallows at all -- it logs and
re-raises.

dlio_benchmark is not a dependency of this repo, so this test fakes just
enough of its import surface via sys.modules, matching the pattern used
throughout this test suite (test_dlio_storage_get_data_range.py etc).
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

import s3dlio  # noqa: E402  (real module; list/mkdir/delete are mocked per-test)
from s3dlio.integrations.dlio import s3dlio_storage  # noqa: E402
from s3dlio.integrations.dlio import s3_torch_storage  # noqa: E402


class PermissionDenied(Exception):
    pass


def _make_storage():
    store = object.__new__(s3dlio_storage.S3dlioStorage)
    store.prefix = "s3://bucket"
    return store


def _make_torch_storage():
    return object.__new__(s3_torch_storage.S3PyTorchConnectorStorage)


# ----------------------------------------------------------------------
# C1 (bug 3.3): walk_node must propagate, not swallow-and-return-[]
# ----------------------------------------------------------------------


class TestWalkNodeDoesNotSwallow:
    def test_s3dlio_storage_walk_node_propagates_listing_error(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, "list", side_effect=PermissionDenied("denied")):
            with pytest.raises(PermissionDenied):
                store.walk_node("prefix")

    def test_torch_storage_walk_node_propagates_listing_error(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, "list", side_effect=PermissionDenied("denied")):
            with pytest.raises(PermissionDenied):
                store.walk_node("s3://bucket/prefix")


# ----------------------------------------------------------------------
# C2 (bug 3.4): create_node(exist_ok=True) must not swallow ANY
# exception -- only a genuine "already exists" signal should return True.
# ----------------------------------------------------------------------


class TestCreateNodeNarrowSwallow:
    def test_s3dlio_storage_propagates_non_exists_error_even_with_exist_ok(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, "mkdir", side_effect=PermissionDenied("denied")):
            with pytest.raises(PermissionDenied):
                store.create_node("dir", exist_ok=True)

    def test_s3dlio_storage_file_exists_error_returns_true_when_exist_ok(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, "mkdir", side_effect=FileExistsError("exists")):
            assert store.create_node("dir", exist_ok=True) is True

    def test_s3dlio_storage_file_exists_error_still_raises_when_not_exist_ok(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, "mkdir", side_effect=FileExistsError("exists")):
            with pytest.raises(FileExistsError):
                store.create_node("dir", exist_ok=False)

    def test_torch_storage_propagates_non_exists_error_even_with_exist_ok(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, "mkdir", side_effect=PermissionDenied("denied")):
            with pytest.raises(PermissionDenied):
                store.create_node("dir", exist_ok=True)

    def test_torch_storage_file_exists_error_returns_true_when_exist_ok(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, "mkdir", side_effect=FileExistsError("exists")):
            assert store.create_node("dir", exist_ok=True) is True


# ----------------------------------------------------------------------
# C3 (bug 3.5): delete_node must not swallow ANY exception -- only a
# genuine "not found" signal should return True (already gone).
# ----------------------------------------------------------------------


class TestDeleteNodeNarrowSwallow:
    def test_s3dlio_storage_propagates_real_error(self):
        store = _make_storage()
        with mock.patch.object(
            s3dlio, "delete", side_effect=PermissionDenied("denied")
        ):
            with pytest.raises(PermissionDenied):
                store.delete_node("key")

    def test_s3dlio_storage_not_found_returns_true(self):
        store = _make_storage()
        with mock.patch.object(s3dlio, "delete", side_effect=FileNotFoundError("gone")):
            assert store.delete_node("key") is True

    def test_torch_storage_propagates_real_error(self):
        store = _make_torch_storage()
        with mock.patch.object(
            s3dlio, "delete", side_effect=PermissionDenied("denied")
        ):
            with pytest.raises(PermissionDenied):
                store.delete_node("s3://bucket/key")

    def test_torch_storage_not_found_returns_true(self):
        store = _make_torch_storage()
        with mock.patch.object(s3dlio, "delete", side_effect=FileNotFoundError("gone")):
            assert store.delete_node("s3://bucket/key") is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
