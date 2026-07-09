#!/usr/bin/env python3
"""
RED-then-GREEN regression test for s3dlio issue #153 bug 3.8 (D7).

Bug: S3PyTorchConnectorStorage.walk_node() compared each full URI returned
by s3dlio.list() (e.g. "s3://bucket/train/a/x") against a bare relative
`prefix` (e.g. "train/") built from the parsed URI's path component. Since
a full URI never starts with a bare relative path, `key.startswith(prefix)`
was always False, so every call fell through to `os.path.basename(key)`,
silently dropping any subdirectory structure: "train/a/x" became just "x"
instead of the expected "a/x".

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 6, bug D7): compare against the full URI prefix instead, matching the
sibling s3dlio_storage.py::walk_node implementation, which already does
this correctly.

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

import s3dlio  # noqa: E402  (real module; list() is mocked per-test)
from s3dlio.integrations.dlio import s3_torch_storage  # noqa: E402


def _make_torch_storage():
    return object.__new__(s3_torch_storage.S3PyTorchConnectorStorage)


class TestWalkNodePreservesSubdirectoryStructure:
    def test_nested_keys_keep_their_subdirectory_prefix(self):
        store = _make_torch_storage()
        with mock.patch.object(
            s3dlio,
            "list",
            return_value=["s3://b/train/a/x", "s3://b/train/b/y"],
        ):
            result = store.walk_node("s3://b/train/")

        assert sorted(result) == ["a/x", "b/y"], (
            "walk_node must preserve the subdirectory component of each key "
            f"(got {result!r} -- if this is ['x', 'y'], the fix regressed to "
            "os.path.basename(), which drops subdirectory structure)"
        )

    def test_flat_keys_still_work(self):
        """Sanity check: keys with no subdirectory component still resolve
        to just their filename, same as before."""
        store = _make_torch_storage()
        with mock.patch.object(
            s3dlio,
            "list",
            return_value=["s3://b/train/x", "s3://b/train/y"],
        ):
            result = store.walk_node("s3://b/train/")

        assert sorted(result) == ["x", "y"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
