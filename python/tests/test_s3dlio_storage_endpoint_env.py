#!/usr/bin/env python3
"""
RED-then-GREEN regression test for s3dlio issue #153 bug 3.9 (D8).

Bug: S3dlioStorage.__init__ set AWS_ENDPOINT_URL via an unconditional
`os.environ["AWS_ENDPOINT_URL"] = selected_endpoint` -- clobbering any
value the user had already set in their own environment before launching
DLIO. This was inconsistent with the `setdefault` (don't-overwrite)
contract already used for AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/
AWS_REGION in the very same constructor.

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 6, bug D8): the pre-existing environment value wins, with a warning
logged when the selected/configured endpoint differs from it, so an
operator has a trail explaining why their endpoint_uris config appeared
to have no effect.

Tests the extracted _apply_selected_endpoint_env() helper directly rather
than through S3dlioStorage.__init__, since constructing a real instance
requires replicating DLIO's own argparse-backed `self._args` machinery
(out of scope for this bug -- see the fake DataStorage stub in other
test files in this suite, which only provides an empty `self._args`).
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

from s3dlio.integrations.dlio import s3dlio_storage  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_endpoint_env():
    saved = os.environ.get("AWS_ENDPOINT_URL")
    os.environ.pop("AWS_ENDPOINT_URL", None)
    yield
    if saved is None:
        os.environ.pop("AWS_ENDPOINT_URL", None)
    else:
        os.environ["AWS_ENDPOINT_URL"] = saved


class TestApplySelectedEndpointEnv:
    def test_does_not_clobber_a_preexisting_env_value(self):
        os.environ["AWS_ENDPOINT_URL"] = "http://user-set-endpoint:9000"

        with mock.patch.object(s3dlio_storage._logger, "warning") as mock_warn:
            s3dlio_storage._apply_selected_endpoint_env("http://config-endpoint:9000")

        assert os.environ["AWS_ENDPOINT_URL"] == "http://user-set-endpoint:9000", (
            "a pre-existing AWS_ENDPOINT_URL must not be clobbered by the "
            "configured/selected endpoint"
        )
        mock_warn.assert_called_once()

    def test_sets_env_when_nothing_preexisting(self):
        assert "AWS_ENDPOINT_URL" not in os.environ

        with mock.patch.object(s3dlio_storage._logger, "warning") as mock_warn:
            s3dlio_storage._apply_selected_endpoint_env("http://config-endpoint:9000")

        assert os.environ["AWS_ENDPOINT_URL"] == "http://config-endpoint:9000"
        mock_warn.assert_not_called()

    def test_no_warning_when_preexisting_value_already_matches(self):
        os.environ["AWS_ENDPOINT_URL"] = "http://same-endpoint:9000"

        with mock.patch.object(s3dlio_storage._logger, "warning") as mock_warn:
            s3dlio_storage._apply_selected_endpoint_env("http://same-endpoint:9000")

        assert os.environ["AWS_ENDPOINT_URL"] == "http://same-endpoint:9000"
        mock_warn.assert_not_called()

    def test_noop_when_no_endpoint_selected(self):
        os.environ["AWS_ENDPOINT_URL"] = "http://user-set-endpoint:9000"

        s3dlio_storage._apply_selected_endpoint_env(None)

        assert os.environ["AWS_ENDPOINT_URL"] == "http://user-set-endpoint:9000"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
