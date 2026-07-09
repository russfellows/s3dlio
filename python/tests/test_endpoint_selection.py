#!/usr/bin/env python3
"""
RED-then-GREEN regression test for s3dlio issue #153 bug 3.10 (B9).

Bug: S3dlioStorage._select_endpoint_via_strategy()'s "round_robin"
strategy actually used `os.getpid() % len(endpoint_uris)` -- a static,
per-process PID hash, not round-robin at all. Every call from a given
process always maps to the same endpoint regardless of MPI rank, so
across a distributed run the endpoint assignment is essentially
arbitrary (whatever the OS handed out as a PID), not an even rotation
across ranks.

Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
Sec 4, bug B9):
1. `round_robin` becomes rank-based deterministic assignment: when any of
   OMPI_COMM_WORLD_RANK / SLURM_PROCID / PMI_RANK is set (the same set
   already read by _select_endpoint_via_mpi), use `rank % len(endpoints)`.
   When no rank var is available, log a warning naming the PID fallback
   and use `pid % len(endpoints)` (documented as best-effort).
2. `random` keeps random.choice() but seeds from rank-if-available or
   pid-otherwise, so re-runs at least aren't identically skewed across
   processes.
3. `least_connections` continues to fall back to round_robin, with the
   fallback message now surfacing via the logging module.

dlio_benchmark is not a dependency of this repo, so this test fakes just
enough of its import surface via sys.modules, matching the pattern in
test_dlio_storage_get_data_range.py / test_s3dlio_storage_multipart_env_vars.py.
"""

import logging
import os
import sys
import types
from collections import Counter
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

RANK_VARS = ("OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "PMI_RANK")


def _make_storage():
    return object.__new__(s3dlio_storage.S3dlioStorage)


@pytest.fixture(autouse=True)
def _clean_rank_env():
    saved = {name: os.environ.get(name) for name in RANK_VARS}
    for name in RANK_VARS:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class TestRoundRobinRankBased:
    def test_even_split_across_ranks(self):
        """32 ranks, 8 endpoints -- each endpoint must serve exactly 4
        ranks. Pre-fix, this used pid % N regardless of rank, so the
        distribution was arbitrary, not an even split."""
        store = _make_storage()
        endpoints = [f"http://ep{i}:9000" for i in range(8)]
        selections = []
        for rank in range(32):
            os.environ["OMPI_COMM_WORLD_RANK"] = str(rank)
            selections.append(
                store._select_endpoint_via_strategy(endpoints, "round_robin")
            )

        counts = Counter(selections)
        assert counts == {ep: 4 for ep in endpoints}

    def test_slurm_procid_also_used(self):
        store = _make_storage()
        endpoints = [f"http://ep{i}:9000" for i in range(4)]
        os.environ["SLURM_PROCID"] = "5"
        # Mock the pid to a value whose pid % 4 differs from rank % 4
        # (5 % 4 == 1) -- otherwise a pre-fix, pid-based implementation
        # could coincidentally "pass" this test using the real live PID.
        with mock.patch("os.getpid", return_value=100):  # 100 % 4 == 0
            result = store._select_endpoint_via_strategy(endpoints, "round_robin")
        assert result == endpoints[5 % 4]

    def test_no_rank_var_falls_back_to_pid_with_warning(self, caplog):
        """No MPI/SLURM/PMI env var set -- falls back to pid % N, but
        must emit a warning naming the fallback (operator visibility)."""
        store = _make_storage()
        endpoints = [f"http://ep{i}:9000" for i in range(4)]
        with (
            mock.patch("os.getpid", return_value=12345),
            caplog.at_level(logging.WARNING),
        ):
            result = store._select_endpoint_via_strategy(endpoints, "round_robin")

        assert result == endpoints[12345 % 4]
        assert any(
            "round_robin" in rec.message.lower() or "pid" in rec.message.lower()
            for rec in caplog.records
        ), "expected a warning naming the PID fallback"


class TestRandomStrategySeeding:
    def test_seeds_from_rank_when_available(self):
        store = _make_storage()
        endpoints = ["http://ep0:9000", "http://ep1:9000"]
        os.environ["OMPI_COMM_WORLD_RANK"] = "7"
        with mock.patch("random.seed") as mock_seed:
            store._select_endpoint_via_strategy(endpoints, "random")
        mock_seed.assert_called_once_with(7)

    def test_seeds_from_pid_when_rank_absent(self):
        store = _make_storage()
        endpoints = ["http://ep0:9000", "http://ep1:9000"]
        with (
            mock.patch("os.getpid", return_value=999),
            mock.patch("random.seed") as mock_seed,
        ):
            store._select_endpoint_via_strategy(endpoints, "random")
        mock_seed.assert_called_once_with(999)


class TestLeastConnectionsFallback:
    def test_falls_back_to_round_robin_with_warning(self, caplog):
        store = _make_storage()
        endpoints = [f"http://ep{i}:9000" for i in range(4)]
        os.environ["OMPI_COMM_WORLD_RANK"] = "2"
        with caplog.at_level(logging.WARNING):
            result = store._select_endpoint_via_strategy(endpoints, "least_connections")

        assert result == endpoints[2 % 4]
        assert any("least_connections" in rec.message.lower() for rec in caplog.records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
