#!/usr/bin/env python3
"""
Tests for the DataGenMode configuration of ``s3dlio.put`` in the Python API.

These run entirely against the local ``file://`` backend in a temporary directory, so
they never contact a live object store (no bucket, no credentials needed).
"""

import time
from pathlib import Path

import pytest

s3dlio = pytest.importorskip("s3dlio")
put = s3dlio.put


def _prefix(tmp_path: Path) -> str:
    return f"file://{tmp_path}/"


def test_default_mode_writes_requested_size(tmp_path):
    """Default mode (streaming) writes objects of exactly the requested size."""
    put(
        _prefix(tmp_path),
        num=1,
        template="default-{}",
        size=1024 * 1024,
        object_type="binary",
        dedup_factor=1,
        compress_factor=1,
    )
    assert (tmp_path / "default-0").stat().st_size == 1024 * 1024


@pytest.mark.parametrize(
    ("mode", "chunk_size"),
    [("streaming", 8192), ("singlepass", 4096), ("single-pass", 4096)],
)
def test_explicit_modes_write_requested_size(tmp_path, mode, chunk_size):
    """Each explicit data generation mode writes objects of exactly the requested size."""
    put(
        _prefix(tmp_path),
        num=2,
        template="obj-{}",
        size=1024 * 1024,
        object_type="binary",
        dedup_factor=1,
        compress_factor=1,
        data_gen_mode=mode,
        chunk_size=chunk_size,
    )
    for i in range(2):
        assert (tmp_path / f"obj-{i}").stat().st_size == 1024 * 1024


def test_unknown_mode_falls_back_to_streaming(tmp_path):
    """An unrecognised data_gen_mode is not an error: it falls back to streaming."""
    put(
        _prefix(tmp_path),
        num=1,
        template="invalid-{}",
        size=1024,
        object_type="binary",
        dedup_factor=1,
        compress_factor=1,
        data_gen_mode="invalidmode",
    )
    assert (tmp_path / "invalid-0").stat().st_size == 1024


@pytest.mark.parametrize("obj_type", ["binary", "hdf5", "npz", "tfrecord"])
def test_different_object_types(tmp_path, obj_type):
    """Every object type is generated with the streaming mode."""
    try:
        put(
            _prefix(tmp_path),
            num=1,
            template=f"{obj_type}-{{}}",
            size=512 * 1024,
            object_type=obj_type,
            dedup_factor=2,
            compress_factor=2,
            data_gen_mode="streaming",
            chunk_size=4096,
        )
    except RuntimeError as exc:
        # Optional format compiled out of this wheel (e.g. hdf5): skip, anything else fails.
        if "not available in this build" in str(exc):
            pytest.skip(f"{obj_type} support not built into this wheel")
        raise
    written = tmp_path / f"{obj_type}-0"
    assert written.is_file()
    assert written.stat().st_size > 0


def test_modes_produce_same_size_objects(tmp_path):
    """Streaming and single-pass modes write the same number of bytes for the same request."""
    size = 4 * 1024 * 1024
    timings = {}
    for mode in ("streaming", "singlepass"):
        start = time.time()
        put(
            _prefix(tmp_path),
            num=1,
            template=f"perf-{mode}-{{}}",
            size=size,
            object_type="binary",
            dedup_factor=1,
            compress_factor=1,
            data_gen_mode=mode,
            chunk_size=8192,
        )
        timings[mode] = time.time() - start
        assert (tmp_path / f"perf-{mode}-0").stat().st_size == size
    print(f"timings: {timings}")
