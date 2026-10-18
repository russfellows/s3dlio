#!/usr/bin/env python3
"""
Verify the DataGenMode parameters of ``s3dlio.put`` are accepted and validated.

All writes go to the local ``file://`` backend in a temporary directory; nothing here
contacts a live object store.
"""

import inspect
from pathlib import Path

import pytest

s3dlio = pytest.importorskip("s3dlio")
put = s3dlio.put

ACCEPTED_CASES = [
    pytest.param({"object_type": "binary", "size": 1024}, id="default-streaming"),
    pytest.param(
        {
            "object_type": "binary",
            "size": 1024,
            "data_gen_mode": "streaming",
            "chunk_size": 8192,
        },
        id="explicit-streaming",
    ),
    pytest.param(
        {
            "object_type": "binary",
            "size": 1024,
            "data_gen_mode": "singlepass",
            "chunk_size": 4096,
        },
        id="singlepass",
    ),
    pytest.param(
        {
            "object_type": "binary",
            "size": 1024 * 1024,
            "data_gen_mode": "streaming",
            "chunk_size": 65536,
        },
        id="streaming-large-chunk",
    ),
]


@pytest.mark.parametrize("params", ACCEPTED_CASES)
def test_parameter_acceptance(tmp_path: Path, params):
    """The new parameters are accepted and the object really is written."""
    put(f"file://{tmp_path}/", num=1, template="test-{}", **params)
    assert (tmp_path / "test-0").stat().st_size == params["size"]


def test_invalid_data_gen_mode_falls_back_to_streaming(tmp_path: Path):
    """An unrecognised data_gen_mode is accepted and generates with streaming."""
    put(
        f"file://{tmp_path}/",
        num=1,
        template="test-{}",
        size=1024,
        object_type="binary",
        data_gen_mode="invalid_mode",
    )
    assert (tmp_path / "test-0").stat().st_size == 1024


def test_prefix_without_scheme_is_rejected(tmp_path: Path):
    """A prefix with no URI scheme is rejected before anything is written."""
    with pytest.raises(Exception, match="scheme"):
        put(str(tmp_path), num=1, template="test-{}", size=1024)
    assert list(tmp_path.iterdir()) == []


def test_put_signature_exposes_data_gen_parameters():
    """put() exposes data_gen_mode / chunk_size with defaults."""
    sig = inspect.signature(put)
    assert "data_gen_mode" in sig.parameters
    assert "chunk_size" in sig.parameters
