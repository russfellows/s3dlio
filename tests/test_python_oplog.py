#!/usr/bin/env python3
"""
Python API op-log test for the file:// and direct:// backends.

Uploads and downloads 10 files through each backend with the op-log active and
checks the finished log: a complete zstd stream, the exact header, 13 columns on
every row, and exactly 20 PUT / 20 GET / 2 LIST rows. (LIST is the directory
listing each download performs; per-object listing is not logged separately.)

Run with pytest (``pytest tests/test_python_oplog.py``) or as a script
(``python tests/test_python_oplog.py``, which runs the same test through pytest).
"""

import os
import shutil
import subprocess
import sys

import pytest

HEADER = (
    "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror"
    "\tstart\tfirst_byte\tend\tduration_ns"
)
EXPECTED_OPS = {"PUT": 20, "GET": 20, "LIST": 2}


def test_python_api_oplog_file_and_direct_backends(tmp_path):
    import s3dlio

    zstd = shutil.which("zstd")
    if zstd is None:
        pytest.skip(
            "the zstd command-line tool is not installed; it is needed to verify "
            "the op-log (install it, e.g. `apt-get install zstd`)"
        )

    src_dir = tmp_path / "src"
    dest_file_dir = tmp_path / "dest_file"
    dest_direct_dir = tmp_path / "dest_direct"
    download_dir = tmp_path / "download"
    log_dir = tmp_path / "oplog"
    for d in (src_dir, dest_file_dir, dest_direct_dir, download_dir, log_dir):
        d.mkdir()

    # 10 test files of 10 KB each.
    for i in range(10):
        (src_dir / f"test_{i:03d}.dat").write_bytes(b"X" * (10 * 1024))

    log_path = log_dir / "test_op_log_python.tsv.zst"
    s3dlio.init_op_log(str(log_path))
    assert s3dlio.is_op_log_active(), "op-log must be active after init_op_log"

    src_pattern = str(src_dir / "*.dat")
    try:
        for scheme, dest_dir in (("file", dest_file_dir), ("direct", dest_direct_dir)):
            # 10 PUTs through this backend
            s3dlio.upload(
                src_patterns=[src_pattern],
                dest_prefix=f"{scheme}://{dest_dir}/",
                max_in_flight=4,
                create_bucket=False,
            )
            # 10 GETs (plus one LIST) through this backend
            out = download_dir / f"from_{scheme}"
            out.mkdir()
            s3dlio.download(
                src_uri=f"{scheme}://{dest_dir}/",
                dest_dir=str(out),
                max_in_flight=4,
                recursive=True,
            )
            assert len(list(out.iterdir())) == 10, f"{scheme}:// download did not fetch 10 files"
    finally:
        s3dlio.finalize_op_log()
    assert not s3dlio.is_op_log_active()

    assert log_path.exists(), f"log file not created: {log_path}"
    # The file must be a COMPLETE zstd stream (finalize finished it).
    tested = subprocess.run([zstd, "-t", str(log_path)], capture_output=True, text=True)
    assert tested.returncode == 0, f"not a complete zstd stream: {tested.stderr}"

    result = subprocess.run([zstd, "-dc", str(log_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"could not decompress the log: {result.stderr}"
    lines = result.stdout.strip().split("\n")
    assert lines[0] == HEADER, f"unexpected header: {lines[0]!r}"

    malformed = [line for line in lines[1:] if len(line.split("\t")) != 13]
    ops = [line.split("\t")[2] for line in lines[1:]]
    actual_ops = {op: ops.count(op) for op in EXPECTED_OPS}
    assert not malformed, f"{len(malformed)} rows without 13 columns, first: {malformed[0]!r}"
    assert actual_ops == EXPECTED_OPS and len(ops) == sum(EXPECTED_OPS.values()), (
        f"expected {EXPECTED_OPS}; got {actual_ops} of {len(ops)} rows"
    )

    print(f"log: {log_path} ({os.path.getsize(log_path)} bytes), {len(ops)} rows: {actual_ops}")
    print(
        "endpoints:",
        {
            "file://": sum(1 for line in lines[1:] if "\tfile://" in line),
            "direct://": sum(1 for line in lines[1:] if "\tdirect://" in line),
        },
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
