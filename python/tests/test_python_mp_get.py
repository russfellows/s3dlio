#!/usr/bin/env python3

"""
Test script for s3dlio Python mp_get functionality.

LIVE TEST: it writes a few small objects to a real S3 bucket and reads them back
with mp_get. It is SKIPPED unless S3DLIO_TEST_BUCKET names a bucket you own and
may write to (there is no default bucket), so a plain ``pytest python/tests``
never touches a live endpoint, even with ambient AWS credentials.
"""

import os
import sys
import uuid

import pytest

S3_BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()

pytestmark = pytest.mark.skipif(
    not S3_BUCKET,
    reason="live S3 test: set S3DLIO_TEST_BUCKET to a writable bucket to run",
)


def test_mp_get_performance():
    """mp_get reads back exactly the objects that were uploaded."""
    import s3dlio

    print("Testing s3dlio Python mp_get...")
    print("=" * 50)

    prefix = f"s3://{S3_BUCKET}/s3dlio-pytest-mp-get-{uuid.uuid4().hex[:12]}/"
    procs = 2
    num = 20  # Number of objects to test
    size = 64 * 1024
    template = "mp_object_{}.dat"

    print(f"URI: {prefix}")
    print(f"Processes: {procs}")
    print(f"Objects: {num} x {size} bytes")
    print()

    try:
        s3dlio.put(prefix, num, template, size=size)
        assert len(s3dlio.list(prefix, recursive=True)) == num

        result = s3dlio.mp_get(uri=prefix, procs=procs, num=num, template=template)

        print("Results:")
        print(f"  Total objects: {result['total_objects']:,}")
        print(f"  Total bytes: {result['total_bytes']:,}")
        print(f"  Duration: {result['duration_seconds']:.2f}s")
        print(f"  Throughput: {result['throughput_mb_s']:.1f} MB/s")
        print(f"  Ops/sec: {result['ops_per_sec']:.1f}")
        print()

        print("Per-worker performance:")
        for worker in result["workers"]:
            print(
                f"  Worker {worker['worker_id']}: {worker['objects']:,} objects, "
                f"{worker['bytes']:,} bytes, {worker['throughput_mb_s']:.1f} MB/s"
            )

        assert result["total_objects"] == num
        assert result["total_bytes"] == num * size
        assert sum(w["objects"] for w in result["workers"]) == num
    finally:
        try:
            for uri in s3dlio.list(prefix, recursive=True):
                s3dlio.delete(uri)
        except Exception as exc:
            print(f"cleanup warning for {prefix}: {exc}")

    print("✅ Python mp_get test completed successfully!")


if __name__ == "__main__":
    if not S3_BUCKET:
        print("S3DLIO_TEST_BUCKET is not set; refusing to contact a live S3 bucket.")
        sys.exit(2)
    test_mp_get_performance()
