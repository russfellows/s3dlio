#!/usr/bin/env python3
"""
Test the Python API with a real S3 backend using different data generation modes.

LIVE TEST: this module writes to a real S3 bucket. It is SKIPPED unless
S3DLIO_TEST_BUCKET names a bucket you own and may write to (there is no default
bucket), so a plain ``pytest python/tests`` never touches a live endpoint, even
on a machine with ambient AWS credentials. For an S3-compatible server also set
AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
"""

import os
import sys
import time
import uuid

import pytest

S3_BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()

pytestmark = pytest.mark.skipif(
    not S3_BUCKET,
    reason="live S3 test: set S3DLIO_TEST_BUCKET to a writable bucket to run",
)

NUM_OBJECTS = 3
OBJECT_SIZE = 4 * 1024 * 1024  # 4 MiB


@pytest.fixture
def live_prefix():
    """A unique per-test prefix in the opted-in bucket; removed afterwards."""
    import s3dlio

    prefix = f"s3://{S3_BUCKET}/s3dlio-pytest-{uuid.uuid4().hex[:12]}/"
    yield prefix
    try:
        for uri in s3dlio.list(prefix, recursive=True):
            s3dlio.delete(uri)
    except Exception as exc:
        print(f"cleanup warning for {prefix}: {exc}")


def _put_and_verify(prefix, **kwargs):
    """put() NUM_OBJECTS objects under prefix and check what landed in S3."""
    import s3dlio

    start = time.time()
    s3dlio.put(
        prefix=prefix,
        num=NUM_OBJECTS,
        template="obj-{}.bin",
        size=OBJECT_SIZE,
        **kwargs,
    )
    elapsed = time.time() - start
    print(f"put of {NUM_OBJECTS} x {OBJECT_SIZE} bytes took {elapsed:.2f}s")

    for i in range(NUM_OBJECTS):
        uri = f"{prefix}obj-{i}.bin"
        assert s3dlio.stat(uri)["size"] == OBJECT_SIZE, f"wrong size for {uri}"
    # Read one object back: the bytes that arrived must be the full object.
    assert len(s3dlio.get(f"{prefix}obj-0.bin")) == OBJECT_SIZE
    assert len(s3dlio.list(prefix, recursive=True)) == NUM_OBJECTS


def test_streaming_mode(live_prefix):
    """Test streaming mode via Python API"""
    _put_and_verify(live_prefix, data_gen_mode="streaming", chunk_size=65536)


def test_single_pass_mode(live_prefix):
    """Test single-pass mode via Python API"""
    _put_and_verify(live_prefix, data_gen_mode="single-pass", chunk_size=65536)


def test_default_mode(live_prefix):
    """Test default mode (should be streaming) via Python API"""
    # Defaults for data_gen_mode and chunk_size.
    _put_and_verify(live_prefix)


if __name__ == "__main__":
    if not S3_BUCKET:
        print("S3DLIO_TEST_BUCKET is not set; refusing to contact a live S3 bucket.")
        sys.exit(2)
    sys.exit(pytest.main([__file__, "-v", "-s"]))
