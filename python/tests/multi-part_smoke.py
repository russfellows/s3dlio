"""Manual multipart-upload smoke script (NOT a pytest test; it writes to a live bucket).

Usage:
    S3DLIO_TEST_BUCKET=<bucket you own> uv run python python/tests/multi-part_smoke.py

For an S3-compatible server also set AWS_ENDPOINT_URL and the AWS credentials.
There is no default bucket: the script exits before any network call if
S3DLIO_TEST_BUCKET is unset.
"""

import os
import sys

import s3dlio

# --- Configuration ---
BUCKET_NAME = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()
if not BUCKET_NAME:
    sys.exit(
        "S3DLIO_TEST_BUCKET is not set; set it to a bucket you own to run this smoke script."
    )
S3_URI = f"s3://{BUCKET_NAME}"
# TEST_PREFIX = "test-multipart/"
# S3_URI_PREFIX = f"s3://{BUCKET_NAME}/{TEST_PREFIX}"


# Optional: dial up logs
os.environ["RUST_LOG"] = "s3dlio=debug,aws_sdk_s3=info"

location = f"{S3_URI}/test-multipart.bin"

print(f"--- Creating S3 bucket {BUCKET_NAME}... ---")
s3dlio.create_bucket(BUCKET_NAME)

N = 64 * 1024 * 1024  # 64 MiB
w = s3dlio.MultipartUploadWriter.from_uri(
    location,
    part_size=32 << 20,
    max_in_flight=16,
)

# true zero-copy into Rust-owned buffer
mv = w.reserve(N)
mv[:] = b"\xab" * N
w.commit(N)
del mv  # don’t use it again after commit

info = w.close()
print("MPU done:", info)
assert s3dlio.stat(location)["size"] == N, "uploaded object has the wrong size"
