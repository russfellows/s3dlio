import s3dlio
import os
import shutil
import time

# --- Configuration ---
BUCKET_NAME = "russ-test6"
S3_URI = f"s3://{BUCKET_NAME}"
#TEST_PREFIX = "test-multipart/"
#S3_URI_PREFIX = f"s3://{BUCKET_NAME}/{TEST_PREFIX}"


# Optional: dial up logs
os.environ["RUST_LOG"] = "s3dlio=debug,aws_sdk_s3=info"

location = f"{S3_URI}/test-multipart.bin"

print(f"--- Creating S3 bucket {BUCKET_NAME}... ---")
s3dlio.create_bucket(BUCKET_NAME)

N = 64 * 1024 * 1024  # 64 MiB
w = s3dlio.MultipartUploadWriter.from_uri(
    location,
    part_size = 32 << 20,
    max_in_flight = 16,
)

# true zero-copy into Rust-owned buffer
mv = w.reserve(N)
mv[:] = b"\xAB" * N
w.commit(N)
del mv      # don’t use it again after commit

info = w.close()
print("MPU done:", info)

