import os
import sys

import jax.numpy as jnp
from s3dlio import S3JaxIterable, list_full_uris, stat

# Manual smoke script (not a pytest test): reads from a live bucket you name explicitly.
# S3DLIO_TEST_BUCKET must contain a "health/" prefix with some objects; no default bucket.
BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()
if not BUCKET:
    sys.exit(
        "S3DLIO_TEST_BUCKET is not set; set it to a bucket you own to run this smoke script."
    )
URI = f"s3://{BUCKET}/health/"

ds = S3JaxIterable.from_prefix(
    URI,
    prefetch=8,
    transform=lambda a: jnp.asarray(a),  # a is np.ndarray(dtype=uint8)
)

# keys = list_keys(URI)
# for i, arr in zip(range(2), ds):
# uri = f"{URI}{keys[i]}"
# meta = stat(uri)
# print(f"[{i}] uri={uri}")
# print(f"    dtype={arr.dtype}, shape={arr.shape}, nbytes={arr.size}")
# print(f"    s3.size={meta['size']}")
# assert arr.size == meta["size"], "byte count mismatch with S3 stat!"

uris = list_full_uris(URI)  # FULL s3://... URIs
for i, arr in zip(range(2), ds):
    meta = stat(uris[i])
    print(f"[{i}] uri={uris[i]}")
    print(f"    dtype={arr.dtype}, shape={arr.shape}, nbytes={arr.size}")
    print(f"    s3.size={meta['size']}")
    assert arr.size == meta["size"]
