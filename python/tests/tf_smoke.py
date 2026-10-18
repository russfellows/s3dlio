import os
import sys

import tensorflow as tf
from s3dlio import list_full_uris, make_tf_dataset, stat

# Manual smoke script (not a pytest test): reads from a live bucket you name explicitly.
# S3DLIO_TEST_BUCKET must contain a "health/" prefix with some objects; no default bucket.
BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()
if not BUCKET:
    sys.exit(
        "S3DLIO_TEST_BUCKET is not set; set it to a bucket you own to run this smoke script."
    )
URI = f"s3://{BUCKET}/health/"

# ds = make_tf_dataset(URI, prefetch=8)   # yields tf.uint8 vectors
# keys = list_keys(URI)
#
# for i, x in enumerate(ds.take(2)):
# uri = f"{URI}{keys[i]}"
# meta = stat(uri)
# print(f"[{i}] tf.tensor: shape={x.shape} dtype={x.dtype}")
# print(f"    nbytes={int(x.shape[0])}, s3.size={meta['size']}")
# tf.debugging.assert_equal(int(x.shape[0]), meta["size"])

ds = make_tf_dataset(URI, prefetch=8)
uris = list_full_uris(URI)

for i, x in enumerate(ds.take(2)):
    meta = stat(uris[i])
    nbytes = int(x.shape[0])  # uint8 1-D tensor => element count == bytes
    print(f"[{i}] tf.tensor: shape={x.shape} dtype={x.dtype}")
    print(f"    nbytes={nbytes}, s3.size={meta['size']}")
    tf.debugging.assert_equal(nbytes, meta["size"])
