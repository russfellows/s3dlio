from s3dlio import S3JaxIterable, list_keys_from_s3, list_uris, stat
import jax.numpy as jnp

URI = "s3://russ-b3/health/"

ds = S3JaxIterable.from_prefix(
    URI,
    prefetch=8,
    transform=lambda a: jnp.asarray(a),  # a is np.ndarray(dtype=uint8)
)

#keys = list_keys_from_s3(URI)
#for i, arr in zip(range(2), ds):
    #uri = f"{URI}{keys[i]}"
    #meta = stat(uri)
    #print(f"[{i}] uri={uri}")
    #print(f"    dtype={arr.dtype}, shape={arr.shape}, nbytes={arr.size}")
    #print(f"    s3.size={meta['size']}")
    #assert arr.size == meta["size"], "byte count mismatch with S3 stat!"

uris = list_uris(URI)  # FULL s3://... URIs
for i, arr in zip(range(2), ds):
    meta = stat(uris[i])
    print(f"[{i}] uri={uris[i]}")
    print(f"    dtype={arr.dtype}, shape={arr.shape}, nbytes={arr.size}")
    print(f"    s3.size={meta['size']}")
    assert arr.size == meta["size"]
