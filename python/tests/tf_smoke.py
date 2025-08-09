from s3dlio import make_tf_dataset, list_keys_from_s3, list_uris, stat
import tensorflow as tf

URI = "s3://russ-b3/health/"

#ds = make_tf_dataset(URI, prefetch=8)   # yields tf.uint8 vectors
#keys = list_keys_from_s3(URI)
#
#for i, x in enumerate(ds.take(2)):
    #uri = f"{URI}{keys[i]}"
    #meta = stat(uri)
    #print(f"[{i}] tf.tensor: shape={x.shape} dtype={x.dtype}")
    #print(f"    nbytes={int(x.shape[0])}, s3.size={meta['size']}")
    #tf.debugging.assert_equal(int(x.shape[0]), meta["size"])

ds = make_tf_dataset(URI, prefetch=8)
uris = list_uris(URI)

for i, x in enumerate(ds.take(2)):
    meta = stat(uris[i])
    nbytes = int(x.shape[0])  # uint8 1-D tensor => element count == bytes
    print(f"[{i}] tf.tensor: shape={x.shape} dtype={x.dtype}")
    print(f"    nbytes={nbytes}, s3.size={meta['size']}")
    tf.debugging.assert_equal(nbytes, meta["size"])
