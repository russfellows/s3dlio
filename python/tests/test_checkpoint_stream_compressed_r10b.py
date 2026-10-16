"""R10b: CheckpointStream.finalize must record the name the shard was stored under.

With a compressing store (``compression_level``), the streaming writer stores
``<key>.zst``; the returned metadata used to name (and stat) the un-suffixed key, so
finalize failed after a successful write or described an older object at that name.
"""

import os
import tempfile

import s3dlio


def _stream_one(store, payload):
    writer = store.writer(world_size=1, rank=0)
    stream = writer.get_distributed_shard_stream(step=3, epoch=1, framework="test")
    stream.write_chunk(payload)
    return stream.finalize()


def test_compressed_stream_records_the_stored_name():
    payload = b"streamed checkpoint bytes " * 200
    with tempfile.TemporaryDirectory() as tmp:
        store = s3dlio.PyCheckpointStore(f"file://{tmp}/ckpt", compression_level=3)
        meta = _stream_one(store, payload)
        assert meta["key"].endswith(".zst"), meta
        stored = os.path.join(tmp, "ckpt", meta["key"])
        assert os.path.isfile(stored), (meta, stored)
        assert meta["size"] == os.path.getsize(stored)
        assert meta["size"] < len(payload)


def test_uncompressed_stream_name_is_unchanged():
    payload = b"streamed checkpoint bytes " * 200
    with tempfile.TemporaryDirectory() as tmp:
        store = s3dlio.PyCheckpointStore(f"file://{tmp}/ckpt")
        meta = _stream_one(store, payload)
        assert not meta["key"].endswith(".zst"), meta
        stored = os.path.join(tmp, "ckpt", meta["key"])
        assert os.path.getsize(stored) == len(payload) == meta["size"]
