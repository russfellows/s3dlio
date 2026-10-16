"""F2: the writer checksum must survive every Python shard-metadata boundary.

A streamed shard's writer computes a CRC-32C of the logical (uncompressed) bytes. That
checksum has to travel ``CheckpointStream.finalize()`` -> ``finalize_distributed_checkpoint``
-> JSON manifest -> ``load_latest_manifest()`` -> ``read_shard_by_rank``, where the read
must validate it. Before the fix every hop dropped it: the manifest said
``"checksum": null`` and a replaced shard was returned without complaint.

CRC-32C("123456789") = 0xe3069283 is the standard check value.
"""

import json
import os
import shutil
import tempfile

import pytest
import s3dlio

KNOWN_PAYLOAD = b"123456789"
KNOWN_CRC32C = "crc32c:e3069283"
STEP = 3
EPOCH = 1


def _stream_one(store, payload):
    writer = store.writer(world_size=1, rank=0)
    stream = writer.get_distributed_shard_stream(
        step=STEP, epoch=EPOCH, framework="test"
    )
    stream.write_chunk(payload)
    return stream.finalize()


def _finalize(store, metas):
    writer = store.writer(world_size=1, rank=0)
    return writer.finalize_distributed_checkpoint(STEP, EPOCH, "test", metas, None)


def _manifest_json(root, manifest_key):
    with open(os.path.join(root, manifest_key), encoding="utf-8") as fh:
        return json.load(fh)


def _read(store):
    reader = store.reader()
    manifest = reader.load_latest_manifest()
    assert manifest is not None
    return manifest, bytes(reader.read_shard_by_rank(manifest, 0))


@pytest.mark.parametrize("compression_level", [None, 3], ids=["plain", "zstd"])
def test_checksum_reaches_metadata_manifest_and_validating_read(compression_level):
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(
            f"file://{root}", compression_level=compression_level
        )

        meta = _stream_one(store, KNOWN_PAYLOAD)
        assert meta["checksum"] == KNOWN_CRC32C, meta

        manifest_key = _finalize(store, [meta])
        on_disk = _manifest_json(root, manifest_key)
        assert [s["checksum"] for s in on_disk["shards"]] == [KNOWN_CRC32C], on_disk

        manifest, data = _read(store)
        assert manifest["shards"][0]["checksum"] == KNOWN_CRC32C, manifest
        assert data == KNOWN_PAYLOAD


def test_replaced_plain_shard_is_rejected_by_the_python_read():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        _finalize(store, [meta])

        # Same length, different content: size and etag-style checks would not notice.
        with open(os.path.join(root, meta["key"]), "wb") as fh:
            fh.write(b"987654321")

        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            _read(store)


def test_replaced_zstd_shard_is_rejected_after_decoding():
    """A different but perfectly valid zstd stream must still be refused: the checksum
    covers the DECODED bytes, so validation has to happen after decompression."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        other_root = os.path.join(tmp, "other")
        store = s3dlio.PyCheckpointStore(f"file://{root}", compression_level=3)
        meta = _stream_one(store, KNOWN_PAYLOAD)
        _finalize(store, [meta])
        assert meta["key"].endswith(".zst")

        # A valid zstd stream with other content, produced by the library itself.
        other = s3dlio.PyCheckpointStore(f"file://{other_root}", compression_level=3)
        other_meta = _stream_one(other, b"987654321")
        shutil.copyfile(
            os.path.join(other_root, other_meta["key"]), os.path.join(root, meta["key"])
        )

        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            _read(store)


def test_manifest_without_checksum_stays_readable():
    """Old manifests (and metadata a caller built by hand) carry no checksum; they are
    still readable, just not validated."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        legacy = {k: v for k, v in meta.items() if k != "checksum"}

        manifest_key = _finalize(store, [legacy])
        assert [
            s["checksum"] for s in _manifest_json(root, manifest_key)["shards"]
        ] == [None]

        manifest, data = _read(store)
        assert manifest["shards"][0].get("checksum") is None
        assert data == KNOWN_PAYLOAD

        # Not validated: replacing the shard goes unnoticed, exactly as before.
        with open(os.path.join(root, meta["key"]), "wb") as fh:
            fh.write(b"987654321")
        _, data = _read(store)
        assert data == b"987654321"


def test_checksum_none_in_metadata_means_absent():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        meta["checksum"] = None

        manifest_key = _finalize(store, [meta])
        assert [
            s["checksum"] for s in _manifest_json(root, manifest_key)["shards"]
        ] == [None]
        _, data = _read(store)
        assert data == KNOWN_PAYLOAD


def test_caller_supplied_checksum_is_written_and_enforced():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        writer = store.writer(world_size=1, rank=0)
        meta = writer.save_distributed_shard(STEP, EPOCH, "test", KNOWN_PAYLOAD)
        assert meta.get("checksum") is None  # one-shot save computes none

        meta["checksum"] = "crc32c:00000000"  # wrong on purpose
        manifest_key = _finalize(store, [meta])
        assert [
            s["checksum"] for s in _manifest_json(root, manifest_key)["shards"]
        ] == ["crc32c:00000000"]
        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            _read(store)


def test_non_string_checksum_is_a_type_error():
    with tempfile.TemporaryDirectory() as tmp:
        store = s3dlio.PyCheckpointStore(f"file://{tmp}/ckpt")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        meta["checksum"] = 12345
        with pytest.raises(TypeError):
            _finalize(store, [meta])


# ---------------------------------------------------------------------------
# F2b: the plain "load latest" loaders validate too, a compressed one-shot save keeps its
# checksum, and a malformed checksum is refused when the checkpoint is finalized.
# ---------------------------------------------------------------------------


def _finalized_store(tmp, compression_level):
    root = os.path.join(tmp, "ckpt")
    store = s3dlio.PyCheckpointStore(
        f"file://{root}", compression_level=compression_level
    )
    meta = _stream_one(store, KNOWN_PAYLOAD)
    _finalize(store, [meta])
    return root, store, meta


@pytest.mark.parametrize("compression_level", [None, 3], ids=["plain", "zstd"])
def test_load_latest_and_load_checkpoint_load_unchanged_content(compression_level):
    with tempfile.TemporaryDirectory() as tmp:
        root, store, _ = _finalized_store(tmp, compression_level)
        assert bytes(store.load_latest()) == KNOWN_PAYLOAD
        assert bytes(s3dlio.load_checkpoint(f"file://{root}")) == KNOWN_PAYLOAD


def test_load_latest_and_load_checkpoint_reject_a_replaced_plain_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root, store, meta = _finalized_store(tmp, None)
        with open(os.path.join(root, meta["key"]), "wb") as fh:
            fh.write(b"987654321")
        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            store.load_latest()
        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            s3dlio.load_checkpoint(f"file://{root}")


def test_load_latest_and_load_checkpoint_reject_a_replaced_zstd_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root, store, meta = _finalized_store(tmp, 3)
        other_root = os.path.join(tmp, "other")
        other = s3dlio.PyCheckpointStore(f"file://{other_root}", compression_level=3)
        other_meta = _stream_one(other, b"987654321")
        shutil.copyfile(
            os.path.join(other_root, other_meta["key"]), os.path.join(root, meta["key"])
        )
        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            store.load_latest()
        with pytest.raises(Exception, match="(?i)integrity|checksum"):
            s3dlio.load_checkpoint(f"file://{root}")


def test_checksum_free_manifest_still_loads_through_load_latest():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        _finalize(store, [{k: v for k, v in meta.items() if k != "checksum"}])
        with open(os.path.join(root, meta["key"]), "wb") as fh:
            fh.write(b"987654321")  # unvalidated, as before
        assert bytes(store.load_latest()) == b"987654321"
        assert bytes(s3dlio.load_checkpoint(f"file://{root}")) == b"987654321"


def test_compressed_one_shot_save_returns_the_logical_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}", compression_level=3)
        writer = store.writer(world_size=1, rank=0)
        meta = writer.save_distributed_shard(STEP, EPOCH, "test", KNOWN_PAYLOAD)
        assert meta["key"].endswith(".zst"), meta
        assert meta["checksum"] == KNOWN_CRC32C, meta
        # And it validates: finalize with it, then the read decodes before checking.
        _finalize(store, [meta])
        assert bytes(store.load_latest()) == KNOWN_PAYLOAD


@pytest.mark.parametrize(
    "bad", ["md5:abc", "zzz", "crc32c:123456789", "0xe3069283", "crc32c:"]
)
def test_malformed_checksum_is_refused_at_finalize(bad):
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        meta["checksum"] = bad
        with pytest.raises(Exception, match="(?i)checksum"):
            _finalize(store, [meta])
        # Nothing was published.
        assert store.reader().load_latest_manifest() is None


def test_empty_checksum_string_means_absent():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(f"file://{root}")
        meta = _stream_one(store, KNOWN_PAYLOAD)
        meta["checksum"] = ""
        manifest_key = _finalize(store, [meta])
        assert [
            s["checksum"] for s in _manifest_json(root, manifest_key)["shards"]
        ] == [None]
        assert bytes(store.load_latest()) == KNOWN_PAYLOAD
