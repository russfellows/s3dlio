"""Regression tests for #286 and #287 (checkpoint Python API).

#286: ``user_meta`` is advertised as ``user_meta=None`` by ``PyCheckpointStore.save``,
``PyCheckpointWriter.finalize_distributed_checkpoint`` and the module functions
``save_checkpoint`` / ``finalize_distributed_checkpoint``, so each must be callable
without it (the wheel used to raise ``TypeError: missing 1 required positional
argument: 'user_meta'``).

#287: ``load_checkpoint_with_validation(uri, validate_integrity=True)`` must check the
loaded bytes against the checksum the checkpoint manifest recorded at write time. It
used to pass no expected checksum, so corrupted data came back without an error.

These tests exercise the installed wheel (file:// only) and pass only on a wheel built
from the fix.
"""

import json
import os
import tempfile

import pytest
import s3dlio

PAYLOAD = b"checkpoint payload for the 286/287 regression tests"


def _uri(root):
    return f"file://{root}"


def _latest_shard(root):
    """(shard URI, shard path on disk, manifest path) of the checkpoint's only shard."""
    store = s3dlio.PyCheckpointStore(_uri(root))
    manifest = store.reader().load_latest_manifest()
    assert manifest is not None
    key = manifest["shards"][0]["key"]
    manifests = os.path.join(root, "manifests")
    manifest_path = os.path.join(manifests, max(os.listdir(manifests)))
    return f"{_uri(root)}/{key}", os.path.join(root, key), manifest_path


# --------------------------------------------------------------------------- #286


def test_module_save_checkpoint_without_user_meta():
    with tempfile.TemporaryDirectory() as tmp:
        key = s3dlio.save_checkpoint(_uri(tmp), 1, 1, "test", PAYLOAD)
        assert "manifests/ckpt-1-" in key


def test_store_save_without_user_meta():
    with tempfile.TemporaryDirectory() as tmp:
        store = s3dlio.PyCheckpointStore(_uri(tmp))
        key = store.save(2, 1, "test", PAYLOAD)
        assert "manifests/ckpt-2-" in key
        assert bytes(store.load_latest()) == PAYLOAD


def test_writer_finalize_without_user_meta():
    with tempfile.TemporaryDirectory() as tmp:
        store = s3dlio.PyCheckpointStore(_uri(tmp))
        writer = store.writer(world_size=1, rank=0)
        meta = writer.save_distributed_shard(3, 1, "test", PAYLOAD)
        key = writer.finalize_distributed_checkpoint(3, 1, "test", [meta])
        assert "manifests/ckpt-3-" in key


def test_module_finalize_without_user_meta():
    with tempfile.TemporaryDirectory() as tmp:
        meta = s3dlio.save_distributed_shard(_uri(tmp), 4, 1, "test", PAYLOAD, 1, 0)
        key = s3dlio.finalize_distributed_checkpoint(
            _uri(tmp), 4, 1, "test", [meta], 1, 0
        )
        assert "manifests/ckpt-4-" in key


def test_user_meta_may_still_be_passed_explicitly():
    with tempfile.TemporaryDirectory() as tmp:
        assert s3dlio.save_checkpoint(_uri(tmp), 5, 1, "test", PAYLOAD, None)
        store = s3dlio.PyCheckpointStore(_uri(tmp))
        assert store.save(6, 1, "test", PAYLOAD, user_meta=None)


# --------------------------------------------------------------------------- #287


@pytest.mark.parametrize("compression_level", [None, 3], ids=["plain", "zstd"])
def test_validated_load_round_trip(compression_level):
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        store = s3dlio.PyCheckpointStore(
            _uri(root), compression_level=compression_level
        )
        store.save(7, 1, "test", PAYLOAD)
        shard_uri, _, _ = _latest_shard(root)

        # The shard of a compressed checkpoint is decoded before it is returned.
        assert (
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)
            == PAYLOAD
        )
        assert s3dlio.load_checkpoint_with_validation(
            shard_uri, validate_integrity=False
        ) == (PAYLOAD if compression_level is None else _stored_bytes(shard_uri))


def _stored_bytes(shard_uri):
    with open(shard_uri.removeprefix("file://"), "rb") as fh:
        return fh.read()


def test_validated_load_rejects_corrupted_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(_uri(root), 8, 1, "test", PAYLOAD)
        shard_uri, shard_path, _ = _latest_shard(root)

        corrupted = bytes([PAYLOAD[0] ^ 0xFF]) + PAYLOAD[1:]
        with open(shard_path, "wb") as fh:
            fh.write(corrupted)

        with pytest.raises(Exception, match="Integrity validation failed"):
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)
        with pytest.raises(Exception, match="Integrity validation failed"):
            s3dlio.load_checkpoint_with_validation(shard_uri)  # True is the default

        # validate_integrity=False reads the stored bytes as before.
        assert (
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=False)
            == corrupted
        )


def test_validated_load_reports_a_shard_with_no_recorded_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(_uri(root), 9, 1, "test", PAYLOAD)
        shard_uri, _, manifest_path = _latest_shard(root)

        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        for shard in manifest["shards"]:
            shard["checksum"] = None
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)

        with pytest.raises(Exception, match="recorded no checksum"):
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)

        # A checkpoint written before checksums were recorded still loads: nothing is
        # recorded, so the plain loaders have nothing to check and skip validation.
        assert bytes(s3dlio.load_checkpoint(_uri(root))) == PAYLOAD
        assert bytes(s3dlio.PyCheckpointStore(_uri(root)).load_latest()) == PAYLOAD
        assert (
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=False)
            == PAYLOAD
        )


def test_validated_load_rejects_a_uri_that_is_not_a_checkpoint_shard():
    with tempfile.TemporaryDirectory() as tmp:
        plain = os.path.join(tmp, "plain.bin")
        with open(plain, "wb") as fh:
            fh.write(PAYLOAD)
        with pytest.raises(Exception, match="not a checkpoint shard URI"):
            s3dlio.load_checkpoint_with_validation(_uri(plain), validate_integrity=True)
        assert (
            s3dlio.load_checkpoint_with_validation(
                _uri(plain), validate_integrity=False
            )
            == PAYLOAD
        )


# ------------------------------------------------- #287: which manifest is believed


def _rewrite_manifest(root, manifest_path, new_name, **changes):
    """Write a copy of the manifest as manifests/<new_name>.json with fields changed.

    ``checksum`` is applied to every shard entry; the other keywords set top-level
    fields. Returns the new path."""
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    checksum = changes.pop("checksum", None)
    manifest.update(changes)
    if checksum is not None:
        for shard in manifest["shards"]:
            shard["checksum"] = checksum
    path = os.path.join(root, "manifests", new_name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return path


BOGUS = "crc32c:00000000"


def test_validated_load_ignores_an_incomplete_newer_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(_uri(root), 10, 1, "test", PAYLOAD)
        shard_uri, _, manifest_path = _latest_shard(root)
        # Rename the real manifest so the lookup cannot find it by the shard's directory
        # name, then add a NEWER manifest that is incomplete and carries a wrong checksum.
        real = _rewrite_manifest(root, manifest_path, "ckpt-10-real")
        os.remove(manifest_path)
        assert os.path.exists(real)
        _rewrite_manifest(
            root,
            real,
            "ckpt-10-zzz",
            status="incomplete",
            wall_time="2999-01-01T00:00:00+00:00",
            checksum=BOGUS,
        )

        assert (
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)
            == PAYLOAD
        )


def test_validated_load_orders_manifests_by_instant_not_by_text():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(_uri(root), 11, 1, "test", PAYLOAD)
        shard_uri, _, manifest_path = _latest_shard(root)
        # 09:00-06:00 is 15:00Z: the newer instant, though its text sorts first.
        _rewrite_manifest(
            root, manifest_path, "ckpt-11-good", wall_time="2026-10-18T09:00:00-06:00"
        )
        _rewrite_manifest(
            root,
            manifest_path,
            "ckpt-11-bad",
            wall_time="2026-10-18T10:00:00+00:00",
            checksum=BOGUS,
        )
        os.remove(manifest_path)

        assert (
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)
            == PAYLOAD
        )


def test_validated_load_reports_an_unreadable_manifest_as_such():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "ckpt")
        s3dlio.save_checkpoint(_uri(root), 12, 1, "test", PAYLOAD)
        shard_uri, _, manifest_path = _latest_shard(root)
        os.rename(manifest_path, manifest_path.replace(".json", "-moved.json"))
        with open(manifest_path.replace(".json", "-broken.json"), "w") as fh:
            fh.write("{ not json")

        with pytest.raises(Exception, match="parse checkpoint manifest") as info:
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)
        assert "lists this shard" not in str(info.value)


def test_validated_load_works_when_the_checkpoint_directory_is_named_shards():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "shards")
        s3dlio.save_checkpoint(_uri(root), 13, 1, "test", PAYLOAD)
        shard_uri, _, _ = _latest_shard(root)
        assert shard_uri.count("/shards/") == 2
        assert (
            s3dlio.load_checkpoint_with_validation(shard_uri, validate_integrity=True)
            == PAYLOAD
        )
