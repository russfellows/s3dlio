"""Smoke tests for ``PyCheckpointStore.delete_checkpoint`` (file:// only).

Contract (docs/PYTHON_API_GUIDE.md, "Deleting a checkpoint"): ``delete_checkpoint(step)``
returns True when a checkpoint of ``step`` was found and deleted, False when there is none;
``latest.json`` is repointed when it named the deleted manifest; a manifest whose file name
disagrees with its recorded ``global_step`` raises with nothing deleted.
"""

import json
import os
import shutil
import tempfile

import pytest
import s3dlio

PAYLOAD_1 = b"payload of checkpoint step one"
PAYLOAD_2 = b"payload of checkpoint step two, a little longer"
PAYLOAD_4 = b"payload of checkpoint step four"


def _uri(root):
    return f"file://{root}"


def _manifest_path(root, step):
    """Path of the only manifest file named for ``step``."""
    manifests = os.path.join(root, "manifests")
    found = [n for n in os.listdir(manifests) if n.startswith(f"ckpt-{step}-")]
    assert len(found) == 1, found
    return os.path.join(manifests, found[0])


def _shard_paths(root, step):
    """On-disk shard paths listed by the manifest of ``step``."""
    with open(_manifest_path(root, step)) as fh:
        manifest = json.load(fh)
    return [os.path.join(root, shard["key"]) for shard in manifest["shards"]]


def _snapshot(paths):
    out = {}
    for path in paths:
        with open(path, "rb") as fh:
            out[path] = fh.read()
    return out


def _steps(store):
    return sorted(info.step for info in store.list_checkpoints())


def _save_two(root):
    store = s3dlio.PyCheckpointStore(_uri(root))
    store.save(1, 1, "test", PAYLOAD_1)
    store.save(2, 1, "test", PAYLOAD_2)
    return store


def test_delete_older_checkpoint_keeps_latest():
    with tempfile.TemporaryDirectory() as tmp:
        store = _save_two(tmp)
        shards = _shard_paths(tmp, 1)
        assert shards
        assert all(os.path.isfile(p) for p in shards)

        assert store.delete_checkpoint(1) is True
        assert bytes(store.load_latest()) == PAYLOAD_2
        assert store.delete_checkpoint(1) is False
        assert _steps(store) == [2]
        assert not any(os.path.exists(p) for p in shards)
        assert all(os.path.isfile(p) for p in _shard_paths(tmp, 2))


def test_delete_latest_repoints_then_removes_pointer():
    with tempfile.TemporaryDirectory() as tmp:
        store = _save_two(tmp)
        latest_path = os.path.join(tmp, "latest.json")

        assert store.delete_checkpoint(2) is True
        assert bytes(store.load_latest()) == PAYLOAD_1
        with open(latest_path) as fh:
            latest = json.load(fh)
        assert latest["global_step"] == 1
        assert os.path.basename(latest["manifest_key"]).startswith("ckpt-1-")
        assert os.path.basename(latest["manifest_key"]) == os.path.basename(
            _manifest_path(tmp, 1)
        )

        assert store.delete_checkpoint(1) is True
        assert store.load_latest() is None
        assert not os.path.exists(latest_path)
        assert _steps(store) == []


def test_delete_refuses_name_vs_global_step_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        store = s3dlio.PyCheckpointStore(_uri(tmp))
        store.save(4, 1, "test", PAYLOAD_4)
        real = _manifest_path(tmp, 4)
        ts = os.path.basename(real)[len("ckpt-4-") :]
        fake = os.path.join(tmp, "manifests", f"ckpt-3-{ts}")
        shutil.copyfile(real, fake)  # named for step 3, body still global_step 4

        shard_paths = _shard_paths(tmp, 4)
        watched = [real, fake, os.path.join(tmp, "latest.json"), *shard_paths]
        before = _snapshot(watched)

        with pytest.raises(Exception) as excinfo:
            store.delete_checkpoint(3)
        assert "is named for step 3 but records global_step 4" in str(excinfo.value)

        assert _snapshot(watched) == before
        assert bytes(store.load_latest()) == PAYLOAD_4


def test_delete_unknown_step_is_false_and_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        store = _save_two(tmp)
        files = []
        for dirpath, _dirs, names in os.walk(tmp):
            files.extend(os.path.join(dirpath, n) for n in names)
        before = _snapshot(files)

        assert store.delete_checkpoint(99) is False

        after_files = []
        for dirpath, _dirs, names in os.walk(tmp):
            after_files.extend(os.path.join(dirpath, n) for n in names)
        assert sorted(after_files) == sorted(files)
        assert _snapshot(after_files) == before
        assert _steps(store) == [1, 2]
        assert bytes(store.load_latest()) == PAYLOAD_2
