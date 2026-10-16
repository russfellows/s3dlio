"""``s3dlio.torch.ObjectStoreMapDataset`` fetches each item by the exact key it listed
(s3dlio issue #223).

The dataset lists the objects under its URI and later fetches item ``i`` with
``__getitem__``. Before the fix the fetch URI was rebuilt as
``<scheme>://<whole dataset URI path>/<listed key>``. The listing returns full
bucket-relative keys, so a dataset rooted below the bucket added the prefix twice:
``s3://bkt/dir/`` listed ``dir/x`` and requested ``dir/dir/x``.

Every scenario runs in a FRESH interpreter (the S3 client is process-global) against
the recording in-process mock S3 server of the #212 delete-contract tests, and asserts
the exact object keys the client asked the server for. ``torch`` is an optional extra
of s3dlio, so a minimal stand-in is installed in the child when it is absent (the
tests use ``return_type="bytes"``, which needs none of torch).

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_map_dataset_keys_223.py``.
"""

import json
import textwrap

import pytest

from test_oplog_lifecycle import run_scenario
from test_python_delete_contract_212 import MockS3, env_for


@pytest.fixture
def mock_factory():
    mocks = []

    def make(keys=(), host="127.0.0.1"):
        m = MockS3(keys, host)
        mocks.append(m)
        return m

    yield make
    for m in mocks:
        m.close()


FAKE_TORCH = """
    import sys, types
    try:
        import torch.utils.data
    except ImportError:
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")
        data.Dataset = type("Dataset", (), {})
        data.IterableDataset = type("IterableDataset", (), {})
        data.get_worker_info = lambda: None
        torch.utils = utils
        utils.data = data
        sys.modules.update({"torch": torch, "torch.utils": utils, "torch.utils.data": data})
"""

# Build the dataset, record what it listed, fetch every item, record what came back.
CHILD = (
    FAKE_TORCH
    + """
    import json
    from s3dlio.torch import ObjectStoreMapDataset
    uri = sys.argv[1]
    out = {"keys": None, "items": [], "exc": None}
    try:
        ds = ObjectStoreMapDataset(uri, return_type="bytes")
        out["keys"] = list(ds._keys)
        for i in range(len(ds)):
            out["items"].append(ds[i].hex())
    except BaseException as e:
        out["exc"] = type(e).__name__ + ": " + str(e)
    print("RESULT " + json.dumps(out))
"""
)


def run_dataset(uri, env=None, cwd=None):
    proc = run_scenario(textwrap.dedent(CHILD), uri, env=env, cwd=cwd)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT ") :])


def object_keys_requested(mock):
    """The key of every non-listing request, in order (path-style: /bkt/<key>)."""
    return [
        r["path"].lstrip("/").partition("/")[2]
        for r in mock.requests()
        if "list-type" not in r["query"] and r["method"] in ("GET", "HEAD")
    ]


S3_CASES = [
    # (dataset URI, bucket contents, keys the dataset must list and fetch)
    ("s3://bkt/", ["x", "y/z"], ["x", "y/z"]),
    ("s3://bkt/dir/", ["dir/x", "dir/sub/y", "other/q"], ["dir/x", "dir/sub/y"]),
    # No trailing slash selects the directory `dir` (and an object named exactly `dir`).
    ("s3://bkt/dir", ["dir/x", "dir/sub/y", "dir.bak/q"], ["dir/x", "dir/sub/y"]),
    ("s3://bkt/a/b/c/", ["a/b/c/x", "a/b/x"], ["a/b/c/x"]),
    # A doubled trailing slash is an exact prefix (#198), not a normalised one.
    ("s3://bkt/dir//", ["dir//x", "dir/x"], ["dir//x"]),
    # A leading "/" in the key: bucket `bkt`, prefix "/".
    ("s3://bkt//", ["/a", "/b/c", "z"], ["/a", "/b/c"]),
]


@pytest.mark.parametrize("uri,contents,expected", S3_CASES)
def test_s3_items_are_fetched_by_the_exact_listed_key(
    mock_factory, uri, contents, expected
):
    mock = mock_factory(contents)
    out = run_dataset(uri, env=env_for(mock))
    detail = f"\n  outcome: {out}\n  requests:\n    {mock.render()}"
    assert out["exc"] is None, f"the dataset failed{detail}"
    assert sorted(out["keys"]) == sorted(expected), f"listed keys{detail}"
    # Every item is fetched, by exactly the key that was listed -- no `dir/dir/x`.
    assert sorted(set(object_keys_requested(mock))) == sorted(expected), (
        f"object requests must name exactly the listed keys{detail}"
    )
    assert len(out["items"]) == len(expected), detail


# ----------------------------------------------------------------------------
# file:// and direct:// roots: real files, compared by content
# ----------------------------------------------------------------------------

TREE = {
    "a.bin": b"alpha",
    "sub/b.bin": b"bravo",
    "sub/deeper/c.bin": b"charlie",
}


def make_tree(root):
    for rel, data in TREE.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


@pytest.mark.parametrize("scheme", ["file", "direct"])
@pytest.mark.parametrize("trailing", ["", "/"])
def test_local_directory_items_are_read_from_their_own_path(tmp_path, scheme, trailing):
    root = tmp_path / "ds"
    make_tree(root)
    out = run_dataset(f"{scheme}://{root}{trailing}")
    assert out["exc"] is None, out
    # Nested files keep their directory under the root, so the key names the file.
    assert sorted(out["keys"]) == sorted(TREE), out
    got = {k: bytes.fromhex(v) for k, v in zip(out["keys"], out["items"])}
    assert got == TREE, out


@pytest.mark.parametrize("scheme", ["file", "direct"])
def test_local_single_file_root_reads_that_file(tmp_path, scheme):
    path = tmp_path / "only.bin"
    path.write_bytes(b"single")
    out = run_dataset(f"{scheme}://{path}")
    assert out["exc"] is None, out
    assert [bytes.fromhex(v) for v in out["items"]] == [b"single"], out


# ----------------------------------------------------------------------------
# The server form `s3://host:port/bucket/key`: the dataset parsed its own root, so the
# fetch URI keeps the right bucket (the first path segment is the server's port, not
# the bucket, here)
# ----------------------------------------------------------------------------

SERVER_CASES = [
    # (dataset path after the server, bucket contents, keys listed and fetched)
    ("bkt/", ["x", "y/z"], ["x", "y/z"]),
    ("bkt/dir/", ["dir/x", "dir/sub/y", "other/q"], ["dir/x", "dir/sub/y"]),
    ("bkt/dir//", ["dir//x", "dir/x"], ["dir//x"]),
    ("bkt//", ["/a", "/b/c", "z"], ["/a", "/b/c"]),
]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
@pytest.mark.parametrize("path,contents,expected", SERVER_CASES)
def test_server_form_items_are_fetched_from_the_right_bucket(
    mock_factory, host, path, contents, expected
):
    try:
        mock = mock_factory(contents, host)
    except OSError as e:
        pytest.skip(f"cannot listen on {host}: {e}")
    uri = f"s3://{mock.server_name}/{path}"
    out = run_dataset(uri, env=env_for(mock))
    detail = f"\n  uri: {uri}\n  outcome: {out}\n  requests:\n    {mock.render()}"
    assert out["exc"] is None, f"the dataset failed{detail}"
    assert sorted(out["keys"]) == sorted(expected), f"listed keys{detail}"
    assert sorted(set(object_keys_requested(mock))) == sorted(expected), (
        f"object requests must name exactly the listed keys{detail}"
    )
    assert len(out["items"]) == len(expected), detail
