"""s3torchconnector compatibility layer: ``prefix`` is an S3 KEY PREFIX (s3dlio issue #213).

``s3dlio.list`` selects an exact name, a directory or a ``*`` glob for an ``s3://`` URI,
but ``s3dlio.compat.s3torchconnector`` mirrors AWS's s3torchconnector, whose ``prefix``
is a key prefix: ``S3Client.list_objects(bucket, "train/img_")``,
``S3IterableDataset.from_prefix("s3://b/train/img_")`` and
``S3MapDataset.from_prefix("s3://b/train/img_")`` return every key that STARTS WITH
``train/img_`` -- ``train/img_1/x`` included, because it does start with it.

* A prefix without ``*`` is listed as ``<prefix>**`` (the S3 request prefix is the prefix
  itself, so the server does the narrowing).
* A prefix that itself contains ``*`` (a literal character in a key prefix) is read by
  listing the directory part of the text before the first ``*`` recursively and keeping
  the keys that start with the prefix.
* A directory prefix (``train/``) is listed as it always was.

Each case runs in a FRESH interpreter (the S3 client is process-global) against the
stateful in-process mock S3 of ``test_python_delete_contract_212.py``, and also asserts
the prefix the library SENT to the server. ``torch`` is replaced by a stub in the child:
the compat module only needs its ``Dataset`` base classes.

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_compat_s3torchconnector_213.py``.
"""

import json
import textwrap

import pytest
from test_oplog_lifecycle import run_scenario
from test_python_delete_contract_212 import MockS3, env_for

CATALOG = [
    "train/img_",
    "train/img_0001",
    "train/img_0002",
    "train/img_1/x",
    "train/img",
    "train/imgs/x",
    "train/other",
    "train/im*g",
    "train/im*g/x",
    "train/imxg",
    "train/sub/img_9",
    "val/img_0001",
    "top.txt",
]


@pytest.fixture
def mock():
    m = MockS3(CATALOG)
    yield m
    m.close()


CHILD = """
    import json, types
    torch = types.ModuleType("torch")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.IterableDataset = object
    torch.utils.data.Dataset = object
    sys.modules.update(
        {"torch": torch, "torch.utils": torch.utils, "torch.utils.data": torch.utils.data}
    )
    from s3dlio.compat.s3torchconnector import S3Client, S3IterableDataset, S3MapDataset

    what, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        if what == "list_objects":
            pages = list(S3Client().list_objects(a, b))
            keys = [o.key for page in pages for o in page.object_info]
        elif what == "iterable":
            keys = list(S3IterableDataset.from_prefix(a)._keys)
        elif what == "map":
            keys = list(S3MapDataset.from_prefix(a)._full_uris)
        else:
            raise ValueError(what)
        out = {"keys": keys}
    except BaseException as e:
        out = {"exc": type(e).__name__, "msg": str(e)}
    print("RESULT " + json.dumps(out))
"""


def call(mock, what, a, b=""):
    proc = run_scenario(textwrap.dedent(CHILD), what, a, b, env=env_for(mock))
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    out = json.loads(lines[-1][len("RESULT ") :])
    assert "keys" in out, f"{what}({a!r}, {b!r}) raised: {out}"
    return sorted(k.removeprefix("s3://bkt/") for k in out["keys"])


def sent(mock):
    return [(r["query"].get("prefix", ""), r["query"].get("delimiter")) for r in mock.lists()]


WANT_IMG = sorted(
    [
        "train/img_",
        "train/img_0001",
        "train/img_0002",
        "train/img_1/x",  # starts with "train/img_": deeper paths are included
    ]
)


def test_list_objects_bare_partial_prefix_returns_every_key_that_starts_with_it(mock):
    assert call(mock, "list_objects", "bkt", "train/img_") == WANT_IMG
    # one listing, whose S3 prefix is the prefix itself; not the directory listing
    assert sent(mock) == [("train/img_", None)], mock.render()


def test_list_objects_a_prefix_that_is_also_a_name_returns_siblings_too(mock):
    # upstream semantics: `train/img` is a prefix, so img_*, img_1/x, imgs/x all start with it
    assert call(mock, "list_objects", "bkt", "train/img") == sorted(
        [
            "train/img",
            "train/img_",
            "train/img_0001",
            "train/img_0002",
            "train/img_1/x",
            "train/imgs/x",
        ]
    )


def test_list_objects_directory_prefix_is_unchanged(mock):
    want = sorted(k for k in CATALOG if k.startswith("train/"))
    assert call(mock, "list_objects", "bkt", "train/") == want
    assert sent(mock) == [("train/", None)], mock.render()


def test_list_objects_empty_prefix_lists_the_bucket(mock):
    assert call(mock, "list_objects", "bkt", "") == sorted(CATALOG)


def test_list_objects_a_full_uri_is_a_directory(mock):
    want = sorted(k for k in CATALOG if k.startswith("train/"))
    assert call(mock, "list_objects", "bkt", "s3://bkt/train") == want


def test_list_objects_no_match_is_empty(mock):
    assert call(mock, "list_objects", "bkt", "nothing/here_") == []


def test_a_prefix_with_a_star_is_literal_and_listed_through_its_directory(mock):
    # `*` is an ordinary character of a key prefix: only keys that START WITH `train/im*`
    assert call(mock, "list_objects", "bkt", "train/im*") == ["train/im*g", "train/im*g/x"]
    # the directory part of the text before the first `*`, recursively; no delimiter
    assert sent(mock) == [("train/", None)], mock.render()


def test_a_prefix_with_a_star_and_no_directory_lists_the_bucket_and_filters(mock):
    assert call(mock, "list_objects", "bkt", "im*") == []
    assert sent(mock) == [("", None)], mock.render()


@pytest.mark.parametrize("what", ["iterable", "map"])
def test_datasets_from_a_bare_partial_prefix_keep_the_upstream_prefix_meaning(mock, what):
    uri = "s3://bkt/train/img_"
    got = call(mock, what, uri)
    assert got == WANT_IMG, got
    assert sent(mock) == [("train/img_", None)], mock.render()


@pytest.mark.parametrize("what", ["iterable", "map"])
def test_datasets_from_a_directory_prefix_are_unchanged(mock, what):
    want = sorted(k for k in CATALOG if k.startswith("train/"))
    assert call(mock, what, "s3://bkt/train/") == want
    assert sent(mock) == [("train/", None)], mock.render()


@pytest.mark.parametrize("what", ["iterable", "map"])
def test_datasets_from_a_prefix_with_a_star_filter_by_str_startswith(mock, what):
    assert call(mock, what, "s3://bkt/train/im*") == ["train/im*g", "train/im*g/x"]


def test_host_form_prefix_returns_the_same_keys(mock):
    host = mock.url.replace("http://", "s3://")
    assert call(mock, "map", f"{host}/bkt/train/img_") == WANT_IMG
    assert call(mock, "map", f"{host}/bkt/train/im*") == ["train/im*g", "train/im*g/x"]
