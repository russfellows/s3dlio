"""Python ``s3dlio.list()`` / ``download()`` selection contract: s3dlio issue #213.

The key part ``K`` of an ``s3://`` URI is never a regular expression and never a bare
string prefix:

* ``list("s3://b/data/run.1")`` lists the object ``data/run.1`` and (recursive) the
  keys under ``data/run.1/`` -- NEVER ``data/run.1.old``, ``data/run_1`` or
  ``data/prerun.1``. Non-recursive it shows ``data/run.1`` and the directory
  ``data/run.1/``.
* ``.``, ``(``, ``)``, ``[``, ``+``, ``$`` and ``?`` in the URI are ordinary characters
  and never fail to parse.
* ``list("s3://b/foo")`` (non-recursive) lists ``foo`` and the directory ``foo/`` --
  not ``foobar``, ``afoo`` or unrelated top-level folders.
* ``*`` makes an explicit glob over the full key (``*`` stays inside one directory
  level, ``**`` crosses ``/``); the request prefix is the text before the first ``*``.
* The regex feature stays, through the explicit ``pattern=`` argument only; the
  pattern is applied AFTER the selection and matched against the key relative to the
  listing base.

Every test drives the real ``s3dlio`` extension module in a FRESH interpreter (the S3
client is process-global) against the stateful in-process mock S3 of
``test_python_delete_contract_212.py`` (prefix and delimiter honoured like S3), and also
asserts the prefix the library SENT to the server.

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_list_literal_213.py``.
"""

import json
import textwrap

import pytest
from test_oplog_lifecycle import run_scenario
from test_python_delete_contract_212 import MockS3, env_for

CATALOG = [
    "data/run.1",
    "data/run.1.old",
    "data/run.1/child",
    "data/run_1",
    "data/prerun.1",
    "data/run.2",
    "foo",
    "foobar",
    "foo/a",
    "foo/b/c",
    "afoo",
    "other/x",
    "zeta/top",
    "weird/p(",
    "weird/p(x",
    "weird/pp",
    "weird/q[",
    "weird/q[1]",
    "weird/qq",
    "weird/r+",
    "weird/r+1",
    "weird/rr",
    "weird/s$",
    "weird/s$1",
    "weird/ss",
    "weird/t?",
    "weird/t?1",
    "weird/tt",
    "dl/a.npz",
    "dl/b.txt",
    "dl/sub/c.npz",
    # decoys for the exact-name rule
    "file.txt",
    "file.txt.bak",
    "file.txt2",
    "dl2/x",
    "run.1.d/x",
    "data/top.npz",
    "data/sub/deep/z.npz",
]


@pytest.fixture
def mock():
    m = MockS3(CATALOG)
    yield m
    m.close()


CHILD = """
    import json
    uri, recursive, pattern = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
    try:
        keys = s3dlio.list(uri, recursive=recursive, pattern=pattern or None)
        out = {"keys": keys}
    except BaseException as e:
        out = {"exc": type(e).__name__, "msg": str(e)}
    print("RESULT " + json.dumps(out))
"""


def call_list(mock, uri, recursive=False, pattern=""):
    proc = run_scenario(
        textwrap.dedent(CHILD),
        uri,
        "1" if recursive else "0",
        pattern,
        env=env_for(mock),
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT ") :])


def uris(*keys):
    return sorted(f"s3://bkt/{k}" for k in keys)


def listed(mock, uri, **kw):
    out = call_list(mock, uri, **kw)
    assert "keys" in out, f"list({uri!r}, {kw}) raised: {out}"
    return sorted(out["keys"])


def sent_prefixes(mock):
    return [r["query"].get("prefix", "") for r in mock.lists()]


def test_run_dot_1_is_the_object_and_its_directory_not_a_regex_and_not_a_prefix(mock):
    uri = "s3://bkt/data/run.1"
    assert listed(mock, uri, recursive=True) == uris("data/run.1", "data/run.1/child")
    assert listed(mock, uri, recursive=False) == uris("data/run.1", "data/run.1/")


def test_an_exact_key_is_two_probes_that_cannot_return_siblings(mock):
    listed(mock, "s3://bkt/data/run.1", recursive=False)
    assert sent_prefixes(mock) == ["data/run.1", "data/run.1/"], mock.render()
    assert [r["query"].get("max-keys") for r in mock.lists()] == ["1", "1"]
    assert [r["query"].get("delimiter") for r in mock.lists()] == [None, None]


def test_file_txt_selects_only_file_txt(mock):
    uri = "s3://bkt/file.txt"
    assert listed(mock, uri) == uris("file.txt")
    assert listed(mock, uri, recursive=True) == uris("file.txt")


def test_a_directory_name_never_crosses_into_a_sibling_directory(mock):
    assert listed(mock, "s3://bkt/dl", recursive=True) == uris(
        "dl/a.npz", "dl/b.txt", "dl/sub/c.npz"
    )


@pytest.mark.parametrize(
    "prefix,want",
    [
        ("weird/p(", ["weird/p("]),
        ("weird/q[", ["weird/q["]),
        ("weird/r+", ["weird/r+"]),
        ("weird/s$", ["weird/s$"]),
        ("weird/t?", ["weird/t?"]),
    ],
)
def test_regex_metacharacters_in_the_uri_are_literal_and_never_fail(mock, prefix, want):
    for recursive in (True, False):
        assert listed(mock, f"s3://bkt/{prefix}", recursive=recursive) == uris(*want)


def test_bare_name_non_recursive_has_no_siblings_or_unrelated_folders(mock):
    assert listed(mock, "s3://bkt/foo") == uris("foo", "foo/")


def test_bare_name_recursive_lists_the_object_and_everything_under_it(mock):
    assert listed(mock, "s3://bkt/foo", recursive=True) == uris("foo", "foo/a", "foo/b/c")


def test_trailing_slash_directory_is_unchanged(mock):
    assert listed(mock, "s3://bkt/foo/") == uris("foo/a", "foo/b/")
    assert listed(mock, "s3://bkt/foo/", recursive=True) == uris("foo/a", "foo/b/c")


# ---- an explicit wildcard ---------------------------------------------------------------


def test_star_is_an_explicit_glob_at_the_same_directory_level(mock):
    want = uris("file.txt", "file.txt.bak", "file.txt2")
    assert listed(mock, "s3://bkt/file.txt*") == want
    assert listed(mock, "s3://bkt/file.txt*", recursive=True) == want
    assert sent_prefixes(mock) == ["file.txt", "file.txt"], mock.render()


def test_single_star_stays_in_one_directory_and_double_star_crosses_slash(mock):
    assert listed(mock, "s3://bkt/data/*.npz", recursive=True) == uris("data/top.npz")
    assert listed(mock, "s3://bkt/data/**.npz", recursive=True) == uris(
        "data/top.npz", "data/sub/deep/z.npz"
    )


def test_a_wildcard_key_ending_in_slash_is_refused(mock):
    out = call_list(mock, "s3://bkt/data/*/", recursive=True)
    assert "exc" in out and "not defined" in out["msg"], out
    assert not mock.lists(), mock.render()


# ---- download -------------------------------------------------------------------------

DOWNLOAD_CHILD = """
    import json, os
    uri, dest, recursive = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
    try:
        s3dlio.download(uri, dest, recursive=recursive)
        # issue #284: each object keeps its key path relative to the listing base
        out = {
            "files": sorted(
                os.path.relpath(os.path.join(root, name), dest).replace(os.sep, "/")
                for root, _dirs, names in os.walk(dest)
                for name in names
            )
        }
    except BaseException as e:
        out = {"exc": type(e).__name__, "msg": str(e)}
    print("RESULT " + json.dumps(out))
"""


def call_download(mock, uri, dest, recursive=True):
    proc = run_scenario(
        textwrap.dedent(DOWNLOAD_CHILD),
        uri,
        str(dest),
        "1" if recursive else "0",
        env=env_for(mock),
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT ") :])


def test_download_of_file_txt_takes_only_file_txt(mock, tmp_path):
    out = call_download(mock, "s3://bkt/file.txt", tmp_path / "d")
    assert out == {"files": ["file.txt"]}, out
    assert [r["path"] for r in mock.requests("GET") if "list-type" not in r["query"]] == [
        "/bkt/file.txt"
    ], mock.render()


def test_download_of_a_wildcard_takes_the_glob_matches(mock, tmp_path):
    out = call_download(mock, "s3://bkt/file.txt*", tmp_path / "d")
    assert out == {"files": ["file.txt", "file.txt.bak", "file.txt2"]}, out


def test_download_of_a_directory_name_does_not_cross_into_dl2(mock, tmp_path):
    out = call_download(mock, "s3://bkt/dl", tmp_path / "d")
    assert out == {"files": ["a.npz", "b.txt", "sub/c.npz"]}, out


# ---- get (an exact single object: untouched, never lists) -------------------------------

GET_CHILD = """
    import json
    uri = sys.argv[1]
    try:
        data = bytes(s3dlio.get(uri))
        out = {"len": len(data)}
    except BaseException as e:
        out = {"exc": type(e).__name__, "msg": str(e)}
    print("RESULT " + json.dumps(out))
"""


def test_get_of_file_txt_reads_that_one_key_and_never_lists(mock):
    proc = run_scenario(textwrap.dedent(GET_CHILD), "s3://bkt/file.txt", env=env_for(mock))
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (proc.stdout, proc.stderr)
    assert not mock.lists(), mock.render()
    assert {r["path"] for r in mock.requests("GET")} == {"/bkt/file.txt"}, mock.render()


# ---- the explicit pattern keeps working ------------------------------------------------


def test_explicit_pattern_filters_relative_to_the_listing_base(mock):
    assert listed(mock, "s3://bkt/dl/", recursive=True, pattern=r"\.npz$") == uris(
        "dl/a.npz", "dl/sub/c.npz"
    )
    # the pattern sees `sub/c.npz`, not the full URI
    assert listed(mock, "s3://bkt/dl/", recursive=True, pattern="^sub/") == uris(
        "dl/sub/c.npz"
    )
    # no trailing slash: the same base, because `dl/a.npz` lies under `dl/`
    assert listed(mock, "s3://bkt/dl", recursive=True, pattern="^a") == uris("dl/a.npz")


def test_explicit_pattern_composes_with_a_selection(mock):
    assert listed(mock, "s3://bkt/data/run.1", recursive=True, pattern="1$") == uris(
        "data/run.1"
    )
    # a pattern narrows a wildcard selection too: relative to the directory before the `*`
    assert listed(mock, "s3://bkt/file.txt*", pattern=r"\.bak$") == uris("file.txt.bak")


def test_a_pattern_does_not_change_what_the_server_is_asked_for(mock):
    listed(mock, "s3://bkt/dl/", recursive=True, pattern=r"\.npz$")
    assert sent_prefixes(mock) == ["dl/"], mock.render()


def test_invalid_pattern_is_an_error(mock):
    out = call_list(mock, "s3://bkt/dl/", recursive=True, pattern="(")
    assert "exc" in out, out
    assert "regex" in out["msg"].lower(), out
