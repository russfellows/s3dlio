"""Python ``s3dlio.list()`` literal-prefix contract: s3dlio issue #213.

A plain S3 URI is a LITERAL key prefix, never a regular expression:

* ``list("s3://b/data/run.1")`` lists ``data/run.1`` and ``data/run.1.old`` (and
  ``data/run.1/...``: every key that STARTS WITH ``data/run.1``), but not
  ``data/run_1`` or ``data/prerun.1`` -- literal prefix is not exact, and not a regex.
* ``.``, ``(``, ``)``, ``[``, ``+``, ``$`` and ``?`` in the URI are ordinary characters
  and never fail to parse.
* ``list("s3://b/foo")`` (non-recursive) lists ``foo``, ``foobar`` and the directory
  ``foo/`` -- not ``afoo`` and not unrelated top-level folders.
* The regex feature stays, through the explicit ``pattern=`` argument only; the
  pattern is matched against the key relative to the listing base.

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
        textwrap.dedent(CHILD), uri, "1" if recursive else "0", pattern, env=env_for(mock)
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT "):])


def uris(*keys):
    return sorted(f"s3://bkt/{k}" for k in keys)


def listed(mock, uri, **kw):
    out = call_list(mock, uri, **kw)
    assert "keys" in out, f"list({uri!r}, {kw}) raised: {out}"
    return sorted(out["keys"])


def sent_prefixes(mock):
    return [r["query"].get("prefix", "") for r in mock.lists()]


def test_run_dot_1_is_a_literal_prefix_not_a_regex_and_not_exact(mock):
    uri = "s3://bkt/data/run.1"
    assert listed(mock, uri, recursive=True) == uris(
        "data/run.1", "data/run.1.old", "data/run.1/child"
    )
    assert listed(mock, uri, recursive=False) == uris(
        "data/run.1", "data/run.1.old", "data/run.1/"
    )


def test_the_whole_key_part_is_sent_as_the_prefix(mock):
    listed(mock, "s3://bkt/data/run.1", recursive=False)
    assert sent_prefixes(mock) == ["data/run.1"], mock.render()
    assert [r["query"].get("delimiter") for r in mock.lists()] == ["/"]


@pytest.mark.parametrize(
    "prefix,want",
    [
        ("weird/p(", ["weird/p(", "weird/p(x"]),
        ("weird/q[", ["weird/q[", "weird/q[1]"]),
        ("weird/r+", ["weird/r+", "weird/r+1"]),
        ("weird/s$", ["weird/s$", "weird/s$1"]),
        ("weird/t?", ["weird/t?", "weird/t?1"]),
    ],
)
def test_regex_metacharacters_in_the_uri_are_literal_and_never_fail(mock, prefix, want):
    for recursive in (True, False):
        assert listed(mock, f"s3://bkt/{prefix}", recursive=recursive) == uris(*want)


def test_bare_name_non_recursive_has_no_unrelated_folders(mock):
    assert listed(mock, "s3://bkt/foo") == uris("foo", "foobar", "foo/")


def test_bare_name_recursive_lists_everything_that_starts_with_it(mock):
    assert listed(mock, "s3://bkt/foo", recursive=True) == uris(
        "foo", "foobar", "foo/a", "foo/b/c"
    )


def test_trailing_slash_directory_is_unchanged(mock):
    assert listed(mock, "s3://bkt/foo/") == uris("foo/a", "foo/b/")
    assert listed(mock, "s3://bkt/foo/", recursive=True) == uris("foo/a", "foo/b/c")


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


def test_explicit_pattern_composes_with_a_literal_prefix(mock):
    assert listed(mock, "s3://bkt/data/run.1", recursive=True, pattern="1$") == uris(
        "data/run.1"
    )


def test_a_pattern_does_not_change_what_the_server_is_asked_for(mock):
    listed(mock, "s3://bkt/dl/", recursive=True, pattern=r"\.npz$")
    assert sent_prefixes(mock) == ["dl/"], mock.render()


def test_invalid_pattern_is_an_error(mock):
    out = call_list(mock, "s3://bkt/dl/", recursive=True, pattern="(")
    assert "exc" in out, out
    assert "regex" in out["msg"].lower(), out
