"""Python ``s3dlio.delete()`` delete contract: s3dlio issue #212.

The contract (the same "contract A" as ``s3-cli delete``, issues #210 / #167):

* ``delete(uri)`` sends ONE delete request for exactly the key named by ``uri``:
  no HEAD first, no listing, nothing broader. An already-missing object is
  success (S3's DELETE succeeds for a missing key); a real failure (e.g. 403 on
  the DELETE) raises ``RuntimeError`` naming the key.
* ``delete(uri, recursive=True)`` deletes the REAL objects the URI selects (issue
  #213): the object named exactly ``K`` plus everything under ``K/`` -- never
  ``K.bak`` -- listed without a delimiter, objects only; a key ending in ``/`` is
  the directory; ``*`` / ``**`` in a recursive delete's URI are an explicit glob.
* Without ``recursive``, ``/``, ``*`` and ``?`` are ordinary key characters, never
  an instruction to list or glob.

Every test drives the real ``s3dlio`` extension module in a FRESH interpreter
(the S3 client is process-global) against a recording in-process mock S3 server
(a Python ``http.server`` in this process). The mock keeps real state -- it
deletes what DeleteObjects / DELETE name -- and logs every request, so each test
asserts both what survived and exactly which requests were sent.

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_delete_contract_212.py``.
Against a wheel without the fix (<= 0.9.122) the data-loss tests fail: that is
the point of them.
"""

import json
import re
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
from xml.sax.saxutils import escape, unescape

import pytest

from test_oplog_lifecycle import run_scenario


# ----------------------------------------------------------------------------
# Recording mock S3
# ----------------------------------------------------------------------------


def _delete_objects_keys(body: bytes) -> list:
    """The <Key> values of a DeleteObjects request body, in order.

    A plain linear scan: no regular expression (CodeQL py/polynomial-redos) and
    no XML parser (CodeQL py/xml-bomb). The mock only ever sees requests from
    this test's own client; keys are XML-escaped by the SDK, so unescape them.
    """
    text = body.decode("utf-8", "replace")
    keys = []
    pos = 0
    while True:
        start = text.find("<Key>", pos)
        if start < 0:
            return keys
        start += len("<Key>")
        end = text.find("</Key>", start)
        if end < 0:
            return keys
        keys.append(unescape(text[start:end]))
        pos = end + len("</Key>")


class MockS3:
    """A path-style S3 server for the single bucket ``bkt``, with real state."""

    def __init__(self, keys=()):
        self.keys = set(keys)
        self.log = []  # dicts: method, path, query, body_keys
        self.delete_status = None  # answer every delete request (DELETE, POST ?delete) with this status
        self.list_status = None  # answer every delimiter-less listing with this status
        self.delete_error_keys = set()  # DeleteObjects reports these per key, deleting nothing
        self.page_size = None  # page delimiter-less listings
        self.list_fail_after_first_page = False  # 403 every delimiter-less listing page after the first
        self.lock = threading.Lock()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # one request per connection

            def log_message(self, *args):
                pass

            def _body(self):
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    out = b""
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            self.rfile.readline()
                            return out
                        out += self.rfile.read(size)
                        self.rfile.readline()
                length = int(self.headers.get("Content-Length", "0") or 0)
                return self.rfile.read(length) if length else b""

            def _handle(self):
                body = self._body()
                split = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(split.query, keep_blank_values=True).items()}
                path = unquote(split.path)
                key = path.lstrip("/").partition("/")[2]
                body_keys = []
                if self.command == "POST" and body:
                    body_keys = _delete_objects_keys(body)
                status, resp = mock.respond(self.command, key, query, body_keys)
                with mock.lock:
                    mock.log.append(
                        {"method": self.command, "path": path, "query": query, "body_keys": body_keys}
                    )
                data = resp.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            do_GET = do_HEAD = do_POST = do_DELETE = do_PUT = _handle

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def respond(self, method, key, query, body_keys):
        with self.lock:
            if method == "GET" and "list-type" in query:
                return self._list(query)
            if self.delete_status is not None and (
                method == "DELETE" or (method == "POST" and "delete" in query)
            ):
                return (
                    self.delete_status,
                    '<?xml version="1.0"?><Error><Code>AccessDenied</Code>'
                    "<Message>mock refuses to delete</Message></Error>",
                )
            if method == "POST" and "delete" in query:
                parts = []
                for k in body_keys:
                    if k in self.delete_error_keys:
                        parts.append(
                            f"<Error><Key>{escape(k)}</Key><Code>AccessDenied</Code>"
                            "<Message>mock refuses this key</Message></Error>"
                        )
                    else:
                        self.keys.discard(k)
                        parts.append(f"<Deleted><Key>{escape(k)}</Key></Deleted>")
                return 200, '<?xml version="1.0"?><DeleteResult>' + "".join(parts) + "</DeleteResult>"
            if method == "DELETE":
                if key in self.delete_error_keys:
                    return (
                        403,
                        '<?xml version="1.0"?><Error><Code>AccessDenied</Code>'
                        "<Message>mock refuses this key</Message></Error>",
                    )
                self.keys.discard(key)
                return 204, ""
            if method == "PUT":
                self.keys.add(key)
                return 200, ""
            if method == "HEAD":
                return (200, "") if key in self.keys else (404, "")
            return 200, ""

    def _list(self, query):
        delimiter = query.get("delimiter")
        if (
            self.list_fail_after_first_page
            and delimiter is None
            and "continuation-token" in query
        ):
            return (
                403,
                '<?xml version="1.0"?><Error><Code>AccessDenied</Code>'
                "<Message>mock refuses to list</Message></Error>",
            )
        if self.list_status is not None and delimiter is None:
            return (
                self.list_status,
                '<?xml version="1.0"?><Error><Code>AccessDenied</Code>'
                "<Message>mock refuses to list</Message></Error>",
            )
        prefix = query.get("prefix", "")
        contents, common = [], set()
        for k in sorted(self.keys):
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            if delimiter is not None and delimiter in rest:
                common.add(prefix + rest[: rest.index(delimiter) + 1])
            else:
                contents.append(k)
        after = query.get("continuation-token")
        if after is not None:
            contents = [k for k in contents if k > after]
        page = self.page_size if (self.page_size and delimiter is None) else len(contents) or 1
        truncated = len(contents) > page
        contents = contents[:page]
        token = f"<NextContinuationToken>{escape(contents[-1])}</NextContinuationToken>" if truncated else ""
        entries = "".join(
            f"<Contents><Key>{escape(k)}</Key><LastModified>2026-01-01T00:00:00.000Z</LastModified>"
            f'<ETag>"x"</ETag><Size>{0 if k.endswith("/") else 1}</Size>'
            "<StorageClass>STANDARD</StorageClass></Contents>"
            for k in contents
        )
        cps = "".join(f"<CommonPrefixes><Prefix>{escape(c)}</Prefix></CommonPrefixes>" for c in sorted(common))
        return (
            200,
            '<?xml version="1.0"?><ListBucketResult><Name>bkt</Name>'
            f"<Prefix>{escape(prefix)}</Prefix><KeyCount>1</KeyCount><MaxKeys>1000</MaxKeys>"
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>{token}{entries}{cps}"
            "</ListBucketResult>",
        )

    # -- what the tests ask -------------------------------------------------
    def requests(self, method=None):
        with self.lock:
            return [r for r in self.log if method is None or r["method"] == method]

    def lists(self):
        return [r for r in self.requests("GET") if "list-type" in r["query"]]

    def sent_delete_keys(self):
        """Every key named in any delete request, one entry per occurrence."""
        out = []
        for r in self.requests():
            if r["method"] == "POST" and "delete" in r["query"]:
                out.extend(r["body_keys"])
            elif r["method"] == "DELETE":
                out.append(r["path"].lstrip("/").partition("/")[2])
        return out

    def render(self):
        return "\n    ".join(
            f"{r['method']} {r['path']}?{r['query']} body-keys={r['body_keys']}"
            for r in self.requests()
        )


@pytest.fixture
def mock_factory():
    mocks = []

    def make(keys=()):
        m = MockS3(keys)
        mocks.append(m)
        return m

    yield make
    for m in mocks:
        m.close()


# ----------------------------------------------------------------------------
# Driving s3dlio.delete in a fresh interpreter
# ----------------------------------------------------------------------------

CHILD = """
    import json
    uri, recursive = sys.argv[1], sys.argv[2] == "1"
    try:
        value = s3dlio.delete(uri, recursive=recursive)
        out = {"returned": repr(value)}
    except BaseException as e:
        out = {"exc": type(e).__name__, "mro": [c.__name__ for c in type(e).__mro__], "msg": str(e)}
    print("RESULT " + json.dumps(out))
"""


def env_for(mock):
    return {
        "AWS_ENDPOINT_URL": mock.url,
        "AWS_ENDPOINT_URL_S3": mock.url,
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def call_delete(mock, uri, recursive=False):
    """Run ``s3dlio.delete(uri, recursive)`` against `mock`; return the outcome dict."""
    proc = run_scenario(
        textwrap.dedent(CHILD), uri, "1" if recursive else "0", env=env_for(mock)
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (
        f"child failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(lines[-1][len("RESULT "):])


def ctx(mock, out):
    return f"\n  outcome: {out}\n  requests:\n    {mock.render()}\n  keys now: {sorted(mock.keys)}"


def assert_raises(out, exc, mock, text=None):
    assert out.get("exc") == exc, f"expected {exc}{ctx(mock, out)}"
    if text is not None:
        assert text in out["msg"], f"expected {text!r} in the message{ctx(mock, out)}"


def assert_returned(out, mock):
    assert "returned" in out, f"expected a normal return{ctx(mock, out)}"
    assert out["returned"] == "None", f"delete returns None{ctx(mock, out)}"


def assert_no_list_no_delete(mock, out):
    assert not mock.lists(), f"a listing was issued{ctx(mock, out)}"
    assert not mock.sent_delete_keys(), f"a delete was issued{ctx(mock, out)}"


def assert_one_delete_no_lookup(mock, out, key):
    """The exact delete is one delete request for `key`: no HEAD, no listing."""
    assert not mock.requests("HEAD"), f"a HEAD was issued{ctx(mock, out)}"
    assert not mock.lists(), f"a listing was issued{ctx(mock, out)}"
    assert mock.sent_delete_keys() == [key], f"expected one delete of {key!r}{ctx(mock, out)}"


# ----------------------------------------------------------------------------
# 1. delete(uri): ONLY the exact object
# ----------------------------------------------------------------------------


def test_trailing_slash_deletes_only_the_marker_object_not_what_is_under_it(mock_factory):
    """The data-loss case: ``foo/`` names the marker object, never "everything under foo/"."""
    m = mock_factory(["foo/", "foo/x", "foo/sub/y", "foobar", "other/", "other/x"])
    out = call_delete(m, "s3://bkt/foo/")
    assert_returned(out, m)
    assert m.keys == {"foo/x", "foo/sub/y", "foobar", "other/", "other/x"}, ctx(m, out)
    assert_one_delete_no_lookup(m, out, "foo/")


def test_trailing_slash_without_marker_deletes_only_that_key_and_nested_objects_survive(mock_factory):
    """#167: ``foo/x`` exists but the object ``foo/`` does not. The delete request for
    ``foo/`` succeeds (S3 semantics); ``foo/x`` and everything else is untouched."""
    seed = ["foo/x", "foo/sub/y", "other/x"]
    m = mock_factory(seed)
    out = call_delete(m, "s3://bkt/foo/")
    assert_returned(out, m)
    assert m.keys == set(seed), ctx(m, out)
    assert_one_delete_no_lookup(m, out, "foo/")


def test_directory_with_only_subdirectories_never_deletes_common_prefixes(mock_factory):
    """A synthetic CommonPrefix (``d/sub/``) is never sent as a key, nor is what is under it."""
    seed = ["d/sub/y", "d/sub/z"]
    m = mock_factory(seed)
    out = call_delete(m, "s3://bkt/d/")
    assert_returned(out, m)
    assert m.keys == set(seed), ctx(m, out)
    assert_one_delete_no_lookup(m, out, "d/")


def test_missing_object_is_success_and_touches_nothing_else(mock_factory):
    """#210: the key does not exist, but other/ and other/x do. One delete for
    ``testprefix`` only; no listing of the bucket root, no deletion of ``other/``."""
    seed = ["other/", "other/x", "testprefix/a"]
    m = mock_factory(seed)
    out = call_delete(m, "s3://bkt/testprefix")
    assert_returned(out, m)
    assert m.keys == set(seed), ctx(m, out)
    assert_one_delete_no_lookup(m, out, "testprefix")


def test_exact_object_is_deleted_and_neighbours_survive(mock_factory):
    m = mock_factory(["foo", "foo/", "foo/x", "foobar"])
    out = call_delete(m, "s3://bkt/foo")
    assert_returned(out, m)
    assert m.keys == {"foo/", "foo/x", "foobar"}, ctx(m, out)
    assert_one_delete_no_lookup(m, out, "foo")


def test_bucket_only_uri_names_no_object_and_sends_nothing(mock_factory):
    seed = ["a", "b/c"]
    m = mock_factory(seed)
    out = call_delete(m, "s3://bkt/")
    assert_raises(out, "RuntimeError", m, "names a bucket")
    assert m.keys == set(seed), ctx(m, out)
    assert not m.requests(), f"a request was sent{ctx(m, out)}"


def test_delete_403_raises_runtime_error_naming_the_key_and_is_never_read_as_absent(mock_factory):
    """A real failure of the delete request is an error; nothing else is tried."""
    seed = ["foo", "foo/x"]
    m = mock_factory(seed)
    m.delete_status = 403
    out = call_delete(m, "s3://bkt/foo")
    assert_raises(out, "RuntimeError", m, "foo")
    assert "failed to delete" in out["msg"], ctx(m, out)
    assert m.keys == set(seed), ctx(m, out)
    assert not m.requests("HEAD"), f"a HEAD was issued{ctx(m, out)}"
    assert not m.lists(), f"a listing was issued{ctx(m, out)}"


@pytest.mark.parametrize(
    "key,seed",
    [
        ("a*", ["a*", "ab", "axb", "a/b"]),
        ("a?b", ["a?b", "axb", "ayb"]),
        ("p*/", ["p*/", "p*/x", "pq/x"]),
    ],
)
def test_star_and_question_mark_are_literal_key_characters(mock_factory, key, seed):
    m = mock_factory(seed)
    out = call_delete(m, f"s3://bkt/{key}")
    assert_returned(out, m)
    assert m.keys == set(seed) - {key}, ctx(m, out)
    assert_one_delete_no_lookup(m, out, key)


@pytest.mark.parametrize("key", ["a*", "a?b"])
def test_star_and_question_mark_missing_object_is_one_literal_delete_not_a_glob(mock_factory, key):
    seed = ["ab", "axb", "a/b"]
    m = mock_factory(seed)
    out = call_delete(m, f"s3://bkt/{key}")
    assert_returned(out, m)
    assert m.keys == set(seed), ctx(m, out)
    assert_one_delete_no_lookup(m, out, key)


# ----------------------------------------------------------------------------
# 2. delete(uri, recursive=True): the selected objects only (issue #213 rule)
# ----------------------------------------------------------------------------


def test_recursive_deletes_the_real_objects_under_the_literal_prefix_only(mock_factory):
    m = mock_factory(
        ["foo/", "foo/x", "foo/sub/y", "foo/sub/deep/z", "foo", "foobar", "other/", "other/x"]
    )
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_returned(out, m)
    assert m.keys == {"foo", "foobar", "other/", "other/x"}, ctx(m, out)
    assert sorted(m.sent_delete_keys()) == ["foo/", "foo/sub/deep/z", "foo/sub/y", "foo/x"], ctx(m, out)
    # one literal listing: prefix as given, no delimiter, so no CommonPrefixes exist
    assert [(r["query"].get("prefix"), r["query"].get("delimiter")) for r in m.lists()] == [
        ("foo/", None)
    ], ctx(m, out)
    # no existence check is made for a recursive delete, and no single-object DELETE
    assert not m.requests("HEAD"), ctx(m, out)


def test_recursive_without_trailing_slash_is_the_exact_object_and_its_directory(mock_factory):
    """``foo`` selects the object ``foo`` and ``foo/...``; NEVER ``foobar``. Nothing is
    appended to the key that is sent for the existence probe."""
    m = mock_factory(["foo", "foo/", "foo/x", "foobar", "fo", "other/x"])
    out = call_delete(m, "s3://bkt/foo", recursive=True)
    assert_returned(out, m)
    assert m.keys == {"foobar", "fo", "other/x"}, ctx(m, out)
    assert sorted(m.sent_delete_keys()) == ["foo", "foo/", "foo/x"], ctx(m, out)
    # two requests, none with a delimiter: a max-keys=1 probe of `foo`, then the pages of `foo/`
    assert [
        (r["query"].get("prefix"), r["query"].get("delimiter"), r["query"].get("max-keys"))
        for r in m.lists()
    ] == [("foo", None, "1"), ("foo/", None, None)], ctx(m, out)


def test_recursive_delete_of_file_txt_leaves_file_txt_bak(mock_factory):
    """THE safety case: ``delete(s3://b/file.txt, recursive=True)`` removes file.txt and
    nothing else."""
    seed = ["file.txt", "file.txt.bak", "file.txt2", "file.txt.d/x", "dl/a", "dl2/x"]
    m = mock_factory(seed)
    out = call_delete(m, "s3://bkt/file.txt", recursive=True)
    assert_returned(out, m)
    assert m.sent_delete_keys() == ["file.txt"], ctx(m, out)
    assert m.keys == {"file.txt.bak", "file.txt2", "file.txt.d/x", "dl/a", "dl2/x"}, ctx(m, out)


def test_recursive_delete_of_a_directory_name_does_not_cross_into_dl2(mock_factory):
    m = mock_factory(["dl/a", "dl/sub/b", "dl2/x", "dlx"])
    out = call_delete(m, "s3://bkt/dl", recursive=True)
    assert_returned(out, m)
    assert sorted(m.sent_delete_keys()) == ["dl/a", "dl/sub/b"], ctx(m, out)
    assert m.keys == {"dl2/x", "dlx"}, ctx(m, out)


@pytest.mark.parametrize(
    "key,seed,survivors,sent",
    [
        # `*` stays inside one directory level
        ("file.txt*", ["file.txt", "file.txt.bak", "file.txt2", "file.txt.d/x", "file.tx"],
         {"file.txt.d/x", "file.tx"}, ["file.txt", "file.txt.bak", "file.txt2"]),
        ("d/*", ["d/*", "d/*/x", "d/y", "d2/z"], {"d/*/x", "d2/z"}, ["d/*", "d/y"]),
        # `**` crosses '/'
        ("d/**", ["d/*", "d/*/x", "d/y", "d2/z"], {"d2/z"}, ["d/*", "d/*/x", "d/y"]),
        ("d/**.npz", ["d/a.npz", "d/s/b.npz", "d/c.txt"], {"d/c.txt"}, ["d/a.npz", "d/s/b.npz"]),
    ],
)
def test_recursive_wildcard_is_an_explicit_glob_over_the_full_key(
    mock_factory, key, seed, survivors, sent
):
    m = mock_factory(seed)
    out = call_delete(m, f"s3://bkt/{key}", recursive=True)
    assert_returned(out, m)
    assert m.keys == survivors, ctx(m, out)
    assert sorted(m.sent_delete_keys()) == sorted(sent), ctx(m, out)


def test_recursive_wildcard_request_prefix_is_the_text_before_the_first_star(mock_factory):
    m = mock_factory(["a1", "ab", "b"])
    out = call_delete(m, "s3://bkt/a*", recursive=True)
    assert_returned(out, m)
    assert [r["query"].get("prefix") for r in m.lists()] == ["a"], ctx(m, out)


@pytest.mark.parametrize(
    "prefix,seed,survivors",
    [
        ("a?", ["a?", "a?b", "ab", "axb"], {"a?b", "ab", "axb"}),
        ("a(b", ["a(b", "a(bc", "a(b/x", "ab"], {"a(bc", "ab"}),
    ],
)
def test_recursive_question_mark_and_regex_characters_are_literal(
    mock_factory, prefix, seed, survivors
):
    m = mock_factory(seed)
    out = call_delete(m, f"s3://bkt/{prefix}", recursive=True)
    assert_returned(out, m)
    assert m.keys == survivors, ctx(m, out)


def test_recursive_with_nothing_under_the_prefix_is_a_no_op(mock_factory):
    seed = ["other/", "other/x"]
    m = mock_factory(seed)
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_returned(out, m)
    assert m.keys == set(seed), ctx(m, out)
    assert not m.sent_delete_keys(), ctx(m, out)


def test_recursive_follows_listing_pages(mock_factory):
    seed = [f"big/k{i:02d}" for i in range(7)] + ["bigger", "other"]
    m = mock_factory(seed)
    m.page_size = 2
    out = call_delete(m, "s3://bkt/big/", recursive=True)
    assert_returned(out, m)
    assert m.keys == {"bigger", "other"}, ctx(m, out)
    assert len(m.lists()) >= 4, ctx(m, out)


def test_recursive_listing_failure_raises_and_deletes_nothing(mock_factory):
    seed = ["foo/x", "foo/y"]
    m = mock_factory(seed)
    m.list_status = 403
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_raises(out, "RuntimeError", m, "nothing was deleted")
    assert m.keys == set(seed), ctx(m, out)
    assert not m.sent_delete_keys(), ctx(m, out)


def test_recursive_per_key_error_in_a_successful_response_raises_and_names_the_key(mock_factory):
    m = mock_factory(["foo/a", "foo/b", "other"])
    m.delete_error_keys = {"foo/b"}
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_raises(out, "RuntimeError", m, "foo/b")
    assert "other" in m.keys and "foo/b" in m.keys, ctx(m, out)


def delete_requests(mock):
    """Every request that deletes: single DELETEs and DeleteObjects POSTs."""
    return [
        r
        for r in mock.requests()
        if r["method"] == "DELETE" or (r["method"] == "POST" and "delete" in r["query"])
    ]


def test_recursive_sends_one_list_then_one_single_key_delete_per_selected_object(mock_factory):
    """Python keeps individual deletes (not batches): one recursive LIST, then one
    delete request per selected object, each naming exactly ONE key. Keys outside the
    literal prefix (foobar, other/x) are never named."""
    m = mock_factory(["foo/", "foo/a", "foo/b/c", "foobar", "other/x"])
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_returned(out, m)
    assert m.keys == {"foobar", "other/x"}, ctx(m, out)
    reqs = m.requests()
    assert reqs[0]["method"] == "GET" and "list-type" in reqs[0]["query"], ctx(m, out)
    assert len(m.lists()) == 1 and m.lists()[0]["query"].get("delimiter") is None, ctx(m, out)
    deletes = delete_requests(m)
    assert len(deletes) == 3, ctx(m, out)
    assert all(r["method"] == "DELETE" or len(r["body_keys"]) == 1 for r in deletes), ctx(m, out)
    assert sorted(m.sent_delete_keys()) == ["foo/", "foo/a", "foo/b/c"], ctx(m, out)
    assert not m.requests("HEAD"), ctx(m, out)


RECURSIVE_OPLOG_CHILD = """
    oplog = sys.argv[1]
    s3dlio.init_op_log(oplog)
    s3dlio.delete("s3://bkt/foo/", recursive=True)
    s3dlio.finalize_op_log()
"""


def test_recursive_records_one_list_row_and_one_delete_row_per_object_never_delete_batch(
    mock_factory, tmp_path
):
    from collections import Counter

    from test_oplog_lifecycle import read_rows

    m = mock_factory(["foo/", "foo/a", "foo/b/c", "foobar"])
    oplog = tmp_path / "ops.tsv.zst"
    proc = run_scenario(textwrap.dedent(RECURSIVE_OPLOG_CHILD), str(oplog), env=env_for(m))
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    rows = read_rows(oplog)
    assert Counter(r["op"] for r in rows) == {"LIST": 1, "DELETE": 3}, rows
    # the op-log's file column is bucket/key
    assert sorted(r["file"] for r in rows if r["op"] == "DELETE") == [
        "bkt/foo/",
        "bkt/foo/a",
        "bkt/foo/b/c",
    ], rows
    assert all(r["error"] == "" for r in rows), rows
    assert m.keys == {"foobar"}, sorted(m.keys)


def test_recursive_stops_after_a_failed_delete_and_reports_accurate_counts(mock_factory):
    """The first object deleted is refused. No request STARTS after the failure is
    seen (requests already in flight, and ones that started before the refusal came
    back, finish), and the counts in the message match what the server saw."""
    seed = [f"foo/k{i:03d}" for i in range(150)] + ["foobar"]
    m = mock_factory(seed)
    m.delete_error_keys = {"foo/k000"}
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_raises(out, "RuntimeError", m, "foo/k000")
    sent = m.sent_delete_keys()
    assert 1 <= len(sent) < 150, ctx(m, out)
    msg = out["msg"]
    match = re.search(r"Sent delete requests for (\d+) of the (\d+) listed", msg)
    assert match, ctx(m, out)
    requested, listed = int(match.group(1)), int(match.group(2))
    assert requested == len(sent), ctx(m, out)
    not_attempted = re.search(r"(\d+) selected objects were NOT attempted", msg)
    skipped = int(not_attempted.group(1)) if not_attempted else 0
    assert requested + skipped == listed, ctx(m, out)
    assert "1 delete request(s) reported errors" in msg, ctx(m, out)
    # the failed key, foobar and everything never sent are still there
    assert "foo/k000" in m.keys and "foobar" in m.keys, ctx(m, out)
    assert len(m.keys) >= len(seed) - (len(sent) - 1), ctx(m, out)


def test_recursive_listing_error_midway_raises_and_sends_nothing_after_it(mock_factory):
    """Page 1 (foo/a, foo/b) lists fine, page 2 is refused. Each object is its own
    request, so a and b were already sent; c, d, e were never listed and are untouched."""
    m = mock_factory(["foo/a", "foo/b", "foo/c", "foo/d", "foo/e"])
    m.page_size = 2
    m.list_fail_after_first_page = True
    out = call_delete(m, "s3://bkt/foo/", recursive=True)
    assert_raises(out, "RuntimeError", m, "mock refuses to list")
    assert "listing failed" in out["msg"], ctx(m, out)
    assert m.keys >= {"foo/c", "foo/d", "foo/e"}, ctx(m, out)
    assert set(m.sent_delete_keys()) <= {"foo/a", "foo/b"}, ctx(m, out)


# ----------------------------------------------------------------------------
# 3. Other backends follow the trait defaults (file://)
# ----------------------------------------------------------------------------


def test_file_backend_exact_delete_is_exact_and_recursive_needs_a_trailing_slash(tmp_path, mock_factory):
    m = mock_factory()  # unused; keeps the helper signature
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "f1").write_bytes(b"1")
    (d / "f2").write_bytes(b"2")
    (d / "sub" / "g").write_bytes(b"3")

    out = call_delete(m, f"file://{d}/f1")
    assert_returned(out, m)
    assert not (d / "f1").exists() and (d / "f2").exists()

    # an already-missing file is success; nothing else is touched
    out = call_delete(m, f"file://{d}/missing")
    assert_returned(out, m)
    assert (d / "f2").exists() and (d / "sub" / "g").exists()

    # -r with no trailing '/': a clear error, nothing deleted (never read as a pattern)
    out = call_delete(m, f"file://{d}", recursive=True)
    assert_raises(out, "RuntimeError", m, "trailing '/'")
    assert (d / "f2").exists() and (d / "sub" / "g").exists()

    out = call_delete(m, f"file://{d}/", recursive=True)
    assert_returned(out, m)
    assert not (d / "f2").exists() and not (d / "sub" / "g").exists()


def test_file_backend_exact_delete_of_a_directory_is_refused_and_deletes_nothing(tmp_path, mock_factory):
    """`delete(dir)` / `delete(dir/)` without recursive=True must not remove the
    directory tree (recursion without recursive=True)."""
    m = mock_factory()  # unused; keeps the helper signature
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "f1").write_bytes(b"1")
    (d / "sub" / "g").write_bytes(b"3")

    for uri in (f"file://{d}", f"file://{d}/"):
        out = call_delete(m, uri)
        assert_raises(out, "RuntimeError", m, "is a directory")
        assert "recursive=True" in out["msg"], out
        assert (d / "f1").exists() and (d / "sub" / "g").exists() and (d / "sub").is_dir()

    # with recursive=True and a trailing '/', the existing -r contract applies
    out = call_delete(m, f"file://{d}/", recursive=True)
    assert_returned(out, m)
    assert not (d / "f1").exists() and not (d / "sub" / "g").exists()

    # a regular file is still deleted, a missing path is still success
    (tmp_path / "plain").write_bytes(b"x")
    assert_returned(call_delete(m, f"file://{tmp_path}/plain"), m)
    assert not (tmp_path / "plain").exists()
    assert_returned(call_delete(m, f"file://{tmp_path}/plain"), m)


# ----------------------------------------------------------------------------
# 4. Op-log: an exact delete is recorded as exactly one DELETE row (no HEAD row)
# ----------------------------------------------------------------------------

OPLOG_CHILD = """
    oplog = sys.argv[1]
    s3dlio.init_op_log(oplog)
    s3dlio.delete("s3://bkt/foo")        # present
    s3dlio.delete("s3://bkt/missing")    # absent: success
    s3dlio.delete("s3://bkt/foo/")       # trailing slash: just a key
    s3dlio.finalize_op_log()
"""


def test_exact_delete_records_one_delete_row_each_and_no_head_row(mock_factory, tmp_path):
    from test_oplog_lifecycle import read_rows

    m = mock_factory(["foo", "foo/x", "keep"])
    oplog = tmp_path / "ops.tsv.zst"
    proc = run_scenario(textwrap.dedent(OPLOG_CHILD), str(oplog), env=env_for(m))
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    rows = read_rows(oplog)
    assert [r["op"] for r in rows] == ["DELETE", "DELETE", "DELETE"], rows
    assert [r["error"] for r in rows] == ["", "", ""], rows
    assert not m.requests("HEAD"), m.render()
    assert not m.lists(), m.render()
    assert m.keys == {"foo/x", "keep"}, sorted(m.keys)
