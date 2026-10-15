"""Release validation for the full-backend wheel (``./build_pyo3.sh full``): the Azure and GCS
code paths that can run offline, through the real Python boundary.

``az://`` runs against a local fake Blob service (``AZURE_STORAGE_ENDPOINT`` on 127.0.0.1, http,
so no credential is involved): put, get, range get, stat, list, delete and ``get_many``, with the
requests the fake saw checked. ``gs://`` has no offline stand-in (its client needs Google
credentials and a bucket-layout RPC), so only what does not touch the network runs: the store is
built (proving the GCS backend is compiled in) and ``get_many`` refuses bad batches before any
request. The whole file is skipped on a wheel built without the backends.
"""

import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import unquote, urlsplit
from xml.sax.saxutils import escape

import pytest
import s3dlio

ACCOUNT = "fullacct"
CONTAINER = "ctr"
REQUEST_CAP = 300


class _FakeBlobs(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    blobs: ClassVar[dict] = {}
    requests: ClassVar[list] = []
    lock: ClassVar[threading.Lock] = threading.Lock()

    def log_message(self, *args):
        pass

    def _reply(self, status, body=b"", headers=None):
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if "Content-Length" not in (headers or {}):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self):
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        body = b""
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            body = self.rfile.read(length)
        with self.lock:
            self.requests.append(
                (self.command, path, parts.query, self.headers.get("Range"))
            )
            over_cap = len(self.requests) > REQUEST_CAP
        if over_cap:
            return self._reply(403)
        prefix = f"/{ACCOUNT}/{CONTAINER}/"
        if (
            self.command == "GET"
            and path == f"/{ACCOUNT}/{CONTAINER}"
            and "comp=list" in parts.query
        ):
            names = sorted(n for n in self.blobs if n.startswith(prefix))
            items = "".join(
                f"<Blob><Name>{escape(n[len(prefix) :])}</Name><Properties>"
                f"<Content-Length>{len(self.blobs[n])}</Content-Length></Properties></Blob>"
                for n in names
            )
            xml = (
                '<?xml version="1.0" encoding="utf-8"?><EnumerationResults '
                f'ContainerName="{CONTAINER}"><Blobs>{items}</Blobs><NextMarker /></EnumerationResults>'
            )
            return self._reply(200, xml.encode(), {"Content-Type": "application/xml"})
        if not path.startswith(prefix):
            return self._reply(404, headers={"x-ms-error-code": "BlobNotFound"})
        if self.command == "PUT":
            self.blobs[path] = body
            return self._reply(
                201,
                headers={
                    "ETag": '"0x1"',
                    "Last-Modified": "Wed, 14 Oct 2026 10:00:00 GMT",
                },
            )
        if self.command == "DELETE":
            existed = self.blobs.pop(path, None) is not None
            return self._reply(
                202 if existed else 404,
                headers=None if existed else {"x-ms-error-code": "BlobNotFound"},
            )
        data = self.blobs.get(path)
        if data is None:
            return self._reply(404, headers={"x-ms-error-code": "BlobNotFound"})
        total = len(data)
        headers = {
            "ETag": '"0x1"',
            "Last-Modified": "Wed, 14 Oct 2026 10:00:00 GMT",
            "x-ms-blob-type": "BlockBlob",
            "Accept-Ranges": "bytes",
        }
        status, chunk = 200, data
        rng = self.headers.get("Range") or self.headers.get("x-ms-range")
        if rng and self.command == "GET":
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng)
            start = int(m.group(1))
            if start >= total:
                return self._reply(
                    416,
                    headers={
                        "x-ms-error-code": "InvalidRange",
                        "Content-Range": f"bytes */{total}",
                    },
                )
            end = min(int(m.group(2)), total - 1) if m.group(2) else total - 1
            status, chunk = 206, data[start : end + 1]
            headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        headers["Content-Length"] = str(total if self.command == "HEAD" else len(chunk))
        return self._reply(status, chunk, headers)

    do_GET = do_HEAD = do_PUT = do_DELETE = _handle


@pytest.fixture()
def fake(monkeypatch):
    _FakeBlobs.blobs = {}
    _FakeBlobs.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBlobs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "AZURE_STORAGE_ENDPOINT", f"http://127.0.0.1:{server.server_port}"
    )
    for var in (
        "S3DLIO_AZURE_CREDENTIAL",
        "S3DLIO_AZURE_SAS_TOKEN",
        "AZURE_BLOB_ENDPOINT_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        yield _FakeBlobs
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _require_backends():
    try:
        s3dlio.mkdir(
            "gs://bucket-for-backend-probe/dir"
        )  # a no-op that still builds the store
    except Exception as e:
        if "not enabled" in str(e):
            pytest.skip("wheel was built without the GCS backend")
        raise
    try:
        s3dlio.stat(f"az://{ACCOUNT}/{CONTAINER}/backend-probe")
    except Exception as e:  # noqa: BLE001
        if "not enabled" in str(e):
            pytest.skip("wheel was built without the Azure backend")


def uri(key):
    return f"az://{ACCOUNT}/{CONTAINER}/{key}"


def payload(n, seed=1):
    return bytes((i * 31 + seed) & 0xFF for i in range(n))


def test_azure_put_get_range_stat_list_delete_round_trip(fake):
    _require_backends()
    small, big = payload(1000), payload(3 * 1024 * 1024 + 17, 7)
    s3dlio.put_bytes(uri("dir/small.bin"), small)
    s3dlio.put_bytes(uri("dir/big.bin"), big)

    assert bytes(s3dlio.get(uri("dir/small.bin"))) == small
    assert bytes(s3dlio.get(uri("dir/big.bin"))) == big
    assert bytes(s3dlio.get_range(uri("dir/big.bin"), 100, 5000)) == big[100:5100]
    assert s3dlio.stat(uri("dir/big.bin"))["size"] == len(big)

    listed = s3dlio.list(f"az://{ACCOUNT}/{CONTAINER}/dir/", recursive=True)
    assert {u.rsplit("/", 1)[-1] for u in listed} == {"small.bin", "big.bin"}, listed

    s3dlio.delete(uri("dir/small.bin"))
    with pytest.raises(RuntimeError):
        s3dlio.get(uri("dir/small.bin"))


def test_azure_get_many_makes_one_get_per_blob_and_no_head(fake):
    _require_backends()
    keys = [f"many/o{i}.bin" for i in range(8)]
    for i, k in enumerate(keys):
        s3dlio.put_bytes(uri(k), payload(2000 + i, i))
    fake.requests.clear()

    got = s3dlio.get_many([uri(k) for k in keys], max_in_flight=4)
    assert [u for u, _ in got] == [uri(k) for k in keys]
    for i, (_, data) in enumerate(got):
        assert bytes(data) == payload(2000 + i, i)
    methods = [m for m, *_ in fake.requests]
    assert methods.count("GET") == len(keys), fake.requests
    assert "HEAD" not in methods, "the default mode must not ask for the size first"


def test_azure_overwritten_blob_is_read_correctly_with_few_requests(fake):
    """The stale-size scenario from the Blue review, through the wheel: a blob read at one
    byte, then replaced by a 1.5 MiB one, must come back whole in a handful of requests."""
    _require_backends()
    s3dlio.put_bytes(uri("grow.bin"), b"x")
    assert bytes(s3dlio.get(uri("grow.bin"))) == b"x"
    big = payload(1_500_000, 3)
    s3dlio.put_bytes(uri("grow.bin"), big)
    fake.requests.clear()
    assert bytes(s3dlio.get(uri("grow.bin"))) == big
    assert len(fake.requests) <= 6, fake.requests[:6]
    # And inside get_many.
    fake.blobs[f"/{ACCOUNT}/{CONTAINER}/grow.bin"] = payload(
        1_700_000, 5
    )  # changed behind its back
    fake.requests.clear()
    [(_, data)] = s3dlio.get_many([uri("grow.bin")])
    assert bytes(data) == payload(1_700_000, 5)
    assert len(fake.requests) <= 6, fake.requests[:6]


def test_azure_get_many_error_names_the_failing_uri_and_stops_the_batch(fake):
    _require_backends()
    s3dlio.put_bytes(uri("ok.bin"), b"ok")
    missing = uri("missing/none.bin")
    with pytest.raises(RuntimeError) as info:
        s3dlio.get_many([missing] + [uri("ok.bin")] * 3, max_in_flight=1)
    assert missing in str(info.value), str(info.value)


def test_gcs_backend_is_compiled_in_and_get_many_refuses_bad_batches_before_any_request(
    fake,
):
    _require_backends()
    fake.requests.clear()
    # The batch checks run before any store is built or request made, so no credentials or
    # network are needed for these.
    with pytest.raises(RuntimeError, match="no key specified"):
        s3dlio.get_many(["gs://some-bucket/"])
    with pytest.raises(RuntimeError, match="no key specified"):
        s3dlio.get_many([uri("fine.bin"), f"az://{ACCOUNT}/{CONTAINER}/"])
    with pytest.raises(RuntimeError, match="same backend scheme"):
        s3dlio.get_many(["gs://some-bucket/a", uri("b")])
    assert fake.requests == [], "no request may be made for a refused batch"


def test_azure_deleting_a_missing_blob_succeeds_and_recursive_delete_works(fake):
    """The idempotent-delete contract through the wheel (the typed BlobNotFound must survive)."""
    _require_backends()
    s3dlio.delete(uri("never/existed.bin"))  # no error
    for i in range(3):
        s3dlio.put_bytes(uri(f"tree/o{i}.bin"), payload(100 + i, i))
    s3dlio.delete(f"az://{ACCOUNT}/{CONTAINER}/tree/", recursive=True)
    assert not [k for k in fake.blobs if "/tree/" in k], fake.blobs.keys()
    s3dlio.delete(
        f"az://{ACCOUNT}/{CONTAINER}/tree/", recursive=True
    )  # again: still fine


def test_azure_list_containers_refuses_malformed_accounts_before_any_request(fake, monkeypatch):
    _require_backends()
    monkeypatch.delenv("AZURE_STORAGE_ENDPOINT")  # the strict, host-name rule
    # A bearer-token mode with no variables: if a credential were built, the error would be about
    # AZURE_TENANT_ID rather than the account.
    monkeypatch.setenv("S3DLIO_AZURE_CREDENTIAL", "client-secret")
    for bad in ("az://attacker.example#", "az://a?b", "az://acct@evil", "az://x%2fy"):
        with pytest.raises(RuntimeError) as info:
            s3dlio.list_containers(bad)
        message = str(info.value)
        assert "invalid Azure account" in message and "AZURE_TENANT_ID" not in message, message
    assert fake.requests == []


def test_azure_malformed_account_names_are_refused_before_any_request(fake):
    _require_backends()
    fake.requests.clear()
    for bad in ("az://attacker.example#/c/k", "az://a?b/c/k", "az://acct@evil/c/k"):
        with pytest.raises(RuntimeError, match="invalid Azure account"):
            s3dlio.get(bad)
    assert fake.requests == []
