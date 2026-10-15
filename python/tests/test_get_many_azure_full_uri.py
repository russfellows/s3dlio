"""Issue #242: ``get_many`` on ``az://`` URIs must hand the Azure store the full URI.

``get_many`` used to strip the scheme and container and pass the bare key to the
Azure store, whose ``get`` parses the account and container out of the URI, so
every key failed while the same URI worked with ``get()``.

No Azure service is used.  ``AZURE_STORAGE_ENDPOINT`` points s3dlio at a tiny
HTTP server on 127.0.0.1 that plays the part of Blob storage (an ``http``
endpoint is used without credentials).  The server records every request path,
so the test can check that the requests went to ``/<account>/<container>/<key>``.

The wheel must be built with the Azure backend (``./build_pyo3.sh full``); the
test is skipped on a wheel without it.  GCS has no equivalent local stand-in
(its client needs Google credentials), so ``gs://`` is covered only by the Rust
unit tests of ``object_store::get_many_with_stores``.
"""

import re
import threading
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import unquote, urlsplit

import pytest
import s3dlio

ACCOUNT = "devacct"
CONTAINER = "ctr"


def _body_for(path):
    """Distinct, deterministic content per blob path."""
    unit = path.encode() + b"|"
    if (
        "big" in path
    ):  # a multi-MiB blob: many chunks, so a buffer that grows would show
        return unit * (3 * 1024 * 1024 // len(unit) + 1)
    return unit * 40


class _FakeBlobService(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: ClassVar[list] = []
    seen_lock: ClassVar[threading.Lock] = threading.Lock()
    queries: ClassVar[list] = []

    def log_message(self, *args):  # keep pytest output quiet
        pass

    def _send_blob(self, send_body):
        path = unquote(
            self.path.split("?", 1)[0]
        )  # the SDK sends "/" in a blob name as %2F
        with self.seen_lock:
            self.seen.append((self.command, path))
            self.queries.append(urlsplit(self.path).query)
        prefix = f"/{ACCOUNT}/{CONTAINER}/"
        if not path.startswith(prefix):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("x-ms-error-code", "BlobNotFound")
            self.end_headers()
            return
        body = _body_for(path)
        total = len(body)
        status = 200
        rng = self.headers.get("Range") or self.headers.get("x-ms-range")
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng)
            start = int(m.group(1))
            end = min(int(m.group(2)), total - 1) if m.group(2) else total - 1
            chunk = body[start : end + 1]
            status = 206
            self.send_response(status)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        else:
            chunk = body
            self.send_response(status)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("ETag", '"0x8D0000000000001"')
        self.send_header("Last-Modified", formatdate(usegmt=True))
        self.send_header("x-ms-blob-type", "BlockBlob")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if send_body:
            self.wfile.write(chunk)

    def do_HEAD(self):
        self._send_blob(send_body=False)

    def do_GET(self):
        self._send_blob(send_body=True)


@pytest.fixture()
def fake_blob_endpoint(monkeypatch):
    _FakeBlobService.seen = []
    _FakeBlobService.queries = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBlobService)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "AZURE_STORAGE_ENDPOINT", f"http://127.0.0.1:{server.server_port}"
    )
    try:
        yield _FakeBlobService.seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _require_azure_backend():
    try:
        s3dlio.get(f"az://{ACCOUNT}/{CONTAINER}/probe-for-backend")
    except Exception as e:  # noqa: BLE001 - only the message matters
        if "Azure backend is not enabled" in str(e):
            pytest.skip("wheel was built without the Azure backend")


def test_get_many_azure_requests_the_full_object_path(fake_blob_endpoint):
    _require_azure_backend()
    keys = [f"dir/sub/object-{i}.bin" for i in range(5)]
    uris = [f"az://{ACCOUNT}/{CONTAINER}/{k}" for k in keys]

    # Same URIs work through get(): the baseline the issue describes.
    single = bytes(s3dlio.get(uris[0]))
    assert single == _body_for(f"/{ACCOUNT}/{CONTAINER}/{keys[0]}")

    fake_blob_endpoint.clear()
    got = s3dlio.get_many(uris, max_in_flight=3)

    assert [u for u, _ in got] == uris, "results stay in input order"
    for (uri, data), key in zip(got, keys):
        assert bytes(data) == _body_for(f"/{ACCOUNT}/{CONTAINER}/{key}"), uri
    paths = {p for _, p in fake_blob_endpoint}
    assert paths == {f"/{ACCOUNT}/{CONTAINER}/{k}" for k in keys}, paths


def test_get_many_azure_multi_mib_blobs_come_back_intact(fake_blob_endpoint):
    _require_azure_backend()
    keys = [f"dir/big-{i}.bin" for i in range(3)]
    uris = [f"az://{ACCOUNT}/{CONTAINER}/{k}" for k in keys]
    got = s3dlio.get_many(uris, max_in_flight=2)
    for (uri, data), key in zip(got, keys):
        want = _body_for(f"/{ACCOUNT}/{CONTAINER}/{key}")
        assert len(want) > 3 * 1024 * 1024
        assert bytes(data) == want, uri


def test_get_many_azure_writes_op_log_rows(fake_blob_endpoint, tmp_path):
    _require_azure_backend()
    keys = [f"dir/log-{i}.bin" for i in range(6)]
    uris = [f"az://{ACCOUNT}/{CONTAINER}/{k}" for k in keys]
    s3dlio.init_op_log(str(tmp_path / "oplog.tsv.zst"))
    try:
        s3dlio.get_many(uris, max_in_flight=3)
    finally:
        s3dlio.finalize_op_log()
    stats = s3dlio.op_log_stats()
    # One GET row per object, as get() writes.
    assert stats["accepted"] >= len(uris), stats


def test_get_many_azure_refuses_a_uri_without_a_key(fake_blob_endpoint):
    _require_azure_backend()
    fake_blob_endpoint.clear()
    with pytest.raises(RuntimeError, match="no key specified"):
        s3dlio.get_many(
            [f"az://{ACCOUNT}/{CONTAINER}/ok.bin", f"az://{ACCOUNT}/{CONTAINER}/"]
        )
    assert fake_blob_endpoint == [], (
        "no request may be made for a batch with a keyless URI"
    )


def test_get_many_azure_error_names_the_failing_uri(fake_blob_endpoint):
    _require_azure_backend()
    missing = f"az://{ACCOUNT}/other-container/missing.bin"  # the fake answers 404
    with pytest.raises(RuntimeError) as info:
        s3dlio.get_many([missing])
    assert missing in str(info.value), str(info.value)


def test_azure_sas_mode_sends_the_token_on_every_request(
    fake_blob_endpoint, monkeypatch
):
    _require_azure_backend()
    monkeypatch.setenv("S3DLIO_AZURE_CREDENTIAL", "sas")
    monkeypatch.setenv("S3DLIO_AZURE_SAS_TOKEN", "?sv=2024-11-04&sig=abc%2Bdef")
    fake_blob_endpoint.clear()
    _FakeBlobService.queries.clear()
    got = s3dlio.get_many([f"az://{ACCOUNT}/{CONTAINER}/dir/sas-0.bin"])
    assert len(got) == 1
    assert _FakeBlobService.queries, "no request reached the fake endpoint"
    for query in _FakeBlobService.queries:
        assert "sv=2024-11-04" in query and "sig=abc%2Bdef" in query, query


def test_azure_unknown_credential_mode_is_an_error_naming_the_valid_ones(
    fake_blob_endpoint, monkeypatch
):
    _require_azure_backend()
    monkeypatch.setenv("S3DLIO_AZURE_CREDENTIAL", "password")
    with pytest.raises(RuntimeError) as info:
        s3dlio.get(f"az://{ACCOUNT}/{CONTAINER}/dir/x.bin")
    message = str(info.value)
    for mode in (
        "auto",
        "managed-identity",
        "client-secret",
        "workload-identity",
        "sas",
        "none",
    ):
        assert mode in message, message


def test_azure_bearer_token_mode_on_http_endpoint_says_it_needs_https(
    fake_blob_endpoint, monkeypatch
):
    _require_azure_backend()
    monkeypatch.setenv("S3DLIO_AZURE_CREDENTIAL", "managed-identity")
    with pytest.raises(RuntimeError, match="https"):
        s3dlio.get(f"az://{ACCOUNT}/{CONTAINER}/dir/x.bin")
