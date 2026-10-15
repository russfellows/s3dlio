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
from urllib.parse import unquote

import pytest
import s3dlio

ACCOUNT = "devacct"
CONTAINER = "ctr"


def _body_for(path):
    """Distinct, deterministic content per blob path."""
    return (path.encode() + b"|") * 40


class _FakeBlobService(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: ClassVar[list] = []
    seen_lock: ClassVar[threading.Lock] = threading.Lock()

    def log_message(self, *args):  # keep pytest output quiet
        pass

    def _send_blob(self, send_body):
        path = unquote(
            self.path.split("?", 1)[0]
        )  # the SDK sends "/" in a blob name as %2F
        with self.seen_lock:
            self.seen.append((self.command, path))
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
