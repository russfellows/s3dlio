"""Python compressed S3 writer: s3dlio issue #218 (silent data loss).

``s3dlio.create_s3_writer(uri, opts)`` with ``opts.with_compression("zstd")`` used to
upload only the tail of the zstd stream for any payload large enough that zstd emits
blocks before ``finalize()``: the object had no frame header and no early blocks, and
could not be decompressed. The caller saw no error.

The test drives the real extension module in a FRESH interpreter (the S3 client is
process-global) against an in-process Python mock S3 server that really implements
multipart upload: it keeps the uploaded parts and joins them on
CompleteMultipartUpload. The parent then decompresses the object the server STORED
with the ``zstd`` command-line tool (``-t`` proves one complete frame) and compares
the decompressed bytes with the payload, byte for byte.

Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_python_compressed_s3_writer_218.py``.
"""

import hashlib
import json
import random
import subprocess
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from test_oplog_lifecycle import run_scenario, zstd_cli

BUCKET = "bkt"
CHUNK = 64 * 1024


class MultipartMock:
    """A path-style S3 server that stores objects, including multipart uploads."""

    def __init__(self):
        self.objects = {}  # key -> bytes of the stored object
        self.uploads = {}  # upload id -> (key, {part number: bytes})
        self.requests = []  # (method, key, sorted query keys)
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
                key = unquote(split.path).lstrip("/").partition("/")[2]
                status, headers, resp = mock.respond(self.command, key, query, body)
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(resp)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(resp)

            do_GET = do_HEAD = do_POST = do_DELETE = do_PUT = _handle

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def respond(self, method, key, query, body):
        with self.lock:
            self.requests.append((method, key, sorted(query)))
            if method == "POST" and "uploads" in query:
                upload_id = f"up-{len(self.uploads) + 1}"
                self.uploads[upload_id] = (key, {})
                xml = (
                    '<?xml version="1.0"?><InitiateMultipartUploadResult>'
                    f"<Bucket>{BUCKET}</Bucket><Key>{key}</Key><UploadId>{upload_id}</UploadId>"
                    "</InitiateMultipartUploadResult>"
                )
                return 200, {"Content-Type": "application/xml"}, xml.encode()
            if method == "PUT" and "partNumber" in query:
                self.uploads[query["uploadId"]][1][int(query["partNumber"])] = body
                return 200, {"ETag": f'"etag-{query["partNumber"]}"'}, b""
            if method == "POST" and "uploadId" in query:
                stored_key, parts = self.uploads.pop(query["uploadId"])
                self.objects[stored_key] = b"".join(parts[n] for n in sorted(parts))
                xml = (
                    '<?xml version="1.0"?><CompleteMultipartUploadResult>'
                    f"<Bucket>{BUCKET}</Bucket><Key>{stored_key}</Key><ETag>\"done\"</ETag>"
                    "</CompleteMultipartUploadResult>"
                )
                return 200, {"Content-Type": "application/xml"}, xml.encode()
            if method == "PUT":
                self.objects[key] = body
                return 200, {"ETag": '"put"'}, b""
            if method == "DELETE":
                self.uploads.clear() if "uploadId" in query else self.objects.pop(key, None)
                return 204, {}, b""
            if method == "HEAD":
                if key in self.objects:
                    return 200, {"Content-Length": str(len(self.objects[key]))}, b""
                return 404, {}, b""
            return 200, {}, b""

    def part_uploads(self):
        with self.lock:
            return sum(1 for m, _k, q in self.requests if m == "PUT" and "partNumber" in q)


@pytest.fixture
def mock():
    m = MultipartMock()
    yield m
    m.close()


def mixed_payload(n: int) -> bytes:
    """64 KiB stretches alternating incompressible and highly compressible bytes."""
    rng = random.Random(218)
    out = bytearray()
    random_stretch = True
    while len(out) < n:
        stretch = min(CHUNK, n - len(out))
        if random_stretch:
            out += rng.randbytes(stretch)
        else:
            text = b"the quick brown fox jumps over the lazy dog; "
            out += (text * (stretch // len(text) + 1))[:stretch]
        random_stretch = not random_stretch
    return bytes(out)


CHILD = """
    import json
    src, uri = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    opts = s3dlio.PyWriterOptions()
    opts.with_compression("zstd", None)
    writer = s3dlio.create_s3_writer(uri, opts)
    for i in range(0, len(data), {chunk}):
        writer.write_chunk(data[i:i + {chunk}])
    written, compressed = writer.finalize()
    print("RESULT " + json.dumps({{"written": written, "compressed": compressed}}))
""".format(chunk=CHUNK)


def env_for(mock):
    return {
        "AWS_ENDPOINT_URL": mock.url,
        "AWS_ENDPOINT_URL_S3": mock.url,
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def upload(mock, tmp_path, data: bytes, key: str):
    src = tmp_path / "payload.bin"
    src.write_bytes(data)
    proc = run_scenario(
        textwrap.dedent(CHILD), str(src), f"s3://{BUCKET}/{key}", env=env_for(mock)
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize(
    "size,min_parts",
    [
        (32 * 1024, 1),  # control: small payloads always worked
        (6 * 1024 * 1024 + 123, 1),  # zstd emits blocks long before finalize
        (18 * 1024 * 1024 + 7, 2),  # compressed object bigger than one 16 MiB part
    ],
)
def test_compressed_writer_uploads_a_complete_zstd_frame(mock, tmp_path, size, min_parts):
    if size > 16 * 1024 * 1024:
        # Incompressible throughout, so the compressed object spans two parts.
        data = random.Random(7).randbytes(size)
    else:
        data = mixed_payload(size)
    result = upload(mock, tmp_path, data, "obj")

    stored = mock.objects.get("obj.zst")
    assert stored is not None, f"no object obj.zst was stored: {sorted(mock.objects)}"
    assert mock.part_uploads() >= min_parts, f"parts: {mock.requests}"
    assert result["written"] == len(data)
    assert result["compressed"] == len(stored), "finalize must report the stored compressed size"

    zstd = zstd_cli()
    packed = tmp_path / "stored.zst"
    packed.write_bytes(stored)
    test = subprocess.run([zstd, "-t", str(packed)], capture_output=True, text=True, check=False)
    assert test.returncode == 0, (
        f"the {len(stored)} stored bytes are not a complete zstd stream: {test.stderr}"
    )
    plain = subprocess.run([zstd, "-dc", str(packed)], capture_output=True, check=True).stdout
    assert len(plain) == len(data), f"decompressed {len(plain)} bytes, wrote {len(data)}"
    assert hashlib.sha256(plain).digest() == hashlib.sha256(data).digest(), (
        "the decompressed bytes differ from the written bytes"
    )
