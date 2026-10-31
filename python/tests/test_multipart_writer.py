# python/tests/test_multipart_writer.py
"""MultipartUploadWriter tests.

The writer speaks S3 only (``from_uri`` rejects every other scheme), so these tests need an
S3 endpoint.  Two modes:

* Live S3: set S3DLIO_TEST_BUCKET (an existing, writable bucket); credentials and endpoint
  come from whatever the environment provides (AWS_* variables, profiles, AWS_ENDPOINT_URL
  for an S3-compatible server).  Objects are written under a unique prefix and deleted
  afterwards.  The tests run in the current process.
* Otherwise the tests run against a strict in-process S3 server (``MiniS3``, multipart
  upload only) in an ISOLATED CHILD PROCESS: the process-wide S3 client is built once per
  process and bound to its endpoint, so pointing it at a throw-away server inside the pytest
  process would leak into every later module (and fail if an earlier module had already
  built the client).  The child sets S3DLIO_MPU_INNER=1, which selects the in-process mode;
  ``TestMultipartWriterIsolated`` spawns it and reports the result.
"""

import contextlib
import hashlib
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
# The default listen backlog of these stand-in servers is 5; concurrent connects beyond it wait for a SYN
# retransmit (1, 2, 4 s). A bigger backlog removes that source of flakiness (issue #304 round 2).
ThreadingHTTPServer.request_queue_size = 512
from importlib import import_module
from urllib.parse import parse_qs, unquote, urlsplit

import s3dlio

LIVE = bool(os.environ.get("S3DLIO_TEST_BUCKET"))
INNER = os.environ.get("S3DLIO_MPU_INNER") == "1"
MIN_PART = 5 << 20  # S3's minimum size of every part except the last

# Everything the S3 client reads that the child process must not inherit.
CLIENT_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_CA_BUNDLE",
    "S3DLIO_TEST_BUCKET",
)


def unique_bucket(prefix="mpu-py"):
    return f"{prefix}-{os.getpid()}-{int(time.time())}"


def etag_of(data):
    return f'"{hashlib.md5(data).hexdigest()}"'


def payload_of(n, seed):
    """Deterministic non-uniform bytes, so misordered or corrupted parts are detectable."""
    return random.Random(seed).randbytes(n)


def s3_error(status, code, message):
    body = f'<?xml version="1.0"?><Error><Code>{code}</Code><Message>{message}</Message></Error>'
    return status, {"Content-Type": "application/xml"}, body.encode()


class MiniS3:
    """A strict path-style S3 server: bucket create/delete, PUT/GET/HEAD/DELETE object
    (with prefix, open-ended and suffix ranges, 416), and multipart upload (initiate, upload
    part, complete, abort).  CompleteMultipartUpload is validated the way S3 does: the body
    is parsed and wrong part ETags, wrong order, unknown or unlisted parts, and non-final
    parts under 5 MiB are refused with S3-style errors."""

    def __init__(self):
        self.objects = {}  # (bucket, key) -> bytes
        self.etags = {}  # (bucket, key) -> ETag of the stored object
        self.uploads = {}  # upload id -> (bucket, key, {part number: (bytes, etag)})
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
                        size = int(
                            self.rfile.readline().split(b";")[0].strip() or b"0", 16
                        )
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
                query = {
                    k: v[0]
                    for k, v in parse_qs(split.query, keep_blank_values=True).items()
                }
                bucket, _, key = unquote(split.path).lstrip("/").partition("/")
                status, headers, resp = mock.respond(
                    self.command, bucket, key, query, body, self.headers.get("Range")
                )
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                if "Content-Length" not in headers:
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

    @staticmethod
    def parse_range(header, size):
        """Return (start, end) inclusive, or None when unsatisfiable (-> 416)."""
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if not m or (m.group(1) == "" and m.group(2) == ""):
            return None
        first, last = m.groups()
        if first == "":  # suffix range: the last N bytes
            n = int(last)
            if n == 0 or size == 0:
                return None
            return max(size - n, 0), size - 1
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            return None
        return start, end

    @staticmethod
    def parse_part_list(body):
        """Return [(int PartNumber, stripped ETag), ...] in document order, or raise
        ValueError when malformed.  No XML parser is used (CodeQL py/xml-bomb): the
        mock only needs the part list, so it rejects any DOCTYPE/ENTITY declaration."""
        text = body.decode("utf-8")  # UnicodeDecodeError is a ValueError
        if "<!" in text:
            raise ValueError("DOCTYPE/ENTITY declarations are not accepted")
        entities = (("&quot;", '"'), ("&apos;", "'"), ("&lt;", "<"), ("&gt;", ">"))

        def field(block, name):
            start = block.find(f"<{name}>")
            end = block.find(f"</{name}>", start)
            if start < 0 or end < 0:
                raise ValueError(f"missing or unterminated <{name}>")
            value = block[start + len(name) + 2 : end]
            if "<" in value:
                raise ValueError(f"unexpected markup inside <{name}>")
            check = value
            for entity, _ in entities + (("&amp;", "&"),):
                check = check.replace(entity, "")
            if "&" in check:  # numeric or unknown reference: treat as malformed
                raise ValueError(f"unsupported entity inside <{name}>")
            for entity, char in entities:
                value = value.replace(entity, char)
            return value.replace("&amp;", "&")  # &amp; last, as ElementTree does

        listed = []
        pos = 0
        while True:
            start = text.find("<Part>", pos)
            if start < 0:
                return listed
            end = text.find("</Part>", start)
            if end < 0:
                raise ValueError("unterminated <Part>")
            block = text[start + len("<Part>") : end]
            listed.append(
                (int(field(block, "PartNumber")), field(block, "ETag").strip())
            )
            pos = end + len("</Part>")

    def complete(self, bucket, key, upload_id, body):
        if upload_id not in self.uploads:
            return s3_error(404, "NoSuchUpload", "The specified upload does not exist.")
        up_bucket, up_key, parts = self.uploads[upload_id]
        try:
            listed = self.parse_part_list(body)
        except ValueError:
            return s3_error(
                400, "MalformedXML", "The XML you provided was not well-formed."
            )
        if not listed:
            return s3_error(400, "MalformedXML", "You must specify at least one part.")
        numbers = [n for n, _ in listed]
        if numbers != sorted(set(numbers)):
            return s3_error(
                400,
                "InvalidPartOrder",
                "The list of parts was not in ascending order.",
            )
        for n, etag in listed:
            if n not in parts:
                return s3_error(400, "InvalidPart", f"Part {n} was never uploaded.")
            if parts[n][1] != etag:
                return s3_error(
                    400, "InvalidPart", f"Part {n} ETag does not match the upload."
                )
        unlisted = sorted(set(parts) - set(numbers))
        if unlisted:
            return s3_error(
                400, "InvalidPart", f"Uploaded part(s) {unlisted} are not listed."
            )
        if numbers != list(range(1, len(numbers) + 1)):
            return s3_error(400, "InvalidPart", "Part numbers must be 1..N.")
        for n in numbers[:-1]:
            if len(parts[n][0]) < MIN_PART:
                return s3_error(
                    400,
                    "EntityTooSmall",
                    f"Part {n} is smaller than the 5 MiB minimum.",
                )
        data = b"".join(parts[n][0] for n in numbers)
        digests = b"".join(hashlib.md5(parts[n][0]).digest() for n in numbers)
        etag = f'"{hashlib.md5(digests).hexdigest()}-{len(numbers)}"'
        self.objects[(up_bucket, up_key)] = data
        self.etags[(up_bucket, up_key)] = etag
        del self.uploads[upload_id]
        xml = {"Content-Type": "application/xml"}
        return (
            200,
            xml,
            (
                '<?xml version="1.0"?><CompleteMultipartUploadResult>'
                f"<Bucket>{up_bucket}</Bucket><Key>{up_key}</Key><ETag>{etag}</ETag>"
                "</CompleteMultipartUploadResult>"
            ).encode(),
        )

    def respond(self, method, bucket, key, query, body, range_header):
        xml = {"Content-Type": "application/xml"}
        with self.lock:
            if not key:  # bucket-level request
                return (204 if method == "DELETE" else 200), {}, b""
            if method == "POST" and "uploads" in query:
                upload_id = f"up-{len(self.uploads) + 1}-{uuid.uuid4().hex[:8]}"
                self.uploads[upload_id] = (bucket, key, {})
                return (
                    200,
                    xml,
                    (
                        '<?xml version="1.0"?><InitiateMultipartUploadResult>'
                        f"<Bucket>{bucket}</Bucket><Key>{key}</Key><UploadId>{upload_id}</UploadId>"
                        "</InitiateMultipartUploadResult>"
                    ).encode(),
                )
            if method == "PUT" and "partNumber" in query:
                upload = self.uploads.get(query.get("uploadId"))
                if upload is None:
                    return s3_error(
                        404, "NoSuchUpload", "The specified upload does not exist."
                    )
                number = int(query["partNumber"])
                if not 1 <= number <= 10000:
                    return s3_error(
                        400, "InvalidArgument", "Part number must be 1..10000."
                    )
                etag = etag_of(body)
                upload[2][number] = (body, etag)
                return 200, {"ETag": etag}, b""
            if method == "POST" and "uploadId" in query:
                return self.complete(bucket, key, query["uploadId"], body)
            if method == "DELETE":
                if "uploadId" in query:
                    self.uploads.pop(query["uploadId"], None)
                else:
                    self.objects.pop((bucket, key), None)
                    self.etags.pop((bucket, key), None)
                return 204, {}, b""
            if method == "PUT":
                self.objects[(bucket, key)] = body
                self.etags[(bucket, key)] = etag_of(body)
                return 200, {"ETag": self.etags[(bucket, key)]}, b""
            if method in ("GET", "HEAD"):
                data = self.objects.get((bucket, key))
                if data is None:
                    return s3_error(
                        404, "NoSuchKey", "The specified key does not exist."
                    )
                etag = self.etags[(bucket, key)]
                if method == "GET" and range_header:
                    span = self.parse_range(range_header, len(data))
                    if span is None:
                        status, headers, resp = s3_error(
                            416,
                            "InvalidRange",
                            "The requested range is not satisfiable",
                        )
                        headers["Content-Range"] = f"bytes */{len(data)}"
                        return status, headers, resp
                    start, end = span
                    return (
                        206,
                        {
                            "ETag": etag,
                            "Content-Range": f"bytes {start}-{end}/{len(data)}",
                        },
                        data[start : end + 1],
                    )
                return (
                    200,
                    {"ETag": etag, "Content-Length": str(len(data))},
                    (b"" if method == "HEAD" else data),
                )
            return 200, {}, b""


def complete_body(parts):
    """CompleteMultipartUpload XML for [(part number, etag), ...]."""
    items = "".join(
        f"<Part><PartNumber>{n}</PartNumber><ETag>{e}</ETag></Part>" for n, e in parts
    )
    return f"<CompleteMultipartUpload>{items}</CompleteMultipartUpload>".encode()


class TestMiniS3Strictness(unittest.TestCase):
    """The mini server itself must refuse what real S3 refuses, or the writer tests above
    could pass on a writer that produces objects S3 would reject.  Pure Python: no S3
    client is involved, so this runs in any process."""

    def setUp(self):
        self.mini = MiniS3()
        self.addCleanup(self.mini.close)

    def start(self, parts):
        """Initiate an upload and upload ``parts`` ({number: bytes}); return (id, etags)."""
        _, _, resp = self.mini.respond("POST", "b", "k", {"uploads": ""}, b"", None)
        upload_id = ET.fromstring(resp).find("UploadId").text
        etags = {}
        for n, data in parts.items():
            status, headers, _ = self.mini.respond(
                "PUT",
                "b",
                "k",
                {"partNumber": str(n), "uploadId": upload_id},
                data,
                None,
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers["ETag"], etag_of(data), "part ETags are md5-hex")
            etags[n] = headers["ETag"]
        return upload_id, etags

    def complete_code(self, upload_id, listed):
        status, _, resp = self.mini.respond(
            "POST", "b", "k", {"uploadId": upload_id}, complete_body(listed), None
        )
        if status == 200:
            return None
        self.assertEqual(status, 400, resp)
        return ET.fromstring(resp).find("Code").text

    def test_part_list_parser_namespaced_root_and_field_order(self):
        body = (
            b'<CompleteMultipartUpload xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            b"<Part><ETag> abc </ETag><PartNumber>2</PartNumber></Part>"
            b"<Part><PartNumber>1</PartNumber><ETag>def</ETag></Part>"
            b"</CompleteMultipartUpload>"
        )
        self.assertEqual(MiniS3.parse_part_list(body), [(2, "abc"), (1, "def")])

    def test_part_list_parser_unescapes_predefined_entities(self):
        body = b"<Part><PartNumber>1</PartNumber><ETag>&quot;a&amp;lt;b&quot;</ETag></Part>"
        self.assertEqual(MiniS3.parse_part_list(body), [(1, '"a&lt;b"')])

    def test_part_list_parser_refuses_doctype_and_malformed(self):
        for body in (
            b'<!DOCTYPE x [<!ENTITY a "b">]><Part></Part>',
            b"<Part><PartNumber>x</PartNumber><ETag>e</ETag></Part>",
            b"<Part><PartNumber>1</PartNumber></Part>",
            b"<Part><PartNumber>1</PartNumber><ETag>e</ETag>",
            b"<Part><PartNumber>1</PartNumber><ETag>&#34;</ETag></Part>",
            b"\xff",
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                MiniS3.parse_part_list(body)

    def test_valid_complete_assembles_in_part_order(self):
        big = b"a" * MIN_PART
        upload_id, etags = self.start({2: b"tail", 1: big})
        self.assertIsNone(self.complete_code(upload_id, [(1, etags[1]), (2, etags[2])]))
        self.assertEqual(self.mini.objects[("b", "k")], big + b"tail")
        self.assertRegex(self.mini.etags[("b", "k")], r'^"[0-9a-f]{32}-2"$')

    def test_wrong_part_etag_refused(self):
        upload_id = self.start({1: b"abc"})[0]
        self.assertEqual(
            self.complete_code(upload_id, [(1, etag_of(b"other"))]), "InvalidPart"
        )
        self.assertNotIn(("b", "k"), self.mini.objects)

    def test_wrong_order_refused(self):
        big = b"a" * MIN_PART
        upload_id, etags = self.start({1: big, 2: b"tail"})
        self.assertEqual(
            self.complete_code(upload_id, [(2, etags[2]), (1, etags[1])]),
            "InvalidPartOrder",
        )

    def test_missing_and_unknown_parts_refused(self):
        big = b"a" * MIN_PART
        upload_id, etags = self.start({1: big, 2: big, 3: b"tail"})
        # part 2 uploaded but not listed
        self.assertEqual(
            self.complete_code(upload_id, [(1, etags[1]), (3, etags[3])]),
            "InvalidPart",
        )
        # part 4 never uploaded
        self.assertEqual(
            self.complete_code(upload_id, [(1, etags[1]), (4, etags[3])]),
            "InvalidPart",
        )
        self.assertEqual(self.complete_code(upload_id, []), "MalformedXML")

    def test_small_non_final_part_refused(self):
        upload_id, etags = self.start({1: b"short", 2: b"tail"})
        self.assertEqual(
            self.complete_code(upload_id, [(1, etags[1]), (2, etags[2])]),
            "EntityTooSmall",
        )

    def test_ranges(self):
        self.mini.respond("PUT", "b", "r", {}, b"0123456789", None)

        def get(rng):
            return self.mini.respond("GET", "b", "r", {}, b"", rng)

        status, headers, body = get("bytes=2-4")
        self.assertEqual((status, body), (206, b"234"))
        self.assertEqual(headers["Content-Range"], "bytes 2-4/10")
        self.assertEqual(get("bytes=7-")[2], b"789")
        self.assertEqual(get("bytes=-3")[2], b"789")  # suffix range
        self.assertEqual(
            get("bytes=-99")[2], b"0123456789"
        )  # suffix larger than object
        self.assertEqual(get("bytes=8-99")[2], b"89")  # end clamped
        status, headers, _ = get("bytes=10-")
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], "bytes */10")
        self.assertEqual(get("bytes=-0")[0], 416)


@unittest.skipUnless(
    LIVE or INNER, "runs live (S3DLIO_TEST_BUCKET) or in the isolated child"
)
class TestMultipartWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_env = None
        cls._server = None
        cls._keys = []
        if LIVE and not INNER:
            cls.bucket = os.environ["S3DLIO_TEST_BUCKET"]
            cls.prefix = f"mpu-py-{uuid.uuid4().hex[:12]}/"
            return
        # Isolated child process: bind its (process-wide) S3 client to the local server.
        cls._saved_env = {n: os.environ.get(n) for n in CLIENT_ENV}
        cls._server = MiniS3()
        for name in CLIENT_ENV:
            os.environ.pop(name, None)
        os.environ["AWS_ACCESS_KEY_ID"] = "test-access-key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret-key"
        os.environ["AWS_ENDPOINT_URL"] = cls._server.url
        os.environ["AWS_REGION"] = "us-east-1"
        cls.bucket = unique_bucket()
        cls.prefix = ""
        try:
            s3dlio.create_bucket(cls.bucket)
        except BaseException:
            cls._restore()
            raise

    @classmethod
    def _restore(cls):
        if cls._saved_env is not None:
            for name, value in cls._saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        if cls._server is not None:
            cls._server.close()
            cls._server = None

    @classmethod
    def tearDownClass(cls):
        try:
            if cls._server is None:  # live: delete what this class wrote
                for key in cls._keys:
                    with contextlib.suppress(Exception):  # best-effort cleanup
                        s3dlio.delete(f"s3://{cls.bucket}/{key}", recursive=False)
            else:
                with contextlib.suppress(Exception):  # best-effort cleanup
                    s3dlio.delete_bucket(cls.bucket)
        finally:
            cls._restore()

    def uri(self, name):
        key = f"{self.prefix}{name}"
        type(self)._keys.append(key)
        return f"s3://{self.bucket}/{key}", key

    def assert_same_bytes(self, actual, expected, what):
        if actual != expected:
            same = min(len(actual), len(expected))
            first = next((i for i in range(same) if actual[i] != expected[i]), same)
            self.fail(
                f"{what}: {len(actual)} bytes, expected {len(expected)}; first difference at {first}"
            )

    def assert_stored(self, uri, key, payload):
        """The object the writer produced is exactly ``payload``: on the server side
        (local server mode) and as read back through the client."""
        if self._server is not None:
            stored = self._server.objects.get((self.bucket, key))
            self.assertIsNotNone(stored, "object was never completed on the server")
            self.assert_same_bytes(stored, payload, "server-side object")
        self.assert_same_bytes(
            bytes(s3dlio.get(uri).memoryview()), payload, "object read back via get()"
        )
        n = len(payload)
        self.assert_same_bytes(
            bytes(s3dlio.get_range(uri, 0, 16).memoryview()), payload[:16], "head range"
        )
        self.assert_same_bytes(
            bytes(s3dlio.get_range(uri, n - 16, 16).memoryview()),
            payload[n - 16 :],
            "tail range",
        )

    def test_reserve_commit_zero_copy(self):
        uri, key = self.uri("ckpt-zero-copy.bin")
        N = (64 << 20) + (1 << 20)  # 65 MiB
        payload = payload_of(N, 1)
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=32 << 20, max_in_flight=16
        )
        mv = w.reserve(N)
        mv[:] = payload
        w.commit(N)
        del mv
        info = w.close()
        self.assertEqual(info["total_bytes"], N)
        self.assertGreaterEqual(info["parts"], 2)
        self.assert_stored(uri, key, payload)

    def test_write_bytes_path(self):
        uri, key = self.uri("ckpt-write.bin")
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=16 << 20, max_in_flight=8
        )
        chunks = [bytes([0x10 + i]) * (4 << 20) for i in range(10)]  # 10 x 4 MiB
        for chunk in chunks:
            w.write(chunk)
        info = w.close()
        self.assertEqual(info["total_bytes"], sum(len(c) for c in chunks))
        self.assertGreaterEqual(info["parts"], 1)
        self.assert_stored(uri, key, b"".join(chunks))

    def test_close_always_has_etag_key(self):
        # Regression test for docs/DESIGN_FFI_BOUNDARY_HARDENING.md §4.3:
        # close()'s docstring promises 'etag': str or None, but the old
        # code only inserted the key when e_tag was Some(...) -- omitting
        # it entirely (not even as None) otherwise, so result['etag']
        # raised KeyError instead of returning None as documented.
        uri, key = self.uri("ckpt-etag-key.bin")
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=16 << 20, max_in_flight=4
        )
        payload = payload_of(1 << 20, 2)
        w.write(payload)
        info = w.close()
        self.assertIn(
            "etag",
            info,
            "'etag' key must always be present (str or None), never omitted",
        )
        self.assert_stored(uri, key, payload)

    def test_reserve_view_is_owned_by_its_exporter_not_by_the_writer(self):
        # Regression test for stage G review G-F1 (and the older
        # docs/DESIGN_FFI_BOUNDARY_HARDENING.md §1.3 / §6.3): reserve() used
        # to return a memoryview via PyMemoryView_FromMemory, which carries no
        # reference to whatever owns the memory. The writer's "self-pin" only
        # covered the reserve()..commit() window, so a view retained past
        # commit()/close()/abort()/del could point at freed memory. The view
        # is now built over an owning exporter (mv.obj); the memory lives as
        # long as any view does and the writer is NOT pinned any more.
        #
        # sys.getrefcount() is CPython-specific; s3dlio only targets CPython.
        uri, key = self.uri("uaf-regression.bin")
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=32 << 20, max_in_flight=4
        )
        baseline = sys.getrefcount(w)
        mv = w.reserve(1024)
        self.assertEqual(
            sys.getrefcount(w),
            baseline,
            "reserve() must not hold an extra reference to the writer",
        )
        self.assertIsInstance(mv.obj, import_module("s3dlio._pymod").ReservedBuffer)
        payload = payload_of(1024, 3)
        mv[:] = payload
        tail = mv[512:]
        w.commit(1024)
        w.close()
        self.assert_stored(uri, key, payload)
        # The views outlive the commit, the close and the writer itself.
        del w
        del mv
        self.assertEqual(bytes(tail), payload[512:])


@unittest.skipIf(
    LIVE or INNER, "live bucket configured (or already the child): tests run in-process"
)
class TestMultipartWriterIsolated(unittest.TestCase):
    """Runs ``TestMultipartWriter`` against the mini server in a fresh child process so the
    S3 client it binds (process-wide, built once) never crosses into other test modules."""

    def test_writer_against_mini_server_in_child_process(self):
        env = {k: v for k, v in os.environ.items() if k not in CLIENT_ENV}
        env["S3DLIO_MPU_INNER"] = "1"
        env["AWS_EC2_METADATA_DISABLED"] = "true"
        env.setdefault("RUST_LOG", "s3dlio=warn,aws_sdk_s3=warn")
        # The child must load the very same s3dlio package as this process.
        package_root = os.path.dirname(
            os.path.dirname(os.path.abspath(s3dlio.__file__))
        )
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (package_root, env.get("PYTHONPATH")) if p
        )
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "-v",
                    "TestMultipartWriter",
                ],
                check=False,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, f"child process failed:\n{out}")
        ran = re.search(r"Ran (\d+) tests?", out)
        self.assertIsNotNone(ran, f"no unittest summary in child output:\n{out}")
        self.assertGreaterEqual(int(ran.group(1)), 4, out)
        self.assertNotIn("skipped", out, f"child tests were skipped:\n{out}")


if __name__ == "__main__":
    unittest.main()
