"""Caller-buffer GET (get_into_async), typed HEAD (head_async) and object_capabilities
against the INSTALLED wheel (v0.10.0 plan, sections 4.1, 4.2, 8 and 9; umbrella issue #264;
also the #250 reproduction check and the #233 / mkdir_async exit-safety checks).

What this module needs
----------------------
Nothing from the network and no environment variable.  The pure parts (capabilities,
synchronous rejections, file:// and direct:// refusals, the #250 check) run in the pytest
process.  Everything that reads an object runs in a FRESH child interpreter (this same file,
executed as a script) because s3dlio latches its S3 client and range settings once per
process: the child is started with AWS_ENDPOINT_URL pointing at a small S3-compatible server
that THIS module runs in the pytest process (``FakeS3``: HEAD, GET, Range, If-Match, ETags,
404/412/416, plus a "stall" switch that holds a response body until released).  The server
records every request, so the tests also assert what was NOT sent (no request at all for a
refused buffer, ``If-Match`` on every ranged GET).  No cloud and no credentials are used; the
child gets fake ones.  The only skips are: numpy missing (the numpy tests), pytest-asyncio
missing (the tests that need it), and a wheel built without native S3 (every child test).

Run (the in-repo convention; one wheel per interpreter):

    S3DLIO_TEST_DIR=<dir on a real filesystem> uv run --no-project --isolated --with <wheel> \
        --with pytest --with pytest-asyncio --with numpy \
        python -m pytest python/tests/test_caller_buffer_get_into.py -v

``S3DLIO_TEST_DIR`` is optional (where the children's scratch directories go).  Every child
runs under a hard timeout; a hang fails the test (the child is asked to stop with SIGTERM,
twice, and a child that ignores both is reported, never force-killed).

Layout: the ``test_*`` functions are the pytest tests; the ``@scenario`` functions in the
second half are what the children run.
"""

import array
import asyncio
import collections
import ctypes
import gc
import hashlib
import importlib.util
import json
import mmap
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import s3dlio

MIB = 1024 * 1024

BUCKET = "cbtest"
FILL = 0xEE  # what a destination holds before a read
CANARY_LEN = 64  # bytes appended after a destination to catch a write past expected_len
AFTER = 0x5A  # what a test writes into a buffer after the awaitable ended

# Object sizes of the round trip: empty, one byte, tiny, just over a page, just over 1 MiB,
# several MiB.  All stay below the 32 MiB default range threshold, so they take the single
# unranged GET unless a test lowers the threshold in the child's environment.
SIZES = (0, 1, 16, 4096 + 7, MIB + 7, 4 * MIB + 3)

STALL_KEY = "stall-object"
STALL_SIZE = 4 * MIB
STALL_SEED = 9
STALL_AFTER = 64 * 1024  # bytes served before the body stalls
HEAD_HOLD_KEY = "head-hold-object"
HEAD_HOLD_SIZE = 12345

HOLD_SECONDS = 25  # longest a server thread waits for a release
CHILD_TIMEOUT = 90

REFUSALS = (BufferError, TypeError, ValueError)

S3_READY = (
    hasattr(s3dlio, "object_capabilities")
    and s3dlio.object_capabilities(f"s3://{BUCKET}/key")["get_into"]
)

needs_asyncio_plugin = pytest.mark.skipif(
    importlib.util.find_spec("pytest_asyncio") is None,
    reason="pytest-asyncio is not installed",
)


def pattern(size, seed=0):
    """`size` deterministic, position-dependent bytes (different between seeds)."""
    return random.Random(seed).randbytes(size)


def object_path(key):
    return f"/{BUCKET}/{key}"


def uri_of(key):
    return f"s3://{BUCKET}/{key}"


def resizable(buf):
    """True when no export is held on the bytearray / array (a resize succeeds)."""
    try:
        buf.append(0)
        del buf[-1:]
        return True
    except BufferError:
        return False


# ---------------------------------------------------------------------------------------------
# A small S3-compatible server (runs in the pytest process; the children talk to it).
# ---------------------------------------------------------------------------------------------

Req = collections.namedtuple("Req", "method path range if_match")

LAST_MODIFIED = "Mon, 01 Jan 2024 00:00:00 GMT"


def same_etag(left, right):
    def normal(value):
        value = value.strip()
        if value.startswith("W/"):
            value = value[2:]
        return value.strip('"')

    return normal(left) == normal(right)


class FakeObject:
    def __init__(self, body, version_id=None):
        self.body = bytes(body)
        self.etag = '"' + hashlib.sha256(self.body).hexdigest()[:32] + '"'
        self.version_id = version_id


class FakeS3:
    """HEAD / GET (Range, If-Match) over path-style URLs, a request log, and two switches:

    ``stall[path] = n``      GETs of `path` send their headers and `n` body bytes, then wait
                             for ``release`` before sending the rest;
    ``head_hold``            HEADs of the listed paths wait for ``release`` before answering.

    ``GET /__ctl/release`` sets ``release`` (it stays set); ``GET /__ctl/stalled`` returns,
    as JSON, how many responses per path have reached their wait so far.
    """

    def __init__(self):
        self.objects = {}
        self.stall = {}
        self.head_hold = set()
        self.stalled = {}
        self.requests = []
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self))
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    def put(self, key, body, version_id=None):
        obj = FakeObject(body, version_id)
        with self.lock:
            self.objects[object_path(key)] = obj
        return obj

    def hold(self, path):
        """Called by a handler thread that reached its wait."""
        with self.lock:
            self.stalled[path] = self.stalled.get(path, 0) + 1
        self.release.wait(HOLD_SECONDS)

    def requests_for(self, key, method=None):
        path = object_path(key)
        with self.lock:
            return [
                r
                for r in self.requests
                if r.path == path and (method is None or r.method == method)
            ]

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(10)


def make_handler(fake):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            self.serve(head=False)

        def do_HEAD(self):
            self.serve(head=True)

        def serve(self, head):
            try:
                self.serve_request(head)
            except OSError:
                # the client went away (a cancelled read, an exiting process)
                self.close_connection = True

        def reply(self, status, body, headers=None, head=False, length=None):
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            size = len(body) if length is None else length
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if body and not head:
                self.wfile.write(body)

        def error(self, status, code, message, headers=None):
            body = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                f"<Error><Code>{code}</Code><Message>{message}</Message>"
                "<RequestId>0</RequestId></Error>"
            ).encode()
            merged = {"Content-Type": "application/xml"}
            merged.update(headers or {})
            self.reply(status, body, merged)

        def control(self, path):
            if path == "/__ctl/release":
                fake.release.set()
                self.reply(200, b"ok", {"Content-Type": "text/plain"})
            elif path == "/__ctl/stalled":
                with fake.lock:
                    body = json.dumps(fake.stalled).encode()
                self.reply(200, body, {"Content-Type": "application/json"})
            else:
                self.reply(404, b"", head=True, length=0)

        def serve_request(self, head):
            path = urllib.parse.unquote(self.path.split("?", 1)[0])
            if path.startswith("/__ctl/"):
                self.control(path)
                return
            range_header = self.headers.get("Range")
            if_match = self.headers.get("If-Match")
            with fake.lock:
                fake.requests.append(
                    Req("HEAD" if head else "GET", path, range_header, if_match)
                )
                obj = fake.objects.get(path)
                stall_after = fake.stall.get(path)
                hold_head = head and path in fake.head_hold
            if hold_head:
                fake.hold(path)
            if obj is None:
                if head:
                    self.reply(404, b"", head=True, length=0)
                else:
                    self.error(404, "NoSuchKey", "The specified key does not exist.")
                return
            headers = {
                "ETag": obj.etag,
                "Accept-Ranges": "bytes",
                "Content-Type": "application/octet-stream",
                "Last-Modified": LAST_MODIFIED,
            }
            if obj.version_id is not None:
                headers["x-amz-version-id"] = obj.version_id
            if head:
                self.reply(200, b"", headers, head=True, length=len(obj.body))
                return
            if (
                if_match is not None
                and if_match.strip() != "*"
                and not same_etag(if_match, obj.etag)
            ):
                self.error(412, "PreconditionFailed", "The condition did not hold.")
                return
            body, status = obj.body, 200
            if range_header is not None:
                total = len(obj.body)
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())
                if match is None:
                    self.error(416, "InvalidRange", "Unsupported range.")
                    return
                first = int(match.group(1))
                last = int(match.group(2)) if match.group(2) else total - 1
                if total == 0 or first >= total:
                    self.error(
                        416,
                        "InvalidRange",
                        "The requested range is not satisfiable.",
                        {"Content-Range": f"bytes */{total}"},
                    )
                    return
                last = min(last, total - 1)
                if last < first:
                    self.error(416, "InvalidRange", "The range is empty.")
                    return
                body = obj.body[first : last + 1]
                status = 206
                headers["Content-Range"] = f"bytes {first}-{last}/{total}"
            if stall_after is not None and 0 < stall_after < len(body):
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[:stall_after])
                self.wfile.flush()
                fake.hold(path)
                self.wfile.write(body[stall_after:])
            else:
                self.reply(status, body, headers)

    return Handler


@pytest.fixture
def fake_s3():
    if not S3_READY:
        pytest.skip(
            "this wheel has no native S3 caller-buffer GET (see object_capabilities)"
        )
    fake = FakeS3()
    try:
        yield fake
    finally:
        fake.close()


def populate(fake):
    """The objects most scenarios read: one per size, key `obj-<size>`."""
    for size in SIZES:
        fake.put(f"obj-{size}", pattern(size, size % 251))


# ---------------------------------------------------------------------------------------------
# Running a scenario in a fresh child interpreter.
# ---------------------------------------------------------------------------------------------

ChildResult = collections.namedtuple("ChildResult", "returncode stdout stderr")

BAD_STDERR = (
    "Fatal Python error",
    "Future exception was never retrieved",
    "Task exception was never retrieved",
    "panicked at",
    "Segmentation fault",
)


def child_env(fake, extra=None):
    env = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith(("AWS_", "S3DLIO_")) or upper in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        ):
            continue
        env[name] = value
    env.update(
        {
            "AWS_ENDPOINT_URL": fake.endpoint,
            "AWS_ACCESS_KEY_ID": "fake-access-key",
            "AWS_SECRET_ACCESS_KEY": "fake-secret-key",
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CONFIG_FILE": os.devnull,
            "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
            "NO_PROXY": "127.0.0.1",
            "S3DLIO_LOAD_DOTENV": "0",
            "S3DLIO_MAX_RETRY_ATTEMPTS": "2",
            "S3DLIO_ENABLE_RANGE_OPTIMIZATION": "1",
            "CB_FAKE_ENDPOINT": fake.endpoint,
        }
    )
    env.update(extra or {})
    return env


def run_child(fake, name, params=None, extra_env=None, timeout=CHILD_TIMEOUT):
    """Run scenario `name` in a fresh interpreter; fail (not hang) after `timeout` seconds."""
    work = tempfile.mkdtemp(
        prefix="cbget-", dir=os.environ.get("S3DLIO_TEST_DIR") or None
    )
    command = [
        sys.executable,
        os.path.abspath(__file__),
        name,
        json.dumps(params or {}),
        work,
    ]
    try:
        proc = subprocess.Popen(
            command,
            env=child_env(fake, extra_env),
            cwd=work,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            fake.release.set()
            proc.terminate()
            try:
                out, err = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                # a second SIGTERM; a child that ignores both is reported, not force-killed
                proc.terminate()
                try:
                    out, err = proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    pytest.fail(
                        f"scenario {name!r} did not finish within {timeout}s and did not "
                        f"exit after two SIGTERMs (pid {proc.pid} left running)"
                    )
            pytest.fail(
                f"scenario {name!r} did not finish within {timeout}s (hang)\n"
                f"stdout:\n{out[-2000:]}\nstderr:\n{err[-3000:]}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return ChildResult(proc.returncode, out, err)


def run_ok(fake, name, params=None, extra_env=None, timeout=CHILD_TIMEOUT):
    result = run_child(fake, name, params, extra_env, timeout)
    detail = f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-4000:]}"
    assert result.returncode == 0, (
        f"scenario {name!r} failed (rc={result.returncode})\n{detail}"
    )
    assert "SCENARIO-OK" in result.stdout, f"scenario {name!r} never finished\n{detail}"
    for bad in BAD_STDERR:
        assert bad not in result.stderr, f"scenario {name!r} logged {bad!r}\n{detail}"
    return result


# =============================================================================================
# In-process tests (no S3 client is ever built here)
# =============================================================================================


def test_the_public_surface_exists_with_the_documented_hierarchy():
    for name in (
        "object_capabilities",
        "get_into_async",
        "head_async",
        "ObjectMetadata",
        "ObjectNotFoundError",
        "LengthMismatchError",
        "UnsupportedOperationError",
    ):
        assert hasattr(s3dlio, name), f"s3dlio.{name} is missing"
    assert issubclass(s3dlio.ObjectNotFoundError, FileNotFoundError)
    assert issubclass(s3dlio.LengthMismatchError, ValueError)
    assert issubclass(s3dlio.UnsupportedOperationError, NotImplementedError)
    assert isinstance(s3dlio.ObjectMetadata, type)
    # the three are raisable and catchable by their built-in bases
    for exc_type, base in (
        (s3dlio.ObjectNotFoundError, FileNotFoundError),
        (s3dlio.LengthMismatchError, ValueError),
        (s3dlio.UnsupportedOperationError, NotImplementedError),
    ):
        with pytest.raises(base):
            raise exc_type("raised for the hierarchy check")


@pytest.mark.parametrize(
    "uri,provider",
    [
        ("s3://bucket/key", "s3"),
        ("file:///tmp/some/file", "file"),
        ("direct:///tmp/some/file", "direct"),
        ("az://account/container/key", "az"),
        ("gs://bucket/key", "gs"),
        ("gcs://bucket/key", "gs"),
        ("nosuchscheme://bucket/key", "unknown"),
    ],
)
def test_object_capabilities_shape_and_provider(uri, provider):
    caps = s3dlio.object_capabilities(uri)
    assert isinstance(caps, dict)
    assert set(caps) == {"provider", "get_into", "head"}
    assert caps["provider"] == provider
    assert isinstance(caps["get_into"], bool) and isinstance(caps["head"], bool)
    if provider in ("file", "direct", "unknown"):
        assert caps["get_into"] is False and caps["head"] is False
    if provider == "s3":
        assert caps["get_into"] is True and caps["head"] is True


def test_object_capabilities_is_pure(monkeypatch):
    """No credentials, endpoint or configuration are needed (nothing reaches a provider)."""
    uris = ("s3://bucket/key", "az://account/container/key", "gs://bucket/key")
    before = {u: s3dlio.object_capabilities(u) for u in uris}
    for name in list(os.environ):
        if name.upper().startswith(("AWS_", "AZURE_", "GOOGLE_", "GCS_", "S3DLIO_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:1")  # nothing listens here
    after = {u: s3dlio.object_capabilities(u) for u in uris}
    assert after == before


@pytest.mark.parametrize(
    "uri",
    [
        "az://account//key",  # empty container (#224)
        "s3://host:09000/bucket/key",  # leading zero in the port
    ],
)
def test_malformed_uri_is_a_value_error_before_anything_else(uri):
    buf = bytearray(8)
    with pytest.raises(ValueError):
        s3dlio.object_capabilities(uri)
    with pytest.raises(ValueError):
        s3dlio.get_into_async(uri, buf, expected_len=8)
    with pytest.raises(ValueError):
        s3dlio.head_async(uri)
    assert resizable(buf)


@pytest.mark.parametrize(
    "uri", ["file:///tmp/cb-unsupported", "direct:///tmp/cb-unsupported"]
)
def test_file_and_direct_are_unsupported_synchronously(uri):
    """Raised by the call itself: no event loop is running, so nothing was awaited."""
    buf = bytearray(8)
    with pytest.raises(s3dlio.UnsupportedOperationError) as info:
        s3dlio.get_into_async(uri, buf, expected_len=8)
    assert isinstance(info.value, NotImplementedError)
    with pytest.raises(s3dlio.UnsupportedOperationError):
        s3dlio.head_async(uri)
    assert resizable(buf), "the refusal left the buffer exported"


@needs_asyncio_plugin
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri", ["file:///tmp/cb-unsupported", "direct:///tmp/cb-unsupported"]
)
async def test_file_and_direct_are_unsupported_inside_a_running_loop_too(uri):
    buf = bytearray(8)
    with pytest.raises(s3dlio.UnsupportedOperationError):
        s3dlio.get_into_async(uri, buf, expected_len=8)
    with pytest.raises(s3dlio.UnsupportedOperationError):
        s3dlio.head_async(uri)
    assert resizable(buf)


@pytest.mark.parametrize(
    "bad,error",
    [
        (True, TypeError),
        (False, TypeError),
        (1.5, TypeError),
        (8.0, TypeError),
        ("8", TypeError),
        (None, TypeError),
        (-1, ValueError),
        (-(2**70), ValueError),
        (2**70, ValueError),
        (2**300, ValueError),
    ],
)
def test_bad_expected_len_is_rejected_synchronously(bad, error):
    buf = bytearray(8)
    with pytest.raises(error):
        s3dlio.get_into_async("s3://bucket/key", buf, expected_len=bad)
    assert resizable(buf), "a rejected expected_len must never export the buffer"


def test_expected_len_is_keyword_only_and_required():
    buf = bytearray(8)
    with pytest.raises(TypeError):
        s3dlio.get_into_async("s3://bucket/key", buf, 8)
    with pytest.raises(TypeError):
        s3dlio.get_into_async("s3://bucket/key", buf)
    assert resizable(buf)


# =============================================================================================
# Child-process tests against the local S3-compatible server
# =============================================================================================


def test_every_size_round_trips_exactly_and_leaves_the_canary(fake_s3):
    populate(fake_s3)
    run_ok(fake_s3, "roundtrip_sizes")
    # one unranged GET per size (nothing near the range threshold)
    for size in SIZES:
        gets = fake_s3.requests_for(f"obj-{size}", "GET")
        assert len(gets) == 1, (size, gets)
        assert gets[0].range is None, (size, gets)


def test_accepted_destination_types(fake_s3):
    populate(fake_s3)
    run_ok(fake_s3, "accepted_buffer_types")


def test_numpy_uint8_and_int8_destinations(fake_s3):
    pytest.importorskip("numpy")
    populate(fake_s3)
    run_ok(fake_s3, "numpy_destinations")


def test_refused_buffers_are_refused_before_any_io_and_released(fake_s3):
    populate(fake_s3)
    run_ok(fake_s3, "refused_buffers")
    # the only requests are those of the good read at the end of the scenario
    others = [r for r in fake_s3.requests if r.path != object_path(f"obj-{4096 + 7}")]
    assert others == [], f"a refused destination reached the network: {others}"
    assert len(fake_s3.requests_for(f"obj-{4096 + 7}", "GET")) == 1, (
        "a refused destination sent a GET; only the final good read may"
    )


def test_numpy_refusals_make_no_request(fake_s3):
    pytest.importorskip("numpy")
    populate(fake_s3)
    run_ok(fake_s3, "numpy_refusals")
    assert fake_s3.requests == []


def test_a_wrong_expected_length_is_a_length_mismatch_with_an_untouched_buffer(
    fake_s3,
):
    fake_s3.put("mismatch-1000", pattern(1000, 3))
    run_ok(fake_s3, "length_mismatch")


def test_missing_object_is_not_found_and_head_reports_none(fake_s3):
    populate(fake_s3)
    run_ok(fake_s3, "not_found")


def test_head_metadata_is_frozen_hashable_and_python_style(fake_s3):
    plain = fake_s3.put("head-plain", pattern(777, 1))
    versioned = fake_s3.put("head-versioned", pattern(888, 2), version_id="ver-0001")
    fake_s3.put("head-empty", b"")
    run_ok(
        fake_s3,
        "head_metadata",
        {"plain_etag": plain.etag, "versioned_etag": versioned.etag},
    )


def test_fan_out_pins_every_range_to_the_head_etag(fake_s3):
    size = 3 * MIB + 7
    obj = fake_s3.put("fanout-object", pattern(size, 5))
    run_ok(
        fake_s3,
        "fanout_route",
        {"size": size},
        extra_env={"S3DLIO_RANGE_THRESHOLD_MB": "1"},
    )
    requests = fake_s3.requests_for("fanout-object")
    assert any(r.method == "HEAD" for r in requests), "the fan-out never sent its HEAD"
    gets = [r for r in requests if r.method == "GET"]
    assert len(gets) >= 3, f"expected several ranged GETs, saw {gets}"
    for r in gets:
        assert r.range is not None, f"an unranged GET on the fan-out route: {r}"
        assert r.if_match is not None and same_etag(r.if_match, obj.etag), (
            f"a range without the pinned If-Match: {r}"
        )
    assert len({r.range for r in gets}) == len(gets), "a range was requested twice"


@pytest.mark.parametrize("key", ["a%2Fb", "100%", "x@y", "k=v", "p/q/r", "name."])
def test_keys_are_literal(fake_s3, key):
    fake_s3.put(key, pattern(33, 7))
    run_ok(fake_s3, "literal_key", {"key": key})


def test_concurrent_reads_into_disjoint_slices_of_one_buffer(fake_s3):
    for i in range(4):
        fake_s3.put(f"slice-{i}", pattern(100_003, 20 + i))
    run_ok(fake_s3, "disjoint_slices")


def test_failing_to_start_outside_a_loop_releases_the_buffer(fake_s3):
    populate(fake_s3)
    run_ok(fake_s3, "start_failure_without_loop")


STALL_VARIANTS = [
    "cancel_pending",
    "cancel_before_first_poll",
    "cancel_after_completion",
    "wait_for_timeout",
    "asyncio_timeout",
    "task_cancel_repeated",
    "task_cancel_tight",
    "stall_released_completes",
]


def stalled_server(fake):
    fake.put(STALL_KEY, pattern(STALL_SIZE, STALL_SEED))
    fake.stall[object_path(STALL_KEY)] = STALL_AFTER


@pytest.mark.parametrize("variant", STALL_VARIANTS)
def test_cancellation_matrix_for_a_blocked_read(fake_s3, variant):
    """Section 9 matrix: Future.cancel(), cancel after completion, wait_for, asyncio.timeout,
    repeated Task.cancel.  In every case the buffer is exported while the read runs, is free
    when (never before) the awaitable has finished, nothing writes into it afterwards, and
    the loop logs no unretrieved exception."""
    stalled_server(fake_s3)
    run_ok(fake_s3, "cancellation", {"variant": variant})


HEAD_VARIANTS = ["cancel", "wait_for", "asyncio_timeout", "released_completes"]


@pytest.mark.parametrize("variant", HEAD_VARIANTS)
def test_cancellation_matrix_for_a_blocked_head(fake_s3, variant):
    fake_s3.put(HEAD_HOLD_KEY, pattern(HEAD_HOLD_SIZE, 4))
    fake_s3.head_hold.add(object_path(HEAD_HOLD_KEY))
    run_ok(fake_s3, "head_cancellation", {"variant": variant})


# ---- interpreter shutdown ---------------------------------------------------------------------

EXIT_REPEATS = 3
EXIT_TIMEOUT = 60


def run_exit_scenario(fake, name, repeats=EXIT_REPEATS):
    """Each run must exit cleanly (rc 0, no fatal error, no abort) within a hard timeout."""
    for i in range(repeats):
        started = time.monotonic()
        result = run_ok(fake, name, timeout=EXIT_TIMEOUT)
        assert result.returncode == 0, f"exit {i}"
        assert time.monotonic() - started < EXIT_TIMEOUT, f"exit {i} was slow"


def test_exit_with_a_get_into_in_flight_is_clean_and_repeatable(fake_s3):
    stalled_server(fake_s3)
    run_exit_scenario(fake_s3, "exit_in_flight_get")


def test_exit_with_a_head_in_flight_is_clean_and_repeatable(fake_s3):
    fake_s3.put(HEAD_HOLD_KEY, pattern(HEAD_HOLD_SIZE, 4))
    fake_s3.head_hold.add(object_path(HEAD_HOLD_KEY))
    run_exit_scenario(fake_s3, "exit_in_flight_head")


def test_exit_right_after_a_get_into_completes_is_clean_and_repeatable(fake_s3):
    populate(fake_s3)
    run_exit_scenario(fake_s3, "exit_after_get")


def test_exit_with_mkdir_async_in_flight_is_clean_and_repeatable(fake_s3):
    run_exit_scenario(fake_s3, "exit_in_flight_mkdir", repeats=5)


def test_exit_right_after_mkdir_async_completes_is_clean_and_repeatable(fake_s3):
    run_exit_scenario(fake_s3, "exit_after_mkdir", repeats=5)


# =============================================================================================
# #250 reproduction check, #233 (these run under pytest-asyncio in the pytest process)
# =============================================================================================


class CustomBytes:
    def __init__(self, data):
        self._data = data

    def __bytes__(self):
        return self._data

    def __len__(self):
        return len(self._data)


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_issue_250_reproduction_check_async_calls_under_pytest_asyncio(tmp_path):
    """#250 REPRODUCTION CHECK.  The issue reports "no running event loop" from the async
    buffer-protocol tests of ``test_buffer_protocol_support.py``.  This runs the same calls
    (put_bytes_async with bytes, bytearray, memoryview and a ``__bytes__`` object, then
    mkdir_async and the new async entry points) under pytest-asyncio against the INSTALLED
    wheel and RECORDS whether the error appears.  The asserted, correct behaviour is that it
    does not: every await succeeds in the running loop and the data round-trips.  If this
    ever fails with "no running event loop", the report is reproduced; if it passes, the
    symptom does not reproduce with pytest-asyncio present and the issue account needs
    correcting rather than a fix being invented."""
    loop = asyncio.get_running_loop()
    assert loop.is_running()
    payloads = {
        "bytes": (b"Hello, async bytes!", b"Hello, async bytes!"),
        "bytearray": (bytearray(b"Hello, bytearray!"), b"Hello, bytearray!"),
        "memoryview": (memoryview(b"Hello, memoryview!"), b"Hello, memoryview!"),
        "custom": (CustomBytes(b"Hello, custom!"), b"Hello, custom!"),
    }
    reproduced = []
    for label, (payload, expected) in payloads.items():
        uri = f"file://{tmp_path}/{label}.bin"
        try:
            await s3dlio.put_bytes_async(uri, payload)
        except RuntimeError as exc:
            reproduced.append((label, str(exc)))
            continue
        assert bytes(s3dlio.get(uri)) == expected
    try:
        made = await s3dlio.mkdir_async(f"file://{tmp_path}/made/sub/")
    except RuntimeError as exc:
        reproduced.append(("mkdir_async", str(exc)))
        made = None
    assert made is None
    # the new entry points refuse file:// by raising, not by needing a loop
    with pytest.raises(s3dlio.UnsupportedOperationError):
        s3dlio.get_into_async(f"file://{tmp_path}/x.bin", bytearray(4), expected_len=4)
    observed = [item for item in reproduced if "no running event loop" in item[1]]
    print(f"\n#250 observed 'no running event loop': {observed or 'no (does not reproduce)'}")
    assert not reproduced, f"#250 reproduces: {reproduced}"


def test_issue_250_the_real_async_tests_pass_under_pytest_asyncio():
    """#250 REPRODUCTION CHECK on the actual file: run the async tests of
    ``test_buffer_protocol_support.py`` in a fresh pytest and require that none of them fails
    with "no running event loop" (and that the run is green)."""
    pytest.importorskip("pytest_asyncio")
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "test_buffer_protocol_support.py")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "async",
            target,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=here,
        check=False,
    )
    output = result.stdout + result.stderr
    print(f"\n#250: reproduces = {'no running event loop' in output}")
    assert "no running event loop" not in output, output[-3000:]
    assert result.returncode == 0, output[-3000:]


@needs_asyncio_plugin
@pytest.mark.asyncio
async def test_mkdir_async_returns_none_and_is_idempotent(tmp_path):
    """#233 through the installed wheel."""
    uri = f"file://{tmp_path}/a/b/c/"
    assert await s3dlio.mkdir_async(uri) is None
    assert await s3dlio.mkdir_async(uri) is None
    assert (tmp_path / "a" / "b" / "c").is_dir()


# =============================================================================================
# Child scenarios.  Everything below runs in the child interpreter (``python <this file>``).
# The child has no pytest involvement: a failed `assert` ends it with a traceback and a
# non-zero exit code, which the parent test reports.
# =============================================================================================

SCENARIOS = {}
KEEP = []  # references a shutdown scenario keeps alive until interpreter exit


def scenario(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


class Ctl:
    """The child's handle on the server in the parent: control calls and the scratch dir."""

    def __init__(self, endpoint, work):
        self.endpoint = endpoint
        self.work = work
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path):
        with self.opener.open(self.endpoint + path, timeout=10) as response:
            return response.read()

    def release(self):
        self.get("/__ctl/release")

    def stalled(self, key):
        return json.loads(self.get("/__ctl/stalled")).get(object_path(key), 0)


async def wait_for_new_stall(ctl, key, baseline, seconds=20):
    deadline = time.monotonic() + seconds
    while ctl.stalled(key) <= baseline:
        assert time.monotonic() < deadline, f"the server never reached its stall for {key}"
        await asyncio.sleep(0.02)


async def failure_of(awaitable):
    try:
        await awaitable
    except Exception as exc:  # noqa: BLE001 - the caller checks the type
        return exc
    raise AssertionError("the awaitable succeeded but was expected to fail")


def expect_refused(dst, expected_len, uri):
    """The call itself must raise (synchronously) one of the refusal errors."""
    try:
        fut = s3dlio.get_into_async(uri, dst, expected_len=expected_len)
    except REFUSALS as exc:
        return exc
    fut.cancel()
    raise AssertionError(f"{type(dst).__name__} destination was not refused")


@scenario
def roundtrip_sizes(ctl, params):
    async def main():
        for size in SIZES:
            expected = pattern(size, size % 251)
            dst = bytearray([FILL]) * (size + CANARY_LEN)
            got = await s3dlio.get_into_async(
                uri_of(f"obj-{size}"), dst, expected_len=size
            )
            assert type(got) is int and got == size, (size, got)
            assert bytes(dst[:size]) == expected, f"content differs at size {size}"
            assert dst[size:].count(FILL) == CANARY_LEN, f"write past expected_len ({size})"
            assert resizable(dst), f"the export outlived the await at size {size}"

    asyncio.run(main())


@scenario
def accepted_buffer_types(ctl, params):
    size = 4096 + 7
    expected = pattern(size, size % 251)
    uri = uri_of(f"obj-{size}")

    def make_cases():
        return {
            "bytearray": bytearray(size),
            "memoryview(bytearray)": memoryview(bytearray(size)),
            "array('B')": array.array("B", bytes(size)),
            "array('b')": array.array("b", bytes(size)),
            "ctypes c_ubyte array": (ctypes.c_ubyte * size)(),
            "ctypes c_char array": (ctypes.c_char * size)(),
            "mmap": mmap.mmap(-1, size),
        }

    async def main():
        for label, dst in make_cases().items():
            got = await s3dlio.get_into_async(uri, dst, expected_len=size)
            assert got == size, label
            with memoryview(dst) as view:
                data = view.tobytes()
            assert data == expected, f"{label}: wrong content"
            if isinstance(dst, (bytearray, array.array)):
                assert resizable(dst), f"{label}: export outlived the await"
            if isinstance(dst, mmap.mmap):
                dst.close()  # BufferError here = the export was not released

        # a slice of a larger buffer: only the slice is written
        base = bytearray([FILL]) * (size + 16)
        with memoryview(base) as whole:
            with whole[8 : 8 + size] as piece:
                await s3dlio.get_into_async(uri, piece, expected_len=size)
        assert bytes(base[8 : 8 + size]) == expected
        assert base[:8].count(FILL) == 8 and base[8 + size :].count(FILL) == 8
        assert resizable(base)

        # address identity: the object passed in is the one filled, in place
        ctype_dst = (ctypes.c_ubyte * size)()
        address = ctypes.addressof(ctype_dst)
        await s3dlio.get_into_async(uri, ctype_dst, expected_len=size)
        assert ctypes.addressof(ctype_dst) == address
        assert bytes(ctype_dst) == expected

    asyncio.run(main())


@scenario
def numpy_destinations(ctl, params):
    import numpy as np

    size = 4096 + 7
    expected = pattern(size, size % 251)
    uri = uri_of(f"obj-{size}")

    async def main():
        for dtype in (np.uint8, np.int8):
            dst = np.zeros(size, dtype=dtype)
            got = await s3dlio.get_into_async(uri, dst, expected_len=size)
            assert got == size
            assert dst.tobytes() == expected, f"numpy {dtype.__name__}: wrong content"
        # a writable view of part of a larger array
        big = np.full(size + 32, FILL, dtype=np.uint8)
        part = big[16 : 16 + size]
        await s3dlio.get_into_async(uri, part, expected_len=size)
        assert part.tobytes() == expected
        assert big[:16].tolist() == [FILL] * 16
        assert big[16 + size :].tolist() == [FILL] * 16

    asyncio.run(main())


@scenario
def refused_buffers(ctl, params):
    size = 100
    uri = uri_of(f"obj-{4096 + 7}")

    async def main():
        resized = []
        cases = [
            ("bytes (read-only)", bytes(size), size),
            ("memoryview of bytes", memoryview(bytes(size)), size),
            ("non-contiguous memoryview", memoryview(bytearray(2 * size))[::2], size),
            ("array('i')", array.array("i", bytes(4 * size)), size),
            ("array('H')", array.array("H", bytes(2 * size)), size),
            ("array('d')", array.array("d", bytes(8 * size)), size),
            ("ctypes c_int array", (ctypes.c_int * size)(), size),
            ("ctypes c_bool array", (ctypes.c_bool * size)(), size),
            ("str", "x" * size, size),
            ("None", None, size),
            ("int", 5, size),
            ("list", [0] * size, size),
            ("bytearray smaller than expected_len", bytearray(size - 1), size),
        ]
        for label, dst, expected_len in cases:
            expect_refused(dst, expected_len, uri)
            if isinstance(dst, (bytearray, array.array)):
                resized.append(label)
                assert resizable(dst), f"{label}: still exported after the refusal"
        assert "array('i')" in resized and "array('H')" in resized

        # a refusal must not poison the next call: a good buffer right afterwards works
        good = bytearray(4096 + 7)
        got = await s3dlio.get_into_async(uri, good, expected_len=len(good))
        assert got == len(good)

    asyncio.run(main())


@scenario
def numpy_refusals(ctl, params):
    import numpy as np

    size = 64
    uri = uri_of(f"obj-{4096 + 7}")

    async def main():
        read_only = np.zeros(size, dtype=np.uint8)
        read_only.flags.writeable = False
        for dst in (
            np.zeros(size, dtype=np.int32),
            np.zeros(size, dtype=np.float32),
            np.zeros(2 * size, dtype=np.uint8)[::2],
            read_only,
        ):
            expect_refused(dst, size, uri)
        expect_refused(np.zeros(size, dtype=np.uint8), size + 1, uri)

    asyncio.run(main())


@scenario
def length_mismatch(ctl, params):
    uri = uri_of("mismatch-1000")
    actual = 1000

    async def main():
        for expected_len in (actual - 1, actual + 1, 0, 1):
            dst = bytearray([FILL]) * 1100
            exc = await failure_of(
                s3dlio.get_into_async(uri, dst, expected_len=expected_len)
            )
            assert isinstance(exc, s3dlio.LengthMismatchError), repr(exc)
            assert isinstance(exc, ValueError)
            assert exc.expected == expected_len, (exc.expected, expected_len)
            assert exc.actual == actual, (exc.actual, actual)
            assert dst.count(FILL) == len(dst), f"expected_len={expected_len}: written"
            assert resizable(dst)
        # and the right length still works afterwards
        dst = bytearray(actual)
        assert await s3dlio.get_into_async(uri, dst, expected_len=actual) == actual
        assert bytes(dst) == pattern(actual, 3)

    asyncio.run(main())


@scenario
def not_found(ctl, params):
    uri = uri_of("absent-key")

    async def main():
        dst = bytearray([FILL]) * 64
        exc = await failure_of(s3dlio.get_into_async(uri, dst, expected_len=64))
        assert isinstance(exc, s3dlio.ObjectNotFoundError), repr(exc)
        assert isinstance(exc, FileNotFoundError)
        assert dst.count(FILL) == len(dst), "a missing object wrote into the buffer"
        assert resizable(dst)
        assert await s3dlio.head_async(uri) is None
        # an existing object is still found afterwards
        meta = await s3dlio.head_async(uri_of("obj-16"))
        assert meta is not None and meta.size == 16

    asyncio.run(main())


REPR_TEXT = (
    r"ObjectMetadata\(size=\d+, etag=('[^']*'|\"[^\"]*\"|None), "
    r"version_id=('[^']*'|None)\)"
)


@scenario
def head_metadata(ctl, params):
    async def main():
        plain = await s3dlio.head_async(uri_of("head-plain"))
        again = await s3dlio.head_async(uri_of("head-plain"))
        versioned = await s3dlio.head_async(uri_of("head-versioned"))
        empty = await s3dlio.head_async(uri_of("head-empty"))
        assert type(plain) is s3dlio.ObjectMetadata
        assert plain.size == 777 and versioned.size == 888 and empty.size == 0
        assert plain.etag is not None
        assert plain.etag.strip('"') == params["plain_etag"].strip('"')
        assert versioned.etag.strip('"') == params["versioned_etag"].strip('"')
        assert plain.version_id is None
        assert versioned.version_id == "ver-0001"
        # frozen
        for name, value in (("size", 1), ("etag", "x"), ("version_id", "v")):
            try:
                setattr(plain, name, value)
            except (AttributeError, TypeError):
                pass
            else:
                raise AssertionError(f"ObjectMetadata.{name} could be assigned")
        # value semantics
        assert plain == again and hash(plain) == hash(again)
        assert plain != versioned
        assert len({plain, again, versioned}) == 2
        assert {plain: "x"}[again] == "x"
        # Python-style repr (quotes and None the Python way)
        for meta in (plain, versioned, empty):
            text = repr(meta)
            assert re.fullmatch(REPR_TEXT, text), text
        assert repr(plain).endswith("version_id=None)")
        assert "version_id='ver-0001'" in repr(versioned)
        # absence is None, not an error
        assert await s3dlio.head_async(uri_of("head-nope")) is None

    asyncio.run(main())


@scenario
def fanout_route(ctl, params):
    size = params["size"]
    expected = pattern(size, 5)

    async def main():
        dst = bytearray([FILL]) * (size + CANARY_LEN)
        got = await s3dlio.get_into_async(
            uri_of("fanout-object"), dst, expected_len=size
        )
        assert got == size
        assert bytes(dst[:size]) == expected, "ranges were placed wrongly"
        assert dst[size:].count(FILL) == CANARY_LEN
        assert resizable(dst)

    asyncio.run(main())


@scenario
def literal_key(ctl, params):
    key = params["key"]
    expected = pattern(33, 7)

    async def main():
        meta = await s3dlio.head_async(uri_of(key))
        assert meta is not None and meta.size == 33, f"HEAD of the literal key {key!r}"
        dst = bytearray(33)
        assert await s3dlio.get_into_async(uri_of(key), dst, expected_len=33) == 33
        assert bytes(dst) == expected

    asyncio.run(main())


@scenario
def disjoint_slices(ctl, params):
    size, stride = 100_003, 100_100

    async def main():
        big = bytearray([FILL]) * (4 * stride)
        whole = memoryview(big)
        pieces = [whole[i * stride : i * stride + size] for i in range(4)]
        results = await asyncio.gather(
            *[
                s3dlio.get_into_async(
                    uri_of(f"slice-{i}"), pieces[i], expected_len=size
                )
                for i in range(4)
            ]
        )
        assert results == [size] * 4
        for i, piece in enumerate(pieces):
            assert piece.tobytes() == pattern(size, 20 + i), f"slice {i}"
            piece.release()
        whole.release()
        for i in range(4):
            gap = big[i * stride + size : (i + 1) * stride]
            assert gap.count(FILL) == len(gap), f"bytes after slice {i} were written"
        assert resizable(big)

    asyncio.run(main())


@scenario
def start_failure_without_loop(ctl, params):
    size = 4096 + 7
    dst = bytearray(size)
    try:
        s3dlio.get_into_async(uri_of(f"obj-{size}"), dst, expected_len=size)
    except RuntimeError:
        pass
    else:
        raise AssertionError("get_into_async outside an event loop did not raise")
    assert resizable(dst), "BufferError right after a get_into_async that failed to start"
    try:
        s3dlio.head_async(uri_of(f"obj-{size}"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("head_async outside an event loop did not raise")


class StallCase:
    """One blocked read of STALL_KEY and the checks every variant shares."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.dst = bytearray([FILL]) * STALL_SIZE
        self.fut = None
        self.task = None
        self.problems = []

    def on_loop_error(self, loop, context):
        self.problems.append(str(context.get("message")))

    async def start(self, wait=True):
        asyncio.get_running_loop().set_exception_handler(self.on_loop_error)
        baseline = self.ctl.stalled(STALL_KEY)
        self.fut = s3dlio.get_into_async(
            uri_of(STALL_KEY), self.dst, expected_len=STALL_SIZE
        )
        if wait:
            await wait_for_new_stall(self.ctl, STALL_KEY, baseline)
            assert not self.fut.done(), "the blocked read finished"
            assert not resizable(self.dst), "the buffer must be exported while reading"

    async def wait_done_observing(self, seconds=30):
        """Wait for the awaitable to be done; whenever it is seen done, the buffer must
        already be free (terminal never comes before the export is released)."""
        deadline = time.monotonic() + seconds
        while True:
            done = self.fut.done()
            free = resizable(self.dst)
            assert not done or free, "terminal while the buffer was still exported"
            if done:
                return
            assert time.monotonic() < deadline, "the awaitable never finished (hang)"
            await asyncio.sleep(0.005)

    async def finish_clean(self):
        """After the awaitable ended: free buffer, no late writer, nothing logged."""
        assert self.fut.done()
        assert resizable(self.dst)
        self.dst[:] = bytes([AFTER]) * len(self.dst)
        self.ctl.release()
        await asyncio.sleep(0.5)
        assert self.dst.count(AFTER) == len(self.dst), (
            "a native write landed in the buffer after the awaitable had finished"
        )
        self.fut = None
        self.task = None
        gc.collect()
        await asyncio.sleep(0.1)
        assert not self.problems, f"the event loop logged: {self.problems}"


async def awaiting(fut):
    return await fut


async def variant_cancel_pending(case):
    await case.start()
    assert case.fut.cancel() is True, "cancel() of a pending read must return True"
    await case.wait_done_observing()
    assert case.fut.cancelled()
    await case.finish_clean()


async def variant_cancel_before_first_poll(case):
    await case.start(wait=False)
    assert case.fut.cancel() is True
    await case.wait_done_observing()
    assert case.fut.cancelled()
    await case.finish_clean()


async def variant_cancel_after_completion(case):
    case.ctl.release()  # nothing stalls: the read completes
    await case.start(wait=False)
    got = await case.fut
    assert got == STALL_SIZE
    assert case.fut.cancel() is False, "cancel() after completion must return False"
    assert case.fut.done() and not case.fut.cancelled()
    assert case.fut.result() == STALL_SIZE, "the result must stand after a late cancel()"
    assert bytes(case.dst) == pattern(STALL_SIZE, STALL_SEED)
    assert resizable(case.dst)
    case.fut = None
    gc.collect()
    await asyncio.sleep(0.1)
    assert not case.problems, f"the event loop logged: {case.problems}"


async def variant_wait_for_timeout(case):
    await case.start()
    try:
        await asyncio.wait_for(case.fut, 0.4)
    except TimeoutError:
        pass
    else:
        raise AssertionError("wait_for did not time out on a blocked read")
    assert case.fut.done(), "TimeoutError delivered before the read had finished"
    assert resizable(case.dst), "TimeoutError delivered with the buffer still exported"
    await case.finish_clean()


async def variant_asyncio_timeout(case):
    await case.start()
    try:
        async with asyncio.timeout(0.4):
            await case.fut
    except TimeoutError:
        pass
    else:
        raise AssertionError("asyncio.timeout did not fire on a blocked read")
    assert case.fut.done(), "TimeoutError delivered before the read had finished"
    assert resizable(case.dst), "TimeoutError delivered with the buffer still exported"
    await case.finish_clean()


async def task_cancel(case, tight):
    await case.start()
    case.task = asyncio.ensure_future(awaiting(case.fut))
    await asyncio.sleep(0.05)
    for _ in range(3):
        case.task.cancel()
        if not tight:
            await asyncio.sleep(0.02)
    done, _ = await asyncio.wait({case.task}, timeout=30)
    assert case.task in done, "the cancelled task never finished (hang)"
    assert case.task.cancelled()
    assert case.fut.done(), "the task was cancelled before the awaitable finished"
    assert resizable(case.dst)
    await case.finish_clean()


async def variant_task_cancel_repeated(case):
    await task_cancel(case, tight=False)


async def variant_task_cancel_tight(case):
    await task_cancel(case, tight=True)


async def variant_stall_released_completes(case):
    await case.start()
    await asyncio.sleep(0.3)
    assert not case.fut.done()
    assert not resizable(case.dst), "the buffer must stay exported while the read blocks"
    case.ctl.release()
    got = await asyncio.wait_for(case.fut, 30)
    assert got == STALL_SIZE
    assert bytes(case.dst) == pattern(STALL_SIZE, STALL_SEED)
    assert resizable(case.dst)
    case.fut = None
    gc.collect()
    await asyncio.sleep(0.1)
    assert not case.problems, f"the event loop logged: {case.problems}"


@scenario
def cancellation(ctl, params):
    variants = {
        "cancel_pending": variant_cancel_pending,
        "cancel_before_first_poll": variant_cancel_before_first_poll,
        "cancel_after_completion": variant_cancel_after_completion,
        "wait_for_timeout": variant_wait_for_timeout,
        "asyncio_timeout": variant_asyncio_timeout,
        "task_cancel_repeated": variant_task_cancel_repeated,
        "task_cancel_tight": variant_task_cancel_tight,
        "stall_released_completes": variant_stall_released_completes,
    }
    body = variants[params["variant"]]

    async def main():
        await body(StallCase(ctl))

    asyncio.run(main())


class HeadCase:
    def __init__(self, ctl):
        self.ctl = ctl
        self.fut = None
        self.problems = []

    def on_loop_error(self, loop, context):
        self.problems.append(str(context.get("message")))

    async def start(self):
        asyncio.get_running_loop().set_exception_handler(self.on_loop_error)
        baseline = self.ctl.stalled(HEAD_HOLD_KEY)
        self.fut = s3dlio.head_async(uri_of(HEAD_HOLD_KEY))
        await wait_for_new_stall(self.ctl, HEAD_HOLD_KEY, baseline)
        assert not self.fut.done(), "the held HEAD finished"

    async def wait_done(self, seconds=30):
        deadline = time.monotonic() + seconds
        while not self.fut.done():
            assert time.monotonic() < deadline, "the HEAD awaitable never finished (hang)"
            await asyncio.sleep(0.005)

    async def finish_clean(self):
        self.ctl.release()
        self.fut = None
        gc.collect()
        await asyncio.sleep(0.3)
        assert not self.problems, f"the event loop logged: {self.problems}"


@scenario
def head_cancellation(ctl, params):
    variant = params["variant"]

    async def main():
        case = HeadCase(ctl)
        await case.start()
        if variant == "cancel":
            assert case.fut.cancel() is True
            await case.wait_done()
            assert case.fut.cancelled()
        elif variant == "wait_for":
            try:
                await asyncio.wait_for(case.fut, 0.4)
            except TimeoutError:
                pass
            else:
                raise AssertionError("wait_for did not time out on a held HEAD")
            assert case.fut.done()
        elif variant == "asyncio_timeout":
            try:
                async with asyncio.timeout(0.4):
                    await case.fut
            except TimeoutError:
                pass
            else:
                raise AssertionError("asyncio.timeout did not fire on a held HEAD")
            assert case.fut.done()
        elif variant == "released_completes":
            await asyncio.sleep(0.3)
            assert not case.fut.done()
            ctl.release()
            meta = await asyncio.wait_for(case.fut, 30)
            assert meta is not None and meta.size == HEAD_HOLD_SIZE
            assert case.fut.cancel() is False  # after completion: no effect
            assert case.fut.result().size == HEAD_HOLD_SIZE
        else:
            raise AssertionError(f"unknown variant {variant!r}")
        await case.finish_clean()

    asyncio.run(main())


# ---- shutdown scenarios: the child ends WITHOUT awaiting; the parent checks the exit --------


@scenario
def exit_in_flight_get(ctl, params):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    dst = bytearray([FILL]) * STALL_SIZE
    baseline = ctl.stalled(STALL_KEY)

    async def start():
        return s3dlio.get_into_async(uri_of(STALL_KEY), dst, expected_len=STALL_SIZE)

    fut = loop.run_until_complete(start())
    loop.run_until_complete(wait_for_new_stall(ctl, STALL_KEY, baseline))
    KEEP.extend([loop, dst, fut])
    # fall off the end: the interpreter exits with the read in flight and its buffer exported


@scenario
def exit_in_flight_head(ctl, params):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    baseline = ctl.stalled(HEAD_HOLD_KEY)

    async def start():
        return s3dlio.head_async(uri_of(HEAD_HOLD_KEY))

    fut = loop.run_until_complete(start())
    loop.run_until_complete(wait_for_new_stall(ctl, HEAD_HOLD_KEY, baseline))
    KEEP.extend([loop, fut])


@scenario
def exit_after_get(ctl, params):
    size = 4096 + 7

    async def main():
        dst = bytearray(size)
        got = await s3dlio.get_into_async(uri_of(f"obj-{size}"), dst, expected_len=size)
        assert got == size
        assert bytes(dst) == pattern(size, size % 251)

    asyncio.run(main())
    # the interpreter exits right after the last completion


@scenario
def exit_in_flight_mkdir(ctl, params):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def start():
        return [s3dlio.mkdir_async(f"file://{ctl.work}/m{i}/a/b/c") for i in range(200)]

    KEEP.extend([loop, loop.run_until_complete(start())])
    # exit at once, without awaiting any of them


@scenario
def exit_after_mkdir(ctl, params):
    async def main():
        results = await asyncio.gather(
            *[s3dlio.mkdir_async(f"file://{ctl.work}/n{i}/x/y") for i in range(50)]
        )
        assert results == [None] * 50

    asyncio.run(main())


def child_main(argv):
    if len(argv) != 3 or argv[0] not in SCENARIOS:
        print(f"usage: <scenario> <json params> <work dir>; scenarios: {sorted(SCENARIOS)}")
        return 2
    ctl = Ctl(os.environ["CB_FAKE_ENDPOINT"], argv[2])
    SCENARIOS[argv[0]](ctl, json.loads(argv[1]))
    print("SCENARIO-OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(child_main(sys.argv[1:]))
