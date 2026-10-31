# tests/test_abort_at_exit_303.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""#303: a MultipartUploadWriter that is still alive (unfinished) when the interpreter exits, or that
is garbage-collected, must have its AbortMultipartUpload DELIVERED before the process is gone.
(The sink's Drop spawns the abort and no longer blocks; the writer's Drop and the atexit hook wait
for it, bounded.) A tiny keep-alive S3 mock in this process logs every request; a child
interpreter makes the upload and ends without finish()."""

import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

LOG = []
ACKED = []


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            return self.rfile.read(n)
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            while True:
                size = int(self.rfile.readline().split(b";")[0], 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b""):
                        pass
                    break
                self.rfile.read(size)
                self.rfile.readline()
        return b""

    def _send(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        self._body()
        if "uploads" in q:
            LOG.append("CREATE")
            self._send(
                200,
                b'<?xml version="1.0"?><InitiateMultipartUploadResult><Bucket>b</Bucket><Key>k</Key><UploadId>UP1</UploadId></InitiateMultipartUploadResult>',
                {"Content-Type": "application/xml"},
            )
        else:
            LOG.append("COMPLETE")
            self._send(
                200,
                b'<?xml version="1.0"?><CompleteMultipartUploadResult><Bucket>b</Bucket><Key>k</Key><ETag>"e"</ETag></CompleteMultipartUploadResult>',
                {"Content-Type": "application/xml"},
            )

    def do_PUT(self):
        self._body()
        LOG.append("PART")
        self._send(200, b"", {"ETag": '"p"'})

    def do_DELETE(self):
        LOG.append("ABORT")
        # a slow abort: only a process that waits for the answer outlives it
        time.sleep(1.0)
        self._send(204)
        ACKED.append(time.time())


@pytest.fixture()
def server():
    LOG.clear()
    ACKED.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()


CHILD = r"""
import os, sys
import s3dlio
w = s3dlio.MultipartUploadWriter.from_uri("s3://bkt/key", part_size=5 << 20)
w.write(bytes(6 << 20))
mode = sys.argv[1]
if mode == "gc":
    del w            # garbage-collected unfinished
elif mode == "module_global":
    globals()["keep"] = w   # still alive at interpreter shutdown
print("child-done", flush=True)
"""


def run(port, mode):
    env = {
        "PATH": "/usr/bin:/bin",
        "AWS_ENDPOINT_URL": f"http://127.0.0.1:{port}",
        "AWS_ACCESS_KEY_ID": "x",
        "AWS_SECRET_ACCESS_KEY": "y",
        "AWS_REGION": "us-east-1",
        "S3DLIO_OPERATION_TIMEOUT_SECS": "10",
    }
    p = subprocess.run(
        [sys.executable, "-c", CHILD, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "child-done" in p.stdout, p.stdout + p.stderr
    return p


@pytest.mark.parametrize("mode", ["gc", "module_global"])
def test_unfinished_writer_abort_is_delivered_before_the_process_exits(server, mode):
    run(server, mode)
    ended = time.time()
    # No waiting: the child process has exited. The abort must have reached the server, and the
    # process must have waited for it (the server answers after 1 s: the answer cannot have been
    # sent after the process was gone).
    assert "CREATE" in LOG
    assert "ABORT" in LOG, f"the upload leaked: {LOG}"
    assert "COMPLETE" not in LOG
    assert ACKED and ACKED[0] <= ended + 0.05, (
        "the process exited before the abort was answered"
    )
