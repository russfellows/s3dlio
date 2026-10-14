"""A tiny S3-speaking HTTP server for the Python regression tests (issues #198, #201, #204).

It runs in its OWN process (``python slow_s3_server.py [delay_seconds]``), not in a
thread of the test process: a server thread needs the GIL to answer, so a binding
that wrongly holds the GIL across its network call would starve the server it is
waiting for. A separate process turns that bug into a measurable stall instead of
a deadlock.

It prints ``PORT <n>`` on stdout once it is listening, then serves path-style S3:

* every request sleeps ``delay_seconds`` before it answers (the "slow server");
* HEAD /bucket/key                         -> 200, Content-Length 4;
* GET  /bucket?list-type=2&prefix=P        -> a listing with the keys ``P + "bar"``,
  ``P + "other/deep.dat"``, ``P + "/lead"`` and ``P + "x//y"`` (the prefix is echoed
  exactly as received; the last two have an empty segment after the prefix);
* GET  /bucket/key                         -> 200, body ``mock`` (4 bytes);
* PUT  /bucket/key[?partNumber=..]         -> 200 with an ETag;
* POST /bucket/key?uploads                 -> a multipart upload id;
* POST /bucket/key?uploadId=..             -> CompleteMultipartUpload result;
* GET  /__stats                            -> JSON ``{"connections": n, "requests": [...]}``
  (served without the delay); each request is ``[method, raw_target]``.

Connections are persistent (HTTP/1.1), so ``connections`` counts TCP connections
accepted: a client that reuses its connection pool opens few, one that builds a
new client per call opens one per call.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

LOCK = threading.Lock()
REQUESTS = []
CONNECTIONS = [0]


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def get_request(self):
        request = super().get_request()
        with LOCK:
            CONNECTIONS[0] += 1
        return request


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the test output quiet
        pass

    def _reply(self, status, body=b"", headers=None, head=False):
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)) if not head else "4")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _xml(self, xml):
        self._reply(200, xml.encode(), {"Content-Type": "application/xml"})

    def _drain(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

    def _handle(self, method):
        self._drain()
        parts = urlsplit(self.path)
        if method == "GET" and parts.path == "/__stats":
            with LOCK:
                body = json.dumps(
                    {"connections": CONNECTIONS[0], "requests": REQUESTS}
                ).encode()
            self._reply(200, body, {"Content-Type": "application/json"})
            return
        with LOCK:
            REQUESTS.append([method, self.path])
        time.sleep(DELAY)
        query = parse_qs(parts.query, keep_blank_values=True)
        if method == "HEAD":
            self._reply(200, headers={"ETag": '"mock"'}, head=True)
        elif method == "GET" and "list-type" in query:
            prefix = unquote(query.get("prefix", [""])[0])
            self._xml(
                '<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>'
                f"<Name>b</Name><Prefix>{prefix}</Prefix><KeyCount>4</KeyCount>"
                "<MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>"
                f"<Contents><Key>{prefix}bar</Key></Contents>"
                f"<Contents><Key>{prefix}other/deep.dat</Key></Contents>"
                f"<Contents><Key>{prefix}/lead</Key></Contents>"
                f"<Contents><Key>{prefix}x//y</Key></Contents>"
                "</ListBucketResult>"
            )
        elif method == "GET":
            self._reply(200, b"mock", {"ETag": '"mock"'})
        elif method == "PUT":
            self._reply(200, headers={"ETag": '"part"'})
        elif method == "POST" and "uploads" in query:
            self._xml(
                '<?xml version="1.0" encoding="UTF-8"?><InitiateMultipartUploadResult>'
                "<Bucket>b</Bucket><Key>k</Key><UploadId>upload-1</UploadId>"
                "</InitiateMultipartUploadResult>"
            )
        elif method == "POST":
            self._xml(
                '<?xml version="1.0" encoding="UTF-8"?><CompleteMultipartUploadResult>'
                "<Bucket>b</Bucket><Key>k</Key><ETag>&quot;done&quot;</ETag>"
                "</CompleteMultipartUploadResult>"
            )
        else:
            self._reply(200)

    def do_GET(self):
        self._handle("GET")

    def do_HEAD(self):
        self._handle("HEAD")

    def do_PUT(self):
        self._handle("PUT")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")


if __name__ == "__main__":
    server = Server(("127.0.0.1", 0), Handler)
    print(f"PORT {server.server_address[1]}", flush=True)
    server.serve_forever()
