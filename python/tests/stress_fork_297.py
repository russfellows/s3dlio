# python/tests/stress_fork_297.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""Fork stress for #297 (not collected by pytest: run it by hand).

    python stress_fork_297.py [CHILDREN=250] [BUSY_THREADS=8] [HANG_S=30]

The parent keeps BUSY_THREADS threads inside s3dlio calls (get / list / put / async get / datagen
/ checkpoint) and PARKED threads (one blocked inside `s3dlio.get` on a mock S3 object that never
answers, and two consumers of one loader iterator over such an object: one blocked in the loader's
blocking receive, the other parked on the iterator's lock), then forks CHILDREN children back to
back. Each child does a small s3dlio workload and, one time in five, forks a grandchild that does
the same.

Prints one JSON summary. A child that hangs is attached to with gdb (when installed) and its
backtraces are written next to the script's scratch directory, so a hang can be attributed (or
not) to the one known residual: pyo3's own reference-pool mutex.
"""

import asyncio
import contextlib
import ctypes
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import s3dlio

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
# S3DLIO_STRESS_NO_ASYNC=1: no async calls in the parent
BUSY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HANG_S = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value

MOCK = r"""
import http.server, socketserver, sys, os, time
port, release = int(sys.argv[1]), sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        while not os.path.exists(release): time.sleep(0.05)      # the parked threads wait here
        self.send_response(200); self.send_header("Content-Length", "5"); self.end_headers(); self.wfile.write(b"hello")
    do_HEAD = do_GET
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True; allow_reuse_address = True
S(("127.0.0.1", port), H).serve_forever()
"""
import socket as _socket

with _socket.socket() as _s:
    _s.bind(("127.0.0.1", 0))
    MOCK_PORT = _s.getsockname()[1]
os.environ.update(
    AWS_ENDPOINT_URL=f"http://127.0.0.1:{MOCK_PORT}",
    AWS_ACCESS_KEY_ID="t",
    AWS_SECRET_ACCESS_KEY="t",
    AWS_REGION="us-east-1",
    AWS_EC2_METADATA_DISABLED="true",
)
D = tempfile.mkdtemp(prefix="stress297_")
RELEASE = f"{D}/release"
_mock = subprocess.Popen([sys.executable, "-c", MOCK, str(MOCK_PORT), RELEASE])
# wait until the mock accepts connections (a loaded host can be slow)
for _ in range(200):
    try:
        _socket.create_connection(("127.0.0.1", MOCK_PORT), timeout=0.2).close()
        break
    except OSError:
        time.sleep(0.05)
for i in range(4):
    with open(f"{D}/o{i}.bin", "wb") as f:
        f.write(bytes([i]) * 65536)
URI = f"file://{D}/o0.bin"
DIR = f"file://{D}/"

stop = threading.Event()


async def _ag():
    return await s3dlio.get_async(URI)


def busy(k):
    st = None
    while not stop.is_set():
        try:
            s3dlio.get(URI)
            s3dlio.list(DIR)
            s3dlio.put_bytes(f"file://{D}/t{k}.bin", b"y" * 100)
            m = k % 3
            if m == 0:
                s3dlio.generate_data(2 << 20)
            elif m == 1:
                if os.environ.get("S3DLIO_STRESS_NO_ASYNC") == "1":
                    # the async delivery path is left out of the parent's load
                    s3dlio.get(URI)
                else:
                    asyncio.run(_ag())
            else:
                if st is None:
                    st = s3dlio.PyCheckpointStore(f"file://{D}/ck{k}")
                st.save(1, 1, "pt", b"x" * 100)
        except BaseException as e:  # noqa: BLE001
            print("busy thread error", k, type(e).__name__, e, flush=True)
            return


def blocked_get():
    with contextlib.suppress(BaseException):
        s3dlio.get("s3://bkt/slow_get")


ds = s3dlio.PyDataset.from_uris(["s3://bkt/slow_loader"])
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"batch_size": 1})
SHARED_IT = iter(loader)


def consumer():
    with contextlib.suppress(BaseException):
        next(SHARED_IT)


threads = [threading.Thread(target=busy, args=(k,), daemon=True) for k in range(BUSY)]
threads += [threading.Thread(target=blocked_get, daemon=True)]
threads += [threading.Thread(target=consumer, daemon=True) for _ in range(2)]
for t in threads:
    t.start()
time.sleep(1.5)
if not all(t.is_alive() for t in threads[BUSY:]):
    sys.exit(
        "stress scenario failed: the parked threads are not parked (mock S3 not answering as designed)"
    )


def child_work(depth):
    with contextlib.suppress(Exception):
        ctypes.CDLL(None).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    s3dlio.get(URI)
    asyncio.run(_ag())
    s3dlio.generate_data(2 << 20)
    g = s3dlio.Generator(size=2 << 20)
    g.fill_chunk(bytearray(2 << 20))
    st = s3dlio.PyCheckpointStore(f"file://{D}/ck_c{os.getpid()}")
    st.save(1, 1, "pt", b"c" * 10)
    list(s3dlio.create_async_loader(DIR, {"batch_size": 2}))
    if depth == 0 and random.random() < 0.2:
        pid = os.fork()
        if pid == 0:
            try:
                child_work(1)
            except BaseException:  # noqa: BLE001
                os._exit(3)
            os._exit(0)
        t0 = time.time()
        while time.time() - t0 < HANG_S / 2:
            p, s = os.waitpid(pid, os.WNOHANG)
            if p:
                if os.WIFSIGNALED(s) or os.WEXITSTATUS(s) != 0:
                    raise RuntimeError(f"grandchild status {s}")
                return
            time.sleep(0.01)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise RuntimeError("grandchild HUNG")


res = {"OK": 0, "HUNG": 0, "CRASH": 0, "FAILED": 0}
crashes, hangs = [], []
os.makedirs(f"{D}/hangs", exist_ok=True)
t_start = time.time()
for i in range(N):
    pid = os.fork()
    if pid == 0:
        try:
            child_work(0)
        except BaseException as e:  # noqa: BLE001
            sys.stderr.write(f"child failed: {type(e).__name__}: {e}\n")
            os._exit(3)
        os._exit(0)
    t0 = time.time()
    status = None
    while time.time() - t0 < HANG_S:
        p, s = os.waitpid(pid, os.WNOHANG)
        if p:
            status = s
            break
        time.sleep(0.005)
    if status is None:
        res["HUNG"] += 1
        bt = ""
        if shutil.which("gdb"):
            bt = subprocess.run(
                ["gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt 25"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            ).stdout
        path = f"{D}/hangs/hang_{i}_{pid}.txt"
        with open(path, "w") as f:
            f.write(bt)
        hangs.append(path)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    elif os.WIFSIGNALED(status):
        res["CRASH"] += 1
        crashes.append((i, os.WTERMSIG(status)))
    elif os.WEXITSTATUS(status) != 0:
        res["FAILED"] += 1
    else:
        res["OK"] += 1
    time.sleep(random.random() * 0.02)

parked_alive = [
    t.is_alive() for t in threads[BUSY:]
]  # the parked threads are still parked
stop.set()
open(RELEASE, "w").close()  # release the parked threads
time.sleep(0.5)
_mock.kill()
summary = {
    "children": N,
    "busy_threads": BUSY,
    "seconds": round(time.time() - t_start, 1),
    "parked_threads_alive_at_end": parked_alive,
    "results": res,
    "crashes": crashes,
    "hang_backtraces": hangs,
    "scratch": D,
}
print("SUMMARY " + json.dumps(summary), flush=True)
os._exit(0 if res["CRASH"] == 0 and res["FAILED"] == 0 else 1)
