# python/tests/stress_fork_297.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""Fork stress for #297 (not collected by pytest: run it by hand).

    python stress_fork_297.py [CHILDREN=250] [BUSY_THREADS=8] [HANG_S=30]

The parent keeps BUSY_THREADS threads inside s3dlio calls (get / list / put / async get / datagen
/ checkpoint) and two kinds of PARKED threads (one blocked inside `s3dlio.get` on a FIFO, two
consumers of one loader iterator, one blocked in the loader's blocking receive while the other
waits for its lock), then forks CHILDREN children back to back. Each child does a small s3dlio
workload and, one time in five, forks a grandchild that does the same.

Prints one JSON summary. A child that hangs is attached to with gdb (when installed) and its
backtraces are written next to the script's scratch directory, so a hang can be attributed (or
not) to the one known residual: pyo3's own reference-pool mutex.
"""

import asyncio
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
BUSY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HANG_S = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value

D = tempfile.mkdtemp(prefix="stress297_")
for i in range(4):
    with open(f"{D}/o{i}.bin", "wb") as f:
        f.write(bytes([i]) * 65536)
URI = f"file://{D}/o0.bin"
DIR = f"file://{D}/"
FIFO_GET = f"{D}/fifo_get"
FIFO_LOADER = f"{D}/fifo_loader"
os.mkfifo(FIFO_GET)
os.mkfifo(FIFO_LOADER)

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
                asyncio.run(_ag())
            else:
                if st is None:
                    st = s3dlio.PyCheckpointStore(f"file://{D}/ck{k}")
                st.save(1, 1, "pt", b"x" * 100)
        except BaseException as e:  # noqa: BLE001
            print("busy thread error", k, type(e).__name__, e, flush=True)
            return


def blocked_get():
    try:
        s3dlio.get(f"file://{FIFO_GET}")
    except BaseException:  # noqa: BLE001
        pass


ds = s3dlio.PyDataset.from_uris([f"file://{FIFO_LOADER}"])
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"batch_size": 1})
SHARED_IT = iter(loader)


def consumer():
    try:
        next(SHARED_IT)
    except BaseException:  # noqa: BLE001
        pass


threads = [threading.Thread(target=busy, args=(k,), daemon=True) for k in range(BUSY)]
threads += [threading.Thread(target=blocked_get, daemon=True)]
threads += [threading.Thread(target=consumer, daemon=True) for _ in range(2)]
for t in threads:
    t.start()
time.sleep(1.0)


def child_work(depth):
    try:
        ctypes.CDLL(None).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass
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
                capture_output=True, text=True, timeout=120,
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

parked_alive = [t.is_alive() for t in threads[BUSY:]]  # the parked threads are still parked
stop.set()
for fifo in (FIFO_GET, FIFO_LOADER):  # release the parked threads
    try:
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.write(fd, b"x" * 10)
        os.close(fd)
    except OSError:
        pass
summary = {
    "children": N, "busy_threads": BUSY, "seconds": round(time.time() - t_start, 1),
    "parked_threads_alive_at_end": parked_alive, "results": res, "crashes": crashes, "hang_backtraces": hangs, "scratch": D,
}
print("SUMMARY " + json.dumps(summary), flush=True)
os._exit(0 if res["CRASH"] == 0 and res["FAILED"] == 0 else 1)
