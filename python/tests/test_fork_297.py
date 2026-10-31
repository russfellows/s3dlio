# python/tests/test_fork_297.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""Fork safety (#297): a forked child must work, or raise a catchable RuntimeError, never hang.

``fork()`` copies only the calling thread. Before the fix a child that called s3dlio after the
parent had used it waited for ever for runtime, rayon and connection threads that do not exist in
the child. The behaviour now:

* every s3dlio entry point works in a fork child (fresh runtimes, pools, store cache and clients
  are built lazily; one stderr line says so; ``S3DLIO_FORK_WARN=0`` silences it);
* an OBJECT created before the fork (checkpoint store, writer, loader, dataset, multi-endpoint
  store, generator, iterators) raises ``RuntimeError("... created before fork() ...")`` when used
  in the child, and is never dropped there (``del``, garbage collection and interpreter exit
  are safe);
* nothing a child does is visible to the parent.

How it runs: this file is its own harness. ``python test_fork_297.py HARNESS WARM`` forks one child
per case from a parent that first did WARM (none, sync, async, datagen, oplog, objects, all), with a
hang limit per child, and prints one JSON line per case. The pytest functions run the harness in a
subprocess (so a hang can never block the test run) and judge the lines.

Run (the in-repo convention):

    uv run --no-project --isolated --with <wheel> --with pytest \
        python -m pytest python/tests/test_fork_297.py -v
"""

import asyncio
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

WARMS = ["none", "sync", "async", "datagen", "oplog", "objects", "all"]
WARN_MARK = "s3dlio: this process was forked after s3dlio had already been used"
INHERITED_MSG = "created before fork()"
HANG_S = float(os.environ.get("S3DLIO_FORK_TEST_HANG_S", "25"))
GEN_BYTES = 32 << 20


# =============================================================================================
# Harness (runs in a subprocess)
# =============================================================================================
def _harness_main(warm, hang_s, only=None):
    import s3dlio

    D = tempfile.mkdtemp(prefix="fork297_")
    for i in range(4):
        with open(f"{D}/o{i}.bin", "wb") as f:
            f.write(bytes([i]) * 65536)
    URI = f"file://{D}/o0.bin"
    URI1 = f"file://{D}/o1.bin"
    DIR = f"file://{D}/"
    N = {}  # objects the parent made before the fork

    def out_uri():
        return f"file://{D}/out_{os.getpid()}_{time.monotonic_ns()}.bin"

    # ------------------------------------------------------------------ parent states
    def warm_sync():
        s3dlio.get(URI)
        s3dlio.list(DIR)
        s3dlio.put_bytes(out_uri(), b"p" * 100)

    def warm_async():
        async def w():
            await s3dlio.get_async(URI)
            await s3dlio.put_bytes_async(out_uri(), b"p" * 100)

        asyncio.run(w())

    def warm_datagen():
        s3dlio.generate_data(GEN_BYTES)
        s3dlio.generate_npz_bytes(shape=[512, 512, 8])
        g = s3dlio.Generator(size=GEN_BYTES)
        g.fill_chunk(bytearray(GEN_BYTES))

    def warm_oplog():
        s3dlio.init_op_log(f"{D}/parent_oplog.tsv.zst")
        s3dlio.get(URI)  # leaves the op-log ACTIVE across the fork

    def warm_objects():
        N["ck"] = s3dlio.PyCheckpointStore(f"file://{D}/ck_parent")
        N["ck"].save(1, 1, "pt", b"x" * 100)
        N["ckw"] = N["ck"].writer(1, 0)
        N["ckr"] = N["ck"].reader()
        N["cks"] = N["ckw"].get_distributed_shard_stream(2, 1, "pt")
        N["cks"].write_chunk(b"s" * 10)  # an open shard stream
        N["mep"] = s3dlio.create_multi_endpoint_store([DIR])
        N["ds"] = s3dlio.PyDataset.from_uris([URI, URI1])
        N["ldr"] = s3dlio.PyBytesAsyncDataLoader(N["ds"], {"batch_size": 1})
        N["it"] = iter(N["ldr"])
        next(N["it"])
        N["vl"] = s3dlio.PyAsyncDataLoader(s3dlio.PyVecDataset(list(range(8))), {"batch_size": 2})
        N["vit"] = N["vl"].__aiter__()  # an async iterator over a producer task of the parent
        N["wr"] = s3dlio.create_filesystem_writer(f"file://{D}/parent_writer.bin")
        N["wr"].write_chunk(b"w" * 10)  # an unfinalised writer
        N["gen"] = s3dlio.Generator(size=GEN_BYTES)
        N["gen"].fill_chunk(bytearray(1 << 20))

    def parent_warm():
        if warm in ("sync", "all"):
            warm_sync()
        if warm in ("async", "all"):
            warm_async()
        if warm in ("datagen", "all"):
            warm_datagen()
        if warm in ("oplog", "all"):
            warm_oplog()
        if warm in ("objects", "all"):
            warm_objects()

    # ------------------------------------------------------------------ child cases
    CASES = {}  # name -> (fn, kind, needs_objects, normal_exit)

    def case(kind="fresh", needs=False, normal_exit=False):
        def deco(f):
            CASES[f.__name__.rstrip("_")] = (f, kind, needs, normal_exit)
            return f

        return deco

    @case()
    def get():
        return len(memoryview(s3dlio.get(URI)))

    @case()
    def get_range():
        return len(memoryview(s3dlio.get_range(URI, 0, 100)))

    @case()
    def put_bytes():
        s3dlio.put_bytes(out_uri(), b"z" * 1000)
        return 1000

    @case()
    def list_():
        return len(s3dlio.list(DIR))

    @case()
    def stat():
        return str(s3dlio.stat(URI))[:40]

    @case()
    def delete():
        u = out_uri()
        s3dlio.put_bytes(u, b"q")
        s3dlio.delete(u, False)
        return 1

    @case()
    def get_many():
        return len(s3dlio.get_many([URI, URI1]))

    @case()
    def get_async():
        async def f():
            return len(memoryview(await s3dlio.get_async(URI)))

        return asyncio.run(f())

    @case()
    def get_into_async():
        async def f():
            b = bytearray(65536)
            return await s3dlio.get_into_async(URI, b, expected_len=65536)

        return asyncio.run(f())

    @case()
    def head_async():
        async def f():
            return (await s3dlio.head_async(URI)).size

        return asyncio.run(f())

    @case()
    def put_bytes_async():
        async def f():
            await s3dlio.put_bytes_async(out_uri(), b"a" * 100)
            return 100

        return asyncio.run(f())

    @case()
    def multi_endpoint_new():
        async def f():
            st = s3dlio.create_multi_endpoint_store([DIR])
            return len(await st.get(URI))

        return asyncio.run(f())

    @case()
    def writer_fs():
        w = s3dlio.create_filesystem_writer(out_uri())
        w.write_chunk(b"w" * 1000)
        w.finalize()
        return 1000

    @case()
    def checkpoint_new():
        st = s3dlio.PyCheckpointStore(f"file://{D}/ck_child_{os.getpid()}")
        st.save(2, 1, "pt", b"y" * 100)
        st.load_latest()
        w = st.writer(1, 0)
        s = w.get_distributed_shard_stream(3, 1, "pt")
        s.write_chunk(b"c" * 10)
        meta = s.finalize()
        return len(st.list_checkpoints()) + (1 if meta else 0)

    @case()
    def loader_new():
        return len(list(s3dlio.create_async_loader(DIR, {"batch_size": 2})))

    @case()
    def bytes_loader_new():
        ds = s3dlio.PyDataset.from_uris([URI, URI1])
        return len(list(s3dlio.PyBytesAsyncDataLoader(ds, {"batch_size": 1})))

    @case()
    def vec_loader_new():
        async def f():
            n = 0
            async for _ in s3dlio.PyAsyncDataLoader(s3dlio.PyVecDataset(list(range(8))), {"batch_size": 2}):
                n += 1
            return n

        return asyncio.run(f())

    @case()
    def dataset_new():
        async def f():
            ds = s3dlio.PyDataset.from_uris([URI])
            return len(await ds.get_item(0))

        return asyncio.run(f())

    @case()
    def generate_data():
        return len(memoryview(s3dlio.generate_data(GEN_BYTES)))

    @case()
    def generate_data_with_threads():
        return len(memoryview(s3dlio.generate_data_with_threads(GEN_BYTES, threads=4)))

    @case()
    def generate_into_buffer():
        return s3dlio.generate_into_buffer(bytearray(GEN_BYTES))

    @case()
    def generator_new():
        g = s3dlio.Generator(size=GEN_BYTES)
        buf = bytearray(GEN_BYTES)
        n = g.fill_chunk(buf)
        return n

    @case()
    def generate_npz_bytes():
        return len(memoryview(s3dlio.generate_npz_bytes(shape=[512, 512, 8])))

    @case()
    def oplog_init():
        p = f"{D}/child_oplog_{os.getpid()}.tsv.zst"
        s3dlio.init_op_log(p)
        s3dlio.get(URI)
        s3dlio.finalize_op_log()
        return 1

    @case()
    def configure_then_get():
        s3dlio.configure_tokio_threads(2)
        return len(memoryview(s3dlio.get(URI)))

    @case()
    def nested_fork():
        # the child forks again: the grandchild must also work
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                len(memoryview(s3dlio.get(URI)))
                asyncio.run(_aget())
                s3dlio.generate_data(GEN_BYTES)
                os.write(w, b"OK")
            except BaseException as e:  # noqa: BLE001
                os.write(w, f"RAISED {type(e).__name__}: {e}".encode())
            os._exit(0)
        os.close(w)
        deadline = time.time() + hang_s / 2
        while time.time() < deadline:
            p, _ = os.waitpid(pid, os.WNOHANG)
            if p:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise AssertionError("grandchild HUNG")
        msg = os.read(r, 4096).decode()
        if msg != "OK":
            raise AssertionError(f"grandchild: {msg}")
        return 1

    async def _aget():
        return len(memoryview(await s3dlio.get_async(URI)))

    # ---- objects created by the parent, used in the child: RuntimeError, never a hang
    def inherited(name, fn):
        CASES[name] = (fn, "inherited", True, False)

    inherited("ckpt_inherited_save", lambda: N["ck"].save(5, 1, "pt", b"z"))
    inherited("ckpt_inherited_list", lambda: N["ck"].list_checkpoints())
    inherited("ckpt_inherited_writer", lambda: N["ck"].writer(1, 0))
    inherited("ckpt_writer_inherited", lambda: N["ckw"].save_distributed_shard(6, 1, "pt", b"z"))
    inherited("ckpt_reader_inherited", lambda: N["ckr"].load_latest_manifest())
    inherited("ckpt_stream_inherited_write", lambda: N["cks"].write_chunk(b"zz"))
    inherited("ckpt_stream_inherited_cancel", lambda: N["cks"].cancel())
    inherited("mep_inherited", lambda: asyncio.run(_mep_get()))
    inherited("dataset_inherited", lambda: len(N["ds"]))
    inherited("loader_inherited", lambda: list(N["ldr"]))
    inherited("iterator_inherited", lambda: next(N["it"]))
    inherited("async_iterator_inherited", lambda: asyncio.run(_aiter_next(N["vit"])))
    inherited("writer_inherited_write", lambda: N["wr"].write_chunk(b"x"))
    inherited("writer_inherited_finalize", lambda: N["wr"].finalize())
    inherited("generator_inherited", lambda: N["gen"].fill_chunk(bytearray(1 << 20)))

    async def _mep_get():
        return await N["mep"].get(URI)

    async def _aiter_next(it):
        return await it.__anext__()

    # ---- dropping / exiting with inherited objects must not hang or touch the parent's state
    @case(kind="drop", needs=True)
    def drop_inherited_objects():
        for k in list(N):
            del N[k]
        gc.collect()
        return 1

    @case(kind="drop", needs=True, normal_exit=True)
    def exit_with_inherited_objects():
        return 1  # the child leaves through normal interpreter shutdown with N alive

    @case(kind="drop", needs=True, normal_exit=True)
    def exit_after_del_inherited_ckpt():
        N.pop("ck", None)
        N.pop("ckw", None)
        N.pop("ckr", None)
        N.pop("cks", None)
        gc.collect()
        return 1

    @case(normal_exit=True)
    def exit_after_use():
        s3dlio.get(URI)
        asyncio.run(_aget())
        return 1

    # ------------------------------------------------------------------ run
    def parent_health():
        try:
            assert len(memoryview(s3dlio.get(URI))) == 65536
            assert asyncio.run(_aget()) == 65536
            assert len(memoryview(s3dlio.generate_data(1 << 20))) == 1 << 20
            for k in ("ck",):
                if k in N:
                    N[k].save(9, 1, "pt", b"h")
            if "wr" in N:
                N["wr"].write_chunk(b"more")
            if "gen" in N:
                N["gen"].fill_chunk(bytearray(1 << 20))
            if "it" in N:
                next(N["it"], None)
            if "mep" in N:
                assert len(asyncio.run(_mep_get())) == 65536
            return True
        except BaseException as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    parent_warm()
    names = [c for c in CASES if (only is None or c in only)]
    recs = []
    for name in names:
        fn, kind, needs, normal_exit = CASES[name]
        if needs and warm not in ("objects", "all"):
            continue
        errf = f"{D}/stderr_{name}.txt"
        r, w = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            fd = os.open(errf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(fd, 2)
            try:
                v = fn()
                msg = {"res": "OK", "v": repr(v)[:60]}
            except BaseException as e:  # noqa: BLE001
                msg = {"res": "RAISED", "type": type(e).__name__, "msg": str(e)[:300]}
            os.write(w, json.dumps(msg).encode())
            os.close(w)
            if normal_exit:
                sys.exit(0)
            os._exit(0)
        os.close(w)
        t0 = time.time()
        status = None
        while time.time() - t0 < hang_s:
            p, st = os.waitpid(pid, os.WNOHANG)
            if p:
                status = st
                break
            time.sleep(0.02)
        rec = {"case": name, "warm": warm, "kind": kind}
        if status is None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            rec["res"] = "HUNG"
            try:
                rec["partial"] = os.read(r, 4096).decode()
            except OSError:
                pass
        else:
            raw = os.read(r, 4096).decode()
            os.close(r)
            if os.WIFSIGNALED(status):
                rec["res"] = f"CRASH:sig{os.WTERMSIG(status)}"
            else:
                rec.update(json.loads(raw) if raw else {"res": "NO-RESULT"})
                rec["exit"] = os.WEXITSTATUS(status)
        try:
            rec["warns"] = open(errf).read().count(WARN_MARK)
        except OSError:
            rec["warns"] = -1
        recs.append(rec)
    # The parent's own health is checked once, AFTER every child: checking between children would
    # change what the parent "did before fork" for the next one (e.g. a `none` parent would stop
    # being one).
    health = parent_health()
    for rec in recs:
        rec["parent_ok"] = health
        print("RESULT " + json.dumps(rec), flush=True)


# =============================================================================================
# pytest side
# =============================================================================================
def _run_harness(warm, env_extra=None, only=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [sys.executable, os.path.abspath(__file__), "HARNESS", warm]
    if only:
        cmd += list(only)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
    recs = {}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            recs[r["case"]] = r
    return recs, p


try:
    import pytest
except ImportError:  # the harness mode needs no pytest
    pytest = None

if pytest is not None:
    _CACHE = {}

    def _results(warm):
        if warm not in _CACHE:
            recs, p = _run_harness(warm)
            assert recs, f"harness produced nothing:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
            _CACHE[warm] = recs
        return _CACHE[warm]

    ALL_CASES = [
        "get", "get_range", "put_bytes", "list", "stat", "delete", "get_many", "get_async",
        "get_into_async", "head_async", "put_bytes_async", "multi_endpoint_new", "writer_fs",
        "checkpoint_new", "loader_new", "bytes_loader_new", "vec_loader_new", "dataset_new",
        "generate_data", "generate_data_with_threads", "generate_into_buffer", "generator_new",
        "generate_npz_bytes", "oplog_init", "configure_then_get", "nested_fork",
        "exit_after_use",
    ]
    INHERITED_CASES = [
        "ckpt_inherited_save", "ckpt_inherited_list", "ckpt_inherited_writer",
        "ckpt_writer_inherited", "ckpt_reader_inherited", "ckpt_stream_inherited_write",
        "ckpt_stream_inherited_cancel", "mep_inherited", "dataset_inherited", "loader_inherited",
        "iterator_inherited", "async_iterator_inherited", "writer_inherited_write",
        "writer_inherited_finalize", "generator_inherited",
    ]
    DROP_CASES = [
        "drop_inherited_objects", "exit_with_inherited_objects", "exit_after_del_inherited_ckpt",
    ]

    @pytest.mark.parametrize("warm", WARMS)
    @pytest.mark.parametrize("case", ALL_CASES)
    def test_every_entry_point_works_in_a_fork_child(warm, case):
        """Whatever the parent did before fork(), the child's call works (never hangs/crashes)."""
        rec = _results(warm)[case]
        assert rec["res"] == "OK", rec
        assert rec["exit"] == 0, rec
        assert rec["parent_ok"] is True, rec

    @pytest.mark.parametrize("case", INHERITED_CASES)
    @pytest.mark.parametrize("warm", ["objects", "all"])
    def test_an_object_made_before_fork_raises_a_catchable_runtimeerror(warm, case):
        """Parent-created objects cannot work in the child: RuntimeError, not a hang, a panic
        (PanicException is a BaseException) or a crash."""
        rec = _results(warm)[case]
        assert rec["res"] == "RAISED", rec
        assert rec["type"] == "RuntimeError", rec
        assert INHERITED_MSG in rec["msg"], rec
        assert rec["parent_ok"] is True, rec

    @pytest.mark.parametrize("case", DROP_CASES)
    @pytest.mark.parametrize("warm", ["objects", "all"])
    def test_dropping_or_exiting_with_inherited_objects_is_safe(warm, case):
        """`del`, garbage collection and normal interpreter exit in the child: no hang, no abort
        of the parent's checkpoint store / writer / generator (review finding B2)."""
        rec = _results(warm)[case]
        assert rec["res"] == "OK", rec
        assert rec["exit"] == 0, rec
        assert rec["parent_ok"] is True, rec

    def test_the_child_warns_once_and_only_after_use():
        # parent never used s3dlio: nothing was inherited, nothing to say
        for case, rec in _results("none").items():
            assert rec["warns"] == 0, rec
        # parent used it: at most ONE line per child process, and the first use says it
        for warm, case in (("sync", "get"), ("async", "get_async"), ("datagen", "generate_data")):
            assert _results(warm)[case]["warns"] == 1, _results(warm)[case]
        for warm in ("sync", "async", "datagen", "all"):
            for case, rec in _results(warm).items():
                assert 0 <= rec["warns"] <= 1, rec

    def test_the_warning_can_be_silenced():
        recs, _ = _run_harness("all", {"S3DLIO_FORK_WARN": "0"}, only=["get", "get_async", "generate_data", "nested_fork"])
        assert set(recs) == {"get", "get_async", "generate_data", "nested_fork"}
        for rec in recs.values():
            assert rec["res"] == "OK" and rec["warns"] == 0, rec

    def test_the_parent_is_unaffected_by_all_the_children():
        for warm in WARMS:
            for rec in _results(warm).values():
                assert rec["parent_ok"] is True, rec


# =============================================================================================
# Scenarios that are not a (case x warm) cell
# =============================================================================================
def _run_script(code, timeout=180, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=env)


if pytest is not None:

    def test_fork_while_other_parent_threads_are_inside_s3dlio_calls():
        """Parent threads loop in get / put / list / datagen / async calls while the main thread
        forks child after child. No child may crash; the children that do their call must finish."""
        code = f"""
import os, sys, time, signal, threading, tempfile, asyncio, json
import s3dlio
D = tempfile.mkdtemp(); open(D + "/o.bin", "wb").write(b"x" * 65536)
U = "file://" + D + "/o.bin"
stop = threading.Event()
async def ag(): return await s3dlio.get_async(U)
def worker(k):
    while not stop.is_set():
        s3dlio.get(U); s3dlio.list("file://" + D + "/"); s3dlio.put_bytes("file://%s/t%d.bin" % (D, k), b"y" * 100)
        if k % 2 == 0:
            s3dlio.generate_data(2 << 20)
        else:
            asyncio.run(ag())
ts = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(6)]
[t.start() for t in ts]; time.sleep(0.3)
res = {{"OK": 0, "HUNG": 0, "CRASH": 0, "RAISED": 0}}
for i in range(40):
    pid = os.fork()
    if pid == 0:
        try:
            s3dlio.get(U); asyncio.run(ag()); s3dlio.generate_data(2 << 20)
        except BaseException:
            os._exit(3)
        os._exit(0)
    t0 = time.time(); st = None
    while time.time() - t0 < {HANG_S}:
        p, s = os.waitpid(pid, os.WNOHANG)
        if p: st = s; break
        time.sleep(0.01)
    if st is None: os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); res["HUNG"] += 1
    elif os.WIFSIGNALED(st): res["CRASH"] += 1
    elif os.WEXITSTATUS(st) != 0: res["RAISED"] += 1
    else: res["OK"] += 1
    time.sleep(0.005)
stop.set()
print("RES", json.dumps(res))
"""
        p = _run_script(code, timeout=1500)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RES ")]
        assert line, (p.stdout[-1500:], p.stderr[-1500:])
        res = json.loads(line[0][4:])
        assert res["CRASH"] == 0 and res["RAISED"] == 0, res
        # the documented residual (a parent thread holding pyo3's own reference-pool lock at the
        # instant of fork, which s3dlio cannot reach) is the only known source of a hang
        assert res["HUNG"] <= 1, res

    def test_multiprocessing_fork_pool_after_the_parent_used_s3dlio():
        code = """
import multiprocessing as mp, os, tempfile, asyncio
import s3dlio
D = tempfile.mkdtemp(); open(D + "/o.bin", "wb").write(b"x" * 65536)
U = "file://" + D + "/o.bin"
async def ag(): return await s3dlio.get_async(U)
def work(i):
    a = len(memoryview(s3dlio.get(U)))
    b = asyncio.run(ag())
    c = len(memoryview(s3dlio.generate_data(8 << 20)))
    return a + len(memoryview(b)) + c
if __name__ == "__main__":
    s3dlio.get(U); asyncio.run(ag()); s3dlio.generate_data(8 << 20)
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        r = pool.map(work, range(8))
    assert r == [65536 * 2 + (8 << 20)] * 8, r
    print("POOL OK")
"""
        p = _run_script(code, timeout=180)
        assert "POOL OK" in p.stdout, (p.stdout[-1500:], p.stderr[-1500:])

    def test_fork_then_exec_is_unaffected():
        """A process that execs after fork (subprocess, os.execv) is a new process: no
        inherited state, no warning."""
        code = """
import os, sys, subprocess, tempfile
import s3dlio
D = tempfile.mkdtemp(); open(D + "/o.bin", "wb").write(b"x" * 65536)
U = "file://" + D + "/o.bin"
s3dlio.get(U); s3dlio.generate_data(8 << 20)
child = "import s3dlio; assert len(memoryview(s3dlio.get(%r))) == 65536; print('EXEC OK')" % U
r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=60)
assert "EXEC OK" in r.stdout and "forked" not in r.stderr, (r.stdout, r.stderr)
pid = os.fork()
if pid == 0:
    os.execv(sys.executable, [sys.executable, "-c", child])
_, st = os.waitpid(pid, 0)
assert os.WEXITSTATUS(st) == 0
print("EXEC DONE")
"""
        p = _run_script(code, timeout=120)
        assert "EXEC DONE" in p.stdout, (p.stdout[-1500:], p.stderr[-1500:])
        assert "forked after s3dlio" not in p.stderr, p.stderr

    def test_a_dataloader_style_worker_pool_with_fork():
        """PyTorch's DataLoader forks its workers by default. torch is optional: without it the
        same shape is run with multiprocessing (fork) workers that stream items back."""
        code = """
import os, tempfile, time
import s3dlio
D = tempfile.mkdtemp()
for i in range(16): open("%s/o%d.bin" % (D, i), "wb").write(bytes([i]) * 4096)
URIS = ["file://%s/o%d.bin" % (D, i) for i in range(16)]
s3dlio.get(URIS[0])                       # the parent used s3dlio
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
if torch is not None:
    class DS(Dataset):
        def __len__(self): return len(URIS)
        def __getitem__(self, i): return torch.tensor(memoryview(s3dlio.get(URIS[i]))[0], dtype=torch.int64)
    dl = DataLoader(DS(), batch_size=4, num_workers=4, timeout=60)
    got = sorted(int(x) for b in dl for x in b)
    assert got == list(range(16)), got
    print("TORCH OK")
else:
    import multiprocessing as mp
    def work(i): return memoryview(s3dlio.get(URIS[i]))[0]
    with mp.get_context("fork").Pool(4) as p:
        got = sorted(p.map(work, range(16)))
    assert got == list(range(16)), got
    print("MP OK")
"""
        p = _run_script(code, timeout=300)
        assert ("TORCH OK" in p.stdout) or ("MP OK" in p.stdout), (p.stdout[-1500:], p.stderr[-1500:])


# =============================================================================================
# Mock S3: the child opens its OWN connection, the parent's survives
# =============================================================================================
def _mock_s3(port, store):
    """A tiny HTTP/1.1 keep-alive S3: GET/HEAD/PUT objects and multipart upload. `log` has
    (method, path-with-query, client port, body length) per request."""
    import http.server
    import socketserver
    import urllib.parse

    log = []
    parts = {}

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _body(self):
            if "chunked" in self.headers.get("Transfer-Encoding", ""):
                out = b""
                while True:
                    n = int(self.rfile.readline().strip().split(b";")[0], 16)
                    if n == 0:
                        while self.rfile.readline().strip():
                            pass
                        return out
                    out += self.rfile.read(n)
                    self.rfile.readline()
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _send(self, code, body=b"", extra=None):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _get(self, with_body):
            log.append((self.command, self.path, self.client_address[1], 0))
            b = store.get(self.path.split("?")[0])
            if b is None:
                return self._send(404)
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("ETag", '"x"')
            self.end_headers()
            if with_body:
                self.wfile.write(b)

        def do_GET(self):
            self._get(True)

        def do_HEAD(self):
            self._get(False)

        def do_PUT(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            b = self._body()
            log.append(("PUT", self.path, self.client_address[1], len(b)))
            if "partNumber" in q:
                parts[int(q["partNumber"][0])] = b
            else:
                store[u.path] = b
            self._send(200, extra={"ETag": '"p%s"' % q.get("partNumber", ["0"])[0]})

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query, keep_blank_values=True)
            b = self._body()
            log.append(("POST", self.path, self.client_address[1], len(b)))
            if "uploads" in q:
                self._send(200, b"<?xml version='1.0'?><InitiateMultipartUploadResult><Bucket>bkt</Bucket>"
                                b"<Key>k</Key><UploadId>UP1</UploadId></InitiateMultipartUploadResult>")
            else:
                store[u.path] = b"".join(parts[k] for k in sorted(parts))
                self._send(200, b"<?xml version='1.0'?><CompleteMultipartUploadResult><Location>x</Location>"
                                b"<Bucket>bkt</Bucket><Key>k</Key><ETag>\"c\"</ETag></CompleteMultipartUploadResult>")

        def do_DELETE(self):
            log.append(("DELETE", self.path, self.client_address[1], 0))
            self._send(204)

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = S(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log


def _mock_env(port):
    return {
        "AWS_ENDPOINT_URL": f"http://127.0.0.1:{port}",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if pytest is not None:

    def test_the_child_opens_its_own_connection_and_the_parent_keeps_its_own():
        port = _free_port()
        store = {f"/bkt/{k}": b"hello" * 1000 for k in ("p1", "p2", "c1", "c2", "p3", "p4")}
        srv, log = _mock_s3(port, store)
        try:
            code = """
import os, sys, time, signal
import s3dlio
def get(k): return len(memoryview(s3dlio.get("s3://bkt/" + k)))
assert get("p1") == 5000 and get("p2") == 5000      # the parent's connection is now pooled
pid = os.fork()
if pid == 0:
    try:
        assert get("c1") == 5000 and get("c2") == 5000
    except BaseException as e:
        print("CHILD RAISED", type(e).__name__, e, flush=True); os._exit(1)
    os._exit(0)
t0 = time.time()
while time.time() - t0 < 60:
    p, st = os.waitpid(pid, os.WNOHANG)
    if p: break
    time.sleep(0.05)
else:
    os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); print("CHILD HUNG", flush=True); sys.exit(2)
assert os.WEXITSTATUS(st) == 0, st
assert get("p3") == 5000 and get("p4") == 5000      # after the child: still works
print("PARENT OK")
"""
            p = _run_script(code, timeout=120, env_extra=_mock_env(port))
            assert "PARENT OK" in p.stdout, (p.stdout[-1500:], p.stderr[-1500:])
            ports = {}
            for cmd, path, cport, _n in log:
                if cmd == "GET":
                    ports.setdefault(path.split("?")[0].split("/")[-1], set()).add(cport)
            assert all(len(v) == 1 for v in ports.values()), ports
            parent_ports = {next(iter(ports[k])) for k in ("p1", "p2", "p3", "p4")}
            child_ports = {next(iter(ports[k])) for k in ("c1", "c2")}
            assert len(parent_ports) == 1, f"the parent's connection must survive the child: {ports}"
            assert len(child_ports) == 1, ports
            assert parent_ports.isdisjoint(child_ports), (
                f"the child must open its own connection, not reuse the parent's: {ports}"
            )
        finally:
            srv.shutdown()



    def test_an_inherited_multipart_writer_is_refused_and_never_aborts_the_parents_upload():
        """The multipart writer's drop aborts its upload. In a fork child that would abort the
        PARENT's upload: the child must refuse the writer, never drop it, and the parent must
        still complete the upload."""
        port = _free_port()
        store = {}
        srv, log = _mock_s3(port, store)
        try:
            code = """
import os, sys, gc, time, signal
import s3dlio
MIB = 1 << 20
w = s3dlio.MultipartUploadWriter.from_uri("s3://bkt/big.bin", part_size=5 * MIB, max_in_flight=2)
w.write(b"a" * (6 * MIB))                       # an upload is open: parts sent, not completed
pid = os.fork()
if pid == 0:
    res = []
    for call in (lambda: w.write(b"b"), lambda: w.abort(), lambda: w.close()):
        try:
            call(); res.append("NO ERROR")
        except RuntimeError as e:
            res.append("fork" if "created before fork()" in str(e) else "other: " + str(e))
    del w; gc.collect()
    print("CHILD", res, flush=True)
    sys.exit(0 if res == ["fork"] * 3 else 4)    # normal interpreter exit: drops, atexit
t0 = time.time()
while time.time() - t0 < 60:
    p, st = os.waitpid(pid, os.WNOHANG)
    if p: break
    time.sleep(0.05)
else:
    os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); print("CHILD HUNG"); sys.exit(2)
assert os.WEXITSTATUS(st) == 0, st
meta = w.close()                                  # the parent completes ITS upload
assert meta["total_bytes"] == 6 * MIB, meta
print("PARENT OK", meta["parts"])
"""
            p = _run_script(code, timeout=120, env_extra=_mock_env(port))
            assert "PARENT OK" in p.stdout, (p.stdout[-1500:], p.stderr[-1500:])
            assert not [e for e in log if e[0] == "DELETE"], f"the parent's upload was aborted: {log}"
            assert len(store["/bkt/big.bin"]) == 6 << 20
        finally:
            srv.shutdown()

# =============================================================================================
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "HARNESS":
        _harness_main(sys.argv[2], HANG_S, only=set(sys.argv[3:]) or None)
    else:
        sys.exit("usage: test_fork_297.py HARNESS WARM [CASE ...]  (or run it with pytest)")
