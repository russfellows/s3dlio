"""Op-log lifecycle PERMUTATION matrix for s3dlio issue #294 (round 2).

Systematic coverage of starting and stopping op-logging from single-threaded,
multi-threaded, asyncio and multi-process Python programs. The op-log is
process-global, so every scenario runs in a FRESH interpreter (see
``run_scenario`` in ``test_oplog_lifecycle.py``); a scenario that deadlocks is
reported with the stack of every thread (``faulthandler`` watchdog in the child)
instead of hanging the suite. Operations use ``file://`` URIs in a tmp dir, so no
server is needed.

Behaviour pinned here (decided for #294):

* init while Active with the SAME file (any spelling, any link to it): no-op.
* init while Active with a DIFFERENT file: ``RuntimeError`` naming the file being
  written and the requested path, saying to finalize first; nothing is created.
* init after finalize: a new session (waits for the old writer to drain).
* a forked child never shares the parent's session (it sees the op-log as not
  initialised) and cannot hang on it or corrupt the parent's file.

Multi-threaded scenarios run in the default (lossy) mode and with
``S3DLIO_OPLOG_LOSSLESS=1``.
"""

import textwrap

import pytest
from test_oplog_lifecycle import HEADER, assert_ok, read_rows, run_scenario

PRE = f'''
import faulthandler, threading, subprocess, shutil, time, random, asyncio, json, signal
faulthandler.dump_traceback_later(int(os.environ["SCEN_WATCHDOG"]), exit=True)
T = sys.argv[1]
HEADER = {HEADER!r}
ZSTD = shutil.which("zstd")


def u(p):
    return "file://" + p


def ztest(p):
    r = subprocess.run([ZSTD, "-t", p], capture_output=True)
    assert r.returncode == 0, p + " is not a complete zstd stream: " + r.stderr.decode()


def rows(p):
    """Complete zstd stream, exact header, 13 columns, sequential idx -> [(op, file)]."""
    ztest(p)
    out = subprocess.run([ZSTD, "-dc", p], capture_output=True, check=True).stdout.decode()
    lines = out.split("\\n")
    assert lines[-1] == "" and lines[0] == HEADER, (p, lines[:2])
    res = []
    for i, line in enumerate(lines[1:-1]):
        c = line.split("\\t")
        assert len(c) == 13 and c[0] == str(i), (p, i, line)
        res.append((c[2], c[7]))
    return res


def run_threads(fn, n, limit=90):
    """Run fn(i) on n threads; return [("ok", value) | ("err", exception)]. A thread that
    does not finish within `limit` seconds is a DEADLOCK."""
    out = [None] * n

    def body(i):
        try:
            out[i] = ("ok", fn(i))
        except BaseException as e:  # noqa: BLE001 - reported to the caller
            out[i] = ("err", e)

    ts = [threading.Thread(target=body, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    end = time.time() + limit
    for t in ts:
        t.join(max(0.1, end - time.time()))
        assert not t.is_alive(), "DEADLOCK: a thread is still running after %ds" % limit
    return out
'''


@pytest.fixture(params=["lossy", "lossless"])
def mode(request):
    return request.param


def run(body, tmp_path, mode="lossy", timeout=180, cwd=None, extra_env=None):
    env = dict(extra_env or {})
    env["SCEN_WATCHDOG"] = str(timeout - 20)
    if mode == "lossless":
        env["S3DLIO_OPLOG_LOSSLESS"] = "1"
    proc = run_scenario(
        PRE + textwrap.dedent(body), str(tmp_path), timeout=timeout, cwd=cwd, env=env
    )
    assert_ok(proc)
    return proc


def ops(path):
    return [r["op"] for r in read_rows(path)]


# --------------------------------------------------------------------------
# Single thread (matrix items 1-7)
# --------------------------------------------------------------------------


def test_01_init_same_file_while_active_is_a_noop(tmp_path):
    run(
        """
        a, f = T + "/a.tsv.zst", T + "/f.bin"
        s3dlio.init_op_log(a)
        s3dlio.put_bytes(u(f), b"x")
        s3dlio.init_op_log(a)                      # no-op: must not truncate
        assert s3dlio.is_op_log_active() is True
        s3dlio.get(u(f))                           # the session continues
        assert s3dlio.op_log_stats()["accepted"] == 2   # still the one session
        s3dlio.finalize_op_log()
        """,
        tmp_path,
    )
    assert ops(tmp_path / "a.tsv.zst") == ["PUT", "GET"]


def test_02_init_different_file_while_active_is_an_error_naming_both(tmp_path):
    run(
        """
        a, b, f = T + "/a.tsv.zst", T + "/b.tsv.zst", T + "/f.bin"
        s3dlio.init_op_log(a)
        s3dlio.put_bytes(u(f), b"x")
        try:
            s3dlio.init_op_log(b)
        except RuntimeError as e:
            msg = str(e)
        else:
            raise AssertionError("init with a different file while Active must raise")
        assert a in msg and b in msg and "finalize" in msg, msg
        assert not os.path.exists(b), "the requested file must not be created"
        assert s3dlio.is_op_log_active() is True
        s3dlio.get(u(f))                           # session A continues unaffected
        assert s3dlio.op_log_stats()["accepted"] == 2
        s3dlio.finalize_op_log()
        assert not os.path.exists(b)
        """,
        tmp_path,
    )
    assert ops(tmp_path / "a.tsv.zst") == ["PUT", "GET"]
    assert not (tmp_path / "b.tsv.zst").exists()


def test_03_same_file_in_every_spelling_is_a_noop_other_files_error(tmp_path):
    run(
        """
        os.mkdir("sub")
        f = T + "/f.bin"
        s3dlio.init_op_log("x")                    # relative: creates ./x
        s3dlio.put_bytes(u(f), b"1")
        os.symlink(T + "/x", "link_to_x")
        os.symlink(T + "/link_to_x", "sub/chain")
        os.link("x", "hard_to_x")
        same = ["x", "./x", T + "/x", "file://" + T + "/x", "sub/../x", T + "/sub/../x",
                "link_to_x", "sub/chain", "hard_to_x", "file://" + T + "/link_to_x"]
        for s in same:
            s3dlio.init_op_log(s)                  # must not raise, must not truncate
        s3dlio.get(u(f))
        assert s3dlio.op_log_stats()["accepted"] == 2
        with open("other", "wb") as fh:
            fh.write(b"precious")
        os.symlink(T + "/other", "link_to_other")
        for d in ["y", "./sub/x", "x2", "sub/y", "no_such_dir/x", "other",
                  "link_to_other", "file://" + T + "/y"]:
            try:
                s3dlio.init_op_log(d)
            except RuntimeError as e:
                assert T + "/x" in str(e), (d, str(e))
            else:
                raise AssertionError(repr(d) + " is a different file: must raise")
        for d in ["file://", "s3://bucket/key"]:   # unusable destinations error too
            try:
                s3dlio.init_op_log(d)
            except RuntimeError:
                pass
            else:
                raise AssertionError(repr(d) + " must raise")
        for gone in ["y", "sub/x", "x2", "sub/y"]:
            assert not os.path.exists(gone), gone
        assert open("other", "rb").read() == b"precious"
        s3dlio.finalize_op_log()
        """,
        tmp_path,
        cwd=tmp_path,
    )
    assert ops(tmp_path / "x") == ["PUT", "GET"]


def test_04_finalize_before_init_twice_and_after_failed_init_are_noops(tmp_path):
    run(
        """
        s3dlio.finalize_op_log()                   # before any init
        s3dlio.finalize_op_log()
        assert s3dlio.is_op_log_active() is False and s3dlio.op_log_stats() is None
        try:
            s3dlio.init_op_log("s3://bucket/key")  # fails: nothing started
        except RuntimeError:
            pass
        else:
            raise AssertionError("must fail")
        s3dlio.finalize_op_log()                   # after a failed init
        s3dlio.finalize_op_log()
        assert s3dlio.is_op_log_active() is False and s3dlio.op_log_stats() is None
        assert os.listdir(T) == [], os.listdir(T)  # no file anywhere
        # and a session still works afterwards, then finalize twice
        s3dlio.init_op_log(T + "/a.tsv.zst")
        s3dlio.finalize_op_log()
        s3dlio.finalize_op_log()
        """,
        tmp_path,
    )
    assert ops(tmp_path / "a.tsv.zst") == []


def test_05_reinit_same_path_replaces_different_path_leaves_old_untouched(tmp_path):
    run(
        """
        a, b, f = T + "/a.tsv.zst", T + "/b.tsv.zst", T + "/f.bin"
        s3dlio.init_op_log(a)
        s3dlio.put_bytes(u(f), b"x")
        s3dlio.get(u(f))
        s3dlio.finalize_op_log()
        assert len(rows(a)) == 2
        s3dlio.init_op_log(a)                      # same path: REPLACED
        s3dlio.stat(u(f))
        s3dlio.finalize_op_log()
        assert rows(a) == [("HEAD", f)], rows(a)
        before = open(a, "rb").read()
        s3dlio.init_op_log(b)                      # different path: a untouched
        s3dlio.get(u(f))
        s3dlio.finalize_op_log()
        assert open(a, "rb").read() == before, "a must be byte-identical"
        assert rows(b) == [("GET", f)]
        """,
        tmp_path,
    )


def test_06_bad_path_from_uninitialized_and_finalized_changes_nothing(tmp_path):
    run(
        """
        f = T + "/f.bin"
        ro = T + "/ro"
        os.mkdir(ro)
        os.chmod(ro, 0o500)
        can_chmod = os.geteuid() != 0 and not os.access(ro, os.W_OK)
        bad = [T + "/missing_dir/x.tsv.zst"] + ([ro + "/x.tsv.zst"] if can_chmod else [])

        def must_fail(path, stats_before):
            try:
                s3dlio.init_op_log(path)
            except RuntimeError as e:
                assert path in str(e), str(e)
            else:
                raise AssertionError(path + " must fail")
            assert s3dlio.is_op_log_active() is False
            assert s3dlio.op_log_stats() == stats_before

        for p in bad:                               # from Uninitialized
            must_fail(p, None)
        a = T + "/a.tsv.zst"
        s3dlio.init_op_log(a)                       # a good init still works
        s3dlio.put_bytes(u(f), b"x")
        s3dlio.finalize_op_log()
        stats = s3dlio.op_log_stats()
        assert stats["accepted"] == 1
        for p in bad:                               # from Finalized: stats unchanged
            must_fail(p, stats)
        s3dlio.init_op_log(T + "/b.tsv.zst")        # and a good init works
        assert s3dlio.is_op_log_active() is True
        s3dlio.finalize_op_log()
        os.chmod(ro, 0o700)
        print("permission case:", can_chmod)
        """,
        tmp_path,
    )
    assert ops(tmp_path / "a.tsv.zst") == ["PUT"]
    assert ops(tmp_path / "b.tsv.zst") == []


def test_07_five_hundred_cycles_leak_no_fds_or_threads(tmp_path):
    run(
        """
        f = T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")

        def cycle(i):
            name = "same" if i % 4 < 2 else ("other" if i % 4 == 3 else "d%d" % i)
            p = T + "/c_" + name + ".tsv.zst"
            s3dlio.init_op_log(p)
            s3dlio.get(u(f))
            s3dlio.finalize_op_log()
            return p

        def count(d):
            return len(os.listdir(d))

        cycle(0)                                    # warm-up: runtime threads/fds exist now
        fds, threads = count("/proc/self/fd"), count("/proc/self/task")
        paths = {cycle(i) for i in range(1, 501)}
        deadline = time.time() + 5
        while count("/proc/self/task") > threads + 2 and time.time() < deadline:
            time.sleep(0.02)
        # Tolerance for threads: the shared tokio runtime may grow/shrink its
        # blocking pool by a thread or two; a leak would add one per cycle (500).
        assert count("/proc/self/fd") <= fds, (count("/proc/self/fd"), fds)
        assert count("/proc/self/task") <= threads + 2, (count("/proc/self/task"), threads)
        for p in paths:
            assert rows(p) == [("GET", f)], p
        """,
        tmp_path,
    )


# --------------------------------------------------------------------------
# Multi-threaded (matrix items 8-11, 13, 14; each in lossy and lossless mode)
# --------------------------------------------------------------------------


def test_08a_sixteen_threads_init_same_file(tmp_path, mode):
    run(
        """
        a, f = T + "/a.tsv.zst", T + "/f.bin"
        barrier = threading.Barrier(16)

        def go(i):
            barrier.wait()
            s3dlio.init_op_log(a)

        res = run_threads(go, 16)
        assert [r[0] for r in res] == ["ok"] * 16, res
        assert os.listdir(T) == ["a.tsv.zst"]
        s3dlio.put_bytes(u(f), b"x")
        s3dlio.finalize_op_log()
        assert rows(a) == [("PUT", f)]
        """,
        tmp_path,
        mode,
    )


def test_08b_sixteen_threads_init_different_files_one_winner(tmp_path, mode):
    run(
        """
        barrier = threading.Barrier(16)
        paths = [T + "/f%d.tsv.zst" % i for i in range(16)]

        def go(i):
            barrier.wait()
            s3dlio.init_op_log(paths[i])
            return paths[i]

        res = run_threads(go, 16)
        winners = [r[1] for r in res if r[0] == "ok"]
        assert len(winners) == 1, res
        losers = [(paths[i], r[1]) for i, r in enumerate(res) if r[0] == "err"]
        assert len(losers) == 15
        for mine, e in losers:
            assert isinstance(e, RuntimeError), repr(e)
            assert winners[0] in str(e) and mine in str(e), str(e)
        assert os.listdir(T) == [os.path.basename(winners[0])], os.listdir(T)
        s3dlio.finalize_op_log()
        assert rows(winners[0]) == []
        """,
        tmp_path,
        mode,
    )


def test_09_concurrent_finalize_returns_only_after_the_file_is_complete(tmp_path, mode):
    run(
        """
        a, f = T + "/a.tsv.zst", T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        for rep in range(10):
            s3dlio.init_op_log(a)
            for _ in range(300):
                s3dlio.get(u(f))
            barrier = threading.Barrier(16)

            def go(i):
                barrier.wait()
                s3dlio.finalize_op_log()
                ztest(a)                            # complete the instant finalize returns
                assert s3dlio.is_op_log_active() is False
                return s3dlio.op_log_stats()["written"]

            res = run_threads(go, 16)
            assert [r[0] for r in res] == ["ok"] * 16, res
            assert {r[1] for r in res} == {300}, res
            assert len(rows(a)) == 300
        """,
        tmp_path,
        mode,
    )


@pytest.mark.parametrize(
    "same_path", [True, False], ids=["same_path", "different_path"]
)
def test_10_init_racing_finalize_never_deadlocks_and_ends_consistent(
    tmp_path, mode, same_path
):
    run(
        f"""
        same = {same_path!r}
        f = T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        outcomes = set()
        for it in range(60):
            a = T + "/a%d.tsv.zst" % it
            p = a if same else T + "/p%d.tsv.zst" % it
            s3dlio.init_op_log(a)
            s3dlio.get(u(f))
            stagger_fin, stagger_init = [(0, 0), (0.002, 0), (0, 0.002)][it % 3]
            barrier = threading.Barrier(2)

            def go(i):
                barrier.wait()
                if i == 0:
                    time.sleep(stagger_fin)
                    s3dlio.finalize_op_log()
                    return None
                time.sleep(stagger_init)
                try:
                    s3dlio.init_op_log(p)
                    return "ok"
                except RuntimeError as e:
                    return str(e)

            res = run_threads(go, 2)
            assert res[0][0] == "ok", res
            assert res[1][0] == "ok", res
            r = res[1][1]
            active = s3dlio.is_op_log_active()
            if same:
                assert r == "ok", r                 # no-op, or a new session on the same file
            elif r == "ok":
                assert active, "init won after finalize: the new session must be Active"
            else:
                assert a in r and p in r and "finalize" in r, r
                assert not active and not os.path.exists(p), (active, p)
            outcomes.add((r == "ok", active))
            s3dlio.finalize_op_log()
            assert s3dlio.is_op_log_active() is False
            for q in {{a, p}}:
                if os.path.exists(q):
                    rows(q)                         # every file that exists is complete
        print("outcomes", sorted(outcomes))
        """,
        tmp_path,
        mode,
    )


def test_10b_init_returns_only_after_the_previous_writer_has_drained(tmp_path, mode):
    """The drain wait: while a finalize on another thread is still draining a big
    backlog, init (same path or another path) must not return until the old file is a
    complete zstd stream, whatever the thread timing. Checked WITHOUT joining the
    finalizing thread."""
    run(
        """
        f = T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        for rnd in range(12):
            a = T + "/a%d.tsv.zst" % rnd
            nxt = a if rnd % 2 else T + "/n%d.tsv.zst" % rnd
            s3dlio.init_op_log(a)
            for _ in range(4000):
                s3dlio.get(u(f))
            fin = threading.Thread(target=s3dlio.finalize_op_log)
            fin.start()
            while s3dlio.is_op_log_active():        # finalize has swapped the state
                pass
            s3dlio.init_op_log(nxt)                 # must wait for the drain
            if nxt != a:
                assert len(rows(a)) == 4000, rnd    # complete NOW (fin not joined)
            fin.join(60)
            assert not fin.is_alive(), "DEADLOCK"
            assert s3dlio.is_op_log_active() is True, "the finalizer closed the new session"
            s3dlio.finalize_op_log()
            rows(nxt)
        """,
        tmp_path,
        mode,
    )


def test_11_workers_run_while_main_cycles_init_finalize(tmp_path, mode):
    """Worker threads (sync module-level ops plus an asyncio thread) run flat out
    while the main thread cycles init/finalize 50 times over alternating and
    repeated paths. What can be asserted EXACTLY, per session g (the main thread
    publishes phases OPEN/CLOSING/CLOSED/INITING and a worker classifies an op by
    reading the phase before and after it):

    * the file is a complete zstd stream, exact header, sequential idx, and its
      row count equals stats["written"] == stats["accepted"] (read right after
      finalize, when the counters are final for every call that began before the
      close);
    * accepted + dropped_full >= the ops that began AND ended in phase OPEN(g)
      ("surely inside": the logger snapshot is taken during the call, finalize had
      not begun) - lossless: dropped_full == 0;
    * no (op, file) row appears twice across all files, and every row is an op a
      worker really issued;
    * overall: sum(accepted + dropped_full + rejected_closed) <= ops issued minus
      the ops that began and ended in phase CLOSED ("surely outside": they cannot
      be logged at all).
    rejected_closed is only a lower bound for calls that lost the race with
    finalize (a call can reach a closed logger after the stats were read), so no
    equality is claimed for ops overlapping a transition: that window is the
    documented 'finalize does not stop a call already holding the logger'.
    """
    proc = run(
        """
        W, K = 6, 50
        base = T + "/work"
        os.mkdir(base)
        shared = base + "/shared"
        with open(shared, "wb") as fh:
            fh.write(b"s")
        ph = [("CLOSED", 0)]
        stop = threading.Event()
        lock = threading.Lock()
        sure_in, sure_out, issued, issued_keys = {}, [0], [0], set()

        def account(before, after):
            with lock:
                issued[0] += 1
                if before == after:
                    if before[0] == "OPEN":
                        sure_in[before[1]] = sure_in.get(before[1], 0) + 1
                    elif before[0] == "CLOSED":
                        sure_out[0] += 1

        def worker(w):
            n = 0
            while not stop.is_set():
                p = base + "/w%d_%d" % (w, n)
                uri = u(p)
                for name, fn in (("PUT", lambda: s3dlio.put_bytes(uri, b"d")),
                                 ("GET", lambda: s3dlio.get(uri)),
                                 ("HEAD", lambda: s3dlio.stat(uri)),
                                 ("LIST", lambda: s3dlio.list(uri))):
                    before = ph[0]
                    fn()
                    account(before, ph[0])
                    with lock:
                        issued_keys.add((name, p))
                n += 1

        def aworker():
            async def go():
                while not stop.is_set():
                    before = ph[0]
                    await s3dlio.get_async(u(shared))
                    account(before, ph[0])
                    before = ph[0]
                    await s3dlio.stat_async(u(shared))
                    account(before, ph[0])
            asyncio.run(go())

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(W)]
        threads.append(threading.Thread(target=aworker))
        for t in threads:
            t.start()
        seq = [T + "/s_%s.tsv.zst" % c for c in "AABCCABBA"]
        sessions = []
        for i in range(K):
            g, path = i + 1, seq[i % len(seq)]
            ph[0] = ("INITING", g)
            s3dlio.init_op_log(path)
            ph[0] = ("OPEN", g)
            time.sleep(random.uniform(0.003, 0.015))
            ph[0] = ("CLOSING", g)
            s3dlio.finalize_op_log()
            ph[0] = ("CLOSED", g)
            sessions.append((g, path, s3dlio.op_log_stats(), rows(path)))
            time.sleep(random.choice([0, 0, 0.001, 0.003]))
        stop.set()
        for t in threads:
            t.join(60)
            assert not t.is_alive(), "DEADLOCK: worker did not stop"

        seen = {}
        total_rows = total_outcomes = 0
        for g, path, st, rs in sessions:
            assert st["writer_error"] is None and st["writer_failed"] == 0, st
            assert st["written"] == len(rs) == st["accepted"], (g, st, len(rs))
            assert st["accepted"] + st["dropped_full"] >= sure_in.get(g, 0), (g, st, sure_in.get(g))
            if st["lossless"]:
                assert st["dropped_full"] == 0, st
            total_rows += len(rs)
            total_outcomes += st["accepted"] + st["dropped_full"] + st["rejected_closed"]
            for key in rs:
                if key[1] != shared:
                    assert key not in seen, ("row duplicated across files", key, g, seen[key])
                    assert key in issued_keys, ("row no worker issued", key)
                    seen[key] = g
        assert total_outcomes <= issued[0] - sure_out[0], (total_outcomes, issued[0], sure_out[0])
        assert total_rows > 0 and sum(sure_in.values()) > 0
        print("sessions", K, "rows", total_rows, "issued", issued[0],
              "surely_in", sum(sure_in.values()), "surely_out", sure_out[0],
              "outcomes", total_outcomes)
        """,
        tmp_path,
        mode,
        timeout=300,
    )
    print(proc.stdout)


def test_12_awaitable_created_in_session_1_writes_nothing_into_session_2(
    tmp_path, mode
):
    """Python has no user-built store that captures the logger (the module-level
    calls snapshot it per call); the closest object is an async call created in
    session 1 and awaited later. It must never write into session 2, while
    module-level calls made in session 2 are logged there. (The Rust-level
    store-capture case is tests/oplog_lifecycle_reinit_old_store.rs.)"""
    run(
        """
        s1, s2 = T + "/s1.tsv.zst", T + "/s2.tsv.zst"
        f1, f2 = T + "/f1.bin", T + "/f2.bin"
        for f in (f1, f2):
            with open(f, "wb") as fh:
                fh.write(b"x")

        async def main():
            s3dlio.init_op_log(s1)
            old = s3dlio.get_async(u(f1))           # created (and its logger snapshot taken) in session 1
            s3dlio.finalize_op_log()
            s3dlio.init_op_log(s2)
            s3dlio.get(u(f2))                       # a session-2 call
            await old
            s3dlio.stat(u(f2))
            s3dlio.finalize_op_log()

        asyncio.run(main())
        r2 = rows(s2)
        assert r2 == [("GET", f2), ("HEAD", f2)], r2
        assert all(f != f1 for _, f in r2)
        rows(s1)
        """,
        tmp_path,
        mode,
    )


def test_13_stats_and_active_hammered_while_cycling(tmp_path, mode):
    run(
        """
        stop = threading.Event()
        f = T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        KEYS = {"accepted", "dropped_full", "rejected_closed", "writer_failed", "lossless",
                "written", "writer_error", "rejected_unformattable",
                "rejected_unsafe_field", "normalized_error_text"}

        def hammer(i):
            seen_stats = False
            n = 0
            while not stop.is_set():
                a = s3dlio.is_op_log_active()
                assert a is True or a is False, a
                st = s3dlio.op_log_stats()
                if st is not None:
                    seen_stats = True
                    assert set(st) == KEYS, st
                    assert all(isinstance(st[k], int) and st[k] >= 0 for k in KEYS
                               if k not in ("lossless", "writer_error")), st
                else:
                    assert not seen_stats, "stats went back to None after a session existed"
                n += 1
            return n

        ts = []
        results = [None] * 6

        def body(i):
            try:
                results[i] = ("ok", hammer(i))
            except BaseException as e:
                results[i] = ("err", e)

        ts = [threading.Thread(target=body, args=(i,)) for i in range(6)]
        for t in ts:
            t.start()
        paths = [T + "/a.tsv.zst", T + "/a.tsv.zst", T + "/b.tsv.zst"]
        for i in range(120):
            p = paths[i % 3]
            s3dlio.init_op_log(p)
            s3dlio.get(u(f))
            s3dlio.finalize_op_log()
        stop.set()
        for t in ts:
            t.join(60)
            assert not t.is_alive(), "DEADLOCK"
        assert all(r[0] == "ok" and r[1] > 0 for r in results), results
        rows(T + "/a.tsv.zst")
        rows(T + "/b.tsv.zst")
        """,
        tmp_path,
        mode,
    )


def test_14_asyncio_loop_runs_while_another_thread_cycles(tmp_path, mode):
    run(
        """
        f = T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        stop = threading.Event()
        done = []

        def loop_thread():
            async def go():
                n = 0
                while not stop.is_set():
                    assert bytes(await s3dlio.get_async(u(f))) == b"x"
                    await s3dlio.stat_async(u(f))
                    await s3dlio.put_bytes_async(u(T + "/g%d.bin" % (n % 5)), b"y")
                    n += 1
                return n
            done.append(asyncio.run(go()))

        errs = []

        def guarded():
            try:
                loop_thread()
            except BaseException as e:
                errs.append(e)

        t = threading.Thread(target=guarded)
        t.start()
        paths = [T + "/a.tsv.zst", T + "/b.tsv.zst", T + "/a.tsv.zst"]
        for i in range(40):
            s3dlio.init_op_log(paths[i % 3])
            time.sleep(0.004)
            s3dlio.finalize_op_log()
            rows(paths[i % 3])
        stop.set()
        t.join(60)
        assert not t.is_alive(), "DEADLOCK"
        assert not errs, errs
        assert done and done[0] > 0
        """,
        tmp_path,
        mode,
    )


# --------------------------------------------------------------------------
# Process level (matrix items 16-17)
# --------------------------------------------------------------------------

FORK_CHILD = """
a, c, f = T + "/a.tsv.zst", T + "/c.tsv.zst", T + "/f.bin"
with open(f, "wb") as fh:                          # no s3dlio I/O before the fork: its
    fh.write(b"x")                                 # runtime threads do not survive fork
s3dlio.init_op_log(a)
pid = os.fork()
if pid == 0:
    code = 0
    try:
        assert s3dlio.is_op_log_active() is False, "child inherited an Active session"
        assert s3dlio.op_log_stats() is None
        for _ in range(500):                       # far more than the queue (S3DLIO_OPLOG_BUF=64)
            s3dlio.get(u(f))                       # must neither block nor be logged
        s3dlio.finalize_op_log()                   # must not wait for a writer that does not exist
        try:
            s3dlio.init_op_log(a)                  # the PARENT's file: refuse, never truncate it
        except RuntimeError as e:
            assert "PARENT" in str(e) and a in str(e), str(e)
        else:
            raise AssertionError("child init of the parent's file must raise")
        s3dlio.init_op_log(c)                      # its own new session works
        s3dlio.stat(u(f))
        s3dlio.finalize_op_log()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 3
    os._exit(code)

end = time.time() + 60
while True:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    if time.time() > end:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
        raise AssertionError("forked child HUNG (no exit within 60 s)")
    time.sleep(0.02)
assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
s3dlio.get(u(f))                                   # the parent's session is unaffected
s3dlio.finalize_op_log()
assert rows(a) == [("GET", f)], rows(a)
assert rows(c) == [("HEAD", f)], rows(c)
"""


def test_16a_forked_child_cannot_hang_or_corrupt_the_parents_log(tmp_path, mode):
    run(FORK_CHILD, tmp_path, mode, extra_env={"S3DLIO_OPLOG_BUF": "64"})


def test_16b_fork_while_another_thread_uses_the_oplog_never_deadlocks(tmp_path, mode):
    run(
        """
        f = T + "/f.bin"
        s3dlio.init_op_log(T + "/a.tsv.zst")
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                s3dlio.is_op_log_active()
                s3dlio.op_log_stats()

        ts = [threading.Thread(target=hammer) for _ in range(6)]
        for t in ts:
            t.start()
        for i in range(120):
            pid = os.fork()
            if pid == 0:
                try:
                    s3dlio.is_op_log_active()
                    s3dlio.op_log_stats()
                    s3dlio.finalize_op_log()
                finally:
                    os._exit(0)
            end = time.time() + 30
            while True:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    break
                if time.time() > end:
                    os.kill(pid, signal.SIGTERM)
                    os.waitpid(pid, 0)
                    raise AssertionError("child %d deadlocked on the lifecycle lock" % i)
                time.sleep(0.005)
            assert os.WEXITSTATUS(status) == 0
        stop.set()
        for t in ts:
            t.join(30)
        s3dlio.finalize_op_log()
        assert rows(T + "/a.tsv.zst") == []
        """,
        tmp_path,
        mode,
    )


def test_16c_spawned_child_starts_clean_and_can_run_its_own_session(tmp_path, mode):
    (tmp_path / "child_mod.py").write_text(
        textwrap.dedent(
            """
            import s3dlio


            def child(path, uri):
                assert s3dlio.is_op_log_active() is False
                assert s3dlio.op_log_stats() is None
                s3dlio.init_op_log(path)
                s3dlio.get(uri)
                s3dlio.finalize_op_log()
            """
        )
    )
    run(
        """
        import multiprocessing
        sys.path.insert(0, T)
        import child_mod
        a, c, f = T + "/a.tsv.zst", T + "/c.tsv.zst", T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        s3dlio.init_op_log(a)
        s3dlio.put_bytes(u(f), b"y")
        p = multiprocessing.get_context("spawn").Process(target=child_mod.child, args=(c, u(f)))
        p.start()
        p.join(90)
        assert not p.is_alive(), "spawned child hung"
        assert p.exitcode == 0, p.exitcode
        s3dlio.get(u(f))
        s3dlio.finalize_op_log()
        assert rows(a) == [("PUT", f), ("GET", f)]
        assert rows(c) == [("GET", f)]
        """,
        tmp_path,
        mode,
    )


def test_16d_multiprocessing_fork_context(tmp_path, mode):
    run(
        """
        import multiprocessing
        a, c, f = T + "/a.tsv.zst", T + "/c.tsv.zst", T + "/f.bin"
        with open(f, "wb") as fh:
            fh.write(b"x")
        s3dlio.init_op_log(a)

        def child():
            assert s3dlio.is_op_log_active() is False
            s3dlio.get(u(f))
            s3dlio.finalize_op_log()
            s3dlio.init_op_log(c)
            s3dlio.stat(u(f))
            s3dlio.finalize_op_log()

        p = multiprocessing.get_context("fork").Process(target=child)
        p.start()
        p.join(90)
        assert not p.is_alive(), "forked child hung"
        assert p.exitcode == 0, p.exitcode
        s3dlio.put_bytes(u(f), b"z")
        s3dlio.finalize_op_log()
        assert rows(a) == [("PUT", f)]
        assert rows(c) == [("HEAD", f)]
        """,
        tmp_path,
        mode,
    )


@pytest.mark.parametrize("busy", [False, True], ids=["idle", "workers_running"])
def test_17_exit_with_a_reinitialised_session_active_does_not_hang(
    tmp_path, mode, busy
):
    """No finalize: the interpreter must still exit promptly and cleanly (queued
    rows and the zstd frame end are lost, as documented for a missing finalize).

    The busy variant keeps daemon threads calling the op-log's own functions
    (the lifecycle lock) up to the exit. It deliberately does NOT keep daemon
    threads inside s3dlio I/O calls: that aborts the interpreter at exit
    ("FATAL: exception not rethrown") with or without an op-log, a pre-existing
    problem unrelated to #294 (reproduced with no init_op_log at all)."""
    run(
        f"""
        f = T + "/f.bin"
        s3dlio.put_bytes(u(f), b"x")
        s3dlio.init_op_log(T + "/a.tsv.zst")
        s3dlio.get(u(f))
        s3dlio.finalize_op_log()
        s3dlio.init_op_log(T + "/b.tsv.zst")        # re-init, never finalized
        for _ in range(100):
            s3dlio.get(u(f))
        if {busy!r}:
            def spin():
                while True:
                    s3dlio.is_op_log_active()
                    s3dlio.op_log_stats()
            for _ in range(4):
                threading.Thread(target=spin, daemon=True).start()
            time.sleep(0.05)
        assert s3dlio.is_op_log_active() is True
        """,
        tmp_path,
        mode,
        timeout=120,
    )
    assert (tmp_path / "b.tsv.zst").exists()
    assert ops(tmp_path / "a.tsv.zst") == ["GET"]
