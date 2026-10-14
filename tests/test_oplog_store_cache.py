"""Op-log vs the Python store cache: s3dlio issue #174.

The module-level functions share a process-wide cache of stores. Whether a call
is recorded must depend only on whether the op-log is ACTIVE when the call is
made: never on which call populated the cache first (``get_async`` used to build
the cached store without the logger and silence every later call), on an
``init_op_log`` that came after the cache was populated, or on a logger that has
since been finalized.

Every scenario runs in a fresh interpreter (the op-log is process-global) and
uses ``file://`` URIs, so no server is needed and no S3 backend can hide a
missing row. Run it against the wheel under test:
``uv run --no-project --with <wheel> --with pytest pytest tests/test_oplog_store_cache.py``.
"""

import textwrap

from test_oplog_lifecycle import assert_ok, read_rows, run_scenario

PRELUDE = """
    oplog, work = sys.argv[1:3]
    import asyncio
    uri = "file://" + work + "/f.bin"
    with open(work + "/f.bin", "wb") as f:
        f.write(b"0123456789")

    async def ga():
        return await s3dlio.get_async(uri)

    async def gra(offset, length):
        return await s3dlio.get_range_async(uri, offset, length)
"""


def run(tmp_path, body):
    work = tmp_path / "work"
    work.mkdir()
    oplog = tmp_path / "ops.tsv.zst"
    proc = run_scenario(
        textwrap.dedent(PRELUDE) + textwrap.dedent(body), str(oplog), str(work)
    )
    assert_ok(proc)
    return proc, oplog


def test_get_async_first_does_not_silence_later_calls(tmp_path):
    """The issue's reproduction: get_async, get, stat on one URI -> three rows."""
    _, oplog = run(
        tmp_path,
        """
        s3dlio.init_op_log(oplog)
        asyncio.run(ga())
        s3dlio.get(uri)
        s3dlio.stat(uri)
        s3dlio.finalize_op_log()
        """,
    )
    rows = read_rows(oplog)
    assert [r["op"] for r in rows] == ["GET", "GET", "HEAD"], rows
    assert all(r["endpoint"] == "file://" for r in rows)


def test_get_range_async_first_does_not_silence_later_calls(tmp_path):
    _, oplog = run(
        tmp_path,
        """
        s3dlio.init_op_log(oplog)
        asyncio.run(gra(2, 4))
        s3dlio.get_range(uri, 2, 4)
        s3dlio.stat(uri)
        s3dlio.finalize_op_log()
        """,
    )
    rows = read_rows(oplog)
    assert [r["op"] for r in rows] == ["GET_RANGE", "GET_RANGE", "HEAD"], rows


def test_each_call_is_recorded_once_whatever_the_cache_history(tmp_path):
    """Sync first, async first and mixed orders all give exactly one row per call."""
    _, oplog = run(
        tmp_path,
        """
        s3dlio.init_op_log(oplog)
        s3dlio.stat(uri)
        asyncio.run(ga())
        s3dlio.get_range(uri, 0, 3)
        asyncio.run(gra(1, 3))
        s3dlio.get(uri)
        s3dlio.finalize_op_log()
        """,
    )
    rows = read_rows(oplog)
    assert [r["op"] for r in rows] == ["HEAD", "GET", "GET_RANGE", "GET_RANGE", "GET"], rows


def test_calls_made_before_init_do_not_hide_calls_made_after(tmp_path):
    """A store cached before init_op_log has no logger; init must still take effect."""
    proc, oplog = run(
        tmp_path,
        """
        s3dlio.get(uri)                       # populates the cache, op-log inactive
        asyncio.run(ga())    # inactive: not recorded
        s3dlio.init_op_log(oplog)
        s3dlio.get(uri)                       # active: recorded
        asyncio.run(ga())    # active: recorded
        s3dlio.stat(uri)                      # active: recorded
        s3dlio.finalize_op_log()
        """,
    )
    rows = read_rows(oplog)
    assert [r["op"] for r in rows] == ["GET", "GET", "HEAD"], rows


def test_calls_after_finalize_are_not_recorded_and_not_counted(tmp_path):
    """A cached store must not keep a dead logger: nothing is recorded or counted after finalize."""
    proc, oplog = run(
        tmp_path,
        """
        s3dlio.init_op_log(oplog)
        s3dlio.get(uri)
        asyncio.run(ga())
        s3dlio.finalize_op_log()
        before = s3dlio.op_log_stats()
        s3dlio.get(uri)
        asyncio.run(ga())
        s3dlio.stat(uri)
        after = s3dlio.op_log_stats()
        assert before == after, (before, after)
        assert after["rejected_closed"] == 0, after
        """,
    )
    rows = read_rows(oplog)
    assert [r["op"] for r in rows] == ["GET", "GET"], rows
    assert "no longer recorded" not in proc.stderr, proc.stderr


def test_calls_racing_init_and_finalize_are_each_recorded_at_most_once(tmp_path):
    """Threads hammer get()/get_async() while the op-log is initialised and finalized."""
    proc, oplog = run(
        tmp_path,
        """
        import threading, time
        stop = threading.Event()
        counts = []

        def hammer(use_async):
            n = 0
            while not stop.is_set():
                if use_async:
                    asyncio.run(ga())
                else:
                    s3dlio.get(uri)
                n += 1
            counts.append(n)

        threads = [threading.Thread(target=hammer, args=(i % 2 == 1,)) for i in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        s3dlio.init_op_log(oplog)
        time.sleep(0.3)
        s3dlio.finalize_op_log()
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join()
        stats = s3dlio.op_log_stats()
        print(stats["accepted"], stats["written"], sum(counts))
        """,
    )
    accepted, written, total = (int(x) for x in proc.stdout.split())
    rows = read_rows(oplog)
    assert len(rows) == written
    # Only calls made while active are recorded: never more rows than calls, and
    # some calls (before init and after finalize) are not rows.
    assert 0 < len(rows) < total
    assert all(r["op"] == "GET" for r in rows)
    assert accepted == written
