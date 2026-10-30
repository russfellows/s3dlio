# tests/test_oplog_async_call_time_294.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""An async call binds to the op-log session that is active WHEN IT IS MADE (issue #294).

The contract: an awaitable created while op-log session 1 is active never writes a row into
a later session 2 (after ``finalize_op_log()`` + ``init_op_log()``), and an awaitable created
while no session is active is never logged, however late the runtime first polls it.

``test_12_awaitable_created_in_session_1_writes_nothing_into_session_2`` (in
``test_oplog_lifecycle_permutations_294.py``) tests this with real timing, so it only failed
about 1 run in 30. These tests make the race DETERMINISTIC with the hidden test hook
``s3dlio._hold_async_runtime()``: it occupies every worker thread of the async runtime and
returns only when all are occupied, so an awaitable created while the hold is active has been
spawned but provably cannot take its first poll until the hold is released. Everything between
the call and the release (``finalize_op_log()``, ``init_op_log(other)``) therefore happens
BEFORE the first poll, every time.

Each scenario runs in a fresh interpreter (the op-log is process-global) on ``file://`` URIs.
Two directions per entry point, plus a control call made in session 2 that must be logged
(so a broken logger cannot make the test pass vacuously).
"""

import shutil
import textwrap

import pytest
from test_oplog_lifecycle import HEADER, assert_ok, run_scenario

pytestmark = pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="the zstd command-line tool is not installed (e.g. `apt-get install zstd`)",
)

# name -> (python source of a function `call(p)` that CREATES the awaitable for path `p`,
#          whether the call writes an op-log row on file://). `p` is the object (or, for mkdir,
#          the directory) the call acts on; `u(p)` is its file:// URI.
ENTRIES = {
    "get_async": ("lambda p: s3dlio.get_async(u(p))", True),
    "get_range_async": ("lambda p: s3dlio.get_range_async(u(p), 0, 1)", True),
    "put_bytes_async": ("lambda p: s3dlio.put_bytes_async(u(p), b'x')", True),
    "put_bytes_async_snapshot": (
        "lambda p: s3dlio.put_bytes_async(u(p), bytearray(b'x'), borrow=False)",
        True,
    ),
    "put_many_async": ("lambda p: s3dlio.put_many_async([(u(p), b'x')])", True),
    "mkdir_async": ("lambda p: s3dlio.mkdir_async(u(p) + '/')", True),
    "stat_async": ("lambda p: s3dlio.stat_async(u(p))", True),
    "exists_async": ("lambda p: s3dlio.exists_async(u(p))", True),
    "head_async": ("lambda p: s3dlio.head_async(u(p))", True),
    "get_into_async": (
        "lambda p: s3dlio.get_into_async(u(p), bytearray(1), expected_len=1)",
        True,
    ),
    # A MultiEndpointStore is built without a logger: on file:// it writes no row at all, so
    # these pin "nothing leaks into session 2" only (the S3 rows of a bucket-form endpoint are
    # covered by the Rust test helper_rows_follow_the_session_of_the_call_not_the_session_at_first_poll).
    "multi_endpoint_get": (
        "lambda p: s3dlio.create_multi_endpoint_store([u(T)]).get(u(p))",
        False,
    ),
    "multi_endpoint_put": (
        "lambda p: s3dlio.create_multi_endpoint_store([u(T)]).put(u(p), b'x')",
        False,
    ),
}

# Every scenario: OLD names the object the held call acts on, NEW the control call's.
BODY = '''
    import asyncio, subprocess, shutil, inspect
    T = os.path.realpath(sys.argv[1])
    DIRECTION, ENTRY = sys.argv[2], sys.argv[3]
    CALL = eval(sys.argv[4])
    EXPECT_ROWS = sys.argv[5] == "1"
    HEADER = {header!r}

    def u(p):
        return "file://" + p

    def rows(p):
        out = subprocess.run([shutil.which("zstd"), "-dc", p], capture_output=True,
                             check=True).stdout.decode()
        lines = out.split("\\n")
        assert lines[-1] == "" and lines[0] == HEADER, (p, lines[:2])
        res = []
        for line in lines[1:-1]:
            c = line.split("\\t")
            assert len(c) == 13, line
            res.append((c[2], c[7]))
        return res

    OLD, NEW = T + "/OLDTOK.bin", T + "/NEWTOK.bin"
    S1, S2 = T + "/s1.tsv.zst", T + "/s2.tsv.zst"
    if ENTRY == "mkdir_async":
        OLD, NEW = T + "/OLDTOK", T + "/NEWTOK"
    for f in (OLD, NEW):
        if "put" not in ENTRY and ENTRY != "mkdir_async":
            with open(f, "wb") as fh:
                fh.write(b"x")

    async def main():
        if DIRECTION == "created_in_session_1":
            s3dlio.init_op_log(S1)
        hold = s3dlio._hold_async_runtime()   # every runtime worker is blocked from here on
        try:
            aw = CALL(OLD)                    # the call is MADE here ...
            s3dlio.finalize_op_log()          # ... session 1 ends (a no-op in the other direction)
            s3dlio.init_op_log(S2)            # ... session 2 starts (or the first one, after the call)
        finally:
            hold.release()                    # only now can the future take its first poll
        await aw
        await CALL(NEW)                       # control: a call MADE in session 2
        s3dlio.finalize_op_log()

    asyncio.run(main())
    r2 = rows(S2)
    assert not any("OLDTOK" in f for _, f in r2), ("the held call wrote into session 2", r2)
    if EXPECT_ROWS:
        assert any("NEWTOK" in f for _, f in r2), ("the session-2 control call was not logged", r2)
    else:
        assert r2 == [], r2
    if DIRECTION == "created_in_session_1":
        r1 = rows(S1)                         # session 1 was finalized before the call ran
        assert not any("OLDTOK" in f for _, f in r1), ("a finalized session gained a row", r1)
'''


@pytest.mark.parametrize("direction", ["created_in_session_1", "created_before_init"])
@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_async_call_is_bound_to_the_session_active_when_it_was_made(
    tmp_path, entry, direction
):
    call_src, expect_rows = ENTRIES[entry]
    proc = run_scenario(
        textwrap.dedent(BODY).format(header=HEADER),
        str(tmp_path),
        direction,
        entry,
        call_src,
        "1" if expect_rows else "0",
        timeout=180,
    )
    assert_ok(proc)
