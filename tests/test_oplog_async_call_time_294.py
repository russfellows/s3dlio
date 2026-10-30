# tests/test_oplog_async_call_time_294.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""Op-log session binding of async calls and of constructed loaders (issue #294).

The contract:

* an awaitable created while op-log session 1 is active never writes a row into a later
  session 2 (after ``finalize_op_log()`` + ``init_op_log()``), and an awaitable created while no
  session is active is never logged, however late the runtime first polls it;
* objects you construct (datasets, loaders, MultiEndpointStore) capture the logger when they are CONSTRUCTED and
  use it for every fetch: one built in session 1 and used after ``finalize_op_log()`` +
  ``init_op_log(other)`` writes nothing into session 2, and one built before ``init_op_log``
  is never logged;
* the synchronous ``get_many`` on ``s3://`` captures the logger once, at entry.

``test_12_awaitable_created_in_session_1_writes_nothing_into_session_2`` (in
``test_oplog_lifecycle_permutations_294.py``) tests the first rule with real timing, so it only
failed about 1 run in 30. These tests make the race DETERMINISTIC with the hidden test hook
``s3dlio._hold_async_runtime()``: it occupies every worker thread of the async runtime and returns
only when all are occupied, so an awaitable created while the hold is active has been spawned but
provably cannot take its first poll until the hold is released. Everything between the call and
the release (``finalize_op_log()``, ``init_op_log(other)``) therefore happens BEFORE the first poll.

THE HOOK EXISTS ONLY IN A WHEEL BUILT WITH THE ``test-hooks`` CARGO FEATURE (CI builds its test
wheel that way; the published wheels do not contain it). Without it these tests skip locally and
FAIL when ``S3DLIO_REQUIRE_TEST_HOOKS=1`` (set in CI), so a mis-built wheel cannot silently skip
them. Local run: ``maturin build --release --features extension-module,backend-azure,test-hooks``.

Each scenario runs in a fresh interpreter (the op-log is process-global). ``file://`` cases need
no server; the S3-only paths (batch helpers, bucket-form MultiEndpointStore, dataset loaders) run
against a small in-process mock S3 server.
"""

import os
import shutil
import textwrap
import time

import pytest
from test_oplog_lifecycle import HEADER, assert_ok, run_scenario
from test_python_delete_contract_212 import MockS3, env_for

pytestmark = pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="the zstd command-line tool is not installed (e.g. `apt-get install zstd`)",
)


def _has_hook():
    import s3dlio

    return getattr(s3dlio, "_hold_async_runtime", None) is not None


@pytest.fixture
def need_hook():
    """The deterministic tests need `s3dlio._hold_async_runtime` (a `test-hooks` build)."""
    if _has_hook():
        return
    msg = "this wheel was built without the `test-hooks` cargo feature (no s3dlio._hold_async_runtime)"
    if os.environ.get("S3DLIO_REQUIRE_TEST_HOOKS") == "1":
        pytest.fail(msg + "; S3DLIO_REQUIRE_TEST_HOOKS=1 forbids skipping")
    pytest.skip(msg)


# ----------------------------------------------------------------------------
# Common child code
# ----------------------------------------------------------------------------

PRELUDE = '''
    import asyncio, subprocess, shutil, inspect, threading, time
    T = os.path.realpath(sys.argv[1])
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

    S1, S2 = T + "/s1.tsv.zst", T + "/s2.tsv.zst"

    def check(r2, expect_rows):
        assert not any("OLDTOK" in f for _, f in r2), ("the OLD call/object wrote into session 2", r2)
        if expect_rows:
            assert any("NEWTOK" in f for _, f in r2), ("the session-2 control was not logged", r2)
'''


def scenario(body, *args, env=None):
    code = textwrap.dedent(PRELUDE).format(header=HEADER) + "\n" + textwrap.dedent(body)
    assert_ok(run_scenario(code, *args, env=env, timeout=240))


# ----------------------------------------------------------------------------
# 1. file:// calls held before their first poll
# ----------------------------------------------------------------------------
# name -> python source of `call(p)` that CREATES the awaitable for the path `p` (`u(p)` is its URI).
# Only entry points whose file:// call writes a row are here: a call that writes no row on file://
# (mkdir_async, an unlogged MultiEndpointStore) cannot be RED-able, so it is not tested on file://.
# The S3 table below covers the MultiEndpointStore, whose S3 backend DOES write rows.
FILE_ENTRIES = {
    "get_async": "lambda p: s3dlio.get_async(u(p))",
    "get_range_async": "lambda p: s3dlio.get_range_async(u(p), 0, 1)",
    "put_bytes_async": "lambda p: s3dlio.put_bytes_async(u(p), b'x')",
    "put_bytes_async_snapshot": "lambda p: s3dlio.put_bytes_async(u(p), bytearray(b'x'), borrow=False)",
    "put_many_async": "lambda p: s3dlio.put_many_async([(u(p), b'x')])",
    # Already correct before the fix (store and logger built on the calling thread): regression pins.
    "stat_async": "lambda p: s3dlio.stat_async(u(p))",
    "exists_async": "lambda p: s3dlio.exists_async(u(p))",
    "head_async": "lambda p: s3dlio.head_async(u(p))",
    "get_into_async": "lambda p: s3dlio.get_into_async(u(p), bytearray(1), expected_len=1)",
}

HELD_BODY = '''
    DIRECTION = sys.argv[2]
    CALL = eval(sys.argv[3])
    NEW_URI_BASE = sys.argv[4]          # "" for file:// (CALL takes a path), else a prefix
    OLD, NEW = (T + "/OLDTOK.bin", T + "/NEWTOK.bin") if not NEW_URI_BASE else ("OLDTOK", "NEWTOK")
    if not NEW_URI_BASE:
        for f in (OLD, NEW):
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
    check(rows(S2), True)
    if DIRECTION == "created_in_session_1":
        assert not any("OLDTOK" in f for _, f in rows(S1)), "a finalized session gained a row"
'''

DIRECTIONS = ["created_in_session_1", "created_before_init"]


@pytest.mark.usefixtures("need_hook")
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("entry", sorted(FILE_ENTRIES))
def test_file_async_call_is_bound_to_the_session_active_when_it_was_made(
    tmp_path, entry, direction
):
    scenario(HELD_BODY, str(tmp_path), direction, FILE_ENTRIES[entry], "")


# ----------------------------------------------------------------------------
# 2. S3-only paths against a mock S3 server
# ----------------------------------------------------------------------------


class DataMock(MockS3):
    """MockS3 that serves 4-byte objects (HEAD advertises the length) and can GATE a GET: a GET
    for a key containing ``GATE`` touches ``<gate_dir>/seen`` and waits for ``<gate_dir>/release``."""

    gate_dir = None

    def respond(self, method, key, query, body_keys):
        if method == "GET" and "list-type" not in query and self.gate_dir and "GATE" in key:
            open(os.path.join(self.gate_dir, "seen"), "w").close()
            end = time.time() + 60
            while not os.path.exists(os.path.join(self.gate_dir, "release")) and time.time() < end:
                time.sleep(0.02)
        if method in ("GET", "HEAD") and "list-type" not in query:
            with self.lock:
                present = key in self.keys
            return (200, "xxxx") if present else (404, "")
        return super().respond(method, key, query, body_keys)


KEYS = (
    "OLDTOK", "NEWTOK", "mep/OLDTOK", "mep/NEWTOK",
    "oldds/OLDTOK0", "newds/NEWTOK0",
    "OLDTOK-GATE-a", "OLDTOK-GATE-b", "NEWTOK-sync",
)


@pytest.fixture
def mock():
    m = DataMock(KEYS)
    yield m
    m.close()


def s3env(m):
    env = env_for(m)
    env.update({"S3DLIO_LOAD_DOTENV": "0"})
    return env


S3_ENTRIES = {
    "get_many_async": 'lambda n: s3dlio.get_many_async(["s3://bkt/" + n])',
    "get_many_stats_async": 'lambda n: s3dlio.get_many_stats_async(["s3://bkt/" + n])',
    "put_async": 'lambda n: s3dlio.put_async("s3://bkt/", 1, template=n + "-{}", size=16)',
}


@pytest.mark.usefixtures("need_hook")
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("entry", sorted(S3_ENTRIES))
def test_s3_async_call_is_bound_to_the_session_active_when_it_was_made(
    tmp_path, mock, entry, direction
):
    scenario(HELD_BODY, str(tmp_path), direction, S3_ENTRIES[entry], "s3", env=s3env(mock))


# ----------------------------------------------------------------------------
# 3. Constructed objects: datasets, loaders and MultiEndpointStores capture the logger at CONSTRUCTION
# ----------------------------------------------------------------------------
# `build(uri)` constructs the object, `use(obj)` is an async function that fetches through it.
MEP_BUILD = (
    'lambda uri: (s3dlio.create_multi_endpoint_store(["s3://bkt/mep/"]), "OLDTOK" if "old" in uri else "NEWTOK")'
)

OBJECTS = {
    "dataset_get_item": (
        "lambda uri: s3dlio.create_dataset(uri)",
        "lambda ds: ds.get_item(0)",
        False,
    ),
    "async_loader": (
        "lambda uri: s3dlio.create_async_loader(uri)",
        "drain",
        False,
    ),
    # A MultiEndpointStore is a store: it captures the logger at construction and uses it for every call
    # (its bucket-form s3:// endpoints write the rows). `build` returns (store, token) so the control
    # store, built in session 2, touches the NEW key.
    "multi_endpoint_get": (
        MEP_BUILD,
        'lambda o: o[0].get("s3://bkt/mep/" + o[1])',
        False,
    ),
    "multi_endpoint_put": (
        MEP_BUILD,
        'lambda o: o[0].put("s3://bkt/mep/" + o[1], b"xxxx")',
        False,
    ),
    "multi_endpoint_get_from_endpoint": (
        MEP_BUILD,
        'lambda o: o[0].get_from_endpoint(0, "s3://bkt/mep/" + o[1])',
        False,
    ),
    "multi_endpoint_put_to_endpoint": (
        MEP_BUILD,
        'lambda o: o[0].put_to_endpoint(0, "s3://bkt/mep/" + o[1], b"xxxx")',
        False,
    ),
    "multi_endpoint_put_all_endpoints": (
        MEP_BUILD,
        'lambda o: o[0].put_all_endpoints("s3://bkt/mep/" + o[1], b"xxxx")',
        False,
    ),
    # (PyS3AsyncDataLoader is a deprecated shim whose __aiter__ returns a loader that is not an async
    # iterator, so it cannot be iterated; create_async_loader is the supported route to S3BytesDataset.)
}

OBJECT_BODY = '''
    DIRECTION = sys.argv[2]
    BUILD = eval(sys.argv[3])
    USE = sys.argv[4]
    HOLD = sys.argv[5] == "1"

    async def drain(obj):
        async for _ in obj:
            pass

    async def use(obj):
        if USE == "drain":
            await drain(obj)
        else:
            await eval(USE)(obj)

    async def main():
        if DIRECTION == "built_in_session_1":
            s3dlio.init_op_log(S1)
        hold = s3dlio._hold_async_runtime() if HOLD else None
        try:
            old = BUILD("s3://bkt/oldds/")        # constructed here ...
            s3dlio.finalize_op_log()              # ... session 1 ends (a no-op in the other direction)
            s3dlio.init_op_log(S2)                # ... session 2 starts
        finally:
            if hold is not None:
                hold.release()
        await use(old)                            # used in session 2
        new = BUILD("s3://bkt/newds/")            # control: constructed in session 2
        await use(new)
        s3dlio.finalize_op_log()

    asyncio.run(main())
    check(rows(S2), True)
    if DIRECTION == "built_in_session_1":
        assert not any("OLDTOK" in f for _, f in rows(S1)), "a finalized session gained a row"
'''


@pytest.mark.usefixtures("need_hook")
@pytest.mark.parametrize("direction", ["built_in_session_1", "built_before_init"])
@pytest.mark.parametrize("obj", sorted(OBJECTS))
def test_constructed_dataset_or_loader_keeps_the_logger_it_was_built_with(
    tmp_path, mock, obj, direction
):
    build, use, hold = OBJECTS[obj]
    scenario(
        OBJECT_BODY, str(tmp_path), direction, build, use, "1" if hold else "0", env=s3env(mock)
    )


# ----------------------------------------------------------------------------
# 4. Synchronous get_many on s3:// captures the logger once, at entry
# ----------------------------------------------------------------------------

SYNC_BODY = '''
    GATE = sys.argv[2]
    s3dlio.init_op_log(S1)
    uris = ["s3://bkt/OLDTOK-GATE-a", "s3://bkt/OLDTOK-GATE-b"]
    out = {}
    def work():
        # max_in_flight=1: the second GET only STARTS after the first finished, i.e. after the switch.
        out["r"] = s3dlio.get_many(uris, max_in_flight=1)
    t = threading.Thread(target=work)
    t.start()
    end = time.time() + 60
    while not os.path.exists(GATE + "/seen"):          # the mock holds the first GET
        assert time.time() < end, "the mock never saw the first GET"
        time.sleep(0.02)
    s3dlio.finalize_op_log()                           # session 1 ends while get_many is in flight ...
    s3dlio.init_op_log(S2)                             # ... session 2 starts
    open(GATE + "/release", "w").close()
    t.join(60)
    assert not t.is_alive() and len(out["r"]) == 2, out
    s3dlio.get_many(["s3://bkt/NEWTOK-sync"])          # control: a call made in session 2
    s3dlio.finalize_op_log()
    check(rows(S2), True)
    assert not any("OLDTOK" in f for _, f in rows(S1)), "a finalized session gained a row"
'''


def test_sync_get_many_s3_takes_the_logger_once_at_entry(tmp_path, mock):
    gate = tmp_path / "gate"
    gate.mkdir()
    mock.gate_dir = str(gate)
    scenario(SYNC_BODY, str(tmp_path), str(gate), env=s3env(mock))
