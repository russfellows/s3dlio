"""The real Python caller-buffer consumer smoke (v0.10.0 plan, section 9 "Before any PR"):
put -> head_async -> get_into_async into a fresh buffer -> byte comparison, content-at-address checks
(the in-flight proof that bytes land in caller memory is in test_caller_buffer_get_into.py,
which can hold a body mid-stream), cancellation of a large read, typed errors, and exact-key cleanup, through the INSTALLED wheel
against one real service.

SKIPPED unless ``S3DLIO_LIVE_SMOKE_URI_PREFIX`` is set.  It names a prefix the runner owns
(for example ``s3://<bucket>/<prefix>``, ``az://<account>/<container>/<prefix>`` or
``gs://<bucket>/<prefix>``), with credentials and any endpoint configured the usual way for
that provider (AWS_* variables, Azure CLI login or AZURE_* variables, application default
credentials).  Nothing here names an account, bucket or host.

Everything lives under ``<prefix>/s3dlio-cb-smoke-<random>/``.  The keys are registered BEFORE
they are written and deleted by exact key afterwards (no listing is ever used to find them),
even when a test fails; each key is then checked absent with ``head_async``.  Data stays at or
below 8 MiB per object; about 22 MiB is written in total and at most about 112 MiB is read
(the worst case, in which every cancelled read happens to complete; a cancelled read stops
early).  The read side is enforced: every get_into_async is charged its full expected_len
against READ_BUDGET (128 MiB for the whole session) before it starts, and the run fails if
the budget would be exceeded.

Run (one wheel per interpreter; the in-repo convention):

    S3DLIO_LIVE_SMOKE_URI_PREFIX='<scheme>://<...>/<unique run prefix>' \
        uv run --no-project --isolated --with <wheel> --with pytest --with pytest-asyncio \
        python -m pytest python/tests/test_caller_buffer_live_smoke.py -v -s

For ``gs://`` use a wheel built with ``./build_pyo3.sh full``; for a GCS S3-interop run use an
``s3://`` prefix with AWS_ENDPOINT_URL and the interop credentials.
"""

import asyncio
import ctypes
import os
import random
import time
import uuid

import pytest
import s3dlio

ENV_PREFIX = "S3DLIO_LIVE_SMOKE_URI_PREFIX"
PREFIX = os.environ.get(ENV_PREFIX, "").strip()

pytestmark = pytest.mark.skipif(not PREFIX, reason=f"{ENV_PREFIX} is not set")

MIB = 1024 * 1024
FILL = 0xEE
CANARY = 0xA5
CANARY_LEN = 64
SENTINEL = 0x5A

# Sizes of the round trip: empty, one byte, just over 1 MiB (not a multiple of any chunk size),
# and several MiB.
SIZES = (0, 1, MIB + 7, 4 * MIB + 3)
LARGE = 8 * MIB  # the object a read is cancelled on
# Cap on the bytes the run may ask to read: every get_into_async is charged its full
# expected_len BEFORE it starts (a cancelled read is charged in full too), summed over the
# whole pytest session.  The tests need about 112 MiB in the worst case, so the cap leaves
# room without allowing a runaway loop.
READ_BUDGET = 128 * MIB
CANCEL_ITERATIONS = 10

LITERAL_KEYS = ("a%2Fb", "100%", "x@y", "k=v", "p/q/r", "name.")


def content(seed, size):
    """`size` deterministic bytes derived from `seed` (the object's full URI)."""
    return random.Random(seed).randbytes(size)


def resizable(buf):
    """True when no export is held on the bytearray (a resize succeeds)."""
    try:
        buf.append(0)
        del buf[-1:]
        return True
    except BufferError:
        return False


class Run:
    """The run's unique base URI and the exact keys written under it."""

    # bytes charged to READ_BUDGET by every Run of this session (see begin_get_into)
    session_read_charged = 0

    def __init__(self):
        self.base = PREFIX.rstrip("/") + "/s3dlio-cb-smoke-" + uuid.uuid4().hex[:12]
        self.written = []
        self.calls = {"put": 0, "head": 0, "get_into": 0, "delete": 0}
        self.bytes = {"put": 0, "get_into": 0}

    def begin_get_into(self, expected_len):
        """Count one get_into_async and charge its full expected_len against READ_BUDGET;
        call it before every get_into_async, whatever the outcome of the read."""
        self.calls["get_into"] += 1
        Run.session_read_charged += expected_len
        assert Run.session_read_charged <= READ_BUDGET, (
            f"read budget exceeded: {Run.session_read_charged} > {READ_BUDGET} bytes"
        )

    def uri(self, key):
        return f"{self.base}/{key}"

    def put(self, key, data):
        """Register the key, THEN write it (so a failed write is still cleaned up)."""
        uri = self.uri(key)
        self.written.append(uri)
        s3dlio.put_bytes(uri, data)
        self.calls["put"] += 1
        self.bytes["put"] += len(data)
        return uri

    def register_absent(self, key):
        """A key the test never writes but still checks absent at the end."""
        uri = self.uri(key)
        self.written.append(uri)
        return uri

    def cleanup(self):
        problems = []
        for uri in self.written:
            try:
                s3dlio.delete(uri)
                self.calls["delete"] += 1
            except Exception as exc:  # noqa: BLE001 - collected and reported below
                problems.append(f"delete {uri}: {exc!r}")
        if self.written:
            # a second exact delete of the first key: already absent must still succeed
            try:
                s3dlio.delete(self.written[0])
                self.calls["delete"] += 1
            except Exception as exc:  # noqa: BLE001
                problems.append(f"repeat delete {self.written[0]}: {exc!r}")

        async def check():
            for uri in self.written:
                self.calls["head"] += 1
                try:
                    meta = await s3dlio.head_async(uri)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"head after delete {uri}: {exc!r}")
                    continue
                if meta is not None:
                    problems.append(f"still present after delete: {uri} {meta!r}")

        asyncio.run(check())
        return problems

    def summary(self):
        return (
            f"calls={self.calls} bytes_written={self.bytes['put']} "
            f"bytes_read_into={self.bytes['get_into']} "
            f"session_read_charged={Run.session_read_charged}/{READ_BUDGET}"
        )


@pytest.fixture
def run(request):
    failed_before = request.session.testsfailed
    state = Run()
    try:
        yield state
    finally:
        problems = state.cleanup()
        print(f"\nlive smoke summary: {state.summary()}")
        # pytest has already logged the test body's own failure by now (teardown follows
        # the call phase): then the cleanup problems are REPORTED beside it, never raised
        # over it, so the original failure stays the one that is shown.
        body_failed = request.session.testsfailed > failed_before
        report = "cleanup left objects behind:\n" + "\n".join(problems)
        if problems and body_failed:
            print(
                f"\n{report}\n(the test body had already failed; see its failure above)"
            )
        else:
            assert not problems, report


def test_the_prefix_provider_supports_the_new_api():
    caps = s3dlio.object_capabilities(PREFIX)
    print(f"\ncapabilities of the prefix: {caps}")
    assert caps["get_into"] is True and caps["head"] is True, (
        f"this wheel/provider has no caller-buffer GET for {PREFIX!r}: {caps}"
    )


@pytest.mark.parametrize("size", SIZES)
def test_put_head_get_into_round_trip_with_identity(run, size):
    data = content(run.uri(f"rt-{size}"), size)
    uri = run.put(f"rt-{size}", data)

    async def main():
        meta = await s3dlio.head_async(uri)
        run.calls["head"] += 1
        assert meta is not None, "HEAD of a just-written object reported absence"
        assert meta.size == size, (meta.size, size)
        assert isinstance(meta.etag, str) and meta.etag, f"no ETag: {meta!r}"
        again = await s3dlio.head_async(uri)
        run.calls["head"] += 1
        assert again == meta, f"identity changed between two HEADs: {meta!r} {again!r}"
        print(f"\nsize {size}: {meta!r}")

        # A fresh bytearray with a canary after the destination.
        buf = bytearray([FILL]) * size + bytearray([CANARY]) * CANARY_LEN
        probe = (ctypes.c_char * len(buf)).from_buffer(buf)
        address = ctypes.addressof(probe)
        del probe  # the probe's own export must not outlive the setup
        run.begin_get_into(size)
        got = await s3dlio.get_into_async(uri, buf, expected_len=size)
        run.bytes["get_into"] += size
        assert got == size
        assert bytes(buf[:size]) == data, "bytes differ from what was written"
        assert buf[size:].count(CANARY) == CANARY_LEN, "a write past expected_len"
        # Content-at-address check, not an identity proof (a staging buffer plus a final copy
        # would also pass): the allocation did not move.  Taken before the resize probe
        # below, which may realloc and move the bytearray.
        after = (ctypes.c_char * len(buf)).from_buffer(buf)
        assert ctypes.addressof(after) == address, "the buffer moved during the read"
        del after
        assert resizable(buf), "the export outlived the await"

        # A ctypes destination: the bytes are readable at the allocation's own address
        # (a content-at-address check; ctypes objects cannot move, so the address assert is
        # trivially true and proves nothing by itself).
        arr = (ctypes.c_ubyte * size)()
        arr_address = ctypes.addressof(arr)
        run.begin_get_into(size)
        got = await s3dlio.get_into_async(uri, arr, expected_len=size)
        run.bytes["get_into"] += size
        assert got == size
        assert ctypes.string_at(arr_address, size) == data
        assert ctypes.addressof(arr) == arr_address

        # A memoryview of a bytearray: the same underlying allocation is filled.
        backing = bytearray([FILL]) * size
        view = memoryview(backing)
        run.begin_get_into(size)
        got = await s3dlio.get_into_async(uri, view, expected_len=size)
        run.bytes["get_into"] += size
        assert got == size
        assert bytes(view) == data and bytes(backing) == data
        view.release()
        assert resizable(backing)

    asyncio.run(main())


def test_keys_are_literal_characters(run):
    async def main():
        for key in LITERAL_KEYS:
            data = content(run.uri(key), 33)
            uri = run.put(key, data)
            meta = await s3dlio.head_async(uri)
            run.calls["head"] += 1
            assert meta is not None and meta.size == 33, f"HEAD of literal key {key!r}"
            buf = bytearray(33)
            run.begin_get_into(33)
            got = await s3dlio.get_into_async(uri, buf, expected_len=33)
            run.bytes["get_into"] += 33
            assert got == 33 and bytes(buf) == data, f"literal key {key!r}"

    asyncio.run(main())


def test_missing_key_is_object_not_found_and_head_is_none(run):
    uri = run.register_absent("never-written")

    async def main():
        buf = bytearray([FILL]) * 64
        run.begin_get_into(64)
        with pytest.raises(s3dlio.ObjectNotFoundError) as info:
            await s3dlio.get_into_async(uri, buf, expected_len=64)
        assert isinstance(info.value, FileNotFoundError)
        assert resizable(buf)
        assert await s3dlio.head_async(uri) is None
        run.calls["head"] += 1

    asyncio.run(main())


def test_wrong_expected_len_is_a_length_mismatch(run):
    size = 100_003
    data = content(run.uri("mismatch"), size)
    uri = run.put("mismatch", data)

    async def main():
        for expected_len in (size - 1, size + 1):
            buf = bytearray([FILL]) * expected_len + bytearray([CANARY]) * CANARY_LEN
            run.begin_get_into(expected_len)
            with pytest.raises(s3dlio.LengthMismatchError) as info:
                await s3dlio.get_into_async(uri, buf, expected_len=expected_len)
            exc = info.value
            assert isinstance(exc, ValueError)
            assert exc.expected == expected_len
            # `actual` is None when the provider did not say; when it does, it is the size
            assert exc.actual is None or exc.actual == size, (exc.actual, size)
            print(f"\nexpected_len={expected_len}: actual={exc.actual}")
            assert buf[expected_len:].count(CANARY) == CANARY_LEN, (
                "a write past expected_len"
            )
            assert resizable(buf)
        # the right length still works
        buf = bytearray(size)
        run.begin_get_into(size)
        assert await s3dlio.get_into_async(uri, buf, expected_len=size) == size
        run.bytes["get_into"] += size
        assert bytes(buf) == data

    asyncio.run(main())


def test_cancelling_a_large_read_frees_the_buffer_only_after_the_awaitable_ends(run):
    """Cancel an 8 MiB read at varying moments.  Whatever the moment (before the first poll,
    mid-flight, or after it completed), the awaitable ends, the buffer is free by the time it
    is seen done, and nothing writes into it afterwards.  The mix of outcomes is printed; a
    fast link may complete most reads before the cancel lands, which is fine for the
    invariants and is why no count of cancellations is asserted."""
    data = content(run.uri("large"), LARGE)
    uri = run.put("large", data)
    outcomes = {"cancelled": 0, "completed": 0}
    rng = random.Random(1)
    problems = []

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(
            lambda lp, ctx: problems.append(str(ctx.get("message")))
        )
        for i in range(CANCEL_ITERATIONS):
            buf = bytearray([FILL]) * LARGE
            run.begin_get_into(LARGE)
            fut = s3dlio.get_into_async(uri, buf, expected_len=LARGE)
            await asyncio.sleep(rng.random() * 0.05)
            was_pending = not fut.done()
            cancel_result = fut.cancel()
            assert cancel_result is was_pending or fut.done(), (
                i,
                cancel_result,
                was_pending,
            )
            deadline = time.monotonic() + 60
            while True:
                done = fut.done()
                free = resizable(buf)
                assert not done or free, (
                    f"iteration {i}: terminal while the buffer was exported"
                )
                if done:
                    break
                assert time.monotonic() < deadline, (
                    f"iteration {i}: never finished (hang)"
                )
                await asyncio.sleep(0.005)
            if fut.cancelled():
                outcomes["cancelled"] += 1
            else:
                assert fut.result() == LARGE, f"iteration {i}"
                assert bytes(buf) == data, (
                    f"iteration {i}: wrong content after completion"
                )
                outcomes["completed"] += 1
                run.bytes["get_into"] += LARGE
            # reuse the buffer at once; a late native write would damage it
            buf[:] = bytes([SENTINEL]) * LARGE
            await asyncio.sleep(0.05)
            assert buf.count(SENTINEL) == LARGE, (
                f"iteration {i}: a late write hit the buffer"
            )
            del fut
        await asyncio.sleep(0.2)

    asyncio.run(main())
    print(f"\ncancellation outcomes over {CANCEL_ITERATIONS} reads: {outcomes}")
    assert sum(outcomes.values()) == CANCEL_ITERATIONS
    assert not problems, f"the event loop logged: {problems}"


def test_wait_for_and_timeout_end_only_after_the_buffer_is_free(run):
    data = content(run.uri("timeout"), LARGE)
    uri = run.put("timeout", data)

    async def main():
        for how in ("wait_for", "timeout"):
            buf = bytearray([FILL]) * LARGE
            run.begin_get_into(LARGE)
            fut = s3dlio.get_into_async(uri, buf, expected_len=LARGE)
            try:
                if how == "wait_for":
                    await asyncio.wait_for(fut, 0.001)
                else:
                    async with asyncio.timeout(0.001):
                        await fut
            except TimeoutError:
                assert fut.done(), f"{how}: TimeoutError before the awaitable ended"
                assert resizable(buf), (
                    f"{how}: TimeoutError with the buffer still exported"
                )
            else:
                # completed inside 1 ms: possible on a very fast link
                assert fut.result() == LARGE
                run.bytes["get_into"] += LARGE
                assert bytes(buf) == data
            buf[:] = bytes([SENTINEL]) * LARGE
            await asyncio.sleep(0.1)
            assert buf.count(SENTINEL) == LARGE, f"{how}: a late write hit the buffer"

    asyncio.run(main())
