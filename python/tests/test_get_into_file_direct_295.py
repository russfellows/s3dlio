# python/tests/test_get_into_file_direct_295.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
"""get_into_async / head_async / object_capabilities for file:// and direct:// (#295, #267).

Runs against the INSTALLED wheel, in process, with no network and no environment variable
(``S3DLIO_TEST_DIR`` optionally names a directory on a real filesystem for the scratch files;
the default is the system temp directory).  direct:// tests are skipped when the scratch
filesystem refuses O_DIRECT, unless S3DLIO_REQUIRE_ODIRECT_TESTS=1 (CI), which makes such a skip
a failure (see conftest.py).

Run (the in-repo convention):

    uv run --no-project --isolated --with <wheel> --with pytest --with numpy \
        python -m pytest python/tests/test_get_into_file_direct_295.py -v
"""

import asyncio
import hashlib
import os
import shutil
import tempfile
import threading
import time

import pytest
import s3dlio

np = pytest.importorskip("numpy")

MIB = 1024 * 1024
PAGE = 4096
FILL = 0xA5
CANARY = 0x5A
CANARY_LEN = 64

SCHEMES = ("file", "direct")


def pattern(size):
    """`size` position-dependent bytes that never contain FILL or CANARY."""
    base = np.arange(size, dtype=np.uint64)
    return ((base % 160) + ((base // 160) % 5)).astype(np.uint8).tobytes()


@pytest.fixture(scope="module")
def scratch():
    base = os.environ.get("S3DLIO_TEST_DIR") or None
    path = tempfile.mkdtemp(prefix="getinto295-", dir=base)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def supports_o_direct(path):
    probe = os.path.join(path, "o_direct_probe")
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
    except OSError:
        return False
    os.close(fd)
    return True


@pytest.fixture(params=SCHEMES)
def scheme(request, scratch):
    if request.param == "direct" and not supports_o_direct(scratch):
        pytest.skip("the scratch filesystem does not support O_DIRECT")
    return request.param


def put_file(scratch, name, data):
    path = os.path.join(scratch, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def uri_for(scheme, path):
    return f"{scheme}://{path}"


def aligned_view(size, offset_in_page=0, tail=CANARY_LEN):
    """A numpy uint8 buffer of `size` bytes whose address is `offset_in_page` bytes past a page
    boundary, inside a larger FILL-ed backing array with CANARY bytes after it."""
    # The view starts at (-address) % PAGE + PAGE + offset_in_page <= 3 * PAGE - 2 bytes into the
    # backing array (the first PAGE bytes are a FILL-ed guard before it), so 3 * PAGE of slack is
    # enough for the view and the canary behind it, whatever address numpy returned.
    assert 0 <= offset_in_page < PAGE
    backing = np.empty(size + 3 * PAGE + tail, dtype=np.uint8)
    start = (-backing.ctypes.data) % PAGE + PAGE + offset_in_page
    backing[:] = FILL
    backing[start + size : start + size + tail] = CANARY
    view = backing[start : start + size]
    assert view.size == size and start + size + tail <= backing.size
    assert view.ctypes.data % PAGE == offset_in_page
    return backing, view, start


def assert_around_untouched(backing, start, size):
    assert (backing[:start] == FILL).all(), "wrote before the destination"
    assert (backing[start + size : start + size + CANARY_LEN] == CANARY).all(), (
        "wrote past the destination"
    )


# ---------------------------------------------------------------------------------------------
# the test helper itself
# ---------------------------------------------------------------------------------------------


def test_aligned_view_is_always_whole_at_every_page_offset():
    """`aligned_view` must hand back exactly `size` bytes at exactly `offset_in_page` past a page
    boundary, with the whole canary after it, whatever address numpy happened to allocate (a view
    cut short by a bad backing size made other tests flaky). Randomised over many allocations
    and every kind of offset, so it cannot regress unnoticed."""
    rng = np.random.default_rng(295)
    offsets = [0, 1, 2, 15, 16, 17, PAGE // 2, PAGE - 17, PAGE - 16, PAGE - 2, PAGE - 1]
    sizes = [1, 7, PAGE - 1, PAGE, PAGE + 1, 64 * 1024 + 5, MIB + 3]
    for _ in range(400):
        offset = (
            int(rng.choice(offsets))
            if rng.random() < 0.5
            else int(rng.integers(0, PAGE))
        )
        size = int(rng.choice(sizes))
        backing, view, start = aligned_view(size, offset)
        assert view.size == size, (size, offset, view.size)
        assert view.ctypes.data % PAGE == offset, (size, offset)
        assert start + size + CANARY_LEN <= backing.size, (size, offset)
        assert (backing[start + size : start + size + CANARY_LEN] == CANARY).all()
        assert (backing[:start] == FILL).all()


# ---------------------------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme_name", SCHEMES)
def test_object_capabilities_are_true(scheme_name):
    caps = s3dlio.object_capabilities(f"{scheme_name}:///tmp/whatever")
    assert caps == {"provider": scheme_name, "get_into": True, "head": True}


# ---------------------------------------------------------------------------------------------
# get_into_async
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 17, PAGE, PAGE + 7, MIB + 7, 8 * MIB])
def test_bytearray_destination_round_trip(scheme, scratch, size):
    data = pattern(size)
    uri = uri_for(scheme, put_file(scratch, f"ba-{scheme}-{size}", data))
    dst = bytearray([FILL]) * size

    async def main():
        return await s3dlio.get_into_async(uri, dst, expected_len=size)

    assert asyncio.run(main()) == size
    assert bytes(dst) == data
    dst.append(0)  # no export left behind: a resize succeeds
    del dst[-1:]


def test_zero_length_object(scheme, scratch):
    uri = uri_for(scheme, put_file(scratch, f"zero-{scheme}", b""))

    async def main():
        assert await s3dlio.get_into_async(uri, bytearray(0), expected_len=0) == 0
        with pytest.raises(s3dlio.LengthMismatchError) as info:
            await s3dlio.get_into_async(uri, bytearray(8), expected_len=8)
        assert (info.value.expected, info.value.actual) == (8, 0)

    asyncio.run(main())


@pytest.mark.parametrize(
    "offset_in_page,size",
    [
        (0, 8 * MIB),  # aligned address and length: O_DIRECT straight into the buffer
        (8, 8 * MIB),  # unaligned address: bounce
        (PAGE // 2, 3 * MIB),
        (0, 8 * MIB + 100),  # unaligned length: bounce
        (0, 40 * MIB + 1),  # more than one internal chunk
    ],
)
def test_numpy_destination_aligned_and_unaligned(scheme, scratch, offset_in_page, size):
    data = pattern(size)
    uri = uri_for(
        scheme, put_file(scratch, f"np-{scheme}-{offset_in_page}-{size}", data)
    )
    backing, view, start = aligned_view(size, offset_in_page)

    async def main():
        return await s3dlio.get_into_async(uri, view, expected_len=size)

    assert asyncio.run(main()) == size
    assert view.tobytes() == data
    assert_around_untouched(backing, start, size)


def test_memoryview_slice_destination(scheme, scratch):
    data = pattern(MIB + 3)
    uri = uri_for(scheme, put_file(scratch, f"mv-{scheme}", data))
    outer = bytearray([FILL]) * (len(data) + 32)
    window = memoryview(outer)[5 : 5 + len(data)]

    async def main():
        return await s3dlio.get_into_async(uri, window, expected_len=len(data))

    assert asyncio.run(main()) == len(data)
    window.release()
    assert bytes(outer[5 : 5 + len(data)]) == data
    assert set(outer[:5]) == {FILL} and set(outer[5 + len(data) :]) == {FILL}


@pytest.mark.parametrize("file_size,expected", [(100, 200), (300, 200), (3 * MIB, MIB)])
def test_length_mismatch_is_typed_and_writes_nothing(
    scheme, scratch, file_size, expected
):
    uri = uri_for(
        scheme, put_file(scratch, f"mm-{scheme}-{file_size}", pattern(file_size))
    )
    backing, view = aligned_view(expected)[:2]
    before = backing.copy()

    async def main():
        with pytest.raises(s3dlio.LengthMismatchError) as info:
            await s3dlio.get_into_async(uri, view, expected_len=expected)
        assert isinstance(info.value, ValueError)
        assert info.value.expected == expected
        assert info.value.actual == file_size

    asyncio.run(main())
    assert np.array_equal(backing, before), (
        "a length mismatch must not write into or past the buffer"
    )


def test_missing_file_is_object_not_found(scheme, scratch):
    uri = uri_for(scheme, os.path.join(scratch, "no-such-file"))
    dst = bytearray([FILL]) * 16

    async def main():
        with pytest.raises(s3dlio.ObjectNotFoundError) as info:
            await s3dlio.get_into_async(uri, dst, expected_len=16)
        assert isinstance(info.value, FileNotFoundError)
        assert await s3dlio.head_async(uri) is None

    asyncio.run(main())
    assert set(dst) == {FILL}


def test_directory_is_an_error(scheme, scratch):
    sub = os.path.join(scratch, f"dir-{scheme}")
    os.makedirs(sub, exist_ok=True)
    uri = uri_for(scheme, sub)
    dst = bytearray([FILL]) * 16

    async def main():
        with pytest.raises(RuntimeError) as info:
            await s3dlio.get_into_async(uri, dst, expected_len=16)
        assert not isinstance(info.value, FileNotFoundError)
        with pytest.raises(RuntimeError):
            await s3dlio.head_async(uri)

    asyncio.run(main())
    assert set(dst) == {FILL}


# ---------------------------------------------------------------------------------------------
# head_async
# ---------------------------------------------------------------------------------------------


def test_head_async_reports_size_and_identity(scheme, scratch):
    uri = uri_for(scheme, put_file(scratch, f"head-{scheme}", pattern(12_345)))

    async def main():
        meta = await s3dlio.head_async(uri)
        assert type(meta) is s3dlio.ObjectMetadata
        assert meta.size == 12_345
        assert isinstance(meta.etag, str) and meta.etag
        assert meta.version_id is None
        assert await s3dlio.head_async(uri) == meta

    asyncio.run(main())


# ---------------------------------------------------------------------------------------------
# the read runs without the GIL and a cancelled read stops writing
# ---------------------------------------------------------------------------------------------


def test_the_event_loop_stays_responsive_during_a_large_read(scheme, scratch):
    size = 512 * MIB
    uri = uri_for(scheme, put_file(scratch, f"big-{scheme}", pattern(size)))
    dst = np.empty(size, dtype=np.uint8)

    async def main():
        ticks = 0
        stop = False

        async def ticker():
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.001)
                ticks += 1

        side = asyncio.ensure_future(ticker())
        started = time.perf_counter()
        await s3dlio.get_into_async(uri, dst, expected_len=size)
        elapsed = time.perf_counter() - started
        stop = True
        await side
        return ticks, elapsed

    ticks, elapsed = asyncio.run(main())
    assert hashlib.sha256(dst.data).digest() == hashlib.sha256(pattern(size)).digest()
    # A native read holding the GIL (or blocking the loop) would leave almost no ticks.
    assert ticks >= max(3, int(elapsed * 1000 * 0.05)), (ticks, elapsed)


def test_other_threads_run_during_the_read(scheme, scratch):
    size = 512 * MIB
    uri = uri_for(scheme, put_file(scratch, f"gil-{scheme}", pattern(size)))
    dst = np.empty(size, dtype=np.uint8)
    ticks = 0
    stop = threading.Event()

    def spin():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.0005)

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        asyncio.run(_read(uri, dst, size))
    finally:
        stop.set()
        thread.join()
    assert ticks >= 5, ticks


async def _read(uri, dst, size):
    return await s3dlio.get_into_async(uri, dst, expected_len=size)


def test_cancelling_the_read_stops_every_write(scheme, scratch):
    size = 512 * MIB
    uri = uri_for(scheme, put_file(scratch, f"cancel-{scheme}", pattern(size)))
    dst = np.full(size, FILL, dtype=np.uint8)

    def digest():
        return hashlib.sha256(dst.data).digest()

    async def main():
        task = asyncio.ensure_future(_read(uri, dst, size))
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            return "finished"  # the read beat the cancel: nothing to check
        # The awaitable has ended: the buffer is ours and must never change again.
        before = digest()
        await asyncio.sleep(0.3)
        assert digest() == before, "a native write happened after the cancel"
        return "cancelled"

    outcome = asyncio.run(main())
    assert outcome in ("finished", "cancelled")
    # The buffer is reusable: a complete read into it succeeds.
    assert asyncio.run(_read(uri, dst, size)) == size
    assert bytes(dst[:4096]) == pattern(4096)
