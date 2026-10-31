# python/tests/test_no_offgil_drop_297.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""#297: s3dlio never drops a Python reference on a thread that does not hold the GIL.

pyo3 registers such a drop in a process-global mutex; a parent thread holding it at a fork() left
the child's first Python call waiting for it for ever. The two sources were an exact-`bytes`
payload dying on a Tokio worker / inside `py.detach` (now queued for a GIL holder) and the
async bridge dropping a spare reference per awaitable (now an s3dlio-owned bridge whose references
are consumed under the GIL).

Two kinds of check:
* counters (any wheel): the queue path is used (`deferred_decrefs` grows) and the async bridge
  never falls back to it (`async_gilref_deferred` stays 0) over many calls, cancellations and threads;
* the ORACLE (CI builds a wheel with `RUSTFLAGS="--cfg pyo3_disable_reference_pool"` and runs this
  file with it): in that build an off-GIL `Py` drop ABORTS the process, so any future regression
  fails these tests by crashing them.
"""
import asyncio
import os
import tempfile
import threading
import time

import pytest
import s3dlio


@pytest.fixture()
def uri():
    d = tempfile.mkdtemp(prefix="offgil297_")
    with open(f"{d}/o.bin", "wb") as f:
        f.write(b"x" * 65536)
    return f"file://{d}/o.bin", d


def stats():
    return s3dlio._put_export_stats()


def test_exact_bytes_puts_release_their_reference_through_the_queue(uri):
    u, d = uri
    before = stats()["deferred_decrefs"]
    for i in range(50):
        s3dlio.put_bytes(f"file://{d}/p{i}.bin", bytes([i % 256]) * 4096)
    assert stats()["deferred_decrefs"] > before, "the exact-bytes drop must go through the queue"
    deadline = time.time() + 5
    while stats()["queued"] and time.time() < deadline:
        time.sleep(0.02)
    assert stats()["queued"] == 0, "the queue is drained"
    assert s3dlio.get(f"file://{d}/p7.bin") is not None


def test_checkpoint_save_and_async_puts_do_not_drop_off_the_gil(uri):
    u, d = uri
    st = s3dlio.PyCheckpointStore(f"file://{d}/ck")
    for i in range(20):
        st.save(i, 1, "pt", bytes([i]) * 1000)

    async def run():
        for i in range(20):
            await s3dlio.put_bytes_async(f"file://{d}/a{i}.bin", bytes([i]) * 1000)

    asyncio.run(run())
    assert stats()["async_gilref_deferred"] == 0


def test_the_async_bridge_never_falls_back_to_the_queue(uri):
    u, d = uri

    async def run():
        for _ in range(100):
            assert len(memoryview(await s3dlio.get_async(u))) == 65536
        # cancelled awaitables: the Rust side is dropped, the references are still consumed under the GIL
        for _ in range(100):
            t = asyncio.ensure_future(s3dlio.get_async(u))
            await asyncio.sleep(0)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        # an error result
        for _ in range(20):
            with pytest.raises(Exception):
                await s3dlio.get_async(u + ".missing")

    asyncio.run(run())
    time.sleep(0.2)
    assert stats()["async_gilref_deferred"] == 0


def test_many_threads_mixing_sync_async_and_puts(uri):
    u, d = uri
    errors = []

    def worker(k):
        try:
            for i in range(30):
                s3dlio.get(u)
                s3dlio.put_bytes(f"file://{d}/t{k}_{i}.bin", b"y" * 2048)
                asyncio.run(_ag(u))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
    assert stats()["async_gilref_deferred"] == 0


async def _ag(u):
    return await s3dlio.get_async(u)
