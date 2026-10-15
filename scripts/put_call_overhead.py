"""Per-call cost of a small put: 256 KiB bytearray, put_bytes and put_bytes_async(borrow=True),
file:// on one path (overwritten each call). Prints medians in microseconds.

Usage: uv run --no-project --isolated --with <wheel> python scripts/put_call_overhead.py <dir> [N]
Measure on a quiet host and compare wheels back to back (alternate the order).
"""

import asyncio
import statistics
import sys
import time

import s3dlio

SIZE = 256 * 1024


def bench_sync(uri, n):
    ba = bytearray(b"x" * SIZE)
    for _ in range(200):
        s3dlio.put_bytes(uri, ba)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        s3dlio.put_bytes(uri, ba)
        ts.append(time.perf_counter() - t)
    return ts


def bench_async_borrow(uri, n):
    ba = bytearray(b"x" * SIZE)

    async def go():
        for _ in range(200):
            await s3dlio.put_bytes_async(uri, ba, borrow=True)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            await s3dlio.put_bytes_async(uri, ba, borrow=True)
            ts.append(time.perf_counter() - t)
        return ts

    return asyncio.run(go())


def report(name, ts):
    us = sorted(x * 1e6 for x in ts)
    print(
        f"{name:28} median {statistics.median(us):8.1f} us   p10 {us[len(us) // 10]:8.1f}   "
        f"p90 {us[9 * len(us) // 10]:8.1f}   n={len(us)}"
    )


if __name__ == "__main__":
    directory = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    uri = f"file://{directory}/overhead.bin"
    report("put_bytes (256 KiB)", bench_sync(uri, n))
    report("put_bytes_async borrow=True", bench_async_borrow(uri, n))
