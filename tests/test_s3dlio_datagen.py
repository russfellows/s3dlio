#!/usr/bin/env python3
"""
tests/test_s3dlio_datagen.py

Verify s3dlio data generation after dgen-data migration.

Tests:
  1. generate_data() — zero-copy BytesView, memoryview, numpy
  2. generate_data() with dedup/compress
  3. generate_data_with_threads() — custom thread count
  4. generate_into_buffer() — fills user buffer in-place
  5. Generator.fill_chunk() — streaming generation, zero-copy write
  6. Generator throughput benchmark
  7. Generator uniqueness — dedup-safe output
  8. Concurrent Generator — N threads sharing global Rayon pool
  9. Utility: default_data_gen_threads(), total_cpus()
  10. Thread safety — concurrent generate_data() calls
  11. Tokio safety — no nested runtime panic

Zero-copy guarantee:
  - generate_data() returns BytesView (buffer protocol); memoryview() is zero-copy
  - Generator.fill_chunk() writes directly into user's bytearray (zero-copy write)
"""

import sys
import time
import threading
import numpy as np

import s3dlio

SIZE_1M = 1 * 1024 * 1024
SIZE_8M = 8 * 1024 * 1024
SIZE_32M = 32 * 1024 * 1024


def banner(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# 1. generate_data() — zero-copy BytesView
# ---------------------------------------------------------------------------
banner("1. s3dlio.generate_data() — BytesView zero-copy")

data1 = s3dlio.generate_data(SIZE_8M)
print(f"  type : {type(data1)}")
assert len(data1) == SIZE_8M, f"Expected {SIZE_8M}, got {len(data1)}"
print(f"  len  : {len(data1)}  ✓")

# Zero-copy memoryview
view1 = memoryview(data1)
assert len(view1) == SIZE_8M
print(f"  memoryview len : {len(view1)}  ✓")

# Zero-copy numpy
arr1 = np.frombuffer(view1, dtype=np.uint8)
assert arr1.shape == (SIZE_8M,)
print(f"  numpy shape    : {arr1.shape}  ✓")

# Uniqueness
data2 = s3dlio.generate_data(SIZE_8M)
arr2 = np.frombuffer(memoryview(data2), dtype=np.uint8)
diff = int(np.sum(arr1 != arr2))
print(f"  bytes differing between two calls: {diff} (expect > 0)  ✓")
assert diff > 0, "Two calls produced identical data!"
print("  PASS")

# ---------------------------------------------------------------------------
# 2. generate_data() with dedup and compress
# ---------------------------------------------------------------------------
banner("2. generate_data(dedup=4, compress=2)")

data_dc = s3dlio.generate_data(SIZE_8M, dedup=4, compress=2)
assert len(data_dc) == SIZE_8M
print(f"  len : {len(data_dc)}  ✓")
print("  PASS")

# ---------------------------------------------------------------------------
# 3. generate_data_with_threads()
# ---------------------------------------------------------------------------
banner("3. s3dlio.generate_data_with_threads(threads=4)")

data_t = s3dlio.generate_data_with_threads(SIZE_8M, threads=4)
assert len(data_t) == SIZE_8M
print(f"  len : {len(data_t)}  ✓")
view_t = memoryview(data_t)
arr_t = np.frombuffer(view_t, dtype=np.uint8)
print(f"  numpy shape : {arr_t.shape}  ✓")
print("  PASS")

# ---------------------------------------------------------------------------
# 4. generate_into_buffer() — fill user bytearray
# ---------------------------------------------------------------------------
banner("4. s3dlio.generate_into_buffer() — in-place fill")

buf4 = bytearray(SIZE_8M)
n = s3dlio.generate_into_buffer(buf4)
assert n == SIZE_8M, f"Expected {SIZE_8M}, got {n}"
arr4 = np.frombuffer(buf4, dtype=np.uint8)
print(f"  wrote {n} bytes, non-zero: {np.count_nonzero(arr4)}  ✓")
assert np.count_nonzero(arr4) > 0
print("  PASS")

# ---------------------------------------------------------------------------
# 5. Generator.fill_chunk() — streaming, zero-copy write
# ---------------------------------------------------------------------------
banner("5. s3dlio.Generator.fill_chunk() — streaming")

gen5 = s3dlio.Generator(size=SIZE_32M, dedup=1, compress=1)
buf5 = bytearray(SIZE_1M)

total5 = 0
chunks5 = 0
while not gen5.is_complete():
    n = gen5.fill_chunk(buf5)
    if n == 0:
        break
    total5 += n
    chunks5 += 1

print(f"  chunks : {chunks5}, total bytes : {total5}")
assert total5 == SIZE_32M, f"Expected {SIZE_32M}, got {total5}"
print(f"  {total5} bytes generated in {chunks5} chunks  ✓")
print("  PASS")

# ---------------------------------------------------------------------------
# 6. Generator throughput
# ---------------------------------------------------------------------------
banner("6. Generator throughput benchmark")

BENCH_SIZE = 512 * SIZE_1M
gen6 = s3dlio.Generator(size=BENCH_SIZE, dedup=1, compress=1)
buf6 = bytearray(SIZE_32M)

t0 = time.perf_counter()
written6 = 0
while not gen6.is_complete():
    n = gen6.fill_chunk(buf6)
    if n == 0:
        break
    written6 += n
elapsed6 = time.perf_counter() - t0

gbps6 = written6 / elapsed6 / 1e9
print(f"  {written6/1e6:.0f} MB in {elapsed6:.3f}s = {gbps6:.2f} GB/s")
print("  PASS")

# ---------------------------------------------------------------------------
# 7. Generator uniqueness — dedup-safe output
# ---------------------------------------------------------------------------
banner("7. Generator uniqueness — dedup-safe output")

gen7a = s3dlio.Generator(size=SIZE_8M, dedup=1, compress=1)
buf7a = bytearray(SIZE_8M)
gen7a.fill_chunk(buf7a)
arr7a = np.frombuffer(buf7a, dtype=np.uint8).copy()

gen7b = s3dlio.Generator(size=SIZE_8M, dedup=1, compress=1)
buf7b = bytearray(SIZE_8M)
gen7b.fill_chunk(buf7b)
arr7b = np.frombuffer(buf7b, dtype=np.uint8)

diff7 = int(np.sum(arr7a != arr7b))
print(f"  bytes differing between two generators: {diff7} (expect > 0)  ✓")
assert diff7 > 0, "Two Generator calls produced identical data!"
print("  PASS")

# ---------------------------------------------------------------------------
# 8. Concurrent Generator — N threads sharing global Rayon pool
# ---------------------------------------------------------------------------
banner("8. Concurrent Generator — N threads, global Rayon pool")

N_CONC8 = 8
results8 = [None] * N_CONC8
errors8 = []

def worker8(tid: int) -> None:
    try:
        gen = s3dlio.Generator(size=SIZE_8M * 4, dedup=1, compress=1)
        buf = bytearray(SIZE_8M)
        total = 0
        while not gen.is_complete():
            n = gen.fill_chunk(buf)
            if n == 0:
                break
            total += n
        assert total == SIZE_8M * 4, f"tid {tid}: expected {SIZE_8M*4}, got {total}"
        results8[tid] = total
    except Exception as exc:
        errors8.append((tid, exc))

threads8 = [threading.Thread(target=worker8, args=(i,)) for i in range(N_CONC8)]
for t in threads8: t.start()
for t in threads8: t.join()

if errors8:
    print(f"  ERRORS: {errors8}")
    sys.exit(1)
assert all(r == SIZE_8M * 4 for r in results8)
print(f"  {N_CONC8} concurrent generators, each {SIZE_8M*4//1024//1024} MiB  ✓")
print("  PASS")

# ---------------------------------------------------------------------------
# 9. Utility functions
# ---------------------------------------------------------------------------
banner("9. Utility: default_data_gen_threads(), total_cpus()")

threads9 = s3dlio.py_default_data_gen_threads()
cpus9 = s3dlio.py_total_cpus()
print(f"  total_cpus()              : {cpus9}")
print(f"  default_data_gen_threads(): {threads9}")
assert cpus9 > 0
assert threads9 > 0
assert threads9 <= cpus9
print("  PASS")

# ---------------------------------------------------------------------------
# 10. Thread safety — multiple threads calling generate_data concurrently
# ---------------------------------------------------------------------------
banner("10. Concurrent generate_data() from multiple threads")

N_THREADS10 = 8
N_ITERS10 = 16
errors10 = []
results10 = [None] * N_THREADS10


def worker10(tid: int) -> None:
    try:
        for _ in range(N_ITERS10):
            d = s3dlio.generate_data(SIZE_1M)
            assert len(d) == SIZE_1M
        results10[tid] = True
    except Exception as exc:
        errors10.append((tid, exc))


threads10 = [threading.Thread(target=worker10, args=(i,)) for i in range(N_THREADS10)]
for t in threads10:
    t.start()
for t in threads10:
    t.join()

if errors10:
    print(f"  ERRORS: {errors10}")
    sys.exit(1)

assert all(r is True for r in results10)
print(f"  {N_THREADS10} threads × {N_ITERS10} calls = {N_THREADS10*N_ITERS10} total  ✓")
print("  PASS")

# ---------------------------------------------------------------------------
# 11. Tokio safety — no "nested runtime" panic when called from async context
# ---------------------------------------------------------------------------
banner("11. generate_data_with_threads() — no nested Tokio panic")

# Verify calling from a standard thread (not async) works fine
# (s3dlio uses Tokio for async I/O, but data generation is sync Rayon — no Tokio involvement)
results11 = []
exceptions11 = []


def worker11() -> None:
    try:
        for _ in range(4):
            d = s3dlio.generate_data_with_threads(SIZE_1M, threads=2)
            assert len(d) == SIZE_1M
        results11.append(True)
    except Exception as exc:
        exceptions11.append(exc)


t11 = threading.Thread(target=worker11)
t11.start()
t11.join()

if exceptions11:
    print(f"  EXCEPTION: {exceptions11}")
    sys.exit(1)

assert results11 == [True]
print("  No Tokio panic  ✓")
print("  PASS")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*62)
print("  ALL TESTS PASSED ✓")
print("="*62)
