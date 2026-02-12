#!/usr/bin/env python3
import s3dlio
import time

print("Config:")
print(f"  CPUs: {s3dlio.py_total_cpus()}")
print(f"  Threads: {s3dlio.py_default_data_gen_threads()}")
print()

# Test 1: Single 100 MB generation
size = 100 * 1024**2
buf = bytearray(size)
start = time.perf_counter()
s3dlio.generate_into_buffer(buf, dedup=1, compress=1)
elapsed = time.perf_counter() - start
print(f"Single 100 MB: {elapsed*1000:.1f} ms = {0.1/elapsed:.1f} GB/s")

# Test 2: 10x 100 MB with warm-up
start = time.perf_counter()
for _ in range(10):
    s3dlio.generate_into_buffer(buf, dedup=1, compress=1)
elapsed = time.perf_counter() - start
print(f"10x 100 MB: {elapsed:.2f}s = {1.0/elapsed:.1f} GB/s")

# Test 3: generate_data
start = time.perf_counter()
for _ in range(10):
    data = s3dlio.generate_data(size, dedup=1, compress=1)
elapsed = time.perf_counter() - start
print(f"10x generate_data(): {elapsed:.2f}s = {1.0/elapsed:.1f} GB/s")
