#!/usr/bin/env python3
"""Test if passing threads parameter helps performance"""
import s3dlio
import time

size = 100 * 1024**2
buf = bytearray(size)

# Test different thread counts
for threads in [None, 1, 4, 8, 12]:
    start = time.perf_counter()
    if threads is None:
        s3dlio.generate_into_buffer(buf, dedup=1, compress=1)
        label = "default"
    else:
        s3dlio.generate_into_buffer(buf, dedup=1, compress=1, threads=threads)
        label = f"{threads} threads"
    elapsed = time.perf_counter() - start
    print(f"{label:15s}: {elapsed*1000:6.1f} ms = {0.1/elapsed:5.1f} GB/s")
