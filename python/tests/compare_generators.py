#!/usr/bin/env python3
"""Fair comparison: dgen-py vs s3dlio using Generator streaming API"""
import dgen_py
import s3dlio
import time

size_per_chunk = 100 * 1024**2  # 100 MB
total_size = 16 * 1024**3  # 16 GB
buf = bytearray(size_per_chunk)

print("=" * 60)
print("Fair Performance Comparison: Generator Streaming API")
print("=" * 60)
print(f"Total size: {total_size / 1024**3:.1f} GB")
print(f"Chunk size: {size_per_chunk / 1024**2:.0f} MB")
print()

# Test dgen-py
print("Testing dgen-py Generator...")
gen1 = dgen_py.Generator(size=total_size, dedup_ratio=1.0, compress_ratio=1.0)

# Warm-up
gen1.fill_chunk(buf)
gen1.reset()

# Timed test
start = time.perf_counter()
while not gen1.is_complete():
    nbytes = gen1.fill_chunk(buf)
    if nbytes == 0:
        break
elapsed_dgen = time.perf_counter() - start

throughput_dgen = (total_size / 1024**3) / elapsed_dgen
print(f"  Time:       {elapsed_dgen:.2f}s")
print(f"  Throughput: {throughput_dgen:.1f} GB/s")
print()

# Test s3dlio
print("Testing s3dlio Generator...")
gen2 = s3dlio.Generator(size=total_size, dedup=1, compress=1)

# Warm-up
gen2.fill_chunk(buf)
gen2.reset()

# Timed test  
start = time.perf_counter()
while not gen2.is_complete():
    nbytes = gen2.fill_chunk(buf)
    if nbytes == 0:
        break
elapsed_s3dlio = time.perf_counter() - start

throughput_s3dlio = (total_size / 1024**3) / elapsed_s3dlio
print(f"  Time:       {elapsed_s3dlio:.2f}s")
print(f"  Throughput: {throughput_s3dlio:.1f} GB/s")
print()

# Results
print("=" * 60)
ratio = throughput_dgen / throughput_s3dlio
if ratio > 1.1:
    print(f"❌ dgen-py {ratio:.1f}x FASTER than s3dlio")
elif ratio < 0.9:
    print(f"✅ s3dlio {1/ratio:.1f}x FASTER than dgen-py")
else:
    print(f"✅ Performance parity: {abs(ratio-1)*100:.1f}% difference")
print("=" * 60)
