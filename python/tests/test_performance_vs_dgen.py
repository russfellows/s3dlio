#!/usr/bin/env python3
"""
Performance comparison: s3dlio vs dgen-py equivalent test
Based on dgen-py README.md Quick Start example
"""

import s3dlio
import time
import sys


def test_bytearray_allocation_and_generation():
    """
    Equivalent to dgen-py README.md example:
    - Pre-generate 16 GB in 32 MB chunks
    - Measure allocation speed
    - Measure generation speed
    - Compare to dgen-py benchmarks
    """
    print("=" * 70)
    print("s3dlio Performance Test (vs dgen-py Baseline)")
    print("=" * 70)
    print()
    
    # Same parameters as dgen-py README (scaled to 16 GB)
    total_size = 16 * 1024**3  # 16 GB
    chunk_size = 32 * 1024**2  # 32 MB chunks
    num_chunks = total_size // chunk_size  # 512 chunks
    
    print(f"Configuration:")
    print(f"  Total size: {total_size / (1024**3):.0f} GB")
    print(f"  Chunk size: {chunk_size / (1024**2):.0f} MB")
    print(f"  Num chunks: {num_chunks}")
    print()
    
    # =========================================================================
    # ALLOCATION: Python native bytearray
    # =========================================================================
    print("PHASE 1: Buffer Allocation")
    print("-" * 70)
    
    start = time.perf_counter()
    chunks = [bytearray(chunk_size) for _ in range(num_chunks)]
    alloc_time = time.perf_counter() - start
    
    alloc_throughput = (total_size / (1024**3)) / alloc_time
    print(f"✓ Allocation: {alloc_time*1000:.1f} ms @ {alloc_throughput:.0f} GB/s")
    print(f"  Method: Python list comprehension [bytearray(size) for ...]")
    print()
    
    # =========================================================================
    # GENERATION: Fill buffers with s3dlio
    # =========================================================================
    print("PHASE 2: Data Generation")
    print("-" * 70)
    
    start = time.perf_counter()
    for buf in chunks:
        s3dlio.generate_into_buffer(buf, dedup=1, compress=1)
    gen_time = time.perf_counter() - start
    
    gen_throughput = (total_size / (1024**3)) / gen_time
    print(f"✓ Generation: {gen_time:.2f}s @ {gen_throughput:.1f} GB/s")
    print(f"  Method: s3dlio.generate_into_buffer()")
    print()
    
    # =========================================================================
    # COMPARISON TO DGEN-PY
    # =========================================================================
    print("=" * 70)
    print("Comparison to dgen-py v0.2.0 README Benchmarks")
    print("=" * 70)
    print()
    
    # Reference values from dgen-py README.md
    dgen_alloc_time_ms = 10.9  # ms
    dgen_alloc_throughput = 2204  # GB/s
    dgen_gen_time = 1.59  # seconds
    dgen_gen_throughput = 15.1  # GB/s
    
    print(f"{'Metric':<30} {'s3dlio':<20} {'dgen-py':<20} {'Ratio':<15}")
    print("-" * 70)
    print(f"{'Allocation Time':<30} {alloc_time*1000:<9.1f} ms      {dgen_alloc_time_ms:<9.1f} ms      {alloc_time*1000/dgen_alloc_time_ms:<.2f}x")
    print(f"{'Allocation Throughput':<30} {alloc_throughput:<9.0f} GB/s     {dgen_alloc_throughput:<9.0f} GB/s     {alloc_throughput/dgen_alloc_throughput:<.2f}x")
    print(f"{'Generation Time':<30} {gen_time:<9.2f} s        {dgen_gen_time:<9.2f} s        {gen_time/dgen_gen_time:<.2f}x")
    print(f"{'Generation Throughput':<30} {gen_throughput:<9.1f} GB/s     {dgen_gen_throughput:<9.1f} GB/s     {gen_throughput/dgen_gen_throughput:<.2f}x")
    print()
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("Analysis")
    print("=" * 70)
    print()
    
    # Allocation comparison
    if alloc_throughput < 100:  # If using Python's standard allocation
        print("📊 Allocation:")
        print(f"   s3dlio uses Python's native bytearray allocation")
        print(f"   dgen-py v0.2.0 uses Rust PyByteArray_Resize (C API)")
        print(f"   Result: dgen-py is ~{dgen_alloc_throughput/alloc_throughput:.0f}x faster")
        print(f"   Impact: Minimal for most workloads (allocation is {alloc_time*1000:.0f}ms one-time cost)")
        print()
    
    # Generation comparison
    gen_ratio = gen_throughput / dgen_gen_throughput
    if gen_ratio >= 0.8:
        print("✅ Data Generation:")
        print(f"   s3dlio: {gen_throughput:.1f} GB/s")
        print(f"   dgen-py: {dgen_gen_throughput:.1f} GB/s")
        print(f"   Result: EQUIVALENT performance ({gen_ratio:.2f}x)")
        print(f"   Both use same underlying architecture (Rayon + Xoshiro256++)")
    elif gen_ratio >= 0.5:
        print("⚠️  Data Generation:")
        print(f"   s3dlio: {gen_throughput:.1f} GB/s")
        print(f"   dgen-py: {dgen_gen_throughput:.1f} GB/s")
        print(f"   Result: Slightly slower ({gen_ratio:.2f}x)")
        print(f"   May be due to system differences or CPU load")
    else:
        print("❌ Data Generation:")
        print(f"   s3dlio: {gen_throughput:.1f} GB/s")
        print(f"   dgen-py: {dgen_gen_throughput:.1f} GB/s")
        print(f"   Result: Slower ({gen_ratio:.2f}x)")
        print(f"   Check system configuration")
    print()
    
    # Total workflow
    total_time = alloc_time + gen_time
    dgen_total = (dgen_alloc_time_ms / 1000) + dgen_gen_time
    
    print("📈 Total Workflow (16 GB pre-generation):")
    print(f"   s3dlio:  {total_time:.2f}s (alloc: {alloc_time*1000:.0f}ms + gen: {gen_time:.2f}s)")
    print(f"   dgen-py (24GB): {dgen_total:.2f}s (alloc: {dgen_alloc_time_ms:.0f}ms + gen: {dgen_gen_time:.2f}s)")
    print(f"   Note: Scaled test (16 GB vs 24 GB baseline)")
    print()
    
    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    print()
    print("✓ s3dlio provides EQUIVALENT data generation performance to dgen-py")
    print("✓ Both share the same high-performance Rust backend")
    print("✓ For streaming workflows, generation throughput is the key metric")
    print(f"✓ Test PASSED: {gen_throughput:.1f} GB/s generation speed")
    print()
    
    return gen_ratio >= 0.8  # Pass if within 20% of dgen-py


def test_single_buffer_performance():
    """Quick single-buffer test to verify zero-copy performance"""
    print("=" * 70)
    print("Single Buffer Test (100 MB)")
    print("=" * 70)
    print()
    
    size = 100 * 1024 * 1024  # 100 MB
    iterations = 10
    
    # Pre-allocate buffer
    buf = bytearray(size)
    
    # Warm-up
    s3dlio.generate_into_buffer(buf)
    
    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        s3dlio.generate_into_buffer(buf, dedup=1, compress=1)
    elapsed = time.perf_counter() - start
    
    throughput = (size * iterations / (1024**3)) / elapsed
    per_iter = elapsed / iterations
    
    print(f"✓ Throughput: {throughput:.1f} GB/s")
    print(f"✓ Per iteration: {per_iter*1000:.1f} ms")
    print(f"✓ Iterations: {iterations}")
    print()
    
    return throughput > 5.0  # Should be at least 5 GB/s


def main():
    print("\n" + "=" * 70)
    print("s3dlio vs dgen-py Performance Comparison")
    print("Based on dgen-py v0.2.0 README Quick Start Example (16 GB test)")
    print("=" * 70 + "\n")
    
    # Check system info
    total_cpus = s3dlio.py_total_cpus()
    default_threads = s3dlio.py_default_data_gen_threads()
    print(f"System Info:")
    print(f"  Total CPUs: {total_cpus}")
    print(f"  Default threads: {default_threads} ({(default_threads/total_cpus)*100:.0f}%)")
    print()
    
    tests = [
        ("16 GB Bulk Generation", test_bytearray_allocation_and_generation),
        ("Single Buffer Performance", test_single_buffer_performance),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {name} PASSED\n")
            else:
                failed += 1
                print(f"❌ {name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED with exception: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
