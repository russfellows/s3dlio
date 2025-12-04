#!/usr/bin/env python3
"""
s3dlio Parallel Operations Example

Demonstrates high-performance parallel I/O operations using s3dlio.
Shows how to efficiently read and write many objects concurrently.

Key concepts:
- max_in_flight: Controls concurrency (number of parallel operations)
- Parallel PUT: Create many objects simultaneously  
- Parallel GET: Read many objects simultaneously
- Performance measurement

Usage:
    python examples/python/parallel_operations.py
    python examples/python/parallel_operations.py s3://mybucket/prefix/
"""

import os
import sys
import tempfile
import time

try:
    import s3dlio
except ImportError:
    print("ERROR: s3dlio module not found.")
    print("Build and install with: ./build_pyo3.sh && ./install_pyo3_wheel.sh")
    sys.exit(1)


def format_throughput(bytes_count: int, elapsed: float) -> str:
    """Format throughput as human-readable string."""
    if elapsed == 0:
        return "N/A"
    mb_per_sec = bytes_count / elapsed / (1024 * 1024)
    return f"{mb_per_sec:.2f} MB/s"


def format_ops(ops_count: int, elapsed: float) -> str:
    """Format operations per second."""
    if elapsed == 0:
        return "N/A"
    return f"{ops_count / elapsed:.1f} ops/s"


def main():
    print("=" * 60)
    print("s3dlio Parallel Operations Example")
    print("=" * 60)
    
    # Determine base URI from command line or use local filesystem
    if len(sys.argv) > 1:
        base_uri = sys.argv[1]
        if not base_uri.endswith('/'):
            base_uri += '/'
    else:
        temp_dir = tempfile.mkdtemp(prefix="s3dlio_parallel_example_")
        base_uri = f"file://{temp_dir}/"
        print(f"\nUsing local filesystem: {base_uri}")
    
    print(f"Target URI: {base_uri}")
    
    # Test parameters
    num_objects = 100
    object_size = 64 * 1024  # 64 KB per object
    total_bytes = num_objects * object_size
    
    print(f"\nTest configuration:")
    print(f"  Objects: {num_objects}")
    print(f"  Object size: {object_size // 1024} KB")
    print(f"  Total data: {total_bytes // (1024 * 1024)} MB")
    
    try:
        # =====================================================================
        # Test 1: Parallel PUT with varying concurrency
        # =====================================================================
        print("\n" + "=" * 60)
        print("📤 Parallel PUT Performance")
        print("=" * 60)
        
        concurrency_levels = [1, 4, 16, 64]
        
        for concurrency in concurrency_levels:
            # Use a unique prefix for each test
            test_prefix = f"{base_uri}put_test_{concurrency}/"
            
            start_time = time.time()
            s3dlio.put(
                prefix=test_prefix,
                num=num_objects,
                template="object_{}.bin",
                size=object_size,
                object_type="zeros",  # Fast to generate
                max_in_flight=concurrency,
            )
            elapsed = time.time() - start_time
            
            print(f"  Concurrency {concurrency:2d}: {elapsed:.3f}s | "
                  f"{format_throughput(total_bytes, elapsed)} | "
                  f"{format_ops(num_objects, elapsed)}")
            
            # Cleanup this test
            s3dlio.delete(test_prefix, recursive=True)
        
        # =====================================================================
        # Test 2: Parallel GET with varying concurrency
        # =====================================================================
        print("\n" + "=" * 60)
        print("📥 Parallel GET Performance")
        print("=" * 60)
        
        # First, create test objects
        test_prefix = f"{base_uri}get_test/"
        print(f"  Creating {num_objects} test objects...")
        s3dlio.put(
            prefix=test_prefix,
            num=num_objects,
            template="object_{}.bin",
            size=object_size,
            object_type="zeros",
            max_in_flight=64,
        )
        
        # List objects to get their URIs
        object_uris = s3dlio.list(test_prefix, recursive=False)
        print(f"  Created {len(object_uris)} objects")
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            results = s3dlio.get_many(object_uris, max_in_flight=concurrency)
            elapsed = time.time() - start_time
            
            bytes_read = sum(len(bytes(data)) for _, data in results)
            
            print(f"  Concurrency {concurrency:2d}: {elapsed:.3f}s | "
                  f"{format_throughput(bytes_read, elapsed)} | "
                  f"{format_ops(len(results), elapsed)}")
        
        # Cleanup
        s3dlio.delete(test_prefix, recursive=True)
        
        # =====================================================================
        # Test 3: Mixed workload (interleaved reads and writes)
        # =====================================================================
        print("\n" + "=" * 60)
        print("🔄 Mixed Workload (PUT + GET)")
        print("=" * 60)
        
        test_prefix = f"{base_uri}mixed_test/"
        iterations = 5
        objects_per_iteration = 20
        
        print(f"  Running {iterations} iterations of PUT {objects_per_iteration} + GET all...")
        
        total_put_bytes = 0
        total_get_bytes = 0
        start_time = time.time()
        
        for i in range(iterations):
            # PUT some objects
            iter_prefix = f"{test_prefix}iter_{i}/"
            s3dlio.put(
                prefix=iter_prefix,
                num=objects_per_iteration,
                template="obj_{}.bin",
                size=object_size,
                object_type="zeros",
                max_in_flight=16,
            )
            total_put_bytes += objects_per_iteration * object_size
            
            # GET all objects created so far
            all_objects = s3dlio.list(test_prefix, recursive=True)
            if all_objects:
                results = s3dlio.get_many(all_objects, max_in_flight=16)
                total_get_bytes += sum(len(bytes(data)) for _, data in results)
        
        elapsed = time.time() - start_time
        
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  PUT: {total_put_bytes // (1024*1024)} MB | {format_throughput(total_put_bytes, elapsed)}")
        print(f"  GET: {total_get_bytes // (1024*1024)} MB | {format_throughput(total_get_bytes, elapsed)}")
        print(f"  Combined: {format_throughput(total_put_bytes + total_get_bytes, elapsed)}")
        
        # Cleanup
        s3dlio.delete(test_prefix, recursive=True)
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("✅ Parallel operations examples completed!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("  - Higher concurrency generally improves throughput")
        print("  - Optimal concurrency depends on backend and network")
        print("  - For S3/Azure/GCS: try 32-128 concurrent operations")
        print("  - For local files: 4-16 is often sufficient")
        
    finally:
        # Clean up temp directory if we created one
        if base_uri.startswith("file://") and "s3dlio_parallel_example_" in base_uri:
            import shutil
            temp_dir = base_uri.replace("file://", "").rstrip('/')
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
