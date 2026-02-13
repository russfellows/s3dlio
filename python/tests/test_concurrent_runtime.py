#!/usr/bin/env python3
"""
Test concurrent Python API calls from multiple threads.

This validates the io_uring-style runtime fix (v0.9.50) that replaced the broken
GLOBAL_RUNTIME.block_on() pattern with submit_io() → run_on_global_rt() which
spawns work onto a single background Tokio runtime via spawn+channel.

The TWO bugs this test validates are fixed:
  1. v0.9.27: Per-call Runtime::new() → "dispatch failure" after ~40 objects
  2. v0.9.40: GLOBAL_RUNTIME.block_on() → "Cannot start a runtime from within
     a runtime" panic when called from ThreadPoolExecutor threads

Usage:
    python tests/test_concurrent_runtime.py [--threads N] [--objects N] [--size N]
"""

import argparse
import os
import sys
import time
import tempfile
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Concurrent s3dlio runtime test")
    p.add_argument("--threads", type=int, default=16,
                   help="Number of ThreadPoolExecutor workers (default: 16)")
    p.add_argument("--objects", type=int, default=200,
                   help="Number of objects to create/read (default: 200)")
    p.add_argument("--size", type=int, default=65536,
                   help="Object size in bytes (default: 65536)")
    p.add_argument("--rounds", type=int, default=3,
                   help="Number of put/get/verify rounds (default: 3)")
    return p.parse_args()


def banner(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def test_concurrent_put_bytes(s3dlio, base_uri, num_threads, num_objects, obj_size):
    """Put objects concurrently from multiple threads — triggers the old bugs."""
    banner(f"PUT {num_objects} objects × {obj_size}B  ({num_threads} threads)")
    data = os.urandom(obj_size)
    errors = []

    def put_one(i):
        uri = f"{base_uri}/obj_{i:06d}.bin"
        s3dlio.put_bytes(uri, data)
        return i

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(put_one, i): i for i in range(num_objects)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:
                errors.append((futs[f], exc))

    elapsed = time.monotonic() - t0
    rate = num_objects / elapsed
    total_mb = (num_objects * obj_size) / (1024 * 1024)

    print(f"  Completed {num_objects - len(errors)}/{num_objects} puts in {elapsed:.2f}s")
    print(f"  {rate:.0f} ops/s  |  {total_mb / elapsed:.1f} MB/s")

    if errors:
        print(f"  *** {len(errors)} ERRORS:")
        for idx, exc in errors[:5]:
            print(f"      obj {idx}: {exc}")
    return errors


def test_concurrent_get(s3dlio, base_uri, num_threads, num_objects, obj_size):
    """Get objects concurrently from multiple threads."""
    banner(f"GET {num_objects} objects  ({num_threads} threads)")
    errors = []

    def get_one(i):
        uri = f"{base_uri}/obj_{i:06d}.bin"
        data = s3dlio.get(uri)
        if len(data) != obj_size:
            raise ValueError(f"Expected {obj_size} bytes, got {len(data)}")
        return i

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(get_one, i): i for i in range(num_objects)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:
                errors.append((futs[f], exc))

    elapsed = time.monotonic() - t0
    rate = num_objects / elapsed
    total_mb = (num_objects * obj_size) / (1024 * 1024)

    print(f"  Completed {num_objects - len(errors)}/{num_objects} gets in {elapsed:.2f}s")
    print(f"  {rate:.0f} ops/s  |  {total_mb / elapsed:.1f} MB/s")

    if errors:
        print(f"  *** {len(errors)} ERRORS:")
        for idx, exc in errors[:5]:
            print(f"      obj {idx}: {exc}")
    return errors


def test_concurrent_stat(s3dlio, base_uri, num_threads, num_objects):
    """Stat objects concurrently from multiple threads."""
    banner(f"STAT {num_objects} objects  ({num_threads} threads)")
    errors = []

    def stat_one(i):
        uri = f"{base_uri}/obj_{i:06d}.bin"
        info = s3dlio.stat(uri)
        return i

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(stat_one, i): i for i in range(num_objects)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:
                errors.append((futs[f], exc))

    elapsed = time.monotonic() - t0
    print(f"  Completed {num_objects - len(errors)}/{num_objects} stats in {elapsed:.2f}s")
    print(f"  {num_objects / elapsed:.0f} ops/s")

    if errors:
        print(f"  *** {len(errors)} ERRORS:")
        for idx, exc in errors[:5]:
            print(f"      obj {idx}: {exc}")
    return errors


def test_concurrent_exists(s3dlio, base_uri, num_threads, num_objects):
    """Exists check concurrently from multiple threads."""
    banner(f"EXISTS {num_objects} objects  ({num_threads} threads)")
    errors = []

    def exists_one(i):
        uri = f"{base_uri}/obj_{i:06d}.bin"
        result = s3dlio.exists(uri)
        if not result:
            raise ValueError(f"Expected object to exist: {uri}")
        return i

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(exists_one, i): i for i in range(num_objects)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:
                errors.append((futs[f], exc))

    elapsed = time.monotonic() - t0
    print(f"  Completed {num_objects - len(errors)}/{num_objects} exists in {elapsed:.2f}s")
    print(f"  {num_objects / elapsed:.0f} ops/s")

    if errors:
        print(f"  *** {len(errors)} ERRORS:")
        for idx, exc in errors[:5]:
            print(f"      obj {idx}: {exc}")
    return errors


def test_concurrent_delete(s3dlio, base_uri, num_threads, num_objects):
    """Delete objects concurrently from multiple threads."""
    banner(f"DELETE {num_objects} objects  ({num_threads} threads)")
    errors = []

    def delete_one(i):
        uri = f"{base_uri}/obj_{i:06d}.bin"
        s3dlio.delete(uri)
        return i

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(delete_one, i): i for i in range(num_objects)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:
                errors.append((futs[f], exc))

    elapsed = time.monotonic() - t0
    print(f"  Completed {num_objects - len(errors)}/{num_objects} deletes in {elapsed:.2f}s")
    print(f"  {num_objects / elapsed:.0f} ops/s")

    if errors:
        print(f"  *** {len(errors)} ERRORS:")
        for idx, exc in errors[:5]:
            print(f"      obj {idx}: {exc}")
    return errors


def test_put_many_batch(s3dlio, base_uri, num_objects, obj_size):
    """Test the new put_many() batch function."""
    banner(f"PUT_MANY batch: {num_objects} objects × {obj_size}B")
    items = [
        (f"{base_uri}/batch_{i:06d}.bin", os.urandom(obj_size))
        for i in range(num_objects)
    ]

    t0 = time.monotonic()
    try:
        s3dlio.put_many(items, max_in_flight=64)
        elapsed = time.monotonic() - t0
        total_mb = (num_objects * obj_size) / (1024 * 1024)
        print(f"  Completed {num_objects} puts in {elapsed:.2f}s")
        print(f"  {num_objects / elapsed:.0f} ops/s  |  {total_mb / elapsed:.1f} MB/s")
        return []
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  *** FAILED after {elapsed:.2f}s: {exc}")
        return [(0, exc)]


def test_mixed_concurrent(s3dlio, base_uri, num_threads, num_objects, obj_size):
    """Mixed PUT + GET + STAT concurrently — stress test."""
    banner(f"MIXED ops: {num_objects * 3} total  ({num_threads} threads)")
    data = os.urandom(obj_size)
    errors = []

    def mixed_op(i):
        """Each task: put → get → stat on the same object."""
        uri = f"{base_uri}/mixed_{i:06d}.bin"
        s3dlio.put_bytes(uri, data)
        got = s3dlio.get(uri)
        if len(got) != obj_size:
            raise ValueError(f"Size mismatch: expected {obj_size}, got {len(got)}")
        s3dlio.stat(uri)
        return i

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(mixed_op, i): i for i in range(num_objects)}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as exc:
                errors.append((futs[f], exc))

    elapsed = time.monotonic() - t0
    total_ops = (num_objects - len(errors)) * 3
    print(f"  Completed {num_objects - len(errors)}/{num_objects} mixed sequences in {elapsed:.2f}s")
    print(f"  {total_ops / elapsed:.0f} total ops/s")

    if errors:
        print(f"  *** {len(errors)} ERRORS:")
        for idx, exc in errors[:5]:
            print(f"      obj {idx}: {exc}")
    return errors


def main():
    args = parse_args()
    print("=" * 60)
    print("  s3dlio Concurrent Runtime Test (v0.9.50 io_uring fix)")
    print("=" * 60)
    print(f"  Threads: {args.threads}  Objects: {args.objects}  Size: {args.size}B")
    print(f"  Rounds:  {args.rounds}")

    import s3dlio
    print(f"\n  s3dlio loaded from: {s3dlio.__file__}")

    # Create temp directory for file:// backend testing
    tmp_dir = tempfile.mkdtemp(prefix="s3dlio_rt_test_")
    base_uri = f"file://{tmp_dir}"
    print(f"  Test URI base: {base_uri}")

    all_errors = []
    try:
        for round_num in range(1, args.rounds + 1):
            round_uri = f"{base_uri}/round_{round_num}"
            print(f"\n{'#' * 60}")
            print(f"  ROUND {round_num}/{args.rounds}")
            print(f"{'#' * 60}")

            # Core operations that triggered the original bugs
            all_errors += test_concurrent_put_bytes(
                s3dlio, round_uri, args.threads, args.objects, args.size)
            all_errors += test_concurrent_get(
                s3dlio, round_uri, args.threads, args.objects, args.size)
            all_errors += test_concurrent_stat(
                s3dlio, round_uri, args.threads, args.objects)
            all_errors += test_concurrent_exists(
                s3dlio, round_uri, args.threads, args.objects)

            # Batch put_many
            batch_uri = f"{base_uri}/batch_round_{round_num}"
            all_errors += test_put_many_batch(
                s3dlio, batch_uri, min(args.objects, 100), args.size)

            # Mixed concurrent operations (put+get+stat per thread)
            mixed_uri = f"{base_uri}/mixed_round_{round_num}"
            all_errors += test_mixed_concurrent(
                s3dlio, mixed_uri, args.threads, args.objects, args.size)

            # Cleanup: delete all objects from this round
            all_errors += test_concurrent_delete(
                s3dlio, round_uri, args.threads, args.objects)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    banner("RESULTS")
    if all_errors:
        print(f"  FAIL: {len(all_errors)} errors across all rounds")
        for idx, exc in all_errors[:10]:
            print(f"    obj {idx}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    else:
        print(f"  PASS: All operations completed without errors")
        print(f"  Threads={args.threads}  Objects={args.objects}  Rounds={args.rounds}")
        sys.exit(0)


if __name__ == "__main__":
    main()
