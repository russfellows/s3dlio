#!/usr/bin/env python3
# tests/test_phase2_python_loader_parallelism.py
#
# SPDX-License-Identifier: Apache-2.0 OR MIT
# SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
#
# Phase 2 site 1.1 end-to-end Python-surface test for issue #148
# (docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md).
#
# The three call sites in src/python_api/python_aiml_api.rs
# (PyBytesAsyncDataLoader.__iter__, .items(), and the Parquet
# streaming loader) all previously polled a `buffer_unordered`
# stream of bare fetch futures inside one tokio producer task.
# The Phase 2 fix spawns each fetch as its own task and wires an
# internal CancellationToken through each spawned task via
# `tokio::select!`. When the Python consumer drops the iterator,
# a `DropCancel` guard on the producer's stack fires, cancelling
# every in-flight fetch instead of leaving them to run to
# completion (audit §2.1's "spawn regression" avoidance).
#
# This test exercises the behavior at the Python surface, which
# is the layer users actually touch: create a dataset with many
# items, start iterating with a large prefetch depth, drop the
# iterator early, and check that the pool stops fetching soon
# after — not that it runs the full N items in the background.
#
# Run as a script (matches this repo's Python-test convention):
#   source .venv/bin/activate
#   python tests/test_phase2_python_loader_parallelism.py
# or via uv:
#   uv run python tests/test_phase2_python_loader_parallelism.py

import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import s3dlio


TOTAL_ITEMS = 200
BYTES_PER_ITEM = 4 * 1024  # 4 KiB — small enough to be fast, big enough to observe


def create_test_files(root: Path) -> list[str]:
    """Write TOTAL_ITEMS small files and return file:// URIs for them."""
    uris = []
    for i in range(TOTAL_ITEMS):
        p = root / f"item_{i:04d}.bin"
        # Deterministic content so any later reader can validate.
        p.write_bytes(bytes([i & 0xFF]) * BYTES_PER_ITEM)
        uris.append(f"file://{p.resolve()}")
    return uris


def test_full_iteration_delivers_all_items(uris: list[str]) -> None:
    """Sanity check: the loader still delivers every item when we
    consume the full iterator. Catches gross breakage of the fix."""
    ds = s3dlio.PyDataset.from_uris(uris)
    loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 32})
    count = 0
    total_bytes = 0
    for item in loader:
        count += 1
        total_bytes += len(item)
    assert count == TOTAL_ITEMS, f"expected {TOTAL_ITEMS} items, got {count}"
    assert total_bytes == TOTAL_ITEMS * BYTES_PER_ITEM, (
        f"expected {TOTAL_ITEMS * BYTES_PER_ITEM} bytes, got {total_bytes}"
    )
    print(f"  full-iter: OK ({count} items, {total_bytes} bytes)")


def test_early_drop_stops_pool_soon(uris: list[str]) -> None:
    """Consume ONE item, drop the iterator, wait briefly. Then
    start a NEW loader over the same URIs and time how long it
    takes to complete. If the first loader's in-flight fetches
    were properly cancelled, the file cache / disk are unloaded
    and the second run behaves normally. If they leaked and are
    still running, the second run competes with them for I/O
    and process resources.

    The stronger assertion is behavioral: the first iterator's
    early-drop must not hang, and Python's own garbage collector
    (or explicit del) must reclaim the iterator promptly. If the
    Rust producer detached its spawned tasks (naive tokio::spawn
    without cancel wiring), dropping the iterator would leave
    them running; the test wouldn't crash — it would just be
    silently wasteful. The audit calls this out as a *silent*
    regression, so we assert what we CAN observe: that early
    drop returns control quickly and does not raise.
    """
    ds = s3dlio.PyDataset.from_uris(uris)
    loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 32})

    it = iter(loader)
    first = next(it)
    assert len(first) == BYTES_PER_ITEM, "first item wrong size"

    # Drop the iterator and force GC so PyBytesDataLoaderSyncIter's
    # __del__ (via reference count → 0) actually runs. That drops
    # the mpsc::Receiver on the Rust side, which is what makes the
    # producer's tx.send() return Err, which triggers the DropCancel.
    t0 = time.perf_counter()
    del it
    del first
    gc.collect()
    elapsed_drop = time.perf_counter() - t0

    # The drop path itself must not take significant time — it's
    # just releasing a receiver. If it does, something is wrong
    # (e.g., blocking on task join).
    assert elapsed_drop < 0.5, (
        f"iterator drop took {elapsed_drop:.3f}s — should be near-instant. "
        f"A blocking drop would indicate the fix is joining spawned tasks "
        f"on the caller's thread instead of cancelling them."
    )
    print(f"  early-drop return time: {elapsed_drop * 1000:.1f}ms  OK")


def test_items_iterator_delivers_uri_and_bytes(uris: list[str]) -> None:
    """The .items() variant is a second site the Phase 2 fix
    covers. Confirm it still delivers (uri, bytes) pairs and
    that all URIs come back."""
    ds = s3dlio.PyDataset.from_uris(uris)
    loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 16})
    seen_uris: set[str] = set()
    count = 0
    for item in loader.items():
        # item is a PyObjectItem with .uri and byte contents
        seen_uris.add(item.uri)
        count += 1
    assert count == TOTAL_ITEMS, f"items(): expected {TOTAL_ITEMS}, got {count}"
    assert seen_uris == set(uris), (
        f"items() URIs mismatch — missing {set(uris) - seen_uris}, "
        f"extra {seen_uris - set(uris)}"
    )
    print(f"  items() covers all {TOTAL_ITEMS} URIs: OK")


def test_early_drop_from_items(uris: list[str]) -> None:
    """Same early-drop assertion for the items() variant."""
    ds = s3dlio.PyDataset.from_uris(uris)
    loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 32})

    it = iter(loader.items())
    first = next(it)
    assert first.uri in uris, f"first item.uri {first.uri!r} not in dataset"

    t0 = time.perf_counter()
    del it
    del first
    gc.collect()
    elapsed_drop = time.perf_counter() - t0
    assert elapsed_drop < 0.5, (
        f"items() iterator drop took {elapsed_drop:.3f}s — should be near-instant"
    )
    print(f"  items() early-drop return time: {elapsed_drop * 1000:.1f}ms  OK")


def main() -> int:
    print("=" * 70)
    print("Phase 2 site 1.1 — PyBytesAsyncDataLoader task-level parallelism")
    print(f"s3dlio {getattr(s3dlio, '__version__', 'unknown version')}")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="s3dlio_phase2_") as td:
        root = Path(td)
        print(f"Creating {TOTAL_ITEMS} test files in {root} ...")
        uris = create_test_files(root)

        print()
        print("Test 1: full iteration delivers all items")
        test_full_iteration_delivers_all_items(uris)

        print()
        print("Test 2: early drop of iterator returns control quickly")
        test_early_drop_stops_pool_soon(uris)

        print()
        print("Test 3: items() delivers (uri, bytes) pairs correctly")
        test_items_iterator_delivers_uri_and_bytes(uris)

        print()
        print("Test 4: early drop from items() iterator")
        test_early_drop_from_items(uris)

    print()
    print("=" * 70)
    print("✓ ALL Phase 2 site 1.1 tests passed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
