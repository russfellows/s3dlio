// tests/test_phase1_zero_copy_assembly.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Phase 1 RED/GREEN tests for issue #148
// (docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md, sections 1.3, 3.3):
//
// The current range-assembly code paths allocate the final output buffer
// AND hold all per-range Bytes simultaneously in an intermediate `parts`
// Vec while the output is built. That drives peak memory to ~2 * total
// object size. The Phase 1 fix pre-allocates the output buffer, processes
// each range's result as it completes, copies into the pre-allocated
// offset, and drops the source Bytes immediately, holding peak memory to
// ~1 * total object size plus a small in-flight budget.
//
// These tests install a peak-tracking global allocator (defined inline
// below — nothing else uses it yet, so it lives here rather than in a
// shared helper module) to make the peak-memory difference observable,
// and assert that peak overhead stays below a threshold that only the
// fixed code can satisfy. Against the unmodified code they FAIL (RED).
// After Patch 3 lands they PASS (GREEN).

use bytes::Bytes;
use s3dlio::range_engine_generic::{RangeEngine, RangeEngineConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Peak-tracking wrapper around System. Reports the maximum
/// (bytes_allocated - bytes_deallocated) observed since the last `reset()`.
/// stats_alloc only exposes cumulative counters, which cannot distinguish
/// a temporal peak from a steady-state footprint of the same size — the
/// bug being tested here is a temporal peak difference, so peak tracking
/// is the metric we need.
struct PeakAlloc {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl PeakAlloc {
    const fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Reset peak to the current live total. Call at the start of the
    /// region to measure, after any warmup has finished.
    fn reset(&self) {
        let current = self.live.load(Ordering::Relaxed);
        self.peak.store(current, Ordering::Relaxed);
    }

    fn note_grow(&self, delta: usize) {
        let new_live = self.live.fetch_add(delta, Ordering::Relaxed) + delta;
        let mut peak = self.peak.load(Ordering::Relaxed);
        while new_live > peak {
            match self.peak.compare_exchange_weak(
                peak,
                new_live,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }
}

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            self.note_grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.live.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size > old_size {
                self.note_grow(new_size - old_size);
            } else if new_size < old_size {
                self.live.fetch_sub(old_size - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc::new();

/// Return a fresh, freshly-allocated Bytes of the requested length filled
/// with a deterministic value derived from the offset. Using
/// `Bytes::from(vec![...])` on every call is intentional — it makes the
/// per-range allocations real so the allocator can see them.
fn allocate_range(offset: u64, length: u64) -> Bytes {
    // Deterministic byte value so we can also verify correctness of the
    // assembled result.
    let fill = ((offset >> 20) & 0xFF) as u8;
    Bytes::from(vec![fill; length as usize])
}

fn make_engine(chunk_size: usize, max_concurrent_ranges: usize) -> RangeEngine {
    RangeEngine::new(RangeEngineConfig {
        chunk_size,
        max_concurrent_ranges,
        // Force range mode even for our small test object.
        min_split_size: 1,
        range_timeout: Duration::from_secs(30),
    })
}

/// Warm up the allocator so any lazy-init allocations (once_cell, ring
/// initialization, etc.) happen before the measured region.
async fn warmup(engine: &RangeEngine) {
    // Small, throwaway download.
    let _ = engine
        .download(
            4096,
            |off, len| async move { Ok(allocate_range(off, len)) },
            None,
        )
        .await
        .expect("warmup download failed");
    // Drop scratch allocations before the next measurement.
    for _ in 0..8 {
        let scratch: Vec<u8> = vec![0; 4096];
        drop(scratch);
    }
}

/// Assert the assembled output has the expected total length and that each
/// byte reflects the expected per-range fill pattern.
fn assert_output_matches(total_size: usize, chunk_size: usize, bytes: &Bytes) {
    assert_eq!(bytes.len(), total_size, "assembled length mismatch");
    for (i, &b) in bytes.iter().enumerate() {
        let range_offset = (i / chunk_size) * chunk_size;
        let expected = ((range_offset as u64 >> 20) & 0xFF) as u8;
        assert_eq!(
            b, expected,
            "byte {} in range starting at {} should equal {}, got {}",
            i, range_offset, expected, b
        );
    }
}

/// Phase 1 RED — peak-memory bound during ordered concurrent-range assembly.
///
/// Configuration:
/// * total_size = 32 MiB
/// * chunk_size = 1 MiB  → 32 ranges
/// * max_concurrent_ranges = 4
///
/// Current (buggy) code accumulates every completed range's Bytes into an
/// intermediate `parts` Vec while separately allocating a `Vec::with_capacity(total_size)`
/// for `assembled`. Peak live memory during the loop is roughly:
///     total_size  (all ranges retained in `parts`)
///   + total_size  (`assembled` being built)
///   = ~2 * total_size = ~64 MiB peak overhead.
///
/// Phase 1 fix pre-allocates the master buffer once and copies each range's
/// Bytes into its offset as it completes, releasing the source Bytes
/// immediately. Peak stays at ~total_size plus at most
/// max_concurrent_ranges * chunk_size in flight ≈ 32 MiB + 4 MiB = ~36 MiB.
///
/// The threshold below (1.5 * total_size = 48 MiB) is well under the
/// current code's ~64 MiB peak and well above the fixed code's ~36 MiB
/// peak, so it cleanly separates the two.
#[tokio::test(flavor = "current_thread")]
async fn range_engine_download_peak_memory_bounded() {
    let chunk_size: usize = 1024 * 1024;
    let n_ranges: usize = 32;
    let total_size: usize = chunk_size * n_ranges;
    let max_concurrent_ranges: usize = 4;

    let engine = make_engine(chunk_size, max_concurrent_ranges);
    warmup(&engine).await;

    ALLOC.reset();
    let baseline_live = ALLOC.live();

    let (bytes, stats) = engine
        .download(
            total_size as u64,
            |off, len| async move { Ok(allocate_range(off, len)) },
            None,
        )
        .await
        .expect("download failed");

    let peak = ALLOC.peak();
    let overhead = peak.saturating_sub(baseline_live);

    // Correctness first.
    assert_eq!(
        stats.ranges_processed, n_ranges,
        "expected {} ranges",
        n_ranges
    );
    assert_output_matches(total_size, chunk_size, &bytes);

    let threshold = (total_size as f64 * 1.5) as usize;
    eprintln!(
        "range_engine peak overhead: {} bytes ({:.2}x total_size {})",
        overhead,
        overhead as f64 / total_size as f64,
        total_size,
    );
    assert!(
        overhead <= threshold,
        "Peak memory overhead {} bytes ({:.2}x total_size {}) exceeds threshold {} bytes (1.50x). \
         Current double-copy code holds all range Bytes in `parts` while separately allocating the \
         assembled Vec, driving peak to ~2x. Patch 3 pre-allocates and copies into offset, keeping \
         peak at ~1.1x. If this fails against unmodified main it's the expected RED state; if it \
         fails after the fix, the fix isn't reducing peak memory as intended.",
        overhead,
        overhead as f64 / total_size as f64,
        total_size,
        threshold,
    );
}

/// Phase 1 correctness regression — many small ranges with a low
/// concurrency cap, exercising the assembly path repeatedly and checking
/// byte-for-byte identity. Passes against both buggy and fixed code; the
/// point is to catch a broken FIX implementation (off-by-one in offset
/// math, wrong split-point, dropped range, etc.), not to prove the
/// pre-fix bug exists.
#[tokio::test(flavor = "current_thread")]
async fn range_engine_download_correct_bytes_many_ranges() {
    let chunk_size: usize = 128 * 1024;
    let n_ranges: usize = 64;
    let total_size: usize = chunk_size * n_ranges;
    let max_concurrent_ranges: usize = 2;

    let engine = make_engine(chunk_size, max_concurrent_ranges);
    let (bytes, stats) = engine
        .download(
            total_size as u64,
            |off, len| async move { Ok(allocate_range(off, len)) },
            None,
        )
        .await
        .expect("download failed");

    assert_eq!(stats.ranges_processed, n_ranges);
    assert_output_matches(total_size, chunk_size, &bytes);
}

/// Phase 1 correctness regression — trailing partial chunk.
///
/// If total_size is not a multiple of chunk_size, the last range is a
/// short read. The assembly must handle the short trailing range
/// correctly (right length, right position). This is exactly the kind of
/// off-by-one a broken fix would introduce.
#[tokio::test(flavor = "current_thread")]
async fn range_engine_download_handles_partial_trailing_chunk() {
    let chunk_size: usize = 256 * 1024;
    let n_full_ranges: usize = 5;
    let trailing: usize = 37 * 1024; // deliberately awkward remainder
    let total_size: usize = chunk_size * n_full_ranges + trailing;
    let max_concurrent_ranges: usize = 3;

    let engine = make_engine(chunk_size, max_concurrent_ranges);
    let (bytes, stats) = engine
        .download(
            total_size as u64,
            |off, len| async move { Ok(allocate_range(off, len)) },
            None,
        )
        .await
        .expect("download failed");

    assert_eq!(bytes.len(), total_size, "total assembled length must match");
    assert_eq!(stats.ranges_processed, n_full_ranges + 1);

    // Full-length ranges: each fully filled with its own byte.
    for r in 0..n_full_ranges {
        let range_offset = r * chunk_size;
        let expected = ((range_offset as u64 >> 20) & 0xFF) as u8;
        for i in range_offset..range_offset + chunk_size {
            assert_eq!(
                bytes[i], expected,
                "byte {} in full range {} should equal {}",
                i, r, expected
            );
        }
    }
    // Trailing partial range.
    let range_offset = n_full_ranges * chunk_size;
    let expected = ((range_offset as u64 >> 20) & 0xFF) as u8;
    for i in range_offset..total_size {
        assert_eq!(
            bytes[i], expected,
            "byte {} in trailing partial range should equal {}",
            i, expected
        );
    }
}

/// Phase 1 correctness regression — closure that returns a shorter Bytes
/// than requested. The engine currently logs a warning but proceeds; the
/// assembly should still produce output whose length matches what was
/// actually returned, not what was requested. This locks in whatever
/// behavior the fix chooses so a later refactor can't silently change it.
#[tokio::test(flavor = "current_thread")]
async fn range_engine_download_short_read_does_not_panic() {
    let chunk_size: usize = 64 * 1024;
    let n_ranges: usize = 4;
    let total_size: usize = chunk_size * n_ranges;
    let short_range_idx: usize = 2;
    let short_range_bytes: u64 = (chunk_size / 2) as u64;

    let engine = make_engine(chunk_size, 2);
    let result = engine
        .download(
            total_size as u64,
            move |off, len| async move {
                // Return a short read on exactly one range.
                let actual_len = if off == (short_range_idx * chunk_size) as u64 {
                    short_range_bytes
                } else {
                    len
                };
                Ok(allocate_range(off, actual_len))
            },
            None,
        )
        .await;

    // The current code accepts the short read (warns) and returns whatever
    // was assembled; the fix should behave the same way (no panic, no UB,
    // no silent truncation-with-wrong-length). Either
    // "Ok with total_size - (chunk_size - short_range_bytes)"
    // or "Err(...)" is acceptable — this test just proves no panic.
    match result {
        Ok((bytes, _stats)) => {
            let expected_len = total_size - (chunk_size - short_range_bytes as usize);
            assert_eq!(
                bytes.len(),
                expected_len,
                "assembled length should equal the sum of returned range lengths"
            );
        }
        Err(_e) => {
            // Explicit error is also acceptable — proves no panic.
        }
    }
}
