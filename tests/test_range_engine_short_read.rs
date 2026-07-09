// tests/test_range_engine_short_read.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #152 sub-bug 2.1
// (audit finding f39, tracked as russfellows/s3dlio#152).
//
// Bug: RangeEngine::download_with_ranges assembled chunks with a running
// write-cursor (`write_offset += bytes.len()`) instead of writing each
// chunk at its declared range offset. A non-final short (or over-length)
// read from the underlying `get_range` closure only produced a
// `tracing::warn!` and was NOT treated as an error, so every subsequent
// chunk was written at the wrong offset in the master buffer and the
// tail was silently truncated. The caller received `Ok(Bytes)` with
// corrupt content and no indication anything was wrong.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §3, bug A1): any short read, over-read, or object-size drift is a
// hard `Err` — there is no "tolerate and reassemble" path, because a
// running-cursor design cannot know where the missing/extra bytes
// belong once one chunk doesn't match its declared length.

use anyhow::Result;
use bytes::Bytes;
use s3dlio::range_engine_generic::{RangeEngine, RangeEngineConfig};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn test_engine(chunk_size: usize) -> RangeEngine {
    RangeEngine::new(RangeEngineConfig {
        chunk_size,
        max_concurrent_ranges: 4,
        min_split_size: 1, // force the concurrent-range path for any size we test
        range_timeout: Duration::from_secs(5),
    })
}

/// Range index 1 of 4 returns 512 KiB instead of the requested 1 MiB;
/// ranges 0, 2, 3 return full-size chunks. Must error, not silently
/// reassemble with shifted offsets.
#[tokio::test]
async fn short_read_on_non_final_range_is_an_error() -> Result<()> {
    let engine = test_engine(1024 * 1024); // 1 MiB chunks -> 4 ranges for a 4 MiB object
    let object_size = 4 * 1024 * 1024u64;
    let data = vec![0xABu8; object_size as usize];

    let get_range = {
        let data = data.clone();
        move |offset: u64, length: u64| {
            let data = data.clone();
            async move {
                // Range index 1 (offset == 1 MiB) returns half of what
                // was requested; every other range is honest.
                let actual_len = if offset == 1024 * 1024 {
                    (length / 2) as usize
                } else {
                    length as usize
                };
                Ok::<Bytes, anyhow::Error>(Bytes::from(
                    data[offset as usize..offset as usize + actual_len].to_vec(),
                ))
            }
        }
    };

    let result = engine.download(object_size, get_range, None).await;

    let err = result.expect_err(
        "short read on a non-final range must be Err — silent reassembly with a shifted \
         write-cursor is the exact silent-data-corruption bug this test guards against",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("short") || msg.contains("expected") || msg.contains("bytes"),
        "error message should name the short-read condition, got: {msg}"
    );

    Ok(())
}

/// A non-final chunk returns MORE bytes than requested, and the final
/// chunk under-reads by the same amount, so the *total* bytes returned
/// still equals `object_size` exactly. This deliberately avoids tripping
/// the master buffer's incidental "would write past the end" overflow
/// check (which only catches over-reads large enough to blow the total
/// budget) — it isolates the actual silent-corruption bug: a compensating
/// short/over pair that sums correctly but shifts every byte in between
/// to the wrong offset, with no size-level signal that anything is wrong.
#[tokio::test]
async fn over_read_on_a_middle_range_is_an_error_even_if_totals_balance() -> Result<()> {
    let engine = test_engine(1024 * 1024);
    let object_size = 3 * 1024 * 1024u64; // 3 ranges of 1 MiB each
    let over_by = 4096usize;
    // Pad the backing buffer so the over-read slice itself doesn't
    // index out of bounds — we're testing the engine's own validation,
    // not Rust's slice bounds checks.
    let data = vec![0xEFu8; object_size as usize + over_by];

    let get_range = {
        let data = data.clone();
        move |offset: u64, length: u64| {
            let data = data.clone();
            async move {
                let actual_len = if offset == 0 {
                    length as usize + over_by // range 0 over-reads
                } else if offset == 2 * 1024 * 1024 {
                    length as usize - over_by // final range under-reads to compensate
                } else {
                    length as usize // middle range is honest
                };
                Ok::<Bytes, anyhow::Error>(Bytes::from(
                    data[offset as usize..offset as usize + actual_len].to_vec(),
                ))
            }
        }
    };

    let result = engine.download(object_size, get_range, None).await;

    result.expect_err(
        "over-read on a non-final range must be Err even when a compensating short \
         final range makes the total byte count come out correct — the sum matching \
         object_size is not proof the data landed at the right offsets",
    );

    Ok(())
}

/// Sanity check: a well-behaved get_range (every chunk exactly the
/// requested length) still round-trips correctly after the fix. Proves
/// the fix didn't overcorrect into rejecting valid transfers.
#[tokio::test]
async fn exact_length_chunks_still_succeed() -> Result<()> {
    let engine = test_engine(1024 * 1024);
    let object_size = 4 * 1024 * 1024u64;
    let data: Vec<u8> = (0..object_size).map(|i| (i % 256) as u8).collect();

    let call_count = Arc::new(AtomicUsize::new(0));
    let get_range = {
        let data = data.clone();
        let call_count = Arc::clone(&call_count);
        move |offset: u64, length: u64| {
            let data = data.clone();
            let call_count = Arc::clone(&call_count);
            async move {
                call_count.fetch_add(1, Ordering::SeqCst);
                Ok::<Bytes, anyhow::Error>(Bytes::from(
                    data[offset as usize..(offset + length) as usize].to_vec(),
                ))
            }
        }
    };

    let (bytes, stats) = engine.download(object_size, get_range, None).await?;

    assert_eq!(bytes.len(), object_size as usize);
    assert_eq!(stats.ranges_processed, 4);
    assert_eq!(call_count.load(Ordering::SeqCst), 4);
    for (i, &b) in bytes.iter().enumerate() {
        assert_eq!(b, (i % 256) as u8, "data mismatch at byte {i}");
    }

    Ok(())
}
