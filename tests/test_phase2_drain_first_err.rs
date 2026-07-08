// tests/test_phase2_drain_first_err.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Phase 2 site 3.2 RED/GREEN tests for issue #148
// (docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md, finding 3.2).
//
// The bug: `get_objects_parallel`, `get_objects_parallel_with_progress`,
// `put_objects_parallel_with_progress`, and the multipart coordinator's
// part-join loop all wrote something like:
//
//     while let Some(res) = futs.next().await {
//         out.push(res??);           // <-- short-circuit
//     }
//
// where `futs` is a `FuturesUnordered<JoinHandle<Result<T>>>` (or a
// `Vec<JoinHandle<...>>` iterated with `for`). On the first error, `res??`
// (or `handle.await??`) returns via `?` — the enclosing async block ends,
// which drops the FuturesUnordered / Vec, which drops the remaining
// JoinHandles. Dropping a JoinHandle **detaches** the task rather than
// aborting it: the in-flight tasks keep running in the background on the
// runtime, silently consuming resources long after the function has
// already returned an error to its caller. Audit §3.2 documents this.
//
// The fix is the "full-drain" pattern already present in
// `src/object_store.rs::generic_upload_files` /
// `generic_download_objects`:
//
//     let mut first_err: Option<anyhow::Error> = None;
//     while let Some(res) = futs.next().await {
//         match res {
//             Ok(Ok(item))    => out.push(item),
//             Ok(Err(e))      => { first_err.get_or_insert(e); }
//             Err(join_err)   => { first_err.get_or_insert(
//                 anyhow::anyhow!("task panicked: {}", join_err)); }
//         }
//     }
//     if let Some(e) = first_err { return Err(e); }
//
// The loop drains every spawned task to completion before returning, so
// no JoinHandle is ever dropped mid-flight.
//
// These tests exercise the two patterns side-by-side on identical
// synthetic tokio::spawn'd tasks so the RED/GREEN distinction is clean
// and cross-platform-reliable. Both tests run on a fixed multi-thread
// runtime with 4 workers.

use futures::stream::{FuturesUnordered, StreamExt};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

/// Build a canonical test scenario:
///   * task 0 succeeds after 100ms (increments `counter`)
///   * task 1 fails IMMEDIATELY
///   * task 2 succeeds after 200ms (increments `counter`)
///   * task 3 succeeds after 300ms (increments `counter`)
///
/// The 300ms slowest-succeeding task is what distinguishes drain (waits)
/// from short-circuit (returns immediately on task 1's failure). Counter
/// == 3 iff every succeeding task ran to completion (and thus incremented
/// the counter before its Ok was recorded).
fn build_scenario(
    counter: Arc<AtomicUsize>,
) -> FuturesUnordered<JoinHandle<anyhow::Result<usize>>> {
    let futs: FuturesUnordered<JoinHandle<anyhow::Result<usize>>> = FuturesUnordered::new();

    // Task 0: 100ms then Ok.
    {
        let counter = Arc::clone(&counter);
        futs.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(0usize)
        }));
    }

    // Task 1: immediate Err.
    futs.push(tokio::spawn(async move {
        Err(anyhow::anyhow!(
            "synthetic failure — the 3.2 short-circuit trigger"
        ))
    }));

    // Task 2: 200ms then Ok.
    {
        let counter = Arc::clone(&counter);
        futs.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(2usize)
        }));
    }

    // Task 3: 300ms then Ok.
    {
        let counter = Arc::clone(&counter);
        futs.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(3usize)
        }));
    }

    futs
}

/// Phase 2 site 3.2 GREEN — the drain-first-then-return-error pattern
/// awaits every spawned task before returning, so no JoinHandle is
/// dropped mid-flight and no task is detached.
///
/// After the async block returns:
///   * The FIRST error is surfaced to the caller (preserves API).
///   * Every successful task has completed (counter == 3).
///   * Wall time reflects waiting for the longest task (~300ms).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_2_drain_pattern_awaits_all_before_returning() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut futs = build_scenario(Arc::clone(&counter));

    let start = Instant::now();

    // This IS the fix pattern. All four buggy sites now use this shape.
    let mut out: Vec<usize> = Vec::new();
    let mut first_err: Option<anyhow::Error> = None;
    while let Some(res) = futs.next().await {
        match res {
            Ok(Ok(item)) => out.push(item),
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            Err(join_err) => {
                first_err.get_or_insert(anyhow::anyhow!("task panicked: {}", join_err));
            }
        }
    }
    let result: anyhow::Result<Vec<usize>> = match first_err {
        Some(e) => Err(e),
        None => Ok(out),
    };

    let elapsed = start.elapsed();

    assert!(
        result.is_err(),
        "expected Err from task 1's synthetic failure"
    );
    assert_eq!(
        counter.load(Ordering::Relaxed),
        3,
        "drain must have awaited every succeeding task (counter should equal 3, meaning \
         all three tasks that returned Ok also ran their body to completion)"
    );
    // Task 3 sleeps for 300ms; drain must wait for it too. Use 250ms as a
    // slightly relaxed floor to absorb scheduling jitter on slow CI.
    assert!(
        elapsed >= Duration::from_millis(250),
        "drain should have waited for the slowest task (300ms); actual elapsed = {:?}",
        elapsed
    );
}

/// Phase 2 site 3.2 RED — the current (buggy) short-circuit pattern
/// returns on first error, dropping the remaining JoinHandles. The
/// dropped handles detach their tasks (do NOT abort), so those tasks
/// continue running in the background — which is the leak the audit
/// warns about.
///
/// Observations at RETURN time:
///   * Result is Err (task 1's failure).
///   * counter is 0 (or maybe 1 by pure luck) — MUCH less than 3.
///   * Wall time is very small (task 1's Err latency), not 300ms.
///
/// This test documents the buggy behavior so it's clear WHY the drain
/// pattern above is necessary. Run against pre-fix `s3_utils.rs`
/// (before this commit) and equivalent bugs manifest exactly this way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_2_short_circuit_pattern_leaks_tasks() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut futs = build_scenario(Arc::clone(&counter));

    let start = Instant::now();

    // OLD (buggy) short-circuit pattern — the exact shape at
    // s3_utils.rs:1587, :1631, :1929 before this commit.
    let result: anyhow::Result<Vec<usize>> = async {
        let mut out: Vec<usize> = Vec::new();
        while let Some(res) = futs.next().await {
            out.push(res??); // <-- returns from THIS closure on first Err
        }
        Ok(out)
    }
    .await;

    let elapsed = start.elapsed();

    assert!(
        result.is_err(),
        "expected Err from task 1's synthetic failure"
    );
    assert!(
        elapsed < Duration::from_millis(50),
        "short-circuit should have returned as soon as task 1 failed; \
         elapsed = {:?}. If this is longer than expected the scheduling is \
         unusually slow and the short-circuit vs drain distinction may be \
         hard to prove.",
        elapsed
    );
    let observed = counter.load(Ordering::Relaxed);
    assert!(
        observed < 3,
        "short-circuit returned before every succeeding task had completed. \
         counter = {} (should be < 3). If this equals 3 the scheduler happened \
         to run every task before the .await'd Err surfaced — increase the \
         sleeps to make the test more robust.",
        observed
    );
}
