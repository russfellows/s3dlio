// tests/test_phase2_join_all_vs_spawn.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Phase 2 site 3.1b and 3.1d RED/GREEN pattern tests for issue #148
// (docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md).
//
// Several sites in the crate build a Vec of bare async blocks and drive
// them via `futures::future::join_all(...)` (site 3.1b uses the pre-stat
// batch phase in `get_objects_parallel`; site 3.1d uses `try_join_all`
// over per-shard checkpoint reads). `join_all` polls every future
// inside the caller's single task — no `tokio::spawn` in sight. Any
// per-future CPU work (header parsing, signing, checksum compute, or
// even a simple `std::thread::sleep` in a mock backend) blocks every
// OTHER future's poll on the same worker thread until it yields.
//
// The Phase 2 fix is identical to site 3.1a: spawn each future as its
// own tokio task, so tokio can distribute polling across worker threads.
// This test proves the pattern-level RED→GREEN difference on 4 workers.
//
// Same test file style as `test_phase2_drain_first_err.rs`: two tests
// side-by-side on synthetic tokio tasks so the RED/GREEN distinction
// is cross-platform reliable and deterministic.

use futures::future::join_all;
use futures::stream::{FuturesUnordered, StreamExt};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

const N_TASKS: usize = 4;
const SYNC_SLEEP_MS: u64 = 100;

/// Build N async blocks that each `std::thread::sleep` synchronously for
/// SYNC_SLEEP_MS. Synchronous sleep holds the tokio worker thread — that's
/// how we make the "serialize on one task's poll" bug observable.
fn synthetic_stat_futures(counter: Arc<AtomicUsize>) -> Vec<impl std::future::Future<Output = ()>> {
    (0..N_TASKS)
        .map(|_| {
            let counter = Arc::clone(&counter);
            async move {
                std::thread::sleep(Duration::from_millis(SYNC_SLEEP_MS));
                counter.fetch_add(1, Ordering::Relaxed);
            }
        })
        .collect()
}

/// Phase 2 site 3.1b/3.1d RED — `join_all` polls every future inside the
/// caller's task, so synchronous work in each future serializes on one
/// worker thread. With 4 tasks × 100ms sync sleep, elapsed ≈ 400ms even
/// though we're on a `worker_threads = 4` runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_1b_join_all_serializes_sync_work() {
    let counter = Arc::new(AtomicUsize::new(0));
    let futs = synthetic_stat_futures(Arc::clone(&counter));

    let start = Instant::now();
    join_all(futs).await;
    let elapsed = start.elapsed();

    assert_eq!(
        counter.load(Ordering::Relaxed),
        N_TASKS,
        "all tasks must have run to completion"
    );
    // Expected: ~400ms serialized (4 tasks × 100ms).
    // Threshold at 350ms cleanly separates from the spawn-based ~150ms.
    assert!(
        elapsed >= Duration::from_millis(350),
        "join_all should serialize N × sync_sleep on ONE task's poll, elapsed = {:?}. \
         If this is much lower, tokio's scheduler is doing something unexpected \
         (e.g., migrating polls across threads mid-await); investigate before \
         trusting this test as a RED baseline.",
        elapsed
    );
}

/// Phase 2 site 3.1b/3.1d GREEN — spawn each future onto its own tokio
/// task via `tokio::spawn`. Now tokio distributes polls across the 4
/// worker threads and the 4 synchronous sleeps run in parallel.
/// Expected: ~100-150ms.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_1b_spawned_futures_run_in_parallel() {
    let counter = Arc::new(AtomicUsize::new(0));

    // Build the same futures shape, but wrap each in tokio::spawn.
    let mut pending: FuturesUnordered<JoinHandle<()>> = FuturesUnordered::new();
    for _ in 0..N_TASKS {
        let counter = Arc::clone(&counter);
        pending.push(tokio::spawn(async move {
            std::thread::sleep(Duration::from_millis(SYNC_SLEEP_MS));
            counter.fetch_add(1, Ordering::Relaxed);
        }));
    }

    let start = Instant::now();
    while let Some(res) = pending.next().await {
        res.expect("spawned task should not panic");
    }
    let elapsed = start.elapsed();

    assert_eq!(counter.load(Ordering::Relaxed), N_TASKS);
    // Expected: ~100-150ms parallel on 4 workers.
    // Threshold at 250ms is well below join_all's ~400ms and comfortably
    // above the ~150ms fixed code should hit.
    assert!(
        elapsed < Duration::from_millis(250),
        "spawned futures should run in parallel on 4 workers (~150ms total); \
         elapsed = {:?}. If this is close to the join_all's ~400ms, tokio's \
         scheduler is not distributing spawned tasks across workers — \
         investigate the runtime configuration.",
        elapsed
    );
}
