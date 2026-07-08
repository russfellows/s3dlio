// tests/test_phase2_loader_parallelism.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Phase 2 site 3.1a RED/GREEN tests for issue #148
// (docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md, findings 1.1 and 3.1a).
//
// The current `AsyncPoolDataLoader::run_async_pool_worker` drives a
// `FuturesUnordered<Pin<Box<...>>>` of bare, unspawned request futures
// entirely from within one tokio task. Tokio can only poll that task on
// one worker thread at a time, so:
//
//   * request-driving work (header parsing, decompression, checksum
//     compute) that consumes CPU inside a fetch's poll blocks every
//     other in-flight fetch until it yields;
//   * an external CancellationToken can only be checked between
//     completions — an in-flight `.await` on a network read cannot be
//     interrupted;
//   * a panic inside a fetch dies inside the single worker task and
//     silently truncates the iterator (Python side sees StopIteration,
//     Rust side sees `Stream::next()` return `None`).
//
// The Phase 2 fix:
//   * `tokio::spawn` each fetch as its own task, so tokio can
//     distribute polling across worker threads (true task-level
//     parallelism, not just async concurrency inside one task);
//   * pass a `CancellationToken` into each spawned task and wrap the
//     fetch in `select! { fetch, token.cancelled() }` — cancellation
//     drops the fetch future immediately, cancelling the in-flight
//     I/O for free;
//   * observe JoinError from panicked tasks and surface them as
//     `DatasetError::Backend` so callers see the error instead of
//     silent truncation.
//
// These tests use a small mock `ObjectStore` that lets each test
// dictate what `get()` does — sleep synchronously (CPU-holding), sleep
// asynchronously (interruptible), or panic. All four tests are pinned
// to a `multi_thread` runtime with 4 workers so the RED vs GREEN
// distinction is deterministic across environments (any host that can
// run tokio, including single-core WSL, spins up multiple worker
// threads on request).

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use s3dlio::data_loader::async_pool_dataloader::{
    AsyncPoolDataLoader, MultiBackendDataset, PoolConfig,
};
use s3dlio::data_loader::LoaderOptions;
use s3dlio::object_store::{ObjectMetadata, ObjectStore, ObjectWriter};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

// ─────────────────────────────────────────────────────────────────────
// Mock ObjectStore — only `get()` is expected to be called by the
// loader; everything else is `unimplemented!()` and will loudly
// surface any accidental use.
// ─────────────────────────────────────────────────────────────────────

/// Configurable behavior for a single MockStore's `get()` responses.
#[derive(Default, Clone)]
struct GetControl {
    /// `std::thread::sleep` this long inside the get() poll before
    /// returning. Synchronous — holds the current tokio worker thread.
    /// Used to prove parallelism: if two calls are truly distributed
    /// across workers, their sync sleeps overlap; if they're serialized
    /// on one worker, they add up.
    sync_sleep: Duration,
    /// `tokio::time::sleep` this long. Properly async — yields the
    /// worker to other tasks. Used to test cancellation interruption.
    async_sleep: Duration,
    /// If `Some(n)`, the nth (0-indexed) call to `get()` panics before
    /// returning. Used to test panic propagation.
    panic_on_call_nth: Option<usize>,
}

struct MockStore {
    control: GetControl,
    /// Incremented at the top of every get() call, regardless of outcome.
    started: AtomicUsize,
    /// Incremented right before get() returns Ok. NOT incremented on
    /// panic or on cancellation-drop of the future.
    completed: AtomicUsize,
}

impl MockStore {
    fn new(control: GetControl) -> Self {
        Self {
            control,
            started: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
        }
    }

    fn completed_count(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }

    fn started_count(&self) -> usize {
        self.started.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ObjectStore for MockStore {
    async fn get(&self, _uri: &str) -> Result<Bytes> {
        let n = self.started.fetch_add(1, Ordering::Relaxed);

        if Some(n) == self.control.panic_on_call_nth {
            panic!("MockStore synthetic panic on call {}", n);
        }

        if !self.control.sync_sleep.is_zero() {
            std::thread::sleep(self.control.sync_sleep);
        }
        if !self.control.async_sleep.is_zero() {
            tokio::time::sleep(self.control.async_sleep).await;
        }

        self.completed.fetch_add(1, Ordering::Relaxed);
        Ok(Bytes::from(vec![0u8; 1024]))
    }

    async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
        unimplemented!("MockStore::get_range should not be called by AsyncPoolDataLoader")
    }
    async fn put(&self, _uri: &str, _data: Bytes) -> Result<()> {
        unimplemented!("MockStore::put should not be called by AsyncPoolDataLoader")
    }
    async fn put_multipart(
        &self,
        _uri: &str,
        _data: Bytes,
        _part_size: Option<usize>,
    ) -> Result<()> {
        unimplemented!("MockStore::put_multipart should not be called by AsyncPoolDataLoader")
    }
    async fn list(&self, _uri_prefix: &str, _recursive: bool) -> Result<Vec<String>> {
        unimplemented!("MockStore::list should not be called by AsyncPoolDataLoader")
    }
    fn list_stream<'a>(
        &'a self,
        _uri_prefix: &'a str,
        _recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        unimplemented!("MockStore::list_stream should not be called by AsyncPoolDataLoader")
    }
    async fn stat(&self, _uri: &str) -> Result<ObjectMetadata> {
        unimplemented!("MockStore::stat should not be called by AsyncPoolDataLoader")
    }
    async fn delete(&self, _uri: &str) -> Result<()> {
        unimplemented!("MockStore::delete should not be called by AsyncPoolDataLoader")
    }
    async fn delete_batch(&self, _uris: &[String]) -> Result<()> {
        unimplemented!("MockStore::delete_batch should not be called by AsyncPoolDataLoader")
    }
    async fn delete_prefix(&self, _uri_prefix: &str) -> Result<()> {
        unimplemented!("MockStore::delete_prefix should not be called by AsyncPoolDataLoader")
    }
    async fn create_container(&self, _name: &str) -> Result<()> {
        unimplemented!("MockStore::create_container should not be called by AsyncPoolDataLoader")
    }
    async fn delete_container(&self, _name: &str) -> Result<()> {
        unimplemented!("MockStore::delete_container should not be called by AsyncPoolDataLoader")
    }
    async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
        unimplemented!("MockStore::get_writer should not be called by AsyncPoolDataLoader")
    }
}

/// Build a `MultiBackendDataset` with `n_items` URIs backed by the
/// given mock store. The URIs are `mock://item{i}` — they never touch
/// a real backend because the mock's get() ignores its input.
fn make_dataset(store: Arc<MockStore>, n_items: usize) -> MultiBackendDataset {
    let store_dyn: Arc<dyn ObjectStore> = store;
    MultiBackendDataset {
        uris: (0..n_items).map(|i| format!("mock://item{}", i)).collect(),
        store: store_dyn,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 — parallelism (RED→GREEN)
//
// Current code: FuturesUnordered polled inside one tokio task; each
// fetch's synchronous CPU work (here, `std::thread::sleep`) blocks the
// current worker until it yields. 4 fetches × 100ms sync sleep should
// therefore serialize into ~400ms of wall time.
//
// Fixed code: each fetch is `tokio::spawn`'d onto its own task; with
// `worker_threads = 4` in the runtime, the four sleeps run in parallel
// on four workers → ~100-150ms of wall time.
//
// Threshold at 250ms cleanly separates the two behaviors.
// ─────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_1a_spawned_fetches_run_in_parallel_across_workers() {
    let control = GetControl {
        sync_sleep: Duration::from_millis(100),
        ..GetControl::default()
    };
    let store = Arc::new(MockStore::new(control));
    let dataset = make_dataset(store.clone(), 4);
    let opts = LoaderOptions {
        batch_size: 4,
        ..LoaderOptions::default()
    };
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let pool = PoolConfig {
        pool_size: 4,
        readahead_batches: 1,
        batch_timeout: Duration::from_secs(30),
        max_inflight: 4,
    };
    let mut stream = loader.stream_with_pool(pool);

    let start = Instant::now();
    let mut count = 0usize;
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("mock store never returns Err");
        count += batch.len();
    }
    let elapsed = start.elapsed();

    assert_eq!(count, 4, "should deliver all 4 items");
    assert_eq!(store.completed_count(), 4);

    assert!(
        elapsed < Duration::from_millis(250),
        "expected 4 parallel fetches (100ms sync sleep each) to complete in ~100-150ms \
         with worker_threads=4, but wall time was {:?}. In the CURRENT code the fetches \
         are polled inside a single task, so their sync sleeps serialize into ~400ms — \
         this is the RED gate for Phase 2's spawn conversion.",
        elapsed
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 — external cancellation interrupts in-flight fetches (RED→GREEN)
//
// Current code: cancel_token.is_cancelled() is only checked between
// completions of pending_requests.next().await, and next() only
// returns when at least one bare future finishes. If every in-flight
// future is awaiting a long async sleep, cancel takes effect only
// after at least one sleep finishes.
//
// Fixed code: each spawned task's `select!` polls the cancellation
// side alongside the fetch — cancel fires, select! drops the fetch,
// task returns, JoinHandle resolves. The whole loader shuts down
// within a small multiple of the cancel signal's own latency.
// ─────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_1a_external_cancel_interrupts_in_flight_fetches() {
    let control = GetControl {
        async_sleep: Duration::from_secs(2),
        ..GetControl::default()
    };
    let store = Arc::new(MockStore::new(control));
    let dataset = make_dataset(store, 10);

    let cancel = CancellationToken::new();
    let opts = LoaderOptions {
        batch_size: 4,
        cancellation_token: Some(cancel.clone()),
        ..LoaderOptions::default()
    };
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let pool = PoolConfig {
        pool_size: 4,
        readahead_batches: 1,
        batch_timeout: Duration::from_secs(30),
        max_inflight: 4,
    };
    let mut stream = loader.stream_with_pool(pool);

    // Fire the cancel after 100ms; at that moment all 4 in-flight fetches
    // are ~1.9s into their async_sleep(2s).
    let cancel_trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel_trigger.cancel();
    });

    let start = Instant::now();
    // Drain whatever the stream produces. We don't care about the
    // number of items — only how quickly the stream closes.
    while let Some(_batch) = stream.next().await {}
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(500),
        "external CancellationToken should interrupt in-flight fetches within 500ms of \
         cancel. Elapsed = {:?}. In the CURRENT code the cancel is only checked between \
         `pending_requests.next().await` completions, so the loader waits ~2s for the \
         first in-flight fetch to finish before honoring the cancel — this is the RED \
         gate for wiring cancel into each spawned task via select!.",
        elapsed
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 — panic inside a fetch surfaces as an error (RED→GREEN)
//
// Current code: a bare future's panic propagates up run_async_pool_worker's
// own poll. Tokio's per-task panic boundary catches it, the whole worker
// task dies silently, `tx` is dropped, the receiver sees the channel
// close, and the caller observes `Stream::next() == None` — silent
// truncation.
//
// Fixed code: each fetch is spawned. A panic in the spawned task
// resolves its `JoinHandle` with `Err(JoinError)` where `is_panic()`
// is true. `run_async_pool_worker` sees the JoinError and forwards
// `Err(DatasetError::Backend(...))` on `tx`. Caller sees the error.
// ─────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_1a_panic_in_fetch_surfaces_as_error_not_truncation() {
    let control = GetControl {
        panic_on_call_nth: Some(1), // the second call panics
        ..GetControl::default()
    };
    let store = Arc::new(MockStore::new(control));
    let dataset = make_dataset(store, 10);
    let opts = LoaderOptions {
        batch_size: 4,
        ..LoaderOptions::default()
    };
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let pool = PoolConfig {
        pool_size: 4,
        readahead_batches: 1,
        batch_timeout: Duration::from_secs(30),
        max_inflight: 4,
    };
    let mut stream = loader.stream_with_pool(pool);

    let mut saw_error = false;
    let mut ok_batches = 0usize;
    // Bound the loop so a broken fix can't hang the test forever.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(500), stream.next()).await {
            Ok(Some(Ok(_batch))) => ok_batches += 1,
            Ok(Some(Err(_e))) => {
                saw_error = true;
                break;
            }
            Ok(None) => break, // stream ended
            Err(_) => break,   // 500ms without progress — treat as ended
        }
    }

    assert!(
        saw_error,
        "expected a panicking fetch to surface as Err(DatasetError::...) to the caller. \
         Got {} Ok batches and no Err. In the CURRENT code the panic dies inside the \
         single run_async_pool_worker task, tx is dropped, and the caller sees the \
         stream simply end — silent truncation. This is the RED gate for wrapping \
         each fetch in tokio::spawn and converting JoinError-with-is_panic into a \
         proper DatasetError::Backend on the channel.",
        ok_batches
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 — dropping the receiver stops in-flight fetches
//
// This is a MUST-STAY-GREEN regression guard, not a RED→GREEN test.
// The CURRENT code passes it because bare futures in FuturesUnordered
// cancel when the enclosing task drops. A NAIVE spawn conversion
// (spawning without wiring cancellation) would FAIL this test — that's
// the audit §2.1 warned-about regression. Its purpose is to catch that
// regression if the fix is written carelessly.
// ─────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn phase2_3_1a_dropping_receiver_stops_in_flight_fetches() {
    let control = GetControl {
        async_sleep: Duration::from_millis(500),
        ..GetControl::default()
    };
    let store = Arc::new(MockStore::new(control));
    let dataset = make_dataset(store.clone(), 100);
    let opts = LoaderOptions {
        batch_size: 4,
        ..LoaderOptions::default()
    };
    let loader = AsyncPoolDataLoader::new(dataset, opts);
    let pool = PoolConfig {
        pool_size: 8,
        readahead_batches: 1,
        batch_timeout: Duration::from_secs(30),
        max_inflight: 8,
    };
    let mut stream = loader.stream_with_pool(pool);

    // Wait long enough for the pool to fill and a first batch to land.
    let _first = tokio::time::timeout(Duration::from_secs(3), stream.next())
        .await
        .expect("first batch should arrive within 3s")
        .expect("first batch should be Some(_)")
        .expect("first batch should be Ok");

    // Give the worker a chance to refill the pool with the next wave of
    // fetches before we drop — otherwise the observation window is so
    // narrow that started can equal completed by pure luck (worker not
    // yet resumed past the send that just delivered our batch).
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Snapshot state just before dropping — we'll use this to prove the
    // pool was actively working (fetches in flight) at drop time.
    let started_at_drop = store.started_count();
    let completed_at_drop = store.completed_count();

    // Now drop the stream and wait long enough that a leaked pool
    // would keep running to completion.
    drop(stream);
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let completed = store.completed_count();

    // Stronger check: at the moment of drop, fetches had been started
    // that hadn't yet completed — the pool was actively working. If
    // this assertion is vacuously false (started == completed), the
    // outer cancellation check tells us nothing.
    assert!(
        started_at_drop > completed_at_drop,
        "expected the pool to have in-flight fetches at drop time \
         (started_at_drop={}, completed_at_drop={}). If they're equal the \
         pool went idle before we dropped the stream — this test wouldn't \
         actually be exercising the cancellation path.",
        started_at_drop,
        completed_at_drop
    );
    // And the final state should not have blown past the pool's snapshot:
    // if in-flight fetches are properly cancelled, `completed` should
    // stay near `completed_at_drop`. If they leak, `completed` climbs
    // toward `started_at_drop` (or beyond, if the leaked pool refilled).
    assert!(
        completed < 30,
        "expected in-flight fetches to be cancelled after the receiver is dropped, \
         but {} of 100 items completed (was {} at drop time). A naive tokio::spawn \
         conversion without cancellation wiring would trip this — see audit §2.1.",
        completed,
        completed_at_drop
    );
}
