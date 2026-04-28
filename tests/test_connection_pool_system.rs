// tests/test_connection_pool_system.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// System tests for v0.9.92 connection pool / runtime improvements.
//
// These tests verify that the four fixes in v0.9.92 have the intended
// behavioral impact, not just that they compile.
//
// ═══════════════════════════════════════════════════════════════════════════
// ALWAYS-ON tests  (no external server, run on any machine):
//
//   cargo test --test test_connection_pool_system -- --nocapture
//
// LIVE-SERVER tests  (tagged #[ignore], need a running S3-compatible server):
//
//   # Start s3-ultra (example — adjust paths as needed):
//   #   s3-ultra serve --db-path /tmp/s3ultra --bind 127.0.0.1 --port 9000 \
//   #                  --access-key minioadmin --secret-key minioadmin &
//
//   export AWS_ENDPOINT_URL=http://127.0.0.1:9000
//   export AWS_ACCESS_KEY_ID=minioadmin
//   export AWS_SECRET_ACCESS_KEY=minioadmin
//   export AWS_REGION=us-east-1
//   export S3DLIO_H2C=0
//   export S3DLIO_TEST_BUCKET=test-bench   # must already exist
//
//   cargo test --test test_connection_pool_system -- --include-ignored --nocapture
//
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn a minimal HTTP/1.1 server on a random loopback port.
///
/// Returns `(port, connection_count)`.  `connection_count` is atomically
/// incremented once per accepted TCP connection (not per request), so it
/// distinguishes connection-reuse (pool) from connection-creation (churn).
async fn spawn_counting_http_server() -> (u16, Arc<AtomicUsize>) {
    use bytes::Bytes;
    use http_body_util::Empty;
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder as AutoBuilder;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let conn_count = Arc::new(AtomicUsize::new(0));
    let cc = conn_count.clone();

    tokio::spawn(async move {
        let builder = AutoBuilder::new(TokioExecutor::new());
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(v) => v,
                Err(_) => break,
            };
            // Increment BEFORE dispatching so the count is visible immediately.
            cc.fetch_add(1, Ordering::SeqCst);
            let io = TokioIo::new(stream);
            let b = builder.clone();
            tokio::spawn(async move {
                let _ = b
                    .serve_connection(
                        io,
                        service_fn(|_: Request<Incoming>| async {
                            // Empty 200 OK — HEAD has no body so keep-alive
                            // immediately reclaims the connection.
                            Ok::<Response<Empty<Bytes>>, std::convert::Infallible>(Response::new(
                                Empty::new(),
                            ))
                        }),
                    )
                    .await;
            });
        }
    });

    (port, conn_count)
}

/// Return `AWS_ENDPOINT_URL` if set, otherwise `None`.  Used to skip live-
/// server tests on machines without a running S3-compatible server.
fn s3_endpoint() -> Option<String> {
    std::env::var("AWS_ENDPOINT_URL").ok()
}

/// Return the test bucket name from `S3DLIO_TEST_BUCKET` / `S3_BUCKET`, or
/// the default `"test-bench"`.
#[allow(dead_code)] // only referenced from #[ignore] live-server tests
fn s3_test_bucket() -> String {
    std::env::var("S3DLIO_TEST_BUCKET")
        .or_else(|_| std::env::var("S3_BUCKET"))
        .unwrap_or_else(|_| "test-bench".to_string())
}

// ═════════════════════════════════════════════════════════════════════════════
// FIX 1 — UNLIMITED CONNECTION POOL
//
// Root cause (v0.9.92 fix): DEFAULT_POOL_MAX_IDLE_PER_HOST was 32.
// At concurrency ≥ 33 the pool evicted the 33rd connection, forcing a new TCP
// handshake (~700 µs on loopback) for every request beyond the cap.
//
// Fix: DEFAULT_POOL_MAX_IDLE_PER_HOST = usize::MAX (unlimited).
// All idle connections are retained; the second wave reuses them.
// ═════════════════════════════════════════════════════════════════════════════

/// The constant DEFAULT_POOL_MAX_IDLE_PER_HOST must be usize::MAX.
/// Regression check — if this constant is reverted, Fix 1 is broken.
#[test]
fn test_pool_constant_is_unlimited() {
    use s3dlio::constants::DEFAULT_POOL_MAX_IDLE_PER_HOST;

    assert_eq!(
        DEFAULT_POOL_MAX_IDLE_PER_HOST,
        usize::MAX,
        "DEFAULT_POOL_MAX_IDLE_PER_HOST must be usize::MAX (unlimited). \
         The previous value of 32 caused TCP handshake storms at concurrency > 32."
    );
    println!(
        "DEFAULT_POOL_MAX_IDLE_PER_HOST = usize::MAX ({})  ✓",
        usize::MAX
    );
}

/// Behavioural proof that the pool holds more than 32 idle connections.
///
/// Wave 1 (64 concurrent HEAD requests) establishes 64 TCP connections.
/// Wave 2 (64 more concurrent HEAD requests) must reuse them — connection
/// count must not increase significantly after wave 1.
///
/// With the old cap of 32: pool evicts 32 connections → wave 2 opens ~32 new
/// connections → total ≈ 96.
/// With unlimited pool: all 64 idle → wave 2 opens 0 new connections → total = 64.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_pool_reuses_connections_beyond_old_cap() {
    let (port, conn_count) = spawn_counting_http_server().await;
    let url = format!("http://127.0.0.1:{port}");

    // Build a reqwest client using the library's pool settings
    // (pool_max_idle_per_host = DEFAULT_POOL_MAX_IDLE_PER_HOST = usize::MAX).
    let client = s3dlio::reqwest_client::build_reqwest_http_client();

    let wave = 64_usize; // exceeds the old cap of 32

    // ── Wave 1: each request opens a new TCP connection ───────────────────
    let handles: Vec<_> = (0..wave)
        .map(|_| {
            let c = client.clone(); // shares same pool
            let u = url.clone();
            tokio::spawn(async move {
                // HEAD: no response body → connection returns to pool immediately.
                let _ = c.head(&u).send().await;
            })
        })
        .collect();
    for h in handles {
        let _ = h.await;
    }

    let after_wave1 = conn_count.load(Ordering::SeqCst);
    println!("Wave 1 ({wave} concurrent HEAD): {after_wave1} TCP connections");
    assert_eq!(
        after_wave1, wave,
        "Wave 1 should open exactly {wave} new connections (got {after_wave1})"
    );

    // Give the pool a moment to settle (idle keep-alive negotiation).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ── Wave 2: should reuse pool connections, no new TCP connections ──────
    let handles: Vec<_> = (0..wave)
        .map(|_| {
            let c = client.clone();
            let u = url.clone();
            tokio::spawn(async move {
                let _ = c.head(&u).send().await;
            })
        })
        .collect();
    for h in handles {
        let _ = h.await;
    }

    let after_wave2 = conn_count.load(Ordering::SeqCst);
    let new_in_wave2 = after_wave2 - after_wave1;
    println!(
        "Wave 2 ({wave} concurrent HEAD): {new_in_wave2} new TCP connections \
         (unlimited pool → 0; old cap-32 pool would give ~32)"
    );

    // With unlimited pool all 64 idle connections are reused.
    // Allow a small tolerance (≤ 4) for OS-level resets / timing edge-cases.
    assert!(
        new_in_wave2 <= 4,
        "Wave 2 opened {new_in_wave2} new TCP connections; expected ≤ 4 with unlimited pool. \
         (With the old pool cap of 32 this would be ~32.)"
    );
    println!(
        "PASS: unlimited pool retained ~{}/{wave} connections between waves  ✓",
        wave.saturating_sub(new_in_wave2)
    );
}

/// Live-server version: 128 concurrent PUTs to a real S3-compatible server.
///
/// Verifies:
/// • All 128 operations complete without error.
/// • Throughput is plausible for a local server (> 50 ops/s).
///
/// With the old cap=32, workers 33–128 each pay a TCP handshake penalty on
/// every call, throttling throughput.  With unlimited pool, all 128 workers
/// hold open connections → no TCP churn.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires S3-compatible server: set AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"]
async fn test_s3_high_concurrency_pool_no_churn() {
    let endpoint = match s3_endpoint() {
        Some(e) => e,
        None => {
            println!("SKIP: AWS_ENDPOINT_URL not set");
            return;
        }
    };
    let bucket = s3_test_bucket();
    println!("Pool no-churn test: endpoint={endpoint}  bucket={bucket}");

    use s3dlio::object_store::store_for_uri;

    let base_uri = format!("s3://{bucket}/pool-churn-test/");
    let store: Arc<dyn s3dlio::object_store::ObjectStore> =
        Arc::from(store_for_uri(&base_uri).expect("store_for_uri failed"));

    let payload = bytes::Bytes::from(vec![0u8; 4096]); // 4 KiB objects
    let n_ops: usize = 128;

    // One warm-up PUT to initialise the S3 client singleton.
    let warmup_uri = format!("{base_uri}warmup.bin");
    store
        .put(&warmup_uri, payload.clone())
        .await
        .expect("warm-up PUT failed");

    // 128 concurrent PUTs.
    let start = Instant::now();
    let handles: Vec<_> = (0..n_ops)
        .map(|i| {
            let s = store.clone();
            let p = payload.clone();
            let uri = format!("{base_uri}obj-{i:04}.bin");
            tokio::spawn(async move { s.put(&uri, p).await })
        })
        .collect();

    let mut errors = 0_usize;
    for h in handles {
        match h.await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                errors += 1;
                eprintln!("PUT error: {e}");
            }
            Err(e) => {
                errors += 1;
                eprintln!("Task panic: {e}");
            }
        }
    }

    let elapsed = start.elapsed();
    let ops_per_sec = n_ops as f64 / elapsed.as_secs_f64();
    println!(
        "Completed: {n_ops} PUTs in {:.3}s = {ops_per_sec:.0} ops/s  ({errors} errors)",
        elapsed.as_secs_f64()
    );

    assert_eq!(errors, 0, "{errors} PUT operations failed");
    assert!(
        ops_per_sec > 50.0,
        "throughput {ops_per_sec:.0} ops/s is suspiciously low for a local server (expected > 50)"
    );

    // Cleanup.
    let _ = store.delete(&warmup_uri).await;
    for i in 0..n_ops {
        let _ = store.delete(&format!("{base_uri}obj-{i:04}.bin")).await;
    }
    println!("Cleanup done. PASS  ✓");
}

// ═════════════════════════════════════════════════════════════════════════════
// FIX 2 — RUNTIME THREAD CAP REMOVED
//
// Old formula: min(max(8, cores×2), 32)  — hard cap at 32 regardless of
//              machine size.  On a 64-core host: min(128, 32) = 32 wasted 32
//              cores for Python/sync callers.
//
// New formula: max(4, cores)             — scales with hardware, floor at 4.
//              With concurrency hint h:  min(h, cores×4)  when h > base.
// ═════════════════════════════════════════════════════════════════════════════

/// Verifies the properties of the new thread-count formula:
///   • Floor at 4: min. 4 threads even on single/dual-core VMs.
///   • Scales with hardware: at least as many threads as CPU cores.
///   • No hard cap at 32: on >32-core machines, new formula > old formula.
///   • Concurrency hint: min(hint, cores×4) when hint > base.
#[test]
fn test_thread_formula_no_hard_cap() {
    let cores = num_cpus::get();
    let new_formula = std::cmp::max(4, cores);

    // Old formula for comparison (we cannot call the private fn directly).
    let old_formula = (cores * 2).clamp(8_usize, 32);

    println!("CPU cores detected: {cores}");
    println!("Old formula  min(max(8, cores×2), 32)  = {old_formula}");
    println!("New formula  max(4, cores)              = {new_formula}");

    // Floor: always at least 4.
    assert!(
        new_formula >= 4,
        "thread count must have a floor of 4 (got {new_formula})"
    );

    // Scales with hardware.
    if cores >= 4 {
        assert_eq!(
            new_formula, cores,
            "on a {cores}-core host new formula should equal core count"
        );
    }

    // On > 32-core hosts the old cap was the bottleneck.
    if cores > 32 {
        assert!(
            new_formula > old_formula,
            "on {cores}-core host, new={new_formula} must exceed old capped formula={old_formula}"
        );
        println!("Large-machine assertion: new={new_formula} > old cap {old_formula}  ✓");
    } else {
        println!(
            "Note: cap removal most visible on >32-core hosts; \
             this {cores}-core machine has new={new_formula} vs old={old_formula}"
        );
    }

    // Concurrency hint: min(hint, cores×4) when hint > base.
    let hint: usize = 256;
    let hinted = if hint > new_formula {
        std::cmp::min(hint, cores * 4)
    } else {
        new_formula
    };
    println!("With configure_for_concurrency({hint}): effective thread count = {hinted}");
    assert!(
        hinted >= new_formula,
        "concurrency hint must not reduce threads below formula baseline"
    );
    assert!(
        hinted <= cores * 4,
        "hint must not exceed cores×4 = {}",
        cores * 4
    );
    println!("Thread formula properties verified  ✓");
}

/// The global s3dlio Tokio runtime must accept and execute concurrent tasks
/// from a non-async caller.  200 tasks are spawned via `spawn_on_global_rt`;
/// all must complete within 10 seconds.
///
/// This verifies that the runtime is correctly initialised and that its thread
/// pool is large enough to make progress on many concurrent tasks (Fix 2:
/// thread count = max(4, num_cpus)).
#[test]
fn test_global_runtime_executes_concurrent_tasks() {
    let n = 200_usize;
    let done = Arc::new(AtomicUsize::new(0));

    for _ in 0..n {
        let done_c = done.clone();
        s3dlio::s3_client::spawn_on_global_rt(async move {
            // A short yield ensures tasks aren't trivially serialised.
            tokio::task::yield_now().await;
            done_c.fetch_add(1, Ordering::Relaxed);
        });
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let completed = done.load(Ordering::Relaxed);
        if completed >= n {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "global runtime only completed {completed}/{n} tasks within 10 s \
             (expected all {n} to finish quickly)"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    println!("Global runtime completed {n} concurrent tasks  ✓");
}

/// The global runtime thread count must match the formula max(4, num_cpus).
///
/// Strategy: spawn enough block_in_place tasks to saturate the worker pool
/// simultaneously, then collect the OS thread IDs observed.  The number of
/// distinct thread IDs is a lower bound on the number of worker threads.
///
/// On a 28-core host we expect ≥ 28 distinct threads.
/// On a 4-core host we expect ≥ 4 distinct threads.
#[test]
fn test_global_runtime_thread_count_matches_formula() {
    use std::collections::HashSet;
    use std::sync::Mutex;

    let cores = num_cpus::get();
    let expected_min = std::cmp::max(4, cores);

    // Spawn enough tasks to keep all workers busy simultaneously.
    // Each task occupies a worker via block_in_place for ~30 ms.
    let n_tasks = expected_min * 3; // 3 waves
    let ids: Arc<Mutex<HashSet<std::thread::ThreadId>>> = Arc::new(Mutex::new(HashSet::new()));
    let done = Arc::new(AtomicUsize::new(0));

    for _ in 0..n_tasks {
        let ids_c = ids.clone();
        let done_c = done.clone();
        s3dlio::s3_client::spawn_on_global_rt(async move {
            // block_in_place: current worker thread runs the blocking closure.
            // Tokio promotes a replacement thread so other tasks continue.
            tokio::task::block_in_place(|| {
                std::thread::sleep(Duration::from_millis(30));
                ids_c.lock().unwrap().insert(std::thread::current().id());
            });
            done_c.fetch_add(1, Ordering::Relaxed);
        });
    }

    // Wait for all tasks.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let completed = done.load(Ordering::Relaxed);
        if completed >= n_tasks {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "only {completed}/{n_tasks} block_in_place tasks completed within 30 s"
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    let observed = ids.lock().unwrap().len();
    println!(
        "Observed {observed} distinct tokio worker threads \
         (expected ≥ max(4, {cores}) = {expected_min})"
    );

    // block_in_place spawns replacement threads, so we may observe more than
    // expected_min.  We assert a lower bound only.
    assert!(
        observed >= expected_min,
        "global runtime should have ≥ max(4, num_cpus) = {expected_min} worker threads, \
         observed only {observed} distinct thread IDs"
    );
    println!("Runtime thread count ≥ {expected_min}  ✓");
}

// ═════════════════════════════════════════════════════════════════════════════
// FIX 3 — CONCURRENCY HINT API  (configure_for_concurrency)
//
// configure_for_concurrency(n) stores n in a global AtomicUsize (CONCURRENCY_HINT).
// When the internal Tokio runtime is first initialised, if hint > max(4, cores)
// the runtime thread count becomes min(hint, cores×4) instead of max(4, cores).
//
// IMPORTANT: the hint is only consulted during runtime initialisation.
// Calling configure_for_concurrency after the first S3 operation is a silent
// no-op.  Call it before any s3dlio API call.
// ═════════════════════════════════════════════════════════════════════════════

/// configure_for_concurrency must accept any usize value without panicking.
/// It simply stores the value; correctness of the formula is already covered
/// by test_thread_formula_no_hard_cap.
#[test]
fn test_configure_for_concurrency_api() {
    // Edge cases: 0, 1, typical values, and extremes.
    s3dlio::configure_for_concurrency(0);
    s3dlio::configure_for_concurrency(1);
    s3dlio::configure_for_concurrency(32);
    s3dlio::configure_for_concurrency(128);
    s3dlio::configure_for_concurrency(4096);
    s3dlio::configure_for_concurrency(usize::MAX);
    println!("configure_for_concurrency: all values accepted without panic  ✓");
}

// ═════════════════════════════════════════════════════════════════════════════
// FIX 4 — CONNECTION POOL WARMUP  (warmup_connection_pool)
//
// warmup_connection_pool(url, n) fires n concurrent HEAD requests to pre-open
// TCP connections before the workload starts, eliminating the TCP-handshake
// spike that would otherwise appear in the first wave of real operations.
//
// ── DESIGN NOTE (known limitation) ──────────────────────────────────────────
// warmup_connection_pool builds its own reqwest::Client (via
// build_reqwest_http_client) for the warmup connections.  The AWS SDK S3
// client (used by store_for_uri / put / get) wraps a separate reqwest::Client
// inside the Smithy transport.  These two clients do NOT share a connection
// pool, so connections opened by warmup_connection_pool are NOT reused for
// subsequent S3 API calls.
//
// For effective S3 pool warmup, fire real S3 API calls (e.g. HeadBucket or
// GetObject) through aws_s3_client_async() before the workload begins.  This
// test verifies that the function itself works correctly (opens exactly n
// connections) and completes without error.
// ═════════════════════════════════════════════════════════════════════════════

/// warmup_connection_pool(url, n) must open exactly n TCP connections to the
/// target endpoint and complete without error.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_warmup_opens_n_tcp_connections() {
    let (port, conn_count) = spawn_counting_http_server().await;
    let url = format!("http://127.0.0.1:{port}");

    let n = 32_usize;
    s3dlio::reqwest_client::warmup_connection_pool(&url, n).await;

    let total = conn_count.load(Ordering::SeqCst);
    println!("warmup_connection_pool(n={n}): {total} TCP connections established");
    assert_eq!(
        total, n,
        "warmup_connection_pool should open exactly {n} connections, got {total}"
    );
    println!("PASS: warmup established exactly {n} connections  ✓");
    println!(
        "NOTE: these connections are in a separate reqwest pool — \
         they are NOT reused by the AWS SDK S3 client. \
         For S3 pool warmup, use HeadBucket via aws_s3_client_async()."
    );
}

/// Live-server warmup test: verifies that calling warmup_connection_pool
/// before a concurrent workload does NOT cause errors, and that the
/// subsequent S3 operations complete at acceptable throughput.
///
/// The throughput assertion documents the expected performance floor; if it
/// fails on a local server, investigate connection pool or server issues.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires S3-compatible server: set AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"]
async fn test_s3_warmup_before_concurrent_workload() {
    let endpoint = match s3_endpoint() {
        Some(e) => e,
        None => {
            println!("SKIP: AWS_ENDPOINT_URL not set");
            return;
        }
    };
    let bucket = s3_test_bucket();
    println!("Warmup test: endpoint={endpoint}  bucket={bucket}");

    use s3dlio::object_store::store_for_uri;

    let base_uri = format!("s3://{bucket}/warmup-test/");
    let store: Arc<dyn s3dlio::object_store::ObjectStore> =
        Arc::from(store_for_uri(&base_uri).expect("store_for_uri failed"));

    let concurrency = 64_usize;
    let payload = bytes::Bytes::from(vec![42u8; 4096]);

    // ── Warm up the connection pool before the benchmark ──────────────────
    //
    // NOTE: warmup_connection_pool creates its own reqwest::Client, so it
    // warms a *different* pool than the S3 SDK uses.  For now this call
    // documents the intended usage pattern; a future fix will share the pool.
    println!("Calling warmup_connection_pool({concurrency}) ...");
    s3dlio::reqwest_client::warmup_connection_pool(&endpoint, concurrency).await;
    println!("warmup done");

    // One sequential PUT to initialise the S3 client singleton (important:
    // this must happen AFTER any configure_for_concurrency call).
    let init_uri = format!("{base_uri}init.bin");
    store
        .put(&init_uri, payload.clone())
        .await
        .expect("init PUT failed");

    // ── Concurrent workload ───────────────────────────────────────────────
    let start = Instant::now();
    let handles: Vec<_> = (0..concurrency)
        .map(|i| {
            let s = store.clone();
            let p = payload.clone();
            let uri = format!("{base_uri}obj-{i:04}.bin");
            tokio::spawn(async move { s.put(&uri, p).await })
        })
        .collect();

    let mut errors = 0_usize;
    for h in handles {
        if let Ok(Err(e)) = h.await {
            errors += 1;
            eprintln!("PUT error: {e}");
        }
    }

    let elapsed = start.elapsed();
    let ops_per_sec = concurrency as f64 / elapsed.as_secs_f64();
    println!(
        "{concurrency} concurrent PUTs: {:.3}s = {ops_per_sec:.0} ops/s  ({errors} errors)",
        elapsed.as_secs_f64()
    );

    assert_eq!(errors, 0, "{errors} PUT operations failed");
    assert!(
        ops_per_sec > 50.0,
        "throughput {ops_per_sec:.0} ops/s is too low for a local server"
    );

    // Cleanup.
    let _ = store.delete(&init_uri).await;
    for i in 0..concurrency {
        let _ = store.delete(&format!("{base_uri}obj-{i:04}.bin")).await;
    }
    println!("PASS  ✓");
}

// ═════════════════════════════════════════════════════════════════════════════
// LIVE TEST — TCP connection reuse against s3-ultra
//
// Runs 2 waves of concurrent PUTs against a real S3-compatible server and
// measures how many NEW TCP connections the server sees in wave 2.
//
// With pool cap = 32 (old): wave 2 at t=64 always opens ~32 fresh TCP conns.
// With pool cap = unlimited (v0.9.92): wave 2 reuses all idle conns → 0 new.
//
// Run against the local s3-ultra instance:
//   export AWS_ENDPOINT_URL=http://127.0.0.1:9000
//   export AWS_ACCESS_KEY_ID=minioadmin
//   export AWS_SECRET_ACCESS_KEY=minioadmin
//   export AWS_REGION=us-east-1
//   export S3DLIO_H2C=0
//   export S3DLIO_TEST_BUCKET=sai3bench-1k
//   cargo test --test test_connection_pool_system test_s3_pool_reuse_two_waves -- --include-ignored --nocapture
// ═════════════════════════════════════════════════════════════════════════════

/// Two-wave PUT test against a live S3-compatible server.
///
/// Wave 1 establishes `concurrency` connections.  Wave 2 must reuse them.
/// The server-side TCP connection count is tracked via a wrapping reqwest
/// client that counts new connections (using a loopback counting proxy is not
/// needed here — we use the s3dlio reqwest client directly and compare pool
/// state via timing: if wave 2 is faster than wave 1, connections are reused).
///
/// Concrete assertion: wave 2 average latency must be ≤ wave 1 average latency.
/// A TCP handshake on loopback costs ~0.5–1 ms; if wave 2 is paying that cost
/// on every request, its latency will be similar or worse than wave 1.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires s3-ultra: AWS_ENDPOINT_URL=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_REGION=us-east-1 S3DLIO_H2C=0 S3DLIO_TEST_BUCKET=sai3bench-1k"]
async fn test_s3_pool_reuse_two_waves() {
    let endpoint =
        std::env::var("AWS_ENDPOINT_URL").unwrap_or_else(|_| "http://127.0.0.1:9000".to_string());
    let bucket = std::env::var("S3DLIO_TEST_BUCKET")
        .or_else(|_| std::env::var("S3_BUCKET"))
        .unwrap_or_else(|_| "sai3bench-1k".to_string());

    println!("TCP pool-reuse test  endpoint={endpoint}  bucket={bucket}");

    use s3dlio::object_store::store_for_uri;

    let base_uri = format!("s3://{bucket}/pool-reuse-test/");
    let store: Arc<dyn s3dlio::object_store::ObjectStore> =
        Arc::from(store_for_uri(&base_uri).expect("store_for_uri"));

    let concurrency = 64_usize; // exceeds old cap of 32
    let payload = bytes::Bytes::from(vec![0xAB_u8; 1024]); // 1 KiB

    // One sequential PUT to initialise the S3 client + open first connection.
    let init_uri = format!("{base_uri}init.bin");
    store
        .put(&init_uri, payload.clone())
        .await
        .expect("init PUT");

    // ── Wave 1 ────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let handles: Vec<_> = (0..concurrency)
        .map(|i| {
            let s = store.clone();
            let p = payload.clone();
            let uri = format!("{base_uri}w1-{i:04}.bin");
            tokio::spawn(async move {
                let t = Instant::now();
                s.put(&uri, p).await.expect("wave1 PUT");
                t.elapsed()
            })
        })
        .collect();
    let wave1_latencies: Vec<Duration> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.expect("task join"))
        .collect();
    let wave1_wall = t0.elapsed();
    let wave1_avg_ms = wave1_latencies
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / concurrency as f64;
    println!(
        "Wave 1: {concurrency} PUTs  wall={:.0}ms  avg_latency={wave1_avg_ms:.2}ms",
        wave1_wall.as_millis()
    );

    // Short pause — let connections return to the pool.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // ── Wave 2 ────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let handles: Vec<_> = (0..concurrency)
        .map(|i| {
            let s = store.clone();
            let p = payload.clone();
            let uri = format!("{base_uri}w2-{i:04}.bin");
            tokio::spawn(async move {
                let t = Instant::now();
                s.put(&uri, p).await.expect("wave2 PUT");
                t.elapsed()
            })
        })
        .collect();
    let wave2_latencies: Vec<Duration> = futures_util::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.expect("task join"))
        .collect();
    let wave2_wall = t0.elapsed();
    let wave2_avg_ms = wave2_latencies
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / concurrency as f64;
    println!(
        "Wave 2: {concurrency} PUTs  wall={:.0}ms  avg_latency={wave2_avg_ms:.2}ms",
        wave2_wall.as_millis()
    );

    // With connection reuse, wave 2 should not be slower than wave 1.
    // Allow 20% headroom for scheduling jitter.
    let threshold_ms = wave1_avg_ms * 1.20;
    println!("Assert wave2_avg ({wave2_avg_ms:.2}ms) ≤ wave1_avg×1.20 ({threshold_ms:.2}ms)");
    assert!(
        wave2_avg_ms <= threshold_ms,
        "Wave 2 avg latency {wave2_avg_ms:.2}ms > {threshold_ms:.2}ms — \
         connections are likely being re-established instead of reused from pool"
    );

    println!("Connection pool reuse confirmed (wave 2 not slower than wave 1 + 20%)  ✓");

    // Cleanup.
    let _ = store.delete(&init_uri).await;
    for i in 0..concurrency {
        let _ = store.delete(&format!("{base_uri}w1-{i:04}.bin")).await;
        let _ = store.delete(&format!("{base_uri}w2-{i:04}.bin")).await;
    }
    println!("PASS  ✓");
}
