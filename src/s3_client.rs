// src/s3_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Owns a single global multi-thread Tokio runtime and the global S3 client.
//!

use anyhow::{bail, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_config::retry::RetryConfig;
use aws_config::timeout::TimeoutConfig;
use aws_sdk_s3::{config::Region, Client};

use aws_smithy_runtime_api::client::http::SharedHttpClient;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::{env, thread, time::Duration};
use tokio::runtime::{Builder as TokioBuilder, Handle};
use tokio::sync::{oneshot, OnceCell};
use tracing::{debug, info}; // For logging

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_REGION: &str = "us-east-1";

// -----------------------------------------------------------------------------
// Global runtime + S3 client (lazy, thread-safe)
// -----------------------------------------------------------------------------
static RT_HANDLE: once_cell::sync::OnceCell<Handle> = once_cell::sync::OnceCell::new();
static CLIENT: OnceCell<Client> = OnceCell::const_new();

/// Concurrency hint set by [`configure_for_concurrency`].
/// Zero means "not set" — `get_runtime_threads` uses the CPU-based default.
static CONCURRENCY_HINT: AtomicUsize = AtomicUsize::new(0);

/// Hard upper-bound on global-rt worker threads set by [`set_rt_threads_limit`].
/// Zero means "not set" — no cap beyond the CPU-based default.
pub(crate) static RT_THREADS_LIMIT: AtomicUsize = AtomicUsize::new(0);

/// Inform s3dlio of your expected peak concurrency level **before** the first
/// S3 operation.
///
/// This hint is consulted exactly once when the global Tokio runtime is
/// first initialized, so it **must** be called before any S3 I/O.
///
/// Effect: ensures the internal Tokio runtime has at least `n` worker threads
/// so that sync/Python callers (which dispatch through the s3dlio runtime)
/// can overlap `n` concurrent requests without thread starvation.
///
/// For async callers (e.g. sai3-bench or any `tokio::main` program) that bring
/// their own runtime, this hint does not affect their thread pool — it only
/// influences the s3dlio-internal runtime used by blocking/Python callers.
///
/// The connection pool is already unlimited by default
/// (see [`crate::constants::DEFAULT_POOL_MAX_IDLE_PER_HOST`]), so no pool
/// configuration is needed for throughput tuning.
///
/// # Example
/// ```no_run
/// s3dlio::configure_for_concurrency(128);
/// // … now call blocking S3 helpers from 128 threads …
/// ```
pub fn configure_for_concurrency(n: usize) {
    CONCURRENCY_HINT.store(n, Ordering::Relaxed);
}

/// Auto-configure every process-wide thread pool s3dlio owns: the global
/// s3dlio Tokio runtime, the `pyo3_async_runtimes` Tokio runtime (Python
/// extension builds only), and Rayon's global pool.
///
/// `threads = 0` means "auto-detect from MPI/distributed-training env vars"
/// (see [`crate::constants::mpi_aware_thread_budget`]); a positive value is
/// an explicit per-process worker-thread count.
///
/// This is the single choke point for the whole class of CPU-oversubscription
/// bugs found while chasing the DLIO unet3d npz-datagen slowdown: previously
/// only 2 of 16 s3dlio-touching call sites in DLIO_local_changes called
/// `configure_tokio_threads()`, and Rayon (used by NPZ/data-gen's
/// `par_chunks_mut` hot paths) had no MPI-awareness or explicit sizing
/// anywhere in s3dlio at all. Calling this unconditionally once at
/// `s3dlio` import time (see `_pymod` in `lib.rs`) means every current and
/// future call site — any data format, reading or writing, Python or
/// pure-Rust — gets a correctly-sized pool without needing to opt in.
///
/// Idempotent / safe to call more than once: the global s3dlio Tokio runtime
/// (`RT_HANDLE`) and Rayon's global pool are each backed by a `OnceLock`-style
/// cell that only takes effect on first build, and `pyo3_async_runtimes::
/// tokio::init()` merely stores the pending builder in a mutex — it does not
/// build the runtime until first use — so a later explicit
/// `configure_tokio_threads(n)` call from Python still fully overrides an
/// earlier automatic call, as long as it happens before the first actual S3
/// operation / `create_async_loader` call, exactly as already documented.
pub fn configure_thread_pools(threads: usize) {
    let threads = if threads > 0 {
        threads
    } else {
        crate::constants::mpi_aware_thread_budget()
    };

    // Global s3dlio Tokio runtime (blocking/sync + Python callers).
    RT_THREADS_LIMIT.store(threads, Ordering::Relaxed);

    // pyo3-async-runtimes Tokio runtime (create_async_loader / Arrow decode
    // streaming pipeline) -- only present in Python extension builds.
    #[cfg(feature = "extension-module")]
    {
        let mut builder = TokioBuilder::new_multi_thread();
        builder.worker_threads(threads);
        builder.enable_all();
        builder.thread_name("s3dlio-decode");
        pyo3_async_runtimes::tokio::init(builder);
    }

    // Rayon global pool (NPZ/data-gen par_chunks_mut hot paths). Ignore the
    // Err case: it just means the pool was already built (e.g. by a prior
    // call to this function, or by Rayon's own lazy default racing ahead of
    // us) -- there is no way to resize an already-built global pool, and
    // that's fine, since whichever call won still sized it correctly.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("s3dlio-rayon-{i}"))
        .build_global();
}

/// Returns `true` when `S3DLIO_UNSIGNED_PAYLOAD=1` (or `true`/`yes`/`on`/`enable`).
///
/// When enabled, S3 PUT requests send `x-amz-content-sha256: UNSIGNED-PAYLOAD`
/// instead of computing a SHA-256 digest of the request body, eliminating
/// per-request CPU cost proportional to object size.
///
/// The result is cached in a `OnceLock` — the env var is read exactly once on
/// first call, so there is zero env-var lookup overhead on the hot path.
///
/// Only use on trusted, internal endpoints (MinIO, s3-ultra, Ceph RGW).
/// See [`crate::constants::ENV_UNSIGNED_PAYLOAD`] for full documentation.
pub fn unsigned_payload_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var(crate::constants::ENV_UNSIGNED_PAYLOAD)
            .map(|v| {
                matches!(
                    v.to_lowercase().as_str(),
                    "1" | "true" | "yes" | "on" | "enable"
                )
            })
            .unwrap_or(crate::constants::DEFAULT_UNSIGNED_PAYLOAD)
    })
}

// Create (once) a background multi-thread Tokio runtime and return its Handle.
// pub(crate) so that other modules (e.g. memory.rs) can spawn onto the same
// runtime, ensuring consistent async behaviour across all URI schemes.
pub(crate) fn global_rt_handle() -> &'static Handle {
    RT_HANDLE.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("s3dlio-rt".to_string())
            .spawn(move || {
                // Intelligent thread count with environment override
                let threads = get_runtime_threads();
                debug!("Creating Tokio runtime with {} worker threads", threads);

                let rt = TokioBuilder::new_multi_thread()
                    .enable_io()
                    .enable_time()
                    .worker_threads(threads)
                    .thread_name("s3dlio-rt-worker")
                    .build()
                    .expect("failed to build global tokio runtime");

                // Send a Handle clone back to the creator, then park the runtime forever.
                let handle = rt.handle().clone();
                tx.send(handle).expect("send runtime handle");
                rt.block_on(async { std::future::pending::<()>().await });
            })
            .expect("failed to spawn s3dlio runtime thread");

        rx.recv().expect("receive runtime handle")
    })
}

/// Read `S3DLIO_RT_THREADS` from the environment and apply the sanity
/// clamp against [`RT_THREADS_LIMIT`] shared by every s3dlio-owned thread
/// pool.  Returns `None` if the env var is unset or unparseable.
///
/// See [`get_runtime_threads`] for the full rationale of the clamp and
/// the `S3DLIO_RT_THREADS_UNSAFE=1` escape hatch.
pub(crate) fn clamped_env_rt_threads() -> Option<usize> {
    let n = std::env::var("S3DLIO_RT_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())?;
    let limit = RT_THREADS_LIMIT.load(Ordering::Relaxed);
    let unsafe_bypass = std::env::var("S3DLIO_RT_THREADS_UNSAFE")
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    if limit > 0 && !unsafe_bypass && n.saturating_mul(4) < limit {
        eprintln!(
            "s3dlio: S3DLIO_RT_THREADS={} is < RT_THREADS_LIMIT/4 (limit={}); \
             using {} instead to avoid Tokio-runtime starvation. Set \
             S3DLIO_RT_THREADS_UNSAFE=1 to bypass this safeguard.",
            n, limit, limit,
        );
        return Some(limit.max(1));
    }
    Some(n.max(1))
}

/// Get optimal number of runtime threads with environment override.
///
/// Resolution order (first match wins):
/// 1. `S3DLIO_RT_THREADS` env var (user override), subject to a sanity
///    clamp against `RT_THREADS_LIMIT` — values below `LIMIT/4` are
///    treated as downstream miscomputation and replaced with `LIMIT`.
///    Set `S3DLIO_RT_THREADS_UNSAFE=1` to bypass the clamp.
/// 2. Concurrency hint from [`configure_for_concurrency`], if > CPU baseline
/// 3. MPI-aware default: when `mpi_world_size() > 1` (i.e. running under
///    `mpirun -n N` / `torchrun --nproc-per-node=N`), `num_cpus / world_size`
///    — otherwise `max(4, num_cpus)`, unchanged from before.
///
/// Folding MPI-awareness into the *default* (rather than requiring every
/// caller to explicitly call `configure_tokio_threads()` first) closes an
/// entire class of CPU-oversubscription bugs: N ranks on the same host each
/// independently claiming the full core count. See
/// [`crate::constants::mpi_aware_thread_budget`] for the shared formula.
///
/// The env-var sanity clamp closes a symmetric class of runtime-starvation
/// bugs: a downstream library (e.g. DLIO's `ObjStoreLibStorage`) that
/// derives `S3DLIO_RT_THREADS` from its own not-yet-finalized state can
/// end up setting it to `1`, which — without the clamp — would build the
/// s3dlio global Tokio runtime with a single worker and force every
/// concurrent MPU part upload to serialize on it (~10x throughput loss
/// on real workloads).  See
/// `docs/investigation/DLIO_UNET3D_DATAGEN_BOTTLENECK_INVESTIGATION_2026-07-10.md`.
///
/// The old `min(cores×2, 32)` hard cap has been removed.  On a 96-core host
/// it capped at 32, wasting 64 cores for Python/sync callers.  Tokio I/O
/// threads sit in epoll/io_uring when idle — no CPU cost.
fn get_runtime_threads() -> usize {
    // Env var wins, subject to a sanity clamp against the configured
    // per-process target (RT_THREADS_LIMIT, set by configure_thread_pools;
    // at s3dlio import time _pymod calls configure_thread_pools(0) which
    // sets it to the MPI-aware auto-budget).
    if let Some(n) = clamped_env_rt_threads() {
        return n;
    }

    let cores = num_cpus::get();
    let base = if crate::constants::mpi_world_size() > 1 {
        crate::constants::mpi_aware_thread_budget()
    } else {
        // Floor at 4 for single/dual-core VMs.
        std::cmp::max(4, cores)
    };

    // Respect a concurrency hint (e.g. configure_for_concurrency(128) on a
    // 4-core machine should give the runtime enough threads to interleave I/O
    // for 128 tasks).  Cap at cores*4 to avoid unbounded thread creation on
    // hosts with a tiny core count and a very large hint.
    let hint = CONCURRENCY_HINT.load(Ordering::Relaxed);
    let threads = if hint > base {
        std::cmp::min(hint, cores * 4)
    } else {
        base
    };

    // Hard upper bound set by configure_tokio_threads() — enforces MPI-aware
    // per-process thread budget so NP processes don't all claim all cores.
    let limit = RT_THREADS_LIMIT.load(Ordering::Relaxed);
    if limit > 0 {
        std::cmp::min(threads, limit).max(1)
    } else {
        threads
    }
}

/// Run an async `fut` on the global runtime and block the **current** thread
/// until it completes. Handles both runtime and non-runtime contexts.
pub fn run_on_global_rt<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    // Check if we're already in a runtime context
    match tokio::runtime::Handle::try_current() {
        Ok(_) => {
            // We're already in a runtime context, but we still need to execute on our global runtime
            // Use spawn and block with a different approach
            let handle = global_rt_handle().clone();
            let (tx, rx) = std::sync::mpsc::channel();

            handle.spawn(async move {
                let result = fut.await;
                let _ = tx.send(result);
            });

            // Use blocking receive which works even from within runtime context
            rx.recv()
                .map_err(|_| anyhow::anyhow!("global runtime task crashed: RecvError(())"))?
        }
        Err(_) => {
            // Not in a runtime, use our original approach with oneshot
            let handle = global_rt_handle().clone();
            let (tx, rx) = oneshot::channel();

            handle.spawn(async move {
                let _ = tx.send(fut.await);
            });

            // Block this plain OS thread until the async result arrives.
            rx.blocking_recv()
                .map_err(|_| anyhow::anyhow!("global runtime task crashed: RecvError(())"))?
        }
    }
}

/// Spawn a task on the global runtime without blocking (non-blocking spawn).
/// Returns a JoinHandle immediately that can be awaited later.
///
/// This is more efficient than `run_on_global_rt` when you just want to
/// kick off async work and poll it later, avoiding the channel overhead.
pub fn spawn_on_global_rt<F, T>(fut: F) -> tokio::task::JoinHandle<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    global_rt_handle().spawn(fut)
}

// -----------------------------------------------------------------------------
// HTTP Client Configuration
// -----------------------------------------------------------------------------

/// Get operation timeout for large file transfers
fn get_operation_timeout() -> Duration {
    std::env::var("S3DLIO_OPERATION_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(crate::constants::DEFAULT_OPERATION_TIMEOUT_SECS))
}

// -----------------------------------------------------------------------------
// Client factory (built on the global runtime)
// -----------------------------------------------------------------------------

/// Synchronous wrapper for places that are not async (e.g., Python entrypoints).
/// Internally hops onto the global runtime via `run_on_global_rt`.
pub fn aws_s3_client() -> Result<Client> {
    run_on_global_rt(async { aws_s3_client_async().await })
}

// -----------------------------------------------------------------------------
// Async S3 client (safe to call from any async context)
// -----------------------------------------------------------------------------
/// Async getter for the global S3 client.
/// Safe to call from any async context; initializes once without blocking.
pub async fn aws_s3_client_async() -> Result<Client> {
    let client_ref = CLIENT
        .get_or_try_init(|| async {
            dotenvy::dotenv().ok();

            if env::var("AWS_ACCESS_KEY_ID").is_err() || env::var("AWS_SECRET_ACCESS_KEY").is_err()
            {
                bail!("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY");
            }
            info!("Initializing S3 client");

            // Build HTTP client — always use the reqwest-based transport so that
            // HTTP version detection, h2c support, and connection pool tuning are
            // available regardless of whether a custom CA bundle is configured.
            let ca_val  = env::var("AWS_CA_BUNDLE").ok();
            let ca_path = ca_val.as_deref().filter(|s| !s.is_empty());
            if let Some(path) = ca_path {
                info!("AWS_CA_BUNDLE set — loading CA bundle from: {}", path);
            } else {
                info!("AWS_CA_BUNDLE not set — using system default TLS trust store");
            }
            let http_client = crate::reqwest_client::build_smithy_http_client(ca_path)?;

            // Optionally wrap with redirect following for AIStore compatibility.
            // AIStore proxy nodes return HTTP 307 → Location: http://target-node/...
            // which the AWS SDK's default HTTP client does not follow cross-host.
            // Enable via: S3DLIO_FOLLOW_REDIRECTS=1  (also: true/yes/on/enable)
            let follow_redirects_env = env::var("S3DLIO_FOLLOW_REDIRECTS").unwrap_or_default();
            let http_client = if matches!(
                follow_redirects_env.to_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "enable"
            ) {
                info!("S3DLIO_FOLLOW_REDIRECTS enabled — following 307/302/308 redirects (AIStore support)");
                crate::redirect_client::make_redirecting_client(http_client)
            } else {
                http_client
            };

            // Region & optional endpoint
            let effective_region = env::var("AWS_REGION")
                .ok()
                .or_else(|| env::var("AWS_DEFAULT_REGION").ok());
            debug!("Region env: {}",
                effective_region.as_deref().unwrap_or("<not set — defaulting to us-east-1>"));
            let region =
                RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
                    .or_default_provider()
                    .or_else(Region::new(DEFAULT_REGION));

            let mut loader =
                aws_config::defaults(aws_config::BehaviorVersion::v2026_01_12()).region(region);
            if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
                if !endpoint.is_empty() {
                    info!("Custom S3 endpoint: {}", endpoint);
                    loader = loader.endpoint_url(endpoint);
                }
            }

            // Load config fully async with optimized timeout configuration.
            // connect_timeout honors S3DLIO_CONNECT_TIMEOUT_SECS (default 20s) —
            // unified with the reqwest transport so both layers agree.
            // Retry count honors S3DLIO_MAX_RETRY_ATTEMPTS (default 3, same as
            // SDK default); set to 1 for fast-fail at warmup.
            let op_timeout = get_operation_timeout();
            let connect_secs = crate::constants::connect_timeout_secs();
            let max_attempts = crate::constants::max_retry_attempts();
            debug!(
                "Timeouts — connect: {}s, operation: {:?}, max_attempts: {}",
                connect_secs, op_timeout, max_attempts
            );
            let timeout_config = TimeoutConfig::builder()
                .connect_timeout(Duration::from_secs(connect_secs))
                .operation_timeout(op_timeout)
                .build();
            let retry_config = RetryConfig::standard().with_max_attempts(max_attempts);

            let mut config_builder = loader
                .timeout_config(timeout_config)
                .retry_config(retry_config);

            // Conditionally set HTTP client only if we have one
            config_builder = config_builder.http_client(http_client);

            let cfg = config_builder.load().await;

            // =========================================================================
            // FORCE PATH-STYLE ADDRESSING (added 2025-12-03)
            // Required for S3-compatible services (MinIO, Ceph, etc.)
            // Virtual-hosted style (bucket.endpoint) doesn't work with custom endpoints.
            // Path-style (endpoint/bucket) is the standard for S3-compatible services.
            // To revert: replace this block with just `Client::new(&cfg)`
            // =========================================================================
            let s3_config = aws_sdk_s3::config::Builder::from(&cfg)
                .force_path_style(true)
                // Disable automatic CRC checksum on PUT and validation on GET.
                // AWS SDK v2026+ defaults to CRC64-NVME on all requests, which
                // breaks S3-compatible servers (s3-ultra, MinIO older builds, etc.)
                // that don't store or return AWS-style checksum headers.
                .request_checksum_calculation(aws_sdk_s3::config::RequestChecksumCalculation::WhenRequired)
                .response_checksum_validation(aws_sdk_s3::config::ResponseChecksumValidation::WhenRequired)
                .build();
            info!("S3 client ready (path-style: forced, endpoint: {})",
                env::var("AWS_ENDPOINT_URL").ok().as_deref().unwrap_or("AWS default"));
            Ok::<_, anyhow::Error>(Client::from_conf(s3_config))
        })
        .await?;

    Ok(client_ref.clone())
}

// -----------------------------------------------------------------------------
// Per-endpoint S3 client factory
// -----------------------------------------------------------------------------

/// Create a new S3 client configured for a specific endpoint URL.
///
/// Unlike `aws_s3_client_async()` which returns the global singleton, this
/// always creates a **new** client with its own connection pool targeting the
/// given endpoint. Used by `S3ObjectStore::for_endpoint()` to achieve per-endpoint
/// client isolation (critical for multi-endpoint high-throughput workloads).
///
/// The optional `http_client` parameter lets callers inject a custom HTTP transport
/// (e.g., reqwest-based h2c). When `None`, a new reqwest client is created.
///
/// # Environment
/// - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — required
/// - `AWS_REGION` — optional, defaults to `us-east-1`
/// - `S3DLIO_H2C=1` — enables HTTP/2 cleartext transport
/// - `S3DLIO_FOLLOW_REDIRECTS=1` — wraps transport with redirect follower
/// - `AWS_CA_BUNDLE` — custom TLS CA certificate file
pub async fn create_s3_client_for_endpoint(
    endpoint_url: &str,
    http_client: Option<SharedHttpClient>,
) -> Result<Client> {
    dotenvy::dotenv().ok();

    if env::var("AWS_ACCESS_KEY_ID").is_err() || env::var("AWS_SECRET_ACCESS_KEY").is_err() {
        bail!("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY");
    }

    let region = RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
        .or_default_provider()
        .or_else(Region::new(DEFAULT_REGION));

    // Build HTTP client: use provided one, or create a new reqwest client.
    // Always use the reqwest-based transport so HTTP version detection and
    // h2c support are available. Load the CA bundle into reqwest when set.
    let http_client = match http_client {
        Some(c) => c,
        None => {
            let ca = env::var("AWS_CA_BUNDLE").ok();
            let ca_path = ca.as_deref().filter(|s| !s.is_empty());
            crate::reqwest_client::build_smithy_http_client(ca_path)?
        }
    };

    // Optionally wrap with redirect follower (AIStore compatibility)
    let follow_redirects_env = env::var("S3DLIO_FOLLOW_REDIRECTS").unwrap_or_default();
    let http_client = if matches!(
        follow_redirects_env.to_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enable"
    ) {
        crate::redirect_client::make_redirecting_client(http_client)
    } else {
        http_client
    };

    let op_timeout = get_operation_timeout();
    // Per-endpoint client: same connect-timeout / retry policy as the global
    // client.  S3DLIO_CONNECT_TIMEOUT_SECS (default 20s) for both transport
    // and SDK layers; S3DLIO_MAX_RETRY_ATTEMPTS (default 3) for SDK retries.
    let timeout_config = TimeoutConfig::builder()
        .connect_timeout(Duration::from_secs(crate::constants::connect_timeout_secs()))
        .operation_timeout(op_timeout)
        .build();
    let retry_config =
        RetryConfig::standard().with_max_attempts(crate::constants::max_retry_attempts());

    let cfg = aws_config::defaults(aws_config::BehaviorVersion::v2026_01_12())
        .region(region)
        .endpoint_url(endpoint_url)
        .timeout_config(timeout_config)
        .retry_config(retry_config)
        .http_client(http_client)
        .load()
        .await;

    let s3_config = aws_sdk_s3::config::Builder::from(&cfg)
        .force_path_style(true)
        .request_checksum_calculation(aws_sdk_s3::config::RequestChecksumCalculation::WhenRequired)
        .response_checksum_validation(aws_sdk_s3::config::ResponseChecksumValidation::WhenRequired)
        .build();

    info!(
        "Per-endpoint S3 client ready (path-style: forced, endpoint: {})",
        endpoint_url
    );
    Ok(Client::from_conf(s3_config))
}

#[cfg(test)]
mod mpi_aware_runtime_sizing_tests {
    use super::*;
    use std::sync::Mutex;

    // get_runtime_threads() reads process-global env vars and atomics
    // (CONCURRENCY_HINT, RT_THREADS_LIMIT); serialize these tests both
    // against each other and reset the atomics we touch so we don't leak
    // state into other tests in this binary.
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    const VARS: [&str; 5] = [
        "S3DLIO_RT_THREADS",
        "S3DLIO_RT_THREADS_UNSAFE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "WORLD_SIZE",
    ];

    fn with_clean_state<F: FnOnce()>(body: F) {
        let _guard = ENV_LOCK.lock().expect("ENV_LOCK poisoned");
        let prev: Vec<Option<String>> = VARS.iter().map(|v| std::env::var(v).ok()).collect();
        for v in VARS {
            std::env::remove_var(v);
        }
        let prev_hint = CONCURRENCY_HINT.swap(0, Ordering::Relaxed);
        let prev_limit = RT_THREADS_LIMIT.swap(0, Ordering::Relaxed);

        body();

        for (v, p) in VARS.iter().zip(prev) {
            match p {
                Some(val) => std::env::set_var(v, val),
                None => std::env::remove_var(v),
            }
        }
        CONCURRENCY_HINT.store(prev_hint, Ordering::Relaxed);
        RT_THREADS_LIMIT.store(prev_limit, Ordering::Relaxed);
    }

    /// Reproduces the CPU-oversubscription bug found while chasing the DLIO
    /// unet3d npz-datagen slowdown: under `mpirun -n N`, every rank's
    /// `s3dlio` process independently defaults `get_runtime_threads()` to
    /// `max(4, num_cpus)` -- N ranks all claim the FULL core count instead
    /// of splitting it, unless something explicitly calls
    /// `configure_tokio_threads()` first (which, in DLIO_local_changes, only
    /// 2 of 16 s3dlio-touching call sites do). The global Tokio runtime
    /// (`get_runtime_threads()`'s caller) must instead default to an
    /// MPI-aware per-process budget whenever MPI env vars are present, with
    /// zero opt-in required.
    #[test]
    fn get_runtime_threads_is_mpi_aware_by_default_without_configure_call() {
        with_clean_state(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            let cpus = num_cpus::get();
            let expected = (cpus / 4).max(1);
            assert_eq!(
                get_runtime_threads(),
                expected,
                "get_runtime_threads() must divide available CPUs across the \
                 MPI world size by default -- it ignored OMPI_COMM_WORLD_SIZE=4 \
                 and returned an unconstrained per-process thread count"
            );
        });
    }

    #[test]
    fn get_runtime_threads_single_process_unchanged_behavior() {
        // world_size=1 (no MPI env vars) must reduce to the pre-existing
        // max(4, num_cpus) default -- zero behavior change for the common
        // single-process case.
        with_clean_state(|| {
            let cpus = num_cpus::get();
            assert_eq!(get_runtime_threads(), std::cmp::max(4, cpus));
        });
    }

    #[test]
    fn get_runtime_threads_explicit_env_var_still_wins_over_mpi_default() {
        with_clean_state(|| {
            std::env::set_var("S3DLIO_RT_THREADS", "7");
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            assert_eq!(get_runtime_threads(), 7);
        });
    }

    #[test]
    fn get_runtime_threads_explicit_limit_still_caps_mpi_default() {
        // The configure_tokio_threads() escape hatch (RT_THREADS_LIMIT) must
        // still be able to further constrain the MPI-aware default.
        with_clean_state(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            RT_THREADS_LIMIT.store(1, Ordering::Relaxed);
            assert_eq!(get_runtime_threads(), 1);
        });
    }

    #[test]
    fn configure_thread_pools_auto_sets_rt_threads_limit_from_mpi_env() {
        with_clean_state(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            let cpus = num_cpus::get();
            let expected = (cpus / 4).max(1);
            configure_thread_pools(0);
            assert_eq!(RT_THREADS_LIMIT.load(Ordering::Relaxed), expected);
        });
    }

    #[test]
    fn configure_thread_pools_explicit_count_overrides_mpi_default() {
        with_clean_state(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            configure_thread_pools(3);
            assert_eq!(RT_THREADS_LIMIT.load(Ordering::Relaxed), 3);
        });
    }

    #[test]
    fn configure_thread_pools_is_idempotent_does_not_panic_on_repeat_calls() {
        // Rayon's global pool can only be *built* once per process; calling
        // this twice (e.g. an automatic call at import followed by an
        // explicit user override) must not panic even though the second
        // build_global() call is necessarily a no-op.
        with_clean_state(|| {
            configure_thread_pools(2);
            configure_thread_pools(4);
            assert_eq!(RT_THREADS_LIMIT.load(Ordering::Relaxed), 4);
        });
    }

    /// RED-then-GREEN regression for the DLIO unet3d datagen bug (found
    /// 2026-07-10, docs/investigation/DLIO_UNET3D_DATAGEN_BOTTLENECK_
    /// INVESTIGATION_2026-07-10.md): DLIO's `ObjStoreLibStorage.__init__`
    /// computes `S3DLIO_RT_THREADS = write_threads * 3 // 2`, and with
    /// Hydra's default `write_threads = 1` that resolves to
    /// `S3DLIO_RT_THREADS=1`, which s3dlio then faithfully obeys by
    /// building a Tokio runtime with ONE worker.  All concurrent MPU part
    /// uploads then serialize on that single worker (~10x throughput
    /// loss).  Fix: env-var values that are far below the configured
    /// per-process target (`RT_THREADS_LIMIT`, set by
    /// `configure_thread_pools(0)` at import time to the MPI-aware
    /// auto-budget) are treated as downstream miscomputation and clamped
    /// up to the target instead of starving the runtime.
    #[test]
    fn get_runtime_threads_clamps_grossly_underprovisioned_env_var_up_to_limit() {
        with_clean_state(|| {
            std::env::set_var("S3DLIO_RT_THREADS", "1");
            RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
            assert_eq!(
                get_runtime_threads(),
                28,
                "S3DLIO_RT_THREADS=1 with RT_THREADS_LIMIT=28 must clamp up \
                 to the configured target (1 is < LIMIT/4 = 7).  Under-\
                 provisioned env var almost always means downstream code \
                 miscomputed the value (e.g. DLIO's write_threads*1.5 with \
                 write_threads defaulted to 1)."
            );
        });
    }

    #[test]
    fn get_runtime_threads_env_var_at_or_above_quarter_of_limit_is_honored_verbatim() {
        // Values above LIMIT/4 are honored as-is -- the sanity clamp is
        // strictly for grossly-wrong values.  User's ability to override
        // downward within a reasonable range is preserved.
        with_clean_state(|| {
            RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
            // Exactly the boundary: LIMIT/4 = 7.  7 >= 7, so honored.
            std::env::set_var("S3DLIO_RT_THREADS", "7");
            assert_eq!(get_runtime_threads(), 7);
        });
        with_clean_state(|| {
            RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
            std::env::set_var("S3DLIO_RT_THREADS", "12");
            assert_eq!(get_runtime_threads(), 12);
        });
    }

    #[test]
    fn get_runtime_threads_unsafe_env_bypasses_the_clamp() {
        // Escape hatch: users who genuinely want a low thread count (e.g.
        // running s3dlio's own single-threaded test suite, or testing
        // fault-injection scenarios) can set S3DLIO_RT_THREADS_UNSAFE=1
        // to bypass the sanity clamp.
        with_clean_state(|| {
            RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
            std::env::set_var("S3DLIO_RT_THREADS", "1");
            std::env::set_var("S3DLIO_RT_THREADS_UNSAFE", "1");
            assert_eq!(get_runtime_threads(), 1);
        });
    }

    #[test]
    fn get_runtime_threads_env_var_low_with_no_limit_set_is_honored() {
        // When configure_thread_pools has not been called (RT_THREADS_LIMIT
        // == 0), there is no target to compare against, so the clamp is
        // inactive and the env var wins verbatim.  This keeps the pure-
        // Rust `use s3dlio::…` path (no _pymod, no auto-configure) fully
        // backward-compatible with any programmatic env-var use.
        with_clean_state(|| {
            std::env::set_var("S3DLIO_RT_THREADS", "1");
            // RT_THREADS_LIMIT stays 0 from with_clean_state.
            assert_eq!(get_runtime_threads(), 1);
        });
    }
}

#[cfg(test)]
mod tier4_reentrancy_tests {
    use super::*;

    /// Reproduces the exact hazard described in
    /// docs/DESIGN_TIER4_FFI_HARDENING.md item 3: 7 call sites in
    /// `python_core_api.rs` used to call `.block_on()` directly instead of
    /// `run_on_global_rt`, which panics with "Cannot start a runtime from
    /// within a runtime" if invoked from a call stack already executing
    /// inside a Tokio runtime. `run_on_global_rt` (the pattern those 7
    /// sites were migrated to) must survive the identical nested condition.
    #[test]
    fn run_on_global_rt_survives_nested_runtime_context() {
        let outer_rt = tokio::runtime::Runtime::new().unwrap();
        let result: Result<i32> = outer_rt.block_on(async {
            // We're now executing on an OS thread that already has a Tokio
            // runtime context entered -- the exact "reentrant" condition.
            run_on_global_rt(async { Ok(42) })
        });
        assert_eq!(result.unwrap(), 42);
    }
}
