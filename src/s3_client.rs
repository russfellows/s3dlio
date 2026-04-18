// src/s3_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Owns a single global multi-thread Tokio runtime and the global S3 client.
//!

use anyhow::{bail, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_config::timeout::TimeoutConfig;
use aws_sdk_s3::{config::Region, Client};

use aws_smithy_runtime_api::client::http::SharedHttpClient;
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

// Create (once) a background multi-thread Tokio runtime and return its Handle.
fn global_rt_handle() -> &'static Handle {
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

/// Get optimal number of runtime threads with environment override
fn get_runtime_threads() -> usize {
    std::env::var("S3DLIO_RT_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            let cores = num_cpus::get();
            let default_threads = std::cmp::max(8, cores * 2);
            // Cap at reasonable maximum to avoid thread explosion
            std::cmp::min(default_threads, 32)
        })
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
        .unwrap_or_else(|| {
            // For fast storage: ~1 second per 8MB object should be plenty
            // 5000 objects * 1 second = 83 minutes max (very conservative)
            // Use more aggressive timeout for better resource management
            Duration::from_secs(120) // 2 minutes per operation - much faster
        })
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
            debug!("AWS_REGION env: {}", env::var("AWS_REGION").as_deref().unwrap_or("<not set — using provider chain>"));
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

            // Load config fully async with optimized timeout configuration
            let op_timeout = get_operation_timeout();
            debug!("Timeouts — connect: 5s, operation: {:?}", op_timeout);
            let timeout_config = TimeoutConfig::builder()
                .connect_timeout(Duration::from_secs(5))  // Quick connection timeout
                .operation_timeout(op_timeout)             // Configurable for large transfers
                .build();

            let mut config_builder = loader.timeout_config(timeout_config);

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
    let timeout_config = TimeoutConfig::builder()
        .connect_timeout(Duration::from_secs(5))
        .operation_timeout(op_timeout)
        .build();

    let cfg = aws_config::defaults(aws_config::BehaviorVersion::v2026_01_12())
        .region(region)
        .endpoint_url(endpoint_url)
        .timeout_config(timeout_config)
        .http_client(http_client)
        .load()
        .await;

    let s3_config = aws_sdk_s3::config::Builder::from(&cfg)
        .force_path_style(true)
        .build();

    info!(
        "Per-endpoint S3 client ready (path-style: forced, endpoint: {})",
        endpoint_url
    );
    Ok(Client::from_conf(s3_config))
}
