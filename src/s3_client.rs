// src/s3_client.rs
// 
// Copyright, 2025.  Signal65 / Futurum Group.
//
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Owns a single global multi-thread Tokio runtime and the global S3 client.
//!

use anyhow::{bail, Context, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_config::timeout::TimeoutConfig;
use aws_sdk_s3::{config::Region, Client};
use aws_smithy_http_client::{tls, Builder as HttpClientBuilder};
use aws_smithy_http_client::tls::rustls_provider::CryptoMode;
#[cfg(feature = "experimental-http-client")]
use aws_smithy_http_client::Connector;
use std::{env, fs, thread, time::Duration};
use tokio::runtime::{Builder as TokioBuilder, Handle};
use tokio::sync::{oneshot, OnceCell};
use aws_smithy_runtime_api::client::http::SharedHttpClient;
use std::path::Path;
use std::sync::mpsc;
use tracing::debug; // For logging


// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_REGION: &str     = "us-east-1";


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



// -----------------------------------------------------------------------------
// TLS helper, for CA bundle
// -----------------------------------------------------------------------------

/// Create a TLS context using a CA bundle file
fn tls_context_from_pem(filename: impl AsRef<Path>) -> Result<tls::TlsContext> {
    // Read the file, but wrap any IO error
    let pem_contents = fs::read(&filename)
        .with_context(|| format!("Failed to read CA bundle file: {}", filename.as_ref().display()))?;

    // Build a trust store containing exactly that PEM
    let trust_store = tls::TrustStore::empty()
        .with_pem_certificate(pem_contents.as_slice());

    // Build the TlsContext, bubbling up any builder error
    tls::TlsContext::builder()
        .with_trust_store(trust_store)
        .build()
        .with_context(|| format!("Failed to build TLS context from PEM {}", filename.as_ref().display()))
}


// -----------------------------------------------------------------------------
// HTTP Client Configuration
// -----------------------------------------------------------------------------

/// Get HTTP configuration values from environment with performance-oriented defaults
#[cfg(feature = "experimental-http-client")]
fn get_max_http_connections() -> usize {
    std::env::var("S3DLIO_MAX_HTTP_CONNECTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            // Conservative optimization: Don't over-allocate connections
            // Too many can cause contention, too few limit throughput
            200  // Reduced from 600 - more conservative approach
        })
}

/// Get HTTP idle timeout optimized for storage speed
/// User suggested: ~100ms per MB for fast local storage
#[cfg(feature = "experimental-http-client")]
fn get_http_idle_timeout() -> Duration {
    std::env::var("S3DLIO_HTTP_IDLE_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| {
            // Back to user's original recommendation: ~100ms per MB
            // For 8MB objects: 800ms timeout  
            // More conservative than our previous 2s timeout
            Duration::from_millis(800)  // Original user recommendation
        })
}

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
            Duration::from_secs(120)  // 2 minutes per operation - much faster
        })
}

/// Create an optimized HTTP client with connection pool configuration
/// 
/// NOTE: This function requires the "experimental-http-client" feature and a patched
/// aws-smithy-http-client to access hyper_builder. Without the patch, this code won't
/// compile. Enable via: cargo build --features experimental-http-client
#[cfg(feature = "experimental-http-client")]
fn create_optimized_http_client() -> Result<SharedHttpClient> {
    // Get performance configuration
    let max_connections = get_max_http_connections();
    let idle_timeout = get_http_idle_timeout();
    
    debug!("Configuring experimental optimized HTTP client: max_connections={}, idle_timeout={:?}", 
           max_connections, idle_timeout);
    
    // Create hyper client with optimized connection pool settings  
    let executor = hyper_util::rt::TokioExecutor::new();
    let mut hyper_builder = hyper_util::client::legacy::Builder::new(executor);
    hyper_builder
        .pool_max_idle_per_host(max_connections)  // Maximum connections per host
        .pool_idle_timeout(idle_timeout)          // Keep-alive timeout
        .timer(hyper_util::rt::TokioTimer::new()) // Use Tokio timer
        .http2_only(false)                        // Allow HTTP/1.1 and HTTP/2
        .http2_adaptive_window(true)              // Enable HTTP/2 adaptive windows
        .http2_keep_alive_interval(Duration::from_secs(30))  // Conservative keep-alive
        .http2_keep_alive_timeout(Duration::from_secs(10));  // Conservative timeout
        
    // Then create a SharedHttpClient from the optimized Connector
    // Since Connector doesn't implement Clone, we need to create it inside the closure
    let http_client = HttpClientBuilder::new()
        .build_with_connector_fn({
            let max_connections = max_connections;
            let idle_timeout = idle_timeout;
            move |_settings, _components| {
                // Recreate the hyper builder inside the closure
                let executor = hyper_util::rt::TokioExecutor::new();
                let mut hyper_builder = hyper_util::client::legacy::Builder::new(executor);
                hyper_builder
                    .pool_max_idle_per_host(max_connections)
                    .pool_idle_timeout(idle_timeout)
                    .timer(hyper_util::rt::TokioTimer::new())
                    .http2_only(false)
                    .http2_adaptive_window(true)
                    .http2_keep_alive_interval(Duration::from_secs(30))
                    .http2_keep_alive_timeout(Duration::from_secs(10));
                    
                Connector::builder()
                    .hyper_builder(hyper_builder)
                    .tls_provider(tls::Provider::Rustls(CryptoMode::AwsLc))
                    .build()
            }
        });
    
    // NOTE: info!() removed 2025-12-03 - causes hangs in async context with tracing
    // info!("Experimental optimized HTTP client created with {} max connections per host", max_connections);
    eprintln!("[s3dlio] Experimental HTTP client: {} max connections/host", max_connections);
    Ok(http_client)
}

/// Create default HTTP client without custom optimizations
/// This is the standard path that doesn't require patched dependencies
#[cfg(not(feature = "experimental-http-client"))]
fn create_optimized_http_client() -> Result<SharedHttpClient> {
    debug!("Using default AWS SDK HTTP client (experimental-http-client feature not enabled)");
    
    // Return standard AWS SDK HTTP client with Rustls TLS
    // build_http() creates an HTTPS client with Rustls by default
    Ok(HttpClientBuilder::new()
        .build_http())
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

            // Create HTTP client with optimized settings
            // NOTE: debug!() commented out 2025-12-03 - causes hangs in async context with tracing
            // debug!("Building HTTP client with optimization settings: max_connections={}, idle_timeout={}ms", 
            //        get_max_http_connections(), get_http_idle_timeout().as_millis());

            let http_client = match env::var("AWS_CA_BUNDLE_PATH") {
                Ok(ca_bundle_path) if !ca_bundle_path.is_empty() => {
                    // User has specified custom CA bundle via environment
                    // This is a standard AWS SDK feature that must work in all builds
                    eprintln!("[s3dlio] Loading CA bundle from: {}", ca_bundle_path);
                    let tls_context = tls_context_from_pem(&ca_bundle_path)?;
                    
                    // Build HTTPS client with custom CA using standard AWS SDK API
                    Some(aws_smithy_http_client::Builder::new()
                        .tls_provider(tls::Provider::Rustls(CryptoMode::AwsLc))
                        .tls_context(tls_context)
                        .build_https())
                },
                _ => {
                    // Check if optimized HTTP client is enabled via environment variable
                    match env::var("S3DLIO_USE_OPTIMIZED_HTTP").unwrap_or_default().to_lowercase().as_str() {
                        "true" | "1" | "yes" | "on" | "enable" => {
                            // Use our optimized HTTP client with connection pooling (opt-in)
                            eprintln!("[s3dlio] HTTP optimization enabled: Enhanced connection pooling");
                            Some(create_optimized_http_client()?)
                        },
                        _ => {
                            // Use default AWS SDK configuration (DEFAULT)
                            None
                        }
                    }
                }
            };

            // Region & optional endpoint
            let region =
                RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
                    .or_default_provider()
                    .or_else(Region::new(DEFAULT_REGION));

            let mut loader =
                aws_config::defaults(aws_config::BehaviorVersion::v2025_08_07()).region(region);
            if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
                if !endpoint.is_empty() {
                    loader = loader.endpoint_url(endpoint);
                }
            }

            // Load config fully async with optimized timeout configuration
            let timeout_config = TimeoutConfig::builder()
                .connect_timeout(Duration::from_secs(5))  // Quick connection timeout
                .operation_timeout(get_operation_timeout()) // Configurable for large transfers
                .build();

            let mut config_builder = loader.timeout_config(timeout_config);
            
            // Conditionally set HTTP client only if we have one
            if let Some(client) = http_client {
                config_builder = config_builder.http_client(client);
            }
            
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
            Ok::<_, anyhow::Error>(Client::from_conf(s3_config))
        })
        .await?;

    Ok(client_ref.clone())
}
