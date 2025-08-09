// src/s3_client.rs
// 
// Copyright, 2025.  Signal65 / Futurum Group.
//
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Owns a single global multi-thread Tokio runtime and the global S3 client.
//!

use anyhow::{bail, Context, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::{config::Region, Client};
use aws_smithy_http_client::tls::{self, rustls_provider::CryptoMode};

//use once_cell::sync::OnceCell;

use tokio::runtime::{Builder, Handle};
use tokio::sync::{oneshot, OnceCell};

use std::{env, fs, thread};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use log::{info, debug}; // For logging


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
                let rt = Builder::new_multi_thread()
                    .enable_io()
                    .enable_time()
                    .worker_threads(2)
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


/// Run an async `fut` on the global runtime and block the **current** (non-Tokio) thread
/// until it completes. This never blocks a Tokio worker thread.
pub fn run_on_global_rt<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let handle = global_rt_handle().clone();
    let (tx, rx) = oneshot::channel();

    handle.spawn(async move {
        let _ = tx.send(fut.await);
    });

    // Block this plain OS thread until the async result arrives.
    rx.blocking_recv()
        .map_err(|_| anyhow::anyhow!("global runtime task crashed: RecvError(())"))?
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

            // Optional custom CA
            let mut http_client_builder =
                aws_smithy_http_client::Builder::new().tls_provider(tls::Provider::Rustls(
                    CryptoMode::AwsLc,
                ));

            if let Ok(ca_bundle_path) = env::var("AWS_CA_BUNDLE_PATH") {
                debug!("Loading CA bundle from environment: {}", ca_bundle_path);
                info!("Loading CA bundle from environment: {}", ca_bundle_path);
                let ca_bundle_path_env = PathBuf::from(ca_bundle_path);
                let tls_context = tls_context_from_pem(ca_bundle_path_env)?;
                http_client_builder = http_client_builder.tls_context(tls_context);
            }

            // Region & optional endpoint
            let region =
                RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
                    .or_default_provider()
                    .or_else(Region::new(DEFAULT_REGION));

            let mut loader =
                aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region);
            if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
                if !endpoint.is_empty() {
                    loader = loader.endpoint_url(endpoint);
                }
            }

            // Load config fully async (no blocking calls)
            let cfg = loader
                .http_client(http_client_builder.build_https())
                .load()
                .await;

            Ok::<_, anyhow::Error>(Client::new(&cfg))
        })
        .await?;

    Ok(client_ref.clone())
}
