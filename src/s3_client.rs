// src/s3_client.rs
// 
// Copyright, 2025.  Signal65 / Futurum Group.
//
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides only the S3 client creation code 
//!

use anyhow::{bail, Context, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_smithy_http_client::tls::{self, rustls_provider::CryptoMode};
use aws_sdk_s3::{config::Region, Client};
use once_cell::sync::OnceCell;
use tokio::{runtime::Handle, task};

use std::{env, fs};
use std::path::{Path, PathBuf};
use log::{info, debug}; // For logging


// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_REGION: &str     = "us-east-1";


// -----------------------------------------------------------------------------
// Global S3 client (lazy, thread‑safe)
// -----------------------------------------------------------------------------
static CLIENT: OnceCell<Client> = OnceCell::new();
static RUNTIME: OnceCell<tokio::runtime::Runtime> = OnceCell::new();   // for block_on fallback



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


/// Instantiate the S3 client
pub fn aws_s3_client() -> Result<Client> {
    CLIENT.get_or_try_init(|| {
        dotenvy::dotenv().ok();
        if env::var("AWS_ACCESS_KEY_ID").is_err() || env::var("AWS_SECRET_ACCESS_KEY").is_err() {
            bail!("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY");
        }

        // --- Begin: Add TLS context based on AWS_CA_BUNDLE_PATH ---
        let mut http_client_builder = aws_smithy_http_client::Builder::new()
            .tls_provider(tls::Provider::Rustls(CryptoMode::AwsLc));

        if let Ok(ca_bundle_path) = env::var("AWS_CA_BUNDLE_PATH") {
            debug!("Loading CA bundle from environment: {}", ca_bundle_path);
            info!("Loading CA bundle from environment: {}", ca_bundle_path);
            let ca_bundle_path_env = PathBuf::from(ca_bundle_path);

            // Build TLS context with custom CA
            let tls_context = tls_context_from_pem(ca_bundle_path_env.clone())?;
            http_client_builder = http_client_builder.tls_context(tls_context);
        }
        // --- End: Add TLS context based on AWS_CA_BUNDLE_PATH ---

        // Add default region
        let region = RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
            .or_default_provider()
            .or_else(Region::new(DEFAULT_REGION));

        // Load the AWS config defaults 
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region);
        if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
            if !endpoint.is_empty() {
                loader = loader.endpoint_url(endpoint);
            }
        }

        // This is the old loader, note we don't use the http_client_builder
        //let fut = loader.load();
        //
        // Use the configured http_client_builder
        let fut = loader.http_client(http_client_builder.build_https()).load();
        let cfg = match Handle::try_current() {
            Ok(handle) => task::block_in_place(|| handle.block_on(fut)),
            Err(_) => {
                let rt = RUNTIME.get_or_init(|| {
                    tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
                });
                rt.block_on(fut)
            }
        };
        Ok::<_, anyhow::Error>(Client::new(&cfg))
    }).map(Clone::clone)
}

// -----------------------------------------------------------------------------
// Helper: synchronously wait on a future
// -----------------------------------------------------------------------------
pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    if let Ok(handle) = Handle::try_current() {
        handle.block_on(fut)
    } else {
        let rt = RUNTIME.get_or_init(|| {
            tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
        });
        rt.block_on(fut)
    }
}

