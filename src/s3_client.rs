// src/s3_client.rs
// 
// Copyright, 2025.  Signal65 / Futurum Group.
//
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides only the S3 client creation code 
//!

use anyhow::{bail, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::{config::Region, Client};
use once_cell::sync::OnceCell;
use std::env;
use tokio::{runtime::Handle, task};


// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_REGION: &str     = "us-east-1";


// -----------------------------------------------------------------------------
// Global S3 client (lazy, thread‑safe)
// -----------------------------------------------------------------------------
static CLIENT: OnceCell<Client> = OnceCell::new();
static RUNTIME: OnceCell<tokio::runtime::Runtime> = OnceCell::new();   // for block_on fallback

pub fn aws_s3_client() -> Result<Client> {
    CLIENT.get_or_try_init(|| {
        dotenvy::dotenv().ok();
        if env::var("AWS_ACCESS_KEY_ID").is_err() || env::var("AWS_SECRET_ACCESS_KEY").is_err() {
            bail!("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY");
        }
        let region = RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
            .or_default_provider()
            .or_else(Region::new(DEFAULT_REGION));
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region);
        if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
            if !endpoint.is_empty() {
                loader = loader.endpoint_url(endpoint);
            }
        }
        let fut = loader.load();
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

