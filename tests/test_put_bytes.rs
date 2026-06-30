// tests/test_put_bytes.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Integration tests for single-part PUT + HEAD verification (storage#593).
//!
//! These tests confirm that:
//!   1. `put_object_uri_async` + `stat_object_uri_async` round-trips return the
//!      correct byte count — the property that `put_verified_with_retry` checks
//!      internally on every call to `s3dlio.put_bytes()`.
//!   2. Various data sizes (small, ~1 MiB, ~20 MiB) all store intact.
//!   3. The bucket `mlp-s3dlio` is reachable and credentials are valid.
//!
//! Run with:
//!   cargo test --test test_put_bytes -- --include-ignored

use anyhow::{ensure, Result};
use bytes::Bytes;

use s3dlio::s3_client::run_on_global_rt;
use s3dlio::s3_utils::{delete_objects_async, put_object_uri_async, stat_object_uri_async};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn bucket() -> String {
    std::env::var("BUCKET").unwrap_or_else(|_| "mlp-s3dlio".to_string())
}

fn unique_key(tag: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("test-put-bytes-{tag}-{ts}.bin")
}

/// PUT data, HEAD-verify the stored size, then clean up.  Returns stored size.
fn put_and_verify(bucket: &str, key: &str, data: Bytes) -> Result<u64> {
    let expected = data.len() as u64;
    let uri = format!("s3://{bucket}/{key}");
    let bucket = bucket.to_owned();
    let key = key.to_owned();

    run_on_global_rt(async move {
        put_object_uri_async(&uri, data).await?;
        let meta = stat_object_uri_async(&uri).await?;
        ensure!(
            meta.size == expected,
            "PUT size mismatch for {uri}: sent {expected} bytes but HEAD reports {} bytes",
            meta.size
        );
        delete_objects_async(&bucket, &[key]).await?;
        Ok(meta.size)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Small object (1 KiB) — confirms basic PUT + HEAD round-trip works.
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_put_bytes_small_object_verified() -> Result<()> {
    dotenvy::dotenv().ok();
    let b = bucket();
    let k = unique_key("small");
    let data = Bytes::from(vec![0xABu8; 1024]);
    let stored = put_and_verify(&b, &k, data)?;
    assert_eq!(stored, 1024, "1 KiB object must store exactly 1024 bytes");
    Ok(())
}

/// Medium object (~1 MiB) — below the multipart threshold; exercises the
/// single-part PUT path that `put_verified_with_retry` guards.
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_put_bytes_medium_object_verified() -> Result<()> {
    dotenvy::dotenv().ok();
    let b = bucket();
    let k = unique_key("medium");
    let size = 1024 * 1024; // 1 MiB
    let data = Bytes::from(vec![0x5Au8; size]);
    let stored = put_and_verify(&b, &k, data)?;
    assert_eq!(
        stored, size as u64,
        "1 MiB object must store exactly {size} bytes"
    );
    Ok(())
}

/// Large object (~20 MiB) — well under default multipart threshold (16 MiB…
/// wait, 20 MiB is above 16 MiB, so DLIO would use MPU; but at the raw
/// `put_object_uri_async` level it is always single-part, confirming single-
/// part PUT handles large objects correctly too).
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_put_bytes_large_single_part_verified() -> Result<()> {
    dotenvy::dotenv().ok();
    let b = bucket();
    let k = unique_key("large");
    let size = 20 * 1024 * 1024; // 20 MiB
    let data = Bytes::from(vec![0xCDu8; size]);
    let stored = put_and_verify(&b, &k, data)?;
    assert_eq!(
        stored, size as u64,
        "20 MiB object must store exactly {size} bytes"
    );
    Ok(())
}

/// Zero-byte object — edge case; HEAD must report 0 bytes.
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_put_bytes_zero_length_object() -> Result<()> {
    dotenvy::dotenv().ok();
    let b = bucket();
    let k = unique_key("zero");
    let data = Bytes::new();
    let stored = put_and_verify(&b, &k, data)?;
    assert_eq!(stored, 0, "empty object must report 0 bytes via HEAD");
    Ok(())
}

/// Confirm `stat_object_uri_async` reports the right size for an object we just
/// replaced — verifies the HEAD path doesn't serve stale metadata.
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_put_bytes_overwrite_size_updates() -> Result<()> {
    dotenvy::dotenv().ok();
    let b = bucket();
    let k = unique_key("overwrite");
    let uri = format!("s3://{b}/{k}");

    run_on_global_rt(async move {
        // First PUT: 4 KiB
        let data_small = Bytes::from(vec![0x11u8; 4096]);
        put_object_uri_async(&uri, data_small).await?;
        let meta1 = stat_object_uri_async(&uri).await?;
        ensure!(
            meta1.size == 4096,
            "first PUT must report 4096 bytes, got {}",
            meta1.size
        );

        // Second PUT (overwrite): 8 KiB
        let data_large = Bytes::from(vec![0x22u8; 8192]);
        put_object_uri_async(&uri, data_large).await?;
        let meta2 = stat_object_uri_async(&uri).await?;
        ensure!(
            meta2.size == 8192,
            "overwrite must report 8192 bytes, got {} (stale metadata?)",
            meta2.size
        );

        delete_objects_async(&b, &[k]).await?;
        Ok(())
    })
}
