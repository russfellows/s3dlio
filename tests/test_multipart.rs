// tests/test_multipart.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::{ensure, Context, Result};
use std::time::{SystemTime, UNIX_EPOCH};

use s3dlio::s3_client::{aws_s3_client_async, run_on_global_rt};
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};

// ---------------------------------------------------------------------------
// BUG #593 Bug 1 — __exit__ silently discards finish_blocking() errors
// ---------------------------------------------------------------------------

/// Proves that `finish_blocking()` CAN return `Err` — the error that
/// `PyMultipartUploadWriter.__exit__` currently silences.
///
/// We trigger the failure by calling `finish_blocking()` immediately after
/// creating the sink, before writing any parts.  `CompleteMultipartUpload`
/// with an empty parts list is rejected by every S3-compatible backend.
///
/// **FAILS before fix**: the assertion `result.is_err()` holds (the error
/// exists), but `__exit__` in python_advanced_api.rs discards it and returns
/// `Ok(())`, so Python sees no exception.
/// **PASSES after fix**: `__exit__` propagates the error as `PyRuntimeError`.
///
/// Run with: `cargo test -- --include-ignored test_finish_err_on_zero_parts`
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_finish_err_on_zero_parts() -> Result<()> {
    dotenvy::dotenv().ok();
    let bucket = std::env::var("BUCKET").unwrap_or_else(|_| "mlp-s3dlio".to_string());
    let cfg = MultipartUploadConfig::default();

    let mut sink = MultipartUploadSink::new(&bucket, "test-bug593-zero-parts.bin", cfg)?;

    // finish_blocking() with no parts written → CompleteMultipartUpload(parts=[])
    // Every S3-compatible backend rejects an empty parts list.
    let result = sink.finish_blocking();
    assert!(
        result.is_err(),
        "finish_blocking() must fail when no parts were uploaded (S3 requires >= 1 part);\
         \nBUG #593 Bug 1: __exit__ currently discards this error — Python sees no exception"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// BUG #593 Bug 2 — no HEAD verification after CompleteMultipartUpload
// ---------------------------------------------------------------------------

/// Verifies that `MultipartCompleteInfo.stored_bytes` (populated by a HEAD
/// request after `CompleteMultipartUpload`) equals the bytes written, when
/// `S3DLIO_MPU_PUT_VERIFY=true` is set.  As of v0.9.106 this HEAD check is
/// opt-in (default off) — see `test_finish_skips_verification_by_default` for
/// the default-off behavior.
///
/// **FAILS TO COMPILE before fix**: `stored_bytes` does not exist on
/// `MultipartCompleteInfo` — the struct is defined in `src/multipart.rs`.
/// **PASSES after fix**: HEAD is issued, `stored_bytes` is populated, and
/// assertion confirms the stored object matches what was written.
///
/// Run with: `cargo test -- --include-ignored test_finish_verifies_stored_bytes`
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_finish_verifies_stored_bytes() -> Result<()> {
    dotenvy::dotenv().ok();
    std::env::set_var("S3DLIO_MPU_PUT_VERIFY", "true");
    let bucket = std::env::var("BUCKET").unwrap_or_else(|_| "mlp-s3dlio".to_string());
    let key = "test-bug593-head-verify.bin";
    let data_size: usize = 12 * 1024 * 1024; // 12 MiB — spans 2 parts at default 8 MiB

    let cfg = MultipartUploadConfig::default();
    let mut sink = MultipartUploadSink::new(&bucket, key, cfg)?;

    let block = vec![0xBEu8; 1024 * 1024]; // 1 MiB blocks
    for _ in 0..(data_size / block.len()) {
        sink.write_blocking(&block)?;
    }

    let info = sink.finish_blocking()?;

    ensure!(
        info.total_bytes == data_size as u64,
        "total_bytes mismatch: wrote {} but got {}",
        data_size,
        info.total_bytes
    );

    // BUG #593 Bug 2: stored_bytes field doesn't exist yet — compile error before fix.
    ensure!(
        info.stored_bytes == data_size as u64, // COMPILE ERROR before fix
        "HEAD-verified stored size mismatch: expected {} but S3 reports {}",
        data_size,
        info.stored_bytes
    );

    // Cleanup
    run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        let _ = client.delete_object().bucket(&bucket).key(key).send().await;
        Ok::<(), anyhow::Error>(())
    })?;

    std::env::remove_var("S3DLIO_MPU_PUT_VERIFY");

    Ok(())
}

// ---------------------------------------------------------------------------
// BUG #593 follow-up — verification is opt-in (default off) as of v0.9.106
// ---------------------------------------------------------------------------

/// Confirms that with `S3DLIO_MPU_PUT_VERIFY` unset (the default), no HEAD is
/// issued after `CompleteMultipartUpload` and `stored_bytes` is set equal to
/// `total_bytes` (unverified-assumed-equal) rather than independently
/// confirmed.  The object must still land correctly — verified here directly
/// by the test via its own HEAD call, independent of s3dlio's internal logic.
///
/// Run with: `cargo test -- --include-ignored test_finish_skips_verification_by_default`
#[test]
#[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
fn test_finish_skips_verification_by_default() -> Result<()> {
    dotenvy::dotenv().ok();
    std::env::remove_var("S3DLIO_MPU_PUT_VERIFY"); // explicit: exercise the default
    let bucket = std::env::var("BUCKET").unwrap_or_else(|_| "mlp-s3dlio".to_string());
    let key = "test-bug593-default-no-verify.bin";
    let data_size: usize = 12 * 1024 * 1024; // 12 MiB — spans 2 parts at default 8 MiB

    let cfg = MultipartUploadConfig::default();
    let mut sink = MultipartUploadSink::new(&bucket, key, cfg)?;

    let block = vec![0xEFu8; 1024 * 1024]; // 1 MiB blocks
    for _ in 0..(data_size / block.len()) {
        sink.write_blocking(&block)?;
    }

    let info = sink.finish_blocking()?;

    ensure!(
        info.total_bytes == data_size as u64,
        "total_bytes mismatch: wrote {} but got {}",
        data_size,
        info.total_bytes
    );

    // Verification was skipped — stored_bytes is set equal to total_bytes
    // (unverified-assumed-equal), not independently confirmed via HEAD.
    ensure!(
        info.stored_bytes == info.total_bytes,
        "stored_bytes must equal total_bytes when verification is disabled: {} vs {}",
        info.stored_bytes,
        info.total_bytes
    );

    // The test itself independently confirms the object landed correctly,
    // proving the upload succeeded even though s3dlio didn't verify it.
    run_on_global_rt({
        let bucket = bucket.clone();
        let key = key.to_string();
        async move {
            let client = aws_s3_client_async().await?;
            let head = client
                .head_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await?;
            let size = head.content_length().unwrap_or_default();
            ensure!(
                size == data_size as i64,
                "independent HEAD size mismatch: {} vs {}",
                size,
                data_size
            );
            let _ = client
                .delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    Ok(())
}

fn unique(prefix: &str) -> String {
    let pid = std::process::id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("s3dlio-{}-{}-{}", prefix, pid, now)
}

#[test]
#[ignore = "requires S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"]
fn multipart_upload_basic() -> Result<()> {
    let bucket = unique("mpu-basic");
    let key = "test-mpu.bin";

    // Create bucket
    run_on_global_rt({
        let bucket = bucket.clone();
        async move {
            let client = aws_s3_client_async().await?;
            client
                .create_bucket()
                .bucket(&bucket)
                .send()
                .await
                .context("create_bucket")?;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    // Upload ~65 MiB in 8 MiB writes
    let cfg = MultipartUploadConfig {
        part_size: 32 * 1024 * 1024,
        max_in_flight: Some(8),
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(&bucket, key, cfg)?;

    let total = (64 * 1024 * 1024) + (3 * 1024 * 1024);
    let block = 8 * 1024 * 1024;
    let pattern = vec![0xABu8; block];

    let mut remaining = total;
    while remaining >= block {
        sink.write_blocking(&pattern)?;
        remaining -= block;
    }
    if remaining > 0 {
        sink.write_blocking(&vec![0xAB; remaining])?;
    }

    let info = sink.finish_blocking()?;
    assert_eq!(info.total_bytes as usize, total, "total bytes mismatch");
    assert!(info.parts >= 2, "expected >=2 parts, got {}", info.parts);

    // HEAD to verify size, then cleanup
    run_on_global_rt({
        let bucket = bucket.clone();
        let key = key.to_string();
        async move {
            let client = aws_s3_client_async().await?;
            let head = client
                .head_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await?;
            let size = head.content_length().unwrap_or_default();
            ensure!(
                size == total as i64,
                "HEAD size mismatch: {} vs {}",
                size,
                total
            );
            let _ = client
                .delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await;
            let _ = client.delete_bucket().bucket(&bucket).send().await;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    Ok(())
}

#[test]
#[ignore = "requires S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"]
fn multipart_upload_abort() -> Result<()> {
    let bucket = unique("mpu-abort");
    let key = "abort.bin";

    // Create bucket
    run_on_global_rt({
        let bucket = bucket.clone();
        async move {
            let client = aws_s3_client_async().await?;
            client
                .create_bucket()
                .bucket(&bucket)
                .send()
                .await
                .context("create_bucket")?;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    // Start upload and then abort
    let cfg = MultipartUploadConfig {
        part_size: 16 * 1024 * 1024,
        max_in_flight: Some(4),
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(&bucket, key, cfg)?;
    sink.write_blocking(&vec![0xCD; 5 * 1024 * 1024])?;
    sink.flush_blocking()?; // force a part upload to start
    sink.abort_blocking()?; // abort the MPU

    // Object should not exist
    run_on_global_rt({
        let bucket = bucket.clone();
        let key = key.to_string();
        async move {
            let client = aws_s3_client_async().await?;
            let head = client.head_object().bucket(&bucket).key(&key).send().await;
            ensure!(head.is_err(), "object unexpectedly exists after abort");
            let _ = client.delete_bucket().bucket(&bucket).send().await;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    Ok(())
}
