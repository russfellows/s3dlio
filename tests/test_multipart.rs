// tests/test_multipart.rs
use anyhow::{Context, Result, ensure};
use std::time::{SystemTime, UNIX_EPOCH};

use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
use s3dlio::s3_client::{aws_s3_client_async, run_on_global_rt};

fn unique(prefix: &str) -> String {
    let pid = std::process::id();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("s3dlio-{}-{}-{}", prefix, pid, now)
}

#[test]
fn multipart_upload_basic() -> Result<()> {
    let bucket = unique("mpu-basic");
    let key = "test-mpu.bin";

    // Create bucket
    run_on_global_rt({
        let bucket = bucket.clone();
        async move {
            let client = aws_s3_client_async().await?;
            client.create_bucket().bucket(&bucket).send().await
                .context("create_bucket")?;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    // Upload ~65 MiB in 8 MiB writes
    let cfg = MultipartUploadConfig { part_size: 32 * 1024 * 1024, max_in_flight: 8, ..Default::default() };
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
            let head = client.head_object().bucket(&bucket).key(&key).send().await?;
            let size = head.content_length().unwrap_or_default();
            ensure!(size == total as i64, "HEAD size mismatch: {} vs {}", size, total);
            let _ = client.delete_object().bucket(&bucket).key(&key).send().await;
            let _ = client.delete_bucket().bucket(&bucket).send().await;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    Ok(())
}

#[test]
fn multipart_upload_abort() -> Result<()> {
    let bucket = unique("mpu-abort");
    let key = "abort.bin";

    // Create bucket
    run_on_global_rt({
        let bucket = bucket.clone();
        async move {
            let client = aws_s3_client_async().await?;
            client.create_bucket().bucket(&bucket).send().await
                .context("create_bucket")?;
            Ok::<(), anyhow::Error>(())
        }
    })?;

    // Start upload and then abort
    let cfg = MultipartUploadConfig { part_size: 16 * 1024 * 1024, max_in_flight: 4, ..Default::default() };
    let mut sink = MultipartUploadSink::new(&bucket, key, cfg)?;
    sink.write_blocking(&vec![0xCD; 5 * 1024 * 1024])?;
    sink.flush_blocking()?;     // force a part upload to start
    sink.abort_blocking()?;     // abort the MPU

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

