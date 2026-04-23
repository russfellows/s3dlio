// src/multipart.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Architecture (v0.9.93+): coordinator task + bounded channel
// ─────────────────────────────────────────────────────────────
// Python write() → part_tx.blocking_send(PartMsg::Part(bytes))
//   • returns immediately if channel not full
//   • parks caller thread only when all max_in_flight slots are occupied
//     (natural backpressure, issue #134 contract preserved)
//
// [Coordinator task — runs entirely on Tokio runtime]
//   loop { recv part → semaphore.acquire().await → spawn UploadPart task }
//   → CompleteMultipartUpload → return MultipartCompleteInfo
//
// Python close() → blocking_send(PartMsg::Finish) → run_on_global_rt(coord.await)
//
// Result: zero run_on_global_rt() calls in the hot write path.
// All Tokio async work stays on the runtime; Python thread is never parked
// while slots are available.

use anyhow::{bail, Context, Result};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::Client;
use bytes::Bytes;

use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;

use crate::constants::{
    DEFAULT_MULTIPART_BUFFER_CAPACITY, DEFAULT_S3_MULTIPART_PART_SIZE, MIN_S3_MULTIPART_PART_SIZE,
};

use std::time::SystemTime;

use crate::s3_client::{aws_s3_client_async, run_on_global_rt, spawn_on_global_rt};
use crate::s3_utils::parse_s3_uri;

#[cfg(feature = "profiling")]
use tracing::instrument;

#[derive(Clone, Debug)]
pub struct MultipartUploadConfig {
    /// Target size of each part in bytes (AWS minimum is 5 MiB, typical 8–64 MiB).
    pub part_size: usize,
    /// Maximum number of concurrent in-flight part uploads.
    /// `None` → auto-computed from `part_size` via `auto_max_in_flight()`.
    /// Explicit values override the auto formula.
    pub max_in_flight: Option<usize>,
    /// Abort the MPU automatically if dropped unfinished.
    pub abort_on_drop: bool,
    /// Optional content-type to set on CreateMultipartUpload.
    pub content_type: Option<String>,
}

impl Default for MultipartUploadConfig {
    fn default() -> Self {
        Self {
            part_size: DEFAULT_S3_MULTIPART_PART_SIZE,
            max_in_flight: None, // auto
            abort_on_drop: true,
            content_type: None,
        }
    }
}

/// Compute the default `max_in_flight` from `part_size`.
///
/// Goal: ensure `max_in_flight` is large enough that a typical object (up to
/// ~512 MiB) can have all its parts queued without batching.
///
/// Formula: `max(32, ceil(512 MiB / part_size))`
///
/// Examples:
/// - part_size =  8 MiB → max(32, 64) = 64  (covers 512 MiB in one batch)
/// - part_size = 16 MiB → max(32, 32) = 32
/// - part_size = 32 MiB → max(32, 16) = 32
/// - part_size = 64 MiB → max(32,  8) = 32
///
/// Memory ceiling: `max_in_flight × part_size`
/// - @ 8 MiB:  64 × 8 MiB  = 512 MiB  (pipeline buffer)
/// - @ 16 MiB: 32 × 16 MiB = 512 MiB
/// - @ 32 MiB: 32 × 32 MiB = 1 GiB
///
/// This is the maximum bytes that can be buffered in the coordinator channel
/// + in-flight UploadPart tasks combined.
pub fn auto_max_in_flight(part_size: usize) -> usize {
    const TARGET_BYTES: usize = 512 * 1024 * 1024; // 512 MiB
    const FLOOR: usize = 32;
    let from_size = TARGET_BYTES.div_ceil(part_size);
    std::cmp::max(FLOOR, from_size)
}

/// Result info returned by finish()
#[derive(Clone, Debug)]
pub struct MultipartCompleteInfo {
    pub e_tag: Option<String>,
    pub total_bytes: u64,
    pub parts: usize,
    pub started_at: SystemTime,
    pub completed_at: SystemTime,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal message type sent from Python thread → coordinator task
// ─────────────────────────────────────────────────────────────────────────────

enum PartMsg {
    /// A complete part ready for upload. `part_number` is 1-based.
    Part { data: Bytes, part_number: i32 },
    /// Signal the coordinator that all parts have been sent; close the channel.
    Finish,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public sink struct
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming sink for multipart upload.
pub struct MultipartUploadSink {
    /// Channel to the coordinator task. Bounded by `max_in_flight`.
    part_tx: mpsc::Sender<PartMsg>,
    /// Coordinator task handle; awaited in finish_blocking().
    coordinator: Option<JoinHandle<Result<MultipartCompleteInfo>>>,

    // buffering (same as before — happens before channel send)
    buf: Vec<u8>,
    next_part_number: i32,
    total_bytes: u64,
    cfg: MultipartUploadConfig,
    /// Resolved max_in_flight (after applying auto formula).
    resolved_mif: usize,
    /// Upload ID for Drop abort.
    upload_id: String,
    client: aws_sdk_s3::Client,
    bucket: String,
    key: String,

    finished: bool,
}

impl Drop for MultipartUploadSink {
    fn drop(&mut self) {
        if self.finished || !self.cfg.abort_on_drop {
            return;
        }
        // Drop the sender — coordinator will see channel closed and can abort.
        // Also fire-and-forget an explicit AbortMultipartUpload for safety.
        let client     = self.client.clone();
        let bucket     = self.bucket.clone();
        let key        = self.key.clone();
        let upload_id  = self.upload_id.clone();
        let _ = run_on_global_rt(async move {
            let _ = client
                .abort_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(upload_id)
                .send()
                .await;
            Ok::<_, anyhow::Error>(())
        });
    }
}

impl MultipartUploadSink {
    /// Create a new MPU sink from an "s3://bucket/key" URI.
    pub fn from_uri(uri: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        let (b, k) = parse_s3_uri(uri)?;
        Self::new(&b, &k, cfg)
    }

    /// Create a new MPU sink for bucket + key.
    pub fn new(bucket: &str, key: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        let bucket = bucket.to_string();
        let key = key.to_string();
        run_on_global_rt(async move { Self::new_async(&bucket, &key, cfg).await })
    }

    /// Async constructor: issues CreateMultipartUpload and starts coordinator task.
    pub async fn new_async(bucket: &str, key: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        if cfg.part_size < MIN_S3_MULTIPART_PART_SIZE {
            bail!("part_size must be at least 5 MiB for S3 Multipart Upload");
        }
        if let Some(mif) = cfg.max_in_flight {
            if mif == 0 {
                bail!("max_in_flight must be >= 1");
            }
        }

        let resolved_mif = cfg.max_in_flight.unwrap_or_else(|| auto_max_in_flight(cfg.part_size));

        let client = aws_s3_client_async().await?;
        let mut req = client.create_multipart_upload().bucket(bucket).key(key);
        if let Some(ct) = &cfg.content_type {
            req = req.content_type(ct);
        }
        let resp = req.send().await.context("CreateMultipartUpload failed")?;
        let upload_id = resp.upload_id().unwrap_or_default().to_string();
        if upload_id.is_empty() {
            bail!("CreateMultipartUpload returned empty upload_id");
        }

        // Channel capacity = resolved_mif.  Python write() will block only
        // when this channel is full — meaning all resolved_mif slots are occupied.
        let (part_tx, part_rx) = mpsc::channel::<PartMsg>(resolved_mif);

        // Spawn the coordinator task on the global runtime.
        let coordinator = spawn_on_global_rt(coordinator_task(
            client.clone(),
            bucket.to_string(),
            key.to_string(),
            upload_id.clone(),
            resolved_mif,
            part_rx,
        ));

        Ok(Self {
            part_tx,
            coordinator: Some(coordinator),
            buf: Vec::with_capacity(DEFAULT_MULTIPART_BUFFER_CAPACITY),
            next_part_number: 1,
            total_bytes: 0,
            cfg,
            resolved_mif,
            upload_id,
            client,
            bucket: bucket.to_string(),
            key: key.to_string(),
            finished: false,
        })
    }

    /// Return the resolved `max_in_flight` (after auto formula, if applicable).
    pub fn max_in_flight(&self) -> usize {
        self.resolved_mif
    }

    // ── write paths ──────────────────────────────────────────────────────────

    /// Blocking write from a borrowed slice.
    pub fn write_blocking(&mut self, data: &[u8]) -> Result<()> {
        if self.buf.is_empty() && data.len() >= self.cfg.part_size {
            let mut offset = 0usize;
            while data.len() - offset >= self.cfg.part_size {
                let end = offset + self.cfg.part_size;
                let chunk = Bytes::copy_from_slice(&data[offset..end]);
                offset = end;
                self.enqueue_part(chunk)?;
            }
            if offset < data.len() {
                self.buf.extend_from_slice(&data[offset..]);
            }
            self.total_bytes += data.len() as u64;
            return Ok(());
        }

        self.buf.extend_from_slice(data);
        self.total_bytes += data.len() as u64;
        while self.buf.len() >= self.cfg.part_size {
            let chunk = Bytes::copy_from_slice(&self.buf[..self.cfg.part_size]);
            self.buf.drain(..self.cfg.part_size);
            self.enqueue_part(chunk)?;
        }
        Ok(())
    }

    /// Async write (kept for backwards compat with async callers).
    #[cfg_attr(feature = "profiling", instrument(
        name = "mpu.write",
        skip(self, data),
        fields(data_len = data.len())
    ))]
    pub async fn write(&mut self, data: Vec<u8>) -> Result<()> {
        // Delegate: async write just calls the blocking version here because
        // the channel send is synchronous from the Rust perspective.
        self.write_blocking(&data)
    }

    /// Zero-copy blocking write: accepts an owned Vec, converts to Bytes.
    pub fn write_owned_blocking(&mut self, data: Vec<u8>) -> Result<()> {
        let data_len = data.len() as u64;

        if self.buf.is_empty() && data.len() >= self.cfg.part_size {
            let bytes = Bytes::from(data);
            let mut offset = 0usize;
            while bytes.len() - offset >= self.cfg.part_size {
                let end = offset + self.cfg.part_size;
                let chunk = bytes.slice(offset..end); // zero-copy slice
                offset = end;
                self.enqueue_part(chunk)?;
            }
            if offset < bytes.len() {
                self.buf.extend_from_slice(&bytes[offset..]);
            }
        } else {
            self.buf.extend_from_slice(&data);
            while self.buf.len() >= self.cfg.part_size {
                let chunk = Bytes::copy_from_slice(&self.buf[..self.cfg.part_size]);
                self.buf.drain(..self.cfg.part_size);
                self.enqueue_part(chunk)?;
            }
        }

        self.total_bytes += data_len;
        Ok(())
    }

    /// Async version of `write_owned_blocking`.
    pub async fn write_owned(&mut self, data: Vec<u8>) -> Result<()> {
        self.write_owned_blocking(data)
    }

    /// Flush any buffered data as a (possibly short) part.
    pub fn flush_blocking(&mut self) -> Result<()> {
        if !self.buf.is_empty() {
            let chunk = Bytes::from(std::mem::take(&mut self.buf));
            self.enqueue_part(chunk)?;
        }
        Ok(())
    }

    pub async fn flush(&mut self) -> Result<()> {
        self.flush_blocking()
    }

    // ── finish / abort ───────────────────────────────────────────────────────

    pub fn finish_blocking(&mut self) -> Result<MultipartCompleteInfo> {
        if self.finished {
            bail!("finish() called more than once");
        }

        // Flush tail
        if !self.buf.is_empty() {
            let chunk = Bytes::from(std::mem::take(&mut self.buf));
            self.enqueue_part(chunk)?;
        }

        // Signal coordinator that all parts have been sent.
        // blocking_send won't block here: Finish is tiny and the channel has
        // capacity == resolved_mif; worst case we've already drained writes.
        self.part_tx
            .blocking_send(PartMsg::Finish)
            .map_err(|_| anyhow::anyhow!("coordinator task exited before Finish was sent"))?;

        // Await the coordinator result — one blocking hop at end of upload.
        let coordinator = self
            .coordinator
            .take()
            .ok_or_else(|| anyhow::anyhow!("coordinator already consumed"))?;

        let info = run_on_global_rt(async move {
            coordinator
                .await
                .map_err(|e| anyhow::anyhow!("coordinator task panicked: {e}"))?
        })?;

        self.finished = true;
        Ok(info)
    }

    pub fn abort_blocking(&mut self) -> Result<()> {
        // Drop the sender — coordinator sees channel closed and exits.
        // The Drop impl fires AbortMultipartUpload.
        self.finished = true; // prevent Drop from double-aborting
        let client    = self.client.clone();
        let bucket    = self.bucket.clone();
        let key       = self.key.clone();
        let upload_id = self.upload_id.clone();

        let _ = run_on_global_rt(async move {
            let _ = client
                .abort_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(upload_id)
                .send()
                .await;
            Ok::<_, anyhow::Error>(())
        });
        Ok(())
    }

    // ── internal ─────────────────────────────────────────────────────────────

    /// Enqueue a complete part for upload via the coordinator channel.
    ///
    /// `blocking_send` parks the caller (Python thread) only when the channel
    /// is full — i.e., `resolved_mif` parts are already queued or in-flight.
    /// This is the backpressure point: memory is bounded at
    /// `resolved_mif × part_size` bytes.
    fn enqueue_part(&mut self, data: Bytes) -> Result<()> {
        let part_number = self.next_part_number;
        self.next_part_number += 1;

        self.part_tx
            .blocking_send(PartMsg::Part { data, part_number })
            .map_err(|_| anyhow::anyhow!("coordinator task exited unexpectedly"))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinator task (runs entirely on Tokio runtime)
// ─────────────────────────────────────────────────────────────────────────────

/// Background task that owns all async S3 work for one multipart upload.
///
/// Receives parts from the bounded channel, acquires semaphore permits
/// asynchronously (never parking the Python caller thread), spawns
/// UploadPart tasks, collects ETags, and finally calls CompleteMultipartUpload.
async fn coordinator_task(
    client: Client,
    bucket: String,
    key: String,
    upload_id: String,
    max_in_flight: usize,
    mut part_rx: mpsc::Receiver<PartMsg>,
) -> Result<MultipartCompleteInfo> {
    let sem = std::sync::Arc::new(Semaphore::new(max_in_flight));
    let started_at = SystemTime::now();
    let mut total_bytes: u64 = 0;
    let mut part_tasks: Vec<JoinHandle<Result<(i32, String)>>> = Vec::new();

    loop {
        match part_rx.recv().await {
            Some(PartMsg::Part { data, part_number }) => {
                total_bytes += data.len() as u64;

                // Acquire a semaphore permit ASYNCHRONOUSLY.
                // This never parks the Python caller thread.
                let permit = sem
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| anyhow::anyhow!("semaphore closed"))?;

                let c  = client.clone();
                let b  = bucket.clone();
                let k  = key.clone();
                let uid = upload_id.clone();

                let handle = tokio::task::spawn(async move {
                    let _permit = permit; // released when task completes

                    let body = ByteStream::from(data);
                    let resp = c
                        .upload_part()
                        .bucket(b)
                        .key(k)
                        .upload_id(uid)
                        .part_number(part_number)
                        .body(body)
                        .send()
                        .await
                        .context("UploadPart failed")?;

                    let etag = resp.e_tag().unwrap_or_default().to_string();
                    if etag.is_empty() {
                        bail!("UploadPart returned empty ETag for part {part_number}");
                    }
                    Ok::<(i32, String), anyhow::Error>((part_number, etag))
                });

                part_tasks.push(handle);
            }
            Some(PartMsg::Finish) | None => {
                // All parts sent. Wait for all in-flight uploads.
                break;
            }
        }
    }

    // Join all part tasks.
    let mut parts: Vec<(i32, String)> = Vec::with_capacity(part_tasks.len());
    for handle in part_tasks {
        let (pn, etag) = handle
            .await
            .map_err(|e| anyhow::anyhow!("part task panicked: {e}"))??;
        parts.push((pn, etag));
    }
    parts.sort_by_key(|(pn, _)| *pn);

    let n_parts = parts.len();
    let completed_parts: Vec<CompletedPart> = parts
        .into_iter()
        .map(|(pn, etag)| {
            CompletedPart::builder()
                .set_e_tag(Some(etag))
                .set_part_number(Some(pn))
                .build()
        })
        .collect();

    let cmu = CompletedMultipartUpload::builder()
        .set_parts(Some(completed_parts))
        .build();

    let resp = client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(key)
        .upload_id(upload_id)
        .multipart_upload(cmu)
        .send()
        .await
        .context("CompleteMultipartUpload failed")?;

    Ok(MultipartCompleteInfo {
        e_tag: resp.e_tag.map(|s| s.to_string()),
        total_bytes,
        parts: n_parts,
        started_at,
        completed_at: SystemTime::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // MultipartUploadConfig validation (issue #134)
    // ------------------------------------------------------------------

    /// The part_size validation in new_async must reject sizes below 5 MiB.
    #[test]
    fn test_multipart_config_rejects_part_size_below_min() {
        let cfg = MultipartUploadConfig {
            part_size: 1024, // 1 KiB — well below the 5 MiB minimum
            ..Default::default()
        };
        let err = MultipartUploadSink::new("test-bucket", "test-key", cfg)
            .err()
            .expect("must fail with part_size < 5 MiB");
        assert!(
            err.to_string().contains("5 MiB"),
            "error must mention 5 MiB minimum, got: {}", err
        );
    }

    /// max_in_flight = Some(0) must be rejected before any S3 call.
    #[test]
    fn test_multipart_config_rejects_zero_max_in_flight() {
        let cfg = MultipartUploadConfig {
            max_in_flight: Some(0),
            ..Default::default()
        };
        let err = MultipartUploadSink::new("test-bucket", "test-key", cfg)
            .err()
            .expect("must fail with max_in_flight = 0");
        assert!(
            err.to_string().contains("max_in_flight"),
            "error must mention max_in_flight, got: {}", err
        );
    }

    /// from_uri must reject non-s3:// URIs before any network call.
    #[test]
    fn test_multipart_from_uri_rejects_invalid_uri() {
        let cfg = MultipartUploadConfig::default();
        let err = MultipartUploadSink::from_uri("not-a-valid-uri", cfg)
            .err()
            .expect("must fail for invalid URI");
        assert!(!err.to_string().is_empty(), "error message must not be empty");
    }

    /// Default MultipartUploadConfig has valid values.
    #[test]
    fn test_multipart_default_config_is_valid() {
        let cfg = MultipartUploadConfig::default();
        assert!(
            cfg.part_size >= MIN_S3_MULTIPART_PART_SIZE,
            "default part_size {} must be >= MIN {}", cfg.part_size, MIN_S3_MULTIPART_PART_SIZE
        );
        assert!(cfg.abort_on_drop, "default abort_on_drop should be true");
        // max_in_flight=None means auto
        assert!(cfg.max_in_flight.is_none(), "default max_in_flight should be None (auto)");
    }

    // ------------------------------------------------------------------
    // auto_max_in_flight formula
    // ------------------------------------------------------------------

    #[test]
    fn test_auto_max_in_flight_floor() {
        // Any part size large enough to produce a value below 32 should still give 32.
        assert_eq!(auto_max_in_flight(64 * 1024 * 1024), 32);  // 64 MiB → 8, floor=32
        assert_eq!(auto_max_in_flight(128 * 1024 * 1024), 32); // 128 MiB → 4, floor=32
    }

    #[test]
    fn test_auto_max_in_flight_scales_with_small_parts() {
        // Small parts produce larger max_in_flight.
        let mif_8mib  = auto_max_in_flight(8 * 1024 * 1024);  // 512/8 = 64
        let mif_16mib = auto_max_in_flight(16 * 1024 * 1024); // 512/16 = 32
        assert_eq!(mif_8mib,  64);
        assert_eq!(mif_16mib, 32);
    }

    #[test]
    fn test_auto_max_in_flight_covers_512mib_object() {
        // For any part size, auto_mif × part_size >= 512 MiB.
        for &ps in &[5, 8, 16, 32, 64, 100] {
            let ps_bytes = ps * 1024 * 1024;
            let mif = auto_max_in_flight(ps_bytes);
            let pipeline_bytes = mif * ps_bytes;
            assert!(
                pipeline_bytes >= 512 * 1024 * 1024,
                "part_size={ps}MiB: mif={mif}, pipeline={} MiB < 512 MiB",
                pipeline_bytes >> 20
            );
        }
    }
}

