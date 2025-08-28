// src/multipart.rs
//
// Streaming, concurrent Multipart Upload (MPU) for very large objects.
// Keeps nearly all MPU logic self-contained in this module.
//
// Design:
// - MultipartUploadConfig configures part sizing and concurrency.
// - MultipartUploadSink provides a streaming writer API:
//     - write(&[u8]) buffers until part_size and schedules concurrent UploadPart
//     - finish() uploads any tail and issues CompleteMultipartUpload
//     - abort() aborts the MPU; also runs on Drop if not finished and abort_on_drop
//
// Integrations:
// - Uses the global Tokio runtime and S3 client from s3_client.rs
// - Accepts "s3://bucket/key" via from_uri() using parse_s3_uri() from s3_utils.rs
//
// Notes:
// - All public APIs return anyhow::Result to match the crate style.
// - Logging hooks can be added later to feed s3_logger; kept minimal here.

use anyhow::{Context, Result, bail};
use aws_sdk_s3::Client;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::primitives::ByteStream;

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::constants::{DEFAULT_S3_MULTIPART_PART_SIZE, MIN_S3_MULTIPART_PART_SIZE, DEFAULT_MULTIPART_BUFFER_CAPACITY};
/*
use tokio::sync::{Semaphore, OwnedSemaphorePermit};
use tokio::task::JoinSet;
*/

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::s3_client::{aws_s3_client_async, run_on_global_rt};
use crate::s3_utils::parse_s3_uri;

#[cfg(feature = "profiling")]
use tracing::instrument;

#[derive(Clone, Debug)]
pub struct MultipartUploadConfig {
    /// Target size of each part in bytes (AWS minimum is 5 MiB, typical 8–64 MiB).
    pub part_size: usize,
    /// Maximum number of concurrent in-flight part uploads.
    pub max_in_flight: usize,
    /// Abort the MPU automatically if dropped unfinished.
    pub abort_on_drop: bool,
    /// Optional content-type to set on CreateMultipartUpload.
    pub content_type: Option<String>,
}

impl Default for MultipartUploadConfig {
    fn default() -> Self {
        Self {
            part_size: DEFAULT_S3_MULTIPART_PART_SIZE,
            max_in_flight: 16,
            abort_on_drop: true,
            content_type: None,
        }
    }
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


/// Streaming sink for multipart upload.
pub struct MultipartUploadSink {
    client: Client,
    bucket: String,
    key: String,
    upload_id: String,
    cfg: MultipartUploadConfig,

    // buffering
    buf: Vec<u8>,
    next_part_number: i32,
    total_bytes: u64,
    started_at: SystemTime,

    // concurrency
    sem: Arc<Semaphore>,
    tasks: Vec<JoinHandle<Result<(i32, String)>>>, // spawned on global runtime
    completed: Arc<Mutex<Vec<(i32, String)>>>,

    finished: bool,
}

impl Drop for MultipartUploadSink {
    fn drop(&mut self) {
        if self.finished || !self.cfg.abort_on_drop {
            return;
        }
        // Best effort abort; we can't async .await here. Fire-and-forget via runtime.
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
        let upload_id = self.upload_id.clone();
        let _ = run_on_global_rt(async move {
            let _ = client.abort_multipart_upload()
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

    /// Async constructor: issues CreateMultipartUpload.
    pub async fn new_async(bucket: &str, key: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        if cfg.part_size < MIN_S3_MULTIPART_PART_SIZE {
            bail!("part_size must be at least 5 MiB for S3 Multipart Upload");
        }
        if cfg.max_in_flight == 0 {
            bail!("max_in_flight must be >= 1");
        }
    
        let client = aws_s3_client_async().await?;
        let mut req = client.create_multipart_upload()
            .bucket(bucket)
            .key(key);
        if let Some(ct) = &cfg.content_type {
            req = req.content_type(ct);
        }
        let resp = req.send().await.context("CreateMultipartUpload failed")?;
        let upload_id = resp.upload_id().unwrap_or_default().to_string();
        if upload_id.is_empty() {
            bail!("CreateMultipartUpload returned empty upload_id");
        }
    
        // Build semaphore BEFORE moving cfg into the struct.
        let sem = Arc::new(Semaphore::new(cfg.max_in_flight));
    
        Ok(Self {
            client,
            bucket: bucket.to_string(),
            key: key.to_string(),
            upload_id,
            cfg, // moved exactly once here
            buf: Vec::with_capacity(DEFAULT_MULTIPART_BUFFER_CAPACITY),
            next_part_number: 1,
            total_bytes: 0,
            started_at: SystemTime::now(),
            sem,
            tasks: Vec::new(),
            completed: Arc::new(Mutex::new(Vec::new())),
            finished: false,
        })
    }

    /// Blocking write from a borrowed slice (no async needed), but do we want run_on_global_rt ?
    pub fn write_blocking(&mut self, data: &[u8]) -> Result<()> {
        // Fast path: if internal buffer is empty and data is already large,
        // cut directly into full-size parts to avoid extra buffering.
        if self.buf.is_empty() && data.len() >= self.cfg.part_size {
            let mut offset = 0usize;
            while data.len() - offset >= self.cfg.part_size {
                let end = offset + self.cfg.part_size;
                let chunk = data[offset..end].to_vec(); // own the part buffer
                offset = end;
                self.spawn_part(chunk)?;
            }
            if offset < data.len() {
                self.buf.extend_from_slice(&data[offset..]);
            }
            self.total_bytes += data.len() as u64;
            return Ok(());
        }

        // Otherwise merge into internal buffer and emit parts as thresholds are crossed.
        self.buf.extend_from_slice(data);
        self.total_bytes += data.len() as u64;
        while self.buf.len() >= self.cfg.part_size {
            let chunk = self.buf.drain(..self.cfg.part_size).collect::<Vec<u8>>();
            self.spawn_part(chunk)?;
        }
        Ok(())
    }


    /// Async write: buffers and schedules part uploads as needed.
    #[cfg_attr(feature = "profiling", instrument(
        name = "mpu.write",
        skip(self, data),
        fields(
            data_len = data.len(),
            buf_len = self.buf.len(),
            part_size = self.cfg.part_size,
            total_bytes = self.total_bytes
        )
    ))]
    pub async fn write(&mut self, data: Vec<u8>) -> Result<()> {
        self.total_bytes += data.len() as u64;

        if self.buf.is_empty() && data.len() >= self.cfg.part_size {
            // Fast path: directly slice full-sized parts out of data to avoid extra copies.
            let mut offset = 0usize;
            while data.len() - offset >= self.cfg.part_size {
                let end = offset + self.cfg.part_size;
                let chunk = data[offset..end].to_vec();
                offset = end;
                self.spawn_part(chunk)?;
            }
            // Remainder goes into internal buffer.
            if offset < data.len() {
                self.buf.extend_from_slice(&data[offset..]);
            }
        } else {
            // Slow path: fill buffer until it reaches part_size
            self.buf.extend_from_slice(&data);
            while self.buf.len() >= self.cfg.part_size {
                let chunk = self.buf.drain(..self.cfg.part_size).collect::<Vec<u8>>();
                self.spawn_part(chunk)?;
            }
        }
        Ok(())
    }

    /// Zero-copy path: accept an *owned* buffer and stream without copying.
    /// This is equivalent to `write()` but consumes `data` directly.
    /// Zero-copy within Rust: accept an owned Vec from Python and stream without another copy,
    /// what about run_on_global_rt() ?
    pub fn write_owned_blocking(&mut self, data: Vec<u8>) -> Result<()> {
        if self.buf.is_empty() && data.len() >= self.cfg.part_size {
            let mut offset = 0usize;
            while data.len() - offset >= self.cfg.part_size {
                let end = offset + self.cfg.part_size;
                let chunk = data[offset..end].to_vec(); // own each part buffer
                offset = end;
                self.spawn_part(chunk)?;
            }
            if offset < data.len() {
                self.buf.extend_from_slice(&data[offset..]);
            }
        } else {
            self.buf.extend_from_slice(&data);
            while self.buf.len() >= self.cfg.part_size {
                let chunk = self.buf.drain(..self.cfg.part_size).collect::<Vec<u8>>();
                self.spawn_part(chunk)?;
            }
        }
    
        self.total_bytes += data.len() as u64;
        Ok(())
    }


    /// Async version of `write_owned_blocking`.
    pub async fn write_owned(&mut self, data: Vec<u8>) -> Result<()> {
        self.total_bytes += data.len() as u64;

        if self.buf.is_empty() && data.len() >= self.cfg.part_size {
            // Fast path: slice full-sized parts out of `data` without extra copies.
            let mut offset = 0usize;
            while data.len() - offset >= self.cfg.part_size {
                let end = offset + self.cfg.part_size;
                // Move out chunks by allocating a new Vec and copying the slice.
                // (No extra Python->Rust copy; the only copy is within Rust when chunking `data`.)
                let chunk = data[offset..end].to_vec();
                offset = end;
                self.spawn_part(chunk)?;
            }
            if offset < data.len() {
                // Move the tail into internal buffer.
                self.buf.extend_from_slice(&data[offset..]);
            }
        } else {
            // Merge into internal buffer; will spawn parts as it crosses thresholds.
            self.buf.extend_from_slice(&data);
            while self.buf.len() >= self.cfg.part_size {
                let chunk = self.buf.drain(..self.cfg.part_size).collect::<Vec<u8>>();
                self.spawn_part(chunk)?;
            }
        }
        Ok(())
    }

    /// Flush any buffered data as a (possibly short) part without finishing the MPU.
    pub fn flush_blocking(&mut self) -> Result<()> {
        if !self.buf.is_empty() {
            let chunk = std::mem::take(&mut self.buf);
            self.spawn_part(chunk)?;
        }
        Ok(())
    }


    pub async fn flush(&mut self) -> Result<()> {
        if !self.buf.is_empty() {
            let chunk = std::mem::take(&mut self.buf);
            self.spawn_part(chunk)?;
        }
        Ok(())
    }


    pub fn finish_blocking(&mut self) -> Result<MultipartCompleteInfo> {
        if self.finished {
            bail!("finish() called more than once");
        }
        // Upload any tail synchronously
        if !self.buf.is_empty() {
            let chunk = std::mem::take(&mut self.buf);
            self.spawn_part(chunk)?;
        }
    
        // Move the JoinHandles out so the future owns them (no borrow of self).
        let tasks = std::mem::take(&mut self.tasks);
    
        // Await all part tasks on the global runtime
        let results: Vec<(i32, String)> = run_on_global_rt(async move {
            let mut out = Vec::with_capacity(tasks.len());
            for h in tasks {
                // First await the join handle, then unwrap the inner Result
                let res = h.await.context("part task join failed")??;
                out.push(res);
            }
            Ok::<_, anyhow::Error>(out)
        })?;
    
        // Record completed parts
        {
            let mut guard = self.completed.lock().unwrap();
            guard.extend(results);
        }
    
        // Build CompletedMultipartUpload
        let mut parts = self.completed.lock().unwrap().clone();
        parts.sort_by_key(|(pn, _)| *pn);
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
    
        // Clone small fields; call CompleteMultipartUpload on the runtime.
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
        let upload_id = self.upload_id.clone();
    
        let resp = run_on_global_rt(async move {
            client
                .complete_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(upload_id)
                .multipart_upload(cmu)
                .send()
                .await
                .context("CompleteMultipartUpload failed")
        })?;
    
        self.finished = true;
    
        Ok(MultipartCompleteInfo {
            e_tag: resp.e_tag.map(|s| s.to_string()),
            total_bytes: self.total_bytes,
            parts: self.next_part_number.saturating_sub(1) as usize,
            started_at: self.started_at,
            completed_at: SystemTime::now(),
        })
    }


    pub fn abort_blocking(&mut self) -> Result<()> {
        // Move JoinHandles out so the future owns them
        let tasks = std::mem::take(&mut self.tasks);
    
        // Best-effort: join and ignore results
        let _ = run_on_global_rt(async move {
            for h in tasks {
                let _ = h.await;
            }
            Ok::<_, anyhow::Error>(())
        });
    
        // Abort MPU
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
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
    
        self.finished = true;
        Ok(())
    }



    fn spawn_part(&mut self, bytes: Vec<u8>) -> Result<()> {
        let part_number = self.next_part_number;
        self.next_part_number += 1;
    
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
        let upload_id = self.upload_id.clone();
        let semaphore = self.sem.clone();
    
        // Spawn the task *on the global Tokio runtime* and capture the JoinHandle.
        let handle: JoinHandle<Result<(i32, String)>> = run_on_global_rt(async move {
            // Spawn a child task on the same runtime; return its JoinHandle
            let h = tokio::spawn(async move {
                // Concurrency permit
                let _permit = semaphore.acquire_owned().await.expect("semaphore closed");
    
                let body = ByteStream::from(bytes);
                let resp = client
                    .upload_part()
                    .bucket(bucket)
                    .key(key)
                    .upload_id(upload_id)
                    .part_number(part_number)
                    .body(body)
                    .send()
                    .await
                    .context("UploadPart failed")?;
    
                let etag = resp.e_tag().unwrap_or_default().to_string();
                if etag.is_empty() {
                    bail!("UploadPart returned empty ETag");
                }
                Ok::<(i32, String), anyhow::Error>((part_number, etag))
            });
            Ok::<_, anyhow::Error>(h)
        })?;
    
        // Store handle to await later in finish_blocking()
        self.tasks.push(handle);
        Ok(())
    }
    
    
}
    
    
    
