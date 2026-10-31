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

use anyhow::{bail, Result};
use aws_sdk_s3::primitives::{ByteStream, SdkBody};
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::Client;
use bytes::{Bytes, BytesMut};
use std::collections::VecDeque;

use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::constants::{
    DEFAULT_S3_MULTIPART_PART_SIZE, MAX_MULTIPART_PARTS, MIN_S3_MULTIPART_PART_SIZE,
};

use std::time::SystemTime;

use crate::delete_batch_plan::parse_s3_uri_for_global_client;
use crate::s3_client::{aws_s3_client_async, run_on_global_rt, spawn_on_global_rt};
use crate::s3_utils::SdkResultExt;

use tracing::{debug, warn};

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
/// Memory ceiling: `~2 × max_in_flight × part_size`
/// (both the channel and the semaphore are sized `max_in_flight`; in the worst
/// case all channel slots are full AND all semaphore slots are in-flight)
/// - @ 8 MiB:  2 × 64 × 8 MiB  ≈ 1 GiB
/// - @ 16 MiB: 2 × 32 × 16 MiB ≈ 1 GiB
/// - @ 32 MiB: 2 × 32 × 32 MiB ≈ 2 GiB
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
    /// Bytes written by the caller (sum of all parts).
    pub total_bytes: u64,
    /// Bytes confirmed stored by HEAD request after CompleteMultipartUpload —
    /// only when `S3DLIO_MPU_PUT_VERIFY=true` (default off; see coordinator_task).
    /// When verification is disabled this is set equal to `total_bytes` (unverified).
    /// When verification is enabled, a mismatch means the backend silently dropped data.
    pub stored_bytes: u64,
    pub parts: usize,
    pub started_at: SystemTime,
    pub completed_at: SystemTime,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal message type sent from Python thread → coordinator task
// ─────────────────────────────────────────────────────────────────────────────

/// Writes smaller than the threshold are COPIED ONCE into the part buffer instead of being
/// queued as refcounted slices (and, from Python, instead of holding a buffer export per write).
/// The default ("auto") threshold is `part_size` for a SIGNED upload and
/// [`UNSIGNED_COALESCE_BELOW`] (2 KiB) for an unsigned one.
///
/// Rule #1 allows a single bounded copy when it is MEASURED faster. SIGNED: borrowing a write
/// smaller than a part does not avoid the copy, because a part is made of several writes and
/// must be gathered into one body anyway. Benchmark (`python/tests/test_zero_copy_multipart.py`,
/// `S3DLIO_ZC_BENCH=1`; 16 MiB parts, 128 MiB per case, best of 3, write loop + `close()`,
/// borrow/copy time): 1.07-1.25 at 1 KiB and below (copy faster), 0.87-1.04 from 2 KiB up
/// (typically ~0.93: a gain of at most ~7%, about the run-to-run noise). The gain is not worth
/// the lock and the pinned buffers, so signed uploads keep the part size.
pub(crate) const COALESCE_AUTO: usize = usize::MAX;

/// The auto threshold of an UNSIGNED upload (`S3DLIO_UNSIGNED_PAYLOAD`): 2 KiB. Unsigned parts
/// that span writes are sent as multi-chunk bodies with no gather copy (the connector streams
/// the segments), so borrowing a sub-part write IS zero-copy end to end. Measured (two runs, best
/// of 3, 16 MiB parts, 128 MiB per case, write loop + `close()`; borrow/copy time): 0.72-0.93
/// from 2 KiB to 32 KiB (faster), 0.94-1.10 from 64 KiB up (parity), 1.05-1.19 at 1 KiB and
/// below (slower: per-write bookkeeping dominates). 2 KiB is the lowest size where borrowing was
/// not slower in both runs. CAVEATS: a multi-chunk body is a streaming body, so reqwest's own
/// redirect policy cannot replay it on a 307/308 (`S3DLIO_FOLLOW_REDIRECTS` still works); and the
/// 2 KiB was measured over plain HTTP/1.1 - TLS or h2 may favour larger writes.
pub(crate) const UNSIGNED_COALESCE_BELOW: usize = 2 * 1024;

static COALESCE_BELOW: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(COALESCE_AUTO);

/// The configured small-write threshold; [`COALESCE_AUTO`] means "the part size".
pub(crate) fn coalesce_below() -> usize {
    COALESCE_BELOW.load(std::sync::atomic::Ordering::Relaxed)
}

/// The threshold a sink with this part size uses (see [`COALESCE_AUTO`]).
pub(crate) fn resolve_coalesce_below(part_size: usize, unsigned: bool) -> usize {
    match coalesce_below() {
        COALESCE_AUTO if unsigned => UNSIGNED_COALESCE_BELOW,
        COALESCE_AUTO => part_size,
        explicit => explicit,
    }
}

/// Set the threshold for sinks created AFTER this call (`COALESCE_AUTO` restores the default);
/// returns the previous value. Test and benchmark hook (hidden Python function
/// `s3dlio._mpu_coalesce_below`).
#[cfg(feature = "extension-module")]
pub(crate) fn set_coalesce_below(bytes: usize) -> usize {
    COALESCE_BELOW.swap(bytes, std::sync::atomic::Ordering::Relaxed)
}

/// Count copied payload bytes in test builds only (proves the copy count in unit tests); expands
/// to nothing in release code.
macro_rules! note_copy {
    ($sink:expr, $n:expr) => {
        #[cfg(test)]
        {
            $sink.copied += $n;
        }
    };
}

/// One part ready for upload: a single contiguous `Bytes`, or - only for an UNSIGNED upload -
/// the refcounted slices of a part that straddles several writes, sent as one multi-chunk body.
struct PartData {
    chunks: Vec<Bytes>,
    len: usize,
}

impl PartData {
    fn new(chunks: Vec<Bytes>) -> Self {
        let len = chunks.iter().map(Bytes::len).sum();
        Self { chunks, len }
    }

    fn single(part: Bytes) -> Self {
        let len = part.len();
        Self {
            chunks: vec![part],
            len,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// The request body. One chunk is the ordinary in-memory body (`ByteStream::from(Bytes)`:
    /// the redirect client's `try_clone()` succeeds, so AIStore 307 redirects on UploadPart are
    /// followed). Several chunks are a body of KNOWN length (so `Content-Length` is sent, not
    /// chunked encoding) that yields the slices as they are; it is retryable and clonable, each
    /// attempt rebuilding the body from the same refcounted slices.
    fn into_byte_stream(mut self) -> ByteStream {
        if self.chunks.len() == 1 {
            if let Some(only) = self.chunks.pop() {
                return ByteStream::from(only);
            }
        }
        let chunks = self.chunks;
        let len = self.len as u64;
        ByteStream::new(SdkBody::retryable(move || {
            SdkBody::from_body_1_x(ChunkedBody {
                chunks: chunks.iter().cloned().collect(),
                remaining: len,
            })
        }))
    }
}

/// An HTTP body that yields a list of `Bytes` slices without copying them and reports its exact
/// length.
struct ChunkedBody {
    chunks: VecDeque<Bytes>,
    remaining: u64,
}

impl http_body::Body for ChunkedBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<std::result::Result<http_body::Frame<Bytes>, Self::Error>>> {
        match self.chunks.pop_front() {
            Some(chunk) => {
                self.remaining = self.remaining.saturating_sub(chunk.len() as u64);
                std::task::Poll::Ready(Some(Ok(http_body::Frame::data(chunk))))
            }
            None => std::task::Poll::Ready(None),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.chunks.is_empty()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::with_exact(self.remaining)
    }
}

enum PartMsg {
    /// A complete part ready for upload. `part_number` is 1-based.
    Part { data: PartData, part_number: i32 },
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
    /// Signaled on Drop / abort() to stop the coordinator BEFORE it can
    /// reach CompleteMultipartUpload. See `cancel_coordinator`
    /// (audit #151 findings f11/f12 — issue #148 CLAUDE.md bug class:
    /// dropping `part_tx` alone is not enough, because dropping a
    /// `JoinHandle` detaches rather than cancels the task it points to).
    cancel: CancellationToken,

    // buffering (same as before — happens before channel send)
    /// Owned input not yet cut into parts: refcounted slices of the callers' data, in order.
    pending: VecDeque<Bytes>,
    /// The part buffer being filled by BORROWED input (copied once into it); it logically follows
    /// `pending`.
    open: BytesMut,
    /// Total bytes in `pending` and `open` (always < `part_size` between writes).
    pending_len: usize,
    /// Payload signing is off (`S3DLIO_UNSIGNED_PAYLOAD`): parts that straddle writes are sent as
    /// multi-chunk bodies instead of being gathered, and every part is sent unsigned.
    unsigned: bool,
    /// Owned writes smaller than this are copied once into the part buffer (see
    /// [`COALESCE_AUTO`]); `0` disables the coalescing.
    coalesce_below: usize,
    /// Test builds: payload bytes copied by the part assembly.
    #[cfg(test)]
    copied: usize,
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
        if self.finished {
            return;
        }
        // abort_on_drop gates only whether the best-effort S3-side Abort
        // fires; the coordinator is ALWAYS cancelled so it can never
        // reach CompleteMultipartUpload for a sink that was dropped
        // without finish() (audit #151 f12 — locked contract, see
        // docs/implementation-plans/v0.9.109-audit-fix-plan.md bug A2).
        //
        // The Abort is SPAWNED on the global runtime and Drop returns at once (#303): a Drop
        // that waited for it blocked the dropping thread, and on a current-thread runtime (the
        // only thread that could drive the connection) it stalled for the whole operation
        // timeout and leaked the upload. The outcome is logged; if the process exits before the
        // request is delivered the upload may remain on the server, so call `abort().await`
        // (or `abort_blocking()`) when delivery must be certain.
        if let Some(req) = self.cancel_coordinator(self.cfg.abort_on_drop) {
            spawn_on_global_rt(async move {
                let what = req.describe();
                match req.send().await {
                    Ok(()) => debug!("s3dlio MPU: dropped sink aborted {what}"),
                    Err(e) => warn_abort_failed(&what, &e),
                }
            });
        }
    }
}

fn warn_abort_failed(what: &str, e: &anyhow::Error) {
    warn!(
        "s3dlio MPU: best-effort AbortMultipartUpload failed for {what}: {e} — \
         the in-progress upload may need manual cleanup"
    );
}

/// One best-effort `AbortMultipartUpload`, detached from the sink so it can run anywhere.
struct AbortRequest {
    client: aws_sdk_s3::Client,
    bucket: String,
    key: String,
    upload_id: String,
}

impl AbortRequest {
    fn describe(&self) -> String {
        format!(
            "s3://{}/{} upload_id={}",
            self.bucket, self.key, self.upload_id
        )
    }

    async fn send(&self) -> Result<()> {
        self.client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&self.upload_id)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("AbortMultipartUpload failed: {e}"))
    }
}

impl MultipartUploadSink {
    /// Create a new MPU sink from an "s3://bucket/key" URI.
    pub fn from_uri(uri: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        let (b, k) = parse_s3_uri_for_global_client(uri)?;
        Self::new(&b, &k, cfg)
    }

    /// Create a new MPU sink for bucket + key.
    pub fn new(bucket: &str, key: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        let bucket = bucket.to_string();
        let key = key.to_string();
        run_on_global_rt(async move { Self::new_async(&bucket, &key, cfg).await })
    }

    /// Create a new MPU sink for bucket + key on a caller-chosen client
    /// (issue #208): a per-endpoint store passes ITS client.
    pub(crate) fn new_with_client(
        client: Client,
        bucket: &str,
        key: &str,
        cfg: MultipartUploadConfig,
    ) -> Result<Self> {
        let bucket = bucket.to_string();
        let key = key.to_string();
        run_on_global_rt(
            async move { Self::new_async_with_client(client, &bucket, &key, cfg).await },
        )
    }

    /// Async constructor: issues CreateMultipartUpload and starts coordinator task.
    pub async fn new_async(bucket: &str, key: &str, cfg: MultipartUploadConfig) -> Result<Self> {
        // Reject a bad config before anything needs credentials.
        Self::validate_config(&cfg)?;
        let client = aws_s3_client_async().await?;
        Self::new_async_with_client(client, bucket, key, cfg).await
    }

    fn validate_config(cfg: &MultipartUploadConfig) -> Result<()> {
        if cfg.part_size < MIN_S3_MULTIPART_PART_SIZE {
            bail!("part_size must be at least 5 MiB for S3 Multipart Upload");
        }
        if let Some(mif) = cfg.max_in_flight {
            if mif == 0 {
                bail!("max_in_flight must be >= 1");
            }
        }
        Ok(())
    }

    /// [`MultipartUploadSink::new_async`] on a caller-chosen client.
    ///
    /// The sink keeps `client` for the whole upload: CreateMultipartUpload, every
    /// UploadPart, CompleteMultipartUpload and AbortMultipartUpload (on drop or
    /// `abort_blocking`) all use it, so an abort always reaches the server that
    /// holds the upload.
    pub(crate) async fn new_async_with_client(
        client: Client,
        bucket: &str,
        key: &str,
        cfg: MultipartUploadConfig,
    ) -> Result<Self> {
        Self::validate_config(&cfg)?;

        let resolved_mif = cfg
            .max_in_flight
            .unwrap_or_else(|| auto_max_in_flight(cfg.part_size));

        let mut req = client.create_multipart_upload().bucket(bucket).key(key);
        if let Some(ct) = &cfg.content_type {
            req = req.content_type(ct);
        }
        let resp = req.send().await.sdk_context(format!(
            "CreateMultipartUpload failed for s3://{}/{}",
            bucket, key
        ))?;
        let upload_id = resp.upload_id().unwrap_or_default().to_string();
        if upload_id.is_empty() {
            bail!("CreateMultipartUpload returned empty upload_id");
        }

        // Channel capacity = resolved_mif.  Python write() will block only
        // when this channel is full — meaning all resolved_mif slots are occupied.
        let (part_tx, part_rx) = mpsc::channel::<PartMsg>(resolved_mif);

        // Cancellation token: Drop / abort() signal this to stop the
        // coordinator BEFORE it can reach CompleteMultipartUpload
        // (audit #151 f11/f12). Cloned into the coordinator task below.
        let cancel = CancellationToken::new();

        // Latched for the process: payload signing off => unsigned UploadPart bodies, multi-chunk
        // straddling parts (no gather copy).
        let unsigned = crate::s3_client::unsigned_payload_enabled();
        // Spawn the coordinator task on the global runtime.
        let coordinator = spawn_on_global_rt(coordinator_task(
            client.clone(),
            UploadTarget {
                bucket: bucket.to_string(),
                key: key.to_string(),
                upload_id: upload_id.clone(),
            },
            resolved_mif,
            part_rx,
            cancel.clone(),
            unsigned,
        ));

        Ok(Self {
            part_tx,
            coordinator: Some(coordinator),
            cancel,
            pending: VecDeque::new(),
            open: BytesMut::new(),
            pending_len: 0,
            unsigned,
            coalesce_below: resolve_coalesce_below(cfg.part_size, unsigned),
            #[cfg(test)]
            copied: 0,
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

    /// Writes smaller than this are copied once into the part buffer (`0`: never).
    #[cfg(feature = "extension-module")]
    pub(crate) fn coalesce_below(&self) -> usize {
        self.coalesce_below
    }

    /// Return the resolved `max_in_flight` (after auto formula, if applicable).
    pub fn max_in_flight(&self) -> usize {
        self.resolved_mif
    }

    // ── write paths ──────────────────────────────────────────────────────────
    //
    // ZERO-COPY PART ASSEMBLY (Rule #1, stage G W3).
    //
    // The stream is held as `pending` (a queue of refcounted slices of OWNED input) followed by
    // `open` (a part buffer being filled by BORROWED input). `pending_len` counts both and is
    // below `part_size` between writes.
    //
    //   * Owned `Bytes` (or a `Vec`, which becomes `Bytes` without a copy) is queued as slices
    //     and never copied. A part that lies inside one slice is a zero-copy `split_to`.
    //   * A part that STRADDLES slices (the tail of one write plus the head of the next) must
    //     be one contiguous request body for a SIGNED upload, so it is gathered with ONE copy
    //     of at most `part_size` bytes (the "non-contiguous source that must be gathered"
    //     exception). When payload signing is off (`S3DLIO_UNSIGNED_PAYLOAD`, see
    //     `unsigned_payload_enabled`) it is NOT gathered: the part is handed to the uploader as
    //     the list of slices and sent as a multi-chunk body of known length, zero copies.
    //   * Borrowed `&[u8]` input cannot outlive the call, so it is copied once, straight into
    //     the part buffer that will be uploaded (`open`, or a fresh part buffer when the input
    //     holds whole parts). A borrowed byte is therefore copied exactly once, whatever the
    //     tail state. (Signed mode only: owned slices queued before a borrowed write are
    //     gathered once into `open` first, so every byte still has at most one copy.)

    /// Test-only copy accounting: the number of payload bytes the assembly has copied.
    #[cfg(test)]
    fn copied_bytes(&self) -> usize {
        self.copied
    }

    /// Move the open (borrowed-input) buffer behind the queued slices, as one more slice.
    fn seal_open(&mut self) {
        if !self.open.is_empty() {
            let sealed = self.open.split().freeze();
            self.pending.push_back(sealed);
        }
    }

    /// Queue owned `Bytes` for part assembly. Signed mode only: when an open part buffer is
    /// filling, the head of `data` that completes it is copied into it (the bounded gather).
    fn push_owned(&mut self, mut data: Bytes) {
        if data.is_empty() {
            return;
        }
        self.total_bytes += data.len() as u64;
        if self.unsigned {
            // No copy ever: the open buffer becomes one more slice, `data` another.
            self.seal_open();
            self.pending_len += data.len();
            self.pending.push_back(data);
            return;
        }
        if !self.open.is_empty() {
            let need = self.cfg.part_size.saturating_sub(self.pending_len);
            let take = need.min(data.len());
            // Gather copy: bounded by part_size, completes the part buffer being filled.
            self.open.extend_from_slice(&data[..take]);
            note_copy!(self, take);
            self.pending_len += take;
            data = data.slice(take..);
            if data.is_empty() {
                return;
            }
            // `open` is now a full part: it goes first, ahead of the rest of `data`.
            self.seal_open();
        }
        self.pending_len += data.len();
        self.pending.push_back(data);
    }

    /// Cut the next complete part (`part_size` bytes) off the stream, if there is one.
    fn next_part(&mut self) -> Result<Option<PartData>> {
        let part_size = self.cfg.part_size;
        if self.pending_len < part_size {
            return Ok(None);
        }
        match self.pending.front_mut() {
            Some(front) if front.len() >= part_size => {
                let part = front.split_to(part_size);
                if front.is_empty() {
                    self.pending.pop_front();
                }
                self.pending_len -= part_size;
                return Ok(Some(PartData::single(part)));
            }
            None => {
                if self.open.len() < part_size {
                    bail!("internal error: pending length exceeds the buffered bytes");
                }
                let part = self.open.split_to(part_size).freeze();
                self.pending_len -= part_size;
                return Ok(Some(PartData::single(part)));
            }
            Some(_) => {}
        }
        // A part that straddles slices.
        let mut chunks: Vec<Bytes> = Vec::new();
        let mut left = part_size;
        while left > 0 {
            match self.pending.front_mut() {
                Some(front) if front.len() <= left => {
                    left -= front.len();
                    if let Some(whole) = self.pending.pop_front() {
                        chunks.push(whole);
                    }
                }
                Some(front) => {
                    chunks.push(front.split_to(left));
                    left = 0;
                }
                None => {
                    if self.open.len() < left {
                        bail!("internal error: pending length exceeds the buffered bytes");
                    }
                    chunks.push(self.open.split_to(left).freeze());
                    left = 0;
                }
            }
        }
        self.pending_len -= part_size;
        if self.unsigned || chunks.len() == 1 {
            // Unsigned: sent as a multi-chunk body, no copy.
            return Ok(Some(PartData::new(chunks)));
        }
        // Signed: one contiguous body is required. Gather copy of exactly `part_size` bytes.
        let mut gathered = BytesMut::with_capacity(part_size);
        for chunk in &chunks {
            gathered.extend_from_slice(chunk);
        }
        note_copy!(self, part_size);
        Ok(Some(PartData::single(gathered.freeze())))
    }

    /// Take everything still pending as the final (possibly short) part.
    fn take_tail(&mut self) -> Option<PartData> {
        if self.pending_len == 0 {
            return None;
        }
        self.pending_len = 0;
        let mut chunks: Vec<Bytes> = self.pending.drain(..).collect();
        if !self.open.is_empty() {
            chunks.push(self.open.split().freeze());
        }
        if chunks.len() > 1 && !self.unsigned {
            // Signed: one contiguous body is required; the tail is shorter than a part.
            let total: usize = chunks.iter().map(Bytes::len).sum();
            let mut gathered = BytesMut::with_capacity(total);
            for chunk in &chunks {
                gathered.extend_from_slice(chunk);
            }
            note_copy!(self, total);
            return Some(PartData::single(gathered.freeze()));
        }
        Some(PartData::new(chunks))
    }

    /// Signed mode, before borrowed input is appended: turn the queued owned slices into the
    /// open part buffer (one gather copy of bytes shorter than a part), so the borrowed bytes
    /// can be copied once, straight behind them, into the buffer that will be uploaded.
    fn gather_queue_into_open(&mut self) {
        let mut open = BytesMut::with_capacity(self.cfg.part_size);
        for chunk in self.pending.drain(..) {
            open.extend_from_slice(&chunk);
            note_copy!(self, chunk.len());
        }
        open.extend_from_slice(&self.open);
        self.open = open;
    }

    /// One step of a borrowed write: copy as much of `rest` as completes (or extends) the
    /// current part, advance `rest`, and return the part if one is now complete. `Ok(None)`
    /// means nothing more is ready and `rest` is fully absorbed.
    fn step_borrowed(&mut self, rest: &mut &[u8]) -> Result<Option<PartData>> {
        // A part left complete by an earlier failed enqueue goes first (in order).
        if let Some(part) = self.next_part()? {
            return Ok(Some(part));
        }
        if rest.is_empty() {
            return Ok(None);
        }
        let part_size = self.cfg.part_size;
        if !self.unsigned && !self.pending.is_empty() {
            self.gather_queue_into_open();
        }
        if self.pending_len == 0 && rest.len() >= part_size {
            // Whole part straight from the borrowed slice: one copy, into the upload buffer.
            let (head, tail) = rest.split_at(part_size);
            let mut part = BytesMut::with_capacity(part_size);
            part.extend_from_slice(head);
            note_copy!(self, part_size);
            *rest = tail;
            self.total_bytes += part_size as u64;
            return Ok(Some(PartData::single(part.freeze())));
        }
        let need = part_size.saturating_sub(self.pending_len);
        let take = need.min(rest.len());
        if self.open.is_empty() && self.open.capacity() < need {
            self.open = BytesMut::with_capacity(need);
        } else if self.open.capacity() - self.open.len() < take {
            self.open.reserve(take);
        }
        let (head, tail) = rest.split_at(take);
        self.open.extend_from_slice(head);
        note_copy!(self, take);
        self.pending_len += take;
        self.total_bytes += take as u64;
        *rest = tail;
        self.next_part()
    }

    /// Blocking write from a borrowed slice.
    ///
    /// `data` cannot outlive this call, so it must be copied: each byte exactly ONCE, straight
    /// into the part buffer that is uploaded (also when it completes a buffered tail). Callers
    /// that own their data should use [`write_bytes_blocking`](Self::write_bytes_blocking) or
    /// [`write_owned_blocking`](Self::write_owned_blocking), which never copy it.
    pub fn write_blocking(&mut self, data: &[u8]) -> Result<()> {
        let mut rest = data;
        while let Some(part) = self.step_borrowed(&mut rest)? {
            self.enqueue_part(part)?;
        }
        Ok(())
    }

    /// Async variant of [`write_blocking`](Self::write_blocking).
    async fn write_borrowed_async(&mut self, data: &[u8]) -> Result<()> {
        let mut rest = data;
        while let Some(part) = self.step_borrowed(&mut rest)? {
            self.enqueue_part_async(part).await?;
        }
        Ok(())
    }

    /// Async write — safe to call from a Tokio async task.
    ///
    /// Uses `part_tx.send().await` so the task yields instead of parking an OS
    /// thread when the channel is full.  The Python hot-path uses
    /// `write_blocking()` which is unchanged.
    ///
    /// The `Vec` is turned into `Bytes` without a copy and sliced into parts (see
    /// [`write_bytes`](Self::write_bytes)).
    #[cfg_attr(feature = "profiling", instrument(
        name = "mpu.write",
        skip(self, data),
        fields(data_len = data.len())
    ))]
    pub async fn write(&mut self, data: Vec<u8>) -> Result<()> {
        self.write_bytes(Bytes::from(data)).await
    }

    /// Async zero-copy write of owned `Bytes`: the data is sliced into parts, never copied
    /// (apart from the bounded gather of a part that straddles two writes in a SIGNED upload).
    pub async fn write_bytes(&mut self, data: Bytes) -> Result<()> {
        if data.len() < self.coalesce_below {
            return self.write_borrowed_async(&data).await;
        }
        self.push_owned(data);
        while let Some(part) = self.next_part()? {
            self.enqueue_part_async(part).await?;
        }
        Ok(())
    }

    /// True zero-copy blocking write: accepts already-owned Bytes (e.g. from BytesView Arc clone).
    /// No memcpy of the payload: the shared allocation is sliced into part-sized chunks. Only a
    /// part that straddles this write and a previously buffered tail is gathered, with one copy
    /// of at most `part_size` bytes, and only when payload signing is on (unsigned uploads send
    /// such a part as a multi-chunk body instead).
    pub fn write_bytes_blocking(&mut self, data: Bytes) -> Result<()> {
        if data.len() < self.coalesce_below {
            // Small write: one bounded copy into the part buffer (see `COALESCE_AUTO`).
            return self.write_blocking(&data);
        }
        self.push_owned(data);
        while let Some(part) = self.next_part()? {
            self.enqueue_part(part)?;
        }
        Ok(())
    }

    /// Zero-copy blocking write: accepts an owned Vec, converts to Bytes (no copy).
    pub fn write_owned_blocking(&mut self, data: Vec<u8>) -> Result<()> {
        self.write_bytes_blocking(Bytes::from(data))
    }

    /// Async version of `write_owned_blocking` — safe to call from a Tokio async task.
    pub async fn write_owned(&mut self, data: Vec<u8>) -> Result<()> {
        self.write(data).await
    }

    /// Flush any buffered data as a (possibly short) part.
    pub fn flush_blocking(&mut self) -> Result<()> {
        if let Some(chunk) = self.take_tail() {
            self.enqueue_part(chunk)?;
        }
        Ok(())
    }

    /// Async flush — safe to call from a Tokio async task.
    pub async fn flush(&mut self) -> Result<()> {
        if let Some(chunk) = self.take_tail() {
            self.enqueue_part_async(chunk).await?;
        }
        Ok(())
    }

    /// Async finish — safe to call from a Tokio async task.
    ///
    /// Flushes any buffered tail, signals the coordinator, and awaits
    /// `CompleteMultipartUpload`.  Use this instead of `finish_blocking()`
    /// when already inside a Tokio runtime.
    pub async fn finish(&mut self) -> Result<MultipartCompleteInfo> {
        if self.finished {
            bail!("finish() called more than once");
        }
        if let Some(chunk) = self.take_tail() {
            self.enqueue_part_async(chunk).await?;
        }
        self.part_tx
            .send(PartMsg::Finish)
            .await
            .map_err(|_| anyhow::anyhow!("coordinator task exited before Finish was sent"))?;
        let coordinator = self
            .coordinator
            .take()
            .ok_or_else(|| anyhow::anyhow!("coordinator already consumed"))?;
        let info = coordinator
            .await
            .map_err(|e| anyhow::anyhow!("coordinator task panicked: {e}"))??;
        self.finished = true;
        Ok(info)
    }

    // ── finish / abort ───────────────────────────────────────────────────────

    pub fn finish_blocking(&mut self) -> Result<MultipartCompleteInfo> {
        if self.finished {
            bail!("finish() called more than once");
        }

        // Flush tail
        if let Some(chunk) = self.take_tail() {
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

    /// Explicitly abort the upload.
    ///
    /// Locked contract (audit #151 f11, docs/implementation-plans/
    /// v0.9.109-audit-fix-plan.md bug A3): always cancels the coordinator
    /// (so it can never reach CompleteMultipartUpload after this call),
    /// always attempts a best-effort `AbortMultipartUpload`, and always
    /// returns `Ok(())` — the caller asked to abort; a transient failure
    /// of the *cleanup* request doesn't change that outcome, it only
    /// means the server-side upload may need manual cleanup (a
    /// `tracing::warn!` records the upload_id for that). Unlike Drop,
    /// this is an explicit user action, so the S3-side Abort always
    /// fires regardless of `abort_on_drop` (that flag only gates
    /// *implicit* cleanup-on-drop).
    ///
    /// Blocks the calling thread until the Abort was attempted; use [`MultipartUploadSink::abort`]
    /// from async code.
    pub fn abort_blocking(&mut self) -> Result<()> {
        if let Some(req) = self.cancel_coordinator(true) {
            let outcome = run_on_global_rt(async move {
                let what = req.describe();
                Ok::<_, anyhow::Error>((what, req.send().await))
            });
            match outcome {
                Ok((_, Ok(()))) => {}
                Ok((what, Err(e))) => warn_abort_failed(&what, &e),
                Err(e) => warn_abort_failed(
                    &format!(
                        "s3://{}/{} upload_id={}",
                        self.bucket, self.key, self.upload_id
                    ),
                    &e,
                ),
            }
        }
        Ok(())
    }

    /// Explicitly abort the upload from async code (#303).
    ///
    /// Same contract as [`MultipartUploadSink::abort_blocking`]: the coordinator is cancelled
    /// first (it can never reach CompleteMultipartUpload after this call), a best-effort
    /// `AbortMultipartUpload` is awaited (it runs on s3dlio's own runtime, so it completes even
    /// when the calling runtime is a current-thread one), a failure of that cleanup request is
    /// only logged (`warn!`, with the upload id) and the call returns `Ok(())`. It fires
    /// regardless of `abort_on_drop`. Calling it again, or after `finish()`, does nothing. If the
    /// returned future is dropped before it completes, the Abort still runs: the cancellation has
    /// already happened and the request is spawned.
    pub async fn abort(&mut self) -> Result<()> {
        if let Some(req) = self.cancel_coordinator(true) {
            let what = req.describe();
            // Spawned (not awaited inline) so that dropping this future cannot cancel the
            // cleanup half-way and leak the upload.
            let task = spawn_on_global_rt(async move { req.send().await });
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn_abort_failed(&what, &e),
                Err(e) => warn_abort_failed(&what, &anyhow::anyhow!("abort task failed: {e}")),
            }
        }
        Ok(())
    }

    // ── internal ─────────────────────────────────────────────────────────────

    /// Stop the coordinator before it can reach Complete (shared by `Drop::drop`,
    /// `abort_blocking()` and `abort()`). Returns the best-effort `AbortMultipartUpload` request
    /// to run, or `None` when none is wanted (`fire_abort == false`, or the sink is already
    /// finished or aborted). `fire_abort=false` is used only from Drop when `abort_on_drop=false`
    /// — the operator explicitly opted into leaving the upload dangling rather than paying for
    /// the extra request, per the locked A2 contract.
    fn cancel_coordinator(&mut self, fire_abort: bool) -> Option<AbortRequest> {
        if self.finished {
            return None;
        }
        self.finished = true;

        // Signal cancellation FIRST, before anything else — this is what
        // stops the coordinator from falling through to Complete once
        // part_tx closes. Cancelling here (rather than relying solely on
        // channel-close) closes the race window entirely: the
        // coordinator's select! observes cancellation independently of
        // channel state.
        self.cancel.cancel();

        fire_abort.then(|| AbortRequest {
            client: self.client.clone(),
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            upload_id: self.upload_id.clone(),
        })
    }

    /// Enqueue a complete part for upload via the coordinator channel.
    ///
    /// `blocking_send` parks the caller (Python thread) only when the channel
    /// is full — i.e., `resolved_mif` parts are already queued or in-flight.
    /// This is the backpressure point: memory is bounded at
    /// `~2 × resolved_mif × part_size` bytes.
    ///
    /// **Must only be called from non-Tokio threads** (e.g. Python threads).
    /// Use `enqueue_part_async` from async Tokio tasks.
    fn enqueue_part(&mut self, data: PartData) -> Result<()> {
        let part_number = self.next_part_number;
        if part_number as usize > MAX_MULTIPART_PARTS {
            bail!(
                "exceeded S3 maximum of {MAX_MULTIPART_PARTS} parts; \
                 use a larger part_size for objects this large"
            );
        }
        self.next_part_number += 1;

        self.part_tx
            .blocking_send(PartMsg::Part { data, part_number })
            .map_err(|_| anyhow::anyhow!("coordinator task exited unexpectedly"))?;
        Ok(())
    }

    /// Async variant of `enqueue_part` — safe to call from a Tokio async task.
    async fn enqueue_part_async(&mut self, data: PartData) -> Result<()> {
        let part_number = self.next_part_number;
        if part_number as usize > MAX_MULTIPART_PARTS {
            bail!(
                "exceeded S3 maximum of {MAX_MULTIPART_PARTS} parts; \
                 use a larger part_size for objects this large"
            );
        }
        self.next_part_number += 1;

        self.part_tx
            .send(PartMsg::Part { data, part_number })
            .await
            .map_err(|_| anyhow::anyhow!("coordinator task exited unexpectedly"))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinator task (runs entirely on Tokio runtime)
// ─────────────────────────────────────────────────────────────────────────────

/// Where one multipart upload goes.
struct UploadTarget {
    bucket: String,
    key: String,
    upload_id: String,
}

/// Background task that owns all async S3 work for one multipart upload.
///
/// Receives parts from the bounded channel, acquires semaphore permits
/// asynchronously (never parking the Python caller thread), spawns
/// UploadPart tasks, collects ETags, and finally calls CompleteMultipartUpload.
async fn coordinator_task(
    client: Client,
    target: UploadTarget,
    max_in_flight: usize,
    mut part_rx: mpsc::Receiver<PartMsg>,
    cancel: CancellationToken,
    unsigned: bool,
) -> Result<MultipartCompleteInfo> {
    let UploadTarget {
        bucket,
        key,
        upload_id,
    } = target;
    let sem = std::sync::Arc::new(Semaphore::new(max_in_flight));
    let started_at = SystemTime::now();
    let mut total_bytes: u64 = 0;
    let mut part_tasks: Vec<JoinHandle<Result<(i32, String)>>> = Vec::new();

    debug!(
        "s3dlio MPU: start upload_id={} s3://{}/{}",
        upload_id, bucket, key
    );

    // Locked contract (audit #151 f11/f12, docs/implementation-plans/
    // v0.9.109-audit-fix-plan.md bug A2/A3): racing cancellation against
    // the receive is what actually stops this task from reaching
    // CompleteMultipartUpload after the sink is dropped or abort() is
    // called. The prior design relied on `part_tx` closing to end the
    // loop, but that path fell through to Complete unconditionally —
    // dropping a `JoinHandle` (as the struct's own Drop does with
    // `coordinator`) detaches the task rather than cancelling it, so
    // there was no way to signal "stop, don't commit" from outside.
    // `biased` ensures a cancellation that raced in alongside a
    // already-buffered part is still honored before processing it.
    let mut cancelled = false;
    loop {
        let msg = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                cancelled = true;
                None
            }
            m = part_rx.recv() => m,
        };

        let (data, part_number) = match msg {
            Some(PartMsg::Part { data, part_number }) => (data, part_number),
            Some(PartMsg::Finish) | None => break,
        };

        let part_bytes = data.len() as u64;
        total_bytes += part_bytes;

        debug!(
            "s3dlio MPU: enqueue part={} size={} total_so_far={} s3://{}/{}",
            part_number, part_bytes, total_bytes, bucket, key
        );

        // Acquire a semaphore permit ASYNCHRONOUSLY.
        // This never parks the Python caller thread. Raced against cancellation so an
        // abort() / drop is honoured promptly even while every upload slot is busy (the part
        // in hand is dropped, releasing its refcount on the caller's buffer).
        let permit = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                cancelled = true;
                break;
            }
            p = sem.clone().acquire_owned() => {
                p.map_err(|_| anyhow::anyhow!("semaphore closed"))?
            }
        };

        let c = client.clone();
        let b = bucket.clone();
        let k = key.clone();
        let uid = upload_id.clone();

        let handle = tokio::task::spawn(async move {
            let _permit = permit; // released when task completes

            // A one-chunk part is an in-memory body (`ByteStream::from(Bytes)`: the redirect
            // client's try_clone() succeeds, so AIStore 307 redirects on UploadPart are
            // followed correctly); a multi-chunk part (unsigned uploads only) is a retryable
            // body of known length built from the same slices.
            let body = data.into_byte_stream();
            // Build the error-context string before moving b/k into the request
            // builder.  Preserves the bucket+key+part_number in the message
            // even when the upload fails with a bare dispatch failure.
            let upload_ctx = format!("UploadPart {} failed for s3://{}/{}", part_number, b, k);
            let request = c
                .upload_part()
                .bucket(&b)
                .key(&k)
                .upload_id(uid)
                .part_number(part_number)
                .body(body);
            // Payload signing off (S3DLIO_UNSIGNED_PAYLOAD, same as PutObject): no SHA-256
            // pass over the part, `x-amz-content-sha256: UNSIGNED-PAYLOAD`.
            let resp = if unsigned {
                request
                    .customize()
                    .disable_payload_signing()
                    .send()
                    .await
                    .sdk_context(upload_ctx)?
            } else {
                request.send().await.sdk_context(upload_ctx)?
            };

            let etag = resp.e_tag().unwrap_or_default().to_string();
            if etag.is_empty() {
                bail!("UploadPart returned empty ETag for part {part_number}");
            }
            debug!(
                "s3dlio MPU: part={} done ETag={} s3://{}/{}",
                part_number, etag, b, k
            );
            Ok::<(i32, String), anyhow::Error>((part_number, etag))
        });

        part_tasks.push(handle);
    }

    if cancelled {
        // Deliberately do NOT join part_tasks or call CompleteMultipartUpload.
        // Already-spawned UploadPart tasks are ABORTED (not merely detached): the upload is
        // being discarded, and a running task would keep a refcount on its part's bytes (for
        // the Python writer, on the caller's borrowed buffer) until its request finished. An
        // aborted task drops its request future promptly, so `abort()` can hand the buffer
        // back. Parts that already landed are harmless uncommitted parts (cleaned up by
        // AbortMultipartUpload, when it fires) or are left dangling if the caller opted out
        // via abort_on_drop=false.
        for handle in &part_tasks {
            handle.abort();
        }
        debug!(
            "s3dlio MPU: coordinator cancelled before Finish for upload_id={} s3://{}/{} \
             ({} part upload task(s) were in flight and will not be awaited) — \
             CompleteMultipartUpload will NOT be called",
            upload_id,
            bucket,
            key,
            part_tasks.len()
        );
        bail!(
            "multipart upload for s3://{}/{} (upload_id={}) was aborted before finish() \
             was called — the sink was dropped or abort() was invoked",
            bucket,
            key,
            upload_id
        );
    }

    // Join all part tasks.
    //
    // Full-drain-first-then-error (issue #148 finding 3.2): the previous
    // shape used `handle.await.map_err(...)??` inside a `for` loop, so on
    // the first failed part every remaining `JoinHandle` in `part_tasks`
    // got dropped — which detaches the tasks rather than aborting them.
    // The result was that in-flight UploadPart calls kept running against
    // S3 after this coordinator had already returned an error to the
    // Python caller, wasting bandwidth and (worse) sometimes producing
    // orphaned parts that showed up in later ListMultipartUploads calls.
    // Draining every task first eliminates the leak; the first observed
    // error is what we surface.
    let mut parts: Vec<(i32, String)> = Vec::with_capacity(part_tasks.len());
    let mut first_err: Option<anyhow::Error> = None;
    for handle in part_tasks {
        match handle.await {
            Ok(Ok(pair)) => parts.push(pair),
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            Err(join_err) => {
                first_err.get_or_insert(anyhow::anyhow!("part task panicked: {}", join_err));
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    parts.sort_by_key(|(pn, _)| *pn);

    let n_parts = parts.len();
    debug!(
        "s3dlio MPU: CompleteMultipartUpload parts={} total_bytes={} s3://{}/{}",
        n_parts, total_bytes, bucket, key
    );

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

    let complete_ctx = format!("CompleteMultipartUpload failed for s3://{}/{}", bucket, key);
    let resp = client
        .complete_multipart_upload()
        .bucket(&bucket)
        .key(&key)
        .upload_id(upload_id)
        .multipart_upload(cmu)
        .send()
        .await
        .sdk_context(complete_ctx)?;

    debug!(
        "s3dlio MPU: CompleteMultipartUpload OK ETag={:?} s3://{}/{}",
        resp.e_tag, bucket, key
    );

    // Verify the stored object size via HEAD — opt-in via S3DLIO_MPU_PUT_VERIFY
    // (default off).  S3-compatible backends (AIStore in particular) have been
    // observed to silently drop parts and still return valid ETags / a 200 OK
    // on CompleteMultipartUpload; without this check a truncated upload
    // appears successful.  This is a server-side contract violation (a 200
    // response should mean the object is fully and durably stored), so the
    // check is opt-in rather than mandatory overhead for every backend.
    // Enable DEBUG logging (RUST_LOG=s3dlio=debug) to trace redirect hops and
    // per-part ETags when diagnosing size mismatches on multi-node AIStore.
    let verify_enabled = std::env::var("S3DLIO_MPU_PUT_VERIFY")
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    let stored_bytes = if !verify_enabled {
        debug!(
            "s3dlio MPU: skipping HEAD verification for s3://{}/{} (set S3DLIO_MPU_PUT_VERIFY=true to enable)",
            bucket, key
        );
        total_bytes
    } else {
        match client.head_object().bucket(&bucket).key(&key).send().await {
            Ok(head) => head.content_length().unwrap_or(0) as u64,
            Err(e) => {
                // audit #151 bug 1.3 (D3): CompleteMultipartUpload above
                // already succeeded (we have its ETag). A transient
                // failure of the HEAD verification REQUEST ITSELF
                // (network blip, throttling) is not evidence the object
                // is corrupt — it's evidence the verification call
                // failed. Treating it as a hard error here would delete
                // a perfectly good object (the size-mismatch branch
                // below does exactly that) on nothing more than a flaky
                // HEAD. Soft-fail: warn and assume the upload is correct
                // rather than manufacturing data-corruption handling out
                // of an unrelated RPC hiccup.
                warn!(
                    "s3dlio MPU: HEAD verification request failed for s3://{}/{} \
                     (CompleteMultipartUpload already succeeded, ETag={:?}) — assuming the \
                     upload is correct rather than treating a failed verification call as \
                     data corruption: {}",
                    bucket, key, resp.e_tag, e
                );
                total_bytes
            }
        }
    };

    if verify_enabled && stored_bytes != total_bytes {
        warn!(
            "s3dlio MPU: size mismatch s3://{}/{} written={} stored={} — \
             possible redirect or AIStore replication issue; \
             set RUST_LOG=s3dlio=debug for per-part trace",
            bucket, key, total_bytes, stored_bytes
        );

        // Delete the truncated object so callers and downstream readers do not
        // silently consume corrupt data.  The streaming write data is already
        // consumed; the caller must restart the upload from scratch.
        let cleanup_ok = client
            .delete_object()
            .bucket(&bucket)
            .key(&key)
            .send()
            .await
            .is_ok();
        if cleanup_ok {
            warn!(
                "s3dlio MPU: deleted truncated object s3://{}/{} — caller must retry upload",
                bucket, key
            );
        } else {
            warn!(
                "s3dlio MPU: failed to delete truncated object s3://{}/{} — \
                 delete it manually before retrying",
                bucket, key
            );
        }

        bail!(
            "Size mismatch after CompleteMultipartUpload for s3://{}/{}: \
             wrote {} bytes but HEAD reports {} bytes stored. \
             {} Retry the upload from scratch.",
            bucket,
            key,
            total_bytes,
            stored_bytes,
            if cleanup_ok {
                "Truncated object has been deleted."
            } else {
                "WARNING: truncated object could not be deleted — remove it manually before retrying."
            }
        );
    }

    if verify_enabled {
        debug!(
            "s3dlio MPU: HEAD verified stored={} bytes s3://{}/{}",
            stored_bytes, bucket, key
        );
    }

    Ok(MultipartCompleteInfo {
        e_tag: resp.e_tag.map(|s| s.to_string()),
        total_bytes,
        stored_bytes,
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
            "error must mention 5 MiB minimum, got: {}",
            err
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
            "error must mention max_in_flight, got: {}",
            err
        );
    }

    /// from_uri must reject non-s3:// URIs before any network call.
    #[test]
    fn test_multipart_from_uri_rejects_invalid_uri() {
        let cfg = MultipartUploadConfig::default();
        let err = MultipartUploadSink::from_uri("not-a-valid-uri", cfg)
            .err()
            .expect("must fail for invalid URI");
        assert!(
            !err.to_string().is_empty(),
            "error message must not be empty"
        );
    }

    /// Default MultipartUploadConfig has valid values.
    #[test]
    fn test_multipart_default_config_is_valid() {
        let cfg = MultipartUploadConfig::default();
        assert!(
            cfg.part_size >= MIN_S3_MULTIPART_PART_SIZE,
            "default part_size {} must be >= MIN {}",
            cfg.part_size,
            MIN_S3_MULTIPART_PART_SIZE
        );
        assert!(cfg.abort_on_drop, "default abort_on_drop should be true");
        // max_in_flight=None means auto
        assert!(
            cfg.max_in_flight.is_none(),
            "default max_in_flight should be None (auto)"
        );
    }

    // ------------------------------------------------------------------
    // auto_max_in_flight formula
    // ------------------------------------------------------------------

    #[test]
    fn test_auto_max_in_flight_floor() {
        // Any part size large enough to produce a value below 32 should still give 32.
        assert_eq!(auto_max_in_flight(64 * 1024 * 1024), 32); // 64 MiB → 8, floor=32
        assert_eq!(auto_max_in_flight(128 * 1024 * 1024), 32); // 128 MiB → 4, floor=32
    }

    #[test]
    fn test_auto_max_in_flight_scales_with_small_parts() {
        // Small parts produce larger max_in_flight.
        let mif_8mib = auto_max_in_flight(8 * 1024 * 1024); // 512/8 = 64
        let mif_16mib = auto_max_in_flight(16 * 1024 * 1024); // 512/16 = 32
        assert_eq!(mif_8mib, 64);
        assert_eq!(mif_16mib, 32);
    }

    /// Proves that calling `blocking_send` from within a Tokio runtime panics.
    /// This is the exact bug class that the old `write()` / `write_owned()` had:
    /// they delegated to `write_blocking()` which calls `blocking_send()` via
    /// `enqueue_part()`.  The fixed async methods use `send().await` instead.
    #[test]
    fn test_blocking_send_panics_inside_tokio_runtime() {
        use std::panic;
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel::<i32>(4);
        // blocking_send must panic when called from within a Tokio runtime.
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            rt.block_on(async {
                let _ = tx.blocking_send(42);
            })
        }));
        assert!(
            result.is_err(),
            "blocking_send must panic inside a Tokio runtime — \
             this is the bug class write()/write_owned() previously had"
        );
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

    // ------------------------------------------------------------------
    // BUG #593 Bug 2 — structural test
    // ------------------------------------------------------------------

    /// `MultipartCompleteInfo` must carry `stored_bytes: u64` populated by a
    /// HEAD request after `CompleteMultipartUpload`.  Without it the
    /// coordinator never verifies that the stored file size equals the bytes
    /// written — AIStore can silently drop parts and return valid ETags.
    ///
    /// This test **FAILS TO COMPILE** before the fix because `stored_bytes`
    /// does not exist on the struct.  After the fix it compiles and the
    /// assertion verifies the field is reachable.
    #[test]
    fn test_complete_info_has_stored_bytes_field() {
        let info = MultipartCompleteInfo {
            e_tag: None,
            total_bytes: 1024,
            stored_bytes: 1024, // BUG: field missing → compile error before fix
            parts: 1,
            started_at: SystemTime::UNIX_EPOCH,
            completed_at: SystemTime::UNIX_EPOCH,
        };
        assert_eq!(
            info.stored_bytes, info.total_bytes,
            "stored size from HEAD must match bytes written"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zero-copy part assembly (stage G, W2/W3): pointer-identity and copy-count tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
impl MultipartUploadSink {
    /// A sink with no coordinator and no S3 behind it: parts land on the returned receiver
    /// instead of being uploaded. `part_size` may be below the S3 minimum (tests only).
    /// `unsigned` selects the unsigned-payload assembly (multi-chunk straddling parts).
    fn for_test(part_size: usize, unsigned: bool) -> (Self, mpsc::Receiver<PartMsg>) {
        let (part_tx, part_rx) = mpsc::channel(4096);
        let conf = aws_sdk_s3::Config::builder()
            .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
            .region(aws_sdk_s3::config::Region::new("us-east-1"))
            .build();
        let sink = Self {
            part_tx,
            coordinator: None,
            cancel: CancellationToken::new(),
            pending: VecDeque::new(),
            open: BytesMut::new(),
            pending_len: 0,
            unsigned,
            coalesce_below: 0,
            copied: 0,
            next_part_number: 1,
            total_bytes: 0,
            cfg: MultipartUploadConfig {
                part_size,
                max_in_flight: Some(4096),
                abort_on_drop: false,
                content_type: None,
            },
            resolved_mif: 4096,
            upload_id: "test-upload".to_string(),
            client: Client::from_conf(conf),
            bucket: "b".to_string(),
            key: "k".to_string(),
            // finished = true: Drop must not try to abort an upload that never existed.
            finished: true,
        };
        (sink, part_rx)
    }
}

#[cfg(test)]
mod zero_copy_tests {
    use super::*;

    const P: usize = 1000;

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn drain(rx: &mut mpsc::Receiver<PartMsg>) -> Vec<PartData> {
        let mut parts = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let PartMsg::Part { data, .. } = msg {
                parts.push(data);
            }
        }
        parts
    }

    fn flat(part: &PartData) -> Vec<u8> {
        part.chunks.iter().flat_map(|c| c.iter().copied()).collect()
    }

    /// `part` lies entirely inside the allocation of `src`: it is a slice of it, not a copy.
    fn is_slice_of(part: &Bytes, src: &Bytes) -> bool {
        let (p, s) = (part.as_ptr() as usize, src.as_ptr() as usize);
        !part.is_empty() && p >= s && p + part.len() <= s + src.len()
    }

    fn single(part: &PartData) -> &Bytes {
        assert_eq!(part.chunks.len(), 1, "a contiguous body was expected");
        &part.chunks[0]
    }

    /// Signed upload, owned inputs of 2.5 and 2.7 parts: the straddling part (tail of the first
    /// write plus head of the second) is the one gather copy (exactly one part); every other part
    /// must be a slice of its input.
    #[test]
    fn bytes_input_is_sliced_even_when_a_tail_is_buffered() {
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, false);
        let a = Bytes::from(pattern(2500, 1));
        let b = Bytes::from(pattern(2700, 2));

        sink.write_bytes_blocking(a.clone()).unwrap();
        let first = drain(&mut rx);
        assert_eq!(first.len(), 2);
        assert!(is_slice_of(single(&first[0]), &a) && is_slice_of(single(&first[1]), &a));
        assert_eq!(
            sink.copied_bytes(),
            0,
            "whole parts of an owned write: no copy"
        );

        sink.write_bytes_blocking(b.clone()).unwrap();
        let second = drain(&mut rx);
        assert_eq!(second.len(), 3, "straddling part plus two whole parts of b");
        let mut straddle = a[2000..].to_vec();
        straddle.extend_from_slice(&b[..500]);
        assert_eq!(flat(&second[0]), straddle, "straddling part content");
        assert!(
            is_slice_of(single(&second[1]), &b) && is_slice_of(single(&second[2]), &b),
            "parts that lie inside one input must be slices of it, not copies"
        );
        assert_eq!(
            sink.copied_bytes(),
            P,
            "only the straddling part is gathered"
        );

        // The tail left over is a slice of b as well, and flushes without a copy.
        sink.flush_blocking().unwrap();
        let tail = drain(&mut rx);
        assert_eq!(tail.len(), 1);
        assert_eq!(flat(&tail[0]), b[2500..].to_vec());
        assert!(
            is_slice_of(single(&tail[0]), &b),
            "a single-chunk tail is not copied"
        );
        assert_eq!(sink.copied_bytes(), P);
    }

    /// Unsigned upload: the same inputs, but the straddling part is NOT gathered. It is the two
    /// slices, each pointing into its own input, and nothing is copied at all.
    #[test]
    fn unsigned_straddling_part_is_a_chunk_list_with_no_copy() {
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, true);
        let a = Bytes::from(pattern(2500, 1));
        let b = Bytes::from(pattern(2700, 2));
        sink.write_bytes_blocking(a.clone()).unwrap();
        sink.write_bytes_blocking(b.clone()).unwrap();
        let parts = drain(&mut rx);
        assert_eq!(parts.len(), 5);
        let straddle = &parts[2];
        assert_eq!(straddle.chunks.len(), 2, "tail of a + head of b");
        assert!(
            is_slice_of(&straddle.chunks[0], &a),
            "first chunk points into a"
        );
        assert!(
            is_slice_of(&straddle.chunks[1], &b),
            "second chunk points into b"
        );
        assert_eq!(straddle.len(), P);
        let mut expected = a[2000..].to_vec();
        expected.extend_from_slice(&b[..500]);
        assert_eq!(flat(straddle), expected);
        assert!(is_slice_of(single(&parts[3]), &b) && is_slice_of(single(&parts[4]), &b));
        assert_eq!(sink.copied_bytes(), 0, "an unsigned upload copies nothing");
        // The multi-chunk body has a known length (Content-Length, not chunked encoding).
        let body = straddle.chunks.clone();
        let stream = PartData::new(body).into_byte_stream();
        assert_eq!(stream.size_hint(), (P as u64, Some(P as u64)));
    }

    /// Many small owned writes in an unsigned upload: a part spanning several slices is a list
    /// of all of them, in order, still without a copy.
    #[test]
    fn unsigned_part_spanning_many_small_writes_has_no_copy() {
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, true);
        let inputs: Vec<Bytes> = (0..10).map(|i| Bytes::from(pattern(300, i))).collect();
        for b in &inputs {
            sink.write_bytes_blocking(b.clone()).unwrap();
        }
        let parts = drain(&mut rx);
        assert_eq!(parts.len(), 3);
        assert_eq!(sink.copied_bytes(), 0);
        let joined: Vec<u8> = parts.iter().flat_map(flat).collect();
        let want: Vec<u8> = inputs.iter().flat_map(|b| b.iter().copied()).collect();
        assert_eq!(&joined[..], &want[..3000]);
        for chunk in parts.iter().flat_map(|p| p.chunks.iter()) {
            assert!(
                inputs.iter().any(|b| is_slice_of(chunk, b)),
                "every chunk is a slice of one of the inputs"
            );
        }
    }

    #[test]
    fn owned_vec_input_is_sliced_even_when_a_tail_is_buffered() {
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, false);
        let first_vec = pattern(2500, 3);
        let second_vec = pattern(3200, 4);
        // Bytes::from(Vec) keeps the Vec's allocation, so the pointer range identifies it.
        let second_ptr = second_vec.as_ptr() as usize;

        sink.write_owned_blocking(first_vec).unwrap();
        sink.write_owned_blocking(second_vec).unwrap();
        let parts = drain(&mut rx);
        assert_eq!(parts.len(), 5);
        // part 3 straddles; parts 4 and 5 lie in the second input's own allocation.
        for p in &parts[3..] {
            let ptr = single(p).as_ptr() as usize;
            assert!(
                ptr >= second_ptr && ptr + p.len() <= second_ptr + 3200,
                "part must point into the owned Vec's allocation"
            );
        }
    }

    #[test]
    fn async_write_slices_owned_vec_with_a_tail() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, false);
        let second = pattern(3200, 6);
        let second_ptr = second.as_ptr() as usize;
        rt.block_on(async {
            sink.write(pattern(2500, 5)).await.unwrap();
            sink.write(second).await.unwrap();
        });
        let parts = drain(&mut rx);
        assert_eq!(parts.len(), 5);
        for p in &parts[3..] {
            let ptr = single(p).as_ptr() as usize;
            assert!(ptr >= second_ptr && ptr + p.len() <= second_ptr + 3200);
        }
    }

    /// Many writes of every kind, in both modes: the whole stream comes out in order, every
    /// part but the last exactly `part_size`, whatever the split points.
    #[test]
    fn many_small_writes_reassemble_exactly() {
        for unsigned in [false, true] {
            let (mut sink, mut rx) = MultipartUploadSink::for_test(P, unsigned);
            let src = pattern(23_457, 9);
            let mut seed = 12345u64;
            let mut at = 0usize;
            while at < src.len() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let n = (((seed >> 33) as usize) % 2300 + 1).min(src.len() - at);
                let chunk = Bytes::from(src[at..at + n].to_vec());
                match (seed >> 20) % 3 {
                    0 => sink.write_bytes_blocking(chunk).unwrap(),
                    1 => sink.write_owned_blocking(chunk.to_vec()).unwrap(),
                    _ => sink.write_blocking(&chunk).unwrap(),
                }
                at += n;
            }
            sink.flush_blocking().unwrap();
            let parts = drain(&mut rx);
            let (last, whole) = parts.split_last().unwrap();
            assert!(whole.iter().all(|p| p.len() == P), "unsigned={unsigned}");
            assert!(last.len() != 0 && last.len() <= P);
            let joined: Vec<u8> = parts.iter().flat_map(flat).collect();
            assert_eq!(joined, src, "unsigned={unsigned}");
            assert_eq!(sink.total_bytes, src.len() as u64);
        }
    }

    /// The borrowed-slice API cannot keep the caller's memory, so it copies - but each byte
    /// exactly ONCE, also when a write completes a buffered tail (a borrowed write straddling a
    /// borrowed tail used to be copied into the tail and then gathered again).
    #[test]
    fn borrowed_slice_straddling_a_buffered_tail_is_copied_exactly_once() {
        for unsigned in [false, true] {
            let (mut sink, mut rx) = MultipartUploadSink::for_test(P, unsigned);
            let src = pattern(7_300, 7);
            sink.write_blocking(&src[..400]).unwrap();
            assert_eq!(sink.copied_bytes(), 400);
            sink.write_blocking(&src[400..5_900]).unwrap();
            sink.write_blocking(&src[5_900..]).unwrap();
            assert_eq!(
                sink.copied_bytes(),
                src.len(),
                "every borrowed byte is copied exactly once (unsigned={unsigned})"
            );
            sink.flush_blocking().unwrap();
            let parts = drain(&mut rx);
            assert_eq!(parts.len(), 8);
            let joined: Vec<u8> = parts.iter().flat_map(flat).collect();
            assert_eq!(joined, src);
            // After flush a single-chunk tail is not copied again.
            assert_eq!(sink.copied_bytes(), src.len());
        }
    }

    /// An owned tail followed by a borrowed write (signed): the tail is gathered once, the
    /// borrowed bytes copied once, so every byte is still copied at most once.
    #[test]
    fn owned_tail_then_borrowed_write_copies_each_byte_at_most_once() {
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, false);
        let owned = Bytes::from(pattern(500, 1));
        let borrowed = pattern(700, 2);
        sink.write_bytes_blocking(owned.clone()).unwrap();
        sink.write_blocking(&borrowed).unwrap();
        assert_eq!(sink.copied_bytes(), 500 + 700);
        let parts = drain(&mut rx);
        assert_eq!(parts.len(), 1);
        let mut want = owned.to_vec();
        want.extend_from_slice(&borrowed[..500]);
        assert_eq!(flat(&parts[0]), want);
    }

    /// A write after a failed enqueue (the coordinator is gone) leaves `pending_len >=
    /// part_size`; the next write must report the failure, not underflow `part_size -
    /// pending_len` (a panic in debug builds).
    #[test]
    fn a_write_after_a_failed_enqueue_is_an_error_not_an_underflow() {
        for unsigned in [false, true] {
            let (mut sink, rx) = MultipartUploadSink::for_test(P, unsigned);
            drop(rx); // every enqueue now fails
            let err = sink
                .write_bytes_blocking(Bytes::from(pattern(2_500, 1)))
                .unwrap_err();
            assert!(err.to_string().contains("coordinator"), "{err}");
            // 1500 bytes are still pending (>= one part). Must not panic.
            let err = sink.write_blocking(&pattern(10, 2)).unwrap_err();
            assert!(err.to_string().contains("coordinator"), "{err}");
            let err = sink
                .write_bytes_blocking(Bytes::from(pattern(600, 3)))
                .unwrap_err();
            assert!(err.to_string().contains("coordinator"), "{err}");
        }
    }

    /// Owned writes below the coalescing threshold are copied once into the part buffer (not
    /// queued as one slice each); bigger writes are still sliced, never copied.
    #[test]
    fn small_owned_writes_are_coalesced_with_one_copy_big_ones_are_sliced() {
        for unsigned in [false, true] {
            let (mut sink, mut rx) = MultipartUploadSink::for_test(P, unsigned);
            sink.coalesce_below = 300;
            let small: Vec<Bytes> = (0..25).map(|i| Bytes::from(pattern(100, i))).collect();
            for b in &small {
                sink.write_bytes_blocking(b.clone()).unwrap();
            }
            let parts = drain(&mut rx);
            assert_eq!(parts.len(), 2);
            assert!(
                parts.iter().all(|p| p.chunks.len() == 1),
                "coalesced into contiguous part buffers"
            );
            assert_eq!(sink.copied_bytes(), 2_500, "each small byte copied once");
            let big = Bytes::from(pattern(2 * P, 9));
            sink.write_bytes_blocking(big.clone()).unwrap();
            let parts = drain(&mut rx);
            assert!(!parts.is_empty());
            let head = if unsigned { 0 } else { 500 };
            assert_eq!(
                sink.copied_bytes(),
                2_500 + head,
                "only the head that completes the open buffer (signed)"
            );
            let last = parts.last().unwrap();
            assert!(is_slice_of(&last.chunks[0], &big), "big write is sliced");
        }
    }

    /// The request body of a straddling (unsigned) part hands the SDK the very slices it was
    /// built from: collecting it yields segments that point into the original writes.
    #[test]
    fn multi_chunk_body_segments_are_the_original_slices() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let a = Bytes::from(pattern(2500, 1));
        let b = Bytes::from(pattern(2700, 2));
        let (mut sink, mut rx) = MultipartUploadSink::for_test(P, true);
        sink.write_bytes_blocking(a.clone()).unwrap();
        sink.write_bytes_blocking(b.clone()).unwrap();
        let straddle = drain(&mut rx).remove(2);
        let segments: Vec<Bytes> = rt
            .block_on(async { straddle.into_byte_stream().collect().await })
            .unwrap()
            .into_segments()
            .collect();
        assert_eq!(segments.len(), 2);
        assert!(is_slice_of(&segments[0], &a), "segment 0 points into a");
        assert!(is_slice_of(&segments[1], &b), "segment 1 points into b");
    }

    /// The default small-write threshold is the part size: a write smaller than a part is
    /// copied once into the part buffer, a write of at least one part is sliced.
    #[test]
    fn the_default_coalescing_threshold_is_the_part_size() {
        assert_eq!(coalesce_below(), COALESCE_AUTO);
        assert_eq!(resolve_coalesce_below(5 << 20, false), 5 << 20);
        assert_eq!(resolve_coalesce_below(16 << 20, false), 16 << 20);
        assert_eq!(
            resolve_coalesce_below(16 << 20, true),
            UNSIGNED_COALESCE_BELOW,
            "unsigned uploads borrow down to a fixed small size"
        );
    }

    /// The multi-chunk body yields its slices in order, byte for byte, and can be rebuilt
    /// (retries) from the same slices.
    #[test]
    fn multi_chunk_body_is_exact_and_rebuildable() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let chunks = vec![
            Bytes::from(pattern(700, 1)),
            Bytes::from(pattern(1, 2)),
            Bytes::from(pattern(3000, 3)),
        ];
        let want: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        let part = PartData::new(chunks);
        assert_eq!(part.len(), want.len());
        let got = rt
            .block_on(async { part.into_byte_stream().collect().await })
            .unwrap()
            .into_bytes();
        assert_eq!(&got[..], &want[..]);
    }
}

#[cfg(all(test, feature = "extension-module"))]
impl MultipartUploadSink {
    /// A sink with no coordinator, for the Python writer's unit tests: parts pile up in a
    /// channel that the returned guard keeps alive. `coalesce_below` is the small-write
    /// threshold (0 borrows every write).
    pub(crate) fn for_writer_test(
        part_size: usize,
        coalesce_below: usize,
    ) -> (Self, WriterTestGuard) {
        let (mut sink, rx) = Self::for_test(part_size, false);
        sink.coalesce_below = coalesce_below;
        (sink, WriterTestGuard(rx))
    }
}

/// Keeps the parts channel of a [`MultipartUploadSink::for_writer_test`] sink alive and lets a
/// test look at the parts the writer produced.
#[cfg(all(test, feature = "extension-module"))]
pub(crate) struct WriterTestGuard(mpsc::Receiver<PartMsg>);

#[cfg(all(test, feature = "extension-module"))]
impl WriterTestGuard {
    /// Every chunk of every part queued so far, taken out of the channel. They are the very
    /// `Bytes` the writer queued (same allocation, nothing copied), so a test can compare
    /// addresses and owner counts.
    pub(crate) fn take_chunks(&mut self) -> Vec<Bytes> {
        let mut out = Vec::new();
        while let Ok(msg) = self.0.try_recv() {
            if let PartMsg::Part { data, .. } = msg {
                out.extend(data.chunks);
            }
        }
        out
    }
}
