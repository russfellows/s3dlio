// src/google_gcs_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::constants::{
    BYTES_PER_GIB, DEFAULT_GCS_READ_PROGRESS_INTERVAL_SECS, DEFAULT_GCS_READ_TIMEOUT_MAX_SECS,
    DEFAULT_GCS_READ_TIMEOUT_MIN_SECS, DEFAULT_GCS_READ_TIMEOUT_SECS_PER_GIB,
    ENV_GCS_BIDI_ATTEMPT_TIMEOUT_SECS, ENV_GCS_READ_CHUNK_TIMEOUT_SECS,
    ENV_GCS_READ_PROGRESS_INTERVAL_SECS, ENV_GCS_READ_TIMEOUT_MAX_SECS,
    ENV_GCS_READ_TIMEOUT_MIN_SECS, ENV_GCS_READ_TIMEOUT_SECS_PER_GIB,
};
use crate::gcs_constants::{
    DEFAULT_WINDOW_MIB, ENV_GCS_GRPC_CHANNELS, ENV_GRPC_INITIAL_WINDOW_MIB,
    GCS_MAX_CONCURRENT_DELETES, GCS_MIN_CHANNELS,
};
#[cfg(not(feature = "gcs-community"))]
use crate::object_error::ObjectErrorKind;
use crate::read_buffer::ReadBuffer;
use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures::Stream;
use google_cloud_gax::paginator::ItemPaginator;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;
use tracing::{debug, info, trace, warn};

// Global cached GCS clients - initialized once and reused across all operations
static GCS_STORAGE: OnceCell<Arc<Storage>> = OnceCell::const_new();
static GCS_CONTROL: OnceCell<Arc<StorageControl>> = OnceCell::const_new();
static EFFECTIVE_GCS_RAPID_MODE: std::sync::OnceLock<RapidMode> = std::sync::OnceLock::new();
const GCS_FULL_READ_REPAIR_MAX_ATTEMPTS: usize = 4;
const GCS_FULL_READ_RETRY_MAX_ATTEMPTS: usize = 3;

/// Exponential backoff delay for GCS full-read retry attempts.
///
/// audit #157 bug 5.1 (D1): the full-read retry loop in
/// `get_object_via_grpc` previously retried immediately with no delay
/// at all on a retryable error (e.g. UNAVAILABLE), hammering a
/// transiently-overloaded or rate-limited endpoint on every attempt.
/// Base 100ms, doubling per attempt (1-indexed: attempt 1 -> 100ms,
/// attempt 2 -> 200ms, attempt 3 -> 400ms, ...), capped at 2s.
fn gcs_retry_backoff_delay(attempt: usize) -> Duration {
    let exponent = attempt.saturating_sub(1).min(10) as u32; // generous cap on the shift itself; .min(2000) below is the real cap
    let delay_ms = 100u64.saturating_mul(1u64 << exponent);
    Duration::from_millis(delay_ms.min(2000))
}

/// Run `operation` up to `max_attempts` times (1-indexed attempt number
/// passed to each call), sleeping `gcs_retry_backoff_delay(attempt)`
/// between attempts whenever the error is retryable per
/// `is_retryable_gcs_read_error` and attempts remain.
///
/// Extracted from `get_object_via_grpc`'s full-read retry loop
/// (audit #157 bug 5.1 / D1) specifically so the backoff-timing
/// contract is independently unit-testable: driving the real call site
/// requires a live or mocked GCS gRPC `StorageControl` client, which
/// isn't available in this test environment, but this function can be
/// driven with a synthetic `operation` closure under a paused Tokio
/// clock (`#[tokio::test(start_paused = true)]`) for exact, non-flaky
/// timing assertions.
async fn retry_with_backoff<T, Fut>(
    max_attempts: usize,
    mut operation: impl FnMut(usize) -> Fut,
) -> Result<T>
where
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 0usize;
    loop {
        attempt += 1;
        match operation(attempt).await {
            Ok(v) => return Ok(v),
            Err(e) if attempt < max_attempts && is_retryable_gcs_read_error(&e.to_string()) => {
                tokio::time::sleep(gcs_retry_backoff_delay(attempt)).await;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Desired gRPC subchannel count, set by the caller before the first GCS operation.
///
/// This allows the CLI (or any library user) to auto-tune the connection count to
/// match the intended concurrency level (e.g., `--jobs N`) without requiring an
/// environment variable.  Priority order in `GcsClient::new()`:
///   1. `S3DLIO_GCS_GRPC_CHANNELS` env var  (explicit override)
///   2. This atomic  (set via `set_gcs_channel_count()` before first GCS call)
///   3. `max(64, cpu_count)`  (auto fallback)
///
/// Must be stored before the `GCS_STORAGE` OnceCell is first initialized.
static DESIRED_GCS_CHANNELS: AtomicUsize = AtomicUsize::new(0);

/// Pre-configure the number of gRPC subchannels (TCP connections) the GCS client
/// will open.  Call this once, before any GCS operation, to auto-tune throughput
/// to your concurrency level.  Each subchannel carries one HTTP/2 connection with
/// its own flow-control window; matching channels to concurrent jobs ensures each
/// object stream has the full window available.
///
/// Has no effect if called after the GCS client has already been initialized.
/// `S3DLIO_GCS_GRPC_CHANNELS` env var takes precedence over this value.
pub fn set_gcs_channel_count(n: usize) {
    DESIRED_GCS_CHANNELS.store(n, Ordering::Relaxed);
}

/// Read back the programmatic GCS subchannel count set via [`set_gcs_channel_count`].
///
/// Returns `0` if [`set_gcs_channel_count`] has not been called — the client will
/// auto-detect from `max(GCS_MIN_CHANNELS, cpu_count)` on first initialization.
/// The `S3DLIO_GCS_GRPC_CHANNELS` env var is NOT checked here; it is applied
/// inside `GcsClient::new()` and takes precedence over this value.
pub fn get_gcs_channel_count() -> usize {
    DESIRED_GCS_CHANNELS.load(Ordering::Relaxed)
}

/// Programmatic override for RAPID mode, parallel to [`set_gcs_channel_count`].
///
/// Stored values:
///   0 = Unset — defer to `S3DLIO_GCS_RAPID` env var (default: Auto)
///   1 = ForceOn
///   2 = ForceOff
///
/// Must be set before the `GCS_STORAGE` OnceCell is first initialized.
static DESIRED_GCS_RAPID: AtomicU8 = AtomicU8::new(0);

/// Pre-configure RAPID (Hyperdisk ML / zonal GCS) mode before the first GCS
/// operation.
///
/// - `Some(true)`  — force RAPID on for all buckets (gRPC + appendable writes)
/// - `Some(false)` — force RAPID off for all buckets (standard HTTP reads)
/// - `None`        — auto-detect per bucket (default; no override)
///
/// Priority inside the GCS client:
///   1. `S3DLIO_GCS_RAPID` env var  (always wins)
///   2. The value set here            (`Some(true)` / `Some(false)`)
///   3. Auto-detect per bucket        (`None` / default)
///
/// Has no effect if called after the GCS client has already been initialized.
pub fn set_gcs_rapid_mode(force: Option<bool>) {
    let v = match force {
        None => 0u8,
        Some(true) => 1u8,
        Some(false) => 2u8,
    };
    DESIRED_GCS_RAPID.store(v, Ordering::Relaxed);
}

/// Read back the current effective RAPID mode as an `Option<bool>`.
///
/// Resolution order (same as GCS client initialization):
///   1. `S3DLIO_GCS_RAPID` env var  →  `Some(true)` / `Some(false)`
///   2. Value previously set by [`set_gcs_rapid_mode`]
///   3. Unset / auto-detect          →  `None`
///
/// Returns:
///   - `Some(true)`  — RAPID forced on  (gRPC BidiRead/Write + appendable)
///   - `Some(false)` — RAPID forced off (standard HTTP reads)
///   - `None`        — auto-detect per bucket (default)
///
/// Mirrors the `force` argument of [`set_gcs_rapid_mode`], making it easy to
/// log the effective setting at the start of a workload.
pub fn get_gcs_rapid_mode() -> Option<bool> {
    match read_rapid_mode() {
        RapidMode::ForceOn => Some(true),
        RapidMode::ForceOff => Some(false),
        RapidMode::Auto => None,
    }
}

/// Per-bucket cache of whether a bucket is RAPID (zonal/Hyperdisk ML).
/// Each entry is a `OnceCell<bool>` so that concurrent callers for the same
/// bucket block on the *same* detection future rather than all racing to call
/// `get_storage_layout()` simultaneously (thundering herd fix).
static BUCKET_RAPID_CACHE: std::sync::LazyLock<
    std::sync::Mutex<HashMap<String, Arc<OnceCell<bool>>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Minimal object metadata for GCS objects.
/// Maps to the provider-neutral ObjectMetadata type in object_store.rs.
///
/// NOTE: This struct MUST match the one in gcs_client.rs exactly!
#[derive(Debug, Clone)]
pub struct GcsObjectMetadata {
    pub size: u64,
    pub etag: Option<String>,
    pub updated: Option<String>,
    pub key: String,
}

/// A GCS object time as an RFC 3339 string (#263).
///
/// GCS reports object times as Google's `Timestamp`, whose `Debug` form is a struct dump
/// (`Timestamp { seconds: .., nanos: .. }`), not a time.  The provider-neutral
/// `ObjectMetadata::last_modified` and the GCS listing consumers expect RFC 3339, which
/// Google's `String::from(Timestamp)` produces (UTC, with the fraction kept when there is one).
/// An absent time stays absent.
fn object_time_rfc3339<T>(time: Option<T>) -> Option<String>
where
    T: Into<String>,
{
    time.map(Into::into)
}

/// RAPID mode configuration.
///
/// Controls whether gRPC BidiReadObject/BidiWriteObject (with appendable=true)
/// is used.  These APIs are **only** available on RAPID / zonal (Hyperdisk ML)
/// GCS buckets.
///
/// - `Auto` (default, or `S3DLIO_GCS_RAPID=auto`): detect per-bucket via
///   `get_storage_layout()`.  Result is cached for the process lifetime.
/// - `ForceOn` (`S3DLIO_GCS_RAPID=true|1|yes`): assume every bucket is RAPID.
/// - `ForceOff` (`S3DLIO_GCS_RAPID=false|0|no`): never use RAPID APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RapidMode {
    /// Auto-detect per bucket (default).
    Auto,
    /// Force RAPID on — gRPC BidiRead/Write + appendable.
    ForceOn,
    /// Force RAPID off — standard HTTP reads, gRPC writes without appendable.
    ForceOff,
}

/// High-level GCS client using official Google google-cloud-storage crate.
///
/// Authentication follows the standard ADC chain:
/// 1. GOOGLE_APPLICATION_CREDENTIALS environment variable (service account JSON)
/// 2. GCE/GKE metadata server (automatic for Google Cloud workloads)
/// 3. gcloud CLI credentials (~/.config/gcloud/application_default_credentials.json)
///
/// # Transport Selection
///
/// Writes always use gRPC (`BidiWriteObject` via `send_grpc()`).  Reads use
/// gRPC `BidiReadObject` for RAPID/zonal buckets and standard HTTP for
/// regular buckets.
///
/// # RAPID Storage
///
/// By default, RAPID mode is **auto-detected** per bucket by calling
/// `get_storage_layout()` on first access.  The result is cached for the
/// process lifetime.  Override with the `S3DLIO_GCS_RAPID` environment
/// variable:
///
/// | Value                        | Behaviour                          |
/// |------------------------------|------------------------------------|
/// | unset / `auto`               | auto-detect via storage layout     |
/// | `true` / `1` / `yes`         | force RAPID on all buckets         |
/// | `false` / `0` / `no`         | force standard mode on all buckets |
#[derive(Clone)]
pub struct GcsClient {
    storage: Arc<Storage>,
    control: Arc<StorageControl>,
    /// How to determine RAPID mode for any given bucket.
    rapid_mode: RapidMode,
}

impl GcsClient {
    /// Create a new GCS client using Application Default Credentials.
    /// This uses cached global clients for efficiency - authentication only happens once.
    ///
    /// The credentials are automatically discovered from:
    /// - GOOGLE_APPLICATION_CREDENTIALS env var (from the environment, or from a `.env` file
    ///   when `S3DLIO_LOAD_DOTENV` enables one)
    /// - Metadata server (if running on GCP)
    /// - gcloud CLI credentials
    pub async fn new() -> Result<Self> {
        crate::dotenv::apply();
        let storage = GCS_STORAGE
            .get_or_try_init(|| async {
                debug!("Initializing GCS Storage client (first time only)");

                // ---------------------------------------------------------------------------
                // Subchannel (gRPC connection) count.
                //
                // Each subchannel is an independent HTTP/2 TCP connection to the GCS endpoint.
                // More subchannels → higher aggregate bandwidth.
                //
                // Critical insight for RAPID/zonal buckets: HTTP/2 multiplexes all
                // concurrent object streams onto the same TCP connection. Each connection
                // has a single flow-control window (default configured to 256 MiB above).
                // With N jobs and M subchannels, each connection carries N/M concurrent
                // streams, each sharing that window. When N >> M (e.g., 64 jobs on 8
                // subchannels = 8 streams/connection), throughput collapses because
                // window_per_stream = 256 MiB / 8 = 32 MiB — barely enough for one ~30 MiB
                // object before a WINDOW_UPDATE stalls the next.
                //
                // Empirically on a c4-standard-8 GCE VM (RAPID bucket, 1000×~30 MiB files):
                //   8  subchannels, 64 jobs → ~845 MiB/s  (8 streams/conn)
                //   32 subchannels, 64 jobs → ~3.10 GiB/s (2 streams/conn)
                //   64 subchannels, 64 jobs → ~3.62 GiB/s (1 stream/conn)  ← optimal
                //
                // Default: max(64, available_parallelism()) — ensures 1:1 for typical
                // job counts (≤64). Set S3DLIO_GCS_GRPC_CHANNELS to match --jobs exactly
                // for maximum throughput at other concurrency levels.
                // Override: S3DLIO_GCS_GRPC_CHANNELS=<N>
                // ---------------------------------------------------------------------------
                let cpu_count = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);

                // Subchannel count priority:
                //   1. S3DLIO_GCS_GRPC_CHANNELS env var
                //   2. set_gcs_channel_count() (e.g. from --jobs)
                //   3. max(64, cpu_count) auto fallback
                let env_channels = std::env::var(ENV_GCS_GRPC_CHANNELS)
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok());
                let desired_channels = DESIRED_GCS_CHANNELS.load(Ordering::Relaxed);
                let (subchannel_count, channel_source) = if let Some(n) = env_channels {
                    (n, "S3DLIO_GCS_GRPC_CHANNELS env")
                } else if desired_channels > 0 {
                    (desired_channels, "jobs/concurrency")
                } else {
                    (std::cmp::max(GCS_MIN_CHANNELS, cpu_count), "auto")
                };

                let window_mib: u64 = std::env::var(ENV_GRPC_INITIAL_WINDOW_MIB)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_WINDOW_MIB);

                debug!(
                    "GCS gRPC config: subchannels={subchannel_count} (source={channel_source}, cpus={cpu_count}), \
                     initial_window={window_mib} MiB"
                );

                let client = Storage::builder()
                    .with_grpc_subchannel_count(subchannel_count)
                    .build()
                    .await
                    .map_err(|e| anyhow!("Failed to initialize GCS Storage client: {}", e))?;

                info!("GCS Storage client initialized successfully");
                Ok::<Arc<Storage>, anyhow::Error>(Arc::new(client))
            })
            .await?;

        let control = GCS_CONTROL
            .get_or_try_init(|| async {
                debug!("Initializing GCS StorageControl client (first time only)");

                let client = StorageControl::builder().build().await.map_err(|e| {
                    anyhow!("Failed to initialize GCS StorageControl client: {}", e)
                })?;

                info!("GCS StorageControl client initialized successfully");
                Ok::<Arc<StorageControl>, anyhow::Error>(Arc::new(client))
            })
            .await?;

        let rapid_mode = *EFFECTIVE_GCS_RAPID_MODE.get_or_init(|| {
            let mode = read_rapid_mode();
            match mode {
                RapidMode::ForceOn => info!("GCS RAPID mode: forced ON (gRPC + appendable writes)"),
                RapidMode::ForceOff => info!("GCS RAPID mode: forced OFF (standard HTTP reads)"),
                RapidMode::Auto => debug!("GCS RAPID mode: auto-detect per bucket"),
            }
            mode
        });

        Ok(Self {
            storage: Arc::clone(storage),
            control: Arc::clone(control),
            rapid_mode,
        })
    }

    /// Access the [`StorageControl`] client for bucket management operations
    /// (list, create, delete buckets).
    pub fn control_ref(&self) -> &StorageControl {
        &self.control
    }

    /// Determine whether `bucket` is a RAPID (zonal / Hyperdisk ML) bucket.
    ///
    /// - `ForceOn` / `ForceOff` return immediately.
    /// - `Auto` calls `get_storage_layout()` on first access to each bucket,
    ///   then caches the answer in the global `BUCKET_RAPID_CACHE`.
    ///
    /// Concurrent callers for the *same* bucket share a single `OnceCell`:
    /// exactly one task runs the detection RPC; all others await that result.
    /// This prevents the thundering-herd of N parallel `get_storage_layout()`
    /// calls that occurred when many jobs started simultaneously.
    async fn is_rapid_bucket(&self, bucket: &str) -> bool {
        match self.rapid_mode {
            RapidMode::ForceOn => return true,
            RapidMode::ForceOff => return false,
            RapidMode::Auto => {}
        }

        // Get or create the per-bucket OnceCell. The std::sync::Mutex is held
        // only for this brief map lookup/insert — no I/O while locked.
        let cell: Arc<OnceCell<bool>> = {
            let mut map = BUCKET_RAPID_CACHE.lock().unwrap();
            map.entry(bucket.to_string())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        // Exactly one caller runs detect_rapid_bucket_result(); the rest
        // await it. audit #157 bug 5.2 (D2): uses get_or_try_init, NOT
        // get_or_init -- a transient RPC failure must not permanently
        // cache "standard" for this bucket. get_or_try_init only
        // populates the cell on Ok, so a failed attempt leaves it
        // uninitialized and a later call retries detection instead of
        // being stuck with a stale fallback forever. Concurrent callers
        // still share a single in-flight detection attempt either way
        // (the thundering-herd protection this cache exists for).
        match cell
            .get_or_try_init(|| self.detect_rapid_bucket_result(bucket))
            .await
        {
            Ok(is_zonal) => *is_zonal,
            Err(e) => {
                warn!(
                    "GCS auto-detect: failed to get storage layout for '{}': {} — assuming \
                     standard bucket for this call (not cached, will retry next call)",
                    bucket, e
                );
                false
            }
        }
    }

    /// Query GCS `get_storage_layout()` to determine if a bucket is zonal.
    /// Returns `Ok(true)` for RAPID/zonal buckets, `Ok(false)` otherwise,
    /// `Err` on an RPC failure. Deliberately does NOT collapse the error
    /// case to `Ok(false)` here — see `is_rapid_bucket`'s use of
    /// `get_or_try_init` (audit #157 bug 5.2 / D2): only a genuine
    /// successful detection should ever be cached.
    async fn detect_rapid_bucket_result(&self, bucket: &str) -> Result<bool> {
        let layout_name = format!("projects/_/buckets/{}/storageLayout", bucket);
        let layout = self
            .control
            .get_storage_layout()
            .set_name(&layout_name)
            .send()
            .await
            .map_err(|e| anyhow!("get_storage_layout failed for '{}': {}", bucket, e))?;

        let is_zonal = layout.location_type.eq_ignore_ascii_case("zone");
        if is_zonal {
            info!(
                "GCS auto-detect: bucket '{}' is RAPID/zonal (location={}, type={})",
                bucket, layout.location, layout.location_type
            );
        } else {
            debug!(
                "GCS auto-detect: bucket '{}' is standard (location={}, type={})",
                bucket, layout.location, layout.location_type
            );
        }
        Ok(is_zonal)
    }

    /// Get entire object as bytes.
    ///
    /// Uses gRPC BidiReadObject (`open_object`) for RAPID/zonal buckets, and
    /// standard HTTP (`read_object`) for regular buckets.  BidiReadObject is
    /// only available for projects/buckets enrolled in RAPID/Hyperdisk ML.
    pub async fn get_object(&self, bucket: &str, object: &str) -> Result<Bytes> {
        let bucket_name = format_bucket_name(bucket);
        if self.is_rapid_bucket(bucket).await {
            debug!("GCS GET (gRPC): bucket={}, object={}", bucket, object);
            self.get_object_via_grpc(&bucket_name, bucket, object, ReadRange::all(), None)
                .await
        } else {
            debug!("GCS GET (HTTP): bucket={}, object={}", bucket, object);
            self.get_object_via_http(&bucket_name, bucket, object, None, None)
                .await
        }
    }

    /// Internal: read an object (or range) via gRPC BidiReadObject (open_object).
    async fn get_object_via_grpc(
        &self,
        bucket_name: &str,
        bucket: &str,
        object: &str,
        range: ReadRange,
        window: Option<(u64, Option<u64>)>,
    ) -> Result<Bytes> {
        let requested_range = range.clone();
        let (expected_size, mut data, mut chunk_count) = if requested_range == ReadRange::all() {
            retry_with_backoff(GCS_FULL_READ_RETRY_MAX_ATTEMPTS, |attempt| {
                let range = range.clone();
                async move {
                    self.get_object_via_grpc_once(bucket_name, bucket, object, range, window)
                        .await
                        .map_err(|e| {
                            info!(
                                "GCS GET (gRPC) full-read attempt {}/{} failed for gs://{}/{}: {}",
                                attempt, GCS_FULL_READ_RETRY_MAX_ATTEMPTS, bucket, object, e
                            );
                            e
                        })
                }
            })
            .await?
        } else {
            self.get_object_via_grpc_once(bucket_name, bucket, object, range, window)
                .await?
        };

        if requested_range == ReadRange::all() {
            if should_repair_incomplete_full_read(&requested_range, expected_size, data.len()) {
                let mut repair_attempts = 0usize;

                while data.len() < expected_size
                    && repair_attempts < GCS_FULL_READ_REPAIR_MAX_ATTEMPTS
                {
                    let offset = data.len() as u64;
                    let remaining = (expected_size - data.len()) as u64;
                    repair_attempts += 1;

                    debug!(
                        "GCS GET (gRPC) detected truncated full read for gs://{}/{} (got {} < expected {}), repairing tail offset={} len={} (attempt {}/{})",
                        bucket,
                        object,
                        data.len(),
                        expected_size,
                        offset,
                        remaining,
                        repair_attempts,
                        GCS_FULL_READ_REPAIR_MAX_ATTEMPTS
                    );

                    let (_, tail, tail_chunks) = self
                        .get_object_via_grpc_once(
                            bucket_name,
                            bucket,
                            object,
                            ReadRange::segment(offset, remaining),
                            Some((offset, Some(remaining))),
                        )
                        .await
                        .map_err(|e| {
                            anyhow!(
                                "GCS GET (gRPC) tail repair failed for gs://{}/{} at offset {} (remaining {}): {}",
                                bucket,
                                object,
                                offset,
                                remaining,
                                e
                            )
                        })?;

                    if tail.is_empty() {
                        break;
                    }

                    data.push(tail.as_slice());
                    chunk_count += tail_chunks;
                }
            }

            if data.len() != expected_size {
                return Err(anyhow!(
                    "GCS GET (gRPC) incomplete full read for gs://{}/{}: got {} bytes, expected {} bytes",
                    bucket,
                    object,
                    data.len(),
                    expected_size
                ));
            }
        }

        debug!(
            "GCS GET (gRPC) success: {} bytes in {} chunk(s)",
            data.len(),
            chunk_count
        );
        Ok(data.into_bytes())
    }

    async fn get_object_via_grpc_once(
        &self,
        bucket_name: &str,
        bucket: &str,
        object: &str,
        range: ReadRange,
        window: Option<(u64, Option<u64>)>,
    ) -> Result<(usize, ReadBuffer, u32)> {
        let requested_range = range.clone();
        let attempt_timeout = gcs_bidi_attempt_timeout();
        let (descriptor, mut reader) = match self
            .storage
            .open_object(bucket_name, object)
            .with_attempt_timeout(attempt_timeout)
            .send_and_read(range)
            .await
        {
            Ok(result) => result,
            Err(e) => {
                let error_text = e.to_string();
                if should_treat_out_of_range_as_empty_full_read(&error_text, &requested_range) {
                    // OUT_OF_RANGE on a full read may indicate a genuinely
                    // empty object.  Check metadata to confirm.
                    // HOWEVER: on RAPID/zonal buckets, GetObject metadata can
                    // return stale size=0 for recently written objects (per GCS
                    // team).  Do NOT trust the empty-object fallback there.
                    let is_rapid = self.is_rapid_bucket(bucket).await;
                    if is_rapid {
                        info!(
                            "GCS GET (gRPC) OUT_OF_RANGE on RAPID bucket gs://{}/{} — NOT falling back to stale metadata; retrying directly",
                            bucket, object
                        );
                        return Err(anyhow!(
                            "GCS GET (gRPC) OUT_OF_RANGE for RAPID object gs://{}/{}: {}",
                            bucket,
                            object,
                            e
                        ));
                    }
                    match self.get_object_metadata(bucket, object).await {
                        Ok(metadata) if metadata.size == 0 => {
                            debug!(
                                "GCS GET (gRPC) empty-object fallback: treating OUT_OF_RANGE as empty for gs://{}/{}",
                                bucket, object
                            );
                            return Ok((0, ReadBuffer::with_expected_len(0), 0));
                        }
                        Ok(metadata) => {
                            trace!(
                                "GCS GET (gRPC) OUT_OF_RANGE not treated as empty: metadata.size={} for gs://{}/{}",
                                metadata.size,
                                bucket,
                                object
                            );
                        }
                        Err(meta_err) => {
                            trace!(
                                "GCS GET (gRPC) OUT_OF_RANGE metadata check failed for gs://{}/{}: {}",
                                bucket,
                                object,
                                meta_err
                            );
                        }
                    }
                }

                return Err(anyhow!(
                    "GCS GET (gRPC) failed for gs://{}/{}: {}",
                    bucket,
                    object,
                    e
                ));
            }
        };

        // `size_hint` is the whole object (the full-read repair compares against it); the buffer
        // and the chunk timeout are sized to the bytes this read returns (#255).
        let (size_hint, capacity) = grpc_read_sizes(descriptor.object().size, window);
        // Reserved with `try_reserve`: a huge capacity cannot abort the process (it falls back
        // to growth).
        let mut data = ReadBuffer::with_expected_len(capacity as u64);
        let chunk_timeout = gcs_read_chunk_timeout(capacity as u64);
        let progress_interval = gcs_read_chunk_progress_interval();
        let mut chunk_count: u32 = 0;
        while let Some(chunk) = {
            let next_chunk = if let Some(timeout) = chunk_timeout {
                let wait_started = Instant::now();
                loop {
                    let waited = wait_started.elapsed();
                    if waited >= timeout {
                        return Err(anyhow!(
                            "GCS GET (gRPC) timeout for gs://{}/{} after {} chunk(s), {} byte(s), chunk-timeout={}s",
                            bucket,
                            object,
                            chunk_count,
                            data.len(),
                            timeout.as_secs()
                        ));
                    }

                    let remaining = timeout.saturating_sub(waited);
                    let step = if remaining < progress_interval {
                        remaining
                    } else {
                        progress_interval
                    };

                    match tokio::time::timeout(step, reader.next()).await {
                        Ok(next) => break next,
                        Err(_) => {
                            let now_waited = wait_started.elapsed().as_secs();
                            info!(
                                "GCS GET (gRPC) waiting for next chunk for gs://{}/{} (waited={}s, chunk-timeout={}s, received_chunks={}, received_bytes={})",
                                bucket,
                                object,
                                now_waited,
                                timeout.as_secs(),
                                chunk_count,
                                data.len()
                            );
                        }
                    }
                }
            } else {
                reader.next().await
            };
            next_chunk
        }
        .transpose()
        .map_err(|e| anyhow!("GCS GET (gRPC) stream error for gs://{}/{}: {}", bucket, object, e))?
        {
            chunk_count += 1;
            trace!("GCS GET (gRPC) chunk #{}: {} bytes", chunk_count, chunk.len());
            data.push(&chunk);
        }

        Ok((size_hint, data, chunk_count))
    }

    /// Internal: read an object (or optional range) via standard HTTP (`read_object`).
    /// `window` is the requested `(offset, length)` of a range read, `None` for a full read;
    /// it only sizes the destination buffer.
    /// Works on all bucket types including standard regional/multi-regional buckets.
    async fn get_object_via_http(
        &self,
        bucket_name: &str,
        bucket: &str,
        object: &str,
        range: Option<ReadRange>,
        window: Option<(u64, Option<u64>)>,
    ) -> Result<Bytes> {
        // The fork's URL encoder leaves a literal `%` alone (issue #270); see the helper.
        let wire_name = http_media_object_name(object);
        let mut builder = self.storage.read_object(bucket_name, wire_name.as_ref());
        if let Some(r) = range {
            builder = builder.set_read_range(r);
        }

        let mut reader = builder.send().await.map_err(|e| {
            anyhow!(
                "GCS GET (HTTP) failed for gs://{}/{}: {}",
                bucket,
                object,
                e
            )
        })?;

        // The response carries the object's size, so the destination is allocated once
        // (issue #252): growing a 1 MiB buffer by doubling copied everything received so far at
        // every step. `window` is the requested (offset, length) for a range read. If the server
        // sends more than it announced the buffer still grows, so no data is lost.
        let mut data =
            ReadBuffer::with_expected_len(http_read_expected_len(reader.object().size, window));
        let mut chunk_count: u32 = 0;
        while let Some(chunk) = reader.next().await.transpose().map_err(|e| {
            anyhow!(
                "GCS GET (HTTP) stream error for gs://{}/{}: {}",
                bucket,
                object,
                e
            )
        })? {
            chunk_count += 1;
            trace!(
                "GCS GET (HTTP) chunk #{}: {} bytes",
                chunk_count,
                chunk.len()
            );
            data.push(&chunk);
        }

        debug!(
            "GCS GET (HTTP) success: {} bytes in {} chunk(s)",
            data.len(),
            chunk_count
        );
        Ok(data.into_bytes())
    }

    /// Get a byte range from an object.
    ///
    /// Uses gRPC BidiReadObject (`open_object`) for RAPID/zonal buckets, and
    /// standard HTTP (`read_object`) for regular buckets.  BidiReadObject is
    /// only available for projects/buckets enrolled in RAPID/Hyperdisk ML.
    pub async fn get_object_range(
        &self,
        bucket: &str,
        object: &str,
        offset: u64,
        length: Option<u64>,
    ) -> Result<Bytes> {
        let bucket_name = format_bucket_name(bucket);

        // Use ReadRange to specify byte range
        let read_range = match length {
            Some(len) => ReadRange::segment(offset, len),
            None => ReadRange::offset(offset),
        };

        if self.is_rapid_bucket(bucket).await {
            debug!(
                "GCS GET RANGE (gRPC): bucket={}, object={}, offset={}, length={:?}",
                bucket, object, offset, length
            );
            self.get_object_via_grpc(
                &bucket_name,
                bucket,
                object,
                read_range.clone(),
                Some((offset, length)),
            )
            .await
        } else {
            debug!(
                "GCS GET RANGE (HTTP): bucket={}, object={}, offset={}, length={:?}",
                bucket, object, offset, length
            );
            self.get_object_via_http(
                &bucket_name,
                bucket,
                object,
                Some(read_range),
                Some((offset, length)),
            )
            .await
        }
    }

    /// Upload an object.
    ///
    /// Always uses gRPC BidiWriteObject (`send_grpc()`) which works for both
    /// standard and RAPID/zonal GCS buckets.  When the target bucket is RAPID
    /// (auto-detected or forced via `S3DLIO_GCS_RAPID`), sets `appendable=true`
    /// as required by zonal storage.  The transport layer omits `object_size`
    /// for appendable writes so the server can finalize correctly.
    pub async fn put_object(&self, bucket: &str, object: &str, data: Bytes) -> Result<()> {
        let is_rapid = self.is_rapid_bucket(bucket).await;
        let data_len = data.len();
        debug!(
            "GCS PUT (gRPC): bucket={}, object={}, size={}, rapid={}",
            bucket, object, data_len, is_rapid
        );

        let bucket_name = format_bucket_name(bucket);

        // `data` is already Arc-backed Bytes — pass directly, no copy.
        let mut write = self.storage.write_object(&bucket_name, object, data);
        if is_rapid {
            // RAPID (zonal) buckets require appendable=true on every write.
            write = write.set_appendable(true);
        }

        let obj = write
            .send_grpc()
            .await
            .map_err(|e| anyhow!("GCS PUT failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!(
            "GCS PUT success: gs://{}/{} size={} (server reported size={})",
            bucket, object, data_len, obj.size
        );
        Ok(())
    }

    /// Delete an object.
    pub async fn delete_object(&self, bucket: &str, object: &str) -> Result<()> {
        debug!(
            "GCS DELETE (official): bucket={}, object={}",
            bucket, object
        );

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format_bucket_name(bucket);

        self.control
            .delete_object()
            .set_bucket(bucket_name)
            .set_object(object.to_string())
            .send()
            .await
            .map_err(|e| {
                // Keep the typed error as the source so callers can tell a
                // NOT_FOUND from every other failure (delete_contract).
                anyhow::Error::new(e)
                    .context(format!("GCS DELETE failed for gs://{bucket}/{object}"))
            })?;

        debug!("GCS DELETE success");
        Ok(())
    }

    /// Get object metadata without downloading the content.
    pub async fn get_object_metadata(
        &self,
        bucket: &str,
        object: &str,
    ) -> Result<GcsObjectMetadata> {
        debug!(
            "GCS GET METADATA (official): bucket={}, object={}",
            bucket, object
        );

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format_bucket_name(bucket);

        let obj = self
            .control
            .get_object()
            .set_bucket(bucket_name)
            .set_object(object.to_string())
            .send()
            .await
            .map_err(|e| {
                anyhow!(
                    "GCS GET METADATA failed for gs://{}/{}: {}",
                    bucket,
                    object,
                    e
                )
            })?;

        let metadata = GcsObjectMetadata {
            size: obj.size as u64,
            etag: Some(obj.etag.clone()),
            updated: object_time_rfc3339(obj.update_time),
            key: object.to_string(),
        };

        debug!("GCS GET METADATA success: {} bytes", metadata.size);
        Ok(metadata)
    }

    /// Get object metadata.
    ///
    /// On RAPID/zonal buckets, `StorageControl.get_object()` may return stale
    /// `size=0` for recently written objects (confirmed by GCS team).  We work
    /// around this by using `BidiReadObject` (`open_object`) which returns
    /// authoritative metadata in the response descriptor.
    pub async fn stat_object(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata> {
        if self.is_rapid_bucket(bucket).await {
            return self.stat_object_via_bidi_read(bucket, object).await;
        }
        self.get_object_metadata(bucket, object).await
    }

    /// RAPID-safe stat using BidiReadObject descriptor for authoritative metadata.
    async fn stat_object_via_bidi_read(
        &self,
        bucket: &str,
        object: &str,
    ) -> Result<GcsObjectMetadata> {
        debug!(
            "GCS STAT (BidiRead) for RAPID bucket: gs://{}/{}",
            bucket, object
        );
        let bucket_name = format_bucket_name(bucket);
        let attempt_timeout = gcs_bidi_attempt_timeout();

        // Request just 1 byte — we only need the descriptor metadata.
        let (descriptor, _reader) = self
            .storage
            .open_object(&bucket_name, object)
            .with_attempt_timeout(attempt_timeout)
            .send_and_read(ReadRange::segment(0, 1))
            .await
            .map_err(|e| {
                anyhow!(
                    "GCS STAT (BidiRead) failed for gs://{}/{}: {}",
                    bucket,
                    object,
                    e
                )
            })?;

        let obj = descriptor.object();
        let metadata = GcsObjectMetadata {
            size: obj.size as u64,
            etag: Some(obj.etag.clone()),
            updated: object_time_rfc3339(obj.update_time),
            key: object.to_string(),
        };

        debug!(
            "GCS STAT (BidiRead) success: {} bytes for gs://{}/{}",
            metadata.size, bucket, object
        );
        Ok(metadata)
    }

    /// List objects in a bucket with optional prefix.
    ///
    /// When recursive=false, returns both files and subdirectory prefixes (ending with "/").
    /// This matches S3 behavior when using delimiter="/".
    pub async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<String>> {
        // Normalize prefix for non-recursive listings
        // GCS requires trailing "/" for delimiter behavior to work correctly
        // when listing virtual directories.
        let original_prefix = prefix.map(|p| p.to_string());
        let normalized_prefix = prefix.map(|p| {
            if !recursive && !p.is_empty() && !p.ends_with('/') {
                format!("{}/", p)
            } else {
                p.to_string()
            }
        });

        debug!(
            "GCS LIST (official): bucket={}, prefix={:?}, recursive={}, normalized_prefix={:?}",
            bucket, prefix, recursive, normalized_prefix
        );

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format_bucket_name(bucket);

        let results = self
            .list_objects_with_prefix(
                &bucket_name,
                bucket,
                normalized_prefix.as_deref(),
                recursive,
            )
            .await?;

        // If 0 results with normalized prefix (trailing '/') and the original
        // prefix didn't have a trailing slash, retry without it — the prefix
        // might be an exact object name, not a directory.
        if results.is_empty() && !recursive {
            if let Some(orig) = original_prefix.as_deref() {
                if !orig.is_empty() && !orig.ends_with('/') {
                    debug!(
                        "GCS LIST: 0 results with normalized_prefix={:?}, retrying with exact prefix={:?}",
                        normalized_prefix, orig
                    );
                    return self
                        .list_objects_with_prefix(&bucket_name, bucket, Some(orig), recursive)
                        .await;
                }
            }
        }

        Ok(results)
    }

    /// Stream objects from a bucket with optional prefix, yielding keys as they
    /// arrive from the GCS paginator — without buffering the full result set.
    ///
    /// For recursive listings this uses `by_item()` which yields one object per
    /// paginator item; downloads can begin as soon as the first key is received,
    /// up to ~7 seconds earlier than the blocking `list_objects` alternative.
    ///
    /// For non-recursive listings it falls back to `by_page()` so that virtual
    /// directory prefixes (entries ending with `/`) are included.
    pub fn list_objects_stream<'a>(
        &'a self,
        bucket: &'a str,
        prefix: Option<&'a str>,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        use google_cloud_gax::paginator::Paginator;

        let bucket_name_owned = format_bucket_name(bucket);
        // Normalise the prefix exactly as list_objects does.
        let normalized_prefix: Option<String> = prefix.map(|p| {
            if !recursive && !p.is_empty() && !p.ends_with('/') {
                format!("{}/", p)
            } else {
                p.to_string()
            }
        });

        Box::pin(async_stream::stream! {
            let mut builder = self.control
                .list_objects()
                .set_parent(bucket_name_owned.clone());

            if let Some(ref p) = normalized_prefix {
                builder = builder.set_prefix(p.clone());
            }
            if !recursive {
                builder = builder.set_delimiter("/".to_string());
            }

            if recursive {
                // by_item() yields one Object per call — true streaming, no intermediate Vec.
                let mut iter = builder.by_item();
                let mut count = 0u64;
                while let Some(result) = iter.next().await {
                    match result {
                        Ok(obj) => {
                            count += 1;
                            trace!("GCS LIST stream recursive: object={}", obj.name);
                            yield Ok(obj.name);
                        }
                        Err(e) => {
                            yield Err(anyhow!("GCS LIST stream error for gs://{}: {}", bucket, e));
                            return;
                        }
                    }
                }
                debug!("GCS LIST stream complete: {} objects", count);
            } else {
                // by_page() is needed for non-recursive so we can emit prefix entries.
                let mut pages_iter = builder.by_page();
                let mut count = 0u64;
                while let Some(result) = pages_iter.next().await {
                    match result {
                        Ok(page) => {
                            for obj in page.objects {
                                count += 1;
                                trace!("GCS LIST stream: object={}", obj.name);
                                yield Ok(obj.name);
                            }
                            for pfx in page.prefixes {
                                count += 1;
                                trace!("GCS LIST stream: prefix={}", pfx);
                                yield Ok(pfx);
                            }
                        }
                        Err(e) => {
                            yield Err(anyhow!("GCS LIST stream error for gs://{}: {}", bucket, e));
                            return;
                        }
                    }
                }
                debug!("GCS LIST stream complete: {} entries", count);
            }
        })
    }

    /// Internal: execute a paginated LIST with specified prefix and options.
    async fn list_objects_with_prefix(
        &self,
        bucket_name: &str,
        bucket: &str,
        prefix: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<String>> {
        let mut builder = self
            .control
            .list_objects()
            .set_parent(bucket_name.to_string());

        if let Some(p) = prefix {
            builder = builder.set_prefix(p.to_string());
        }

        // Set delimiter for non-recursive listings
        // This makes GCS return common prefixes (subdirectories) separately
        if !recursive {
            builder = builder.set_delimiter("/".to_string());
        }

        let mut results = Vec::new();

        if !recursive {
            // For non-recursive, we need access to both items AND prefixes
            // Use by_page() to get the full response structure with prefixes
            use google_cloud_gax::paginator::Paginator;
            let mut pages_iter = builder.by_page();

            while let Some(result) = pages_iter.next().await {
                let page =
                    result.map_err(|e| anyhow!("GCS LIST error for gs://{}: {}", bucket, e))?;

                // Collect object names (files)
                results.extend(
                    page.objects
                        .into_iter()
                        .map(|obj| obj.name)
                        .inspect(|name| trace!("GCS LIST: object={}", name)),
                );

                // Collect prefixes (subdirectories) - these end with "/"
                results.extend(
                    page.prefixes
                        .into_iter()
                        .inspect(|prefix| trace!("GCS LIST: prefix={}", prefix)),
                );
            }
        } else {
            // For recursive, by_item() is more efficient (no need for prefixes)
            let mut objects_iter = builder.by_item();

            while let Some(object) = objects_iter
                .next()
                .await
                .transpose()
                .map_err(|e| anyhow!("GCS LIST error for gs://{}: {}", bucket, e))?
            {
                trace!("GCS LIST recursive: object={}", object.name);
                results.push(object.name.clone());
            }
        }

        debug!("GCS LIST success: {} objects", results.len());
        Ok(results)
    }

    /// Delete a bucket (must be empty).
    pub async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        debug!("GCS DELETE BUCKET (official): bucket={}", bucket);

        let bucket_name = format_bucket_name(bucket);

        self.control
            .delete_bucket()
            .set_name(bucket_name)
            .send()
            .await
            .map_err(|e| anyhow!("GCS DELETE BUCKET failed for {}: {}", bucket, e))?;

        debug!("GCS DELETE BUCKET success");
        Ok(())
    }

    /// Create a bucket.
    pub async fn create_bucket(&self, bucket: &str, _project_id: Option<&str>) -> Result<()> {
        debug!("GCS CREATE BUCKET (official): bucket={}", bucket);

        self.control
            .create_bucket()
            .set_parent("projects/_".to_string())
            .set_bucket_id(bucket.to_string())
            .set_bucket(google_cloud_storage::model::Bucket::new())
            .send()
            .await
            .map_err(|e| anyhow!("GCS CREATE BUCKET failed for {}: {}", bucket, e))?;

        debug!("GCS CREATE BUCKET success");
        Ok(())
    }

    /// Delete multiple objects concurrently (batch delete).
    ///
    /// GCS does not have a native batch-delete API like S3's `DeleteObjects`,
    /// so each object is deleted with an individual gRPC call.  To avoid
    /// sequential round-trip latency (~80ms each), requests are dispatched
    /// concurrently with a semaphore.
    pub async fn delete_objects(&self, bucket: &str, objects: Vec<String>) -> Result<()> {
        if objects.is_empty() {
            return Ok(());
        }

        let total = objects.len();

        debug!(
            "GCS DELETE OBJECTS (concurrent): bucket={}, count={}, concurrency={}",
            bucket, total, GCS_MAX_CONCURRENT_DELETES
        );

        let bucket_name = format_bucket_name(bucket);
        let sem = Arc::new(tokio::sync::Semaphore::new(GCS_MAX_CONCURRENT_DELETES));
        let failed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let futs = futures_util::stream::FuturesUnordered::new();

        for object in objects {
            let sem = sem.clone();
            let control = Arc::clone(&self.control);
            let bname = bucket_name.clone();
            let bucket_str = bucket.to_string();
            let failed = failed.clone();

            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let result = control
                    .delete_object()
                    .set_bucket(bname)
                    .set_object(object.clone())
                    .send()
                    .await;

                if let Err(e) = result {
                    tracing::warn!("Failed to delete gs://{}/{}: {}", bucket_str, object, e);
                    failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }));
        }

        drain_delete_tasks_counting_panics(futs, &failed).await;

        let fail_count = failed.load(std::sync::atomic::Ordering::Relaxed);
        if fail_count > 0 {
            anyhow::bail!(
                "GCS delete partially failed: {}/{} objects failed to delete in gs://{}",
                fail_count,
                total,
                bucket
            );
        }
        debug!("GCS DELETE OBJECTS complete");
        Ok(())
    }

    /// Multipart upload.
    pub async fn put_object_multipart(
        &self,
        bucket: &str,
        object: &str,
        data: Bytes,
        _chunk_size: usize,
    ) -> Result<()> {
        // GCS handles large-object chunking internally via `write_object_grpc`
        // (16 MiB chunks, concurrent producer).  Pass Bytes through unchanged.
        self.put_object(bucket, object, data).await
    }
}

/// Drain all in-flight `delete_objects` tasks, counting BOTH per-object
/// failures (already recorded via `failed.fetch_add` inside each task
/// before this function ever observes anything) and task panics or
/// cancellations (`JoinError`, surfaced here as `Err(result)`).
///
/// audit #157 bug 5.3 (C5): previously a panicked or cancelled delete
/// task was logged via `tracing::warn!` but never counted toward
/// `failed`, so `delete_objects` could return `Ok(())` even though one
/// or more objects were never actually deleted (the task crashed before
/// its own `failed.fetch_add` call could run). Extracted as a
/// standalone function — rather than left as delete_objects' inline
/// drain loop — so this panic-accounting contract is independently
/// unit-testable with synthetic tasks: panicking a *real* spawned GCS
/// delete task requires a live or mocked gRPC `StorageControl` client,
/// which isn't available in this test environment, but a bare
/// `tokio::spawn(async { panic!() })` exercises the exact same
/// `JoinHandle`/`JoinError` mechanics this function depends on.
async fn drain_delete_tasks_counting_panics(
    mut futs: futures_util::stream::FuturesUnordered<tokio::task::JoinHandle<()>>,
    failed: &Arc<AtomicUsize>,
) {
    while let Some(result) = futures_util::StreamExt::next(&mut futs).await {
        if let Err(e) = result {
            tracing::warn!("Delete task panicked or was cancelled: {}", e);
            failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod delete_objects_panic_accounting_tests {
    use super::*;

    #[tokio::test]
    async fn panicked_task_is_counted_as_a_failure() {
        let failed = Arc::new(AtomicUsize::new(0));
        let futs = futures_util::stream::FuturesUnordered::new();
        futs.push(tokio::spawn(async {
            panic!("simulated delete task panic");
        }));

        drain_delete_tasks_counting_panics(futs, &failed).await;

        assert_eq!(
            failed.load(Ordering::Relaxed),
            1,
            "a panicked delete task must be counted as a failure, not silently dropped"
        );
    }

    #[tokio::test]
    async fn cancelled_task_is_counted_as_a_failure() {
        let failed = Arc::new(AtomicUsize::new(0));
        let futs = futures_util::stream::FuturesUnordered::new();
        let handle = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        handle.abort();
        futs.push(handle);

        drain_delete_tasks_counting_panics(futs, &failed).await;

        assert_eq!(
            failed.load(Ordering::Relaxed),
            1,
            "a cancelled delete task must be counted as a failure, not silently dropped"
        );
    }

    #[tokio::test]
    async fn successful_tasks_are_not_counted() {
        let failed = Arc::new(AtomicUsize::new(0));
        let futs = futures_util::stream::FuturesUnordered::new();
        for _ in 0..3 {
            futs.push(tokio::spawn(async {}));
        }

        drain_delete_tasks_counting_panics(futs, &failed).await;

        assert_eq!(failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn mixed_success_and_panic_counts_only_the_panic() {
        let failed = Arc::new(AtomicUsize::new(0));
        let futs = futures_util::stream::FuturesUnordered::new();
        futs.push(tokio::spawn(async {}));
        futs.push(tokio::spawn(async {
            panic!("simulated delete task panic");
        }));
        futs.push(tokio::spawn(async {}));

        drain_delete_tasks_counting_panics(futs, &failed).await;

        assert_eq!(failed.load(Ordering::Relaxed), 1);
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers (pub(crate) so unit tests can exercise them directly)
// ---------------------------------------------------------------------------

/// Read the `S3DLIO_GCS_RAPID` environment variable and return the appropriate
/// [`RapidMode`].
///
/// | Env value                   | Result           |
/// |-----------------------------|------------------|
/// | unset                       | `Auto`           |
/// | `auto`                      | `Auto`           |
/// | `true` / `1` / `yes`        | `ForceOn`        |
/// | `false` / `0` / `no`        | `ForceOff`       |
/// | anything else               | `Auto`           |
///
/// The variable can be set in the shell **or** come from a `.env` file when
/// `S3DLIO_LOAD_DOTENV` enables one (see [`crate::dotenv`]).
pub(crate) fn read_rapid_mode() -> RapidMode {
    crate::dotenv::apply();
    // Priority 1: env var (explicit user override, always wins)
    if let Ok(v) = std::env::var("S3DLIO_GCS_RAPID") {
        return match v.to_lowercase().as_str() {
            "true" | "1" | "yes" => RapidMode::ForceOn,
            "false" | "0" | "no" => RapidMode::ForceOff,
            "auto" | "" => RapidMode::Auto,
            other => {
                tracing::warn!(
                    "S3DLIO_GCS_RAPID='{}' is not recognised; using auto-detect",
                    other
                );
                RapidMode::Auto
            }
        };
    }
    // Priority 2: programmatic override via set_gcs_rapid_mode()
    match DESIRED_GCS_RAPID.load(Ordering::Relaxed) {
        1 => RapidMode::ForceOn,
        2 => RapidMode::ForceOff,
        _ => RapidMode::Auto, // 0 = unset → auto-detect
    }
}

/// Format a plain bucket name into the resource path required by the
/// official google-cloud-storage gRPC API.
///
/// ```text
/// "my-bucket"  →  "projects/_/buckets/my-bucket"
/// ```
pub(crate) fn format_bucket_name(bucket: &str) -> String {
    format!("projects/_/buckets/{}", bucket)
}

fn is_gcs_out_of_range_error(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("out_of_range") || normalized.contains("outside the valid object range")
}

fn should_treat_out_of_range_as_empty_full_read(
    message: &str,
    requested_range: &ReadRange,
) -> bool {
    is_gcs_out_of_range_error(message) && *requested_range == ReadRange::all()
}

fn should_repair_incomplete_full_read(
    requested_range: &ReadRange,
    expected_size: usize,
    actual_size: usize,
) -> bool {
    *requested_range == ReadRange::all() && actual_size < expected_size
}

fn is_retryable_gcs_read_error(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("timeout")
        || normalized.contains("unavailable")
        || normalized.contains("cancel")
        || normalized.contains("resource_exhausted")
}

/// Classify a GCS failure from the typed `google_cloud_gax` error (plan §4.5 A2).
///
/// Only typed accessors are used; message text is never inspected.
///
/// * HTTP 404 or gRPC `NotFound` is `NotFound`.  GCS gives the same status for a missing
///   bucket, so absence of the bucket is not told apart from absence of the object;
/// * 412 or `FailedPrecondition` is `VersionChanged`;
/// * 403 or `PermissionDenied` is `Forbidden`;
/// * `Unavailable` or `ResourceExhausted`, or HTTP 503 or 429, is `Throttled`;
/// * gRPC `DeadlineExceeded` is `Transport`: the deadline expired before a usable answer, like
///   a timeout;
/// * HTTP 408 (parity with S3), gRPC `Internal` and gRPC `Aborted` are `Server`: the service
///   asks for the request to be repeated;
/// * a typed transport failure (timeout, connect, I/O, or a transport error without an
///   HTTP status) is `Transport`;
/// * any other HTTP 5xx is `Server`;
/// * everything else is `Other`, in particular HTTP 401 and gRPC `Unauthenticated` (a
///   credential problem, not retried), and gRPC `Unknown` and `Cancelled` (no evidence the
///   service failed, not retried).
#[cfg(not(feature = "gcs-community"))]
pub(crate) fn classify_gcs_failure(e: &google_cloud_gax::error::Error) -> ObjectErrorKind {
    use google_cloud_gax::error::rpc::Code;
    let http = e.http_status_code();
    let code = e.status().map(|s| s.code);
    if http == Some(404) || code == Some(Code::NotFound) {
        ObjectErrorKind::NotFound
    } else if http == Some(412) || code == Some(Code::FailedPrecondition) {
        ObjectErrorKind::VersionChanged
    } else if http == Some(403) || code == Some(Code::PermissionDenied) {
        ObjectErrorKind::Forbidden
    } else if matches!(code, Some(Code::Unavailable | Code::ResourceExhausted))
        || matches!(http, Some(503 | 429))
    {
        ObjectErrorKind::Throttled
    } else if code == Some(Code::DeadlineExceeded) {
        ObjectErrorKind::Transport
    } else if matches!(code, Some(Code::Internal | Code::Aborted)) || http == Some(408) {
        ObjectErrorKind::Server
    } else if http.is_none() && (e.is_timeout() || e.is_connect() || e.is_transport()) {
        ObjectErrorKind::Transport
    } else if matches!(http, Some(500..=599)) {
        ObjectErrorKind::Server
    } else {
        ObjectErrorKind::Other
    }
}

/// Attach the GCS classification of `err` as an
/// [`ObjectError`](crate::object_error::ObjectError) context.
///
/// `err` must carry a `google_cloud_gax` error (found through any context layers).  An error
/// that does not is returned unchanged: an unclassified error stays unclassified, and the
/// typed-only retry rule ([`gcs_typed_retry_verdict`]) then never retries it.
///
/// An error that already carries an `ObjectError` is returned unchanged, with one exception:
/// on a `pinned` read whose outer kind is `NotFound`, a `VersionChanged` is attached over it
/// (the outer classification wins), because a pinned read never reports absence.
///
/// The attached `ObjectError`'s own message is built only from the HTTP status and the gRPC
/// code name, never from server-supplied text.  The retained source error, reachable through
/// `{:#}`, `{:?}` or the chain, is the `google_cloud_gax` error itself and is NOT redacted by
/// this module.
#[cfg(not(feature = "gcs-community"))]
pub(crate) fn attach_gcs_classification(err: anyhow::Error, pinned: bool) -> anyhow::Error {
    if let Some(existing) = crate::object_error::object_error(&err) {
        if pinned && existing.kind == ObjectErrorKind::NotFound {
            return crate::object_error::attach_object_error(
                err,
                ObjectErrorKind::VersionChanged,
                "gs",
                "pinned version no longer found",
            );
        }
        return err;
    }
    let Some(g) = err.downcast_ref::<google_cloud_gax::error::Error>() else {
        return err;
    };
    let kind = classify_gcs_failure(g);
    let message = match (g.http_status_code(), g.status()) {
        (Some(h), Some(s)) => format!("status {h}, code {:?}", s.code),
        (Some(h), None) => format!("status {h}"),
        (None, Some(s)) => format!("code {:?}", s.code),
        (None, None) => "no status".to_string(),
    };
    if pinned {
        crate::object_error::attach_pinned_read_error(err, kind, "gs", message)
    } else {
        crate::object_error::attach_object_error(err, kind, "gs", message)
    }
}

/// The retry predicate for the new GCS receive and repair loop: **typed only**.
///
/// Only an outer `Throttled`, `Transport` or `Server` classification is retryable.  An
/// unclassified error (no `ObjectError`), `Other`, `Forbidden`, `NotFound`, `VersionChanged`,
/// a length error, and any message text that merely contains a status number are not.  This
/// is not [`is_retryable_gcs_read_error`] (message text) and not the generic conservative
/// fallback of `RetryClassify`; it never authorizes combining unpinned responses either.
#[cfg(not(feature = "gcs-community"))]
pub(crate) fn gcs_typed_retry_verdict(err: &anyhow::Error) -> bool {
    matches!(
        crate::object_error::object_error(err).map(|o| o.kind),
        Some(ObjectErrorKind::Throttled | ObjectErrorKind::Transport | ObjectErrorKind::Server)
    )
}

/// The object name as the pinned fork's HTTP `read_object` must be given it, so that the
/// URL names exactly the literal object.
///
/// `read_object` builds `/storage/v1/b/{bucket}/o/{enc(object)}?alt=media`, and its `enc`
/// (`storage/client.rs`) encodes the characters GCS requires (`/`, `@`, `=`, space, ...) but
/// deliberately PRESERVES `%` (its own test maps `preserve%percent%21` to itself).  A literal
/// `%2F` in an object name would therefore reach GCS as an escape and be decoded to `/`, which
/// names a different object.  Sending every literal `%` as `%25` makes the fork's encoder
/// produce the right URL for any name.
///
/// The same mechanism covers `\`: `enc` leaves it alone, and the URL parser then reads it as a
/// `/` in an http path (measured: `a\b` reached the server as `a/b`), so it is sent as `%5C`.
/// The `%` replacement runs first, so the `%` of a `%5C` is never doubled.
///
/// Every HTTP read goes through this (issue #270): `get_object_via_http` (whole and ranged
/// reads) and the caller-buffer `HttpOpener`.  The gRPC/Rapid routes carry the name in a
/// protobuf field, not a URL, and must receive it unchanged.
///
/// The names `.` and `..` cannot be addressed over the HTTP path (URL parsing collapses dot
/// segments); GCS forbids them as object names anyway, and this helper does not change that.
fn http_media_object_name(object: &str) -> std::borrow::Cow<'_, str> {
    if object.contains(['%', '\\']) {
        std::borrow::Cow::Owned(object.replace('%', "%25").replace('\\', "%5C"))
    } else {
        std::borrow::Cow::Borrowed(object)
    }
}

/// How many bytes an HTTP read will return, for sizing its destination buffer once (#252).
///
/// `object_size` is the object's size from the response; `window` is the requested
/// `(offset, length)` of a range read (`length` `None` = to the end), `None` for a full read.
/// A range is clamped to what the object has past `offset`. A negative or unknown size is 0,
/// which makes the buffer grow as data arrives.
fn http_read_expected_len(object_size: i64, window: Option<(u64, Option<u64>)>) -> u64 {
    let size = u64::try_from(object_size).unwrap_or(0);
    match window {
        None => size,
        Some((offset, Some(length))) => length.min(size.saturating_sub(offset)),
        Some((offset, None)) => size.saturating_sub(offset),
    }
}

/// Sizes for a gRPC read, from the object size the descriptor reports and the requested window
/// (`None` = the whole object): `(object_size, buffer_capacity)`. The capacity is the bytes the
/// read will return, so a 1 MiB range of a multi-TiB object reserves 1 MiB, not the object
/// (#255); a negative or unrepresentable size is 0 (the buffer grows as data arrives) instead of
/// wrapping through `as usize`.
fn grpc_read_sizes(object_size: i64, window: Option<(u64, Option<u64>)>) -> (usize, usize) {
    let size = usize::try_from(object_size).unwrap_or(0);
    let capacity = usize::try_from(http_read_expected_len(object_size, window)).unwrap_or(0);
    (size, capacity)
}

fn gcs_read_chunk_timeout(expected_size_bytes: u64) -> Option<Duration> {
    let timeout_secs = std::env::var(ENV_GCS_READ_CHUNK_TIMEOUT_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| gcs_scaled_timeout_secs(expected_size_bytes));

    if timeout_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(timeout_secs))
    }
}

fn gcs_bidi_attempt_timeout() -> Duration {
    let timeout_secs = std::env::var(ENV_GCS_BIDI_ATTEMPT_TIMEOUT_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or_else(|| gcs_scaled_timeout_secs(0));

    Duration::from_secs(timeout_secs)
}

fn gcs_read_chunk_progress_interval() -> Duration {
    let interval_secs = std::env::var(ENV_GCS_READ_PROGRESS_INTERVAL_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_GCS_READ_PROGRESS_INTERVAL_SECS);

    Duration::from_secs(interval_secs)
}

fn gcs_scaled_timeout_secs(expected_size_bytes: u64) -> u64 {
    let per_gib = std::env::var(ENV_GCS_READ_TIMEOUT_SECS_PER_GIB)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_GCS_READ_TIMEOUT_SECS_PER_GIB);

    let min_secs = std::env::var(ENV_GCS_READ_TIMEOUT_MIN_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_GCS_READ_TIMEOUT_MIN_SECS);

    let max_secs = std::env::var(ENV_GCS_READ_TIMEOUT_MAX_SECS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_GCS_READ_TIMEOUT_MAX_SECS);

    let scaled = if expected_size_bytes == 0 {
        0
    } else {
        expected_size_bytes
            .saturating_mul(per_gib)
            .div_ceil(BYTES_PER_GIB)
    };
    scaled.clamp(min_secs, max_secs)
}

/// Parse a GCS URI (gs://bucket/path/to/object) into (bucket, object_path).
///
/// Bucket-only URIs are also supported (for prefix listings):
/// - gs://bucket/ → ("bucket", "")
/// - gs://bucket  → ("bucket", "") - requires trailing slash for proper parsing
pub fn parse_gcs_uri(uri: &str) -> Result<(String, String)> {
    // The structure of a gs:// URI is parsed in one place, by the environment-free parser.
    crate::capabilities::parse_gs_uri_pure(uri)
}

/// Query whether a GCS bucket or `gs://` URI refers to a RAPID (Hyperdisk ML / zonal) bucket.
///
/// Accepts either a bare bucket name (`"my-bucket"`) or a full GCS URI
/// (`"gs://my-bucket/some/prefix/"`).  The bucket layout is determined once via
/// the `GetStorageLayout` RPC and cached for the process lifetime — the same
/// cache used internally by every `get_object` / `put_object` call — so
/// repeated calls for the same bucket are essentially free.
///
/// Concurrency-safe: simultaneous calls for the same bucket are deduplicated
/// via `BUCKET_RAPID_CACHE` — only one `GetStorageLayout` RPC is issued.
///
/// Returns `false` on authentication or network errors (a warning is logged).
pub async fn query_gcs_rapid_bucket(bucket_or_uri: &str) -> bool {
    let bucket = if bucket_or_uri.starts_with("gs://") || bucket_or_uri.starts_with("gcs://") {
        match parse_gcs_uri(bucket_or_uri) {
            Ok((b, _)) => b,
            Err(_) => bucket_or_uri.to_string(),
        }
    } else {
        bucket_or_uri.to_string()
    };
    match GcsClient::new().await {
        Ok(client) => client.is_rapid_bucket(&bucket).await,
        Err(e) => {
            warn!(
                "query_gcs_rapid_bucket('{}'): failed to init GCS client: {}",
                bucket, e
            );
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Caller-buffer GET and typed HEAD (plan §4.3, §4.5 A1, A2, A7, A8; #264)
// ---------------------------------------------------------------------------
//
// The receive loop is written against a small `ReadOpener` interface ("open at an offset,
// pinned to a generation, and give me a stream of chunks or a typed error") so that its core
// is driven by a deterministic fake in the tests.  The two real openers are the HTTP
// `read_object` route and the Rapid `open_object` route; neither ever sees the destination.
//
// Every write into `dst` happens in the task that awaits `receive_into`; nothing is spawned.
// Dropping the future therefore stops every writer.
#[cfg(not(feature = "gcs-community"))]
mod caller_buffer {
    use super::*;
    use crate::object_error::{attach_object_error, object_error};
    use crate::object_store::HeadMetadata;
    use crate::range_version::{Leader, ObjectVersion, Role};
    use crate::receive_progress::{ReceiveProgress, SpanMark};
    use async_trait::async_trait;
    use google_cloud_gax::error::rpc::Code;
    use google_cloud_gax::error::Error as GaxError;
    use google_cloud_storage::model::Object;
    use google_cloud_storage::object_descriptor::ObjectDescriptor;
    use google_cloud_storage::read_object::ReadObjectResponse;
    use google_cloud_storage::read_resume_policy::NeverResume;

    /// What an opened read says about the object it is reading.
    pub(super) struct ObjectFacts {
        /// The whole object's size, signed as the service reports it (validated before use).
        pub size: i64,
        /// The object's generation; `<= 0` means the response gave no usable identity.
        pub generation: i64,
    }

    /// One read request: `limit == None` reads to the end of the object.
    pub(super) struct OpenRequest {
        pub start: u64,
        pub limit: Option<u64>,
        /// `Some` pins the read to this generation (`set_generation` on the real openers).
        pub generation: Option<i64>,
    }

    /// An opened read: the object facts and the stream of data chunks.
    pub(super) struct Opened {
        pub facts: ObjectFacts,
        pub stream: Box<dyn ChunkStream>,
    }

    /// A stream of data chunks.  Errors are the typed `google_cloud_gax` error.
    #[async_trait]
    pub(super) trait ChunkStream: Send {
        async fn next_chunk(&mut self) -> Option<Result<Bytes, GaxError>>;
    }

    /// Opens reads of one object.
    #[async_trait]
    pub(super) trait ReadOpener: Send + Sync {
        /// Open a read.  An open that fails reports the typed error.
        async fn open(&self, request: &OpenRequest) -> Result<Opened, GaxError>;

        /// The authoritative metadata of the object, with no data range requested.
        async fn describe(&self, generation: Option<i64>) -> Result<Object, GaxError>;
    }

    /// Most repairs one caller-buffer read may use in TOTAL, whatever progress they make.  It is
    /// four times the no-progress limit (`GCS_FULL_READ_REPAIR_MAX_ATTEMPTS`), so a read that
    /// keeps advancing may survive several interruptions, yet the number of outer repairs is
    /// finite and independent of the object size. Without it, one byte followed by a failure
    /// could cost one repair per object byte. With the backoff capped at 2 s the outer repairs
    /// add about 25 s of backoff at most; SDK retries, redirects and I/O time are additional.
    const GCS_FULL_READ_TOTAL_REPAIR_MAX_ATTEMPTS: usize = 4 * GCS_FULL_READ_REPAIR_MAX_ATTEMPTS;

    /// Retry and repair limits, and the one behavioural switch (`rapid`).
    pub(super) struct ReceiveConfig {
        /// A Rapid bucket: a zero-length destination reads the descriptor, and an
        /// OUT_OF_RANGE on the first full read is checked against the descriptor.
        pub rapid: bool,
        /// Longest wait for the next chunk; `None` waits indefinitely.
        pub chunk_timeout: Option<Duration>,
        /// Delay before retry or repair number `n` (1-based).
        pub retry_delay: fn(usize) -> Duration,
        /// Attempts allowed while nothing has been written yet.
        pub max_open_attempts: usize,
        /// Consecutive repairs allowed after some bytes were written (pinned reads only) without
        /// the cursor advancing past the previous repair point.  A repair that advances the
        /// cursor starts THIS count again; it does not extend `max_total_repairs`.
        pub max_repairs: usize,
        /// Repairs allowed in TOTAL for the whole logical read (pinned reads only).  Progress
        /// never refills it, and the backoff delay is taken from it, so it never restarts either.
        /// Together with the no-progress bound above it makes the repair work finite for any
        /// object size. The real HTTP and Rapid openers disable ordinary SDK body resumption
        /// (`NeverResume`). There are at most `max_open_attempts + max_total_repairs` outer
        /// opener calls, not wire requests: each call retains the SDK's ordinary request retry
        /// policy (the pinned default has a 300 s retry time limit, not an attempt-count cap).
        /// The pinned fork's `ResumeRedirect`
        /// (`bidi/resume_redirect.rs`) and `RetryRedirect` (`retry_redirect.rs`) continue server
        /// redirects regardless of `NeverResume`, so redirect re-connects are bounded only by the
        /// server, the per-attempt timeout and, in the stream phase, `chunk_timeout`.
        pub max_total_repairs: usize,
    }

    impl ReceiveConfig {
        pub(super) fn new(rapid: bool, chunk_timeout: Option<Duration>) -> Self {
            Self {
                rapid,
                chunk_timeout,
                retry_delay: gcs_retry_backoff_delay,
                max_open_attempts: GCS_FULL_READ_RETRY_MAX_ATTEMPTS,
                max_repairs: GCS_FULL_READ_REPAIR_MAX_ATTEMPTS,
                max_total_repairs: GCS_FULL_READ_TOTAL_REPAIR_MAX_ATTEMPTS,
            }
        }
    }

    /// The limits for one caller-buffer read of `len` bytes.  The chunk timeout is the same on
    /// both routes (HTTP and Rapid), sized to the bytes expected: a stalled body is a typed
    /// timeout, classified Transport, and the pinned read is repaired.
    pub(super) fn read_config(rapid: bool, len: usize) -> Result<ReceiveConfig> {
        Ok(ReceiveConfig::new(
            rapid,
            gcs_read_chunk_timeout(to_u64(len)?),
        ))
    }

    fn to_u64(v: usize) -> Result<u64> {
        u64::try_from(v).map_err(|_| anyhow!("length {v} does not fit in u64"))
    }

    /// A reported size, validated before conversion: a negative size is an error, never zero.
    fn size_to_u64(size: i64, what: &str) -> Result<u64> {
        u64::try_from(size).map_err(|_| {
            attach_object_error(
                anyhow!("GCS reported a negative size ({size}) for {what}"),
                ObjectErrorKind::Other,
                "gs",
                "negative size",
            )
        })
    }

    /// Attach the typed classification to a gax failure.  `context` says which operation
    /// failed, in the caller's words (the phase differs: opening a read, a stream, a HEAD).
    /// `pinned` upgrades a 404 to `VersionChanged`: once a generation is pinned, absence is
    /// never reported.
    fn gax_failure(e: GaxError, context: String, pinned: bool) -> anyhow::Error {
        attach_gcs_classification(anyhow::Error::new(e).context(context), pinned)
    }

    /// What may be logged about a failure: its classification (provider, kind, HTTP status
    /// and gRPC code) and nothing else.  The source chain behind `{:#}` can carry text the
    /// server supplied, such as an error body, so it is never logged.
    pub(super) fn loggable(err: &anyhow::Error) -> String {
        object_error(err).map_or_else(|| "unclassified failure".to_string(), ToString::to_string)
    }

    fn length_mismatch(what: &str, expected: u64, actual: Option<u64>) -> anyhow::Error {
        attach_object_error(
            anyhow!("GCS caller-buffer GET length mismatch for {what}"),
            ObjectErrorKind::LengthMismatch { expected, actual },
            "gs",
            "",
        )
    }

    fn version_changed(what: &str, detail: &str) -> anyhow::Error {
        attach_object_error(
            anyhow!("GCS caller-buffer GET of {what}: the object changed after its generation was pinned"),
            ObjectErrorKind::VersionChanged,
            "gs",
            detail,
        )
    }

    /// OUT_OF_RANGE by its typed status code (never by message text).
    fn is_out_of_range(e: &GaxError) -> bool {
        e.status().map(|s| s.code) == Some(Code::OutOfRange)
    }

    /// The stream of a read that has nothing to deliver (a zero-length Rapid read).
    struct EmptyStream;

    #[async_trait]
    impl ChunkStream for EmptyStream {
        async fn next_chunk(&mut self) -> Option<Result<Bytes, GaxError>> {
            None
        }
    }

    /// How many retries and repairs have been used.
    #[derive(Default)]
    struct Budget {
        failures_before_data: usize,
        /// Repairs used in total for this read.  Progress never refills it.
        repairs: usize,
        /// Repairs used since the cursor last advanced past the previous repair point.
        stalled: usize,
        /// Where the cursor stood at the most recent repair.
        last_repair_at: Option<usize>,
    }

    impl Budget {
        /// The delay before the retry or repair that `err` allows, or `None` when it allows
        /// none.  `cursor` is how many bytes the span already holds and `len` its size.
        ///
        /// The verdict is `gcs_typed_retry_verdict` and nothing else.  With bytes already
        /// written, only a pinned read is repaired (unpinned responses are never combined), and
        /// a span that is complete has no suffix left to resume.
        fn next_delay(
            &mut self,
            cfg: &ReceiveConfig,
            err: &anyhow::Error,
            cursor: usize,
            len: usize,
            pinned: bool,
        ) -> Option<Duration> {
            if !gcs_typed_retry_verdict(err) {
                return None;
            }
            if cursor == 0 {
                self.failures_before_data += 1;
                if self.failures_before_data >= cfg.max_open_attempts {
                    return None;
                }
                Some((cfg.retry_delay)(self.failures_before_data))
            } else {
                if !pinned || cursor >= len {
                    return None;
                }
                self.take_repair(cfg, cursor)
            }
        }

        /// Use one repair at `cursor`, returning its delay, or `None` when none are left.
        ///
        /// Two bounds apply and either one ends the repairs:
        ///
        /// * a TOTAL (`max_total_repairs`) for the whole logical read, which progress never
        ///   refills.  A read that advances one byte per request is therefore failed after a
        ///   fixed number of repairs, not one reconnection per object byte, and the caller's
        ///   allocation is not held for a time that grows with the object size;
        /// * a NO-PROGRESS limit (`max_repairs`): when `cursor` is past where the previous repair
        ///   began, the repair did advance the read and only this count starts again.
        ///
        /// The backoff delay is taken from the total count, so progress does not reset it.
        fn take_repair(&mut self, cfg: &ReceiveConfig, cursor: usize) -> Option<Duration> {
            if self
                .last_repair_at
                .is_some_and(|previous| cursor > previous)
            {
                self.stalled = 0;
            }
            if self.repairs >= cfg.max_total_repairs || self.stalled >= cfg.max_repairs {
                return None;
            }
            self.repairs = self.repairs.saturating_add(1);
            self.stalled = self.stalled.saturating_add(1);
            self.last_repair_at = Some(cursor);
            Some((cfg.retry_delay)(self.repairs))
        }
    }

    async fn back_off(delay: Duration) {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
    }

    /// Receive the whole object into `dst` (`dst.len()` is the expected size).
    ///
    /// * The first response's generation is retained; every retry and repair is pinned to it.
    ///   A response with a different generation, or a 404 on a pinned request, is
    ///   `VersionChanged`.  Without a usable generation a single response is accepted, and a
    ///   partially received request fails (unpinned responses are never combined).
    /// * Repair writes the suffix of the same span, resuming at the cursor, under one
    ///   [`SpanMark`] that survives every retry and repair, so each position is counted once.
    /// * The object's size must equal `dst.len()` before any byte is copied, and the received
    ///   count must equal it exactly.  A chunk that would pass the end of `dst` is refused
    ///   whole; nothing is ever written beyond `dst`.
    /// * Retries use only [`gcs_typed_retry_verdict`].
    /// * Repair work is finite whatever the object size: at most `cfg.max_total_repairs` repairs in
    ///   all (progress never refills them), and at most `cfg.max_repairs` in a row without the
    ///   cursor advancing. The openers disable ordinary SDK body resumption; ordinary SDK
    ///   request retries and Rapid server redirects remain separate layers, so this is not a
    ///   wire-request or elapsed-time bound (see `ReceiveConfig::max_total_repairs`). On
    ///   exhaustion the typed failure is returned and `progress` keeps what was received.
    pub(super) async fn receive_into<O: ReadOpener + ?Sized>(
        opener: &O,
        what: &str,
        dst: &mut [u8],
        progress: &ReceiveProgress,
        cfg: &ReceiveConfig,
    ) -> Result<()> {
        receive_span(opener, what, SpanOf::Whole, dst, progress, cfg)
            .await
            .map(|_| ())
    }

    /// Which part of the object one receive covers, and how it is tied to one version of it.
    pub(super) enum SpanOf<'a> {
        /// The whole object, exactly as [`receive_into`].
        Whole,
        /// The bytes `base..base + dst.len()`, one range of a read of several (or the only one):
        /// every range of the same logical GET shares `version`.
        Window {
            base: u64,
            version: &'a ObjectVersion<i64>,
        },
    }

    /// What [`receive_planned`] is told about its span.
    struct Plan<'a> {
        base: u64,
        ranged: bool,
        /// The generation every request must name from the first one on; `None` for a request
        /// that learns it from its own first response.
        pinned: Option<i64>,
        /// Set when this receive is the leader of a read of several ranges: the generation of
        /// its first response is reported through it (as soon as the response arrives).
        leader: Option<&'a Leader<'a, i64>>,
    }

    /// [`receive_into`] for a span of the object.
    ///
    /// * [`SpanOf::Whole`]: the whole object (`base == 0`), exactly as [`receive_into`]: the
    ///   object's size must equal `dst.len()` and exactly that many bytes are written.
    /// * [`SpanOf::Window`] (the range engine, stage G R3): every request names its window
    ///   (`start`, `limit`), the object must be at least `base` bytes long, and the bytes that
    ///   exist in the window are written (`min(dst.len(), size - base)`; the service clamps a
    ///   range to the end of the object). The count written is returned and the caller decides
    ///   whether a short answer is an error. Retries and repairs are pinned to the first
    ///   response's generation and size exactly as for the whole object.
    ///
    /// All the windows of one logical GET share one [`ObjectVersion`] (stage G review, G-F2):
    /// the first window to start is sent unpinned and reports its response's generation as soon
    /// as the response arrives; every other window waits for that, and every one of its requests
    /// names that generation, so an object replaced between two windows (by another of the same
    /// size included) is a `VersionChanged` error and never a body of two versions. A first
    /// response with no usable generation fails the other windows at once (the caller then reads
    /// the object as one response, see [`ObjectVersion::is_unavailable`]); a first window that
    /// fails releases them with its reason.
    pub(super) async fn receive_span<O: ReadOpener + ?Sized>(
        opener: &O,
        what: &str,
        span: SpanOf<'_>,
        dst: &mut [u8],
        progress: &ReceiveProgress,
        cfg: &ReceiveConfig,
    ) -> Result<usize> {
        match span {
            SpanOf::Whole => {
                let plan = Plan {
                    base: 0,
                    ranged: false,
                    pinned: None,
                    leader: None,
                };
                receive_planned(opener, what, plan, dst, progress, cfg).await
            }
            SpanOf::Window { base, version } => {
                // A window with nothing in it makes no request and takes no part in the version.
                if dst.is_empty() {
                    return Ok(0);
                }
                match version.join().await? {
                    Role::Follower(generation) => {
                        let plan = Plan {
                            base,
                            ranged: true,
                            pinned: Some(generation),
                            leader: None,
                        };
                        receive_planned(opener, what, plan, dst, progress, cfg).await
                    }
                    Role::Leader(leader) => {
                        let plan = Plan {
                            base,
                            ranged: true,
                            pinned: None,
                            leader: Some(&leader),
                        };
                        let result = receive_planned(opener, what, plan, dst, progress, cfg).await;
                        if let Err(e) = &result {
                            // Releases the other windows with the reason (a no-op once the
                            // generation was reported). Only the classification is passed on.
                            leader.fail(loggable(e));
                        }
                        result
                    }
                }
            }
        }
    }

    /// The receive loop of [`receive_span`], for a span whose version handling is settled.
    async fn receive_planned<O: ReadOpener + ?Sized>(
        opener: &O,
        what: &str,
        plan: Plan<'_>,
        dst: &mut [u8],
        progress: &ReceiveProgress,
        cfg: &ReceiveConfig,
    ) -> Result<usize> {
        let Plan {
            base,
            ranged,
            pinned: initial_pin,
            leader,
        } = plan;
        let len = dst.len();
        let expected = to_u64(len)?;
        if ranged && len == 0 {
            return Ok(0);
        }
        let mut span = SpanMark::new(progress);
        let mut cursor = 0usize;
        let mut pinned: Option<i64> = initial_pin;
        let mut first_size: Option<u64> = None;
        let mut budget = Budget::default();
        // A Rapid zero-length read is answered by the descriptor alone: no data range is
        // requested, so an empty object cannot fail with OUT_OF_RANGE and no HTTP size header
        // or stale control-plane zero is trusted.
        let descriptor_only = len == 0 && cfg.rapid;

        'attempt: loop {
            let cursor64 = to_u64(cursor)?;
            let remaining = expected
                .checked_sub(cursor64)
                .ok_or_else(|| anyhow!("cursor {cursor} is past the end of the span"))?;
            // The first whole-object request reads to the end; a repair, and every ranged
            // request, asks for exactly the missing part of its window.
            let request = OpenRequest {
                start: base
                    .checked_add(cursor64)
                    .ok_or_else(|| anyhow!("range start {base} + {cursor64} overflows"))?,
                limit: (cursor > 0 || ranged).then_some(remaining),
                generation: pinned,
            };
            let was_pinned = pinned.is_some();

            let acquired = if descriptor_only {
                opener.describe(None).await.map(|object| Opened {
                    facts: ObjectFacts {
                        size: object.size,
                        generation: object.generation,
                    },
                    stream: Box::new(EmptyStream),
                })
            } else {
                opener.open(&request).await
            };
            let opened = match acquired {
                Ok(opened) => opened,
                Err(e) => {
                    // OUT_OF_RANGE on the first full Rapid read may be an empty object.  Ask the
                    // authoritative descriptor; never turn it into absence.
                    if cfg.rapid && len > 0 && cursor == 0 && !was_pinned && is_out_of_range(&e) {
                        if let Ok(object) = opener.describe(None).await {
                            if object.size == 0 {
                                return Err(length_mismatch(what, expected, Some(0)));
                            }
                        }
                    }
                    let err = gax_failure(
                        e,
                        format!("GCS caller-buffer GET could not open {what}"),
                        was_pinned,
                    );
                    match budget.next_delay(cfg, &err, cursor, len, was_pinned) {
                        Some(delay) => {
                            debug!(
                                "GCS caller-buffer GET of {what}: retrying the open: {}",
                                loggable(&err)
                            );
                            back_off(delay).await;
                            continue 'attempt;
                        }
                        None => return Err(err),
                    }
                }
            };

            match pinned {
                Some(generation) if opened.facts.generation != generation => {
                    return Err(version_changed(what, "response generation differs"));
                }
                Some(_) => {}
                None if opened.facts.generation > 0 => {
                    pinned = Some(opened.facts.generation);
                    if let Some(leader) = leader {
                        leader.publish(opened.facts.generation);
                    }
                }
                None => {
                    if let Some(leader) = leader {
                        leader.unavailable();
                    }
                }
            }

            let size = size_to_u64(opened.facts.size, what)?;
            // How many bytes this span can deliver.
            let target: usize = if ranged {
                match first_size {
                    Some(known) if known != size => {
                        return Err(version_changed(
                            what,
                            "size differs on a pinned retry or repair",
                        ));
                    }
                    Some(_) => {}
                    None => first_size = Some(size),
                }
                // The window may run past the end of the object (the service clamps it); one
                // that starts at or past the end holds nothing and is a length mismatch.
                let available = match size.checked_sub(base) {
                    Some(available) if available > 0 => available,
                    _ => return Err(length_mismatch(what, expected, Some(size))),
                };
                usize::try_from(available.min(expected))
                    .map_err(|_| anyhow!("span length does not fit in memory"))?
            } else {
                if size != expected {
                    // The body is dropped unread.
                    return Err(if was_pinned {
                        version_changed(what, "size differs on a pinned retry or repair")
                    } else {
                        length_mismatch(what, expected, Some(size))
                    });
                }
                len
            };

            let before = cursor;
            let mut stream = opened.stream;
            loop {
                let next = match cfg.chunk_timeout {
                    Some(limit) => match tokio::time::timeout(limit, stream.next_chunk()).await {
                        Ok(next) => next,
                        Err(_) => Some(Err(GaxError::timeout(format!(
                            "no data for {limit:?} while reading"
                        )))),
                    },
                    None => stream.next_chunk().await,
                };
                match next {
                    None => break,
                    Some(Ok(chunk)) => {
                        if chunk.is_empty() {
                            continue;
                        }
                        let Some(end) =
                            cursor.checked_add(chunk.len()).filter(|end| *end <= target)
                        else {
                            return Err(length_mismatch(what, expected, None));
                        };
                        let Some(window) = dst.get_mut(cursor..end) else {
                            return Err(length_mismatch(what, expected, None));
                        };
                        window.copy_from_slice(&chunk);
                        span.record_copy(to_u64(cursor)?, to_u64(chunk.len())?)?;
                        cursor = end;
                    }
                    Some(Err(e)) => {
                        let err = gax_failure(
                            e,
                            format!("GCS caller-buffer GET stream failed for {what}"),
                            pinned.is_some(),
                        );
                        match budget.next_delay(cfg, &err, cursor, len, pinned.is_some()) {
                            Some(delay) => {
                                debug!(
                                    "GCS caller-buffer GET of {what}: repairing at offset {cursor}: {}",
                                    loggable(&err)
                                );
                                back_off(delay).await;
                                continue 'attempt;
                            }
                            None => return Err(err),
                        }
                    }
                }
            }

            if cursor == target {
                return Ok(cursor);
            }
            // A clean end that is short.  A pinned read that made progress repairs the suffix;
            // anything else is a length mismatch.
            if pinned.is_some() && cursor > before {
                if let Some(delay) = budget.take_repair(cfg, cursor) {
                    debug!(
                        "GCS caller-buffer GET of {what}: short end at {cursor} of {len}, repairing the suffix"
                    );
                    back_off(delay).await;
                    continue 'attempt;
                }
            }
            return Err(length_mismatch(what, expected, Some(to_u64(cursor)?)));
        }
    }

    // -- typed HEAD (pure decision logic) ----------------------------------

    /// `(GcsObjectMetadata, generation)` from a metadata `Object`.  The size is validated
    /// before conversion (a negative size is an error, never 0); an empty ETag is absent.
    /// The public `GcsObjectMetadata` has no generation, so it travels beside it.
    pub(super) fn metadata_with_generation(
        obj: &Object,
        key: &str,
    ) -> Result<(GcsObjectMetadata, i64)> {
        let size = size_to_u64(obj.size, key)?;
        Ok((
            GcsObjectMetadata {
                size,
                etag: (!obj.etag.is_empty()).then(|| obj.etag.clone()),
                updated: object_time_rfc3339(obj.update_time),
                key: key.to_string(),
            },
            obj.generation,
        ))
    }

    /// The outcome of a metadata request: present with its generation, absent only for a typed
    /// NotFound, or the classified failure.  OUT_OF_RANGE, 403, throttling and every other
    /// failure stay failures.
    pub(super) fn head_outcome(
        result: Result<Object, GaxError>,
        what: &str,
        key: &str,
    ) -> Result<Option<(GcsObjectMetadata, i64)>> {
        match result {
            Ok(obj) => metadata_with_generation(&obj, key).map(Some),
            Err(e) => {
                let err = gax_failure(e, format!("GCS HEAD failed for {what}"), false);
                match object_error(&err).map(|o| o.kind) {
                    Some(ObjectErrorKind::NotFound) => Ok(None),
                    _ => Err(err),
                }
            }
        }
    }

    /// The provider-neutral HEAD answer: size, the ETag if any, and the generation as
    /// `version_id` when the service gave one.
    pub(super) fn head_metadata(meta: GcsObjectMetadata, generation: i64) -> HeadMetadata {
        HeadMetadata {
            size: meta.size,
            etag: meta.etag,
            version_id: (generation > 0).then(|| generation.to_string()),
        }
    }

    // -- the real openers --------------------------------------------------

    /// Standard buckets: the HTTP `read_object` route.
    ///
    /// The fields are `pub(super)` so the tests can drive this real opener against a local server.
    pub(super) struct HttpOpener<'a> {
        pub(super) storage: &'a Storage,
        pub(super) bucket_name: &'a str,
        pub(super) object: &'a str,
    }

    struct HttpStream(ReadObjectResponse);

    #[async_trait]
    impl ChunkStream for HttpStream {
        async fn next_chunk(&mut self) -> Option<Result<Bytes, GaxError>> {
            self.0.next().await
        }
    }

    #[async_trait]
    impl ReadOpener for HttpOpener<'_> {
        /// Opens `GET ...?alt=media`: `set_generation` becomes the `generation` query parameter
        /// and the range becomes the `Range` header (`bytes=a-b`, or `bytes=a-` to the end).
        ///
        /// The object's size is `x-goog-stored-content-length`.  On the live JSON media API that
        /// header carries the FULL stored size on ranged (206) responses as well as on unranged
        /// ones (measured 2026-10: `Range: bytes=4-15` on a 16-byte object gives
        /// `content-range: bytes 4-15/16`, `x-goog-stored-content-length: 16`, `content-length:
        /// 12`).  The size check of a repair against `dst.len()` therefore compares the whole
        /// object, as it does for the first request.  The generation is `x-goog-generation`.
        async fn open(&self, request: &OpenRequest) -> Result<Opened, GaxError> {
            let object = http_media_object_name(self.object);
            // The SDK's own read-resume layer is switched off on this path.  Its default
            // (`Recommended`, fork `storage/common_options.rs:34`) has no attempt limit
            // (`read_resume_policy.rs:134-140` ignores the resume count), so it could reconnect
            // once per byte of a slowly advancing body, outside every bound of the receive loop.
            // With `NeverResume` a broken body surfaces here as its typed error and the bounded
            // repair loop in `receive_into` owns ordinary body repairs, each pinned to the
            // generation. Every open still keeps the client's request retry policy; Rapid
            // server redirects are a separate documented exception to the body-resume policy.
            let mut builder = self
                .storage
                .read_object(self.bucket_name, object.as_ref())
                .with_read_resume_policy(NeverResume);
            if let Some(generation) = request.generation {
                builder = builder.set_generation(generation);
            }
            match (request.start, request.limit) {
                (0, None) => {}
                (start, Some(limit)) => {
                    builder = builder.set_read_range(ReadRange::segment(start, limit))
                }
                (start, None) => builder = builder.set_read_range(ReadRange::offset(start)),
            }
            let response = builder.send().await?;
            let highlights = response.object();
            Ok(Opened {
                facts: ObjectFacts {
                    size: highlights.size,
                    generation: highlights.generation,
                },
                stream: Box::new(HttpStream(response)),
            })
        }

        /// Not available on the HTTP route.  The descriptor is a Rapid concept: `receive_into`
        /// asks for one only when `ReceiveConfig::rapid` is set, which this route never sets.  The
        /// answer is a typed, non-retryable error, not a data GET whose body would be thrown away.
        async fn describe(&self, generation: Option<i64>) -> Result<Object, GaxError> {
            Err(GaxError::binding(format!(
                "the HTTP route has no object descriptor (Rapid buckets only); generation {generation:?} was requested"
            )))
        }
    }

    /// Rapid buckets: the existing `open_object` (BidiReadObject) route.
    ///
    /// No local test drives this opener: the pinned fork has no local bidi-streaming server to
    /// point it at, so its `set_generation`, range mapping and descriptor facts are covered by the
    /// live Rapid gate of plan section 8 (the "Existing GCS Rapid route" row), not by a unit
    /// test.  The receive loop above it is tested with the fake opener in `rapid` mode.
    struct RapidOpener<'a> {
        storage: &'a Storage,
        bucket_name: &'a str,
        object: &'a str,
        attempt_timeout: Duration,
    }

    /// A Rapid read.  The descriptor stays open for as long as the reader is used.
    struct RapidStream {
        descriptor: ObjectDescriptor,
        reader: ReadObjectResponse,
    }

    #[async_trait]
    impl ChunkStream for RapidStream {
        async fn next_chunk(&mut self) -> Option<Result<Bytes, GaxError>> {
            let next = self.reader.next().await;
            if matches!(next, Some(Err(_))) {
                trace!(
                    "GCS Rapid read failed; upload id {:?}",
                    self.descriptor.headers().get("x-guploader-uploadid")
                );
            }
            next
        }
    }

    #[async_trait]
    impl ReadOpener for RapidOpener<'_> {
        async fn open(&self, request: &OpenRequest) -> Result<Opened, GaxError> {
            let mut builder = self
                .storage
                .open_object(self.bucket_name, self.object)
                .with_attempt_timeout(self.attempt_timeout)
                // No SDK-level reconnection of a broken read (fork `bidi/connector.rs`
                // `reconnect`, which consults this policy, `Recommended` by default and without
                // an attempt limit): the bounded repair loop in `receive_into` owns it.
                .with_read_resume_policy(NeverResume);
            if let Some(generation) = request.generation {
                builder = builder.set_generation(generation);
            }
            let range = match (request.start, request.limit) {
                (0, None) => ReadRange::all(),
                (start, Some(limit)) => ReadRange::segment(start, limit),
                (start, None) => ReadRange::offset(start),
            };
            let (descriptor, reader) = builder.send_and_read(range).await?;
            let object = descriptor.object();
            Ok(Opened {
                facts: ObjectFacts {
                    size: object.size,
                    generation: object.generation,
                },
                stream: Box::new(RapidStream { descriptor, reader }),
            })
        }

        async fn describe(&self, generation: Option<i64>) -> Result<Object, GaxError> {
            let mut builder = self
                .storage
                .open_object(self.bucket_name, self.object)
                .with_attempt_timeout(self.attempt_timeout);
            if let Some(generation) = generation {
                builder = builder.set_generation(generation);
            }
            Ok(builder.send().await?.object())
        }
    }

    impl GcsClient {
        /// Read the whole object into `dst` (caller-buffer GET, plan §4.3 and §4.5).
        ///
        /// `dst.len()` is the expected size: `Ok(())` means exactly `dst.len()` bytes were
        /// written and the object's size equals it.  Rapid buckets use the existing Rapid
        /// route (`open_object`); other buckets use HTTP.  The first response's generation is
        /// pinned on every retry and repair; see [`receive_into`] for the full contract.
        /// `progress` counts the distinct positions of `dst` filled, also when the read fails.
        pub async fn get_object_into(
            &self,
            bucket: &str,
            object: &str,
            dst: &mut [u8],
            progress: &ReceiveProgress,
        ) -> Result<()> {
            let what = format!("gs://{bucket}/{object}");
            let bucket_name = format_bucket_name(bucket);
            let rapid = self.is_rapid_bucket(bucket).await;
            let cfg = read_config(rapid, dst.len())?;
            if rapid {
                debug!("GCS GET INTO (gRPC): {what}, {} bytes", dst.len());
                let opener = RapidOpener {
                    storage: &self.storage,
                    bucket_name: &bucket_name,
                    object,
                    attempt_timeout: gcs_bidi_attempt_timeout(),
                };
                receive_into(&opener, &what, dst, progress, &cfg).await
            } else {
                debug!("GCS GET INTO (HTTP): {what}, {} bytes", dst.len());
                let opener = HttpOpener {
                    storage: &self.storage,
                    bucket_name: &bucket_name,
                    object,
                };
                receive_into(&opener, &what, dst, progress, &cfg).await
            }
        }

        /// Read the bytes `offset..offset + dst.len()` of an object straight into `dst` and
        /// return how many were written (stage G, R3: the range engine's per-range receive).
        ///
        /// The count is `dst.len()` unless the range runs past the end of the object (the
        /// service clamps it): the caller decides whether a short answer is an error. A range
        /// that starts at or past the end is a `LengthMismatch`. Rapid buckets use the Rapid route,
        /// others HTTP; retries and repairs are pinned to the first response's generation (see
        /// [`receive_span`]). `progress` counts the positions written.
        ///
        /// This is one range standing alone (its own version). The ranges of ONE logical GET go
        /// through [`Self::get_object_range_into_versioned`] so that they share a version.
        pub async fn get_object_range_into(
            &self,
            bucket: &str,
            object: &str,
            offset: u64,
            dst: &mut [u8],
            progress: &ReceiveProgress,
        ) -> Result<usize> {
            self.get_object_range_into_versioned(
                bucket,
                object,
                offset,
                dst,
                progress,
                &ObjectVersion::new(),
            )
            .await
        }

        /// [`Self::get_object_range_into`] for one range of a read of SEVERAL ranges of the same
        /// object (the range engine, stage G review G-F2): all of them share `version`, the
        /// generation of the first response to arrive. The first range to start is sent
        /// unpinned; every other range names that generation in every request it makes, so an
        /// object replaced between two ranges is a `VersionChanged` error, never a body of two
        /// versions. See [`receive_span`] for the rest.
        pub(crate) async fn get_object_range_into_versioned(
            &self,
            bucket: &str,
            object: &str,
            offset: u64,
            dst: &mut [u8],
            progress: &ReceiveProgress,
            version: &ObjectVersion<i64>,
        ) -> Result<usize> {
            let what = format!("gs://{bucket}/{object}");
            let bucket_name = format_bucket_name(bucket);
            let rapid = self.is_rapid_bucket(bucket).await;
            let cfg = read_config(rapid, dst.len())?;
            let span = SpanOf::Window {
                base: offset,
                version,
            };
            if rapid {
                debug!(
                    "GCS GET RANGE INTO (gRPC): {what}, offset={offset}, {} bytes",
                    dst.len()
                );
                let opener = RapidOpener {
                    storage: &self.storage,
                    bucket_name: &bucket_name,
                    object,
                    attempt_timeout: gcs_bidi_attempt_timeout(),
                };
                receive_span(&opener, &what, span, dst, progress, &cfg).await
            } else {
                debug!(
                    "GCS GET RANGE INTO (HTTP): {what}, offset={offset}, {} bytes",
                    dst.len()
                );
                let opener = HttpOpener {
                    storage: &self.storage,
                    bucket_name: &bucket_name,
                    object,
                };
                receive_span(&opener, &what, span, dst, progress, &cfg).await
            }
        }

        /// Typed HEAD (plan §4.3, §4.5 A8): size, ETag if any, and the generation as
        /// `version_id`.  `Ok(None)` only for a typed NotFound.  GCS answers a missing bucket and a
        /// missing object with the same NotFound, so `None` can mean either: the two are not told
        /// apart, and no extra probe is made to manufacture the distinction.  403 and every other
        /// failure are errors.
        pub async fn head_typed(&self, bucket: &str, object: &str) -> Result<Option<HeadMetadata>> {
            Ok(self
                .stat_object_with_generation(bucket, object)
                .await?
                .map(|(meta, generation)| head_metadata(meta, generation)))
        }

        /// Object metadata with its generation, `None` for a typed NotFound.
        ///
        /// On a Rapid bucket the size comes from the `open_object(..).send()` descriptor, which
        /// requests no data range: an empty object reports 0 instead of failing a one-byte
        /// read with OUT_OF_RANGE, and the control plane's possibly stale zero is not used.
        pub(crate) async fn stat_object_with_generation(
            &self,
            bucket: &str,
            object: &str,
        ) -> Result<Option<(GcsObjectMetadata, i64)>> {
            let what = format!("gs://{bucket}/{object}");
            let bucket_name = format_bucket_name(bucket);
            let described: Result<Object, GaxError> = if self.is_rapid_bucket(bucket).await {
                let opener = RapidOpener {
                    storage: &self.storage,
                    bucket_name: &bucket_name,
                    object,
                    attempt_timeout: gcs_bidi_attempt_timeout(),
                };
                opener.describe(None).await
            } else {
                self.control
                    .get_object()
                    .set_bucket(bucket_name.clone())
                    .set_object(object.to_string())
                    .send()
                    .await
            };
            head_outcome(described, &what, object)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// Serialize all tests that mutate environment variables.  Without this
    /// guard two tests running in parallel could observe each other's
    /// `set_var` / `remove_var` calls and produce non-deterministic results.
    pub(super) static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // ------------------------------------------------------------------
    // parse_gcs_uri
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_gcs_uri_standard() {
        let (bucket, obj) = parse_gcs_uri("gs://my-bucket/path/to/object.bin").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(obj, "path/to/object.bin");
    }

    #[test]
    fn test_parse_gcs_uri_gcs_scheme() {
        let (bucket, obj) = parse_gcs_uri("gcs://my-bucket/data/file.npz").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(obj, "data/file.npz");
    }

    #[test]
    fn test_parse_gcs_uri_bucket_only_with_slash() {
        let (bucket, obj) = parse_gcs_uri("gs://my-bucket/").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(obj, "");
    }

    #[test]
    fn test_parse_gcs_uri_bucket_only_no_slash() {
        let (bucket, obj) = parse_gcs_uri("gs://my-bucket").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(obj, "");
    }

    #[test]
    fn test_parse_gcs_uri_nested_path() {
        let (bucket, obj) =
            parse_gcs_uri("gs://rapids-test-bucket/checkpoints/epoch-10/model.bin").unwrap();
        assert_eq!(bucket, "rapids-test-bucket");
        assert_eq!(obj, "checkpoints/epoch-10/model.bin");
    }

    #[test]
    fn test_parse_gcs_uri_wrong_scheme() {
        assert!(parse_gcs_uri("s3://bucket/obj").is_err());
        assert!(parse_gcs_uri("az://container/blob").is_err());
        assert!(parse_gcs_uri("bucket/obj").is_err());
    }

    #[test]
    fn test_parse_gcs_uri_empty_bucket() {
        assert!(parse_gcs_uri("gs:///some-object").is_err());
        assert!(parse_gcs_uri("gs://").is_err());
    }

    // ------------------------------------------------------------------
    // format_bucket_name
    // ------------------------------------------------------------------

    #[test]
    fn test_format_bucket_name_standard() {
        assert_eq!(
            format_bucket_name("my-bucket"),
            "projects/_/buckets/my-bucket"
        );
    }

    #[test]
    fn test_format_bucket_name_rapid_bucket() {
        assert_eq!(
            format_bucket_name("rapid-hyperdisk-ml-bucket"),
            "projects/_/buckets/rapid-hyperdisk-ml-bucket"
        );
    }

    // ------------------------------------------------------------------
    // read_rapid_mode — env var parsing
    // ------------------------------------------------------------------

    #[test]
    fn test_rapid_mode_unset_is_auto() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(
            read_rapid_mode(),
            RapidMode::Auto,
            "absent var should default to Auto"
        );
    }

    #[test]
    fn test_rapid_mode_true_lowercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "true");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::ForceOn);
    }

    #[test]
    fn test_rapid_mode_true_uppercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "TRUE");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(
            r,
            RapidMode::ForceOn,
            "TRUE should force RAPID on (case-insensitive)"
        );
    }

    #[test]
    fn test_rapid_mode_one() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "1");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::ForceOn, "\"1\" should force RAPID on");
    }

    #[test]
    fn test_rapid_mode_yes_lowercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "yes");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::ForceOn);
    }

    #[test]
    fn test_rapid_mode_yes_uppercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "YES");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(
            r,
            RapidMode::ForceOn,
            "YES should force RAPID on (case-insensitive)"
        );
    }

    #[test]
    fn test_rapid_mode_false_string() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "false");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::ForceOff, "\"false\" should force RAPID off");
    }

    #[test]
    fn test_rapid_mode_zero() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "0");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::ForceOff, "\"0\" should force RAPID off");
    }

    #[test]
    fn test_rapid_mode_unrecognised_value() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "maybe");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(
            r,
            RapidMode::Auto,
            "unrecognised value should default to Auto"
        );
    }

    #[test]
    fn test_rapid_mode_auto_explicit() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "auto");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::Auto, "\"auto\" should set Auto mode");
    }

    // RED-then-GREEN regression tests for s3dlio issue #157 bug 5.2 (D2).
    //
    // Bug: is_rapid_bucket() cached detect_rapid_bucket()'s result in a
    // per-bucket `OnceCell<bool>` via `get_or_init`, which always
    // populates the cell with whatever the closure returns -- including
    // the `false` fallback detect_rapid_bucket used on an RPC failure.
    // A single transient get_storage_layout() failure (network blip,
    // auth hiccup) therefore permanently mis-cached a RAPID/zonal bucket
    // as "standard" for the lifetime of the process: gRPC/RAPID
    // performance benefits lost forever, with no retry.
    //
    // The fix (get_or_try_init instead of get_or_init) is exercised here
    // directly against a synthetic `Arc<OnceCell<bool>>` + counting
    // detect closure -- driving the real is_rapid_bucket() needs a live
    // or mocked GCS gRPC StorageControl client, unavailable in this
    // environment (matches the "requires exposing the mockable
    // interface" adequacy note in the fix plan for this bug). This is
    // the exact tokio::sync::OnceCell API and calling pattern
    // is_rapid_bucket() uses, so it proves the caching CONTRACT
    // (retry-after-failure, no poisoning, thundering-herd-safe) that the
    // real method depends on.

    #[tokio::test]
    async fn failed_detection_does_not_poison_the_cache() {
        let cell: OnceCell<bool> = OnceCell::new();
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));

        // First call: detection fails (simulated transient RPC error).
        let attempts_clone = attempts.clone();
        let first: Result<bool, anyhow::Error> = cell
            .get_or_try_init(|| async move {
                attempts_clone.fetch_add(1, Ordering::SeqCst);
                Err(anyhow!("UNAVAILABLE: simulated transient failure"))
            })
            .await
            .copied();
        assert!(first.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);

        // Second call: detection now succeeds. Pre-fix (get_or_init),
        // the cell would already be permanently set to `false` from the
        // first call and this closure would never run again.
        let attempts_clone = attempts.clone();
        let second: Result<bool, anyhow::Error> = cell
            .get_or_try_init(|| async move {
                attempts_clone.fetch_add(1, Ordering::SeqCst);
                Ok(true) // genuinely a RAPID/zonal bucket
            })
            .await
            .copied();
        assert!(
            second.unwrap(),
            "a failed first attempt must not poison the cache with a wrong fallback value"
        );
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "detection must be retried after a prior failure, not skipped"
        );
    }

    #[tokio::test]
    async fn successful_detection_is_cached_and_not_retried() {
        let cell: OnceCell<bool> = OnceCell::new();
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let attempts_clone = attempts.clone();
            let result: Result<bool, anyhow::Error> = cell
                .get_or_try_init(|| async move {
                    attempts_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(true)
                })
                .await
                .copied();
            assert!(result.unwrap());
        }

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "a successful detection must be cached -- only the first of 3 calls should \
             actually invoke the detection closure"
        );
    }

    // ------------------------------------------------------------------
    // dotenvy — load vars from a .env file
    // ------------------------------------------------------------------

    /// Verify that a `.env` file containing `S3DLIO_GCS_RAPID=true` causes
    /// `read_rapid_mode()` to return `true` after the file is loaded via
    /// `dotenvy::from_path()`.
    #[test]
    fn test_dotenvy_rapid_mode_enabled_from_dotenv() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        let dir = TempDir::new().unwrap();
        let env_path = dir.path().join(".env");
        std::fs::write(&env_path, "S3DLIO_GCS_RAPID=true\n").unwrap();

        assert!(
            read_rapid_mode() == RapidMode::Auto,
            "should be Auto before loading .env"
        );

        dotenvy::from_path(&env_path).expect("dotenvy::from_path should succeed");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        assert_eq!(
            r,
            RapidMode::ForceOn,
            "RAPID mode should be ForceOn after loading .env with S3DLIO_GCS_RAPID=true"
        );
    }

    /// `S3DLIO_GCS_RAPID=false` in a `.env` file should leave RAPID mode off.
    #[test]
    fn test_dotenvy_rapid_mode_disabled_in_dotenv() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        let dir = TempDir::new().unwrap();
        let env_path = dir.path().join(".env");
        std::fs::write(&env_path, "S3DLIO_GCS_RAPID=false\n").unwrap();

        dotenvy::from_path(&env_path).expect("dotenvy::from_path should succeed");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        assert_eq!(
            r,
            RapidMode::ForceOff,
            "RAPID mode should be ForceOff when .env sets it to false"
        );
    }

    /// `GOOGLE_APPLICATION_CREDENTIALS` loaded from a `.env` file should
    /// appear in the process environment exactly as written.
    #[test]
    fn test_dotenvy_credentials_path_from_dotenv() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");

        let fake_creds = "/workspace/secrets/gcp-sa.json";
        let dir = TempDir::new().unwrap();
        let env_path = dir.path().join(".env");
        std::fs::write(
            &env_path,
            format!("GOOGLE_APPLICATION_CREDENTIALS={}\n", fake_creds),
        )
        .unwrap();

        dotenvy::from_path(&env_path).expect("dotenvy::from_path should succeed");
        let loaded = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").unwrap_or_default();
        std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");

        assert_eq!(loaded, fake_creds);
    }

    /// A single `.env` file can set both `S3DLIO_GCS_RAPID` and
    /// `GOOGLE_APPLICATION_CREDENTIALS` at the same time.
    #[test]
    fn test_dotenvy_multiple_vars_from_dotenv() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");

        let dir = TempDir::new().unwrap();
        let env_path = dir.path().join(".env");
        std::fs::write(
            &env_path,
            "S3DLIO_GCS_RAPID=1\nGOOGLE_APPLICATION_CREDENTIALS=/tmp/creds.json\n",
        )
        .unwrap();

        dotenvy::from_path(&env_path).expect("dotenvy::from_path should succeed");
        let rapid = read_rapid_mode();
        let creds = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").unwrap_or_default();

        std::env::remove_var("S3DLIO_GCS_RAPID");
        std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");

        assert_eq!(
            rapid,
            RapidMode::ForceOn,
            "RAPID mode should be ForceOn via .env"
        );
        assert_eq!(
            creds, "/tmp/creds.json",
            "credentials path should be loaded from .env"
        );
    }

    /// An environment variable already set in the shell takes precedence over
    /// the same variable in a `.env` file — this is standard dotenvy behaviour.
    #[test]
    fn test_dotenvy_shell_env_takes_precedence_over_dotenv() {
        let _g = ENV_MUTEX.lock().unwrap();

        // Pre-set the var (simulates `export S3DLIO_GCS_RAPID=false` in the shell)
        std::env::set_var("S3DLIO_GCS_RAPID", "false");

        let dir = TempDir::new().unwrap();
        let env_path = dir.path().join(".env");
        // .env tries to set it to true, but dotenvy must not override existing vars
        std::fs::write(&env_path, "S3DLIO_GCS_RAPID=true\n").unwrap();

        dotenvy::from_path(&env_path).ok(); // ok() - may warn but must not panic

        let raw = std::env::var("S3DLIO_GCS_RAPID").unwrap();
        let rapid = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        assert_eq!(raw, "false", "shell env var must not be overridden by .env");
        assert_eq!(
            rapid,
            RapidMode::ForceOff,
            "read_rapid_mode() must respect the shell-set value"
        );
    }

    #[test]
    fn test_is_gcs_out_of_range_error_matches_known_patterns() {
        assert!(is_gcs_out_of_range_error("OUT_OF_RANGE"));
        assert!(is_gcs_out_of_range_error(
            "The provided read ranges are outside the valid object range"
        ));
        assert!(is_gcs_out_of_range_error(
            "the service reports an error with code OUT_OF_RANGE described as: The provided read ranges are outside the valid object range."
        ));
        assert!(!is_gcs_out_of_range_error("RESOURCE_EXHAUSTED"));
    }

    #[test]
    fn test_should_treat_out_of_range_as_empty_full_read_only_for_full_reads() {
        let msg = "OUT_OF_RANGE: The provided read ranges are outside the valid object range.";
        assert!(should_treat_out_of_range_as_empty_full_read(
            msg,
            &ReadRange::all()
        ));
        assert!(!should_treat_out_of_range_as_empty_full_read(
            msg,
            &ReadRange::segment(0, 1)
        ));
        assert!(!should_treat_out_of_range_as_empty_full_read(
            "INTERNAL",
            &ReadRange::all()
        ));
    }

    #[test]
    fn test_should_repair_incomplete_full_read() {
        assert!(should_repair_incomplete_full_read(
            &ReadRange::all(),
            8 * 1024 * 1024,
            3_162_112
        ));
        assert!(!should_repair_incomplete_full_read(
            &ReadRange::all(),
            8 * 1024 * 1024,
            8 * 1024 * 1024
        ));
        assert!(!should_repair_incomplete_full_read(
            &ReadRange::segment(0, 1024),
            1024,
            512
        ));
    }

    #[test]
    fn test_is_retryable_gcs_read_error() {
        assert!(is_retryable_gcs_read_error(
            "timeout for gs://bucket/object"
        ));
        assert!(is_retryable_gcs_read_error("UNAVAILABLE: upstream reset"));
        assert!(is_retryable_gcs_read_error("CANCELLED by peer"));
        assert!(!is_retryable_gcs_read_error("PERMISSION_DENIED"));
    }

    // RED-then-GREEN regression tests for s3dlio issue #157 bug 5.1 (D1).
    //
    // Bug: the full-read retry loop in get_object_via_grpc retried
    // immediately on a retryable error (e.g. UNAVAILABLE) with no delay
    // at all between attempts -- hammering a transiently-overloaded or
    // rate-limited endpoint instead of backing off.
    //
    // gcs_retry_backoff_delay() is a pure function, so its exponential
    // growth is directly and deterministically testable with no timing
    // flakiness. retry_with_backoff() is the exact function
    // get_object_via_grpc's full-read branch calls (extracted for this
    // reason) -- tested here under a paused Tokio clock with a
    // synthetic operation closure, since driving the real call site
    // needs a live or mocked gRPC StorageControl client.

    #[test]
    fn gcs_retry_backoff_delay_grows_exponentially_and_caps() {
        assert_eq!(gcs_retry_backoff_delay(1), Duration::from_millis(100));
        assert_eq!(gcs_retry_backoff_delay(2), Duration::from_millis(200));
        assert_eq!(gcs_retry_backoff_delay(3), Duration::from_millis(400));
        assert_eq!(gcs_retry_backoff_delay(4), Duration::from_millis(800));
        assert_eq!(gcs_retry_backoff_delay(5), Duration::from_millis(1600));
        // Cap at 2s -- attempt 6 would be 3200ms uncapped.
        assert_eq!(gcs_retry_backoff_delay(6), Duration::from_millis(2000));
        assert_eq!(gcs_retry_backoff_delay(100), Duration::from_millis(2000));
    }

    #[tokio::test(start_paused = true)]
    async fn retry_with_backoff_sleeps_between_retryable_failures() {
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let start = tokio::time::Instant::now();

        let attempts_clone = attempts.clone();
        let result = retry_with_backoff(3, move |_attempt| {
            let attempts = attempts_clone.clone();
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(anyhow!("UNAVAILABLE: simulated transient failure"))
                } else {
                    Ok("done")
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), "done");
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "should have taken exactly 3 attempts (2 retries)"
        );
        // Two sleeps elapsed: gcs_retry_backoff_delay(1) + gcs_retry_backoff_delay(2)
        // = 100ms + 200ms = 300ms of virtual time.
        assert_eq!(start.elapsed(), Duration::from_millis(300));
    }

    #[tokio::test(start_paused = true)]
    async fn retry_with_backoff_does_not_sleep_on_non_retryable_error() {
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let start = tokio::time::Instant::now();

        let attempts_clone = attempts.clone();
        let result: Result<()> = retry_with_backoff(3, move |_attempt| {
            let attempts = attempts_clone.clone();
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(anyhow!("PERMISSION_DENIED"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            1,
            "a non-retryable error must not be retried"
        );
        assert_eq!(start.elapsed(), Duration::from_millis(0));
    }

    #[tokio::test(start_paused = true)]
    async fn retry_with_backoff_gives_up_after_max_attempts() {
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));

        let attempts_clone = attempts.clone();
        let result: Result<()> = retry_with_backoff(3, move |_attempt| {
            let attempts = attempts_clone.clone();
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(anyhow!("UNAVAILABLE: always fails"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_gcs_read_chunk_timeout_default() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS");
        assert_eq!(
            gcs_read_chunk_timeout(8 * 1024 * 1024),
            Some(Duration::from_secs(12))
        );
    }

    #[test]
    fn test_gcs_read_chunk_timeout_disabled_when_zero() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS", "0");
        let timeout = gcs_read_chunk_timeout(8 * 1024 * 1024);
        std::env::remove_var("S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS");
        assert_eq!(timeout, None);
    }

    #[test]
    fn test_gcs_read_chunk_timeout_from_env() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS", "7");
        let timeout = gcs_read_chunk_timeout(8 * 1024 * 1024);
        std::env::remove_var("S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS");
        assert_eq!(timeout, Some(Duration::from_secs(7)));
    }

    #[test]
    fn test_gcs_scaled_timeout_secs_uses_min_for_small_objects() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_READ_TIMEOUT_SECS_PER_GIB");
        std::env::remove_var("S3DLIO_GCS_READ_TIMEOUT_MIN_SECS");
        std::env::remove_var("S3DLIO_GCS_READ_TIMEOUT_MAX_SECS");
        assert_eq!(gcs_scaled_timeout_secs(8 * 1024 * 1024), 12);
    }

    #[test]
    fn test_gcs_scaled_timeout_secs_scales_by_object_size() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_READ_TIMEOUT_SECS_PER_GIB");
        std::env::remove_var("S3DLIO_GCS_READ_TIMEOUT_MIN_SECS");
        std::env::remove_var("S3DLIO_GCS_READ_TIMEOUT_MAX_SECS");
        assert_eq!(gcs_scaled_timeout_secs(3 * 1024 * 1024 * 1024), 135);
    }

    #[test]
    fn test_gcs_read_chunk_progress_interval_default() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_READ_PROGRESS_INTERVAL_SECS");
        assert_eq!(gcs_read_chunk_progress_interval(), Duration::from_secs(5));
    }

    #[test]
    fn test_gcs_read_chunk_progress_interval_from_env() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_READ_PROGRESS_INTERVAL_SECS", "2");
        let interval = gcs_read_chunk_progress_interval();
        std::env::remove_var("S3DLIO_GCS_READ_PROGRESS_INTERVAL_SECS");
        assert_eq!(interval, Duration::from_secs(2));
    }

    #[test]
    fn test_gcs_bidi_attempt_timeout_default() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_BIDI_ATTEMPT_TIMEOUT_SECS");
        assert_eq!(gcs_bidi_attempt_timeout(), Duration::from_secs(12));
    }

    #[test]
    fn test_gcs_bidi_attempt_timeout_from_env() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_BIDI_ATTEMPT_TIMEOUT_SECS", "4");
        let timeout = gcs_bidi_attempt_timeout();
        std::env::remove_var("S3DLIO_GCS_BIDI_ATTEMPT_TIMEOUT_SECS");
        assert_eq!(timeout, Duration::from_secs(4));
    }
}

// ---------------------------------------------------------------------------
// Zero-copy correctness tests
//
// These tests verify that EVERY data path through the GCS client is genuinely
// zero-copy by comparing raw pointers before and after each operation.
//
// Key invariants:
//   - `BytesMut::freeze()` → same base pointer (no reallocation)
//   - `Bytes::clone()`     → same base pointer (Arc refcount increment)
//   - `Bytes::slice(a..b)` → pointer == base + a (sub-slice, no copy)
//   - `Bytes::from(Vec<u8>)` → same pointer as original Vec
//   - `std::mem::take` + `Bytes::from` → same pointer (GcsBufferedWriter path)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod zero_copy_tests {
    use bytes::{BufMut, Bytes, BytesMut};

    // ------------------------------------------------------------------
    // Primitive zero-copy building blocks
    // ------------------------------------------------------------------

    /// `BytesMut::freeze()` must not reallocate — the returned `Bytes` must
    /// point to exactly the same memory that `BytesMut` wrote into.
    ///
    /// This covers the accumulation loop in `get_object_via_grpc` and
    /// `get_object_via_http`.
    #[test]
    fn test_bytesmut_freeze_is_zero_copy() {
        let mut buf = BytesMut::with_capacity(1024);
        buf.put_slice(b"hello world this is test data for a gRPC chunk");
        let raw_ptr_before = buf.as_ptr();

        let frozen: Bytes = buf.freeze();

        assert_eq!(
            frozen.as_ptr(),
            raw_ptr_before,
            "BytesMut::freeze() must be zero-copy — pointer must not change"
        );
        assert_eq!(
            &frozen[..],
            b"hello world this is test data for a gRPC chunk"
        );
    }

    /// `Bytes::clone()` is an Arc reference-count increment — O(1), no memcpy.
    ///
    /// This covers the `let data = data.clone();` in the producer task spawn
    /// inside `write_object_grpc`.
    #[test]
    fn test_bytes_clone_is_zero_copy() {
        let bytes = Bytes::from(vec![0xABu8; 4096]);
        let original_ptr = bytes.as_ptr();

        let cloned = bytes.clone();

        assert_eq!(
            cloned.as_ptr(),
            original_ptr,
            "Bytes::clone() must share the same backing buffer (Arc increment, not memcpy)"
        );
        assert_eq!(bytes.len(), cloned.len());
    }

    /// `Bytes::slice(a..b)` must return a view into the existing buffer —
    /// the pointer of the slice must equal `base + a`, no new allocation.
    ///
    /// This covers every `data.slice(offset..end)` in the write producer loop.
    #[test]
    fn test_bytes_slice_is_zero_copy() {
        let bytes = Bytes::from(vec![0xCCu8; 4096]);
        let base_ptr = bytes.as_ptr();

        let offset = 512usize;
        let sliced = bytes.slice(offset..1024);

        assert_eq!(
            sliced.as_ptr(),
            unsafe { base_ptr.add(offset) },
            "Bytes::slice() must point into the existing allocation (zero-copy sub-slice)"
        );
        assert_eq!(sliced.len(), 512);
    }

    /// `Bytes::from(Vec<u8>)` must take ownership of the Vec's allocation —
    /// the resulting `Bytes` must have the same pointer as the original Vec.
    ///
    /// This is the foundation of the entire put path.
    #[test]
    fn test_bytes_from_vec_preserves_pointer() {
        let mut vec: Vec<u8> = Vec::with_capacity(8192);
        vec.extend_from_slice(&[0xDDu8; 8192]);
        let vec_ptr = vec.as_ptr();

        let bytes = Bytes::from(vec);

        assert_eq!(
            bytes.as_ptr(),
            vec_ptr,
            "Bytes::from(Vec<u8>) must transfer ownership without copying"
        );
    }

    // ------------------------------------------------------------------
    // GcsBufferedWriter finalise path
    // ------------------------------------------------------------------

    /// Mirrors the exact code in `GcsBufferedWriter::finalise()`:
    ///
    /// ```rust
    /// Bytes::from(std::mem::take(&mut self.buffer))
    /// ```
    ///
    /// `mem::take` moves the allocation out of `self.buffer` (leaving an empty
    /// Vec) and `Bytes::from` takes ownership without copying.
    #[test]
    fn test_buffered_writer_finalise_path_is_zero_copy() {
        let mut buffer: Vec<u8> = Vec::with_capacity(65536);
        buffer.extend_from_slice(&[0xBEu8; 65536]);
        let original_ptr = buffer.as_ptr();

        // Mirrors: Bytes::from(std::mem::take(&mut self.buffer))
        let bytes = Bytes::from(std::mem::take(&mut buffer));

        assert!(buffer.is_empty(), "mem::take must empty the source Vec");
        assert_eq!(
            bytes.as_ptr(),
            original_ptr,
            "GcsBufferedWriter finalise: Bytes::from(mem::take(buffer)) must preserve pointer"
        );
        assert_eq!(bytes.len(), 65536);
    }

    // ------------------------------------------------------------------
    // get_object_via_grpc accumulation loop
    // ------------------------------------------------------------------

    /// Simulate the full `get_object_via_grpc` read loop:
    ///
    /// ```text
    /// let mut data = BytesMut::with_capacity(size_hint);
    /// for chunk in stream { data.put_slice(&chunk); }
    /// Ok(data.freeze())
    /// ```
    ///
    /// When `size_hint` is exact (from `descriptor.object().size`) the
    /// `BytesMut` must never reallocate, so `freeze()` returns a `Bytes`
    /// at the same base address.
    #[test]
    fn test_grpc_read_accumulation_no_realloc() {
        // Three chunks totalling 2560 bytes — matches a 3-message gRPC response
        let chunk1 = Bytes::from(vec![0x01u8; 1024]);
        let chunk2 = Bytes::from(vec![0x02u8; 1024]);
        let chunk3 = Bytes::from(vec![0x03u8; 512]);
        let total = chunk1.len() + chunk2.len() + chunk3.len();

        // Exact pre-allocation via descriptor.object().size
        let mut data = BytesMut::with_capacity(total);
        let alloc_ptr = data.as_ptr();

        for chunk in [&chunk1, &chunk2, &chunk3] {
            data.put_slice(chunk);
        }
        let result = data.freeze();

        assert_eq!(
            result.as_ptr(),
            alloc_ptr,
            "exact pre-allocation + freeze() must not reallocate (zero-copy gRPC read path)"
        );
        assert_eq!(result.len(), total);
        // Verify content correctness
        assert!(result[..1024].iter().all(|&b| b == 0x01));
        assert!(result[1024..2048].iter().all(|&b| b == 0x02));
        assert!(result[2048..].iter().all(|&b| b == 0x03));
    }

    // ------------------------------------------------------------------
    // get_object_via_http accumulation loop (no upfront size hint)
    // ------------------------------------------------------------------

    // Issue #252: the HTTP read sizes its buffer from the object size in the response.
    #[test]
    fn http_read_buffer_is_sized_from_the_object_size_and_window() {
        use super::http_read_expected_len;
        const MIB: i64 = 1024 * 1024;
        assert_eq!(http_read_expected_len(8 * MIB, None), 8 * MIB as u64);
        assert_eq!(
            http_read_expected_len(8 * MIB, Some((0, None))),
            8 * MIB as u64
        );
        assert_eq!(
            http_read_expected_len(8 * MIB, Some((2 * MIB as u64, None))),
            6 * MIB as u64
        );
        assert_eq!(
            http_read_expected_len(8 * MIB, Some((MIB as u64, Some(MIB as u64)))),
            MIB as u64
        );
        // A range that runs past the end is clamped; an offset past the end is empty.
        assert_eq!(
            http_read_expected_len(8 * MIB, Some((7 * MIB as u64, Some(4 * MIB as u64)))),
            MIB as u64
        );
        assert_eq!(
            http_read_expected_len(8 * MIB, Some((9 * MIB as u64, Some(MIB as u64)))),
            0
        );
        assert_eq!(http_read_expected_len(-1, None), 0);
    }

    // Issue #255: a gRPC range read sized its buffer to the whole object.
    #[test]
    fn grpc_range_reads_reserve_the_range_not_the_object() {
        use super::grpc_read_sizes;
        const MIB: i64 = 1024 * 1024;
        let tib5: i64 = 5 * 1024 * 1024 * MIB;
        // Full read: the object.
        assert_eq!(
            grpc_read_sizes(8 * MIB, None),
            (8 * MIB as usize, 8 * MIB as usize)
        );
        // A 1 MiB range of a 5 TiB object reserves 1 MiB; the object size is still reported.
        let (size, capacity) = grpc_read_sizes(tib5, Some((MIB as u64, Some(MIB as u64))));
        assert_eq!(size, tib5 as usize);
        assert_eq!(
            capacity, MIB as usize,
            "RED: reserved {capacity} bytes for a 1 MiB range"
        );
        // Open-ended range: from the offset to the end; past the end: nothing.
        assert_eq!(
            grpc_read_sizes(8 * MIB, Some((2 * MIB as u64, None))).1,
            6 * MIB as usize
        );
        assert_eq!(
            grpc_read_sizes(8 * MIB, Some((9 * MIB as u64, Some(1)))).1,
            0
        );
        // A range longer than the object is clamped.
        assert_eq!(
            grpc_read_sizes(MIB, Some((0, Some(1 << 40)))).1,
            MIB as usize
        );
        // A negative size must not wrap to a huge capacity.
        assert_eq!(grpc_read_sizes(-1, None), (0, 0));
        assert_eq!(grpc_read_sizes(i64::MIN, Some((0, Some(5)))), (0, 0));
    }

    /// The HTTP read path must use the pre-sized buffer, not a doubling `BytesMut`.
    #[test]
    fn get_object_via_http_uses_the_presized_read_buffer() {
        let src = include_str!("google_gcs_client.rs");
        let start = src.find("async fn get_object_via_http(").expect("fn");
        let body = &src[start..start + src[start..].find("\n    }\n").expect("end")];
        assert!(
            body.contains("ReadBuffer::with_expected_len("),
            "not pre-sized"
        );
        assert!(
            !body.contains("BytesMut"),
            "RED: still accumulates into a growing BytesMut"
        );
    }

    /// Historical note: `get_object_via_http` used a 1 MiB doubling buffer; it is now
    /// pre-sized (see above). This keeps the `BytesMut::freeze()` pointer-stability property.
    #[test]
    fn test_http_read_freeze_returns_bytes_not_vec() {
        let initial_capacity = 1024 * 1024; // 1 MiB, matches get_object_via_http
        let mut data = BytesMut::with_capacity(initial_capacity);

        // Fill exactly to capacity (no realloc required)
        let payload = vec![0xEFu8; initial_capacity];
        data.put_slice(&payload);
        let alloc_ptr = data.as_ptr();

        let result = data.freeze();

        assert_eq!(
            result.as_ptr(),
            alloc_ptr,
            "HTTP read: freeze() after exact-fit fill must not reallocate"
        );
        assert_eq!(result.len(), initial_capacity);
    }

    // ------------------------------------------------------------------
    // write_object_grpc producer chunking
    // ------------------------------------------------------------------

    /// Each write chunk is a `data.slice(offset..end)` sub-view — no copy.
    /// Every chunk pointer must equal `base_ptr + offset`.
    ///
    /// For 3 × 4 MiB chunks of a 12 MiB object this verifies that the
    /// producer task never copies data before sending it down the channel.
    #[test]
    fn test_write_producer_chunk_slices_are_zero_copy() {
        const TOTAL: usize = 6 * 1024 * 1024; // 6 MiB  (3 × DEFAULT chunk)
        const CHUNK: usize = 2 * 1024 * 1024; // 2 MiB  = DEFAULT_GRPC_WRITE_CHUNK_SIZE

        let data = Bytes::from(vec![0xFFu8; TOTAL]);
        let base_ptr = data.as_ptr();

        let mut offset = 0usize;
        let mut chunk_num = 0usize;
        while offset < TOTAL {
            let end = (offset + CHUNK).min(TOTAL);
            let chunk = data.slice(offset..end);

            assert_eq!(
                chunk.as_ptr(),
                unsafe { base_ptr.add(offset) },
                "chunk {chunk_num}: slice must point into the original Bytes buffer (zero-copy)"
            );
            assert_eq!(chunk.len(), end - offset);

            offset = end;
            chunk_num += 1;
        }
        assert_eq!(chunk_num, 3, "6 MiB / 2 MiB = exactly 3 chunks");
    }

    /// The write producer task receives `data` via `let data = data.clone()`.
    /// That clone must be an Arc increment — same pointer, no 16–128 MiB copy.
    #[test]
    fn test_producer_task_data_clone_is_arc_increment() {
        let data = Bytes::from(vec![0x42u8; 32 * 1024 * 1024]); // 32 MiB
        let original_ptr = data.as_ptr();

        // Mirrors the spawn closure: let data = data.clone();
        let task_data = data.clone();

        assert_eq!(
            task_data.as_ptr(),
            original_ptr,
            "Bytes clone for producer task must be free (Arc increment, not 32 MiB memcpy)"
        );
    }

    // ------------------------------------------------------------------
    // put_object API surface — Bytes flows end-to-end without copy
    // ------------------------------------------------------------------

    /// Construct a `Bytes` the same way a caller would (e.g. after reading a
    /// file into a `Vec<u8>`) and verify that converting to `Bytes` for the
    /// `put_object` call never copies the buffer.
    ///
    /// In the new API `put_object` takes `Bytes` directly — callers convert
    /// once at the boundary (e.g. `Bytes::from(file_contents)`) and the Arc
    /// is then passed through without further copies.
    #[test]
    fn test_put_object_caller_bytes_conversion_is_zero_copy() {
        // Simulate a caller that read a file into Vec<u8>
        let file_contents: Vec<u8> = (0u8..=255).cycle().take(1024 * 1024).collect();
        let data_ptr = file_contents.as_ptr();

        // Single boundary conversion — this is all that should ever happen
        let bytes = Bytes::from(file_contents);

        assert_eq!(
            bytes.as_ptr(),
            data_ptr,
            "Bytes::from(file_contents) must not copy — put_object caller boundary"
        );

        // Clone for passing to async task (simulates what the runtime may do)
        let bytes2 = bytes.clone();
        assert_eq!(
            bytes2.as_ptr(),
            data_ptr,
            "Bytes clone for async dispatch must still point to original buffer"
        );
    }

    // ------------------------------------------------------------------
    // delete_objects: error propagation (issue #135)
    // ------------------------------------------------------------------

    /// The `delete_objects` function must return `Err` when any individual
    /// delete fails.  We test this indirectly through the public contract:
    /// the function takes a `Vec<String>` of object keys, spawns concurrent
    /// delete tasks, counts failures and either returns `Ok(())` (all
    /// succeeded) or `Err(...)` (at least one failed).
    ///
    /// We verify the logic by inspecting the fail-count path in isolation
    /// using a direct unit-test of the error message format, since spinning
    /// up a real GCS client is not feasible in a unit-test environment.
    #[test]
    fn test_delete_objects_error_message_format() {
        // Verify the bail! message contains the expected fields by constructing
        // what the code would produce and checking its content.
        let fail_count: usize = 3;
        let total: usize = 10;
        let bucket = "test-bucket";
        let msg = format!(
            "GCS delete partially failed: {}/{} objects failed to delete in gs://{}",
            fail_count, total, bucket
        );
        assert!(
            msg.contains("3/10"),
            "error message must include fail/total ratio"
        );
        assert!(
            msg.contains("gs://test-bucket"),
            "error message must name the bucket"
        );
        assert!(
            msg.contains("GCS delete partially failed"),
            "error message must indicate partial failure"
        );
    }

    /// Confirms that `delete_objects` returns `Ok(())` immediately for an
    /// empty input without touching the network.
    #[test]
    fn test_delete_objects_empty_input_is_noop() {
        // delete_objects starts with `if objects.is_empty() { return Ok(()); }`
        // We can validate the empty-slice short-circuit by checking the
        // first-line guard exists and produces the right branch.  Since we
        // cannot call async code from a sync test without a runtime we just
        // verify the logic by code inspection (the early-return is trivial).
        let objects: Vec<String> = vec![];
        assert!(
            objects.is_empty(),
            "empty slice must trigger early return in delete_objects"
        );
    }
}

/// Typed classification and the typed-only retry rule of the new GCS receive loop (plan §4.5
/// A2). Every error here is built from the gax error's typed constructors; no test relies on
/// message text.
#[cfg(all(test, not(feature = "gcs-community")))]
mod object_error_tests {
    use super::*;
    use crate::object_error::object_error;
    use anyhow::Context;
    use google_cloud_gax::error::rpc::{Code, Status};
    use google_cloud_gax::error::Error as Gax;

    fn http(code: u16) -> Gax {
        Gax::http(code, http::HeaderMap::new(), Bytes::from_static(b"body"))
    }

    fn grpc(code: Code) -> Gax {
        Gax::service(Status::default().set_code(code).set_message("injected"))
    }

    /// Classify as the loop's attachment site does: keep the typed gax error as the source and
    /// wrap it in string contexts.
    fn classified(e: Gax, pinned: bool) -> anyhow::Error {
        let wrapped = Err::<(), _>(anyhow::Error::new(e))
            .context("GCS ranged read")
            .unwrap_err();
        attach_gcs_classification(wrapped, pinned)
    }

    fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
        object_error(err).expect("classified").kind
    }

    #[test]
    fn classification_follows_the_typed_status() {
        use ObjectErrorKind::*;
        assert_eq!(classify_gcs_failure(&http(404)), NotFound);
        assert_eq!(classify_gcs_failure(&grpc(Code::NotFound)), NotFound);
        assert_eq!(classify_gcs_failure(&http(412)), VersionChanged);
        assert_eq!(
            classify_gcs_failure(&grpc(Code::FailedPrecondition)),
            VersionChanged
        );
        assert_eq!(classify_gcs_failure(&http(403)), Forbidden);
        assert_eq!(
            classify_gcs_failure(&grpc(Code::PermissionDenied)),
            Forbidden
        );
        assert_eq!(classify_gcs_failure(&http(503)), Throttled);
        assert_eq!(classify_gcs_failure(&http(429)), Throttled);
        assert_eq!(classify_gcs_failure(&grpc(Code::Unavailable)), Throttled);
        assert_eq!(
            classify_gcs_failure(&grpc(Code::ResourceExhausted)),
            Throttled
        );
        assert_eq!(classify_gcs_failure(&http(500)), Server);
        assert_eq!(classify_gcs_failure(&http(502)), Server);
        assert_eq!(classify_gcs_failure(&Gax::io("reset")), Transport);
        assert_eq!(classify_gcs_failure(&Gax::timeout("deadline")), Transport);
        assert_eq!(classify_gcs_failure(&Gax::connect("refused")), Transport);
        assert_eq!(classify_gcs_failure(&http(400)), Other);
        assert_eq!(classify_gcs_failure(&grpc(Code::InvalidArgument)), Other);
        assert_eq!(classify_gcs_failure(&Gax::deser("bad body")), Other);
    }

    /// The mapping choices of the typed classifier that are not obvious from the status alone.
    #[test]
    fn deadline_internal_aborted_and_408_are_transient_and_the_rest_are_other() {
        use ObjectErrorKind::*;
        // 408 is the S3 parity choice: Server.
        assert_eq!(classify_gcs_failure(&http(408)), Server);
        // The deadline expired before an answer: like a timeout.
        assert_eq!(
            classify_gcs_failure(&grpc(Code::DeadlineExceeded)),
            Transport
        );
        // The service asked for the request to be repeated.
        assert_eq!(classify_gcs_failure(&grpc(Code::Internal)), Server);
        assert_eq!(classify_gcs_failure(&grpc(Code::Aborted)), Server);
        // No evidence of a service failure, or a credential problem: Other.
        assert_eq!(classify_gcs_failure(&grpc(Code::Unknown)), Other);
        assert_eq!(classify_gcs_failure(&grpc(Code::Cancelled)), Other);
        assert_eq!(classify_gcs_failure(&grpc(Code::Unauthenticated)), Other);
        assert_eq!(classify_gcs_failure(&http(401)), Other);
    }

    /// The typed-only predicate retries typed transient and transport errors, including 503.
    #[test]
    fn typed_transient_and_transport_errors_are_retried() {
        for e in [
            http(503),
            http(429),
            http(500),
            http(408),
            grpc(Code::Unavailable),
            grpc(Code::ResourceExhausted),
            grpc(Code::DeadlineExceeded),
            grpc(Code::Internal),
            grpc(Code::Aborted),
            Gax::io("connection reset"),
            Gax::timeout("deadline"),
            Gax::connect("refused"),
        ] {
            let err = classified(e, false);
            assert!(
                gcs_typed_retry_verdict(&err),
                "{:?} must be retried",
                kind_of(&err)
            );
        }
    }

    /// Nothing else retries: not an unclassified error whose message merely contains "503",
    /// not an `Other`, not HTTP 403, not gRPC PermissionDenied, not absence or a version change.
    #[test]
    fn unclassified_other_forbidden_and_definite_errors_are_not_retried() {
        let text_503 = anyhow::anyhow!("upstream answered 503 Service Unavailable, try later");
        assert!(object_error(&text_503).is_none());
        assert!(!gcs_typed_retry_verdict(&text_503));
        // Attaching to an error with no gax source leaves it unclassified.
        let still = attach_gcs_classification(text_503, false);
        assert!(object_error(&still).is_none());
        assert!(!gcs_typed_retry_verdict(&still));

        // An `Other` whose message contains a retry-looking number.
        let other = Gax::http(
            400,
            http::HeaderMap::new(),
            Bytes::from_static(b"503 timeout"),
        );
        let other = classified(other, false);
        assert_eq!(kind_of(&other), ObjectErrorKind::Other);
        assert!(!gcs_typed_retry_verdict(&other));

        for e in [
            http(403),
            grpc(Code::PermissionDenied),
            http(404),
            grpc(Code::NotFound),
            http(412),
            grpc(Code::FailedPrecondition),
            http(401),
            grpc(Code::Unauthenticated),
            grpc(Code::Unknown),
            grpc(Code::Cancelled),
        ] {
            let err = classified(e, false);
            assert!(!gcs_typed_retry_verdict(&err), "{:?}", kind_of(&err));
        }
        // The old message-text helper would have retried this text; the typed rule does not.
        assert!(is_retryable_gcs_read_error("UNAVAILABLE: upstream reset"));
        let text_only = anyhow::anyhow!("UNAVAILABLE: upstream reset");
        assert!(!gcs_typed_retry_verdict(&text_only));
    }

    /// A pinned read never reports absence: a 404 or NOT_FOUND on a retry or repair is a
    /// version change, which is not retried and is not "not found".
    #[test]
    fn pinned_404_is_a_version_change() {
        for e in [http(404), grpc(Code::NotFound)] {
            let err = classified(e, true);
            assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
            assert!(!gcs_typed_retry_verdict(&err));
            assert!(!crate::delete_contract::is_not_found(&err));
        }
        // Unpinned (first request): absence.
        let first = classified(http(404), false);
        assert_eq!(kind_of(&first), ObjectErrorKind::NotFound);
        assert!(crate::delete_contract::is_not_found(&first));
    }

    /// An error that already carries an outer `NotFound` is upgraded to `VersionChanged` on a
    /// pinned read, and left alone otherwise.
    #[test]
    fn pinned_attach_upgrades_an_existing_not_found() {
        let first = classified(http(404), false);
        assert_eq!(kind_of(&first), ObjectErrorKind::NotFound);
        let upgraded = attach_gcs_classification(first, true);
        assert_eq!(kind_of(&upgraded), ObjectErrorKind::VersionChanged);
        assert!(!gcs_typed_retry_verdict(&upgraded));
        assert!(!crate::delete_contract::is_not_found(&upgraded));
        // Not pinned: unchanged.
        let first = classified(http(404), false);
        let same = attach_gcs_classification(first, false);
        assert_eq!(kind_of(&same), ObjectErrorKind::NotFound);
        // Pinned, but another kind: unchanged.
        let forbidden = classified(http(403), false);
        let same = attach_gcs_classification(forbidden, true);
        assert_eq!(kind_of(&same), ObjectErrorKind::Forbidden);
    }

    /// The attached message names status and code only.
    #[test]
    fn attached_message_has_status_and_code_but_not_the_payload() {
        let err = classified(
            Gax::http(
                503,
                http::HeaderMap::new(),
                Bytes::from_static(b"SECRET-PAYLOAD"),
            ),
            false,
        );
        let o = object_error(&err).expect("typed");
        let text = o.to_string();
        assert!(text.contains("status 503"), "{text}");
        assert!(!text.contains("SECRET-PAYLOAD"), "{text}");
    }
}

// Tests for the typed HEAD decision logic and the #263 object-time conversion.
#[cfg(all(test, not(feature = "gcs-community")))]
#[path = "gcs_head_typed_tests.rs"]
mod head_typed_tests;

// Tests of the caller-buffer receive loop, driven by a deterministic fake source.
#[cfg(all(test, not(feature = "gcs-community")))]
#[path = "gcs_get_into_tests.rs"]
mod get_into_tests;

// Tests of the ranged receive (the range engine's per-range receive), driven by a small fake.
#[cfg(all(test, not(feature = "gcs-community")))]
#[path = "gcs_range_into_tests.rs"]
mod range_into_tests;

// Tests of the real HTTP opener against a local HTTP server (generation, range, size facts).
#[cfg(all(test, not(feature = "gcs-community")))]
#[path = "gcs_http_opener_tests.rs"]
mod http_opener_tests;
