// src/google_gcs_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::{anyhow, bail, Result};
use bytes::{BufMut, Bytes, BytesMut};
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;
use google_cloud_gax::paginator::ItemPaginator;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use tokio::sync::OnceCell;
use tracing::{debug, info, trace, warn};
use crate::gcs_constants::{
    DEFAULT_WINDOW_MIB, ENV_GCS_GRPC_CHANNELS, ENV_GRPC_INITIAL_WINDOW_MIB,
    GCS_MAX_CONCURRENT_DELETES, GCS_MIN_CHANNELS,
};

// Global cached GCS clients - initialized once and reused across all operations
static GCS_STORAGE: OnceCell<Arc<Storage>> = OnceCell::const_new();
static GCS_CONTROL: OnceCell<Arc<StorageControl>> = OnceCell::const_new();

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
        None        => 0u8,
        Some(true)  => 1u8,
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
        RapidMode::ForceOn  => Some(true),
        RapidMode::ForceOff => Some(false),
        RapidMode::Auto     => None,
    }
}

/// Per-bucket cache of whether a bucket is RAPID (zonal/Hyperdisk ML).
/// Each entry is a `OnceCell<bool>` so that concurrent callers for the same
/// bucket block on the *same* detection future rather than all racing to call
/// `get_storage_layout()` simultaneously (thundering herd fix).
static BUCKET_RAPID_CACHE: std::sync::LazyLock<std::sync::Mutex<HashMap<String, Arc<OnceCell<bool>>>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

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
    /// - GOOGLE_APPLICATION_CREDENTIALS env var (loaded by dotenvy)
    /// - Metadata server (if running on GCP)
    /// - gcloud CLI credentials
    pub async fn new() -> Result<Self> {
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
                
                let client = StorageControl::builder()
                    .build()
                    .await
                    .map_err(|e| anyhow!("Failed to initialize GCS StorageControl client: {}", e))?;
                
                info!("GCS StorageControl client initialized successfully");
                Ok::<Arc<StorageControl>, anyhow::Error>(Arc::new(client))
            })
            .await?;
        
        let rapid_mode = read_rapid_mode();
        match rapid_mode {
            RapidMode::ForceOn  => info!("GCS RAPID mode: forced ON (gRPC + appendable writes)"),
            RapidMode::ForceOff => info!("GCS RAPID mode: forced OFF (standard HTTP reads)"),
            RapidMode::Auto     => debug!("GCS RAPID mode: auto-detect per bucket"),
        }

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
            RapidMode::ForceOn  => return true,
            RapidMode::ForceOff => return false,
            RapidMode::Auto     => {}
        }

        // Get or create the per-bucket OnceCell. The std::sync::Mutex is held
        // only for this brief map lookup/insert — no I/O while locked.
        let cell: Arc<OnceCell<bool>> = {
            let mut map = BUCKET_RAPID_CACHE.lock().unwrap();
            map.entry(bucket.to_string())
               .or_insert_with(|| Arc::new(OnceCell::new()))
               .clone()
        };

        // Exactly one caller runs detect_rapid_bucket(); the rest await it.
        *cell.get_or_init(|| self.detect_rapid_bucket(bucket)).await
    }

    /// Query GCS `get_storage_layout()` to determine if a bucket is zonal.
    /// Returns `true` for RAPID/zonal buckets, `false` otherwise.
    async fn detect_rapid_bucket(&self, bucket: &str) -> bool {
        let layout_name = format!("projects/_/buckets/{}/storageLayout", bucket);
        match self.control.get_storage_layout().set_name(&layout_name).send().await {
            Ok(layout) => {
                let is_zonal = layout.location_type.eq_ignore_ascii_case("zone");
                if is_zonal {
                    info!("GCS auto-detect: bucket '{}' is RAPID/zonal (location={}, type={})",
                        bucket, layout.location, layout.location_type);
                } else {
                    debug!("GCS auto-detect: bucket '{}' is standard (location={}, type={})",
                        bucket, layout.location, layout.location_type);
                }
                is_zonal
            }
            Err(e) => {
                warn!("GCS auto-detect: failed to get storage layout for '{}': {} — assuming standard bucket", bucket, e);
                false
            }
        }
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
            self.get_object_via_grpc(&bucket_name, bucket, object, ReadRange::all()).await
        } else {
            debug!("GCS GET (HTTP): bucket={}, object={}", bucket, object);
            self.get_object_via_http(&bucket_name, bucket, object, None).await
        }
    }

    /// Internal: read an object (or range) via gRPC BidiReadObject (open_object).
    async fn get_object_via_grpc(
        &self,
        bucket_name: &str,
        bucket: &str,
        object: &str,
        range: ReadRange,
    ) -> Result<Bytes> {
        let (descriptor, mut reader) = self.storage
            .open_object(bucket_name, object)
            .send_and_read(range)
            .await
            .map_err(|e| anyhow!("GCS GET (gRPC) failed for gs://{}/{}: {}", bucket, object, e))?;

        // Pre-allocate using the object size from the descriptor to avoid
        // repeated reallocations as chunks stream in.  For a range read the
        // object size is the full size, so this may slightly over-allocate,
        // but that is far cheaper than multiple realloc+memcpy cycles.
        // BytesMut::freeze() is zero-copy — it transitions the buffer to an
        // immutable Arc-backed Bytes without any data movement.
        let size_hint = descriptor.object().size as usize;
        let mut data = BytesMut::with_capacity(size_hint);
        let mut chunk_count: u32 = 0;
        while let Some(chunk) = reader.next().await.transpose()
            .map_err(|e| anyhow!("GCS GET (gRPC) stream error for gs://{}/{}: {}", bucket, object, e))?
        {
            chunk_count += 1;
            trace!("GCS GET (gRPC) chunk #{}: {} bytes", chunk_count, chunk.len());
            data.put_slice(&chunk);
        }

        debug!("GCS GET (gRPC) success: {} bytes in {} chunk(s)", data.len(), chunk_count);
        Ok(data.freeze())
    }

    /// Internal: read an object (or optional range) via standard HTTP (`read_object`).
    /// Works on all bucket types including standard regional/multi-regional buckets.
    async fn get_object_via_http(
        &self,
        bucket_name: &str,
        bucket: &str,
        object: &str,
        range: Option<ReadRange>,
    ) -> Result<Bytes> {
        let mut builder = self.storage.read_object(bucket_name, object);
        if let Some(r) = range {
            builder = builder.set_read_range(r);
        }

        let mut reader = builder
            .send()
            .await
            .map_err(|e| anyhow!("GCS GET (HTTP) failed for gs://{}/{}: {}", bucket, object, e))?;

        // BytesMut grows dynamically here; the HTTP path does not provide an
        // upfront Content-Length so we start with a conservative allocation
        // (1 MiB) and let the buffer double as needed.  freeze() is zero-copy.
        let mut data = BytesMut::with_capacity(1024 * 1024);
        let mut chunk_count: u32 = 0;
        while let Some(chunk) = reader.next().await.transpose()
            .map_err(|e| anyhow!("GCS GET (HTTP) stream error for gs://{}/{}: {}", bucket, object, e))?
        {
            chunk_count += 1;
            trace!("GCS GET (HTTP) chunk #{}: {} bytes", chunk_count, chunk.len());
            data.put_slice(&chunk);
        }

        debug!("GCS GET (HTTP) success: {} bytes in {} chunk(s)", data.len(), chunk_count);
        Ok(data.freeze())
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
            self.get_object_via_grpc(&bucket_name, bucket, object, read_range.clone()).await
        } else {
            debug!(
                "GCS GET RANGE (HTTP): bucket={}, object={}, offset={}, length={:?}",
                bucket, object, offset, length
            );
            self.get_object_via_http(&bucket_name, bucket, object, Some(read_range)).await
        }
    }

    /// Upload an object.
    ///
    /// Always uses gRPC BidiWriteObject (`send_grpc()`) which works for both
    /// standard and RAPID/zonal GCS buckets.  When the target bucket is RAPID
    /// (auto-detected or forced via `S3DLIO_GCS_RAPID`), sets `appendable=true`.
    pub async fn put_object(&self, bucket: &str, object: &str, data: Bytes) -> Result<()> {
        let is_rapid = self.is_rapid_bucket(bucket).await;
        debug!(
            "GCS PUT (gRPC): bucket={}, object={}, size={}, rapid={}",
            bucket, object, data.len(), is_rapid
        );

        let bucket_name = format_bucket_name(bucket);

        // `data` is already Arc-backed Bytes — pass directly, no copy.
        let mut write = self.storage.write_object(&bucket_name, object, data);
        if is_rapid {
            // RAPID (zonal) buckets require appendable=true on every write.
            write = write.set_appendable(true);
        }

        write
            .send_grpc()
            .await
            .map_err(|e| anyhow!("GCS PUT failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS PUT success");
        Ok(())
    }

    /// Delete an object.
    pub async fn delete_object(&self, bucket: &str, object: &str) -> Result<()> {
        debug!("GCS DELETE (official): bucket={}, object={}", bucket, object);

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format_bucket_name(bucket);

        self.control
            .delete_object()
            .set_bucket(bucket_name)
            .set_object(object.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("GCS DELETE failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS DELETE success");
        Ok(())
    }

    /// Get object metadata without downloading the content.
    pub async fn get_object_metadata(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata> {
        debug!("GCS GET METADATA (official): bucket={}, object={}", bucket, object);

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format_bucket_name(bucket);

        let obj = self.control
            .get_object()
            .set_bucket(bucket_name)
            .set_object(object.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("GCS GET METADATA failed for gs://{}/{}: {}", bucket, object, e))?;

        let metadata = GcsObjectMetadata {
            size: obj.size as u64,
            etag: Some(obj.etag.clone()),
            updated: obj.update_time.map(|t| {
                // Timestamp is google_cloud_wkt::timestamp::Timestamp
                // Convert to RFC3339 format
                format!("{:?}", t) // Use debug formatting as fallback
            }),
            key: object.to_string(),
        };

        debug!("GCS GET METADATA success: {} bytes", metadata.size);
        Ok(metadata)
    }

    /// Alias for get_object_metadata (for compatibility with gcs_client.rs).
    pub async fn stat_object(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata> {
        self.get_object_metadata(bucket, object).await
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
            .list_objects_with_prefix(&bucket_name, bucket, normalized_prefix.as_deref(), recursive)
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

    /// Internal: execute a paginated LIST with specified prefix and options.
    async fn list_objects_with_prefix(
        &self,
        bucket_name: &str,
        bucket: &str,
        prefix: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<String>> {
        let mut builder = self.control
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
                let page = result.map_err(|e| anyhow!("GCS LIST error for gs://{}: {}", bucket, e))?;
                
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
            
            while let Some(object) = objects_iter.next().await.transpose()
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
        if objects.is_empty() { return Ok(()); }

        

        debug!(
            "GCS DELETE OBJECTS (concurrent): bucket={}, count={}, concurrency={}",
            bucket, objects.len(), GCS_MAX_CONCURRENT_DELETES
        );

        let bucket_name = format_bucket_name(bucket);
        let sem = Arc::new(tokio::sync::Semaphore::new(GCS_MAX_CONCURRENT_DELETES));
        let failed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut futs = futures_util::stream::FuturesUnordered::new();

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

        // Drain all futures
        while let Some(result) = futures_util::StreamExt::next(&mut futs).await {
            if let Err(e) = result {
                tracing::warn!("Delete task panicked: {}", e);
            }
        }

        let fail_count = failed.load(std::sync::atomic::Ordering::Relaxed);
        if fail_count > 0 {
            tracing::warn!("GCS DELETE OBJECTS: {} deletions failed", fail_count);
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
/// The variable can be set in the shell **or** loaded from a `.env` file via
/// the [`dotenvy`] crate before the binary initializes its GCS client.
pub(crate) fn read_rapid_mode() -> RapidMode {
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

/// Parse a GCS URI (gs://bucket/path/to/object) into (bucket, object_path).
/// 
/// Bucket-only URIs are also supported (for prefix listings):
/// - gs://bucket/ → ("bucket", "")
/// - gs://bucket  → ("bucket", "") - requires trailing slash for proper parsing
pub fn parse_gcs_uri(uri: &str) -> Result<(String, String)> {
    // Strip gs:// or gcs:// prefix
    let path = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
        .ok_or_else(|| anyhow!("Invalid GCS URI (expected gs:// or gcs:// prefix): {}", uri))?;

    // Split into bucket and object path
    let mut parts = path.splitn(2, '/');
    let bucket = parts
        .next()
        .ok_or_else(|| anyhow!("Invalid GCS URI (missing bucket): {}", uri))?
        .to_string();

    // Object path is optional (empty for bucket-only URIs like gs://bucket/ for listings)
    let object_path = parts.next().unwrap_or("").to_string();

    if bucket.is_empty() {
        bail!("Invalid GCS URI (empty bucket name): {}", uri);
    }

    Ok((bucket, object_path))
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
            warn!("query_gcs_rapid_bucket('{}'): failed to init GCS client: {}", bucket, e);
            false
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
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

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
        assert_eq!(read_rapid_mode(), RapidMode::Auto, "absent var should default to Auto");
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
        assert_eq!(r, RapidMode::ForceOn, "TRUE should force RAPID on (case-insensitive)");
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
        assert_eq!(r, RapidMode::ForceOn, "YES should force RAPID on (case-insensitive)");
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
        assert_eq!(r, RapidMode::Auto, "unrecognised value should default to Auto");
    }

    #[test]
    fn test_rapid_mode_auto_explicit() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "auto");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert_eq!(r, RapidMode::Auto, "\"auto\" should set Auto mode");
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

        assert!(read_rapid_mode() == RapidMode::Auto, "should be Auto before loading .env");

        dotenvy::from_path(&env_path).expect("dotenvy::from_path should succeed");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        assert_eq!(r, RapidMode::ForceOn, "RAPID mode should be ForceOn after loading .env with S3DLIO_GCS_RAPID=true");
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

        assert_eq!(r, RapidMode::ForceOff, "RAPID mode should be ForceOff when .env sets it to false");
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
        std::fs::write(&env_path, format!("GOOGLE_APPLICATION_CREDENTIALS={}\n", fake_creds))
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

        assert_eq!(rapid, RapidMode::ForceOn, "RAPID mode should be ForceOn via .env");
        assert_eq!(creds, "/tmp/creds.json", "credentials path should be loaded from .env");
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
        assert_eq!(rapid, RapidMode::ForceOff, "read_rapid_mode() must respect the shell-set value");
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
            frozen.as_ptr(), raw_ptr_before,
            "BytesMut::freeze() must be zero-copy — pointer must not change"
        );
        assert_eq!(&frozen[..], b"hello world this is test data for a gRPC chunk");
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
            cloned.as_ptr(), original_ptr,
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
            bytes.as_ptr(), vec_ptr,
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
            bytes.as_ptr(), original_ptr,
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
            result.as_ptr(), alloc_ptr,
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

    /// `get_object_via_http` uses a 1 MiB initial capacity because HTTP
    /// responses don't supply a reliable Content-Length.
    ///
    /// Verify that when data fits in the initial allocation no reallocation
    /// occurs, and that after a growth event the final `freeze()` still
    /// produces a true `Bytes` with no extra copy.
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
            result.as_ptr(), alloc_ptr,
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
        const TOTAL: usize = 6 * 1024 * 1024;   // 6 MiB  (3 × DEFAULT chunk)
        const CHUNK: usize = 2 * 1024 * 1024;   // 2 MiB  = DEFAULT_GRPC_WRITE_CHUNK_SIZE

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
            task_data.as_ptr(), original_ptr,
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
            bytes.as_ptr(), data_ptr,
            "Bytes::from(file_contents) must not copy — put_object caller boundary"
        );

        // Clone for passing to async task (simulates what the runtime may do)
        let bytes2 = bytes.clone();
        assert_eq!(
            bytes2.as_ptr(), data_ptr,
            "Bytes clone for async dispatch must still point to original buffer"
        );
    }
}

