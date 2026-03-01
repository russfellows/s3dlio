// src/google_gcs_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;
use google_cloud_gax::paginator::ItemPaginator;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::{debug, info, trace};

// Global cached GCS clients - initialized once and reused across all operations
static GCS_STORAGE: OnceCell<Arc<Storage>> = OnceCell::const_new();
static GCS_CONTROL: OnceCell<Arc<StorageControl>> = OnceCell::const_new();

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

/// High-level GCS client using official Google google-cloud-storage crate.
///
/// Authentication follows the standard ADC chain:
/// 1. GOOGLE_APPLICATION_CREDENTIALS environment variable (service account JSON)
/// 2. GCE/GKE metadata server (automatic for Google Cloud workloads)
/// 3. gcloud CLI credentials (~/.config/gcloud/application_default_credentials.json)
///
/// # gRPC Transport
///
/// All reads and writes use gRPC (`BidiReadObject` / `BidiWriteObject`) which
/// is faster than the JSON API and works universally for both standard and
/// RAPID/zonal (Hyperdisk ML) buckets.
///
/// # RAPID Storage
///
/// Set the environment variable `S3DLIO_GCS_RAPID=true` (or `1`, `yes`) to
/// enable appendable-object writes for GCS RAPID / Hyperdisk ML buckets.
/// Without this flag, writes to RAPID buckets fail with:
/// *"This bucket requires appendable objects."*
pub struct GcsClient {
    storage: Arc<Storage>,      // For read/write operations
    control: Arc<StorageControl>, // For metadata/list/delete operations
    /// True when writing to a GCS RAPID (Hyperdisk ML) bucket.
    /// Set via S3DLIO_GCS_RAPID=true|1|yes environment variable.
    pub rapid_mode: bool,
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
                
                let client = Storage::builder()
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
        
        // RAPID mode: set appendable=true on every write for RAPID/Hyperdisk ML buckets.
        // All reads/writes use gRPC regardless of RAPID mode.
        let rapid_mode = read_rapid_mode();
        if rapid_mode {
            info!("GCS RAPID mode enabled (gRPC + appendable writes)");
        } else {
            debug!("GCS mode: standard (gRPC reads/writes, S3DLIO_GCS_RAPID not set)");
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

    /// Get entire object as bytes.
    ///
    /// Always uses gRPC BidiReadObject (`open_object`) which works for both
    /// standard and RAPID/zonal GCS buckets.  This avoids the JSON API path
    /// (`read_object`) which is rejected by zonal buckets with HTTP 400.
    pub async fn get_object(&self, bucket: &str, object: &str) -> Result<Bytes> {
        debug!("GCS GET (gRPC): bucket={}, object={}", bucket, object);
        let bucket_name = format_bucket_name(bucket);
        self.get_object_via_grpc(&bucket_name, bucket, object, ReadRange::all()).await
    }

    /// Internal: read an object (or range) via gRPC BidiReadObject (open_object).
    async fn get_object_via_grpc(
        &self,
        bucket_name: &str,
        bucket: &str,
        object: &str,
        range: ReadRange,
    ) -> Result<Bytes> {
        let (_descriptor, mut reader) = self.storage
            .open_object(bucket_name, object)
            .send_and_read(range)
            .await
            .map_err(|e| anyhow!("GCS GET (gRPC) failed for gs://{}/{}: {}", bucket, object, e))?;

        let mut data = Vec::new();
        let mut chunk_count: u32 = 0;
        while let Some(chunk) = reader.next().await.transpose()
            .map_err(|e| anyhow!("GCS GET (gRPC) stream error for gs://{}/{}: {}", bucket, object, e))?
        {
            chunk_count += 1;
            trace!("GCS GET (gRPC) chunk #{}: {} bytes", chunk_count, chunk.len());
            data.extend_from_slice(&chunk);
        }

        debug!("GCS GET (gRPC) success: {} bytes in {} chunk(s)", data.len(), chunk_count);
        Ok(Bytes::from(data))
    }

    /// Get a byte range from an object.
    ///
    /// Always uses gRPC BidiReadObject (`open_object`) which works for both
    /// standard and RAPID/zonal GCS buckets.
    pub async fn get_object_range(
        &self,
        bucket: &str,
        object: &str,
        offset: u64,
        length: Option<u64>,
    ) -> Result<Bytes> {
        debug!(
            "GCS GET RANGE (gRPC): bucket={}, object={}, offset={}, length={:?}",
            bucket, object, offset, length
        );

        let bucket_name = format_bucket_name(bucket);

        // Use ReadRange to specify byte range
        let read_range = match length {
            Some(len) => ReadRange::segment(offset, len),
            None => ReadRange::offset(offset),
        };

        self.get_object_via_grpc(&bucket_name, bucket, object, read_range).await
    }

    /// Upload an object.
    ///
    /// Always uses gRPC BidiWriteObject (`send_grpc`) which works for both
    /// standard and RAPID/zonal GCS buckets.  When RAPID mode is enabled
    /// (`S3DLIO_GCS_RAPID=true`), sets `appendable=true` on the write spec.
    pub async fn put_object(&self, bucket: &str, object: &str, data: &[u8]) -> Result<()> {
        debug!(
            "GCS PUT (gRPC): bucket={}, object={}, size={}, rapid={}",
            bucket, object, data.len(), self.rapid_mode
        );

        let bucket_name = format_bucket_name(bucket);

        // Convert slice to Bytes for upload
        let bytes = Bytes::copy_from_slice(data);

        let mut write = self.storage.write_object(&bucket_name, object, bytes);
        if self.rapid_mode {
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

        const MAX_CONCURRENT_DELETES: usize = 64;

        debug!(
            "GCS DELETE OBJECTS (concurrent): bucket={}, count={}, concurrency={}",
            bucket, objects.len(), MAX_CONCURRENT_DELETES
        );

        let bucket_name = format_bucket_name(bucket);
        let sem = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_DELETES));
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
        data: &[u8],
        _chunk_size: usize,
    ) -> Result<()> {
        // For now, just use regular put_object
        // Google Cloud Storage handles chunking automatically for large objects
        self.put_object(bucket, object, data).await
    }
}


// ---------------------------------------------------------------------------
// Free-standing helpers (pub(crate) so unit tests can exercise them directly)
// ---------------------------------------------------------------------------

/// Read the `S3DLIO_GCS_RAPID` environment variable and return `true` when it
/// is set to `true`, `1`, or `yes` (case-insensitive).  Returns `false` for
/// any other value or when the variable is absent.
///
/// The variable can be set in the shell **or** loaded from a `.env` file via
/// the [`dotenvy`] crate before the binary initializes its GCS client.
pub(crate) fn read_rapid_mode() -> bool {
    std::env::var("S3DLIO_GCS_RAPID")
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(false)
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
    fn test_rapid_mode_unset_is_false() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(!read_rapid_mode(), "absent var should default to false");
    }

    #[test]
    fn test_rapid_mode_true_lowercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "true");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(r);
    }

    #[test]
    fn test_rapid_mode_true_uppercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "TRUE");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(r, "TRUE should enable RAPID mode (case-insensitive)");
    }

    #[test]
    fn test_rapid_mode_one() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "1");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(r, "\"1\" should enable RAPID mode");
    }

    #[test]
    fn test_rapid_mode_yes_lowercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "yes");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(r);
    }

    #[test]
    fn test_rapid_mode_yes_uppercase() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "YES");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(r, "YES should enable RAPID mode (case-insensitive)");
    }

    #[test]
    fn test_rapid_mode_false_string() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "false");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(!r, "\"false\" should NOT enable RAPID mode");
    }

    #[test]
    fn test_rapid_mode_zero() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "0");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(!r, "\"0\" should NOT enable RAPID mode");
    }

    #[test]
    fn test_rapid_mode_unrecognised_value() {
        let _g = ENV_MUTEX.lock().unwrap();
        std::env::set_var("S3DLIO_GCS_RAPID", "maybe");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");
        assert!(!r, "unrecognised value should NOT enable RAPID mode");
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

        assert!(!read_rapid_mode(), "should be off before loading .env");

        dotenvy::from_path(&env_path).expect("dotenvy::from_path should succeed");
        let r = read_rapid_mode();
        std::env::remove_var("S3DLIO_GCS_RAPID");

        assert!(r, "RAPID mode should be on after loading .env with S3DLIO_GCS_RAPID=true");
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

        assert!(!r, "RAPID mode should remain off when .env sets it to false");
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

        assert!(rapid, "RAPID mode should be enabled via .env");
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
        assert!(!rapid, "read_rapid_mode() must respect the shell-set value");
    }
}

