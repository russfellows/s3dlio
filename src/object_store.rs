// src/object_store.rs
//
// Pluggable object-store abstraction.
// Backends: FileSystem, S3, and Azure Blob.
// Features: Checksums, Compression, and Integrity Validation

use anyhow::anyhow;
use anyhow::{bail, Result};
use async_trait::async_trait;
use crc32fast::Hasher;
use std::io::Write;
use std::collections::HashMap;
use tracing::{debug, warn, info, trace};
use regex::Regex;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;

// Helper function for integrity validation
fn compute_checksum(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    format!("{:08x}", hasher.finalize())
}

// --- S3 ----------------------------------------------------------------------
use crate::s3_utils::{
    // Reuse existing S3 helpers
    ObjectStat as S3ObjectStat,
    parse_s3_uri,
    list_objects as s3_list_objects,
    list_objects_stream as s3_list_objects_stream,
    get_object_uri_async as s3_get_object_uri_async,
    get_object_range_uri_async as s3_get_object_range_uri_async,
    stat_object_uri_async as s3_stat_object_uri_async,
    delete_objects as s3_delete_objects,
    create_bucket as s3_create_bucket,
    delete_bucket as s3_delete_bucket,
    // NEW: PUT operations via ObjectStore
    put_object_uri_async as s3_put_object_uri_async,
    put_object_multipart_uri_async as s3_put_object_multipart_uri_async,
};

use crate::s3_logger::global_logger;

// Expose FS adapter (already implemented in src/file_store.rs)
use crate::file_store::FileSystemObjectStore;

// Expose enhanced FS adapter with O_DIRECT support
use crate::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};

// --- Azure ---------------------------------------------------
use bytes::Bytes;
use futures::stream;
use crate::azure_client::{AzureBlob, AzureBlobProperties};
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use crate::constants::{
    DEFAULT_RANGE_ENGINE_CHUNK_SIZE,
    DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,
    DEFAULT_RANGE_ENGINE_THRESHOLD,
    DEFAULT_RANGE_TIMEOUT_SECS,
};
use std::time::Duration;

// --- GCS - Feature-gated backend selection -------------------
#[cfg(feature = "gcs-community")]
use crate::gcs_client::{GcsClient, GcsObjectMetadata, parse_gcs_uri};

#[cfg(feature = "gcs-official")]
use crate::google_gcs_client::{GcsClient, GcsObjectMetadata, parse_gcs_uri};

/// Provider-neutral object metadata. For now this aliases S3's metadata.
pub type ObjectMetadata = S3ObjectStat;

// -----------------------------------------------------------------------------
// Performance Configuration Helpers
// -----------------------------------------------------------------------------

/// Get the threshold size for using concurrent range requests.
/// Objects/transfers above this size will use concurrent range GET for optimal performance.
fn get_concurrent_threshold() -> u64 {
    std::env::var("S3DLIO_CONCURRENT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32 * 1024 * 1024) // Default: 32MB threshold
}

/// Compression configuration for ObjectWriter implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionConfig {
    None,
    Zstd { level: i32 },
}

impl Default for CompressionConfig {
    fn default() -> Self {
        CompressionConfig::None
    }
}

impl CompressionConfig {
    /// Create Zstd compression with default level (3)
    pub fn zstd_default() -> Self {
        CompressionConfig::Zstd { level: 3 }
    }
    
    /// Create Zstd compression with custom level (1-22)
    pub fn zstd_level(level: i32) -> Self {
        CompressionConfig::Zstd { level: level.clamp(1, 22) }
    }
    
    /// Get the file extension for this compression format
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionConfig::None => "",
            CompressionConfig::Zstd { .. } => ".zst",
        }
    }
    
    /// Check if compression is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, CompressionConfig::None)
    }
}


/// Configuration options for creating object writers
#[derive(Debug, Clone, Default)]
pub struct WriterOptions {
    /// Optional compression configuration
    pub compression: Option<CompressionConfig>,
    /// Buffer size hint for streaming operations
    pub buffer_size: Option<usize>,
    /// Maximum part size for multipart uploads
    pub part_size: Option<usize>,
    /// Optional adaptive configuration for auto-tuning (default: disabled)
    pub adaptive: Option<crate::adaptive_config::AdaptiveConfig>,
}

impl WriterOptions {
    /// Create new WriterOptions with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set compression configuration
    pub fn with_compression(mut self, compression: CompressionConfig) -> Self {
        self.compression = Some(compression);
        self
    }
    
    /// Set buffer size hint
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }
    
    /// Set maximum part size for multipart uploads
    pub fn with_part_size(mut self, size: usize) -> Self {
        self.part_size = Some(size);
        self
    }
    
    /// Enable adaptive tuning with default configuration
    pub fn with_adaptive(mut self) -> Self {
        self.adaptive = Some(crate::adaptive_config::AdaptiveConfig::enabled());
        self
    }
    
    /// Set custom adaptive configuration
    pub fn with_adaptive_config(mut self, config: crate::adaptive_config::AdaptiveConfig) -> Self {
        self.adaptive = Some(config);
        self
    }
    
    /// Compute effective part size considering adaptive tuning
    /// 
    /// If part_size is explicitly set, it is always used.
    /// Otherwise, if adaptive is enabled, it computes the optimal part size.
    /// Falls back to default if neither is set.
    pub fn effective_part_size(&self, file_size: Option<usize>) -> usize {
        use crate::adaptive_config::AdaptiveParams;
        
        // If adaptive config is provided, use it to compute effective part size
        if let Some(ref adaptive_cfg) = self.adaptive {
            let params = AdaptiveParams::new(adaptive_cfg.clone());
            params.compute_part_size(file_size, self.part_size)
        } else {
            // No adaptive config: use explicit part_size or default
            self.part_size.unwrap_or(crate::constants::DEFAULT_S3_MULTIPART_PART_SIZE)
        }
    }
    
    /// Compute effective buffer size considering adaptive tuning
    pub fn effective_buffer_size(&self, operation_type: &str) -> usize {
        use crate::adaptive_config::AdaptiveParams;
        
        // If adaptive config is provided, use it to compute effective buffer size
        if let Some(ref adaptive_cfg) = self.adaptive {
            let params = AdaptiveParams::new(adaptive_cfg.clone());
            params.compute_buffer_size(operation_type, self.buffer_size)
        } else {
            // No adaptive config: use explicit buffer_size or default
            self.buffer_size.unwrap_or(1024 * 1024) // 1 MB default
        }
    }
}
/// Supported schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    File,
    Direct,
    S3,
    Azure,
    Gcs,
    Unknown,
}
/// Object properties for metadata update operations (v0.10.0+)
/// 
/// Used with `update_properties()` to modify object metadata without re-uploading data.
/// Only non-None fields will be updated. For cloud backends, this typically requires
/// copying the object with new metadata.
#[derive(Debug, Clone, Default)]
pub struct ObjectProperties {
    /// MIME content type (e.g., "application/json", "text/plain")
    pub content_type: Option<String>,
    
    /// HTTP Cache-Control header (e.g., "max-age=3600, public")
    pub cache_control: Option<String>,
    
    /// Content encoding (e.g., "gzip", "br")
    pub content_encoding: Option<String>,
    
    /// Content language (e.g., "en-US", "fr-FR")
    pub content_language: Option<String>,
    
    /// Content disposition (e.g., "attachment; filename=data.json")
    pub content_disposition: Option<String>,
    
    /// HTTP Expires header (RFC 2822 date format)
    pub expires: Option<String>,
    
    /// Storage class/tier for cost optimization
    /// - S3: "STANDARD", "INTELLIGENT_TIERING", "GLACIER", "DEEP_ARCHIVE"
    /// - GCS: "STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"
    /// - Azure: "Hot", "Cool", "Archive"
    pub storage_class: Option<String>,
}


/// Best-effort scheme inference from a URI.
pub fn infer_scheme(uri: &str) -> Scheme {
    if uri.starts_with("file://") { Scheme::File }
    else if uri.starts_with("direct://") { Scheme::Direct }
    else if uri.starts_with("s3://") { Scheme::S3 }
    else if uri.starts_with("az://") || uri.contains(".blob.core.windows.net/") { Scheme::Azure }
    else if uri.starts_with("gs://") || uri.starts_with("gcs://") { Scheme::Gcs }
    else { Scheme::Unknown }
}

#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get entire object into memory.
    /// 
    /// Returns `Bytes` for zero-copy efficiency. The S3/Azure SDKs return data as `Bytes`,
    /// allowing us to avoid unnecessary allocations. Use `.as_ref()` to get &[u8] or
    /// `.to_vec()` only when ownership is required.
    async fn get(&self, uri: &str) -> Result<Bytes>;

    /// Get a byte-range. If `length` is None, read from `offset` to end.
    /// 
    /// Returns `Bytes` for zero-copy efficiency.
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes>;

    /// Put full object (single-shot).
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()>;

    /// Put via multipart semantics (or an equivalent high-throughput path).
    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()>;

    /// List objects under a prefix. Returns full URIs.
    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>>;

    /// List objects as a stream for memory-efficient iteration over large result sets.
    /// Results are yielded as they arrive from the backend (typically in 1000-object pages).
    /// This allows displaying progress and avoids buffering millions of URIs in memory.
    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;

    /// Stat a single object (HEAD-like).
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata>;

    /// Delete a single object.
    async fn delete(&self, uri: &str) -> Result<()>;

    /// Delete multiple objects efficiently (batched when backend supports it).
    /// Backends should override for optimal batch delete (e.g., S3 DeleteObjects API).
    async fn delete_batch(&self, uris: &[String]) -> Result<()>;

    /// Delete all objects under a prefix.
    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()>;

    /// Create a top-level container (S3: bucket; Azure: container; File: directory).
    async fn create_container(&self, name: &str) -> Result<()>;

    /// Delete a top-level container.
    async fn delete_container(&self, name: &str) -> Result<()>;

    /// Check existence via `stat()`. Backends can override for efficiency.
    async fn exists(&self, uri: &str) -> Result<bool> {
        Ok(self.stat(uri).await.is_ok())
    }
    
    /// Get object with integrity validation.
    /// Returns the data and validates it against the expected checksum if provided.
    async fn get_with_validation(&self, uri: &str, expected_checksum: Option<&str>) -> Result<Bytes> {
        let data = self.get(uri).await?;
        
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Integrity validation failed for {}: expected {}, got {}", uri, expected, actual);
            }
        }
        
        Ok(data)
    }
    
    /// Get byte range with integrity validation.
    /// For partial reads, validates against the partial data checksum if provided.
    async fn get_range_with_validation(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        expected_checksum: Option<&str>
    ) -> Result<Bytes> {
        let data = self.get_range(uri, offset, length).await?;
        
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Integrity validation failed for range {}[{}:{}]: expected {}, got {}", 
                      uri, offset, offset + data.len() as u64, expected, actual);
            }
        }
        
        Ok(data)
    }
    
    /// Load and validate checkpoint data with integrity checking.
    /// This method is specifically designed for checkpoint loading scenarios.
    async fn load_checkpoint_with_validation(
        &self, 
        checkpoint_uri: &str, 
        expected_checksum: Option<&str>
    ) -> Result<Vec<u8>> {
        // Get checkpoint data
        let data = self.get(checkpoint_uri).await?;
        
        // Validate integrity if checksum provided
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Checkpoint integrity validation failed for {}: expected checksum {}, got {}", 
                      checkpoint_uri, expected, actual);
            }
        }
        
        // Convert Bytes to Vec<u8> for backward compatibility
        Ok(data.to_vec())
    }

    /// Default copy reads then writes. Backends can override with server-side copy.
    async fn copy(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        let data = self.get(src_uri).await?;
        self.put(dst_uri, &data).await
    }

    /// Atomic rename/move operation. For file:// backends, this uses filesystem rename
    /// which is atomic. For cloud backends, this typically falls back to copy+delete.
    /// Returns an error if the operation cannot be completed atomically.
    async fn rename(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        // Default implementation: copy then delete (not atomic, but works)
        self.copy(src_uri, dst_uri).await?;
        self.delete(src_uri).await
    }

    /// Get a streaming writer for zero-copy operations.
    /// This enables writing large objects without buffering the entire content in memory.
    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>>;

    /// Get a streaming writer with compression support for zero-copy operations.
    /// This enables writing large objects with compression without buffering the entire content in memory.
    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        // Default implementation: use get_writer (no compression)
        if compression.is_enabled() {
            bail!("Compression not supported by this ObjectStore implementation");
        }
        self.get_writer(uri).await
    }

    /// Create a new streaming writer with specified options.
    /// This is the unified interface for creating writers with configurable options.
    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        // Default implementation: use get_writer_with_compression if compression is specified
        if let Some(compression) = options.compression {
            self.get_writer_with_compression(uri, compression).await
        } else {
            self.get_writer(uri).await
        }
    }

    /// High-performance optimized GET operation for large objects.
    /// Automatically chooses between single GET and concurrent range requests
    /// based on object size and configured thresholds. This method provides
    /// the best performance for large object retrieval.
    async fn get_optimized(&self, uri: &str) -> Result<Bytes> {
        // Default implementation: delegate to regular get()
        // S3ObjectStore will override this with concurrent range logic
        self.get(uri).await
    }

    /// High-performance optimized range GET operation.
    /// Uses concurrent range requests for large transfers to maximize throughput.
    async fn get_range_optimized(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        _chunk_size: Option<usize>,
        _max_concurrency: Option<usize>
    ) -> Result<Bytes> {
        // Default implementation: delegate to regular get_range()
        // S3ObjectStore will override this with concurrent range logic
        self.get_range(uri, offset, length).await
    }

    /// Pre-stat multiple objects concurrently to populate size cache (v0.9.10+)
    /// 
    /// This is a performance optimization for workloads where object URIs are known
    /// upfront (e.g., benchmark tools like sai3-bench, batch processing pipelines).
    /// By pre-statting objects concurrently, we eliminate per-object stat latency
    /// during the download phase.
    /// 
    /// # Performance Impact
    /// 
    /// For 1000 object benchmark workload:
    /// - Without pre-stat: 1000 × 20ms stat = 20 seconds overhead (61% of total time)
    /// - With pre-stat: Pre-stat in 200ms (100 concurrent), then zero per-object overhead
    /// - Result: 2.5x faster (32.8s → 13.0s), 2.5x higher throughput (1.95 GB/s → 4.92 GB/s)
    /// 
    /// # Arguments
    /// 
    /// * `uris` - List of object URIs to stat
    /// * `max_concurrent` - Maximum concurrent stat operations (recommended: 100)
    /// 
    /// # Returns
    /// 
    /// Map of URI → size for successfully statted objects. Failed stats are logged
    /// and omitted from the result (graceful degradation).
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// use s3dlio::object_store::ObjectStore;
    /// 
    /// # async fn example(store: &dyn ObjectStore) -> anyhow::Result<()> {
    /// let uris = vec![
    ///     "s3://bucket/object1.dat".to_string(),
    ///     "s3://bucket/object2.dat".to_string(),
    ///     // ... 1000 more objects
    /// ];
    /// 
    /// // Pre-stat all objects concurrently
    /// let size_map = store.pre_stat_objects(&uris, 100).await?;
    /// println!("Pre-statted {} objects", size_map.len());
    /// 
    /// // Now downloads can use cached sizes (if backend supports it)
    /// for uri in &uris {
    ///     let data = store.get(uri).await?;
    ///     // No stat overhead here!
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// 
    /// # Backward Compatibility
    /// 
    /// This method has a default implementation that stats objects CONCURRENTLY
    /// using the provided max_concurrent limit. No need to override unless you
    /// want custom behavior.
    async fn pre_stat_objects(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<std::collections::HashMap<String, u64>> {
        use futures::stream::{self, StreamExt};
        
        tracing::debug!("Pre-statting {} objects (concurrent, max={})", uris.len(), max_concurrent);
        
        // Clone URIs to avoid lifetime issues in async closures
        let uri_vec: Vec<String> = uris.to_vec();
        
        // Use futures::stream to stat objects concurrently
        let results: Vec<Option<(String, u64)>> = stream::iter(uri_vec)
            .map(|uri| async move {
                match self.stat(&uri).await {
                    Ok(metadata) => {
                        tracing::trace!("Pre-stat success: {} ({} bytes)", uri, metadata.size);
                        Some((uri, metadata.size))
                    }
                    Err(e) => {
                        tracing::warn!("Pre-stat failed for {}: {}", uri, e);
                        None
                    }
                }
            })
            .buffer_unordered(max_concurrent)
            .collect()
            .await;
        
        // Collect successful results
        let size_map: std::collections::HashMap<String, u64> = results
            .into_iter()
            .flatten()
            .collect();
        
        tracing::info!(
            "Pre-statted {}/{} objects successfully (concurrent)",
            size_map.len(),
            uris.len()
        );
        
        Ok(size_map)
    }

    /// Pre-stat objects and populate internal size cache (v0.9.10+)
    /// 
    /// This is a higher-level convenience method that pre-stats objects AND caches
    /// the results internally. After calling this, subsequent `get()` calls will
    /// use cached sizes and skip the per-object stat operation.
    /// 
    /// # Arguments
    /// 
    /// * `uris` - List of object URIs to stat
    /// * `max_concurrent` - Maximum concurrent stat operations (recommended: 100)
    /// 
    /// # Returns
    /// 
    /// Count of successfully cached entries
    /// 
    /// # Example (sai3-bench usage pattern)
    /// 
    /// ```no_run
    /// use s3dlio::object_store::ObjectStore;
    /// use std::time::Instant;
    /// 
    /// # async fn example(store: &dyn ObjectStore) -> anyhow::Result<()> {
    /// let object_uris: Vec<String> = vec![/* 1000 objects */];
    /// 
    /// // PHASE 1: Pre-stat all objects (runs once at start)
    /// let start = Instant::now();
    /// let cached = store.pre_stat_and_cache(&object_uris, 100).await?;
    /// println!("Pre-statted {} objects in {:?}", cached, start.elapsed());
    /// 
    /// // PHASE 2: Download with zero stat overhead
    /// for uri in &object_uris {
    ///     let data = store.get(uri).await?;  // Uses cached size!
    ///     // Process data...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// 
    /// # Backward Compatibility
    /// 
    /// Default implementation calls `pre_stat_objects()` but doesn't cache results
    /// (returns the count but doesn't enable size cache benefits). Backends with
    /// size cache support will override this method.
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Default: just pre-stat without caching (backward compatible no-op)
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;
        Ok(size_map.len())
    }
    
    // =========================================================================
    // Metadata Operations (v0.10.0+)
    // =========================================================================
    
    /// Create a directory (POSIX) or prefix marker (cloud).
    /// 
    /// **Backend behavior**:
    /// - `file://`, `direct://`: Creates actual directory with `create_dir_all()`
    /// - `s3://`, `gs://`, `az://`: Creates empty marker object (e.g., `.keep`)
    /// 
    /// # Arguments
    /// * `uri` - Full URI including scheme (e.g., "file:///tmp/dir", "s3://bucket/prefix/")
    async fn mkdir(&self, uri: &str) -> Result<()> {
        let _ = uri;
        bail!("mkdir not implemented for this backend")
    }

    /// Remove a directory (POSIX) or delete all objects under prefix (cloud).
    /// 
    /// **Backend behavior**:
    /// - `file://`, `direct://`: Removes directory (must be empty unless `recursive=true`)
    /// - `s3://`, `gs://`, `az://`: Deletes all objects under prefix
    /// 
    /// # Arguments
    /// * `uri` - Full URI including scheme
    /// * `recursive` - If true, delete recursively; if false, fail if not empty
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let _ = (uri, recursive);
        bail!("rmdir not implemented for this backend")
    }

    /// Update custom object metadata (cloud-specific, x-amz-meta-*, x-goog-meta-*, x-ms-meta-*).
    /// 
    /// **Note**: For cloud backends, this typically requires copying the object with new metadata.
    /// For file backends, this is not applicable.
    async fn update_metadata(&self, uri: &str, metadata: &HashMap<String, String>) -> Result<()> {
        let _ = (uri, metadata);
        bail!("update_metadata not supported for this backend")
    }

    /// Update object properties (content-type, cache-control, storage class, etc).
    /// 
    /// **Note**: For cloud backends, changing properties requires copying the object.
    /// For file backends, limited support.
    async fn update_properties(&self, uri: &str, properties: &ObjectProperties) -> Result<()> {
        let _ = (uri, properties);
        bail!("update_properties not supported for this backend")
    }


}

/// Concurrent deletion helper for efficient batch deletions across all backends.
/// 
/// This function deletes multiple objects concurrently through the ObjectStore trait,
/// providing significant performance improvements over sequential deletion while
/// maintaining universal backend compatibility.
/// 
/// # Adaptive Concurrency
/// - For small batches (< 10 objects): Sequential deletion (concurrency = 1)
/// - For medium batches: 10% of total objects (with min 10, max 1000)
/// - For large batches (10,000+ objects): Caps at 1000 concurrent operations
/// 
/// # Progress Updates
/// - Batched updates (every 50 operations) to minimize overhead
/// - Allows 98% reduction in progress bar overhead vs per-object updates
/// - Final update ensures accurate completion count
/// 
/// # Arguments
/// * `store` - Any ObjectStore implementation (S3, Azure, GCS, file, direct)
/// * `keys` - Slice of object URIs to delete
/// * `progress_callback` - Optional callback for progress updates (called every 50 deletions)
/// 
/// # Returns
/// * `Ok(())` - All objects deleted successfully
/// * `Err(_)` - First error encountered (other deletions may still be in progress)
/// 
/// # Example
/// ```ignore
/// let keys = vec!["s3://bucket/obj1", "s3://bucket/obj2"];
/// delete_objects_concurrent(&store, &keys, Some(|count| {
///     println!("Deleted {} objects", count);
/// })).await?;
/// ```
pub async fn delete_objects_concurrent<F>(
    store: &dyn ObjectStore,
    keys: &[String],
    progress_callback: Option<F>,
) -> Result<()>
where
    F: Fn(usize) + Send + Sync + 'static,
{
    use futures::stream::{self, StreamExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let total = keys.len();
    
    if total == 0 {
        return Ok(());
    }

    // Adaptive concurrency: 10% of total objects with reasonable min/max bounds
    let max_concurrency = if total < 10 {
        1  // Very small batches: sequential
    } else if total < 100 {
        10  // Small batches: minimum 10 concurrent
    } else if total < 10_000 {
        (total / 10).max(10).min(100)  // Medium batches: 10% with max 100
    } else {
        (total / 10).min(1000)  // Large batches: 10% capped at 1000
    };

    let completed = Arc::new(AtomicUsize::new(0));
    let last_reported = Arc::new(AtomicUsize::new(0));
    
    // Wrap callback in Arc for sharing across async tasks
    let callback_arc = progress_callback.map(Arc::new);
    
    // Progress reporting frequency: every 50 deletions or at completion
    let report_interval = 50;

    // Create concurrent deletion stream
    let deletions = stream::iter(keys.iter().enumerate())
        .map(|(idx, key)| {
            let key = key.clone();
            let completed = Arc::clone(&completed);
            let last_reported = Arc::clone(&last_reported);
            let callback = callback_arc.clone();
            
            async move {
                // Delete the object
                store.delete(&key).await?;
                
                // Update counter
                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                
                // Report progress in batches or on completion
                if let Some(ref cb) = callback {
                    let last = last_reported.load(Ordering::Relaxed);
                    if count - last >= report_interval || count == total || idx == keys.len() - 1 {
                        if last_reported.compare_exchange(last, count, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                            cb(count);
                        }
                    }
                }
                
                Ok::<_, anyhow::Error>(())
            }
        })
        .buffer_unordered(max_concurrency);

    // Execute all deletions and collect results
    let results: Vec<Result<()>> = deletions.collect().await;
    
    // Check for errors
    for result in results {
        result?;
    }

    // Final progress update to ensure accurate count
    if let Some(callback) = callback_arc {
        let final_count = completed.load(Ordering::Relaxed);
        callback(final_count);
    }

    Ok(())
}

/// Streaming writer interface for zero-copy object uploads
#[async_trait]
pub trait ObjectWriter: Send + Sync {
    /// Write a chunk of data to the object stream.
    /// Chunks are written in order and the implementation handles buffering/uploading.
    /// If compression is enabled, data is compressed before storage.
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()>;

    /// Write owned bytes for zero-copy optimization.
    /// Takes ownership of the data to avoid copying.
    async fn write_owned_bytes(&mut self, data: Vec<u8>) -> Result<()> {
        // Default implementation: convert to slice and call write_chunk
        self.write_chunk(&data).await
    }
    
    /// Finalize the object upload. Must be called to complete the write.
    /// After calling finalize(), the writer should not be used again.
    async fn finalize(self: Box<Self>) -> Result<()>;
    
    /// Get the total number of uncompressed bytes written so far.
    fn bytes_written(&self) -> u64;
    
    /// Get the total number of compressed bytes (if compression is enabled).
    /// Returns the same as bytes_written() if compression is disabled.
    fn compressed_bytes(&self) -> u64 {
        self.bytes_written() // Default: no compression
    }
    
    /// Get the computed checksum for the uncompressed data written so far.
    /// Returns None if no checksum has been computed yet.
    fn checksum(&self) -> Option<String>;
    
    /// Get compression configuration for this writer.
    fn compression(&self) -> CompressionConfig {
        CompressionConfig::None // Default: no compression
    }
    
    /// Get compression ratio (compressed_size / uncompressed_size).
    /// Returns 1.0 if compression is disabled.
    fn compression_ratio(&self) -> f64 {
        if self.bytes_written() == 0 {
            1.0
        } else {
            self.compressed_bytes() as f64 / self.bytes_written() as f64
        }
    }
    
    /// Cancel the upload and clean up any partial data.
    /// This is called automatically if the writer is dropped without finalize().
    async fn cancel(self: Box<Self>) -> Result<()> {
        // Default implementation: just drop
        Ok(())
    }
}

/// Default buffered implementation that collects chunks and calls put()
pub struct BufferedObjectWriter<'a> {
    uri: String,
    store: &'a dyn ObjectStore,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
}

impl<'a> BufferedObjectWriter<'a> {
    pub fn new(uri: String, store: &'a dyn ObjectStore) -> Self {
        Self {
            uri,
            store,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
        }
    }
}

#[async_trait]
impl<'a> ObjectWriter for BufferedObjectWriter<'a> {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        self.buffer.extend_from_slice(chunk);
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        self.store.put(&self.uri, &self.buffer).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        Ok(())
    }
}

// ============================================================================
// FileSystem adapter (already implemented)
// ============================================================================
impl FileSystemObjectStore {
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }
}

// ============================================================================
// S3 adapter that calls straight into your existing helpers
// ============================================================================

/// Configuration for S3ObjectStore
/// 
/// v0.9.10: Added size_cache_ttl for pre-stat optimization
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Enable RangeEngine for concurrent range downloads
    /// Default: false (v0.9.6+) - must opt-in to avoid stat overhead on every GET
    pub enable_range_engine: bool,
    
    /// RangeEngine configuration
    /// Network-optimized defaults: 16 MiB threshold, 32 concurrent ranges, 64 MiB chunks
    pub range_engine: RangeEngineConfig,
    
    /// Time-to-live for cached object sizes
    /// Default: 60 seconds
    /// Set to 0 to disable caching
    pub size_cache_ttl_secs: u64,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            enable_range_engine: false,  // Disabled by default due to stat overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE,  // 64 MiB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,  // 32 parallel
                min_split_size: DEFAULT_RANGE_ENGINE_THRESHOLD,  // 16 MiB threshold
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS),  // 30s
            },
            size_cache_ttl_secs: 60,  // 60 second TTL for size cache
        }
    }
}

#[derive(Clone)]
pub struct S3ObjectStore {
    // Note: S3ObjectStore doesn't store config because it doesn't support RangeEngine yet
    // (unlike GCS/Azure which do). The config is only used during construction to set
    // the cache TTL. If RangeEngine support is added later, we can add the config field.
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
}

impl S3ObjectStore {
    pub fn new() -> Self {
        let config = S3Config::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }
    
    pub fn with_config(config: S3Config) -> Self {
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }
    
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self::new()) }
    
    /// Get object size, checking cache first
    /// 
    /// v0.9.10: Added to optimize get_optimized() and get_range_optimized()
    /// by eliminating redundant stat() calls.
    async fn get_object_size(&self, uri: &str) -> Result<u64> {
        // Check cache first
        if let Some(cached_size) = self.size_cache.get(uri).await {
            return Ok(cached_size);
        }
        
        // Cache miss - perform stat and cache result
        let metadata = self.stat(uri).await?;
        self.size_cache.put(uri.to_string(), metadata.size).await;
        Ok(metadata.size)
    }
}

#[async_trait]
impl ObjectStore for S3ObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_get_object_uri_async(uri).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_get_object_range_uri_async(uri, offset, length).await
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_put_object_uri_async(uri, data).await
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_put_object_multipart_uri_async(uri, data, part_size).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (bucket, key_prefix) = parse_s3_uri(uri_prefix)?;
        let keys = s3_list_objects(&bucket, &key_prefix, recursive)?;
        Ok(keys.into_iter().map(|k| format!("s3://{}/{}", bucket, k)).collect())
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let (bucket, key_prefix) = match parse_s3_uri(uri_prefix) {
                Ok((b, k)) => (b, k),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            
            let mut stream = match s3_list_objects_stream(bucket.clone(), key_prefix, recursive).await {
                Ok(s) => s,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(key) => yield Ok(format!("s3://{}/{}", bucket, key)),
                    Err(e) => yield Err(e),
                }
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_stat_object_uri_async(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        let (bucket, key) = parse_s3_uri(uri)?;
        s3_delete_objects(&bucket, &vec![key])
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        if uris.is_empty() { return Ok(()); }
        
        // Extract bucket and keys from URIs
        let (bucket, _) = parse_s3_uri(&uris[0])?;
        let keys: Vec<String> = uris.iter()
            .filter_map(|uri| parse_s3_uri(uri).ok().map(|(_, key)| key))
            .collect();
        
        // Use S3 batch delete API (1000 objects per request)
        s3_delete_objects(&bucket, &keys)
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (bucket, mut key_prefix) = parse_s3_uri(uri_prefix)?;
        if !key_prefix.is_empty() && !key_prefix.ends_with('/') { key_prefix.push('/'); }
        let keys = s3_list_objects(&bucket, &key_prefix, true)?;
        if keys.is_empty() { return Ok(()); }
        s3_delete_objects(&bucket, &keys)
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        s3_create_bucket(name)
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        s3_delete_bucket(name)
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For S3, use buffered writer that collects chunks then uses put_multipart
        Ok(Box::new(S3BufferedWriter::new(uri.to_string())))
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        // For S3, use buffered writer with compression support
        Ok(Box::new(S3BufferedWriter::new_with_compression(uri.to_string(), compression)))
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        // For S3, use buffered writer that collects chunks then uses put_multipart
        if let Some(compression) = options.compression {
            Ok(Box::new(S3BufferedWriter::new_with_compression(uri.to_string(), compression)))
        } else {
            Ok(Box::new(S3BufferedWriter::new(uri.to_string())))
        }
    }

    async fn get_optimized(&self, uri: &str) -> Result<Bytes> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        
        // Get object size from cache or stat (v0.9.10: cache optimization)
        let object_size = self.get_object_size(uri).await?;
        
        // Use threshold-based decision for optimization
        let threshold = get_concurrent_threshold();
        
        if object_size >= threshold {
            debug!("Using concurrent range GET for large object: {} bytes", object_size);
            // Use concurrent range GET for large objects
            crate::s3_utils::get_object_concurrent_range_async(uri, 0, None, None, None).await
        } else {
            debug!("Using standard GET for small object: {} bytes", object_size);
            // Use standard GET for small objects
            self.get(uri).await
        }
    }

    async fn get_range_optimized(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        chunk_size: Option<usize>,
        max_concurrency: Option<usize>
    ) -> Result<Bytes> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        
        let transfer_size = match length {
            Some(len) => len,
            None => {
                // Get object size from cache or stat (v0.9.10: cache optimization)
                let object_size = self.get_object_size(uri).await?;
                object_size.saturating_sub(offset)
            }
        };
        
        // Use threshold-based decision for optimization
        let threshold = get_concurrent_threshold();
        
        if transfer_size >= threshold {
            debug!("Using concurrent range GET for large transfer: {} bytes", transfer_size);
            // Use concurrent range GET for large transfers
            crate::s3_utils::get_object_concurrent_range_async(uri, offset, length, chunk_size, max_concurrency).await
        } else {
            debug!("Using standard range GET for small transfer: {} bytes", transfer_size);
            // Use standard range GET for small transfers
            self.get_range(uri, offset, length).await
        }
    }
    
    /// Pre-stat objects and populate the size cache
    /// 
    /// v0.9.10: Override default implementation to populate internal size cache.
    /// This enables subsequent get_optimized() calls to skip redundant stat operations.
    /// 
    /// # Performance Impact
    /// 
    /// For workloads that download many objects (e.g., benchmarking 1000+ objects):
    /// - Eliminates per-object stat latency (typically 10-50ms each)
    /// - Trades one-time concurrent pre-stat (e.g., 200ms for 1000 objects @ 100 concurrent)
    ///   for N × stat_latency savings (e.g., 1000 × 20ms = 20 seconds)
    /// - Expected speedup: 2-3x for large object sets
    /// 
    /// # Example
    /// 
    /// ```rust,no_run
    /// use s3dlio::api::store_for_uri;
    /// 
    /// # async fn example() -> anyhow::Result<()> {
    /// let store = store_for_uri("s3://my-bucket/prefix/")?;
    /// let objects: Vec<String> = vec![/* 1000 s3:// URIs */];
    /// 
    /// // Pre-stat phase (once at start)
    /// let cached = store.pre_stat_and_cache(&objects, 100).await?;
    /// println!("Cached {} object sizes", cached);
    /// 
    /// // Download phase (benefits from cached sizes)
    /// for uri in &objects {
    ///     let data = store.get(&uri).await?;  // No stat overhead!
    /// }
    /// # Ok(())
    /// # }
    /// ```
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;
        
        // Populate size cache with results
        for (uri, size) in size_map.iter() {
            self.size_cache.put(uri.clone(), *size).await;
        }
        
        Ok(size_map.len())
    }
}

/// Buffered writer for S3 that collects chunks and uses multipart upload
pub struct S3BufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

impl S3BufferedWriter {
    pub fn new(uri: String) -> Self {
        Self::new_with_compression(uri, CompressionConfig::None)
    }
    
    pub fn new_with_compression(uri: String, compression: CompressionConfig) -> Self {
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                match zstd::Encoder::new(Vec::new(), *level) {
                    Ok(encoder) => Some(encoder),
                    Err(_) => None, // Fall back to no compression on error
                }
            }
        };
        
        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        }
    }
}

#[async_trait]
impl ObjectWriter for S3BufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        
        // Handle compression
        if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and add to buffer
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            self.buffer.append(compressed);
        } else {
            // No compression - direct to buffer
            self.buffer.extend_from_slice(chunk);
        }
        
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        
        // Finalize compression if enabled
        if let Some(compressor) = self.compressor.take() {
            let compressed_data = compressor.finish()?;
            self.buffer = compressed_data;
        }
        
        self.compressed_bytes = self.buffer.len() as u64;
        
        // Append compression extension if needed
        let mut final_uri = self.uri.clone();
        if self.compression.is_enabled() {
            if !final_uri.ends_with(self.compression.extension()) {
                final_uri.push_str(self.compression.extension());
            }
        }
        
        // Use S3 multipart upload for the buffered data
        s3_put_object_multipart_uri_async(&final_uri, &self.buffer, None).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    fn compression(&self) -> CompressionConfig {
        self.compression.clone()
    }
    
    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use current buffer size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.buffer.len() as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        Ok(())
    }
}

// ============================================================================
// Azure adapter
// ============================================================================
fn parse_azure_uri(uri: &str) -> Result<(String, String, String)> {
    // Supports:
    // - az://{account}/{container}/{key...}
    // - https://{account}.blob.core.windows.net/{container}/{key...}
    if let Some(rest) = uri.strip_prefix("az://") {
        let mut it = rest.splitn(3, '/');
        let account = it.next().ok_or_else(|| anyhow!("missing account in az:// URI"))?;
        let container = it.next().ok_or_else(|| anyhow!("missing container in az:// URI"))?;
        let key = it.next().unwrap_or("").to_string();
        return Ok((account.to_string(), container.to_string(), key));
    }

    //if let Some(host_i) = uri.find(".blob.core.windows.net/") {
    if uri.contains(".blob.core.windows.net/") {
        // crude parse: "https://{account}.blob.core.windows.net/{container}/{key...}"
        // find "https://" then account up to first '.'
        let after_scheme = uri.strip_prefix("https://").ok_or_else(|| anyhow!("expected https:// for Azure URL"))?;
        let mut host_and_path = after_scheme.splitn(2, '/');
        let host = host_and_path.next().unwrap_or("");
        let path = host_and_path.next().unwrap_or("");
        let account = host.split('.').next().ok_or_else(|| anyhow!("bad Azure host"))?;
        let mut segs = path.split('/').filter(|s| !s.is_empty());
        let container = segs.next().ok_or_else(|| anyhow!("missing container in URL path"))?;
        let key = segs.collect::<Vec<_>>().join("/");
        return Ok((account.to_string(), container.to_string(), key));
    }

    bail!("not a recognized Azure URI: {}", uri)
}

fn az_uri(account: &str, container: &str, key: &str) -> String {
    if key.is_empty() {
        format!("az://{}/{}", account, container)
    } else {
        format!("az://{}/{}/{}", account, container, key)
    }
}

fn az_props_to_meta(p: &AzureBlobProperties) -> ObjectMetadata {
    ObjectMetadata {
        size: p.content_length,
        last_modified: p.last_modified.clone(),
        e_tag: p.etag.clone(),
        content_type: None,
        content_language: None,
        content_encoding: None,
        cache_control: None,
        content_disposition: None,
        expires: None,
        storage_class: None,
        server_side_encryption: None,
        ssekms_key_id: None,
        sse_customer_algorithm: None,
        version_id: None,
        replication_status: None,
        metadata: Default::default(),
    }
}

/// Configuration for Azure Blob Storage backend with RangeEngine support
/// 
/// Azure benefits significantly from concurrent range downloads due to network latency.
/// However, RangeEngine is **disabled by default** (v0.9.6+) to avoid stat overhead.
/// Enable explicitly for large-file workloads where the benefit outweighs HEAD request cost.
#[derive(Debug, Clone)]
pub struct AzureConfig {
    /// Enable RangeEngine for concurrent range downloads
    /// Default: false (v0.9.6+) - must opt-in to avoid stat overhead on every GET
    pub enable_range_engine: bool,
    
    /// RangeEngine configuration
    /// Network-optimized defaults: 16 MiB threshold, 32 concurrent ranges, 64 MiB chunks
    pub range_engine: RangeEngineConfig,
    
    /// Time-to-live for cached object sizes
    /// Default: 60 seconds
    /// Set to 0 to disable caching
    /// 
    /// v0.9.10: Added for pre-stat optimization
    pub size_cache_ttl_secs: u64,
}

impl Default for AzureConfig {
    fn default() -> Self {
        Self {
            enable_range_engine: false,  // Disabled by default due to stat overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE,  // 64 MiB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,  // 32 parallel
                min_split_size: DEFAULT_RANGE_ENGINE_THRESHOLD,  // 16 MiB threshold
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS),  // 30s
            },
            size_cache_ttl_secs: 60,  // 60 second TTL for size cache
        }
    }
}


#[derive(Clone)]
pub struct AzureObjectStore {
    config: AzureConfig,
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
}


impl AzureObjectStore {
    pub fn new() -> Self {
        let config = AzureConfig::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }
    
    pub fn with_config(config: AzureConfig) -> Self {
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }
    
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }
    
    /// Get object size, checking cache first
    /// 
    /// v0.9.10: Added to optimize get() and get_with_range_engine()
    /// by eliminating redundant stat() calls.
    async fn get_object_size(&self, uri: &str) -> Result<u64> {
        // Check cache first
        if let Some(cached_size) = self.size_cache.get(uri).await {
            return Ok(cached_size);
        }
        
        // Cache miss - perform stat and cache result
        let metadata = self.stat(uri).await?;
        self.size_cache.put(uri.to_string(), metadata.size).await;
        Ok(metadata.size)
    }

    fn client_for_uri(uri: &str) -> Result<(AzureBlob, String, String, String)> {
        let (account, container, key) = parse_azure_uri(uri)?;
        // If caller provided an https:// URL, we still build via account name to ensure
        // we normalize the list() return URIs as az://...
        let cli = AzureBlob::with_default_credential(&account, &container)?;
        Ok((cli, account, container, key))
    }

    fn client_for_prefix(uri_prefix: &str) -> Result<(AzureBlob, String, String, String)> {
        Self::client_for_uri(uri_prefix)
    }
    
    /// Download using RangeEngine for concurrent range requests
    /// 
    /// This method uses the generic RangeEngine to split large Azure blobs into
    /// concurrent range requests, significantly improving throughput by hiding
    /// network latency.
    /// 
    /// # Performance
    /// 
    /// Expected improvements for large blobs (> 4MB):
    /// - Medium blobs (4-64MB): 20-40% faster
    /// - Large blobs (> 64MB): 30-50% faster
    /// - Huge blobs (> 1GB): 40-60% faster
    async fn get_with_range_engine(&self, uri: &str, object_size: u64) -> Result<Bytes> {
        let engine = RangeEngine::new(self.config.range_engine.clone());
        
        // Create closure that captures uri for get_range calls
        let uri_owned = uri.to_string();
        let self_clone = self.clone();
        
        let get_range_fn = move |offset: u64, length: u64| {
            let uri = uri_owned.clone();
            let store = self_clone.clone();
            async move {
                store.get_range(&uri, offset, Some(length)).await
            }
        };
        
        let (bytes, stats) = engine.download(object_size, get_range_fn, None).await?;
        
        info!(
            "RangeEngine (Azure) downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );
        
        Ok(bytes)
    }
}


#[async_trait]
impl ObjectStore for AzureObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        
        // Get blob size from cache or stat (v0.9.10: cache optimization)
        let object_size = self.get_object_size(uri).await?;
        
        // Use RangeEngine for large blobs if enabled
        if self.config.enable_range_engine && object_size >= self.config.range_engine.min_split_size {
            debug!(
                "Azure blob size {} >= threshold {}, using RangeEngine for {}",
                object_size,
                self.config.range_engine.min_split_size,
                uri
            );
            return self.get_with_range_engine(uri, object_size).await;
        }
        
        // Simple sequential download for small blobs
        debug!(
            "Azure blob size {} < threshold {}, using simple download for {}",
            object_size,
            self.config.range_engine.min_split_size,
            uri
        );
        
        let b = cli.get(&key).await?; // Bytes - return directly for zero-copy
        Ok(b)
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let end = length.map(|len| offset + len - 1);
        let b = cli.get_range(&key, offset, end).await?; // Bytes - return directly for zero-copy
        Ok(b)
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        cli.put(&key, Bytes::from(data.to_vec()), true).await
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let part = part_size.unwrap_or(crate::constants::DEFAULT_AZURE_MULTIPART_PART_SIZE);
        let max_in_flight = std::env::var("AZURE_MAX_INFLIGHT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(crate::constants::DEFAULT_CONCURRENT_UPLOADS);

        // Stream the provided buffer as Bytes chunks of size `part`
        let chunks = data
            .chunks(part)
            .map(|c| Bytes::copy_from_slice(c))
            .collect::<Vec<_>>();
        let stream = stream::iter(chunks);

        cli.upload_multipart_stream(&key, stream, part, max_in_flight).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (cli, account, container, key_prefix) = Self::client_for_prefix(uri_prefix)?;
        // Azure's flat list is already recursive (prefix-constrained).
        let prefix = if recursive {
            Some(key_prefix.as_str())
        } else {
            // emulate "shallow" by trimming after next '/':
            // We’ll still ask Azure for full prefix, then post-filter.
            Some(key_prefix.as_str())
        };

        let mut keys = cli.list(prefix).await?;
        if !recursive && !key_prefix.is_empty() {
            let base = if key_prefix.ends_with('/') { key_prefix.clone() } else { format!("{}/", key_prefix) };
            keys.retain(|k| {
                if let Some(rest) = k.strip_prefix(&base) {
                    !rest.contains('/')
                } else {
                    // if it doesn't start with base, keep only if it's exactly the same path
                    k == &key_prefix
                }
            });
        }

        Ok(keys.into_iter().map(|k| az_uri(&account, &container, &k)).collect())
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let (cli, account, container, key_prefix) = match Self::client_for_prefix(uri_prefix) {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let prefix = if key_prefix.is_empty() { None } else { Some(key_prefix.as_str()) };
            let base = if !key_prefix.is_empty() && !key_prefix.ends_with('/') { 
                format!("{}/", key_prefix) 
            } else { 
                key_prefix.clone() 
            };

            // Use streaming list to get blobs page by page
            let mut stream = cli.list_stream(prefix);
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(key) => {
                        // For non-recursive, filter to immediate children only
                        if !recursive && !key_prefix.is_empty() {
                            if let Some(rest) = key.strip_prefix(&base) {
                                if rest.contains('/') {
                                    continue; // Skip nested items
                                }
                            } else if key != key_prefix {
                                continue;
                            }
                        }
                        yield Ok(az_uri(&account, &container, &key));
                    }
                    Err(e) => yield Err(e),
                }
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let p = cli.stat(&key).await?;
        Ok(az_props_to_meta(&p))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        cli.delete_objects(&[key]).await.map_err(Into::into)
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        if uris.is_empty() { return Ok(()); }
        
        // Azure Blob batch delete: already supports batching
        let (cli, _acct, _cont, _) = Self::client_for_uri(&uris[0])?;
        let keys: Vec<String> = uris.iter()
            .filter_map(|uri| {
                Self::client_for_uri(uri).ok().map(|(_, _, _, key)| key)
            })
            .collect();
        
        cli.delete_objects(&keys).await.map_err(Into::into)
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (cli, _acct, _cont, key_prefix) = Self::client_for_prefix(uri_prefix)?;
        let keys = cli.list(Some(&key_prefix)).await?;
        if keys.is_empty() { return Ok(()); }
        cli.delete_objects(&keys).await.map_err(Into::into)
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        // interpret "name" as "{account}/{container}"
        let mut it = name.splitn(2, '/');
        let account = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let container = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let cli = AzureBlob::with_default_credential(account, container)?;
        // best-effort create
        let _ = cli.create_container_if_missing().await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let mut it = name.splitn(2, '/');
        let account = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let container = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let cli = AzureBlob::with_default_credential(account, container)?;
        let _ = cli.delete_container().await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer that collects chunks then uses put_multipart
        Ok(Box::new(AzureBufferedWriter::new(uri.to_string())))
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer with compression support
        Ok(Box::new(AzureBufferedWriter::new_with_compression(uri.to_string(), compression)))
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer that collects chunks then uses put_multipart
        if let Some(compression) = options.compression {
            Ok(Box::new(AzureBufferedWriter::new_with_compression(uri.to_string(), compression)))
        } else {
            Ok(Box::new(AzureBufferedWriter::new(uri.to_string())))
        }
    }
    
    /// Pre-stat objects and populate the size cache
    /// 
    /// v0.9.10: Override default implementation to populate internal size cache.
    /// This enables subsequent get() calls to skip redundant stat operations.
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;
        
        // Populate size cache with results
        for (uri, size) in size_map.iter() {
            self.size_cache.put(uri.clone(), *size).await;
        }
        
        Ok(size_map.len())
    }
}

/// Buffered writer for Azure that collects chunks and uses multipart upload

pub struct AzureBufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}


impl AzureBufferedWriter {
    pub fn new(uri: String) -> Self {
        Self::new_with_compression(uri, CompressionConfig::None)
    }
    
    pub fn new_with_compression(uri: String, compression: CompressionConfig) -> Self {
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                match zstd::Encoder::new(Vec::new(), *level) {
                    Ok(encoder) => Some(encoder),
                    Err(_) => None, // Fall back to no compression on error
                }
            }
        };
        
        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        }
    }
}


#[async_trait]
impl ObjectWriter for AzureBufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        
        // Handle compression
        if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and add to buffer
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            self.buffer.append(compressed);
        } else {
            // No compression - direct to buffer
            self.buffer.extend_from_slice(chunk);
        }
        
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        
        // Finalize compression if enabled
        if let Some(compressor) = self.compressor.take() {
            let compressed_data = compressor.finish()?;
            self.buffer = compressed_data;
        }
        
        self.compressed_bytes = self.buffer.len() as u64;
        
        // Append compression extension if needed
        let (cli, _acct, _cont, key) = AzureObjectStore::client_for_uri(&self.uri)?;
        let final_key = if self.compression.is_enabled() && !key.ends_with(self.compression.extension()) {
            format!("{}{}", key, self.compression.extension())
        } else {
            key
        };
        
        // Use Azure multipart upload for the buffered data
        cli.put(&final_key, Bytes::from(self.buffer.clone()), true).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    fn compression(&self) -> CompressionConfig {
        self.compression.clone()
    }
    
    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use current buffer size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.buffer.len() as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        Ok(())
    }
}

// ============================================================================
// Google Cloud Storage (GCS) adapter
// ============================================================================

/// Helper function to create GCS URI from bucket and key
fn gcs_uri(bucket: &str, key: &str) -> String {
    if key.is_empty() {
        format!("gs://{}", bucket)
    } else {
        format!("gs://{}/{}", bucket, key)
    }
}

/// Convert GCS object metadata to provider-neutral ObjectMetadata
fn gcs_meta_to_object_meta(meta: &GcsObjectMetadata) -> ObjectMetadata {
    ObjectMetadata {
        size: meta.size,
        last_modified: meta.updated.clone(),
        e_tag: meta.etag.clone(),
        content_type: None,
        content_language: None,
        content_encoding: None,
        cache_control: None,
        content_disposition: None,
        expires: None,
        storage_class: None,
        server_side_encryption: None,
        ssekms_key_id: None,
        sse_customer_algorithm: None,
        version_id: None,
        replication_status: None,
        metadata: Default::default(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GCS Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Google Cloud Storage backend
/// 
/// Supports RangeEngine for concurrent range downloads on network storage.
/// However, RangeEngine is **disabled by default** (v0.9.6+) to avoid stat overhead.
/// Enable explicitly for large-file workloads where the benefit outweighs HEAD request cost.
/// 
/// v0.9.10: Added size_cache_ttl_secs for pre-stat optimization
#[derive(Clone, Debug)]
pub struct GcsConfig {
    /// Enable RangeEngine for concurrent range downloads
    /// Default: false (v0.9.6+) - must opt-in to avoid stat overhead on every GET
    pub enable_range_engine: bool,
    
    /// RangeEngine configuration
    /// Network-optimized defaults: 16 MiB threshold, 32 concurrent ranges, 64 MiB chunks
    pub range_engine: RangeEngineConfig,
    
    /// Time-to-live for cached object sizes
    /// Default: 60 seconds
    /// Set to 0 to disable caching
    pub size_cache_ttl_secs: u64,
}

impl Default for GcsConfig {
    fn default() -> Self {
        Self {
            enable_range_engine: false,  // Disabled by default due to stat overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE,  // 64 MiB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,  // 32 parallel
                min_split_size: DEFAULT_RANGE_ENGINE_THRESHOLD,  // 16 MiB threshold
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS),  // 30s
            },
            size_cache_ttl_secs: 60,  // 60 second TTL for size cache
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GCS ObjectStore Implementation
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GcsObjectStore {
    config: GcsConfig,
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
}

impl GcsObjectStore {
    pub fn new() -> Self {
        let config = GcsConfig::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }
    
    pub fn with_config(config: GcsConfig) -> Self {
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }
    
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }
    
    /// Get object size, checking cache first
    /// 
    /// v0.9.10: Added to optimize get() and get_with_range_engine()
    /// by eliminating redundant stat() calls.
    async fn get_object_size(&self, uri: &str) -> Result<u64> {
        // Check cache first
        if let Some(cached_size) = self.size_cache.get(uri).await {
            return Ok(cached_size);
        }
        
        // Cache miss - perform stat and cache result
        let metadata = self.stat(uri).await?;
        self.size_cache.put(uri.to_string(), metadata.size).await;
        Ok(metadata.size)
    }

    async fn get_client() -> Result<GcsClient> {
        GcsClient::new().await
    }
    
    /// Download using RangeEngine for concurrent range requests
    /// 
    /// This method uses the generic RangeEngine to split large GCS objects into
    /// concurrent range requests, significantly improving throughput by hiding
    /// network latency.
    /// 
    /// # Performance
    /// 
    /// Expected improvements for large objects (> 4MB):
    /// - Medium objects (4-64MB): 20-40% faster
    /// - Large objects (> 64MB): 30-50% faster
    /// - Huge objects (> 1GB): 40-60% faster
    async fn get_with_range_engine(&self, uri: &str, object_size: u64) -> Result<Bytes> {
        let engine = RangeEngine::new(self.config.range_engine.clone());
        
        // Create closure that captures uri for get_range calls
        let uri_owned = uri.to_string();
        let self_clone = self.clone();
        
        let get_range_fn = move |offset: u64, length: u64| {
            let uri = uri_owned.clone();
            let store = self_clone.clone();
            async move {
                store.get_range(&uri, offset, Some(length)).await
            }
        };
        
        let (bytes, stats) = engine.download(object_size, get_range_fn, None).await?;
        
        info!(
            "RangeEngine (GCS) downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );
        
        Ok(bytes)
    }
}

#[async_trait]
impl ObjectStore for GcsObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = Self::get_client().await?;
        
        // Check if RangeEngine is enabled and object is large enough
        if !self.config.enable_range_engine {
            trace!("RangeEngine disabled for GCS, using simple download");
            return client.get_object(&bucket, &object).await;
        }
        
        // Get object size from cache or stat (v0.9.10: cache optimization)
        let object_size = self.get_object_size(uri).await?;
        
        // Use RangeEngine for large objects (default 4MB+)
        if object_size >= self.config.range_engine.min_split_size {
            debug!(
                "GCS object size {} >= threshold {}, using RangeEngine for {}",
                object_size,
                self.config.range_engine.min_split_size,
                uri
            );
            return self.get_with_range_engine(uri, object_size).await;
        }
        
        // Simple sequential download for small objects
        debug!(
            "GCS object size {} < threshold {}, using simple download for {}",
            object_size,
            self.config.range_engine.min_split_size,
            uri
        );
        
        client.get_object(&bucket, &object).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = Self::get_client().await?;
        client.get_object_range(&bucket, &object, offset, length).await
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = Self::get_client().await?;
        client.put_object(&bucket, &object, data).await
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let chunk_size = part_size.unwrap_or(crate::constants::DEFAULT_S3_MULTIPART_PART_SIZE);
        let client = Self::get_client().await?;
        client.put_object_multipart(&bucket, &object, data, chunk_size).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        // parse_gcs_uri now handles bucket-only URIs (gs://bucket/ → ("bucket", ""))
        let (bucket, key_prefix) = parse_gcs_uri(uri_prefix)?;

        let client = Self::get_client().await?;
        let prefix = if key_prefix.is_empty() { None } else { Some(key_prefix.as_str()) };
        let keys = client.list_objects(&bucket, prefix, recursive).await?;

        // Convert keys to full URIs
        Ok(keys.into_iter().map(|k| gcs_uri(&bucket, &k)).collect())
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            // parse_gcs_uri now handles bucket-only URIs (gs://bucket/ → ("bucket", ""))
            let (bucket, key_prefix) = match parse_gcs_uri(uri_prefix) {
                Ok((b, k)) => (b, k),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let client = match Self::get_client().await {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            // Use streaming list to get objects page by page
            let prefix = if key_prefix.is_empty() { None } else { Some(key_prefix.as_str()) };
            let mut stream = client.list_objects_stream(&bucket, prefix, recursive);
            
            use futures::stream::StreamExt;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(key) => yield Ok(gcs_uri(&bucket, &key)),
                    Err(e) => yield Err(e),
                }
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = Self::get_client().await?;
        let meta = client.stat_object(&bucket, &object).await?;
        Ok(gcs_meta_to_object_meta(&meta))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = Self::get_client().await?;
        client.delete_object(&bucket, &object).await
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        if uris.is_empty() { return Ok(()); }
        
        // GCS batch delete: extract bucket and objects
        let (bucket, _) = parse_gcs_uri(&uris[0])?;
        let objects: Vec<String> = uris.iter()
            .filter_map(|uri| parse_gcs_uri(uri).ok().map(|(_, obj)| obj))
            .collect();
        
        let client = Self::get_client().await?;
        client.delete_objects(&bucket, objects).await
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (bucket, key_prefix) = parse_gcs_uri(uri_prefix)
            .or_else(|_| {
                // Handle bucket-only URIs
                if let Some(rest) = uri_prefix.strip_prefix("gs://").or_else(|| uri_prefix.strip_prefix("gcs://")) {
                    let bucket = rest.trim_end_matches('/').to_string();
                    if !bucket.is_empty() {
                        return Ok((bucket, String::new()));
                    }
                }
                bail!("Invalid GCS URI for delete_prefix operation: {}", uri_prefix)
            })?;

        let client = Self::get_client().await?;
        // List all objects with the prefix
        let keys = client.list_objects(&bucket, Some(&key_prefix), true).await?;
        
        if keys.is_empty() {
            return Ok(());
        }

        // Delete all objects
        client.delete_objects(&bucket, keys).await
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        // For GCS, the "container" is a bucket
        let client = Self::get_client().await?;
        client.create_bucket(name, None).await
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        // For GCS, the "container" is a bucket
        let client = Self::get_client().await?;
        client.delete_bucket(name).await
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For GCS, use buffered writer
        Ok(Box::new(GcsBufferedWriter::new(uri.to_string())))
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        // For GCS, use buffered writer with compression support
        Ok(Box::new(GcsBufferedWriter::new_with_compression(uri.to_string(), compression)))
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        // For GCS, use buffered writer
        if let Some(compression) = options.compression {
            Ok(Box::new(GcsBufferedWriter::new_with_compression(uri.to_string(), compression)))
        } else {
            Ok(Box::new(GcsBufferedWriter::new(uri.to_string())))
        }
    }
    
    /// Pre-stat objects and populate the size cache
    /// 
    /// v0.9.10: Override default implementation to populate internal size cache.
    /// This enables subsequent get() calls to skip redundant stat operations.
    /// 
    /// Critical for benchmarking workloads that download many GCS objects,
    /// eliminating per-object HEAD request latency (typically 10-50ms each).
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;
        
        // Populate size cache with results
        for (uri, size) in size_map.iter() {
            self.size_cache.put(uri.clone(), *size).await;
        }
        
        Ok(size_map.len())
    }
}

/// Buffered writer for GCS that collects chunks and uses multipart upload
pub struct GcsBufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

impl GcsBufferedWriter {
    pub fn new(uri: String) -> Self {
        Self::new_with_compression(uri, CompressionConfig::None)
    }
    
    pub fn new_with_compression(uri: String, compression: CompressionConfig) -> Self {
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                match zstd::Encoder::new(Vec::new(), *level) {
                    Ok(encoder) => Some(encoder),
                    Err(_) => None, // Fall back to no compression on error
                }
            }
        };
        
        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        }
    }
}

#[async_trait]
impl ObjectWriter for GcsBufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        
        // Handle compression
        if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and add to buffer
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            self.buffer.append(compressed);
        } else {
            // No compression - direct to buffer
            self.buffer.extend_from_slice(chunk);
        }
        
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        
        // Finalize compression if enabled
        if let Some(compressor) = self.compressor.take() {
            let compressed_data = compressor.finish()?;
            self.buffer = compressed_data;
        }
        
        self.compressed_bytes = self.buffer.len() as u64;
        
        // Append compression extension if needed
        let mut final_uri = self.uri.clone();
        if self.compression.is_enabled() {
            if !final_uri.ends_with(self.compression.extension()) {
                final_uri.push_str(self.compression.extension());
            }
        }
        
        // Parse URI and upload via GCS client
        let (bucket, object) = parse_gcs_uri(&final_uri)?;
        let client = GcsClient::new().await?;
        
        // Use multipart for large objects, simple put for small ones
        if self.buffer.len() > crate::constants::DEFAULT_S3_MULTIPART_PART_SIZE {
            client.put_object_multipart(&bucket, &object, &self.buffer, crate::constants::DEFAULT_S3_MULTIPART_PART_SIZE).await
        } else {
            client.put_object(&bucket, &object, &self.buffer).await
        }
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    fn compression(&self) -> CompressionConfig {
        self.compression.clone()
    }
    
    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use current buffer size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.buffer.len() as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        Ok(())
    }
}

// ============================================================================
// Convenience factory that picks a backend from a URI
// ============================================================================
pub fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    store_for_uri_with_logger(uri, None)
}

/// Create a store for the given URI with optional operation logging.
/// 
/// When a logger is provided, all ObjectStore operations will be traced to the op-log.
/// This enables performance analysis and debugging for all backends (file://, s3://, az://, direct://).
///
/// # Arguments
/// * `uri` - Storage URI (e.g., "s3://bucket/", "file:///path/", "az://container/")
/// * `logger` - Optional Logger instance for operation tracing
///
/// # Example
/// ```rust,no_run
/// use s3dlio::{store_for_uri_with_logger, init_op_logger, global_logger};
///
/// # async fn example() -> anyhow::Result<()> {
/// // Initialize op-log
/// init_op_logger("trace.tsv.zst")?;
/// let logger = global_logger();
///
/// // Create store with logging
/// let store = store_for_uri_with_logger("file:///tmp/data", logger)?;
///
/// // Operations are now traced
/// let data = store.get("file:///tmp/data/test.txt").await?;
/// # Ok(())
/// # }
/// ```
pub fn store_for_uri_with_logger(uri: &str, logger: Option<crate::s3_logger::Logger>) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;
    
    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File  => FileSystemObjectStore::boxed(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3    => S3ObjectStore::boxed(),
        Scheme::Azure => AzureObjectStore::boxed(),
        Scheme::Gcs   => GcsObjectStore::boxed(),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };
    
    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::new(Arc::from(store), logger)))
    } else {
        Ok(store)
    }
}

/// Enhanced factory that supports configuration options for file I/O
pub fn store_for_uri_with_config(uri: &str, file_config: Option<FileSystemConfig>) -> Result<Box<dyn ObjectStore>> {
    store_for_uri_with_config_and_logger(uri, file_config, None)
}

/// Enhanced factory with configuration and optional logger support
pub fn store_for_uri_with_config_and_logger(
    uri: &str, 
    file_config: Option<FileSystemConfig>,
    logger: Option<crate::s3_logger::Logger>
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;
    
    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => {
            if let Some(config) = file_config {
                ConfigurableFileSystemObjectStore::boxed(config)
            } else {
                FileSystemObjectStore::boxed()
            }
        }
        Scheme::Direct => {
            if let Some(config) = file_config {
                ConfigurableFileSystemObjectStore::boxed(config)
            } else {
                ConfigurableFileSystemObjectStore::boxed_direct_io()
            }
        }
        Scheme::S3 => S3ObjectStore::boxed(),
        Scheme::Azure => AzureObjectStore::boxed(),
        Scheme::Gcs => GcsObjectStore::boxed(),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };
    
    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::new(Arc::from(store), logger)))
    } else {
        Ok(store)
    }
}

/// Factory for creating file stores with O_DIRECT enabled for AI/ML workloads
pub fn direct_io_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    direct_io_store_for_uri_with_logger(uri, None)
}

/// Factory for creating file stores with O_DIRECT and optional logger
pub fn direct_io_store_for_uri_with_logger(uri: &str, logger: Option<crate::s3_logger::Logger>) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;
    
    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => S3ObjectStore::boxed(),
        Scheme::Azure => AzureObjectStore::boxed(),
        Scheme::Gcs => GcsObjectStore::boxed(),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };
    
    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::new(Arc::from(store), logger)))
    } else {
        Ok(store)
    }
}

/// Factory for creating high-performance stores optimized for AI/ML workloads
pub fn high_performance_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    high_performance_store_for_uri_with_logger(uri, None)
}

/// Factory for creating high-performance stores with optional logger
pub fn high_performance_store_for_uri_with_logger(uri: &str, logger: Option<crate::s3_logger::Logger>) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;
    
    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => ConfigurableFileSystemObjectStore::boxed_high_performance(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => S3ObjectStore::boxed(),
        Scheme::Azure => AzureObjectStore::boxed(),
        Scheme::Gcs => GcsObjectStore::boxed(),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };
    
    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::new(Arc::from(store), logger)))
    } else {
        Ok(store)
    }
}

/// Factory for creating cloud storage backends with RangeEngine enabled
/// 
/// This factory enables high-performance concurrent downloads for cloud storage (S3, Azure, GCS)
/// by enabling RangeEngine with network-optimized settings. Use this when:
/// - Testing within the cloud provider's network (low latency, high bandwidth)
/// - Working with large objects (> 16 MiB) where parallelism improves throughput
/// - Network conditions support concurrent connections (> 10 Gbps, < 3 ms latency)
/// 
/// RangeEngine is disabled by default to avoid stat() overhead on small files.
/// This factory explicitly enables it for high-performance scenarios.
/// 
/// # Arguments
/// * `uri` - Storage URI (e.g., "s3://bucket/", "az://container/", "gs://bucket/")
/// 
/// # Network Scenarios
/// 
/// **High-performance (local/in-cloud)**: > 10 Gbps, < 3 ms latency
/// - Use this factory to enable RangeEngine
/// - Expected improvement: 30-60% for large objects
/// 
/// **Remote/low-bandwidth**: < 1 Gbps, > 30 ms latency
/// - Use standard `store_for_uri()` (RangeEngine disabled by default)
/// - Avoids stat() overhead that may not be worth parallelism cost
/// 
/// # Example
/// ```rust,no_run
/// use s3dlio::object_store::store_for_uri_with_high_performance_cloud;
/// 
/// # async fn example() -> anyhow::Result<()> {
/// // High-performance mode for in-cloud testing
/// let store = store_for_uri_with_high_performance_cloud("s3://my-bucket/")?;
/// let data = store.get("s3://my-bucket/large-file.bin").await?;
/// # Ok(())
/// # }
/// ```
pub fn store_for_uri_with_high_performance_cloud(uri: &str) -> Result<Box<dyn ObjectStore>> {
    store_for_uri_with_high_performance_cloud_and_logger(uri, None)
}

/// Factory for creating cloud storage backends with RangeEngine enabled and optional logger
pub fn store_for_uri_with_high_performance_cloud_and_logger(
    uri: &str,
    logger: Option<crate::s3_logger::Logger>
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;
    
    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => ConfigurableFileSystemObjectStore::boxed_high_performance(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => {
            let mut config = S3Config::default();
            config.enable_range_engine = true;  // Enable for high-performance
            Box::new(S3ObjectStore::with_config(config))
        },
        Scheme::Azure => {
            let mut config = AzureConfig::default();
            config.enable_range_engine = true;  // Enable for high-performance
            Box::new(AzureObjectStore::with_config(config))
        },
        Scheme::Gcs => {
            let mut config = GcsConfig::default();
            config.enable_range_engine = true;  // Enable for high-performance
            Box::new(GcsObjectStore::with_config(config))
        },
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };
    
    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::new(Arc::from(store), logger)))
    } else {
        Ok(store)
    }
}

// ============================================================================
// Generic Upload/Download Functions with Progress Tracking
// ============================================================================

use std::path::Path;
use std::sync::Arc;
use futures::stream::FuturesUnordered;
// StreamExt already imported at top of file
use tokio::sync::Semaphore;
use glob;
use crate::progress::ProgressCallback;

/// Generic upload function that works with any ObjectStore backend
/// and supports progress tracking. Supports glob patterns (*,?) and regex patterns.
pub async fn generic_upload_files<P: AsRef<Path>>(
    dest_prefix: &str,
    patterns: &[P],
    max_in_flight: usize,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    // Expand patterns (globs, regex, and single paths)
    let mut paths = Vec::new();
    for pat in patterns {
        let s = pat.as_ref().to_string_lossy();
        
        // Handle different pattern types
        if s.contains('*') || s.contains('?') {
            // Glob pattern - use glob crate
            for entry in glob::glob(&s).map_err(|e| anyhow!("Glob pattern error: {}", e))? {
                match entry {
                    Ok(pb) => {
                        if pb.is_file() {
                            paths.push(pb);
                        }
                    },
                    Err(e) => warn!("Glob error for pattern {}: {}", s, e),
                }
            }
        } else if s.contains('^') || s.contains('$') || s.contains('[') || s.contains('(') || s.contains('\\') || s.contains('.') {
            // Regex pattern - scan directory and apply regex
            let (dir_part, pattern_part) = match s.rfind('/') {
                Some(index) => {
                    let (dir, pattern) = s.split_at(index + 1);
                    (dir, pattern)
                }
                None => ("./", s.as_ref()),
            };
            
            let pattern_part = if pattern_part.is_empty() { ".*" } else { pattern_part };
            debug!("Trying regex pattern '{}' in directory '{}'", pattern_part, dir_part);
            
            match Regex::new(pattern_part) {
                Ok(re) => {
                    let dir_path = std::path::Path::new(dir_part);
                    if dir_path.is_dir() {
                        for entry in std::fs::read_dir(dir_path)
                            .map_err(|e| anyhow!("Cannot read directory '{}': {}", dir_part, e))? {
                            if let Ok(entry) = entry {
                                let path = entry.path();
                                if path.is_file() {
                                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                                        if re.is_match(filename) {
                                            debug!("Regex matched file: {:?}", path);
                                            paths.push(path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                Err(_) => {
                    // Not a valid regex, treat as literal file path
                    debug!("Invalid regex pattern '{}', treating as literal path", pattern_part);
                    let path = pat.as_ref();
                    if path.is_file() {
                        paths.push(path.to_path_buf());
                    }
                }
            }
        } else {
            // Regular file path
            let path = pat.as_ref();
            if path.is_file() {
                paths.push(path.to_path_buf());
            } else if path.is_dir() {
                // If it's a directory, add all files in it
                for entry in std::fs::read_dir(path)
                    .map_err(|e| anyhow!("Cannot read directory '{:?}': {}", path, e))? {
                    if let Ok(entry) = entry {
                        let entry_path = entry.path();
                        if entry_path.is_file() {
                            paths.push(entry_path);
                        }
                    }
                }
            } else {
                warn!("Path does not exist or is not accessible: {:?}", path);
            }
        }
    }

    if paths.is_empty() {
        bail!("No files matched for upload");
    }

    // Cap the number of concurrent tasks
    let effective_jobs = std::cmp::min(max_in_flight, paths.len());

    info!(
        "Starting upload of {} file(s) to {} (jobs={})",
        paths.len(),
        dest_prefix,
        effective_jobs
    );

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let mut futs = FuturesUnordered::new();

    for path in paths.clone() {
        debug!("queueing upload for {:?}", path);
        let sem = sem.clone();
        let store = store_for_uri_with_logger(dest_prefix, global_logger())?; // Each task gets its own store instance
        let progress = progress_callback.clone();
        let dest_base = dest_prefix.to_string();

        let fname = path.file_name()
            .ok_or_else(|| anyhow!("Bad path {:?}", path))?
            .to_string_lossy();

        // Construct destination URI
        let dest_uri = if dest_base.ends_with('/') {
            format!("{}{}", dest_base, fname)
        } else {
            format!("{}/{}", dest_base, fname)
        };

        // Get file size for progress tracking
        let file_size = std::fs::metadata(&path)?.len();

        futs.push(tokio::spawn(async move {
            debug!("starting upload of {:?} → {}", path, dest_uri);
            let _permit = sem.acquire_owned().await.unwrap();

            // Read file data
            let data = tokio::fs::read(&path).await?;

            // Upload via ObjectStore trait
            store.put(&dest_uri, &data).await?;

            debug!("finished upload of {:?} → {}", path, dest_uri);

            // Update progress if callback provided
            if let Some(ref progress) = progress {
                progress.object_completed(file_size);
            }

            Ok::<(), anyhow::Error>(())
        }));
    }

    // Wait for all uploads to complete
    while let Some(join_res) = futs.next().await {
        join_res??;
    }

    info!("Finished upload of {} file(s) to {}", paths.len(), dest_prefix);
    Ok(())
}

/// Generic download function that works with any ObjectStore backend
/// and supports progress tracking.
pub async fn generic_download_objects(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let store = store_for_uri_with_logger(src_uri, global_logger())?;

    // List objects to download
    let keys = store.list(src_uri, recursive).await?;

    if keys.is_empty() {
        bail!("No objects matched for download");
    }

    // Ensure destination directory exists
    std::fs::create_dir_all(dest_dir)?;

    // Cap the number of concurrent tasks
    let effective_jobs = std::cmp::min(max_in_flight, keys.len());

    info!(
        "Starting download of {} object(s) from {} to {:?} (jobs={})",
        keys.len(),
        src_uri,
        dest_dir,
        effective_jobs
    );

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let mut futs = FuturesUnordered::new();

    for uri in keys.clone() {
        let sem = sem.clone();
        let store = store_for_uri_with_logger(&uri, global_logger())?; // Each task gets its own store instance
        let progress = progress_callback.clone();
        let out_dir = dest_dir.to_path_buf();

        futs.push(tokio::spawn(async move {
            debug!("starting download of {} → {:?}", uri, out_dir);
            let _permit = sem.acquire_owned().await.unwrap();

            // Skip "directories" (URIs that end with slash)
            if uri.ends_with('/') {
                return Ok::<(), anyhow::Error>(());
            }

            // Download the object
            let bytes = store.get(&uri).await?;
            let byte_count = bytes.len() as u64;

            // Extract filename from URI
            let fname = uri.split('/').last()
                .ok_or_else(|| anyhow!("Cannot extract filename from URI: {}", uri))?;
            let out_path = out_dir.join(fname);

            // Write to disk
            tokio::fs::write(&out_path, bytes).await?;

            debug!("finished download of {} → {:?}", uri, out_path);

            // Update progress if callback provided
            if let Some(ref progress) = progress {
                progress.object_completed(byte_count);
            }

            Ok::<(), anyhow::Error>(())
        }));
    }

    // Wait for all downloads to complete
    while let Some(join_res) = futs.next().await {
        join_res??;
    }

    info!("Finished download of {} object(s) to {:?}", keys.len(), dest_dir);
    Ok(())
}

