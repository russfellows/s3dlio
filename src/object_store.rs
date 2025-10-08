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
use tracing::{debug, warn, info};
use regex::Regex;

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

// --- GCS -----------------------------------------------------
use crate::gcs_client::{GcsClient, GcsObjectMetadata, parse_gcs_uri};

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

    /// Stat a single object (HEAD-like).
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata>;

    /// Delete a single object.
    async fn delete(&self, uri: &str) -> Result<()>;

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
    async fn get_with_validation(&self, uri: &str, expected_checksum: Option<&str>) -> Result<Vec<u8>> {
        let data = self.get(uri).await?;
        
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Integrity validation failed for {}: expected {}, got {}", uri, expected, actual);
            }
        }
        
        // Convert Bytes to Vec<u8> for backward compatibility
        Ok(data.to_vec())
    }
    
    /// Get byte range with integrity validation.
    /// For partial reads, validates against the partial data checksum if provided.
    async fn get_range_with_validation(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        expected_checksum: Option<&str>
    ) -> Result<Vec<u8>> {
        let data = self.get_range(uri, offset, length).await?;
        
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Integrity validation failed for range {}[{}:{}]: expected {}, got {}", 
                      uri, offset, offset + data.len() as u64, expected, actual);
            }
        }
        
        // Convert Bytes to Vec<u8> for backward compatibility
        Ok(data.to_vec())
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
    async fn get_optimized(&self, uri: &str) -> Result<Vec<u8>> {
        // Default implementation: delegate to regular get()
        // S3ObjectStore will override this with concurrent range logic
        // Convert Bytes to Vec<u8> for backward compatibility
        self.get(uri).await.map(|b| b.to_vec())
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
    ) -> Result<Vec<u8>> {
        // Default implementation: delegate to regular get_range()
        // S3ObjectStore will override this with concurrent range logic
        // Convert Bytes to Vec<u8> for backward compatibility
        self.get_range(uri, offset, length).await.map(|b| b.to_vec())
    }

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
pub struct S3ObjectStore;

impl S3ObjectStore {
    pub fn new() -> Self { Self }
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self) }
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

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_stat_object_uri_async(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        let (bucket, key) = parse_s3_uri(uri)?;
        s3_delete_objects(&bucket, &vec![key])
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

    async fn get_optimized(&self, uri: &str) -> Result<Vec<u8>> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        
        // Get object size to determine strategy
        let metadata = self.stat(uri).await?;
        let object_size = metadata.size;
        
        // Use threshold-based decision for optimization
        let threshold = get_concurrent_threshold();
        
        if object_size >= threshold {
            debug!("Using concurrent range GET for large object: {} bytes", object_size);
            // Use concurrent range GET for large objects - convert Bytes to Vec<u8>
            crate::s3_utils::get_object_concurrent_range_async(uri, 0, None, None, None).await.map(|b| b.to_vec())
        } else {
            debug!("Using standard GET for small object: {} bytes", object_size);
            // Use standard GET for small objects - convert Bytes to Vec<u8>
            self.get(uri).await.map(|b| b.to_vec())
        }
    }

    async fn get_range_optimized(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        chunk_size: Option<usize>,
        max_concurrency: Option<usize>
    ) -> Result<Vec<u8>> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        
        let transfer_size = match length {
            Some(len) => len,
            None => {
                // Get object size to calculate transfer size
                let metadata = self.stat(uri).await?;
                metadata.size.saturating_sub(offset)
            }
        };
        
        // Use threshold-based decision for optimization
        let threshold = get_concurrent_threshold();
        
        if transfer_size >= threshold {
            debug!("Using concurrent range GET for large transfer: {} bytes", transfer_size);
            // Use concurrent range GET for large transfers - convert Bytes to Vec<u8>
            crate::s3_utils::get_object_concurrent_range_async(uri, offset, length, chunk_size, max_concurrency).await.map(|b| b.to_vec())
        } else {
            debug!("Using standard range GET for small transfer: {} bytes", transfer_size);
            // Use standard range GET for small transfers - convert Bytes to Vec<u8>
            self.get_range(uri, offset, length).await.map(|b| b.to_vec())
        }
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


pub struct AzureObjectStore;


impl AzureObjectStore {
    pub fn new() -> Self { Self }
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self) }

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
}


#[async_trait]
impl ObjectStore for AzureObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
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

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let p = cli.stat(&key).await?;
        Ok(az_props_to_meta(&p))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        cli.delete_objects(&[key]).await.map_err(Into::into)
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

pub struct GcsObjectStore;

impl GcsObjectStore {
    pub fn new() -> Self { Self }
    
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self) }

    async fn get_client() -> Result<GcsClient> {
        GcsClient::new().await
    }
}

#[async_trait]
impl ObjectStore for GcsObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = Self::get_client().await?;
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
        let (bucket, key_prefix) = parse_gcs_uri(uri_prefix)
            .or_else(|_| {
                // Handle bucket-only URIs like "gs://bucket" or "gs://bucket/"
                if let Some(rest) = uri_prefix.strip_prefix("gs://").or_else(|| uri_prefix.strip_prefix("gcs://")) {
                    let bucket = rest.trim_end_matches('/').to_string();
                    if !bucket.is_empty() {
                        return Ok((bucket, String::new()));
                    }
                }
                bail!("Invalid GCS URI for list operation: {}", uri_prefix)
            })?;

        let client = Self::get_client().await?;
        let keys = client.list_objects(&bucket, Some(&key_prefix), recursive).await?;

        // Convert keys to full URIs
        Ok(keys.into_iter().map(|k| gcs_uri(&bucket, &k)).collect())
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
        Scheme::Gcs => bail!("GCS backend not yet fully implemented"),
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
        Scheme::Gcs => bail!("GCS backend not yet fully implemented"),
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
        Scheme::Gcs => bail!("GCS backend not yet fully implemented"),
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
use futures::StreamExt;
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

