use tracing::{debug, info, trace};
// src/file_store.rs
//
// FileSystemObjectStore implementation for POSIX file I/O
// This provides the same ObjectStore interface for local filesystem operations

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use crate::object_store::WriterOptions;
use crate::page_cache::{apply_page_cache_hint};
use crate::data_loader::options::PageCacheMode;
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use crc32fast::Hasher;

use crate::object_store::{ObjectStore, ObjectMetadata, ObjectWriter, CompressionConfig, ObjectProperties};
use crate::constants::DEFAULT_FILE_RANGE_ENGINE_THRESHOLD;

/// Configuration for FileSystemObjectStore
/// 
/// **Performance Note**: Range parallelism may be **counterproductive** for local files
/// - Sequential reads are typically faster due to OS page cache optimization
/// - Concurrent ranges introduce seek overhead and disk contention
/// - RangeEngine is **disabled by default** (v0.9.6+) due to performance testing
/// - Benefits are primarily for network storage (S3/Azure/GCS)
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    /// Enable concurrent range downloads for large files
    /// Default: false (v0.9.6+) - local files rarely benefit from range parallelism
    pub enable_range_engine: bool,
    
    /// Range engine configuration
    pub range_engine: RangeEngineConfig,
    
    /// Page cache behavior hint (maps to posix_fadvise on Linux/Unix)
    /// - None: Use Auto mode (Sequential for large files >=64MB, Random for small)
    /// - Some(mode): Explicitly set Sequential, Random, DontNeed, or Normal
    /// Default: None (Auto mode)
    pub page_cache_mode: Option<PageCacheMode>,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            enable_range_engine: false,  // Disabled by default due to local FS seek overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                min_split_size: DEFAULT_FILE_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            page_cache_mode: None,  // Use Auto mode by default
        }
    }
}

/// FileSystem adapter that implements ObjectStore for local POSIX file operations.
/// 
/// URI Mapping:
/// - `file:///absolute/path/to/file` -> `/absolute/path/to/file`
/// - `file://./relative/path/to/file` -> `./relative/path/to/file`
/// - `file://../relative/path/to/file` -> `../relative/path/to/file`
///
/// Container Operations:
/// - create_container: creates directories
/// - delete_container: removes empty directories
/// 
/// Performance:
/// - Files >= 4MB use concurrent range downloads (configurable)
/// - Files < 4MB use simple sequential reads
/// - Expected 30-50% throughput improvement for large files
#[derive(Clone)]
pub struct FileSystemObjectStore {
    config: Arc<FileSystemConfig>,
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
}

/// Streaming writer for filesystem operations
pub struct FileSystemWriter {
    file: Option<fs::File>,
    path: PathBuf,
    bytes_written: u64,
    compressed_bytes: u64,
    finalized: bool,
    hasher: Hasher,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
}

impl FileSystemWriter {
    async fn new(path: PathBuf) -> Result<Self> {
        Self::new_with_compression(path, CompressionConfig::None).await
    }
    
    pub async fn new_with_compression(path: PathBuf, compression: CompressionConfig) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Determine final path with compression extension if needed
        let final_path = if compression.is_enabled() {
            let extension = path.extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let new_extension = if extension.is_empty() {
                compression.extension().trim_start_matches('.')
            } else {
                &format!("{}{}", extension, compression.extension())
            };
            path.with_extension(new_extension)
        } else {
            path.clone()
        };
        
        // Create and open the file for writing
        let file = fs::File::create(&final_path).await?;
        
        let compressor = match compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                let mut encoder = zstd::Encoder::new(Vec::new(), level)?;
                encoder.include_checksum(false)?; // We handle checksums ourselves
                Some(encoder)
            }
        };
        
        Ok(Self {
            file: Some(file),
            path: final_path,
            bytes_written: 0,
            compressed_bytes: 0,
            finalized: false,
            hasher: Hasher::new(),
            compression,
            compressor,
        })
    }
}

#[async_trait]
impl ObjectWriter for FileSystemWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        debug!("FileSystemWriter::write_chunk: writing {} bytes", chunk.len());
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        
        // Update checksum with original data (before compression)
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        
        if let Some(ref mut file) = self.file {
            if let Some(ref mut compressor) = self.compressor {
                // Compress the chunk using std::io::Write trait
                use std::io::Write;
                compressor.write_all(chunk)?;
                
                // Don't update compressed_bytes here - we don't know the compressed size until finalize()
            } else {
                // No compression - write directly to file
                file.write_all(chunk).await?;
            }
            Ok(())
        } else {
            bail!("Writer has been finalized or cancelled");
        }
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        debug!("FileSystemWriter::finalize: starting finalization");
        if self.finalized {
            return Ok(());
        }
        
        if let Some(mut file) = self.file.take() {
            // Finalize compression if enabled
            if let Some(compressor) = self.compressor.take() {
                debug!("FileSystemWriter::finalize: finalizing compression, path: {}", self.path.display());
                let final_compressed = compressor.finish()?;
                self.compressed_bytes = final_compressed.len() as u64; // Update with actual compressed size
                file.write_all(&final_compressed).await?;
            }
            
            file.flush().await?;
            file.sync_all().await?;
        }
        
        self.finalized = true;
        Ok(())
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    fn compressed_bytes(&self) -> u64 {
        self.compressed_bytes
    }
    
    fn compression(&self) -> CompressionConfig {
        self.compression
    }
    
    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else if self.compressed_bytes == 0 {
            // Not yet finalized - can't determine ratio
            1.0
        } else {
            self.compressed_bytes as f64 / self.bytes_written as f64
        }
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        if let Some(_file) = self.file.take() {
            // File will be closed when dropped
        }
        
        // Remove the partial file
        if self.path.exists() {
            let _ = fs::remove_file(&self.path).await; // Ignore errors
        }
        
        Ok(())
    }
}

impl FileSystemObjectStore {
    /// Create a new FileSystemObjectStore with default configuration
    /// 
    /// v0.9.10: Uses TTL=0 for size cache (effectively disabled) because:
    /// - File metadata can change rapidly on disk
    /// - Local stat operations are fast (<1ms vs 10-50ms for network storage)
    /// - Cache provides no meaningful performance benefit
    pub fn new() -> Self {
        use std::time::Duration;
        Self {
            config: Arc::new(FileSystemConfig::default()),
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(Duration::from_secs(0))),
        }
    }
    
    /// Create with custom configuration
    /// 
    /// v0.9.10: TTL=0 for file:// backend (no caching needed for local filesystem)
    pub fn with_config(config: FileSystemConfig) -> Self {
        use std::time::Duration;
        Self {
            config: Arc::new(config),
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(Duration::from_secs(0))),
        }
    }
    
    /// Convert a URI to a filesystem path
    fn uri_to_path(uri: &str) -> Result<PathBuf> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expects file:// URI, got: {}", uri);
        }
        
        // Strip file:// prefix
        let path = &uri[7..];
        Ok(PathBuf::from(path))
    }
    
    /// Convert a filesystem path back to a URI for list operations
    fn path_to_uri(path: &Path) -> String {
        if path.is_absolute() {
            format!("file://{}", path.display())
        } else {
            path.display().to_string()
        }
    }
    
    /// Create ObjectMetadata from filesystem metadata
    async fn metadata_from_path(path: &Path) -> Result<ObjectMetadata> {
        let metadata = fs::metadata(path).await?;
        
        // Create ObjectMetadata compatible with S3ObjectStat
        let last_modified = metadata.modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
            .map(|duration| {
                // Convert to RFC 3339 format like S3 uses
                let secs = duration.as_secs();
                let datetime = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs);
                format!("{:?}", datetime) // Simple formatting for now
            });
            
        let file_hash = format!("file-{}-{}", 
            metadata.len(),
            metadata.modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        
        Ok(ObjectMetadata {
            size: metadata.len(),
            last_modified,
            e_tag: Some(file_hash),
            content_type: None, // Could infer from file extension
            content_language: None,
            content_encoding: None,
            cache_control: None,
            content_disposition: None,
            expires: None,
            storage_class: Some("STANDARD".to_string()), // Default for files
            server_side_encryption: None,
            ssekms_key_id: None,
            sse_customer_algorithm: None,
            version_id: None,
            replication_status: None,
            metadata: HashMap::new(),
        })
    }
    
    /// Recursively collect files in a directory
    async fn collect_files_recursive(dir: &Path, prefix: &str, results: &mut Vec<String>) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let file_name = entry_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
                
            if entry_path.is_dir() {
                let new_prefix = if prefix.is_empty() {
                    file_name.to_string()
                } else {
                    format!("{}/{}", prefix, file_name)
                };
                Box::pin(Self::collect_files_recursive(&entry_path, &new_prefix, results)).await?;
            } else {
                let file_uri = if prefix.is_empty() {
                    Self::path_to_uri(&entry_path)
                } else {
                    Self::path_to_uri(&entry_path)
                };
                results.push(file_uri);
            }
        }
        
        Ok(())
    }
    
    /// Download using RangeEngine for concurrent range requests
    async fn get_with_range_engine(&self, uri: &str, file_size: u64) -> Result<Bytes> {
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
        
        let (bytes, stats) = engine.download(file_size, get_range_fn, None).await?;
        
        info!(
            "RangeEngine downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );
        
        Ok(bytes)
    }
    
    /// Simple sequential read for small files
    async fn get_simple(&self, uri: &str, file_size: u64) -> Result<Bytes> {
        let path = Self::uri_to_path(uri)?;
        
        // Open file and apply page cache hint based on size
        let file = fs::File::open(&path).await?;
        let std_file = file.try_into_std().map_err(|_| anyhow::anyhow!("Failed to convert to std file"))?;
        
        // Apply page cache hint (use configured mode or Auto by default)
        let cache_mode = self.config.page_cache_mode.unwrap_or(PageCacheMode::Auto);
        let _ = apply_page_cache_hint(&std_file, cache_mode, file_size);
        
        // Convert back to tokio file and read
        let mut file = fs::File::from_std(std_file);
        
        // Pre-allocate to avoid reallocation churn (v0.9.9+)
        let mut data = Vec::with_capacity(file_size as usize);
        file.read_to_end(&mut data).await?;
        
        // Convert to Bytes (cheap, just wraps in Arc)
        Ok(Bytes::from(data))
    }
}

#[async_trait]
impl ObjectStore for FileSystemObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        // Get file size to decide download strategy
        let metadata = fs::metadata(&path).await?;
        let file_size = metadata.len();
        
        // Use RangeEngine for large files if enabled
        if self.config.enable_range_engine && file_size >= self.config.range_engine.min_split_size {
            trace!(
                "File size {} >= threshold {}, using RangeEngine for {}",
                file_size,
                self.config.range_engine.min_split_size,
                uri
            );
            return self.get_with_range_engine(uri, file_size).await;
        }
        
        // Simple sequential read for small files
        trace!(
            "File size {} < threshold {}, using simple read for {}",
            file_size,
            self.config.range_engine.min_split_size,
            uri
        );
        self.get_simple(uri, file_size).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        // Get file size and apply page cache hint
        let metadata = fs::metadata(&path).await?;
        let file_size = metadata.len();
        
        let file = fs::File::open(&path).await?;
        let std_file = file.try_into_std().map_err(|_| anyhow::anyhow!("Failed to convert to std file"))?;
        
        // Apply page cache hint (use configured mode, or Random for range reads by default)
        let cache_mode = self.config.page_cache_mode.unwrap_or(PageCacheMode::Random);
        let _ = apply_page_cache_hint(&std_file, cache_mode, file_size);
        
        let mut file = fs::File::from_std(std_file);
        file.seek(std::io::SeekFrom::Start(offset)).await?;
        
        let read_length = length.unwrap_or(u64::MAX);
        let mut buffer = Vec::new();
        
        if read_length == u64::MAX {
            // Read to end of file
            file.read_to_end(&mut buffer).await?;
        } else {
            // Read specific length - use read_exact to ensure we get all bytes
            buffer.resize(read_length as usize, 0);
            file.read_exact(&mut buffer).await?;
        }
        
        // Convert to Bytes (cheap, just wraps in Arc)
        Ok(Bytes::from(buffer))
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        fs::write(&path, data).await?;
        Ok(())
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], _part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        // For filesystem, multipart is the same as regular put
        // In a more sophisticated implementation, we could write in chunks
        self.put(uri, data).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        if !uri_prefix.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let base_path = Self::uri_to_path(uri_prefix)?;
        let mut results = Vec::new();
        
        if !base_path.exists() {
            return Ok(results); // Empty list for non-existent paths
        }
        
        if base_path.is_file() {
            // If the prefix points to a file, return just that file
            results.push(Self::path_to_uri(&base_path));
            return Ok(results);
        }
        
        if base_path.is_dir() {
            if recursive {
                Self::collect_files_recursive(&base_path, "", &mut results).await?;
            } else {
                // Non-recursive: only direct children
                let mut entries = fs::read_dir(&base_path).await?;
                while let Some(entry) = entries.next_entry().await? {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        results.push(Self::path_to_uri(&entry_path));
                    }
                }
            }
        }
        
        Ok(results)
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            // For file://, delegate to buffered list() since filesystem operations are fast
            // Streaming doesn't provide significant benefit for local filesystem
            match self.list(uri_prefix, recursive).await {
                Ok(keys) => {
                    for key in keys {
                        yield Ok(key);
                    }
                }
                Err(e) => yield Err(e),
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        Self::metadata_from_path(&path).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            // Already deleted, consider it success
            return Ok(());
        }
        
        if path.is_file() {
            fs::remove_file(&path).await?;
        } else if path.is_dir() {
            fs::remove_dir_all(&path).await?;
        }
        
        Ok(())
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        // FileSystemObjectStore: delete files concurrently
        use futures::stream::{self, StreamExt};
        
        let max_concurrency = (uris.len() / 10).max(10).min(100);
        let uris_owned: Vec<String> = uris.to_vec();
        
        let deletions = stream::iter(uris_owned.into_iter())
            .map(|uri| async move { self.delete(&uri).await })
            .buffer_unordered(max_concurrency);
        
        let results: Vec<Result<()>> = deletions.collect().await;
        for result in results {
            result?;
        }
        
        Ok(())
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        if !uri_prefix.starts_with("file://") { bail!("FileSystemObjectStore expected file:// URI"); }
        let base_path = Self::uri_to_path(uri_prefix)?;
        
        if !base_path.exists() {
            return Ok(()); // Nothing to delete
        }
        
        if base_path.is_file() {
            fs::remove_file(&base_path).await?;
        } else if base_path.is_dir() {
            // Collect all files under the prefix and delete them
            let files = self.list(uri_prefix, true).await?;
            for file_uri in files {
                self.delete(&file_uri).await?;
            }
            
            // Try to remove the directory if it's empty
            if let Err(_) = fs::remove_dir(&base_path).await {
                // Directory might not be empty due to subdirectories
                // For now, we'll leave non-empty directories
            }
        }
        
        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);
        fs::create_dir_all(&path).await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);
        
        if !path.exists() {
            return Ok(()); // Already deleted
        }
        
        if path.is_dir() {
            fs::remove_dir(&path).await?; // Only removes empty directories
        } else {
            bail!("Path is not a directory: {}", path.display());
        }
        
        Ok(())
    }

    async fn rename(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        if !src_uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI for source"); 
        }
        if !dst_uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI for destination"); 
        }
        
        let src_path = Self::uri_to_path(src_uri)?;
        let dst_path = Self::uri_to_path(dst_uri)?;
        
        if !src_path.exists() {
            bail!("Source file not found: {}", src_path.display());
        }
        
        // Create parent directories for destination if they don't exist
        if let Some(parent) = dst_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Use tokio::fs::rename for atomic filesystem rename operation
        fs::rename(&src_path, &dst_path).await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI"); 
        }
        
        let path = Self::uri_to_path(uri)?;
        let writer = FileSystemWriter::new(path).await?;
        Ok(Box::new(writer))
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") { 
            bail!("FileSystemObjectStore expected file:// URI"); 
        }
        
        let path = Self::uri_to_path(uri)?;
        let writer = FileSystemWriter::new_with_compression(path, compression).await?;
        Ok(Box::new(writer))
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        
        let path = Self::uri_to_path(uri)?;
        if let Some(compression) = options.compression {
            let writer = FileSystemWriter::new_with_compression(path, compression).await?;
            Ok(Box::new(writer))
        } else {
            let writer = FileSystemWriter::new(path).await?;
            Ok(Box::new(writer))
        }
    }
    
    /// Pre-stat objects and populate the size cache
    /// 
    /// v0.9.10: For file:// backend, size_cache has TTL=0 (effectively disabled)
    /// because local filesystem metadata operations are fast (~1ms vs 10-50ms for
    /// network storage) and files can change rapidly on disk.
    /// 
    /// This method still works to maintain API compatibility, but provides minimal
    /// performance benefit for local files.
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;
        
        // Populate size cache with results (TTL=0 means immediate expiration)
        for (uri, size) in size_map.iter() {
            self.size_cache.put(uri.clone(), *size).await;
        }
        
        Ok(size_map.len())
    }
    // =========================================================================
    // Metadata Operations (v0.10.0+) - ported from sai3-bench fs_metadata.rs
    // =========================================================================
    
    async fn mkdir(&self, uri: &str) -> Result<()> {
        let path = Self::uri_to_path(uri)?;
        
        tokio::fs::create_dir_all(&path)
            .await
            .with_context(|| format!("Failed to create directory: {}", path.display()))?;
        
        Ok(())
    }

    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let path = Self::uri_to_path(uri)?;
        
        let result = if recursive {
            tokio::fs::remove_dir_all(&path).await
        } else {
            tokio::fs::remove_dir(&path).await
        };
        
        match result {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Directory doesn't exist - treat as success (idempotent operation)
                tracing::debug!("Directory already removed or never existed: {}", path.display());
                Ok(())
            }
            Err(e) => {
                if recursive {
                    Err(e).with_context(|| format!("Failed to remove directory recursively: {}", path.display()))
                } else {
                    Err(e).with_context(|| format!("Failed to remove directory (not empty?): {}", path.display()))
                }
            }
        }
    }

    async fn update_metadata(&self, _uri: &str, _metadata: &HashMap<String, String>) -> Result<()> {
        // File systems don't support custom metadata keys like cloud storage
        bail!("Custom metadata not supported for file:// backend")
    }

    async fn update_properties(&self, uri: &str, properties: &ObjectProperties) -> Result<()> {
        let path = Self::uri_to_path(uri)?;
        
        // File systems have limited property support
        // We can only really handle storage_class as a hint (ignored for now)
        if properties.content_type.is_some() 
            || properties.cache_control.is_some() 
            || properties.content_encoding.is_some() {
            bail!("HTTP properties (content-type, cache-control, etc.) not supported for file:// backend")
        }
        
        // storage_class could map to filesystem features (btrfs compression, ZFS tiers)
        // but for now we just accept it as a no-op
        if let Some(ref storage_class) = properties.storage_class {
            tracing::debug!("Ignoring storage_class '{}' for file:// backend at {}", storage_class, path.display());
        }
        
        Ok(())
    }


}
