// src/file_store_direct.rs
//
// Enhanced FileSystemObjectStore with O_DIRECT support for AI/ML workloads
// This module provides direct I/O capabilities that bypass the page cache

use anyhow::{bail, Result};
use bytes::{Bytes, BytesMut};
use crate::object_store::WriterOptions;
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use crate::memory::{BufferPool, BufferPoolConfig, AlignedBuf};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use crc32fast::Hasher;
use std::io::Write;
use tracing::{debug, info, trace, warn};

use crate::constants::{DEFAULT_PAGE_SIZE, DEFAULT_MIN_IO_SIZE, DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD};
use crate::object_store::{ObjectStore, ObjectMetadata, ObjectWriter, CompressionConfig};

/// Configuration options for FileSystem operations
/// 
/// **DirectIO Performance Characteristics**:
/// - O_DIRECT bypasses page cache - already very fast for sequential reads
/// - Range parallelism provides **limited benefit** compared to network storage
/// - Alignment overhead (page-size rounding) adds complexity
/// - Expected improvement: 20-40% vs 30-50% for network storage
/// - **Recommendation**: Higher thresholds (16-32MB) and lower concurrency (8-16)
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    /// Enable O_DIRECT I/O to bypass page cache (Linux/Unix only)
    pub direct_io: bool,
    /// Buffer alignment for O_DIRECT (typically system page size)
    pub alignment: usize,
    /// Minimum I/O size for O_DIRECT operations
    pub min_io_size: usize,
    /// Enable O_SYNC for synchronous writes
    pub sync_writes: bool,
    /// Enable concurrent range downloads for large files
    /// Default: false (v0.9.6+) - DirectIO rarely benefits from range parallelism
    pub enable_range_engine: bool,
    /// Range engine configuration
    pub range_engine: RangeEngineConfig,
    /// Buffer pool for reusable aligned buffers (v0.9.9+)
    /// Eliminates allocation churn and improves performance by 15-20%
    pub buffer_pool: Option<Arc<BufferPool>>,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        // Get system page size for proper alignment
        let page_size = get_system_page_size();
        
        Self {
            direct_io: false,
            alignment: page_size,
            min_io_size: page_size,
            sync_writes: false,
            enable_range_engine: false,  // Disabled by default due to O_DIRECT seek overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                min_split_size: DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            buffer_pool: None,  // No pool by default (v0.9.9+)
        }
    }
}

impl FileSystemConfig {
    /// Create a new configuration with O_DIRECT enabled for AI/ML workloads
    pub fn direct_io() -> Self {
        let page_size = get_system_page_size();
        
        // Create buffer pool for DirectIO operations (v0.9.9+)
        // Pool size matches range concurrency to avoid blocking
        let buffer_pool = Some(BufferPoolConfig {
            capacity: 32,  // 32 buffers in pool
            buffer_size: 64 * 1024 * 1024,  // 64MB per buffer (matches chunk_size)
            alignment: page_size,
        }.build());
        
        Self {
            direct_io: true,
            alignment: page_size,
            min_io_size: page_size,
            sync_writes: false,
            enable_range_engine: false,  // Disabled by default due to O_DIRECT seek overhead (v0.9.6+)
            // DirectIO: higher threshold (16MB), lower concurrency (16)
            range_engine: RangeEngineConfig {
                chunk_size: 64 * 1024 * 1024,     // 64MB chunks
                max_concurrent_ranges: 16,         // Lower concurrency for O_DIRECT
                min_split_size: DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            buffer_pool,
        }
    }

    /// Create a configuration optimized for high-performance AI/ML workloads
    pub fn high_performance() -> Self {
        let page_size = get_system_page_size();
        
        // Create buffer pool for high-performance DirectIO (v0.9.9+)
        let buffer_pool = Some(BufferPoolConfig {
            capacity: 32,
            buffer_size: 64 * 1024 * 1024,
            alignment: page_size,
        }.build());
        
        Self {
            direct_io: true,
            alignment: page_size,
            min_io_size: DEFAULT_MIN_IO_SIZE,
            sync_writes: true,          // Ensure data hits storage
            enable_range_engine: false,  // Disabled by default due to O_DIRECT seek overhead (v0.9.6+)
            // High performance: same as direct_io
            range_engine: RangeEngineConfig {
                chunk_size: 64 * 1024 * 1024,
                max_concurrent_ranges: 16,
                min_split_size: DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            buffer_pool,
        }
    }
}

/// Get the system page size for proper O_DIRECT alignment
fn get_system_page_size() -> usize {
    #[cfg(unix)]
    {
        unsafe {
            let page_size = libc::sysconf(libc::_SC_PAGESIZE);
            if page_size > 0 {
                let detected_size = page_size as usize;
                
                // Validate the detected page size is within reasonable bounds
                // Most systems use 4KB pages, some use 8KB, 16KB, or 64KB
                if detected_size >= 512 && detected_size <= 65536 && detected_size.is_power_of_two() {
                    detected_size
                } else {
                    // Invalid page size detected, use safe default
                    DEFAULT_PAGE_SIZE
                }
            } else {
                DEFAULT_PAGE_SIZE // Fallback if sysconf fails
            }
        }
    }
    
    #[cfg(not(unix))]
    {
        DEFAULT_PAGE_SIZE // Default fallback for non-Unix systems
    }
}

/// Create a page-aligned buffer for O_DIRECT operations
fn create_aligned_buffer(size: usize, alignment: usize) -> Vec<u8> {
    #[cfg(unix)]
    {
        // Use posix_memalign to create properly aligned buffer
        use std::alloc::{alloc, Layout};
        
        if size == 0 {
            return Vec::new();
        }
        
        // Ensure size is aligned to alignment boundary
        let aligned_size = ((size + alignment - 1) / alignment) * alignment;
        
        unsafe {
            let layout = Layout::from_size_align(aligned_size, alignment)
                .expect("Invalid layout for aligned buffer");
            let ptr = alloc(layout);
            
            if ptr.is_null() {
                panic!("Failed to allocate aligned buffer");
            }
            
            // Initialize with zeros
            std::ptr::write_bytes(ptr, 0, aligned_size);
            
            // Create Vec from raw parts
            Vec::from_raw_parts(ptr, 0, aligned_size)
        }
    }
    
    #[cfg(not(unix))]
    {
        // Fallback for non-Unix systems - regular Vec
        vec![0u8; size]
    }
}

/// Resize an aligned buffer, preserving alignment
fn resize_aligned_buffer(buffer: &mut Vec<u8>, new_size: usize, alignment: usize) {
    if new_size <= buffer.capacity() {
        // Simple case - just adjust length
        unsafe {
            buffer.set_len(new_size);
        }
        return;
    }
    
    // Need to reallocate with larger capacity
    let mut new_buffer = create_aligned_buffer(new_size, alignment);
    let copy_size = std::cmp::min(buffer.len(), new_size);
    new_buffer[..copy_size].copy_from_slice(&buffer[..copy_size]);
    unsafe {
        new_buffer.set_len(buffer.len());
    }
    *buffer = new_buffer;
}

/// Streaming writer for direct I/O filesystem operations
pub struct DirectIOWriter {
    file: Option<fs::File>,
    path: PathBuf,
    config: FileSystemConfig,
    bytes_written: u64,
    finalized: bool,
    buffer: Vec<u8>, // Buffer for alignment and minimum I/O size requirements
    hasher: Hasher,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

impl DirectIOWriter {
    pub async fn new(path: PathBuf, config: FileSystemConfig) -> Result<Self> {
        Self::new_with_compression(path, config, CompressionConfig::None).await
    }
    
    pub async fn new_with_compression(path: PathBuf, config: FileSystemConfig, compression: CompressionConfig) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Append compression extension to path if needed
        let final_path = if compression.is_enabled() {
            let mut path_with_ext = path.clone();
            let ext = match &compression {
                CompressionConfig::None => "",
                CompressionConfig::Zstd { .. } => ".zst",
            };
            if let Some(existing_ext) = path_with_ext.extension() {
                let new_ext = format!("{}{}", existing_ext.to_str().unwrap_or(""), ext);
                path_with_ext.set_extension(new_ext);
            } else {
                let new_name = format!("{}{}", path_with_ext.file_name().unwrap_or_default().to_str().unwrap_or(""), ext);
                path_with_ext.set_file_name(new_name);
            }
            path_with_ext
        } else {
            path.clone()
        };
        
        // Create compressor if needed
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                match zstd::Encoder::new(Vec::new(), *level) {
                    Ok(encoder) => Some(encoder),
                    Err(_) => None, // Fall back to no compression on error
                }
            }
        };
        
        // Create and open the file for writing with appropriate flags
        let file = if config.direct_io {
            #[cfg(unix)]
            {
                use tokio::fs::OpenOptions;
                
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&final_path)
                    .await?
            }
            #[cfg(not(unix))]
            {
                // Fallback to regular I/O on non-Unix systems
                fs::File::create(&final_path).await?
            }
        } else {
            fs::File::create(&final_path).await?
        };
        
        // Create aligned buffer for O_DIRECT operations
        let buffer = if config.direct_io {
            create_aligned_buffer(config.min_io_size * 2, config.alignment)
        } else {
            Vec::new()
        };
        
        Ok(Self {
            file: Some(file),
            path: final_path,
            config,
            bytes_written: 0,
            finalized: false,
            buffer,
            hasher: Hasher::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        })
    }
}

#[async_trait]
impl ObjectWriter for DirectIOWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        
        // Update checksum for all written data
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        
        // Handle compression
        let data_to_write = if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and collect compressed data
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            let result = compressed.clone();
            compressed.clear();
            result
        } else {
            // No compression - use original chunk
            chunk.to_vec()
        };
        
        let file = match self.file.as_mut() {
            Some(f) => f,
            None => bail!("Writer has been finalized or cancelled"),
        };

        if !self.config.direct_io {
            // Regular I/O - write directly
            file.write_all(&data_to_write).await?;
            
            if self.config.sync_writes {
                file.sync_all().await?;
            }
        } else {
            // DirectIO - need to handle alignment and buffering with proper page alignment
            let alignment = self.config.alignment;
            let min_io_size = self.config.min_io_size;
            
            // Ensure buffer has enough capacity for the new data
            let new_len = self.buffer.len() + data_to_write.len();
            if new_len > self.buffer.capacity() {
                let new_capacity = ((new_len + min_io_size - 1) / min_io_size) * min_io_size;
                resize_aligned_buffer(&mut self.buffer, new_capacity, alignment);
            }
            
            // Copy data to aligned buffer
            let old_len = self.buffer.len();
            unsafe {
                let dst = self.buffer.as_mut_ptr().add(old_len);
                std::ptr::copy_nonoverlapping(data_to_write.as_ptr(), dst, data_to_write.len());
                self.buffer.set_len(old_len + data_to_write.len());
            }
            
            // Write aligned chunks when we have enough data
            while self.buffer.len() >= min_io_size {
                let write_size = (self.buffer.len() / alignment) * alignment;
                if write_size > 0 {
                    // Create aligned write buffer by copying from our aligned buffer
                    let write_slice = &self.buffer[..write_size];
                    
                    // Write the aligned data directly from our aligned buffer
                    file.write_all(write_slice).await?;
                    
                    // For O_DIRECT, we must sync immediately to ensure data persistence  
                    file.sync_all().await?;
                    
                    // Move remaining data to front of buffer
                    let remaining = self.buffer.len() - write_size;
                    if remaining > 0 {
                        unsafe {
                            let src = self.buffer.as_ptr().add(write_size);
                            let dst = self.buffer.as_mut_ptr();
                            std::ptr::copy(src, dst, remaining);
                        }
                    }
                    unsafe {
                        self.buffer.set_len(remaining);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        
        if let Some(mut file) = self.file.take() {
            // Finalize compression if enabled
            if let Some(compressor) = self.compressor.take() {
                let compressed_data = compressor.finish()?;
                self.compressed_bytes = compressed_data.len() as u64;
                
                if !compressed_data.is_empty() {
                    if !self.config.direct_io {
                        // Regular I/O - write directly
                        file.write_all(&compressed_data).await?;
                    } else {
                        // DirectIO - add to aligned buffer properly
                        let alignment = self.config.alignment;
                        let min_io_size = self.config.min_io_size;
                        
                        // Ensure buffer has enough capacity for the compressed data
                        let new_len = self.buffer.len() + compressed_data.len();
                        if new_len > self.buffer.capacity() {
                            let new_capacity = ((new_len + min_io_size - 1) / min_io_size) * min_io_size;
                            resize_aligned_buffer(&mut self.buffer, new_capacity, alignment);
                        }
                        
                        // Copy compressed data to aligned buffer
                        let old_len = self.buffer.len();
                        unsafe {
                            let dst = self.buffer.as_mut_ptr().add(old_len);
                            std::ptr::copy_nonoverlapping(compressed_data.as_ptr(), dst, compressed_data.len());
                            self.buffer.set_len(old_len + compressed_data.len());
                        }
                    }
                }
            } else {
                self.compressed_bytes = self.bytes_written;
            }
            
            // Write any remaining buffered data for DirectIO
            if self.config.direct_io && !self.buffer.is_empty() {
                // For O_DIRECT, we use the hybrid I/O approach:
                // 1. Write aligned chunks with O_DIRECT for performance
                // 2. Write the final unaligned data with standard buffered I/O
                
                let alignment = self.config.alignment;
                let buffer_len = self.buffer.len();
                
                if buffer_len >= alignment {
                    // Write the aligned portion using the existing O_DIRECT file
                    let aligned_len = (buffer_len / alignment) * alignment;
                    
                    let result = file.write_all(&self.buffer[..aligned_len]).await;
                    match result {
                        Ok(()) => {},
                        Err(e) => {
                            return Err(e.into());
                        }
                    }
                    
                    // Keep only the unaligned remainder - move data to preserve alignment
                    let remaining = buffer_len - aligned_len;
                    if remaining > 0 {
                        unsafe {
                            let src = self.buffer.as_ptr().add(aligned_len);
                            let dst = self.buffer.as_mut_ptr();
                            std::ptr::copy(src, dst, remaining);
                        }
                    }
                    unsafe {
                        self.buffer.set_len(remaining);
                    }
                }
                
                // If there's still unaligned data, write it using buffered I/O
                if !self.buffer.is_empty() {
                    // First, ensure all O_DIRECT data is synced before switching to buffered I/O
                    file.sync_all().await?;
                    
                    // Close the O_DIRECT file and reopen without O_DIRECT for the final write
                    drop(file);
                    
                    // Reopen the file in append mode without O_DIRECT
                    let mut buffered_file = tokio::fs::OpenOptions::new()
                        .append(true)  // Only append - this preserves existing content
                        .open(&self.path)
                        .await?;
                    
                    // Write the remaining unaligned data using standard buffered I/O
                    buffered_file.write_all(&self.buffer).await?;
                    buffered_file.flush().await?;
                    buffered_file.sync_all().await?;
                } else {
                    // No unaligned data, just sync the O_DIRECT file
                    // Note: O_DIRECT bypasses page cache, so flush() is not needed
                    let sync_result = file.sync_all().await;
                    match sync_result {
                        Ok(()) => {},
                        Err(e) => {
                            return Err(e.into());
                        }
                    }
                }
            } else {
                // Standard file I/O or no remaining buffer
                if !self.config.direct_io {
                    // Only flush if not using O_DIRECT (O_DIRECT bypasses page cache)
                    let flush_result = file.flush().await;
                    match flush_result {
                        Ok(()) => {},
                        Err(e) => {
                            return Err(e.into());
                        }
                    }
                }
                file.sync_all().await?;
            }
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
        self.buffer.clear(); // Clear any buffered data
        
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        
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

/// Enhanced FileSystem adapter with O_DIRECT support
pub struct ConfigurableFileSystemObjectStore {
    config: Arc<FileSystemConfig>,
}

impl Clone for ConfigurableFileSystemObjectStore {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
        }
    }
}

impl ConfigurableFileSystemObjectStore {
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config: Arc::new(config) }
    }

    pub fn with_config(config: FileSystemConfig) -> Self {
        Self::new(config)
    }

    pub fn with_direct_io() -> Self {
        Self::new(FileSystemConfig::direct_io())
    }

    pub fn high_performance() -> Self {
        Self::new(FileSystemConfig::high_performance())
    }

    /// Test if the filesystem at the given path supports O_DIRECT
    pub async fn test_direct_io_support(path: &Path) -> bool {
        #[cfg(unix)]
        {
            // Create a test file to check O_DIRECT support
            let test_path = path.join(".test_direct_io");
            let result = tokio::task::spawn_blocking(move || -> bool {
                use std::fs::OpenOptions;
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                
                // Try to create a small file with O_DIRECT
                match OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&test_path)
                {
                    Ok(mut file) => {
                        // Try to write aligned data
                        let page_size = get_system_page_size();
                        let test_data = vec![0u8; page_size];
                        let write_result = file.write_all(&test_data);
                        
                        // Clean up test file
                        let _ = std::fs::remove_file(&test_path);
                        
                        write_result.is_ok()
                    }
                    Err(_) => false
                }
            }).await;
            
            result.unwrap_or(false)
        }
        
        #[cfg(not(unix))]
        {
            false // O_DIRECT not supported on non-Unix systems
        }
    }

    /// Convert a URI to a filesystem path (accepts both file:// and direct:// schemes)
    fn uri_to_path(uri: &str) -> Result<PathBuf> {
        let path = if uri.starts_with("file://") {
            &uri[7..] // Strip "file://" prefix
        } else if uri.starts_with("direct://") {
            &uri[9..] // Strip "direct://" prefix  
        } else {
            bail!("FileSystemObjectStore expects file:// or direct:// URI, got: {}", uri);
        };
        
        Ok(PathBuf::from(path))
    }

    /// Check if URI uses a valid file scheme (file:// or direct://)
    fn is_valid_file_uri(uri: &str) -> bool {
        uri.starts_with("file://") || uri.starts_with("direct://")
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
            content_type: None,
            content_language: None,
            content_encoding: None,
            cache_control: None,
            content_disposition: None,
            expires: None,
            storage_class: Some("DIRECT_IO".to_string()),
            server_side_encryption: None,
            ssekms_key_id: None,
            sse_customer_algorithm: None,
            version_id: None,
            replication_status: None,
            metadata: HashMap::new(),
        })
    }

    /// Align buffer for O_DIRECT operations
    fn align_buffer(&self, data: &[u8]) -> Vec<u8> {
        if !self.config.direct_io {
            return data.to_vec();
        }

        let alignment = self.config.alignment;
        let aligned_size = ((data.len() + alignment - 1) / alignment) * alignment;
        let mut aligned_buffer = create_aligned_buffer(aligned_size, alignment);
        unsafe { 
            aligned_buffer.set_len(aligned_size);
            aligned_buffer[..data.len()].copy_from_slice(data);
            // Zero out the padding
            if data.len() < aligned_size {
                aligned_buffer[data.len()..].fill(0);
            }
        }
        aligned_buffer
    }

    /// Create OpenOptions with appropriate flags for the configuration
    fn create_open_options(&self, read: bool, write: bool) -> fs::OpenOptions {
        let mut options = fs::OpenOptions::new();
        options.read(read).write(write);

        if write {
            options.create(true).truncate(true);
        }

        #[cfg(unix)]
        {
            if self.config.direct_io {
                // O_DIRECT flag (Linux/Unix)
                options.custom_flags(libc::O_DIRECT);
            }
            if self.config.sync_writes && write {
                // O_SYNC flag for synchronous writes
                options.custom_flags(libc::O_SYNC);
            }
        }

        options
    }

    /// Read file with O_DIRECT if configured
    async fn read_file_direct(&self, path: &Path) -> Result<Bytes> {
        if !self.config.direct_io {
            // Fallback to normal tokio read
            let data = fs::read(path).await?;
            return Ok(Bytes::from(data));
        }

        // For O_DIRECT, we need to be careful about alignment and may need to fall back
        match self.try_read_file_direct(path).await {
            Ok(data) => Ok(Bytes::from(data)),
            Err(e) => {
                // If O_DIRECT fails, fall back to regular I/O
                warn!("O_DIRECT read failed for {}, falling back to regular I/O: {}", path.display(), e);
                let data = fs::read(path).await?;
                Ok(Bytes::from(data))
            }
        }
    }

    /// Try to read file with O_DIRECT, may fail due to alignment or filesystem support
    async fn try_read_file_direct(&self, path: &Path) -> Result<Vec<u8>> {
        let metadata = fs::metadata(path).await?;
        let file_size = metadata.len() as usize;
        
        if file_size == 0 {
            return Ok(Vec::new());
        }
        
        // For small files or when O_DIRECT would be inefficient, use regular I/O
        if file_size < self.config.min_io_size {
            return Ok(fs::read(path).await?);
        }
        
        #[cfg(unix)]
        {
            let path = path.to_owned();
            let alignment = self.config.alignment;
            
            // Use spawn_blocking to run sync I/O in thread pool
            tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
                use std::fs::OpenOptions;
                use std::io::Read;
                use std::os::unix::fs::OpenOptionsExt;
                
                let mut file = OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)?;
                
                // Read in aligned chunks using aligned buffer
                let mut result = Vec::new();
                let mut temp_buffer = create_aligned_buffer(alignment, alignment);
                unsafe { temp_buffer.set_len(alignment); }
                let mut total_read = 0;
                
                while total_read < file_size {
                    let bytes_read = file.read(&mut temp_buffer)?;
                    if bytes_read == 0 {
                        break;
                    }
                    
                    let copy_size = std::cmp::min(bytes_read, file_size - total_read);
                    result.extend_from_slice(&temp_buffer[..copy_size]);
                    total_read += copy_size;
                }
                
                result.truncate(file_size);
                Ok(result)
            }).await?
        }
        
        #[cfg(not(unix))]
        {
            // Fallback for non-Unix systems
            Ok(fs::read(path).await?)
        }
    }

    /// Write file with O_DIRECT if configured
    async fn write_file_direct(&self, path: &Path, data: &[u8]) -> Result<()> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        if !self.config.direct_io {
            // Fallback to normal tokio write
            return Ok(fs::write(path, data).await?);
        }

        // Try O_DIRECT write, fall back to regular write if it fails
        match self.try_write_file_direct(path, data).await {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!("O_DIRECT write failed for {}, falling back to regular I/O: {}", path.display(), e);
                Ok(fs::write(path, data).await?)
            }
        }
    }

    /// Try to write file with O_DIRECT, may fail due to alignment or filesystem support
    async fn try_write_file_direct(&self, path: &Path, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(fs::write(path, data).await?);
        }

        // For small files, use regular I/O
        if data.len() < self.config.min_io_size {
            return Ok(fs::write(path, data).await?);
        }

        #[cfg(unix)]
        {
            let path = path.to_owned();
            let data = data.to_vec();
            let alignment = self.config.alignment;
            let sync_writes = self.config.sync_writes;
            
            // Use spawn_blocking for sync I/O
            tokio::task::spawn_blocking(move || -> Result<()> {
                use std::fs::OpenOptions;
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                
                let mut flags = libc::O_DIRECT;
                if sync_writes {
                    flags |= libc::O_SYNC;
                }
                
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .custom_flags(flags)
                    .open(&path)?;
                
                // Write data in aligned chunks
                let mut written = 0;
                
                // Create single aligned buffer for all writes
                let mut aligned_chunk = create_aligned_buffer(alignment, alignment);
                unsafe { aligned_chunk.set_len(alignment); }
                
                while written < data.len() {
                    let remaining = data.len() - written;
                    let chunk_size = std::cmp::min(remaining, alignment);
                    
                    // Clear and copy data to aligned buffer
                    unsafe { aligned_chunk.set_len(0); aligned_chunk.set_len(alignment); }
                    aligned_chunk[..chunk_size].copy_from_slice(&data[written..written + chunk_size]);
                    if chunk_size < alignment {
                        aligned_chunk[chunk_size..].fill(0);
                    }
                    
                    // Write full aligned chunk
                    file.write_all(&aligned_chunk)?;
                    written += chunk_size;
                }
                
                file.flush()?;
                
                // Truncate to original size
                file.set_len(data.len() as u64)?;
                
                Ok(())
            }).await?
        }
        
        #[cfg(not(unix))]
        {
            // Fallback for non-Unix systems
            Ok(fs::write(path, data).await?)
        }
    }

    /// Read range with O_DIRECT support
    async fn read_range_direct(&self, path: &Path, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !self.config.direct_io {
            // Fallback to normal implementation
            let mut file = fs::File::open(path).await?;
            file.seek(std::io::SeekFrom::Start(offset)).await?;
            
            let read_length = length.unwrap_or(u64::MAX);
            let mut buffer = Vec::new();
            
            if read_length == u64::MAX {
                file.read_to_end(&mut buffer).await?;
            } else {
                buffer.resize(read_length as usize, 0);
                file.read_exact(&mut buffer).await?;
            }
            
            return Ok(Bytes::from(buffer));
        }

        // Try O_DIRECT first, fall back to regular I/O if it fails
        match self.try_read_range_direct(path, offset, length).await {
            Ok(data) => Ok(data),  // Already Bytes now (v0.9.9+)
            Err(e) => {
                warn!("O_DIRECT range read failed for {}, falling back to regular I/O: {}", path.display(), e);
                // Fall back to regular I/O
                let mut file = fs::File::open(path).await?;
                file.seek(std::io::SeekFrom::Start(offset)).await?;
                
                let read_length = length.unwrap_or(u64::MAX);
                let mut buffer = Vec::new();
                
                if read_length == u64::MAX {
                    file.read_to_end(&mut buffer).await?;
                } else {
                    buffer.resize(read_length as usize, 0);
                    file.read_exact(&mut buffer).await?;
                }
                
                Ok(Bytes::from(buffer))
            }
        }
    }

    /// Try to read file range with O_DIRECT, may fail due to alignment or filesystem support
    /// 
    /// **v0.9.9 Enhancement**: Uses buffer pool to eliminate allocation churn
    /// - Borrows aligned buffer from pool (reused across operations)
    /// - Returns only requested subrange (minimal copy vs full buffer copy)
    /// - Returns buffer to pool for future reuse
    async fn try_read_range_direct(&self, path: &Path, offset: u64, length: Option<u64>) -> Result<Bytes> {
        // O_DIRECT range read requires careful alignment
        let alignment = self.config.alignment as u64;
        let aligned_offset = (offset / alignment) * alignment;
        let offset_adjustment = (offset - aligned_offset) as usize;
        
        let read_length = length.unwrap_or_else(|| {
            // Read to end of file
            let metadata = std::fs::metadata(path).unwrap();
            metadata.len() - offset
        });
        
        let total_length = offset_adjustment + read_length as usize;
        let aligned_length = ((total_length + self.config.alignment - 1) / self.config.alignment) * self.config.alignment;
        
        // Borrow aligned buffer from pool if available, otherwise allocate fresh (v0.9.9+)
        let mut aligned: AlignedBuf = if let Some(pool) = &self.config.buffer_pool {
            let b = pool.take().await;
            // Grow buffer if pooled buffer is too small
            if b.len() < aligned_length {
                trace!("Pool buffer too small ({} < {}), allocating larger buffer", b.len(), aligned_length);
                AlignedBuf::new(aligned_length, self.config.alignment)
            } else {
                b
            }
        } else {
            // No pool configured, allocate fresh buffer
            AlignedBuf::new(aligned_length, self.config.alignment)
        };
        
        let options = self.create_open_options(true, false);
        let mut file = options.open(path).await?;
        
        file.seek(std::io::SeekFrom::Start(aligned_offset)).await?;
        let bytes_read = file.read(aligned.as_mut_slice()).await?;
        
        // Extract the requested subrange (single small copy vs entire buffer copy)
        let start = offset_adjustment.min(bytes_read);
        let end = start.saturating_add(read_length as usize).min(bytes_read);
        let needed = end.saturating_sub(start);
        
        let mut out = BytesMut::with_capacity(needed);
        out.extend_from_slice(&aligned.as_slice()[start..end]);
        let result = out.freeze();
        
        // Return aligned buffer to pool for reuse (v0.9.9+)
        if let Some(pool) = &self.config.buffer_pool {
            pool.give(aligned).await;
        }
        
        Ok(result)
    }

    /// Get file using RangeEngine for concurrent downloads
    async fn get_with_range_engine(&self, uri: &str, object_size: u64) -> Result<Bytes> {
        debug!(
            "DirectIO using RangeEngine: uri={}, size={} MB, threshold={} MB",
            uri,
            object_size / (1024 * 1024),
            self.config.range_engine.min_split_size / (1024 * 1024)
        );

        let engine = RangeEngine::new(self.config.range_engine.clone());
        let store = self.clone();
        let uri_owned = uri.to_string();

        let get_range_fn = move |offset: u64, length: u64| {
            let store = store.clone();
            let uri = uri_owned.clone();
            async move { store.get_range(&uri, offset, Some(length)).await }
        };

        let (bytes, stats) = engine.download(object_size, get_range_fn, None).await?;

        info!(
            "DirectIO RangeEngine complete: {} MB in {:.2}s ({:.2} MB/s, {} ranges)",
            bytes.len() / (1024 * 1024),
            stats.elapsed_time.as_secs_f64(),
            stats.throughput_mbps(),
            stats.ranges_processed
        );

        Ok(bytes)
    }

    /// Simple file read without RangeEngine (for small files)
    async fn get_simple(&self, uri: &str, _size: u64) -> Result<Bytes> {
        let path = Self::uri_to_path(uri)?;
        trace!("DirectIO using simple read for file: {}", path.display());
        self.read_file_direct(&path).await
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
                let file_uri = Self::path_to_uri(&entry_path);
                results.push(file_uri);
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl ObjectStore for ConfigurableFileSystemObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if !Self::is_valid_file_uri(uri) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        // Get file size to determine if we should use RangeEngine
        let metadata = fs::metadata(&path).await?;
        let size = metadata.len();

        // Use RangeEngine for large files if enabled
        if self.config.enable_range_engine && size >= self.config.range_engine.min_split_size {
            return self.get_with_range_engine(uri, size).await;
        }

        // Use simple read for small files or when RangeEngine disabled
        self.get_simple(uri, size).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !Self::is_valid_file_uri(uri) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }
        
        self.read_range_direct(&path, offset, length).await
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        if !Self::is_valid_file_uri(uri) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
        let path = Self::uri_to_path(uri)?;
        
        self.write_file_direct(&path, data).await
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        if !Self::is_valid_file_uri(uri) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
        
        if !self.config.direct_io {
            // Fallback to regular put for non-direct I/O
            return self.put(uri, data).await;
        }
        
        // For O_DIRECT multipart, write in aligned chunks
        let chunk_size = part_size.unwrap_or(self.config.min_io_size);
        
        let path = Self::uri_to_path(uri)?;
        
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        let options = self.create_open_options(false, true);
        let mut file = options.open(&path).await?;
        
        for chunk in data.chunks(chunk_size) {
            let aligned_chunk = self.align_buffer(chunk);
            file.write_all(&aligned_chunk).await?;
        }
        
        file.flush().await?;
        
        // Truncate to exact size
        file.set_len(data.len() as u64).await?;
        
        Ok(())
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        if !Self::is_valid_file_uri(uri_prefix) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
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
        if !Self::is_valid_file_uri(uri) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
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
        if !Self::is_valid_file_uri(uri) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
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
        if !Self::is_valid_file_uri(uri_prefix) { bail!("FileSystemObjectStore expected file:// or direct:// URI"); }
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
        if !Self::is_valid_file_uri(src_uri) { 
            bail!("FileSystemObjectStore expected file:// or direct:// URI for source"); 
        }
        if !Self::is_valid_file_uri(dst_uri) { 
            bail!("FileSystemObjectStore expected file:// or direct:// URI for destination"); 
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
        if !Self::is_valid_file_uri(uri) { 
            bail!("FileSystemObjectStore expected file:// or direct:// URI"); 
        }
        
        let path = Self::uri_to_path(uri)?;
        
        // For direct I/O, we might want a specialized writer, but for now use the same
        // as regular filesystem with potential for future direct I/O optimization
        Ok(Box::new(DirectIOWriter::new(path, (*self.config).clone()).await?))
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        if !Self::is_valid_file_uri(uri) { 
            bail!("FileSystemObjectStore expected file:// or direct:// URI"); 
        }
        
        let path = Self::uri_to_path(uri)?;
        
        // For direct I/O with compression
        Ok(Box::new(DirectIOWriter::new_with_compression(path, (*self.config).clone(), compression).await?))
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        
        let path = Self::uri_to_path(uri)?;
        
        // Create DirectIOWriter with optional compression
        if let Some(compression) = options.compression {
            Ok(Box::new(DirectIOWriter::new_with_compression(path, (*self.config).clone(), compression).await?))
        } else {
            Ok(Box::new(DirectIOWriter::new(path, (*self.config).clone()).await?))
        }
    }
}

impl ConfigurableFileSystemObjectStore {
    #[inline]
    pub fn boxed(config: FileSystemConfig) -> Box<dyn ObjectStore> {
        Box::new(Self::new(config))
    }

    #[inline]
    pub fn boxed_direct_io() -> Box<dyn ObjectStore> {
        Box::new(Self::with_direct_io())
    }

    #[inline]
    pub fn boxed_high_performance() -> Box<dyn ObjectStore> {
        Box::new(Self::high_performance())
    }
}
