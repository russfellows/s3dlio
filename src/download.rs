// src/download.rs
//
// High-performance download engine for S3 operations
// Optimized for AI/ML workloads with memory-first architecture and optional O_DIRECT file I/O

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::memory::{BufferPool, AlignedBuf};

/// Download target specification
#[derive(Debug, Clone)]
pub enum Target {
    /// Download directly to memory using aligned buffer pools
    /// This is the primary path for AI/ML workloads
    Memory {
        /// Buffer pool for managing aligned memory
        pool: Arc<BufferPool>,
    },
    /// Download to file using O_DIRECT for maximum performance
    /// Used when data doesn't fit in memory (3x+ memory size workloads)
    DirectFile {
        /// Target file path
        path: PathBuf,
        /// Pre-allocate file space (recommended for large files)
        preallocate: bool,
    },
}

/// Configuration for download operations
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Size of each range part in bytes (default: 16 MiB)
    pub part_size: usize,
    /// Maximum concurrent range requests per object (default: 64)
    pub max_concurrent: usize,
    /// Number of HTTP client pools to use (default: 4, reduces contention)
    pub client_pools: usize,
    /// Timeout for individual range requests
    pub request_timeout: std::time::Duration,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            part_size: 16 * 1024 * 1024,  // 16 MiB parts
            max_concurrent: 64,            // 64 concurrent ranges
            client_pools: 4,               // 4 client pools to reduce contention
            request_timeout: std::time::Duration::from_secs(30),
        }
    }
}

/// Represents a completed part of a download
#[derive(Debug)]
pub struct CompletedPart {
    /// Object identifier
    pub object_id: String,
    /// Part index within the object
    pub part_index: usize,
    /// Byte offset within the object
    pub offset: u64,
    /// Number of bytes in this part
    pub bytes_len: usize,
    /// The actual data (for Memory target)
    pub buffer: Option<AlignedBuf>,
}

/// Statistics for a completed download operation
#[derive(Debug, Clone)]
pub struct DownloadStats {
    /// Total number of objects processed
    pub objects_count: usize,
    /// Total number of parts processed
    pub parts_count: usize,
    /// Total bytes downloaded
    pub total_bytes: u64,
    /// Time taken for the download
    pub elapsed: std::time::Duration,
    /// Average throughput in bytes per second
    pub throughput_bps: f64,
}

/// High-performance download engine
pub struct DownloadEngine {
    config: DownloadConfig,
}

impl DownloadEngine {
    /// Create a new download engine with the specified configuration
    pub fn new(config: DownloadConfig) -> Self {
        Self { config }
    }

    /// Download a single object to the specified target
    /// 
    /// # Arguments
    /// * `uri` - S3 URI of the object to download
    /// * `target` - Where to store the downloaded data
    /// 
    /// # Returns
    /// * `DownloadStats` - Statistics about the download operation
    pub async fn download_object(&self, uri: &str, target: Target) -> Result<DownloadStats> {
        let start_time = std::time::Instant::now();
        
        match target {
            Target::Memory { pool } => {
                self.download_to_memory(uri, pool).await
            }
            Target::DirectFile { path, preallocate } => {
                self.download_to_file(uri, path, preallocate).await
            }
        }
        .map(|mut stats| {
            stats.elapsed = start_time.elapsed();
            stats.throughput_bps = stats.total_bytes as f64 / stats.elapsed.as_secs_f64();
            stats
        })
    }

    /// Download multiple objects concurrently
    /// 
    /// # Arguments
    /// * `uris` - List of S3 URIs to download
    /// * `target_fn` - Function to determine target for each URI
    /// 
    /// # Returns
    /// * `DownloadStats` - Aggregated statistics for all downloads
    pub async fn download_objects<F>(&self, uris: &[String], mut target_fn: F) -> Result<DownloadStats>
    where
        F: FnMut(&str) -> Target,
    {
        let start_time = std::time::Instant::now();
        
        // For now, implement sequential downloads
        // TODO: Implement true concurrent downloads with proper resource management
        let mut total_stats = DownloadStats {
            objects_count: 0,
            parts_count: 0,
            total_bytes: 0,
            elapsed: std::time::Duration::default(),
            throughput_bps: 0.0,
        };

        for uri in uris {
            let target = target_fn(uri);
            let stats = self.download_object(uri, target).await?;
            
            total_stats.objects_count += stats.objects_count;
            total_stats.parts_count += stats.parts_count;
            total_stats.total_bytes += stats.total_bytes;
        }

        total_stats.elapsed = start_time.elapsed();
        total_stats.throughput_bps = total_stats.total_bytes as f64 / total_stats.elapsed.as_secs_f64();

        Ok(total_stats)
    }

    /// Stream download parts as they complete (for Memory target)
    /// 
    /// This is the zero-copy path for AI/ML workloads - parts are delivered
    /// as they complete without waiting for the entire object.
    pub async fn stream_download_parts(
        &self,
        uri: &str,
        pool: Arc<BufferPool>,
    ) -> Result<mpsc::Receiver<CompletedPart>> {
        let (tx, _rx) = mpsc::channel::<CompletedPart>(self.config.max_concurrent);
        
        // Close the channel immediately for now - this is a placeholder
        // In a real implementation, we would spawn tasks to download parts
        // and send them through tx as they complete
        drop(tx);
        
        // For now, return an error indicating this is not yet implemented
        anyhow::bail!("Stream download not yet implemented for URI: {}, pool capacity: {}", uri, pool.capacity())
    }

    // Private implementation methods
    
    async fn download_to_memory(&self, uri: &str, pool: Arc<BufferPool>) -> Result<DownloadStats> {
        // Placeholder implementation - will be replaced when range engine is integrated
        anyhow::bail!("Memory download not yet implemented. URI: {}, Pool capacity: {}", 
                     uri, pool.capacity())
    }

    async fn download_to_file(&self, uri: &str, path: PathBuf, preallocate: bool) -> Result<DownloadStats> {
        // Placeholder implementation - will be replaced when O_DIRECT file backend is implemented
        anyhow::bail!("File download not yet implemented. URI: {}, Path: {}, Preallocate: {}", 
                     uri, path.display(), preallocate)
    }
}

/// Builder for creating download configurations
pub struct DownloadConfigBuilder {
    config: DownloadConfig,
}

impl DownloadConfigBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: DownloadConfig::default(),
        }
    }

    /// Set the part size for range requests
    pub fn part_size(mut self, size: usize) -> Self {
        self.config.part_size = size;
        self
    }

    /// Set the maximum concurrent requests
    pub fn max_concurrent(mut self, concurrent: usize) -> Self {
        self.config.max_concurrent = concurrent;
        self
    }

    /// Set the number of HTTP client pools
    pub fn client_pools(mut self, pools: usize) -> Self {
        self.config.client_pools = pools;
        self
    }

    /// Set the request timeout
    pub fn request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.request_timeout = timeout;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> DownloadConfig {
        self.config
    }
}

impl Default for DownloadConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::BufferPoolConfig;

    #[test]
    fn download_config_builder() {
        let config = DownloadConfigBuilder::new()
            .part_size(32 * 1024 * 1024)
            .max_concurrent(128)
            .client_pools(8)
            .build();

        assert_eq!(config.part_size, 32 * 1024 * 1024);
        assert_eq!(config.max_concurrent, 128);
        assert_eq!(config.client_pools, 8);
    }

    #[tokio::test]
    async fn target_memory() {
        let pool = BufferPoolConfig::default().build();
        let target = Target::Memory { pool: pool.clone() };
        
        match target {
            Target::Memory { .. } => (),
            _ => panic!("Expected Memory target"),
        }
    }

    #[test]
    fn target_direct_file() {
        let target = Target::DirectFile {
            path: "/tmp/test.dat".into(),
            preallocate: true,
        };
        
        match target {
            Target::DirectFile { path, preallocate } => {
                assert_eq!(path, PathBuf::from("/tmp/test.dat"));
                assert!(preallocate);
            }
            _ => panic!("Expected DirectFile target"),
        }
    }

    #[tokio::test]
    async fn download_engine_creation() {
        let config = DownloadConfig::default();
        let engine = DownloadEngine::new(config);
        
        // Just verify it was created successfully
        assert_eq!(engine.config.part_size, 16 * 1024 * 1024);
    }
}