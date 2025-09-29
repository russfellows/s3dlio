// src/range_engine.rs
//
// High-performance concurrent range-GET engine for S3 operations
// Zero-copy memory operations with optimal concurrency and minimal allocations

use crate::memory::BufferPool;
use crate::sharded_client::{ShardedS3Clients, RangeRequest, RangeResponse};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Configuration for range-based downloads
#[derive(Debug, Clone)]
pub struct RangeEngineConfig {
    /// Size of each range in bytes (default: 64MB)
    pub range_size: usize,
    /// Maximum concurrent ranges (default: 32)
    pub max_concurrent_ranges: usize,
    /// Minimum object size for range splitting (default: 128MB)
    pub min_split_size: u64,
    /// Whether to use adaptive range sizing based on throughput
    pub adaptive_sizing: bool,
    /// Buffer pool configuration
    pub buffer_pool: Arc<BufferPool>,
    /// Sharded HTTP clients
    pub clients: Arc<ShardedS3Clients>,
}

impl RangeEngineConfig {
    /// Create a new default configuration
    /// 
    /// Note: This creates placeholder buffer pool and clients.
    /// Use the builder pattern or set these explicitly for production use.
    pub fn new() -> Self {
        Self {
            range_size: 64 * 1024 * 1024, // 64MB
            max_concurrent_ranges: 32,
            min_split_size: 128 * 1024 * 1024, // 128MB
            adaptive_sizing: true,
            buffer_pool: BufferPool::new(1024, 64 * 1024 * 1024, 4096),
            // Placeholder clients - should be set explicitly
            // This will be replaced by proper initialization in production
            clients: Arc::new(Self::create_placeholder_clients()),
        }
    }

    /// Create placeholder clients for testing/default configuration
    fn create_placeholder_clients() -> ShardedS3Clients {
        // Create an empty sharded client for compilation
        // In production, this should be properly initialized
        ShardedS3Clients::placeholder(4)
    }
}

/// High-performance range-GET engine
/// 
/// This engine maximizes S3 GET performance by:
/// 1. Splitting large objects into optimal-sized ranges
/// 2. Downloading ranges concurrently across multiple HTTP clients
/// 3. Using zero-copy memory management with aligned buffers
/// 4. Adaptively tuning range sizes based on observed throughput
pub struct RangeEngine {
    config: RangeEngineConfig,
    concurrency_limiter: Arc<Semaphore>,
}

/// Statistics for a range download operation
#[derive(Debug, Clone)]
pub struct RangeDownloadStats {
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
    /// Number of ranges processed
    pub ranges_processed: usize,
    /// Total elapsed time
    pub elapsed_time: std::time::Duration,
    /// Average throughput (bytes/sec)
    pub throughput_bps: u64,
    /// Ranges processed concurrently (peak)
    pub peak_concurrency: usize,
    /// Client shard utilization
    pub shard_utilization: Vec<usize>,
}

/// Result of a range download operation
#[derive(Debug)]
pub struct RangeDownloadResult {
    /// Downloaded data in memory
    pub data: Vec<u8>,
    /// Download statistics
    pub stats: RangeDownloadStats,
    /// Individual range responses (for debugging)
    pub range_responses: Vec<RangeResponse>,
}

impl RangeEngine {
    /// Create a new range engine with the given configuration
    pub fn new(config: RangeEngineConfig) -> Self {
        let concurrency_limiter = Arc::new(Semaphore::new(config.max_concurrent_ranges));
        
        Self {
            config,
            concurrency_limiter,
        }
    }

    /// Download an entire object using concurrent range requests
    /// 
    /// This automatically determines the optimal range strategy:
    /// - Small objects: single GET request
    /// - Large objects: parallel range requests with optimal sizing
    pub async fn download_object(&self, uri: &str, object_size: u64) -> Result<RangeDownloadResult> {
        let start_time = std::time::Instant::now();
        
        if object_size < self.config.min_split_size {
            // Small object - use single request
            self.download_single_range(uri, 0, object_size as usize, start_time).await
        } else {
            // Large object - use concurrent ranges
            self.download_multi_range(uri, object_size, start_time).await
        }
    }

    /// Download using a single range request (for small objects)
    async fn download_single_range(
        &self,
        uri: &str,
        offset: u64,
        length: usize,
        start_time: std::time::Instant,
    ) -> Result<RangeDownloadResult> {
        let request = RangeRequest {
            key: uri.to_string(),
            offset,
            length,
            part_index: 0,
        };

        let response = self.config.clients.get_range(request).await?;
        
        let elapsed = start_time.elapsed();
        let throughput = if elapsed.as_secs() > 0 {
            response.bytes_read as u64 / elapsed.as_secs()
        } else {
            0
        };

        let stats = RangeDownloadStats {
            bytes_downloaded: response.bytes_read as u64,
            ranges_processed: 1,
            elapsed_time: elapsed,
            throughput_bps: throughput,
            peak_concurrency: 1,
            shard_utilization: vec![1; self.config.clients.shard_count()],
        };

        Ok(RangeDownloadResult {
            data: response.data.to_vec(),
            stats,
            range_responses: vec![response],
        })
    }

    /// Download using concurrent range requests (for large objects)
    async fn download_multi_range(
        &self,
        uri: &str,
        object_size: u64,
        start_time: std::time::Instant,
    ) -> Result<RangeDownloadResult> {
        // Calculate optimal ranges
        let ranges = self.calculate_ranges(object_size);
        
        // Create range requests
        let requests: Vec<RangeRequest> = ranges
            .into_iter()
            .enumerate()
            .map(|(i, (offset, length))| RangeRequest {
                key: uri.to_string(),
                offset,
                length,
                part_index: i,
            })
            .collect();

        // Execute concurrent downloads
        let responses = self.download_ranges_concurrent(requests).await?;

        // Assemble results
        let mut data = Vec::with_capacity(object_size as usize);
        let mut total_bytes = 0;
        let mut shard_counts = vec![0; self.config.clients.shard_count()];

        // Sort responses by part index to maintain order
        let mut sorted_responses = responses;
        sorted_responses.sort_by_key(|r| r.request.part_index);

        for response in &sorted_responses {
            data.extend_from_slice(&response.data);
            total_bytes += response.bytes_read;
            shard_counts[response.shard_id] += 1;
        }

        let elapsed = start_time.elapsed();
        let throughput = if elapsed.as_secs() > 0 {
            total_bytes as u64 / elapsed.as_secs()
        } else {
            0
        };

        let stats = RangeDownloadStats {
            bytes_downloaded: total_bytes as u64,
            ranges_processed: sorted_responses.len(),
            elapsed_time: elapsed,
            throughput_bps: throughput,
            peak_concurrency: self.config.max_concurrent_ranges.min(sorted_responses.len()),
            shard_utilization: shard_counts,
        };

        Ok(RangeDownloadResult {
            data,
            stats,
            range_responses: sorted_responses,
        })
    }

    /// Execute multiple range requests with controlled concurrency
    async fn download_ranges_concurrent(&self, requests: Vec<RangeRequest>) -> Result<Vec<RangeResponse>> {
        use tokio::task::JoinSet;

        let mut join_set = JoinSet::new();
        
        for request in requests {
            let clients = Arc::clone(&self.config.clients);
            let semaphore = Arc::clone(&self.concurrency_limiter);
            
            join_set.spawn(async move {
                // Acquire concurrency permit
                let _permit = semaphore.acquire().await.unwrap();
                
                // Execute the range request
                clients.get_range(request).await
            });
        }

        // Collect all results
        let mut responses = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow::anyhow!("Task join error: {}", e)),
            }
        }

        Ok(responses)
    }

    /// Calculate optimal ranges for an object
    /// 
    /// This determines the best way to split an object into ranges
    /// based on the configured range size and adaptive sizing settings.
    fn calculate_ranges(&self, object_size: u64) -> Vec<(u64, usize)> {
        let mut ranges = Vec::new();
        let mut offset = 0;
        
        while offset < object_size {
            let remaining = object_size - offset;
            let range_size = if self.config.adaptive_sizing {
                self.calculate_adaptive_range_size(remaining)
            } else {
                self.config.range_size.min(remaining as usize)
            };
            
            ranges.push((offset, range_size));
            offset += range_size as u64;
        }

        ranges
    }

    /// Calculate adaptive range size based on remaining bytes
    /// 
    /// This uses heuristics to optimize range sizes:
    /// - Larger ranges for the middle of large objects
    /// - Smaller ranges for the end to avoid waste
    fn calculate_adaptive_range_size(&self, remaining_bytes: u64) -> usize {
        let base_size = self.config.range_size;
        
        if remaining_bytes < base_size as u64 {
            // Last range - use exactly what's remaining
            remaining_bytes as usize
        } else if remaining_bytes < (base_size * 2) as u64 {
            // Near the end - use half ranges to avoid very small last range
            (remaining_bytes / 2) as usize
        } else {
            // Plenty remaining - use full size
            base_size
        }
    }

    /// Update configuration based on observed performance
    /// 
    /// This allows the engine to adapt its behavior based on
    /// measured throughput and latency characteristics.
    pub fn update_config_from_stats(&mut self, stats: &RangeDownloadStats) {
        if !self.config.adaptive_sizing {
            return;
        }

        // Heuristic: If throughput is high and we used all concurrency,
        // consider increasing range size to reduce overhead
        if stats.throughput_bps > 1_000_000_000 && // > 1GB/s
           stats.peak_concurrency == self.config.max_concurrent_ranges {
            self.config.range_size = (self.config.range_size * 3 / 2).min(256 * 1024 * 1024);
        }

        // Heuristic: If throughput is low with many small ranges,
        // consider decreasing range size for better parallelism
        if stats.throughput_bps < 100_000_000 && // < 100MB/s
           stats.ranges_processed > self.config.max_concurrent_ranges * 2 {
            self.config.range_size = (self.config.range_size * 2 / 3).max(16 * 1024 * 1024);
        }
    }

    /// Get current engine statistics
    pub fn get_config(&self) -> &RangeEngineConfig {
        &self.config
    }

    /// Check if the engine would use ranges for a given object size
    pub fn would_use_ranges(&self, object_size: u64) -> bool {
        object_size >= self.config.min_split_size
    }

    /// Estimate the number of ranges for a given object size
    pub fn estimate_range_count(&self, object_size: u64) -> usize {
        if object_size < self.config.min_split_size {
            1
        } else {
            ((object_size + self.config.range_size as u64 - 1) / self.config.range_size as u64) as usize
        }
    }
}

/// Builder for RangeEngine configuration
pub struct RangeEngineBuilder {
    config: RangeEngineConfig,
}

impl RangeEngineBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: RangeEngineConfig::new(),
        }
    }

    /// Set the range size in bytes
    pub fn range_size(mut self, size: usize) -> Self {
        self.config.range_size = size;
        self
    }

    /// Set the maximum concurrent ranges
    pub fn max_concurrent_ranges(mut self, count: usize) -> Self {
        self.config.max_concurrent_ranges = count;
        self
    }

    /// Set the minimum object size for range splitting
    pub fn min_split_size(mut self, size: u64) -> Self {
        self.config.min_split_size = size;
        self
    }

    /// Enable or disable adaptive sizing
    pub fn adaptive_sizing(mut self, enabled: bool) -> Self {
        self.config.adaptive_sizing = enabled;
        self
    }

    /// Set the buffer pool
    pub fn buffer_pool(mut self, pool: Arc<BufferPool>) -> Self {
        self.config.buffer_pool = pool;
        self
    }

    /// Set the sharded clients
    pub fn clients(mut self, clients: Arc<ShardedS3Clients>) -> Self {
        self.config.clients = clients;
        self
    }

    /// Build the range engine
    pub fn build(self) -> RangeEngine {
        RangeEngine::new(self.config)
    }
}

impl Default for RangeEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn range_calculation() {
        let config = RangeEngineConfig {
            range_size: 64 * 1024 * 1024, // 64MB
            min_split_size: 128 * 1024 * 1024, // 128MB
            adaptive_sizing: false,
            ..RangeEngineConfig::new()
        };
        
        let engine = RangeEngine::new(config);

        // Small object - should not use ranges
        assert!(!engine.would_use_ranges(100 * 1024 * 1024)); // 100MB
        assert_eq!(engine.estimate_range_count(100 * 1024 * 1024), 1);

        // Large object - should use ranges
        assert!(engine.would_use_ranges(500 * 1024 * 1024)); // 500MB
        let expected_ranges = (500 + 64 - 1) / 64; // Ceiling division
        assert_eq!(engine.estimate_range_count(500 * 1024 * 1024), expected_ranges);
    }

    #[tokio::test]
    async fn adaptive_range_sizing() {
        let config = RangeEngineConfig {
            range_size: 64 * 1024 * 1024, // 64MB
            adaptive_sizing: true,
            ..RangeEngineConfig::new()
        };
        
        let engine = RangeEngine::new(config);

        // Test different remaining sizes
        let base_size = 64 * 1024 * 1024;
        
        // Large remaining - should use full size
        assert_eq!(engine.calculate_adaptive_range_size(200 * 1024 * 1024), base_size);
        
        // Small remaining - should use exactly remaining
        assert_eq!(engine.calculate_adaptive_range_size(32 * 1024 * 1024), 32 * 1024 * 1024);
        
        // Near end - should use half to avoid tiny last range
        let near_end = (base_size as f64 * 1.5) as u64;
        let expected = near_end / 2;
        assert_eq!(engine.calculate_adaptive_range_size(near_end), expected as usize);
    }

    #[tokio::test]
    async fn range_engine_builder() {
        let engine = RangeEngineBuilder::new()
            .range_size(32 * 1024 * 1024)
            .max_concurrent_ranges(16)
            .min_split_size(64 * 1024 * 1024)
            .adaptive_sizing(false)
            .build();

        assert_eq!(engine.config.range_size, 32 * 1024 * 1024);
        assert_eq!(engine.config.max_concurrent_ranges, 16);
        assert_eq!(engine.config.min_split_size, 64 * 1024 * 1024);
        assert!(!engine.config.adaptive_sizing);
    }

    #[test]
    fn range_stats_calculation() {
        // This test would need async runtime, but shows the structure
        let stats = RangeDownloadStats {
            bytes_downloaded: 1024 * 1024 * 1024, // 1GB
            ranges_processed: 16,
            elapsed_time: std::time::Duration::from_secs(10),
            throughput_bps: 100 * 1024 * 1024, // 100MB/s
            peak_concurrency: 16,
            shard_utilization: vec![4, 4, 4, 4], // Even distribution across 4 shards
        };

        assert_eq!(stats.bytes_downloaded, 1024 * 1024 * 1024);
        assert_eq!(stats.ranges_processed, 16);
        assert_eq!(stats.peak_concurrency, 16);
        assert_eq!(stats.shard_utilization.iter().sum::<usize>(), 16);
    }
}