// src/range_engine_generic.rs
//
// Universal stream-based range engine for concurrent downloads
// Works with ANY backend that implements async get_range(offset, length)

use bytes::Bytes;
use futures::stream::{self, StreamExt};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use anyhow::{Result, bail};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::constants::{
    DEFAULT_RANGE_ENGINE_CHUNK_SIZE,
    DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,
    DEFAULT_FILE_RANGE_ENGINE_THRESHOLD,
    DEFAULT_RANGE_TIMEOUT_SECS,
};

/// Configuration for range-based concurrent downloads
/// 
/// **Performance Considerations**:
/// - **Local File Systems**: Range parallelism may be **slower** due to:
///   - Seek overhead (random access vs sequential)
///   - Disk I/O contention
///   - Page cache already optimizes sequential reads
///   - Consider disabling or using higher thresholds (16-64MB) for file:// URIs
/// 
/// - **Network Storage (S3/Azure/GCS)**: Benefits significantly from range parallelism:
///   - Hides network latency with concurrent requests
///   - 30-50% throughput improvement for large files
///   - Lower thresholds (4MB) work well
/// 
/// - **DirectIO**: Limited benefit since O_DIRECT already bypasses page cache
///   - Higher threshold (16MB) recommended due to alignment overhead
///   - Lower concurrency (16) to avoid excessive parallel seeks
#[derive(Debug, Clone)]
pub struct RangeEngineConfig {
    /// Size of each range chunk in bytes (default: 64MB)
    pub chunk_size: usize,
    
    /// Maximum concurrent range requests (default: 32)
    pub max_concurrent_ranges: usize,
    
    /// Minimum object size to trigger range splitting (default: 4MB)
    /// Objects smaller than this use simple single-request downloads
    /// **WARNING**: For local filesystems, consider higher thresholds or disable entirely
    pub min_split_size: u64,
    
    /// Timeout per range request (default: 30s)
    pub range_timeout: Duration,
}

impl Default for RangeEngineConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE,
            max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,
            min_split_size: DEFAULT_FILE_RANGE_ENGINE_THRESHOLD,
            range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS),
        }
    }
}

/// Statistics collected during range download
#[derive(Debug, Clone)]
pub struct RangeDownloadStats {
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
    
    /// Number of range requests made
    pub ranges_processed: usize,
    
    /// Total elapsed time
    pub elapsed_time: Duration,
    
    /// Average throughput in bytes per second
    pub throughput_bps: u64,
}

impl RangeDownloadStats {
    /// Throughput in megabytes per second
    pub fn throughput_mbps(&self) -> f64 {
        (self.throughput_bps as f64) / (1024.0 * 1024.0)
    }
    
    /// Throughput in gigabits per second
    pub fn throughput_gbps(&self) -> f64 {
        (self.throughput_bps as f64 * 8.0) / (1_000_000_000.0)
    }
}

/// Universal range-based download engine
/// 
/// This engine provides high-performance concurrent downloads for ANY backend
/// that implements async `get_range(offset, length)`. It uses:
/// 
/// - Stream-based architecture with `stream::iter().buffered()`
/// - Controlled concurrency via semaphore
/// - Cancellation token support for clean shutdown
/// - Timeout per range request
/// - Ordered reassembly of chunks
/// 
/// # Example
/// 
/// ```no_run
/// use s3dlio::range_engine_generic::{RangeEngine, RangeEngineConfig};
/// 
/// # async fn example() -> anyhow::Result<()> {
/// let engine = RangeEngine::new(RangeEngineConfig::default());
/// 
/// // Works with ANY async get_range function
/// let get_range = |offset, length| async move {
///     // Your backend's get_range implementation
///     my_backend.get_range(offset, length).await
/// };
/// 
/// let (bytes, stats) = engine.download(file_size, get_range, None).await?;
/// println!("Downloaded {} bytes in {} ranges at {} MB/s",
///     stats.bytes_downloaded, stats.ranges_processed, stats.throughput_mbps());
/// # Ok(())
/// # }
/// ```
pub struct RangeEngine {
    config: RangeEngineConfig,
    concurrency_limiter: Arc<Semaphore>,
}

impl RangeEngine {
    /// Create a new range engine with the given configuration
    pub fn new(config: RangeEngineConfig) -> Self {
        let concurrency_limiter = Arc::new(Semaphore::new(config.max_concurrent_ranges));
        Self { config, concurrency_limiter }
    }
    
    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RangeEngineConfig::default())
    }

    /// Download object using concurrent range requests
    /// 
    /// This method automatically decides the optimal download strategy:
    /// - Small objects (< min_split_size): Single request
    /// - Large objects: Concurrent range requests with streaming
    /// 
    /// # Arguments
    /// 
    /// * `object_size` - Total size of the object in bytes
    /// * `get_range` - Async function that fetches a range: `fn(offset, length) -> Future<Result<Bytes>>`
    /// * `cancel` - Optional cancellation token for clean shutdown
    /// 
    /// # Returns
    /// 
    /// Tuple of (downloaded bytes, statistics)
    pub async fn download<F, Fut>(
        &self,
        object_size: u64,
        get_range: F,
        cancel: Option<CancellationToken>,
    ) -> Result<(Bytes, RangeDownloadStats)>
    where
        F: Fn(u64, u64) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Bytes>> + Send,
    {
        if object_size == 0 {
            bail!("Cannot download zero-sized object");
        }
        
        let start_time = Instant::now();
        
        // Small objects: use single request (no overhead from range splitting)
        if object_size < self.config.min_split_size {
            tracing::debug!(
                "Object size {} < threshold {}, using single request",
                object_size, self.config.min_split_size
            );
            return self.download_single(object_size, get_range, start_time).await;
        }
        
        // Large objects: use concurrent range requests with streams
        tracing::debug!(
            "Object size {} >= threshold {}, using concurrent ranges",
            object_size, self.config.min_split_size
        );
        self.download_with_ranges(object_size, get_range, cancel, start_time).await
    }

    /// Download using a single range request (for small objects)
    async fn download_single<F, Fut>(
        &self,
        object_size: u64,
        get_range: F,
        start_time: Instant,
    ) -> Result<(Bytes, RangeDownloadStats)>
    where
        F: Fn(u64, u64) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<Bytes>> + Send,
    {
        // Fetch entire object as single range
        let bytes = tokio::time::timeout(
            self.config.range_timeout,
            get_range(0, object_size)
        )
        .await
        .map_err(|_| anyhow::anyhow!("Single request timeout after {:?}", self.config.range_timeout))?
        .map_err(|e| anyhow::anyhow!("Single request failed: {}", e))?;
        
        let bytes_downloaded = bytes.len() as u64;
        let elapsed = start_time.elapsed();
        
        let stats = RangeDownloadStats {
            bytes_downloaded,
            ranges_processed: 1,
            elapsed_time: elapsed,
            throughput_bps: Self::calculate_throughput(bytes_downloaded, elapsed),
        };
        
        Ok((bytes, stats))
    }

    /// Download using concurrent range requests (stream-based for large objects)
    async fn download_with_ranges<F, Fut>(
        &self,
        object_size: u64,
        get_range: F,
        cancel: Option<CancellationToken>,
        start_time: Instant,
    ) -> Result<(Bytes, RangeDownloadStats)>
    where
        F: Fn(u64, u64) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Bytes>> + Send,
    {
        // Calculate optimal range splits
        let ranges = self.calculate_ranges(object_size);
        let n_ranges = ranges.len();
        
        tracing::debug!(
            "Splitting {} bytes into {} ranges of ~{} bytes each",
            object_size, n_ranges, self.config.chunk_size
        );
        
        let semaphore = Arc::clone(&self.concurrency_limiter);
        let timeout = self.config.range_timeout;
        
        // Create stream of concurrent range requests
        // This is the key pattern: stream::iter().map().buffered()
        let mut chunks = stream::iter(ranges)
            .enumerate()
            .map(|(idx, (offset, length))| {
                let get_range = get_range.clone();
                let semaphore = Arc::clone(&semaphore);
                let cancel = cancel.clone();
                
                async move {
                    // Check cancellation before starting
                    if let Some(ref token) = cancel {
                        if token.is_cancelled() {
                            return Err(anyhow::anyhow!("Download cancelled by user"));
                        }
                    }
                    
                    // Acquire concurrency permit (backpressure control)
                    let _permit = semaphore.acquire().await
                        .map_err(|e| anyhow::anyhow!("Semaphore acquisition failed: {}", e))?;
                    
                    tracing::trace!("Fetching range {}: offset={}, length={}", idx, offset, length);
                    
                    // Execute range request with timeout
                    let bytes = tokio::time::timeout(timeout, get_range(offset, length))
                        .await
                        .map_err(|_| anyhow::anyhow!(
                            "Range {} timeout after {:?} (offset={}, length={})",
                            idx, timeout, offset, length
                        ))?
                        .map_err(|e| anyhow::anyhow!(
                            "Range {} request failed (offset={}, length={}): {}",
                            idx, offset, length, e
                        ))?;
                    
                    // Verify we got the expected amount of data
                    if bytes.len() != length as usize {
                        tracing::warn!(
                            "Range {} returned {} bytes, expected {} (offset={}, last_range={})",
                            idx, bytes.len(), length, offset, idx == n_ranges - 1
                        );
                    }
                    
                    Ok((idx, bytes))
                }
            })
            .buffered(self.config.max_concurrent_ranges);
        
        // Collect results with ordered reassembly
        let mut parts: Vec<(usize, Bytes)> = Vec::with_capacity(n_ranges);
        while let Some(result) = chunks.next().await {
            let (idx, bytes) = result?;
            parts.push((idx, bytes));
        }
        
        // Sort by index to ensure correct order
        // (buffered() doesn't guarantee output order)
        parts.sort_by_key(|(idx, _)| *idx);
        
        // Assemble final buffer
        let total_size: usize = parts.iter().map(|(_, b)| b.len()).sum();
        let mut assembled = Vec::with_capacity(total_size);
        
        for (idx, bytes) in parts {
            tracing::trace!("Assembling range {} ({} bytes)", idx, bytes.len());
            assembled.extend_from_slice(&bytes);
        }
        
        let bytes_downloaded = assembled.len() as u64;
        let elapsed = start_time.elapsed();
        
        let stats = RangeDownloadStats {
            bytes_downloaded,
            ranges_processed: n_ranges,
            elapsed_time: elapsed,
            throughput_bps: Self::calculate_throughput(bytes_downloaded, elapsed),
        };
        
        tracing::info!(
            "Downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );
        
        Ok((Bytes::from(assembled), stats))
    }

    /// Calculate optimal range splits for an object
    /// 
    /// Splits the object into chunks of approximately chunk_size,
    /// with the last chunk possibly being smaller.
    fn calculate_ranges(&self, object_size: u64) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        let mut offset = 0u64;
        let chunk_size = self.config.chunk_size as u64;
        
        while offset < object_size {
            let remaining = object_size - offset;
            let length = remaining.min(chunk_size);
            ranges.push((offset, length));
            offset += length;
        }
        
        ranges
    }

    /// Calculate throughput in bytes per second
    fn calculate_throughput(bytes: u64, elapsed: Duration) -> u64 {
        let elapsed_secs = elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            (bytes as f64 / elapsed_secs) as u64
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    #[tokio::test]
    async fn test_small_object_single_request() {
        let engine = RangeEngine::with_defaults();
        
        // 1MB object (< 4MB threshold)
        let object_size = 1024 * 1024;
        let data = vec![0u8; object_size as usize];
        
        let get_range = move |offset: u64, length: u64| {
            let data = data.clone();
            async move {
                Ok(Bytes::from(data[offset as usize..(offset + length) as usize].to_vec()))
            }
        };
        
        let (bytes, stats) = engine.download(object_size, get_range, None).await.unwrap();
        
        assert_eq!(bytes.len(), object_size as usize);
        assert_eq!(stats.ranges_processed, 1);
        assert_eq!(stats.bytes_downloaded, object_size);
    }
    
    #[tokio::test]
    async fn test_large_object_concurrent_ranges() {
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024 * 1024,  // 1MB chunks
            max_concurrent_ranges: 4,
            min_split_size: 2 * 1024 * 1024,  // 2MB threshold
            range_timeout: Duration::from_secs(5),
        });
        
        // 10MB object (> 2MB threshold)
        let object_size = 10 * 1024 * 1024;
        let data = (0..object_size).map(|i| (i % 256) as u8).collect::<Vec<_>>();
        
        let concurrent_count = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        
        let get_range = {
            let data = data.clone();
            let concurrent_count = Arc::clone(&concurrent_count);
            let max_concurrent = Arc::clone(&max_concurrent);
            
            move |offset: u64, length: u64| {
                let data = data.clone();
                let concurrent_count = Arc::clone(&concurrent_count);
                let max_concurrent = Arc::clone(&max_concurrent);
                
                async move {
                    // Track concurrency
                    let current = concurrent_count.fetch_add(1, Ordering::SeqCst) + 1;
                    max_concurrent.fetch_max(current, Ordering::SeqCst);
                    
                    // Simulate network delay
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    
                    let result = Bytes::from(
                        data[offset as usize..(offset + length) as usize].to_vec()
                    );
                    
                    concurrent_count.fetch_sub(1, Ordering::SeqCst);
                    Ok(result)
                }
            }
        };
        
        let (bytes, stats) = engine.download(object_size, get_range, None).await.unwrap();
        
        // Verify correctness
        assert_eq!(bytes.len(), object_size as usize);
        assert_eq!(stats.bytes_downloaded, object_size);
        assert_eq!(stats.ranges_processed, 10); // 10 x 1MB chunks
        
        // Verify concurrency occurred
        let max_concurrent_seen = max_concurrent.load(Ordering::SeqCst);
        assert!(max_concurrent_seen > 1, "Expected concurrent execution, got max_concurrent={}", max_concurrent_seen);
        assert!(max_concurrent_seen <= 4, "Should not exceed max_concurrent_ranges");
        
        // Verify data integrity
        for (i, &byte) in bytes.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data mismatch at byte {}", i);
        }
    }
    
    #[tokio::test]
    async fn test_cancellation() {
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024 * 1024,
            max_concurrent_ranges: 4,
            min_split_size: 2 * 1024 * 1024,
            range_timeout: Duration::from_secs(5),
        });
        
        let object_size = 10 * 1024 * 1024;
        let cancel_token = CancellationToken::new();
        
        let get_range = {
            let cancel_token = cancel_token.clone();
            move |_offset: u64, _length: u64| {
                let cancel_token = cancel_token.clone();
                async move {
                    // Cancel after first request
                    cancel_token.cancel();
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(Bytes::from(vec![0u8; 1024 * 1024]))
                }
            }
        };
        
        let result = engine.download(object_size, get_range, Some(cancel_token.clone())).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }
    
    #[tokio::test]
    async fn test_timeout() {
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024 * 1024,
            max_concurrent_ranges: 2,
            min_split_size: 2 * 1024 * 1024,
            range_timeout: Duration::from_millis(100),  // Very short timeout
        });
        
        let object_size = 5 * 1024 * 1024;
        
        let get_range = |_offset: u64, _length: u64| async move {
            // Simulate slow request (exceeds timeout)
            tokio::time::sleep(Duration::from_secs(1)).await;
            Ok(Bytes::from(vec![0u8; 1024 * 1024]))
        };
        
        let result = engine.download(object_size, get_range, None).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timeout"));
    }
}
