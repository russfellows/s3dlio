// src/concurrency/scheduler.rs
//
// Adaptive Concurrency Scheduler
// Based on AWS S3 Transfer Manager patterns for intelligent throughput optimization

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, TryAcquireError};
use anyhow::Result;

/// Target throughput in bytes per second
#[derive(Debug, Clone, Copy)]
pub struct Throughput {
    bytes_per_sec: u64,
}

impl Throughput {
    pub fn new_bytes_per_sec(bytes_per_sec: u64) -> Self {
        Self { bytes_per_sec }
    }
    
    pub fn new_mbps(mbps: u64) -> Self {
        Self { bytes_per_sec: mbps * 1_000_000 }
    }
    
    pub fn new_gbps(gbps: u64) -> Self {
        Self { bytes_per_sec: gbps * 1_000_000_000 }
    }
    
    pub fn bytes_per_sec(&self) -> u64 {
        self.bytes_per_sec
    }
    
    pub fn as_mbps(&self) -> f64 {
        self.bytes_per_sec as f64 / 1_000_000.0
    }
    
    pub fn as_gbps(&self) -> f64 {
        self.bytes_per_sec as f64 / 1_000_000_000.0
    }
}

/// S3 performance characteristics based on AWS Transfer Manager research
#[derive(Debug, Clone)]
pub struct S3PerformanceProfile {
    /// Maximum throughput per connection for downloads
    pub max_download_per_connection: Throughput,
    /// Maximum throughput per connection for uploads  
    pub max_upload_per_connection: Throughput,
    /// Estimated round-trip latency
    pub estimated_latency: Duration,
    /// Whether this endpoint supports HTTP/2
    pub supports_http2: bool,
}

impl Default for S3PerformanceProfile {
    fn default() -> Self {
        // Conservative defaults based on AWS Transfer Manager constants
        Self {
            max_download_per_connection: Throughput::new_mbps(90), // 90 MB/s per connection
            max_upload_per_connection: Throughput::new_mbps(20),   // 20 MB/s per connection  
            estimated_latency: Duration::from_millis(50),          // 50ms RTT
            supports_http2: false,                                  // Conservative default
        }
    }
}

impl S3PerformanceProfile {
    /// Create profile for high-performance S3-compatible storage
    pub fn high_performance() -> Self {
        Self {
            max_download_per_connection: Throughput::new_mbps(150),
            max_upload_per_connection: Throughput::new_mbps(100),
            estimated_latency: Duration::from_millis(25),
            supports_http2: true,
        }
    }
    
    /// Create profile for AWS S3 (based on Transfer Manager data)
    pub fn aws_s3() -> Self {
        Self {
            max_download_per_connection: Throughput::new_mbps(90),
            max_upload_per_connection: Throughput::new_mbps(20),
            estimated_latency: Duration::from_millis(100),
            supports_http2: false, // AWS S3 doesn't support HTTP/2
        }
    }
}

/// Transfer direction for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransferDirection {
    Upload,
    Download,
}

/// Concurrency configuration mode
#[derive(Debug, Clone)]
pub enum ConcurrencyMode {
    /// Fixed number of concurrent operations
    Explicit(usize),
    /// Target throughput - scheduler calculates optimal concurrency
    TargetThroughput(Throughput),
    /// Auto-detect based on system and S3 performance
    Auto,
}

/// Adaptive concurrency scheduler with throughput optimization
#[derive(Debug)]
pub struct AdaptiveScheduler {
    /// Performance profile for the target S3 endpoint
    profile: S3PerformanceProfile,
    /// Current concurrency mode
    mode: ConcurrencyMode,
    /// Semaphore for limiting concurrent operations
    semaphore: Arc<Semaphore>,
    /// Metrics for adaptive adjustment
    metrics: SchedulerMetrics,
    /// Current optimal concurrency (calculated)
    optimal_concurrency: AtomicUsize,
}

#[derive(Debug)]
pub struct SchedulerMetrics {
    /// Total bytes transferred
    total_bytes: AtomicU64,
    /// Total transfer time
    total_time_ms: AtomicU64,
    /// Number of completed operations
    completed_operations: AtomicU64,
    /// Current in-flight operations
    inflight_operations: AtomicUsize,
    /// Last throughput measurement
    last_throughput: AtomicU64,
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self {
            total_bytes: AtomicU64::new(0),
            total_time_ms: AtomicU64::new(0),
            completed_operations: AtomicU64::new(0),
            inflight_operations: AtomicUsize::new(0),
            last_throughput: AtomicU64::new(0),
        }
    }
}

impl SchedulerMetrics {
    /// Calculate current throughput in bytes per second
    pub fn current_throughput(&self) -> Throughput {
        let total_bytes = self.total_bytes.load(Ordering::Relaxed);
        let total_time_ms = self.total_time_ms.load(Ordering::Relaxed);
        
        if total_time_ms > 0 {
            let bytes_per_sec = (total_bytes * 1000) / total_time_ms;
            Throughput::new_bytes_per_sec(bytes_per_sec)
        } else {
            Throughput::new_bytes_per_sec(0)
        }
    }
    
    /// Get number of in-flight operations
    pub fn inflight(&self) -> usize {
        self.inflight_operations.load(Ordering::Relaxed)
    }
    
    /// Record completion of an operation
    pub fn record_completion(&self, bytes_transferred: u64, duration: Duration) {
        self.total_bytes.fetch_add(bytes_transferred, Ordering::Relaxed);
        self.total_time_ms.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        self.completed_operations.fetch_add(1, Ordering::Relaxed);
        
        // Update last throughput measurement
        let bytes_per_sec = if duration.as_millis() > 0 {
            (bytes_transferred * 1000) / duration.as_millis() as u64
        } else {
            bytes_transferred // Instantaneous
        };
        self.last_throughput.store(bytes_per_sec, Ordering::Relaxed);
    }
    
    /// Get average throughput across all operations
    pub fn average_throughput(&self) -> Throughput {
        self.current_throughput() // Same calculation for now
    }
    
    /// Get total number of operations completed
    pub fn total_operations(&self) -> u64 {
        self.completed_operations.load(Ordering::Relaxed)
    }
}

impl AdaptiveScheduler {
    /// Create new scheduler with performance profile
    pub fn new(mode: ConcurrencyMode, profile: S3PerformanceProfile) -> Self {
        let initial_permits = match &mode {
            ConcurrencyMode::Explicit(n) => *n,
            ConcurrencyMode::TargetThroughput(target) => {
                // Calculate initial concurrency based on target throughput
                Self::calculate_optimal_concurrency(target, &profile, TransferDirection::Download)
            },
            ConcurrencyMode::Auto => {
                // Start with conservative default, will adapt
                48 // Your proven sweet spot
            }
        };
        
        Self {
            profile,
            mode,
            semaphore: Arc::new(Semaphore::new(initial_permits)),
            metrics: SchedulerMetrics::default(),
            optimal_concurrency: AtomicUsize::new(initial_permits),
        }
    }
    
    /// Calculate optimal concurrency for target throughput
    fn calculate_optimal_concurrency(
        target: &Throughput,
        profile: &S3PerformanceProfile,
        direction: TransferDirection,
    ) -> usize {
        let per_connection_throughput = match direction {
            TransferDirection::Upload => profile.max_upload_per_connection,
            TransferDirection::Download => profile.max_download_per_connection,
        };
        
        // Calculate needed connections to achieve target throughput
        let needed_connections = target.bytes_per_sec() / per_connection_throughput.bytes_per_sec();
        
        // Apply reasonable bounds
        let min_concurrency = 8;
        let max_concurrency = 256;
        
        std::cmp::max(min_concurrency, std::cmp::min(max_concurrency, needed_connections as usize))
    }
    
    /// Acquire permit for operation
    pub async fn acquire_permit(&self) -> Result<SchedulerPermit<'_>> {
        // Try to acquire permit
        let permit = self.semaphore.clone().acquire_owned().await
            .map_err(|_| anyhow::anyhow!("Semaphore closed"))?;
        
        // Increment inflight counter
        self.metrics.inflight_operations.fetch_add(1, Ordering::Relaxed);
        
        Ok(SchedulerPermit {
            _permit: permit,
            scheduler: self,
            start_time: Instant::now(),
        })
    }
    
    /// Try to acquire permit without waiting
    pub fn try_acquire_permit(&self) -> Result<Option<SchedulerPermit<'_>>> {
        match self.semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                self.metrics.inflight_operations.fetch_add(1, Ordering::Relaxed);
                Ok(Some(SchedulerPermit {
                    _permit: permit,
                    scheduler: self,
                    start_time: Instant::now(),
                }))
            },
            Err(TryAcquireError::NoPermits) => Ok(None),
            Err(TryAcquireError::Closed) => Err(anyhow::anyhow!("Semaphore closed")),
        }
    }
    
    /// Get current metrics
    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }
    
    /// Get performance profile
    pub fn profile(&self) -> &S3PerformanceProfile {
        &self.profile
    }
    
    /// Adapt concurrency based on measured performance
    pub fn adapt_concurrency(&self) {
        if let ConcurrencyMode::Auto | ConcurrencyMode::TargetThroughput(_) = &self.mode {
            let current_throughput = self.metrics.current_throughput();
            let inflight = self.metrics.inflight();
            
            // Simple adaptation logic - can be made more sophisticated
            if current_throughput.bytes_per_sec() > 0 && inflight > 0 {
                let throughput_per_connection = current_throughput.bytes_per_sec() / inflight as u64;
                
                // If we're getting good throughput per connection, we might be able to add more
                // If throughput per connection is low, we might have too many connections
                
                let target_throughput = match &self.mode {
                    ConcurrencyMode::TargetThroughput(target) => target.bytes_per_sec(),
                    _ => current_throughput.bytes_per_sec() * 2, // Try to double current performance
                };
                
                let new_optimal = (target_throughput / throughput_per_connection) as usize;
                let new_optimal = std::cmp::max(8, std::cmp::min(256, new_optimal));
                
                self.optimal_concurrency.store(new_optimal, Ordering::Relaxed);
            }
        }
    }
    
    /// Update the concurrency mode dynamically
    pub fn update_mode(&mut self, new_mode: ConcurrencyMode) -> Result<()> {
        self.mode = new_mode;
        // Re-initialize semaphore if needed
        let new_concurrency = match &self.mode {
            ConcurrencyMode::Explicit(n) => *n,
            _ => self.optimal_concurrency.load(Ordering::Relaxed),
        };
        self.semaphore = Arc::new(tokio::sync::Semaphore::new(new_concurrency));
        Ok(())
    }
    
    /// Update the performance profile
    pub fn update_profile(&mut self, new_profile: S3PerformanceProfile) -> Result<()> {
        self.profile = new_profile;
        Ok(())
    }
    
    /// Get optimal part size for given object size
    pub fn optimal_part_size(&self, object_size: u64) -> u64 {
        // Use 10MB as default part size (your proven optimum)
        calculate_optimal_part_size(object_size, 10 * 1024 * 1024)
    }
    
    /// Get current concurrency level
    pub fn current_concurrency(&self) -> usize {
        match &self.mode {
            ConcurrencyMode::Explicit(n) => *n,
            _ => self.optimal_concurrency.load(Ordering::Relaxed),
        }
    }
    
    /// Get comprehensive statistics
    pub fn statistics(&self) -> SchedulerStatistics {
        let metrics = &self.metrics;
        SchedulerStatistics {
            current_concurrency: self.current_concurrency(),
            average_throughput: metrics.average_throughput(),
            total_operations: metrics.total_operations(),
            inflight_operations: metrics.inflight(),
        }
    }
}

/// Permit for performing an operation with the scheduler
pub struct SchedulerPermit<'a> {
    _permit: tokio::sync::OwnedSemaphorePermit,
    scheduler: &'a AdaptiveScheduler,
    start_time: Instant,
}

impl<'a> SchedulerPermit<'a> {
    /// Mark operation as completed with transfer stats
    pub fn complete(self, bytes_transferred: u64) {
        let duration = self.start_time.elapsed();
        self.scheduler.metrics.record_completion(bytes_transferred, duration);
        self.scheduler.metrics.inflight_operations.fetch_sub(1, Ordering::Relaxed);
        
        // Trigger adaptation
        self.scheduler.adapt_concurrency();
        
        // Permit is automatically dropped, releasing the semaphore slot
    }
}

impl<'a> Drop for SchedulerPermit<'a> {
    fn drop(&mut self) {
        // Ensure inflight counter is decremented even if complete() wasn't called
        self.scheduler.metrics.inflight_operations.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Statistics from the adaptive scheduler
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    pub current_concurrency: usize,
    pub average_throughput: Throughput,
    pub total_operations: u64,
    pub inflight_operations: usize,
}

/// Calculate optimal part size based on object size and S3 limits
pub fn calculate_optimal_part_size(object_size: u64, target_part_size: u64) -> u64 {
    const MAX_PARTS: u64 = 10_000;
    const MIN_PART_SIZE: u64 = 5 * 1024 * 1024; // 5 MiB minimum
    const MAX_PART_SIZE: u64 = 5 * 1024 * 1024 * 1024; // 5 GiB maximum
    
    // If object is small, use minimum part size
    if object_size <= target_part_size {
        return std::cmp::max(MIN_PART_SIZE, object_size);
    }
    
    // Calculate part size needed to stay under MAX_PARTS
    let required_part_size = object_size.div_ceil(MAX_PARTS);
    
    // Use the larger of target or required part size
    let optimal_size = std::cmp::max(target_part_size, required_part_size);
    
    // Ensure within bounds
    std::cmp::max(MIN_PART_SIZE, std::cmp::min(MAX_PART_SIZE, optimal_size))
}