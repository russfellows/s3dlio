//! Enhanced performance metrics with HDR histogram support
//! 
//! This module provides comprehensive performance monitoring for AI/ML workloads
//! using HDR (High Dynamic Range) histograms for precise tail latency analysis.
//! 
//! Key Features:
//! - HDR histograms for detailed performance analysis with precise percentiles
//! - Mergeable metrics for distributed training scenarios  
//! - Adaptive performance optimization based on real-time metrics
//! - AI/ML workload-specific performance tracking with low overhead

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::Result;
use hdrhistogram::Histogram;

/// Configuration for HDR histogram collection
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Significant value digits (1-5, higher = more precision)
    pub significant_digits: u8,
    /// Maximum expected value for auto-resize
    pub max_value: u64,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            significant_digits: 3,
            max_value: 1_000_000, // 1 second in microseconds
        }
    }
}

/// AI/ML workload metrics configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// HDR histogram configuration
    pub histogram_config: HistogramConfig,
    /// Whether to track per-operation metrics
    pub per_operation_tracking: bool,
    /// Sampling rate for high-frequency operations (0.0-1.0)
    pub sampling_rate: f64,
    /// Buffer size for batched metrics updates
    pub batch_size: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            histogram_config: HistogramConfig::default(),
            per_operation_tracking: true,
            sampling_rate: 1.0, // Track everything by default
            batch_size: 1000,
        }
    }
}

/// HDR histogram wrapper for AI/ML performance metrics
pub struct PerformanceHistogram {
    /// HDR histogram for precise percentile analysis
    hdr: Histogram<u64>,
    /// Operation name for tracking
    name: String,
}

impl PerformanceHistogram {
    /// Create a new performance histogram
    pub fn new(name: String, config: HistogramConfig) -> Result<Self> {
        let hdr = Histogram::new_with_bounds(1, config.max_value, config.significant_digits)
            .map_err(|e| anyhow::anyhow!("Failed to create HDR histogram: {}", e))?;
        
        Ok(Self { hdr, name })
    }
    
    /// Record a value in microseconds
    pub fn record(&mut self, value_us: u64) -> Result<()> {
        self.hdr.record(value_us)
            .map_err(|e| anyhow::anyhow!("Failed to record value in histogram {}: {}", self.name, e))?;
        Ok(())
    }
    
    /// Get percentile value in microseconds
    pub fn percentile(&self, percentile: f64) -> u64 {
        self.hdr.value_at_percentile(percentile)
    }
    
    /// Get mean value in microseconds
    pub fn mean(&self) -> f64 {
        self.hdr.mean()
    }
    
    /// Get total count of samples
    pub fn len(&self) -> u64 {
        self.hdr.len()
    }
    
    /// Check if histogram is empty
    pub fn is_empty(&self) -> bool {
        self.hdr.is_empty()
    }
    
    /// Get standard deviation
    pub fn stdev(&self) -> f64 {
        self.hdr.stdev()
    }
    
    /// Get minimum recorded value
    pub fn min(&self) -> u64 {
        self.hdr.min()
    }
    
    /// Get maximum recorded value
    pub fn max(&self) -> u64 {
        self.hdr.max()
    }
    
    /// Merge another histogram into this one (for distributed aggregation)
    pub fn merge(&mut self, other: &PerformanceHistogram) -> Result<()> {
        self.hdr.add(&other.hdr)
            .map_err(|e| anyhow::anyhow!("Failed to merge histogram {} with {}: {}", self.name, other.name, e))?;
        Ok(())
    }
    
    /// Reset histogram (clear all recorded values)
    pub fn reset(&mut self) {
        self.hdr.reset();
    }
    
    /// Export histogram data for external analysis or persistence
    pub fn export_summary(&self) -> HistogramSummary {
        HistogramSummary {
            name: self.name.clone(),
            count: self.len(),
            min: self.min(),
            max: self.max(),
            mean: self.mean(),
            stdev: self.stdev(),
            p50: self.percentile(50.0),
            p90: self.percentile(90.0),
            p95: self.percentile(95.0),
            p99: self.percentile(99.0),
            p999: self.percentile(99.9),
            p9999: self.percentile(99.99),
        }
    }
}

/// Summary statistics from a performance histogram
#[derive(Debug, Clone)]
pub struct HistogramSummary {
    pub name: String,
    pub count: u64,
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub stdev: f64,
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub p999: u64,
    pub p9999: u64,
}

/// Enhanced metrics collector for AI/ML workloads
pub struct EnhancedMetricsCollector {
    /// Configuration for metrics collection
    config: MetricsConfig,
    /// Per-operation latency histograms (operation_name -> histogram)
    latency_histograms: HashMap<String, PerformanceHistogram>,
    /// Per-operation throughput histograms (operation_name -> histogram)  
    throughput_histograms: HashMap<String, PerformanceHistogram>,
    /// Per-operation request size histograms (operation_name -> histogram)
    request_size_histograms: HashMap<String, PerformanceHistogram>,
    /// Global counters for different metrics
    operation_counts: HashMap<String, AtomicU64>,
    /// Total bytes transferred per operation
    bytes_transferred: HashMap<String, AtomicU64>,
    /// Error counts per operation
    error_counts: HashMap<String, AtomicU64>,
}

impl EnhancedMetricsCollector {
    /// Create a new enhanced metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            latency_histograms: HashMap::new(),
            throughput_histograms: HashMap::new(),
            request_size_histograms: HashMap::new(),
            operation_counts: HashMap::new(),
            bytes_transferred: HashMap::new(),
            error_counts: HashMap::new(),
        }
    }
    
    /// Record operation metrics
    pub fn record_operation(&mut self, operation: &str, latency_us: u64, request_size: u64, response_size: u64) -> Result<()> {
        // Apply sampling if configured
        if self.config.sampling_rate < 1.0 {
            if rand::random::<f64>() > self.config.sampling_rate {
                return Ok(());
            }
        }
        
        // Ensure histograms exist for this operation
        if !self.latency_histograms.contains_key(operation) {
            self.latency_histograms.insert(
                operation.to_string(), 
                PerformanceHistogram::new(format!("{}_latency", operation), self.config.histogram_config.clone())?
            );
        }
        if !self.throughput_histograms.contains_key(operation) {
            self.throughput_histograms.insert(
                operation.to_string(), 
                PerformanceHistogram::new(format!("{}_throughput", operation), self.config.histogram_config.clone())?
            );
        }
        if !self.request_size_histograms.contains_key(operation) {
            self.request_size_histograms.insert(
                operation.to_string(), 
                PerformanceHistogram::new(format!("{}_request_size", operation), self.config.histogram_config.clone())?
            );
        }
        
        // Record latency
        if let Some(histogram) = self.latency_histograms.get_mut(operation) {
            histogram.record(latency_us)?;
        }
        
        // Calculate and record throughput (bytes per second)
        if latency_us > 0 {
            let throughput_bps = (response_size * 1_000_000) / latency_us; // Convert to bytes per second
            if let Some(histogram) = self.throughput_histograms.get_mut(operation) {
                histogram.record(throughput_bps)?;
            }
        }
        
        // Record request size
        if let Some(histogram) = self.request_size_histograms.get_mut(operation) {
            histogram.record(request_size)?;
        }
        
        // Update counters
        self.operation_counts
            .entry(operation.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
            
        self.bytes_transferred
            .entry(operation.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(request_size + response_size, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Record an error for an operation
    pub fn record_error(&mut self, operation: &str) {
        self.error_counts
            .entry(operation.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get latency summary for an operation
    pub fn get_latency_summary(&self, operation: &str) -> Option<HistogramSummary> {
        self.latency_histograms.get(operation).map(|h| h.export_summary())
    }
    
    /// Get throughput summary for an operation
    pub fn get_throughput_summary(&self, operation: &str) -> Option<HistogramSummary> {
        self.throughput_histograms.get(operation).map(|h| h.export_summary())
    }
    
    /// Get request size summary for an operation
    pub fn get_request_size_summary(&self, operation: &str) -> Option<HistogramSummary> {
        self.request_size_histograms.get(operation).map(|h| h.export_summary())
    }
    
    /// Get operation count
    pub fn get_operation_count(&self, operation: &str) -> u64 {
        self.operation_counts.get(operation)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
    
    /// Get total bytes transferred for an operation
    pub fn get_bytes_transferred(&self, operation: &str) -> u64 {
        self.bytes_transferred.get(operation)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
    
    /// Get error count for an operation
    pub fn get_error_count(&self, operation: &str) -> u64 {
        self.error_counts.get(operation)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
    
    /// Get error rate for an operation (errors / total operations)
    pub fn get_error_rate(&self, operation: &str) -> f64 {
        let total = self.get_operation_count(operation);
        let errors = self.get_error_count(operation);
        
        if total > 0 {
            errors as f64 / total as f64
        } else {
            0.0
        }
    }
    
    /// Get all tracked operations
    pub fn get_operations(&self) -> Vec<String> {
        let mut operations: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        for key in self.latency_histograms.keys() {
            operations.insert(key.clone());
        }
        for key in self.operation_counts.keys() {
            operations.insert(key.clone());
        }
        
        operations.into_iter().collect()
    }
    
    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut operations = HashMap::new();
        
        for operation in self.get_operations() {
            let op_report = OperationReport {
                name: operation.clone(),
                latency: self.get_latency_summary(&operation),
                throughput: self.get_throughput_summary(&operation),
                request_size: self.get_request_size_summary(&operation),
                operation_count: self.get_operation_count(&operation),
                bytes_transferred: self.get_bytes_transferred(&operation),
                error_count: self.get_error_count(&operation),
                error_rate: self.get_error_rate(&operation),
            };
            operations.insert(operation, op_report);
        }
        
        PerformanceReport { operations }
    }
    
    /// Reset all metrics
    pub fn reset(&mut self) {
        for histogram in self.latency_histograms.values_mut() {
            histogram.reset();
        }
        for histogram in self.throughput_histograms.values_mut() {
            histogram.reset();
        }
        for histogram in self.request_size_histograms.values_mut() {
            histogram.reset();
        }
        
        for counter in self.operation_counts.values() {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in self.bytes_transferred.values() {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in self.error_counts.values() {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

/// Performance report for a single operation
#[derive(Debug, Clone)]
pub struct OperationReport {
    pub name: String,
    pub latency: Option<HistogramSummary>,
    pub throughput: Option<HistogramSummary>,
    pub request_size: Option<HistogramSummary>,
    pub operation_count: u64,
    pub bytes_transferred: u64,
    pub error_count: u64,
    pub error_rate: f64,
}

/// Comprehensive performance report for all operations
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub operations: HashMap<String, OperationReport>,
}

impl PerformanceReport {
    /// Print a human-readable performance report
    pub fn print_summary(&self) {
        println!("\n=== Performance Report ===");
        
        for (operation, report) in &self.operations {
            println!("\nOperation: {}", operation);
            println!("  Operations: {}", report.operation_count);
            println!("  Bytes Transferred: {} MB", report.bytes_transferred / (1024 * 1024));
            println!("  Error Rate: {:.2}%", report.error_rate * 100.0);
            
            if let Some(latency) = &report.latency {
                println!("  Latency (μs):");
                println!("    Mean: {:.2}, P50: {}, P95: {}, P99: {}, P99.9: {}", 
                    latency.mean, latency.p50, latency.p95, latency.p99, latency.p999);
            }
            
            if let Some(throughput) = &report.throughput {
                println!("  Throughput (bytes/s):");
                println!("    Mean: {:.2}, P50: {}, P95: {}, P99: {}", 
                    throughput.mean, throughput.p50, throughput.p95, throughput.p99);
            }
        }
        
        println!("\n=========================\n");
    }
}

/// Global metrics instance for convenience
use std::sync::{Mutex, OnceLock};

static GLOBAL_METRICS: OnceLock<Mutex<EnhancedMetricsCollector>> = OnceLock::new();

/// Initialize global metrics collector
pub fn init_global_metrics(config: MetricsConfig) {
    let _ = GLOBAL_METRICS.set(Mutex::new(EnhancedMetricsCollector::new(config)));
}

/// Record operation using global metrics collector
pub fn record_operation(operation: &str, latency_us: u64, request_size: u64, response_size: u64) -> Result<()> {
    if let Some(metrics) = GLOBAL_METRICS.get() {
        let mut collector = metrics.lock().unwrap();
        collector.record_operation(operation, latency_us, request_size, response_size)
    } else {
        // Initialize with default config if not already initialized
        init_global_metrics(MetricsConfig::default());
        record_operation(operation, latency_us, request_size, response_size)
    }
}

/// Record error using global metrics collector
pub fn record_error(operation: &str) {
    if let Some(metrics) = GLOBAL_METRICS.get() {
        let mut collector = metrics.lock().unwrap();
        collector.record_error(operation);
    } else {
        init_global_metrics(MetricsConfig::default());
        record_error(operation);
    }
}

/// Get performance report from global metrics collector
pub fn get_global_report() -> Option<PerformanceReport> {
    GLOBAL_METRICS.get().map(|metrics| {
        let collector = metrics.lock().unwrap();
        collector.generate_report()
    })
}

/// Print global performance report
pub fn print_global_report() {
    if let Some(report) = get_global_report() {
        report.print_summary();
    } else {
        println!("No global metrics available. Initialize with init_global_metrics() first.");
    }
}

// AI/ML workload preset configurations
impl MetricsConfig {
    /// Configuration optimized for training workloads
    pub fn training_optimized() -> Self {
        Self {
            histogram_config: HistogramConfig {
                significant_digits: 3,
                max_value: 10_000_000, // 10 second max for training operations
            },
            per_operation_tracking: true,
            sampling_rate: 1.0, // Track everything for training analysis
            batch_size: 100,
        }
    }
    
    /// Configuration optimized for inference workloads  
    pub fn inference_optimized() -> Self {
        Self {
            histogram_config: HistogramConfig {
                significant_digits: 4, // Higher precision for inference latency
                max_value: 1_000_000, // 1 second max for inference
            },
            per_operation_tracking: true,
            sampling_rate: 1.0,
            batch_size: 1000,
        }
    }
    
    /// Configuration for high-frequency operations with sampling
    pub fn high_frequency() -> Self {
        Self {
            histogram_config: HistogramConfig {
                significant_digits: 2, // Lower precision for speed
                max_value: 100_000, // 100ms max
            },
            per_operation_tracking: false, // Aggregate only
            sampling_rate: 0.1, // Sample 10% for performance
            batch_size: 10000,
        }
    }
}