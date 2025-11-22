//! Multi-endpoint storage support with configurable load balancing and per-endpoint thread/process control.
//!
//! This module provides a wrapper around multiple ObjectStore instances, enabling:
//! - Load balancing across multiple storage endpoints (round-robin, least-connections)
//! - Per-endpoint statistics tracking (requests, bytes, errors, latency)
//! - Configurable thread counts per endpoint for optimal scaling
//! - Process affinity hints for multi-process scenarios
//!
//! # Examples
//!
//! ## Simple usage with round-robin:
//! ```no_run
//! use s3dlio::multi_endpoint::{MultiEndpointStore, LoadBalanceStrategy};
//!
//! let uris = vec![
//!     "s3://endpoint1:9000/bucket/".to_string(),
//!     "s3://endpoint2:9000/bucket/".to_string(),
//!     "s3://endpoint3:9000/bucket/".to_string(),
//! ];
//! let store = MultiEndpointStore::new(uris, LoadBalanceStrategy::RoundRobin, None)?;
//! ```
//!
//! ## Advanced usage with per-endpoint thread configuration:
//! ```no_run
//! use s3dlio::multi_endpoint::{MultiEndpointStore, MultiEndpointStoreConfig, EndpointConfig, LoadBalanceStrategy};
//!
//! let config = MultiEndpointStoreConfig {
//!     endpoints: vec![
//!         EndpointConfig {
//!             uri: "s3://fast-ssd:9000/bucket/".to_string(),
//!             thread_count: Some(32),  // High-performance endpoint gets more threads
//!             process_affinity: Some(0),
//!         },
//!         EndpointConfig {
//!             uri: "s3://slow-hdd:9000/bucket/".to_string(),
//!             thread_count: Some(8),   // Slower endpoint gets fewer threads
//!             process_affinity: Some(1),
//!         },
//!     ],
//!     strategy: LoadBalanceStrategy::LeastConnections,
//!     default_thread_count: Some(16),  // Fallback for endpoints without explicit config
//! };
//! let store = MultiEndpointStore::from_config(config)?;
//! ```

use crate::object_store::{ObjectStore, store_for_uri, ObjectMetadata};
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use bytes::Bytes;

/// Load balancing strategy for distributing requests across endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Simple round-robin: cycle through endpoints sequentially.
    /// Low overhead, good for uniform endpoints with similar performance.
    RoundRobin,
    
    /// Least connections: route to endpoint with fewest active requests.
    /// Better for heterogeneous endpoints or when request latencies vary.
    /// Slightly higher overhead due to active connection tracking.
    LeastConnections,
}

/// Configuration for a single storage endpoint.
#[derive(Debug, Clone)]
pub struct EndpointConfig {
    /// Storage URI (e.g., "s3://host:9000/bucket/", "file:///mnt/storage1/")
    pub uri: String,
    
    /// Optional thread count override for this endpoint.
    /// If None, uses the default_thread_count from MultiEndpointStoreConfig.
    /// This allows dedicating more threads to faster endpoints or
    /// limiting threads for slower/rate-limited endpoints.
    pub thread_count: Option<usize>,
    
    /// Optional process affinity hint for multi-process scenarios.
    /// When running multiple processes, this hints which process should
    /// primarily handle this endpoint. Useful for:
    /// - NUMA optimization (process on same NUMA node as storage)
    /// - CPU pinning (dedicate cores to specific endpoints)
    /// - Resource isolation (separate processes for different storage tiers)
    pub process_affinity: Option<usize>,
}

impl EndpointConfig {
    /// Create a simple endpoint config with just a URI (uses defaults for thread/process settings).
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            thread_count: None,
            process_affinity: None,
        }
    }
    
    /// Create an endpoint config with custom thread count.
    pub fn with_threads(uri: impl Into<String>, thread_count: usize) -> Self {
        Self {
            uri: uri.into(),
            thread_count: Some(thread_count),
            process_affinity: None,
        }
    }
    
    /// Create an endpoint config with process affinity hint.
    pub fn with_process_affinity(uri: impl Into<String>, process_id: usize) -> Self {
        Self {
            uri: uri.into(),
            thread_count: None,
            process_affinity: Some(process_id),
        }
    }
}

/// Configuration for multi-endpoint store with global defaults and per-endpoint overrides.
#[derive(Debug, Clone)]
pub struct MultiEndpointStoreConfig {
    /// List of storage endpoints with optional per-endpoint settings.
    pub endpoints: Vec<EndpointConfig>,
    
    /// Load balancing strategy to use.
    pub strategy: LoadBalanceStrategy,
    
    /// Default thread count for endpoints that don't specify thread_count.
    /// If None, lets the underlying ObjectStore implementation choose.
    pub default_thread_count: Option<usize>,
}

impl MultiEndpointStoreConfig {
    /// Create a simple config from URIs with round-robin strategy and default threading.
    pub fn from_uris(uris: Vec<String>) -> Self {
        Self {
            endpoints: uris.into_iter().map(EndpointConfig::new).collect(),
            strategy: LoadBalanceStrategy::RoundRobin,
            default_thread_count: None,
        }
    }
    
    /// Create a config with a specific load balancing strategy.
    pub fn with_strategy(uris: Vec<String>, strategy: LoadBalanceStrategy) -> Self {
        Self {
            endpoints: uris.into_iter().map(EndpointConfig::new).collect(),
            strategy,
            default_thread_count: None,
        }
    }
}

/// Per-endpoint statistics tracked with lock-free atomic operations.
#[derive(Debug)]
pub struct EndpointStats {
    /// Total number of requests sent to this endpoint.
    pub total_requests: AtomicU64,
    
    /// Total bytes read from this endpoint.
    pub bytes_read: AtomicU64,
    
    /// Total bytes written to this endpoint.
    pub bytes_written: AtomicU64,
    
    /// Number of errors encountered on this endpoint.
    pub error_count: AtomicU64,
    
    /// Current number of active requests to this endpoint.
    /// Used by LeastConnections strategy to route to least-loaded endpoint.
    pub active_requests: AtomicUsize,
}

impl EndpointStats {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            active_requests: AtomicUsize::new(0),
        }
    }
    
    /// Get a snapshot of current statistics (non-atomic, may be slightly inconsistent).
    pub fn snapshot(&self) -> EndpointStatsSnapshot {
        EndpointStatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of endpoint statistics at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct EndpointStatsSnapshot {
    pub total_requests: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub error_count: u64,
    pub active_requests: usize,
}

/// Internal endpoint information combining ObjectStore instance with metadata.
struct EndpointInfo {
    /// Underlying ObjectStore implementation for this endpoint.
    store: Box<dyn ObjectStore>,
    
    /// Original URI for this endpoint.
    uri: String,
    
    /// Statistics for this endpoint.
    stats: Arc<EndpointStats>,
    
    /// Configured thread count for this endpoint.
    thread_count: usize,
    
    /// Process affinity hint (if configured).
    process_affinity: Option<usize>,
}

/// Multi-endpoint ObjectStore wrapper with load balancing and per-endpoint configuration.
///
/// This wrapper implements the ObjectStore trait and distributes requests across
/// multiple backend storage endpoints according to the configured load balancing strategy.
pub struct MultiEndpointStore {
    /// List of configured endpoints.
    endpoints: Vec<EndpointInfo>,
    
    /// Load balancing strategy.
    strategy: LoadBalanceStrategy,
    
    /// Round-robin counter (used by RoundRobin strategy).
    next_index: AtomicUsize,
}

impl MultiEndpointStore {
    /// Create a new multi-endpoint store from a list of URIs.
    ///
    /// # Arguments
    /// * `uris` - List of storage URIs (all must use the same scheme)
    /// * `strategy` - Load balancing strategy to use
    /// * `default_thread_count` - Optional default thread count per endpoint
    ///
    /// # Errors
    /// Returns an error if:
    /// - URI list is empty
    /// - URIs use different schemes (e.g., mixing s3:// and file://)
    /// - Any URI is invalid or store creation fails
    pub fn new(
        uris: Vec<String>,
        strategy: LoadBalanceStrategy,
        default_thread_count: Option<usize>,
    ) -> Result<Self> {
        let config = MultiEndpointStoreConfig {
            endpoints: uris.into_iter().map(EndpointConfig::new).collect(),
            strategy,
            default_thread_count,
        };
        Self::from_config(config)
    }
    
    /// Create a new multi-endpoint store from a full configuration.
    ///
    /// This allows per-endpoint thread count and process affinity configuration.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Endpoint list is empty
    /// - Endpoints use different URI schemes
    /// - Any store creation fails
    pub fn from_config(config: MultiEndpointStoreConfig) -> Result<Self> {
        if config.endpoints.is_empty() {
            return Err(anyhow!("Multi-endpoint store requires at least one endpoint"));
        }
        
        // Validate all endpoints use the same scheme
        let first_scheme = crate::uri_utils::infer_scheme_from_uri(&config.endpoints[0].uri)?;
        for endpoint in &config.endpoints[1..] {
            let scheme = crate::uri_utils::infer_scheme_from_uri(&endpoint.uri)?;
            if scheme != first_scheme {
                return Err(anyhow!(
                    "All endpoints must use the same URI scheme. Found {} and {}",
                    first_scheme, scheme
                ));
            }
        }
        
        // Create ObjectStore instances for each endpoint
        let mut endpoints = Vec::with_capacity(config.endpoints.len());
        for endpoint_config in config.endpoints {
            let store = store_for_uri(&endpoint_config.uri)
                .with_context(|| format!("Failed to create store for URI: {}", endpoint_config.uri))?;
            
            let thread_count = endpoint_config.thread_count
                .or(config.default_thread_count)
                .unwrap_or_else(num_cpus::get);
            
            endpoints.push(EndpointInfo {
                store,
                uri: endpoint_config.uri.clone(),
                stats: Arc::new(EndpointStats::new()),
                thread_count,
                process_affinity: endpoint_config.process_affinity,
            });
        }
        
        Ok(Self {
            endpoints,
            strategy: config.strategy,
            next_index: AtomicUsize::new(0),
        })
    }
    
    /// Select the next endpoint to use based on load balancing strategy.
    fn select_endpoint(&self) -> &EndpointInfo {
        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.next_index.fetch_add(1, Ordering::Relaxed);
                &self.endpoints[idx % self.endpoints.len()]
            }
            LoadBalanceStrategy::LeastConnections => {
                // Find endpoint with minimum active requests
                self.endpoints.iter()
                    .min_by_key(|e| e.stats.active_requests.load(Ordering::Acquire))
                    .expect("endpoints list is non-empty")
            }
        }
    }
    
    /// Get statistics for all endpoints.
    pub fn get_all_stats(&self) -> Vec<(String, EndpointStatsSnapshot)> {
        self.endpoints.iter()
            .map(|e| (e.uri.clone(), e.stats.snapshot()))
            .collect()
    }
    
    /// Get total aggregated statistics across all endpoints.
    pub fn get_total_stats(&self) -> EndpointStatsSnapshot {
        let mut total = EndpointStatsSnapshot {
            total_requests: 0,
            bytes_read: 0,
            bytes_written: 0,
            error_count: 0,
            active_requests: 0,
        };
        
        for endpoint in &self.endpoints {
            let stats = endpoint.stats.snapshot();
            total.total_requests += stats.total_requests;
            total.bytes_read += stats.bytes_read;
            total.bytes_written += stats.bytes_written;
            total.error_count += stats.error_count;
            total.active_requests += stats.active_requests;
        }
        
        total
    }
    
    /// Get endpoint configuration details (URIs, thread counts, process affinity).
    pub fn get_endpoint_configs(&self) -> Vec<(String, usize, Option<usize>)> {
        self.endpoints.iter()
            .map(|e| (e.uri.clone(), e.thread_count, e.process_affinity))
            .collect()
    }
    
    /// Get the current load balancing strategy.
    pub fn strategy(&self) -> LoadBalanceStrategy {
        self.strategy
    }
    
    /// Get the number of configured endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }
}

#[async_trait]
impl ObjectStore for MultiEndpointStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.get(uri).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        match &result {
            Ok(data) => {
                endpoint.stats.bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
            }
            Err(_) => {
                endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        result
    }
    
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.get_range(uri, offset, length).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        match &result {
            Ok(data) => {
                endpoint.stats.bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
            }
            Err(_) => {
                endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        result
    }
    
    async fn get_optimized(&self, uri: &str) -> Result<Bytes> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.get_optimized(uri).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        match &result {
            Ok(data) => {
                endpoint.stats.bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
            }
            Err(_) => {
                endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        result
    }
    
    async fn get_range_optimized(
        &self,
        uri: &str,
        offset: u64,
        length: Option<u64>,
        chunk_size: Option<usize>,
        max_concurrency: Option<usize>,
    ) -> Result<Bytes> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.get_range_optimized(
            uri, offset, length, chunk_size, max_concurrency
        ).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        match &result {
            Ok(data) => {
                endpoint.stats.bytes_read.fetch_add(data.len() as u64, Ordering::Relaxed);
            }
            Err(_) => {
                endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        result
    }
    
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let data_len = data.len() as u64;
        let result = endpoint.store.put(uri, data).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        match &result {
            Ok(_) => {
                endpoint.stats.bytes_written.fetch_add(data_len, Ordering::Relaxed);
            }
            Err(_) => {
                endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        result
    }
    
    async fn delete(&self, uri: &str) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.delete(uri).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(uris.len() as u64, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.delete_batch(uris).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn list(&self, prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.list(prefix, recursive).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String>> + Send + 'a>> {
        use futures::stream::StreamExt;
        
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        
        let mut stream = endpoint.store.list_stream(uri_prefix, recursive);
        let stats = endpoint.stats.clone();
        let stats_for_end = stats.clone();
        let mut request_ended = false;
        
        Box::pin(async_stream::stream! {
            while let Some(result) = stream.next().await {
                if result.is_err() && !request_ended {
                    stats.error_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    stats.active_requests.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                    request_ended = true;
                }
                yield result;
            }
            
            // Decrement active count if not already done due to error
            if !request_ended {
                stats_for_end.active_requests.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            }
        })
    }
    
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.stat(uri).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let data_len = data.len() as u64;
        let result = endpoint.store.put_multipart(uri, data, part_size).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        match &result {
            Ok(_) => {
                endpoint.stats.bytes_written.fetch_add(data_len, Ordering::Relaxed);
            }
            Err(_) => {
                endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        result
    }
    
    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.delete_prefix(uri_prefix).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn create_container(&self, name: &str) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.create_container(name).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn delete_container(&self, name: &str) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.delete_container(name).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn get_writer(&self, uri: &str) -> Result<Box<dyn crate::object_store::ObjectWriter>> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        
        let result = endpoint.store.get_writer(uri).await;
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn mkdir(&self, uri: &str) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.mkdir(uri).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
    
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let endpoint = self.select_endpoint();
        endpoint.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        endpoint.stats.active_requests.fetch_add(1, Ordering::AcqRel);
        
        let result = endpoint.store.rmdir(uri, recursive).await;
        
        endpoint.stats.active_requests.fetch_sub(1, Ordering::AcqRel);
        
        if result.is_err() {
            endpoint.stats.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_endpoint_config_builders() {
        let cfg1 = EndpointConfig::new("s3://host/bucket/");
        assert_eq!(cfg1.uri, "s3://host/bucket/");
        assert_eq!(cfg1.thread_count, None);
        assert_eq!(cfg1.process_affinity, None);
        
        let cfg2 = EndpointConfig::with_threads("s3://host/bucket/", 32);
        assert_eq!(cfg2.thread_count, Some(32));
        
        let cfg3 = EndpointConfig::with_process_affinity("s3://host/bucket/", 2);
        assert_eq!(cfg3.process_affinity, Some(2));
    }
    
    #[test]
    fn test_multi_endpoint_config_builders() {
        let uris = vec![
            "file:///tmp/test1/".to_string(),
            "file:///tmp/test2/".to_string(),
        ];
        
        let cfg1 = MultiEndpointStoreConfig::from_uris(uris.clone());
        assert_eq!(cfg1.endpoints.len(), 2);
        assert_eq!(cfg1.strategy, LoadBalanceStrategy::RoundRobin);
        
        let cfg2 = MultiEndpointStoreConfig::with_strategy(uris, LoadBalanceStrategy::LeastConnections);
        assert_eq!(cfg2.strategy, LoadBalanceStrategy::LeastConnections);
    }
    
    #[test]
    fn test_endpoint_stats() {
        let stats = EndpointStats::new();
        
        stats.total_requests.fetch_add(100, Ordering::Relaxed);
        stats.bytes_read.fetch_add(1024, Ordering::Relaxed);
        stats.active_requests.fetch_add(5, Ordering::Relaxed);
        
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 100);
        assert_eq!(snapshot.bytes_read, 1024);
        assert_eq!(snapshot.active_requests, 5);
    }
    
    #[test]
    fn test_multi_endpoint_store_creation() {
        // Create temporary directories for file:// endpoints
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        let tmp3 = TempDir::new().unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
            format!("file://{}/", tmp3.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        assert_eq!(store.endpoint_count(), 3);
        assert_eq!(store.strategy(), LoadBalanceStrategy::RoundRobin);
    }
    
    #[test]
    fn test_multi_endpoint_store_with_config() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let config = MultiEndpointStoreConfig {
            endpoints: vec![
                EndpointConfig {
                    uri: format!("file://{}/", tmp1.path().display()),
                    thread_count: Some(16),
                    process_affinity: Some(0),
                },
                EndpointConfig {
                    uri: format!("file://{}/", tmp2.path().display()),
                    thread_count: Some(8),
                    process_affinity: Some(1),
                },
            ],
            strategy: LoadBalanceStrategy::LeastConnections,
            default_thread_count: Some(12),
        };
        
        let store = MultiEndpointStore::from_config(config).unwrap();
        
        assert_eq!(store.endpoint_count(), 2);
        assert_eq!(store.strategy(), LoadBalanceStrategy::LeastConnections);
        
        let configs = store.get_endpoint_configs();
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].1, 16); // First endpoint: 16 threads
        assert_eq!(configs[1].1, 8);  // Second endpoint: 8 threads
        assert_eq!(configs[0].2, Some(0)); // Process affinity
        assert_eq!(configs[1].2, Some(1));
    }
    
    #[test]
    fn test_mixed_schemes_error() {
        let tmp1 = TempDir::new().unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            "s3://bucket/prefix/".to_string(), // Different scheme!
        ];
        
        let result = MultiEndpointStore::new(uris, LoadBalanceStrategy::RoundRobin, None);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("same URI scheme"));
    }
    
    #[test]
    fn test_empty_endpoints_error() {
        let result = MultiEndpointStore::new(vec![], LoadBalanceStrategy::RoundRobin, None);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("at least one endpoint"));
    }
    
    #[tokio::test]
    async fn test_round_robin_load_balancing() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        let tmp3 = TempDir::new().unwrap();
        
        // Create test files in each directory
        let test_data = b"test content";
        fs::write(tmp1.path().join("test.txt"), test_data).unwrap();
        fs::write(tmp2.path().join("test.txt"), test_data).unwrap();
        fs::write(tmp3.path().join("test.txt"), test_data).unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
            format!("file://{}/", tmp3.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        // Perform multiple operations and verify they're distributed
        for _ in 0..9 {
            let uri = format!("file://{}/test.txt", tmp1.path().display());
            let _ = store.get(&uri).await;
        }
        
        let stats = store.get_all_stats();
        assert_eq!(stats.len(), 3);
        
        // With round-robin, each endpoint should get 3 requests
        for (_, stat) in &stats {
            assert_eq!(stat.total_requests, 3);
        }
    }
    
    #[tokio::test]
    async fn test_least_connections_load_balancing() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let test_data = b"test content";
        fs::write(tmp1.path().join("test.txt"), test_data).unwrap();
        fs::write(tmp2.path().join("test.txt"), test_data).unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::LeastConnections,
            None,
        ).unwrap();
        
        // Perform operations
        for _ in 0..10 {
            let uri = format!("file://{}/test.txt", tmp1.path().display());
            let _ = store.get(&uri).await;
        }
        
        let stats = store.get_all_stats();
        assert_eq!(stats.len(), 2);
        
        // Verify requests were distributed (not necessarily evenly due to timing)
        let total: u64 = stats.iter().map(|(_, s)| s.total_requests).sum();
        assert_eq!(total, 10);
    }
    
    #[tokio::test]
    async fn test_put_get_operations() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        // Put data
        let test_data = b"Hello, multi-endpoint world!";
        let uri = format!("file://{}/output.txt", tmp1.path().display());
        store.put(&uri, test_data).await.unwrap();
        
        // Get data back
        let retrieved = store.get(&uri).await.unwrap();
        assert_eq!(&retrieved[..], test_data);
        
        // Check stats
        let stats = store.get_total_stats();
        assert!(stats.total_requests >= 2); // At least put + get
        assert!(stats.bytes_written >= test_data.len() as u64);
        assert!(stats.bytes_read >= test_data.len() as u64);
    }
    
    #[tokio::test]
    async fn test_list_operations() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        // Create multiple files
        fs::write(tmp1.path().join("file1.txt"), b"content1").unwrap();
        fs::write(tmp1.path().join("file2.txt"), b"content2").unwrap();
        fs::create_dir(tmp1.path().join("subdir")).unwrap();
        fs::write(tmp1.path().join("subdir/file3.txt"), b"content3").unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        // List non-recursive
        let prefix = format!("file://{}/", tmp1.path().display());
        let files = store.list(&prefix, false).await.unwrap();
        assert!(files.len() >= 2); // At least file1.txt and file2.txt
        
        // List recursive
        let files_recursive = store.list(&prefix, true).await.unwrap();
        assert!(files_recursive.len() >= 3); // Should include subdir/file3.txt
    }
    
    #[tokio::test]
    async fn test_stat_operations() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let test_data = b"test content for stat";
        fs::write(tmp1.path().join("stat_test.txt"), test_data).unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        let uri = format!("file://{}/stat_test.txt", tmp1.path().display());
        let metadata = store.stat(&uri).await.unwrap();
        
        assert_eq!(metadata.size, test_data.len() as u64);
    }
    
    #[tokio::test]
    async fn test_delete_operations() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        fs::write(tmp1.path().join("delete_me.txt"), b"will be deleted").unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        let uri = format!("file://{}/delete_me.txt", tmp1.path().display());
        
        // Verify file exists
        assert!(tmp1.path().join("delete_me.txt").exists());
        
        // Delete it
        store.delete(&uri).await.unwrap();
        
        // Verify it's gone
        assert!(!tmp1.path().join("delete_me.txt").exists());
    }
    
    #[tokio::test]
    async fn test_error_counting() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        // Try to get a non-existent file
        let uri = format!("file://{}/nonexistent.txt", tmp1.path().display());
        let result = store.get(&uri).await;
        assert!(result.is_err());
        
        // Check that error was counted
        let stats = store.get_all_stats();
        let total_errors: u64 = stats.iter().map(|(_, s)| s.error_count).sum();
        assert!(total_errors > 0);
    }
    
    #[tokio::test]
    async fn test_get_range_operations() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let test_data = b"0123456789ABCDEFGHIJ";
        fs::write(tmp1.path().join("range_test.txt"), test_data).unwrap();
        
        let uris = vec![
            format!("file://{}/", tmp1.path().display()),
            format!("file://{}/", tmp2.path().display()),
        ];
        
        let store = MultiEndpointStore::new(
            uris.clone(),
            LoadBalanceStrategy::RoundRobin,
            None,
        ).unwrap();
        
        let uri = format!("file://{}/range_test.txt", tmp1.path().display());
        
        // Get range [5:10)
        let range_data = store.get_range(&uri, 5, Some(5)).await.unwrap();
        assert_eq!(&range_data[..], b"56789");
        
        // Check stats tracked the read
        let stats = store.get_total_stats();
        assert!(stats.bytes_read >= 5);
    }
    
    #[test]
    fn test_default_thread_count() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let config = MultiEndpointStoreConfig {
            endpoints: vec![
                EndpointConfig::new(format!("file://{}/", tmp1.path().display())),
                EndpointConfig::new(format!("file://{}/", tmp2.path().display())),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
            default_thread_count: Some(24),
        };
        
        let store = MultiEndpointStore::from_config(config).unwrap();
        
        let configs = store.get_endpoint_configs();
        // Both should use default thread count
        assert_eq!(configs[0].1, 24);
        assert_eq!(configs[1].1, 24);
    }
    
    #[test]
    fn test_thread_count_override() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        
        let config = MultiEndpointStoreConfig {
            endpoints: vec![
                EndpointConfig::with_threads(format!("file://{}/", tmp1.path().display()), 64),
                EndpointConfig::new(format!("file://{}/", tmp2.path().display())),
            ],
            strategy: LoadBalanceStrategy::RoundRobin,
            default_thread_count: Some(16),
        };
        
        let store = MultiEndpointStore::from_config(config).unwrap();
        
        let configs = store.get_endpoint_configs();
        // First endpoint uses override (64), second uses default (16)
        assert_eq!(configs[0].1, 64);
        assert_eq!(configs[1].1, 16);
    }
}
