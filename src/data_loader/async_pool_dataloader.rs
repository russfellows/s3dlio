// src/data_loader/async_pool_dataloader.rs
//
// Enhanced DataLoader with async request pooling and dynamic batch formation
// Implements out-of-order completion with multi-backend support

use crate::object_store::{ObjectStore, store_for_uri};
use crate::data_loader::dataset::{DatasetError, Dataset};
use crate::data_loader::options::{LoaderOptions, LoadingMode};
use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{StreamExt, FuturesUnordered};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

/// Request metadata for tracking async operations
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RequestInfo {
    index: usize,
    uri: String,
    submitted_at: Instant,
}

/// Completed request with its data
#[derive(Debug)]
#[allow(dead_code)]
struct CompletedRequest {
    info: RequestInfo,
    data: Vec<u8>,
    completed_at: Instant,
}

/// Multi-backend dataset using unified ObjectStore
#[derive(Clone)]
pub struct MultiBackendDataset {
    pub uris: Vec<String>,
    pub store: Arc<dyn ObjectStore>,
}

impl MultiBackendDataset {
    /// Create dataset from URI prefix using appropriate backend
    pub async fn from_prefix(prefix_uri: &str) -> Result<Self> {
        let store: Arc<dyn ObjectStore> = Arc::from(store_for_uri(prefix_uri)?);
        let uris = store.list(prefix_uri, true).await?;
        
        Ok(Self { uris, store })
    }

    /// Create dataset from explicit list of URIs
    pub fn from_uris(uris: Vec<String>) -> Result<Self> {
        if uris.is_empty() {
            return Ok(Self {
                uris,
                store: Arc::from(store_for_uri("file://dummy")?), // Won't be used
            });
        }
        
        // Use first URI to determine backend
        let store: Arc<dyn ObjectStore> = Arc::from(store_for_uri(&uris[0])?);
        Ok(Self { uris, store })
    }
    
    pub fn len(&self) -> usize {
        self.uris.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.uris.is_empty()
    }
    
    /// Get URI by index
    pub fn get_uri(&self, index: usize) -> Option<&str> {
        self.uris.get(index).map(|s| s.as_str())
    }
}

#[async_trait]
impl Dataset for MultiBackendDataset {
    type Item = Vec<u8>;

    fn len(&self) -> Option<usize> {
        Some(self.uris.len())
    }

    async fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        let uri = self.uris.get(idx)
            .ok_or(DatasetError::IndexOutOfRange(idx))?;
        
        // Convert Bytes to Vec<u8> for Dataset API compatibility
        self.store.get(uri).await
            .map(|bytes| bytes.to_vec())
            .map_err(|e| DatasetError::from(e.to_string()))
    }
}

/// Enhanced DataLoader with async request pooling
pub struct AsyncPoolDataLoader {
    dataset: Arc<MultiBackendDataset>,
    options: LoaderOptions,
}

/// Configuration for async request pooling
#[derive(Debug, Clone, PartialEq)]
pub struct PoolConfig {
    /// Number of concurrent requests to maintain
    pub pool_size: usize,
    /// Target number of read-ahead batches 
    pub readahead_batches: usize,
    /// Maximum time to wait for batch completion
    pub batch_timeout: Duration,
    /// Maximum requests in flight globally
    pub max_inflight: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 64,
            readahead_batches: 4,
            batch_timeout: Duration::from_secs(30),
            max_inflight: 256,
        }
    }
}

impl PoolConfig {
    /// Create PoolConfig with sensible scaling from LoaderOptions
    /// 
    /// Maps LoaderOptions fields to pool configuration:
    /// - `pool_size` = `num_workers * 16` (scale parallelism)
    /// - `readahead_batches` = `prefetch.max(2)` (minimum prefetch depth)
    /// 
    /// This provides a reasonable starting point for users who want to derive
    /// pool configuration from their training parameters.
    /// 
    /// # Example
    /// ```ignore
    /// let options = LoaderOptions { num_workers: 4, prefetch: 3, ..Default::default() };
    /// let pool_config = PoolConfig::from_loader_options(&options);
    /// // pool_size = 64, readahead_batches = 3
    /// ```
    pub fn from_loader_options(opts: &LoaderOptions) -> Self {
        Self {
            pool_size: if opts.num_workers > 0 {
                opts.num_workers * 16
            } else {
                64  // Default when num_workers is 0 (auto)
            },
            readahead_batches: opts.prefetch.max(2),
            ..Default::default()
        }
    }
}


impl AsyncPoolDataLoader {
    pub fn new(dataset: MultiBackendDataset, options: LoaderOptions) -> Self {
        Self {
            dataset: Arc::new(dataset),
            options,
        }
    }

    /// Create enhanced dataloader from URI prefix
    pub async fn from_prefix(prefix_uri: &str, options: LoaderOptions) -> Result<Self> {
        let dataset = MultiBackendDataset::from_prefix(prefix_uri).await?;
        Ok(Self::new(dataset, options))
    }

    /// Enhanced stream with async pooling and dynamic batch formation
    /// Enhanced stream with async pooling and dynamic batch formation
    /// 
    /// Supports graceful cancellation via `LoaderOptions::cancellation_token`.
    /// When cancelled, the pool stops submitting new requests and drains pending ones.
    pub fn stream_with_pool(self, pool_config: PoolConfig) -> ReceiverStream<Result<Vec<Vec<u8>>, DatasetError>> {
        let batch_size = self.options.batch_size.max(1);
        let drop_last = self.options.drop_last;
        let dataset_len = self.dataset.len();
        let cancel_token = self.options.cancellation_token.clone();
        
        let (tx, rx) = mpsc::channel::<Result<Vec<Vec<u8>>, DatasetError>>(pool_config.readahead_batches);
        
        let dataset = Arc::clone(&self.dataset);
        
        tokio::spawn(async move {
            if let Err(e) = Self::run_async_pool_worker(
                dataset,
                tx,
                batch_size,
                drop_last,
                pool_config,
                dataset_len,
                cancel_token,
            ).await {
                eprintln!("AsyncPoolDataLoader error: {}", e);
            }
        });
        
        ReceiverStream::new(rx)
    }

    /// Core async pooling worker with cancellation support
    async fn run_async_pool_worker(
        dataset: Arc<MultiBackendDataset>,
        tx: mpsc::Sender<Result<Vec<Vec<u8>>, DatasetError>>,
        batch_size: usize,
        drop_last: bool,
        pool_config: PoolConfig,
        dataset_len: usize,
        cancel_token: Option<CancellationToken>,
    ) -> Result<()> {
        type RequestFuture = Pin<Box<dyn std::future::Future<Output = (usize, Result<Vec<u8>, anyhow::Error>)> + Send>>;
        
        let mut pending_requests: FuturesUnordered<RequestFuture> = FuturesUnordered::new();
        let mut next_index = 0;
        let mut completed_data = std::collections::HashMap::new();
        let mut current_batch = Vec::new();
        let total_items = dataset_len;
        let timeout = pool_config.batch_timeout;
        
        // Start initial pool of requests
        for _ in 0..pool_config.pool_size.min(total_items) {
            // Check cancellation before submitting initial requests
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;
                }
            }
            
            if next_index < total_items {
                if let Some(uri) = dataset.get_uri(next_index) {
                    let store = dataset.store.clone();
                    let uri = uri.to_string();
                    let index = next_index;
                    
                    let fut: RequestFuture = Box::pin(async move {
                        let result = match tokio::time::timeout(timeout, store.get(&uri)).await {
                            Ok(Ok(data)) => Ok(data.to_vec()), // Convert Bytes to Vec<u8>
                            Ok(Err(e)) => Err(anyhow::anyhow!("Store error: {}", e)),
                            Err(_) => Err(anyhow::anyhow!("Request timeout after {:?}", timeout)),
                        };
                        (index, result)
                    });
                    pending_requests.push(fut);
                    next_index += 1;
                }
            }
        }
        
        // Process completions and maintain pool
        while !pending_requests.is_empty() {
            // Check cancellation before processing next completion
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    break;
                }
            }
            
            if let Some((index, result)) = pending_requests.next().await {
                match result {
                    Ok(data) => {
                        completed_data.insert(index, data);
                        
                        // Add more requests to maintain pool size
                        if next_index < total_items {
                            // Check cancellation before submitting new requests
                            if let Some(ref token) = cancel_token {
                                if token.is_cancelled() {
                                    // Don't submit new requests, but continue processing pending ones
                                    continue;
                                }
                            }
                            
                            if let Some(uri) = dataset.get_uri(next_index) {
                                let store = dataset.store.clone();
                                let uri = uri.to_string();
                                let req_index = next_index;
                                
                                let fut: RequestFuture = Box::pin(async move {
                                    let result = match tokio::time::timeout(timeout, store.get(&uri)).await {
                                        Ok(Ok(data)) => Ok(data.to_vec()), // Convert Bytes to Vec<u8>
                                        Ok(Err(e)) => Err(anyhow::anyhow!("Store error: {}", e)),
                                        Err(_) => Err(anyhow::anyhow!("Request timeout after {:?}", timeout)),
                                    };
                                    (req_index, result)
                                });
                                pending_requests.push(fut);
                                next_index += 1;
                            }
                        }
                        
                        // Try to form batches from completed data (out-of-order completion)
                        while current_batch.len() < batch_size && !completed_data.is_empty() {
                            // Take any available completed item (out-of-order)
                            if let Some(&key) = completed_data.keys().next() {
                                let data = completed_data.remove(&key).unwrap();
                                current_batch.push(data);
                            } else {
                                break;
                            }
                        }
                        
                        // Send complete batch
                        if current_batch.len() == batch_size {
                            if tx.send(Ok(std::mem::take(&mut current_batch))).await.is_err() {
                                break; // Receiver dropped
                            }
                        }
                    }
                    Err(e) => {
                        if tx.send(Err(DatasetError::Backend(e))).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                }
            }
        }
        
        // Send any remaining data after all requests complete
        // Process any remaining completed data into final batches
        
        // First, process any remaining completed data
        while !completed_data.is_empty() {
            while current_batch.len() < batch_size && !completed_data.is_empty() {
                if let Some(&key) = completed_data.keys().next() {
                    let data = completed_data.remove(&key).unwrap();
                    current_batch.push(data);
                } else {
                    break;
                }
            }
            
            // Send complete batch
            if current_batch.len() == batch_size {
                if tx.send(Ok(std::mem::take(&mut current_batch))).await.is_err() {
                    break;
                }
            }
        }
        
        // Finally, send any remaining partial batch
        if !current_batch.is_empty() && !drop_last {
            let _ = tx.send(Ok(current_batch)).await;
        }
        
        Ok(())
    }

    /// Standard stream interface (maintains compatibility)
    pub fn stream(self) -> ReceiverStream<Result<Vec<Vec<u8>>, DatasetError>> {
        self.stream_with_pool(PoolConfig::default())
    }
}

/// Unified DataLoader for MultiBackendDataset that supports both Sequential and AsyncPool modes
pub struct UnifiedDataLoader {
    dataset: MultiBackendDataset,
    options: LoaderOptions,
}

impl UnifiedDataLoader {
    /// Create a new unified dataloader
    pub fn new(dataset: MultiBackendDataset, options: LoaderOptions) -> Self {
        Self { dataset, options }
    }

    /// Create from URI prefix using appropriate backend
    pub async fn from_prefix(prefix_uri: &str, options: LoaderOptions) -> Result<Self> {
        let dataset = MultiBackendDataset::from_prefix(prefix_uri).await?;
        Ok(Self::new(dataset, options))
    }

    /// Create from explicit list of URIs
    pub fn from_uris(uris: Vec<String>, options: LoaderOptions) -> Result<Self> {
        let dataset = MultiBackendDataset::from_uris(uris)?;
        Ok(Self::new(dataset, options))
    }

    /// Get stream using the configured loading mode
    pub fn stream(self) -> ReceiverStream<Result<Vec<Vec<u8>>, DatasetError>> {
        let loading_mode = self.options.loading_mode.clone();
        match loading_mode {
            LoadingMode::Sequential => {
                // Use traditional sequential loading
                let traditional_loader = crate::data_loader::dataloader::DataLoader::new(
                    self.dataset, 
                    self.options
                );
                traditional_loader.stream()
            }
            LoadingMode::AsyncPool(pool_config) => {
                // Use async pool loading
                let async_loader = AsyncPoolDataLoader::new(self.dataset, self.options);
                async_loader.stream_with_pool(pool_config)
            }
        }
    }
}

/// Extensions to LoaderOptions for async pooling
impl LoaderOptions {
    /// Create pool configuration from loader options
    pub fn to_pool_config(&self) -> PoolConfig {
        PoolConfig {
            pool_size: self.max_inflight_parts.max(32),
            readahead_batches: 4,
            batch_timeout: Duration::from_secs(10),
            max_inflight: self.max_inflight_parts * 2,
        }
    }
    
    /// Enhanced stream with automatic pool configuration
    pub fn enhanced_stream(self, dataset: MultiBackendDataset) -> ReceiverStream<Result<Vec<Vec<u8>>, DatasetError>> {
        let pool_config = self.to_pool_config();
        AsyncPoolDataLoader::new(dataset, self).stream_with_pool(pool_config)
    }
}
