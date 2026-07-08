// src/data_loader/async_pool_dataloader.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::data_loader::dataset::{Dataset, DatasetError};
use crate::data_loader::options::{LoaderOptions, LoadingMode};
use crate::data_loader::parallel_fetch::DropCancel;
use crate::object_store::{store_for_uri, ObjectStore};
use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
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
    type Item = Bytes;

    fn len(&self) -> Option<usize> {
        Some(self.uris.len())
    }

    /// Return all URIs — enables URI-carrying iterators (e.g. `PyBytesAsyncDataLoader.items()`).
    fn keys(&self) -> Option<Vec<String>> {
        Some(self.uris.clone())
    }

    async fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        let uri = self
            .uris
            .get(idx)
            .ok_or(DatasetError::IndexOutOfRange(idx))?;

        // Return Bytes directly - zero-copy!
        self.store
            .get(uri)
            .await
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
                64 // Default when num_workers is 0 (auto)
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
    pub fn stream_with_pool(
        self,
        pool_config: PoolConfig,
    ) -> ReceiverStream<Result<Vec<Bytes>, DatasetError>> {
        let batch_size = self.options.batch_size.max(1);
        let drop_last = self.options.drop_last;
        let dataset_len = self.dataset.len();
        let cancel_token = self.options.cancellation_token.clone();

        let (tx, rx) =
            mpsc::channel::<Result<Vec<Bytes>, DatasetError>>(pool_config.readahead_batches);

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
            )
            .await
            {
                eprintln!("AsyncPoolDataLoader error: {}", e);
            }
        });

        ReceiverStream::new(rx)
    }

    /// Core async pooling worker with cancellation support.
    ///
    /// Each fetch is `tokio::spawn`'d as its own task so tokio can
    /// distribute polling across worker threads (issue #148, audit
    /// §1.1 / §3.1a). To make that safe for early-drop and external-
    /// cancel scenarios, an internal `CancellationToken` is wired
    /// through each spawned task via `tokio::select!` — cancelling
    /// the internal token drops the fetch future, cancelling the
    /// in-flight I/O for free (no JoinHandle tracking / abort() needed).
    ///
    /// The internal token is a child of the external `cancel_token`
    /// when the caller supplied one, so external cancels propagate
    /// automatically. A `DropCancel` guard on this function's stack
    /// cancels the internal token on ANY exit path (normal return,
    /// early `break` from receiver-drop, panic on this function's
    /// own frame) — that's what keeps a "the worker went home"
    /// event from leaving detached fetch tasks running.
    async fn run_async_pool_worker(
        dataset: Arc<MultiBackendDataset>,
        tx: mpsc::Sender<Result<Vec<Bytes>, DatasetError>>,
        batch_size: usize,
        drop_last: bool,
        pool_config: PoolConfig,
        dataset_len: usize,
        cancel_token: Option<CancellationToken>,
    ) -> Result<()> {
        // Child of the external token when supplied; otherwise a
        // fresh independent token. Cancelling the child does NOT
        // cancel the parent, so users' external tokens are safe from
        // us; but the child DOES get cancelled when the parent does.
        let internal_token = match &cancel_token {
            Some(ext) => ext.child_token(),
            None => CancellationToken::new(),
        };
        // Belt: cancel on any function exit path.
        let _drop_cancel = DropCancel(internal_token.clone());

        type SpawnedFetch = JoinHandle<(usize, Result<Bytes, anyhow::Error>)>;

        let mut pending_requests: FuturesUnordered<SpawnedFetch> = FuturesUnordered::new();
        let mut next_index = 0;
        let mut completed_data = std::collections::HashMap::new();
        let mut current_batch = Vec::new();
        let total_items = dataset_len;
        let timeout = pool_config.batch_timeout;

        // Helper: build one spawned fetch task with cancellation wired
        // through a select!. Returning the JoinHandle keeps the FuturesUnordered
        // homogeneous.
        let spawn_fetch = |index: usize| -> SpawnedFetch {
            let uri = dataset.get_uri(index).unwrap_or_default().to_string();
            let store = dataset.store.clone();
            let token = internal_token.clone();
            tokio::spawn(async move {
                tokio::select! {
                    // Cancellation arm: dropping the fetch future
                    // aborts the in-flight I/O.
                    _ = token.cancelled() => {
                        (index, Err(anyhow::anyhow!("Request cancelled")))
                    }
                    // Fetch arm.
                    r = tokio::time::timeout(timeout, store.get(&uri)) => {
                        let result = match r {
                            Ok(Ok(data)) => Ok(data),
                            Ok(Err(e)) => Err(anyhow::anyhow!("Store error: {}", e)),
                            Err(_) => Err(anyhow::anyhow!("Request timeout after {:?}", timeout)),
                        };
                        (index, result)
                    }
                }
            })
        };

        // Start initial pool of requests.
        for _ in 0..pool_config.pool_size.min(total_items) {
            if internal_token.is_cancelled() {
                break;
            }
            if next_index < total_items && dataset.get_uri(next_index).is_some() {
                pending_requests.push(spawn_fetch(next_index));
                next_index += 1;
            }
        }

        // Process completions and maintain pool.
        while !pending_requests.is_empty() {
            if internal_token.is_cancelled() {
                break;
            }

            let join_res = match pending_requests.next().await {
                Some(r) => r,
                None => continue,
            };

            // Convert a spawned-task outcome into a (index, Result<Bytes>)
            // pair. A panic in the fetch surfaces as JoinError::is_panic() —
            // we translate it into a DatasetError::Backend so the caller
            // sees the error instead of silent truncation (audit §2.1's
            // "bonus fix").
            let (index, result) = match join_res {
                Ok(pair) => pair,
                Err(join_err) if join_err.is_panic() => {
                    let msg = format!("fetch task panicked: {}", join_err);
                    let _ = tx
                        .send(Err(DatasetError::Backend(anyhow::anyhow!(msg))))
                        .await;
                    // A panicked task can't tell us its index, so the
                    // pool bookkeeping (completed_data, batching)
                    // can't recover it. Break to end gracefully.
                    break;
                }
                Err(_) => {
                    // Cancelled or otherwise — happens during shutdown.
                    // The DropCancel + select! path is the intended
                    // shutdown flow; skip silently and continue draining.
                    continue;
                }
            };

            match result {
                Ok(data) => {
                    completed_data.insert(index, data);

                    // Refill: submit another fetch if items remain and
                    // we haven't been asked to stop taking new work.
                    if next_index < total_items
                        && !internal_token.is_cancelled()
                        && dataset.get_uri(next_index).is_some()
                    {
                        pending_requests.push(spawn_fetch(next_index));
                        next_index += 1;
                    }

                    // Try to form batches from completed data (out-of-order completion).
                    while current_batch.len() < batch_size && !completed_data.is_empty() {
                        if let Some(&key) = completed_data.keys().next() {
                            let data = completed_data.remove(&key).unwrap();
                            current_batch.push(data);
                        } else {
                            break;
                        }
                    }

                    // Send complete batch.
                    if current_batch.len() == batch_size
                        && tx
                            .send(Ok(std::mem::take(&mut current_batch)))
                            .await
                            .is_err()
                    {
                        break; // Receiver dropped
                    }
                }
                Err(e) => {
                    if tx.send(Err(DatasetError::Backend(e))).await.is_err() {
                        break; // Receiver dropped
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
            if current_batch.len() == batch_size
                && tx
                    .send(Ok(std::mem::take(&mut current_batch)))
                    .await
                    .is_err()
            {
                break;
            }
        }

        // Finally, send any remaining partial batch
        if !current_batch.is_empty() && !drop_last {
            let _ = tx.send(Ok(current_batch)).await;
        }

        Ok(())
    }

    /// Standard stream interface (maintains compatibility)
    pub fn stream(self) -> ReceiverStream<Result<Vec<Bytes>, DatasetError>> {
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
    pub fn stream(self) -> ReceiverStream<Result<Vec<Bytes>, DatasetError>> {
        let loading_mode = self.options.loading_mode.clone();
        match loading_mode {
            LoadingMode::Sequential => {
                // Use traditional sequential loading
                let traditional_loader =
                    crate::data_loader::dataloader::DataLoader::new(self.dataset, self.options);
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
    pub fn enhanced_stream(
        self,
        dataset: MultiBackendDataset,
    ) -> ReceiverStream<Result<Vec<Bytes>, DatasetError>> {
        let pool_config = self.to_pool_config();
        AsyncPoolDataLoader::new(dataset, self).stream_with_pool(pool_config)
    }
}
