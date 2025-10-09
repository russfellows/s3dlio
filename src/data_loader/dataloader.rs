// src/data_loader/dataloader.rs
//
// Stage-2 DataLoader: batching, shuffle+seed, prefetch; returns a concrete
// ReceiverStream so tests can use `.next().await` and `.collect().await`
// without pinning/boxing issues.

use crate::data_loader::dataset::{Dataset, DatasetError};
use crate::data_loader::options::{LoaderOptions, LoadingMode};
use crate::data_loader::sampler::{Sampler, SequentialSampler, ShuffleSampler};

use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;

/// High-level iterator that yields batches from a `Dataset`.
pub struct DataLoader<D: Dataset> {
    ds: D,
    opts: LoaderOptions,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(ds: D, opts: LoaderOptions) -> Self {
        Self { ds, opts }
    }

    /// Produce an async stream of `Result<Vec<Item>>` batches.
    /// Returned type is a concrete `ReceiverStream`, which is `Unpin` and `Sized`.
    /// 
    /// Supports graceful cancellation via `LoaderOptions::cancellation_token`.
    /// When cancelled, prefetch loops exit cleanly without producing new batches.
    pub fn stream(self) -> ReceiverStream<Result<Vec<D::Item>, DatasetError>> {
        // Check if async pooling was requested
        if let LoadingMode::AsyncPool(_) = self.opts.loading_mode {
            eprintln!("Warning: AsyncPool loading mode requested but only supported for MultiBackendDataset");
            eprintln!("Falling back to Sequential loading mode. Use AsyncPoolDataLoader directly for async pooling.");
        }

        let batch = self.opts.batch_size.max(1);
        let drop_last = self.opts.drop_last;
        let shuffle = self.opts.shuffle;
        let seed = self.opts.seed;
        let prefetch = self.opts.prefetch.max(1);
        let cancel_token = self.opts.cancellation_token.clone();

        let (tx, rx) = mpsc::channel::<Result<Vec<D::Item>, DatasetError>>(prefetch);

        // Shared dataset for async task(s).
        let ds = Arc::new(self.ds);

        if let Some(n) = ds.len() {
            // -------- Map-style with known length: drive indices from a sampler --------
            let mut sampler: Box<dyn Sampler + Send> = if shuffle {
                Box::new(ShuffleSampler::new(n, seed))
            } else {
                Box::new(SequentialSampler::new(n))
            };
            let tx_idx = tx.clone();
            let ds_idx = ds.clone();
            let cancel_token_idx = cancel_token.clone();
            tokio::spawn(async move {
                let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
                while let Some(idx) = sampler.next_index() {
                    // Check cancellation before fetching
                    if let Some(ref token) = cancel_token_idx {
                        if token.is_cancelled() {
                            break;  // Clean exit
                        }
                    }
                    
                    match ds_idx.get(idx).await {
                        Ok(item) => {
                            buf.push(item);
                            if buf.len() == batch {
                                if tx_idx.send(Ok(std::mem::take(&mut buf))).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx_idx.send(Err(e)).await;
                            break;
                        }
                    }
                }
                if !drop_last && !buf.is_empty() {
                    let _ = tx_idx.send(Ok(buf)).await;
                }
            });
        } else if let Some(mut stream) = ds.as_stream() {
            // -------- Iterable-style: drain underlying stream, form batches --------
            // NEW (Stage 0): optional sharding across distributed ranks + DataLoader workers
            let shard_world = self.opts.shard_world_size.max(1);
            let worker_n    = self.opts.num_workers_pytorch.max(1);
            let shard_mod   = shard_world * worker_n;

            // Clamp shard indices to valid ranges to be defensive
            let rank       = if shard_world > 0 { self.opts.shard_rank.min(shard_world - 1) } else { 0 };
            let worker_id  = if worker_n > 0 { self.opts.worker_id.min(worker_n - 1) } else { 0 };
            let shard_id   = rank * worker_n + worker_id;

            let tx_it = tx.clone();
            let cancel_token_it = cancel_token.clone();
            tokio::spawn(async move {
                let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
                let mut i: usize = 0; // item counter for round-robin sharding
                while let Some(next) = stream.next().await {
                    // Check cancellation before processing
                    if let Some(ref token) = cancel_token_it {
                        if token.is_cancelled() {
                            break;  // Clean exit
                        }
                    }
                    
                    match next {
                        Ok(item) => {
                            // Keep item if this shard "owns" position i in round-robin
                            let take = if shard_mod <= 1 { true } else { (i % shard_mod) == shard_id };
                            i = i.wrapping_add(1);

                            if !take { continue; }

                            buf.push(item);
                            if buf.len() == batch {
                                if tx_it.send(Ok(std::mem::take(&mut buf))).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx_it.send(Err(e)).await;
                            break;
                        }
                    }
                }
                if !drop_last && !buf.is_empty() {
                    let _ = tx_it.send(Ok(buf)).await;
                }
            });
        } else {
            // -------- Map-style with unknown length: probe indices until IndexOutOfRange --------
            let tx_un = tx.clone();
            let ds_un = ds.clone();
            let cancel_token_un = cancel_token;
            tokio::spawn(async move {
                let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
                let mut idx = 0usize;
                loop {
                    // Check cancellation before probing
                    if let Some(ref token) = cancel_token_un {
                        if token.is_cancelled() {
                            break;  // Clean exit
                        }
                    }
                    
                    match ds_un.get(idx).await {
                        Ok(item) => {
                            buf.push(item);
                            if buf.len() == batch {
                                if tx_un.send(Ok(std::mem::take(&mut buf))).await.is_err() {
                                    break;
                                }
                            }
                            idx += 1;
                        }
                        Err(DatasetError::IndexOutOfRange(_)) => {
                            // graceful end
                            break;
                        }
                        Err(e) => {
                            let _ = tx_un.send(Err(e)).await;
                            break;
                        }
                    }
                }
                if !drop_last && !buf.is_empty() {
                    let _ = tx_un.send(Ok(buf)).await;
                }
            });
        }

        ReceiverStream::new(rx)
    }
}

