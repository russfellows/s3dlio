// src/data_loader/dataloader.rs
//
// Stage-2 DataLoader: batching, shuffle+seed, prefetch; returns a concrete
// ReceiverStream so tests can use `.next().await` and `.collect().await`
// without pinning/boxing issues.

use crate::data_loader::dataset::{Dataset, DatasetError};
use crate::data_loader::options::LoaderOptions;
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
    pub fn stream(self) -> ReceiverStream<Result<Vec<D::Item>, DatasetError>> {
        let batch = self.opts.batch_size.max(1);
        let drop_last = self.opts.drop_last;
        let shuffle = self.opts.shuffle;
        let seed = self.opts.seed;
        let prefetch = self.opts.prefetch.max(1);

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
            tokio::spawn(async move {
                let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
                while let Some(idx) = sampler.next_index() {
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
            let tx_it = tx.clone();
            tokio::spawn(async move {
                let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
                while let Some(next) = stream.next().await {
                    match next {
                        Ok(item) => {
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
            tokio::spawn(async move {
                let mut buf: Vec<D::Item> = Vec::with_capacity(batch);
                let mut idx = 0usize;
                loop {
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

