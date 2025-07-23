//! Minimal sequential `DataLoader` (Stage 1).
//!
//! * Handles **map‑style** and **iterable** datasets transparently.
//! * Yields `Result<Vec<Item>, DatasetError>` where each `Vec` is a batch.
//! * No shuffling, parallelism, or prefetching yet—those come in Stage 2.

use crate::data_loader::dataset::{Dataset, DatasetError};
use crate::data_loader::options::LoaderOptions;

use async_stream::try_stream;
use futures_core::stream::Stream;
use futures_util::StreamExt;
use std::pin::Pin;
use std::sync::Arc;

/// High‑level iterator that produces batched samples from a dataset.
pub struct DataLoader<D>
where
    D: Dataset,
{
    dataset: Arc<D>,
    opts: LoaderOptions,
}

impl<D> DataLoader<D>
where
    D: Dataset,
{
    /// Create a new loader.
    pub fn new(dataset: D, opts: LoaderOptions) -> Self {
        Self {
            dataset: Arc::new(dataset),
            opts,
        }
    }

    /// Return an **async stream** over the dataset that yields batches.
    ///
    /// ```ignore
    /// # use s3dlio::{DataLoader, LoaderOptions};
    /// # async fn demo<D: s3dlio::Dataset>(ds: D) -> anyhow::Result<()> {
    /// let loader = DataLoader::new(ds, LoaderOptions::default());
    /// tokio::pin!(let batches = loader.stream());
    /// while let Some(batch) = batches.next().await {
    ///     let data = batch?; // Vec<D::Item>
    ///     // training step ...
    /// }
    /// # Ok(()) }
    /// ```
    pub fn stream(
        self,
    ) -> Pin<
        Box<dyn Stream<Item = Result<Vec<D::Item>, DatasetError>> + Send + 'static>,
    > {
        let ds = self.dataset.clone();
        let opts = self.opts.clone();

        Box::pin(try_stream! {
            let bs = opts.batch_size;

            // -------- Iterable dataset -----------------------------------
            if let Some(mut st) = ds.as_stream() {
                let mut acc = Vec::with_capacity(bs);
                while let Some(item) = st.next().await {
                    acc.push(item?);
                    if acc.len() == bs {
                        yield std::mem::take(&mut acc);
                    }
                }
                if !acc.is_empty() && !opts.drop_last {
                    yield acc;
                }
                return;
            }

            // -------- Map‑style dataset -----------------------------------
            match ds.len() {
                Some(total) => {
                    // Known length – straightforward indexing.
                    let mut index = 0;
                    while index < total {
                        let mut batch = Vec::with_capacity(bs);
                        let end = (index + bs).min(total);
                        while index < end {
                            batch.push(ds.get(index).await?);
                            index += 1;
                        }
                        if batch.len() == bs || !opts.drop_last {
                            yield batch;
                        }
                    }
                }
                None => {
                    // Unknown length – keep reading until we hit EOF.
                    let mut index = 0usize;
                    loop {
                        let mut batch = Vec::with_capacity(bs);
                        for _ in 0..bs {
                            match ds.get(index).await {
                                Ok(item) => batch.push(item),
                                Err(DatasetError::IndexOutOfRange(_)) => {
                                    if !batch.is_empty() && !opts.drop_last {
                                        yield batch;
                                    }
                                    return;
                                }
                                Err(e) => Err(e)?,
                            }
                            index += 1;
                        }
                        yield batch;
                    }
                }
            }
        })
    }
}

impl<D> std::fmt::Debug for DataLoader<D>
where
    D: Dataset,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLoader")
            .field("batch_size", &self.opts.batch_size)
            .finish()
    }
}

