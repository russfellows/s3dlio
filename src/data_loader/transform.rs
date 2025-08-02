

// Similarly you could add FilterDataset and FlatMapDataset here...
// For brevity you can start with MapDataset only and add others later.//! Dataset combinators: map, filter, flat_map (Stage 2).
//!
//! We currently implement MapDataset as an example; add FilterDataset and FlatMapDataset similarly.

use async_trait::async_trait;
use futures_util::StreamExt;
use crate::{DatasetError, Dataset};
use crate::data_loader::dataset::DynStream;

/// Map – applies a function `F` to each item.
#[derive(Debug, Clone)]
pub struct MapDataset<D, F, O>
where
    D: Dataset,
    F: Fn(D::Item) -> O + Clone + Send + Sync + 'static,
    O: Send + 'static,
{
    inner: D,
    func: F,
}

impl<D, F, O> MapDataset<D, F, O>
where
    D: Dataset,
    F: Fn(D::Item) -> O + Clone + Send + Sync + 'static,
    O: Send + 'static,
{
    pub fn new(inner: D, func: F) -> Self {
        Self { inner, func }
    }
}

#[async_trait]
impl<D, F, O> Dataset for MapDataset<D, F, O>
where
    D: Dataset,
    F: Fn(D::Item) -> O + Clone + Send + Sync + 'static,
    O: Send + 'static,
{
    type Item = O;

    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    async fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let v = self.inner.get(index).await?;
        Ok((self.func)(v))
    }

    fn as_stream(&self) -> Option<DynStream<Self::Item>> {
        if let Some(mut st) = self.inner.as_stream() {
            let func = self.func.clone();
            let mapped = async_stream::try_stream! {
                while let Some(item) = st.next().await {
                    let v = item?;
                    yield func(v);
                }
            };
            Some(Box::pin(mapped))
        } else {
            None
        }
    }
}