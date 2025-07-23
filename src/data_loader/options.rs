//! User‑visible knobs for configuring a [`DataLoader`](crate::dataloader::DataLoader).

/// Parameters that control batching behaviour.
///
/// Stage 1 exposes only the two essentials.  Additional fields (shuffle,
/// workers, prefetch, etc.) will be added in Stage 2 without breaking the
/// existing API.
#[derive(Debug, Clone)]
pub struct LoaderOptions {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to drop the final, possibly incomplete batch.
    pub drop_last: bool,
}

impl Default for LoaderOptions {
    fn default() -> Self {
        Self {
            batch_size: 32,
            drop_last: false,
        }
    }
}

impl LoaderOptions {
    /// Builder‑style helper: change the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Builder‑style helper: set `drop_last`.
    pub fn drop_last(mut self, yes: bool) -> Self {
        self.drop_last = yes;
        self
    }
}

