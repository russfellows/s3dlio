//! Loader options for s3dlio data loader (Stage 2 additions).
//!
//! Stage 1 exposed only `batch_size` and `drop_last`.
//! Stage 2 adds shuffle, num_workers, prefetch, and auto_tune.
//!
//! Builder helpers are provided so callers can write a fluent style:
//!
//! let opts = LoaderOptions::default()
//!     .with_batch_size(128)
//!     .drop_last(true)
//!     .shuffle(true, 42)
//!     .num_workers(8)
//!     .prefetch(16)
//!     .auto_tune(false);
//!

#[derive(Debug, Clone)]
pub struct LoaderOptions {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to drop the final, possibly incomplete batch.
    pub drop_last: bool,

    // ── New in Stage 2 ──────────────────────────────────────────────────────
    /// If true, use a shuffled sampler (deterministic with `seed`).
    pub shuffle: bool,
    /// RNG seed used when `shuffle == true`. Ignored otherwise.
    pub seed: u64,
    /// Number of parallel workers. `0` means "auto" (use number of CPUs).
    pub num_workers: usize,
    /// Size of the bounded prefetch queue. `0` disables prefetching.
    pub prefetch: usize,
    /// If true, adapt `num_workers`/`prefetch` during runtime (simple heuristic).
    pub auto_tune: bool,
}

impl Default for LoaderOptions {
    fn default() -> Self {
        Self {
            batch_size: 32,
            drop_last: false,
            shuffle: false,
            seed: 0,
            num_workers: 0,
            prefetch: 0,
            auto_tune: false,
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

    /// Enable/disable shuffling and set seed.
    ///
    /// When `on` is false, the seed is left unchanged but ignored.
    pub fn shuffle(mut self, on: bool, seed: u64) -> Self {
        self.shuffle = on;
        if on {
            self.seed = seed;
        }
        self
    }

    /// Set the number of worker tasks/threads used for fetching/decoding.
    ///
    /// `0` means "auto", which the loader interprets as the number of CPUs.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the number of elements buffered ahead of the consumer.
    ///
    /// `0` disables prefetching.
    pub fn prefetch(mut self, n: usize) -> Self {
        self.prefetch = n;
        self
    }

    /// Enable/disable auto‑tuning of `num_workers` and `prefetch`.
    pub fn auto_tune(mut self, on: bool) -> Self {
        self.auto_tune = on;
        self
    }
}
