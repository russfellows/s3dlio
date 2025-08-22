// src/data_loader/options.rs
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

//
// NEW:
// LoaderOptions for the s3dlio data loader.
//
// Stage 0 additions:
// - ReaderMode (Sequential | Range)
// - Range reader tuning: part_size, max_inflight_parts
// - Sharding hints: shard_rank, shard_world_size, worker_id, num_workers_pytorch

/// How the dataset reads object bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderMode {
    /// Fetch the whole object with a single GET.
    Sequential,
    /// Fetch object via HTTP byte-ranges (optionally in parts).
    Range,
}

impl Default for ReaderMode {
    fn default() -> Self { ReaderMode::Sequential }
}

/// How the dataloader processes batches.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadingMode {
    /// Traditional sequential loading - maintains order, waits for full batches
    Sequential,
    /// Async pooling with out-of-order completion and dynamic batch formation
    AsyncPool(crate::data_loader::async_pool_dataloader::PoolConfig),
}

impl Default for LoadingMode {
    fn default() -> Self { LoadingMode::Sequential }
}

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

    // ── New in Stage 3 (loading strategy) ───────────────────────────────────
    /// Loading strategy: Sequential (traditional) or AsyncPool (dynamic batching)
    pub loading_mode: LoadingMode,

    // ── New in Stage 0 (reader strategy) ────────────────────────────────────
    /// Reader strategy used to fetch object bytes.
    pub reader_mode: ReaderMode,
    /// Range-reader part size (bytes). Ignored when `reader_mode == Sequential`.
    pub part_size: usize,
    /// Max concurrent in-flight range parts (>=1). Ignored for Sequential.
    pub max_inflight_parts: usize,

    // ── New in Stage 0 (sharding) ───────────────────────────────────────────
    /// Global rank in the distributed job (0-based).
    pub shard_rank: usize,
    /// Total number of ranks in the distributed job.
    pub shard_world_size: usize,
    /// Worker id within PyTorch DataLoader (0-based).
    pub worker_id: usize,
    /// Total PyTorch DataLoader workers for this dataset instance.
    pub num_workers_pytorch: usize,
}

impl Default for LoaderOptions {
    fn default() -> Self {
        Self {
            // original defaults
            batch_size: 32,
            drop_last: false,
            shuffle: false,
            seed: 0,
            num_workers: 0,
            prefetch: 0,
            auto_tune: false,

            // new defaults (Stage 3)
            loading_mode: LoadingMode::Sequential,

            // new defaults (Stage 0)
            reader_mode: ReaderMode::Sequential,
            part_size: 8 * 1024 * 1024, // 8 MiB
            max_inflight_parts: 4,

            shard_rank: 0,
            shard_world_size: 1,
            worker_id: 0,
            num_workers_pytorch: 1,
        }
    }
}

impl LoaderOptions {
    /// Builder-style helper: change the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Builder-style helper: set `drop_last`.
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

    /// Enable/disable auto-tuning of `num_workers` and `prefetch`.
    pub fn auto_tune(mut self, on: bool) -> Self {
        self.auto_tune = on;
        self
    }

    // ── New in Stage 0: builder helpers ─────────────────────────────────────

    /// Select reader strategy.
    pub fn reader_mode(mut self, mode: ReaderMode) -> Self {
        self.reader_mode = mode;
        self
    }

    /// Set range-reader part size (bytes).
    pub fn part_size(mut self, bytes: usize) -> Self {
        self.part_size = bytes.max(1);
        self
    }

    /// Set max concurrent in-flight parts (>=1).
    pub fn max_inflight_parts(mut self, n: usize) -> Self {
        self.max_inflight_parts = n.max(1);
        self
    }

    /// Set distributed sharding (rank/world_size).
    pub fn shard(mut self, rank: usize, world: usize) -> Self {
        self.shard_rank = rank;
        self.shard_world_size = world.max(1);
        self
    }

    /// Set DataLoader worker sharding (worker_id/num_workers).
    pub fn worker(mut self, worker_id: usize, num_workers: usize) -> Self {
        self.worker_id = worker_id;
        self.num_workers_pytorch = num_workers.max(1);
        self
    }

    /// Use traditional sequential loading (maintains order, waits for full batches)
    pub fn sequential_loading(mut self) -> Self {
        self.loading_mode = LoadingMode::Sequential;
        self
    }

    /// Use async pool loading with default configuration
    pub fn async_pool_loading(mut self) -> Self {
        self.loading_mode = LoadingMode::AsyncPool(
            crate::data_loader::async_pool_dataloader::PoolConfig::default()
        );
        self
    }

    /// Use async pool loading with custom configuration
    pub fn async_pool_loading_with_config(mut self, config: crate::data_loader::async_pool_dataloader::PoolConfig) -> Self {
        self.loading_mode = LoadingMode::AsyncPool(config);
        self
    }
}

