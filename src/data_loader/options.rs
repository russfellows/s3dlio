// src/data_loader/options.rs
//!
//! Stage 1 exposed only `batch_size` and `drop_last`.
//! Stage 2 adds shuffle, num_workers, prefetch, and auto_tune.
//! Stage 3 adds AI/ML realism knobs: pin_memory, persistent_workers, timeout, etc.
//!
//! Builder helpers are provided so callers can write a fluent style:
//!
//! let opts = LoaderOptions::default()
//!     .with_batch_size(128)
//!     .drop_last(true)
//!     .shuffle(true, 42)
//!     .num_workers(8)
//!     .prefetch(16)
//!     .pin_memory(true)
//!     .persistent_workers(true)
//!     .timeout_seconds(30.0)
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

/// Multiprocessing context for worker processes (mirrors PyTorch DataLoader options)
#[derive(Debug, Clone, PartialEq)]
pub enum MultiprocessingContext {
    /// Use spawn method (safest, slower startup)
    Spawn,
    /// Use fork method (faster startup, potential issues with CUDA)
    Fork,
    /// Use forkserver method (balance of safety and performance)
    ForkServer,
}

impl Default for MultiprocessingContext {
    fn default() -> Self { MultiprocessingContext::Spawn }
}

/// Sampling strategy for dataset iteration
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerType {
    /// Sequential iteration through dataset
    Sequential,
    /// Random sampling (with replacement if specified)
    Random { replacement: bool },
    /// Weighted random sampling for imbalanced datasets
    WeightedRandom { weights: Vec<f64> },
    /// Distributed sampling for multi-GPU training
    DistributedRandom { rank: usize, world_size: usize },
}

impl Default for SamplerType {
    fn default() -> Self { SamplerType::Sequential }
}

/// Memory format optimization for tensor layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFormat {
    /// Channels-last format (NHWC) - better for some GPUs
    ChannelsLast,
    /// Channels-first format (NCHW) - traditional PyTorch default
    ChannelsFirst,
    /// Let the framework decide optimal format
    Auto,
}

impl Default for MemoryFormat {
    fn default() -> Self { MemoryFormat::Auto }
}

/// Page cache behavior hint for file I/O (maps to posix_fadvise on Linux/Unix)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageCacheMode {
    /// Sequential access pattern - prefetch data ahead (POSIX_FADV_SEQUENTIAL)
    Sequential,
    /// Random access pattern - don't prefetch (POSIX_FADV_RANDOM)
    Random,
    /// Don't pollute page cache - data won't be reused (POSIX_FADV_DONTNEED)
    DontNeed,
    /// Let OS decide based on default heuristics (POSIX_FADV_NORMAL)
    Normal,
    /// Automatically choose based on file size (Sequential for large, Random for small)
    Auto,
}

impl Default for PageCacheMode {
    fn default() -> Self { PageCacheMode::Auto }
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

    // ── New in Stage 3 (AI/ML realism knobs) ────────────────────────────────
    /// Pin memory for faster CPU→GPU transfers (critical for GPU training)
    pub pin_memory: bool,
    /// Keep workers alive between epochs (massive performance improvement)
    pub persistent_workers: bool,
    /// Worker timeout in seconds (handles stuck/slow workers)
    pub timeout_seconds: Option<f64>,
    /// Multiprocessing context (spawn/fork/forkserver)
    pub multiprocessing_context: MultiprocessingContext,
    /// Sampling strategy for dataset iteration
    pub sampler_type: SamplerType,
    /// Memory format optimization for tensor layouts
    pub memory_format: MemoryFormat,
    /// Non-blocking transfers for performance (when supported)
    pub non_blocking: bool,
    /// Custom generator seed for enhanced reproducibility
    pub generator_seed: Option<u64>,
    /// Enable dataset-level transforms/augmentations
    pub enable_transforms: bool,
    /// Batch collation buffer size for optimization
    pub collate_buffer_size: usize,
    /// Page cache behavior hint for file I/O operations
    pub page_cache_mode: PageCacheMode,
    
    // ── New in v0.9.0 (adaptive tuning) ─────────────────────────────────────
    /// Optional adaptive configuration for auto-tuning (default: None/disabled)
    pub adaptive: Option<crate::adaptive_config::AdaptiveConfig>,
    
    // ── New in v0.9.2 (cancellation support) ────────────────────────────────
    /// Optional cancellation token for graceful shutdown of async operations
    /// When set, all prefetch loops and async workers will cooperatively cancel
    /// Enables clean Ctrl-C handling and prevents orphaned background tasks
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
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

            // new defaults (Stage 3 - AI/ML realism)
            pin_memory: false,
            persistent_workers: false,
            timeout_seconds: None,
            multiprocessing_context: MultiprocessingContext::default(),
            sampler_type: SamplerType::default(),
            memory_format: MemoryFormat::default(),
            non_blocking: false,
            generator_seed: None,
            enable_transforms: false,
            collate_buffer_size: 1024, // 1KB default buffer
            page_cache_mode: PageCacheMode::default(),
            
            // new defaults (v0.9.0 - adaptive tuning)
            adaptive: None, // Disabled by default - users opt-in
            
            // new defaults (v0.9.2 - cancellation support)
            cancellation_token: None, // No cancellation by default
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

    // ── New in Stage 3: AI/ML realism builder helpers ───────────────────────

    /// Enable/disable memory pinning for faster CPU→GPU transfers
    pub fn pin_memory(mut self, pin: bool) -> Self {
        self.pin_memory = pin;
        self
    }

    /// Enable/disable persistent workers (keep workers alive between epochs)
    pub fn persistent_workers(mut self, persistent: bool) -> Self {
        self.persistent_workers = persistent;
        self
    }

    /// Set worker timeout in seconds (None = no timeout)
    pub fn timeout_seconds(mut self, timeout: Option<f64>) -> Self {
        self.timeout_seconds = timeout;
        self
    }

    /// Set timeout in seconds (convenience method)
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }

    /// Set multiprocessing context
    pub fn multiprocessing_context(mut self, context: MultiprocessingContext) -> Self {
        self.multiprocessing_context = context;
        self
    }

    /// Use spawn multiprocessing (safest)
    pub fn use_spawn(mut self) -> Self {
        self.multiprocessing_context = MultiprocessingContext::Spawn;
        self
    }

    /// Use fork multiprocessing (faster startup)
    pub fn use_fork(mut self) -> Self {
        self.multiprocessing_context = MultiprocessingContext::Fork;
        self
    }

    /// Use forkserver multiprocessing (balanced)
    pub fn use_forkserver(mut self) -> Self {
        self.multiprocessing_context = MultiprocessingContext::ForkServer;
        self
    }

    /// Set sampling strategy
    pub fn sampler_type(mut self, sampler: SamplerType) -> Self {
        self.sampler_type = sampler;
        self
    }

    /// Use random sampling
    pub fn random_sampling(mut self, replacement: bool) -> Self {
        self.sampler_type = SamplerType::Random { replacement };
        self
    }

    /// Use weighted random sampling for imbalanced datasets
    pub fn weighted_sampling(mut self, weights: Vec<f64>) -> Self {
        self.sampler_type = SamplerType::WeightedRandom { weights };
        self
    }

    /// Use distributed sampling for multi-GPU training
    pub fn distributed_sampling(mut self, rank: usize, world_size: usize) -> Self {
        self.sampler_type = SamplerType::DistributedRandom { rank, world_size };
        self
    }

    /// Set memory format for tensor layout optimization
    pub fn memory_format(mut self, format: MemoryFormat) -> Self {
        self.memory_format = format;
        self
    }

    /// Use channels-last memory format (NHWC)
    pub fn channels_last(mut self) -> Self {
        self.memory_format = MemoryFormat::ChannelsLast;
        self
    }

    /// Use channels-first memory format (NCHW)
    pub fn channels_first(mut self) -> Self {
        self.memory_format = MemoryFormat::ChannelsFirst;
        self
    }

    /// Enable/disable non-blocking transfers
    pub fn non_blocking(mut self, non_blocking: bool) -> Self {
        self.non_blocking = non_blocking;
        self
    }

    /// Set custom generator seed for enhanced reproducibility
    pub fn generator_seed(mut self, seed: Option<u64>) -> Self {
        self.generator_seed = seed;
        self
    }

    /// Set generator seed (convenience method)
    pub fn with_generator_seed(mut self, seed: u64) -> Self {
        self.generator_seed = Some(seed);
        self
    }

    /// Enable/disable dataset transforms/augmentations
    pub fn enable_transforms(mut self, enable: bool) -> Self {
        self.enable_transforms = enable;
        self
    }

    /// Set collate buffer size for batch optimization
    pub fn collate_buffer_size(mut self, size: usize) -> Self {
        self.collate_buffer_size = size.max(64); // Minimum 64 bytes
        self
    }

    /// Configure for high-performance GPU training (common preset)
    pub fn gpu_optimized(mut self) -> Self {
        self.pin_memory = true;
        self.persistent_workers = true;
        self.non_blocking = true;
        self.memory_format = MemoryFormat::ChannelsFirst;
        self.multiprocessing_context = MultiprocessingContext::Spawn;
        self
    }

    /// Configure for distributed training (common preset)
    pub fn distributed_optimized(self, rank: usize, world_size: usize) -> Self {
        self.gpu_optimized()
            .distributed_sampling(rank, world_size)
            .shard(rank, world_size)
    }

    /// Configure for development/debugging (safe settings)
    pub fn debug_mode(mut self) -> Self {
        self.num_workers = 0; // Single-threaded
        self.persistent_workers = false;
        self.timeout_seconds = Some(10.0); // Short timeout
        self.auto_tune = false;
        self
    }

    /// Set page cache behavior hint for file I/O
    pub fn with_page_cache_mode(mut self, mode: PageCacheMode) -> Self {
        self.page_cache_mode = mode;
        self
    }
    
    // ── New in v0.9.0: adaptive tuning builder helpers ──────────────────────
    
    /// Enable adaptive tuning with default configuration
    /// 
    /// When enabled, part_size and num_workers will be auto-tuned based on
    /// workload characteristics. Explicit settings always override adaptive.
    pub fn with_adaptive(mut self) -> Self {
        self.adaptive = Some(crate::adaptive_config::AdaptiveConfig::enabled());
        self
    }
    
    /// Set custom adaptive configuration
    pub fn with_adaptive_config(mut self, config: crate::adaptive_config::AdaptiveConfig) -> Self {
        self.adaptive = Some(config);
        self
    }
    
    // ── New in v0.9.2: cancellation support builder helpers ─────────────────
    
    /// Set cancellation token for graceful shutdown
    /// 
    /// When set, all async operations (prefetch loops, pool workers) will
    /// cooperatively check for cancellation and exit cleanly.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// use s3dlio::data_loader::*;
    /// use tokio_util::sync::CancellationToken;
    /// 
    /// # async fn example() -> anyhow::Result<()> {
    /// let cancel_token = CancellationToken::new();
    /// 
    /// // Setup Ctrl-C handler
    /// let token_clone = cancel_token.clone();
    /// tokio::spawn(async move {
    ///     tokio::signal::ctrl_c().await.expect("Failed to listen for Ctrl-C");
    ///     token_clone.cancel();
    /// });
    /// 
    /// let opts = LoaderOptions::default()
    ///     .with_batch_size(32)
    ///     .with_cancellation_token(cancel_token);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_cancellation_token(mut self, token: tokio_util::sync::CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }
    
    /// Remove cancellation token (disable cancellation)
    pub fn without_cancellation(mut self) -> Self {
        self.cancellation_token = None;
        self
    }
    
    /// Compute effective part size considering adaptive tuning
    /// 
    /// CRITICAL: Explicit part_size always overrides adaptive behavior
    pub fn effective_part_size(&self, file_size: Option<usize>) -> usize {
        use crate::adaptive_config::AdaptiveParams;
        
        if let Some(ref adaptive_cfg) = self.adaptive {
            let params = AdaptiveParams::new(adaptive_cfg.clone());
            // If part_size was explicitly set, it's used; otherwise adaptive computes it
            params.compute_part_size(file_size, Some(self.part_size))
        } else {
            // No adaptive: use explicit part_size
            self.part_size
        }
    }
    
    /// Compute effective concurrency considering adaptive tuning
    /// 
    /// CRITICAL: Explicit num_workers always overrides adaptive behavior
    pub fn effective_concurrency(&self, workload: Option<crate::adaptive_config::WorkloadType>) -> usize {
        use crate::adaptive_config::AdaptiveParams;
        
        if let Some(ref adaptive_cfg) = self.adaptive {
            let params = AdaptiveParams::new(adaptive_cfg.clone());
            // If num_workers was explicitly set (non-zero), use it; otherwise adaptive computes
            let explicit = if self.num_workers > 0 { Some(self.num_workers) } else { None };
            params.compute_concurrency(workload, explicit)
        } else {
            // No adaptive: use explicit num_workers (0 means auto = CPU count)
            if self.num_workers > 0 {
                self.num_workers
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(8)
            }
        }
    }
}

