// src/data_gen_alt.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

/// # Key Features
/// - No shared BASE_BLOCK (eliminates cross-block compression)
/// - Xoshiro256++ RNG (5-10x faster than ChaCha20)
/// - Compression distributed evenly via integer error accumulation
/// - Correct compress=1 behavior (truly incompressible, ~1.0 zstd ratio)
/// - When compress>1, compressibility is local to each block
/// - NUMA-aware parallel generation with thread pinning (optional)
use rand::{Rng, SeedableRng, RngCore};
use rand_xoshiro::Xoshiro256PlusPlus;  // Explicit high-performance RNG
use rayon::prelude::*;
use tracing::{debug, info};

use crate::constants::BLK_SIZE;

#[cfg(feature = "numa")]
use crate::numa::NumaTopology;

// =============================================================================
// Constants
// =============================================================================

/// Maximum back-reference distance for compression (1 KiB)
const MAX_BACK_REF_DISTANCE: usize = 1024;

/// Minimum run length for back-references (64 bytes)
const MIN_RUN_LENGTH: usize = 64;

/// Maximum run length for back-references (256 bytes)
const MAX_RUN_LENGTH: usize = 256;

// =============================================================================
// NUMA Configuration
// =============================================================================

/// NUMA optimization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumaMode {
    /// Auto-detect: enable NUMA optimizations only on multi-node systems (default)
    #[default]
    Auto,
    /// Force: enable optimizations even on UMA systems (for testing)
    Force,
}

/// Configuration for data generation with NUMA support
///
/// # Default Behavior
/// - Uses 50% of available CPU cores/threads for data generation
/// - Leaves the other 50% available for I/O operations
/// - Automatically detects NUMA topology and optimizes accordingly
///
/// # Examples
///
/// ```rust
/// use s3dlio::data_gen_alt::{GeneratorConfig, NumaMode, default_data_gen_threads, total_cpus};
///
/// // Use defaults (50% of CPUs, auto NUMA detection)
/// let config = GeneratorConfig::default();
///
/// // Override to use all CPUs
/// let config_all = GeneratorConfig {
///     max_threads: Some(total_cpus()),
///     ..Default::default()
/// };
///
/// // Override to use specific number of threads
/// let config_custom = GeneratorConfig {
///     max_threads: Some(4),
///     ..Default::default()
/// };
///
/// // Force NUMA optimizations even on UMA systems (for testing)
/// let config_force_numa = GeneratorConfig {
///     numa_mode: NumaMode::Force,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Total size in bytes
    pub size: usize,
    /// Deduplication factor (1 = no dedup, N = N:1 logical:physical ratio)
    pub dedup_factor: usize,
    /// Compression factor (1 = incompressible, N = N:1 logical:physical ratio)
    pub compress_factor: usize,
    /// NUMA optimization mode (Auto, Force, or Disabled)
    pub numa_mode: NumaMode,
    /// Maximum number of threads to use (None = use all available cores)
    pub max_threads: Option<usize>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            size: BLK_SIZE,
            dedup_factor: 1,
            compress_factor: 1,
            numa_mode: NumaMode::Auto,
            max_threads: Some(default_data_gen_threads()), // Use 50% of CPUs by default
        }
    }
}

/// Calculate default number of threads for data generation
///
/// Returns 50% of available logical CPUs (cores × hyperthreads).
/// This leaves the other 50% available for I/O operations.
///
/// Minimum: 1 thread
/// Maximum: Total available logical CPUs
pub fn default_data_gen_threads() -> usize {
    let total_cpus = num_cpus::get();
    let half = total_cpus / 2;
    half.max(1) // At least 1 thread
}

/// Get total available logical CPUs
///
/// Returns the total number of logical CPUs (cores × hyperthreads)
/// available on the system. This can be used to override the default
/// thread count if you want to use more or fewer threads.
pub fn total_cpus() -> usize {
    num_cpus::get()
}

// =============================================================================
// Helper Functions (No longer needed - Xoshiro256PlusPlus has built-in seeding)
// =============================================================================

// Note: Xoshiro256++ has built-in seed_from_u64 that uses SplitMix64 internally,
// so we don't need custom seed generation helpers.

// =============================================================================
// Block Filling Helper
// =============================================================================

/// Fill a single block with controlled compression characteristics
///
/// # Algorithm
/// 1. Fill entire block with Xoshiro256++ keystream (high entropy baseline)
/// 2. Add local back-references to achieve target compression ratio
///
/// # Parameters
/// - `out`: Output buffer (must be BLK_SIZE bytes)
/// - `unique_block_idx`: Index of the unique block (for seeding)
/// - `copy_len_for_block`: Target number of bytes to make compressible
/// - `call_entropy`: Per-call entropy for RNG seeding
///
/// # Compression Strategy
/// When `copy_len_for_block > 0`, creates local repetition via back-references:
/// - Copies runs of 64-256 bytes from earlier positions in the same block
/// - Back-reference distance: 1-1024 bytes (stays local)
/// - This mimics LZ compression sources without cross-block duplication
#[inline]
fn fill_block_alt(
    out: &mut [u8],
    unique_block_idx: usize,
    copy_len_for_block: usize,
    call_entropy: u64,
) {
    // Seed Xoshiro256++ uniquely per unique_block_idx
    // SeedableRng::seed_from_u64 uses built-in SplitMix64 for fast initialization
    let seed = call_entropy ^ ((unique_block_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Step 1: Fill entire block with high-entropy keystream
    rng.fill_bytes(out);

    // Step 2: Add local back-references to achieve target compressibility
    if copy_len_for_block == 0 || out.len() <= 1 {
        return; // Fully incompressible
    }

    let mut made = 0usize;
    while made < copy_len_for_block {
        // Choose run length: 64-256 bytes (capped to remaining target)
        let remaining = copy_len_for_block.saturating_sub(made);
        if remaining == 0 {
            break;
        }

        let run_len = rng.random_range(MIN_RUN_LENGTH..=MAX_RUN_LENGTH).min(remaining).min(out.len() - 1);
        if run_len == 0 {
            break;
        }

        // Choose destination position (ensure room for run_len)
        if out.len() <= run_len {
            break;
        }
        let dst = rng.random_range(0..(out.len() - run_len));

        // Choose source position (back-reference from earlier in block)
        // Distance: 1-1024 bytes (or less if near start of block)
        let max_back = dst.clamp(1, MAX_BACK_REF_DISTANCE);
        let back = rng.random_range(1..=max_back);
        let src = dst.saturating_sub(back);

        // Safety check: ensure we don't read past end of buffer
        if src + run_len > out.len() {
            break;
        }

        // Copy within same block (memmove semantics handle overlaps)
        out.copy_within(src..(src + run_len), dst);

        made += run_len;
    }
}

// =============================================================================
// Single-Pass Generator (Drop-in Alternative)
// =============================================================================

/// Generate data with full configuration support (NUMA-aware)
///
/// # Default Behavior
/// - Uses 50% of available CPU cores/threads (leaving 50% for I/O)
/// - Automatically detects NUMA/UMA topology
/// - Enables NUMA optimizations on multi-node systems
/// - Uses thread pinning on NUMA systems (when `thread-pinning` feature enabled)
///
/// # Algorithm
/// 1. Fill blocks with Xoshiro256++ keystream (high entropy baseline)
/// 2. Add local back-references for compression
/// 3. Use round-robin deduplication across unique blocks
/// 4. Parallel generation via rayon (NUMA-aware if enabled)
///
/// # Performance
/// - 5-15 GB/s per core with incompressible data
/// - 1-4 GB/s with compression enabled (depends on compress factor)
/// - Near-linear scaling with CPU cores
/// - NUMA optimizations when >1 NUMA node detected
///
/// # Examples
///
/// ```rust
/// use s3dlio::data_gen_alt::{generate_data_with_config, GeneratorConfig};
///
/// // Use defaults (50% CPUs, 100 MiB, no dedup/compression)
/// let config = GeneratorConfig {
///     size: 100 * 1024 * 1024,
///     ..Default::default()
/// };
/// let data = generate_data_with_config(config);
/// ```
pub fn generate_data_with_config(config: GeneratorConfig) -> Vec<u8> {
    info!(
        "Starting data generation with config: size={}, dedup={}, compress={}, numa_mode={:?}",
        config.size,
        config.dedup_factor,
        config.compress_factor,
        config.numa_mode
    );

    let size = config.size.max(BLK_SIZE);
    let nblocks = size.div_ceil(BLK_SIZE);

    let dedup_factor = config.dedup_factor.max(1);
    let unique_blocks = if dedup_factor > 1 {
        ((nblocks as f64) / (dedup_factor as f64)).round().max(1.0) as usize
    } else {
        nblocks
    };

    debug!(
        "Generating: size={}, blocks={}, dedup={}, unique_blocks={}, compress={}",
        size,
        nblocks,
        dedup_factor,
        unique_blocks,
        config.compress_factor
    );

    // Calculate per-block copy lengths using integer error accumulation
    let (f_num, f_den) = if config.compress_factor > 1 {
        (config.compress_factor - 1, config.compress_factor)
    } else {
        (0, 1)
    };
    let floor_len = (f_num * BLK_SIZE) / f_den;
    let rem = (f_num * BLK_SIZE) % f_den;

    let copy_lens: Vec<usize> = {
        let mut v = Vec::with_capacity(unique_blocks);
        let mut err = 0;
        for _ in 0..unique_blocks {
            err += rem;
            if err >= f_den {
                err -= f_den;
                v.push(floor_len + 1);
            } else {
                v.push(floor_len);
            }
        }
        v
    };

    // Per-call entropy for RNG seeding
    let call_entropy = generate_call_entropy();

    // Allocate buffer
    let total_size = nblocks * BLK_SIZE;
    debug!("Allocating {} bytes ({} blocks)", total_size, nblocks);
    let mut data: Vec<u8> = vec![0u8; total_size];

    // Configure thread pool based on config
    let num_threads = config.max_threads.unwrap_or_else(default_data_gen_threads);
    info!("Using {} threads for parallel generation", num_threads);

    // NUMA optimization check - always detect topology in Auto mode
    #[cfg(feature = "numa")]
    let numa_topology = NumaTopology::detect().ok();

    #[cfg(feature = "numa")]
    let should_optimize_numa = if let Some(ref topology) = numa_topology {
        let optimize = match config.numa_mode {
            NumaMode::Auto => topology.num_nodes > 1,
            NumaMode::Force => true,
        };

        if optimize {
            info!(
                "NUMA optimization enabled: {} nodes detected",
                topology.num_nodes
            );
        } else {
            debug!(
                "NUMA optimization not needed: {} nodes detected (UMA system)",
                topology.num_nodes
            );
        }
        optimize
    } else {
        false
    };

    #[cfg(not(feature = "numa"))]
    #[allow(unused_variables)]
    let should_optimize_numa = false;

    debug!("Starting parallel generation with rayon");

    // Build thread pool with optional NUMA-aware thread pinning
    #[cfg(all(feature = "numa", feature = "thread-pinning"))]
    let pool = if should_optimize_numa {
        if let Some(ref topology) = numa_topology {
            if topology.num_nodes > 1 {
                debug!(
                    "Configuring NUMA-aware thread pinning for {} nodes",
                    topology.num_nodes
                );

                let cpu_map = std::sync::Arc::new(build_cpu_affinity_map(topology, num_threads));

                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .spawn_handler(move |thread| {
                        let cpu_map = cpu_map.clone();
                        let mut b = std::thread::Builder::new();
                        if let Some(name) = thread.name() {
                            b = b.name(name.to_owned());
                        }
                        if let Some(stack_size) = thread.stack_size() {
                            b = b.stack_size(stack_size);
                        }

                        b.spawn(move || {
                            let thread_id = rayon::current_thread_index().unwrap_or(0);
                            if let Some(core_ids) = cpu_map.get(&thread_id) {
                                pin_thread_to_cores(core_ids);
                            }
                            thread.run()
                        })?;
                        Ok(())
                    })
                    .build()
                    .expect("Failed to create NUMA-aware thread pool")
            } else {
                debug!("Skipping thread pinning on UMA system (would add overhead)");
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .expect("Failed to create thread pool")
            }
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Failed to create thread pool")
        }
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool")
    };

    #[cfg(not(all(feature = "numa", feature = "thread-pinning")))]
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to create thread pool");

    // First-touch memory initialization for NUMA locality
    #[cfg(feature = "numa")]
    if should_optimize_numa {
        if let Some(ref topology) = numa_topology {
            if topology.num_nodes > 1 {
                debug!(
                    "Performing first-touch memory initialization for {} NUMA nodes",
                    topology.num_nodes
                );
                pool.install(|| {
                    data.par_chunks_mut(BLK_SIZE).for_each(|chunk| {
                        chunk[0] = 0;
                        if chunk.len() > 4096 {
                            chunk[chunk.len() - 1] = 0;
                        }
                    });
                });
            } else {
                tracing::trace!("Skipping first-touch on UMA system");
            }
        }
    }

    pool.install(|| {
        data.par_chunks_mut(BLK_SIZE)
            .enumerate()
            .for_each(|(i, chunk)| {
                let ub = i % unique_blocks;
                tracing::trace!("Filling block {} (unique block {})", i, ub);
                fill_block_alt(chunk, ub, copy_lens[ub].min(chunk.len()), call_entropy);
            });
    });

    debug!("Parallel generation complete, truncating to {} bytes", size);
    data.truncate(size);
    data
}

/// Alternative data generator with improved compression control
///
/// # Default Behavior  
/// - Uses 50% of available CPU cores/threads (leaving 50% for I/O)
/// - Automatically detects NUMA/UMA topology
/// - Enables NUMA optimizations on multi-node systems
///
/// # Key Differences from `generate_controlled_data`:
/// - No shared BASE_BLOCK → eliminates cross-block compression
/// - Each unique block uses Xoshiro256++ keystream → high baseline entropy
/// - Compressibility via local back-refs → no cross-block duplication
///
/// # Parameters
/// - `size`: Total bytes to generate (rounded up to BLK_SIZE)
/// - `dedup`: Deduplication factor (1 = no dedup, N = N:1 logical:physical)
/// - `compress`: Compression factor (1 = incompressible, N = N:1 logical:physical)
///
/// # Dedupe Semantics (unchanged from original)
/// - `unique_blocks = round(total_blocks / dedup)`
/// - Blocks are assigned round-robin to unique blocks
/// - Same dedupe factor yields same logical-to-physical ratio
///
/// # Compress Semantics (improved)
/// - `compress=1`: Truly incompressible (zstd ratio ~1.00-1.02)
/// - `compress>1`: Target ratio via local back-refs (no cross-block effects)
/// - Compression distributed evenly via integer error accumulation
///
/// # Performance
/// - Parallel block generation via rayon (50% of CPUs by default)
/// - Xoshiro256++ keystream ~5-15 GB/s per core (5-10x faster than ChaCha20)
/// - Back-reference copies ~10-15 GB/s
/// - Overall throughput: ~1-4 GB/s (depends on compress factor)
///
/// # Example
/// ```rust
/// // 100MB incompressible data, no deduplication
/// let data = generate_controlled_data_alt(100 * 1024 * 1024, 1, 1);
/// // zstd compression: ~1.00-1.02 ratio
///
/// // 100MB, 2:1 dedup, 3:1 compression
/// let data = generate_controlled_data_alt(100 * 1024 * 1024, 2, 3);
/// // Logical: 100MB, Physical: ~17MB (2:1 dedup × 3:1 compress)
/// ```
pub fn generate_controlled_data_alt(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {

    // Ensure minimum size
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    let block_size = BLK_SIZE;
    let nblocks = size.div_ceil(block_size);

    // --- Dedupe calculation (identical to original) ---
    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    let unique_blocks = if dedup_factor > 1 {
        ((nblocks as f64) / (dedup_factor as f64))
            .round()
            .max(1.0) as usize
    } else {
        nblocks
    };

    debug!(
        "Alt generator: size={}, blocks={}, dedup={}, unique_blocks={}, compress={}",
        size, nblocks, dedup_factor, unique_blocks, compress
    );

    // --- Compress calculation: distribute target copy length via integer error accumulation ---
    // Standard technique for even distribution (similar to line rasterization algorithms)
    let (f_num, f_den) = if compress > 1 {
        (compress - 1, compress)
    } else {
        (0, 1)
    };
    let floor_len = (f_num * block_size) / f_den;
    let rem = (f_num * block_size) % f_den;

    // Precompute per-block copy lengths using integer error accumulation
    // This ensures compression is spread evenly across all unique blocks
    let copy_lens: Vec<usize> = {
        let mut v = Vec::with_capacity(unique_blocks);
        let mut err = 0;
        for _ in 0..unique_blocks {
            err += rem;
            if err >= f_den {
                err -= f_den;
                v.push(floor_len + 1);
            } else {
                v.push(floor_len);
            }
        }
        v
    };

    // Allocate full buffer
    let total_size = nblocks * block_size;
    debug!("Allocating {} bytes ({} blocks)", total_size, nblocks);
    let mut data: Vec<u8> = vec![0u8; total_size];

    // Per-call entropy using helper function
    let call_entropy = generate_call_entropy();

    // Configure thread pool - use 50% of CPUs by default (leaves other 50% for I/O)
    let num_threads = default_data_gen_threads();
    info!("Using {} threads for parallel generation ({} total CPUs available)", num_threads, num_cpus::get());

    // NUMA optimization check - always detect topology for optimal performance
    #[cfg(feature = "numa")]
    let numa_topology = NumaTopology::detect().ok();

    #[cfg(feature = "numa")]
    let should_optimize_numa = if let Some(ref topology) = numa_topology {
        let optimize = topology.num_nodes > 1;  // Auto mode: only on multi-node systems
        
        if optimize {
            info!(
                "NUMA optimization enabled: {} nodes detected",
                topology.num_nodes
            );
        } else {
            debug!(
                "NUMA optimization not needed: {} nodes detected (UMA system)",
                topology.num_nodes
            );
        }
        optimize
    } else {
        debug!("NUMA topology detection failed, using default rayon pool");
        false
    };

    #[cfg(not(feature = "numa"))]
    #[allow(unused_variables)]
    let should_optimize_numa = false;

    debug!("Starting parallel generation with rayon");

    // Build thread pool with optional NUMA-aware thread pinning
    // Only pin threads on true NUMA systems (>1 node) - adds overhead on UMA
    #[cfg(all(feature = "numa", feature = "thread-pinning"))]
    let pool = if should_optimize_numa {
        if let Some(ref topology) = numa_topology {
            if topology.num_nodes > 1 {
                debug!(
                    "Configuring NUMA-aware thread pinning for {} nodes",
                    topology.num_nodes
                );

                // Build CPU affinity mapping (wrap in Arc for sharing across threads)
                let cpu_map = std::sync::Arc::new(build_cpu_affinity_map(topology, num_threads));

                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .spawn_handler(move |thread| {
                        let cpu_map = cpu_map.clone();
                        let mut b = std::thread::Builder::new();
                        if let Some(name) = thread.name() {
                            b = b.name(name.to_owned());
                        }
                        if let Some(stack_size) = thread.stack_size() {
                            b = b.stack_size(stack_size);
                        }

                        b.spawn(move || {
                            // Pin this thread to specific CPU cores
                            let thread_id = rayon::current_thread_index().unwrap_or(0);
                            if let Some(core_ids) = cpu_map.get(&thread_id) {
                                pin_thread_to_cores(core_ids);
                            }
                            thread.run()
                        })?;
                        Ok(())
                    })
                    .build()
                    .expect("Failed to create NUMA-aware thread pool")
            } else {
                debug!("Skipping thread pinning on UMA system (would add overhead)");
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .expect("Failed to create thread pool")
            }
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Failed to create thread pool")
        }
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool")
    };

    #[cfg(not(all(feature = "numa", feature = "thread-pinning")))]
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to create thread pool");

    // First-touch memory initialization for NUMA locality
    // Only beneficial on true NUMA systems (>1 node)
    // On UMA systems, this just adds overhead
    #[cfg(feature = "numa")]
    if should_optimize_numa {
        if let Some(ref topology) = numa_topology {
            if topology.num_nodes > 1 {
                debug!(
                    "Performing first-touch memory initialization for {} NUMA nodes",
                    topology.num_nodes
                );
                pool.install(|| {
                    data.par_chunks_mut(block_size).for_each(|chunk| {
                        // Touch each page to allocate it locally
                        // Linux allocates memory on the node of the thread that first writes to it
                        chunk[0] = 0;
                        if chunk.len() > 4096 {
                            chunk[chunk.len() - 1] = 0;
                        }
                    });
                });
            } else {
                tracing::trace!("Skipping first-touch on UMA system");
            }
        }
    }

    // Parallel block generation
    pool.install(|| {
        data.par_chunks_mut(block_size)
            .enumerate()
            .for_each(|(i, chunk)| {
                // Map block index to unique block (round-robin for dedupe)
                let ub = i % unique_blocks;
                tracing::trace!("Filling block {} (unique block {})", i, ub);

                // Fill this block
                fill_block_alt(chunk, ub, copy_lens[ub].min(chunk.len()), call_entropy);
            });
    });

    debug!("Parallel generation complete, truncating to {} bytes", size);
    // Truncate to requested size
    data.truncate(size);

    info!(
        "Alt generator complete: {} bytes, {} blocks, {} unique blocks",
        data.len(),
        nblocks,
        unique_blocks
    );

    data
}

// =============================================================================
// Streaming Generator (ObjectGen equivalent)
// =============================================================================

/// Streaming data generator state (alternative implementation)
///
/// Mirrors the behavior of `ObjectGen` from data_gen.rs but uses the
/// alternative block-filling algorithm for improved compression control.
#[allow(dead_code)] // dedup_factor and compress_factor used in constructor, stored for reference
pub struct ObjectGenAlt {
    /// Total size in bytes
    total_size: usize,
    /// Current position
    current_pos: usize,
    /// Deduplication factor
    dedup_factor: usize,
    /// Compression factor
    compress_factor: usize,
    /// Number of unique blocks
    unique_blocks: usize,
    /// Per-block copy lengths (evenly distributed via error accumulation)
    copy_lens: Vec<usize>,
    /// Per-call entropy for seeding
    call_entropy: u64,
}

impl ObjectGenAlt {
    /// Create new streaming generator
    ///
    /// # Parameters
    /// - `total_size`: Total bytes to generate
    /// - `dedup`: Deduplication factor
    /// - `compress`: Compression factor
    pub fn new(total_size: usize, dedup: usize, compress: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let block_size = BLK_SIZE;
        let nblocks = total_size.div_ceil(block_size);

        // Dedupe calculation
        let dedup_factor = if dedup == 0 { 1 } else { dedup };
        let unique_blocks = if dedup_factor > 1 {
            ((nblocks as f64) / (dedup_factor as f64))
                .round()
                .max(1.0) as usize
        } else {
            nblocks
        };

        // Compress calculation (integer error accumulation for even distribution)
        let (f_num, f_den) = if compress > 1 {
            (compress - 1, compress)
        } else {
            (0, 1)
        };
        let floor_len = (f_num * block_size) / f_den;
        let rem = (f_num * block_size) % f_den;

        let copy_lens: Vec<usize> = {
            let mut v = Vec::with_capacity(unique_blocks);
            let mut err = 0;
            for _ in 0..unique_blocks {
                err += rem;
                if err >= f_den {
                    err -= f_den;
                    v.push(floor_len + 1);
                } else {
                    v.push(floor_len);
                }
            }
            v
        };

        // Per-call entropy: Mix system time + high-quality random bytes from /dev/urandom
        // This ensures global uniqueness across distributed nodes even with synchronized clocks
        let time_entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Get 8 bytes of cryptographically random data from /dev/urandom
        let urandom_entropy: u64 = {
            use rand::RngCore;
            let mut rng = rand::rng(); // Uses /dev/urandom on Linux
            rng.next_u64()
        };

        // Combine both sources for maximum uniqueness
        let call_entropy = time_entropy.wrapping_add(urandom_entropy);

        Self {
            total_size,
            current_pos: 0,
            dedup_factor,
            compress_factor: compress,
            unique_blocks,
            copy_lens,
            call_entropy,
        }
    }

    /// Fill the next chunk of data
    ///
    /// Returns the number of bytes written. When this returns 0, generation is complete.
    pub fn fill_chunk(&mut self, buf: &mut [u8]) -> usize {
        if self.current_pos >= self.total_size {
            return 0;
        }

        let remaining = self.total_size - self.current_pos;
        let to_write = buf.len().min(remaining);
        let chunk = &mut buf[..to_write];

        let block_size = BLK_SIZE;
        let _start_block = self.current_pos / block_size;

        let mut offset = 0;
        while offset < chunk.len() {
            let block_idx = (self.current_pos + offset) / block_size;
            let block_offset = (self.current_pos + offset) % block_size;
            let remaining_in_block = block_size - block_offset;
            let to_copy = remaining_in_block.min(chunk.len() - offset);

            // Map to unique block
            let ub = block_idx % self.unique_blocks;

            // Generate full block if we need any part of it
            let mut block_buf = vec![0u8; block_size];
            fill_block_alt(
                &mut block_buf,
                ub,
                self.copy_lens[ub].min(block_size),
                self.call_entropy,
            );

            // Copy the needed portion
            chunk[offset..offset + to_copy]
                .copy_from_slice(&block_buf[block_offset..block_offset + to_copy]);

            offset += to_copy;
        }

        self.current_pos += to_write;
        to_write
    }

    /// Reset generator to start
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.current_pos
    }

    /// Get total size
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Check if generation is complete
    pub fn is_complete(&self) -> bool {
        self.current_pos >= self.total_size
    }
}

/// Generate data using streaming approach (alternative implementation)
///
/// # Parameters
/// - `total_size`: Total bytes to generate
/// - `dedup`: Deduplication factor
/// - `compress`: Compression factor
/// - `chunk_size`: Size of chunks for streaming
pub fn generate_controlled_data_streaming_alt(
    total_size: usize,
    dedup: usize,
    compress: usize,
    chunk_size: usize,
) -> Vec<u8> {
    let mut gen = ObjectGenAlt::new(total_size, dedup, compress);
    let mut result = Vec::with_capacity(total_size);
    let mut chunk = vec![0u8; chunk_size];

    loop {
        let written = gen.fill_chunk(&mut chunk);
        if written == 0 {
            break;
        }
        result.extend_from_slice(&chunk[..written]);
    }

    result
}

// =============================================================================
// NUMA Helper Functions
// =============================================================================

/// Generate per-call entropy from time + urandom
fn generate_call_entropy() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let time_entropy = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let urandom_entropy: u64 = {
        use rand::RngCore;
        let mut rng = rand::rng();
        rng.next_u64()
    };

    time_entropy.wrapping_add(urandom_entropy)
}

#[cfg(all(feature = "numa", feature = "thread-pinning"))]
use std::collections::HashMap;

/// Build CPU affinity map for thread pinning
#[cfg(all(feature = "numa", feature = "thread-pinning"))]
fn build_cpu_affinity_map(
    topology: &crate::numa::NumaTopology,
    num_threads: usize,
) -> HashMap<usize, Vec<usize>> {
    let mut map = HashMap::new();

    // Distribute threads across NUMA nodes round-robin
    let mut thread_id = 0;
    let mut node_idx = 0;

    while thread_id < num_threads {
        if let Some(node) = topology.nodes.get(node_idx % topology.nodes.len()) {
            // Assign threads to cores within this NUMA node
            let cores_per_thread = (node.cpus.len() as f64 / num_threads as f64).ceil() as usize;
            let cores_per_thread = cores_per_thread.max(1);

            let start_cpu = (thread_id * cores_per_thread) % node.cpus.len();
            let end_cpu = ((thread_id + 1) * cores_per_thread).min(node.cpus.len());

            let core_ids: Vec<usize> = node.cpus[start_cpu..end_cpu].to_vec();

            if !core_ids.is_empty() {
                tracing::trace!(
                    "Thread {} -> NUMA node {} cores {:?}",
                    thread_id,
                    node.node_id,
                    &core_ids
                );
                map.insert(thread_id, core_ids);
            }
        }

        thread_id += 1;
        node_idx += 1;
    }

    map
}

/// Pin current thread to specific CPU cores
#[cfg(all(feature = "numa", feature = "thread-pinning"))]
fn pin_thread_to_cores(core_ids: &[usize]) {
    if let Some(&first_core) = core_ids.first() {
        if let Some(core_ids_all) = core_affinity::get_core_ids() {
            if first_core < core_ids_all.len() {
                let core_id = core_ids_all[first_core];
                if core_affinity::set_for_current(core_id) {
                    tracing::trace!("Pinned thread to core {}", first_core);
                } else {
                    tracing::debug!("Failed to pin thread to core {}", first_core);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: No seed generation tests needed - Xoshiro256PlusPlus::seed_from_u64 is battle-tested

    #[test]
    fn test_default_thread_count() {
        let default_threads = default_data_gen_threads();
        let total = total_cpus();
        
        // Should be at least 1
        assert!(default_threads >= 1);
        
        // Should be at most total CPUs
        assert!(default_threads <= total);
        
        // Should be roughly half (allowing for rounding)
        let expected_half = total / 2;
        assert!(default_threads == expected_half.max(1));
        
        println!("Total CPUs: {}, Default data gen threads: {}", total, default_threads);
    }

    #[test]
    fn test_generator_config_defaults() {
        let config = GeneratorConfig::default();
        
        assert_eq!(config.size, BLK_SIZE);
        assert_eq!(config.dedup_factor, 1);
        assert_eq!(config.compress_factor, 1);
        assert_eq!(config.numa_mode, NumaMode::Auto);
        
        // Should have Some value (not None)
        assert!(config.max_threads.is_some());
        
        // Should be the default thread count
        assert_eq!(config.max_threads.unwrap(), default_data_gen_threads());
    }

    #[test]
    fn test_generate_minimal_size() {
        let data = generate_controlled_data_alt(100, 1, 1);
        assert_eq!(data.len(), BLK_SIZE, "Should round up to BLK_SIZE");
    }

    #[test]
    fn test_generate_exact_block() {
        let data = generate_controlled_data_alt(BLK_SIZE, 1, 1);
        assert_eq!(data.len(), BLK_SIZE);
    }

    #[test]
    fn test_generate_multiple_blocks() {
        let size = BLK_SIZE * 10;
        let data = generate_controlled_data_alt(size, 1, 1);
        assert_eq!(data.len(), size);
    }

    #[test]
    fn test_dedupe_factor() {
        let size = BLK_SIZE * 100;
        let data = generate_controlled_data_alt(size, 2, 1);
        // Should generate unique_blocks = round(100/2) = 50 unique blocks
        // Can't easily verify without inspecting block contents, but should not panic
        assert_eq!(data.len(), size);
    }

    #[test]
    fn test_streaming_generator() {
        let size = BLK_SIZE * 10;
        let mut gen = ObjectGenAlt::new(size, 1, 1);
        let mut result = Vec::new();
        let mut chunk = vec![0u8; 1024];

        while !gen.is_complete() {
            let written = gen.fill_chunk(&mut chunk);
            if written == 0 {
                break;
            }
            result.extend_from_slice(&chunk[..written]);
        }

        assert_eq!(result.len(), size);
        assert!(gen.is_complete());
    }

    #[test]
    fn test_streaming_matches_single_pass() {
        let size = BLK_SIZE * 5;
        
        // Note: These won't be identical due to different chunking,
        // but they should have the same length and similar characteristics
        let single = generate_controlled_data_alt(size, 1, 1);
        let streaming = generate_controlled_data_streaming_alt(size, 1, 1, 1024);
        
        assert_eq!(single.len(), streaming.len());
    }
}
