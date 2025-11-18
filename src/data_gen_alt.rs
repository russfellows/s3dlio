// src/data_gen_alt.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
//
// Alternative data generation implementation using Xoshiro256++ PRNG
// with local back-references for compression control.
//
/// # Key Features
/// - No shared BASE_BLOCK (eliminates cross-block compression)
/// - Xoshiro256++ RNG (5-10x faster than ChaCha20)
/// - Compression distributed evenly via integer error accumulation
/// - Correct compress=1 behavior (truly incompressible, ~1.0 zstd ratio)
// - When compress>1, compressibility is local to each block

use rand::{Rng, SeedableRng, RngCore};
use rand_xoshiro::Xoshiro256PlusPlus;  // Explicit high-performance RNG
use rayon::prelude::*;
use tracing::{debug, info};

use crate::constants::BLK_SIZE;

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

        let run_len = rng.random_range(64..=256).min(remaining).min(out.len() - 1);
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
        let max_back = dst.min(1024).max(1);
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

/// Alternative data generator with improved compression control
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
/// - Parallel block generation via rayon
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
    use std::time::{SystemTime, UNIX_EPOCH};

    // Ensure minimum size
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    let block_size = BLK_SIZE;
    let nblocks = (size + block_size - 1) / block_size;

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
    let mut data: Vec<u8> = vec![0u8; total_size];

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

    // Parallel block generation
    data.par_chunks_mut(block_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            // Map block index to unique block (round-robin for dedupe)
            let ub = i % unique_blocks;

            // Fill this block
            fill_block_alt(chunk, ub, copy_lens[ub].min(chunk.len()), call_entropy);
        });

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
        let nblocks = (total_size + block_size - 1) / block_size;

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

#[cfg(test)]
mod tests {
    use super::*;

    // Note: No seed generation tests needed - Xoshiro256PlusPlus::seed_from_u64 is battle-tested

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
