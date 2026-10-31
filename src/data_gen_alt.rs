// src/data_gen_alt.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Compatibility shim — all core data-generation logic now lives in the
//! `dgen-data` crate.  This module re-exports everything that was previously
//! implemented here so that other s3dlio modules continue to compile unchanged.
//!
//! New code should import from `dgen_data` directly.

// Re-export the full public API from dgen_data so existing `use crate::data_gen_alt::*`
// imports keep working without any changes.
pub use dgen_data::{
    generate_data, generate_data_simple, DataBuffer, DataGenerator, GeneratorConfig, NumaMode,
};

// ---------------------------------------------------------------------------
// Helper shims — functions that lived here but belong in hardware or are
// trivially delegated.
// ---------------------------------------------------------------------------

/// Total logical CPU count.
#[inline]
pub fn total_cpus() -> usize {
    crate::hardware::total_cpus()
}

/// Recommended data-generation thread count (≈ 50 % of logical CPUs).
#[inline]
pub fn default_data_gen_threads() -> usize {
    crate::hardware::recommended_data_gen_threads(None, None)
}

/// Optimal streaming chunk size for the given total object size.
///
/// - ≥ 64 MiB → 64 MiB
/// - ≥ 32 MiB → 32 MiB
/// - ≥ 16 MiB → 16 MiB
/// - < 16 MiB → `total_size` (single allocation, fast enough)
#[inline]
pub fn optimal_chunk_size(total_size: usize) -> usize {
    if total_size >= 64 * 1024 * 1024 {
        64 * 1024 * 1024
    } else if total_size >= 32 * 1024 * 1024 {
        32 * 1024 * 1024
    } else if total_size >= 16 * 1024 * 1024 {
        16 * 1024 * 1024
    } else {
        total_size
    }
}

/// Single-call generation returning `bytes::Bytes` (zero-copy for UMA).
#[inline]
pub fn generate_data_with_config(config: GeneratorConfig) -> bytes::Bytes {
    // dgen-data's one-shot generator uses rayon's GLOBAL pool; run it inside s3dlio's own,
    // fork-aware pool instead (#297).
    crate::s3_client::with_rayon_pool(|| generate_data(config)).into_bytes()
}

/// First epoch (+1) in which this process lineage created a dgen-data streaming generator
/// (0 = none yet). dgen-data keeps its own process-global rayon pool, built by the first
/// `DataGenerator::new` and NOT fork-aware (a private `OnceLock` in the crate); a streaming
/// generator that runs its parallel path in a fork child of a process that built that pool waits
/// for ever on threads that do not exist there. s3dlio cannot rebuild a pool it does not own.
static DGEN_POOL_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The config for a new dgen-data streaming generator. In a fork child whose ancestor already
/// built dgen-data's pool the generator is limited to ONE thread (its sequential path never
/// touches the dead pool): correct and non-hanging, but slower. The real fix is a fork-aware
/// pool in dgen-data (reported in #297).
pub(crate) fn fork_safe_config(mut config: GeneratorConfig) -> GeneratorConfig {
    use std::sync::atomic::Ordering;
    // A Rust caller that only ever uses streaming generators has touched no `ForkCell`: make
    // sure the fork handler exists, or a child would never see its epoch change.
    crate::fork_guard::register();
    let epoch = crate::fork_guard::epoch();
    match DGEN_POOL_EPOCH.compare_exchange(0, epoch + 1, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {}
        Err(first) if first - 1 < epoch => {
            crate::fork_guard::warn_abandoned();
            config.max_threads = Some(1);
        }
        Err(_) => {}
    }
    config
}

/// Generate `size` bytes with controlled dedup/compress, returning `bytes::Bytes`.
///
/// When `seed` is `Some(s)`, identical seeds produce identical byte sequences.
/// Internally uses `DataGenerator` (streaming path) which honours the seed;
/// the batch `generate_data` function ignores `config.seed` entirely.
#[inline]
pub fn generate_controlled_data_alt(
    size: usize,
    dedup: usize,
    compress: usize,
    seed: Option<u64>,
) -> bytes::Bytes {
    let config = GeneratorConfig {
        size,
        dedup_factor: dedup.max(1),
        compress_factor: compress.max(1),
        seed,
        ..Default::default()
    };
    generate_data_with_config(config)
}

// ---------------------------------------------------------------------------
// ObjectGenAlt — streaming generator shim wrapping dgen_data::DataGenerator.
//
// Preserves the interface used by `data_gen.rs::ObjectGen`.
// ---------------------------------------------------------------------------

/// Streaming per-object generator, delegating to `dgen_data::DataGenerator`.
pub struct ObjectGenAlt {
    inner: DataGenerator,
}

impl ObjectGenAlt {
    /// Create a new streaming generator using system entropy as the seed.
    pub fn new(total_size: usize, dedup: usize, compress: usize) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self::new_with_seed(total_size, dedup, compress, seed)
    }

    /// Create a new streaming generator for one object.
    pub fn new_with_seed(total_size: usize, dedup: usize, compress: usize, seed: u64) -> Self {
        let config = fork_safe_config(GeneratorConfig {
            size: total_size,
            dedup_factor: dedup.max(1),
            compress_factor: compress.max(1),
            seed: Some(seed),
            ..Default::default()
        });
        Self {
            inner: DataGenerator::new(config),
        }
    }

    /// Fill `buf` with the next chunk of generated data.
    ///
    /// Returns the number of bytes written (0 when complete).
    #[inline]
    pub fn fill_chunk(&mut self, buf: &mut [u8]) -> usize {
        self.inner.fill_chunk(buf)
    }

    /// Returns `true` when all bytes for this object have been generated.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Resets the generator back to the start of the object.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset()
    }

    /// Number of bytes already written for this object.
    #[inline]
    pub fn position(&self) -> usize {
        self.inner.position()
    }

    /// Total byte count this object will produce.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.inner.total_size()
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 1024 * 1024; // 1 MiB — matches dgen_data BLOCK_SIZE

    // -------------------------------------------------------------------------
    // generate_data_simple / generate_data_with_config
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_minimal() {
        // generate_data_simple returns exactly the requested size.
        // Use BLOCK (1 MiB) as a convenient round size.
        let data = generate_data_simple(BLOCK, 1, 1);
        assert_eq!(
            data.len(),
            BLOCK,
            "generate_data_simple must return exactly the requested byte count"
        );
    }

    #[test]
    fn test_generate_exact_block() {
        let data = generate_data_simple(BLOCK, 1, 1);
        assert_eq!(data.len(), BLOCK);
    }

    #[test]
    fn test_generate_multiple_blocks() {
        let size = BLOCK * 4;
        let data = generate_data_simple(size, 1, 1);
        assert_eq!(data.len(), size);
    }

    #[test]
    fn test_generate_data_with_config_size() {
        let size = BLOCK * 3;
        let bytes = generate_data_with_config(GeneratorConfig {
            size,
            dedup_factor: 2,
            compress_factor: 2,
            ..Default::default()
        });
        assert_eq!(bytes.len(), size);
    }

    // -------------------------------------------------------------------------
    // generate_controlled_data_alt
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_controlled_data_alt_basic() {
        let size = BLOCK * 2;
        let bytes = generate_controlled_data_alt(size, 1, 1, None);
        assert_eq!(bytes.len(), size);
    }

    #[test]
    fn test_generate_controlled_data_alt_with_dedup_compress() {
        let size = BLOCK * 4;
        let bytes = generate_controlled_data_alt(size, 4, 2, None);
        assert_eq!(bytes.len(), size);
    }

    #[test]
    fn test_generate_controlled_data_alt_seeded_reproducible() {
        let size = BLOCK * 2;
        let a = generate_controlled_data_alt(size, 1, 1, Some(42));
        let b = generate_controlled_data_alt(size, 1, 1, Some(42));
        assert_eq!(a, b, "Same seed must produce identical output");
    }

    #[test]
    fn test_generate_controlled_data_alt_different_seeds() {
        let size = BLOCK * 2;
        let a = generate_controlled_data_alt(size, 1, 1, Some(1));
        let b = generate_controlled_data_alt(size, 1, 1, Some(2));
        assert_ne!(a, b, "Different seeds must produce different output");
    }

    // -------------------------------------------------------------------------
    // ObjectGenAlt — streaming interface
    // -------------------------------------------------------------------------

    #[test]
    fn test_object_gen_alt_streaming_correct_length() {
        let total = BLOCK * 5;
        let chunk_buf_size = BLOCK;

        let mut gen = ObjectGenAlt::new_with_seed(total, 1, 1, 0xdeadbeef);
        let mut collected = 0usize;
        let mut buf = vec![0u8; chunk_buf_size];

        while !gen.is_complete() {
            let n = gen.fill_chunk(&mut buf);
            assert!(n > 0, "fill_chunk returned 0 before is_complete()");
            collected += n;
        }

        assert_eq!(
            collected, total,
            "Streaming must yield exactly {total} bytes"
        );
        assert_eq!(
            gen.fill_chunk(&mut buf),
            0,
            "fill_chunk after complete must return 0"
        );
    }

    #[test]
    fn test_object_gen_alt_small_chunk() {
        // DataGenerator minimum size is BLOCK_SIZE (1 MiB); use BLOCK as total.
        // The test verifies that streaming with a tiny chunk_size buffer works correctly.
        let total = BLOCK;
        let chunk_size = 512;

        let mut gen = ObjectGenAlt::new_with_seed(total, 1, 1, 99);
        let mut buf = vec![0u8; chunk_size];
        let mut collected = 0usize;

        while !gen.is_complete() {
            let n = gen.fill_chunk(&mut buf);
            if n == 0 {
                break;
            }
            collected += n;
        }

        assert_eq!(collected, total);
    }

    // -------------------------------------------------------------------------
    // Utility shims
    // -------------------------------------------------------------------------

    #[test]
    fn test_total_cpus_positive() {
        assert!(total_cpus() > 0);
    }

    #[test]
    fn test_default_data_gen_threads_bounds() {
        let t = default_data_gen_threads();
        assert!(t > 0);
        assert!(t <= total_cpus());
    }

    #[test]
    fn test_optimal_chunk_size_nonzero() {
        assert!(optimal_chunk_size(1024 * 1024) > 0);
    }
}
