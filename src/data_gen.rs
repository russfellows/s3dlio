// src/data_gen.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! s3dlio-specific data generation glue.
//!
//! Core generation logic (DataGenerator, GeneratorConfig, generate_data, …) lives
//! in the `dgen-data` crate and is re-exported from `data_gen_alt`.  This module
//! contains only the s3dlio-specific helpers that wrap Config/ObjectType and the
//! public `fill_controlled_data` API.

use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tracing::{debug, info};

use crate::config::Config;
use crate::config::ObjectType;
use crate::constants::{A_BASE_BLOCK, BASE_BLOCK, BLK_SIZE, HALF_BLK, MOD_SIZE};
#[cfg(feature = "hdf5")]
use crate::data_formats::build_hdf5;
use crate::data_formats::{build_npz, build_raw, build_tfrecord};

// =============================================================================
// Public API: generate_object
// =============================================================================

/// Build an object payload in the requested format.
pub fn generate_object(cfg: &Config) -> anyhow::Result<bytes::Bytes> {
    use crate::config::DataGenMode;

    let total_bytes = cfg.elements * cfg.element_size;
    info!(
        "Generating object: type={:?}, elements={}, element_size={} bytes, total_bytes={}, algorithm={:?}, mode={:?}",
        cfg.object_type, cfg.elements, cfg.element_size, total_bytes, cfg.data_gen_algorithm, cfg.data_gen_mode
    );

    let data = if cfg.use_controlled {
        match cfg.data_gen_mode {
            DataGenMode::Streaming => {
                debug!(
                    "Using streaming controlled data: dedup={}, compress={}, chunk_size={}",
                    cfg.dedup_factor, cfg.compress_factor, cfg.chunk_size
                );
                generate_controlled_data_streaming(
                    total_bytes,
                    cfg.dedup_factor,
                    cfg.compress_factor,
                    cfg.chunk_size,
                )
            }
            DataGenMode::SinglePass => {
                debug!(
                    "Using single-pass controlled data: dedup={}, compress={}",
                    cfg.dedup_factor, cfg.compress_factor
                );
                crate::data_gen_alt::generate_controlled_data_alt(
                    total_bytes,
                    cfg.dedup_factor,
                    cfg.compress_factor,
                    None,
                )
                .to_vec()
            }
        }
    } else {
        debug!("Using generic random data");
        generate_random_data(total_bytes)
    };

    let object = match cfg.object_type {
        ObjectType::Npz => build_npz(cfg.elements, cfg.element_size, &data)?,
        #[cfg(feature = "hdf5")]
        ObjectType::Hdf5 => build_hdf5(cfg.elements, cfg.element_size, &data)?,
        #[cfg(not(feature = "hdf5"))]
        ObjectType::Hdf5 => anyhow::bail!(
            "HDF5 format is not available in this build.\n\
             If you installed via pip, rebuild from source with the hdf5 extra:\n\
             \n\
             pip install --no-binary s3dlio \\'s3dlio[hdf5]\\' \\\n\
             --config-settings cargo-extra-args=\"--features hdf5,extension-module\"\n\
             \n\
             This also requires libhdf5 to be installed on your system:\n\
             Ubuntu/Debian: sudo apt-get install libhdf5-dev\n\
             RHEL/Fedora:   sudo dnf install hdf5-devel\n\
             macOS:         brew install hdf5"
        ),
        ObjectType::TfRecord => build_tfrecord(cfg.elements, cfg.element_size, &data)?,
        ObjectType::Raw => build_raw(&data)?,
    };

    debug!("Generated payload: {} bytes", object.len());
    Ok(object)
}

// =============================================================================
// generate_random_data — simple pseudo-random fallback (no controlled dedup)
// =============================================================================

/// Generates a buffer of `size` random bytes using a per-block BASE_BLOCK template
/// with randomised header/tail bytes.
pub fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    for chunk in data.chunks_mut(BLK_SIZE) {
        let len = chunk.len();
        chunk.copy_from_slice(&BASE_BLOCK[..len]);
    }

    let mut rng = rand::rngs::ThreadRng::default();
    let mut offset = 0;
    while offset < size {
        let block_end = std::cmp::min(offset + BLK_SIZE, size);
        let block_size = block_end - offset;

        if block_size > 0 {
            let first_len = if block_size >= MOD_SIZE { MOD_SIZE } else { block_size };
            rng.fill(&mut data[offset..offset + first_len]);
        }

        if block_size > HALF_BLK {
            rng.fill(&mut data[block_end - MOD_SIZE..block_end]);
        }

        offset += BLK_SIZE;
    }

    data
}

// =============================================================================
// fill_controlled_data — in-place generation using the GLOBAL Rayon pool
// =============================================================================

/// Fill a buffer in-place with controlled random data (dedup/compress).
///
/// **CRITICAL**: This uses the global Rayon pool, which means it respects any
/// `rayon::ThreadPool::install()` context.  Use this when you need to control
/// which thread pool does the parallel work.
///
/// Unlike `generate_data_simple` which allocates a new `Vec<u8>`, this fills
/// an existing buffer — ideal for streaming workloads with pre-allocated buffers.
///
/// # Parameters
/// - `buf`: Mutable buffer to fill with generated data
/// - `dedup`: Deduplication factor (0 or 1 = no dedup, N = N:1 dedup ratio)
/// - `compress`: Compression factor (1 = incompressible, N = N:1 compressible)
pub fn fill_controlled_data(buf: &mut [u8], dedup: usize, compress: usize) {
    use std::time::{SystemTime, UNIX_EPOCH};

    if buf.is_empty() {
        return;
    }

    let size = buf.len();
    let block_size = BLK_SIZE;
    let nblocks = size.div_ceil(block_size);

    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    let unique_blocks = if dedup_factor > 1 {
        ((nblocks as f64) / (dedup_factor as f64)).round().max(1.0) as usize
    } else {
        nblocks
    };

    let (f_num, f_den) = if compress > 1 {
        (compress - 1, compress)
    } else {
        (0, 1)
    };
    let floor_len = (f_num * block_size) / f_den;
    let rem = (f_num * block_size) % f_den;

    let const_lens: Vec<usize> = {
        let mut v = Vec::with_capacity(unique_blocks);
        let mut err_acc = 0;
        for _ in 0..unique_blocks {
            err_acc += rem;
            if err_acc >= f_den {
                err_acc -= f_den;
                v.push(floor_len + 1);
            } else {
                v.push(floor_len);
            }
        }
        v
    };

    let call_entropy = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // FILL IN PLACE using the global Rayon pool (respects install() context!)
    buf.par_chunks_mut(block_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let unique_block_idx = i % unique_blocks;
            let seed = (unique_block_idx as u64).wrapping_add(call_entropy);
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

            let src = &*A_BASE_BLOCK;
            let len = chunk.len();
            chunk.copy_from_slice(&src[..len]);

            let const_len = const_lens[unique_block_idx].min(len);
            chunk[..const_len].fill(0);

            let region_start = const_len;
            let region_len = len - region_start;
            let modify_len = region_len.min(MOD_SIZE);

            if modify_len > 0 {
                rng.fill(&mut chunk[region_start..region_start + modify_len]);
                let second_offset = HALF_BLK.max(region_start);
                if second_offset + modify_len <= len {
                    rng.fill(&mut chunk[second_offset..second_offset + modify_len]);
                }
            }
        });
}

// =============================================================================
// Streaming API — DataGenerator / ObjectGen
// Used by streaming_writer.rs
// =============================================================================

/// Generate a complete object using the streaming DataGenerator.
pub fn generate_controlled_data_streaming(
    size: usize,
    dedup: usize,
    compress: usize,
    chunk_size: usize,
) -> Vec<u8> {
    let generator = DataGenerator::new(None);
    let mut object_gen = generator.begin_object(size, dedup, compress);

    let mut result = Vec::with_capacity(size);
    while !object_gen.is_complete() {
        match object_gen.fill_chunk(chunk_size) {
            Some(chunk) => result.extend_from_slice(&chunk),
            None => break,
        }
    }
    result
}

/// Thin wrapper around `ObjectGenAlt` providing the `new(seed)` / `begin_object()` API
/// used by `streaming_writer.rs`.
pub struct DataGenerator {
    instance_entropy: u64,
}

impl DataGenerator {
    /// Create a new `DataGenerator`.
    ///
    /// - `seed`: `None` = use system entropy (unique per instance).
    ///           `Some(s)` = reproducible data.
    pub fn new(seed: Option<u64>) -> Self {
        Self::new_impl(seed)
    }

    /// Convenience constructor for reproducible data (equivalent to `new(Some(seed))`).
    pub fn new_with_seed(seed: u64) -> Self {
        Self::new(Some(seed))
    }

    fn new_impl(seed: Option<u64>) -> Self {
        let instance_entropy = match seed {
            Some(s) => s,
            None => {
                use std::cell::Cell;
                use std::time::{SystemTime, UNIX_EPOCH};
                thread_local! {
                    static ENTROPY_COUNTER: Cell<u64> = const { Cell::new(0) };
                }
                let base = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let counter = ENTROPY_COUNTER.with(|c| {
                    let val = c.get();
                    c.set(val.wrapping_add(1));
                    val
                });
                base.wrapping_add(counter)
            }
        };
        Self { instance_entropy }
    }

    /// Begin streaming generation of one object.
    pub fn begin_object(&self, size: usize, dedup: usize, compress: usize) -> ObjectGen {
        ObjectGen::from_alt(size, dedup, compress, self.instance_entropy)
    }
}

impl Default for DataGenerator {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Per-object streaming generator (wraps `data_gen_alt::ObjectGenAlt`).
pub struct ObjectGen {
    alt_gen: crate::data_gen_alt::ObjectGenAlt,
}

impl ObjectGen {
    fn from_alt(total_size: usize, dedup: usize, compress: usize, call_entropy: u64) -> Self {
        Self {
            alt_gen: crate::data_gen_alt::ObjectGenAlt::new_with_seed(
                total_size,
                dedup,
                compress,
                call_entropy,
            ),
        }
    }

    /// Fill the next `chunk_size` bytes of data.
    ///
    /// Returns `None` when all data for this object has been generated.
    pub fn fill_chunk(&mut self, chunk_size: usize) -> Option<Vec<u8>> {
        assert!(chunk_size > 0, "Chunk size must be greater than 0");
        let mut buf = vec![0u8; chunk_size];
        let written = self.alt_gen.fill_chunk(&mut buf);
        if written == 0 {
            return None;
        }
        buf.truncate(written);
        Some(buf)
    }

    /// Returns `true` when the full object has been generated.
    pub fn is_complete(&self) -> bool {
        self.alt_gen.is_complete()
    }

    /// Resets this object generator to the start so data can be re-read.
    pub fn reset(&mut self) {
        self.alt_gen.reset();
    }

    /// Number of bytes generated so far for this object.
    pub fn position(&self) -> usize {
        self.alt_gen.position()
    }

    /// Total byte count this object will produce.
    pub fn total_size(&self) -> usize {
        self.alt_gen.total_size()
    }

    /// Generate all remaining bytes in one call and return them as `Vec<u8>`.
    pub fn fill_remaining(&mut self) -> Vec<u8> {
        const CHUNK: usize = 32 * 1024 * 1024; // 32 MiB
        let remaining = self.total_size().saturating_sub(self.position());
        let mut result = Vec::with_capacity(remaining);
        while !self.is_complete() {
            match self.fill_chunk(CHUNK) {
                Some(chunk) => result.extend_from_slice(&chunk),
                None => break,
            }
        }
        result
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 1024 * 1024; // 1 MiB

    // -------------------------------------------------------------------------
    // generate_random_data
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_random_data_size() {
        let data = generate_random_data(BLOCK * 3);
        assert_eq!(data.len(), BLOCK * 3);
    }

    #[test]
    fn test_generate_random_data_non_uniform() {
        // Data should not be all-zero (extremely unlikely with random fill)
        let data = generate_random_data(BLOCK);
        let non_zero = data.iter().filter(|&&b| b != 0).count();
        assert!(non_zero > 0, "Random data should not be all zeros");
    }

    // -------------------------------------------------------------------------
    // fill_controlled_data
    // -------------------------------------------------------------------------

    #[test]
    fn test_fill_controlled_data_size_preserved() {
        let mut buf = vec![0u8; BLOCK * 2];
        fill_controlled_data(&mut buf, 1, 1);
        assert_eq!(buf.len(), BLOCK * 2, "fill_controlled_data must not change buffer length");
    }

    #[test]
    fn test_fill_controlled_data_produces_non_zero() {
        let mut buf = vec![0u8; BLOCK];
        fill_controlled_data(&mut buf, 1, 1);
        let non_zero = buf.iter().filter(|&&b| b != 0).count();
        assert!(non_zero > 0, "fill_controlled_data produced all zeros");
    }

    #[test]
    fn test_fill_controlled_data_two_calls_differ() {
        let mut a = vec![0u8; BLOCK];
        let mut b = vec![0u8; BLOCK];
        fill_controlled_data(&mut a, 1, 1);
        fill_controlled_data(&mut b, 1, 1);
        // Each call uses current-time entropy so they should differ
        // (not a strict guarantee but overwhelmingly likely in practice)
        let differs = a.iter().zip(b.iter()).any(|(x, y)| x != y);
        assert!(differs, "Two fill_controlled_data calls should produce different data");
    }

    #[test]
    fn test_fill_controlled_data_empty_buf() {
        // Should not panic
        let mut buf: Vec<u8> = vec![];
        fill_controlled_data(&mut buf, 1, 1);
    }

    #[test]
    fn test_fill_controlled_data_with_dedup() {
        let mut buf = vec![0u8; BLOCK * 4];
        fill_controlled_data(&mut buf, 4, 1);
        assert_eq!(buf.len(), BLOCK * 4);
    }

    #[test]
    fn test_fill_controlled_data_with_compress() {
        let mut buf = vec![0u8; BLOCK * 2];
        fill_controlled_data(&mut buf, 1, 4);
        assert_eq!(buf.len(), BLOCK * 2);
    }

    // -------------------------------------------------------------------------
    // generate_controlled_data_streaming
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_controlled_data_streaming_size() {
        let size = BLOCK * 5;
        let data = generate_controlled_data_streaming(size, 1, 1, BLOCK);
        assert_eq!(data.len(), size);
    }

    #[test]
    fn test_generate_controlled_data_streaming_small_chunk() {
        let size = BLOCK * 3;
        let data = generate_controlled_data_streaming(size, 1, 1, 4096);
        assert_eq!(data.len(), size);
    }

    // -------------------------------------------------------------------------
    // DataGenerator / ObjectGen (streaming_writer.rs interface)
    // -------------------------------------------------------------------------

    #[test]
    fn test_data_generator_begin_object_full_read() {
        let total = BLOCK * 4;
        let gen = DataGenerator::new(None);
        let mut obj = gen.begin_object(total, 1, 1);

        let mut collected = 0usize;
        while !obj.is_complete() {
            match obj.fill_chunk(BLOCK) {
                Some(chunk) => {
                    assert!(!chunk.is_empty());
                    collected += chunk.len();
                }
                None => break,
            }
        }

        assert_eq!(collected, total);
        assert!(obj.is_complete());
        assert!(obj.fill_chunk(BLOCK).is_none(), "fill_chunk after complete must return None");
    }

    #[test]
    fn test_data_generator_seeded() {
        let total = BLOCK * 2;
        let gen = DataGenerator::new(Some(42));
        let mut obj = gen.begin_object(total, 1, 1);

        let mut data = Vec::with_capacity(total);
        while let Some(chunk) = obj.fill_chunk(BLOCK) {
            data.extend_from_slice(&chunk);
        }
        assert_eq!(data.len(), total);
    }

    #[test]
    fn test_data_generator_default() {
        // Default should not panic and should produce data
        let gen = DataGenerator::default();
        let mut obj = gen.begin_object(4096, 1, 1);
        let chunk = obj.fill_chunk(4096).expect("First fill_chunk must return data");
        assert_eq!(chunk.len(), 4096);
    }
}
