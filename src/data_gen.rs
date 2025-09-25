// src/data_gen.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 

//use anyhow::{Result, bail};

use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use log::{info, debug};

use crate::constants::{BLK_SIZE, HALF_BLK, MOD_SIZE, A_BASE_BLOCK, BASE_BLOCK};
use crate::data_formats::{build_npz, build_hdf5, build_tfrecord, build_raw};
use crate::config::Config;
use crate::config::ObjectType;

// -----------------------------------------------------------------------------
// Generate a buffer of random bytes.
// -----------------------------------------------------------------------------
// Data generation constants moved to src/constants.rs



// -------------------------------------------------------------
// Public API to generate a specific object type, of a given size
// -------------------------------------------------------------
/// A function to build objects in the correct format
pub fn generate_object(cfg: &Config) -> anyhow::Result<bytes::Bytes> {
    use crate::config::DataGenMode;
    
    let total_bytes = cfg.elements * cfg.element_size;
    info!(
        "Generating object: type={:?}, elements={}, element_size={} bytes, total_bytes={}, mode={:?}",
        cfg.object_type, cfg.elements, cfg.element_size, total_bytes, cfg.data_gen_mode
    );
    
    let data = if cfg.use_controlled {
        match cfg.data_gen_mode {
            DataGenMode::Streaming => {
                debug!("Using streaming controlled data: dedup={}, compress={}, chunk_size={}", 
                    cfg.dedup_factor, cfg.compress_factor, cfg.chunk_size);
                generate_controlled_data_streaming(total_bytes, cfg.dedup_factor, cfg.compress_factor, cfg.chunk_size)
            },
            DataGenMode::SinglePass => {
                debug!("Using single-pass controlled data: dedup={}, compress={}", 
                    cfg.dedup_factor, cfg.compress_factor);
                generate_controlled_data(total_bytes, cfg.dedup_factor, cfg.compress_factor)
            }
        }
    } else {
        debug!("Using generic random data");
        generate_random_data(total_bytes)
    };

    // Old
    /*
     * let object: Bytes = match cfg.format.as_str() {
        "NPZ"      => build_npz(cfg.elements, cfg.element_size, &data)?,
        "HDF5"     => build_hdf5(cfg.elements, cfg.element_size, &data)?,
        "TFRecord" => build_tfrecord(cfg.elements, cfg.element_size, &data)?,
        "RAW"      => build_raw(&data)?,                                    // raw passthrough
    };
    */

    // New, uses type
    let object = match cfg.object_type {
        ObjectType::Npz      => build_npz(cfg.elements, cfg.element_size, &data)?,
        ObjectType::Hdf5     => build_hdf5(cfg.elements, cfg.element_size, &data)?,
        ObjectType::TfRecord => build_tfrecord(cfg.elements, cfg.element_size, &data)?,
        ObjectType::Raw      => build_raw(&data)?,
    };

    debug!("Generated paylod: {} bytes", object.len());
    Ok(object)
}

/*
 * Not used, instead we use the generate_object() function above
 *
// For now, each of our 4 object types just calls the same function
pub fn generate_npz(size: usize) -> Vec<u8> {
    //generate_random_data(size)
    generate_controlled_data(size, 1, 1)
} 

pub fn generate_tfrecord(size: usize) -> Vec<u8> {
    //generate_random_data(size)
    generate_controlled_data(size, 1, 1)
} 

pub fn generate_hdf5(size: usize) -> Vec<u8> {
    //generate_random_data(size)
    generate_controlled_data(size, 1, 1)
} 

pub fn generate_raw_data(size: usize) -> Vec<u8> {
    //generate_random_data(size)
    generate_controlled_data(size, 1, 1)
} 
*
*/

/// Generates a buffer of `size` random bytes by:
/// 1. Enforcing a minimum size of BLK_SIZE bytes.
/// 2. Filling each BLK_SIZE-byte block with a static base block.
/// 3. Modifying the first MOD_SIZE bytes of each block,
///    and modifying the last MOD_SIZE bytes only if the block is larger than 128 bytes.
///
/// This ensures each BLK_SIZE-byte block is unique while avoiding the need to generate a whole new
/// random buffer on every call.
pub fn generate_random_data(mut size: usize) -> Vec<u8> {
    // Enforce a minimum size of BLK_SIZE bytes.
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    // Allocate the buffer.
    let mut data = vec![0u8; size];

    // Fill each BLK_SIZE-byte block by copying from the static base block.
    for chunk in data.chunks_mut(BLK_SIZE) {
        let len = chunk.len();
        chunk.copy_from_slice(&BASE_BLOCK[..len]);
    }

    let mut rng = rand::rngs::ThreadRng::default();
    let mut offset = 0;
    while offset < size {
        let block_end = std::cmp::min(offset + BLK_SIZE, size);
        let block_size = block_end - offset;

        // Modify the first MOD_SIZE bytes (or the full block if it's smaller).
        if block_size > 0 {
            let first_len = if block_size >= MOD_SIZE { MOD_SIZE } else { block_size };
            rng.fill(&mut data[offset .. offset + first_len]);
        }

        // Modify the last MOD_SIZE bytes only if the block is larger than 128 bytes.
        if block_size > HALF_BLK {
            rng.fill(&mut data[block_end - MOD_SIZE .. block_end]);
        }

        offset += BLK_SIZE;
    }

    data
}


/// Generates a buffer of `size` bytes with controlled deduplication and compressibility.
///
/// # Parameters
/// - `size`: The total size (in bytes) of the returned buffer; if less than BLK_SIZE, it is raised to BLK_SIZE.
/// - `dedup`: The deduplication factor. For example, dedup = 3 means that roughly one out of every 3 blocks
///   is unique; a value of 1 produces fully unique blocks. (If dedup is 0, it is treated as 1.)
/// - `compress`: The compressibility factor. For values greater than 1, each BLK_SIZE-byte block is generated so that
///   a fraction f = (compress - 1) / compress of the block is constant while the rest is random. For instance,
///   compress = 2 produces roughly 50% constant data per block (and 50% random), making the block roughly 2:1 compressible.
///   A value of 1 produces all-random data.
///
/// # Implementation Details
/// - The function works on fixed BLK_SIZE-byte blocks. A minimal size of BLK_SIZE bytes is enforced.
/// - It builds a base block according to the compress parameter: the first `constant_length` bytes are constant
///   (set here to zero) and the remainder of the block is random.
/// - Then it creates a set of "unique" blocks by cloning the base block and modifying only a small portion:
///   the first MOD_SIZE bytes, and if the block is larger than 128 bytes, also the last MOD_SIZE bytes.
/// - Finally, the unique blocks are repeated in round-robin order to fill the requested output size.
/// - The final assembly step is parallelized using Rayon for efficiency on large buffers.
///
/// # Returns
/// A `Vec<u8>` with the requested size, containing data with the specified deduplication and compressibility characteristics.

/// Enhanced single-pass data generation function that supports specifying deduplication and compression ratios.
/// 
/// This version eliminates the intermediate `unique` vector and generates data directly into the final buffer
/// in a single pass, while preserving exact dedup/compress ratios and uniqueness guarantees.
///
/// # Performance improvements over the two-pass version:
/// - Eliminates intermediate allocations (no `unique` vector)
/// - Reduces memory passes from 2 to 1  
/// - Uses deterministic per-block seeding for deduplication consistency
/// - Maintains identical dedup/compress math and block-level uniqueness
/// Generate controlled data using streaming approach for optimal performance
/// Based on benchmarks showing streaming is faster for most workload sizes
pub fn generate_controlled_data_streaming(size: usize, dedup: usize, compress: usize, chunk_size: usize) -> Vec<u8> {
    let generator = DataGenerator::new();
    let mut object_gen = generator.begin_object(size, dedup, compress);
    
    let mut result = Vec::with_capacity(size);
    
    while !object_gen.is_complete() {
        let remaining = object_gen.total_size() - object_gen.position();
        let current_chunk_size = remaining.min(chunk_size);
        
        if let Some(chunk) = object_gen.fill_chunk(current_chunk_size) {
            result.extend_from_slice(&chunk);
        }
    }
    
    result
}

pub fn generate_controlled_data(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {
    use rand::Rng;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Enforce a minimum size of BLK_SIZE bytes.
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    let block_size = BLK_SIZE;
    let nblocks = (size + block_size - 1) / block_size;

    // Determine deduplication factor and number of unique blocks (identical to original)
    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    let unique_blocks = if dedup_factor > 1 {
        ((nblocks as f64) / (dedup_factor as f64)).round().max(1.0) as usize
    } else {
        nblocks
    };

    // Prepare parameters for compression: fraction = (compress-1)/compress (identical to original)
    let (f_num, f_den) = if compress > 1 {
        (compress - 1, compress)
    } else {
        (0, 1)
    };
    let floor_len = (f_num * block_size) / f_den;
    let rem = (f_num * block_size) % f_den;

    // Precompute zero-prefix lengths per unique block (Bresenham distribution - identical to original)
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

    // SINGLE-PASS OPTIMIZATION: Generate directly into final buffer
    let total_size = nblocks * block_size;
    let mut data: Vec<u8> = Vec::with_capacity(total_size);
    unsafe {
        data.set_len(total_size);
    }

    // Generate call-specific entropy while preserving deduplication within this call
    let call_entropy = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // Generate each block directly in place - FIXED to properly handle deduplication
    data.par_chunks_mut(block_size).enumerate().for_each(|(i, chunk)| {
        // Use deterministic RNG seeded by unique block index for proper deduplication
        let unique_block_idx = i % unique_blocks;
        // Add call-specific entropy to differentiate between different calls
        // while maintaining identical blocks within the same call for deduplication
        let seed = (unique_block_idx as u64).wrapping_add(call_entropy);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

        // 1) Copy from shared base block (identical to original)
        let src = &*A_BASE_BLOCK;
        let len = chunk.len();
        chunk.copy_from_slice(&src[..len]);

        // 2) Apply zero-prefix using same Bresenham distribution (identical to original)
        let const_len = const_lens[unique_block_idx].min(len);
        chunk[..const_len].fill(0);

        // 3) Inject uniqueness in same two regions (identical to original)
        // CRITICAL FIX: Only blocks with same unique_block_idx get same randomness
        let region_start = const_len;
        let region_len = len - region_start;
        let modify_len = region_len.min(MOD_SIZE);

        if modify_len > 0 {
            // First region: at start of random region
            rng.fill(&mut chunk[region_start..region_start + modify_len]);
            
            // Second region: at HALF_BLK offset if it fits
            let second_offset = HALF_BLK.max(region_start);
            if second_offset + modify_len <= len {
                rng.fill(&mut chunk[second_offset..second_offset + modify_len]);
            }
        }
    });

    // Trim to exact size (identical to original)
    data.truncate(size);
    data
}


/// Two-pass version (original implementation) - kept for testing parity
pub fn generate_controlled_data_two_pass(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {
    // Enforce a minimum size of BLK_SIZE bytes.
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    let block_size = BLK_SIZE;
    let nblocks = (size + block_size - 1) / block_size;

    // Determine deduplication: target ratio = 1/dedup_factor
    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    // Round to nearest number of unique blocks for better approximation
    let unique_blocks = if dedup_factor > 1 {
        // Round(nblocks/dedup_factor)
        let ub = ((nblocks as f64) / (dedup_factor as f64)).round() as usize;
        ub.max(1)
    } else {
        nblocks
    };

    // Prepare parameters for compression spread across blocks
    let (f_num, f_den) = if compress > 1 {
        (compress - 1, compress)
    } else {
        (0, 1)
    };
    // Base zero prefix length (floor)
    let floor_len = (f_num * block_size) / f_den;
    // Remainder gives extra zeros to distribute
    let rem = (f_num * block_size) % f_den;

    // Generate unique blocks with per-block varying zero-prefix lengths
    let mut unique: Vec<Vec<u8>> = Vec::with_capacity(unique_blocks);
    {
        let mut rng = rand::rngs::ThreadRng::default();
        let mut err_acc = 0;

        for _ in 0..unique_blocks {
            // Start from the base random block
            let mut block = BASE_BLOCK.clone();

            // Determine this block's constant prefix length via Bresenham error accumulation
            err_acc += rem;
            let const_len = if err_acc >= f_den {
                err_acc -= f_den;
                floor_len + 1
            } else {
                floor_len
            };

            // Zero out the constant prefix
            for j in 0..const_len {
                block[j] = 0;
            }

            // Compute region for unique modifications
            let region_start = const_len;
            let region_len = block_size - region_start;
            let modify_len = std::cmp::min(MOD_SIZE, region_len);

            // 1) Modify at the start of the random region
            rng.fill(&mut block[region_start..region_start + modify_len]);

            // 2) Optionally modify a second chunk (if it fits within the random region)
            let second_offset = std::cmp::max(HALF_BLK, region_start);
            if second_offset + modify_len <= block_size {
                rng.fill(&mut block[second_offset..second_offset + modify_len]);
            }

            unique.push(block);
        }
    }

    // Build the final output buffer in parallel
    let total_size = nblocks * block_size;
    let mut data = vec![0u8; total_size];
    data.par_chunks_mut(block_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = i % unique.len();
            chunk.copy_from_slice(&unique[idx]);
        });

    // Trim to exact requested size (may leave a partial block)
    data.truncate(size);
    data
}

// =============================================================================
// STREAMING DATA GENERATION API (v0.8.1+ Enhancement)
// =============================================================================

/// Object-scoped data generator that maintains consistent block indexing across streaming chunks.
/// 
/// This enables streaming data generation while preserving exact deduplication and compression 
/// semantics. Each object maintains a global block index that ensures identical dedup/compress
/// ratios regardless of how the data is chunked during streaming.
pub struct DataGenerator {
    /// Instance-specific entropy generated at creation time to differentiate between different 
    /// DataGenerator instances while maintaining deterministic behavior for the same instance
    instance_entropy: u64,
}

impl DataGenerator {
    /// Create a new DataGenerator.
    /// 
    /// Each instance gets unique entropy at creation time, ensuring different
    /// generators produce different data while the same generator remains
    /// deterministic for the same parameters.
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        // Generate instance-specific entropy at creation time
        // Add thread-local counter to ensure uniqueness even with rapid creation
        use std::cell::Cell;
        thread_local! {
            static ENTROPY_COUNTER: Cell<u64> = Cell::new(0);
        }
        
        let base_entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        let counter = ENTROPY_COUNTER.with(|c| {
            let val = c.get();
            c.set(val.wrapping_add(1));
            val
        });
        
        let instance_entropy = base_entropy.wrapping_add(counter);
            
        Self { instance_entropy }
    }    /// Begin generating a new object with the specified parameters.
    /// 
    /// This creates an ObjectGen instance that can generate data in chunks
    /// while maintaining exact compatibility with generate_controlled_data.
    /// Uses the instance's entropy to ensure deterministic behavior.
    /// 
    /// # Parameters
    /// - `size`: Total size of the object in bytes
    /// - `dedup`: Deduplication factor (0 treated as 1)
    /// - `compress`: Compression factor for controllable compressibility
    /// 
    /// # Returns
    /// An ObjectGen instance for streaming generation of the object
    pub fn begin_object(&self, size: usize, dedup: usize, compress: usize) -> ObjectGen {
        ObjectGen::new(size, dedup, compress, self.instance_entropy)
    }
}

impl Default for DataGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-object generator that maintains state for streaming generation of a single object.
/// 
/// This struct tracks the global block index and dedup/compress parameters to ensure
/// that chunks generated in any order produce the same result as if the entire object
/// was generated at once.
pub struct ObjectGen {
    /// Total size of the object being generated
    total_size: usize,
    /// Number of unique blocks (for deduplication)
    unique_blocks: usize,
    /// Precomputed compression lengths per unique block (Bresenham distribution)
    const_lens: Vec<usize>,
    /// Call entropy from the parent DataGenerator
    call_entropy: u64,
    /// Current position (in bytes) for sequential generation
    current_pos: usize,
}

impl ObjectGen {
    /// Create a new ObjectGen for the specified object parameters.
    /// 
    /// This performs the same dedup/compress calculations as generate_controlled_data
    /// but stores them for use across multiple chunk generations.
    fn new(mut total_size: usize, dedup: usize, compress: usize, call_entropy: u64) -> Self {
        // Enforce minimum size (same as generate_controlled_data)
        if total_size < BLK_SIZE {
            total_size = BLK_SIZE;
        }
        
        let total_blocks = (total_size + BLK_SIZE - 1) / BLK_SIZE;
        
        // Calculate deduplication (identical to generate_controlled_data)
        let dedup_factor = if dedup == 0 { 1 } else { dedup };
        let unique_blocks = if dedup_factor > 1 {
            ((total_blocks as f64) / (dedup_factor as f64)).round().max(1.0) as usize
        } else {
            total_blocks
        };
        
        // Calculate compression parameters (identical to generate_controlled_data)
        let (f_num, f_den) = if compress > 1 {
            (compress - 1, compress)
        } else {
            (0, 1)
        };
        let floor_len = (f_num * BLK_SIZE) / f_den;
        let rem = (f_num * BLK_SIZE) % f_den;
        
        // Precompute Bresenham distribution (identical to generate_controlled_data)
        let mut const_lens = Vec::with_capacity(unique_blocks);
        let mut err_acc = 0;
        for _ in 0..unique_blocks {
            err_acc += rem;
            if err_acc >= f_den {
                err_acc -= f_den;
                const_lens.push(floor_len + 1);
            } else {
                const_lens.push(floor_len);
            }
        }
        
        Self {
            total_size,
            unique_blocks,
            const_lens,
            call_entropy,
            current_pos: 0,
        }
    }
    
    /// Fill a chunk with generated data starting at the current position.
    /// 
    /// This generates data with the same semantics as generate_controlled_data but allows
    /// for arbitrary chunk sizes. The chunk_size should ideally be a multiple of BLK_SIZE
    /// for optimal performance, but any size is supported.
    /// 
    /// # Parameters
    /// - `chunk_size`: Size of the chunk to generate
    /// 
    /// # Returns
    /// A Vec<u8> with generated data, or None if all data has been generated
    /// 
    /// # Panics
    /// Panics if chunk_size is 0
    pub fn fill_chunk(&mut self, chunk_size: usize) -> Option<Vec<u8>> {
        assert!(chunk_size > 0, "Chunk size must be greater than 0");
        
        // Check if we've reached the end of the object
        if self.current_pos >= self.total_size {
            return None;
        }
        
        // Determine actual chunk size (may be smaller at end of object)
        let remaining = self.total_size - self.current_pos;
        let actual_chunk_size = chunk_size.min(remaining);
        
        // Create buffer for the chunk
        let mut chunk_data = vec![0u8; actual_chunk_size];
        
        // Find which blocks intersect with this chunk
        let start_block = self.current_pos / BLK_SIZE;
        let end_pos = self.current_pos + actual_chunk_size;
        let end_block = (end_pos + BLK_SIZE - 1) / BLK_SIZE;
        
        // Generate data block by block to match single-pass logic exactly
        for block_idx in start_block..end_block {
            let block_start_global = block_idx * BLK_SIZE;
            let block_end_global = (block_start_global + BLK_SIZE).min(self.total_size);
            
            // Find intersection with current chunk
            let chunk_start_in_block = self.current_pos.max(block_start_global);
            let chunk_end_in_block = end_pos.min(block_end_global);
            
            if chunk_start_in_block < chunk_end_in_block {
                // Generate full block using EXACT same logic as single-pass
                let unique_block_idx = block_idx % self.unique_blocks;
                let seed = (unique_block_idx as u64).wrapping_add(self.call_entropy);
                let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                
                // 1) Start with base block data
                let block_size = block_end_global - block_start_global;
                let mut block_data = A_BASE_BLOCK[..block_size].to_vec();
                
                // 2) Apply compression (zero prefix) - identical to single-pass
                let const_len = self.const_lens[unique_block_idx].min(block_size);
                block_data[..const_len].fill(0);
                
                // 3) Apply uniqueness modifications - IDENTICAL to single-pass
                let region_start = const_len;
                let region_len = block_size - region_start;
                let modify_len = region_len.min(MOD_SIZE);
                
                if modify_len > 0 {
                    // First region: at start of random region - EXACT same as single-pass
                    rng.fill(&mut block_data[region_start..region_start + modify_len]);
                    
                    // Second region: at HALF_BLK offset if it fits - EXACT same as single-pass
                    let second_offset = HALF_BLK.max(region_start);
                    if second_offset + modify_len <= block_size {
                        rng.fill(&mut block_data[second_offset..second_offset + modify_len]);
                    }
                }
                
                // Copy intersection to chunk
                let chunk_offset_start = chunk_start_in_block - self.current_pos;
                let chunk_offset_end = chunk_end_in_block - self.current_pos;
                let block_offset_start = chunk_start_in_block - block_start_global;
                let block_offset_end = chunk_end_in_block - block_start_global;
                
                chunk_data[chunk_offset_start..chunk_offset_end]
                    .copy_from_slice(&block_data[block_offset_start..block_offset_end]);
            }
        }
        
        // Update position for next chunk
        self.current_pos += actual_chunk_size;
        
        Some(chunk_data)
    }
    
    /// Fill all remaining chunks and collect into a single buffer.
    /// 
    /// This is a convenience method that generates all remaining data at once.
    /// Equivalent to calling fill_chunk repeatedly with a large chunk size.
    /// 
    /// # Returns
    /// All remaining data as a single Vec<u8>, or empty Vec if no data remains
    pub fn fill_remaining(&mut self) -> Vec<u8> {
        let remaining = self.total_size - self.current_pos;
        if remaining == 0 {
            return Vec::new();
        }
        
        self.fill_chunk(remaining).unwrap_or_default()
    }
    
    /// Get the current position in the object (bytes generated so far).
    pub fn position(&self) -> usize {
        self.current_pos
    }
    
    /// Get the total size of the object being generated.
    pub fn total_size(&self) -> usize {
        self.total_size
    }
    
    /// Check if all data has been generated.
    pub fn is_complete(&self) -> bool {
        self.current_pos >= self.total_size
    }
    
    /// Reset the generator to the beginning of the object.
    /// 
    /// This allows re-generating the same object data, useful for retries or testing.
    pub fn reset(&mut self) {
        self.current_pos = 0;
    }
}



