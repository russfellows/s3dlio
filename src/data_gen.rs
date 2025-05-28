// src/data_gen.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 

//use anyhow::{Result, bail};

use once_cell::sync::Lazy;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use log::{info, debug};

use crate::data_formats::{build_npz, build_hdf5, build_tfrecord, build_raw};
use crate::config::Config;
use crate::config::ObjectType;

// -----------------------------------------------------------------------------
// Generate a buffer of random bytes.
// -----------------------------------------------------------------------------
//
pub const BLK_SIZE: usize = 512; // This is used elsewhere, hence pub
const HALF_BLK: usize = BLK_SIZE / 2;
const MOD_SIZE: usize = 32;

/// New version
/// A base random block of BLK_SIZE bytes, generated once and shared.
static A_BASE_BLOCK: Lazy<Arc<Vec<u8>>> = Lazy::new(|| {
    let mut block = vec![0u8; BLK_SIZE];
    rand::rngs::ThreadRng::default().fill(&mut block[..]);
    Arc::new(block)
});

/// Original version
/// A base random block of BLK_SIZE bytes, generated once.
static BASE_BLOCK: Lazy<Vec<u8>> = Lazy::new(|| {
    let mut block = vec![0u8; BLK_SIZE];
    rand::rngs::ThreadRng::default().fill(&mut block[..]);
    block
});

// -------------------------------------------------------------
// Public API to generate a specific object type, of a given size
// -------------------------------------------------------------
/// A function to build objects in the correct format
pub fn generate_object(cfg: &Config) -> anyhow::Result<bytes::Bytes> {
    // create the payload once
    //
    let total_bytes = cfg.elements * cfg.element_size;
    info!(
        "Generating object: type={:?}, elements={}, element_size={} bytes, total_bytes={}",
        cfg.object_type, cfg.elements, cfg.element_size, total_bytes
    );
    let data = if cfg.use_controlled {
        debug!("Using controlled data: dedup={}, compress={}", 
            cfg.dedup_factor, cfg.compress_factor
        );
        generate_controlled_data(total_bytes, cfg.dedup_factor, cfg.compress_factor)
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

/// Start of a data generation function that supports specifying deduplication and compression ratios of data created
pub fn generate_controlled_data(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {
    // Enforce a minimum size of BLK_SIZE bytes.
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    let block_size = BLK_SIZE;
    let nblocks = (size + block_size - 1) / block_size;

    // Determine deduplication factor and number of unique blocks
    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    let unique_blocks = if dedup_factor > 1 {
        ((nblocks as f64) / (dedup_factor as f64)).round().max(1.0) as usize
    } else {
        nblocks
    };

    // Prepare parameters for compression: fraction = (compress-1)/compress
    let (f_num, f_den) = if compress > 1 {
        (compress - 1, compress)
    } else {
        (0, 1)
    };
    let floor_len = (f_num * block_size) / f_den;
    let rem = (f_num * block_size) % f_den;

    // Precompute zero-prefix lengths per unique block (Bresenham distribution)
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

    // Generate unique blocks in parallel
    let unique: Vec<Arc<Vec<u8>>> = const_lens.into_par_iter().map(|const_len| {
        let mut rng = rand::rngs::ThreadRng::default();
        // Clone the shared base block cheaply
        let mut block_arc = Arc::clone(&A_BASE_BLOCK);
        // Obtain mutable access (clone-on-write if needed)
        let block = Arc::make_mut(&mut block_arc);

        // Zero out the constant prefix using slice fill
        block[..const_len].fill(0);

        // Compute modification region
        let region_start = const_len;
        let region_len = block_size - region_start;
        let modify_len = region_len.min(MOD_SIZE);

        // 1) Modify at the start of the random region
        rng.fill(&mut block[region_start..region_start + modify_len]);

        // 2) Optionally modify a second chunk if it fits
        let second_offset = HALF_BLK.max(region_start);
        if second_offset + modify_len <= block_size {
            rng.fill(&mut block[second_offset..second_offset + modify_len]);
        }

        block_arc
    }).collect();

    // Build the final output buffer without an initial zero fill
    let total_size = nblocks * block_size;
    let mut data: Vec<u8> = Vec::with_capacity(total_size);
    unsafe {
        data.set_len(total_size);
    }

    data.par_chunks_mut(block_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = i % unique.len();
            let src = &unique[idx];
            chunk.copy_from_slice(src);
        });

    // Trim to exact size
    data.truncate(size);
    data
}


/// Last version of code, should be accurate, but slightly inefficient
pub fn generate_controlled_data0(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {
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



