// src/data_gen.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 

use once_cell::sync::Lazy;
use rand::Rng;
use rayon::prelude::*;

// -----------------------------------------------------------------------------
// Generate a buffer of random bytes.
// -----------------------------------------------------------------------------
//
const BLK_SIZE: usize = 512;
const HALF_BLK: usize = BLK_SIZE / 2;

/// A base random block of BLK_SIZE bytes, generated once.
static BASE_BLOCK: Lazy<Vec<u8>> = Lazy::new(|| {
    let mut block = vec![0u8; BLK_SIZE];
    rand::rngs::ThreadRng::default().fill(&mut block[..]);
    block
});

// For now, each of our 4 object types just calls the same function
pub fn generate_npz(size: usize) -> Vec<u8> {
    generate_random_data(size)
} 

pub fn generate_tfrecord(size: usize) -> Vec<u8> {
    generate_random_data(size)
} 

pub fn generate_hdf5(size: usize) -> Vec<u8> {
    generate_random_data(size)
} 

pub fn generate_raw_data(size: usize) -> Vec<u8> {
    generate_random_data(size)
} 

/// Generates a buffer of `size` random bytes by:
/// 1. Enforcing a minimum size of BLK_SIZE bytes.
/// 2. Filling each BLK_SIZE-byte block with a static base block.
/// 3. Modifying the first 32 bytes of each block,
///    and modifying the last 32 bytes only if the block is larger than 128 bytes.
///
/// This ensures each BLK_SIZE-byte block is unique while avoiding the need to generate a whole new
/// random buffer on every call.
//pub fn generate_random_data(mut size: usize) -> Vec<u8> {
fn generate_random_data(mut size: usize) -> Vec<u8> {
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

        // Modify the first 32 bytes (or the full block if it's smaller).
        if block_size > 0 {
            let first_len = if block_size >= 32 { 32 } else { block_size };
            rng.fill(&mut data[offset .. offset + first_len]);
        }

        // Modify the last 32 bytes only if the block is larger than 128 bytes.
        if block_size > HALF_BLK {
            rng.fill(&mut data[block_end - 32 .. block_end]);
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
///   the first 32 bytes, and if the block is larger than 128 bytes, also the last 32 bytes.
/// - Finally, the unique blocks are repeated in round-robin order to fill the requested output size.
/// - The final assembly step is parallelized using Rayon for efficiency on large buffers.
///
/// # Returns
/// A `Vec<u8>` with the requested size, containing data with the specified deduplication and compressibility characteristics.

#[allow(dead_code)]
/// Start of a data generation function that supports specifying deduplication and compression ratios of data created
//pub fn generate_controlled_data(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {
fn generate_controlled_data(mut size: usize, dedup: usize, compress: usize) -> Vec<u8> {
    // Enforce a minimum size of BLK_SIZE bytes.
    if size < BLK_SIZE {
        size = BLK_SIZE;
    }

    let block_size = BLK_SIZE;
    let nblocks = (size + block_size - 1) / block_size;

    // Determine the number of unique blocks based on dedup factor.
    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    let unique_blocks = (nblocks + dedup_factor - 1) / dedup_factor;

    // Compute the compressibility fraction:
    // For compress > 1: f = (compress - 1) / compress, meaning that f of the block is constant.
    let f = if compress > 1 { (compress - 1) as f64 / compress as f64 } else { 0.0 };
    let constant_length = (f * block_size as f64).round() as usize;
    // The rest (block_size - constant_length) is left to be random.

    // Build a base BLK_SIZE-byte block that already has the desired compressible layout.
    let mut base_block = vec![0u8; block_size];
    // Fill the constant portion (first constant_length bytes) with a fixed pattern (here, zeros).
    for j in 0..constant_length {
        base_block[j] = 0;
    }
    {
        // Fill the remaining bytes with random data.
        let mut rng = rand::rngs::ThreadRng::default();
        rng.fill(&mut base_block[constant_length..]);
    }

    // Generate the unique blocks.
    let mut unique: Vec<Vec<u8>> = Vec::with_capacity(unique_blocks);
    {
        let mut rng = rand::rngs::ThreadRng::default();
        for i in 0..unique_blocks {
            let _ = i; // Because we never explicitly use i, its just an index
            // Start with a clone of the base block
            let mut block = base_block.clone();
            // Uniquify by modifying the first 32 bytes (or less if the block is smaller).
            let modify_len = std::cmp::min(32, block_size);
            rng.fill(&mut block[..modify_len]);
            //
            // If the block is larger than HALF_BLK bytes, also modify the first 32 bytes of the
            // second half of our block.
            if block_size > HALF_BLK {
                let offset = HALF_BLK;
                let end = offset + 32;
                rng.fill(&mut block[offset..end]);
            }
            /*
             * Old, incorrect, remove soon
            if block_size > 128 {
                rng.fill(&mut block[block_size - 32..block_size]);
            }
            */
            unique.push(block);
        }
    }

    // Build the final output buffer.
    let total_size = nblocks * block_size;
    let mut data = vec![0u8; total_size];
    // Use Rayon to fill each block in parallel.
    data.par_chunks_mut(block_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let unique_index = i % unique.len();
            chunk.copy_from_slice(&unique[unique_index]);
        });

    // Trim the buffer to the exact requested size.
    data.truncate(size);
    data
}
