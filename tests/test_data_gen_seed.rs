// Test explicit seed functionality for DataGenerator
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use s3dlio::data_gen::DataGenerator;

#[test]
fn test_explicit_seed_produces_identical_data() {
    println!("\n=== Testing Explicit Seed Produces Identical Data ===");
    
    let seed = 12345u64;
    let size = 1024 * 1024; // 1 MB
    
    // Create two generators with the same seed
    let gen1 = DataGenerator::new_with_seed(seed);
    let gen2 = DataGenerator::new_with_seed(seed);
    
    // Generate data from both
    let mut obj1 = gen1.begin_object(size, 2, 1);
    let mut obj2 = gen2.begin_object(size, 2, 1);
    
    let mut data1 = Vec::new();
    let mut data2 = Vec::new();
    
    while let Some(chunk) = obj1.fill_chunk(128 * 1024) {
        data1.extend(chunk);
    }
    
    while let Some(chunk) = obj2.fill_chunk(128 * 1024) {
        data2.extend(chunk);
    }
    
    assert_eq!(data1.len(), data2.len(), "Sizes should match");
    assert_eq!(data1, data2, "Data should be identical with same seed");
    
    println!("✓ Generated {} bytes identically with seed {}", data1.len(), seed);
}

#[test]
fn test_different_seeds_produce_different_data() {
    println!("\n=== Testing Different Seeds Produce Different Data ===");
    
    let size = 512 * 1024; // 512 KB
    
    // Create generators with different seeds
    let gen1 = DataGenerator::new_with_seed(11111);
    let gen2 = DataGenerator::new_with_seed(22222);
    
    let mut obj1 = gen1.begin_object(size, 1, 1);
    let mut obj2 = gen2.begin_object(size, 1, 1);
    
    let mut data1 = Vec::new();
    let mut data2 = Vec::new();
    
    while let Some(chunk) = obj1.fill_chunk(64 * 1024) {
        data1.extend(chunk);
    }
    
    while let Some(chunk) = obj2.fill_chunk(64 * 1024) {
        data2.extend(chunk);
    }
    
    assert_eq!(data1.len(), data2.len(), "Sizes should match");
    assert_ne!(data1, data2, "Data should be different with different seeds");
    
    println!("✓ Different seeds produced different data");
}

#[test]
fn test_seed_vs_system_entropy() {
    println!("\n=== Testing Seed vs System Entropy ===");
    
    let size = 256 * 1024; // 256 KB
    
    // Explicit seed generator
    let gen_seed = DataGenerator::new_with_seed(99999);
    
    // System entropy generator (default)
    let gen_random1 = DataGenerator::new(None);
    let gen_random2 = DataGenerator::new(None);
    
    // Seed-based should be deterministic
    let mut obj_seed1 = gen_seed.begin_object(size, 1, 1);
    let mut obj_seed2 = gen_seed.begin_object(size, 1, 1);
    
    let mut data_seed1 = Vec::new();
    let mut data_seed2 = Vec::new();
    
    while let Some(chunk) = obj_seed1.fill_chunk(32 * 1024) {
        data_seed1.extend(chunk);
    }
    
    while let Some(chunk) = obj_seed2.fill_chunk(32 * 1024) {
        data_seed2.extend(chunk);
    }
    
    assert_eq!(data_seed1, data_seed2, "Seeded generator should be deterministic");
    println!("✓ Seeded generator is deterministic");
    
    // System entropy should be unique per instance
    let mut obj_rand1 = gen_random1.begin_object(size, 1, 1);
    let mut obj_rand2 = gen_random2.begin_object(size, 1, 1);
    
    let mut data_rand1 = Vec::new();
    let mut data_rand2 = Vec::new();
    
    while let Some(chunk) = obj_rand1.fill_chunk(32 * 1024) {
        data_rand1.extend(chunk);
    }
    
    while let Some(chunk) = obj_rand2.fill_chunk(32 * 1024) {
        data_rand2.extend(chunk);
    }
    
    assert_ne!(data_rand1, data_rand2, "System entropy generators should be unique");
    println!("✓ System entropy generators are unique");
    
    // Seeded data should differ from system entropy data
    assert_ne!(data_seed1, data_rand1, "Seeded should differ from system entropy");
    println!("✓ Seeded differs from system entropy");
}

#[test]
fn test_reproducible_across_chunk_sizes() {
    println!("\n=== Testing Reproducibility Across Different Chunk Sizes ===");
    
    let seed = 54321u64;
    let size = 512 * 1024; // 512 KB
    
    // Generate with small chunks
    let gen1 = DataGenerator::new_with_seed(seed);
    let mut obj1 = gen1.begin_object(size, 1, 1);
    let mut data_small_chunks = Vec::new();
    
    while let Some(chunk) = obj1.fill_chunk(16 * 1024) {  // 16 KB chunks
        data_small_chunks.extend(chunk);
    }
    
    // Generate with large chunks
    let gen2 = DataGenerator::new_with_seed(seed);
    let mut obj2 = gen2.begin_object(size, 1, 1);
    let mut data_large_chunks = Vec::new();
    
    while let Some(chunk) = obj2.fill_chunk(256 * 1024) {  // 256 KB chunks
        data_large_chunks.extend(chunk);
    }
    
    assert_eq!(data_small_chunks.len(), data_large_chunks.len(), "Sizes should match");
    assert_eq!(data_small_chunks, data_large_chunks, 
               "Data should be identical regardless of chunk size");
    
    println!("✓ Same seed produces identical data with different chunk sizes");
}
