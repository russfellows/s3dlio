// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// High-performance data generation validation test
// Based on dgen-rs benchmark methodology (achieves 47+ GB/s)
//
// Key patterns for maximum performance:
// 1. Use DataBuffer directly (zero-copy)
// 2. Use 32 MB chunks (optimal for L3 cache)
// 3. Use 1 MB internal block size (good parallelism)
// 4. Reuse buffer allocation across fills
// 5. Use streaming API for large datasets
// 6. Generate 100 GB to hit peak throughput (cache warming)

use std::time::Instant;
use s3dlio::data_gen_alt::{DataGenerator, GeneratorConfig, NumaMode, generate_data_simple};
use s3dlio::constants::DGEN_BLOCK_SIZE;

// Conditional compilation for debug vs release builds
// Debug builds are much slower, so we reduce test size and expectations
#[cfg(not(debug_assertions))]
const MIN_THROUGHPUT_GBS: f64 = 35.0; // Release: 35 GB/s (dgen-rs achieves 47 GB/s)
#[cfg(not(debug_assertions))]
const TEST_SIZE: usize = 100 * 1024 * 1024 * 1024; // Release: 100 GB
#[cfg(not(debug_assertions))]
const WARMUP_SIZE: usize = 1024 * 1024 * 1024; // Release: 1 GB warmup

#[cfg(debug_assertions)]
const MIN_THROUGHPUT_GBS: f64 = 0.5; // Debug: 500 MB/s (much slower without optimizations)
#[cfg(debug_assertions)]
const TEST_SIZE: usize = 16 * 1024 * 1024 * 1024; // Debug: 16 GB
#[cfg(debug_assertions)]
const WARMUP_SIZE: usize = 256 * 1024 * 1024; // Debug: 256 MB warmup

/// Full performance test matching dgen-rs methodology
/// Release: Generates 100 GB with 32 MB chunks after 1 GB warmup (~2-3 seconds)
/// Debug: Generates 16 GB with 32 MB chunks after 256 MB warmup (slower without optimization)
#[test]
fn test_datagen_streaming_throughput_100gb() {
    const CHUNK_SIZE: usize = 32 * 1024 * 1024; // 32 MB (optimal for L3 cache)
    
    println!("\n=== s3dlio Data Generation Performance (dgen-rs methodology) ===");
    println!("Warmup: {} GB", WARMUP_SIZE as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("Test size: {} GB", TEST_SIZE as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("Chunk size: {} MB", CHUNK_SIZE / (1024 * 1024));
    println!("Block size: {} MB", DGEN_BLOCK_SIZE / (1024 * 1024));
    
    // Pre-allocate buffer (reused for all chunks)
    let mut buffer = vec![0u8; CHUNK_SIZE];
    
    // === WARMUP PHASE ===
    println!("\nWarming up with {} GB...", WARMUP_SIZE as f64 / 1024.0 / 1024.0 / 1024.0);
    let warmup_config = GeneratorConfig {
        size: WARMUP_SIZE,
        dedup_factor: 1,
        compress_factor: 1,
        numa_mode: NumaMode::Auto,
        max_threads: None,
        numa_node: None,
        block_size: None,
        seed: None,
    };
    
    let mut warmup_gen = DataGenerator::new(warmup_config);
    while !warmup_gen.is_complete() {
        warmup_gen.fill_chunk(&mut buffer);
    }
    println!("Warmup complete.");
    
    // === BENCHMARK PHASE ===
    println!("\nGenerating {} GB...", TEST_SIZE as f64 / 1024.0 / 1024.0 / 1024.0);
    
    let config = GeneratorConfig {
        size: TEST_SIZE,
        dedup_factor: 1,      // No dedup = fastest
        compress_factor: 1,   // No compression = fastest
        numa_mode: NumaMode::Auto,
        max_threads: None,    // Use all cores
        numa_node: None,
        block_size: None,     // Use DGEN_BLOCK_SIZE (1 MB)
        seed: None,
    };
    
    let mut generator = DataGenerator::new(config);
    
    let start = Instant::now();
    let mut total_bytes = 0usize;
    
    while !generator.is_complete() {
        let n = generator.fill_chunk(&mut buffer);
        total_bytes += n;
    }
    
    let duration = start.elapsed();
    let duration_secs = duration.as_secs_f64();
    let throughput_gbs = (total_bytes as f64 / 1024.0 / 1024.0 / 1024.0) / duration_secs;
    
    println!("\nResults:");
    println!("  Generated: {:.2} GB", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  Duration:  {:.3} seconds", duration_secs);
    println!("  Throughput: {:.2} GB/s", throughput_gbs);
    println!("  Minimum:   {:.2} GB/s (dgen-rs achieves ~47 GB/s)", MIN_THROUGHPUT_GBS);
    
    // Validate minimum throughput
    assert!(
        throughput_gbs >= MIN_THROUGHPUT_GBS,
        "Throughput {:.2} GB/s below minimum {:.2} GB/s. \
         Expected ~47 GB/s based on dgen-rs benchmarks.",
        throughput_gbs,
        MIN_THROUGHPUT_GBS
    );
    
    println!("✅ PASSED: Throughput {:.2} GB/s meets minimum {:.2} GB/s", 
             throughput_gbs, MIN_THROUGHPUT_GBS);
}

/// Quick sanity test (for CI - runs in <1 second)
/// Just validates the API works, doesn't check peak throughput
#[test]
fn test_datagen_api_sanity() {
    const SIZE: usize = 64 * 1024 * 1024; // 64 MB
    const CHUNK_SIZE: usize = 32 * 1024 * 1024; // 32 MB
    
    let config = GeneratorConfig {
        size: SIZE,
        dedup_factor: 1,
        compress_factor: 1,
        numa_mode: NumaMode::Auto,
        max_threads: None,
        numa_node: None,
        block_size: None,
        seed: None,
    };
    
    let mut generator = DataGenerator::new(config);
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut total_bytes = 0usize;
    
    while !generator.is_complete() {
        let n = generator.fill_chunk(&mut buffer);
        total_bytes += n;
    }
    
    assert_eq!(total_bytes, SIZE, "Should generate exactly {} bytes", SIZE);
    println!("✅ API sanity test passed: generated {} MB", total_bytes / (1024 * 1024));
}

/// Test batch generation API
#[test]
fn test_datagen_batch_api() {
    const SIZE: usize = 64 * 1024 * 1024; // 64 MB
    
    let data = generate_data_simple(SIZE, 1, 1);
    
    assert_eq!(data.len(), SIZE, "Should generate exactly {} bytes", SIZE);
    
    // Verify data is not all zeros (actually generated)
    let non_zero = data.as_slice().iter().filter(|&&b| b != 0).count();
    assert!(non_zero > SIZE / 2, "Data should be mostly non-zero (got {} non-zero bytes)", non_zero);
    
    println!("✅ Batch API test passed: generated {} MB", SIZE / (1024 * 1024));
}

/// Test dedup and compression produce expected ratios
#[test]
fn test_datagen_dedup_compress_ratios() {
    const SIZE: usize = 64 * DGEN_BLOCK_SIZE; // 64 blocks = 64 MB
    
    // Test dedup=2 (should produce 50% unique blocks)
    let data = generate_data_simple(SIZE, 2, 1);
    assert_eq!(data.len(), SIZE);
    
    // Count unique 1 MB blocks
    let mut unique_blocks = std::collections::HashSet::new();
    let slice = data.as_slice();
    for i in 0..(SIZE / DGEN_BLOCK_SIZE) {
        let start = i * DGEN_BLOCK_SIZE;
        let end = start + DGEN_BLOCK_SIZE;
        unique_blocks.insert(&slice[start..end]);
    }
    
    let expected_unique = SIZE / DGEN_BLOCK_SIZE / 2; // 50% unique
    let actual_unique = unique_blocks.len();
    
    println!("Dedup=2: {} unique blocks out of {} (expected ~{})", 
             actual_unique, SIZE / DGEN_BLOCK_SIZE, expected_unique);
    
    // Allow 10% tolerance
    let ratio_diff = (actual_unique as f64 - expected_unique as f64).abs() / (expected_unique as f64);
    assert!(
        ratio_diff < 0.1,
        "Dedup ratio off: got {} unique, expected ~{}", actual_unique, expected_unique
    );
    
    // Test compress=2 (should produce 50% zeros)
    let data = generate_data_simple(SIZE, 1, 2);
    let zeros = data.as_slice().iter().filter(|&&b| b == 0).count();
    let zero_ratio = zeros as f64 / SIZE as f64;
    
    println!("Compress=2: {:.1}% zeros (expected ~50%)", zero_ratio * 100.0);
    
    assert!(
        (zero_ratio - 0.5).abs() < 0.1,
        "Compression ratio off: got {:.1}% zeros, expected ~50%", zero_ratio * 100.0
    );
    
    println!("✅ Dedup/compress ratio test passed");
}
