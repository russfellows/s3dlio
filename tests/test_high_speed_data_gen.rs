// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// High-speed data generation test - validates s3dlio can match dgen-rs performance
// Target: 10+ GB/s per core, 50+ GB/s aggregate on multi-core systems
//
// Based on dgen-rs benchmarking methodology:
// - Uses 1 MiB block size (DGEN_BLOCK_SIZE)
// - Uses 32 MB chunk size for optimal streaming
// - Measures aggregate throughput with rayon parallelization

use s3dlio::data_gen_alt::{generate_data_simple, generate_data, DataGenerator, GeneratorConfig};
use s3dlio::constants::DGEN_BLOCK_SIZE;
use std::time::Instant;

/// Target throughput based on dgen-rs benchmarks on this system (~50 GB/s)
const MINIMUM_EXPECTED_THROUGHPUT_GBPS: f64 = 10.0; // Conservative: at least 10 GB/s
const TEST_SIZE_BYTES: usize = 1024 * 1024 * 1024; // 1 GB test for speed

/// Quick sanity check that data generation is functional and reasonably fast
/// This test runs quickly (< 2 seconds) and validates basic performance
#[test]
fn test_data_gen_sanity_check() {
    println!("\n=== Data Generation Sanity Check ===");
    
    // Generate 64 MB - should complete in < 100ms on a modern system
    let size = 64 * 1024 * 1024;
    
    let start = Instant::now();
    let data = generate_data_simple(size, 1, 1);
    let elapsed = start.elapsed();
    
    let throughput_gbps = (data.len() as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed.as_secs_f64();
    
    println!("Generated {} MiB in {:?}", data.len() / (1024 * 1024), elapsed);
    println!("Throughput: {:.2} GB/s", throughput_gbps);
    
    assert_eq!(data.len(), size, "Generated data size mismatch");
    assert!(throughput_gbps > 1.0, "Throughput too low: {:.2} GB/s (expected > 1 GB/s)", throughput_gbps);
    
    println!("✅ Sanity check passed");
}

/// Validate that deduplication produces correct block patterns
/// With dedup=2, exactly 50% of blocks should be unique
#[test]
fn test_dedup_correctness() {
    println!("\n=== Deduplication Correctness Test ===");
    
    // Generate 64 MiB with dedup=2 (50% unique blocks)
    let num_blocks = 64;
    let size = num_blocks * DGEN_BLOCK_SIZE;
    let dedup = 2;
    
    let data = generate_data_simple(size, dedup, 1);
    
    // Count unique blocks
    let mut unique_blocks = std::collections::HashSet::new();
    for i in 0..num_blocks {
        let start = i * DGEN_BLOCK_SIZE;
        let end = start + DGEN_BLOCK_SIZE;
        unique_blocks.insert(&data.as_slice()[start..end]);
    }
    
    let unique_count = unique_blocks.len();
    let expected_unique = num_blocks / dedup;
    let dedup_ratio = unique_count as f64 / num_blocks as f64;
    
    println!("Total blocks: {}", num_blocks);
    println!("Unique blocks: {} (expected ~{})", unique_count, expected_unique);
    println!("Dedup ratio: {:.4} (expected {:.4})", dedup_ratio, 1.0 / dedup as f64);
    
    // Allow 10% tolerance
    let expected_ratio = 1.0 / dedup as f64;
    assert!((dedup_ratio - expected_ratio).abs() < 0.10,
        "Dedup ratio {:.4} not within 10% of expected {:.4}", dedup_ratio, expected_ratio);
    
    println!("✅ Deduplication correctness verified");
}

/// Validate that compression produces correct zero-fill patterns
/// With compress=2, exactly 50% of each block should be zeros
#[test]
fn test_compression_correctness() {
    println!("\n=== Compression Correctness Test ===");
    
    // Generate 16 MiB with compress=2 (50% zeros)
    let num_blocks = 16;
    let size = num_blocks * DGEN_BLOCK_SIZE;
    let compress = 2;
    
    let data = generate_data_simple(size, 1, compress);
    
    // Count zero bytes
    let zero_count = data.as_slice().iter().filter(|&&b| b == 0).count();
    let zero_ratio = zero_count as f64 / data.len() as f64;
    let expected_ratio = (compress as f64 - 1.0) / compress as f64;
    
    println!("Total bytes: {}", data.len());
    println!("Zero bytes: {} ({:.2}%)", zero_count, zero_ratio * 100.0);
    println!("Expected zero ratio: {:.4} ({:.2}%)", expected_ratio, expected_ratio * 100.0);
    
    // Allow 5% tolerance
    assert!((zero_ratio - expected_ratio).abs() < 0.05,
        "Compression ratio {:.4} not within 5% of expected {:.4}", zero_ratio, expected_ratio);
    
    println!("✅ Compression correctness verified");
}

/// Validate that the same seed produces the same data (deterministic RNG)
/// This ensures reproducibility for testing
#[test]
fn test_seed_determinism() {
    println!("\n=== Seed Determinism Test ===");
    
    let size = 8 * DGEN_BLOCK_SIZE; // 8 MiB
    let seed = 0xDEADBEEF_u64;
    
    // Generate twice with the same seed
    let config1 = GeneratorConfig {
        size,
        dedup_factor: 2,
        compress_factor: 2,
        seed: Some(seed),
        ..Default::default()
    };
    let data1 = generate_data(config1);
    
    let config2 = GeneratorConfig {
        size,
        dedup_factor: 2,
        compress_factor: 2,
        seed: Some(seed),
        ..Default::default()
    };
    let data2 = generate_data(config2);
    
    assert_eq!(data1.as_slice(), data2.as_slice(),
        "Same seed should produce identical data");
    println!("✅ Identical seed produces identical data");
    
    // Verify different seeds produce different data
    let config3 = GeneratorConfig {
        size,
        dedup_factor: 2,
        compress_factor: 2,
        seed: Some(0x12345678),
        ..Default::default()
    };
    let data3 = generate_data(config3);
    
    assert_ne!(data1.as_slice(), data3.as_slice(),
        "Different seeds should produce different data");
    println!("✅ Different seeds produce different data");
    
    println!("✅ Seed determinism verified");
}

/// High-speed throughput benchmark - validates multi-GB/s generation
/// This test generates 1 GB and measures throughput
/// 
/// NOTE: Run in release mode for accurate results:
/// cargo test --release test_high_speed_throughput -- --nocapture
#[test]
fn test_high_speed_throughput() {
    println!("\n=== High-Speed Throughput Benchmark ===");
    println!("Target: {} GB/s (based on dgen-rs performance)", MINIMUM_EXPECTED_THROUGHPUT_GBPS);
    println!("Test size: {} MiB", TEST_SIZE_BYTES / (1024 * 1024));
    println!();
    
    // Warmup: generate 64 MB to warm caches and JIT
    let _ = generate_data_simple(64 * 1024 * 1024, 1, 1);
    
    // Actual benchmark
    let start = Instant::now();
    let data = generate_data_simple(TEST_SIZE_BYTES, 1, 1);
    let elapsed = start.elapsed();
    
    let size_gb = data.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    let throughput_gbps = size_gb / elapsed.as_secs_f64();
    
    println!("Generated: {} MiB", data.len() / (1024 * 1024));
    println!("Duration:  {:?}", elapsed);
    println!("Throughput: {:.2} GB/s", throughput_gbps);
    
    // In debug mode, we'll be slower - adjust threshold
    #[cfg(debug_assertions)]
    let adjusted_threshold = MINIMUM_EXPECTED_THROUGHPUT_GBPS / 20.0; // ~0.5 GB/s in debug
    #[cfg(not(debug_assertions))]
    let adjusted_threshold = MINIMUM_EXPECTED_THROUGHPUT_GBPS;
    
    if throughput_gbps >= adjusted_threshold {
        println!("✅ PASSED: {:.2} GB/s >= {:.2} GB/s threshold", throughput_gbps, adjusted_threshold);
    } else {
        println!("⚠️  Below threshold: {:.2} GB/s < {:.2} GB/s", throughput_gbps, adjusted_threshold);
        #[cfg(not(debug_assertions))]
        panic!("Throughput {:.2} GB/s below minimum {:.2} GB/s", throughput_gbps, adjusted_threshold);
    }
    
    // Verify data is not all zeros (basic sanity)
    let non_zero_count = data.as_slice().iter().filter(|&&b| b != 0).count();
    assert!(non_zero_count > data.len() / 2, 
        "Data appears to be mostly zeros - RNG may not be working");
    
    println!("\n=== Comparison to dgen-rs ===");
    println!("dgen-rs achieves ~50 GB/s aggregate on this system");
    println!("s3dlio achieved: {:.2} GB/s", throughput_gbps);
    if throughput_gbps >= 40.0 {
        println!("✅ Excellent: Within 80% of dgen-rs performance");
    } else if throughput_gbps >= 20.0 {
        println!("⚡ Good: Within 40% of dgen-rs performance");
    } else if throughput_gbps >= 10.0 {
        println!("📊 Acceptable: Meets minimum threshold");
    }
}

/// Streaming throughput test with 32 MB chunks (optimal per dgen-rs)
/// This matches the dgen-rs streaming benchmark methodology
#[test]
fn test_streaming_throughput_optimal_chunks() {
    println!("\n=== Streaming Throughput (32 MB chunks) ===");
    
    let total_size = 512 * 1024 * 1024; // 512 MiB
    let chunk_size = 32 * 1024 * 1024;  // 32 MiB chunks (optimal per dgen-rs)
    
    let config = GeneratorConfig {
        size: total_size,
        dedup_factor: 1,
        compress_factor: 1,
        ..Default::default()
    };
    let mut gen = DataGenerator::new(config);
    let mut buffer = vec![0u8; chunk_size];
    
    // Warmup
    let _ = gen.fill_chunk(&mut buffer[..1024 * 1024]);
    
    // Reset generator
    let config = GeneratorConfig {
        size: total_size,
        dedup_factor: 1,
        compress_factor: 1,
        ..Default::default()
    };
    let mut gen = DataGenerator::new(config);
    
    let start = Instant::now();
    let mut total_bytes = 0;
    
    while !gen.is_complete() {
        let bytes = gen.fill_chunk(&mut buffer);
        total_bytes += bytes;
    }
    
    let elapsed = start.elapsed();
    let size_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let throughput_gbps = size_gb / elapsed.as_secs_f64();
    
    println!("Generated: {} MiB in {} chunks", total_bytes / (1024 * 1024), total_bytes / chunk_size);
    println!("Duration:  {:?}", elapsed);
    println!("Throughput: {:.2} GB/s", throughput_gbps);
    
    assert_eq!(total_bytes, total_size, "Streaming generated wrong size");
    
    #[cfg(not(debug_assertions))]
    assert!(throughput_gbps > 5.0, "Streaming throughput too low: {:.2} GB/s", throughput_gbps);
    
    println!("✅ Streaming throughput test passed");
}
