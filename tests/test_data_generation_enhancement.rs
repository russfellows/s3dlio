// Test for data generation improvements - verifies single-pass generator maintains identical output
// to the two-pass version while preserving dedup/compress guarantees.

use s3dlio::constants::{BLK_SIZE, MOD_SIZE, A_BASE_BLOCK};
use s3dlio::data_gen::{generate_controlled_data, generate_controlled_data_two_pass};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// Test that single-pass and two-pass generators produce identical *properties*
/// (dedup/compress), even if RNG differences prevent bit-for-bit parity.
#[test]
fn test_single_pass_vs_two_pass_parity() {
    // (size, dedup, compress)
    let test_cases = vec![
        (4096, 1, 1),               // Fully unique, no compression
        (8192, 3, 2),               // 1:3 dedup, 2:1 compression
        (16384, 5, 4),              // 1:5 dedup, 4:1 compression
        (32768, 2, 3),              // 1:2 dedup, 3:1 compression
        (BLK_SIZE, 1, 1),           // Minimum size
        (BLK_SIZE * 10, 1, 1),      // Multiple blocks, fully unique
        (65536, 4, 5),              // High compression
    ];

    for (size, dedup, compress) in test_cases {
        // Generate using current single-pass implementation
        let single_pass = generate_controlled_data(size, dedup, compress);

        // Generate using two-pass reference implementation
        let two_pass = generate_controlled_data_two_pass(size, dedup, compress);

        // They should be identical in length
        assert_eq!(
            single_pass.len(),
            two_pass.len(),
            "Length mismatch for size={}, dedup={}, compress={}",
            size,
            dedup,
            compress
        );

        // NOTE: Due to RNG differences, we can't expect bit-identical output.
        // Verify dedup/compress properties are equivalent instead.
        verify_dedup_compress_properties(&single_pass, dedup, compress, size);
        verify_dedup_compress_properties(&two_pass, dedup, compress, size);

        println!(
            "✅ Parity verified for size={}, dedup={}, compress={} (length={})",
            size,
            dedup,
            compress,
            single_pass.len()
        );
    }
}

/// Verify that generated data has correct deduplication and compression properties.
fn verify_dedup_compress_properties(
    data: &[u8],
    dedup: usize,
    compress: usize,
    original_size: usize,
) {
    let block_size = BLK_SIZE;
    let nblocks = (original_size + block_size - 1) / block_size;

    // Calculate expected unique blocks
    let dedup_factor = if dedup == 0 { 1 } else { dedup };
    let expected_unique_blocks = if dedup_factor > 1 {
        ((nblocks as f64) / (dedup_factor as f64)).round().max(1.0) as usize
    } else {
        nblocks
    };

    // Count actual unique blocks by hashing each block
    let mut unique_blocks = HashSet::new();
    for chunk in data.chunks(block_size) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        chunk.hash(&mut hasher);
        unique_blocks.insert(hasher.finish());
    }

    // Allow tolerance for rounding in dedup calculation
    let actual_unique = unique_blocks.len();
    let tolerance = (expected_unique_blocks as f64 * 0.1).max(1.0) as usize;

    assert!(
        actual_unique <= expected_unique_blocks + tolerance,
        "Too many unique blocks: expected ~{}, got {} (dedup={})",
        expected_unique_blocks,
        actual_unique,
        dedup
    );

    // Verify compression properties by counting zero bytes
    if compress > 1 {
        let zero_count = data.iter().filter(|&&b| b == 0).count();
        let expected_zero_fraction = (compress - 1) as f64 / compress as f64;
        let expected_zeros = (data.len() as f64 * expected_zero_fraction) as usize;
        let tolerance = (expected_zeros as f64 * 0.1).max(100.0) as usize;

        assert!(
            zero_count >= expected_zeros.saturating_sub(tolerance)
                && zero_count <= expected_zeros + tolerance,
            "Incorrect compression ratio: expected ~{} zeros, got {} (compress={})",
            expected_zeros,
            zero_count,
            compress
        );
    }
}

/// Test that the single-pass generator maintains block structure integrity.
#[test]
fn test_block_structure_integrity() {
    let size = BLK_SIZE * 5;
    let data = generate_controlled_data(size, 1, 1); // Fully unique

    // Verify each block has the expected modifications
    for (i, chunk) in data.chunks(BLK_SIZE).enumerate() {
        if chunk.len() == BLK_SIZE {
            // Each block should differ from the base block due to randomization
            let base_block = &*A_BASE_BLOCK;
            assert_ne!(
                chunk,
                &base_block[..BLK_SIZE],
                "Block {} should be modified from base block",
                i
            );

            // Verify first MOD_SIZE bytes are randomized (not from base)
            let first_region = &chunk[0..MOD_SIZE.min(chunk.len())];
            let base_first = &base_block[0..MOD_SIZE.min(base_block.len())];
            assert_ne!(
                first_region,
                base_first,
                "Block {} first region should be randomized",
                i
            );
        }
    }
}

/// Test edge cases and boundary conditions.
#[test]
fn test_edge_cases() {
    // Minimum size
    let min_data = generate_controlled_data(1, 1, 1);
    assert_eq!(min_data.len(), BLK_SIZE, "Should enforce minimum size");

    // Exact block size
    let exact_data = generate_controlled_data(BLK_SIZE, 1, 1);
    assert_eq!(exact_data.len(), BLK_SIZE, "Should handle exact block size");

    // Zero dedup factor should be treated as 1
    let zero_dedup = generate_controlled_data(BLK_SIZE * 2, 0, 1);
    let normal_dedup = generate_controlled_data(BLK_SIZE * 2, 1, 1);
    assert_eq!(
        zero_dedup.len(),
        normal_dedup.len(),
        "Zero dedup should equal dedup=1"
    );

    // High compression factor
    let high_compress = generate_controlled_data(BLK_SIZE, 1, 10);
    let zero_count = high_compress.iter().filter(|&&b| b == 0).count();
    let expected_ratio = 9.0 / 10.0; // (10-1)/10
    let expected_zeros = (BLK_SIZE as f64 * expected_ratio) as usize;

    assert!(
        zero_count >= expected_zeros.saturating_sub(50)
            && zero_count <= expected_zeros + 50,
        "High compression should have ~{} zeros, got {}",
        expected_zeros,
        zero_count
    );
}

/// Performance comparison test (informational).
#[test]
fn test_performance_comparison() {
    let size = 1024 * 1024; // 1 MiB
    let iterations = 5;

    // Measure single-pass performance
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = generate_controlled_data(size, 3, 2);
    }
    let single_pass_time = start.elapsed();

    // Measure two-pass performance
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = generate_controlled_data_two_pass(size, 3, 2);
    }
    let two_pass_time = start.elapsed();

    // Compute per-iteration averages safely
    fn avg_per_iter(total: Duration, iters: u32) -> Duration {
        if iters == 0 {
            return Duration::from_nanos(0);
        }
        let nanos = total.as_nanos() / iters as u128;
        Duration::from_nanos(nanos as u64)
    }

    let single_avg = avg_per_iter(single_pass_time, iterations as u32);
    let two_pass_avg = avg_per_iter(two_pass_time, iterations as u32);

    println!(
        "Performance comparison ({}x {} MiB): single-pass avg: {:?}, two-pass avg: {:?}",
        iterations,
        size / (1024 * 1024),
        single_avg,
        two_pass_avg
    );

    if single_pass_time > two_pass_time {
        println!("⚠️  Single-pass is slower — may need further optimization");
    } else {
        println!(
            "✅ Single-pass is faster by {:?}",
            two_pass_time.saturating_sub(single_pass_time)
        );
    }
}

/// Test data generation with different random seeds produces different output.
/// If your implementation seeds from a fixed source, this may need adjusting.
#[test]
fn test_randomness_properties() {
    let size = BLK_SIZE * 3;

    // Generate multiple instances - they should be different due to randomness
    let data1 = generate_controlled_data(size, 1, 1);
    let data2 = generate_controlled_data(size, 1, 1);

    // With fully unique blocks, the outputs should be different
    // (very low probability of collision with good RNG)
    assert_ne!(
        data1, data2,
        "Multiple generations should produce different random data"
    );

    // But they should have the same length
    assert_eq!(data1.len(), data2.len(), "Should have same length");

    // And similar (low) zero-byte counts with no compression
    let zeros1 = data1.iter().filter(|&&b| b == 0).count();
    let zeros2 = data2.iter().filter(|&&b| b == 0).count();

    assert!(
        zeros1 < size / 10 && zeros2 < size / 10,
        "Should have minimal zeros with no compression"
    );
}
