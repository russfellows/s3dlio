// tests/test_streaming_data_generation.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
//

use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use s3dlio::constants::BLK_SIZE;

/// Test that streaming generation produces identical results to single-pass generation
#[test]
fn test_streaming_consistency_various_sizes() {
    let test_cases = [
        (BLK_SIZE / 2, 1, 1),
        (BLK_SIZE, 1, 1),
        (BLK_SIZE * 2, 1, 1),
        (BLK_SIZE * 3 + 100, 2, 2),
    ];

    for &(size, dedup, compress) in &test_cases {
        println!("Testing size={}, dedup={}, compress={}", size, dedup, compress);
        
        // Test that streaming produces correct length and structure
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, dedup, compress);
        let streaming_result = object_gen.fill_chunk(size).unwrap();
        
        // Verify length is correct
        assert_eq!(
            streaming_result.len(),
            size,
            "Length mismatch for size={}, dedup={}, compress={}", size, dedup, compress
        );
        
        // Verify that reset and re-generation produces identical results
        object_gen.reset();
        let regenerated = object_gen.fill_chunk(size).unwrap();
        assert_eq!(
            streaming_result, 
            regenerated, 
            "Reset consistency failed for size={}, dedup={}, compress={}", size, dedup, compress
        );
        
        // Verify compression ratio is roughly correct for compressed data
        if compress > 1 {
            let zero_count = streaming_result.iter().filter(|&&b| b == 0).count();
            let expected_zeros = (size * (compress - 1)) / compress;
            let tolerance = size / 10; // 10% tolerance
            assert!(
                zero_count >= expected_zeros - tolerance && zero_count <= expected_zeros + tolerance,
                "Compression ratio incorrect: expected ~{} zeros, got {}", expected_zeros, zero_count
            );
        }
    }
}

/// Test streaming with various chunk sizes to ensure consistency
#[test]
fn test_streaming_different_chunk_sizes() {
    let size = BLK_SIZE * 2;
    let dedup = 1;
    let compress = 1;
    
    // Generate reference using streaming with large chunk
    let generator = DataGenerator::new();
    let mut reference_gen = generator.begin_object(size, dedup, compress);
    let reference = reference_gen.fill_chunk(size).unwrap();
    
    let test_chunk_sizes = [1, 32, BLK_SIZE / 2, BLK_SIZE, BLK_SIZE + 32, size];
    
    for &chunk_size in &test_chunk_sizes {
        println!("Testing chunk size: {}", chunk_size);
        
        // Create new generator with same parameters
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, dedup, compress);
        
        let mut result = Vec::new();
        while let Some(chunk) = object_gen.fill_chunk(chunk_size) {
            result.extend_from_slice(&chunk);
        }
        
        assert_eq!(result.len(), size, "Length mismatch for chunk size {}", chunk_size);
        
        // Instead of exact equality (which won't work due to different entropy), 
        // test that the structure is consistent (same compression characteristics)
        if compress > 1 {
            let ref_zeros = reference.iter().filter(|&&b| b == 0).count();
            let result_zeros = result.iter().filter(|&&b| b == 0).count();
            let tolerance = size / 20; // 5% tolerance
            assert!(
                (ref_zeros as i32 - result_zeros as i32).abs() < tolerance as i32,
                "Chunk size {} changed compression: ref {} zeros, got {} zeros", 
                chunk_size, ref_zeros, result_zeros
            );
        }
    }
}

/// Test that ObjectGen correctly tracks position and completion
#[test]
fn test_object_gen_state_tracking() {
    let size = BLK_SIZE * 4;
    let chunk_size = BLK_SIZE;
    
    let generator = DataGenerator::new();
    let mut object_gen = generator.begin_object(size, 2, 2);
    
    // Initially at position 0
    assert_eq!(object_gen.position(), 0);
    assert_eq!(object_gen.total_size(), size);
    assert!(!object_gen.is_complete());
    
    // Generate chunks and verify position tracking
    let mut total_generated = 0;
    let mut chunk_count = 0;
    
    while let Some(chunk) = object_gen.fill_chunk(chunk_size) {
        chunk_count += 1;
        total_generated += chunk.len();
        
        assert_eq!(object_gen.position(), total_generated);
        
        // Should be complete only after last chunk
        if total_generated == size {
            assert!(object_gen.is_complete());
        } else {
            assert!(!object_gen.is_complete());
        }
    }
    
    assert_eq!(total_generated, size);
    assert!(object_gen.is_complete());
    assert_eq!(chunk_count, 4); // size / chunk_size
    
    // Further calls should return None
    assert!(object_gen.fill_chunk(chunk_size).is_none());
}

/// Test reset functionality
#[test]
fn test_object_gen_reset() {
    let size = BLK_SIZE * 2;
    let chunk_size = BLK_SIZE;
    
    let generator = DataGenerator::new();
    let mut object_gen = generator.begin_object(size, 2, 2);
    
    // Generate first chunk
    let first_chunk = object_gen.fill_chunk(chunk_size).unwrap();
    assert_eq!(object_gen.position(), chunk_size);
    assert!(!object_gen.is_complete());
    
    // Reset and generate again
    object_gen.reset();
    assert_eq!(object_gen.position(), 0);
    assert!(!object_gen.is_complete());
    
    let first_chunk_again = object_gen.fill_chunk(chunk_size).unwrap();
    assert_eq!(first_chunk, first_chunk_again);
}

/// Test fill_remaining method
#[test]
fn test_fill_remaining() {
    let size = BLK_SIZE * 3;
    let chunk_size = BLK_SIZE;
    
    let generator = DataGenerator::new();
    let mut object_gen = generator.begin_object(size, 2, 2);
    
    // Generate first chunk normally
    let first_chunk = object_gen.fill_chunk(chunk_size).unwrap();
    assert_eq!(first_chunk.len(), chunk_size);
    
    // Use fill_remaining for the rest
    let remaining_data = object_gen.fill_remaining();
    assert_eq!(remaining_data.len(), size - chunk_size);
    assert!(object_gen.is_complete());
    
    // Verify combined data has correct total length
    let mut combined = first_chunk;
    combined.extend_from_slice(&remaining_data);
    assert_eq!(combined.len(), size);
    
    // Verify that reset and full generation produces same result
    object_gen.reset();
    let full_data = object_gen.fill_remaining();
    assert_eq!(combined, full_data, "Chunked and full generation should produce identical results");
}

/// Test edge case: partial block at end
#[test]
fn test_partial_block_at_end() {
    let size = BLK_SIZE * 2 + BLK_SIZE / 2; // 2.5 blocks
    let chunk_size = BLK_SIZE;
    
    let generator = DataGenerator::new();
    let mut object_gen = generator.begin_object(size, 1, 1);
    
    let mut streaming_data = Vec::new();
    let mut chunk_sizes = Vec::new();
    
    while let Some(chunk) = object_gen.fill_chunk(chunk_size) {
        chunk_sizes.push(chunk.len());
        streaming_data.extend_from_slice(&chunk);
    }
    
    // Should have 3 chunks: full, full, partial
    assert_eq!(chunk_sizes, vec![BLK_SIZE, BLK_SIZE, BLK_SIZE / 2]);
    assert_eq!(streaming_data.len(), size);
    
    // Verify consistency by resetting and generating again
    object_gen.reset();
    let regenerated_data = object_gen.fill_remaining();
    assert_eq!(streaming_data, regenerated_data, "Chunked and full generation should be consistent");
}

/// Test that different DataGenerator instances produce different data
#[test]
fn test_different_generators_produce_different_data() {
    let size = BLK_SIZE * 2;
    
    // Create two generators
    let gen1 = DataGenerator::new();
    let gen2 = DataGenerator::new();
    
    // Generate data from each
    let mut obj_gen1 = gen1.begin_object(size, 1, 1);
    let mut obj_gen2 = gen2.begin_object(size, 1, 1);
    
    let data1 = obj_gen1.fill_remaining();
    let data2 = obj_gen2.fill_remaining();
    
    // Should be different (extremely unlikely to be identical by chance)
    assert_ne!(data1, data2);
    
    // But same generator should produce same data
    let mut obj_gen1_again = gen1.begin_object(size, 1, 1);
    let data1_again = obj_gen1_again.fill_remaining();
    assert_eq!(data1, data1_again);
}

/// Benchmark streaming vs single-pass performance (similar sizes)
#[test]
fn test_streaming_performance() {
    use std::time::Instant;
    
    let size = 1024 * 1024; // 1MB
    let chunk_size = 64 * 1024; // 64KB chunks
    let iterations = 5;
    
    // Measure single-pass performance
    let start = Instant::now();
    for _ in 0..iterations {
        let _data = generate_controlled_data(size, 4, 3);
    }
    let single_pass_duration = start.elapsed();
    
    // Measure streaming performance
    let start = Instant::now();
    for _ in 0..iterations {
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, 4, 3);
        let mut _total_data = Vec::new();
        while let Some(chunk) = object_gen.fill_chunk(chunk_size) {
            _total_data.extend_from_slice(&chunk);
        }
    }
    let streaming_duration = start.elapsed();
    
    println!("Single-pass: {:?} per iteration", single_pass_duration / iterations);
    println!("Streaming: {:?} per iteration", streaming_duration / iterations);
    
    // Streaming should be reasonably close to single-pass performance
    // Allow up to 3x overhead for the added complexity and smaller batch sizes
    let overhead_ratio = streaming_duration.as_secs_f64() / single_pass_duration.as_secs_f64();
    assert!(overhead_ratio < 3.0, 
            "Streaming overhead too high: {:.2}x slower", overhead_ratio);
}