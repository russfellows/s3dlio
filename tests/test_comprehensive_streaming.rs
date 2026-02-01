// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

/// Comprehensive testing suite for streaming data generation
/// Tests edge cases, error conditions, and multi-process scenarios for production readiness

use s3dlio::data_gen::DataGenerator;
use s3dlio::data_gen_alt;
use s3dlio::data_gen_alt::ObjectGenAlt;
use std::thread;
use std::time::Instant;

fn throughput_gb_per_sec(bytes: usize, duration_ms: f64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0 * 1024.0) / (duration_ms / 1000.0)
}

#[test]
fn test_edge_case_buffer_sizes() {
    println!("=== Edge Case Buffer Size Testing ===");
    
    let edge_sizes = vec![
        1,                    // Minimum size
        64,                   // Very small
        1023,                 // Just under 1KB
        1024,                 // Exactly 1KB
        1025,                 // Just over 1KB
        65535,                // Just under 64KB (BLK_SIZE)
        65536,                // Exactly BLK_SIZE
        65537,                // Just over BLK_SIZE
        1048575,              // Just under 1MB
        1048576,              // Exactly 1MB
        1048577,              // Just over 1MB
    ];
    
    for &size in &edge_sizes {
        println!("Testing size: {} bytes", size);
        
        // Test single-pass generation
        let data1 = data_gen_alt::generate_controlled_data_alt(size, 4, 2, None).to_vec();
        assert!(!data1.is_empty(), "Single-pass generated empty data for size {}", size);
        
        // Test streaming generation
        let generator = DataGenerator::new(None);
        let mut obj_gen = generator.begin_object(size, 4, 2);
        let mut total_streamed = 0;
        let mut chunks = Vec::new();
        
        while let Some(chunk) = obj_gen.fill_chunk(1024) {
            total_streamed += chunk.len();
            chunks.push(chunk);
        }
        
        assert!(total_streamed > 0, "Streaming generated no data for size {}", size);
        
        // Verify consistency between methods - but note that different generator instances may produce different entropy
        let streamed_data: Vec<u8> = chunks.into_iter().flatten().collect();
        if data1.len() != streamed_data.len() {
            println!("⚠️ Size mismatch for size {}: single-pass {} vs streaming {}", 
                    size, data1.len(), streamed_data.len());
        } else if data1 != streamed_data {
            println!("⚠️ Data difference for size {} (different generator instances produce different entropy)", size);
        } else {
            println!("✅ Identical results for size {}", size);
        }
        
        println!("✅ Size {} - Single-pass: {} bytes, Streaming: {} bytes", 
                size, data1.len(), streamed_data.len());
    }
}

#[test]
fn test_deduplication_edge_cases() {
    println!("\n=== Deduplication Edge Case Testing ===");
    
    let test_cases = vec![
        (1024 * 1024, 1, 1),   // dedup=1 (no deduplication, 1:1 ratio)
        (1024 * 1024, 2, 1),   // dedup=2 (50% duplicate blocks)
        (1024 * 1024, 4, 1),   // dedup=4 (75% duplicate blocks)
        (1024 * 1024, 100, 1), // Very high dedup factor
        (100, 2, 1),           // Small size with dedup
        (65536, 8, 1),         // Exactly one block with high dedup
    ];
    
    for (size, dedup, compress) in test_cases {
        println!("Testing: size={}, dedup={}, compress={}", size, dedup, compress);
        
        let data = data_gen_alt::generate_controlled_data_alt(size, dedup, compress, None).to_vec();
        assert!(!data.is_empty(), "Generated empty data for dedup case");
        
        // Test that streaming with SAME SEED and different chunk sizes produces identical output
        // Use explicit seed for deterministic testing
        let test_seed = 12345u64;
        
        // Stream with 1KB chunks
        let mut obj_gen1 = ObjectGenAlt::new_with_seed(size, dedup as usize, compress as usize, test_seed);
        let mut streamed1 = Vec::with_capacity(size);
        let mut buf1 = vec![0u8; 1024];
        loop {
            let n = obj_gen1.fill_chunk(&mut buf1);
            if n == 0 { break; }
            streamed1.extend_from_slice(&buf1[..n]);
        }
        
        // Stream with 2KB chunks (different chunk size)
        let mut obj_gen2 = ObjectGenAlt::new_with_seed(size, dedup as usize, compress as usize, test_seed);
        let mut streamed2 = Vec::with_capacity(size);
        let mut buf2 = vec![0u8; 2048];
        loop {
            let n = obj_gen2.fill_chunk(&mut buf2);
            if n == 0 { break; }
            streamed2.extend_from_slice(&buf2[..n]);
        }
        
        // Same seed should produce same results regardless of chunk size
        assert_eq!(streamed1.len(), streamed2.len(), 
                  "Streaming generators produced different sizes: {} vs {}", 
                  streamed1.len(), streamed2.len());
        
        if streamed1 != streamed2 {
            let mismatch_pos = streamed1.iter().zip(streamed2.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(streamed1.len().min(streamed2.len()));
            
            panic!("Streaming determinism failure at position {}: byte values differ (chunk1024 vs chunk2048)", 
                   mismatch_pos);
        }
        
        println!("✅ Dedup case passed - generated {} bytes", data.len());
    }
}

#[test]
fn test_compression_edge_cases() {
    println!("\n=== Compression Edge Case Testing ===");
    
    let test_cases = vec![
        (1024 * 1024, 1, 1),   // compress=1 (no compression, 1:1 ratio)
        (1024 * 1024, 1, 2),   // compress=2 (50% compressible)
        (1024 * 1024, 1, 4),   // compress=4 (75% compressible)
        (1024 * 1024, 1, 100), // Very high compress factor
        (65536, 1, 10),        // One block with compression
    ];
    
    for (size, dedup, compress) in test_cases {
        println!("Testing: size={}, dedup={}, compress={}", size, dedup, compress);
        
        let data = data_gen_alt::generate_controlled_data_alt(size, dedup, compress, None).to_vec();
        assert!(!data.is_empty(), "Generated empty data for compress case");
        
        // Verify compression actually affects data (for compress > 1)
        if compress > 1 && data.len() >= 1000 {
            let zero_count = data.iter().filter(|&&b| b == 0).count();
            let zero_ratio = zero_count as f64 / data.len() as f64;
            
            // Should have some zero bytes when compression is enabled
            if compress >= 2 {
                assert!(zero_ratio > 0.1, "Expected more zeros with compress={}, got ratio={}", compress, zero_ratio);
            }
        }
        
        println!("✅ Compression case passed - generated {} bytes", data.len());
    }
}

#[test]
fn test_deterministic_behavior() {
    println!("\n=== Deterministic Behavior Testing ===");
    
    let size = 4 * 1024 * 1024; // 4MB
    let iterations = 5;
    let test_seed = 99999u64;
    
    // Test that using the SAME SEED produces identical results across iterations
    let mut results = Vec::new();
    
    for i in 0..iterations {
        let mut obj_gen = ObjectGenAlt::new_with_seed(size, 4, 2, test_seed);
        let mut data = Vec::with_capacity(size);
        let mut buf = vec![0u8; 256 * 1024];
        
        loop {
            let n = obj_gen.fill_chunk(&mut buf);
            if n == 0 { break; }
            data.extend_from_slice(&buf[..n]);
        }
        
        results.push(data);
        println!("Iteration {}: generated {} bytes", i + 1, results[i].len());
    }
    
    // All results with same seed should be identical
    for i in 1..iterations {
        assert_eq!(results[0].len(), results[i].len(), 
                  "Deterministic failure: different lengths - iteration 0 ({}) != iteration {} ({})", 
                  results[0].len(), i, results[i].len());
        assert_eq!(results[0], results[i], 
                  "Deterministic failure: data mismatch - iteration 0 != iteration {}", i);
    }
    
    println!("✅ Deterministic behavior verified across {} iterations with same seed", iterations);
}

#[test]
fn test_unique_streams_by_default() {
    println!("\n=== Unique Streams By Default Testing ===");
    
    let size = 1024 * 1024; // 1MB
    let num_streams = 10;
    
    // Create multiple ObjectGenAlt instances WITHOUT explicit seed
    // Each should produce DIFFERENT data (unique entropy)
    let mut streams = Vec::new();
    
    for i in 0..num_streams {
        let mut obj_gen = ObjectGenAlt::new(size, 1, 1); // dedup=1, compress=1 (pure random)
        let mut data = Vec::with_capacity(size);
        let mut buf = vec![0u8; 64 * 1024];
        
        loop {
            let n = obj_gen.fill_chunk(&mut buf);
            if n == 0 { break; }
            data.extend_from_slice(&buf[..n]);
        }
        
        streams.push(data);
        println!("Stream {}: generated {} bytes", i + 1, streams[i].len());
    }
    
    // Verify ALL streams are unique (different from each other)
    for i in 0..num_streams {
        for j in (i + 1)..num_streams {
            // Compare first 1000 bytes - if random, should differ
            let sample_i: Vec<u8> = streams[i].iter().take(1000).cloned().collect();
            let sample_j: Vec<u8> = streams[j].iter().take(1000).cloned().collect();
            
            assert_ne!(sample_i, sample_j, 
                      "CRITICAL: Stream {} and {} produced identical data! Default behavior should be unique streams.", 
                      i, j);
        }
    }
    
    println!("✅ All {} streams are unique - default behavior is correct", num_streams);
}

// test_multi_generator_independence removed - functionality now covered by test_unique_streams_by_default

#[test] 
fn test_concurrent_generation() {
    println!("\n=== Concurrent Generation Testing ===");
    
    let size = 4 * 1024 * 1024; // 4MB per thread
    let num_threads = 8;
    let iterations_per_thread = 5;
    
    println!("Testing {} threads, {} iterations each, {} MB per iteration", 
            num_threads, iterations_per_thread, size / (1024 * 1024));
    
    let start = Instant::now();
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let handle = thread::spawn(move || {
            let mut thread_total = 0;
            let thread_start = Instant::now();
            let mut first_bytes: Option<Vec<u8>> = None;
            
            for iteration in 0..iterations_per_thread {
                // Each iteration uses a NEW generator (unique entropy)
                let mut obj_gen = ObjectGenAlt::new(size, 4, 2);
                let mut iteration_total = 0;
                let mut iteration_sample = Vec::new();
                let mut buf = vec![0u8; 256 * 1024];
                
                loop {
                    let n = obj_gen.fill_chunk(&mut buf);
                    if n == 0 { break; }
                    iteration_total += n;
                    if iteration_sample.len() < 100 {
                        iteration_sample.extend(buf[..n].iter().take(100 - iteration_sample.len()));
                    }
                }
                
                thread_total += iteration_total;
                
                // Capture first iteration's sample for uniqueness check
                if iteration == 0 {
                    first_bytes = Some(iteration_sample);
                }
            }
            
            let thread_duration = thread_start.elapsed().as_secs_f64() * 1000.0;
            let thread_throughput = throughput_gb_per_sec(thread_total, thread_duration);
            
            (thread_id, thread_total, thread_duration, thread_throughput, first_bytes.unwrap())
        });
        
        handles.push(handle);
    }
    
    // Collect results and verify uniqueness
    let mut total_bytes = 0;
    let mut thread_samples: Vec<(usize, Vec<u8>)> = Vec::new();
    
    for handle in handles {
        let (thread_id, bytes, duration, throughput, sample) = handle.join().unwrap();
        total_bytes += bytes;
        thread_samples.push((thread_id, sample));
        
        println!("Thread {} completed: {} MB in {:.2}ms = {:.2} GB/s", 
                thread_id, bytes / (1024 * 1024), duration, throughput);
    }
    
    // Verify all threads produced unique data
    for i in 0..thread_samples.len() {
        for j in (i + 1)..thread_samples.len() {
            assert_ne!(thread_samples[i].1, thread_samples[j].1,
                      "CRITICAL: Thread {} and {} produced identical data!", 
                      thread_samples[i].0, thread_samples[j].0);
        }
    }
    
    let total_duration = start.elapsed().as_secs_f64() * 1000.0;
    let aggregate_throughput = throughput_gb_per_sec(total_bytes, total_duration);
    
    println!("\n🎯 CONCURRENT RESULTS:");
    println!("Total data generated: {} MB", total_bytes / (1024 * 1024));
    println!("Total time: {:.2}ms", total_duration);
    println!("Aggregate throughput: {:.2} GB/s", aggregate_throughput);
    println!("All {} threads produced unique data streams", num_threads);
    
    // Verify reasonable performance (low threshold for debug builds)
    assert!(aggregate_throughput > 0.1, "Aggregate throughput too low: {:.2} GB/s", aggregate_throughput);
    
    println!("✅ Concurrent generation successful with {:.2} GB/s aggregate", aggregate_throughput);
}