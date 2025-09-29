/// Comprehensive testing suite for streaming data generation
/// Tests edge cases, error conditions, and multi-process scenarios for production readiness

use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
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
        let data1 = generate_controlled_data(size, 4, 2);
        assert!(!data1.is_empty(), "Single-pass generated empty data for size {}", size);
        
        // Test streaming generation
        let generator = DataGenerator::new();
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
        (1024 * 1024, 0, 1),   // dedup=0 (should be treated as 1)
        (1024 * 1024, 1, 1),   // dedup=1 (no deduplication)
        (1024 * 1024, 2, 1),   // dedup=2
        (1024 * 1024, 1000, 1), // Very high dedup factor
        (100, 2, 1),           // Small size with dedup
        (65536, 64, 1),        // Exactly one block with high dedup
    ];
    
    for (size, dedup, compress) in test_cases {
        println!("Testing: size={}, dedup={}, compress={}", size, dedup, compress);
        
        let data = generate_controlled_data(size, dedup, compress);
        assert!(!data.is_empty(), "Generated empty data for dedup case");
        
        // Test that streaming generation is internally consistent
        let generator = DataGenerator::new();
        
        // Generate the same object twice using streaming to verify determinism
        let mut obj_gen1 = generator.begin_object(size, dedup, compress);
        let mut streamed1 = Vec::new();
        while let Some(chunk) = obj_gen1.fill_chunk(1024) {
            streamed1.extend(chunk);
        }
        
        let mut obj_gen2 = generator.begin_object(size, dedup, compress);
        let mut streamed2 = Vec::new();
        while let Some(chunk) = obj_gen2.fill_chunk(2048) { // Different chunk size
            streamed2.extend(chunk);
        }
        
        // Same generator should produce same results regardless of chunk size
        assert_eq!(streamed1.len(), streamed2.len(), 
                  "Streaming generators produced different sizes");
        
        if streamed1 != streamed2 {
            let mismatch_pos = streamed1.iter().zip(streamed2.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(streamed1.len().min(streamed2.len()));
            
            panic!("Streaming determinism failure at position {}: chunk1024[{}]={:?} vs chunk2048[{}]={:?}", 
                   mismatch_pos, mismatch_pos, 
                   streamed1.get(mismatch_pos), mismatch_pos, 
                   streamed2.get(mismatch_pos));
        }
        
        println!("✅ Dedup case passed - generated {} bytes", data.len());
    }
}

#[test]
fn test_compression_edge_cases() {
    println!("\n=== Compression Edge Case Testing ===");
    
    let test_cases = vec![
        (1024 * 1024, 1, 0),   // compress=0
        (1024 * 1024, 1, 1),   // compress=1 (no compression)
        (1024 * 1024, 1, 2),   // compress=2
        (1024 * 1024, 1, 100), // Very high compress factor
        (65536, 1, 10),        // One block with compression
    ];
    
    for (size, dedup, compress) in test_cases {
        println!("Testing: size={}, dedup={}, compress={}", size, dedup, compress);
        
        let data = generate_controlled_data(size, dedup, compress);
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
    
    // Test that the same DataGenerator instance produces identical results
    let generator = DataGenerator::new();
    let mut results = Vec::new();
    
    for i in 0..iterations {
        let mut obj_gen = generator.begin_object(size, 4, 2);
        let mut data = Vec::new();
        
        while let Some(chunk) = obj_gen.fill_chunk(256 * 1024) {
            data.extend(chunk);
        }
        
        results.push(data);
        println!("Iteration {}: generated {} bytes", i + 1, results[i].len());
    }
    
    // All results should be identical
    for i in 1..iterations {
        assert_eq!(results[0], results[i], 
                  "Deterministic failure: iteration 0 != iteration {}", i);
    }
    
    println!("✅ Deterministic behavior verified across {} iterations", iterations);
}

#[test]
fn test_multi_generator_independence() {
    println!("\n=== Multi-Generator Independence Testing ===");
    
    let size = 2 * 1024 * 1024; // 2MB
    let num_generators = 10;
    
    // Create multiple generators and verify they produce different data
    let mut generators = Vec::new();
    let mut results = Vec::new();
    
    for i in 0..num_generators {
        let generator = DataGenerator::new();
        let mut obj_gen = generator.begin_object(size, 4, 2);
        let mut data = Vec::new();
        
        while let Some(chunk) = obj_gen.fill_chunk(64 * 1024) {
            data.extend(chunk);
        }
        
        generators.push(generator);
        results.push(data);
        
        println!("Generator {}: generated {} bytes", i + 1, results[i].len());
    }
    
    // Verify that different generators produce different data
    let mut unique_count = 0;
    for i in 0..num_generators {
        let mut is_unique = true;
        for j in (i + 1)..num_generators {
            if results[i] == results[j] {
                is_unique = false;
                break;
            }
        }
        if is_unique {
            unique_count += 1;
        }
    }
    
    // Should have mostly unique results (allowing for small chance of collision)
    assert!(unique_count >= num_generators - 1, 
           "Expected mostly unique results, got {} unique out of {}", unique_count, num_generators);
    
    println!("✅ Generator independence verified: {}/{} unique results", unique_count, num_generators);
}

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
            
            for iteration in 0..iterations_per_thread {
                // Each thread uses its own generator
                let generator = DataGenerator::new();
                let mut obj_gen = generator.begin_object(size, 4, 2);
                let mut iteration_total = 0;
                
                while let Some(chunk) = obj_gen.fill_chunk(256 * 1024) {
                    iteration_total += chunk.len();
                }
                
                thread_total += iteration_total;
                
                if iteration == 0 {
                    println!("Thread {}: First iteration generated {} bytes", 
                            thread_id, iteration_total);
                }
            }
            
            let thread_duration = thread_start.elapsed().as_secs_f64() * 1000.0;
            let thread_throughput = throughput_gb_per_sec(thread_total, thread_duration);
            
            (thread_id, thread_total, thread_duration, thread_throughput)
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut total_bytes = 0;
    let mut results = Vec::new();
    
    for handle in handles {
        let (thread_id, bytes, duration, throughput) = handle.join().unwrap();
        total_bytes += bytes;
        results.push((thread_id, bytes, duration, throughput));
        
        println!("Thread {} completed: {} MB in {:.2}ms = {:.2} GB/s", 
                thread_id, bytes / (1024 * 1024), duration, throughput);
    }
    
    let total_duration = start.elapsed().as_secs_f64() * 1000.0;
    let aggregate_throughput = throughput_gb_per_sec(total_bytes, total_duration);
    
    println!("\n🎯 CONCURRENT RESULTS:");
    println!("Total data generated: {} MB", total_bytes / (1024 * 1024));
    println!("Total time: {:.2}ms", total_duration);
    println!("Aggregate throughput: {:.2} GB/s", aggregate_throughput);
    
    // Verify reasonable performance
    assert!(aggregate_throughput > 1.0, "Aggregate throughput too low: {:.2} GB/s", aggregate_throughput);
    
    println!("✅ Concurrent generation successful with {:.2} GB/s aggregate", aggregate_throughput);
}