/// Error handling and edge case tests for streaming data generation
/// Ensures robust behavior under adverse conditions

use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};

#[test]
fn test_zero_size_handling() {
    println!("=== Zero Size Handling ===");
    
    // Test zero size generation
    let data = generate_controlled_data(0, 4, 2);
    println!("Zero size single-pass result: {} bytes", data.len());
    
    // Test streaming with zero size
    let generator = DataGenerator::new();
    let mut obj_gen = generator.begin_object(0, 4, 2);
    let mut total_bytes = 0;
    let mut chunk_count = 0;
    
    while let Some(chunk) = obj_gen.fill_chunk(1024) {
        total_bytes += chunk.len();
        chunk_count += 1;
        
        // Prevent infinite loops
        if chunk_count > 100 {
            panic!("Too many chunks for zero-size generation");
        }
    }
    
    println!("Zero size streaming result: {} bytes in {} chunks", total_bytes, chunk_count);
    println!("✅ Zero size handling completed");
}

#[test]
fn test_extreme_parameters() {
    println!("\n=== Extreme Parameter Testing ===");
    
    let test_cases = vec![
        ("Max dedup", 1024 * 1024, 1_000_000, 1),
        ("Max compress", 1024 * 1024, 1, 1_000_000),
        ("Both max", 1024 * 1024, 1_000_000, 1_000_000),
        ("Large size", 100 * 1024 * 1024, 4, 2), // 100MB
    ];
    
    for (name, size, dedup, compress) in test_cases {
        println!("Testing {}: size={}, dedup={}, compress={}", name, size, dedup, compress);
        
        let start = Instant::now();
        
        // Test single-pass (but limit to reasonable time)
        let data = generate_controlled_data(size, dedup, compress);
        let generation_time = start.elapsed();
        
        println!("  {} generated {} bytes in {:.2}ms", 
                name, data.len(), generation_time.as_secs_f64() * 1000.0);
        
        // Basic sanity checks
        assert!(!data.is_empty(), "{} generated empty data", name);
        
        if generation_time.as_secs() < 10 {
            println!("  ✅ {} completed in reasonable time", name);
        } else {
            println!("  ⚠️ {} took longer than expected: {:.2}s", name, generation_time.as_secs_f64());
        }
    }
}

#[test]
fn test_memory_pressure() {
    println!("\n=== Memory Pressure Testing ===");
    
    let buffer_size = 16 * 1024 * 1024; // 16MB buffers
    let num_concurrent = 10;
    
    println!("Testing {} concurrent {} MB generations", num_concurrent, buffer_size / (1024 * 1024));
    
    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    
    for thread_id in 0..num_concurrent {
        let results_clone = Arc::clone(&results);
        
        let handle = thread::spawn(move || {
            let start = Instant::now();
            
            // Generate large buffer using streaming to control memory usage
            let generator = DataGenerator::new();
            let mut obj_gen = generator.begin_object(buffer_size, 4, 2);
            let mut total_bytes = 0;
            let mut max_chunk_size = 0;
            
            while let Some(chunk) = obj_gen.fill_chunk(256 * 1024) { // 256KB chunks
                total_bytes += chunk.len();
                max_chunk_size = max_chunk_size.max(chunk.len());
                
                // Simulate processing delay
                thread::sleep(Duration::from_millis(1));
            }
            
            let duration = start.elapsed();
            
            let mut results = results_clone.lock().unwrap();
            results.push((thread_id, total_bytes, max_chunk_size, duration));
            
            (total_bytes, duration)
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads and collect results
    let mut total_generated = 0;
    let mut max_duration = Duration::from_secs(0);
    
    for handle in handles {
        let (bytes, duration) = handle.join().unwrap();
        total_generated += bytes;
        max_duration = max_duration.max(duration);
    }
    
    println!("📊 Memory pressure results:");
    println!("Total generated: {} MB", total_generated / (1024 * 1024));
    println!("Max thread duration: {:.2}s", max_duration.as_secs_f64());
    println!("Peak memory estimate: ~{} MB", (num_concurrent * 256) / 1024); // Conservative chunk-based estimate
    
    assert!(total_generated > 0, "No data generated under memory pressure");
    assert!(max_duration.as_secs() < 30, "Generation took too long under memory pressure");
    
    println!("✅ Memory pressure test completed successfully");
}

#[test]
fn test_rapid_generator_creation() {
    println!("\n=== Rapid Generator Creation Testing ===");
    
    let num_generators = 1000;
    let small_size = 64 * 1024; // 64KB per generation
    
    println!("Creating {} generators rapidly with {} KB each", num_generators, small_size / 1024);
    
    let start = Instant::now();
    let mut total_bytes = 0;
    
    for i in 0..num_generators {
        let generator = DataGenerator::new();
        let mut obj_gen = generator.begin_object(small_size, 2, 2);
        
        while let Some(chunk) = obj_gen.fill_chunk(8192) {
            total_bytes += chunk.len();
        }
        
        if i % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            println!("  Created {} generators in {:.3}s", i + 1, elapsed);
        }
    }
    
    let duration = start.elapsed();
    let creation_rate = num_generators as f64 / duration.as_secs_f64();
    
    println!("📊 Rapid creation results:");
    println!("Created {} generators in {:.3}s", num_generators, duration.as_secs_f64());
    println!("Creation rate: {:.1} generators/second", creation_rate);
    println!("Total data generated: {} MB", total_bytes / (1024 * 1024));
    
    assert!(creation_rate > 100.0, "Generator creation rate too slow: {:.1}/s", creation_rate);
    
    println!("✅ Rapid generator creation successful");
}

#[test]
fn test_chunk_size_variations() {
    println!("\n=== Chunk Size Variation Testing ===");
    
    let object_size = 4 * 1024 * 1024; // 4MB object
    let chunk_sizes = vec![
        1,           // Minimal chunks
        1024,        // 1KB chunks  
        8192,        // 8KB chunks
        64 * 1024,   // 64KB chunks
        256 * 1024,  // 256KB chunks
        1024 * 1024, // 1MB chunks
        2 * 1024 * 1024, // 2MB chunks (larger than typical)
        object_size + 1, // Larger than object
    ];
    
    let generator = DataGenerator::new();
    
    for &chunk_size in &chunk_sizes {
        println!("Testing chunk size: {} bytes", chunk_size);
        
        let start = Instant::now();
        let mut obj_gen = generator.begin_object(object_size, 4, 2);
        let mut total_bytes = 0;
        let mut chunk_count = 0;
        let mut actual_chunk_sizes = Vec::new();
        
        while let Some(chunk) = obj_gen.fill_chunk(chunk_size) {
            total_bytes += chunk.len();
            chunk_count += 1;
            actual_chunk_sizes.push(chunk.len());
            
            // Safety check - for very small chunks, allow more iterations but with timeout
            if chunk_count > 100000 {
                println!("⚠️ Stopping after {} chunks for chunk size {} (likely edge case)", chunk_count, chunk_size);
                break;
            }
        }
        
        let duration = start.elapsed();
        let throughput_mb_s = (total_bytes as f64) / (1024.0 * 1024.0) / duration.as_secs_f64();
        
        println!("  Generated {} bytes in {} chunks ({:.2} MB/s)", 
                total_bytes, chunk_count, throughput_mb_s);
        
        if !actual_chunk_sizes.is_empty() {
            let min_chunk = *actual_chunk_sizes.iter().min().unwrap();
            let max_chunk = *actual_chunk_sizes.iter().max().unwrap();
            println!("  Chunk sizes: {} to {} bytes", min_chunk, max_chunk);
        }
        
        // Verify reasonable behavior
        assert!(total_bytes > 0, "No data generated for chunk size {}", chunk_size);
        assert!(chunk_count > 0, "No chunks generated for chunk size {}", chunk_size);
        
        // For reasonable chunk sizes, verify efficiency
        if chunk_size >= 1024 && chunk_size <= 1024 * 1024 {
            assert!(throughput_mb_s > 100.0, "Low throughput for chunk size {}: {:.2} MB/s", 
                    chunk_size, throughput_mb_s);
        }
    }
    
    println!("✅ Chunk size variation testing completed");
}

#[test]
fn test_streaming_consistency_across_restarts() {
    println!("\n=== Streaming Consistency Across Restarts ===");
    
    let total_size = 2 * 1024 * 1024; // 2MB total
    let chunk_size = 128 * 1024; // 128KB chunks
    
    // Generate in one continuous stream
    let generator1 = DataGenerator::new();
    let mut obj_gen1 = generator1.begin_object(total_size, 4, 2);
    let mut continuous_data = Vec::new();
    
    while let Some(chunk) = obj_gen1.fill_chunk(chunk_size) {
        continuous_data.extend(chunk);
    }
    
    // Generate with the same generator but restarted
    let mut obj_gen2 = generator1.begin_object(total_size, 4, 2);
    let mut restarted_data = Vec::new();
    
    while let Some(chunk) = obj_gen2.fill_chunk(chunk_size) {
        restarted_data.extend(chunk);
    }
    
    // Verify they're identical (same generator instance should produce same results)
    assert_eq!(continuous_data.len(), restarted_data.len(), 
               "Size mismatch between continuous and restarted generation");
    assert_eq!(continuous_data, restarted_data,
               "Data mismatch between continuous and restarted generation");
    
    println!("Generated {} bytes continuously and restarted - identical results", 
             continuous_data.len());
    
    // Test with different generators (should produce different data)
    let generator3 = DataGenerator::new();
    let mut obj_gen3 = generator3.begin_object(total_size, 4, 2);
    let mut different_data = Vec::new();
    
    while let Some(chunk) = obj_gen3.fill_chunk(chunk_size) {
        different_data.extend(chunk);
    }
    
    assert_ne!(continuous_data, different_data,
               "Different generators produced identical data (unexpected)");
    
    println!("Different generator produced different data as expected");
    println!("✅ Streaming consistency verified");
}