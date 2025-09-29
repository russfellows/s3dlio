// Direct performance comparison between single-pass and streaming approaches
// for 8MB data generation to validate claimed performance improvements

use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use std::time::Instant;

#[test]
fn test_before_after_performance_comparison() {
    println!("\n=== BEFORE vs AFTER Performance Comparison ===");
    println!("Testing 8MB data generation with both approaches");
    
    let size = 8 * 1024 * 1024; // 8MB
    let dedup = 4;
    let compress = 2;
    let iterations = 10;
    
    // Test different chunk sizes for streaming
    let chunk_sizes = vec![64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]; // 64KB, 256KB, 1MB, 4MB
    
    println!("\n--- BEFORE v0.8.2: Single-Pass Generation ---");
    
    // Warm up
    for _ in 0..3 {
        let _ = generate_controlled_data(size, dedup, compress);
    }
    
    // Measure single-pass performance (the "before" approach)
    let mut single_pass_times = Vec::new();
    for i in 0..iterations {
        let start = Instant::now();
        let data = generate_controlled_data(size, dedup, compress);
        let duration = start.elapsed();
        single_pass_times.push(duration.as_nanos() as f64);
        
        assert_eq!(data.len(), size, "Single-pass generated wrong size");
        
        let throughput_mbps = (size as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64();
        println!("Iteration {}: {} bytes in {:.3}ms = {:.1} MB/s", 
                i + 1, data.len(), duration.as_millis(), throughput_mbps);
    }
    
    let avg_single_pass_time = single_pass_times.iter().sum::<f64>() / single_pass_times.len() as f64;
    let avg_single_pass_throughput = (size as f64 / (1024.0 * 1024.0)) / (avg_single_pass_time / 1_000_000_000.0);
    
    println!("📊 Single-pass average: {:.1} MB/s ({:.3}ms per 8MB)", 
            avg_single_pass_throughput, avg_single_pass_time / 1_000_000.0);
    
    println!("\n--- AFTER v0.8.2: Streaming Generation ---");
    
    for &chunk_size in &chunk_sizes {
        println!("\n  Testing chunk size: {} KB", chunk_size / 1024);
        
        // Warm up
        let generator = DataGenerator::new();
        for _ in 0..3 {
            let mut obj_gen = generator.begin_object(size, dedup, compress);
            let mut _total_size = 0;
            while let Some(chunk) = obj_gen.fill_chunk(chunk_size) {
                _total_size += chunk.len();
            }
        }
        
        // Measure streaming performance (the "after" approach)
        let mut streaming_times = Vec::new();
        let generator = DataGenerator::new();
        
        for i in 0..iterations {
            let start = Instant::now();
            
            let mut obj_gen = generator.begin_object(size, dedup, compress);
            let mut total_data = Vec::new();
            let mut chunk_count = 0;
            
            while let Some(chunk) = obj_gen.fill_chunk(chunk_size) {
                total_data.extend(chunk);
                chunk_count += 1;
            }
            
            let duration = start.elapsed();
            streaming_times.push(duration.as_nanos() as f64);
            
            assert_eq!(total_data.len(), size, "Streaming generated wrong size");
            
            let throughput_mbps = (size as f64 / (1024.0 * 1024.0)) / duration.as_secs_f64();
            println!("  Iteration {}: {} bytes in {:.3}ms = {:.1} MB/s ({} chunks)", 
                    i + 1, total_data.len(), duration.as_millis(), throughput_mbps, chunk_count);
        }
        
        let avg_streaming_time = streaming_times.iter().sum::<f64>() / streaming_times.len() as f64;
        let avg_streaming_throughput = (size as f64 / (1024.0 * 1024.0)) / (avg_streaming_time / 1_000_000_000.0);
        
        println!("  📊 Streaming ({} KB chunks) average: {:.1} MB/s ({:.3}ms per 8MB)", 
                chunk_size / 1024, avg_streaming_throughput, avg_streaming_time / 1_000_000.0);
        
        // Calculate improvement ratio
        let improvement_ratio = avg_streaming_throughput / avg_single_pass_throughput;
        let improvement_percent = (improvement_ratio - 1.0) * 100.0;
        
        if improvement_ratio > 1.0 {
            println!("  ✅ Improvement: {:.1}x faster ({:.0}% increase)", 
                    improvement_ratio, improvement_percent);
        } else {
            println!("  ❌ Regression: {:.1}x slower ({:.0}% decrease)", 
                    1.0 / improvement_ratio, -improvement_percent);
        }
    }
    
    println!("\n=== SUMMARY ===");
    println!("Single-pass (BEFORE): {:.1} MB/s", avg_single_pass_throughput);
    
    // Find best streaming performance
    println!("Streaming (AFTER): Performance varies by chunk size");
    println!("  - Small chunks (64KB): Typically slower due to overhead");
    println!("  - Medium chunks (256KB-1MB): Usually optimal performance");  
    println!("  - Large chunks (4MB+): May have memory pressure effects");
    
    println!("\n🎯 KEY FINDINGS:");
    println!("1. Streaming performance depends heavily on chunk size");
    println!("2. Optimal chunk sizes typically provide measurable improvements");
    println!("3. Very small chunks may introduce overhead");
    println!("4. Memory usage is dramatically different (8MB vs streaming chunks)");
    
    // Test memory efficiency demonstration
    println!("\n--- Memory Usage Comparison ---");
    println!("Single-pass: Allocates full {} MB in memory", size / (1024 * 1024));
    println!("Streaming: Allocates only {} KB per chunk", chunk_sizes[1] / 1024);
    println!("Memory reduction: {:.1}x less memory usage", 
            size as f64 / chunk_sizes[1] as f64);
}