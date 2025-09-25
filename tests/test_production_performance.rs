/// Large-scale performance test for production streaming scenarios
/// Tests the actual usage pattern: generating multiple 1-8 MB buffers in a loop
use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use std::time::Instant;

fn throughput_mb_per_sec(bytes: usize, duration_ms: f64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0) / (duration_ms / 1000.0)
}

#[test]
fn test_production_streaming_performance() {
    println!("=== Production Streaming Performance Test ===");
    
    let sizes = vec![
        1 * 1024 * 1024,   // 1 MB
        2 * 1024 * 1024,   // 2 MB  
        4 * 1024 * 1024,   // 4 MB
        8 * 1024 * 1024,   // 8 MB
    ];
    
    let iterations = 10; // Generate 10 buffers of each size
    
    for &size in &sizes {
        println!("\n--- Testing {} MB buffers (10 iterations) ---", size / (1024 * 1024));
        
        // Test single-pass generation (current baseline)
        let start = Instant::now();
        let mut total_bytes = 0;
        for _ in 0..iterations {
            let data = generate_controlled_data(size, 4, 2);
            total_bytes += data.len();
        }
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        let throughput = throughput_mb_per_sec(total_bytes, duration_ms);
        println!("Single-pass: {:.1} MB/s ({:.2}ms total for {} MB)", 
                throughput, duration_ms, total_bytes / (1024 * 1024));
        
        // Test streaming generation 
        let start = Instant::now();
        let mut total_bytes = 0;
        for _ in 0..iterations {
            let generator = DataGenerator::new();
            let mut obj_gen = generator.begin_object(size, 4, 2);
            
            // Generate in 256KB chunks (typical streaming pattern)
            let chunk_size = 256 * 1024;
            while let Some(chunk) = obj_gen.fill_chunk(chunk_size) {
                total_bytes += chunk.len();
            }
        }
        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        let throughput = throughput_mb_per_sec(total_bytes, duration_ms);
        println!("Streaming:   {:.1} MB/s ({:.2}ms total for {} MB)", 
                throughput, duration_ms, total_bytes / (1024 * 1024));
    }
}

#[test]
fn test_sustained_generation_performance() {
    println!("\n=== Sustained Generation Test (Target: Multi-GB/s) ===");
    
    let buffer_size = 4 * 1024 * 1024; // 4 MB buffers
    let target_total = 1024 * 1024 * 1024; // Generate 1 GB total
    let iterations = target_total / buffer_size;
    
    println!("Generating {} buffers of {} MB each (1 GB total)", iterations, buffer_size / (1024 * 1024));
    
    // Single-pass sustained test
    let start = Instant::now();
    let mut total_bytes = 0;
    for i in 0..iterations {
        let data = generate_controlled_data(buffer_size, 4, 2);
        total_bytes += data.len();
        
        if i % 64 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let current_throughput = throughput_mb_per_sec(total_bytes, elapsed * 1000.0);
            println!("Progress: {} MB generated, {:.1} MB/s", 
                    total_bytes / (1024 * 1024), current_throughput);
        }
    }
    
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let throughput = throughput_mb_per_sec(total_bytes, duration_ms);
    println!("\nSustained single-pass: {:.1} MB/s ({:.2}s total)", 
            throughput, duration_ms / 1000.0);
    
    if throughput >= 1000.0 {
        println!("✅ ACHIEVED MULTI-GB/s TARGET!");
    } else {
        println!("⚠️  Below multi-GB/s target ({:.1} MB/s)", throughput);
    }
}

#[test]
fn test_rng_overhead_measurement() {
    println!("\n=== RNG Overhead Detailed Analysis ===");
    
    let size = 4 * 1024 * 1024; // 4 MB buffer
    let iterations = 100;
    
    // Measure just the data generation without any RNG operations
    // This gives us the theoretical maximum
    println!("Testing {} iterations of {} MB buffers", iterations, size / (1024 * 1024));
    
    let start = Instant::now();
    let mut total_bytes = 0;
    for _ in 0..iterations {
        let data = generate_controlled_data(size, 4, 2);
        total_bytes += data.len();
    }
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    let throughput = throughput_mb_per_sec(total_bytes, duration_ms);
    
    println!("Current implementation: {:.1} MB/s", throughput);
    println!("Total time: {:.2}ms for {} MB", duration_ms, total_bytes / (1024 * 1024));
    
    // Calculate overhead per buffer
    let time_per_buffer = duration_ms / iterations as f64;
    let mb_per_buffer = (size as f64) / (1024.0 * 1024.0);
    println!("Per buffer: {:.2}ms for {:.1} MB = {:.1} MB/s per buffer", 
            time_per_buffer, mb_per_buffer, mb_per_buffer / (time_per_buffer / 1000.0));
    
    // Estimate RNG overhead
    let block_size = 65536; // BLK_SIZE
    let blocks_per_buffer = (size + block_size - 1) / block_size;
    let unique_blocks = blocks_per_buffer / 4; // dedup_factor = 4
    println!("Blocks per buffer: {}, Unique blocks: {}", blocks_per_buffer, unique_blocks);
    println!("RNG creations per buffer: {} (current) vs {} (optimized)", 
            blocks_per_buffer, unique_blocks);
}