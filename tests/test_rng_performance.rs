use std::time::Instant;
use rand::{Rng, SeedableRng};

use s3dlio::{
    data_gen::{generate_controlled_data, DataGenerator},
    constants::BLK_SIZE,
};

#[test]
fn performance_baseline_single_pass() {
    println!("=== RNG Performance Baseline Tests ===");
    
    let sizes = vec![BLK_SIZE, BLK_SIZE * 4, BLK_SIZE * 16, BLK_SIZE * 64];
    
    for size in sizes {
        let start = Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let _data = generate_controlled_data(size, 1, 1);
        }
        
        let elapsed = start.elapsed();
        let avg_time = elapsed / iterations;
        let throughput = (size as f64) / avg_time.as_secs_f64() / (1024.0 * 1024.0);
        
        println!("Single-pass: {} bytes, avg: {:?}, throughput: {:.1} MB/s", 
            size, avg_time, throughput);
    }
}

#[test]
fn performance_baseline_streaming() {
    println!("=== Streaming Performance Tests ===");
    
    let sizes = vec![BLK_SIZE, BLK_SIZE * 4, BLK_SIZE * 16, BLK_SIZE * 64];
    
    for size in sizes {
        let start = Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let generator = DataGenerator::new();
            let mut object_gen = generator.begin_object(size, 1, 1);
            let _data = object_gen.fill_remaining();
        }
        
        let elapsed = start.elapsed();
        let avg_time = elapsed / iterations;
        let throughput = (size as f64) / avg_time.as_secs_f64() / (1024.0 * 1024.0);
        
        println!("Streaming: {} bytes, avg: {:?}, throughput: {:.1} MB/s", 
            size, avg_time, throughput);
    }
}

#[test] 
fn performance_streaming_chunked() {
    println!("=== Streaming Chunked Performance Tests ===");
    
    let size = BLK_SIZE * 16; // 1MB
    let chunk_sizes = vec![4096, 16384, 65536, 262144]; // 4KB to 256KB
    
    for chunk_size in chunk_sizes {
        let start = Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let generator = DataGenerator::new();
            let mut object_gen = generator.begin_object(size, 1, 1);
            let mut _total_data = Vec::new();
            
            while let Some(chunk) = object_gen.fill_chunk(chunk_size) {
                _total_data.extend_from_slice(&chunk);
            }
        }
        
        let elapsed = start.elapsed();
        let avg_time = elapsed / iterations;
        let throughput = (size as f64) / avg_time.as_secs_f64() / (1024.0 * 1024.0);
        
        println!("Chunked ({} KB): avg: {:?}, throughput: {:.1} MB/s", 
            chunk_size / 1024, avg_time, throughput);
    }
}

#[test]
fn performance_rng_comparison() {
    println!("=== RNG Performance Comparison ===");
    
    let block_count = 256;
    let block_size = 4096;
    let iterations = 100;
    
    // Test current approach: SmallRng per block with seeding
    let start = Instant::now();
    for _ in 0..iterations {
        let mut _total_bytes = 0;
        for i in 0..block_count {
            let seed = i as u64;
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut data = vec![0u8; block_size];
            rng.fill(&mut data[..]);
            _total_bytes += data.len();
        }
    }
    let elapsed_per_block = start.elapsed();
    let per_block_throughput = (block_count * block_size * iterations) as f64 
        / elapsed_per_block.as_secs_f64() / (1024.0 * 1024.0);
    
    // Test optimized approach: Single RNG reused
    let start = Instant::now();
    for _ in 0..iterations {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(12345);
        let mut _total_bytes = 0;
        for _ in 0..block_count {
            let mut data = vec![0u8; block_size];
            rng.fill(&mut data[..]);
            _total_bytes += data.len();
        }
    }
    let elapsed_reused = start.elapsed();
    let reused_throughput = (block_count * block_size * iterations) as f64 
        / elapsed_reused.as_secs_f64() / (1024.0 * 1024.0);
    
    println!("RNG per block: {:?}, {:.1} MB/s", elapsed_per_block, per_block_throughput);
    println!("RNG reused: {:?}, {:.1} MB/s", elapsed_reused, reused_throughput);
    println!("Speedup from reuse: {:.2}x", reused_throughput / per_block_throughput);
}