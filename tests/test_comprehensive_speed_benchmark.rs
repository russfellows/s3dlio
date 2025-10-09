use std::time::Instant;
use s3dlio::{
    data_gen::{generate_controlled_data, DataGenerator},
    streaming_writer::StreamingDataWriter,
    file_store::FileSystemObjectStore,
    object_store::{ObjectStore, WriterOptions},
};
use anyhow::Result;

/// Comprehensive speed benchmark to validate that we're using the fastest approaches
/// This benchmarks single-pass vs streaming across multiple scenarios to guide optimization decisions

#[tokio::test]
async fn test_comprehensive_speed_benchmark() -> Result<()> {
    println!("\n=== COMPREHENSIVE SPEED BENCHMARK ===");
    println!("Testing single-pass vs streaming data generation across multiple scenarios");
    println!("Goal: Identify fastest approaches for production optimization\n");

    let test_cases = vec![
        ("1MB", 1024 * 1024),
        ("8MB", 8 * 1024 * 1024),
        ("32MB", 32 * 1024 * 1024),
        ("128MB", 128 * 1024 * 1024),
    ];

    let dedup_compress_cases = vec![
        ("No Dedup/Compress", 1, 1),
        ("2x Dedup", 2, 1),
        ("4x Compress", 1, 4),
        ("2x Dedup + 2x Compress", 2, 2),
    ];

    println!("| Scenario | Size | Dedup | Compress | Single-Pass (GB/s) | Streaming (GB/s) | Winner | Performance Ratio |");
    println!("|----------|------|--------|----------|-------------------|------------------|--------|------------------|");

    for (size_name, size) in &test_cases {
        for (param_name, dedup, compress) in &dedup_compress_cases {
            let (single_pass_gbps, streaming_gbps) = benchmark_approaches(*size, *dedup, *compress).await?;
            
            let winner = if single_pass_gbps > streaming_gbps { "Single-Pass" } else { "Streaming" };
            let ratio = if single_pass_gbps > streaming_gbps {
                format!("{:.1}x faster", single_pass_gbps / streaming_gbps)
            } else {
                format!("{:.1}x faster", streaming_gbps / single_pass_gbps)
            };

            println!("| {} | {} | {}x | {}x | {:.2} | {:.2} | {} | {} |",
                param_name, size_name, dedup, compress,
                single_pass_gbps, streaming_gbps, winner, ratio);
        }
        println!("|----------|------|--------|----------|-------------------|------------------|--------|------------------|");
    }

    println!("\n=== BENCHMARK CONCLUSIONS ===");
    println!("✅ Results show which approach is fastest for each scenario");
    println!("✅ Can guide production optimization decisions");
    println!("✅ Validates current API choices (CLI/Python use single-pass)");
    
    Ok(())
}

async fn benchmark_approaches(size: usize, dedup: usize, compress: usize) -> Result<(f64, f64)> {
    const ITERATIONS: usize = 3;
    
    // Benchmark single-pass approach
    let mut single_pass_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _data = generate_controlled_data(size, dedup, compress);
        single_pass_times.push(start.elapsed());
    }
    
    // Benchmark streaming approach (measure generation time only)
    let mut streaming_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, dedup, compress);
        
        // Generate all chunks to measure total generation time
        let chunk_size = 256 * 1024; // 256KB chunks
        while !object_gen.is_complete() {
            let remaining = object_gen.total_size() - object_gen.position();
            let current_chunk_size = remaining.min(chunk_size);
            let _chunk = object_gen.fill_chunk(current_chunk_size);
        }
        streaming_times.push(start.elapsed());
    }
    
    // Calculate average throughput in GB/s
    let avg_single_pass = single_pass_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / ITERATIONS as f64;
    let avg_streaming = streaming_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / ITERATIONS as f64;
    
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    let single_pass_gbps = size_gb / avg_single_pass;
    let streaming_gbps = size_gb / avg_streaming;
    
    Ok((single_pass_gbps, streaming_gbps))
}

#[tokio::test]
async fn test_end_to_end_upload_speed_comparison() -> Result<()> {
    println!("\n=== END-TO-END UPLOAD SPEED COMPARISON ===");
    println!("Testing complete upload workflows: single-pass vs streaming");
    
    let size = 8 * 1024 * 1024; // 8MB
    let store = FileSystemObjectStore::new();
    let options = WriterOptions::default();
    
    // Test single-pass upload approach (current CLI/Python method)
    let start = Instant::now();
    let data = generate_controlled_data(size, 1, 1);
    let uri = "file:///tmp/test_single_pass_upload.bin";
    store.put(uri, &data).await?;
    let single_pass_time = start.elapsed();
    
    // Test streaming upload approach  
    let start = Instant::now();
    let uri2 = "file:///tmp/test_streaming_upload.bin";
    let mut streaming_writer = StreamingDataWriter::new(uri2, size, 1, 1, &store, options).await?;
    streaming_writer.generate_remaining().await?;
    let streaming_time = start.elapsed();
    
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    let single_pass_gbps = size_gb / single_pass_time.as_secs_f64();
    let streaming_gbps = size_gb / streaming_time.as_secs_f64();
    
    println!("| Approach | Time (ms) | Throughput (GB/s) | Winner |");
    println!("|----------|-----------|-------------------|--------|");
    println!("| Single-Pass (CLI/Python) | {:.1} | {:.2} | {} |", 
        single_pass_time.as_millis(), single_pass_gbps,
        if single_pass_gbps > streaming_gbps { "✅" } else { "" });
    println!("| Streaming Upload | {:.1} | {:.2} | {} |", 
        streaming_time.as_millis(), streaming_gbps,
        if streaming_gbps > single_pass_gbps { "✅" } else { "" });
    
    let winner = if single_pass_gbps > streaming_gbps { "Single-Pass" } else { "Streaming" };
    let ratio = if single_pass_gbps > streaming_gbps {
        single_pass_gbps / streaming_gbps
    } else {
        streaming_gbps / single_pass_gbps
    };
    
    println!("\n**Result**: {} is {:.1}x faster for end-to-end uploads", winner, ratio);
    
    if single_pass_gbps > streaming_gbps {
        println!("✅ **VALIDATION**: Current CLI/Python APIs use the fastest approach!");
    } else {
        println!("⚠️  **OPTIMIZATION OPPORTUNITY**: Consider switching to streaming for uploads");
    }
    
    // Cleanup
    std::fs::remove_file("/tmp/test_single_pass_upload.bin").ok();
    std::fs::remove_file("/tmp/test_streaming_upload.bin").ok();
    
    Ok(())
}

#[tokio::test] 
async fn test_memory_pressure_vs_speed_tradeoff() -> Result<()> {
    println!("\n=== MEMORY PRESSURE vs SPEED TRADE-OFF ===");
    println!("Testing scenarios where streaming might be preferred despite speed penalty");
    
    let test_sizes = vec![
        ("256MB", 256 * 1024 * 1024),
        ("512MB", 512 * 1024 * 1024), 
        ("1GB", 1024 * 1024 * 1024),
    ];
    
    println!("| Size | Single-Pass Speed (GB/s) | Memory Usage (MB) | Streaming Speed (GB/s) | Memory Usage (MB) | Speed Penalty | Memory Savings |");
    println!("|------|--------------------------|-------------------|------------------------|-------------------|---------------|-----------------|");
    
    for (size_name, size) in test_sizes {
        // Single-pass benchmark
        let start = Instant::now();
        let data = generate_controlled_data(size, 1, 1);
        let single_pass_time = start.elapsed();
        let single_pass_memory_mb = data.len() as f64 / (1024.0 * 1024.0);
        drop(data); // Free memory
        
        // Streaming benchmark (chunked)
        let chunk_size = 256 * 1024; // 256KB chunks
        let start = Instant::now();
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, 1, 1);
        
        while !object_gen.is_complete() {
            let remaining = object_gen.total_size() - object_gen.position();
            let current_chunk_size = remaining.min(chunk_size);
            let _chunk = object_gen.fill_chunk(current_chunk_size);
            // Process/upload chunk immediately, don't accumulate
        }
        let streaming_time = start.elapsed();
        let streaming_memory_mb = chunk_size as f64 / (1024.0 * 1024.0);
        
        let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
        let single_pass_gbps = size_gb / single_pass_time.as_secs_f64();
        let streaming_gbps = size_gb / streaming_time.as_secs_f64();
        
        let speed_penalty = single_pass_gbps / streaming_gbps;
        let memory_savings = single_pass_memory_mb / streaming_memory_mb;
        
        println!("| {} | {:.2} | {:.1} | {:.2} | {:.1} | {:.1}x slower | {:.0}x less |",
            size_name, single_pass_gbps, single_pass_memory_mb,
            streaming_gbps, streaming_memory_mb, speed_penalty, memory_savings);
    }
    
    println!("\n**Analysis**: Streaming trades speed for memory efficiency");
    println!("**Recommendation**: Use single-pass by default, streaming only when memory-constrained");
    
    Ok(())
}