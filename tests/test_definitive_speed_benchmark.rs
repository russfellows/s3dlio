use std::time::Instant;
use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use anyhow::Result;

/// DEFINITIVE speed benchmark with high precision timing
/// This resolves conflicts between previous benchmarks by using consistent methodology

#[tokio::test]
async fn test_definitive_speed_benchmark() -> Result<()> {
    println!("\n=== DEFINITIVE SPEED BENCHMARK ===");
    println!("Resolving conflicts between previous benchmarks");
    println!("Using high-precision timing and consistent methodology\n");

    let test_cases = vec![
        ("1MB", 1024 * 1024),
        ("8MB", 8 * 1024 * 1024),
        ("32MB", 32 * 1024 * 1024),
        ("128MB", 128 * 1024 * 1024),
    ];

    println!("| Size | Single-Pass (GB/s) | Streaming-256KB (GB/s) | Winner | Ratio |");
    println!("|------|-------------------|-------------------------|--------|-------|");

    for (size_name, size) in &test_cases {
        let (single_pass_gbps, streaming_gbps) = precise_benchmark(*size).await?;
        
        let winner = if single_pass_gbps > streaming_gbps { "Single-Pass" } else { "Streaming" };
        let ratio = if single_pass_gbps > streaming_gbps {
            format!("{:.1}x", single_pass_gbps / streaming_gbps)
        } else {
            format!("{:.1}x", streaming_gbps / single_pass_gbps)
        };

        println!("| {} | {:.2} | {:.2} | {} | {} |",
            size_name, single_pass_gbps, streaming_gbps, winner, ratio);
    }

    println!("\n=== DEFINITIVE CONCLUSIONS ===");
    println!("This benchmark uses consistent methodology and high-precision timing");
    println!("Results should be used for production optimization decisions");
    
    Ok(())
}

async fn precise_benchmark(size: usize) -> Result<(f64, f64)> {
    const ITERATIONS: usize = 100; // More iterations for precision
    const WARMUP_ITERATIONS: usize = 10; // Warmup to stabilize performance
    
    // Warmup phase
    for _ in 0..WARMUP_ITERATIONS {
        let _data = generate_controlled_data(size, 1, 1);
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, 1, 1);
        let _remaining = object_gen.fill_remaining();
    }
    
    // Benchmark single-pass approach
    let mut single_pass_durations = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _data = generate_controlled_data(size, 1, 1);
        single_pass_durations.push(start.elapsed());
    }
    
    // Benchmark streaming approach with 256KB chunks (same as original test)
    let mut streaming_durations = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, 1, 1);
        
        let chunk_size = 256 * 1024; // 256KB chunks to match original test
        while !object_gen.is_complete() {
            let remaining = object_gen.total_size() - object_gen.position();
            let current_chunk_size = remaining.min(chunk_size);
            let _chunk = object_gen.fill_chunk(current_chunk_size);
        }
        streaming_durations.push(start.elapsed());
    }
    
    // Calculate statistics - use median to avoid outliers
    single_pass_durations.sort();
    streaming_durations.sort();
    
    let single_pass_median = single_pass_durations[ITERATIONS / 2];
    let streaming_median = streaming_durations[ITERATIONS / 2];
    
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    let single_pass_gbps = size_gb / single_pass_median.as_secs_f64();
    let streaming_gbps = size_gb / streaming_median.as_secs_f64();
    
    Ok((single_pass_gbps, streaming_gbps))
}

#[tokio::test]
async fn test_chunk_size_optimization() -> Result<()> {
    println!("\n=== CHUNK SIZE OPTIMIZATION ===");
    println!("Testing different chunk sizes to find optimal streaming performance");
    
    let size = 8 * 1024 * 1024; // 8MB test case
    let chunk_sizes = vec![
        ("64KB", 64 * 1024),
        ("256KB", 256 * 1024),
        ("1MB", 1024 * 1024), 
        ("4MB", 4 * 1024 * 1024),
    ];
    
    // Single-pass baseline
    let single_pass_baseline = {
        let mut times = Vec::new();
        for _ in 0..50 {
            let start = Instant::now();
            let _data = generate_controlled_data(size, 1, 1);
            times.push(start.elapsed());
        }
        times.sort();
        let median = times[25];
        let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
        size_gb / median.as_secs_f64()
    };
    
    println!("Single-Pass Baseline: {:.2} GB/s", single_pass_baseline);
    println!("\n| Chunk Size | Streaming (GB/s) | vs Single-Pass | Recommendation |");
    println!("|------------|------------------|-----------------|----------------|");
    
    let mut best_streaming = 0.0f64;
    let mut best_chunk = "";
    
    for (chunk_name, chunk_size) in &chunk_sizes {
        let streaming_performance = {
            let mut times = Vec::new();
            for _ in 0..50 {
                let start = Instant::now();
                let generator = DataGenerator::new();
                let mut object_gen = generator.begin_object(size, 1, 1);
                
                while !object_gen.is_complete() {
                    let remaining = object_gen.total_size() - object_gen.position();
                    let current_chunk_size = remaining.min(*chunk_size);
                    let _chunk = object_gen.fill_chunk(current_chunk_size);
                }
                times.push(start.elapsed());
            }
            times.sort();
            let median = times[25];
            let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
            size_gb / median.as_secs_f64()
        };
        
        let vs_single_pass = if streaming_performance > single_pass_baseline {
            format!("{:.1}x faster", streaming_performance / single_pass_baseline)
        } else {
            format!("{:.1}x slower", single_pass_baseline / streaming_performance)
        };
        
        let recommendation = if streaming_performance > single_pass_baseline {
            "✅ Use streaming"
        } else if streaming_performance / single_pass_baseline > 0.8 {
            "⚠️  Close, memory benefit"
        } else {
            "❌ Use single-pass"
        };
        
        if streaming_performance > best_streaming {
            best_streaming = streaming_performance;
            best_chunk = chunk_name;
        }
        
        println!("| {} | {:.2} | {} | {} |",
            chunk_name, streaming_performance, vs_single_pass, recommendation);
    }
    
    println!("\n**OPTIMIZATION RESULT**:");
    println!("Best streaming configuration: {} chunks ({:.2} GB/s)", best_chunk, best_streaming);
    
    if best_streaming > single_pass_baseline {
        println!("✅ **RECOMMENDATION**: Use streaming with {} chunks for best performance", best_chunk);
    } else {
        println!("✅ **RECOMMENDATION**: Use single-pass generation for best performance");
        println!("   Streaming provides {:.0}x memory reduction at {:.1}x speed cost", 
            (size as f64) / (256.0 * 1024.0), single_pass_baseline / best_streaming);
    }
    
    Ok(())
}