use std::time::Instant;
use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use anyhow::Result;

/// Real-world performance benchmark for typical s3dlio usage patterns
/// Tests 1MB-100MB objects with common dedup/compression rates (1:1 and 2:1)
/// Memory constraint: Up to 1GB available, so memory efficiency below 100MB is not a concern

#[tokio::test]
async fn test_real_world_performance_benchmark() -> Result<()> {
    println!("\n=== REAL-WORLD PERFORMANCE BENCHMARK ===");
    println!("Testing typical s3dlio usage: 1MB-100MB objects with common dedup/compression rates");
    println!("Memory constraint: Up to 1GB available (memory not a concern for these sizes)\n");

    let test_cases = vec![
        ("1MB", 1024 * 1024),
        ("4MB", 4 * 1024 * 1024),
        ("8MB", 8 * 1024 * 1024),
        ("16MB", 16 * 1024 * 1024),
        ("32MB", 32 * 1024 * 1024),
        ("64MB", 64 * 1024 * 1024),
        ("100MB", 100 * 1024 * 1024),
    ];

    let data_configs = vec![
        ("No Dedup/Compress (1:1)", 1, 1),
        ("2:1 Deduplication", 2, 1),
        ("2:1 Compression", 1, 2),
        ("2:1 Dedup + 2:1 Compress", 2, 2),
    ];

    println!("| Size | Config | Single-Pass (GB/s) | Streaming-256KB (GB/s) | Winner | Advantage | Recommendation |");
    println!("|------|--------|-------------------|-------------------------|--------|-----------|----------------|");

    let mut results = Vec::new();

    for (size_name, size) in &test_cases {
        for (config_name, dedup, compress) in &data_configs {
            let (single_pass_gbps, streaming_gbps) = benchmark_real_world_scenario(*size, *dedup, *compress).await?;
            
            let (winner, advantage, recommendation) = if single_pass_gbps > streaming_gbps {
                let ratio = single_pass_gbps / streaming_gbps;
                ("Single-Pass", format!("{:.1}x", ratio), 
                 if ratio > 1.2 { "✅ Use Single-Pass" } else { "⚠️  Either OK" })
            } else {
                let ratio = streaming_gbps / single_pass_gbps;
                ("Streaming", format!("{:.1}x", ratio),
                 if ratio > 1.2 { "✅ Use Streaming" } else { "⚠️  Either OK" })
            };

            println!("| {} | {} | {:.2} | {:.2} | {} | {} | {} |",
                size_name, config_name, single_pass_gbps, streaming_gbps, winner, advantage, recommendation);
            
            results.push((*size, *dedup, *compress, single_pass_gbps, streaming_gbps, winner));
        }
        println!("|------|--------|-------------------|-------------------------|--------|-----------|----------------|");
    }

    analyze_results(&results);
    
    Ok(())
}

async fn benchmark_real_world_scenario(size: usize, dedup: usize, compress: usize) -> Result<(f64, f64)> {
    const ITERATIONS: usize = 50; // Sufficient for precision without being slow
    const WARMUP_ITERATIONS: usize = 5;
    
    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        let _data = generate_controlled_data(size, dedup, compress);
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, dedup, compress);
        let _remaining = object_gen.fill_remaining();
    }
    
    // Benchmark single-pass
    let mut single_pass_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _data = generate_controlled_data(size, dedup, compress);
        single_pass_times.push(start.elapsed());
    }
    
    // Benchmark streaming with optimal 256KB chunks
    let mut streaming_times = Vec::new();
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(size, dedup, compress);
        
        let chunk_size = 256 * 1024; // Optimal chunk size from previous tests
        while !object_gen.is_complete() {
            let remaining = object_gen.total_size() - object_gen.position();
            let current_chunk_size = remaining.min(chunk_size);
            let _chunk = object_gen.fill_chunk(current_chunk_size);
        }
        streaming_times.push(start.elapsed());
    }
    
    // Use median to avoid outliers
    single_pass_times.sort();
    streaming_times.sort();
    
    let single_pass_median = single_pass_times[ITERATIONS / 2];
    let streaming_median = streaming_times[ITERATIONS / 2];
    
    let size_gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    let single_pass_gbps = size_gb / single_pass_median.as_secs_f64();
    let streaming_gbps = size_gb / streaming_median.as_secs_f64();
    
    Ok((single_pass_gbps, streaming_gbps))
}

fn analyze_results(results: &[(usize, usize, usize, f64, f64, &str)]) {
    println!("\n=== ANALYSIS FOR PRODUCTION OPTIMIZATION ===");
    
    // Count wins by method
    let single_pass_wins = results.iter().filter(|(_, _, _, _, _, winner)| *winner == "Single-Pass").count();
    let streaming_wins = results.iter().filter(|(_, _, _, _, _, winner)| *winner == "Streaming").count();
    
    println!("**Overall Performance Winner Count**:");
    println!("- Single-Pass wins: {} scenarios", single_pass_wins);
    println!("- Streaming wins: {} scenarios", streaming_wins);
    
    // Analyze by size ranges
    println!("\n**Performance by Object Size**:");
    let small_objects: Vec<_> = results.iter().filter(|(size, _, _, _, _, _)| *size <= 8 * 1024 * 1024).collect();
    let medium_objects: Vec<_> = results.iter().filter(|(size, _, _, _, _, _)| *size > 8 * 1024 * 1024 && *size <= 32 * 1024 * 1024).collect();
    let large_objects: Vec<_> = results.iter().filter(|(size, _, _, _, _, _)| *size > 32 * 1024 * 1024).collect();
    
    analyze_size_range("Small (≤8MB)", &small_objects);
    analyze_size_range("Medium (8-32MB)", &medium_objects);
    analyze_size_range("Large (>32MB)", &large_objects);
    
    // Analyze by data configuration
    println!("\n**Performance by Data Configuration**:");
    let no_dedup_compress: Vec<_> = results.iter().filter(|(_, dedup, compress, _, _, _)| *dedup == 1 && *compress == 1).collect();
    let with_dedup_compress: Vec<_> = results.iter().filter(|(_, dedup, compress, _, _, _)| *dedup > 1 || *compress > 1).collect();
    
    analyze_config_range("No Dedup/Compress (1:1)", &no_dedup_compress);
    analyze_config_range("With Dedup/Compress (2:1)", &with_dedup_compress);
    
    // Final recommendation
    println!("\n=== 🎯 FINAL PRODUCTION RECOMMENDATIONS ===");
    
    if single_pass_wins > streaming_wins {
        println!("✅ **RECOMMENDATION**: Continue using single-pass generation (current CLI/Python behavior)");
        println!("   - Single-pass wins in {}/{} scenarios ({:.1}%)", 
            single_pass_wins, results.len(), (single_pass_wins as f64 / results.len() as f64) * 100.0);
        println!("   - Optimal for typical workloads in 1MB-100MB range");
        println!("   - Memory usage up to 100MB is acceptable (within 1GB constraint)");
    } else {
        println!("⚠️  **OPTIMIZATION OPPORTUNITY**: Consider switching to streaming generation");
        println!("   - Streaming wins in {}/{} scenarios ({:.1}%)", 
            streaming_wins, results.len(), (streaming_wins as f64 / results.len() as f64) * 100.0);
        println!("   - May provide performance benefits for typical workloads");
    }
    
    println!("\n**Current Status**: CLI and Python APIs use single-pass generation");
    println!("**Memory Constraint**: 1GB available - memory efficiency not critical below 100MB");
}

fn analyze_size_range(name: &str, objects: &[&(usize, usize, usize, f64, f64, &str)]) {
    if objects.is_empty() { return; }
    
    let single_pass_wins = objects.iter().filter(|(_, _, _, _, _, winner)| *winner == "Single-Pass").count();
    let streaming_wins = objects.iter().filter(|(_, _, _, _, _, winner)| *winner == "Streaming").count();
    
    let avg_single_pass: f64 = objects.iter().map(|(_, _, _, sp, _, _)| sp).sum::<f64>() / objects.len() as f64;
    let avg_streaming: f64 = objects.iter().map(|(_, _, _, _, st, _)| st).sum::<f64>() / objects.len() as f64;
    
    let winner = if single_pass_wins > streaming_wins { "Single-Pass" } else { "Streaming" };
    
    println!("- {}: {} wins ({} vs {}), Avg perf: {:.1} GB/s vs {:.1} GB/s → **Use {}**", 
        name, winner, single_pass_wins, streaming_wins, avg_single_pass, avg_streaming, winner);
}

fn analyze_config_range(name: &str, objects: &[&(usize, usize, usize, f64, f64, &str)]) {
    if objects.is_empty() { return; }
    
    let single_pass_wins = objects.iter().filter(|(_, _, _, _, _, winner)| *winner == "Single-Pass").count();
    let streaming_wins = objects.iter().filter(|(_, _, _, _, _, winner)| *winner == "Streaming").count();
    
    let avg_single_pass: f64 = objects.iter().map(|(_, _, _, sp, _, _)| sp).sum::<f64>() / objects.len() as f64;
    let avg_streaming: f64 = objects.iter().map(|(_, _, _, _, st, _)| st).sum::<f64>() / objects.len() as f64;
    
    let winner = if single_pass_wins > streaming_wins { "Single-Pass" } else { "Streaming" };
    
    println!("- {}: {} wins ({} vs {}), Avg perf: {:.1} GB/s vs {:.1} GB/s → **Use {}**", 
        name, winner, single_pass_wins, streaming_wins, avg_single_pass, avg_streaming, winner);
}

#[tokio::test]
async fn test_multiple_8mb_objects_with_dedup_compress() -> Result<()> {
    println!("\n=== MULTIPLE 8MB OBJECTS TEST ===");
    println!("Testing the specific scenario: 8MB objects with 2:1 dedup + 2:1 compress");
    println!("Simulating batch processing of multiple objects (common CLI usage pattern)\n");
    
    const NUM_OBJECTS: usize = 10;
    const OBJECT_SIZE: usize = 8 * 1024 * 1024; // 8MB
    const DEDUP_FACTOR: usize = 2;
    const COMPRESS_FACTOR: usize = 2;
    
    println!("Test Parameters:");
    println!("- Object count: {}", NUM_OBJECTS);
    println!("- Object size: 8MB each");  
    println!("- Total data: {}MB", (NUM_OBJECTS * OBJECT_SIZE) / (1024 * 1024));
    println!("- Deduplication: 2:1");
    println!("- Compression: 2:1\n");
    
    // Test single-pass approach (current CLI behavior)
    println!("Testing Single-Pass Generation (Current CLI/Python Behavior):");
    let start = Instant::now();
    let mut single_pass_objects = Vec::new();
    
    for i in 0..NUM_OBJECTS {
        let obj_start = Instant::now();
        let data = generate_controlled_data(OBJECT_SIZE, DEDUP_FACTOR, COMPRESS_FACTOR);
        let obj_time = obj_start.elapsed();
        single_pass_objects.push(data.len());
        
        if i < 3 {  // Show details for first 3 objects
            println!("  Object {}: {} bytes generated in {:.1}ms", 
                i + 1, data.len(), obj_time.as_millis());
        }
    }
    let single_pass_total_time = start.elapsed();
    
    // Test streaming approach
    println!("\nTesting Streaming Generation (Alternative Approach):");
    let start = Instant::now();
    let mut streaming_objects = Vec::new();
    
    for i in 0..NUM_OBJECTS {
        let obj_start = Instant::now();
        let generator = DataGenerator::new();
        let mut object_gen = generator.begin_object(OBJECT_SIZE, DEDUP_FACTOR, COMPRESS_FACTOR);
        
        let mut total_generated = 0;
        let chunk_size = 256 * 1024; // 256KB optimal chunk size
        
        while !object_gen.is_complete() {
            let remaining = object_gen.total_size() - object_gen.position();
            let current_chunk_size = remaining.min(chunk_size);
            if let Some(chunk) = object_gen.fill_chunk(current_chunk_size) {
                total_generated += chunk.len();
            }
        }
        
        let obj_time = obj_start.elapsed();
        streaming_objects.push(total_generated);
        
        if i < 3 {  // Show details for first 3 objects
            println!("  Object {}: {} bytes generated in {:.1}ms", 
                i + 1, total_generated, obj_time.as_millis());
        }
    }
    let streaming_total_time = start.elapsed();
    
    // Calculate and display results
    let total_bytes = single_pass_objects.iter().sum::<usize>();
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    
    let single_pass_gbps = total_gb / single_pass_total_time.as_secs_f64();
    let streaming_gbps = total_gb / streaming_total_time.as_secs_f64();
    
    println!("\n=== BATCH PROCESSING RESULTS ===");
    println!("| Method | Total Time | Throughput | Winner |");
    println!("|--------|------------|------------|--------|");
    println!("| Single-Pass | {:.1}ms | {:.2} GB/s | {} |", 
        single_pass_total_time.as_millis(), single_pass_gbps,
        if single_pass_gbps > streaming_gbps { "✅" } else { "" });
    println!("| Streaming | {:.1}ms | {:.2} GB/s | {} |", 
        streaming_total_time.as_millis(), streaming_gbps,
        if streaming_gbps > single_pass_gbps { "✅" } else { "" });
    
    let winner = if single_pass_gbps > streaming_gbps { "Single-Pass" } else { "Streaming" };
    let advantage = if single_pass_gbps > streaming_gbps {
        single_pass_gbps / streaming_gbps
    } else {
        streaming_gbps / single_pass_gbps
    };
    
    println!("\n**RESULT FOR 8MB OBJECTS WITH 2:1 DEDUP+COMPRESS**:");
    println!("🏆 {} is {:.1}x faster ({:.2} GB/s vs {:.2} GB/s)", 
        winner, advantage, 
        if single_pass_gbps > streaming_gbps { single_pass_gbps } else { streaming_gbps },
        if single_pass_gbps > streaming_gbps { streaming_gbps } else { single_pass_gbps });
    
    if winner == "Single-Pass" {
        println!("✅ **VALIDATION**: Current CLI/Python APIs use the optimal approach!");
    } else {
        println!("⚠️  **OPTIMIZATION**: Consider switching to streaming for better performance");
    }
    
    // Memory usage analysis
    let single_pass_peak_memory = OBJECT_SIZE; // One object at a time
    let streaming_peak_memory = 256 * 1024; // 256KB chunks
    
    println!("\n**Memory Usage Analysis**:");
    println!("- Single-Pass peak memory: {:.1}MB per object", single_pass_peak_memory as f64 / (1024.0 * 1024.0));
    println!("- Streaming peak memory: {:.1}MB per object", streaming_peak_memory as f64 / (1024.0 * 1024.0));
    println!("- Memory constraint: 1GB available");
    println!("- **Memory verdict**: Both approaches are well within limits for 8MB objects");
    
    Ok(())
}