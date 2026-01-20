// tests/test_cpu_utilization.rs
//
// Standalone test to demonstrate CPU utilization and NUMA detection in s3dlio

use s3dlio::data_gen_alt::{GeneratorConfig, total_cpus, default_data_gen_threads};
use std::time::Instant;

#[test]
#[ignore] // Run with: cargo test --release test_cpu_utilization -- --ignored --nocapture
fn test_cpu_utilization() {
    println!("\n=== s3dlio CPU Utilization Test ===\n");
    
    // System information
    let total_cores = total_cpus();
    let default_threads = default_data_gen_threads();
    
    println!("System Information:");
    println!("  Total CPUs: {}", total_cores);
    println!("  Default threads: {} ({}%)", default_threads, (default_threads * 100) / total_cores);
    
    #[cfg(feature = "numa")]
    {
        println!("\n  NUMA feature: ENABLED");
        if let Ok(topology) = s3dlio::numa::NumaTopology::detect() {
            println!("  NUMA nodes: {}", topology.num_nodes);
            println!("  System type: {}", if topology.num_nodes > 1 { "NUMA" } else { "UMA" });
            
            for node in &topology.nodes {
                println!("    Node {}: {} CPUs", node.node_id, node.cpus.len());
            }
        } else {
            println!("  NUMA topology detection failed");
        }
    }
    
    #[cfg(not(feature = "numa"))]
    {
        println!("\n  NUMA feature: DISABLED (compile with --features numa to enable)");
    }
    
    // WARMUP: Generate 1 GB to enable caching (code, data, CPU)
    println!("\nWarmup Run:");
    println!("  Generating 1 GB to warm up caches...");
    let warmup_config = GeneratorConfig {
        size: 1 * 1024 * 1024 * 1024, // 1 GB
        dedup_factor: 1,
        compress_factor: 1,
        ..Default::default()
    };
    let warmup_start = std::time::Instant::now();
    let warmup_data = s3dlio::data_gen_alt::generate_data(warmup_config);
    let warmup_elapsed = warmup_start.elapsed();
    println!("  Warmup: {:.2} GB in {:.3}s = {:.2} GB/s", 
             warmup_data.len() as f64 / (1024.0 * 1024.0 * 1024.0),
             warmup_elapsed.as_secs_f64(),
             (warmup_data.len() as f64 / (1024.0 * 1024.0 * 1024.0)) / warmup_elapsed.as_secs_f64());
    drop(warmup_data);

    // Performance test - generate 100 GB with all CPUs using streaming API (like dgen-rs)
    println!("\nPerformance Test:");
    let total_size: usize = 100 * 1024 * 1024 * 1024; // 100 GB total
    let chunk_size: usize = 64 * 1024 * 1024; // 64 MB chunks (EXACTLY like dgen-rs)
    println!("  Generating {} GB of data in {} MB chunks (streaming API like dgen-rs)...", 
             total_size / (1024 * 1024 * 1024), 
             chunk_size / (1024 * 1024));
    
    let config = GeneratorConfig {
        size: total_size,
        dedup_factor: 1,
        compress_factor: 1,
        ..Default::default()
    };
    
    // Create ONE generator (creates thread pool once)
    let mut generator = s3dlio::data_gen_alt::DataGenerator::new(config);
    let mut buffer = vec![0u8; chunk_size];
    
    let start = Instant::now();
    let mut total_bytes = 0;
    let mut chunk_num = 0;
    
    // Stream through all data (reuses thread pool!)
    while !generator.is_complete() {
        chunk_num += 1;
        if chunk_num % 100 == 0 {
            print!("  Chunk {}... ", chunk_num);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        
        let nbytes = generator.fill_chunk(&mut buffer);
        if nbytes == 0 {
            break;
        }
        total_bytes += nbytes;
        if chunk_num % 100 == 0 {
            println!("✓ ({:.2} GB so far)", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        }
    }
    
    let elapsed = start.elapsed();
    
    let throughput_gbps = (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed.as_secs_f64();
    
    println!("\n  Generated: {:.2} GB", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  Time: {:.2} seconds", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} GB/s", throughput_gbps);
    println!("  Per-core throughput: {:.2} GB/s", throughput_gbps / 6.0); // 6 physical cores
    
    // Verify all CPUs were utilized (6 physical cores × 8 GB/s = 48 GB/s expected)
    let expected_min_total_throughput = 6.0 * 8.0; // 48 GB/s for 6 physical cores
    println!("\n  Expected minimum: {:.2} GB/s (6 physical cores × 8 GB/s)", expected_min_total_throughput);
    
    assert!(throughput_gbps >= expected_min_total_throughput * 0.8, 
            "Throughput {:.2} GB/s is below 80% of expected {:.2} GB/s - CPUs may not be fully utilized",
            throughput_gbps, expected_min_total_throughput);
    
    println!("\n✓ CPU utilization test PASSED");
    println!("  All {} CPUs appear to be working optimally\n", total_cores);
}
