// tests/test_optimal_chunking.rs
//
// Test optimal chunk size selection for data generation

use s3dlio::data_gen_alt::{generate_controlled_data_alt, optimal_chunk_size};
use std::time::Instant;

#[test]
fn test_optimal_chunk_sizes() {
    // Test that optimal_chunk_size returns expected values
    assert_eq!(optimal_chunk_size(10 * 1024 * 1024), 10 * 1024 * 1024); // 10 MB → returns itself
    assert_eq!(optimal_chunk_size(16 * 1024 * 1024), 16 * 1024 * 1024); // 16 MB → 16 MB
    assert_eq!(optimal_chunk_size(20 * 1024 * 1024), 16 * 1024 * 1024); // 20 MB → 16 MB
    assert_eq!(optimal_chunk_size(32 * 1024 * 1024), 32 * 1024 * 1024); // 32 MB → 32 MB
    assert_eq!(optimal_chunk_size(50 * 1024 * 1024), 32 * 1024 * 1024); // 50 MB → 32 MB
    assert_eq!(optimal_chunk_size(64 * 1024 * 1024), 64 * 1024 * 1024); // 64 MB → 64 MB
    assert_eq!(optimal_chunk_size(100 * 1024 * 1024), 64 * 1024 * 1024); // 100 MB → 64 MB
    assert_eq!(optimal_chunk_size(1024 * 1024 * 1024), 64 * 1024 * 1024); // 1 GB → 64 MB
}

#[test]
#[ignore] // Run with: cargo test --release test_performance_various_sizes -- --ignored --nocapture
fn test_performance_various_sizes() {
    println!("\n=== Testing Performance Across Various Sizes ===\n");
    
    let test_sizes = vec![
        ("10 MB", 10 * 1024 * 1024),
        ("16 MB", 16 * 1024 * 1024),
        ("32 MB", 32 * 1024 * 1024),
        ("64 MB", 64 * 1024 * 1024),
        ("128 MB", 128 * 1024 * 1024),
        ("256 MB", 256 * 1024 * 1024),
        ("512 MB", 512 * 1024 * 1024),
        ("1 GB", 1024 * 1024 * 1024),
    ];
    
    for (label, size) in test_sizes {
        let chunk_size = optimal_chunk_size(size);
        
        let start = Instant::now();
        let data = generate_controlled_data_alt(size, 1, 1, None);
        let elapsed = start.elapsed();
        
        let throughput = (size as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        
        println!(
            "{:>8}: {:.3}s | {:.2} GB/s | chunk_size={} MB | data_size={} bytes",
            label,
            elapsed.as_secs_f64(),
            throughput,
            chunk_size / (1024 * 1024),
            data.len()
        );
        
        assert_eq!(data.len(), size);
    }
    
    println!("\nNOTE: Throughput should increase significantly for sizes >= 64 MB");
}

#[test]
#[ignore] // Run with: cargo test --release test_small_vs_large -- --ignored --nocapture
fn test_small_vs_large() {
    println!("\n=== Comparing Small (single alloc) vs Large (streaming) ===\n");
    
    // Small size (< 16 MB) - should use single allocation
    let small_size = 10 * 1024 * 1024; // 10 MB
    let start = Instant::now();
    let small_data = generate_controlled_data_alt(small_size, 1, 1, None);
    let small_elapsed = start.elapsed();
    let small_throughput = (small_size as f64) / small_elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    
    println!("Small (10 MB, single alloc):");
    println!("  Time: {:.4}s", small_elapsed.as_secs_f64());
    println!("  Throughput: {:.2} GB/s", small_throughput);
    
    // Large size (>= 64 MB) - should use 64 MB chunks
    let large_size = 1024 * 1024 * 1024; // 1 GB
    let start = Instant::now();
    let large_data = generate_controlled_data_alt(large_size, 1, 1, None);
    let large_elapsed = start.elapsed();
    let large_throughput = (large_size as f64) / large_elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    
    println!("\nLarge (1 GB, 64 MB chunks):");
    println!("  Time: {:.4}s", large_elapsed.as_secs_f64());
    println!("  Throughput: {:.2} GB/s", large_throughput);
    
    println!("\nComparison:");
    println!("  Large is {:.2}x faster (per-byte)", large_throughput / small_throughput);
    
    assert_eq!(small_data.len(), small_size);
    assert_eq!(large_data.len(), large_size);
    
    // Large should be significantly faster (at least 2x per-byte throughput)
    assert!(
        large_throughput > small_throughput * 2.0,
        "Large data generation should be at least 2x faster per-byte"
    );
}
