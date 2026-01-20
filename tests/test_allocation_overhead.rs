// Test to measure Vec allocation overhead vs actual generation

use std::time::Instant;

#[test]
#[ignore]
fn test_allocation_overhead() {
    let size = 10 * 1024 * 1024 * 1024; // 10 GB
    
    println!("\n=== Allocation Overhead Test ===\n");
    
    // Test 1: Just allocate
    println!("Test 1: vec![0u8; {}] allocation...", size);
    let start = Instant::now();
    let vec = vec![0u8; size];
    let alloc_time = start.elapsed();
    println!("  Allocation time: {:.3} seconds", alloc_time.as_secs_f64());
    println!("  Throughput: {:.2} GB/s", (size as f64 / (1024.0 * 1024.0 * 1024.0)) / alloc_time.as_secs_f64());
    drop(vec);
    
    // Test 2: Allocate with capacity (no initialization)
    println!("\nTest 2: Vec::with_capacity({}) + unsafe set_len...", size);
    let start = Instant::now();
    let mut vec: Vec<u8> = Vec::with_capacity(size);
    unsafe { vec.set_len(size); }
    let capacity_time = start.elapsed();
    println!("  Allocation time: {:.3} seconds", capacity_time.as_secs_f64());
    println!("  Throughput: {:.2} GB/s", (size as f64 / (1024.0 * 1024.0 * 1024.0)) / capacity_time.as_secs_f64());
    drop(vec);
    
    println!("\nConclusion:");
    println!("  vec![0u8] zeros memory: {:.3}s ({:.2} GB/s)", 
             alloc_time.as_secs_f64(),
             (size as f64 / (1024.0 * 1024.0 * 1024.0)) / alloc_time.as_secs_f64());
    println!("  Vec::with_capacity skips zeroing: {:.3}s ({:.2} GB/s)", 
             capacity_time.as_secs_f64(),
             (size as f64 / (1024.0 * 1024.0 * 1024.0)) / capacity_time.as_secs_f64());
}
