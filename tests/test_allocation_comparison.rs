// tests/test_allocation_comparison.rs
//
// Allocation overhead comparison test for Phase 1 buffer pool optimization
//
// This test attempts to measure the allocation overhead difference between:
// 1. DEFAULT path: No buffer pool (allocates fresh Vec on each range read)
// 2. OPTIMIZED path: Buffer pool enabled (reuses pooled buffers)
//
// Without specialized profiling tools, we can't measure exact allocations,
// but we CAN measure the observable effects:
// - Execution time (allocations cause overhead)
// - Memory stats from /proc/self/status (Linux only)
//
// For precise allocation counting, use: perf stat -e page-faults,syscalls:sys_enter_mmap

use anyhow::Result;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::ObjectStore;
use std::time::Instant;
use tempfile::TempDir;

/// Helper function to perform N range reads and measure time
async fn benchmark_range_reads(
    store: &ConfigurableFileSystemObjectStore,
    uri: &str,
    iterations: usize,
) -> Result<std::time::Duration> {
    let start = Instant::now();
    
    for i in 0..iterations {
        let offset = (i * 8192) % (16 * 1024 * 1024 - 8192);
        let _data = store.get_range(uri, offset as u64, Some(8192)).await?;
    }
    
    Ok(start.elapsed())
}

#[tokio::test]
async fn test_allocation_overhead_comparison() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create 16MB test file (large enough to avoid caching effects)
    let test_data: Vec<u8> = (0..16 * 1024 * 1024).map(|i| (i % 256) as u8).collect();
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Allocation Overhead Comparison Test - Order 1 (Old→New)");
    println!("═══════════════════════════════════════════════════════════");
    
    // Test 1: WITHOUT buffer pool (baseline)
    println!("\n📊 Test 1: WITHOUT Buffer Pool (Baseline)");
    println!("   Config: FileSystemConfig::default() - no pool");
    
    let config_no_pool = FileSystemConfig::default();
    let store_no_pool = ConfigurableFileSystemObjectStore::new(config_no_pool);
    let uri_no_pool = format!("file://{}/test_no_pool.dat", base_path.to_str().unwrap());
    
    store_no_pool.put(&uri_no_pool, &test_data).await?;
    
    // Warmup
    let _ = benchmark_range_reads(&store_no_pool, &uri_no_pool, 10).await?;
    
    // Actual benchmark
    let time_no_pool_1 = benchmark_range_reads(&store_no_pool, &uri_no_pool, 1000).await?;
    println!("   Time for 1000 range reads: {:?}", time_no_pool_1);
    println!("   Average per read: {:.2}µs", time_no_pool_1.as_micros() as f64 / 1000.0);
    
    store_no_pool.delete(&uri_no_pool).await?;
    drop(store_no_pool);
    
    // Test 2: WITH buffer pool (optimized)
    println!("\n📊 Test 2: WITH Buffer Pool (Optimized)");
    println!("   Config: FileSystemConfig::direct_io() - pool of 32 × 64MB buffers");
    
    let config_with_pool = FileSystemConfig::direct_io();
    let store_with_pool = ConfigurableFileSystemObjectStore::new(config_with_pool);
    let uri_with_pool = format!("file://{}/test_with_pool.dat", base_path.to_str().unwrap());
    
    store_with_pool.put(&uri_with_pool, &test_data).await?;
    
    // Warmup
    let _ = benchmark_range_reads(&store_with_pool, &uri_with_pool, 10).await?;
    
    // Actual benchmark
    let time_with_pool_1 = benchmark_range_reads(&store_with_pool, &uri_with_pool, 1000).await?;
    println!("   Time for 1000 range reads: {:?}", time_with_pool_1);
    println!("   Average per read: {:.2}µs", time_with_pool_1.as_micros() as f64 / 1000.0);
    
    store_with_pool.delete(&uri_with_pool).await?;
    drop(store_with_pool);
    
    // Now run in REVERSE order: New→Old
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Allocation Overhead Comparison Test - Order 2 (New→Old)");
    println!("═══════════════════════════════════════════════════════════");
    
    // Test 3: WITH buffer pool first
    println!("\n📊 Test 3: WITH Buffer Pool (Optimized) - Run First");
    
    let config_with_pool_2 = FileSystemConfig::direct_io();
    let store_with_pool_2 = ConfigurableFileSystemObjectStore::new(config_with_pool_2);
    let uri_with_pool_2 = format!("file://{}/test_with_pool_2.dat", base_path.to_str().unwrap());
    
    store_with_pool_2.put(&uri_with_pool_2, &test_data).await?;
    let _ = benchmark_range_reads(&store_with_pool_2, &uri_with_pool_2, 10).await?;
    let time_with_pool_2 = benchmark_range_reads(&store_with_pool_2, &uri_with_pool_2, 1000).await?;
    println!("   Time for 1000 range reads: {:?}", time_with_pool_2);
    println!("   Average per read: {:.2}µs", time_with_pool_2.as_micros() as f64 / 1000.0);
    
    store_with_pool_2.delete(&uri_with_pool_2).await?;
    drop(store_with_pool_2);
    
    // Test 4: WITHOUT buffer pool second
    println!("\n📊 Test 4: WITHOUT Buffer Pool (Baseline) - Run Second");
    
    let config_no_pool_2 = FileSystemConfig::default();
    let store_no_pool_2 = ConfigurableFileSystemObjectStore::new(config_no_pool_2);
    let uri_no_pool_2 = format!("file://{}/test_no_pool_2.dat", base_path.to_str().unwrap());
    
    store_no_pool_2.put(&uri_no_pool_2, &test_data).await?;
    let _ = benchmark_range_reads(&store_no_pool_2, &uri_no_pool_2, 10).await?;
    let time_no_pool_2 = benchmark_range_reads(&store_no_pool_2, &uri_no_pool_2, 1000).await?;
    println!("   Time for 1000 range reads: {:?}", time_no_pool_2);
    println!("   Average per read: {:.2}µs", time_no_pool_2.as_micros() as f64 / 1000.0);
    
    store_no_pool_2.delete(&uri_no_pool_2).await?;
    
    // Compare results across both orderings
    println!("\n═══════════════════════════════════════════════════════════");
    println!("📈 Performance Summary Across Test Orders:");
    println!("═══════════════════════════════════════════════════════════");
    
    println!("\n📊 Order 1 (Old→New):");
    println!("   Baseline (no pool):  {:?}", time_no_pool_1);
    println!("   Optimized (pool):    {:?}", time_with_pool_1);
    
    let diff_1 = if time_with_pool_1 < time_no_pool_1 {
        let improvement = (time_no_pool_1.as_secs_f64() - time_with_pool_1.as_secs_f64()) 
                         / time_no_pool_1.as_secs_f64() * 100.0;
        println!("   Result: {:.1}% faster with pool", improvement);
        improvement
    } else {
        let regression = (time_with_pool_1.as_secs_f64() - time_no_pool_1.as_secs_f64()) 
                        / time_no_pool_1.as_secs_f64() * 100.0;
        println!("   Result: {:.1}% slower with pool", regression);
        -regression
    };
    
    println!("\n📊 Order 2 (New→Old):");
    println!("   Optimized (pool):    {:?}", time_with_pool_2);
    println!("   Baseline (no pool):  {:?}", time_no_pool_2);
    
    let diff_2 = if time_with_pool_2 < time_no_pool_2 {
        let improvement = (time_no_pool_2.as_secs_f64() - time_with_pool_2.as_secs_f64()) 
                         / time_no_pool_2.as_secs_f64() * 100.0;
        println!("   Result: {:.1}% faster with pool", improvement);
        improvement
    } else {
        let regression = (time_with_pool_2.as_secs_f64() - time_no_pool_2.as_secs_f64()) 
                        / time_no_pool_2.as_secs_f64() * 100.0;
        println!("   Result: {:.1}% slower with pool", regression);
        -regression
    };
    
    println!("\n📊 Cross-Order Analysis:");
    println!("   Test order impact: {:.1}% difference between orderings", (diff_1 - diff_2).abs());
    
    if (diff_1 - diff_2).abs() > 10.0 {
        println!("   ⚠️  WARNING: >10% variance suggests caching/timing effects");
    } else {
        println!("   ✅ Test ordering has <10% impact (acceptable variance)");
    }
    
    println!("\n💡 Notes:");
    println!("   • Microbenchmarks on small files may not show full benefit");
    println!("   • Buffer pool shines with large files + high concurrency");
    println!("   • For accurate profiling:");
    println!("     - Use sai3-bench with --file-size 1GB --range-size 256KB");
    println!("     - Profile with: perf stat -e page-faults,syscalls:sys_enter_mmap");
    println!("     - Run with DirectIO + RangeEngine enabled");
    
    println!("\n═══════════════════════════════════════════════════════════");
    
    Ok(())
}

/// Test memory stats difference (Linux only)
#[cfg(target_os = "linux")]
#[tokio::test]
async fn test_memory_allocation_stats() -> Result<()> {
    use std::fs;
    
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Helper to read VmRSS from /proc/self/status
    fn get_vm_rss_kb() -> Result<usize> {
        let status = fs::read_to_string("/proc/self/status")?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return Ok(parts[1].parse()?);
                }
            }
        }
        anyhow::bail!("VmRSS not found in /proc/self/status")
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Memory Allocation Stats Test (Linux)");
    println!("═══════════════════════════════════════════════════════════");
    
    // Create test file
    let test_data: Vec<u8> = (0..64 * 1024 * 1024).map(|i| (i % 256) as u8).collect();
    
    // Test WITHOUT buffer pool
    println!("\n📊 Memory usage WITHOUT buffer pool:");
    
    let config_no_pool = FileSystemConfig::default();
    let store_no_pool = ConfigurableFileSystemObjectStore::new(config_no_pool);
    let uri_no_pool = format!("file://{}/test_mem_no_pool.dat", base_path.to_str().unwrap());
    store_no_pool.put(&uri_no_pool, &test_data).await?;
    
    let rss_before = get_vm_rss_kb()?;
    println!("   RSS before reads: {} KB", rss_before);
    
    // Perform many range reads
    for i in 0..100 {
        let offset = (i * 1024 * 1024) % (64 * 1024 * 1024 - 1024 * 1024);
        let _data = store_no_pool.get_range(&uri_no_pool, offset, Some(1024 * 1024)).await?;
    }
    
    let rss_after = get_vm_rss_kb()?;
    println!("   RSS after 100 reads: {} KB", rss_after);
    println!("   RSS increase: {} KB", rss_after.saturating_sub(rss_before));
    
    store_no_pool.delete(&uri_no_pool).await?;
    drop(store_no_pool);
    
    // Force garbage collection
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    
    // Test WITH buffer pool
    println!("\n📊 Memory usage WITH buffer pool:");
    
    let config_with_pool = FileSystemConfig::direct_io();
    let store_with_pool = ConfigurableFileSystemObjectStore::new(config_with_pool);
    let uri_with_pool = format!("file://{}/test_mem_with_pool.dat", base_path.to_str().unwrap());
    store_with_pool.put(&uri_with_pool, &test_data).await?;
    
    let rss_before_pool = get_vm_rss_kb()?;
    println!("   RSS before reads: {} KB", rss_before_pool);
    
    // Perform same number of range reads
    for i in 0..100 {
        let offset = (i * 1024 * 1024) % (64 * 1024 * 1024 - 1024 * 1024);
        let _data = store_with_pool.get_range(&uri_with_pool, offset, Some(1024 * 1024)).await?;
    }
    
    let rss_after_pool = get_vm_rss_kb()?;
    println!("   RSS after 100 reads: {} KB", rss_after_pool);
    println!("   RSS increase: {} KB", rss_after_pool.saturating_sub(rss_before_pool));
    
    println!("\n💡 Note: RSS measurements are coarse and affected by:");
    println!("   - OS page cache behavior");
    println!("   - Background allocator behavior");
    println!("   - Timing of garbage collection");
    println!("   Use 'perf stat' for precise allocation counting");
    
    println!("═══════════════════════════════════════════════════════════");
    
    store_with_pool.delete(&uri_with_pool).await?;
    Ok(())
}
