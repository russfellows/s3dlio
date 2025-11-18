// Manual performance comparison with detailed metrics
//
// Run with: cargo test --release --test performance_comparison -- --nocapture --test-threads=1

use std::time::Instant;
use s3dlio::data_gen::{generate_controlled_data, DataGenerator};
use s3dlio::data_gen_alt::{generate_controlled_data_alt, ObjectGenAlt};

/// Get RSS memory in MB
#[cfg(target_os = "linux")]
fn get_memory_mb() -> f64 {
    use std::fs;
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0; // Convert to MB
                    }
                }
            }
        }
    }
    0.0
}

#[cfg(not(target_os = "linux"))]
fn get_memory_mb() -> f64 {
    0.0
}

/// Get CPU time in seconds (user + system)
#[cfg(target_os = "linux")]
fn get_cpu_time_secs() -> f64 {
    use std::fs;
    if let Ok(stat) = fs::read_to_string("/proc/self/stat") {
        let fields: Vec<&str> = stat.split_whitespace().collect();
        if fields.len() > 14 {
            let utime = fields[13].parse::<u64>().unwrap_or(0);
            let stime = fields[14].parse::<u64>().unwrap_or(0);
            let clock_ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
            let total_ticks = (utime + stime) as f64;
            return total_ticks / clock_ticks;
        }
    }
    0.0
}

#[cfg(not(target_os = "linux"))]
fn get_cpu_time_secs() -> f64 {
    0.0
}

/// Run timed test and return (throughput, cpu_efficiency)
fn benchmark<F>(name: &str, size: usize, iterations: usize, mut f: F) -> (f64, f64)
where
    F: FnMut() -> Vec<u8>,
{
    println!("\n=== {} ===", name);
    println!("Size: {} bytes ({} MB)", size, size / 1_048_576);
    println!("Iterations: {}", iterations);
    
    let mem_start = get_memory_mb();
    let cpu_start = get_cpu_time_secs();
    let wall_start = Instant::now();
    
    for i in 0..iterations {
        let data = f();
        assert_eq!(data.len(), size, "Size mismatch at iteration {}", i);
        drop(data); // Explicit drop
    }
    
    let wall_elapsed = wall_start.elapsed().as_secs_f64();
    let cpu_elapsed = get_cpu_time_secs() - cpu_start;
    let mem_end = get_memory_mb();
    
    let total_bytes = (size * iterations) as f64;
    let throughput = total_bytes / wall_elapsed / 1_048_576.0;
    let cpu_efficiency = if cpu_elapsed > 0.0 { wall_elapsed / cpu_elapsed } else { 0.0 };
    
    println!("Wall time: {:.3}s", wall_elapsed);
    println!("CPU time: {:.3}s (user+system)", cpu_elapsed);
    println!("CPU efficiency: {:.1}% (lower is better - less CPU per MB)", cpu_efficiency * 100.0);
    println!("Throughput: {:.2} MB/s", throughput);
    println!("Memory start: {:.2} MB, end: {:.2} MB, delta: {:.2} MB", 
             mem_start, mem_end, mem_end - mem_start);
    
    (throughput, cpu_efficiency)
}

/// Streaming benchmark - returns (throughput, cpu_efficiency)
fn benchmark_streaming<F>(name: &str, total_size: usize, chunk_size: usize, iterations: usize, mut f: F) -> (f64, f64)
where
    F: FnMut(),
{
    println!("\n=== {} ===", name);
    println!("Total size: {} bytes ({} MB), Chunk: {} KB", 
             total_size, total_size / 1_048_576, chunk_size / 1024);
    println!("Iterations: {}", iterations);
    
    let mem_start = get_memory_mb();
    let cpu_start = get_cpu_time_secs();
    let wall_start = Instant::now();
    
    for _ in 0..iterations {
        f();
    }
    
    let wall_elapsed = wall_start.elapsed().as_secs_f64();
    let cpu_elapsed = get_cpu_time_secs() - cpu_start;
    let mem_end = get_memory_mb();
    
    let total_bytes = (total_size * iterations) as f64;
    let throughput = total_bytes / wall_elapsed / 1_048_576.0;
    let cpu_efficiency = if cpu_elapsed > 0.0 { wall_elapsed / cpu_elapsed } else { 0.0 };
    
    println!("Wall time: {:.3}s", wall_elapsed);
    println!("CPU time: {:.3}s (user+system)", cpu_elapsed);
    println!("CPU efficiency: {:.1}% (lower is better)", cpu_efficiency * 100.0);
    println!("Throughput: {:.2} MB/s", throughput);
    println!("Memory start: {:.2} MB, end: {:.2} MB, delta: {:.2} MB", 
             mem_start, mem_end, mem_end - mem_start);
    
    (throughput, cpu_efficiency)
}

#[test]
fn test_performance_1mb_compress1() {
    let size = 1_048_576; // 1MB
    let iterations = 100;
    
    let (old_throughput, old_cpu) = benchmark(
        "OLD: 1MB compress=1 (incompressible)",
        size,
        iterations,
        || generate_controlled_data(size, 1, 1)
    );
    
    let (new_throughput, new_cpu) = benchmark(
        "NEW: 1MB compress=1 (incompressible)",
        size,
        iterations,
        || generate_controlled_data_alt(size, 1, 1)
    );
    
    println!("\n📊 COMPARISON:");
    println!("Speed:     OLD: {:.2} MB/s  vs  NEW: {:.2} MB/s  ({:.2}x)", 
             old_throughput, new_throughput, new_throughput / old_throughput);
    println!("CPU usage: OLD: {:.1}%      vs  NEW: {:.1}%      ({:.2}x)", 
             old_cpu * 100.0, new_cpu * 100.0, new_cpu / old_cpu);
    
    // NEW should be within 0.5x to 2x of OLD (allow for variance)
    assert!(new_throughput > old_throughput * 0.5, 
            "NEW is too slow: {:.2} MB/s vs OLD {:.2} MB/s", new_throughput, old_throughput);
}

#[test]
fn test_performance_16mb_compress1() {
    let size = 16_777_216; // 16MB
    let iterations = 20;
    
    let (old_throughput, old_cpu) = benchmark(
        "OLD: 16MB compress=1 (incompressible)",
        size,
        iterations,
        || generate_controlled_data(size, 1, 1)
    );
    
    let (new_throughput, new_cpu) = benchmark(
        "NEW: 16MB compress=1 (incompressible)",
        size,
        iterations,
        || generate_controlled_data_alt(size, 1, 1)
    );
    
    println!("\n📊 COMPARISON:");
    println!("Speed:     OLD: {:.2} MB/s  vs  NEW: {:.2} MB/s  ({:.2}x)", 
             old_throughput, new_throughput, new_throughput / old_throughput);
    println!("CPU usage: OLD: {:.1}%      vs  NEW: {:.1}%      ({:.2}x)", 
             old_cpu * 100.0, new_cpu * 100.0, new_cpu / old_cpu);
    
    assert!(new_throughput > old_throughput * 0.5, 
            "NEW is too slow: {:.2} MB/s vs OLD {:.2} MB/s", new_throughput, old_throughput);
}

#[test]
fn test_performance_64mb_compress1() {
    let size = 67_108_864; // 64MB
    let iterations = 5;
    
    let (old_throughput, old_cpu) = benchmark(
        "OLD: 64MB compress=1 (incompressible)",
        size,
        iterations,
        || generate_controlled_data(size, 1, 1)
    );
    
    let (new_throughput, new_cpu) = benchmark(
        "NEW: 64MB compress=1 (incompressible)",
        size,
        iterations,
        || generate_controlled_data_alt(size, 1, 1)
    );
    
    println!("\n📊 COMPARISON:");
    println!("Speed:     OLD: {:.2} MB/s  vs  NEW: {:.2} MB/s  ({:.2}x)", 
             old_throughput, new_throughput, new_throughput / old_throughput);
    println!("CPU usage: OLD: {:.1}%      vs  NEW: {:.1}%      ({:.2}x)", 
             old_cpu * 100.0, new_cpu * 100.0, new_cpu / old_cpu);
    
    assert!(new_throughput > old_throughput * 0.5, 
            "NEW is too slow: {:.2} MB/s vs OLD {:.2} MB/s", new_throughput, old_throughput);
}

#[test]
fn test_performance_compress5() {
    let size = 16_777_216; // 16MB
    let iterations = 20;
    
    let (old_throughput, old_cpu) = benchmark(
        "OLD: 16MB compress=5",
        size,
        iterations,
        || generate_controlled_data(size, 1, 5)
    );
    
    let (new_throughput, new_cpu) = benchmark(
        "NEW: 16MB compress=5",
        size,
        iterations,
        || generate_controlled_data_alt(size, 1, 5)
    );
    
    println!("\n📊 COMPARISON:");
    println!("Speed:     OLD: {:.2} MB/s  vs  NEW: {:.2} MB/s  ({:.2}x)", 
             old_throughput, new_throughput, new_throughput / old_throughput);
    println!("CPU usage: OLD: {:.1}%      vs  NEW: {:.1}%      ({:.2}x)", 
             old_cpu * 100.0, new_cpu * 100.0, new_cpu / old_cpu);
    
    assert!(new_throughput > old_throughput * 0.5, 
            "NEW is too slow: {:.2} MB/s vs OLD {:.2} MB/s", new_throughput, old_throughput);
}

#[test]
fn test_performance_streaming() {
    let total_size = 16_777_216; // 16MB
    let chunk_size = 262_144; // 256KB chunks
    let iterations = 20;
    
    let (old_throughput, old_cpu) = benchmark_streaming(
        "OLD: Streaming 16MB in 256KB chunks",
        total_size,
        chunk_size,
        iterations,
        || {
            let mut data_gen = DataGenerator::new();
            let mut obj_gen = data_gen.begin_object(total_size, 1, 1);
            while !obj_gen.is_complete() {
                if let Some(chunk) = obj_gen.fill_chunk(chunk_size) {
                    std::hint::black_box(&chunk);
                }
            }
        }
    );
    
    let (new_throughput, new_cpu) = benchmark_streaming(
        "NEW: Streaming 16MB in 256KB chunks",
        total_size,
        chunk_size,
        iterations,
        || {
            let mut obj_gen = ObjectGenAlt::new(total_size, 1, 1);
            let mut buffer = vec![0u8; chunk_size];
            while !obj_gen.is_complete() {
                obj_gen.fill_chunk(&mut buffer);
            }
        }
    );
    
    println!("\n📊 COMPARISON:");
    println!("Speed:     OLD: {:.2} MB/s  vs  NEW: {:.2} MB/s  ({:.2}x)", 
             old_throughput, new_throughput, new_throughput / old_throughput);
    println!("CPU usage: OLD: {:.1}%      vs  NEW: {:.1}%      ({:.2}x)", 
             old_cpu * 100.0, new_cpu * 100.0, new_cpu / old_cpu);
    
    assert!(new_throughput > old_throughput * 0.5, 
            "NEW is too slow: {:.2} MB/s vs OLD {:.2} MB/s", new_throughput, old_throughput);
}

#[test]
fn test_performance_with_dedup() {
    let size = 16_777_216; // 16MB
    let dedup = 4;
    let iterations = 20;
    
    let (old_throughput, old_cpu) = benchmark(
        "OLD: 16MB dedup=4",
        size,
        iterations,
        || generate_controlled_data(size, dedup, 1)
    );
    
    let (new_throughput, new_cpu) = benchmark(
        "NEW: 16MB dedup=4",
        size,
        iterations,
        || generate_controlled_data_alt(size, dedup, 1)
    );
    
    println!("\n📊 COMPARISON:");
    println!("Speed:     OLD: {:.2} MB/s  vs  NEW: {:.2} MB/s  ({:.2}x)", 
             old_throughput, new_throughput, new_throughput / old_throughput);
    println!("CPU usage: OLD: {:.1}%      vs  NEW: {:.1}%      ({:.2}x)", 
             old_cpu * 100.0, new_cpu * 100.0, new_cpu / old_cpu);
    
    assert!(new_throughput > old_throughput * 0.5, 
            "NEW is too slow: {:.2} MB/s vs OLD {:.2} MB/s", new_throughput, old_throughput);
}
