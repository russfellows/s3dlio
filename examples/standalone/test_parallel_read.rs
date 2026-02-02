// Test parallel reads with rayon
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn format_throughput(bytes: usize, duration: std::time::Duration) -> String {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let secs = duration.as_secs_f64();
    format!("{:.2} GB/s", gb / secs)
}

fn test_parallel_read(total_mb: usize, threads: usize) -> std::io::Result<()> {
    let chunk_mb = total_mb / threads;
    let chunk_bytes = chunk_mb * 1024 * 1024;
    let total_bytes_read = Arc::new(AtomicUsize::new(0));
    
    let start = Instant::now();
    
    let handles: Vec<_> = (0..threads).map(|_| {
        let total_ref = Arc::clone(&total_bytes_read);
        std::thread::spawn(move || -> std::io::Result<()> {
            let mut file = File::open("/dev/zero")?;
            let mut buffer = vec![0u8; chunk_bytes];
            let bytes = file.read(&mut buffer)?;
            total_ref.fetch_add(bytes, Ordering::Relaxed);
            Ok(())
        })
    }).collect();
    
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    let duration = start.elapsed();
    let total = total_bytes_read.load(Ordering::Relaxed);
    
    println!("{:4} MB ({} threads × {} MB): read {} bytes in {:?} ({})", 
             total_mb, threads, chunk_mb, total, duration, format_throughput(total, duration));
    
    Ok(())
}

fn main() -> std::io::Result<()> {
    println!("Testing parallel /dev/zero reads:");
    println!("{}", "=".repeat(70));
    
    // Test different thread counts with 4 GB total
    let total_mb = 4096;
    println!("\nTotal size: {} MB", total_mb);
    
    for thread_count in [1, 2, 4, 8, 16, 32] {
        test_parallel_read(total_mb, thread_count)?;
    }
    
    println!("\nComparison:");
    println!("  dd if=/dev/zero: 9.9 GB/s");
    println!("  fio (sync): 9.3 GB/s");
    println!("  fio (io_uring): 7.1 GB/s");
    println!("  Rust File::read (1 thread): 2.8 GB/s");
    
    Ok(())
}
