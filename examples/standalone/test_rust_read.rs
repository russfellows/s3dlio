// Test different Rust I/O methods for performance comparison
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;

fn format_throughput(bytes: usize, duration: std::time::Duration) -> String {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let secs = duration.as_secs_f64();
    format!("{:.2} GB/s", gb / secs)
}

fn test_std_fs_read(path: &str, size: usize) -> std::io::Result<std::time::Duration> {
    let start = Instant::now();
    let data = std::fs::read(path)?;
    let duration = start.elapsed();
    println!("  std::fs::read: read {} bytes in {:?} ({})", 
             data.len().min(size), duration, format_throughput(size, duration));
    Ok(duration)
}

fn test_file_read(path: &str, size: usize) -> std::io::Result<std::time::Duration> {
    let start = Instant::now();
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; size];
    let bytes_read = file.read(&mut buffer)?;
    let duration = start.elapsed();
    println!("  File::read: read {} bytes in {:?} ({})", 
             bytes_read, duration, format_throughput(bytes_read, duration));
    Ok(duration)
}

fn test_buffered_read(path: &str, size: usize) -> std::io::Result<std::time::Duration> {
    let start = Instant::now();
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024 * 1024, file); // 64 MB buffer
    let mut buffer = vec![0u8; size];
    let bytes_read = reader.read(&mut buffer)?;
    let duration = start.elapsed();
    println!("  BufReader::read (64MB): read {} bytes in {:?} ({})", 
             bytes_read, duration, format_throughput(bytes_read, duration));
    Ok(duration)
}

fn test_read_exact(path: &str, size: usize) -> std::io::Result<std::time::Duration> {
    let start = Instant::now();
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)?;
    let duration = start.elapsed();
    println!("  File::read_exact: read {} bytes in {:?} ({})", 
             buffer.len(), duration, format_throughput(buffer.len(), duration));
    Ok(duration)
}

fn main() -> std::io::Result<()> {
    let test_file = "/tmp/test_4gb.dat";
    let sizes_mb = vec![64, 128, 256, 512, 1024];
    
    println!("Testing Rust I/O methods:");
    println!("{}", "=".repeat(60));
    
    for size_mb in sizes_mb {
        let size_bytes = size_mb * 1024 * 1024;
        println!("\n{} MB:", size_mb);
        
        // Test 1: std::fs::read (allocates Vec internally)
        test_std_fs_read(test_file, size_bytes)?;
        
        // Test 2: File::read with pre-allocated buffer
        test_file_read(test_file, size_bytes)?;
        
        // Test 3: BufReader with 64 MB buffer
        test_buffered_read(test_file, size_bytes)?;
        
        // Test 4: File::read_exact
        test_read_exact(test_file, size_bytes)?;
    }
    
    Ok(())
}
