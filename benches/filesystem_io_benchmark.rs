// Comprehensive filesystem I/O benchmarks for s3dlio
// 
// Purpose: Determine optimal I/O patterns for both direct and buffered I/O
// to improve s3dlio's file backend performance
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use s3dlio::data_gen::fill_controlled_data;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::PathBuf;

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

const TEST_SIZES: &[usize] = &[
    4 << 10,      // 4 KB
    64 << 10,     // 64 KB
    1 << 20,      // 1 MB
    4 << 20,      // 4 MB
    16 << 20,     // 16 MB
    64 << 20,     // 64 MB
];

/// Get the test directory from environment variable or use default
fn get_test_dir() -> PathBuf {
    env::var("S3DLIO_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let default = PathBuf::from("/tmp/s3dlio-bench");
            eprintln!("⚠️  S3DLIO_BENCH_DIR not set, using default: {}", default.display());
            eprintln!("   Set S3DLIO_BENCH_DIR to test different filesystems");
            eprintln!("   Example: S3DLIO_BENCH_DIR=/mnt/scratch cargo bench --bench filesystem_io_benchmark");
            default
        })
}

/// Ensure test directory exists
fn setup_test_dir() -> io::Result<PathBuf> {
    let dir = get_test_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Cleanup test files
fn cleanup_test_files(dir: &PathBuf, pattern: &str) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                if name.contains(pattern) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
}

// ============================================================================
// BUFFERED I/O BENCHMARKS
// ============================================================================

/// Benchmark: Buffered sequential write
fn bench_buffered_write(c: &mut Criterion) {
    let test_dir = setup_test_dir().expect("Failed to create test directory");
    let mut group = c.benchmark_group("buffered_write");
    
    for &size in TEST_SIZES.iter() {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Pre-generate test data
        let mut data = vec![0u8; size];
        fill_controlled_data(&mut data, 1, 1);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    let path = test_dir.join(format!("buffered_write_{}.tmp", size));
                    let mut file = File::create(&path).expect("Failed to create file");
                    file.write_all(&data).expect("Failed to write");
                    file.sync_all().expect("Failed to sync");
                    let _ = fs::remove_file(path);
                });
            },
        );
    }
    
    cleanup_test_files(&test_dir, "buffered_write");
    group.finish();
}

/// Benchmark: Buffered sequential read
fn bench_buffered_read(c: &mut Criterion) {
    let test_dir = setup_test_dir().expect("Failed to create test directory");
    let mut group = c.benchmark_group("buffered_read");
    
    for &size in TEST_SIZES.iter() {
        // Pre-create test file
        let path = test_dir.join(format!("buffered_read_{}.dat", size));
        let mut data = vec![0u8; size];
        fill_controlled_data(&mut data, 1, 1);
        {
            let mut file = File::create(&path).expect("Failed to create file");
            file.write_all(&data).expect("Failed to write");
            file.sync_all().expect("Failed to sync");
        }
        
        group.throughput(Throughput::Bytes(size as u64));
        
        let mut read_buffer = vec![0u8; size];
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut file = File::open(&path).expect("Failed to open file");
                    file.read_exact(&mut read_buffer).expect("Failed to read");
                });
            },
        );
        
        let _ = fs::remove_file(path);
    }
    
    cleanup_test_files(&test_dir, "buffered_read");
    group.finish();
}

/// Benchmark: Buffered random read (seek patterns)
fn bench_buffered_random_read(c: &mut Criterion) {
    let test_dir = setup_test_dir().expect("Failed to create test directory");
    let mut group = c.benchmark_group("buffered_random_read");
    
    let file_size = 64 << 20;  // 64 MB file
    let read_sizes = vec![4 << 10, 64 << 10, 1 << 20];  // 4KB, 64KB, 1MB
    
    // Create test file
    let path = test_dir.join("buffered_random_read.dat");
    {
        let mut data = vec![0u8; file_size];
        fill_controlled_data(&mut data, 1, 1);
        let mut file = File::create(&path).expect("Failed to create file");
        file.write_all(&data).expect("Failed to write");
        file.sync_all().expect("Failed to sync");
    }
    
    for &read_size in read_sizes.iter() {
        group.throughput(Throughput::Bytes(read_size as u64));
        
        let mut buffer = vec![0u8; read_size];
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", read_size)),
            &read_size,
            |b, _| {
                let mut file = File::open(&path).expect("Failed to open file");
                let mut offset = 0u64;
                
                b.iter(|| {
                    file.seek(SeekFrom::Start(offset)).expect("Failed to seek");
                    file.read_exact(&mut buffer).expect("Failed to read");
                    
                    // Move to next random offset
                    offset = (offset + read_size as u64 * 7) % (file_size - read_size) as u64;
                });
            },
        );
    }
    
    let _ = fs::remove_file(path);
    group.finish();
}

// ============================================================================
// DIRECT I/O BENCHMARKS (Linux only)
// ============================================================================

#[cfg(target_os = "linux")]
fn bench_direct_io_write(c: &mut Criterion) {
    let test_dir = setup_test_dir().expect("Failed to create test directory");
    let mut group = c.benchmark_group("direct_io_write");
    
    // Direct I/O requires aligned buffers (typically 4KB)
    const ALIGNMENT: usize = 4096;
    
    for &size in TEST_SIZES.iter() {
        // Size must be multiple of alignment
        if size % ALIGNMENT != 0 {
            continue;
        }
        
        group.throughput(Throughput::Bytes(size as u64));
        
        // Create aligned buffer
        let mut data = vec![0u8; size + ALIGNMENT];
        let offset = data.as_ptr().align_offset(ALIGNMENT);
        let aligned_data = &mut data[offset..offset + size];
        fill_controlled_data(aligned_data, 1, 1);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    let path = test_dir.join(format!("direct_write_{}.tmp", size));
                    let file = OpenOptions::new()
                        .create(true)
                        .write(true)
                        .custom_flags(libc::O_DIRECT)
                        .open(&path)
                        .expect("Failed to open with O_DIRECT");
                    
                    use std::os::unix::io::AsRawFd;
                    use std::io::Write;
                    
                    let fd = file.as_raw_fd();
                    let written = unsafe {
                        libc::write(
                            fd,
                            aligned_data.as_ptr() as *const libc::c_void,
                            aligned_data.len(),
                        )
                    };
                    
                    assert_eq!(written, aligned_data.len() as isize);
                    
                    unsafe { libc::fsync(fd) };
                    let _ = fs::remove_file(path);
                });
            },
        );
    }
    
    cleanup_test_files(&test_dir, "direct_write");
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_direct_io_read(c: &mut Criterion) {
    let test_dir = setup_test_dir().expect("Failed to create test directory");
    let mut group = c.benchmark_group("direct_io_read");
    
    const ALIGNMENT: usize = 4096;
    
    for &size in TEST_SIZES.iter() {
        if size % ALIGNMENT != 0 {
            continue;
        }
        
        // Create test file with aligned writes
        let path = test_dir.join(format!("direct_read_{}.dat", size));
        {
            let mut data = vec![0u8; size + ALIGNMENT];
            let offset = data.as_ptr().align_offset(ALIGNMENT);
            let aligned_data = &mut data[offset..offset + size];
            fill_controlled_data(aligned_data, 1, 1);
            
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .custom_flags(libc::O_DIRECT)
                .open(&path)
                .expect("Failed to open with O_DIRECT");
            
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            unsafe {
                libc::write(
                    fd,
                    aligned_data.as_ptr() as *const libc::c_void,
                    aligned_data.len(),
                );
                libc::fsync(fd);
            }
        }
        
        group.throughput(Throughput::Bytes(size as u64));
        
        // Prepare aligned read buffer
        let mut read_buffer = vec![0u8; size + ALIGNMENT];
        let read_offset = read_buffer.as_ptr().align_offset(ALIGNMENT);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    let file = OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_DIRECT)
                        .open(&path)
                        .expect("Failed to open with O_DIRECT");
                    
                    use std::os::unix::io::AsRawFd;
                    let fd = file.as_raw_fd();
                    let aligned_buf = &mut read_buffer[read_offset..read_offset + size];
                    
                    let bytes_read = unsafe {
                        libc::read(
                            fd,
                            aligned_buf.as_mut_ptr() as *mut libc::c_void,
                            size,
                        )
                    };
                    
                    assert_eq!(bytes_read, size as isize);
                });
            },
        );
        
        let _ = fs::remove_file(path);
    }
    
    cleanup_test_files(&test_dir, "direct_read");
    group.finish();
}

// ============================================================================
// PARALLEL I/O PATTERNS
// ============================================================================

/// Benchmark: Parallel buffered writes
fn bench_parallel_buffered_write(c: &mut Criterion) {
    let test_dir = setup_test_dir().expect("Failed to create test directory");
    let mut group = c.benchmark_group("parallel_buffered_write");
    
    let size = 16 << 20;  // 16 MB total
    let thread_counts = vec![1, 2, 4, 8, 16];
    
    for &threads in thread_counts.iter() {
        let chunk_size = size / threads;
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", threads)),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    use std::thread;
                    
                    let handles: Vec<_> = (0..threads).map(|i| {
                        let test_dir = test_dir.clone();
                        
                        thread::spawn(move || {
                            let path = test_dir.join(format!("parallel_write_{}_{}.tmp", threads, i));
                            let mut data = vec![0u8; chunk_size];
                            fill_controlled_data(&mut data, 1, 1);
                            
                            let mut file = File::create(&path).expect("Failed to create file");
                            file.write_all(&data).expect("Failed to write");
                            file.sync_all().expect("Failed to sync");
                            let _ = fs::remove_file(path);
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().expect("Thread panicked");
                    }
                });
            },
        );
    }
    
    cleanup_test_files(&test_dir, "parallel_write");
    group.finish();
}

// ============================================================================
// CRITERION SETUP
// ============================================================================

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_buffered_write,
    bench_buffered_read,
    bench_buffered_random_read,
    bench_direct_io_write,
    bench_direct_io_read,
    bench_parallel_buffered_write,
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_buffered_write,
    bench_buffered_read,
    bench_buffered_random_read,
    bench_parallel_buffered_write,
);

criterion_main!(benches);
