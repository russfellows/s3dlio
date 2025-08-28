use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use s3dlio::{data_gen::generate_controlled_data, s3_utils::*};
use std::env;

/// Benchmark data splitting and buffer management for multipart uploads
fn bench_multipart_splits(c: &mut Criterion) {
    let mut group = c.benchmark_group("multipart_buffer_ops");
    
    // Test different part sizes that are commonly used
    for part_size in [4 << 20, 8 << 20, 16 << 20, 64 << 20] {  // 4MB, 8MB, 16MB, 64MB
        group.throughput(Throughput::Bytes(part_size as u64 * 2)); // 2 parts worth
        
        group.bench_with_input(
            format!("split_to_{}MB", part_size >> 20), 
            &part_size, 
            |b, &part_size| {
                b.iter(|| {
                    // Simulate realistic buffer operations for multipart upload
                    let mut buffer = bytes::BytesMut::with_capacity(part_size * 2);
                    buffer.resize(part_size * 2, 42u8);
                    
                    // Split off first part (zero-copy operation)
                    let first_part = buffer.split_to(part_size).freeze();
                    criterion::black_box(first_part);
                    
                    // Remaining data (simulates leftover in buffer)
                    criterion::black_box(buffer);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark data generation performance (baseline for comparison)
fn bench_data_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_generation");
    
    // Test different data sizes
    for size in [1 << 20, 4 << 20, 16 << 20] {  // 1MB, 4MB, 16MB
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            format!("generate_{}MB", size >> 20),
            &size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_controlled_data(
                        size,
                        10, // dedup_factor
                        1   // compress_factor
                    );
                    criterion::black_box(data);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark vector operations commonly used in S3 operations
fn bench_vector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");
    
    for size in [1 << 20, 4 << 20, 16 << 20] {  // 1MB, 4MB, 16MB
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark Vec::with_capacity + extend_from_slice (common pattern)
        group.bench_with_input(
            format!("vec_extend_{}MB", size >> 20),
            &size,
            |b, &size| {
                let source_data = vec![42u8; size];
                b.iter(|| {
                    let mut target = Vec::with_capacity(size);
                    target.extend_from_slice(&source_data);
                    criterion::black_box(target);
                });
            },
        );
        
        // Benchmark pre-allocated buffer copy (optimal pattern)
        group.bench_with_input(
            format!("vec_copy_{}MB", size >> 20),
            &size,
            |b, &size| {
                let source_data = vec![42u8; size];
                b.iter(|| {
                    let mut target = vec![0u8; size];
                    target.copy_from_slice(&source_data);
                    criterion::black_box(target);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark URI parsing (happens on every S3 operation)
fn bench_uri_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("uri_parsing");
    
    let test_uris = vec![
        "s3://my-bucket/key",
        "s3://my-bucket/path/to/nested/key",
        "s3://my-bucket/very/deeply/nested/path/with/many/segments/file.dat",
    ];
    
    for uri in &test_uris {
        group.bench_with_input(
            format!("parse_uri_{}_segments", uri.matches('/').count()),
            uri,
            |b, uri| {
                b.iter(|| {
                    let result = parse_s3_uri(uri);
                    criterion::black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches, 
    bench_multipart_splits,
    bench_data_generation,
    bench_vector_ops,
    bench_uri_parsing
);
criterion_main!(benches);
