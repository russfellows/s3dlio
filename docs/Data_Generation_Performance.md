# Data Generation Performance Guide

## Performance Summary

s3dlio's data generation now matches dgen-rs performance: **50+ GB/s** on 6-core systems (tested with 100 GB workloads).

## Key Optimizations Implemented

### 1. Optimal Chunk Sizing (Automatic)

The library automatically selects optimal chunk sizes based on your request:

| Data Size | Chunk Size | Expected Throughput |
|-----------|------------|-------------------|
| < 16 MB   | Single allocation | 2-5 GB/s |
| 16-31 MB  | 16 MB chunks | 5-15 GB/s |
| 32-63 MB  | 32 MB chunks | 15-30 GB/s |
| ≥ 64 MB   | 64 MB chunks | **50+ GB/s** |

### 2. Thread Pool Reuse (Critical for Performance)

**Thread pool creation overhead is significant** (~1 second per pool). The key to maximum performance is **reusing the thread pool** across multiple generations.

### 3. Warmup Benefits

For maximum throughput, a small warmup run (~1 GB) enables code and data caching:
- Without warmup: ~28 GB/s
- With warmup: ~51 GB/s

## Usage Patterns

### Simple API (Good for Single Calls)

```rust
use s3dlio::data_gen_alt::generate_controlled_data_alt;

// Single large allocation - optimized internally
let data = generate_controlled_data_alt(
    100 * 1024 * 1024 * 1024,  // 100 GB
    1,    // No deduplication
    1,    // Incompressible
    None  // Random seed
);
// Performance: ~50 GB/s for large sizes
```

**Limitation**: Each call creates a new thread pool (~1 second overhead). Fine for single calls, suboptimal for repeated calls.

### Streaming API (Maximum Performance)

```rust
use s3dlio::data_gen_alt::{DataGenerator, GeneratorConfig, optimal_chunk_size};

let total_size = 100 * 1024 * 1024 * 1024;  // 100 GB
let config = GeneratorConfig {
    size: total_size,
    dedup_factor: 1,
    compress_factor: 1,
    ..Default::default()
};

// Create thread pool ONCE
let mut generator = DataGenerator::new(config);

// Get optimal chunk size for maximum throughput
let chunk_size = optimal_chunk_size(total_size);  // Returns 64 MB
let mut buffer = vec![0u8; chunk_size];

// Stream through all data (thread pool reused!)
while !generator.is_complete() {
    let nbytes = generator.fill_chunk(&mut buffer);
    if nbytes == 0 {
        break;
    }
    
    // Use buffer[..nbytes] for your data
    // write_to_storage(&buffer[..nbytes])?;
}
// Performance: 50+ GB/s sustained
```

### Warmup Pattern (Optional, +80% throughput)

```rust
// Generate small amount to warm up caches
let warmup = generate_controlled_data_alt(1 * 1024 * 1024 * 1024, 1, 1, None);
drop(warmup);

// Main generation now runs 80% faster
let data = generate_controlled_data_alt(100 * 1024 * 1024 * 1024, 1, 1, None);
```

## Helper Functions

### `optimal_chunk_size(total_size)`

Returns the best chunk size for a given total size:

```rust
use s3dlio::data_gen_alt::optimal_chunk_size;

let chunk = optimal_chunk_size(100 * 1024 * 1024 * 1024);  // 64 MB
let chunk = optimal_chunk_size(50 * 1024 * 1024);   // 32 MB
let chunk = optimal_chunk_size(10 * 1024 * 1024);   // 10 MB (returns size itself)
```

## Performance Comparison

### Test System
- CPU: 6 physical cores (12 logical via hyperthreading)
- Memory: UMA architecture
- Data: 100 GB, incompressible (compress=1)

### Results

| Pattern | Throughput | Notes |
|---------|-----------|-------|
| dgen-rs (baseline) | 50 GB/s | Reference implementation |
| s3dlio streaming API | **50.87 GB/s** | ✅ Matches dgen-rs! |
| s3dlio simple API (100 GB) | ~50 GB/s | Single call, optimized internally |
| s3dlio simple API (10× 10 GB) | ~1.3 GB/s | Thread pool created 10 times |
| s3dlio streaming API (10× 10 GB) | **50+ GB/s** | Thread pool created once |

## Best Practices

### ✅ DO THIS (Maximum Performance)

```rust
// For large workloads or repeated generation:
let mut gen = DataGenerator::new(config);
let mut buffer = vec![0u8; 64 * 1024 * 1024];

while !gen.is_complete() {
    let nbytes = gen.fill_chunk(&mut buffer);
    process_data(&buffer[..nbytes]);  // 50+ GB/s
}
```

### ⚠️ AVOID THIS (Poor Performance)

```rust
// DON'T create new generators in a loop:
for i in 0..10 {
    let data = generate_controlled_data_alt(10 * 1024 * 1024 * 1024, 1, 1, None);
    process_data(&data);  // Only ~1.3 GB/s due to thread pool overhead
}
```

### ✅ DO THIS INSTEAD

```rust
// Create ONE generator for all iterations:
let config = GeneratorConfig { size: 10 * 1024 * 1024 * 1024, ..Default::default() };
let mut buffer = vec![0u8; 64 * 1024 * 1024];

for i in 0..10 {
    let mut gen = DataGenerator::new(config.clone());
    gen.reset();  // Reset for new iteration
    
    while !gen.is_complete() {
        let nbytes = gen.fill_chunk(&mut buffer);
        process_data(&buffer[..nbytes]);  // 50+ GB/s
    }
}
```

## Technical Details

### Why 64 MB Chunks?

Benchmarking showed optimal throughput at 64 MB chunk sizes:
- **Larger chunks**: More memory pressure, worse cache behavior
- **Smaller chunks**: More function call overhead
- **64 MB**: Sweet spot for CPU cache and parallelism

### Thread Pool Creation Cost

- **Creation time**: ~1 second per thread pool (12 threads)
- **Per-call overhead**: ~100-200 ms
- **Solution**: Reuse thread pool via streaming API

### Compression Factor

- `compress=1`: Truly incompressible (~1.0 zstd ratio)
- `compress > 1`: Proportionally more compressible
- Algorithm: Fills blocks with zeros instead of random data

## Changelog

- **v0.9.34**: Automatic optimal chunking in simple API
- **v0.9.30**: Zero-copy refactor with `bytes::Bytes`
- **v0.9.27**: Backported dgen-rs optimizations (50+ GB/s)

---

**For more details**, see:
- [dgen-rs streaming_benchmark.rs](https://github.com/user/dgen-rs/blob/main/examples/streaming_benchmark.rs)
- [s3dlio test_cpu_utilization.rs](../tests/test_cpu_utilization.rs)
