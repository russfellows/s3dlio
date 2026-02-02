# Filesystem I/O Benchmarking Guide

**Purpose**: Determine optimal I/O patterns for s3dlio's file backend to maximize performance across different storage systems.

**Date**: January 31, 2026

---

## Overview

This benchmark suite tests various I/O patterns to help optimize s3dlio's file operations:

- **Buffered I/O** vs **Direct I/O** (O_DIRECT)
- **Sequential** vs **Random** access patterns
- **Single-threaded** vs **Parallel** I/O
- **Different transfer sizes** (4 KB to 64 MB)

The results will guide implementation decisions for the file backend in s3dlio.

---

## Quick Start

### Default Testing (tmpfs)

```bash
cargo bench --bench filesystem_io_benchmark
```

This uses `/tmp/s3dlio-bench` by default (usually tmpfs/memory).

### Test Specific Filesystem

Set `S3DLIO_BENCH_DIR` to test different storage backends:

#### NVMe Direct
```bash
# Best performance - direct NVMe access
S3DLIO_BENCH_DIR=/mnt/scratch cargo bench --bench filesystem_io_benchmark
```

#### tmpfs (memory-backed)
```bash
# Fast but limited by RAM
S3DLIO_BENCH_DIR=/tmp cargo bench --bench filesystem_io_benchmark
```

#### Network Storage (NFS)
```bash
# Higher latency, good bandwidth
S3DLIO_BENCH_DIR=/mnt/vast1 cargo bench --bench filesystem_io_benchmark
```

#### Home Directory
```bash
# Typical SSD/HDD performance
S3DLIO_BENCH_DIR=$HOME/bench cargo bench --bench filesystem_io_benchmark
```

---

## Benchmark Categories

### 1. Buffered Write (`buffered_write`)

Tests standard file writes using Rust's `File::write_all()`:
- Uses OS page cache
- Multiple transfer sizes (4 KB - 64 MB)
- Includes `sync_all()` to ensure data is flushed

**What it measures**: Best-case write throughput with OS caching.

### 2. Buffered Read (`buffered_read`)

Tests standard file reads using `File::read_exact()`:
- Benefits from OS page cache
- Sequential access pattern
- Pre-warms cache with initial write

**What it measures**: Sequential read throughput with page cache.

### 3. Buffered Random Read (`buffered_random_read`)

Tests seek-based random access:
- 64 MB test file
- Various read sizes (4 KB, 64 KB, 1 MB)
- Pseudo-random seek pattern (stride-7 to avoid cache patterns)

**What it measures**: Random access overhead and seek performance.

### 4. Direct I/O Write (`direct_io_write`) [Linux only]

Tests O_DIRECT writes (bypass page cache):
- Requires 4096-byte aligned buffers
- No OS caching
- Direct-to-device writes

**What it measures**: Raw device write speed without buffering.

**Requirements**:
- Linux only
- Buffer size must be multiple of 4096 bytes
- Path must be on filesystem supporting O_DIRECT

### 5. Direct I/O Read (`direct_io_read`) [Linux only]

Tests O_DIRECT reads:
- Aligned buffers (4096 bytes)
- Bypasses page cache
- True device read speed

**What it measures**: Raw device read speed, important for large sequential I/O.

### 6. Parallel Buffered Write (`parallel_buffered_write`)

Tests concurrent writes across threads:
- 16 MB total transfer
- 1, 2, 4, 8, 16 thread configurations
- Each thread writes to separate file

**What it measures**: Parallel I/O scalability and thread contention.

---

## Understanding Results

### Key Metrics

**Throughput**: MB/s or GB/s - higher is better
**Latency**: Time per operation - lower is better

### Expected Performance Ranges

Based on loki-node3 testing (32-core system):

| Storage Backend | Write Speed | Read Speed | Use Case |
|----------------|-------------|------------|----------|
| NVMe direct (`/mnt/scratch`) | 6-7 GB/s | 12+ GB/s | Production, large files |
| tmpfs (`/tmp`) | 2-3 GB/s | 9-10 GB/s | Testing, temp files |
| NFS (`/mnt/vast1`) | 1-1.5 GB/s | 1-1.5 GB/s | Shared storage, moderate I/O |

### Performance Patterns to Look For

#### 1. **Transfer Size Sweet Spot**
- Small transfers (4 KB): High overhead, low throughput
- Medium transfers (64 KB - 1 MB): Best efficiency
- Large transfers (16-64 MB): Maximum throughput but higher latency

**Decision**: Choose default chunk size for s3dlio file operations.

#### 2. **Buffered vs Direct I/O**
- **Buffered faster**: Use for small files, random access
- **Direct I/O faster**: Use for large sequential operations (>10 MB)

**Decision**: When to use O_DIRECT in file backend.

#### 3. **Parallel Scaling**
- **Good scaling (linear)**: Use thread pool for I/O
- **Poor scaling (contention)**: Single-threaded or limit concurrency

**Decision**: Thread pool size for parallel file operations.

#### 4. **Random Access Penalty**
- **Low penalty**: Filesystem has good caching/indexing
- **High penalty (10x+ slower)**: Avoid random access, prefer sequential

**Decision**: Whether to optimize for sequential-only access.

---

## Comparing Storage Backends

Run benchmarks on multiple filesystems and compare:

```bash
# NVMe
S3DLIO_BENCH_DIR=/mnt/scratch cargo bench --bench filesystem_io_benchmark > results_nvme.txt

# tmpfs
S3DLIO_BENCH_DIR=/tmp cargo bench --bench filesystem_io_benchmark > results_tmpfs.txt

# NFS
S3DLIO_BENCH_DIR=/mnt/vast1 cargo bench --bench filesystem_io_benchmark > results_nfs.txt

# Compare
grep "time:" results_*.txt | sort
```

---

## Common Issues

### Direct I/O Failures

**Error**: "Invalid argument" when opening with O_DIRECT

**Causes**:
1. Filesystem doesn't support O_DIRECT (tmpfs, some network filesystems)
2. Buffer not aligned to 4096 bytes
3. Transfer size not multiple of 4096 bytes

**Solution**: Only use on local filesystems (ext4, xfs, btrfs).

### Permission Errors

**Error**: "Permission denied" creating test directory

**Solution**: Ensure `S3DLIO_BENCH_DIR` is writable:
```bash
mkdir -p /your/test/path
chmod 755 /your/test/path
```

### Low Performance

**Possible causes**:
- Disk is busy (check `iostat -x 1`)
- Filesystem is full (check `df -h`)
- Network issues (for NFS/remote storage)
- CPU throttling (check CPU governor)

---

## Using Results to Optimize s3dlio

### Step 1: Identify Bottlenecks

Look at benchmark output for:
- Which I/O pattern is fastest for your target filesystem?
- Does Direct I/O help or hurt performance?
- What transfer size gives best throughput?

### Step 2: Apply Findings

Update `file_store.rs` and `file_store_direct.rs`:

```rust
// Example: If benchmarks show 1 MB chunks are optimal
const OPTIMAL_CHUNK_SIZE: usize = 1 << 20;  // 1 MB

// Example: If Direct I/O shows 2x improvement on NVMe
#[cfg(target_os = "linux")]
fn should_use_direct_io(file_size: usize) -> bool {
    file_size > 16 << 20  // Use O_DIRECT for files >16 MB
}
```

### Step 3: Verify Improvements

Re-run application benchmarks to confirm optimization:
```bash
# Your actual workload
cargo run --release --example your_s3_test
```

---

## Advanced Testing

### Custom Transfer Sizes

Edit `benches/filesystem_io_benchmark.rs`:

```rust
const TEST_SIZES: &[usize] = &[
    256 << 10,    // 256 KB
    512 << 10,    // 512 KB  
    2 << 20,      // 2 MB
    8 << 20,      // 8 MB
];
```

### Custom Thread Counts

```rust
let thread_counts = vec![1, 4, 8, 16, 32];  // Test up to 32 threads
```

### Test Specific Operation

Run single benchmark:
```bash
S3DLIO_BENCH_DIR=/mnt/scratch cargo bench --bench filesystem_io_benchmark -- buffered_read
```

---

## Interpreting Criterion Output

```
buffered_write/4194304  time:   [1.2345 ms 1.2567 ms 1.2789 ms]
                        thrpt:  [3.12 GiB/s 3.18 GiB/s 3.24 GiB/s]
```

- **time**: Lower is better (time per operation)
- **thrpt**: Higher is better (throughput)
- **First number**: Lower bound (fastest)
- **Middle number**: Median (typical)
- **Last number**: Upper bound (slowest)

**Focus on median values** for typical performance.

---

## Next Steps

After gathering results:

1. **Document findings** in this file or `s3dlio/docs/Changelog.md`
2. **Update file backend** based on optimal patterns
3. **Add adaptive logic** to choose I/O method based on file size/pattern
4. **Create regression tests** to catch performance degradation

---

## Example Results Template

Document your findings:

```markdown
## Benchmark Results - loki-node3 (Jan 31, 2026)

**System**: 32-core AMD, NVMe SSD
**Filesystem**: ext4 on /mnt/scratch

### Findings

1. **Optimal chunk size**: 1-4 MB for sequential I/O
2. **Direct I/O**: 15% faster for files >16 MB
3. **Parallel I/O**: Linear scaling up to 8 threads
4. **Random access**: 8x slower than sequential (avoid if possible)

### Recommendations

- Use 1 MB chunks for file operations
- Enable O_DIRECT for large files (>16 MB)
- Limit parallelism to 8 concurrent file operations
- Optimize for sequential access patterns
```

---

## Troubleshooting

### Benchmarks Running Slowly

```bash
# Run subset of sizes
S3DLIO_BENCH_DIR=/mnt/scratch cargo bench --bench filesystem_io_benchmark -- "buffered_write/1048576"
```

### Need Faster Iteration

```bash
# Quick mode (fewer samples)
cargo bench --bench filesystem_io_benchmark -- --quick
```

### Clean Up Test Files

Benchmarks auto-cleanup, but if interrupted:
```bash
rm -f /tmp/s3dlio-bench/*.tmp /tmp/s3dlio-bench/*.dat
```

---

**Remember**: The goal is to find the **BEST I/O patterns** for s3dlio, not just the fastest filesystem. Test realistic workloads and prioritize patterns that will benefit real users.
