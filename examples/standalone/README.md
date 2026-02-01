# Standalone Storage Performance Tests

These tests are designed to run independently without s3dlio dependencies, allowing you to compare raw Rust I/O performance against s3dlio's implementation.

## Tests

### 1. test_rust_read.rs - I/O Method Comparison

Compares different Rust standard library I/O methods:
- `std::fs::read()` - Convenience method, allocates Vec internally
- `File::read()` - Direct file read with pre-allocated buffer
- `BufReader::read()` - Buffered reader with 64MB buffer
- `File::read_exact()` - Ensures complete read

**Build:**
```bash
rustc -O test_rust_read.rs -o test_rust_read
```

**Prepare:**
```bash
# Create 4 GB test file
dd if=/dev/zero of=/tmp/test_4gb.dat bs=1M count=4096
```

**Run:**
```bash
./test_rust_read
```

**Expected Output:**
- BufReader: ~9+ GB/s (best for sequential reads)
- File::read: ~8 GB/s
- std::fs::read: ~3 GB/s (slower due to dynamic allocation)

### 2. test_parallel_read.rs - Multi-threaded Performance

Tests parallel reads from `/dev/zero` with 1-32 threads.

**Build:**
```bash
rustc -O test_parallel_read.rs -o test_parallel_read
```

**Run:**
```bash
./test_parallel_read
```

**Expected Output:**
- 1 thread: ~2.8 GB/s (baseline)
- 4 threads: ~10 GB/s
- 16 threads: ~27 GB/s
- 32 threads: ~30 GB/s (near memory bandwidth)

**Comparison:**
- dd if=/dev/zero: 9.9 GB/s
- fio (sync): 9.3 GB/s
- fio (io_uring): 7.1 GB/s

## Quick Test on Remote System

```bash
# 1. Extract tarball
tar xzf s3dlio-bench.tar.gz
cd s3dlio/examples/standalone

# 2. Build tests
rustc -O test_rust_read.rs -o test_rust_read
rustc -O test_parallel_read.rs -o test_parallel_read

# 3. Prepare test data
dd if=/dev/zero of=/tmp/test_4gb.dat bs=1M count=4096

# 4. Run tests
./test_rust_read
./test_parallel_read
```

## Comparing Against s3dlio

After running these baselines, compare against s3dlio:

```bash
cd ../..  # Back to s3dlio root

# Build s3dlio
cargo build --release

# Run s3dlio Direct I/O tests
cargo run --release --example test_direct_io_comprehensive

# Run s3dlio production performance tests
cargo test --release --test test_production_performance -- --nocapture
```

## Performance Expectations

### Local VM / vSAN (Your Current System)
- Sequential read: 1-3 GB/s
- Parallel read (8+ threads): 3-8 GB/s
- Page cache: 10-20 GB/s

### Remote NVMe (Target Systems)
- Sequential read: 3-8 GB/s
- Parallel read (16+ threads): 15-30 GB/s
- Page cache: 20-50 GB/s

### Cloud Instance (AWS/Azure/GCP)
- EBS/Disk: 1-3 GB/s
- Instance store NVMe: 3-8 GB/s
- Page cache: 15-40 GB/s
