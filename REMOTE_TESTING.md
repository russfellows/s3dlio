# Remote System Testing Guide

This guide helps you run s3dlio storage performance tests on remote systems with fast NVMe storage.

## Quick Start

### 1. Transfer Tarball to Remote System

```bash
# From your local machine
scp s3dlio-bench.tar.gz user@remote-system:/tmp/

# Or use rsync
rsync -avz s3dlio-bench.tar.gz user@remote-system:/tmp/
```

### 2. Extract on Remote System

```bash
ssh user@remote-system
cd /tmp
tar xzf s3dlio-bench.tar.gz
cd s3dlio
```

### 3. Build s3dlio (Release Mode)

```bash
# Build the main library
cargo build --release

# Build with Direct I/O support (recommended)
cargo build --release --features direct-io
```

### 4. Run Quick Validation Tests

```bash
# Run standalone comparison tests first (no dependencies)
cd examples/standalone
rustc -O test_parallel_read.rs -o test_parallel_read
./test_parallel_read

rustc -O test_rust_read.rs -o test_rust_read
dd if=/dev/zero of=/tmp/test_4gb.dat bs=1M count=4096
./test_rust_read

# Back to s3dlio root
cd ../..
```

### 5. Run s3dlio Performance Tests

```bash
# Comprehensive Direct I/O test (RECOMMENDED)
cargo run --release --example test_direct_io_comprehensive

# All Direct I/O examples
cargo run --release --example test_direct_io
cargo run --release --example test_hybrid_io_debug
cargo run --release --example test_aligned_buffers

# Production performance test
cargo test --release --test test_production_performance -- --nocapture

# Full benchmark suite (30-60 minutes)
cargo bench
```

## Expected Performance on NVMe

### Gen4 NVMe (7+ GB/s sequential)
- test_parallel_read (16 threads): 20-40 GB/s
- test_rust_read (BufReader): 10-20 GB/s
- s3dlio Direct I/O: 5-15 GB/s
- s3dlio Buffered I/O (warm): 20-50 GB/s

### Gen3 NVMe (3-5 GB/s sequential)
- test_parallel_read (16 threads): 15-30 GB/s
- test_rust_read (BufReader): 8-15 GB/s
- s3dlio Direct I/O: 3-8 GB/s
- s3dlio Buffered I/O (warm): 15-40 GB/s

## Test Matrix

Run these tests in order to build a complete performance profile:

### Phase 1: Baseline (No s3dlio)
```bash
cd examples/standalone
./test_parallel_read        # Memory/CPU baseline
./test_rust_read            # Filesystem baseline
```

### Phase 2: s3dlio Direct I/O (Unbuffered)
```bash
cd ../..
cargo run --release --example test_direct_io_comprehensive
cargo run --release --example test_aligned_buffers
```

### Phase 3: s3dlio Buffered I/O (Page Cache)
```bash
cargo run --release --example test_hybrid_io_debug
cargo test --release --test test_production_performance -- --nocapture
```

### Phase 4: Full Benchmarks (Statistical Analysis)
```bash
cargo bench --bench performance_microbenchmarks
cargo bench --bench rng_performance_benchmark
```

## Collecting Results

### Save Test Output
```bash
# Create results directory
mkdir -p ~/s3dlio-results

# Run tests and capture output
./test_parallel_read > ~/s3dlio-results/01_parallel_baseline.txt
./test_rust_read > ~/s3dlio-results/02_rust_io_baseline.txt
cargo run --release --example test_direct_io_comprehensive > ~/s3dlio-results/03_direct_io.txt
cargo test --release --test test_production_performance -- --nocapture > ~/s3dlio-results/04_production.txt

# Benchmark results are in target/criterion/
tar czf ~/s3dlio-results/criterion-reports.tar.gz target/criterion/
```

### Transfer Results Back
```bash
# From remote system
cd ~/s3dlio-results
tar czf ../results-$(hostname)-$(date +%Y%m%d).tar.gz .

# From local machine
scp user@remote-system:~/results-*.tar.gz ./
```

## System Information to Collect

For comparison and analysis, capture:

```bash
# CPU info
lscpu > ~/s3dlio-results/00_cpu_info.txt

# Memory info
free -h > ~/s3dlio-results/00_mem_info.txt

# Storage info
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE > ~/s3dlio-results/00_storage_info.txt
df -h > ~/s3dlio-results/00_disk_usage.txt

# NVMe details (if applicable)
sudo nvme list > ~/s3dlio-results/00_nvme_info.txt 2>&1 || true
sudo smartctl -a /dev/nvme0n1 > ~/s3dlio-results/00_nvme_smart.txt 2>&1 || true
```

## Troubleshooting

### Build Fails - Missing Dependencies
```bash
# Install Rust if not present
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install build tools
sudo apt-get install build-essential pkg-config libssl-dev
```

### Direct I/O Fails
```bash
# Ensure direct-io feature is enabled
cargo build --release --features direct-io

# Check filesystem supports O_DIRECT
mount | grep $(df /tmp | tail -1 | awk '{print $1}')
```

### Performance Lower Than Expected
```bash
# Check CPU governor (should be "performance")
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Drop page cache before tests
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Check for thermal throttling
watch -n1 'cat /sys/class/thermal/thermal_zone*/temp'
```

## Documentation

See these files in the tarball:
- **FILE_BENCH.md** - Complete testing guide
- **examples/standalone/README.md** - Standalone test details
- **README.md** - s3dlio library documentation
- **docs/Changelog.md** - Version history

## Summary of Test Files

### Standalone Tests (No Dependencies)
- `examples/standalone/test_parallel_read.rs` - Multi-threaded /dev/zero reads
- `examples/standalone/test_rust_read.rs` - Rust I/O method comparison

### s3dlio Examples (Interactive)
- `examples/rust/test_direct_io_comprehensive.rs` - **RECOMMENDED START**
- `examples/rust/test_direct_io.rs` - Basic Direct I/O
- `examples/rust/test_hybrid_io_debug.rs` - Buffered vs Direct
- `examples/rust/test_aligned_buffers.rs` - Alignment impact

### Integration Tests
- `tests/test_production_performance.rs` - Sustained multi-GB/s
- `tests/test_backend_parity.rs` - Multi-cloud comparison

### Benchmarks (Criterion)
- `benches/performance_microbenchmarks.rs` - Micro-benchmarks
- `benches/rng_performance_benchmark.rs` - RNG throughput

---

**Estimated Total Test Time**: 15-30 minutes (excluding full benchmark suite)  
**Tarball Created**: $(date)  
**s3dlio Version**: 0.9.37+
