# Data Generation Guide

> **Version**: v0.9.34 (January 2026)  
> **Status**: Production-Ready with NUMA Optimization

This guide covers s3dlio's high-performance data generation capabilities with NUMA-aware optimization.

---

## Overview

s3dlio includes a high-performance random data generator optimized for benchmarking and testing storage I/O operations. As of v0.9.34, the generator automatically detects system topology (NUMA vs UMA) and intelligently allocates CPU resources to balance data generation with I/O operations.

### Key Features

- **Smart CPU Allocation**: Uses 50% of available CPUs by default, reserving the other 50% for I/O operations
- **Auto NUMA Detection**: Automatically detects and optimizes for multi-socket (NUMA) vs single-socket (UMA) systems  
- **Thread Pinning**: On NUMA systems, pins threads to specific CPU cores for better cache locality
- **Zero-Copy Architecture**: Uses `bytes::Bytes` throughout for minimal memory overhead
- **Controllable Characteristics**:
  - Deduplication factor (1:1 to N:1 logical-to-physical ratio)
  - Compression factor (1:1 incompressible to N:1 compressible)

---

## Default Behavior

### CPU Allocation

By default, s3dlio uses **50% of available logical CPUs** (cores × hyperthreads) for data generation:

```python
import s3dlio
import multiprocessing

# System has 12 logical CPUs
print(f"Total CPUs: {multiprocessing.cpu_count()}")  # 12

# s3dlio uses 6 CPUs for data gen, leaving 6 for I/O
data = s3dlio.generate_data(size=100 * 1024 * 1024)  # 100 MiB
```

This balanced approach ensures optimal performance for mixed workloads (e.g., PUT operations that combine data generation with network upload).

### NUMA Optimization

The system automatically detects your hardware topology:

- **UMA Systems** (cloud VMs, workstations): Uses standard rayon thread pool
- **NUMA Systems** (multi-socket servers): Enables thread pinning and first-touch memory initialization

No configuration required—it just works!

---

## Rust API

### Basic Usage

The simplest API uses default settings (50% CPUs, no dedup/compression):

```rust
use s3dlio::data_gen_alt::generate_controlled_data_alt;

// Generate 100 MiB of incompressible data
let data = generate_controlled_data_alt(
    100 * 1024 * 1024,  // size
    1,                   // dedup factor (1 = no dedup)
    1,                   // compress factor (1 = incompressible)
);

assert_eq!(data.len(), 100 * 1024 * 1024);
```

### Advanced Configuration

Use `GeneratorConfig` for fine-grained control:

```rust
use s3dlio::data_gen_alt::{
    generate_data_with_config,
    GeneratorConfig,
    NumaMode,
    default_data_gen_threads,
    total_cpus,
};

// Use defaults (50% CPUs, auto NUMA detection)
let config = GeneratorConfig {
    size: 100 * 1024 * 1024,
    dedup_factor: 2,     // 2:1 dedup ratio
    compress_factor: 3,  // 3:1 compression ratio
    ..Default::default()
};
let data = generate_data_with_config(config);

// Override to use all CPUs
let config_all = GeneratorConfig {
    max_threads: Some(total_cpus()),
    ..Default::default()
};

// Override to use specific thread count
let config_custom = GeneratorConfig {
    max_threads: Some(4),
    ..Default::default()
};

// Force NUMA optimizations (for testing)
let config_force = GeneratorConfig {
    numa_mode: NumaMode::Force,
    ..Default::default()
};
```

### Helper Functions

```rust
use s3dlio::data_gen_alt::{default_data_gen_threads, total_cpus};

// Get default thread count (50% of CPUs)
let default_threads = default_data_gen_threads();
println!("Default threads: {}", default_threads);

// Get total available CPUs
let total = total_cpus();
println!("Total CPUs: {}", total);
```

---

## Python API

### Basic Usage

```python
import s3dlio

# Generate incompressible data (uses 50% of CPUs)
data = s3dlio.generate_data(
    size=100 * 1024 * 1024,  # 100 MiB
    dedup=1,                  # No deduplication
    compress=1,               # Incompressible
)

print(f"Generated {len(data)} bytes")
```

### With Deduplication and Compression

```python
# Generate data with 2:1 dedup and 3:1 compression
# Logical size: 100 MiB
# Physical size: ~17 MiB (100 / 2 / 3)
data = s3dlio.generate_data(
    size=100 * 1024 * 1024,
    dedup=2,
    compress=3,
)
```

### Custom Thread Count

```python
import s3dlio

# Use all available CPUs (not recommended for I/O workloads)
data = s3dlio.generate_data_with_threads(
    size=100 * 1024 * 1024,
    threads=s3dlio.total_cpus(),
)

# Use specific thread count
data = s3dlio.generate_data_with_threads(
    size=100 * 1024 * 1024,
    threads=4,
)
```

---

## Performance Characteristics

### Throughput

Per-core performance (Xoshiro256++ RNG):

- **Incompressible data** (compress=1): 5-15 GB/s per core
- **Compressible data** (compress>1): 1-4 GB/s (depends on compression factor)

Parallel scaling:

- Near-linear scaling up to 50% of CPUs (default)
- Leaves CPU headroom for concurrent I/O operations

### NUMA Benefits

On multi-socket systems (2+ NUMA nodes):

- **Thread Pinning**: Reduces cross-node memory access by 30-50%
- **First-Touch Init**: Allocates memory on local NUMA node
- **Cache Locality**: Better L3 cache hit rates

### Memory Footprint

- **Block-based generation**: 4 MiB blocks processed in parallel
- **Streaming mode**: Minimal memory overhead (generates on-demand)
- **Zero-copy**: Uses `bytes::Bytes` for efficient memory sharing

---

## Cargo Features

NUMA optimizations are available via optional Cargo features:

```toml
[dependencies]
s3dlio = { version = "0.9.34", features = ["numa", "thread-pinning"] }
```

### Available Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `numa` | NUMA topology detection | `hwloc2 = "2.2"` |
| `thread-pinning` | CPU core affinity | `core_affinity = "0.8"` |

**Default build**: NUMA features disabled (no extra dependencies)

**Recommended for bare-metal servers**: Enable both features for optimal performance

---

## Use Cases

### Benchmarking Storage I/O

Generate test data for PUT operations:

```rust
use s3dlio::data_gen_alt::generate_controlled_data_alt;

// Generate 1 GiB of test data
let data = generate_controlled_data_alt(1024 * 1024 * 1024, 1, 1);

// Upload to S3 (data gen used 50% of CPUs, upload uses the other 50%)
s3dlio::put("s3://bucket/test-file", &data).await?;
```

### Simulating Real-World Data

Mix deduplication and compression to mimic real workloads:

```rust
// Database backup simulation (medium dedup, low compression)
let db_data = generate_controlled_data_alt(size, 4, 2);

// Log file simulation (low dedup, high compression)
let log_data = generate_controlled_data_alt(size, 2, 8);

// VM image simulation (high dedup, medium compression)
let vm_data = generate_controlled_data_alt(size, 8, 4);
```

---

## Best Practices

### For I/O Benchmarking

1. **Use default thread count** (50% of CPUs) for mixed PUT workloads
2. **Use all CPUs** only for pure data generation benchmarks
3. **Enable NUMA features** on bare-metal multi-socket servers
4. **Measure end-to-end** (data gen + I/O) for realistic results

### For Memory Efficiency

1. **Use streaming mode** for large datasets that don't fit in RAM
2. **Generate in chunks** if processing incrementally
3. **Tune block size** based on available memory

### For Reproducibility

1. **Document CPU count** used in benchmark results
2. **Note NUMA topology** (nodes, cores per node)
3. **Specify thread count** if overriding defaults
4. **Include dedup/compress factors** in test metadata

---

## Troubleshooting

### Performance Lower Than Expected

**Check CPU allocation**:
```rust
use s3dlio::data_gen_alt::{default_data_gen_threads, total_cpus};

println!("Using {} of {} CPUs", default_data_gen_threads(), total_cpus());
```

**Try using more threads**:
```rust
let config = GeneratorConfig {
    max_threads: Some(total_cpus()),
    ..Default::default()
};
```

### NUMA Optimizations Not Working

**Verify features are enabled**:
```bash
cargo build --features numa,thread-pinning
```

**Check NUMA detection**:
```bash
cargo test numa::tests::test_detect_topology -- --nocapture
```

Expected output:
```
NUMA topology: NumaTopology { num_nodes: 2, ... }
```

### Out of Memory

**Use streaming mode**:
```rust
use s3dlio::data_gen_alt::ObjectGenAlt;

let mut gen = ObjectGenAlt::new(size, dedup, compress);
let mut chunk = vec![0u8; 4 * 1024 * 1024];  // 4 MiB chunks

while !gen.is_complete() {
    let written = gen.fill_chunk(&mut chunk);
    // Process chunk...
}
```

---

## See Also

- [Performance Profiling Guide](../performance/Performance_Profiling_Guide.md)
- [ZERO-COPY-API-REFERENCE.md](ZERO-COPY-API-REFERENCE.md)
- [Changelog.md](../Changelog.md) - v0.9.34 release notes
