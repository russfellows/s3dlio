# Performance Profiling Guide for s3dlio

This guide covers the comprehensive profiling infrastructure built into s3dlio to help identify and resolve performance bottlenecks in S3 operations.

## Overview

s3dlio includes multiple profiling approaches to analyze different aspects of performance:

1. **CPU Profiling (Flamegraphs)** - Whole-program sampling to identify compute hotspots
2. **Async Task Monitoring** - Tokio runtime analysis for concurrency bottlenecks  
3. **Microbenchmarks** - Detailed analysis of inner loops and buffer operations
4. **System-level Profiling** - OS-level metrics for I/O and memory analysis

## Quick Start

### 1. Enable Profiling

Build with the profiling feature enabled:

```bash
# Basic profiling build
RUSTFLAGS="--cfg tokio_unstable" cargo build --release --features profiling

# For tokio-console support (optional)
RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling
```

### 2. Initialize Profiling in Your Code

```rust
use s3dlio::profiling::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize comprehensive profiling
    init_profiling()?;
    
    // Your S3 operations here...
    
    Ok(())
}
```

### 3. Run with Profiling

```bash
# Set profiling parameters
export S3DLIO_PROFILE_FREQ=100          # 100Hz sampling (adjust for detail vs. overhead)
export RUST_LOG="s3dlio=debug,info"     # Enable detailed logging

# Run your application
./target/release/your_app
```

## Profiling Features

### CPU Profiling with Flamegraphs

Flamegraphs show where CPU time is spent across your entire application.

#### Automatic Profiling
When you call `init_profiling()`, CPU sampling starts automatically:

```rust
use s3dlio::profiling::*;

#[tokio::main] 
async fn main() -> anyhow::Result<()> {
    init_profiling()?;  // Starts CPU sampling
    
    // Run your workload
    for uri in &large_object_uris {
        let data = s3dlio::s3_utils::get_object_uri_optimized_async(uri).await?;
        process_data(&data);
    }
    
    Ok(())
    // Profiling data collected automatically
}
```

#### Targeted Section Profiling
Profile specific operations in isolation:

```rust
use s3dlio::profiling::*;

fn analyze_upload_performance() -> anyhow::Result<()> {
    let profiler = profile_section("multipart_upload_analysis")?;
    
    // Your multipart upload code here
    for chunk in &data_chunks {
        writer.write_chunk(chunk).await?;
    }
    writer.finalize().await?;
    
    // Save flamegraph for this operation only
    profiler.save_flamegraph("target/upload_profile.svg")?;
    Ok(())
}
```

#### Instrumentation Macros
Add structured profiling to functions:

```rust
use s3dlio::{profile_span, profile_async};

#[cfg_attr(feature = "profiling", tracing::instrument(
    name = "custom.data_processing",
    skip(data),
    fields(data_size = data.len())
))]
async fn process_large_dataset(data: &[u8]) -> Result<ProcessedData> {
    let _span = profile_span!("preprocessing", size = data.len());
    
    let processed = profile_async!("compression", async {
        compress_data(data).await
    }).await?;
    
    Ok(processed)
}
```

### Async Task Monitoring with tokio-console

Monitor task scheduling, contention, and async runtime behavior:

```bash
# Terminal 1: Run your app with console support
export S3DLIO_TOKIO_CONSOLE=1
RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling

# Terminal 2: Connect tokio-console
tokio-console
```

This shows:
- Task spawn/completion rates
- Semaphore and mutex contention
- Waker churn and scheduling delays
- Resource utilization per task

### Microbenchmarks

Run focused benchmarks on core operations:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench --bench s3_microbenchmarks multipart_buffer_ops

# Generate HTML reports
cargo bench -- --output-format html
```

Benchmark categories:
- **Buffer Operations**: Vector allocation, copying, splitting
- **Data Generation**: Synthetic data creation performance
- **URI Parsing**: S3 URI parsing overhead
- **Vector Operations**: Common data manipulation patterns

## Environment Variables for Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_PROFILE_FREQ` | `100` | CPU sampling frequency (Hz). Higher = more detail, more overhead |
| `S3DLIO_TOKIO_CONSOLE` | unset | Set to `1` to enable tokio-console integration |
| `S3DLIO_SAVE_INDIVIDUAL_PROFILES` | unset | Set to save flamegraph for each profiled section |
| `RUST_LOG` | unset | Set to `s3dlio=debug,info` for detailed tracing output |

## Performance Analysis Workflow

### 1. Establish Baseline

```bash
# Measure current performance
time your_s3_workload

# Run microbenchmarks for baseline
cargo bench
```

### 2. Identify Hotspots

```bash
# Profile CPU usage
RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling
# Check flamegraph output for time distribution
```

Look for:
- High CPU time in specific functions
- Excessive time in memory operations (memcpy, allocation)
- Time spent in AWS SDK internals vs. your code

### 3. Analyze Async Behavior

```bash
# Monitor async runtime
S3DLIO_TOKIO_CONSOLE=1 cargo run --release --features profiling &
tokio-console
```

Look for:
- Tasks blocked on semaphores/mutexes
- High waker churn indicating contention
- Imbalanced task distribution

### 4. Deep Dive with System Tools

```bash
# System-level flamegraph (includes OS/kernel time)
cargo flamegraph --release --features profiling

# Performance counters
perf stat -e cache-misses,cache-references,page-faults,context-switches your_program
```

### 5. Optimize and Validate

After making changes:
```bash
# Re-run benchmarks to measure improvement
cargo bench

# Profile again to verify hotspot resolution
RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling
```

## Real-World Examples

### Example 1: Optimizing Large Object Downloads

```rust
use s3dlio::profiling::*;

async fn download_large_objects(uris: &[String]) -> Result<()> {
    init_profiling()?;
    
    let profiler = profile_section("large_object_downloads")?;
    
    for uri in uris {
        let _span = profile_span!("single_download", uri = %uri);
        
        // This will automatically use concurrent range GET for large objects
        let data = s3dlio::s3_utils::get_object_uri_optimized_async(uri).await?;
        
        // Profile data processing separately
        let _process_span = profile_span!("data_processing", size = data.len());
        process_downloaded_data(&data).await?;
    }
    
    profiler.save_flamegraph("target/download_profile.svg")?;
    Ok(())
}
```

**What to look for in flamegraph:**
- Time in `get_object_concurrent_range_async` vs. `get_object`  
- Time in HTTP client operations vs. data copying
- Balance between network I/O and CPU processing

### Example 2: Profiling Multipart Uploads

```rust
use s3dlio::{profiling::*, multipart::*};

async fn upload_with_profiling(data: &[u8], uri: &str) -> Result<()> {
    init_profiling()?;
    
    let upload_profiler = profile_section("multipart_upload")?;
    
    let config = MultipartUploadConfig {
        part_size: 8 * 1024 * 1024,  // 8MB parts
        max_in_flight: 10,
        abort_on_drop: true,
        content_type: None,
    };
    
    let mut writer = MultipartUploadSink::from_uri_async(uri, config).await?;
    
    // Profile chunking and upload separately
    for chunk in data.chunks(config.part_size) {
        let _chunk_span = profile_span!("write_chunk", size = chunk.len());
        writer.write(chunk.to_vec()).await?;
    }
    
    let _finalize_span = profile_span!("finalize_upload");
    writer.finish().await?;
    
    upload_profiler.save_flamegraph("target/upload_profile.svg")?;
    Ok(())
}
```

**What to look for:**
- Time in `write` vs. `spawn_part` (buffering vs. uploading)
- Concurrency utilization in `concurrent_range_get_impl`
- Memory allocation patterns during chunking

## Profiling Overhead

| Profiling Type | Overhead | Use Case |
|----------------|----------|----------|
| Sampling (100Hz) | ~1-2% | Production-safe general profiling |
| Sampling (1000Hz) | ~5-10% | Detailed development analysis |
| Tracing spans | <1% | Structured logging with minimal overhead |
| tokio-console | ~2-5% | Async runtime analysis |
| Microbenchmarks | N/A | Isolated component testing |

## Common Performance Bottlenecks

Based on flamegraph analysis, common bottlenecks include:

### 1. Memory Operations
```
Symptom: High time in memcpy, Vec::extend_from_slice
Solution: Use pre-allocated buffers, reduce copying
```

### 2. HTTP Connection Pool Exhaustion
```
Symptom: Tasks blocked on semaphore acquire
Solution: Increase S3DLIO_HTTP_MAX_CONNECTIONS
```

### 3. Suboptimal Chunk Sizes
```
Symptom: Many small HTTP requests or excessive memory usage
Solution: Tune S3DLIO_RANGE_CHUNK_SIZE_MB
```

### 4. Runtime Thread Starvation
```
Symptom: High task scheduler overhead in tokio-console
Solution: Increase S3DLIO_RUNTIME_THREADS
```

## Troubleshooting

### Profiling Not Working
```bash
# Ensure feature is enabled
cargo build --release --features profiling

# Check for required RUSTFLAGS
RUSTFLAGS="--cfg tokio_unstable" cargo build --release --features profiling
```

### No Flamegraph Output
```bash
# Ensure profiling is initialized
init_profiling()?;

# Check environment variables
export S3DLIO_PROFILE_FREQ=100
```

### tokio-console Not Connecting
```bash
# Ensure both flags are set
export S3DLIO_TOKIO_CONSOLE=1
RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling
```

## Integration with External Tools

### cargo flamegraph
```bash
# Install if needed
cargo install flamegraph

# Generate system-wide flamegraph
cargo flamegraph --release --features profiling -- your_args
```

### perf integration
```bash
# Detailed system analysis
perf record --call-graph dwarf ./target/release/your_app
perf report

# Cache analysis
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./target/release/your_app
```

This profiling infrastructure provides comprehensive insights into s3dlio performance characteristics, enabling data-driven optimization decisions.
