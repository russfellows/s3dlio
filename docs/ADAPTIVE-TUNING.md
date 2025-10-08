# Adaptive Tuning Guide (v0.9.0+)

## Overview

s3dlio v0.9.0 introduces **optional** adaptive tuning that automatically optimizes performance parameters based on workload characteristics. Adaptive tuning is:

- **OPTIONAL**: Disabled by default
- **OPT-IN**: Users explicitly enable via API
- **OVERRIDABLE**: Explicit settings always take precedence

## Key Principle

**Explicit settings ALWAYS override adaptive behavior.**

If you set `part_size`, `buffer_size`, or `num_workers` explicitly, those values are used regardless of adaptive configuration. Adaptive tuning only applies when these parameters are left unspecified.

## Quick Start

### Rust API

```rust
use s3dlio::api::{WriterOptions, LoaderOptions};

// Enable adaptive tuning for uploads
let opts = WriterOptions::new().with_adaptive();
let writer = store.create_writer("file.dat", opts).await?;

// Enable adaptive tuning for data loading
let loader_opts = LoaderOptions::default().with_adaptive();
let dataset = S3BytesDataset::from_prefix_with_opts("s3://bucket/", &loader_opts)?;
```

### Python API

```python
import s3dlio

# Enable adaptive tuning for uploads
opts = s3dlio.WriterOptions()
opts.enable_adaptive()
writer = store.create_writer("file.dat", opts)

# Enable adaptive tuning for data loading
loader_opts = s3dlio.LoaderOptions()
loader_opts.enable_adaptive()
dataset = s3dlio.S3BytesDataset.from_prefix("s3://bucket/", loader_opts)
```

## What Gets Tuned?

### 1. Part Size (Upload Operations)

Adaptive part sizing based on file size:

| File Size    | Adaptive Part Size | Rationale                           |
|--------------|-------------------|-------------------------------------|
| < 16 MB      | 8 MB              | Avoid unnecessary chunking          |
| 16-256 MB    | 16 MB (default)   | Proven optimal for most workloads   |
| > 256 MB     | 32 MB             | Better throughput for large files   |

**Override Example**:
```rust
// Adaptive would choose 32 MB for 1 GB file, but explicit 20 MB is used
let opts = WriterOptions::new()
    .with_adaptive()
    .with_part_size(20 * 1024 * 1024); // Explicit 20 MB overrides adaptive
```

### 2. Concurrency (Data Loading)

Adaptive concurrency based on workload type:

| Workload Type | Concurrency Formula | Example (8 CPUs) |
|---------------|---------------------|------------------|
| Small Files   | 8x CPU count        | 64 workers       |
| Medium Files  | 4x CPU count        | 32 workers       |
| Large Files   | 2x CPU count        | 16 workers       |
| Batch Ops     | 8x CPU count        | 64 workers       |

**Override Example**:
```rust
// Adaptive would choose 64 for small files, but explicit 16 is used
let opts = LoaderOptions::default()
    .with_adaptive()
    .num_workers(16); // Explicit 16 overrides adaptive
```

### 3. Buffer Size (Streaming Operations)

Adaptive buffer sizing based on operation type:

- **Upload**: 2 MB buffers
- **Download**: 4 MB buffers (read-ahead optimization)
- **Other**: 1 MB default

## Custom Adaptive Configuration

You can customize adaptive bounds and behavior:

```rust
use s3dlio::api::{AdaptiveConfig, WriterOptions};

let custom_adaptive = AdaptiveConfig::enabled()
    .with_part_size_bounds(10 * 1024 * 1024, 40 * 1024 * 1024) // 10-40 MB
    .with_concurrency_bounds(4, 32); // 4-32 workers

let opts = WriterOptions::new().with_adaptive_config(custom_adaptive);
```

## Workload Type Hints

You can provide workload hints to guide adaptive behavior:

```rust
use s3dlio::api::{AdaptiveConfig, WorkloadType};

let adaptive = AdaptiveConfig::enabled()
    .with_workload_type(WorkloadType::LargeFile);
```

Available workload types:
- `SmallFile`: < 16 MB, low latency priority
- `MediumFile`: 16-256 MB, balanced performance
- `LargeFile`: > 256 MB, throughput priority
- `Batch`: Many small objects, high concurrency
- `Unknown`: Mixed or unknown workload (uses defaults)

## When to Use Adaptive Tuning

### ✅ Good Use Cases

- **Prototyping**: Quick development without tuning overhead
- **Mixed workloads**: Varying file sizes and patterns
- **General-purpose applications**: Default optimization is sufficient
- **Non-critical pipelines**: Convenience over absolute control

### ❌ When NOT to Use

- **Production ML training**: Explicit settings for reproducibility
- **Performance-critical paths**: Manual tuning for maximum throughput
- **Benchmarking**: Consistent parameters for fair comparison
- **Compliance requirements**: Predictable resource usage

## Performance Impact

Adaptive tuning adds **minimal overhead** (microseconds):
- Parameter computation happens once at configuration time
- No runtime overhead during actual I/O operations
- Same code paths used whether adaptive is enabled or not

## Migration from Explicit Settings

Adaptive tuning is **fully backward compatible**. Existing code with explicit settings continues to work identically:

```rust
// Existing code (pre-v0.9.0) - still works perfectly
let opts = WriterOptions::new().with_part_size(16 * 1024 * 1024);

// New code (v0.9.0+) - opt-in to adaptive
let opts = WriterOptions::new().with_adaptive();

// Hybrid approach - explicit overrides adaptive when needed
let opts = WriterOptions::new()
    .with_adaptive() // Use adaptive by default
    .with_part_size(custom_size); // Override for specific case
```

## Testing Recommendations

When testing adaptive tuning:

1. **Verify override behavior**: Confirm explicit settings are always used
2. **Test boundary conditions**: Small/large files, low/high concurrency
3. **Compare performance**: Benchmark adaptive vs manual tuning for your workload
4. **Document decisions**: Record why adaptive is enabled/disabled

## Debugging Adaptive Behavior

To see what adaptive tuning selected:

```rust
let opts = WriterOptions::new().with_adaptive();
let effective_part_size = opts.effective_part_size(Some(file_size));
println!("Adaptive chose part size: {} MB", effective_part_size / (1024 * 1024));
```

## Example: Complete Workflow

```rust
use s3dlio::api::{store_for_uri, AdaptiveConfig, WriterOptions};

async fn upload_with_adaptive(uri: &str, data: &[u8]) -> anyhow::Result<()> {
    let store = store_for_uri(uri)?;
    
    // Enable adaptive with custom bounds
    let adaptive = AdaptiveConfig::enabled()
        .with_part_size_bounds(8 * 1024 * 1024, 32 * 1024 * 1024);
    
    let opts = WriterOptions::new().with_adaptive_config(adaptive);
    
    // Adaptive will choose optimal part size based on data.len()
    let mut writer = store.create_writer("output.dat", opts).await?;
    writer.write_chunk(data).await?;
    writer.finalize().await?;
    
    Ok(())
}
```

## Advanced: Direct AdaptiveParams Usage

For advanced use cases, you can compute parameters directly:

```rust
use s3dlio::api::{AdaptiveConfig, AdaptiveParams, WorkloadType};

let config = AdaptiveConfig::enabled();
let params = AdaptiveParams::new(config);

// Compute optimal part size for specific file
let part_size = params.compute_part_size(Some(500 * 1024 * 1024), None);

// Compute optimal concurrency for workload
let concurrency = params.compute_concurrency(Some(WorkloadType::Batch), None);

// Compute optimal buffer size
let buffer = params.compute_buffer_size("upload", None);
```

## Summary

- **Default**: Adaptive tuning is **DISABLED**
- **Opt-in**: Use `.with_adaptive()` to enable
- **Override**: Explicit settings **ALWAYS** take precedence
- **Control**: You maintain full control over I/O behavior
- **Convenience**: Adaptive picks sensible defaults when you don't specify

For most users, adaptive tuning provides a good balance of convenience and performance. For critical workloads, explicit configuration remains the recommended approach.
