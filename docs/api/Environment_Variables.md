# s3dlio Environment Variables Reference

This document provides a comprehensive reference for all environment variables supported by s3dlio v0.7.5+.

## HTTP Client & Performance Optimization

### HTTP Client Control
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_USE_OPTIMIZED_HTTP` | `false` | Enable optimized HTTP client with enhanced connection pooling |
| `S3DLIO_MAX_HTTP_CONNECTIONS` | `200` | Maximum connections per host (when optimization enabled) |
| `S3DLIO_HTTP_IDLE_TIMEOUT_MS` | `800` | Connection idle timeout in milliseconds |
| `S3DLIO_OPERATION_TIMEOUT_SECS` | `120` | Total operation timeout in seconds |

### Tokio Runtime Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_RT_THREADS` | `max(8, cores*2)` | Number of Tokio runtime worker threads (capped at 32) |

## Range GET Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_ENABLE_RANGE_OPTIMIZATION` | `false` | Enable parallel range downloads for large S3 objects (opt-in to avoid HEAD overhead) |
| `S3DLIO_RANGE_THRESHOLD_MB` | `64` | Minimum object size (MB) to trigger range GET optimization (only when enabled) |
| `S3DLIO_RANGE_CONCURRENCY` | Auto-tuned | Number of concurrent range requests for large objects |
| `S3DLIO_CHUNK_SIZE` | Auto-calculated | Chunk size for range requests (default: 1-8 MB based on object size) |
| `S3DLIO_CONCURRENT_THRESHOLD` | Auto-tuned | Threshold for enabling concurrent operations |

### Range Optimization Details

**Why disabled by default?**
- Requires HEAD request to determine object size (adds ~10-20ms latency)
- Small objects (< 64 MB) faster with single GET request
- Best for workloads with large, known-size objects (> 100 MB)

**When to enable:**
```bash
# Enable for large object workloads (> 100 MB objects)
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=1
export S3DLIO_RANGE_THRESHOLD_MB=64  # Conservative default

# For very large objects (> 500 MB), use aggressive settings
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=1
export S3DLIO_RANGE_THRESHOLD_MB=128
export S3DLIO_RANGE_CONCURRENCY=32
export S3DLIO_CHUNK_SIZE=16777216  # 16 MB chunks
```

**Performance impact (measured with 148 MB objects):**
- Objects < threshold: Single GET request (fast path, no HEAD)
- Objects ≥ threshold: HEAD + parallel range GETs (58-76% faster for large objects)

**Actual benchmark results (16x 148 MB objects, MinIO):**

| Threshold | Time | Throughput | Speedup |
|-----------|------|------------|---------|
| Disabled (baseline) | 5.52s | 429 MB/s (0.42 GB/s) | 1.00x |
| 8 MB | 3.50s | 676 MB/s (0.66 GB/s) | 1.58x (58% faster) |
| 16 MB | 3.27s | 725 MB/s (0.71 GB/s) | 1.69x (69% faster) |
| 32 MB | 3.23s | 732 MB/s (0.71 GB/s) | 1.71x (71% faster) |
| **64 MB (default)** | **3.14s** | **755 MB/s (0.74 GB/s)** | **1.76x (76% faster)** 🏆 |
| 128 MB | 3.22s | 735 MB/s (0.72 GB/s) | 1.71x (71% faster) |

**Key findings:**
- **64 MB threshold (default) is optimal** for 148 MB objects
- 16-64 MB range provides excellent performance (69-76% faster)
- Even aggressive 8 MB threshold shows 58% improvement
- HEAD overhead (~10-20ms) is well amortized by parallel download

## S3 Operation Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_OPLOG_LOSSLESS` | `false` | Enable lossless operation logging (higher memory usage) |
| `S3DLIO_OPLOG_BUF` | `8192` | Operation log buffer capacity |
| `S3DLIO_OPLOG_WBUFCAP` | `4096` | Write buffer capacity for operation logging |
| `S3DLIO_OPLOG_LEVEL` | `1` | Operation logging level (0=off, 1=basic, 2=verbose) |

## AWS Configuration

Standard AWS environment variables are also supported:

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_SESSION_TOKEN` | AWS session token (for temporary credentials) |
| `AWS_REGION` | AWS region (e.g., `us-east-1`) |
| `AWS_ENDPOINT_URL` | Custom S3 endpoint URL (for MinIO or other S3-compatible storage) |
| `AWS_CA_BUNDLE_PATH` | Path to custom CA certificate bundle |

## Azure Blob Storage Configuration

| Variable | Description |
|----------|-------------|
| `AZURE_STORAGE_ACCOUNT` | Azure storage account name |
| `AZURE_STORAGE_KEY` | Azure storage account key |
| `AZURE_CLIENT_ID` | Azure AD client ID (for service principal auth) |
| `AZURE_CLIENT_SECRET` | Azure AD client secret |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_STORAGE_ENDPOINT` | **Custom Azure endpoint** (e.g., `http://localhost:10000` for Azurite) |
| `AZURE_BLOB_ENDPOINT_URL` | Alternative for `AZURE_STORAGE_ENDPOINT` |

### Azure Custom Endpoint Examples

```bash
# Azurite (local emulator)
export AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000
sai3-bench util ls az://devstoreaccount1/testcontainer/

# Multi-protocol proxy
export AZURE_STORAGE_ENDPOINT=http://localhost:9001
sai3-bench util ls az://myaccount/mycontainer/
```

## Google Cloud Storage Configuration

| Variable | Description |
|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCS service account JSON key file |
| `GCS_ENDPOINT_URL` | **Custom GCS endpoint** (e.g., `http://localhost:4443` for fake-gcs-server) |
| `STORAGE_EMULATOR_HOST` | GCS emulator convention (`host:port`), `http://` prepended if missing |

### GCS Custom Endpoint Examples

```bash
# fake-gcs-server (local emulator)
export GCS_ENDPOINT_URL=http://localhost:4443
sai3-bench util ls gs://testbucket/

# Using STORAGE_EMULATOR_HOST convention
export STORAGE_EMULATOR_HOST=localhost:4443
sai3-bench util ls gs://testbucket/

# Multi-protocol proxy
export GCS_ENDPOINT_URL=http://localhost:9002
sai3-bench util ls gs://testbucket/
```

## Usage Examples

### High-Performance Configuration
```bash
# Optimize for high-throughput workloads
export S3DLIO_USE_OPTIMIZED_HTTP=true
export S3DLIO_MAX_HTTP_CONNECTIONS=400
export S3DLIO_HTTP_IDLE_TIMEOUT_MS=2000
export S3DLIO_RT_THREADS=32
export S3DLIO_RANGE_CONCURRENCY=64

./target/release/s3-cli get s3://bucket/large-dataset/
```

### Memory-Constrained Configuration
```bash
# Optimize for lower memory usage
export S3DLIO_RT_THREADS=8
export S3DLIO_MAX_HTTP_CONNECTIONS=50
export S3DLIO_RANGE_CONCURRENCY=8
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0  # Disable range optimization

./target/release/s3-cli get s3://bucket/files/
```

### Debugging Configuration
```bash
# Enable detailed operation logging
export S3DLIO_OPLOG_LEVEL=2
export S3DLIO_OPLOG_LOSSLESS=true
export S3DLIO_OPLOG_BUF=16384

./target/release/s3-cli --op-log operations.tsv.zst get s3://bucket/test/
```

### Conservative Configuration
```bash
# Use AWS SDK defaults with minimal optimization
export S3DLIO_USE_OPTIMIZED_HTTP=false
export S3DLIO_RT_THREADS=8

./target/release/s3-cli get s3://bucket/files/
```

## Performance Tuning Guidelines

### For Large Objects (>100MB)
- **Enable range optimization**: `S3DLIO_ENABLE_RANGE_OPTIMIZATION=1`
- Set `S3DLIO_RANGE_THRESHOLD_MB=64` (or higher for very large objects)
- Increase `S3DLIO_RANGE_CONCURRENCY` to 32-64
- Use `S3DLIO_USE_OPTIMIZED_HTTP=true`
- Set `S3DLIO_MAX_HTTP_CONNECTIONS=400`
- Consider `S3DLIO_RT_THREADS=32` on high-core systems

### For Many Small Objects
- **Disable range optimization**: `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` (default)
- Use `S3DLIO_USE_OPTIMIZED_HTTP=true` 
- Set `S3DLIO_MAX_HTTP_CONNECTIONS=200-400`
- Increase `S3DLIO_RT_THREADS` for better parallelism

### For Bandwidth-Limited Networks
- Reduce `S3DLIO_RANGE_CONCURRENCY` to 8-16
- Increase `S3DLIO_HTTP_IDLE_TIMEOUT_MS` to 2000-5000
- Use moderate `S3DLIO_MAX_HTTP_CONNECTIONS=100`

### For High-Latency Networks
- Increase `S3DLIO_OPERATION_TIMEOUT_SECS` to 300-600
- Use `S3DLIO_HTTP_IDLE_TIMEOUT_MS=5000` or higher
- Enable `S3DLIO_USE_OPTIMIZED_HTTP=true` for better connection reuse

## Version History

- **v0.7.5**: Added HTTP client optimization variables
- **v0.7.4**: Added runtime threading control
- **v0.7.0+**: Range GET optimization variables
- **v0.6.0+**: Operation logging variables

## See Also

- [Phase1 GET Optimization Plan](Phase1_GET_Optimization_Plan.md) - Implementation details
- [Performance Optimization Summary](Performance_Optimization_Summary.md) - Performance analysis
- [README.md](../README.md) - Main project documentation
