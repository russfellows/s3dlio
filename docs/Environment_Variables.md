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
| `S3DLIO_RANGE_CONCURRENCY` | Auto-tuned | Number of concurrent range requests for large objects |
| `S3DLIO_RANGE_THRESHOLD_MB` | `32` | Minimum object size (MB) to trigger range GET optimization |
| `S3DLIO_CHUNK_SIZE` | Auto-calculated | Chunk size for range requests |
| `S3DLIO_CONCURRENT_THRESHOLD` | Auto-tuned | Threshold for enabling concurrent operations |

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
| `AWS_ENDPOINT_URL` | Custom S3 endpoint URL |
| `AWS_CA_BUNDLE_PATH` | Path to custom CA certificate bundle |

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
export S3DLIO_RANGE_THRESHOLD_MB=64

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
- Increase `S3DLIO_RANGE_CONCURRENCY` to 32-64
- Use `S3DLIO_USE_OPTIMIZED_HTTP=true`
- Set `S3DLIO_MAX_HTTP_CONNECTIONS=400`
- Consider `S3DLIO_RT_THREADS=32` on high-core systems

### For Many Small Objects
- Use `S3DLIO_USE_OPTIMIZED_HTTP=true` 
- Set `S3DLIO_MAX_HTTP_CONNECTIONS=200-400`
- Increase `S3DLIO_RT_THREADS` for better parallelism
- Set `S3DLIO_RANGE_THRESHOLD_MB=64` to avoid unnecessary range requests

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
