# s3dlio Environment Variables Reference

This document provides a comprehensive reference for all environment variables supported by s3dlio v0.7.5+.

## HTTP Client & Performance Optimization

### Redirect Following (for AIStore and other S3-compatible services that use 3xx redirects)
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_FOLLOW_REDIRECTS` | `0` | Enable 3xx redirect following (`1`, `true`, `yes`, `on`, or `enable`). Required for NVIDIA AIStore (HTTP 307 routing). Disabled by default because the AWS SDK intentionally does not follow cross-host redirects. |
| `S3DLIO_REDIRECT_MAX` | `5` | Maximum redirect hops to follow per request. Set to a lower value to detect redirect loops sooner. |

### HTTP Client Control
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_H2C` | `0` (HTTP/1.1) | HTTP/2 cleartext (h2c) mode for `http://` endpoints. **Default (unset or `0`)** = always HTTP/1.1 — benchmarking showed HTTP/2 reduces throughput on plain-HTTP endpoints vs HTTP/1.1 with an unlimited connection pool (changed from auto-probe in v0.9.92). **`1`** (or `true`, `yes`, `on`, `enable`) = force h2c prior-knowledge, no fallback — use for storage systems that require HTTP/2 on their `http://` API endpoint. **`auto`** = probe h2c on the first plain-HTTP connection and fall back to HTTP/1.1 if the server rejects it (pre-v0.9.92 behaviour). Has **no effect** on `https://` connections — those negotiate HTTP/2 automatically via TLS ALPN. |
| `S3DLIO_POOL_MAX_IDLE_PER_HOST` | unlimited | Maximum idle connections kept in the reqwest connection pool per host. Default changed to unlimited in v0.9.92: previously 32, which caused TCP connection churn at concurrency levels above 32 (each worker paid a full handshake penalty when the pool was full). Idle connections are still evicted after `S3DLIO_POOL_IDLE_TIMEOUT_SECS` seconds. Set to a positive integer to impose a hard ceiling. |
| `S3DLIO_POOL_IDLE_TIMEOUT_SECS` | `90` | Seconds before an idle pooled connection is closed. |
| `S3DLIO_H2_ADAPTIVE_WINDOW` | `1` (enabled) | HTTP/2 flow-control window mode. **`1`** (or `true`, `yes`, `on`) = adaptive (BDP estimator): hyper measures bandwidth-delay product via H2 PINGs and auto-tunes the window from 64 KB up to hundreds of MiB. Best for most workloads. **`0`** = static windows controlled by `S3DLIO_H2_STREAM_WINDOW_MB` / `S3DLIO_H2_CONN_WINDOW_MB`. Only active when `S3DLIO_H2C=1`. |
| `S3DLIO_H2_STREAM_WINDOW_MB` | `4` | HTTP/2 per-stream flow-control window in MiB (static mode only, i.e. `S3DLIO_H2_ADAPTIVE_WINDOW=0`). Clamped to 256 MiB maximum. |
| `S3DLIO_H2_CONN_WINDOW_MB` | `4×stream` | HTTP/2 connection-level flow-control window in MiB (static mode only). Defaults to 4× `S3DLIO_H2_STREAM_WINDOW_MB`, capped at 256 MiB. |

### HTTP/2 on TLS endpoints (`https://`)

No configuration is needed.  s3dlio's reqwest client (rustls + aws-lc-rs) advertises
`["h2", "http/1.1"]` in every TLS ClientHello.  If the server selects `h2`, HTTP/2 is
used automatically; otherwise HTTP/1.1 is used.  The negotiated protocol is reported in
startup INFO logs and in the PUT summary line (`protocol=HTTP/2` or `protocol=HTTP/1.1`).

### HTTP/2 on cleartext endpoints (`http://`)

```bash
# Default: HTTP/1.1 (S3DLIO_H2C unset or 0 — changed from auto-probe in v0.9.92)
AWS_ENDPOINT_URL=http://storage-host:9000 s3-cli stat s3://bucket/key

# Force h2c (for systems that require HTTP/2 cleartext)
S3DLIO_H2C=1 AWS_ENDPOINT_URL=http://storage-host:9000 s3-cli put s3://bucket/prefix -n 100

# Force HTTP/1.1 explicitly (same as default)
S3DLIO_H2C=0 AWS_ENDPOINT_URL=http://storage-host:9000 s3-cli ls s3://bucket/

# Re-enable pre-v0.9.92 auto-probe behaviour (probe h2c, fall back to HTTP/1.1)
S3DLIO_H2C=auto AWS_ENDPOINT_URL=http://storage-host:9000 s3-cli put s3://bucket/prefix -n 100
```

### HTTP/2 flow-control window tuning

Applies only when `S3DLIO_H2C=1` (cleartext HTTP/2).

#### What the three knobs actually do

Every HTTP/2 connection carries two flow-control windows at all times:

**Per-stream window** (`S3DLIO_H2_STREAM_WINDOW_MB`)
Controls how much DATA the remote peer may send on a single stream before it must stop and wait for the client to issue a `WINDOW_UPDATE` frame.  The H2 spec default is **65,535 bytes (~64 KB)** — far too small for high-throughput storage I/O.  For GET operations this window directly caps download throughput: if the server exhausts the window mid-object it stalls until we acknowledge receipt.

**Connection-level window** (`S3DLIO_H2_CONN_WINDOW_MB`)
The aggregate receive budget across *all* concurrent streams on one TCP connection.  Must be ≥ the per-stream window to be useful.  Setting a large stream window but a small connection window makes the connection window the bottleneck instead (e.g. stream = 16 MiB but conn = 4 MiB → the connection stalls after 4 MiB regardless of how many streams are open).  H2 spec default is also 65,535 bytes.

**Adaptive BDP estimator** (`S3DLIO_H2_ADAPTIVE_WINDOW`)
When enabled (the default), hyper sends periodic H2 `PING` frames, measures the round-trip time, and computes the Bandwidth-Delay Product:

```
BDP = throughput × RTT
```

It then proactively issues `WINDOW_UPDATE` frames to keep the window ≥ BDP at all times.  This eliminates stalls on any network without manual tuning.  **When adaptive mode is ON, both static size variables (`S3DLIO_H2_STREAM_WINDOW_MB` and `S3DLIO_H2_CONN_WINDOW_MB`) are completely ignored — reqwest/hyper overrides them.**

#### GET vs PUT behaviour

- **GETs**: the window governs how much response data the server may send before the client acknowledges.  A 64 KB default window on a 100 Gb/s LAN limits a single stream to ~64 KB per RTT (~64 KB ÷ 0.1 ms = ~640 MB/s maximum per stream before stalling).  Adaptive mode eliminates this cap automatically.
- **PUTs**: the server's *own* advertised window (not these settings) controls how much of the request body we can send.  These client-side variables affect only how quickly we receive the `200 OK` response body (~200 bytes), so they have negligible impact on upload throughput.  Nevertheless, a large window still avoids any framing stalls in bidirectional scenarios.

#### When to use static mode

Use `S3DLIO_H2_ADAPTIVE_WINDOW=0` when you need:
- **Reproducible benchmarks** — adaptive mode changes window sizes mid-run; static windows keep the protocol behaviour constant across runs.
- **Known network characteristics** — if BDP is measured and fixed (e.g. 10 GbE at 0.1 ms RTT ≈ 125 KB), you can set windows exactly.
- **Adaptive-mode overhead avoidance** — each PING adds a small round-trip; on very-low-latency (<< 0.1 ms) local connections the overhead is sometimes visible.

For all other workloads, leave adaptive ON (the default).

```bash
# Default: adaptive BDP estimator (recommended for most workloads)
S3DLIO_H2C=1 s3-cli get s3://bucket/large-dataset/ -o /data/

# Static windows — useful for reproducible benchmarks
S3DLIO_H2C=1 S3DLIO_H2_ADAPTIVE_WINDOW=0 \
  S3DLIO_H2_STREAM_WINDOW_MB=16 \
  S3DLIO_H2_CONN_WINDOW_MB=64 \
  s3-cli get s3://bucket/large-dataset/ -o /data/

# High-throughput static config (many large objects, low-latency network)
S3DLIO_H2C=1 S3DLIO_H2_ADAPTIVE_WINDOW=0 \
  S3DLIO_H2_STREAM_WINDOW_MB=64 \
  s3-cli put s3://bucket/prefix/ -n 500
```

### Tokio Runtime Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_RT_THREADS` | `max(8, cores*2)` | Number of Tokio runtime worker threads (capped at 32) |

## Range GET Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_ENABLE_RANGE_OPTIMIZATION` | `true` (v0.9.60+) | Parallel range downloads for large S3 objects. Set to `0` to disable (e.g., for small-object or memory-constrained workloads) |
| `S3DLIO_RANGE_THRESHOLD_MB` | `32` (v0.9.60+) | Minimum object size (MB) to trigger range GET optimization |
| `S3DLIO_RANGE_CONCURRENCY` | Auto-tuned | Number of concurrent range requests for large objects |
| `S3DLIO_CHUNK_SIZE` | Auto-calculated | Chunk size for range requests (default: 1-8 MB based on object size) |
| `S3DLIO_CONCURRENT_THRESHOLD` | Auto-tuned | Threshold for enabling concurrent operations |

### Range Optimization Details

**Enabled by default (v0.9.60+)**
- Range optimization is on by default; requires a HEAD request to determine object size (~10-20ms overhead)
- Objects below the threshold use a single GET request (no HEAD)
- Default threshold is 32 MB — covers most AI/ML dataset file sizes

**When to disable** (small-object or memory-constrained workloads):
```bash
# Disable range optimization (e.g., many small objects < 32 MB)
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0

# Raise the threshold instead of disabling entirely
export S3DLIO_RANGE_THRESHOLD_MB=128  # Only for very large objects

# For very large objects, aggressive parallel settings
export S3DLIO_RANGE_THRESHOLD_MB=32
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
| 64 MB | 3.14s | 755 MB/s (0.74 GB/s) | 1.76x (76% faster) 🏆 |
| 128 MB | 3.22s | 735 MB/s (0.72 GB/s) | 1.71x (71% faster) |

**Key findings:**
- **32 MB default threshold** is a good balance for typical AI/ML objects (covers most dataset files)
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
| `AWS_CA_BUNDLE` | Path to custom CA certificate bundle (standard AWS SDK name) |

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
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0  # Disable range optimization (on by default since v0.9.60)

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
- Range optimization is **enabled by default** (v0.9.60+); no action needed
- Lower `S3DLIO_RANGE_THRESHOLD_MB` if needed (default is 32 MB)
- Increase `S3DLIO_RANGE_CONCURRENCY` to 32-64
- Use `S3DLIO_USE_OPTIMIZED_HTTP=true`
- Set `S3DLIO_MAX_HTTP_CONNECTIONS=400`
- Consider `S3DLIO_RT_THREADS=32` on high-core systems

### For Many Small Objects
- **Disable range optimization**: `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` (enabled by default since v0.9.60)
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

- **v0.9.60**: Range optimization enabled by default; default threshold changed 64 MB → 32 MB; set `=0` to disable
- **v0.7.5**: Added HTTP client optimization variables
- **v0.7.4**: Added runtime threading control
- **v0.7.0+**: Range GET optimization variables
- **v0.6.0+**: Operation logging variables

## See Also

- [Phase1 GET Optimization Plan](Phase1_GET_Optimization_Plan.md) - Implementation details
- [Performance Optimization Summary](Performance_Optimization_Summary.md) - Performance analysis
- [README.md](../README.md) - Main project documentation
