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
| `S3DLIO_RT_THREADS` | `max(4, cores)`, scaled by concurrency hint up to `cores * 4` | Number of Tokio runtime worker threads.  May be further bounded by `configure_tokio_threads()` for MPI-aware per-process budgets.  Note: the historical "capped at 32" docstring referred to the pre-v0.9.92 ceiling that was removed when the auto-scaling formula was rewritten. |

### Connection Timeouts and Retries
| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_CONNECT_TIMEOUT_SECS` | `20` | TCP connect timeout in seconds.  Covers only the SYN → SYN-ACK handshake.  Honored by both the reqwest transport and the AWS SDK `TimeoutConfig` layer (unified in v0.9.102).  Default bumped from 10 s to 20 s in v0.9.102 after mlcommons/storage#506 showed the previous 5 s SDK-layer ceiling was the trigger for cold-start dispatch failures.  Raise further (e.g. `30` or `60`) for extreme fan-out scenarios where the endpoint's TCP accept queue is briefly saturated by thousands of concurrent connects. |
| `S3DLIO_OPERATION_TIMEOUT_SECS` | `60` | Full request/response cycle timeout in seconds (excluding the connect handshake).  60 s is sufficient for ~6 GB at 100 MB/s and ~60 GB at 1 GB/s.  Raise for very large single objects or slow networks. |
| `S3DLIO_MAX_RETRY_ATTEMPTS` | `3` | Maximum number of attempts (1 initial + N−1 retries) the AWS SDK makes per operation before propagating the error.  Matches the SDK's own default.  Set to `1` for **fast-fail at warmup** (no retries; surface a dispatch failure in one connect budget instead of three) — useful for debugging mlcommons/storage#506-style cold-start issues.  Set to `5` or higher to ride out flaky-network bursts.  Clamped to a minimum of 1; the SDK rejects 0. |

## Data Integrity: Write Verification and Retry (v0.9.104+, opt-in as of v0.9.106)

s3dlio can guard object writes with a HEAD verification step and automatic retry: corrupt or truncated objects are detected immediately after the write, deleted, and re-uploaded without requiring any action from the caller.  This protects against backends that return a successful PUT / CompleteMultipartUpload response for an object that is not actually fully or durably stored (see mlcommons/storage#593) — a server-side contract violation observed on some S3-compatible backends (e.g. AIStore) under certain conditions.

**Disabled by default as of v0.9.106.**  HEAD-after-write verification adds one extra round-trip per object.  For benchmark workloads writing many small objects this can meaningfully reduce measured throughput, and every other major S3 client library (e.g. the AWS CRT-based connectors) trusts the server's success response without a follow-up HEAD.  s3dlio now matches that default: a single PUT (or a single CompleteMultipartUpload), no extra round-trip, relying on `S3DLIO_MAX_RETRY_ATTEMPTS` (SDK layer) to cover transient network failures.  Set the verify flags below to opt back into the HEAD-verify-retry behavior on backends where silent truncation has actually been observed.

**Two independent layers, two independent verify flags** — single-part PUT is handled entirely inside the Rust library; multipart upload verification is also in the Rust library, but multipart *retry-the-whole-upload* orchestration is handled by the DLIO Python integration layer.  You can enable verification for one path and not the other.

### Single-Part PUT — Rust library layer

These variables are read by `put_bytes()` / `put_bytes_async()` inside the s3dlio Rust library.  They apply to **every** call to those functions regardless of the calling application.

| Variable | Default | Allowable values | Description |
|----------|---------|-----------------|-------------|
| `S3DLIO_PUT_VERIFY` | `false` | `1`/`true`/`yes`/`on` (case-insensitive) = enabled; anything else / unset = disabled | Opt-in master switch for single-part write verification.  When `false` (default), `put_bytes()` issues a single PUT and returns — no HEAD, no extra round-trip, matching the cost of every other S3 client library.  When `true`, the HEAD-verify-retry behavior below is active. |
| `S3DLIO_PUT_MAX_RETRIES` | `3` | Integer ≥ 1 | **Only takes effect when `S3DLIO_PUT_VERIFY=true`.**  Total PUT attempts (1 initial + N−1 retries) before `put_bytes()` raises an error.  After each attempt the library issues a HEAD request and compares the reported byte count with the expected size.  On mismatch the truncated object is deleted before the next attempt so the retry always writes to a clean slot.  **`1`** = no retries (fail-fast — useful in tests to surface corruption immediately).  **`5`+** = extra resilience against flaky networks.  Non-numeric or < 1 values are silently treated as `3` (the default). |
| `S3DLIO_PUT_RETRY_DELAY_MS` | `1000` | Integer ≥ 0 (milliseconds) | **Only takes effect when `S3DLIO_PUT_VERIFY=true`.**  Milliseconds to wait between PUT retry attempts.  Applies to both network-error retries and size-mismatch retries.  **`0`** = back-to-back retries with no sleep (useful in unit/integration tests where the backend is local and instant).  **`5000`+** recommended on unreliable WAN links to give the storage backend time to recover.  Non-numeric values are silently treated as `1000`. |

> **`file://` and `direct://` bypass:** local stores have OS-level write durability so the HEAD verification round-trip is skipped entirely for those URI schemes, regardless of `S3DLIO_PUT_VERIFY`.

> **Debug tracing:** set `RUST_LOG=s3dlio=debug` to see per-attempt PUT and HEAD trace lines including URI, byte counts, and attempt numbers (only emitted when verification is enabled).

```bash
# Default: single PUT, no verification, no extra round-trip.
# No action needed — this is the out-of-the-box behavior.

# Opt in to write verification with default retry settings (3 attempts, 1s delay).
export S3DLIO_PUT_VERIFY=true

# Opt in, fast-fail for tests and debugging: one attempt, no delay.
export S3DLIO_PUT_VERIFY=true
export S3DLIO_PUT_MAX_RETRIES=1
export S3DLIO_PUT_RETRY_DELAY_MS=0

# Opt in, maximum resilience for unreliable WAN or overloaded cluster.
export S3DLIO_PUT_VERIFY=true
export S3DLIO_PUT_MAX_RETRIES=5
export S3DLIO_PUT_RETRY_DELAY_MS=5000

# Enable per-attempt debug trace for all s3dlio operations.
export RUST_LOG=s3dlio=debug
```

### Multipart Upload — Rust verification + DLIO Python retry orchestration

`S3DLIO_MPU_PUT_VERIFY` is read by the s3dlio Rust library's multipart coordinator (`MultipartUploadWriter`).  `S3DLIO_MULTIPART_THRESHOLD_MB`, `S3DLIO_MPU_MAX_RETRIES`, and `S3DLIO_MPU_RETRY_DELAY_S` are read by `ObjStoreLibStorage` in DLIO (`dlio_benchmark/storage/obj_store_lib.py`) and apply only when DLIO is the calling application (i.e., training data generation / upload via `--storage-library s3dlio`).

| Variable | Layer | Default | Allowable values | Description |
|----------|-------|---------|-----------------|-------------|
| `S3DLIO_MPU_PUT_VERIFY` | Rust | `false` | `1`/`true`/`yes`/`on` (case-insensitive) = enabled; anything else / unset = disabled | Opt-in master switch for multipart write verification.  When `false` (default), `MultipartUploadWriter` does not issue a HEAD after `CompleteMultipartUpload`; `stored_bytes` in the result is set equal to `total_bytes` (unverified-assumed-equal).  When `true`, a HEAD confirms the stored size and a mismatch raises an error — see below for what happens next on the DLIO side. |
| `S3DLIO_MULTIPART_THRESHOLD_MB` | DLIO (Python) | `16` | Integer ≥ 0 (MiB) | Object size threshold in MiB above which DLIO switches from `put_bytes()` (single-part PUT) to `MultipartUploadWriter` (concurrent multi-part upload).  **`0`** = always use multipart (not recommended for small objects — adds MPU overhead with no throughput benefit).  Set to a very large value (e.g. `999999`) to force all writes through `put_bytes()` regardless of size.  Default 16 MiB is a good balance for MinIO with AI/ML dataset files. |
| `S3DLIO_MPU_MAX_RETRIES` | DLIO (Python) | `3` | Integer ≥ 1 | **Only meaningful when `S3DLIO_MPU_PUT_VERIFY=true`** — without it, the Rust layer never raises on a size mismatch, so this retry loop is only reached on genuine API errors (e.g. a failed `UploadPart`).  Total multipart upload attempts before DLIO raises `RuntimeError`.  On each failure DLIO calls `writer.abort()` to free the in-progress upload slot on the server, then starts a fresh `MultipartUploadWriter` for the next attempt.  Because the payload is already in memory (no disk re-read), retrying is cheap.  **`1`** = no retries (raise on the first failure — useful for debugging).  Non-numeric or < 1 values are silently treated as `3`. |
| `S3DLIO_MPU_RETRY_DELAY_S` | DLIO (Python) | `5` | Number ≥ 0 (seconds, floats accepted) | Seconds to sleep between multipart upload retry attempts.  **`0`** = back-to-back retries with no sleep (useful in unit tests).  **`30`+** recommended when the storage backend needs time to recover between attempts (e.g. after an upload quota event).  Non-numeric values are silently treated as `5`. |

```bash
# Default: no HEAD verification after CompleteMultipartUpload, no extra round-trip.
# No action needed — this is the out-of-the-box behavior.

# Opt in to multipart write verification (recommended on backends where
# silent truncation has been observed, e.g. some AIStore configurations).
export S3DLIO_MPU_PUT_VERIFY=true

# Raise the single-part/multipart threshold so only very large objects use multipart.
export S3DLIO_MULTIPART_THRESHOLD_MB=64

# Force all uploads through put_bytes (bypass MultipartUploadWriter entirely).
export S3DLIO_MULTIPART_THRESHOLD_MB=999999

# Opt in to verification with aggressive retry for unreliable storage.
export S3DLIO_MPU_PUT_VERIFY=true
export S3DLIO_MPU_MAX_RETRIES=5
export S3DLIO_MPU_RETRY_DELAY_S=30

# Opt in, fast-fail for debugging (first failure raises immediately, no sleep).
export S3DLIO_MPU_PUT_VERIFY=true
export S3DLIO_MPU_MAX_RETRIES=1
export S3DLIO_MPU_RETRY_DELAY_S=0
```

### How the two layers interact

For any single object write, exactly one path is active — the two are mutually exclusive:

- **`payload_size < S3DLIO_MULTIPART_THRESHOLD_MB MiB`** → DLIO calls `put_bytes()` → the s3dlio Rust library's `S3DLIO_PUT_*` logic runs (HEAD verify only if `S3DLIO_PUT_VERIFY=true`).
- **`payload_size ≥ S3DLIO_MULTIPART_THRESHOLD_MB MiB`** → DLIO calls `MultipartUploadWriter` → the Rust library's `S3DLIO_MPU_PUT_VERIFY` check runs after `CompleteMultipartUpload`; if it raises, DLIO's `S3DLIO_MPU_*` retry loop orchestrates the retry.

There is no double-retry: if the multipart path is active, the single-part Rust retry path is not, and vice versa.  Verification is independently configurable per path — for example, enabling `S3DLIO_MPU_PUT_VERIFY` for large checkpoint objects while leaving `S3DLIO_PUT_VERIFY` off for the high-volume small-object datagen path.

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
# Optimize for high-throughput workloads (large objects, many concurrent GETs)
export S3DLIO_RT_THREADS=32                  # Tokio worker threads
export S3DLIO_RANGE_CONCURRENCY=64           # concurrent range chunks per object
export S3DLIO_RANGE_THRESHOLD_MB=32          # objects ≥ 32 MB use parallel range GETs
export S3DLIO_POOL_MAX_IDLE_PER_HOST=0       # 0 = unlimited; keep idle conns alive between requests
export S3DLIO_POOL_IDLE_TIMEOUT_SECS=300     # don't tear down idle conns for 5 min
export S3DLIO_OPERATION_TIMEOUT_SECS=120     # raise from 60 s default for very large objects

./target/release/s3-cli get s3://bucket/large-dataset/
```

### Cold-Start Fan-Out Configuration (mlperf-storage scale)
```bash
# Mitigate "dispatch failure" at warmup when hundreds of worker processes
# all issue their first GETs simultaneously (e.g. retinanet B200 with 128
# DataLoader workers × 64 prefetch_window = 8 K concurrent connects).
export S3DLIO_CONNECT_TIMEOUT_SECS=30        # raise from 20 s default — the
                                             # endpoint's TCP accept queue may
                                             # take longer under burst
export S3DLIO_OPERATION_TIMEOUT_SECS=120     # plus a generous full-request budget
export S3DLIO_MAX_RETRY_ATTEMPTS=5           # SDK rides out brief transient failures
export S3DLIO_POOL_MAX_IDLE_PER_HOST=0       # never tear down warm conns
export S3DLIO_RT_THREADS=16                  # per-process; tune to ranks_per_node
```

### Debug Configuration — Fast-Fail on Cold-Start Issues
```bash
# When investigating dispatch-failure root cause, you usually want to fail
# in one connect budget instead of three.  This stops retries from masking
# the original symptom and shrinks the time-to-failure roughly 3×.
export S3DLIO_MAX_RETRY_ATTEMPTS=1           # one shot, no retries
export S3DLIO_CONNECT_TIMEOUT_SECS=10        # back to the previous default for a tighter loop
export RUST_BACKTRACE=full                   # full backtrace in the panic / error path
```

### Memory-Constrained Configuration
```bash
# Optimize for lower memory usage
export S3DLIO_RT_THREADS=8
export S3DLIO_RANGE_CONCURRENCY=8
export S3DLIO_POOL_MAX_IDLE_PER_HOST=8       # cap idle conns per host
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0    # Disable range optimization (on by default since v0.9.60)

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

### Write Integrity Verification — Debug / Fast-Fail

Write verification is off by default (see "Data Integrity" above).  To debug a
suspected silent-corruption issue against a local or test backend, opt in
with fast-fail settings and verbose trace:

```bash
export S3DLIO_PUT_VERIFY=true           # opt in: single-part HEAD verification
export S3DLIO_PUT_MAX_RETRIES=1         # single-part: fail immediately on any PUT error
export S3DLIO_PUT_RETRY_DELAY_MS=0      # no sleep between (there are no) retries
export S3DLIO_MPU_PUT_VERIFY=true       # opt in: multipart HEAD verification
export S3DLIO_MPU_MAX_RETRIES=1         # multipart (DLIO layer): same, fail immediately
export S3DLIO_MPU_RETRY_DELAY_S=0
export RUST_LOG=s3dlio=debug            # see each PUT attempt + HEAD verification line
```

### Write Integrity Verification — Production / Backend With Known Truncation Issues

For a backend where silent truncation has actually been observed (e.g. some
AIStore configurations under storage#593), opt in to verification with
resilient retry settings for a high-latency or intermittently overloaded
cluster:

```bash
export S3DLIO_PUT_VERIFY=true
export S3DLIO_PUT_MAX_RETRIES=5
export S3DLIO_PUT_RETRY_DELAY_MS=5000   # 5-second backoff between single-part retries
export S3DLIO_MPU_PUT_VERIFY=true
export S3DLIO_MPU_MAX_RETRIES=5
export S3DLIO_MPU_RETRY_DELAY_S=30      # 30-second backoff between multipart retries
```

### Write Integrity Verification — Default (No Action Needed)

```bash
# Out-of-the-box behavior as of v0.9.106: no HEAD verification, no extra
# round-trip per object. Matches the default behavior of other S3 client
# libraries. Nothing to set.
```

### Conservative Configuration
```bash
# Use AWS SDK defaults with minimal optimization
export S3DLIO_RT_THREADS=8
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0    # disable range-split GETs
# (do not set the high-throughput vars above; defaults are conservative)

./target/release/s3-cli get s3://bucket/files/
```

## Performance Tuning Guidelines

### For Large Objects (>100MB)
- Range optimization is **enabled by default** (v0.9.60+); no action needed
- Lower `S3DLIO_RANGE_THRESHOLD_MB` if needed (default is 32 MB)
- Increase `S3DLIO_RANGE_CONCURRENCY` to 32-64
- Set `S3DLIO_POOL_MAX_IDLE_PER_HOST=0` (unlimited) to amortize TCP handshakes across requests
- Raise `S3DLIO_OPERATION_TIMEOUT_SECS` if a single object can exceed the 60 s default
- Consider `S3DLIO_RT_THREADS=32` on high-core systems

> **Note on deprecated/dead names.** Older versions of this guide and external
> tutorials sometimes reference `S3DLIO_USE_OPTIMIZED_HTTP`,
> `S3DLIO_MAX_HTTP_CONNECTIONS`, `S3DLIO_HTTP_IDLE_TIMEOUT_MS`,
> `S3DLIO_MAX_CONCURRENCY`, `S3DLIO_CONNECTION_TIMEOUT`,
> `S3DLIO_READ_TIMEOUT`, `S3DLIO_MULTIPART_THRESHOLD`, or
> `S3DLIO_PART_SIZE`.  **None of these are read by s3dlio source code as of
> v0.9.102.**  Use the wired equivalents above
> (`S3DLIO_POOL_MAX_IDLE_PER_HOST`, `S3DLIO_POOL_IDLE_TIMEOUT_SECS`,
> `S3DLIO_CONNECT_TIMEOUT_SECS`, `S3DLIO_OPERATION_TIMEOUT_SECS`,
> `S3DLIO_RANGE_CONCURRENCY`) instead.

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

- **v0.9.106**: Write verification (single-part `S3DLIO_PUT_VERIFY` and multipart `S3DLIO_MPU_PUT_VERIFY`) changed from always-on to **opt-in, default `false`** — matches the default behavior of other S3 client libraries and avoids an extra round-trip on every object write for benchmark workloads.  `S3DLIO_PUT_MAX_RETRIES`/`S3DLIO_PUT_RETRY_DELAY_MS` and `S3DLIO_MPU_MAX_RETRIES`/`S3DLIO_MPU_RETRY_DELAY_S` are unchanged but now only take effect when the corresponding verify flag is enabled (mlcommons/storage#593)
- **v0.9.104**: Added `S3DLIO_PUT_MAX_RETRIES` and `S3DLIO_PUT_RETRY_DELAY_MS` — single-part PUT with HEAD verification and automatic retry for all network backends (S3, Azure, GCS); `file://`/`direct://` bypass.  DLIO integration layer adds `S3DLIO_MPU_MAX_RETRIES` and `S3DLIO_MPU_RETRY_DELAY_S` for multipart upload retry (mlcommons/storage#593)
- **v0.9.102**: Added `S3DLIO_CONNECT_TIMEOUT_SECS` (default 20 s, up from 5 s SDK default), `S3DLIO_OPERATION_TIMEOUT_SECS`, `S3DLIO_MAX_RETRY_ATTEMPTS`; unified AWS SDK and reqwest timeout layers
- **v0.9.92**: `S3DLIO_H2C` default changed from `auto` to `0` (HTTP/1.1); `S3DLIO_POOL_MAX_IDLE_PER_HOST` default changed to unlimited; `S3DLIO_H2_ADAPTIVE_WINDOW`, `S3DLIO_H2_STREAM_WINDOW_MB`, `S3DLIO_H2_CONN_WINDOW_MB` added
- **v0.9.60**: Range optimization enabled by default; default threshold changed 64 MB → 32 MB; set `=0` to disable
- **v0.7.5**: Added HTTP client optimization variables
- **v0.7.4**: Added runtime threading control
- **v0.7.0+**: Range GET optimization variables
- **v0.6.0+**: Operation logging variables

## See Also

- [Phase1 GET Optimization Plan](Phase1_GET_Optimization_Plan.md) - Implementation details
- [Performance Optimization Summary](Performance_Optimization_Summary.md) - Performance analysis
- [README.md](../README.md) - Main project documentation
