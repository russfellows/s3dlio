# AWS SDK vs Thin Client Benchmark: DLRM Parquet Row-Group GETs

## Summary

The AWS Rust SDK imposes ~41 ms of per-request middleware overhead compared to
thin S3 clients (minio-go, raw reqwest) against the same S3-compatible endpoint.
At the optimal concurrency level this limits sai3-bench to ~9.7 GB/s while an
equivalent thin client (warp/minio-go) achieves ~34 GB/s against the same server.

## Test Setup

- **Server**: s3-ultra v0.2.0 on `localhost:9200`
- **Dataset**: 64 Parquet files × 123 row groups/file = 7,872 RGs, ~7.9 MiB/RG
  (`s3://mlp-flux/data/dlrm/train/`)
- **Operation**: Byte-range GETs at real Parquet row-group offsets (both clients
  parse the footer and use actual byte offsets — no random ranges)
- **Concurrency**: 72–80 in-flight requests (optimal for this server)
- **Date**: May 2026, loopback (single-host, no network latency)

## Results

### warp (minio-go thin client, SigV4 signed)

```
Average:   34,579 MiB/s  (33.8 GiB/s)
Fastest:   35,560 MiB/s  (34.7 GiB/s)
Ops/s:     1,095 ops/s  (each op = footer cache check + 4 RG GETs)
HTTP req/s: ~5,475 req/s
Per-RG latency: ~18 ms
```

### sai3-bench (AWS SDK Rust, SigV4 signed)

Optimal concurrency = 18 workers × 4 rg_reads = 72 in-flight GETs:

```
Throughput:   9,690 MiB/s  (9.5 GiB/s)
Ops/s:        1,231 ops/s  (each op = 1 individual RG GET)
HTTP req/s:   1,231 req/s
Per-RG latency (p50): 58.9 ms
```

## Analysis

### Accounting for the Different "op" Definitions

warp counts `footer_check + 4 × RG_GET` as one op.  
sai3-bench counts each individual RG GET as one op.

Normalized to per-HTTP-request:

| Metric | warp | sai3-bench | Ratio |
|---|---|---|---|
| Req/s | ~5,475 | 1,231 | 4.4× |
| Per-req latency | ~18 ms | 58.9 ms | **3.3×** |
| Throughput | 34,579 MiB/s | 9,690 MiB/s | **3.6×** |

### Root Cause: AWS SDK Middleware Stack

Both clients use SigV4 signing.  The gap is not signing computation — it is the
**tower middleware layers** that wrap every AWS SDK request:
credentials provider, SigV4 interceptor, retry middleware, timeout middleware,
Smithy request serializer, and response deserializer.

On loopback (actual network RTT < 0.1 ms), these layers collectively add
approximately **41 ms per request** — more than doubling the effective latency
relative to a thin client doing the same work with the same signing.

### Little's Law Confirms the Bottleneck

```
sai3-bench throughput = 72 in-flight / 0.059 s × 7.9 MiB ≈ 9,637 MiB/s  ✓
warp throughput       = 80 in-flight / 0.018 s × 7.9 MiB ≈ 35,111 MiB/s  ✓
```

Pushing sai3-bench past 72 in-flight (e.g., 128 at concurrency=32) inflates
queuing latency further and *reduces* throughput — the server is not the
bottleneck, the client middleware queue is.

### Concurrency Sweep (sai3-bench, channel pipeline)

| Workers | In-flight GETs | TTFB p50 | Throughput |
|---|---|---|---|
| 18 (optimal) | 72 | 59 ms | 9,690 MiB/s |
| 32 (over-driven) | 128 | 95–115 ms | 9,163 MiB/s |

## Proposed Fix

See [AWS_SDK_OVERHEAD_BYPASS.md](../enhancement/AWS_SDK_OVERHEAD_BYPASS.md) for
the proposal to use direct `reqwest` + lightweight SigV4 on the hot GET path,
bypassing the tower middleware stack.

Expected result after fix: ~31 GB/s at the same 72 in-flight concurrency level.
