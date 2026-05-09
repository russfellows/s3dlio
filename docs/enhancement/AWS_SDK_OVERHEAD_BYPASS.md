# Enhancement: AWS SDK Middleware Overhead Bypass for High-Throughput GET Paths

## Status

**Proposed** — Not yet implemented.

## Problem

The AWS Rust SDK adds approximately **41 ms of per-request overhead** compared to
thin clients (minio-go, raw reqwest) making requests against the same S3-compatible
endpoint.  This overhead is large enough to cap throughput regardless of how fast
the server can respond.

### Measured Evidence (May 2026, loopback against s3-ultra)

| Client | Per-RG-GET latency (p50) | Effective req/s | Throughput |
|---|---|---|---|
| **warp** (minio-go, SigV4 signed) | ~18 ms | ~5,475 req/s | ~34 GB/s |
| **sai3-bench** (AWS SDK Rust) | ~59 ms | ~1,220 req/s | ~9.7 GB/s |

Both clients:
- Hit the same s3-ultra server on `localhost:9200`
- Perform identical operations: byte-range GETs at real Parquet row-group offsets
- Use SigV4 signing with the same credentials
- Use 72 concurrent in-flight requests

The **~41 ms gap is purely client-side** — it is the cost of the AWS SDK's
tower middleware stack executing for every request.

### AWS SDK Middleware Layers (per request, hot path)

Each `GetObject` call traverses:

1. Credentials provider (resolved at request build time)
2. SigV4 request signer interceptor (tower layer)
3. Retry middleware (tower layer, even when retries=0)
4. Timeout middleware (tower layer)
5. Smithy request serializer / endpoint resolver
6. `reqwest` (actual TCP send — this is the fast part)
7. Smithy response deserializer
8. Response type mapping

Steps 1–5 and 7–8 are pure overhead for a fixed-size range GET of a known
object.  On loopback (sub-millisecond network RTT), these layers dominate
total request latency.

### Concurrency and Little's Law

At the optimal in-flight count (~72 concurrent GETs for this workload):

```
Throughput = in_flight / latency_per_req × bytes_per_req
           = 72 / 0.059 s × 7.9 MiB
           ≈ 9,650 MiB/s                       ← matches measured 9.7 GB/s
```

Adding more concurrency past the knee (e.g., 128 in-flight) only inflates
queueing latency further, reducing throughput. The only way to improve is to
reduce per-request latency.

## Proposed Enhancement: Direct reqwest GET Path

Replace the AWS SDK path for the common `GET object-range` operation with a
direct `reqwest` call + lightweight SigV4 signing.

`reqwest` is **already the underlying HTTP transport** in s3dlio — the AWS SDK
wraps it via the `aws-smithy-runtime` reqwest adapter.  We can call it directly
with a hand-crafted signed `Range` request, bypassing all SDK middleware.

### Interface Sketch

```rust
/// High-performance GET for a byte range.  Uses direct reqwest + SigV4 signing,
/// bypassing the AWS SDK middleware stack.  For trusted S3-compatible endpoints
/// (MinIO, s3-ultra, Ceph RGW) where retry/timeout middleware adds overhead
/// without benefit.
///
/// Enable via S3DLIO_FAST_GET=1 (or always-on once validated).
pub async fn get_range_fast(
    uri: &str,
    offset: u64,
    length: u64,
) -> Result<Bytes>;
```

### Expected Outcome

Replacing the ~59 ms SDK path with ~18 ms direct reqwest would yield:

```
Throughput ≈ 72 / 0.018 s × 7.9 MiB ≈ 31,600 MiB/s (~31 GB/s)
```

This would close the gap with warp to within measurement noise.

## Alternative: Unsigned Requests (S3DLIO_NO_SIGNING)

For completely trusted, internal endpoints, request signing can be
disabled entirely.  The AWS SDK supports an anonymous credential provider.
This eliminates signing CPU but does NOT eliminate the tower middleware layers —
so the latency reduction would be partial.

Estimated benefit: ~5–10 ms reduction (signing computation), not the full 41 ms.

## Scope

Changes are isolated to s3dlio's GET path (`s3_utils.rs` / `data_loader/parquet_rg.rs`).
No changes needed in sai3-bench or dl-driver — they call the s3dlio API which
would transparently use the fast path.

The existing `get_object_range_uri_async` function would remain as the default;
the fast path would be opt-in via environment variable, graduating to default
once validated against real AWS S3.

## Dependencies

- `reqwest` (already a dependency, already used as underlying transport)
- A lightweight SigV4 signing crate, or the existing `aws-sigv4` crate used
  directly without the full SDK middleware (it is already in the dependency tree
  via `aws-sdk-s3`)

## References

- [Performance Optimization Summary](../performance/Performance_Optimization_Summary.md)
- [Benchmarking Analysis: sai3-bench vs warp](../performance/AWS_SDK_vs_Thin_Client_Benchmark.md)
- s3dlio `src/s3_client.rs` — global AWS SDK client initialization
- s3dlio `src/s3_utils.rs` — `get_object_range_uri_async`, `get_object_range_uri_timed_async`
- s3dlio `src/data_loader/parquet_rg.rs` — `ParquetRowGroupDataset::get_timed`
