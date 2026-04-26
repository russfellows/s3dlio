# Custom S3 Client Analysis: Build vs. Buy

**Date**: April 23, 2026  
**Context**: Performance investigation of sai3-bench → s3-ultra loopback benchmark  
**Status**: Analysis only — no implementation decisions made

---

## Background and Motivation

A profiling session comparing two benchmark paths revealed a 7.6× throughput gap:

| Benchmark | Ops/s | Client stack |
|-----------|-------|-------------|
| `s3bench` (internal loopback) | ~240,000 | Raw `reqwest`, no auth, no AWS SDK |
| `sai3-bench` → `s3-ultra` (loopback TCP) | ~46,000 | AWS SDK (`aws-sdk-s3` v1.129) |

Both paths use the same loopback TCP transport. Both send 1 KiB PUT objects. The server
(`s3-ultra`) is identical. The only material difference is the client: `s3bench` sends
raw unsigned HTTP requests; `sai3-bench` sends them through the full AWS SDK middleware stack.

Flamegraph profiling (`perf record + inferno`) of the sai3-bench process identified these
per-request costs as attributable to the AWS SDK:

| Category | CPU% | Notes |
|----------|------|-------|
| SHA-256 (SigV4 request signing) | 4.34% | `sha2::sha256::compress256` |
| SigV4 canonical request construction | 0.59% | `aws_sigv4::http_request::CanonicalRequest::from` |
| AWS SDK interceptor chain (13 distinct hooks) | ~5.0% | Each `Interceptors<I>::*` method ≈ 0.35% |
| Config bag iteration per request | 3.65% | `aws_smithy_types::config_bag::ItemIter::next` |
| `RuntimeComponentsBuilder::merge_from` | 0.54% | Called on every request setup |
| `regex_lite` URL validation | 2.11% | `pikevm::epsilon_closure` + `search` |
| IDNA hostname normalization | 0.40% | `idna::uts46::Uts46::process_innermost` |
| URL parser | 0.39% | `url::parser::Parser::after_double_slash` |
| Heap allocation/deallocation churn | ~18.0% | `malloc`/`_int_free`/`realloc`; majority SDK-driven |
| **Total SDK-attributable** | **~35%** | Estimate; SigV4 unavoidable for real AWS |
| TCP/Tokio/kernel scheduling | ~5–6% | Inherent cost of any async TCP client |

**The critical insight**: roughly 30% of client CPU (everything except SHA-256) is AWS SDK
infrastructure overhead that runs on every single request regardless of payload size.

---

## What `s3s` Is — And Why Its Code Is Still Relevant

The crate `s3s` (used by `s3-ultra`) is a **server-side** S3 protocol library. It parses
incoming HTTP requests and routes them to a trait-based handler. It has no HTTP client
or request-sending functionality — `s3s` cannot directly replace `aws-sdk-s3`.

However, **the SigV4 math is identical from both sides of the connection**, and this is the key
insight. Both client and server process the same canonical request format, the same HMAC-SHA256
key derivation, and the same string-to-sign construction. A server *verifies* these; a client
*produces* them. The algorithm is the same.

`s3s`'s `sig_v4/methods.rs` module contains clean, pure implementations of:
- `create_canonical_request(method, uri_path, query, headers, payload_hash)`
- `calculate_signature(signing_key, string_to_sign)`
- `hmac_sha256(key, data)`

These functions have **zero AWS SDK middleware dependencies** — no config bags, no interceptors,
no orchestrator. They take `&str`, do the math, return `String`.

**What this means for Option B (thin wrapper)**: instead of reaching for `aws-sigv4` as the
signing primitive and writing your own canonical request builder, `s3s`'s signing code could
be used directly. It is ~600 lines of clean, tested SigV4 math that `s3-ultra` already runs
in production on this benchmark. s3s also uses `SmallVec<[u8; 512]>` for URI encoding to
avoid heap allocation for short paths — an optimization the AWS SDK does not have.

The practical approach: pull in `s3s` as a dependency, call `s3s::sig_v4::methods::*`
directly from a reqwest-based client, and avoid the full AWS SDK orchestrator entirely.
The "server code building a client" framing is misleading — what you're actually doing is
using the signing primitives from a reference implementation that happens to be server-facing.

`s3-ultra` *uses* `s3s` to *receive* S3 requests. `s3dlio` *sends* S3 requests using
`aws-sdk-s3`. These are opposite roles — but the signing mathematics they share is the
same ~600 lines of code.

---

## Option Analysis

Three implementation paths exist for a lighter S3 client.

---

### Option A: Build a Custom S3 Client from Scratch

**Architecture**: raw `reqwest` + standalone `aws-sigv4` crate + hand-written request/response
serialization for the ~8 operations `s3dlio` actually uses
(PutObject, GetObject, GetObjectRange, HeadObject, DeleteObject, ListObjectsV2, CreateBucket,
DeleteBucket).

**What would need to be written**:
- SigV4 signing wrapper (the `aws-sigv4` crate already exists standalone — ~2k LOC to integrate)
- HTTP request construction for each operation (~50–100 lines per operation = ~600–800 LOC total)
- Response parsing (XML for ListObjects/Error, headers only for others = ~300 LOC)
- Error type mapping (S3 XML error codes → `anyhow::Error` / custom type)
- Multipart upload coordination (`s3dlio` uses this for large objects)
- Presigned URL generation (used by Python bindings)
- Credential provider (environment, profile, IMDS for EC2 — currently free from AWS SDK)
- Regional endpoint resolution (bucket-regional endpoints, path vs. virtual-hosted style)

**Total estimated effort**: 4,000–7,000 lines of new code, plus test coverage.

**Ongoing maintenance concerns**:
- AWS adds API changes and new authentication schemes regularly (e.g., SigV4a for multi-region,
  S3 Express One Zone, object checksums CRC32C/SHA-1/SHA-256 headers added in 2023–2024)
- Any new feature used by downstream tools (sai3-bench, dl-driver) requires manual implementation
- Security patches to SigV4 (e.g., header injection vulnerabilities) must be tracked and applied
  independently
- Credential refresh (STS AssumeRole, web identity tokens) is complex and security-sensitive

**Security testing difficulty**: HIGH
- SigV4 has subtle correctness requirements (canonical URI encoding, header normalization,
  trailing newline in string-to-sign, etc.) that have historically caused vulnerabilities
- The AWS SDK has been security-audited; a custom impl starts from zero trust
- Testing against real AWS requires live credentials; testing against s3-ultra does not verify
  real-world security correctness
- Misimplementing SigV4 is not just a performance issue — it can lead to request forgery
  if the canonical form differs from what a validator expects

---

### Option B: Thin Reqwest Wrapper (Partial SDK Bypass)

**Architecture**: Keep `aws-sigv4` (standalone signing crate) and `reqwest` but **eliminate the
`aws-sdk-s3` middleware stack entirely**. This is not the same as building from scratch —
it reuses the already-audited SigV4 signing logic while discarding the interceptor/orchestrator
machinery that accounts for ~25–30% of the per-request overhead.

**What this eliminates** vs. Option A:
- No credential provider rewrite — use `aws-config` for credential loading only (no request overhead)
- No SigV4 reimplementation — use `aws-sigv4` crate standalone
- No security regression on the crypto path — same code, just invoked differently

**What still needs writing**:
- HTTP verb + header construction per operation (~600–800 LOC, same as Option A)
- XML response parser for ListObjects and error responses
- Multipart upload state machine
- Regional endpoint resolution

**Estimated effort**: 2,500–4,000 LOC (smaller than Option A because credential loading
and SigV4 are reused).

**Maintenance burden**: Medium. AWS SDK API changes don't automatically flow through,
but the signed-URL and credential pieces track separately. New S3 features still require
manual implementation.

---

### Option C: Switch to an Existing Lightweight S3 Crate

Several third-party Rust S3 clients exist outside the AWS SDK:

| Crate | Stars (Apr 2026) | Last updated | Key tradeoffs |
|-------|-----------------|--------------|---------------|
| `rusty-s3` | ~400 | Active | HTTP-agnostic (caller drives reqwest/hyper), no async, pure URL+signing |
| `opendal` (Apache) | ~4,000 | Active | Multi-backend abstraction; S3 backend uses reqwest directly |
| `minio/minio-rs` | ~200 | Sporadic | MinIO-specific extensions; wraps AWS SDK internally |
| `aws-s3-client` (mountpoint) | ~800 | Active | AWS's own high-throughput client for Mountpoint-S3; optimized for throughput |

**`aws-s3-client` (Mountpoint)** is the most interesting option here. AWS built it
specifically for high-throughput workloads (Mountpoint for Amazon S3, their FUSE driver).
It uses `aws-crt-s3` (the C-based AWS Common Runtime) via FFI, which internally uses
multiplexed HTTP/2 and has been benchmarked at multi-GB/s. However:
- It requires linking against the CRT (C dependencies, non-trivial build)
- It targets throughput for large objects, not low-latency small-object ops/s

**`opendal`** is worth understanding in depth. OpenDAL (Open Data Access Layer) is an
Apache incubator project providing a unified interface for reading and writing data
across heterogeneous storage backends — S3, Azure Blob, GCS, HDFS, local filesystem,
MinIO, and ~50 others — through a single Rust trait:

```rust
let op = Operator::new(S3::default())?
    .layer(RetryLayer::new())
    .finish();
let data = op.read("path/to/object").await?;
op.write("path/to/object", data).await?;
```

The S3 backend in OpenDAL uses `reqwest` directly — **not** the AWS SDK. It implements
SigV4 signing using the `reqsign` crate, which is OpenDAL's purpose-built signing
library (~3k LOC). `reqsign` does the same math as `aws-sigv4` but has no
`aws-smithy-runtime` dependency: no config bags, no interceptors, no orchestrator.

**How OpenDAL's architecture compares to what s3dlio already does**:

```
s3dlio today:
  ObjectStore trait → S3Backend → aws-sdk-s3 → aws-smithy-runtime → reqwest_client.rs

OpenDAL integration path:
  ObjectStore trait → S3Backend → OpenDAL S3 layer → reqsign → reqwest
```

OpenDAL would replace `aws-sdk-s3` and `aws-smithy-runtime` (the overhead sources)
while `s3dlio`'s higher-level value-add is preserved:
- Multi-endpoint load balancing (`MultiEndpointStore`) — not in OpenDAL
- The custom `reqwest_client.rs` h2c/h2 transport tuning — would need re-integration
- Page cache and prefetch machinery — not in OpenDAL
- Python bindings (PyO3) — not in OpenDAL
- HDR histogram metrics — not in OpenDAL

**Maintenance model**: The Apache OpenDAL project (~4,000 GitHub stars, ~150 contributors
as of early 2026) owns S3 protocol correctness. New AWS S3 API features tracked by that
project flow to s3dlio through a version bump rather than requiring manual implementation.

**Downside**: The `reqwest` transport customizations (h2c protocol probing, connection
pool limit overrides, h2 flow control window tuning) that live in `s3dlio`'s
`reqwest_client.rs` would need to be re-integrated since OpenDAL's S3 backend manages
its own `reqwest::Client` configuration. This is solvable but adds ~1 week of work.

**`aws-s3-client` (Mountpoint)** is the most interesting option for large-object throughput.
AWS built it specifically for high-throughput workloads (Mountpoint for Amazon S3, their FUSE
driver). It uses `aws-crt-s3` (the C-based AWS Common Runtime) via FFI, which internally uses
multiplexed HTTP/2 and has been benchmarked at multi-GB/s. However:
- It requires linking against the CRT (C dependencies, non-trivial build)
- It targets throughput for large objects, not low-latency small-object ops/s

## Performance Estimate

The profiling data gives a concrete upper bound on the gains from removing SDK overhead.

**Conservative estimate** (eliminating non-SigV4 SDK overhead only):

| Removed overhead | Client CPU recovered |
|-----------------|---------------------|
| 13 interceptor hooks | ~5.0% |
| Config bag iteration | ~3.65% |
| regex_lite URL validation | ~2.11% |
| IDNA + URL parsing | ~0.79% |
| RuntimeComponentsBuilder | ~0.54% |
| Reduced heap churn (SDK alloc patterns) | ~8–12% (conservative estimate) |
| **Total** | **~20–24%** |

Applying 20–24% more available client CPU to actual work, starting from 46k ops/s,
**projects to approximately 55–57k ops/s** — roughly a **20–24% improvement**.

**Optimistic estimate** (including jemalloc + connection pool tuning + removed overhead):
Potentially 60–70k ops/s, a **30–50% improvement**.

**Important ceiling**: Even with a perfect zero-overhead client, the server would become
the bottleneck. With `--skip-sig-verify` removing ~14% of server CPU overhead, the
combined server+client improvement could push further, but `s3bench`'s 240k ops/s
remains a theoretical ceiling that would require eliminating TCP round trips entirely
— not a realistic goal.

The **20% performance threshold is likely achievable**, but only narrowly, and only with
Option B or Option C. Option A would require near-perfect execution to hit 20%.

---

## Summary Evaluation Against the Four Concerns

### 1. Effort Required

| Option | Estimated LOC | Initial dev time |
|--------|--------------|-----------------|
| A: From scratch | 4,000–7,000 | 6–12 weeks |
| B: Thin wrapper (reuse sigv4) | 2,500–4,000 | 3–6 weeks |
| C: Adopt opendal S3 backend | ~500 (integration) | 1–2 weeks |
| C: Adopt aws-s3-client (CRT) | ~200 (integration) | 2–4 weeks (CRT build) |

These estimates do not include multipart upload, presigned URLs, or Python binding regression
testing — each adds 1–3 weeks.

### 2. Ongoing Maintenance

This is the most serious concern. The AWS S3 API is not static:

- **2023–2024 changes**: Default checksum validation (`x-amz-checksum-*`), request payer
  headers, bucket ownership controls, Express One Zone storage class
- **Historical pace**: AWS makes 50–200 S3 API additions/changes per year, most minor but
  some (checksum enforcement, SigV4 changes) requiring client updates
- **AWS SDK handles this automatically** — its maintainers track API changes in the `smithy`
  model and regenerate code
- A custom client would require tracking AWS changelog and manually implementing anything
  new that downstream tools use

**Honest assessment**: For a team of 1–2 people working on multiple projects simultaneously,
maintaining a custom S3 client long-term is a significant ongoing tax. The AWS SDK overhead
is real, but so is the cost of being one breaking change behind.

### 3. Security Implications

SigV4 signing is security-critical. A subtle encoding bug in canonical URI normalization
or header sorting does not cause test failures — it causes silent incorrect behavior or,
in worst cases, allows spoofed requests to pass a misconfigured server.

The AWS SDK's SigV4 implementation is:
- Maintained by AWS security engineers
- Fuzz-tested against the AWS SigV4 test suite
- Audited as part of AWS's security program

A custom implementation:
- Cannot easily be tested against real AWS (signatures are rejected, not just logged)
- Would need its own SigV4 test vector suite (AWS publishes some, but not comprehensive)
- Has no external audit

**Mitigation**: Option B (reuse `aws-sigv4` standalone crate) fully avoids this risk.
The `aws-sigv4` crate is the same signing code the SDK uses, just callable without the
full middleware stack.

### 4. Expected Performance Delta

Based on profiling analysis: **20–24% improvement is probable** (just at the stated
threshold), with potential for 30–50% under the optimistic scenario with jemalloc
added. The performance argument is real but not overwhelming.

Critically: **the performance gap is not caused by one thing**. It is a sum of
small overheads (~2–5% each) distributed across 8–10 SDK components. Any implementation
that reuses even one of those components (e.g., retaining the interceptor chain) will
only capture part of the gain.

---

## Recommendation

**Do not pursue Option A** (from scratch). The maintenance and security burden is
disproportionate to the performance gain, especially for a 2-person-equivalent
project with multiple active codebases.

**Evaluate Option B as a targeted optimization**, scoped specifically to the PUT/GET
hot path used in benchmarking. The key insight is that `s3dlio` already has a custom
`reqwest_client.rs` (a custom HTTP transport for the AWS SDK). Going one step further
— replacing the SDK's orchestration layer with a thin manual layer for the 3–4 hot
operations — is bounded work (2–3 weeks) that would directly close the performance gap.

This is not "rewriting the AWS SDK." It is writing ~600 lines of HTTP request
construction that bypasses the middleware for `PutObject` and `GetObject`, while
keeping the AWS SDK for everything else (multipart, list, credential loading, etc.).

**Alternatively, evaluate `opendal`** as a drop-in for the S3 backend. It is actively
maintained (Apache project), eliminates SDK overhead, and provides a clean abstraction
layer that maps naturally onto `s3dlio`'s existing `ObjectStore` trait. The integration
risk is low compared to Option A.

**What to do first**: Add `--skip-sig-verify` to s3-ultra (already done) and re-run the
benchmark. This will quantify how much of the gap is server-side SigV4. That number tells
us exactly how much budget remains for the client-side optimization to close the gap to
the 20% improvement target.

---

## Appendix: Overhead Not Captured by This Analysis

The analysis above is based on a single benchmark configuration (1 KiB PUT, t=128, loopback
TCP). The following factors would shift the conclusion:

- **Large objects (>1 MB)**: The SDK overhead becomes a smaller fraction of total time;
  the performance argument for a custom client weakens significantly
- **GETs vs. PUTs**: GET performance involves streaming response bodies; the overhead profile
  is different (fewer allocation round-trips)
- **Real AWS with network latency**: At 10–100ms round trips, SDK overhead at ~0.1ms/request
  is negligible; the performance argument disappears entirely
- **Multi-endpoint**: `s3dlio`'s `MultiEndpointStore` adds its own per-request overhead;
  this should be profiled separately before attributing all overhead to the SDK

The performance case for a custom S3 client is strongest specifically for high-concurrency,
small-object, low-latency benchmarking scenarios (exactly the sai3-bench use case). It
weakens substantially for production AI/ML workloads reading large files from real AWS.

---

## See Also

[SHA256_SIGV4_OPTIMIZATION_ANALYSIS.md](SHA256_SIGV4_OPTIMIZATION_ANALYSIS.md) — deep dive
into the specific SHA-256 and SigV4 optimization opportunities: hardware acceleration status
(SHA-NI vs. software fallback on this CPU), signing key caching, the `UNSIGNED-PAYLOAD`
fast path, and PR candidates for the upstream `aws-sigv4` crate.
