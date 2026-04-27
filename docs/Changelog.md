# s3dlio Changelog

## Version 0.9.95 — O_DIRECT fix, BufferPool deadlock fix (April 27, 2026)

### Fix: `direct://` URIs now correctly use O_DIRECT in `get_many()` (`src/python_api/python_core_api.rs`)

`Scheme::Direct` was previously handled by the same code branch as `Scheme::File`, causing `direct://` URIs to silently fall through to `tokio::fs::read()` — the standard buffered I/O path. Page-cache bypass was never actually engaged in `get_many()`.

**Fix:** Split into two separate match arms. `Scheme::Direct` now creates a `ConfigurableFileSystemObjectStore::with_direct_io()` and submits all reads through it, bypassing the OS page cache as intended. `Scheme::File` retains its existing `tokio::fs::read()` path with the stale `direct://` prefix-stripping code removed.

**Impact:** `direct://` reads in Python (`get_many(["direct:///path/file", ...])`) now actually bypass the page cache. This was a silent correctness bug — the API appeared to work but provided no O_DIRECT benefit.

### Fix: `BufferPool::give()` deadlock (`src/memory.rs`, `src/file_store_direct.rs`)

**Root cause:** `BufferPool::new(n, …)` spawns a pre-allocation task on the global background runtime that fills the channel to capacity `n`. If `take()` runs before pre-alloc completes (the normal case — the caller's runtime is usually faster), it allocates a new buffer via the fallback path, creating a `n+1`th buffer that the channel cannot hold. The previous `async fn give()` called `tx.send(buf).await`, which blocks indefinitely on a full channel with no concurrent drainer → **permanent deadlock**.

This is a real production bug, not just a test issue. Any caller using `BufferPool` under high concurrency where `take()`'s fallback path fires (by design) could deadlock.

**Fix:** `give()` changed from `async fn` using `tx.send().await` to a sync `fn` using `tx.try_send()`. If the channel is full, the buffer is simply dropped — correct behaviour, since fallback-allocated buffers are not needed by the pool. All call sites updated (`src/file_store_direct.rs`, `tests/test_buffer_pool_directio.rs`).

**Test hardening:** `buffer_pool_basic` now wraps the test body in `tokio::time::timeout(5s)` so a regression will fail with a clear message instead of hanging indefinitely.

### Internal: `global_rt_handle()` made `pub(crate)` (`src/s3_client.rs`)

Allows `memory.rs` to spawn `BufferPool` pre-allocation tasks on the same background runtime used by all other async operations — consistent runtime behaviour across all URI schemes.

---

## Version 0.9.94 — Fast NPZ generation, zero-copy BytesView upload path (April 25, 2026)

### New: `generate_npz_bytes()` — Rust NPZ builder (`src/data_formats/npz.rs`, `src/python_api/python_datagen_api.rs`)

Generates a complete NumPy `.npz` archive in Rust without holding the GIL.

**Design:**
- Single `Vec<u8>` allocation sized exactly to the final file — no intermediate copies
- Random data (`x.npy`) filled in-place with Xoshiro256++ via Rayon (2 MiB chunks, each with an independent seed — page faults distributed across all cores)
- CRC32 computed by `crc32fast` (SSE4.2/NEON hardware acceleration) on already-hot pages
- ZIP structure (local file headers, central directory, EOCD) written with fixed-offset pointer arithmetic — no `ZipWriter` state machine overhead
- Labels (`y.npy`) are int64 zeros, matching the unet3d training format

**Performance (140 MiB file, 28-core machine):**

| Method | Latency |
|--------|---------|
| `numpy.savez()` (Python CRC32 via `zlib`) | ~178 ms |
| `generate_npz_bytes_raw()` (Rust, this change) | ~20 ms |

The returned `Vec<u8>` is wrapped in `Bytes::from(vec)` (zero-cost Arc) and returned as a `PyBytesView`.

### Fix: Duplicate `PyBytesView` causing silent zero-copy failure (`src/python_api/python_datagen_api.rs`)

`python_datagen_api` previously defined its own `PyBytesView` (backed by `DataBuffer`) and registered it as Python's `BytesView` class. Because `python_datagen_api` was registered after `python_core_api`, the datagen version silently replaced the core version at the Python class level. `MultipartUploadWriter.write()`'s fast path tried to extract `python_core_api::PyBytesView` — the extraction always failed and fell through to the `PyBuffer` path, which held the GIL during a full `memcpy` (~109 ms for 140 MiB, serializing all concurrent threads).

**Fix:** Removed the duplicate struct entirely. `python_datagen_api` now imports and uses `python_core_api::PyBytesView`. Generate functions use `DataBuffer::into_bytes()` (zero-cost for `Uma(Vec<u8>)` via `Bytes::from(vec)`) before wrapping.

**Impact:** Write latency for 140 MiB BytesView dropped from ~109 ms → ~6 ms. Sustained upload throughput at N=48 improved from ~1,250 MiB/s → ~2,440 MiB/s (1.95×).

### New: `write_bytes_blocking()` on `MultipartUploadSink` (`src/multipart.rs`)

Zero-copy write method accepting owned `bytes::Bytes`. For large inputs (≥ `part_size`), slices the shared `Arc<[u8]>` into part-sized chunks using `Bytes::slice(offset..end)` — no `memcpy`. Only the final sub-part tail (< `part_size`, typically ≤ 12 MiB for 16 MiB parts) is copied into the accumulation buffer. Enqueues parts to the coordinator channel (~1–70 µs per `blocking_send`).

### New: `PyBytesView` fast path in `MultipartUploadWriter.write()` (`src/python_api/python_advanced_api.rs`)

Added path 0 (before the existing `PyBuffer` path) that detects `BytesView` objects via `data.extract::<PyRef<PyBytesView>>()`, performs an Arc refcount increment (`bv.bytes.clone()` — O(1), no data movement), releases the GIL, and calls `write_bytes_blocking()`. The existing `PyBuffer` path remains as fallback for plain `bytes`, `bytearray`, `memoryview`, and NumPy arrays.

### Fix: `PyBytesView::bytes` field made `pub` (`src/python_api/python_core_api.rs`)

Required for `python_advanced_api` to Arc-clone the underlying `Bytes` without unsafe code.

### Fix: README typo (`README.md`)

Trailing `m` removed from v0.9.92 blurb (`"fixesm"` → `"fixes"`); blurb replaced with v0.9.94 highlights.

### Docs added (`docs/`)

- `CUSTOM_S3_CLIENT_ANALYSIS.md` — Analysis of AWS SDK overhead vs. lightweight reqwest client; motivates future custom SigV4 signer work
- `SHA256_SIGV4_OPTIMIZATION_ANALYSIS.md` — Deep-dive on SHA-256 hot path in SigV4 signing; benchmark data from sai3-bench → s3-ultra loopback profiling

---

## Version 0.9.92 — Post-review async safety fixes, MAX_MULTIPART_PARTS guard, clippy (April 23, 2026)

_This entry documents the second round of fixes committed on top of the coordinator-task rewrite below._

### Fixes (`src/multipart.rs`, `src/bin/cli.rs`)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🔴 Latent panic | `write()`, `write_owned()`, `flush()` called `blocking_send` from async context | Added `enqueue_part_async()` using `send().await`; rewrote all three methods to use it; added `async fn finish()` |
| 2 | 🔴 Policy | 2 clippy warnings (coordinator `loop { match }`, manual `Option::filter`) | `while let` loop in coordinator; `.filter()` in `cli.rs` |
| 3 | 🟡 UX | `MAX_MULTIPART_PARTS` (10 000) defined but never enforced | Checked in both `enqueue_part()` and `enqueue_part_async()` with a clear bail message |
| 4 | 🟡 Docs | Memory ceiling doc off by ~2× | Updated to `~2 × max_in_flight × part_size` with corrected examples |

A new unit test `test_blocking_send_panics_inside_tokio_runtime` was added to prove the pre-fix
panic class via `std::panic::catch_unwind`.

**Test count**: 580 passing (247 Rust unit tests + Python integration tests).

### CI / Release workflow updates (`.github/workflows/`)

| File | Change |
|---|---|
| `ci.yml` | Removed Python 3.10 (unsupported by our PyO3 version); added `ubuntu-24.04-arm` runner — CI now tests 3.11, 3.12, 3.13 on both x86_64 and aarch64 (6 matrix jobs) |
| `publish-pypi.yml` | Added `aarch64` to wheel build matrix; `runs-on` selects `ubuntu-24.04-arm` natively for ARM builds; result: 6 wheels published per release (3 Python versions × 2 architectures) plus sdist |

---

## Version 0.9.92 — Multipart upload: coordinator task, auto-scale max_in_flight, panic fix (April 2026)

_This entry covers work added on top of the base v0.9.92 release (connection pool / runtime changes documented below)._

### Multipart upload rewrite (`src/multipart.rs`)

The multipart upload implementation was fully rewritten to resolve two correctness issues and
improve throughput at high part counts.

**Change 1 — Coordinator task + bounded mpsc channel**

Previously each `write()` call ran `run_on_global_rt(semaphore.acquire_owned())`, crossing the
Python→Tokio boundary on every part. Under high concurrency this caused stalls because
`run_on_global_rt` parks the calling thread until the future resolves.

The new design uses a background `coordinator_task` running entirely on the Tokio runtime. It owns
a `tokio::sync::mpsc` bounded channel (capacity = `max_in_flight`). Python `write()` calls
`blocking_send()` on the channel — this parks the Python thread only when all slots are genuinely
occupied (true backpressure). The coordinator acquires the semaphore and spawns UploadPart tasks
async-natively. `finish_blocking()` sends a `Finish` sentinel and makes a single `run_on_global_rt`
call to await the coordinator, replacing one blocking cross-thread trip per part with one per upload.

**Change 2 — Auto-scale `max_in_flight`**

`MultipartUploadConfig.max_in_flight` changed from `usize` to `Option<usize>`:
- `None` (new default) → auto: `max(32, ⌈512 MiB ÷ part_size⌉)`
- `Some(n)` → explicit override, same as before

At the default 16 MiB part size this raises the default from 16 → 32 concurrent upload slots,
doubling the parallelism for callers that never passed an explicit value. At 8 MiB parts it becomes
64 slots, scaling to match the higher part count for large objects.

Python callers that pass `max_in_flight=N` explicitly (e.g. `dlio_benchmark` checkpointing, which
reads `S3DLIO_MULTIPART_MAX_IN_FLIGHT`) are unaffected — `Some(N)` is used as before.

**Change 3 — Fix latent panic in `put_object_multipart_uri_async` (`src/s3_utils.rs`)**

`put_object_multipart_uri_async` is an `async fn` called from within the Tokio runtime (via
`ObjectStore::put_multipart`), but it previously called `sink.write_blocking()` and
`sink.finish_blocking()` directly. These call `blocking_send()` and `run_on_global_rt()`, which
park a Tokio worker thread when the channel fills. For objects larger than
`max_in_flight × part_size` (~512 MiB at the old defaults) this would panic.

Fixed by wrapping the entire sink lifecycle in `tokio::task::spawn_blocking`, so blocking calls
run on a dedicated blocking thread rather than a worker thread. `run_on_global_rt` detects the
runtime context and uses `std::sync::mpsc` blocking recv, which is safe from blocking threads.

### Benchmark results (real MinIO, HTTPS, vs s3torchconnector 1.5.0)

| Object size | NP=1 | NP=4 |
|---|---|---|
| 512 MiB | 0.659 GB/s (+33% vs baseline) | 0.944 GB/s (+10%) |
| 5 GiB | 0.941 GB/s (1.10× s3torchconnector) | 1.015 GB/s (1.03×) |

### New tests

- `python/tests/test_issue134_backpressure.py` — 7 tests (integrity, OOM, concurrency, throughput)
  all passing on real MinIO
- `python/tests/test_mpu_throughput.py` — 7 throughput regression tests
- `python/tests/bench_multipart_vs_s3torchconnector.py` — benchmark harness, 5 GiB default

### Summary table

| Change | File(s) | Detail |
|--------|---------|--------|
| Coordinator task | `src/multipart.rs` | `coordinator_task` + `mpsc::bounded(max_in_flight)` replaces per-part `run_on_global_rt` |
| Auto-scale | `src/multipart.rs`, `src/python_api/python_advanced_api.rs` | `max_in_flight: usize` → `Option<usize>`; `None` = auto formula |
| Panic fix | `src/s3_utils.rs` | `put_object_multipart_uri_async` wrapped in `spawn_blocking` |
| Test compat | `tests/test_multipart.rs` | Integration test struct literals updated to `Some(n)` |

---

## Version 0.9.92 — Default HTTP/2 off, unlimited connection pool, runtime scaling, concurrency API (April 2026)

### Summary of code changes

| Fix | File(s) | Change |
|-----|---------|--------|
| 1 — Unlimited connection pool | `src/constants.rs` | `DEFAULT_POOL_MAX_IDLE_PER_HOST`: `32` → `usize::MAX`. Pool still shrinks via `pool_idle_timeout` (90 s). Doc comment explains the 794 µs root cause at high concurrency. |
| 2 — Thread cap removed | `src/s3_client.rs` | `get_runtime_threads()`: replaced `min(max(8, cores*2), 32)` with `max(4, cores)`. 28-core machine unchanged (28 threads); 96-core machine goes 32 → 96. |
| 3 — Concurrency hint API | `src/s3_client.rs`, `src/lib.rs` | Added `CONCURRENCY_HINT: AtomicUsize` static and `pub fn configure_for_concurrency(n: usize)`, exported at crate root as `s3dlio::configure_for_concurrency`. If the hint exceeds the CPU baseline, runtime thread count is `min(hint, cores*4)`. Must be called before the first S3 operation. |
| 4 — Connection pool warmup | `src/reqwest_client.rs` | Added `pub async fn warmup_connection_pool(endpoint_url: &str, connections: usize)` — fires `connections` concurrent HEAD requests to pre-fill the pool and eliminate TCP handshake storms at benchmark start. |
| 5 — HTTP/2 default off | `src/constants.rs`, `src/reqwest_client.rs` | `DEFAULT_H2C_ENABLED = false` added. `h2c_mode_from_env()` now returns `ForceHttp1` when `S3DLIO_H2C` is unset (was `Auto`). Benchmarking showed HTTP/2 reduces throughput on plain `http://` endpoints. Set `S3DLIO_H2C=1` to opt in. |

All changes compile with zero warnings; **243 tests pass**.

### Breaking change: `S3DLIO_H2C` default changed to HTTP/1.1

HTTP/2 on plain `http://` endpoints is now **off by default**. Benchmarking on loopback TCP
showed HTTP/2 reduces PUT/GET throughput compared with HTTP/1.1 and an unlimited connection pool.
The `Auto` mode (probe h2c once, fall back if rejected) is no longer the default.

| Scenario | Before v0.9.92 | v0.9.92+ |
|---|---|---|
| `S3DLIO_H2C` not set, `http://` endpoint | h2c probe once, HTTP/1.1 fallback | **HTTP/1.1 (no probe)** |
| `S3DLIO_H2C=1`, `http://` endpoint | Force h2c, no fallback | Force h2c, no fallback (unchanged) |
| `S3DLIO_H2C=0`, `http://` endpoint | Force HTTP/1.1 | Force HTTP/1.1 (unchanged) |
| Any `https://` endpoint | HTTP/2 via TLS ALPN | HTTP/2 via TLS ALPN (unchanged) |

Set `S3DLIO_H2C=1` to restore h2c for storage systems that require HTTP/2 on `http://` endpoints.
The new default is documented in `src/constants.rs` as `DEFAULT_H2C_ENABLED = false`.

### Performance: connection pool default changed to unlimited

`S3DLIO_POOL_MAX_IDLE_PER_HOST` default changed from `32` to `usize::MAX` (unlimited).

With t=128 concurrent workers and a pool cap of 32, the 96 workers that couldn't hold an idle
connection paid a full TCP handshake penalty on every request (~794 µs on loopback). This capped
throughput at ~47k ops/s externally vs ~240k ops/s in-process. With an unlimited pool, idle
connections are retained for all workers; the `S3DLIO_POOL_IDLE_TIMEOUT_SECS` (default 90 s)
eviction timer handles cleanup when load drops.

Set `S3DLIO_POOL_MAX_IDLE_PER_HOST=<n>` to impose a hard ceiling in memory-constrained environments.

### Performance: s3dlio internal runtime thread cap removed

`get_runtime_threads()` previously capped at 32 even on large machines (e.g. 96-core hosts got
32 Tokio threads). The cap is removed; the default is now `max(4, num_cpus)`. The floor of 4
prevents thread starvation on single/dual-core VMs. This primarily benefits Python/sync callers
that route through the s3dlio-internal runtime.

### New API: `configure_for_concurrency(n)`

```rust
s3dlio::configure_for_concurrency(128);
// … then call blocking S3 helpers from 128 threads …
```

Must be called before the first S3 operation. Ensures the internal Tokio runtime has enough
threads to overlap `n` concurrent requests for blocking/Python callers. Ignored by async callers
(e.g. sai3-bench) that bring their own runtime.

### New API: `reqwest_client::warmup_connection_pool(endpoint_url, n)`

```rust
s3dlio::reqwest_client::warmup_connection_pool("http://127.0.0.1:9000", 128).await;
```

Fires `n` concurrent HEAD requests to pre-fill the connection pool before benchmarking, eliminating
the TCP handshake storm at burst start that biases early-window latency measurements.

### CLI improvements (carried from in-progress work)

- `parse_human_count()`: `--num` argument accepts human suffixes (`1k`, `100m`, `1g`, binary `1ki`,
  `1mi`, `1gi`) and underscore separators (`100_000_000`).
- `parse_jobs()`: all `--jobs` / `--concurrent` arguments clamped to `MAX_JOBS = 4_096` with a
  clear error on overflow.
- `MAX_JOBS = 4_096` added to `src/constants.rs`.

---

## Version 0.9.90 - AIStore Full Support, TLS Security Fixes, HTTP/2, 5 Issues Closed (April 2026)

### Feature: Full NVIDIA AIStore support — redirects + complete TLS security (closes #126)

This release promotes AIStore support from "tacit" to **fully implemented and security-hardened**.
`RedirectFollowingConnector` now enforces all four redirect security policies when
`S3DLIO_FOLLOW_REDIRECTS=1`:

1. **Cross-host `Authorization` header stripping** (RFC 9110 §11.6.2) — always active
2. **Standard TLS cert chain + hostname verification** (rustls WebPKI) — always active
3. **HTTPS→HTTP scheme downgrade prevention** — was dead code (`cert_store: None`) since
   `make_redirecting_client()` never attached the store; now active via
   `cert_store: Some(CertVerifyStore::new())`
4. **Certificate pinning across redirect chain** — implemented via `RecordingVerifier`
   (a `rustls::client::danger::ServerCertVerifier` that delegates to WebPKI first and records
   the leaf cert DER only on success) + pre-flight TLS probe via `tokio-rustls`. This is the only
   viable approach given that the AWS Rust SDK's internal TLS chain is entirely `pub(crate)`-gated.

**Root cause of the security gaps:** both TLS policy gates in `follow_redirects()` are guarded by
`if let Some(ref store) = cert_store`. With `cert_store: None`, neither gate was ever entered —
all TLS security logic was unreachable in production. Unit tests bypassed `make_redirecting_client()`
entirely, so they passed while production was unprotected.

**New direct dependencies** (were already transitive via `aws-smithy-http-client`):
- `rustls = { version = "0.23", features = ["aws-lc-rs"] }`
- `tokio-rustls = "0.26"`
- `rustls-native-certs = "0.8"` (loads platform root CAs for production probes)

**End-to-end test coverage — all 4 redirect scenarios validated with real OS-assigned ports:**

| # | Origin | Redirect target | Result | Test |
|---|---|---|---|---|
| 1 | `https://` (real TLS server) | `http://` (real TCP port) | ❌ Refused — scheme downgrade | `end_to_end_https_to_http_redirect_refused_real_servers` |
| 2 | `https://` cert A | `https://` cert B ≠ A | ❌ Refused — cert mismatch | `end_to_end_https_cert_mismatch_refused_real_servers` |
| 3 | `https://` cert A | `https://` cert A (shared) | ✅ 200 OK — cert match | `end_to_end_https_same_cert_redirect_passes_real_servers` |
| 4 | `http://` | `http://` | ✅ 200 OK — no TLS checks | `end_to_end_http_to_http_redirect_passes_real_servers` |

Total redirect tests: **27** (13 RFC 9110 conformance, 4 TLS policy unit, 2 TLS probe integration,
4 end-to-end). See [`docs/AIStore_redirect_implementation_v0.9.90.md`](AIStore_redirect_implementation_v0.9.90.md)
for the full implementation reference including the AWS SDK API dead-end investigation.

### Bug fix: GCS delete errors silently swallowed — exit code always 0 (closes #135)
- `GcsClient::delete_objects()` logged `warn!()` for individual delete failures but returned
  `Ok(())`, causing `s3-cli rm` to report "Successfully deleted all objects" even when objects
  were not deleted. A subsequent `ls` would reveal the surviving objects.
- Fix: `anyhow::bail!()` with a descriptive message including `fail_count / total` and the bucket
  name. Errors now propagate correctly to the CLI caller and produce a non-zero exit code.

### Bug fix: `s3-cli put` silently rounds up objects < 4 KiB to 4096 bytes (closes #136)
- Small object generation clamped the payload to a 4096-byte minimum in the random data path.
  A request for a 1 KiB object silently produced a 4 KiB PUT. Fixed in the Python API layer.

### Feature: expose `list_containers()` / `list_buckets()` to Python API (closes #133)
- `list_containers(uri)` was implemented in `src/list_containers.rs` but never registered in the
  Python API. Now exported as `list_containers_py()` in `src/python_api/python_core_api.rs`.
- Returns a `list[dict]` with keys `"name"`, `"uri"`, `"creation_date"` for every container found.
- Works for all backends: `s3://`, `az://`, `gs://`, `file://`, `direct://`.

### Bug documented: multipart upload blocks Python writer thread (closes #134)
- `MultipartUploadSink::spawn_part_bytes()` acquires the concurrency semaphore via
  `run_on_global_rt(sem.acquire_owned())` — a blocking wait on the calling (Python writer) thread.
  This causes a ~3× throughput regression vs non-blocking clients (AWS CRT) for the same object
  and endpoint. The root cause, measured impact, proposed fix (coordinator task + bounded channel),
  and memory profile comparison are documented in issue #134.
- This release adds `MultipartUploadConfig` validation tests; the coordinator-task fix is
  scheduled for a follow-on PR.

---

## Version 0.9.90 - HTTP/2 & h2c Support, ForceH2c Routing Fix, TLS Test Server (April 2026)

### Code cleanup: removed dead-code modules `sharded_client` and `range_engine` (April 2026)
- Deleted `src/sharded_client.rs` — a never-completed stub that proposed splitting HTTP traffic
  across multiple `aws_sdk_s3::Client` instances to reduce per-client contention.  The premise
  is wrong for reqwest 0.13 + HTTP/2 (the connection pool is already concurrent); the
  implementation's core `get_range()` returned `Bytes::new()` (empty placeholder) and was never
  wired into any call site.  Removed `pub mod sharded_client` from `lib.rs`.
- Deleted `src/range_engine.rs` — an S3-specific range-GET engine that depended on
  `sharded_client`.  Also a stub with placeholder implementations; zero callers outside itself.
  Removed `pub mod range_engine` from `lib.rs`.
- The **production** range engine, `src/range_engine_generic.rs`, is unchanged and continues to
  be used by `file_store.rs`, `file_store_direct.rs`, and `object_store.rs` (Azure/GCS backends).
  A rationalization comment was added to `range_engine_generic.rs` explaining the history and
  noting that S3ObjectStore does not yet use this engine (future work item).
- Build is clean with zero warnings after removal.

### Documentation: `store_for_uri()` async limitation (April 2026)
- Added an in-code comment block above `store_for_uri()` in `object_store.rs` documenting:
  - Why the function is intentionally synchronous (all current backends construct without I/O).
  - The known limitation: callers passing `s3://host:port/bucket/` directly to `store_for_uri()`
    always get the global singleton client — NOT a per-endpoint isolated client — because
    `S3ObjectStore::for_endpoint()` is async and cannot be called from a sync context.
  - How `MultiEndpointStore::from_config()` works around this via `run_on_global_rt()`.
  - What a future `store_for_uri_async()` would look like if per-endpoint isolation is ever
    needed outside `MultiEndpointStore`.

### Future work items (range engine)
- `S3ObjectStore` does not yet use `range_engine_generic` for large-object downloads.  S3 range
  GETs are handled natively by the AWS SDK streaming path.  Adding an opt-in
  `enable_range_engine` flag to an `S3Config` struct (mirroring `AzureConfig`/`GcsConfig`)
  would be the natural next step if S3 workloads would benefit from explicit range splitting.

### Feature: HTTP/2 and h2c (cleartext HTTP/2) support via `S3DLIO_H2C`
- Added `S3DLIO_H2C` environment variable to control the HTTP version used by the S3 client:
  - `S3DLIO_H2C=1` — force h2c (HTTP/2 prior-knowledge) on `http://` endpoints; on `https://`
    endpoints, ALPN negotiation proceeds normally (server selects HTTP/2 or HTTP/1.1).
  - `S3DLIO_H2C=0` — force HTTP/1.1 on all endpoints.
  - Unset (default `Auto`) — probes h2c once on the first `http://` connection; falls back to
    HTTP/1.1 if the server rejects the HTTP/2 preface. On `https://`, ALPN handles version
    negotiation transparently with no extra probe.
- Implemented in `src/reqwest_client.rs` as `H2cMode` enum with `build_smithy_http_client()`.
  Two reqwest clients are built at startup (`http1_client` and `h2c_client`); the correct one is
  selected per-request based on scheme and current probe state via `select_client()`.
- CA bundle (`AWS_CA_BUNDLE`) is supported independently of the HTTP version mode.

### Bug fix: `ForceH2c` on HTTPS caused "broken pipe"
- Root cause: `H2cMode::ForceH2c` unconditionally routed all requests (including `https://`) to
  the h2c client, which sends an HTTP/2 prior-knowledge preface without TLS. TLS servers reject
  this with a broken-pipe error.
- Fix: `ForceH2c` now only routes `http://` requests to the h2c client. `https://` requests fall
  through to the standard TLS client so ALPN negotiation selects the protocol.
- Routing logic extracted into a pure `pub(crate) fn select_client(mode, is_plain_http,
  auto_state) -> ClientChoice` for testability.

### Feature: Startup INFO logging for HTTP version mode and CA bundle
- `build_smithy_http_client()` logs the resolved `H2cMode` at `INFO` on every client creation so
  operators can confirm which mode is active without needing a packet capture.
- `s3_client.rs` logs both the "CA bundle loaded from <path>" and "CA bundle not set — using
  system default TLS trust store" cases, eliminating a previous silent no-op for the not-set path.
- First response HTTP version (HTTP/2 or HTTP/1.1) is logged once via `PROTOCOL_LOGGED` atomic.

### New: `examples/tls_test_server` — local TLS+HTTP/2 test server for ALPN verification
- Added `examples/tls_test_server.rs`: a minimal HTTPS server that generates a self-signed
  certificate at runtime (via `rcgen`) for `127.0.0.1`/`localhost`, installs `aws_lc_rs` as the
  rustls crypto provider, advertises ALPN `["h2", "http/1.1"]`, and serves both HTTP/1.1 and
  HTTP/2 via `hyper_util`'s `AutoBuilder`.
- The server writes its certificate to `/tmp/tls_test_server.crt` for use by curl or `s3-cli`.
- Logs `ALPN negotiated = "h2"` (or `"http/1.1"`) and the HTTP version of every request.
- Verified end-to-end: `s3-cli stat` and `s3-cli put` against this server both log
  `HTTP protocol (first response): HTTP/2.0` and `protocol=HTTP/2` in PUT summary.
- Start it with: `cargo run --example tls_test_server`
- Test with: `AWS_CA_BUNDLE=/tmp/tls_test_server.crt AWS_ENDPOINT_URL=https://127.0.0.1:9443`
- See [`docs/HTTP2_ALPN_INVESTIGATION.md`](HTTP2_ALPN_INVESTIGATION.md) for full usage and
  expected output.
- New dev-dependencies added to `Cargo.toml`: `rcgen 0.13`, `rustls 0.23 (aws-lc-rs)`,
  `tokio-rustls 0.26`, `hyper 1 (server)`, `hyper-util 0.1 (server-auto)`.

### Tests: 10 new unit tests for HTTP version routing logic
- `test_select_client_force_h2c_plain_http` — ForceH2c + http:// → h2c client
- `test_select_client_force_h2c_tls` — ForceH2c + https:// → http1 client (the bug case)
- `test_select_client_force_http1_plain_http` — ForceHttp1 + http:// → http1 client
- `test_select_client_force_http1_tls` — ForceHttp1 + https:// → http1 client
- `test_select_client_auto_plain_http_first_connection` — Auto + http:// + Unknown → h2c probe
- `test_select_client_auto_plain_http_probe_succeeded` — Auto + http:// + H2C_AUTO_OK → h2c
- `test_select_client_auto_plain_http_probe_failed` — Auto + http:// + H2C_AUTO_FAILED → http1
- `test_select_client_auto_tls_unknown` — Auto + https:// → http1 (ALPN, any state)
- `test_select_client_auto_tls_ok` — Auto + https:// → http1 (ALPN)
- `test_select_client_auto_tls_failed` — Auto + https:// → http1 (ALPN, state irrelevant)
- All tests live in `src/reqwest_client.rs` under `#[cfg(test)]`.

### Documentation: HTTP2_ALPN_INVESTIGATION.md
- Created `docs/HTTP2_ALPN_INVESTIGATION.md`: documents the full ALPN investigation (MinIO
  always selects `http/1.1` in ALPN — its server does not accept h2), the routing logic table,
  the `tls_test_server` usage guide with expected curl and `s3-cli` output, and an index of the
  new unit tests. This doc is the canonical reference for HTTP version behavior in s3dlio.
- Updated `docs/CLI_GUIDE.md`: added `S3DLIO_H2C` to the environment variables table with
  cleartext and TLS usage examples.
- Updated `docs/api/Environment_Variables.md`: replaced stale HTTP client variables with current
  `S3DLIO_H2C` and connection pool settings, with cleartext and TLS configuration examples.

### Verification
- `cargo build --bin s3-cli` — zero warnings ✅
- `cargo test` — 225 unit tests + 38 doc tests = 263 passing, 0 failed ✅
- `s3-cli stat` against `tls_test_server` → `HTTP protocol (first response): HTTP/2.0` ✅
- `s3-cli put` against `tls_test_server` → `PUT summary: ... protocol=HTTP/2` ✅

---

## Version 0.9.86 - Redirect Follower / Tacit AIStore Support, Redirect Security (March 2026)

### Feature: HTTP 3xx redirect-following connector (`S3DLIO_FOLLOW_REDIRECTS`)
- Added `RedirectFollowingHttpClient` / `RedirectFollowingConnector` in `src/redirect_client.rs`.
  When the environment variable `S3DLIO_FOLLOW_REDIRECTS=1` is set, the S3 client wraps the
  default AWS SDK HTTP client with a connector that transparently follows 3xx redirects up to
  a configurable maximum (default: 5, override via `S3DLIO_REDIRECT_MAX`).
- The primary motivation is **NVIDIA AIStore**, which routes S3 GET/PUT/DELETE requests from a
  stateless proxy node to the specific storage target via HTTP 307 Temporary Redirect. The AWS
  SDK's default HTTP client intentionally does not follow cross-host redirects; this connector
  closes that gap. See [`docs/AIStore_307_Redirect_Proposal.md`](AIStore_307_Redirect_Proposal.md)
  for full design rationale and the rejected alternative approaches.

  > **⚠️ Note:** AIStore compatibility is *tacit* support — s3dlio has not been tested directly
  > against NVIDIA AIStore at this time. The redirect mechanism conforms to the protocol as
  > described in AIStore's documentation, but end-to-end validation is pending.

### Feature: Redirect security — scheme downgrade prevention (ACTIVE)
- HTTPS → HTTP redirects are now **refused** with a `ConnectorError`. Following an HTTPS-to-HTTP
  redirect would transmit the S3 `Authorization` header (HMAC-SHA256 signature + credentials) in
  plaintext, enabling credential capture and replay attacks.
- This protection is unconditional when `S3DLIO_FOLLOW_REDIRECTS=1` is active.

### Security gap: Certificate pinning across redirect chain (NOT YET IMPLEMENTED)
- A `CertVerifyStore` structure is in place and the redirect-chain cert-comparison policy is
  implemented and fully unit-tested — but the store is currently empty in production because there
  is no point at which the TLS handshake's peer certificate is recorded into it.
- Implementing this requires a pre-flight TLS probe using `tokio_rustls` to extract the peer
  certificate before each redirect hop. The AWS Smithy SDK's built-in TLS connector chain does not
  expose a public injection point for a custom `rustls::client::danger::ServerCertVerifier`, and
  all internal paths (`create_rustls_client_config`, `wrap_connector`) are `pub(crate)`.
- The planned approach (pre-flight probe via `aws-smithy-runtime-api`'s public `HttpConnector`
  trait) requires no SDK fork and is fully documented. See
  [`docs/security/HTTPS_Redirect_Security_Issues.md`](security/HTTPS_Redirect_Security_Issues.md)
  §7 for the complete API investigation, dead-end analysis, viable implementation path, and
  future PR checklist.
- **Risk:** MEDIUM. WebPKI validation (hostname + CA chain) continues to run on every hop.
  The scheme downgrade risk (HIGH) is now blocked. Cert pinning adds defense in depth for
  environments with non-standard CA configurations.

### Tests: 21 new unit tests for redirect behavior
- All redirect tests live in `src/redirect_client.rs` under `#[cfg(test)]`.
- Coverage includes: basic 302/307 redirect following; cross-host `Authorization` stripping
  (RFC 9110 §11.6.2); max-redirect loop termination; scheme downgrade refusal (HTTPS → HTTP);
  cert pinning match/mismatch; RFC-correct handling of 301/302 POST→GET conversion; 303 See Other;
  and the scheme-downgrade regression test that confirmed the production gap and its fix.

### Verification
- `cargo build --release` — zero warnings ✅
- `cargo test` — all tests pass ✅

---

## Version 0.9.84 - HEAD Elimination, OnceLock Caching, Lock-Free Range Assembly, Env Var Rename (March 2026)

### Performance: process-global ObjectSizeCache eliminates redundant HEAD requests
- Added `ObjectSizeCache` (TTL: 1 hour, override via `S3DLIO_SIZE_CACHE_TTL_SECS`) integrated
  into `get_objects_parallel()`. A pre-stat phase issues N concurrent HEADs before any GETs begin;
  results are stored in the process-global cache. From batch 2 onward (and all subsequent epochs),
  `get_object_uri_optimized_async()` finds the size in cache and skips its HEAD entirely. For
  typical training workloads (same files repeated across batches/epochs), this reduces S3 request
  count by up to 50%.

### Performance: OnceLock-based env var caching on the hot path
- `get_object_uri_optimized_async()` previously called `std::env::var()` on every invocation
  (one syscall per object). Three `OnceLock<T>` statics now cache `S3DLIO_ENABLE_RANGE_OPTIMIZATION`,
  `S3DLIO_RANGE_THRESHOLD_MB`, and the `ObjectSizeCache` instance once per process on first call.

### Bug fix: `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` was a no-op on the `get_many()` path
- The env var previously only applied to `S3ObjectStore::get()` in `object_store.rs`. The Python
  `get_many()` path routes through `get_objects_parallel()` → `get_object_uri_optimized_async()`
  which never checked the var. Now all paths are consistent: setting `=0` disables range splitting
  and suppresses the HEAD that was needed only to determine the threshold.

### Performance: lock-free chunk assembly in `concurrent_range_get_impl()`
- Replaced `Arc<Mutex<BytesMut>>` shared buffer (serialised writes from up to 37 concurrent chunk
  tasks) with a collect-then-sort-then-assemble pattern. Each chunk future returns
  `(buffer_offset, Bytes)` independently; results are sorted by offset and assembled in one
  sequential pass — no lock contention in the concurrent phase.

### Fix: rename `AWS_CA_BUNDLE_PATH` → `AWS_CA_BUNDLE` (standard AWS SDK name)
- `s3_client.rs`, `aws-env`, `docs/api/Environment_Variables.md`, and `python/tests/test_new_dlio_s3.py`
  updated. The previous name was non-standard; the AWS SDK uses `AWS_CA_BUNDLE`.

### Observability: replace `eprintln!` with structured tracing
- Replaced all `eprintln!("[s3dlio] ...")` calls in `s3_client.rs` with `info!()` and `debug!()`
  from the `tracing` crate. Log output is now controlled by `S3DLIO_LOG_LEVEL` / `RUST_LOG` and
  captured correctly by the Python logging integration.

### Verification
- `cargo build --release` — zero warnings ✅
- `cargo check` — zero warnings ✅

---

## Version 0.9.82 - Multipart Upload Backpressure Fix, DLIO Multipart Integration (March 2026)

### Bug fix: Multipart upload semaphore acquired before spawn (backpressure)
- `spawn_part()` and `spawn_part_bytes()` in `src/multipart.rs` previously
  acquired the concurrency semaphore *inside* the spawned task, meaning all
  parts were immediately spawned with no bound on concurrent memory usage.
  The semaphore is now acquired on the caller thread (via `run_on_global_rt`)
  **before** `spawn_on_global_rt`, providing true backpressure: at most
  `max_in_flight` tasks are ever live, capping peak memory at
  `max_in_flight × part_size` bytes. This mirrors the `_throttle()` pattern
  used by minio's Python client.

### Performance: concurrent part-joining in `finish()`
- `MultipartUploadSink::finish()` previously joined upload tasks sequentially
  (`for h in tasks { h.await }`). Changed to `futures::future::join_all(tasks)`
  so all remaining in-flight parts are awaited concurrently — reduces tail
  latency on the final flush when parts finish out of order.
- `OwnedSemaphorePermit` import moved from commented-out block to active use.

### DLIO integration: automatic multipart upload for large objects
- `S3dlioStorage.put_data()` in
  `python/s3dlio/integrations/dlio/s3dlio_storage.py` now automatically selects
  upload strategy based on object size:
  - **< 32 MiB**: single `put_bytes()` call — lowest overhead for small objects.
  - **≥ 32 MiB**: `MultipartUploadWriter` with 32 MiB parts and up to 8
    concurrent in-flight parts — avoids the S3 5 GiB single-PUT limit and
    achieves higher throughput on large checkpoints.
- Threshold constants (`_MULTIPART_THRESHOLD`, `_MULTIPART_PART_SIZE`,
  `_MULTIPART_MAX_IN_FLIGHT`) defined at module level and aligned with
  `src/constants.rs` defaults.

### Build fix: vendored OpenSSL for manylinux wheel compatibility
- Added `openssl = { version = "0.10", features = ["vendored"] }` as a direct
  dependency. The Azure SDK (`typespec_client_core`) transitively enables
  `native-tls` on `reqwest`, which pulls `openssl-sys` and requires system OpenSSL
  headers. Manylinux containers do not reliably have these installed. The vendored
  feature compiles OpenSSL from source (statically linked) so no system package is
  ever required — in any container, any OS, any CI environment.
- Set `default-features = false` on our own `reqwest` dependency (defense in
  depth — ensures s3dlio's own HTTP paths never introduce `native-tls`).
- CI `before-script-linux` already installs `openssl-devel`/`libssl-dev` as a
  fallback; the vendored fix makes this redundant but harmless.

### Verification
- `cargo build --release --features full-backends,direct-io,enhanced-http` — zero warnings ✅
- `./build_pyo3.sh` — Python 3.12 and 3.13 wheels built successfully ✅
- `cargo check --features full-backends,direct-io,enhanced-http` — zero warnings ✅

---

## Version 0.9.80 - Python List Hang Fix, Tracing Deadlock Fix, Async S3 Operations (March 2026)

### Bug fix: Python list() hung indefinitely on non-AWS endpoints (HIGH severity)
- Removed legacy `let _ = build_ops_async().await?...` calls from `list_objects()`
  and `stat_object()` in `src/s3_utils.rs`. These were leftover debug artifacts
  from an earlier `S3Ops` implementation whose results were discarded. On non-AWS
  endpoints (MinIO, on-prem S3), the AWS SDK's IMDSv2 credential probe
  (`http://169.254.169.254/…`) never times out, causing all Python `list()`,
  `list_keys()`, `list_full_uris()`, and `list_uris()` calls to hang indefinitely
  with no Ctrl-C recovery. Bug closed by this release.

### Bug fix: Tracing deadlock inside `tokio::spawn` with AWS SDK S3 operations
- Refactored `list_objects_stream` in `src/s3_utils.rs` from `tokio::spawn` +
  `mpsc::channel` + `ReceiverStream` to an inline `async_stream::stream!` macro
  (no spawned task). When `RUST_LOG=debug` or `trace` was set, the previous
  `tokio::spawn`-based implementation could deadlock due to tracing-subscriber
  lock contention between the spawned task's tracing events and the AWS SDK's
  internal tracing. The inline stream runs entirely on the caller's task,
  eliminating this class of deadlock. Bug closed by this release; also filed upstream as aws-sdk-rust#1388.

### S3ObjectStore: replace sync helpers with async variants
- `S3ObjectStore::list()` now drives `s3_list_objects_stream` (async, inline)
  instead of calling the sync `s3_list_objects()` from inside the global runtime
  (which would deadlock via nested `run_on_global_rt`).
- Added `delete_objects_async()`, `create_bucket_async()`, `delete_bucket_async()`
  to `src/s3_utils.rs` — safe to call from within async contexts.
- `S3ObjectStore::delete()`, `delete_batch()`, `delete_prefix()`,
  `create_container()`, and `delete_container()` all updated to use the new async
  variants.

### Python API cleanup
- Removed stale deprecated helpers (`list_keys_from_s3`, `list_uris`) from
  `python/s3dlio/__init__.py`; marked remaining deprecated section with TODO.

### Verification
- `cargo check` — zero warnings ✅
- All list, stat, delete operations confirmed working against MinIO ✅
- `RUST_LOG=debug` no longer causes hangs ✅

---

## Version 0.9.76 - Version fix for PyPI publish (March 2026)

- Corrected `pyproject.toml` version from `0.9.70` to `0.9.75` (was accidentally not bumped before the v0.9.75 release tag was created, causing the PyPI workflow to upload `0.9.70` wheels). No functional code changes from v0.9.75.

## Version 0.9.75 - GCS RAPID/Zonal Support, Debug Logging, RUST_LOG Hang Fix (March 2026)

### GCS RAPID / zonal bucket support
- Stat: use BidiReadObject for authoritative size metadata on RAPID buckets (StorageControl returns stale size=0).
- Get: handle OUT_OF_RANGE errors on RAPID zonal buckets by bypassing stale-metadata fallback path.
- Put: leverage `google-cloud-rust` fork with BidiWriteObject two-phase finalize to avoid PUT truncation.
- Auto-detect RAPID mode per bucket via `S3DLIO_GCS_RAPID` env var (`true` / `false` / `auto`); cached once per process.
- Pinned `google-cloud-rust` fork at `release-20260212-rf-gcsrapid-20260317.1` for PUT truncation fix.

### RUST_LOG=debug hang fix (issue #105)
- Added `crate_log_caps()` in `src/bin/cli.rs` that applies `warn`-level `EnvFilter` directives for noisy async crates (`h2`, `hyper`, `hyper_util`, `hyper_rustls`, `tonic`, `tower`, `reqwest`, `rustls`, `aws_config`, `aws_sdk_s3`, `aws_smithy_runtime`, `aws_smithy_http`).
- Prevents debug-level tracing from triggering async deadlocks inside HTTP/gRPC stacks.
- User code remains at the level requested by `RUST_LOG`; only the third-party crates are capped.

### Debug logging restored across all five backends
- `S3ObjectStore` + `s3_utils.rs`: entry/exit debug logging on all operations (get, get_range, put, put_multipart, list, stat, delete, delete_batch, delete_prefix).
- `AzureObjectStore` + `azure_client.rs`: consistent debug logging on all 12 Blob Storage methods.
- `GcsObjectStore` in `object_store.rs`: entry debug logging (inner `google_gcs_client.rs` already had 35+ statements).
- `FileSystemObjectStore` (`file_store.rs`): debug logging on all POSIX-path operations.
- `ConfigurableFileSystemObjectStore` (`file_store_direct.rs`): debug logging on all O_DIRECT operations.
- `MultiEndpointStore` and `LoggedObjectStore` intentionally not modified (delegate/TSV paths avoid duplication).

### CLI improvements
- `delete` command gains `rm` as a visible alias (`s3-cli rm ...`).
- Command naming corrected: `list` is now the primary name, `ls` is the alias (was reversed). Help output now reads `list ... [aliases: ls]`.
- Shortened `list` and `tfrecord-index` one-line help strings to prevent wrapping in terminal help output.
- `create-bucket` and `delete-bucket` commands made fully generic: now accept any URI scheme (`s3://`, `gs://`, `az://`, `file://`, `direct://`) instead of bare bucket names. Auth check is URI-scheme-aware (only S3 checks AWS credentials). Uses `store_for_uri_with_logger()` dispatch, consistent with all other commands.
- `extract_container_name(uri)` helper: pure string parser extracting the backend-appropriate container identifier from any URI scheme.
- Command help ordering: bucket commands (`create-bucket`, `delete-bucket`, `list-buckets`) listed first, remaining commands sorted alphabetically.
- 17 unit tests added for `extract_container_name()` covering all five backends and error cases (19/19 `cargo test --bin s3-cli` passing).
- `crate_log_caps()` pins noisy async crates (`h2`, `hyper`, `tonic`, `tower`, `reqwest`, etc.) to `warn` level, preventing `RUST_LOG=debug` from triggering async deadlocks in gRPC/HTTP stacks.

### Python binding fix
- Removed `LogTracer::init()` call from `python_core_api.rs` that conflicted with Python's own logging initialization.

### Issue #125 confirmed resolved
- Default build (`cargo build`) excludes optional Azure/GCS backends; `--features full-backends` enables all five.

### Integration test hygiene
- Fixed 11 integration test files that referenced feature-gated Azure/GCS structs without being feature-gated themselves. All files now carry the correct `#![cfg(feature = "backend-azure")]` or `#![cfg(feature = "backend-gcs")]` crate-level attribute.
- `tests/test_backend_parity.rs`: corrected stale assertion `assert!(azure_result.is_ok())` — wrapped Azure instantiation in `#[cfg(feature = "backend-azure")]` / `#[cfg(not(...))]` blocks; default builds now correctly verify that Azure returns an error when the feature is absent.
- `tests/test_compression_all_backends.rs` and `tests/test_object_store_integration.rs`: individual Azure/GCS tests gated with `#[cfg(feature = "backend-azure")]` to prevent spurious failures in default builds.
- `cargo test` (default features) returns zero failures across all test suites.

### GCS documentation consolidated
- Removed 5 GCS docs that documented in-progress investigation and now-resolved issues: `GCS-API-Configuration.md`, `GCS_BIDI_HANG_ANALYSIS.md`, `GCS_DEBUG_SUMMARY.md`, `GCS-gRPC_Fixes.md`, `GCS-gRPC-Transport.md`.
- Replaced with single authoritative reference: `docs/supplemental/GCS-Backend.md` — current architecture, RAPID two-phase finalize, HTTP/2 window tuning, google-cloud-rust fork commit history, API reference, performance results.

### Verification
- `cargo check --features full-backends` — zero warnings ✅
- `cargo build --release --features full-backends` — success ✅
- `cargo test` (default features) — zero failures ✅
- Manual GCS RAPID cluster test: 1000-object delete via `s3-cli -vv rm -r gs://…` — debug logging visible, `rm` alias functional ✅

---

## Version 0.9.70 - Backend-Profiled Builds, GCS Hot-Path Reuse, and RAPID Runtime Improvements (March 2026)

### Why this release was needed
- We needed to separate "default" packaging behavior from "all backends enabled" behavior.
- The Python wheel path needed a predictable profile model for maintainers and users.
- GCS hot code paths were creating lightweight wrappers frequently, adding avoidable overhead and log noise.
- RAPID mode logs were emitted too often under high concurrency, making real diagnostics harder.
- Runtime tuning around GCS channel count needed clearer wiring from CLI concurrency settings.

### Build and packaging model changes
- Added explicit backend profile behavior for Python wheel builds.
- Default Python build profile now targets a smaller S3-focused wheel.
- Full cloud backend build remains available via full-backends feature selection.
- `build_pyo3.sh` now accepts profile arguments from CLI instead of relying on environment-only control.
- Added named-flag CLI support in `build_pyo3.sh`:
    - `--profile full`
    - `--profile default`
    - `-p full`
    - `-p default`
- Positional profile form is still supported:
    - `./build_pyo3.sh`
    - `./build_pyo3.sh full`
    - `./build_pyo3.sh default`
- Added robust argument validation and help output in `build_pyo3.sh`.
- Updated `pyproject.toml` guidance/comments to align with current build/install workflows.

### Cargo feature and dependency updates
- Refined optional backend dependency usage for clearer backend selection boundaries.
- Preserved full backend behavior through `full-backends` feature.
- Updated lockfile to reflect new dependency graph resulting from feature-path updates.

### GCS performance and hot-path optimizations
- Confirmed expensive GCS transport/auth clients remain globally reused via async once-initialization.
- Added wrapper-level reuse in operational paths to avoid repeated `GcsClient::new()` wrapper churn.
- Added per-store lazy GCS wrapper caching in object-store path.
- Added global wrapper reuse for additional GCS paths that execute outside per-store reuse scope.
- Extended reuse into write finalize path so per-object write completion avoids unnecessary wrapper construction.
- Extended reuse into GCS container-list path to reduce repeated wrapper initialization in repeated listing workflows.

### RAPID mode behavior improvements
- Cached effective RAPID mode once per process lifecycle.
- Reduced RAPID mode log emission to one initialization-time message rather than per wrapper construction.
- Kept per-bucket RAPID auto-detection behavior and cache strategy intact.
- Preserved explicit override behavior from environment and API configuration.

### CLI/runtime tuning improvements
- Added/strengthened pre-tuning of GCS subchannel count before first GCS client initialization.
- Propagated concurrency intent from CLI options into GCS setup earlier in command flow.
- Added/updated constants for GCS read timeout/progress environment control.

### Documentation updates
- Updated README to explain:
    - default vs full backend build behavior,
    - explicit full-backend CLI build command,
    - Python build profile usage for `build_pyo3.sh`,
    - source-build commands for full backend Python installs.
- Updated backend options guidance to reflect current feature model.
- Added investigation docs for GCS bidi read behavior and debugging findings.
- Consolidated GCS debug document roles:
    - one document as deep technical analysis,
    - one as concise incident/debug summary with references.

### Tests and verification
- Updated tests impacted by new backend-feature defaults.
- Made list-container backend expectation tests feature-aware.
- Validation performed during this release cycle:
    - `cargo check`
    - `cargo check --features backend-gcs`
    - `cargo check --features full-backends`
    - `cargo test --lib`

### User-visible outcomes
- Cleaner default Python install footprint.
- Straightforward path to full cloud backend builds when required.
- Reduced GCS runtime overhead in hot operational paths.
- Significantly reduced RAPID log spam under concurrency.
- Better maintainability through explicit build profiles and clearer docs.

### Notes for maintainers
- Keep `build_pyo3.sh` profile UX stable (`default|slim|full`, `--profile|-p`).
- Keep README and changelog examples aligned with actual Cargo features.
- Treat GCS debug docs as a paired set:
    - technical root-cause deep dive,
    - concise debugging summary and triage notes.

## Version 0.9.65 - GCS PUT/Performance Fixes: Chunk Size, Constants, Zero-Copy (March 2026)

### GCS PUT RESOURCE_EXHAUSTED Fix
- Fixed `RESOURCE_EXHAUSTED: SERVER: Received message larger than max` errors on all GCS PUT operations
- Root cause: `DEFAULT_GRPC_WRITE_CHUNK_SIZE` was 16 MiB; GCS server's limit is **4 MiB per serialised protobuf message** (not just data payload)
- Protobuf framing overhead (~89 bytes per message) means the data payload must be smaller than the server ceiling
- New constant hierarchy in `vendor/google-cloud-gax-internal/src/gcs_constants.rs`:
  - `GCS_SERVER_MAX_MESSAGE_SIZE = 4 MiB` — raw server ceiling
  - `MAX_GRPC_WRITE_CHUNK_SIZE = GCS_SERVER_MAX_MESSAGE_SIZE - 64 KiB` — max safe data payload (63 × 64 KiB)
  - `DEFAULT_GRPC_WRITE_CHUNK_SIZE = 2 MiB` — conservative default (32 × 64 KiB), well below ceiling
- All values are 64 KiB-aligned; env-var override `S3DLIO_GRPC_WRITE_CHUNK_SIZE` silently clamped to `MAX_GRPC_WRITE_CHUNK_SIZE`
- Measured upload performance: **3.83 GB/s** for 1,000 × 32 MiB objects on RAPID bucket (32 jobs)

### GCS Constants Centralisation
- **NEW**: `vendor/google-cloud-gax-internal/src/gcs_constants.rs` — single source of truth for all GCS/gRPC protocol constants
- **NEW**: `src/gcs_constants.rs` — application-layer re-exports + channel floor, concurrent delete limit, env-var names
- All magic numbers removed from `transport.rs`, `grpc.rs`, `google_gcs_client.rs`
- `google-cloud-gax-internal` added as direct dependency in `Cargo.toml` (needed for re-exports)

### gcs-official Feature Flag Removed
- `google-cloud-storage` and `google-cloud-gax` are now always-on unconditional dependencies
- All `#[cfg(feature = "gcs-official")]` guards removed across 6 source files
- Projects depending on s3dlio no longer need `features = ["gcs-official"]`

### Zero-Copy Write Pipeline
- `put_object` and `put_object_multipart` now take `Bytes` directly — eliminates the `Bytes::copy_from_slice` on every write
- Bounded channel (capacity 8) + `tokio::spawn` producer: CRC32C now runs concurrently with gRPC network I/O
- Read path: `BytesMut::with_capacity(size_hint)` pre-allocated from `ObjectDescriptor` — zero reallocations

### RAPID Auto-Detection
- `RapidMode` enum (`Auto` | `ForceOn` | `ForceOff`) with per-bucket cache
- `get_storage_layout()` auto-detects RAPID on first access; no manual configuration required
- `S3DLIO_GCS_RAPID=auto|true|false` controls override behaviour

### Subchannel Auto-Tune
- `set_gcs_channel_count(n)` pre-init hook wired to `--jobs N` in CLI
- Three-tier priority: env var > API call > `max(64, cpu_count)` floor
- HTTP/2 window patched to 128 MiB (env `S3DLIO_GRPC_INITIAL_WINDOW_MIB`)

### New Zero-Copy Unit Tests (10 tests)
- `test_bytesmut_freeze_is_zero_copy`
- `test_bytes_clone_is_zero_copy`
- `test_bytes_slice_is_zero_copy`
- `test_bytes_from_vec_preserves_pointer`
- `test_buffered_writer_finalise_path_is_zero_copy`
- `test_grpc_read_accumulation_no_realloc`
- `test_http_read_freeze_returns_bytes_not_vec`
- `test_write_producer_chunk_slices_are_zero_copy`
- `test_producer_task_data_clone_is_arc_increment`
- `test_put_object_caller_bytes_conversion_is_zero_copy`

### Documentation
- `docs/supplemental/GCS-Backend.md` — comprehensive combined doc covering all 6 root-cause issues, all fixes, constants architecture, zero-copy data flow, performance results (3.83 GB/s), env-var reference, files-changed table (formerly `GCS-gRPC_Fixes.md`)
- `docs/performance/RANGE_OPTIMIZATION_IMPLEMENTATION.md` — clarified as S3-only; added §Backend-Specific Read Architectures comparing S3/GCS/Azure/File transport layers

### GCS API Completion (read-back, query, Python bindings)
- **NEW**: `s3dlio::get_gcs_channel_count() -> usize` — read back the programmatically-configured subchannel count (`0` = not set, auto-detect will apply)
- **NEW**: `s3dlio::get_gcs_rapid_mode() -> Option<bool>` — read back the current effective RAPID mode, resolving env var + programmatic override (`None` = auto-detect)
- **NEW**: `s3dlio::query_gcs_rapid_bucket(bucket_or_uri) -> bool` (async) — query whether a bucket or `gs://` URI is RAPID (Hyperdisk ML); result cached per bucket for process lifetime, deduplicated under concurrent access
- **NEW** Python API: `gcs_set_channel_count`, `gcs_set_rapid_mode`, `gcs_get_channel_count`, `gcs_get_rapid_mode`, `gcs_query_rapid_bucket` — full GCS tuning control from Python without requiring environment variables
- `docs/supplemental/GCS-Backend.md` — complete reference for all 5 public GCS API functions, constants, env vars, and Python usage (see §Runtime Configuration)
- `.github/copilot-instructions.md` — added Prime Directive and Secondary Directive

---

## Version 0.9.60 - GCS gRPC Transport, RAPID Storage & Multi-Protocol List Buckets (February 2026)

### GCS gRPC Transport (All Operations)
- **All GCS reads** now use gRPC `BidiReadObject` via `open_object()` — replaces JSON/REST `read_object()`
- **All GCS writes** now use gRPC `BidiWriteObject` via `send_grpc()` — replaces JSON/REST multipart uploads
- Chunked writes with 2 MiB default chunk size (`S3DLIO_GRPC_WRITE_CHUNK_SIZE` to override)
- Per-chunk CRC32C checksums via `ChecksummedData` + whole-object CRC32C on final message
- TRACE-level logging for gRPC write chunk size, count, offsets, and CRC32C values
- See [docs/supplemental/GCS-Backend.md](supplemental/GCS-Backend.md) for full architecture details

### GCS RAPID / Hyperdisk ML Storage Support
- Vendored `google-cloud-storage` v1.8.0 fork with gRPC write support:
  - `stub.rs`: `write_object_grpc()` trait method
  - `transport.rs`: Full `BidiWriteObject` implementation (~200 lines)
  - `write_object.rs`: `send_grpc()` + `set_appendable()` builder methods
  - `perform_upload.rs`: `appendable=true` query parameter for legacy JSON path
- New env var `S3DLIO_GCS_RAPID=true|1|yes` enables `appendable=true` on writes
- `.env` file support via `dotenvy` for persistent configuration
- gRPC reads work with RAPID/zonal buckets automatically (no special flag needed)

### GCS Concurrent Batch Deletes
- `delete_objects()` now dispatches up to **64 concurrent** gRPC delete requests
- Uses `FuturesUnordered` + `Semaphore` for bounded parallelism
- ~35x speedup for batch deletes (859 objects: ~69s → ~2s)
- Errors collected and reported after all requests complete (no early abort)

### GCS List Prefix Normalization Fix
- Fixed bug where `list_objects()` appended `/` to exact object names (e.g. `object.dat/`)
- Now retries without trailing slash when normalized prefix returns empty results
- Extracted reusable `list_objects_with_prefix()` paginated helper

### List Buckets Multi-Protocol (Issue #121)
- Fixed hardcoded `us-east-1` for custom S3 endpoints in `list_buckets()`
- New `list_containers(uri)` function dispatches across S3, Azure, GCS, and local fs
- CLI `list-buckets` command now accepts optional URI argument for non-S3 backends

### Range GET Optimization Defaults Changed
- `S3DLIO_ENABLE_RANGE_OPTIMIZATION` is now **enabled by default**; set to `0` to disable
- Default threshold changed from 64 MB → **32 MB** (`S3DLIO_RANGE_THRESHOLD_MB`)
- Setting the variable to `1` is still accepted (no-op, already enabled)

### Build & Dependencies
- `gcs-official` (gRPC) is now the default GCS backend; `gcs-community` demoted
- `hdf5` and `numa` (hwloc) are now **optional** Cargo features — removes `libhdf5-dev` and `libhwloc-dev` as build dependencies
- Vendor fork wired via `[patch.crates-io]` in `Cargo.toml`

### Other
- Comprehensive `trace!`/`debug!` instrumentation across GCS, Azure, S3 list paths
- Doc-test fence annotations updated; `doctest-threads = 1` to prevent memory thrash
- Consolidated 4 GCS design docs into single `docs/GCS-gRPC-Transport.md` (subsequently merged into `docs/supplemental/GCS-Backend.md` in v0.9.75)

---

## Version 0.9.50 - Critical Python Runtime Fix & s3torchconnector Compat (February 2026)

### 🚨 **CRITICAL: Python Multi-Threaded Runtime Fix**

**Fixed two catastrophic bugs** that prevented s3dlio from being used in multi-threaded Python applications:

| Bug | Version | Symptom | Root Cause |
|-----|---------|---------|------------|
| Runtime Churn | v0.9.27 | `RuntimeError: dispatch failure` after ~40 objects | Every API call created a new Tokio runtime (320 calls = 256+ OS threads) |
| Nested Runtime | v0.9.40 | `PanicException: Cannot start a runtime from within a runtime` | `GLOBAL_RUNTIME.block_on()` panics when called from Tokio-managed threads |

**Solution — io_uring-style Submit Pattern:**

Replaced all `block_on()` calls with a spawn+channel pattern that works from ANY thread context:

```
Python Thread → handle.spawn(async work) → mpsc::channel.recv() → result
```

- The calling thread blocks on `channel.recv()` (NOT `block_on()`), so there are no Tokio runtime conflicts
- A single global Tokio runtime (created once via `once_cell::Lazy`) handles all async I/O
- `submit_io<F, T>()` wrapper converts `anyhow::Result<T>` → `PyResult<T>` for clean error propagation
- Zero-copy: `Bytes` (Arc-based) moves through the channel — no data copied

**Files Changed:**
- `src/python_api/python_core_api.rs` — Removed `GLOBAL_RUNTIME`, added `submit_io()`, updated 12 functions
- `src/python_api/python_aiml_api.rs` — Updated 2 functions to use `run_on_global_rt()`

**Test Result:** 16 threads × 200 objects × 3 rounds — ALL OPERATIONS PASSED (ThreadPoolExecutor)

### ✨ **New: `put_many()` Batch Upload**

Upload multiple objects in a single call with parallel execution:

```python
items = [
    ("s3://bucket/file1.bin", b"data1"),
    ("s3://bucket/file2.bin", b"data2"),
    ("s3://bucket/file3.bin", b"data3"),
]
s3dlio.put_many(items)

# Async version
await s3dlio.put_many_async(items)
```

### 🔧 **s3torchconnector Compatibility Layer Rewrite**

Complete rewrite of `python/s3dlio/compat/s3torchconnector.py` for zero-copy performance:

| Component | Before (Broken) | After (v0.9.50) |
|-----------|-----------------|------------------|
| `S3Client.get_object` (range) | Downloaded entire object + `bytes()` copy | Uses `get_range()` — server-side range, returns BytesView zero-copy |
| `S3Client.put_object` | `list[bytes]` + `b''.join()` (two copies) | Accumulates in BytesIO, single `put_bytes()` call |
| `S3Checkpoint.writer` | `self.buffer.read()` (copy) | `getbuffer()` memoryview — no copy |
| `S3Checkpoint.reader` | `io.BytesIO(bytes(data))` (two copies) | New `_BytesViewIO(io.RawIOBase)` wrapping BytesView via memoryview, wrapped in `io.BufferedReader` |
| `S3IterableDataset.__iter__` | Lost URIs (placeholder base_uri) | Pre-lists via `list()`, iterates with full URIs |
| `S3MapDataset` | Depended on `ObjectStoreMapDataset._keys` private attr | Uses `list()` + `get()` directly |
| `S3Client.list_objects` | Called non-existent `list_prefix` | Uses `list(uri, recursive=True)` |
| Module imports | Per-class `from .. import _pymod` | Single module-level `from .. import _pymod as _core` |
| `S3Item.close()` | No-op | Sets `_data = None` to release Rust Bytes Arc reference early |

**New class: `_BytesViewIO(io.RawIOBase)`** — Zero-copy seekable file-like wrapper around BytesView. `readinto()` uses `memoryview` for zero-copy slice into caller's buffer. Used by `S3Checkpoint.reader()` so `torch.load()` can read directly from Rust memory.

### � **Range Download Optimization (Opt-In)**

Parallel range downloads for large S3 objects, achieving **76% performance improvement** for 148 MB objects:

**New Environment Variables:**
- `S3DLIO_ENABLE_RANGE_OPTIMIZATION` — Opt-in flag (default: disabled to avoid HEAD overhead on small objects)
- `S3DLIO_RANGE_THRESHOLD_MB` — Minimum object size to trigger parallel download (default: 64 MB)

**Performance (Measured with 16x 148 MB objects on MinIO):**

| Threshold | Time | Throughput | Speedup |
|-----------|------|------------|---------|
| Disabled (baseline) | 5.52s | 429 MB/s (0.42 GB/s) | 1.00x |
| 8 MB | 3.50s | 676 MB/s (0.66 GB/s) | **1.58x** (58% faster) |
| 16 MB | 3.27s | 725 MB/s (0.71 GB/s) | **1.69x** (69% faster) |
| 32 MB | 3.23s | 732 MB/s (0.71 GB/s) | **1.71x** (71% faster) |
| **64 MB (default)** | **3.14s** | **755 MB/s (0.74 GB/s)** | **1.76x** (76% faster) 🏆 |
| 128 MB | 3.22s | 735 MB/s (0.72 GB/s) | **1.71x** (71% faster) |

**Key Findings:**
- 64 MB threshold (default) is optimal for 148 MB objects
- Sweet spot: 16-64 MB thresholds provide 69-76% faster downloads
- Even aggressive 8 MB threshold shows 58% improvement
- HEAD overhead (~10-20ms) is well amortized by parallel download
- **Fixes issue** where s3dlio was 25% slower than s3torchconnector for 148 MB objects — now 61-76% faster

**Implementation:**
- `S3ObjectStore::get()` now uses `get_object_uri_optimized_async()` when enabled
- S3 backend now matches Azure/GCS parallel range download capability
- Conservative opt-in design: disabled by default to avoid HEAD overhead on small objects

**Python Usage:**
```python
import os
os.environ['S3DLIO_ENABLE_RANGE_OPTIMIZATION'] = '1'
os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '64'  # Objects ≥ 64 MB use parallel ranges

import s3dlio
data = s3dlio.get("s3://bucket/checkpoint-148mb.bin")  # 76% faster!
```

**Files Changed:**
- `src/s3_utils.rs` — Updated threshold default from 4 MB to 64 MB, added enable check
- `src/object_store.rs` — Wired S3ObjectStore::get() to use optimized path when enabled

### ⚡ **Multipart Upload Performance Improvements**

Zero-copy chunking and non-blocking spawn for streaming multipart uploads:

**Optimizations:**
- **Zero-copy chunking**: Use `Bytes::slice()` instead of `Vec::to_vec()` for part extraction
- **Non-blocking spawn**: Use `spawn_on_global_rt()` instead of `run_on_global_rt()` — eliminates double-spawn overhead
- **New `spawn_part_bytes()` method**: Direct Bytes handling without Vec<u8> conversion
- **No more Python thread blocking**: Upload tasks spawn immediately and return JoinHandle

**Performance Impact:**
- Eliminates one full data copy per multipart chunk (16-64 MB chunks)
- Removes blocking wait during task spawn (was: channel send/recv overhead)
- Python thread now submits upload and continues immediately

**Files Changed:**
- `src/multipart.rs` — Zero-copy Bytes::slice() in write_owned() and write_owned_blocking(), new spawn_part_bytes()
- `src/s3_client.rs` — Added spawn_on_global_rt() helper function

### 📚 **Documentation**

- Rewrote `docs/PYTHON_API_GUIDE.md` — complete API reference for v0.9.50
- Updated `docs/S3TORCHCONNECTOR_MIGRATION.md` — reflects zero-copy compat layer
- Updated `docs/api/Environment_Variables.md` — Full threshold comparison table with actual benchmark data
- Updated `docs/performance/MultiPart_README.md` — Large object download section with performance results
- **New**: `docs/performance/RANGE_OPTIMIZATION_IMPLEMENTATION.md` — Complete implementation summary

### 🏗️ **Architecture**

**Global Client Cache (DashMap):**
- `StoreKey(scheme, endpoint, region) → Arc<dyn ObjectStore>`
- >99% cache hit rate, <100ns lookup (DashMap sharded locking)
- Process-global and automatic — no manual client passing required
- Superior to MinIO's `urllib3.PoolManager(maxsize=10)` per-instance pattern

**Zero-Copy Data Flow:**
```
Rust async I/O → Bytes (Arc) → mpsc channel → PyBytesView (buffer protocol)
                                                    ↓
                    memoryview / np.frombuffer / torch.frombuffer (zero-copy)
```

---

## Version 0.9.40 - Critical Performance Fix & Zero-Copy Architecture (February 2026)

### 🚀 **Critical Performance Fix**

**Zero-Copy DataBuffer Architecture:**
- **FIXED**: 48x performance regression in Python data generation (was 0.8 GB/s, now 42+ GB/s)
- Eliminated `bytes::Bytes` intermediate copy in `generate_into_buffer()` and `generate_data()`
- Changed from `generate_data_with_config()` → `generate_data()` to return `DataBuffer` directly
- Python buffer protocol now exposes `DataBuffer` raw pointer (zero-copy NUMA memory access)
- **Performance**: Achieved parity with dgen-py (41.3 GB/s vs 42.5 GB/s = 2.9% difference)

**New Streaming Generator API:**
- Added `Generator` class to Python API matching dgen-py's streaming interface
- Allows efficient chunk-by-chunk generation with state reuse
- Avoids expensive generator recreation for repeated operations
- Recommended for benchmarking and large-scale data generation

**NUMA/Thread-Pinning Enabled by Default:**
- Enabled `numa` and `thread-pinning` features in default build
- Changed dependency from `hwloc2` to `hwlocality = "1.0.0-alpha.11"`
- Fixed `hardware.rs` to use `NumaTopology::detect()` instead of legacy API
- Automatic NUMA-aware allocation for >16 MB buffers on NUMA systems

**Build Optimizations:**
- Added release profile: `lto=true`, `codegen-units=1`, `opt-level=3`
- Matches dgen-rs optimization level for consistent performance

### 📚 **Documentation & Testing Improvements**

**Python Bytearray Feature Documentation:**
- Enhanced documentation for `generate_into_buffer()` bytearray support
- Added comprehensive performance benchmarks showing 2.5-3.0x speedup with pre-allocated buffers
- Documented best practices for streaming workflows and memory-efficient generation
- Added usage patterns: when to use bytearray vs on-demand allocation

**New Test Suite:**
- Added `test_bytearray_allocation.py` - comprehensive bytearray feature testing
- Added `compare_generators.py` - fair performance comparison with dgen-py using streaming API
- 5 new test cases covering: basic allocation, reuse patterns, performance comparison, various sizes, NumPy compatibility
- All tests passing with excellent performance (2.98x speedup measured with bytearray reuse)

**Updated Documentation:**
- Enhanced `ZERO_COPY_IMPLEMENTATION.md` with bytearray allocation section and DataBuffer architecture
- Performance comparison examples with real benchmarks (16 GB generation in <0.4s)
- Streaming generation patterns with reusable buffers
- Memory efficiency guidelines for large-scale data generation

**Test Status:**
- **Rust tests**: 186/186 passing (100%)
- **Python tests**: All existing + new tests passing
- Zero warnings policy maintained

**Architecture Note:**
- This version prepares for future migration to `dgen-data` crate (published on crates.io)
- Current implementation uses internal `data_gen_alt.rs` module
- Future versions may depend on `dgen-data` to eliminate code duplication

---

## Version 0.9.39 - s3dlio-oplog API Fix (February 2026)

### 🐛 **Bug Fix**

**s3dlio-oplog compatibility:**
- Fixed `s3dlio-oplog` crate to use new `data_gen_alt::generate_controlled_data_alt()` API
- Removed usage of deprecated `generate_controlled_data()` function
- Improved zero-copy performance: `generate_controlled_data_alt()` returns `Bytes` directly

**Impact:** Fixes compilation errors when using s3dlio-oplog with s3dlio v0.9.38+

---

## Version 0.9.38 - Critical Compression Fixes (February 2026)

### 🐛 **Critical Bug Fixes**

**Data Generation Compression:**
- Fixed zero-prefix compression calculation for objects <1 MiB
- Fixed small object alignment in `generate_controlled_data`
- All object sizes now compress to exact target ratios

**GCS Backend:**
- Fixed GCS API compatibility (removed non-existent `list_objects_stream()` calls)
- Replaced with `list_objects()` and manual chunking

**Code Quality:**
- Fixed all clippy warnings (zero-warning policy maintained)
- Added missing benchmark configurations
- Repository cleanup

---

## Version 0.9.37 - Test Suite Modernization (January 2026)

### 🧹 **Build & Test Cleanup**

Comprehensive test suite modernization and build cleanup to achieve zero warnings.

**Changes:**
- Updated 25+ test files to use `bytes::Bytes::from()` for `ObjectStore::put()` calls
- Added `#[ignore]` to slow benchmark tests with proper run instructions in comments
- Deprecated `performance_comparison.rs` (old vs new algorithm comparison)
- Added `#[allow(deprecated)]` where legacy APIs are intentionally tested
- Fixed clippy warning in `object_format_tests.rs` (redundant clone)
- Fixed `test_comprehensive_streaming.rs` to use `ObjectGenAlt` API correctly
- Removed obsolete `test_data_generation_enhancement.rs` (two-pass comparison)

**New Test Files:**
- `test_datagen_performance_validation.rs` - Validates 35+ GB/s generation throughput
- `test_high_speed_data_gen.rs` - Comprehensive data generation benchmarks

**Deprecated Function Suppression:**
- Added `#[allow(deprecated)]` to re-exports and call sites for backward compatibility
- `generate_controlled_data()` remains available but deprecated (use `data_gen_alt` APIs)

**Build Status:** Zero warnings in library, binary, and test targets.

---

## Version 0.9.36 - BREAKING: Zero-Copy API (January 2026)

### ⚠️ **BREAKING CHANGE: ObjectStore API Zero-Copy Conversion**

**Critical Performance Fix**: Eliminated internal `.to_vec()` call that forced memcpy, defeating zero-copy architecture.

**What Changed**:
```rust
// OLD API (v0.9.35 and earlier)
pub trait ObjectStore {
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()>;
    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()>;
}

// NEW API (v0.9.36+) - Zero-copy throughout
pub trait ObjectStore {
    async fn put(&self, uri: &str, data: Bytes) -> Result<()>;
    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()>;
}
```

**Why This Matters**:
- **Eliminated Hidden Copy**: Previous API took `&[u8]` but internally converted to `Vec<u8>` via `.to_vec()`, forcing memcpy
- **True Zero-Copy**: `Bytes` is reference-counted (Arc-like) - no memcpy when passed, cloned, or sliced
- **Maintains Performance**: 20-40 GB/s streaming writes now achievable without internal copies
- **Consistent Architecture**: Matches get() API which already returns `Bytes`

**Migration Guide**:
```rust
// Before (v0.9.35):
let data = vec![0u8; 1024];
store.put("s3://bucket/key", &data).await?;

// After (v0.9.36):
let data = Bytes::from(vec![0u8; 1024]);  // Or Bytes::copy_from_slice(&data) for &[u8]
store.put("s3://bucket/key", data).await?;
```

**Zero-Copy Conversions**:
- `Bytes::from(vec)` - Takes ownership (zero-copy, just wraps Arc)
- `Bytes::copy_from_slice(&[u8])` - Allocates new buffer (one copy, unavoidable)
- `bytes.as_ref()` → `&[u8]` - Zero-copy view (fat pointer only)
- `bytes.slice(range)` - Zero-copy subslice (shares Arc)

**Internal Changes**:
- S3: `Bytes` → `ByteStream` via `.into()` (zero-copy)
- Azure: `Bytes` passed directly to SDK (zero-copy)
- GCS: `Bytes` → `&[u8]` via `.as_ref()` (zero-copy view)
- File/Direct: `Bytes` → `&[u8]` via `.as_ref()` (zero-copy view)
- MultiEndpoint: `Bytes` cloned cheaply (reference count, not data)

### ✨ **New: `fill_controlled_data()` - In-Place Buffer Filling**

**High-performance data generation without allocation** (86-163 GB/s with Rayon parallelism):

**Key Innovation**: Fill pre-allocated buffers instead of allocating new `Vec<u8>`:
```rust
/// Fill existing buffer with controlled data (ZERO-COPY workflows)
pub fn fill_controlled_data(buf: &mut [u8], dedup: usize, compress: usize)
```

**Why This Matters**:
- **Streaming Workflows**: Pre-allocate buffers once, reuse for entire stream
- **Zero-Copy Architecture**: Works with `BytesMut` → `Bytes` pipeline
- **Rayon Pool Control**: Respects `ThreadPool::install()` context (unlike allocation-based methods)
- **86-163 GB/s**: Parallel generation scales across all cores

**Usage Example**:
```rust
use bytes::BytesMut;
use s3dlio::fill_controlled_data;

// Allocate buffer once
let mut buf = BytesMut::zeroed(4 * 1024 * 1024);  // 4 MB

// Fill in-place (no additional allocation)
fill_controlled_data(&mut buf, 1, 1);  // Incompressible, no dedup

// Convert to Bytes (zero-copy)
let data: Bytes = buf.freeze();
```

**Benefits**:
- **Memory Efficiency**: Reuse buffers in streaming loops
- **Performance**: 20-50x faster than deprecated `generate_controlled_data_prand()`
- **Thread Control**: Works with custom Rayon thread pools

### 🚨 **Deprecated: `generate_controlled_data_prand()`**

**Old slow method marked deprecated** - replaced by `fill_controlled_data()`:

```rust
#[deprecated(since = "0.9.36", note = "Use fill_controlled_data() instead - MUCH faster (86-163 GB/s vs 3-4 GB/s)")]
pub fn generate_controlled_data_prand(size: usize, dedup: usize, compress: usize) -> Vec<u8>
```

**Performance Comparison**:
- **Old**: `generate_controlled_data_prand()` - 3-4 GB/s (sequential, Vec allocation)
- **New**: `fill_controlled_data()` - 86-163 GB/s (parallel Rayon, in-place)

**Migration**:
```rust
// OLD (deprecated):
let data = generate_controlled_data_prand(size, 1, 1);

// NEW (recommended):
let mut buf = BytesMut::zeroed(size);
fill_controlled_data(&mut buf, 1, 1);
let data = buf.freeze();
```

**Note**: `DataGenAlgorithm::Prand` enum variant removed - all code now uses high-performance Random algorithm.

---

## Version 0.9.35 - Runtime Hardware Detection API & Data Generation Optimization (January 2026)

### 🔍 **New: Public Hardware Detection API**

**Key Innovation**: Hardware detection is now **always available** at runtime, regardless of compile-time feature flags.

**Why This Matters**:
- Binaries built on one system automatically optimize for hardware on another system
- No more "build with NUMA flag" requirements - detection happens at runtime
- External tools (sai3-bench, dl-driver, etc.) can leverage the same hardware detection API
- Single binary works optimally from single-core VMs to multi-socket NUMA servers

**New Public API** (`s3dlio::hardware` module):
```rust
// CPU Detection (always available, zero dependencies on Linux)
pub fn get_affinity_cpu_count() -> usize;         // CPUs available to this process
pub fn total_cpus() -> usize;                      // Total system CPUs
pub fn is_numa_available() -> bool;                 // NUMA hardware detection
pub fn recommended_data_gen_threads() -> usize;     // Optimal thread count
```

**Usage Example**:
```rust
use s3dlio::hardware;

// Automatically scale to available hardware
let threads = hardware::recommended_data_gen_threads();
println!("Using {} threads for data generation", threads);

// Respects cgroup limits, taskset, Docker CPU constraints
let cpus = hardware::get_affinity_cpu_count();

// Check NUMA availability at runtime
if hardware::is_numa_available() {
    println!("NUMA topology detected - optimizing placement");
}
```

### ⚡ **Enhanced: Data Generation Optimization**

**Backported from dgen-rs** high-performance data generation library:

**Optimal Defaults**:
- **1 MB block size** - Optimal CPU cache utilization (was 4 MB)
- **All available cores** - Auto-detected via hardware API (was 50%)
- **NumaMode::Auto** - Runtime NUMA adaptation
- **51.09 GB/s validated** - 100 GB in 1.96s on 12-core system

**New GeneratorConfig Fields** (all optional, backward compatible):
```rust
pub struct GeneratorConfig {
    // Existing fields...
    pub numa_node: Option<usize>,    // NEW: Pin to specific NUMA node
    pub block_size: Option<usize>,   // NEW: Override block size (1-32 MB)
    pub seed: Option<u64>,           // NEW: RNG seed for reproducibility
}
```

**Example - Reproducible Data Generation**:
```rust
use s3dlio::data_gen_alt::generate_controlled_data_alt;

// Generate with explicit seed for reproducibility
let data = generate_controlled_data_alt(
    100 * 1024 * 1024,  // 100 MB
    1,                   // No deduplication
    1,                   // Incompressible
    Some(12345),        // NEW: Reproducible seed
);
```

### 🐍 **Updated: Python Bindings**

**All Python APIs updated** to support new features:
- Added seed parameter support
- Updated GeneratorConfig with new fields
- Maintained zero-copy buffer protocol
- Successfully builds: `s3dlio-0.9.35-cp313-cp313-manylinux_2_39_x86_64.whl`

### 📚 **Documentation**

**New Guides**:
- `docs/Hardware_Detection_API.md` (429 lines) - Complete API reference and examples
- `docs/Data_Generation_Performance.md` (198 lines) - Performance analysis and tuning
- `examples/hardware_detection.rs` - Practical usage demonstration

### 🧪 **Testing**

**New Tests**:
- `test_cpu_utilization.rs` (119 lines) - Validates 50+ GB/s performance
- `test_allocation_overhead.rs` (38 lines) - Memory efficiency validation
- `test_optimal_chunking.rs` (99 lines) - Cache optimization verification

**Results**: ✅ 178 library tests passing, zero warnings

### 🔧 **Code Quality Improvements**

**Fixed**:
- Removed unused imports in 6 test files
- Fixed `test_rmdir_on_nonexistent_directory_is_idempotent` to match idempotent design
- Updated benchmark `data_gen_comparison.rs` with correct API signatures
- Removed non-existent `s3_backend_comparison` example reference
- Zero compiler warnings (verified with `cargo clippy`)

### 📊 **Performance Impact**

**Data Generation**:
- Throughput: **51.09 GB/s** maintained (100 GB in 1.96s)
- CPU Utilization: ~100% across all available cores
- Memory: Zero additional allocations in hot path
- Startup: <1ms overhead for hardware detection

**Binary Size**: +15 KB (hardware detection code)

### 🔄 **Backward Compatibility**

**Breaking Changes**: ✅ **NONE**

**Migration**: ✅ **NOT REQUIRED** - All existing code works unchanged

**Optional Enhancements**:
```rust
// Before (still works):
let data = generate_controlled_data_alt(size, dedup, compress);

// After (with explicit seed):
let data = generate_controlled_data_alt(size, dedup, compress, Some(42));
```

### 🎯 **Use Cases**

**For Benchmarking Tools** (sai3-bench, dl-driver):
```rust
use s3dlio::hardware;

// Automatically scale data generation to available hardware
let threads = hardware::recommended_data_gen_threads();
config.set_threads(threads);
```

**For Reproducible Testing**:
```rust
// Generate identical data across runs for validation
let data = generate_controlled_data_alt(size, 1, 1, Some(fixed_seed));
```

**For Container Environments**:
```rust
// Respects cgroup CPU limits automatically
let cpus = hardware::get_affinity_cpu_count();  // Respects --cpus=4 in Docker
```

See `examples/hardware_detection.rs` for complete examples.

---

## Version 0.9.34 - NUMA-Aware Data Generation (January 2026)

### 🚀 **Enhanced Data Generation with NUMA Optimization**

**Intelligent CPU Allocation for Data Generation + I/O Workloads**:

**Key Features**:
- **Smart Default**: Uses 50% of available CPUs for data generation, leaving 50% for I/O operations
- **Auto NUMA Detection**: Automatically detects UMA (single-node) vs NUMA (multi-socket) systems
- **Thread Pinning**: Pins threads to specific CPU cores on NUMA systems for better cache locality
- **First-Touch Initialization**: Memory allocated on local NUMA node for optimal performance
- **Zero Breaking Changes**: All existing APIs work unchanged

**New Components**:
- `src/numa.rs`: NUMA topology detection module (Linux)
  - Detects number of NUMA nodes, physical cores, logical CPUs
  - Reads from `/sys/devices/system/node` and `/proc/cpuinfo`
- `NumaMode` enum: Auto (default) or Force (testing)
- `GeneratorConfig` struct: Fine-grained control over data generation
- Helper functions:
  - `default_data_gen_threads()`: Returns 50% of available CPUs
  - `total_cpus()`: Returns total logical CPUs available

**Performance Benefits**:
- **NUMA Systems**: Thread pinning + first-touch reduces cross-node memory access
- **All Systems**: Reserves CPU capacity for concurrent I/O operations
- **Balanced Workloads**: Optimal for PUT operations (data gen + upload)

**Optional Cargo Features**:
- `numa`: Enable NUMA topology detection (requires hwloc2)
- `thread-pinning`: Enable CPU core affinity (requires core_affinity)

**Example Usage**:
```rust
// Use defaults (50% CPUs, auto NUMA detection)
let data = generate_controlled_data_alt(100 * 1024 * 1024, 1, 1);

// Override to use all CPUs
let config = GeneratorConfig {
    max_threads: Some(total_cpus()),
    ..Default::default()
};
let data = generate_data_with_config(config);
```

**New Tests** (8 added):
- `test_default_thread_count`: Verify 50% CPU allocation
- `test_generator_config_defaults`: Verify default configuration
- `test_detect_topology`: NUMA topology detection (when feature enabled)
- 5 existing tests updated for new defaults

**Dependencies Added**:
- `hwloc2 = "2.2"` (optional, for NUMA)
- `core_affinity = "0.8"` (optional, for thread pinning)

---

## Version 0.9.33 - Clippy Cleanup (December 26, 2025)

### 🧹 **Code Quality: Zero Clippy Warnings**

**Comprehensive clippy cleanup across library and binary targets**:

**Phase 1 Fixes** (69 warnings resolved):
- Fixed `empty_line_after_doc_comments` (3 instances)
- Fixed `unused_unit` in profiling macros
- Fixed `redundant_field_names` (key: key → key)
- Fixed `derivable_impls` (7 enums with #[derive(Default)])
- Fixed `collapsible_if` (5 instances)
- Fixed `manual_div_ceil` (2 instances with .div_ceil())
- Fixed `manual_clamp` (2 instances with .clamp())
- Fixed `let_unit_value` warnings (5 instances)
- Fixed `writeln_empty_string` warnings (3 instances)
- Fixed `clone_on_copy`, `unnecessary_cast`, `unnecessary_map_or`
- Implemented FromStr trait properly for Strategy enum

**Phase 2 Fixes** (additional cleanup):
- Fixed `manual_strip`: Use strip_prefix/strip_suffix methods
- Fixed `explicit_counter_loop`: Use zip with enumerate for s3_logger
- Fixed `manual_clamp`: Replace max().min() with clamp()
- Fixed `manual_ok`: Replace match Ok/Err patterns with .ok()
- Fixed `needless_range_loop`: Use iterator patterns
- Fixed `if_same_then_else`: Simplify redundant conditional branches
- Fixed `doc_lazy_continuation`: Add proper indentation
- Fixed `redundant_pattern_matching`: Use is_err() methods
- Fixed unused imports: Remove Hash, keep BuildHasher for hash_one
- Fixed `too_many_arguments`: Add allow annotations where necessary
- Applied auto-fixes for 51 additional warnings

**Result**:
- ✅ **Library (lib)**: Zero warnings (was 69)
- ✅ **Binary (s3-cli)**: Zero warnings (was 2)
- ⏭️ **Tests**: 32 warnings remain (deferred to future work)

**Files Modified**: 16 files (src/object_store.rs, src/file_store.rs, src/file_store_direct.rs, src/data_gen.rs, src/s3_logger.rs, src/mp.rs, src/bin/cli.rs, and others)

---

## Version 0.9.32 - Memory-Efficient delete_prefix() (December 2025)

### 🐛 **Fixed: delete_prefix() Memory Bloat with Millions of Objects**

**Problem**: The `delete_prefix()` method loaded ALL object keys into memory before deletion:
```rust
let keys = client.list_objects(&bucket, Some(&key_prefix), true).await?;  // BAD!
client.delete_objects(&bucket, keys).await?;
```

For millions of objects, this consumed gigabytes of memory (~80 bytes per key × 100M objects = ~8GB).

**Solution**: Converted all backends to use streaming list + batched deletion:
- ✅ **S3**: Now uses `s3_list_objects_stream()` with 1000-object batches
- ✅ **Azure**: Now uses `list_stream()` with 1000-object batches  
- ✅ **GCS**: Now uses `list_objects_stream()` with 1000-object batches

**Memory Impact**:
- **Before**: O(N) memory - loaded all N object keys
- **After**: O(1) memory - max 1000 keys in memory at once
- **Example**: 100M objects: 8GB → 80KB memory usage

**Performance**: No degradation - deletion happens in parallel batches as objects are listed.

**Affected Methods**:
- `GcsObjectStore::delete_prefix()` - src/object_store.rs line ~2173
- `S3ObjectStore::delete_prefix()` - src/object_store.rs line ~1050  
- `AzureObjectStore::delete_prefix()` - src/object_store.rs line ~1705

**Testing**: Existing tests pass (test_file_store, dl-driver checkpoint tests).

---

## Version 0.9.33 - Issue #110 Investigation Results (December 2025)

### 📝 **Issue #110: GCS Console Shows Directory Structure After Delete**

**Investigation**: User reported that after using `s3-cli delete -r` on GCS buckets, empty directory structures remained visible in the GCS Console UI.

**Root Cause Analysis**: Extensive testing revealed that:
- ✅ `delete -r` successfully deletes ALL objects (verified with `gsutil ls -r`)
- ✅ GCS API correctly reports zero objects in bucket
- ✅ Both s3-cli and gsutil confirm bucket is empty
- ❌ **GCS Console UI continues to show "folder icons"** (📁)

**Conclusion**: The directory icons are **virtual UI artifacts** created by the GCS Console based on object path structure (the `/` delimiter). These are NOT actual objects in storage - they're purely a Console rendering/caching issue that eventually expires (hours to days).

**Verification Test**:
```bash
# Delete all objects with s3-cli v0.9.25 (before any "fix")
./s3-cli delete -r gs://bucket/
# Deleted: 25088 objects ✅

# Verify with gsutil
gsutil ls -r gs://bucket/
# (no output - bucket is empty) ✅

# Verify with s3-cli
./s3-cli ls -r gs://bucket/
# Total objects: 0 ✅

# Check Console UI
# Still shows directory icons ❌ (UI caching artifact)
```

**Resolution**: No code changes required. This is expected GCS Console behavior. The bucket is actually empty according to the API - the Console UI just hasn't invalidated its cache.

**Workarounds for Console Display**:
1. **Wait**: Console cache expires eventually (hours to days)
2. **Different browser/incognito mode**: May show correct (empty) state
3. **Ignore**: Doesn't affect storage costs, API operations, or functionality

**Status**: Closing as "Not a Bug" - this is GCS Console UI behavior, not a code issue.

---

## Version 0.9.32 - Bug Fix: FileSystemConfig Type Mismatch (December 2025)

### 🐛 **Fixed Issue #85: FileSystemConfig Type Mismatch**

**Problem**: The public API exported `crate::file_store::FileSystemConfig` (with `page_cache_mode`) but `store_for_uri_with_config()` used `crate::file_store_direct::FileSystemConfig` (with `direct_io`, `alignment`, etc.). This made it impossible to configure page cache behavior via the public API.

**Solution**: Introduced unified `StorageConfig` enum:

```rust
pub enum StorageConfig {
    File(FileSystemConfig),      // For file:// URIs
    Direct(DirectFileSystemConfig),  // For direct:// URIs
}
```

**⚠️ BREAKING CHANGE**: `store_for_uri_with_config()` signature changed:

```rust
// OLD (v0.9.31 and earlier)
store_for_uri_with_config(uri, Some(config))

// NEW (v0.9.32+)
store_for_uri_with_config(uri, Some(StorageConfig::File(config)))
store_for_uri_with_config(uri, Some(StorageConfig::Direct(config)))
```

**Migration**:
- Import `StorageConfig` from `s3dlio::api`
- Wrap `FileSystemConfig` in `StorageConfig::File(...)` for file:// URIs
- Wrap `DirectFileSystemConfig` in `StorageConfig::Direct(...)` for direct:// URIs
- Type checking now prevents using wrong config with wrong URI scheme

**Improvements**:
- ✅ Public API now compiles and works as documented
- ✅ Page cache hints (Sequential, Random, DontNeed) now functional
- ✅ Added debug logging to show page cache mode being applied
- ✅ Clear error messages if wrong config type used with URI scheme
- ✅ Comprehensive tests for all configuration scenarios

**New Exports**:
- `pub use crate::api::StorageConfig` - Unified config enum
- `pub use crate::file_store_direct::FileSystemConfig as DirectFileSystemConfig` - O_DIRECT config

**Example**:

```rust
use s3dlio::api::{FileSystemConfig, StorageConfig, PageCacheMode, store_for_uri_with_config};

let config = FileSystemConfig {
    enable_range_engine: false,
    range_engine: Default::default(),
    page_cache_mode: Some(PageCacheMode::Sequential),
};

// This now works!
let store = store_for_uri_with_config(
    "file:///data/test/", 
    Some(StorageConfig::File(config))
)?;
```

## Version 0.9.31 - Maintenance Release (December 2025)

- Changed license to Apache-2.0
- Updated all source files to SPDX-compliant headers
- Removed fork-patches directory

## Version 0.9.30 - Zero-Copy Refactor & PyO3 0.27 (December 2025)

### 🆕 **Zero-Copy Architecture**

**True Zero-Copy Data Path**
- Refactored entire read path to use `bytes::Bytes` throughout the stack
- Eliminated intermediate allocations between storage backends and Python
- `BytesView` wrapper provides direct access to storage data without copying

**Python Buffer Protocol**
- Implemented `__getbuffer__` / `__releasebuffer__` for `BytesView` class
- Enables true zero-copy access via `memoryview(data)`
- Compatible with NumPy, PyTorch, and other frameworks that use buffer protocol

```python
import s3dlio

# Zero-copy read - data stays in same memory location
data = s3dlio.get("s3://bucket/file.bin")
view = memoryview(data)  # No copy! Direct buffer access
arr = np.frombuffer(view, dtype=np.float32)  # Still no copy
```

### 🔄 **PyO3 0.27 Migration**

- Upgraded PyO3 from 0.25 to 0.27.2
- Updated all APIs to use non-deprecated patterns:
  - `Python::attach()` (was `Python::with_gil`)
  - `py.detach()` (was `py.allow_threads`)
  - `Py<PyAny>` (was `PyObject`)
  - `.cast()` (was `.downcast()`)
- Updated dependencies: pyo3-async-runtimes 0.27.0, numpy 0.27.1

### 🧹 **API Cleanup: Protocol Equality**

**Removed S3-Specific Public APIs**
- All storage protocols now treated equally (S3, Azure, GCS, file://, direct://)
- Internal S3 functions changed from `pub fn` to `pub(crate) fn`
- Removed deprecated Python functions:
  - `list_objects(bucket, prefix)` → Use `list(uri)`
  - `get_object(bucket, key, offset, length)` → Use `get(uri)` or `get_range(uri, offset, length)`

**CLI: `list` is Now a True Alias for `ls`**
- Both commands execute identical code paths
- All protocols supported: `s3://`, `az://`, `gs://`, `file://`, `direct://`
- Same options available: `-r`, `-p`, `-c`

```bash
s3dlio ls s3://bucket/prefix/ -r     # Works
s3dlio list az://container/path/ -r  # Also works - same command
```

### 📦 **Technical Details**

- **Branch:** fix/multi-backend-zero-copy
- **Files changed:** 12
- **Tests:** 175 Rust tests passing, Python tests passing
- **Warnings:** Zero compiler warnings

---

## Version 0.9.27 - PyPI Release & Bug Fixes (December 2025)

### 🆕 **First PyPI Release**

s3dlio is now available on PyPI! Install with:
```bash
pip install s3dlio
```

### 🐛 **Bug Fixes**

**Optional ML Framework Imports**
- Fixed crash when importing s3dlio without PyTorch/JAX/TensorFlow installed
- ML-specific loaders (S3MapDataset, S3IterableDataset, etc.) now return `None` if frameworks unavailable
- Core storage functions work without any ML dependencies

**Universal Backend Support for Async Functions**
- Fixed `exists_async()` to work with all backends (was S3-only)
- Fixed `stat_async()` to work with all backends (was S3-only)
- Both now use the universal ObjectStore interface like their sync counterparts

### 📦 **Package Metadata**

- Added author information: Russ Fellows
- Proper package metadata for PyPI listing

### ⚠️ **Note**

v0.9.26 was yanked from PyPI due to the import bug. Use v0.9.27.

---

## Version 0.9.26 - DLIO Benchmark Integration (December 2025)

### 🆕 **New Features**

**DLIO Benchmark Integration**
- Added comprehensive integration support for [Argonne DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark)
- Two installation options:
  - **Option 1 (Recommended):** New `storage_type: s3dlio` with explicit configuration
  - **Option 2:** Drop-in replacement for existing S3 configurations
- Enables DLIO to use all s3dlio backends: S3, Azure, GCS, file://, direct://

**Zero-Copy Write Functions (Rust + Python)**
- `put_bytes(uri, data)` - Write bytes to any backend using zero-copy from Python memory
- `put_bytes_async(uri, data)` - Async version
- `mkdir(uri)` - Create directories/prefixes via ObjectStore trait
- `mkdir_async(uri)` - Async version
- Uses `PyBytes.as_bytes()` for zero-copy data transfer, avoiding allocation overhead

### 📦 **New Files**

```
python/s3dlio/integrations/dlio/
├── __init__.py           # Helper functions for installation
├── s3dlio_storage.py     # Option 1: S3dlioStorage class
└── s3_torch_storage.py   # Option 2: Drop-in replacement

docs/integration/
└── DLIO_BENCHMARK_INTEGRATION.md  # Comprehensive guide (477 lines)
```

### 🔧 **Other Changes**

**Azure SDK Update**
- Updated from Azure SDK 0.4.0 to 0.7.0 API
- Maintains backward compatibility with existing functionality

### 📝 **Documentation**

- Added `docs/integration/DLIO_BENCHMARK_INTEGRATION.md` - comprehensive guide with:
  - Architecture diagram
  - Step-by-step instructions for both integration options
  - Environment variable configuration (S3/Azure/GCS)
  - API mapping table (DLIO → s3dlio)
  - Troubleshooting section
  - Multi-protocol examples
  - Advanced features (multi-endpoint load balancing, performance tuning)
  - Feature compatibility table
- Updated `docs/PYTHON_API_GUIDE.md` with new functions

---

## Version 0.9.24 - S3-Compatible Endpoint Fix & Tracing Workaround (December 2025)

### 🐛 **Bug Fixes**

**S3-Compatible Endpoint Support (force_path_style)**
- Fixed S3 requests to custom endpoints (MinIO, Ceph, etc.)
- Added `force_path_style(true)` to S3 config builder in `s3_client.rs`
- **Root cause**: AWS SDK defaults to virtual-hosted style addressing (e.g., `bucket.endpoint.com`) which doesn't work with custom endpoints that expect path-style (e.g., `endpoint.com/bucket`)
- Now matches AWS CLI behavior when using custom endpoints

**Tracing Hang Workaround**
- Fixed hang when using verbose flags (`-v`, `-vv`) with `s3-cli`
- **Root cause**: `tracing::debug!()` macros inside `tokio::spawn` async tasks cause indefinite hangs when AWS SDK S3 operations are also running in that task
- **Solution**: CLI now uses `warn,s3dlio=debug` filter to exclude AWS SDK debug logging
- Commented out debug statements inside `tokio::spawn` in `s3_utils.rs` as an additional safeguard

### 🐛 **Known Issues**

**AWS SDK Tracing Hang Bug** ([aws-sdk-rust#1388](https://github.com/awslabs/aws-sdk-rust/issues/1388))
- `tracing::debug!()` inside `tokio::spawn` + AWS SDK operations = hang
- Affects both SDK 1.104.0 and 1.116.0 (not a recent regression)
- **Workaround**: Filter tracing with `RUST_LOG=warn,s3dlio=debug`
- Debug statements outside `tokio::spawn` work correctly
- The `s3-cli` `-v` and `-vv` flags use the workaround filter automatically

### 📦 **Examples & Project Organization**

**Python Examples** (6 comprehensive examples in `examples/python/`)
- `basic_operations.py` - Core put/get/list/stat/delete operations
- `parallel_operations.py` - High-performance parallel get/put with concurrency tuning
- `data_loader.py` - ML data loader patterns with batch iteration
- `streaming_writer.py` - Chunked/streaming upload API with compression
- `upload_download.py` - File upload/download workflows
- `oplog_example.py` - Operation logging/tracing demonstration

**Examples Directory Reorganization**
- Moved Python examples to `examples/python/`
- Moved Rust examples to `examples/rust/`
- Moved shell scripts to `scripts/`
- Deleted broken examples that used outdated APIs

**Op-Log Fixes**
- Fixed `get()` and `delete()` in Python API to use `store_for_uri_with_logger()`
- Fixed `put_objects_parallel_with_progress()` to use logger
- All Python operations now properly logged via `LoggedObjectStore` wrapper

**Code Quality**
- Fixed unused import warning (`info` in `s3_client.rs`)
- Added `#[cfg(feature = "experimental-http-client")]` to functions only used with that feature
- Zero non-deprecation warnings

### 📝 **Documentation**

- Created `docs/bugs/AWS_SDK_TRACING_HANG_BUG_REPORT.md` with full investigation details
- Added doc comments to `list_objects_stream()` in `s3_utils.rs` warning about the bug
- Updated README.md with v0.9.24 release notes

---

## Version 0.9.23 - Azure Blob & GCS Custom Endpoint Support (December 3, 2025)

### 🆕 **New Features**

**Custom Endpoint Support for Azure Blob Storage**
- Added environment variable support for custom Azure endpoints
- Primary: `AZURE_STORAGE_ENDPOINT` (e.g., `http://localhost:10000`)
- Alternative: `AZURE_BLOB_ENDPOINT_URL`
- Enables use with Azurite or other Azure-compatible emulators/proxies
- Account name is appended to endpoint URL automatically

Usage:
```bash
# Azurite (local emulator)
export AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000
sai3-bench util ls az://devstoreaccount1/testcontainer/

# Multi-protocol proxy
export AZURE_STORAGE_ENDPOINT=http://localhost:9001
sai3-bench util ls az://myaccount/mycontainer/
```

**Custom Endpoint Support for Google Cloud Storage**
- Added environment variable support for custom GCS endpoints
- Primary: `GCS_ENDPOINT_URL` (e.g., `http://localhost:4443`)
- Alternative: `STORAGE_EMULATOR_HOST` (GCS emulator convention, `http://` prepended if missing)
- Enables use with fake-gcs-server or other GCS-compatible emulators/proxies
- Anonymous authentication used automatically for custom endpoints (typical for emulators)

Usage:
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

### 📝 **Documentation**

- Updated `docs/api/Environment_Variables.md` with Azure and GCS endpoint configuration
- Added new constants in `src/constants.rs` for endpoint environment variable names:
  - `ENV_AZURE_STORAGE_ENDPOINT`
  - `ENV_AZURE_BLOB_ENDPOINT_URL`
  - `ENV_GCS_ENDPOINT_URL`
  - `ENV_STORAGE_EMULATOR_HOST`

### ⚡ **Compatibility**

**Backwards Compatibility**: No breaking changes
- When environment variables are not set, behavior remains identical to previous versions
- Connects to public cloud endpoints by default (Azure Blob, GCS)
- S3 custom endpoint support via `AWS_ENDPOINT_URL` remains unchanged

**Related Issue**: https://github.com/russfellows/sai3-bench/issues/56

---

## Version 0.9.22 - Client ID & First Byte Tracking (November 25, 2025)

### 🆕 **New Features**

**Client ID Support for Multi-Agent Operation Logging**
- Added `set_client_id()` and `get_client_id()` public functions
- All operation log entries now include client_id field
- Enables identification of which client/agent performed each operation
- Thread-safe implementation using `OnceCell<Mutex<String>>`
- Minimal overhead: ~10ns per log entry (mutex lock + clone)
- Use case: Distributed benchmarking with multiple agents writing to separate oplogs

API Usage:
```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Set client identifier (agent ID, hostname, custom ID, etc.)
let client_id = std::env::var("CLIENT_ID").unwrap_or_else(|_| "standalone".to_string());
s3dlio::set_client_id(&client_id)?;

// All future log entries tagged with this client_id
```

**Approximate First Byte Tracking** (See docs/OPERATION_LOGGING.md for details)
- `first_byte_time` field now populated in operation logs
- GET operations: first_byte ≈ end (when complete data is available)
- PUT operations: first_byte = start (upload begins immediately)
- Metadata operations (LIST, HEAD, DELETE): first_byte = None (not applicable)

**Important**: This is an *approximate* implementation due to ObjectStore trait limitations:
- Current API returns `Bytes` (complete data), not `Stream<Bytes>`
- Can't distinguish HTTP header receipt from body completion
- For small objects (<1MB): approximation is acceptable for throughput analysis
- For true TTFB metrics: Use streaming APIs (future enhancement) or dedicated HTTP tools

See [OPERATION_LOGGING.md](OPERATION_LOGGING.md) for comprehensive documentation on:
- Why first_byte is approximate
- When to use vs when to avoid
- Future enhancement plans (streaming GET API)
- Recommendations for different use cases

### 📝 **Documentation**

**New: Operation Logging Guide** (docs/OPERATION_LOGGING.md)
- Comprehensive explanation of operation logging architecture
- First byte tracking strategy with detailed rationale
- Clock offset synchronization patterns
- Client identification best practices
- Example usage for standalone and distributed scenarios
- Performance impact analysis
- Future enhancement roadmap

**Updated: Code Comments**
- Extensive inline documentation in `object_store_logger.rs`
- 40+ lines explaining first_byte tracking approach and limitations
- Clear guidance on when approximation is acceptable
- Future enhancement notes for streaming APIs

**Clarification: Operation Log Sorting**
- Added section on post-processing oplogs in OPERATION_LOGGING.md
- Clarified that logs are NOT sorted during capture (due to concurrent writes)
- Documented proper sorting workflow using sai3-bench sort command
- Note: Sorted logs compress ~30-40% better than unsorted

### ⚠️ **Important Notes**

**first_byte_time Interpretation**:
- **DO**: Use for throughput analysis and relative comparisons
- **DO**: Use for small object (<1MB) performance benchmarking
- **DON'T**: Assume it represents exact time of first byte arrival
- **DON'T**: Use for precise TTFB analysis on large objects (>10MB)

**Backwards Compatibility**:
- Existing code continues to work (client_id defaults to empty string)
- TSV format unchanged (first_byte column existed but was empty before)
- No breaking API changes

## Version 0.9.21 - Clock Offset Support & Pseudo-Random Data Generation (November 25, 2025)

### 🆕 **New Features**

**Clock Offset Support for Distributed Op-Log Synchronization** (Issue #100)
- Added `set_clock_offset()` and `get_clock_offset()` public functions
- Logger now supports timestamp correction for distributed systems
- Enables accurate global timeline reconstruction when agents have clock skew
- Thread-safe implementation using `Arc<AtomicI64>`
- Minimal overhead: single atomic read per log entry
- Use case: sai3-bench and dl-driver distributed benchmarking

API Usage:
```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Calculate clock offset during agent sync
let offset = (local_time - controller_time).as_nanos() as i64;

// Set offset for all future log entries
s3dlio::set_clock_offset(offset)?;
```

**Pseudo-Random Data Generation Method** (Issue #98 resolution)
- Added `generate_controlled_data_prand()` function
- Uses original BASE_BLOCK algorithm (~3-4 GB/s, consistent performance)
- Provides "prand" option alongside "random" (new Xoshiro256++ algorithm)
- Public API in `s3dlio::api::advanced` module
- When to use:
  - `random`: Truly incompressible data (compress=1 → ~1.0 zstd ratio), slower but more realistic
  - `prand`: Maximum CPU efficiency, faster but allows cross-block compression patterns

### ✅ **Bug Fixes & Improvements**

**Issue #95: Range Engine Messages** (Already Fixed)
- Confirmed Range Engine messages now at `trace!` level (not `debug!`)
- Fixed in commit edee657 (November 16, 2025)
- No longer overwhelms debug output

**Issue #98: Old Data Generation Code**
- Resolution: Old code now serves as "prand" method, not removed
- Provides performance option for CPU-constrained scenarios
- Both algorithms available for different use cases

### 📝 **Documentation**

- Added comprehensive clock offset documentation in `s3_logger.rs`
- Updated API documentation for both data generation methods
- Added usage examples for distributed op-log synchronization

---

## Version 0.9.20 - High-Performance List & Delete Optimizations (November 22, 2025)

### 🚀 **Performance Improvements for Large Object Operations**

Major optimizations for workloads with 100K-1M+ objects, targeting 5x performance improvement (28 minutes → 5-8 minutes for 1M objects).

**Phase 1: Batch Delete API**
- Added `delete_batch()` method to ObjectStore trait
- S3: DeleteObjects API (1000 objects/request)
- Azure: Batch API with pipeline (up to 256 operations)
- GCS: Batch delete with parallel execution
- File/Direct: Parallel deletion with configurable concurrency
- All 7 backends implement efficient batch operations

**Phase 2: Streaming List with Progress**
- Added `list_stream()` returning `Pin<Box<dyn Stream<Item = Result<String>>>>`
- S3: True streaming via paginated ListObjectsV2 (1000-object pages)
- Azure/GCS/File: Efficient buffered implementation
- MultiEndpoint: Wraps stream with per-endpoint statistics
- ObjectStoreLogger: Proper op-log integration for workload replay
- CLI: Added `-c/--count-only` flag with progress indicators
- CLI: Rate formatting with comma separators (e.g., "rate: 276,202 objects/s")

**Phase 3: List+Delete Pipeline**
- Concurrent lister/deleter tasks via tokio channels
- 1000-object batches, 10-batch buffer (10K objects in-flight)
- Overlaps LIST and DELETE operations for maximum throughput
- Progress reporting every 10K objects

**CLI Improvements:**
```bash
# Count objects with streaming progress
s3-cli ls -rc s3://bucket/prefix/
# Output: Total objects: 1,234,567 (12.345s, rate: 100,000 objects/s)

# Fast deletion with pipeline
s3-cli delete s3://bucket/prefix/
# Uses concurrent list+delete pipeline automatically
```

**Architecture:**
- Clean abstractions maintained (no backend-specific CLI code)
- Proper op-log integration for workload replay capability
- Zero-copy streaming where possible
- All optimizations work across all 7 storage backends

**Testing:**
- Comprehensive test suite with file:// backend
- Pattern matching validated (preserves non-matching objects)
- Pipeline performance verified (10K files in 0.213s)
- Progress indicators and rate formatting confirmed

**Documentation:**
- `docs/LIST_DELETE_PERFORMANCE_OPTIMIZATION.md` - Complete 3-phase plan
- All phases implemented and tested

### 📝 **API Stability**

**Rust API:**
- New trait methods: `ObjectStore::delete_batch()`, `ObjectStore::list_stream()`
- Backward compatible - existing code continues to work
- Python API automatically benefits from underlying optimizations

**Python API:**
- No changes required - `delete()` and `list()` use optimized implementations
- Batch operations work transparently under the hood

---

## Version 0.9.18 - Data Generation Bug Fix & Algorithm Migration (November 17-18, 2025)

### 🔧 **Update (November 18, 2025): RNG Optimization & Distributed Safety**

**Performance Optimization:**
- Explicit Xoshiro256PlusPlus RNG (removed StdRng abstraction)
- 5-24% performance improvement in data generation
- Same or lower CPU usage across all workloads
- Added `rand_xoshiro = "^0.7"` dependency

**Distributed Deployment Enhancement:**
- Enhanced entropy source: SystemTime + `/dev/urandom`
- Prevents data collision across distributed workers
- Critical for orchestrated environments (Kubernetes, SLURM)
- Ensures global uniqueness even with synchronized clocks

**Code Quality:**
- Updated comments: removed "Bresenham" terminology
- Clarified as "integer error accumulation" (standard distribution technique)
- No patent concerns or algorithmic attribution issues

**Comprehensive Testing:**
- New `tests/performance_comparison.rs` with CPU/memory metrics
- All 162 library tests passing
- Performance validated across 6 workload scenarios
- API compatibility verified (zero breaking changes)

**Performance Results:**
```
Test                 | OLD Speed  | NEW Speed  | Speedup | CPU Δ
---------------------|------------|------------|---------|-------
1MB compress=1       | 3,474 MB/s | 3,436 MB/s | 0.99x   | +15%
16MB compress=1      | 6,319 MB/s | 6,621 MB/s | 1.05x   | 0%
64MB compress=1      | 5,800 MB/s | 6,283 MB/s | 1.08x   | -10%
16MB compress=5      | 5,660 MB/s | 7,009 MB/s | 1.24x   | -3%
Streaming 16MB       | 2,355 MB/s | 2,530 MB/s | 1.07x   | 0%
16MB dedup=4         | 6,936 MB/s | 7,553 MB/s | 1.09x   | -3%
```

**Documentation:**
- `API_COMPATIBILITY_REPORT.md` - Complete API analysis
- Verified sai3-bench (6 call sites) and dl-driver (12 call sites) compatibility

### 🐛 **Critical Bug Fix: Cross-Block Compression**

Fixed a critical bug in the data generation algorithm where `compress=1` (incompressible data) incorrectly produced 7.68:1 compression ratio instead of ~1.0.

**Root Cause:**
- Original algorithm used shared `BASE_BLOCK` template across all unique blocks
- Zstd compressor found cross-block patterns, defeating incompressibility guarantee
- Affected all compress levels (1-6) when combined with dedup > 1

**Solution:**
- New algorithm uses per-block Xoshiro256++ RNG initialization
- Each unique block gets independent high-entropy keystream
- Local back-references within blocks for controlled compressibility
- `compress=1` now correctly produces ratio ~1.0000 ✅

### ✨ **New Data Generation Algorithm**

Introduced `data_gen_alt.rs` with improved correctness and performance:

**Features:**
- Per-block RNG seeding (prevents cross-block compression)
- Xoshiro256++ RNG (5-10x faster than ChaCha20)
- Streaming generation via `ObjectGenAlt`
- Parallel single-pass generation for large datasets
- Performance: 1-7 GB/s depending on size

**API Changes:**
- **Zero breaking changes** - all existing code works unchanged
- `generate_controlled_data()` transparently redirected to new algorithm
- `ObjectGen` now wraps `ObjectGenAlt` internally
- Old implementations preserved as commented-out code (removal: December 2025)

### 📊 **Validation Results**

**Compression Ratios (16MB test):**
- compress=1: 1.0000 ✅ (was 7.6845 ❌)
- compress=5: 1.3734 ✅
- compress=6: 1.3929 ✅

**Performance:**
- 1MB: 954 MB/s (3.76x faster than old algorithm)
- 16MB: 2,816 MB/s
- 64MB: 7,351 MB/s
- Streaming: 1,374 MB/s

**Testing:**
- All 162 library tests passing ✅
- Comprehensive test suite added (`tests/test_data_gen_alt.rs`)
- Deduplication behavior validated (dedup=2,6)
- Old algorithm bug confirmed via regression test

### 📝 **Documentation**

Added comprehensive migration documentation:
- `docs/DATA_GEN_MIGRATION_SUMMARY.md` - Technical details (9.2KB)
- `.github/ISSUE_TEMPLATE/data_gen_migration.md` - Tracking issue template (5.4KB)
- `.github/copilot-instructions.md` - Migration checklist
- `src/data_gen.rs` - 73-line header explaining changes

### 🔄 **Migration Timeline**

**Phase 1: Production Validation (Nov-Dec 2025)**
- Extended testing with real workloads
- Performance monitoring across platforms
- Compatibility verification with downstream tools

**Phase 2: Code Cleanup (December 2025)**
- Remove commented-out old algorithm code
- Update inline documentation
- Consider renaming data_gen_alt.rs → data_gen.rs

**Phase 3: Optimization (Q1 2026)**
- Profile Xoshiro256++ performance
- Evaluate SIMD opportunities
- Benchmark against industry tools

### 🔧 **Dependencies**

- Updated `rand_chacha` 0.3 → 0.9
- Fixed API breaking changes (`gen_range` → `random_range`)
- Updated test code for modern APIs

### 📦 **Test Suite**

- **Total tests**: 162 (all passing)
- **New tests**: 7 comprehensive tests for data_gen_alt
- **Coverage**: Compression ratios, dedup behavior, streaming, performance, regression

---

## Version 0.9.17 - NPY/NPZ Enhancements & TFRecord Index API (November 16, 2025)

### 🎯 **Multi-Array NPZ Support**

Added `build_multi_npz()` function for creating NumPy ZIP archives with multiple named arrays, enabling PyTorch/JAX-style dataset creation with data, labels, and metadata in a single file.

**New API:**

```rust
use s3dlio::data_formats::npz::build_multi_npz;
use ndarray::ArrayD;

// Create multi-array NPZ (PyTorch/JAX pattern)
let data = ArrayD::zeros(vec![224, 224, 3]);
let labels = ArrayD::ones(vec![10]);
let metadata = ArrayD::from_elem(vec![5], 42.0);

let arrays = vec![
    ("data", &data),
    ("labels", &labels),
    ("metadata", &metadata),
];

let npz_bytes = build_multi_npz(arrays)?;
// Write npz_bytes to file or object storage
```

**Python Interoperability:**

```python
import numpy as np

# Load multi-array NPZ created by Rust
data = np.load("dataset.npz")
print(data.files)  # ['data', 'labels', 'metadata']

images = data['data']      # Shape: (224, 224, 3)
labels = data['labels']    # Shape: (10,)
metadata = data['metadata'] # Shape: (5,)
```

**Key Features:**
- **Zero-copy design**: Uses `Bytes` for efficient memory handling
- **Proper ZIP structure**: Compatible with NumPy's `np.load()`
- **Named arrays**: Custom names for each array in the archive
- **Type support**: f32 arrays (primary ML use case)
- **Comprehensive tests**: 5 new tests covering single/multi-array scenarios

**Use Cases:**
- AI/ML dataset generation (images + labels + metadata)
- Scientific computing (simulation results + parameters + timestamps)
- dl-driver workload generation (simplified from 150+ lines to 80 lines)

---

### 🔧 **TFRecord Index Generation API**

Exported `build_tfrecord_with_index()` function and `TfRecordWithIndex` struct for creating TFRecord files with accompanying index files, enabling compatibility with TensorFlow Data Service.

**New Exports:**

```rust
use s3dlio::data_formats::{build_tfrecord_with_index, TfRecordWithIndex};

// Generate TFRecord with index in single pass
let raw_data = s3dlio::generate_controlled_data(102400, 1, 1);
let result = build_tfrecord_with_index(
    100,    // num_records
    1024,   // record_size
    &raw_data
)?;

// result.data: Bytes containing TFRecord file
// result.index: Bytes containing index file (16 bytes per record)

// Write both files
store.put("dataset.tfrecord", &result.data).await?;
store.put("dataset.tfrecord.index", &result.index).await?;
```

**Index Format** (TensorFlow Data Service compatible):
```
For each record:
  - offset: u64 (8 bytes, little-endian) - Byte offset in TFRecord file
  - length: u64 (8 bytes, little-endian) - Record length in bytes
Total: 16 bytes per record
```

**Key Features:**
- **Zero overhead**: Index generated during TFRecord creation (single pass)
- **Standard format**: Compatible with TensorFlow Data Service expectations
- **Efficient**: Returns `Bytes` for zero-copy I/O
- **Documented**: Clear API for downstream tools (dl-driver, custom tools)

**Performance:**
- No additional I/O operations
- Minimal memory overhead (16 bytes per record)
- Example: 1000 records → 16KB index file

**Background:**

TensorFlow Data Service can leverage index files to optimize random access patterns and enable efficient dataset sharding across distributed workers. This API enables tools to generate properly formatted indices alongside TFRecord data files.

---

### 🔄 **Custom NPY/NPZ Implementation**

Previously in v0.9.16, replaced `ndarray-npy` dependency with custom 328-line implementation for better control, zero-copy performance, and elimination of version conflicts.

**Features (continued from v0.9.16):**
- Full NPY format support (header + data serialization)
- Multi-array NPZ with proper ZIP structure (NEW in v0.9.17)
- TFRecord index generation API (NEW in v0.9.17)
- Zero-copy `Bytes`-based design
- Python/NumPy interoperability verified
- 11 comprehensive tests (6 NPY + 5 multi-NPZ)

---

## Version 0.9.16 - Optional Op-Log Sorting (November 7, 2025)

### 📊 **Configurable Operation Log Sorting**

Added optional automatic sorting of operation logs by start timestamp, addressing chronological ordering requirements for multi-threaded workloads while maintaining high performance for large-scale logging (10M+ operations).

**Key Changes:**

- **Optional Auto-Sort at Shutdown** - Controlled via `S3DLIO_OPLOG_SORT` environment variable
  - **Default behavior**: Streaming write (no sorting, zero memory overhead, immediate output)
  - **Opt-in sorting**: Set `S3DLIO_OPLOG_SORT=1` to enable chronological sorting
  - **Performance**: ~1.2μs per entry overhead (~4% for 210K entries)
  - **Use case**: Small to medium workloads (<1M operations) requiring sorted output

- **Streaming Sort Window Constant** - `DEFAULT_OPLOG_SORT_WINDOW = 1000`
  - Documented for future streaming sort implementations
  - Sized based on observation that operations are rarely >1000 lines out of order
  - Enables constant-memory sorting for huge files (50M+ operations)

**Background:**

Multi-threaded operation logging writes entries as they complete, not in start-time order. Variable I/O latency causes operations to finish out of sequence. For workloads requiring chronological analysis (replay, performance analysis), sorting is now optionally available.

**Environment Variables:**

```bash
# Default: Fast streaming write (unsorted)
sai3-bench run --op-log /tmp/ops.tsv --config test.yaml

# Opt-in: Auto-sort at shutdown (sorted output)
S3DLIO_OPLOG_SORT=1 sai3-bench run --op-log /tmp/ops.tsv --config test.yaml
```

**Configuration Constants:**

- `ENV_OPLOG_SORT` - Environment variable name for auto-sort control
- `DEFAULT_OPLOG_SORT_WINDOW` - Window size for streaming sort algorithms (1000 lines)

**Implementation Details:**

- Sort-on-write path collects all entries in `Vec<LogEntry>`, sorts by `start_time`, then writes
- No-sort path streams directly to file (zero buffering, minimal memory)
- Both paths use zstd compression (level 1) and auto-add `.zst` extension
- Proper `info!()` logging for transparency during long sorts

**Rationale:**

This opt-in approach balances three competing needs:
1. **Performance**: Default streaming mode has zero sorting overhead
2. **Scalability**: Large workloads (10M+ ops) avoid memory pressure
3. **Usability**: Small workloads can enable convenient auto-sort

For very large files requiring sorting, downstream tools (sai3-bench) provide streaming window-based offline sorting with constant memory usage.

---

## Version 0.9.15 - S3 URI Endpoint Parsing (November 6, 2025)

### 🔧 **Enhanced URI Parsing for Multi-Endpoint Scenarios**

Added utilities for parsing S3 URIs with optional custom endpoints, supporting MinIO, Ceph, and other S3-compatible storage systems with explicit endpoint specifications.

**New Functions:**

**Rust API:**
```rust
use s3dlio::{parse_s3_uri_full, S3UriComponents};

// Standard AWS format
let components = parse_s3_uri_full("s3://mybucket/data.bin")?;
assert_eq!(components.endpoint, None);  // Uses AWS_ENDPOINT_URL env var
assert_eq!(components.bucket, "mybucket");
assert_eq!(components.key, "data.bin");

// MinIO/Ceph with custom endpoint
let components = parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")?;
assert_eq!(components.endpoint, Some("192.168.100.1:9001".to_string()));
assert_eq!(components.bucket, "mybucket");
assert_eq!(components.key, "data.bin");
```

**Python API:**
```python
import s3dlio

# Parse standard AWS URI
result = s3dlio.parse_s3_uri_full("s3://mybucket/data.bin")
# {'endpoint': None, 'bucket': 'mybucket', 'key': 'data.bin'}

# Parse MinIO URI with endpoint
result = s3dlio.parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")
# {'endpoint': '192.168.100.1:9001', 'bucket': 'mybucket', 'key': 'data.bin'}
```

**Key Features:**
- Heuristic endpoint detection (IP addresses, hostnames, ports)
- Backwards compatible with existing `parse_s3_uri()`
- Useful for multi-process benchmarking tools (sai3-bench, dl-driver)
- Zero overhead - parsing utilities only

**Use Case:**
Multi-process testing with different endpoints per process:
```bash
# Process 1
AWS_ENDPOINT_URL=http://192.168.100.1:9001 sai3bench-agent

# Process 2  
AWS_ENDPOINT_URL=http://192.168.100.2:9001 sai3bench-agent
```

Tools can use `parse_s3_uri_full()` to extract endpoint information from config files for validation and process orchestration.

---

## Version 0.9.14 - Multi-Endpoint Storage (November 6, 2025)

### 🎯 **Multi-Endpoint Load Balancing for High-Throughput Workloads**

Added comprehensive multi-endpoint support enabling load distribution across multiple storage backends for improved performance, scalability, and fault tolerance.

**New Core Components:**

**1. MultiEndpointStore** - Load balancing wrapper implementing full `ObjectStore` trait
```rust
use s3dlio::{MultiEndpointStore, MultiEndpointStoreConfig, EndpointConfig, LoadBalanceStrategy};

let config = MultiEndpointStoreConfig {
    endpoints: vec![
        EndpointConfig { uri: "s3://bucket-1/data".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket-2/data".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket-3/data".to_string(), ..Default::default() },
    ],
    strategy: LoadBalanceStrategy::RoundRobin,  // or LeastConnections
    default_thread_count: 4,
};

let store = MultiEndpointStore::new(config).await?;
```

**2. Load Balancing Strategies:**
- **RoundRobin**: Sequential distribution for uniform workloads (minimal overhead)
- **LeastConnections**: Adaptive routing based on active connections (self-balancing)

**3. Per-Endpoint Configuration:**
```rust
EndpointConfig {
    uri: "s3://high-perf-bucket/data".to_string(),
    thread_count: Some(16),          // Override default thread count
    process_affinity: Some(0),       // NUMA node assignment for multi-socket systems
}
```

**4. URI Template Expansion** - New `uri_utils` module:
```rust
// Expand numeric ranges with optional zero-padding
expand_uri_template("s3://bucket-{1...10}/data")?;
// → ["s3://bucket-1/data", ..., "s3://bucket-10/data"]

expand_uri_template("direct://192.168.1.{01...10}:9000")?;
// → ["direct://192.168.1.01:9000", ..., "direct://192.168.1.10:9000"]

// Load URIs from configuration file (one per line)
load_uris_from_file("endpoints.txt")?;
```

**5. Statistics and Monitoring:**
```rust
// Per-endpoint statistics (lock-free atomics)
let stats = store.get_endpoint_stats();
for (i, stat) in stats.iter().enumerate() {
    println!("Endpoint {}: {} requests, {} bytes, {} active connections",
             i, stat.requests, stat.bytes_transferred, stat.active_connections);
}

// Total aggregated statistics
let total = store.get_total_stats();
println!("Total: {} requests, {} bytes, {} errors",
         total.total_requests, total.total_bytes, total.total_errors);
```

**6. Python API** - Zero-copy support via `BytesView`:
```python
import s3dlio

# Create multi-endpoint store
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://bucket-1/data", "s3://bucket-2/data", "s3://bucket-3/data"],
    strategy="least_connections"  # or "round_robin"
)

# Zero-copy data access (buffer protocol support)
data = store.get("s3://bucket-1/large-file.bin")
mv = memoryview(data)  # No copy!
array = np.frombuffer(mv, dtype=np.float32)

# Template expansion
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://shard-{1...10}/dataset",
    strategy="round_robin"
)

# File-based configuration
store = s3dlio.create_multi_endpoint_store_from_file(
    file_path="endpoints.txt",
    strategy="least_connections"
)

# Statistics
stats = store.get_endpoint_stats()
total = store.get_total_stats()
```

### ✨ **Features**

**Architecture:**
- Thread-safe design with atomic statistics (no lock contention)
- Schema validation: all endpoints must use same URI scheme (s3://, az://, gs://, file://, direct://)
- Transparent ObjectStore implementation - works anywhere single store is expected
- Per-endpoint thread pool control for optimal resource utilization

**Performance:**
- Lock-free atomic counters for statistics (zero contention overhead)
- Round-robin: ~10ns overhead per request
- Least-connections: ~50ns overhead per request
- Zero-copy Python interface via `BytesView` (buffer protocol)
- Negligible overhead vs latency (< 0.01% for typical cloud requests)

**Use Cases:**
- **Distributed ML training**: Access sharded datasets across multiple S3 buckets
- **High-throughput benchmarking**: Aggregate bandwidth from multiple storage servers
- **Fault tolerance**: Continue operations if individual endpoints fail
- **Geographic distribution**: Route requests to regionally-optimal endpoints
- **Load testing**: Distribute synthetic workloads across multiple backends

### 📝 **Testing**

**Comprehensive Test Coverage:**
- 33 new tests (16 in `uri_utils`, 17 in `multi_endpoint`)
- Total test count: 141 tests (all passing)
- Zero-copy validation tests in Python suite
- Load balancing distribution tests
- Error handling and validation tests

**Python Test Suite** (`tests/test_multi_endpoint.py`):
- Store creation from URIs, templates, and files
- Zero-copy data access via `memoryview()`
- NumPy/PyTorch integration tests
- Load balancing statistics validation
- Error handling for invalid configurations

### 🔧 **API Reference**

**Rust API:**
```rust
// Module exports
pub use multi_endpoint::{
    MultiEndpointStore,
    MultiEndpointStoreConfig,
    EndpointConfig,
    LoadBalanceStrategy,
    EndpointStats,
    TotalStats,
};

pub use uri_utils::{
    expand_uri_template,
    parse_uri_list,
    load_uris_from_file,
    infer_scheme_from_uri,  // Now public for validation
};
```

**Python API:**
```python
# Factory functions
s3dlio.create_multi_endpoint_store(uris, strategy) -> PyMultiEndpointStore
s3dlio.create_multi_endpoint_store_from_template(template, strategy) -> PyMultiEndpointStore  
s3dlio.create_multi_endpoint_store_from_file(file_path, strategy) -> PyMultiEndpointStore

# PyMultiEndpointStore methods
store.get(uri) -> BytesView                           # Zero-copy
store.get_range(uri, offset, length) -> BytesView     # Zero-copy
store.put(uri, data) -> None
store.list(prefix, recursive) -> List[str]
store.delete(uri) -> None
store.get_endpoint_stats() -> List[Dict]
store.get_total_stats() -> Dict
```

### 📖 **Documentation**

**New Documentation:**
- **[MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md)** - Comprehensive guide with:
  - Architecture overview and design principles
  - Load balancing strategy comparison
  - Rust and Python API examples
  - Configuration methods (URIs, templates, files)
  - Performance tuning guidelines
  - Best practices and use cases
  - Advanced topics (NUMA affinity, monitoring)
  - Troubleshooting guide

**Updated Documentation:**
- **[README.md](../README.md)** - Added multi-endpoint quick example in Python section
- **[docs/README.md](README.md)** - Added MULTI_ENDPOINT_GUIDE.md to Technical References

### 🔄 **Backward Compatibility**

✅ **Zero Breaking Changes:**
- New optional feature - existing code unaffected
- Single-endpoint workflows continue working unchanged
- Can wrap single endpoint in multi-endpoint store for future flexibility

### 🎯 **Use Case Examples**

**1. Distributed ML Training:**
```python
# Access sharded training data across 10 S3 buckets
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://training-shard-{1...10}/data",
    strategy="least_connections"
)

# Zero-copy data loading
for uri in file_list:
    data = store.get(uri)
    tensor = torch.frombuffer(memoryview(data), dtype=torch.float32)
```

**2. High-Throughput Benchmarking:**
```rust
// Aggregate bandwidth across 4 storage servers
let endpoints = vec![
    "s3://benchmark-1/test", "s3://benchmark-2/test",
    "s3://benchmark-3/test", "s3://benchmark-4/test",
];

// Round-robin for predictable load distribution
let config = MultiEndpointStoreConfig {
    endpoints: endpoints.iter().map(|uri| EndpointConfig {
        uri: uri.to_string(),
        thread_count: Some(8),
        ..Default::default()
    }).collect(),
    strategy: LoadBalanceStrategy::RoundRobin,
    ..Default::default()
};
```

**3. Geographic Distribution:**
```python
# Route to fastest regional endpoint automatically
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://data-us-west-2/dataset",
        "s3://data-us-east-1/dataset", 
        "s3://data-eu-west-1/dataset",
    ],
    strategy="least_connections"  # Naturally favors lower-latency endpoints
)
```

### 🔍 **Implementation Details**

**Zero-Copy Design:**
- Python `get()` and `get_range()` return `PyBytesView` objects
- `PyBytesView` implements Python buffer protocol
- Allows `memoryview()` access without copying data
- Compatible with NumPy, PyTorch, and other buffer-aware libraries
- Maintains Arc-counted `Bytes` reference (cheap clone, automatic cleanup)

**Thread Safety:**
- All statistics use `AtomicU64` counters (lock-free)
- Endpoint selection protected by `Mutex` (minimal contention)
- Concurrent operations across different endpoints fully parallel
- No performance degradation under high concurrency

**Schema Validation:**
- All endpoints must use same URI scheme (enforced at creation)
- Prevents mixing incompatible backends (e.g., s3:// + file://)
- Clear error messages for configuration mistakes

### 📊 **Performance Characteristics**

**Throughput Scaling:**
- 2 endpoints: 1.8-2.0× baseline throughput
- 4 endpoints: 3.5-4.0× baseline throughput  
- 8 endpoints: 6.5-8.0× baseline throughput
- 16+ endpoints: 12-16× baseline throughput

**Overhead:**
- RoundRobin: ~10ns per request (atomic increment)
- LeastConnections: ~50ns per request (atomic read + compare)
- For typical cloud requests (1-100ms), overhead < 0.01%

### 🔧 **Configuration Examples**

**endpoints.txt** (file-based configuration):
```text
# Production multi-region setup
s3://prod-us-west-2/data
s3://prod-us-east-1/data
s3://prod-eu-west-1/data

# Comments and blank lines ignored
```

**Template Patterns:**
```rust
// Simple numeric range
"s3://bucket-{1...5}/data"

// Zero-padded range  
"s3://bucket-{01...10}/data"

// IP addresses for direct I/O
"direct://192.168.1.{1...10}:9000"

// Multiple ranges (Cartesian product)
"s3://rack{1...3}-node{1...4}"  // → 12 URIs
```

---

## Version 0.9.12 - GCS Factory Fixes (November 3, 2025)

### 🐛 **Fixed**
- Fixed 4 misleading "GCS backend not yet fully implemented" errors in enhanced factory functions
- `store_for_uri_with_config_and_logger()` now properly supports GCS
- `direct_io_store_for_uri_with_logger()` now properly supports GCS  
- `high_performance_store_for_uri_with_logger()` now properly supports GCS

### ✨ **Added**
- `store_for_uri_with_high_performance_cloud()` - Enable RangeEngine for cloud backends
- `store_for_uri_with_high_performance_cloud_and_logger()` - With logging support
- Documentation for high-performance cloud factory functions

### 📝 **Notes**
- All basic GCS operations worked correctly even before this fix
- The errors only appeared in enhanced factory functions (not used by most code)
- dl-driver checkpoint plugin was never affected (uses basic `store_for_uri()`)

---

## Version 0.9.11 - Directory Operations (November 2024)

### 🎯 **Unified Directory Management Across All Backends**

Added `mkdir` and `rmdir` operations to the `ObjectStore` trait for consistent directory handling across file systems and cloud storage.

**New API Methods:**
```rust
async fn mkdir(&self, uri: &str) -> Result<()>
async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()>
```

**Backend Implementations:**
- **File/DirectIO** (`file://`, `direct://`): Creates/removes actual POSIX directories
  - `mkdir`: Uses `tokio::fs::create_dir_all()` for recursive creation
  - `rmdir`: Supports both empty (`remove_dir`) and recursive (`remove_dir_all`) deletion
  
- **Cloud Storage** (`s3://`, `az://`, `gs://`): Manages prefix markers
  - `mkdir`: Creates empty marker objects (e.g., `.keep`) to represent directories
  - `rmdir`: Deletes all objects under prefix (always recursive for cloud)

**Integration:**
- Required for sai3-bench v0.7.0+ directory tree workloads
- Enables consistent directory operations across hybrid storage environments
- Maintains backend-specific optimizations (e.g., cloud prefix semantics)

**Backward Compatibility:**
- ✅ Default trait implementations error gracefully for backends without support
- ✅ No breaking changes to existing APIs
- ✅ Optional functionality - existing workloads unaffected

---

## Version 0.9.10 - Pre-Stat Size Cache for Benchmarking (December 2024)

### 🚀 **Major Performance Improvement for Multi-Object Workloads**

#### **ObjectSizeCache - 2.5x Faster Benchmarking with Pre-Stat Optimization**

Eliminated redundant stat/HEAD operations in multi-object download workloads through intelligent size caching:

**Problem Solved:**
- Benchmarking tools downloading 1000+ objects spent 60% of time on sequential stat operations
- Each `get()` called stat before download: 1000 objects × 20ms stat = 20 seconds of pure overhead
- Network latency dominated total time (32.8s benchmark: 20s stat + 12.8s download)
- No way to amortize stat cost across multiple objects

**Solution - Pre-Stat Optimization:**
- New `ObjectSizeCache` module: Thread-safe size cache with configurable TTL (default 60s)
- New `pre_stat_objects()` trait method: Concurrent stat for multiple objects (100 concurrent default)
- New `pre_stat_and_cache()` API: Pre-stat all objects once, cache sizes, eliminate per-object stat overhead
- Integrated into S3, GCS, Azure backends automatically

**Performance Impact (Measured on 1000 × 64MB benchmark):**
- **Total time**: 32.8s → 13.0s (2.5x faster)
- **Stat overhead**: 20.0s → 0.2s (99% reduction)
- **Effective throughput**: 1.95 GB/s → 4.92 GB/s (2.5x improvement)
- **Pattern**: Pre-stat 1000 objects in 200ms (concurrent) vs 20s (sequential)

**Use Cases:**
- Benchmarking tools (`sai3-bench`, `io-bench`) downloading many objects
- Dataset pre-loading in ML training pipelines
- Batch processing of S3/GCS object collections
- Any workload with predictable object access patterns

**Backward Compatibility:**
- ✅ **Zero breaking changes** - All existing code works unchanged
- ✅ **Default trait methods** - All backends get concurrent pre-stat automatically
- ✅ **Opt-in API** - `pre_stat_and_cache()` is optional, regular `get()` still works
- ✅ **Graceful degradation** - Cache miss falls back to stat (same as before)

**API Usage - Before (v0.9.9 - Sequential stat overhead):**

```rust
// OLD: Each get() calls stat internally (20ms overhead per object)
let store = store_for_uri("s3://bucket/data/")?;
let objects = get_1000_object_uris();

for uri in &objects {
    let data = store.get(uri).await?;  // ❌ stat + download (20ms + 128ms = 148ms each)
    process(data);
}
// Total time: ~148 seconds for 1000 objects
```

**API Usage - After (v0.9.10 - Pre-stat optimization):**

```rust
// NEW: Pre-stat all objects once, then download without stat overhead
let store = store_for_uri("s3://bucket/data/")?;
let objects = get_1000_object_uris();

// PHASE 1: Pre-stat all objects concurrently (once at start)
store.pre_stat_and_cache(&objects, 100).await?;  // ✅ 200ms for 1000 objects

// PHASE 2: Download with zero stat overhead
for uri in &objects {
    let data = store.get(uri).await?;  // ✅ cached size, no stat! (128ms download only)
    process(data);
}
// Total time: ~128 seconds for 1000 objects (13% faster)
```

**Example: Benchmarking Tool Integration**

```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

async fn benchmark_download(bucket: &str, prefix: &str, object_count: usize) -> Result<()> {
    let store = store_for_uri(&format!("s3://{}/{}", bucket, prefix))?;
    
    // Get list of objects to benchmark
    let all_objects = store.list(&format!("s3://{}/{}", bucket, prefix), true).await?;
    let objects: Vec<String> = all_objects.into_iter().take(object_count).collect();
    
    println!("Benchmarking {} objects...", objects.len());
    
    // PRE-STAT PHASE: Load all object sizes concurrently (v0.9.10 NEW!)
    let start = std::time::Instant::now();
    let cached = store.pre_stat_and_cache(&objects, 100).await?;
    println!("Pre-statted {} objects in {:?}", cached, start.elapsed());
    
    // DOWNLOAD PHASE: Now downloads have zero stat overhead
    let download_start = std::time::Instant::now();
    let mut total_bytes = 0u64;
    
    for uri in &objects {
        let data = store.get(uri).await?;  // ✅ Uses cached size!
        total_bytes += data.len() as u64;
    }
    
    let duration = download_start.elapsed();
    let throughput_mbps = (total_bytes as f64 / 1_000_000.0) / duration.as_secs_f64();
    
    println!("Downloaded {} MB in {:?} ({:.2} MB/s)", 
             total_bytes / 1_000_000, duration, throughput_mbps);
    
    Ok(())
}
```

**Example: ML Dataset Pre-Loading**

```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

async fn preload_training_data(dataset_uris: Vec<String>) -> Result<Vec<Vec<u8>>> {
    let store = store_for_uri(&dataset_uris[0])?;
    
    // Pre-stat all files to populate size cache
    // This eliminates stat overhead during actual data loading
    store.pre_stat_and_cache(&dataset_uris, 200).await?;
    
    // Now load data with zero stat overhead
    let mut dataset = Vec::new();
    for uri in dataset_uris {
        let data = store.get(&uri).await?;
        dataset.push(data.to_vec());
    }
    
    Ok(dataset)
}
```

**Configuration Options:**

```rust
use s3dlio::object_store::{S3ObjectStore, S3Config, GcsObjectStore, GcsConfig};
use std::time::Duration;

// Custom TTL for size cache (default is 60 seconds)
let config = S3Config {
    enable_range_engine: false,
    range_engine: Default::default(),
    size_cache_ttl_secs: 120,  // 2-minute cache for longer workloads
};
let store = S3ObjectStore::with_config(config);

// Or use defaults (60 second TTL)
let store = S3ObjectStore::new();  // ✅ 60s TTL automatic
```

**Backend-Specific Behavior:**

| Backend | Cache TTL | Rationale |
|---------|-----------|-----------|
| **S3** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **GCS** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **Azure** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **file://** | 0 seconds (disabled) | Local stat is fast (~1ms), files can change on disk |
| **direct://** | 0 seconds (disabled) | Local stat is fast (~1ms), files can change on disk |

**Technical Details:**
- **Cache structure**: Thread-safe `RwLock<HashMap<String, CachedSize>>`
- **TTL mechanism**: Per-entry `Instant` timestamp, checked on `get()`
- **Concurrency**: `pre_stat_objects()` uses `futures::stream::buffer_unordered()`
- **Memory**: ~100 KB for 1000 entries (String + u64 + Instant per entry)
- **Backward compatibility**: Default trait methods ensure all backends work

**New Trait Methods:**

```rust
#[async_trait]
pub trait ObjectStore: Send + Sync {
    // ... existing methods ...
    
    /// Stat multiple objects concurrently (NEW in v0.9.10)
    async fn pre_stat_objects(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<HashMap<String, u64>> {
        // Default concurrent implementation provided
    }
    
    /// Pre-stat and cache object sizes for later use (NEW in v0.9.10)
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Default implementation (backends with cache override this)
    }
}
```

**When to Use Pre-Stat:**
- ✅ Benchmarking many objects (100+)
- ✅ Known object list before download
- ✅ Objects don't change during workload
- ✅ Network storage (S3, GCS, Azure)
- ❌ Single object downloads
- ❌ Objects changing frequently
- ❌ Local file:// or direct:// storage

**Performance Scaling:**

| Objects | Sequential Stat | Concurrent Pre-Stat | Speedup |
|---------|-----------------|---------------------|---------|
| 100 | 2.0s | 20ms | 100x faster |
| 500 | 10.0s | 100ms | 100x faster |
| 1000 | 20.0s | 200ms | 100x faster |
| 5000 | 100.0s | 1.0s | 100x faster |

**Testing:**
- 13 ObjectSizeCache unit tests (basic ops, TTL, concurrency, memory efficiency)
- Integration tests for file:// backend
- Performance validation tests (benchmarking scenarios)

**Migration Guide:**

No migration required! All existing code works without changes. To opt-in to performance improvements:

```rust
// Add this one line before your download loop:
store.pre_stat_and_cache(&object_uris, 100).await?;

// Then use store.get() as normal - it will use cached sizes
```

---

## Version 0.9.9 - Buffer Pool Optimization for DirectIO (October 2025)

### 🚀 **Major Performance Improvement**

#### **Buffer Pool for DirectIO Hot Path - 15-20% Throughput Gain**

Eliminated allocation churn in DirectIO range reads by integrating buffer pool infrastructure:

**Problem Solved:**
- DirectIO range reads allocated fresh aligned buffers on every operation
- Full buffer copies via `buffer[start..end].to_vec()` caused CPU overhead
- Allocator churn resulted in excessive page faults (20-25% throughput gap vs vdbench)

**Solution - Phase 1:**
- Wired existing `BufferPool` infrastructure into DirectIO hot path
- Pool of 32 × 64MB aligned buffers automatically initialized in `direct_io()` and `high_performance()` constructors
- Optimized `try_read_range_direct()` with borrow/return pattern:
  - **Before**: Fresh allocation + full buffer copy per range
  - **After**: Pool borrow + small copy (only requested bytes) + pool return
  - **Eliminated**: 2 allocations + 1 full copy per range operation

**Performance Impact (Expected):**
- **Throughput**: +15-20% on DirectIO with RangeEngine enabled
- **CPU**: -10-15% utilization (less memcpy, less malloc/free)
- **Page faults**: -30-50% reduction (less allocator activity)
- **Allocator calls**: -90% reduction (buffer reuse)

**Backward Compatibility:**
- ✅ **Zero breaking changes** - buffer pool is optional (`Option<Arc<BufferPool>>`)
- ✅ **Automatic** - `direct_io()` and `high_performance()` constructors initialize pool automatically
- ✅ **Graceful fallback** - Falls back to allocation if pool disabled or exhausted
- ✅ **Default unchanged** - `FileSystemConfig::default()` has no pool (backward compatible)

**API Usage (No Code Changes Required):**

```rust
// Using factory functions (pool auto-initialized for DirectIO)
let store = direct_io_store_for_uri("file:///data/")?;  // ✅ Pool automatic

// Using constructors (pool auto-initialized)
let config = FileSystemConfig::direct_io();  // ✅ Pool automatic (32 × 64MB buffers)
let config = FileSystemConfig::high_performance();  // ✅ Pool automatic

// Default (no pool, backward compatible)
let config = FileSystemConfig::default();  // ✅ No pool (compatible with v0.9.8)
```

**Technical Details:**
- Pool capacity: 32 buffers
- Buffer size: 64 MB per buffer
- Alignment: System page size (typically 4096 bytes)
- Grow-on-demand: Allocates new buffer if pool exhausted
- Async-safe: Uses tokio channels for thread-safe borrow/return

**Why Only DirectIO?**
- **S3/Azure/GCS**: Network latency (5-50ms) >> allocation (<0.1ms), pool would provide <1% benefit
- **Regular file I/O**: Kernel page cache already handles efficiency
- **DirectIO**: Aligned allocations expensive + frequent operations = pool is critical

**Files Modified:**
- `src/file_store_direct.rs`: Buffer pool integration (~80 lines)
- `src/file_store.rs`: Pre-sizing optimization (~2 lines)
- `tests/test_buffer_pool_directio.rs`: 12 comprehensive functional tests (new)
- `tests/test_allocation_comparison.rs`: Allocation overhead comparison (new)
- `docs/implementation-plans/v0.9.9-buffer-pool-enhancement.md`: Complete implementation plan (new)
- `docs/testing/v0.9.9-phase1-testing-summary.md`: Testing analysis (new)

**Testing:**
- ✅ 12/12 functional tests passing
- ✅ Concurrent access validated (64 tasks with pool size 32)
- ✅ Edge cases covered (unaligned offsets, oversized ranges, fallback)
- ✅ Data integrity verified across multiple chunks
- ✅ Zero new compiler warnings
- ⏳ Performance validation pending with sai3-bench (realistic workloads)

**Future Phases:**
- Phase 2: Writer path optimizations (optional)
- Phase 3: Zero-copy via custom Bytes deallocator (future)
- Target: Close remaining performance gap to <10% vs vdbench

---

### 📚 **Documentation Cleanup**

**Streamlined documentation from 114 to 34 files (70% reduction):**

**Removed:**
- Entire `docs/archive/` directory (584KB of historical docs)
  - Old API versions (pre-v0.9.0)
  - Completed implementation plans (v0.8.x)
  - Old performance reports and benchmarks
  - Pre-v0.8.0 changelog and release notes
- Version-specific docs (v0.9.4, v0.9.5, v0.9.6)
- Old API snapshots (v0.9.2, v0.9.3)
- Detailed release plans for v0.9.0-v0.9.2 (15 files)
- Completed implementation plans (4 files)

**Added:**
- `docs/STREAMING-ARCHITECTURE.md`: Comprehensive guide to stream-based patterns
  - RangeEngine streaming patterns
  - Backpressure control mechanisms
  - Cancellation support
  - DataLoader streaming architecture

**Kept (Essential):**
- Core guides: README, Changelog, RELEASE-CHECKLIST, TESTING-GUIDE
- Configuration: CONFIGURATION-HIERARCHY, VERSION-MANAGEMENT
- API references: ZERO-COPY-API-REFERENCE, TFRECORD-INDEX-QUICKREF
- Performance: Performance_Optimization_Summary, Performance_Profiling_Guide

**Rationale:**
- Changelog.md is authoritative source for version history
- Reduced maintenance burden and improved navigation
- Current API documented in code and reference docs

---

## Version 0.9.8 - Optional GCS Backends & Page Cache Configuration (October 2025)

### 🚀 **New Features**

#### **Optional Google Cloud Storage (GCS) Backends**

Added **two mutually exclusive GCS backend implementations** selectable at compile time:

1. **`gcs-community`** (default) - Uses community-maintained `gcloud-storage` v1.1 crate
   - ✅ **Production Ready**: All tests pass reliably (10/10)
   - ✅ Stable and proven in production workloads
   - ✅ Full ADC (Application Default Credentials) support
   - ✅ Supports all GCS operations: GET, PUT, DELETE, LIST, STAT, range reads

2. **`gcs-official`** (experimental) - Uses official Google `google-cloud-storage` v1.1 crate
   - ⚠️ **Experimental**: Known transport flakes in test suites (10-20% failure rate)
   - ✅ Individual operations work correctly when tested in isolation
   - ⚠️ Full test suite has intermittent "transport error" failures (improved to 8-9/10 tests passing)
   - 🐛 **Root Cause**: Upstream flake in `google-cloud-rust` library
     - **Bug Report**: https://github.com/googleapis/google-cloud-rust/issues/3574
     - **Related Issue**: https://github.com/googleapis/google-cloud-rust/issues/3412
   - ✅ **MAJOR IMPROVEMENT**: Implemented global client singleton pattern
     - **Before**: 30% pass rate (3/10 tests) - StorageControl operations failed consistently
     - **After**: 80-90% pass rate (8-9/10 tests) - Most operations now work reliably
     - **Implementation**: `once_cell::Lazy<OnceCell<GcsClient>>` for single client per process
   - 📊 **Current Status**: Acceptable for development/testing but expect occasional flakes until upstream fix

**Build Options:**

```bash
# Default (community backend - RECOMMENDED)
cargo build --release

# Explicit community backend
cargo build --release --features gcs-community

# Official backend (experimental - for testing only)
cargo build --release --no-default-features --features native-backends,s3,gcs-official

# ❌ Cannot use both (compile error by design)
cargo build --features gcs-community,gcs-official  # ERROR
```

**Why Two Backends?**

The dual-backend approach allows:
- ✅ Production stability with `gcs-community` (default)
- ✅ Future migration path when upstream transport issues are resolved
- ✅ A/B performance testing and benchmarking
- ✅ Easy switching without code changes (compile-time selection only)

**Recommendation**: Use `gcs-community` (default) for all production workloads. The `gcs-official` backend is provided for experimentation and to track upstream development, but is not suitable for production due to the transport flakes documented in Issue #3574.

**Files Changed:**
- `src/google_gcs_client.rs`: New implementation using official `google-cloud-storage` crate
- `src/gcs_client.rs`: Community implementation (default)
- `src/object_store.rs`: Factory pattern with compile-time backend selection
- `tests/test_gcs_community.rs`: Complete test suite for community backend (10/10 pass ✅)
- `tests/test_gcs_official.rs`: Complete test suite for official backend (3/10 pass, 7/10 fail ⚠️)
- `tests/common/mod.rs`: Shared test utilities
- `docs/GCS-BACKEND-SELECTION.md`: Comprehensive backend selection guide
- Bug report filed: https://github.com/googleapis/google-cloud-rust/issues/3574

---

#### **Configurable `posix_fadvise` Hints for File I/O**

Added ability to configure page cache behavior for file system operations (`file://` and `direct://` URIs) via the new `page_cache_mode` field in `FileSystemConfig`.

**What is `posix_fadvise`?**  
On Linux/Unix systems, `posix_fadvise()` provides hints to the kernel about expected file access patterns, allowing the OS to optimize page cache behavior for better performance.

**Available Modes:**
- **`Auto`** (default for full GETs): Automatically selects Sequential for large files (>= 64 MiB), Random for small files
- **`Sequential`**: Prefetch data ahead - optimal for streaming reads, large sequential scans
- **`Random`**: Don't prefetch - optimal for random access patterns, database queries
- **`DontNeed`**: Don't cache pages - optimal for one-time reads, streaming that won't be re-read
- **`Normal`**: Let OS use default heuristics

**Default Behavior (No Configuration Required):**
- Full GET operations: `Auto` mode (intelligent based on file size)
- Range GET operations: `Random` mode (typical for random access)

**Rust API Usage:**

```rust
use s3dlio::object_store::{store_for_uri_with_config, FileSystemConfig, PageCacheMode};

// Example 1: Sequential access pattern (streaming, large files)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::Sequential),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///data/", Some(config))?;

// Example 2: Random access pattern (database, random seeks)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::Random),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///db/", Some(config))?;

// Example 3: Don't pollute cache (one-time streaming)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::DontNeed),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///stream/", Some(config))?;

// Example 4: Use defaults (Auto for full GET, Random for range GET)
let store = store_for_uri("file:///data/")?;  // No config needed
```

**When to Use:**
- ✅ **Sequential**: Large file streaming, sequential scans, media processing
- ✅ **Random**: Database files, random access patterns, sparse file access
- ✅ **DontNeed**: Large one-time reads that won't be re-accessed (prevents cache pollution)
- ✅ **Auto**: Default - works well for most workloads

**Performance Impact:**
- Sequential mode: Can provide 2-3x improvement for large sequential reads
- Random mode: Reduces memory pressure for random access patterns
- DontNeed mode: Prevents cache pollution for one-time large file operations

**Files Changed:**
- `src/file_store.rs`: Added `page_cache_mode` field to `FileSystemConfig`, updated GET operations
- `src/api.rs`: Exported `PageCacheMode`, `FileSystemConfig`, and `store_for_uri_with_config`
- `tests/test_file_range_engine.rs`: Updated tests for new field

**Backward Compatibility:**  
✅ **Fully backward compatible** - existing code continues to work with Auto/Random defaults

---

## Version 0.9.6 - RangeEngine Disabled by Default (October 2025)

### ⚠️ **BREAKING CHANGES**

#### **RangeEngine Disabled by Default (ALL Backends)**

**Problem:** Performance testing revealed that RangeEngine causes **up to 50% slowdown** for typical workloads due to the extra HEAD/STAT request required on every GET operation to determine object size. While RangeEngine provides 30-50% throughput improvement for large files (>= 16 MiB), the mandatory stat overhead makes it counterproductive for most real-world workloads.

**Solution:** RangeEngine is now **disabled by default** across all storage backends. Users must explicitly opt-in for large-file workloads where benefits outweigh the stat overhead.

**Affected Backends:**
- ✅ **Azure Blob Storage** (`az://`) - `enable_range_engine: false`
- ✅ **Google Cloud Storage** (`gs://`, `gcs://`) - `enable_range_engine: false`
- ✅ **Local File System** (`file://`) - `enable_range_engine: false`
- ✅ **DirectIO** (`direct://`) - `enable_range_engine: false`
- ✅ **S3** (`s3://`) - RangeEngine not yet implemented

**Performance Impact:**
- **Before (v0.9.5)**: Every GET operation performed HEAD + GET (2 requests)
- **After (v0.9.6)**: GET operations use single request (no stat overhead)
- **Typical workloads**: 50% faster due to elimination of HEAD requests
- **Large-file workloads**: Must enable RangeEngine explicitly to get 30-50% benefit

**Migration Guide:**

**No changes required for most users** - default behavior now faster for typical workloads.

**For large-file workloads (>= 16 MiB objects), explicitly enable RangeEngine:**

```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig, RangeEngineConfig};

// Enable RangeEngine for large-file Azure workload
let config = AzureConfig {
    enable_range_engine: true,  // Explicitly enable (was: default true, now: default false)
    range_engine: RangeEngineConfig {
        min_split_size: 16 * 1024 * 1024,  // 16 MiB threshold (default)
        max_concurrent_ranges: 32,          // 32 parallel ranges
        chunk_size: 64 * 1024 * 1024,      // 64 MiB chunks
        ..Default::default()
    },
};
let store = AzureObjectStore::with_config(config);
```

**Same pattern for all backends:**

```rust
// Google Cloud Storage
use s3dlio::object_store::{GcsObjectStore, GcsConfig};
let config = GcsConfig {
    enable_range_engine: true,  // Opt-in for large files
    ..Default::default()
};
let store = GcsObjectStore::with_config(config);

// Local File System
use s3dlio::file_store::{FileSystemObjectStore, FileSystemConfig};
let config = FileSystemConfig {
    enable_range_engine: true,  // Rarely beneficial for local FS
    ..Default::default()
};
let store = FileSystemObjectStore::with_config(config);

// DirectIO
use s3dlio::file_store_direct::{FileSystemConfig};
let config = FileSystemConfig::direct_io();  // Still disabled by default
// Explicitly enable if needed:
let mut config = FileSystemConfig::direct_io();
config.enable_range_engine = true;
```

**When to Enable RangeEngine:**
- ✅ Large-file workloads (average object size >= 64 MiB)
- ✅ High-bandwidth networks (>= 1 Gbps) with high latency
- ✅ Dedicated large-object operations (media processing, ML training)
- ❌ Mixed workloads with small and large objects
- ❌ Benchmarks with small objects (< 16 MiB)
- ❌ Local file systems (seek overhead usually outweighs benefit)

---

### 🔧 **Configuration Changes**

#### **All Backend Configs Updated**

1. **`AzureConfig`** (`src/object_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation updated to reflect opt-in behavior

2. **`GcsConfig`** (`src/object_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation updated to reflect opt-in behavior

3. **`FileSystemConfig`** (`src/file_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation emphasizes local FS rarely benefits

4. **`FileSystemConfig` (DirectIO)** (`src/file_store_direct.rs`)
   - `enable_range_engine`: `true` → `false` in:
     - `Default::default()`
     - `direct_io()`
     - `high_performance()`
   - All presets now disable RangeEngine by default

#### **Constants Documentation Updated**

- `DEFAULT_RANGE_ENGINE_THRESHOLD` documentation clarifies:
  - RangeEngine is **disabled by default**
  - 16 MiB threshold applies only when explicitly enabled
  - Explains performance trade-offs
  - Provides example configurations for enabling

---

### 📊 **Performance Summary**

| Workload Type | v0.9.5 Performance | v0.9.6 Performance (Default) | v0.9.6 with RangeEngine Enabled |
|---------------|-------------------|------------------------------|----------------------------------|
| Small objects (< 16 MiB) | **Slow** (2x requests: HEAD + GET) | **Fast** (1x request: GET only) | **Slow** (2x requests: HEAD + GET) |
| Large files (>= 64 MiB) | **Medium** (HEAD + single GET) | **Medium** (single GET) | **Fast** (HEAD + parallel range GETs) |
| Mixed workloads | **Slow overall** (all GETs statted) | **Fast** (no stat overhead) | **Slow overall** (all GETs statted) |

**Key Insight:** RangeEngine provides benefit **only for large-file workloads**. For typical workloads with mixed object sizes, the stat overhead on small objects outweighs gains on large objects.

---

### 📚 **Documentation Updates**

- **Constants (`src/constants.rs`)**: Comprehensive RangeEngine configuration guide
- **Config Structs**: All backend configs document opt-in behavior
- **README.md**: Updated to reflect disabled-by-default status
- **Copilot Instructions**: Updated RangeEngine guidance

---

### 🔄 **Backward Compatibility**

**Breaking Change:** Existing code that relies on RangeEngine being enabled by default will see different behavior. However, this change **improves performance for most workloads**.

**Migration Path:**
1. **If your workload uses small/mixed objects**: No changes needed - enjoy faster performance
2. **If your workload uses large files (>= 64 MiB)**: Add explicit `enable_range_engine: true` to config

**Deprecations:** None. All APIs remain stable.

---

## Version 0.9.5 - Performance Fixes & RangeEngine Tuning (October 2025)

### 🚀 **PERFORMANCE IMPROVEMENTS**

#### **1. Adaptive Concurrency for Delete Operations (10-70x faster)**

Fixed critical performance regression introduced in v0.8.23 where progress bar implementation caused delete operations to become sequential instead of concurrent.

**Performance Gains:**
- **500 objects**: ~0.7 seconds (was ~50 seconds) - **70x faster**
- **7,000 objects**: ~5.5 seconds (was ~70-140 seconds) - **12-25x faster**
- **93,000 objects**: ~90 seconds (was ~15+ minutes) - **10x+ faster**

**Implementation Details:**

1. **New `delete_objects_concurrent()` Helper Function** (`src/object_store.rs`)
   - Universal implementation works with all backends (S3, Azure, GCS, file://, direct://)
   - Uses `futures::stream` with `buffer_unordered` for concurrent deletions
   - Batched progress updates (every 50 operations) instead of per-object updates
   - 98% reduction in progress bar overhead

2. **Adaptive Concurrency Algorithm**
   
   Automatically scales concurrency based on total object count (10% of total):
   
   | Total Objects | Concurrency | Reasoning |
   |--------------|-------------|-----------|
   | < 10 | 1 | Very small: sequential is efficient |
   | 10-99 | 10 | Small: minimum viable concurrency |
   | 100-9,999 | total/10 (max 100) | Medium: scales with workload |
   | 10,000+ | total/10 (max 1,000) | Large: capped to avoid overwhelming backends |
   
   **Examples:**
   - 100 objects → 10 concurrent deletions
   - 500 objects → 50 concurrent deletions
   - 5,000 objects → 100 concurrent deletions (capped)
   - 93,000 objects → 1,000 concurrent deletions (capped)

3. **CLI Updates** (`src/bin/cli.rs`)
   - Both pattern-filtered and full-prefix deletions now use concurrent helper
   - Displays adaptive concurrency level in user messages
   - Maintains smooth progress bar with batched updates

**Testing:**
- ✅ Tested with Google Cloud Storage: 7,010 objects deleted in 5.5 seconds
- ✅ Throughput: ~1,280 deletions/second
- ✅ Progress bar verified working correctly
- ✅ Universal backend compatibility confirmed

**Technical Notes:**
- Progress updates use `Arc<AtomicUsize>` for lock-free concurrent counting
- Callback wrapped in `Arc` for sharing across async tasks
- ProgressBar cloned to avoid move issues in closures
- Final progress update ensures accurate completion count

---

#### **2. RangeEngine Threshold Increased to 16 MiB (Fixes 10% regression)**

**Problem:** v0.9.3 introduced a 10% performance regression for small object workloads (e.g., 1 MiB objects) due to an extra HEAD request on every GET operation to check object size for RangeEngine eligibility.

**Root Cause Analysis:**
- v0.9.3 used 4 MiB threshold for RangeEngine
- For objects < 4 MiB, code still performed HEAD + GET (2 requests instead of 1)
- Benchmarks with 1 MiB objects saw 60% more total requests → ~10% slowdown
- The stat overhead outweighed any potential benefit for small objects

**Solution:** Raised default threshold to **16 MiB** for all network backends (S3, Azure, GCS).

**Changes:**

1. **New Universal Constant** (`src/constants.rs`)
   ```rust
   /// Universal default minimum object size to trigger RangeEngine (16 MiB)
   pub const DEFAULT_RANGE_ENGINE_THRESHOLD: u64 = 16 * 1024 * 1024;
   ```
   
   - Replaces backend-specific thresholds (4 MiB)
   - Legacy aliases deprecated but maintained for compatibility
   - Comprehensive documentation explaining threshold selection

2. **Updated Backend Configs**
   - `AzureConfig`: Now uses 16 MiB threshold
   - `GcsConfig`: Now uses 16 MiB threshold
   - `S3Config`: Now uses 16 MiB threshold (when RangeEngine added)

**Performance Impact:**
- **Small objects (< 16 MiB)**: Restored to v0.8.22 performance (single GET request)
- **Medium objects (16-64 MiB)**: Still benefit from RangeEngine (20-40% faster)
- **Large objects (> 64 MiB)**: Maximum RangeEngine benefit (30-60% faster)
- **Benchmarks**: 1 MiB object workloads no longer see regression

**When to Override:**

```rust
use s3dlio::object_store::{GcsConfig, RangeEngineConfig};

// Lower threshold for large-file workloads (higher latency networks)
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        min_split_size: 4 * 1024 * 1024,  // 4 MiB threshold
        ..Default::default()
    },
};

// Higher threshold to avoid stat overhead on most objects
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        min_split_size: 64 * 1024 * 1024,  // 64 MiB threshold
        ..Default::default()
    },
};

// Disable RangeEngine entirely for small-object benchmarks
let benchmark_config = GcsConfig {
    enable_range_engine: false,
    ..Default::default()
};
```

**Documentation Updates:**
- Added detailed comments in `src/constants.rs` explaining threshold rationale
- Updated Azure/GCS config documentation to reflect 16 MiB threshold
- Performance regression analysis documented

**Testing:**
- ✅ Benchmarks with 1 MiB objects now match v0.8.22 performance
- ✅ Large file downloads still use RangeEngine (>= 16 MiB)
- ✅ No extra HEAD requests for typical workloads
- ✅ Configuration override tested and working

---

### 📊 **Performance Summary**

| Change | Workload | Performance Gain |
|--------|----------|------------------|
| Adaptive Delete | 7,000 objects | 12-25x faster (5.5s vs 70-140s) |
| RangeEngine Fix | 1 MiB GETs | 10% regression eliminated |
| Combined | Mixed workloads | Restored + improved performance |

---

### 🔧 **Breaking Changes**
None. All changes are backward compatible with optional configuration overrides.

---

### 📚 **Deprecations**
- `DEFAULT_S3_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`
- `DEFAULT_AZURE_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`
- `DEFAULT_GCS_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`

Legacy constants remain functional but emit deprecation warnings.

---

## Version 0.9.4 - S3-Specific API Deprecation (October 2025)

### ⚠️ **DEPRECATION NOTICES**

#### **Python & Rust API: S3-Specific Data Functions Deprecated (Removal in v1.0.0)**

Two S3-specific data operation functions have been deprecated in favor of universal URI-based alternatives. **These functions will be removed in v1.0.0.**

**IMPORTANT**: 
- This deprecation affects **BOTH Python AND Rust APIs**
- `create_bucket()` and `delete_bucket()` are **NOT deprecated** - they will be made universal in future releases

**Deprecated Functions:**

1. **`list_objects(bucket, prefix, recursive)`** → Use `list(uri, recursive, pattern)`
   
   **Python:**
   ```python
   # OLD (deprecated)
   objects = s3dlio.list_objects("bucket", "prefix/", recursive=True)
   
   # NEW (universal)
   objects = s3dlio.list("s3://bucket/prefix/", recursive=True)
   ```
   
   **Rust:**
   ```rust
   // OLD (deprecated)
   use s3dlio::s3_utils::list_objects;
   let objects = list_objects("bucket", "prefix/", true)?;
   
   // NEW (universal)
   use s3dlio::api::{store_for_uri, ObjectStore};
   let store = store_for_uri("s3://bucket/prefix/")?;
   let objects = store.list("", true, None).await?;
   ```

2. **`get_object(bucket, key, offset, length)`** → Use `get(uri)` or `get_range(uri, offset, length)`
   
   **Python:**
   ```python
   # OLD (deprecated)
   data = s3dlio.get_object("bucket", "key", offset=1024, length=4096)
   
   # NEW (universal)
   data = s3dlio.get_range("s3://bucket/key", offset=1024, length=4096)
   ```

**Timeline:**
- **v0.9.4**: Functions work with deprecation warnings (stderr + compile-time)
- **v0.9.x → v1.0.0-rc**: Functions continue working with warnings
- **v1.0.0**: Functions removed

**See**: [DEPRECATION-NOTICE-v0.9.4.md](./DEPRECATION-NOTICE-v0.9.4.md) for complete migration guide.

---

## Version 0.9.3 - RangeEngine for Azure & GCS Backends (October 2025)

#### **1. RangeEngine Integration for Azure Blob Storage**
Concurrent range downloads significantly improve throughput for large Azure blobs by hiding network latency with parallel range requests.

**Performance Gains:**
- Medium blobs (4-64MB): 20-40% faster
- Large blobs (> 64MB): 30-50% faster
- Huge blobs (> 1GB): 40-60% faster on high-bandwidth networks

**Implementation:**
- Created `AzureConfig` with configurable RangeEngine settings
- Refactored `AzureObjectStore` from unit struct to stateful struct with Clone support
- Added `get_with_range_engine()` helper for concurrent downloads
- Size-based strategy: files >= 4MB use RangeEngine automatically

**Configuration (network-optimized defaults):**
```rust
pub struct AzureConfig {
    pub enable_range_engine: bool,  // Default: true
    pub range_engine: RangeEngineConfig {
        chunk_size: 64 * 1024 * 1024,        // 64MB chunks
        max_concurrent_ranges: 32,            // 32 parallel ranges
        min_split_size: 4 * 1024 * 1024,     // 4MB threshold
        range_timeout: Duration::from_secs(30),
    },
}
```

**Usage:**
```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig};

// Default configuration (RangeEngine enabled)
let store = AzureObjectStore::new();

// Custom configuration
let config = AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig::default(),
};
let store = AzureObjectStore::with_config(config);

// Downloads automatically use RangeEngine for files >= 4MB
let data = store.get("az://account/container/large-file.bin").await?;
```

**Validation:**
- ✅ Python tests: 16.54 MB/s download (8MB blob)
- ✅ Rust tests: All ObjectStore methods validated
- ✅ Zero-copy Bytes API compatibility
- ✅ Builds with zero warnings

---

#### **2. RangeEngine Integration for Google Cloud Storage**
Complete GCS backend enhancement with concurrent range downloads following the same pattern as Azure.

**Performance:**
- 128MB file: 44-46 MB/s with 2 concurrent ranges
- Validated on production GCS buckets (signal65-russ-b1)
- Expected 30-50% gains on high-bandwidth networks (>1 Gbps)

**Implementation:**
- Created `GcsConfig` with identical structure to AzureConfig
- Refactored `GcsObjectStore` from unit struct to stateful struct
- Added `get_with_range_engine()` for concurrent downloads
- Same 4MB threshold and network-optimized defaults

**Configuration:**
```rust
use s3dlio::object_store::{GcsObjectStore, GcsConfig};

// Default configuration (RangeEngine enabled)
let store = GcsObjectStore::new();

// Custom configuration
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 64 * 1024 * 1024,
        max_concurrent_ranges: 32,
        min_split_size: 4 * 1024 * 1024,
        range_timeout: Duration::from_secs(30),
    },
};
let store = GcsObjectStore::with_config(config);
```

**Validation:**
- ✅ Python tests: 45.61 MB/s download (128MB file, 2 concurrent ranges)
- ✅ Rust smoke tests: All 5 tests passing (8.06s total)
  - Small files (1MB): Simple download below threshold
  - Large files (128MB): 44.60 MB/s with RangeEngine
  - Range requests: Partial reads validated
  - Metadata: stat() with ETag
  - Listings: Directory operations working
- ✅ Debug logging confirms concurrent range execution
- ✅ Zero-copy Bytes API compatibility

---

#### **3. Universal Python API Enhancements**
All Python API functions now work universally across all 5 backends (S3, Azure, GCS, file://, direct://).

**Changes:**
- `put()`: Template parameter now optional (default: "object-{}")
- `get()`: Universal implementation via `store_for_uri()` + `ObjectStore::get()`
- `delete()`: Works with all URI schemes, supports pattern matching
- `build_uri_list()`: Generic URI parsing for any scheme

**Before (v0.9.2 - S3-specific):**
```python
# Only worked with s3:// URIs
s3dlio.put("s3://bucket/prefix/", num=3, template="object-{}", size=1024*1024)
```

**After (v0.9.3 - Universal):**
```python
# Works with all backends
s3dlio.put("s3://bucket/prefix/", num=3, size=1024*1024)        # S3
s3dlio.put("az://account/container/prefix/", num=3, size=1024*1024)  # Azure
s3dlio.put("gs://bucket/prefix/", num=3, size=1024*1024)        # GCS
s3dlio.put("file:///local/path/", num=3, size=1024*1024)        # Local
s3dlio.put("direct:///local/path/", num=3, size=1024*1024)      # DirectIO

# get() works universally
data = s3dlio.get("gs://bucket/file.bin")  # Returns BytesView (zero-copy)

# delete() works with patterns
s3dlio.delete(["az://container/prefix/*"])  # Deletes all matching
```

**Validation:**
- ✅ Azure Python tests: All operations working (put, get, get_range, large files)
- ✅ GCS Python tests: All operations working (128MB with RangeEngine)
- ✅ Zero-copy BytesView wrapper preserved
- ✅ Pattern matching in delete() validated

---

### 🧪 **Testing & Validation**

#### **New Test Suites:**
1. **`python/tests/test_azure_api.py`**: Comprehensive Azure backend validation
2. **`python/tests/test_gcs_api.py`**: GCS backend with RangeEngine tests (128MB files)
3. **`tests/test_gcs_smoke.rs`**: Rust integration tests (5 test cases, all passing)

#### **Test Coverage:**
- Azure: 17.08 MB/s download (8MB with RangeEngine)
- GCS Python: 45.61 MB/s download (128MB, 2 concurrent ranges)
- GCS Rust: 44.60 MB/s download (128MB, 2 concurrent ranges)
- All ObjectStore methods validated: get, put, delete, list, stat, get_range
- Zero-copy Bytes API compatibility confirmed
- Debug logging added for RangeEngine activity tracking

---

### 📦 **Infrastructure Updates**

- **Build System**: Zero warnings policy enforced across all builds
- **Tracing**: Debug logging support for RangeEngine activity (`s3dlio.init_logging("debug")`)
- **Documentation**: Added GCS login helper script (`scripts/gcs-login.sh`)
- **Architecture**: Consistent backend patterns across Azure and GCS
  - Both use stateful structs with Config
  - Both support Clone for closure compatibility
  - Both use identical RangeEngine configuration

---

### 🔧 **Breaking Changes**
None. This is a backward-compatible feature release.

- Existing Azure code continues to work (defaults to RangeEngine enabled)
- Existing GCS code continues to work (defaults to RangeEngine enabled)
- Python API changes are purely additive (template parameter optional)
- All v0.9.2 code compiles and runs without modification

---

### 📊 **Performance Summary**

| Backend | File Size | Method | Throughput | Improvement |
|---------|-----------|--------|------------|-------------|
| Azure | 8MB | RangeEngine (1 range) | 16.54 MB/s | Baseline |
| GCS | 8MB | RangeEngine (1 range) | 22.97 MB/s | Baseline |
| GCS | 128MB | RangeEngine (2 ranges) | 44-46 MB/s | Network-limited |
| Expected | >1GB | RangeEngine (multi-range) | 30-50% faster | On fast networks |

**Note**: Performance gains depend on network bandwidth and latency. High-bandwidth networks (>1 Gbps) with higher latency will see the most benefit from concurrent range requests.

---

### 🚀 **Migration Guide**

#### **No Code Changes Required**
This release is fully backward compatible. All existing code continues to work without modification.

#### **Optional: Customize RangeEngine Configuration**
```rust
// Azure custom config
use s3dlio::object_store::{AzureObjectStore, AzureConfig, RangeEngineConfig};

let config = AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 32 * 1024 * 1024,  // 32MB chunks instead of 64MB
        max_concurrent_ranges: 16,      // 16 concurrent instead of 32
        min_split_size: 8 * 1024 * 1024,  // 8MB threshold instead of 4MB
        range_timeout: Duration::from_secs(60),
    },
};
let store = AzureObjectStore::with_config(config);
```

#### **Optional: Disable RangeEngine**
```rust
let config = AzureConfig {
    enable_range_engine: false,  // Disable concurrent ranges
    ..Default::default()
};
let store = AzureObjectStore::with_config(config);
```

---

### 📚 **Documentation Updates**
- Updated README.md with v0.9.3 features
- Updated Changelog.md (this file)
- API documentation includes RangeEngine configuration examples
- Test guides updated with new test suites

---

## Version 0.9.2 - CancellationToken & Configuration Rationalization (October 2025)

### 🎯 **New Features**

#### **1. CancellationToken Infrastructure for Graceful Shutdown**
Comprehensive cancellation support across all DataLoader components enables clean shutdown of data loading operations.

**Components Updated**:
- `LoaderOptions`: Added `cancellation_token` field with builder methods
- `spawn_prefetch()`: Added cancel_token parameter with loop check
- `DataLoader`: Cancellation checks in all 3 spawn paths (map-known, iterable, map-unknown)
- `AsyncPoolDataLoader`: Cancellation support in async pool worker with 3 checkpoints

**Usage**:
```rust
use tokio_util::sync::CancellationToken;

let cancel_token = CancellationToken::new();

let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token.clone());

// Spawn Ctrl-C handler
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    cancel_token.cancel();
});

let loader = DataLoader::new(dataset, options);
let mut stream = loader.stream();

while let Some(batch) = stream.next().await {
    train_step(batch?).await?;
}
// Clean shutdown on Ctrl-C
```

**Behavior**:
- ✅ Workers exit cleanly without submitting new requests
- ✅ In-flight requests drain naturally
- ✅ MPSC channels properly closed
- ✅ No orphaned background tasks
- ✅ Zero overhead when token not cancelled

---

#### **2. Configuration Hierarchy Rationalization**
Clear three-level configuration design aligned with PyTorch DataLoader concepts for ML practitioners.

**Documentation Added**:
- `docs/CONFIGURATION-HIERARCHY.md` - Comprehensive analysis
- `docs/api/rust-api-v0.9.2.md` - Updated from v0.9.0 with hierarchy section
- `docs/api/python-api-v0.9.2.md` - Updated from v0.9.0 with Python-specific guidance

**Three Levels**:
1. **LoaderOptions** (User-Facing, Training-Centric)
   - Like PyTorch's `DataLoader(batch_size, num_workers, ...)`
   - Controls WHAT batches to create and HOW to iterate
   - Always visible to users
   
2. **PoolConfig** (Performance Tuning, Optional)
   - Like PyTorch's internal worker pool management
   - Controls HOW data is fetched efficiently
   - Good defaults via `stream()`, advanced tuning via `stream_with_pool()`
   
3. **RangeEngineConfig** (Internal Optimization, Hidden)
   - Like file I/O internals in PyTorch Dataset
   - Controls storage-layer parallel range requests
   - Backend-specific, automatically configured

**Code Enhancement**:
```rust
// NEW: Convenience constructor to bridge Level 1 → Level 2
let pool_config = PoolConfig::from_loader_options(&options);
// Derives: pool_size from num_workers, readahead_batches from prefetch
```

**Design Philosophy**:
- Progressive complexity: Simple by default, powerful when needed
- PyTorch alignment: Familiar concepts for ML practitioners
- Separation of concerns: Training logic vs storage optimization
- Good defaults: Most users never touch Level 2 or 3

---

### 🧪 **Testing Improvements**

#### **Cancellation Test Suite**
Comprehensive test coverage in `tests/test_cancellation.rs`:

**9 Tests (All Passing ✅)**:
- DataLoader tests (5): pre-cancellation, during streaming, delayed, no-token, shared token
- AsyncPoolDataLoader tests (4): pre-cancellation, during streaming, stops new requests, idempotent

**Test Patterns**:
- Validates drain behavior (in-flight requests complete)
- Timeout-based completion checks (2s)
- Mock URIs for control flow testing
- Ctrl-C handler examples

**Documentation**:
- Updated `docs/TESTING-GUIDE.md` with cancellation testing section
- Test patterns, running instructions, expected behavior
- Configuration hierarchy integration notes
- Checklist for new components

---

### 📚 **Documentation Updates**

#### **API Documentation Versioned to v0.9.2**:
- Renamed `rust-api-v0.9.0.md` → `rust-api-v0.9.2.md`
- Renamed `python-api-v0.9.0.md` → `python-api-v0.9.2.md`
- Added "Configuration Hierarchy" sections
- Added "Graceful Shutdown" section with CancellationToken examples
- Updated "What's New" sections

#### **New Documentation Files**:
- `docs/CONFIGURATION-HIERARCHY.md` - Deep dive into three-level design
- PyTorch comparison tables
- Side-by-side examples (PyTorch vs s3dlio)
- Recommendations for users
- When to tune each level

---

### 🔧 **API Additions**

**LoaderOptions**:
```rust
pub struct LoaderOptions {
    // ... existing fields ...
    
    /// Optional cancellation token for graceful shutdown (NEW in v0.9.2)
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl LoaderOptions {
    pub fn with_cancellation_token(self, token: CancellationToken) -> Self;
    pub fn without_cancellation(self) -> Self;
}
```

**PoolConfig**:
```rust
impl PoolConfig {
    /// NEW: Derive from LoaderOptions with sensible scaling
    pub fn from_loader_options(opts: &LoaderOptions) -> Self;
}
```

**spawn_prefetch**:
```rust
// NEW: Added cancel_token parameter
pub fn spawn_prefetch<F, Fut, T>(
    cap: usize,
    producer: F,
    cancel_token: Option<CancellationToken>,  // NEW
) -> Receiver<Result<T, DatasetError>>
```

---

### ⚡ **Performance**

- **Zero overhead** when cancellation token not used
- **Clean shutdown** typically completes within 2 seconds
- **No breaking changes** to existing code without cancellation
- **Maintains** 5+ GB/s reads, 2.5+ GB/s writes

---

### 🛠️ **Build Status**

- ✅ Zero compiler warnings
- ✅ All tests passing (including 9 new cancellation tests)
- ✅ Clean release builds
- ✅ Documentation comprehensive and versioned

---

### 📦 **Migration Notes**

**No breaking changes** - all features are additive:

**To add graceful shutdown**:
```rust
// Before (v0.9.1)
let options = LoaderOptions::default().with_batch_size(32);

// After (v0.9.2) - optional cancellation
let cancel_token = CancellationToken::new();
let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token);
```

**To use PoolConfig convenience constructor**:
```rust
// NEW in v0.9.2
let pool_config = PoolConfig::from_loader_options(&options);
let stream = loader.stream_with_pool(pool_config);
```

---

### 🎯 **Key Use Cases**

**Training Loops with Ctrl-C Handling**:
- Clean shutdown prevents corrupted checkpoints
- No orphaned background tasks
- Proper resource cleanup

**Multi-Loader Coordination**:
- Single token cancels multiple loaders
- Synchronized shutdown across components

**Long-Running Jobs**:
- Graceful termination on SIGTERM
- Clean exit from infinite data streams

---

### 📖 **Further Reading**

- **Configuration Guide**: `docs/CONFIGURATION-HIERARCHY.md`
- **Rust API**: `docs/api/rust-api-v0.9.2.md`
- **Python API**: `docs/api/python-api-v0.9.2.md`
- **Testing Guide**: `docs/TESTING-GUIDE.md` (cancellation section)

---

## Version 0.9.1 - True Zero-Copy Python API (October 2025)

### 🎯 **Critical Fixes**

#### **1. Zero-Copy Python API (TRUE Implementation)**
- **Fixed false claim**: v0.9.0 claimed zero-copy but actually copied data to Python bytes
- **New `BytesView` class**: Wraps Rust `Bytes`, exposes `.memoryview()` for true zero-copy
- **Python buffer protocol**: Native support without numpy dependency
- **Memory reduction**: 10-15% reduction in typical AI/ML workflows

**Zero-Copy Functions (Returns `BytesView`)**:
- `get(uri)` → `BytesView` (was `bytes`)
- `get_range(uri, offset, length)` → `BytesView` (NEW)
- `get_many(uris)` → `List[(str, BytesView)]` (was `List[(str, bytes)]`)
- `get_object(bucket, key)` → `BytesView` (was `bytes`)
- `CheckpointStore.load_latest()` → `Optional[BytesView]` (was `Optional[bytes]`)
- `CheckpointReader.read_shard_by_rank()` → `BytesView` (was `bytes`)

**Usage**:
```python
# Zero-copy access
view = s3dlio.get("s3://bucket/data.bin")
arr = np.frombuffer(view.memoryview(), dtype=np.float32)  # No copy!

# Or copy if needed
data = view.to_bytes()  # Explicit copy
```

**NOT Zero-Copy** (copies data):
- `PyDataset.get_item()` → still returns `bytes` (legacy trait)
- `PyBytesAsyncDataLoader.__next__()` → returns `bytes` (uses dataset)

See `docs/ZERO-COPY-API-REFERENCE.md` for complete details.

---

#### **2. Universal Backend Support for get_many()**
- **Fixed limitation**: `get_many()` was S3-only in v0.9.0
- **Universal support**: Now works with S3, Azure, GCS, File, DirectIO
- **Backend detection**: Automatically routes to appropriate implementation
- **S3**: Uses optimized `get_objects_parallel()`
- **File/DirectIO**: Parallel `tokio::fs::read()` with semaphore
- **Azure/GCS**: Uses universal `store_for_uri()` factory

**All URIs must use same scheme**:
```python
# ✅ Valid - all file://
s3dlio.get_many(["file:///a.bin", "file:///b.bin"])

# ❌ Invalid - mixed schemes
s3dlio.get_many(["s3://a", "file:///b"])  # Error
```

---

### ✨ **New Features**

#### **1. Universal Range Request Support**

**Python API**:
```python
# Read byte range (zero-copy)
view = s3dlio.get_range("s3://bucket/file", offset=1000, length=1024*1024)
arr = np.frombuffer(view.memoryview(), dtype=np.uint8)

# Read from offset to end
view = s3dlio.get_range("file:///path", 5000, None)
```

**CLI**:
```bash
# Range request
s3-cli get s3://bucket/file --offset 1000 --length 1024

# From offset to end
s3-cli get file:///path/file --offset 5000

# Single-object only (not with --recursive or --keylist)
```

**Backends**: S3, Azure, GCS, File, DirectIO

---

#### **2. Comprehensive Zero-Copy Refactor**

**Rust Library**:
- All `ObjectStore` helper methods return `Bytes` (not `Vec<u8>`)
- `get_with_validation()` → `Bytes`
- `get_range_with_validation()` → `Bytes`
- `get_optimized()` → `Bytes`
- `get_range_optimized()` → `Bytes`

**Checkpoint System**:
- All checkpoint reader methods return `Bytes`
- `read_shard()` → `Bytes`
- `read_shard_by_rank()` → `Bytes`
- `read_all_shards()` → `Vec<(u32, Bytes)>`
- `load_latest()` → `Option<Bytes>`

**S3 Utils**:
- `get_range()` uses `.slice()` not `.to_vec()`

---

### 🧪 **Testing**

**New Test Suites**:
1. **test_zero_copy_comprehensive.py** (6/6 tests pass):
   - BytesView structure validation
   - get() returns BytesView with working memoryview
   - NumPy array from memoryview (zero-copy verified)
   - get_many() universal backend support
   - Large file handling (1 MB)
   - BytesView immutability

2. **test_range_requests.py** (4/4 tests pass):
   - Range with offset+length
   - Range with offset only (to end)
   - Range from beginning
   - Full file comparison

**Existing Tests**:
- ✅ Rust library: 91/91 tests pass
- ✅ Python functionality: 27/27 tests pass
- ✅ Zero warnings in all builds

---

### 📝 **Documentation**

**New Documentation**:
- `docs/ZERO-COPY-API-REFERENCE.md`: Complete zero-copy vs copy operations guide
- `docs/v0.9.1-ZERO-COPY-TEST-SUMMARY.md`: Implementation and test summary

**Clarifications**:
- GCS backend: Production-ready, works well in testing
- Performance targets: 5+ GB/s reads, 2.5+ GB/s writes maintained

---

### 🔧 **Implementation Notes**

**Zero-Copy Architecture**:
- `PyBytesView` wraps `Bytes` with PyO3 buffer protocol support
- `.memoryview()` → zero-copy access (read-only)
- `.to_bytes()` → explicit copy for compatibility
- `__len__()` → zero-copy length query

**Type Conversions**:
- Functions returning `PyBytesView`: Direct return
- Functions returning `PyObject`: Use `.into_py_any(py)?`
- Functions returning `Option<PyObject>`: Nested map with `Python::with_gil()`

**Build Quality**:
- Zero warnings in all builds (`cargo build`, `cargo clippy`, PyO3)
- Fixed all Rust test assertions for `Bytes` comparisons
- Proper error handling and validation

---

### 📊 **Performance Impact**

**Memory Savings**:
```python
# v0.9.0 (copies data)
data = s3dlio.get("s3://bucket/1gb.bin")  # bytes
arr = np.frombuffer(data, dtype=np.uint8)
# Memory: 2 GB (bytes + NumPy array)

# v0.9.1 (zero-copy)
view = s3dlio.get("s3://bucket/1gb.bin")  # BytesView
arr = np.frombuffer(view.memoryview(), dtype=np.uint8)
# Memory: 1 GB (shared buffer)
# Savings: 50%
```

**Throughput**:
- Reduced GC pressure: 15-20% faster in batch operations
- Universal `get_many()`: Works across all backends
- Range requests: Fetch only needed bytes

---

### 🔄 **Migration from v0.9.0**

**Python API Changes**:
```python
# Old (v0.9.0)
data = s3dlio.get(uri)  # bytes
arr = np.frombuffer(data, dtype=np.float32)

# New (v0.9.1) - Zero-copy
view = s3dlio.get(uri)  # BytesView
arr = np.frombuffer(view.memoryview(), dtype=np.float32)

# Backward compatibility (copies)
view = s3dlio.get(uri)
data = view.to_bytes()  # Convert to bytes if needed
```

**No Breaking Changes**:
- `BytesView` supports buffer protocol → most code works unchanged
- Explicit `.to_bytes()` available for compatibility
- Dataset API unchanged (still returns bytes)

---

### ✅ **Verified Compatibility**

**Frameworks**:
- ✅ NumPy: `np.frombuffer(view.memoryview())`
- ✅ PyTorch: `torch.from_numpy(np.frombuffer(view.memoryview()))`
- ✅ TensorFlow: `tf.constant(view.memoryview())`
- ✅ JAX: `jnp.array(np.frombuffer(view.memoryview()))`

**Backends**:
- ✅ S3: AWS S3, MinIO, Vast
- ✅ Azure: Azure Blob Storage
- ✅ GCS: Google Cloud Storage
- ✅ File: Local filesystem
- ✅ DirectIO: Direct I/O filesystem

---


---

**For v0.9.0 and earlier releases, see [archive/Changelog_pre_v090.md](archive/Changelog_pre_v090.md)**
