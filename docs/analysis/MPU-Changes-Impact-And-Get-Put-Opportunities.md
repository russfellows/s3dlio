# Analysis: MPU Changes Impact & Get/Put Channel Opportunities

**Date**: April 22, 2026  
**Branch**: `feature/connection-pool-autoscale`  
**Commit**: `19cd2c4`  
**Scope**: Impact analysis of `src/multipart.rs` v0.9.92 changes on Rust and Python users, assessment of sai3-bench compatibility, and investigation of whether the coordinator/channel technique could improve single-object Get and Put performance.

---

## 1. Summary of Changes (v0.9.92)

Two changes were made to `src/multipart.rs`:

### Change 1 — Auto-scale `max_in_flight`
`MultipartUploadConfig.max_in_flight` changed from `usize` to `Option<usize>`.
- `None` (new default) → auto-computed via `auto_max_in_flight(part_size) = max(32, ceil(512 MiB / part_size))`
- `Some(n)` → explicit override, identical to prior behaviour

### Change 2 — Coordinator task + bounded mpsc channel
The hot write path was re-architected:

| Before | After |
|---|---|
| `write_blocking` → `run_on_global_rt(semaphore.acquire().await)` per part | `write_blocking` → `blocking_send` into bounded channel |
| N `run_on_global_rt` calls for N parts | 1 `run_on_global_rt` call at `finish_blocking` |
| Python thread parks in Tokio bridge per part | Python thread parks only when channel is full |
| Coordinator logic in-line with Python thread | Coordinator task runs entirely on Tokio runtime |

---

## 2. Impact on Python Library Users

### 2.1 API compatibility — Python

All Python-facing APIs are unchanged. The `MultipartUploadWriter` PyO3 class and `max_in_flight` keyword argument work identically:

```python
# max_in_flight=None (new default) → auto-computed
w = s3dlio.MultipartUploadWriter("s3://bucket/key")

# max_in_flight=N → explicit, same as before  
w = s3dlio.MultipartUploadWriter("s3://bucket/key", max_in_flight=16)
```

The Python binding was updated in `python_advanced_api.rs`:
```rust
// Before
cfg.max_in_flight = mif;

// After (correct)
cfg.max_in_flight = Some(mif);  // None passed through correctly when Python sends None
```

### 2.2 Behaviour changes — Python

**Memory usage increase (minor)**: The old default was `max_in_flight=16` (hardcoded). The new auto formula gives 32 at 32 MiB parts and 64 at 8 MiB parts. Maximum buffered memory increases from `16 × part_size` to `32–64 × part_size`.

- 8 MiB parts: 128 MiB → 512 MiB (worst case, rarely exercised)
- 32 MiB parts: 512 MiB → 1 GiB (worst case)

Python callers who relied on low memory usage with the old default can restore the old behaviour with `max_in_flight=16`.

**Throughput improvement**: 33% single-writer, 10% 4-writer at 512 MiB. See benchmark results in `docs/enhancement/Async-Runtime-Enhancement.md`.

**Backpressure contract preserved**: `blocking_send` parks the Python thread only when all `max_in_flight` channel slots are occupied. Memory stays bounded at `max_in_flight × part_size` bytes. Issue #134 contract holds.

---

## 3. Impact on Rust Library Users

### 3.1 BREAKING API CHANGE — struct literal construction

`MultipartUploadConfig.max_in_flight: usize` → `max_in_flight: Option<usize>` is a **source-incompatible change** for any Rust code that directly sets this field in a struct literal:

```rust
// This BREAKS with v0.9.92:
let cfg = MultipartUploadConfig {
    max_in_flight: 16,  // ← type error: expected Option<usize>, found usize
    ..Default::default()
};

// Fix:
let cfg = MultipartUploadConfig {
    max_in_flight: Some(16),
    ..Default::default()
};

// Or, to use auto (new behaviour):
let cfg = MultipartUploadConfig::default();  // max_in_flight = None = auto
```

**Scope of breakage**: Searched all known downstream crates (sai3-bench, dl-driver). **None construct `MultipartUploadConfig` directly** — they use `..Default::default()` or go through `ObjectStore::put_multipart`. No current downstream breakage found. However, external users of the library are affected.

**Recommendation**: This change warrants a minor semver bump (0.9.x → 0.10.0 or 0.9.92 → 0.10.0) before public release, as it breaks the struct field type. At minimum, it should be called out in the changelog as a breaking change.

### 3.2 LATENT BUG — `put_object_multipart_uri_async` calls blocking APIs from async context

`s3_utils.rs` contains this async function (unchanged from before our commit, but now more dangerous):

```rust
pub async fn put_object_multipart_uri_async(uri: &str, data: Bytes, part_size: Option<usize>)
    -> Result<()>
{
    let mut sink = MultipartUploadSink::from_uri(uri, cfg)?;  // ← calls run_on_global_rt (blocks worker thread)
    sink.write_blocking(&data)?;   // ← calls blocking_send (PANICS if channel full in async context!)
    sink.finish_blocking()?;       // ← calls run_on_global_rt (blocks worker thread)
    Ok(())
}
```

**Where it's called**: `S3ObjectStore::put_multipart()` → `s3_put_object_multipart_uri_async()`. This is an async method on the `ObjectStore` trait, so it is always called from within a Tokio runtime (`.await`'d by the caller).

**The panic condition**:  
Tokio's `blocking_send` panics with `"called blocking_send when the Tokio runtime entered the thread"` if called from within a Tokio runtime AND the channel is full (i.e., actual blocking is needed).

When the channel fills:
- `max_in_flight` auto = 32 at 16 MiB parts → channel capacity = 32 parts
- Channel fills when object has > 32 parts, i.e., object > 32 × 16 MiB = **512 MiB**
- For a 1 GiB object at 16 MiB parts: 64 parts → channel fills on part 33 → **panic**

**Why this wasn't a panic before our changes**: The old implementation called `run_on_global_rt` per part. `run_on_global_rt` from within a Tokio context uses `std::sync::mpsc::channel` with `rx.recv()` — this blocks the Tokio worker thread but does NOT panic. The new `blocking_send` panics when it would need to block from within a Tokio context.

**Current risk level**: **Low but real.** `put_multipart` is not called in sai3-bench's hot path. Python callers never reach this code path (Python is always outside the Tokio runtime). The risk is for external Rust callers who call `store.put_multipart()` for objects > 512 MiB.

**Fix** (not implemented in this commit — analysis only):  
`put_object_multipart_uri_async` should use the async API rather than blocking stubs:

```rust
// Instead of:
sink.write_blocking(&data)?;
sink.finish_blocking()?;

// Use async-native equivalents that call part_tx.send().await:
sink.write_async(&data).await?;
sink.finish_async().await?;
```

The `MultipartUploadSink` would need truly async `write_async()` and `finish_async()` methods that use `part_tx.send().await` and `coordinator.await` directly, without going through `run_on_global_rt`. These are separate from the blocking variants which remain the primary path for Python callers.

### 3.3 Worker thread starvation in async contexts (pre-existing)

`MultipartUploadSink::new()` calls `run_on_global_rt()` synchronously. When called from within an async task (e.g., if sai3-bench calls `store.put_multipart()` concurrently for N objects), each construction call blocks a Tokio worker thread for the duration of `CreateMultipartUpload` (a network round-trip). With N concurrent callers, this occupies N worker threads. This is a pre-existing issue not introduced by our changes, but worth noting.

---

## 4. Impact on sai3-bench

### 4.1 Direct use of `MultipartUploadSink` — None

A complete search of `sai3-bench/src/` found **zero direct uses** of `MultipartUploadSink` or `MultipartUploadConfig`. sai3-bench never constructs the sink directly.

### 4.2 Use via `ObjectStore::put_multipart` — Not in hot path

sai3-bench implements `ObjectStore::put_multipart` on `ArcMultiEndpointWrapper` (delegates to `MultiEndpointStore`), but the actual benchmark loops in `agent.rs` and `replay.rs` use `store.put(uri, data).await` — regular single PUT — for all write operations. No call site to `put_multipart` was found in the benchmark execution paths.

**Conclusion**: sai3-bench's hot path is unaffected by any of the `multipart.rs` changes.

### 4.3 Use via `ObjectStore::get` / `ObjectStore::put` — Fully async, unchanged

sai3-bench's benchmark loops:

```rust
// agent.rs — PUT workload
store.put(&full_uri, d).await?;

// agent.rs — GET workload  
let bytes = store.get(&full_uri).await?;

// workload.rs — multi-backend PUT
workload::put_object_multi_backend(uri, data).await?;

// workload.rs — multi-backend GET
workload::get_object_multi_backend(uri).await?;
```

All of these go through the `ObjectStore` trait's async `get()` and `put()` methods, which call:
- `s3_put_object_uri_async()` / `s3_get_object_uri_async()` — fully async, no blocking calls in hot path
- No `MultipartUploadSink` involvement

**These code paths are completely unaffected by the multipart.rs changes.**

### 4.4 Could the changes help sai3-bench?

**Today**: No direct benefit, since multipart is not used.

**If sai3-bench adds large-object write support**: The auto-scale formula would automatically give optimal parallelism without configuration. A sai3-bench workload writing 512 MiB–5 GiB objects would benefit from the same 1.03–1.10× improvement vs s3torchconnector observed in benchmarks.

**Indirect benefit**: The coordinator architecture cleanly separates the sai3-bench thread (or Tokio task) from the Tokio I/O work. Once the latent `put_object_multipart_uri_async` async-safety issue is fixed (§3.2), sai3-bench could safely use `put_multipart` from its async workload tasks for large-object writes.

---

## 5. Can the Coordinator/Channel Technique Help Get and Put?

### 5.1 Current architecture of single-object Get and Put

```
Python get(uri)
  → submit_io(store.get(uri).await)        ← 1 run_on_global_rt call
      → S3ObjectStore::get → AWS SDK GetObject
      ← Bytes returned
  ← PyBytesView

Python put(uri, data)  
  → submit_io(store.put(uri, data).await)  ← 1 run_on_global_rt call
      → S3ObjectStore::put → AWS SDK PutObject
      ← ()
```

**The MPU architecture helped because** the old code called `run_on_global_rt` N times (once per part). For a 512 MiB / 32 MiB = 16 part upload: 16 × channel round-trips, each parking the Python thread briefly even when slots were available.

For single GET/PUT: there is only **ONE** `run_on_global_rt` call wrapping the entire operation. The channel overhead is a one-time cost (50–150 µs) and is not repeated per chunk. **There is no multi-call hot-path to eliminate.**

### 5.2 Where the coordinator pattern IS applicable for Get/Put

The coordinator/channel technique helps when there is a **repeated interaction between the Python thread and the Tokio runtime for a single logical operation**. Three scenarios qualify:

#### Scenario A — Streaming object downloader (high-value)

**Problem**: `get_objects_parallel` blocks Python until ALL N objects are downloaded. For a training loop consuming `batch_size=256` objects, Python waits the duration of the slowest download before processing any.

**Coordinator solution**:
```
Python loop calls next_object()
  → blocking_recv() on results channel  ← parks Python only when no result ready
                                          returns immediately when one is ready

[Background coordinator task — Tokio runtime]
  loop:
    uri = uri_rx.recv().await
    permit = sem.acquire().await
    spawn(get object, send result to results_tx)
```

This is **exactly the MPU pattern inverted for reads**. Benefits:
- Python processes object N while objects N+1..N+32 are downloading in parallel
- Time-to-first-object: near zero (first object ready while rest still download)
- Throughput pipelining: Python processing and network I/O overlap
- Backpressure: `results_tx` is bounded → coordinator stops issuing GETs if Python can't keep up

**Estimated benefit**: For a 256-object batch at 100 MiB/s per object, the first object would be available in ~80 ms instead of ~204 ms (time for slowest object in batch). For AI/ML training loops, this directly reduces idle GPU time.

#### Scenario B — Persistent PUT coordinator for high-IOPS small object writes

**Problem**: Each Python `put(uri, data)` call incurs one `run_on_global_rt` round-trip (50–150 µs channel latency + thread scheduling). For 10,000 small (64 KiB) objects/sec, this adds 500 ms–1.5 s of pure bridging overhead per second — the bridge itself becomes the bottleneck.

**Coordinator solution**:
```
Python submit_put(uri, data)
  → blocking_send(PutRequest { uri, data })  ← parks Python only when all slots full

[Background coordinator — Tokio runtime]
  loop:
    req = rx.recv().await
    permit = sem.acquire().await
    spawn(put_object(req.uri, req.data))
    
Python wait_all() → send Drain msg → coordinator joins all tasks
```

Benefits:
- Eliminates per-put channel round-trip; only one blocking call across all puts
- Fire-and-forget semantics with controlled parallelism
- Better connection reuse: coordinator can reuse the same HTTP/2 connections

**Estimated benefit**: For high-IOPS small-object workloads, 10–30% reduction in wall-clock time. For large objects (>1 MiB), negligible because network time dominates.

#### Scenario C — Prefetch coordinator (most impactful for AI/ML reads)

**Problem**: Sequential epoch reads with no lookahead. Python finishes processing batch N, then starts downloading batch N+1. GPU sits idle during each download phase.

**Coordinator solution**: A persistent background prefetch coordinator receives a **future work queue** of URIs (e.g., the next 2–3 batches), downloads them proactively, and stores results in a bounded in-memory cache. Python's `get()` call returns from cache with near-zero latency.

This is the `prefetch.rs` module's purpose, but a channel-based coordinator would:
1. Accept URI queues pushed by Python as it constructs each epoch
2. Download with configurable lookahead (e.g., 2 batches ahead)
3. Bound memory use via channel capacity (discard old results if Python falls behind)

**Estimated benefit**: Could entirely eliminate download latency from the Python-visible critical path for sequential workloads. The ceiling is 100% overlap of compute and I/O.

### 5.3 Where the coordinator pattern does NOT help

| Scenario | Why it doesn't help |
|---|---|
| Single `get(uri)` | Already 1 `run_on_global_rt` call; no hot-path repetition |
| Single `put(uri, data)` | Same — 1 call, dominated by network time |
| `get_objects_parallel` (existing) | Already fully async inside 1 `run_on_global_rt`; coordinator is already there in spirit (tokio::spawn + Semaphore) |
| PUT parallelism (`put_objects_parallel`) | Same — N concurrent PUTs already run inside 1 `run_on_global_rt` |

### 5.4 What actually limits Get performance today

Before implementing any coordinator, the real bottlenecks for Get should be understood:

1. **HTTP/2 stream concurrency**: The AWS Rust SDK may be reusing connections but stream count per connection is bounded (typically 100–128 concurrent streams per HTTP/2 connection). At high parallelism, new connections must be established.

2. **Connection pool size**: Default Hyper/reqwest connection pool may limit concurrent connections to the same host. This is where s3torchconnector (which uses CRT with aggressive connection management) gains its edge.

3. **TLS session resumption**: Each new connection pays a TLS handshake cost. Session tickets are typically cached, but with many parallel workers, cold-start connections dominate.

4. **`run_on_global_rt` overhead**: Measured at ~50–150 µs per call. For objects > 1 MiB, this is noise. For objects < 64 KiB at high IOPS, this CAN dominate.

**Recommendation**: Profile a real sai3-bench GET workload with `S3DLIO_RT_THREADS` tuned to `num_cpus × 2` and check connection-pool metrics before adding coordinator complexity.

---

## 6. Summary and Recommendations

### 6.1 Impact summary table

| Caller | Impact | Risk |
|---|---|---|
| Python library users | +33% PUT throughput, default memory increased | Low — backward compat, can revert with `max_in_flight=N` |
| Rust library users (struct literal) | **Breaking**: `usize → Option<usize>` | Medium — compile error, easy to fix |
| Rust `put_multipart` async callers | Latent panic for objects > 512 MiB | Medium — not triggered today, but real |
| sai3-bench hot path | **Zero impact** — uses `put/get` not multipart | None |
| sai3-bench if multipart is added | Positive — auto-scale + coordinator benefits | None |

### 6.2 Action items

| Priority | Action | Effort |
|---|---|---|
| **High** | Fix `put_object_multipart_uri_async` — add truly async `write_async()`/`finish_async()` methods on `MultipartUploadSink` that use `part_tx.send().await` | Small |
| **High** | Version bump to 0.10.0 or document breaking `Option<usize>` change in changelog | Trivial |
| **Medium** | Implement streaming object downloader (Scenario A) | Medium |
| **Low** | Persistent PUT coordinator for high-IOPS small objects (Scenario B) | Medium |
| **Low** | Profile connection pool as root cause of remaining 10% Get gap before adding coordinator | Trivial |

### 6.3 The correct fix for `put_object_multipart_uri_async`

The simplest safe fix is to avoid calling any blocking method from within this async function:

```rust
// src/s3_utils.rs — put_object_multipart_uri_async
pub async fn put_object_multipart_uri_async(
    uri: &str,
    data: Bytes,
    part_size: Option<usize>,
) -> Result<()> {
    use crate::multipart::{MultipartUploadConfig, MultipartUploadSink};
    let cfg = MultipartUploadConfig {
        part_size: part_size.unwrap_or(16 * 1024 * 1024),
        ..Default::default()
    };

    // Use new_async() directly — avoids run_on_global_rt() blocking a worker thread
    let mut sink = MultipartUploadSink::new_async(
        &parse_s3_uri(uri)?.0,
        &parse_s3_uri(uri)?.1,
        cfg
    ).await?;

    // Use async write — needs part_tx.send().await, NOT blocking_send
    sink.write_async_safe(&data).await?;       // new method needed
    sink.finish_async_safe().await?;           // new method needed

    Ok(())
}
```

The `write_async_safe` and `finish_async_safe` methods would use `part_tx.send().await` and `coordinator.await` directly — no `run_on_global_rt` needed since we're already in an async context.

---

## Appendix: Call chain reference

```
sai3-bench (async Tokio runtime)
  │
  ├── store.get(uri).await
  │     └── S3ObjectStore::get → s3_get_object_uri_optimized_async → AWS SDK GetObject
  │         ✅ Fully async, no blocking calls
  │
  ├── store.put(uri, data).await  
  │     └── S3ObjectStore::put → s3_put_object_uri_async → AWS SDK PutObject
  │         ✅ Fully async, no blocking calls
  │
  └── store.put_multipart(uri, data, part_size).await  [NOT USED in hot path]
        └── S3ObjectStore::put_multipart → s3_put_object_multipart_uri_async
              ├── MultipartUploadSink::from_uri → run_on_global_rt  ← blocks worker thread
              ├── sink.write_blocking → blocking_send               ← PANICS if channel full
              └── sink.finish_blocking → run_on_global_rt           ← blocks worker thread

Python (plain OS thread, outside Tokio runtime)
  │
  ├── s3dlio.get(uri)
  │     └── submit_io(store.get(uri).await)  ← 1 run_on_global_rt call
  │         ✅ Correct use of blocking API from non-async context
  │
  └── MultipartUploadWriter.write(data) / .close()
        └── MultipartUploadSink::write_blocking / finish_blocking
              ├── blocking_send  ← safe from Python thread (not in Tokio runtime)
              └── run_on_global_rt  ← safe from Python thread
                  ✅ Correct use of blocking APIs from non-async context
```
