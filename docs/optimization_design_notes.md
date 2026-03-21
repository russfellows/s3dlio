# s3dlio Optimization Design Notes

**Author:** Russ Fellows / GitHub Copilot  
**Date:** March 2026  
**Version context:** v0.9.84 (implemented and shipped)

---

## Overview

This document captures the reasoning behind the performance optimizations introduced based on
profiling the DLIO training workload (168 × 147 MB NPZ files). It explains what was changed, why,
and importantly what must **not** be over-tuned for the small test environment.

---

## 1. Backward Compatibility — Nothing Was Broken

### Public API is unchanged

All public function signatures are identical to v0.9.30:

| Function | Signature change? |
|---|---|
| `get_object_uri_optimized_async(uri)` | **No** — same name, same signature |
| `get_objects_parallel(uris, max_in_flight)` | **No** — same signature |
| `get_objects_parallel_with_progress(...)` | **No** — same signature |
| `get_object_concurrent_range_async(...)` | **No** — still public, still callable |

### What was added (internal only)

`get_object_with_known_size_async()` is a **private** `async fn` (no `pub`) that is called only
by `get_object_uri_optimized_async()` when the object size is already known from the size cache.
This eliminates HEAD request #2 — the one that `get_object_concurrent_range_async()` would have
otherwise issued. External callers are unaffected.

---

## 2. Changes Made and Why

### 2.1 `S3DLIO_ENABLE_RANGE_OPTIMIZATION` — Bug Fix (All Paths Now Consistent)

**Problem:** The env var was documented as a global switch but only applied to
`S3ObjectStore::get()` in `object_store.rs`. The `get_many()` Python path routes through
`get_objects_parallel()` → `get_object_uri_optimized_async()` in `s3_utils.rs`, which did not
check the env var. Setting `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` was a confirmed no-op on the
`get_many()` path.

**Fix:** `get_object_uri_optimized_async()` now checks `get_range_opt_enabled()` as its first
action. If range opt is disabled, it returns immediately via single GET — no HEAD, no range split.
Both code paths now behave consistently.

**Key secondary effect:** When `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0`, the HEAD request that
the old code always issued (to learn object size for the threshold decision) is also eliminated.
This means the env var is now meaningful for performance, not just a no-op.

### 2.2 OnceLock for Env Var Caching

**Problem:** `get_object_uri_optimized_async()` previously called `std::env::var()` on every
invocation — one syscall per object, hot path.

**Fix:** Three `OnceLock<T>` statics cache the values on first call:

```rust
static RANGE_OPT_ENABLED: OnceLock<bool> = OnceLock::new();
static RANGE_THRESHOLD_BYTES: OnceLock<u64> = OnceLock::new();
static GLOBAL_SIZE_CACHE: OnceLock<Arc<ObjectSizeCache>> = OnceLock::new();
```

**MPI note:** Each MPI rank is a separate OS process, each with its own statics. So
`S3DLIO_RT_THREADS=8` (or any other env var) is read per-rank on first call, as expected.
Changing env vars after process start will not be picked up — this is the expected and
documented behaviour for OnceLock-based config.

### 2.3 ObjectSizeCache Integration — Eliminating Double HEAD

**Problem:** For a batch of N objects, the original code path was:
1. `get_objects_parallel`: for each URI, call `get_object_uri_optimized_async(uri)`
2. `get_object_uri_optimized_async`: issue HEAD to get object size → decide threshold
3. `get_object_concurrent_range_async`: issue **another** HEAD to get total size for range calc

That's 2 × N HEAD requests per batch (Finding 1 in the analysis doc).

**Fix:**
- **Pre-stat phase** in `get_objects_parallel()`: before starting any GETs, issue N concurrent
  HEADs (rate-limited to `max_in_flight`), store results in the process-global `ObjectSizeCache`.
- **`get_object_uri_optimized_async()`**: checks cache first; on a hit, skips its HEAD entirely
  and passes the known size to `get_object_with_known_size_async()`, which also skips HEAD.
- **Result:** epoch 1, batch 1 = N HEADs (pre-stat) + N GETs. Epoch 1, batch 2+ = 0 HEADs.
  Epoch 2+ = 0 HEADs (cache warm). For training workloads where the same files repeat, this is
  optimal.

**For workloads where files do NOT repeat** (e.g., first-time data processing): pre-stat still
issues the same number of HEADs as before, now just concurrent and upfront rather than serialised
inside each GET. No regression.

### 2.4 ObjectSizeCache TTL — 1 Hour Default

**Problem:** The initial implementation used a 5-minute TTL. For large-scale training on 10TB+
datasets where a single epoch takes 10–30 minutes, cache entries expire mid-training, causing
redundant HEADs on every subsequent epoch — defeating the purpose.

**Fix:** Default TTL is **3600 seconds (1 hour)**. Override with:
```
S3DLIO_SIZE_CACHE_TTL_SECS=7200   # 2 hours for very slow storage
S3DLIO_SIZE_CACHE_TTL_SECS=60     # Short TTL if objects are frequently replaced
```

Object files in training datasets are immutable for the lifetime of a run. 1 hour is conservative
and appropriate for all realistic training job durations.

### 2.5 Mutex Elimination in `concurrent_range_get_impl()`

**Problem:** All concurrent range chunk writers shared a single `Arc<Mutex<BytesMut>>`. With up
to 37 concurrent chunk requests per 147 MB file, each chunk writer blocked every other on
completion to copy its data into the shared buffer. Pure serialisation at the merge point.

**Fix:** Each chunk future now returns `(buffer_offset, Bytes)` independently — no shared state.
All chunks collected into a `Vec`, sorted by offset (`sort_unstable_by_key`, O(N log N) where N ≤
~37), then assembled in one sequential pass into a final `BytesMut`. Lock-free except for the
sort, which is on a tiny N.

### 2.6 O(N²) → O(N log N) Sort in `get_objects_parallel()`

**Problem:** After concurrent GETs completed, results were reordered to match the input slice via
`out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap())`. This is O(N²) — for each
of the N results, a linear scan of the N-element input slice. For large batches this is
measurable.

**Fix:** Before dispatch, build a `HashMap<String, usize>` mapping URI → input position (O(N)).
Sort is then `sort_by_key(|(u, _)| uri_positions.get(u).copied().unwrap_or(0))` — O(N log N),
no inner scan.

---

## 3. What Was Deliberately NOT Changed

### 3.1 Range Threshold Default (32 MB)

The test environment had a 1 Gbps NIC. For 147 MB files on a 1 Gbps link:
- A single TCP flow saturates ~120 MB/s
- Range splitting creates 37 sub-requests that open 37 TCP flows — massive overhead at 1 Gbps  
- Solution for that environment: `S3DLIO_RANGE_THRESHOLD_MB=1000` (threshold above file size)

**For production environments (100+ Gbps, RDMA, high-performance object storage):**
- A single TCP flow caps at 4–8 GB/s due to OS TCP window and buffer limits
- Reaching 10–100 GB/s requires multiple parallel flows — exactly what range splitting provides
- The 32 MB default is **correct** for production: any file > 32 MB will use concurrent ranges
- Users at 1 Gbps should set `S3DLIO_RANGE_THRESHOLD_MB=<larger_than_file_size_MB>`

### 3.2 Tokio Thread Count Default

`S3DLIO_RT_THREADS` controls the Tokio async runtime thread count per process. The default (32)
was reduced to 8 in the test environment because MPI NP=1 on a ~16-core test machine with 32
Tokio threads caused scheduling overhead.

**For production:**
- A 128-core machine running NP=4 MPI ranks: 128/4 = 32 cores per rank → 32 Tokio threads = good
- A 64-core machine running NP=8 MPI ranks: 64/8 = 8 cores per rank → 8 Tokio threads = good
- Rule of thumb: `S3DLIO_RT_THREADS = total_cores / NP`

The default of 32 is kept for single-process (NP=1) workloads on production hardware. MPI users
should set `S3DLIO_RT_THREADS` explicitly in their launch scripts.

### 3.3 `get_object_concurrent_range_async()` Still Exists and Is Public

This function is still public and callable. Internal code that knew the object size in advance
can still call it directly. `object_store.rs` uses it in two places and those calls are unchanged.

---

## 4. MPI Considerations

s3dlio is frequently called from within MPI-parallel DLIO. Key facts:

- **Each MPI rank = separate OS process** with its own Tokio runtime, OnceLock statics,
  and ObjectSizeCache. No cross-rank sharing, no distributed coordination needed.
- **Concurrency per rank = `max_in_flight` × MPI NP total concurrent connections.** With
  NP=8 and max_in_flight=16: 128 concurrent S3 connections across ranks. Most object stores
  handle this fine; very high NP with high max_in_flight can cause connection exhaustion on
  poorly-configured MinIO. Tune `max_in_flight` downward if needed.
- **Pre-stat phase in `get_objects_parallel()`**: each rank pre-stats its own slice of the
  batch (DLIO distributes files across ranks). So if NP=8 and 168 files/epoch, each rank
  pre-stats ~21 files = 21 HEADs total. Negligible.
- **OnceLock env vars**: read on first call in each rank's process. Setting per-rank env vars
  (e.g., different `S3DLIO_RT_THREADS` per rank) is supported — MPI launchers can set them.

---

## 5. Prefetch Module Notes

### `src/prefetch.rs` (Prefetcher handle)

The simple top-level `Prefetcher` struct uses `get_object(&bucket, &key)` — the basic path without
cache or range-split logic. This is intentional for streaming use cases where the caller controls
the iteration. For the DLIO training workload, `get_many()`/`get_objects_parallel()` is used
instead (batch per step), not the Prefetcher.

**Future enhancement:** Update `Prefetcher` to use `get_object_uri_optimized_async()` so range
splitting and size cache apply to the streaming path as well. Low priority since DLIO doesn't use
this interface.

### `src/data_loader/prefetch.rs` (DataLoader-integrated prefetch)

The data_loader prefetcher uses the `S3BytesDataset` with `ReaderMode::Sequential` or
`ReaderMode::Range`. These go through `get_object_uri_async()` or `get_object_range_uri_async()`
respectively — not the optimized path. The data_loader is a separate abstraction designed for
Rust-native use and the PyTorch DataLoader protocol; DLIO uses `get_many()` directly.

---

## 6. Environment Variable Summary

| Variable | Default | Effect |
|---|---|---|
| `S3DLIO_ENABLE_RANGE_OPTIMIZATION` | `1` (enabled) | `0/false/no/off` → skip HEAD + single GET on ALL paths |
| `S3DLIO_RANGE_THRESHOLD_MB` | `32` | Files above this MB use concurrent range GETs |
| `S3DLIO_RT_THREADS` | `32` | Tokio async runtime threads per process |
| `S3DLIO_SIZE_CACHE_TTL_SECS` | `3600` | Object size cache TTL in seconds |

**Production tuning cheat sheet:**

```bash
# 100+ Gbps, large files (e.g., 512MB checkpoints), fast object storage:
# Use defaults — range splitting helps

# 1–10 Gbps, 100–200 MB files, limited parallelism:
export S3DLIO_RANGE_THRESHOLD_MB=1000   # Disable splitting for this file size

# MPI, e.g. NP=8 on a 64-core machine:
export S3DLIO_RT_THREADS=8              # 1 Tokio thread per core on this rank

# Very short-lived training jobs (< 5 min):
export S3DLIO_SIZE_CACHE_TTL_SECS=3600  # Default is fine; set lower to free RAM early
```

---

## 7. What Remains for Future Work

| Item | Description | Where |
|---|---|---|
| HEAD #1 elimination | When file lists include sizes (e.g. from `list_objects()` returning stats), pass sizes into `get_objects_parallel()` to skip all HEADs | `s3_utils.rs` |
| Adaptive range threshold | Auto-detect link speed to set threshold without env var | `s3_utils.rs` |
| Prefetcher cache integration | Update `src/prefetch.rs` to use `get_object_uri_optimized_async()` | `prefetch.rs` |
| Data loader pipeline | Consider using DataLoader + S3BytesDataset for DLIO to get prefetch windowing | TBD |
| Cache memory bound | Add max-entry count to ObjectSizeCache to prevent unbounded growth with unique-file workloads | `object_size_cache.rs` |
