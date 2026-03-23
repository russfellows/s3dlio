# Object Storage Performance Analysis — DLIO Training Workload

**Date:** March 20, 2026  
**Author:** Analysis via GitHub Copilot  
**Status:** HISTORICAL — Findings 1, 2, 4, 5, 6 resolved in s3dlio v0.9.84; Finding 3 mitigated via `S3DLIO_RT_THREADS=8`. See `dlio_mpi_object_results.md` for current benchmark results.

---

## 1. Executive Summary

At NP=1, s3dlio **baseline** (v0.9.82 defaults) achieved **332 MB/s** while minio-py achieved **459 MB/s** — a 1.38× deficit. After applying all Tier 1+2 fixes (env vars + zero-copy reader), throughput rose to **408 MB/s** (workaround) and then **413 ± 2 MB/s** (v0.9.84 bug-fix wheel with `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0`). At NP=4 both s3dlio and minio converge to **~1087–1097 MB/s** — within 1% of each other — confirming that the gap at NP=1 was caused by the six root causes below, not a fundamental library capability difference.

After examining the code paths from Python reader through to the Rust async engine, six
root causes were identified:

| # | Finding | Severity | Fastest fix | Where | **v0.9.84 Status** |
|---|---------|----------|-------------|-------|---|
| 1 | Double HEAD request per object file | **Critical** | `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` (v0.9.84 — now works on `get_many` path) | s3dlio Rust | **✅ RESOLVED** |
| 2 | Range splitting (37 requests per file) | **Major** | `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` (correct knob in v0.9.84) | s3dlio Rust | **✅ RESOLVED** |
| 3 | Tokio runtime over-provisioning | **Major** | `S3DLIO_RT_THREADS=8` | Env var | **⚪ MITIGATED** (env var; architectural fix pending) |
| 4 | `bytes(data)` Python copy (147 MB/file) | **Major** | Zero-copy `_BytesViewIO` wrapper | mlp-storage reader | **✅ RESOLVED** |
| 5 | Mutex contention in range assembly | Moderate | Mutex-free collect-then-assemble | s3dlio Rust | **✅ RESOLVED v0.9.82** |
| 6 | O(N²) sort in `get_objects_parallel` | Minor | HashMap pre-index O(N log N) sort | s3dlio Rust | **✅ RESOLVED v0.9.82** |

> **⚠️ `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` is confirmed as a no-op for `get_many()`.**
> Do not use it. The correct env var for the `get_many()` path is `S3DLIO_RANGE_THRESHOLD_MB`.

**Priority order — highest impact, lowest effort first:**
1. **Env vars** (`S3DLIO_RANGE_THRESHOLD_MB=1000` + `S3DLIO_RT_THREADS=8`) — zero effort, do now
2. **Python reader fix** (`bytes(data)` → `_BytesViewIO`) — 3-line mlp-storage change, no s3dlio rebuild
3. **All s3dlio Rust changes in one batch** — each release cycle costs ~1.5 hours; never make a partial Rust change


Results updated. Now for your architectural question:

Yes, you are correct — it's a bug. S3DLIO_ENABLE_RANGE_OPTIMIZATION is documented and named as a global flag controlling range optimization behavior, but it only applies to one of two code paths:

Code path	Checks S3DLIO_ENABLE_RANGE_OPTIMIZATION?	Checks S3DLIO_RANGE_THRESHOLD_MB?
S3ObjectStore::get() → object_store.rs:1008	✅ YES	No (delegates to get_object_uri_optimized_async)
get_object_uri_optimized_async() → s3_utils.rs:1247	❌ NO	✅ YES
The get_many() path (which DLIO exclusively uses) goes through get_object_uri_optimized_async(), which silently ignores S3DLIO_ENABLE_RANGE_OPTIMIZATION. A user setting =0 has every reasonable expectation that range splitting stops on ALL paths.

The correct fix is to add the S3DLIO_ENABLE_RANGE_OPTIMIZATION check at the top of get_object_uri_optimized_async() in s3_utils.rs, before the HEAD request:

Edited
dlio_mpi_object_results.md+30-10
This has a secondary benefit: when =0, it also eliminates the HEAD request that get_object_uri_optimized_async() currently always issues to determine object size (the HEAD that contributes to Finding 1's double-HEAD cost). So S3DLIO_ENABLE_RANGE_OPTIMIZATION=0 would become a meaningful optimization flag rather than a no-op on the hot path.

Long-term, the env var reads should be cached with OnceLock (called once per process) rather than re-reading from environment on every object fetch — this is flagged in my memory notes as a pattern to use for process-wide config in s3dlio.

I've only documented this analysis — let me know if you want me to implement the fix in s3_utils.rs.

---

## 2. Benchmark Context

| Parameter | Value |
|-----------|-------|
| Dataset | 168 × 146.6 MB NPZ files = 24.63 GB |
| Network ceiling | 1.2 GB/s (anything above = page cache) |
| DLIO DataLoader workers | 4 per rank (multiprocessing_context: spawn) |
| Object format | NPZ (numpy archive, compressed, `["x"]` key) |
| Training epochs | 5, batch_size=7 |

**Observed throughput (wall-clock, NP=1):**

| Library | MB/s | GB/s | Relative |
|---------|------|------|----------|
| s3dlio | 332 ± 0.7 | 0.332 | 1.00× (baseline) |
| s3torchconnector | 303.0 ± 1.1 | 0.303 | 0.91× (8% slower than s3dlio) |
| minio-py | 459 ± 1 | 0.459 | **1.38× faster than s3dlio** |

> **Note:** See [dlio_mpi_object_results.md](dlio_mpi_object_results.md) for complete per-epoch
> data for all three libraries across NP=1/2/4/8. s3torchconnector results use the real
> `S3IterableDataset.from_objects()` API with `S3ReaderConstructor.sequential()` — single
> streaming GET per file, no range splitting, no HEAD requests.

---

## 3. Code Path Traced

```
DLIO DataLoader worker (spawned process)
  └─ NPZReaderS3Iterable.next()
       └─ _prefetch()
            │
            ├─ [s3dlio / s3torchconnector] → _prefetch_s3dlio()
            │    └─ s3dlio.get_many(uris, max_in_flight=64)     ← Rust
            │         └─ [S3 path] get_objects_parallel(uris, max_in_flight)
            │               └─ for each URI (up to 64 concurrent):
            │                    └─ get_object_uri_optimized_async(uri)
            │                          ├─ HEAD #1: head_object() ← size check
            │                          └─ [size >= 32MB: 147MB > 32MB]
            │                               └─ get_object_concurrent_range_async()
            │                                     ├─ HEAD #2: head_object() ← AGAIN!
            │                                     └─ split into N×4MB range GETs (37 for 147MB)
            │
            └─ [minio] → _prefetch_minio()
                 └─ ThreadPoolExecutor (16 threads)
                      └─ for each file:
                           └─ Minio.get_object()  ← 1 streaming GET
```

**Requests per file, s3dlio vs minio:**

| Operation | s3dlio | minio |
|-----------|--------|-------|
| HEAD (size probe) | **2** | 0 |
| GET (full or ranged) | **37** (4 MB chunks × 37) | **1** streaming |
| **Total requests per 147 MB file** | **39** | **1** |

With 42 files per DataLoader worker (NP=1, 4 workers, 168 files → 42/worker):

| Library | Requests per worker per epoch | Total (× 4 workers) |
|---------|------------------------------|---------------------|
| s3dlio | 42 × 39 = **1,638** | **6,552** |
| minio | 42 × 1 = **42** | **168** |

s3dlio sends **39× more S3 requests** to fetch the same data.

---

## 4. Root Cause Analysis

### Finding 1: Double HEAD Request Per Object — [RESOLVED v0.9.84]

**Location:** `s3dlio/src/s3_utils.rs`  
**Functions:** `get_object_uri_optimized_async()` and `get_object_concurrent_range_async()`

```rust
// In get_object_uri_optimized_async():
let head_resp = client.head_object()           // ← HEAD request #1
    .bucket(&bucket).key(&key).send().await;
let object_size = head_resp.content_length();

if object_size >= range_threshold {  // 147 MB >= 32 MB → TRUE
    get_object_concurrent_range_async(uri, ...).await  // calls...
}

// In get_object_concurrent_range_async():
let head_resp = client.head_object()           // ← HEAD request #2 (redundant!)
    .bucket(&bucket).key(&key).send().await;
let object_size = head_resp.content_length();
```

The first HEAD was added to decide whether to range-split. The second HEAD exists
independently inside `get_object_concurrent_range_async` because that function also
needs the size to compute range boundaries. The size from HEAD #1 is **not passed
down** — it is simply discarded.

**Impact per file:** +1 unnecessary HEAD round-trip (~5–15 ms each at 1 Gbps).  
**At scale:** 168 files × 1 extra HEAD = 168 extra round-trips per epoch.

---

### Finding 2: Range Splitting Threshold Too Low for This 1 Gbps Environment — [RESOLVED v0.9.84]

> **⚠️ Context-dependent finding:** Range splitting is *essential* on high-bandwidth systems
> (100 Gbps / high-performance S3) and *counterproductive* on bandwidth-saturated
> 1 Gbps systems like this MinIO test setup. Both are correct — see analysis below.

**Location:** `s3dlio/src/s3_utils.rs` — `get_optimal_chunk_size()` and
`concurrent_range_get_impl()`

For objects between 16 MB and 256 MB, s3dlio defaults to **4 MB chunks**:

```rust
fn get_optimal_chunk_size(total_bytes: u64) -> usize {
    if total_bytes < 16 * 1024 * 1024 {
        1024 * 1024   // 1 MB chunks
    } else if total_bytes < 256 * 1024 * 1024 {
        4 * 1024 * 1024  // ← 4 MB chunks (our 147 MB files land here)
    } else {
        8 * 1024 * 1024
    }
}
```

For a 147 MB file: ⌈147 / 4⌉ = **37 HTTP range requests** instead of 1.

**Range splitting is network-bandwidth-dependent — both of these are true:**

| Environment | Network | Single-TCP throughput | Range splitting? |
|-------------|---------|----------------------|------------------|
| This MinIO test system | ~1.2 GB/s | ~1.1 GB/s (fills the pipe) | **Harmful** — 37× request overhead, zero throughput gain |
| High-perf S3 (100G network) | 10–100 Gbps | ~4–8 Gbps (OS/NIC limited) | **Essential** — parallel ranges aggregate multiple TCP flows |

**On this 1 Gbps system (current test environment):**
- A single streaming GET for 147 MB fills the 1.2 GB/s link in ≈ 136 ms
- Range splitting adds 37× HTTP round-trips, connection setup, TLS overhead, and buffer management
- All CPU cost, zero throughput benefit
- **Fix: raise `S3DLIO_RANGE_THRESHOLD_MB` above the file size (e.g. 1000 for 147 MB files)**

**On a 100 Gbps high-performance system (user-confirmed with high-performance S3 storage):**
- A single TCP connection is OS-limited to ~4–8 Gbps (socket buffer, TCP window, stack overhead)
- 16 MB range chunks open 10–20 parallel TCP flows, each contributing ~4 Gbps separately
- **Massive improvement from 16 MB range splitting was observed on such a system** — this is expected and correct
- Raising the threshold on a high-bandwidth system would be a significant regression

The changelog (v0.9.50) benchmarked range splitting with 16 objects on a high-bandwidth system,
showing 76% improvement — consistent with the high-bandwidth picture above. The current default
32 MB threshold serves high-bandwidth environments well but is too aggressive for 1 Gbps deployments.

**The root issue is not that range splitting exists — it is that the default threshold has no
network-awareness.** The optimal design is auto-detection (see §6 Recommendation 3c).

**Impact on this system:** 37× HTTP request overhead per file, high CPU utilization, higher memory
pressure from managing 37 concurrent chunk downloads per file.

---

### Finding 3: Tokio Runtime Thread Over-Provisioning — [MITIGATED: S3DLIO_RT_THREADS=8]

**Location:** `s3dlio/src/s3_client.rs` — `get_runtime_threads()`

```rust
fn get_runtime_threads() -> usize {
    std::env::var("S3DLIO_RT_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            let cores = num_cpus::get();
            let default_threads = std::cmp::max(8, cores * 2);
            std::cmp::min(default_threads, 32)  // ← up to 32 Tokio threads!
        })
}
```

On a 16-core machine: `min(max(8, 32), 32)` = **32 Tokio threads**.

With `multiprocessing_context: spawn`, each DataLoader worker is an independent
process. Each process creates its own Tokio runtime. For NP=4 with 4 workers:

```
4 MPI ranks × 4 DataLoader workers = 16 worker processes
16 worker processes × 32 Tokio threads = 512 background Tokio threads (s3dlio only)
```

minio's equivalent:
```
16 worker processes × min(16, n_files) threads via ThreadPoolExecutor ≈ 256 threads
(and these are Python threads, not continuously-running OS threads)
```

The critical difference: Tokio's 32 worker threads per process are **always running**
(blocking on work queue), whereas ThreadPoolExecutor threads are only active during
the fetch window. The constant OS scheduling overhead across 512 active threads explains
the elevated CPU baselines seen even during compute time in DLIO traces.

**Impact:** Elevated CPU utilization at all times, OS scheduler contention, cache
thrashing between threads managing 37-chunk concurrent downloads per file.

---

### Finding 4: Python-Side `bytes(data)` Memory Copy — [RESOLVED: zero-copy fix applied]

**Location:** `mlp-storage/dlio_benchmark/dlio_benchmark/reader/npz_reader_s3_iterable.py`  
**Function:** `_prefetch_s3dlio()`

```python
def _prefetch_s3dlio(self, filenames: list) -> dict:
    import s3dlio
    uris = [self._uri_for_filename(f) for f in filenames]
    results = s3dlio.get_many(uris)
    cache = {}
    for uri, data in results:
        fname = uri_to_fname.get(uri, uri)
        cache[fname] = np.load(io.BytesIO(bytes(data)), allow_pickle=True)["x"]
                                          ^^^^^^^^^^
                                          This copies ALL 147 MB!
    return cache
```

`data` is a `PyBytesView` — a zero-copy view of Rust-owned memory. It exposes the
Python buffer protocol, meaning clients can read it without copying. However,
`bytes(data)` calls `PyBytesView.__bytes__()`:

```rust
fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &self.bytes)  // Full memcopy into Python heap!
}
```

This discards the zero-copy design. Then `io.BytesIO(...)` wraps the new Python bytes
object. Both the Rust `BytesMut` (from range assembly) and the Python `bytes` copy
coexist in memory simultaneously.

**Memory accounting per worker per epoch (42 files, max_in_flight=64):**

| Object | Size | Lifetime |
|--------|------|----------|
| Rust BytesMut (range buffer) | 147 MB × 64 = **9.4 GB** | Freed after `get_many` returns |
| Python `bytes` copy | 147 MB × 42 = **6.2 GB** | Freed after `np.load` completes per file |
| Peak | | **up to 15.6 GB per worker process** |

With 4 workers that's 62 GB peak resident memory — significantly more than minio's
approach, where each streaming response is consumed and freed file-by-file.

**Note:** A zero-copy alternative already exists in the codebase —
`_BytesViewIO` in `s3dlio/python/s3dlio/compat/s3torchconnector.py`. This class wraps
a `BytesView` as a seekable `io.RawIOBase`, allowing `np.load()` to read directly
from Rust memory with `readinto()` via `memoryview` — **no copy**. It is not currently
used in the DLIO reader.

---

### Finding 5: Mutex Contention in Range Buffer Assembly — [RESOLVED v0.9.82]

**Location:** `s3dlio/src/s3_utils.rs` — `concurrent_range_get_impl()`

```rust
// Pre-allocate full object buffer
let result = Arc::new(std::sync::Mutex::new(BytesMut::zeroed(total_bytes)));

// 37 concurrent tasks, each writing their chunk:
let mut result_guard = result.lock()?;  // ← All 37 writers serialize here
let end_pos = buffer_offset + chunk_data.len();
result_guard[buffer_offset..end_pos].copy_from_slice(&chunk_data);
```

With 37 concurrent writers (for a 147 MB file) each acquiring the same mutex to write
non-overlapping segments, there is severe lock contention. Writing disjoint memory
segments through a shared mutex serializes otherwise-parallel work and inflates CPU
cycles. The correct design for disjoint-range writes into a pre-allocated buffer is to
use `unsafe` pointer arithmetic from separate tasks (no mutex needed for disjoint
ranges), or a `Mutex<[u8]>` with per-chunk locking.

Additionally, `BytesMut::zeroed(total_bytes)` pre-zeroes 147 MB before downloads even
start — a `calloc`-equivalent that adds ≈ 150 μs per file before any I/O.

**Impact:** CPU overhead proportional to chunk count × concurrency, especially visible
when many files are being range-assembled simultaneously.

---

### Finding 6: O(N²) Sort for Input Order Preservation — [RESOLVED v0.9.82]

**Location:** `s3dlio/src/s3_utils.rs` — `get_objects_parallel()`

```rust
out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap());
//                         ^^^ linear scan of uris Vec for every element → O(N²)
```

For N=42 files per worker this is negligible, but the correct implementation is
`HashMap<&str, usize>` pre-built from the input slice, giving O(N log N) sort.

---

## 5. Why minio-py is Faster

### 5a. minio-py Source Code Review Findings

**Repository:** `https://github.com/minio/minio-py` (reviewed March 2026)  
**Files examined:** `minio/minio.py`, `minio/helpers.py`

#### `get_object()` — single streaming GET, zero HEAD requests

```python
# minio/minio.py — get_object()
def get_object(self, *, bucket_name, object_name, ...):
    # No HEAD request — goes straight to GET
    headers = self._gen_read_headers(ssec=ssec, offset=offset, ...)
    response = self._execute(
        method="GET",
        ...
        preload_content=False,   # ← urllib3 returns streaming response, not buffered
    )
    return GetObjectResponse(response=response, ...)
```

`preload_content=False` instructs urllib3 to return the socket handle without
buffering the body. The DLIO reader then calls `resp.read()` once, which materializes
the full 147 MB into a Python `bytes` object via a single socket read. This is one
copy: socket → Python bytes. Not truly zero-copy, but **one full-object copy** with
no TLS/HTTP overhead per chunk.

Contrast with s3dlio's default path (without env var override): 37 separate
`urllib`/`hyper` HTTP streams (one per range), each with its own HTTP headers, flow
control, and reconnection logic.

#### `fget_object()` vs `get_object()` — when HEAD happens

```python
# minio/minio.py — fget_object() DOES HEAD (for atomic ETag-verified write):
head_response = self._head_object(...)     # HEAD #1
response = self.get_object(...)            # GET
# Writes atomically to tmp_file_path → os.rename

# get_object() — does NOT HEAD. Direct GET.
```

The HEAD in `fget_object` is for ETag-based atomicity (writes to a temp file, verifies
ETag before rename). Our DLIO reader uses `get_object()`, not `fget_object()`, so
**zero HEAD requests** from minio-py. This is the key difference from s3dlio's two
HEADs per file.

#### urllib3 connection pool configuration

```python
# minio/minio.py — __init__()
self._http = urllib3.PoolManager(
    timeout=Timeout(connect=300, read=300),  # 5-minute timeout
    maxsize=10,                               # 10 keep-alive connections per host
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where(),
    retries=Retry(
        total=5,
        backoff_factor=0.2,                  # 0.2s, 0.4s, 0.8s... backoff
        status_forcelist=[500, 502, 503, 504]
    )
)
```

Key parameters:
- **`maxsize=10`**: Up to 10 keep-alive TCP connections per host. Once established,
  successive `get_object()` calls reuse connections — no TCP handshake per file.
- **`total=5` retries**: Transparent retry on server errors, no app-level retry logic needed.
- **`backoff_factor=0.2`**: Gentle exponential backoff avoids thundering-herd on
  transient server errors.

With DLIO's `ThreadPoolExecutor(max_workers=min(16, len(filenames)))` = 16 threads but
only 10 pool connections, 6 threads wait for a free connection. For a bandwidth-limited
workload this is fine since bandwidth (not connection count) is the bottleneck. Could
be improved by using `maxsize=16` to match the thread count.

#### `ThreadPool` for uploads only — reads are Python-level

`helpers.py`'s `ThreadPool` class (using `Queue`, `BoundedSemaphore`, `Thread`) is
**only used for multipart uploads** (`put_object()` with `num_parallel_uploads > 1`).
For reads, minio-py is single-threaded per client, with file-level parallelism
entirely at the DLIO reader level via `ThreadPoolExecutor`. This is simpler and avoids
the Tokio scheduling overhead.

#### No range splitting — by design (and a design limitation at high bandwidth)

minio-py has **no range splitting logic whatsoever** for reads. Even for a 1 GB object,
minio-py issues a single `GET` with no `Range` header. This is a deliberate design
choice: the library trusts the OS TCP stack and network infrastructure to saturate the
available bandwidth from a single streaming connection.

**Where this works well:** LAN/data center environments with ≤ 1 Gbps effective bandwidth —
exactly our MinIO test system at 1.2 GB/s. A single TCP connection fills the pipe, and
adding more requests only adds overhead. minio-py achieves 96% of that line rate at NP=4.

**Where this is a design limitation:** High-performance S3 systems (AWS S3 with
large instances, or any 10/100 Gbps interconnect with a fast S3-compatible backend). At those bandwidths, a single TCP connection
tops out at ~4–8 Gbps due to OS TCP stack and socket buffer limits. An application that
never issues range requests leaves 85–95% of available bandwidth on the table. minio-py
would achieve the same ~4–8 Gbps single-stream throughput on a 100 Gbps link as on
a 10 Gbps link — entirely bottlenecked at the connection level.

**s3dlio's adaptive range splitting is architecturally the correct design** — the issue
is that the default threshold (32 MB) is tuned for high-bandwidth systems and fires
too eagerly in bandwidth-saturated 1 Gbps environments. The fix is better defaults
or auto-detection, not removing range splitting.

#### Key implication for s3dlio design

On this 1 Gbps system, minio-py's no-range approach is validated by > 1 GB/s aggregate
throughput at NP=4 (1.097 GB/s observed), achieving 96% of line rate. But this does
**not** generalize to high-bandwidth environments. The user's observation of massive
improvement from 16 MB range splitting on high-performance S3 systems is equally valid.
Both results are correct and reflect the bandwidth-dependent nature of range optimization.
s3dlio needs better auto-detection, not a policy change.

---

### 5b. Observed performance comparison (NP=1, warm epochs)

minio-py's `_prefetch_minio()` uses a simple but effective pattern:

```python
def _fetch_one(filename):
    resp = client.get_object(bucket, key)  # Single streaming GET
    raw = resp.read()                       # Consume entire response at once
    resp.close(); resp.release_conn()
    return filename, np.load(io.BytesIO(raw), allow_pickle=True)["x"]

with ThreadPoolExecutor(max_workers=min(16, len(filenames))) as pool:
    for fname, arr in pool.map(_fetch_one, filenames):
        cache[fname] = arr
```

| Property | s3dlio | minio |
|----------|--------|-------|
| HTTP requests per file | 39 (2 HEAD + 37 GET) | 1 streaming GET |
| Request overhead | High (37 TLS handshakes, 37 HTTP parsings) | Minimal |
| Concurrency model | Tokio async (32 threads) | ThreadPoolExecutor (16 threads) |
| Memory per file | 2× (Rust BytesMut + Python bytes) | 1× (Python bytes only) |
| Thread count at NP=4 | ~512 Tokio threads | ~256 Python threads |

The minio speedup is not from minio being "faster" at S3 — it is from doing **38×
fewer round trips**, using **half the threads**, and **halving peak memory**.

---

## 6. Recommendations

> **s3dlio release cost constraint:** Each Rust change requires ~1.5 hours: code →
> `cargo test` → `cargo build --release` → push git tag → rebuild PyPI wheel → `pip install`
> update in all consumers. **All Rust changes must be planned completely and executed in a
> single batch.** Do not make any s3dlio Rust change until the full list is known, validated
> by env var experiments, and confirmed to be complete.

**Execution order:**
- **Step 1 (now):** Tier 1 env vars — zero effort, validates hypothesis, fully reversible
- **Step 2 (alongside / immediately after):** Tier 2 Python reader fixes — mlp-storage only, no s3dlio rebuild needed
- **Step 3 (after Steps 1+2 measured and complete Rust list confirmed):** Single s3dlio Rust release

---

### Tier 1: Immediate — Environment Variables (Zero Code Changes)

These can be set in the training script before launching `mpirun`. They are completely
reversible. They test the hypothesis that range optimization is the primary bottleneck.

> **⚠️ Important: `S3DLIO_ENABLE_RANGE_OPTIMIZATION` is a no-op for `get_many()`**
>
> This was tested at NP=1 (result: **329.5 ± 0.9 MB/s** vs original **332 ± 0.7 MB/s** — no change).
> Root cause: `S3DLIO_ENABLE_RANGE_OPTIMIZATION` is only read inside `S3ObjectStore::get()`
> (`object_store.rs`). The `get_many()` path routes through `get_objects_parallel()` →
> `get_object_uri_optimized_async()` in `s3_utils.rs`, which checks only
> `S3DLIO_RANGE_THRESHOLD_MB`. Setting `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` in the
> training script has zero effect on range splitting during DLIO training.
>
> **The correct knob for the `get_many` path is `S3DLIO_RANGE_THRESHOLD_MB`.**

```bash
# CORRECT: Raise the range-split threshold above our 147 MB file size
# → files below this threshold get a single streaming GET (no range splitting)
# → eliminates the 37-chunk overhead and both HEAD requests
export S3DLIO_RANGE_THRESHOLD_MB=1000

# Reduce Tokio runtime threads per process:
# → 8 threads is sufficient for an I/O-bound workload
# → expected: CPU drops, less OS scheduler contention at higher NP
export S3DLIO_RT_THREADS=8
```

**What NOT to use (confirmed ineffective):**
```bash
# ❌ No-op for get_many() — only affects S3ObjectStore::get() which DLIO never calls
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0
```

**Expected impact (still to be measured with correct env var):** Bringing s3dlio to
within 10–20% of minio-py at NP=1, and reducing CPU utilization by an estimated 50–70%.

---

### Tier 2: Python Reader Changes (mlp-storage only — no s3dlio rebuild)

These changes are in the DLIO reader only. No s3dlio rebuild or PyPI update required.

#### 2a. Use zero-copy IO wrapper instead of `bytes(data)`

In `npz_reader_s3_iterable.py`, `_prefetch_s3dlio()`:

**Current (copies 147 MB per file):**
```python
cache[fname] = np.load(io.BytesIO(bytes(data)), allow_pickle=True)["x"]
```

**Proposed (zero-copy from Rust memory):**
```python
from s3dlio.compat.s3torchconnector import _BytesViewIO
# ...
raw = io.BufferedReader(_BytesViewIO(data))
cache[fname] = np.load(raw, allow_pickle=True)["x"]
```

`_BytesViewIO` already exists in s3dlio and uses `memoryview`-based `readinto()` to
read directly from the Rust buffer — no copy. This halves peak memory usage.

#### 2b. Cache the minio client per worker

`_prefetch_minio()` creates a new `Minio()` client on every call (every epoch's
prefetch). Client creation is cheap but not free. Cache it as `self._minio_client`
during `__init__`.

#### 2c. Align `max_in_flight` with actual file count

`s3dlio.get_many(uris, max_in_flight=64)` when `uris` has only 21–42 elements (for
NP=4 to NP=1) creates 64 semaphore permits that are never acquired. This is harmless
but creates unnecessary Semaphore allocation overhead. Use
`max_in_flight=min(64, len(uris))`.

---

### Tier 3: s3dlio Rust Changes — Plan Fully, Execute Once

**Do not start until Tier 1 env var results and Tier 2 Python fix are both measured.**
All of the following must be batched into a single release cycle.

These require changes to the s3dlio Rust library.

#### 3a. Eliminate the redundant HEAD request

Pass the object size from `get_object_uri_optimized_async()` directly into
`get_object_concurrent_range_async()` instead of re-HEADing:

```rust
// In get_object_uri_optimized_async():
let object_size = head_resp.content_length().unwrap_or(0) as u64;
if object_size >= range_threshold {
    // Pass known_size to avoid the second HEAD
    get_object_concurrent_range_with_known_size(uri, 0, object_size, None, None).await
}
```

#### 3b. Replace `Mutex<BytesMut>` with direct disjoint-range writes

For non-overlapping range writes into a pre-allocated buffer, no mutex is needed.
Use `tokio::task::spawn_blocking` with a split `BytesMut` or raw pointer writes
(after checking ranges are non-overlapping). This eliminates all mutex contention
in `concurrent_range_get_impl`.

The pattern used by AWS's own transfer manager (`aws-s3-transfer-manager-rs`) and
`object_store` crate uses `BytesMut::split_off` to give each task its own slice
ownership, then assembles with `BytesMut::unsplit()`.

#### 3c. Add auto-detect for bandwidth-limited environments

Introduce a lightweight heuristic or config flag `S3DLIO_BANDWIDTH_LIMIT_GBPS`.
If the configured or detected network bandwidth is at or near saturation, disable
range optimization automatically (single streaming GET). Range optimization only
benefits when per-connection throughput < network_bandwidth / N_connections.

#### 3d. Fix O(N²) sort in `get_objects_parallel`

```rust
// Current (O(N²)):
out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap());

// Proposed (O(N log N) sort after O(N) hashmap build):
let uri_order: HashMap<&str, usize> = uris.iter().enumerate()
    .map(|(i, u)| (u.as_str(), i))
    .collect();
out.sort_by_key(|(u, _)| *uri_order.get(u.as_str()).unwrap_or(&usize::MAX));
```

#### 3e. Object size cache (LRU or epoch-bounded)

For repeated epoch training, s3dlio HEADs the same objects every epoch. An in-process
LRU cache of `(bucket, key) → size` would eliminate all HEAD requests after the
first epoch. At 168 objects × ~500 bytes per HEAD response = 84 KB — negligible
memory for significant latency savings.

---

### Tier 4: Architectural Considerations

#### Streaming vs buffering

The current design buffers the full object in Rust (`BytesMut::zeroed(total_bytes)`)
before returning anything to Python. A streaming design would return a `read()`-compatible
object that Python can consume while Rust is still downloading remaining ranges. NumPy's
`np.load()` reads sequentially through the ZIP central directory then the compressed
chunk — the beginning of the file can be decompressed while the end is still downloading.

This is a significant refactor but would reduce peak memory from O(object_size) to
O(chunk_size) per file.

#### The "s3torchconnector is the same" implication

Since `npz_reader_s3_iterable.py` routes s3torchconnector through `_prefetch_s3dlio`,
the s3torchconnector backend offers no practical advantage in this setup. The
s3torchconnector S3Client (the real AWS-built library) is only used in `obj_store_lib.py`
for single-file `get_data()` calls, not for the batch prefetch path that drives training
throughput. If s3torchconnector is to be benchmarked independently, the DLIO reader
would need a dedicated `_prefetch_s3torchconnector()` method using the actual
`S3Client.get_object()` API.

---

## 7. Expected Impact Summary

| Priority | Change | Effort | CPU ↓ | Memory ↓ | Throughput ↑ |
|----------|--------|--------|-------|----------|--------------|
| 1 | `S3DLIO_RANGE_THRESHOLD_MB=1000` | Env var (zero) | ~50% | ~30% | **+40–60%** |
| 1 | `S3DLIO_RT_THREADS=8` | Env var (zero) | additional ~15% | ~5% | +5% |
| 2 | Fix `bytes(data)` → `_BytesViewIO` | Python reader (~3 lines, no rebuild) | ~5% | **~40%** | +2–5% |
| 2 | Cache minio client per worker | Python reader (easy) | — | — | +1% |
| 3 (batch) | Eliminate redundant HEAD | s3dlio Rust | ~5% | — | +3–5% |
| 3 (batch) | Raise default threshold for on-prem | s3dlio Rust | — | — | systemic |
| 3 (batch) | Fix O(N²) sort | s3dlio Rust | negligible | — | negligible |
| 4 (optional) | Replace Mutex with disjoint writes | s3dlio Rust (complex) | ~10% | — | +5–10% |

**After Priority 1+2 (env vars + Python fix):** Expected s3dlio throughput ≥ 430–480 MB/s,
within ~10% of minio's 459 MB/s warm baseline. Remaining gap, if any, closed by the
batched s3dlio Rust release.

---

## 8. Files Examined

| File | Purpose |
|------|---------|
| `dlio_benchmark/reader/npz_reader_s3_iterable.py` | Batch prefetch reader — entry point for all three libraries |
| `dlio_benchmark/storage/obj_store_lib.py` | Storage abstraction (s3dlio, s3torchconnector, minio) |
| `s3dlio/src/python_api/python_core_api.rs` | `get()`, `get_many()`, `PyBytesView` |
| `s3dlio/src/s3_utils.rs` | `get_objects_parallel()`, `get_object_uri_optimized_async()`, `get_object_concurrent_range_async()`, `concurrent_range_get_impl()`, `get_optimal_chunk_size()` |
| `s3dlio/src/s3_client.rs` | Tokio runtime config, thread count, `run_on_global_rt()` |
| `s3dlio/src/object_store.rs` | `S3ObjectStore::get()`, range optimization enable check |
| `s3dlio/python/s3dlio/compat/s3torchconnector.py` | `_BytesViewIO` zero-copy wrapper |
| `s3dlio/docs/Changelog.md` | v0.9.50 range optimization origin and benchmarks |
| `configs/dlio/workload/unet3d_h100_s3dlio.yaml` | DLIO workload config |

---

## 9. Recommended Next Steps

### Step 1 — Env var experiment (zero risk, do now)

```bash
NP=1 S3DLIO_RANGE_THRESHOLD_MB=1000 S3DLIO_RT_THREADS=8 \
    bash tests/object-store/dlio_s3dlio_train.sh
```

**Do NOT use `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0`** — confirmed no-op for this code path.
Target: ≥ 430 MB/s warm throughput (within ~10% of minio's 459 MB/s). If confirmed,
the range-splitting hypothesis is validated.

### Step 2 — Python reader fix (no s3dlio rebuild, do alongside Step 1)

Replace `bytes(data)` with `_BytesViewIO` in `npz_reader_s3_iterable.py` (~3 lines).
Run NP=1 again with env vars still set. Measure the delta. This requires no s3dlio
rebuild — it is a mlp-storage-only change.

### Step 3 — Compile the complete s3dlio Rust change list (do NOT start coding yet)

Before touching s3dlio Rust, confirm the full list of changes needed so they can be
batched into a single ~1.5-hour release cycle. Current known list:

| Change | Function | Complexity |
|--------|----------|------------|
| Pass `object_size` to eliminate HEAD #2 | `get_object_uri_optimized_async()` → `get_object_concurrent_range_async()` | Low |
| Raise default `RANGE_THRESHOLD_MB` for on-prem (detect non-AWS endpoint or set 500 MB default) | `get_object_uri_optimized_async()` | Low |
| Fix O(N²) sort | `get_objects_parallel()` | Low |
| *(Optional)* Replace `Mutex<BytesMut>` with disjoint-range writes | `concurrent_range_get_impl()` | High — add only if Steps 1+2 data shows measurable residual gap |

### Step 4 — Single s3dlio Rust release

Only after Steps 1–3 are complete and the change list is final: implement all planned
changes, run `cargo test`, `cargo build --release`, push git tag, update PyPI, update
mlp-storage `pip install`. Bump version in `Cargo.toml` + `pyproject.toml`, document
in `Changelog.md`. Re-baseline all benchmarks after the update.
