# Async Runtime Enhancement — Multipart Upload Throughput

**Date**: April 22, 2026  
**Branch**: `feature/connection-pool-autoscale`  
**Version**: v0.9.92  

---

## Background

During testing of the v0.9.92 connection pool fixes and issue #134 backpressure work, a head-to-head
throughput benchmark was run comparing `s3dlio.MultipartUploadWriter` against `s3torchconnector 1.5.0`
against a real MinIO server (`https://172.16.1.40:9000`).

### Benchmark configuration

| Parameter | Value |
|---|---|
| Object size | 512 MiB |
| Part size | 32 MiB |
| Total parts | 16 |
| `max_in_flight` | 8 |
| Parallelism measured | NP=1 and NP=4 |
| Target | Real MinIO (LAN, HTTPS, TLS) |

### Initial benchmark results

| Library | NP=1 | NP=4 |
|---|---|---|
| s3dlio v0.9.92 | 0.457 GB/s | 0.855 GB/s |
| s3torchconnector 1.5.0 | 0.814 GB/s | 0.927 GB/s |
| s3dlio ratio | **1.78× slower** | **1.08× slower** |

s3dlio closes the gap significantly at NP=4 (4 concurrent writers) but is substantially slower
at NP=1 (single writer). The question: why can't s3dlio be fully async?

---

## Root Cause Analysis

### Problem 1 — Batching due to `max_in_flight` < total parts (primary)

With 16 total parts and `max_in_flight=8`, s3dlio uploads in **two sequential batches**:

```
Current execution (max_in_flight=8, 16 parts):

  write(part_1..8)  → semaphore acquire → all 8 permits taken → spawn 8 upload tasks
  write(part_9)     → semaphore.acquire() → NO PERMIT → Python thread parks here
                                             ↑
                                       blocked until one of parts 1-8 finishes
  write(part_10..16)→ each waits for a slot to free up
  close()           → join remaining tasks

  Total wall time ≈ T(batch_1) + T(batch_2)
```

s3torchconnector (AWS CRT-based) enqueues all parts asynchronously and returns from `write()`
immediately. All 16 parts can be in-flight simultaneously:

```
s3torchconnector (all 16 in flight):

  write(part_1..16) → enqueued to CRT pipeline → returns immediately each time
  close()           → blocks until all 16 done

  Total wall time ≈ T(single_batch_of_16)
```

### Problem 2 — `run_on_global_rt(semaphore.acquire_owned())` overhead (secondary)

Even when permits are immediately available (parts 1-8), each `spawn_part()` call executes:

```rust
let permit = run_on_global_rt(async move {
    semaphore.acquire_owned().await  // uncontended, but still async
})?;
```

`run_on_global_rt` spawns a Tokio task, sends through a channel, and does a blocking `recv()` on
the Python caller thread — **even when the semaphore is immediately available**. This is 16
unnecessary round-trips through the Tokio scheduler for a 16-part upload.

---

## Findings Confirmed by Experiment

The batching hypothesis was validated by sweeping `max_in_flight` at fixed object/part size:

### Experiment: `max_in_flight` sweep (512 MiB, 32 MiB parts = 16 total parts)

```
max_in_flight=  8:  0.495 GB/s  (1.08s)   ← 2 batches
max_in_flight= 16:  0.645 GB/s  (0.83s)   ← 1 batch, 30% faster
max_in_flight= 32:  0.627 GB/s  (0.86s)   ← excess permits, marginal overhead
s3torchconnector:   0.814 GB/s  (0.61s)   ← baseline
```

**Key conclusions:**
1. Increasing `max_in_flight` from 8 → 16 recovers **~41% of lost throughput** (0.495 → 0.645 GB/s)
   with zero code changes — pure parameter tuning
2. The remaining ~1.26× gap at `max_in_flight=16` is attributable to `run_on_global_rt` overhead
   per part (16 × Tokio channel round-trips, even uncontended)
3. `max_in_flight=32` provides no gain over 16 — confirming the bottleneck is not permit scarcity
   but rather the semaphore-per-part dispatch overhead

---

## Proposed Changes

### Change 1 — Auto-scale `max_in_flight` based on object size

Replace the fixed default of `max_in_flight=16` with a formula that scales with expected part count.
The goal is to ensure `max_in_flight ≥ ceil(total_size / part_size)` so all parts can be queued
without batching. Since total object size isn't always known upfront, the formula should use a
reasonable upper bound:

```
auto_max_in_flight(part_size) = max(32, ceil(256 MiB / part_size))
```

This ensures:
- `part_size=32 MiB` → `max_in_flight = max(32, 8) = 32` (covers 1 GiB objects without batching)
- `part_size=8 MiB`  → `max_in_flight = max(32, 32) = 32` (covers 256 MiB objects without batching)
- `part_size=64 MiB` → `max_in_flight = max(32, 4) = 32`

Memory ceiling stays bounded: `max_in_flight × part_size` caps at `32 × 32 MiB = 1 GiB` worst case.
For the common 8 MiB part size this is `32 × 8 MiB = 256 MiB` — acceptable.

When `max_in_flight` is explicitly specified by the caller it overrides the auto value.

### Change 2 — Coordinator task + bounded channel (architectural)

Replace the synchronous `run_on_global_rt(semaphore.acquire_owned())` per-part call with a
**background coordinator task** and a **bounded `tokio::sync::mpsc` channel** as the
backpressure mechanism.

**New execution model:**

```
Python write() → part_tx.blocking_send(PartCmd::Part(chunk))
                  └── if channel full → parks Python thread (backpressure)
                  └── otherwise → returns immediately

[Coordinator task — runs entirely on Tokio runtime, never touches Python thread]
  loop:
    chunk = rx.recv().await
    permit = semaphore.acquire_owned().await   ← async, no Python thread parking
    spawn upload_task(chunk, permit)

Python close() → part_tx.blocking_send(PartCmd::Finish)
              → run_on_global_rt(coordinator.await)  ← one blocking call at end
```

**Channel capacity = `max_in_flight`**. Backpressure is preserved:
- `blocking_send()` only blocks when channel is full AND coordinator is waiting on semaphore
- Memory stays bounded at `max_in_flight × part_size` (same guarantee as current design)
- **Zero `run_on_global_rt` calls in the hot write path** — all async work stays on the Tokio runtime

**Struct changes:**

```rust
pub struct MultipartUploadSink {
    part_tx: tokio::sync::mpsc::Sender<PartCmd>,  // bounded channel to coordinator
    coordinator: JoinHandle<Result<MultipartCompleteInfo>>,
    buf: Vec<u8>,
    cfg: MultipartUploadConfig,
    total_bytes: u64,
    finished: bool,
}

enum PartCmd {
    Part { data: Bytes, part_number: i32 },
    Finish,
}
```

---

## Expected Outcome

After both changes:

| Scenario | Before | Expected After | Target |
|---|---|---|---|
| NP=1, max_in_flight=8 (default) | 0.457 GB/s | 0.75–0.85 GB/s | ≥ 0.80 GB/s |
| NP=1, max_in_flight=16 | 0.645 GB/s | 0.75–0.85 GB/s | ≥ 0.80 GB/s |
| NP=4 | 0.855 GB/s | 0.90+ GB/s | ≥ 0.90 GB/s |
| vs s3torchconnector NP=1 | 1.78× slower | ≤ 1.15× slower | ≤ 1.15× |

Change 1 alone recovers ~41% of the single-writer gap. Change 2 eliminates the remaining
Tokio-channel overhead. Together they should bring s3dlio within ~15% of s3torchconnector's
single-writer throughput while preserving all backpressure guarantees from issue #134.

---

## Backpressure Contract Preserved

Both changes maintain the issue #134 backpressure contract:
- Memory is still bounded at `max_in_flight × part_size` bytes
- Writes block when the pipeline is full
- No unbounded task accumulation
- `finish()` / `close()` still waits for all in-flight parts to complete

The difference is *where* the blocking happens: currently on every `spawn_part()` call even
when slots are available; after Change 2, only when the channel is full (i.e., all
`max_in_flight` slots are genuinely occupied).
