# Performance & Concurrency Audit — Issue #148 and Related Findings

**Date**: 2026-07-07
**Revised**: 2026-07-07 — incorporated corrections from the [adversarial review](./PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md) (5 corrections, see §6) and a maintainer design decision on HTTP/2 opt-out controls (§2.2, Phase 3)
**Status**: Proposed — awaiting developer review before implementation
**Origin**: [mlcommons/storage#701](https://github.com/mlcommons/storage/issues/701) (unet3d object-storage benchmark, client-bound at ~1.1 GB/s/process against a ~10 GB/s/node endpoint), tracked in [russfellows/s3dlio#148](https://github.com/russfellows/s3dlio/issues/148)
**Verified against**: s3dlio `main` @ `38fe812` (past v0.9.106)
**Context**: s3dlio is approaching a 1.0 release. The priority for every item below is **stability first** — nothing here should ship without the tests called out in the plan. Several items are genuine correctness risks (not just performance), including one credible silent-data-corruption path; those are flagged explicitly.

---

## How this document is organized

1. [Root causes confirmed in issue #148](#1-root-causes-confirmed-in-issue-148) — the four originally-reported problems, verified against current code.
2. [Deep correctness review of the four proposed patches](#2-deep-correctness-review-of-the-four-proposed-patches) — adversarial review of each patch as submitted, with verdicts.
3. [Additional issues found during audit](#3-additional-issues-found-during-audit) — the same bug classes found elsewhere in the crate, not part of the original report.
4. [Proposed implementation plan](#4-proposed-implementation-plan) — phased, dependency-ordered, with required tests per phase.
5. [Quick-reference table](#5-quick-reference-table) — every file:line finding in one place.
6. [Revision history](#6-revision-history) — corrections incorporated from the adversarial review.

---

## 1. Root causes confirmed in issue #148

All four claims in the original report were independently re-verified by reading the actual current source (not just the issue's diff snippets). All four are real and unchanged in current `main`.

### 1.1 — Loader drives all fetches from one async task

**File**: `src/python_api/python_aiml_api.rs` — three call sites: `PyBytesAsyncDataLoader::__iter__` (~line 239), `items()` (~line 372), the Parquet streaming loader (~line 2213).

```rust
let mut stream = stream::iter(0..len)
    .map(|idx| {
        let ds = dataset.clone();
        async move { ds.inner.get(idx).await }
    })
    .buffer_unordered(prefetch);
```

`buffer_unordered(prefetch)` interleaves polling up to `prefetch` futures — but only *within the single task* driving `stream.next().await`. That's async concurrency, not parallelism: Tokio can only run that one task on one worker thread at a time, so raising `prefetch` doesn't help. All request-driving and body-accumulation work is funneled through ~one core's capacity regardless of depth. This produces the reported "flat ~1.1 GB/s/process regardless of prefetch depth" symptom.

### 1.2 — https client never gets HTTP/2 window tuning; no override exists

**File**: `src/reqwest_client.rs`, `build_reqwest_client_raw` (~line 424).

H2 window tuning (`http2_adaptive_window`, `http2_initial_stream_window_size`, etc.) is applied only inside `if h2c { ... }` — the branch for plain **cleartext** `http://` connections. For `https://`, HTTP/2 is negotiated automatically via TLS ALPN with **no** window configuration applied at all, and there is **no existing knob** to change this. Confirmed in the crate's own documentation, in three places:

- `src/constants.rs:410` — *"`https://` endpoints always negotiate HTTP/2 via TLS ALPN regardless of this setting."*
- `src/constants.rs:419` — *"`https://` endpoints are unaffected — HTTP/2 is still negotiated automatically via TLS ALPN."*
- `src/reqwest_client.rs:10` — *"Automatic HTTP/2 negotiation via TLS ALPN on `https://` endpoints (**always on**)"*

**Important, and a source of confusion during the initial investigation**: the existing `S3DLIO_H2C` env var does *not* cover this. It only controls h2c (HTTP/2 cleartext) on `http://`. There is no equivalent for `https://`. On an h2-capable https endpoint (most modern S3-compatible stores, including MinIO), ALPN silently negotiates HTTP/2 and hyper multiplexes the connection with a **default 5 MiB window** — an effectively uncapped-by-config, ~5 MiB/RTT per-process ceiling once storage throughput exceeds what that window allows.

### 1.3 — Range-GET assembly double-copies every byte

**File**: `src/s3_utils.rs`, `concurrent_range_get_impl` (~line 1154).

```rust
let body = resp.body.collect().await.context("collect chunk body")?;
let chunk_data = body.into_bytes();
// ...later, after ALL chunks have arrived:
let mut output = BytesMut::with_capacity(total_bytes);
for (_, chunk) in chunks {
    output.extend_from_slice(&chunk);
}
```

Each chunk is collected into its own `Bytes`, then every chunk is copied *again* into a second assembled buffer — a full extra copy of every byte of the object, plus a transient ~2x memory peak (all per-chunk buffers and the assembled buffer alive simultaneously).

### 1.4 — Connector fully buffers every response body before the SDK sees byte one

**File**: `src/reqwest_client.rs` (connector, ~line 317).

```rust
let resp_body = resp.bytes().await.map_err(|e| ConnectorError::io(e.into()))?;
// ...
SdkBody::from(resp_body),
```

The connector buffers the **entire** response body in memory before `SdkBody::from(resp_body)` ever hands anything to the SDK/caller. No receive/consume overlap; an extra whole-body buffer per GET; `first_byte_time` metrics that in practice measure last-byte.

---

## 2. Deep correctness review of the four proposed patches

Each patch was reviewed adversarially against the actual current code (not just the diffs), including reading the real `bytes` and `sync_wrapper` crate sources on disk to verify claimed semantics rather than trusting documentation summaries.

### 2.1 — Patch 1: `tokio::spawn` each fetch (loader) — **NEEDS SIGNIFICANT REWORK AS SCOPED**

**Proposed change**: wrap each fetch in `tokio::spawn(async move { ds.inner.get(idx).await })` at all three sites in §1.1, converting the resulting `JoinHandle`'s `Result<T, JoinError>` into the existing `DatasetError` type.

**Confirmed regression, must be fixed in the same change, not deferred:**
Grepped the entire loader file and `data_loader/` module: there is **no `Drop` impl, no `__del__`, no `.abort()` anywhere** on the Python-facing iterator wrappers (`PyBytesDataLoaderSyncIter`, `PyObjectDataLoaderSyncIter`, `ParquetStreamIter`) — they hold only `rx: Mutex<Receiver<...>>`.

- **Today**: dropping the Python iterator drops `rx` → the producer's `tx.send()` fails → producer breaks → the `stream` (holding *bare, unspawned* futures) drops → `buffer_unordered` cancels the in-flight futures for free, because dropping the stream drops the futures it owns.
- **After the patch as submitted**: those futures become `JoinHandle`s. Dropping a `JoinHandle` **detaches, it does not abort** — this is documented Tokio semantics, not a bug in the patch's logic, but a semantic change the patch introduces. Every early Python-side break (validation subsampling, early stop, an exception) leaves up to `prefetch` in-flight GETs running to completion in the background — burning bandwidth/CPU fetching data nobody wants anymore.

This is bounded (self-terminating, not an unbounded leak) but is a genuine functional regression from current behavior. **Fix has a ready precedent already in this codebase**: `tokio_util::sync::CancellationToken` is already a dependency and already used for exactly this class of problem in `src/data_loader/async_pool_dataloader.rs` (lines 194, 201, 228, 244–288), already plumbed through `LoaderOptions.cancellation_token` (`src/data_loader/options.rs:181`). Note that pattern is *cooperative* (checked between iterations) and would not interrupt an already-in-flight `.await` on its own — a correct fix needs either tracked-and-explicitly-`.abort()`'d `JoinHandle`s, or a `select!` around each spawned fetch against the cancellation token.

**A bonus, currently-hidden correctness bug this patch incidentally fixes:**
Today, a panic inside a bare (unspawned) fetch future propagates up through the *producer* task's own poll. Tokio's per-task panic boundary catches it there, silently terminating the whole producer task — the channel just closes, and Python sees a plain `StopIteration`, **not an error** — silently truncating the dataset with no indication anything went wrong. Wrapping in `tokio::spawn` isolates the panic into its own task, correctly surfaces as `JoinError::is_panic()`, and (once mapped to `DatasetError::Backend`) becomes a proper `PyRuntimeError` in Python. Worth keeping and highlighting in the eventual PR description as a fixed bug, not just a side effect.

**Checked and ruled out as concerns:**
- Send/`'static` bounds: `Dataset: Send + Sync + 'static`, `Item: Send + 'static` (`src/data_loader/dataset.rs:57,61`); `get()` is `#[async_trait]` (not `?Send`), so the future is already `Send`. `tokio::spawn` adds no new bound violation.
- Ordering: `buffer_unordered` was already unordered; no regression.
- Error plumbing: `DatasetError` already has `#[from] AnyError`, so mapping a `JoinError` in is a trivial, non-breaking addition.

**Verdict**: implement `tokio::spawn` **together with** cancellation handling in the same change — not as a follow-up. Model the cancellation approach on `async_pool_dataloader.rs`'s existing use of `CancellationToken`.

### 2.2 — Patch 2: force HTTP/1.1 for all https clients — **REWORKED: WINDOW TUNING PLUS EXPLICIT, MAINTAINER-DIRECTED OPT-OUT CONTROLS**

**Proposed change** (as originally submitted):
```diff
     Ok(builder
+    } else {
+        builder = builder.http1_only();
     }
```

**In-repo context, stated with appropriate weight (corrected from an earlier overstated version of this critique — see §6.2)**: `src/range_engine_generic.rs:13-18` documents that a *prior* multi-connection client design (`sharded_client.rs`) was deliberately removed, because "that premise is incorrect for reqwest 0.13 + HTTP/2: the connection pool is already concurrent by design." This codebase's architecture leans on H2 multiplexing rather than proliferating TCP/TLS connections — but that history is evidence the tradeoff was considered once, not proof that an https HTTP/1.1 opt-out would be wrong today.

The original critique claimed forced HTTP/1.1 would require "each chunk request to do its own TCP+TLS handshake." That overstates the cost: `DEFAULT_POOL_MAX_IDLE_PER_HOST = usize::MAX` (`src/constants.rs:464`) means the reqwest pool already keeps large numbers of idle HTTP/1.1 keep-alive connections around, so **HTTP/1.1 requests can and do reuse pooled connections** rather than paying a fresh handshake every time. The defensible, weaker claim is: forcing HTTP/1.1 gives up H2 multiplexing, likely needs more concurrent TCP/TLS connections for the same fan-out (e.g. `concurrent_range_get_impl`'s many parallel range-GETs against one object), and so *could* trade one bottleneck (uncapped H2 window) for another (connection-count/handshake overhead) — a real but conditional, benchmark-dependent concern, not a certainty.

**Maintainer decision (final, 2026-07-07 revision, supersedes the earlier "opt-out" direction)**: HTTP/2 is *nearly always slower* than HTTP/1.1 in practice for this workload class, in the maintainer's own experience. The default must therefore **reverse**: HTTP/1.1 becomes the default for **both** `http://` and `https://`, and HTTP/2 becomes explicitly opt-in for each. This is a breaking change to the default `https://` behavior (was ALPN-negotiated H2, now HTTP/1.1); it lands in a minor-version bump under the pre-1.0 semver-cargo convention and is called out in the changelog.

**Revised scope** — three pieces, all opt-in, all defaulting to HTTP/1.1:

1. **`S3DLIO_H2C=1`** — unchanged. Continues to opt-in to h2c (HTTP/2 cleartext prior-knowledge) on `http://`. Existing users of this var see no change.
2. **`S3DLIO_HTTPS_H2=1`** — new. Opt-in to HTTP/2 on `https://` via TLS ALPN. When set, the reqwest builder does *not* call `.http1_only()` and reqwest offers `h2` in its ALPN advertisement so the server may negotiate H2. When unset (the new default), the reqwest builder calls `.http1_only()`, restricting the connection to HTTP/1.1 regardless of ALPN.
3. **`S3DLIO_ENABLE_HTTP2=1`** — new master switch. When set, implies both `S3DLIO_H2C=1` and `S3DLIO_HTTPS_H2=1` — a single, memorable var for users who want H2 wherever it's available. Precedence is a simple OR: H2 is enabled for scheme S iff the per-scheme var for S is set OR the master switch is set.

**Window tuning** (originally the recommended alternative to Patch 2) still lands: extend `H2WindowConfig::from_env()` (currently gated to the `h2c=true` branch) to also apply when `https_h2=true`. Fixes the uncapped default receive window for anyone who opts in — the fix is only relevant when H2 is on.

**Correction to the earlier draft of this section**: the previous rev proposed `S3DLIO_DISABLE_HTTP2` as a master OFF switch on top of an https-H2-on-by-default. That framing is superseded — since the default is now OFF, an OFF-master would be redundant, and the ON-master (`S3DLIO_ENABLE_HTTP2`) is the useful one.

**Verdict**: implement all four pieces (three env vars + window tuning) together in Phase 3 (see §4). Default behavior *changes* for https: H2 users must now set `S3DLIO_HTTPS_H2=1` or `S3DLIO_ENABLE_HTTP2=1`. This is called out as a breaking change in the changelog. No design sign-off remaining — the shape above is the agreed design.

### 2.3 — Patch 3: pre-split segments, O(1) unsplit (range engine) — **SAFE TO IMPLEMENT AS-IS**

Verified directly against the actual `bytes-1.11.1` crate source on disk (`~/.cargo/registry/.../bytes-1.11.1/src/bytes_mut.rs`), not just documentation:

- `BytesMut::zeroed(len)` = `BytesMut::from_vec(vec![0; len])` — fully zero-initialized, no uninitialized-memory risk.
- `split_to(at)` has `assert!(at <= self.len())` — a real panic precondition, but the current range-building loop (`src/s3_utils.rs:1166-1173`) guarantees `range_end > range_start` strictly and segment sizes sum exactly to `total_bytes`, so the assert cannot trip given correct range math.
- **Important correction to the general understanding of this technique**: `unsplit()` does **not** panic on non-adjacent buffers. Read directly (lines 907-916 of `bytes_mut.rs`): `unsplit` calls `try_unsplit`, and on failure (non-contiguous) falls back to `self.extend_from_slice(other.as_ref())` — a real copy, but never a panic or UB. So even a latent ordering bug in the eventual implementation degrades to "correct but slow" (loses the zero-copy win), never to a crash or memory corruption. This is meaningfully lower-risk than it might appear.
- Segments are produced by ascending `split_to` calls in range order; sorting `parts` by `idx` before `unsplit`-chaining restores exactly that order, so the O(1) fast path is correctly triggered by construction.
- No `unsafe` needed anywhere; `copy_from_slice` into `&mut [u8]` is fully safe; error paths (`?`/`bail!`) drop partially-filled `BytesMut`s cleanly with no UB regardless of how much of a segment was written when an error fires.

**One implementation nuance for the real PR** (not verifiable from a design sketch alone): the bounds check (`if end > seg.len() { bail!(...) }`) must run **before** `seg[written..end].copy_from_slice(&frag)`, since `copy_from_slice` panics (not `Result`) on a length mismatch — confirm the actual diff checks length before the copy, not after.

**Minor, non-blocking implementation note**: a design sketch using `body.try_next()` implies `ByteStream: TryStream`, which needs `aws-smithy-types` feature flags not currently requested (`Cargo.toml:63` has no features listed). `ByteStream` already has an inherent `pub async fn next(&mut self) -> Option<Result<Bytes, Error>>` (verified in `aws-smithy-types-1.4.7/src/byte_stream.rs:299`) that achieves the identical loop with **zero new feature flags** — simpler, and avoids coupling to whatever feature flags Patch 4 may also be turning on.

**Verdict**: safe to implement as-is. Lowest-risk of the four.

### 2.4 — Patch 4: streaming connector bodies + distributed retries — **NEEDS A SPECIFIC FAULT-INJECTION TEST BEFORE MERGE**

**Proposed change**: connector hands the SDK a live stream (`SdkBody::from_body_1_x` over a `StreamBody`/`sync_wrapper::SyncStream` wrapping `resp.bytes_stream()`) instead of a fully-buffered body. Because `send()` now resolves at response headers rather than after the full body downloads, the SDK's own retry no longer covers body-transfer failures — compensated with bounded whole-request retry loops (reusing `crate::constants::max_retry_attempts().max(1)`, linear backoff `100ms * attempt`) independently at four call sites: `S3Ops::get_object`, `get_object_range`, `S3ObjectStore::get`, and the range-chunk task inside `concurrent_range_get_impl`. New Cargo surface: `http-body`, `sync_wrapper` (features=["futures"]), `aws-smithy-types` gains feature `http-body-1-x`.

**Things verified sound, not just assumed sound:**
- **`SyncStream`/`SyncWrapper` soundness** — read `sync_wrapper-1.0.2/src/lib.rs` directly: `SyncStream<S>::poll_next` requires `Pin<&mut Self>`. Rust's own aliasing rules make it *impossible* to hold two live `&mut` references to the same value simultaneously, from any thread — concurrent access through the safe API is statically excluded by the type system, not just discouraged by convention. There's no runtime check to "fail," because the compiler already prevents the unsafe scenario at compile time. Confirmed why it's needed: `aws-smithy-types::body::SdkBody::from_body_1_x_internal` boxes into `http_body_util::combinators::BoxBody<Bytes, Error>`, which requires `Send + Sync`.
- **New Cargo deps are not actually new** — `sync_wrapper 1.0.2` (with the exact `"futures"` feature needed) and `http-body 1.0.1` are already present in `Cargo.lock` as *transitive* dependencies (verified directly). Promoting them to direct deps introduces no new supply-chain surface. `aws-smithy-types`'s `http-body-1-x` feature is a legitimate, existing opt-in feature not currently enabled by anything else in the dependency graph.
- **No public API break** — `S3Ops::get_object`/`get_object_range` and `S3ObjectStore::get`/`get_range` all remain `pub`/`pub(crate) async fn ... -> Result<Bytes>`; consumers are unaffected signature-wise.
- **Retry idempotency for the three simple whole-object paths is genuinely safe** — each currently does `send().await` then `.collect().await?.into_bytes()`, producing a fresh, independently-owned `Bytes` per call with no shared mutable buffer across attempts. A retry loop wrapping the whole `send+collect` pair per attempt is trivially idempotent there.

**The one specific, high-severity risk that needs a real test, not just a code read — a direct consequence of combining this patch with Patch 3:**

The range-chunk retry inside `concurrent_range_get_impl`, once Patch 3 lands, writes streamed bytes **directly into a shared, pre-allocated `seg: BytesMut` slice** via `copy_from_slice` at an incrementing `written` offset — Patch 3's whole design point is removing the "fresh `Bytes` per chunk" safety net that made a naive retry trivially safe before. If a retry loop re-issues the range GET after a partial failure but does **not** reset `written = 0` fresh at the top of *every* attempt — or if a later attempt writes fewer bytes than a previous failed attempt already wrote into `seg` — stale bytes from the failed attempt could silently remain in `seg[0..old_written]`, mixed with new bytes from the successful retry. **No error would be raised**, because Patch 3's own bounds checks only catch length mismatches (`written != seg.len()`), not "correct total length, wrong content" contamination across retry attempts.

This is silent data corruption in a storage client approaching 1.0 — the worst class of bug to ship undetected. This could **not** be confirmed or ruled out from a design sketch alone; it depends entirely on the literal implementation of the retry loop around the segment-fill logic.

**Other, lower-severity findings:**
- The retry loop is duplicated across 4 call sites with minor structural differences (some also do `log_op`/`LogContext` bookkeeping) — see §3.4 for the recommendation to introduce a small shared helper as part of this same patch, not as separate follow-up cleanup.

**Verdict**: everything about the streaming-connector mechanism itself (SyncStream soundness, Cargo deps, API surface, simple-path retry idempotency) checks out as safe based on direct verification. **Block merge on an explicit fault-injection test**: inject a failure partway through a range-chunk's body stream (after N of M bytes written into its segment), confirm the retry resets `written = 0` and the segment is refilled from scratch, and assert the final reassembled object is byte-for-byte correct — not just correct length.

---

## 3. Additional issues found during audit

A broader sweep of the ~49k-line crate for the same bug *classes* (not just the four originally-reported instances) found meaningfully more scope than the original issue implied.

### 3.1 — The same missing-`tokio::spawn` bug, in more places

**High confidence / high severity:**

| File:line | What | Why it matters |
|---|---|---|
| `src/data_loader/async_pool_dataloader.rs:234-315` (`run_async_pool_worker`) | The **actual Rust-level core DataLoader**, one layer below the Python bindings audited in §1.1. A single `tokio::spawn` launches one worker task; *inside* that task, `FuturesUnordered` holds `Box::pin(async move { store.get(&uri).await })` items — never individually spawned. | This is the identical bug, in the layer the Python loader is built on top of. Fixing only the Python-binding layer (§2.1) would leave this lower layer's own callers (if any exist independent of the Python bindings) still affected. |
| `src/s3_utils.rs:1483-1504` (`get_objects_parallel`, pre-stat phase) | `join_all(stat_futs)` with no spawn — **6 lines later, in the same function**, the GET phase correctly does `tokio::spawn`. | The internal inconsistency within one function is strong evidence this is an oversight, not a deliberate design choice — a template for the fix already exists two lines away. |
| `src/data_loader/parquet_file_cache.rs:110` (`fetch_and_parse`), called from `parquet_rg.rs:567-602` and `parquet_index.rs:316-329` | `join_all`/no-spawn over futures that do network fetch **and** Thrift metadata parsing (genuinely CPU-bound). | Stronger case than a pure-I/O GET — CPU-bound decode work for N files serializes onto one core at epoch-1 cold start across a whole dataset. |
| `src/checkpoint/reader.rs:203-244` (`read_all_shards_concurrent`, `*_with_validation`) | `try_join_all(futures)` over per-shard GETs, no spawn. | Same GET-with-body-accumulation shape as the confirmed bug, applied to distributed-checkpoint shard reads across ranks. |
| `src/azure_client.rs:436-472` (`upload_multipart_stream`) | `FuturesUnordered` over `stage_block` calls, no spawn at all. | Azure's own block-upload path — worse than S3's multipart (which does spawn; see §3.2 for its own separate gap). |
| `src/range_engine_generic.rs:305-369` (`RangeEngine::download_with_ranges`) | Shared single-large-object range-split engine used by Azure and GCS (`object_store.rs` `get_with_range_engine` for both). No spawn. | Currently limited exposure: `RangeEngine` is disabled by default via the Azure/GCS store config defaults (`src/object_store.rs`) and the file-store config defaults (`src/file_store.rs`, `src/file_store_direct.rs`) — not `src/constants.rs`, which documents the feature but isn't the authoritative source of the default (corrected per §6.5). `file_store`/`file_store_direct` backends that also use this engine are unaffected in practice because their `get_range` delegates to `tokio::fs`/`spawn_blocking` internally. Still worth fixing for consistency and for when users opt in for large Azure/GCS objects. |

**Medium confidence:**

| File:line | What | Note |
|---|---|---|
| `src/object_store.rs:565-583` (default `pre_stat`), `src/s3_utils.rs:1364-1371` (`stat_object_many_async`) | `buffer_unordered`/`try_join_all` over HEAD-only stat calls, no spawn. | Lighter than a GET (no body to accumulate), but at the docstring-recommended `max_concurrent` of 100+, request-signing/header-parsing work for all of them still serializes onto one core. |
| `src/data_loader/s3_bytes.rs:97-110` (`ReaderMode::Range`) | `.buffered(max_inflight)` over range GET + body, no spawn. | Same shape as the confirmed bug; flagged medium rather than high only because current exposure (how often `ReaderMode::Range` is actually selected vs. the default reader mode) wasn't confirmed in this pass. **Note this file also has the double-copy bug, §3.3** — worth fixing both in the same change. |

**Checked and ruled out** (pattern-matched `buffer_unordered`/`join_all` but not flagged): `multi_endpoint.rs:608-643` (`list_all_endpoints`, capped at the small number of configured endpoints — spawning wouldn't change scheduling meaningfully), `reqwest_client.rs:580-599` (`warmup_connection_pool`, one-time startup cost, not steady-state), and several delete-only batch operations across `gcs_client.rs`, `file_store.rs`, `file_store_direct.rs`, `object_store.rs` (no body to accumulate; file-store variants already delegate real work to `tokio::fs`/`spawn_blocking`).

### 3.2 — A second, related bug: "drop doesn't abort" in code that *already* spawns correctly

This is distinct from §3.1 — it affects places that already use `tokio::spawn` correctly for concurrency, but handle early-termination incorrectly.

**The pattern that gets it wrong** (early-return-on-first-error):

```rust
// multipart.rs:556-627 (S3 multipart part upload), and similarly
// s3_utils.rs:1508-1520, :1554-1567, :1843-1856
// (get_objects_parallel, *_with_progress, put_objects_parallel_with_progress)
while let Some(res) = futs.next().await {
    out.push(res??);   // <-- short-circuits on the FIRST failure
}
```

On the first error, `?` (or `??`) returns from the function immediately, dropping the remaining `FuturesUnordered`/`Vec<JoinHandle>` before all tasks finish. Dropping a `JoinHandle` detaches, does not abort, the task — the remaining in-flight `UploadPart` (or GET/PUT) requests keep running in the background after the function has already returned an error to its caller.

**The pattern that gets it right** (full-drain, already in this codebase — use as the template):

```rust
// object_store.rs:3646-3722 (generic_upload_files, generic_download_objects)
// and prefetch.rs:38-54 (start_prefetch)
while let Some(join_res) = futs.next().await {
    match join_res {
        Ok(Ok(_)) => summary.succeeded += 1,
        Ok(Err(e)) => summary.failed.push(e),   // record, don't bail
        Err(join_err) => summary.failed.push(join_err.into()),
    }
    // loop ALWAYS runs to completion
}
```

`tokio::spawn` + `Arc<Semaphore>` + `FuturesUnordered<JoinHandle<...>>`, but the drain loop always runs to completion regardless of individual task failures, accumulating a succeeded/failed summary instead of short-circuiting. Nothing is ever dropped early, so there's no detach-leak.

**Affected sites needing the full-drain fix**: `multipart.rs:556-627`, `s3_utils.rs:1508-1520`, `s3_utils.rs:1554-1567`, `s3_utils.rs:1843-1856`.

### 3.3 — The same buffer double-copy bug, in more places

**High confidence:**

`src/range_engine_generic.rs:371-389` (`download_with_ranges`) — nearly a copy-paste of the S3 bug already tracked as Patch 3:

```rust
let mut parts: Vec<(usize, Bytes)> = Vec::with_capacity(n_ranges);
while let Some(result) = chunks.next().await {
    let (idx, bytes) = result?;
    parts.push((idx, bytes));
}
parts.sort_by_key(|(idx, _)| *idx);
let mut assembled = Vec::with_capacity(total_size);
for (idx, bytes) in parts {
    assembled.extend_from_slice(&bytes);   // second copy
}
```

This is the shared engine backing Azure and GCS large-object range downloads (`object_store.rs` `AzureObjectStore::get_with_range_engine` and `GcsObjectStore::get_with_range_engine`), plus `file_store`/`file_store_direct` when `enable_range_engine` is set. `RangeEngine` is off by default, but it's the documented, recommended mechanism specifically for large Azure/GCS file workloads — exactly the hot path it exists to serve, whenever a user turns it on. The same fix technique as Patch 3 applies directly (ranges are known up front before I/O starts).

**Medium-high confidence:**

`src/data_loader/s3_bytes.rs:99-114` (`ReaderMode::Range`) — a third, independent occurrence:

```rust
let mut out = Vec::with_capacity(size as usize);
let mut chunks = stream::iter(0..n_parts).map(|i| { /* range GET */ }).buffered(max_inflight);
while let Some(res) = chunks.next().await {
    let bytes = res.map_err(DatasetError::from)?;
    out.extend_from_slice(&bytes);   // second copy, even though out is pre-sized
}
```

`out` is at least pre-sized (`Vec::with_capacity(size)`, no reallocation churn), but every byte of every part is still copied a second time. This is the dataset-sample fetch path used by `ReaderMode::Range` — invoked per-sample, potentially many times per training epoch, for samples that can be multi-MB. Unlike the range-engine finding, this isn't behind an opt-in flag off by default — it's the direct-call path a user gets by choosing `ReaderMode::Range`.

**Minor, low-priority:**

`src/file_store_direct.rs:819-869` (`try_read_file_direct`, O_DIRECT read loop) — `result` starts as `Vec::new()` and grows by repeated amortized reallocation instead of `Vec::with_capacity(file_size)`, even though `file_size` is known before the loop starts. The per-chunk copy from the aligned scratch buffer into `result` is itself necessary (O_DIRECT requires reads to land in a kernel-aligned buffer) — only the missing capacity hint is wasteful. One-line fix, near-zero risk, lower priority than the two findings above.

**Checked and ruled out**: `multipart.rs` upload path (legitimate bounded streaming accumulator — total size isn't known upfront from arbitrary incremental `write()` calls, so there's no equivalent "pre-split one big buffer" available), the various `BufferedObjectWriter`/`ArrowWriter` implementations (same legitimate pattern, final `Bytes::from(mem::take(&mut buffer))` is zero-copy), `multi_endpoint.rs` (dispatches to a single endpoint, no multi-source aggregation), `s3_ops.rs`/`s3_copy.rs` (single-shot GETs, no multi-chunk assembly), `parquet_rg.rs::S3AsyncFileReader::get_bytes` (delegates directly, no local re-copy), GCS's own internal accumulation in `google_gcs_client.rs` (already streams into one growable `BytesMut` correctly), and small fixed-size header/pattern construction in `npz.rs`/`data_gen.rs`/`tfrecord_index.rs`.

### 3.4 — Retry logic: three independently hand-rolled shapes, no shared helper

Cataloged every retry implementation in the crate (grep for retry/attempt/sleep across `src/`):

1. **SDK-level retry** (`src/s3_client.rs:317-326,422-428`) — `RetryConfig::standard().with_max_attempts(crate::constants::max_retry_attempts())`. AWS smithy's own exponential-backoff-with-jitter. Applied uniformly to both the global client and per-endpoint clients. This is the one "off the shelf" mechanism, and it's exactly the coverage Patch 4 (§2.4) partially removes for body-transfer failures.

2. **`put_verified_with_retry`** (`src/python_api/python_core_api.rs:171-294`) — opt-in via `S3DLIO_PUT_VERIFY`; bounded loop, budget `S3DLIO_PUT_MAX_RETRIES` (default 3), **fixed** delay via `S3DLIO_PUT_RETRY_DELAY_MS` (default 1000ms, no exponential backoff, no jitter). Sound and self-contained; retries a whole-object PUT (idempotent overwrite) plus HEAD-verify, deleting the truncated object before retrying.

3. **GCS full-read retry** (`src/google_gcs_client.rs:404-483`) — full-read retry loop and a separate tail-repair loop for truncated reads. Constants are **hardcoded**, not env-configurable (`GCS_FULL_READ_RETRY_MAX_ATTEMPTS = 3`, `GCS_FULL_READ_REPAIR_MAX_ATTEMPTS = 4`, lines 35-36). Gated by string-matching on error text ("timeout"/"unavailable"/"cancel"/"resource_exhausted"). **No delay/backoff at all between attempts** — immediate retry. Reads are idempotent so this isn't unsafe, but three immediate retries on `RESOURCE_EXHAUSTED`/`UNAVAILABLE` could tighten a retry storm against a struggling backend instead of relieving it. This is inconsistent with the rest of the crate's philosophy of making tunables env-configurable (per `constants.rs`'s own stated design goal).

**No shared retry helper exists anywhere** in the crate — every hand-rolled loop duplicates its own `for attempt in 1..=N { match ... }` shape from scratch. Patch 4 (§2.4) as scoped would add a **fourth** independently-invented shape, duplicated across its 4 call sites (effectively 4 more copies).

**Recommendation**: given Patch 4 already needs this exact retry shape (bounded attempts from the existing `max_retry_attempts()` budget, applied specifically to body-transfer-phase failures) at 4 call sites, introduce one small, narrowly-scoped shared helper (e.g. `retry_get_body<F, Fut, T>(op: F) -> Result<T>`) as part of implementing Patch 4 — not as separate cleanup, and not as an attempt to unify with `put_verified_with_retry` or the GCS retry loop (those have different semantics — idempotent-write-with-verify vs. read-body-resume — and forcing one general helper to cover all three would be over-engineering for what should stay a narrowly-scoped perf fix). Retrofitting a helper later means touching the same 4 call sites a second time.

**Reassuring finding, worth stating plainly**: no non-idempotent retries and no unbounded retry loops were found anywhere in the crate. Every retry loop observed is bounded by an explicit attempt counter, and every retried operation (GET, PUT-with-verify-and-cleanup) is safe to retry. This is a genuinely solid property for a library approaching 1.0.

### 3.5 — Areas checked and found clean (no action needed)

- **HTTP client construction is not duplicated or drifted.** There is exactly **one** place in the entire ~49k-line crate that builds a `reqwest::Client`: `src/reqwest_client.rs`. `src/redirect_client.rs` (AIStore 307-redirect wrapper) and `src/multi_endpoint.rs` both correctly *delegate* to the shared client rather than building their own — no independent copies of the h2c/window-tuning logic to drift out of sync.
- **GCS and Azure use their own SDKs' HTTP stacks entirely** (`gcloud_storage`, `azure_core`/`azure_storage_blob`), not `reqwest_client.rs`. This means the crate's H2-window env vars silently do nothing for GCS/Azure traffic — worth a documentation note (the env var docs don't currently scope this), but it's an architectural fact, not a bug, and out of scope for a narrow reqwest fix.

---

## 4. Proposed implementation plan

Ordered by risk and dependency. Each phase should land as its own PR (or small set of PRs), not combined, to keep the change bisectable if something regresses. **Do not implement Phase 4 before Phase 1** — Phase 4's fault-injection requirement is directly coupled to Phase 1's shared-segment-buffer design (§2.4). Phase 3 has no hard dependency on Phase 1 (corrected per §6.3 — the original claim that Phase 3 must also wait was overstated); it's *preferred* after Phase 1 only for cleaner throughput-regression benchmarking (see Phase 3's required tests), not because of an implementation dependency. Phase 2 and Phase 3 can proceed in either order relative to each other.

### Phase 1 — Low-risk, self-contained buffer-copy fixes

**Scope:**
- Implement Patch 3 as reviewed in §2.3 (S3 `concurrent_range_get_impl`), with the bounds-check-before-copy ordering confirmed and using `ByteStream::next()` rather than adding new `aws-smithy-types` feature flags.
- Apply the identical technique to `range_engine_generic.rs::download_with_ranges` (§3.3) — same fix, same safety argument, shared by Azure/GCS/file-store range downloads.
- Apply the identical technique to `s3_bytes.rs::ReaderMode::Range` (§3.3).
- Trivial: add `Vec::with_capacity(file_size)` in `file_store_direct.rs::try_read_file_direct` (§3.3).

**Risk**: low. Self-contained, no retry/error-handling semantics touched, `unsplit()`'s documented non-panicking fallback behavior (§2.3) means even an implementation slip degrades to "slower," not "broken."

**Required tests** (scope corrected per §6.1 — the three sites do not all use the same combinator):
- Existing range-GET correctness tests must continue to pass unmodified.
- New test, `concurrent_range_get_impl` (`s3_utils.rs`) only: this site uses `FuturesUnordered`, so completion order is genuinely non-deterministic — assert final bytes are correct and in the right order regardless of completion order.
- New test, `range_engine_generic.rs::download_with_ranges` and `s3_bytes.rs::ReaderMode::Range`: both use `.buffered(...)`, which is order-preserving by construction (built on `FuturesOrdered` — verified against `futures-util` source), so an out-of-order-completion test is not meaningful here. Instead require correctness tests around chunk-boundary edge cases and short/over-long reads.
- New test: short-read and over-long-read error paths still bail cleanly (no panic, no partial/corrupt result silently returned) at each site.
- While touching `range_engine_generic.rs`: remove or correct its source comment claiming `buffered()` does not guarantee output order — verified false against the `futures-util` implementation (§6.1).

### Phase 2 — Loader parallelism, with mandatory cancellation handling

**Scope:**
- Implement Patch 1 (§2.1) at all 3 sites in `python_aiml_api.rs`, **bundled with** cancellation handling modeled on `async_pool_dataloader.rs`'s existing `CancellationToken` usage — tracked `JoinHandle`s explicitly aborted on early iterator drop, or a `select!` against the token around each spawned fetch.
- Fix the identical bug in `async_pool_dataloader.rs` itself (§3.1) — same fix shape, same cancellation-handling requirement (it already has the `CancellationToken` plumbing; extend its use to cover the spawn-then-cancel path correctly).
- As a second wave (can be split into smaller PRs given the number of sites): apply the same spawn-and-cancel fix to the remaining §3.1 findings — `get_objects_parallel` pre-stat phase, `parquet_file_cache.rs` fetch-and-parse (called from `parquet_rg.rs` and `parquet_index.rs`), `checkpoint/reader.rs` shard reads, `azure_client.rs` multipart upload, `range_engine_generic.rs`'s Azure/GCS combinator (can be combined with Phase 1's fix to the same function), and the medium-confidence pre-stat/HEAD-only batch calls in `object_store.rs`/`s3_utils.rs`.
- Separately (does not require adding new `tokio::spawn` calls — this is a fix to *existing* spawn sites' cleanup behavior): retrofit the full-drain pattern (§3.2) from `object_store.rs:3646-3722` into `multipart.rs:556-627` and the three `s3_utils.rs` parallel get/put helpers, replacing early-return-on-first-error with drain-to-completion + accumulated-summary.

**Risk**: medium. The core technique (spawn + track + abort/cancel) is proven in this codebase already; the risk is in breadth (many call sites) and in getting the cancellation semantics right at each one, not in novelty.

**Required tests**:
- For each site converted: a test that drops the operation mid-flight (early iterator break, simulated cancellation) and asserts no dangling background work continues — e.g., via a call counter or a channel that would receive unwanted results if a "cancelled" fetch actually completed.
- A panic-inside-fetch test confirming the panic surfaces as a proper error to the caller (Python `PyRuntimeError` for the loader sites; `Result::Err` for the Rust-only sites), not a silent truncation — this locks in the bonus fix noted in §2.1.
- For the full-drain retrofit: a test with a mix of succeeding and failing tasks in the same batch, asserting *all* tasks are awaited to completion (no detached stragglers) and the returned summary correctly reflects every outcome, not just the first failure.

### Phase 3 — Reverse the HTTP/2 default: everything opt-in (Patch 2 final form) — design decided, ready to implement

**Scope** (see §2.2 for full rationale — this is a maintainer-directed design change; the *default* now flips to HTTP/1.1 for both schemes, and this is a breaking change called out in the changelog):

1. **Default HTTP/1.1 for both schemes.** The generic reqwest client (used for `https://` and for `http://` non-H2C fallback) is now built with `.http1_only()` unless an opt-in var enables H2 for the target scheme. This changes `https://` behavior: previously it always negotiated H2 via TLS ALPN; now it only does so if the user opts in.
2. **`S3DLIO_H2C=1`** — unchanged. Continues to opt in to h2c for `http://`.
3. **`S3DLIO_HTTPS_H2=1`** — new. Opt in to HTTP/2 (via TLS ALPN) for `https://`. When set, the generic client is *not* built with `.http1_only()` and window tuning is applied.
4. **`S3DLIO_ENABLE_HTTP2=1`** — new master switch. When set, implies both of the above. Precedence: H2 is enabled for scheme S iff (per-scheme var for S is set) OR (master switch is set).
5. **Window tuning**: extend `H2WindowConfig::from_env()` (currently gated to `h2c=true`) to also apply when `https_h2=true`. Only relevant when H2 is opted in on some scheme.
6. **Doc updates** in `reqwest_client.rs`, `constants.rs`, and `docs/Environment_Variables.md`. Retire the "https-ALPN-H2 is always on" claims and replace with "HTTP/1.1 by default; opt in via `S3DLIO_HTTPS_H2=1` or `S3DLIO_ENABLE_HTTP2=1`."
7. **Changelog entry** in `docs/Changelog.md` explicitly flagging the `https://` default flip as a breaking change from `v0.9.106`.

**Risk**: medium — the `https://` default behavior changes for every user of the crate. Mitigation: the change is trivially reversible for any user who wants the prior behavior (set `S3DLIO_HTTPS_H2=1`), and the change reflects real-world benchmarking (H2 is frequently slower than H1.1 for this workload class).

**Required tests**:
- Env-var parsing unit tests for the new `Http2Modes::from_env_values` helper, covering: unset defaults, `S3DLIO_H2C=1` sets `h2c` only, `S3DLIO_HTTPS_H2=1` sets `https_h2` only, `S3DLIO_ENABLE_HTTP2=1` sets both, and combinations (both scheme + master; contradictory values).
- Wire-level RED-then-GREEN test: hit a local TLS test server (self-signed cert; server offers both `h2` and `http/1.1` in ALPN) with the built reqwest client and assert `response.version()`:
  - Default (no opt-in vars) → HTTP/1.1. **RED against unmodified main** (which negotiates H2 via ALPN).
  - `Http2Modes { h2c: false, https_h2: true }` → HTTP/2. RED against unmodified main (env var doesn't exist).
  - `Http2Modes { h2c: false, https_h2: false }` with client hitting an http:// server → HTTP/1.1 (unchanged).
- H2 window-tuning applies on the https-H2 client when `https_h2=true` (mirror the existing h2c-only window tests).
- Precedence test: `S3DLIO_ENABLE_HTTP2=1` alone produces `Http2Modes { h2c: true, https_h2: true }`.
- Throughput regression check against Phase 1's range-GET path, if Phase 3 lands after Phase 1 (preferred order for cleaner benchmarking, not a hard dependency — §6.3).

### Phase 4 — Streaming connector + centralized retry (highest risk, do last)

**Scope:**
- Introduce the shared `retry_get_body()`-style helper (§3.4) first, as a small, independently-testable unit, before wiring it into the connector change.
- Implement the streaming-connector change (§2.4): `SdkBody::from_body_1_x` over `StreamBody`/`SyncStream`, applied at the 4 call sites, using the new shared retry helper instead of 4 independently hand-rolled loops.
- **Hard requirement, blocking**: the fault-injection test described in §2.4 — inject a body-read failure partway through a range-chunk's stream (after N of M bytes written into its `seg`), confirm the retry resets `written = 0` and refills from scratch, and assert the final reassembled object is byte-for-byte correct (not just correct length). This test must exist and pass before this phase merges, regardless of how confident a code review is otherwise.
- Do not combine this phase's PR with Phase 3's — keep them separately bisectable, since both touch `reqwest_client.rs` and interact with the same hot paths.

**Risk**: highest of the four. Everything about the mechanism itself is verified sound (§2.4), but this phase depends on Phase 1 already being in place (the shared-segment-buffer design that creates the retry/corruption interaction), and the one concrete risk identified is a silent-data-corruption class of bug — the standard for "done" here should be the fault-injection test passing, not code review alone.

**Required tests**:
- The fault-injection test above (mandatory, blocking).
- A test confirming retry idempotency for the 3 simple whole-object paths (`get_object`, `get_object_range`, `S3ObjectStore::get`) under injected mid-body failures — same technique as the range-chunk test, simpler because there's no shared buffer to worry about there, just confirming a clean fresh retry.
- A test confirming the retry budget (`max_retry_attempts()`) is actually respected (fails after N attempts, not retried indefinitely) for the new streaming-body-failure path specifically, separate from the existing SDK-level retry-budget tests (since this is a *new* retry path bypassing the SDK's own budget enforcement).

---

## 5. Quick-reference table

| # | File:line | Issue class | Severity/Confidence | Phase |
|---|---|---|---|---|
| 1.1 | `python_api/python_aiml_api.rs:239,372,~2213` | Single-task async concurrency | Confirmed (original report) | 2 |
| 1.2 | `reqwest_client.rs:424` (`build_reqwest_client_raw`) | No H2 window tuning on https | Confirmed (original report) | 3 |
| 1.3 | `s3_utils.rs:1154` (`concurrent_range_get_impl`) | Double-copy buffer assembly | Confirmed (original report) | 1 |
| 1.4 | `reqwest_client.rs:317` (connector) | Full-buffer before streaming | Confirmed (original report) | 4 |
| 3.1a | `data_loader/async_pool_dataloader.rs:234-315` | Single-task async concurrency | High | 2 |
| 3.1b | `s3_utils.rs:1483-1504` (pre-stat) | Single-task async concurrency | High | 2 |
| 3.1c | `data_loader/parquet_file_cache.rs:110` (+2 callers) | Single-task async concurrency (CPU-bound) | High | 2 |
| 3.1d | `checkpoint/reader.rs:203-244` | Single-task async concurrency | High | 2 |
| 3.1e | `azure_client.rs:436-472` | Single-task async concurrency (no spawn at all) | High | 2 |
| 3.1f | `range_engine_generic.rs:305-369` | Single-task async concurrency | High (low current exposure — disabled by default) | 2 |
| 3.1g | `object_store.rs:565-583`, `s3_utils.rs:1364-1371` | Single-task async concurrency (HEAD-only) | Medium | 2 |
| 3.1h | `data_loader/s3_bytes.rs:97-110` | Single-task async concurrency | Medium | 2 |
| 3.2 | `multipart.rs:556-627`, `s3_utils.rs:1508-1520,1554-1567,1843-1856` | Drop-doesn't-abort on existing spawns | High | 2 |
| 3.3a | `range_engine_generic.rs:371-389` | Double-copy buffer assembly | High | 1 |
| 3.3b | `data_loader/s3_bytes.rs:99-114` | Double-copy buffer assembly | Medium-high | 1 |
| 3.3c | `file_store_direct.rs:819-869` | Missing capacity hint | Medium (low impact) | 1 |
| 3.4 | Crate-wide | No shared retry helper (3 divergent shapes, Patch 4 would add a 4th) | High (design smell, not a bug) | 4 |
| 3.4a | `google_gcs_client.rs:404-483` | Retry with no backoff | Medium | Not scheduled — flag for separate GCS-focused pass |

---

## 6. Revision history

An independent [adversarial review](./PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md) (2026-07-07) re-verified this audit against current source and confirmed the four primary findings, the Patch 1 cancellation objection, and the Patch 4 fault-injection requirement. It also identified five corrections, all incorporated into this document:

1. **`buffered` vs `buffer_unordered` conflation** — only `concurrent_range_get_impl` (`s3_utils.rs`) uses an unordered combinator (`FuturesUnordered`); `range_engine_generic.rs::download_with_ranges` and `s3_bytes.rs::ReaderMode::Range` use `.buffered(...)`, which is order-preserving. Fixed in Phase 1's required-tests scope (§4).
2. **Patch 2's handshake-cost claim was overstated** — reqwest's connection pool (`DEFAULT_POOL_MAX_IDLE_PER_HOST = usize::MAX`) allows HTTP/1.1 keep-alive reuse, so forced-H1.1 does not mean "one handshake per request." Softened in §2.2; superseded in substance by the maintainer's decision to ship HTTP/2 opt-out controls regardless of this tradeoff.
3. **Phase 3/4 dependency overstated** — only Phase 4 has a hard dependency on Phase 1 (the fault-injection test needs Phase 1's shared-segment design in place). Phase 3 is independent; "after Phase 1" is a benchmarking preference, not a blocker. Fixed in §4's plan introduction and Phase 3's required tests.
4. **Closing summary understated live defects** — reworded to distinguish "no confirmed active data-corruption bug" (still true) from "no currently-live defects at all" (false — the throughput ceiling and `first_byte_time` distortion are present-tense). Fixed in the closing summary.
5. **Wrong citation for RangeEngine's default-disabled state** — the authoritative source is the Azure/GCS/file-store config defaults (`object_store.rs`, `file_store.rs`, `file_store_direct.rs`), not `constants.rs`. Fixed in §3.1's high-confidence table.

Separately, incorporated in this revision: a maintainer design decision (2026-07-07) that HTTP/2 must be trivially disableable for both `http://` and `https://` traffic, based on operational experience that H2 is frequently slower than H1.1 for this workload class. This resolves Phase 3's original "needs design sign-off" status and expands its scope beyond window-tuning-only to include the `S3DLIO_HTTPS_H2` and `S3DLIO_DISABLE_HTTP2` opt-out controls described in §2.2 and §4.

---

## Summary for the reviewing developer

Corrected per §6.4: no confirmed active silent-data-corruption bug was found in current code, but several currently-live performance and observability defects were confirmed — the client throughput ceiling (§1.1/§1.2) and the `first_byte_time` metric distortion (§1.4) are present-tense behavior in `main` today, not merely theoretical risk. What this audit found is: (a) two of the four patches originally proposed in #148 needed rework before they're safe to merge — Patch 2's blunt-instrument approach has been reworked into an opt-in, maintainer-directed set of HTTP/2 controls rather than a default-changing blanket flip (§2.2); Patch 4 has a specific, credible data-corruption interaction with Patch 3 that needs a fault-injection test, not just review — and (b) the same two bug *classes* (missing task-level parallelism, avoidable double-copies) recur in roughly a dozen more places across the crate, including the core Rust-level DataLoader itself. None of this needs to land in one giant change — the phased plan above is designed so each phase is independently reviewable, testable, and revertible, with the riskiest work (Phase 4) both last and most heavily gated by a specific required test.
