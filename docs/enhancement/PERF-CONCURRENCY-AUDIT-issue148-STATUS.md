# Issue #148 — Where we are, what's next

**Date paused**: 2026-07-07
**Branch**: `perf/148-phase3-http2-optin` (13 commits ahead of `main`, plus 2 audit-docs + parent-CLAUDE commits that already merged into `main`)
**Working tree**: clean, nothing uncommitted
**Push state**: **NOTHING PUSHED FROM THE BRANCH**. All 13 branch commits are local-only. The two commits already on `main` (`8a42ebb`, `e4b9ae4`) are also local-only — `main` itself is ahead of `origin/main` and has not been pushed. Per Prime Directive #1, do not push anything without an explicit instruction.

Detailed audit lives at [`PERF-CONCURRENCY-AUDIT-issue148.md`](PERF-CONCURRENCY-AUDIT-issue148.md); adversarial review at [`PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md`](PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md). This doc is the operational handoff: what's on the branch, what proved GREEN, and exactly where to pick up.

---

## 30-second summary

| Phase | Scope | Items | Status |
|---|---|---|---|
| **1** | Range-assembly double-copy + capacity hint | 4 / 4 | ✅ **DONE** — peak memory 2.28× → 1.03× |
| **3** | HTTP/2 opt-in reversal + window tuning for https | 1 / 1 | ✅ **DONE** — default now HTTP/1.1 on both schemes |
| **2** | Loader task-level parallelism + drop-doesn't-abort | 0 / 10 | ⏳ **NEXT UP** |
| **4** | Streaming connector + centralized retry | 0 / 2 | ⏳ Blocked on Phase 2 |
| Deferred | GCS retry-with-no-backoff (finding 3.4a) | 0 / 1 | Separate GCS-focused pass |

**Overall: 5 in-scope + 1 deferred out of 17 in-scope + 1 deferred.** Version bumped `0.9.106` → **`0.9.108`** locally in `Cargo.toml`, `pyproject.toml`, `docs/Changelog.md`, and `docs/Environment_Variables.md`.

---

## What's committed

### Already on local `main` (2 commits, also local-only — `main` is ahead of `origin/main`)

```
e4b9ae4 docs: add project-level CLAUDE.md with RED/GREEN policy + s3dlio-specific rules
8a42ebb docs(enhancement): add issue #148 performance/concurrency audit + adversarial review
```

These were committed before the branch was cut, so they show as "on main" but they too are unpushed — `origin/main` is still at `38fe812` (the v0.9.106 baseline).

### On branch `perf/148-phase3-http2-optin` (13 commits, oldest first)

```
01898ca test(148/phase1): add RED peak-memory test for range assembly double-copy
d4632a8 fix(148/phase1): pre-allocate range-assembly buffers to halve peak memory
b3dd699 chore(148/phase1): eliminate warnings by scope-fixing shared test module
1fe9067 docs(148/phase3): reverse Phase 3 direction — HTTP/2 is now opt-in for both schemes
61f0731 test(148/phase3): add RED wire-level test asserting default https is HTTP/1.1
1451c4f fix(148/phase3): reverse HTTP/2 default — HTTP/1.1 unless explicitly opted in
5294959 docs(148/phase3): env-var reference + changelog for HTTP/2 opt-in reversal
e47cce2 release: bump version 0.9.106 → 0.9.108 for the issue #148 Phase 1 + Phase 3 changes
e8f4774 release: bump s3dlio/pyproject.toml to 0.9.108 + harden CLAUDE.md scope rule
44510e8 chore: cargo update + CLAUDE.md — document uv and ./build_pyo3.sh
e5f3bec docs(148): mark Phase 1 + Phase 3 done in the audit progress table
08bcda6 chore: relax AWS SDK pins, keep Azure at 0.31/0.8
6ab7547 docs(148): add STATUS handoff document for resuming after the pause  ← this doc
```

Phase 1 was originally on a sub-branch `perf/148-phase1-buffer-copies`, then Phase 3 branched off it — Phase 1's commits show up in Phase 3's history as ancestors. The old parent branches `perf/concurrency-audit-issue148` and `perf/148-phase1-buffer-copies` still exist locally and can be deleted (`git branch -d <name>`) if you don't need them for reference.

---

## What's proven GREEN

Run this to reproduce the full gate at any time:

```bash
cd /home/eval/Documents/Code/s3dlio
cargo test --lib                                        # 337/337 pass
cargo test --test test_phase1_zero_copy_assembly        # 4/4 pass (peak overhead 1.03×)
cargo test --test test_file_range_engine \
           --test test_directio_range_engine \
           --test test_range_engine_defaults \
           --test test_range_engine_cache_integration \
           --test test_file_store \
           --test test_direct_io \
           --test test_buffer_pool_directio             # 48/48 pass
cargo fmt --all -- --check                              # clean
cargo clippy --lib --tests --no-deps -- -D warnings     # clean, zero warnings
```

Additionally, the reqwest_client wire-level tests (inside `cargo test --lib`) cover:
- `phase3_default_https_client_negotiates_http1` — default https client speaks HTTP/1.1 (RED gate for Phase 3).
- `phase3_https_h2_opt_in_negotiates_h2` — `Http2Modes { https_h2: true }` → HTTP/2 via ALPN.
- `phase3_https_h2_opt_in_falls_back_when_server_only_offers_http1` — opt-in is permission, not force.
- 9 `test_http2_modes_*` — `Http2Modes::from_env_values` precedence rules across all env-var combinations.

---

## Phase 2 — Next up (pick up here)

Phase 2 is the biggest of the remaining phases: **10 call sites** across the loader, checkpoint reader, parquet cache, and both S3/Azure batch operations. Two distinct bug classes to fix:

### Bug class A — "single-task async concurrency" (findings 1.1, 3.1a–h)

Places where `buffer_unordered(N)` or `.buffered(N)` drives many futures inside **one** async task. Tokio can only run that one task on one worker thread, so no matter how deep the buffer, all request-driving + body-accumulation work funnels through ~one core's capacity. Fix: `tokio::spawn` each fetch so the runtime can multi-thread them.

**Sites** (in the audit's order — recommend implementing in this order, since each is a variation on the same technique and it's easier to review sequentially):

| # | File:line | Notes |
|---|---|---|
| 1.1 | `src/python_api/python_aiml_api.rs:239, 372, ~2213` | Three call sites in the Python bindings — the original issue #148 report. |
| 3.1a | `src/data_loader/async_pool_dataloader.rs:234-315` | Core Rust-level DataLoader; the Python bindings sit on top. **Same bug in the layer below** the Python fix. |
| 3.1b | `src/s3_utils.rs:1483-1504` | `get_objects_parallel` pre-stat phase. Only 6 lines away, the GET phase *does* spawn — internal inconsistency is strong evidence it's an oversight. |
| 3.1c | `src/data_loader/parquet_file_cache.rs:110` (+2 callers at `parquet_rg.rs:567-602`, `parquet_index.rs:316-329`) | CPU-bound Thrift metadata parsing — strongest case for spawning since it's CPU, not just I/O. |
| 3.1d | `src/checkpoint/reader.rs:203-244` | `read_all_shards_concurrent` distributed-checkpoint reads. |
| 3.1e | `src/azure_client.rs:436-472` | Azure multipart upload `stage_block` fan-out. No spawn at all. |
| 3.1f | `src/range_engine_generic.rs:305-369` | Azure/GCS shared range engine (used via `RangeEngine::download_with_ranges`, disabled by default in backend configs). Lower urgency but same technique. |
| 3.1g | `src/object_store.rs:565-583` + `src/s3_utils.rs:1364-1371` | HEAD-only stat batch calls. Lower per-item cost (no body), but at high concurrency the request-signing work still bottlenecks one core. |
| 3.1h | `src/data_loader/s3_bytes.rs:97-110` | `ReaderMode::Range`'s loop. Note this file *also* has finding 3.3b (double-copy) which Phase 1 already fixed — the file has both bug classes. |

**Critical**: converting bare futures to `tokio::spawn` changes cancellation semantics. Dropping a bare unspawned future cancels it; dropping a `JoinHandle` **detaches** (task keeps running). The audit §2.1 has a detailed "cancellation must be fixed in the same change" discussion. The precedent already in this codebase is `CancellationToken`-based cooperative cancellation — see `src/data_loader/async_pool_dataloader.rs:194, 201, 228, 244-288` and `src/data_loader/options.rs:181` for how it's wired.

### Bug class B — "drop doesn't abort" on *existing* spawns (finding 3.2)

Distinct from A: these places already spawn correctly, but the drain loop short-circuits on the first error (`res??`), dropping the remaining `Vec<JoinHandle>` before other tasks finish — which detaches them, not aborts them.

**Sites** (all can be fixed with the same "full-drain + summary" pattern):

| File:line | What |
|---|---|
| `src/multipart.rs:556-627` | S3 multipart part upload |
| `src/s3_utils.rs:1508-1520` | `get_objects_parallel` GET phase |
| `src/s3_utils.rs:1554-1567` | `*_with_progress` variant |
| `src/s3_utils.rs:1843-1856` | `put_objects_parallel_with_progress` |

The audit §3.2 shows the **known-good template** already in this codebase: `src/object_store.rs:3646-3722` (`generic_upload_files`, `generic_download_objects`) — full-drain over `FuturesUnordered<JoinHandle<...>>`, accumulate succeeded/failed into a summary struct, never bail on first error. Also `src/prefetch.rs:38-54`. Retrofit those 4 sites to match.

### Phase 2 RED/GREEN test plan

Per the CLAUDE.md RED/GREEN rule, each fix ships with a test that goes RED against pre-fix code:

- **For class A (single-task → spawn)**: hardest to test at the throughput level — the difference is in parallelism, not correctness. Two options:
  1. **Behavioral**: spawn N slow futures via a mock closure; measure wall time. Sequential polling ~= N × delay; parallel spawning ~= max(delay). Wall-time bounded assertion — RED against pre-fix (single-task) code, GREEN after spawn. **Fragile** (CI variability).
  2. **Cancellation-drop test**: start the iterator, drop it early, assert no dangling background work continues (via a counter or a channel that would receive unwanted results if a "cancelled" fetch actually completed). This is real RED against unmodified code today (bare futures cancel for free, so it's a no-op) but real RED against a **naive spawn** without cancellation handling (leaks). Doubles as the safety net for the spawn conversion.
  3. **Panic-inside-fetch test**: assert a panic in a fetch surfaces as a proper error to the caller (Python `PyRuntimeError` for the loader sites; `Result::Err` for the Rust-only sites) rather than silently truncating the iterator — locks in the bonus fix noted in §2.1.

- **For class B (full-drain retrofit)**: test that a mix of succeeding and failing tasks all get awaited (not just the first failure), and the returned summary reflects every outcome. RED against short-circuit code (fails: only the first outcome is captured), GREEN after retrofit.

The audit's §4 "Phase 2 required tests" section has the full authoritative list — cross-check it before finalizing.

### Suggested branch structure for Phase 2

Given the size (10 sites × 2 bug classes), don't do one giant commit. Suggested:

1. Branch `perf/148-phase2-loader-parallelism` off current `perf/148-phase3-http2-optin`.
2. **Sub-commit A**: bug class A at site 1.1 only (Python loader — the original issue #148 report) + cancellation handling. RED test + GREEN.
3. **Sub-commit B**: bug class A at site 3.1a (`async_pool_dataloader.rs` — the core Rust layer). Same technique, different call site.
4. **Sub-commits C, D, E, F, G, H**: remaining §3.1 sites in the audit's order. Group logically — the parquet-cache sites (3.1c) are three call sites for one shared function, so they belong together.
5. **Sub-commit I**: bug class B — the four `drop-doesn't-abort` retrofits at once (they share the exact same pattern; one commit is fine).

Each sub-commit follows RED-then-GREEN with the new failing test committed alongside the fix. See how Phase 1 was structured (`01898ca` RED → `d4632a8` GREEN) as the template.

---

## Phase 4 — After Phase 2

Blocked on Phase 2 landing first. Two items:

- **Finding 1.4** (`src/reqwest_client.rs:317` connector) — the smithy HTTP connector currently does `resp.bytes().await` and hands a fully-buffered body to the SDK. Fix: stream via `SdkBody::from_body_1_x` over `StreamBody`/`sync_wrapper::SyncStream` so bytes flow through as they arrive.
- **Finding 3.4** (crate-wide retry) — three independently hand-rolled retry-loop shapes (`src/s3_client.rs`, `src/python_api/python_core_api.rs`, `src/google_gcs_client.rs`) and Patch 4 would add a fourth. Introduce a single shared `retry_get_body()`-shaped helper as part of the same change.

**Mandatory before merge**: the fault-injection test described in audit §2.4. Inject a body-read failure partway through a range chunk's stream (after N of M bytes written into its segment), confirm the retry resets `written = 0` and refills from scratch, assert the reassembled object is byte-for-byte correct — **not just correct length**. This is the silent-data-corruption gate.

Audit §4 "Phase 4" and §2.4 have the full details.

---

## Deferred: finding 3.4a — GCS retry with no backoff

`src/google_gcs_client.rs:404-483` — the full-read retry loop uses hardcoded constants and **no delay between attempts**. Reads are idempotent so it's not unsafe, but three immediate retries on `RESOURCE_EXHAUSTED`/`UNAVAILABLE` could tighten a retry storm against a struggling backend. Flagged for a separate GCS-focused pass. Not scheduled.

---

## Related deferred items

### Azure SDK 0.31 → 1.0 upgrade

`azure_core`, `azure_identity`, `azure_storage_blob` all reached 1.0 in early 2026 with non-trivial breaking changes. Cargo.toml keeps them at exact `=0.31.0` / `=0.8.0` pins with a comment recording the deferral. Task for another day — separate from anything in the audit.

### Downstream pyproject.toml pins (out of scope — DO NOT touch without explicit instruction)

- `DLIO_local_changes/pyproject.toml` — still pins `s3dlio>=0.9.106` on `main` (fast-forwarded during this session, no local edits made).
- `mlp-storage/pyproject.toml` — has WIP from a prior session bumping to `s3dlio>=0.9.104` + local-dev wheel path. **Not touched.**

The `feedback_cargo_pyproject_version_sync` memory has been updated to make it explicit that this rule is a **reminder to ask**, not pre-authorization to edit downstream repos. When you want the downstream pins bumped to `0.9.108`, tell me explicitly which repos and which branches.

---

## How to resume

```bash
cd /home/eval/Documents/Code/s3dlio
git status
git log --oneline main..HEAD    # should show 13 commits ending in 6ab7547
git branch --show-current       # perf/148-phase3-http2-optin

# Sanity check: re-run the gate
cargo test --lib                 # expect 337/337
cargo test --test test_phase1_zero_copy_assembly    # expect 4/4
cargo fmt --all -- --check
cargo clippy --lib --tests --no-deps -- -D warnings

# Then to start Phase 2:
git checkout -b perf/148-phase2-loader-parallelism
# Open src/python_api/python_aiml_api.rs at ~line 239 (PyBytesAsyncDataLoader::__iter__)
# and start with the RED test for site 1.1 per the plan above.
```

Read this file, [`PERF-CONCURRENCY-AUDIT-issue148.md`](PERF-CONCURRENCY-AUDIT-issue148.md) (especially §2.1 for cancellation, §3.1 for the site list, §3.2 for the drop-doesn't-abort template, §4 for the phased plan and required tests), and [`PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md`](PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md) (§6 corrections, especially §6.1 on `buffered` vs `buffer_unordered`) before starting the first change.

---

## Files most useful for context when resuming

| File | Why |
|---|---|
| `docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md` | The plan. §5 has the progress table; §2 has per-patch design; §3 has the site list; §4 has the phased implementation plan; §6 has the correction history. |
| `docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md` | Independent review that surfaced 5 corrections to the audit; §6.1 (combinator semantics) is important for Phase 2 test design. |
| `CLAUDE.md` (this repo) | Standing rules: RED/GREEN, zero warnings, no-underscore-hack, uv + `./build_pyo3.sh` for Python, no pushes without explicit instruction, version-bump scope. |
| `../CLAUDE.md` (parent) | Prime Directives #1 (never push without instruction) and #2 (never touch a repo not named in the current instruction). |
| `src/reqwest_client.rs` | Home of Phase 3's Http2Modes struct + wire tests (search `phase3_`). Reference for how end-to-end wire tests are structured. |
| `src/range_engine_generic.rs`, `src/s3_utils.rs`, `src/data_loader/s3_bytes.rs`, `src/file_store_direct.rs` | The four sites Phase 1 fixed. Reference for the pre-allocate-and-copy-into-offset technique. |
| `tests/test_phase1_zero_copy_assembly.rs` | Reference for the peak-tracking global allocator pattern (may be reusable for Phase 2 if a memory-based assertion is needed). |
