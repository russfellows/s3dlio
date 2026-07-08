# Issue #148 — Where we are, what's next

**Last updated**: 2026-07-07 (after Phase 2 sites 3.1f + 3.1g + 3.1h landed — Phase 2 bug class A COMPLETE)
**Branch**: `perf/148-phase2-loader-parallelism` at `82a638f`, 28 commits ahead of local `main`
**Working tree**: clean (impact-analysis doc untracked, not staged)
**Push state**: **NOTHING PUSHED**. The branch is local-only. Local `main` is also 2 commits ahead of `origin/main` (the audit docs + parent CLAUDE.md — `8a42ebb`, `e4b9ae4`) and unpushed. Per Prime Directive #1, do not push anything without an explicit instruction.

Detailed audit lives at [`PERF-CONCURRENCY-AUDIT-issue148.md`](PERF-CONCURRENCY-AUDIT-issue148.md); adversarial review at [`PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md`](PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md). This doc is the operational handoff: what's on the branch, what proved GREEN, and exactly where to pick up.

---

## 30-second summary

| Phase | Scope | Items | Status |
|---|---|---|---|
| **1** | Range-assembly double-copy + capacity hint | **4 / 4** | ✅ **DONE** — peak memory 2.28× → 1.03× total_size |
| **3** | HTTP/2 opt-in reversal + window tuning for https | **1 / 1** | ✅ **DONE** — default now HTTP/1.1 on both schemes |
| **2 — bug class A** (spawn + cancel) | Task-level parallelism (findings 1.1, 3.1a–h) | **9 / 9** | ✅ **DONE** — all 9 sites converted to `tokio::spawn` + DropCancel + select! |
| **2 — bug class B** (drop doesn't abort) | Full-drain-first-then-error (finding 3.2) | **1 / 1** | ✅ **DONE** — all 4 short-circuit sites retrofitted |
| **4** | Streaming connector + centralized retry (findings 1.4, 3.4) | **0 / 2** | ⏳ Not started; hard-depends on Phase 1 (already done) |
| Deferred | GCS retry-with-no-backoff (finding 3.4a) | 0 / 1 | Separate GCS-focused pass; unscheduled |

**Overall: 15 in-scope + 1 deferred out of 17 in-scope + 1 deferred.** Phase 2 is fully complete. Only Phase 4 (2 items) remains. Version bumped `0.9.106` → **`0.9.108`** locally in `Cargo.toml`, `pyproject.toml`, `docs/Changelog.md`, and `docs/Environment_Variables.md`.

---

## What's committed

### Already on local `main` (2 commits, also unpushed — `main` is ahead of `origin/main`)

```
e4b9ae4 docs: add project-level CLAUDE.md with RED/GREEN policy + s3dlio-specific rules
8a42ebb docs(enhancement): add issue #148 performance/concurrency audit + adversarial review
```

`origin/main` is still at `38fe812` (the v0.9.106 baseline).

### On branch `perf/148-phase2-loader-parallelism` (28 commits, oldest first)

Phase 1 (peak-memory range-assembly fix):
```
01898ca test(148/phase1): add RED peak-memory test for range assembly double-copy
d4632a8 fix(148/phase1): pre-allocate range-assembly buffers to halve peak memory
b3dd699 chore(148/phase1): eliminate warnings by scope-fixing shared test module
```

Phase 3 (HTTP/2 opt-in reversal):
```
1fe9067 docs(148/phase3): reverse Phase 3 direction — HTTP/2 is now opt-in for both schemes
61f0731 test(148/phase3): add RED wire-level test asserting default https is HTTP/1.1
1451c4f fix(148/phase3): reverse HTTP/2 default — HTTP/1.1 unless explicitly opted in
5294959 docs(148/phase3): env-var reference + changelog for HTTP/2 opt-in reversal
```

Release + tooling:
```
e47cce2 release: bump version 0.9.106 → 0.9.108 for the issue #148 Phase 1 + Phase 3 changes
e8f4774 release: bump s3dlio/pyproject.toml to 0.9.108 + harden CLAUDE.md scope rule
44510e8 chore: cargo update + CLAUDE.md — document uv and ./build_pyo3.sh
e5f3bec docs(148): mark Phase 1 + Phase 3 done in the audit progress table
08bcda6 chore: relax AWS SDK pins, keep Azure at 0.31/0.8
6ab7547 docs(148): add STATUS handoff document for resuming after the pause
20d4887 docs(148): correct STATUS commit listing — separate branch commits from main commits
```

Phase 2 (task-level parallelism + drain-first-then-error):
```
63651f4 test(148/phase2): add RED tests for async_pool_dataloader.rs task-level parallelism
bfd9527 fix(148/phase2): spawn each fetch as its own tokio task in async_pool_dataloader  ← 3.1a
1070a93 fix(148/phase2): apply task-level parallelism to python_aiml_api.rs (site 1.1)
053583f fix(148/phase2): drain-first-then-error at 4 short-circuit-on-error sites (site 3.2)
84ce783 docs(148): refresh STATUS + audit §5 to reflect Phase 2 progress (3.1a + 1.1 + 3.2)
6c53f12 fix(148/phase2): spawn stat tasks in get_objects_parallel pre-stat phase (site 3.1b)
262014d fix(148/phase2): spawn parquet footer-fetch tasks in 3 call sites (site 3.1c)
d2bc6cd fix(148/phase2): spawn per-shard reads in checkpoint::Reader (site 3.1d)
ec7c098 fix(148/phase2): spawn stage_block in Azure upload_multipart_stream (site 3.1e)
a8a2a9a docs(148): refresh STATUS + audit §5 for Phase 2 sites 3.1b-3.1e
4c07d1b docs(148): mark sites 3.1b-3.1e ✅ done in audit §5
0f542ec fix(148/phase2): spawn range fetches in range_engine_generic (site 3.1f)
c157013 fix(148/phase2): spawn stat_object_many_async + document trait limit (site 3.1g)
82a638f fix(148/phase2): spawn range GETs in ReaderMode::Range (site 3.1h)  ← current tip
```

The branch history has ancestors on `perf/148-phase1-buffer-copies` and `perf/148-phase3-http2-optin`. Those old branches still exist locally; if you don't need them for reference they can be deleted with `git branch -d <name>` once this rolls up.

---

## What's proven GREEN

Run this to reproduce the full gate at any time (Rust side):

```bash
cd /home/eval/Documents/Code/s3dlio
cargo test --lib                                              # 337/337 pass
cargo test --test test_phase1_zero_copy_assembly              # 4/4 pass  (peak overhead 1.03×)
cargo test --test test_phase2_loader_parallelism              # 4/4 pass  (site 3.1a)
cargo test --test test_phase2_drain_first_err                 # 2/2 pass  (site 3.2 patterns)
cargo test --test test_async_pool_dataloader                  # 6/6 pass  (integration)
cargo test --test test_file_range_engine \
           --test test_directio_range_engine \
           --test test_range_engine_defaults \
           --test test_range_engine_cache_integration \
           --test test_file_store \
           --test test_direct_io \
           --test test_buffer_pool_directio                   # 48/48 pass
cargo fmt --all -- --check                                    # clean
cargo clippy --lib --tests --no-deps -- -D warnings           # clean, zero warnings
```

Python side (requires the `0.9.108` wheel installed in the venv — see below):

```bash
source .venv/bin/activate
./build_pyo3.sh                                                            # cp312 + cp313 wheels
uv pip install --reinstall target/wheels/s3dlio-0.9.108-cp313-*.whl
uv run python tests/test_phase2_python_loader_parallelism.py               # 4/4 subtests pass
uv run python tests/test_loader_return_type.py                             # OK
uv run python tests/test_zero_copy.py                                      # 12 pass
uv run python tests/test_python_oplog.py                                   # OK
uv run python tests/test_s3dlio_datagen.py                                 # OK
```

**Key wire/measurement observations captured across the RED→GREEN transitions:**

Phase 1 (peak memory during range assembly, 32 MiB / 1 MiB chunks / 4 concurrent):
- Pre-fix: 2.28× total_size (~76 MiB overhead — all chunks retained in `parts` while `assembled` grew)
- Post-fix: 1.03× total_size (~1 MiB overhead)

Phase 3 (default https protocol via ALPN wire test):
- Pre-fix: `HTTP/2.0` (server offered `h2` + `http/1.1`)
- Post-fix: `HTTP/1.1` unless `S3DLIO_HTTPS_H2=1` or `S3DLIO_ENABLE_HTTP2=1` is set

Phase 2 site 3.1a (parallelism, 4 × 100ms sync-sleep, worker_threads=4):
- Pre-fix: 401ms (serialized on one task)
- Post-fix: ~150ms (parallel across workers)

Phase 2 site 3.1a (external cancel latency):
- Pre-fix: 2.001s (waited for at least one in-flight fetch to complete)
- Post-fix: <500ms (`select!` drops the fetch immediately)

Phase 2 site 3.1a (panic-in-fetch):
- Pre-fix: silent truncation (Python side sees `StopIteration`)
- Post-fix: `Err(DatasetError::Backend(...))` surfaced to caller

Phase 2 site 1.1 (Python early-drop iterator latency, 200 files, prefetch=32):
- Post-fix: 43–65ms drop return time; regression guard, would trip a naive `tokio::spawn` without cancel wiring

Phase 2 site 3.2 (drain vs short-circuit pattern, synthetic scenario):
- Short-circuit (RED): elapsed <50ms, counter <3 (leaks in-flight tasks)
- Drain (GREEN): elapsed ≥250ms (waited for 300ms slowest task), counter ==3 (all tasks completed)

---

## Phase 2 bug class A — COMPLETE

All 9 sites have been converted:

| # | File | Commit |
|---|---|---|
| 3.1a | `src/data_loader/async_pool_dataloader.rs::run_async_pool_worker` | `bfd9527` |
| 1.1 | `src/python_api/python_aiml_api.rs` (3 iterator sites) | `1070a93` |
| 3.1b | `src/s3_utils.rs::get_objects_parallel` pre-stat | `6c53f12` |
| 3.1c | `src/data_loader/parquet_rg.rs` + `parquet_index.rs` (3 sites) | `262014d` |
| 3.1d | `src/checkpoint/reader.rs` (2 sites) | `d2bc6cd` |
| 3.1e | `src/azure_client.rs::upload_multipart_stream` | `ec7c098` |
| 3.1f | `src/range_engine_generic.rs::download_with_ranges` | `0f542ec` |
| 3.1g | `src/s3_utils.rs::stat_object_many_async` (+ trait-default doc note) | `c157013` |
| 3.1h | `src/data_loader/s3_bytes.rs::ReaderMode::Range` | `82a638f` |

Common pattern across every site: `tokio::spawn` per unit of work + `DropCancel` guard on the enclosing task + `tokio::select!` inside each spawn against a `CancellationToken` so mid-flight fetches bail on error/drop instead of running to completion. Panics surface as errors instead of silently truncating results.

Site-specific adaptations:
- Sites 3.1f (range_engine_generic) and 3.1h (s3_bytes) use `FuturesOrdered` + a bounded prime-and-refill pool to preserve the running-write-offset assembly (short-read semantics) and cap peak memory. Without the pool cap, spawned JoinHandles would each hold a chunk-sized Bytes after completion, blowing peak memory to `n_parts * part_size` and regressing the Phase 1 guarantee.
- Site 3.1g documents (rather than fixes) the `ObjectStore::pre_stat_objects` trait default. That default takes `&self` through `dyn ObjectStore`, so `tokio::spawn` (which needs `Arc<Self>: 'static`) is not possible without a breaking trait change. The comment points backends at `stat_object_many_async` as the pattern to copy if they need N-core stat parallelism.

**Only Phase 4 remains** — see below.

---

## Phase 4 — After Phase 2

Blocked on Phase 2 finishing (not on any specific site, but on the overall Phase 2 wrap-up per the audit's dependency plan). Two items:

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
git log --oneline main..HEAD    # should show 28 commits ending in 82a638f
git branch --show-current       # perf/148-phase2-loader-parallelism

# Sanity check: re-run the gate (see "What's proven GREEN" above)
cargo test --lib
cargo test --test test_phase2_loader_parallelism
cargo test --test test_phase2_drain_first_err
cargo test --test test_phase2_join_all_vs_spawn
cargo test --test test_phase1_zero_copy_assembly
cargo fmt --all -- --check
cargo clippy --lib --tests --no-deps -- -D warnings

# Phase 2 is complete. Next: Phase 4 (streaming connector + centralized
# retry — see below). Fault-injection test is mandatory at merge.
```

Read this file, [`PERF-CONCURRENCY-AUDIT-issue148.md`](PERF-CONCURRENCY-AUDIT-issue148.md) (especially §2.1 for cancellation, §3.1 for the site list, §3.2 for the drop-doesn't-abort template — now done, §4 for the phased plan), and [`PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md`](PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md) (§6 corrections) before starting the first change.

---

## Files most useful for context when resuming

| File | Why |
|---|---|
| `docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md` | The plan. §5 has the progress table; §2 has per-patch design; §3 has the site list; §4 has the phased implementation plan; §6 has the correction history. |
| `docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148-adversarial-review.md` | Independent review that surfaced 5 corrections to the audit; §6.1 (combinator semantics) is important for the remaining Phase 2 sites' test design. |
| `CLAUDE.md` (this repo) | Standing rules: RED/GREEN, zero warnings, no-underscore-hack, uv + `./build_pyo3.sh` for Python, no pushes without explicit instruction, version-bump scope. |
| `../CLAUDE.md` (parent) | Prime Directives #1 (never push without instruction) and #2 (never touch a repo not named in the current instruction). |
| `src/data_loader/parallel_fetch.rs` | Home of `DropCancel`. Shared type used by every Phase 2 site so far. |
| `src/data_loader/async_pool_dataloader.rs` | Reference implementation of the Phase 2 spawn + CancellationToken + select! pattern (site 3.1a). |
| `src/python_api/python_aiml_api.rs` | Same pattern applied at three PyO3 iterator sites (site 1.1). |
| `src/s3_utils.rs`, `src/multipart.rs` | The four drain-first-then-error sites (finding 3.2, all done). |
| `src/reqwest_client.rs` | Home of Phase 3's `Http2Modes` struct + wire tests (search `phase3_`). Reference for end-to-end wire-level test structure. |
| `tests/test_phase1_zero_copy_assembly.rs` | Reference for the peak-tracking global allocator pattern. |
| `tests/test_phase2_loader_parallelism.rs` | Reference for how site 3.1a's tests were structured (mock ObjectStore + tokio::test with pinned worker_threads). |
| `tests/test_phase2_drain_first_err.rs` | Side-by-side RED vs GREEN pattern documentation for the drop-doesn't-abort fix. |
| `tests/test_phase2_python_loader_parallelism.py` | End-to-end Python-surface test for site 1.1. |
