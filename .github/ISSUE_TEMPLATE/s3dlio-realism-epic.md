---
name: "Realism Epic: Loader knobs, metrics, trace (s3dlio)"
about: Track the work to add realism controls, metrics, caching policies, and trace record/replay to s3dlio.
title: "[Realism] s3dlio: loader options, metrics, trace & cache policy"
labels: enhancement, performance, io, epic
assignees: ""
---

## Summary
Add the loader controls, metrics, cache policies, and trace record/replay needed for **realistic AI/ML storage workloads** and for parity with dl-driver profiles.

## Goals
- Framework-equivalent knobs (prefetch, workers, request shaping, inflight caps, shuffle intensity).
- First-class metrics (req-size histograms, latency, in-flight bytes, QD).
- Page cache policy controls (Buffered FS with fadvise, DirectIO).
- Trace **record** and **replay** primitives.
- Consistent Python factory API across `file://`, `direct://`, `s3://`, `az://`.

## Non-Goals
- Implementing model training or transforms.
- Vendor/SDK benchmarking beyond required metrics.

---

## Scope & Tasks

### 1) Python multi-backend factories (parity)
- [ ] Export/confirm `create_dataset(...)` and `create_async_loader(...)` in Python surface.
- [ ] Support `file://`, `direct://`, `s3://`, `az://` uniformly (clear error on unknown scheme).
- [ ] Ensure options dict passes through consistently to Rust (batch_size, num_workers, prefetch, shuffle, reader_mode, loading_mode).
- [ ] **Tests:** `python/tests/test_multi_backend_iteration.py` iterates 2 batches per scheme (CI).

### 2) LoaderOptions & PoolConfig realism knobs
- [ ] Extend `LoaderOptions`:
  - [ ] `target_request_bytes: Option<usize>`
  - [ ] `max_bytes_in_flight: Option<usize>`
  - [ ] `random_index_mode: enum { None, Uniform, Zipfian{theta:f64} }`
  - [ ] `page_cache_mode: enum { Default, FadviseRandom, FadviseSequential, DirectIO }`
- [ ] Implement request shaping (range reads for object stores, chunked `pread` for FS).
- [ ] Enforce inflight cap backpressure in async pool.
- [ ] **Tests:** unit tests for parsing; integration verifying req-size histogram peak and inflight cap.

### 3) Metrics & histograms
- [ ] Add `MetricsSnapshot` with:
  - [ ] Request-bytes histogram
  - [ ] Latency (µs) histogram
  - [ ] Max queue depth
  - [ ] Peak bytes-in-flight
  - [ ] S3/Azure: count of range GETs; FS: cache hints applied
- [ ] Expose `to_json()`/`to_csv()`; bind to Python (PyO3).
- [ ] **Tests:** golden JSON/CSV schema snapshot + sanity test.

### 4) Trace record & replay (feature-flagged)
- [ ] Trace recorder: JSONL per-op `{uri, offset, len, t_submit[, sample_id]}`.
- [ ] Trace replayer: issues reads with same cadence; respects inflight caps & cache policy.
- [ ] Feature flag: `--features trace`.
- [ ] **Tests:** record from small dataset then replay; compare req-size hist/throughput envelope within ±10–15%.

### 5) Page-cache & DirectIO policy
- [ ] FS backend: wire `posix_fadvise(SEQUENTIAL|RANDOM|DONTNEED)` based on `page_cache_mode`.
- [ ] DirectIO backend: validate alignment and document constraints.
- [ ] **Tests:** A/B runs show expected latency/throughput divergence between modes.

---

## API Changes (Draft)
```rust
// src/data_loader/loader_options.rs
pub enum RandomIndexMode { None, Uniform, Zipfian { theta: f64 } }
pub enum PageCacheMode { Default, FadviseRandom, FadviseSequential, DirectIO }

#[derive(Clone, Default)]
pub struct LoaderOptions {
  pub batch_size: usize,
  pub num_workers: usize,
  pub prefetch: usize,
  pub shuffle: bool,
  pub reader_mode: ReaderMode,
  pub loading_mode: LoadingMode,
  pub target_request_bytes: Option<usize>,
  pub max_bytes_in_flight: Option<usize>,
  pub random_index_mode: RandomIndexMode,
  pub page_cache_mode: PageCacheMode,
}
```

## Acceptance Criteria
- [ ] Python factories run on all 4 schemes with identical option names.
- [ ] Metrics JSON includes `{req_bytes_hist, latency_us_hist, max_qd, peak_inflight, backend_counters}`.
- [ ] Inflight cap prevents RAM ballooning; verified via test.
- [ ] Trace replay matches recorded histogram and average bytes/step within ±10–15%.
- [ ] Docs updated with new options + metrics schema.

## Test Plan
- [ ] Unit: option parsing, metrics serialization.
- [ ] Integration: backend iteration per scheme.
- [ ] Perf sanity: request histogram peaks where configured.
- [ ] Trace record/replay parity test.

## Docs
- [ ] Update `docs/api/python-api-*.md` with new options and examples.
- [ ] Add `docs/metrics.md` and `docs/trace.md`.

## Risks / Mitigations
- Alignment/DirectIO errors → preflight checks & clear errors.
- Histogram overhead → feature flag or sampling rate.