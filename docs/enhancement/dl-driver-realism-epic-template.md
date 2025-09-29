---
name: "Realism Epic: Framework profiles, trace, validation (dl-driver)"
about: Track work to add framework-equivalent profiles, trace capabilities, and validation harness to dl-driver.
title: "[Realism] dl-driver: framework profiles, trace validation, acceptance harness"
labels: enhancement, performance, io, epic, realism
assignees: ""
---

## Summary
Add framework-equivalent profiles, trace record/replay integration, and validation harness to make dl-driver provably "realistic" for AI/ML storage workloads.

## Goals
- Framework profile presets (Torch-like, TF-like, JAX-like) that map to s3dlio loader options.
- Trace record/replay integration for validation against real training runs.
- End-to-end acceptance harness with statistical validation.
- DLIO configuration bridge for drop-in compatibility.
- Memory-bounded storage stress testing capabilities.

## Non-Goals
- Implementing actual ML training or model transforms.
- Framework C++ integration (emulation is sufficient).

---

## Scope & Tasks

### 1) Framework profile presets
- [ ] Define preset configurations:
  - [ ] **Torch-Like**: `num_workers=N`, `prefetch=2*N`, `target_request_bytes≈128–256 KiB`, `random_index_mode=Uniform`
  - [ ] **TF-Like**: Sequential pipeline with map/batch/prefetch, larger `target_request_bytes`
  - [ ] **JAX-Like**: Large batches, fast prefetch with fewer workers
  - [ ] **ImageNet-Like**: Small-read dominated (2-6 KB), frequent open/close/metadata ops
  - [ ] **WebDataset-Like**: Shard+index workflows with sequential+random patterns
- [ ] Wire presets into existing framework profiles layer from v0.5.0.
- [ ] **Tests:** For each preset, assert request histogram, inflight bytes, overlap are in expected ranges.

### 2) Trace integration with s3dlio
- [ ] CLI flags: `--trace-record path.jsonl` / `--trace-replay path.jsonl`.
- [ ] Integration with s3dlio trace recorder/replayer.
- [ ] Configuration bridge: accept DLIO YAML and map to s3dlio options.
- [ ] **Tests:** Record from small PyTorch dataset; replay in dl-driver; compare within ±10% histogram/throughput.

### 3) Cache policy & DirectIO surface
- [ ] Expose cache policies: `cache_policy: default|random|sequential|direct`.
- [ ] Integration with s3dlio page cache controls.
- [ ] Add `--drop-cache` helper for test isolation.
- [ ] DirectIO alignment validation with clear error messages.
- [ ] **Tests:** A/B runs showing expected cache behavior differences.

### 4) Enhanced metrics & validation
- [ ] Integrate s3dlio metrics snapshots into dl-driver output.
- [ ] Per-rank + aggregated metrics: request histograms, queue depth, inflight bytes.
- [ ] Wall-clock based throughput calculations (already implemented in v0.6.2).
- [ ] **Tests:** Metrics JSON schema validation; sanity checks on aggregated values.

### 5) End-to-end acceptance harness
- [ ] Statistical validation framework:
  - [ ] **Storage envelope**: Mean/median req-size (±20%), tail latency p99 (±25%)
  - [ ] **Overlap validation**: Background I/O queue never starved during compute (min QD > 0 for ≥95% of steps)
  - [ ] **Cache scenario matrix**: Buffered vs DirectIO show expected divergence
- [ ] Output: Single JSON summary + CSV histograms.
- [ ] **Tests:** Green/red validation on known-good configurations.

### 6) DLIO compatibility bridge
- [ ] YAML parser for subset of DLIO configuration format.
- [ ] Mapping from DLIO options to s3dlio LoaderOptions + PoolConfig.
- [ ] Example configs: `torch-like.yaml`, `tf-like.yaml`, `trace-replay.yaml`.
- [ ] **Tests:** Parse example DLIO configs; verify option translation.

---

## CLI Changes (Draft)
```rust
#[derive(clap::Args)]
pub struct ProfileArgs {
    /// torch-like|tf-like|jax-like|imagenet-like|webdataset-like|custom
    #[clap(long, default_value="torch-like")]
    pub profile: String,
    
    /// Record I/O trace to JSONL file
    #[clap(long)]
    pub trace_record: Option<PathBuf>,
    
    /// Replay I/O trace from JSONL file
    #[clap(long)]
    pub trace_replay: Option<PathBuf>,
    
    /// Cache policy: default|random|sequential|direct
    #[clap(long, default_value="default")]
    pub cache_policy: String,
    
    /// Drop page cache before run (requires sudo helper)
    #[clap(long)]
    pub drop_cache: bool,
    
    /// DLIO-compatible YAML configuration
    #[clap(long)]
    pub dlio_config: Option<PathBuf>,
}
```

## Acceptance Criteria
- [ ] Framework presets generate expected I/O patterns (request size, parallelism, cache behavior).
- [ ] Trace replay matches recorded patterns within ±10–15% on key metrics.
- [ ] Acceptance harness provides clear pass/fail on realism criteria.
- [ ] DLIO config bridge handles common YAML patterns.
- [ ] Memory usage stays bounded via s3dlio inflight caps.
- [ ] Documentation shows validation methodology and example results.

## Test Plan
- [ ] Unit: profile preset option generation, YAML parsing.
- [ ] Integration: trace record/replay round-trip validation.
- [ ] Performance: acceptance harness on known configurations.
- [ ] Compatibility: DLIO YAML examples parsed and executed correctly.

## Docs
- [ ] README section: "Validating realism" with acceptance harness usage.
- [ ] Example configurations with expected results.
- [ ] Migration guide from DLIO to dl-driver.
- [ ] Performance tuning guide for realistic workloads.

## Risks / Mitigations
- Trace replay timing sensitivity → configurable tolerance bands.
- Complex DLIO YAML features → start with common subset, expand iteratively.
- Framework preset accuracy → validate against published framework behavior studies.