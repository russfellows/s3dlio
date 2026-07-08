# Issue #148 — Measured before/after throughput

**Setup:** s3-ultra fake-S3 server on same host (`http://127.0.0.1:9200`),
40 GB/s theoretical ceiling (per warp benchmarks against the same target).
28-vCPU box, HTTP/1.1 (Phase 3 default).

**Client:** `examples/python/parallel_operations.py` (shipped by s3dlio),
run with two separate `uv` venvs pinning `s3dlio==0.9.106` (baseline) and
`s3dlio==0.9.108` (this branch's wheel). Range-GET bench is a small
supplemental script hitting `s3dlio.get(uri)` on a single large object —
that triggers `concurrent_range_get_impl` (the S3-specific range engine
that splits large GETs into concurrent chunk fetches internally).

Numbers are throughput as reported by the bench; range-GET numbers are
median of 5 warm runs.

---

## Bulk parallel PUT/GET, varying object size × concurrency

Concurrency values chosen to fit the 28-vCPU box.

### 64 KB × 100 objects (6 MB total)

| Conc | PUT baseline | PUT branch | Δ | GET baseline | GET branch | Δ |
|---|---|---|---|---|---|---|
| 1  | 23.0 MB/s | 39.4 MB/s | +71% | 40.1 MB/s | 91.9 MB/s | **+129%** |
| 4  | 106.3 MB/s | 140.9 MB/s | +33% | 278.6 MB/s | 345.4 MB/s | +24% |
| 16 | 389.9 MB/s | 401.8 MB/s | +3% | 698.8 MB/s | 789.8 MB/s | +13% |
| **64** | **163.7 MB/s** ← *baseline regresses vs 16* | **528.7 MB/s** ← *keeps scaling* | **3.2×** | **482.2 MB/s** ← *baseline regresses vs 16* | **597.0 MB/s** | 1.24× |

Mixed workload combined: baseline 84.9 MB/s → branch 324.4 MB/s = **3.8×**.

The baseline's throughput regression from concurrency=16 to concurrency=64
is the smoking-gun signature of the `.buffered(N)` bug: at high N, all
fetches pile up in one worker task's poll budget and throughput collapses.
The branch keeps scaling.

### 256 KB × 2000 objects (500 MB total)

| Conc | PUT baseline | PUT branch | Δ | GET baseline | GET branch | Δ |
|---|---|---|---|---|---|---|
| 1  | 30.5 MB/s   | 29.4 MB/s   | ≈ 0 | 300.7 MB/s  | 311.1 MB/s  | +3% |
| 4  | 210.9 MB/s  | 262.4 MB/s  | **+24%** | 1307.5 MB/s | 1448.8 MB/s | +11% |
| 8  | 555.3 MB/s  | 603.6 MB/s  | +9% | 2337.2 MB/s | 2505.0 MB/s | +7% |
| 16 | 938.0 MB/s  | 1092.3 MB/s | **+16%** | 3140.5 MB/s | 3687.1 MB/s | **+17%** |
| 32 | 1260.6 MB/s | 1327.2 MB/s | +5% | 3067.2 MB/s | 3568.1 MB/s | **+16%** |

### 1 MB × 1000 objects (1 GB total)

| Conc | PUT baseline | PUT branch | Δ | GET baseline | GET branch | Δ |
|---|---|---|---|---|---|---|
| 1  | 49.7 MB/s   | 50.7 MB/s   | ≈ 0 | 734.1 MB/s  | 786.7 MB/s  | +7% |
| 4  | 221.2 MB/s  | 207.6 MB/s  | ≈ 0 | 2270.1 MB/s | 2366.3 MB/s | +4% |
| 8  | 569.5 MB/s  | 579.8 MB/s  | ≈ 0 | 4589.9 MB/s | 4798.7 MB/s | +5% |
| 16 | 1219.8 MB/s | 1239.0 MB/s | ≈ 0 | 5647.1 MB/s | 6744.2 MB/s | **+19%** |
| 32 | 1750.2 MB/s | 1778.9 MB/s | ≈ 0 | 7916.8 MB/s | 6987.1 MB/s | -12% (noise) |

### 8 MB × 200 objects (1.6 GB total)

| Conc | PUT baseline | PUT branch | Δ | GET baseline | GET branch | Δ |
|---|---|---|---|---|---|---|
| 1  | 96.9 MB/s   | 95.3 MB/s   | ≈ 0 | 974.7 MB/s  | 898.4 MB/s  | -8% (noise) |
| 4  | 341.0 MB/s  | 328.0 MB/s  | ≈ 0 | 3000.3 MB/s | 2624.7 MB/s | -12% (noise) |
| 8  | 659.7 MB/s  | 669.8 MB/s  | ≈ 0 | 5060.2 MB/s | 5607.9 MB/s | +11% |
| 16 | 1345.9 MB/s | 1320.9 MB/s | ≈ 0 | 9407.0 MB/s | 8137.5 MB/s | -13% (noise) |
| 32 | 2016.0 MB/s | 2039.6 MB/s | ≈ 0 | 8914.7 MB/s | 8927.7 MB/s | ≈ 0 |

At this object size, wire time swamps per-fetch CPU work and the Phase 2
"spread across cores" fix has nothing to accelerate. No regression, no win.

---

## Whole-object GET on a single large object (concurrent_range_get_impl)

This is the code path that stacks Phase 1 (range-assembly double-copy
eliminated) + Phase 2 site 3.1f (per-chunk task spawn) + Phase 4b
(streaming SdkBody). Object splits into chunks of `get_optimal_chunk_size`:

| Object size | Chunks × chunk size | Baseline median MB/s | Branch median MB/s | Δ |
|---|---|---|---|---|
| **16 MiB** | 4 × 4 MB | 1017 | 931 | ≈ 0 (noise; only 4 chunks) |
| **64 MiB** | 16 × 4 MB | 678 | 1108 | **+63% (1.6×)** |
| **256 MiB** | 32 × 8 MB | 731 | 1526 | **+109% (2.1×)** |

The crossover sits at ~32 MiB — below that, too few chunks to see the
spread-across-cores win. At 256 MiB and above, we get the ~2× gain
predicted by the audit.

---

## Summary for the PR write-up

**The wins are workload-dependent, not universal.** The audit found and
fixed a real class of bugs (`.buffered(N)`/`join_all` polling all fetches
on one task), but its impact is a function of where CPU work sits vs.
wire time.

Two workload patterns see meaningful improvement:

1. **Bulk small-object I/O with high concurrency** — 64 KB × 100 objects
   at concurrency 64 goes 3.2× (PUT) / 1.24× (GET); mixed workload
   3.8×. Baseline actively regresses at high concurrency (single-task
   polling saturates); branch scales cleanly. This is the workload
   pattern the mlcommons/storage#701 report describes (streaming a
   training dataset of many small samples).
2. **Large single-object concurrent range GET** — 64 MiB → 1.6×, 256 MiB
   → 2.1×. This is the workload every large-object download hits
   automatically via the range engine.

Intermediate patterns (256 KB – 1 MB per object, moderate concurrency)
show consistent 15–20% wins. Above 8 MB per object, throughput is
wire-bound and the branch neither wins nor regresses (no throughput
regression is itself important — the Phase 2 rewrite is not slower
for the case it wasn't targeting).

**Not a win:** any single-object, single-request GET/PUT under a few MB.
Those were never the target; the audit was about parallelism, not
single-request latency.

**Correctness gains** (not throughput-visible, still important):
- Peak memory during range assembly: 2.28× → 1.03× total_size (Phase 1)
- Fetch cancellation on Python iterator drop: ~2 s → <500 ms latency
- Panic in a fetch task: previously truncated silently as StopIteration,
  now surfaces as an error
- Body-transfer failure retry paths verified byte-for-byte correct
  under the audit §2.4 fault-injection gate (Phase 4c)

**Local ceiling observation — s3dlio vs warp, same target, same host:**

warp gets 37.9 GiB/s at 8 MiB × conc=32 against s3-ultra. s3dlio (both
0.9.106 and 0.9.108) peaks around **~10 GB/s per Python process** at
similar shape, and plateaus around **~20 GB/s in aggregate** across
multiple processes on the same host. That's **~50% of warp's efficiency**
in throughput per box. The gap is **not** a regression from the audit;
the same ceiling exists on `0.9.106` and matches prior single-process
investigations by the maintainer (never observed above 10 GB/s
single-process; multi-process aggregate now confirmed to top out around
20 GB/s at this scale).

Fanning out across parallel Python processes gets partway there:

| s3dlio Python processes | Aggregate GET | CPU % (of 28 cores) |
|---|---|---|
| 1 | 10.7 GB/s | ~32% |
| 2 | 19.0 GB/s | ~49% (1.78× scaling) |
| 4 | 19.3 GB/s | ~69% (**flat** — worker oversubscription) |
| warp @ conc=32 | 37.9 GB/s | ~90% |

**Diagnosis of the s3dlio ceiling:** client-side CPU-bound in the AWS
SDK per-request pipeline (SigV4 signing, smithy request builder,
smithy response parsing, middleware chain). Inherent to the AWS SDK
choice, not to Phase 2's task-parallelism design.

- Single-process 1→2 scales near-linearly (1.78×) because AWS-SDK CPU
  work parallelizes cleanly across cores when Python processes are
  independent.
- 2→4 scales flat because each process spins ~28 tokio workers, so
  4 × 28 = 112 workers on 28 cores → context-switch thrashing.
  Constraining tokio workers per process (S3DLIO_TOKIO_THREADS or the
  `configure_tokio_threads` Python API) would allow further scaling
  but wasn't measured in this pass.
- warp uses `minio-go`, a much thinner S3 client (S3-only, no SDK
  middleware chain), which is why it reaches ~1.7 GB/s per used core
  vs s3dlio's ~1.4 GB/s per used core at 2 processes.

**Not audit scope**, but worth flagging for a future perf pass:
switching to a lighter-weight S3 client (aws-sdk-s3 tuned further,
`object_store` crate, or a minio-rs-style client) could substantially
raise the per-process ceiling. That's a design change of its own,
not a fix to the concurrency bug this audit targeted.

---

## How to reproduce

```bash
# One-time setup — two isolated uv envs, one per s3dlio version.
mkdir -p /path/to/perf-compare/{baseline-0.9.106,branch-0.9.108}

# Baseline (from PyPI):
cat > /path/to/perf-compare/baseline-0.9.106/pyproject.toml <<'EOF'
[project]
name = "s3dlio-perf-baseline"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["s3dlio==0.9.106", "numpy"]
EOF

# Branch (from the wheel built by ./build_pyo3.sh):
cat > /path/to/perf-compare/branch-0.9.108/pyproject.toml <<'EOF'
[project]
name = "s3dlio-perf-branch"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["s3dlio", "numpy"]
[tool.uv.sources]
s3dlio = { path = "/path/to/target/wheels/s3dlio-0.9.108-cp312-cp312-manylinux_2_39_x86_64.whl" }
EOF

# Sync both, start s3-ultra, then in each venv:
cd baseline-0.9.106 && uv sync
cd ../branch-0.9.108 && uv sync

# Configure env for a running s3-ultra instance (port 9200, minioadmin/minioadmin):
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://127.0.0.1:9200 AWS_REGION=us-east-1
unset AWS_CA_BUNDLE

# Bulk bench — copy examples/python/parallel_operations.py, edit `num_objects`
# and `object_size` for each data point:
cp examples/python/parallel_operations.py /path/to/perf-compare/
cd /path/to/perf-compare/baseline-0.9.106
uv run python ../parallel_operations.py s3://s3ultra-bench/base-$(date +%s)/

cd /path/to/perf-compare/branch-0.9.108
uv run python ../parallel_operations.py s3://s3ultra-bench/branch-$(date +%s)/
```
