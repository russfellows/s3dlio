# DLIO unet3d datagen bottleneck investigation — 2026-07-10

**Status: RESOLVED (s3dlio side) — 2026-07-11.**  See
[Resolution](#resolution) at the end for the root cause, the fix, and the
live before/after numbers.  A DLIO-side follow-up (raising the default
`write_threads`, or delaying `S3DLIO_RT_THREADS` computation until after
`write_threads` is finalized) is still worth doing but no longer blocks
throughput — s3dlio now clamps the miscomputed env var back up.

Investigation into why `mlpstorage whatif training unet3d datagen object` (which drives DLIO's NPZ generator through s3dlio's MultipartUploadWriter into s3-ultra on localhost) ceilings at ~200 MB/s per rank while isolated s3dlio benchmarks against the same s3-ultra hit 2500 MB/s per rank.

## Numbers observed

| Test | Wall throughput |
|---|---|
| Isolated NP=1, 32 threads, bytes → MultipartUploadWriter | **1265 MB/s** |
| Isolated NP=1, 32 threads, **BytesView** → MultipartUploadWriter | **2518 MB/s** |
| Isolated NP=1, 32 threads, memoryview → MultipartUploadWriter | 1240 MB/s |
| Isolated NP=1, dlio-exact semaphore + submit + generate_npz_bytes | **2108 MB/s** |
| Isolated NP=1, same but `mpirun -n 1 --bind-to none --map-by socket` | **2120 MB/s** |
| Isolated NP=1, with DLIO stack imported (torch, tensorflow, mpi4py) | **2185 MB/s** |
| Isolated NP=1, after LoadConfig+derive_configurations | **1294 MB/s** |
| Actual `dlio_benchmark` NP=1 via `mlpstorage whatif` | **~150–247 MB/s** |
| Actual `dlio_benchmark` NP=4 via `mlpstorage whatif` | **~800 MB/s aggregate** |

**~10× gap** between what s3dlio+s3-ultra can do and what DLIO's actual generate flow achieves at NP=1.

## What I ruled out

Every one of these was tested with a dedicated benchmark that reproduced the exact code path in question. Not one moved the number:

1. **Buffer type** — Confirmed DLIO passes `BytesView` (the fast path). `put_data.hasattr(getbuffer)` = False for BytesView → falls to `payload = data` → hits `PyRef<PyBytesView>` extract → Arc clone (no memcpy). Fast path.
2. **NPZ generation speed** — A/B test in `rayon_ab_test.py`: 9.2 GB/s aggregate for NPZ fill alone at NP=4. Not the bottleneck.
3. **Semaphore + `pool.submit` + `_futures` list pattern** — `dlio_pattern_bench.py` reproduces DLIO's exact loop and gets 2108 MB/s. Pattern is fine.
4. **mpi4py MPI_Init** — `dlio_mpi_bench.py` under `mpirun -n 1 --bind-to none --map-by socket` still gets 2120 MB/s.
5. **DLIO's heavy imports** (torch, tensorflow, hydra) — `dlio_imported_bench.py` gets 2185 MB/s after importing them all.
6. **DFTracer** — `DFTRACER_ENABLE=False` and `dft_fn.log(f)` wraps in a wrapper that is a straight passthrough when disabled (verified via `inspect.getsource`).
7. **DLIO's `LoadConfig` + `derive_configurations`** — `dlio_init_then_bench.py` runs full config load and still gets 1294 MB/s.
8. **`S3DLIO_RT_THREADS=48`** (which DLIO's `ObjStoreLibStorage.__init__` sets) — `buffer_type_bench.py` with `S3DLIO_RT_THREADS=48` pre-set gets 2557 MB/s. Not the issue.
9. **`record_length_bytes_stdev=68341808`** (file-size variance) — set to 0, still slow. Not the issue.
10. **s3-ultra saturation** — while a slow DLIO run was in progress, an isolated `dlio_pattern_bench.py` in a separate process got 1280 MB/s. **s3-ultra is not the bottleneck.**
11. **AWS_ENDPOINT_URL / URI format** — DLIO and my benchmarks use identical URI shapes.

## Timing breakdown of DLIO's slow MultipartUploadWriter calls

Instrumented via `instrument2.py` monkey-patching `MultipartUploadWriter.from_uri/write/close`:

- `from_uri()` (CreateMultipartUpload): **p50 = 983 ms**, p90 = 1334, p99 = 4573 ms — should be single-digit ms
- `write()` (queue chunks): p50 = 7.8 ms — fine
- **`close()` (upload + Complete): p50 = 22843 ms** — should be ~1800 ms in isolated case

Aggregate per-second bytes transferred matches per-MPU stats × 32 concurrent = ~200 MB/s.

## The finding that mattered

`/proc/<pid>/task/*/comm` on the running DLIO benchmark process during the upload phase showed:

```
36 python3               (ThreadPoolExecutor workers + Python bookkeeping)
28 s3dlio-rayon-0..27    (Rayon global pool — sized correctly by my _pymod auto-init)
 1 s3dlio-rt             (runtime driver thread)
 1 s3dlio-rt-worke       (Tokio worker — TRUNCATED to 15 chars)
 1 jemalloc_bg_thd
```

**Only ONE `s3dlio-rt-worker` thread**, despite `S3DLIO_RT_THREADS=48` set in env at that point. Confirmed the name truncation is not hiding 47 more — `uniq -c` shows count 1 for that name.

The s3dlio global Tokio runtime driving `MultipartUploadWriter`'s coordinator + all in-flight UploadPart tasks has 1 worker. Every `run_on_global_rt` in `MultipartUploadSink::finish_blocking` funnels through that single worker's queue. 32 concurrent MPUs × 32 in-flight parts each = 1024 async tasks waiting on one core.

I did NOT yet capture the same measurement on the FAST isolated bench during its upload phase (kept catching the parent, not the mp.Process child) — but every symptom fits: 22 s per close, ~200 MB/s aggregate, low CPU (171%).

## Where the count-of-1 could come from

Not yet fully proven, but the plumbing to look at next session:

- `src/s3_client.rs:98–125` `global_rt_handle()` lazily spawns a fresh OS thread named `s3dlio-rt`, which creates the runtime via `TokioBuilder::new_multi_thread().worker_threads(get_runtime_threads())`. `get_runtime_threads()` at [s3_client.rs:137](../../src/s3_client.rs#L137) reads `S3DLIO_RT_THREADS` env var *first*. In DLIO's flow this env var IS set to 48 by the time first S3 op runs (verified in `env_diff.py`).
- Either `get_runtime_threads()` is returning 1 (contra reading), or `worker_threads(1)` is coming from somewhere else, or Tokio is falling back to 1 due to a build-time cfg. **Read `get_runtime_threads()` at the exact moment the first `global_rt_handle()` call fires** — add an `eprintln!` and re-run.
- Also check whether `configure_thread_pools(0)`'s call to `pyo3_async_runtimes::tokio::init(builder)` at import time (my new fix) somehow blocks/init-freezes the s3dlio global runtime side. They are supposed to be independent runtimes.

## What's already landed in s3dlio (worth keeping regardless)

Verified: `_pymod` auto-config → 28 Rayon threads created at import time, exactly as designed. That part of the thread-pool fix DOES work at NP=1 on this machine (28 `s3dlio-rayon-N` threads visible). And the RED-then-GREEN tests for MPI-aware `get_runtime_threads` / `configure_thread_pools` / `build_checkpoint_runtime` / `recommended_data_gen_threads` all pass cleanly.

## Scratchpad artifacts (for pickup next session)

All under `/tmp/claude-1000/-home-eval-Documents-Code/264eebff-b762-4534-aa79-c34c5ca0e64d/scratchpad/`:

- `lo_monitor.py` — `/proc/net/dev` bandwidth sampler
- `put_only_bench.py` — pure MPU with pre-generated payload
- `gen_plus_put_bench.py` — MPU + generate per-file
- `buffer_type_bench.py` — bytes vs BytesView vs memoryview
- `dlio_pattern_bench.py` — DLIO's exact semaphore+submit pattern
- `dlio_mpi_bench.py` — same, under mpirun + mpi4py init
- `dlio_imported_bench.py` — same, after importing torch/tensorflow/hydra
- `dlio_init_then_bench.py` — same, after LoadConfig+derive
- `instrument.py`, `instrument2.py`, `instrument3.py`, `instrument4.py` — DLIO monkey-patching to measure from_uri/write/close and submit gaps
- `env_diff.py` — dumps env vars right before generate() call

## Next-session first move

1. Add an `eprintln!("s3dlio-rt: get_runtime_threads() -> {}", n)` inside `global_rt_handle()` in `src/s3_client.rs` right before `.worker_threads(n)`.
2. Run `instrument4.py` — see what n actually gets printed at first S3 op.
3. If it's 1 → the bug is in `get_runtime_threads()` under DLIO's env. If it's 48 → the bug is in the Tokio side or in how I'm counting worker threads (Tokio may share names with truncation).

## Resolution

Executed the next-session plan on 2026-07-11:

1. Added a logfile diagnostic (`/tmp/s3dlio_rt_diag.log`) at
   `global_rt_handle()`.  Rebuilt the wheel via `./build_pyo3.sh`,
   reinstalled into the mlp-storage venv, re-ran `instrument4.py`.
2. Diagnostic showed `S3DLIO_RT_THREADS=Some("1")` — not 48.  Root cause
   confirmed: **DLIO** was setting the env var to 1, not 48.

### Where the 1 came from

`DLIO_local_changes/dlio_benchmark/utils/config.py:227`:

```python
write_threads: int = 1
```

`DLIO_local_changes/dlio_benchmark/storage/obj_store_lib.py:321-324` (in
`ObjStoreLibStorage.__init__`, runs at storage-construction time,
**before** DLIO auto-tunes `write_threads`):

```python
_write_threads = getattr(self._args, "write_threads", 8)
_rt_threads = min(_write_threads * 3 // 2, 128)
os.environ["S3DLIO_RT_THREADS"] = str(_rt_threads)
```

With `write_threads=1`, `_rt_threads = min(1*3//2, 128) = 1`.  DLIO's
`_S3DLIO_RT_AUTO` sentinel logic is designed to let this recompute later
if `ObjStoreLibStorage.__init__` runs again, but for a NP=1 run there is
only ever **one** storage instantiation, so the recompute never fires.

### s3dlio-side fix

Even though the *value* is DLIO's misjudgment, the fact that s3dlio
silently obeyed it and built a 1-worker Tokio runtime is a hardening
bug on the s3dlio side.  Fixed in v0.9.112: `get_runtime_threads()`
and `effective_thread_budget()` now share
`s3_client::clamped_env_rt_threads()`, which:

- If `S3DLIO_RT_THREADS` is set, its value is clamped to `>= RT_THREADS_LIMIT`
  whenever the requested value is below `RT_THREADS_LIMIT / 4`.
  `RT_THREADS_LIMIT` is set at s3dlio import time by
  `configure_thread_pools(0)` to the MPI-aware per-process budget
  (`num_cpus / world_size`, floor 1).
- Emits a stderr warning the first time the clamp fires so operators
  can spot the miscomputation upstream.
- `S3DLIO_RT_THREADS_UNSAFE=1` bypasses the clamp for legitimate
  low-thread scenarios (single-thread test suites, fault injection).

### Live before/after (RED-then-GREEN, same s3-ultra target, 28-core host)

DLIO `unet3d_datagen`, 40 files × 146 MB, NP=1, `instrument4.py` /
`instrument5.py`:

| Metric               | Pre-fix (0.9.110)         | Post-fix (0.9.112)      |
| -------------------- | ------------------------- | ----------------------- |
| Wall time            | 27.4 s                    | **2.9 s**               |
| Throughput           | 214 MB/s                  | **~1928 MB/s** (9×)     |
| Upload `close()` p50 | 22 843 ms                 | **309 ms** (77×)        |
| Upload `close()` p99 | 26 368 ms                 | 529 ms                  |
| Rt-worker threads    | 1                         | 28                      |

Cross-backend live smoke test with `S3DLIO_RT_THREADS=1` forced (see
`scratchpad/live_backend_verify.py`): `file://` 397 MB/s, `direct://` on
`/mnt/nvme_data` 655 MB/s, `s3://` (s3-ultra) 919 MB/s.
`S3DLIO_RT_THREADS_UNSAFE=1` correctly re-enables the pathological
1-worker behavior for testing purposes.  Sane env values
(e.g. `S3DLIO_RT_THREADS=16`) pass through unchanged with no warning
and full throughput (1.1 GB/s).

### Test coverage (all RED→GREEN)

Added in `src/s3_client.rs` `mpi_aware_runtime_sizing_tests`:
- `get_runtime_threads_clamps_grossly_underprovisioned_env_var_up_to_limit`
- `get_runtime_threads_env_var_at_or_above_quarter_of_limit_is_honored_verbatim`
- `get_runtime_threads_unsafe_env_bypasses_the_clamp`
- `get_runtime_threads_env_var_low_with_no_limit_set_is_honored`

Added in `src/constants.rs` `effective_thread_budget_tests`:
- `test_effective_budget_clamps_grossly_underprovisioned_env_var`
- `test_effective_budget_unsafe_bypass_respects_low_env_var`

All verified RED against pre-fix `get_runtime_threads()` and
`effective_thread_budget()`; all pass under the fix.  Full pre-push
gate clean (fmt, clippy `-D warnings` for both `default` and
`--features extension-module`, 404+412 unit tests).

### DLIO-side follow-up (out of s3dlio scope)

The s3dlio-side clamp is a *defense in depth*.  The upstream bug —
DLIO setting `S3DLIO_RT_THREADS=1` because its own `write_threads`
default is 1 at the time `ObjStoreLibStorage.__init__` runs — is
still worth fixing on the DLIO side.  Two clean options:

1. Raise the DLIO Hydra default `write_threads` from 1 to something
   like `8` or `min(cpus, 32)`.  (`dlio_benchmark/utils/config.py:227`.)
2. Delay the `S3DLIO_RT_THREADS` derivation to a later hook when
   `write_threads` has been auto-tuned (or call `s3dlio.configure_tokio_threads(n)`
   explicitly from that hook instead of writing to the env var).

Not urgent given the s3dlio clamp, but would eliminate the warning
banner and simplify reasoning about the invariant.
