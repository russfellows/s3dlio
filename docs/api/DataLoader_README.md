# s3dlio DataLoader Test Guide (vers. 0.4.3)

This guide shows how to run **all** verification scripts that exercise the Stage-0 features in `s3dlio`, validate API compatibility with AWS’s `s3torchconnector`, and interpret results.

> All scripts live under `python/tests/`. Commands below assume you’re in the project root.
> If your local filenames differ (e.g., `*_vers_4-3_demo.py`), adjust accordingly.

---

## 0) Prerequisites

* **Python** ≥ 3.10
* A built/installed `s3dlio` (local wheel or `maturin develop`)
* **AWS credentials** or a compatible S3 endpoint (MinIO/Ceph/etc.)

Required environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
# For S3-compatible services:
# export AWS_ENDPOINT_URL=https://minio.example.com
```

Optional packages:

```bash
# PyTorch (CPU wheel; use CUDA wheel if preferred)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# TensorFlow + JAX for the TF/JAX demo
pip install tensorflow-cpu jax jaxlib

# AWS comparison
pip install s3torchconnector
```

---

## 1) Quick Start (seed + one run)

```bash
# Seed 64 x 4 KiB objects under a test prefix
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --setup 64 --size 4096

# Iterable demo (sequential reader) – good for small objects
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --iterable --reader sequential --batch-size 4 --prefetch 16 --take 64
```

Example output:

```
[iterable] items=64 bytes=262144 (0.25 MiB) time=0.15s -> 427.5 it/s, 1.67 MiB/s
```

---

## 2) PyTorch demo (iterable/map, sharding, self-tests)

**Script:** `python/tests/torch_stage0_demo.py`

### 2.1 Seed data

```bash
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --setup 128 --size 4096 --jobs 64
```

### 2.2 Iterable dataset

```bash
# sequential reader (best for small objects)
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --iterable --reader sequential --batch-size 4 --prefetch 16 --take 64

# range reader (try with larger objects + higher inflight/prefetch)
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --iterable --reader range --inflight 8 --prefetch 32 --batch-size 4 --take 64
```

### 2.3 Map dataset

```bash
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --map --batch-size 8 --shuffle --take 64
```

### 2.4 Sharding (iterable across ranks × workers)

```bash
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --iterable --enable-sharding --reader sequential \
  --batch-size 1 --num-workers 4 --take 64
# Under DDP, launch with torchrun --nproc-per-node=N
```

### 2.5 Built-in self-tests (validates return types)

```bash
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --selftest --reader sequential --take 32
```

Expected:

```
[PASS] tensor: ...
[PASS] bytes:  ...
[PASS] reader: ...
```

---

## 3) TensorFlow / JAX demo

**Script:** `python/tests/jax_tf_stage0_demo.py`

> Seed the prefixes first (you can use the Torch script’s `--setup`):

### TensorFlow

```bash
python python/tests/jax_tf_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/tf \
  --tf --reader range --prefetch 16 --batch-size 4 --take 32
```

### JAX

```bash
python python/tests/jax_tf_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/jax \
  --jax --reader sequential --take 16
```

---

## 4) AWS vs s3dlio — simple throughput comparison

**Script:** `python/tests/compare_aws_s3dlio_loaders.py`

```bash
python python/tests/compare_aws_s3dlio_loaders.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --region us-east-1 \
  --reader sequential --prefetch 16 --batch-size 4 --take 128
```

Outputs per-side throughput (items/s, MiB/s).
For small (\~4 KiB) objects, `reader=sequential` is best; for multi-MiB objects, try `reader=range` with higher `--inflight` / `--prefetch`.

---

## 5) Full parity suite — throughput + integrity + reader semantics

**Script:** `python/tests/compare_parity_suite.py`
(Your file may be named `aws-s3dlio_compare_suite.py`)

### 5.1 Reader-parity run (closest to AWS)

```bash
python python/tests/compare_parity_suite.py \
  --prefix s3://<bucket>/s3dlio-test/torch \
  --region us-east-1 \
  --return-type reader --reader sequential \
  --batch-size 4 --prefetch 16 --take 128 \
  --integrity --semantics
```

You should see:

* **Throughput** for both libraries
* **Integrity PASS** (order-agnostic SHA-256 matching)
* **Reader semantics PASS** (partial read → remainder → EOF → close)

### 5.2 Large-object range test

```bash
# Seed larger objects once (e.g., 64 x 16 MiB)
python python/tests/torch_stage0_demo.py \
  --prefix s3://<bucket>/s3dlio-test/big \
  --setup 64 --size $((16*1024*1024))

# Range reader with concurrency
python python/tests/compare_parity_suite.py \
  --prefix s3://<bucket>/s3dlio-test/big \
  --region us-east-1 \
  --return-type reader --reader range \
  --inflight 8 --prefetch 32 --batch-size 2 --take 32 \
  --integrity
```

---

## 6) Compatibility notes — `return_type` & collate

`s3dlio.torch.S3IterableDataset.from_prefix(..., return_type=...)` supports:

* `return_type="tensor"` (default): yields `torch.Tensor([L], dtype=uint8)`
  **Works with default PyTorch collate**.
* `return_type="bytes"`: yields Python `bytes`
  **Use** `DataLoader(..., collate_fn=lambda b: b)` (identity collate).
* `return_type="reader"`: yields a file-like object with `.read()`, `.close()` (AWS-style)
  **Use** `collate_fn=lambda b: b`.

Example (drop-in for `s3torchconnector`):

```python
from torch.utils.data import DataLoader
from s3dlio.torch import S3IterableDataset

ds = S3IterableDataset.from_prefix(
    "s3://bucket/prefix",
    reader_mode="sequential",
    return_type="reader",
)
loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=lambda x: x)
for batch in loader:
    for r in batch:
        blob = r.read()
        r.close()
```

---

## 7) Choosing `reader_mode` & tuning tips

| Dataset & Object Size      | Recommended Mode   | Tips                                                                |
| -------------------------- | ------------------ | ------------------------------------------------------------------- |
| Many tiny files (≤ 16 KiB) | `sequential`       | Keep `--inflight` low (1–4); raise `--prefetch` (16–64).            |
| Multi-MiB objects          | `range`            | Raise `--inflight` (8–32) and `--prefetch` (32–128).                |
| Mixed sizes                | start `sequential` | If profiling shows large objects dominate, try `range` with tuning. |

**Rule of thumb:** `range` adds overhead (HEAD + range GETs). It shines when objects are **large enough** to amortize that and when **intra-object parallelism** helps.

---

## 8) Suggested CI smoke tests

* Seed a tiny prefix (e.g., 16 × 4 KiB).
* Torch self-test:

  ```bash
  python python/tests/torch_stage0_demo.py \
    --prefix s3://<bucket>/ci/torch \
    --setup 16 --size 4096 \
    --selftest --take 16
  ```
* Parity throughput (small take):

  ```bash
  python python/tests/compare_parity_suite.py \
    --prefix s3://<bucket>/ci/torch --region us-east-1 \
    --return-type reader --reader sequential \
    --batch-size 2 --prefetch 8 --take 16 \
    --integrity --semantics
  ```

---

## 9) Troubleshooting

* **0 items**: Prefix empty or wrong path. Seed with the Torch script’s `--setup`, or verify the prefix.
* **Auth/region**: Check env vars. For S3-compatible endpoints, set `AWS_ENDPOINT_URL`.
* **Collate errors**: If using `bytes` or `reader`, set `collate_fn=lambda b: b`.
* **Slow range on tiny files**: Use `reader=sequential` for small objects; `range` is for big ones.
* **Module import**: After editing Python sources, reinstall the wheel or `pip install -e .`.

---

## 10) Clean up

```bash
python - <<'PY'
from s3dlio import _pymod as _core
_core.delete("s3://<bucket>/s3dlio-test/", recursive=True)
print("Deleted test prefixes.")
PY
```

---

## Flags reference (by script)

| Flag                                  | Scripts                     | Meaning                                                              |
| ------------------------------------- | --------------------------- | -------------------------------------------------------------------- |
| `--prefix`                            | all                         | `s3://bucket/prefix/` to read                                        |
| `--setup N` / `--size` / `--jobs`     | torch demo                  | Seed **N** objects of size **size** bytes (writer inflight **jobs**) |
| `--iterable` / `--map`                | torch demo                  | Choose dataset style                                                 |
| `--selftest`                          | torch demo                  | Validate `tensor` / `bytes` / `reader` return types                  |
| `--reader {sequential,range}`         | torch, tf/jax, parity suite | Reader strategy                                                      |
| `--part-size BYTES`                   | torch, tf/jax, parity suite | Range GET part size                                                  |
| `--inflight N`                        | torch, tf/jax, parity suite | Max concurrent range parts                                           |
| `--prefetch N`                        | all                         | Loader queue capacity                                                |
| `--batch-size N`                      | all                         | Framework batch size                                                 |
| `--num-workers N`                     | torch demo                  | PyTorch DataLoader workers                                           |
| `--enable-sharding`                   | torch demo                  | Shard iterable across ranks × workers                                |
| `--return-type {tensor,bytes,reader}` | parity suite, s3dlio APIs   | s3dlio output type (tensor default; use reader for AWS parity)       |
| `--take N`                            | all                         | Stop after N items                                                   |
| `--integrity`                         | parity suite                | Order-agnostic SHA-256 multiset comparison                           |
| `--semantics`                         | parity suite                | Probe `.read()` partials/EOF/close                                   |

Happy testing!

