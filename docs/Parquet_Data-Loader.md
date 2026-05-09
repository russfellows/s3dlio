# Parquet DataLoader — s3dlio v0.9.98

**Last Updated:** May 9, 2026

## Table of Contents

1. [Overview](#overview)
2. [Feature Flags](#feature-flags)
3. [Memory Model](#memory-model)
4. [API Reference](#api-reference)
   - [Rust API](#rust-api)
   - [Python API](#python-api)
5. [Decode Modes](#decode-modes)
6. [Epoch-2+ Fast Path](#epoch-2-fast-path)
7. [Performance Model](#performance-model)
8. [Configuration Options](#configuration-options)
9. [Python Usage Examples](#python-usage-examples)
10. [Rust Usage Examples](#rust-usage-examples)
11. [Testing](#testing)
12. [Architecture Deep Dive](#architecture-deep-dive)

---

## Overview

The **Parquet DataLoader** provides a high-performance, epoch-aware training data loader for
Parquet files stored on s3dlio -compatible storage. Each dataset item corresponds to one
Parquet **row group** — the natural unit of parallelism and I/O for Parquet files.

Key properties:

- **Zero re-fetch on epoch 2+**: Row-group byte ranges (offsets + lengths) are cached in a
  process-global `DashMap` after the first epoch. Subsequent epochs skip all S3 footer GETs.
- **Minimal RAM**: The dataset holds only metadata in RAM. Each `get()` call fetches one row
  group payload which is freed when Python releases it.
- **Two decode modes**: `Raw` (Python receives compressed Parquet bytes) and `ArrowIpc`
  (Rust decodes to Arrow RecordBatch, returns IPC stream bytes).
- **Concurrent prefetch**: The Python-facing `PyParquetStreamLoader` keeps N row-group GETs
  in flight simultaneously using Tokio `buffer_unordered`.
- **8-instance safe**: All process-global caches are shared, so 8 concurrent dataset
  instances incur no 8× duplication of metadata.

---

## Feature Flags

| Cargo feature | Included in default? | What it enables |
|---------------|---------------------|-----------------|
| `parquet` | ✅ Yes | `ParquetRowGroupDataset`, `ParquetIndex`, `parquet_file_cache`, `Raw` decode mode, footer parsing |
| `parquet-arrow` | ✅ Yes | `ArrowIpc` decode mode — Rust decodes Parquet column data to Arrow `RecordBatch`, serialises to Arrow IPC stream |
| `arrow-backend` | ❌ No (mutually exclusive with `native-backends`) | Apache Arrow's `object_store` crate as the S3 object-storage backend — **completely separate from the Parquet DataLoader** |

> **Important**: `parquet-arrow` (Arrow IPC decode for the DataLoader) is **not** the same as
> `arrow-backend` (Arrow's `object_store` as the storage client). The former is in the default
> build; the latter is not and cannot coexist with `native-backends`.

Both `parquet` and `parquet-arrow` are built by default. There is no action required to
enable the Parquet DataLoader.

---

## Memory Model

The Parquet DataLoader is designed for concurrent training with 8+ workers sharing a single
process. Only metadata is held in RAM at all times; bulk data is fetched on demand and freed
immediately.

| What | Size | Scope | Lifetime |
|------|------|-------|----------|
| `extents` (byte ranges per RG) | ~48 B × N_rgs | Per dataset instance | Until dataset is dropped |
| `parquet_file_cache` (parsed footer) | a few MB per file (Arc-shared) | Process-global | Until `clear()` or process exit |
| `parquet_index` DashMap entries | ~80 B × N_rgs | Process-global singleton | Persists across epochs |
| Active `get()` payload | 1 row-group ≈ 1–100 MB | Python object | Freed when Python releases `bytes` |

### Implication for 8 Concurrent Workers

8 dataset instances that cover the same prefix **share** the process-global caches — no
8× duplication. Peak RAM is approximately:

```
metadata overhead    ≈ (files × footer_size) + (N_rgs × 80 B index entries)
active data in RAM   ≈ prefetch_depth × rg_size × num_workers
```

**Example** (64 files, 1,968 RGs, 100 MB per RG, prefetch=2, 8 workers):

```
metadata: ~20 MB (footers) + ~160 KB (index) ≈ 20 MB shared
data:     100 MB × 2 × 8 = 1,600 MB = 1.6 GB
total:    ≈ 1.6 GB
```

For tighter RAM budgets, reduce `prefetch` (the `concurrency` parameter).

---

## API Reference

### Rust API

#### `ParquetRowGroupDataset`

```rust
pub struct ParquetRowGroupDataset { /* ... */ }

impl ParquetRowGroupDataset {
    /// Construct the dataset.
    ///
    /// * `uri_prefix`  — S3 prefix, e.g. `"s3://bucket/data/train/"`
    /// * `col_indices` — column indices to include per RG; `None` = all columns
    /// * `footer_cap`  — bytes to fetch from file tail for footer parsing (≥ largest footer)
    /// * `decode_mode` — `Raw` or `ArrowIpc`
    pub fn new(
        uri_prefix: &str,
        col_indices: Option<&[usize]>,
        footer_cap: usize,
        decode_mode: ParquetDecodeMode,
    ) -> Result<Self, DatasetError>;

    /// Total number of row groups across all files.
    pub fn len(&self) -> Option<usize>;

    /// Number of rows in the row group at `global_rg_idx`.
    pub fn num_rows_in_rg(&self, global_rg_idx: usize) -> Option<i64>;

    /// S3 URI of the file containing `global_rg_idx`.
    pub fn file_uri_for_rg(&self, global_rg_idx: usize) -> Option<&str>;

    /// The active decode mode for this dataset instance.
    pub fn decode_mode(&self) -> ParquetDecodeMode;

    /// Return `(file_uri_idx, rg_idx_in_file)` for every RG in order.
    pub fn rg_info_vec(&self) -> Vec<(usize, usize)>;

    /// Shared reference to the file URI list.
    pub fn file_uris_arc(&self) -> Arc<Vec<String>>;
}
```

#### `ParquetDecodeMode`

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParquetDecodeMode {
    /// Compressed Parquet column-chunk bytes — Python decodes or discards.
    #[default]
    Raw,
    /// Rust decodes to Arrow RecordBatch → Arrow IPC stream bytes.
    /// Requires the `parquet-arrow` feature (in default build).
    #[cfg(feature = "parquet-arrow")]
    ArrowIpc,
}
```

#### `ParquetIndex` (process-global singleton)

```rust
// Access via:
use s3dlio::data_loader::parquet_index;

let gidx = parquet_index::global();   // &'static ParquetIndex

// Query cached extents for a file (byte_offset, byte_length, num_rows per RG):
let extents: Option<Vec<(u64, u64, i64)>> = gidx.file_extents("s3://bucket/data/train/shard_00.parquet");

// Check whether a file is indexed:
let indexed: bool = gidx.is_indexed("s3://bucket/...");

// Row count for a single RG:
let rows: Option<i64> = parquet_index::rg_num_rows("s3://bucket/...", rg_idx);

// Invalidate one file (forces re-fetch next epoch):
gidx.invalidate_uri("s3://bucket/...");

// Clear all entries (forces full re-fetch for all files):
gidx.clear();
```

#### `DEFAULT_FOOTER_CAP`

```rust
/// Default bytes fetched from the tail of each file for footer parsing (4 MiB).
/// Covers all known production workloads (DLRM footers ≈ 2.66 MiB).
pub const DEFAULT_FOOTER_CAP: usize = 4 * 1024 * 1024;
```

---

### Python API

#### `create_async_loader(uri, opts)` — Parquet mode

```python
import s3dlio

loader = s3dlio.create_async_loader(
    "s3://bucket/data/train/",
    {
        "format":     "parquet",       # Required to activate Parquet mode
        "decode":     "raw",           # "raw" (default) or "arrow"
        "columns":    None,            # list[int] or None = all columns
        "footer_cap": 4 * 1024 * 1024, # bytes for footer parsing (default: 4 MiB)
        "prefetch":   32,              # concurrent in-flight row-group GETs (default: 32)
    }
)

for item in loader:
    # item is a dict: {"data": bytes, "uri": str, "rg_idx": int}
    raw_bytes = item["data"]
    source_uri = item["uri"]
    rg_index   = item["rg_idx"]
```

| Option key | Type | Default | Description |
|------------|------|---------|-------------|
| `"format"` | `str` | — | Must be `"parquet"` to activate Parquet mode |
| `"decode"` | `str` | `"raw"` | `"raw"` — Python receives Parquet bytes; `"arrow"` — Rust decodes to Arrow IPC |
| `"columns"` | `list[int]` or `None` | `None` | Column subset; `None` = all columns |
| `"footer_cap"` | `int` | `4194304` | Bytes to fetch from file tail for footer parsing |
| `"prefetch"` | `int` | `32` | Concurrent in-flight row-group GETs (bounded channel capacity) |

#### Iterator items

Each iteration yields a `dict` with:

| Key | Type | Value |
|-----|------|-------|
| `"data"` | `bytes` | Row-group payload (Parquet column chunks in `Raw` mode; Arrow IPC bytes in `ArrowIpc` mode) |
| `"uri"` | `str` | S3 URI of the source file |
| `"rg_idx"` | `int` | Row-group index within the file |

---

## Decode Modes

### `Raw` mode (default)

The dataset returns the raw, compressed Parquet column-chunk bytes exactly as stored on S3.
Python (or dlio_benchmark) is responsible for decoding.

```python
import pyarrow.parquet as pq
import io

for item in loader:
    buf = io.BytesIO(item["data"])
    table = pq.read_table(buf)
    # table is a pyarrow.Table with all columns (or the selected subset)
```

**When to use `Raw`**:
- Pure storage benchmarks (discard bytes without decoding)
- Python-side control over decoding (custom projections, filters, etc.)
- Minimum Rust CPU overhead — Rust only fetches, never decodes

**`Raw` mode fast path**: On epoch 2+, when `col_indices=None`, `build_extents()` reads from the
process-global `DashMap` and returns an empty `file_metadata` Vec. `get_raw()` never accesses
per-file metadata, so no cache lookups are needed on the critical path.

### `ArrowIpc` mode (requires `parquet-arrow` feature — in default build)

Rust decodes the Parquet column data to an Arrow `RecordBatch` and serialises it to Arrow IPC
stream format. Python receives ready-to-use IPC bytes.

```python
import pyarrow as pa

for item in loader:
    buf = pa.py_buffer(item["data"])
    reader = pa.ipc.open_stream(buf)
    batch = reader.read_next_batch()
    # batch is a pyarrow.RecordBatch — decoded, in Arrow memory format
    arr = batch.column(0).to_pylist()
```

**When to use `ArrowIpc`**:
- CPU-bound training loops where Python decode becomes a bottleneck
- When you need Arrow memory format directly (e.g. for JAX/TF integration)
- When Rust's parallel Tokio threads can absorb decode cost more efficiently than Python

**`ArrowIpc` mode always populates `file_metadata`** (the Arrow decoder needs `ParquetMetaData`
for the schema), even on epoch 2+. It still benefits from the global index for byte extents —
only the file cache is consulted, not re-fetched from S3.

---

## Epoch-2+ Fast Path

After the first full construction (`new()`) for a set of files, all row-group byte ranges
are stored in `parquet_index::global()` — a process-global `OnceLock<ParquetIndex>` backed
by a `DashMap<String, Vec<RgIndexEntry>>`. The key is the S3 URI string.

### What happens on epoch 1 (slow path):

1. `list_objects` — lists `.parquet` files under the prefix (one S3 request)
2. For each file: range GET of the last `footer_cap` bytes from S3
3. Parse Parquet footer → `ParquetMetaData`
4. Cache in `parquet_file_cache` (Arc-shared, de-duplicates concurrent fetches)
5. Compute `RgExtent` for each row group from column chunk metadata
6. Insert into `parquet_index::global()` so epoch 2+ can skip steps 2–5

### What happens on epoch 2+ (fast path):

1. `list_objects` — still hits S3 (can't be skipped — files may be added/removed)
2. For each file: `gidx.is_indexed(uri)` — DashMap lookup, no network I/O
3. Read `RgExtent` entries directly from DashMap — zero S3 I/O
4. Return empty `file_metadata` Vec (Raw mode) or consult warm file cache (ArrowIpc mode)

**Fast path trigger condition**: `col_indices.is_none()` AND all listed files are in the
global index. If any file is missing from the index (new file appeared, or `invalidate_uri`
was called), the slow path runs for that file only and re-indexes it.

### Measured speedup (MinIO, 2 files, 4 RGs total):

```
Epoch-1 construction: 20.4 ms  (list + 2 footer GETs)
Epoch-2 construction:  8.3 ms  (list only; DashMap lookup ≈ 0 ms)
Speedup:               2.5×
```

At scale (64 files), epoch-1 fetches 64 footers concurrently (~0.5 s total); epoch-2 does
zero footer GETs (<0.1 s total), giving a much larger proportional speedup.

---

## Performance Model

| Step | Cost (per worker) | Notes |
|------|-------------------|-------|
| Epoch-1 construction | ~0.5 s | List + 64 concurrent footer GETs |
| Epoch-2+ construction | < 0.1 s | List only; DashMap lookup ≈ zero |
| 1,968 range GETs @ 10 GiB/s | ~0.33 s | Main I/O cost, runs every epoch |
| Rust Arrow decode (`ArrowIpc`) | ~0.05 s | Parallel on Tokio worker threads |
| Python iteration overhead | caller-dependent | No extra copies inside s3dlio |

### Concurrency recommendation

```
prefetch = min(cpu_count * 4, network_bandwidth_MBps / avg_rg_size_MB)
```

The default of 32 concurrent GETs is appropriate for 1 GiB/s network and ~30 MB row groups.
For 10 GiB/s+ networks, increase `prefetch` to 64–128.

---

## Configuration Options

### `footer_cap` — footer parsing buffer size

The Parquet footer is located at the end of the file. `footer_cap` controls how many bytes
are fetched from the file tail in one range GET to parse the footer.

| Workload | Typical footer size | Recommended `footer_cap` |
|----------|--------------------|-----------------------------|
| Small files (< 100 MB) | < 100 KB | 1 MiB (default works) |
| ImageNet NPZ | ~500 KB | 4 MiB (default) |
| DLRM recommendation | ~2.66 MiB | 4 MiB (default) |
| Very large schemas | > 4 MiB | Set to `max(footer_sizes) + 1 MiB` |

If `footer_cap` is smaller than the actual footer, parsing will fail with an error. Use the
default (4 MiB) unless you have measured that footers are larger.

### `col_indices` — column projection

Pass a list of 0-based column indices to fetch only those columns per row group.

```python
loader = s3dlio.create_async_loader(
    "s3://bucket/data/",
    {"format": "parquet", "columns": [0, 2, 5]}  # only columns 0, 2, and 5
)
```

> **Note**: Column projection only applies to the byte range fetched from S3. The global
> index fast path is **disabled** when `col_indices` is not `None`, because the cached byte
> ranges cover all columns and per-column extents are not separately stored in the index.
> Epoch-2 speedup is only available for full-column (all-column) workloads.

---

## Python Usage Examples

### Minimal read loop (all columns, Raw mode)

```python
import s3dlio

loader = s3dlio.create_async_loader(
    "s3://mlp-bucket/train/",
    {"format": "parquet"}
)

total_bytes = 0
for item in loader:
    total_bytes += len(item["data"])

print(f"Read {total_bytes / 1e9:.2f} GB across all row groups")
```

### Training loop with PyArrow decode (Raw mode)

```python
import s3dlio
import pyarrow.parquet as pq
import io
import torch

loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "prefetch": 64}
)

for epoch in range(10):
    for item in loader:
        table = pq.read_table(io.BytesIO(item["data"]))
        features = torch.from_numpy(table.column("feature").to_pydict()["feature"])
        labels = torch.from_numpy(table.column("label").to_pydict()["label"])
        # ... train step ...
```

### Arrow IPC decode (Rust-side decode)

```python
import s3dlio
import pyarrow as pa

loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "decode": "arrow", "prefetch": 32}
)

for item in loader:
    buf = pa.py_buffer(item["data"])
    batch = pa.ipc.open_stream(buf).read_next_batch()
    # batch: pyarrow.RecordBatch, decoded and ready
    arr = batch.column(0).to_pylist()
```

### dlio_benchmark integration

s3dlio's Parquet DataLoader is a drop-in replacement for `dlio_benchmark`'s native reader.
Set `format: parquet` and `record_length` to your average row-group size:

```yaml
# dlio_benchmark config
dataset:
  data_folder: s3://bucket/train/
  format: parquet
  record_length: 52428800   # 50 MiB (average row-group size)

reader:
  reader_type: s3dlio
  batch_size: 1
```

In Python (or DLIO plugin code):

```python
loader = s3dlio.create_async_loader(
    config.data_folder,
    {
        "format":   "parquet",
        "decode":   "raw",       # DLIO's decode_mode=none equivalent
        "prefetch": config.prefetch_size,
    }
)
```

### Epoch-2+ fast path — explicit cache management

```python
import s3dlio
from s3dlio._internal import parquet_index_clear, parquet_file_cache_clear

for epoch in range(num_epochs):
    loader = s3dlio.create_async_loader(
        "s3://bucket/train/",
        {"format": "parquet", "prefetch": 32}
    )
    for item in loader:
        train(item["data"])
    # No need to clear caches between epochs — index persists for fast epoch-2+

# Clear caches if files are replaced between runs:
# parquet_index_clear()
# parquet_file_cache_clear()
```

---

## Rust Usage Examples

```rust
use s3dlio::data_loader::{
    Dataset, DatasetError, ParquetDecodeMode, ParquetRowGroupDataset, DEFAULT_FOOTER_CAP,
};

// Build dataset (epoch 1 — fetches footers from S3)
let ds = ParquetRowGroupDataset::new(
    "s3://bucket/data/train/",
    None,              // all columns
    DEFAULT_FOOTER_CAP,
    ParquetDecodeMode::Raw,
)?;

println!("Total row groups: {}", ds.len().unwrap_or(0));

// Iterate over all row groups
for rg_idx in 0..ds.len().unwrap_or(0) {
    let payload: bytes::Bytes = ds.get(rg_idx).await?;
    println!("RG {}: {} bytes, {} rows",
        rg_idx,
        payload.len(),
        ds.num_rows_in_rg(rg_idx).unwrap_or(0)
    );
}

// Epoch 2 — fast path, no footer GETs
let ds2 = ParquetRowGroupDataset::new(
    "s3://bucket/data/train/",
    None,
    DEFAULT_FOOTER_CAP,
    ParquetDecodeMode::Raw,
)?;  // construction time: < 0.1 s (DashMap lookups only)
```

---

## Testing

### Unit tests (no network required)

Run with:

```bash
cargo test --features parquet-arrow -- parquet --nocapture
```

**17 unit tests** covering:

| Test | What it proves |
|------|---------------|
| `test_parse_footer_magic_check` | Footer parser validates PAR1 magic bytes |
| `test_parse_footer_bad_magic` | Footer parser rejects invalid magic bytes |
| `test_parse_footer_too_small` | Footer parser rejects under-size buffers |
| `test_decode_mode_default_is_raw` | `ParquetDecodeMode::default()` is `Raw` |
| `test_decode_mode_variants_compile` | Both decode mode variants compile and are distinct |
| `test_rg_extent_round_trip` | Write a real Parquet file in memory, verify extents |
| `test_needs_file_metadata_raw_is_false` | `Raw` mode does not need file_metadata |
| `test_needs_file_metadata_arrow_ipc_is_true` | `ArrowIpc` mode requires file_metadata |
| `test_needs_file_metadata_modes_differ` | The two modes return different values |
| `test_global_index_is_singleton` | `global()` returns the same address on every call |
| `test_global_index_col_indices_is_none` | Global index stores all-column extents |
| `test_index_insert_and_file_extents` | Insert entries, read back correct extents |
| `test_rg_for_sample_binary_search` | Binary search finds correct RG for sample |
| `test_rg_range_lookup` | Range lookup returns correct row range |
| `test_check_and_mark_epoch` | Epoch counter increments correctly |
| `test_invalidate_uri` | `invalidate_uri` removes file from index |
| `test_rg_lookup_combined` | Combined insert/lookup/invalidate round-trip |

### Integration tests (requires live MinIO)

Run with:

```bash
set -a && source .env && set +a && \
  cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture
```

> **Important**: Use `--test-threads=1`. The tests share the process-global `ParquetIndex`
> and intentionally call `clear()` between tests. Parallel execution causes races.

**6 integration tests** covering:

| Test | What it proves |
|------|---------------|
| `test_parquet_dataset_epoch1_construction` | Epoch-1 constructs dataset, populates global index, `get(0)` returns non-empty bytes |
| `test_parquet_dataset_epoch2_fast_path` | Epoch-2 produces identical results to epoch-1 |
| `test_parquet_index_row_offsets_consistency` | `rg_num_rows()` matches `num_rows_in_rg()` across all RGs |
| `test_epoch2_faster_than_epoch1` | **Timing proof**: epoch-2 construction ≤ epoch-1 construction; epoch-2 < 2 s absolute |
| `test_raw_fast_path_skips_file_metadata` | Raw epoch-2 `file_metadata_len()` == 0 |
| `test_arrow_ipc_fast_path_keeps_file_metadata` | ArrowIpc epoch-2 `file_metadata_len()` == num_files |

### Python integration tests

```bash
set -a && source .env && set +a && \
  python -m pytest tests/test_parquet_dataloader.py -v
```

Tests in `tests/test_parquet_dataloader.py`:

- `TestParquetDataloaderBasic`: list, raw loader, bytes non-empty, Parquet magic bytes, num_rows, rg_info_vec
- `TestParquetEpochFastPath`: same RG count across epochs, identical per-RG row counts
- `TestParquetArrowIpc`: Arrow IPC decode returns valid RecordBatch, column values correct

---

## Architecture Deep Dive

### Component relationships

```
Python caller
    │
    ▼
PyParquetStreamLoader              ← Python object (__iter__ returns ParquetStreamIter)
    │ contains Arc<>
    ▼
ParquetRowGroupDataset             ← Rust struct; immutable after construction
    ├── file_uris: Arc<Vec<String>>       ← S3 URIs (sorted by list order)
    ├── extents: Arc<Vec<RgExtent>>       ← byte range per RG (start, length, num_rows)
    ├── file_metadata: Arc<Vec<Arc<ParquetMetaData>>>  ← per-file footer (empty in Raw fast path)
    ├── col_indices: Arc<Option<Vec<usize>>>           ← column projection
    └── decode_mode: ParquetDecodeMode
    │
    ├── get(idx) → Bytes     ← single range GET; dispatches to get_raw() or get_arrow_ipc()
    │
    │ construction calls ──►
    │
    ├── parquet_file_cache::get_or_fetch(uri, footer_cap)
    │       └── DashMap<String, Arc<CachedFileMeta>>  (process-global, shared via Arc<>)
    │               └── CachedFileMeta { parquet_meta: Arc<ParquetMetaData>, rg_row_offsets }
    │
    └── parquet_index::global()
            └── OnceLock<ParquetIndex>  (process-global singleton)
                    └── DashMap<String, Vec<RgIndexEntry>>  (uri → [(start, length, num_rows)])
```

### Concurrency model

- **Construction** (`new()`): single call to `run_on_global_rt(build_extents())`. Footer
  fetches inside `build_extents` run concurrently via `futures::stream::buffer_unordered`.
- **Iteration** (`__iter__`): Tokio task spawned by `PyParquetStreamLoader::__iter__`; bounded
  `mpsc::channel(concurrency)` provides backpressure. Python blocks on `rx.blocking_recv()`.
- **Global index writes**: protected by `DashMap` shard-level RwLock (lock-free reads for
  already-indexed files).
- **File cache writes**: `tokio::sync::OnceCell` per URI prevents duplicate fetches when
  multiple concurrent epoch-1 constructors race on the same file.

### Thread safety guarantees

- `ParquetRowGroupDataset` is `Send + Sync` — multiple Python threads can share one instance.
- `parquet_index::global()` returns `&'static ParquetIndex`; `DashMap` is `Sync`.
- `parquet_file_cache` uses `DashMap<String, Arc<OnceCell<Arc<CachedFileMeta>>>>` — concurrent
  callers for the same URI block on the `OnceCell` until the first fetch completes, then all
  share the same `Arc`.

---

*For questions, see the [PYTHON_API_GUIDE.md](PYTHON_API_GUIDE.md) parquet section, or the
inline Rust docs in `src/data_loader/parquet_rg.rs` and `src/data_loader/parquet_index.rs`.*
