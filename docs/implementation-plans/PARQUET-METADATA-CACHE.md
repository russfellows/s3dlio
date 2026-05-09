# Parquet Metadata Cache (`parquet_file_cache`)

**Status**: Planned  
**Target version**: v0.9.98  
**Authors**: Russ Fellows / GitHub Copilot  
**Date**: 2026-05-06

---

## Problem Statement

Every call to `create_async_loader(uri, {format: "parquet", ...})` constructs a
`ParquetRowGroupDataset`, which unconditionally does:

1. `ListObjects` — find all `.parquet` files under the URI prefix
2. `HeadObject` × N — stat each file to get its size
3. Range `GET` × N — fetch the last `footer_cap` (4 MiB) bytes of every file
4. Thrift decode × N — parse each Parquet footer into `ParquetMetaData`

For a DLRM workload with 64 files, that is **64 stat + 64 range GETs** every time
a loader is created.  In the DLIO benchmark with `read_threads = 4` workers and
`ON_DEMAND` disabled, each worker creates one loader per unique file per epoch.
With a fresh process, this means **256+ HTTP round-trips just to open files before
a single byte of training data is fetched**.

A second problem is the Python-side `_build_rg_offsets()` method, which *also*
reads each file's footer using PyArrow + `_S3RangeFile`.  This duplicates the
same stat + range-GET work that Rust already did.

---

## What the Parquet Footer Actually Contains

A Parquet footer (Thrift-encoded `FileMetaData`) contains:

| Field | Size (DLRM) | Needed? |
|---|---|---|
| Schema (column names, types) | ~2 KB | Yes, for ArrowIpc decode |
| Row-group list: offset, length, num_rows, col stats, encodings | ~2.6 MiB for 123 RGs × ~21 KB/RG | Mostly yes (offsets + num_rows); stats no |
| Created-by string | tiny | No |
| Key-value metadata | tiny | No |

For the common **Raw / `decode_mode=none`** path:
- We need: `(start: u64, length: u64, num_rows: i64)` per row group
- We **do not** need: column statistics, encodings, dictionary info, created-by

For the **ArrowIpc** path:
- We need the full `Arc<ParquetMetaData>` so `ParquetRecordBatchStreamBuilder`
  can build its projection mask and row-group reader.

Since we must support ArrowIpc, we need to retain the full `ParquetMetaData`.
But we can cache it so it is fetched exactly once per file URI per process lifetime.

---

## Design

### Data Structures

```rust
/// Everything we derive from one file's Parquet footer, cached for
/// the process lifetime.
pub struct CachedFileMeta {
    /// Full parquet metadata — required for ArrowIpc decode path.
    /// Also contains column stats if callers ever need them.
    pub parquet_meta: Arc<ParquetMetaData>,

    /// Cumulative row-count offsets across row groups.
    ///
    /// `rg_row_offsets[i]` = total rows before row group i.
    /// `rg_row_offsets[num_rgs]` = total rows in file.
    ///
    /// Length = num_row_groups + 1.
    ///
    /// Used by Python reader to map `sample_index → rg_idx` without a
    /// second footer fetch.
    pub rg_row_offsets: Vec<i64>,
}
```

### Cache

```rust
// In src/data_loader/parquet_file_cache.rs (new file)

static CACHE: OnceLock<DashMap<String, Arc<CachedFileMeta>>> = OnceLock::new();

fn cache() -> &'static DashMap<String, Arc<CachedFileMeta>> {
    CACHE.get_or_init(DashMap::new)
}

/// Returns cached metadata for `uri`, fetching and caching on first call.
/// Concurrent callers for the same URI will both issue a fetch; the second
/// result is discarded (last-writer-wins, both get the same data).
pub async fn get_or_fetch(uri: &str, footer_cap: u64) -> anyhow::Result<Arc<CachedFileMeta>> { ... }

/// Evict a single URI.  Useful for testing or when a file is known to have
/// been rewritten.
pub fn invalidate(uri: &str) { ... }

/// Clear the entire cache.
pub fn clear() { ... }

/// Return the number of cached entries.
pub fn len() -> usize { ... }
```

**Concurrency note**: DashMap is sharded and lock-free for reads.  On the first
call for a URI, two threads may both issue a footer GET and one result is thrown
away.  This is safe — both fetches return identical bytes.  A more complex
"in-flight deduplication" with `tokio::sync::OnceCell` is possible but not
worth the complexity for an init path.

### Integration into `build_extents`

Replace the current per-file `stat + GET + parse` sequence:

```rust
// Before: unconditional stat + GET per file
let stat  = stat_object_uri_async(&uri).await?;
let bytes = get_object_range_uri_async(&uri, offset, len).await?;
let meta  = parse_footer(stat.size, &bytes)?;

// After: cache lookup
let cached = parquet_file_cache::get_or_fetch(&uri, footer_cap).await?;
let meta = &cached.parquet_meta;
```

The `rg_byte_extent` computation over the cached metadata remains unchanged —
it is O(num_columns) arithmetic, not I/O.

### New Python API

```python
import s3dlio

# Returns list[int] of length num_row_groups+1.
# Indexes: offsets[i] = first sample row in rg i, offsets[-1] = total rows.
offsets = s3dlio.parquet_rg_row_offsets("s3://bucket/path/to/file.parquet")
```

Rust side:
```rust
#[pyfunction]
pub fn parquet_rg_row_offsets(uri: &str) -> PyResult<Vec<i64>> {
    let cached = run_on_global_rt(
        parquet_file_cache::get_or_fetch(uri, DEFAULT_FOOTER_CAP as u64)
    ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cached.rg_row_offsets.clone())
}
```

This replaces the Python `_build_rg_offsets()` method entirely:

```python
# Before (2 × HTTP round-trips per file via _S3RangeFile + PyArrow)
rg_offsets = self._build_rg_offsets(uri, filename)

# After (zero HTTP — data already in Rust cache from loader construction)
rg_offsets = s3dlio.parquet_rg_row_offsets(uri)
```

---

## What Gets Cached vs. Recomputed

| Item | Cached? | Why |
|---|---|---|
| `ParquetMetaData` (full Thrift struct) | ✅ Per URI | Expensive to fetch; immutable for a given file |
| `rg_row_offsets: Vec<i64>` | ✅ Per URI | Cheap to derive once; avoids Python footer read |
| `RgExtent` list (column-specific offsets) | ❌ Per `ParquetRowGroupDataset` instance | Depends on `col_indices`; pure arithmetic, O(µs) |
| File listing (which URIs exist under a prefix) | ❌ | ListObjects is fast; prefix scope varies |

This means `ParquetRowGroupDataset::new()` for a file that is already cached does:
- `ListObjects` (one call per prefix, typically one call per worker startup) — can't cache without knowing the prefix scope
- `rg_byte_extent` arithmetic over cached metadata — O(num_rgs × num_cols), microseconds
- **Zero network I/O** for metadata

---

## Expected Performance Impact

| Scenario | Before | After |
|---|---|---|
| First `create_async_loader` for N files | N stat + N range GETs | N stat + N range GETs (same) |
| Second call (same process, same files) | N stat + N range GETs | 0 stat + 0 range GETs |
| Python `_build_rg_offsets` per file | 1 stat + 2–3 range GETs (PyArrow) | 0 (deleted, use Rust cache) |
| Epoch 2+ (same files) | N stat + N range GETs | 0 |

DLRM, 64 files, 4 read_threads, 2 epochs = 512 stat + 512 range GETs today.
After: 64 stat + 64 range GETs on epoch 1, zero on epoch 2+.

The Python `_build_rg_offsets` adds ~2–3 more range GETs per file on top.
Total current: ~640 HTTP calls for metadata alone.  After: 64.

---

## Files Changed

| File | Change |
|---|---|
| `src/data_loader/parquet_file_cache.rs` | **New** — static cache module |
| `src/data_loader/mod.rs` | Add `pub mod parquet_file_cache;` |
| `src/data_loader/parquet_rg.rs` | Replace `build_extents` inner fetch loop with cache lookup |
| `src/python_api/python_aiml_api.rs` | Add `parquet_rg_row_offsets` pyfunction |
| `src/python_api/python_core_api.rs` | Register new function in module |
| `dlio_benchmark/.../parquet_reader_s3dlio.py` | Replace `_build_rg_offsets()` call with `s3dlio.parquet_rg_row_offsets()` |

---

## Non-Goals

- Persisting cache to disk (future: `.parquet_meta_cache/` directory)
- Partial column-chunk caching (future: column-specific `RgExtent` caching)
- TTL / LRU eviction (files don't change during a benchmark run; process lifetime is fine)
- Cross-process shared cache (each MPI rank caches independently; network is fast enough for 64 files)
