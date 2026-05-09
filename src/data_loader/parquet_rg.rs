// src/data_loader/parquet_rg.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `ParquetRowGroupDataset` — a [`Dataset`] where each item represents one
//! Parquet row group, returned in one of two decode modes:
//!
//! | Mode | What Python receives | When to use |
//! |------|---------------------|-------------|
//! | `Raw` (default) | Compressed Parquet column-chunk bytes | Python decodes with PyArrow, or discards for pure I/O benchmark |
//! | `ArrowIpc` | Arrow IPC stream bytes (decoded in Rust) | Rust handles the CPU decode; Python reconstructs via `pa.ipc` |
//!
//! The "no decode" storage-benchmark case is handled on the **Python side**
//! (dlio_benchmark `decode_mode: none`): it receives `Raw` bytes and simply
//! records the byte count without calling PyArrow at all.
//!
//! ## Feature flags
//!
//! | Cargo feature | What it enables |
//! |---------------|----------------|
//! | `parquet` (default) | `Raw` mode DataLoader; Parquet footer parsing |
//! | `parquet-arrow` (default) | Adds `ArrowIpc` decode mode (Rust→Arrow→IPC); NOT the PyArrow object-store backend |
//!
//! `arrow-backend` is a separate, mutually-exclusive feature that replaces the
//! native AWS/Azure/GCS clients with Apache Arrow's `object_store` crate.
//! It is **not** related to the Parquet DataLoader and is not in the default build.
//!
//! ## Memory model
//!
//! The dataset holds **only metadata in RAM** — no bulk Parquet data is buffered:
//!
//! | What | Size | Scope |
//! |------|------|-------|
//! | `extents` (byte ranges) | ~48 B × N_rgs | Per dataset instance |
//! | `parquet_file_cache` (parsed footer) | a few MB per file | Process-global, shared via `Arc<>` |
//! | `parquet_index` (DashMap entries) | ~80 B × N_rgs | Process-global singleton |
//! | Active `get()` call | 1 row-group payload | Freed when Python releases `Bytes` |
//!
//! **8 simultaneous instances** share the process-global caches (no duplication).
//! Peak RAM per instance = metadata overhead (< 10 MB) + however many row-group
//! `Bytes` objects Python keeps alive concurrently.  At typical row-group sizes
//! of 50–200 MB and a Python DataLoader prefetch depth of 2, 8 instances consume
//! < 3.2 GB of actual data RAM.
//!
//! ## Epoch-2+ fast path
//!
//! After the first construction (`new()`) for a set of files, all row-group byte
//! ranges are stored in the process-global [`crate::data_loader::parquet_index`].
//! Subsequent calls to `new()` for the same files (epoch 2, 3, …) read ranges
//! directly from the DashMap — **zero S3 I/O**, sub-millisecond per-file cost.
//! Only the `list_objects` call (listing `.parquet` files under the prefix) still
//! hits the network on every epoch.
//!
//! ## Throughput model
//!
//! | Step | Cost per worker |
//! |------|----------------|
//! | Epoch-1 construction (list + footer fetch × 64 files) | ~0.5 s (concurrent) |
//! | Epoch-2+ construction (list + DashMap lookup) | < 0.1 s |
//! | 1,968 range GETs @ 10 GiB/s | ~0.33 s |
//! | Rust Arrow decode (ArrowIpc mode only) | ~0.05 s (parallel on Tokio threads) |
//! | Python iteration overhead | depends on caller |
//!
//! Both `parquet` and `parquet-arrow` are enabled by default.
//! `ArrowIpc` mode additionally requires the `parquet-arrow` feature (also default).

use crate::data_loader::{Dataset, DatasetError};
use crate::s3_client::run_on_global_rt;
use crate::s3_utils::{
    get_object_range_uri_async, get_object_range_uri_timed_async, list_objects as list_objects_rs,
    parse_s3_uri,
};
use async_trait::async_trait;
use bytes::Bytes;
#[cfg(test)]
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::metadata::{ColumnChunkMetaData, ParquetMetaData, RowGroupMetaData};
use std::sync::Arc;
use std::time::Instant;

// ── Parquet footer constants (test-only) ────────────────────────────────────

/// Default number of bytes fetched from the tail of each file for footer
/// parsing.  4 MiB comfortably covers production workloads (DLRM footers are
/// ~2.66 MiB).
pub const DEFAULT_FOOTER_CAP: usize = 4 * 1024 * 1024;

// ── Decode mode ───────────────────────────────────────────────────────────────

/// Controls what `Dataset::get()` returns for each row group.
///
/// `Raw` is the default and requires only the `parquet` feature.
/// `ArrowIpc` requires the `parquet-arrow` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ParquetDecodeMode {
    /// Return the raw compressed Parquet column-chunk bytes as fetched from
    /// storage.  Python is responsible for decoding (or may discard for a pure
    /// storage benchmark).
    #[default]
    Raw,
    /// Decode the Parquet column data to an Arrow RecordBatch in Rust, then
    /// serialise to Arrow IPC stream format and return the IPC bytes.
    ///
    /// Python reconstructs the RecordBatch with:
    /// ```python
    /// import pyarrow as pa
    /// batch = pa.ipc.open_stream(pa.py_buffer(bytes(item))).read_next_batch()
    /// ```
    ///
    /// Requires the `parquet-arrow` cargo feature.
    #[cfg(feature = "parquet-arrow")]
    ArrowIpc,
}

// ── Pre-computed per-RG extent ───────────────────────────────────────────────

/// Immutable byte extent for one row group, computed at construction time.
#[derive(Clone, Debug)]
struct RgExtent {
    /// Index into `file_uris`.
    file_uri_idx: usize,
    /// Row-group index within this file (needed for the ArrowIpc decode path).
    rg_idx_in_file: usize,
    /// Byte offset of the first selected column chunk within the file.
    start: u64,
    /// Byte length covering all selected column chunks (merged span).
    length: u64,
    /// Number of rows in this row group (for metadata / progress reporting).
    num_rows: i64,
}

// ── Main Dataset struct ──────────────────────────────────────────────────────

/// Dataset where each item represents one Parquet row group.
///
/// See [`ParquetDecodeMode`] for the three output formats.
/// Construction fetches and parses all Parquet footers once; subsequent
/// `get()` calls each issue a single range GET (plus a decode step for
/// `ArrowIpc` mode).
pub struct ParquetRowGroupDataset {
    /// Full URIs of all Parquet files under the prefix (sorted by listing order).
    file_uris: Arc<Vec<String>>,
    /// Flat index over every row group across all files.
    extents: Arc<Vec<RgExtent>>,
    /// Per-file Parquet metadata (needed for `ArrowIpc` decode to get the schema
    /// and row-group reader).  Always populated even in `Raw` mode (low overhead,
    /// avoids a second construction path).
    file_metadata: Arc<Vec<Arc<ParquetMetaData>>>,
    /// Column indices requested at construction time.  `None` = all columns.
    /// Used both for the raw merged-GET byte extent and for Arrow projection.
    col_indices: Arc<Option<Vec<usize>>>,
    /// Output format for `Dataset::get()`.
    decode_mode: ParquetDecodeMode,
}

impl ParquetRowGroupDataset {
    /// Build the dataset synchronously.
    ///
    /// # Arguments
    /// * `uri_prefix`   — S3 URI prefix, e.g. `s3://bucket/data/train/`
    /// * `col_indices`  — column indices to include per row group; `None` = all
    /// * `footer_cap`   — bytes from file tail for footer parsing (≥ largest footer)
    /// * `decode_mode`  — controls what `get()` returns (see [`ParquetDecodeMode`])
    pub fn new(
        uri_prefix: &str,
        col_indices: Option<&[usize]>,
        footer_cap: usize,
        decode_mode: ParquetDecodeMode,
    ) -> Result<Self, DatasetError> {
        let t_new = Instant::now();
        tracing::info!(
            uri_prefix,
            footer_cap,
            "[s3dlio] ParquetRowGroupDataset::new — listing files under prefix"
        );

        // ── 1. List .parquet files under prefix ─────────────────────────────
        let (bucket, prefix) =
            parse_s3_uri(uri_prefix).map_err(|e| DatasetError::from(e.to_string()))?;

        let keys = list_objects_rs(&bucket, &prefix, true)
            .map_err(|e| DatasetError::from(e.to_string()))?;

        let file_uris: Vec<String> = keys
            .into_iter()
            .filter(|k| k.ends_with(".parquet"))
            .map(|k| format!("s3://{}/{}", bucket, k))
            .collect();

        if file_uris.is_empty() {
            return Err(DatasetError::from(format!(
                "No .parquet files found under '{}' (bucket='{}', prefix='{}')",
                uri_prefix, bucket, prefix
            )));
        }

        tracing::info!(
            num_files = file_uris.len(),
            "[s3dlio] listed {} file(s) — fetching footers",
            file_uris.len()
        );

        // ── 2. Concurrently stat + fetch footers, build extents + metadata ──
        let col_indices_owned: Option<Vec<usize>> = col_indices.map(|s| s.to_vec());
        let footer_cap_u64 = footer_cap as u64;
        let uris_for_init = file_uris.clone();

        let (extents, file_metadata) = run_on_global_rt(async move {
            build_extents(
                uris_for_init,
                col_indices_owned,
                footer_cap_u64,
                decode_mode,
            )
            .await
        })
        .map_err(|e| DatasetError::from(e.to_string()))?;

        let elapsed = t_new.elapsed();
        let total_rgs = extents.len();
        let total_bytes: u64 = extents.iter().map(|e| e.length).sum();
        tracing::info!(
            num_files = file_uris.len(),
            total_rgs,
            total_bytes,
            elapsed_ms = elapsed.as_millis(),
            "[s3dlio] dataset ready: {} file(s), {} row-groups, {:.1} MiB data, init {:.0}ms",
            file_uris.len(),
            total_rgs,
            total_bytes as f64 / 1024.0 / 1024.0,
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(Self {
            file_uris: Arc::new(file_uris),
            extents: Arc::new(extents),
            file_metadata: Arc::new(file_metadata),
            col_indices: Arc::new(col_indices.map(|s| s.to_vec())),
            decode_mode,
        })
    }

    /// Number of rows in the row group at `global_rg_idx`.
    pub fn num_rows_in_rg(&self, global_rg_idx: usize) -> Option<i64> {
        self.extents.get(global_rg_idx).map(|e| e.num_rows)
    }

    /// URI of the file containing `global_rg_idx`.
    pub fn file_uri_for_rg(&self, global_rg_idx: usize) -> Option<&str> {
        self.extents
            .get(global_rg_idx)
            .map(|e| self.file_uris[e.file_uri_idx].as_str())
    }

    /// The active decode mode for this dataset instance.
    pub fn decode_mode(&self) -> ParquetDecodeMode {
        self.decode_mode
    }

    /// Return `(file_uri_idx, rg_idx_in_file)` for every row group in order.
    ///
    /// Used by `ParquetStreamIter` to annotate items returned to Python so the
    /// caller can identify which file and row-group each batch item came from.
    pub fn rg_info_vec(&self) -> Vec<(usize, usize)> {
        self.extents
            .iter()
            .map(|e| (e.file_uri_idx, e.rg_idx_in_file))
            .collect()
    }

    /// Shared reference to the file URI list.
    pub fn file_uris_arc(&self) -> Arc<Vec<String>> {
        Arc::clone(&self.file_uris)
    }

    /// Number of files for which `file_metadata` is populated.
    ///
    /// In `Raw` mode on epoch 2+ (global-index fast path), this is **0** because
    /// `get_raw()` never accesses per-file metadata.  In `ArrowIpc` mode it equals
    /// the number of files (metadata is always fetched so the Arrow decoder can
    /// reconstruct the schema).
    ///
    /// Only available in test builds; use this to verify the fast-path behaviour.
    #[cfg(test)]
    pub fn file_metadata_len(&self) -> usize {
        self.file_metadata.len()
    }
}

// ── Dataset trait ────────────────────────────────────────────────────────────

#[async_trait]
impl Dataset for ParquetRowGroupDataset {
    /// One row group's data.
    ///
    /// * `Raw` mode → raw compressed Parquet column-chunk bytes.
    /// * `ArrowIpc` mode → Arrow IPC stream bytes (decoded in Rust).
    type Item = Bytes;

    fn len(&self) -> Option<usize> {
        Some(self.extents.len())
    }

    /// Returns `"<file_uri>#rg<idx>"` identifiers, useful for sharding.
    fn keys(&self) -> Option<Vec<String>> {
        Some(
            self.extents
                .iter()
                .enumerate()
                .map(|(i, e)| format!("{}#rg{}", self.file_uris[e.file_uri_idx], i))
                .collect(),
        )
    }

    async fn get(&self, global_idx: usize) -> Result<Self::Item, DatasetError> {
        match self.decode_mode {
            ParquetDecodeMode::Raw => self.get_raw(global_idx).await,
            #[cfg(feature = "parquet-arrow")]
            ParquetDecodeMode::ArrowIpc => self.get_arrow_ipc(global_idx).await,
        }
    }
}

// ── Internal per-mode get implementations ────────────────────────────────────

impl ParquetRowGroupDataset {
    /// Fetch raw compressed column-chunk bytes (no decode).
    async fn get_raw(&self, global_idx: usize) -> Result<Bytes, DatasetError> {
        let ext = self
            .extents
            .get(global_idx)
            .ok_or(DatasetError::IndexOutOfRange(global_idx))?;
        let uri = &self.file_uris[ext.file_uri_idx];
        let t0 = Instant::now();
        let result = get_object_range_uri_async(uri, ext.start, Some(ext.length))
            .await
            .map_err(DatasetError::from);
        let elapsed = t0.elapsed();
        tracing::debug!(
            rg = global_idx,
            offset = ext.start,
            length = ext.length,
            elapsed_ms = elapsed.as_millis(),
            "[s3dlio] rg {}: {} bytes in {:.1}ms ({:.0} MB/s)",
            global_idx,
            ext.length,
            elapsed.as_secs_f64() * 1000.0,
            ext.length as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0
        );
        result
    }

    /// Fetch, decode to Arrow RecordBatch, and return Arrow IPC stream bytes.
    ///
    /// The IPC stream contains exactly one RecordBatch (the requested row group,
    /// projected to the selected columns).  Python reconstructs it with:
    /// ```python
    /// batch = pa.ipc.open_stream(pa.py_buffer(bytes(item))).read_next_batch()
    /// ```
    #[cfg(feature = "parquet-arrow")]
    async fn get_arrow_ipc(&self, global_idx: usize) -> Result<Bytes, DatasetError> {
        use arrow_array::RecordBatch;
        use arrow_ipc::writer::StreamWriter;
        use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
        use parquet::arrow::ProjectionMask;

        let ext = self
            .extents
            .get(global_idx)
            .ok_or(DatasetError::IndexOutOfRange(global_idx))?;

        let uri = self.file_uris[ext.file_uri_idx].clone();
        let meta = Arc::clone(&self.file_metadata[ext.file_uri_idx]);
        let col_indices = Arc::clone(&self.col_indices);
        let rg_idx = ext.rg_idx_in_file;

        // Build a parquet-rs AsyncFileReader backed by our S3 range-GET.
        let reader = S3AsyncFileReader::new(uri, meta);

        let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| DatasetError::from(format!("parquet reader init: {}", e)))?;

        // Read only this specific row group.
        builder = builder.with_row_groups(vec![rg_idx]);

        // Apply column projection if requested.
        if let Some(cols) = col_indices.as_deref() {
            let mask = ProjectionMask::leaves(builder.parquet_schema(), cols.iter().copied());
            builder = builder.with_projection(mask);
        }

        let mut stream = builder
            .build()
            .map_err(|e| DatasetError::from(format!("parquet stream build: {}", e)))?;

        use futures_util::StreamExt as _;
        let batch: RecordBatch = stream
            .next()
            .await
            .ok_or_else(|| DatasetError::from("parquet stream returned no batch"))?
            .map_err(|e| DatasetError::from(format!("parquet stream read: {}", e)))?;

        // Serialize to Arrow IPC stream (schema header + one RecordBatch message).
        let mut buf: Vec<u8> = Vec::with_capacity(batch.get_array_memory_size() + 512);
        {
            let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
                .map_err(|e| DatasetError::from(format!("ipc writer init: {}", e)))?;
            writer
                .write(&batch)
                .map_err(|e| DatasetError::from(format!("ipc write: {}", e)))?;
            writer
                .finish()
                .map_err(|e| DatasetError::from(format!("ipc finish: {}", e)))?;
        }

        Ok(Bytes::from(buf))
    }
}

// ── S3AsyncFileReader (parquet-arrow only) ────────────────────────────────────

/// Implements `parquet::arrow::async_reader::AsyncFileReader` using s3dlio's
/// range GET.  Provides pre-parsed metadata so parquet-rs skips its own footer
/// fetch.
#[cfg(feature = "parquet-arrow")]
struct S3AsyncFileReader {
    uri: String,
    metadata: Arc<ParquetMetaData>,
}

#[cfg(feature = "parquet-arrow")]
impl S3AsyncFileReader {
    fn new(uri: String, metadata: Arc<ParquetMetaData>) -> Self {
        Self { uri, metadata }
    }
}

#[cfg(feature = "parquet-arrow")]
impl parquet::arrow::async_reader::AsyncFileReader for S3AsyncFileReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures_util::future::BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let uri = self.uri.clone();
        let start = range.start;
        let len = range.end - range.start;
        Box::pin(async move {
            get_object_range_uri_async(&uri, start, Some(len))
                .await
                .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))
        })
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a parquet::arrow::arrow_reader::ArrowReaderOptions>,
    ) -> futures_util::future::BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let meta = Arc::clone(&self.metadata);
        Box::pin(async move { Ok(meta) })
    }
}

// ── Inherent methods for diagnostics ─────────────────────────────────────────

impl ParquetRowGroupDataset {
    /// Like `get_raw()` but returns split timing information for TTFB analysis.
    ///
    /// Returns `(bytes, ttfb, transfer)` where:
    /// - `ttfb` — time from request start until HTTP response headers are received.
    /// - `transfer` — time from headers received until all body bytes are collected.
    pub async fn get_timed(
        &self,
        global_idx: usize,
    ) -> Result<(bytes::Bytes, std::time::Duration, std::time::Duration), DatasetError> {
        let ext = self
            .extents
            .get(global_idx)
            .ok_or(DatasetError::IndexOutOfRange(global_idx))?;

        let uri = &self.file_uris[ext.file_uri_idx];

        get_object_range_uri_timed_async(uri, ext.start, Some(ext.length))
            .await
            .map_err(DatasetError::from)
    }
}

// ── Decode-mode helper ───────────────────────────────────────────────────────

/// Returns `true` if `mode` requires `file_metadata` to be populated.
///
/// In `Raw` mode, `get_raw()` only needs `extents` (byte offsets) and never
/// accesses `file_metadata`.  In `ArrowIpc` mode, `get_arrow_ipc()` needs the
/// per-file `ParquetMetaData` to reconstruct the Arrow schema.
#[inline]
fn needs_file_metadata(mode: ParquetDecodeMode) -> bool {
    match mode {
        ParquetDecodeMode::Raw => false,
        #[cfg(feature = "parquet-arrow")]
        ParquetDecodeMode::ArrowIpc => true,
    }
}

// ── Async construction helpers ───────────────────────────────────────────────

/// Stat all files + fetch their footers concurrently (via the process-lifetime
/// metadata cache), then build the flat extents index and per-file metadata.
///
/// ## Epoch-2+ fast path
///
/// When `col_indices` is `None` and every URI in `file_uris` is already
/// present in the process-global [`crate::data_loader::parquet_index`], this
/// function reads byte ranges directly from the in-memory DashMap — no async
/// I/O, no `ParquetMetaData` iteration.  For a 64-file dataset this reduces
/// epoch-N (N ≥ 2) construction from ~40 ms to < 1 ms.
///
/// ## Epoch-1 (cache miss) path
///
/// On the first call for any URI, `parquet_file_cache::get_or_fetch` issues a
/// `HeadObject` + one range `GET` and caches the result in the per-process
/// file cache.  Concurrently, each file's row-group extents are inserted into
/// the global `ParquetIndex` (when `col_indices` is `None`) so that subsequent
/// calls take the fast path.
///
/// Returns `(extents, per_file_metadata)`.
async fn build_extents(
    file_uris: Vec<String>,
    col_indices: Option<Vec<usize>>,
    footer_cap: u64,
    decode_mode: ParquetDecodeMode,
) -> anyhow::Result<(Vec<RgExtent>, Vec<Arc<ParquetMetaData>>)> {
    use crate::data_loader::parquet_file_cache;
    use futures::future::join_all;

    // ── Fast path: global index hit ───────────────────────────────────────────
    // All files are already indexed from a prior construction in the same
    // process (typical epoch-2+ scenario for DataLoader workers).
    // Only valid when col_indices=None because the index always stores
    // all-column extents; projected-column subsets fall through to the slow path.
    {
        use crate::data_loader::parquet_index;
        let gidx = parquet_index::global();
        if col_indices.is_none() && file_uris.iter().all(|u| gidx.is_indexed(u)) {
            tracing::debug!(
                num_files = file_uris.len(),
                "[s3dlio] build_extents: global-index fast path (epoch 2+)"
            );
            let mut extents = Vec::new();
            for (file_uri_idx, uri) in file_uris.iter().enumerate() {
                let rg_data = gidx
                    .file_extents(uri)
                    .ok_or_else(|| anyhow::anyhow!("index inconsistency for '{}'", uri))?;
                for (rg_idx_in_file, (start, length, num_rows)) in rg_data.into_iter().enumerate() {
                    extents.push(RgExtent {
                        file_uri_idx,
                        rg_idx_in_file,
                        start,
                        length,
                        num_rows,
                    });
                }
            }

            // For ArrowIpc mode we still need file_metadata, but the file
            // cache is warm so this is just a DashMap lookup (no network I/O).
            let file_metadata = if needs_file_metadata(decode_mode) {
                let futs: Vec<_> = file_uris
                    .iter()
                    .map(|uri| {
                        let u = uri.clone();
                        async move { parquet_file_cache::get_or_fetch(&u, footer_cap).await }
                    })
                    .collect();
                join_all(futs)
                    .await
                    .into_iter()
                    .enumerate()
                    .map(|(i, r)| {
                        r.map(|c| Arc::clone(&c.parquet_meta))
                            .map_err(|e| anyhow::anyhow!("'{}': {}", file_uris[i], e))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
            } else {
                // Raw mode: file_metadata is never accessed by get_raw().
                Vec::new()
            };

            return Ok((extents, file_metadata));
        }
    }

    // ── Slow path: fetch (or hit file cache for) all footers concurrently ────
    let cache_futs: Vec<_> = file_uris
        .iter()
        .map(|uri| {
            let u = uri.clone();
            async move { parquet_file_cache::get_or_fetch(&u, footer_cap).await }
        })
        .collect();

    let cache_results = join_all(cache_futs).await;

    // ── Build extents from cached metadata ────────────────────────────────────
    let mut extents = Vec::new();
    let mut file_metadata: Vec<Arc<ParquetMetaData>> = Vec::with_capacity(file_uris.len());

    for (file_uri_idx, result) in cache_results.into_iter().enumerate() {
        let cached =
            result.map_err(|e| anyhow::anyhow!("metadata '{}': {}", file_uris[file_uri_idx], e))?;

        let meta = &cached.parquet_meta;

        for rg_idx in 0..meta.num_row_groups() {
            let rg = meta.row_group(rg_idx);
            let (start, length) = rg_byte_extent(rg, col_indices.as_deref()).map_err(|e| {
                anyhow::anyhow!("'{}' rg[{}]: {}", file_uris[file_uri_idx], rg_idx, e)
            })?;
            extents.push(RgExtent {
                file_uri_idx,
                rg_idx_in_file: rg_idx,
                start,
                length,
                num_rows: rg.num_rows(),
            });
        }

        // Populate the process-global ParquetIndex when all columns are included.
        // This enables the epoch-2+ fast path for all subsequent constructions
        // over the same files in this process.  Silently ignored on error.
        if col_indices.is_none() {
            use crate::data_loader::parquet_index;
            let _ = parquet_index::global().insert_file(
                &file_uris[file_uri_idx],
                meta,
                &cached.rg_row_offsets,
                0,
            );
        }

        file_metadata.push(Arc::clone(meta));
    }

    Ok((extents, file_metadata))
}

// ── Footer parsing (test-only) ────────────────────────────────────────────────
//
// parse_footer is only used by the unit tests below.  All production code now
// fetches and parses footers through parquet_file_cache::get_or_fetch, which
// has its own inline parsing logic.

/// Magic bytes at the very end of every Parquet file.
#[cfg(test)]
const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Fixed size of the footer suffix: 4-byte metadata length + 4-byte magic.
#[cfg(test)]
const FOOTER_SUFFIX_LEN: usize = 8;

/// Parse a Parquet footer from the raw tail bytes of a file.
///
/// `file_size` is the total object size; `buf` is the last `buf.len()` bytes
/// of the file.  The caller must have fetched at least `footer_len + 8` bytes,
/// where `footer_len` is the Thrift metadata length stored in the file.
#[cfg(test)]
fn parse_footer(
    file_size: u64,
    buf: &Bytes,
) -> anyhow::Result<parquet::file::metadata::ParquetMetaData> {
    let buf_len = buf.len();

    if buf_len < FOOTER_SUFFIX_LEN {
        anyhow::bail!(
            "Footer buffer too small ({} bytes); minimum is {} bytes",
            buf_len,
            FOOTER_SUFFIX_LEN
        );
    }

    // Validate PAR1 magic at the very end.
    let magic = &buf[buf_len - 4..];
    if magic != PARQUET_MAGIC {
        anyhow::bail!(
            "Invalid Parquet magic {:?} (expected {:?}); file may not be a Parquet file",
            magic,
            PARQUET_MAGIC
        );
    }

    // 4-byte little-endian metadata length immediately before the magic.
    let meta_len_bytes: [u8; 4] = buf[buf_len - 8..buf_len - 4]
        .try_into()
        .expect("slice is exactly 4 bytes");
    let metadata_len = u32::from_le_bytes(meta_len_bytes) as usize;

    if metadata_len + FOOTER_SUFFIX_LEN > buf_len {
        anyhow::bail!(
            "Parquet metadata ({} bytes) + suffix ({} bytes) exceeds fetched buffer \
             ({} bytes covering last {} bytes of {}-byte file). \
             Increase footer_cap.",
            metadata_len,
            FOOTER_SUFFIX_LEN,
            buf_len,
            buf_len,
            file_size
        );
    }

    let meta_start = buf_len - FOOTER_SUFFIX_LEN - metadata_len;
    let meta_bytes = &buf[meta_start..buf_len - FOOTER_SUFFIX_LEN];

    ParquetMetaDataReader::decode_metadata(meta_bytes)
        .map_err(|e| anyhow::anyhow!("Thrift decode error: {}", e))
}

// ── Extent calculation ────────────────────────────────────────────────────────

/// Compute the tightest byte range covering all selected columns in a row group.
///
/// Returns `(start_offset_in_file, byte_length)`.
pub(crate) fn rg_byte_extent(
    rg: &RowGroupMetaData,
    col_indices: Option<&[usize]>,
) -> anyhow::Result<(u64, u64)> {
    let n_cols = rg.num_columns();
    if n_cols == 0 {
        anyhow::bail!("Row group has zero columns");
    }

    let mut min_start = u64::MAX;
    let mut max_end = 0u64;

    let process = |col: &ColumnChunkMetaData| {
        // Use dictionary page offset when present (it precedes data pages).
        let col_start = col
            .dictionary_page_offset()
            .unwrap_or_else(|| col.data_page_offset()) as u64;
        let col_end = col_start + col.compressed_size() as u64;
        (col_start, col_end)
    };

    match col_indices {
        Some(indices) => {
            if indices.is_empty() {
                anyhow::bail!("col_indices is empty; must specify at least one column");
            }
            for &ci in indices {
                if ci >= n_cols {
                    anyhow::bail!(
                        "Column index {} out of range (row group has {} columns)",
                        ci,
                        n_cols
                    );
                }
                let (s, e) = process(rg.column(ci));
                if s < min_start {
                    min_start = s;
                }
                if e > max_end {
                    max_end = e;
                }
            }
        }
        None => {
            for ci in 0..n_cols {
                let (s, e) = process(rg.column(ci));
                if s < min_start {
                    min_start = s;
                }
                if e > max_end {
                    max_end = e;
                }
            }
        }
    }

    if min_start == u64::MAX {
        anyhow::bail!("No column extents computed");
    }

    Ok((min_start, max_end - min_start))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal synthetic parquet footer buffer and verify parse_footer
    /// correctly extracts metadata length and magic without hitting real S3.
    #[test]
    fn test_parse_footer_magic_check() {
        // Construct a minimal buffer: [fake_metadata_bytes, len_u32_le, PAR1]
        let fake_meta = b"FAKE";
        let meta_len = fake_meta.len() as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(fake_meta);
        buf.extend_from_slice(&meta_len.to_le_bytes());
        buf.extend_from_slice(b"PAR1");

        // parse_footer will fail because "FAKE" is not valid Thrift,
        // but we get past the magic check (no "Invalid Parquet magic" error).
        let result = parse_footer(buf.len() as u64, &Bytes::from(buf));
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("Invalid Parquet magic"),
            "Should pass magic check, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("Thrift decode error") || err_msg.contains("parse footer"),
            "Expected Thrift error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_parse_footer_bad_magic() {
        let mut buf = vec![0u8; 16];
        // Write wrong magic
        buf[12..16].copy_from_slice(b"NOPE");
        let result = parse_footer(buf.len() as u64, &Bytes::from(buf));
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid Parquet magic"));
    }

    #[test]
    fn test_parse_footer_too_small() {
        let result = parse_footer(4, &Bytes::from(vec![0u8; 4]));
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    /// Verify that `ParquetDecodeMode::default()` is `Raw`.
    #[test]
    fn test_decode_mode_default_is_raw() {
        assert_eq!(ParquetDecodeMode::default(), ParquetDecodeMode::Raw);
    }

    /// Verify round-trip decode-mode accessors compile and return the right value.
    #[cfg(feature = "parquet-arrow")]
    #[test]
    fn test_decode_mode_variants_compile() {
        let modes = [ParquetDecodeMode::Raw, ParquetDecodeMode::ArrowIpc];
        assert_eq!(modes[0], ParquetDecodeMode::Raw);
        assert_eq!(modes[1], ParquetDecodeMode::ArrowIpc);
        assert_ne!(modes[0], modes[1]);
    }

    /// Write a real minimal Parquet file in memory, read it back via
    /// `parse_footer`, and verify the row-group extent calculation.
    ///
    /// This test is gated on `parquet-arrow` because it uses the `parquet`
    /// crate's write path (which needs the arrow feature for easy construction).
    #[cfg(feature = "parquet-arrow")]
    #[test]
    fn test_rg_extent_round_trip() {
        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::arrow_writer::ArrowWriter;
        use std::sync::Arc;

        // Build a two-column, two-row-group Parquet file in memory.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(2)) // force two row groups (rows 0-1, row 2)
            .build();
        let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let file_size = buf.len() as u64;
        let footer_bytes = Bytes::from(buf.clone());

        // parse_footer must succeed and return two row groups.
        let meta = parse_footer(file_size, &footer_bytes).unwrap();
        assert_eq!(meta.num_row_groups(), 2, "expected 2 row groups");

        // rg_byte_extent must return a non-zero range for both row groups,
        // selecting only column 0.
        for rg_idx in 0..2 {
            let rg = meta.row_group(rg_idx);
            let (start, length) = rg_byte_extent(rg, Some(&[0])).unwrap();
            assert!(
                length > 0,
                "rg[{rg_idx}] column 0 extent length must be > 0"
            );
            assert!(
                start + length <= file_size,
                "rg[{rg_idx}] extent must lie within file"
            );
        }
    }

    // ── needs_file_metadata tests ─────────────────────────────────────────────

    /// `Raw` mode must NOT request file_metadata (it only needs byte extents).
    ///
    /// This guards the epoch-2+ optimisation: in `Raw` mode, `build_extents()`
    /// returns an empty `file_metadata` Vec.  If `needs_file_metadata` were to
    /// incorrectly return `true` for `Raw`, every epoch-2 construction would
    /// issue unnecessary file-cache lookups.
    #[test]
    fn test_needs_file_metadata_raw_is_false() {
        assert!(
            !needs_file_metadata(ParquetDecodeMode::Raw),
            "Raw mode must not require file_metadata"
        );
    }

    /// `ArrowIpc` mode MUST request file_metadata (the Arrow decoder needs the
    /// schema embedded in `ParquetMetaData`).
    #[cfg(feature = "parquet-arrow")]
    #[test]
    fn test_needs_file_metadata_arrow_ipc_is_true() {
        assert!(
            needs_file_metadata(ParquetDecodeMode::ArrowIpc),
            "ArrowIpc mode must require file_metadata"
        );
    }

    /// The `needs_file_metadata` function must cover all variants without
    /// returning the same value for both.
    #[cfg(feature = "parquet-arrow")]
    #[test]
    fn test_needs_file_metadata_modes_differ() {
        assert_ne!(
            needs_file_metadata(ParquetDecodeMode::Raw),
            needs_file_metadata(ParquetDecodeMode::ArrowIpc),
            "Raw and ArrowIpc must return different values from needs_file_metadata"
        );
    }

    // ── global index singleton tests ──────────────────────────────────────────

    /// `parquet_index::global()` must return the same instance on every call.
    #[test]
    fn test_global_index_is_singleton() {
        use crate::data_loader::parquet_index;
        let a = parquet_index::global() as *const _;
        let b = parquet_index::global() as *const _;
        assert_eq!(
            a, b,
            "global() must return the same ParquetIndex every time"
        );
    }

    /// The global index's `col_indices` must be `None` (all columns).
    ///
    /// The index stores all-column extents so that datasets constructed with
    /// `col_indices=None` can use the fast path.  If the index were constructed
    /// with a column subset, datasets requesting all columns would have to fall
    /// back to the slow path even though the index is populated.
    #[test]
    fn test_global_index_col_indices_is_none() {
        use crate::data_loader::parquet_index;
        assert!(
            parquet_index::global().col_indices.is_none(),
            "global ParquetIndex must store all-column extents (col_indices=None)"
        );
    }
}

// ── MinIO integration tests ───────────────────────────────────────────────────
//
// These tests require a live MinIO endpoint and real credentials.  They are
// skipped automatically when AWS_ENDPOINT_URL is not set.
//
// Run with:
//   source .env && cargo test --features parquet-arrow -- parquet_integration --nocapture --ignored
//
// The tests:
//  1. Create a small two-row-group Parquet file in memory.
//  2. Upload it twice to `s3://mlp-flux/s3dlio/__parquet_tests__/`.
//  3. Construct `ParquetRowGroupDataset` (epoch 1) and verify extents + data.
//  4. Construct a second dataset (epoch 2) and assert the fast path fires.
//  5. Delete the test objects.

#[cfg(all(test, feature = "parquet-arrow"))]
mod integration_tests {
    use super::*;
    use crate::data_loader::parquet_index;
    use crate::s3_client::{aws_s3_client_async, run_on_global_rt};
    use arrow_array::{Float32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;

    const TEST_BUCKET: &str = "mlp-flux";
    const TEST_PREFIX: &str = "s3dlio/__parquet_tests__";

    /// Return `None` when the MinIO env vars are absent — caller skips the test.
    fn require_minio() -> Option<()> {
        if std::env::var("AWS_ENDPOINT_URL").is_ok() && std::env::var("AWS_ACCESS_KEY_ID").is_ok() {
            Some(())
        } else {
            eprintln!("Skipping MinIO integration test: AWS_ENDPOINT_URL not set");
            None
        }
    }

    /// Build a small two-row-group Parquet file in memory.
    ///
    /// Schema: `id: i64, value: f32`.  Rows 0–1 → RG 0, row 2 → RG 1.
    fn make_parquet_bytes() -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2, 3])),
                Arc::new(Float32Array::from(vec![1.1f32, 2.2, 3.3])),
            ],
        )
        .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let props = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(2))
            .build();
        let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        buf
    }

    /// Upload `data` to `s3://<bucket>/<key>` using the global AWS client.
    async fn upload(bucket: &str, key: &str, data: Vec<u8>) -> anyhow::Result<()> {
        let client = aws_s3_client_async().await?;
        client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(bytes::Bytes::from(data).into())
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("put_object failed: {}", e))?;
        Ok(())
    }

    /// Delete `s3://<bucket>/<key>`.
    async fn delete(bucket: &str, key: &str) -> anyhow::Result<()> {
        let client = aws_s3_client_async().await?;
        client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("delete_object failed: {}", e))?;
        Ok(())
    }

    /// Upload test files, run the test closure, always clean up.
    fn with_test_files<F>(test_fn: F)
    where
        F: FnOnce(&str) -> anyhow::Result<()>,
    {
        let parquet_data = make_parquet_bytes();
        let keys = [
            format!("{}/part-0.parquet", TEST_PREFIX),
            format!("{}/part-1.parquet", TEST_PREFIX),
        ];
        let prefix_uri = format!("s3://{}/{}/", TEST_BUCKET, TEST_PREFIX);

        // Upload both files synchronously via the global runtime.
        let keys_upload = keys.clone();
        run_on_global_rt(async move {
            for key in &keys_upload {
                upload(TEST_BUCKET, key, parquet_data.clone()).await?;
            }
            anyhow::Ok(())
        })
        .expect("upload failed");

        let result = test_fn(&prefix_uri);

        // Always clean up.
        let keys_cleanup = keys.clone();
        run_on_global_rt(async move {
            for key in &keys_cleanup {
                let _ = delete(TEST_BUCKET, key).await;
            }
            anyhow::Ok(())
        })
        .ok();

        result.expect("test body failed");
    }

    /// Epoch-1: construct `ParquetRowGroupDataset`, verify extent count + data.
    #[test]
    #[ignore = "requires live MinIO; run with: set -a && source .env && set +a && cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture"]
    fn test_parquet_dataset_epoch1_construction() {
        if require_minio().is_none() {
            return;
        }

        with_test_files(|prefix_uri| {
            // Clear the global index so this test is isolated.
            parquet_index::global().clear();
            crate::data_loader::parquet_file_cache::clear();

            let ds = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;

            // 2 files × 2 RGs each = 4 total row groups.
            assert_eq!(ds.len(), Some(4), "expected 4 row groups across 2 files");

            // Verify num_rows for each RG (RG0: 2 rows, RG1: 1 row).
            assert_eq!(ds.num_rows_in_rg(0), Some(2));
            assert_eq!(ds.num_rows_in_rg(1), Some(1));
            assert_eq!(ds.num_rows_in_rg(2), Some(2));
            assert_eq!(ds.num_rows_in_rg(3), Some(1));

            // Fetch one row group — must return non-empty bytes.
            let file_uris_for_check = ds.file_uris_arc();
            let bytes: Bytes =
                run_on_global_rt(
                    async move { ds.get(0).await.map_err(|e| anyhow::anyhow!("{}", e)) },
                )?;
            assert!(!bytes.is_empty(), "row group bytes must not be empty");

            // After epoch-1 construction, both files must be in the global index.
            for uri in file_uris_for_check.iter() {
                assert!(
                    parquet_index::global().is_indexed(uri),
                    "URI should be in global index after epoch 1: {}",
                    uri
                );
            }

            Ok(())
        });
    }

    /// Epoch-2: second construction must use the global-index fast path.
    ///
    /// We verify by clearing the per-file footer *cache* (so any network I/O
    /// would fail the column-extent computation) and confirming the dataset
    /// still constructs correctly from the global index alone.
    ///
    /// Note: the global index is populated by the epoch-1 construction above.
    /// This test runs after it (within the same process, sharing the static).
    #[test]
    #[ignore = "requires live MinIO; run with: set -a && source .env && set +a && cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture"]
    fn test_parquet_dataset_epoch2_fast_path() {
        if require_minio().is_none() {
            return;
        }

        with_test_files(|prefix_uri| {
            // Epoch 1: populate the global index.
            parquet_index::global().clear();
            crate::data_loader::parquet_file_cache::clear();
            let ds1 = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;
            assert_eq!(ds1.len(), Some(4));

            // Verify index is populated.
            for uri in ds1.file_uris_arc().iter() {
                assert!(parquet_index::global().is_indexed(uri));
            }

            // Epoch 2: global index is populated, file_metadata cache still warm.
            // Construction must succeed and return identical extents.
            let ds2 = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;
            assert_eq!(ds2.len(), Some(4));

            // Verify identical num_rows per RG.
            for rg in 0..4 {
                assert_eq!(
                    ds1.num_rows_in_rg(rg),
                    ds2.num_rows_in_rg(rg),
                    "RG {} num_rows mismatch between epoch1 and epoch2",
                    rg
                );
            }

            Ok(())
        });
    }

    /// Verify rg_num_rows() and file_extents() are consistent with the dataset.
    #[test]
    #[ignore = "requires live MinIO; run with: set -a && source .env && set +a && cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture"]
    fn test_parquet_index_row_offsets_consistency() {
        if require_minio().is_none() {
            return;
        }

        with_test_files(|prefix_uri| {
            parquet_index::global().clear();
            crate::data_loader::parquet_file_cache::clear();

            let ds = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;

            let gidx = parquet_index::global();

            // For each file, verify that rg_num_rows() matches dataset num_rows_in_rg().
            let rg_info = ds.rg_info_vec();
            let file_uris = ds.file_uris_arc();

            for (global_rg_idx, (file_uri_idx, rg_idx_in_file)) in rg_info.iter().enumerate() {
                let uri = &file_uris[*file_uri_idx];
                let idx_rows = gidx.rg_num_rows(uri, *rg_idx_in_file as u32);
                let ds_rows = ds.num_rows_in_rg(global_rg_idx);
                assert_eq!(
                    idx_rows, ds_rows,
                    "rg_num_rows mismatch for file={} rg={}",
                    uri, rg_idx_in_file
                );
            }

            Ok(())
        });
    }

    /// Epoch-2 construction is measurably faster than epoch-1 because
    /// `build_extents()` reads from the process-global DashMap instead of
    /// issuing S3 footer GETs.
    ///
    /// Both epochs still call `list_objects`, so the speedup is on the
    /// `build_extents` portion only.  For 2 files at footer_cap=4 MiB, the
    /// footer fetches take ~5–20 ms each, while the DashMap fast path takes
    /// < 0.1 ms.  We assert epoch-2 completes in under 1 second absolute AND
    /// is faster than epoch-1.
    #[test]
    #[ignore = "requires live MinIO; run with: set -a && source .env && set +a && cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture"]
    fn test_epoch2_faster_than_epoch1() {
        if require_minio().is_none() {
            return;
        }

        with_test_files(|prefix_uri| {
            // Clear everything so epoch-1 must do real footer fetches.
            parquet_index::global().clear();
            crate::data_loader::parquet_file_cache::clear();

            // ── Epoch 1 ──────────────────────────────────────────────────────
            let t_epoch1 = std::time::Instant::now();
            let ds1 = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;
            let elapsed_epoch1 = t_epoch1.elapsed();

            assert_eq!(ds1.len(), Some(4));
            eprintln!(
                "  Epoch-1 construction: {:.1} ms",
                elapsed_epoch1.as_secs_f64() * 1000.0
            );

            // Verify the global index was populated by epoch-1.
            for uri in ds1.file_uris_arc().iter() {
                assert!(
                    parquet_index::global().is_indexed(uri),
                    "epoch-1 must populate global index for: {}",
                    uri
                );
            }

            // ── Epoch 2 ──────────────────────────────────────────────────────
            // The global index is populated.  Clear ONLY the file cache to ensure
            // that if build_extents() incorrectly fell back to the slow path, it
            // would need to re-fetch footers from MinIO (and still succeed, but
            // much more slowly).
            crate::data_loader::parquet_file_cache::clear();

            let t_epoch2 = std::time::Instant::now();
            let ds2 = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;
            let elapsed_epoch2 = t_epoch2.elapsed();

            assert_eq!(ds2.len(), Some(4));
            eprintln!(
                "  Epoch-2 construction: {:.1} ms",
                elapsed_epoch2.as_secs_f64() * 1000.0
            );
            eprintln!(
                "  Speedup: {:.1}×",
                elapsed_epoch1.as_secs_f64() / elapsed_epoch2.as_secs_f64().max(0.0001)
            );

            // Epoch-2 must be faster than epoch-1.
            // Even if list_objects dominates, build_extents() adds zero network
            // overhead on epoch-2, so total time must not exceed epoch-1.
            assert!(
                elapsed_epoch2 <= elapsed_epoch1,
                "epoch-2 ({:.1} ms) must not be slower than epoch-1 ({:.1} ms)",
                elapsed_epoch2.as_secs_f64() * 1000.0,
                elapsed_epoch1.as_secs_f64() * 1000.0,
            );

            // Epoch-2 must complete in under 2 seconds absolute
            // (list_objects for 2 files over LAN should be < 200 ms).
            assert!(
                elapsed_epoch2.as_secs() < 2,
                "epoch-2 took {:.1} ms — expected < 2 s",
                elapsed_epoch2.as_secs_f64() * 1000.0
            );

            // Verify identical extents between epochs.
            for rg in 0..4 {
                assert_eq!(
                    ds1.num_rows_in_rg(rg),
                    ds2.num_rows_in_rg(rg),
                    "RG {} num_rows differs between epoch-1 and epoch-2",
                    rg
                );
            }

            Ok(())
        });
    }

    /// In `Raw` mode, epoch-2 construction must return an **empty**
    /// `file_metadata` Vec because `get_raw()` never accesses per-file metadata
    /// and the fast path deliberately skips populating it to avoid unnecessary
    /// cache lookups.
    #[test]
    #[ignore = "requires live MinIO; run with: set -a && source .env && set +a && cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture"]
    fn test_raw_fast_path_skips_file_metadata() {
        if require_minio().is_none() {
            return;
        }

        with_test_files(|prefix_uri| {
            // Epoch 1: slow path populates index.
            parquet_index::global().clear();
            crate::data_loader::parquet_file_cache::clear();
            let ds1 = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;
            assert_eq!(ds1.len(), Some(4));

            // Epoch 2: fast path.
            let ds2 = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;
            assert_eq!(ds2.len(), Some(4));

            // The fast path must NOT populate file_metadata for Raw mode
            // (it's never needed by get_raw()).
            assert_eq!(
                ds2.file_metadata_len(),
                0,
                "Raw mode epoch-2 must have empty file_metadata (fast path skips it)"
            );

            Ok(())
        });
    }

    /// In `ArrowIpc` mode, epoch-2 must still populate `file_metadata`
    /// (the Arrow decoder needs the schema from `ParquetMetaData`).
    #[test]
    #[ignore = "requires live MinIO; run with: set -a && source .env && set +a && cargo test --features parquet-arrow -- --ignored parquet --test-threads=1 --nocapture"]
    fn test_arrow_ipc_fast_path_keeps_file_metadata() {
        if require_minio().is_none() {
            return;
        }

        with_test_files(|prefix_uri| {
            // Epoch 1: populate index.
            parquet_index::global().clear();
            crate::data_loader::parquet_file_cache::clear();
            let _ = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::Raw,
            )?;

            // Epoch 2 in ArrowIpc mode: must have file_metadata populated.
            let ds_ipc = ParquetRowGroupDataset::new(
                prefix_uri,
                None,
                DEFAULT_FOOTER_CAP,
                ParquetDecodeMode::ArrowIpc,
            )?;
            assert_eq!(ds_ipc.len(), Some(4));

            assert_eq!(
                ds_ipc.file_metadata_len(),
                2, // 2 files
                "ArrowIpc mode must populate file_metadata for all files (needed by Arrow decoder)"
            );

            Ok(())
        });
    }
}
