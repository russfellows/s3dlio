// src/data_loader/parquet_rg.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `ParquetRowGroupDataset` — a [`Dataset`] where each item is the raw merged
//! column-chunk bytes for one Parquet row group across a set of selected columns.
//!
//! ## Design rationale
//!
//! The standard map-style DLIO reader calls `read_index()` once per *sample*
//! (e.g. 16 million times per worker for DLRM), even though only ~1,968 unique
//! I/O operations are needed (one per row group).  By making the row group the
//! unit of iteration and returning raw bytes, we shift the hot path entirely
//! into Rust:
//!
//! * **Construction** (sync) — lists files, stats them, fetches footers, parses
//!   Parquet metadata, and pre-computes the byte extent for every row group.
//!   All stat and footer GET requests are issued concurrently.
//!
//! * **`get(global_rg_idx)`** (async) — issues exactly one range GET covering
//!   the min–max byte span of the selected columns within that row group.
//!
//! ## Throughput model
//!
//! | Step | Cost per worker |
//! |------|----------------|
//! | Construction (stat + footer fetch × 64 files) | ~0.5 s (concurrent) |
//! | 1,968 range GETs @ 10 GiB/s | ~0.33 s |
//! | Python iteration overhead | depends on caller |
//!
//! Requires the `parquet` cargo feature (no arrow/datafusion pulled in).

use crate::data_loader::{Dataset, DatasetError};
use crate::s3_client::run_on_global_rt;
use crate::s3_utils::{
    get_object_range_uri_async, list_objects as list_objects_rs, parse_s3_uri,
    stat_object_uri_async,
};
use async_trait::async_trait;
use bytes::Bytes;
use parquet::file::metadata::{ColumnChunkMetaData, ParquetMetaDataReader, RowGroupMetaData};
use std::sync::Arc;

// ── Parquet footer constants ─────────────────────────────────────────────────

/// Magic bytes at the very end of every Parquet file.
const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Fixed size of the footer suffix: 4-byte metadata length + 4-byte magic.
const FOOTER_SUFFIX_LEN: usize = 8;

/// Default number of bytes fetched from the tail of each file for footer
/// parsing.  4 MiB comfortably covers production workloads (DLRM footers are
/// ~2.66 MiB).
pub const DEFAULT_FOOTER_CAP: usize = 4 * 1024 * 1024;

// ── Pre-computed per-RG extent ───────────────────────────────────────────────

/// Immutable byte extent for one row group, computed at construction time.
#[derive(Clone, Debug)]
struct RgExtent {
    /// Index into `file_uris`.
    file_uri_idx: usize,
    /// Byte offset of the first selected column chunk within the file.
    start: u64,
    /// Byte length covering all selected column chunks (merged span).
    length: u64,
    /// Number of rows in this row group (for metadata / progress reporting).
    num_rows: i64,
}

// ── Main Dataset struct ──────────────────────────────────────────────────────

/// Dataset where each item is the merged column-chunk bytes for one Parquet
/// row group.
///
/// Construction fetches and parses all Parquet footers once; subsequent
/// `get()` calls each issue a single range GET.
pub struct ParquetRowGroupDataset {
    /// Full URIs of all Parquet files under the prefix (sorted by listing order).
    file_uris: Arc<Vec<String>>,
    /// Flat index over every row group across all files.
    extents: Arc<Vec<RgExtent>>,
}

impl ParquetRowGroupDataset {
    /// Build the dataset synchronously.
    ///
    /// # Arguments
    /// * `uri_prefix`  — S3 URI prefix, e.g. `s3://bucket/data/train/`
    /// * `col_indices` — which column indices to include in each range GET;
    ///                   `None` means all columns
    /// * `footer_cap`  — bytes to fetch from the tail of each file for footer
    ///                   parsing; must be ≥ the largest footer in the dataset.
    ///                   Use [`DEFAULT_FOOTER_CAP`] when unsure.
    pub fn new(
        uri_prefix: &str,
        col_indices: Option<&[usize]>,
        footer_cap: usize,
    ) -> Result<Self, DatasetError> {
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

        // ── 2. Concurrently stat + fetch footers, then build extents ────────
        let col_indices_owned: Option<Vec<usize>> = col_indices.map(|s| s.to_vec());
        let footer_cap_u64 = footer_cap as u64;
        let uris_for_init = file_uris.clone();

        let extents: Vec<RgExtent> =
            run_on_global_rt(async move {
                build_extents(uris_for_init, col_indices_owned, footer_cap_u64).await
            })
            .map_err(|e| DatasetError::from(e.to_string()))?;

        Ok(Self {
            file_uris: Arc::new(file_uris),
            extents: Arc::new(extents),
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
}

// ── Dataset trait ────────────────────────────────────────────────────────────

#[async_trait]
impl Dataset for ParquetRowGroupDataset {
    /// Raw merged column bytes for one row group.
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
        let ext = self
            .extents
            .get(global_idx)
            .ok_or(DatasetError::IndexOutOfRange(global_idx))?;

        let uri = &self.file_uris[ext.file_uri_idx];

        get_object_range_uri_async(uri, ext.start, Some(ext.length))
            .await
            .map_err(DatasetError::from)
    }
}

// ── Async construction helpers ───────────────────────────────────────────────

/// Stat all files + fetch their footers concurrently, then parse and build
/// the flat extents index.
async fn build_extents(
    file_uris: Vec<String>,
    col_indices: Option<Vec<usize>>,
    footer_cap: u64,
) -> anyhow::Result<Vec<RgExtent>> {
    use futures::future::join_all;

    // ── Stat all files concurrently ─────────────────────────────────────────
    let stat_futs: Vec<_> = file_uris
        .iter()
        .map(|uri| {
            let u = uri.clone();
            async move { stat_object_uri_async(&u).await }
        })
        .collect();

    let stat_results = join_all(stat_futs).await;

    // Propagate any stat error; collect file sizes.
    let sizes: Vec<u64> = stat_results
        .into_iter()
        .zip(file_uris.iter())
        .map(|(r, uri)| {
            r.map(|s| s.size)
                .map_err(|e| anyhow::anyhow!("stat '{}': {}", uri, e))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // ── Fetch footer regions concurrently ────────────────────────────────────
    let fetch_futs: Vec<_> = file_uris
        .iter()
        .zip(sizes.iter())
        .map(|(uri, &size)| {
            let u = uri.clone();
            async move {
                let fetch_len = footer_cap.min(size);
                let offset = size - fetch_len;
                let bytes = get_object_range_uri_async(&u, offset, Some(fetch_len)).await?;
                Ok::<(u64, Bytes), anyhow::Error>((size, bytes))
            }
        })
        .collect();

    let footer_results = join_all(fetch_futs).await;

    // ── Parse footers and populate extents ───────────────────────────────────
    let mut extents = Vec::new();

    for (file_uri_idx, footer_res) in footer_results.into_iter().enumerate() {
        let (size, footer_bytes) = footer_res.map_err(|e| {
            anyhow::anyhow!("footer fetch '{}': {}", file_uris[file_uri_idx], e)
        })?;

        let meta =
            parse_footer(size, &footer_bytes).map_err(|e| {
                anyhow::anyhow!("parse footer '{}': {}", file_uris[file_uri_idx], e)
            })?;

        for rg_idx in 0..meta.num_row_groups() {
            let rg = meta.row_group(rg_idx);
            let (start, length) = rg_byte_extent(rg, col_indices.as_deref()).map_err(|e| {
                anyhow::anyhow!(
                    "'{}' rg[{}]: {}",
                    file_uris[file_uri_idx],
                    rg_idx,
                    e
                )
            })?;
            extents.push(RgExtent {
                file_uri_idx,
                start,
                length,
                num_rows: rg.num_rows(),
            });
        }
    }

    Ok(extents)
}

// ── Footer parsing ────────────────────────────────────────────────────────────

/// Parse a Parquet footer from the raw tail bytes of a file.
///
/// `file_size` is the total object size; `buf` is the last `buf.len()` bytes
/// of the file.  The caller must have fetched at least `footer_len + 8` bytes,
/// where `footer_len` is the Thrift metadata length stored in the file.
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
fn rg_byte_extent(
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
        assert!(result.unwrap_err().to_string().contains("Invalid Parquet magic"));
    }

    #[test]
    fn test_parse_footer_too_small() {
        let result = parse_footer(4, &Bytes::from(vec![0u8; 4]));
        assert!(result.unwrap_err().to_string().contains("too small"));
    }
}
