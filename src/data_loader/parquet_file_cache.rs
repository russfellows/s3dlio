// src/data_loader/parquet_file_cache.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Process-lifetime cache for Parquet file metadata.
//!
//! Fetching and parsing a Parquet footer is expensive: one `HeadObject` + one
//! multi-MiB range `GET` per file.  For DLRM with 64 files this is 128 HTTP
//! round-trips per worker, per epoch.  After the first open the data is
//! immutable for the lifetime of a benchmark run.
//!
//! This module maintains a static [`DashMap`] keyed by file URI.  The first
//! caller for a given URI pays the fetch cost; every subsequent caller gets
//! the cached value instantly with no locking overhead (DashMap reads are
//! lock-free per shard).
//!
//! ## What is cached
//!
//! | Field | Purpose |
//! |---|---|
//! | `parquet_meta: Arc<ParquetMetaData>` | Full Thrift metadata — needed for ArrowIpc decode |
//! | `rg_row_offsets: Vec<i64>` | Cumulative row counts — replaces Python `_build_rg_offsets()` |
//!
//! ## What is NOT cached
//!
//! - `RgExtent` lists (column-specific byte offsets): depend on the caller's
//!   `col_indices` selection; pure arithmetic over the cached metadata, O(µs).
//! - File listings: prefix scope varies per call; `ListObjects` is fast.

use dashmap::DashMap;
use parquet::file::metadata::ParquetMetaData;
use std::sync::{Arc, OnceLock};

// ── Cached value ─────────────────────────────────────────────────────────────

/// Everything derived from a single Parquet file's footer, ready to use
/// without any further network I/O.
pub struct CachedFileMeta {
    /// Full parquet-rs metadata.  Required for `ArrowIpc` decode path and for
    /// computing per-column-group byte extents.
    pub parquet_meta: Arc<ParquetMetaData>,

    /// Cumulative row counts across row groups.
    ///
    /// `rg_row_offsets[i]` = total rows before row group `i` (0-based).  
    /// `rg_row_offsets[num_rgs]` = total rows in the file.  
    /// Length = `num_row_groups + 1`.
    ///
    /// Example: file with 3 row groups of 8192, 8192, 4000 rows:
    /// ```text
    /// [0, 8192, 16384, 20384]
    /// ```
    pub rg_row_offsets: Vec<i64>,
}

// ── Static cache ─────────────────────────────────────────────────────────────

static CACHE: OnceLock<DashMap<String, Arc<CachedFileMeta>>> = OnceLock::new();

fn cache() -> &'static DashMap<String, Arc<CachedFileMeta>> {
    CACHE.get_or_init(DashMap::new)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Return cached metadata for `uri`, fetching from storage on first call.
///
/// `footer_cap` is the number of bytes fetched from the file tail for footer
/// parsing.  Must be ≥ the largest footer in your dataset (default 4 MiB).
///
/// **Concurrency**: two concurrent callers for the same uncached URI will both
/// issue a fetch; the second result is discarded (last-writer-wins).  Both
/// callers receive correct data.  For typical Parquet workloads (one listing
/// phase before training starts) this never happens.
pub async fn get_or_fetch(
    uri: &str,
    footer_cap: u64,
) -> anyhow::Result<Arc<CachedFileMeta>> {
    let c = cache();

    // Fast path: already cached.
    if let Some(entry) = c.get(uri) {
        tracing::debug!("[s3dlio cache] hit  {}", uri);
        return Ok(Arc::clone(&*entry));
    }

    // Slow path: fetch stat + footer bytes, parse, store.
    tracing::info!("[s3dlio cache] miss {}", uri);
    let meta = fetch_and_parse(uri, footer_cap).await?;
    let arc = Arc::new(meta);
    c.insert(uri.to_owned(), Arc::clone(&arc));
    Ok(arc)
}

/// Remove a single URI from the cache (e.g. after a file is rewritten).
pub fn invalidate(uri: &str) {
    cache().remove(uri);
}

/// Clear the entire cache.
pub fn clear() {
    cache().clear();
}

/// Number of files currently cached.
pub fn len() -> usize {
    cache().len()
}

// ── Internal fetch + parse ────────────────────────────────────────────────────

async fn fetch_and_parse(uri: &str, footer_cap: u64) -> anyhow::Result<CachedFileMeta> {
    use crate::s3_utils::{get_object_range_uri_async, stat_object_uri_async};
    use parquet::file::metadata::ParquetMetaDataReader;
    use bytes::Bytes;

    // 1. Stat to get file size.
    let stat = stat_object_uri_async(uri)
        .await
        .map_err(|e| anyhow::anyhow!("stat '{}': {}", uri, e))?;
    let file_size = stat.size;

    // 2. Fetch last `footer_cap` bytes (or the whole file if smaller).
    let fetch_len = footer_cap.min(file_size);
    let offset = file_size - fetch_len;
    let buf: Bytes = get_object_range_uri_async(uri, offset, Some(fetch_len))
        .await
        .map_err(|e| anyhow::anyhow!("footer GET '{}': {}", uri, e))?;

    // 3. Validate PAR1 magic.
    const MAGIC: &[u8; 4] = b"PAR1";
    const SUFFIX: usize = 8; // 4-byte length + 4-byte magic

    let buf_len = buf.len();
    if buf_len < SUFFIX {
        anyhow::bail!("footer buffer too small ({} bytes) for '{}'", buf_len, uri);
    }
    if &buf[buf_len - 4..] != MAGIC {
        anyhow::bail!("'{}' is not a valid Parquet file (bad magic bytes)", uri);
    }

    // 4. Decode Thrift metadata length (little-endian u32 before magic).
    let meta_len = u32::from_le_bytes(
        buf[buf_len - 8..buf_len - 4]
            .try_into()
            .expect("slice is exactly 4 bytes"),
    ) as usize;

    if meta_len + SUFFIX > buf_len {
        anyhow::bail!(
            "'{}': Parquet metadata ({} bytes) + suffix ({} bytes) exceeds \
             fetched buffer ({} bytes). Increase footer_cap (currently {} bytes).",
            uri, meta_len, SUFFIX, buf_len, footer_cap
        );
    }

    // 5. Parse Thrift.
    let meta_start = buf_len - SUFFIX - meta_len;
    let parquet_meta = ParquetMetaDataReader::decode_metadata(&buf[meta_start..buf_len - SUFFIX])
        .map_err(|e| anyhow::anyhow!("Thrift decode '{}': {}", uri, e))?;

    // 6. Build cumulative row offsets.
    let num_rgs = parquet_meta.num_row_groups();
    let mut rg_row_offsets = Vec::with_capacity(num_rgs + 1);
    rg_row_offsets.push(0i64);
    for rg_idx in 0..num_rgs {
        let prev = *rg_row_offsets.last().unwrap();
        rg_row_offsets.push(prev + parquet_meta.row_group(rg_idx).num_rows());
    }

    tracing::info!(
        "[s3dlio cache] cached '{}': {} row-groups, {} total rows",
        uri,
        num_rgs,
        rg_row_offsets.last().copied().unwrap_or(0)
    );

    Ok(CachedFileMeta {
        parquet_meta: Arc::new(parquet_meta),
        rg_row_offsets,
    })
}
