// src/data_loader/parquet_index.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Epoch-aware Parquet row-group byte-range index.
//!
//! `ParquetIndex` is a persistent, multi-epoch-aware map of
//! `(file_id, rg_idx) → (byte_offset, byte_length, last_epoch)`.
//!
//! ## Design
//!
//! Parquet readers normally issue 2–3 HTTP round-trips per file open:
//! `HeadObject` → range-GET footer → range-GET data.  At scale (thousands of
//! Parquet objects) this metadata overhead dominates.  `ParquetIndex` eliminates
//! it:
//!
//! 1. Footers are fetched **lazily in batches** (via the existing
//!    `parquet_file_cache`, which de-duplicates concurrent fetches per URI).
//! 2. After extraction, only the 16-byte `(byte_offset, byte_length)` range per
//!    row group is retained in a lock-free DashMap.  The heavy `ParquetMetaData`
//!    is released from our local `Arc<CachedFileMeta>` reference (the global
//!    file cache retains its own `Arc`).
//! 3. `last_epoch` tracks whether each row group has been fetched in the current
//!    training epoch, enabling correct multi-epoch reuse without re-fetching.
//!
//! ## Multi-epoch reuse
//!
//! Entries are **never evicted** automatically.  This is intentional:
//! - DashMap cost ≈ 80 bytes/entry regardless of epoch count.
//! - 10 K files × 123 RGs/file ≈ 1.2 M entries × 80 bytes ≈ 96 MB.
//! - 100 K files × 123 RGs/file ≈ 12 M entries × 80 bytes ≈ 960 MB.
//! - Evicting entries would force re-fetching footers on every epoch.
//! - `invalidate_uri()` and `clear()` provide explicit eviction when needed.
//!
//! ## URI interning
//!
//! File URIs are interned to compact `u32` IDs to minimise DashMap key size.
//! The interning table is never evicted; IDs are stable for the process lifetime.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::data_loader::parquet_file_cache;
use crate::data_loader::parquet_rg::rg_byte_extent;
use crate::s3_client::run_on_global_rt;

// ── Process-lifetime singleton ────────────────────────────────────────────────

static GLOBAL_INDEX: OnceLock<ParquetIndex> = OnceLock::new();

/// Access the process-lifetime global [`ParquetIndex`].
///
/// Stores all-column byte extents (`col_indices = None`) for every Parquet
/// file that [`crate::data_loader::parquet_rg::ParquetRowGroupDataset`] has
/// opened.  `build_extents()` populates this automatically on first open; on
/// epoch 2+ (and for any second worker in the same process), `build_extents()`
/// reads from the index instead of re-fetching footer metadata from S3.
pub fn global() -> &'static ParquetIndex {
    GLOBAL_INDEX.get_or_init(|| {
        ParquetIndex::new(
            None,
            crate::data_loader::parquet_rg::DEFAULT_FOOTER_CAP as u64,
        )
    })
}

// ── Row-group entry ───────────────────────────────────────────────────────────

/// Byte range and epoch-tracking state for one Parquet row group.
#[derive(Clone, Debug)]
struct RgEntry {
    byte_offset: u64,
    byte_length: u64,
    /// Epoch number when this row group was last fetched from storage.
    /// `0` means never fetched.  Updated by `mark_fetched()`.
    last_epoch: u32,
}

// ── URI interning ─────────────────────────────────────────────────────────────

#[derive(Default)]
struct UriInterner {
    uri_to_id: HashMap<String, u32>,
    id_to_uri: Vec<String>,
}

impl UriInterner {
    /// Return the existing ID for `uri`, or mint a new u32.
    fn intern(&mut self, uri: &str) -> u32 {
        if let Some(&id) = self.uri_to_id.get(uri) {
            return id;
        }
        let id = self.id_to_uri.len() as u32;
        self.id_to_uri.push(uri.to_owned());
        self.uri_to_id.insert(uri.to_owned(), id);
        id
    }

    fn get_id(&self, uri: &str) -> Option<u32> {
        self.uri_to_id.get(uri).copied()
    }
}

// ── Main struct ───────────────────────────────────────────────────────────────

/// Process-lifetime Parquet row-group byte-range index.
///
/// Thread-safe — wrap in `Arc<ParquetIndex>` to share across threads.
pub struct ParquetIndex {
    /// URI ↔ u32 interning (RwLock: many concurrent readers, rare writers).
    interner: RwLock<UriInterner>,
    /// `(file_id, rg_idx) → (byte_offset, byte_length, last_epoch)`.
    entries: DashMap<(u32, u32), RgEntry>,
    /// `file_id → cumulative row offsets` for sample-to-RG index lookup.
    ///
    /// `row_offsets[file_id][i]` = total rows before row group `i` (0-based).
    /// Length = `num_row_groups + 1`.
    row_offsets: DashMap<u32, Vec<i64>>,
    /// Column indices to include in the byte-range computation.  `None` = all.
    pub col_indices: Option<Vec<usize>>,
    /// Bytes fetched from each file tail for Parquet footer parsing.
    pub footer_cap: u64,
}

impl ParquetIndex {
    /// Create a new empty index.
    ///
    /// * `col_indices` — columns to include in byte-range computation;
    ///   `None` = all columns.  Must be consistent for all files.
    /// * `footer_cap` — bytes fetched from each file tail for footer parsing.
    ///   Default 4 MiB covers DLRM and most production workloads.
    pub fn new(col_indices: Option<Vec<usize>>, footer_cap: u64) -> Self {
        Self {
            interner: RwLock::new(UriInterner::default()),
            entries: DashMap::new(),
            row_offsets: DashMap::new(),
            col_indices,
            footer_cap,
        }
    }

    // ── Interning helpers ─────────────────────────────────────────────────────

    fn intern(&self, uri: &str) -> u32 {
        // Fast path: read lock (no writer contention).
        {
            let r = self.interner.read().expect("interner read lock poisoned");
            if let Some(id) = r.get_id(uri) {
                return id;
            }
        }
        // Slow path: write lock.
        self.interner
            .write()
            .expect("interner write lock poisoned")
            .intern(uri)
    }

    fn get_id(&self, uri: &str) -> Option<u32> {
        self.interner
            .read()
            .expect("interner read lock poisoned")
            .get_id(uri)
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// `true` if all row groups for `uri` are already in the index.
    pub fn is_indexed(&self, uri: &str) -> bool {
        self.get_id(uri)
            .map_or(false, |id| self.row_offsets.contains_key(&id))
    }

    /// Number of row-group entries in the index (across all files).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of files whose row groups are indexed.
    pub fn file_count(&self) -> usize {
        self.row_offsets.len()
    }

    /// Number of row groups in `uri`, or `None` if not yet indexed.
    pub fn file_rg_count(&self, uri: &str) -> Option<u32> {
        self.get_id(uri).and_then(|id| {
            self.row_offsets
                .get(&id)
                .map(|v| v.len().saturating_sub(1) as u32)
        })
    }

    /// Find which row group contains `sample_idx` (0-based within the file).
    ///
    /// Returns the row-group index (0-based), or an error if `uri` is not
    /// indexed or `sample_idx` is out of range.
    pub fn rg_for_sample(&self, uri: &str, sample_idx: i64) -> anyhow::Result<u32> {
        let file_id = self
            .get_id(uri)
            .ok_or_else(|| anyhow::anyhow!("URI not indexed: {}", uri))?;
        let offsets = self
            .row_offsets
            .get(&file_id)
            .ok_or_else(|| anyhow::anyhow!("row_offsets not found for URI: {}", uri))?;
        let n = offsets.len();
        if n < 2 {
            anyhow::bail!("No row groups for URI: {}", uri);
        }
        // Binary search: find the last i such that offsets[i] <= sample_idx.
        let mut lo = 0usize;
        let mut hi = n - 1; // n-1 = num_rgs (exclusive upper bound = total rows)
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if offsets[mid] <= sample_idx {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(lo as u32)
    }

    /// Byte range `(byte_offset, byte_length)` for `(uri, rg_idx)`.
    pub fn rg_range(&self, uri: &str, rg_idx: u32) -> anyhow::Result<(u64, u64)> {
        let file_id = self
            .get_id(uri)
            .ok_or_else(|| anyhow::anyhow!("URI not indexed: {}", uri))?;
        let entry = self
            .entries
            .get(&(file_id, rg_idx))
            .ok_or_else(|| anyhow::anyhow!("RG ({}, {}) not in index", uri, rg_idx))?;
        Ok((entry.byte_offset, entry.byte_length))
    }

    /// Combined lookup: sample → RG index + byte range, in one call.
    ///
    /// Eliminates two separate PyO3 round-trips on the Python hot path.
    /// Returns `(rg_idx, byte_offset, byte_length)`.
    pub fn rg_lookup(&self, uri: &str, sample_idx: i64) -> anyhow::Result<(u32, u64, u64)> {
        let rg_idx = self.rg_for_sample(uri, sample_idx)?;
        let (offset, length) = self.rg_range(uri, rg_idx)?;
        Ok((rg_idx, offset, length))
    }

    /// `true` if `(uri, rg_idx)` was fetched during epoch `epoch`.
    pub fn was_fetched(&self, uri: &str, rg_idx: u32, epoch: u32) -> bool {
        self.get_id(uri).map_or(false, |file_id| {
            self.entries
                .get(&(file_id, rg_idx))
                .map_or(false, |e| e.last_epoch >= epoch)
        })
    }

    /// Record that `(uri, rg_idx)` was fetched during epoch `epoch`.
    pub fn mark_fetched(&self, uri: &str, rg_idx: u32, epoch: u32) {
        if let Some(file_id) = self.get_id(uri) {
            if let Some(mut entry) = self.entries.get_mut(&(file_id, rg_idx)) {
                entry.last_epoch = epoch;
            }
        }
    }

    /// Combined `was_fetched` + `mark_fetched` in one DashMap access.
    ///
    /// Returns `true` if already fetched this epoch (caller should skip the GET).
    /// If not fetched, atomically sets `last_epoch = epoch` and returns `false`.
    pub fn check_and_mark(&self, uri: &str, rg_idx: u32, epoch: u32) -> bool {
        if let Some(file_id) = self.get_id(uri) {
            if let Some(mut entry) = self.entries.get_mut(&(file_id, rg_idx)) {
                if entry.last_epoch >= epoch {
                    return true;
                }
                entry.last_epoch = epoch;
                return false;
            }
        }
        false
    }

    // ── Indexing ───────────────────────────────────────────────────────────────

    /// Ensure all URIs in `uris` are indexed, fetching footers concurrently in
    /// batches of `batch_size`.  Already-indexed URIs are skipped.
    ///
    /// Footers are fetched via `parquet_file_cache::get_or_fetch`, which
    /// de-duplicates concurrent fetches and caches results for the process
    /// lifetime.  After extraction, our local `Arc<CachedFileMeta>` is dropped;
    /// the file cache retains its own `Arc`.
    pub fn ensure_indexed(
        &self,
        uris: &[String],
        epoch: u32,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        let to_fetch: Vec<String> = uris
            .iter()
            .filter(|u| !self.is_indexed(u))
            .cloned()
            .collect();

        if to_fetch.is_empty() {
            return Ok(());
        }

        let footer_cap = self.footer_cap;

        for chunk in to_fetch.chunks(batch_size.max(1)) {
            let chunk_owned: Vec<String> = chunk.to_vec();

            let results = run_on_global_rt(async move {
                use futures::future::join_all;
                let futs: Vec<_> = chunk_owned
                    .iter()
                    .map(|uri| {
                        let u = uri.clone();
                        async move {
                            let meta = parquet_file_cache::get_or_fetch(&u, footer_cap).await?;
                            anyhow::Ok((u, meta))
                        }
                    })
                    .collect();
                Ok(join_all(futs).await)
            })?;

            for result in results {
                let (uri, cached) = result?;
                // Extract byte ranges and drop our Arc<CachedFileMeta> reference.
                // The global parquet_file_cache keeps its own Arc.
                self.insert_file(
                    &uri,
                    &cached.parquet_meta,
                    &cached.rg_row_offsets,
                    epoch,
                )?;
                // `cached` Arc dropped here — heavy ParquetMetaData released.
            }
        }

        Ok(())
    }

    /// Insert all row groups for one file.  Called internally after footer fetch.
    pub(crate) fn insert_file(
        &self,
        uri: &str,
        meta: &parquet::file::metadata::ParquetMetaData,
        rg_row_offsets: &[i64],
        _epoch: u32,
    ) -> anyhow::Result<()> {
        let file_id = self.intern(uri);

        // Skip if another thread raced us.
        if self.row_offsets.contains_key(&file_id) {
            return Ok(());
        }

        for rg_idx in 0..meta.num_row_groups() {
            let rg = meta.row_group(rg_idx);
            let (offset, length) = rg_byte_extent(rg, self.col_indices.as_deref())
                .map_err(|e| anyhow::anyhow!("'{}' rg[{}]: {}", uri, rg_idx, e))?;
            self.entries.insert(
                (file_id, rg_idx as u32),
                RgEntry {
                    byte_offset: offset,
                    byte_length: length,
                    last_epoch: 0, // not yet fetched
                },
            );
        }

        // Store cumulative row offsets for sample→RG lookup.
        self.row_offsets.insert(file_id, rg_row_offsets.to_vec());

        Ok(())
    }

    // ── Bulk queries ──────────────────────────────────────────────────────────

    /// Retrieve all row-group extents for `uri` in order.
    ///
    /// Returns `None` if `uri` is not yet indexed.  Each tuple is
    /// `(byte_offset, byte_length, num_rows)` for that row group.
    ///
    /// This is the key method used by the epoch-2+ fast path in
    /// `build_extents()`: all three fields needed to reconstruct a
    /// [`RgExtent`] are returned in one DashMap read per row group.
    pub fn file_extents(&self, uri: &str) -> Option<Vec<(u64, u64, i64)>> {
        let file_id = self.get_id(uri)?;
        let offsets = self.row_offsets.get(&file_id)?;
        let n_rgs = offsets.len().saturating_sub(1);
        let mut result = Vec::with_capacity(n_rgs);
        for rg_idx in 0..n_rgs as u32 {
            // Return None if any entry is missing (partial/corrupt index).
            let entry = self.entries.get(&(file_id, rg_idx))?;
            let num_rows = offsets[(rg_idx + 1) as usize] - offsets[rg_idx as usize];
            result.push((entry.byte_offset, entry.byte_length, num_rows));
        }
        Some(result)
    }

    /// Number of rows in `(uri, rg_idx)`, derived from the stored row offsets.
    ///
    /// Returns `None` if `uri` is not indexed or `rg_idx` is out of range.
    pub fn rg_num_rows(&self, uri: &str, rg_idx: u32) -> Option<i64> {
        let file_id = self.get_id(uri)?;
        let offsets = self.row_offsets.get(&file_id)?;
        let i = rg_idx as usize;
        if i + 1 < offsets.len() {
            Some(offsets[i + 1] - offsets[i])
        } else {
            None
        }
    }

    // ── Eviction ──────────────────────────────────────────────────────────────

    /// Remove all row-group entries for `uri` (e.g. after a file is rewritten).
    pub fn invalidate_uri(&self, uri: &str) {
        if let Some(file_id) = self.get_id(uri) {
            self.entries.retain(|(fid, _), _| *fid != file_id);
            self.row_offsets.remove(&file_id);
        }
    }

    /// Clear all row-group entries (keeps URI interning table intact).
    pub fn clear(&self) {
        self.entries.clear();
        self.row_offsets.clear();
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory index and verify intern + insert + file_extents.
    #[test]
    fn test_index_insert_and_file_extents() {
        let idx = ParquetIndex::new(None, 4 * 1024 * 1024);

        let uri = "s3://test-bucket/data/part-0.parquet";
        assert!(!idx.is_indexed(uri));
        assert_eq!(idx.file_extents(uri), None);
        assert_eq!(idx.rg_num_rows(uri, 0), None);

        // Manually populate via the interning + entries DashMap
        // (insert_file needs real ParquetMetaData; test RgEntry directly).
        let file_id = idx.intern(uri);
        idx.entries.insert((file_id, 0), RgEntry { byte_offset: 100, byte_length: 500, last_epoch: 0 });
        idx.entries.insert((file_id, 1), RgEntry { byte_offset: 600, byte_length: 400, last_epoch: 0 });
        // row_offsets: [0, 128, 384] → rg0 has 128 rows, rg1 has 256 rows.
        idx.row_offsets.insert(file_id, vec![0i64, 128, 384]);

        assert!(idx.is_indexed(uri));
        assert_eq!(idx.file_rg_count(uri), Some(2));
        assert_eq!(idx.rg_num_rows(uri, 0), Some(128));
        assert_eq!(idx.rg_num_rows(uri, 1), Some(256));
        assert_eq!(idx.rg_num_rows(uri, 2), None); // out of range

        let extents = idx.file_extents(uri).expect("should be indexed");
        assert_eq!(extents.len(), 2);
        assert_eq!(extents[0], (100, 500, 128));
        assert_eq!(extents[1], (600, 400, 256));
    }

    /// Verify rg_for_sample binary search across two row groups.
    #[test]
    fn test_rg_for_sample_binary_search() {
        let idx = ParquetIndex::new(None, 4 * 1024 * 1024);
        let uri = "s3://test-bucket/data/part-1.parquet";
        let file_id = idx.intern(uri);
        idx.entries.insert((file_id, 0), RgEntry { byte_offset: 0, byte_length: 256, last_epoch: 0 });
        idx.entries.insert((file_id, 1), RgEntry { byte_offset: 256, byte_length: 256, last_epoch: 0 });
        idx.entries.insert((file_id, 2), RgEntry { byte_offset: 512, byte_length: 128, last_epoch: 0 });
        // rg0: 0..100, rg1: 100..200, rg2: 200..250
        idx.row_offsets.insert(file_id, vec![0i64, 100, 200, 250]);

        assert_eq!(idx.rg_for_sample(uri, 0).unwrap(), 0);
        assert_eq!(idx.rg_for_sample(uri, 99).unwrap(), 0);
        assert_eq!(idx.rg_for_sample(uri, 100).unwrap(), 1);
        assert_eq!(idx.rg_for_sample(uri, 199).unwrap(), 1);
        assert_eq!(idx.rg_for_sample(uri, 200).unwrap(), 2);
        assert_eq!(idx.rg_for_sample(uri, 249).unwrap(), 2);
    }

    /// Verify rg_range lookup returns correct byte offsets.
    #[test]
    fn test_rg_range_lookup() {
        let idx = ParquetIndex::new(None, 4 * 1024 * 1024);
        let uri = "s3://test-bucket/data/part-2.parquet";
        let file_id = idx.intern(uri);
        idx.entries.insert((file_id, 0), RgEntry { byte_offset: 1024, byte_length: 8192, last_epoch: 0 });
        idx.row_offsets.insert(file_id, vec![0i64, 64]);

        let (off, len) = idx.rg_range(uri, 0).unwrap();
        assert_eq!(off, 1024);
        assert_eq!(len, 8192);
        assert!(idx.rg_range(uri, 1).is_err()); // rg1 not in index
    }

    /// Verify check_and_mark epoch logic.
    #[test]
    fn test_check_and_mark_epoch() {
        let idx = ParquetIndex::new(None, 4 * 1024 * 1024);
        let uri = "s3://test-bucket/data/part-3.parquet";
        let file_id = idx.intern(uri);
        idx.entries.insert((file_id, 0), RgEntry { byte_offset: 0, byte_length: 64, last_epoch: 0 });
        idx.row_offsets.insert(file_id, vec![0i64, 32]);

        // Epoch 1: not yet fetched → returns false, marks it.
        assert!(!idx.check_and_mark(uri, 0, 1));
        // Same epoch again → already marked, returns true.
        assert!(idx.check_and_mark(uri, 0, 1));
        // Epoch 2 → not yet fetched in epoch 2.
        assert!(!idx.check_and_mark(uri, 0, 2));
        assert!(idx.check_and_mark(uri, 0, 2));
    }

    /// Verify invalidate_uri removes entries but keeps other URIs.
    #[test]
    fn test_invalidate_uri() {
        let idx = ParquetIndex::new(None, 4 * 1024 * 1024);
        let uri_a = "s3://test-bucket/data/a.parquet";
        let uri_b = "s3://test-bucket/data/b.parquet";
        for uri in [uri_a, uri_b] {
            let fid = idx.intern(uri);
            idx.entries.insert((fid, 0), RgEntry { byte_offset: 0, byte_length: 64, last_epoch: 0 });
            idx.row_offsets.insert(fid, vec![0i64, 10]);
        }
        assert_eq!(idx.file_count(), 2);
        idx.invalidate_uri(uri_a);
        assert!(!idx.is_indexed(uri_a));
        assert!(idx.is_indexed(uri_b));
        assert_eq!(idx.file_count(), 1);
    }

    /// Verify rg_lookup combines rg_for_sample + rg_range in one call.
    #[test]
    fn test_rg_lookup_combined() {
        let idx = ParquetIndex::new(None, 4 * 1024 * 1024);
        let uri = "s3://test-bucket/data/part-4.parquet";
        let file_id = idx.intern(uri);
        idx.entries.insert((file_id, 0), RgEntry { byte_offset: 200, byte_length: 300, last_epoch: 0 });
        idx.entries.insert((file_id, 1), RgEntry { byte_offset: 500, byte_length: 400, last_epoch: 0 });
        idx.row_offsets.insert(file_id, vec![0i64, 50, 100]);

        // Sample 30 → rg 0 (rows 0..50)
        let (rg, off, len) = idx.rg_lookup(uri, 30).unwrap();
        assert_eq!((rg, off, len), (0, 200, 300));
        // Sample 75 → rg 1 (rows 50..100)
        let (rg, off, len) = idx.rg_lookup(uri, 75).unwrap();
        assert_eq!((rg, off, len), (1, 500, 400));
    }
}
