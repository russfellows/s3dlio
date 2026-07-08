// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use super::latest::{read_latest, Latest};
use super::manifest::Manifest;
use super::paths::KeyLayout;
use crate::data_loader::parallel_fetch::DropCancel;
use crate::object_store::{store_for_uri, ObjectStore};
use anyhow::Result;
use bytes::Bytes;
use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub struct Reader<'a> {
    pub store: &'a dyn ObjectStore,
    pub base_uri: String,
}

/// Read one shard by relative key from an owned store handle.
///
/// Functionally equivalent to `Reader::read_shard(_with_validation)?` but
/// takes an owned `Arc<dyn ObjectStore>` and a base URI directly, so it
/// can be called from a `tokio::spawn`ed task (which requires `'static`
/// captures — the borrowed `Reader<'a>` cannot cross that boundary).
///
/// Used by `read_all_shards_concurrent` and its `_with_validation`
/// companion (issue #148 site 3.1d). When `expected_checksum` is `Some`,
/// dispatches to `store.get_with_validation`; otherwise plain
/// `store.get`. `.zst` shards are decompressed identically to the
/// `Reader` methods.
async fn read_shard_owned(
    store: &Arc<dyn ObjectStore>,
    base_uri: &str,
    shard_rel_key: &str,
    expected_checksum: Option<&str>,
) -> Result<Bytes> {
    let uri = format!("{}/{}", base_uri, shard_rel_key);
    let data = match expected_checksum {
        Some(cs) => store.get_with_validation(&uri, Some(cs)).await?,
        None => store.get(&uri).await?,
    };

    if shard_rel_key.ends_with(".zst") {
        use std::io::Read;
        let mut decoder = zstd::Decoder::new(&data[..])?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(Bytes::from(decompressed))
    } else {
        Ok(data)
    }
}

impl<'a> Reader<'a> {
    pub fn new(store: &'a dyn ObjectStore, base_uri: String) -> Self {
        Self {
            store,
            base_uri: base_uri.trim_end_matches('/').to_string(),
        }
    }

    /// Get the latest checkpoint pointer
    pub async fn get_latest(&self) -> Result<Option<Latest>> {
        let latest_uri = format!("{}/latest.json", self.base_uri);
        read_latest(self.store, &latest_uri).await
    }

    /// Load a manifest by its relative key
    pub async fn load_manifest(&self, manifest_rel_key: &str) -> Result<Manifest> {
        let uri = format!("{}/{}", self.base_uri, manifest_rel_key);
        let bytes = self.store.get(&uri).await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Load the latest complete manifest
    pub async fn load_latest_manifest(&self) -> Result<Option<Manifest>> {
        match self.get_latest().await? {
            Some(latest) => {
                let manifest = self.load_manifest(&latest.manifest_key).await?;
                Ok(Some(manifest))
            }
            None => {
                // Fallback: scan for latest complete manifest
                self.scan_latest_complete().await
            }
        }
    }

    /// Fallback if latest.json is missing or racy: scan manifests/ and select the max (step, ts)
    pub async fn scan_latest_complete(&self) -> Result<Option<Manifest>> {
        let prefix = format!("{}/manifests/", self.base_uri);
        let uris = self.store.list(&prefix, false).await?;

        let mut best: Option<(u64, String, Manifest)> = None;

        for uri in uris {
            if !uri.ends_with(".json") {
                continue;
            }

            // Try to load and parse the manifest
            let bytes = match self.store.get(&uri).await {
                Ok(b) => b,
                Err(_) => continue,
            };

            let manifest = match serde_json::from_slice::<Manifest>(&bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Only consider complete manifests
            if manifest.status != "complete" {
                continue;
            }

            // Check if this is better than current best
            let is_better = match &best {
                None => true,
                Some((prev_step, prev_ts, _)) => {
                    (manifest.global_step, manifest.wall_time.clone())
                        > (*prev_step, prev_ts.clone())
                }
            };

            if is_better {
                best = Some((manifest.global_step, manifest.wall_time.clone(), manifest));
            }
        }

        Ok(best.map(|(_, _, manifest)| manifest))
    }

    /// Read a complete shard by its relative key
    pub async fn read_shard(&self, shard_rel_key: &str) -> Result<Bytes> {
        let uri = format!("{}/{}", self.base_uri, shard_rel_key);
        let data = self.store.get(&uri).await?;

        // Check if this is a compressed file and decompress if needed
        if shard_rel_key.ends_with(".zst") {
            use std::io::Read;
            let mut decoder = zstd::Decoder::new(&data[..])?;
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            // Convert Vec<u8> to Bytes for consistency
            Ok(Bytes::from(decompressed))
        } else {
            // Return Bytes directly - zero copy!
            Ok(data)
        }
    }

    /// Read a complete shard by its relative key with integrity validation
    pub async fn read_shard_with_validation(
        &self,
        shard_rel_key: &str,
        expected_checksum: Option<&str>,
    ) -> Result<Bytes> {
        let uri = format!("{}/{}", self.base_uri, shard_rel_key);
        let data = self
            .store
            .get_with_validation(&uri, expected_checksum)
            .await?;

        // Check if this is a compressed file and decompress if needed
        if shard_rel_key.ends_with(".zst") {
            use std::io::Read;
            let mut decoder = zstd::Decoder::new(&data[..])?;
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            // Convert Vec<u8> to Bytes for consistency
            Ok(Bytes::from(decompressed))
        } else {
            // Return Bytes directly - zero copy!
            Ok(data)
        }
    }

    /// Read a shard by rank from a manifest
    pub async fn read_shard_by_rank(&self, manifest: &Manifest, rank: u32) -> Result<Bytes> {
        let shard = manifest
            .shards
            .iter()
            .find(|s| s.rank == rank)
            .ok_or_else(|| anyhow::anyhow!("Shard for rank {} not found in manifest", rank))?;

        self.read_shard(&shard.key).await
    }

    /// Read a shard by rank from a manifest with integrity validation
    pub async fn read_shard_by_rank_with_validation(
        &self,
        manifest: &Manifest,
        rank: u32,
    ) -> Result<Bytes> {
        let shard = manifest
            .shards
            .iter()
            .find(|s| s.rank == rank)
            .ok_or_else(|| anyhow::anyhow!("Shard for rank {} not found in manifest", rank))?;

        self.read_shard_with_validation(&shard.key, shard.checksum.as_deref())
            .await
    }

    /// Read all shards from a manifest
    pub async fn read_all_shards(&self, manifest: &Manifest) -> Result<Vec<(u32, Bytes)>> {
        let mut results = Vec::new();

        for shard in &manifest.shards {
            let data = self.read_shard(&shard.key).await?;
            results.push((shard.rank, data));
        }

        // Sort by rank for consistent ordering
        results.sort_by_key(|(rank, _)| *rank);
        Ok(results)
    }

    /// Read all shards from a manifest with integrity validation
    pub async fn read_all_shards_with_validation(
        &self,
        manifest: &Manifest,
    ) -> Result<Vec<(u32, Bytes)>> {
        let mut results = Vec::new();

        for shard in &manifest.shards {
            let data = self
                .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
                .await?;
            results.push((shard.rank, data));
        }

        // Sort by rank for consistent ordering
        results.sort_by_key(|(rank, _)| *rank);
        Ok(results)
    }

    /// Read all shards concurrently for better performance.
    ///
    /// Task-level parallelism (issue #148 site 3.1d): each shard read is
    /// `tokio::spawn`'d so tokio can distribute both the network I/O and
    /// any zstd decode work (see `read_shard`) across worker threads.
    /// `DropCancel` on this frame + `select!` in each spawn cancels
    /// in-flight reads on any exit path (normal return, panic here, or
    /// the task holding this Reader being dropped).
    ///
    /// Because `Reader<'a>` borrows `&'a dyn ObjectStore` — non-`'static`,
    /// so `tokio::spawn` can't capture it — an owned `Arc<dyn ObjectStore>`
    /// is looked up once via `store_for_uri(&self.base_uri)` and cloned
    /// into each spawn. `store_for_uri` is deterministic for a given URI,
    /// so the freshly-looked-up store is functionally equivalent to the
    /// borrowed `self.store` for the read+decompress leaf operation.
    pub async fn read_all_shards_concurrent(
        &self,
        manifest: &Manifest,
    ) -> Result<Vec<(u32, Bytes)>> {
        let cancel = CancellationToken::new();
        let _drop_cancel = DropCancel(cancel.clone());
        let store_arc: Arc<dyn ObjectStore> = Arc::from(store_for_uri(&self.base_uri)?);
        let base_uri = self.base_uri.clone();

        let handles: Vec<JoinHandle<Result<(u32, Bytes)>>> = manifest
            .shards
            .iter()
            .map(|shard| {
                let rank = shard.rank;
                let key = shard.key.clone();
                let store = Arc::clone(&store_arc);
                let base_uri = base_uri.clone();
                let token = cancel.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = token.cancelled() => Err(anyhow::anyhow!(
                            "checkpoint shard read cancelled"
                        )),
                        r = read_shard_owned(&store, &base_uri, &key, None) => {
                            r.map(|data| (rank, data))
                        }
                    }
                })
            })
            .collect();

        // Drain-first-then-first-error (site 3.2 pattern): no JoinHandle
        // is dropped mid-flight, so no shard read is left detached.
        let mut results = Vec::with_capacity(handles.len());
        let mut first_err: Option<anyhow::Error> = None;
        for h in handles {
            match h.await {
                Ok(Ok(pair)) => results.push(pair),
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(join_err) if join_err.is_panic() => {
                    first_err.get_or_insert(anyhow::anyhow!(
                        "checkpoint shard read task panicked: {}",
                        join_err
                    ));
                }
                Err(_) => {} // cancelled during shutdown
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        results.sort_by_key(|(rank, _)| *rank);
        Ok(results)
    }

    /// Read all shards concurrently with integrity validation.
    ///
    /// Task-level parallelism companion to `read_all_shards_concurrent` —
    /// same DropCancel + spawn + select! pattern; the only difference is
    /// each shard is read via `read_shard_owned` with the per-shard
    /// expected checksum passed through for validation.
    pub async fn read_all_shards_concurrent_with_validation(
        &self,
        manifest: &Manifest,
    ) -> Result<Vec<(u32, Bytes)>> {
        let cancel = CancellationToken::new();
        let _drop_cancel = DropCancel(cancel.clone());
        let store_arc: Arc<dyn ObjectStore> = Arc::from(store_for_uri(&self.base_uri)?);
        let base_uri = self.base_uri.clone();

        let handles: Vec<JoinHandle<Result<(u32, Bytes)>>> = manifest
            .shards
            .iter()
            .map(|shard| {
                let rank = shard.rank;
                let key = shard.key.clone();
                let checksum = shard.checksum.clone();
                let store = Arc::clone(&store_arc);
                let base_uri = base_uri.clone();
                let token = cancel.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = token.cancelled() => Err(anyhow::anyhow!(
                            "checkpoint shard read cancelled"
                        )),
                        r = read_shard_owned(&store, &base_uri, &key, checksum.as_deref()) => {
                            r.map(|data| (rank, data))
                        }
                    }
                })
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        let mut first_err: Option<anyhow::Error> = None;
        for h in handles {
            match h.await {
                Ok(Ok(pair)) => results.push(pair),
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(join_err) if join_err.is_panic() => {
                    first_err.get_or_insert(anyhow::anyhow!(
                        "checkpoint shard read task panicked: {}",
                        join_err
                    ));
                }
                Err(_) => {}
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        results.sort_by_key(|(rank, _)| *rank);
        Ok(results)
    }

    /// Validate an entire checkpoint including all shards with integrity checking
    pub async fn validate_checkpoint_integrity(&self, manifest: &Manifest) -> Result<bool> {
        // Check if all shards have checksums
        for shard in &manifest.shards {
            if shard.checksum.is_none() {
                return Ok(false); // Cannot validate without checksums
            }
        }

        // Attempt to read all shards with validation
        match self.read_all_shards_with_validation(manifest).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get checkpoint metadata without downloading data
    pub async fn get_checkpoint_info(&self, step: Option<u64>) -> Result<Option<Manifest>> {
        match step {
            Some(step) => {
                // Try to find specific step
                self.find_manifest_by_step(step).await
            }
            None => {
                // Get latest
                self.load_latest_manifest().await
            }
        }
    }

    /// Find a manifest by step number
    pub async fn find_manifest_by_step(&self, step: u64) -> Result<Option<Manifest>> {
        let prefix = format!("{}/manifests/", self.base_uri);
        let uris = self.store.list(&prefix, false).await?;

        for uri in uris {
            if !uri.ends_with(".json") {
                continue;
            }

            // Extract step from filename
            if let Ok(rel_key) = self.extract_relative_key(&uri) {
                if let Ok((manifest_step, _)) = KeyLayout::parse_manifest_key(&rel_key) {
                    if manifest_step == step {
                        return Ok(Some(self.load_manifest(&rel_key).await?));
                    }
                }
            }
        }

        Ok(None)
    }

    /// List all available checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<(u64, String, String)>> {
        let prefix = format!("{}/manifests/", self.base_uri);
        let uris = self.store.list(&prefix, false).await?;

        let mut checkpoints = Vec::new();

        for uri in uris {
            if !uri.ends_with(".json") {
                continue;
            }

            if let Ok(rel_key) = self.extract_relative_key(&uri) {
                if let Ok((step, timestamp)) = KeyLayout::parse_manifest_key(&rel_key) {
                    // Try to load manifest to get status
                    if let Ok(manifest) = self.load_manifest(&rel_key).await {
                        checkpoints.push((step, timestamp, manifest.status));
                    }
                }
            }
        }

        // Sort by step
        checkpoints.sort_by_key(|(step, _, _)| *step);
        Ok(checkpoints)
    }

    /// Delete a checkpoint (manifest and all shards)
    pub async fn delete_checkpoint(&self, step: u64) -> Result<bool> {
        if let Some(manifest) = self.find_manifest_by_step(step).await? {
            // Delete all shards
            for shard in &manifest.shards {
                let shard_uri = format!("{}/{}", self.base_uri, shard.key);
                let _ = self.store.delete(&shard_uri).await; // Best effort
            }

            // Delete manifest
            let _manifest_uri = format!("{}/manifests/ckpt-{}-*.json", self.base_uri, step);
            let prefix = format!("{}/manifests/", self.base_uri);
            let uris = self.store.list(&prefix, false).await?;

            for uri in uris {
                if let Ok(rel_key) = self.extract_relative_key(&uri) {
                    if let Ok((manifest_step, _)) = KeyLayout::parse_manifest_key(&rel_key) {
                        if manifest_step == step {
                            self.store.delete(&uri).await?;
                            break;
                        }
                    }
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Extract relative key from full URI
    fn extract_relative_key(&self, full_uri: &str) -> Result<String> {
        let expected_prefix = format!("{}/", self.base_uri);
        if let Some(relative) = full_uri.strip_prefix(&expected_prefix) {
            Ok(relative.to_string())
        } else {
            anyhow::bail!("URI does not match base: {}", full_uri)
        }
    }

    /// Validate checkpoint integrity
    pub async fn validate_checkpoint(&self, manifest: &Manifest) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        // Validate manifest itself
        if let Err(e) = manifest.validate() {
            errors.push(format!("Manifest validation failed: {}", e));
        }

        // Check that all shards exist and have correct sizes
        for shard in &manifest.shards {
            let shard_uri = format!("{}/{}", self.base_uri, shard.key);
            match self.store.stat(&shard_uri).await {
                Ok(meta) => {
                    if meta.size != shard.size {
                        errors.push(format!(
                            "Shard {} size mismatch: expected {}, got {}",
                            shard.rank, shard.size, meta.size
                        ));
                    }
                    // Check ETag if available
                    if let (Some(expected), Some(actual)) = (&shard.etag, &meta.e_tag) {
                        if expected != actual {
                            errors.push(format!(
                                "Shard {} ETag mismatch: expected {}, got {}",
                                shard.rank, expected, actual
                            ));
                        }
                    }
                }
                Err(e) => {
                    errors.push(format!("Shard {} not accessible: {}", shard.rank, e));
                }
            }
        }

        Ok(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::writer::Writer;
    use crate::object_store::store_for_uri;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_single_checkpoint() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        // Write a checkpoint first
        let writer = Writer::new(&*store, base_uri.clone(), 1, 0);
        let data = b"test checkpoint data";
        writer.save_checkpoint(100, 10, "torch", data, None).await?;

        // Now read it back
        let reader = Reader::new(&*store, base_uri);

        let latest = reader.get_latest().await?.unwrap();
        assert_eq!(latest.global_step, 100);

        let manifest = reader.load_latest_manifest().await?.unwrap();
        assert_eq!(manifest.global_step, 100);
        assert_eq!(manifest.world_size, 1);

        let shard_data = reader.read_shard_by_rank(&manifest, 0).await?;
        assert_eq!(&shard_data[..], data);

        Ok(())
    }

    #[tokio::test]
    async fn test_read_distributed_checkpoint() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        // Write a distributed checkpoint
        let writer_0 = Writer::new(&*store, base_uri.clone(), 2, 0);
        let writer_1 = Writer::new(&*store, base_uri.clone(), 2, 1);

        let (layout, shard_0) = writer_0
            .save_distributed_shard(200, 20, "torch", b"rank 0 data")
            .await?;
        let (_, shard_1) = writer_1
            .save_distributed_shard(200, 20, "torch", b"rank 1 data")
            .await?;

        writer_0
            .finalize_distributed_checkpoint(&layout, "torch", 20, vec![shard_0, shard_1], None)
            .await?;

        // Now read it back
        let reader = Reader::new(&*store, base_uri);

        let manifest = reader.load_latest_manifest().await?.unwrap();
        assert_eq!(manifest.world_size, 2);

        let all_shards = reader.read_all_shards(&manifest).await?;
        assert_eq!(all_shards.len(), 2);
        assert_eq!(all_shards[0].0, 0); // rank 0
        assert_eq!(all_shards[1].0, 1); // rank 1
        assert_eq!(&all_shards[0].1[..], b"rank 0 data");
        assert_eq!(&all_shards[1].1[..], b"rank 1 data");

        Ok(())
    }

    #[tokio::test]
    async fn test_list_and_find_checkpoints() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        let writer = Writer::new(&*store, base_uri.clone(), 1, 0);

        // Create multiple checkpoints
        writer
            .save_checkpoint(100, 10, "torch", b"data1", None)
            .await?;
        writer
            .save_checkpoint(200, 20, "torch", b"data2", None)
            .await?;
        writer
            .save_checkpoint(300, 30, "torch", b"data3", None)
            .await?;

        let reader = Reader::new(&*store, base_uri);

        // List all checkpoints
        let checkpoints = reader.list_checkpoints().await?;
        assert_eq!(checkpoints.len(), 3);
        assert_eq!(checkpoints[0].0, 100);
        assert_eq!(checkpoints[1].0, 200);
        assert_eq!(checkpoints[2].0, 300);

        // Find specific checkpoint
        let manifest = reader.find_manifest_by_step(200).await?.unwrap();
        assert_eq!(manifest.global_step, 200);

        // Latest should be step 300
        let latest_manifest = reader.load_latest_manifest().await?.unwrap();
        assert_eq!(latest_manifest.global_step, 300);

        Ok(())
    }

    #[tokio::test]
    async fn test_checkpoint_validation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        let writer = Writer::new(&*store, base_uri.clone(), 1, 0);
        writer
            .save_checkpoint(100, 10, "torch", b"test data", None)
            .await?;

        let reader = Reader::new(&*store, base_uri);
        let manifest = reader.load_latest_manifest().await?.unwrap();

        let errors = reader.validate_checkpoint(&manifest).await?;
        assert!(errors.is_empty(), "Validation should pass: {:?}", errors);

        Ok(())
    }
}
