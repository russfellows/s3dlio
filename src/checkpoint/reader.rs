// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use super::latest::{read_latest, Latest};
use super::manifest::Manifest;
use super::paths::KeyLayout;
use crate::data_loader::parallel_fetch::DropCancel;
use crate::object_store::{store_for_uri_with_helper_rows, ObjectStore};
use anyhow::{Context, Result};
use bytes::Bytes;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub struct Reader<'a> {
    pub store: &'a dyn ObjectStore,
    pub base_uri: String,
}

/// How many manifests a search reads at once.
const MANIFEST_SCAN_CONCURRENCY: usize = 16;

/// The step and manifest key a shard's directory implies: `<..>/ckpt-<step>-<ts>/<file>` is
/// listed by `manifests/ckpt-<step>-<ts>.json` (see [`KeyLayout`]). `None` when the shard is
/// not in a `ckpt-<step>-<ts>` directory (a caller-chosen key).
fn manifest_hint(shard_rel_key: &str) -> Option<(u64, String)> {
    let mut segments = shard_rel_key.rsplit('/');
    segments.next()?; // the shard file name
    let dir = segments.next()?;
    let id = dir.strip_prefix("ckpt-")?;
    let (step, ts) = id.split_once('-')?;
    if ts.is_empty() {
        return None;
    }
    Some((step.parse().ok()?, format!("manifests/ckpt-{id}.json")))
}

/// A complete manifest that lists a shard, reduced to what the choice between several needs.
struct ManifestMatch {
    /// `wall_time` as an instant; `None` when it is not valid RFC 3339.
    instant: Option<chrono::DateTime<chrono::FixedOffset>>,
    manifest_key: String,
    checksum: Option<String>,
}

impl ManifestMatch {
    /// Newer instant wins; an unparseable `wall_time` is older than any parseable one; a tie
    /// goes to the lexically larger manifest key.
    fn is_newer_than(&self, other: &Self) -> bool {
        (self.instant, &self.manifest_key) > (other.instant, &other.manifest_key)
    }
}

/// `Some` when `manifest` is complete and lists `shard_rel_key`.
fn manifest_match(
    manifest: &Manifest,
    manifest_key: &str,
    shard_rel_key: &str,
) -> Option<ManifestMatch> {
    if manifest.status != "complete" {
        return None;
    }
    let shard = manifest.shards.iter().find(|s| s.key == shard_rel_key)?;
    Some(ManifestMatch {
        instant: chrono::DateTime::parse_from_rfc3339(&manifest.wall_time).ok(),
        manifest_key: manifest_key.to_string(),
        checksum: shard.checksum.clone(),
    })
}

/// Fetch a shard and decode it, validating `expected_checksum` when there is one.
///
/// A checksum a writer reports (and the manifest stores) covers the UNCOMPRESSED shard
/// bytes. A `.zst` shard is stored compressed, so for those the object is fetched,
/// decompressed, and the decompressed bytes are checked; checking the stored (compressed)
/// bytes could never match. An unsupported or malformed checksum is refused before
/// anything is fetched.
async fn fetch_shard_validated(
    store: &dyn ObjectStore,
    uri: &str,
    shard_rel_key: &str,
    expected_checksum: Option<&str>,
) -> Result<Bytes> {
    if let Some(expected) = expected_checksum {
        crate::checksum::parse_expected_crc32c(expected)?;
    }
    if shard_rel_key.ends_with(".zst") {
        use std::io::Read;
        let stored = store.get(uri).await?;
        let mut decoder = zstd::Decoder::new(&stored[..])?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        if let Some(expected) = expected_checksum {
            crate::checksum::verify_crc32c(&decompressed, expected, uri)?;
        }
        Ok(Bytes::from(decompressed))
    } else {
        // Return Bytes directly - zero copy!
        store.get_with_validation(uri, expected_checksum).await
    }
}

/// Split a shard URI into the checkpoint base URI and the shard key relative to it.
///
/// Shards always live under the base's `shards/` directory whatever the key-layout
/// strategy (`shards/ckpt-..`, `shards/rr/..`, `shards/bin/..`), so the last `/shards/`
/// separates the two (issue #287). Only the path is searched: a bucket or container that is
/// itself named `shards` (`s3://shards/rank-0.bin`) is the authority, not a shard directory.
pub fn split_shard_uri(uri: &str) -> Result<(&str, &str)> {
    // The path starts at the first '/' after `scheme://authority`; without a scheme the whole
    // string is the path.
    let path_start = match uri.find("://") {
        Some(scheme_end) => {
            let authority_start = scheme_end + "://".len();
            uri[authority_start..]
                .find('/')
                .map_or(uri.len(), |i| authority_start + i)
        }
        None => 0,
    };
    match uri[path_start..].rfind("/shards/") {
        Some(rel) if path_start + rel > 0 && uri.len() > path_start + rel + "/shards/".len() => {
            let pos = path_start + rel;
            Ok((&uri[..pos], &uri[pos + 1..]))
        }
        _ => anyhow::bail!(
            "cannot validate integrity of '{uri}': not a checkpoint shard URI \
             (expected '<checkpoint base>/shards/<shard key>'); pass validate_integrity=False \
             to read it without verification"
        ),
    }
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
    match expected_checksum {
        Some(_) => {
            fetch_shard_validated(store.as_ref(), &uri, shard_rel_key, expected_checksum).await
        }
        None => {
            let data = store.get(&uri).await?;
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
        // A `.zst` shard is decompressed first: the checksum covers the uncompressed bytes.
        fetch_shard_validated(self.store, &uri, shard_rel_key, expected_checksum).await
    }

    /// Read the shard at `shard_rel_key` and verify it against the checksum its checkpoint
    /// manifest recorded when the shard was written (issue #287).
    ///
    /// Only manifests whose `status` is `complete` count (as in
    /// [`Reader::scan_latest_complete`]). Precedence, exactly:
    /// 1. the manifest named after the shard directory (`shards/ckpt-<step>-<ts>/..` is listed
    ///    by `manifests/ckpt-<step>-<ts>.json`, which `save_checkpoint` and
    ///    `save_distributed_shard` produce) is used when it is complete and lists the shard --
    ///    one GET in the common case, and no other manifest is compared with it, even a
    ///    newer one;
    /// 2. only otherwise, the newest complete manifest of the same step
    ///    (`manifests/ckpt-<step>-*.json`, where a writer that stamps the manifest later than
    ///    the shard puts it) that lists the shard is used, "newest" being the parsed
    ///    `wall_time` compared as an RFC 3339 instant;
    /// 3. only when none of those lists it, the newest such manifest among all the others
    ///    (a shard stored under a caller-chosen key), by the same rule.
    ///
    /// A manifest whose `wall_time` does not parse ranks older than every parseable one, and
    /// ties go to the lexically larger manifest key.
    ///
    /// The check is never skipped silently: when no complete manifest lists the shard, or the
    /// manifest recorded no checksum for it, this returns an error that says so, and a
    /// mismatch is an "Integrity validation failed" error. A `.zst` shard is decoded first,
    /// exactly as in [`Reader::read_shard_with_validation`].
    ///
    /// A manifest that cannot be read is never taken for "does not list this shard": in
    /// steps 2 and 3 a listed manifest that is gone (typed not-found) is skipped, but any other
    /// failure to read, parse or validate a manifest is returned as the error. (Step 1 is a
    /// guess at a name; if it fails for any reason the listing steps decide.)
    pub async fn read_shard_against_manifest(&self, shard_rel_key: &str) -> Result<Bytes> {
        let hint = manifest_hint(shard_rel_key);
        let mut tried: HashSet<String> = HashSet::new();

        // Step 1: the manifest the shard directory names.
        if let Some((_, manifest_key)) = &hint {
            if let Ok(outcome) = self.fetch_complete_manifest(manifest_key).await {
                tried.insert(manifest_key.clone());
                if let Some(found) =
                    outcome.and_then(|m| manifest_match(&m, manifest_key, shard_rel_key))
                {
                    return self.read_with_recorded(shard_rel_key, found.checksum).await;
                }
            }
        }

        // Steps 2 and 3 share one LIST of the manifests.
        let prefix = format!("{}/manifests/", self.base_uri);
        let mut keys: Vec<String> = Vec::new();
        for uri in self.store.list(&prefix, false).await? {
            if !uri.ends_with(".json") {
                continue;
            }
            if let Ok(rel_key) = self.extract_relative_key(&uri) {
                if !tried.contains(&rel_key) {
                    keys.push(rel_key);
                }
            }
        }

        let mut found = None;
        if let Some((step, _)) = &hint {
            let same_step: Vec<String> = keys
                .iter()
                .filter(|k| matches!(KeyLayout::parse_manifest_key(k), Ok((s, _)) if s == *step))
                .cloned()
                .collect();
            found = self.newest_listing(&same_step, shard_rel_key).await?;
            tried.extend(same_step);
        }
        if found.is_none() {
            keys.retain(|k| !tried.contains(k));
            found = self.newest_listing(&keys, shard_rel_key).await?;
        }

        match found {
            None => anyhow::bail!(
                "cannot validate integrity of '{}/{}': no complete checkpoint manifest under \
                 '{}' lists this shard (pass validate_integrity=False to read it without \
                 verification)",
                self.base_uri,
                shard_rel_key,
                self.base_uri
            ),
            Some(found) => self.read_with_recorded(shard_rel_key, found.checksum).await,
        }
    }

    /// Validate-read `shard_rel_key` against the checksum a manifest recorded for it.
    async fn read_with_recorded(
        &self,
        shard_rel_key: &str,
        checksum: Option<String>,
    ) -> Result<Bytes> {
        match checksum {
            None => anyhow::bail!(
                "cannot validate integrity of '{}/{}': the checkpoint manifest recorded no \
                 checksum for this shard (pass validate_integrity=False to read it without \
                 verification)",
                self.base_uri,
                shard_rel_key
            ),
            Some(checksum) => {
                self.read_shard_with_validation(shard_rel_key, Some(&checksum))
                    .await
            }
        }
    }

    /// Read the manifest at `manifest_rel_key`. `Ok(None)` when it is absent (typed
    /// not-found) or not `complete`; any other failure is an error.
    async fn fetch_complete_manifest(&self, manifest_rel_key: &str) -> Result<Option<Manifest>> {
        let uri = format!("{}/{}", self.base_uri, manifest_rel_key);
        let bytes = match self.store.get(&uri).await {
            Ok(bytes) => bytes,
            Err(e) if crate::delete_contract::is_not_found(&e) => return Ok(None),
            Err(e) => return Err(e.context(format!("failed to read checkpoint manifest '{uri}'"))),
        };
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse checkpoint manifest '{uri}'"))?;
        if manifest.status != "complete" {
            return Ok(None);
        }
        manifest
            .validate()
            .with_context(|| format!("invalid checkpoint manifest '{uri}'"))?;
        Ok(Some(manifest))
    }

    /// `Some` when the manifest at `manifest_key` is complete and lists `shard_rel_key`.
    async fn listing_of(
        &self,
        manifest_key: &str,
        shard_rel_key: &str,
    ) -> Result<Option<ManifestMatch>> {
        let manifest = self.fetch_complete_manifest(manifest_key).await?;
        Ok(manifest.and_then(|m| manifest_match(&m, manifest_key, shard_rel_key)))
    }

    /// Read the named manifests (a bounded number at a time) and return the newest complete
    /// one that lists `shard_rel_key`. The first read failure aborts the search.
    async fn newest_listing(
        &self,
        manifest_keys: &[String],
        shard_rel_key: &str,
    ) -> Result<Option<ManifestMatch>> {
        let mut best: Option<ManifestMatch> = None;
        for chunk in manifest_keys.chunks(MANIFEST_SCAN_CONCURRENCY) {
            // The futures are built up front (not through a stream adaptor holding a closure),
            // which keeps this future provably `Send` for callers that spawn it.
            let reads: Vec<_> = chunk
                .iter()
                .map(|key| self.listing_of(key, shard_rel_key))
                .collect();
            for candidate in futures::future::try_join_all(reads)
                .await?
                .into_iter()
                .flatten()
            {
                if best.as_ref().is_none_or(|b| candidate.is_newer_than(b)) {
                    best = Some(candidate);
                }
            }
        }
        Ok(best)
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
    /// is looked up once via `store_for_uri_with_helper_rows(&self.base_uri)` (as before W2)
    /// and cloned into each spawn. The factory is deterministic for a given URI,
    /// so the freshly-looked-up store is functionally equivalent to the
    /// borrowed `self.store` for the read+decompress leaf operation.
    pub async fn read_all_shards_concurrent(
        &self,
        manifest: &Manifest,
    ) -> Result<Vec<(u32, Bytes)>> {
        let cancel = CancellationToken::new();
        let _drop_cancel = DropCancel(cancel.clone());
        let store_arc: Arc<dyn ObjectStore> =
            Arc::from(store_for_uri_with_helper_rows(&self.base_uri)?);
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
        let store_arc: Arc<dyn ObjectStore> =
            Arc::from(store_for_uri_with_helper_rows(&self.base_uri)?);
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

    #[test]
    fn split_shard_uri_finds_base_and_key_for_every_layout() {
        assert_eq!(
            split_shard_uri("file:///d/run/shards/ckpt-1-ts/rank-0.bin").unwrap(),
            ("file:///d/run", "shards/ckpt-1-ts/rank-0.bin")
        );
        assert_eq!(
            split_shard_uri("s3://b/p/shards/rr/03/ckpt-1-ts/rank-3.bin.zst").unwrap(),
            ("s3://b/p", "shards/rr/03/ckpt-1-ts/rank-3.bin.zst")
        );
        // A base that itself contains "shards" splits at the LAST separator.
        assert_eq!(
            split_shard_uri("file:///a/shards/run/shards/ckpt-1-ts/rank-0.bin").unwrap(),
            ("file:///a/shards/run", "shards/ckpt-1-ts/rank-0.bin")
        );
        for bad in [
            "file:///d/run",
            "file:///d/run/shards/",
            "/shards/x",
            "shards/x",
        ] {
            assert!(split_shard_uri(bad).is_err(), "{bad} must be rejected");
        }
    }

    #[tokio::test]
    async fn read_shard_against_manifest_round_trip_and_corruption() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        let writer = Writer::new(&*store, base_uri.clone(), 1, 0);
        let data = b"payload recorded with a checksum";
        writer.save_checkpoint(5, 1, "torch", data, None).await?;

        let reader = Reader::new(&*store, base_uri.clone());
        let manifest = reader.load_latest_manifest().await?.unwrap();
        let shard = &manifest.shards[0];
        assert!(
            shard.checksum.is_some(),
            "an uncompressed shard must record its checksum"
        );

        let got = reader.read_shard_against_manifest(&shard.key).await?;
        assert_eq!(&got[..], data);

        // Corrupt the stored shard in place: the recorded checksum must reject it.
        let path = temp_dir.path().join(&shard.key);
        let mut bad = data.to_vec();
        bad[0] ^= 0xff;
        std::fs::write(&path, &bad)?;
        let err = reader
            .read_shard_against_manifest(&shard.key)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Integrity validation failed"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_reports_missing_checksum_and_unlisted_shard() -> Result<()>
    {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        // A checkpoint whose manifest recorded no checksum (streamed shard without one).
        let writer = Writer::new(&*store, base_uri.clone(), 1, 0);
        let layout = KeyLayout::new(base_uri.clone(), 9);
        let key = layout.shard_key(0, writer.strategy);
        store
            .put(
                &layout.to_uri(&base_uri, &key),
                Bytes::from_static(b"no sum"),
            )
            .await?;
        let meta = writer.finalize_shard_meta(&layout, key.clone()).await?;
        assert!(meta.checksum.is_none());
        writer
            .finalize_distributed_checkpoint(&layout, "torch", 1, vec![meta], None)
            .await?;

        let reader = Reader::new(&*store, base_uri);
        let err = reader.read_shard_against_manifest(&key).await.unwrap_err();
        assert!(
            err.to_string().contains("recorded no checksum"),
            "unexpected error: {err}"
        );

        let err = reader
            .read_shard_against_manifest("shards/ckpt-0-none/rank-0.bin")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("lists this shard"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    // ---- #287 follow-up: manifest lookup scaling, selection and error handling ----

    use crate::object_store::{ObjectMetadata, ObjectWriter};
    use async_trait::async_trait;
    use futures::Stream;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// How a manifest GET fails in [`ProbeStore`].
    enum Injected {
        /// A typed "object not found" (a listed key that vanished).
        NotFound,
        /// An untyped failure such as a dropped connection or throttling.
        Transient,
    }

    /// A store that delegates to `inner`, counts GETs of manifest objects, and can make
    /// the GET of one manifest fail.
    struct ProbeStore {
        inner: Box<dyn ObjectStore>,
        manifest_gets: AtomicUsize,
        fail_suffix: Option<(String, Injected)>,
    }

    impl ProbeStore {
        fn new(inner: Box<dyn ObjectStore>) -> Self {
            Self {
                inner,
                manifest_gets: AtomicUsize::new(0),
                fail_suffix: None,
            }
        }

        fn failing(inner: Box<dyn ObjectStore>, suffix: &str, how: Injected) -> Self {
            Self {
                fail_suffix: Some((suffix.to_string(), how)),
                ..Self::new(inner)
            }
        }

        fn manifest_gets(&self) -> usize {
            self.manifest_gets.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ObjectStore for ProbeStore {
        async fn get(&self, uri: &str) -> Result<Bytes> {
            if uri.contains("/manifests/") {
                self.manifest_gets.fetch_add(1, Ordering::SeqCst);
                if let Some((suffix, how)) = &self.fail_suffix {
                    if uri.ends_with(suffix.as_str()) {
                        return Err(match how {
                            Injected::NotFound => crate::object_error::attach_object_error(
                                anyhow::anyhow!("injected 404"),
                                crate::object_error::ObjectErrorKind::NotFound,
                                "test",
                                "not found",
                            ),
                            Injected::Transient => anyhow::anyhow!("injected transient failure"),
                        });
                    }
                }
            }
            self.inner.get(uri).await
        }
        async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
            self.inner.get_range(uri, offset, length).await
        }
        async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
            self.inner.put(uri, data).await
        }
        async fn put_multipart(
            &self,
            uri: &str,
            data: Bytes,
            part_size: Option<usize>,
        ) -> Result<()> {
            self.inner.put_multipart(uri, data, part_size).await
        }
        async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
            self.inner.list(uri_prefix, recursive).await
        }
        fn list_stream<'a>(
            &'a self,
            uri_prefix: &'a str,
            recursive: bool,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            self.inner.list_stream(uri_prefix, recursive)
        }
        async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
            self.inner.stat(uri).await
        }
        fn list_objects_with_prefix<'a>(
            &'a self,
            uri_prefix: &'a str,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            self.inner.list_objects_with_prefix(uri_prefix)
        }
        async fn delete(&self, uri: &str) -> Result<()> {
            self.inner.delete(uri).await
        }
        async fn delete_batch(&self, uris: &[String]) -> Result<()> {
            self.inner.delete_batch(uris).await
        }
        async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
            self.inner.delete_prefix(uri_prefix).await
        }
        async fn create_container(&self, name: &str) -> Result<()> {
            self.inner.create_container(name).await
        }
        async fn delete_container(&self, name: &str) -> Result<()> {
            self.inner.delete_container(name).await
        }
        async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
            self.inner.get_writer(uri).await
        }
    }

    const SHARD_DATA: &[u8] = b"the real shard bytes";

    fn good_checksum() -> String {
        crate::checksum::crc32c_label(crate::checksum::crc32c_of(SHARD_DATA))
    }

    fn bogus_checksum() -> String {
        crate::checksum::crc32c_label(crate::checksum::crc32c_of(b"not the shard"))
    }

    /// Put a one-rank manifest named `manifests/<name>.json` that lists `shard_key`.
    async fn put_manifest(
        store: &dyn ObjectStore,
        base_uri: &str,
        name: &str,
        wall_time: &str,
        status: &str,
        shard_key: &str,
        checksum: Option<String>,
    ) -> Result<()> {
        let (step, _) = KeyLayout::parse_manifest_key(&format!("{name}.json"))?;
        let mut manifest = Manifest::new("torch".to_string(), step, 1, 1);
        manifest.wall_time = wall_time.to_string();
        manifest.status = status.to_string();
        manifest.add_shard(crate::checkpoint::manifest::ShardMeta {
            rank: 0,
            key: shard_key.to_string(),
            size: SHARD_DATA.len() as u64,
            etag: None,
            checksum,
        });
        store
            .put(
                &format!("{base_uri}/manifests/{name}.json"),
                Bytes::from(serde_json::to_vec(&manifest)?),
            )
            .await
    }

    async fn put_shard(store: &dyn ObjectStore, base_uri: &str, shard_key: &str) -> Result<()> {
        store
            .put(
                &format!("{base_uri}/{shard_key}"),
                Bytes::from_static(SHARD_DATA),
            )
            .await
    }

    /// `n` unrelated complete checkpoints (steps 1000..) so a scan has something to read.
    async fn put_fillers(store: &dyn ObjectStore, base_uri: &str, n: u64) -> Result<()> {
        for i in 0..n {
            let step = 1000 + i;
            put_manifest(
                store,
                base_uri,
                &format!("ckpt-{step}-fill"),
                "2026-01-01T00:00:00Z",
                "complete",
                &format!("shards/ckpt-{step}-fill/rank-0.bin"),
                Some(bogus_checksum()),
            )
            .await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_reads_one_manifest_when_names_match() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let plain = store_for_uri(&base_uri)?;
        put_fillers(&*plain, &base_uri, 60).await?;
        let key = "shards/ckpt-5-A/rank-0.bin";
        put_shard(&*plain, &base_uri, key).await?;
        put_manifest(
            &*plain,
            &base_uri,
            "ckpt-5-A",
            "2026-10-18T10:00:00Z",
            "complete",
            key,
            Some(good_checksum()),
        )
        .await?;

        let probe = ProbeStore::new(store_for_uri(&base_uri)?);
        let got = Reader::new(&probe, base_uri)
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);
        assert_eq!(
            probe.manifest_gets(),
            1,
            "61 manifests exist; one is needed"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_narrows_to_the_shards_step() -> Result<()> {
        // The Python writer stamps the shard directory and the manifest at different
        // instants, so the manifest name does not match the shard directory.
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let plain = store_for_uri(&base_uri)?;
        put_fillers(&*plain, &base_uri, 60).await?;
        let key = "shards/ckpt-7-shardtime/rank-0.bin";
        put_shard(&*plain, &base_uri, key).await?;
        put_manifest(
            &*plain,
            &base_uri,
            "ckpt-7-manifesttime",
            "2026-10-18T10:00:00Z",
            "complete",
            key,
            Some(good_checksum()),
        )
        .await?;

        let probe = ProbeStore::new(store_for_uri(&base_uri)?);
        let got = Reader::new(&probe, base_uri)
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);
        // The one failed direct probe (no such manifest) plus the one step-7 manifest.
        assert!(
            probe.manifest_gets() <= 2,
            "read {} manifests",
            probe.manifest_gets()
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_falls_back_to_a_full_scan_for_a_custom_key() -> Result<()>
    {
        // put_shard_with_key can store a shard anywhere; the manifest then names that key.
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let plain = store_for_uri(&base_uri)?;
        put_fillers(&*plain, &base_uri, 5).await?;
        let key = "shards/custom/weights.bin";
        put_shard(&*plain, &base_uri, key).await?;
        put_manifest(
            &*plain,
            &base_uri,
            "ckpt-2-x",
            "2026-10-18T10:00:00Z",
            "complete",
            key,
            Some(good_checksum()),
        )
        .await?;

        let got = Reader::new(&*plain, base_uri)
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_ignores_an_incomplete_newer_manifest() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;
        let key = "shards/ckpt-3-S/rank-0.bin";
        put_shard(&*store, &base_uri, key).await?;
        put_manifest(
            &*store,
            &base_uri,
            "ckpt-3-A",
            "2026-10-18T10:00:00Z",
            "complete",
            key,
            Some(good_checksum()),
        )
        .await?;
        // Newer, but still incomplete and carrying a bogus checksum.
        put_manifest(
            &*store,
            &base_uri,
            "ckpt-3-B",
            "2026-10-18T11:00:00Z",
            "incomplete",
            key,
            Some(bogus_checksum()),
        )
        .await?;

        let got = Reader::new(&*store, base_uri)
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_skips_an_incomplete_manifest_named_after_the_shard(
    ) -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;
        let key = "shards/ckpt-3-S/rank-0.bin";
        put_shard(&*store, &base_uri, key).await?;
        // The manifest the shard directory names is incomplete; the complete one is elsewhere.
        put_manifest(
            &*store,
            &base_uri,
            "ckpt-3-S",
            "2026-10-18T11:00:00Z",
            "incomplete",
            key,
            Some(bogus_checksum()),
        )
        .await?;
        put_manifest(
            &*store,
            &base_uri,
            "ckpt-3-A",
            "2026-10-18T10:00:00Z",
            "complete",
            key,
            Some(good_checksum()),
        )
        .await?;

        let got = Reader::new(&*store, base_uri)
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_orders_by_instant_not_by_string() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;
        let key = "shards/ckpt-3-S/rank-0.bin";
        put_shard(&*store, &base_uri, key).await?;
        // 09:00-06:00 is 15:00Z, the NEWER instant, but its text sorts before "10:00:00Z".
        put_manifest(
            &*store,
            &base_uri,
            "ckpt-3-A",
            "2026-10-18T09:00:00-06:00",
            "complete",
            key,
            Some(good_checksum()),
        )
        .await?;
        put_manifest(
            &*store,
            &base_uri,
            "ckpt-3-B",
            "2026-10-18T10:00:00Z",
            "complete",
            key,
            Some(bogus_checksum()),
        )
        .await?;

        let got = Reader::new(&*store, base_uri)
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);
        Ok(())
    }

    #[tokio::test]
    async fn read_shard_against_manifest_surfaces_a_manifest_load_error() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let plain = store_for_uri(&base_uri)?;
        let key = "shards/ckpt-3-S/rank-0.bin";
        put_shard(&*plain, &base_uri, key).await?;
        for name in ["ckpt-3-A", "ckpt-3-B"] {
            put_manifest(
                &*plain,
                &base_uri,
                name,
                "2026-10-18T10:00:00Z",
                "complete",
                key,
                Some(good_checksum()),
            )
            .await?;
        }

        // A transient failure must not be reported as a missing or corrupt shard.
        let probe = ProbeStore::failing(
            store_for_uri(&base_uri)?,
            "ckpt-3-B.json",
            Injected::Transient,
        );
        let err = Reader::new(&probe, base_uri.clone())
            .read_shard_against_manifest(key)
            .await
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("injected transient failure"),
            "unexpected error: {err:#}"
        );

        // A listed manifest that is gone (typed not-found) may be skipped.
        let probe = ProbeStore::failing(
            store_for_uri(&base_uri)?,
            "ckpt-3-B.json",
            Injected::NotFound,
        );
        let got = Reader::new(&probe, base_uri.clone())
            .read_shard_against_manifest(key)
            .await?;
        assert_eq!(&got[..], SHARD_DATA);

        // An unparseable manifest surfaces too.
        plain
            .put(
                &format!("{base_uri}/manifests/ckpt-3-B.json"),
                Bytes::from_static(b"{ not json"),
            )
            .await?;
        let err = Reader::new(&*plain, base_uri)
            .read_shard_against_manifest(key)
            .await
            .unwrap_err();
        assert!(
            !err.to_string().contains("lists this shard"),
            "a corrupt manifest was reported as an unlisted shard: {err:#}"
        );
        Ok(())
    }

    #[test]
    fn split_shard_uri_ignores_a_bucket_named_shards() {
        assert!(split_shard_uri("s3://shards/rank-0.bin").is_err());
        assert!(split_shard_uri("az://shards/rank-0.bin").is_err());
        assert_eq!(
            split_shard_uri("s3://shards/run/shards/ckpt-1-ts/rank-0.bin").unwrap(),
            ("s3://shards/run", "shards/ckpt-1-ts/rank-0.bin")
        );
        assert_eq!(
            split_shard_uri("s3://shards/shards/ckpt-1-ts/rank-0.bin").unwrap(),
            ("s3://shards", "shards/ckpt-1-ts/rank-0.bin")
        );
    }
}
