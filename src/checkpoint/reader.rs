// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use super::latest::{delete_latest, read_latest, write_latest, Latest};
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

/// The `<step>` of a `ckpt-<step>-<ts>` path component (a checkpoint's shard directory, or the
/// stem of its manifest), `None` for any other name. Shares its rule with [`manifest_hint`].
fn ckpt_dir_step(component: &str) -> Option<u64> {
    let id = component.strip_prefix("ckpt-")?;
    let (step, ts) = id.split_once('-')?;
    if ts.is_empty() {
        return None;
    }
    step.parse().ok()
}

/// Check that `key`, a shard key read from a manifest, is safe to delete (review of F-F2,
/// issue #291). `own_step` is the step of the manifest being deleted (from its key).
///
/// A manifest is data in the checkpoint prefix, so a shard key is not trusted. The writer's own
/// keys are `shards/ckpt-<step>-<ts>/rank-<n>.bin` (also under `shards/rr/..` and `shards/bin/..`),
/// but `ShardWriter::put_shard_with_key` legitimately stores a shard under any caller-chosen key
/// (`custom/one.bin`, `shard-0`), so the rule cannot require the `shards/` prefix. It requires:
///
/// * a relative key: non-empty, no leading `/`, no backslash or NUL (a `file://` base would turn a
///   backslash into a separator on some hosts);
/// * no `?`, no `#` and no percent-encoded dot (`%2e`, any case, anywhere): an HTTP-based store
///   parses the key as a URL, which truncates it at `?`/`#` and normalises `%2e%2e` to `..`, so
///   it could address an object outside the base. (A `%` that does not encode a dot is fine.)
/// * every `/`-separated component non-empty and neither `.` nor `..` (no escape from the base on
///   a `file://` base, no `a//b`, and no trailing `/`, so the final component names an object
///   and not a prefix);
/// * not a checkpoint-control object or directory, at ANY position: no component named
///   `manifests` (another checkpoint's manifest), `latest` (the append-only pointer markers),
///   `latest.json` or `latest.json.tmp`, and no FINAL component `shards` (the directory of every
///   checkpoint's shards, which a `file://` store would remove whole);
/// * no component naming the shard directory `ckpt-<step>-<ts>` of a DIFFERENT step, at any
///   position. A distributed save stamps each rank's directory with its own timestamp, so the
///   timestamp of the manifest cannot be required to match, but the step can. (Two saves of one
///   step cannot be told apart by name; `Reader::delete_checkpoint` keeps a shard that another
///   complete manifest lists too.)
///
/// On a `file://` or `direct://` base `Reader::delete_checkpoint` also stats every target and
/// refuses a directory (its store would remove it whole).
///
/// A key that fails the rule makes `delete_checkpoint` fail before anything is deleted.
fn validate_shard_key(key: &str, own_step: u64) -> Result<()> {
    let reject = |why: &str| -> Result<()> {
        anyhow::bail!(
            "refusing to delete: manifest shard key \"{}\" is not a safe shard key ({why})",
            crate::redact::redact_and_escape(key)
        )
    };
    if key.is_empty() {
        return reject("empty");
    }
    if key.starts_with('/') {
        return reject("absolute");
    }
    if key.contains('\\') || key.contains('\0') {
        return reject("contains a backslash or NUL");
    }
    if key.contains('?') || key.contains('#') {
        return reject("contains `?` or `#`, which a URL-based store reads as a query or fragment");
    }
    if key.to_ascii_lowercase().contains("%2e") {
        return reject("contains a percent-encoded dot, which a URL-based store may normalise");
    }
    let mut last = "";
    for component in key.split('/') {
        match component {
            "" => return reject("empty path component or trailing `/`"),
            "." | ".." => return reject("`.` or `..` path component"),
            "manifests" | "latest" | "latest.json" | "latest.json.tmp" => {
                return reject("a checkpoint control name (manifests, latest, latest.json)");
            }
            _ => {}
        }
        if ckpt_dir_step(component).is_some_and(|step| step != own_step) {
            return reject("names the shard directory of another checkpoint's step");
        }
        last = component;
    }
    if last == "shards" {
        return reject("the directory of every checkpoint's shards, not an object");
    }
    Ok(())
}

/// The Azure account as the store that lists it spells it (so as it appears in listed entries).
///
/// This is the rule of `azure_client::canonical_account_name` without its side effects: with a
/// custom Azure endpoint configured the account is an emulator path segment and is kept as
/// written, otherwise it is a host label and is lower-cased. It only READS the process
/// environment (the store that listed the entry has already applied the `.env` policy before
/// it listed), so using the reader never loads `.env` itself, and it needs no Azure feature.
fn listed_account_name(account: &str) -> String {
    let custom_endpoint = [
        crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
        crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
    ]
    .iter()
    .any(|var| std::env::var(var).is_ok_and(|v| !v.trim().is_empty()));
    if custom_endpoint {
        account.to_string()
    } else {
        account.to_ascii_lowercase()
    }
}

/// Where an object lives, by the rules of the provider its URI names, whatever the spelling.
struct ObjectIdentity {
    /// The bucket (`s3:<bucket>`, `gs:<bucket>`) or account and container (`az:<account>/<container>`)
    /// as the provider names it: `gcs://` and `gs://` are one, an Azure blob host URL and `az://`
    /// are one, `s3://host:port/bucket/key` and `s3://bucket/key` are one.
    container: String,
    /// The server an `s3://host:port/...` URI names, lower-cased; `None` for bucket form.
    endpoint: Option<String>,
    /// The key, exactly as written (no decoding, no normalisation).
    key: String,
}

impl ObjectIdentity {
    /// `None` for a URI no provider rule places (a `file://` path, an unknown scheme, a URI the
    /// pure parsers refuse). `uri` is expected to contain a `/` after its container.
    fn of(uri: &str) -> Option<Self> {
        use crate::capabilities::{parse_azure_uri_pure, parse_gs_uri_pure, parse_s3_uri_pure};
        use crate::object_store::{infer_scheme, Scheme};
        match infer_scheme(uri) {
            Scheme::S3 => {
                let parts = parse_s3_uri_pure(uri).ok()?;
                Some(Self {
                    container: format!("s3:{}", parts.bucket),
                    endpoint: parts.endpoint.map(|e| e.to_ascii_lowercase()),
                    key: parts.key,
                })
            }
            Scheme::Azure => {
                let (account, container, key) = parse_azure_uri_pure(uri).ok()?;
                Some(Self {
                    container: format!("az:{}/{container}", listed_account_name(&account)),
                    endpoint: None,
                    key,
                })
            }
            Scheme::Gcs => {
                let (bucket, key) = parse_gs_uri_pure(uri).ok()?;
                Some(Self {
                    container: format!("gs:{bucket}"),
                    endpoint: None,
                    key,
                })
            }
            _ => None,
        }
    }

    /// Whether both name the same container of the same provider (and, when both name a server,
    /// the same server).
    fn same_container(&self, other: &Self) -> bool {
        self.container == other.container
            && match (&self.endpoint, &other.endpoint) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            }
    }
}

/// The key of the listed object `full_uri` relative to `base_uri` (no trailing `/`), or `None`
/// when it is not under the base. The base is followed by exactly one `/` in every spelling
/// (`ckpt` does not own `ckpt2/..`, `ckptmanifests/..` or `ckpt//..`).
///
/// In order: the literal base; for a `file://` or `direct://` base the base with its scheme
/// stripped (their stores list a relative path as a scheme-less `rel/ck/...`); and last the
/// provider identity, which compares the bucket (`gcs://` is `gs://`), the Azure account and
/// container (an Azure blob host URL is `az://`), and the S3 server when both name one
/// (`s3://host:port/bucket/key` is `s3://bucket/key`), then the key against the base's key.
fn relative_to_base(base_uri: &str, full_uri: &str) -> Option<String> {
    // A relative key is not empty (that is the base itself) and never starts with `/` (that is
    // an empty path segment: `ckpt//manifests`).
    let accept = |relative: &str| {
        (!relative.is_empty() && !relative.starts_with('/')).then(|| relative.to_string())
    };

    if let Some(relative) = full_uri.strip_prefix(&format!("{base_uri}/")) {
        if let Some(found) = accept(relative) {
            return Some(found);
        }
    }
    for scheme in ["file://", "direct://"] {
        if let Some(bare) = base_uri.strip_prefix(scheme).filter(|b| !b.is_empty()) {
            if let Some(found) = full_uri.strip_prefix(&format!("{bare}/")).and_then(accept) {
                return Some(found);
            }
        }
    }
    // `base_uri` has no trailing `/`: add one so a bucket-only base (`s3://bucket`) parses.
    let base = ObjectIdentity::of(&format!("{base_uri}/"))?;
    let listed = ObjectIdentity::of(full_uri)?;
    if !base.same_container(&listed) {
        return None;
    }
    let base_key = base.key.trim_end_matches('/');
    if base_key.is_empty() {
        accept(&listed.key)
    } else {
        listed
            .key
            .strip_prefix(base_key)
            .and_then(|rest| rest.strip_prefix('/'))
            .and_then(accept)
    }
}

/// What `latest.json` holds.
enum LatestState {
    /// Missing, or the GET failed.
    Absent,
    /// Readable, but not a pointer.
    Corrupt,
    Valid(Latest),
}

/// The outcome of reading every manifest listed under `manifests/`.
struct ManifestScan {
    /// The newest complete, valid manifest, with its key relative to the base.
    best: Option<(String, Manifest)>,
    /// Listed manifests that could not be read, parsed or placed under the base (redacted).
    unreadable: Vec<String>,
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

    /// Load the latest complete manifest.
    ///
    /// `latest.json` names the manifest. When it names one that no longer exists (typed
    /// not-found, or a listing of `manifests/` that does not hold it: the checkpoint was deleted
    /// and the pointer left behind, issue #290) the newest complete, valid manifest is found by
    /// [`Reader::scan_latest_complete`], as when `latest.json` is absent. An unreadable or
    /// corrupt `latest.json` counts as absent. A manifest that exists but cannot be read, parsed
    /// or validated is still an error.
    ///
    /// Tradeoff of the fallback: the pointer is only taken for dangling when the GET of the
    /// named manifest fails with something other than a typed not-found AND a successful listing
    /// of `manifests/` does not show it. A transient, untyped GET failure of a manifest that
    /// exists but is not yet visible in the listing (an eventually consistent store) therefore
    /// loads the newest remaining complete checkpoint, which can be older than the one named.
    pub async fn load_latest_manifest(&self) -> Result<Option<Manifest>> {
        match self.latest_state().await {
            LatestState::Valid(latest) => {
                match self.load_manifest_if_present(&latest.manifest_key).await? {
                    Some(manifest) => Ok(Some(manifest)),
                    None => self.scan_latest_complete().await,
                }
            }
            // Missing, unreadable or corrupt: scan for the latest complete manifest.
            LatestState::Absent | LatestState::Corrupt => self.scan_latest_complete().await,
        }
    }

    /// What `latest.json` holds: absent (missing, or the GET failed), corrupt (readable but not
    /// a pointer) or a pointer.
    async fn latest_state(&self) -> LatestState {
        let latest_uri = format!("{}/latest.json", self.base_uri);
        match self.store.get(&latest_uri).await {
            Err(_) => LatestState::Absent,
            Ok(bytes) => match serde_json::from_slice::<Latest>(&bytes) {
                Ok(latest) => LatestState::Valid(latest),
                Err(_) => LatestState::Corrupt,
            },
        }
    }

    /// [`Reader::load_manifest`], but `Ok(None)` when the manifest is gone. Gone means the GET
    /// failed with a typed not-found, or failed some other way (the file store's not-found is an
    /// untyped error) and a successful listing of `manifests/` does not hold the object. Any
    /// other failure, and any failure to read or parse a manifest that exists, is an error.
    async fn load_manifest_if_present(&self, manifest_rel_key: &str) -> Result<Option<Manifest>> {
        let uri = format!("{}/{}", self.base_uri, manifest_rel_key);
        let bytes = match self.store.get(&uri).await {
            Ok(bytes) => bytes,
            Err(e) => {
                if crate::delete_contract::is_not_found(&e) {
                    return Ok(None);
                }
                let prefix = format!("{}/manifests/", self.base_uri);
                let listed = match self.store.list(&prefix, false).await {
                    Ok(listed) => listed,
                    Err(_) => return Err(e),
                };
                let held = listed.iter().any(|entry| {
                    relative_to_base(&self.base_uri, entry).as_deref() == Some(manifest_rel_key)
                });
                return if held { Err(e) } else { Ok(None) };
            }
        };
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        Ok(Some(manifest))
    }

    /// Fallback if latest.json is missing, corrupt or racy: scan manifests/ and select the max
    /// (step, ts) among the complete manifests that pass [`Manifest::validate`] and whose key is
    /// `ckpt-<step>-<ts>.json` with `<step>` equal to the manifest's `global_step`. A manifest
    /// that cannot be read, parsed or validated, or whose key does not match, is skipped.
    pub async fn scan_latest_complete(&self) -> Result<Option<Manifest>> {
        Ok(self
            .scan_manifests(None)
            .await?
            .best
            .map(|(_, manifest)| manifest))
    }

    /// Read every listed manifest (a bounded number at a time) and keep the newest complete,
    /// valid one (max `(global_step, wall_time)`), leaving out the listed manifest whose key
    /// relative to the base is `exclude`. Reads go through the caller's base, like every other
    /// read of a manifest. A manifest that is not complete, or fails `validate`, is skipped; one
    /// whose name is `ckpt-<step>-<ts>.json` and that cannot be read or parsed is recorded in
    /// [`ManifestScan::unreadable`] (it may be the newest checkpoint).
    async fn scan_manifests(&self, exclude: Option<&str>) -> Result<ManifestScan> {
        let prefix = format!("{}/manifests/", self.base_uri);
        let uris = self.store.list(&prefix, false).await?;

        let mut scan = ManifestScan {
            best: None,
            unreadable: Vec::new(),
        };
        let mut rel_keys: Vec<String> = Vec::new();
        for uri in uris {
            if !uri.ends_with(".json") {
                continue;
            }
            match relative_to_base(&self.base_uri, &uri) {
                Some(rel) if exclude == Some(rel.as_str()) => {}
                Some(rel) => rel_keys.push(rel),
                None => scan.unreadable.push(crate::redact::redact_and_escape(&uri)),
            }
        }

        for chunk in rel_keys.chunks(MANIFEST_SCAN_CONCURRENCY) {
            let reads: Vec<_> = chunk
                .iter()
                .map(|rel| self.load_manifest_lenient(rel))
                .collect();
            for (rel, read) in chunk.iter().zip(futures::future::join_all(reads).await) {
                let manifest = match read {
                    Ok(manifest) => manifest,
                    Err(_) => {
                        if KeyLayout::parse_manifest_key(rel).is_ok() {
                            scan.unreadable.push(crate::redact::redact_and_escape(rel));
                        }
                        continue;
                    }
                };
                // Only complete, valid manifests are candidates, and only ones whose key is
                // `ckpt-<step>-<ts>.json` with `<step>` equal to their `global_step`.
                if manifest.status != "complete" || manifest.validate().is_err() {
                    continue;
                }
                if !matches!(
                    KeyLayout::parse_manifest_key(rel),
                    Ok((step, _)) if step == manifest.global_step
                ) {
                    continue;
                }
                let is_better = match &scan.best {
                    None => true,
                    Some((_, prev)) => {
                        (manifest.global_step, &manifest.wall_time)
                            > (prev.global_step, &prev.wall_time)
                    }
                };
                if is_better {
                    scan.best = Some((rel.clone(), manifest));
                }
            }
        }
        Ok(scan)
    }

    /// GET and parse a manifest, without validating it.
    async fn load_manifest_lenient(&self, manifest_rel_key: &str) -> Result<Manifest> {
        let uri = format!("{}/{}", self.base_uri, manifest_rel_key);
        let bytes = self.store.get(&uri).await?;
        Ok(serde_json::from_slice(&bytes)?)
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
    ///
    /// A store lists manifests in its own spelling of the base (`az://` for an Azure blob host
    /// URL, `gs://` for `gcs://`, bucket-form `s3://` for `s3://host:port/bucket/..`); they are
    /// matched to the base by provider identity (review F-F2). A listed `.json` object that
    /// cannot be placed under the base is an error naming it, never a manifest silently left
    /// out.
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
            let rel_key = self.extract_relative_key(&uri)?;
            if !tried.contains(&rel_key) {
                keys.push(rel_key);
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

            // Extract step from filename (an entry that cannot be placed under the base is an error)
            let rel_key = self.extract_relative_key(&uri)?;
            if let Ok((manifest_step, _)) = KeyLayout::parse_manifest_key(&rel_key) {
                if manifest_step == step {
                    return Ok(Some(self.load_manifest(&rel_key).await?));
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

            let rel_key = self.extract_relative_key(&uri)?;
            if let Ok((step, timestamp)) = KeyLayout::parse_manifest_key(&rel_key) {
                // Try to load manifest to get status
                if let Ok(manifest) = self.load_manifest(&rel_key).await {
                    checkpoints.push((step, timestamp, manifest.status));
                }
            }
        }

        // Sort by step
        checkpoints.sort_by_key(|(step, _, _)| *step);
        Ok(checkpoints)
    }

    /// Delete a checkpoint (manifest and all shards).
    ///
    /// Returns `Ok(true)` when a checkpoint of `step` was found and its manifest deleted,
    /// `Ok(false)` when none exists.
    ///
    /// **One checkpoint per call.** Several manifests can share a step (two saves of one step at
    /// different times). Exactly one is deleted per call: the first the store lists (the
    /// listing order is the store's and is not specified), whatever its status. Call again to
    /// delete the next; the result is `Ok(false)` once none is left. Nothing is deleted for the
    /// other manifests of the step, and `latest.json` is only changed when it names the manifest
    /// that was deleted (or is corrupt).
    ///
    /// The manifest is selected by its FILE NAME (`ckpt-<step>-<ts>.json`) and must also record
    /// that step (`global_step`); otherwise the call fails with nothing deleted.
    ///
    /// **Coordinate saves and deletes.** The caller must quiesce checkpoint saves and
    /// finalizations under the same base while a delete runs: no transaction or lock is
    /// provided, and shared-shard protection uses a listing snapshot, so a checkpoint published
    /// after it can lose a shard it lists.
    ///
    /// Everything that can fail is settled BEFORE the first delete: the manifests are listed
    /// once, every listed `.json` entry must be placeable under the base (an entry that is not
    /// is an error, as in `find_manifest_by_step`), the manifest of `step` is read, parsed and
    /// validated, and every shard key it lists must be a safe key (issue #291): relative, with a non-empty
    /// final component, no `.`/`..`/empty component, no `?`, `#` or percent-encoded dot, no
    /// component named `manifests`, `latest`, `latest.json` or `latest.json.tmp` and no final
    /// component `shards` (at any position), no shard directory of another step (at any
    /// position), and not a directory (a `file://` or `direct://` base is statted: its store
    /// would remove a directory whole). See `validate_shard_key`. An unsafe key makes the call
    /// fail with nothing deleted.
    ///
    /// **Shared shards are kept.** Every other listed manifest is read (16 at a time; one that
    /// does not parse or is not complete is ignored). A shard key that another COMPLETE manifest
    /// lists too belongs to that checkpoint as well, so it is not deleted: this checkpoint's
    /// manifest and the shards no other complete manifest lists are deleted, the shared shards
    /// are kept (logged at info level), and the other checkpoint stays whole. If another manifest
    /// cannot be READ (any failure but not-found) the call fails with nothing deleted, since it
    /// could list a shard of this checkpoint. The cost is one listing plus one GET of each other
    /// manifest per call.
    ///
    /// Then, in order:
    /// 1. the manifest is deleted FIRST. If that fails the call fails and nothing else has been
    ///    deleted (shards and manifest are all still present);
    /// 2. when `latest.json` names the deleted manifest, or is corrupt (an unreadable or corrupt
    ///    `latest.json` counts as absent for readers, and a corrupt one is rewritten here), it is
    ///    pointed at the newest remaining complete, valid checkpoint, or deleted when none
    ///    remains (issue #290). Only manifests named `ckpt-<step>-<ts>.json` whose `<step>` equals
    ///    their `global_step` are candidates. This scans (one GET per remaining manifest) only
    ///    when the pointer needs changing. Every remaining listed manifest must be readable for
    ///    this: if one named `ckpt-<step>-<ts>.json` cannot be read or parsed, it could be the
    ///    newest, so
    ///    `latest.json` is left UNTOUCHED and the error says so. The pointer is read again just
    ///    before it is written and left alone if a concurrent save has changed it; a save that
    ///    publishes between that read and the write can still be overwritten (the pointer
    ///    store has no conditional write; readers fall back to a scan only for a dangling
    ///    pointer);
    /// 3. the shards are deleted, best effort: a failure to delete a shard is ignored and the
    ///    call still returns `Ok(true)`. The manifest is already gone, so such a shard is
    ///    unreferenced garbage that a rerun cannot find (the step now reports `Ok(false)`). With
    ///    saves under the base quiesced as required above, no complete manifest points at a
    ///    deleted shard; a checkpoint published concurrently with the delete can.
    ///
    /// If step 2 fails (an unverifiable scan, a failed write) the shards are still deleted and
    /// the call then returns an error saying the checkpoint WAS deleted and `latest.json` was
    /// not updated; `load_latest_manifest` still finds the newest remaining checkpoint, because
    /// it falls back to a scan when the pointer names a missing manifest.
    pub async fn delete_checkpoint(&self, step: u64) -> Result<bool> {
        let prefix = format!("{}/manifests/", self.base_uri);
        let mut manifest_rel_key: Option<String> = None;
        let mut listed: Vec<String> = Vec::new();
        for uri in self.store.list(&prefix, false).await? {
            if !uri.ends_with(".json") {
                continue;
            }
            let rel_key = self.extract_relative_key(&uri)?;
            if manifest_rel_key.is_none()
                && matches!(KeyLayout::parse_manifest_key(&rel_key), Ok((s, _)) if s == step)
            {
                manifest_rel_key = Some(rel_key.clone());
            }
            listed.push(rel_key);
        }
        let Some(rel_key) = manifest_rel_key else {
            return Ok(false);
        };
        let manifest = self.load_manifest(&rel_key).await?;
        // The manifest was selected by its file name: its body must record the same step, or
        // the shards below would belong to a different checkpoint. Nothing is deleted.
        if manifest.global_step != step {
            anyhow::bail!(
                "checkpoint step {step} was not deleted: manifest '{}' is named for step {step} \
                 but records global_step {}",
                crate::redact::redact_and_escape(&rel_key),
                manifest.global_step
            );
        }
        let shared = self
            .check_shards_deletable(step, &rel_key, &listed, &manifest)
            .await
            .with_context(|| format!("checkpoint step {step} was not deleted"))?;

        // Through the caller's base, like every read of this manifest: the listed spelling can
        // be one the store does not accept back (a scheme-less path, `az://` for a blob URL).
        self.store
            .delete(&format!("{}/{}", self.base_uri, rel_key))
            .await?;
        let pointer = self.repoint_latest_after_delete(&rel_key).await;
        for shard in &manifest.shards {
            if shared.contains(&shard.key) {
                tracing::info!(
                    "checkpoint step {step}: kept shard \"{}\", which another complete \
                     checkpoint manifest also lists",
                    crate::redact::redact_and_escape(&shard.key)
                );
                continue;
            }
            let shard_uri = format!("{}/{}", self.base_uri, shard.key);
            let _ = self.store.delete(&shard_uri).await; // Best effort
        }
        pointer.with_context(|| {
            format!(
                "checkpoint step {step} was deleted, but latest.json was not updated \
                 (load_latest still falls back to the newest remaining checkpoint)"
            )
        })?;
        Ok(true)
    }

    /// Every shard key of `manifest` (the manifest `rel_key`, of `step`) must be safe to delete:
    /// see [`Reader::delete_checkpoint`]. Deletes nothing. Returns the keys of `manifest` that
    /// another complete manifest of `listed` also lists: those shards must be kept.
    async fn check_shards_deletable(
        &self,
        step: u64,
        rel_key: &str,
        listed: &[String],
        manifest: &Manifest,
    ) -> Result<HashSet<String>> {
        for shard in &manifest.shards {
            validate_shard_key(&shard.key, step)?;
        }

        // A `file://` or `direct://` store removes a directory whole: stat every target.
        let local_root = ["file://", "direct://"]
            .iter()
            .find_map(|scheme| self.base_uri.strip_prefix(scheme));
        if let Some(root) = local_root {
            for shard in &manifest.shards {
                let path = std::path::Path::new(root).join(&shard.key);
                match tokio::fs::metadata(&path).await {
                    Ok(meta) if meta.is_dir() => anyhow::bail!(
                        "refusing to delete: manifest shard key \"{}\" is not a safe shard key \
                         (it names a directory, not an object)",
                        crate::redact::redact_and_escape(&shard.key)
                    ),
                    Ok(_) => {}
                    Err(e)
                        if matches!(
                            e.kind(),
                            std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                        ) => {}
                    Err(e) => {
                        return Err(anyhow::Error::new(e).context(format!(
                            "cannot check that manifest shard key \"{}\" is not a directory",
                            crate::redact::redact_and_escape(&shard.key)
                        )))
                    }
                }
            }
        }

        // A key another complete manifest lists is that checkpoint's shard as well: keep it.
        let own: HashSet<&str> = manifest.shards.iter().map(|s| s.key.as_str()).collect();
        let mut shared: HashSet<String> = HashSet::new();
        let others: Vec<&String> = listed.iter().filter(|k| k.as_str() != rel_key).collect();
        for chunk in others.chunks(MANIFEST_SCAN_CONCURRENCY) {
            let reads: Vec<_> = chunk
                .iter()
                .map(|key| self.fetch_complete_manifest_unvalidated(key))
                .collect();
            for other in futures::future::join_all(reads).await {
                let Some(other) = other? else { continue };
                for shard in &other.shards {
                    if own.contains(shard.key.as_str()) {
                        shared.insert(shard.key.clone());
                    }
                }
            }
        }
        Ok(shared)
    }

    /// The manifest at `manifest_rel_key` when it is complete. `Ok(None)` when it is gone (typed
    /// not-found), not complete, or does not parse (not a usable checkpoint); any other failure
    /// to read it is an error.
    async fn fetch_complete_manifest_unvalidated(
        &self,
        manifest_rel_key: &str,
    ) -> Result<Option<Manifest>> {
        let uri = format!("{}/{}", self.base_uri, manifest_rel_key);
        let bytes = match self.store.get(&uri).await {
            Ok(bytes) => bytes,
            Err(e) if crate::delete_contract::is_not_found(&e) => return Ok(None),
            Err(e) => {
                return Err(e.context(format!(
                    "cannot rule out that checkpoint manifest '{}' lists a shard key of this \
                     checkpoint",
                    crate::redact::redact_and_escape(manifest_rel_key)
                )))
            }
        };
        Ok(serde_json::from_slice::<Manifest>(&bytes)
            .ok()
            .filter(|m| m.status == "complete"))
    }

    /// After the manifest `deleted_rel_key` is deleted: when `latest.json` names it, or is
    /// corrupt, point `latest.json` at the newest remaining complete, valid manifest, or delete
    /// it when none remains. Leaves `latest.json` alone when it names another manifest, when it
    /// cannot be read at all, and (with an error) when a remaining manifest could not be read
    /// or parsed, since a pointer is only written when it is certain.
    async fn repoint_latest_after_delete(&self, deleted_rel_key: &str) -> Result<()> {
        let latest_uri = format!("{}/latest.json", self.base_uri);
        let needs_repoint = |state: LatestState| match state {
            LatestState::Absent => false,
            LatestState::Corrupt => true,
            LatestState::Valid(current) => current.manifest_key == deleted_rel_key,
        };
        if !needs_repoint(self.latest_state().await) {
            return Ok(());
        }
        let scan = self.scan_manifests(Some(deleted_rel_key)).await?;
        if !scan.unreadable.is_empty() {
            let shown: Vec<&str> = scan.unreadable.iter().take(3).map(String::as_str).collect();
            anyhow::bail!(
                "latest.json was left unchanged: {} remaining manifest(s) could not be read or \
                 parsed ({}{}), so the newest checkpoint cannot be determined",
                scan.unreadable.len(),
                shown.join(", "),
                if scan.unreadable.len() > shown.len() {
                    ", ..."
                } else {
                    ""
                }
            );
        }
        // A concurrent save may have published since the first read: leave its pointer alone.
        if !needs_repoint(self.latest_state().await) {
            return Ok(());
        }
        match scan.best {
            Some((rel_key, manifest)) => {
                let ts = KeyLayout::parse_manifest_key(&rel_key)
                    .map(|(_, ts)| ts)
                    .unwrap_or_else(|_| manifest.wall_time.clone());
                let latest = Latest::new(rel_key, manifest.global_step, ts);
                write_latest(self.store, &latest_uri, &latest).await
            }
            None => match delete_latest(self.store, &latest_uri).await {
                Err(e) if !crate::delete_contract::is_not_found(&e) => Err(e),
                _ => Ok(()),
            },
        }
    }

    /// The key of a LISTED object relative to the checkpoint base (`manifests/ckpt-1-x.json`).
    ///
    /// A store lists an object in its own spelling, which need not be the caller's spelling of
    /// the base: an Azure blob host URL lists as `az://account/container/key`, `gcs://` as
    /// `gs://`, an endpoint-qualified `s3://host:port/bucket/key` as `s3://bucket/key`, and a
    /// relative `file://rel/ck` as the scheme-less path `rel/ck/...`. See [`relative_to_base`].
    /// The result is the listed key exactly as the store returned it; the reads that follow
    /// still go through `self.base_uri`, so the caller's store and endpoint are preserved.
    ///
    /// An entry that cannot be placed under the base (another bucket, container, account or
    /// server, a sibling prefix, or a URI no rule parses) is an error naming it (redacted),
    /// never a silent omission.
    fn extract_relative_key(&self, full_uri: &str) -> Result<String> {
        relative_to_base(&self.base_uri, full_uri).ok_or_else(|| {
            anyhow::anyhow!(
                "listed checkpoint object {} is not under the checkpoint base {}: the store \
                 listed it under a different bucket, container, account or server, or in a \
                 spelling that cannot be mapped to the base",
                crate::redact::redact_and_escape(full_uri),
                crate::redact::redact_and_escape(&self.base_uri)
            )
        })
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

    /// The shard-key rule of `delete_checkpoint` (issue #291).
    #[test]
    fn validate_shard_key_accepts_writer_and_caller_chosen_keys_only() {
        for good in [
            "shards/ckpt-1-t/rank-0.bin",
            "shards/rr/0a/ckpt-1-t/rank-10.bin",
            "shards/bin/0001/ckpt-1-t/rank-1.bin",
            "shards/ckpt-1-t/rank-0.bin.zst",
            "custom/one.bin",
            "shard-0",
            "a.b/c..d/e",
            "manifestsx/a",
            "latest.json.bak",
            // A distributed save stamps each rank's directory itself: same step is fine.
            "shards/ckpt-1-other/rank-1.bin",
            "shards/ckpt-x-y/a",
        ] {
            assert!(validate_shard_key(good, 1).is_ok(), "{good}");
        }
        for bad in [
            "",
            "/abs",
            "../x",
            "a/../x",
            "a/./x",
            "a//x",
            "a/",
            "..",
            ".",
            "a\\b",
            "a\0b",
            "manifests/ckpt-1-t.json",
            "latest/ckpt-1-t.json",
            "latest.json",
            "latest.json.tmp",
            "shards",
            "shards/",
            "shards/ckpt-2-t/rank-0.bin",
            "shards/ckpt-2-t",
            "shards/rr/00/ckpt-2-t/rank-0.bin",
            "shards/bin/0000/ckpt-0-t/rank-0.bin",
        ] {
            assert!(validate_shard_key(bad, 1).is_err(), "{bad:?}");
        }
    }

    /// An HTTP-based store may normalise or truncate a key as a URL, and a control name hidden
    /// deep in a key is still that object (review of the delete work).
    #[test]
    fn validate_shard_key_refuses_url_metacharacters_and_control_components_anywhere() {
        for good in [
            "100%.bin",
            "a%20b",
            "a%2fb",
            "shards/rr/0a/ckpt-1-t/rank-10.bin.zst",
            "shards/bin/00ff/ckpt-1-t/rank-0.bin",
            "shards/ckpt-1-t/shards-0.bin",
            "custom/ckpt-1-t/rank-0.bin",
            "x/manifestsx/y",
            "x/latest.json.bak",
        ] {
            assert!(validate_shard_key(good, 1).is_ok(), "{good}");
        }
        for bad in [
            "a?b",
            "shards/x?y",
            "a#b",
            "shards/ckpt-1-t/rank-0.bin#frag",
            "a/%2e%2e/b",
            "a/%2E%2E/b",
            "a/%2e./b",
            "a/.%2E/b",
            "shards/a%2Ebin",
            "shards/ckpt-1-t/%2e",
            "x/manifests/ckpt-1-t.json",
            "x/y/manifests",
            "x/latest/y",
            "x/latest",
            "x/latest.json",
            "x/latest.json.tmp",
            "x/shards",
            "shards/rr/shards",
            "custom/ckpt-2-t/rank-0.bin",
            "a/b/ckpt-0-t/rank-0.bin",
        ] {
            assert!(validate_shard_key(bad, 1).is_err(), "{bad:?}");
        }
    }

    /// The base is followed by exactly one `/` in every spelling.
    #[test]
    fn relative_to_base_requires_a_single_slash_boundary() {
        let m = "manifests/ckpt-1-x.json";
        for (base, listed_root) in [
            ("s3://bkt/ckpt", "s3://bkt/ckpt"),
            ("s3://127.0.0.1:9000/bkt/ckpt", "s3://bkt/ckpt"),
            (
                "s3://127.0.0.1:9000/bkt/ckpt",
                "s3://127.0.0.1:9000/bkt/ckpt",
            ),
            ("gcs://bkt/ckpt", "gs://bkt/ckpt"),
            ("file:///abs/ckpt", "file:///abs/ckpt"),
        ] {
            assert_eq!(
                relative_to_base(base, &format!("{listed_root}/{m}")).as_deref(),
                Some(m),
                "{base} vs {listed_root}"
            );
            let sibling_root = listed_root.rsplit_once('/').map_or("", |(head, _)| head);
            for bad in [
                format!("{listed_root}2/{m}"),
                format!("{sibling_root}/ckptmanifests/ckpt-1-x.json"),
                format!("{listed_root}//{m}"),
                format!("{listed_root}/"),
                listed_root.to_string(),
            ] {
                assert_eq!(
                    relative_to_base(base, &bad),
                    None,
                    "{base} must not own {bad}"
                );
            }
        }
    }

    /// A bucket-only base owns the whole bucket, but still not an empty path segment.
    #[test]
    fn relative_to_base_bucket_only_base() {
        assert_eq!(
            relative_to_base("s3://bkt", "s3://bkt/manifests/a.json").as_deref(),
            Some("manifests/a.json")
        );
        assert_eq!(
            relative_to_base("s3://bkt", "s3://bkt//manifests/a.json"),
            None
        );
        assert_eq!(
            relative_to_base("s3://bkt", "s3://other/manifests/a.json"),
            None
        );
    }

    /// Two servers are two stores; the same server matches whatever its spelling.
    #[test]
    fn relative_to_base_compares_the_s3_server_when_both_name_one() {
        let m = "manifests/a.json";
        assert_eq!(
            relative_to_base(
                "s3://hosta:9000/bkt/c",
                &format!("s3://hostb:9000/bkt/c/{m}")
            ),
            None
        );
        assert_eq!(
            relative_to_base(
                "s3://HostA:9000/bkt/c",
                &format!("s3://hosta:9000/bkt/c/{m}")
            )
            .as_deref(),
            Some(m)
        );
        assert_eq!(
            relative_to_base(
                "s3://[::1]:9000/bkt/c",
                &format!("s3://[::1]:9000/bkt/c/{m}")
            )
            .as_deref(),
            Some(m)
        );
        assert_eq!(
            relative_to_base(
                "s3://[::1]:9000/bkt/c",
                &format!("s3://[::2]:9000/bkt/c/{m}")
            ),
            None
        );
        assert_eq!(
            relative_to_base("s3://[::1]:9000/bkt/c", &format!("s3://bkt/c/{m}")).as_deref(),
            Some(m)
        );
    }

    /// The stores list a relative `file://` / `direct://` base as a scheme-less path.
    #[test]
    fn relative_to_base_strips_the_scheme_of_a_file_or_direct_base() {
        let m = "manifests/a.json";
        for scheme in ["file", "direct"] {
            let base = format!("{scheme}://rel/ck");
            assert_eq!(
                relative_to_base(&base, &format!("rel/ck/{m}")).as_deref(),
                Some(m),
                "{base}"
            );
            assert_eq!(
                relative_to_base(&base, &format!("../up/{m}")),
                None,
                "{base}"
            );
            assert_eq!(relative_to_base(&base, &format!("rel/ck2/{m}")), None);
            assert_eq!(relative_to_base(&base, "rel/ckmanifests/a.json"), None);
            assert_eq!(relative_to_base(&base, &format!("rel/ck//{m}")), None);
            assert_eq!(relative_to_base(&base, &format!("other/rel/ck/{m}")), None);
            // An absolute base is listed with its scheme: the literal prefix.
            let abs = format!("{scheme}:///abs/ck");
            assert_eq!(
                relative_to_base(&abs, &format!("{scheme}:///abs/ck/{m}")).as_deref(),
                Some(m)
            );
            assert_eq!(
                relative_to_base(&abs, &format!("{scheme}:///abs/ck2/{m}")),
                None
            );
        }
    }

    /// The identity check must not load `.env` (a library call must not touch the environment).
    #[test]
    fn listed_account_name_follows_the_endpoint_variables_without_loading_env() {
        // Pure function of the process environment: the same answer as the production rule.
        let custom = [
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
        ]
        .iter()
        .any(|v| std::env::var(v).is_ok_and(|x| !x.trim().is_empty()));
        let expect = if custom { "MyAcct" } else { "myacct" };
        assert_eq!(listed_account_name("MyAcct"), expect);
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
