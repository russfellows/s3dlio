// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Review finding F-F2 (plan v0.10.0, #264/#287): the checkpoint reader must find the manifests
//! a store LISTS even when the store spells the checkpoint base differently from the caller.
//!
//! Supported stores return one object under several spellings: an Azure blob host URL lists as
//! `az://account/container/key`, `gcs://` lists as `gs://`, and an endpoint-qualified S3 URI
//! (`s3://host:port/bucket/key`) lists bucket-form `s3://bucket/key`. The fallback of
//! `Reader::read_shard_against_manifest` (the manifest is not the one the shard directory
//! names: distributed finalisation stamped it later, or the shard has a caller-chosen key) used
//! to discard every listed manifest whose text did not start with the caller's literal base, so
//! a valid checkpoint failed its integrity load.
//!
//! The store here is an in-memory fake that knows each provider's identity rule and lists in the
//! provider's canonical spelling, whatever spelling the caller used. No network, no cloud.

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::{self, Stream, StreamExt};
use s3dlio::checkpoint::manifest::{Manifest, ShardMeta};
use s3dlio::checkpoint::{CheckpointStore, Latest, Reader};
use s3dlio::object_store::ObjectWriter;
use s3dlio::{ObjectMetadata, ObjectStore};
use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Mutex;

const SHARD_DATA: &[u8] = b"the real shard bytes";
const REPLACED_DATA: &[u8] = b"the FAKE shard bytes";

fn checksum_of(data: &[u8]) -> String {
    s3dlio::checksum::crc32c_label(s3dlio::checksum::crc32c_of(data))
}

/// An object's identity and its canonical spelling, by the rules of the provider it names.
/// `None` for a URI no provider rule recognises.
fn identity(uri: &str) -> Option<(String, String)> {
    // Returns (identity string used as the map key, canonical URI).
    if let Some(rest) = uri.strip_prefix("az://") {
        let mut it = rest.splitn(3, '/');
        let (account, container, key) = (it.next()?, it.next()?, it.next().unwrap_or(""));
        let account = account.to_ascii_lowercase();
        return Some((
            format!("az/{account}/{container}/{key}"),
            format!("az://{account}/{container}/{key}"),
        ));
    }
    if let Some((scheme, rest)) = uri.split_once("://") {
        if scheme == "https" || scheme == "http" {
            let (authority, path) = rest.split_once('/')?;
            let host = authority.split(':').next()?.to_ascii_lowercase();
            let account = host.strip_suffix(".blob.core.windows.net")?;
            let (container, key) = path.split_once('/').unwrap_or((path, ""));
            return Some((
                format!("az/{account}/{container}/{key}"),
                format!("az://{account}/{container}/{key}"),
            ));
        }
        if scheme == "gs" || scheme == "gcs" {
            let (bucket, key) = rest.split_once('/').unwrap_or((rest, ""));
            return Some((format!("gs/{bucket}/{key}"), format!("gs://{bucket}/{key}")));
        }
        if scheme == "s3" {
            let (first, after) = rest.split_once('/')?;
            let (bucket, key) = if first.contains(':') {
                // s3://host:port/bucket/key names a server first.
                after.split_once('/').unwrap_or((after, ""))
            } else {
                (first, after)
            };
            return Some((format!("s3/{bucket}/{key}"), format!("s3://{bucket}/{key}")));
        }
    }
    None
}

/// The in-memory provider: accepts any spelling of a name, lists only the canonical one.
#[derive(Default)]
struct SpellingStore {
    objects: Mutex<BTreeMap<String, (String, Bytes)>>,
    /// Entries a hostile or buggy listing adds after the real ones.
    extra_listed: Mutex<Vec<String>>,
    /// Entries a hostile or buggy listing returns ahead of the real ones.
    extra_listed_first: Mutex<Vec<String>>,
}

impl SpellingStore {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<String, (String, Bytes)>>> {
        self.objects.lock().map_err(|_| anyhow!("poisoned"))
    }

    fn not_found(uri: &str) -> anyhow::Error {
        s3dlio::object_error::attach_object_error(
            anyhow!("no such object: {uri}"),
            s3dlio::object_error::ObjectErrorKind::NotFound,
            "test",
            "not found",
        )
    }
}

#[async_trait]
impl ObjectStore for SpellingStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (id, _) = identity(uri).ok_or_else(|| anyhow!("unrecognised URI {uri}"))?;
        match self.lock()?.get(&id) {
            Some((_, data)) => Ok(data.clone()),
            None => Err(Self::not_found(uri)),
        }
    }
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let data = self.get(uri).await?;
        let start = (offset as usize).min(data.len());
        let end = length.map_or(data.len(), |l| (start + l as usize).min(data.len()));
        Ok(data.slice(start..end))
    }
    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        let (id, canonical) = identity(uri).ok_or_else(|| anyhow!("unrecognised URI {uri}"))?;
        self.lock()?.insert(id, (canonical, data));
        Ok(())
    }
    async fn put_multipart(&self, uri: &str, data: Bytes, _part_size: Option<usize>) -> Result<()> {
        self.put(uri, data).await
    }
    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (prefix_id, _) =
            identity(uri_prefix).ok_or_else(|| anyhow!("unrecognised URI {uri_prefix}"))?;
        let mut out: Vec<String> = self
            .extra_listed_first
            .lock()
            .map_err(|_| anyhow!("poisoned"))?
            .clone();
        for (id, (canonical, _)) in self.lock()?.iter() {
            if let Some(rest) = id.strip_prefix(&prefix_id) {
                if recursive || !rest.contains('/') {
                    out.push(canonical.clone());
                }
            }
        }
        out.extend(
            self.extra_listed
                .lock()
                .map_err(|_| anyhow!("poisoned"))?
                .iter()
                .cloned(),
        );
        Ok(out)
    }
    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(
            stream::once(async move { self.list(uri_prefix, recursive).await }).flat_map(|batch| {
                match batch {
                    Ok(items) => stream::iter(items.into_iter().map(Ok)).left_stream(),
                    Err(e) => stream::once(async move { Err(e) }).right_stream(),
                }
            }),
        )
    }
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        bail!("stat is not part of this fake ({uri})")
    }
    async fn delete(&self, uri: &str) -> Result<()> {
        let (id, _) = identity(uri).ok_or_else(|| anyhow!("unrecognised URI {uri}"))?;
        self.lock()?.remove(&id);
        Ok(())
    }
    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        for uri in uris {
            self.delete(uri).await?;
        }
        Ok(())
    }
    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        bail!("delete_prefix is not part of this fake ({uri_prefix})")
    }
    async fn create_container(&self, name: &str) -> Result<()> {
        bail!("create_container is not part of this fake ({name})")
    }
    async fn delete_container(&self, name: &str) -> Result<()> {
        bail!("delete_container is not part of this fake ({name})")
    }
    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        bail!("get_writer is not part of this fake ({uri})")
    }
}

/// Put a one-rank manifest `manifests/<name>.json` listing `shard_key`.
async fn put_manifest(
    store: &SpellingStore,
    base: &str,
    name: &str,
    status: &str,
    shard_key: &str,
    data: &[u8],
) -> Result<()> {
    let step: u64 = name
        .strip_prefix("ckpt-")
        .and_then(|rest| rest.split('-').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("bad manifest name {name}"))?;
    let mut manifest = Manifest::new("torch".to_string(), step, 1, 1);
    manifest.wall_time = "2026-10-18T10:00:00Z".to_string();
    manifest.status = status.to_string();
    manifest.add_shard(ShardMeta {
        rank: 0,
        key: shard_key.to_string(),
        size: data.len() as u64,
        etag: None,
        checksum: Some(checksum_of(data)),
    });
    store
        .put(
            &format!("{base}/manifests/{name}.json"),
            Bytes::from(serde_json::to_vec(&manifest)?),
        )
        .await
}

/// The spellings of F-F2 (plus a mixed-case Azure host and a bracketed IPv6 server): the
/// caller's base, which the store never lists back as written.
const ALL_BASES: &[(&str, &str)] = &[
    (
        "azure-https",
        "https://acct.blob.core.windows.net/ctr/run/ckpts",
    ),
    (
        "azure-https-mixed-case-host",
        "https://Acct.Blob.Core.Windows.Net/ctr/run/ckpts",
    ),
    ("gcs", "gcs://bkt/run/ckpts"),
    ("s3-endpoint", "s3://127.0.0.1:9000/bkt/run/ckpts"),
    ("s3-endpoint-ipv6", "s3://[::1]:9000/bkt/run/ckpts"),
];

/// Whether a custom Azure endpoint is configured in this process. The library then keeps an
/// account name exactly as written (it is a path segment of an emulator URL), so a mixed-case
/// host no longer folds to the lower-case account this fake lists. The environment is only read
/// here, never changed.
fn custom_azure_endpoint_set() -> bool {
    ["AZURE_STORAGE_ENDPOINT", "AZURE_BLOB_ENDPOINT_URL"]
        .iter()
        .any(|var| std::env::var(var).is_ok_and(|v| !v.trim().is_empty()))
}

/// `ALL_BASES`, minus the case that is not defined while a custom Azure endpoint is set.
fn bases() -> Vec<(&'static str, &'static str)> {
    ALL_BASES
        .iter()
        .copied()
        .filter(|(label, _)| {
            let skip = *label == "azure-https-mixed-case-host" && custom_azure_endpoint_set();
            if skip {
                eprintln!(
                    "SKIP {label}: AZURE_STORAGE_ENDPOINT/AZURE_BLOB_ENDPOINT_URL is set, so the \
                     library keeps the account case as written"
                );
            }
            !skip
        })
        .collect()
}

/// A shard whose directory names `ckpt-5-shardtime`, but whose complete manifest was stamped
/// later (`ckpt-5-manifesttime`): the guessed manifest `ckpt-5-shardtime.json` is ABSENT.
const STEP_SHARD: &str = "shards/ckpt-5-shardtime/rank-0.bin";
/// A caller-chosen shard key, which names no manifest at all.
const CUSTOM_SHARD: &str = "shards/custom/weights.bin";

async fn fallback_case(
    label: &str,
    base: &str,
    shard_key: &str,
    manifest_name: &str,
) -> Result<()> {
    let store = SpellingStore::default();
    store
        .put(
            &format!("{base}/{shard_key}"),
            Bytes::from_static(SHARD_DATA),
        )
        .await?;
    put_manifest(
        &store,
        base,
        manifest_name,
        "complete",
        shard_key,
        SHARD_DATA,
    )
    .await?;
    // The guess a direct lookup makes is absent, so only the listing can find the manifest.
    if let Some(dir) = shard_key.rsplit('/').nth(1) {
        if let Some(id) = dir.strip_prefix("ckpt-") {
            let guessed = format!("{base}/manifests/ckpt-{id}.json");
            let absent = store.get(&guessed).await;
            assert!(
                absent.is_err(),
                "{label}: the guessed manifest must be absent"
            );
        }
    }

    // Original content: validates.
    let got = Reader::new(&store, base.to_string())
        .read_shard_against_manifest(shard_key)
        .await
        .map_err(|e| anyhow!("{label}: valid checkpoint failed integrity loading: {e:#}"))?;
    assert_eq!(&got[..], SHARD_DATA, "{label}");

    // Replacement content of the same length: rejected as an integrity failure, not as a
    // manifest that went missing.
    store
        .put(
            &format!("{base}/{shard_key}"),
            Bytes::from_static(REPLACED_DATA),
        )
        .await?;
    let err = Reader::new(&store, base.to_string())
        .read_shard_against_manifest(shard_key)
        .await
        .expect_err("replacement content must be rejected");
    let text = format!("{err:#}");
    assert!(
        text.contains("Integrity validation failed") || text.to_lowercase().contains("checksum"),
        "{label}: rejected for the wrong reason: {text}"
    );
    assert!(
        !text.contains("no complete checkpoint manifest"),
        "{label}: the manifest was lost, not compared: {text}"
    );
    Ok(())
}

#[tokio::test]
async fn listed_manifest_of_the_same_step_is_found_under_every_store_spelling() -> Result<()> {
    let mut failures = Vec::new();
    let n_bases = bases().len();
    for (label, base) in bases() {
        if let Err(e) = fallback_case(label, base, STEP_SHARD, "ckpt-5-manifesttime").await {
            failures.push(format!("{e:#}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} spellings failed:\n{}",
        failures.len(),
        n_bases,
        failures.join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn listed_manifest_of_a_caller_chosen_shard_key_is_found_under_every_store_spelling(
) -> Result<()> {
    let mut failures = Vec::new();
    let n_bases = bases().len();
    for (label, base) in bases() {
        if let Err(e) = fallback_case(label, base, CUSTOM_SHARD, "ckpt-2-x").await {
            failures.push(format!("{e:#}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} spellings failed:\n{}",
        failures.len(),
        n_bases,
        failures.join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn an_incomplete_listed_manifest_is_still_not_trusted_under_every_spelling() -> Result<()> {
    for (label, base) in bases() {
        let store = SpellingStore::default();
        store
            .put(
                &format!("{base}/{CUSTOM_SHARD}"),
                Bytes::from_static(SHARD_DATA),
            )
            .await?;
        put_manifest(
            &store,
            base,
            "ckpt-2-x",
            "incomplete",
            CUSTOM_SHARD,
            SHARD_DATA,
        )
        .await?;
        let err = Reader::new(&store, base.to_string())
            .read_shard_against_manifest(CUSTOM_SHARD)
            .await
            .expect_err("an incomplete manifest must not validate a shard");
        assert!(
            format!("{err:#}").contains("no complete checkpoint manifest"),
            "{label}: {err:#}"
        );
    }
    Ok(())
}

/// A fake that holds one complete checkpoint (step 2, manifest `ckpt-2-x`, shard `CUSTOM_SHARD`).
async fn store_with_checkpoint(base: &str) -> Result<SpellingStore> {
    let store = SpellingStore::default();
    store
        .put(
            &format!("{base}/{CUSTOM_SHARD}"),
            Bytes::from_static(SHARD_DATA),
        )
        .await?;
    put_manifest(
        &store,
        base,
        "ckpt-2-x",
        "complete",
        CUSTOM_SHARD,
        SHARD_DATA,
    )
    .await?;
    Ok(store)
}

fn push(list: &Mutex<Vec<String>>, uri: &str) -> Result<()> {
    list.lock()
        .map_err(|_| anyhow!("poisoned"))?
        .push(uri.to_string());
    Ok(())
}

/// The shard and the manifest are both still there.
async fn assert_nothing_deleted(label: &str, base: &str, store: &SpellingStore) -> Result<()> {
    let shard = store.get(&format!("{base}/{CUSTOM_SHARD}")).await;
    assert!(shard.is_ok(), "{label}: the shard was deleted");
    let manifest = store.get(&format!("{base}/manifests/ckpt-2-x.json")).await;
    assert!(manifest.is_ok(), "{label}: the manifest was deleted");
    Ok(())
}

fn assert_names_entry_without_secret(label: &str, what: &str, err: &anyhow::Error) {
    let text = format!("{err:#}");
    assert!(
        text.contains("ckpt-5-z.json"),
        "{label}: {what}: the error must name the entry: {text}"
    );
    assert!(
        !text.contains("SECRETPW"),
        "{label}: {what}: the credential leaked: {text}"
    );
}

/// A listed `.json` entry that is not under the base is an error in EVERY consumer of the
/// listing, wherever in the listing it sits, and `delete_checkpoint` deletes nothing then.
async fn assert_unplaceable_is_an_error(label: &str, base: &str, foreign: &str) -> Result<()> {
    // Listed after the real manifest.
    let store = store_with_checkpoint(base).await?;
    push(&store.extra_listed, foreign)?;
    let reader = Reader::new(&store, base.to_string());

    let err = reader
        .read_shard_against_manifest(CUSTOM_SHARD)
        .await
        .expect_err("a listed manifest that cannot be placed must be an error");
    assert_names_entry_without_secret(label, "read_shard_against_manifest", &err);
    // Step 99 does not exist, so the whole listing is walked.
    let err = reader
        .find_manifest_by_step(99)
        .await
        .expect_err("find must not skip an unplaceable entry");
    assert_names_entry_without_secret(label, "find_manifest_by_step", &err);
    let err = reader
        .list_checkpoints()
        .await
        .expect_err("list must not skip an unplaceable entry");
    assert_names_entry_without_secret(label, "list_checkpoints", &err);
    let err = reader
        .delete_checkpoint(2)
        .await
        .expect_err("delete must refuse when the listing holds an unplaceable entry");
    assert_names_entry_without_secret(label, "delete_checkpoint (listed after)", &err);
    assert_nothing_deleted(label, base, &store).await?;

    // Listed ahead of the real manifest.
    let store = store_with_checkpoint(base).await?;
    push(&store.extra_listed_first, foreign)?;
    let reader = Reader::new(&store, base.to_string());
    let err = reader
        .find_manifest_by_step(2)
        .await
        .expect_err("find must not skip an unplaceable entry");
    assert_names_entry_without_secret(label, "find_manifest_by_step (listed first)", &err);
    let err = reader
        .delete_checkpoint(2)
        .await
        .expect_err("delete must refuse when the listing holds an unplaceable entry");
    assert_names_entry_without_secret(label, "delete_checkpoint (listed first)", &err);
    assert_nothing_deleted(label, base, &store).await?;
    Ok(())
}

/// A listing that returns an entry which is not under the checkpoint base (another bucket,
/// container or account, a URL carrying credentials, a sibling prefix, or another server) is an
/// error that names the entry, never a silent loss.
#[tokio::test]
async fn an_unconvertible_listed_manifest_is_surfaced_with_a_redacted_uri() -> Result<()> {
    let cases: &[(&str, &str, &str)] = &[
        (
            "s3-endpoint",
            "s3://127.0.0.1:9000/bkt/run/ckpts",
            "s3://other-bucket/run/ckpts/manifests/ckpt-5-z.json",
        ),
        (
            "gcs",
            "gcs://bkt/run/ckpts",
            "gs://other-bucket/run/ckpts/manifests/ckpt-5-z.json",
        ),
        (
            "azure-https",
            "https://acct.blob.core.windows.net/ctr/run/ckpts",
            "az://otheracct/ctr/run/ckpts/manifests/ckpt-5-z.json",
        ),
        (
            "azure-https-userinfo",
            "https://acct.blob.core.windows.net/ctr/run/ckpts",
            "https://user:SECRETPW@acct.blob.core.windows.net/ctr/run/ckpts/manifests/ckpt-5-z.json",
        ),
    ];
    for (label, base, foreign) in cases {
        assert_unplaceable_is_an_error(label, base, foreign).await?;
    }
    Ok(())
}

/// The base `.../ckpt` must not claim `ckpt2/...`, `ckptmanifests/...` or `ckpt//...`: the base
/// is followed by exactly one `/`.
#[tokio::test]
async fn a_sibling_prefix_of_the_base_is_not_under_the_base() -> Result<()> {
    let bases: &[(&str, &str, &str)] = &[
        ("s3", "s3://bkt/ckpt", "s3://bkt"),
        ("s3-endpoint", "s3://127.0.0.1:9000/bkt/ckpt", "s3://bkt"),
        ("gcs", "gcs://bkt/ckpt", "gs://bkt"),
        (
            "azure-https",
            "https://acct.blob.core.windows.net/ctr/ckpt",
            "az://acct/ctr",
        ),
    ];
    for (label, base, listed_root) in bases {
        for sibling in [
            "ckpt2/manifests/ckpt-5-z.json",
            "ckptmanifests/ckpt-5-z.json",
            "ckpt//manifests/ckpt-5-z.json",
        ] {
            let foreign = format!("{listed_root}/{sibling}");
            assert_unplaceable_is_an_error(&format!("{label} {sibling}"), base, &foreign).await?;
        }
    }
    Ok(())
}

/// Two servers are two stores: `hostA:9000` does not hold what `hostB:9000` lists.
#[tokio::test]
async fn an_entry_from_another_s3_endpoint_is_not_under_the_base() -> Result<()> {
    assert_unplaceable_is_an_error(
        "endpoint",
        "s3://hosta:9000/bkt/run/ckpts",
        "s3://hostb:9000/bkt/run/ckpts/manifests/ckpt-5-z.json",
    )
    .await?;
    assert_unplaceable_is_an_error(
        "ipv6",
        "s3://[::1]:9000/bkt/run/ckpts",
        "s3://[::2]:9000/bkt/run/ckpts/manifests/ckpt-5-z.json",
    )
    .await
}

/// An endpoint-qualified entry listed for the SAME server (and a different case of its name)
/// is under the base, IPv6 included.
#[tokio::test]
async fn an_entry_listed_with_the_same_s3_endpoint_is_under_the_base() -> Result<()> {
    for (label, base, listed) in [
        (
            "ipv4",
            "s3://HostA:9000/bkt/run/ckpts",
            "s3://hosta:9000/bkt/run/ckpts/manifests/ckpt-2-x.json",
        ),
        (
            "ipv6",
            "s3://[::1]:9000/bkt/run/ckpts",
            "s3://[::1]:9000/bkt/run/ckpts/manifests/ckpt-2-x.json",
        ),
    ] {
        let store = store_with_checkpoint(base).await?;
        // The fake lists bucket form; this listing names the server as well.
        {
            let mut objects = store.lock()?;
            let key = objects
                .keys()
                .find(|k| k.ends_with("manifests/ckpt-2-x.json"))
                .cloned()
                .ok_or_else(|| anyhow!("{label}: no manifest"))?;
            if let Some((canonical, _)) = objects.get_mut(&key) {
                *canonical = listed.to_string();
            }
        }
        let reader = Reader::new(&store, base.to_string());
        let found = reader.find_manifest_by_step(2).await;
        assert!(
            matches!(found, Ok(Some(_))),
            "{label}: not found: {found:?}"
        );
        let listed = reader.list_checkpoints().await?;
        assert_eq!(listed.len(), 1, "{label}: {listed:?}");
    }
    Ok(())
}

/// `delete_checkpoint` ignores a listed entry that is not a manifest (it is filtered to `.json`
/// like the other consumers) and still deletes the checkpoint.
#[tokio::test]
async fn delete_ignores_an_unplaceable_entry_that_is_not_a_manifest() -> Result<()> {
    let base = "s3://127.0.0.1:9000/bkt/run/ckpts";
    let store = store_with_checkpoint(base).await?;
    // Listed ahead of the manifest: the old delete deleted the shards, then failed on it.
    push(
        &store.extra_listed_first,
        "s3://other-bucket/run/ckpts/manifests/notes.txt",
    )?;
    let reader = Reader::new(&store, base.to_string());
    assert!(reader.delete_checkpoint(2).await?);
    assert!(store
        .get(&format!("{base}/manifests/ckpt-2-x.json"))
        .await
        .is_err());
    assert!(store.get(&format!("{base}/{CUSTOM_SHARD}")).await.is_err());
    Ok(())
}

/// The other listing consumers of the reader share the defect and the fix.
#[tokio::test]
async fn find_list_and_delete_by_step_work_under_every_store_spelling() -> Result<()> {
    for (label, base) in bases() {
        let store = SpellingStore::default();
        store
            .put(
                &format!("{base}/{STEP_SHARD}"),
                Bytes::from_static(SHARD_DATA),
            )
            .await?;
        put_manifest(
            &store,
            base,
            "ckpt-5-manifesttime",
            "complete",
            STEP_SHARD,
            SHARD_DATA,
        )
        .await?;
        let reader = Reader::new(&store, base.to_string());

        let found = reader
            .find_manifest_by_step(5)
            .await?
            .unwrap_or_else(|| panic!("{label}: step 5 not found"));
        assert_eq!(found.global_step, 5, "{label}");

        let listed = reader.list_checkpoints().await?;
        assert_eq!(listed.len(), 1, "{label}: {listed:?}");
        assert_eq!(listed[0].0, 5, "{label}");
        assert_eq!(listed[0].1, "manifesttime", "{label}");
        assert_eq!(listed[0].2, "complete", "{label}");

        assert!(reader.delete_checkpoint(5).await?, "{label}");
        assert!(
            reader.find_manifest_by_step(5).await?.is_none(),
            "{label}: the manifest must be gone after delete_checkpoint"
        );
        assert!(
            store.get(&format!("{base}/{STEP_SHARD}")).await.is_err(),
            "{label}: the shard must be gone after delete_checkpoint"
        );
    }
    Ok(())
}

/// A relative `file://` base: `FileSystemObjectStore` lists its objects as scheme-less
/// relative paths (`rel/ck/manifests/x.json`), and the reader finds, lists and deletes them.
///
/// The test never changes the working directory: it spells a fresh temp directory as a path
/// relative to the current directory (`../../..` up to the root, then down), and skips, with a
/// printed reason, if that spelling does not name the same directory.
#[tokio::test]
async fn a_relative_file_base_finds_lists_and_deletes_scheme_less_entries() -> Result<()> {
    use std::path::{Component, Path, PathBuf};

    let dir = tempfile::TempDir::new()?;
    let abs = std::fs::canonicalize(dir.path())?;
    let cwd = std::env::current_dir()?;
    let ups = cwd
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .count();
    let tail: PathBuf = abs
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .collect();
    let mut relative = String::new();
    for _ in 0..ups {
        relative.push_str("../");
    }
    relative.push_str(&tail.to_string_lossy());
    let same = std::fs::canonicalize(Path::new(&relative)).is_ok_and(|p| p == abs);
    if !same || relative.starts_with('/') {
        eprintln!("SKIP: cannot spell {abs:?} relative to {cwd:?} (got {relative:?})");
        return Ok(());
    }

    // Write the checkpoint straight to the directory; the reader goes through the store.
    let ck = abs.join("ck");
    std::fs::create_dir_all(ck.join("manifests"))?;
    std::fs::create_dir_all(ck.join("shards/ckpt-5-t"))?;
    std::fs::write(ck.join("shards/ckpt-5-t/rank-0.bin"), SHARD_DATA)?;
    let mut manifest = Manifest::new("torch".to_string(), 5, 1, 1);
    manifest.wall_time = "2026-10-18T10:00:00Z".to_string();
    manifest.status = "complete".to_string();
    manifest.add_shard(ShardMeta {
        rank: 0,
        key: "shards/ckpt-5-t/rank-0.bin".to_string(),
        size: SHARD_DATA.len() as u64,
        etag: None,
        checksum: Some(checksum_of(SHARD_DATA)),
    });
    std::fs::write(
        ck.join("manifests/ckpt-5-t.json"),
        serde_json::to_vec(&manifest)?,
    )?;

    let base = format!("file://{relative}/ck");
    let store = s3dlio::object_store::store_for_uri(&base)?;
    // The premise: the store really lists scheme-less.
    let listed = store.list(&format!("{base}/manifests/"), false).await?;
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert!(
        !listed[0].starts_with("file://"),
        "the premise changed: the store now lists {:?}",
        listed[0]
    );

    let reader = Reader::new(store.as_ref(), base.clone());
    let found = reader.find_manifest_by_step(5).await?;
    assert!(found.is_some(), "step 5 not found");
    let infos = reader.list_checkpoints().await?;
    assert_eq!(infos.len(), 1, "{infos:?}");
    assert!(reader.delete_checkpoint(5).await?);
    assert!(!ck.join("manifests/ckpt-5-t.json").exists());
    assert!(!ck.join("shards/ckpt-5-t/rank-0.bin").exists());
    assert!(reader.find_manifest_by_step(5).await?.is_none());
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// delete_checkpoint: ordering and failure outcomes (fault injection), the latest.json pointer
// (#290) and shard-key validation (#291).
// ---------------------------------------------------------------------------------------------

/// A store that wraps the in-memory fake, records every SUCCESSFUL delete in order, and can be
/// told to fail a delete, a put or a get whose URI ends with a given suffix.
#[derive(Default)]
struct FaultStore {
    inner: SpellingStore,
    fail_delete_suffixes: Mutex<Vec<String>>,
    fail_put_suffixes: Mutex<Vec<String>>,
    fail_get_suffixes: Mutex<Vec<String>>,
    /// Gets that fail only once something has been deleted (a failure during the repoint).
    fail_get_after_delete_suffixes: Mutex<Vec<String>>,
    deleted: Mutex<Vec<String>>,
}

impl FaultStore {
    fn matches(list: &Mutex<Vec<String>>, uri: &str) -> Result<bool> {
        Ok(list
            .lock()
            .map_err(|_| anyhow!("poisoned"))?
            .iter()
            .any(|suffix| uri.ends_with(suffix.as_str())))
    }

    fn deleted(&self) -> Result<Vec<String>> {
        Ok(self
            .deleted
            .lock()
            .map_err(|_| anyhow!("poisoned"))?
            .clone())
    }
}

#[async_trait]
impl ObjectStore for FaultStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if Self::matches(&self.fail_get_suffixes, uri)? {
            bail!("injected get failure for {uri}");
        }
        if !self.deleted()?.is_empty() && Self::matches(&self.fail_get_after_delete_suffixes, uri)?
        {
            bail!("injected get failure after a delete for {uri}");
        }
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        self.inner.get_range(uri, offset, length).await
    }
    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        if Self::matches(&self.fail_put_suffixes, uri)? {
            bail!("injected put failure for {uri}");
        }
        self.inner.put(uri, data).await
    }
    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()> {
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
    async fn delete(&self, uri: &str) -> Result<()> {
        if Self::matches(&self.fail_delete_suffixes, uri)? {
            bail!("injected delete failure for {uri}");
        }
        self.inner.delete(uri).await?;
        self.deleted
            .lock()
            .map_err(|_| anyhow!("poisoned"))?
            .push(uri.to_string());
        Ok(())
    }
    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        for uri in uris {
            self.delete(uri).await?;
        }
        Ok(())
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

const FAULT_BASE: &str = "s3://bkt/run/ckpts";

/// Steps 1 and 2, each a complete one-shard checkpoint (`manifests/ckpt-<n>-t<n>.json`,
/// `shards/ckpt-<n>-t<n>/rank-0.bin`). `latest.json` names step 2, as the writer leaves it.
async fn two_step_store() -> Result<FaultStore> {
    let store = FaultStore::default();
    for step in [1u64, 2] {
        let shard = format!("shards/ckpt-{step}-t{step}/rank-0.bin");
        store
            .put(
                &format!("{FAULT_BASE}/{shard}"),
                Bytes::from_static(SHARD_DATA),
            )
            .await?;
        // wall_time orders the steps equally; the step decides.
        put_manifest_in(
            &store,
            &format!("ckpt-{step}-t{step}"),
            "complete",
            &shard,
            SHARD_DATA,
        )
        .await?;
    }
    put_latest(&store, 2).await?;
    Ok(store)
}

async fn put_manifest_in(
    store: &FaultStore,
    name: &str,
    status: &str,
    shard_key: &str,
    data: &[u8],
) -> Result<()> {
    put_manifest(&store.inner, FAULT_BASE, name, status, shard_key, data).await
}

async fn put_latest(store: &FaultStore, step: u64) -> Result<()> {
    let latest = Latest::new(
        format!("manifests/ckpt-{step}-t{step}.json"),
        step,
        format!("t{step}"),
    );
    store
        .put(
            &format!("{FAULT_BASE}/latest.json"),
            Bytes::from(serde_json::to_vec(&latest)?),
        )
        .await
}

async fn latest_of(store: &FaultStore) -> Result<Option<Latest>> {
    match store.get(&format!("{FAULT_BASE}/latest.json")).await {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(_) => Ok(None),
    }
}

async fn present(store: &FaultStore, key: &str) -> bool {
    store.get(&format!("{FAULT_BASE}/{key}")).await.is_ok()
}

const M1: &str = "manifests/ckpt-1-t1.json";
const M2: &str = "manifests/ckpt-2-t2.json";
const S1: &str = "shards/ckpt-1-t1/rank-0.bin";
const S2: &str = "shards/ckpt-2-t2/rank-0.bin";

fn fail(list: &Mutex<Vec<String>>, suffix: &str) -> Result<()> {
    push(list, suffix)
}

/// The manifest is deleted first. If its delete fails, NOTHING is deleted and the call fails.
/// (A shards-first delete leaves a complete manifest pointing at deleted shards here.)
#[tokio::test]
async fn a_failed_manifest_delete_deletes_nothing_and_errors() -> Result<()> {
    let store = two_step_store().await?;
    fail(&store.fail_delete_suffixes, "manifests/ckpt-2-t2.json")?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());

    let err = reader
        .delete_checkpoint(2)
        .await
        .expect_err("a manifest that cannot be deleted must fail the call");
    assert!(
        format!("{err:#}").contains("injected delete failure"),
        "{err:#}"
    );
    assert!(present(&store, M2).await, "the manifest must still exist");
    assert!(present(&store, S2).await, "the shard must still exist");
    assert!(store.deleted()?.is_empty(), "{:?}", store.deleted()?);
    // The pointer is untouched too: it still names the (existing) manifest.
    assert_eq!(
        latest_of(&store).await?.map(|l| l.manifest_key).as_deref(),
        Some(M2)
    );
    // Step 1 is not involved.
    assert!(present(&store, M1).await && present(&store, S1).await);
    Ok(())
}

/// A shard that cannot be deleted does not fail the call: the manifest is already gone, the
/// call returns `Ok(true)`, and the undeleted shard is unreferenced garbage a rerun cannot find.
#[tokio::test]
async fn a_failed_shard_delete_leaves_the_manifest_gone_and_returns_true() -> Result<()> {
    let store = two_step_store().await?;
    fail(&store.fail_delete_suffixes, S2)?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());

    assert!(reader.delete_checkpoint(2).await?);
    assert!(
        !present(&store, M2).await,
        "the manifest must already be gone"
    );
    assert!(
        present(&store, S2).await,
        "the shard whose delete failed stays"
    );
    assert!(reader.find_manifest_by_step(2).await?.is_none());
    // A rerun cannot find the garbage shard: the step is simply not there any more.
    assert!(!reader.delete_checkpoint(2).await?);
    // The checkpoint of step 1 is intact.
    assert!(present(&store, M1).await && present(&store, S1).await);
    Ok(())
}

/// The order of successful deletes: the manifest, then its shards.
#[tokio::test]
async fn the_manifest_is_deleted_before_its_shards() -> Result<()> {
    let store = two_step_store().await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(2).await?);
    let deleted = store.deleted()?;
    // latest.json is repointed (a put), never deleted, while step 1 remains.
    assert_eq!(
        deleted,
        vec![format!("{FAULT_BASE}/{M2}"), format!("{FAULT_BASE}/{S2}")]
    );
    Ok(())
}

/// #290: deleting the checkpoint `latest.json` names points `latest.json` at the newest
/// remaining complete checkpoint, and `load_latest_manifest` returns it.
#[tokio::test]
async fn deleting_the_latest_checkpoint_repoints_latest_json() -> Result<()> {
    let store = two_step_store().await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(2).await?);

    let latest = latest_of(&store).await?.expect("latest.json must remain");
    assert_eq!(latest.manifest_key, M1);
    assert_eq!(latest.global_step, 1);
    assert_eq!(latest.ts, "t1");
    let loaded = reader.load_latest_manifest().await?.expect("step 1");
    assert_eq!(loaded.global_step, 1);
    Ok(())
}

/// #290: deleting the only checkpoint removes `latest.json`; `load_latest_manifest` is `None`.
#[tokio::test]
async fn deleting_the_last_checkpoint_removes_latest_json() -> Result<()> {
    let store = two_step_store().await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(1).await?); // not the one latest names
    assert_eq!(
        latest_of(&store).await?.map(|l| l.manifest_key).as_deref(),
        Some(M2),
        "deleting another checkpoint must leave the pointer alone"
    );
    assert!(reader.delete_checkpoint(2).await?);
    assert!(latest_of(&store).await?.is_none(), "latest.json must go");
    assert!(reader.load_latest_manifest().await?.is_none());
    Ok(())
}

/// #290: an incomplete newer manifest is not chosen as the new latest.
#[tokio::test]
async fn the_new_latest_is_the_newest_complete_checkpoint() -> Result<()> {
    let store = two_step_store().await?;
    put_manifest_in(
        &store,
        "ckpt-3-t3",
        "in_progress",
        "shards/ckpt-3-t3/rank-0.bin",
        b"x",
    )
    .await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(2).await?);
    let latest = latest_of(&store).await?.expect("latest.json");
    assert_eq!(latest.manifest_key, M1);
    Ok(())
}

/// #290: a failure to update the pointer is reported, after the checkpoint WAS deleted, and the
/// shards are still deleted; `load_latest_manifest` still works through its fallback.
#[tokio::test]
async fn a_failed_pointer_update_is_reported_without_undoing_the_delete() -> Result<()> {
    let store = two_step_store().await?;
    fail(&store.fail_put_suffixes, "latest.json")?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());

    let err = reader
        .delete_checkpoint(2)
        .await
        .expect_err("a pointer that cannot be updated must be reported");
    assert!(format!("{err:#}").contains("was deleted"), "{err:#}");
    assert!(!present(&store, M2).await);
    assert!(!present(&store, S2).await, "the shards are still deleted");
    // The pointer dangles, and the reader copes with it.
    let loaded = reader.load_latest_manifest().await?.expect("step 1");
    assert_eq!(loaded.global_step, 1);
    Ok(())
}

/// #290, the fallback on its own: a dangling pointer (manifest deleted behind its back) makes
/// `load_latest_manifest` scan instead of failing.
#[tokio::test]
async fn load_latest_falls_back_when_the_pointer_names_a_missing_manifest() -> Result<()> {
    let store = two_step_store().await?;
    store.delete(&format!("{FAULT_BASE}/{M2}")).await?; // latest.json still names it
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    let loaded = reader.load_latest_manifest().await?.expect("step 1");
    assert_eq!(loaded.global_step, 1);

    // Nothing left at all: no checkpoint, not an error.
    store.delete(&format!("{FAULT_BASE}/{M1}")).await?;
    assert!(reader.load_latest_manifest().await?.is_none());
    Ok(())
}

/// The fallback is for a MISSING manifest only: a manifest that exists but cannot be read or
/// parsed is still an error.
#[tokio::test]
async fn load_latest_does_not_hide_a_manifest_that_exists_but_fails() -> Result<()> {
    let store = two_step_store().await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());

    fail(&store.fail_get_suffixes, M2)?;
    let err = reader
        .load_latest_manifest()
        .await
        .expect_err("a transient failure on an existing manifest must surface");
    assert!(
        format!("{err:#}").contains("injected get failure"),
        "{err:#}"
    );
    store
        .fail_get_suffixes
        .lock()
        .map_err(|_| anyhow!("poisoned"))?
        .clear();

    store
        .put(
            &format!("{FAULT_BASE}/{M2}"),
            Bytes::from_static(b"{ not json"),
        )
        .await?;
    assert!(reader.load_latest_manifest().await.is_err());
    Ok(())
}

/// #290 through the public store on a real `file://` base (whose not-found is untyped): save
/// steps 1 and 2, delete step 2, `load_latest` returns step 1; and with the pointer left
/// dangling (manifest removed behind the store's back) it still does.
#[tokio::test]
async fn load_latest_returns_the_older_checkpoint_after_a_delete_on_a_file_base() -> Result<()> {
    for dangling in [false, true] {
        let dir = tempfile::TempDir::new()?;
        let base = format!("file://{}", dir.path().display());
        let store = CheckpointStore::open(&base)?;
        let key1 = store.save(1, 1, "torch", b"step one", None).await?;
        let key2 = store.save(2, 2, "torch", b"step two", None).await?;
        assert_eq!(
            &store.load_latest().await?.expect("step 2")[..],
            b"step two"
        );

        if dangling {
            std::fs::remove_file(dir.path().join(&key2))?;
        } else {
            assert!(store.delete_checkpoint(2).await?);
        }
        let loaded = store.load_latest().await?;
        assert_eq!(
            loaded.as_deref(),
            Some(&b"step one"[..]),
            "dangling pointer: {dangling}"
        );
        assert!(dir.path().join(&key1).exists());
        if !dangling {
            let latest = store.reader().get_latest().await?.expect("latest.json");
            assert_eq!(latest.manifest_key, key1);
        }
    }
    Ok(())
}

/// #291: a manifest whose shard key is not a safe shard key makes delete fail, and nothing is
/// deleted: not the manifest, not its other shards, not the object the key names.
#[tokio::test]
async fn an_unsafe_manifest_shard_key_refuses_the_delete() -> Result<()> {
    let bad_keys = [
        // Another checkpoint's manifest, and the control objects.
        M1,
        "manifests/anything.json",
        "latest.json",
        "latest.json.tmp",
        "latest/ckpt-1-t1.json",
        // Control components anywhere in the key, a directory named `shards`, and a shard
        // directory of another step at any position.
        "custom/manifests/ckpt-1-t1.json",
        "a/latest/x",
        "a/latest.json",
        "a/latest.json.tmp",
        "a/latest",
        "a/shards",
        "custom/ckpt-1-t1/rank-0.bin",
        "custom/ckpt-2-t2/rank-0.bin",
        // URL metacharacters and percent-encoded dots (an HTTP-based store may normalise or
        // truncate them into a different object).
        "shards/x?y",
        "shards/x#y",
        "shards/%2e%2e/ckpt-1-t1/rank-0.bin",
        "shards/%2E%2e/x",
        "shards/a%2ebin",
        // Not relative keys.
        "",
        "/etc/passwd",
        "../outside",
        "shards/../../outside",
        "shards/./x",
        "shards//x",
        "shards/x/",
        "shards\\x",
        ".",
        "..",
    ];
    for bad in bad_keys {
        let store = two_step_store().await?;
        // Step 3 lists a real shard and then the bad key.
        store
            .put(
                &format!("{FAULT_BASE}/shards/ckpt-3-t3/rank-0.bin"),
                Bytes::from_static(b"s"),
            )
            .await?;
        let mut manifest = Manifest::new("torch".to_string(), 3, 1, 2);
        manifest.wall_time = "2026-10-18T10:00:00Z".to_string();
        manifest.status = "complete".to_string();
        for (rank, key) in [(0, "shards/ckpt-3-t3/rank-0.bin"), (1, bad)] {
            manifest.add_shard(ShardMeta {
                rank,
                key: key.to_string(),
                size: 1,
                etag: None,
                checksum: None,
            });
        }
        store
            .put(
                &format!("{FAULT_BASE}/manifests/ckpt-3-t3.json"),
                Bytes::from(serde_json::to_vec(&manifest)?),
            )
            .await?;
        let before = store.inner.lock()?.len();

        let reader = Reader::new(&store, FAULT_BASE.to_string());
        let err = reader
            .delete_checkpoint(3)
            .await
            .expect_err(&format!("shard key {bad:?} must be refused"));
        assert!(
            format!("{err:#}").contains("not a safe shard key"),
            "{bad:?}: {err:#}"
        );
        assert!(
            store.deleted()?.is_empty(),
            "{bad:?}: {:?}",
            store.deleted()?
        );
        assert_eq!(store.inner.lock()?.len(), before, "{bad:?}");
        assert!(present(&store, "manifests/ckpt-3-t3.json").await, "{bad:?}");
        assert!(
            present(&store, M1).await && present(&store, M2).await,
            "{bad:?}"
        );
    }
    Ok(())
}

/// Caller-chosen keys outside `shards/` (the writer's `put_shard_with_key`) are still deletable.
#[tokio::test]
async fn a_caller_chosen_shard_key_is_still_deletable() -> Result<()> {
    for key in [
        "custom/one.bin",
        "shard-0",
        "shards/custom/weights.bin",
        "a.b/c..d/e",
    ] {
        let store = FaultStore::default();
        store
            .put(
                &format!("{FAULT_BASE}/{key}"),
                Bytes::from_static(SHARD_DATA),
            )
            .await?;
        put_manifest_in(&store, "ckpt-4-t4", "complete", key, SHARD_DATA).await?;
        let reader = Reader::new(&store, FAULT_BASE.to_string());
        assert!(reader.delete_checkpoint(4).await?, "{key}");
        assert!(!present(&store, key).await, "{key}");
    }
    Ok(())
}

/// #291 on a real `file://` base: a `../x` shard key would delete a file outside the checkpoint
/// directory. Refused; neither the file nor the manifest is touched.
#[tokio::test]
async fn a_dotdot_shard_key_on_a_file_base_deletes_nothing() -> Result<()> {
    let dir = tempfile::TempDir::new()?;
    let ck = dir.path().join("ck");
    std::fs::create_dir_all(ck.join("manifests"))?;
    let outside = dir.path().join("x");
    std::fs::write(&outside, b"not a shard")?;
    let mut manifest = Manifest::new("torch".to_string(), 7, 1, 1);
    manifest.wall_time = "2026-10-18T10:00:00Z".to_string();
    manifest.status = "complete".to_string();
    manifest.add_shard(ShardMeta {
        rank: 0,
        key: "../x".to_string(),
        size: 1,
        etag: None,
        checksum: None,
    });
    let manifest_path = ck.join("manifests/ckpt-7-t.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;

    let base = format!("file://{}", ck.display());
    let store = s3dlio::object_store::store_for_uri(&base)?;
    let reader = Reader::new(store.as_ref(), base);
    let err = reader
        .delete_checkpoint(7)
        .await
        .expect_err("a `..` shard key must be refused");
    assert!(
        format!("{err:#}").contains("not a safe shard key"),
        "{err:#}"
    );
    assert!(
        outside.exists(),
        "the file outside the checkpoint was deleted"
    );
    assert!(manifest_path.exists(), "the manifest was deleted");
    Ok(())
}

/// Two manifests of one step: one `delete_checkpoint` call deletes exactly one (the first
/// listed); the other stays until the next call.
#[tokio::test]
async fn delete_checkpoint_removes_one_manifest_of_a_step_per_call() -> Result<()> {
    let store = FaultStore::default();
    for name in ["ckpt-5-a", "ckpt-5-b"] {
        let shard = format!("shards/{name}/rank-0.bin");
        store
            .put(
                &format!("{FAULT_BASE}/{shard}"),
                Bytes::from_static(SHARD_DATA),
            )
            .await?;
        put_manifest_in(&store, name, "complete", &shard, SHARD_DATA).await?;
    }
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    let remaining = |store: &FaultStore| -> Result<usize> {
        Ok(store
            .inner
            .lock()?
            .keys()
            .filter(|k| k.contains("/manifests/"))
            .count())
    };
    assert_eq!(remaining(&store)?, 2);
    assert!(reader.delete_checkpoint(5).await?);
    assert_eq!(remaining(&store)?, 1);
    assert!(reader.delete_checkpoint(5).await?);
    assert_eq!(remaining(&store)?, 0);
    assert!(!reader.delete_checkpoint(5).await?);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Review round 2 of delete_checkpoint / latest.json: directory-valued and cross-checkpoint shard
// keys (#291), a repoint that must never write a pointer it is not sure of (#290), valid-only
// choice of the new latest, and a corrupt latest.json.
// ---------------------------------------------------------------------------------------------

/// Put `manifests/<name>.json` for `step` listing `keys` as ranks 0.. (`world_size` ranks are
/// declared; a `world_size` above `keys.len()` makes a complete manifest that fails `validate`).
async fn put_manifest_keys(
    store: &FaultStore,
    name: &str,
    status: &str,
    world_size: u32,
    keys: &[&str],
) -> Result<()> {
    let step: u64 = name
        .strip_prefix("ckpt-")
        .and_then(|rest| rest.split('-').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("bad manifest name {name}"))?;
    let mut manifest = Manifest::new("torch".to_string(), step, 1, world_size);
    manifest.wall_time = "2026-10-18T10:00:00Z".to_string();
    manifest.status = status.to_string();
    for (rank, key) in keys.iter().enumerate() {
        manifest.add_shard(ShardMeta {
            rank: rank as u32,
            key: key.to_string(),
            size: 1,
            etag: None,
            checksum: None,
        });
    }
    store
        .put(
            &format!("{FAULT_BASE}/manifests/{name}.json"),
            Bytes::from(serde_json::to_vec(&manifest)?),
        )
        .await
}

/// A key that names a directory (`shards`), or a shard of ANOTHER checkpoint, in the manifest of
/// step 3 makes delete fail with nothing deleted; both other checkpoints stay whole.
#[tokio::test]
async fn directory_and_cross_checkpoint_shard_keys_are_refused() -> Result<()> {
    for bad in [
        "shards",
        "shards/",
        S2,
        S1,
        "shards/ckpt-2-t2",
        "shards/rr/00/ckpt-2-t2/rank-0.bin",
        "shards/bin/0000/ckpt-1-t1/rank-0.bin",
    ] {
        let store = two_step_store().await?;
        store
            .put(
                &format!("{FAULT_BASE}/shards/ckpt-3-t3/rank-0.bin"),
                Bytes::from_static(b"s"),
            )
            .await?;
        put_manifest_keys(
            &store,
            "ckpt-3-t3",
            "complete",
            2,
            &["shards/ckpt-3-t3/rank-0.bin", bad],
        )
        .await?;
        let reader = Reader::new(&store, FAULT_BASE.to_string());
        let err = reader
            .delete_checkpoint(3)
            .await
            .expect_err(&format!("shard key {bad:?} must be refused"));
        assert!(
            format!("{err:#}").contains("not a safe shard key"),
            "{bad:?}: {err:#}"
        );
        assert!(
            store.deleted()?.is_empty(),
            "{bad:?}: {:?}",
            store.deleted()?
        );
        for key in [M1, M2, S1, S2, "manifests/ckpt-3-t3.json"] {
            assert!(present(&store, key).await, "{bad:?}: {key} was deleted");
        }
        assert!(
            present(&store, "shards/ckpt-3-t3/rank-0.bin").await,
            "{bad:?}: the checkpoint's own shard was deleted"
        );
    }
    Ok(())
}

/// A custom key listed by another complete manifest is that checkpoint's shard too: the delete
/// succeeds, removes the manifest and only the shards no other complete manifest lists, and KEEPS
/// the shared shard, so the other checkpoint still loads (and both stay deletable).
#[tokio::test]
async fn a_shard_shared_with_another_complete_manifest_is_kept() -> Result<()> {
    let store = FaultStore::default();
    for key in ["custom/shared.bin", "custom/own.bin"] {
        store
            .put(&format!("{FAULT_BASE}/{key}"), Bytes::from_static(b"s"))
            .await?;
    }
    put_manifest_keys(
        &store,
        "ckpt-4-t4",
        "complete",
        2,
        &["custom/own.bin", "custom/shared.bin"],
    )
    .await?;
    put_manifest_keys(&store, "ckpt-5-t5", "complete", 1, &["custom/shared.bin"]).await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(4).await?);
    assert!(!present(&store, "manifests/ckpt-4-t4.json").await);
    assert!(!present(&store, "custom/own.bin").await);
    assert!(
        present(&store, "custom/shared.bin").await,
        "a shard another complete manifest lists must be kept"
    );
    assert!(
        !store.deleted()?.iter().any(|u| u.ends_with("shared.bin")),
        "{:?}",
        store.deleted()?
    );
    // The other checkpoint still loads whole.
    let other = reader.load_manifest("manifests/ckpt-5-t5.json").await?;
    assert_eq!(other.shards.len(), 1);
    assert_eq!(&reader.read_shard("custom/shared.bin").await?[..], b"s");

    // Once it is the last to list the key, deleting it removes the shard.
    assert!(reader.delete_checkpoint(5).await?);
    assert!(!present(&store, "custom/shared.bin").await);
    Ok(())
}

/// Another manifest that cannot be READ (not a not-found) leaves the delete refused with
/// nothing deleted: it could list a shard of this checkpoint.
#[tokio::test]
async fn an_unreadable_other_manifest_refuses_the_delete() -> Result<()> {
    let store = FaultStore::default();
    store
        .put(
            &format!("{FAULT_BASE}/custom/own.bin"),
            Bytes::from_static(b"s"),
        )
        .await?;
    put_manifest_keys(&store, "ckpt-4-t4", "complete", 1, &["custom/own.bin"]).await?;
    put_manifest_keys(&store, "ckpt-5-t5", "complete", 1, &["custom/other.bin"]).await?;
    fail(&store.fail_get_suffixes, "manifests/ckpt-5-t5.json")?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    let err = reader
        .delete_checkpoint(4)
        .await
        .expect_err("an unreadable other manifest must refuse the delete");
    assert!(format!("{err:#}").contains("was not deleted"), "{err:#}");
    assert!(store.deleted()?.is_empty(), "{:?}", store.deleted()?);
    assert!(present(&store, "manifests/ckpt-4-t4.json").await);
    assert!(present(&store, "custom/own.bin").await);
    Ok(())
}

/// Put `manifests/<key_stem>.json` as a complete one-rank manifest whose `global_step` is
/// `global_step` (which need not match the name).
async fn put_decoy_manifest(
    store: &FaultStore,
    key_stem: &str,
    global_step: u64,
    wall_time: &str,
) -> Result<()> {
    let mut manifest = Manifest::new("torch".to_string(), global_step, 1, 1);
    manifest.wall_time = wall_time.to_string();
    manifest.status = "complete".to_string();
    manifest.add_shard(ShardMeta {
        rank: 0,
        key: format!("shards/decoy-{key_stem}/rank-0.bin"),
        size: 1,
        etag: None,
        checksum: None,
    });
    store
        .put(
            &format!("{FAULT_BASE}/manifests/{key_stem}.json"),
            Bytes::from(serde_json::to_vec(&manifest)?),
        )
        .await
}

/// A candidate for the new `latest.json` (and for the `load_latest` fallback scan) must be named
/// `ckpt-<step>-<ts>.json` with `<step>` equal to its `global_step`; others are skipped.
#[tokio::test]
async fn repoint_and_fallback_skip_manifests_whose_name_does_not_match_their_step() -> Result<()> {
    let decoys = [
        // Newer by step and time, but the name says step 7 and global_step is 9...
        ("ckpt-7-t7", 9u64),
        // ...or the name is not a checkpoint manifest name at all.
        ("notes", 8u64),
        ("ckpt-x-y", 8u64),
    ];

    // The repoint.
    let store = two_step_store().await?;
    for (stem, gs) in decoys {
        put_decoy_manifest(&store, stem, gs, "2030-01-01T00:00:00Z").await?;
    }
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(2).await?);
    let latest = latest_of(&store).await?.expect("latest.json");
    assert_eq!(latest.manifest_key, M1, "{latest:?}");

    // The fallback scan with no pointer.
    let store = two_step_store().await?;
    for (stem, gs) in decoys {
        put_decoy_manifest(&store, stem, gs, "2030-01-01T00:00:00Z").await?;
    }
    store.delete(&format!("{FAULT_BASE}/latest.json")).await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert_eq!(
        reader.scan_latest_complete().await?.map(|m| m.global_step),
        Some(2)
    );
    assert_eq!(
        reader.load_latest_manifest().await?.map(|m| m.global_step),
        Some(2)
    );
    Ok(())
}

/// A distributed checkpoint: each rank stamps its own shard directory, so a shard of the SAME
/// step in a directory with another timestamp is the checkpoint's own and is deleted.
#[tokio::test]
async fn shards_of_the_same_step_in_other_stamped_directories_are_deletable() -> Result<()> {
    let store = FaultStore::default();
    let keys = ["shards/ckpt-6-ta/rank-0.bin", "shards/ckpt-6-tb/rank-1.bin"];
    for key in keys {
        store
            .put(&format!("{FAULT_BASE}/{key}"), Bytes::from_static(b"s"))
            .await?;
    }
    put_manifest_keys(&store, "ckpt-6-ta", "complete", 2, &keys).await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(6).await?);
    for key in keys {
        assert!(!present(&store, key).await, "{key}");
    }
    Ok(())
}

/// A real `file://` base: `shards` (which would `remove_dir_all` every shard directory) and a
/// key into another checkpoint's shard directory are refused, and a key naming the checkpoint's
/// OWN shard directory is refused by a stat of the target; nothing is removed.
#[tokio::test]
async fn directory_valued_shard_keys_on_a_file_base_delete_nothing() -> Result<()> {
    for bad in [
        "shards",
        "shards/ckpt-1-x/rank-0.bin", // shard of ANOTHER checkpoint (made below)
        "shards/ckpt-3-own",          // its own directory, not a file
        "shards/ckpt-3-own/sub",      // a directory below it
    ] {
        let dir = tempfile::TempDir::new()?;
        let base = format!("file://{}", dir.path().display());
        let store = CheckpointStore::open(&base)?;
        let key1 = store.save(1, 1, "torch", b"one", None).await?;
        let key2 = store.save(2, 2, "torch", b"two", None).await?;
        let own_dir = dir.path().join("shards/ckpt-3-own");
        std::fs::create_dir_all(own_dir.join("sub"))?;
        std::fs::write(own_dir.join("rank-0.bin"), b"s")?;
        // A shard of checkpoint 1 under the name the bad key uses.
        let shard1 = dir.path().join("shards/ckpt-1-x");
        std::fs::create_dir_all(&shard1)?;
        std::fs::write(shard1.join("rank-0.bin"), b"theirs")?;

        let mut manifest = Manifest::new("torch".to_string(), 3, 1, 2);
        manifest.wall_time = "2026-10-18T10:00:00Z".to_string();
        manifest.status = "complete".to_string();
        for (rank, key) in [(0u32, "shards/ckpt-3-own/rank-0.bin"), (1, bad)] {
            manifest.add_shard(ShardMeta {
                rank,
                key: key.to_string(),
                size: 1,
                etag: None,
                checksum: None,
            });
        }
        let manifest_path = dir.path().join("manifests/ckpt-3-own.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;

        let fs_store = s3dlio::object_store::store_for_uri(&base)?;
        let reader = Reader::new(fs_store.as_ref(), base.clone());
        let err = reader
            .delete_checkpoint(3)
            .await
            .expect_err(&format!("{bad:?} must be refused"));
        assert!(
            format!("{err:#}").contains("not a safe shard key"),
            "{bad:?}: {err:#}"
        );
        assert!(manifest_path.exists(), "{bad:?}: manifest deleted");
        assert!(own_dir.join("rank-0.bin").exists(), "{bad:?}: own shard");
        assert!(own_dir.join("sub").is_dir(), "{bad:?}: own subdirectory");
        assert!(shard1.join("rank-0.bin").exists(), "{bad:?}: other shard");
        for key in [&key1, &key2] {
            assert!(dir.path().join(key).exists(), "{bad:?}: {key}");
        }
        assert_eq!(
            store.load_latest().await?.as_deref(),
            Some(&b"two"[..]),
            "{bad:?}"
        );
    }
    Ok(())
}

/// A newer manifest that cannot be read or parsed during the repoint scan might be the right
/// new latest: `latest.json` is left untouched and the error says so. The delete itself happened.
#[tokio::test]
async fn an_unreadable_manifest_during_the_repoint_leaves_latest_json_untouched() -> Result<()> {
    for how in ["get-fails", "corrupt"] {
        let store = two_step_store().await?;
        // Step 3 is newer than step 1 and sits behind the one `latest.json` names (step 2)...
        put_manifest_in(
            &store,
            "ckpt-3-t3",
            "complete",
            "shards/ckpt-3-t3/rank-0.bin",
            b"x",
        )
        .await?;
        if how == "get-fails" {
            fail(
                &store.fail_get_after_delete_suffixes,
                "manifests/ckpt-3-t3.json",
            )?;
        } else {
            store
                .put(
                    &format!("{FAULT_BASE}/manifests/ckpt-3-t3.json"),
                    Bytes::from_static(b"{ not json"),
                )
                .await?;
        }
        let reader = Reader::new(&store, FAULT_BASE.to_string());
        let err = reader
            .delete_checkpoint(2)
            .await
            .expect_err(&format!("{how}: an unverifiable repoint must be reported"));
        let text = format!("{err:#}");
        assert!(text.contains("was deleted"), "{how}: {text}");
        assert!(text.contains("latest.json"), "{how}: {text}");
        assert!(!present(&store, M2).await, "{how}: the delete happened");
        assert!(!present(&store, S2).await, "{how}");
        assert_eq!(
            latest_of(&store).await?.map(|l| l.manifest_key).as_deref(),
            Some(M2),
            "{how}: latest.json must be untouched"
        );
        assert!(!store.deleted()?.iter().any(|u| u.ends_with("latest.json")));
    }
    Ok(())
}

/// A complete manifest that fails `validate` (a rank missing) is not chosen as the new latest,
/// and the `load_latest` fallback scan does not return it either.
#[tokio::test]
async fn an_invalid_complete_manifest_is_never_chosen() -> Result<()> {
    let store = two_step_store().await?;
    // Newest by step, "complete", but declares 2 ranks and lists one.
    put_manifest_keys(
        &store,
        "ckpt-3-t3",
        "complete",
        2,
        &["shards/ckpt-3-t3/rank-0.bin"],
    )
    .await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());

    // The fallback scan (pointer dangling after the manifest it names is removed).
    store.delete(&format!("{FAULT_BASE}/{M2}")).await?;
    let loaded = reader.load_latest_manifest().await?.expect("step 1");
    assert_eq!(loaded.global_step, 1);
    assert_eq!(
        reader.scan_latest_complete().await?.map(|m| m.global_step),
        Some(1)
    );

    // The repoint: latest names step 2 again; deleting it points at step 1, not step 3.
    let store = two_step_store().await?;
    put_manifest_keys(
        &store,
        "ckpt-3-t3",
        "complete",
        2,
        &["shards/ckpt-3-t3/rank-0.bin"],
    )
    .await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    assert!(reader.delete_checkpoint(2).await?);
    let latest = latest_of(&store).await?.expect("latest.json");
    assert_eq!(latest.manifest_key, M1);
    Ok(())
}

/// A corrupt `latest.json` counts as absent for `load_latest` (it falls back to the scan) and
/// as a dangling pointer for the repoint (rewritten, or removed when nothing remains).
#[tokio::test]
async fn a_corrupt_latest_json_is_treated_as_dangling() -> Result<()> {
    let store = two_step_store().await?;
    let latest_uri = format!("{FAULT_BASE}/latest.json");
    store
        .put(&latest_uri, Bytes::from_static(b"{ not json"))
        .await?;
    let reader = Reader::new(&store, FAULT_BASE.to_string());
    let loaded = reader.load_latest_manifest().await?.expect("newest");
    assert_eq!(loaded.global_step, 2, "the fallback scan picks the newest");

    // Deleting ANY checkpoint rewrites the corrupt pointer at the newest remaining one.
    assert!(reader.delete_checkpoint(2).await?);
    assert_eq!(
        latest_of(&store).await?.map(|l| l.manifest_key).as_deref(),
        Some(M1)
    );

    // Nothing remains: a corrupt pointer is removed.
    store
        .put(&latest_uri, Bytes::from_static(b"\xff\xfe"))
        .await?;
    assert!(reader.delete_checkpoint(1).await?);
    assert!(!present(&store, "latest.json").await);
    assert!(reader.load_latest_manifest().await?.is_none());
    Ok(())
}

/// The same on a real `file://` base through the public store.
#[tokio::test]
async fn a_corrupt_latest_json_on_a_file_base_falls_back_and_is_rewritten() -> Result<()> {
    let dir = tempfile::TempDir::new()?;
    let base = format!("file://{}", dir.path().display());
    let store = CheckpointStore::open(&base)?;
    let key1 = store.save(1, 1, "torch", b"one", None).await?;
    store.save(2, 2, "torch", b"two", None).await?;
    std::fs::write(dir.path().join("latest.json"), b"{ not json")?;
    assert_eq!(store.load_latest().await?.as_deref(), Some(&b"two"[..]));

    assert!(store.delete_checkpoint(2).await?);
    let latest = store.reader().get_latest().await?.expect("rewritten");
    assert_eq!(latest.manifest_key, key1);
    assert_eq!(store.load_latest().await?.as_deref(), Some(&b"one"[..]));
    Ok(())
}
