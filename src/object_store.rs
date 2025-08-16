// src/object_store.rs
//
// Pluggable object-store abstraction.
// Step 1: Ship with an S3 adapter that delegates to existing s3_utils.rs.
// Step 2: We'll add an Azure adapter behind a feature flag.

use anyhow::{bail, Result};
use async_trait::async_trait;

use crate::s3_utils::{
    // Reuse your existing types & fns
    ObjectStat as S3ObjectStat,
    parse_s3_uri,
    list_objects as s3_list_objects,
    get_object_uri_async as s3_get_object_uri_async,
    get_object_range_uri_async as s3_get_object_range_uri_async,
    stat_object_uri_async as s3_stat_object_uri_async,
    delete_objects as s3_delete_objects,
    create_bucket as s3_create_bucket,
    delete_bucket as s3_delete_bucket,
};

/// Provider-neutral object metadata. For now this aliases S3's metadata.
/// In Step 2, Azure will populate the same fields.
pub type ObjectMetadata = S3ObjectStat;

/// A minimal scheme enum so we can route URIs.
/// (We’ll fill AZURE in Step 2.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    S3,
    Azure,
    Unknown,
}

/// Best-effort scheme inference from a URI.
pub fn infer_scheme(uri: &str) -> Scheme {
    if uri.starts_with("s3://") { Scheme::S3 }
    else if uri.starts_with("az://") || uri.contains(".blob.core.windows.net/") { Scheme::Azure }
    else { Scheme::Unknown }
}

#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get entire object into memory.
    async fn get(&self, uri: &str) -> Result<Vec<u8>>;

    /// Get a byte-range. If `length` is None, read from `offset` to end.
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>>;

    /// List objects under a prefix. Returns full URIs.
    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>>;

    /// Stat a single object (HEAD-like).
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata>;

    /// Delete all objects under a prefix.
    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()>;

    /// Create a top-level container (S3: bucket; Azure: container).
    async fn create_container(&self, name: &str) -> Result<()>;

    /// Delete a top-level container.
    async fn delete_container(&self, name: &str) -> Result<()>;
}

/// S3 adapter that calls straight into your existing helpers.
///
/// Notes:
/// - We keep the trait **URI-based** so datasets/consumers don’t need to split
///   bucket/key; the adapter does the parse.
/// - list/delete_prefix expect a *prefix* URI like `s3://bucket/path/`.
pub struct S3ObjectStore;

impl S3ObjectStore {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl ObjectStore for S3ObjectStore {
    async fn get(&self, uri: &str) -> Result<Vec<u8>> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_get_object_uri_async(uri).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_get_object_range_uri_async(uri, offset, length).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (bucket, key_prefix) = parse_s3_uri(uri_prefix)?;
        let keys = s3_list_objects(&bucket, &key_prefix, recursive)?;
        // normalize to full URIs
        Ok(keys.into_iter().map(|k| format!("s3://{}/{}", bucket, k)).collect())
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_stat_object_uri_async(uri).await
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (bucket, mut key_prefix) = parse_s3_uri(uri_prefix)?;
        // Normalize: ensure trailing "/" so list_objects treats it as a pure prefix
        if !key_prefix.is_empty() && !key_prefix.ends_with('/') {
            key_prefix.push('/');
        }
        // Gather all keys under the prefix (recursive delete semantics)
        let keys = s3_list_objects(&bucket, &key_prefix, true)?;
        if keys.is_empty() {
            return Ok(());
        }
        s3_delete_objects(&bucket, &keys)
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        s3_create_bucket(name)
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        s3_delete_bucket(name)
    }
}

/// Convenience factory that picks a backend from a URI.
/// Step 1 returns S3 only; Step 2 will return Azure when enabled.
pub fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match infer_scheme(uri) {
        Scheme::S3    => Ok(Box::new(S3ObjectStore::new())),
        Scheme::Azure => bail!("Azure not implemented yet — enable in Step 2"),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    }
}

