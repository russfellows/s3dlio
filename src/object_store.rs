// src/object_store.rs
//
// Pluggable object-store abstraction with consistent URI schemes.
// Supported schemes: s3://, az://, file://
// Phase 1: S3 adapter (s3://) - delegates to existing s3_utils.rs
// Phase 2: File adapter (file://) - POSIX filesystem operations  
// Phase 3: Azure adapter (az://) - Azure Blob Storage operations

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
/// For now, we only support S3; Azure will be added later.
/// File is a placeholder for local file system access, if needed.
/// This enum helps us infer the backend from a URI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    S3,
    Azure,
    File,
    Unknown,
}

/// Best-effort scheme inference from a URI.
pub fn infer_scheme(uri: &str) -> Scheme {
    if uri.starts_with("s3://") { Scheme::S3 }
    else if uri.starts_with("az://") || uri.contains(".blob.core.windows.net/") { Scheme::Azure }
    else if uri.starts_with("file://") { Scheme::File }
    else { Scheme::Unknown }
}

/// ObjectStore trait for pluggable storage backends.
/// 
/// This trait defines the methods needed for an object store.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get entire object into memory.
    async fn get(&self, uri: &str) -> Result<Vec<u8>>;

    /// Get a byte-range. If `length` is None, read from `offset` to end.
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>>;

    /// Put object data to storage.
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()>;

    /// Put object data with multipart upload for large objects.
    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()>;

    /// List objects under a prefix. Returns full URIs.
    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>>;

    /// Stat a single object (HEAD-like).
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata>;

    /// Delete a single object.
    async fn delete(&self, uri: &str) -> Result<()>;

    /// Delete all objects under a prefix.
    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()>;

    /// Create a top-level container (S3: bucket; Azure: container; File: directory).
    async fn create_container(&self, name: &str) -> Result<()>;

    /// Delete a top-level container.
    async fn delete_container(&self, name: &str) -> Result<()>;

    /// Copy object from one location to another (can be cross-storage).
    async fn copy(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        let data = self.get(src_uri).await?;
        self.put(dst_uri, &data).await
    }

    /// Check if an object exists.
    async fn exists(&self, uri: &str) -> Result<bool> {
        match self.stat(uri).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
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

    async fn put(&self, uri: &str, _data: &[u8]) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        // TODO: Implement put using existing s3_utils functions
        // This requires adding put_object_uri_async to s3_utils.rs
        bail!("S3 put not implemented yet")
    }

    async fn put_multipart(&self, uri: &str, _data: &[u8], _part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        // TODO: Implement multipart put using existing multipart functions
        bail!("S3 multipart put not implemented yet")
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        let (bucket, key) = parse_s3_uri(uri)?;
        s3_delete_objects(&bucket, &vec![key])
    }
    
    async fn create_container(&self, name: &str) -> Result<()> {
        s3_create_bucket(name)
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        s3_delete_bucket(name)
    }
}

/// Convenience factory that picks a backend from a URI.
/// Supports consistent URI schemes: s3://, az://, file://
pub fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match infer_scheme(uri) {
        Scheme::S3    => Ok(Box::new(S3ObjectStore::new())),
        Scheme::Azure => bail!("Azure not implemented yet — enable in Phase 3.3"),
        Scheme::File  => Ok(Box::new(crate::file_store::FileSystemObjectStore::new())),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}. Supported schemes: s3://, az://, file://"),
    }
}

