// src/object_store.rs
//
// Pluggable object-store abstraction.
// Backends: FileSystem, S3, and Azure Blob.
// Features: Checksums, Compression, and Integrity Validation

use anyhow::anyhow;
use anyhow::{bail, Result};
use async_trait::async_trait;
use crc32fast::Hasher;

// Helper function for integrity validation
fn compute_checksum(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    format!("{:08x}", hasher.finalize())
}

// --- S3 ----------------------------------------------------------------------
use crate::s3_utils::{
    // Reuse existing S3 helpers
    ObjectStat as S3ObjectStat,
    parse_s3_uri,
    list_objects as s3_list_objects,
    get_object_uri_async as s3_get_object_uri_async,
    get_object_range_uri_async as s3_get_object_range_uri_async,
    stat_object_uri_async as s3_stat_object_uri_async,
    delete_objects as s3_delete_objects,
    create_bucket as s3_create_bucket,
    delete_bucket as s3_delete_bucket,
    // NEW: PUT operations via ObjectStore
    put_object_uri_async as s3_put_object_uri_async,
    put_object_multipart_uri_async as s3_put_object_multipart_uri_async,
};

// Expose FS adapter (already implemented in src/file_store.rs)
use crate::file_store::FileSystemObjectStore;

// Expose enhanced FS adapter with O_DIRECT support
use crate::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};

// --- Azure ---------------------------------------------------
use bytes::Bytes;
use futures::stream;
use crate::azure_client::{AzureBlob, AzureBlobProperties};

/// Provider-neutral object metadata. For now this aliases S3's metadata.
pub type ObjectMetadata = S3ObjectStat;

/// Compression configuration for ObjectWriter implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionConfig {
    None,
    Zstd { level: i32 },
}

impl Default for CompressionConfig {
    fn default() -> Self {
        CompressionConfig::None
    }
}

impl CompressionConfig {
    /// Create Zstd compression with default level (3)
    pub fn zstd_default() -> Self {
        CompressionConfig::Zstd { level: 3 }
    }
    
    /// Create Zstd compression with custom level (1-22)
    pub fn zstd_level(level: i32) -> Self {
        CompressionConfig::Zstd { level: level.clamp(1, 22) }
    }
    
    /// Get the file extension for this compression format
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionConfig::None => "",
            CompressionConfig::Zstd { .. } => ".zst",
        }
    }
    
    /// Check if compression is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, CompressionConfig::None)
    }
}

/// Supported schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    File,
    Direct,
    S3,
    Azure,
    Unknown,
}

/// Best-effort scheme inference from a URI.
pub fn infer_scheme(uri: &str) -> Scheme {
    if uri.starts_with("file://") { Scheme::File }
    else if uri.starts_with("direct://") { Scheme::Direct }
    else if uri.starts_with("s3://") { Scheme::S3 }
    else if uri.starts_with("az://") || uri.contains(".blob.core.windows.net/") { Scheme::Azure }
    else { Scheme::Unknown }
}

#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get entire object into memory.
    async fn get(&self, uri: &str) -> Result<Vec<u8>>;

    /// Get a byte-range. If `length` is None, read from `offset` to end.
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>>;

    /// Put full object (single-shot).
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()>;

    /// Put via multipart semantics (or an equivalent high-throughput path).
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

    /// Check existence via `stat()`. Backends can override for efficiency.
    async fn exists(&self, uri: &str) -> Result<bool> {
        Ok(self.stat(uri).await.is_ok())
    }
    
    /// Get object with integrity validation.
    /// Returns the data and validates it against the expected checksum if provided.
    async fn get_with_validation(&self, uri: &str, expected_checksum: Option<&str>) -> Result<Vec<u8>> {
        let data = self.get(uri).await?;
        
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Integrity validation failed for {}: expected {}, got {}", uri, expected, actual);
            }
        }
        
        Ok(data)
    }
    
    /// Get byte range with integrity validation.
    /// For partial reads, validates against the partial data checksum if provided.
    async fn get_range_with_validation(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        expected_checksum: Option<&str>
    ) -> Result<Vec<u8>> {
        let data = self.get_range(uri, offset, length).await?;
        
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Integrity validation failed for range {}[{}:{}]: expected {}, got {}", 
                      uri, offset, offset + data.len() as u64, expected, actual);
            }
        }
        
        Ok(data)
    }
    
    /// Load and validate checkpoint data with integrity checking.
    /// This method is specifically designed for checkpoint loading scenarios.
    async fn load_checkpoint_with_validation(
        &self, 
        checkpoint_uri: &str, 
        expected_checksum: Option<&str>
    ) -> Result<Vec<u8>> {
        // Get checkpoint data
        let data = self.get(checkpoint_uri).await?;
        
        // Validate integrity if checksum provided
        if let Some(expected) = expected_checksum {
            let actual = compute_checksum(&data);
            if actual != expected {
                bail!("Checkpoint integrity validation failed for {}: expected checksum {}, got {}", 
                      checkpoint_uri, expected, actual);
            }
        }
        
        Ok(data)
    }

    /// Default copy reads then writes. Backends can override with server-side copy.
    async fn copy(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        let data = self.get(src_uri).await?;
        self.put(dst_uri, &data).await
    }

    /// Atomic rename/move operation. For file:// backends, this uses filesystem rename
    /// which is atomic. For cloud backends, this typically falls back to copy+delete.
    /// Returns an error if the operation cannot be completed atomically.
    async fn rename(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        // Default implementation: copy then delete (not atomic, but works)
        self.copy(src_uri, dst_uri).await?;
        self.delete(src_uri).await
    }

    /// Get a streaming writer for zero-copy operations.
    /// This enables writing large objects without buffering the entire content in memory.
    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>>;

}

/// Streaming writer interface for zero-copy object uploads
#[async_trait]
pub trait ObjectWriter: Send + Sync {
    /// Write a chunk of data to the object stream.
    /// Chunks are written in order and the implementation handles buffering/uploading.
    /// If compression is enabled, data is compressed before storage.
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()>;
    
    /// Finalize the object upload. Must be called to complete the write.
    /// After calling finalize(), the writer should not be used again.
    async fn finalize(self: Box<Self>) -> Result<()>;
    
    /// Get the total number of uncompressed bytes written so far.
    fn bytes_written(&self) -> u64;
    
    /// Get the total number of compressed bytes (if compression is enabled).
    /// Returns the same as bytes_written() if compression is disabled.
    fn compressed_bytes(&self) -> u64 {
        self.bytes_written() // Default: no compression
    }
    
    /// Get the computed checksum for the uncompressed data written so far.
    /// Returns None if no checksum has been computed yet.
    fn checksum(&self) -> Option<String>;
    
    /// Get compression configuration for this writer.
    fn compression(&self) -> CompressionConfig {
        CompressionConfig::None // Default: no compression
    }
    
    /// Get compression ratio (compressed_size / uncompressed_size).
    /// Returns 1.0 if compression is disabled.
    fn compression_ratio(&self) -> f64 {
        if self.bytes_written() == 0 {
            1.0
        } else {
            self.compressed_bytes() as f64 / self.bytes_written() as f64
        }
    }
    
    /// Cancel the upload and clean up any partial data.
    /// This is called automatically if the writer is dropped without finalize().
    async fn cancel(self: Box<Self>) -> Result<()> {
        // Default implementation: just drop
        Ok(())
    }
}

/// Default buffered implementation that collects chunks and calls put()
pub struct BufferedObjectWriter<'a> {
    uri: String,
    store: &'a dyn ObjectStore,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
}

impl<'a> BufferedObjectWriter<'a> {
    pub fn new(uri: String, store: &'a dyn ObjectStore) -> Self {
        Self {
            uri,
            store,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
        }
    }
}

#[async_trait]
impl<'a> ObjectWriter for BufferedObjectWriter<'a> {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        self.buffer.extend_from_slice(chunk);
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        self.store.put(&self.uri, &self.buffer).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        Ok(())
    }
}

// ============================================================================
// FileSystem adapter (already implemented)
// ============================================================================
impl FileSystemObjectStore {
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }
}

// ============================================================================
// S3 adapter that calls straight into your existing helpers
// ============================================================================
pub struct S3ObjectStore;

impl S3ObjectStore {
    pub fn new() -> Self { Self }
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self) }
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

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_put_object_uri_async(uri, data).await
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_put_object_multipart_uri_async(uri, data, part_size).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (bucket, key_prefix) = parse_s3_uri(uri_prefix)?;
        let keys = s3_list_objects(&bucket, &key_prefix, recursive)?;
        Ok(keys.into_iter().map(|k| format!("s3://{}/{}", bucket, k)).collect())
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        s3_stat_object_uri_async(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("s3://") { bail!("S3ObjectStore expected s3:// URI"); }
        let (bucket, key) = parse_s3_uri(uri)?;
        s3_delete_objects(&bucket, &vec![key])
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (bucket, mut key_prefix) = parse_s3_uri(uri_prefix)?;
        if !key_prefix.is_empty() && !key_prefix.ends_with('/') { key_prefix.push('/'); }
        let keys = s3_list_objects(&bucket, &key_prefix, true)?;
        if keys.is_empty() { return Ok(()); }
        s3_delete_objects(&bucket, &keys)
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        s3_create_bucket(name)
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        s3_delete_bucket(name)
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For S3, use buffered writer that collects chunks then uses put_multipart
        Ok(Box::new(S3BufferedWriter::new(uri.to_string())))
    }
}

/// Buffered writer for S3 that collects chunks and uses multipart upload
pub struct S3BufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
}

impl S3BufferedWriter {
    pub fn new(uri: String) -> Self {
        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
        }
    }
}

#[async_trait]
impl ObjectWriter for S3BufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        self.buffer.extend_from_slice(chunk);
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        
        // Use S3 multipart upload for the buffered data
        s3_put_object_multipart_uri_async(&self.uri, &self.buffer, None).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        Ok(())
    }
}

// ============================================================================
// Azure adapter
// ============================================================================
fn parse_azure_uri(uri: &str) -> Result<(String, String, String)> {
    // Supports:
    // - az://{account}/{container}/{key...}
    // - https://{account}.blob.core.windows.net/{container}/{key...}
    if let Some(rest) = uri.strip_prefix("az://") {
        let mut it = rest.splitn(3, '/');
        let account = it.next().ok_or_else(|| anyhow!("missing account in az:// URI"))?;
        let container = it.next().ok_or_else(|| anyhow!("missing container in az:// URI"))?;
        let key = it.next().unwrap_or("").to_string();
        return Ok((account.to_string(), container.to_string(), key));
    }

    //if let Some(host_i) = uri.find(".blob.core.windows.net/") {
    if uri.contains(".blob.core.windows.net/") {
        // crude parse: "https://{account}.blob.core.windows.net/{container}/{key...}"
        // find "https://" then account up to first '.'
        let after_scheme = uri.strip_prefix("https://").ok_or_else(|| anyhow!("expected https:// for Azure URL"))?;
        let mut host_and_path = after_scheme.splitn(2, '/');
        let host = host_and_path.next().unwrap_or("");
        let path = host_and_path.next().unwrap_or("");
        let account = host.split('.').next().ok_or_else(|| anyhow!("bad Azure host"))?;
        let mut segs = path.split('/').filter(|s| !s.is_empty());
        let container = segs.next().ok_or_else(|| anyhow!("missing container in URL path"))?;
        let key = segs.collect::<Vec<_>>().join("/");
        return Ok((account.to_string(), container.to_string(), key));
    }

    bail!("not a recognized Azure URI: {}", uri)
}

fn az_uri(account: &str, container: &str, key: &str) -> String {
    if key.is_empty() {
        format!("az://{}/{}", account, container)
    } else {
        format!("az://{}/{}/{}", account, container, key)
    }
}

fn az_props_to_meta(p: &AzureBlobProperties) -> ObjectMetadata {
    ObjectMetadata {
        size: p.content_length,
        last_modified: p.last_modified.clone(),
        e_tag: p.etag.clone(),
        content_type: None,
        content_language: None,
        content_encoding: None,
        cache_control: None,
        content_disposition: None,
        expires: None,
        storage_class: None,
        server_side_encryption: None,
        ssekms_key_id: None,
        sse_customer_algorithm: None,
        version_id: None,
        replication_status: None,
        metadata: Default::default(),
    }
}


pub struct AzureObjectStore;


impl AzureObjectStore {
    pub fn new() -> Self { Self }
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self) }

    fn client_for_uri(uri: &str) -> Result<(AzureBlob, String, String, String)> {
        let (account, container, key) = parse_azure_uri(uri)?;
        // If caller provided an https:// URL, we still build via account name to ensure
        // we normalize the list() return URIs as az://...
        let cli = AzureBlob::with_default_credential(&account, &container)?;
        Ok((cli, account, container, key))
    }

    fn client_for_prefix(uri_prefix: &str) -> Result<(AzureBlob, String, String, String)> {
        Self::client_for_uri(uri_prefix)
    }
}


#[async_trait]
impl ObjectStore for AzureObjectStore {
    async fn get(&self, uri: &str) -> Result<Vec<u8>> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let b = cli.get(&key).await?; // Bytes
        Ok(b.to_vec())
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let end = length.map(|len| offset + len - 1);
        let b = cli.get_range(&key, offset, end).await?;
        Ok(b.to_vec())
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        cli.put(&key, Bytes::from(data.to_vec()), true).await
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let part = part_size.unwrap_or(crate::constants::DEFAULT_AZURE_MULTIPART_PART_SIZE);
        let max_in_flight = std::env::var("AZURE_MAX_INFLIGHT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(crate::constants::DEFAULT_CONCURRENT_UPLOADS);

        // Stream the provided buffer as Bytes chunks of size `part`
        let chunks = data
            .chunks(part)
            .map(|c| Bytes::copy_from_slice(c))
            .collect::<Vec<_>>();
        let stream = stream::iter(chunks);

        cli.upload_multipart_stream(&key, stream, part, max_in_flight).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (cli, account, container, key_prefix) = Self::client_for_prefix(uri_prefix)?;
        // Azure's flat list is already recursive (prefix-constrained).
        let prefix = if recursive {
            Some(key_prefix.as_str())
        } else {
            // emulate "shallow" by trimming after next '/':
            // We’ll still ask Azure for full prefix, then post-filter.
            Some(key_prefix.as_str())
        };

        let mut keys = cli.list(prefix).await?;
        if !recursive && !key_prefix.is_empty() {
            let base = if key_prefix.ends_with('/') { key_prefix.clone() } else { format!("{}/", key_prefix) };
            keys.retain(|k| {
                if let Some(rest) = k.strip_prefix(&base) {
                    !rest.contains('/')
                } else {
                    // if it doesn't start with base, keep only if it's exactly the same path
                    k == &key_prefix
                }
            });
        }

        Ok(keys.into_iter().map(|k| az_uri(&account, &container, &k)).collect())
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let p = cli.stat(&key).await?;
        Ok(az_props_to_meta(&p))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        cli.delete_objects(&[key]).await.map_err(Into::into)
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (cli, _acct, _cont, key_prefix) = Self::client_for_prefix(uri_prefix)?;
        let keys = cli.list(Some(&key_prefix)).await?;
        if keys.is_empty() { return Ok(()); }
        cli.delete_objects(&keys).await.map_err(Into::into)
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        // interpret "name" as "{account}/{container}"
        let mut it = name.splitn(2, '/');
        let account = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let container = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let cli = AzureBlob::with_default_credential(account, container)?;
        // best-effort create
        let _ = cli.create_container_if_missing().await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let mut it = name.splitn(2, '/');
        let account = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let container = it.next().ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let cli = AzureBlob::with_default_credential(account, container)?;
        let _ = cli.delete_container().await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer that collects chunks then uses put_multipart
        Ok(Box::new(AzureBufferedWriter::new(uri.to_string())))
    }
}

/// Buffered writer for Azure that collects chunks and uses multipart upload

pub struct AzureBufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Hasher,
}


impl AzureBufferedWriter {
    pub fn new(uri: String) -> Self {
        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Hasher::new(),
        }
    }
}


#[async_trait]
impl ObjectWriter for AzureBufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        self.buffer.extend_from_slice(chunk);
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;
        Ok(())
    }
    
    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;
        
        // Use Azure multipart upload for the buffered data
        let (cli, _acct, _cont, key) = AzureObjectStore::client_for_uri(&self.uri)?;
        cli.put(&key, Bytes::from(self.buffer.clone()), true).await
    }
    
    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }
    
    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        Ok(())
    }
}

// ============================================================================
// Convenience factory that picks a backend from a URI
// ============================================================================
pub fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match infer_scheme(uri) {
        Scheme::File  => Ok(FileSystemObjectStore::boxed()),
        Scheme::Direct => Ok(ConfigurableFileSystemObjectStore::boxed_direct_io()),
        Scheme::S3    => Ok(S3ObjectStore::boxed()),
        Scheme::Azure => Ok(AzureObjectStore::boxed()),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    }
}

/// Enhanced factory that supports configuration options for file I/O
pub fn store_for_uri_with_config(uri: &str, file_config: Option<FileSystemConfig>) -> Result<Box<dyn ObjectStore>> {
    match infer_scheme(uri) {
        Scheme::File => {
            if let Some(config) = file_config {
                Ok(ConfigurableFileSystemObjectStore::boxed(config))
            } else {
                Ok(FileSystemObjectStore::boxed())
            }
        }
        Scheme::Direct => {
            if let Some(config) = file_config {
                Ok(ConfigurableFileSystemObjectStore::boxed(config))
            } else {
                Ok(ConfigurableFileSystemObjectStore::boxed_direct_io())
            }
        }
        Scheme::S3 => Ok(S3ObjectStore::boxed()),
        Scheme::Azure => Ok(AzureObjectStore::boxed()),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    }
}

/// Factory for creating file stores with O_DIRECT enabled for AI/ML workloads
pub fn direct_io_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match infer_scheme(uri) {
        Scheme::File => Ok(ConfigurableFileSystemObjectStore::boxed_direct_io()),
        Scheme::Direct => Ok(ConfigurableFileSystemObjectStore::boxed_direct_io()),
        Scheme::S3 => Ok(S3ObjectStore::boxed()),
        Scheme::Azure => Ok(AzureObjectStore::boxed()),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    }
}

/// Factory for creating high-performance stores optimized for AI/ML workloads
pub fn high_performance_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match infer_scheme(uri) {
        Scheme::File => Ok(ConfigurableFileSystemObjectStore::boxed_high_performance()),
        Scheme::Direct => Ok(ConfigurableFileSystemObjectStore::boxed_direct_io()),
        Scheme::S3 => Ok(S3ObjectStore::boxed()),
        Scheme::Azure => Ok(AzureObjectStore::boxed()),
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    }
}

