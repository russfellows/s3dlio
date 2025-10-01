// src/object_store_arrow.rs
//
// Arrow-backed adapter for s3dlio's ObjectStore trait.
// Routes I/O through the Apache Arrow `object_store` crate.
//
// Build: gated behind feature "arrow-backend" (EXPERIMENTAL - NOT production ready).
//
// NOTE: This backend is experimental and has NOT shown performance benefits over
// native-backends in production testing. Kept for comparison testing only.
//
// This module is intentionally "stateless": each call parses the full URI and
// constructs (or reuses) the appropriate backend via object_store::parse_url.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use crc32fast::Hasher;
// Note: Futures stream traits not needed for simple implementation
use std::collections::HashMap;
use std::sync::Arc;
use url::Url;

use crate::object_store::{
    ObjectStore as S3DlioObjectStore,
    ObjectWriter as S3DlioObjectWriter,
    ObjectMetadata,             // alias to s3_utils::ObjectStat
    WriterOptions,
    CompressionConfig,
};

use object_store as aos;
use aos::{
    path::Path,
    ObjectStore as AosObjectStore,
    GetOptions,
    GetRange,
    PutPayload,
    aws::AmazonS3Builder,
};

// ----------------------------------------------------------------------------
// Adapter object store
// ----------------------------------------------------------------------------

/// Arrow-backed implementation.
/// Stateless: we re-parse the URL each call to honor full-URI semantics in s3dlio.
pub struct ArrowObjectStore;

impl ArrowObjectStore {
    pub fn new() -> Self { Self }
    
    #[inline]
    pub fn boxed() -> Box<dyn S3DlioObjectStore> { Box::new(Self) }

    /// parse a full URI into (store, path) using explicit S3 configuration.
    fn parse(uri: &str) -> Result<(Arc<dyn AosObjectStore>, Path, Url)> {
        let url = Url::parse(uri).context("invalid URI")?;
        
        match url.scheme() {
            "s3" => {
                // Extract bucket and path from URL
                let bucket = url.host_str().ok_or_else(|| anyhow::anyhow!("No bucket in S3 URL"))?;
                let path = Path::from(url.path().trim_start_matches('/'));
                
                // Build S3 client explicitly to control credentials
                let mut builder = AmazonS3Builder::new()
                    .with_bucket_name(bucket);
                
                // Use credentials from environment
                if let (Ok(access_key), Ok(secret_key)) = (
                    std::env::var("AWS_ACCESS_KEY_ID"), 
                    std::env::var("AWS_SECRET_ACCESS_KEY")
                ) {
                    builder = builder.with_access_key_id(&access_key)
                                   .with_secret_access_key(&secret_key);
                } else {
                    bail!("AWS credentials not found in environment");
                }
                
                // Set region
                if let Ok(region) = std::env::var("AWS_DEFAULT_REGION").or_else(|_| std::env::var("AWS_REGION")) {
                    builder = builder.with_region(&region);
                }
                
                // Set custom endpoint if provided
                if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
                    builder = builder.with_endpoint(&endpoint);
                    // For custom endpoints, we often need allow_http
                    if endpoint.starts_with("http://") {
                        builder = builder.with_allow_http(true);
                    }
                }
                
                let store = builder.build().context("Failed to build S3 client")?;
                Ok((Arc::new(store), path, url))
            },
            _ => {
                // For non-S3 URLs, fall back to generic parse_url
                let (store, path) = aos::parse_url(&url).context("object_store::parse_url failed")?;
                Ok((Arc::from(store), path, url))
            }
        }
    }

    /// convenience for prefix listing: if the URI is a "directory-like" prefix,
    /// just parse it; listing handles both object prefixes and virtual directories.
    fn parse_prefix(prefix: &str) -> Result<(Arc<dyn AosObjectStore>, Path, Url)> {
        Self::parse(prefix)
    }

    /// Build a full URI string for a returned `location` using the original scheme/authority.
    /// Example: scheme+host from input URL + '/' + returned path.
    fn rebuild_full_uri(base_url: &Url, location: &Path) -> String {
        // Preserve scheme+authority (+ bucket/account in host/path, as provided by parse_url config).
        // Fall back to base for everything before path; append location's string.
        let mut rebuilt = String::new();
        if let Some(scheme) = Some(base_url.scheme()) {
            rebuilt.push_str(scheme);
            rebuilt.push_str("://");
        }
        // authority (host[:port]) if any
        if let Some(host) = base_url.host_str() {
            rebuilt.push_str(host);
        }
        // carry through base_url.path "bucket" portion if present (e.g., https endpoints)
        // Note: parse_url usually embeds bucket/container into store config,
        // but for https-style endpoints keep the path prefix from the URL up to the "prefix" we parsed.
        // Keep it simple: append base path minus any trailing slash.
        let base_path = base_url.path().trim_end_matches('/').to_string();
        // Avoid duplicate slashes on pure virtual schemes like s3://bucket
        if !base_path.is_empty() && !base_path.starts_with('/') {
            rebuilt.push('/');
        }
        if !base_path.is_empty() {
            rebuilt.push_str(&base_path);
        }
        if !rebuilt.ends_with('/') { rebuilt.push('/'); }

        // Append Arrow location (never starts with '/')
        rebuilt.push_str(&location.to_string());
        rebuilt
    }

    /// Map Arrow ObjectMeta → s3dlio ObjectMetadata (S3ObjectStat shape).
    fn to_object_metadata(meta: &aos::ObjectMeta) -> ObjectMetadata {
        // Arrow ObjectMeta doesn't have a metadata field, so we use empty metadata
        let user_meta: HashMap<String, String> = HashMap::new();
        ObjectMetadata {
            size: meta.size,
            last_modified: Some(meta.last_modified.to_rfc3339()),
            e_tag: meta.e_tag.clone(),
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
            version_id: meta.version.clone(),
            replication_status: None,
            metadata: user_meta,
        }
    }
}

// ----------------------------------------------------------------------------
// ObjectStore impl
// ----------------------------------------------------------------------------

#[async_trait]
impl S3DlioObjectStore for ArrowObjectStore {
    async fn get(&self, uri: &str) -> Result<Vec<u8>> {
        let (store, path, _url) = Self::parse(uri)?;
        let reader = store.get(&path).await?;
        let bytes = reader.bytes().await?;
        Ok(bytes.to_vec())
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
        let (store, path, _url) = Self::parse(uri)?;
        let range = if let Some(len) = length {
            GetRange::Bounded(offset..(offset + len))
        } else {
            GetRange::Offset(offset)
        };
        let opts = GetOptions {
            range: Some(range),
            ..Default::default()
        };
        let reader = store.get_opts(&path, opts).await?;
        let bytes = reader.bytes().await?;
        Ok(bytes.to_vec())
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        let (store, path, _url) = Self::parse(uri)?;
        store.put(&path, PutPayload::from(Bytes::copy_from_slice(data))).await?;
        Ok(())
    }

    // For now route multipart to Arrow's regular `put` (Arrow chooses optimal strategy internally).
    // You can switch to `WriteMultipart` later if you want strict part size / concurrency control.
    async fn put_multipart(&self, uri: &str, data: &[u8], _part_size: Option<usize>) -> Result<()> {
        self.put(uri, data).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (store, base, url) = Self::parse_prefix(uri_prefix)?;
        let mut out = Vec::new();

        if recursive {
            // Simple implementation: collect all objects manually
            let result = store.list_with_delimiter(Some(&base)).await?;
            for obj in result.objects {
                out.push(Self::rebuild_full_uri(&url, &obj.location));
            }
            // For recursive, we'd need to recursively list common_prefixes too
            // For now, keep it simple and just list direct objects
        } else {
            let result = store.list_with_delimiter(Some(&base)).await?;
            for obj in result.objects {
                out.push(Self::rebuild_full_uri(&url, &obj.location));
            }
            for prefix in result.common_prefixes {
                out.push(Self::rebuild_full_uri(&url, &prefix));
            }
        }
        Ok(out)
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (store, path, _url) = Self::parse(uri)?;
        let meta = store.head(&path).await?;
        Ok(Self::to_object_metadata(&meta))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let (store, path, _url) = Self::parse(uri)?;
        store.delete(&path).await?;
        Ok(())
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (store, base, _url) = Self::parse_prefix(uri_prefix)?;
        
        // Simple implementation: list objects and delete them one by one
        let result = store.list_with_delimiter(Some(&base)).await?;
        for obj in result.objects {
            store.delete(&obj.location).await?;
        }
        
        Ok(())
    }

    async fn create_container(&self, _name: &str) -> Result<()> {
        // Not supported by Arrow's ObjectStore trait
        bail!("create_container is not supported by the Arrow object_store backend");
    }

    async fn delete_container(&self, _name: &str) -> Result<()> {
        // Not supported by Arrow's ObjectStore trait
        bail!("delete_container is not supported by the Arrow object_store backend");
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn S3DlioObjectWriter>> {
        // No compression
        ArrowWriter::new(uri, CompressionConfig::None).await
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: CompressionConfig) -> Result<Box<dyn S3DlioObjectWriter>> {
        ArrowWriter::new(uri, compression).await
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn S3DlioObjectWriter>> {
        let compression = options.compression.unwrap_or(CompressionConfig::None);
        ArrowWriter::new(uri, compression).await
    }

    // For now, fall back to default optimized paths (S3-specific concurrency doesn't apply here).
    // If you want, you can later implement `get_ranges` coalescing for Arrow to speed large reads.
    async fn get_optimized(&self, uri: &str) -> Result<Vec<u8>> {
        self.get(uri).await
    }

    async fn get_range_optimized(
        &self, 
        uri: &str, 
        offset: u64, 
        length: Option<u64>,
        _chunk_size: Option<usize>,
        _max_concurrency: Option<usize>
    ) -> Result<Vec<u8>> {
        self.get_range(uri, offset, length).await
    }
}

// ----------------------------------------------------------------------------
// Streaming writer
// ----------------------------------------------------------------------------

/// Arrow-backed streaming writer.
/// Since Arrow's BufWriter doesn't implement Sync (required by ObjectWriter), 
/// we implement a simple buffered writer that collects all data and uses put() at finalize.
/// - Maintains CRC32C over the **uncompressed** stream (parity with s3dlio writers).
/// - Optional Zstd compression (matches your FileSystemWriter behavior).
/// - Finalizes by putting all buffered data to the store.
struct ArrowWriter {
    // destination store and path
    store: Arc<dyn AosObjectStore>,
    path: Path,

    // buffer for collecting data
    buffer: Vec<u8>,

    // metrics & integrity
    hasher: Hasher,
    bytes_written: u64,
    compressed_bytes: u64,

    // compression
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,

    // finalized flag
    finalized: bool,
}

impl ArrowWriter {
    async fn new(dest_uri: &str, compression: CompressionConfig) -> Result<Box<dyn S3DlioObjectWriter>> {
        let (store, path, _url) = ArrowObjectStore::parse(dest_uri)?;
        
        // Adjust path for compression extension
        let final_path = if compression.is_enabled() {
            Path::from(format!("{}{}", path.to_string(), compression.extension()))
        } else {
            path
        };

        // Set up compressor if needed
        let compressor = match compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                let mut enc = zstd::Encoder::new(Vec::new(), level)?;
                enc.include_checksum(false)?; // we compute our own CRC
                Some(enc)
            }
        };

        Ok(Box::new(Self {
            store,
            path: final_path,
            buffer: Vec::new(),
            hasher: Hasher::new(),
            bytes_written: 0,
            compressed_bytes: 0,
            compression,
            compressor,
            finalized: false,
        }))
    }
}

#[async_trait]
impl S3DlioObjectWriter for ArrowWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }

        // Update integrity and logical byte count (uncompressed)
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;

        // Handle compression or append directly to buffer
        if let Some(compressor) = &mut self.compressor {
            use std::io::Write;
            compressor.write_all(chunk)?;
            // Note: zstd accumulates data internally, we get output in finalize()
        } else {
            // No compression: append directly to buffer
            self.buffer.extend_from_slice(chunk);
            self.compressed_bytes += chunk.len() as u64;
        }

        Ok(())
    }

    async fn write_owned_bytes(&mut self, data: Vec<u8>) -> Result<()> {
        self.write_chunk(&data).await
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        // Finish compression if enabled
        if let Some(compressor) = self.compressor.take() {
            let final_compressed = compressor.finish()?;
            self.buffer.extend_from_slice(&final_compressed);
            self.compressed_bytes = final_compressed.len() as u64; // Set total compressed size
        }

        // Put the entire buffer to the store
        let payload = PutPayload::from(Bytes::from(self.buffer.clone()));
        self.store.put(&self.path, payload).await?;
        
        Ok(())
    }

    fn bytes_written(&self) -> u64 { self.bytes_written }

    fn compressed_bytes(&self) -> u64 { self.compressed_bytes }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compression(&self) -> CompressionConfig { self.compression }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 { 1.0 } else { self.compressed_bytes as f64 / self.bytes_written as f64 }
    }

    async fn cancel(mut self: Box<Self>) -> Result<()> {
        // Just mark as finalized without uploading
        self.finalized = true;
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Factory function for creating Arrow-backed stores
// ----------------------------------------------------------------------------

// Set up Arrow environment on module load
fn setup_arrow_environment() {
    // Load environment variables from .env file (same as rest of s3dlio)
    let _ = dotenvy::dotenv();
    
    // Try multiple approaches to disable EC2 metadata service
    unsafe { 
        std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");
        std::env::set_var("AWS_IMDS_DISABLED", "true");
        std::env::set_var("AWS_EC2_METADATA_SERVICE_DISABLED", "true");
    }
    
    // Ensure required region is set
    if std::env::var("AWS_DEFAULT_REGION").is_err() && std::env::var("AWS_REGION").is_ok() {
        if let Ok(region) = std::env::var("AWS_REGION") {
            unsafe { std::env::set_var("AWS_DEFAULT_REGION", region); }
        }
    }
    
    // Log what we found for debugging (without exposing secrets)
    if std::env::var("AWS_ACCESS_KEY_ID").is_ok() && std::env::var("AWS_SECRET_ACCESS_KEY").is_ok() {
        println!("🔧 Arrow backend: Using credentials from environment");
    }
    if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
        println!("🔧 Arrow backend: Using custom endpoint: {}", endpoint);
    }
    if let Ok(region) = std::env::var("AWS_DEFAULT_REGION") {
        println!("🔧 Arrow backend: Using region: {}", region);
    }
}

/// Create an Arrow-backed ObjectStore from a URI.
/// This function provides the same interface as your existing store_for_uri factory.
pub fn store_for_uri(_uri: &str) -> Result<Box<dyn S3DlioObjectStore>> {
    // Ensure environment is set up before any Arrow operations
    setup_arrow_environment();
    
    // Arrow's object_store will automatically pick up standard AWS environment variables:
    // - AWS_ACCESS_KEY_ID
    // - AWS_SECRET_ACCESS_KEY  
    // - AWS_DEFAULT_REGION
    // - AWS_ENDPOINT_URL (for custom S3 endpoints)
    // - AWS_ALLOW_HTTP (for non-HTTPS endpoints)
    
    // Arrow's parse_url handles scheme detection automatically, so we can 
    // use a single ArrowObjectStore instance for all URI types
    Ok(ArrowObjectStore::boxed())
}