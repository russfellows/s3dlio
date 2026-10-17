// src/object_store.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::checksum::Crc32c;
use anyhow::anyhow;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use regex::Regex;
use std::collections::HashMap;
use std::io::Write;
use std::pin::Pin;
#[cfg(feature = "backend-gcs")]
use tokio::sync::OnceCell;
#[cfg(feature = "backend-gcs")]
use tracing::trace;
use tracing::{debug, info, warn};

// --- S3 ----------------------------------------------------------------------
use crate::delete_batch_plan::{
    aggregate_failures, global_client_server, group_uris_by_bucket, parse_s3_uri_for_server,
    parse_s3_uri_resolving_server, BucketGroup, GroupFailure, StoreServer,
};
use crate::s3_utils::{
    create_bucket_async as s3_create_bucket_async,
    delete_bucket_async as s3_delete_bucket_async,
    delete_objects_async as s3_delete_objects_async,
    get_object_range_uri_async as s3_get_object_range_uri_async,
    // The S3 backend of a store writes no row of its own: the store's caller records the call (issue #169).
    get_object_uri_optimized_rows as s3_get_object_uri_optimized_rows,
    get_object_uri_rows as s3_get_object_uri_rows,
    list_objects_stream as s3_list_objects_stream,
    put_object_multipart_uri_async as s3_put_object_multipart_uri_async,
    // NEW: PUT operations via ObjectStore
    put_object_uri_rows as s3_put_object_uri_rows,
    stat_object_uri_async as s3_stat_object_uri_async,
    // Reuse existing S3 helpers
    ObjectStat as S3ObjectStat,
    SdkResultExt,
};

use crate::s3_logger::global_logger;

// Expose FS adapter (already implemented in src/file_store.rs)
use crate::file_store::FileSystemObjectStore;

// Expose enhanced FS adapter with O_DIRECT support
use crate::file_store_direct::ConfigurableFileSystemObjectStore;

// --- Azure ---------------------------------------------------
#[cfg(feature = "backend-azure")]
use crate::azure_client::{AzureBlob, AzureBlobProperties};
#[cfg(feature = "backend-azure")]
use crate::constants::{
    DEFAULT_AZURE_DOWNLOAD_CONCURRENCY, DEFAULT_AZURE_DOWNLOAD_PART_SIZE,
    ENV_AZURE_CONCURRENT_ENGINE, ENV_AZURE_DOWNLOAD_CONCURRENCY, ENV_AZURE_DOWNLOAD_MODE,
    ENV_AZURE_DOWNLOAD_PART_SIZE_MB, ENV_AZURE_DOWNLOAD_THRESHOLD_MB,
};
use crate::constants::{
    DEFAULT_RANGE_ENGINE_CHUNK_SIZE, DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,
    DEFAULT_RANGE_ENGINE_THRESHOLD, DEFAULT_RANGE_TIMEOUT_SECS,
};
#[cfg(any(feature = "backend-azure", feature = "backend-gcs"))]
use crate::range_engine_generic::RangeEngine;
use crate::range_engine_generic::RangeEngineConfig;
use bytes::Bytes;
#[cfg(feature = "backend-azure")]
use futures::stream;
use std::time::Duration;

// --- GCS - Feature-gated backend selection -------------------
#[cfg(all(feature = "backend-gcs", feature = "gcs-community"))]
use crate::gcs_client::{parse_gcs_uri, GcsClient, GcsObjectMetadata};

#[cfg(all(feature = "backend-gcs", not(feature = "gcs-community")))]
use crate::google_gcs_client::{parse_gcs_uri, GcsClient, GcsObjectMetadata};

#[cfg(feature = "backend-gcs")]
static GLOBAL_GCS_CLIENT: OnceCell<GcsClient> = OnceCell::const_new();

#[cfg(feature = "backend-gcs")]
async fn get_global_gcs_client() -> Result<GcsClient> {
    let client = GLOBAL_GCS_CLIENT
        .get_or_try_init(|| async { GcsClient::new().await })
        .await?;
    Ok(client.clone())
}

/// Provider-neutral object metadata. For now this aliases S3's metadata.
pub type ObjectMetadata = S3ObjectStat;

/// Progress of one caller-buffer receive; see [`ObjectStore::get_into`].
pub use crate::receive_progress::ReceiveProgress;

/// Metadata from a typed HEAD ([`ObjectStore::head`]).
///
/// The Rust name differs from the [`ObjectMetadata`] alias, which is the provider-neutral result
/// of [`ObjectStore::stat`]. Only what a HEAD establishes safely is here: the size, and the
/// opaque identity (ETag, version) of the object it described.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadMetadata {
    /// The object size in bytes.
    pub size: u64,
    /// The entity tag, exactly as the provider reports it. Opaque: not assumed to be an MD5.
    pub etag: Option<String>,
    /// The provider's version identifier (S3 version id, GCS generation), when it reports one.
    pub version_id: Option<String>,
}

// -----------------------------------------------------------------------------
// Performance Configuration Helpers
// -----------------------------------------------------------------------------

/// Get the threshold size for using concurrent range requests.
/// Objects/transfers above this size will use concurrent range GET for optimal performance.
fn get_concurrent_threshold() -> u64 {
    // The `.env` policy first (#243, review R9/F): `get_range_optimized` with a known length reads this before any client exists.
    crate::dotenv::apply();
    std::env::var("S3DLIO_CONCURRENT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32 * 1024 * 1024) // Default: 32MB threshold
}

/// Compression configuration for ObjectWriter implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionConfig {
    #[default]
    None,
    Zstd {
        level: i32,
    },
}

impl CompressionConfig {
    /// Create Zstd compression with default level (3)
    pub fn zstd_default() -> Self {
        CompressionConfig::Zstd { level: 3 }
    }

    /// Create Zstd compression with custom level (1-22)
    pub fn zstd_level(level: i32) -> Self {
        CompressionConfig::Zstd {
            level: level.clamp(1, 22),
        }
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

    /// The name a compressing writer stores `name` under: the compression extension is
    /// appended only when the name does not already end with it (`a.bin` -> `a.bin.zst`,
    /// `a.zst` -> `a.zst`). With no compression the name is unchanged.
    ///
    /// This is the ONE suffix rule: every compressing writer uses it for the physical
    /// object name and every checkpoint helper uses it for the manifest name, so the
    /// two cannot disagree.
    pub fn stored_name(&self, name: &str) -> String {
        if self.is_enabled() && !name.ends_with(self.extension()) {
            format!("{name}{}", self.extension())
        } else {
            name.to_string()
        }
    }

    /// [`stored_name`](Self::stored_name) for a filesystem path (the extension is
    /// appended to the final component).
    pub fn stored_path(&self, path: &std::path::Path) -> std::path::PathBuf {
        if self.is_enabled()
            && !path
                .as_os_str()
                .to_string_lossy()
                .ends_with(self.extension())
        {
            let mut os = path.as_os_str().to_owned();
            os.push(self.extension());
            std::path::PathBuf::from(os)
        } else {
            path.to_path_buf()
        }
    }
}

/// Configuration options for creating object writers
#[derive(Debug, Clone, Default)]
pub struct WriterOptions {
    /// Optional compression configuration
    pub compression: Option<CompressionConfig>,
    /// Buffer size hint for streaming operations
    pub buffer_size: Option<usize>,
    /// Maximum part size for multipart uploads
    pub part_size: Option<usize>,
    /// Optional adaptive configuration for auto-tuning (default: disabled)
    pub adaptive: Option<crate::adaptive_config::AdaptiveConfig>,
}

impl WriterOptions {
    /// Create new WriterOptions with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression configuration
    pub fn with_compression(mut self, compression: CompressionConfig) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Set buffer size hint
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }

    /// Set maximum part size for multipart uploads
    pub fn with_part_size(mut self, size: usize) -> Self {
        self.part_size = Some(size);
        self
    }

    /// Enable adaptive tuning with default configuration
    pub fn with_adaptive(mut self) -> Self {
        self.adaptive = Some(crate::adaptive_config::AdaptiveConfig::enabled());
        self
    }

    /// Set custom adaptive configuration
    pub fn with_adaptive_config(mut self, config: crate::adaptive_config::AdaptiveConfig) -> Self {
        self.adaptive = Some(config);
        self
    }

    /// Compute effective part size considering adaptive tuning
    ///
    /// If part_size is explicitly set, it is always used.
    /// Otherwise, if adaptive is enabled, it computes the optimal part size.
    /// Falls back to default if neither is set.
    pub fn effective_part_size(&self, file_size: Option<usize>) -> usize {
        use crate::adaptive_config::AdaptiveParams;

        // If adaptive config is provided, use it to compute effective part size
        if let Some(ref adaptive_cfg) = self.adaptive {
            let params = AdaptiveParams::new(adaptive_cfg.clone());
            params.compute_part_size(file_size, self.part_size)
        } else {
            // No adaptive config: use explicit part_size or default
            self.part_size
                .unwrap_or(crate::constants::DEFAULT_S3_MULTIPART_PART_SIZE)
        }
    }

    /// Compute effective buffer size considering adaptive tuning
    pub fn effective_buffer_size(&self, operation_type: &str) -> usize {
        use crate::adaptive_config::AdaptiveParams;

        // If adaptive config is provided, use it to compute effective buffer size
        if let Some(ref adaptive_cfg) = self.adaptive {
            let params = AdaptiveParams::new(adaptive_cfg.clone());
            params.compute_buffer_size(operation_type, self.buffer_size)
        } else {
            // No adaptive config: use explicit buffer_size or default
            self.buffer_size.unwrap_or(1024 * 1024) // 1 MB default
        }
    }
}
/// Supported schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    File,
    Direct,
    S3,
    Azure,
    Gcs,
    Unknown,
}
/// Object properties for metadata update operations (v0.10.0+)
///
/// Used with `update_properties()` to modify object metadata without re-uploading data.
/// Only non-None fields will be updated. For cloud backends, this typically requires
/// copying the object with new metadata.
#[derive(Debug, Clone, Default)]
pub struct ObjectProperties {
    /// MIME content type (e.g., "application/json", "text/plain")
    pub content_type: Option<String>,

    /// HTTP Cache-Control header (e.g., "max-age=3600, public")
    pub cache_control: Option<String>,

    /// Content encoding (e.g., "gzip", "br")
    pub content_encoding: Option<String>,

    /// Content language (e.g., "en-US", "fr-FR")
    pub content_language: Option<String>,

    /// Content disposition (e.g., "attachment; filename=data.json")
    pub content_disposition: Option<String>,

    /// HTTP Expires header (RFC 2822 date format)
    pub expires: Option<String>,

    /// Storage class/tier for cost optimization
    /// - S3: "STANDARD", "INTELLIGENT_TIERING", "GLACIER", "DEEP_ARCHIVE"
    /// - GCS: "STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"
    /// - Azure: "Hot", "Cool", "Archive"
    pub storage_class: Option<String>,
}

/// Returns true when `uri`'s authority (host[:port]) component is, or ends
/// with, an Azure Blob Storage hostname (`*.blob.core.windows.net`).
///
/// audit #154 bug 4.2 (E1): the previous check was
/// `uri.contains(".blob.core.windows.net/")`, a substring match against the
/// *entire* URI. That falsely matches any URI whose path/key happens to
/// contain that literal text -- e.g. an S3 object key that mirrors an Azure
/// blob path, `s3://backups/mirror/acct.blob.core.windows.net/file.dat` --
/// misrouting it to the Azure backend. Restrict the check to the authority
/// component only, and require it as a suffix rather than a substring.
///
/// This is only the classifier behind [`infer_scheme`] and the pure Azure parser
/// (`capabilities::parse_azure_uri_pure`); both call this one function, so they cannot disagree
/// about what is an Azure host URL. It deliberately does NOT refuse userinfo or a backslash in
/// the authority: such a URI is still an Azure URL, so it is classified Azure and the parser then
/// refuses it with a clear error. Classifying it as "not Azure" instead would turn a smuggling
/// attempt into a silent `"unknown"` scheme. The account is never taken from this function.
pub(crate) fn is_azure_blob_host(uri: &str) -> bool {
    let Some(after_scheme_idx) = uri.find("://") else {
        return false;
    };
    let after_scheme = &uri[after_scheme_idx + 3..];
    let authority = after_scheme.split('/').next().unwrap_or("");
    // Strip optional "user@" userinfo and a trailing ":port".
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host);
    // Host names are case-insensitive.
    host.to_ascii_lowercase()
        .ends_with(".blob.core.windows.net")
}

/// Best-effort scheme inference from a URI.
pub fn infer_scheme(uri: &str) -> Scheme {
    if uri.starts_with("file://") {
        Scheme::File
    } else if uri.starts_with("direct://") {
        Scheme::Direct
    } else if uri.starts_with("s3://") {
        Scheme::S3
    } else if uri.starts_with("az://") || is_azure_blob_host(uri) {
        Scheme::Azure
    } else if uri.starts_with("gs://") || uri.starts_with("gcs://") {
        Scheme::Gcs
    } else {
        Scheme::Unknown
    }
}

/// The provider name the URI's scheme implies, as [`crate::object_error::ObjectError`] and
/// [`crate::capabilities::Capabilities`] spell it.
fn provider_for_uri(uri: &str) -> &'static str {
    match infer_scheme(uri) {
        Scheme::S3 => "s3",
        Scheme::Azure => "az",
        Scheme::Gcs => "gs",
        Scheme::File => "file",
        Scheme::Direct => "direct",
        Scheme::Unknown => "unknown",
    }
}

/// The error the default [`ObjectStore::get_into`] and [`ObjectStore::head`] return: kind
/// [`ObjectErrorKind::Unsupported`], attached over a plain description.
fn unsupported_operation(uri: &str, operation: &str) -> anyhow::Error {
    crate::object_error::attach_object_error(
        anyhow!("this store does not implement {operation}"),
        crate::object_error::ObjectErrorKind::Unsupported,
        provider_for_uri(uri),
        format!("{operation} is not supported by this store"),
    )
}

#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Get entire object into memory.
    ///
    /// Returns `Bytes` for zero-copy efficiency. The S3/Azure SDKs return data as `Bytes`,
    /// allowing us to avoid unnecessary allocations. Use `.as_ref()` to get &[u8] or
    /// `.to_vec()` only when ownership is required.
    async fn get(&self, uri: &str) -> Result<Bytes>;

    /// Get a byte-range. If `length` is None, read from `offset` to end.
    ///
    /// Returns `Bytes` for zero-copy efficiency.
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes>;

    /// Put full object (single-shot).
    ///
    /// Accepts `Bytes` for zero-copy efficiency (no memcpy when converting to SDK types).
    async fn put(&self, uri: &str, data: Bytes) -> Result<()>;

    /// Put via multipart semantics (or an equivalent high-throughput path).
    ///
    /// Accepts `Bytes` for zero-copy efficiency.
    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()>;

    /// List objects under a prefix. Returns full URIs.
    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>>;

    /// List objects as a stream for memory-efficient iteration over large result sets.
    /// Results are yielded as they arrive from the backend (typically in 1000-object pages).
    /// This allows displaying progress and avoids buffering millions of URIs in memory.
    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;

    /// Stat a single object (HEAD-like).
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata>;

    /// Stream the URIs of the REAL objects selected by `uri_prefix`, recursively (no
    /// delimiter, no regex, no synthetic directory entries).
    ///
    /// Used by `s3-cli delete -r` and the Python `delete(..., recursive=True)`.
    /// The S3 store applies the `s3://` selection rule (issue #213,
    /// `crate::s3_selection`): `s3://b/K` with no `*` selects the object `K` and
    /// everything under `K/` -- never `K.bak`; a trailing `/` selects everything under
    /// it; `*` / `**` make an explicit glob over the full key. Only `Contents` entries are
    /// yielded.  The default (other backends) streams their existing recursive listing and
    /// only accepts a prefix ending in `/`, so a bare name is never read as a pattern.
    fn list_objects_with_prefix<'a>(
        &'a self,
        uri_prefix: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        if uri_prefix.ends_with('/') {
            self.list_stream(uri_prefix, true)
        } else {
            let msg = format!(
                "recursive listing of '{uri_prefix}' is not supported on this backend without a \
                 trailing '/'"
            );
            Box::pin(futures::stream::once(
                async move { Err(anyhow::anyhow!(msg)) },
            ))
        }
    }

    /// Delete a single object.
    async fn delete(&self, uri: &str) -> Result<()>;

    /// Delete multiple objects efficiently (batched when backend supports it).
    /// Backends should override for optimal batch delete (e.g., S3 DeleteObjects API).
    ///
    /// # Semantics (S3, GCS and Azure stores; issue #179)
    ///
    /// - The batch may name objects in several buckets / containers; each URI is
    ///   deleted in the bucket (Azure: account + container) it names, never in
    ///   the bucket of the first URI.
    /// - **Every URI is validated before anything is deleted.**  A batch that
    ///   contains a URI the store cannot use fails as a whole, naming each such
    ///   URI, and deletes nothing.  This is a behaviour change: an unparseable
    ///   URI past the first used to be silently skipped while the rest of the
    ///   batch was deleted and `Ok(())` returned.
    /// - A store talks to exactly one server.  For S3, a URI of the host form
    ///   (`s3://host:port/bucket/key`) names a server; if it is not the
    ///   store's own server (a per-endpoint store: its endpoint host:port,
    ///   compared case-insensitively, with the scheme's default port filled in
    ///   and no DNS resolution; a store on the process-wide client: any
    ///   server-naming URI) the whole batch is rejected before any request.
    /// - Buckets are processed in order of first appearance, keys in caller
    ///   order within a bucket.  A failure in one bucket does not stop the
    ///   others; the returned error names the URIs not confirmed deleted, and
    ///   the first failure's cause chain is its source.
    async fn delete_batch(&self, uris: &[String]) -> Result<()>;

    /// Delete all objects under a prefix.
    ///
    /// The prefix names a directory-like name, never a string prefix of longer
    /// names (#231).  On every object store (S3, Azure, GCS) a non-empty key
    /// prefix that does not end in `/` gets one `/` appended before listing, so
    /// `delete_prefix(".../foo")` deletes only objects under `foo/`, never
    /// siblings such as `foobar`, `foo.bak` or `foo2/...`.  An empty key prefix
    /// (the bucket / container root) deletes everything.  The prefix is matched
    /// literally: no wildcard.
    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()>;

    /// Create a top-level container (S3: bucket; Azure: container; File: directory).
    async fn create_container(&self, name: &str) -> Result<()>;

    /// Delete a top-level container.
    async fn delete_container(&self, name: &str) -> Result<()>;

    /// Check existence via `stat()`. Backends can override for efficiency.
    async fn exists(&self, uri: &str) -> Result<bool> {
        Ok(self.stat(uri).await.is_ok())
    }

    /// Read the whole object into the caller's buffer.
    ///
    /// `dst.len()` is the expected length. `Ok(())` means exactly `dst.len()` bytes were
    /// written and the whole-object size equals it; any other outcome is an error, with
    /// the typed classification [`crate::object_error::object_error`] reports.
    /// `progress` reports the distinct destination positions filled, while the read runs
    /// and after it fails or is cancelled. It says which positions were touched, never
    /// that their data is valid.
    ///
    /// Every write into `dst` happens inside the returned future, so dropping it stops
    /// every writer.
    ///
    /// The default returns an error of kind
    /// [`ObjectErrorKind::Unsupported`](crate::object_error::ObjectErrorKind::Unsupported).
    /// It never falls back to a temporary download plus a copy. A store that supports the
    /// operation overrides it, and so must any wrapper that should forward it.
    async fn get_into(
        &self,
        uri: &str,
        _dst: &mut [u8],
        _progress: &ReceiveProgress,
    ) -> Result<()> {
        Err(unsupported_operation(uri, "get_into"))
    }

    /// Typed HEAD: the size and identity of an object, or `None` when it does not exist.
    ///
    /// Absence is `Ok(None)`. Every other failure, including a missing bucket or container,
    /// a refusal and a transport failure, is an error.
    ///
    /// The default returns an error of kind
    /// [`ObjectErrorKind::Unsupported`](crate::object_error::ObjectErrorKind::Unsupported).
    async fn head(&self, uri: &str) -> Result<Option<HeadMetadata>> {
        Err(unsupported_operation(uri, "head"))
    }

    /// Get object with integrity validation.
    ///
    /// `expected_checksum` is `<algorithm>:<hex>` (what writers report and manifests
    /// store; only `crc32c` is supported) or bare CRC-32C hex; see
    /// [`crate::checksum::parse_expected_crc32c`]. An unsupported or malformed value is an
    /// error before anything is fetched. Returns the data once it matches.
    async fn get_with_validation(
        &self,
        uri: &str,
        expected_checksum: Option<&str>,
    ) -> Result<Bytes> {
        if let Some(expected) = expected_checksum {
            crate::checksum::parse_expected_crc32c(expected)?;
        }
        let data = self.get(uri).await?;
        if let Some(expected) = expected_checksum {
            crate::checksum::verify_crc32c(&data, expected, uri)?;
        }
        Ok(data)
    }

    /// Get byte range with integrity validation.
    /// For partial reads, validates against the partial data checksum if provided
    /// (same checksum forms as [`ObjectStore::get_with_validation`]).
    async fn get_range_with_validation(
        &self,
        uri: &str,
        offset: u64,
        length: Option<u64>,
        expected_checksum: Option<&str>,
    ) -> Result<Bytes> {
        if let Some(expected) = expected_checksum {
            crate::checksum::parse_expected_crc32c(expected)?;
        }
        let data = self.get_range(uri, offset, length).await?;
        if let Some(expected) = expected_checksum {
            let what = format!("range {}[{}:{}]", uri, offset, offset + data.len() as u64);
            crate::checksum::verify_crc32c(&data, expected, &what)?;
        }
        Ok(data)
    }

    /// Load and validate checkpoint data with integrity checking.
    /// This method is specifically designed for checkpoint loading scenarios.
    /// (Same checksum forms as [`ObjectStore::get_with_validation`].)
    async fn load_checkpoint_with_validation(
        &self,
        checkpoint_uri: &str,
        expected_checksum: Option<&str>,
    ) -> Result<Vec<u8>> {
        if let Some(expected) = expected_checksum {
            crate::checksum::parse_expected_crc32c(expected)?;
        }
        let data = self.get(checkpoint_uri).await?;
        if let Some(expected) = expected_checksum {
            crate::checksum::verify_crc32c(&data, expected, checkpoint_uri)?;
        }

        // Convert Bytes to Vec<u8> for backward compatibility
        Ok(data.to_vec())
    }

    /// `true` only when this store's [`ObjectStore::copy`] is a NATIVE operation of the
    /// backend (a server-side copy, `copy_file_range`, ...) and not the trait default
    /// (a `get` followed by a `put` on this same store).
    ///
    /// A wrapper that records calls (`LoggedObjectStore`) needs to know: a native
    /// operation is one call to the backend that no op name describes, so the wrapper
    /// forwards it; the default is made of real `get` and `put` calls, which the
    /// wrapper must run over ITSELF so that each of them is recorded. A backend that
    /// overrides `copy` with a native operation must return `true` here.
    fn has_native_copy(&self) -> bool {
        false
    }

    /// `true` only when this store's [`ObjectStore::rename`] is a NATIVE operation of
    /// the backend (`rename(2)` on `file://` and `direct://`) and not the trait default
    /// (`copy` then `delete`). See [`ObjectStore::has_native_copy`] for why a call
    /// recorder needs to know. A backend that overrides `rename` natively must return
    /// `true` here.
    fn has_native_rename(&self) -> bool {
        false
    }

    /// Record object sizes this store has ALREADY learned (the answers of `stat`
    /// calls the caller made) in the store's size cache, without making any request.
    /// Default: the store has no size cache, nothing happens.
    ///
    /// `pre_stat_and_cache` is "stat these objects, then cache the sizes". A call
    /// recorder runs the stats through itself so that each HEAD is recorded, and then
    /// hands the sizes to the store it wraps with this method, so the cache fill the
    /// store's own `pre_stat_and_cache` would do still happens, with no extra request.
    async fn seed_size_cache(&self, sizes: &HashMap<String, u64>) {
        let _ = sizes;
    }

    /// Default copy reads then writes. Backends can override with server-side copy.
    async fn copy(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        let data = self.get(src_uri).await?;
        self.put(dst_uri, data).await
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

    /// Get a streaming writer with compression support for zero-copy operations.
    /// This enables writing large objects with compression without buffering the entire content in memory.
    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        // Default implementation: use get_writer (no compression)
        if compression.is_enabled() {
            bail!("Compression not supported by this ObjectStore implementation");
        }
        self.get_writer(uri).await
    }

    /// Create a new streaming writer with specified options.
    /// This is the unified interface for creating writers with configurable options.
    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        // Default implementation: use get_writer_with_compression if compression is specified
        if let Some(compression) = options.compression {
            self.get_writer_with_compression(uri, compression).await
        } else {
            self.get_writer(uri).await
        }
    }

    /// High-performance optimized GET operation for large objects.
    /// Automatically chooses between single GET and concurrent range requests
    /// based on object size and configured thresholds. This method provides
    /// the best performance for large object retrieval.
    async fn get_optimized(&self, uri: &str) -> Result<Bytes> {
        // Default implementation: delegate to regular get()
        // S3ObjectStore will override this with concurrent range logic
        self.get(uri).await
    }

    /// High-performance optimized range GET operation.
    /// Uses concurrent range requests for large transfers to maximize throughput.
    async fn get_range_optimized(
        &self,
        uri: &str,
        offset: u64,
        length: Option<u64>,
        _chunk_size: Option<usize>,
        _max_concurrency: Option<usize>,
    ) -> Result<Bytes> {
        // Default implementation: delegate to regular get_range()
        // S3ObjectStore will override this with concurrent range logic
        self.get_range(uri, offset, length).await
    }

    /// Pre-stat multiple objects concurrently to populate size cache (v0.9.10+)
    ///
    /// This is a performance optimization for workloads where object URIs are known
    /// upfront (e.g., benchmark tools like sai3-bench, batch processing pipelines).
    /// By pre-statting objects concurrently, we eliminate per-object stat latency
    /// during the download phase.
    ///
    /// # Performance Impact
    ///
    /// For 1000 object benchmark workload:
    /// - Without pre-stat: 1000 × 20ms stat = 20 seconds overhead (61% of total time)
    /// - With pre-stat: Pre-stat in 200ms (100 concurrent), then zero per-object overhead
    /// - Result: 2.5x faster (32.8s → 13.0s), 2.5x higher throughput (1.95 GB/s → 4.92 GB/s)
    ///
    /// # Arguments
    ///
    /// * `uris` - List of object URIs to stat
    /// * `max_concurrent` - Maximum concurrent stat operations (recommended: 100)
    ///
    /// # Returns
    ///
    /// Map of URI → size for successfully statted objects. Failed stats are logged
    /// and omitted from the result (graceful degradation).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use s3dlio::object_store::ObjectStore;
    ///
    /// # async fn example(store: &dyn ObjectStore) -> anyhow::Result<()> {
    /// let uris = vec![
    ///     "s3://bucket/object1.dat".to_string(),
    ///     "s3://bucket/object2.dat".to_string(),
    ///     // ... 1000 more objects
    /// ];
    ///
    /// // Pre-stat all objects concurrently
    /// let size_map = store.pre_stat_objects(&uris, 100).await?;
    /// println!("Pre-statted {} objects", size_map.len());
    ///
    /// // Now downloads can use cached sizes (if backend supports it)
    /// for uri in &uris {
    ///     let data = store.get(uri).await?;
    ///     // No stat overhead here!
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Backward Compatibility
    ///
    /// This method has a default implementation that stats objects CONCURRENTLY
    /// using the provided max_concurrent limit. No need to override unless you
    /// want custom behavior.
    ///
    /// Issue #148 site 3.1g caveat: the default implementation uses
    /// `.buffer_unordered(N)` which polls every in-flight stat inside a
    /// single tokio task. This limits per-stat CPU work (URI parsing,
    /// signing, HEAD response parsing) to one worker thread. Full task
    /// parallelism would require `tokio::spawn`, but that needs
    /// `Arc<Self>: 'static + Send + Sync` — not obtainable from a
    /// `&self` method through a `dyn ObjectStore`. Backends that need
    /// N-core parallelism should override with a spawn-based
    /// implementation using their own owned handle (see
    /// `crate::s3_utils::stat_object_many_async` for the pattern). For
    /// most callers, HEAD is small enough that this is not the
    /// bottleneck.
    async fn pre_stat_objects(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<std::collections::HashMap<String, u64>> {
        use futures::stream::{self, StreamExt};

        // Issue #211: `buffer_unordered(0)` never polls anything and hangs.
        crate::concurrency::require_concurrency_at_least_one("max_concurrent", max_concurrent)?;

        tracing::debug!(
            "Pre-statting {} objects (concurrent, max={})",
            uris.len(),
            max_concurrent
        );

        // Clone URIs to avoid lifetime issues in async closures
        let uri_vec: Vec<String> = uris.to_vec();

        // Use futures::stream to stat objects concurrently
        let results: Vec<Option<(String, u64)>> = stream::iter(uri_vec)
            .map(|uri| async move {
                match self.stat(&uri).await {
                    Ok(metadata) => {
                        tracing::trace!("Pre-stat success: {} ({} bytes)", uri, metadata.size);
                        Some((uri, metadata.size))
                    }
                    Err(e) => {
                        tracing::warn!("Pre-stat failed for {}: {}", uri, e);
                        None
                    }
                }
            })
            .buffer_unordered(max_concurrent)
            .collect()
            .await;

        // Collect successful results
        let size_map: std::collections::HashMap<String, u64> =
            results.into_iter().flatten().collect();

        tracing::info!(
            "Pre-statted {}/{} objects successfully (concurrent)",
            size_map.len(),
            uris.len()
        );

        Ok(size_map)
    }

    /// Pre-stat objects and populate internal size cache (v0.9.10+)
    ///
    /// This is a higher-level convenience method that pre-stats objects AND caches
    /// the results internally. After calling this, subsequent `get()` calls will
    /// use cached sizes and skip the per-object stat operation.
    ///
    /// # Arguments
    ///
    /// * `uris` - List of object URIs to stat
    /// * `max_concurrent` - Maximum concurrent stat operations (recommended: 100)
    ///
    /// # Returns
    ///
    /// Count of successfully cached entries
    ///
    /// # Example (sai3-bench usage pattern)
    ///
    /// ```no_run
    /// use s3dlio::object_store::ObjectStore;
    /// use std::time::Instant;
    ///
    /// # async fn example(store: &dyn ObjectStore) -> anyhow::Result<()> {
    /// let object_uris: Vec<String> = vec![/* 1000 objects */];
    ///
    /// // PHASE 1: Pre-stat all objects (runs once at start)
    /// let start = Instant::now();
    /// let cached = store.pre_stat_and_cache(&object_uris, 100).await?;
    /// println!("Pre-statted {} objects in {:?}", cached, start.elapsed());
    ///
    /// // PHASE 2: Download with zero stat overhead
    /// for uri in &object_uris {
    ///     let data = store.get(uri).await?;  // Uses cached size!
    ///     // Process data...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Backward Compatibility
    ///
    /// Default implementation calls `pre_stat_objects()` but doesn't cache results
    /// (returns the count but doesn't enable size cache benefits). Backends with
    /// size cache support will override this method.
    async fn pre_stat_and_cache(&self, uris: &[String], max_concurrent: usize) -> Result<usize> {
        // Default: just pre-stat without caching (backward compatible no-op)
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;
        Ok(size_map.len())
    }

    // =========================================================================
    // Metadata Operations (v0.10.0+)
    // =========================================================================

    /// Create a directory.
    ///
    /// **Backend behavior**:
    /// - `file://`, `direct://`: creates the directory and any missing parents
    ///   (`create_dir_all`); an existing directory is success. `direct://` is the
    ///   `file://` behaviour -- O_DIRECT has no meaning for a directory.
    /// - `s3://`, `gs://`, `az://`: a **no-op**. An object store has no directories;
    ///   a prefix exists once objects are written under it, so there is nothing to
    ///   create. The URI is validated, NO request is sent and no marker object is
    ///   written.
    ///
    /// Never recorded in the op-log.
    ///
    /// # Arguments
    /// * `uri` - Full URI including scheme (e.g., "file:///tmp/dir", "s3://bucket/prefix/")
    async fn mkdir(&self, uri: &str) -> Result<()> {
        let _ = uri;
        bail!("mkdir not implemented for this backend")
    }

    /// Remove a directory.
    ///
    /// **Backend behavior**:
    /// - `file://`, `direct://`: `recursive == false` removes only an EMPTY directory
    ///   (a non-empty one is an error and nothing is removed); `recursive == true`
    ///   removes the whole tree. A missing directory is success. `direct://` is the
    ///   `file://` behaviour.
    /// - `s3://`, `gs://`, `az://`: a **no-op**, `recursive` or not. An object store
    ///   has no directories, so there is nothing to remove. The URI is validated, NO
    ///   request is sent and NOTHING is deleted. To delete the objects under a prefix
    ///   use [`ObjectStore::delete_prefix`] (Python `delete(uri, recursive=True)`,
    ///   `s3-cli delete -r`).
    ///
    /// Recorded in the op-log as one `DELETE_PREFIX` row on `file://` and `direct://`
    /// only (issue #228); an object-store `rmdir` deletes nothing and writes no row.
    ///
    /// # Arguments
    /// * `uri` - Full URI including scheme
    /// * `recursive` - `file://` / `direct://`: if true, remove the whole tree; if
    ///   false, fail if not empty. Ignored on object stores.
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let _ = (uri, recursive);
        bail!("rmdir not implemented for this backend")
    }

    /// Update custom object metadata (cloud-specific, x-amz-meta-*, x-goog-meta-*, x-ms-meta-*).
    ///
    /// **Note**: For cloud backends, this typically requires copying the object with new metadata.
    /// For file backends, this is not applicable.
    async fn update_metadata(&self, uri: &str, metadata: &HashMap<String, String>) -> Result<()> {
        let _ = (uri, metadata);
        bail!("update_metadata not supported for this backend")
    }

    /// Update object properties (content-type, cache-control, storage class, etc).
    ///
    /// **Note**: For cloud backends, changing properties requires copying the object.
    /// For file backends, limited support.
    async fn update_properties(&self, uri: &str, properties: &ObjectProperties) -> Result<()> {
        let _ = (uri, properties);
        bail!("update_properties not supported for this backend")
    }
}

/// Concurrent deletion helper for efficient batch deletions across all backends.
///
/// This function deletes multiple objects concurrently through the ObjectStore trait,
/// providing significant performance improvements over sequential deletion while
/// maintaining universal backend compatibility.
///
/// # Adaptive Concurrency
/// - For small batches (< 10 objects): Sequential deletion (concurrency = 1)
/// - For medium batches: 10% of total objects (with min 10, max 1000)
/// - For large batches (10,000+ objects): Caps at 1000 concurrent operations
///
/// # Progress Updates
/// - Batched updates (every 50 operations) to minimize overhead
/// - Allows 98% reduction in progress bar overhead vs per-object updates
/// - Final update ensures accurate completion count
///
/// # Arguments
/// * `store` - Any ObjectStore implementation (S3, Azure, GCS, file, direct)
/// * `keys` - Slice of object URIs to delete
/// * `progress_callback` - Optional callback for progress updates (called every 50 deletions)
///
/// # Returns
/// * `Ok(())` - All objects deleted successfully
/// * `Err(_)` - First error encountered (other deletions may still be in progress)
///
/// # Example
/// ```ignore
/// let keys = vec!["s3://bucket/obj1", "s3://bucket/obj2"];
/// delete_objects_concurrent(&store, &keys, Some(|count| {
///     println!("Deleted {} objects", count);
/// })).await?;
/// ```
pub async fn delete_objects_concurrent<F>(
    store: &dyn ObjectStore,
    keys: &[String],
    progress_callback: Option<F>,
) -> Result<()>
where
    F: Fn(usize) + Send + Sync + 'static,
{
    use futures::stream::{self, StreamExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let total = keys.len();

    if total == 0 {
        return Ok(());
    }

    // Adaptive concurrency: 10% of total objects with reasonable min/max bounds
    let max_concurrency = if total < 10 {
        1 // Very small batches: sequential
    } else if total < 100 {
        10 // Small batches: minimum 10 concurrent
    } else if total < 10_000 {
        (total / 10).clamp(10, 100) // Medium batches: 10% with max 100
    } else {
        (total / 10).min(1000) // Large batches: 10% capped at 1000
    };

    debug!(
        "delete_objects_concurrent: {} objects, concurrency={}",
        total, max_concurrency
    );

    let completed = Arc::new(AtomicUsize::new(0));
    let last_reported = Arc::new(AtomicUsize::new(0));

    // Wrap callback in Arc for sharing across async tasks
    let callback_arc = progress_callback.map(Arc::new);

    // Progress reporting frequency: every 50 deletions or at completion
    let report_interval = 50;

    // Create concurrent deletion stream
    let deletions = stream::iter(keys.iter().enumerate())
        .map(|(idx, key)| {
            let key = key.clone();
            let completed = Arc::clone(&completed);
            let last_reported = Arc::clone(&last_reported);
            let callback = callback_arc.clone();

            async move {
                // Delete the object
                store.delete(&key).await?;

                // Update counter
                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;

                // Report progress in batches or on completion
                if let Some(ref cb) = callback {
                    let last = last_reported.load(Ordering::Relaxed);
                    if (count - last >= report_interval || count == total || idx == keys.len() - 1)
                        && last_reported
                            .compare_exchange(last, count, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                    {
                        cb(count);
                    }
                }

                Ok::<_, anyhow::Error>(())
            }
        })
        .buffer_unordered(max_concurrency);

    // Execute all deletions and collect results
    let results: Vec<Result<()>> = deletions.collect().await;

    // Check for errors
    for result in results {
        result?;
    }

    // Final progress update to ensure accurate count
    if let Some(callback) = callback_arc {
        let final_count = completed.load(Ordering::Relaxed);
        callback(final_count);
    }

    debug!("delete_objects_concurrent: completed {} deletions", total);

    Ok(())
}

/// Streaming writer interface for zero-copy object uploads
#[async_trait]
pub trait ObjectWriter: Send + Sync {
    /// Write a chunk of data to the object stream.
    /// Chunks are written in order and the implementation handles buffering/uploading.
    /// If compression is enabled, data is compressed before storage.
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()>;

    /// Write owned bytes for zero-copy optimization.
    /// Takes ownership of the data to avoid copying.
    async fn write_owned_bytes(&mut self, data: Vec<u8>) -> Result<()> {
        // Default implementation: convert to slice and call write_chunk
        self.write_chunk(&data).await
    }

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
    hasher: Crc32c,
}

impl<'a> BufferedObjectWriter<'a> {
    pub fn new(uri: String, store: &'a dyn ObjectStore) -> Self {
        Self {
            uri,
            store,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Crc32c::new(),
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
        self.store
            .put(&self.uri, Bytes::from(std::mem::take(&mut self.buffer)))
            .await
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

/// Configuration for S3ObjectStore
///
/// v0.9.10: Added size_cache_ttl for pre-stat optimization
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Enable RangeEngine for concurrent range downloads (struct config field)
    ///
    /// Default: false — set to true for large-file workloads.
    /// Note: S3 also has a separate env-var optimization path (`S3DLIO_ENABLE_RANGE_OPTIMIZATION`)
    /// that is enabled by default (v0.9.60+) and does not require this field to be set.
    pub enable_range_engine: bool,

    /// RangeEngine configuration
    /// Network-optimized defaults: 32 MiB threshold (v0.9.60+), 32 concurrent ranges, 64 MiB chunks
    pub range_engine: RangeEngineConfig,

    /// Time-to-live for cached object sizes
    /// Default: 60 seconds
    /// Set to 0 to disable caching
    pub size_cache_ttl_secs: u64,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            enable_range_engine: false, // Struct field off by default; S3 env-var optimization (S3DLIO_ENABLE_RANGE_OPTIMIZATION) is on by default (v0.9.60+)
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE, // 64 MiB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT, // 32 parallel
                min_split_size: DEFAULT_RANGE_ENGINE_THRESHOLD, // 32 MiB threshold (v0.9.60+)
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS), // 30s
            },
            size_cache_ttl_secs: 60, // 60 second TTL for size cache
        }
    }
}

#[derive(Clone)]
pub struct S3ObjectStore {
    // Note: S3ObjectStore doesn't use range_engine_generic (unlike GCS/Azure).
    // S3 concurrent range splitting is implemented in s3_utils::get_object_concurrent_range_async()
    // and wired in via get_optimized() / get_range_optimized().  See the comment at the top of
    // range_engine_generic.rs for full context and future consolidation notes.
    size_cache: std::sync::Arc<crate::object_size_cache::ObjectSizeCache>,
    /// Per-endpoint AWS S3 client. When `Some`, all operations use this client directly
    /// instead of the global singleton, enabling true per-endpoint connection pools.
    client: Option<std::sync::Arc<aws_sdk_s3::Client>>,
    /// The endpoint URL this store was created for (for logging / debugging).
    endpoint_url: Option<String>,
    /// The store is the endpoint store of a `MultiEndpointStore` on a bucket-form S3
    /// URI: its `get`/`put` write the S3 helper row (`S3Ops` shape) that they wrote
    /// before W2, because a `MultiEndpointStore` is never wrapped in a
    /// `LoggedObjectStore` and so has no other owner for the call (issue #175).
    /// `false` everywhere else: a store's caller records the call (issue #169).
    helper_rows: bool,
}

impl Default for S3ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl S3ObjectStore {
    pub fn new() -> Self {
        let config = S3Config::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            size_cache: std::sync::Arc::new(crate::object_size_cache::ObjectSizeCache::new(
                cache_ttl,
            )),
            client: None,
            endpoint_url: None,
            helper_rows: false,
        }
    }

    pub fn with_config(config: S3Config) -> Self {
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            size_cache: std::sync::Arc::new(crate::object_size_cache::ObjectSizeCache::new(
                cache_ttl,
            )),
            client: None,
            endpoint_url: None,
            helper_rows: false,
        }
    }

    /// Create an `S3ObjectStore` with its own `aws_sdk_s3::Client` configured for a    /// specific endpoint URL.
    ///
    /// Unlike [`S3ObjectStore::new`] (which uses the process-wide global client), this
    /// constructor creates a **dedicated** client with its own connection pool.  Each
    /// endpoint in a `MultiEndpointStore` should use a separate `for_endpoint` store so
    /// that PUT/GET traffic is truly distributed across connection pools.
    ///
    /// The endpoint URL should include the scheme and port, e.g.:
    /// - `"http://10.9.0.17:9000"` — MinIO / VAST S3-compatible
    /// - `"https://s3.us-east-1.amazonaws.com"` — AWS S3
    pub async fn for_endpoint(endpoint_url: &str) -> anyhow::Result<Self> {
        let client = crate::s3_client::create_s3_client_for_endpoint(endpoint_url, None).await?;
        let config = S3Config::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Ok(Self {
            size_cache: std::sync::Arc::new(crate::object_size_cache::ObjectSizeCache::new(
                cache_ttl,
            )),
            client: Some(std::sync::Arc::new(client)),
            endpoint_url: Some(endpoint_url.to_string()),
            helper_rows: false,
        })
    }

    /// Returns the endpoint URL this store was created for, if any.
    ///
    /// Returns `Some` only for stores created via [`S3ObjectStore::for_endpoint`].
    /// Returns `None` for stores that use the process-wide global client.
    pub fn endpoint_url(&self) -> Option<&str> {
        self.endpoint_url.as_deref()
    }

    /// Test-only constructor: records an endpoint URL without building a real
    /// AWS client.  Used in unit tests that verify `endpoint_url()` behaviour
    /// without requiring credentials or network access.
    #[cfg(test)]
    fn for_test(endpoint_url: &str) -> Self {
        let config = S3Config::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            size_cache: std::sync::Arc::new(crate::object_size_cache::ObjectSizeCache::new(
                cache_ttl,
            )),
            client: None,
            endpoint_url: Some(endpoint_url.to_string()),
            helper_rows: false,
        }
    }

    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }

    /// A store on the process-wide client whose `get`/`put` write the S3 helper
    /// row (the row the helpers `get_object` / `put_object_async` write when the
    /// op-log is active). Only for the endpoint stores of a `MultiEndpointStore`
    /// on bucket-form S3 URIs, which are never wrapped in a `LoggedObjectStore`;
    /// see the `helper_rows` field.
    pub(crate) fn with_helper_rows(mut self) -> Self {
        self.helper_rows = true;
        self
    }

    /// Whether `get` / `put` on the process-wide client write the helper row.
    fn rows(&self) -> crate::s3_utils::Rows {
        if self.helper_rows {
            crate::s3_utils::Rows::Write
        } else {
            crate::s3_utils::Rows::Skip
        }
    }

    /// Get object size, checking cache first
    ///
    /// v0.9.10: Added to optimize get_optimized() and get_range_optimized()
    /// by eliminating redundant stat() calls.
    async fn get_object_size(&self, uri: &str) -> Result<u64> {
        // Check cache first
        if let Some(cached_size) = self.size_cache.get(uri).await {
            return Ok(cached_size);
        }

        // Cache miss - perform stat and cache result
        let metadata = self.stat(uri).await?;
        self.size_cache.put(uri.to_string(), metadata.size).await;
        Ok(metadata.size)
    }

    /// The one server this store talks to, for comparison with the server a
    /// URI names: the endpoint of a per-endpoint store, or -- for a store on
    /// the process-wide client -- the server `AWS_ENDPOINT_URL` names (`None`
    /// when that is unset: the default AWS endpoint).
    fn server_identity(&self) -> Result<Option<StoreServer>> {
        match self.endpoint_url.as_deref() {
            Some(url) => StoreServer::from_endpoint_url(url)
                .map(Some)
                .context("S3ObjectStore endpoint URL is not usable"),
            None => Ok(global_client_server()),
        }
    }

    /// `parse_s3_uri` for this store: additionally rejects a URI that names a
    /// server other than the store's own (issues #179, #197).  The store would
    /// otherwise apply that URI's bucket/key to ITS server.  The store's server
    /// is only worked out when the URI names one, so bucket-form URIs (the hot
    /// path) pay nothing for the check.
    fn parse_uri_for_this_store(&self, uri: &str) -> Result<(String, String)> {
        parse_s3_uri_resolving_server(uri, || self.server_identity())
            .with_context(|| format!("S3 URI '{uri}' cannot be used with this store"))
    }

    /// The buffered writer for `uri` on THIS store (issue #208).
    ///
    /// A URI naming another server is refused (#197).  A per-endpoint store hands
    /// the writer its own client and the bucket/key it parsed, so the writer's final
    /// multipart upload -- create, parts, complete, and abort on a failure -- goes
    /// to the store's server; a store on the process-wide client builds the writer
    /// exactly as before.
    fn buffered_writer_for(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;
        let writer = S3BufferedWriter::new_with_compression(uri.to_string(), compression);
        Ok(Box::new(match &self.client {
            Some(client) => writer.with_target(client.clone(), bucket, key),
            None => writer,
        }))
    }

    /// Delete every key of one bucket group (issue #179).
    ///
    /// On failure returns the index (into `group.uris`) of the first URI that
    /// was not confirmed deleted, plus the error.  Requests are issued in
    /// order, at most `DELETE_BATCH_MAX_KEYS` keys each, and stop at the first
    /// failing request, exactly as the pre-#179 single-bucket loop did.
    async fn delete_bucket_group(
        &self,
        group: &BucketGroup<String>,
    ) -> std::result::Result<(), (usize, anyhow::Error)> {
        let bucket = &group.target;

        if let Some(client) = &self.client {
            let mut done = 0usize;
            for (_chunk_uris, chunk_keys) in group.chunks() {
                Self::delete_chunk_with_client(client, bucket, chunk_keys)
                    .await
                    .map_err(|e| (done, e))?;
                done += chunk_keys.len();
            }
            return Ok(());
        }

        // Global-client path: s3_delete_objects_async chunks at 1000 itself
        // and checks the per-key results of every response.
        s3_delete_objects_async(bucket, &group.keys)
            .await
            .map_err(|e| (0, e))
    }

    /// One DeleteObjects request (<= 1000 keys) through a per-endpoint client.
    async fn delete_chunk_with_client(
        client: &aws_sdk_s3::Client,
        bucket: &str,
        keys: &[String],
    ) -> Result<()> {
        // Fallible collect: a key that cannot be turned into an identifier is
        // an error, never silently left out of the request.
        let objects = keys
            .iter()
            .map(|k| {
                aws_sdk_s3::types::ObjectIdentifier::builder()
                    .key(k)
                    .build()
                    .with_context(|| format!("Failed to build DeleteObjects entry for key '{k}'"))
            })
            .collect::<Result<Vec<aws_sdk_s3::types::ObjectIdentifier>>>()?;
        let delete = aws_sdk_s3::types::Delete::builder()
            .set_objects(Some(objects))
            .build()
            .context("Failed to build Delete request")?;
        let output = client
            .delete_objects()
            .bucket(bucket)
            .delete(delete)
            .send()
            .await
            .with_context(|| format!("S3 batch DELETE failed for bucket '{}'", bucket))?;

        // S3 reports per-key failures inside an HTTP 200 body (see
        // s3_utils::delete_objects_async, audit #151 f36); do not discard them.
        let errs = output.errors();
        if !errs.is_empty() {
            let detail = errs
                .iter()
                .map(|e| {
                    format!(
                        "{}: {} ({})",
                        e.key().unwrap_or("<unknown key>"),
                        e.code().unwrap_or("<unknown code>"),
                        e.message().unwrap_or("<no message>")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "S3 batch DELETE for bucket '{}' partially failed: {} of {} object(s) not \
                 deleted: {}",
                bucket,
                errs.len(),
                keys.len(),
                detail
            );
        }
        Ok(())
    }

    /// List what the key part `key` of an `s3://` URI selects, using a given client (shared
    /// implementation for list() and list_stream()).
    ///
    /// The rule (issue #213) is `crate::s3_selection`: a key with no `*` that does not end
    /// in `/` selects that exact object and (when `recursive`) everything under `key/`,
    /// never `key.bak`; a key ending in `/` selects everything under it; a key with `*` is
    /// a glob over the full key. Non-recursive, directories appear as `.../` entries.
    /// There is no client-side regex here; an explicit one is applied by the caller
    /// (`ListPattern`).
    async fn list_with_client(
        client: &aws_sdk_s3::Client,
        bucket: &str,
        key: &str,
        recursive: bool,
    ) -> Result<Vec<String>> {
        use futures::StreamExt;
        let mut stream = crate::s3_selection::select_key_stream(
            client.clone(),
            bucket.to_string(),
            key,
            recursive,
            false,
        );
        let mut keys = Vec::new();
        while let Some(item) = stream.next().await {
            keys.push(format!("s3://{}/{}", bucket, item?));
        }
        keys.sort();
        Ok(keys)
    }

    /// The keys under the LITERAL prefix `prefix` (`s3://bucket/<key>` URIs), read as a plain
    /// S3 key prefix: no wildcard, no exact-object rule. Used by `delete_prefix`, whose
    /// prefix always ends in `/`.
    async fn list_literal_prefix_with_client(
        client: &aws_sdk_s3::Client,
        bucket: &str,
        prefix: &str,
    ) -> Result<Vec<String>> {
        use futures::StreamExt;
        let mut stream = crate::s3_selection::literal_prefix_key_stream(
            client.clone(),
            bucket.to_string(),
            prefix.to_string(),
        );
        let mut keys = Vec::new();
        while let Some(item) = stream.next().await {
            keys.push(format!("s3://{}/{}", bucket, item?));
        }
        keys.sort();
        Ok(keys)
    }
}

#[async_trait]
impl ObjectStore for S3ObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!("S3ObjectStore::get uri='{}'", uri);

        if let Some(client) = &self.client {
            let (bucket, key) = self.parse_uri_for_this_store(uri)?;
            // v0.9.108+ (issue #148 Phase 4b): wrap send()+collect() in
            // retry_get_body so body-transfer failures (no longer
            // covered by smithy's send()-level retry once the connector
            // streams) are retried at this caller level. GETs are
            // idempotent; each attempt allocates a fresh Bytes.
            let data = crate::retry::retry_get_body(|| async {
                let resp = client
                    .get_object()
                    .bucket(&bucket)
                    .key(&key)
                    .send()
                    .await
                    .sdk_context(format!("S3 GET failed for '{}'", uri))?;
                let data = resp
                    .body
                    .collect()
                    .await
                    .context("Failed to read S3 GET response body")?;
                Ok::<_, anyhow::Error>(data.into_bytes())
            })
            .await?;
            return Ok(data);
        }

        // Range optimization is ENABLED by default (v0.9.60+).
        // Set S3DLIO_ENABLE_RANGE_OPTIMIZATION=0 (or false/no/off/disable) to disable.
        // Setting it to 1/true/yes/on/enable is accepted but has no effect (already on).
        // The policy first: this read is per call (not latched), so it must see the file.
        crate::dotenv::apply();
        let use_optimized = std::env::var("S3DLIO_ENABLE_RANGE_OPTIMIZATION")
            .ok()
            .map(|v| {
                !matches!(
                    v.to_lowercase().as_str(),
                    "0" | "false" | "no" | "off" | "disable" | "disabled"
                )
            })
            .unwrap_or(true); // Default: enabled

        if use_optimized {
            s3_get_object_uri_optimized_rows(uri, self.rows()).await
        } else {
            s3_get_object_uri_rows(uri, self.rows()).await
        }
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!(
            "S3ObjectStore::get_range uri='{}', offset={}, length={:?}",
            uri, offset, length
        );

        // audit #152 finding 2.5 (bug B4): a zero-length range request has
        // no valid `bytes=start-end` header at all (RFC 7233 can't express
        // an empty range) — short-circuit it here instead of falling into
        // the `offset + len - 1` underflow below.
        if length == Some(0) {
            return Ok(Bytes::new());
        }

        if let Some(client) = &self.client {
            let (bucket, key) = self.parse_uri_for_this_store(uri)?;
            let range_header = match length {
                Some(len) => format!(
                    "bytes={}-{}",
                    offset,
                    crate::range_engine_generic::range_end_inclusive(offset, len)?
                ),
                None => format!("bytes={}-", offset),
            };
            // v0.9.108+ (issue #148 Phase 4b): same reasoning as the
            // whole-object get() above — wrap send()+collect() in the
            // shared retry helper.
            let data = crate::retry::retry_get_body(|| async {
                let resp = client
                    .get_object()
                    .bucket(&bucket)
                    .key(&key)
                    .range(&range_header)
                    .send()
                    .await
                    .sdk_context(format!("S3 range GET failed for '{}'", uri))?;
                let data = resp
                    .body
                    .collect()
                    .await
                    .context("Failed to read S3 range GET response body")?;
                Ok::<_, anyhow::Error>(data.into_bytes())
            })
            .await?;
            return Ok(data);
        }

        s3_get_object_range_uri_async(uri, offset, length).await
    }

    /// Caller-buffer GET (plan §4.5 A1, A6, A8). A per-endpoint store reads through its OWN
    /// client and writes no row; a store on the process-wide client passes `self.rows()`, so
    /// only a helper-row store (an endpoint store of a `MultiEndpointStore`) writes the S3
    /// helper row. Otherwise the caller (`LoggedObjectStore`) writes the one logical row.
    async fn get_into(&self, uri: &str, dst: &mut [u8], progress: &ReceiveProgress) -> Result<()> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!(
            "S3ObjectStore::get_into uri='{}', expected_len={}",
            uri,
            dst.len()
        );
        // Issue #197: a URI naming another server is an error, never run here.
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;

        match &self.client {
            Some(client) => {
                crate::s3_utils::get_into_with_client(client, &bucket, &key, dst, progress).await
            }
            None => {
                crate::s3_utils::get_into_global(&bucket, &key, dst, progress, self.rows()).await
            }
        }
    }

    /// Typed HEAD: `Ok(None)` only for typed absence. Client choice and row ownership are
    /// those of [`S3ObjectStore::get_into`].
    async fn head(&self, uri: &str) -> Result<Option<HeadMetadata>> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!("S3ObjectStore::head uri='{}'", uri);
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;

        match &self.client {
            Some(client) => crate::s3_utils::head_typed_with_client(client, &bucket, &key).await,
            None => crate::s3_utils::head_typed(&bucket, &key, self.rows()).await,
        }
    }

    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!("S3ObjectStore::put uri='{}', {} bytes", uri, data.len());

        if let Some(client) = &self.client {
            let (bucket, key) = self.parse_uri_for_this_store(uri)?;
            if crate::s3_client::unsigned_payload_enabled() {
                client
                    .put_object()
                    .bucket(&bucket)
                    .key(&key)
                    .body(data.into())
                    .customize()
                    .disable_payload_signing()
                    .send()
                    .await
                    .sdk_context(format!("S3 PUT (unsigned) failed for '{}'", uri))?;
            } else {
                client
                    .put_object()
                    .bucket(&bucket)
                    .key(&key)
                    .body(data.into())
                    .send()
                    .await
                    .sdk_context(format!("S3 PUT failed for '{}'", uri))?;
            }
            return Ok(());
        }

        s3_put_object_uri_rows(uri, data, self.rows()).await
    }

    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!(
            "S3ObjectStore::put_multipart uri='{}', {} bytes, part_size={:?}",
            uri,
            data.len(),
            part_size
        );
        // Issue #197: a URI naming another server is an error, never run here.
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;

        // Issue #208: a per-endpoint store uploads (create, parts, complete, abort)
        // with its OWN client.
        if let Some(client) = &self.client {
            return crate::s3_utils::put_object_multipart_with_client(
                client, &bucket, &key, data, part_size,
            )
            .await;
        }

        s3_put_object_multipart_uri_async(uri, data, part_size).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        debug!(
            "S3ObjectStore::list prefix='{}', recursive={}",
            uri_prefix, recursive
        );
        let (bucket, key_prefix) = self.parse_uri_for_this_store(uri_prefix)?;

        if let Some(client) = &self.client {
            return Self::list_with_client(client, &bucket, &key_prefix, recursive).await;
        }

        // Use the async streaming version to avoid nested run_on_global_rt deadlock.
        // s3_list_objects (sync) cannot be called from inside the global runtime.
        let mut stream = s3_list_objects_stream(bucket.clone(), key_prefix, recursive).await?;
        let mut keys = Vec::new();
        use futures::StreamExt;
        while let Some(result) = stream.next().await {
            let key = result?;
            keys.push(format!("s3://{}/{}", bucket, key));
        }
        keys.sort();
        debug!("S3ObjectStore::list result: {} keys", keys.len());
        Ok(keys)
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let (bucket, key_prefix) = match self.parse_uri_for_this_store(uri_prefix) {
                Ok((b, k)) => (b, k),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            if let Some(client) = &self.client {
                match S3ObjectStore::list_with_client(client, &bucket, &key_prefix, recursive).await {
                    Ok(keys) => {
                        for key in keys { yield Ok(key); }
                    }
                    Err(e) => { yield Err(e); }
                }
                return;
            }

            let mut stream = match s3_list_objects_stream(bucket.clone(), key_prefix, recursive).await {
                Ok(s) => s,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            while let Some(result) = stream.next().await {
                match result {
                    Ok(key) => yield Ok(format!("s3://{}/{}", bucket, key)),
                    Err(e) => yield Err(e),
                }
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!("S3ObjectStore::stat uri='{}'", uri);

        if let Some(client) = &self.client {
            let (bucket, key) = self.parse_uri_for_this_store(uri)?;
            // Issue #172: the key the caller names is the key sent -- no
            // stripping (get_range above already used it verbatim).
            let resp = client
                .head_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
                .sdk_context(format!("S3 HEAD failed for '{}'", uri))?;
            return Ok(crate::s3_utils::ObjectStat {
                size: resp.content_length().unwrap_or_default() as u64,
                last_modified: resp.last_modified().map(|t| t.to_string()),
                e_tag: resp.e_tag().map(|s| s.to_string()),
                content_type: resp.content_type().map(|s| s.to_string()),
                content_language: resp.content_language().map(|s| s.to_string()),
                content_encoding: resp.content_encoding().map(|s| s.to_string()),
                cache_control: resp.cache_control().map(|s| s.to_string()),
                content_disposition: resp.content_disposition().map(|s| s.to_string()),
                expires: resp.expires_string().map(|s| s.to_string()),
                storage_class: resp.storage_class().map(|s| s.as_str().to_string()),
                server_side_encryption: resp
                    .server_side_encryption()
                    .map(|s| s.as_str().to_string()),
                ssekms_key_id: resp.ssekms_key_id().map(|s| s.to_string()),
                sse_customer_algorithm: resp.sse_customer_algorithm().map(|s| s.to_string()),
                version_id: resp.version_id().map(|s| s.to_string()),
                replication_status: resp.replication_status().map(|s| s.as_str().to_string()),
                metadata: resp.metadata().cloned().unwrap_or_default(),
            });
        }

        s3_stat_object_uri_async(uri).await
    }

    fn list_objects_with_prefix<'a>(
        &'a self,
        uri_prefix: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let (bucket, key_prefix) = match self.parse_uri_for_this_store(uri_prefix) {
                Ok(parsed) => parsed,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            let client = match &self.client {
                Some(client) => (**client).clone(),
                None => match crate::s3_client::aws_s3_client_async().await {
                    Ok(client) => client,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                },
            };
            let mut stream =
                crate::s3_utils::list_object_uris_with_prefix_stream(client, bucket, key_prefix);
            while let Some(item) = stream.next().await {
                yield item;
            }
        })
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        debug!("S3ObjectStore::delete uri='{}'", uri);

        // A URI naming a server other than this store's own is an error, not
        // a delete of the same bucket/key on this store's server (#179).
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;
        if key.is_empty() {
            bail!("cannot delete '{uri}': it names a bucket, not an object");
        }

        if let Some(client) = &self.client {
            client
                .delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
                .with_context(|| format!("S3 DELETE failed for '{}'", uri))?;
            return Ok(());
        }

        s3_delete_objects_async(&bucket, &[key]).await
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        if uris.is_empty() {
            return Ok(());
        }
        debug!("S3ObjectStore::delete_batch {} URIs", uris.len());

        // Issue #179: a batch may span several buckets, and DeleteObjects
        // addresses exactly one.  Group by bucket (validating every URI first,
        // so a batch we cannot fully interpret deletes nothing), then issue
        // the requests per bucket.  A single-bucket batch is one group and
        // behaves exactly as before.  A store talks to ONE server, so a URI
        // naming any other server is rejected in the same up-front pass.
        let server = self.server_identity()?;
        let groups = group_uris_by_bucket(uris, |u| parse_s3_uri_for_server(u, server.as_ref()))?;
        let mut failures: Vec<GroupFailure> = Vec::new();
        for group in &groups {
            if let Err((first_undeleted, error)) = self.delete_bucket_group(group).await {
                failures.push(GroupFailure {
                    target: format!("s3://{}", group.target),
                    uris: group.uris.iter().skip(first_undeleted).cloned().collect(),
                    error,
                });
            }
        }
        aggregate_failures(uris.len(), failures)
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        use futures::stream::StreamExt;
        debug!("S3ObjectStore::delete_prefix prefix='{}'", uri_prefix);

        let (bucket, key_prefix) = self.parse_uri_for_this_store(uri_prefix)?;
        let key_prefix = directory_prefix(&key_prefix);

        if let Some(client) = &self.client {
            let listed =
                Self::list_literal_prefix_with_client(client, &bucket, &key_prefix).await?;
            // `list_literal_prefix_with_client` returns `s3://<bucket>/<key>`: strip that
            // head rather than re-parsing (a re-parse could misread the key,
            // and a listing entry that does not parse must be an error, never
            // a key silently left out of the delete).
            let uri_head = format!("s3://{bucket}/");
            let just_keys = listed
                .iter()
                .map(|u| {
                    u.strip_prefix(uri_head.as_str())
                        .map(str::to_string)
                        .with_context(|| {
                            format!("unexpected listing entry '{u}' (not under '{uri_head}')")
                        })
                })
                .collect::<Result<Vec<String>>>()?;
            let total = just_keys.len();
            let mut confirmed = 0usize;
            for chunk in just_keys.chunks(crate::delete_batch_plan::DELETE_BATCH_MAX_KEYS) {
                // Checks the per-key results inside the HTTP 200 body (#196).
                if let Err(e) = Self::delete_chunk_with_client(client, &bucket, chunk).await {
                    return Err(e.context(format!(
                        "S3 delete_prefix of 's3://{bucket}/{key_prefix}' failed: {confirmed} of \
                         {total} key(s) were confirmed deleted before the failing request, {} \
                         later key(s) were not attempted",
                        total - confirmed - chunk.len()
                    )));
                }
                confirmed += chunk.len();
            }
            return Ok(());
        }

        // Use streaming list to avoid loading millions of keys into memory. The prefix is
        // LITERAL here (it always ends in '/'): a '*' in it is a key character, not a glob.
        let global_client = crate::s3_client::aws_s3_client_async().await?;
        let mut stream = crate::s3_selection::literal_prefix_key_stream(
            global_client,
            bucket.clone(),
            key_prefix,
        );
        let mut batch = Vec::with_capacity(1000);
        let mut total_deleted = 0;

        while let Some(result) = stream.next().await {
            match result {
                Ok(key) => {
                    batch.push(key);

                    // Delete in batches of 1000 (S3 DeleteObjects API limit)
                    if batch.len() >= 1000 {
                        s3_delete_objects_async(&bucket, &batch).await?;
                        total_deleted += batch.len();
                        batch.clear();
                    }
                }
                Err(e) => return Err(e),
            }
        }

        // Delete final partial batch if any
        if !batch.is_empty() {
            s3_delete_objects_async(&bucket, &batch).await?;
            total_deleted += batch.len();
        }

        debug!("S3 delete_prefix deleted {} objects total", total_deleted);
        Ok(())
    }

    /// No-op: an object store has no directories (issue #229). A prefix exists once
    /// objects are written under it, so there is nothing to create. The URI is
    /// validated; NO request is sent and no object (marker or otherwise) is written.
    async fn mkdir(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        self.parse_uri_for_this_store(uri)?;
        debug!(
            "S3ObjectStore::mkdir is a no-op on an object store: '{}'",
            uri
        );
        Ok(())
    }

    /// No-op: an object store has no directories (issue #229), so there is nothing to
    /// remove -- `recursive` or not. The URI is validated; NO request is sent and
    /// nothing is deleted. To delete the objects under a prefix use
    /// [`ObjectStore::delete_prefix`] (Python `delete(uri, recursive=True)`,
    /// `s3-cli delete -r`).
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let _ = recursive;
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        self.parse_uri_for_this_store(uri)?;
        debug!(
            "S3ObjectStore::rmdir is a no-op on an object store: '{}'",
            uri
        );
        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        // A per-endpoint store (built for `s3://host:port/...`) must create the
        // bucket on ITS server, not on the global singleton's (issue #194).
        match &self.client {
            Some(client) => crate::s3_utils::create_bucket_with_client(client, name).await,
            None => s3_create_bucket_async(name).await,
        }
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        match &self.client {
            Some(client) => crate::s3_utils::delete_bucket_with_client(client, name).await,
            None => s3_delete_bucket_async(name).await,
        }
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // Issue #197: a URI naming another server is an error, never run here.
        // For S3, use buffered writer that collects chunks then uses put_multipart
        self.buffered_writer_for(uri, CompressionConfig::None)
    }

    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        // For S3, use buffered writer with compression support
        self.buffered_writer_for(uri, compression)
    }

    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        // For S3, use buffered writer that collects chunks then uses put_multipart
        self.buffered_writer_for(uri, options.compression.unwrap_or(CompressionConfig::None))
    }

    async fn get_optimized(&self, uri: &str) -> Result<Bytes> {
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }

        // Issue #197: a URI naming another server is an error, never run here.
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;
        // Get object size from cache or stat (v0.9.10: cache optimization)
        let object_size = self.get_object_size(uri).await?;

        // Use threshold-based decision for optimization
        let threshold = get_concurrent_threshold();

        if object_size >= threshold {
            debug!(
                "Using concurrent range GET for large object: {} bytes",
                object_size
            );
            // Use concurrent range GET for large objects.  A per-endpoint store
            // reads with its OWN client (issue #208), whatever the object size.
            match &self.client {
                Some(client) => {
                    crate::s3_utils::get_object_concurrent_range_with_client(
                        client, &bucket, &key, 0, None, None, None,
                    )
                    .await
                }
                None => {
                    crate::s3_utils::get_object_concurrent_range_async(uri, 0, None, None, None)
                        .await
                }
            }
        } else {
            debug!("Using standard GET for small object: {} bytes", object_size);
            // Use standard GET for small objects
            self.get(uri).await
        }
    }

    async fn get_range_optimized(
        &self,
        uri: &str,
        offset: u64,
        length: Option<u64>,
        chunk_size: Option<usize>,
        max_concurrency: Option<usize>,
    ) -> Result<Bytes> {
        // Issue #211: a limit of 0 is an error whatever the transfer size, so the
        // small-transfer path (which ignores the limits) accepts exactly what the
        // concurrent path accepts. Nothing is sent for a 0.
        crate::s3_utils::require_concurrent_range_limits(chunk_size, max_concurrency)?;
        if !uri.starts_with("s3://") {
            bail!("S3ObjectStore expected s3:// URI");
        }
        // Issue #197: a URI naming another server is an error, never run here.
        let (bucket, key) = self.parse_uri_for_this_store(uri)?;

        let transfer_size = match length {
            Some(len) => len,
            None => {
                // Get object size from cache or stat (v0.9.10: cache optimization)
                let object_size = self.get_object_size(uri).await?;
                object_size.saturating_sub(offset)
            }
        };

        // Use threshold-based decision for optimization
        let threshold = get_concurrent_threshold();

        if transfer_size >= threshold {
            debug!(
                "Using concurrent range GET for large transfer: {} bytes",
                transfer_size
            );
            // Use concurrent range GET for large transfers.  A per-endpoint store
            // reads with its OWN client (issue #208), whatever the transfer size.
            match &self.client {
                Some(client) => {
                    crate::s3_utils::get_object_concurrent_range_with_client(
                        client,
                        &bucket,
                        &key,
                        offset,
                        length,
                        chunk_size,
                        max_concurrency,
                    )
                    .await
                }
                None => {
                    crate::s3_utils::get_object_concurrent_range_async(
                        uri,
                        offset,
                        length,
                        chunk_size,
                        max_concurrency,
                    )
                    .await
                }
            }
        } else {
            debug!(
                "Using standard range GET for small transfer: {} bytes",
                transfer_size
            );
            // Use standard range GET for small transfers
            self.get_range(uri, offset, length).await
        }
    }

    /// Pre-stat objects and populate the size cache
    ///
    /// v0.9.10: Override default implementation to populate internal size cache.
    /// This enables subsequent get_optimized() calls to skip redundant stat operations.
    ///
    /// # Performance Impact
    ///
    /// For workloads that download many objects (e.g., benchmarking 1000+ objects):
    /// - Eliminates per-object stat latency (typically 10-50ms each)
    /// - Trades one-time concurrent pre-stat (e.g., 200ms for 1000 objects @ 100 concurrent)
    ///   for N × stat_latency savings (e.g., 1000 × 20ms = 20 seconds)
    /// - Expected speedup: 2-3x for large object sets
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use s3dlio::api::store_for_uri;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let store = store_for_uri("s3://my-bucket/prefix/")?;
    /// let objects: Vec<String> = vec![/* 1000 s3:// URIs */];
    ///
    /// // Pre-stat phase (once at start)
    /// let cached = store.pre_stat_and_cache(&objects, 100).await?;
    /// println!("Cached {} object sizes", cached);
    ///
    /// // Download phase (benefits from cached sizes)
    /// for uri in &objects {
    ///     let data = store.get(&uri).await?;  // No stat overhead!
    /// }
    /// # Ok(())
    /// # }
    /// ```
    async fn pre_stat_and_cache(&self, uris: &[String], max_concurrent: usize) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;

        // Populate size cache with results
        self.seed_size_cache(&size_map).await;

        Ok(size_map.len())
    }

    /// Fill the size cache with sizes the caller already learned from `stat` calls
    /// (no request is made); see [`ObjectStore::seed_size_cache`].
    async fn seed_size_cache(&self, sizes: &HashMap<String, u64>) {
        for (uri, size) in sizes.iter() {
            self.size_cache.put(uri.clone(), *size).await;
        }
    }
}

/// Where a per-endpoint store's buffered writer uploads (issue #208): the store's own
/// client and the bucket/key the store parsed from the writer's URI.
struct S3WriterTarget {
    client: std::sync::Arc<aws_sdk_s3::Client>,
    bucket: String,
    key: String,
}

/// Buffered writer for S3 that collects chunks and uses multipart upload
pub struct S3BufferedWriter {
    uri: String,
    /// `Some` for a writer made by a per-endpoint store: the final upload goes to
    /// that store's server.  `None`: the process-wide client, as `uri` names it.
    target: Option<S3WriterTarget>,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Crc32c,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

impl S3BufferedWriter {
    pub fn new(uri: String) -> Self {
        Self::new_with_compression(uri, CompressionConfig::None)
    }

    pub fn new_with_compression(uri: String, compression: CompressionConfig) -> Self {
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => zstd::Encoder::new(Vec::new(), *level).ok(),
        };

        Self {
            uri,
            target: None,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Crc32c::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        }
    }

    /// Upload to a per-endpoint store's server: `client`, with the `bucket`/`key` the
    /// store parsed from this writer's URI (issue #208).
    fn with_target(
        mut self,
        client: std::sync::Arc<aws_sdk_s3::Client>,
        bucket: String,
        key: String,
    ) -> Self {
        self.target = Some(S3WriterTarget {
            client,
            bucket,
            key,
        });
        self
    }

    /// Close the zstd frame (if compressing) and fix `compressed_bytes`.  The frame
    /// header and every block the encoder already emitted during `write_chunk` are in
    /// `buffer`; the encoder's final output is APPENDED to them, never assigned over them
    /// (issue #218).
    fn seal_compression(&mut self) -> Result<()> {
        if let Some(compressor) = self.compressor.take() {
            let tail = compressor.finish()?;
            self.buffer.extend_from_slice(&tail);
        }
        self.compressed_bytes = self.buffer.len() as u64;
        Ok(())
    }
}

#[async_trait]
impl ObjectWriter for S3BufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }

        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;

        // Handle compression
        if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and add to buffer
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            self.buffer.append(compressed);
        } else {
            // No compression - direct to buffer
            self.buffer.extend_from_slice(chunk);
        }

        Ok(())
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        self.seal_compression()?;

        // Append compression extension if needed
        let final_uri = self.compression.stored_name(&self.uri);

        // A writer made by a per-endpoint store uploads with that store's client
        // (issue #208): create, parts, complete and abort all reach its server.
        if let Some(target) = self.target.take() {
            let key = self.compression.stored_name(&target.key);
            return crate::s3_utils::put_object_multipart_with_client(
                &target.client,
                &target.bucket,
                &key,
                Bytes::from(std::mem::take(&mut self.buffer)),
                None,
            )
            .await;
        }

        // Use S3 multipart upload for the buffered data
        s3_put_object_multipart_uri_async(
            &final_uri,
            Bytes::from(std::mem::take(&mut self.buffer)),
            None,
        )
        .await
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compression(&self) -> CompressionConfig {
        self.compression
    }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use current buffer size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.buffer.len() as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }

    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        Ok(())
    }
}

/// The listing prefix `delete_prefix` uses on every object store (#231): a
/// non-empty key that does not end in `/` gets one `/` appended, so the name
/// selects only the objects under it and never longer siblings (`foo` selects
/// `foo/...`, not `foobar` or `foo.bak`).  An empty key (bucket / container
/// root) is unchanged.  The key is treated literally; nothing else is altered.
/// Shared by the S3, Azure and GCS stores so they cannot drift.
fn directory_prefix(key: &str) -> String {
    let mut prefix = key.to_string();
    if !prefix.is_empty() && !prefix.ends_with('/') {
        prefix.push('/');
    }
    prefix
}

// ============================================================================
// Azure adapter
// ============================================================================
/// `(account, container, listing_prefix)` for an Azure `delete_prefix` URI.
/// A pure step (no client, no request) so the listing prefix can be tested.
#[cfg(feature = "backend-azure")]
fn azure_delete_prefix_target(uri_prefix: &str) -> Result<(String, String, String)> {
    let (account, container, key) = parse_azure_uri(uri_prefix)?;
    Ok((account, container, directory_prefix(&key)))
}

/// The key a store's size cache uses for an Azure object: `az://account/container/key`, so the
/// same blob read through `https://account.blob.core.windows.net/container/key` and deleted
/// through `az://...` is one entry.
#[cfg(feature = "backend-azure")]
fn azure_size_key(uri: &str) -> String {
    parse_azure_uri(uri)
        .map(|(account, container, key)| format!("az://{account}/{container}/{key}"))
        .unwrap_or_else(|_| uri.to_string())
}

/// The size-cache key for a GCS object: `gs://bucket/object` for both `gs://` and `gcs://`.
#[cfg(feature = "backend-gcs")]
fn gcs_size_key(uri: &str) -> String {
    parse_gcs_uri(uri)
        .map(|(bucket, object)| format!("gs://{bucket}/{object}"))
        .unwrap_or_else(|_| uri.to_string())
}

/// Run the s3dlio RangeEngine for `object_size` bytes.
///
/// The engine plans its ranges from `object_size` and treats any short or over read as an
/// error. When `guarded` (the size came from a cache and may be stale) the LAST range asks for
/// one byte more than planned, so a blob that has grown makes that range an over read, and one
/// that has shrunk a short read or a range error: the caller sees an error instead of a silently
/// truncated body, at no extra request. An unchanged blob returns exactly the planned bytes.
///
/// An object below `min_split_size` is one request, and the engine holds that request to the
/// same exact length. A CACHED size of 0 is an error too (the engine would return empty without
/// any request): the caller's fresh-size retry settles it, and a size just learned from the
/// service (`guarded == false`) stays a legitimate empty read.
#[cfg(any(feature = "backend-azure", feature = "backend-gcs"))]
async fn range_engine_download<F, Fut>(
    config: RangeEngineConfig,
    object_size: u64,
    guarded: bool,
    get_range: F,
) -> Result<(Bytes, crate::range_engine_generic::RangeDownloadStats)>
where
    F: Fn(u64, u64) -> Fut + Send + Sync + Clone + 'static,
    Fut: std::future::Future<Output = Result<Bytes>> + Send,
{
    // The engine reads a zero size as an empty object WITHOUT a request. That is right for a size
    // just learned from the service, but a cached zero may be stale (the object may have grown),
    // and returning empty would be a silently wrong success. Fail so the caller re-reads with a
    // fresh size.
    if guarded && object_size == 0 {
        bail!("cached object size 0 cannot be verified without a request; refresh the size");
    }
    let engine = RangeEngine::new(config);
    let get_range = move |offset: u64, length: u64| {
        let ask = if guarded && offset.checked_add(length) == Some(object_size) {
            length.saturating_add(1)
        } else {
            length
        };
        get_range(offset, ask)
    };
    engine.download(object_size, get_range, None).await
}

/// Split an Azure URI into `(account, container, key)`.
///
/// - `az://{account}/{container}/{key...}`
/// - `http(s)://{account}.blob.core.windows.net/{container}/{key...}`
///
/// The STRUCTURE (account, container and key, an empty container refused per #224, unsafe
/// account characters refused) is validated by the environment-free
/// [`crate::capabilities::parse_azure_uri_pure`], which this delegates to. This function then adds
/// the endpoint-dependent step: it canonicalises the account according to the configured
/// endpoint (lower-cased for Azure, kept as written for an emulator, length-checked for Azure).
/// The pure parser is structural validation, not full parity with this function.
#[cfg(feature = "backend-azure")]
fn parse_azure_uri(uri: &str) -> Result<(String, String, String)> {
    let (account, container, key) = crate::capabilities::parse_azure_uri_pure(uri)?;
    // Refuse an account that could redirect the request before anything uses it.
    let account = crate::azure_client::canonical_account_name(&account)?;
    Ok((account, container, key))
}

/// `delete_batch` grouping key for Azure URIs: `((account, container), key)`.
#[cfg(feature = "backend-azure")]
fn azure_group_key(uri: &str) -> Result<((String, String), String)> {
    let (account, container, key) = parse_azure_uri(uri)?;
    Ok(((account, container), key))
}

#[cfg(feature = "backend-azure")]
fn az_uri(account: &str, container: &str, key: &str) -> String {
    if key.is_empty() {
        format!("az://{}/{}", account, container)
    } else {
        format!("az://{}/{}/{}", account, container, key)
    }
}

#[cfg(feature = "backend-azure")]
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

/// Configuration for Azure Blob Storage download behavior.
///
/// The default `auto` mode uses sequential transfer below the configured size
/// threshold and Azure SDK managed concurrency above it. Callers can instead
/// select s3dlio's RangeEngine or fully sequential downloads through explicit
/// configuration or the documented Azure environment variables.
#[cfg(feature = "backend-azure")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureDownloadMode {
    Auto,
    AzureSdk,
    S3dlio,
    Sequential,
}

#[cfg(feature = "backend-azure")]
impl std::str::FromStr for AzureDownloadMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "auto" => Ok(Self::Auto),
            "azure-sdk" | "sdk" | "azure" => Ok(Self::AzureSdk),
            "s3dlio" | "range-engine" => Ok(Self::S3dlio),
            "sequential" | "single" => Ok(Self::Sequential),
            _ => Err(format!(
                "invalid Azure download mode '{value}'; expected auto, azure-sdk, s3dlio, or sequential"
            )),
        }
    }
}

#[cfg(feature = "backend-azure")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureConcurrentEngine {
    AzureSdk,
    S3dlio,
}

#[cfg(feature = "backend-azure")]
impl std::str::FromStr for AzureConcurrentEngine {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "azure-sdk" | "sdk" | "azure" => Ok(Self::AzureSdk),
            "s3dlio" | "range-engine" => Ok(Self::S3dlio),
            _ => Err(format!(
                "invalid Azure concurrent engine '{value}'; expected azure-sdk or s3dlio"
            )),
        }
    }
}

#[cfg(feature = "backend-azure")]
fn azure_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

#[cfg(feature = "backend-azure")]
fn azure_env_mib(name: &str, default_bytes: usize) -> usize {
    azure_env_usize(name, default_bytes / (1024 * 1024))
        .checked_mul(1024 * 1024)
        .unwrap_or(default_bytes)
}

#[cfg(feature = "backend-azure")]
#[derive(Debug, Clone)]
pub struct AzureConfig {
    /// Download strategy. Explicit config values override environment-derived defaults.
    pub download_mode: AzureDownloadMode,

    /// Concurrent engine selected by [`AzureDownloadMode::Auto`].
    pub auto_concurrent_engine: AzureConcurrentEngine,

    /// Azure SDK managed-download concurrency.
    pub sdk_download_concurrency: usize,

    /// Azure SDK managed-download partition size in bytes.
    pub sdk_download_part_size: usize,

    /// Enable RangeEngine for concurrent range downloads
    /// Legacy compatibility switch. When true, it selects s3dlio RangeEngine
    /// above `range_engine.min_split_size` regardless of `download_mode`.
    pub enable_range_engine: bool,

    /// RangeEngine configuration
    /// Network-optimized defaults: 32 MiB threshold (v0.9.60+), 32 concurrent ranges, 64 MiB chunks
    pub range_engine: RangeEngineConfig,

    /// Time-to-live for cached object sizes
    /// Default: 60 seconds
    /// Set to 0 to disable caching
    ///
    /// v0.9.10: Added for pre-stat optimization
    pub size_cache_ttl_secs: u64,
}

#[cfg(feature = "backend-azure")]
impl Default for AzureConfig {
    fn default() -> Self {
        // The `.env` policy first (#243, review R9): this public `Default` (and
        // `AzureObjectStore::new`, which uses it) reads the Azure tuning settings before any
        // Azure client exists.
        crate::dotenv::apply();
        let download_mode = std::env::var(ENV_AZURE_DOWNLOAD_MODE)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(AzureDownloadMode::Auto);
        let auto_concurrent_engine = std::env::var(ENV_AZURE_CONCURRENT_ENGINE)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(AzureConcurrentEngine::AzureSdk);
        let range_threshold = azure_env_mib(
            ENV_AZURE_DOWNLOAD_THRESHOLD_MB,
            DEFAULT_RANGE_ENGINE_THRESHOLD as usize,
        ) as u64;

        Self {
            download_mode,
            auto_concurrent_engine,
            sdk_download_concurrency: azure_env_usize(
                ENV_AZURE_DOWNLOAD_CONCURRENCY,
                DEFAULT_AZURE_DOWNLOAD_CONCURRENCY,
            ),
            sdk_download_part_size: azure_env_mib(
                ENV_AZURE_DOWNLOAD_PART_SIZE_MB,
                DEFAULT_AZURE_DOWNLOAD_PART_SIZE,
            ),
            enable_range_engine: false,
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE, // 64 MiB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT, // 32 parallel
                min_split_size: range_threshold,
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS), // 30s
            },
            size_cache_ttl_secs: 60, // 60 second TTL for size cache
        }
    }
}

#[cfg(feature = "backend-azure")]
#[derive(Clone)]
pub struct AzureObjectStore {
    config: AzureConfig,
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
}

#[cfg(feature = "backend-azure")]
impl Default for AzureObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "backend-azure")]
impl AzureObjectStore {
    pub fn new() -> Self {
        let config = AzureConfig::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }

    pub fn with_config(config: AzureConfig) -> Self {
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
        }
    }

    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }

    /// The object size and whether it came from the size cache (and so may be stale).
    async fn get_object_size_with_origin(&self, uri: &str) -> Result<(u64, bool)> {
        if let Some(cached_size) = self.size_cache.get(&azure_size_key(uri)).await {
            return Ok((cached_size, true));
        }
        Ok((self.fresh_object_size(uri).await?, false))
    }

    /// The object size from a `HEAD` now, whatever the cache holds; the cache is updated.
    async fn fresh_object_size(&self, uri: &str) -> Result<u64> {
        let metadata = self.stat(uri).await?;
        self.size_cache
            .put(azure_size_key(uri), metadata.size)
            .await;
        Ok(metadata.size)
    }

    /// Forget the sizes of the keys one `delete_prefix` batch tried to delete.
    async fn forget_prefix_batch(&self, account: &str, container: &str, keys: &[String]) {
        let uris: Vec<String> = keys
            .iter()
            .map(|k| format!("az://{account}/{container}/{k}"))
            .collect();
        self.forget_sizes(uris.iter().map(String::as_str)).await;
    }

    /// Forget the cached sizes of objects that were written or deleted. A miss costs only the
    /// cache's shared read lock.
    async fn forget_sizes<'a>(&self, uris: impl IntoIterator<Item = &'a str>) {
        let keys: Vec<String> = uris.into_iter().map(azure_size_key).collect();
        self.size_cache
            .invalidate_many(keys.iter().map(String::as_str))
            .await;
    }

    /// The body of `delete_batch`, separate so the caller can clear the size cache after it.
    async fn delete_batch_inner(&self, uris: &[String]) -> Result<()> {
        // Issue #179: a batch may span several containers (or accounts); a
        // client addresses exactly one container.  Group by (account,
        // container), validating every URI first, then delete per container.
        let groups = group_uris_by_bucket(uris, azure_group_key)?;
        let mut failures: Vec<GroupFailure> = Vec::new();
        for group in &groups {
            let (account, container) = &group.target;
            let outcome = match group.uris.first().map(|u| Self::client_for_uri(u)) {
                Some(Ok((cli, _acct, _cont, _))) => cli.delete_objects(&group.keys).await,
                Some(Err(e)) => Err(e),
                None => continue, // unreachable: a group always holds >= 1 URI
            };
            if let Err(error) = outcome {
                failures.push(GroupFailure {
                    target: format!("az://{account}/{container}"),
                    uris: group.uris.clone(),
                    error,
                });
            }
        }
        aggregate_failures(uris.len(), failures)
    }

    /// The body of `delete_prefix`, separate so the keys it deletes can be forgotten per batch.
    async fn delete_prefix_inner(&self, uri_prefix: &str) -> Result<()> {
        use futures::stream::StreamExt;
        debug!("AzureObjectStore::delete_prefix prefix='{}'", uri_prefix);

        let (account, container, key_prefix) = azure_delete_prefix_target(uri_prefix)?;
        let cli = AzureBlob::with_default_credential(&account, &container)?;

        // Use streaming list to avoid loading millions of keys into memory
        let mut stream = cli.list_stream(Some(&key_prefix));
        let mut batch = Vec::with_capacity(1000);
        let mut total_deleted = 0;

        while let Some(result) = stream.next().await {
            match result {
                Ok(key) => {
                    batch.push(key);

                    // Delete in batches of 1000 to avoid memory bloat
                    if batch.len() >= 1000 {
                        let outcome = cli.delete_objects(&batch).await;
                        self.forget_prefix_batch(&account, &container, &batch).await;
                        outcome?;
                        total_deleted += batch.len();
                        batch.clear();
                    }
                }
                Err(e) => return Err(e),
            }
        }

        // Delete final partial batch if any
        if !batch.is_empty() {
            let outcome = cli.delete_objects(&batch).await;
            self.forget_prefix_batch(&account, &container, &batch).await;
            outcome?;
            total_deleted += batch.len();
        }

        debug!(
            "Azure delete_prefix deleted {} objects total",
            total_deleted
        );
        Ok(())
    }

    /// Whether a full read needs the blob's size before it starts. The SDK's managed download
    /// learns the length from its first (ranged) response, so the default (`Auto` with the
    /// Azure SDK engine) and an explicit `AzureSdk` mode need no `HEAD`; the s3dlio
    /// RangeEngine plans its ranges from the size, and `Sequential` uses it to choose a
    /// partition size.
    fn full_read_needs_size(&self) -> bool {
        if self.config.enable_range_engine {
            return true;
        }
        match self.config.download_mode {
            AzureDownloadMode::AzureSdk => false,
            AzureDownloadMode::Auto => {
                self.config.auto_concurrent_engine != AzureConcurrentEngine::AzureSdk
            }
            AzureDownloadMode::S3dlio | AzureDownloadMode::Sequential => true,
        }
    }

    fn download_mode_for_size(&self, object_size: u64) -> AzureDownloadMode {
        if self.config.enable_range_engine {
            return if object_size >= self.config.range_engine.min_split_size {
                AzureDownloadMode::S3dlio
            } else {
                AzureDownloadMode::Sequential
            };
        }

        match self.config.download_mode {
            AzureDownloadMode::Auto if object_size < self.config.range_engine.min_split_size => {
                AzureDownloadMode::Sequential
            }
            AzureDownloadMode::Auto => match self.config.auto_concurrent_engine {
                AzureConcurrentEngine::AzureSdk => AzureDownloadMode::AzureSdk,
                AzureConcurrentEngine::S3dlio => AzureDownloadMode::S3dlio,
            },
            mode => mode,
        }
    }

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

    /// Download using RangeEngine for concurrent range requests
    ///
    /// This method uses the generic RangeEngine to split large Azure blobs into
    /// concurrent range requests, significantly improving throughput by hiding
    /// network latency.
    ///
    /// # Performance
    ///
    /// Concurrent ranges hide per-request latency on large blobs. (An earlier version of this
    /// comment quoted percentage speed-ups; they were never measured and are removed.)
    ///
    /// `guarded`: the size came from a cache; see [`range_engine_download`].
    async fn get_with_range_engine(
        &self,
        uri: &str,
        object_size: u64,
        guarded: bool,
    ) -> Result<Bytes> {
        // Create closure that captures uri for get_range calls
        let uri_owned = uri.to_string();
        let self_clone = self.clone();

        let get_range_fn = move |offset: u64, length: u64| {
            let uri = uri_owned.clone();
            let store = self_clone.clone();
            async move { store.get_range(&uri, offset, Some(length)).await }
        };

        let (bytes, stats) = range_engine_download(
            self.config.range_engine.clone(),
            object_size,
            guarded,
            get_range_fn,
        )
        .await?;

        info!(
            "RangeEngine (Azure) downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );

        Ok(bytes)
    }
}

#[async_trait]
#[cfg(feature = "backend-azure")]
impl ObjectStore for AzureObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;

        if !self.full_read_needs_size() {
            // No HEAD: the SDK discovers the length from its first response, and the buffer is
            // sized from that response.
            return cli
                .get_with_sdk_concurrency_sized(
                    &key,
                    Some(self.config.sdk_download_concurrency),
                    Some(self.config.sdk_download_part_size),
                    None,
                )
                .await;
        }

        // The size comes from the cache (v0.9.10) or a HEAD. A cached size can be stale.
        let (object_size, from_cache) = self.get_object_size_with_origin(uri).await?;

        match self.download_mode_for_size(object_size) {
            AzureDownloadMode::S3dlio => {
                // The RangeEngine plans exactly object_size bytes of ranges. A size from the
                // cache is used WITHOUT a HEAD but guarded (the last range asks for one byte
                // more); if the blob changed size the engine errors, and the read is retried
                // once with a size from a fresh HEAD.
                match self
                    .get_with_range_engine(uri, object_size, from_cache)
                    .await
                {
                    Err(e) if from_cache => {
                        debug!("Azure RangeEngine read with cached size {object_size} failed ({e:#}); re-reading with a fresh size");
                        self.forget_sizes([uri]).await;
                        let fresh = self.fresh_object_size(uri).await?;
                        self.get_with_range_engine(uri, fresh, false).await
                    }
                    other => other,
                }
            }
            AzureDownloadMode::AzureSdk => {
                cli.get_with_sdk_concurrency_sized(
                    &key,
                    Some(self.config.sdk_download_concurrency),
                    Some(self.config.sdk_download_part_size),
                    Some(object_size),
                )
                .await
            }
            AzureDownloadMode::Sequential => cli.get_sequential(&key, object_size).await,
            AzureDownloadMode::Auto => unreachable!("auto mode must resolve before download"),
        }
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!(
            "AzureObjectStore::get_range uri='{}', offset={}, length={:?}",
            uri, offset, length
        );
        // audit #152 finding 2.6 (bug B4): same zero-length short-circuit
        // as S3ObjectStore::get_range above — avoids the `offset + len - 1`
        // underflow and the pointless network round-trip.
        if length == Some(0) {
            return Ok(Bytes::new());
        }
        let end = length
            .map(|len| crate::range_engine_generic::range_end_inclusive(offset, len))
            .transpose()?;
        let b = cli.get_range(&key, offset, end).await?; // Bytes - return directly for zero-copy
        Ok(b)
    }

    /// Caller-buffer GET (plan §4.5 A1, A8): the URI goes through the same parser as every
    /// other Azure operation. The row, if any, is the `LoggedObjectStore` wrapper's.
    async fn get_into(&self, uri: &str, dst: &mut [u8], progress: &ReceiveProgress) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!(
            "AzureObjectStore::get_into uri='{}', expected_len={}",
            uri,
            dst.len()
        );
        cli.get_into(&key, dst, progress).await
    }

    /// Typed HEAD: `Ok(None)` only for a typed missing blob (a missing container is an error).
    async fn head(&self, uri: &str) -> Result<Option<HeadMetadata>> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!("AzureObjectStore::head uri='{}'", uri);
        cli.head_typed(&key).await
    }

    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!("AzureObjectStore::put uri='{}', {} bytes", uri, data.len());
        // Bytes→Bytes is zero-copy (reference counted buffer)
        let outcome = cli.put(&key, data, true).await;
        // The cached size (if any) is the old object's: forget it (the next read re-learns it).
        self.forget_sizes([uri]).await;
        outcome
    }

    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!(
            "AzureObjectStore::put_multipart uri='{}', {} bytes, part_size={:?}",
            uri,
            data.len(),
            part_size
        );
        let part = part_size.unwrap_or(crate::constants::DEFAULT_AZURE_MULTIPART_PART_SIZE);
        let max_in_flight = std::env::var("AZURE_MAX_INFLIGHT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(crate::constants::DEFAULT_CONCURRENT_UPLOADS);

        // Stream the Bytes buffer as chunks of size `part` (zero-copy via Bytes::slice)
        let chunks = (0..data.len())
            .step_by(part)
            .map(|offset| {
                let end = (offset + part).min(data.len());
                data.slice(offset..end) // Zero-copy slice (shares same Arc)
            })
            .collect::<Vec<_>>();
        let stream = stream::iter(chunks);

        let outcome = cli
            .upload_multipart_stream(&key, stream, part, max_in_flight)
            .await;
        self.forget_sizes([uri]).await;
        outcome
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (cli, account, container, key_prefix) = Self::client_for_prefix(uri_prefix)?;
        debug!(
            "AzureObjectStore::list prefix='{}', recursive={}",
            uri_prefix, recursive
        );
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
            let base = if key_prefix.ends_with('/') {
                key_prefix.clone()
            } else {
                format!("{}/", key_prefix)
            };
            keys.retain(|k| {
                if let Some(rest) = k.strip_prefix(&base) {
                    !rest.contains('/')
                } else {
                    // if it doesn't start with base, keep only if it's exactly the same path
                    k == &key_prefix
                }
            });
        }

        Ok(keys
            .into_iter()
            .map(|k| az_uri(&account, &container, &k))
            .collect())
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            debug!("AzureObjectStore::list_stream prefix='{}', recursive={}", uri_prefix, recursive);
            let (cli, account, container, key_prefix) = match Self::client_for_prefix(uri_prefix) {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let prefix = if key_prefix.is_empty() { None } else { Some(key_prefix.as_str()) };
            let base = if !key_prefix.is_empty() && !key_prefix.ends_with('/') {
                format!("{}/", key_prefix)
            } else {
                key_prefix.clone()
            };

            // Use streaming list to get blobs page by page
            let mut stream = cli.list_stream(prefix);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(key) => {
                        // For non-recursive, filter to immediate children only
                        if !recursive && !key_prefix.is_empty() {
                            if let Some(rest) = key.strip_prefix(&base) {
                                if rest.contains('/') {
                                    continue; // Skip nested items
                                }
                            } else if key != key_prefix {
                                continue;
                            }
                        }
                        yield Ok(az_uri(&account, &container, &key));
                    }
                    Err(e) => yield Err(e),
                }
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!("AzureObjectStore::stat uri='{}'", uri);
        let p = cli.stat(&key).await?;
        Ok(az_props_to_meta(&p))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        debug!("AzureObjectStore::delete uri='{}'", uri);
        if key.is_empty() {
            bail!("cannot delete '{uri}': it names a container, not an object");
        }
        let outcome = cli.delete_objects(&[key]).await;
        // Whether or not the delete succeeded, the cached size of THIS object is not to be
        // trusted any more (neighbours keep theirs).
        self.forget_sizes([uri]).await;
        outcome
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        if uris.is_empty() {
            return Ok(());
        }
        debug!("AzureObjectStore::delete_batch {} URIs", uris.len());
        let outcome = self.delete_batch_inner(uris).await;
        self.forget_sizes(uris.iter().map(String::as_str)).await;
        outcome
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        self.delete_prefix_inner(uri_prefix).await
    }

    /// No-op: an object store has no directories (issue #229). A prefix exists once
    /// objects are written under it, so there is nothing to create. The URI is
    /// validated; NO request is sent and no object (marker or otherwise) is written.
    async fn mkdir(&self, uri: &str) -> Result<()> {
        parse_azure_uri(uri)?;
        debug!(
            "AzureObjectStore::mkdir is a no-op on an object store: '{}'",
            uri
        );
        Ok(())
    }

    /// No-op: an object store has no directories (issue #229), so there is nothing to
    /// remove -- `recursive` or not. The URI is validated; NO request is sent and
    /// nothing is deleted. To delete the objects under a prefix use
    /// [`ObjectStore::delete_prefix`] (Python `delete(uri, recursive=True)`,
    /// `s3-cli delete -r`).
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let _ = recursive;
        parse_azure_uri(uri)?;
        debug!(
            "AzureObjectStore::rmdir is a no-op on an object store: '{}'",
            uri
        );
        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        // interpret "name" as "{account}/{container}"
        let mut it = name.splitn(2, '/');
        let account = it
            .next()
            .ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let container = it
            .next()
            .ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let cli = AzureBlob::with_default_credential(account, container)?;
        // best-effort create
        let _ = cli.create_container_if_missing().await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let mut it = name.splitn(2, '/');
        let account = it
            .next()
            .ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let container = it
            .next()
            .ok_or_else(|| anyhow!("expected \"account/container\""))?;
        let cli = AzureBlob::with_default_credential(account, container)?;
        let _ = cli.delete_container().await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer that collects chunks then uses put_multipart
        Ok(Box::new(AzureBufferedWriter::new(uri.to_string())))
    }

    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer with compression support
        Ok(Box::new(AzureBufferedWriter::new_with_compression(
            uri.to_string(),
            compression,
        )))
    }

    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        // For Azure, use buffered writer that collects chunks then uses put_multipart
        if let Some(compression) = options.compression {
            Ok(Box::new(AzureBufferedWriter::new_with_compression(
                uri.to_string(),
                compression,
            )))
        } else {
            Ok(Box::new(AzureBufferedWriter::new(uri.to_string())))
        }
    }

    /// Pre-stat objects and populate the size cache
    ///
    /// v0.9.10: Override default implementation to populate internal size cache.
    /// This enables subsequent get() calls to skip redundant stat operations.
    async fn pre_stat_and_cache(&self, uris: &[String], max_concurrent: usize) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;

        // Populate size cache with results
        self.seed_size_cache(&size_map).await;

        Ok(size_map.len())
    }

    /// Fill the size cache with sizes the caller already learned from `stat` calls
    /// (no request is made); see [`ObjectStore::seed_size_cache`].
    async fn seed_size_cache(&self, sizes: &HashMap<String, u64>) {
        for (uri, size) in sizes.iter() {
            self.size_cache.put(azure_size_key(uri), *size).await;
        }
    }
}

/// Buffered writer for Azure that collects chunks and uses multipart upload
#[cfg(feature = "backend-azure")]
pub struct AzureBufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Crc32c,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

#[cfg(feature = "backend-azure")]
impl AzureBufferedWriter {
    pub fn new(uri: String) -> Self {
        Self::new_with_compression(uri, CompressionConfig::None)
    }

    pub fn new_with_compression(uri: String, compression: CompressionConfig) -> Self {
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => zstd::Encoder::new(Vec::new(), *level).ok(),
        };

        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Crc32c::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        }
    }

    /// Close the zstd frame (if compressing) and fix `compressed_bytes`.  The frame
    /// header and every block the encoder already emitted during `write_chunk` are in
    /// `buffer`; the encoder's final output is APPENDED to them, never assigned over them
    /// (issue #218).
    fn seal_compression(&mut self) -> Result<()> {
        if let Some(compressor) = self.compressor.take() {
            let tail = compressor.finish()?;
            self.buffer.extend_from_slice(&tail);
        }
        self.compressed_bytes = self.buffer.len() as u64;
        Ok(())
    }
}

#[async_trait]
#[cfg(feature = "backend-azure")]
impl ObjectWriter for AzureBufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }

        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;

        // Handle compression
        if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and add to buffer
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            self.buffer.append(compressed);
        } else {
            // No compression - direct to buffer
            self.buffer.extend_from_slice(chunk);
        }

        Ok(())
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        self.seal_compression()?;

        // Append compression extension if needed
        let (cli, _acct, _cont, key) = AzureObjectStore::client_for_uri(&self.uri)?;
        let final_key = self.compression.stored_name(&key);

        // Use Azure multipart upload for the buffered data
        cli.put(&final_key, Bytes::from(self.buffer.clone()), true)
            .await
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compression(&self) -> CompressionConfig {
        self.compression
    }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use current buffer size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.buffer.len() as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }

    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        Ok(())
    }
}

// ============================================================================
// Google Cloud Storage (GCS) adapter
// ============================================================================

/// Helper function to create GCS URI from bucket and key
#[cfg(feature = "backend-gcs")]
fn gcs_uri(bucket: &str, key: &str) -> String {
    if key.is_empty() {
        format!("gs://{}", bucket)
    } else {
        format!("gs://{}/{}", bucket, key)
    }
}

/// Convert GCS object metadata to provider-neutral ObjectMetadata
#[cfg(feature = "backend-gcs")]
fn gcs_meta_to_object_meta(meta: &GcsObjectMetadata) -> ObjectMetadata {
    ObjectMetadata {
        size: meta.size,
        last_modified: meta.updated.clone(),
        e_tag: meta.etag.clone(),
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

// ────────────────────────────────────────────────────────────────────────────
// GCS Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Google Cloud Storage backend
///
/// Supports RangeEngine for concurrent range downloads on network storage.
/// RangeEngine is **disabled by default** — enable explicitly for large-file workloads
/// where the benefit outweighs the HEAD request cost. Unlike S3, there is no
/// env-var default for GCS; this config field must be set explicitly.
///
/// v0.9.10: Added size_cache_ttl_secs for pre-stat optimization
#[cfg(feature = "backend-gcs")]
#[derive(Clone, Debug)]
pub struct GcsConfig {
    /// Enable RangeEngine for concurrent range downloads (per-store config)
    /// Default: false — must be set explicitly for GCS large-file workloads
    pub enable_range_engine: bool,

    /// RangeEngine configuration
    /// Network-optimized defaults: 32 MiB threshold (v0.9.60+), 32 concurrent ranges, 64 MiB chunks
    pub range_engine: RangeEngineConfig,

    /// Time-to-live for cached object sizes
    /// Default: 60 seconds
    /// Set to 0 to disable caching
    pub size_cache_ttl_secs: u64,
}

#[cfg(feature = "backend-gcs")]
impl Default for GcsConfig {
    fn default() -> Self {
        Self {
            enable_range_engine: false, // Disabled by default; must be set explicitly for GCS large-file workloads
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE, // 64 MiB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT, // 32 parallel
                min_split_size: DEFAULT_RANGE_ENGINE_THRESHOLD, // 32 MiB threshold (v0.9.60+)
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS), // 30s
            },
            size_cache_ttl_secs: 60, // 60 second TTL for size cache
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GCS ObjectStore Implementation
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "backend-gcs")]
#[derive(Clone)]
pub struct GcsObjectStore {
    config: GcsConfig,
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
    client: Arc<OnceCell<GcsClient>>,
}

#[cfg(feature = "backend-gcs")]
impl Default for GcsObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

/// `(bucket, listing_prefix)` for a GCS `delete_prefix` URI.  A pure step (no
/// client, no request) so the listing prefix can be tested (#231).
#[cfg(feature = "backend-gcs")]
fn gcs_delete_prefix_target(uri_prefix: &str) -> Result<(String, String)> {
    let (bucket, key_prefix) = parse_gcs_uri(uri_prefix).or_else(|_| {
        // Handle bucket-only URIs
        if let Some(rest) = uri_prefix
            .strip_prefix("gs://")
            .or_else(|| uri_prefix.strip_prefix("gcs://"))
        {
            let bucket = rest.trim_end_matches('/').to_string();
            if !bucket.is_empty() {
                return Ok((bucket, String::new()));
            }
        }
        bail!(
            "Invalid GCS URI for delete_prefix operation: {}",
            uri_prefix
        )
    })?;
    Ok((bucket, directory_prefix(&key_prefix)))
}

#[cfg(feature = "backend-gcs")]
impl GcsObjectStore {
    pub fn new() -> Self {
        let config = GcsConfig::default();
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
            client: Arc::new(OnceCell::const_new()),
        }
    }

    pub fn with_config(config: GcsConfig) -> Self {
        let cache_ttl = Duration::from_secs(config.size_cache_ttl_secs);
        Self {
            config,
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(cache_ttl)),
            client: Arc::new(OnceCell::const_new()),
        }
    }

    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }

    /// The object size from a `stat` now, whatever the cache holds; the cache is updated.
    async fn fresh_object_size(&self, uri: &str) -> Result<u64> {
        let metadata = self.stat(uri).await?;
        self.size_cache.put(gcs_size_key(uri), metadata.size).await;
        Ok(metadata.size)
    }

    /// Forget the cached sizes of objects that were written or deleted. A miss costs only the
    /// cache's shared read lock.
    async fn forget_sizes<'a>(&self, uris: impl IntoIterator<Item = &'a str>) {
        let keys: Vec<String> = uris.into_iter().map(gcs_size_key).collect();
        self.size_cache
            .invalidate_many(keys.iter().map(String::as_str))
            .await;
    }

    async fn get_client(&self) -> Result<GcsClient> {
        let client = self
            .client
            .get_or_try_init(|| async { GcsClient::new().await })
            .await?;
        Ok(client.clone())
    }

    /// Download using RangeEngine for concurrent range requests
    ///
    /// This method uses the generic RangeEngine to split large GCS objects into
    /// concurrent range requests, hiding per-request latency. (An earlier version of this
    /// comment quoted percentage speed-ups; they were never measured and are removed.)
    ///
    /// `guarded`: the size came from a cache; see [`range_engine_download`].
    async fn get_with_range_engine(
        &self,
        uri: &str,
        object_size: u64,
        guarded: bool,
    ) -> Result<Bytes> {
        // Create closure that captures uri for get_range calls
        let uri_owned = uri.to_string();
        let self_clone = self.clone();

        let get_range_fn = move |offset: u64, length: u64| {
            let uri = uri_owned.clone();
            let store = self_clone.clone();
            async move { store.get_range(&uri, offset, Some(length)).await }
        };

        let (bytes, stats) = range_engine_download(
            self.config.range_engine.clone(),
            object_size,
            guarded,
            get_range_fn,
        )
        .await?;

        info!(
            "RangeEngine (GCS) downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );

        Ok(bytes)
    }
}

#[async_trait]
#[cfg(feature = "backend-gcs")]
impl ObjectStore for GcsObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        let client = self.get_client().await?;

        // Check if RangeEngine is enabled and object is large enough
        if !self.config.enable_range_engine {
            trace!("RangeEngine disabled for GCS, using simple download");
            return client.get_object(&bucket, &object).await;
        }

        // Get object size from cache or stat (v0.9.10: cache optimization). A cached size can
        // be stale.
        let cached = self.size_cache.get(&gcs_size_key(uri)).await;
        let object_size = match cached {
            Some(size) => size,
            None => self.fresh_object_size(uri).await?,
        };

        // Use RangeEngine for large objects (default 4MB+)
        if object_size >= self.config.range_engine.min_split_size {
            debug!(
                "GCS object size {} >= threshold {}, using RangeEngine for {}",
                object_size, self.config.range_engine.min_split_size, uri
            );
            // A cached size is used WITHOUT a HEAD but guarded (the last range asks for one byte
            // more); if the object changed size the engine errors and the read is retried once
            // with a size from a fresh stat.
            return match self
                .get_with_range_engine(uri, object_size, cached.is_some())
                .await
            {
                Err(e) if cached.is_some() => {
                    debug!("GCS RangeEngine read with cached size {object_size} failed ({e:#}); re-reading with a fresh size");
                    self.forget_sizes([uri]).await;
                    let fresh = self.fresh_object_size(uri).await?;
                    self.get_with_range_engine(uri, fresh, false).await
                }
                other => other,
            };
        }

        // Simple sequential download for small objects
        debug!(
            "GCS object size {} < threshold {}, using simple download for {}",
            object_size, self.config.range_engine.min_split_size, uri
        );

        client.get_object(&bucket, &object).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        debug!(
            "GcsObjectStore::get_range uri='{}', offset={}, length={:?}",
            uri, offset, length
        );
        let client = self.get_client().await?;
        client
            .get_object_range(&bucket, &object, offset, length)
            .await
    }

    /// Caller-buffer GET (plan §4.5 A1, A8), official GCS client only: the community client
    /// keeps the `Unsupported` trait default. The row, if any, is the wrapper's.
    #[cfg(not(feature = "gcs-community"))]
    async fn get_into(&self, uri: &str, dst: &mut [u8], progress: &ReceiveProgress) -> Result<()> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        debug!(
            "GcsObjectStore::get_into uri='{}', expected_len={}",
            uri,
            dst.len()
        );
        let client = self.get_client().await?;
        client
            .get_object_into(&bucket, &object, dst, progress)
            .await
    }

    /// Typed HEAD, official GCS client only. GCS cannot tell an absent bucket from an absent
    /// object, so both are `Ok(None)`.
    #[cfg(not(feature = "gcs-community"))]
    async fn head(&self, uri: &str) -> Result<Option<HeadMetadata>> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        debug!("GcsObjectStore::head uri='{}'", uri);
        let client = self.get_client().await?;
        client.head_typed(&bucket, &object).await
    }

    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        debug!("GcsObjectStore::put uri='{}', {} bytes", uri, data.len());
        let client = self.get_client().await?;
        let outcome = client.put_object(&bucket, &object, data).await;
        // The cached size (if any) is the old object's: forget it (the next read re-learns it).
        self.forget_sizes([uri]).await;
        outcome
    }

    async fn put_multipart(&self, uri: &str, data: Bytes, _part_size: Option<usize>) -> Result<()> {
        // audit #157 bug 5.4 (D5): the community GCS client has no real
        // chunked/resumable upload implementation (see put_object's doc
        // comment in gcs_client.rs) — `_part_size` has no effect and is
        // accepted only to satisfy the ObjectStore trait's shared
        // signature with the S3/Azure backends, which do support it.
        let (bucket, object) = parse_gcs_uri(uri)?;
        debug!(
            "GcsObjectStore::put_multipart uri='{}', {} bytes (community GCS client has no \
             chunked upload; routed to simple upload)",
            uri,
            data.len()
        );
        let client = self.get_client().await?;
        let outcome = client.put_object(&bucket, &object, data).await;
        self.forget_sizes([uri]).await;
        outcome
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        // parse_gcs_uri now handles bucket-only URIs (gs://bucket/ → ("bucket", ""))
        let (bucket, key_prefix) = parse_gcs_uri(uri_prefix)?;
        debug!(
            "GcsObjectStore::list prefix='{}', recursive={}",
            uri_prefix, recursive
        );

        let client = self.get_client().await?;
        let prefix = if key_prefix.is_empty() {
            None
        } else {
            Some(key_prefix.as_str())
        };
        let keys = client.list_objects(&bucket, prefix, recursive).await?;

        // Convert keys to full URIs
        Ok(keys.into_iter().map(|k| gcs_uri(&bucket, &k)).collect())
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let (bucket, key_prefix) = match parse_gcs_uri(uri_prefix) {
                Ok((b, k)) => (b, k),
                Err(e) => { yield Err(e); return; }
            };

            let client = match self.get_client().await {
                Ok(c) => c,
                Err(e) => { yield Err(e); return; }
            };

            // Route through the true streaming lister — yields keys as GCS pages
            // arrive, so downloads begin in parallel with the listing rather than
            // waiting for all 50 K+ entries to be collected first.
            let prefix = if key_prefix.is_empty() { None } else { Some(key_prefix.as_str()) };
            let mut stream = client.list_objects_stream(&bucket, prefix, recursive);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(key) => yield Ok(gcs_uri(&bucket, &key)),
                    Err(e) => { yield Err(e); return; }
                }
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let (bucket, object) = parse_gcs_uri(uri)?;
        debug!("GcsObjectStore::stat uri='{}'", uri);
        let client = self.get_client().await?;
        let meta = client.stat_object(&bucket, &object).await?;
        Ok(gcs_meta_to_object_meta(&meta))
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let outcome: Result<()> = async {
            let (bucket, object) = parse_gcs_uri(uri)?;
            debug!("GcsObjectStore::delete uri='{}'", uri);
            if object.is_empty() {
                bail!("cannot delete '{uri}': it names a bucket, not an object");
            }
            let client = self.get_client().await?;
            client.delete_object(&bucket, &object).await
        }
        .await;
        // Whether or not the delete succeeded, the cached size of THIS object is not to be
        // trusted any more (neighbours keep theirs).
        self.forget_sizes([uri]).await;
        outcome
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        let outcome: Result<()> = async {
            if uris.is_empty() {
                return Ok(());
            }
            debug!("GcsObjectStore::delete_batch {} URIs", uris.len());

            // Issue #179: a batch may span several buckets; delete_objects
            // addresses exactly one.  Group by bucket, validating every URI first
            // (a batch we cannot fully interpret deletes nothing), then delete per
            // bucket.  A single-bucket batch is one group, as before.
            let groups = group_uris_by_bucket(uris, parse_gcs_uri)?;
            let client = self.get_client().await?;
            let mut failures: Vec<GroupFailure> = Vec::new();
            for group in &groups {
                if let Err(error) = client
                    .delete_objects(&group.target, group.keys.clone())
                    .await
                {
                    failures.push(GroupFailure {
                        target: format!("gs://{}", group.target),
                        uris: group.uris.clone(),
                        error,
                    });
                }
            }
            aggregate_failures(uris.len(), failures)
        }
        .await;
        self.forget_sizes(uris.iter().map(String::as_str)).await;
        outcome
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let outcome: Result<()> = async {
            debug!("GcsObjectStore::delete_prefix prefix='{}'", uri_prefix);
            let (bucket, key_prefix) = gcs_delete_prefix_target(uri_prefix)?;

            let client = self.get_client().await?;

            // Use streaming list to avoid loading millions of keys into memory
            // List all objects (returns Vec, not a stream)
            let keys = client
                .list_objects(&bucket, Some(&key_prefix), true)
                .await?;
            let total_deleted = keys.len();

            // Delete in batches of 1000 to avoid overloading the API
            for chunk in keys.chunks(1000) {
                let outcome = client.delete_objects(&bucket, chunk.to_vec()).await;
                let uris: Vec<String> =
                    chunk.iter().map(|k| format!("gs://{bucket}/{k}")).collect();
                self.forget_sizes(uris.iter().map(String::as_str)).await;
                outcome?;
            }

            debug!("GCS delete_prefix deleted {} objects total", total_deleted);
            Ok(())
        }
        .await;
        outcome
    }

    /// No-op: an object store has no directories (issue #229). A prefix exists once
    /// objects are written under it, so there is nothing to create. The URI is
    /// validated; NO request is sent and no object (marker or otherwise) is written.
    async fn mkdir(&self, uri: &str) -> Result<()> {
        parse_gcs_uri(uri)?;
        debug!(
            "GcsObjectStore::mkdir is a no-op on an object store: '{}'",
            uri
        );
        Ok(())
    }

    /// No-op: an object store has no directories (issue #229), so there is nothing to
    /// remove -- `recursive` or not. The URI is validated; NO request is sent and
    /// nothing is deleted. To delete the objects under a prefix use
    /// [`ObjectStore::delete_prefix`] (Python `delete(uri, recursive=True)`,
    /// `s3-cli delete -r`).
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let _ = recursive;
        parse_gcs_uri(uri)?;
        debug!(
            "GcsObjectStore::rmdir is a no-op on an object store: '{}'",
            uri
        );
        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        // For GCS, the "container" is a bucket
        let client = self.get_client().await?;
        client.create_bucket(name, None).await
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        // For GCS, the "container" is a bucket
        let client = self.get_client().await?;
        client.delete_bucket(name).await
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // For GCS, use buffered writer
        Ok(Box::new(GcsBufferedWriter::new(uri.to_string())))
    }

    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        // For GCS, use buffered writer with compression support
        Ok(Box::new(GcsBufferedWriter::new_with_compression(
            uri.to_string(),
            compression,
        )))
    }

    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        // For GCS, use buffered writer
        if let Some(compression) = options.compression {
            Ok(Box::new(GcsBufferedWriter::new_with_compression(
                uri.to_string(),
                compression,
            )))
        } else {
            Ok(Box::new(GcsBufferedWriter::new(uri.to_string())))
        }
    }

    /// Pre-stat objects and populate the size cache
    ///
    /// v0.9.10: Override default implementation to populate internal size cache.
    /// This enables subsequent get() calls to skip redundant stat operations.
    ///
    /// Critical for benchmarking workloads that download many GCS objects,
    /// eliminating per-object HEAD request latency (typically 10-50ms each).
    async fn pre_stat_and_cache(&self, uris: &[String], max_concurrent: usize) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;

        // Populate size cache with results
        self.seed_size_cache(&size_map).await;

        Ok(size_map.len())
    }

    /// Fill the size cache with sizes the caller already learned from `stat` calls
    /// (no request is made); see [`ObjectStore::seed_size_cache`].
    async fn seed_size_cache(&self, sizes: &HashMap<String, u64>) {
        for (uri, size) in sizes.iter() {
            self.size_cache.put(gcs_size_key(uri), *size).await;
        }
    }
}

/// Buffered writer for GCS that collects chunks and uses multipart upload
#[cfg(feature = "backend-gcs")]
pub struct GcsBufferedWriter {
    uri: String,
    buffer: Vec<u8>,
    bytes_written: u64,
    finalized: bool,
    hasher: Crc32c,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

#[cfg(feature = "backend-gcs")]
impl GcsBufferedWriter {
    pub fn new(uri: String) -> Self {
        Self::new_with_compression(uri, CompressionConfig::None)
    }

    pub fn new_with_compression(uri: String, compression: CompressionConfig) -> Self {
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => zstd::Encoder::new(Vec::new(), *level).ok(),
        };

        Self {
            uri,
            buffer: Vec::new(),
            bytes_written: 0,
            finalized: false,
            hasher: Crc32c::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        }
    }

    /// Close the zstd frame (if compressing) and fix `compressed_bytes`.  The frame
    /// header and every block the encoder already emitted during `write_chunk` are in
    /// `buffer`; the encoder's final output is APPENDED to them, never assigned over them
    /// (issue #218).
    fn seal_compression(&mut self) -> Result<()> {
        if let Some(compressor) = self.compressor.take() {
            let tail = compressor.finish()?;
            self.buffer.extend_from_slice(&tail);
        }
        self.compressed_bytes = self.buffer.len() as u64;
        Ok(())
    }
}

#[async_trait]
#[cfg(feature = "backend-gcs")]
impl ObjectWriter for GcsBufferedWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }

        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;

        // Handle compression
        if let Some(ref mut compressor) = self.compressor {
            // Compress the chunk and add to buffer
            compressor.write_all(chunk)?;
            let compressed = compressor.get_mut();
            self.buffer.append(compressed);
        } else {
            // No compression - direct to buffer
            self.buffer.extend_from_slice(chunk);
        }

        Ok(())
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        self.seal_compression()?;

        // Append compression extension if needed
        let final_uri = self.compression.stored_name(&self.uri);

        // Parse URI and upload via GCS client
        let (bucket, object) = parse_gcs_uri(&final_uri)?;
        let client = get_global_gcs_client().await?;

        // audit #157 bug 5.4 (D5): the community GCS client has no real
        // chunked/resumable upload — both branches used to call
        // functions with byte-for-byte identical bodies under different
        // names. `Bytes::from(Vec<u8>)` transfers ownership without
        // copying — zero-cost. `std::mem::take` moves the buffer out so
        // the encoder can be dropped too.
        client
            .put_object(
                &bucket,
                &object,
                Bytes::from(std::mem::take(&mut self.buffer)),
            )
            .await
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compression(&self) -> CompressionConfig {
        self.compression
    }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use current buffer size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.buffer.len() as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }

    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.buffer.clear();
        if let Some(compressor) = self.compressor {
            drop(compressor); // Clean up compressor
        }
        Ok(())
    }
}

// ============================================================================
// Convenience factory that picks a backend from a URI
// ============================================================================
//
// NOTE — Why store_for_uri() is synchronous and why that is intentional
// -----------------------------------------------------------------------
// This function and its siblings (store_for_uri_with_logger,
// store_for_uri_with_config, store_for_uri_with_config_and_logger) are
// deliberately synchronous (non-async).
//
// All backends that it dispatches to are constructed synchronously:
//   • file://   → FileSystemObjectStore::boxed()                   — sync ✓
//   • direct:// → ConfigurableFileSystemObjectStore::boxed_direct_io() — sync ✓
//   • s3://     → S3ObjectStore  (see below)                       — sync ✓
//   • az://     → AzureObjectStore::boxed()                        — sync ✓
//   • gs://     → GcsObjectStore::boxed()                          — sync ✓
//
// The S3 global-singleton client is lazily initialised on the *first request*
// (via OnceCell inside aws_s3_client_async), not at construction time.  So
// store_for_uri("s3://...") makes no network calls and needs no await.
//
// PER-ENDPOINT S3 STORES
// ----------------------
// When the URI contains an explicit host:port (e.g. s3://10.9.0.17:80/...),
// s3_endpoint_url_from_uri() extracts the HTTP endpoint URL and
// store_for_uri_with_logger() calls S3ObjectStore::for_endpoint() via
// run_on_global_rt() to create a **dedicated** AWS SDK client with its own
// connection pool.  This client ignores AWS_ENDPOINT_URL entirely.
//
// Plain bucket URIs (s3://my-bucket/prefix/) have no port and continue to
// use the global singleton client (which reads AWS_ENDPOINT_URL).  This is
// correct for single-endpoint and AWS S3 deployments.

/// Extract a per-endpoint HTTP URL from an S3 URI whose first segment names a
/// server (an explicit `host:port` / `[ipv6]:port`).
///
/// - `s3://10.9.0.17:80/bucket/` → `Ok(Some("http://10.9.0.17:80"))`
/// - `s3://my-bucket/prefix/`, `s3://data.example.com/key` → `Ok(None)` (a bucket;
///   issues #178 / #193)
/// - `s3://10.0.0.5/bkt/key` (portless IP) → `Err` telling the caller to add `:port`
///
/// The server-vs-bucket decision is made by
/// [`crate::s3_utils::classify_s3_first_segment`], the same rule the URI parser,
/// `MultiEndpointStore` and the CLI use.
///
/// Used by [`store_for_uri_with_logger`] to create per-endpoint S3 clients
/// rather than falling back to the global singleton (which honours
/// `AWS_ENDPOINT_URL` and would route all traffic to one host).
pub(crate) fn s3_endpoint_url_from_uri(uri: &str) -> Result<Option<String>> {
    let Some(after_scheme) = uri.strip_prefix("s3://") else {
        return Ok(None);
    };
    // The first segment is everything before the first '/'
    let first_segment = after_scheme.split('/').next().unwrap_or("");
    match crate::s3_utils::classify_s3_first_segment(first_segment)
        .with_context(|| format!("invalid S3 URI '{uri}'"))?
    {
        crate::s3_utils::S3FirstSegment::Server(authority) => {
            s3_endpoint_url_from_authority(authority)
                .map(Some)
                .with_context(|| format!("invalid S3 URI '{uri}'"))
        }
        crate::s3_utils::S3FirstSegment::Bucket(_) => Ok(None),
    }
}

/// Build the endpoint URL (`https://host:port` or `http://host:port`) for a
/// URI authority of the form `host:port` / `[ipv6]:port`.
///
/// This is the single place that decides the scheme of a host-form S3
/// endpoint. Both the single-endpoint path ([`s3_endpoint_url_from_uri`]) and
/// `MultiEndpointStore` use it, so they can never disagree (issue #180).
///
/// Errors (never `None`/a silent fallback) when `authority` is not a server per
/// [`crate::s3_utils::classify_s3_first_segment`]: the classifier's own error is
/// propagated, and a bucket-shaped authority yields a "not a server" error.
pub(crate) fn s3_endpoint_url_from_authority(authority: &str) -> Result<String> {
    let authority = match crate::s3_utils::classify_s3_first_segment(authority)? {
        crate::s3_utils::S3FirstSegment::Server(authority) => authority,
        crate::s3_utils::S3FirstSegment::Bucket(name) => bail!(
            "'{name}' is not a server: a server is host:port or [ipv6]:port with an explicit port"
        ),
    };
    // `authority` is `host:port` or `[ipv6]:port` with a valid port.
    let port_str = authority
        .rsplit_once(':')
        .map(|(_, port)| port)
        .ok_or_else(|| anyhow::anyhow!("no port in server authority '{authority}'"))?;

    // Determine HTTP scheme from port: 443 → https, everything else → http,
    // unless the operator has named this port as TLS-terminated via
    // S3DLIO_S3_ENDPOINT_TLS_PORTS (audit #154 bug 4.3 / E2 -- TLS-terminated
    // S3-compatible endpoints on non-standard ports, e.g. a load balancer
    // terminating TLS on :9001, otherwise had no way to get an https:// URL
    // out of this heuristic short of literally using port 443).
    let port: u16 = port_str
        .parse()
        .with_context(|| format!("invalid port '{port_str}' in server authority '{authority}'"))?;
    let http_scheme = if port == 443 || s3_endpoint_tls_ports_from_env().contains(&port) {
        "https"
    } else {
        "http"
    };
    Ok(format!("{http_scheme}://{authority}"))
}

/// Additional ports to treat as TLS-terminated for [`s3_endpoint_url_from_uri`],
/// beyond the always-TLS default of 443. Comma-separated list of u16 ports in
/// `S3DLIO_S3_ENDPOINT_TLS_PORTS`; invalid tokens are ignored.
fn s3_endpoint_tls_ports_from_env() -> std::collections::HashSet<u16> {
    // The `.env` policy first (#243, review R9): this decides http versus https for a server-named URI, and is reached from
    // public constructors (`MultiEndpointStore::new`/`from_config`) before any client exists.
    crate::dotenv::apply();
    std::env::var("S3DLIO_S3_ENDPOINT_TLS_PORTS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|tok| tok.trim().parse::<u16>().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// The per-endpoint routing decision the `store_for_uri*` factories make for
/// `uri`: `Some(endpoint_url)` when they build a dedicated client for the
/// server the URI names, `None` when they use the process-global client (or a
/// non-S3 backend).
///
/// This is the single source of that decision.  The factories call it to pick
/// the client, and any cache of stores built by the factories (the Python
/// `STORE_CACHE`, issue #171) must key on its result, so that the cache and the
/// factories cannot disagree about which server a URI reaches.
///
/// A URI whose first segment is an invalid server (for example a portless IP,
/// issue #178) is an `Err`, never `Ok(None)`: callers must propagate it before
/// consulting or populating any cache, so a bad URI can never be served by a
/// store cached for a valid one.
pub(crate) fn store_routing_endpoint_for_uri(uri: &str) -> Result<Option<String>> {
    match infer_scheme(uri) {
        Scheme::S3 => s3_endpoint_url_from_uri(uri),
        _ => Ok(None),
    }
}

/// A store with its own client for the server at `endpoint_url` (dedicated
/// connection pool, ignores `AWS_ENDPOINT_URL`).
fn s3_per_endpoint_store(endpoint_url: String) -> Result<Box<dyn ObjectStore>> {
    crate::s3_client::run_on_global_rt(async move {
        S3ObjectStore::for_endpoint(&endpoint_url)
            .await
            .map(|s| Box::new(s) as Box<dyn ObjectStore>)
    })
}

/// The S3 store for `uri`: a per-endpoint store when the URI names a server
/// (`s3://host:port/...`), the process-wide-client store otherwise.  Used by
/// the factories that return a store for a URI, so none of them can hand back
/// the global store for a URI naming another server (issue #197).
fn s3_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match store_routing_endpoint_for_uri(uri)? {
        Some(endpoint_url) => s3_per_endpoint_store(endpoint_url),
        None => Ok(S3ObjectStore::boxed()),
    }
}

/// Check that every S3 URI in `uris` can be served by the ONE store built for
/// `uris[0]` (as the data loaders do): each must name no server or the server
/// that store talks to.  The first URI naming any other server is an error, so
/// a loader built over mixed servers fails when it is built, before any request.
pub(crate) fn ensure_s3_uris_share_first_uris_server(uris: &[String]) -> Result<()> {
    let Some(first) = uris.first() else {
        return Ok(());
    };
    if infer_scheme(first) != Scheme::S3 {
        return Ok(());
    }
    let server = match store_routing_endpoint_for_uri(first)? {
        Some(url) => Some(StoreServer::from_endpoint_url(&url)?),
        None => global_client_server(),
    };
    for uri in uris.iter().filter(|u| infer_scheme(u) == Scheme::S3) {
        parse_s3_uri_resolving_server(uri, || Ok(server.clone())).with_context(|| {
            format!(
                "S3 URI '{uri}' cannot be served by the store built for '{first}' (one store \
                 serves all URIs of this loader)"
            )
        })?;
    }
    Ok(())
}

/// Serialises every test that reads or writes `S3DLIO_S3_ENDPOINT_TLS_PORTS`
/// (shared with the multi-endpoint tests, which decide URL schemes the same way).
#[cfg(test)]
pub(crate) static ENDPOINT_TLS_PORTS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    store_for_uri_with_logger(uri, None)
}

/// Read many objects concurrently, each through the store `store_for` returns for its URI,
/// and return `(uri, bytes)` pairs in input order. At most `max_in_flight` reads run at once.
///
/// Used by the Python `get_many` for `az://` and `gs://`. Every store's `get` takes the
/// FULL object URI (`AzureObjectStore::get` and `GcsObjectStore::get` parse the account /
/// container / bucket out of it), so each read is handed the URI exactly as the caller wrote
/// it. Issue #242: `get_many` used to strip the scheme and container and pass a bare key,
/// which those stores cannot parse, so every key failed.
///
/// Error semantics are those of the S3 reader (`get_objects_parallel`, #148 3.2): the call
/// does not return while any task it spawned is still running, and it returns the first
/// error observed, as `GET <uri>: <cause>` (the cause chain is kept; print it with `{:#}`).
/// One difference: once a read has failed, reads that have not yet started are skipped
/// instead of downloaded, so a failed batch stops spending bandwidth after the reads already
/// in flight finish. Dropping the returned future aborts every task (they live in a `JoinSet`).
///
/// `store_for` runs on whichever runtime worker polls the task, so it must not call
/// `run_on_global_rt` (it would block a global-runtime worker on a task for that runtime;
/// issue #173). Return cached stores; building one must be synchronous and runtime-free.
#[cfg(any(test, feature = "extension-module"))]
pub(crate) async fn get_many_with_stores<F>(
    uris: Vec<String>,
    max_in_flight: usize,
    store_for: F,
) -> Result<Vec<(String, Bytes)>>
where
    F: Fn(&str) -> Result<std::sync::Arc<dyn ObjectStore>> + Send + Sync + 'static,
{
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use tokio::task::JoinSet;

    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    // Refuse the whole batch before any read starts if a URI names no object.
    for uri in &uris {
        require_object_key(uri)?;
    }
    let store_for = Arc::new(store_for);
    let sem = Arc::new(Semaphore::new(max_in_flight));
    let failed = Arc::new(AtomicBool::new(false));
    let mut tasks: JoinSet<Result<Option<(usize, String, Bytes)>>> = JoinSet::new();
    // Which URI each task reads, so a task that panics is still reported by URI.
    let mut task_uris: std::collections::HashMap<tokio::task::Id, String> =
        std::collections::HashMap::new();

    for (index, uri) in uris.into_iter().enumerate() {
        let sem = Arc::clone(&sem);
        let store_for = Arc::clone(&store_for);
        let failed = Arc::clone(&failed);
        let task_uri = uri.clone();
        let handle = tasks.spawn(async move {
            let _permit = sem
                .acquire_owned()
                .await
                .map_err(|e| anyhow!("get_many semaphore closed: {e}"))?;
            if failed.load(Ordering::Acquire) {
                return Ok(None); // an earlier read failed: do not start this one
            }
            let read = async {
                let store = store_for(&uri)?;
                store.get(&uri).await
            }
            .await;
            match read {
                Ok(bytes) => Ok(Some((index, uri, bytes))),
                Err(e) => {
                    failed.store(true, Ordering::Release);
                    Err(e.context(format!("GET {uri}")))
                }
            }
        });
        task_uris.insert(handle.id(), task_uri);
    }

    let mut out = Vec::new();
    let mut first_err: Option<anyhow::Error> = None;
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok(Ok(Some(item))) => out.push(item),
            Ok(Ok(None)) => {}
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            Err(join_err) => {
                failed.store(true, Ordering::Release);
                let uri = task_uris
                    .get(&join_err.id())
                    .map(String::as_str)
                    .unwrap_or("?");
                first_err.get_or_insert(anyhow!("GET {uri}: the read task failed: {join_err}"));
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }

    // Maintain input order
    out.sort_by_key(|(index, _, _)| *index);
    Ok(out
        .into_iter()
        .map(|(_, uri, bytes)| (uri, bytes))
        .collect())
}

/// Refuse an `az://`, `gs://`/`gcs://` or Azure `https://` URI that names no object: the
/// account/container (or bucket) is present but the key is empty (`az://acct/container`,
/// `az://acct/container/`, `gs://bucket/`). Keys that merely look odd (`dir/`, `//k`) are keys.
#[cfg(any(test, feature = "extension-module"))]
fn require_object_key(uri: &str) -> Result<()> {
    let Some((scheme, rest)) = uri.split_once("://") else {
        return Ok(()); // not a URL: left to the store's own parser
    };
    // Path segments before the key: account + container (az://, Azure https URLs have
    // host + container), or the bucket (gs://).
    let leading = match scheme {
        "gs" | "gcs" => 1,
        "az" | "http" | "https" => 2,
        _ => return Ok(()),
    };
    let key = rest.splitn(leading + 1, '/').nth(leading).unwrap_or("");
    if key.is_empty() {
        bail!("Cannot GET: no key specified in URI: {uri}");
    }
    Ok(())
}

/// [`store_for_uri`] for the library entry points that always built an unlogged
/// store and relied on the S3 backend's own row (checkpoint save/load, the
/// concurrent shard reader, `MultiBackendDataset`, Python
/// `load_checkpoint_with_validation`).
///
/// Identical to `store_for_uri` except for ONE case: a bucket-form S3 URI (the
/// process-wide client) gets a store whose `get`/`put` write the S3 helper row
/// (`S3Ops` shape) when the op-log is active, exactly the row they wrote before
/// W2 (issue #169). Every other URI (host-form S3, `file://`, `direct://`,
/// `az://`, `gs://`) gets the plain unlogged store and writes no row, as before.
pub(crate) fn store_for_uri_with_helper_rows(uri: &str) -> Result<Box<dyn ObjectStore>> {
    if infer_scheme(uri) == Scheme::S3 && store_routing_endpoint_for_uri(uri)?.is_none() {
        return Ok(Box::new(S3ObjectStore::new().with_helper_rows()));
    }
    store_for_uri(uri)
}

/// The URI prefix a bare container name is joined to when a logged store records
/// `create_container` / `delete_container`, so the row has the endpoint and file
/// the same store's other rows have.
///
/// `scheme://` for every backend (`s3://`, `file://`, `az://`, ...), and
/// `s3://host:port/` for an S3 host-form store, whose other rows are
/// `s3://host:port` + `host:port/bucket/key`.
pub(crate) fn container_uri_prefix(uri: &str) -> String {
    let scheme_end = uri.find("://").map(|i| i + 3).unwrap_or(0);
    // An invalid first segment is an Err here; the factory has already propagated it
    // (store_routing_endpoint_for_uri) before it asks for the prefix, so it is not
    // a host-form URI as far as the prefix is concerned.
    if matches!(s3_endpoint_url_from_uri(uri), Ok(Some(_))) {
        // Authority = everything up to the first '/'; keep it, then the '/'.
        let authority = uri[scheme_end..].split('/').next().unwrap_or_default();
        return format!("{}{authority}/", &uri[..scheme_end]);
    }
    uri[..scheme_end].to_string()
}

/// Create a store for the given URI with optional operation logging.
///
/// When a logger is provided, all ObjectStore operations will be traced to the op-log.
/// This enables performance analysis and debugging for all backends (file://, s3://, az://, direct://).
///
/// # Arguments
/// * `uri` - Storage URI (e.g., "s3://bucket/", "file:///path/", "az://account/container/")
/// * `logger` - Optional Logger instance for operation tracing
///
/// # Example
/// ```rust,no_run
/// use s3dlio::{store_for_uri_with_logger, init_op_logger, global_logger};
///
/// # async fn example() -> anyhow::Result<()> {
/// // Initialize op-log
/// init_op_logger("trace.tsv.zst")?;
/// let logger = global_logger();
///
/// // Create store with logging
/// let store = store_for_uri_with_logger("file:///tmp/data", logger)?;
///
/// // Operations are now traced
/// let data = store.get("file:///tmp/data/test.txt").await?;
/// # Ok(())
/// # }
/// ```
pub fn store_for_uri_with_logger(
    uri: &str,
    logger: Option<crate::s3_logger::Logger>,
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;

    // The `.env` policy applies before any store reads a setting (#243).
    crate::dotenv::apply();

    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => FileSystemObjectStore::boxed(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => {
            if let Some(endpoint_url) = store_routing_endpoint_for_uri(uri)? {
                // Per-endpoint store: dedicated connection pool, ignores AWS_ENDPOINT_URL.
                crate::s3_client::run_on_global_rt(async move {
                    S3ObjectStore::for_endpoint(&endpoint_url)
                        .await
                        .map(|s| Box::new(s) as Box<dyn ObjectStore>)
                })?
            } else {
                S3ObjectStore::boxed()
            }
        }
        Scheme::Azure => {
            #[cfg(feature = "backend-azure")]
            {
                AzureObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-azure"))]
            {
                bail!("Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'.")
            }
        }
        Scheme::Gcs => {
            #[cfg(feature = "backend-gcs")]
            {
                GcsObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-gcs"))]
            {
                bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
            }
        }
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };

    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::with_container_prefix(
            Arc::from(store),
            logger,
            container_uri_prefix(uri),
        )))
    } else {
        Ok(store)
    }
}

/// Enhanced factory that supports configuration options for file I/O
pub fn store_for_uri_with_config(
    uri: &str,
    config: Option<crate::api::StorageConfig>,
) -> Result<Box<dyn ObjectStore>> {
    store_for_uri_with_config_and_logger(uri, config, None)
}

/// Enhanced factory with configuration and optional logger support
pub fn store_for_uri_with_config_and_logger(
    uri: &str,
    config: Option<crate::api::StorageConfig>,
    logger: Option<crate::s3_logger::Logger>,
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;

    // The `.env` policy applies before any store reads a setting (#243).
    crate::dotenv::apply();

    let store: Box<dyn ObjectStore> = match (infer_scheme(uri), config) {
        (Scheme::File, Some(crate::api::StorageConfig::File(cfg))) => {
            Box::new(FileSystemObjectStore::with_config(cfg))
        }
        (Scheme::File, None) => FileSystemObjectStore::boxed(),
        (Scheme::Direct, Some(crate::api::StorageConfig::Direct(cfg))) => {
            ConfigurableFileSystemObjectStore::boxed(cfg)
        }
        (Scheme::Direct, None) => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        (Scheme::Direct, Some(crate::api::StorageConfig::File(_))) => {
            bail!("Cannot use FileSystemConfig with direct:// URI - use DirectFileSystemConfig instead")
        }
        (Scheme::File, Some(crate::api::StorageConfig::Direct(_))) => {
            bail!(
                "Cannot use DirectFileSystemConfig with file:// URI - use FileSystemConfig instead"
            )
        }
        (Scheme::S3, _) => {
            if let Some(endpoint_url) = store_routing_endpoint_for_uri(uri)? {
                // Per-endpoint store: dedicated connection pool, ignores AWS_ENDPOINT_URL.
                crate::s3_client::run_on_global_rt(async move {
                    S3ObjectStore::for_endpoint(&endpoint_url)
                        .await
                        .map(|s| Box::new(s) as Box<dyn ObjectStore>)
                })?
            } else {
                S3ObjectStore::boxed()
            }
        }
        (Scheme::Azure, _) => {
            #[cfg(feature = "backend-azure")]
            {
                AzureObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-azure"))]
            {
                bail!("Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'.")
            }
        }
        (Scheme::Gcs, _) => {
            #[cfg(feature = "backend-gcs")]
            {
                GcsObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-gcs"))]
            {
                bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
            }
        }
        (Scheme::Unknown, _) => bail!("Unable to infer backend from URI: {uri}"),
    };

    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::with_container_prefix(
            Arc::from(store),
            logger,
            container_uri_prefix(uri),
        )))
    } else {
        Ok(store)
    }
}

/// Factory for creating file stores with O_DIRECT enabled for AI/ML workloads
pub fn direct_io_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    direct_io_store_for_uri_with_logger(uri, None)
}

/// Factory for creating file stores with O_DIRECT and optional logger
pub fn direct_io_store_for_uri_with_logger(
    uri: &str,
    logger: Option<crate::s3_logger::Logger>,
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;

    // The `.env` policy applies before any store reads a setting (#243), as in `store_for_uri*`.
    crate::dotenv::apply();

    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => s3_store_for_uri(uri)?,
        Scheme::Azure => {
            #[cfg(feature = "backend-azure")]
            {
                AzureObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-azure"))]
            {
                bail!("Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'.")
            }
        }
        Scheme::Gcs => {
            #[cfg(feature = "backend-gcs")]
            {
                GcsObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-gcs"))]
            {
                bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
            }
        }
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };

    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::with_container_prefix(
            Arc::from(store),
            logger,
            container_uri_prefix(uri),
        )))
    } else {
        Ok(store)
    }
}

/// Factory for creating high-performance stores optimized for AI/ML workloads
pub fn high_performance_store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    high_performance_store_for_uri_with_logger(uri, None)
}

/// Factory for creating high-performance stores with optional logger
pub fn high_performance_store_for_uri_with_logger(
    uri: &str,
    logger: Option<crate::s3_logger::Logger>,
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;

    // The `.env` policy applies before any store reads a setting (#243), as in `store_for_uri*`.
    crate::dotenv::apply();

    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => ConfigurableFileSystemObjectStore::boxed_high_performance(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => s3_store_for_uri(uri)?,
        Scheme::Azure => {
            #[cfg(feature = "backend-azure")]
            {
                AzureObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-azure"))]
            {
                bail!("Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'.")
            }
        }
        Scheme::Gcs => {
            #[cfg(feature = "backend-gcs")]
            {
                GcsObjectStore::boxed()
            }
            #[cfg(not(feature = "backend-gcs"))]
            {
                bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
            }
        }
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };

    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::with_container_prefix(
            Arc::from(store),
            logger,
            container_uri_prefix(uri),
        )))
    } else {
        Ok(store)
    }
}

/// Factory for creating cloud storage backends with RangeEngine enabled
///
/// This factory enables high-performance concurrent downloads for cloud storage (S3, Azure, GCS)
/// by enabling RangeEngine with network-optimized settings. Use this when:
/// - Testing within the cloud provider's network (low latency, high bandwidth)
/// - Working with large objects (> 32 MiB) where parallelism improves throughput
/// - Network conditions support concurrent connections (> 10 Gbps, < 3 ms latency)
///
/// Note: For S3, range optimization is also available via `S3DLIO_ENABLE_RANGE_OPTIMIZATION`
/// (enabled by default, v0.9.60+) without needing this factory.
/// This factory explicitly enables the struct-based RangeEngine for all backends.
///
/// # Arguments
/// * `uri` - Storage URI (e.g., "s3://bucket/", "az://account/container/", "gs://bucket/")
///
/// # Network Scenarios
///
/// **High-performance (local/in-cloud)**: > 10 Gbps, < 3 ms latency
/// - Use this factory to enable RangeEngine
/// - Expected improvement: 30-60% for large objects
///
/// **Remote/low-bandwidth**: < 1 Gbps, > 30 ms latency
/// - Use standard `store_for_uri()` (struct RangeEngine disabled by default)
/// - Avoids stat() overhead that may not be worth parallelism cost
///
/// # Example
/// ```rust,no_run
/// use s3dlio::object_store::store_for_uri_with_high_performance_cloud;
///
/// # async fn example() -> anyhow::Result<()> {
/// // High-performance mode for in-cloud testing
/// let store = store_for_uri_with_high_performance_cloud("s3://my-bucket/")?;
/// let data = store.get("s3://my-bucket/large-file.bin").await?;
/// # Ok(())
/// # }
/// ```
pub fn store_for_uri_with_high_performance_cloud(uri: &str) -> Result<Box<dyn ObjectStore>> {
    store_for_uri_with_high_performance_cloud_and_logger(uri, None)
}

/// Factory for creating cloud storage backends with RangeEngine enabled and optional logger
pub fn store_for_uri_with_high_performance_cloud_and_logger(
    uri: &str,
    logger: Option<crate::s3_logger::Logger>,
) -> Result<Box<dyn ObjectStore>> {
    use crate::object_store_logger::LoggedObjectStore;
    use std::sync::Arc;

    // The `.env` policy applies before any store reads a setting (#243), as in `store_for_uri*`.
    crate::dotenv::apply();

    let store: Box<dyn ObjectStore> = match infer_scheme(uri) {
        Scheme::File => ConfigurableFileSystemObjectStore::boxed_high_performance(),
        Scheme::Direct => ConfigurableFileSystemObjectStore::boxed_direct_io(),
        Scheme::S3 => match store_routing_endpoint_for_uri(uri)? {
            // A URI naming a server gets that server's store (issue #197).
            Some(endpoint_url) => s3_per_endpoint_store(endpoint_url)?,
            None => {
                let config = S3Config {
                    enable_range_engine: true, // Enable for high-performance
                    ..Default::default()
                };
                Box::new(S3ObjectStore::with_config(config))
            }
        },
        Scheme::Azure => {
            #[cfg(feature = "backend-azure")]
            {
                let config = AzureConfig {
                    enable_range_engine: true, // Enable for high-performance
                    ..Default::default()
                };
                Box::new(AzureObjectStore::with_config(config))
            }
            #[cfg(not(feature = "backend-azure"))]
            {
                bail!("Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'.")
            }
        }
        Scheme::Gcs => {
            #[cfg(feature = "backend-gcs")]
            {
                let config = GcsConfig {
                    enable_range_engine: true, // Enable for high-performance
                    ..Default::default()
                };
                Box::new(GcsObjectStore::with_config(config))
            }
            #[cfg(not(feature = "backend-gcs"))]
            {
                bail!("GCS backend is not enabled in this build. Rebuild with feature 'backend-gcs' or 'full-backends'.")
            }
        }
        Scheme::Unknown => bail!("Unable to infer backend from URI: {uri}"),
    };

    // Wrap with logger if provided
    if let Some(logger) = logger {
        Ok(Box::new(LoggedObjectStore::with_container_prefix(
            Arc::from(store),
            logger,
            container_uri_prefix(uri),
        )))
    } else {
        Ok(store)
    }
}

// ============================================================================
// Generic Upload/Download Functions with Progress Tracking
// ============================================================================

use futures::stream::FuturesUnordered;
use std::path::Path;
use std::sync::Arc;
// StreamExt already imported at top of file
use crate::progress::ProgressCallback;
use glob;
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Copy, Default)]
pub struct TransferSummary {
    pub attempted: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub bytes_transferred: u64,
}

/// Generic upload function that works with any ObjectStore backend
/// and supports progress tracking. Supports glob patterns (*,?) and regex patterns.
pub async fn generic_upload_files<P: AsRef<Path>>(
    dest_prefix: &str,
    patterns: &[P],
    max_in_flight: usize,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let summary =
        generic_upload_files_with_summary(dest_prefix, patterns, max_in_flight, progress_callback)
            .await?;
    if summary.failed > 0 {
        bail!(
            "Upload completed with failures: attempted={}, succeeded={}, failed={}",
            summary.attempted,
            summary.succeeded,
            summary.failed
        );
    }
    Ok(())
}

/// Generic upload function with detailed transfer summary.
pub async fn generic_upload_files_with_summary<P: AsRef<Path>>(
    dest_prefix: &str,
    patterns: &[P],
    max_in_flight: usize,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<TransferSummary> {
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    // Expand patterns (globs, regex, and single paths)
    let mut paths = Vec::new();
    for pat in patterns {
        let s = pat.as_ref().to_string_lossy();

        // Handle different pattern types
        if s.contains('*') || s.contains('?') {
            // Glob pattern - use glob crate
            for entry in glob::glob(&s).map_err(|e| anyhow!("Glob pattern error: {}", e))? {
                match entry {
                    Ok(pb) => {
                        if pb.is_file() {
                            paths.push(pb);
                        }
                    }
                    Err(e) => warn!("Glob error for pattern {}: {}", s, e),
                }
            }
        } else if s.contains('^')
            || s.contains('$')
            || s.contains('[')
            || s.contains('(')
            || s.contains('\\')
            || s.contains('.')
        {
            // Regex pattern - scan directory and apply regex
            let (dir_part, pattern_part) = match s.rfind('/') {
                Some(index) => {
                    let (dir, pattern) = s.split_at(index + 1);
                    (dir, pattern)
                }
                None => ("./", s.as_ref()),
            };

            let pattern_part = if pattern_part.is_empty() {
                ".*"
            } else {
                pattern_part
            };
            debug!(
                "Trying regex pattern '{}' in directory '{}'",
                pattern_part, dir_part
            );

            match Regex::new(pattern_part) {
                Ok(re) => {
                    let dir_path = std::path::Path::new(dir_part);
                    if dir_path.is_dir() {
                        for entry in (std::fs::read_dir(dir_path)
                            .map_err(|e| anyhow!("Cannot read directory '{}': {}", dir_part, e))?)
                        .flatten()
                        {
                            let path = entry.path();
                            if path.is_file() {
                                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                                    if re.is_match(filename) {
                                        debug!("Regex matched file: {:?}", path);
                                        paths.push(path);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // Not a valid regex, treat as literal file path
                    debug!(
                        "Invalid regex pattern '{}', treating as literal path",
                        pattern_part
                    );
                    let path = pat.as_ref();
                    if path.is_file() {
                        paths.push(path.to_path_buf());
                    }
                }
            }
        } else {
            // Regular file path
            let path = pat.as_ref();
            if path.is_file() {
                paths.push(path.to_path_buf());
            } else if path.is_dir() {
                // If it's a directory, add all files in it
                for entry in (std::fs::read_dir(path)
                    .map_err(|e| anyhow!("Cannot read directory '{:?}': {}", path, e))?)
                .flatten()
                {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        paths.push(entry_path);
                    }
                }
            } else {
                warn!("Path does not exist or is not accessible: {:?}", path);
            }
        }
    }

    if paths.is_empty() {
        bail!("No files matched for upload");
    }

    // Cap the number of concurrent tasks
    let effective_jobs = std::cmp::min(max_in_flight, paths.len());

    info!(
        "Starting upload of {} file(s) to {} (jobs={})",
        paths.len(),
        dest_prefix,
        effective_jobs
    );

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let mut futs = FuturesUnordered::new();

    for path in paths.clone() {
        debug!("queueing upload for {:?}", path);
        let sem = sem.clone();
        let store = store_for_uri_with_logger(dest_prefix, global_logger())?; // Each task gets its own store instance
        let progress = progress_callback.clone();
        let dest_base = dest_prefix.to_string();

        let fname = path
            .file_name()
            .ok_or_else(|| anyhow!("Bad path {:?}", path))?
            .to_string_lossy();

        // Construct destination URI
        let dest_uri = if dest_base.ends_with('/') {
            format!("{}{}", dest_base, fname)
        } else {
            format!("{}/{}", dest_base, fname)
        };

        // Get file size for progress tracking
        let file_size = std::fs::metadata(&path)?.len();

        futs.push(tokio::spawn(async move {
            debug!("starting upload of {:?} → {}", path, dest_uri);
            let _permit = sem.acquire_owned().await.unwrap();

            // Read file data
            let data = tokio::fs::read(&path).await?;

            // Upload via ObjectStore trait
            store.put(&dest_uri, Bytes::from(data)).await?;

            debug!("finished upload of {:?} → {}", path, dest_uri);

            // Update progress if callback provided
            if let Some(ref progress) = progress {
                progress.object_completed(file_size);
            }

            Ok::<(), anyhow::Error>(())
        }));
    }

    let mut summary = TransferSummary {
        attempted: paths.len() as u64,
        ..Default::default()
    };

    // Wait for all uploads to complete
    while let Some(join_res) = futs.next().await {
        match join_res {
            Ok(Ok(())) => {
                summary.succeeded += 1;
            }
            Ok(Err(e)) => {
                summary.failed += 1;
                warn!("Upload task failed: {}", e);
            }
            Err(e) => {
                summary.failed += 1;
                warn!("Upload task join failed: {}", e);
            }
        }
    }

    info!(
        "Finished upload of {} file(s) to {}",
        paths.len(),
        dest_prefix
    );
    if let Some(progress) = &progress_callback {
        summary.bytes_transferred = progress
            .bytes_transferred
            .load(std::sync::atomic::Ordering::Relaxed);
    }
    Ok(summary)
}

/// Generic download function that works with any ObjectStore backend
/// and supports progress tracking.
pub async fn generic_download_objects(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let summary = generic_download_objects_with_summary(
        src_uri,
        dest_dir,
        max_in_flight,
        recursive,
        progress_callback,
    )
    .await?;
    if summary.failed > 0 {
        bail!(
            "Download completed with failures: attempted={}, succeeded={}, failed={}",
            summary.attempted,
            summary.succeeded,
            summary.failed
        );
    }
    Ok(())
}

/// Generic download function with detailed transfer summary.
///
/// `src_uri` selects the objects; on `s3://` the rule is [`crate::s3_selection`] (issue #213):
/// a name selects that object (and its `name/` directory when recursive) and never
/// `name.bak`, a trailing `/` a directory, `*` / `**` an explicit glob.
/// Use [`generic_download_objects_matching_with_summary`] to narrow the selection with an
/// explicit regex.
pub async fn generic_download_objects_with_summary(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<TransferSummary> {
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    generic_download_objects_matching_with_summary(
        src_uri,
        dest_dir,
        max_in_flight,
        recursive,
        None,
        progress_callback,
    )
    .await
}

/// [`generic_download_objects_with_summary`] with an optional explicit regex filter.
///
/// Backend-aware (S3, Azure, GCS, `file://`, `direct://`): the listing goes through the
/// store chosen by `src_uri`. When `pattern` is given, the listed entries are narrowed with
/// [`ListPattern::retain`] (the pattern is matched against the entry relative to the listing
/// base of `src_uri`) BEFORE the destination directory is created or anything is downloaded.
/// Compile the [`ListPattern`] before calling, so an invalid regex fails without any request.
pub async fn generic_download_objects_matching_with_summary(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    pattern: Option<&crate::list_pattern::ListPattern>,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<TransferSummary> {
    // Issue #211: a limit of 0 builds a semaphore with no permits and hangs.  This is the
    // implementation every backend-aware download funnels through (and a public entry of
    // its own), so the guard lives here: before the store is chosen, anything is listed or
    // the destination directory is created.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    // Issue #234: the store that lists is the store that fetches.  A host-form URI lists on a
    // per-endpoint store, whose entries come back in bucket form; a store built from such an
    // entry would use the process-wide client (another server).  The per-endpoint store
    // accepts the bucket-form URIs of its own bucket and uses its own client for them.
    let store: Arc<dyn ObjectStore> =
        Arc::from(store_for_uri_with_logger(src_uri, global_logger())?);

    // List objects to download
    let mut keys = store.list(src_uri, recursive).await?;
    if let Some(pattern) = pattern {
        pattern.retain(src_uri, &mut keys);
    }

    if keys.is_empty() {
        bail!("No objects matched for download");
    }

    // Ensure destination directory exists
    std::fs::create_dir_all(dest_dir)?;

    // Cap the number of concurrent tasks
    let effective_jobs = std::cmp::min(max_in_flight, keys.len());

    info!(
        "Starting download of {} object(s) from {} to {:?} (jobs={})",
        keys.len(),
        src_uri,
        dest_dir,
        effective_jobs
    );

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let mut futs = FuturesUnordered::new();

    for uri in keys.clone() {
        let sem = sem.clone();
        let store = Arc::clone(&store);
        let progress = progress_callback.clone();
        let out_dir = dest_dir.to_path_buf();

        futs.push(tokio::spawn(async move {
            debug!("starting download of {} → {:?}", uri, out_dir);
            let _permit = sem.acquire_owned().await.unwrap();

            // Skip "directories" (URIs that end with slash)
            if uri.ends_with('/') {
                return Ok::<(), anyhow::Error>(());
            }

            // Download the object
            let bytes = store.get(&uri).await?;
            let byte_count = bytes.len() as u64;

            // Extract filename from URI
            let fname = uri
                .split('/')
                .next_back()
                .ok_or_else(|| anyhow!("Cannot extract filename from URI: {}", uri))?;
            let out_path = out_dir.join(fname);

            // Write to disk
            tokio::fs::write(&out_path, bytes).await?;

            debug!("finished download of {} → {:?}", uri, out_path);

            // Update progress if callback provided
            if let Some(ref progress) = progress {
                progress.object_completed(byte_count);
            }

            Ok::<(), anyhow::Error>(())
        }));
    }

    let mut summary = TransferSummary {
        attempted: keys.iter().filter(|k| !k.ends_with('/')).count() as u64,
        ..Default::default()
    };

    // Wait for all downloads to complete
    while let Some(join_res) = futs.next().await {
        match join_res {
            Ok(Ok(())) => {
                summary.succeeded += 1;
            }
            Ok(Err(e)) => {
                summary.failed += 1;
                warn!("Download task failed: {}", e);
            }
            Err(e) => {
                summary.failed += 1;
                warn!("Download task join failed: {}", e);
            }
        }
    }

    info!(
        "Finished download of {} object(s) to {:?}",
        keys.len(),
        dest_dir
    );
    if let Some(progress) = &progress_callback {
        summary.bytes_transferred = progress
            .bytes_transferred
            .load(std::sync::atomic::Ordering::Relaxed);
    }
    Ok(summary)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for S3ObjectStore per-endpoint isolation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "backend-azure"))]
mod azure_download_mode_tests {
    use super::*;

    #[test]
    fn parses_all_azure_download_modes_and_aliases() {
        assert_eq!("auto".parse(), Ok(AzureDownloadMode::Auto));
        assert_eq!("azure-sdk".parse(), Ok(AzureDownloadMode::AzureSdk));
        assert_eq!("sdk".parse(), Ok(AzureDownloadMode::AzureSdk));
        assert_eq!("s3dlio".parse(), Ok(AzureDownloadMode::S3dlio));
        assert_eq!("range_engine".parse(), Ok(AzureDownloadMode::S3dlio));
        assert_eq!("sequential".parse(), Ok(AzureDownloadMode::Sequential));
        assert!("invalid".parse::<AzureDownloadMode>().is_err());
    }

    #[test]
    fn auto_mode_uses_threshold_and_selected_concurrent_engine() {
        let mut config = AzureConfig {
            download_mode: AzureDownloadMode::Auto,
            auto_concurrent_engine: AzureConcurrentEngine::AzureSdk,
            ..AzureConfig::default()
        };
        config.range_engine.min_split_size = 32 * 1024 * 1024;
        let store = AzureObjectStore::with_config(config.clone());

        assert_eq!(
            store.download_mode_for_size(1024),
            AzureDownloadMode::Sequential
        );
        assert_eq!(
            store.download_mode_for_size(32 * 1024 * 1024),
            AzureDownloadMode::AzureSdk
        );

        config.auto_concurrent_engine = AzureConcurrentEngine::S3dlio;
        let store = AzureObjectStore::with_config(config);
        assert_eq!(
            store.download_mode_for_size(32 * 1024 * 1024),
            AzureDownloadMode::S3dlio
        );
    }

    #[test]
    fn explicit_mode_bypasses_auto_threshold() {
        let config = AzureConfig {
            download_mode: AzureDownloadMode::AzureSdk,
            ..AzureConfig::default()
        };
        let store = AzureObjectStore::with_config(config);
        assert_eq!(store.download_mode_for_size(1), AzureDownloadMode::AzureSdk);
    }

    #[test]
    fn legacy_range_engine_switch_remains_compatible() {
        let mut config = AzureConfig {
            download_mode: AzureDownloadMode::AzureSdk,
            enable_range_engine: true,
            ..AzureConfig::default()
        };
        config.range_engine.min_split_size = 1024;
        let store = AzureObjectStore::with_config(config);

        assert_eq!(
            store.download_mode_for_size(1023),
            AzureDownloadMode::Sequential
        );
        assert_eq!(
            store.download_mode_for_size(1024),
            AzureDownloadMode::S3dlio
        );
    }
}

#[cfg(test)]
mod s3_object_store_tests {
    use super::*;

    /// Mutex to serialize tests that manipulate AWS credential env vars.
    static CRED_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    // ── Tests that need NO credentials / NO network ──────────────────────────

    /// `S3ObjectStore::new()` must NOT pre-configure an endpoint.
    /// It represents the global-client path (backward-compatible).
    #[test]
    fn test_s3_object_store_new_has_no_endpoint_url() {
        let store = S3ObjectStore::new();
        assert!(
            store.endpoint_url().is_none(),
            "S3ObjectStore::new() should use the global client (endpoint_url=None)"
        );
    }

    /// `S3ObjectStore::with_config()` must also produce no endpoint URL.
    #[test]
    fn test_s3_object_store_with_config_has_no_endpoint_url() {
        let store = S3ObjectStore::with_config(S3Config::default());
        assert!(
            store.endpoint_url().is_none(),
            "S3ObjectStore::with_config() should use the global client (endpoint_url=None)"
        );
    }

    // ── Tests that need fake credentials (no network calls) ──────────────────
    //
    // `S3ObjectStore::for_endpoint()` checks that `AWS_ACCESS_KEY_ID` and
    // `AWS_SECRET_ACCESS_KEY` are set, then builds an SDK config — all
    // synchronous.  No actual TCP connection is made at construction time.
    //
    // CRED_LOCK serializes these tests to avoid env-var races in parallel runs.

    /// `for_endpoint()` must store the endpoint URL for diagnostics / assertions.
    #[tokio::test]
    async fn test_s3_object_store_for_endpoint_sets_endpoint_url() {
        let _guard = CRED_LOCK.lock().await;

        let saved_key = std::env::var("AWS_ACCESS_KEY_ID").ok();
        let saved_secret = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
        let saved_region = std::env::var("AWS_REGION").ok();

        #[allow(deprecated)]
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        #[allow(deprecated)]
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        #[allow(deprecated)]
        std::env::set_var("AWS_REGION", "us-east-1");

        let result = S3ObjectStore::for_endpoint("http://127.0.0.1:9000").await;

        // Restore env
        restore_env("AWS_ACCESS_KEY_ID", saved_key);
        restore_env("AWS_SECRET_ACCESS_KEY", saved_secret);
        restore_env("AWS_REGION", saved_region);

        let store = result.expect("for_endpoint() should succeed with fake credentials");
        assert_eq!(
            store.endpoint_url(),
            Some("http://127.0.0.1:9000"),
            "endpoint_url() must reflect the URL passed to for_endpoint()"
        );
    }

    /// Two stores constructed for **different** endpoint URLs must each record
    /// their own URL — proving that `endpoint_url` is stored per-instance.
    ///
    /// This is the regression test for the original bug where
    /// `MultiEndpointStore` allocated a single global `S3ObjectStore` for all
    /// endpoints, causing all traffic to share one connection pool.
    ///
    /// Uses `for_test()` to avoid env-var manipulation: the invariant being
    /// tested is purely about per-instance state, not credential handling.
    #[test]
    fn test_two_stores_for_different_endpoints_have_different_urls() {
        let s1 = S3ObjectStore::for_test("http://10.9.0.17:9000");
        let s2 = S3ObjectStore::for_test("http://10.9.0.18:9000");

        // Each store records its own endpoint URL
        assert_eq!(s1.endpoint_url(), Some("http://10.9.0.17:9000"));
        assert_eq!(s2.endpoint_url(), Some("http://10.9.0.18:9000"));

        // Critical: they must be different (not the same global client)
        assert_ne!(
            s1.endpoint_url(),
            s2.endpoint_url(),
            "Stores for different endpoints must report different endpoint URLs"
        );
    }

    // ── E1 (audit #154 bug 4.2) — infer_scheme / is_azure_blob_host ──────────
    //
    // RED-then-GREEN regression tests: pre-fix, `infer_scheme` used
    // `uri.contains(".blob.core.windows.net/")` against the *entire* URI.
    // Adversarial verify on the original audit finding confirmed `s3://...`
    // can't reproduce the misroute (it short-circuits earlier in the
    // if/else chain), but `gs://...` genuinely can: the Azure substring
    // check runs *before* the `gs://` prefix check, so a GCS URI whose key
    // happens to contain that literal text was misrouted to Azure.

    #[test]
    fn infer_scheme_routes_azure_https_hostname_to_azure() {
        assert_eq!(
            infer_scheme("https://myacct.blob.core.windows.net/c/k"),
            Scheme::Azure
        );
    }

    #[test]
    fn infer_scheme_does_not_misroute_gcs_key_that_mentions_azure_host_in_path() {
        // Pre-fix: `.contains(".blob.core.windows.net/")` matched here
        // because the substring appears in the *key*, and that check ran
        // before the `gs://` prefix check -- misrouting this to Azure.
        assert_eq!(
            infer_scheme("gs://bucket/mirror/acct.blob.core.windows.net/file.dat"),
            Scheme::Gcs
        );
    }

    #[test]
    fn infer_scheme_still_recognizes_az_scheme() {
        assert_eq!(infer_scheme("az://container/blob"), Scheme::Azure);
    }

    // parse_azure_uri() is only compiled under backend-azure; this is the
    // other half of E1 -- the audit's actual failure scenario is that
    // infer_scheme() correctly routes an `http://...blob.core.windows.net/...`
    // URL to Azure, but parse_azure_uri() then hard-required a "https://"
    // prefix and bailed with a confusing "expected https://" error instead
    // of parsing it.
    #[cfg(feature = "backend-azure")]
    #[test]
    fn parse_azure_uri_accepts_http_scheme_not_just_https() {
        let (account, container, key) =
            parse_azure_uri("http://myacct.blob.core.windows.net/container/key/nested")
                .expect("http:// Azure blob URL must parse, not require https://");
        assert_eq!(account, "myacct");
        assert_eq!(container, "container");
        assert_eq!(key, "key/nested");
    }

    // Issue #198: the https form of an Azure URI names the same blob as the az:// form,
    // key included. Only the single `/` that separates the container from the key is
    // structural; an empty segment (`c//k`, key `/k`) is part of the key.
    #[cfg(feature = "backend-azure")]
    #[test]
    fn parse_azure_https_uri_keeps_the_key_exactly() {
        for key in ["x", "/x", "a//b", "//x", "a/b/", "dir/"] {
            let https = format!("https://acct.blob.core.windows.net/cont/{key}");
            let az = format!("az://acct/cont/{key}");
            let (h_account, h_container, h_key) = parse_azure_uri(&https).expect("https parses");
            let (a_account, a_container, a_key) = parse_azure_uri(&az).expect("az parses");
            assert_eq!(h_key, key, "{https}");
            assert_eq!(
                (h_account, h_container, h_key),
                (a_account, a_container, a_key),
                "https and az:// must name the same blob for {key:?}"
            );
        }
    }

    #[cfg(feature = "backend-azure")]
    #[test]
    fn azure_hosts_are_recognised_whatever_their_case() {
        assert!(is_azure_blob_host(
            "https://MyAcct.Blob.Core.Windows.NET/c/k"
        ));
        assert!(!is_azure_blob_host(
            "https://evil.example/acct.blob.core.windows.net/k"
        ));
        let (account, _, _) = parse_azure_uri("https://MyAcct.Blob.Core.Windows.NET/c/k").unwrap();
        assert_eq!(account, "myacct", "the host's account is lowercased");
    }

    // A URI that carries userinfo or a backslash in its authority is still classified Azure (so
    // it reaches the parser, which refuses it) and is never read as an account.
    #[test]
    fn azure_host_urls_with_a_smuggling_authority_are_classified_azure_and_refused() {
        for uri in [
            "https://abc.def@real.blob.core.windows.net/c/k",
            "https://evil.example\\@real.blob.core.windows.net/c/k",
        ] {
            assert_eq!(infer_scheme(uri), Scheme::Azure, "{uri}");
            assert!(
                crate::capabilities::parse_azure_uri_pure(uri).is_err(),
                "{uri}"
            );
        }
    }

    #[cfg(feature = "backend-azure")]
    #[test]
    fn parse_azure_https_uri_container_only_has_an_empty_key() {
        for uri in [
            "https://acct.blob.core.windows.net/cont",
            "https://acct.blob.core.windows.net/cont/",
        ] {
            let (account, container, key) = parse_azure_uri(uri).expect("parses");
            assert_eq!(
                (account.as_str(), container.as_str(), key.as_str()),
                ("acct", "cont", "")
            );
        }
    }

    // An empty container is not a container: refuse it rather than shift the key left.
    #[cfg(feature = "backend-azure")]
    #[test]
    fn parse_azure_https_uri_without_a_container_is_an_error() {
        for uri in [
            "https://acct.blob.core.windows.net",
            "https://acct.blob.core.windows.net/",
            "https://acct.blob.core.windows.net//k",
        ] {
            assert!(parse_azure_uri(uri).is_err(), "{uri}");
        }
    }

    // Issue #224: `az://account//key` has an EMPTY container. It used to parse (container "",
    // key "key") and reach the client builder; it must now be refused by parsing alone, like the
    // https form above.
    #[cfg(feature = "backend-azure")]
    #[test]
    fn parse_azure_az_uri_with_an_empty_container_is_an_error() {
        for uri in [
            "az://account//key",
            "az://account//",
            "az://account/",
            "az://account",
            "az://account///key",
        ] {
            let err = parse_azure_uri(uri).expect_err(uri);
            assert!(format!("{err:#}").contains("container"), "{uri}: {err:#}");
        }
    }

    // The valid forms #224 must not disturb: roots, nested keys, percent signs, and keys that
    // are literal (`%2F` stays three characters).
    #[cfg(feature = "backend-azure")]
    #[test]
    fn parse_azure_az_uri_valid_roots_and_literal_keys_still_parse() {
        for (uri, want) in [
            ("az://account/cont", ("account", "cont", "")),
            ("az://account/cont/", ("account", "cont", "")),
            ("az://account/cont/k", ("account", "cont", "k")),
            ("az://account/cont//k", ("account", "cont", "/k")),
            ("az://account/cont/a%2Fb", ("account", "cont", "a%2Fb")),
            (
                "az://account/cont/100%25/x%",
                ("account", "cont", "100%25/x%"),
            ),
        ] {
            let (account, container, key) = parse_azure_uri(uri).expect(uri);
            assert_eq!(
                (account.as_str(), container.as_str(), key.as_str()),
                want,
                "{uri}"
            );
        }
    }

    // ── E2 (audit #154 bug 4.3) — s3_endpoint_url_from_uri ────────────────────
    //
    // RED-then-GREEN regression test: pre-fix, scheme selection was purely
    // `port == 443`, with no way for an operator to mark a TLS-terminated
    // S3-compatible endpoint running on a non-standard port (e.g. a load
    // balancer terminating TLS on :9001) as https. S3DLIO_S3_ENDPOINT_TLS_PORTS
    // closes that gap.

    #[test]
    fn container_uri_prefix_is_the_scheme_plus_the_host_authority_for_host_form_s3() {
        assert_eq!(container_uri_prefix("s3://bucket/key"), "s3://");
        assert_eq!(container_uri_prefix("file:///tmp/x"), "file://");
        assert_eq!(container_uri_prefix("direct:///tmp/x"), "direct://");
        assert_eq!(container_uri_prefix("az://account/container/b"), "az://");
        assert_eq!(container_uri_prefix("gs://bucket/o"), "gs://");
        assert_eq!(
            container_uri_prefix("s3://10.9.0.17:80/bucket/key"),
            "s3://10.9.0.17:80/"
        );
        assert_eq!(
            container_uri_prefix("s3://[::1]:9000/bucket/key"),
            "s3://[::1]:9000/"
        );
        assert_eq!(container_uri_prefix("not-a-uri"), "");
    }

    #[test]
    fn s3_endpoint_url_from_uri_plain_bucket_has_no_endpoint() {
        assert_eq!(
            s3_endpoint_url_from_uri("s3://my-bucket/prefix/key").unwrap(),
            None
        );
    }

    #[test]
    fn s3_endpoint_url_from_uri_host_port_uses_http() {
        assert_eq!(
            s3_endpoint_url_from_uri("s3://10.9.0.17:80/bucket/key").unwrap(),
            Some("http://10.9.0.17:80".to_string())
        );
    }

    #[test]
    fn s3_endpoint_url_from_uri_port_443_uses_https() {
        assert_eq!(
            s3_endpoint_url_from_uri("s3://minio.example.com:443/bucket/key").unwrap(),
            Some("https://minio.example.com:443".to_string())
        );
    }

    #[test]
    fn s3_endpoint_url_from_uri_bracketed_ipv6_with_port() {
        assert_eq!(
            s3_endpoint_url_from_uri("s3://[::1]:9000/bucket/key").unwrap(),
            Some("http://[::1]:9000".to_string())
        );
    }

    #[test]
    fn s3_endpoint_url_from_uri_custom_tls_port_via_env_var() {
        let _guard = ENDPOINT_TLS_PORTS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let saved = std::env::var("S3DLIO_S3_ENDPOINT_TLS_PORTS").ok();

        #[allow(deprecated)]
        std::env::set_var("S3DLIO_S3_ENDPOINT_TLS_PORTS", "9001,9443");
        let result = s3_endpoint_url_from_uri("s3://minio.example:9001/bucket/key").unwrap();
        restore_env("S3DLIO_S3_ENDPOINT_TLS_PORTS", saved);

        assert_eq!(
            result,
            Some("https://minio.example:9001".to_string()),
            "a port named in S3DLIO_S3_ENDPOINT_TLS_PORTS must produce an https:// endpoint"
        );
    }

    #[test]
    fn s3_endpoint_url_from_uri_unlisted_custom_port_still_uses_http() {
        let _guard = ENDPOINT_TLS_PORTS_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let saved = std::env::var("S3DLIO_S3_ENDPOINT_TLS_PORTS").ok();

        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_S3_ENDPOINT_TLS_PORTS");
        let result = s3_endpoint_url_from_uri("s3://minio.example:9001/bucket/key").unwrap();
        restore_env("S3DLIO_S3_ENDPOINT_TLS_PORTS", saved);

        assert_eq!(result, Some("http://minio.example:9001".to_string()));
    }

    // Issues #178 / #193: only an explicit port makes the first segment a server.
    #[test]
    fn s3_endpoint_url_from_uri_portless_dns_name_is_a_bucket() {
        for uri in [
            "s3://data.example.com/dir/file.bin",
            "s3://minio.example.com/bkt/key",
            "s3://localhost/bkt/key",
        ] {
            assert_eq!(s3_endpoint_url_from_uri(uri).unwrap(), None, "{uri}");
        }
    }

    #[test]
    fn s3_endpoint_url_from_uri_portless_ip_is_an_error() {
        for uri in [
            "s3://10.0.0.5/bkt/key",
            "s3://[::1]/bkt/key",
            "s3://::1/bkt/key",
        ] {
            let err = s3_endpoint_url_from_uri(uri).unwrap_err();
            assert!(format!("{err:#}").contains(":port"), "{uri}: {err:#}");
        }
    }

    // F12: a classification error must propagate, never become "not a server".
    #[test]
    fn s3_endpoint_url_from_authority_propagates_classification_errors() {
        // Not servers at all (buckets): an error saying so.
        for bucket in ["data.example.com", "localhost"] {
            let err = s3_endpoint_url_from_authority(bucket).unwrap_err();
            assert!(
                format!("{err:#}").contains("not a server"),
                "{bucket}: {err:#}"
            );
        }
        // Classifier errors keep the classifier's own message.
        for (bad, hint) in [
            ("10.0.0.5", "IP address with no port"),
            ("host:notaport", "invalid port 'notaport'"),
            ("host:0", "invalid port '0'"),
            ("http://host:9000", "looks like a URL"),
            // Host grammar: no endpoint URL is ever built from these.
            ("node/prefix:9000", "invalid character '/'"),
            ("a b:9000", "invalid character ' '"),
            ("ho%st:9000", "invalid character '%'"),
            ("host.:9000", "empty label"),
        ] {
            let err = s3_endpoint_url_from_authority(bad).unwrap_err();
            assert!(format!("{err:#}").contains(hint), "{bad}: {err:#}");
        }
        assert_eq!(
            s3_endpoint_url_from_authority("host:9000").unwrap(),
            "http://host:9000"
        );
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn restore_env(key: &str, saved: Option<String>) {
        match saved {
            #[allow(deprecated)]
            Some(v) => std::env::set_var(key, v),
            #[allow(deprecated)]
            None => std::env::remove_var(key),
        }
    }
}

// =============================================================================
// Issue #229: mkdir / rmdir are successful NO-OPS on object stores
// =============================================================================
//
// An object store has no directories, so `mkdir` and `rmdir` (recursive or not)
// validate the URI and return Ok without any request. Azure and GCS have no mock
// server in this repository: these unit tests run the REAL store (no client is ever
// built, so no credentials or network are needed) and prove the no-op and the URI
// validation. They are not an end-to-end exercise against the cloud service.
#[cfg(test)]
mod issue229_object_store_dir_noop_tests {
    #[allow(unused_imports)]
    use super::*;

    #[cfg(feature = "backend-azure")]
    #[tokio::test]
    async fn azure_mkdir_rmdir_are_noops_and_validate_the_uri() {
        let store = AzureObjectStore::new();
        for uri in [
            "az://acct/cont/dir",
            "az://acct/cont/dir/",
            "az://acct/cont/",
        ] {
            store.mkdir(uri).await.unwrap();
            store.rmdir(uri, true).await.unwrap();
            store.rmdir(uri, false).await.unwrap();
        }
        assert!(store.mkdir("az://acct").await.is_err(), "no container");
        assert!(
            store.rmdir("s3://b/dir", true).await.is_err(),
            "wrong scheme"
        );
    }

    #[cfg(feature = "backend-gcs")]
    #[tokio::test]
    async fn gcs_mkdir_rmdir_are_noops_and_validate_the_uri() {
        let store = GcsObjectStore::new();
        for uri in ["gs://bkt/dir", "gs://bkt/dir/", "gs://bkt/", "gs://bkt"] {
            store.mkdir(uri).await.unwrap();
            store.rmdir(uri, true).await.unwrap();
            store.rmdir(uri, false).await.unwrap();
        }
        assert!(store.mkdir("gs:///dir").await.is_err(), "empty bucket");
        assert!(
            store.rmdir("s3://b/dir", true).await.is_err(),
            "wrong scheme"
        );
    }
}

// =============================================================================
// ObjectStore::list() — common-prefix (virtual directory) tests
//
// These tests use `file://` URIs so they run without any cloud credentials.
// They specifically guard against regressions where non-recursive listing
// silently discards common prefixes (subdirectory entries).
// =============================================================================
#[cfg(test)]
mod list_common_prefix_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Build a temporary directory tree and return its path:
    ///   <root>/
    ///     top_file.txt
    ///     subdir/
    ///       nested_file.txt
    fn make_tree() -> TempDir {
        let dir = TempDir::new().expect("create tempdir");
        fs::write(dir.path().join("top_file.txt"), b"top").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();
        fs::write(dir.path().join("subdir").join("nested_file.txt"), b"nested").unwrap();
        dir
    }

    /// Non-recursive list of the root must include the subdirectory entry AND
    /// the top-level file.  Before the fix, directory entries were silently dropped.
    #[tokio::test]
    async fn test_non_recursive_list_includes_subdirectory() {
        let tree = make_tree();
        let uri = format!("file://{}/", tree.path().display());
        let store = store_for_uri_with_logger(&uri, None).expect("create file store");
        let keys = store.list(&uri, /*recursive=*/ false).await.expect("list");

        let has_subdir = keys.iter().any(|k| k.contains("subdir"));
        let has_top_file = keys.iter().any(|k| k.contains("top_file.txt"));

        assert!(
            has_subdir,
            "non-recursive list must include subdirectory entries; got: {:?}",
            keys
        );
        assert!(
            has_top_file,
            "non-recursive list must include top-level files; got: {:?}",
            keys
        );
    }

    /// Recursive list must include the nested file but NOT add an extra synthetic
    /// directory entry for the subdirectory.
    #[tokio::test]
    async fn test_recursive_list_includes_nested_file() {
        let tree = make_tree();
        let uri = format!("file://{}/", tree.path().display());
        let store = store_for_uri_with_logger(&uri, None).expect("create file store");
        let keys = store.list(&uri, /*recursive=*/ true).await.expect("list");

        let has_nested = keys.iter().any(|k| k.contains("nested_file.txt"));
        assert!(
            has_nested,
            "recursive list must include nested files; got: {:?}",
            keys
        );
    }

    /// Non-recursive list of a subdirectory must only return its own contents,
    /// not sibling directories.
    #[tokio::test]
    async fn test_non_recursive_list_of_subdir() {
        let tree = make_tree();
        let uri = format!("file://{}/subdir/", tree.path().display());
        let store = store_for_uri_with_logger(&uri, None).expect("create file store");
        let keys = store.list(&uri, /*recursive=*/ false).await.expect("list");

        assert!(
            keys.iter().any(|k| k.contains("nested_file.txt")),
            "listing subdir/ must include nested_file.txt; got: {:?}",
            keys
        );
        assert!(
            !keys.iter().any(|k| k.contains("top_file.txt")),
            "listing subdir/ must NOT include sibling top_file.txt; got: {:?}",
            keys
        );
    }
}

// ── delete_batch URI grouping for the GCS and Azure adapters (issue #179) ──
//
// The S3 adapter is covered end-to-end by tests/test_delete_batch_buckets.rs
// (mock endpoint).  GCS and Azure need live credentials for a store-level
// test, so their bucket/container grouping is pinned here with the exact
// parser each adapter hands to `group_uris_by_bucket`.
#[cfg(test)]
mod delete_batch_grouping_tests {
    #[cfg(any(feature = "backend-gcs", feature = "backend-azure"))]
    use super::*;

    #[cfg(feature = "backend-gcs")]
    #[test]
    fn gcs_batch_is_split_per_bucket_and_gcs_scheme_alias_groups_together() {
        let uris: Vec<String> = ["gs://bucket1/p/a", "gs://bucket2/p/b", "gcs://bucket1/p/c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let g = group_uris_by_bucket(&uris, parse_gcs_uri).expect("group");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].target, "bucket1");
        assert_eq!(g[0].keys, vec!["p/a", "p/c"]);
        assert_eq!(g[1].target, "bucket2");
        assert_eq!(g[1].keys, vec!["p/b"]);
    }

    #[cfg(feature = "backend-gcs")]
    #[test]
    fn gcs_batch_with_unparseable_uri_is_an_error() {
        let uris: Vec<String> = ["gs://bucket1/a", "s3://wrong-scheme/b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let err = group_uris_by_bucket(&uris, parse_gcs_uri).unwrap_err();
        assert!(format!("{err:#}").contains("s3://wrong-scheme/b"));
    }

    #[cfg(feature = "backend-azure")]
    #[test]
    fn azure_batch_is_split_per_account_and_container() {
        let uris: Vec<String> = [
            "az://acct1/cont1/a",
            "az://acct1/cont2/b",
            "az://acct2/cont1/c",
            "https://acct1.blob.core.windows.net/cont1/d",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let g = group_uris_by_bucket(&uris, azure_group_key).expect("group");
        let targets: Vec<(&str, &str)> = g
            .iter()
            .map(|x| (x.target.0.as_str(), x.target.1.as_str()))
            .collect();
        assert_eq!(
            targets,
            vec![("acct1", "cont1"), ("acct1", "cont2"), ("acct2", "cont1")]
        );
        assert_eq!(g[0].keys, vec!["a", "d"]);
    }

    #[cfg(feature = "backend-azure")]
    #[test]
    fn azure_batch_with_unparseable_uri_is_an_error() {
        let uris: Vec<String> = ["az://acct1/cont1/a", "s3://nope/b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let err = group_uris_by_bucket(&uris, azure_group_key).unwrap_err();
        assert!(format!("{err:#}").contains("s3://nope/b"));
    }
}

// ============================================================================
// Issue #218: closing a compressed buffered writer must keep what the encoder
// already emitted.  The S3, Azure and GCS writers share the same shape: write_chunk
// moves the encoder's output into `buffer`, and the frame is closed before upload.
// The cloud upload itself needs a server (S3 is covered end to end in
// tests/test_issue208_per_endpoint_client.rs), so these check the bytes the writer
// would upload.
// ============================================================================
#[cfg(test)]
mod compressed_writer_seal_tests_218 {
    use super::*;

    /// 64 KiB stretches alternating incompressible and compressible bytes.
    fn mixed_payload(n: usize) -> Vec<u8> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(n);
        let mut random = true;
        while out.len() < n {
            let stretch = (64 * 1024).min(n - out.len());
            if random {
                for _ in 0..stretch {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    out.push((state >> 24) as u8);
                }
            } else {
                out.extend(b"the quick brown fox; ".iter().cycle().take(stretch));
            }
            random = !random;
        }
        out
    }

    /// One complete zstd frame that decodes to `data`.
    fn assert_complete_frame_of(sealed: &[u8], data: &[u8]) {
        let frame = zstd::zstd_safe::find_frame_compressed_size(sealed).unwrap_or_else(|e| {
            panic!(
                "the {} sealed bytes are not a complete zstd frame: {}",
                sealed.len(),
                zstd::zstd_safe::get_error_name(e)
            )
        });
        assert_eq!(frame, sealed.len(), "the frame must be the whole buffer");
        let plain = zstd::decode_all(sealed).expect("the sealed bytes decode");
        assert!(plain == data, "decoded bytes differ from the written bytes");
    }

    macro_rules! seal_test {
        ($name:ident, $writer:ty, $($cfg:meta),*) => {
            $(#[$cfg])*
            #[tokio::test]
            async fn $name() {
                for len in [32 * 1024, 6 * 1024 * 1024 + 123] {
                    let data = mixed_payload(len);
                    let mut w = <$writer>::new_with_compression(
                        "unused://bucket/key".to_string(),
                        CompressionConfig::zstd_default(),
                    );
                    for chunk in data.chunks(64 * 1024) {
                        w.write_chunk(chunk).await.expect("write_chunk");
                    }
                    w.seal_compression().expect("seal");
                    assert_complete_frame_of(&w.buffer, &data);
                    assert_eq!(w.compressed_bytes, w.buffer.len() as u64, "{len}");
                }
            }
        };
    }

    seal_test!(
        s3_writer_seal_keeps_the_early_blocks,
        S3BufferedWriter,
        cfg(all())
    );
    seal_test!(
        azure_writer_seal_keeps_the_early_blocks,
        AzureBufferedWriter,
        cfg(feature = "backend-azure")
    );
    seal_test!(
        gcs_writer_seal_keeps_the_early_blocks,
        GcsBufferedWriter,
        cfg(feature = "backend-gcs")
    );
}

// #231: `delete_prefix` must never select longer siblings.  S3, Azure and GCS
// all compute their listing prefix through `directory_prefix`.  Azure and GCS
// have no mock server, so the pure target-computation step each store's
// `delete_prefix` calls is tested here.
#[cfg(test)]
mod delete_prefix_listing_tests {
    use super::*;

    #[test]
    fn directory_prefix_appends_one_slash_to_a_non_empty_key() {
        assert_eq!(directory_prefix("foo"), "foo/");
        assert_eq!(directory_prefix("foo/"), "foo/");
        assert_eq!(directory_prefix(""), "");
        assert_eq!(directory_prefix("a//b"), "a//b/");
        assert_eq!(directory_prefix("/x"), "/x/");
        assert_eq!(directory_prefix("a/b"), "a/b/");
        // Literal: wildcard characters are not interpreted.
        assert_eq!(directory_prefix("foo*"), "foo*/");
    }

    #[cfg(feature = "backend-gcs")]
    #[test]
    fn gcs_delete_prefix_lists_only_under_the_directory() {
        let t = |u: &str| gcs_delete_prefix_target(u).expect(u);
        assert_eq!(t("gs://b/foo"), ("b".to_string(), "foo/".to_string()));
        assert_eq!(t("gs://b/foo/"), ("b".to_string(), "foo/".to_string()));
        assert_eq!(t("gs://b"), ("b".to_string(), String::new()));
        assert_eq!(t("gs://b/"), ("b".to_string(), String::new()));
        assert_eq!(t("gcs://b/foo"), ("b".to_string(), "foo/".to_string()));
        assert!(gcs_delete_prefix_target("gs://").is_err());
        assert!(gcs_delete_prefix_target("s3://b/foo").is_err());
    }

    #[cfg(feature = "backend-azure")]
    #[test]
    fn azure_delete_prefix_lists_only_under_the_directory() {
        let t = |u: &str| azure_delete_prefix_target(u).expect(u);
        let want = |p: &str| ("acct".to_string(), "c".to_string(), p.to_string());
        assert_eq!(t("az://acct/c/foo"), want("foo/"));
        assert_eq!(t("az://acct/c/foo/"), want("foo/"));
        assert_eq!(t("az://acct/c"), want(""));
        assert_eq!(t("az://acct/c/"), want(""));
        assert_eq!(t("https://acct.blob.core.windows.net/c/foo"), want("foo/"));
        assert!(azure_delete_prefix_target("s3://b/foo").is_err());
    }
}

// Issue #242: the Python `get_many` for `az://` / `gs://` must hand each store the full URI.
// A real Azure or GCS service is not available to unit tests, so the store is a fake that
// behaves like the real ones in the one way that matters: `get` only accepts a URI with a
// scheme (the real stores parse the account/container/bucket out of it) and records it.
#[cfg(test)]
mod get_many_with_stores_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    type Seen = Arc<Mutex<Vec<String>>>;

    /// How many reads the fake store has started and finished.
    #[derive(Default)]
    struct Reads {
        started: std::sync::atomic::AtomicUsize,
        completed: std::sync::atomic::AtomicUsize,
    }

    impl Reads {
        fn snapshot(&self) -> (usize, usize) {
            use std::sync::atomic::Ordering::SeqCst;
            (self.started.load(SeqCst), self.completed.load(SeqCst))
        }
    }

    struct UriOnlyStore {
        seen: Seen,
        reads: Arc<Reads>,
    }

    #[async_trait]
    impl ObjectStore for UriOnlyStore {
        async fn get(&self, uri: &str) -> Result<Bytes> {
            use std::sync::atomic::Ordering::SeqCst;
            self.seen.lock().unwrap().push(uri.to_string());
            self.reads.started.fetch_add(1, SeqCst);
            if !uri.contains("://") {
                bail!("invalid object URI '{uri}': missing scheme");
            }
            // A URI ending in "bad" fails at once; one ending in "slow" takes a while.
            if uri.ends_with("bad") {
                bail!("backend refused {uri}");
            }
            if uri.ends_with("panic") {
                panic!("injected panic reading {uri}");
            }
            if uri.ends_with("slow") {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            }
            self.reads.completed.fetch_add(1, SeqCst);
            Ok(Bytes::from(format!("body-of:{uri}")))
        }
        async fn get_range(&self, _: &str, _: u64, _: Option<u64>) -> Result<Bytes> {
            bail!("unused")
        }
        async fn put(&self, _: &str, _: Bytes) -> Result<()> {
            bail!("unused")
        }
        async fn put_multipart(&self, _: &str, _: Bytes, _: Option<usize>) -> Result<()> {
            bail!("unused")
        }
        async fn list(&self, _: &str, _: bool) -> Result<Vec<String>> {
            bail!("unused")
        }
        fn list_stream<'a>(
            &'a self,
            _: &'a str,
            _: bool,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            Box::pin(futures::stream::empty())
        }
        async fn stat(&self, _: &str) -> Result<ObjectMetadata> {
            bail!("unused")
        }
        fn list_objects_with_prefix<'a>(
            &'a self,
            _: &'a str,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            Box::pin(futures::stream::empty())
        }
        async fn delete(&self, _: &str) -> Result<()> {
            bail!("unused")
        }
        async fn delete_batch(&self, _: &[String]) -> Result<()> {
            bail!("unused")
        }
        async fn delete_prefix(&self, _: &str) -> Result<()> {
            bail!("unused")
        }
        async fn create_container(&self, _: &str) -> Result<()> {
            bail!("unused")
        }
        async fn delete_container(&self, _: &str) -> Result<()> {
            bail!("unused")
        }
        async fn get_writer(&self, _: &str) -> Result<Box<dyn ObjectWriter>> {
            bail!("unused")
        }
    }

    fn factory(
        seen: &Seen,
    ) -> impl Fn(&str) -> Result<Arc<dyn ObjectStore>> + Send + Sync + 'static {
        counting_factory(seen, &Arc::default())
    }

    fn counting_factory(
        seen: &Seen,
        reads: &Arc<Reads>,
    ) -> impl Fn(&str) -> Result<Arc<dyn ObjectStore>> + Send + Sync + 'static {
        let seen = Arc::clone(seen);
        let reads = Arc::clone(reads);
        move |_uri: &str| {
            Ok(Arc::new(UriOnlyStore {
                seen: Arc::clone(&seen),
                reads: Arc::clone(&reads),
            }) as Arc<dyn ObjectStore>)
        }
    }

    #[tokio::test]
    async fn azure_and_gcs_uris_reach_the_store_unchanged_and_come_back_in_order() {
        let uris: Vec<String> = (0..12)
            .flat_map(|i| {
                [
                    format!("az://acct/container/dir/obj{i}.bin"),
                    format!("gs://bucket/dir/obj{i}.bin"),
                ]
            })
            .collect();
        let seen: Seen = Arc::default();
        let got = get_many_with_stores(uris.clone(), 3, factory(&seen))
            .await
            .expect("every full URI is accepted by the store");

        assert_eq!(got.len(), uris.len());
        for (i, (uri, bytes)) in got.iter().enumerate() {
            assert_eq!(uri, &uris[i], "results stay in input order");
            assert_eq!(bytes.as_ref(), format!("body-of:{uri}").as_bytes());
        }
        let mut seen = seen.lock().unwrap().clone();
        let mut want = uris;
        seen.sort();
        want.sort();
        assert_eq!(
            seen, want,
            "the store was handed the full URIs, not bare keys"
        );
    }

    #[tokio::test]
    async fn a_store_error_names_the_failing_read() {
        let uris = vec!["az://acct/container/ok".to_string()];
        let failing = |_: &str| -> Result<Arc<dyn ObjectStore>> { bail!("no such backend") };
        let err = get_many_with_stores(uris, 1, failing).await.unwrap_err();
        assert!(format!("{err:#}").contains("no such backend"), "{err:#}");
    }

    // Issue #242 review: an error used to return at once and drop the JoinHandles, so the
    // reads still in flight (and the ones not yet started) kept running after the call had
    // returned. Like the S3 path (#148 3.2) the call now waits for the reads already started
    // before it returns the first error, and starts no more after the first error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn no_read_runs_after_get_many_returns_an_error() {
        let seen: Seen = Arc::default();
        let reads: Arc<Reads> = Arc::default();
        let mut uris = vec!["az://acct/c/first-bad".to_string()];
        uris.extend((0..8).map(|i| format!("az://acct/c/obj{i}-slow")));

        let err = get_many_with_stores(uris, 3, counting_factory(&seen, &reads))
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("az://acct/c/first-bad") && message.contains("backend refused"),
            "the error names the failing URI and keeps the cause: {message}"
        );

        let at_return = reads.snapshot();
        tokio::time::sleep(std::time::Duration::from_millis(900)).await;
        let later = reads.snapshot();
        assert_eq!(
            at_return, later,
            "RED: reads started/completed changed after get_many returned: \
             {at_return:?} -> {later:?}"
        );
        assert_eq!(
            at_return.0,
            at_return.1 + 1,
            "every read that started finished, except the one that failed: {at_return:?}"
        );
    }

    // The reads that have not started when a read fails are skipped, not downloaded. Made
    // deterministic: one permit and a current-thread runtime, so the tasks run in spawn order
    // and the first (failing) read finishes before any other acquires the permit.
    #[tokio::test]
    async fn reads_that_have_not_started_are_skipped_after_a_failure() {
        let seen: Seen = Arc::default();
        let reads: Arc<Reads> = Arc::default();
        let mut uris = vec!["az://acct/c/first-bad".to_string()];
        uris.extend((0..6).map(|i| format!("az://acct/c/obj{i}-slow")));

        let err = get_many_with_stores(uris, 1, counting_factory(&seen, &reads))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("first-bad"), "{err:#}");
        assert_eq!(
            reads.snapshot(),
            (1, 0),
            "RED: reads started after the first failure (started, completed)"
        );
    }

    // A panicking read is an error that still names its URI.
    #[tokio::test]
    async fn a_panicking_read_is_reported_with_its_uri() {
        let seen: Seen = Arc::default();
        let uris = vec![
            "az://acct/c/fine".to_string(),
            "az://acct/c/the-one-panic".to_string(),
        ];
        let err = get_many_with_stores(uris, 2, factory(&seen))
            .await
            .unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("GET az://acct/c/the-one-panic"),
            "RED: the URI is missing from: {message}"
        );
    }

    // A URI that names no object is refused up front, before any store is built or any read
    // starts (get_many used to say "no key specified in URI" for these).
    #[tokio::test]
    async fn uris_without_an_object_key_are_refused_before_any_read() {
        for bad in [
            "az://acct",
            "az://acct/",
            "az://acct/container",
            "az://acct/container/",
            "gs://bucket",
            "gs://bucket/",
            "gcs://bucket",
            "https://acct.blob.core.windows.net/container",
            "https://acct.blob.core.windows.net/container/",
        ] {
            let seen: Seen = Arc::default();
            let uris = vec!["az://acct/container/fine".to_string(), bad.to_string()];
            let err = get_many_with_stores(uris, 2, factory(&seen))
                .await
                .expect_err(bad);
            let message = format!("{err:#}");
            assert!(
                message.contains("no key specified") && message.contains(bad),
                "RED: {bad} was not refused as keyless: {message}"
            );
            assert!(
                seen.lock().unwrap().is_empty(),
                "RED: a read started although {bad} was keyless"
            );
        }
        // Keys that look odd but are keys stay accepted.
        for ok in [
            "az://acct/container/k",
            "az://acct/container/dir/",
            "az://acct/container//k",
            "gs://bucket/k",
            "gs://bucket//k",
            "https://acct.blob.core.windows.net/container/k",
        ] {
            let seen: Seen = Arc::default();
            get_many_with_stores(vec![ok.to_string()], 1, factory(&seen))
                .await
                .unwrap_or_else(|e| panic!("{ok} was refused: {e:#}"));
        }
    }

    #[tokio::test]
    async fn zero_max_in_flight_is_refused() {
        let seen: Seen = Arc::default();
        let err = get_many_with_stores(vec!["gs://b/k".to_string()], 0, factory(&seen))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("max_in_flight"), "{err:#}");
        assert!(seen.lock().unwrap().is_empty());
    }
}

// Blue review 2026-10-15: a size cached by `get` must not outlive a write or delete through the
// same store. The Azure behaviour is tested end to end against a fake Blob service
// (tests/azure_stale_size.rs). GCS cannot run offline: its client needs Google credentials and a
// bucket-layout RPC before any request, so these tests drive the store's own size bookkeeping
// (what `put`/`delete*` call) directly, and pin by source that every mutating method calls it.
#[cfg(all(test, feature = "backend-gcs"))]
mod gcs_size_cache_hygiene_tests {
    use super::*;

    async fn cached(store: &GcsObjectStore, uri: &str) -> Option<u64> {
        store.size_cache.get(&gcs_size_key(uri)).await
    }

    #[tokio::test]
    async fn forgetting_a_size_removes_only_that_object_whichever_scheme_spelling_was_used() {
        let store = GcsObjectStore::new();
        for (uri, size) in [
            ("gs://b/dir/a", 1),
            ("gs://b/dir/b", 2),
            ("gs://b/other", 3),
        ] {
            store.size_cache.put(gcs_size_key(uri), size).await;
        }
        // `gcs://` and `gs://` name the same object.
        store.forget_sizes(["gcs://b/dir/a"]).await;
        assert_eq!(
            cached(&store, "gs://b/dir/a").await,
            None,
            "RED: not forgotten"
        );
        assert_eq!(cached(&store, "gs://b/dir/b").await, Some(2));
        store
            .forget_sizes(["gs://b/dir/b", "gs://b/never-cached"])
            .await;
        assert_eq!(cached(&store, "gs://b/dir/b").await, None);
        assert_eq!(
            cached(&store, "gs://b/other").await,
            Some(3),
            "a neighbour keeps its size"
        );
    }

    #[test]
    fn size_keys_are_canonical() {
        assert_eq!(gcs_size_key("gcs://b/o/p"), "gs://b/o/p");
        assert_eq!(gcs_size_key("gs://b/o/p"), "gs://b/o/p");
    }

    #[test]
    fn every_mutating_method_forgets_the_sizes_it_makes_stale() {
        let src = include_str!("object_store.rs");
        let start = src
            .find("impl ObjectStore for GcsObjectStore {")
            .expect("impl");
        let body = &src[start..src[start..].find("\n}\n").expect("end") + start];
        for sig in [
            "async fn put(",
            "async fn put_multipart(",
            "async fn delete(",
            "async fn delete_batch(",
            "async fn delete_prefix(",
        ] {
            let a = body.find(sig).unwrap_or_else(|| panic!("{sig} not found"));
            let end = body[a..].find("\n    }\n").expect("end of method") + a;
            assert!(
                body[a..end].contains("forget_sizes("),
                "RED: {sig} leaves cached sizes of changed objects"
            );
        }
    }

    #[test]
    fn the_range_engine_is_guarded_when_it_uses_a_cached_size() {
        let src = include_str!("object_store.rs");
        let start = src
            .find("impl ObjectStore for GcsObjectStore {")
            .expect("impl");
        let body = &src[start..];
        let a = body.find("async fn get(").expect("get");
        let get = &body[a..body[a..].find("\n    }\n").expect("end") + a];
        assert!(get.contains("get_with_range_engine(uri, object_size, cached.is_some())"));
        assert!(
            get.contains("fresh_object_size(uri)"),
            "no fresh-size retry"
        );
    }
}
