// src/object_store_arrow.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::checksum::Crc32c;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
// Note: Futures stream traits not needed for simple implementation
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use url::{Position, Url};

use crate::delete_contract::{DELETE_REQUESTS_IN_FLIGHT, DELETE_REQUEST_KEYS};

use crate::object_store::{
    CompressionConfig,
    ObjectMetadata, // alias to s3_utils::ObjectStat
    ObjectStore as S3DlioObjectStore,
    ObjectWriter as S3DlioObjectWriter,
    WriterOptions,
};

use aos::{
    aws::AmazonS3Builder, path::Path, GetOptions, GetRange, ObjectStore as AosObjectStore,
    ObjectStoreExt as _, PutPayload,
};
use object_store as aos;

// ----------------------------------------------------------------------------
// Adapter object store
// ----------------------------------------------------------------------------

/// Arrow-backed implementation.
/// Stateless: we re-parse the URL each call to honor full-URI semantics in s3dlio.
pub struct ArrowObjectStore;

impl Default for ArrowObjectStore {
    fn default() -> Self {
        Self
    }
}

impl ArrowObjectStore {
    pub fn new() -> Self {
        Self
    }

    #[inline]
    pub fn boxed() -> Box<dyn S3DlioObjectStore> {
        Box::new(Self)
    }

    /// parse a full URI naming ONE OBJECT into (store, path) using explicit S3 configuration.
    fn parse(uri: &str) -> Result<(Arc<dyn AosObjectStore>, Path, Url)> {
        Self::parse_as(uri, false)
    }

    /// The shared parser; `is_prefix` is true when the URI names a listing prefix.
    fn parse_as(uri: &str, is_prefix: bool) -> Result<(Arc<dyn AosObjectStore>, Path, Url)> {
        let url = Url::parse(uri).context("invalid URI")?;

        match url.scheme() {
            "s3" => {
                let (bucket, path) = s3_target(&url, uri, is_prefix)?;

                // Build S3 client explicitly to control credentials
                let mut builder = AmazonS3Builder::new().with_bucket_name(&bucket);

                // Use credentials from environment
                if let (Ok(access_key), Ok(secret_key)) = (
                    std::env::var("AWS_ACCESS_KEY_ID"),
                    std::env::var("AWS_SECRET_ACCESS_KEY"),
                ) {
                    builder = builder
                        .with_access_key_id(&access_key)
                        .with_secret_access_key(&secret_key);
                } else {
                    bail!("AWS credentials not found in environment");
                }

                // Set region
                if let Ok(region) =
                    std::env::var("AWS_DEFAULT_REGION").or_else(|_| std::env::var("AWS_REGION"))
                {
                    builder = builder.with_region(&region);
                }

                // Set custom endpoint if provided
                if let Some(endpoint) = crate::s3_client::s3_endpoint_url_from_env() {
                    builder = builder.with_endpoint(&endpoint);
                    // For custom endpoints, we often need allow_http
                    if endpoint.starts_with("http://") {
                        builder = builder.with_allow_http(true);
                    }
                }

                let store = builder.build().context("Failed to build S3 client")?;
                Ok((Arc::new(store), path, url))
            }
            _ => {
                // For non-S3 URLs, fall back to generic parse_url
                let (store, path) =
                    aos::parse_url(&url).context("object_store::parse_url failed")?;
                Ok((Arc::from(store), path, url))
            }
        }
    }

    /// convenience for prefix listing: if the URI is a "directory-like" prefix,
    /// just parse it; listing handles both object prefixes and virtual directories.
    fn parse_prefix(prefix: &str) -> Result<(Arc<dyn AosObjectStore>, Path, Url)> {
        Self::parse_as(prefix, true)
    }

    /// Build the full URI of a listed `location` from the URL the listing was requested with.
    ///
    /// Every location `object_store` returns is relative to the STORE ROOT, not to the
    /// listing URL: for `s3://b/dir/` the store is the bucket `b` and the location of
    /// `dir/x` is `dir/x`, which already contains the listing prefix `dir` (`base`). The
    /// URL's own key path must therefore not be appended again (issue #222). What the
    /// store root adds in front of the key is the part of the URL path before `base`:
    /// nothing for `s3://`, `gs://`, `az://`, `memory://` (the bucket or container is the
    /// URL's authority), but the bucket for a path-style endpoint such as
    /// `https://s3.us-east-1.amazonaws.com/bucket/dir/`.
    fn rebuild_full_uri(base_url: &Url, base: &Path, location: &Path) -> String {
        // `scheme://[userinfo@]host[:port]`, kept exactly as given (`az://container@account`).
        let authority = &base_url[..Position::BeforePath];
        // The root of the store is the URL path minus the trailing `base.parts()` segments.
        let url_segments: Vec<&str> = base_url
            .path_segments()
            .map(|segments| segments.filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();
        let root_len = url_segments.len().saturating_sub(base.parts().count());

        let mut rebuilt = String::from(authority);
        rebuilt.push('/');
        for segment in &url_segments[..root_len] {
            rebuilt.push_str(segment);
            rebuilt.push('/');
        }
        // Arrow locations never start with '/'.
        rebuilt.push_str(location.as_ref());
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

/// The bucket and the key (or listing prefix) an `s3://bucket/key` URL names.
fn s3_target(url: &Url, uri: &str, is_prefix: bool) -> Result<(String, Path)> {
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("No bucket in S3 URL"))?;
    let path = key_path(url.path(), is_prefix).with_context(|| format!("cannot address {uri}"))?;
    Ok((bucket.to_string(), path))
}

/// The `object_store` [`Path`] for the key in an `s3://` URI's path component.
///
/// Keys are used exactly as written (issue #198): the first `/` of `url_path` is the
/// separator between the bucket and the key, and everything after it is the key.
/// `object_store::path::Path` cannot hold a key with an empty segment: `Path::from` drops
/// the empty segments, so the objects `/x`, `x`, and `a//b`, `a/b` would silently collapse
/// onto one another. Such a key is therefore refused with an error that names the
/// problem, never mapped onto a different object.
///
/// A listing prefix (`is_prefix`) may end in one `/`, the directory delimiter. An object
/// key may not (`dir/` is a different object than `dir`).
fn key_path(url_path: &str, is_prefix: bool) -> Result<Path> {
    let key = url_path.strip_prefix('/').unwrap_or(url_path);
    let body = match key.strip_suffix('/') {
        Some(dir) if is_prefix && !dir.is_empty() => dir,
        _ => key,
    };
    if !body.is_empty() && body.split('/').any(str::is_empty) {
        bail!(
            "the key {key:?} has an empty path segment (a leading '/', a '//', or a trailing \
             '/' on an object key), which the Arrow object_store backend cannot represent; \
             using it would address a different object"
        );
    }
    Ok(Path::from(body))
}

/// The entries under `base` in `store` as a stream of full URIs rebuilt from `url`, the URL
/// the listing was requested with.
///
/// The location `base` itself is not an entry under it (see [`delete_under`]) and is
/// skipped.
///
/// `recursive` lists every object under `base` at any depth and no directory entries
/// (issue #245), streamed as the backend returns its pages: nothing is buffered, so a
/// listing of any size runs in constant memory. Otherwise only the direct children are
/// listed, objects first and then the directory entries (common prefixes):
/// `list_with_delimiter` requests the pages one at a time but returns only after all of
/// them have arrived, so memory is O(direct children) and nothing is yielded before the
/// last page. `object_store` matches a prefix per path segment, so `p` selects `p/...`
/// and never `pfoo/...` or `p2/...`. Folder marker objects appear under their stripped
/// name (`p/sub/` as `s3://b/p/sub`) and cannot be addressed afterwards.
fn list_stream_under(
    store: Arc<dyn AosObjectStore>,
    base: Path,
    url: Url,
    recursive: bool,
) -> futures::stream::BoxStream<'static, Result<String>> {
    use futures::StreamExt as _;

    if recursive {
        // An empty prefix is the whole store (the bucket root).
        let prefix = (!base.as_ref().is_empty()).then(|| base.clone());
        store
            .list(prefix.as_ref())
            .filter_map(move |entry| {
                std::future::ready(match entry {
                    Ok(meta) if meta.location == base => None,
                    Ok(meta) => Some(Ok(ArrowObjectStore::rebuild_full_uri(
                        &url,
                        &base,
                        &meta.location,
                    ))),
                    Err(e) => Some(Err(anyhow::Error::new(e))),
                })
            })
            .boxed()
    } else {
        async_stream::try_stream! {
            let result = store.list_with_delimiter(Some(&base)).await?;
            for obj in result.objects {
                if obj.location != base {
                    yield ArrowObjectStore::rebuild_full_uri(&url, &base, &obj.location);
                }
            }
            for prefix in result.common_prefixes {
                if prefix != base {
                    yield ArrowObjectStore::rebuild_full_uri(&url, &base, &prefix);
                }
            }
        }
        .boxed()
    }
}

/// [`list_stream_under`], collected.
async fn list_under(
    store: Arc<dyn AosObjectStore>,
    base: Path,
    url: Url,
    recursive: bool,
) -> Result<Vec<String>> {
    use futures::TryStreamExt as _;

    list_stream_under(store, base, url, recursive)
        .try_collect()
        .await
}

/// Delete every object under `base` in `store`, at any depth (issue #232).
///
/// The location `base` itself is never deleted: S3 returns the folder marker object `p/`
/// as the location `p` (`object_store` strips the trailing '/'), which is the prefix, not
/// an entry under it, and deleting it would delete the unrelated object `p`. Folder
/// marker objects (`p/`, and `p/sub/` which lists as `p/sub`) cannot be addressed through
/// `object_store`: they may remain, and a delete of the location `p/sub` reports success
/// for the key `p/sub` (which does not exist) while `p/sub/` stays.
///
/// `object_store` evaluates a prefix on a path-segment basis: `p` covers `p/a` and
/// `p/b/c` but never `pfoo/x` or `p2/y`, so an unrelated sibling is never selected. The
/// listing is recursive (`list_with_delimiter` returns the immediate children only) and
/// is consumed as a stream, never collected whole.
///
/// The S3 store in `object_store` has no single-object DELETE: every delete, even of one
/// key, is a `DeleteObjects` request (`POST ?delete`), and `delete_stream` sends one per
/// 1000 keys. Deleting key by key would therefore cost a request per key. The keys are
/// grouped here into chunks of [`DELETE_REQUEST_KEYS`] (the `DeleteObjects` limit), each
/// chunk is one `delete_stream` call, and [`DELETE_REQUESTS_IN_FLIGHT`] chunks run at a
/// time.
///
/// **The bound of 10 requests in flight is S3-specific.** It counts chunks, and one chunk
/// is exactly one request only for S3's `DeleteObjects`. In the pinned `object_store`
/// 0.13.2 the other stores split a chunk further inside `delete_stream`: Azure sends
/// batch requests of 256 keys (up to four per chunk, so up to 40 in flight), and the
/// GCS and plain-HTTP stores send one DELETE per object, 10 at a time per chunk (up to
/// 100 in flight). S3 with bulk delete disabled does the same per-object delete (20 per
/// chunk, up to 200). Only the S3 bound is a request bound; for the others it multiplies.
/// An endpoint without `DeleteObjects` (Google Cloud Storage's S3 interoperability
/// endpoint) is NOT supported for delete by this backend: every request fails and so does
/// the call, naming the keys. Use `gs://` for GCS.
///
/// A failure does not stop the others. A key counts as failed unless the store confirmed
/// it deleted, so a request that fails as a whole counts every key in it. A listing error
/// ends the listing, counts as one failure and fails the call ("listing incomplete").
async fn delete_under(store: &dyn AosObjectStore, base: &Path) -> Result<()> {
    delete_under_in_chunks(store, base, DELETE_REQUEST_KEYS).await
}

/// What deleting one chunk of keys (or finding the listing broken) came to.
#[derive(Default)]
struct DeleteOutcome {
    deleted: usize,
    failed: usize,
    failed_keys: Vec<String>,
    causes: Vec<String>,
    listing_error: Option<String>,
}

/// Delete one chunk of keys with one `delete_stream` call.
async fn delete_chunk(store: &dyn AosObjectStore, keys: Vec<Path>) -> DeleteOutcome {
    use futures::StreamExt as _;

    let mut results =
        store.delete_stream(futures::stream::iter(keys.clone().into_iter().map(Ok)).boxed());
    let mut confirmed: std::collections::HashSet<Path> = std::collections::HashSet::new();
    let mut outcome = DeleteOutcome::default();
    while let Some(result) = results.next().await {
        match result {
            Ok(path) => {
                confirmed.insert(path);
            }
            Err(e) => {
                if outcome.causes.len() < MAX_NAMED_DELETE_FAILURES {
                    outcome.causes.push(e.to_string());
                }
            }
        }
    }
    for key in keys {
        if confirmed.contains(&key) {
            outcome.deleted += 1;
        } else {
            outcome.failed += 1;
            if outcome.failed_keys.len() < MAX_NAMED_DELETE_FAILURES {
                outcome.failed_keys.push(key.to_string());
            }
        }
    }
    outcome
}

/// [`delete_under`] with the number of keys per delete request given.
async fn delete_under_in_chunks(
    store: &dyn AosObjectStore,
    base: &Path,
    chunk_keys: usize,
) -> Result<()> {
    use futures::StreamExt as _;

    // An empty prefix is the whole store (the bucket root).
    let prefix = (!base.as_ref().is_empty()).then_some(base);
    let mut listing = store.list(prefix);
    // Chunks of keys, or the error that ended the listing.
    let chunks = async_stream::stream! {
        let mut chunk: Vec<Path> = Vec::new();
        while let Some(entry) = listing.next().await {
            match entry {
                Ok(meta) if meta.location == *base => {}
                Ok(meta) => {
                    chunk.push(meta.location);
                    if chunk.len() >= chunk_keys {
                        yield Ok(std::mem::take(&mut chunk));
                    }
                }
                Err(e) => {
                    if !chunk.is_empty() {
                        yield Ok(std::mem::take(&mut chunk));
                    }
                    yield Err(e.to_string());
                    return;
                }
            }
        }
        if !chunk.is_empty() {
            yield Ok(chunk);
        }
    };
    let outcomes = chunks
        .map(|chunk| async move {
            match chunk {
                Ok(keys) => delete_chunk(store, keys).await,
                Err(listing_error) => DeleteOutcome {
                    failed: 1,
                    listing_error: Some(listing_error),
                    ..Default::default()
                },
            }
        })
        .buffer_unordered(DELETE_REQUESTS_IN_FLIGHT);
    tokio::pin!(outcomes);

    let mut total = DeleteOutcome::default();
    while let Some(outcome) = outcomes.next().await {
        total.deleted += outcome.deleted;
        total.failed += outcome.failed;
        total.listing_error = total.listing_error.or(outcome.listing_error);
        for key in outcome.failed_keys {
            if total.failed_keys.len() < MAX_NAMED_DELETE_FAILURES {
                total.failed_keys.push(key);
            }
        }
        for cause in outcome.causes {
            if total.causes.len() < MAX_NAMED_DELETE_FAILURES {
                total.causes.push(cause);
            }
        }
    }
    if total.failed > 0 {
        let listing = match &total.listing_error {
            Some(e) => format!("; listing incomplete: {e}"),
            None => String::new(),
        };
        let keys = if total.failed_keys.is_empty() {
            String::new()
        } else {
            format!("; first failed keys: {}", total.failed_keys.join(", "))
        };
        bail!(
            "delete_prefix: {} failed under {:?} ({} deleted){listing}{keys}; first errors: {}",
            total.failed,
            base.as_ref(),
            total.deleted,
            total.causes.join("; ")
        );
    }
    Ok(())
}

/// How many individual delete errors `delete_prefix` quotes in its error.
const MAX_NAMED_DELETE_FAILURES: usize = 5;

/// Read `length` bytes from `offset` (to the end of the object if `length` is `None`).
async fn read_range(
    store: &dyn AosObjectStore,
    path: &Path,
    offset: u64,
    length: Option<u64>,
) -> Result<Bytes> {
    // A zero-length range cannot be expressed as an HTTP range header (the end would be
    // `offset - 1`); like the native backends, answer it as empty without a request.
    if length == Some(0) {
        return Ok(Bytes::new());
    }
    let range = match length {
        Some(len) => {
            let end = offset.checked_add(len).ok_or_else(|| {
                anyhow::anyhow!(
                    "range end overflow: offset={offset} + length={len} exceeds u64::MAX"
                )
            })?;
            GetRange::Bounded(offset..end)
        }
        None => GetRange::Offset(offset),
    };
    let opts = GetOptions {
        range: Some(range),
        ..Default::default()
    };
    let reader = store.get_opts(path, opts).await?;
    Ok(reader.bytes().await?)
}

// ----------------------------------------------------------------------------
// ObjectStore impl
// ----------------------------------------------------------------------------

#[async_trait]
impl S3DlioObjectStore for ArrowObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (store, path, _url) = Self::parse(uri)?;
        let reader = store.get(&path).await?;
        Ok(reader.bytes().await?)
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let (store, path, _url) = Self::parse(uri)?;
        read_range(store.as_ref(), &path, offset, length).await
    }

    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        let (store, path, _url) = Self::parse(uri)?;
        store.put(&path, PutPayload::from(data)).await?;
        Ok(())
    }

    // For now route multipart to Arrow's regular `put` (Arrow chooses optimal strategy internally).
    // You can switch to `WriteMultipart` later if you want strict part size / concurrency control.
    async fn put_multipart(&self, uri: &str, data: Bytes, _part_size: Option<usize>) -> Result<()> {
        self.put(uri, data).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let (store, base, url) = Self::parse_prefix(uri_prefix)?;
        list_under(store, base, url, recursive).await
    }

    // Same entries as `list`, yielded as the backend returns them: a recursive listing is
    // never buffered whole (issue #245 follow-up).
    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> Pin<Box<dyn futures::stream::Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            use futures::StreamExt as _;
            match Self::parse_prefix(uri_prefix) {
                Ok((store, base, url)) => {
                    let mut uris = list_stream_under(store, base, url, recursive);
                    while let Some(uri) = uris.next().await {
                        yield uri;
                    }
                }
                Err(e) => yield Err(e),
            }
        })
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

    // Validates every URI first: a batch with an unusable URI deletes nothing and names
    // each such URI. Then deletes one by one; a failure does not stop the others, and the
    // returned error names every URI not confirmed deleted.
    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        let mut parsed = Vec::with_capacity(uris.len());
        let mut invalid: Vec<String> = Vec::new();
        for uri in uris {
            match Self::parse(uri) {
                Ok((store, path, _url)) => parsed.push((uri, store, path)),
                Err(e) => invalid.push(format!("{uri} ({e:#})")),
            }
        }
        if !invalid.is_empty() {
            bail!(
                "delete_batch rejected, nothing deleted; unusable URIs: {}",
                invalid.join(", ")
            );
        }

        let mut failed: Vec<String> = Vec::new();
        let mut first_cause: Option<anyhow::Error> = None;
        for (uri, store, path) in parsed {
            if let Err(e) = store.delete(&path).await {
                failed.push(uri.clone());
                if first_cause.is_none() {
                    first_cause = Some(anyhow::Error::new(e));
                }
            }
        }
        match first_cause {
            None => Ok(()),
            Some(cause) => Err(cause.context(format!(
                "delete_batch: {} of {} objects not confirmed deleted: {}",
                failed.len(),
                uris.len(),
                failed.join(", ")
            ))),
        }
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let (store, base, _url) = Self::parse_prefix(uri_prefix)?;
        delete_under(store.as_ref(), &base).await
    }

    /// No-op: an object store has no directories (issue #229). A prefix exists once
    /// objects are written under it, so there is nothing to create. (A marker object
    /// is also impossible here: `object_store::path::Path` cannot represent a key
    /// ending in `/`.) The URI is validated; NO request is sent.
    async fn mkdir(&self, uri: &str) -> Result<()> {
        Url::parse(uri).context("invalid URI")?;
        Ok(())
    }

    /// No-op, `recursive` or not: an object store has no directories (issue #229), so
    /// there is nothing to remove. The URI is validated; NO request is sent and
    /// nothing is deleted. Use [`S3DlioObjectStore::delete_prefix`] to delete objects.
    async fn rmdir(&self, uri: &str, _recursive: bool) -> Result<()> {
        Url::parse(uri).context("invalid URI")?;
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
        ArrowWriter::open(uri, CompressionConfig::None).await
    }

    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn S3DlioObjectWriter>> {
        ArrowWriter::open(uri, compression).await
    }

    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn S3DlioObjectWriter>> {
        let compression = options.compression.unwrap_or(CompressionConfig::None);
        ArrowWriter::open(uri, compression).await
    }

    // For now, fall back to default optimized paths (S3-specific concurrency doesn't apply here).
    // If you want, you can later implement `get_ranges` coalescing for Arrow to speed large reads.
    async fn get_optimized(&self, uri: &str) -> Result<Bytes> {
        self.get(uri).await
    }

    async fn get_range_optimized(
        &self,
        uri: &str,
        offset: u64,
        length: Option<u64>,
        _chunk_size: Option<usize>,
        _max_concurrency: Option<usize>,
    ) -> Result<Bytes> {
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
    hasher: Crc32c,
    bytes_written: u64,
    compressed_bytes: u64,

    // compression
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,

    // finalized flag
    finalized: bool,
}

/// The path a compressing Arrow writer stores `path` under: the one suffix rule
/// ([`CompressionConfig::stored_name`]) that the checkpoint manifest names also use.
fn compressed_object_path(path: Path, compression: &CompressionConfig) -> Path {
    if compression.is_enabled() {
        Path::from(compression.stored_name(path.as_ref()))
    } else {
        path
    }
}

impl ArrowWriter {
    async fn open(
        dest_uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn S3DlioObjectWriter>> {
        let (store, path, _url) = ArrowObjectStore::parse(dest_uri)?;

        // Adjust path for compression extension
        let final_path = compressed_object_path(path, &compression);

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
            hasher: Crc32c::new(),
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

    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn compressed_bytes(&self) -> u64 {
        self.compressed_bytes
    }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compression(&self) -> CompressionConfig {
        self.compression
    }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 {
            1.0
        } else {
            self.compressed_bytes as f64 / self.bytes_written as f64
        }
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
            unsafe {
                std::env::set_var("AWS_DEFAULT_REGION", region);
            }
        }
    }

    // Log what we found for debugging (without exposing secrets)
    if std::env::var("AWS_ACCESS_KEY_ID").is_ok() && std::env::var("AWS_SECRET_ACCESS_KEY").is_ok()
    {
        println!("🔧 Arrow backend: Using credentials from environment");
    }
    if let Some(endpoint) = crate::s3_client::s3_endpoint_url_from_env() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    // R10: a key that already ends in the compression extension is stored as it is, so the
    // checkpoint manifest (same rule) names the stored object; any other key gains it once.
    #[test]
    fn compressed_object_path_appends_the_extension_once() {
        let zstd = CompressionConfig::zstd_default();
        for (key, stored) in [
            ("dir/plain.bin", "dir/plain.bin.zst"),
            ("noext", "noext.zst"),
            ("dir/custom.zst", "dir/custom.zst"),
        ] {
            let got = compressed_object_path(Path::from(key), &zstd);
            assert_eq!(got, Path::from(stored), "{key}");
            assert_eq!(got.as_ref(), zstd.stored_name(key), "{key}");
        }
        let none = CompressionConfig::None;
        assert_eq!(
            compressed_object_path(Path::from("dir/custom.zst"), &none),
            Path::from("dir/custom.zst")
        );
    }

    // The factory must hand back a store usable through the current trait object.
    #[test]
    fn factory_returns_trait_object() {
        assert!(store_for_uri("s3://bucket/key").is_ok());
    }

    // An unusable URI anywhere in the batch rejects the whole batch (nothing deleted)
    // and names the URI; this is decided before any request, so no network is needed.
    #[tokio::test]
    async fn delete_batch_rejects_unusable_uri_and_names_it() {
        let store = ArrowObjectStore::new();
        let uris = vec!["not a uri".to_string()];
        let err = store.delete_batch(&uris).await.expect_err("must reject");
        let msg = format!("{err:#}");
        assert!(msg.contains("not a uri"), "error must name the URI: {msg}");
        assert!(msg.contains("nothing deleted"), "{msg}");
    }

    async fn memory_store(contents: &'static [u8]) -> (aos::memory::InMemory, Path) {
        let store = aos::memory::InMemory::new();
        let path = Path::from("obj.bin");
        store
            .put(&path, PutPayload::from(Bytes::from_static(contents)))
            .await
            .expect("put fixture");
        (store, path)
    }

    // A zero-length range is empty, at offset 0 and elsewhere, and makes no request
    // (the underlying range header cannot express it).
    #[tokio::test]
    async fn read_range_zero_length_is_empty() {
        let (store, path) = memory_store(b"0123456789").await;
        for offset in [0u64, 5, 10, 1000, u64::MAX] {
            let got = read_range(&store, &path, offset, Some(0)).await;
            assert_eq!(got.expect("zero-length range").len(), 0, "offset {offset}");
        }
    }

    // offset + length beyond u64::MAX is an error, never a panic or a wrapped range.
    #[tokio::test]
    async fn read_range_offset_plus_length_overflow_is_an_error() {
        let (store, path) = memory_store(b"0123456789").await;
        let err = read_range(&store, &path, u64::MAX, Some(2))
            .await
            .expect_err("overflow must be rejected");
        assert!(format!("{err:#}").contains("overflow"), "{err:#}");
        assert!(read_range(&store, &path, u64::MAX - 1, Some(5))
            .await
            .is_err());
    }

    // Normal ranges still return the right bytes.
    #[tokio::test]
    async fn read_range_returns_requested_bytes() {
        let (store, path) = memory_store(b"0123456789").await;
        let got = read_range(&store, &path, 2, Some(3))
            .await
            .expect("bounded");
        assert_eq!(&got[..], b"234");
        let got = read_range(&store, &path, 7, None)
            .await
            .expect("open-ended");
        assert_eq!(&got[..], b"789");
    }

    // -- issue #198: keys are used exactly as written ------------------------------------

    // The reproduction: `object_store::path::Path` cannot hold a key with an empty
    // segment. `Path::from` silently drops the empty segments, so `/x` IS `x` and `a//b`
    // IS `a/b`: two different S3 objects collapse onto one. `Path::parse` refuses them.
    #[tokio::test]
    async fn object_store_path_drops_empty_segments_so_it_cannot_name_such_keys() {
        assert_eq!(Path::from("/x"), Path::from("x"));
        assert_eq!(Path::from("a//b"), Path::from("a/b"));
        assert_eq!(Path::from("a/b/"), Path::from("a/b"));
        // `Path::parse` does not help: it refuses `a//b` and trims the leading `/` of `/x`.
        assert!(Path::parse("a//b").is_err());
        assert_eq!(Path::parse("/x").expect("parse trims"), Path::from("x"));

        // The consequence at the store level: writing the object `/x` is reading `x`.
        let store = aos::memory::InMemory::new();
        store
            .put(
                &Path::from("/x"),
                PutPayload::from(Bytes::from_static(b"slash-x")),
            )
            .await
            .expect("put");
        let got = store
            .get(&Path::from("x"))
            .await
            .expect("`x` aliases `/x` in object_store")
            .bytes()
            .await
            .expect("bytes");
        assert_eq!(&got[..], b"slash-x");
    }

    #[test]
    fn key_path_keeps_representable_keys_exactly() {
        // `s3://b/<key>`: the URL path is `/<key>`; one `/` is the separator.
        assert_eq!(key_path("/x", false).expect("x"), Path::from("x"));
        assert_eq!(key_path("/a/b", false).expect("a/b"), Path::from("a/b"));
        assert_eq!(key_path("/x.bin", false).expect("x.bin").as_ref(), "x.bin");
        // The bucket itself: no key.
        assert_eq!(key_path("", true).expect("bucket"), Path::from(""));
        assert_eq!(key_path("/", true).expect("bucket"), Path::from(""));
        // A listing prefix may end in the directory delimiter.
        assert_eq!(key_path("/dir/", true).expect("dir/"), Path::from("dir"));
        assert_eq!(key_path("/a/b/", true).expect("a/b/"), Path::from("a/b"));
    }

    #[test]
    fn key_path_rejects_keys_object_store_cannot_represent() {
        for (url_path, why) in [
            ("//x", "key `/x`"),
            ("//", "key `/`"),
            ("/a//b", "key `a//b`"),
            ("///x", "key `//x`"),
            ("//a/b/", "prefix `/a/b/`"),
        ] {
            for is_prefix in [false, true] {
                let err = key_path(url_path, is_prefix)
                    .expect_err(&format!("{why} (prefix={is_prefix}) must be refused"));
                let msg = format!("{err:#}");
                assert!(msg.contains("empty path segment"), "{why}: {msg}");
            }
        }
        // A key that ends in `/` names a different object than the Path it would become.
        let err = key_path("/dir/", false).expect_err("object key `dir/` must be refused");
        assert!(format!("{err:#}").contains("empty path segment"), "{err:#}");
    }

    // Through the public surface: refused before any credential lookup or request.
    #[tokio::test]
    async fn operations_refuse_unrepresentable_keys_instead_of_addressing_another_object() {
        let store = ArrowObjectStore::new();
        for uri in ["s3://bucket//x", "s3://bucket/a//b", "s3://bucket///x"] {
            let msg = |e: anyhow::Error| format!("{e:#}");
            let all = [
                store.get(uri).await.map(|_| ()).map_err(msg),
                store
                    .get_range(uri, 0, Some(1))
                    .await
                    .map(|_| ())
                    .map_err(msg),
                store.put(uri, Bytes::from_static(b"d")).await.map_err(msg),
                store.stat(uri).await.map(|_| ()).map_err(msg),
                store.delete(uri).await.map_err(msg),
                store.list(uri, false).await.map(|_| ()).map_err(msg),
                store.delete_batch(&[uri.to_string()]).await.map_err(msg),
            ];
            for r in all {
                let m = r.expect_err(&format!("{uri} must be refused"));
                assert!(m.contains("empty path segment"), "{uri}: {m}");
                assert!(m.contains(uri), "the error must name the URI: {m}");
            }
        }
    }

    // The prefix/object split reaches the parser: `s3://b/dir/` is a valid LISTING prefix
    // (the most common call shape for `list` / `delete_prefix`) and an invalid OBJECT key.
    // The refusal tests above only prove errors; this one proves the success side of the
    // `parse_prefix` -> `parse_as(.., true)` wiring. Building the client needs credentials
    // but makes no request. Env use is serialised by the crate-wide test guard and restored
    // on drop.
    #[test]
    fn parse_prefix_accepts_a_trailing_slash_that_parse_refuses() {
        let _guard = crate::test_env::lock();
        let _env =
            crate::test_env::EnvRestore::capture(&["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]);
        unsafe {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-access-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret-key");
        }

        let prefix = ArrowObjectStore::parse_prefix("s3://b/dir/");
        let object = ArrowObjectStore::parse("s3://b/dir/");

        let (_store, path, _url) = prefix.expect("`s3://b/dir/` is a valid listing prefix");
        assert_eq!(path, Path::from("dir"));
        let err = object.expect_err("`s3://b/dir/` is not an object key");
        let msg = format!("{err:#}");
        assert!(msg.contains("empty path segment"), "{msg}");
        assert!(msg.contains("s3://b/dir/"), "{msg}");
    }

    // list_stream reports a listing failure as a single Err item, like `list`.
    #[tokio::test]
    async fn list_stream_yields_error_for_unusable_prefix() {
        let store = ArrowObjectStore::new();
        let items: Vec<Result<String>> = store.list_stream("not a uri", false).collect().await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_err());
    }

    // #229: mkdir / rmdir are no-ops on an object store -- Ok, no request (no store is
    // even built), the URI is only validated.
    #[tokio::test]
    async fn mkdir_and_rmdir_are_noops_and_validate_the_uri() {
        let store = ArrowObjectStore::new();
        for uri in ["s3://bucket/dir", "s3://bucket/dir/", "s3://bucket/"] {
            store.mkdir(uri).await.unwrap();
            store.rmdir(uri, true).await.unwrap();
            store.rmdir(uri, false).await.unwrap();
        }
        assert!(store.mkdir("not a uri").await.is_err());
        assert!(store.rmdir("not a uri", true).await.is_err());
    }

    // -- issues #222 and #232: listing URIs and recursive prefix delete -------------------

    /// An in-memory store holding an empty object at each key.
    async fn seeded(keys: &[&str]) -> aos::memory::InMemory {
        let store = aos::memory::InMemory::new();
        for key in keys {
            store
                .put(
                    &Path::from(*key),
                    PutPayload::from(Bytes::from_static(b"d")),
                )
                .await
                .expect("seed");
        }
        store
    }

    /// Every key left in `store`, sorted.
    async fn remaining(store: &aos::memory::InMemory) -> Vec<String> {
        let mut keys: Vec<String> = store
            .list(None)
            .map(|m| m.expect("list").location.to_string())
            .collect()
            .await;
        keys.sort();
        keys
    }

    /// The (prefix path, URL) the adapter derives from an `s3://` listing URI.
    fn s3_prefix(uri: &str) -> (Path, Url) {
        let url = Url::parse(uri).expect("url");
        let (_bucket, base) = s3_target(&url, uri, true).expect("s3_target");
        (base, url)
    }

    async fn s3_list(store: &aos::memory::InMemory, uri: &str, recursive: bool) -> Vec<String> {
        let (base, url) = s3_prefix(uri);
        let mut got = list_under(Arc::new(store.clone()), base, url, recursive)
            .await
            .expect("list");
        got.sort();
        got
    }

    // #222: the reproduction from the issue. `dir/x` listed through `s3://b/dir/` came back
    // as `s3://b/dir/dir/x`.
    #[tokio::test]
    async fn s3_listing_with_a_key_prefix_does_not_double_the_prefix() {
        let store = seeded(&["dir/x"]).await;
        for recursive in [false, true] {
            assert_eq!(
                s3_list(&store, "s3://b/dir/", recursive).await,
                vec!["s3://b/dir/x".to_string()],
                "recursive={recursive}"
            );
        }
    }

    #[tokio::test]
    async fn s3_listing_nested_prefixes_return_the_exact_key() {
        let store = seeded(&["a/b/c/x", "a/b/c/y", "a/b/other", "a/z"]).await;
        assert_eq!(
            s3_list(&store, "s3://b/a/b/c/", false).await,
            vec!["s3://b/a/b/c/x".to_string(), "s3://b/a/b/c/y".to_string()]
        );
        // One level up: the objects and the directory entry, each addressed by its full key.
        assert_eq!(
            s3_list(&store, "s3://b/a/b/", false).await,
            vec!["s3://b/a/b/c".to_string(), "s3://b/a/b/other".to_string()]
        );
    }

    // A prefix without a trailing slash is the exact name's directory (#213): it lists
    // `dir/...` and never `dir2/...`, `dir.bak/...` or the sibling object `dirx`.
    #[tokio::test]
    async fn s3_listing_a_prefix_without_trailing_slash_lists_its_directory_only() {
        let store = seeded(&["dir/x", "dir/sub/y", "dir2/w", "dir.bak/v", "dirx"]).await;
        let expected = vec!["s3://b/dir/sub".to_string(), "s3://b/dir/x".to_string()];
        assert_eq!(s3_list(&store, "s3://b/dir", false).await, expected);
        assert_eq!(s3_list(&store, "s3://b/dir/", false).await, expected);
    }

    #[tokio::test]
    async fn s3_listing_the_bucket_root_returns_keys_under_the_bucket() {
        let store = seeded(&["x", "dir/y"]).await;
        let expected = vec!["s3://b/dir".to_string(), "s3://b/x".to_string()];
        assert_eq!(s3_list(&store, "s3://b/", false).await, expected);
        assert_eq!(s3_list(&store, "s3://b", false).await, expected);
    }

    // The point of the issue: a listed URI must address the listed object.
    #[tokio::test]
    async fn s3_listed_uris_round_trip_to_the_listed_objects() {
        let store = seeded(&["dir/x", "dir/y", "a/b/c/z"]).await;
        for (listing, count) in [("s3://b/dir/", 2), ("s3://b/a/b/c/", 1)] {
            let uris = s3_list(&store, listing, false).await;
            assert_eq!(uris.len(), count);
            for uri in uris {
                let url = Url::parse(&uri).expect("url");
                let (_bucket, key) =
                    s3_target(&url, &uri, false).expect("a listed URI parses as an object");
                store
                    .get(&key)
                    .await
                    .unwrap_or_else(|e| panic!("{uri} does not address a listed object: {e}"));
            }
        }
    }

    // `rebuild_full_uri` for every URI shape `parse_url` serves: the URL's own path
    // (`base`) is already part of each listed location, so only the part of the URL path
    // BEFORE the key (the store root, e.g. `/bucket` of a path-style https endpoint) is kept.
    #[test]
    fn rebuild_full_uri_is_correct_for_every_scheme() {
        let key = |s: &str| Path::from(s);
        let cases: [(&str, &str, &str, &str); 10] = [
            // (listing URL, key prefix the store sees, listed location, expected URI)
            ("s3://b/dir/", "dir", "dir/x", "s3://b/dir/x"),
            ("s3://b/", "", "x", "s3://b/x"),
            ("s3://b", "", "x", "s3://b/x"),
            ("s3a://b/a/b/", "a/b", "a/b/x", "s3a://b/a/b/x"),
            ("gs://b/dir/", "dir", "dir/x", "gs://b/dir/x"),
            ("az://c/dir/", "dir", "dir/x", "az://c/dir/x"),
            ("az://c@acct/dir/", "dir", "dir/x", "az://c@acct/dir/x"),
            (
                "https://s3.us-east-1.amazonaws.com/bkt/dir/",
                "dir",
                "dir/x",
                "https://s3.us-east-1.amazonaws.com/bkt/dir/x",
            ),
            (
                "https://example.com/dir/",
                "dir",
                "dir/x",
                "https://example.com/dir/x",
            ),
            ("memory:///dir/", "dir", "dir/x", "memory:///dir/x"),
        ];
        for (url, base, location, expected) in cases {
            let url = Url::parse(url).expect("url");
            assert_eq!(
                ArrowObjectStore::rebuild_full_uri(&url, &key(base), &key(location)),
                expected,
                "listing {url}"
            );
        }
    }

    // #232: `p/b/c` and `p/b/d/e` are not immediate children of `p` and were left behind
    // while the call returned success. Siblings that merely share the name's characters
    // (`pfoo`, `p2`) are not under `p/` and must survive.
    #[tokio::test]
    async fn delete_prefix_removes_every_depth_and_only_the_prefix() {
        for prefix in ["s3://b/p", "s3://b/p/"] {
            let store = seeded(&["p/a", "p/b/c", "p/b/d/e", "pfoo/x", "p2/y"]).await;
            let (base, _url) = s3_prefix(prefix);
            delete_under(&store, &base).await.expect("delete_prefix");
            assert_eq!(
                remaining(&store).await,
                vec!["p2/y".to_string(), "pfoo/x".to_string()],
                "prefix {prefix}"
            );
        }
    }

    // A prefix that holds nothing is not an error and deletes nothing.
    #[tokio::test]
    async fn delete_prefix_on_an_empty_prefix_is_a_noop() {
        let store = seeded(&["other/x"]).await;
        let (base, _url) = s3_prefix("s3://b/none/");
        delete_under(&store, &base).await.expect("delete_prefix");
        assert_eq!(remaining(&store).await, vec!["other/x".to_string()]);
    }

    /// An in-memory store that behaves like S3 in the ways the adapter must survive:
    /// - its delete refuses every key containing `bad`;
    /// - when `marker` is set, a listing of the prefix `marker` also returns the object at
    ///   `marker` itself. That is how a folder marker object `p/` comes back from S3: its
    ///   key is parsed with `Path::parse`, which strips the trailing '/', so the location
    ///   is `p`, the prefix.
    #[derive(Debug)]
    struct Probe {
        inner: aos::memory::InMemory,
        marker: Option<Path>,
        /// Like the GCS S3-interop endpoint: ANY `DeleteObjects` request (`POST ?delete`)
        /// fails. On the S3 client every delete is one, even for a single key
        /// (`ObjectStoreExt::delete` is `delete_stream` of one location), and a request
        /// that fails as a whole yields ONE error, not one per key.
        rejects_delete_objects: bool,
        /// A delete request naming this key fails as a whole (one error for the request).
        reject_requests_containing: Option<String>,
        /// The number of keys in each delete request this store received.
        requests: Arc<std::sync::Mutex<Vec<usize>>>,
        /// A listing that ends with an error item, like a failed continuation page.
        list_error: bool,
        /// A listing that returns its entries and then never ends (a huge listing).
        list_never_ends: bool,
        in_flight: Arc<std::sync::atomic::AtomicUsize>,
        max_in_flight: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Probe {
        fn new(inner: aos::memory::InMemory) -> Self {
            Self {
                inner,
                marker: None,
                rejects_delete_objects: false,
                reject_requests_containing: None,
                requests: Arc::default(),
                list_error: false,
                list_never_ends: false,
                in_flight: Arc::default(),
                max_in_flight: Arc::default(),
            }
        }

        fn with_marker(inner: aos::memory::InMemory, marker: &str) -> Self {
            Self {
                marker: Some(Path::from(marker)),
                ..Self::new(inner)
            }
        }

        /// The sizes of the delete requests received, sorted.
        fn request_sizes(&self) -> Vec<usize> {
            let mut sizes = self.requests.lock().expect("requests").clone();
            sizes.sort_unstable();
            sizes
        }

        /// The most delete requests that were in flight at the same time.
        fn max_in_flight(&self) -> usize {
            self.max_in_flight.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// The marker's own entry, when `prefix` is the marker.
        fn marker_for(&self, prefix: Option<&Path>) -> Option<Path> {
            match (&self.marker, prefix) {
                (Some(m), Some(p)) if m == p => Some(m.clone()),
                _ => None,
            }
        }
    }

    impl std::fmt::Display for Probe {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Probe")
        }
    }

    #[async_trait]
    impl AosObjectStore for Probe {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: aos::PutOptions,
        ) -> aos::Result<aos::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: aos::PutMultipartOptions,
        ) -> aos::Result<Box<dyn aos::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> aos::Result<aos::GetResult> {
            self.inner.get_opts(location, options).await
        }
        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<'static, aos::Result<Path>>,
        ) -> futures::stream::BoxStream<'static, aos::Result<Path>> {
            use std::sync::atomic::Ordering::SeqCst;
            let inner = self.inner.clone();
            let rejects = self.rejects_delete_objects;
            let reject_containing = self.reject_requests_containing.clone();
            let requests = self.requests.clone();
            let (current, max) = (self.in_flight.clone(), self.max_in_flight.clone());
            let refused = |why: String| aos::Error::Generic {
                store: "Probe",
                source: why.into(),
            };
            futures::stream::once(async move {
                let items: Vec<aos::Result<Path>> = locations.collect().await;
                requests.lock().expect("requests").push(items.len());
                if rejects && !items.is_empty() {
                    return vec![Err(refused("DeleteObjects is not supported".to_string()))];
                }
                if let Some(key) = &reject_containing {
                    if items
                        .iter()
                        .any(|i| matches!(i, Ok(p) if p.as_ref() == key))
                    {
                        return vec![Err(refused("the request failed".to_string()))];
                    }
                }
                let now = current.fetch_add(1, SeqCst) + 1;
                max.fetch_max(now, SeqCst);
                // Let the other requests of a concurrent delete start.
                tokio::task::yield_now().await;
                let checked: Vec<aos::Result<Path>> = items
                    .into_iter()
                    .map(|l| match l {
                        Ok(p) if p.as_ref().contains("bad") => Err(refused(format!("refused {p}"))),
                        other => other,
                    })
                    .collect();
                let out = inner
                    .delete_stream(futures::stream::iter(checked).boxed())
                    .collect::<Vec<_>>()
                    .await;
                current.fetch_sub(1, SeqCst);
                out
            })
            .flat_map(futures::stream::iter)
            .boxed()
        }
        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, aos::Result<aos::ObjectMeta>> {
            let mut listed = self.inner.list(prefix);
            if self.list_error {
                listed = listed
                    .chain(futures::stream::once(std::future::ready(Err(
                        aos::Error::Generic {
                            store: "Probe",
                            source: "listing page failed".into(),
                        },
                    ))))
                    .boxed();
            }
            if self.list_never_ends {
                listed = listed.chain(futures::stream::pending()).boxed();
            }
            match self.marker_for(prefix) {
                Some(marker) => {
                    let inner = self.inner.clone();
                    futures::stream::once(async move { inner.head(&marker).await })
                        .chain(listed)
                        .boxed()
                }
                None => listed,
            }
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> aos::Result<aos::ListResult> {
            let mut result = self.inner.list_with_delimiter(prefix).await?;
            if let Some(marker) = self.marker_for(prefix) {
                result.objects.push(self.inner.head(&marker).await?);
            }
            Ok(result)
        }
        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: aos::CopyOptions,
        ) -> aos::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    // A delete that fails for some objects must not be reported as success (the #232 shape:
    // objects left behind while the call returns Ok). The good objects are still deleted.
    #[tokio::test]
    async fn delete_prefix_reports_objects_it_could_not_delete() {
        let store = Probe::new(seeded(&["p/a", "p/bad1", "p/x/bad2", "p/x/c", "q/z"]).await);
        let (base, _url) = s3_prefix("s3://b/p/");
        let err = delete_under(&store, &base).await.expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("2 failed"), "{msg}");
        assert!(
            msg.contains("refused p/bad1") || msg.contains("refused p/x/bad2"),
            "{msg}"
        );
        assert_eq!(
            remaining(&store.inner).await,
            vec![
                "p/bad1".to_string(),
                "p/x/bad2".to_string(),
                "q/z".to_string()
            ]
        );
    }

    // #245: `recursive=true` lists every object under the prefix at any depth, with no
    // directory entries; `recursive=false` keeps the direct children (objects and
    // directory entries). Segment matching holds: `p` never selects `pfoo` or `p2`.
    #[tokio::test]
    async fn recursive_listing_returns_every_depth_and_only_the_prefix() {
        let store = seeded(&["p/a", "p/b/c", "p/b/d/e", "pfoo/x", "p2/y"]).await;
        let deep = vec![
            "s3://b/p/a".to_string(),
            "s3://b/p/b/c".to_string(),
            "s3://b/p/b/d/e".to_string(),
        ];
        for uri in ["s3://b/p", "s3://b/p/"] {
            assert_eq!(s3_list(&store, uri, true).await, deep, "recursive {uri}");
            assert_eq!(
                s3_list(&store, uri, false).await,
                vec!["s3://b/p/a".to_string(), "s3://b/p/b".to_string()],
                "non-recursive {uri}"
            );
        }
        // The bucket root, recursively: every object.
        assert_eq!(s3_list(&store, "s3://b/", true).await.len(), 5);
    }

    // Every URI a recursive listing returns addresses a real object, deep keys included.
    #[tokio::test]
    async fn recursive_listed_uris_round_trip_for_deep_keys() {
        let store = seeded(&["p/a", "p/b/c", "p/b/d/e", "q/z"]).await;
        let uris = s3_list(&store, "s3://b/p/", true).await;
        assert_eq!(uris.len(), 3);
        for uri in uris {
            let url = Url::parse(&uri).expect("url");
            let (_bucket, key) =
                s3_target(&url, &uri, false).expect("a listed URI parses as an object");
            store
                .get(&key)
                .await
                .unwrap_or_else(|e| panic!("{uri} does not address a listed object: {e}"));
        }
    }

    // On S3 every delete is a `DeleteObjects` request that takes up to 1000 keys. The
    // adapter forms its own chunks of that size: 2500 objects are 3 requests, not 2500.
    #[tokio::test]
    async fn delete_prefix_batches_keys_into_delete_objects_requests() {
        let keys: Vec<String> = (0..2500).map(|i| format!("p/d{}/k{i}", i % 7)).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let store = Probe::new(seeded(&refs).await);
        let (base, _url) = s3_prefix("s3://b/p/");
        delete_under(&store, &base).await.expect("delete_prefix");
        let sizes = store.request_sizes();
        assert_eq!(
            (
                sizes.len(),
                sizes.iter().sum::<usize>(),
                sizes.last().copied()
            ),
            (3, 2500, Some(1000)),
            "request sizes (count, total, largest)"
        );
        assert_eq!(sizes[0], 500);
        assert_eq!(remaining(&store.inner).await, Vec::<String>::new());
    }

    // The GCS S3-interop endpoint rejects every `DeleteObjects`. The Arrow backend cannot
    // delete there: the call must fail, name the keys and count each as failed, never
    // report success.
    #[tokio::test]
    async fn delete_prefix_on_an_endpoint_without_delete_objects_fails_naming_each_key() {
        let mut store = Probe::new(seeded(&["p/a", "p/b/c", "p/d"]).await);
        store.rejects_delete_objects = true;
        let (base, _url) = s3_prefix("s3://b/p/");
        let err = delete_under(&store, &base).await.expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("3 failed"), "{msg}");
        for key in ["p/a", "p/b/c", "p/d"] {
            assert!(msg.contains(key), "{key} must be named: {msg}");
        }
        assert!(msg.contains("DeleteObjects is not supported"), "{msg}");
        assert_eq!(remaining(&store.inner).await.len(), 3);
    }

    // At most DELETE_REQUESTS_IN_FLIGHT delete REQUESTS at once (the repo-wide bound), and
    // they do overlap. Chunks of 5 keys make 12 requests from 60 objects.
    #[tokio::test]
    async fn delete_prefix_bounds_and_uses_request_concurrency() {
        let keys: Vec<String> = (0..60).map(|i| format!("p/k{i}")).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let store = Probe::new(seeded(&refs).await);
        let (base, _url) = s3_prefix("s3://b/p/");
        delete_under_in_chunks(&store, &base, 5)
            .await
            .expect("delete_prefix");
        assert_eq!(store.request_sizes(), vec![5; 12]);
        let max = store.max_in_flight();
        assert!(
            (2..=crate::delete_contract::DELETE_REQUESTS_IN_FLIGHT).contains(&max),
            "max in flight {max}"
        );
        assert_eq!(remaining(&store.inner).await, Vec::<String>::new());
    }

    // A request that fails as a whole yields ONE error; every key of its chunk is still
    // counted as failed, and the other chunks' keys are deleted.
    #[tokio::test]
    async fn a_failed_request_counts_every_key_of_its_chunk() {
        let keys: Vec<String> = (0..12).map(|i| format!("p/k{i:02}")).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let mut store = Probe::new(seeded(&refs).await);
        store.reject_requests_containing = Some("p/k07".to_string());
        let (base, _url) = s3_prefix("s3://b/p/");
        let err = delete_under_in_chunks(&store, &base, 4)
            .await
            .expect_err("must fail");
        let msg = format!("{err:#}");
        // The chunk holding k07 is one of three of 4 keys: all 4 failed, 8 deleted.
        assert!(msg.contains("4 failed"), "{msg}");
        assert!(msg.contains("(8 deleted)"), "{msg}");
        assert_eq!(remaining(&store.inner).await.len(), 4);
    }

    // Every failed object counts on its own, however many requests there are, and the
    // error quotes only the first few.
    #[tokio::test]
    async fn delete_prefix_counts_every_failed_object() {
        let mut keys: Vec<String> = (0..7).map(|i| format!("p/bad{i}")).collect();
        keys.extend((0..20).map(|i| format!("p/ok{i}")));
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let store = Probe::new(seeded(&refs).await);
        let (base, _url) = s3_prefix("s3://b/p/");
        let err = delete_under(&store, &base).await.expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("7 failed"), "{msg}");
        assert!(msg.contains("20 deleted"), "{msg}");
        assert!(!msg.contains("listing incomplete"), "{msg}");
        assert_eq!(
            msg.matches("refused").count(),
            MAX_NAMED_DELETE_FAILURES,
            "{msg}"
        );
        assert_eq!(remaining(&store.inner).await.len(), 7);
    }

    // A failed listing is an error of the call, not a short success; the objects listed
    // before the failure are still deleted.
    #[tokio::test]
    async fn delete_prefix_fails_when_the_listing_fails() {
        let mut store = Probe::new(seeded(&["p/a", "p/b"]).await);
        store.list_error = true;
        let (base, _url) = s3_prefix("s3://b/p/");
        let err = delete_under(&store, &base).await.expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("1 failed"), "{msg}");
        assert!(msg.contains("listing incomplete"), "{msg}");
        assert!(msg.contains("listing page failed"), "{msg}");
        assert_eq!(remaining(&store.inner).await, Vec::<String>::new());
    }

    // F3: a recursive listing is a stream, not a buffer: the first entries arrive while the
    // backend is still listing. The store here returns two entries and then never ends; a
    // buffering implementation never yields.
    #[tokio::test]
    async fn recursive_list_stream_yields_entries_before_the_listing_ends() {
        let mut probe = Probe::new(seeded(&["p/a", "p/b"]).await);
        probe.list_never_ends = true;
        let (base, url) = s3_prefix("s3://b/p/");
        let mut stream = list_stream_under(Arc::new(probe), base, url, true);
        let mut got = Vec::new();
        for _ in 0..2 {
            let next = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
                .await
                .expect("entries must arrive before the listing ends")
                .expect("item")
                .expect("ok");
            got.push(next);
        }
        got.sort();
        assert_eq!(
            got,
            vec!["s3://b/p/a".to_string(), "s3://b/p/b".to_string()]
        );
    }

    // The stream and `list` return the same entries; a listing error comes through as an Err
    // item after the entries that preceded it.
    #[tokio::test]
    async fn list_stream_under_matches_list_under_and_reports_errors() {
        let mut probe = Probe::new(seeded(&["p/a", "p/b/c"]).await);
        probe.list_error = true;
        let probe = Arc::new(probe);
        let (base, url) = s3_prefix("s3://b/p/");
        let items: Vec<Result<String>> =
            list_stream_under(probe.clone(), base.clone(), url.clone(), true)
                .collect()
                .await;
        assert_eq!(items.iter().filter(|i| i.is_ok()).count(), 2);
        assert_eq!(items.iter().filter(|i| i.is_err()).count(), 1);
        assert!(
            items.last().expect("items").is_err(),
            "the error comes last"
        );
        assert!(list_under(probe, base, url, true).await.is_err());
    }

    // `rebuild_full_uri` with the `base` the adapter really derives: `s3://` goes through
    // `s3_target` (as `parse_as` does) and every other scheme through
    // `ObjectStoreScheme::parse`, the function `object_store::parse_url` takes its path
    // from. No store is built, so the test depends on no environment variable. A
    // hand-written `base` could agree with the code under test by construction; these
    // cannot.
    #[test]
    fn rebuild_full_uri_matches_the_base_the_parser_derives() {
        let cases = [
            ("s3://b/dir/", "s3://b/dir/x"),
            ("s3://b/dir", "s3://b/dir/x"),
            ("s3://b/a/b/c/", "s3://b/a/b/c/x"),
            ("s3://b/", "s3://b/x"),
            ("s3://b", "s3://b/x"),
            ("s3a://b/dir/", "s3a://b/dir/x"),
            ("gs://b/dir/", "gs://b/dir/x"),
            ("gs://b/", "gs://b/x"),
            (
                "abfss://c@acct.dfs.core.windows.net/a/b/",
                "abfss://c@acct.dfs.core.windows.net/a/b/x",
            ),
            (
                "https://s3.us-east-1.amazonaws.com/bkt/dir/",
                "https://s3.us-east-1.amazonaws.com/bkt/dir/x",
            ),
            (
                "https://s3.us-east-1.amazonaws.com/bkt/",
                "https://s3.us-east-1.amazonaws.com/bkt/x",
            ),
            (
                "https://acct.blob.core.windows.net/cont/dir/",
                "https://acct.blob.core.windows.net/cont/dir/x",
            ),
            ("https://example.com/dir/", "https://example.com/dir/x"),
            ("memory:///dir/", "memory:///dir/x"),
        ];
        for (listing, expected) in cases {
            let url = Url::parse(listing).expect("url");
            let base = if url.scheme() == "s3" {
                s3_target(&url, listing, true).expect("s3_target").1
            } else {
                // `parse_url` takes its prefix path from `ObjectStoreScheme::parse` and only
                // then builds the store (from environment variables such as
                // GOOGLE_APPLICATION_CREDENTIALS, which a test must not depend on). The
                // path is the part `parse_as` uses, so it is derived the same way, without
                // building a store.
                aos::ObjectStoreScheme::parse(&url)
                    .unwrap_or_else(|e| panic!("ObjectStoreScheme::parse({listing}): {e}"))
                    .1
            };
            let location = base.clone().join("x");
            assert_eq!(
                ArrowObjectStore::rebuild_full_uri(&url, &base, &location),
                expected,
                "listing {listing} (base {base:?})"
            );
        }
    }

    // F1: S3 stores the folder marker `p/` as an object. `object_store` parses listed keys
    // with `Path::parse`, which strips a trailing '/', so the marker comes back with the
    // location `p` -- the prefix itself (`Probe::with_marker`). It is not an entry UNDER the
    // prefix: it must not be listed as a child, and above all `delete_prefix` must not
    // delete the unrelated object `p` that shares its Path.
    #[tokio::test]
    async fn the_prefix_itself_is_not_an_entry_under_the_prefix() {
        for recursive in [false, true] {
            let store = Arc::new(Probe::with_marker(seeded(&["p", "p/a"]).await, "p"));
            let (base, url) = s3_prefix("s3://b/p/");
            let mut got = list_under(store, base, url, recursive).await.expect("list");
            got.sort();
            assert_eq!(got, vec!["s3://b/p/a".to_string()], "recursive={recursive}");
        }
        let store = Probe::with_marker(seeded(&["p", "p/a", "p/b/c", "other"]).await, "p");
        let (base, _url) = s3_prefix("s3://b/p/");
        delete_under(&store, &base).await.expect("delete_prefix");
        assert_eq!(
            remaining(&store.inner).await,
            vec!["other".to_string(), "p".to_string()]
        );
    }
}
