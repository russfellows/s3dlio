// src/azure_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use futures::{stream::FuturesUnordered, Stream, StreamExt};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::azure_credential::{self, CredentialMode};
use crate::data_loader::parallel_fetch::DropCancel;
use crate::read_buffer::ReadBuffer;

use azure_core::http::{NoFormat, RequestContent, Url, XmlFormat};

use azure_storage_blob::clients::{
    BlobClient, BlobContainerClient, BlobServiceClient, BlobServiceClientOptions, BlockBlobClient,
};
use azure_storage_blob::models::{
    BlobClientDownloadOptions, BlobClientGetPropertiesOptions,
    BlobClientGetPropertiesResultHeaders, BlobClientUploadOptions,
    BlobContainerClientListBlobsOptions, BlockBlobClientCommitBlockListOptions,
    BlockBlobClientStageBlockOptions, BlockList, BlockListType, BlockLookupList, HttpRange,
};
use tracing::{debug, warn};

/// The partition size for a sequential download of a blob believed to be `object_size` bytes.
fn sequential_partition_size(object_size: u64) -> usize {
    usize::try_from(object_size)
        .unwrap_or(usize::MAX)
        .max(crate::constants::AZURE_MIN_DOWNLOAD_PARTITION_SIZE)
}

/// How many bytes a range read reserves before the data arrives: the range length, never more
/// than [`crate::constants::AZURE_RANGE_PRESIZE_MAX`].
fn range_presize(range_len: Option<u64>) -> Option<u64> {
    range_len.map(|len| len.min(crate::constants::AZURE_RANGE_PRESIZE_MAX))
}

/// The blob's total length from a download response's headers: the total in `Content-Range`
/// (`bytes 0-1023/4096`) if the response was a partial one, else `Content-Length`.
fn total_len_from_headers(content_range: Option<&str>, content_length: Option<u64>) -> Option<u64> {
    match content_range {
        Some(range) => range.rsplit('/').next().and_then(|t| t.trim().parse().ok()),
        None => content_length,
    }
}

/// Receive a download body into one buffer sized from `expected_len` (see
/// [`AzureBlob::get_with_sdk_concurrency_sized`]); without a length it grows as data arrives.
///
/// The SDK's `ResponseBody::collect()` appends to an empty `BytesMut` with `reserve`, so a large
/// blob is copied again at every growth step; `collect_into` needs the exact length up front and
/// fails if the blob is larger. Reading the stream here keeps one copy per byte and still
/// tolerates a size that changed since the `stat`.
async fn collect_body<S>(body: S, expected_len: Option<u64>) -> Result<Bytes>
where
    S: Stream<Item = azure_core::Result<Bytes>> + Unpin,
{
    let buffer = fill_read_buffer(
        body,
        ReadBuffer::with_expected_len(expected_len.unwrap_or(0)),
    )
    .await?;
    Ok(buffer.into_bytes())
}

async fn fill_read_buffer<S>(mut body: S, mut buffer: ReadBuffer) -> Result<ReadBuffer>
where
    S: Stream<Item = azure_core::Result<Bytes>> + Unpin,
{
    while let Some(chunk) = body.next().await {
        buffer.push(&chunk?);
    }
    Ok(buffer)
}

/// Process-wide `BlobServiceClient`s, one per account URL and credential mode.
///
/// A client owns the HTTP pipeline: its connection pool, TLS state and the bearer-token
/// cache of its authorization policy. Building one per operation threw all of that away,
/// so every object paid a new TCP/TLS handshake and a new token request (for the CLI mode, a
/// new `az account get-access-token` process, twice: HEAD and GET). The map is bounded by
/// [`crate::constants::AZURE_SERVICE_CLIENT_CACHE_MAX`]; at the limit one entry is evicted
/// (a client still in use stays alive through its `Arc`). Building a client is synchronous
/// and does no I/O, so it is done under the lock and every caller gets the same client.
type ServiceClientCache =
    std::sync::Mutex<std::collections::HashMap<String, (Arc<BlobServiceClient>, CredentialMode)>>;
static SERVICE_CLIENTS: std::sync::LazyLock<ServiceClientCache> =
    std::sync::LazyLock::new(Default::default);

/// The (cached) service client for `account_url`, and the credential mode it uses.
///
/// The mode comes from `S3DLIO_AZURE_CREDENTIAL` (see [`crate::azure_credential`]):
/// * a bearer-token mode attaches the process-wide credential for that mode, and needs an
///   `https` endpoint (the SDK refuses a bearer token on http);
/// * `sas` attaches no credential and appends `S3DLIO_AZURE_SAS_TOKEN` to the endpoint URL,
///   from which the SDK derives every request URL; `none` attaches nothing;
/// * `auto` on an `http` endpoint (an emulator such as Azurite) is `none`, as before.
fn service_client_for_url(account_url: &str) -> Result<(Arc<BlobServiceClient>, CredentialMode)> {
    let mut service_url = Url::parse(account_url)
        .map_err(|e| anyhow!("invalid Azure account URL '{account_url}': {e}"))?;
    let https = service_url.scheme() == "https";
    let mode = azure_credential::mode_for_endpoint(https)?;
    azure_credential::warn_unused_sas(mode);

    // The cache key never contains the SAS token itself, only a hash of it.
    let plain_url = service_url.to_string();
    let mut key = format!("{mode}|{plain_url}");
    if mode == CredentialMode::Sas {
        use std::hash::{Hash, Hasher};
        let token = azure_credential::sas_token(&|n| std::env::var(n).ok())?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        key.push_str(&format!("|sas:{:016x}", hasher.finish()));
        service_url.set_query(Some(&token));
    }

    let mut clients = SERVICE_CLIENTS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((client, mode)) = clients.get(&key) {
        return Ok((Arc::clone(client), *mode));
    }
    let credential = if mode.uses_bearer_token() {
        if !https {
            bail!(
                "Azure credential mode {mode} sends a bearer token and needs an https endpoint, \
                 but the endpoint is {plain_url}; use S3DLIO_AZURE_CREDENTIAL=none or sas for \
                 an http endpoint such as Azurite"
            );
        }
        Some(azure_credential::credential_for(mode)?)
    } else {
        None
    };
    #[cfg(test)]
    SERVICE_CLIENT_BUILD_LOG
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(plain_url);
    let client = Arc::new(
        BlobServiceClient::new(
            service_url,
            credential,
            Some(BlobServiceClientOptions::default()),
        )
        .map_err(|e| anyhow!(azure_credential::redact_sas(&e.to_string())))?,
    );
    if clients.len() >= crate::constants::AZURE_SERVICE_CLIENT_CACHE_MAX {
        if let Some(evict) = clients.keys().next().cloned() {
            clients.remove(&evict);
        }
    }
    clients.insert(key, (Arc::clone(&client), mode));
    Ok((client, mode))
}

#[cfg(test)]
fn cached_service_client_count() -> usize {
    SERVICE_CLIENTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .len()
}

/// Test hook: the URL of every `BlobServiceClient` this module built (as opposed to served
/// from its cache), in order.
#[cfg(test)]
static SERVICE_CLIENT_BUILD_LOG: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Azure's documented maximum number of uncommitted blocks per block blob.
/// See <https://learn.microsoft.com/en-us/rest/api/storageservices/put-block>.
/// Exceeding this is a `commit_block_list` failure on Azure's side; audit
/// #151 bug 1.6 (E3) is that `upload_multipart_stream` never checked this
/// client-side, so a caller with a `part_size` too small for their object
/// would stage tens of thousands of blocks only to have the final commit
/// fail server-side, with all staged blocks wasted.
const AZURE_MAX_BLOCK_COUNT: usize = 50_000;

fn request_content_from_bytes(body: Bytes) -> RequestContent<Bytes, NoFormat> {
    body.into()
}

/// Minimal properties surfaced by `stat`.
#[derive(Debug, Clone)]
pub struct AzureBlobProperties {
    pub content_length: u64,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// High-level client bound to one container.
pub struct AzureBlob {
    service_client: Arc<BlobServiceClient>,
    /// The credential mode this client authenticates with (named in authorization errors).
    credential_mode: CredentialMode,
    pub container: String,
}

impl AzureBlob {
    /// Public Azure endpoint for an account name.
    fn account_url_from_account(account: &str) -> String {
        format!("https://{}.blob.core.windows.net", account)
    }

    /// Azurite helper, e.g. http://127.0.0.1:10000/{account}
    #[allow(dead_code)]
    pub fn azurite_url(host: &str, port: u16, account: &str) -> String {
        format!("http://{}:{}/{}", host, port, account)
    }

    /// Build with the credential mode selected by `S3DLIO_AZURE_CREDENTIAL` (default `auto`; see
    /// [`crate::azure_credential`]). Account-key (Shared Key) authentication is not supported by
    /// the SDK. `auto` on an `http` endpoint (emulators) sends no credential.
    ///
    /// Supports custom endpoints via environment variables for local emulators and proxies:
    /// - `AZURE_STORAGE_ENDPOINT`: Primary endpoint URL (e.g., http://localhost:10000)
    /// - `AZURE_BLOB_ENDPOINT_URL`: Alternative endpoint URL
    ///
    /// When a custom endpoint is set, the account name is appended to form the full URL.
    /// Example: AZURE_STORAGE_ENDPOINT=http://localhost:10000 + account="devstoreaccount1"
    ///          → http://localhost:10000/devstoreaccount1
    pub fn with_default_credential(account: &str, container: &str) -> Result<Self> {
        // Check for custom endpoint (for Azurite or other emulators/proxies)
        if let Ok(endpoint) = std::env::var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT)
            .or_else(|_| std::env::var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL))
        {
            // Use custom endpoint (e.g., http://localhost:10000/account)
            let account_url = if endpoint.ends_with('/') {
                format!("{}{}", endpoint, account)
            } else {
                format!("{}/{}", endpoint, account)
            };
            tracing::info!("Using custom Azure endpoint: {}", account_url);
            return Self::with_default_credential_from_url(&account_url, container);
        }

        Self::with_default_credential_from_url(&Self::account_url_from_account(account), container)
    }

    /// Same, when a full endpoint URL (possibly emulator) is provided.
    pub fn with_default_credential_from_url(account_url: &str, container: &str) -> Result<Self> {
        let (service_client, credential_mode) = service_client_for_url(account_url)?;
        Ok(Self {
            service_client,
            credential_mode,
            container: container.to_string(),
        })
    }

    /// Turn an SDK error into an `anyhow` error. A 401 or 403 names the credential mode that
    /// was used (a bare "403" does not say which identity was refused), and any configured
    /// SAS token is removed from the message.
    fn auth_err(&self, e: azure_core::Error) -> anyhow::Error {
        let rejected = matches!(
            e.http_status(),
            Some(
                azure_core::http::StatusCode::Unauthorized
                    | azure_core::http::StatusCode::Forbidden
            )
        );
        // The SDK's top-level message is often generic ("non-transport error occurred ..."); the
        // cause (a failed token request, the response body) is in the source chain.
        let mut text = e.to_string();
        let mut source = std::error::Error::source(&e);
        while let Some(cause) = source {
            let cause_text = cause.to_string();
            if !text.contains(&cause_text) {
                text.push_str(": ");
                text.push_str(&cause_text);
            }
            source = cause.source();
        }
        let text = crate::azure_credential::redact_sas(&text);
        if rejected {
            anyhow!(
                "Azure credential mode {} was used and the request was rejected: {text}",
                self.credential_mode
            )
        } else {
            anyhow!(text)
        }
    }

    /// Blob service (rarely needed directly).
    #[allow(dead_code)]
    fn service_client(&self) -> Result<Arc<BlobServiceClient>> {
        Ok(Arc::clone(&self.service_client))
    }

    fn container_client(&self) -> Result<BlobContainerClient> {
        Ok(self.service_client.blob_container_client(&self.container))
    }

    fn blob_client(&self, blob: &str) -> Result<BlobClient> {
        Ok(self.service_client.blob_client(&self.container, blob))
    }

    fn block_blob_client(&self, blob: &str) -> Result<BlockBlobClient> {
        Ok(self.blob_client(blob)?.block_blob_client())
    }

    // ----------------------------------------------------------------------
    // Basic ops (single-shot upload, full/range download, stat, list)
    // ----------------------------------------------------------------------

    /// Simple upload (single request). For large bodies prefer multipart helpers below.
    pub async fn put(&self, key: &str, body: Bytes, overwrite: bool) -> Result<()> {
        debug!(
            "AzureBlob::put container='{}', key='{}', size={}, overwrite={}",
            self.container,
            key,
            body.len(),
            overwrite
        );
        let blob = self.blob_client(key)?;
        let data = request_content_from_bytes(body);
        let options = (!overwrite).then(|| BlobClientUploadOptions::default().if_not_exists());
        let _resp = blob
            .upload(data, options)
            .await
            .map_err(|e| self.auth_err(e))?;
        Ok(())
    }

    /// Range GET. If `end` is None → open-ended range.
    pub async fn get_range(&self, key: &str, start: u64, end: Option<u64>) -> Result<Bytes> {
        debug!(
            "AzureBlob::get_range container='{}', key='{}', start={}, end={:?}",
            self.container, key, start, end
        );
        let blob = self.blob_client(key)?;
        let range = match end {
            Some(e) if e >= start => HttpRange::new(start, e - start + 1),
            Some(e) => bail!("invalid Azure byte range: end {e} precedes start {start}"),
            None => HttpRange::from_offset(start),
        };
        // One request for the whole range: with the SDK's default 4 MiB partition a 64 MiB
        // range was sixteen serial sub-requests, each through an SDK partition buffer.
        let opts = BlobClientDownloadOptions {
            range: Some(range),
            parallel: NonZeroUsize::new(1),
            partition_size: end
                .and_then(|e| usize::try_from(e - start + 1).ok())
                .and_then(NonZeroUsize::new),
            ..Default::default()
        };
        let resp = blob
            .download(Some(opts))
            .await
            .map_err(|e| self.auth_err(e))?;
        let expected = range_presize(end.map(|e| e - start + 1));
        let body = collect_body(resp.body, expected).await?;
        debug!(
            "AzureBlob::get_range success: key='{}', {} bytes",
            key,
            body.len()
        );
        Ok(body)
    }

    /// Full GET (single buffer).
    pub async fn get(&self, key: &str) -> Result<Bytes> {
        self.get_with_sdk_concurrency(key, None, None).await
    }

    /// One stream, one request for a blob up to the partition size. `object_size` is a hint (a
    /// cached size can be stale): it sizes the destination buffer and may enlarge the partition
    /// (fewer requests for a large blob), but the partition is never smaller than
    /// [`crate::constants::AZURE_MIN_DOWNLOAD_PARTITION_SIZE`], so a stale tiny hint cannot turn a
    /// bigger blob into one request per byte.
    pub async fn get_sequential(&self, key: &str, object_size: u64) -> Result<Bytes> {
        let partition_size = sequential_partition_size(object_size);
        self.get_with_sdk_concurrency_sized(key, Some(1), Some(partition_size), Some(object_size))
            .await
    }

    pub async fn get_with_sdk_concurrency(
        &self,
        key: &str,
        concurrency: Option<usize>,
        partition_size: Option<usize>,
    ) -> Result<Bytes> {
        self.get_with_sdk_concurrency_sized(key, concurrency, partition_size, None)
            .await
    }

    /// [`Self::get_with_sdk_concurrency`] when the blob's size is known (from a `stat`):
    /// the destination buffer is allocated once at that size, so each received byte is copied
    /// once (issue #252; the SDK's own `collect()` grows its buffer as chunks arrive). If the
    /// blob turns out larger the buffer grows, so a stale size never loses data.
    pub async fn get_with_sdk_concurrency_sized(
        &self,
        key: &str,
        concurrency: Option<usize>,
        partition_size: Option<usize>,
        expected_len: Option<u64>,
    ) -> Result<Bytes> {
        debug!(
            "AzureBlob::get container='{}', key='{}'",
            self.container, key
        );
        let blob = self.blob_client(key)?;
        let options = BlobClientDownloadOptions {
            parallel: concurrency.and_then(NonZeroUsize::new),
            partition_size: partition_size.and_then(NonZeroUsize::new),
            ..Default::default()
        };
        let resp = blob
            .download(Some(options))
            .await
            .map_err(|e| self.auth_err(e))?;
        // Without a size hint, take the blob's length from the response (the SDK's first
        // request is a ranged GET, so `Content-Range` carries the total).
        let expected_len = expected_len.or_else(|| {
            let headers = &resp.headers;
            total_len_from_headers(
                headers.get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                    "content-range",
                )),
                headers
                    .get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                        "content-length",
                    ))
                    .and_then(|v| v.trim().parse().ok()),
            )
        });
        let body = collect_body(resp.body, expected_len).await?;
        debug!(
            "AzureBlob::get success: key='{}', {} bytes",
            key,
            body.len()
        );
        Ok(body)
    }

    /// Stat: read size, etag, last-modified from typed response headers.
    pub async fn stat(&self, key: &str) -> Result<AzureBlobProperties> {
        debug!(
            "AzureBlob::stat container='{}', key='{}'",
            self.container, key
        );
        let blob = self.blob_client(key)?;
        let resp = blob
            .get_properties(Some(BlobClientGetPropertiesOptions::default()))
            .await
            .map_err(|e| self.auth_err(e))?;
        let content_length = resp.content_length()?.unwrap_or(0);
        let etag = resp.etag()?.map(|e| e.to_string());
        let last_modified = resp.last_modified()?.map(|dt| dt.to_string());
        debug!(
            "AzureBlob::stat success: key='{}', content_length={}",
            key, content_length
        );
        Ok(AzureBlobProperties {
            content_length,
            etag,
            last_modified,
        })
    }

    /// Flat list with optional prefix.
    /// In SDK 0.7+, the Pager yields BlobItemInternal directly (not Response pages).
    pub async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        debug!(
            "AzureBlob::list container='{}', prefix={:?}",
            self.container, prefix
        );
        let container = self.container_client()?;
        let mut opts = BlobContainerClientListBlobsOptions::default();
        if let Some(p) = prefix {
            if !p.is_empty() {
                opts.prefix = Some(p.to_string());
            }
        }
        let mut pager = container.list_blobs(Some(opts))?;
        let mut out = Vec::new();

        // In 0.7.0, pager yields Result<BlobItemInternal> directly
        while let Some(item_result) = pager.next().await {
            let item = item_result.map_err(|e| self.auth_err(e))?;
            if let Some(name) = item.name {
                out.push(name);
            }
        }
        debug!("AzureBlob::list success: {} objects", out.len());
        Ok(out)
    }

    /// List blobs as a stream, yielding results one by one.
    /// This is more memory-efficient for large listings and enables progress updates.
    pub fn list_stream<'a>(
        &'a self,
        prefix: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let container = match self.container_client() {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mut opts = BlobContainerClientListBlobsOptions::default();
            if let Some(p) = prefix {
                if !p.is_empty() { opts.prefix = Some(p.to_string()); }
            }

            let mut pager = match container.list_blobs(Some(opts)) {
                Ok(p) => p,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            // In 0.7.0, pager yields Result<BlobItemInternal> directly
            while let Some(item_result) = pager.next().await {
                let item = match item_result {
                    Ok(i) => i,
                    Err(e) => {
                        yield Err(self.auth_err(e));
                        return;
                    }
                };

                if let Some(name) = item.name {
                    yield Ok(name);
                }
            }
        })
    }

    // src/azure_client.rs  (inside impl AzureBlob)
    /// Delete multiple blobs (simple loop; batch is possible later).
    pub async fn delete_objects(&self, blobs: &[String]) -> anyhow::Result<()> {
        debug!(
            "AzureBlob::delete_objects container='{}', count={}",
            self.container,
            blobs.len()
        );
        let container = self.container_client()?;
        for name in blobs {
            let b = container.blob_client(name);
            b.delete(None).await.map_err(|e| self.auth_err(e))?;
        }
        Ok(())
    }

    // ----------------------------------------------------------------------
    // Multipart (block blob) helpers
    // ----------------------------------------------------------------------

    /// Stage a block (non-committal). `block_id` is raw bytes; SDK base64-encodes on wire.
    pub async fn stage_block(&self, key: &str, block_id: &[u8], chunk: Bytes) -> Result<()> {
        debug!(
            "AzureBlob::stage_block container='{}', key='{}', chunk_size={}",
            self.container,
            key,
            chunk.len()
        );
        let bb = self.block_blob_client(key)?;
        let content_len = chunk.len() as u64;
        let body = request_content_from_bytes(chunk);
        let _resp = bb
            .stage_block(
                block_id,
                content_len,
                body,
                Some(BlockBlobClientStageBlockOptions::default()),
            )
            .await
            .map_err(|e| self.auth_err(e))?;
        Ok(())
    }

    /// Commit previously staged block IDs (order matters).
    pub async fn commit_block_list(
        &self,
        key: &str,
        committed_block_ids: Vec<Vec<u8>>,
    ) -> Result<()> {
        debug!(
            "AzureBlob::commit_block_list container='{}', key='{}', blocks={}",
            self.container,
            key,
            committed_block_ids.len()
        );
        let bb = self.block_blob_client(key)?;
        let lookup = BlockLookupList {
            committed: None,
            latest: Some(committed_block_ids),
            uncommitted: None,
        };
        let body: RequestContent<BlockLookupList, XmlFormat> = lookup.try_into()?;
        let _resp = bb
            .commit_block_list(body, Some(BlockBlobClientCommitBlockListOptions::default()))
            .await
            .map_err(|e| self.auth_err(e))?;
        Ok(())
    }

    /// Return committed block IDs (raw bytes that correspond to your passed IDs).
    pub async fn get_block_list_committed(&self, key: &str) -> Result<Vec<Vec<u8>>> {
        debug!(
            "AzureBlob::get_block_list_committed container='{}', key='{}'",
            self.container, key
        );
        let bb = self.block_blob_client(key)?;
        let resp = bb
            .get_block_list(BlockListType::Committed, None)
            .await
            .map_err(|e| self.auth_err(e))?;
        // In 0.7.0, Response::into_model() deserializes directly (not async)
        let list: BlockList = resp.into_model()?;
        let mut out = Vec::new();
        if let Some(blocks) = list.committed_blocks {
            for b in blocks {
                if let Some(id) = b.name {
                    out.push(id);
                }
            }
        }
        Ok(out)
    }

    // ----------------------------------------------------------------------
    // High-throughput multipart upload (bounded concurrency), additive API
    // ----------------------------------------------------------------------

    /// High-throughput uploader: feeds chunks to `stage_block` with bounded concurrency.
    /// - `stream`: yields Bytes chunks (already sized to `part_size`, except final)
    /// - `part_size`: hint for generating stable, fixed-width block IDs
    /// - `max_in_flight`: bounds the number of concurrent `stage_block` calls
    pub async fn upload_multipart_stream<S>(
        &self,
        key: &str,
        mut stream: S,
        part_size: usize,
        max_in_flight: usize,
    ) -> Result<()>
    where
        S: Stream<Item = Bytes> + Unpin + Send + 'static,
    {
        debug!("AzureBlob::upload_multipart_stream container='{}', key='{}', part_size={}, max_in_flight={}", self.container, key, part_size, max_in_flight);

        // Task-level parallelism (issue #148 site 3.1e): each stage_block
        // is `tokio::spawn`'d so tokio can distribute request signing +
        // upload work across worker threads instead of funneling every
        // block through this task's polling budget. DropCancel + select!
        // on each spawn honors both early-drop and mid-stream errors:
        // when we detect an error (either from a backpressure-drained
        // task or a final-drain task), we `cancel` explicitly to let
        // remaining in-flight blocks bail quickly, then drain to
        // completion so no JoinHandle is dropped mid-flight.
        let cancel = CancellationToken::new();
        let _drop_cancel = DropCancel(cancel.clone());
        let mut in_flight: FuturesUnordered<JoinHandle<Result<()>>> = FuturesUnordered::new();
        let mut next_idx: u64 = 0;
        let mut committed_ids: Vec<Vec<u8>> = Vec::new();
        let mut first_err: Option<anyhow::Error> = None;

        while let Some(chunk) = stream.next().await {
            // Stop enqueuing new work once we've seen an error — but keep
            // draining in_flight below so no spawned task is left detached.
            if first_err.is_some() {
                break;
            }

            // audit #151 bug 1.6 (E3): client-side safety cap so an
            // object whose part_size is too small for its size fails
            // fast with a clear, actionable error instead of staging
            // tens of thousands of blocks only to have commit_block_list
            // reject the blob server-side once the limit is exceeded.
            if exceeds_azure_block_count_limit(next_idx as usize) {
                first_err = Some(azure_block_count_limit_error(
                    &self.container,
                    key,
                    part_size,
                ));
                break;
            }

            // Fixed-width raw bytes (SDK will base64 on the wire)
            let id_str = format!("{:016x}-{:08x}", next_idx, part_size as u32);
            let id_bytes = id_str.as_bytes().to_vec();

            // Maintain order of IDs to match blob composition.
            committed_ids.push(id_bytes.clone());

            // Backpressure — await the oldest completion when the pool
            // is full. Record (but do not bail on) any error observed so
            // we continue draining rather than leaking the rest.
            if in_flight.len() >= max_in_flight {
                if let Some(join_res) = in_flight.next().await {
                    absorb_stage_block_result(join_res, &mut first_err, &cancel);
                }
            }

            let this = self.clone_for_upload();
            let key_owned = key.to_string();
            let token = cancel.clone();
            in_flight.push(tokio::spawn(async move {
                tokio::select! {
                    _ = token.cancelled() => Err(anyhow!("stage_block cancelled")),
                    r = this.stage_block(&key_owned, &id_bytes, chunk) => r,
                }
            }));

            next_idx += 1;
        }

        // Drain remaining tasks — cancellation may have fired above; any
        // still-running spawn will bail through its select! arm quickly.
        while let Some(join_res) = in_flight.next().await {
            absorb_stage_block_result(join_res, &mut first_err, &cancel);
        }

        if let Some(e) = first_err {
            // audit #151 bug 1.5 (D4): previously returned bare Err with
            // no log at all — an operator watching logs had no way to
            // tell a multipart Azure upload failed mid-stream, or how
            // much was staged before it did. commit_block_list is never
            // called in this branch, so the already-staged blocks stay
            // uncommitted; Azure garbage-collects uncommitted blocks
            // automatically (no explicit delete API exists for "just
            // the uncommitted blocks" short of committing an empty/
            // unrelated list, which risks clobbering any pre-existing
            // committed data at this key — not attempted here).
            warn!(
                "{}",
                staged_blocks_failure_warning(&self.container, key, committed_ids.len(), &e)
            );
            return Err(e);
        }

        // Commit in produced order.
        self.commit_block_list(key, committed_ids).await
    }

    fn clone_for_upload(&self) -> Self {
        Self {
            service_client: Arc::clone(&self.service_client),
            credential_mode: self.credential_mode,
            container: self.container.clone(),
        }
    }
}

/// Build the operator-visibility warning logged when
/// `upload_multipart_stream` fails mid-stream (audit #151 bug 1.5 / D4).
/// Extracted as a pure function — independent of any Azure client or
/// network call — so its content (naming the container, blob key, and
/// how many blocks were staged before the failure) is directly
/// unit-testable without needing an Azure mock server, which this repo
/// does not have (unlike the S3 mock harness in tests/common/).
fn staged_blocks_failure_warning(
    container: &str,
    key: &str,
    staged_count: usize,
    err: &anyhow::Error,
) -> String {
    format!(
        "s3dlio Azure MPU: upload_multipart_stream failed for container='{container}' key='{key}' \
         after staging {staged_count} block(s) — those staged blocks were never committed and will \
         be garbage-collected automatically by Azure: {err}"
    )
}

/// Would staging a block at `next_block_index` (0-based) exceed Azure's
/// `AZURE_MAX_BLOCK_COUNT`-block-per-blob limit?
///
/// audit #151 bug 1.6 (E3): pure decision function extracted so the
/// client-side safety cap is directly unit-testable without staging tens
/// of thousands of real blocks against a live Azure account (this repo has
/// no Azure mock server, unlike the S3 mock harness in tests/common/).
fn exceeds_azure_block_count_limit(next_block_index: usize) -> bool {
    next_block_index >= AZURE_MAX_BLOCK_COUNT
}

/// Build the error returned when `upload_multipart_stream` hits the
/// client-side block-count safety cap (audit #151 bug 1.6 / E3), before
/// ever attempting `commit_block_list` (which would otherwise fail
/// server-side, wasting every block staged up to that point).
fn azure_block_count_limit_error(container: &str, key: &str, part_size: usize) -> anyhow::Error {
    anyhow!(
        "s3dlio Azure MPU: upload_multipart_stream aborted for container='{container}' \
         key='{key}' — object requires more than Azure's {AZURE_MAX_BLOCK_COUNT}-block-per-blob \
         limit at part_size={part_size} bytes; increase part_size (or S3DLIO_MULTIPART_PART_SIZE_MB) \
         to reduce the block count"
    )
}

/// Fold one spawned `stage_block` outcome into the first-error slot,
/// firing the shared cancellation token on the first observed error so
/// still-running blocks can bail quickly through their `select!` arm.
///
/// Extracted from `upload_multipart_stream` (issue #148 site 3.1e) so
/// both the backpressure step and the final drain reuse the same
/// bookkeeping.
fn absorb_stage_block_result(
    join_res: std::result::Result<Result<()>, tokio::task::JoinError>,
    first_err: &mut Option<anyhow::Error>,
    cancel: &CancellationToken,
) {
    match join_res {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            if first_err.is_none() {
                *first_err = Some(e);
                cancel.cancel();
            }
        }
        Err(join_err) if join_err.is_panic() => {
            if first_err.is_none() {
                *first_err = Some(anyhow::anyhow!("stage_block task panicked: {}", join_err));
                cancel.cancel();
            }
        }
        Err(_) => {
            // Task was cancelled (via the select! arm) — expected during
            // shutdown after an earlier error already fired cancel.
        }
    }
}

impl AzureBlob {
    // ----------------------------------------------------------------------
    // Container helpers (optional)
    // ----------------------------------------------------------------------

    #[allow(dead_code)]
    pub async fn create_container_if_missing(&self) -> Result<()> {
        let container = self.container_client()?;
        if !container.exists().await.map_err(|e| self.auth_err(e))? {
            container.create(None).await.map_err(|e| self.auth_err(e))?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn delete_container(&self) -> Result<()> {
        self.container_client()?
            .delete(None)
            .await
            .map_err(|e| self.auth_err(e))?;
        Ok(())
    }
}

// ============================================================================
// Helper Functions for Custom Endpoint URL Construction
// ============================================================================

/// Constructs the Azure account URL based on environment variables.
///
/// Returns the custom endpoint URL if `AZURE_STORAGE_ENDPOINT` or `AZURE_BLOB_ENDPOINT_URL`
/// is set, otherwise returns the standard Azure Blob endpoint.
///
/// This is extracted as a pure function for testability.
pub fn resolve_azure_account_url(account: &str) -> String {
    if let Ok(endpoint) = std::env::var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT)
        .or_else(|_| std::env::var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL))
    {
        // Use custom endpoint (e.g., http://localhost:10000/account)
        if endpoint.ends_with('/') {
            format!("{}{}", endpoint, account)
        } else {
            format!("{}/{}", endpoint, account)
        }
    } else {
        // Default: public Azure endpoint
        format!("https://{}.blob.core.windows.net", account)
    }
}

// ---------------------------------------------------------------------------
// Service-level helpers
// ---------------------------------------------------------------------------

/// List all containers in an Azure Blob Storage account.
///
/// Returns `Vec<(container_name, Option<last_modified_string>)>`.
///
/// Credentials are chosen exactly as for [`AzureBlob::with_default_credential`]
/// (`S3DLIO_AZURE_CREDENTIAL`, see [`crate::azure_credential`]).
pub async fn list_account_containers(account: &str) -> Result<Vec<(String, Option<String>)>> {
    list_account_containers_at(&resolve_azure_account_url(account)).await
}

/// [`list_account_containers`] against an explicit account URL.
async fn list_account_containers_at(account_url: &str) -> Result<Vec<(String, Option<String>)>> {
    let (service_client, _mode) = service_client_for_url(account_url)?;

    let mut containers: Vec<(String, Option<String>)> = Vec::new();
    let mut pager = service_client
        .list_containers(None)
        .map_err(|e| anyhow!("Azure list_containers failed: {}", e))?;

    while let Some(item) = pager.next().await {
        let item = item.map_err(|e| anyhow!("Azure list_containers item error: {e}"))?;
        let name = item.name.unwrap_or_default();
        let date = item
            .properties
            .and_then(|p| p.last_modified)
            .map(|t| t.to_string());
        containers.push((name, date));
    }

    Ok(containers)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to serialize tests that modify environment variables
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn builds_for(url: &str) -> usize {
        SERVICE_CLIENT_BUILD_LOG
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|built| built.as_str() == url)
            .count()
    }

    // Issue #252: a known-size download is received into one allocation. The SDK's
    // ResponseBody::collect() grows a BytesMut chunk by chunk.
    #[tokio::test]
    async fn collect_body_receives_a_known_size_blob_without_growth() {
        let chunk = Bytes::from(vec![0x5Au8; 64 * 1024]);
        let chunks = 128; // 8 MiB
        let body =
            futures::stream::iter((0..chunks).map(|_| Ok::<_, azure_core::Error>(chunk.clone())));
        let expected = (chunk.len() * chunks) as u64;
        let buffer = fill_read_buffer(body, ReadBuffer::with_expected_len(expected))
            .await
            .unwrap();
        assert_eq!(
            buffer.reallocations(),
            0,
            "RED: an 8 MiB body grew its buffer {} time(s)",
            buffer.reallocations()
        );
        assert_eq!(buffer.len() as u64, expected);

        // A body larger than the stale size is kept whole.
        let body =
            futures::stream::iter((0..chunks).map(|_| Ok::<_, azure_core::Error>(chunk.clone())));
        let got = collect_body(body, Some(expected / 2)).await.unwrap();
        assert_eq!(got.len() as u64, expected);
        // No size known at all.
        let body = futures::stream::iter((0..4).map(|_| Ok::<_, azure_core::Error>(chunk.clone())));
        assert_eq!(
            collect_body(body, None).await.unwrap().len(),
            4 * chunk.len()
        );
    }

    // Issue #242 review: every get/stat/get_range used to build a new BlobServiceClient (a new
    // HTTP connection pool, TLS state and empty bearer-token cache). Calls for one account
    // URL now share one client.
    #[test]
    fn one_service_client_per_account_url() {
        let url = "http://127.0.0.1:59301/acct242";
        let clients: Vec<AzureBlob> = (0..20)
            .map(|i| AzureBlob::with_default_credential_from_url(url, &format!("c{i}")).unwrap())
            .collect();
        assert_eq!(
            builds_for("http://127.0.0.1:59301/acct242"),
            1,
            "RED: 20 clients for one account URL built {} service clients",
            builds_for("http://127.0.0.1:59301/acct242")
        );
        for c in &clients[1..] {
            assert!(Arc::ptr_eq(&clients[0].service_client, &c.service_client));
        }
        // The container stays per-client.
        assert_eq!(clients[3].container, "c3");

        let other =
            AzureBlob::with_default_credential_from_url("http://127.0.0.1:59302/other", "c")
                .unwrap();
        assert!(!Arc::ptr_eq(
            &clients[0].service_client,
            &other.service_client
        ));
    }

    // The cache is bounded: a process that touches many accounts does not grow it forever.
    #[test]
    fn the_service_client_cache_is_bounded() {
        for i in 0..(crate::constants::AZURE_SERVICE_CLIENT_CACHE_MAX + 10) {
            AzureBlob::with_default_credential_from_url(
                &format!("http://127.0.0.1:59400/bounded{i}"),
                "c",
            )
            .unwrap();
        }
        assert!(cached_service_client_count() <= crate::constants::AZURE_SERVICE_CLIENT_CACHE_MAX);
    }

    // Issue #242 review: list_account_containers handed a bearer-token credential to every
    // client, so an http:// endpoint (Azurite) was refused by the SDK with "must use https".
    // It now follows with_default_credential_from_url: no credential for http.
    #[tokio::test]
    async fn list_account_containers_on_an_http_endpoint_sends_no_credential() {
        // With a credential the SDK refuses the client at once ("must use https"). Without
        // one the call goes on to the network, where nothing listens and the SDK retries for
        // a long while, so a timeout means the client was built, which is what is tested.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            list_account_containers_at("http://127.0.0.1:9/acct"),
        )
        .await;
        if let Ok(Err(e)) = outcome {
            let message = format!("{e:#}");
            assert!(
                !message.contains("must use https"),
                "RED: an http endpoint was given a credential: {message}"
            );
        }
    }

    #[test]
    fn request_content_from_bytes_preserves_sliced_allocation() {
        let source = Bytes::from(vec![7_u8; 4096]);
        let slice = source.slice(128..3968);
        let expected_ptr = slice.as_ptr();
        let expected_len = slice.len();

        let content = request_content_from_bytes(slice);
        let azure_core::http::Body::Bytes(actual) = content.body() else {
            panic!("Bytes conversion must produce an in-memory Azure request body");
        };

        assert_eq!(
            actual.as_ptr(),
            expected_ptr,
            "upload conversion copied data"
        );
        assert_eq!(actual.len(), expected_len);
    }

    // RED-then-GREEN regression tests for s3dlio issue #151 bug 1.5 (D4).
    //
    // Bug: upload_multipart_stream's failure branch returned a bare
    // `Err(e)` with no log at all when a stage_block failed mid-stream
    // -- an operator watching logs had no way to tell an Azure multipart
    // upload failed, or how many blocks were staged (and left
    // uncommitted, pending Azure's automatic GC) before it did.
    //
    // staged_blocks_failure_warning() is a pure function extracted from
    // the failure branch specifically so its content is testable without
    // an Azure mock server (this repo has one for S3, not Azure).

    #[test]
    fn staged_blocks_failure_warning_names_container_key_and_count() {
        let err = anyhow::anyhow!("stage_block cancelled");
        let msg = staged_blocks_failure_warning("my-container", "my-blob.bin", 7, &err);

        assert!(
            msg.contains("my-container"),
            "warning must name the container: {msg}"
        );
        assert!(
            msg.contains("my-blob.bin"),
            "warning must name the blob key: {msg}"
        );
        assert!(
            msg.contains('7'),
            "warning must include the staged block count: {msg}"
        );
        assert!(
            msg.contains("stage_block cancelled"),
            "warning must include the underlying error: {msg}"
        );
    }

    #[test]
    fn staged_blocks_failure_warning_handles_zero_staged_blocks() {
        // The very first block failed -- nothing was staged yet.
        let err = anyhow::anyhow!("connection refused");
        let msg = staged_blocks_failure_warning("c", "k", 0, &err);
        assert!(
            msg.contains('0'),
            "warning must report 0 staged blocks: {msg}"
        );
    }

    // RED-then-GREEN regression tests for s3dlio issue #151 bug 1.6 (E3).
    //
    // Bug: upload_multipart_stream never checked Azure's documented
    // 50,000-block-per-blob limit, so a caller whose part_size was too
    // small for their object would stage tens of thousands of blocks only
    // to have commit_block_list fail server-side, wasting every staged
    // block. exceeds_azure_block_count_limit() is the pure decision
    // extracted from that loop so the cap is testable without staging
    // 50,000+ real blocks against a live Azure account.

    #[test]
    fn exceeds_azure_block_count_limit_false_below_cap() {
        assert!(!exceeds_azure_block_count_limit(0));
        assert!(!exceeds_azure_block_count_limit(AZURE_MAX_BLOCK_COUNT - 1));
    }

    #[test]
    fn exceeds_azure_block_count_limit_true_at_and_above_cap() {
        assert!(exceeds_azure_block_count_limit(AZURE_MAX_BLOCK_COUNT));
        assert!(exceeds_azure_block_count_limit(AZURE_MAX_BLOCK_COUNT + 1));
    }

    #[test]
    fn azure_block_count_limit_error_names_container_key_and_part_size() {
        let err = azure_block_count_limit_error("my-container", "my-blob.bin", 4 * 1024 * 1024);
        let msg = err.to_string();
        assert!(
            msg.contains("my-container"),
            "error must name the container: {msg}"
        );
        assert!(
            msg.contains("my-blob.bin"),
            "error must name the blob key: {msg}"
        );
        assert!(
            msg.contains("50000") || msg.contains("50,000"),
            "error must name the block-count limit: {msg}"
        );
        assert!(
            msg.contains(&(4 * 1024 * 1024).to_string()),
            "error must name the part_size that caused the overflow: {msg}"
        );
    }

    #[test]
    fn test_account_url_from_account() {
        let url = AzureBlob::account_url_from_account("mystorageaccount");
        assert_eq!(url, "https://mystorageaccount.blob.core.windows.net");
    }

    #[test]
    fn test_azurite_url() {
        let url = AzureBlob::azurite_url("127.0.0.1", 10000, "devstoreaccount1");
        assert_eq!(url, "http://127.0.0.1:10000/devstoreaccount1");
    }

    #[test]
    fn test_resolve_azure_account_url_default() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Clear any existing endpoint env vars
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);

        let url = resolve_azure_account_url("mystorageaccount");
        assert_eq!(url, "https://mystorageaccount.blob.core.windows.net");
    }

    #[test]
    fn test_resolve_azure_account_url_with_primary_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set primary env var
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://localhost:10000",
        );
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);

        let url = resolve_azure_account_url("devstoreaccount1");
        assert_eq!(url, "http://localhost:10000/devstoreaccount1");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
    }

    #[test]
    fn test_resolve_azure_account_url_with_alternative_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set alternative env var (primary not set)
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::set_var(
            crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
            "http://127.0.0.1:9001",
        );

        let url = resolve_azure_account_url("testaccount");
        assert_eq!(url, "http://127.0.0.1:9001/testaccount");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);
    }

    #[test]
    fn test_resolve_azure_account_url_with_trailing_slash() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set env var with trailing slash
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://localhost:10000/",
        );
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);

        let url = resolve_azure_account_url("devstoreaccount1");
        assert_eq!(url, "http://localhost:10000/devstoreaccount1");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
    }

    #[test]
    fn test_resolve_azure_account_url_primary_takes_precedence() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set both env vars - primary should take precedence
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://primary:10000",
        );
        std::env::set_var(
            crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
            "http://alternative:9001",
        );

        let url = resolve_azure_account_url("testaccount");
        assert_eq!(url, "http://primary:10000/testaccount");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);
    }
}
