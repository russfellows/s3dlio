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
use crate::object_error::ObjectErrorKind;
use crate::object_store::HeadMetadata;
use crate::range_version::{Leader, ObjectVersion, Role};
use crate::read_buffer::ReadBuffer;
use crate::receive_progress::{ReceiveProgress, SpanMark};

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

/// The Azure SDK's own default partition size for a managed download (4 MiB;
/// `azure_storage_blob` 1.1.0 `partitioned_transfer/defaults.rs`, crate-private there). A
/// download that names no partition size gets the same first request the SDK would have made.
const SDK_DEFAULT_DOWNLOAD_PARTITION: usize = 4 * 1024 * 1024;

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

/// A numeric response header.
fn header_u64(headers: &azure_core::http::headers::Headers, name: &'static str) -> Option<u64> {
    headers
        .get_optional_str(&azure_core::http::headers::HeaderName::from_static(name))
        .and_then(|v| v.trim().parse().ok())
}

/// How many bytes a range read reserves: what the response's `Content-Length` says it carries,
/// else the requested length, capped (a caller-chosen length is not a promise).
fn range_expected_len(content_length: Option<u64>, requested: Option<u64>) -> Option<u64> {
    content_length.or_else(|| range_presize(requested))
}

/// Receive a download body into one buffer sized from `expected_len` (see
/// [`AzureBlob::get_with_sdk_concurrency_sized`]); without a length it grows as data arrives.
///
/// The SDK's `ResponseBody::collect()` appends to an empty `BytesMut` with `reserve`, so a large
/// blob is copied again at every growth step; `collect_into` needs the exact length up front and
/// fails if the blob is larger. Reading the stream here copies each chunk once, into a buffer that
/// is not reallocated, and still tolerates a size that changed since the `stat`. That is one
/// copy inside s3dlio after the SDK delivers the chunks: the SDK's partitioned download (blobs
/// above the partition size) buffers each partition itself first.
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
struct ServiceClientCache {
    clients: std::sync::Mutex<
        std::collections::HashMap<String, (Arc<BlobServiceClient>, CredentialMode)>,
    >,
    max: usize,
}

impl ServiceClientCache {
    fn new(max: usize) -> Self {
        Self {
            clients: Default::default(),
            max,
        }
    }

    /// The client cached under `key`, or the one `build` makes (and caches, evicting one
    /// entry first if the cache is full). `build` runs under the lock.
    fn get_or_build(
        &self,
        key: String,
        build: impl FnOnce() -> Result<(Arc<BlobServiceClient>, CredentialMode)>,
    ) -> Result<(Arc<BlobServiceClient>, CredentialMode)> {
        let mut clients = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((client, mode)) = clients.get(&key) {
            return Ok((Arc::clone(client), *mode));
        }
        let (client, mode) = build()?;
        if clients.len() >= self.max {
            if let Some(evict) = clients.keys().next().cloned() {
                clients.remove(&evict);
            }
        }
        clients.insert(key, (Arc::clone(&client), mode));
        Ok((client, mode))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.clients.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// Fork-aware (#297): the cached clients own connection pools whose tasks ran on the parent's
/// runtime, and the map's lock may have been held by a parent thread at the fork. A fork child
/// builds its own cache.
static SERVICE_CLIENTS_CELL: crate::fork_guard::ForkCell<ServiceClientCache> =
    crate::fork_guard::ForkCell::new();

fn service_clients() -> &'static ServiceClientCache {
    SERVICE_CLIENTS_CELL
        .get_or_init(|| ServiceClientCache::new(crate::constants::AZURE_SERVICE_CLIENT_CACHE_MAX))
}

/// An Azure SDK error with a message that names the credential mode (for a 401/403) and has any
/// SAS token removed, while the typed SDK error stays available to callers that classify errors
/// (the idempotent-delete contract looks for `BlobNotFound`; see
/// [`crate::delete_contract::is_not_found`]). `Display` shows only the redacted message and
/// `source()` is `None`, so printing the chain never shows the unredacted SDK text.
pub struct AzureRequestError {
    message: String,
    inner: azure_core::Error,
}

// A derived Debug would print the SDK error, which can hold the request URL (and so a SAS).
impl std::fmt::Debug for AzureRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureRequestError")
            .field("message", &self.message)
            .field("http_status", &self.inner.http_status())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for AzureRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AzureRequestError {}

impl AzureRequestError {
    /// The request failed because the blob does not exist.
    pub fn is_blob_not_found(&self) -> bool {
        azure_error_is_blob_not_found(&self.inner)
    }

    /// The HTTP status of the failed request, if there was a response.
    pub fn http_status(&self) -> Option<azure_core::http::StatusCode> {
        self.inner.http_status()
    }

    /// The service error code of the failed request (`BlobNotFound`, ...), from the
    /// SDK error's `ErrorKind::HttpResponse`.  `None` when there was no HTTP response or
    /// the response carried no code.
    pub(crate) fn error_code(&self) -> Option<&str> {
        azure_error_code(&self.inner)
    }

    /// The typed classification of this failure (plan §4.5 A2), see
    /// [`classify_azure_failure`].
    pub(crate) fn object_error_kind(&self) -> ObjectErrorKind {
        classify_azure_failure(
            self.inner.http_status().map(u16::from),
            self.error_code(),
            azure_error_is_transport(&self.inner),
        )
    }
}

/// The service error code of an SDK error that is an HTTP response error.
fn azure_error_code(e: &azure_core::Error) -> Option<&str> {
    match e.kind() {
        azure_core::error::ErrorKind::HttpResponse { error_code, .. } => error_code.as_deref(),
        _ => None,
    }
}

/// Whether an SDK error is a failure to get any response: a connection that could not be
/// made, or an I/O failure while sending or reading.
fn azure_error_is_transport(e: &azure_core::Error) -> bool {
    matches!(
        e.kind(),
        azure_core::error::ErrorKind::Connection | azure_core::error::ErrorKind::Io
    )
}

/// Classify an Azure failure from its HTTP `status`, service error `code` and whether it was
/// a transport failure (plan §4.5 A2).  Pure.
///
/// * `BlobNotFound` is `NotFound` (404 with that code); a missing container
///   (`ContainerNotFound`) is `Other`, never object absence;
/// * 412 is `VersionChanged`; 403 is `Forbidden`;
/// * 429 and 503 are `Throttled`; any other 5xx is `Server`;
/// * a failure with no response (connection, I/O) is `Transport`;
/// * anything else, including every other 404, is `Other`.
pub(crate) fn classify_azure_failure(
    status: Option<u16>,
    code: Option<&str>,
    transport: bool,
) -> ObjectErrorKind {
    match (status, code) {
        (Some(404), Some("BlobNotFound")) => ObjectErrorKind::NotFound,
        (Some(404), Some("ContainerNotFound")) => ObjectErrorKind::Other,
        (Some(412), _) => ObjectErrorKind::VersionChanged,
        (Some(403), _) => ObjectErrorKind::Forbidden,
        (Some(429 | 503), _) => ObjectErrorKind::Throttled,
        (Some(500..=599), _) => ObjectErrorKind::Server,
        (None, _) if transport => ObjectErrorKind::Transport,
        _ => ObjectErrorKind::Other,
    }
}

/// Attach the Azure classification of `err` as an
/// [`ObjectError`](crate::object_error::ObjectError) context.  `err` must carry an
/// [`AzureRequestError`]; one that does not is returned unchanged.
///
/// An error that already carries an `ObjectError` is returned unchanged, with one exception:
/// on a `pinned` read whose outer kind is `NotFound`, a `VersionChanged` is attached over it
/// (the outer classification wins), because a pinned read never reports absence.
///
/// The attached `ObjectError`'s own message is built only from the HTTP status and a
/// sanitised service code (`[A-Za-z0-9._-]`, at most 64 characters).  The retained source
/// error, reachable through `{:#}`, `{:?}` or the chain, is not redacted by this module; for
/// Azure it is the [`AzureRequestError`], which `azure_error` already redacted (SAS tokens
/// removed).
pub(crate) fn attach_azure_classification(err: anyhow::Error, pinned: bool) -> anyhow::Error {
    if let Some(existing) = crate::object_error::object_error(&err) {
        if pinned && existing.kind == ObjectErrorKind::NotFound {
            return crate::object_error::attach_object_error(
                err,
                ObjectErrorKind::VersionChanged,
                "az",
                "pinned version no longer found",
            );
        }
        return err;
    }
    let Some(az) = err.downcast_ref::<AzureRequestError>() else {
        return err;
    };
    let kind = az.object_error_kind();
    let message =
        crate::object_error::status_code_message(az.http_status().map(u16::from), az.error_code());
    if pinned {
        crate::object_error::attach_pinned_read_error(err, kind, "az", message)
    } else {
        crate::object_error::attach_object_error(err, kind, "az", message)
    }
}

/// Whether an SDK error is the service answering 404 `BlobNotFound`.
pub fn azure_error_is_blob_not_found(e: &azure_core::Error) -> bool {
    matches!(
        e.kind(),
        azure_core::error::ErrorKind::HttpResponse { status, error_code, .. }
            if *status == azure_core::http::StatusCode::NotFound
                && error_code.as_deref() == Some("BlobNotFound")
    )
}

/// Wrap an SDK error as an [`AzureRequestError`]: the message includes the cause chain (the
/// SDK's top-level text is often generic, "non-transport error occurred ...") and, for a 401 or
/// 403, the credential mode that was used (a bare "403" does not say which identity was
/// refused); a configured SAS token is removed.
fn azure_error(mode: CredentialMode, e: azure_core::Error) -> anyhow::Error {
    let rejected = matches!(
        e.http_status(),
        Some(azure_core::http::StatusCode::Unauthorized | azure_core::http::StatusCode::Forbidden)
    );
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
    let text = azure_credential::redact_sas(&text);
    let message = if rejected {
        format!("Azure credential mode {mode} was used and the request was rejected: {text}")
    } else {
        text
    };
    anyhow::Error::new(AzureRequestError { message, inner: e })
}

/// Builds the error the client reports for an SDK error, for tests of its text.
#[doc(hidden)]
pub fn request_error_for_test(inner: azure_core::Error) -> anyhow::Error {
    azure_error(CredentialMode::None, inner)
}

/// `url` without its query string or fragment, for messages and logs (a query may hold a SAS
/// token or other secrets).
pub fn redact_url(url: &str) -> String {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let url = &url[..end];
    // `scheme://user:password@host/...` -> `scheme://host/...`
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_once('/').map_or((rest, ""), |(a, p)| (a, p));
            let authority = authority.rsplit('@').next().unwrap_or(authority);
            if path.is_empty() && !rest.contains('/') {
                format!("{scheme}://{authority}")
            } else {
                format!("{scheme}://{authority}/{path}")
            }
        }
        None => url.to_string(),
    }
}

/// Whether `account` can be used to build a request URL or path safely. Azure storage account
/// names are 3-24 lowercase letters and digits. With a custom endpoint (an emulator or proxy)
/// that rule is relaxed (emulators may use other names), but a name that could change which host
/// or path a request goes to (`/ ? # @ \`, whitespace, control characters, `.`-only names) is
/// never accepted. Checked before any request is made or credential used.
pub fn validate_account_name(account: &str) -> Result<()> {
    canonical_account_name(account).map(|_| ())
}

/// The account name to use in a URL: validated, and lowercased when it becomes a host name
/// (host names are case-insensitive and storage account names are lowercase, so `MyAccount`
/// means `myaccount`). With a custom endpoint the name is a path segment for an emulator and is
/// kept exactly as given. Every place that puts an account into a URL goes through this.
pub fn canonical_account_name(account: &str) -> Result<String> {
    // The custom-endpoint decision reads the environment: apply the `.env` policy first (#243).
    crate::dotenv::apply();
    let custom = custom_endpoint_set();
    let name = if custom {
        account.to_string()
    } else {
        account.to_ascii_lowercase()
    };
    account_name_problem(&name, custom).map_or(Ok(name), |why| {
        Err(anyhow!("invalid Azure account name {account:?}: {why}"))
    })
}

fn custom_endpoint_set() -> bool {
    [
        crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
        crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
    ]
    .iter()
    .any(|var| std::env::var(var).is_ok_and(|v| !v.trim().is_empty()))
}

/// The environment-dependent account rules. The structural part (empty, or a character that
/// could redirect the request) is shared with the environment-free URI parser, and so is the
/// path-segment rule used with a custom endpoint; only the real-account length and alphabet
/// rule of the default endpoint is decided here.
fn account_name_problem(account: &str, custom_endpoint: bool) -> Option<&'static str> {
    if let Some(why) = crate::capabilities::azure_account_redirect_problem(account) {
        return Some(why);
    }
    if custom_endpoint {
        return crate::capabilities::azure_account_problem_pure(account);
    }
    let well_formed = (3..=24).contains(&account.len())
        && account
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
    (!well_formed).then_some("storage account names are 3-24 lowercase letters and digits")
}

/// A host the machine itself serves: `localhost` or a loopback address.
fn is_loopback_host(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// The (cached) service client for `account_url`, and the credential mode it uses.
///
/// The mode comes from `S3DLIO_AZURE_CREDENTIAL` (see [`crate::azure_credential`]):
/// * a bearer-token mode attaches the process-wide credential for that mode, and needs an
///   `https` endpoint (the SDK refuses a bearer token on http);
/// * `sas` attaches no credential and appends `S3DLIO_AZURE_SAS_TOKEN` to the endpoint URL,
///   from which the SDK derives every request URL; `none` attaches nothing;
/// * `auto` on an `http` endpoint (an emulator such as Azurite) is `none`, as before.
fn service_client_for_url(account_url: &str) -> Result<(Arc<BlobServiceClient>, CredentialMode)> {
    let mut service_url = Url::parse(account_url).map_err(|e| {
        anyhow!(
            "invalid Azure account URL '{}': {e}",
            redact_url(account_url)
        )
    })?;
    let https = service_url.scheme() == "https";
    let mode = azure_credential::mode_for_endpoint(https)?;
    azure_credential::warn_unused_sas(mode);

    // The cache key never contains the SAS token itself, only a hash of it. Messages and logs
    // use the URL without any query string.
    let plain_url = redact_url(service_url.as_str());
    if mode == CredentialMode::Sas
        && !https
        && !service_url.host_str().is_some_and(is_loopback_host)
    {
        bail!(
            "Azure credential mode sas: refusing to send the SAS token over plain http to {plain_url}; use an https endpoint (http is allowed only for localhost / loopback emulators)"
        );
    }
    let mut key = format!("{mode}|{plain_url}");
    if mode == CredentialMode::Sas {
        use std::hash::{Hash, Hasher};
        let token = azure_credential::sas_token(&|n| std::env::var(n).ok())?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        key.push_str(&format!("|sas:{:016x}", hasher.finish()));
        service_url.set_query(Some(&token));
    }

    service_clients().get_or_build(key, || {
        let credential = if mode.uses_bearer_token() {
            if !https {
                bail!(
                    "Azure credential mode {mode} sends a bearer token and needs an https \
                     endpoint, but the endpoint is {plain_url}; use S3DLIO_AZURE_CREDENTIAL=none \
                     or sas for an http endpoint such as Azurite"
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
            .push(plain_url.clone());
        let client = Arc::new(
            BlobServiceClient::new(
                service_url,
                credential,
                Some(BlobServiceClientOptions::default()),
            )
            .map_err(|e| anyhow!(azure_credential::redact_sas(&e.to_string())))?,
        );
        Ok((client, mode))
    })
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

// ---------------------------------------------------------------------------
// Caller-buffer GET and typed HEAD: pure pieces (plan v0.10.0 §4.3, §4.5 A8; #264)
// ---------------------------------------------------------------------------

/// A typed Azure error that is not an HTTP failure of the service: a response that cannot be
/// trusted or used (a bad `Content-Range`, a missing size or ETag). Classified
/// [`ObjectErrorKind::Other`], so it is never retried.
fn az_other(message: impl std::fmt::Display) -> anyhow::Error {
    crate::object_error::attach_object_error(
        anyhow!("{message}"),
        ObjectErrorKind::Other,
        "az",
        "unusable response",
    )
}

/// The typed error for a length that is not the expected one. `actual` is `None` when the true
/// size is not known (a body that ran past the destination).
fn az_length_mismatch(expected: u64, actual: Option<u64>, detail: &str) -> anyhow::Error {
    crate::object_error::attach_object_error(
        anyhow!("Azure blob length mismatch: expected {expected} bytes, {detail}"),
        ObjectErrorKind::LengthMismatch { expected, actual },
        "az",
        "",
    )
}

/// A `usize` length as `u64`, checked (never `as`).
fn len_u64(len: usize) -> Result<u64> {
    u64::try_from(len).map_err(|_| az_other("a length does not fit in 64 bits"))
}

/// A parsed `Content-Range` response header (RFC 9110 section 14.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContentRangeValue {
    /// The first and last byte of the part, both inclusive; `None` for `*`.
    pub(crate) bytes: Option<(u64, u64)>,
    /// The complete length of the blob; `None` for `*` (unknown).
    pub(crate) total: Option<u64>,
}

/// An unsigned decimal: ASCII digits only (no sign, no whitespace), within `u64`.
fn parse_decimal(text: &str) -> Option<u64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Parse `bytes <first>-<last>/<total>`, `bytes <first>-<last>/*` or `bytes */<total>`.
///
/// The SDK has its own parser, but its type is crate-private and it takes `usize`; the reply's
/// size is checked here, in `u64`, before any byte is copied. Strict: the unit must be `bytes`,
/// numbers are plain digits, `first <= last`, `last < total` when the total is known, and
/// `*/*` is refused.
pub(crate) fn parse_content_range(value: &str) -> Result<ContentRangeValue> {
    let shown: String = value.chars().take(64).collect();
    let malformed = || az_other(format!("malformed Content-Range header {shown:?}"));
    let rest = value.trim().strip_prefix("bytes ").ok_or_else(malformed)?;
    let (range, total) = rest.split_once('/').ok_or_else(malformed)?;
    let bytes = if range == "*" {
        None
    } else {
        let (first, last) = range.split_once('-').ok_or_else(malformed)?;
        let first = parse_decimal(first).ok_or_else(malformed)?;
        let last = parse_decimal(last).ok_or_else(malformed)?;
        if first > last {
            return Err(malformed());
        }
        Some((first, last))
    };
    let total = if total == "*" {
        None
    } else {
        Some(parse_decimal(total).ok_or_else(malformed)?)
    };
    match (bytes, total) {
        (None, None) => Err(malformed()),
        (Some((_, last)), Some(total)) if last >= total => Err(malformed()),
        _ => Ok(ContentRangeValue { bytes, total }),
    }
}

/// What a first response says about the blob and about the body it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResponseLengths {
    /// The whole blob, in bytes.
    pub(crate) total: u64,
    /// How many body bytes this response carries.
    pub(crate) body_len: u64,
}

/// The blob size and the body length a download response declares, for a request for bytes
/// `0..request_len`. The caller-buffer GET passes `request_len = expected_len + 1` (review C-F1,
/// see `checked_request_len`).
///
/// * With a `Content-Range` (a 206): the range must start at 0, the total must be known
///   (`*` cannot be checked), the range must be exactly `min(total, request_len)` bytes, and a
///   `Content-Length` must agree with it.
/// * Without one (a 200: the service ignored the range, or an unranged retry after the 416 that
///   an empty blob gives): `Content-Length` is the blob size and the body length. The SDK does not
///   expose the status, so a missing `Content-Range` is read as a whole-blob reply.
///
/// A 206 whose `Content-Range` was stripped (by a broken service or proxy) is read the same way,
/// but it cannot pass for the whole blob: the request is one byte longer than the expected size
/// `N`, so a blob larger than `N` answers with at least `N + 1` bytes, and the caller, which
/// requires a total of exactly `N`, refuses it as a `LengthMismatch` before copying anything.
/// A blob of exactly `N` bytes answers `N` bytes, which is its true size.
pub(crate) fn response_lengths(
    content_range: Option<&str>,
    content_length: Option<u64>,
    request_len: u64,
) -> Result<ResponseLengths> {
    let Some(header) = content_range else {
        let length = content_length
            .ok_or_else(|| az_other("the response has neither Content-Range nor Content-Length"))?;
        return Ok(ResponseLengths {
            total: length,
            body_len: length,
        });
    };
    let parsed = parse_content_range(header)?;
    let (first, last) = parsed
        .bytes
        .ok_or_else(|| az_other("a successful response carries a Content-Range without a range"))?;
    let total = parsed.total.ok_or_else(|| {
        az_other("the Content-Range total is unknown (*), so the blob size cannot be checked")
    })?;
    if first != 0 {
        return Err(az_other(format!(
            "the response range starts at byte {first}, not 0"
        )));
    }
    // `last < total` was checked by the parser, so `last + 1` cannot overflow.
    let body_len = last
        .checked_add(1)
        .ok_or_else(|| az_other("the Content-Range end overflows"))?;
    if body_len != total.min(request_len) {
        return Err(az_other(format!(
            "the response range covers {body_len} bytes, but bytes 0..{request_len} of a \
             {total}-byte blob were requested"
        )));
    }
    if content_length.is_some_and(|length| length != body_len) {
        return Err(az_other(
            "the Content-Length disagrees with the Content-Range",
        ));
    }
    Ok(ResponseLengths { total, body_len })
}

/// The length of the range a caller-buffer GET of `expected_len` bytes asks for: `expected_len +
/// 1` (review C-F1), checked. The one extra byte is a probe, never a byte of the result: a blob
/// of exactly `expected_len` bytes ends inside the range, and a larger one exposes at least
/// `expected_len + 1` bytes even when the response carries no `Content-Range` (a proxy that
/// strips it from a 206), which `response_lengths` then refuses before anything is copied. An
/// `expected_len == 0` request is the one-byte probe `0..1`. A length whose successor overflows
/// is refused here, before any request is built or sent.
fn checked_request_len(expected_len: u64) -> Result<u64> {
    expected_len
        .checked_add(1)
        .ok_or_else(|| az_other("the expected length is too large to request one byte beyond it"))
}

/// The download options for a caller-buffer GET of `expected_len` bytes (plan §4.3, amended by
/// review C-F1).
///
/// One bounded range `0..N+1` (`N = expected_len`; see `checked_request_len`: the extra byte
/// detects a blob larger than `N`, and `expected_len == 0` asks for one byte), `parallel = 1` and
/// `partition_size = N+1`. The destination and the expected blob size stay `N`. That makes the
/// SDK's first response the only one it needs. In `azure_storage_blob` 1.1.0
/// `partitioned_transfer/download.rs` (`analyze_initial_response`, line 524) computes
/// `remainder_end = min(request_end, resource_len)` and the remaining ranges as
/// `(received_end..remainder_end)`:
///
/// * a blob of exactly `N` bytes: `resource_len = N`, so `remainder_end = min(N+1, N) = N`, the
///   response ended at `received_end = N`, and the remaining ranges are empty;
/// * a larger blob: the response covers `0..N+1` (`received_end = N+1`) and `remainder_end =
///   min(N+1, resource_len) = N+1`, so the remaining ranges are empty as well.
///
/// When the remaining ranges are empty, `download` (the `remaining_ranges.is_empty()` branch, line 84)
/// returns the first response's body directly, with no partition buffer and no worker task. So
/// the exact-size read keeps the no-partition-buffer path, and so does the oversized one, which
/// is refused from its headers. `if_match` pins a retry to the first response's ETag.
///
/// The no-remaining-ranges path assumes the service honours the range. A reply shorter than the
/// range asked for (a short 206 that is not the tail of the blob) leaves the SDK with ranges
/// still to fetch, so it starts its partition task (a zero-filled buffer of `partition_size`
/// bytes that reads the first body) BEFORE `get_into` sees the response and refuses it (see
/// `response_lengths`). The refusal still happens before any byte reaches the destination, but
/// that buffer exists, and the SDK's follow-up requests for the missing ranges are driven only
/// if the refused response's body is polled, which `get_into` does not do.
fn get_into_options(
    expected_len: u64,
    if_match: Option<&str>,
) -> Result<BlobClientDownloadOptions<'static>> {
    let range_len = checked_request_len(expected_len)?;
    let partition_size = usize::try_from(range_len)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| az_other("the expected length does not fit in this platform's memory"))?;
    Ok(BlobClientDownloadOptions {
        range: Some(HttpRange::new(0, range_len)),
        parallel: NonZeroUsize::new(1),
        partition_size: Some(partition_size),
        if_match: if_match.map(azure_core::http::Etag::from),
        ..Default::default()
    })
}

/// An ETag that can pin a read: present, non-empty and strong (`If-Match` compares strongly, so
/// a weak `W/"..."` tag cannot match itself).
fn usable_etag(etag: Option<&azure_core::http::Etag>) -> Option<String> {
    let text = AsRef::<str>::as_ref(etag?).trim();
    (!text.is_empty() && !text.starts_with("W/")).then(|| text.to_string())
}

/// How one ranged request relates to the version of the blob (stage G review, G-F2).
enum RangeVersion<'a> {
    /// No version is checked: a request that is the whole read, or that stands alone.
    Unpinned,
    /// The request carries `If-Match: etag` and its reply must carry the same strong ETag.
    Pinned(&'a str),
    /// The first range of a read of several: sent unpinned, and the strong ETag of its reply (as
    /// soon as the headers arrive, before the body) is reported to the other ranges through the
    /// leader, or "unavailable" when the reply has none.
    Lead(&'a Leader<'a, String>),
}

/// The blob's sizes as a first partition reports them, for a blob larger than that partition.
#[derive(Clone, Copy)]
struct LargeBlob {
    /// The whole blob, in bytes.
    total: u64,
    /// How many bytes the first response carries.
    first_body_len: u64,
}

/// How many body bytes a response to a ranged request for bytes starting at `start` carries,
/// from its headers; checked arithmetic throughout, because the headers are the server's.
///
/// * With a `Content-Range`: it must give a range that starts at `start` (anything else would
///   put the wrong bytes at the window), and a `Content-Length` must agree with it.
/// * Without one: only a whole-blob reply to a range that starts at 0 can be read this way (the
///   service ignored the range, or the SDK repeated the request without a range for an empty
///   blob); then `Content-Length` is the body length. For `start > 0` it is an error.
fn range_response_body_len(
    content_range: Option<&str>,
    content_length: Option<u64>,
    start: u64,
) -> Result<u64> {
    let Some(header) = content_range else {
        if start != 0 {
            return Err(az_other(
                "the response to a range request carries no Content-Range",
            ));
        }
        return content_length
            .ok_or_else(|| az_other("the response has neither Content-Range nor Content-Length"));
    };
    let parsed = parse_content_range(header)?;
    let (first, last) = parsed
        .bytes
        .ok_or_else(|| az_other("a successful response carries a Content-Range without a range"))?;
    if first != start {
        return Err(az_other(format!(
            "the response range starts at byte {first}, not {start}"
        )));
    }
    let body_len = last
        .checked_sub(first)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| az_other("the Content-Range length overflows"))?;
    if content_length.is_some_and(|length| length != body_len) {
        return Err(az_other(
            "the Content-Length disagrees with the Content-Range",
        ));
    }
    Ok(body_len)
}

/// Receive a response body into `dst`, which is the exact expected length.
///
/// A checked loop over the public response-body stream (not the SDK's `collect_into`, which
/// returns early on a mid-stream error and loses the partial count). Each chunk is bounds-checked
/// BEFORE it is copied, so nothing is ever written past `dst`; the progress is recorded after
/// each copy, so it survives an error and cancellation (dropping this future). The cursor starts
/// at 0 for every attempt; `span` keeps the high-water mark across attempts.
///
/// * a chunk that would pass the end of `dst`: `LengthMismatch` (the chunk is not copied);
/// * a stream that ends before `dst` is full: a `Transport` error (a truncated transfer);
/// * a stream error: an [`AzureRequestError`] (SAS redacted, credential mode named for a 401/403).
async fn receive_into<S>(
    mut body: S,
    dst: &mut [u8],
    span: &mut SpanMark<'_>,
    mode: CredentialMode,
) -> Result<()>
where
    S: Stream<Item = azure_core::Result<Bytes>> + Unpin,
{
    let expected = len_u64(dst.len())?;
    let mut cursor: usize = 0;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| azure_error(mode, e))?;
        if chunk.is_empty() {
            continue;
        }
        let end = cursor
            .checked_add(chunk.len())
            .filter(|&end| end <= dst.len())
            .ok_or_else(|| az_length_mismatch(expected, None, "the body is longer"))?;
        let window = dst
            .get_mut(cursor..end)
            .ok_or_else(|| az_other("a chunk does not fit the destination"))?;
        window.copy_from_slice(&chunk);
        span.record_copy(len_u64(cursor)?, len_u64(chunk.len())?)
            .map_err(|e| az_other(format!("progress accounting failed: {e}")))?;
        cursor = end;
    }
    if cursor != dst.len() {
        return Err(crate::object_error::attach_object_error(
            anyhow!(
                "Azure response body ended after {cursor} of {} bytes",
                dst.len()
            ),
            ObjectErrorKind::Transport,
            "az",
            "truncated response body",
        ));
    }
    Ok(())
}

/// Whether a failed `get_into` attempt may be repeated (plan §4.3, §4.5 A2).
///
/// Only a typed transient failure (`Throttled`, `Transport`, `Server`) is retried, never a
/// definite answer or an unclassified error. And only while every byte already received is
/// bound to one version: a retry is `If-Match`-pinned to the first response's ETag, or nothing
/// has been written yet. Unpinned responses are never combined.
fn may_retry_get_into(err: &anyhow::Error, pinned: bool, written: u64) -> bool {
    let transient = crate::object_error::object_error(err).is_some_and(|o| {
        matches!(
            o.kind,
            ObjectErrorKind::Throttled | ObjectErrorKind::Transport | ObjectErrorKind::Server
        )
    });
    transient && (pinned || written == 0)
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
    fn account_url_from_account(account: &str) -> Result<String> {
        Ok(format!(
            "https://{}.blob.core.windows.net",
            canonical_account_name(account)?
        ))
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
        crate::dotenv::apply();
        let account = canonical_account_name(account)?;
        let account = account.as_str();
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
            tracing::info!("Using custom Azure endpoint: {}", redact_url(&account_url));
            return Self::with_default_credential_from_url(&account_url, container);
        }

        Self::with_default_credential_from_url(&Self::account_url_from_account(account)?, container)
    }

    /// Same, when a full endpoint URL (possibly emulator) is provided.
    pub fn with_default_credential_from_url(account_url: &str, container: &str) -> Result<Self> {
        crate::dotenv::apply();
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
        azure_error(self.credential_mode, e)
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
        // The range length, with checked arithmetic: `get_range(0, Some(u64::MAX))` is a
        // legal "to the end" request and must not overflow.
        let range_len = match end {
            Some(e) => Some(
                e.checked_sub(start)
                    .and_then(|d| d.checked_add(1))
                    .ok_or_else(|| {
                        anyhow!("invalid Azure byte range: end {e} precedes start {start}, or the range is too long")
                    })?,
            ),
            None => None,
        };
        let range = match range_len {
            Some(len) => HttpRange::new(start, len),
            None => HttpRange::from_offset(start),
        };
        // One request for the whole range: with the SDK's default 4 MiB partition a 64 MiB
        // range was sixteen serial sub-requests, each through an SDK partition buffer. An
        // open-ended range uses the default download partition (32 MiB).
        let partition_size = match range_len {
            Some(len) => usize::try_from(len).ok(),
            None => Some(crate::constants::DEFAULT_AZURE_DOWNLOAD_PART_SIZE),
        };
        let opts = BlobClientDownloadOptions {
            range: Some(range),
            parallel: NonZeroUsize::new(1),
            partition_size: partition_size.and_then(NonZeroUsize::new),
            ..Default::default()
        };
        let resp = blob
            .download(Some(opts))
            .await
            .map_err(|e| self.auth_err(e))?;
        // The response says how many bytes it carries (one request covers the range): reserve
        // that. Without the header, reserve the requested length, capped.
        let expected = range_expected_len(header_u64(&resp.headers, "content-length"), range_len);
        let body = collect_body(resp.body, expected).await?;
        debug!(
            "AzureBlob::get_range success: key='{}', {} bytes",
            key,
            body.len()
        );
        Ok(body)
    }

    /// Range GET into the caller's buffer (stage G, R3): the bytes `start..start + dst.len()` of
    /// the blob are received straight into `dst`, one copy from each SDK chunk to its place, and
    /// the number of bytes written is returned.
    ///
    /// That number is `dst.len()` unless the range runs past the end of the blob, which the
    /// service answers by sending only what exists: the caller decides whether a shorter answer
    /// is an error (the range engine does, except for its one-byte growth probe). A range that
    /// STARTS at or past the end is the service's `416`, an error, with one exception: an EMPTY
    /// blob read from offset 0 is the SDK's unranged retry of that 416 and returns `Ok(0)`. A reply
    /// carrying more than `dst.len()` bytes, a `Content-Range` that does not start at `start`, or
    /// (for `start > 0`) no `Content-Range` at all, is refused before any byte is copied. No byte
    /// is written past `dst`. `progress` counts the positions written.
    pub async fn get_range_into(
        &self,
        key: &str,
        start: u64,
        dst: &mut [u8],
        progress: &ReceiveProgress,
    ) -> Result<usize> {
        let wanted = len_u64(dst.len())?;
        debug!(
            "AzureBlob::get_range_into container='{}', key='{}', start={}, len={}",
            self.container, key, start, wanted
        );
        if dst.is_empty() {
            return Ok(0);
        }
        let blob = self.blob_client(key)?;
        let mut span = SpanMark::new(progress);
        self.fetch_range_into(&blob, start, dst, &mut span, RangeVersion::Unpinned)
            .await
    }

    /// [`Self::get_range_into`] for one range of a read of SEVERAL ranges of the same blob (the
    /// range engine, stage G review G-F2): all of them share `version`, so they are all of ONE
    /// version of the blob.
    ///
    /// The first range to ask leads: it is sent unpinned and reports the strong ETag of its reply
    /// as soon as the headers arrive. Every other range waits for that and is sent with
    /// `If-Match` of that ETag, so a blob replaced by another (of any size, the same size
    /// included) between two ranges is refused by the service (`412`) or, should a reply carry
    /// another ETag, by the check in [`Self::fetch_range_into`]: a `VersionChanged` error, never a
    /// body of two versions. A first reply with no strong ETag cannot pin anything: the other
    /// ranges fail at once (see [`ObjectVersion::is_unavailable`]) and the caller reads the blob as
    /// one response. Nothing else differs from [`Self::get_range_into`]; each chunk is still
    /// copied once, into its place in `dst`.
    pub(crate) async fn get_range_into_versioned(
        &self,
        key: &str,
        start: u64,
        dst: &mut [u8],
        progress: &ReceiveProgress,
        version: &ObjectVersion<String>,
    ) -> Result<usize> {
        if dst.is_empty() {
            return Ok(0);
        }
        let blob = self.blob_client(key)?;
        let mut span = SpanMark::new(progress);
        match version.join().await? {
            Role::Follower(etag) => self
                .fetch_range_into(&blob, start, dst, &mut span, RangeVersion::Pinned(&etag))
                .await
                .map_err(|e| attach_azure_classification(e, true)),
            Role::Leader(leader) => {
                let result = self
                    .fetch_range_into(&blob, start, dst, &mut span, RangeVersion::Lead(&leader))
                    .await;
                if let Err(e) = &result {
                    // Releases the other ranges with the reason (a no-op once it has reported).
                    leader.fail(format!("{e:#}"));
                }
                result
            }
        }
    }

    /// One ranged request for `start..start + dst.len()` and its checked receive into `dst`;
    /// returns the number of bytes written (see [`Self::get_range_into`]). `version` says how the
    /// request relates to the blob's version: `Pinned` sends `If-Match` (the service answers 412
    /// for another version, and a reply that does not carry the same strong ETag is a
    /// `VersionChanged`), `Lead` reports the reply's strong ETag to the other ranges of the read,
    /// `Unpinned` checks nothing.
    async fn fetch_range_into(
        &self,
        blob: &BlobClient,
        start: u64,
        dst: &mut [u8],
        span: &mut SpanMark<'_>,
        version: RangeVersion<'_>,
    ) -> Result<usize> {
        let wanted = len_u64(dst.len())?;
        start
            .checked_add(wanted)
            .ok_or_else(|| az_other("the byte range runs past the largest possible offset"))?;
        let partition_size = NonZeroUsize::new(dst.len())
            .ok_or_else(|| az_other("a zero-length destination has no range"))?;
        // One request for the whole range, and the SDK hands its body up without a partition
        // buffer (the range ends where the first partition does: nothing remains to fetch).
        let options = BlobClientDownloadOptions {
            range: Some(HttpRange::new(start, wanted)),
            parallel: NonZeroUsize::new(1),
            partition_size: Some(partition_size),
            if_match: match version {
                RangeVersion::Pinned(pinned) => Some(azure_core::http::Etag::from(pinned)),
                RangeVersion::Unpinned | RangeVersion::Lead(_) => None,
            },
            ..Default::default()
        };
        let resp = blob
            .download(Some(options))
            .await
            .map_err(|e| self.auth_err(e))?;
        let response_etag = usable_etag(resp.properties.etag.as_ref());
        match version {
            RangeVersion::Unpinned => {}
            RangeVersion::Pinned(pinned) => {
                if response_etag.as_deref() != Some(pinned) {
                    return Err(crate::object_error::attach_object_error(
                        anyhow!("the blob's ETag changed between requests"),
                        ObjectErrorKind::VersionChanged,
                        "az",
                        "response ETag differs from the pinned one",
                    ));
                }
            }
            RangeVersion::Lead(leader) => match response_etag {
                Some(etag) => leader.publish(etag),
                None => leader.unavailable(),
            },
        }
        let content_range = resp
            .headers
            .get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                "content-range",
            ))
            .map(str::to_string);
        let body_len = range_response_body_len(
            content_range.as_deref(),
            header_u64(&resp.headers, "content-length"),
            start,
        )?;
        if body_len > wanted {
            return Err(az_length_mismatch(
                wanted,
                None,
                "the response carries more bytes than the range asked for",
            ));
        }
        let received = usize::try_from(body_len)
            .map_err(|_| az_other("the response length does not fit in memory"))?;
        let window = dst
            .get_mut(..received)
            .ok_or_else(|| az_other("the response does not fit the destination"))?;
        receive_into(resp.body, window, span, self.credential_mode).await?;
        Ok(received)
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
    /// once (issue #252; the SDK's own `collect()` grows its buffer as chunks arrive).
    ///
    /// `expected_len` is only a hint, possibly stale: the length the server reports in this very
    /// response is authoritative and the hint decides nothing.
    ///
    /// # Large blobs: one copy (stage G, R5)
    ///
    /// The first request is one ranged GET for the first partition (`partition_size`; the SDK's
    /// own default of 4 MiB when `None`), which is also how the SDK's managed download starts
    /// and needs no `HEAD`: its `Content-Range` carries the blob's size. A blob that fits in
    /// that partition is received from that response into one buffer sized from the response.
    ///
    /// A larger blob is received into ONE buffer of the blob's size: the first response's body
    /// goes straight into the front of it, and the rest is read in ranged requests of one
    /// partition each, `concurrency` at a time, each received straight into its own disjoint
    /// slice of the buffer and pinned to the first response's ETag with `If-Match`. The ranged
    /// reads are futures of THIS call (a bounded `FuturesUnordered`; nothing is spawned), so
    /// dropping or cancelling the call, or the first failed partition, stops every request.
    /// Each received chunk is copied exactly once, into its final place. The SDK's plain
    /// `download` instead copies every partition into a buffer of its own and hands those up, and
    /// a second copy into the destination follows: twice the blob live and two copies of every
    /// byte. (The SDK's `download_into` is not used: it spawns a manager task that nobody aborts,
    /// which keeps requesting the rest of the blob after the caller has gone.)
    ///
    /// A first response that cannot be pinned (no ETag, or a weak one) starts no ranged tail:
    /// unpinned responses are never combined, so the blob is read again as ONE request into the
    /// buffer ([`Self::get_into`]). A first response with no `Content-Range` that is exactly as
    /// long as the range asked for cannot be told from a truncated blob and is an error.
    pub async fn get_with_sdk_concurrency_sized(
        &self,
        key: &str,
        concurrency: Option<usize>,
        partition_size: Option<usize>,
        expected_len: Option<u64>,
    ) -> Result<Bytes> {
        debug!(
            "AzureBlob::get container='{}', key='{}', size hint {:?}",
            self.container, key, expected_len
        );
        let blob = self.blob_client(key)?;
        let first_len = partition_size
            .filter(|p| *p > 0)
            .unwrap_or(SDK_DEFAULT_DOWNLOAD_PARTITION);
        let first_requested = len_u64(first_len)?;
        let first_options = BlobClientDownloadOptions {
            range: Some(HttpRange::new(0, first_requested)),
            parallel: NonZeroUsize::new(1),
            partition_size: NonZeroUsize::new(first_len),
            ..Default::default()
        };
        let resp = blob
            .download(Some(first_options))
            .await
            .map_err(|e| self.auth_err(e))?;
        let content_range = resp
            .headers
            .get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                "content-range",
            ))
            .map(str::to_string);
        let content_length = header_u64(&resp.headers, "content-length");

        let whole_blob_len = match content_range.as_deref() {
            Some(header) => {
                let parsed = parse_content_range(header)?;
                let (Some((first_at, last)), Some(total)) = (parsed.bytes, parsed.total) else {
                    return Err(az_other(
                        "the response's Content-Range does not give both the range and the \
                         blob's size, so the blob cannot be read completely",
                    ));
                };
                if first_at != 0 {
                    return Err(az_other(format!(
                        "the response range starts at byte {first_at}, not 0"
                    )));
                }
                let first_body_len = last
                    .checked_add(1)
                    .ok_or_else(|| az_other("the Content-Range end overflows"))?;
                if content_length.is_some_and(|length| length != first_body_len) {
                    return Err(az_other(
                        "the Content-Length disagrees with the Content-Range",
                    ));
                }
                if total > first_body_len {
                    let body = self
                        .receive_large_blob(
                            &blob,
                            key,
                            resp,
                            LargeBlob {
                                total,
                                first_body_len,
                            },
                            concurrency,
                            first_len,
                        )
                        .await?;
                    debug!(
                        "AzureBlob::get success: key='{}', {} bytes",
                        key,
                        body.len()
                    );
                    return Ok(body);
                }
                total
            }
            // No `Content-Range`. A reply SHORTER than the range asked for is the whole blob
            // (a range of a larger blob always carries the full range), and a LONGER one is a
            // service that ignored the range and sent the whole blob. A reply exactly as long
            // as the range cannot be told from the first partition of a larger blob whose
            // `Content-Range` was stripped: taking it for the whole blob would silently
            // truncate it, so it is refused.
            None => match content_length {
                Some(length) if length == first_requested => {
                    return Err(az_other(format!(
                        "the response to a request for the first {first_requested} bytes carries \
                         no Content-Range and is exactly that long: it cannot be told from a \
                         truncated read of a larger blob"
                    )));
                }
                Some(length) => length,
                None => {
                    return Err(az_other(
                        "the response has neither Content-Range nor Content-Length, so the \
                         blob cannot be read completely",
                    ));
                }
            },
        };

        // The whole blob is in this one response.
        let body = collect_body(resp.body, Some(whole_blob_len)).await?;
        debug!(
            "AzureBlob::get success: key='{}', {} bytes",
            key,
            body.len()
        );
        Ok(body)
    }

    /// The large-blob half of [`Self::get_with_sdk_concurrency_sized`]: the first response (the
    /// first `blob_sizes.first_body_len` bytes of a `blob_sizes.total`-byte blob) into the front
    /// of one buffer, the rest in ranged reads of `partition` bytes, `concurrency` at a time,
    /// into the remainder, all driven by this one future.
    async fn receive_large_blob(
        &self,
        blob: &BlobClient,
        key: &str,
        first: azure_storage_blob::models::BlobClientDownloadResult,
        blob_sizes: LargeBlob,
        concurrency: Option<usize>,
        partition: usize,
    ) -> Result<Bytes> {
        let LargeBlob {
            total,
            first_body_len,
        } = blob_sizes;
        let total_usize = usize::try_from(total)
            .map_err(|_| az_other("the blob is larger than this platform's memory"))?;
        let head_len = usize::try_from(first_body_len)
            .map_err(|_| az_other("the first range does not fit in memory"))?;
        let mut buffer = crate::read_buffer::try_zeroed_vec(total_usize).ok_or_else(|| {
            az_other(format!(
                "cannot allocate {total} bytes for the blob the service reports"
            ))
        })?;

        // The remaining ranges are pinned to the version the first response came from. Without
        // a strong ETag nothing can pin them, and unpinned responses are never combined: read
        // the whole blob again as one request.
        let Some(etag) = usable_etag(first.properties.etag.as_ref()) else {
            drop(first);
            debug!(
                "AzureBlob::get key='{}': the first response has no strong ETag; reading the \
                 {} bytes as one request",
                key, total
            );
            self.get_into(key, &mut buffer, &ReceiveProgress::new())
                .await?;
            return Ok(Bytes::from(buffer));
        };

        let (head, tail) = buffer
            .split_at_mut_checked(head_len)
            .ok_or_else(|| az_other("the first range is longer than the blob"))?;
        let concurrency = concurrency.filter(|c| *c > 0).unwrap_or_else(|| {
            // The SDK's own default for a managed download.
            std::thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1)
                .clamp(8, 96)
        });

        let first_progress = ReceiveProgress::new();
        let mut first_span = SpanMark::new(&first_progress);
        let first_part = receive_into(first.body, head, &mut first_span, self.credential_mode);
        let rest =
            self.receive_tail_parts(blob, tail, first_body_len, partition, concurrency, &etag);
        futures::try_join!(first_part, rest)?;
        // `Bytes::from(Vec)` takes the allocation over: no copy.
        Ok(Bytes::from(buffer))
    }

    /// Read `tail` (the blob's bytes from `tail_start` on) in ranged requests of `partition`
    /// bytes, at most `concurrency` at a time, each straight into its own slice of `tail`, every
    /// request carrying `If-Match: etag`. The requests are futures of this call: the first error
    /// returns at once and drops the others, and so does dropping this future.
    async fn receive_tail_parts(
        &self,
        blob: &BlobClient,
        tail: &mut [u8],
        tail_start: u64,
        partition: usize,
        concurrency: usize,
        etag: &str,
    ) -> Result<()> {
        let mut parts = tail
            .chunks_mut(partition.max(1))
            .scan(tail_start, |offset, slice| {
                let at = *offset;
                *offset = offset.saturating_add(u64::try_from(slice.len()).unwrap_or(u64::MAX));
                Some((at, slice))
            });
        let mut in_flight = FuturesUnordered::new();
        loop {
            while in_flight.len() < concurrency {
                let Some((at, slice)) = parts.next() else {
                    break;
                };
                in_flight.push(self.receive_tail_part(blob, at, slice, etag));
            }
            match in_flight.next().await {
                Some(Ok(())) => {}
                Some(Err(e)) => return Err(e),
                None => return Ok(()),
            }
        }
    }

    /// One tail part: bytes `at..at + part.len()`, pinned by `If-Match`, exactly that long.
    async fn receive_tail_part(
        &self,
        blob: &BlobClient,
        at: u64,
        part: &mut [u8],
        etag: &str,
    ) -> Result<()> {
        let wanted = part.len();
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let received = self
            .fetch_range_into(blob, at, part, &mut span, RangeVersion::Pinned(etag))
            .await
            .map_err(|e| attach_azure_classification(e, true))?;
        if received != wanted {
            return Err(az_length_mismatch(
                len_u64(wanted)?,
                Some(len_u64(received)?),
                "a ranged part of the blob came back short",
            ));
        }
        Ok(())
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

    /// Typed HEAD (plan §4.3, §4.5 A8): the blob's size, ETag and version id, or `None` when the
    /// blob does not exist.
    ///
    /// Only `BlobNotFound` is absence. A missing container (`ContainerNotFound`) is an error
    /// (`Other`), as are authorization, throttling, transport and server failures, each typed.
    /// The size is parsed as an unsigned number by the SDK (a negative or non-numeric
    /// `Content-Length` is an error), and a response without one is an error: there is no
    /// default of 0.
    pub async fn head_typed(&self, key: &str) -> Result<Option<HeadMetadata>> {
        debug!(
            "AzureBlob::head_typed container='{}', key='{}'",
            self.container, key
        );
        let blob = self.blob_client(key)?;
        let resp = match blob
            .get_properties(Some(BlobClientGetPropertiesOptions::default()))
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                let err = attach_azure_classification(self.auth_err(e), false);
                return match crate::object_error::object_error(&err).map(|o| o.kind) {
                    Some(ObjectErrorKind::NotFound) => Ok(None),
                    _ => Err(err),
                };
            }
        };
        let header_error = |what: &str, e: azure_core::Error| {
            attach_azure_classification(
                self.auth_err(e)
                    .context(format!("invalid {what} header in a HEAD response")),
                false,
            )
        };
        let size = resp
            .content_length()
            .map_err(|e| header_error("Content-Length", e))?
            .ok_or_else(|| az_other("the HEAD response has no Content-Length"))?;
        let etag = resp.etag().map_err(|e| header_error("ETag", e))?;
        let version_id = resp
            .version_id()
            .map_err(|e| header_error("x-ms-version-id", e))?;
        Ok(Some(HeadMetadata {
            size,
            etag: etag.map(|e| e.to_string()).filter(|e| !e.trim().is_empty()),
            version_id: version_id.filter(|v| !v.trim().is_empty()),
        }))
    }

    /// Receive the whole blob into `dst` (plan §4.3, §4.5 A8). `dst.len()` is the expected
    /// length: `Ok(())` means exactly that many bytes were written and the blob is exactly that
    /// long.
    ///
    /// * `dst.len() > 0`: one bounded range `0..dst.len() + 1` (one byte more than the
    ///   destination, review C-F1), `parallel = 1`,
    ///   `partition_size = dst.len() + 1` (see `get_into_options`), so the SDK returns the first
    ///   response's body directly, with no partition buffer. The blob size comes from the
    ///   `Content-Range` total (206) or `Content-Length` (200) and must equal `dst.len()` BEFORE
    ///   any byte is copied; otherwise `LengthMismatch` (a non-empty oversized or undersized blob
    ///   is a mismatch, as is an empty one).
    /// * `dst.len() == 0`: a bounded `0..1` request. An empty blob answers 416, which the SDK
    ///   retries unranged, giving a 200 of length 0: success. A non-empty blob is a
    ///   `LengthMismatch`.
    /// * Retries keep the first response's ETag (`If-Match`). A 412, or a 404 on a pinned retry,
    ///   is `VersionChanged`. A response without a strong ETag pins nothing: once a byte has
    ///   been written, a failure is final rather than retried from pieces of unknown versions.
    ///
    /// * Two retry layers act on a request. The SDK pipeline's own retries (408, 429, 500, 502,
    ///   503, 504 and every connection or I/O failure; `max_retries` 8 by default, exponential
    ///   back-off from 200 ms, 60 s in all) run INSIDE each s3dlio attempt and resend the same
    ///   request, `If-Match` included. `S3DLIO_MAX_RETRY_ATTEMPTS` counts s3dlio attempts only,
    ///   so one attempt that keeps failing with a status the SDK retries can send up to nine
    ///   requests.
    ///
    /// The SDK hides the HTTP status, so a 206 whose `Content-Range` was stripped cannot be told
    /// from a 200. The extra requested byte makes that harmless: a blob larger than `dst.len()`
    /// answers with more than `dst.len()` bytes and is a `LengthMismatch` (see
    /// `response_lengths`); the extra byte is never written to `dst`.
    ///
    /// `progress` counts the distinct positions written and survives errors and cancellation.
    /// Every writer runs inside this future, so dropping it stops all writes. A failure never
    /// says the bytes written are valid.
    pub async fn get_into(
        &self,
        key: &str,
        dst: &mut [u8],
        progress: &ReceiveProgress,
    ) -> Result<()> {
        // The destination length is the expected length, as a checked u64.
        let expected = len_u64(dst.len())?;
        debug!(
            "AzureBlob::get_into container='{}', key='{}', expected_len={}",
            self.container, key, expected
        );
        let blob = self.blob_client(key)?;
        let mut span = SpanMark::new(progress);
        let mut pin: Option<String> = None;
        let attempts = crate::constants::max_retry_attempts();
        let mut attempt: u32 = 1;
        loop {
            let err = match self
                .get_into_attempt(&blob, expected, dst, &mut span, &mut pin)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => attach_azure_classification(e, pin.is_some()),
            };
            // Every failure leaves here typed.
            let err = if crate::object_error::object_error(&err).is_some() {
                err
            } else {
                crate::object_error::attach_object_error(
                    err,
                    ObjectErrorKind::Other,
                    "az",
                    "request failed",
                )
            };
            if attempt >= attempts || !may_retry_get_into(&err, pin.is_some(), span.mark()) {
                return Err(err);
            }
            debug!("AzureBlob::get_into attempt {attempt} of {attempts} failed; retrying");
            tokio::time::sleep(std::time::Duration::from_millis(100 * u64::from(attempt))).await;
            attempt += 1;
        }
    }

    /// One request and its checked receive. Pins `pin` from the first response that carries a
    /// strong ETag, and requires every later response to carry the same one.
    async fn get_into_attempt(
        &self,
        blob: &BlobClient,
        expected: u64,
        dst: &mut [u8],
        span: &mut SpanMark<'_>,
        pin: &mut Option<String>,
    ) -> Result<()> {
        let options = get_into_options(expected, pin.as_deref())?;
        let resp = blob
            .download(Some(options))
            .await
            .map_err(|e| self.auth_err(e))?;

        let response_etag = usable_etag(resp.properties.etag.as_ref());
        match (pin.as_deref(), response_etag) {
            (Some(pinned), Some(seen)) if pinned != seen => {
                return Err(crate::object_error::attach_object_error(
                    anyhow!("the blob's ETag changed between attempts"),
                    ObjectErrorKind::VersionChanged,
                    "az",
                    "response ETag differs from the pinned one",
                ));
            }
            (Some(_), None) => {
                return Err(az_other(
                    "the response carries no usable ETag to confirm the pinned version",
                ));
            }
            (None, Some(seen)) => *pin = Some(seen),
            _ => {}
        }

        let lengths = response_lengths(
            resp.headers
                .get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                    "content-range",
                )),
            resp.properties.content_length,
            checked_request_len(expected)?,
        )?;
        if lengths.total != expected {
            // The body is dropped unread.
            return Err(az_length_mismatch(
                expected,
                Some(lengths.total),
                &format!("the blob is {} bytes", lengths.total),
            ));
        }
        if lengths.body_len != expected {
            return Err(az_length_mismatch(
                expected,
                Some(lengths.total),
                "the response body length differs",
            ));
        }
        receive_into(resp.body, dst, span, self.credential_mode).await
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
            match b.delete(None).await {
                Ok(_) => {}
                // Already gone is the state a delete asks for (S3's DeleteObjects treats a
                // missing key the same way); only the service's own `BlobNotFound` counts.
                Err(e) if azure_error_is_blob_not_found(&e) => {
                    debug!("AzureBlob::delete_objects: '{name}' was already absent");
                }
                Err(e) => return Err(self.auth_err(e)),
            }
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
///
/// The account is validated first ([`canonical_account_name`]): a name that could change the
/// host or path of the URL is an error, before anything is built or sent.
pub fn resolve_azure_account_url(account: &str) -> Result<String> {
    // The `.env` policy first (#243, review R9/F): a public setting reader (the endpoint may come from the opted-in file).
    crate::dotenv::apply();
    let account = canonical_account_name(account)?;
    Ok(
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
        },
    )
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
    crate::dotenv::apply();
    list_account_containers_at(&resolve_azure_account_url(account)?).await
}

/// [`list_account_containers`] against an explicit account URL.
async fn list_account_containers_at(account_url: &str) -> Result<Vec<(String, Option<String>)>> {
    let (service_client, mode) = service_client_for_url(account_url)?;

    let mut containers: Vec<(String, Option<String>)> = Vec::new();
    let mut pager = service_client
        .list_containers(None)
        .map_err(|e| azure_error(mode, e))?;

    while let Some(item) = pager.next().await {
        let item = item.map_err(|e| azure_error(mode, e))?;
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

    /// Serialise with other environment-changing tests and start from a known environment:
    /// nothing that selects an Azure credential or endpoint may leak in from the developer's
    /// shell (or from a test that ran before).
    fn clean_azure_env() -> std::sync::MutexGuard<'static, ()> {
        let guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        reset_azure_env();
        guard
    }

    /// The reset alone, for an async test that must not hold the lock across an await.
    fn reset_azure_env() {
        for (name, _) in std::env::vars() {
            if name.starts_with("AZURE_")
                || name.starts_with("S3DLIO_AZURE_")
                || matches!(
                    name.as_str(),
                    "IDENTITY_ENDPOINT"
                        | "IDENTITY_HEADER"
                        | "IMDS_ENDPOINT"
                        | "MSI_ENDPOINT"
                        | "MSI_SECRET"
                )
            {
                std::env::remove_var(name);
            }
        }
    }

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
        let _env = clean_azure_env();
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
    // A private cache instance, so the test cannot evict the global cache's entries that other
    // tests are counting.
    #[test]
    fn the_service_client_cache_is_bounded() {
        let cache = ServiceClientCache::new(3);
        let build = |n: usize| {
            move || {
                let url = Url::parse(&format!("http://127.0.0.1:59400/bounded{n}")).unwrap();
                let client = BlobServiceClient::new(url, None, None).map_err(|e| anyhow!(e))?;
                Ok((Arc::new(client), CredentialMode::None))
            }
        };
        let mut first = None;
        for i in 0..10 {
            let (client, _) = cache.get_or_build(format!("k{i}"), build(i)).unwrap();
            first.get_or_insert(client);
            assert!(
                cache.len() <= 3,
                "{} entries after {} accounts",
                cache.len(),
                i + 1
            );
        }
        assert_eq!(cache.len(), 3);
        // A hit does not build and does not evict.
        let (again, _) = cache
            .get_or_build("k9".into(), || -> Result<_> { panic!("must be a hit") })
            .unwrap();
        assert!(again.url().as_str().contains("bounded9"));
    }

    #[test]
    fn range_reads_reserve_at_most_the_cap_and_sequential_partitions_have_a_floor() {
        let cap = crate::constants::AZURE_RANGE_PRESIZE_MAX;
        assert_eq!(range_presize(None), None);
        assert_eq!(range_presize(Some(1000)), Some(1000));
        assert_eq!(range_presize(Some(cap)), Some(cap));
        assert_eq!(range_presize(Some(1 << 40)), Some(cap));

        let floor = crate::constants::AZURE_MIN_DOWNLOAD_PARTITION_SIZE;
        for stale in [0u64, 1, 4096, floor as u64 - 1] {
            assert_eq!(
                sequential_partition_size(stale),
                floor,
                "a stale {stale}-byte hint"
            );
        }
        assert_eq!(sequential_partition_size(floor as u64 * 4), floor * 4);
    }

    #[test]
    fn account_names_are_validated_strictly_without_a_custom_endpoint() {
        for good in ["abc", "devstoreaccount1", "a1b2c3", "x".repeat(24).as_str()] {
            assert_eq!(account_name_problem(good, false), None, "{good}");
        }
        for bad in [
            "",
            "ab",
            "UPPER",
            "evil.example",
            "with-dash",
            "under_score",
            "a?b",
            "a#b",
            "a/b",
            "a@b",
            "a\\b",
            "a b",
            "a\tb",
            &"x".repeat(25),
            "attacker.example#",
        ] {
            assert!(account_name_problem(bad, false).is_some(), "{bad:?}");
        }
    }

    // With a custom endpoint (an emulator) the naming rule is relaxed, but nothing that could
    // change the host or path of a request is accepted.
    #[test]
    fn account_names_with_a_custom_endpoint_still_refuse_redirecting_characters() {
        for ok in ["devstoreaccount1", "Dev_Store-1", "evil.example", "ab"] {
            assert_eq!(account_name_problem(ok, true), None, "{ok}");
        }
        for bad in [
            "", "a/b", "a?b", "a#b", "a@b", "a\\b", "a b", "a\nb", "..", ".", "%2e", "%2e%2e",
            "a%2fb",
        ] {
            assert!(account_name_problem(bad, true).is_some(), "{bad:?}");
        }
    }

    #[test]
    fn host_name_accounts_are_lowercased_and_validated_in_one_place() {
        let _env = clean_azure_env();
        assert_eq!(canonical_account_name("MyAccount1").unwrap(), "myaccount1");
        assert_eq!(
            resolve_azure_account_url("MyAccount1").unwrap(),
            "https://myaccount1.blob.core.windows.net"
        );
        for bad in ["attacker.example#", "a?b", "acct@evil", "x%2fy", "ab", ""] {
            assert!(canonical_account_name(bad).is_err(), "{bad:?}");
            assert!(resolve_azure_account_url(bad).is_err(), "{bad:?}");
            assert!(AzureBlob::account_url_from_account(bad).is_err(), "{bad:?}");
        }
        // With a custom endpoint the name is a path segment and keeps its case.
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://127.0.0.1:1",
        );
        assert_eq!(canonical_account_name("DevStore").unwrap(), "DevStore");
        assert!(canonical_account_name("a%2fb").is_err());
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
    }

    #[test]
    fn urls_are_logged_without_their_userinfo_query_or_fragment() {
        assert_eq!(
            redact_url("https://user:pw@h:1/a/b?sig=S#f"),
            "https://h:1/a/b"
        );
        assert_eq!(redact_url("http://tok@h"), "http://h");
        assert_eq!(redact_url("https://h/p@q"), "https://h/p@q");
    }

    #[test]
    fn urls_are_logged_without_their_query_or_fragment() {
        assert_eq!(
            redact_url("http://h:1/acct?sv=1&sig=SECRET#frag"),
            "http://h:1/acct"
        );
        assert_eq!(
            redact_url("https://acct.blob.core.windows.net"),
            "https://acct.blob.core.windows.net"
        );
        assert_eq!(redact_url("not a url?sig=SECRET"), "not a url");
    }

    #[test]
    fn only_localhost_and_loopback_addresses_are_local() {
        for local in [
            "localhost",
            "LOCALHOST",
            "127.0.0.1",
            "127.9.9.9",
            "::1",
            "[::1]",
        ] {
            assert!(is_loopback_host(local), "{local}");
        }
        for remote in [
            "203.0.113.9",
            "10.0.0.1",
            "example.com",
            "0.0.0.0",
            "localhost.evil.com",
        ] {
            assert!(!is_loopback_host(remote), "{remote}");
        }
    }

    #[test]
    fn range_reads_reserve_what_the_response_says_else_the_capped_request() {
        let cap = crate::constants::AZURE_RANGE_PRESIZE_MAX;
        assert_eq!(range_expected_len(Some(4096), Some(1 << 40)), Some(4096));
        assert_eq!(range_expected_len(None, Some(1 << 40)), Some(cap));
        assert_eq!(range_expected_len(None, Some(10)), Some(10));
        assert_eq!(range_expected_len(None, None), None);
    }

    #[test]
    fn a_range_reply_length_comes_from_a_consistent_content_range() {
        // A window in the middle, with and without a Content-Length.
        assert_eq!(
            range_response_body_len(Some("bytes 100-199/1000"), Some(100), 100).unwrap(),
            100
        );
        assert_eq!(
            range_response_body_len(Some("bytes 100-199/*"), None, 100).unwrap(),
            100
        );
        // The range must start where the window does.
        assert!(range_response_body_len(Some("bytes 4-7/100"), None, 0).is_err());
        // A Content-Length that disagrees is refused.
        assert!(range_response_body_len(Some("bytes 0-9/100"), Some(11), 0).is_err());
        // No Content-Range: a whole-blob reply to a range at 0 is read by its Content-Length ...
        assert_eq!(range_response_body_len(None, Some(77), 0).unwrap(), 77);
        assert!(range_response_body_len(None, None, 0).is_err());
        // ... but never for a range that starts later.
        assert!(range_response_body_len(None, Some(77), 5).is_err());
        // `bytes */n` carries no range.
        assert!(range_response_body_len(Some("bytes */100"), None, 0).is_err());
    }

    /// The length of a range is computed with checked arithmetic: a header the server controls
    /// must never overflow `u64` (a panic with overflow checks, a wrapped length without them).
    #[test]
    fn a_hostile_content_range_never_overflows_the_length() {
        assert!(range_response_body_len(Some("bytes 0-18446744073709551615/*"), None, 0).is_err());
        assert!(range_response_body_len(
            Some("bytes 18446744073709551614-18446744073709551615/*"),
            None,
            u64::MAX - 1
        )
        .is_ok());
        assert!(range_response_body_len(Some("bytes 5-4/*"), None, 5).is_err());
    }

    // Issue #242 review: list_account_containers handed a bearer-token credential to every
    // client, so an http:// endpoint (Azurite) was refused by the SDK with "must use https".
    // It now follows with_default_credential_from_url: no credential for http.
    #[tokio::test]
    async fn list_account_containers_on_an_http_endpoint_sends_no_credential() {
        reset_azure_env();
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
        let url = AzureBlob::account_url_from_account("mystorageaccount").unwrap();
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

        let url = resolve_azure_account_url("mystorageaccount").unwrap();
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

        let url = resolve_azure_account_url("devstoreaccount1").unwrap();
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

        let url = resolve_azure_account_url("testaccount").unwrap();
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

        let url = resolve_azure_account_url("devstoreaccount1").unwrap();
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

        let url = resolve_azure_account_url("testaccount").unwrap();
        assert_eq!(url, "http://primary:10000/testaccount");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);
    }
}

/// Typed classification of Azure failures (plan §4.5 A2), built from the SDK's typed
/// `ErrorKind`; no test relies on message text.
#[cfg(test)]
mod object_error_tests {
    use super::*;
    use crate::object_error::object_error;
    use crate::retry::RetryClassify;
    use azure_core::error::ErrorKind;
    use azure_core::http::StatusCode;

    fn http_error(status: u16, code: Option<&str>) -> anyhow::Error {
        let sdk = azure_core::Error::with_message(
            ErrorKind::HttpResponse {
                status: StatusCode::from(status),
                error_code: code.map(str::to_string),
                raw_response: None,
            },
            "azure",
        );
        request_error_for_test(sdk)
    }

    fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
        object_error(err).expect("classified").kind
    }

    #[test]
    fn error_code_comes_from_the_http_response_kind() {
        let e = http_error(404, Some("BlobNotFound"));
        let az = e.downcast_ref::<AzureRequestError>().expect("typed");
        assert_eq!(az.error_code(), Some("BlobNotFound"));
        let e = http_error(500, None);
        let az = e.downcast_ref::<AzureRequestError>().expect("typed");
        assert_eq!(az.error_code(), None);
        let other = request_error_for_test(azure_core::Error::with_message(ErrorKind::Io, "reset"));
        let az = other.downcast_ref::<AzureRequestError>().expect("typed");
        assert_eq!(az.error_code(), None);
    }

    #[test]
    fn mapping_follows_the_plan() {
        use ObjectErrorKind::*;
        let k = |s, c| kind_of(&attach_azure_classification(http_error(s, c), false));
        assert_eq!(k(404, Some("BlobNotFound")), NotFound);
        assert_eq!(k(404, Some("ContainerNotFound")), Other);
        assert_eq!(k(404, None), Other);
        assert_eq!(k(412, Some("ConditionNotMet")), VersionChanged);
        assert_eq!(k(403, Some("AuthorizationFailure")), Forbidden);
        assert_eq!(k(429, None), Throttled);
        assert_eq!(k(503, Some("ServerBusy")), Throttled);
        assert_eq!(k(500, Some("InternalError")), Server);
        assert_eq!(k(502, None), Server);
        assert_eq!(k(400, Some("InvalidInput")), Other);
        // Transport: no HTTP response at all.
        for kind in [ErrorKind::Io, ErrorKind::Connection] {
            let e = request_error_for_test(azure_core::Error::with_message(kind, "reset"));
            assert_eq!(kind_of(&attach_azure_classification(e, false)), Transport);
        }
        // A credential or data-conversion failure is not transport.
        for kind in [
            ErrorKind::Credential,
            ErrorKind::DataConversion,
            ErrorKind::Other,
        ] {
            let e = request_error_for_test(azure_core::Error::with_message(kind, "x"));
            assert_eq!(kind_of(&attach_azure_classification(e, false)), Other);
        }
    }

    /// A pinned read never reports absence, and a missing container is not absence.
    #[test]
    fn pinned_blob_not_found_is_a_version_change() {
        let first = attach_azure_classification(http_error(404, Some("BlobNotFound")), false);
        assert_eq!(kind_of(&first), ObjectErrorKind::NotFound);
        let pinned = attach_azure_classification(http_error(404, Some("BlobNotFound")), true);
        assert_eq!(kind_of(&pinned), ObjectErrorKind::VersionChanged);
        assert!(!pinned.is_retryable());
        assert!(!crate::delete_contract::is_not_found(&pinned));
        let container =
            attach_azure_classification(http_error(404, Some("ContainerNotFound")), false);
        assert!(!crate::delete_contract::is_not_found(&container));
    }

    /// An error that already carries an outer `NotFound` is upgraded to `VersionChanged` on a
    /// pinned read, and left alone otherwise.
    #[test]
    fn pinned_attach_upgrades_an_existing_not_found() {
        let first = attach_azure_classification(http_error(404, Some("BlobNotFound")), false);
        assert_eq!(kind_of(&first), ObjectErrorKind::NotFound);
        let upgraded = attach_azure_classification(first, true);
        assert_eq!(kind_of(&upgraded), ObjectErrorKind::VersionChanged);
        assert!(!upgraded.is_retryable());
        assert!(!crate::delete_contract::is_not_found(&upgraded));
        // Not pinned: unchanged.
        let first = attach_azure_classification(http_error(404, Some("BlobNotFound")), false);
        let same = attach_azure_classification(first, false);
        assert_eq!(kind_of(&same), ObjectErrorKind::NotFound);
        // Pinned, but another kind: unchanged.
        let forbidden = attach_azure_classification(http_error(403, None), false);
        let same = attach_azure_classification(forbidden, true);
        assert_eq!(kind_of(&same), ObjectErrorKind::Forbidden);
    }

    /// A server-supplied code with control characters, or a very long one, never reaches the
    /// attached message intact.
    #[test]
    fn hostile_codes_are_sanitised_in_the_attached_message() {
        let e = attach_azure_classification(http_error(400, Some("Bad\r\nCode\x07 x")), false);
        let text = object_error(&e).expect("typed").to_string();
        assert!(text.contains("status 400, code BadCodex"), "{text:?}");
        assert!(!text.contains('\n') && !text.contains('\x07'), "{text:?}");
        let long = "Z".repeat(500);
        let e = attach_azure_classification(http_error(400, Some(&long)), false);
        let text = object_error(&e).expect("typed").to_string();
        assert!(text.len() < 150, "{} bytes", text.len());
        assert!(text.contains(&"Z".repeat(64)) && !text.contains(&"Z".repeat(65)));
    }

    /// The attached message names status and code only, and the redacted source text is
    /// unchanged below it.
    #[test]
    fn classification_keeps_the_typed_source_and_adds_no_source_text() {
        let e = attach_azure_classification(http_error(503, Some("ServerBusy")), false);
        let text = object_error(&e).expect("typed").to_string();
        assert!(
            text.contains("status 503") && text.contains("ServerBusy"),
            "{text}"
        );
        assert!(e.downcast_ref::<AzureRequestError>().is_some());
        // Not an Azure error: returned unchanged.
        let plain = attach_azure_classification(anyhow::anyhow!("x"), false);
        assert!(object_error(&plain).is_none());
    }
}

/// The caller-buffer GET's pure pieces (plan v0.10.0 §4.3, §4.5 A8; #264): the `Content-Range`
/// parser, the length plan, the SDK options, the retry rule and the checked receive loop. The
/// requests themselves are covered against a local fake service in
/// `tests/azure_get_into_fake.rs`.
#[cfg(test)]
mod get_into_tests {
    use super::*;
    use crate::object_error::object_error;
    use crate::retry::RetryClassify;
    use azure_core::error::ErrorKind;

    const CANARY: u8 = 0xEE;

    fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
        object_error(err).expect("typed error").kind
    }

    fn body(parts: &[&[u8]]) -> impl Stream<Item = azure_core::Result<Bytes>> + Unpin {
        let items: Vec<azure_core::Result<Bytes>> = parts
            .iter()
            .map(|p| Ok(Bytes::copy_from_slice(p)))
            .collect();
        futures::stream::iter(items)
    }

    // ---- Content-Range -------------------------------------------------------------------

    #[test]
    fn content_range_valid_forms() {
        let ok = |text: &str| parse_content_range(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(
            ok("bytes 0-9/100"),
            ContentRangeValue {
                bytes: Some((0, 9)),
                total: Some(100)
            }
        );
        assert_eq!(
            ok("bytes 0-0/1"),
            ContentRangeValue {
                bytes: Some((0, 0)),
                total: Some(1)
            }
        );
        // The total may be unknown, or the range may be (the 416 form).
        assert_eq!(ok("bytes 5-9/*").total, None);
        assert_eq!(ok("bytes 5-9/*").bytes, Some((5, 9)));
        assert_eq!(ok("bytes */100").bytes, None);
        assert_eq!(ok("bytes */100").total, Some(100));
        // Surrounding whitespace is tolerated; u64 extremes parse exactly (no usize, no cast).
        assert_eq!(ok("  bytes 0-9/100 ").total, Some(100));
        let max = u64::MAX;
        let big = ok(&format!("bytes 0-{}/{max}", max - 1));
        assert_eq!(big.bytes, Some((0, max - 1)));
        assert_eq!(big.total, Some(max));
    }

    #[test]
    fn content_range_malformed_is_a_typed_error() {
        let max_plus_one = "18446744073709551616";
        let bad = [
            "",
            "bytes",
            "bytes ",
            "bytes 0-9",
            "bytes 0-9/",
            "bytes -9/100",
            "bytes 0-/100",
            "bytes 9-0/100",     // reversed
            "bytes 0-100/100",   // last byte at or past the total
            "bytes 0-9/5",       // last byte past the total
            "bytes 0-99/100x",   // trailing junk
            "bytes +0-9/100",    // a sign is not a digit
            "bytes 0-9/-100",    // negative
            "bytes 0-9/1e2",     // not decimal
            "items 0-9/100",     // wrong unit
            "bytes */*",         // nothing known
            "bytes 0-9/100/200", // extra slash
            "bytes 0 - 9/100",   // inner whitespace
            "bytes 0-9 /100",
            "bytes \u{660}-\u{669}/100", // non-ASCII digits
            "bytes 0-9/0x64",
        ];
        for text in bad {
            let err = parse_content_range(text).expect_err(text);
            assert_eq!(kind_of(&err), ObjectErrorKind::Other, "{text:?}");
        }
        // Overflow of u64 is malformed, not a wrap.
        for text in [
            format!("bytes 0-9/{max_plus_one}"),
            format!("bytes {max_plus_one}-{max_plus_one}/*"),
        ] {
            assert!(parse_content_range(&text).is_err(), "{text}");
        }
    }

    #[test]
    fn content_range_error_text_is_bounded() {
        let hostile = format!("bytes {}\r\nX-Injected: 1", "9".repeat(5000));
        let text = parse_content_range(&hostile).expect_err("bad").to_string();
        assert!(text.len() < 200, "{} bytes", text.len());
        assert!(!text.contains('\n'), "{text:?}");
    }

    // ---- 200 vs 206 lengths ----------------------------------------------------------------

    #[test]
    fn response_lengths_for_206_and_200() {
        let lens = |range: Option<&str>, length: Option<u64>, request: u64| {
            response_lengths(range, length, request)
        };
        let l = |total, body_len| ResponseLengths { total, body_len };
        // 206, blob exactly as long as asked.
        assert_eq!(
            lens(Some("bytes 0-99/100"), Some(100), 100).unwrap(),
            l(100, 100)
        );
        // 206, blob longer than asked: the body is the asked range, the total is the truth.
        assert_eq!(
            lens(Some("bytes 0-99/250"), Some(100), 100).unwrap(),
            l(250, 100)
        );
        // 206, blob shorter than asked.
        assert_eq!(
            lens(Some("bytes 0-39/40"), Some(40), 100).unwrap(),
            l(40, 40)
        );
        // Content-Length may be absent beside a Content-Range.
        assert_eq!(
            lens(Some("bytes 0-99/100"), None, 100).unwrap(),
            l(100, 100)
        );
        // 200: the service ignored the range; Content-Length is the blob.
        assert_eq!(lens(None, Some(100), 100).unwrap(), l(100, 100));
        assert_eq!(lens(None, Some(250), 100).unwrap(), l(250, 250));
        // The empty blob after the 416 retry: 200, length 0.
        assert_eq!(lens(None, Some(0), 1).unwrap(), l(0, 0));
        // The zero-length probe (bytes 0..1) against a non-empty blob: 1 byte of 7.
        assert_eq!(lens(Some("bytes 0-0/7"), Some(1), 1).unwrap(), l(7, 1));
    }

    /// Review C-F1: the SDK hides the status, so a 206 stripped of `Content-Range` reads as a
    /// whole blob of the length it carries. The request is `N + 1` bytes, so a blob larger than
    /// `N = 60` carries 61 bytes and reports a total of 61, never 60: the caller's check of the
    /// total against `N` refuses it. A blob of exactly `N` bytes carries 60.
    #[test]
    fn a_206_without_content_range_for_a_larger_blob_never_reports_the_expected_total() {
        let expected = 60;
        let request = checked_request_len(expected).unwrap();
        assert_eq!(request, 61);
        // A 100-byte blob, range honoured, header stripped: 61 bytes arrive.
        let larger = response_lengths(None, Some(61), request).unwrap();
        assert_eq!(
            larger,
            ResponseLengths {
                total: 61,
                body_len: 61
            }
        );
        assert_ne!(larger.total, expected);
        // A 60-byte blob: the reply is the whole blob.
        let exact = response_lengths(None, Some(60), request).unwrap();
        assert_eq!(exact.total, expected);
    }

    #[test]
    fn response_lengths_refuses_what_it_cannot_check() {
        let refused = |range: Option<&str>, length: Option<u64>, request: u64| {
            let err = response_lengths(range, length, request).expect_err("must refuse");
            assert_eq!(kind_of(&err), ObjectErrorKind::Other);
        };
        refused(None, None, 100); // no size at all: not 0
        refused(Some("bytes 0-99/*"), Some(100), 100); // unknown total
        refused(Some("bytes */100"), None, 100); // no range
        refused(Some("bytes 10-19/100"), Some(10), 100); // does not start at 0
        refused(Some("bytes 0-9/100"), Some(10), 100); // shorter than asked of a longer blob
        refused(Some("bytes 0-99/100"), Some(50), 100); // Content-Length disagrees
        refused(Some("garbage"), Some(100), 100);
    }

    // ---- SDK options: the no-remaining-ranges shape -----------------------------------------

    #[test]
    fn options_ask_one_bounded_range_with_partition_equal_to_it() {
        let mut sizes = vec![1u64, 7, 4 * 1024 * 1024, 4 * 1024 * 1024 + 1];
        // 8 GiB fits a usize on a 64-bit platform only: there it must work, and on a 32-bit one
        // it must be refused rather than truncated.
        let huge = 1u64 << 33;
        if usize::try_from(huge).is_ok() {
            sizes.push(huge);
        } else {
            assert!(get_into_options(huge, None).is_err());
        }
        for expected in sizes {
            let o = get_into_options(expected, None).expect("options");
            // One byte more than the destination (review C-F1).
            assert_eq!(o.range, Some(HttpRange::new(0, expected + 1)));
            assert_eq!(o.parallel, NonZeroUsize::new(1));
            // partition_size == the range length is what leaves the SDK no remaining ranges.
            assert_eq!(
                o.partition_size.and_then(|p| u64::try_from(p.get()).ok()),
                Some(expected + 1),
                "{expected}"
            );
            assert!(o.if_match.is_none());
        }
        let o = get_into_options(1000, None).expect("options");
        assert_eq!(o.range.expect("range").to_string(), "bytes=0-1000");
    }

    /// `N + 1` overflowing is refused before a request exists, with a typed error.
    #[test]
    fn options_refuse_a_length_whose_successor_overflows() {
        let err = get_into_options(u64::MAX, None).expect_err("N + 1 overflows");
        assert_eq!(kind_of(&err), ObjectErrorKind::Other);
        assert!(checked_request_len(u64::MAX).is_err());
        assert_eq!(checked_request_len(u64::MAX - 1).unwrap(), u64::MAX);
        assert_eq!(checked_request_len(0).unwrap(), 1);
    }

    #[test]
    fn options_for_a_zero_length_destination_probe_one_byte() {
        let o = get_into_options(0, None).expect("options");
        assert_eq!(o.range.expect("range").to_string(), "bytes=0-0");
        assert_eq!(o.partition_size, NonZeroUsize::new(1));
        assert_eq!(o.parallel, NonZeroUsize::new(1));
    }

    #[test]
    fn options_carry_if_match_on_a_retry() {
        let o = get_into_options(10, Some("\"0x8DC\"")).expect("options");
        assert_eq!(
            o.if_match.as_ref().map(AsRef::<str>::as_ref),
            Some("\"0x8DC\"")
        );
    }

    // ---- ETag pinning and the retry rule -----------------------------------------------------

    #[test]
    fn only_a_strong_nonempty_etag_pins() {
        let etag = |s: &str| usable_etag(Some(&azure_core::http::Etag::from(s)));
        assert_eq!(usable_etag(None), None);
        assert_eq!(etag(""), None);
        assert_eq!(etag("   "), None);
        assert_eq!(etag("W/\"weak\""), None, "If-Match compares strongly");
        assert_eq!(etag("\"0x8DC\""), Some("\"0x8DC\"".to_string()));
        assert_eq!(etag("  \"0x8DC\" "), Some("\"0x8DC\"".to_string()));
    }

    #[test]
    fn retry_needs_a_transient_typed_error_and_a_pinned_version_or_no_bytes() {
        use crate::object_error::attach_object_error;
        let typed = |kind| attach_object_error(anyhow!("x"), kind, "az", "m");
        let transient = [
            ObjectErrorKind::Throttled,
            ObjectErrorKind::Transport,
            ObjectErrorKind::Server,
        ];
        let definite = [
            ObjectErrorKind::NotFound,
            ObjectErrorKind::VersionChanged,
            ObjectErrorKind::Forbidden,
            ObjectErrorKind::Unsupported,
            ObjectErrorKind::Other,
            ObjectErrorKind::LengthMismatch {
                expected: 1,
                actual: Some(2),
            },
        ];
        for kind in transient {
            // Pinned: retried whatever was written. Unpinned: only before the first byte.
            assert!(may_retry_get_into(&typed(kind), true, 0), "{kind:?}");
            assert!(may_retry_get_into(&typed(kind), true, 4096), "{kind:?}");
            assert!(may_retry_get_into(&typed(kind), false, 0), "{kind:?}");
            assert!(
                !may_retry_get_into(&typed(kind), false, 1),
                "unpinned pieces must never be combined: {kind:?}"
            );
        }
        for kind in definite {
            for (pinned, written) in [(true, 0), (true, 9), (false, 0), (false, 9)] {
                assert!(
                    !may_retry_get_into(&typed(kind), pinned, written),
                    "{kind:?}"
                );
            }
        }
        // An unclassified error is never retried here (the generic fallback does not apply).
        assert!(!may_retry_get_into(
            &anyhow!("transient-looking: 503"),
            true,
            0
        ));
    }

    // ---- the checked receive loop ------------------------------------------------------------

    /// A destination inside a buffer with canaries before and after it.
    fn with_canaries(len: usize) -> Vec<u8> {
        vec![CANARY; len + 16]
    }

    #[tokio::test]
    async fn receive_exact_in_chunks_counts_each_position_once() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(10);
        let dst = &mut buf[8..18];
        receive_into(
            body(&[b"abc", b"", b"defg", b"hij"]),
            dst,
            &mut span,
            CredentialMode::None,
        )
        .await
        .expect("exact body");
        assert_eq!(&buf[8..18], b"abcdefghij");
        assert!(buf[..8].iter().chain(&buf[18..]).all(|&b| b == CANARY));
        assert_eq!(progress.filled(), 10);
        assert!(progress.first_byte().is_some());
    }

    #[tokio::test]
    async fn receive_short_body_is_a_transport_error_with_progress_kept() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(8);
        let err = receive_into(
            body(&[b"abc", b"de"]),
            &mut buf[8..16],
            &mut span,
            CredentialMode::None,
        )
        .await
        .expect_err("5 of 8 bytes");
        assert_eq!(kind_of(&err), ObjectErrorKind::Transport);
        assert_eq!(progress.filled(), 5, "the five bytes that arrived");
        assert_eq!(&buf[8..13], b"abcde");
        assert!(buf[13..].iter().all(|&b| b == CANARY), "the rest untouched");
        assert!(buf[..8].iter().all(|&b| b == CANARY));
    }

    #[tokio::test]
    async fn receive_overflow_is_a_length_mismatch_and_never_writes_past_dst() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(6);
        // 4 + 4 bytes into 6: the second chunk would cross the end.
        let err = receive_into(
            body(&[b"abcd", b"efgh"]),
            &mut buf[8..14],
            &mut span,
            CredentialMode::None,
        )
        .await
        .expect_err("overflow");
        assert_eq!(
            kind_of(&err),
            ObjectErrorKind::LengthMismatch {
                expected: 6,
                actual: None
            }
        );
        assert!(!err.is_retryable());
        assert_eq!(
            &buf[8..12],
            b"abcd",
            "the chunk before the overflow is kept"
        );
        assert!(
            buf[12..].iter().all(|&b| b == CANARY),
            "nothing past dst, no partial chunk"
        );
        assert!(buf[..8].iter().all(|&b| b == CANARY));
        assert_eq!(progress.filled(), 4);
    }

    #[tokio::test]
    async fn receive_one_oversized_chunk_writes_nothing() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(3);
        let err = receive_into(
            body(&[b"abcdefgh"]),
            &mut buf[8..11],
            &mut span,
            CredentialMode::None,
        )
        .await
        .expect_err("overflow");
        assert!(matches!(
            kind_of(&err),
            ObjectErrorKind::LengthMismatch { expected: 3, .. }
        ));
        assert!(buf.iter().all(|&b| b == CANARY));
        assert_eq!(progress.filled(), 0);
        assert!(progress.first_byte().is_none());
    }

    #[tokio::test]
    async fn receive_into_an_empty_destination() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        // Only empty chunks (or none): success, nothing counted.
        receive_into(body(&[b"", b""]), &mut [], &mut span, CredentialMode::None)
            .await
            .expect("empty body");
        receive_into(body(&[]), &mut [], &mut span, CredentialMode::None)
            .await
            .expect("no body");
        assert_eq!(progress.filled(), 0);
        // Any byte is an overflow; there is nowhere to put it.
        let err = receive_into(body(&[b"x"]), &mut [], &mut span, CredentialMode::None)
            .await
            .expect_err("one byte into zero");
        assert!(matches!(
            kind_of(&err),
            ObjectErrorKind::LengthMismatch { expected: 0, .. }
        ));
    }

    #[tokio::test]
    async fn receive_mid_stream_error_keeps_progress_and_redacts_the_sas() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(8);
        let failing = futures::stream::iter(vec![
            Ok(Bytes::from_static(b"abcd")),
            Err(azure_core::Error::with_message(
                ErrorKind::Io,
                "connection reset reading https://acct.blob.core.windows.net/c/b?sv=2024-11-04&sig=TOPSECRET%2Fvalue",
            )),
            Ok(Bytes::from_static(b"never")),
        ]);
        let err = receive_into(failing, &mut buf[8..16], &mut span, CredentialMode::Sas)
            .await
            .expect_err("stream error");
        // The count the collect_into helper would have lost.
        assert_eq!(progress.filled(), 4);
        assert_eq!(&buf[8..12], b"abcd");
        assert!(buf[12..].iter().all(|&b| b == CANARY));
        let typed = attach_azure_classification(err, false);
        assert_eq!(kind_of(&typed), ObjectErrorKind::Transport);
        for text in [format!("{typed:#}"), format!("{typed:?}")] {
            assert!(!text.contains("TOPSECRET"), "SAS leaked: {text}");
        }
    }

    #[tokio::test]
    async fn a_retry_restarts_at_zero_and_counts_each_position_once() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(8);
        let first = futures::stream::iter(vec![
            Ok(Bytes::from_static(b"abcde")),
            Err(azure_core::Error::with_message(ErrorKind::Io, "reset")),
        ]);
        receive_into(first, &mut buf[8..16], &mut span, CredentialMode::None)
            .await
            .expect_err("first attempt fails");
        assert_eq!(progress.filled(), 5);
        // The retry starts over at offset 0 and passes the old mark.
        receive_into(
            body(&[b"ab", b"cdefgh"]),
            &mut buf[8..16],
            &mut span,
            CredentialMode::None,
        )
        .await
        .expect("second attempt");
        assert_eq!(&buf[8..16], b"abcdefgh");
        assert_eq!(progress.filled(), 8, "5 + 3 new positions, not 5 + 8");
    }

    #[tokio::test]
    async fn cancellation_keeps_what_was_copied() {
        let progress = ReceiveProgress::new();
        let mut span = SpanMark::new(&progress);
        let mut buf = with_canaries(8);
        // Two chunks arrive, then the stream never yields again.
        let stalled = futures::stream::iter(vec![
            Ok(Bytes::from_static(b"ab")),
            Ok(Bytes::from_static(b"cd")),
        ])
        .chain(futures::stream::pending::<azure_core::Result<Bytes>>());
        let outcome = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            receive_into(stalled, &mut buf[8..16], &mut span, CredentialMode::None),
        )
        .await;
        assert!(outcome.is_err(), "the receive must still be pending");
        // The future was dropped by the timeout: the destination is ours again, and the count
        // says which positions were touched.
        assert_eq!(progress.filled(), 4);
        assert_eq!(&buf[8..12], b"abcd");
        assert!(buf[12..].iter().all(|&b| b == CANARY));
        assert!(progress.first_byte().is_some());
    }

    // ---- typed error shapes ------------------------------------------------------------------

    #[test]
    fn length_mismatch_carries_expected_and_actual_and_is_final() {
        let e = az_length_mismatch(100, Some(40), "the blob is 40 bytes");
        assert_eq!(
            kind_of(&e),
            ObjectErrorKind::LengthMismatch {
                expected: 100,
                actual: Some(40)
            }
        );
        assert!(!e.is_retryable());
        let e = az_length_mismatch(100, None, "the body is longer");
        assert!(matches!(
            kind_of(&e),
            ObjectErrorKind::LengthMismatch { actual: None, .. }
        ));
    }
}
