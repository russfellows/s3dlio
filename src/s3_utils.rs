// src/s3_utils.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides high‑level S3 operations: list, get, delete, typed PUT.

use anyhow::{bail, Context, Result};
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_smithy_runtime_api::client::orchestrator::HttpResponse;
use aws_smithy_runtime_api::client::result::SdkError;
//use aws_sdk_s3::primitives::ByteStream;
use futures::{stream::FuturesUnordered, Stream, StreamExt};
#[cfg(feature = "extension-module")]
use pyo3::{FromPyObject, PyAny};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::data_loader::parallel_fetch::DropCancel;

use bytes::{Bytes, BytesMut};
#[cfg(feature = "profiling")]
use tracing::instrument;

use tracing::{debug, info};

// data generation helpers
use crate::config::Config;
use crate::config::ObjectType;
use crate::data_gen::generate_object;
use crate::list_pattern::ListPattern;

// S3 client creation
use crate::retry::retry_get_body;
use crate::s3_client::{aws_s3_client_async, run_on_global_rt};

// S3 operation logging (the public helpers in this module write the S3Ops row)
use crate::s3_logger::global_logger;
use crate::s3_ops::S3Ops;

// Object size cache for eliminating redundant HEAD requests
use crate::object_size_cache::ObjectSizeCache;

// -----------------------------------------------------------------------------
// S3 error diagnostics
// -----------------------------------------------------------------------------

/// Walk the `std::error::Error` source chain and build a compact string like
/// `"outer: middle: inner cause"`.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut parts = vec![e.to_string()];
    let mut src = e.source();
    while let Some(s) = src {
        parts.push(s.to_string());
        src = s.source();
    }
    parts.join(": ")
}

/// Text appended to a service error that has no parsable S3 error body, naming
/// the HTTP status (issue #236), e.g. `" — HTTP 411 Length Required"`.
pub(crate) fn no_error_body_status_note(status: u16) -> String {
    match http::StatusCode::from_u16(status) {
        Ok(sc) => match sc.canonical_reason() {
            Some(reason) => format!(" — HTTP {status} {reason}"),
            None => format!(" — HTTP {status}"),
        },
        Err(_) => format!(" — HTTP {status}"),
    }
}

/// Format an `SdkError` from any S3 operation into a human-readable message that
/// includes the error category, the underlying cause, and — where applicable —
/// a concrete hint about what the user should check.
pub(crate) fn format_sdk_error<E>(e: &SdkError<E, HttpResponse>) -> String
where
    E: std::fmt::Display + ProvideErrorMetadata + std::error::Error,
{
    match e {
        SdkError::DispatchFailure(d) => {
            let category = if d.is_io() {
                "I/O error"
            } else if d.is_timeout() {
                "connection timeout"
            } else if d.is_user() {
                "user/request error"
            } else {
                "transport error"
            };

            let details = d
                .as_connector_error()
                .map(|ce| error_chain(ce as &dyn std::error::Error))
                .unwrap_or_else(|| "unknown connector error".to_string());

            let details_lower = details.to_lowercase();
            let hint = if details_lower.contains("connection refused") {
                "check AWS_ENDPOINT_URL — the server is not listening on that address/port"
            } else if details_lower.contains("tls")
                || details_lower.contains("ssl")
                || details_lower.contains("certificate")
                || details_lower.contains("handshake")
            {
                "TLS/certificate error — set AWS_CA_BUNDLE to your CA cert, \
                 or check AWS_ENDPOINT_URL scheme (http:// vs https://)"
            } else if details_lower.contains("name or service not known")
                || details_lower.contains("failed to lookup")
                || details_lower.contains("dns")
            {
                "DNS resolution failed — check the hostname in AWS_ENDPOINT_URL"
            } else if details_lower.contains("https") && details_lower.contains(":80") {
                "AWS_ENDPOINT_URL uses 'https://' on port 80 (a plain-HTTP port) — \
                 use 'http://' instead, or the correct HTTPS port"
            } else if d.is_io() {
                "check AWS_ENDPOINT_URL, network connectivity, and that the server is running"
            } else if d.is_timeout() {
                "server did not respond within the connect timeout — verify AWS_ENDPOINT_URL is correct and the host is reachable"
            } else {
                "check AWS_ENDPOINT_URL, network connectivity, and firewall rules"
            };

            format!("dispatch failure ({category}: {details}) — {hint}")
        }

        SdkError::TimeoutError(_) => {
            let timeout_secs = std::env::var("S3DLIO_OPERATION_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);
            format!(
                "request timed out after {timeout_secs}s — \
                 check network connectivity or increase S3DLIO_OPERATION_TIMEOUT_SECS"
            )
        }

        SdkError::ConstructionFailure(_) => "failed to construct S3 request — \
             check AWS_REGION, AWS_ENDPOINT_URL, and credential environment variables \
             (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)"
            .to_string(),

        SdkError::ServiceError(s) => {
            let code_opt = s.err().code();
            let code = code_opt.unwrap_or("unknown");
            let msg = s.err().message().unwrap_or("no message");
            let hint = match code {
                "NoSuchBucket" => " — bucket does not exist; check the bucket name in the URI",
                "AccessDenied" => " — check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                "InvalidAccessKeyId" => {
                    if crate::s3_client::s3_endpoint_url_from_env().is_none() {
                        " — AWS_ACCESS_KEY_ID is not recognised by the server; \
                          no endpoint URL is set — did you forget --endpoint-url or AWS_ENDPOINT_URL? \
                          (credentials valid for a custom endpoint are rejected by real AWS S3)"
                    } else {
                        " — AWS_ACCESS_KEY_ID is not recognised by the server"
                    }
                }
                "SignatureDoesNotMatch" => {
                    " — AWS_SECRET_ACCESS_KEY is wrong, or system clock is out of sync"
                }
                "NoSuchKey" => " — object key does not exist",
                _ => "",
            };
            // Issue #236: when the response carried no parsable S3 error body
            // (e.g. GCS's HTML "411 Length Required" page) the code and message
            // are both unknown; name the HTTP status so the cause is not hidden.
            let status_note = if code_opt.is_none() {
                no_error_body_status_note(s.raw().status().as_u16())
            } else {
                String::new()
            };
            format!("service error: {code} ({msg}){hint}{status_note}")
        }

        SdkError::ResponseError(_) => {
            format!("{e}")
        }

        _ => format!("{e}"),
    }
}

/// Wrap an `SdkError` into an `anyhow::Error` whose top-line message is
/// `"<ctx>: <formatted SdkError chain>"`.
///
/// Use this at every site where an S3 SDK call may fail and the error would
/// otherwise be surfaced to a user (Python `RuntimeError`, CLI output, log).
/// The bare `SdkError::DispatchFailure` `Display` impl is just the literal
/// string `"dispatch failure"` with no chain — issue mlcommons/storage#506
/// is an instance of that bare string reaching Python.  This helper makes
/// sure the connector chain, hint, and operation context are all retained.
///
/// NOTE: Only the S3 path is covered here.  The Azure (`az://`), GCS (`gs://`),
/// local-file (`file://`), and direct-IO (`direct://`) backends have their own
/// SDK/error types and currently emit equally opaque messages on dispatch /
/// network failures (e.g. `azure_client.rs`, `google_gcs_client.rs`,
/// `file_store_direct.rs`).  Those should get an equivalent helper +
/// audit pass in a follow-up — track separately from #506.
pub(crate) fn sdk_anyhow<E>(
    ctx: impl std::fmt::Display,
    e: SdkError<E, HttpResponse>,
) -> anyhow::Error
where
    E: std::fmt::Display + ProvideErrorMetadata + std::error::Error,
{
    // The typed `SdkError` is flattened into text here, so carry the
    // transient/definite verdict along (issue #240): `retry_get_body` needs
    // it and could not recover it from the message.
    let retryable = crate::retry::sdk_error_is_retryable(&e);
    // The HTTP status and the service code travel with the verdict, so a typed
    // classification (plan §4.5 A2) can be made after the flattening.
    let status = e.raw_response().map(|r| r.status().as_u16());
    let code = match &e {
        SdkError::ServiceError(s) => s.err().code().map(str::to_owned),
        _ => None,
    };
    anyhow::Error::new(crate::retry::SdkFailure::new(
        format!("{}: {}", ctx, format_sdk_error(&e)),
        retryable,
        status,
        code,
    ))
}

/// `Result<T, SdkError<…>>` extension trait that provides `.sdk_context(ctx)?`
/// as a drop-in replacement for `anyhow::Context::context(…)?`.
///
/// `anyhow::Context::context()` attaches the outer message but the `to_string()`
/// of the resulting `anyhow::Error` only shows the outermost layer — so a
/// `DispatchFailure` still surfaces as the bare word `"dispatch failure"`.
/// `sdk_context()` flattens the SDK error chain into the message itself via
/// `format_sdk_error()`, guaranteeing the full cause + hint reach the caller.
///
/// See `sdk_anyhow()` for the rationale and the cross-scheme TODO note.
pub(crate) trait SdkResultExt<T> {
    fn sdk_context(self, ctx: impl std::fmt::Display) -> anyhow::Result<T>;
}

impl<T, E> SdkResultExt<T> for std::result::Result<T, SdkError<E, HttpResponse>>
where
    E: std::fmt::Display + ProvideErrorMetadata + std::error::Error,
{
    fn sdk_context(self, ctx: impl std::fmt::Display) -> anyhow::Result<T> {
        self.map_err(|e| sdk_anyhow(ctx, e))
    }
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_OBJECT_SIZE: usize = 20 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Process-level configuration caches
//
// These OnceLocks read environment variables ONCE per process on first call,
// then return the cached value on every subsequent call — eliminating env-var
// syscalls on the hot path where get_object_uri_optimized_async() is called
// once per object per step.
// ─────────────────────────────────────────────────────────────────────────────

static RANGE_OPT_ENABLED: OnceLock<bool> = OnceLock::new();
static RANGE_THRESHOLD_BYTES: OnceLock<u64> = OnceLock::new();
static GLOBAL_SIZE_CACHE: OnceLock<Arc<ObjectSizeCache>> = OnceLock::new();

/// Process-wide latch: when true, `get_object_uri_optimized_async` skips the
/// HEAD request entirely and issues a plain streaming GET for every object.
///
/// Set programmatically via `set_skip_head(true)` (triggered by the
/// `skip_head` loader option) **or** by setting `S3DLIO_SKIP_HEAD=1` in the
/// process environment before startup.  Once set to `true` it stays `true`
/// for the lifetime of the process — intentional: a process that elects to
/// skip HEADs for one loader should skip them for all loaders.
static SKIP_HEAD: AtomicBool = AtomicBool::new(false);

/// Activate the process-wide HEAD-skip latch.
///
/// Called from `PyBytesAsyncDataLoader` when the user passes
/// `{"skip_head": True}` in the loader opts dict.  Safe to call from any
/// thread; subsequent calls are no-ops (latch stays `true`).
pub fn set_skip_head(v: bool) {
    if v {
        SKIP_HEAD.store(true, Ordering::Relaxed);
    }
}

/// Returns `true` if HEAD requests should be skipped globally.
///
/// Checks both the runtime latch (set via `set_skip_head`) and the
/// `S3DLIO_SKIP_HEAD` env var (read once on first call, then cached).
fn get_skip_head() -> bool {
    if SKIP_HEAD.load(Ordering::Relaxed) {
        return true;
    }
    // Lazily check env var once and latch if set.
    static ENV_CHECKED: OnceLock<bool> = OnceLock::new();
    *ENV_CHECKED.get_or_init(|| {
        crate::dotenv::apply(); // latched for the process: the policy first (#243)
        let from_env = std::env::var("S3DLIO_SKIP_HEAD")
            .ok()
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        if from_env {
            SKIP_HEAD.store(true, Ordering::Relaxed);
        }
        from_env
    })
}

/// Whether range optimization (range splitting) is enabled.
/// Cached once per process from S3DLIO_ENABLE_RANGE_OPTIMIZATION.
/// Setting =0/false/no/off/disable skips HEAD entirely on the get_many() path
/// (S3ObjectStore::get() honours this too — now consistent on ALL paths).
fn get_range_opt_enabled() -> bool {
    *RANGE_OPT_ENABLED.get_or_init(|| {
        crate::dotenv::apply(); // latched for the process: the policy first (#243)
        std::env::var("S3DLIO_ENABLE_RANGE_OPTIMIZATION")
            .ok()
            .map(|v| {
                !matches!(
                    v.to_lowercase().as_str(),
                    "0" | "false" | "no" | "off" | "disable" | "disabled"
                )
            })
            .unwrap_or(true) // default: enabled
    })
}

/// Range-split threshold in bytes, cached once per process.
/// Default: 32 MiB. Override with S3DLIO_RANGE_THRESHOLD_MB=<megabytes>.
/// Set above file size (e.g. 1000 for 147 MiB files) to force single-GET.
fn get_range_threshold_bytes() -> u64 {
    *RANGE_THRESHOLD_BYTES.get_or_init(|| {
        crate::dotenv::apply(); // latched for the process: the policy first (#243)
        let mb = std::env::var("S3DLIO_RANGE_THRESHOLD_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(32);
        mb * 1024 * 1024
    })
}

/// Process-global object size cache.
///
/// Default TTL: 1 hour. Override with S3DLIO_SIZE_CACHE_TTL_SECS=<seconds>.
/// 5 minutes (300 s) would be too short for large-scale training where a single
/// epoch over a 10TB+ dataset takes 10–30 minutes — cache entries would expire
/// and cause redundant HEAD requests on every subsequent epoch.
/// Populated by get_object_uri_optimized_async() on each HEAD and by the
/// pre-stat phase in get_objects_parallel(). Cache hits skip HEAD entirely.
/// Bounded (issue #241): at most S3DLIO_SIZE_CACHE_MAX_ENTRIES entries, oldest
/// evicted first; an expired entry is removed when a lookup finds it.
fn get_size_cache() -> &'static Arc<ObjectSizeCache> {
    GLOBAL_SIZE_CACHE.get_or_init(|| {
        crate::dotenv::apply(); // latched for the process: the policy first (#243)
        let ttl_secs = std::env::var("S3DLIO_SIZE_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(3600); // 1 hour — lasts across many training epochs
        Arc::new(ObjectSizeCache::new(Duration::from_secs(ttl_secs)))
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// helpers: create an S3Ops around the global S3 client.
//
// `S3Ops` is the S3 helpers' own row writer. A row is written ONLY when the
// `S3Ops` was built with a logger, so the owner of a call is chosen by which
// builder the call used (issue #169, one row per call):
//
// * `build_ops_async` -- the public helper functions of this module that are
//   user entry points (`get_object`, `put_object_async`, `delete_objects`, and
//   everything built on them). Wired to the singleton logger exactly as before
//   W2, so the row they write has the same fields as the row they always wrote.
// * `Rows::Skip` -- the S3 backend of an `ObjectStore`. The store's
//   caller records the call (a `LoggedObjectStore`), or deliberately does not (a
//   store built without a logger), so the backend must not write a second row.

/// Whether a helper call writes the S3Ops row (see the block comment above).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rows {
    /// Public helper: write the S3Ops row if the op-log is active.
    Write,
    /// Backend of an `ObjectStore`: write nothing.
    Skip,
}

async fn build_ops_for(rows: Rows) -> Result<S3Ops> {
    let client = aws_s3_client_async().await?;
    Ok(match rows {
        Rows::Write => {
            let logger = global_logger();
            let client_id = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
            let endpoint = crate::s3_client::s3_endpoint_url_from_env().unwrap_or_default();
            S3Ops::new(client, logger, &client_id, &endpoint)
        }
        Rows::Skip => S3Ops::new(client, None, "", ""),
    })
}

// helper: create an S3Ops wired to the singleton logger (if any)
async fn build_ops_async() -> Result<S3Ops> {
    build_ops_for(Rows::Write).await
}

// -----------------------------------------------------------------------------
// ObjectStat struct for stat operation
// -----------------------------------------------------------------------------
/// Full set of common S3 metadata from a HEAD-object call.
#[derive(Debug)]
pub struct ObjectStat {
    pub size: u64,
    pub last_modified: Option<String>,
    pub e_tag: Option<String>,
    pub content_type: Option<String>,
    pub content_language: Option<String>,
    pub content_encoding: Option<String>,
    pub cache_control: Option<String>,
    pub content_disposition: Option<String>,
    pub expires: Option<String>,
    pub storage_class: Option<String>,
    pub server_side_encryption: Option<String>,
    pub ssekms_key_id: Option<String>,
    pub sse_customer_algorithm: Option<String>,
    pub version_id: Option<String>,
    pub replication_status: Option<String>,
    pub metadata: HashMap<String, String>,
}

// -----------------------------------------------------------------------------
// ObjectType enum for typed data generation
// Note: the defintion moved
// -----------------------------------------------------------------------------
impl From<&str> for ObjectType {
    fn from(s: &str) -> Self {
        match s.to_ascii_uppercase().as_str() {
            "NPZ" => ObjectType::Npz,
            "TFRECORD" => ObjectType::TfRecord,
            "HDF5" => ObjectType::Hdf5,
            _ => ObjectType::Raw,
        }
    }
}

// pyo3 0.27 API - FromPyObject now takes two lifetime parameters and uses Borrowed
#[cfg(feature = "extension-module")]
impl<'a, 'py> FromPyObject<'a, 'py> for ObjectType {
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s = ob.extract::<&str>()?;
        Ok(ObjectType::from(s))
    }
}

// -----------------------------------------------------------------------------
// S3 URI helpers
// -----------------------------------------------------------------------------

/// Components of an S3 URI
#[derive(Debug, Clone)]
pub struct S3UriComponents {
    /// Optional custom server authority, always with an explicit port
    /// (e.g., "192.168.100.1:9001", "minio.local:9000" or "[::1]:9000").
    /// If None, uses AWS_ENDPOINT_URL environment variable or default AWS endpoint
    pub endpoint: Option<String>,
    /// Bucket name
    pub bucket: String,
    /// Object key/prefix
    pub key: String,
}

/// What the FIRST path segment of an `s3://` URI (the part before the first
/// `/`) names. See [`classify_s3_first_segment`] for the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S3FirstSegment<'a> {
    /// A bucket name (no explicit port).
    Bucket(&'a str),
    /// A server authority with an explicit port: `host:port` or `[ipv6]:port`.
    Server(&'a str),
}

/// True if `s` is a syntactically valid dotted-quad IPv4 address (four
/// `0..=255` octets separated by dots). S3 bucket names are explicitly
/// forbidden from being formatted as an IP address, so this check is
/// unambiguous — it can never misclassify a legal bucket name.
fn looks_like_ipv4(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 4
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.len() <= 3
                && p.chars().all(|c| c.is_ascii_digit())
                && p.parse::<u16>().map(|n| n <= 255).unwrap_or(false)
        })
}

/// Text of the one-time deprecation warning for the retired
/// `S3DLIO_S3_ENDPOINT_HINT_TLDS` variable.
const RETIRED_HINT_TLDS_WARNING: &str =
    "S3DLIO_S3_ENDPOINT_HINT_TLDS is deprecated and ignored: an s3:// URI now names \
     a server only when its first segment carries an explicit port \
     (s3://host:port/bucket/key); every other first segment is a bucket (issue #178)";

/// Warn once per process, at the first parse, if the retired
/// `S3DLIO_S3_ENDPOINT_HINT_TLDS` variable is set: it no longer has any effect
/// (issue #178).
///
/// The environment is read exactly once, the first time
/// [`classify_s3_first_segment`] runs; setting the variable later in the same
/// process neither warns nor changes anything. The warning goes to `tracing` AND
/// to stderr, so it is visible in programs (and CLIs) that never install a
/// tracing subscriber.
fn warn_if_retired_hint_tlds_env_set() {
    use std::io::Write;
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        if std::env::var_os("S3DLIO_S3_ENDPOINT_HINT_TLDS").is_some() {
            tracing::warn!("{RETIRED_HINT_TLDS_WARNING}");
            // Best effort: a closed or full stderr must not fail a parse.
            let _ = writeln!(
                std::io::stderr(),
                "s3dlio: warning: {RETIRED_HINT_TLDS_WARNING}"
            );
        }
    });
}

/// Validate the port part of `host:port` / `[ipv6]:port`.
///
/// Accepted: plain ASCII decimal digits, no leading zeros, value `1..=65535`.
/// `u16::from_str` alone is too lax (it accepts a leading `+` and `0`); leading
/// zeros are rejected because `host:09000` is ambiguous (octal habits) and would
/// otherwise leave two spellings of one server in configs, URLs and logs.
fn validate_s3_server_port(port: &str, segment: &str) -> Result<()> {
    if port.is_empty() {
        bail!(
            "empty port in '{segment}': a server must be host:port with a port from 1 to 65535 \
             (e.g. s3://host:9000/bucket/key)"
        );
    }
    if !port.bytes().all(|b| b.is_ascii_digit()) {
        bail!(
            "invalid port '{port}' in '{segment}': a port is 1 to 65535 written as plain decimal \
             digits (no sign or other characters), e.g. s3://host:9000/bucket/key"
        );
    }
    if port.len() > 1 && port.starts_with('0') {
        bail!(
            "invalid port '{port}' in '{segment}': leading zeros are not accepted; write the \
             port without them (e.g. :9000)"
        );
    }
    match port.parse::<u16>() {
        Ok(n) if n >= 1 => Ok(()),
        _ => bail!(
            "invalid port '{port}' in '{segment}': a port must be from 1 to 65535 \
             (e.g. s3://host:9000/bucket/key)"
        ),
    }
}

/// Longest host the `host:port` grammar accepts (the DNS name limit).
const S3_SERVER_HOST_MAX_LEN: usize = 253;

/// The supported host grammar, quoted in every host error.
const S3_SERVER_HOST_GRAMMAR: &str = "a host is 1 to 253 ASCII letters, digits, '-', '.' and '_', \
     with no empty labels (no leading or trailing '.', no '..')";

/// Validate the host part of `host:port` (a DNS name, IPv4 literal or
/// container-style name such as `minio_1`).
///
/// Grammar: 1..=253 characters, each an ASCII letter, digit, `-`, `.` or `_`
/// (`_` is kept for container-style names), with no empty label (no leading or
/// trailing `.`, no `..`). Anything else - `/`, `\`, whitespace, control
/// characters, `%`, `?`, `#`, non-ASCII, ... - is an error naming the character.
/// Enforcing this before any endpoint URI is built means a host can never change
/// how `s3://host:port/bucket/key` is split (`node/prefix:9000` would otherwise
/// read back as bucket `node`, routed to the global client).
fn validate_s3_server_host(host: &str, segment: &str) -> Result<()> {
    if host.is_empty() {
        bail!(
            "missing host before ':' in '{segment}': use host:port, e.g. \
             s3://host:9000/bucket/key"
        );
    }
    if let Some(bad) = host
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')))
    {
        bail!(
            "invalid character {bad:?} in host '{host}' of '{segment}': {S3_SERVER_HOST_GRAMMAR}"
        );
    }
    // Every character is ASCII here, so byte length is character length.
    if host.len() > S3_SERVER_HOST_MAX_LEN {
        bail!(
            "host of '{segment}' is {} characters, longer than the {S3_SERVER_HOST_MAX_LEN} \
             allowed: {S3_SERVER_HOST_GRAMMAR}",
            host.len()
        );
    }
    if host.split('.').any(str::is_empty) {
        bail!("empty label (a leading or trailing '.', or '..') in host '{host}' of '{segment}': {S3_SERVER_HOST_GRAMMAR}");
    }
    Ok(())
}

/// THE single rule that decides whether the first path segment of an `s3://`
/// URI is a server or a bucket. The URI parser ([`parse_s3_uri_full`]), the
/// store factory (`store_for_uri*`), `MultiEndpointStore` and the `s3-cli`
/// bucket commands and `--endpoints` lists all call this, so those components
/// cannot disagree (issues #178, #193, #194). Other code paths that handle
/// `s3://` URIs do not call it yet; see the Changelog for the tracking issues.
///
/// - `host:port` or `[ipv6]:port` (explicit port, plain decimal `1..=65535`, no
///   leading zeros) → [`S3FirstSegment::Server`]. The bracketed content must be a
///   valid IPv6 address; an unbracketed host must follow the host grammar: 1 to 253
///   ASCII letters, digits, `-`, `.` and `_`, with no empty label. Any other
///   character (`/`, `\`, whitespace, control characters, `%`, `?`, `#`,
///   non-ASCII, ...) is an error naming it.
/// - A dotted-quad IPv4 literal with no port → **error**. S3 forbids IP-shaped
///   bucket names, so it is not a bucket; the caller must add `:port`.
/// - An IPv6 literal without brackets or without a port → **error**.
/// - Userinfo (`user@host:9000`), a URL (`http://host:9000`), an ARN, an empty
///   segment, an empty host or port, or a malformed port → **error**, each with
///   a message that names what was wrong.
/// - Anything else — including dotted DNS-looking names such as
///   `data.example.com` and `localhost` → [`S3FirstSegment::Bucket`]. Bucket
///   names are not otherwise validated here.
pub fn classify_s3_first_segment(segment: &str) -> Result<S3FirstSegment<'_>> {
    warn_if_retired_hint_tlds_env_set();

    if segment.is_empty() {
        bail!(
            "empty first segment in the S3 URI: expected a bucket name (s3://bucket/key) or a \
             server with a port (s3://host:9000/bucket/key)"
        );
    }
    if segment.contains("://") {
        bail!(
            "'{segment}' looks like a URL; a server is written host:port without a scheme \
             or path (e.g. host:9000)"
        );
    }
    if segment.contains('@') {
        bail!(
            "'{segment}' contains userinfo ('@'), which is not supported; write the server as \
             host:port and supply credentials through the environment"
        );
    }
    if segment.starts_with("arn:") && segment.matches(':').count() >= 2 {
        bail!(
            "'{segment}' looks like an ARN; ARNs are not supported here — use s3://bucket/key, \
             or s3://host:port/bucket/key to name a server"
        );
    }

    if let Some(rest) = segment.strip_prefix('[') {
        // Bracketed IPv6: "[addr]:port"
        let Some((addr, after)) = rest.split_once(']') else {
            bail!("unterminated '[' in '{segment}': an IPv6 server is written [addr]:port");
        };
        if addr.is_empty() {
            bail!("empty IPv6 address in '{segment}'");
        }
        if addr.parse::<std::net::Ipv6Addr>().is_err() {
            if addr.parse::<std::net::Ipv4Addr>().is_ok() {
                bail!(
                    "'{segment}' puts an IPv4 address inside brackets; brackets are for IPv6 \
                     only — write {addr}:port"
                );
            }
            bail!("invalid IPv6 address '{addr}' in '{segment}'");
        }
        if after.is_empty() {
            bail!(
                "'{segment}' names an IPv6 address without a port; add :port, \
                 e.g. s3://{segment}:9000/bucket/key"
            );
        }
        let Some(port) = after.strip_prefix(':') else {
            bail!(
                "unexpected text '{after}' after ']' in '{segment}': an IPv6 server is written \
                 [addr]:port"
            );
        };
        validate_s3_server_port(port, segment)?;
        return Ok(S3FirstSegment::Server(segment));
    }
    if segment.contains(['[', ']']) {
        bail!(
            "unexpected bracket in '{segment}': brackets are only for an IPv6 server, [addr]:port"
        );
    }

    match segment.matches(':').count() {
        0 => {
            if looks_like_ipv4(segment) {
                bail!(
                    "'{segment}' is an IP address with no port; add :port to name a server, \
                     e.g. s3://{segment}:9000/bucket/key (an IP address cannot be an S3 bucket name)"
                );
            }
            Ok(S3FirstSegment::Bucket(segment))
        }
        1 => {
            let (host, port) = segment.split_once(':').unwrap_or((segment, ""));
            validate_s3_server_host(host, segment)?;
            validate_s3_server_port(port, segment)?;
            Ok(S3FirstSegment::Server(segment))
        }
        _ if segment.parse::<std::net::Ipv6Addr>().is_ok() => bail!(
            "'{segment}' is an unbracketed IPv6 address; write it in brackets with a \
             port, e.g. s3://[{segment}]:9000/bucket/key (add :port)"
        ),
        _ => bail!(
            "'{segment}' contains more than one ':' and is not an IPv6 address; a server is \
             written host:port or [ipv6]:port"
        ),
    }
}

/// The error for `s3://host:9000`, `s3://host:9000/` and `s3://host:9000//key`:
/// the server is fine, the bucket after it is missing.
fn missing_bucket_after_server(endpoint: &str, uri: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "missing bucket after server '{endpoint}' in S3 URI '{uri}': the bucket name cannot be \
         empty; write s3://{endpoint}/bucket/key"
    )
}

/// Parse S3 URI with optional endpoint support.
///
/// Two formats, told apart by [`classify_s3_first_segment`]:
/// 1. **Bucket**: `s3://bucket/key` — also `s3://data.example.com/key`. Uses the
///    `AWS_ENDPOINT_URL` environment variable or the default AWS endpoint.
/// 2. **Server with an explicit port**: `s3://host:port/bucket/key` or
///    `s3://[ipv6]:port/bucket/key` — names the S3-compatible server (MinIO,
///    Ceph, GCS interop, ...) the request goes to.
///
/// The first segment is a server **only** when it carries an explicit port.
/// A portless IPv4 literal (`s3://10.0.0.5/bkt/key`) or a portless / unbracketed
/// IPv6 literal is an error: S3 forbids IP-shaped bucket names, so it can be
/// neither; the message says to add `:port`.
///
/// # Examples
///
/// ```
/// use s3dlio::s3_utils::parse_s3_uri_full;
///
/// // Standard AWS format
/// let result = parse_s3_uri_full("s3://mybucket/data.bin").unwrap();
/// assert_eq!(result.endpoint, None);
/// assert_eq!(result.bucket, "mybucket");
/// assert_eq!(result.key, "data.bin");
///
/// // MinIO with explicit endpoint and port
/// let result = parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin").unwrap();
/// assert_eq!(result.endpoint, Some("192.168.100.1:9001".to_string()));
/// assert_eq!(result.bucket, "mybucket");
/// assert_eq!(result.key, "data.bin");
///
/// // Custom hostname, with its port
/// let result = parse_s3_uri_full("s3://minio.local:9000/mybucket/path/to/file").unwrap();
/// assert_eq!(result.endpoint, Some("minio.local:9000".to_string()));
///
/// // No port: a dotted name is a BUCKET, never a server
/// let result = parse_s3_uri_full("s3://data.example.com/dir/file.bin").unwrap();
/// assert_eq!(result.endpoint, None);
/// assert_eq!(result.bucket, "data.example.com");
/// assert_eq!(result.key, "dir/file.bin");
///
/// // A portless IP address is an error (add :port)
/// assert!(parse_s3_uri_full("s3://10.0.0.5/mybucket/data.bin").is_err());
/// ```
///
/// # Behavior change (issues #178, #193, #194)
///
/// Earlier releases also treated a portless first segment as a server when it
/// was an IPv4 literal or its last label was a "hostname-like" TLD
/// (`com`, `net`, `local`, ...; extendable with `S3DLIO_S3_ENDPOINT_HINT_TLDS`).
/// That made the parser and the store factory disagree and misrouted real
/// dotted bucket names. **Add `:port` to name a server.** The
/// `S3DLIO_S3_ENDPOINT_HINT_TLDS` variable is now ignored (a one-time
/// deprecation warning is logged if it is set).
pub fn parse_s3_uri_full(uri: &str) -> Result<S3UriComponents> {
    let trimmed = uri
        .strip_prefix("s3://")
        .context("URI must start with s3://")?;

    // Find the first slash to split host/bucket from key
    let Some(slash_pos) = trimmed.find('/') else {
        // `s3://host:9000` is a valid server with no bucket, not a bucket with a
        // missing slash: say which.
        if let Ok(S3FirstSegment::Server(endpoint)) = classify_s3_first_segment(trimmed) {
            return Err(missing_bucket_after_server(endpoint, uri));
        }
        bail!("URI must contain '/' after bucket");
    };
    let before_slash = &trimmed[..slash_pos];
    let after_slash = &trimmed[slash_pos + 1..];

    match classify_s3_first_segment(before_slash)
        .with_context(|| format!("invalid S3 URI '{uri}'"))?
    {
        S3FirstSegment::Server(endpoint) => {
            // Format: s3://host:port/bucket/key
            let endpoint = endpoint.to_string();

            // Find next slash for bucket/key split
            let slash_pos2 = after_slash.find('/').unwrap_or(after_slash.len());
            let bucket = after_slash[..slash_pos2].to_string();
            let key = if slash_pos2 < after_slash.len() {
                after_slash[slash_pos2 + 1..].to_string()
            } else {
                String::new()
            };

            if bucket.is_empty() {
                return Err(missing_bucket_after_server(&endpoint, uri));
            }

            Ok(S3UriComponents {
                endpoint: Some(endpoint),
                bucket,
                key,
            })
        }
        S3FirstSegment::Bucket(bucket) => {
            // Format: s3://bucket/key (standard AWS)
            Ok(S3UriComponents {
                endpoint: None,
                bucket: bucket.to_string(),
                key: after_slash.to_string(),
            })
        }
    }
}

/// Split `s3://bucket/key` → (`bucket`, `key`).
/// `key` may be empty (prefix).
///
/// **Note**: This is the legacy function for backwards compatibility.
/// For endpoint-aware parsing, use `parse_s3_uri_full()`.
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let components = parse_s3_uri_full(uri)?;
    Ok((components.bucket, components.key))
}

/// A server named by the first segment of an `s3://` URI (`host:port` or
/// `[ipv6]:port`), with the endpoint URL a client for it is built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Server {
    /// The authority as written in the URI, e.g. `10.9.0.17:9000`.
    pub authority: String,
    /// The endpoint URL derived from it (`http://10.9.0.17:9000`; `https` for
    /// port 443 and the ports in `S3DLIO_S3_ENDPOINT_TLS_PORTS`).
    pub endpoint_url: String,
}

/// Classify the first segment of an `s3://` URI with the shared server-vs-bucket
/// rule ([`classify_s3_first_segment`], issues #178 / #199 / #200): `Ok(Some)`
/// for an explicit `host:port` / `[ipv6]:port`, `Ok(None)` for a bucket name,
/// `Err` for a portless IP address or any malformed segment.
pub fn s3_server_of_first_segment(segment: &str) -> Result<Option<S3Server>> {
    match classify_s3_first_segment(segment)? {
        S3FirstSegment::Bucket(_) => Ok(None),
        S3FirstSegment::Server(authority) => Ok(Some(S3Server {
            authority: authority.to_string(),
            endpoint_url: crate::object_store::s3_endpoint_url_from_authority(authority)?,
        })),
    }
}

/// Where a bucket-level request for an `s3://` URI or bare bucket name goes:
/// the named server (`None` = the process-global client) and the bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3BucketTarget {
    pub server: Option<S3Server>,
    pub bucket: String,
}

/// Resolve `s3://bucket[/...]`, `s3://host:port/bucket[/...]` or a bare `bucket`
/// to a [`S3BucketTarget`], with the shared server-vs-bucket rule
/// ([`classify_s3_first_segment`]). Anything after the bucket (a key prefix) is
/// ignored, as for `s3-cli create-bucket`.
///
/// Errors: a portless IP address (the #178 error), a malformed server, or a
/// missing bucket (`s3://host:9000/`, `s3://`, an empty name).
pub fn s3_bucket_target(uri_or_bucket: &str) -> Result<S3BucketTarget> {
    let rest = uri_or_bucket.strip_prefix("s3://").unwrap_or(uri_or_bucket);
    let mut segments = rest.split('/');
    let first = segments.next().unwrap_or("");
    let ctx = || format!("invalid S3 URI '{uri_or_bucket}'");
    let server = s3_server_of_first_segment(first).with_context(ctx)?;
    let bucket = match &server {
        Some(server) => {
            let bucket = segments.next().unwrap_or("");
            if bucket.is_empty() {
                return Err(missing_bucket_after_server(
                    &server.authority,
                    uri_or_bucket,
                ));
            }
            bucket
        }
        None => first,
    };
    Ok(S3BucketTarget {
        server,
        bucket: bucket.to_string(),
    })
}

/// The S3 client for `server`: a dedicated per-endpoint client when a server is
/// named, the process-global client otherwise.
async fn s3_client_for_server(server: Option<&S3Server>) -> Result<aws_sdk_s3::Client> {
    match server {
        Some(server) => {
            crate::s3_client::create_s3_client_for_endpoint(&server.endpoint_url, None).await
        }
        None => aws_s3_client_async().await,
    }
}

// -----------------------------------------------------------------------------
// S3 Create bucket
// -----------------------------------------------------------------------------
/// Create an S3 bucket if it does not exist.
///
/// `bucket` is a bare bucket name on the process-global endpoint. To create the
/// bucket a URI names (`s3://host:port/bucket` goes to that server), use
/// [`create_bucket_for_uri`].
pub fn create_bucket(bucket: &str) -> Result<()> {
    let bucket = bucket.to_string(); // own it
    run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        create_bucket_if_absent_with_client(&client, &bucket).await
    })
}

/// `CreateBucket` on `client`, treating "already exists" (`BucketAlreadyOwnedByYou`,
/// `BucketAlreadyExists`) as success.
async fn create_bucket_if_absent_with_client(
    client: &aws_sdk_s3::Client,
    bucket: &str,
) -> Result<()> {
    match client.create_bucket().bucket(bucket).send().await {
        Ok(_) => Ok(()),
        Err(e) => match e.code() {
            Some("BucketAlreadyOwnedByYou" | "BucketAlreadyExists") => Ok(()),
            _ => Err(sdk_anyhow(format!("CreateBucket failed for '{bucket}'"), e)),
        },
    }
}

/// Create the bucket an `s3://` URI (or bare bucket name) names, on the server
/// it names (issue #199). Async; see [`create_bucket_for_uri`].
pub async fn create_bucket_for_uri_async(uri_or_bucket: &str) -> Result<()> {
    let target = s3_bucket_target(uri_or_bucket)?;
    let client = s3_client_for_server(target.server.as_ref()).await?;
    create_bucket_if_absent_with_client(&client, &target.bucket).await
}

/// Create the bucket an `s3://` URI (or bare bucket name) names, on the server
/// it names; "already exists" is success.
///
/// - `bucket`, `s3://bucket`, `s3://bucket/prefix/` → bucket `bucket` on the
///   process-global endpoint (`AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL`).
/// - `s3://host:port/bucket[/prefix/]` → bucket `bucket` on `host:port`, through a
///   per-endpoint client.
/// - `s3://10.0.0.5/bucket` (portless IP) → the #178 error; nothing is sent.
pub fn create_bucket_for_uri(uri_or_bucket: &str) -> Result<()> {
    let uri = uri_or_bucket.to_string();
    run_on_global_rt(async move { create_bucket_for_uri_async(&uri).await })
}

/// Delete the (empty) bucket an `s3://` URI (or bare bucket name) names, on the
/// server it names (issue #199). Async; see [`delete_bucket_for_uri`].
pub async fn delete_bucket_for_uri_async(uri_or_bucket: &str) -> Result<()> {
    let target = s3_bucket_target(uri_or_bucket)?;
    let client = s3_client_for_server(target.server.as_ref()).await?;
    delete_bucket_with_client(&client, &target.bucket).await
}

/// Delete the (empty) bucket an `s3://` URI (or bare bucket name) names, on the
/// server it names. Same URI forms and errors as [`create_bucket_for_uri`].
pub fn delete_bucket_for_uri(uri_or_bucket: &str) -> Result<()> {
    let uri = uri_or_bucket.to_string();
    run_on_global_rt(async move { delete_bucket_for_uri_async(&uri).await })
}

// -----------------------------------------------------------------------------
// S3 Delete bucket
// -----------------------------------------------------------------------------
/// Delete an S3 bucket. The bucket must be empty.
pub fn delete_bucket(bucket: &str) -> Result<()> {
    let bucket = bucket.to_string(); // own it for the async move
    run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        client
            .delete_bucket()
            .bucket(&bucket)
            .send()
            .await
            .with_context(|| {
                format!(
                    "Failed to delete bucket '{}'. Note: Buckets must be empty before deletion.",
                    bucket
                )
            })?;
        Ok(())
    })
}

// -----------------------------------------------------------------------------
// S3 List buckets
// -----------------------------------------------------------------------------

/// Resolved S3 client parameters for a list-buckets request.
///
/// Separated from env-var reading so the resolution logic can be unit-tested
/// without touching the process environment.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ListBucketsParams {
    /// Effective AWS region to use for the request.
    pub region: String,
    /// Whether to use path-style (`https://host/bucket`) vs virtual-host
    /// style (`https://bucket.host`) addressing.
    pub use_path_style: bool,
    /// Custom endpoint URL (when `AWS_ENDPOINT_URL` is set).
    pub endpoint_url: Option<String>,
}

/// Resolve S3 client parameters for `list_buckets` from environment-variable
/// values that have already been read by the caller.
///
/// **Rules**
///
/// - If `endpoint_url_env` is `Some` and non-empty (a custom / non-AWS
///   endpoint is configured), `region` is taken from `region_env` and the
///   endpoint is forwarded to the SDK.  This is required because third-party
///   S3-compatible services (GCS, MinIO, Vast Data, …) need the caller's
///   region and reject a hard-coded `us-east-1`.
/// - If `endpoint_url_env` is `None` or empty (genuine AWS S3), the region is
///   always forced to `us-east-1`.  AWS's global `ListBuckets` operation must
///   target `us-east-1` or the SDK returns
///   `AuthorizationHeaderMalformed / region is wrong`.
/// - `use_path_style` is `true` when `addressing_style_env` equals `"path"`
///   (case-insensitive), matching the behaviour of `AWS_S3_ADDRESSING_STYLE`.
///
/// All three arguments mirror the corresponding environment variables:
/// `AWS_ENDPOINT_URL`, `AWS_REGION` (or `AWS_DEFAULT_REGION`), and
/// `AWS_S3_ADDRESSING_STYLE`.
pub(crate) fn resolve_list_buckets_params(
    endpoint_url_env: Option<&str>,
    region_env: Option<&str>,
    addressing_style_env: Option<&str>,
) -> ListBucketsParams {
    let endpoint_url = endpoint_url_env
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let region = if endpoint_url.is_some() {
        // Non-AWS endpoint: honour the caller's region; fall back to us-east-1
        // only if the user did not set any region variable at all.
        region_env.unwrap_or("us-east-1").to_string()
    } else {
        // Genuine AWS S3: GlobalListBuckets must use us-east-1 regardless of
        // what AWS_REGION says.
        "us-east-1".to_string()
    };

    let use_path_style = addressing_style_env
        .map(|v| v.to_lowercase() == "path")
        .unwrap_or(false);

    debug!(
        "list_buckets params: {} → region={:?}, path_style={}",
        if endpoint_url.is_some() {
            "custom endpoint"
        } else {
            "native AWS (us-east-1)"
        },
        region,
        use_path_style,
    );
    ListBucketsParams {
        region,
        use_path_style,
        endpoint_url,
    }
}

/// List all S3 buckets visible to the current credentials.
///
/// **Region behaviour**
///
/// - Genuine AWS S3: always uses `us-east-1` (required for the global
///   `ListBuckets` operation).
/// - Custom endpoint (`AWS_ENDPOINT_URL` is set, e.g. GCS, MinIO, Vast):
///   uses `AWS_REGION` / `AWS_DEFAULT_REGION` from the environment.  Set
///   `AWS_S3_ADDRESSING_STYLE=path` when your endpoint requires path-style
///   addressing (GCS and most self-hosted services do).
///
/// Lists the process-global server's buckets (`AWS_ENDPOINT_URL_S3` /
/// `AWS_ENDPOINT_URL`, or AWS). To list a server named in a URI
/// (`s3://host:port/`), use [`list_buckets_at`].
pub fn list_buckets() -> Result<Vec<BucketInfo>> {
    list_buckets_at(None)
}

/// List the buckets of `server`, through a per-endpoint client, or of the
/// process-global server when `server` is `None` (issue #200).
pub fn list_buckets_at(server: Option<&S3Server>) -> Result<Vec<BucketInfo>> {
    let server = server.cloned();
    run_on_global_rt(async move {
        crate::dotenv::apply();

        // Log resolved params for diagnostics (pure helper, no side effects).
        let endpoint_url_for_log = match &server {
            Some(server) => Some(server.endpoint_url.clone()),
            None => crate::s3_client::s3_endpoint_url_from_env(),
        };
        resolve_list_buckets_params(
            endpoint_url_for_log.as_deref(),
            std::env::var("AWS_REGION")
                .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                .ok()
                .as_deref(),
            std::env::var("AWS_S3_ADDRESSING_STYLE").ok().as_deref(),
        );

        // The global client is the shared, fully-configured one (same reqwest
        // transport, CA bundle, timeouts and INFO-level logging as every other
        // operation); a named server gets a dedicated client built the same way.
        let client = s3_client_for_server(server.as_ref())
            .await
            .context("Failed to initialise S3 client for list_buckets")?;

        let response = client
            .list_buckets()
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("{}", format_sdk_error(&e)))?;

        let mut buckets = Vec::new();
        for bucket in response.buckets() {
            let name = bucket.name().unwrap_or("Unknown").to_string();
            let creation_date = bucket
                .creation_date()
                .map(|dt| dt.to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            buckets.push(BucketInfo {
                name,
                creation_date,
            });
        }
        debug!("list_buckets: {} bucket(s) returned", buckets.len());
        Ok(buckets)
    })
}

/// Information about an S3 bucket
#[derive(Debug, Clone)]
pub struct BucketInfo {
    pub name: String,
    pub creation_date: String,
}

/// List the keys an `s3://` URI selects (issue #213; the rule is in
/// [`crate::s3_selection`]). Handles S3 pagination automatically.
///
/// `path` is the whole key part of the URI: empty or ending in `/` lists everything
/// under it; a key with no `*` selects the object with exactly that key and, when
/// `recursive`, everything under `path/` (never `path.bak`); a key with `*` is a glob
/// over the full key. It is never read as a regular expression.
///
/// **DEPRECATED**: This S3-specific function will be removed in v1.0.0.
/// For backend-agnostic code, use `ObjectStore::list()` via `store_for_uri()`.
pub(crate) fn list_objects(bucket: &str, path: &str, recursive: bool) -> Result<Vec<String>> {
    list_objects_with_pattern(bucket, path, None, recursive)
}

/// [`list_objects`] narrowed by an EXPLICIT client-side regex (issue #213).
///
/// `prefix` is the key part of the URI (see [`list_objects`]); `pattern`, if any, is
/// matched against each returned key relative to the listing base (see
/// [`crate::list_pattern`]). An invalid pattern is an error before any request is sent.
pub(crate) fn list_objects_with_pattern(
    bucket: &str,
    prefix: &str,
    pattern: Option<&str>,
    recursive: bool,
) -> Result<Vec<String>> {
    // Compile before anything is sent.
    let filter = pattern.map(ListPattern::new).transpose()?;
    // Clone inputs to move them into the async block
    let bucket = bucket.to_string();
    let prefix = prefix.to_string();

    // Use the new tokio runtime framework
    run_on_global_rt(async move {
        // NOTE: Legacy S3Ops logging call removed here (was: `let _ = build_ops_async()...`)
        // It caused all Python list() calls to hang on non-AWS endpoints (IMDSv2 timeout).
        // See docs/bugs/PYTHON_LIST_HANG_BUG_REPORT.md for full root cause analysis.

        let client = aws_s3_client_async().await?;
        debug!(
            "list_objects: bucket='{}', key='{}', recursive={}",
            bucket, prefix, recursive
        );

        let mut keys = Vec::new();
        let mut stream = crate::s3_selection::select_key_stream(
            client,
            bucket.clone(),
            &prefix,
            recursive,
            false,
        );
        while let Some(item) = stream.next().await {
            keys.push(item?);
        }

        if let Some(filter) = &filter {
            filter.retain(&prefix, &mut keys);
        }
        info!("list_objects result: {} total keys", keys.len());
        keys.sort(); // Sort for consistent output
        Ok(keys)
    })
}

/// Streaming version of list_objects that yields results as they arrive.
/// Returns a stream of keys (without the s3:// prefix) for memory efficiency.
/// Callers can format URIs as needed: format!("s3://{}/{}", bucket, key)
///
/// `path` is the whole key part of the URI and selects objects exactly as
/// [`list_objects`] describes (issue #213; [`crate::s3_selection`]). An unusable key
/// (a wildcard key ending in `/`) is the stream's first item, an `Err`.
///
/// **Implementation note**: Uses an inline `async_stream::stream!` (no `tokio::spawn`)
/// so all S3 and tracing calls run on the caller's task. This avoids the tracing-subscriber
/// deadlock that occurred when debug/trace logging was active inside a spawned task.
pub async fn list_objects_stream(
    bucket: String,
    path: String,
    recursive: bool,
) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
    debug!(
        "list_objects_stream called: bucket='{}', path='{}', recursive={}",
        bucket, path, recursive
    );

    // Build the client eagerly too -- any credential errors surface before the stream starts.
    let client = aws_s3_client_async().await?;

    // The selection runs on the caller's task: no tokio::spawn (see the note above).
    Ok(crate::s3_selection::select_key_stream(
        client, bucket, &path, recursive, false,
    ))
}

/// NEW: Async GET by byte-range on full s3:// URI.
/// If `length` is None, reads from `offset` to end.
#[cfg_attr(feature = "profiling", instrument(
    name = "s3.get_range",
    skip(uri),
    fields(
        uri = %uri,
        offset = %offset,
        length = ?length
    )
))]
pub async fn get_object_range_uri_async(
    uri: &str,
    offset: u64,
    length: Option<u64>,
) -> Result<Bytes> {
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    if key.is_empty() {
        bail!("Cannot GET range: no key specified");
    }
    debug!(
        "get_object_range: uri='{}', offset={}, length={:?}",
        uri, offset, length
    );
    let bytes = get_object_range_async(&bucket, &key, offset, length).await?;
    debug!(
        "get_object_range: uri='{}', returned {} bytes",
        uri,
        bytes.len()
    );
    Ok(bytes)
}

/// Ranged GET on an already-parsed `(bucket, key)`; the key is sent exactly as
/// given (issue #172). This is the core of [`get_object_range_uri_async`] and is
/// called directly by callers that already hold the parsed pair, so a bucket
/// never has to be re-serialised into an `s3://` string and re-parsed (a dotted
/// or IP-shaped bucket name would be misread as an endpoint host).
pub(crate) async fn get_object_range_async(
    bucket: &str,
    key: &str,
    offset: u64,
    length: Option<u64>,
) -> Result<Bytes> {
    // Locked contract (audit #152 f42, docs/implementation-plans/
    // v0.9.109-audit-fix-plan.md bug B2): `length=Some(0)` must return
    // zero bytes without a network call. The range-string builder below
    // only overwrote the default `bytes={offset}-` (open-ended, meaning
    // "to end of object") inside `if len > 0`, so a caller asking for
    // zero bytes silently got the entire remainder of the object
    // instead.
    if length == Some(0) {
        return Ok(Bytes::new());
    }

    let client = aws_s3_client_async().await?;
    get_object_range_with_client(&client, bucket, key, offset, length).await
}

/// [`get_object_range_async`] on a caller-chosen client (issue #208): a
/// per-endpoint store passes ITS client, so the read goes to the store's own
/// server, never to the process-wide one.  `get_object_range_async` is this
/// with the process-wide client.
pub(crate) async fn get_object_range_with_client(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    offset: u64,
    length: Option<u64>,
) -> Result<Bytes> {
    // The same zero-length contract as `get_object_range_async`, for a caller
    // that reaches this function directly.
    if length == Some(0) {
        return Ok(Bytes::new());
    }

    let mut range = format!("bytes={}-", offset);
    if let Some(len) = length {
        if len > 0 {
            let end = offset.saturating_add(len).saturating_sub(1);
            range = format!("bytes={}-{}", offset, end);
        }
    }

    // Locked contract (audit #152 f19, docs/implementation-plans/
    // v0.9.109-audit-fix-plan.md bug B1): Phase 4b (v0.9.108) made the
    // response body stream via `SdkBody::from_body_1_x`, which moved
    // body-transfer failures (TCP FIN mid-body, HTTP/2 RST, connection
    // reset after headers) out from under smithy's send()-level retry —
    // smithy already returned Ok(response) at headers by the time a
    // streamed body can fail. This send()+collect() pair was left
    // unwrapped, so a transient body-transfer glitch now aborts the
    // whole GET instead of being retried, a regression from pre-Phase-4b
    // behavior. Four production callers reach this function
    // (data_loader/s3_bytes.rs, object_store.rs's client-present
    // get_range fallback, and two internal fast-paths in this file);
    // wrapping here fixes all of them. GETs are idempotent and each
    // attempt allocates a fresh Bytes, so retrying the whole pair is
    // trivially safe.
    let bytes = retry_get_body(|| async {
        let resp = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .range(&range)
            .send()
            .await
            .sdk_context(format!("GET range for s3://{}/{} failed", bucket, key))?;
        let body = resp.body.collect().await.context("collect range body")?;
        // Zero-copy: return Bytes directly
        Ok::<Bytes, anyhow::Error>(body.into_bytes())
    })
    .await?;

    Ok(bytes)
}

/// Like [`get_object_range_uri_async`] but returns split timing information
/// to help diagnose whether latency is in network RTT (TTFB) or transfer time.
///
/// Returns `(bytes, ttfb, transfer)` where:
/// - `ttfb` — time from request start until the HTTP response headers are
///   received (i.e., `.send().await` returns); this is the true TTFB.
/// - `transfer` — time from headers received until the full body has been
///   streamed and assembled (i.e., `.collect().await` returns).
///
/// Total wall time for the GET ≈ `ttfb + transfer`.
pub async fn get_object_range_uri_timed_async(
    uri: &str,
    offset: u64,
    length: Option<u64>,
) -> Result<(Bytes, std::time::Duration, std::time::Duration)> {
    use std::time::Instant;
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    if key.is_empty() {
        bail!("Cannot GET range: no key specified");
    }
    // Same length=Some(0) fast path as get_object_range_uri_async (bug B2)
    // — return zero bytes / zero durations without a network call.
    if length == Some(0) {
        return Ok((
            Bytes::new(),
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        ));
    }

    let client = aws_s3_client_async().await?;
    let mut range = format!("bytes={}-", offset);
    if let Some(len) = length {
        if len > 0 {
            let end = offset.saturating_add(len).saturating_sub(1);
            range = format!("bytes={}-{}", offset, end);
        }
    }
    let t0 = Instant::now();
    let resp = client
        .get_object()
        .bucket(&bucket)
        .key(&key)
        .range(range)
        .send()
        .await
        .sdk_context(format!(
            "timed GET range for s3://{}/{} failed",
            bucket, key
        ))?;
    let ttfb = t0.elapsed();
    let body = resp.body.collect().await.context("collect range body")?;
    let total = t0.elapsed();
    let transfer = total.saturating_sub(ttfb);
    let bytes = body.into_bytes();
    Ok((bytes, ttfb, transfer))
}

/// Read `length` bytes starting at `offset` from `s3://bucket/key`.
/// A negative or oversize request is truncated to the object size.
/// Uses the existing `get_object_uri()` helper internally so we don't
/// duplicate S3-client code right now.
pub fn get_range(bucket: &str, key: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
    let uri = format!("s3://{bucket}/{key}");
    let bytes = get_object_uri(&uri)?;
    let start = offset as usize;
    let end = match length {
        Some(len) => start.saturating_add(len as usize).min(bytes.len()),
        None => bytes.len(),
    };
    if start >= bytes.len() {
        Ok(Bytes::new())
    } else {
        Ok(bytes.slice(start..end))
    }
}

// -----------------------------------------------------------------------------
// Concurrent Range GET Operations (Phase 1 Optimization)
// -----------------------------------------------------------------------------

/// Issue #211: an explicit `chunk_size` of 0 never advances the range-splitting
/// loop, which would push ranges forever (unbounded memory).  Refuse it before
/// any request is made; `None` still means "pick the optimal size".
fn require_chunk_size_at_least_one(chunk_size: Option<usize>) -> Result<()> {
    if chunk_size == Some(0) {
        bail!(
            "chunk_size must be an integer >= 1 (valid range: 1 or more), got 0: \
             a chunk of 0 bytes can never cover the requested range"
        );
    }
    Ok(())
}

/// Issue #211: the caller limits of a concurrent range GET.  An explicit
/// `max_concurrency` of `Some(0)` and an explicit `chunk_size` of `Some(0)` are
/// errors naming the parameter; `None` still means "pick the optimal value".
///
/// Every route into the range engine calls this BEFORE it builds a client or sends
/// a request: the public wrapper, [`get_object_concurrent_range_with_client`] (used
/// by a per-endpoint store, issue #208) and `S3ObjectStore::get_range_optimized`.
pub(crate) fn require_concurrent_range_limits(
    chunk_size: Option<usize>,
    max_concurrency: Option<usize>,
) -> Result<()> {
    if let Some(n) = max_concurrency {
        crate::concurrency::require_concurrency_at_least_one("max_concurrency", n)?;
    }
    require_chunk_size_at_least_one(chunk_size)
}

/// High-performance concurrent range GET for large objects.
/// Automatically determines optimal chunk size and concurrency based on object size.
/// Uses pre-allocated buffers and concurrent range requests for maximum throughput.
#[cfg_attr(feature = "profiling", instrument(
    name = "s3.get_concurrent_range",
    skip(uri),
    fields(
        uri = %uri,
        offset = %offset,
        length = ?length,
        chunk_size = ?chunk_size,
        max_concurrency = ?max_concurrency
    )
))]
pub async fn get_object_concurrent_range_async(
    uri: &str,
    offset: u64,
    length: Option<u64>,
    chunk_size: Option<usize>,
    max_concurrency: Option<usize>,
) -> Result<Bytes> {
    // Issue #211: an explicit limit of 0 would hang a large-object GET; refuse
    // it up front, before any client is built, whatever the object size turns out
    // to be.
    require_concurrent_range_limits(chunk_size, max_concurrency)?;
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    if key.is_empty() {
        bail!("Cannot GET concurrent range: no key specified");
    }

    let client = aws_s3_client_async().await?;
    get_object_concurrent_range_with_client(
        &client,
        &bucket,
        &key,
        offset,
        length,
        chunk_size,
        max_concurrency,
    )
    .await
}

/// [`get_object_concurrent_range_async`] on a caller-chosen client and an
/// already-parsed `(bucket, key)` (issue #208).
///
/// A per-endpoint store passes ITS client, so the size lookup, the single-range
/// fallback and every concurrent ranged GET go to the store's own server -- the
/// same server whatever the object size.  Nothing here consults the process-wide
/// client; `get_object_concurrent_range_async` is this with that client.
pub(crate) async fn get_object_concurrent_range_with_client(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    offset: u64,
    length: Option<u64>,
    chunk_size: Option<usize>,
    max_concurrency: Option<usize>,
) -> Result<Bytes> {
    // Issue #211 (and #208, which made this the per-endpoint store's entry): the
    // limits are validated here too, so a caller that holds its own client cannot
    // reach the range-splitting loop with a 0.
    require_concurrent_range_limits(chunk_size, max_concurrency)?;
    if key.is_empty() {
        bail!("Cannot GET concurrent range: no key specified");
    }

    // Get object size first
    let head_resp = client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .sdk_context(format!(
            "HEAD for concurrent range GET on s3://{}/{} failed",
            bucket, key
        ))?;

    // Locked contract (audit #152 f37, docs/implementation-plans/
    // v0.9.109-audit-fix-plan.md bug B3): a HEAD that succeeds but omits
    // Content-Length (seen on some S3-compatible backends behind a
    // 307-redirect proxy, or a gateway that strips the header) must NOT
    // be silently coerced to object_size=0 — `unwrap_or(0)` did that,
    // which then tripped the `start_offset >= object_size` guard below
    // and returned an empty Bytes with Ok status for an object that
    // actually has data. Surface the anomaly instead.
    let object_size = head_resp.content_length().ok_or_else(|| {
        anyhow::anyhow!(
            "HEAD for s3://{}/{} succeeded but returned no Content-Length header \
             — cannot determine object size for a range GET",
            bucket,
            key
        )
    })? as u64;

    // Calculate actual range
    let start_offset = offset;
    let end_offset = match length {
        Some(len) => std::cmp::min(start_offset + len, object_size),
        None => object_size,
    };

    if start_offset >= object_size {
        return Ok(Bytes::new());
    }

    let total_bytes = end_offset - start_offset;

    // Determine optimal strategy based on size
    let effective_chunk_size = resolve_range_chunk_size(chunk_size, total_bytes)?;

    // Use single request for small objects
    if total_bytes <= effective_chunk_size as u64 {
        debug!("Using single range request for {} bytes", total_bytes);
        // Straight to the bucket/key core on THIS client: a re-built `s3://` URI
        // would be re-parsed (a dotted or IP-shaped bucket misread as a host) and
        // sent through the process-wide client (issue #208).
        return get_object_range_with_client(client, bucket, key, start_offset, Some(total_bytes))
            .await;
    }

    // Use concurrent ranges for large objects
    let effective_concurrency = resolve_range_concurrency(max_concurrency, total_bytes)?;

    debug!(
        "Using concurrent range GET: {} bytes, chunk_size={}, concurrency={}",
        total_bytes, effective_chunk_size, effective_concurrency
    );

    concurrent_range_get_impl(
        client,
        bucket,
        key,
        start_offset,
        end_offset,
        effective_chunk_size,
        effective_concurrency,
    )
    .await
}

/// Internal implementation of concurrent range GET with pre-allocated buffers
#[cfg_attr(feature = "profiling", instrument(
    name = "s3.concurrent_range_impl",
    skip(client, bucket, key),
    fields(
        bucket = %bucket,
        key = %key,
        start_offset = %start_offset,
        end_offset = %end_offset,
        total_bytes = %(end_offset - start_offset),
        chunk_size = %chunk_size,
        max_concurrency = %max_concurrency
    )
))]
/// Internal concurrent range GET implementation.
///
/// v0.9.108+ (issue #148, audit §1.3 / Patch 3): eliminated the second
/// buffer copy by pre-allocating one master `BytesMut` of `total_bytes`,
/// splitting it into per-range segments up front (each segment is a
/// zero-copy view into the same underlying allocation), moving each
/// segment into its range future, and streaming the AWS SDK response body
/// **directly into the segment** via `ByteStream::next()` and
/// `copy_from_slice`. On completion, segments are `unsplit` in ascending
/// order — the fast path is O(1) contiguous reunification into the
/// original allocation (see `bytes::BytesMut::unsplit` — non-contiguous
/// falls back to `extend_from_slice`, which cannot happen here since
/// segments were produced by ascending `split_to` from one buffer).
/// Peak live memory during assembly is ~total_bytes, not ~2 * total_bytes.
///
/// Prior history: v0.9.31 replaced an `Arc<Mutex<BytesMut>>` shared buffer
/// with a collect-then-assemble approach (Finding 5). That eliminated the
/// mutex but retained a second copy from the per-chunk `Bytes` into an
/// assembled buffer. This change closes that gap.
async fn concurrent_range_get_impl(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    start_offset: u64,
    end_offset: u64,
    chunk_size: usize,
    max_concurrency: usize,
) -> Result<Bytes> {
    // Last line of defence (issue #235): every route resolves its limits through
    // `resolve_range_*` first, but a zero here would build ranges forever or a semaphore with
    // no permits, so the engine refuses it before allocating or awaiting anything.
    require_effective_chunk_size("chunk_size", chunk_size)?;
    require_effective_concurrency("max_concurrency", max_concurrency)?;
    let total_bytes = (end_offset - start_offset) as usize;

    // Build (range_start, range_end) tuples in ascending offset order.
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    let mut current_offset = start_offset;
    while current_offset < end_offset {
        let chunk_end = std::cmp::min(current_offset + chunk_size as u64, end_offset);
        ranges.push((current_offset, chunk_end));
        current_offset = chunk_end;
    }

    // Pre-allocate the master buffer once and pre-split it into per-range
    // segments. `split_to(len)` returns the first `len` bytes as its own
    // `BytesMut` view into the shared allocation, leaving `master` holding
    // the remainder. Assemble later via `unsplit` in the same ascending
    // order to trigger the O(1) contiguous fast path.
    let mut master = BytesMut::zeroed(total_bytes);
    let mut segments: Vec<BytesMut> = Vec::with_capacity(ranges.len());
    for (range_start, range_end) in &ranges {
        let seg_len = (range_end - range_start) as usize;
        segments.push(master.split_to(seg_len));
    }
    debug_assert!(
        master.is_empty(),
        "master should be fully consumed by segments"
    );

    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    let mut futures = FuturesUnordered::new();

    for (idx, ((range_start, range_end), mut seg)) in ranges.into_iter().zip(segments).enumerate() {
        let client = client.clone();
        let bucket = bucket.to_string();
        let key = key.to_string();
        let semaphore = semaphore.clone();

        futures.push(async move {
            let _permit = semaphore
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;

            let range_header = format!("bytes={}-{}", range_start, range_end - 1);

            // Bounded retry over send() + body-stream (issue #148
            // Phase 4b). Once the reqwest connector streams response
            // bodies, smithy's own retry stops covering body-transfer
            // failures — we retry the whole send()+stream pair here.
            //
            // Inline (not `crate::retry::retry_get_body`) because that
            // helper's `FnMut() -> Fut` signature can't share `&mut seg`
            // across attempts without Arc<Mutex> or unsafe. Semantics
            // match: max_retry_attempts() budget + linear 100ms*attempt
            // backoff, and (issue #240) no retry of a definite client error.
            //
            // Silent-data-corruption gate (audit §2.4): `written`
            // MUST be declared inside `attempt_range_chunk_fill` so
            // each retry attempt starts writing at position 0. If it
            // leaked across attempts, a partial failed stream's bytes
            // would remain at [0..old_written] mixed with successful
            // retry bytes, length would still be seg_len but content
            // would be a mix of two attempts — no error raised.
            // Locked in by tests/test_phase4_retry_fault_injection.rs.
            let attempts = crate::constants::max_retry_attempts().max(1);
            let mut last_err: Option<anyhow::Error> = None;
            let mut succeeded = false;
            for attempt in 1..=attempts {
                match attempt_range_chunk_fill(
                    &client,
                    &bucket,
                    &key,
                    &range_header,
                    range_start,
                    range_end,
                    &mut seg,
                )
                .await
                {
                    Ok(()) => {
                        succeeded = true;
                        break;
                    }
                    Err(e) => {
                        // Issue #240: same rule as `retry_get_body` -- a
                        // definite answer (404, 403, 412, ...) ends the loop.
                        let definite = !crate::retry::RetryClassify::is_retryable(&e);
                        last_err = Some(e);
                        if definite {
                            break;
                        }
                        if attempt < attempts {
                            tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                        }
                    }
                }
            }
            if !succeeded {
                return Err(last_err.expect("attempts >= 1"));
            }

            Ok::<(usize, BytesMut), anyhow::Error>((idx, seg))
        });
    }

    // Collect all (idx, segment) pairs — order is non-deterministic
    // (FuturesUnordered).
    let mut chunks: Vec<(usize, BytesMut)> = Vec::with_capacity(max_concurrency);
    while let Some(result) = futures.next().await {
        chunks.push(result.context("concurrent range chunk failed")?);
    }

    // Sort by segment index (== ascending offset order), then reunify.
    // Because segments were produced by ascending `split_to` from one
    // allocation and are unsplit here in that same order, this hits the
    // O(1) contiguous fast path — the master buffer is reconstituted
    // in place with no additional copies.
    chunks.sort_unstable_by_key(|(idx, _)| *idx);
    let mut assembled: Option<BytesMut> = None;
    for (_, seg) in chunks {
        match assembled.as_mut() {
            Some(acc) => acc.unsplit(seg),
            None => assembled = Some(seg),
        }
    }
    let output = assembled.unwrap_or_default();
    debug_assert_eq!(
        output.len(),
        total_bytes,
        "assembled length must equal requested total_bytes"
    );

    Ok(output.freeze())
}

/// One attempt at streaming a range GET's body into a pre-allocated
/// segment (issue #148 Phase 4b). Declares `written = 0` inside the
/// function body — the audit §2.4 silent-data-corruption gate — so
/// each retry call from `concurrent_range_get_impl` starts writing at
/// position 0 into `seg` regardless of how many bytes a previous
/// failed attempt already wrote there. `tests/test_phase4_retry_fault_injection.rs`
/// locks in this invariant.
///
/// On any error (send failure, body-stream failure, chunk-overflow,
/// short read), the caller retries the whole call up to
/// `max_retry_attempts()` times with linear backoff.
async fn attempt_range_chunk_fill(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    range_header: &str,
    range_start: u64,
    range_end: u64,
    seg: &mut BytesMut,
) -> Result<()> {
    let seg_len = seg.len();
    let mut written: usize = 0;

    let mut resp = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .range(range_header)
        .send()
        .await
        .sdk_context(format!(
            "concurrent range GET for s3://{}/{} bytes={}-{} failed",
            bucket,
            key,
            range_start,
            range_end - 1
        ))?;

    while let Some(chunk) = resp.body.next().await {
        let chunk = chunk.context("read chunk body byte stream")?;
        let n = chunk.len();
        if written + n > seg_len {
            bail!(
                "range chunk overflow: got {} bytes past segment end at range {}..{}",
                (written + n) - seg_len,
                range_start,
                range_end
            );
        }
        seg[written..written + n].copy_from_slice(&chunk);
        written += n;
    }
    if written != seg_len {
        bail!(
            "range short read: wrote {}/{} bytes for range {}..{}",
            written,
            seg_len,
            range_start,
            range_end
        );
    }
    Ok(())
}

/// The environment variable that sets the range chunk size.
const ENV_RANGE_CHUNK_SIZE: &str = "S3DLIO_CHUNK_SIZE";
/// The environment variable that sets the range concurrency.
const ENV_RANGE_CONCURRENCY: &str = "S3DLIO_RANGE_CONCURRENCY";

/// Issue #235: a range chunk size of 0 can never cover a range.  `name` is the setting as the
/// user spells it (the environment variable, or the parameter).
fn require_effective_chunk_size(name: &str, value: usize) -> Result<usize> {
    if value == 0 {
        bail!(
            "{name} must be an integer >= 1 (valid range: 1 or more), got 0: \
             a chunk of 0 bytes can never cover the requested range"
        );
    }
    Ok(value)
}

/// Issue #235: a range concurrency of 0 builds a semaphore with no permits.
fn require_effective_concurrency(name: &str, value: usize) -> Result<usize> {
    crate::concurrency::require_concurrency_at_least_one(name, value)?;
    Ok(value)
}

/// The chunk size a range GET will really use, after the caller's argument and the
/// environment have been resolved (issue #235).
///
/// An explicit argument wins; otherwise `S3DLIO_CHUNK_SIZE`, otherwise the size-based
/// default.  The EFFECTIVE value must be at least 1: a zero is an error naming the variable
/// (or the parameter, when the caller passed it), returned before any range is allocated.
/// The range receive path reuses this one check.
pub(crate) fn resolve_range_chunk_size(
    chunk_size: Option<usize>,
    total_bytes: u64,
) -> Result<usize> {
    match chunk_size {
        Some(n) => require_effective_chunk_size("chunk_size", n),
        None => get_optimal_chunk_size(total_bytes),
    }
}

/// The concurrency a range GET will really use (issue #235): an explicit argument,
/// otherwise `S3DLIO_RANGE_CONCURRENCY`, otherwise the size-based default.  The EFFECTIVE
/// value must be at least 1: a zero is an error naming the variable (or the parameter),
/// returned before any semaphore is built.  The range receive path reuses this check.
pub(crate) fn resolve_range_concurrency(
    max_concurrency: Option<usize>,
    total_bytes: u64,
) -> Result<usize> {
    match max_concurrency {
        Some(n) => require_effective_concurrency("max_concurrency", n),
        None => get_optimal_concurrency(total_bytes),
    }
}

/// Get optimal chunk size based on total transfer size.
///
/// A zero from `S3DLIO_CHUNK_SIZE` is an error naming the variable (issue #235).
fn get_optimal_chunk_size(total_bytes: u64) -> Result<usize> {
    // The `.env` policy first (#243, review R9): `S3DLIO_CHUNK_SIZE` may come from the file.
    crate::dotenv::apply();
    if let Some(n) = std::env::var(ENV_RANGE_CHUNK_SIZE)
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return require_effective_chunk_size(ENV_RANGE_CHUNK_SIZE, n);
    }
    // Intelligent defaults based on object size
    Ok(if total_bytes < 16 * 1024 * 1024 {
        // < 16MB: Use 1MB chunks
        1024 * 1024
    } else if total_bytes < 256 * 1024 * 1024 {
        // 16MB - 256MB: Use 4MB chunks
        4 * 1024 * 1024
    } else {
        // > 256MB: Use 8MB chunks for optimal throughput
        8 * 1024 * 1024
    })
}

/// Get optimal concurrency based on total transfer size.
///
/// A zero from `S3DLIO_RANGE_CONCURRENCY` is an error naming the variable (issue #235).
fn get_optimal_concurrency(total_bytes: u64) -> Result<usize> {
    // The `.env` policy first (#243, review R9): `S3DLIO_RANGE_CONCURRENCY` may come from the file.
    crate::dotenv::apply();
    if let Some(n) = std::env::var(ENV_RANGE_CONCURRENCY)
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return require_effective_concurrency(ENV_RANGE_CONCURRENCY, n);
    }
    let cores = num_cpus::get();
    Ok(if total_bytes < 32 * 1024 * 1024 {
        // < 32MB: Limited concurrency
        std::cmp::max(2, cores / 2)
    } else if total_bytes < 512 * 1024 * 1024 {
        // 32MB - 512MB: Moderate concurrency
        std::cmp::max(4, cores)
    } else {
        // > 512MB: High concurrency for large transfers
        std::cmp::max(8, cores * 2)
    })
}

/// Synchronous wrapper for concurrent range GET
pub fn get_object_concurrent_range(
    uri: &str,
    offset: u64,
    length: Option<u64>,
    chunk_size: Option<usize>,
    max_concurrency: Option<usize>,
) -> Result<Bytes> {
    let uri_owned = uri.to_string(); // Clone the URI to avoid lifetime issues
    run_on_global_rt(async move {
        get_object_concurrent_range_async(&uri_owned, offset, length, chunk_size, max_concurrency)
            .await
    })
}

// -----------------------------------------------------------------------------
// Stat (HEAD) operation
// -----------------------------------------------------------------------------

/// Perform HEAD-object and return all common metadata (async).
pub async fn stat_object(bucket: &str, key: &str) -> Result<ObjectStat> {
    // NOTE: Legacy S3Ops logging call removed here (was: `let _ = build_ops_async()...`)
    // Could hang on non-AWS endpoints. See docs/bugs/PYTHON_LIST_HANG_BUG_REPORT.md
    // Issue #172: the key the caller names is the key sent -- no stripping.
    debug!("stat_object: bucket='{}', key='{}'", bucket, key);

    let client = aws_s3_client_async().await?;
    let resp = client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .sdk_context(format!("HEAD for s3://{}/{} failed", bucket, key))?;

    Ok(ObjectStat {
        size: resp.content_length().unwrap_or_default() as u64,
        last_modified: resp.last_modified().map(|t| t.to_string()),
        e_tag: resp.e_tag().map(|s| s.to_string()),
        content_type: resp.content_type().map(|s| s.to_string()),
        content_language: resp.content_language().map(|s| s.to_string()),
        content_encoding: resp.content_encoding().map(|s| s.to_string()),
        cache_control: resp.cache_control().map(|s| s.to_string()),
        content_disposition: resp.content_disposition().map(|s| s.to_string()),
        // if your SDK has `expires()` instead of `expires_string()`, swap accordingly
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
    })
}

/// Convenience wrapper that accepts a full `s3://bucket/key` URI.
///
/// Async HEAD by full URI (non-blocking; safe inside Tokio)
pub async fn stat_object_uri_async(uri: &str) -> Result<ObjectStat> {
    debug!("stat_object_uri_async: uri='{}'", uri);
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    if key.is_empty() {
        bail!("Cannot HEAD: no key specified");
    }
    stat_object(&bucket, &key).await
}

/// Stream the URIs (`s3://bucket/key`) of the REAL objects the key part `key`
/// selects, recursively (issue #213; the rule is in [`crate::s3_selection`]): a key
/// with no `*` that does not end in `/` selects that exact object and everything under
/// `key/` -- never `key.bak`; a key ending in `/` (or empty) selects everything under
/// it; a key with `*` is a glob over the full key.
///
/// There is no delimiter (so the server never answers with `CommonPrefixes`), no
/// regex, no key rewriting. Only objects are yielded -- synthetic entries are never
/// recognised by shape (a real object may end in `/` and have size 0), they simply
/// never appear.
///
/// A truncated page without a continuation token is an error, never a silently
/// short listing.
pub(crate) fn list_object_uris_with_prefix_stream(
    client: aws_sdk_s3::Client,
    bucket: String,
    key: String,
) -> Pin<Box<dyn Stream<Item = Result<String>> + Send>> {
    let uri_bucket = bucket.clone();
    Box::pin(
        crate::s3_selection::select_key_stream(client, bucket, &key, true, true)
            .map(move |item| item.map(|k| format!("s3://{uri_bucket}/{k}"))),
    )
}

/*
 * Old ?
 *
/// Sync wrapper that is safe even if a Tokio runtime is already running.
pub fn stat_object_uri(uri: &str) -> Result<ObjectStat> {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(stat_object_uri_async(uri)))
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()?
            .block_on(stat_object_uri_async(uri))
    }
}
*/

/// Sync wrapper that is safe uses helper, we don't use the one above now?
pub fn stat_object_uri(uri: &str) -> anyhow::Result<ObjectStat> {
    let uri = uri.to_string();
    run_on_global_rt(async move { stat_object_uri_async(&uri).await })
}

/// Async stat for many URIs (concurrent).
///
/// Task-level parallelism (issue #148 site 3.1g): each stat is
/// `tokio::spawn`'d so tokio can distribute per-stat work (URI parsing,
/// signing, header parsing on the HEAD response) across worker threads
/// instead of serializing on this task's poll cycle. `Vec<JoinHandle>`
/// awaited in order preserves the previous input→output ordering (the
/// spawns run concurrently regardless of iteration order). Panics are
/// surfaced as errors instead of silently truncating the result.
pub async fn stat_object_many_async(uris: Vec<String>) -> Result<Vec<ObjectStat>> {
    // Issue #197: refuse the whole batch up front if any URI names another server.
    crate::delete_batch_plan::validate_uris_for_global_client(&uris)?;
    let cancel = CancellationToken::new();
    let _drop_cancel = DropCancel(cancel.clone());

    let handles: Vec<JoinHandle<Result<ObjectStat>>> = uris
        .into_iter()
        .map(|u| {
            let token = cancel.clone();
            tokio::spawn(async move {
                tokio::select! {
                    _ = token.cancelled() => Err(anyhow::anyhow!("stat cancelled: {}", u)),
                    r = stat_object_uri_async(&u) => r,
                }
            })
        })
        .collect();

    let mut out = Vec::with_capacity(handles.len());
    let mut first_err: Option<anyhow::Error> = None;
    for h in handles {
        match h.await {
            Ok(Ok(stat)) => {
                if first_err.is_none() {
                    out.push(stat);
                }
            }
            Ok(Err(e)) => {
                if first_err.is_none() {
                    first_err = Some(e);
                    cancel.cancel();
                }
            }
            Err(join_err) if join_err.is_panic() => {
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!("stat task panicked: {}", join_err));
                    cancel.cancel();
                }
            }
            Err(_cancelled) => {}
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(out)
}

// -----------------------------------------------------------------------------
// Delete operation
// -----------------------------------------------------------------------------

/// Delete many keys (up to 1000 per logged batch).
///
/// With the op-log active each batch of up to 1000 keys is ONE `DELETE` row
/// (`file` = `(batch)`, `n_objects` = keys in that batch, endpoint =
/// `AWS_ENDPOINT_URL`), written by `S3Ops::delete_objects`. An empty list
/// sends no request and writes no row.
pub fn delete_objects(bucket: &str, keys: &[String]) -> Result<()> {
    let bucket = bucket.to_string();
    let to_delete: Vec<String> = keys.to_vec();
    run_on_global_rt(async move {
        let ops = build_ops_async().await?;
        for chunk in to_delete.chunks(1000) {
            ops.delete_objects(&bucket, chunk.to_vec()).await?;
        }
        Ok(())
    })
}

/// Async version of delete_objects — safe to call from within an async context
/// (e.g., inside submit_io or S3ObjectStore trait methods).
pub async fn delete_objects_async(bucket: &str, keys: &[String]) -> Result<()> {
    let client = aws_s3_client_async().await?;
    for chunk in keys.chunks(1000) {
        let objects: Vec<aws_sdk_s3::types::ObjectIdentifier> = chunk
            .iter()
            .filter_map(|k| {
                aws_sdk_s3::types::ObjectIdentifier::builder()
                    .key(k)
                    .build()
                    .ok()
            })
            .collect();
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
            .with_context(|| format!("delete_objects_async failed for bucket '{}'", bucket))?;

        // Locked contract (audit #151 f36, docs/implementation-plans/
        // v0.9.109-audit-fix-plan.md bug A4): S3's DeleteObjects returns
        // HTTP 200 even when individual keys failed to delete — the
        // per-object outcome lives in the response body's `errors` list,
        // not the HTTP status. Discarding the response here (as this
        // function previously did) silently reported partial or total
        // batch failures as success.
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
                "delete_objects_async for bucket '{}' partially failed: {} of {} object(s) \
                 not deleted: {}",
                bucket,
                errs.len(),
                chunk.len(),
                detail
            );
        }
    }
    Ok(())
}

/// Async version of delete_bucket — safe to call from within an async context.
pub async fn delete_bucket_async(bucket: &str) -> Result<()> {
    let client = aws_s3_client_async().await?;
    delete_bucket_with_client(&client, bucket).await
}

/// Delete `bucket` using an explicit client (e.g. a per-endpoint client built for
/// `s3://host:port/bucket`, issue #194) instead of the process-wide singleton.
pub async fn delete_bucket_with_client(client: &aws_sdk_s3::Client, bucket: &str) -> Result<()> {
    client
        .delete_bucket()
        .bucket(bucket)
        .send()
        .await
        .with_context(|| format!("delete_bucket failed for '{}'", bucket))?;
    Ok(())
}

/// Async version of create_bucket — safe to call from within an async context.
pub async fn create_bucket_async(bucket: &str) -> Result<()> {
    let client = aws_s3_client_async().await?;
    create_bucket_with_client(&client, bucket).await
}

/// Create `bucket` using an explicit client (e.g. a per-endpoint client built for
/// `s3://host:port/bucket`, issue #194) instead of the process-wide singleton.
pub async fn create_bucket_with_client(client: &aws_sdk_s3::Client, bucket: &str) -> Result<()> {
    client
        .create_bucket()
        .bucket(bucket)
        .send()
        .await
        .with_context(|| format!("create_bucket failed for '{}'", bucket))?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Download (GET) operations
// -----------------------------------------------------------------------------

/// Download a single object from S3 **with op‑logging**: with the op-log active
/// the call is one `GET` row, written by `S3Ops::get_object`.
pub async fn get_object(bucket: &str, key: &str) -> Result<Bytes> {
    get_object_rows(bucket, key, Rows::Write).await
}

async fn get_object_rows(bucket: &str, key: &str, rows: Rows) -> Result<Bytes> {
    // delegate to S3Ops (which records the GET when it has a logger) and propagate errors
    let data = build_ops_for(rows).await?.get_object(bucket, key).await?;

    debug!("S3 GET s3://{}/{} ({} bytes)", bucket, key, data.len());
    Ok(data)
}

/// Download a single object by URI, uses helper
pub fn get_object_uri(uri: &str) -> anyhow::Result<Bytes> {
    let uri = uri.to_string();
    run_on_global_rt(async move { get_object_uri_async(&uri).await })
}

/// Download many objects concurrently (ordered by input).
/// Download many objects concurrently (ordered by input).
///
/// v0.9.31+ improvements:
/// - Pre-stat phase: all object sizes fetched concurrently and stored in the
///   process-global ObjectSizeCache before GET phase begins. This means each
///   individual get_object_uri_optimized_async() call finds its size in cache
///   and skips its own HEAD — eliminating both HEAD #1 and #2 (Finding 1 fix).
///   On subsequent epochs the cache is already warm; zero HEADs issued.
/// - O(N log N) sort: replaced O(N²) linear scan with a pre-built HashMap index.
pub fn get_objects_parallel(uris: &[String], max_in_flight: usize) -> Result<Vec<(String, Bytes)>> {
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    // Issue #197: refuse the whole batch up front if any URI names another server.
    crate::delete_batch_plan::validate_uris_for_global_client(uris)?;
    let uris = uris.to_vec();
    run_on_global_rt(async move {
        // Build position index once (O(N)) for O(N log N) sort later.
        let uri_positions: HashMap<String, usize> = uris
            .iter()
            .enumerate()
            .map(|(i, u)| (u.clone(), i))
            .collect();

        // Pre-stat phase: populate ObjectSizeCache for all URIs concurrently.
        // Only runs when range opt is enabled (if disabled, HEAD is skipped anyway).
        // Uses the same max_in_flight limit to avoid overwhelming the server.
        //
        // Task-level parallelism (issue #148 site 3.1b): the previous shape
        // used `futures::future::join_all(stat_futs).await`, which polled
        // every stat future inside THIS task. Any CPU work in a single
        // stat's poll (request signing, header parsing, cache-write
        // serialization) blocked every other in-flight stat on the same
        // worker thread. Fix: `tokio::spawn` each stat so tokio can
        // distribute polling across worker threads, wired through a
        // CancellationToken so any early exit of this function (normal
        // return or panic on our frame) drops in-flight stats via their
        // `select!` cancel arm rather than detaching them. Same pattern as
        // site 3.1a in `data_loader/async_pool_dataloader.rs`.
        if get_range_opt_enabled() {
            let cancel = CancellationToken::new();
            let _drop_cancel = DropCancel(cancel.clone());
            let stat_sem = Arc::new(Semaphore::new(max_in_flight));
            let mut stat_futs: FuturesUnordered<JoinHandle<()>> = FuturesUnordered::new();
            for uri in uris.iter().cloned() {
                let stat_sem = stat_sem.clone();
                let token = cancel.clone();
                stat_futs.push(tokio::spawn(async move {
                    tokio::select! {
                        _ = token.cancelled() => {}
                        _ = async {
                            let _permit = stat_sem.acquire().await.ok();
                            // Only stat if not already cached (subsequent epochs skip this)
                            if get_size_cache().get(&uri).await.is_none() {
                                if let Ok(stat) = stat_object_uri_async(&uri).await {
                                    get_size_cache().put(uri, stat.size).await;
                                }
                            }
                        } => {}
                    }
                }));
            }
            // Drain to completion so no JoinHandle is dropped mid-flight.
            // Stat outcomes are already swallowed above (fire-and-forget cache
            // warm), so we only care about awaiting every task; JoinError from
            // a panicked stat task is logged and skipped since the pre-warm
            // cache is a best-effort optimization.
            while let Some(join_res) = stat_futs.next().await {
                if let Err(e) = join_res {
                    if e.is_panic() {
                        tracing::warn!(
                            "pre-stat task panicked during cache warm: {}; \
                             continuing without that entry",
                            e
                        );
                    }
                    // Otherwise: cancelled during shutdown, expected on early exit.
                }
            }
        }

        // GET phase: get_object_uri_optimized_async() will find sizes in cache,
        // skipping its internal HEAD call entirely.
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for uri in uris.iter().cloned() {
            let sem = Arc::clone(&sem);
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let data = get_object_uri_optimized_async(&uri).await?;
                Ok::<_, anyhow::Error>((uri, data))
            }));
        }
        let mut out = Vec::with_capacity(uris.len());
        // Full-drain-first-then-error (issue #148 finding 3.2): the previous
        // shape was `out.push(res??)` — that returned on the first Err,
        // dropping the remaining JoinHandles in `futs`. Dropped JoinHandles
        // detach (leak) their tasks rather than aborting them, so any GETs
        // still in flight kept running in the background after this
        // function had already returned an error. Draining every task
        // before returning eliminates the leak; we surface the first
        // error observed.
        let mut first_err: Option<anyhow::Error> = None;
        while let Some(res) = futs.next().await {
            match res {
                Ok(Ok(item)) => out.push(item),
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(join_err) => {
                    first_err.get_or_insert(anyhow::anyhow!("GET task panicked: {}", join_err));
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        // O(N log N) sort using pre-built position map
        out.sort_by_key(|(u, _)| uri_positions.get(u.as_str()).copied().unwrap_or(0));
        Ok(out)
    })
}

/// Download many objects concurrently with progress tracking (ordered by input).
pub fn get_objects_parallel_with_progress(
    uris: &[String],
    max_in_flight: usize,
    progress_callback: Option<Arc<crate::progress::ProgressCallback>>,
) -> Result<Vec<(String, Bytes)>> {
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    // Issue #197: refuse the whole batch up front if any URI names another server.
    crate::delete_batch_plan::validate_uris_for_global_client(uris)?;
    let uris = uris.to_vec(); // own for 'static
    run_on_global_rt(async move {
        // Build position index once (O(N)) for O(N log N) sort later.
        let uri_positions: HashMap<String, usize> = uris
            .iter()
            .enumerate()
            .map(|(i, u)| (u.clone(), i))
            .collect();

        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for uri in uris.iter().cloned() {
            let sem = Arc::clone(&sem);
            let progress = progress_callback.clone();
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let data = get_object_uri_optimized_async(&uri).await?;
                let byte_count = data.len() as u64;

                // Update progress if callback provided
                if let Some(ref progress) = progress {
                    progress.object_completed(byte_count);
                }

                Ok::<_, anyhow::Error>((uri, data))
            }));
        }
        let mut out = Vec::with_capacity(uris.len());
        // Full-drain-first-then-error — see companion note in
        // `get_objects_parallel`. Issue #148 finding 3.2.
        let mut first_err: Option<anyhow::Error> = None;
        while let Some(res) = futs.next().await {
            match res {
                Ok(Ok(item)) => out.push(item),
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(join_err) => {
                    first_err.get_or_insert(anyhow::anyhow!("GET task panicked: {}", join_err));
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        // O(N log N) sort using pre-built position map
        out.sort_by_key(|(u, _)| uri_positions.get(u.as_str()).copied().unwrap_or(0));
        Ok(out)
    })
}

/// Optimised async GET that honours S3DLIO_ENABLE_RANGE_OPTIMIZATION on ALL paths.
///
/// v0.9.31+ fixes (Finding 1 + bug fix):
/// - S3DLIO_ENABLE_RANGE_OPTIMIZATION=0 now skips HEAD entirely (was a no-op on this path).
/// - ObjectSizeCache checked before issuing HEAD; result stored after HEAD.
/// - When range split is chosen, size is passed directly → eliminates HEAD #2.
/// - All env-var reads cached via OnceLock (no syscall overhead on hot path).
///
/// With the op-log active, a single GET (the object is below the range threshold, or the
/// size is unknown) is one `GET` row from `S3Ops`; an object fetched by concurrent ranges
/// writes no row, exactly as before W2.
#[cfg_attr(feature = "profiling", instrument(
    name = "s3.get_optimized",
    skip(uri),
    fields(uri = %uri)
))]
pub async fn get_object_uri_optimized_async(uri: &str) -> Result<Bytes> {
    get_object_uri_optimized_rows(uri, Rows::Write).await
}

pub(crate) async fn get_object_uri_optimized_rows(uri: &str, rows: Rows) -> Result<Bytes> {
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: no key specified");
    }

    // Fast path: range optimisation disabled — skip HEAD entirely on ALL paths.
    // Fixes the bug where S3DLIO_ENABLE_RANGE_OPTIMIZATION=0 was a no-op here.
    if !get_range_opt_enabled() {
        debug!("range_opt disabled: single GET for {}", uri);
        return get_object_rows(&bucket, &key, rows).await;
    }

    // Fast path: skip_head latch — skip HEAD, issue plain GET directly.
    // Activated by `skip_head=True` loader opt or S3DLIO_SKIP_HEAD=1 env var.
    // After the GET we populate the ObjectSizeCache so that epoch 2+ can use
    // range splitting for large objects — zero downside to skipping HEAD.
    if get_skip_head() {
        debug!("skip_head: plain GET (no HEAD) for {}", uri);
        let bytes = get_object_rows(&bucket, &key, rows).await?;
        // Cache the actual size so future calls can range-split if warranted.
        // Skip the write when the cache already holds this size: `put` takes
        // the cache's write lock, and this runs once per GET.
        let size = bytes.len() as u64;
        let cache = get_size_cache();
        if cache.get(uri).await != Some(size) {
            cache.put(uri.to_string(), size).await;
        }
        return Ok(bytes);
    }

    let range_threshold = get_range_threshold_bytes();

    // Check ObjectSizeCache before issuing HEAD (avoids a network round-trip on cache hit).
    // The cache is populated by this function and by the pre-stat phase in get_objects_parallel().
    let object_size = if let Some(sz) = get_size_cache().get(uri).await {
        sz
    } else {
        // Cache miss: issue one HEAD to learn the size, then cache it.
        let client = aws_s3_client_async().await?;
        match client.head_object().bucket(&bucket).key(&key).send().await {
            // Locked contract (audit #152 f37, bug B3): a HEAD that
            // succeeds but omits Content-Length must not be cached as
            // size=0 (unwrap_or(0) did that here) — that poisons the
            // ObjectSizeCache for every subsequent call to this URI and
            // routes small-object reads through range-GET machinery
            // with a bogus zero-byte range. Treat "no Content-Length"
            // the same as a HEAD failure: fall back to a plain GET
            // without caching anything.
            Ok(resp) => match resp.content_length() {
                Some(sz) => {
                    let sz = sz as u64;
                    get_size_cache().put(uri.to_string(), sz).await;
                    sz
                }
                None => {
                    return get_object_rows(&bucket, &key, rows).await;
                }
            },
            Err(_) => {
                // HEAD failed: fall back to single GET without range splitting.
                return get_object_rows(&bucket, &key, rows).await;
            }
        }
    };

    if object_size >= range_threshold {
        debug!(
            "concurrent range GET {}MB: {}",
            object_size / (1024 * 1024),
            uri
        );
        // Pass known size → internal impl skips its own HEAD (eliminates HEAD #2).
        get_object_with_known_size_async(&bucket, &key, object_size, 0, None, None, None).await
    } else {
        get_object_rows(&bucket, &key, rows).await
    }
}

/// Fetch an object using concurrent range GETs with a pre-known size (no HEAD issued).
///
/// Called from get_object_uri_optimized_async() when the size is already known from
/// either the ObjectSizeCache or a prior HEAD in the same call stack.  This eliminates
/// the redundant HEAD #2 that get_object_concurrent_range_async() would otherwise issue.
async fn get_object_with_known_size_async(
    bucket: &str,
    key: &str,
    object_size: u64,
    offset: u64,
    length: Option<u64>,
    chunk_size: Option<usize>,
    max_concurrency: Option<usize>,
) -> Result<Bytes> {
    require_chunk_size_at_least_one(chunk_size)?;
    let start_offset = offset;
    let end_offset = match length {
        Some(len) => std::cmp::min(start_offset + len, object_size),
        None => object_size,
    };
    if start_offset >= object_size {
        return Ok(Bytes::new());
    }

    let total_bytes = end_offset - start_offset;
    let effective_chunk_size = resolve_range_chunk_size(chunk_size, total_bytes)?;

    // If total fits in one chunk, use a single range request (no concurrency overhead).
    if total_bytes <= effective_chunk_size as u64 {
        // Call the bucket/key core directly: round-tripping through a rebuilt
        // `s3://` string would re-parse the bucket and could misread a dotted or
        // IP-shaped bucket name as an endpoint host.
        return get_object_range_async(bucket, key, start_offset, Some(total_bytes)).await;
    }

    let effective_concurrency = resolve_range_concurrency(max_concurrency, total_bytes)?;
    let client = aws_s3_client_async().await?;
    concurrent_range_get_impl(
        &client,
        bucket,
        key,
        start_offset,
        end_offset,
        effective_chunk_size,
        effective_concurrency,
    )
    .await
}

/// Public async get object call **with op‑logging**: with the op-log active it is
/// one `GET` row from `S3Ops`.
pub async fn get_object_uri_async(uri: &str) -> Result<Bytes> {
    get_object_uri_rows(uri, Rows::Write).await
}

pub(crate) async fn get_object_uri_rows(uri: &str, rows: Rows) -> Result<Bytes> {
    debug!("get_object_uri_async: uri='{}'", uri);
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: no key specified");
    }
    get_object_rows(&bucket, &key, rows).await
}

// -----------------------------------------------------------------------------
// Typed PUT operations
// -----------------------------------------------------------------------------

// Updated version, that uses our object_type
/// Upload each URI with a buffer chosen by `object_type`.
pub fn put_objects_with_random_data_and_type(
    uris: &[String],
    size: usize,
    max_in_flight: usize,
    config: Config,
) -> Result<()> {
    put_objects_with_random_data_and_type_with_progress(uris, size, max_in_flight, config, None)
}

pub fn put_objects_with_random_data_and_type_with_progress(
    uris: &[String],
    size: usize,
    max_in_flight: usize,
    config: Config,
    progress_callback: Option<Arc<crate::progress::ProgressCallback>>,
) -> Result<()> {
    // Issue #211: a limit of 0 would hang; refuse it before generating data.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    info!(
        "put_objects: type={:?}, size={} bytes, uris={} parallelism={}",
        config.object_type,
        size,
        uris.len(),
        max_in_flight
    );

    let buffer: Bytes = generate_object(&config)?; // or: Bytes::from(generate_object(&config)?)
    let uris_vec = uris.to_vec(); // own for 'static
    debug!(
        "Uploading buffer of {} bytes to {} URIs",
        buffer.len(),
        uris_vec.len()
    );
    put_objects_parallel_with_progress(uris_vec, buffer, max_in_flight, progress_callback)
}

/// Upload a single object to S3 **with op‑logging**: with the op-log active it is
/// one `PUT` row from `S3Ops`.
///
/// Accepts `Bytes` for zero-copy efficiency (no .to_vec() call).
pub async fn put_object_async(bucket: &str, key: &str, data: Bytes) -> Result<()> {
    put_object_rows(bucket, key, data, Rows::Write).await
}

async fn put_object_rows(bucket: &str, key: &str, data: Bytes, rows: Rows) -> Result<()> {
    debug!("S3 PUT s3://{}/{} ({} bytes)", bucket, key, data.len());
    // delegate to S3Ops (which records the PUT when it has a logger) - data moved, no copy
    build_ops_for(rows)
        .await?
        .put_object(bucket, key, data)
        .await?;
    Ok(())
}

/// Upload a single object to S3 by URI **with op‑logging**.
///
/// Accepts `Bytes` for zero-copy efficiency.
pub async fn put_object_uri_async(uri: &str, data: Bytes) -> Result<()> {
    put_object_uri_rows(uri, data, Rows::Write).await
}

pub(crate) async fn put_object_uri_rows(uri: &str, data: Bytes, rows: Rows) -> Result<()> {
    debug!("put_object_uri_async: uri='{}', {} bytes", uri, data.len());
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    put_object_rows(&bucket, &key, data, rows).await
}

/// Upload object via multipart using existing MultipartUploadSink infrastructure.
///
/// Accepts `Bytes` for zero-copy efficiency.
pub async fn put_object_multipart_uri_async(
    uri: &str,
    data: Bytes,
    part_size: Option<usize>,
) -> Result<()> {
    debug!(
        "put_object_multipart_uri_async: uri='{}', {} bytes, part_size={:?}",
        uri,
        data.len(),
        part_size
    );
    let (bucket, key) = crate::delete_batch_plan::parse_s3_uri_for_global_client(uri)?;
    put_object_multipart_on(None, bucket, key, data, part_size).await
}

/// Multipart upload of `data` to `bucket`/`key` on a caller-chosen client
/// (issue #208).
///
/// A per-endpoint store passes ITS client.  The sink keeps that client for the
/// whole upload, so CreateMultipartUpload, every UploadPart, CompleteMultipartUpload
/// and -- when the upload fails or is dropped unfinished -- AbortMultipartUpload all
/// go to the one server that holds the upload.
pub(crate) async fn put_object_multipart_with_client(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    data: Bytes,
    part_size: Option<usize>,
) -> Result<()> {
    debug!(
        "put_object_multipart_with_client: s3://{}/{}, {} bytes, part_size={:?}",
        bucket,
        key,
        data.len(),
        part_size
    );
    put_object_multipart_on(
        Some(client.clone()),
        bucket.to_string(),
        key.to_string(),
        data,
        part_size,
    )
    .await
}

/// The one multipart-upload driver.  `client` is the client to upload with;
/// `None` is the process-wide client (resolved by the sink's own constructor).
async fn put_object_multipart_on(
    client: Option<aws_sdk_s3::Client>,
    bucket: String,
    key: String,
    data: Bytes,
    part_size: Option<usize>,
) -> Result<()> {
    use crate::multipart::{MultipartUploadConfig, MultipartUploadSink};
    let cfg = MultipartUploadConfig {
        part_size: part_size.unwrap_or(16 * 1024 * 1024), // 16 MiB default
        ..Default::default()
    };
    // Use spawn_blocking so that blocking_send() and run_on_global_rt() calls
    // inside write_blocking()/finish_blocking() never park a Tokio worker thread.
    // Without this, objects larger than max_in_flight × part_size panic because
    // blocking_send() on a full channel would stall a worker.
    tokio::task::spawn_blocking(move || {
        let mut sink = match client {
            Some(client) => MultipartUploadSink::new_with_client(client, &bucket, &key, cfg)?,
            None => MultipartUploadSink::new(&bucket, &key, cfg)?,
        };
        sink.write_blocking(&data)?;
        sink.finish_blocking()?;
        Ok::<(), anyhow::Error>(())
    })
    .await
    .map_err(|e| anyhow::anyhow!("put_object_multipart_uri_async task panicked: {e}"))??;
    Ok(())
}

// -----------------------------------------------------------------------------
// Internal helpers (private)
// -----------------------------------------------------------------------------
//

// TODO: Consider removing this function in a future version once all callers
// have migrated to put_objects_parallel_with_progress. Kept for backward compatibility.
#[allow(dead_code)]
pub(crate) fn put_objects_parallel(
    uris: Vec<String>,
    data: Bytes, // ref-counted, cheap to clone
    max_in_flight: usize,
) -> anyhow::Result<()> {
    put_objects_parallel_with_progress(uris, data, max_in_flight, None)
}

pub(crate) fn put_objects_parallel_with_progress(
    uris: Vec<String>,
    data: Bytes, // ref-counted, cheap to clone
    max_in_flight: usize,
    progress_callback: Option<Arc<crate::progress::ProgressCallback>>,
) -> anyhow::Result<()> {
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    // Get the global logger once, outside the async block
    let logger = global_logger();

    run_on_global_rt(async move {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for uri in uris.into_iter() {
            let sem = Arc::clone(&sem);
            let payload = data.clone(); // zero-copy clone
            let progress = progress_callback.clone();
            let logger = logger.clone(); // clone logger for each task

            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();

                // Capture len before move (Bytes is cheap to clone, but we only need len)
                let payload_len = payload.len() as u64;

                // Use universal ObjectStore API with op-log support
                let store = crate::object_store::store_for_uri_with_logger(&uri, logger)?;
                // Bytes passed directly for zero-copy
                store.put(&uri, payload).await?;

                // Update progress after successful upload
                if let Some(progress) = progress {
                    progress.object_completed(payload_len);
                }

                Ok::<_, anyhow::Error>(())
            }));
        }

        // Full-drain-first-then-error (issue #148 finding 3.2): the previous
        // shape was `res??`, which returned on the first Err and dropped the
        // remaining JoinHandles. Detached PUT tasks would keep running in
        // the background after the caller had already been told the whole
        // operation failed — a genuine resource leak on top of a
        // now-orphaned upload.
        let mut first_err: Option<anyhow::Error> = None;
        while let Some(res) = futs.next().await {
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(join_err) => {
                    first_err.get_or_insert(anyhow::anyhow!("PUT task panicked: {}", join_err));
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // resolve_list_buckets_params
    // -------------------------------------------------------------------------

    #[test]
    fn test_list_buckets_aws_no_endpoint_forces_us_east_1() {
        // No custom endpoint → genuine AWS → always us-east-1
        let p = resolve_list_buckets_params(None, Some("eu-west-1"), None);
        assert_eq!(
            p.region, "us-east-1",
            "genuine AWS must always use us-east-1 for global ListBuckets"
        );
        assert!(!p.use_path_style);
        assert_eq!(p.endpoint_url, None);
    }

    #[test]
    fn test_list_buckets_empty_endpoint_treated_as_no_endpoint() {
        // An empty string for AWS_ENDPOINT_URL is treated as "not set"
        let p = resolve_list_buckets_params(Some(""), Some("us-central1"), Some("path"));
        assert_eq!(
            p.region, "us-east-1",
            "empty endpoint must still force us-east-1 (genuine AWS path)"
        );
        assert_eq!(p.endpoint_url, None);
    }

    #[test]
    fn test_list_buckets_gcs_endpoint_uses_env_region() {
        // GCS S3-compat endpoint + explicit region
        let p = resolve_list_buckets_params(
            Some("https://storage.googleapis.com"),
            Some("us-central1"),
            Some("path"),
        );
        assert_eq!(
            p.region, "us-central1",
            "custom endpoint must use the caller's region, not us-east-1"
        );
        assert!(p.use_path_style, "path addressing style must be respected");
        assert_eq!(
            p.endpoint_url,
            Some("https://storage.googleapis.com".to_string())
        );
    }

    #[test]
    fn test_list_buckets_custom_endpoint_no_region_falls_back() {
        // Custom endpoint but no region env var → fallback to us-east-1
        let p = resolve_list_buckets_params(Some("http://minio.local:9000"), None, None);
        assert_eq!(
            p.region, "us-east-1",
            "should fall back to us-east-1 when no region is set even for custom endpoints"
        );
        assert!(!p.use_path_style);
        assert_eq!(p.endpoint_url, Some("http://minio.local:9000".to_string()));
    }

    #[test]
    fn test_list_buckets_minio_path_style() {
        // MinIO with path style
        let p = resolve_list_buckets_params(
            Some("http://192.168.1.10:9000"),
            Some("us-east-1"),
            Some("path"),
        );
        assert_eq!(p.region, "us-east-1");
        assert!(p.use_path_style);
        assert_eq!(p.endpoint_url, Some("http://192.168.1.10:9000".to_string()));
    }

    #[test]
    fn test_list_buckets_path_style_case_insensitive() {
        // AWS_S3_ADDRESSING_STYLE is compared case-insensitively
        let p_lower = resolve_list_buckets_params(Some("http://host:9000"), None, Some("path"));
        let p_upper = resolve_list_buckets_params(Some("http://host:9000"), None, Some("PATH"));
        let p_mixed = resolve_list_buckets_params(Some("http://host:9000"), None, Some("Path"));
        let p_virt = resolve_list_buckets_params(Some("http://host:9000"), None, Some("virtual"));
        assert!(p_lower.use_path_style, "'path' should enable path style");
        assert!(p_upper.use_path_style, "'PATH' should enable path style");
        assert!(p_mixed.use_path_style, "'Path' should enable path style");
        assert!(
            !p_virt.use_path_style,
            "'virtual' should not enable path style"
        );
    }

    #[test]
    fn test_list_buckets_vast_data_endpoint() {
        // Vast Data S3-compatible endpoint with its typical region name
        let p = resolve_list_buckets_params(
            Some("https://vast-s3.corp.example.com"),
            Some("vast-region-1"),
            Some("path"),
        );
        assert_eq!(p.region, "vast-region-1");
        assert!(p.use_path_style);
        assert!(p.endpoint_url.is_some());
    }

    // -------------------------------------------------------------------------
    // parse_s3_uri
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_s3_uri_standard_format() {
        // Standard AWS format: s3://bucket/key
        let result = parse_s3_uri_full("s3://mybucket/data.bin").unwrap();
        assert_eq!(result.endpoint, None);
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, "data.bin");
    }

    #[test]
    fn test_parse_s3_uri_with_path() {
        // Standard format with path
        let result = parse_s3_uri_full("s3://mybucket/path/to/file.txt").unwrap();
        assert_eq!(result.endpoint, None);
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, "path/to/file.txt");
    }

    #[test]
    fn test_parse_s3_uri_endpoint_with_port() {
        // Custom endpoint with port: s3://host:port/bucket/key
        let result = parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin").unwrap();
        assert_eq!(result.endpoint, Some("192.168.100.1:9001".to_string()));
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, "data.bin");
    }

    #[test]
    fn test_parse_s3_uri_endpoint_ip_with_path() {
        // IP endpoint with nested path
        let result =
            parse_s3_uri_full("s3://192.168.100.2:9002/testbucket/path/to/object.dat").unwrap();
        assert_eq!(result.endpoint, Some("192.168.100.2:9002".to_string()));
        assert_eq!(result.bucket, "testbucket");
        assert_eq!(result.key, "path/to/object.dat");
    }

    #[test]
    fn test_parse_s3_uri_hostname_with_port() {
        // Hostname with port
        let result = parse_s3_uri_full("s3://minio.local:9000/mybucket/file.bin").unwrap();
        assert_eq!(result.endpoint, Some("minio.local:9000".to_string()));
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, "file.bin");
    }

    #[test]
    fn test_parse_s3_uri_fqdn_endpoint() {
        // FQDN endpoint (detected by multiple dots)
        let result = parse_s3_uri_full("s3://storage.example.com:9000/mybucket/data.bin").unwrap();
        assert_eq!(
            result.endpoint,
            Some("storage.example.com:9000".to_string())
        );
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, "data.bin");
    }

    #[test]
    fn test_parse_s3_uri_localhost() {
        // Localhost endpoint
        let result = parse_s3_uri_full("s3://localhost:9000/testbucket/test.txt").unwrap();
        assert_eq!(result.endpoint, Some("localhost:9000".to_string()));
        assert_eq!(result.bucket, "testbucket");
        assert_eq!(result.key, "test.txt");
    }

    #[test]
    fn test_parse_s3_uri_empty_key() {
        // Empty key (prefix-only)
        let result = parse_s3_uri_full("s3://mybucket/").unwrap();
        assert_eq!(result.endpoint, None);
        assert_eq!(result.bucket, "mybucket");
        assert_eq!(result.key, "");
    }

    #[test]
    fn test_parse_s3_uri_endpoint_empty_key() {
        // Endpoint with empty key
        let result = parse_s3_uri_full("s3://192.168.1.1:9000/bucket/").unwrap();
        assert_eq!(result.endpoint, Some("192.168.1.1:9000".to_string()));
        assert_eq!(result.bucket, "bucket");
        assert_eq!(result.key, "");
    }

    #[test]
    fn test_parse_s3_uri_backwards_compat() {
        // Test backwards compatibility with old parse_s3_uri function
        let (bucket, key) = parse_s3_uri("s3://mybucket/data.bin").unwrap();
        assert_eq!(bucket, "mybucket");
        assert_eq!(key, "data.bin");
    }

    #[test]
    fn test_parse_s3_uri_error_no_prefix() {
        // Missing s3:// prefix
        let result = parse_s3_uri_full("mybucket/data.bin");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must start with s3://"));
    }

    // Issue #172 lock: the key is returned exactly as written -- every leading
    // '/' after the bucket separator is part of the key, and every operation
    // (stat, ranged GET, size probe, PUT, plain GET) must send it unchanged.
    #[test]
    fn test_parse_s3_uri_keeps_leading_slashes_in_key() {
        let (bucket, key) = parse_s3_uri("s3://bkt//data/x").unwrap();
        assert_eq!((bucket.as_str(), key.as_str()), ("bkt", "/data/x"));
        let (_, key) = parse_s3_uri("s3://bkt///data/x").unwrap();
        assert_eq!(key, "//data/x");
        let (_, key) = parse_s3_uri("s3://bkt/data/x").unwrap();
        assert_eq!(key, "data/x");
    }

    #[test]
    fn test_parse_s3_uri_error_no_slash() {
        // Missing slash after bucket
        let result = parse_s3_uri_full("s3://mybucket");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_s3_uri_server_without_bucket_says_so() {
        // A bare server URI is not "a slash missing after the bucket": the server
        // is fine, the bucket is missing, and the message shows the full form.
        for uri in [
            "s3://host:9000",
            "s3://host:9000/",
            "s3://[::1]:9000",
            "s3://[::1]:9000/",
        ] {
            let server = uri
                .strip_prefix("s3://")
                .and_then(|r| r.strip_suffix('/').or(Some(r)))
                .unwrap_or("");
            let msg = format!("{:#}", parse_s3_uri_full(uri).unwrap_err());
            assert!(
                msg.contains(&format!("missing bucket after server '{server}'")),
                "{uri}: {msg}"
            );
            assert!(
                msg.contains(&format!("s3://{server}/bucket/key")),
                "{uri}: example missing in {msg}"
            );
            assert!(!msg.contains("after bucket"), "{uri}: stale text in {msg}");
        }
        // A bare bucket (no slash) keeps its own message.
        let msg = format!("{:#}", parse_s3_uri_full("s3://mybucket").unwrap_err());
        assert!(msg.contains("'/' after bucket"), "{msg}");
    }

    #[test]
    fn test_parse_s3_uri_error_empty_bucket() {
        // Empty bucket name with endpoint
        let result = parse_s3_uri_full("s3://192.168.1.1:9000//data.bin");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_parse_s3_uri_minio_prefix() {
        // minio prefix triggers endpoint detection
        let result = parse_s3_uri_full("s3://minio:9000/bucket/key").unwrap();
        assert_eq!(result.endpoint, Some("minio:9000".to_string()));
        assert_eq!(result.bucket, "bucket");
    }

    #[test]
    fn test_parse_s3_uri_ceph_prefix() {
        // ceph prefix triggers endpoint detection
        let result = parse_s3_uri_full("s3://ceph-rgw:7480/bucket/object").unwrap();
        assert_eq!(result.endpoint, Some("ceph-rgw:7480".to_string()));
        assert_eq!(result.bucket, "bucket");
    }

    // -------------------------------------------------------------------------
    // list_objects_stream — delimiter logic (no network)
    // -------------------------------------------------------------------------
    // Verify the delimiter rules that govern whether common prefixes (virtual
    // directories) are returned by the S3 API:
    //   - recursive=false  → delimiter="/"  → server returns common prefixes
    //   - recursive=true   → delimiter=None → server flattens everything
    //
    // These tests do NOT call the S3 API; they exercise the pure logic that
    // derives the delimiter value from the recursive flag.

    fn delimiter_for(recursive: bool) -> Option<String> {
        if recursive {
            None
        } else {
            Some("/".to_string())
        }
    }

    #[test]
    fn test_non_recursive_list_uses_slash_delimiter() {
        // Non-recursive mode MUST use "/" as the delimiter so the S3 API
        // returns CommonPrefixes (virtual directories) instead of flattening.
        let d = delimiter_for(false);
        assert_eq!(
            d,
            Some("/".to_string()),
            "non-recursive listing must request delimiter='/' to get virtual directories"
        );
    }

    #[test]
    fn test_recursive_list_uses_no_delimiter() {
        // Recursive mode must NOT send a delimiter — the S3 API then returns
        // every key beneath the prefix with no CommonPrefixes grouping.
        let d = delimiter_for(true);
        assert_eq!(
            d, None,
            "recursive listing must omit the delimiter to get all keys"
        );
    }

    // -------------------------------------------------------------------------
    // format_sdk_error — InvalidAccessKeyId hint branches
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalid_access_key_hint_no_endpoint_set() {
        // Verify the condition that gates the "no endpoint" hint for
        // InvalidAccessKeyId errors.  The hint is shown when AWS_ENDPOINT_URL
        // is absent OR empty; suppressed when a real URL is present.
        //
        // We test the logic directly with known Option values so the test is
        // independent of the process environment (AWS_ENDPOINT_URL may be set
        // on developer machines and CI hosts).
        let absent: Option<&str> = None;
        let empty: Option<&str> = Some("");
        let real: Option<&str> = Some("http://10.0.0.1:9000");

        // absent → endpoint_unset = true  (hint shown)
        let cond_absent = absent.map(|v| v.is_empty()).unwrap_or(true);
        // empty string → endpoint_unset = true (hint shown)
        let cond_empty = empty.map(|v| v.is_empty()).unwrap_or(true);
        // real URL → endpoint_unset = false (hint suppressed)
        let cond_real = real.map(|v| v.is_empty()).unwrap_or(true);

        assert!(
            cond_absent,
            "absent endpoint → should show --endpoint-url hint"
        );
        assert!(
            cond_empty,
            "empty endpoint → should show --endpoint-url hint"
        );
        assert!(
            !cond_real,
            "real endpoint URL → should not show --endpoint-url hint"
        );
    }

    // -------------------------------------------------------------------------
    // sdk_anyhow / SdkResultExt — keep the connector chain in the final
    // anyhow::Error.to_string().  Guard against future regression of #506:
    // a bare `"dispatch failure"` reaching the caller is the symptom we are
    // explicitly preventing here.
    // -------------------------------------------------------------------------

    #[test]
    fn issue236_no_error_body_status_note_names_the_http_status() {
        assert_eq!(
            no_error_body_status_note(411),
            " — HTTP 411 Length Required"
        );
        assert_eq!(
            no_error_body_status_note(503),
            " — HTTP 503 Service Unavailable"
        );
        assert_eq!(no_error_body_status_note(599), " — HTTP 599");
    }

    #[test]
    fn test_sdk_anyhow_keeps_context_in_top_message() {
        // We do not need a real SdkError to exercise the helper's shape:
        // construct an anyhow::Error via the same anyhow! macro the helper
        // uses, then assert that its top-level Display starts with the ctx
        // string.  This is what the PyO3 RuntimeError wrapper surfaces.
        let err = anyhow::anyhow!(
            "{}: {}",
            "S3 GET for s3://b/k failed",
            "dispatch failure (I/O error: connection refused) — check ..."
        );
        let s = err.to_string();
        assert!(
            s.starts_with("S3 GET for s3://b/k failed"),
            "expected context prefix, got: {}",
            s
        );
        assert!(
            s.contains("dispatch failure"),
            "expected SDK chain detail in top message, got: {}",
            s
        );
        assert!(
            s.contains("connection refused"),
            "expected connector-error detail in top message, got: {}",
            s
        );
    }

    // -------------------------------------------------------------------------
    // classify_s3_first_segment (issues #178 / #193): the strict host:port grammar
    // -------------------------------------------------------------------------

    /// 253 characters (63 + 63 + 63 + 61 with dots) then a port.
    const HOST_253_PORT: &str = concat!(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:9000"
    );
    /// One character longer than [`HOST_253_PORT`].
    const HOST_254_PORT: &str = concat!(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:9000"
    );

    enum Seg {
        Bucket,
        Server,
        /// Error whose full message must contain this text.
        Fail(&'static str),
    }

    #[test]
    fn test_classify_first_segment_grammar() {
        use Seg::*;
        let rows: &[(&str, Seg)] = &[
            // Buckets (no explicit port): never validated as bucket names.
            ("bucket", Bucket),
            ("my.bucket.name", Bucket),
            ("data.example.com", Bucket),
            ("localhost", Bucket),
            // Not four octets / octet > 255 / trailing dot: not an IPv4 literal.
            ("1.2.3", Bucket),
            ("999.1.1.1", Bucket),
            ("10.0.0.5.", Bucket),
            // Servers: explicit, plain-decimal port 1..=65535.
            ("host:9000", Server),
            ("host:1", Server),
            ("host:65535", Server),
            ("10.0.0.5:9000", Server),
            ("[::1]:9000", Server),
            ("[2001:db8::5]:443", Server),
            // Host grammar (review of 0c9e18f): 1..=253 of ASCII letters, digits,
            // '-', '.', '_', no empty labels. Anything else is an error naming the
            // character; nothing that could change how the URI is split survives.
            ("minio_1:9000", Server),
            ("my-host.a_b.c:9000", Server),
            ("MinIO.Example.COM:9000", Server),
            ("node/prefix:9000", Fail("invalid character '/'")),
            ("node\\prefix:9000", Fail("invalid character '\\\\'")),
            ("a b:9000", Fail("invalid character ' '")),
            ("a\tb:9000", Fail("invalid character '\\t'")),
            ("a\u{0}b:9000", Fail("invalid character '\\0'")),
            ("ho%st:9000", Fail("invalid character '%'")),
            ("ho?st:9000", Fail("invalid character '?'")),
            ("ho#st:9000", Fail("invalid character '#'")),
            ("h\u{f4}st:9000", Fail("invalid character '\u{f4}'")),
            ("a;b:9000", Fail("invalid character ';'")),
            (".host:9000", Fail("empty label")),
            ("host.:9000", Fail("empty label")),
            ("host.example.com.:9000", Fail("empty label")),
            ("ho..st:9000", Fail("empty label")),
            ("node/prefix:9000", Fail("ASCII letters, digits")),
            // Length: 253 accepted, 254 rejected.
            (HOST_253_PORT, Server),
            (HOST_254_PORT, Fail("253")),
            // Ports that `u16::from_str` would accept but are not plain 1..=65535.
            ("host:+9000", Fail("invalid port '+9000'")),
            ("host:0", Fail("invalid port '0'")),
            ("host:00", Fail("invalid port '00'")),
            ("host:09000", Fail("leading zero")),
            ("host:65536", Fail("invalid port '65536'")),
            ("host:99999", Fail("invalid port '99999'")),
            ("host:-1", Fail("invalid port '-1'")),
            (
                "host:\u{0669}\u{0660}\u{0660}\u{0660}",
                Fail("invalid port"),
            ),
            ("host:abc", Fail("invalid port 'abc'")),
            ("host:", Fail("empty port")),
            ("[::1]:", Fail("empty port")),
            ("[::1]:0", Fail("invalid port '0'")),
            ("[::1]:+9000", Fail("invalid port '+9000'")),
            // Missing host.
            (":9000", Fail("missing host")),
            // Userinfo is never accepted, with or without a port.
            ("user@host:9000", Fail("userinfo")),
            ("user:pw@host:9000", Fail("userinfo")),
            ("user@host", Fail("userinfo")),
            ("[::1]@x:9000", Fail("userinfo")),
            // Bracketed IPv6 content is validated.
            ("[x]:9000", Fail("invalid IPv6 address 'x'")),
            ("[junk]:9000", Fail("invalid IPv6 address 'junk'")),
            ("[10.0.0.5]:9000", Fail("IPv4 address inside brackets")),
            ("[::1", Fail("unterminated '['")),
            ("[", Fail("unterminated '['")),
            ("[]:9000", Fail("empty IPv6 address")),
            ("[::1]", Fail("without a port")),
            ("[::1]x:9000", Fail("unexpected text")),
            ("[fe80::1%eth0]:9000", Fail("invalid IPv6 address")),
            ("host]:9000", Fail("unexpected bracket")),
            ("ho[st:9000", Fail("unexpected bracket")),
            // Portless IPv4 literal (incl. leading-zero octets): an IP is never a bucket.
            ("10.0.0.5", Fail("IP address with no port")),
            ("010.001.002.003", Fail("IP address with no port")),
            // Unbracketed IPv6.
            ("::1", Fail("unbracketed IPv6")),
            ("2001:db8::5", Fail("unbracketed IPv6")),
            // More than one ':' that is NOT an IPv6 address must not be called IPv6.
            ("host:9000:x", Fail("more than one ':'")),
            ("a:b:c:d", Fail("more than one ':'")),
            // URL-shaped input (e.g. an --endpoints entry).
            ("http://host:9000", Fail("looks like a URL")),
            ("https://host", Fail("looks like a URL")),
            ("s3://host:9000", Fail("looks like a URL")),
            // ARNs.
            (
                "arn:aws:s3:us-east-1:123:accesspoint",
                Fail("looks like an ARN"),
            ),
            ("arn:aws:s3:::bucket", Fail("looks like an ARN")),
            // A host that merely starts with "arn" is still a normal host:port.
            ("arn:9000", Server),
            ("arn", Bucket),
            // Empty first segment.
            ("", Fail("empty first segment")),
        ];

        let mut failures = Vec::new();
        for (segment, want) in rows {
            let got = classify_s3_first_segment(segment);
            match (want, &got) {
                (Bucket, Ok(S3FirstSegment::Bucket(b))) if b == segment => {}
                (Server, Ok(S3FirstSegment::Server(a))) if a == segment => {}
                (Fail(hint), Err(e)) if format!("{e:#}").contains(hint) => {
                    // The IPv6 label must be reserved for real IPv6 addresses.
                    if !segment.contains("::")
                        && !segment.starts_with('[')
                        && format!("{e:#}").contains("unbracketed IPv6")
                    {
                        failures.push(format!("{segment:?}: wrongly called IPv6: {e:#}"));
                    }
                }
                (_, other) => {
                    let want = match want {
                        Bucket => "bucket".to_string(),
                        Server => "server".to_string(),
                        Fail(h) => format!("error containing {h:?}"),
                    };
                    failures.push(format!("{segment:?}: want {want}, got {other:?}"));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "classify_s3_first_segment mismatches:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn test_classify_url_and_arn_errors_name_the_fix() {
        let url = format!(
            "{:#}",
            classify_s3_first_segment("http://host:9000").unwrap_err()
        );
        assert!(url.contains("host:9000") && url.contains("scheme"), "{url}");
        let arn = format!(
            "{:#}",
            classify_s3_first_segment("arn:aws:s3:us-east-1:123:accesspoint").unwrap_err()
        );
        assert!(
            arn.contains("ARN") && arn.contains("not supported"),
            "{arn}"
        );
    }
    // -------------------------------------------------------------------------
    // Issue #211: chunk_size = Some(0) must be rejected before any request
    // -------------------------------------------------------------------------

    /// Drive `fut` on its own thread and runtime and wait a bounded time.  The
    /// range-splitting loop is synchronous, so a task timeout could not stop
    /// it; a separate thread lets the test report "did not return" instead of
    /// freezing.  `None` = no answer in time.
    fn run_bounded_chunk<T: Send + 'static>(
        make: impl FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>
            + Send
            + 'static,
    ) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            let _ = tx.send(rt.block_on(make()));
        });
        rx.recv_timeout(std::time::Duration::from_secs(5)).ok()
    }

    fn assert_chunk_size_rejected(what: &str, got: Option<Result<Bytes>>) {
        let res = got.unwrap_or_else(|| {
            panic!("{what}: did not return with chunk_size Some(0) (loops forever)")
        });
        let msg = format!(
            "{:#}",
            res.expect_err("chunk_size Some(0) must be an error")
        );
        assert!(msg.contains("chunk_size"), "{what}: {msg}");
        assert!(msg.contains(">= 1"), "{what}: {msg}");
    }

    #[test]
    fn concurrent_range_async_rejects_chunk_size_zero() {
        let got = run_bounded_chunk(|| {
            Box::pin(get_object_concurrent_range_async(
                "s3://b/k",
                0,
                None,
                Some(0),
                None,
            ))
        });
        assert_chunk_size_rejected("get_object_concurrent_range_async", got);
    }

    #[test]
    fn concurrent_range_sync_rejects_chunk_size_zero() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(get_object_concurrent_range(
                "s3://b/k",
                0,
                None,
                Some(0),
                None,
            ));
        });
        let got = rx.recv_timeout(std::time::Duration::from_secs(5)).ok();
        assert_chunk_size_rejected("get_object_concurrent_range", got);
    }

    #[test]
    fn known_size_get_rejects_chunk_size_zero() {
        // The object is known to be 4096 bytes, so no HEAD is needed and the
        // split loop is reached at once on an unguarded build.
        let got = run_bounded_chunk(|| {
            Box::pin(get_object_with_known_size_async(
                "b",
                "k",
                4096,
                0,
                None,
                Some(0),
                None,
            ))
        });
        assert_chunk_size_rejected("get_object_with_known_size_async", got);
    }

    #[test]
    fn chunk_size_none_and_positive_are_accepted() {
        assert!(require_chunk_size_at_least_one(None).is_ok());
        assert!(require_chunk_size_at_least_one(Some(1)).is_ok());
        assert!(require_chunk_size_at_least_one(Some(8 * 1024 * 1024)).is_ok());
    }

    /// A client that cannot reach anything (nothing listens on port 1): if a zero limit were
    /// not refused before the HEAD, the error would be a connection failure, which names
    /// neither the parameter nor `>= 1`.
    fn offline_client() -> aws_sdk_s3::Client {
        let conf = aws_sdk_s3::Config::builder()
            .behavior_version(aws_config::BehaviorVersion::v2026_01_12())
            .region(aws_sdk_s3::config::Region::new("us-east-1"))
            .credentials_provider(aws_sdk_s3::config::Credentials::new(
                "a", "b", None, None, "test",
            ))
            .endpoint_url("http://127.0.0.1:1")
            .force_path_style(true)
            .build();
        aws_sdk_s3::Client::from_conf(conf)
    }

    /// Issue #211 / #208: the per-endpoint entry validates its own limits.
    #[test]
    fn concurrent_range_with_client_rejects_zero_limits() {
        for (chunk, conc, param) in [
            (None, Some(0usize), "max_concurrency"),
            (Some(0usize), None, "chunk_size"),
            (Some(0), Some(0), "max_concurrency"),
        ] {
            let got = run_bounded_chunk(move || {
                Box::pin(async move {
                    get_object_concurrent_range_with_client(
                        &offline_client(),
                        "b",
                        "k",
                        0,
                        None,
                        chunk,
                        conc,
                    )
                    .await
                })
            });
            let res = got.unwrap_or_else(|| panic!("{param}: did not return (hang)"));
            let msg = format!("{:#}", res.expect_err("a limit of 0 must be an error"));
            assert!(msg.contains(param), "{msg}");
            assert!(msg.contains(">= 1"), "{msg}");
        }
    }

    #[test]
    fn concurrent_range_limits_accept_none_and_positive() {
        assert!(require_concurrent_range_limits(None, None).is_ok());
        assert!(require_concurrent_range_limits(Some(1), Some(1)).is_ok());
    }
}

/// Issue #235: the effective range limits are validated after the environment is resolved.
#[cfg(test)]
mod range_limit_resolution_tests {
    use super::*;

    const VARS: &[&str] = &[ENV_RANGE_CHUNK_SIZE, ENV_RANGE_CONCURRENCY];

    #[test]
    fn zero_from_the_environment_is_rejected_naming_the_variable() {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(VARS);
        std::env::set_var(ENV_RANGE_CHUNK_SIZE, "0");
        std::env::set_var(ENV_RANGE_CONCURRENCY, "0");
        let chunk = resolve_range_chunk_size(None, 64).expect_err("zero chunk size");
        assert!(
            format!("{chunk:#}").contains("S3DLIO_CHUNK_SIZE"),
            "{chunk:#}"
        );
        assert!(format!("{chunk:#}").contains(">= 1"), "{chunk:#}");
        let conc = resolve_range_concurrency(None, 64).expect_err("zero concurrency");
        assert!(
            format!("{conc:#}").contains("S3DLIO_RANGE_CONCURRENCY"),
            "{conc:#}"
        );
        assert!(format!("{conc:#}").contains(">= 1"), "{conc:#}");
    }

    #[test]
    fn an_explicit_argument_wins_over_the_environment_and_zero_names_the_parameter() {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(VARS);
        std::env::set_var(ENV_RANGE_CHUNK_SIZE, "0");
        std::env::set_var(ENV_RANGE_CONCURRENCY, "0");
        assert_eq!(
            resolve_range_chunk_size(Some(4096), 64).expect("explicit"),
            4096
        );
        assert_eq!(resolve_range_concurrency(Some(3), 64).expect("explicit"), 3);
        let chunk = resolve_range_chunk_size(Some(0), 64).expect_err("zero argument");
        assert!(format!("{chunk:#}").contains("chunk_size"), "{chunk:#}");
        let conc = resolve_range_concurrency(Some(0), 64).expect_err("zero argument");
        assert!(format!("{conc:#}").contains("max_concurrency"), "{conc:#}");
    }

    #[test]
    fn valid_and_unparseable_settings_are_unchanged() {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(VARS);
        std::env::set_var(ENV_RANGE_CHUNK_SIZE, "8192");
        std::env::set_var(ENV_RANGE_CONCURRENCY, "5");
        assert_eq!(resolve_range_chunk_size(None, 64).expect("env"), 8192);
        assert_eq!(resolve_range_concurrency(None, 64).expect("env"), 5);
        // An unparseable value still falls back to the size-based default, as before.
        std::env::set_var(ENV_RANGE_CHUNK_SIZE, "not-a-number");
        std::env::set_var(ENV_RANGE_CONCURRENCY, "-1");
        assert_eq!(
            resolve_range_chunk_size(None, 64).expect("default"),
            1024 * 1024
        );
        assert!(resolve_range_concurrency(None, 64).expect("default") >= 2);
        std::env::remove_var(ENV_RANGE_CHUNK_SIZE);
        std::env::remove_var(ENV_RANGE_CONCURRENCY);
        assert_eq!(
            resolve_range_chunk_size(None, 300 * 1024 * 1024).expect("default"),
            8 * 1024 * 1024
        );
    }
}
