// src/s3_utils.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides high‑level S3 operations: list, get, delete, typed PUT.

use anyhow::{bail, Context, Result};
use aws_sdk_s3::error::ProvideErrorMetadata;
//use aws_sdk_s3::primitives::ByteStream;
use futures::{stream::FuturesUnordered, Stream, StreamExt};
#[cfg(feature = "extension-module")]
use pyo3::{FromPyObject, PyAny};
use regex::Regex;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::Semaphore;

use bytes::{Bytes, BytesMut};
#[cfg(feature = "profiling")]
use tracing::instrument;

use tracing::{debug, info};

// data generation helpers
use crate::config::Config;
use crate::config::ObjectType;
use crate::data_gen::generate_object;

// S3 client creation
use crate::s3_client::{aws_s3_client_async, run_on_global_rt};

// S3 operation logging
use crate::s3_logger::global_logger;
use crate::s3_ops::S3Ops;

// Object size cache for eliminating redundant HEAD requests
use crate::object_size_cache::ObjectSizeCache;

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

/// Whether range optimization (range splitting) is enabled.
/// Cached once per process from S3DLIO_ENABLE_RANGE_OPTIMIZATION.
/// Setting =0/false/no/off/disable skips HEAD entirely on the get_many() path
/// (S3ObjectStore::get() honours this too — now consistent on ALL paths).
fn get_range_opt_enabled() -> bool {
    *RANGE_OPT_ENABLED.get_or_init(|| {
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
fn get_size_cache() -> &'static Arc<ObjectSizeCache> {
    GLOBAL_SIZE_CACHE.get_or_init(|| {
        let ttl_secs = std::env::var("S3DLIO_SIZE_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(3600); // 1 hour — lasts across many training epochs
        Arc::new(ObjectSizeCache::new(Duration::from_secs(ttl_secs)))
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// helper: create an S3Ops wired to the singleton logger (if any)
async fn build_ops_async() -> Result<S3Ops> {
    let client = aws_s3_client_async().await?;
    let logger = global_logger();
    let client_id = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
    let endpoint = std::env::var("AWS_ENDPOINT_URL").unwrap_or_default();
    Ok(S3Ops::new(client, logger, &client_id, &endpoint))
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
    /// Optional custom endpoint (e.g., "192.168.100.1:9001" or "minio.local:9000")
    /// If None, uses AWS_ENDPOINT_URL environment variable or default AWS endpoint
    pub endpoint: Option<String>,
    /// Bucket name
    pub bucket: String,
    /// Object key/prefix
    pub key: String,
}

/// Parse S3 URI with optional endpoint support (Hybrid Approach - Option C)
///
/// Supports two formats:
/// 1. **Standard**: `s3://bucket/key`
///    - Uses AWS_ENDPOINT_URL environment variable or default AWS endpoint
///    - Compatible with existing code
///
/// 2. **With custom endpoint**: `s3://host:port/bucket/key` or `s3://host/bucket/key`
///    - Explicitly specifies endpoint for MinIO, Ceph, or other S3-compatible storage
///    - Host can be IP address, hostname, or domain
///    - Port is optional (defaults to 9000 for custom endpoints, 443 for AWS)
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
/// // Custom hostname
/// let result = parse_s3_uri_full("s3://minio.local:9000/mybucket/path/to/file").unwrap();
/// assert_eq!(result.endpoint, Some("minio.local:9000".to_string()));
/// ```
pub fn parse_s3_uri_full(uri: &str) -> Result<S3UriComponents> {
    let trimmed = uri
        .strip_prefix("s3://")
        .context("URI must start with s3://")?;

    // Find the first slash to split host/bucket from key
    let slash_pos = trimmed
        .find('/')
        .context("URI must contain '/' after bucket")?;
    let before_slash = &trimmed[..slash_pos];
    let after_slash = &trimmed[slash_pos + 1..];

    // Heuristic to detect if first part is an endpoint:
    // - Contains ':' (has port) → definitely endpoint
    // - Starts with digit → likely IP address → endpoint
    // - Contains multiple dots → likely IP or FQDN → endpoint
    // - Common hostnames like "minio", "ceph" → endpoint
    let is_endpoint = before_slash.contains(':')
        || before_slash.starts_with(|c: char| c.is_ascii_digit())
        || before_slash.matches('.').count() >= 2
        || before_slash.starts_with("minio")
        || before_slash.starts_with("ceph")
        || before_slash.contains("localhost");

    if is_endpoint {
        // Format: s3://endpoint:port/bucket/key or s3://endpoint/bucket/key
        let endpoint = before_slash.to_string();

        // Find next slash for bucket/key split
        let slash_pos2 = after_slash.find('/').unwrap_or(after_slash.len());
        let bucket = after_slash[..slash_pos2].to_string();
        let key = if slash_pos2 < after_slash.len() {
            after_slash[slash_pos2 + 1..].to_string()
        } else {
            String::new()
        };

        if bucket.is_empty() {
            bail!("Bucket name cannot be empty in URI: {}", uri);
        }

        Ok(S3UriComponents {
            endpoint: Some(endpoint),
            bucket,
            key,
        })
    } else {
        // Format: s3://bucket/key (standard AWS)
        let bucket = before_slash.to_string();
        let key = after_slash.to_string();

        Ok(S3UriComponents {
            endpoint: None,
            bucket,
            key,
        })
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

// -----------------------------------------------------------------------------
// S3 Create bucket
// -----------------------------------------------------------------------------
/// Create an S3 bucket if it does not exist.
pub fn create_bucket(bucket: &str) -> Result<()> {
    let bucket = bucket.to_string(); // own it
    run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        match client.create_bucket().bucket(&bucket).send().await {
            Ok(_) => Ok(()),
            Err(e) => {
                if let Some(code) = e.code() {
                    if code == "BucketAlreadyOwnedByYou" || code == "BucketAlreadyExists" {
                        Ok(())
                    } else {
                        Err(e.into())
                    }
                } else {
                    Err(e.into())
                }
            }
        }
    })
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
pub fn list_buckets() -> Result<Vec<BucketInfo>> {
    run_on_global_rt(async move {
        use aws_config::timeout::TimeoutConfig;
        use aws_sdk_s3::config::Region;
        use std::time::Duration;
        use tracing::debug;

        dotenvy::dotenv().ok();

        // Read env vars once and resolve config through the pure helper so the
        // logic stays testable without mutating the process environment.
        let params = resolve_list_buckets_params(
            std::env::var("AWS_ENDPOINT_URL").ok().as_deref(),
            std::env::var("AWS_REGION")
                .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                .ok()
                .as_deref(),
            std::env::var("AWS_S3_ADDRESSING_STYLE").ok().as_deref(),
        );

        debug!(
            "ListBuckets params: region={}, path_style={}, endpoint={:?}",
            params.region, params.use_path_style, params.endpoint_url
        );

        let timeout_config = TimeoutConfig::builder()
            .connect_timeout(Duration::from_secs(5))
            .operation_timeout(Duration::from_secs(120))
            .build();

        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::v2026_01_12())
            .region(Region::new(params.region))
            .timeout_config(timeout_config);

        if let Some(endpoint) = params.endpoint_url {
            loader = loader.endpoint_url(endpoint);
        }

        let sdk_cfg = loader.load().await;

        let s3_cfg = aws_sdk_s3::config::Builder::from(&sdk_cfg)
            .force_path_style(params.use_path_style)
            .build();

        let client = aws_sdk_s3::Client::from_conf(s3_cfg);

        let response = client
            .list_buckets()
            .send()
            .await
            .context("Failed to list S3 buckets")?;

        let mut buckets = Vec::new();
        let bucket_list = response.buckets();
        for bucket in bucket_list {
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

/// List keys under a path, with regex matching on the final path component.
/// Handles S3 pagination automatically.
///
/// **DEPRECATED**: This S3-specific function will be removed in v1.0.0.
/// For backend-agnostic code, use `ObjectStore::list()` via `store_for_uri()`.
///
/// # Migration
/// ```ignore
/// // Old (S3-specific):
/// let objects = list_objects("bucket", "prefix/", true)?;
///
/// // New (Universal):
/// use s3dlio::api::{store_for_uri, ObjectStore};
/// let store = store_for_uri("s3://bucket/prefix/")?;
/// let objects = store.list("", true, None).await?;
/// ```
pub(crate) fn list_objects(bucket: &str, path: &str, recursive: bool) -> Result<Vec<String>> {
    // Clone inputs to move them into the async block
    let bucket = bucket.to_string();
    let path = path.to_string();

    // Use the new tokio runtime framework
    run_on_global_rt(async move {
        // 1. Determine the S3 prefix and the regex pattern from the input path.
        let (prefix, pattern) = match path.rfind('/') {
            Some(index) => {
                let (p, pat) = path.split_at(index + 1);
                (p, pat)
            }
            None => ("", path.as_str()),
        };

        // If the pattern is empty (e.g., path ends in "/"), default to matching everything.
        let final_pattern = if pattern.is_empty() { ".*" } else { pattern };

        // Use the recursive flag as-is - don't auto-enable recursion
        let effective_recursive = recursive;

        // 2. Compile the regex.
        let re = Regex::new(final_pattern)
            .with_context(|| format!("Invalid regex pattern: '{}'", final_pattern))?;

        // NOTE: Legacy S3Ops logging call removed here (was: `let _ = build_ops_async()...`)
        // It caused all Python list() calls to hang on non-AWS endpoints (IMDSv2 timeout).
        // See docs/bugs/PYTHON_LIST_HANG_BUG_REPORT.md for full root cause analysis.

        let client = aws_s3_client_async().await?;
        let mut keys = Vec::new();
        let mut cont: Option<String> = None;

        // 3. Set the delimiter for the S3 API call based on the effective recursive flag.
        let delimiter = if effective_recursive { None } else { Some("/") };

        debug!(
            "list_objects: bucket='{}', prefix='{}', pattern='{}', recursive={}, effective_recursive={}",
            bucket, prefix, final_pattern, recursive, effective_recursive
        );

        loop {
            let mut req_builder = client.list_objects_v2().bucket(&bucket).prefix(prefix);

            if let Some(d) = delimiter {
                req_builder = req_builder.delimiter(d);
            }

            if let Some(token) = &cont {
                req_builder = req_builder.continuation_token(token);
                debug!("  loop: continuation_token={:?}", cont);
            }

            let resp = req_builder.send().await.context("list_objects_v2 failed")?;
            debug!(
                "  page received: {} contents, {} common prefixes",
                resp.contents().len(),
                resp.common_prefixes().len()
            );

            // 4. Filter and collect results based on the regex.

            // Handle objects (files)
            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    if let Some(basename) = key.strip_prefix(prefix) {
                        if re.is_match(basename) {
                            debug!("    matched key: '{}'", key);
                            keys.push(key.to_string());
                        }
                    }
                }
            }

            // Handle common prefixes (directories) selectively in non-recursive mode
            // Only process CommonPrefixes that represent single-character separators (like "/")
            // This handles the case where objects with leading slashes are treated as directories by S3
            if !effective_recursive {
                for common_prefix in resp.common_prefixes() {
                    if let Some(prefix_key) = common_prefix.prefix() {
                        if let Some(basename) = prefix_key.strip_prefix(prefix) {
                            // Only process single-character prefixes like "/" that might contain root-level objects
                            // Skip multi-character directory prefixes like "dir1/", "subdir/"
                            // Note: Don't check if pattern matches the prefix - check if objects under the prefix match
                            if basename.len() == 1 && basename == "/" {
                                debug!("    found single-slash common prefix, querying recursively: '{}'", prefix_key);
                                // Make one recursive call to get objects under the "/" prefix
                                // This should be safe since we're only going one level deep
                                if let Ok(client_async) = aws_s3_client_async().await {
                                    let recursive_req = client_async
                                        .list_objects_v2()
                                        .bucket(&bucket)
                                        .prefix(prefix_key);
                                    // No delimiter for recursive call

                                    if let Ok(recursive_resp) = recursive_req.send().await {
                                        for obj in recursive_resp.contents() {
                                            if let Some(key) = obj.key() {
                                                if let Some(obj_basename) = key.strip_prefix(prefix)
                                                {
                                                    if re.is_match(obj_basename) {
                                                        debug!("    matched object under slash prefix: '{}'", key);
                                                        keys.push(key.to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                debug!("    skipping multi-char common prefix: '{}'", prefix_key);
                            }
                        }
                    }
                }
            }

            if let Some(next) = resp.next_continuation_token() {
                cont = Some(next.to_string());
                debug!("  next_continuation_token='{}'", cont.as_ref().unwrap());
            } else {
                debug!("  no more pages, breaking");
                break;
            }
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

    // Determine the S3 prefix and regex pattern from the input path.
    // Do this eagerly (before returning the stream) so callers get parse errors immediately.
    let (prefix_str, pattern_str) = match path.rfind('/') {
        Some(index) => {
            let (p, pat) = path.split_at(index + 1);
            (p.to_string(), pat.to_string())
        }
        None => (String::new(), path.clone()),
    };
    let final_pattern = if pattern_str.is_empty() {
        ".*".to_string()
    } else {
        pattern_str
    };
    let re = Regex::new(&final_pattern)
        .with_context(|| format!("Invalid regex pattern '{}'", final_pattern))?;

    let delimiter = if recursive {
        None
    } else {
        Some("/".to_string())
    };

    debug!(
        "list_objects_stream: prefix='{}', pattern='{}', recursive={}, delimiter={:?}",
        prefix_str, final_pattern, recursive, delimiter
    );

    // Build the client eagerly too — any credential errors surface before the stream starts.
    let client = aws_s3_client_async().await?;

    // Return an inline pinned stream: no tokio::spawn, runs entirely on the caller's task.
    // This avoids tracing-subscriber deadlocks when RUST_LOG=debug/trace is set.
    Ok(Box::pin(async_stream::stream! {
        let mut cont: Option<String> = None;
        let mut page_count: u32 = 0;
        let mut total_keys: u64 = 0;

        loop {
            let mut req_builder = client.list_objects_v2().bucket(&bucket).prefix(prefix_str.as_str());

            if let Some(ref d) = delimiter {
                req_builder = req_builder.delimiter(d.as_str());
            }

            if let Some(ref token) = cont {
                req_builder = req_builder.continuation_token(token.as_str());
            }

            let resp = match req_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("list_objects_v2 failed: {}", e));
                    return;
                }
            };

            page_count += 1;
            debug!(
                "  page {}: {} contents, {} common prefixes",
                page_count, resp.contents().len(), resp.common_prefixes().len()
            );

            // Yield matching objects from this page.
            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    if let Some(basename) = key.strip_prefix(prefix_str.as_str()) {
                        if re.is_match(basename) {
                            yield Ok(key.to_string());
                        }
                    }
                }
            }

            // Handle common prefixes (non-recursive mode only).
            if !recursive {
                for common_prefix in resp.common_prefixes() {
                    if let Some(prefix_key) = common_prefix.prefix() {
                        if let Some(basename) = prefix_key.strip_prefix(prefix_str.as_str()) {
                            if basename.len() == 1 && basename == "/" {
                                let recursive_req = client
                                    .list_objects_v2()
                                    .bucket(&bucket)
                                    .prefix(prefix_key);
                                match recursive_req.send().await {
                                    Ok(recursive_resp) => {
                                        for obj in recursive_resp.contents() {
                                            if let Some(key) = obj.key() {
                                                if let Some(obj_basename) = key.strip_prefix(prefix_str.as_str()) {
                                                    if re.is_match(obj_basename) {
                                                        yield Ok(key.to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        yield Err(anyhow::anyhow!("recursive list_objects_v2 failed: {}", e));
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            total_keys += resp.contents().len() as u64;

            if let Some(next) = resp.next_continuation_token() {
                cont = Some(next.to_string());
            } else {
                debug!("  list_objects_stream done: {} pages, {} total keys streamed", page_count, total_keys);
                break;
            }
        }
    }))
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
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET range: no key specified");
    }
    debug!(
        "get_object_range: uri='{}', offset={}, length={:?}",
        uri, offset, length
    );
    let client = aws_s3_client_async().await?;
    let mut range = format!("bytes={}-", offset);
    if let Some(len) = length {
        if len > 0 {
            let end = offset.saturating_add(len).saturating_sub(1);
            range = format!("bytes={}-{}", offset, end);
        }
    }
    let resp = client
        .get_object()
        .bucket(&bucket)
        .key(key.trim_start_matches('/'))
        .range(range)
        .send()
        .await
        .context("get_object(range) failed")?;
    let body = resp.body.collect().await.context("collect range body")?;
    // Zero-copy: return Bytes directly
    let bytes = body.into_bytes();
    debug!(
        "get_object_range: uri='{}', returned {} bytes",
        uri,
        bytes.len()
    );
    Ok(bytes)
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
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET concurrent range: no key specified");
    }

    // Get object size first
    let client = aws_s3_client_async().await?;
    let head_resp = client
        .head_object()
        .bucket(&bucket)
        .key(key.trim_start_matches('/'))
        .send()
        .await
        .context("head_object failed for concurrent range GET")?;

    let object_size = head_resp.content_length().unwrap_or(0) as u64;

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
    let effective_chunk_size = chunk_size.unwrap_or_else(|| get_optimal_chunk_size(total_bytes));

    // Use single request for small objects
    if total_bytes <= effective_chunk_size as u64 {
        debug!("Using single range request for {} bytes", total_bytes);
        return get_object_range_uri_async(uri, start_offset, Some(total_bytes)).await;
    }

    // Use concurrent ranges for large objects
    let effective_concurrency =
        max_concurrency.unwrap_or_else(|| get_optimal_concurrency(total_bytes));

    debug!(
        "Using concurrent range GET: {} bytes, chunk_size={}, concurrency={}",
        total_bytes, effective_chunk_size, effective_concurrency
    );

    concurrent_range_get_impl(
        &client,
        &bucket,
        &key,
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
/// v0.9.31+: Replaced Arc<Mutex<BytesMut>> shared buffer with a collect-then-assemble
/// approach (Finding 5 fix). Each chunk future returns its (buffer_offset, data) pair
/// independently — no shared state, no lock contention. Chunks are sorted by offset
/// once at the end and assembled into a single BytesMut with one sequential pass.
/// This eliminates mutex serialisation across up to 37 concurrent writers per file.
async fn concurrent_range_get_impl(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    start_offset: u64,
    end_offset: u64,
    chunk_size: usize,
    max_concurrency: usize,
) -> Result<Bytes> {
    let total_bytes = (end_offset - start_offset) as usize;

    // Build (range_start, range_end, buffer_offset) tuples
    let mut ranges: Vec<(u64, u64, usize)> = Vec::new();
    let mut current_offset = start_offset;
    while current_offset < end_offset {
        let chunk_end = std::cmp::min(current_offset + chunk_size as u64, end_offset);
        let buffer_start = (current_offset - start_offset) as usize;
        ranges.push((current_offset, chunk_end, buffer_start));
        current_offset = chunk_end;
    }

    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    let mut futures = FuturesUnordered::new();

    for (range_start, range_end, buffer_offset) in ranges {
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
            let resp = client
                .get_object()
                .bucket(&bucket)
                .key(key.trim_start_matches('/'))
                .range(range_header)
                .send()
                .await
                .context("concurrent range request failed")?;

            let body = resp.body.collect().await.context("collect chunk body")?;
            let chunk_data = body.into_bytes();

            // Return (buffer_offset, data) — no shared mutex, no contention
            Ok::<(usize, Bytes), anyhow::Error>((buffer_offset, chunk_data))
        });
    }

    // Collect all (offset, chunk) pairs — order is non-deterministic (FuturesUnordered)
    let mut chunks: Vec<(usize, Bytes)> = Vec::new();
    while let Some(result) = futures.next().await {
        chunks.push(result.context("concurrent range chunk failed")?);
    }

    // Sort by buffer offset, then assemble with a single sequential pass
    chunks.sort_unstable_by_key(|(offset, _)| *offset);

    let mut output = BytesMut::with_capacity(total_bytes);
    for (_, chunk) in chunks {
        output.extend_from_slice(&chunk);
    }

    Ok(output.freeze())
}

/// Get optimal chunk size based on total transfer size
fn get_optimal_chunk_size(total_bytes: u64) -> usize {
    std::env::var("S3DLIO_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or({
            // Intelligent defaults based on object size
            if total_bytes < 16 * 1024 * 1024 {
                // < 16MB: Use 1MB chunks
                1024 * 1024
            } else if total_bytes < 256 * 1024 * 1024 {
                // 16MB - 256MB: Use 4MB chunks
                4 * 1024 * 1024
            } else {
                // > 256MB: Use 8MB chunks for optimal throughput
                8 * 1024 * 1024
            }
        })
}

/// Get optimal concurrency based on total transfer size
fn get_optimal_concurrency(total_bytes: u64) -> usize {
    std::env::var("S3DLIO_RANGE_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            let cores = num_cpus::get();
            if total_bytes < 32 * 1024 * 1024 {
                // < 32MB: Limited concurrency
                std::cmp::max(2, cores / 2)
            } else if total_bytes < 512 * 1024 * 1024 {
                // 32MB - 512MB: Moderate concurrency
                std::cmp::max(4, cores)
            } else {
                // > 512MB: High concurrency for large transfers
                std::cmp::max(8, cores * 2)
            }
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
    let key_clean = key.trim_start_matches('/');

    debug!("stat_object: bucket='{}', key='{}'", bucket, key_clean);

    let client = aws_s3_client_async().await?;
    let resp = client
        .head_object()
        .bucket(bucket)
        .key(key.trim_start_matches('/'))
        .send()
        .await
        .context("head_object failed")?;

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
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot HEAD: no key specified");
    }
    stat_object(&bucket, &key).await
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
pub async fn stat_object_many_async(uris: Vec<String>) -> Result<Vec<ObjectStat>> {
    use futures_util::future::try_join_all;
    let futs = uris
        .into_iter()
        .map(|u| async move { stat_object_uri_async(&u).await });
    try_join_all(futs).await
}

// -----------------------------------------------------------------------------
// Delete operation
// -----------------------------------------------------------------------------

/// Delete many keys (up to 1000 per logged batch).
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
        client
            .delete_objects()
            .bucket(bucket)
            .delete(delete)
            .send()
            .await
            .with_context(|| format!("delete_objects_async failed for bucket '{}'", bucket))?;
    }
    Ok(())
}

/// Async version of delete_bucket — safe to call from within an async context.
pub async fn delete_bucket_async(bucket: &str) -> Result<()> {
    let client = aws_s3_client_async().await?;
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

/// Download a single object from S3 **with op‑logging**.
pub async fn get_object(bucket: &str, key: &str) -> Result<Bytes> {
    // delegate to S3Ops (which records the GET) and propagate errors
    let data = build_ops_async().await?.get_object(bucket, key).await?;

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
        if get_range_opt_enabled() {
            let stat_sem = Arc::new(Semaphore::new(max_in_flight));
            let stat_futs: Vec<_> = uris
                .iter()
                .map(|uri| {
                    let uri = uri.clone();
                    let stat_sem = stat_sem.clone();
                    async move {
                        let _permit = stat_sem.acquire().await.ok();
                        // Only stat if not already cached (subsequent epochs skip this)
                        if get_size_cache().get(&uri).await.is_none() {
                            if let Ok(stat) = stat_object_uri_async(&uri).await {
                                get_size_cache().put(uri, stat.size).await;
                            }
                        }
                    }
                })
                .collect();
            futures::future::join_all(stat_futs).await;
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
        while let Some(res) = futs.next().await {
            out.push(res??);
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
        while let Some(res) = futs.next().await {
            out.push(res??);
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
#[cfg_attr(feature = "profiling", instrument(
    name = "s3.get_optimized",
    skip(uri),
    fields(uri = %uri)
))]
pub async fn get_object_uri_optimized_async(uri: &str) -> Result<Bytes> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: no key specified");
    }

    // Fast path: range optimisation disabled — skip HEAD entirely on ALL paths.
    // Fixes the bug where S3DLIO_ENABLE_RANGE_OPTIMIZATION=0 was a no-op here.
    if !get_range_opt_enabled() {
        debug!("range_opt disabled: single GET for {}", uri);
        return get_object(&bucket, &key).await;
    }

    let range_threshold = get_range_threshold_bytes();

    // Check ObjectSizeCache before issuing HEAD (avoids a network round-trip on cache hit).
    // The cache is populated by this function and by the pre-stat phase in get_objects_parallel().
    let object_size = if let Some(sz) = get_size_cache().get(uri).await {
        sz
    } else {
        // Cache miss: issue one HEAD to learn the size, then cache it.
        let client = aws_s3_client_async().await?;
        match client
            .head_object()
            .bucket(&bucket)
            .key(key.trim_start_matches('/'))
            .send()
            .await
        {
            Ok(resp) => {
                let sz = resp.content_length().unwrap_or(0) as u64;
                get_size_cache().put(uri.to_string(), sz).await;
                sz
            }
            Err(_) => {
                // HEAD failed: fall back to single GET without range splitting.
                return get_object(&bucket, &key).await;
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
        get_object(&bucket, &key).await
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
    let start_offset = offset;
    let end_offset = match length {
        Some(len) => std::cmp::min(start_offset + len, object_size),
        None => object_size,
    };
    if start_offset >= object_size {
        return Ok(Bytes::new());
    }

    let total_bytes = end_offset - start_offset;
    let effective_chunk_size = chunk_size.unwrap_or_else(|| get_optimal_chunk_size(total_bytes));

    // If total fits in one chunk, use a single range request (no concurrency overhead).
    if total_bytes <= effective_chunk_size as u64 {
        let uri = format!("s3://{}/{}", bucket, key.trim_start_matches('/'));
        return get_object_range_uri_async(&uri, start_offset, Some(total_bytes)).await;
    }

    let effective_concurrency =
        max_concurrency.unwrap_or_else(|| get_optimal_concurrency(total_bytes));
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

/// Public async get object call
pub async fn get_object_uri_async(uri: &str) -> Result<Bytes> {
    debug!("get_object_uri_async: uri='{}'", uri);
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: no key specified");
    }
    get_object(&bucket, &key).await
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

/// Upload a single object to S3 **with op‑logging**.
///
/// Accepts `Bytes` for zero-copy efficiency (no .to_vec() call).
pub async fn put_object_async(bucket: &str, key: &str, data: Bytes) -> Result<()> {
    debug!("S3 PUT s3://{}/{} ({} bytes)", bucket, key, data.len());
    // delegate to S3Ops (which records the PUT) - data moved, no copy
    build_ops_async()
        .await?
        .put_object(bucket, key, data)
        .await?;
    Ok(())
}

/// Upload a single object to S3 by URI **with op‑logging**.
///
/// Accepts `Bytes` for zero-copy efficiency.
pub async fn put_object_uri_async(uri: &str, data: Bytes) -> Result<()> {
    debug!("put_object_uri_async: uri='{}', {} bytes", uri, data.len());
    let (bucket, key) = parse_s3_uri(uri)?;
    put_object_async(&bucket, &key, data).await
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
    use crate::multipart::{MultipartUploadConfig, MultipartUploadSink};
    let uri_owned = uri.to_string();
    let cfg = MultipartUploadConfig {
        part_size: part_size.unwrap_or(16 * 1024 * 1024), // 16 MiB default
        ..Default::default()
    };
    // Use spawn_blocking so that blocking_send() and run_on_global_rt() calls
    // inside write_blocking()/finish_blocking() never park a Tokio worker thread.
    // Without this, objects larger than max_in_flight × part_size panic because
    // blocking_send() on a full channel would stall a worker.
    tokio::task::spawn_blocking(move || {
        let mut sink = MultipartUploadSink::from_uri(&uri_owned, cfg)?;
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

        while let Some(res) = futs.next().await {
            res??;
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

    #[test]
    fn test_parse_s3_uri_error_no_slash() {
        // Missing slash after bucket
        let result = parse_s3_uri_full("s3://mybucket");
        assert!(result.is_err());
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
}
