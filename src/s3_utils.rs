// src/s3_utils.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides high‑level S3 operations: list, get, delete, typed PUT.


use anyhow::{bail, Context, Result};
use aws_sdk_s3::error::ProvideErrorMetadata;
//use aws_sdk_s3::primitives::ByteStream;
use futures::{stream::FuturesUnordered, StreamExt};
#[cfg(feature = "extension-module")]
use pyo3::{FromPyObject, PyAny, PyResult};
#[cfg(feature = "extension-module")]
use pyo3::types::PyAnyMethods;
use std::sync::Arc;
use std::collections::HashMap;
use regex::Regex;
use tokio::sync::Semaphore;

#[cfg(feature = "profiling")]
use tracing::instrument;
use bytes::Bytes;

use log::{info, debug};

// data generation helpers
use crate::data_gen::{generate_object};
use crate::config::Config;
use crate::config::ObjectType;

// S3 client creation
use crate::s3_client::{aws_s3_client_async, run_on_global_rt};

// S3 operation logging
use crate::s3_logger::global_logger;
use crate::s3_ops::S3Ops;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_OBJECT_SIZE: usize = 20 * 1024 * 1024;


// ─────────────────────────────────────────────────────────────────────────────
// helper: create an S3Ops wired to the singleton logger (if any)
async fn build_ops_async() -> Result<S3Ops> {
    let client = aws_s3_client_async().await?;
    let logger = global_logger();
    let client_id = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
    let endpoint  = std::env::var("AWS_ENDPOINT_URL").unwrap_or_default();
    Ok(S3Ops::new(client, logger, &client_id, &endpoint))
}

// -----------------------------------------------------------------------------
// ObjectStat struct for stat operation 
// -----------------------------------------------------------------------------
/// Full set of common S3 metadata from a HEAD-object call.
#[derive(Debug)]
pub struct ObjectStat {
    pub size:                     u64,
    pub last_modified:            Option<String>,
    pub e_tag:                    Option<String>,
    pub content_type:             Option<String>,
    pub content_language:         Option<String>,
    pub content_encoding:         Option<String>,
    pub cache_control:            Option<String>,
    pub content_disposition:      Option<String>,
    pub expires:                  Option<String>,
    pub storage_class:            Option<String>,
    pub server_side_encryption:   Option<String>,
    pub ssekms_key_id:            Option<String>,
    pub sse_customer_algorithm:   Option<String>,
    pub version_id:               Option<String>,
    pub replication_status:       Option<String>,
    pub metadata:                 HashMap<String, String>,
}

// -----------------------------------------------------------------------------
// ObjectType enum for typed data generation
// Note: the defintion moved
// -----------------------------------------------------------------------------
impl From<&str> for ObjectType {
    fn from(s: &str) -> Self {
        match s.to_ascii_uppercase().as_str() {
            "NPZ"      => ObjectType::Npz,
            "TFRECORD" => ObjectType::TfRecord,
            "HDF5"     => ObjectType::Hdf5,
            _           => ObjectType::Raw,
        }
    }
}


// New pyo3 version 0.25 API
#[cfg(feature = "extension-module")]
impl<'source> FromPyObject<'source> for ObjectType {
    fn extract_bound(ob: &pyo3::Bound<'source, PyAny>) -> PyResult<Self> {
        let s = ob.extract::<&str>()?;
        Ok(ObjectType::from(s))
    }
}


// -----------------------------------------------------------------------------
// S3 URI helpers
// -----------------------------------------------------------------------------
/// Split `s3://bucket/key` → (`bucket`, `key`).
/// `key` may be empty (prefix).
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let trimmed = uri.strip_prefix("s3://").context("URI must start with s3://")?;
    let (bucket, key) = trimmed.split_once('/').context("URI must contain '/' after bucket")?;
    Ok((bucket.to_string(), key.to_string()))
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
/// List all S3 buckets in the account.
/// Returns a vector of bucket names with their creation dates.
pub fn list_buckets() -> Result<Vec<BucketInfo>> {
    run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        let response = client
            .list_buckets()
            .send()
            .await
            .context("Failed to list S3 buckets")?;

        let mut buckets = Vec::new();
        let bucket_list = response.buckets();
        for bucket in bucket_list {
            let name = bucket.name().unwrap_or("Unknown").to_string();
            let creation_date = bucket.creation_date()
                .map(|dt| dt.to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            
            buckets.push(BucketInfo {
                name,
                creation_date,
            });
        }
        
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
pub fn list_objects(bucket: &str, path: &str, recursive: bool) -> Result<Vec<String>> {
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

        // 2. Compile the regex.
        let re = Regex::new(final_pattern)
            .with_context(|| format!("Invalid regex pattern: '{}'", final_pattern))?;

        // This logging operation can be adapted or removed, but here's how to keep it:
        let _ = build_ops_async().await?.list_objects(&bucket, prefix).await;

        let client = aws_s3_client_async().await?;
        let mut keys = Vec::new();
        let mut cont: Option<String> = None;

        // 3. Set the delimiter for the S3 API call based on the recursive flag.
        let delimiter = if recursive { None } else { Some("/") };

        debug!(
            "list_objects: bucket='{}', prefix='{}', pattern='{}', recursive={}",
            bucket, prefix, final_pattern, recursive
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

            let resp = req_builder
                .send()
                .await
                .context("list_objects_v2 failed")?;
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

            // Handle common prefixes (directories), only present in non-recursive mode
            for common_prefix in resp.common_prefixes() {
                if let Some(prefix_key) = common_prefix.prefix() {
                    if let Some(basename) = prefix_key.strip_prefix(prefix) {
                        if re.is_match(basename) {
                            debug!("    matched common prefix: '{}'", prefix_key);
                            keys.push(prefix_key.to_string());
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
pub async fn get_object_range_uri_async(uri: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() { bail!("Cannot GET range: no key specified"); }
    let client = aws_s3_client_async().await?;
    let mut range = format!("bytes={}-", offset);
    if let Some(len) = length {
        if len > 0 {
            let end = offset.saturating_add(len).saturating_sub(1);
            range = format!("bytes={}-{}", offset, end);
        }
    }
    let resp = client.get_object()
        .bucket(&bucket)
        .key(key.trim_start_matches('/'))
        .range(range)
        .send()
        .await
        .context("get_object(range) failed")?;
    let body = resp.body.collect().await.context("collect range body")?;
    Ok(body.into_bytes().to_vec())
}

/// Read `length` bytes starting at `offset` from `s3://bucket/key`.
/// A negative or oversize request is truncated to the object size.
/// Uses the existing `get_object_uri()` helper internally so we don't
/// duplicate S3-client code right now.
pub fn get_range(
    bucket: &str,
    key: &str,
    offset: u64,
    length: Option<u64>,
) -> Result<Vec<u8>> {
    let uri = format!("s3://{bucket}/{key}");
    let bytes = get_object_uri(&uri)?;
    let start = offset as usize;
    let end = match length {
        Some(len) => start.saturating_add(len as usize).min(bytes.len()),
        None => bytes.len(),
    };
    if start >= bytes.len() {
        Ok(Vec::new())
    } else {
        Ok(bytes[start..end].to_vec())
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
) -> Result<Vec<u8>> {
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
        return Ok(Vec::new());
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
    let effective_concurrency = max_concurrency.unwrap_or_else(|| get_optimal_concurrency(total_bytes));
    
    debug!(
        "Using concurrent range GET: {} bytes, chunk_size={}, concurrency={}",
        total_bytes, effective_chunk_size, effective_concurrency
    );

    concurrent_range_get_impl(&client, &bucket, &key, start_offset, end_offset, effective_chunk_size, effective_concurrency).await
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
async fn concurrent_range_get_impl(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    start_offset: u64,
    end_offset: u64,
    chunk_size: usize,
    max_concurrency: usize,
) -> Result<Vec<u8>> {
    let total_bytes = end_offset - start_offset;
    
    // Calculate chunk ranges
    let mut ranges = Vec::new();
    let mut current_offset = start_offset;
    
    while current_offset < end_offset {
        let chunk_end = std::cmp::min(current_offset + chunk_size as u64, end_offset);
        let buffer_start = (current_offset - start_offset) as usize;
        ranges.push((current_offset, chunk_end, buffer_start));
        current_offset = chunk_end;
    }
    
    // Create semaphore for concurrency control
    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    
    // Use Arc<Mutex<Vec<u8>>> to store results
    let result = Arc::new(std::sync::Mutex::new(vec![0u8; total_bytes as usize]));
    
    // Execute concurrent range requests
    let mut futures = FuturesUnordered::new();
    
    for (range_start, range_end, buffer_offset) in ranges {
        let client = client.clone();
        let bucket = bucket.to_string();
        let key = key.to_string();
        let semaphore = semaphore.clone();
        let result = result.clone();
        
        let future = async move {
            let _permit = semaphore.acquire().await.map_err(|e| anyhow::anyhow!("Semaphore error: {}", e))?;
            
            let range_header = format!("bytes={}-{}", range_start, range_end - 1);
            let resp = client
                .get_object()
                .bucket(&bucket)
                .key(key.trim_start_matches('/'))
                .range(range_header)
                .send()
                .await
                .context("concurrent range request failed")?;
            
            let body = resp.body.collect().await.context("collect concurrent range body")?;
            let chunk_data = body.into_bytes();
            
            // Write to shared buffer
            {
                let mut result_guard = result.lock().map_err(|e| anyhow::anyhow!("Mutex lock error: {}", e))?;
                let end_pos = buffer_offset + chunk_data.len();
                result_guard[buffer_offset..end_pos].copy_from_slice(&chunk_data);
            }
            
            Ok::<_, anyhow::Error>(())
        };
        
        futures.push(future);
    }
    
    // Wait for all requests to complete
    while let Some(result_future) = futures.next().await {
        result_future.context("concurrent range request failed")?;
    }
    
    // Extract final result
    let final_result = Arc::try_unwrap(result)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap result Arc"))?
        .into_inner()
        .map_err(|e| anyhow::anyhow!("Mutex poison error: {}", e))?;
    
    Ok(final_result)
}

/// Get optimal chunk size based on total transfer size
fn get_optimal_chunk_size(total_bytes: u64) -> usize {
    std::env::var("S3DLIO_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
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
) -> Result<Vec<u8>> {
    let uri_owned = uri.to_string(); // Clone the URI to avoid lifetime issues
    run_on_global_rt(async move {
        get_object_concurrent_range_async(&uri_owned, offset, length, chunk_size, max_concurrency).await
    })
}

// -----------------------------------------------------------------------------
// Stat (HEAD) operation
// -----------------------------------------------------------------------------

/// Perform HEAD-object and return all common metadata (async).
pub async fn stat_object(bucket: &str, key: &str) -> Result<ObjectStat> {

    // optional: log the HEAD via S3Ops (does not return the metadata)
    let key_clean = key.trim_start_matches('/');
    let _ = build_ops_async().await?.stat_object(bucket, key_clean).await;

    debug!(
        "stat_object → bucket='{}', key='{}', key_clean='{}'",
        bucket, key, key_clean
    );

    let client = aws_s3_client_async().await?;
    let resp = client.head_object()
        .bucket(bucket)
        .key(key.trim_start_matches('/'))
        .send()
        .await
        .context("head_object failed")?;

    Ok(ObjectStat {
        size:                   resp.content_length().unwrap_or_default() as u64,
        last_modified:          resp.last_modified().map(|t| t.to_string()),
        e_tag:                  resp.e_tag().map(|s| s.to_string()),
        content_type:           resp.content_type().map(|s| s.to_string()),
        content_language:       resp.content_language().map(|s| s.to_string()),
        content_encoding:       resp.content_encoding().map(|s| s.to_string()),
        cache_control:          resp.cache_control().map(|s| s.to_string()),
        content_disposition:    resp.content_disposition().map(|s| s.to_string()),
        // if your SDK has `expires()` instead of `expires_string()`, swap accordingly
        expires:                resp.expires_string().map(|s| s.to_string()),
        storage_class:          resp.storage_class().map(|s| s.as_str().to_string()),
        server_side_encryption: resp.server_side_encryption().map(|s| s.as_str().to_string()),
        ssekms_key_id:          resp.ssekms_key_id().map(|s| s.to_string()),
        sse_customer_algorithm: resp.sse_customer_algorithm().map(|s| s.to_string()),
        version_id:             resp.version_id().map(|s| s.to_string()),
        replication_status:     resp.replication_status().map(|s| s.as_str().to_string()),
        metadata:               resp.metadata().cloned().unwrap_or_default(),
    })
}

/// Convenience wrapper that accepts a full `s3://bucket/key` URI.
///
/// Async HEAD by full URI (non-blocking; safe inside Tokio)
pub async fn stat_object_uri_async(uri: &str) -> Result<ObjectStat> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() { bail!("Cannot HEAD: no key specified"); }
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
    let futs = uris.into_iter().map(|u| async move { stat_object_uri_async(&u).await });
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



// -----------------------------------------------------------------------------
// Download (GET) operations
// -----------------------------------------------------------------------------

/// Download a single object from S3 **with op‑logging**.
pub async fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>> {
    // delegate to S3Ops (which records the GET) and propagate errors
    let data = build_ops_async().await?.get_object(bucket, key).await?;
    
    // Emit an info‑level message
    info!("S3 GET s3://{}/{}", bucket, key);
    Ok(data)
}

/// Download a single object by URI, uses helper
pub fn get_object_uri(uri: &str) -> anyhow::Result<Vec<u8>> {
    let uri = uri.to_string();
    run_on_global_rt(async move { get_object_uri_async(&uri).await })
}

/// Download many objects concurrently (ordered by input).
pub fn get_objects_parallel(
    uris: &[String],
    max_in_flight: usize,
) -> Result<Vec<(String, Vec<u8>)>> {
    let uris = uris.to_vec(); // own for 'static
    run_on_global_rt(async move {
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
        // keep input order
        out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap());
        Ok(out)
    })
}


/// Optimized async get object call that uses concurrent range downloads for larger objects
#[cfg_attr(feature = "profiling", instrument(
    name = "s3.get_optimized",
    skip(uri),
    fields(uri = %uri)
))]
pub async fn get_object_uri_optimized_async(uri: &str) -> Result<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: no key specified");
    }
    
    // Use environment variable to control range optimization threshold
    let range_threshold = std::env::var("S3DLIO_RANGE_THRESHOLD_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(4) * 1024 * 1024; // Default 4MB threshold
    
    // Get object size first to decide strategy
    let client = aws_s3_client_async().await?;
    let head_resp = client
        .head_object()
        .bucket(&bucket)
        .key(key.trim_start_matches('/'))
        .send()
        .await;
        
    match head_resp {
        Ok(resp) => {
            let object_size = resp.content_length().unwrap_or(0) as u64;
            
            // For objects larger than threshold, use concurrent range downloads
            if object_size >= range_threshold {
                debug!("Using concurrent range download for {}MB object: {}", 
                       object_size / (1024 * 1024), uri);
                get_object_concurrent_range_async(uri, 0, None, None, None).await
            } else {
                // For smaller objects, use regular download
                get_object(&bucket, &key).await
            }
        },
        Err(_) => {
            // If HEAD fails, fall back to regular download
            get_object(&bucket, &key).await
        }
    }
}


/// Public async get object call
pub async fn get_object_uri_async(uri: &str) -> Result<Vec<u8>> {
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
    object_type: ObjectType,
    dedup_factor: usize,
    compress_factor: usize,
) -> Result<()> {
    info!(
        "put_objects: type={:?}, size={} bytes, uris={} parallelism={}",
        object_type, size, uris.len(), max_in_flight
    );

    let cfg = Config {
        object_type,
        elements: 1,
        element_size: size,
        use_controlled: dedup_factor != 1 || compress_factor != 1,
        dedup_factor,
        compress_factor,
    };

    let buffer: Bytes = generate_object(&cfg)?.into();  // or: Bytes::from(generate_object(&cfg)?)
    let uris_vec = uris.to_vec();                     // own for 'static
    debug!("Uploading buffer of {} bytes to {} URIs", buffer.len(), uris_vec.len());
    put_objects_parallel(uris_vec, buffer, max_in_flight)
}



/// Upload a single object to S3 **with op‑logging**.
pub async fn put_object_async(bucket: &str, key: &str, data: &[u8]) -> Result<()> {
    // delegate to S3Ops (which records the PUT) and propagate errors
    build_ops_async().await?.put_object(bucket, key, data.to_vec()).await?;
    
    // Emit an info‑level message 
    info!("S3 PUT s3://{}/{} ({} bytes)", bucket, key, data.len());
    Ok(())
}

/// Upload a single object to S3 by URI **with op‑logging**.
pub async fn put_object_uri_async(uri: &str, data: &[u8]) -> Result<()> {
    let (bucket, key) = parse_s3_uri(uri)?;
    put_object_async(&bucket, &key, data).await
}

/// Upload object via multipart using existing MultipartUploadSink infrastructure.
pub async fn put_object_multipart_uri_async(
    uri: &str,
    data: &[u8],
    part_size: Option<usize>,
) -> Result<()> {
    use crate::multipart::{MultipartUploadConfig, MultipartUploadSink};
    let cfg = MultipartUploadConfig {
        part_size: part_size.unwrap_or(16 * 1024 * 1024), // 16 MiB default
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::from_uri(uri, cfg)?;
    sink.write_blocking(data)?;
    sink.finish_blocking()?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Internal helpers (private)
// -----------------------------------------------------------------------------
//

pub(crate) fn put_objects_parallel(
    uris: Vec<String>,
    data: Bytes,                    // ref-counted, cheap to clone
    max_in_flight: usize,
) -> anyhow::Result<()> {
    run_on_global_rt(async move {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for uri in uris.into_iter() {
            let sem = Arc::clone(&sem);
            let payload = data.clone();           // zero-copy clone

            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let (b, k) = parse_s3_uri(&uri)?;
                // &[u8] view over Bytes — no copy here
                put_object_async(&b, &k, payload.as_ref()).await
            }));
        }

        while let Some(res) = futs.next().await {
            res??;
        }
        Ok(())
    })
}

