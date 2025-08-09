// src/s3_utils.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides high‑level S3 operations: list, get, delete, typed PUT.


use anyhow::{bail, Context, Result};
use aws_sdk_s3::error::ProvideErrorMetadata;
use futures::{stream::FuturesUnordered, StreamExt};
use pyo3::{FromPyObject, PyAny, PyResult};
use pyo3::types::PyAnyMethods;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Semaphore;
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
// List operation
// -----------------------------------------------------------------------------
/// List every key under `prefix` (handles pagination).
pub fn list_objects(bucket: &str, prefix: &str) -> Result<Vec<String>> {
    let bucket = bucket.to_string();
    let prefix = prefix.to_string();
    run_on_global_rt(async move {
        // log one LIST
        let _ = build_ops_async().await?.list_objects(&bucket, &prefix).await;
        let client = aws_s3_client_async().await?;   // reuse raw client for pagination

        let mut keys = Vec::new();
        let mut cont: Option<String> = None;

        debug!("list_objects: bucket='{}' prefix='{}'", bucket, prefix);

        loop {
            debug!("  loop: continuation_token={:?}", cont);
            let mut req = client.list_objects_v2().bucket(&bucket).prefix(&prefix);
            if let Some(token) = &cont {
                req = req.continuation_token(token);
                debug!("    attached continuation_token='{}'", token);
            }
            let resp = req.send().await.context("list_objects_v2 failed")?;
            debug!("    page received: {} contents", resp.contents().len());

            for obj in resp.contents() {
                if let Some(k) = obj.key() {
                    debug!("      found key: '{}'", k);
                    keys.push(k.to_string());
                }
            }
            if let Some(next) = resp.next_continuation_token() {
                cont = Some(next.to_string());
                debug!("    next_continuation_token='{}'", cont.as_ref().unwrap());
            } else {
                debug!("    no more pages, breaking");
                break;
            }
        }
        debug!("list_objects result: {} total keys", keys.len());
        Ok(keys)
    })
}


/// Read `length` bytes starting at `offset` from `s3://bucket/key`.
/// A negative or oversize request is truncated to the object size.
/// Uses the existing `get_object_uri()` helper internally so we don’t
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
                let data = get_object_uri_async(&uri).await?;
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

