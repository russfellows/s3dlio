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

use log::{info, debug};

// data generation helpers
use crate::data_gen::{generate_object};
use crate::config::Config;

// S3 client creation
use crate::s3_client::{aws_s3_client, block_on};

// S3 operation logging
use crate::s3_logger::global_logger;
use crate::s3_ops::S3Ops;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_OBJECT_SIZE: usize = 20 * 1024 * 1024;


// ─────────────────────────────────────────────────────────────────────────────
// helper: create an S3Ops wired to the singleton logger (if any)
fn build_ops() -> Result<S3Ops> {
    let client = aws_s3_client()?;
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

/*
 * Old definition with pyo3 v0.20
 *
impl<'source> FromPyObject<'source> for ObjectType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s = ob.extract::<&str>()?;
        Ok(ObjectType::from(s))
    }
}
*/

use crate::config::ObjectType;

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
    block_on(async {
        let client = aws_s3_client()?;
        match client.create_bucket().bucket(bucket).send().await {
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
    block_on(async {
        // Old, pre logger
//        let client = aws_s3_client()?;

        // Fire one lightweight LIST to get the op into the log.
        let _ = build_ops()?.list_objects(bucket, prefix).await;

        let client = aws_s3_client()?;   // reuse raw client for pagination
        let mut keys = Vec::new();
        let mut cont: Option<String> = None;

        // If debug show what we have
        debug!("list_objects: bucket='{}' prefix='{}'", bucket, prefix);

        loop {
            let mut _req = client.list_objects_v2().bucket(bucket).prefix(prefix);

            debug!("  loop: continuation_token={:?}", cont);
            let req = client
                .list_objects_v2()
                .bucket(bucket)
                .prefix(prefix);


            if let Some(token) = &cont {
                _req = req.continuation_token(token);
                debug!("    attached continuation_token='{}'", token);
            }
            let resp = _req.send().await.context("list_objects_v2 failed")?;
            debug!("    page received: {} contents", resp.contents().len());
            
            // iterating over objects in the response
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

// -----------------------------------------------------------------------------
// Stat (HEAD) operation
// -----------------------------------------------------------------------------
/// Perform HEAD-object and return **all** common metadata. 
pub fn stat_object(bucket: &str, prefix: &str, key: &str) -> Result<ObjectStat> {

    // Clean off any leading slash from the incoming key
    let key_clean = key.trim_start_matches('/');
    // Trim slashes off prefix, then join prefix and key with exactly one "/"
    let prefix_clean = prefix.trim_matches('/');
    let full_key = if prefix_clean.is_empty() {
        key_clean.to_string()
    } else {
        format!("{}/{}", prefix_clean, key_clean)
    };
    debug!(
        "stat_object → bucket='{}', prefix='{}', key='{}', full_key='{}'",
        bucket, prefix, key, full_key
    );

    block_on(async {
        // Old pre logger
        //let client = aws_s3_client()?;
        //
        // New, with logger
        // Log the HEAD request
        let _ = build_ops()?.stat_object(bucket, &full_key).await;

        let client = aws_s3_client()?;
        let resp = client.head_object()
            .bucket(bucket)
            .key(&full_key)
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
            expires:                resp.expires_string().map(|s| s.to_string()),
            storage_class:          resp.storage_class().map(|s| s.as_str().to_string()),
            server_side_encryption: resp.server_side_encryption().map(|s| s.as_str().to_string()),
            ssekms_key_id:          resp.ssekms_key_id().map(|s| s.to_string()),
            sse_customer_algorithm: resp.sse_customer_algorithm().map(|s| s.to_string()),
            version_id:             resp.version_id().map(|s| s.to_string()),
            replication_status:     resp.replication_status().map(|s| s.as_str().to_string()),
            metadata:               resp.metadata().cloned().unwrap_or_default(),
        })
    })
}



/// Convenience wrapper that accepts a full `s3://bucket/key` URI.
pub fn stat_object_uri(uri: &str) -> Result<ObjectStat> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot STAT: no key specified (use full key)");
    }
    //stat_object(&bucket, &key)
    // If no prefix, pass empty string for now
    stat_object(&bucket, "", &key)
}

// -----------------------------------------------------------------------------
// Delete operation
// -----------------------------------------------------------------------------
/*
 * Old - pre logger
 *
/// Delete many keys (up to 1000 per call).
pub fn delete_objects(bucket: &str, keys: &[String]) -> Result<()> {
    block_on(async {
        let client = aws_s3_client()?;
        for chunk in keys.chunks(1000) {
            let objs: Vec<ObjectIdentifier> = chunk.iter()
                .map(|k| ObjectIdentifier::builder().key(k).build().map_err(anyhow::Error::from))
                .collect::<Result<_>>()?;
            let delete = Delete::builder().set_objects(Some(objs)).build().map_err(anyhow::Error::from)?;
            client.delete_objects().bucket(bucket).delete(delete).send().await?;
        }
        Ok(())
    })
}
*/

/// Delete many keys (up to 1000 per logged batch).
pub fn delete_objects(bucket: &str, keys: &[String]) -> Result<()> {
    block_on(async {
        let ops = build_ops()?;
        for chunk in keys.chunks(1000) {
            ops.delete_objects(bucket, chunk.to_vec()).await?;
        }
        Ok(())
    })
}

// -----------------------------------------------------------------------------
// Download (GET) operations
// -----------------------------------------------------------------------------
/*
 * Old, pre logger
 *
/// Download a single object from S3 (bucket/key).
pub async fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>> {
    let client = aws_s3_client()?;
    let resp = client.get_object().bucket(bucket).key(key)
        .send().await.context("get_object failed")?;
    let data = resp.body.collect().await.context("collect body failed")?.into_bytes();

    info!("S3 GET s3://{}/{} ", bucket, key);
    Ok(data.to_vec())
}
*/

/// Download a single object from S3 **with op‑logging**.
pub async fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>> {
    // delegate to S3Ops (which records the GET) and propagate errors
    let data = build_ops()?.get_object(bucket, key).await?;
    
    // Emit an info‑level message
    info!("S3 GET s3://{}/{}", bucket, key);
    Ok(data)
}

/// Download a single object by URI.
pub fn get_object_uri(uri: &str) -> Result<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: no key specified (use full key)");
    }
    block_on(get_object(&bucket, &key))
}

/// Download many objects concurrently (ordered by input).
pub fn get_objects_parallel(
    uris: &[String], max_in_flight: usize,
) -> Result<Vec<(String, Vec<u8>)>> {
    block_on(async {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();
        for uri in uris.iter().cloned() {
            let sem = sem.clone();
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
        out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap());
        Ok(out)
    })
}

async fn get_object_uri_async(uri: &str) -> Result<Vec<u8>> {
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
    /* build a tiny config just for data generation */

    info!(
        "put_objects: type={:?}, size={} bytes, uris={} parallelism={}",
        object_type,
        size,
        uris.len(),
        max_in_flight
    );

    let cfg = Config {
        object_type,
        elements: 1,                // 1 record
        element_size: size,         // whole buffer is the record
        use_controlled: dedup_factor != 1 || compress_factor != 1,  // We use controlled data if
                                                                    // dedup or compress are set 
        dedup_factor,
        compress_factor,
    };

    let buffer = generate_object(&cfg)?;
    debug!("Uploading buffer of {} bytes to {} URIs", buffer.len(), uris.len());
    put_objects_parallel(&uris, &buffer, max_in_flight)
}

/*
 * Old, pre logger
 *
/// Upload an object
pub async fn put_object_async(bucket: &str, key: &str, data: &[u8]) -> Result<()> {
    let client = aws_s3_client()?;
    let body = ByteStream::from(data.to_vec());
    info!("S3 PUT s3://{}/{} ({} bytes)", bucket, key, data.len());

    client.put_object().bucket(bucket).key(key).body(body).send().await?;
    Ok(())
}
*/

/// Upload a single object to S3 **with op‑logging**.
pub async fn put_object_async(bucket: &str, key: &str, data: &[u8]) -> Result<()> {
    // delegate to S3Ops (which records the PUT) and propagate errors
    build_ops()?.put_object(bucket, key, data.to_vec()).await?;
    
    // Emit an info‑level message if you like
    info!("S3 PUT s3://{}/{} ({} bytes)", bucket, key, data.len());
    Ok(())
}

// -----------------------------------------------------------------------------
// Internal helpers (private)
// -----------------------------------------------------------------------------

pub(crate) fn put_objects_parallel(uris: &[String], data: &[u8], max_in_flight: usize) -> Result<()> {
    block_on(async {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();
        for uri in uris.iter().cloned() {
            let sem = sem.clone();
            let data = data.to_vec();
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let (b, k) = parse_s3_uri(&uri)?;
                put_object_async(&b, &k, &data).await
            }));
        }
        while let Some(res) = futs.next().await {
            res??;
        }
        Ok(())
    })
}

