//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
// src/s3_utils.rs
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides helpers used by both the CLI and the forthcoming PyO3 bindings.

use anyhow::{bail, Context, Result};
use clap::ValueEnum;

use aws_config::meta::region::RegionProviderChain;
//use aws_sdk_s3::{config::Region, types::{Delete, ObjectIdentifier}, Client};
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::{config::Region, Client};
use aws_sdk_s3::primitives::ByteStream;

use futures::{stream::FuturesUnordered, StreamExt};
use once_cell::sync::{Lazy, OnceCell};
use std::{env, sync::Arc};
use tokio::{runtime::Handle, sync::Semaphore, task};

use pyo3::{PyAny, FromPyObject, PyResult};

use rand::Rng;

use crate::data_gen::{generate_npz, generate_tfrecord, generate_hdf5, generate_raw_data};

// End imports


// Default size: 20 MB.
pub const DEFAULT_OBJECT_SIZE: usize = 20 * 1024 * 1024;

pub const DEFAULT_REGION: &str = "us-east-1";

// -----------------------------------------------------------------------------
// Enum of our data type, for now just 4 types 
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "UPPERCASE")]   // enum variants match exactly their uppercase names
pub enum ObjectType {
    Npz,
    TfRecord,
    Hdf5,
    Raw,
}

// map &str → ObjectType
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

// allow PyO3 to extract a Python string into our enum
impl<'source> FromPyObject<'source> for ObjectType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Pull out a &str, the delegate fto our From<&str> impl 
        let s: &str = ob.extract()?;
        Ok(ObjectType::from(s))
    }
}



// -----------------------------------------------------------------------------
// Generate a buffer of random bytes.
// -----------------------------------------------------------------------------

/// A base random block of 512 bytes, generated once.
static BASE_BLOCK: Lazy<Vec<u8>> = Lazy::new(|| {
    let mut block = vec![0u8; 512];
    rand::rngs::ThreadRng::default().fill(&mut block[..]);
    block
});

/// Generates a buffer of `size` random bytes by:
/// 1. Enforcing a minimum size of 512 bytes.
/// 2. Filling each 512-byte block with a static base block.
/// 3. Modifying the first 32 bytes of each block,
///    and modifying the last 32 bytes only if the block is larger than 128 bytes.
///
/// This ensures each 512-byte block is unique while avoiding the need to generate a whole new
/// random buffer on every call.
pub fn generate_random_data(mut size: usize) -> Vec<u8> {
    // Enforce a minimum size of 512 bytes.
    if size < 512 {
        size = 512;
    }

    // Allocate the buffer.
    let mut data = vec![0u8; size];

    // Fill each 512-byte block by copying from the static base block.
    for chunk in data.chunks_mut(512) {
        let len = chunk.len();
        chunk.copy_from_slice(&BASE_BLOCK[..len]);
    }

    let mut rng = rand::rngs::ThreadRng::default();
    let mut offset = 0;
    while offset < size {
        let block_end = std::cmp::min(offset + 512, size);
        let block_size = block_end - offset;

        // Modify the first 32 bytes (or the full block if it's smaller).
        if block_size > 0 {
            let first_len = if block_size >= 32 { 32 } else { block_size };
            rng.fill(&mut data[offset .. offset + first_len]);
        }

        // Modify the last 32 bytes only if the block is larger than 128 bytes.
        if block_size > 128 {
            rng.fill(&mut data[block_end - 32 .. block_end]);
        }

        offset += 512;
    }

    data
}

// -----------------------------------------------------------------------------
//  Global S3 client (lazy, thread‑safe) ---------------------------------------
// -----------------------------------------------------------------------------
static CLIENT: OnceCell<Client> = OnceCell::new();

fn client() -> Result<Client> {
    CLIENT
        .get_or_try_init(|| {
            // Load .env first so AWS_* vars are available.
            dotenvy::dotenv().ok();

            // Check for required AWS credentials.
            if std::env::var("AWS_ACCESS_KEY_ID").is_err() || std::env::var("AWS_SECRET_ACCESS_KEY").is_err() {
                return Err(anyhow::anyhow!(
                    "Missing required environment variables: AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY. \
                    Please set these variables (and optionally AWS_REGION) in your environment or .env file."
                ));
            }

            // Build AWS config loader.
            let region = RegionProviderChain::first_try(
                env::var("AWS_REGION").ok().map(Region::new),
            )
            .or_default_provider()
            .or_else(Region::new(DEFAULT_REGION));

            let mut loader =
                aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region);

            if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
                if !endpoint.is_empty() {
                    loader = loader.endpoint_url(endpoint);
                }
            }

            let fut = loader.load();

            // Resolve the future regardless of whether we are inside a runtime.
            let cfg = match Handle::try_current() {
                Ok(handle) => task::block_in_place(|| handle.block_on(fut)),
                Err(_) => {
                    static RT: Lazy<tokio::runtime::Runtime> =
                        Lazy::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));
                    RT.block_on(fut)
                }
            };
            Ok::<_, anyhow::Error>(Client::new(&cfg))
        })
        .map(Clone::clone)
}

// -----------------------------------------------------------------------------
//  Helper: synchronously wait on a future -------------------------------------
// -----------------------------------------------------------------------------
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    if let Ok(handle) = Handle::try_current() {
        handle.block_on(fut)
    } else {
        static RT: Lazy<tokio::runtime::Runtime> =
            Lazy::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));
        RT.block_on(fut)
    }
}

// -----------------------------------------------------------------------------
//  URI helpers ----------------------------------------------------------------
// -----------------------------------------------------------------------------

/// Split `s3://bucket/key` → (`bucket`, `key`). `key` may be empty (prefix).
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let trimmed = uri
        .strip_prefix("s3://")
        .context("URI must start with s3://")?;
    let (bucket, key) = trimmed
        .split_once('/')
        .context("URI must contain a '/' after bucket")?;
    Ok((bucket.to_owned(), key.to_owned()))
}

/// Create an S3 bucket. If the bucket already exists, ignore the error.
pub fn create_bucket(bucket: &str) -> Result<()> {
    block_on(async {
        let client = client()?;
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


// ---------------------
// Put operations 
// ---------------------

/// Asynchronously uploads an object to S3.
pub async fn put_object_async(bucket: &str, key: &str, data: &[u8]) -> Result<()> {
    //let config = aws_config::load_from_env().await; // Old definition
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);
    let body = ByteStream::from(data.to_vec());
    client.put_object().bucket(bucket).key(key).body(body).send().await?;
    Ok(())
}

/// Synchronous wrapper for uploading an object.
pub fn put_object(bucket: &str, key: &str, data: &[u8]) -> Result<()> {
    block_on(put_object_async(bucket, key, data))
}

/// Upload multiple objects in parallel using the same data buffer.
pub fn put_objects_parallel(
    uris: &[String],
    data: &[u8],
    max_in_flight: usize,
) -> Result<()> {
    block_on(async {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for uri in uris.iter().cloned() {
            let sem = sem.clone();
            let data = data.to_vec();
            let (bucket, key) = parse_s3_uri(&uri)?;
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                put_object_async(&bucket, &key, &data).await
            }));
        }
        while let Some(res) = futs.next().await {
            res??;
        }
        Ok(())
    })
}

/// Helper: Upload a single object with random data.
pub fn put_object_with_random_data(uri: &str, size: usize) -> Result<()> {
    let (bucket, key) = parse_s3_uri(uri)?;
    let data = generate_random_data(size);
    put_object(&bucket, &key, &data)
}

/// Helper: Upload many objects concurrently using the same random data buffer.
pub fn put_objects_with_random_data(uris: &[String], size: usize, max_in_flight: usize) -> Result<()> {
    let data = generate_random_data(size);
    put_objects_parallel(uris, &data, max_in_flight)
}

/// Upload each URI with a buffer chosen by `object_type`.
pub fn put_objects_with_random_data_and_type(
    uris: &[String],
    size: usize,
    max_in_flight: usize,
    object_type: ObjectType,
) -> Result<(), anyhow::Error> {
    // pick the generator
    let buffer: Vec<u8> = match object_type {
        ObjectType::Npz      => generate_npz(size),
        ObjectType::TfRecord => generate_tfrecord(size),
        ObjectType::Hdf5     => generate_hdf5(size),
        ObjectType::Raw      => generate_raw_data(size),
    };
    // flood‑fill in parallel
    put_objects_parallel(uris, &buffer, max_in_flight)
        .context("S3 PUT failure")?;
    Ok(())
}

// -----------------------------------------------------------------------------
//  Blocking object operations --------------------------------------------------
// -----------------------------------------------------------------------------

/// List every key that starts with `prefix` (handles pagination).
pub fn list_objects(bucket: &str, prefix: &str) -> Result<Vec<String>> {
    block_on(async {
        let client = client()?;
        let mut keys = Vec::new();
        let mut cont: Option<String> = None;
        loop {
            let mut req = client.list_objects_v2().bucket(bucket).prefix(prefix);
            if let Some(token) = &cont {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.context("list_objects_v2 failed")?;
            for obj in resp.contents() {
                if let Some(k) = obj.key() {
                    keys.push(k.to_owned());
                }
            }
            if let Some(token) = resp.next_continuation_token() {
                cont = Some(token.to_string());
            } else {
                break;
            }
        }
        Ok(keys)
    })
}

/// Delete many keys, batching at 1 000 objects per call.
pub fn delete_objects(bucket: &str, keys: &[String]) -> Result<()> {
    use aws_sdk_s3::types::{Delete, ObjectIdentifier};

    block_on(async {
        let client = client()?;

        for chunk in keys.chunks(1_000) {
            // 1. Build ObjectIdentifier list, propagating any build errors.
            let objs: Vec<ObjectIdentifier> = chunk
                .iter()
                .map(|k| {
                    ObjectIdentifier::builder()
                        .key(k)
                        .build()
                        .map_err(anyhow::Error::from)
                })
                .collect::<Result<_>>()?;

            // 2. Build the Delete struct (also returns Result).
            let delete: Delete = Delete::builder()
                .set_objects(Some(objs))
                .build()
                .map_err(anyhow::Error::from)?;

            // 3. Call S3.
            client
                .delete_objects()
                .bucket(bucket)
                .delete(delete)
                .send()
                .await
                .context("delete_objects failed")?;
        }
        Ok(())
    })
}

// ----------------------------
// Get operations
// ----------------------------

/// Download a single object into memory.
pub fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>> {
    block_on(async {
        let client = client()?;
        let resp = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .context("get_object failed")?;
        let data = resp
            .body
            .collect()
            .await
            .context("collect body failed")?
            .into_bytes();
        Ok(data.to_vec())
    })
}

/// Convenience wrapper that takes a full URI.
pub fn get_object_uri(uri: &str) -> Result<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: URI has no object key – use a full key or the bulk mode");
    }
    get_object(&bucket, &key)
}

// -----------------------------------------------------------------------------
//  Async helpers (used by bulk downloader) ------------------------------------
// -----------------------------------------------------------------------------

async fn get_object_async(bucket: &str, key: &str) -> Result<Vec<u8>> {
    let client = client()?;
    let resp = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await?
        .body
        .collect()
        .await?;
    Ok(resp.into_bytes().to_vec())
}

async fn get_object_uri_async(uri: &str) -> Result<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri)?;
    if key.is_empty() {
        bail!("Cannot GET: URI has no object key – use a full key or the bulk mode");
    }
    get_object_async(&bucket, &key).await
}

// -----------------------------------------------------------------------------
//  Bulk parallel download ------------------------------------------------------
// -----------------------------------------------------------------------------

/// Download many objects concurrently (bounded by `max_in_flight`).
/// Returns results in the **same order** as `uris`.
pub fn get_objects_parallel(
    uris: &[String],
    max_in_flight: usize,
) -> Result<Vec<(String, Vec<u8>)>> {
    block_on(async {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for uri in uris.iter().cloned() {
            let sem = sem.clone();
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                //let bytes = get_object_uri_async(&uri).await?;
                let bytes = get_object_uri_async(&uri).await.map_err(|e| {
                    eprintln!("Error retrieving object {}: {}", uri, e);
                    e
                })?;
                Ok::<_, anyhow::Error>((uri, bytes))
            }));
        }

        let mut out = Vec::with_capacity(uris.len());
        while let Some(res) = futs.next().await {
            out.push(res??);
        }
        // Preserve caller order.
        out.sort_by_key(|(uri, _)| uris.iter().position(|u| u == uri).unwrap());
        Ok(out)
    })
}

