// src/s3_utils.rs
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides high‑level S3 operations: list, get, delete, typed PUT.

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::{config::Region, Client};
use aws_sdk_s3::primitives::ByteStream;
use futures::{stream::FuturesUnordered, StreamExt};
use once_cell::sync::OnceCell;
use std::{env, sync::Arc};
use tokio::{runtime::Handle, sync::Semaphore, task};    // import task for block_in_place
use pyo3::{FromPyObject, PyAny, PyResult};

// data generation helpers
use crate::data_gen::{generate_npz, generate_tfrecord, generate_hdf5, generate_raw_data};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_OBJECT_SIZE: usize = 20 * 1024 * 1024;
pub const DEFAULT_REGION: &str     = "us-east-1";

// -----------------------------------------------------------------------------
// ObjectType enum for typed data generation
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "UPPERCASE")] // CLI shows NPZ, TFRECORD, HDF5, RAW
pub enum ObjectType {
    Npz,
    TfRecord,
    Hdf5,
    Raw,
}

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

impl<'source> FromPyObject<'source> for ObjectType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s = ob.extract::<&str>()?;
        Ok(ObjectType::from(s))
    }
}

// -----------------------------------------------------------------------------
// Global S3 client (lazy, thread‑safe)
// -----------------------------------------------------------------------------
static CLIENT: OnceCell<Client> = OnceCell::new();
static RUNTIME: OnceCell<tokio::runtime::Runtime> = OnceCell::new();   // for block_on fallback

fn client() -> Result<Client> {
    CLIENT.get_or_try_init(|| {
        dotenvy::dotenv().ok();
        if env::var("AWS_ACCESS_KEY_ID").is_err() || env::var("AWS_SECRET_ACCESS_KEY").is_err() {
            bail!("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY");
        }
        let region = RegionProviderChain::first_try(env::var("AWS_REGION").ok().map(Region::new))
            .or_default_provider()
            .or_else(Region::new(DEFAULT_REGION));
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest()).region(region);
        if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
            if !endpoint.is_empty() {
                loader = loader.endpoint_url(endpoint);
            }
        }
        let fut = loader.load();
        let cfg = match Handle::try_current() {
            Ok(handle) => task::block_in_place(|| handle.block_on(fut)),
            Err(_) => {
                let rt = RUNTIME.get_or_init(|| {
                    tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
                });
                rt.block_on(fut)
            }
        };
        Ok::<_, anyhow::Error>(Client::new(&cfg))
    }).map(Clone::clone)
}

// -----------------------------------------------------------------------------
// Helper: synchronously wait on a future
// -----------------------------------------------------------------------------
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    if let Ok(handle) = Handle::try_current() {
        handle.block_on(fut)
    } else {
        let rt = RUNTIME.get_or_init(|| {
            tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
        });
        rt.block_on(fut)
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

/// Create an S3 bucket if it does not exist.
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

// -----------------------------------------------------------------------------
// List and Delete operations
// -----------------------------------------------------------------------------
/// List every key under `prefix` (handles pagination).
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
            //
            // Four attempts at iterating over objects in the response
            //for obj in resp.contents().as_ref().unwrap_or_default() {
            //for obj in resp.contents().iter().flatten() {
            //for obj in resp.contents().unwrap_or(&[]) {
            //for obj in resp.contents().map(|v| v.as_slice()).unwrap_or(&[]) {
            
            for obj in resp.contents() {
                if let Some(k) = obj.key() {
                    keys.push(k.to_string());
                }
            }
            if let Some(next) = resp.next_continuation_token() {
                cont = Some(next.to_string());
            } else {
                break;
            }
        }
        Ok(keys)
    })
}

/// Delete many keys (up to 1000 per call).
pub fn delete_objects(bucket: &str, keys: &[String]) -> Result<()> {
    use aws_sdk_s3::types::{Delete, ObjectIdentifier};
    block_on(async {
        let client = client()?;
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

// -----------------------------------------------------------------------------
// Download (GET) operations
// -----------------------------------------------------------------------------
/// Download a single object from S3 (bucket/key).
async fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>> {
    let client = client()?;
    let resp = client.get_object().bucket(bucket).key(key)
        .send().await.context("get_object failed")?;
    let data = resp.body.collect().await.context("collect body failed")?.into_bytes();
    Ok(data.to_vec())
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
/// Upload each URI with a buffer chosen by `object_type`.
pub fn put_objects_with_random_data_and_type(
    uris: &[String], size: usize, max_in_flight: usize,
    object_type: ObjectType,
) -> Result<()> {
    let buffer = match object_type {
        ObjectType::Npz      => generate_npz(size),
        ObjectType::TfRecord => generate_tfrecord(size),
        ObjectType::Hdf5     => generate_hdf5(size),
        ObjectType::Raw      => generate_raw_data(size),
    };
    put_objects_parallel(&uris, &buffer, max_in_flight)
}

// -----------------------------------------------------------------------------
// Internal helpers (private)
// -----------------------------------------------------------------------------
async fn put_object_async(bucket: &str, key: &str, data: &[u8]) -> Result<()> {
    let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = Client::new(&cfg);
    let body = ByteStream::from(data.to_vec());
    client.put_object().bucket(bucket).key(key).body(body).send().await?;
    Ok(())
}


//fn put_objects_parallel(uris: &[String], data: &[u8], max_in_flight: usize) -> Result<()> {
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

