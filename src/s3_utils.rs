// src/s3_utils.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
//! Thread‑safe, blocking wrapper around the async AWS Rust SDK.
//! Provides high‑level S3 operations: list, get, delete, typed PUT.


use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::types::{Delete, ObjectIdentifier};
use futures::{stream::FuturesUnordered, StreamExt};
use pyo3::{FromPyObject, PyAny, PyResult};
use std::sync::Arc;
use tokio::sync::Semaphore;

// data generation helpers
use crate::data_gen::{generate_npz, generate_tfrecord, generate_hdf5, generate_raw_data};

// S3 client creation
use crate::s3_client::{aws_s3_client, block_on};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DEFAULT_OBJECT_SIZE: usize = 20 * 1024 * 1024;

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
        let client = aws_s3_client()?;
        let mut keys = Vec::new();
        let mut cont: Option<String> = None;
        loop {
            let mut req = client.list_objects_v2().bucket(bucket).prefix(prefix);
            if let Some(token) = &cont {
                req = req.continuation_token(token);
            }
            let resp = req.send().await.context("list_objects_v2 failed")?;
            
            // iterating over objects in the response
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


// -----------------------------------------------------------------------------
// Delete operation
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Download (GET) operations
// -----------------------------------------------------------------------------
/// Download a single object from S3 (bucket/key).
async fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>> {
    let client = aws_s3_client()?;
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
    let client = aws_s3_client()?;
    let body = ByteStream::from(data.to_vec());
    client.put_object().bucket(bucket).key(key).body(body).send().await?;
    Ok(())
}


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

