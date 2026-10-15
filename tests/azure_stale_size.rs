// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Azure downloads and the object-size cache (#242 follow-up, Blue review 2026-10-15).
//!
//! `AzureObjectStore` caches the size a `HEAD` reported for 60 s. A blob that is overwritten
//! with a different size inside that window used to make the sequential download plan its
//! requests from the OLD size: a blob cached as 0 or 1 byte and then grown to 1 MiB became about
//! a million one-byte range requests, issued one after another (the SDK learns the real length
//! from the first response but keeps the caller's partition size).
//!
//! Everything runs against a local fake Blob service on http (`auto` sends no credential there).
//! The fake answers 403 after `REQUEST_CAP` requests, so a regression fails in milliseconds
//! instead of hanging.

#![cfg(feature = "backend-azure")]

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use s3dlio::object_store::{AzureConfig, AzureDownloadMode, AzureObjectStore, ObjectStore};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

const REQUEST_CAP: usize = 300;
const MIB: usize = 1024 * 1024;

#[derive(Default)]
struct Fake {
    blobs: HashMap<String, Vec<u8>>,
    /// (method, range header) of every request, in order.
    requests: Vec<(String, Option<String>)>,
}

type State = Arc<Mutex<Fake>>;

fn plain(status: StatusCode) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

async fn handle(state: State, req: Request<Incoming>) -> Response<Full<Bytes>> {
    let method = req.method().clone();
    let path = req.uri().path().trim_start_matches('/').to_string();
    let range = req
        .headers()
        .get("range")
        .or_else(|| req.headers().get("x-ms-range"))
        .map(|v| v.to_str().unwrap_or("").to_string());
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes())
        .unwrap_or_default();

    let mut fake = state.lock().unwrap();
    fake.requests.push((method.to_string(), range.clone()));
    if fake.requests.len() > REQUEST_CAP {
        return plain(StatusCode::FORBIDDEN);
    }
    match method {
        Method::PUT => {
            fake.blobs.insert(path, body.to_vec());
            Response::builder()
                .status(StatusCode::CREATED)
                .header("etag", "\"0x1\"")
                .header("last-modified", "Wed, 14 Oct 2026 10:00:00 GMT")
                .body(Full::new(Bytes::new()))
                .unwrap()
        }
        Method::DELETE => {
            let existed = fake.blobs.remove(&path).is_some();
            plain(if existed {
                StatusCode::ACCEPTED
            } else {
                StatusCode::NOT_FOUND
            })
        }
        Method::GET | Method::HEAD => {
            let Some(data) = fake.blobs.get(&path).cloned() else {
                return plain(StatusCode::NOT_FOUND);
            };
            let total = data.len();
            let mut builder = Response::builder()
                .header("etag", "\"0x1\"")
                .header("last-modified", "Wed, 14 Oct 2026 10:00:00 GMT")
                .header("x-ms-blob-type", "BlockBlob")
                .header("accept-ranges", "bytes");
            let (status, chunk) = match range.as_deref() {
                Some(r) if method == Method::GET => {
                    let (s, e) = r.trim_start_matches("bytes=").split_once('-').unwrap();
                    let start: usize = s.parse().unwrap();
                    if start >= total {
                        // What Azure answers for a range past the end (and for any range of an
                        // empty blob).
                        return Response::builder()
                            .status(StatusCode::RANGE_NOT_SATISFIABLE)
                            .header("x-ms-error-code", "InvalidRange")
                            .header("content-range", format!("bytes */{total}"))
                            .body(Full::new(Bytes::new()))
                            .unwrap();
                    }
                    let end = if e.is_empty() {
                        total - 1
                    } else {
                        e.parse::<usize>().unwrap().min(total - 1)
                    };
                    builder =
                        builder.header("content-range", format!("bytes {start}-{end}/{total}"));
                    (StatusCode::PARTIAL_CONTENT, data[start..=end].to_vec())
                }
                _ => (StatusCode::OK, data),
            };
            builder = builder.status(status);
            if method == Method::HEAD {
                builder
                    .header("content-length", total.to_string())
                    .body(Full::new(Bytes::new()))
                    .unwrap()
            } else {
                builder
                    .header("content-length", chunk.len().to_string())
                    .body(Full::new(Bytes::from(chunk)))
                    .unwrap()
            }
        }
        _ => plain(StatusCode::METHOD_NOT_ALLOWED),
    }
}

async fn start() -> (State, u16) {
    let state: State = Arc::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let shared = Arc::clone(&state);
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                continue;
            };
            let state = Arc::clone(&shared);
            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let state = Arc::clone(&state);
                    async move { Ok::<_, Infallible>(handle(state, req).await) }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(tcp), service)
                    .await;
            });
        }
    });
    (state, port)
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn uri(key: &str) -> String {
    format!("az://acct/ctr/{key}")
}

fn set_blob(state: &State, key: &str, data: Vec<u8>) {
    state
        .lock()
        .unwrap()
        .blobs
        .insert(format!("acct/ctr/{key}"), data);
}

fn requests_since(state: &State, mark: usize) -> Vec<(String, Option<String>)> {
    state.lock().unwrap().requests[mark..].to_vec()
}

fn mark(state: &State) -> usize {
    state.lock().unwrap().requests.len()
}

fn count(requests: &[(String, Option<String>)], method: &str) -> usize {
    requests.iter().filter(|(m, _)| m == method).count()
}

fn store(mode: AzureDownloadMode, range_engine: bool) -> AzureObjectStore {
    let mut config = AzureConfig {
        download_mode: mode,
        enable_range_engine: range_engine,
        ..AzureConfig::default()
    };
    config.range_engine.min_split_size = 64 * 1024;
    config.range_engine.chunk_size = 256 * 1024;
    AzureObjectStore::with_config(config)
}

/// Read `key` once, then grow it on the fake, then read it again inside the size-cache TTL.
async fn stale_then_grow(state: &State, store: &AzureObjectStore, key: &str, first: usize) {
    set_blob(state, key, pattern(first, 1));
    let got = store
        .get(&uri(key))
        .await
        .unwrap_or_else(|e| panic!("first read of {first} bytes: {e:#}"));
    assert_eq!(got.as_ref(), pattern(first, 1).as_slice());

    let grown = pattern(MIB + 12_345, 7);
    set_blob(state, key, grown.clone());
    let before = mark(state);
    let got = store
        .get(&uri(key))
        .await
        .unwrap_or_else(|e| panic!("read after growing from {first} bytes: {e:#}"));
    assert_eq!(
        got.len(),
        grown.len(),
        "short or wrong-length body after the blob grew from {first} bytes"
    );
    assert_eq!(got.as_ref(), grown.as_slice());
    let seen = requests_since(state, before);
    assert!(
        seen.len() <= 12,
        "RED: re-reading a blob that grew from {first} bytes to {} made {} requests (cap {REQUEST_CAP}): {:?}",
        grown.len(),
        seen.len(),
        &seen[..seen.len().min(5)]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn downloads_do_not_plan_requests_from_a_stale_size() {
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    for var in [
        "S3DLIO_AZURE_CREDENTIAL",
        "S3DLIO_AZURE_SAS_TOKEN",
        "AZURE_BLOB_ENDPOINT_URL",
    ] {
        std::env::remove_var(var);
    }

    // 1. Sequential (explicit): the cached size is only a hint. 0 and 1 byte are the worst cases.
    let sequential = store(AzureDownloadMode::Sequential, false);
    stale_then_grow(&state, &sequential, "seq-zero", 0).await;
    stale_then_grow(&state, &sequential, "seq-one", 1).await;

    // 2. The default (Auto, Azure SDK engine) does not even ask for the size.
    let default = AzureObjectStore::new();
    stale_then_grow(&state, &default, "auto-one", 1).await;

    // 3. s3dlio's RangeEngine plans its ranges from the size, so it must not trust a cached one.
    let engine = store(AzureDownloadMode::Auto, true);
    stale_then_grow(&state, &engine, "engine-128k", 128 * 1024).await;

    // 4. Default mode: no HEAD at all, one GET per small blob.
    let n = 5;
    for i in 0..n {
        set_blob(&state, &format!("plain-{i}"), pattern(1000 + i, 3));
    }
    let before = mark(&state);
    for i in 0..n {
        let got = default.get(&uri(&format!("plain-{i}"))).await.unwrap();
        assert_eq!(got.as_ref(), pattern(1000 + i, 3).as_slice());
    }
    let seen = requests_since(&state, before);
    assert_eq!(
        count(&seen, "HEAD"),
        0,
        "RED: the default mode asked for the size of every blob first: {seen:?}"
    );
    assert_eq!(count(&seen, "GET"), n, "{seen:?}");

    // 5. A write through the store updates the cached size; a delete drops it.
    set_blob(&state, "mut", pattern(10, 5));
    sequential.get(&uri("mut")).await.unwrap(); // caches 10
    let big = pattern(MIB, 9);
    sequential
        .put(&uri("mut"), Bytes::from(big.clone()))
        .await
        .unwrap();
    assert_eq!(
        sequential.get(&uri("mut")).await.unwrap().as_ref(),
        big.as_slice()
    );
    sequential.delete(&uri("mut")).await.unwrap();
    let before = mark(&state);
    assert!(
        sequential.get(&uri("mut")).await.is_err(),
        "the blob is gone"
    );
    let seen = requests_since(&state, before);
    assert!(
        count(&seen, "HEAD") >= 1,
        "RED: after delete the next read trusted the cached size and made no HEAD: {seen:?}"
    );
}
