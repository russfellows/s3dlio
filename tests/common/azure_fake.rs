// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! A local fake Azure Blob service over plain http, for tests of the Azure backend.
//!
//! Blobs live under `/<account>/<container>/<key>`. The fake supports HEAD, GET (with `Range`
//! and `Content-Range`, and Azure's `416 InvalidRange` for a range past the end), PUT, DELETE
//! (404 `BlobNotFound` for a missing blob), and a flat container listing. It records every
//! request and answers 403 after `cap` requests so a request storm fails fast.
//!
//! Include with `#[path = "common/azure_fake.rs"] mod azure_fake;`.

#![allow(dead_code)]

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

pub const MIB: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub struct Req {
    pub method: String,
    /// Percent-decoded path.
    pub path: String,
    pub query: String,
    pub range: Option<String>,
}

pub struct Fake {
    /// `acct/ctr/key` -> content.
    pub blobs: HashMap<String, Vec<u8>>,
    pub requests: Vec<Req>,
    /// After this many requests every request is answered 403.
    pub cap: usize,
    /// Accounts whose every request is answered 403 with the request URL echoed in the body
    /// (as a misconfigured proxy might).
    pub forbidden_accounts: HashSet<String>,
    /// Names the listing reports although no such blob exists (it vanished after the listing).
    pub list_ghosts: Vec<String>,
}

impl Default for Fake {
    fn default() -> Self {
        Self {
            blobs: HashMap::new(),
            requests: Vec::new(),
            cap: 300,
            forbidden_accounts: HashSet::new(),
            list_ghosts: Vec::new(),
        }
    }
}

pub type State = Arc<Mutex<Fake>>;

fn plain(status: StatusCode) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

pub fn pct_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Ok(b) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

async fn handle(state: State, req: Request<Incoming>) -> Response<Full<Bytes>> {
    let method = req.method().clone();
    let full_uri = req.uri().to_string();
    let path = pct_decode(req.uri().path().trim_start_matches('/'));
    let query = req.uri().query().unwrap_or("").to_string();
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
    fake.requests.push(Req {
        method: method.to_string(),
        path: path.clone(),
        query: query.clone(),
        range: range.clone(),
    });
    if fake.requests.len() > fake.cap {
        return plain(StatusCode::FORBIDDEN);
    }
    let account = path.split('/').next().unwrap_or("").to_string();
    if fake.forbidden_accounts.contains(&account) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .header("x-ms-error-code", "AuthorizationFailure")
            .body(Full::new(Bytes::from(format!("forbidden: {full_uri}"))))
            .unwrap();
    }

    // Container listing: GET /<account>/<container>?restype=container&comp=list[&prefix=..]
    let mut segs = path.splitn(3, '/');
    let (_, container, key) = (
        segs.next(),
        segs.next().unwrap_or(""),
        segs.next().unwrap_or(""),
    );
    if method == Method::GET && key.is_empty() && query.contains("comp=list") {
        let prefix = query
            .split('&')
            .find_map(|p| p.strip_prefix("prefix="))
            .map(pct_decode)
            .unwrap_or_default();
        let base = format!("{account}/{container}/");
        let mut names: Vec<String> = fake
            .blobs
            .keys()
            .filter_map(|n| n.strip_prefix(&base))
            .filter(|n| n.starts_with(&prefix))
            .map(str::to_string)
            .collect();
        names.extend(
            fake.list_ghosts
                .iter()
                .filter(|n| n.starts_with(&prefix))
                .cloned(),
        );
        names.sort();
        let items: String = names
            .iter()
            .map(|n| {
                let len = fake.blobs.get(&format!("{base}{n}")).map_or(0, Vec::len);
                format!(
                    "<Blob><Name>{}</Name><Properties><Content-Length>{len}</Content-Length></Properties></Blob>",
                    xml_escape(n)
                )
            })
            .collect();
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/xml")
            .body(Full::new(Bytes::from(format!(
                "<?xml version=\"1.0\" encoding=\"utf-8\"?><EnumerationResults ContainerName=\"{container}\">\
                 <Blobs>{items}</Blobs><NextMarker /></EnumerationResults>"
            ))))
            .unwrap();
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
            if fake.blobs.remove(&path).is_some() {
                plain(StatusCode::ACCEPTED)
            } else {
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("x-ms-error-code", "BlobNotFound")
                    .body(Full::new(Bytes::new()))
                    .unwrap()
            }
        }
        Method::GET | Method::HEAD => {
            let Some(data) = fake.blobs.get(&path).cloned() else {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("x-ms-error-code", "BlobNotFound")
                    .body(Full::new(Bytes::new()))
                    .unwrap();
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

/// Start the fake on a free local port.
pub async fn start() -> (State, u16) {
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

pub fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

pub fn uri(key: &str) -> String {
    format!("az://acct/ctr/{key}")
}

pub fn set_blob(state: &State, key: &str, data: Vec<u8>) {
    state
        .lock()
        .unwrap()
        .blobs
        .insert(format!("acct/ctr/{key}"), data);
}

pub fn mark(state: &State) -> usize {
    state.lock().unwrap().requests.len()
}

pub fn requests_since(state: &State, mark: usize) -> Vec<Req> {
    state.lock().unwrap().requests[mark..].to_vec()
}

pub fn count(requests: &[Req], method: &str) -> usize {
    requests.iter().filter(|r| r.method == method).count()
}

/// Remove every variable that selects an Azure endpoint or credential, so a test starts from a
/// known environment whatever the developer's shell holds.
pub fn clear_azure_env() {
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

/// One test at a time may change the process environment; hold this for the whole test.
pub static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
