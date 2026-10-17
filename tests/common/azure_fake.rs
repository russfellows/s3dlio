// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! A local fake Azure Blob service over plain http, for tests of the Azure backend.
//!
//! Blobs live under `/<account>/<container>/<key>`. The fake supports HEAD, GET (with `Range`
//! and `Content-Range`, and Azure's `416 InvalidRange` for a range past the end), PUT, DELETE
//! (404 `BlobNotFound` for a missing blob), and a flat container listing. It records every
//! request and answers 403 after `cap` requests so a request storm fails fast.
//!
//! For the caller-buffer GET tests (#264) it also: enforces `If-Match` (412 `ConditionNotMet`),
//! serves per-blob ETags and version ids (or none), can answer a missing container, can cap the
//! length of a ranged reply, and can cut a GET body short after the headers (a mid-stream
//! failure), optionally changing the blob afterwards.
//!
//! Include with `#[path = "common/azure_fake.rs"] mod azure_fake;`.

#![allow(dead_code)]

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

pub const MIB: usize = 1024 * 1024;

/// A response body: whole, or a stream that can fail part-way.
pub type Body = BoxBody<Bytes, std::io::Error>;

fn full(bytes: Bytes) -> Body {
    Full::new(bytes).map_err(|never| match never {}).boxed()
}

/// `sent` and then a connection failure: the headers (with the full declared length) and the
/// first bytes go out, then the server aborts the connection.
fn cut_body(sent: Bytes) -> Body {
    use futures::StreamExt;
    let mut frames: Vec<Result<Frame<Bytes>, std::io::Error>> = Vec::new();
    if !sent.is_empty() {
        frames.push(Ok(Frame::data(sent)));
    }
    // The pause lets the server flush the headers and the bytes before the connection is cut.
    let cut = futures::stream::once(async {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "fake: connection cut",
        ))
    });
    BodyExt::boxed(StreamBody::new(futures::stream::iter(frames).chain(cut)))
}

#[derive(Clone, Debug)]
pub struct Req {
    pub method: String,
    /// Percent-decoded path.
    pub path: String,
    pub query: String,
    pub range: Option<String>,
    pub if_match: Option<String>,
}

/// What happens to the blob after a faulted GET was served.
#[derive(Clone, Debug)]
pub enum AfterFault {
    Nothing,
    /// The blob is deleted.
    Delete,
    /// The blob is replaced by this content and ETag.
    Replace(Vec<u8>, String),
    /// Later replies carry no ETag.
    OmitEtag,
}

/// One scripted GET failure: the reply declares the full length, sends `truncate_after` bytes of
/// it and then the connection is cut; afterwards `then` is applied to the blob.
#[derive(Clone, Debug)]
pub struct GetFault {
    pub truncate_after: usize,
    pub then: AfterFault,
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
    /// ETag per `acct/ctr/key`; a blob without an entry has `"0x1"` (a quoted strong tag).
    pub etags: HashMap<String, String>,
    /// No ETag header at all on blob replies.
    pub omit_etag: bool,
    /// `x-ms-version-id` per `acct/ctr/key`.
    pub version_ids: HashMap<String, String>,
    /// Containers (`acct/ctr`) that do not exist: every blob request answers 404 `ContainerNotFound`.
    pub missing_containers: HashSet<String>,
    /// A ranged reply carries at most this many bytes (a service answering less than asked).
    pub range_cap: Option<usize>,
    /// Ranged GETs are answered 200 with the whole blob (a service that ignores `Range`).
    pub ignore_range: bool,
    /// `If-Match` is not enforced (a service or proxy that ignores the condition).
    pub ignore_if_match: bool,
    /// Scripted GET failures, consumed first in first out by GETs of an existing blob.
    pub get_faults: VecDeque<GetFault>,
}

impl Default for Fake {
    fn default() -> Self {
        Self {
            blobs: HashMap::new(),
            requests: Vec::new(),
            cap: 300,
            forbidden_accounts: HashSet::new(),
            list_ghosts: Vec::new(),
            etags: HashMap::new(),
            omit_etag: false,
            version_ids: HashMap::new(),
            missing_containers: HashSet::new(),
            range_cap: None,
            ignore_range: false,
            ignore_if_match: false,
            get_faults: VecDeque::new(),
        }
    }
}

pub type State = Arc<Mutex<Fake>>;

fn plain(status: StatusCode) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(full(Bytes::new()))
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

async fn handle(state: State, req: Request<Incoming>) -> Response<Body> {
    let method = req.method().clone();
    let full_uri = req.uri().to_string();
    let path = pct_decode(req.uri().path().trim_start_matches('/'));
    let query = req.uri().query().unwrap_or("").to_string();
    let range = req
        .headers()
        .get("range")
        .or_else(|| req.headers().get("x-ms-range"))
        .map(|v| v.to_str().unwrap_or("").to_string());
    let if_match = req
        .headers()
        .get("if-match")
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
        if_match: if_match.clone(),
    });
    if fake.requests.len() > fake.cap {
        return plain(StatusCode::FORBIDDEN);
    }
    let account = path.split('/').next().unwrap_or("").to_string();
    if fake.forbidden_accounts.contains(&account) {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .header("x-ms-error-code", "AuthorizationFailure")
            .body(full(Bytes::from(format!("forbidden: {full_uri}"))))
            .unwrap();
    }

    // Container listing: GET /<account>/<container>?restype=container&comp=list[&prefix=..]
    let mut segs = path.splitn(3, '/');
    let (_, container, key) = (
        segs.next(),
        segs.next().unwrap_or(""),
        segs.next().unwrap_or(""),
    );
    if method != Method::PUT
        && fake
            .missing_containers
            .contains(&format!("{account}/{container}"))
    {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("x-ms-error-code", "ContainerNotFound")
            .body(full(Bytes::new()))
            .unwrap();
    }
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
            .body(full(Bytes::from(format!(
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
                .body(full(Bytes::new()))
                .unwrap()
        }
        Method::DELETE => {
            if fake.blobs.remove(&path).is_some() {
                plain(StatusCode::ACCEPTED)
            } else {
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("x-ms-error-code", "BlobNotFound")
                    .body(full(Bytes::new()))
                    .unwrap()
            }
        }
        Method::GET | Method::HEAD => {
            let Some(data) = fake.blobs.get(&path).cloned() else {
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("x-ms-error-code", "BlobNotFound")
                    .body(full(Bytes::new()))
                    .unwrap();
            };
            let total = data.len();
            let etag = fake
                .etags
                .get(&path)
                .cloned()
                .unwrap_or_else(|| "\"0x1\"".to_string());
            // `If-Match`: a list of tags, or `*`. A mismatch is 412, evaluated before the range.
            if let Some(wanted) = if_match.as_deref().filter(|_| !fake.ignore_if_match) {
                let matches = wanted
                    .split(',')
                    .map(str::trim)
                    .any(|w| w == "*" || w == etag);
                if !matches {
                    return Response::builder()
                        .status(StatusCode::PRECONDITION_FAILED)
                        .header("x-ms-error-code", "ConditionNotMet")
                        .body(full(Bytes::new()))
                        .unwrap();
                }
            }
            let mut builder = Response::builder()
                .header("last-modified", "Wed, 14 Oct 2026 10:00:00 GMT")
                .header("x-ms-blob-type", "BlockBlob")
                .header("accept-ranges", "bytes");
            if !fake.omit_etag {
                builder = builder.header("etag", etag);
            }
            if let Some(version) = fake.version_ids.get(&path) {
                builder = builder.header("x-ms-version-id", version.as_str());
            }
            let (status, chunk) = match range.as_deref() {
                Some(r) if method == Method::GET && !fake.ignore_range => {
                    let (s, e) = r.trim_start_matches("bytes=").split_once('-').unwrap();
                    let start: usize = s.parse().unwrap();
                    if start >= total {
                        return Response::builder()
                            .status(StatusCode::RANGE_NOT_SATISFIABLE)
                            .header("x-ms-error-code", "InvalidRange")
                            .header("content-range", format!("bytes */{total}"))
                            .body(full(Bytes::new()))
                            .unwrap();
                    }
                    let mut end = if e.is_empty() {
                        total - 1
                    } else {
                        e.parse::<usize>().unwrap().min(total - 1)
                    };
                    if let Some(cap) = fake.range_cap {
                        end = end.min(start + cap.max(1) - 1);
                    }
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
                    .body(full(Bytes::new()))
                    .unwrap()
            } else {
                // A scripted failure: declare the full length, send part of it. The headers and
                // the first `truncate_after` bytes go out, then the connection is cut.
                let fault = fake.get_faults.pop_front();
                let declared = chunk.len();
                let cut = fault.is_some();
                let sent = match &fault {
                    Some(f) => chunk[..f.truncate_after.min(declared)].to_vec(),
                    None => chunk,
                };
                if let Some(f) = fault {
                    match f.then {
                        AfterFault::Nothing => {}
                        AfterFault::Delete => {
                            fake.blobs.remove(&path);
                        }
                        AfterFault::Replace(content, new_etag) => {
                            fake.blobs.insert(path.clone(), content);
                            fake.etags.insert(path.clone(), new_etag);
                        }
                        AfterFault::OmitEtag => {
                            fake.omit_etag = true;
                        }
                    }
                }
                let body = if cut {
                    cut_body(Bytes::from(sent))
                } else {
                    full(Bytes::from(sent))
                };
                builder
                    .header("content-length", declared.to_string())
                    .body(body)
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

/// Give one blob (`key` under `acct/ctr`) an ETag; the default is `"0x1"` (quoted, strong).
pub fn set_etag(state: &State, key: &str, etag: &str) {
    state
        .lock()
        .unwrap()
        .etags
        .insert(format!("acct/ctr/{key}"), etag.to_string());
}

/// Give one blob (`key` under `acct/ctr`) a version id.
pub fn set_version_id(state: &State, key: &str, version: &str) {
    state
        .lock()
        .unwrap()
        .version_ids
        .insert(format!("acct/ctr/{key}"), version.to_string());
}

/// Queue a scripted GET failure (see [`GetFault`]).
pub fn push_get_fault(state: &State, truncate_after: usize, then: AfterFault) {
    state.lock().unwrap().get_faults.push_back(GetFault {
        truncate_after,
        then,
    });
}
