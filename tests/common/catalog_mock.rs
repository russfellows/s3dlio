// tests/common/catalog_mock.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// A small in-process S3 server that serves REAL bytes from a catalog, so a test can run two
// servers (A and B) holding DIFFERENT bytes under the same bucket and keys and tell, from
// the bytes it gets back and from the requests each server recorded, which server a
// per-endpoint store (or a download) really talked to.
//
// Supported: ListObjectsV2 (prefix, delimiter; one page), GetObject (optionally ranged),
// HeadObject, DeleteObject, DeleteObjects (POST ?delete). Everything else is a 501.
//
// Include with `#[path = "common/catalog_mock.rs"] mod catalog_mock;`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

/// One request as the server received it.
#[derive(Clone, Debug)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub query: String,
}

#[derive(Default)]
struct State {
    requests: Vec<Req>,
    /// bucket -> key -> bytes
    buckets: BTreeMap<String, BTreeMap<String, Vec<u8>>>,
}

pub struct CatalogMock {
    pub port: u16,
    state: Arc<Mutex<State>>,
}

impl CatalogMock {
    /// Start a server holding `objects` as `(bucket, key, bytes)`. The listener and its
    /// threads live for the rest of the test process.
    pub fn start(objects: &[(&str, &str, &[u8])]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind catalog mock");
        let port = listener.local_addr().expect("local addr").port();
        let mut st = State::default();
        for (bucket, key, bytes) in objects {
            st.buckets
                .entry((*bucket).to_string())
                .or_default()
                .insert((*key).to_string(), bytes.to_vec());
        }
        let state = Arc::new(Mutex::new(st));
        let shared = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let shared = Arc::clone(&shared);
                thread::spawn(move || serve_one(stream, shared));
            }
        });
        CatalogMock { port, state }
    }

    /// Add (or replace) one object after the server started.
    pub fn seed(&self, bucket: &str, key: &str, bytes: &[u8]) {
        if let Ok(mut st) = self.state.lock() {
            st.buckets
                .entry(bucket.to_string())
                .or_default()
                .insert(key.to_string(), bytes.to_vec());
        }
    }

    /// `http://127.0.0.1:<port>`: the value for `AWS_ENDPOINT_URL` or `for_endpoint`.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// `127.0.0.1:<port>`: the authority of a host-form `s3://` URI.
    pub fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// Every request whose path is `bucket` or inside it, in arrival order.
    pub fn requests_for(&self, bucket: &str) -> Vec<Req> {
        let inside = format!("/{bucket}/");
        let root = format!("/{bucket}");
        self.state
            .lock()
            .map(|s| {
                s.requests
                    .iter()
                    .filter(|r| r.path.starts_with(&inside) || r.path == root)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The keys currently in `bucket`.
    pub fn keys(&self, bucket: &str) -> Vec<String> {
        self.state
            .lock()
            .ok()
            .and_then(|s| s.buckets.get(bucket).map(|b| b.keys().cloned().collect()))
            .unwrap_or_default()
    }
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (k == name).then(|| percent_decode(v))
    })
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn http(status: &str, headers: &[(&str, String)], body: &[u8], with_body: bool) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status}\r\n");
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("Connection: close\r\n\r\n");
    let mut bytes = out.into_bytes();
    if with_body {
        bytes.extend_from_slice(body);
    }
    bytes
}

fn xml(status: &str, body: &str) -> Vec<u8> {
    http(
        status,
        &[
            ("Content-Type", "application/xml".to_string()),
            ("Content-Length", body.len().to_string()),
        ],
        body.as_bytes(),
        true,
    )
}

fn serve_one(mut stream: TcpStream, shared: Arc<Mutex<State>>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let request_line = head.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let (path, query) = (path.to_string(), query.to_string());
    let header = |name: &str| -> Option<String> {
        head.lines().find_map(|l| {
            let (n, v) = l.split_once(':')?;
            n.eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    };
    let content_length = header("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let range = header("range");
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }
    let response = respond(
        &shared,
        Req {
            method,
            path,
            query,
        },
        &body,
        range,
    );
    let _ = stream.write_all(&response).and_then(|()| stream.flush());
}

fn respond(shared: &Arc<Mutex<State>>, req: Req, body: &[u8], range: Option<String>) -> Vec<u8> {
    let Ok(mut st) = shared.lock() else {
        return xml(
            "500 Internal Server Error",
            "<Error><Code>Poisoned</Code></Error>",
        );
    };
    st.requests.push(req.clone());
    let trimmed = req.path.trim_start_matches('/');
    let (bucket, key) = trimmed.split_once('/').unwrap_or((trimmed, ""));
    let bucket = percent_decode(bucket);
    let key = percent_decode(key);
    let not_found = || {
        xml(
            "404 Not Found",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>NoSuchKey</Code>\
             <Message>not found</Message></Error>",
        )
    };

    match req.method.as_str() {
        "GET" if query_param(&req.query, "list-type").is_some() => {
            let prefix = query_param(&req.query, "prefix").unwrap_or_default();
            let delimiter = query_param(&req.query, "delimiter").filter(|d| !d.is_empty());
            let mut contents = String::new();
            let mut commons: Vec<String> = Vec::new();
            let mut count = 0usize;
            if let Some(objects) = st.buckets.get(&bucket) {
                for (k, v) in objects {
                    let Some(rest) = k.strip_prefix(&prefix) else {
                        continue;
                    };
                    match delimiter
                        .as_deref()
                        .and_then(|d| rest.find(d).map(|i| (d, i)))
                    {
                        Some((d, i)) => {
                            let common = format!("{prefix}{}", &rest[..i + d.len()]);
                            if !commons.contains(&common) {
                                commons.push(common);
                                count += 1;
                            }
                        }
                        None => {
                            contents.push_str(&format!(
                                "<Contents><Key>{}</Key><Size>{}</Size></Contents>",
                                xml_escape(k),
                                v.len()
                            ));
                            count += 1;
                        }
                    }
                }
            }
            let common_xml: String = commons
                .iter()
                .map(|c| {
                    format!(
                        "<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>",
                        xml_escape(c)
                    )
                })
                .collect();
            xml(
                "200 OK",
                &format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult>\
                     <Name>{bucket}</Name><Prefix>{}</Prefix><KeyCount>{count}</KeyCount>\
                     <MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>\
                     {contents}{common_xml}</ListBucketResult>",
                    xml_escape(&prefix)
                ),
            )
        }
        "GET" | "HEAD" => {
            let Some(bytes) = st.buckets.get(&bucket).and_then(|b| b.get(&key)) else {
                return not_found();
            };
            let is_get = req.method == "GET";
            let (status, slice, first) = match range
                .as_deref()
                .and_then(|r| r.strip_prefix("bytes="))
                .filter(|_| is_get)
            {
                Some(spec) => {
                    let (a, b) = spec.split_once('-').unwrap_or((spec, ""));
                    let start = a.parse::<usize>().unwrap_or(0).min(bytes.len());
                    let end = b
                        .parse::<usize>()
                        .map(|e| (e + 1).min(bytes.len()))
                        .unwrap_or(bytes.len())
                        .max(start);
                    ("206 Partial Content", &bytes[start..end], start)
                }
                None => ("200 OK", &bytes[..], 0),
            };
            let mut headers = vec![
                ("Content-Length", slice.len().to_string()),
                ("ETag", "\"catalog-mock\"".to_string()),
                ("Accept-Ranges", "bytes".to_string()),
            ];
            if status.starts_with("206") {
                headers.push((
                    "Content-Range",
                    format!(
                        "bytes {first}-{}/{}",
                        first + slice.len().saturating_sub(1),
                        bytes.len()
                    ),
                ));
            }
            if req.method == "HEAD" {
                // HEAD reports the whole object's length.
                headers[0] = ("Content-Length", bytes.len().to_string());
            }
            http(status, &headers, slice, is_get)
        }
        "DELETE" => {
            if let Some(b) = st.buckets.get_mut(&bucket) {
                b.remove(&key);
            }
            http("204 No Content", &[], &[], false)
        }
        "POST" if query_param(&req.query, "delete").is_some() => {
            let text = String::from_utf8_lossy(body).into_owned();
            let mut deleted = String::new();
            for part in text.split("<Key>").skip(1) {
                let Some((k, _)) = part.split_once("</Key>") else {
                    continue;
                };
                let k = k
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&amp;", "&");
                if let Some(b) = st.buckets.get_mut(&bucket) {
                    b.remove(&k);
                }
                deleted.push_str(&format!("<Deleted><Key>{}</Key></Deleted>", xml_escape(&k)));
            }
            xml(
                "200 OK",
                &format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult>{deleted}</DeleteResult>"
                ),
            )
        }
        _ => xml(
            "501 Not Implemented",
            "<Error><Code>NotImplemented</Code></Error>",
        ),
    }
}
