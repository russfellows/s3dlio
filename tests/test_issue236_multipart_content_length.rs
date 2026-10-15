// tests/test_issue236_multipart_content_length.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #236: over HTTP/1.1 (the default transport) the
// body-less `POST /<bucket>/<key>?uploads` (CreateMultipartUpload) went out with NO
// `Content-Length` header.  Google Cloud Storage's S3-interoperability endpoint answers
// that with `411 Length Required`, so every multipart upload to GCS failed.
//
// The mock below behaves like GCS in that respect: it answers 411 to any POST/PUT/PATCH
// that carries neither `Content-Length` nor `Transfer-Encoding`, and otherwise serves a
// real multipart upload (the parts are kept and joined on Complete).  It records every
// request's headers.  The tests drive real multipart uploads through s3dlio and assert
// that the `?uploads` POST carried `content-length: 0`, that the upload completed, and
// that the assembled bytes are the bytes uploaded.
//
// Covered client paths: the process-wide client, a per-endpoint client (#208), and the
// per-endpoint client wrapped by the redirect follower (S3DLIO_FOLLOW_REDIRECTS=1).

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::thread;

use bytes::Bytes;
use s3dlio::object_store::{ObjectStore, S3ObjectStore, WriterOptions};

// ----------------------------------------------------------------------------
// The GCS-like mock
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    query: String,
    /// Lower-cased header name -> value.
    headers: HashMap<String, String>,
}

#[derive(Default)]
struct State {
    requests: Vec<Req>,
    /// Requests refused with 411.
    refused_411: usize,
    uploads: HashMap<String, (String, BTreeMap<i32, Vec<u8>>)>,
    completed: HashMap<String, Vec<u8>>,
    next_upload: usize,
}

struct Gcs411Mock {
    port: u16,
    state: Arc<Mutex<State>>,
}

impl Gcs411Mock {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock listener");
        let port = listener.local_addr().expect("mock local addr").port();
        let state = Arc::new(Mutex::new(State::default()));
        let shared = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let shared = Arc::clone(&shared);
                thread::spawn(move || serve_one(stream, shared));
            }
        });
        Gcs411Mock { port, state }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn for_bucket(&self, bucket: &str) -> Vec<Req> {
        let prefix = format!("/{bucket}/");
        self.state
            .lock()
            .map(|s| {
                s.requests
                    .iter()
                    .filter(|r| r.path.starts_with(&prefix))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn refused_411(&self) -> usize {
        self.state.lock().map(|s| s.refused_411).unwrap_or(0)
    }

    fn completed(&self, bucket: &str, key: &str) -> Option<Vec<u8>> {
        self.state
            .lock()
            .ok()?
            .completed
            .get(&format!("/{bucket}/{key}"))
            .cloned()
    }
}

fn query_param<'a>(query: &'a str, name: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (k == name).then_some(v)
    })
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
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let headers: HashMap<String, String> = lines
        .filter_map(|l| {
            let (n, v) = l.split_once(':')?;
            Some((n.trim().to_ascii_lowercase(), v.trim().to_string()))
        })
        .collect();
    let content_length = headers
        .get("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }

    let req = Req {
        method,
        path: path.to_string(),
        query: query.to_string(),
        headers,
    };
    let response = respond(&shared, req, body);
    let _ = stream.write_all(&response).and_then(|()| stream.flush());
}

fn http(status: &str, headers: &[(&str, String)], body: &[u8]) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status}\r\n");
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("Connection: close\r\n\r\n");
    let mut bytes = out.into_bytes();
    bytes.extend_from_slice(body);
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
    )
}

fn respond(shared: &Arc<Mutex<State>>, req: Req, body: Vec<u8>) -> Vec<u8> {
    let Ok(mut st) = shared.lock() else {
        return xml(
            "500 Internal Server Error",
            "<Error><Code>Poisoned</Code></Error>",
        );
    };
    st.requests.push(req.clone());

    // GCS behaviour: a POST/PUT/PATCH with neither Content-Length nor
    // Transfer-Encoding is "411 Length Required", with an HTML (non-S3) body.
    let carries_body = matches!(req.method.as_str(), "POST" | "PUT" | "PATCH");
    if carries_body
        && !req.headers.contains_key("content-length")
        && !req.headers.contains_key("transfer-encoding")
    {
        st.refused_411 += 1;
        let page = "<html><title>Error 411 (Length Required)</title></html>";
        return http(
            "411 Length Required",
            &[
                ("Content-Type", "text/html".to_string()),
                ("Content-Length", page.len().to_string()),
            ],
            page.as_bytes(),
        );
    }

    let path = req.path.clone();
    let bucket_and_key = path.trim_start_matches('/');
    let (bucket, key) = bucket_and_key
        .split_once('/')
        .unwrap_or((bucket_and_key, ""));

    match req.method.as_str() {
        "POST" if query_param(&req.query, "uploads").is_some() => {
            st.next_upload += 1;
            let id = format!("up-{}", st.next_upload);
            st.uploads
                .insert(id.clone(), (path.clone(), BTreeMap::new()));
            xml(
                "200 OK",
                &format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?><InitiateMultipartUploadResult>\
                     <Bucket>{bucket}</Bucket><Key>{key}</Key><UploadId>{id}</UploadId>\
                     </InitiateMultipartUploadResult>"
                ),
            )
        }
        "POST" if query_param(&req.query, "uploadId").is_some() => {
            let id = query_param(&req.query, "uploadId").unwrap_or("");
            match st.uploads.remove(id) {
                Some((upload_path, parts)) => {
                    let joined: Vec<u8> = parts.into_values().flatten().collect();
                    st.completed.insert(upload_path, joined);
                    xml(
                        "200 OK",
                        &format!(
                            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                             <CompleteMultipartUploadResult><Bucket>{bucket}</Bucket>\
                             <Key>{key}</Key><ETag>\"complete-etag\"</ETag>\
                             </CompleteMultipartUploadResult>"
                        ),
                    )
                }
                None => xml("404 Not Found", "<Error><Code>NoSuchUpload</Code></Error>"),
            }
        }
        "PUT" if query_param(&req.query, "partNumber").is_some() => {
            let number = query_param(&req.query, "partNumber")
                .and_then(|n| n.parse::<i32>().ok())
                .unwrap_or(0);
            let id = query_param(&req.query, "uploadId")
                .unwrap_or("")
                .to_string();
            if let Some((_, parts)) = st.uploads.get_mut(&id) {
                parts.insert(number, body);
            }
            http(
                "200 OK",
                &[
                    ("ETag", format!("\"etag-{number}\"")),
                    ("Content-Length", "0".to_string()),
                ],
                b"",
            )
        }
        "DELETE" => {
            let id = query_param(&req.query, "uploadId").unwrap_or("");
            st.uploads.remove(id);
            http("204 No Content", &[], b"")
        }
        _ => http("200 OK", &[("Content-Length", "0".to_string())], b""),
    }
}

// ----------------------------------------------------------------------------
// Fixture
// ----------------------------------------------------------------------------

static GLOBAL: OnceLock<Gcs411Mock> = OnceLock::new();
static ENV: Once = Once::new();
static COUNTER: AtomicUsize = AtomicUsize::new(0);
/// Serialises the tests that build a per-endpoint client, so one test can set
/// `S3DLIO_FOLLOW_REDIRECTS` for exactly the client it builds.
static CLIENT_BUILD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The server `AWS_ENDPOINT_URL` names; set before anything builds the global client.
fn global_server() -> &'static Gcs411Mock {
    ENV.call_once(|| {
        let g = Gcs411Mock::start();
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
            std::env::remove_var("AWS_ENDPOINT_URL_S3");
            std::env::remove_var("S3DLIO_FOLLOW_REDIRECTS");
            std::env::set_var("AWS_ENDPOINT_URL", g.url());
            // A failing request must fail fast, not after SDK back-off.
            std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "1");
        }
        let _ = GLOBAL.set(g);
    });
    GLOBAL.get().expect("global mock set by call_once")
}

fn unique_bucket(tag: &str) -> String {
    format!("b236-{tag}-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn payload(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i.wrapping_mul(7) ^ (i >> 8)) % 253) as u8)
        .collect()
}

/// Assert the multipart upload of `/bucket/key` ran through to completion and that
/// the `?uploads` POST carried `Content-Length: 0`.
fn assert_multipart_ok(server: &Gcs411Mock, label: &str, bucket: &str, key: &str, data: &[u8]) {
    let reqs = server.for_bucket(bucket);
    let creates: Vec<&Req> = reqs
        .iter()
        .filter(|r| r.method == "POST" && query_param(&r.query, "uploads").is_some())
        .collect();
    assert_eq!(
        creates.len(),
        1,
        "{label}: one CreateMultipartUpload: {reqs:?}"
    );
    assert_eq!(
        creates[0].headers.get("content-length").map(String::as_str),
        Some("0"),
        "{label}: the ?uploads POST must carry `content-length: 0`: {:?}",
        creates[0].headers
    );
    assert_eq!(
        server.refused_411(),
        0,
        "{label}: the server answered 411 Length Required to a request"
    );
    assert_eq!(
        server.completed(bucket, key).as_deref(),
        Some(data),
        "{label}: the object the server assembled must equal what was uploaded"
    );
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[tokio::test]
async fn global_client_put_multipart_sends_content_length_zero() {
    let server = global_server();
    let store = S3ObjectStore::new();
    let part = 5 * 1024 * 1024;
    let data = payload(2 * part + 12_345);
    let bucket = unique_bucket("global-mp");
    let uri = format!("s3://{bucket}/obj");

    store
        .put_multipart(&uri, Bytes::from(data.clone()), Some(part))
        .await
        .unwrap_or_else(|e| panic!("global put_multipart must complete: {e:#}"));

    assert_multipart_ok(server, "global put_multipart", &bucket, "obj", &data);
}

#[tokio::test]
async fn global_client_create_writer_sends_content_length_zero() {
    let server = global_server();
    let store = S3ObjectStore::new();
    // The buffered writer uploads with the default 16 MiB part: 17 MiB is two parts.
    let data = payload(17 * 1024 * 1024);
    let bucket = unique_bucket("global-wr");
    let uri = format!("s3://{bucket}/obj");

    let mut writer = store
        .create_writer(&uri, WriterOptions::default())
        .await
        .unwrap_or_else(|e| panic!("create_writer: {e:#}"));
    for chunk in data.chunks(3 * 1024 * 1024) {
        writer
            .write_chunk(chunk)
            .await
            .unwrap_or_else(|e| panic!("write_chunk: {e:#}"));
    }
    writer
        .finalize()
        .await
        .unwrap_or_else(|e| panic!("finalize must complete the upload: {e:#}"));

    assert_multipart_ok(server, "global create_writer", &bucket, "obj", &data);
}

#[tokio::test]
async fn per_endpoint_client_put_multipart_sends_content_length_zero() {
    global_server();
    let b = Gcs411Mock::start();
    let store = {
        let _guard = CLIENT_BUILD.lock().await;
        S3ObjectStore::for_endpoint(&b.url())
            .await
            .expect("for_endpoint with fake credentials")
    };
    let part = 5 * 1024 * 1024;
    let data = payload(2 * part + 54_321);
    let bucket = unique_bucket("ep-mp");
    let uri = format!("s3://{bucket}/obj");

    store
        .put_multipart(&uri, Bytes::from(data.clone()), Some(part))
        .await
        .unwrap_or_else(|e| panic!("per-endpoint put_multipart must complete: {e:#}"));

    assert_multipart_ok(&b, "per-endpoint put_multipart", &bucket, "obj", &data);
}

#[tokio::test]
async fn per_endpoint_redirect_wrapped_client_sends_content_length_zero() {
    global_server();
    let b = Gcs411Mock::start();
    let store = {
        let _guard = CLIENT_BUILD.lock().await;
        #[allow(deprecated)]
        std::env::set_var("S3DLIO_FOLLOW_REDIRECTS", "1");
        let built = S3ObjectStore::for_endpoint(&b.url()).await;
        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_FOLLOW_REDIRECTS");
        built.expect("for_endpoint with fake credentials")
    };
    let part = 5 * 1024 * 1024;
    let data = payload(2 * part + 777);
    let bucket = unique_bucket("ep-redirect-mp");
    let uri = format!("s3://{bucket}/obj");

    store
        .put_multipart(&uri, Bytes::from(data.clone()), Some(part))
        .await
        .unwrap_or_else(|e| panic!("redirect-wrapped put_multipart must complete: {e:#}"));

    assert_multipart_ok(&b, "redirect-wrapped put_multipart", &bucket, "obj", &data);
}

/// With no parsable S3 error body, the failure must name the HTTP status (#236, part 2).
/// A bare `POST` to a server that answers 411 to everything produces exactly that.
#[tokio::test]
async fn http_status_is_named_when_the_error_body_is_not_s3_xml() {
    global_server();
    // A server that always answers 411 with an HTML page, whatever it is sent.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            thread::spawn(move || {
                let mut buf = [0u8; 65536];
                let mut seen = Vec::new();
                while !seen.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => seen.extend_from_slice(&buf[..n]),
                    }
                }
                let page = "<html><title>Error 411 (Length Required)</title></html>";
                let _ = stream.write_all(
                    http(
                        "411 Length Required",
                        &[
                            ("Content-Type", "text/html".to_string()),
                            ("Content-Length", page.len().to_string()),
                        ],
                        page.as_bytes(),
                    )
                    .as_slice(),
                );
            });
        }
    });
    let store = {
        let _guard = CLIENT_BUILD.lock().await;
        S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
            .await
            .expect("for_endpoint with fake credentials")
    };
    let bucket = unique_bucket("err411");
    let err = store
        .put_multipart(
            &format!("s3://{bucket}/obj"),
            Bytes::from(vec![1u8; 1024]),
            None,
        )
        .await
        .expect_err("a 411 from the server must fail the upload");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("HTTP 411 Length Required"),
        "the error must name the HTTP status, got: {msg}"
    );
}

/// The `Content-Length: 0` header is added by the HTTP connector AFTER the AWS SDK
/// has signed the request (`sign_request` runs before `connector.call` in the SDK's
/// orchestrator), so it must not appear in the signature's `SignedHeaders`, and the
/// list must be exactly what it was before the fix.
#[tokio::test]
async fn content_length_is_added_after_signing_and_is_not_signed() {
    global_server();
    let b = Gcs411Mock::start();
    let store = {
        let _guard = CLIENT_BUILD.lock().await;
        S3ObjectStore::for_endpoint(&b.url())
            .await
            .expect("for_endpoint with fake credentials")
    };
    let bucket = unique_bucket("signed");
    let part = 5 * 1024 * 1024;
    let data = payload(part + 1);
    // The outcome is irrelevant here (it fails before the fix, succeeds after): only
    // the recorded `?uploads` POST is inspected.
    let _ = store
        .put_multipart(&format!("s3://{bucket}/obj"), Bytes::from(data), Some(part))
        .await;

    let reqs = b.for_bucket(&bucket);
    let create = reqs
        .iter()
        .find(|r| r.method == "POST" && query_param(&r.query, "uploads").is_some())
        .unwrap_or_else(|| panic!("no ?uploads POST recorded: {reqs:?}"));
    let auth = create
        .headers
        .get("authorization")
        .unwrap_or_else(|| panic!("the request is not signed: {:?}", create.headers));
    let signed = auth
        .split("SignedHeaders=")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .unwrap_or_else(|| panic!("no SignedHeaders in: {auth}"));
    assert!(
        !signed.split(';').any(|h| h == "content-length"),
        "content-length must not be in SignedHeaders: {signed}"
    );
    assert_eq!(
        signed, EXPECTED_SIGNED_HEADERS,
        "SignedHeaders must be unchanged from before the fix"
    );
}

/// The SignedHeaders of the `?uploads` POST as signed by the SDK, recorded from a run
/// on f1d9add (before the fix).
const EXPECTED_SIGNED_HEADERS: &str = "host;x-amz-content-sha256;x-amz-date;x-amz-user-agent";
