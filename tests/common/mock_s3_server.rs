// tests/common/mock_s3_server.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Minimal in-process S3 multipart-upload mock server (H1 test harness,
// docs/implementation-plans/v0.9.109-audit-fix-plan.md §9). Serves just
// enough of the S3 multipart API — CreateMultipartUpload, UploadPart,
// CompleteMultipartUpload, AbortMultipartUpload — to drive
// `s3dlio::multipart::MultipartUploadSink` through failure paths that a
// live S3-compatible backend would rarely exhibit on demand (a transient
// AbortMultipartUpload failure, for instance).
//
// Design notes:
//
// - `s3dlio::s3_client::aws_s3_client_async()` caches the S3 client in a
//   process-global `tokio::sync::OnceCell`, initialized from whatever
//   `AWS_ENDPOINT_URL` / credential env vars are set on first call. All
//   tests in one `tests/*.rs` binary that touch the multipart sink must
//   therefore share ONE mock server endpoint — there is no way to point
//   different tests at different mock servers within the same binary.
//
// - `#[tokio::test]` gives each test function its own short-lived Tokio
//   runtime. A server task spawned with `tokio::spawn` inside one test's
//   runtime is dropped when that runtime shuts down. To outlive every
//   test function in the binary, the mock server runs on a dedicated
//   background OS thread with its own long-lived Tokio runtime — the
//   same pattern `s3dlio::s3_client::global_rt_handle()` uses internally
//   for the crate's own coordinator tasks.
//
// - Per-test behavior (e.g. "the next AbortMultipartUpload must fail")
//   is selected by encoding a marker in the S3 *key* the test uses, not
//   via shared mutable server state — this keeps tests safe to run
//   concurrently (the default `cargo test` behavior) without a Mutex.

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

/// Key-path marker that makes the mock server return a 500 for
/// AbortMultipartUpload on that specific object. Bake this into the S3
/// key a test uses (e.g. `format!("mpu-abort-fails-{unique_suffix}")`)
/// to exercise the transient-abort-failure path deterministically.
pub const ABORT_FAILS_MARKER: &str = "abort-fails";

/// Key-path marker that makes the mock server return a 500 for every
/// UploadPart call on that specific object — used to force the
/// coordinator into its error path without ever reaching Complete.
pub const UPLOAD_PART_FAILS_MARKER: &str = "part-fails";

/// Key substring marker that makes the mock server report a per-object
/// error for that key in a `DeleteObjects` response, while still
/// returning HTTP 200 overall — this is exactly the real S3 behavior
/// bug A4 guards against (per-object failure inside a 200 response).
/// Any key containing this substring comes back in the response's
/// `<Error>` list instead of `<Deleted>`.
pub const DELETE_FAILS_MARKER: &str = "delete-fails";

/// Call counts scoped to a single S3 key. `cargo test` runs integration
/// tests in the same binary concurrently by default, so a single set of
/// process-global counters would let one test observe another test's
/// traffic. Every test in this harness must use a unique key (see
/// `unique_test_key()`) and read back counts only for that key.
#[derive(Clone, Copy, Default, Debug)]
pub struct PathCallCounts {
    pub create_calls: usize,
    pub upload_part_calls: usize,
    pub complete_calls: usize,
    pub abort_calls: usize,
    pub abort_failures_returned: usize,
}

#[derive(Clone, Default)]
pub struct MockS3State {
    per_path: Arc<Mutex<HashMap<String, PathCallCounts>>>,
}

impl MockS3State {
    fn record(&self, key_path: &str, f: impl FnOnce(&mut PathCallCounts)) {
        let mut map = self.per_path.lock().unwrap();
        let entry = map.entry(key_path.to_string()).or_default();
        f(entry);
    }

    /// Snapshot of call counts observed for one S3 key path so far.
    pub fn counts_for(&self, key_path: &str) -> PathCallCounts {
        self.per_path
            .lock()
            .unwrap()
            .get(key_path)
            .copied()
            .unwrap_or_default()
    }

    /// Poll `counts_for(key_path)` until `predicate` returns true or
    /// `timeout` elapses. Returns the last observed snapshot either way
    /// — callers assert on the returned value so a timeout still
    /// produces a useful failure message instead of a bare bool.
    pub fn wait_for(
        &self,
        key_path: &str,
        timeout: Duration,
        predicate: impl Fn(&PathCallCounts) -> bool,
    ) -> PathCallCounts {
        let deadline = Instant::now() + timeout;
        loop {
            let snap = self.counts_for(key_path);
            if predicate(&snap) || Instant::now() >= deadline {
                return snap;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

/// A unique S3 key for one test, so its mock-server call counts can never
/// be conflated with another concurrently-running test's traffic.
pub fn unique_test_key(prefix: &str) -> String {
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{prefix}-{nanos}-{n}")
}

/// The exact request-path string the mock server records call counts
/// under for a given bucket+key — `force_path_style(true)` (set in
/// `s3dlio::s3_client::aws_s3_client_async`) means every request path is
/// `/{bucket}/{key}`. Tests must use this to look up their own counts;
/// the mock server keys on `req.uri().path()` directly.
pub fn key_path(bucket: &str, key: &str) -> String {
    format!("/{bucket}/{key}")
}

static MOCK_SERVER: OnceLock<(u16, MockS3State)> = OnceLock::new();
static ENV_INIT: std::sync::Once = std::sync::Once::new();

/// Ensure the shared mock S3 multipart server is running and the process
/// environment points `aws_s3_client_async()` at it. Idempotent and safe
/// to call from every test — the server is started and the env vars are
/// set exactly once per test binary process.
pub fn ensure_mock_s3_server() -> (u16, MockS3State) {
    let (port, state) = MOCK_SERVER
        .get_or_init(|| {
            let state = MockS3State::default();
            let server_state = state.clone();
            let (tx, rx) = std::sync::mpsc::channel();

            std::thread::Builder::new()
                .name("mock-s3-server".to_string())
                .spawn(move || {
                    let rt = tokio::runtime::Runtime::new()
                        .expect("failed to build mock-s3-server runtime");
                    rt.block_on(async move {
                        let listener = TcpListener::bind("127.0.0.1:0")
                            .await
                            .expect("failed to bind mock S3 server");
                        let port = listener.local_addr().unwrap().port();
                        tx.send(port).expect("send mock S3 server port");
                        serve(listener, server_state).await;
                    });
                })
                .expect("failed to spawn mock-s3-server thread");

            let port = rx.recv().expect("receive mock S3 server port");
            (port, state)
        })
        .clone();

    ENV_INIT.call_once(|| {
        std::env::set_var("AWS_ACCESS_KEY_ID", "mock-access-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "mock-secret-key");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"));
    });

    (port, state)
}

async fn serve(listener: TcpListener, state: MockS3State) {
    let builder = AutoBuilder::new(TokioExecutor::new());
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(v) => v,
            Err(_) => break,
        };
        let io = TokioIo::new(stream);
        let b = builder.clone();
        let st = state.clone();
        tokio::spawn(async move {
            let _ = b
                .serve_connection(
                    io,
                    service_fn(move |req: Request<Incoming>| {
                        let st = st.clone();
                        async move { Ok::<_, std::convert::Infallible>(handle(req, st).await) }
                    }),
                )
                .await;
        });
    }
}

async fn handle(req: Request<Incoming>, state: MockS3State) -> Response<Full<Bytes>> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req.uri().query().unwrap_or("").to_string();
    // Collect the body — DeleteObjects needs to inspect which keys were
    // requested (see below); every other route just needs it drained so
    // the connection can be cleanly reused/closed.
    let body_bytes = req
        .into_body()
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();

    // DeleteObjects: POST /{bucket}?delete, XML body listing <Object><Key>...
    // Any key containing DELETE_FAILS_MARKER comes back as a per-object
    // <Error> entry instead of <Deleted> — HTTP status stays 200 either
    // way, matching real S3 behavior (bug A4's whole point).
    if method == Method::POST && query.contains("delete") {
        let body_str = String::from_utf8_lossy(&body_bytes);
        let keys = extract_xml_tag_values(&body_str, "Key");
        let mut deleted_xml = String::new();
        let mut error_xml = String::new();
        for key in &keys {
            if key.contains(DELETE_FAILS_MARKER) {
                error_xml.push_str(&format!(
                    "<Error><Key>{key}</Key><Code>AccessDenied</Code>\
                     <Message>mock server injected per-object delete failure</Message></Error>"
                ));
            } else {
                deleted_xml.push_str(&format!("<Deleted><Key>{key}</Key></Deleted>"));
            }
        }
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult>{deleted_xml}{error_xml}\
             </DeleteResult>"
        );
        return xml_response(StatusCode::OK, &xml);
    }

    // ListObjectsV2: GET /{bucket}?list-type=2&prefix=...
    //
    // Always returns exactly two synthetic keys under whatever prefix was
    // requested: "{prefix}bar" and "{prefix}other/deep.dat" — chosen so a
    // test can pass a `key_prefix` ending in `/bar` and get one key whose
    // tail-past-prefix matches the resulting regex ("bar") and one whose
    // tail doesn't ("other/deep.dat"), directly reproducing the B7 dead-
    // filter scenario from the audit (`list("s3://bucket/foo/bar")`
    // returning `foo/other/deep.dat` alongside `foo/bar`).
    if method == Method::GET && query.contains("list-type=2") {
        let prefix = query
            .split('&')
            .find_map(|kv| kv.strip_prefix("prefix="))
            .map(percent_decode)
            .unwrap_or_default();
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult>\
             <Name>mock-bucket</Name><Prefix>{prefix}</Prefix>\
             <KeyCount>2</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>\
             <Contents><Key>{prefix}bar</Key></Contents>\
             <Contents><Key>{prefix}other/deep.dat</Key></Contents>\
             </ListBucketResult>"
        );
        return xml_response(StatusCode::OK, &xml);
    }

    // CreateMultipartUpload: POST .../{key}?uploads[=...]
    if method == Method::POST && query.contains("uploads") && !query.contains("uploadId") {
        state.record(&path, |c| c.create_calls += 1);
        let xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <InitiateMultipartUploadResult>\
              <Bucket>mock-bucket</Bucket><Key>mock-key</Key>\
              <UploadId>mock-upload-id</UploadId>\
            </InitiateMultipartUploadResult>";
        return xml_response(StatusCode::OK, xml);
    }

    // UploadPart: PUT .../{key}?partNumber=N&uploadId=...
    if method == Method::PUT && query.contains("partNumber") {
        state.record(&path, |c| c.upload_part_calls += 1);
        if path.contains(UPLOAD_PART_FAILS_MARKER) {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "MockUploadPartFailure");
        }
        return Response::builder()
            .status(StatusCode::OK)
            .header("ETag", "\"mock-part-etag\"")
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    // AbortMultipartUpload: DELETE .../{key}?uploadId=...
    if method == Method::DELETE && query.contains("uploadId") {
        state.record(&path, |c| c.abort_calls += 1);
        if path.contains(ABORT_FAILS_MARKER) {
            state.record(&path, |c| c.abort_failures_returned += 1);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "MockAbortFailure");
        }
        return Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    // CompleteMultipartUpload: POST .../{key}?uploadId=...
    if method == Method::POST && query.contains("uploadId") {
        state.record(&path, |c| c.complete_calls += 1);
        let xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <CompleteMultipartUploadResult>\
              <Location>http://mock/mock-bucket/mock-key</Location>\
              <Bucket>mock-bucket</Bucket><Key>mock-key</Key>\
              <ETag>\"mock-complete-etag\"</ETag>\
            </CompleteMultipartUploadResult>";
        return xml_response(StatusCode::OK, xml);
    }

    // HeadObject (used by S3DLIO_MPU_PUT_VERIFY — not exercised by A2/A3,
    // but harmless to answer so any accidental call doesn't hang).
    if method == Method::HEAD {
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Length", "0")
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Full::new(Bytes::from("mock-s3-server: unhandled route")))
        .unwrap()
}

fn xml_response(status: StatusCode, xml: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/xml")
        .body(Full::new(Bytes::from(xml.to_string())))
        .unwrap()
}

fn error_response(status: StatusCode, code: &str) -> Response<Full<Bytes>> {
    let xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <Error><Code>{code}</Code><Message>mock server injected failure</Message>\
         <RequestId>mock-request-id</RequestId></Error>"
    );
    Response::builder()
        .status(status)
        .header("content-type", "application/xml")
        .body(Full::new(Bytes::from(xml)))
        .unwrap()
}

/// Minimal `%XX` percent-decoder for query-string values — good enough
/// for the ASCII prefixes this test harness uses (slashes, hyphens,
/// alphanumerics); not a general URL-decoding implementation.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Extract every `<tag>value</tag>` occurrence's inner text from a flat
/// XML body. Good enough for the DeleteObjects request shape the AWS SDK
/// generates (`<Object><Key>...</Key></Object>` repeated) — this is a
/// test-only mock, not a general XML parser.
fn extract_xml_tag_values(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut values = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        rest = &rest[start + open.len()..];
        if let Some(end) = rest.find(&close) {
            values.push(rest[..end].to_string());
            rest = &rest[end + close.len()..];
        } else {
            break;
        }
    }
    values
}
