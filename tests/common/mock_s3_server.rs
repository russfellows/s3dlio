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
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
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

/// Key substring marker that makes the mock server's HeadObject response
/// omit the `Content-Length` header entirely (still HTTP 200) — used to
/// reproduce bug B3 (a HEAD that succeeds but doesn't report a size).
pub const NO_CONTENT_LENGTH_MARKER: &str = "no-content-length";

/// Key substring marker that makes the mock server's GetObject handler
/// fail the FIRST request to that key with a transient 500 and succeed
/// on every request after — used to prove a caller retries the whole
/// request rather than surfacing the first transient failure (bug B1).
pub const GET_FAILS_ONCE_MARKER: &str = "get-fails-once";

/// Key substring marker that makes the mock server's HeadObject handler
/// return a 500 for every HEAD request on that key — used to reproduce
/// a HEAD *request* failure (distinct from `NO_CONTENT_LENGTH_MARKER`,
/// where the HEAD succeeds but omits a header) for bug D3: the
/// S3DLIO_MPU_PUT_VERIFY HEAD check must not treat a failed
/// verification *call* as proof the just-completed upload is corrupt.
pub const HEAD_FAILS_MARKER: &str = "head-fails";

/// Key substring marker that makes the mock server treat the object as a
/// real, [`SIZED_OBJECT_BODY`]-byte object: HeadObject reports its true
/// `Content-Length`, and GetObject honours a `Range: bytes=a-b` header by
/// returning exactly that slice (206), or the whole body (200) when no
/// Range header is sent. Every other key keeps the fixed 4-byte `mock`
/// behaviour that the older tests rely on. Used by the issue #172 tests,
/// which must reach the code path that issues one ranged GET per chunk.
pub const SIZED_OBJECT_MARKER: &str = "sized-obj";

/// Body served for keys containing [`SIZED_OBJECT_MARKER`].
pub const SIZED_OBJECT_BODY: &[u8; 16] = b"0123456789abcdef";

/// Key substring marker that makes the mock server answer a HeadObject
/// request with `404 Not Found` (an object that does not exist). Used by the
/// op-log tests that need a missing object on S3.
pub const HEAD_NOT_FOUND_MARKER: &str = "head-404";

/// Key substring marker that makes the mock server answer a plain PutObject
/// with `403 AccessDenied` (a persistent, non-retried failure).
pub const PUT_FORBIDDEN_MARKER: &str = "put-403";

/// Key substring marker that makes the mock server answer a plain PutObject
/// with `500` every time. The S3 SDK retries a 500, so one call reaches the
/// server several times (`PathCallCounts::put_calls` counts every attempt).
pub const PUT_FAILS_MARKER: &str = "put-500";

/// Key substring marker that makes the mock server answer GetObject with
/// `403 AccessDenied` (a persistent, non-retried failure).
pub const GET_FORBIDDEN_MARKER: &str = "get-403";

/// Key substring marker that makes the mock server answer GetObject with
/// `404 NoSuchKey` every time (issue #240: a definite answer that must not be
/// retried by the s3dlio retry loop).
pub const GET_NOT_FOUND_MARKER: &str = "get-404";

/// Key substring marker that makes the mock server answer GetObject with
/// `503 SlowDown` every time (issue #240: a transient failure that must still
/// be retried).
pub const GET_UNAVAILABLE_MARKER: &str = "get-503";

/// Key substring marker that makes the mock server answer the FIRST
/// [`GET_TIMEOUT_FAILURES`] GetObject requests for that key with
/// `400 RequestTimeoutException` and every later one normally. The AWS SDK's own retry
/// layer recognises this code and spends its whole attempt budget on it, so the error
/// the s3dlio code sees is the SDK's FINAL one; the later success proves the OUTER
/// retry loop still starts another attempt (issue #240 follow-up).
pub const GET_TIMEOUT_THEN_OK_MARKER: &str = "get-timeout-then-ok";

/// How many requests [`GET_TIMEOUT_THEN_OK_MARKER`] keys fail: the SDK's attempt
/// budget (`S3DLIO_MAX_RETRY_ATTEMPTS=3`, pinned by the tests that use the marker).
pub const GET_TIMEOUT_FAILURES: usize = 3;

/// Bucket-name marker: a seeded bucket whose name contains this answers its LAST
/// ListObjectsV2 page (`IsTruncated=false`) with a non-empty `NextContinuationToken`
/// too, as some S3-compatible servers do. A listing must stop on `IsTruncated=false`.
pub const LAST_PAGE_TOKEN_MARKER: &str = "last-page-token";

/// Bucket-name marker: a seeded bucket whose name contains this answers every page
/// that has more keys behind it with `IsTruncated=true` but NO `NextContinuationToken`
/// (a broken server). A listing that drives a delete must treat that as an error.
pub const TRUNCATED_NO_TOKEN_MARKER: &str = "truncated-no-token";

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
    pub get_calls: usize,
    pub head_calls: usize,
    /// Single-object `DELETE /{bucket}/{key}` requests.
    pub delete_calls: usize,
    /// Keys named in a batch `DeleteObjects` (`POST /{bucket}?delete`) body,
    /// counted under `/{bucket}/{key}`.
    pub batch_delete_calls: usize,
    /// Plain (non-multipart) `PUT /{bucket}/{key}` requests.
    pub put_calls: usize,
}

/// One request exactly as the server received it: `path` and `query` are the
/// raw (still percent-encoded) request-target components, and `range` is the
/// `Range` header if one was sent. Used by tests that must prove which
/// object path / which kind of GET (ranged or plain) went over the wire.
#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub query: String,
    pub range: Option<String>,
    /// The `Host` header (`host:port` the client addressed), if one was sent.
    pub host: Option<String>,
    /// The `If-Match` header, if one was sent.
    pub if_match: Option<String>,
}

#[derive(Clone, Default)]
pub struct MockS3State {
    per_path: Arc<Mutex<HashMap<String, PathCallCounts>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    /// Seeded buckets (see [`MockS3State::seed_bucket`]): bucket -> sorted keys.
    catalogs: Arc<Mutex<HashMap<String, Vec<String>>>>,
    /// Registered objects (see [`MockS3State::put_object`]): raw request path -> object.
    objects: Arc<Mutex<HashMap<String, MockObject>>>,
}

impl MockS3State {
    /// Seed `bucket` with exactly `keys`. ListObjectsV2 on a seeded bucket is
    /// answered from this catalog the way S3 answers it: only keys that start
    /// with `prefix`; with `delimiter`, keys whose remainder (after the prefix)
    /// contains the delimiter collapse into one `CommonPrefixes` entry; results
    /// come back in UTF-8 key order, `max-keys` per page, with
    /// `continuation-token` paging. A bucket that was never seeded keeps the
    /// legacy synthetic answer (`{prefix}bar` and `{prefix}other/deep.dat`).
    pub fn seed_bucket(&self, bucket: &str, keys: &[&str]) {
        let mut sorted: Vec<String> = keys.iter().map(|k| (*k).to_string()).collect();
        sorted.sort();
        sorted.dedup();
        self.catalogs
            .lock()
            .unwrap()
            .insert(bucket.to_string(), sorted);
    }

    /// The seeded catalog of `bucket`, if it has one.
    fn catalog(&self, bucket: &str) -> Option<Vec<String>> {
        self.catalogs.lock().unwrap().get(bucket).cloned()
    }

    /// The keys still in the seeded `bucket`, sorted. A seeded bucket loses the keys that
    /// DeleteObjects / DeleteObject name (a key carrying [`DELETE_FAILS_MARKER`] is never
    /// removed), so a test can assert exactly what survived a delete.
    pub fn remaining_keys(&self, bucket: &str) -> Vec<String> {
        self.catalog(bucket).unwrap_or_default()
    }

    /// Remove `key` from the seeded catalog of `bucket`, if there is one.
    fn remove_key(&self, bucket: &str, key: &str) {
        if let Some(keys) = self.catalogs.lock().unwrap().get_mut(bucket) {
            keys.retain(|k| k != key);
        }
    }

    fn record(&self, key_path: &str, f: impl FnOnce(&mut PathCallCounts)) {
        let mut map = self.per_path.lock().unwrap();
        let entry = map.entry(key_path.to_string()).or_default();
        f(entry);
    }

    /// Like `record`, but returns the value of `get_calls` AFTER
    /// applying `f` — lets the GetObject handler decide "is this the
    /// first attempt?" atomically with incrementing the counter (no
    /// separate read-then-write race under concurrent requests).
    fn record_and_get(&self, key_path: &str, f: impl FnOnce(&mut PathCallCounts)) -> usize {
        let mut map = self.per_path.lock().unwrap();
        let entry = map.entry(key_path.to_string()).or_default();
        f(entry);
        entry.get_calls
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

    fn log_request(&self, req: RecordedRequest) {
        self.requests.lock().unwrap().push(req);
    }

    /// Every request received so far whose raw path or raw query contains
    /// `tag`, in arrival order. A test embeds a unique ASCII tag in its key,
    /// so this returns that test's traffic -- INCLUDING requests for variant
    /// paths (stripped, decoded, double-encoded) that `counts_for` cannot
    /// enumerate because it needs the exact path up front.
    pub fn requests_containing(&self, tag: &str) -> Vec<RecordedRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.path.contains(tag) || r.query.contains(tag))
            .cloned()
            .collect()
    }

    /// Every request received so far for EXACTLY the path-style bucket `bucket` (the first
    /// path segment), in arrival order. Use this, not [`Self::requests_containing`], to
    /// filter by bucket: `requests_containing("b-1")` is a substring match and also returns
    /// the traffic of `b-10` .. `b-19` created by tests running concurrently on the same mock.
    pub fn requests_for_bucket(&self, bucket: &str) -> Vec<RecordedRequest> {
        self.requests_containing(bucket)
            .into_iter()
            .filter(|r| r.path.trim_start_matches('/').split('/').next() == Some(bucket))
            .collect()
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
    let (port, state) = MOCK_SERVER.get_or_init(spawn_mock_server).clone();

    ENV_INIT.call_once(|| {
        std::env::set_var("AWS_ACCESS_KEY_ID", "mock-access-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "mock-secret-key");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"));
    });

    (port, state)
}

// ─────────────────────────── registered objects ───────────────────────────
//
// A registered object is served the way an S3 server serves a real one: the true size on
// HEAD, `Range` answered with 206 and a `Content-Range` that carries the total, 416 for a
// range that starts at or past the end (and for ANY range on an empty object), `If-Match`
// answered with 412 when it does not match the current ETag (ranged or not), 404 for an
// absent key (with or without `If-Match`), and quoted hex ETags. Scripted faults (see
// [`FaultRule`]) change one answer at a time. Objects are keyed by their raw request path,
// so a test registers `key_path(bucket, key)` and no marker is needed in the key.

/// The body type every answer of the mock server uses.
pub type MockBody = BoxBody<Bytes, std::io::Error>;

/// How long a streamed body waits before it aborts, so the bytes before the abort are flushed
/// to the client first (an abort straight after the first frame would drop buffered data).
const ABORT_DELAY: Duration = Duration::from_millis(40);

/// One scripted change to an answer.
#[derive(Clone, Debug)]
pub enum ObjectFault {
    /// Answer with this status and S3 error code instead of serving the request.
    Status(u16, &'static str),
    /// Serve a correct header (the full `Content-Length`) but only `send` bytes of the body,
    /// then abort the connection: a body that breaks mid-stream.
    TruncateBody { send: usize },
    /// Serve a correct header and `send` bytes of the body, then stall for ever: a read that
    /// never finishes (for cancellation tests).
    StallBody { send: usize },
    /// Serve a `Content-Range` of `bytes first-last/total`, whatever was asked (a 206 for a
    /// ranged GET; the header is added to the 200 of an unranged GET too).
    ContentRange { first: u64, last: u64, total: u64 },
    /// Answer without an `ETag` header (the object is otherwise served honestly).
    OmitEtag,
    /// Ignore the `Range` header: a 200 with the whole object and no `Content-Range`.
    IgnoreRange,
    /// Serve a body `delta` bytes longer (zero padding) or shorter than the answer should be,
    /// with the `Content-Length` of the body actually sent (none when `get_content_length` is
    /// false).  `Content-Range` is unchanged.  With a length header this is a
    /// Content-Length that disagrees with the range; without one, a body that ends clean but
    /// short, or runs long.
    BodyLength { delta: isize },
    /// Serve a 206 with no `Content-Range` header.
    NoContentRange,
    /// Replace the object's body and ETag BEFORE answering this request, as a concurrent
    /// writer would. The request is then answered against the new object.
    OverwriteFirst { body: Vec<u8>, etag: Option<String> },
}

/// A fault applied to the next `times` GETs that match `range_start`.
#[derive(Clone, Debug)]
pub struct FaultRule {
    /// Only a ranged GET whose first byte is this; `None` matches every GET.
    pub range_start: Option<u64>,
    /// How many matching GETs it applies to.
    pub times: usize,
    pub fault: ObjectFault,
}

/// What the mock serves for one registered key.
#[derive(Clone, Debug)]
pub struct MockObject {
    pub body: Vec<u8>,
    /// The `ETag` header (a quoted string); `None` omits it everywhere.
    pub etag: Option<String>,
    /// The `x-amz-version-id` header, if any.
    pub version_id: Option<String>,
    /// HEAD carries the ETag (only when `etag` is set).
    pub head_etag: bool,
    /// HEAD carries `Content-Length`.
    pub head_content_length: bool,
    /// GET carries `Content-Length` (when false the body is chunked and has no length).
    pub get_content_length: bool,
    /// HEAD waits this long before it answers (for cancellation tests).
    pub head_delay: Option<Duration>,
    /// HEAD fails with this status instead of describing the object.
    pub head_status: Option<u16>,
    /// `If-Match` is answered with 412 when it differs from the current ETag.
    pub honour_if_match: bool,
    /// The key does not exist: 404 on GET and HEAD, before any `If-Match` is looked at.
    pub missing: bool,
    pub faults: Vec<FaultRule>,
}

/// A deterministic quoted 32-digit hex ETag for `body` (FNV-1a, two lanes).
pub fn etag_for(body: &[u8]) -> String {
    let (mut a, mut b) = (0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64);
    for &byte in body {
        a = (a ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        b = (b.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("\"{a:016x}{b:016x}\"")
}

impl MockObject {
    /// An object holding `body`, with the ETag [`etag_for`] gives, a length on HEAD and GET,
    /// and `If-Match` honoured.
    pub fn new(body: impl Into<Vec<u8>>) -> Self {
        let body = body.into();
        Self {
            etag: Some(etag_for(&body)),
            body,
            version_id: None,
            head_etag: true,
            head_content_length: true,
            get_content_length: true,
            head_delay: None,
            head_status: None,
            honour_if_match: true,
            missing: false,
            faults: Vec::new(),
        }
    }

    /// A key that does not exist.
    pub fn missing() -> Self {
        Self {
            missing: true,
            ..Self::new(Vec::new())
        }
    }

    /// Append a scripted fault.
    pub fn with_fault(
        mut self,
        range_start: Option<u64>,
        times: usize,
        fault: ObjectFault,
    ) -> Self {
        self.faults.push(FaultRule {
            range_start,
            times,
            fault,
        });
        self
    }
}

impl MockS3State {
    /// Register (or replace) the object at `bucket`/`key`.
    pub fn put_object(&self, bucket: &str, key: &str, object: MockObject) {
        self.objects
            .lock()
            .unwrap()
            .insert(key_path(bucket, key), object);
    }

    /// Replace the body and ETag of a registered object, as a concurrent writer would.
    pub fn overwrite_object(&self, bucket: &str, key: &str, body: Vec<u8>, etag: Option<String>) {
        if let Some(o) = self.objects.lock().unwrap().get_mut(&key_path(bucket, key)) {
            o.body = body;
            o.etag = etag;
        }
    }

    /// Add a scripted fault to a registered object.
    pub fn add_fault(&self, bucket: &str, key: &str, rule: FaultRule) {
        if let Some(o) = self.objects.lock().unwrap().get_mut(&key_path(bucket, key)) {
            o.faults.push(rule);
        }
    }

    fn is_registered(&self, path: &str) -> bool {
        self.objects.lock().unwrap().contains_key(path)
    }

    /// Every request received for `bucket`/`key` so far, in arrival order.
    pub fn requests_for(&self, bucket: &str, key: &str) -> Vec<RecordedRequest> {
        let want = key_path(bucket, key);
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.path == want)
            .cloned()
            .collect()
    }

    /// Every request received at all, in arrival order.
    pub fn all_requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }
}

/// `bytes=a-b` / `bytes=a-` parsed to `(a, Some(b))` / `(a, None)`.
fn parse_range_header(header: &str) -> Option<(u64, Option<u64>)> {
    let spec = header.strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()))
}

fn full_body(bytes: Bytes) -> MockBody {
    Full::new(bytes).map_err(|never| match never {}).boxed()
}

fn boxed(resp: Response<Full<Bytes>>) -> Response<MockBody> {
    resp.map(|b| b.map_err(|never| match never {}).boxed())
}

/// How a streamed body ends after its data frame.
enum BodyEnd {
    Complete,
    /// Wait, then fail the stream (the connection is aborted).
    Abort,
    /// Never end.
    Stall,
}

/// A body that sends `data` as one frame (with no declared length if the response has none),
/// then ends as `end` says.
fn streamed_body(data: Bytes, end: BodyEnd) -> MockBody {
    let frames = async_stream::stream! {
        yield Ok::<_, std::io::Error>(Frame::data(data));
        match end {
            BodyEnd::Complete => {}
            BodyEnd::Abort => {
                tokio::time::sleep(ABORT_DELAY).await;
                yield Err(std::io::Error::other("mock server injected body abort"));
            }
            BodyEnd::Stall => std::future::pending::<()>().await,
        }
    };
    StreamBody::new(frames).boxed()
}

fn plain_error(status: u16, code: &str) -> Response<MockBody> {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    boxed(error_response(status, code))
}

/// Route one request: a registered object is served by [`serve_object`], everything else by
/// the marker-driven [`handle`].
async fn dispatch(req: Request<Incoming>, state: MockS3State) -> Response<MockBody> {
    let is_object_verb = req.method() == Method::GET || req.method() == Method::HEAD;
    let is_list = req.uri().query().is_some_and(|q| q.contains("list-type"));
    if is_object_verb && !is_list && state.is_registered(req.uri().path()) {
        if req.method() == Method::HEAD {
            let delay = state
                .objects
                .lock()
                .unwrap()
                .get(req.uri().path())
                .and_then(|o| o.head_delay);
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
        }
        return serve_object(req, state);
    }
    boxed(handle(req, state).await)
}

fn serve_object(req: Request<Incoming>, state: MockS3State) -> Response<MockBody> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let header = |name: &str| {
        req.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let range_header = header("range");
    let if_match = header("if-match");
    state.log_request(RecordedRequest {
        method: method.to_string(),
        path: path.clone(),
        query: req.uri().query().unwrap_or("").to_string(),
        range: range_header.clone(),
        host: header("host"),
        if_match: if_match.clone(),
    });
    let mut range = range_header.as_deref().and_then(parse_range_header);

    if method == Method::HEAD {
        state.record(&path, |c| c.head_calls += 1);
    } else {
        state.record(&path, |c| c.get_calls += 1);
    }

    // Decide under the lock, answer outside it.
    let mut objects = state.objects.lock().unwrap();
    let Some(obj) = objects.get_mut(&path) else {
        return plain_error(404, "NoSuchKey");
    };

    if method == Method::HEAD {
        if obj.missing {
            return boxed(
                Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Full::new(Bytes::new()))
                    .unwrap(),
            );
        }
        if let Some(status) = obj.head_status {
            return plain_error(status, "MockHeadFailure");
        }
        let mut b = Response::builder().status(StatusCode::OK);
        if obj.head_content_length {
            b = b.header("Content-Length", obj.body.len().to_string());
        }
        if let (true, Some(e)) = (obj.head_etag, &obj.etag) {
            b = b.header("ETag", e.as_str());
        }
        if let Some(v) = &obj.version_id {
            b = b.header("x-amz-version-id", v.as_str());
        }
        return boxed(b.body(Full::new(Bytes::new())).unwrap());
    }

    // GET. Scripted faults first (the first rule that still has uses and matches).
    let mut truncate: Option<(usize, BodyEnd)> = None;
    let mut forced_range: Option<Option<(u64, u64, u64)>> = None;
    let mut omit_etag = false;
    let mut body_delta: isize = 0;
    let start = range.map(|(a, _)| a);
    let mut action: Option<ObjectFault> = None;
    for rule in obj.faults.iter_mut() {
        if rule.times > 0 && (rule.range_start.is_none() || rule.range_start == start) {
            rule.times -= 1;
            action = Some(rule.fault.clone());
            break;
        }
    }
    if let Some(fault) = action {
        match fault {
            ObjectFault::Status(status, code) => return plain_error(status, code),
            ObjectFault::TruncateBody { send } => truncate = Some((send, BodyEnd::Abort)),
            ObjectFault::StallBody { send } => truncate = Some((send, BodyEnd::Stall)),
            ObjectFault::ContentRange { first, last, total } => {
                forced_range = Some(Some((first, last, total)))
            }
            ObjectFault::NoContentRange => forced_range = Some(None),
            ObjectFault::OmitEtag => omit_etag = true,
            ObjectFault::IgnoreRange => range = None,
            ObjectFault::BodyLength { delta } => body_delta = delta,
            ObjectFault::OverwriteFirst { body, etag } => {
                obj.body = body;
                obj.etag = etag;
            }
        }
    }

    if obj.missing {
        return plain_error(404, "NoSuchKey");
    }
    if let (true, Some(wanted), Some(current)) = (obj.honour_if_match, &if_match, &obj.etag) {
        if wanted != "*" && wanted != current {
            return plain_error(412, "PreconditionFailed");
        }
    }

    let len = obj.body.len() as u64;
    let mut b = Response::builder();
    if let (false, Some(e)) = (omit_etag, &obj.etag) {
        b = b.header("ETag", e.as_str());
    }
    if let Some(v) = &obj.version_id {
        b = b.header("x-amz-version-id", v.as_str());
    }

    let (status, mut slice, content_range) = match range {
        None => (
            StatusCode::OK,
            obj.body.clone(),
            forced_range
                .flatten()
                .map(|(f, l, t)| format!("bytes {f}-{l}/{t}")),
        ),
        Some((first, last)) => {
            if len == 0 || first >= len {
                return plain_error(416, "InvalidRange");
            }
            let last = last.map_or(len - 1, |l| l.min(len - 1));
            if last < first {
                return plain_error(416, "InvalidRange");
            }
            let slice = obj.body[first as usize..=last as usize].to_vec();
            let cr = match forced_range {
                Some(forced) => forced.map(|(f, l, t)| format!("bytes {f}-{l}/{t}")),
                None => Some(format!("bytes {first}-{last}/{len}")),
            };
            (StatusCode::PARTIAL_CONTENT, slice, cr)
        }
    };
    if let Some(cr) = content_range {
        b = b.header("Content-Range", cr);
    }
    b = b.status(status);

    // `BodyLength`: send a body that is not the length the answer describes.
    let new_len = slice.len().saturating_add_signed(body_delta);
    slice.resize(new_len, 0);
    let slice_len = slice.len();
    if let Some((send, end)) = truncate {
        let head = Bytes::copy_from_slice(&slice[..send.min(slice_len)]);
        let resp = b
            .header("Content-Length", slice_len.to_string())
            .body(streamed_body(head, end))
            .unwrap();
        return resp;
    }
    if !obj.get_content_length {
        // No Content-Length: a streamed body of unknown size is sent chunked.
        return b
            .body(streamed_body(Bytes::from(slice), BodyEnd::Complete))
            .unwrap();
    }
    b.header("Content-Length", slice_len.to_string())
        .body(full_body(Bytes::from(slice)))
        .unwrap()
}

/// Start a SEPARATE mock server (its own port and its own state) without touching the
/// process environment, for a test that needs a second endpoint.
pub fn start_isolated_mock_s3_server() -> (u16, MockS3State) {
    spawn_mock_server()
}

fn spawn_mock_server() -> (u16, MockS3State) {
    let state = MockS3State::default();
    let server_state = state.clone();
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::Builder::new()
        .name("mock-s3-server".to_string())
        .spawn(move || {
            let rt =
                tokio::runtime::Runtime::new().expect("failed to build mock-s3-server runtime");
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
                        async move { Ok::<_, std::convert::Infallible>(dispatch(req, st).await) }
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
    let range_header = req
        .headers()
        .get("range")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let if_match_header = req
        .headers()
        .get("if-match")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let host_header = req
        .headers()
        .get("host")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| req.uri().authority().map(|a| a.to_string()));
    state.log_request(RecordedRequest {
        method: method.to_string(),
        path: path.clone(),
        query: query.clone(),
        range: range_header.clone(),
        host: host_header,
        if_match: if_match_header,
    });
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
            state.record(&format!("{}/{key}", path.trim_end_matches('/')), |c| {
                c.batch_delete_calls += 1
            });
            if key.contains(DELETE_FAILS_MARKER) {
                error_xml.push_str(&format!(
                    "<Error><Key>{key}</Key><Code>AccessDenied</Code>\
                     <Message>mock server injected per-object delete failure</Message></Error>"
                ));
            } else {
                state.remove_key(path.trim_matches('/'), key);
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
    // A bucket seeded with `MockS3State::seed_bucket` is answered from its
    // catalog (prefix, delimiter and paging honoured, like S3). Every other
    // bucket gets exactly two synthetic keys under whatever prefix was
    // requested -- "{prefix}bar" and "{prefix}other/deep.dat" -- which older
    // tests rely on to see the prefix the client SENT.
    if method == Method::GET && query.contains("list-type=2") {
        let prefix = query_param(&query, "prefix").unwrap_or_default();
        let bucket = path.trim_matches('/').split('/').next().unwrap_or("");
        if let Some(catalog) = state.catalog(bucket) {
            let xml = seeded_list_response(&catalog, bucket, &prefix, &query);
            return xml_response(StatusCode::OK, &xml);
        }
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

    // DeleteObject: DELETE .../{key} (no uploadId -- that is the abort above).
    if method == Method::DELETE {
        state.record(&path, |c| c.delete_calls += 1);
        if let Some((bucket, key)) = path.trim_start_matches('/').split_once('/') {
            state.remove_key(bucket, &percent_decode(key));
        }
        return Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    // PutObject: PUT .../{key} (no partNumber -- that is UploadPart above).
    if method == Method::PUT {
        if path.contains(PUT_FORBIDDEN_MARKER) {
            state.record(&path, |c| c.put_calls += 1);
            return error_response(StatusCode::FORBIDDEN, "AccessDenied");
        }
        if path.contains(PUT_FAILS_MARKER) {
            state.record(&path, |c| c.put_calls += 1);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "MockPutFailure");
        }
        state.record(&path, |c| c.put_calls += 1);
        return Response::builder()
            .status(StatusCode::OK)
            .header("ETag", "\"mock-put-etag\"")
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

    // HeadObject (used by S3DLIO_MPU_PUT_VERIFY, and by concurrent-range
    // GET's size lookup — bug B3). A key containing NO_CONTENT_LENGTH_MARKER
    // gets a 200 with no Content-Length header at all, reproducing the
    // "HEAD succeeded but didn't report a size" scenario.
    if method == Method::HEAD {
        state.record(&path, |c| c.head_calls += 1);
        if path.contains(HEAD_FAILS_MARKER) {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "MockHeadObjectFailure");
        }
        if path.contains(HEAD_NOT_FOUND_MARKER) {
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Full::new(Bytes::new()))
                .unwrap();
        }
        if path.contains(NO_CONTENT_LENGTH_MARKER) {
            return Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::new()))
                .unwrap();
        }
        let head_len = if path.contains(SIZED_OBJECT_MARKER) {
            SIZED_OBJECT_BODY.len()
        } else {
            0
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Length", head_len.to_string())
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    // GetObject (plain or ranged): GET on a key path with no other
    // matched query shape. Tracked so tests can assert a fast-path
    // return (e.g. length=Some(0), bug B2) never reached the network.
    //
    // A key containing GET_FAILS_ONCE_MARKER returns a transient 500 on
    // its FIRST request and succeeds on every subsequent one — used to
    // prove a caller retries the whole send()+collect() pair (bug B1)
    // rather than literally reproducing a mid-body TCP truncation
    // (which Full<Bytes> response bodies can't under-deliver).
    if method == Method::GET {
        let attempt = state.record_and_get(&path, |c| c.get_calls += 1);
        if path.contains(GET_FORBIDDEN_MARKER) {
            return error_response(StatusCode::FORBIDDEN, "AccessDenied");
        }
        if path.contains(GET_NOT_FOUND_MARKER) {
            return error_response(StatusCode::NOT_FOUND, "NoSuchKey");
        }
        if path.contains(GET_UNAVAILABLE_MARKER) {
            return error_response(StatusCode::SERVICE_UNAVAILABLE, "SlowDown");
        }
        if path.contains(GET_TIMEOUT_THEN_OK_MARKER) && attempt <= GET_TIMEOUT_FAILURES {
            return error_response(StatusCode::BAD_REQUEST, "RequestTimeoutException");
        }
        if path.contains(GET_FAILS_ONCE_MARKER) && attempt == 1 {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "MockTransientGetFailure");
        }
        if path.contains(SIZED_OBJECT_MARKER) {
            return sized_object_get(range_header.as_deref());
        }
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Length", "4")
            .body(Full::new(Bytes::from_static(b"mock")))
            .unwrap();
    }

    // DeleteBucket / DeleteObject: any DELETE that is not an AbortMultipartUpload.
    if method == Method::DELETE {
        return Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    // PutObject (single shot) and CreateBucket: PUT on a path. The multipart UploadPart
    // PUT (query `partNumber`) was answered above.
    if method == Method::PUT {
        state.record(&path, |c| c.put_calls += 1);
        return Response::builder()
            .status(StatusCode::OK)
            .header("ETag", "\"mock-put-etag\"")
            .body(Full::new(Bytes::new()))
            .unwrap();
    }

    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Full::new(Bytes::from("mock-s3-server: unhandled route")))
        .unwrap()
}

/// GetObject for a [`SIZED_OBJECT_MARKER`] key: no Range header returns the
/// whole body (200); `bytes=a-b` / `bytes=a-` returns that inclusive slice
/// (206), clamped to the body like real S3.
fn sized_object_get(range_header: Option<&str>) -> Response<Full<Bytes>> {
    let body = SIZED_OBJECT_BODY;
    let spec = range_header.and_then(|h| h.strip_prefix("bytes="));
    let Some(spec) = spec else {
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Length", body.len().to_string())
            .body(Full::new(Bytes::from_static(body)))
            .unwrap();
    };
    let (start_s, end_s) = spec.split_once('-').unwrap_or((spec, ""));
    let start = start_s.parse::<usize>().unwrap_or(0);
    let last = body.len() - 1;
    let end = end_s.parse::<usize>().map(|e| e.min(last)).unwrap_or(last);
    if start > end {
        return error_response(StatusCode::RANGE_NOT_SATISFIABLE, "InvalidRange");
    }
    let slice = &body[start..=end];
    Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header("Content-Length", slice.len().to_string())
        .header(
            "Content-Range",
            format!("bytes {start}-{end}/{}", body.len()),
        )
        .body(Full::new(Bytes::copy_from_slice(slice)))
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

/// The decoded value of query parameter `name`, if present.
fn query_param(query: &str, name: &str) -> Option<String> {
    let want = format!("{name}=");
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix(want.as_str()))
        .map(percent_decode)
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// A ListObjectsV2 answer built from a seeded, sorted catalog (see
/// [`MockS3State::seed_bucket`]).
fn seeded_list_response(catalog: &[String], bucket: &str, prefix: &str, query: &str) -> String {
    let delimiter = query_param(query, "delimiter").filter(|d| !d.is_empty());
    let max_keys: usize = query_param(query, "max-keys")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let token = query_param(query, "continuation-token");

    // (name, is_common_prefix), in key order, common prefixes collapsed.
    let mut entries: Vec<(String, bool)> = Vec::new();
    for key in catalog {
        let Some(rest) = key.strip_prefix(prefix) else {
            continue;
        };
        match delimiter
            .as_deref()
            .and_then(|d| rest.find(d).map(|i| (d, i)))
        {
            Some((d, i)) => {
                let common = format!("{prefix}{}", &rest[..i + d.len()]);
                if entries.last().map(|(n, _)| n.as_str()) != Some(common.as_str()) {
                    entries.push((common, true));
                }
            }
            None => entries.push((key.clone(), false)),
        }
    }
    if let Some(after) = &token {
        entries.retain(|(name, _)| name.as_str() > after.as_str());
    }
    let truncated = entries.len() > max_keys;
    entries.truncate(max_keys);

    let mut contents = String::new();
    let mut common_prefixes = String::new();
    for (name, is_common) in &entries {
        if *is_common {
            common_prefixes.push_str(&format!(
                "<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>",
                xml_escape(name)
            ));
        } else {
            contents.push_str(&format!(
                "<Contents><Key>{}</Key><Size>4</Size></Contents>",
                xml_escape(name)
            ));
        }
    }
    let next = match (truncated, entries.last()) {
        _ if bucket.contains(TRUNCATED_NO_TOKEN_MARKER) => String::new(),
        (false, _) if bucket.contains(LAST_PAGE_TOKEN_MARKER) => {
            "<NextContinuationToken>constant-last-page-token</NextContinuationToken>".to_string()
        }
        (true, Some((name, _))) => {
            format!(
                "<NextContinuationToken>{}</NextContinuationToken>",
                xml_escape(name)
            )
        }
        _ => String::new(),
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult>\
         <Name>{bucket}</Name><Prefix>{}</Prefix><KeyCount>{}</KeyCount>\
         <MaxKeys>{max_keys}</MaxKeys><IsTruncated>{truncated}</IsTruncated>{next}\
         {contents}{common_prefixes}</ListBucketResult>",
        xml_escape(prefix),
        entries.len()
    )
}
