// tests/test_issue208_per_endpoint_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #208: a per-endpoint `S3ObjectStore` (one that
// holds its own client, e.g. `S3ObjectStore::for_endpoint(<B>)`) must run EVERY
// operation against ITS server, whatever the size of the object:
//
//   * `get_optimized` / `get_range_optimized` above the concurrent-range threshold,
//   * `put_multipart` (CreateMultipartUpload, UploadPart, CompleteMultipartUpload and
//     -- on a failure -- AbortMultipartUpload),
//   * the final upload of a buffered writer (`get_writer`, `get_writer_with_compression`,
//     `create_writer`).
//
// Before the fix these paths used the process-wide client (the server
// `AWS_ENDPOINT_URL` names) and, after #197, were REJECTED with a "not yet supported"
// error. They now route to the store's own server.
//
//   A = the process-wide (global) client's server (`AWS_ENDPOINT_URL`)
//   B = the server a per-endpoint store talks to
//   C = a third server, named by a URI, that the store must refuse (#197 stays)
//
// Both A and B are in-process mock S3 servers that really serve objects (HEAD, ranged
// GET) and really implement multipart (the parts are kept and joined on Complete), so
// the tests check the BYTES, not only where the requests went. The two servers serve
// DIFFERENT data (a different seed), so a read that reached the wrong server cannot
// return the right bytes by accident.
//
// One global mock A for the whole binary (the library's global client is built once,
// from the environment); every test uses bucket names unique to it.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::thread;

use bytes::Bytes;
use s3dlio::object_store::{CompressionConfig, ObjectStore, S3ObjectStore, WriterOptions};

// ----------------------------------------------------------------------------
// The mock
// ----------------------------------------------------------------------------

/// One request as the mock received it.
#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    query: String,
    range: Option<String>,
}

#[derive(Default)]
struct State {
    requests: Vec<Req>,
    /// upload id -> (object path, parts by part number)
    uploads: HashMap<String, (String, BTreeMap<i32, Vec<u8>>)>,
    /// object path -> the bytes of its completed multipart upload
    completed: HashMap<String, Vec<u8>>,
    next_upload: usize,
}

/// A mock S3 server. Every object it is asked for is `object_size` bytes of
/// `pattern(seed, i)`. A key containing `part-fails` makes every UploadPart a 500.
struct ObjMock {
    port: u16,
    state: Arc<Mutex<State>>,
}

fn pattern(seed: u8, i: u64) -> u8 {
    ((i as u32).wrapping_mul(31).wrapping_add(u32::from(seed)) % 251) as u8
}

/// The bytes `[offset, offset + len)` of an object served by a mock with `seed`.
fn expected(seed: u8, offset: u64, len: u64) -> Vec<u8> {
    (offset..offset + len).map(|i| pattern(seed, i)).collect()
}

impl ObjMock {
    fn start(seed: u8, object_size: u64) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock S3 listener");
        let port = listener.local_addr().expect("mock S3 local addr").port();
        let state = Arc::new(Mutex::new(State::default()));
        let shared = Arc::clone(&state);
        // The thread (and the listener) live for the rest of the test process.
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let shared = Arc::clone(&shared);
                thread::spawn(move || serve_one(stream, shared, seed, object_size));
            }
        });
        ObjMock { port, state }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Every request whose path is inside `bucket`, in arrival order.
    fn for_bucket(&self, bucket: &str) -> Vec<Req> {
        let prefix = format!("/{bucket}/");
        self.state
            .lock()
            .map(|s| {
                s.requests
                    .iter()
                    .filter(|r| r.path.starts_with(&prefix) || r.path == format!("/{bucket}"))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All requests, any bucket.
    fn all(&self) -> Vec<Req> {
        self.state
            .lock()
            .map(|s| s.requests.clone())
            .unwrap_or_default()
    }

    /// The bytes of the completed multipart upload of `/bucket/key`, if any.
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

fn serve_one(mut stream: TcpStream, shared: Arc<Mutex<State>>, seed: u8, object_size: u64) {
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
            range,
        },
        body,
        seed,
        object_size,
    );
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

fn respond(
    shared: &Arc<Mutex<State>>,
    req: Req,
    body: Vec<u8>,
    seed: u8,
    object_size: u64,
) -> Vec<u8> {
    let Ok(mut st) = shared.lock() else {
        return xml(
            "500 Internal Server Error",
            "<Error><Code>Poisoned</Code></Error>",
        );
    };
    st.requests.push(req.clone());
    let path = req.path.clone();
    let bucket_and_key = path.trim_start_matches('/');
    let (bucket, key) = bucket_and_key
        .split_once('/')
        .unwrap_or((bucket_and_key, ""));

    match req.method.as_str() {
        // CreateMultipartUpload / CompleteMultipartUpload
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
        // UploadPart
        "PUT" if query_param(&req.query, "partNumber").is_some() => {
            if path.contains("part-fails") {
                return xml(
                    "500 Internal Server Error",
                    "<Error><Code>MockUploadPartFailure</Code></Error>",
                );
            }
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
        // AbortMultipartUpload
        "DELETE" if query_param(&req.query, "uploadId").is_some() => {
            let id = query_param(&req.query, "uploadId").unwrap_or("");
            st.uploads.remove(id);
            http("204 No Content", &[], b"")
        }
        "HEAD" => http(
            "200 OK",
            &[("Content-Length", object_size.to_string())],
            b"",
        ),
        "GET" => {
            let spec = req.range.as_deref().and_then(|r| r.strip_prefix("bytes="));
            let Some(spec) = spec else {
                let all = expected(seed, 0, object_size);
                return http("200 OK", &[("Content-Length", all.len().to_string())], &all);
            };
            let (start_s, end_s) = spec.split_once('-').unwrap_or((spec, ""));
            let start = start_s.parse::<u64>().unwrap_or(0);
            let last = object_size.saturating_sub(1);
            let end = end_s.parse::<u64>().map(|e| e.min(last)).unwrap_or(last);
            if start > end {
                return xml(
                    "416 Range Not Satisfiable",
                    "<Error><Code>InvalidRange</Code></Error>",
                );
            }
            let slice = expected(seed, start, end - start + 1);
            http(
                "206 Partial Content",
                &[
                    ("Content-Length", slice.len().to_string()),
                    (
                        "Content-Range",
                        format!("bytes {start}-{end}/{object_size}"),
                    ),
                ],
                &slice,
            )
        }
        _ => http("200 OK", &[("Content-Length", "0".to_string())], b""),
    }
}

// ----------------------------------------------------------------------------
// Fixture
// ----------------------------------------------------------------------------

const OBJECT_SIZE: u64 = 1024 * 1024;
const SEED_A: u8 = 11;
const SEED_B: u8 = 77;
/// The concurrent-range threshold the tests set: a 1 MiB object is far above it.
const THRESHOLD: u64 = 64 * 1024;
/// Range chunk size: a 1 MiB object splits into 8 ranged GETs.
const CHUNK: u64 = 128 * 1024;

static GLOBAL_A: OnceLock<ObjMock> = OnceLock::new();
static ENV: Once = Once::new();
static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A, the server `AWS_ENDPOINT_URL` names. The environment is set before anything
/// touches the process-wide client (it is built once from it).
fn global_a() -> &'static ObjMock {
    ENV.call_once(|| {
        let a = ObjMock::start(SEED_A, OBJECT_SIZE);
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
            std::env::remove_var("AWS_ENDPOINT_URL_S3");
            std::env::set_var("AWS_ENDPOINT_URL", a.url());
            // A small object is "large": the concurrent-range path is taken above 64 KiB.
            std::env::set_var("S3DLIO_CONCURRENT_THRESHOLD", THRESHOLD.to_string());
            std::env::set_var("S3DLIO_CHUNK_SIZE", CHUNK.to_string());
            // A failing part must fail fast, not after SDK back-off.
            std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "1");
        }
        let _ = GLOBAL_A.set(a);
    });
    GLOBAL_A.get().expect("global mock set by call_once")
}

fn unique_bucket(tag: &str) -> String {
    format!("b208-{tag}-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

async fn store_for(server: &ObjMock) -> S3ObjectStore {
    global_a();
    S3ObjectStore::for_endpoint(&server.url())
        .await
        .expect("for_endpoint with fake credentials")
}

/// `n` bytes of a pattern that is not periodic at any part boundary.
fn payload(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i.wrapping_mul(7) ^ (i >> 8)) % 253) as u8)
        .collect()
}

fn count(reqs: &[Req], method: &str, query_has: &str) -> usize {
    reqs.iter()
        .filter(|r| r.method == method && r.query.contains(query_has))
        .count()
}

/// The two URI spellings that name the store's own server: bucket form and host form.
fn uri_forms(b: &ObjMock, bucket: &str, key: &str) -> [(&'static str, String); 2] {
    [
        ("bucket-form", format!("s3://{bucket}/{key}")),
        (
            "host-form",
            format!("s3://127.0.0.1:{}/{bucket}/{key}", b.port),
        ),
    ]
}

// ----------------------------------------------------------------------------
// (1) get_optimized above the concurrent-range threshold
// ----------------------------------------------------------------------------

#[tokio::test]
async fn get_optimized_above_threshold_reaches_only_the_stores_server() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;

    for (form, tag) in [("bucket-form", "get-bf"), ("host-form", "get-hf")] {
        let bucket = unique_bucket(tag);
        let uri = if form == "bucket-form" {
            format!("s3://{bucket}/obj")
        } else {
            format!("s3://127.0.0.1:{}/{bucket}/obj", b.port)
        };

        let got = store.get_optimized(&uri).await.unwrap_or_else(|e| {
            panic!("{form}: get_optimized must reach the store's server: {e:#}")
        });

        assert_eq!(got.len() as u64, OBJECT_SIZE, "{form}: length");
        assert_eq!(
            got.as_ref(),
            expected(SEED_B, 0, OBJECT_SIZE).as_slice(),
            "{form}: bytes must be B's"
        );
        assert_eq!(
            a.for_bucket(&bucket).len(),
            0,
            "{form}: A got a request: {:?}",
            a.for_bucket(&bucket)
        );
        let reqs = b.for_bucket(&bucket);
        let ranged = reqs
            .iter()
            .filter(|r| r.method == "GET" && r.range.is_some())
            .count();
        assert_eq!(
            ranged,
            (OBJECT_SIZE / CHUNK) as usize,
            "{form}: every ranged GET must reach B, got {reqs:?}"
        );
        assert_eq!(
            reqs.iter().filter(|r| r.method == "GET").count(),
            ranged,
            "{form}: no un-ranged GET expected on the concurrent path, got {reqs:?}"
        );
    }
}

// ----------------------------------------------------------------------------
// (2) get_range_optimized above the concurrent-range threshold
// ----------------------------------------------------------------------------

#[tokio::test]
async fn get_range_optimized_above_threshold_reaches_only_the_stores_server() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;

    // An explicit length, an explicit chunk and concurrency.
    let bucket = unique_bucket("rng-len");
    let (offset, length) = (100_000u64, 500_000u64);
    let got = store
        .get_range_optimized(
            &format!("s3://{bucket}/obj"),
            offset,
            Some(length),
            Some(100_000),
            Some(3),
        )
        .await
        .expect("get_range_optimized must reach the store's server");
    assert_eq!(
        got.as_ref(),
        expected(SEED_B, offset, length).as_slice(),
        "bytes must be B's, from the requested range"
    );
    assert_eq!(a.for_bucket(&bucket).len(), 0, "A got a request");
    let ranged = b
        .for_bucket(&bucket)
        .iter()
        .filter(|r| r.method == "GET" && r.range.is_some())
        .count();
    assert_eq!(ranged, 5, "500000 bytes in 100000-byte ranges, all on B");

    // No length: the size comes from a HEAD, which must also go to B.
    let bucket = unique_bucket("rng-nolen");
    let offset = 200_000u64;
    let got = store
        .get_range_optimized(
            &format!("s3://127.0.0.1:{}/{bucket}/obj", b.port),
            offset,
            None,
            None,
            None,
        )
        .await
        .expect("get_range_optimized (host form, no length) must reach the store's server");
    assert_eq!(
        got.as_ref(),
        expected(SEED_B, offset, OBJECT_SIZE - offset).as_slice(),
        "bytes must be B's, to the end of the object"
    );
    assert_eq!(a.for_bucket(&bucket).len(), 0, "A got a request");
    assert!(
        count(&b.for_bucket(&bucket), "HEAD", "") >= 1,
        "B must get the size lookup"
    );
}

// ----------------------------------------------------------------------------
// (2a) the single-range fallback inside the concurrent helper
// ----------------------------------------------------------------------------

/// A transfer at or above the concurrent threshold but no larger than ONE chunk
/// takes the helper's single-range fallback (`total_bytes <= effective_chunk_size`),
/// not the multi-range split and not the small-read path:
///
///   * `length = 100_000` is >= `THRESHOLD` (64 KiB), so `get_range_optimized`
///     enters `get_object_concurrent_range_with_client` (a smaller length would be
///     served by `get_range` without ever reaching the helper);
///   * `chunk_size = CHUNK` (128 KiB) is >= `length`, so the helper issues ONE
///     ranged GET through `get_object_range_with_client` instead of splitting.
///
/// That fallback must use the store's own client: A (the process-wide client's
/// server) gets nothing, B gets the HEAD and exactly one ranged GET.
#[tokio::test]
async fn the_single_range_fallback_reaches_only_the_stores_server() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;

    let bucket = unique_bucket("rng-single");
    let (offset, length) = (100_000u64, 100_000u64);
    assert!(
        (THRESHOLD..=CHUNK).contains(&length),
        "the parameters must reach the single-range fallback"
    );
    let got = store
        .get_range_optimized(
            &format!("s3://{bucket}/obj"),
            offset,
            Some(length),
            Some(CHUNK as usize),
            Some(3),
        )
        .await
        .expect("the single-range fallback must reach the store's server");

    assert_eq!(
        a.for_bucket(&bucket).len(),
        0,
        "A got a request: {:?}",
        a.for_bucket(&bucket)
    );
    assert!(
        got.as_ref() == expected(SEED_B, offset, length).as_slice(),
        "bytes must be B's, from the requested range (got {} bytes)",
        got.len()
    );
    let reqs = b.for_bucket(&bucket);
    assert!(
        count(&reqs, "HEAD", "") >= 1,
        "B must get the size lookup, got {reqs:?}"
    );
    let gets: Vec<&Req> = reqs.iter().filter(|r| r.method == "GET").collect();
    assert_eq!(
        gets.len(),
        1,
        "exactly one GET (single range, not a split), got {reqs:?}"
    );
    assert!(
        gets[0].range.is_some(),
        "the one GET must be a ranged GET, got {reqs:?}"
    );
}

// ----------------------------------------------------------------------------
// (2b) the server must not depend on the object size
// ----------------------------------------------------------------------------

#[tokio::test]
async fn the_selected_server_is_the_same_below_and_above_the_threshold() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;

    // Below the threshold: an explicit length of 1 KiB takes the plain ranged GET.
    let small = unique_bucket("small");
    let got = store
        .get_range_optimized(&format!("s3://{small}/obj"), 10, Some(1024), None, None)
        .await
        .expect("a small range read");
    assert_eq!(got.as_ref(), expected(SEED_B, 10, 1024).as_slice());

    // Above it: the concurrent path.
    let large = unique_bucket("large");
    let got = store
        .get_range_optimized(
            &format!("s3://{large}/obj"),
            10,
            Some(OBJECT_SIZE - 10),
            None,
            None,
        )
        .await
        .expect("a large range read");
    assert_eq!(
        got.as_ref(),
        expected(SEED_B, 10, OBJECT_SIZE - 10).as_slice()
    );

    for bucket in [&small, &large] {
        assert_eq!(
            a.for_bucket(bucket).len(),
            0,
            "A got a request for {bucket}"
        );
        assert!(
            b.for_bucket(bucket)
                .iter()
                .any(|r| r.method == "GET" && r.range.is_some()),
            "B must serve the ranged GETs of {bucket}"
        );
    }
}

// ----------------------------------------------------------------------------
// (3) put_multipart
// ----------------------------------------------------------------------------

#[tokio::test]
async fn put_multipart_reaches_only_the_stores_server() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;
    let part = 5 * 1024 * 1024;
    let data = payload(2 * part + 1_234_567);

    for (form, tag) in [("bucket-form", "mp-bf"), ("host-form", "mp-hf")] {
        let bucket = unique_bucket(tag);
        let key = "obj";
        let uri = uri_forms(&b, &bucket, key)
            .into_iter()
            .find(|(f, _)| *f == form)
            .map(|(_, u)| u)
            .expect("form");

        store
            .put_multipart(&uri, Bytes::from(data.clone()), Some(part))
            .await
            .unwrap_or_else(|e| panic!("{form}: put_multipart must run on B: {e:#}"));

        assert_eq!(a.for_bucket(&bucket).len(), 0, "{form}: A got a request");
        let reqs = b.for_bucket(&bucket);
        assert_eq!(
            count(&reqs, "POST", "uploads"),
            1,
            "{form}: Create on B: {reqs:?}"
        );
        assert_eq!(
            count(&reqs, "PUT", "partNumber"),
            3,
            "{form}: 3 parts on B: {reqs:?}"
        );
        assert_eq!(
            count(&reqs, "POST", "uploadId"),
            1,
            "{form}: Complete on B: {reqs:?}"
        );
        assert_eq!(count(&reqs, "DELETE", ""), 0, "{form}: no Abort on success");
        assert_eq!(
            b.completed(&bucket, key).as_deref(),
            Some(data.as_slice()),
            "{form}: the object B assembled must equal what was uploaded"
        );
    }
}

// ----------------------------------------------------------------------------
// (3b) the buffered writer's final upload
// ----------------------------------------------------------------------------

#[tokio::test]
async fn buffered_writers_upload_to_the_stores_server() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;
    // The writer uploads with the default 16 MiB part size: 17 MiB is two parts.
    let large = payload(17 * 1024 * 1024);

    for name in [
        "get_writer",
        "get_writer_with_compression",
        "create_writer",
        "create_writer_compressed",
    ] {
        let bucket = unique_bucket(name);
        let uri = format!("s3://{bucket}/obj");
        let compressed = name.ends_with("compression") || name.ends_with("compressed");
        let data = &large;
        let mut writer = match name {
            "get_writer" => store.get_writer(&uri).await,
            "get_writer_with_compression" => {
                store
                    .get_writer_with_compression(&uri, CompressionConfig::zstd_default())
                    .await
            }
            "create_writer" => store.create_writer(&uri, WriterOptions::default()).await,
            _ => {
                let options = WriterOptions {
                    compression: Some(CompressionConfig::zstd_default()),
                    ..WriterOptions::default()
                };
                store.create_writer(&uri, options).await
            }
        }
        .unwrap_or_else(|e| panic!("{name}: the writer must be created: {e:#}"));

        for chunk in data.chunks(3 * 1024 * 1024) {
            writer
                .write_chunk(chunk)
                .await
                .unwrap_or_else(|e| panic!("{name}: write_chunk: {e:#}"));
        }
        writer
            .finalize()
            .await
            .unwrap_or_else(|e| panic!("{name}: finalize must upload to B: {e:#}"));

        assert_eq!(a.for_bucket(&bucket).len(), 0, "{name}: A got a request");
        let reqs = b.for_bucket(&bucket);
        assert_eq!(
            count(&reqs, "POST", "uploads"),
            1,
            "{name}: Create on B: {reqs:?}"
        );
        assert!(
            count(&reqs, "PUT", "partNumber") >= 1,
            "{name}: parts on B: {reqs:?}"
        );
        assert_eq!(
            count(&reqs, "POST", "uploadId"),
            1,
            "{name}: Complete on B: {reqs:?}"
        );

        let (key, stored) = if compressed {
            let stored = b.completed(&bucket, "obj.zst");
            ("obj.zst", stored)
        } else {
            ("obj", b.completed(&bucket, "obj"))
        };
        let stored = stored.unwrap_or_else(|| panic!("{name}: B has no completed /{bucket}/{key}"));
        let plain = if compressed {
            zstd::decode_all(stored.as_slice()).expect("zstd stream written by the writer")
        } else {
            stored
        };
        assert_eq!(
            &plain, data,
            "{name}: the uploaded object must equal the written data"
        );
    }
}

// ----------------------------------------------------------------------------
// (4) a failing part aborts the upload on the SAME server
// ----------------------------------------------------------------------------

#[tokio::test]
async fn a_failing_part_aborts_the_upload_on_the_stores_server() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;
    let part = 5 * 1024 * 1024;
    let data = payload(2 * part + 17);

    let bucket = unique_bucket("abort-mp");
    let err = store
        .put_multipart(
            &format!("s3://{bucket}/part-fails"),
            Bytes::from(data.clone()),
            Some(part),
        )
        .await
        .expect_err("every UploadPart fails");
    assert!(
        !format!("{err:#}").contains("not yet supported"),
        "the failure must be the part's, got: {err:#}"
    );
    assert_eq!(a.for_bucket(&bucket).len(), 0, "A got a request");
    let reqs = b.for_bucket(&bucket);
    assert_eq!(count(&reqs, "POST", "uploads"), 1, "Create on B: {reqs:?}");
    assert!(
        count(&reqs, "PUT", "partNumber") >= 1,
        "UploadPart on B: {reqs:?}"
    );
    assert_eq!(
        count(&reqs, "DELETE", "uploadId"),
        1,
        "the Abort must reach B, the server that has the upload: {reqs:?}"
    );
    assert_eq!(
        count(&reqs, "POST", "uploadId"),
        0,
        "a failed upload must never be completed: {reqs:?}"
    );
    assert!(b.completed(&bucket, "part-fails").is_none());

    // The same through the buffered writer.
    let bucket = unique_bucket("abort-wr");
    let mut writer = store
        .get_writer(&format!("s3://{bucket}/part-fails"))
        .await
        .expect("writer");
    writer.write_chunk(&data).await.expect("buffered");
    writer.finalize().await.expect_err("every UploadPart fails");
    assert_eq!(a.for_bucket(&bucket).len(), 0, "A got a request (writer)");
    let reqs = b.for_bucket(&bucket);
    assert_eq!(
        count(&reqs, "DELETE", "uploadId"),
        1,
        "the writer's Abort must reach B: {reqs:?}"
    );
    assert_eq!(
        count(&reqs, "POST", "uploadId"),
        0,
        "never completed: {reqs:?}"
    );
}

// ----------------------------------------------------------------------------
// (5) regression guards: the global server is still served as before
// ----------------------------------------------------------------------------

#[tokio::test]
async fn a_store_bound_to_the_global_server_and_the_default_store_still_use_it() {
    let a = global_a();
    let part = 5 * 1024 * 1024;
    let data = payload(part + 99);

    let bound = S3ObjectStore::for_endpoint(&a.url())
        .await
        .expect("for_endpoint with fake credentials");
    let default = S3ObjectStore::new();

    for (name, store) in [("for_endpoint(A)", &bound), ("default store", &default)] {
        let tag = name.replace(['(', ')', ' '], "-");

        // multipart
        let bucket = unique_bucket(&format!("g-mp-{tag}"));
        store
            .put_multipart(
                &format!("s3://{bucket}/obj"),
                Bytes::from(data.clone()),
                Some(part),
            )
            .await
            .unwrap_or_else(|e| panic!("{name}: put_multipart: {e:#}"));
        assert_eq!(
            a.completed(&bucket, "obj").as_deref(),
            Some(data.as_slice()),
            "{name}: multipart bytes"
        );

        // writer
        let bucket = unique_bucket(&format!("g-wr-{tag}"));
        let mut writer = store
            .get_writer(&format!("s3://{bucket}/obj"))
            .await
            .unwrap_or_else(|e| panic!("{name}: get_writer: {e:#}"));
        writer.write_chunk(&data).await.expect("buffered");
        writer
            .finalize()
            .await
            .unwrap_or_else(|e| panic!("{name}: finalize: {e:#}"));
        assert_eq!(
            a.completed(&bucket, "obj").as_deref(),
            Some(data.as_slice()),
            "{name}: writer bytes"
        );

        // large GET and range GET
        let bucket = unique_bucket(&format!("g-get-{tag}"));
        let got = store
            .get_optimized(&format!("s3://{bucket}/obj"))
            .await
            .unwrap_or_else(|e| panic!("{name}: get_optimized: {e:#}"));
        assert_eq!(
            got.as_ref(),
            expected(SEED_A, 0, OBJECT_SIZE).as_slice(),
            "{name}: GET"
        );
        let ranged = a
            .for_bucket(&bucket)
            .iter()
            .filter(|r| r.method == "GET" && r.range.is_some())
            .count();
        assert_eq!(
            ranged,
            (OBJECT_SIZE / CHUNK) as usize,
            "{name}: ranged GETs on A"
        );

        let bucket = unique_bucket(&format!("g-rng-{tag}"));
        let got = store
            .get_range_optimized(&format!("s3://{bucket}/obj"), 5, Some(700_000), None, None)
            .await
            .unwrap_or_else(|e| panic!("{name}: get_range_optimized: {e:#}"));
        assert_eq!(
            got.as_ref(),
            expected(SEED_A, 5, 700_000).as_slice(),
            "{name}: range"
        );
    }
}

// ----------------------------------------------------------------------------
// (6) #197 stays: a URI naming a DIFFERENT server than the store's is refused
// ----------------------------------------------------------------------------

#[tokio::test]
async fn a_uri_naming_a_different_server_is_still_rejected() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let c = ObjMock::start(33, OBJECT_SIZE);
    let store = store_for(&b).await;
    let bucket = unique_bucket("other");

    // C is a third server; A is the global server, which is not THIS store's server.
    for (who, port) in [("C", c.port), ("A", a.port)] {
        let uri = format!("s3://127.0.0.1:{port}/{bucket}/obj");
        let data = Bytes::from(payload(1024));
        let mut errors = vec![
            (
                "put_multipart",
                store
                    .put_multipart(&uri, data.clone(), None)
                    .await
                    .expect_err("a different server must be refused"),
            ),
            (
                "get_optimized",
                store
                    .get_optimized(&uri)
                    .await
                    .expect_err("a different server must be refused"),
            ),
            (
                "get_range_optimized",
                store
                    .get_range_optimized(&uri, 0, Some(OBJECT_SIZE), None, None)
                    .await
                    .expect_err("a different server must be refused"),
            ),
        ];
        for (name, result) in [
            ("get_writer", store.get_writer(&uri).await),
            (
                "get_writer_with_compression",
                store
                    .get_writer_with_compression(&uri, CompressionConfig::zstd_default())
                    .await,
            ),
            (
                "create_writer",
                store.create_writer(&uri, WriterOptions::default()).await,
            ),
        ] {
            match result {
                Ok(_) => panic!("{name} on a URI naming server {who}: must be refused"),
                Err(e) => errors.push((name, e)),
            }
        }
        for (name, err) in errors {
            let text = format!("{err:#}");
            assert!(
                text.contains("cannot be used with this store"),
                "{name} on a URI naming server {who}: expected the #197 refusal, got: {text}"
            );
        }
    }

    assert_eq!(c.all().len(), 0, "C got a request");
    assert_eq!(
        b.for_bucket(&bucket).len(),
        0,
        "B got a request for a URI it was not named in"
    );
    assert_eq!(a.for_bucket(&bucket).len(), 0, "A got a request");
}

// ----------------------------------------------------------------------------
// (6) issue #218: a compressed buffered writer must upload the WHOLE zstd frame
// ----------------------------------------------------------------------------
//
// zstd emits compressed blocks while it is still being fed (once a block of input is
// full), not only at `finish()`.  A writer that keeps only the encoder's final tail
// uploads an object with no frame header and no early blocks: it cannot be decoded.
// Small payloads hide that (nothing is emitted before finish), so these tests write
// several MiB in many chunks, through the process-wide store and a per-endpoint
// store, fetch the bytes the server really stored, and decode them.

/// `n` bytes: alternating 64 KiB stretches of incompressible pseudo-random bytes and
/// highly compressible repeated text, so zstd emits output of both kinds early.
fn mixed_payload(n: usize, incompressible_only: bool) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let text = b"the quick brown fox jumps over the lazy dog; ";
    let mut out = Vec::with_capacity(n);
    let mut random_stretch = true;
    while out.len() < n {
        let stretch = (64 * 1024).min(n - out.len());
        if random_stretch || incompressible_only {
            out.extend((0..stretch).map(|_| (next() >> 24) as u8));
        } else {
            out.extend(text.iter().cycle().take(stretch));
        }
        random_stretch = !random_stretch;
    }
    out
}

/// `stored` must be exactly ONE complete zstd frame (no truncation, no trailing
/// bytes) that decodes to `data`, byte for byte.
fn assert_complete_frame_of(name: &str, stored: &[u8], data: &[u8]) {
    let frame = zstd::zstd_safe::find_frame_compressed_size(stored).unwrap_or_else(|e| {
        panic!(
            "{name}: the {} stored bytes are not a complete zstd frame: {}",
            stored.len(),
            zstd::zstd_safe::get_error_name(e)
        )
    });
    assert_eq!(
        frame,
        stored.len(),
        "{name}: the first frame must be the whole stored object"
    );
    let plain = zstd::decode_all(stored)
        .unwrap_or_else(|e| panic!("{name}: the stored object does not decode: {e}"));
    assert_eq!(
        plain.len(),
        data.len(),
        "{name}: decoded length differs from the written length"
    );
    assert!(
        plain == data,
        "{name}: the decoded bytes differ from the written bytes"
    );
}

/// Write `data` in 64 KiB chunks through a compressed writer of `store` for `uri`, via
/// the named API, and finalize it.
async fn write_compressed(store: &S3ObjectStore, api: &str, uri: &str, data: &[u8]) {
    let mut writer = match api {
        "get_writer_with_compression" => {
            store
                .get_writer_with_compression(uri, CompressionConfig::zstd_default())
                .await
        }
        _ => {
            let options = WriterOptions {
                compression: Some(CompressionConfig::zstd_default()),
                ..WriterOptions::default()
            };
            store.create_writer(uri, options).await
        }
    }
    .unwrap_or_else(|e| panic!("{api}: the writer must be created: {e:#}"));
    for chunk in data.chunks(64 * 1024) {
        writer
            .write_chunk(chunk)
            .await
            .unwrap_or_else(|e| panic!("{api}: write_chunk: {e:#}"));
    }
    writer
        .finalize()
        .await
        .unwrap_or_else(|e| panic!("{api}: finalize: {e:#}"));
}

/// (name, payload).  `small` is the control that always worked; `mixed` is far past the
/// point where zstd emits output before finish; `multipart` is incompressible, so the
/// compressed object is bigger than the writer's 16 MiB part size and the upload is
/// two parts.
fn issue218_payloads() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("small", mixed_payload(32 * 1024, false)),
        ("mixed", mixed_payload(6 * 1024 * 1024 + 123, false)),
        ("multipart", mixed_payload(18 * 1024 * 1024 + 7, true)),
    ]
}

#[tokio::test]
async fn compressed_writer_uploads_the_whole_frame_on_the_global_store() {
    let a = global_a();
    let store = S3ObjectStore::new();
    for api in ["get_writer_with_compression", "create_writer"] {
        for (size, data) in issue218_payloads() {
            let name = format!("global/{api}/{size}");
            let bucket = unique_bucket(&format!("c218-g-{size}"));
            write_compressed(&store, api, &format!("s3://{bucket}/obj"), &data).await;

            let reqs = a.for_bucket(&bucket);
            assert_eq!(count(&reqs, "POST", "uploads"), 1, "{name}: {reqs:?}");
            if size == "multipart" {
                assert!(
                    count(&reqs, "PUT", "partNumber") >= 2,
                    "{name}: the compressed object must span parts: {reqs:?}"
                );
            }
            let stored = a
                .completed(&bucket, "obj.zst")
                .unwrap_or_else(|| panic!("{name}: A has no completed /{bucket}/obj.zst"));
            assert_complete_frame_of(&name, &stored, &data);
        }
    }
}

#[tokio::test]
async fn compressed_writer_uploads_the_whole_frame_on_a_per_endpoint_store() {
    let a = global_a();
    let b = ObjMock::start(SEED_B, OBJECT_SIZE);
    let store = store_for(&b).await;
    for api in ["get_writer_with_compression", "create_writer"] {
        for (size, data) in issue218_payloads() {
            for form in ["bucket-form", "host-form"] {
                let name = format!("endpoint/{api}/{size}/{form}");
                let bucket = unique_bucket(&format!("c218-e-{size}"));
                let uri = if form == "bucket-form" {
                    format!("s3://{bucket}/obj")
                } else {
                    format!("s3://127.0.0.1:{}/{bucket}/obj", b.port)
                };
                write_compressed(&store, api, &uri, &data).await;

                assert_eq!(a.for_bucket(&bucket).len(), 0, "{name}: A got a request");
                let reqs = b.for_bucket(&bucket);
                assert_eq!(count(&reqs, "POST", "uploads"), 1, "{name}: {reqs:?}");
                if size == "multipart" {
                    assert!(
                        count(&reqs, "PUT", "partNumber") >= 2,
                        "{name}: the compressed object must span parts: {reqs:?}"
                    );
                }
                let stored = b
                    .completed(&bucket, "obj.zst")
                    .unwrap_or_else(|| panic!("{name}: B has no completed /{bucket}/obj.zst"));
                assert_complete_frame_of(&name, &stored, &data);
            }
        }
    }
}
