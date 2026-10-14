// Regression tests for issue #179: `ObjectStore::delete_batch` must delete each
// URI in the bucket that URI names, never in the bucket of `uris[0]`.
//
// These tests need no cloud: `S3ObjectStore::for_endpoint` is pointed at an
// in-process mock S3 endpoint (path-style) that records every DeleteObjects
// request (bucket + keys) it receives.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// How the mock answers a DeleteObjects request for a given bucket.
#[derive(Clone)]
enum Reply {
    /// HTTP 200 with an empty `<DeleteResult/>` (all keys deleted).
    Ok,
    /// HTTP 403 AccessDenied (non-retryable whole-request failure).
    Denied,
    /// HTTP 200 whose body reports every key as an individual `<Error>`.
    PerKeyError,
    /// The first `n` requests for the bucket succeed, every later one is a
    /// 403 AccessDenied (used to fail the second chunk of a large batch).
    DeniedAfter(usize),
}

/// One recorded DeleteObjects request.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Seen {
    bucket: String,
    keys: Vec<String>,
}

struct MockS3 {
    endpoint: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    /// Request paths of single-object `DELETE /<bucket>/<key>` calls.
    singles: Arc<Mutex<Vec<String>>>,
}

impl MockS3 {
    fn requests(&self) -> Vec<Seen> {
        self.seen.lock().expect("seen lock").clone()
    }

    /// Paths of single-object DELETE requests received.
    fn single_deletes(&self) -> Vec<String> {
        self.singles.lock().expect("singles lock").clone()
    }

    /// Keys received, per bucket, in arrival order.
    fn keys_by_bucket(&self) -> HashMap<String, Vec<String>> {
        let mut m: HashMap<String, Vec<String>> = HashMap::new();
        for s in self.requests() {
            m.entry(s.bucket).or_default().extend(s.keys);
        }
        m
    }
}

static ENV: Once = Once::new();

/// Listener reserved for the process-wide (global) S3 client's mock endpoint.
/// `init_env` binds it and points `AWS_ENDPOINT_URL` at it; only the one test
/// that exercises the global-client path takes it and serves on it.
static GLOBAL_LISTENER: Mutex<Option<std::net::TcpListener>> = Mutex::new(None);

/// Fake credentials/region for the SDK; the mock never validates signatures.
/// Also routes the global client (never the per-endpoint ones) to a local
/// port, so a test of the global path can never reach a real endpoint.
fn init_env() {
    ENV.call_once(|| {
        let global = std::net::TcpListener::bind("127.0.0.1:0").expect("bind global mock");
        global.set_nonblocking(true).expect("nonblocking");
        let addr = global.local_addr().expect("global mock addr");
        *GLOBAL_LISTENER.lock().expect("listener lock") = Some(global);
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
            std::env::set_var("AWS_ENDPOINT_URL", format!("http://{addr}"));
        }
    });
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Undo the XML escaping of an element's text.
fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Extract every `<Key>..</Key>` value from a request body (XML-unescaped).
fn keys_in_body(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(s) = rest.find("<Key>") {
        let after = &rest[s + "<Key>".len()..];
        match after.find("</Key>") {
            Some(e) => {
                out.push(xml_unescape(&after[..e]));
                rest = &after[e + "</Key>".len()..];
            }
            None => break,
        }
    }
    out
}

async fn serve_conn(
    mut sock: TcpStream,
    seen: Arc<Mutex<Vec<Seen>>>,
    singles: Arc<Mutex<Vec<String>>>,
    replies: Arc<HashMap<String, Reply>>,
) {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        // Read until the end of the header block.
        let hdr_end = loop {
            if let Some(p) = find_subslice(&buf, b"\r\n\r\n") {
                break p + 4;
            }
            let mut tmp = [0u8; 8192];
            match sock.read(&mut tmp).await {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        };
        let head = String::from_utf8_lossy(&buf[..hdr_end]).to_string();
        let mut lines = head.split("\r\n");
        let request_line = lines.next().unwrap_or("").to_string();
        let method = request_line.split(' ').next().unwrap_or("").to_string();
        let path = request_line.split(' ').nth(1).unwrap_or("/").to_string();
        let mut content_length = 0usize;
        let mut chunked = false;
        for l in lines {
            let lower = l.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            }
            if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
                chunked = true;
            }
        }

        // Read the body.
        let body_bytes: Vec<u8>;
        if chunked {
            loop {
                // A chunked body ends with a zero-size chunk.
                if find_subslice(&buf[hdr_end..], b"\r\n0\r\n").is_some()
                    || buf[hdr_end..].starts_with(b"0\r\n")
                {
                    // Wait for the trailing CRLF after any trailers.
                    if buf.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let mut tmp = [0u8; 8192];
                match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                }
            }
            body_bytes = buf[hdr_end..].to_vec();
            buf.clear();
        } else {
            while buf.len() < hdr_end + content_length {
                let mut tmp = [0u8; 8192];
                match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                }
            }
            body_bytes = buf[hdr_end..hdr_end + content_length].to_vec();
            buf.drain(..hdr_end + content_length);
        }
        let body = String::from_utf8_lossy(&body_bytes).to_string();

        // Single-object delete: DELETE /<bucket>/<key> -> 204 No Content.
        if method == "DELETE" {
            let object_path = path.split('?').next().unwrap_or("").to_string();
            singles.lock().expect("singles lock").push(object_path);
            let resp = "HTTP/1.1 204 No Content\r\nx-amz-request-id: 1\r\n\r\n";
            if sock.write_all(resp.as_bytes()).await.is_err() {
                return;
            }
            continue;
        }

        // Path-style: /<bucket>?delete  or  /<bucket>/?delete
        let bucket = path
            .trim_start_matches('/')
            .split(['/', '?'])
            .next()
            .unwrap_or("")
            .to_string();
        let keys = keys_in_body(&body);
        // Number of earlier requests for this bucket (for `DeniedAfter`).
        let earlier = {
            let mut g = seen.lock().expect("seen lock");
            let n = g.iter().filter(|s| s.bucket == bucket).count();
            g.push(Seen {
                bucket: bucket.clone(),
                keys: keys.clone(),
            });
            n
        };

        let reply = match replies.get(&bucket).cloned().unwrap_or(Reply::Ok) {
            Reply::DeniedAfter(n) if earlier < n => Reply::Ok,
            Reply::DeniedAfter(_) => Reply::Denied,
            other => other,
        };
        let (status, payload) = match reply {
            Reply::Ok => (
                "200 OK",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult \
                 xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"></DeleteResult>"
                    .to_string(),
            ),
            Reply::Denied | Reply::DeniedAfter(_) => (
                "403 Forbidden",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>AccessDenied</Code>\
                 <Message>mock denied</Message><RequestId>1</RequestId></Error>"
                    .to_string(),
            ),
            Reply::PerKeyError => {
                let errs: String = keys
                    .iter()
                    .map(|k| {
                        format!(
                            "<Error><Key>{k}</Key><Code>InternalError</Code>\
                             <Message>mock per-key failure</Message></Error>"
                        )
                    })
                    .collect();
                (
                    "200 OK",
                    format!(
                        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult \
                         xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">{errs}</DeleteResult>"
                    ),
                )
            }
        };
        let resp = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/xml\r\nContent-Length: {}\r\n\
             x-amz-request-id: 1\r\n\r\n{payload}",
            payload.len()
        );
        if sock.write_all(resp.as_bytes()).await.is_err() {
            return;
        }
    }
}

async fn start_mock(replies: HashMap<String, Reply>) -> MockS3 {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
    serve_mock(listener, replies)
}

/// Start serving the mock on an already-bound listener.
fn serve_mock(listener: TcpListener, replies: HashMap<String, Reply>) -> MockS3 {
    let addr = listener.local_addr().expect("mock addr");
    let seen: Arc<Mutex<Vec<Seen>>> = Arc::new(Mutex::new(Vec::new()));
    let replies = Arc::new(replies);
    let singles: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_srv = seen.clone();
    let singles_srv = singles.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((sock, _)) => {
                    tokio::spawn(serve_conn(
                        sock,
                        seen_srv.clone(),
                        singles_srv.clone(),
                        replies.clone(),
                    ));
                }
                Err(_) => return,
            }
        }
    });
    MockS3 {
        endpoint: format!("http://{addr}"),
        seen,
        singles,
    }
}

async fn store_for(mock: &MockS3) -> S3ObjectStore {
    init_env();
    S3ObjectStore::for_endpoint(&mock.endpoint)
        .await
        .expect("for_endpoint with fake credentials")
}

fn uris(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// The bug: every key was sent to the bucket of `uris[0]`.  The decoy-shaped
/// case from the issue: `bkt-a/p/a` and `bkt-b/p/b` -- the key `p/b` must be
/// deleted in `bkt-b`, and must NOT be sent to `bkt-a`.
#[tokio::test]
async fn mixed_buckets_each_key_goes_to_its_own_bucket() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    store
        .delete_batch(&uris(&["s3://bkt-a/p/a", "s3://bkt-b/p/b"]))
        .await
        .expect("delete_batch should succeed");

    let by_bucket = mock.keys_by_bucket();
    assert_eq!(
        by_bucket.get("bkt-a").cloned().unwrap_or_default(),
        vec!["p/a".to_string()],
        "bkt-a must receive only p/a (p/b in bkt-a is the DECOY the caller never named)"
    );
    assert_eq!(
        by_bucket.get("bkt-b").cloned().unwrap_or_default(),
        vec!["p/b".to_string()],
        "bkt-b must receive p/b"
    );
    assert_eq!(by_bucket.len(), 2, "no other bucket may be touched");
}

/// Interleaved buckets: grouping must preserve caller order within a bucket.
#[tokio::test]
async fn interleaved_buckets_preserve_order_within_each_bucket() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    store
        .delete_batch(&uris(&[
            "s3://bkt-a/1",
            "s3://bkt-b/1",
            "s3://bkt-a/2",
            "s3://bkt-b/2",
            "s3://bkt-a/3",
        ]))
        .await
        .expect("delete_batch should succeed");

    let by_bucket = mock.keys_by_bucket();
    assert_eq!(by_bucket["bkt-a"], vec!["1", "2", "3"]);
    assert_eq!(by_bucket["bkt-b"], vec!["1", "2"]);
    assert_eq!(by_bucket.len(), 2);
}

/// Single-bucket fast path: 2500 keys -> exactly three DeleteObjects requests
/// of 1000, 1000 and 500 keys, all to that bucket, in caller order.
#[tokio::test]
async fn single_bucket_chunks_at_1000() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    let all: Vec<String> = (0..2500).map(|i| format!("s3://bkt-a/k{i:05}")).collect();
    store.delete_batch(&all).await.expect("delete_batch");

    let reqs = mock.requests();
    let sizes: Vec<usize> = reqs.iter().map(|r| r.keys.len()).collect();
    assert_eq!(sizes, vec![1000, 1000, 500], "chunking at 1000 keys");
    assert!(reqs.iter().all(|r| r.bucket == "bkt-a"));
    let flat: Vec<String> = reqs.into_iter().flat_map(|r| r.keys).collect();
    let expect: Vec<String> = (0..2500).map(|i| format!("k{i:05}")).collect();
    assert_eq!(flat, expect, "order preserved across chunks");
}

/// An unparseable URI is an error, never silently dropped, and nothing is
/// deleted before the whole batch has been validated.
#[tokio::test]
async fn unparseable_uri_is_an_error_and_nothing_is_deleted() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    let err = store
        .delete_batch(&uris(&["s3://bkt-a/ok", "not-a-uri", "s3://bkt-b/ok"]))
        .await
        .expect_err("an unparseable URI must fail the batch");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not-a-uri"),
        "error must name the offending URI: {msg}"
    );
    assert!(
        mock.requests().is_empty(),
        "no DeleteObjects may be issued when the batch contains an unparseable URI"
    );
}

/// One bucket failing must not hide the other bucket's outcome: the healthy
/// bucket is still processed, and the error names the URIs that failed.
#[tokio::test]
async fn per_bucket_failure_is_reported_with_failed_uris() {
    let mut replies = HashMap::new();
    replies.insert("bkt-bad".to_string(), Reply::Denied);
    let mock = start_mock(replies).await;
    let store = store_for(&mock).await;

    let err = store
        .delete_batch(&uris(&[
            "s3://bkt-bad/x/one",
            "s3://bkt-good/y/two",
            "s3://bkt-bad/x/three",
        ]))
        .await
        .expect_err("a failing bucket must fail the batch");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("s3://bkt-bad/x/one"),
        "failed URI named: {msg}"
    );
    assert!(
        msg.contains("s3://bkt-bad/x/three"),
        "failed URI named: {msg}"
    );
    assert!(
        !msg.contains("s3://bkt-good/y/two"),
        "a URI that was deleted must not be reported as failed: {msg}"
    );
    assert_eq!(
        mock.keys_by_bucket()
            .get("bkt-good")
            .cloned()
            .unwrap_or_default(),
        vec!["y/two".to_string()],
        "the healthy bucket must still be processed"
    );
}

/// S3 reports per-key failures inside an HTTP 200 body; the per-endpoint
/// client path must surface them (it previously discarded the response).
#[tokio::test]
async fn per_key_errors_in_a_200_response_are_an_error() {
    let mut replies = HashMap::new();
    replies.insert("bkt-a".to_string(), Reply::PerKeyError);
    let mock = start_mock(replies).await;
    let store = store_for(&mock).await;

    let err = store
        .delete_batch(&uris(&["s3://bkt-a/k1"]))
        .await
        .expect_err("per-key errors in the response body must fail the batch");
    let msg = format!("{err:#}");
    assert!(msg.contains("k1"), "error must name the failed key: {msg}");
}

/// Empty batch stays a no-op with no request.
#[tokio::test]
async fn empty_batch_is_a_noop() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;
    store.delete_batch(&[]).await.expect("empty batch");
    assert!(mock.requests().is_empty());
}

// ---------------------------------------------------------------------------
// Second round (#179): server identity, chunk offsets, key shapes, sizes.
// ---------------------------------------------------------------------------

/// `host:port` of a mock (the authority a URI uses to name it).
fn authority(mock: &MockS3) -> String {
    mock.endpoint.trim_start_matches("http://").to_string()
}

/// A batch that also names ANOTHER server must delete nothing at all.  The
/// pre-round-2 code ignored the host and sent both keys to this store's
/// server: it deleted `data/k` here twice and never touched the other server
/// (a data-loss bug of the same class as the bucket one).
#[tokio::test]
async fn uri_naming_a_different_server_fails_the_batch_and_deletes_nothing() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;
    let here = authority(&mock);
    let elsewhere = "s3://127.0.0.1:9/bkt-a/data/k";

    let outcome = store
        .delete_batch(&uris(&[&format!("s3://{here}/bkt-a/data/k"), elsewhere]))
        .await;
    assert!(
        mock.requests().is_empty(),
        "no request may be sent for a batch naming another server (outcome {outcome:?}): {:?}",
        mock.requests()
    );
    let err = outcome.expect_err("a URI naming another server must fail the batch");
    let msg = format!("{err:#}");
    assert!(msg.contains(elsewhere), "error must name the URI: {msg}");
    assert!(
        !msg.contains(&format!("s3://{here}/bkt-a/data/k")),
        "the URI naming THIS server is fine and must not be blamed: {msg}"
    );
}

/// Same-host-different-port is a different server too.
#[tokio::test]
async fn same_host_other_port_is_a_different_server() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;
    let addr = mock.endpoint.trim_start_matches("http://");
    let port: u16 = addr.rsplit(':').next().expect("port").parse().expect("u16");
    let other = format!("s3://127.0.0.1:{}/bkt-a/k", port.wrapping_add(1));

    let outcome = store.delete_batch(&uris(&[&other])).await;
    assert!(
        mock.requests().is_empty(),
        "nothing may be sent (outcome {outcome:?}): {:?}",
        mock.requests()
    );
    let err = outcome.expect_err("another port is another server");
    assert!(format!("{err:#}").contains(&other));
}

/// URIs that name this store's own server (explicit host:port, any host-name
/// case) are accepted, group by bucket, and mix freely with bucket-form URIs.
#[tokio::test]
async fn uris_naming_this_stores_own_server_are_accepted() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;
    let here = authority(&mock);

    store
        .delete_batch(&uris(&[
            &format!("s3://{here}/bkt-a/1"),
            "s3://bkt-b/2",
            &format!("s3://{here}/bkt-b/3"),
            "s3://bkt-a/4",
        ]))
        .await
        .expect("URIs naming this server are valid");

    let by_bucket = mock.keys_by_bucket();
    assert_eq!(by_bucket["bkt-a"], vec!["1", "4"]);
    assert_eq!(by_bucket["bkt-b"], vec!["2", "3"]);
    assert_eq!(by_bucket.len(), 2);
}

/// A store built on the process-wide client talks to the server named by the
/// environment, never to one a URI names: such a URI is an error, and nothing
/// is sent.  (The env points the global client at a local mock, so the unfixed
/// code is observable as requests received instead of an error.)
#[tokio::test]
async fn global_client_store_rejects_a_uri_naming_a_server() {
    init_env();
    let listener = GLOBAL_LISTENER
        .lock()
        .expect("listener lock")
        .take()
        .expect("global listener is taken by exactly one test");
    let global_mock = serve_mock(
        TcpListener::from_std(listener).expect("tokio listener"),
        HashMap::new(),
    );
    let store = S3ObjectStore::new();
    let named = "s3://127.0.0.1:9/bkt-a/data/k";

    let outcome = store.delete_batch(&uris(&["s3://bkt-a/ok", named])).await;
    assert!(
        global_mock.requests().is_empty(),
        "nothing may be sent (outcome {outcome:?}): {:?}",
        global_mock.requests()
    );
    let err = outcome.expect_err("a global-client store cannot honour a URI-named server");
    let msg = format!("{err:#}");
    assert!(msg.contains(named), "error must name the URI: {msg}");
}

/// Single-object delete has the same blindness and gets the same check.
#[tokio::test]
async fn single_delete_of_a_uri_naming_another_server_is_an_error() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;
    let elsewhere = "s3://127.0.0.1:9/bkt-a/data/k";

    let outcome = store.delete(elsewhere).await;
    assert!(
        mock.requests().is_empty() && mock.single_deletes().is_empty(),
        "no request of any kind may be sent (outcome {outcome:?}): singles {:?}",
        mock.single_deletes()
    );
    let err = outcome.expect_err("delete() of a URI naming another server must fail");
    assert!(format!("{err:#}").contains(elsewhere));

    // ...while the URI naming this very server is deleted normally.
    store
        .delete(&format!("s3://{}/bkt-a/data/k", authority(&mock)))
        .await
        .expect("own-server single delete");
    assert_eq!(mock.single_deletes(), vec!["/bkt-a/data/k".to_string()]);
}

/// The second DeleteObjects request fails: only the URIs of that chunk and
/// later are reported as not deleted, never the 1000 already confirmed.
#[tokio::test]
async fn failure_in_the_second_chunk_reports_only_unconfirmed_uris() {
    let mut replies = HashMap::new();
    replies.insert("bkt-a".to_string(), Reply::DeniedAfter(1));
    let mock = start_mock(replies).await;
    let store = store_for(&mock).await;

    let all: Vec<String> = (0..1500).map(|i| format!("s3://bkt-a/k{i:04}")).collect();
    let err = store
        .delete_batch(&all)
        .await
        .expect_err("chunk 2 is denied");
    let msg = format!("{err:#}");

    assert!(msg.contains("500 of 1500"), "offset arithmetic: {msg}");
    assert!(
        msg.contains("s3://bkt-a/k1000"),
        "first unconfirmed URI named: {msg}"
    );
    assert!(
        !msg.contains("s3://bkt-a/k0999"),
        "a confirmed URI must not be reported: {msg}"
    );
    let sizes: Vec<usize> = mock.requests().iter().map(|r| r.keys.len()).collect();
    assert_eq!(sizes, vec![1000, 500], "two requests, then stop");
}

/// Keys with spaces, unicode, XML metacharacters, '%' and a leading slash
/// reach the server byte-for-byte, in the bucket the URI names.
#[tokio::test]
async fn awkward_key_shapes_are_sent_verbatim() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    let keys = [
        "dir with spaces/file name.bin",
        "unicode/\u{00e9}\u{4e2d}\u{6587}/\u{1f600}.dat",
        "amp&lt;ersand/<tag>\"q\"'a'.txt",
        "percent/100%/a%20b%2Fc",
        "/leading-slash",
    ];
    let batch: Vec<String> = keys.iter().map(|k| format!("s3://bkt-a/{k}")).collect();
    store.delete_batch(&batch).await.expect("delete_batch");

    let got = mock.keys_by_bucket();
    let expect: Vec<String> = keys.iter().map(|k| k.to_string()).collect();
    assert_eq!(got["bkt-a"], expect);
    assert_eq!(got.len(), 1);
}

/// Duplicate URIs are passed through unchanged (the pre-#179 behaviour; S3
/// treats a repeated delete as idempotent) -- nothing is dropped or merged.
#[tokio::test]
async fn duplicate_uris_are_passed_through() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    store
        .delete_batch(&uris(&["s3://bkt-a/x", "s3://bkt-b/y", "s3://bkt-a/x"]))
        .await
        .expect("duplicates are not an error");
    let by_bucket = mock.keys_by_bucket();
    assert_eq!(by_bucket["bkt-a"], vec!["x", "x"]);
    assert_eq!(by_bucket["bkt-b"], vec!["y"]);
}

/// Exactly 1000 keys is one request; 1001 is two (1000 + 1).
#[tokio::test]
async fn chunk_boundaries_1000_and_1001() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    let exactly: Vec<String> = (0..1000).map(|i| format!("s3://bkt-a/e{i}")).collect();
    store.delete_batch(&exactly).await.expect("1000");
    let sizes: Vec<usize> = mock.requests().iter().map(|r| r.keys.len()).collect();
    assert_eq!(sizes, vec![1000]);

    let over: Vec<String> = (0..1001).map(|i| format!("s3://bkt-b/o{i}")).collect();
    store.delete_batch(&over).await.expect("1001");
    let sizes: Vec<usize> = mock
        .requests()
        .iter()
        .filter(|r| r.bucket == "bkt-b")
        .map(|r| r.keys.len())
        .collect();
    assert_eq!(sizes, vec![1000, 1]);
}

/// When several buckets fail, the first failure's cause chain is kept as the
/// error source (not flattened to text) and the others are listed.
#[tokio::test]
async fn several_failing_buckets_keep_the_first_cause_chain() {
    let mut replies = HashMap::new();
    replies.insert("bkt-x".to_string(), Reply::Denied);
    replies.insert("bkt-y".to_string(), Reply::Denied);
    let mock = start_mock(replies).await;
    let store = store_for(&mock).await;

    let err = store
        .delete_batch(&uris(&["s3://bkt-x/1", "s3://bkt-y/2"]))
        .await
        .expect_err("both buckets are denied");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("s3://bkt-x/1") && msg.contains("s3://bkt-y/2"),
        "{msg}"
    );
    assert!(
        err.chain().count() >= 3,
        "the first failure's SDK cause chain must survive as the source, got {} link(s): {msg}",
        err.chain().count()
    );
}

// ---------------------------------------------------------------------------
// Cross-fix interaction (#179 delete_batch x #178 "a server needs a port").
//
// `delete_batch` decides whether a URI names another server through the URI
// parser. After #178 the parser applies the single `classify_s3_first_segment`
// rule, so a portless IP literal is an ERROR (not a bucket, not a server) and a
// portless DNS name is a BUCKET (not a server). Neither branch could assert this
// alone: #179 alone classified by the old suffix-hint rule, #178 alone had a
// delete_batch that ignored the host entirely.
// ---------------------------------------------------------------------------

/// A portless IP literal is the #178 error ("add :port") and nothing is sent,
/// for `delete_batch` (anywhere in the batch) and for a single `delete`.
#[tokio::test]
async fn portless_ip_uri_is_the_178_error_and_deletes_nothing() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;
    let portless = "s3://10.0.0.5/bkt/key";

    for batch in [
        uris(&[portless]),
        uris(&["s3://bkt-a/ok", portless]),
        uris(&[portless, "s3://bkt-a/ok"]),
    ] {
        let outcome = store.delete_batch(&batch).await;
        assert!(
            mock.requests().is_empty(),
            "no request may be sent (outcome {outcome:?}): {:?}",
            mock.requests()
        );
        let err = outcome.expect_err("a portless IP is an error, never 'no server'");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("10.0.0.5") && msg.contains("add :port"),
            "must be the #178 message that says to add :port: {msg}"
        );
    }

    let outcome = store.delete(portless).await;
    assert!(
        mock.requests().is_empty() && mock.single_deletes().is_empty(),
        "no request of any kind may be sent (outcome {outcome:?})"
    );
    let msg = format!(
        "{:#}",
        outcome.expect_err("delete() of a portless IP must fail")
    );
    assert!(msg.contains("add :port"), "{msg}");
}

/// A portless DNS name is a BUCKET (#178), so it is not "another server": the
/// batch is accepted and the name is sent as the bucket on the store's server.
#[tokio::test]
async fn portless_dns_name_is_a_bucket_not_another_server() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    store
        .delete_batch(&uris(&["s3://minio.example.com/data/k", "s3://bkt-a/1"]))
        .await
        .expect("a dotted portless name is a bucket on this store's server");

    let by_bucket = mock.keys_by_bucket();
    assert_eq!(by_bucket["minio.example.com"], vec!["data/k"]);
    assert_eq!(by_bucket["bkt-a"], vec!["1"]);
    assert_eq!(by_bucket.len(), 2);
}

/// Leading-'/' keys (#172) survive the #179 grouping: `s3://b//k` is the key
/// `/k` in bucket `b`, distinct from `k`, and is sent verbatim.
#[tokio::test]
async fn leading_slash_key_is_deleted_as_written() {
    let mock = start_mock(HashMap::new()).await;
    let store = store_for(&mock).await;

    store
        .delete_batch(&uris(&["s3://bkt-a//data/x", "s3://bkt-a/data/y"]))
        .await
        .expect("delete_batch");

    assert_eq!(
        mock.keys_by_bucket()["bkt-a"],
        vec!["/data/x", "data/y"],
        "the leading '/' is part of the key"
    );
}
