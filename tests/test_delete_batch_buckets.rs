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
}

impl MockS3 {
    fn requests(&self) -> Vec<Seen> {
        self.seen.lock().expect("seen lock").clone()
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

/// Fake credentials/region for the SDK; the mock never validates signatures.
fn init_env() {
    ENV.call_once(|| {
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
        }
    });
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Extract every `<Key>..</Key>` value from a request body.
fn keys_in_body(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(s) = rest.find("<Key>") {
        let after = &rest[s + "<Key>".len()..];
        match after.find("</Key>") {
            Some(e) => {
                out.push(after[..e].to_string());
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

        // Path-style: /<bucket>?delete  or  /<bucket>/?delete
        let bucket = path
            .trim_start_matches('/')
            .split(['/', '?'])
            .next()
            .unwrap_or("")
            .to_string();
        let keys = keys_in_body(&body);
        seen.lock().expect("seen lock").push(Seen {
            bucket: bucket.clone(),
            keys: keys.clone(),
        });

        let reply = replies.get(&bucket).cloned().unwrap_or(Reply::Ok);
        let (status, payload) = match reply {
            Reply::Ok => (
                "200 OK",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult \
                 xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"></DeleteResult>"
                    .to_string(),
            ),
            Reply::Denied => (
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
    let addr = listener.local_addr().expect("mock addr");
    let seen: Arc<Mutex<Vec<Seen>>> = Arc::new(Mutex::new(Vec::new()));
    let replies = Arc::new(replies);
    let seen_srv = seen.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((sock, _)) => {
                    tokio::spawn(serve_conn(sock, seen_srv.clone(), replies.clone()));
                }
                Err(_) => return,
            }
        }
    });
    MockS3 {
        endpoint: format!("http://{addr}"),
        seen,
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
