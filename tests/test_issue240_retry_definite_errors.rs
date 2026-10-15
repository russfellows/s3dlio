// tests/test_issue240_retry_definite_errors.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression tests for s3dlio issue #240.
//
// Bug: `retry_get_body` retried EVERY error, including a definite server
// answer such as 404 NoSuchKey or 403 AccessDenied. A GET of a missing key
// therefore cost `S3DLIO_MAX_RETRY_ATTEMPTS` requests and at least 300 ms of
// backoff before the error came back.
//
// Contract under test: definite client errors are returned after exactly ONE
// request; transient failures (503 SlowDown, a connection cut mid-body) are
// still retried.
//
// Mechanism: the in-process mock S3 server (tests/common/mock_s3_server.rs)
// counts GET requests per key path. Each test uses a unique key carrying one
// of the mock's behaviour markers and asserts on that key's `get_calls`.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, GET_FORBIDDEN_MARKER, GET_NOT_FOUND_MARKER,
    GET_UNAVAILABLE_MARKER, SIZED_OBJECT_MARKER,
};
use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use std::time::{Duration, Instant};

const BUCKET: &str = "mock-bucket";

fn init() -> (u16, common::mock_s3_server::MockS3State) {
    // Pin the knobs the assertions depend on, before any client reads them.
    // Set unconditionally so a value inherited from the developer's shell
    // cannot change the outcome. The SDK's own retry layer and the outer
    // s3dlio loop share this budget (3 attempts each).
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        std::env::set_var("S3DLIO_SKIP_HEAD", "1");
        std::env::set_var("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "0");
    });
    ensure_mock_s3_server()
}

async fn endpoint_store(port: u16) -> S3ObjectStore {
    S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint")
}

fn uri_for(key: &str) -> String {
    format!("s3://{BUCKET}/{key}")
}

/// A missing key through a per-endpoint store: exactly one GET, prompt error.
#[tokio::test]
async fn per_endpoint_get_of_missing_key_is_one_request() {
    let (port, state) = init();
    let store = endpoint_store(port).await;
    let key = unique_test_key(GET_NOT_FOUND_MARKER);

    let start = Instant::now();
    let err = store.get(&uri_for(&key)).await.expect_err("404 must fail");
    let elapsed = start.elapsed();

    let msg = format!("{err:#}");
    assert!(msg.contains("NoSuchKey"), "unexpected error: {msg}");
    let c = state.counts_for(&key_path(BUCKET, &key));
    assert_eq!(
        c.get_calls, 1,
        "a 404 is a definite answer: it must cost exactly one GET, saw {}",
        c.get_calls
    );
    // The unfixed loop sleeps 100 ms + 200 ms before giving up; a generous
    // bound (the structural count above is the real proof).
    assert!(
        elapsed < Duration::from_millis(250),
        "404 took {elapsed:?}; the retry backoff must not run"
    );
}

/// Same for a ranged GET through a per-endpoint store.
#[tokio::test]
async fn per_endpoint_get_range_of_missing_key_is_one_request() {
    let (port, state) = init();
    let store = endpoint_store(port).await;
    let key = unique_test_key(GET_NOT_FOUND_MARKER);

    let err = store
        .get_range(&uri_for(&key), 0, Some(4))
        .await
        .expect_err("404 must fail");
    assert!(format!("{err:#}").contains("NoSuchKey"));
    let c = state.counts_for(&key_path(BUCKET, &key));
    assert_eq!(c.get_calls, 1, "ranged 404 must cost exactly one GET");
}

/// 403 AccessDenied is also definite.
#[tokio::test]
async fn per_endpoint_get_of_forbidden_key_is_one_request() {
    let (port, state) = init();
    let store = endpoint_store(port).await;
    let key = unique_test_key(GET_FORBIDDEN_MARKER);

    let err = store.get(&uri_for(&key)).await.expect_err("403 must fail");
    assert!(format!("{err:#}").contains("AccessDenied"));
    let c = state.counts_for(&key_path(BUCKET, &key));
    assert_eq!(c.get_calls, 1, "403 must cost exactly one GET");
}

/// The process-global client path (`S3ObjectStore::new()` -> s3_utils) reaches
/// the same loop through the range-GET helper.
#[tokio::test]
async fn global_client_range_get_of_missing_key_is_one_request() {
    let (_, state) = init();
    let key = unique_test_key(GET_NOT_FOUND_MARKER);
    let uri = uri_for(&key);

    // Warm the global client's pooled connection inside THIS runtime so the
    // SDK has no reason to retry a dead pooled connection on the real call.
    let warm = unique_test_key("warm");
    s3dlio::s3_utils::get_object_range_uri_async(&uri_for(&warm), 0, Some(4))
        .await
        .expect("warm-up GET");

    let err = s3dlio::s3_utils::get_object_range_uri_async(&uri, 0, Some(4))
        .await
        .expect_err("404 must fail");
    assert!(format!("{err:#}").contains("NoSuchKey"), "{err:#}");
    let c = state.counts_for(&key_path(BUCKET, &key));
    assert_eq!(c.get_calls, 1, "global-path 404 must cost exactly one GET");
}

/// The concurrent range-GET path has its own per-chunk retry loop (it cannot
/// use `retry_get_body`); it must also stop at a definite answer. The key
/// carries both the sized-object marker (so HEAD succeeds with 16 bytes and the
/// chunked path is entered) and the 404 marker (so every chunk GET is a 404).
#[tokio::test]
async fn concurrent_range_chunks_do_not_retry_a_404() {
    let (_, state) = init();
    let key = unique_test_key(&format!("{SIZED_OBJECT_MARKER}-{GET_NOT_FOUND_MARKER}"));

    let start = Instant::now();
    let err = s3dlio::s3_utils::get_object_concurrent_range_async(
        &uri_for(&key),
        0,
        None,
        Some(4),
        Some(1),
    )
    .await
    .expect_err("404 must fail");
    let elapsed = start.elapsed();

    assert!(format!("{err:#}").contains("NoSuchKey"), "{err:#}");
    let c = state.counts_for(&key_path(BUCKET, &key));
    // One chunk in flight at a time and the first 404 ends the download, so
    // exactly one GET. The unfixed loop sent that chunk three times.
    assert_eq!(c.get_calls, 1, "a 404 chunk must cost exactly one GET");
    assert!(
        elapsed < Duration::from_millis(250),
        "404 took {elapsed:?}; the chunk retry backoff must not run"
    );
}

/// A 503 SlowDown is transient: the s3dlio loop must still retry it on top of
/// the SDK's own attempts. Both layers use the pinned budget of 3, so a 503
/// retried by both reaches the server up to 3 x 3 = 9 times, while one that
/// only the SDK retried would reach it exactly 3 times.
///
/// The upper bound is deliberately not pinned to exactly 9: the SDK's
/// standard retry mode has its own internals (a shared retry-token bucket, its
/// own classification of 503/SlowDown) that can change the SDK-side count.
#[tokio::test]
async fn throttled_get_is_still_retried_by_the_outer_loop() {
    let (port, state) = init();
    let store = endpoint_store(port).await;
    let key = unique_test_key(GET_UNAVAILABLE_MARKER);

    store.get(&uri_for(&key)).await.expect_err("503 must fail");
    let c = state.counts_for(&key_path(BUCKET, &key));
    assert!(
        c.get_calls > 3 && c.get_calls <= 9,
        "503 must be retried by BOTH layers (more than the SDK's 3, at most 3 x 3), saw {}",
        c.get_calls
    );
}

/// A connection cut mid-body is a body-transfer failure: retried, and the
/// second attempt succeeds. The shared mock cannot under-deliver a body, so
/// this test runs its own tiny raw-TCP server: the first connection answers
/// `200` with `Content-Length: 100`, sends 4 body bytes and closes; the
/// second connection serves a complete 4-byte object.
#[tokio::test]
async fn mid_body_failure_is_still_retried() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    init();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(AtomicUsize::new(0));
    let served = Arc::clone(&requests);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut buf = [0u8; 4096];
            let mut req = Vec::new();
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => req.extend_from_slice(&buf[..n]),
                }
            }
            if req.is_empty() {
                continue;
            }
            let n = served.fetch_add(1, Ordering::SeqCst) + 1;
            let resp: &[u8] = if n == 1 {
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nmock"
            } else {
                b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\nmock"
            };
            let _ = stream.write_all(resp);
            let _ = stream.flush();
            // Dropping the stream closes it: for request 1 that is the
            // mid-body cut (4 of 100 bytes delivered).
        }
    });

    let store = endpoint_store(port).await;
    let data = store
        .get(&uri_for("mid-body/object"))
        .await
        .expect("second attempt must succeed");
    assert_eq!(&data[..], b"mock");
    assert_eq!(
        requests.load(Ordering::SeqCst),
        2,
        "truncated body then success = 2 GETs"
    );
}
