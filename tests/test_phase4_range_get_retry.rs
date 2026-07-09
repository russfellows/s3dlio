// tests/test_phase4_range_get_retry.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #152 sub-bug 2.2
// (audit finding f19) — a Phase 4b (v0.9.108) regression.
//
// Bug: `get_object_range_uri_async` did a bare `send()` then
// `body.collect()` with no `retry_get_body` wrap. Once Phase 4b
// (v0.9.108) made the response body stream via
// `SdkBody::from_body_1_x`, transient body-transfer failures (TCP FIN
// mid-body, HTTP/2 RST, connection reset after headers) moved out from
// under smithy's send()-level retry — smithy had already returned
// Ok(response) at headers by the time a streamed body can fail. This
// function's send()+collect() pair was left unwrapped, so what used to
// be auto-retried by the SDK now aborts the whole GET on the first
// transient failure. Four production callers reach this function
// (data_loader/s3_bytes.rs's per-part range GETs, object_store.rs's
// client-present get_range fallback, and two internal fast-paths in
// s3_utils.rs), so fixing the shared function fixes all of them.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §4, bug B1): wrap the whole send()+collect() pair in
// `crate::retry::retry_get_body` — same pattern already used by
// `S3Ops::get_object`/`get_object_range` since Phase 4b.
//
// Test infrastructure note: this needs a GENUINE mid-body connection
// truncation (headers accepted with a promised Content-Length, then
// the socket closes before that many bytes arrive) — a full HTTP-level
// error response (e.g. 500) does NOT distinguish pre-fix from
// post-fix, because smithy's OWN send()-level retry already covers a
// complete failed response; only a failure that occurs AFTER headers
// are accepted (mid-body) is unique to the Phase 4b regression this
// bug is about. The shared H1 mock harness (tests/common/mock_s3_server.rs)
// is built on hyper's `Full<Bytes>` body type, which cannot under-deliver
// a declared Content-Length — there's no API for it to close a
// connection mid-body. So this file runs its own dedicated raw-TCP
// mock server that speaks just enough HTTP/1.1 to control exactly
// that: first connection gets truncated mid-body, second gets a
// complete response.

use anyhow::Result;
use s3dlio::s3_utils::get_object_range_uri_async;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Spawn a raw-TCP mock S3 endpoint on the CURRENT test's own runtime
/// (fine here because this file has exactly one test — no need for the
/// shared harness's dedicated-background-thread trick to outlive other
/// tests). First connection: writes valid headers claiming
/// `Content-Length: 4` then closes after writing only 2 bytes of body
/// (mid-body truncation). Every connection after that: a complete,
/// correct 4-byte response.
async fn spawn_truncating_get_server() -> (u16, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let connection_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&connection_count);

    tokio::spawn(async move {
        loop {
            let (mut stream, _) = match listener.accept().await {
                Ok(v) => v,
                Err(_) => break,
            };
            let n = cc.fetch_add(1, Ordering::SeqCst);

            // Drain the request (don't need to parse it — every request
            // in this test targets the same object).
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await;

            if n == 0 {
                // First connection: promise 4 bytes, deliver 2, then
                // close the socket without sending the rest — a real
                // mid-body connection failure, not an HTTP-level error.
                let headers = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(headers).await;
                let _ = stream.write_all(b"mo").await;
                let _ = stream.flush().await;
                drop(stream); // abrupt close — client sees a truncated body
            } else {
                let headers = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(headers).await;
                let _ = stream.write_all(b"mock").await;
                let _ = stream.flush().await;
                drop(stream);
            }
        }
    });

    (port, connection_count)
}

#[tokio::test]
async fn mid_body_truncation_on_first_attempt_is_retried_and_succeeds() -> Result<()> {
    let (port, connection_count) = spawn_truncating_get_server().await;

    std::env::set_var("AWS_ACCESS_KEY_ID", "mock-access-key");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "mock-secret-key");
    std::env::set_var("AWS_REGION", "us-east-1");
    std::env::set_var("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"));
    std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");

    let bytes = get_object_range_uri_async("s3://mock-bucket/mock-key", 0, Some(4))
        .await
        .expect(
            "a mid-body connection truncation on the first attempt must be retried and \
             ultimately succeed (audit f19: the unwrapped send()+collect() pair surfaced \
             the truncation directly instead of retrying it — this is the exact Phase 4b \
             regression bug B1 targets).",
        );
    assert_eq!(&bytes[..], b"mock");

    let connections = connection_count.load(Ordering::SeqCst);
    assert!(
        connections >= 2,
        "expected at least 2 connection attempts (1 truncated + 1 retried-success), \
         observed {connections}"
    );

    Ok(())
}
