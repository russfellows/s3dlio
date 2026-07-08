// tests/test_range_length_zero.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #152 sub-bug 2.3
// (audit finding f42).
//
// Bug: `get_object_range_uri_async` / `get_object_range_uri_timed_async`
// only overwrote the default open-ended range string
// (`bytes={offset}-`, meaning "from offset to end of object") inside an
// `if len > 0` guard. A caller passing `length=Some(0)` — expecting
// zero bytes back — instead got the range header left as
// `bytes={offset}-`, which S3 interprets as "everything from offset to
// EOF": the full remainder of the object was silently downloaded.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §4, bug B2): `length=Some(0)` returns `Ok(Bytes::new())` immediately,
// with NO network call at all.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, key_path};
use s3dlio::s3_utils::{get_object_range_uri_async, get_object_range_uri_timed_async};

fn unique_key(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{prefix}-{nanos}-{n}")
}

#[tokio::test]
async fn length_zero_returns_empty_bytes_with_no_network_call() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_key("range-zero");
    let path = key_path("mock-bucket", &key);
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = get_object_range_uri_async(&uri, 100, Some(0))
        .await
        .expect("length=Some(0) must succeed, not error");

    assert_eq!(
        bytes.len(),
        0,
        "length=Some(0) must return zero bytes, not the full remainder of the object \
         (audit f42: the range-string builder left `bytes={{offset}}-` — open-ended — \
         when len==0 skipped the overwrite branch)"
    );

    let snap = state.counts_for(&path);
    assert_eq!(
        snap.get_calls, 0,
        "length=Some(0) must short-circuit before any network call, observed {} GET(s)",
        snap.get_calls
    );
}

#[tokio::test]
async fn length_zero_still_works_for_nonzero_offset_timed_variant() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_key("range-zero-timed");
    let path = key_path("mock-bucket", &key);
    let uri = format!("s3://mock-bucket/{key}");

    let (bytes, ttfb, transfer) = get_object_range_uri_timed_async(&uri, 12345, Some(0))
        .await
        .expect("length=Some(0) must succeed, not error");

    assert_eq!(bytes.len(), 0, "length=Some(0) must return zero bytes");
    assert_eq!(ttfb, std::time::Duration::ZERO);
    assert_eq!(transfer, std::time::Duration::ZERO);

    let snap = state.counts_for(&path);
    assert_eq!(
        snap.get_calls, 0,
        "length=Some(0) must short-circuit before any network call, observed {} GET(s)",
        snap.get_calls
    );
}

/// Sanity check: a real nonzero length still round-trips through the
/// mock server correctly (the fast path doesn't overcorrect).
#[tokio::test]
async fn nonzero_length_still_issues_a_network_call() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_key("range-nonzero");
    let path = key_path("mock-bucket", &key);
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = get_object_range_uri_async(&uri, 0, Some(4))
        .await
        .expect("a normal ranged GET should succeed against the mock server");
    assert_eq!(&bytes[..], b"mock");

    let snap = state.counts_for(&path);
    assert_eq!(snap.get_calls, 1);
}
