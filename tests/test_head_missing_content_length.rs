// tests/test_head_missing_content_length.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #152 sub-bug 2.4
// (audit finding f37).
//
// Bug: `get_object_concurrent_range_async` collapsed a HEAD response
// with no `Content-Length` header into `object_size=0` via
// `.unwrap_or(0)`. With `object_size=0`, the very next guard
// (`start_offset >= object_size`) was trivially true for any
// `start_offset >= 0`, so the function returned `Ok(Bytes::new())` —
// an empty payload reported as SUCCESS for an object that may well
// have real data. The sibling cache-populating HEAD site inside
// `get_object_uri_optimized_async` had the same pattern, but poisoned
// the ObjectSizeCache with a bogus 0 instead of returning empty bytes
// directly.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §4, bug B3): a HEAD that succeeds but omits Content-Length must be
// treated as an anomaly, not silently coerced to size=0.
// `get_object_concurrent_range_async` now returns `Err`.
// `get_object_uri_optimized_async`'s cache-populating site now falls
// back to a plain GET (same as an outright HEAD failure) instead of
// caching size=0.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, key_path, NO_CONTENT_LENGTH_MARKER};
use s3dlio::s3_utils::{get_object_concurrent_range_async, get_object_uri_optimized_async};

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
async fn concurrent_range_get_errors_when_head_omits_content_length() {
    let (_, _state) = ensure_mock_s3_server();
    let key = unique_key(&format!("crg-{NO_CONTENT_LENGTH_MARKER}"));
    let uri = format!("s3://mock-bucket/{key}");

    let result = get_object_concurrent_range_async(&uri, 0, None, None, None).await;

    let err = result.expect_err(
        "get_object_concurrent_range_async must return Err when HEAD succeeds but omits \
         Content-Length, not silently return Ok(Bytes::new()) as if the object were empty \
         (audit f37: unwrap_or(0) coerced object_size to 0, which then tripped the \
         start_offset >= object_size early-return guard).",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Content-Length") || msg.contains("content-length") || msg.contains("size"),
        "error message should name the missing Content-Length, got: {msg}"
    );
}

/// Sanity check: normal HEAD responses (with Content-Length) are
/// unaffected — the mock server's default HEAD handler returns
/// Content-Length: 0 for an unmarked key, which is a legitimately empty
/// object, not a missing-header anomaly, so this must still succeed.
#[tokio::test]
async fn concurrent_range_get_succeeds_on_a_genuinely_empty_object() {
    let (_, _state) = ensure_mock_s3_server();
    let key = unique_key("crg-genuinely-empty");
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = get_object_concurrent_range_async(&uri, 0, None, None, None)
        .await
        .expect("a HEAD that reports Content-Length: 0 is a real empty object, not an error");
    assert_eq!(bytes.len(), 0);
}

/// get_object_uri_optimized_async's cache-populating HEAD site: a
/// missing Content-Length must fall back to a plain GET (same as a
/// HEAD failure) rather than caching a bogus size=0 for this URI.
///
/// The single-call return value doesn't distinguish pre-fix from
/// post-fix here: this mock's synthetic object is small either way, so
/// both the "cache 0, route by size" (buggy) and "don't cache, fall
/// back to plain GET" (fixed) paths happen to call plain GET on the
/// FIRST request and return the same bytes. The actual damage is cache
/// poisoning — a bogus `size=0` cached against this URI. That's
/// observable on a SECOND call: with the bug, the poisoned cache entry
/// causes the second call to skip HEAD entirely (cache hit); fixed,
/// nothing was cached, so the second call issues a fresh HEAD too.
#[tokio::test]
async fn optimized_get_does_not_poison_the_size_cache_when_head_omits_content_length() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_key(&format!("opt-{NO_CONTENT_LENGTH_MARKER}"));
    let path = key_path("mock-bucket", &key);
    let uri = format!("s3://mock-bucket/{key}");

    let first = get_object_uri_optimized_async(&uri)
        .await
        .expect("must fall back to a plain GET, not error and not silently return empty");
    assert_eq!(&first[..], b"mock");

    let second = get_object_uri_optimized_async(&uri)
        .await
        .expect("second call must also succeed");
    assert_eq!(&second[..], b"mock");

    let snap = state.counts_for(&path);
    assert_eq!(
        snap.head_calls, 2,
        "expected a fresh HEAD on every call to a URI whose HEAD omits Content-Length — \
         observed {} HEAD call(s). If this is 1, the first call's missing-Content-Length \
         response got cached as size=0 (audit f37), so the second call hit the poisoned \
         cache entry and skipped HEAD entirely.",
        snap.head_calls
    );
}
