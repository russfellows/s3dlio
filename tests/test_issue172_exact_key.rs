// tests/test_issue172_exact_key.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression tests for s3dlio issue #172.
//
// Bug: `stat`, ranged GETs and the size-probe HEAD-before-GET stripped every
// leading '/' from the key (`key.trim_start_matches('/')`) before building
// the request, so for the URI `s3://bkt//data/x` (key "/data/x") they acted
// on the DIFFERENT object "data/x". PUT and plain GET kept the key, so
// s3dlio was internally inconsistent -- on GCS, `stat` returned another
// object's size and `get_range` returned another object's bytes.
//
// Contract under test: the key the caller names -- exactly as
// `parse_s3_uri_full` returns it -- is the key EVERY operation sends.
//
// Mechanism: the in-process mock S3 server (tests/common/mock_s3_server.rs)
// records every request under its raw request path (`/{bucket}/{key}`,
// path-style). Each test uses a unique key, issues one operation, and then
// asserts on the request counts of BOTH the exact path and the path with the
// leading '/' removed: the exact path must have been hit, and the stripped
// path -- a different object -- must never have been touched.
//
// Each test runs both key shapes: the leading-slash key ("/<u>/.../data/x",
// RED on unfixed code) and the plain key ("<u>/.../data/x", a control that
// is GREEN on both unfixed and fixed code and proves the harness itself).

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, MockS3State, PathCallCounts, SIZED_OBJECT_BODY,
    SIZED_OBJECT_MARKER,
};
use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use s3dlio::s3_utils::{
    get_object, get_object_concurrent_range_async, get_object_range_uri_async,
    get_object_range_uri_timed_async, get_object_uri_optimized_async, stat_object,
    stat_object_uri_async,
};

const BUCKET: &str = "mock-bucket";

fn init() -> (u16, MockS3State) {
    // The size-probe path only range-splits objects at or above this
    // threshold (default 32 MiB). The mock objects are 16 bytes, so lower it
    // to 0 to drive `get_object_uri_optimized_async` through
    // `get_object_with_known_size_async` (the `format!("s3://..")` site).
    // The library reads it once, so every test sets the same value before
    // the first library call.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::env::set_var("S3DLIO_RANGE_THRESHOLD_MB", "0"));
    ensure_mock_s3_server()
}

/// A unique, sized-object key. `leading_slash` selects the buggy shape:
/// "/<u>/sized-obj/data/x" versus "<u>/sized-obj/data/x".
fn unique_key(tag: &str, leading_slash: bool) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let rest = format!("i172-{tag}-{nanos}-{n}/{SIZED_OBJECT_MARKER}/data/x");
    if leading_slash {
        format!("/{rest}")
    } else {
        rest
    }
}

fn uri_for(key: &str) -> String {
    format!("s3://{BUCKET}/{key}")
}

// NOTE on `>=`: the SDK client is process-global while each `#[tokio::test]`
// runtime is short-lived, so a pooled connection can die between tests and the
// SDK transparently retries -- an exact key may legitimately be hit more than
// once. What must hold is "the exact key was requested" and "its stripped twin
// never was".

/// Request counts seen by the exact key and by its leading-'/'-stripped twin.
fn counts(state: &MockS3State, key: &str) -> (PathCallCounts, PathCallCounts) {
    let exact = state.counts_for(&key_path(BUCKET, key));
    let stripped = state.counts_for(&key_path(BUCKET, key.trim_start_matches('/')));
    (exact, stripped)
}

/// For a plain key the "stripped twin" is the key itself, so the
/// "different object untouched" assertion only applies to the slash shape.
fn assert_other_object_untouched(
    leading_slash: bool,
    stripped: &PathCallCounts,
    what: &str,
    key: &str,
) {
    if leading_slash {
        assert_eq!(
            (stripped.head_calls, stripped.get_calls),
            (0, 0),
            "{what}: key {key:?} was rewritten to {:?} -- a DIFFERENT object -- \
             and requests were sent for it (issue #172)",
            key.trim_start_matches('/'),
        );
    }
}

async fn stat_object_uri_async_heads_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("stat-uri", slash);
    let stat = stat_object_uri_async(&uri_for(&key)).await.expect("stat");
    assert_eq!(stat.size, SIZED_OBJECT_BODY.len() as u64);
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.head_calls >= 1,
        "stat_object_uri_async: HEAD must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "stat_object_uri_async", &key);
}

async fn stat_object_heads_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("stat-bk", slash);
    stat_object(BUCKET, &key).await.expect("stat");
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.head_calls >= 1,
        "stat_object: HEAD must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "stat_object", &key);
}

async fn s3_object_store_stat_heads_the_exact_key(slash: bool) {
    let (port, state) = init();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let key = unique_key("store-stat", slash);
    let meta = store.stat(&uri_for(&key)).await.expect("stat");
    assert_eq!(meta.size, SIZED_OBJECT_BODY.len() as u64);
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.head_calls >= 1,
        "S3ObjectStore::stat: HEAD must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "S3ObjectStore::stat", &key);
}

async fn ranged_get_uses_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("range", slash);
    let bytes = get_object_range_uri_async(&uri_for(&key), 0, Some(5))
        .await
        .expect("range");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[..5]);
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.get_calls >= 1,
        "get_object_range_uri_async: ranged GET must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "get_object_range_uri_async", &key);
}

async fn timed_ranged_get_uses_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("range-timed", slash);
    let (bytes, _, _) = get_object_range_uri_timed_async(&uri_for(&key), 2, Some(4))
        .await
        .expect("timed range");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[2..6]);
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.get_calls >= 1,
        "get_object_range_uri_timed_async: ranged GET must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "timed range", &key);
}

/// Covers BOTH the size-probe HEAD inside `get_object_concurrent_range_async`
/// and the per-chunk ranged GETs (`attempt_range_chunk_fill`).

async fn concurrent_range_get_probes_and_fetches_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("conc-range", slash);
    // 16 bytes, 4-byte chunks => four ranged GETs.
    let bytes = get_object_concurrent_range_async(&uri_for(&key), 0, None, Some(4), Some(2))
        .await
        .expect("concurrent range");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[..]);
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.head_calls >= 1,
        "concurrent range GET: size-probe HEAD must go to the exact key {key:?}"
    );
    assert!(
        exact.get_calls >= 4,
        "concurrent range GET: every chunk GET must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "concurrent range GET", &key);
}

/// The size-probe HEAD before GET, and (with the threshold lowered to 0 in
/// `init`) the follow-on known-size range path that rebuilds an `s3://` URI.

async fn optimized_get_probes_and_fetches_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("optimized", slash);
    let bytes = get_object_uri_optimized_async(&uri_for(&key))
        .await
        .expect("optimized get");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[..]);
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.head_calls >= 1,
        "get_object_uri_optimized_async: size-probe HEAD must go to the exact key {key:?}"
    );
    assert!(
        exact.get_calls >= 1,
        "get_object_uri_optimized_async: the GET must go to the exact key {key:?}"
    );
    assert_other_object_untouched(slash, &stripped, "optimized get", &key);
}

/// `S3ObjectStore::get_range` already used the exact key; kept here so the
/// trait-level stat/get_range pair is proven consistent from now on.

async fn s3_object_store_range_and_stat_address_the_same_object(slash: bool) {
    let (port, state) = init();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let key = unique_key("store-pair", slash);
    let uri = uri_for(&key);
    let bytes = store.get_range(&uri, 0, Some(5)).await.expect("range");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[..5]);
    store.stat(&uri).await.expect("stat");
    let (exact, stripped) = counts(&state, &key);
    assert!(
        exact.get_calls >= 1,
        "get_range must hit the exact key {key:?}"
    );
    assert!(exact.head_calls >= 1, "stat must hit the exact key {key:?}");
    assert_other_object_untouched(slash, &stripped, "store range+stat", &key);
}

/// Control: plain GET has always kept the key. GREEN before and after.

async fn plain_get_uses_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("plain-get", slash);
    let bytes = get_object(BUCKET, &key).await.expect("get");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[..]);
    let (exact, stripped) = counts(&state, &key);
    assert!(exact.get_calls >= 1);
    assert_other_object_untouched(slash, &stripped, "plain get", &key);
}

/// Generate `<name>_leading_slash_key` (the #172 shape, RED on unfixed code)
/// and `<name>_plain_key` (the control, GREEN on both) from one case.
macro_rules! both_shapes {
    ($($case:ident => $slash:ident, $plain:ident;)*) => {
        $(
            #[tokio::test]
            async fn $slash() {
                $case(true).await;
            }

            #[tokio::test]
            async fn $plain() {
                $case(false).await;
            }
        )*
    };
}

both_shapes! {
    stat_object_uri_async_heads_the_exact_key
        => stat_uri_leading_slash_key, stat_uri_plain_key;
    stat_object_heads_the_exact_key
        => stat_bucket_key_leading_slash_key, stat_bucket_key_plain_key;
    s3_object_store_stat_heads_the_exact_key
        => store_stat_leading_slash_key, store_stat_plain_key;
    ranged_get_uses_the_exact_key
        => range_leading_slash_key, range_plain_key;
    timed_ranged_get_uses_the_exact_key
        => timed_range_leading_slash_key, timed_range_plain_key;
    concurrent_range_get_probes_and_fetches_the_exact_key
        => concurrent_range_leading_slash_key, concurrent_range_plain_key;
    optimized_get_probes_and_fetches_the_exact_key
        => size_probe_leading_slash_key, size_probe_plain_key;
    s3_object_store_range_and_stat_address_the_same_object
        => store_pair_leading_slash_key, store_pair_plain_key;
    plain_get_uses_the_exact_key
        => plain_get_leading_slash_key, plain_get_plain_key;
}
