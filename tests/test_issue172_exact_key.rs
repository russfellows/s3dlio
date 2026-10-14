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

use bytes::Bytes;
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
    // to 0 to drive `get_object_uri_optimized_async` through the known-size
    // (large-object) path.
    //
    // How this is made independent of test order: the library latches the
    // threshold (and the range-optimisation / skip-HEAD switches) in
    // process-global `OnceLock`s on first use. EVERY test in this binary calls
    // `init()` before its first library call, and `Once::call_once` makes all
    // concurrent callers wait until the closure has finished, so the
    // variables are always in place before any latch is read -- whichever
    // test the harness schedules first, and however many threads it uses. The
    // three variables are set unconditionally (not "if unset"), so a value
    // inherited from the developer's shell cannot change the outcome.
    // `optimized_get_probes_and_fetches_the_exact_key` additionally ASSERTS
    // that a ranged GET arrived, so a latch that somehow missed this setup
    // fails loudly there instead of silently testing the plain-GET fallback.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("S3DLIO_RANGE_THRESHOLD_MB", "0");
        std::env::set_var("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "1");
        std::env::set_var("S3DLIO_SKIP_HEAD", "0");
    });
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
/// `init`) the follow-on known-size range path.
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

    // Prove the KNOWN-SIZE path was the one taken. The plain-GET fallback
    // (`get_object`) sends no Range header; the known-size path issues a
    // ranged GET for the exact key. If this fails, the S3DLIO_RANGE_THRESHOLD_MB
    // setup in `init()` did not take effect and the test is not exercising the
    // large-object code at all.
    let reqs = state.requests_containing(tag_of(&key));
    let exact_path = key_path(BUCKET, &key);
    assert!(
        reqs.iter()
            .any(|r| r.method == "GET" && r.path == exact_path && r.range.is_some()),
        "known-size path not taken: no ranged GET reached {exact_path:?}; saw {reqs:?}"
    );
    assert!(
        reqs.iter()
            .all(|r| !(r.method == "GET" && r.path == exact_path && r.range.is_none())),
        "the plain-GET fallback was used for {exact_path:?}: {reqs:?}"
    );
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

// ---------------------------------------------------------------------------
// Follow-up round (F5/F6/F7): stronger known-size proof, dotted bucket,
// delete / list / multipart put, special-character keys, and the
// MultiEndpointStore relative-path join against an S3 endpoint.
// ---------------------------------------------------------------------------

/// The unique ASCII token embedded in every key made by `unique_key` /
/// `unique_special_key`: the first path segment of the key, which is what
/// `MockS3State::requests_containing` filters on.
fn tag_of(key: &str) -> &str {
    key.trim_start_matches('/').split('/').next().unwrap_or("")
}

/// RFC 3986 percent-encoding of an object key as the AWS SDK puts it in a
/// path-style request path: unreserved characters and '/' stay, every other
/// byte becomes `%XX` (uppercase hex). Independent of the library under test.
fn encode_key_path(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for b in key.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Assert that EVERY request carrying `tag` (in its path or query) went to
/// exactly one of `allowed_paths` -- i.e. no variant of the key (stripped,
/// decoded, double-encoded, bucket/key mis-split) was ever requested.
fn assert_only_paths(state: &MockS3State, tag: &str, allowed_paths: &[String], what: &str) {
    let reqs = state.requests_containing(tag);
    assert!(
        !reqs.is_empty(),
        "{what}: no request carrying tag {tag:?} reached the server"
    );
    for r in &reqs {
        assert!(
            allowed_paths.contains(&r.path),
            "{what}: {} request went to {:?} (query {:?}); only {allowed_paths:?} \
             may be requested -- a variant of the key was addressed (issue #172)",
            r.method,
            r.path,
            r.query,
        );
    }
}

// ----- F6: dotted bucket through the known-size (large-object) path --------

/// `get_object_with_known_size_async` used to rebuild `s3://{bucket}/{key}` and
/// re-parse it. A dotted bucket whose last label looks like a hostname TLD
/// (`data.example.com`) is then read as an endpoint host, so the bucket/key
/// split moved by one segment and the request went to the wrong object.
/// The URI below names an explicit endpoint, so the FIRST parse correctly
/// yields bucket `data.example.com`; only the internal round-trip broke it.
async fn dotted_bucket_known_size_get_addresses_the_exact_bucket_and_key(slash: bool) {
    let (port, state) = init();
    const DOTTED: &str = "data.example.com";
    let key = unique_key("dotted", slash);
    let uri = format!("s3://127.0.0.1:{port}/{DOTTED}/{key}");
    let bytes = get_object_uri_optimized_async(&uri)
        .await
        .expect("optimized get with a dotted bucket");
    assert_eq!(&bytes[..], &SIZED_OBJECT_BODY[..]);

    let exact = key_path(DOTTED, &key);
    assert_only_paths(
        &state,
        tag_of(&key),
        std::slice::from_ref(&exact),
        "dotted-bucket GET",
    );
    let reqs = state.requests_containing(tag_of(&key));
    assert!(
        reqs.iter()
            .any(|r| r.method == "GET" && r.path == exact && r.range.is_some()),
        "dotted-bucket GET: the known-size path must send a ranged GET to {exact:?}; \
         saw {reqs:?}"
    );
}

// ----- F7: delete ----------------------------------------------------------

async fn single_delete_removes_the_exact_key(slash: bool) {
    let (port, state) = init();
    let key = unique_key("delete", slash);
    let uri = uri_for(&key);
    // client-present store (dedicated per-endpoint client)
    let with_client = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    with_client.delete(&uri).await.expect("delete (client)");
    let exact = key_path(BUCKET, &key);
    assert!(state.counts_for(&exact).delete_calls >= 1);
    // global-client store: its single delete is a one-key DeleteObjects batch
    S3ObjectStore::new().delete(&uri).await.expect("delete");
    assert!(state.counts_for(&exact).delete_calls >= 1);
    assert!(state.counts_for(&exact).batch_delete_calls >= 1);
    if slash {
        let twin = key_path(BUCKET, key.trim_start_matches('/'));
        let t = state.counts_for(&twin);
        assert_eq!((t.delete_calls, t.batch_delete_calls), (0, 0));
    }
    assert_only_paths(&state, tag_of(&key), &[exact], "single delete");
}

async fn batch_delete_names_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("delete-batch", slash);
    s3dlio::s3_utils::delete_objects_async(BUCKET, std::slice::from_ref(&key))
        .await
        .expect("delete_objects_async");
    let exact = key_path(BUCKET, &key);
    assert!(
        state.counts_for(&exact).batch_delete_calls >= 1,
        "delete_objects_async must name the exact key {key:?} in the batch body"
    );
    if slash {
        assert_eq!(
            state
                .counts_for(&key_path(BUCKET, key.trim_start_matches('/')))
                .batch_delete_calls,
            0,
            "the stripped twin of {key:?} was named in a DeleteObjects body (issue #172)"
        );
    }

    // The store's batch delete derives keys from URIs: same contract.
    let key2 = unique_key("delete-batch-store", slash);
    let store = S3ObjectStore::new();
    store
        .delete_batch(&[uri_for(&key2)])
        .await
        .expect("delete_batch");
    assert!(
        state
            .counts_for(&key_path(BUCKET, &key2))
            .batch_delete_calls
            >= 1,
        "S3ObjectStore::delete_batch must name the exact key {key2:?}"
    );
    if slash {
        assert_eq!(
            state
                .counts_for(&key_path(BUCKET, key2.trim_start_matches('/')))
                .batch_delete_calls,
            0,
            "S3ObjectStore::delete_batch named the stripped twin of {key2:?} (issue #172)"
        );
    }
}

// ----- F7: list ------------------------------------------------------------

/// The mock answers every ListObjectsV2 with `{prefix}bar` and
/// `{prefix}other/deep.dat`. With the URI prefix `s3://b/<key-prefix>/foo/bar`
/// the request prefix must be `<key-prefix>/foo/` verbatim (leading '/' kept),
/// and the returned URIs must carry the real, slash-bearing key.
async fn list_uses_the_prefix_verbatim_and_returns_real_keys(slash: bool) {
    let (port, state) = init();
    let tag = unique_key("list", false);
    let tag = tag_of(&tag).to_string();
    let key_prefix = if slash {
        format!("/{tag}/foo/")
    } else {
        format!("{tag}/foo/")
    };
    let uri_prefix = format!("s3://{BUCKET}/{key_prefix}bar");
    let want_uri = format!("s3://{BUCKET}/{key_prefix}bar");
    let want_query_prefix = format!(
        "prefix={}",
        encode_key_path(&key_prefix).replace('/', "%2F")
    );

    // client-present path
    let with_client = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let listed = with_client.list(&uri_prefix, true).await.expect("list");
    assert_eq!(
        listed,
        vec![want_uri.clone()],
        "list (client) returned URIs that do not carry the real key"
    );
    // global-client path (list_objects_stream)
    let listed = S3ObjectStore::new()
        .list(&uri_prefix, true)
        .await
        .expect("list");
    assert_eq!(
        listed,
        vec![want_uri],
        "list (global client) returned URIs that do not carry the real key"
    );

    let reqs = state.requests_containing(&tag);
    assert!(
        reqs.len() >= 2,
        "both list calls must have reached the server: {reqs:?}"
    );
    for r in &reqs {
        assert_eq!(r.method, "GET");
        assert_eq!(
            r.path,
            format!("/{BUCKET}/"),
            "list goes to the bucket root"
        );
        assert!(
            r.query.split('&').any(|kv| kv == want_query_prefix),
            "the list prefix must be sent verbatim: want {want_query_prefix:?} in query {:?}",
            r.query
        );
    }
}

// ----- F7: multipart put ---------------------------------------------------

async fn multipart_put_create_upload_complete_use_the_exact_key(slash: bool) {
    let (_, state) = init();
    let key = unique_key("mpu", slash);
    let part = 5 * 1024 * 1024;
    // two parts: one full minimum-size part plus a small tail
    let data = Bytes::from(vec![0xA5u8; part + 1024]);
    s3dlio::s3_utils::put_object_multipart_uri_async(&uri_for(&key), data, Some(part))
        .await
        .expect("multipart put");
    let exact = key_path(BUCKET, &key);
    let c = state.counts_for(&exact);
    assert!(
        c.create_calls >= 1,
        "CreateMultipartUpload must use {key:?}"
    );
    assert!(c.upload_part_calls >= 2, "UploadPart must use {key:?}");
    assert!(
        c.complete_calls >= 1,
        "CompleteMultipartUpload must use {key:?}"
    );
    assert_eq!(c.abort_calls, 0);
    assert_only_paths(&state, tag_of(&key), &[exact], "multipart put");
}

// ----- F7: percent-sequences / spaces / unicode ----------------------------

/// A key full of characters that a careless "normalise the key" step would
/// alter: a literal `%2F` and `%20` (must NOT be decoded), a space, unicode,
/// and a doubled slash inside the key.
fn unique_special_key(leading_slash: bool) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let rest = format!(
        "i172-special-{nanos}-{n}/100%25 done%2Fx %20/sp ace//ü-é-日本/{SIZED_OBJECT_MARKER}/x"
    );
    if leading_slash {
        format!("/{rest}")
    } else {
        rest
    }
}

async fn special_character_keys_reach_the_wire_unaltered(slash: bool) {
    let (port, state) = init();
    let key = unique_special_key(slash);
    let uri = uri_for(&key);
    let exact = key_path(BUCKET, &encode_key_path(&key));

    stat_object_uri_async(&uri).await.expect("stat");
    get_object_range_uri_async(&uri, 0, Some(5))
        .await
        .expect("range");
    get_object(BUCKET, &key).await.expect("get");
    s3dlio::s3_utils::put_object_uri_async(&uri, Bytes::from_static(b"payload"))
        .await
        .expect("put");
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    store.stat(&uri).await.expect("store stat");
    store.delete(&uri).await.expect("delete");

    let c = state.counts_for(&exact);
    assert!(c.head_calls >= 2, "stat x2 must HEAD {exact:?}: {c:?}");
    assert!(c.get_calls >= 2, "range + get must GET {exact:?}: {c:?}");
    assert!(c.put_calls >= 1, "put must PUT {exact:?}: {c:?}");
    assert!(c.delete_calls >= 1, "delete must DELETE {exact:?}: {c:?}");
    assert_only_paths(&state, tag_of(&key), &[exact], "special-character key");
}

// ----- F7: MultiEndpointStore relative-path join against an S3 endpoint ----

/// Endpoint `s3://mock-bucket` (NO trailing '/') plus relative path
/// `/<tag>/...` must address key `<tag>/...` for every operation (it used to
/// address `/<tag>/...` for PUT/GET/DELETE and `<tag>/...` for stat/range).
/// The same call against `s3://mock-bucket/` and with a path that has no
/// leading '/' must agree.
async fn multi_endpoint_relative_join_addresses_one_key_on_s3(
    endpoint_has_trailing_slash: bool,
    path_has_leading_slash: bool,
) {
    use s3dlio::multi_endpoint::{LoadBalanceStrategy, MultiEndpointStore};
    let (_, state) = init();
    let key = unique_key("multi-ep", false); // "<tag>/sized-obj/data/x"
    let endpoint = if endpoint_has_trailing_slash {
        format!("s3://{BUCKET}/")
    } else {
        format!("s3://{BUCKET}")
    };
    let rel = if path_has_leading_slash {
        format!("/{key}")
    } else {
        key.clone()
    };
    let store = MultiEndpointStore::new(vec![endpoint], LoadBalanceStrategy::RoundRobin, Some(1))
        .expect("MultiEndpointStore::new");

    store.get(&rel).await.expect("get");
    store.get_range(&rel, 0, Some(5)).await.expect("get_range");
    store.stat(&rel).await.expect("stat");
    store
        .put(&rel, Bytes::from_static(b"payload"))
        .await
        .expect("put");
    store.delete(&rel).await.expect("delete");
    store
        .get_from_endpoint(0, &rel)
        .await
        .expect("get_from_endpoint");

    let exact = key_path(BUCKET, &key);
    let c = state.counts_for(&exact);
    assert!(
        c.get_calls >= 3,
        "get, get_range, get_from_endpoint -> {exact:?}: {c:?}"
    );
    assert!(c.head_calls >= 1, "stat -> {exact:?}: {c:?}");
    assert!(c.put_calls >= 1, "put -> {exact:?}: {c:?}");
    assert!(
        c.delete_calls + c.batch_delete_calls >= 1,
        "delete -> {exact:?}: {c:?}"
    );
    assert_only_paths(
        &state,
        tag_of(&key),
        &[exact],
        "multi-endpoint relative join",
    );
}

macro_rules! plain_cases {
    ($($case:ident => $name:ident, ($($arg:expr),*);)*) => {
        $(
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn $name() {
                $case($($arg),*).await;
            }
        )*
    };
}

macro_rules! both_shapes_mt {
    ($($case:ident => $slash:ident, $plain:ident;)*) => {
        $(
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn $slash() {
                $case(true).await;
            }

            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn $plain() {
                $case(false).await;
            }
        )*
    };
}

both_shapes_mt! {
    dotted_bucket_known_size_get_addresses_the_exact_bucket_and_key
        => dotted_bucket_leading_slash_key, dotted_bucket_plain_key;
    single_delete_removes_the_exact_key
        => delete_leading_slash_key, delete_plain_key;
    batch_delete_names_the_exact_key
        => batch_delete_leading_slash_key, batch_delete_plain_key;
    list_uses_the_prefix_verbatim_and_returns_real_keys
        => list_leading_slash_prefix, list_plain_prefix;
    multipart_put_create_upload_complete_use_the_exact_key
        => multipart_put_leading_slash_key, multipart_put_plain_key;
    special_character_keys_reach_the_wire_unaltered
        => special_chars_leading_slash_key, special_chars_plain_key;
}

plain_cases! {
    multi_endpoint_relative_join_addresses_one_key_on_s3
        => multi_endpoint_no_trailing_slash_endpoint_leading_slash_path, (false, true);
    multi_endpoint_relative_join_addresses_one_key_on_s3
        => multi_endpoint_no_trailing_slash_endpoint_plain_path, (false, false);
    multi_endpoint_relative_join_addresses_one_key_on_s3
        => multi_endpoint_trailing_slash_endpoint_leading_slash_path, (true, true);
    multi_endpoint_relative_join_addresses_one_key_on_s3
        => multi_endpoint_trailing_slash_endpoint_plain_path, (true, false);
}
