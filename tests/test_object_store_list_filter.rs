// tests/test_object_store_list_filter.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #154 sub-bug 4.4
// (audit finding f24).
//
// Bug: `S3ObjectStore::list_with_client`'s filter was
// `re.is_match(key) || key.starts_with(&prefix_str)`. Since the S3 API
// was already queried with `prefix(prefix_str)`, EVERY key in the
// response is guaranteed by the SDK to start with `prefix_str` — so the
// right-hand side of that `||` was always true, making the regex check
// dead code. `list("s3://bucket/foo/bar", recursive=true)` (which
// splits into prefix_str="foo/", pattern_str="bar") returned
// "foo/other/deep.dat" right alongside "foo/bar", even though
// "other/deep.dat" doesn't match the "bar" pattern at all.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §4, bug B7): match the regex against the key's tail PAST the prefix,
// not the whole key (which trivially always starts with the prefix).

#[allow(dead_code)]
mod common;

use common::mock_s3_server::ensure_mock_s3_server;
use s3dlio::object_store::{ObjectStore, S3ObjectStore};

fn unique_prefix(name: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{name}-{nanos}-{n}")
}

/// The mock server's ListObjectsV2 handler always returns exactly two
/// keys under the requested prefix: "{prefix}bar" (tail "bar", matches
/// the "bar" pattern) and "{prefix}other/deep.dat" (tail
/// "other/deep.dat", does NOT match "bar"). Only the first should come
/// back from `list()`.
#[tokio::test]
async fn list_excludes_keys_whose_tail_does_not_match_the_pattern() {
    let (port, _) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let unique = unique_prefix("b7-list-filter");
    let uri_prefix = format!("s3://mock-bucket/{unique}/foo/bar");

    let keys = store
        .list(&uri_prefix, true)
        .await
        .expect("list() should succeed against the mock server");

    let matching_key = format!("s3://mock-bucket/{unique}/foo/bar");
    let non_matching_key = format!("s3://mock-bucket/{unique}/foo/other/deep.dat");

    assert!(
        keys.contains(&matching_key),
        "the key whose tail matches the pattern must be present: {keys:?}"
    );
    assert!(
        !keys.contains(&non_matching_key),
        "list() must not return a key whose tail-past-prefix doesn't match the derived \
         pattern — got {keys:?}. This is the dead-regex-filter bug (audit f24): \
         `key.starts_with(&prefix_str)` is always true once S3 already filtered by that \
         same prefix server-side, so the `||` let every key under the prefix through \
         regardless of the pattern."
    );
}
