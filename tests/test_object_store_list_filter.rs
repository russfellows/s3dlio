// tests/test_object_store_list_filter.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// History: this file began as the RED-then-GREEN test for s3dlio issue #154
// sub-bug 4.4 (audit finding f24), where `S3ObjectStore::list_with_client`'s
// filter `re.is_match(key) || key.starts_with(&prefix_str)` was a vacuous OR, so
// `list("s3://bucket/foo/bar", recursive=true)` returned `foo/other/deep.dat`
// along with `foo/bar`. The #154 fix made the last URI segment an (unanchored)
// regex matched against the key tail.
//
// Issue #213 REPLACES that contract. A plain URI is a LITERAL key prefix, never
// a regex; S3 applies the prefix, and the client does no filtering of its own.
// What #154 guaranteed -- `foo/other/deep.dat` does not come back for
// `foo/bar` -- still holds, now because it does not start with `foo/bar`. The
// regex feature is kept, but only as an explicit pattern (see
// `tests/test_list_literal_prefix_213.rs`, `s3_cli_ls_literal_prefix_213.rs` and
// `tests/test_python_list_literal_213.py`).

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

/// A key that does not start with the URI's key part is never returned, and the
/// last segment is a literal prefix: `foo/bar` lists `foo/bar` and `foo/barn`
/// but not `foo/other/deep.dat`, `foo/xbar` or `foo/b-a-r`.
#[tokio::test]
async fn list_returns_exactly_the_keys_that_start_with_the_literal_prefix() {
    let (port, state) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let unique = unique_prefix("b7-list-filter");
    let bucket = format!("lf-{}", unique.to_ascii_lowercase());
    let keys = [
        format!("{unique}/foo/bar"),
        format!("{unique}/foo/barn"),
        format!("{unique}/foo/bar/inner"),
        format!("{unique}/foo/other/deep.dat"),
        format!("{unique}/foo/xbar"),
        format!("{unique}/foo/b-a-r"),
    ];
    let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    state.seed_bucket(&bucket, &refs);

    let uri_prefix = format!("s3://{bucket}/{unique}/foo/bar");
    let listed = store
        .list(&uri_prefix, true)
        .await
        .expect("list() should succeed against the mock server");

    let want: Vec<String> = [
        format!("s3://{bucket}/{unique}/foo/bar"),
        format!("s3://{bucket}/{unique}/foo/bar/inner"),
        format!("s3://{bucket}/{unique}/foo/barn"),
    ]
    .to_vec();
    assert_eq!(
        listed, want,
        "list() must return exactly the keys that start with the literal prefix"
    );
}

/// The regex feature is kept, but only as an EXPLICIT pattern applied to the
/// literal-prefix listing (the shared `ListPattern` behind Python
/// `list(pattern=)` and `s3-cli ls --pattern`): it narrows the result, matched
/// against the key relative to the listing base, and never changes what is
/// asked of the server.
#[tokio::test]
async fn an_explicit_pattern_narrows_a_literal_prefix_listing() {
    use s3dlio::list_pattern::ListPattern;

    let (port, state) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let unique = unique_prefix("b7-list-pattern");
    let bucket = format!("lp-{}", unique.to_ascii_lowercase());
    let keys = [
        format!("{unique}/foo/a.dat"),
        format!("{unique}/foo/b.txt"),
        format!("{unique}/foo/sub/c.dat"),
    ];
    let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    state.seed_bucket(&bucket, &refs);

    let uri = format!("s3://{bucket}/{unique}/foo/");
    let mut listed = store.list(&uri, true).await.expect("list");
    assert_eq!(listed.len(), 3, "{listed:?}");

    ListPattern::new(r"\.dat$")
        .expect("pattern")
        .retain(&uri, &mut listed);
    assert_eq!(
        listed,
        vec![
            format!("s3://{bucket}/{unique}/foo/a.dat"),
            format!("s3://{bucket}/{unique}/foo/sub/c.dat"),
        ]
    );

    // the pattern is relative to the base: `^sub/` selects only the nested key
    let mut listed = store.list(&uri, true).await.expect("list");
    ListPattern::new("^sub/")
        .expect("pattern")
        .retain(&uri, &mut listed);
    assert_eq!(
        listed,
        vec![format!("s3://{bucket}/{unique}/foo/sub/c.dat")]
    );
}
