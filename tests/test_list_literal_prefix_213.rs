// tests/test_list_literal_prefix_213.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #213: a plain S3 URI is a LITERAL key prefix,
// never a regular expression.
//
// On v0.9.124 the S3 listings (`S3ObjectStore::list` / `list_stream` with a
// per-endpoint client, with the process-wide client, and the `s3_utils` helpers
// behind `s3_copy::download_objects` and `S3BytesDataset`) split the URI at its
// last '/', sent only the left part to the server, and compiled the remainder as
// an UNANCHORED regular expression over the keys that came back. So
//
//   * `s3://b/data/run.1` also listed `data/run_1` and `data/prerun.1`
//     (`.` matches any character, unanchored matches anywhere);
//   * a key with `(` or `[` in it made the listing fail with "Invalid regex";
//   * `s3://b/foo` (non-recursive) listed every top-level folder, because common
//     prefixes were added unfiltered.
//
// The contract: the whole key part of the URI is the prefix (LITERAL PREFIX is
// not EXACT: `run.1` also lists `run.1.old` and `run.1/…`); S3 applies it, and
// with the '/' delimiter returns the matching objects AND common prefixes. A
// regular expression only ever comes from an explicit pattern (`pattern=` /
// `--pattern`, or `s3_copy::download_objects_matching`).
//
// The mock S3 server answers from a seeded catalog that honours prefix,
// delimiter and paging like S3 does (`MockS3State::seed_bucket`).

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, MockS3State};
use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use std::sync::atomic::{AtomicU64, Ordering};

/// Every key of the shared catalog. Chosen so each assertion has a decoy that a
/// regex reading of the same URI would let through.
const CATALOG: &[&str] = &[
    // the reviewer's example: literal prefix `data/run.1`
    "data/run.1",
    "data/run.1.old",
    "data/run.1/child",
    "data/run_1",
    "data/prerun.1",
    "data/run.2",
    // a bare name at the bucket root
    "foo",
    "foobar",
    "foo/a",
    "foo/b/c",
    "afoo",
    "other/x",
    "zeta/top",
    // characters that are special in a regular expression
    "weird/p(",
    "weird/p(x",
    "weird/pp",
    "weird/q[",
    "weird/q[1]",
    "weird/qq",
    "weird/r+",
    "weird/r+1",
    "weird/rr",
    "weird/s$",
    "weird/s$1",
    "weird/ss",
    "weird/t?",
    "weird/t?1",
    "weird/tt",
    // download fixtures
    "dl/a.npz",
    "dl/b.npz",
    "dl/c.txt",
    "dl/sub/d.npz",
];

/// A fresh seeded bucket (a bucket per test: the mock is process-wide).
fn seeded_bucket() -> (u16, MockS3State, String) {
    static N: AtomicU64 = AtomicU64::new(0);
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("lit213-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, CATALOG);
    (port, state, bucket)
}

fn uris(bucket: &str, keys: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = keys.iter().map(|k| format!("s3://{bucket}/{k}")).collect();
    v.sort();
    v
}

/// `list` through BOTH S3 code paths: the per-endpoint client
/// (`list_with_client`) and the process-wide client (`list_objects_stream`).
async fn list_both_ways(port: u16, uri: &str, recursive: bool) -> Vec<(&'static str, Vec<String>)> {
    let per_endpoint = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let a = per_endpoint
        .list(uri, recursive)
        .await
        .unwrap_or_else(|e| panic!("per-endpoint list({uri}, {recursive}) failed: {e:#}"));
    let b = S3ObjectStore::new()
        .list(uri, recursive)
        .await
        .unwrap_or_else(|e| panic!("global-client list({uri}, {recursive}) failed: {e:#}"));
    // list_stream is a separate entry point over the same helpers
    let mut c: Vec<String> = Vec::new();
    {
        use futures::StreamExt;
        let mut s = per_endpoint.list_stream(uri, recursive);
        while let Some(item) = s.next().await {
            c.push(item.unwrap_or_else(|e| panic!("list_stream({uri}) failed: {e:#}")));
        }
        c.sort();
    }
    let mut d: Vec<String> = Vec::new();
    {
        use futures::StreamExt;
        let global = S3ObjectStore::new();
        let mut s = global.list_stream(uri, recursive);
        while let Some(item) = s.next().await {
            d.push(item.unwrap_or_else(|e| panic!("global list_stream({uri}) failed: {e:#}")));
        }
        d.sort();
    }
    vec![
        ("per-endpoint list", a),
        ("global-client list", b),
        ("per-endpoint list_stream", c),
        ("global-client list_stream", d),
    ]
}

async fn assert_lists(port: u16, uri: &str, recursive: bool, want: Vec<String>) {
    for (which, got) in list_both_ways(port, uri, recursive).await {
        assert_eq!(
            got, want,
            "{which}({uri}, recursive={recursive}): a URI is a LITERAL key prefix"
        );
    }
}

// ----------------------------------------------------------------------------
// The reviewer's example: literal PREFIX, not regex, not exact
// ----------------------------------------------------------------------------

#[tokio::test]
async fn run_dot_1_lists_its_prefix_matches_and_never_a_regex_look_alike() {
    let (port, _state, b) = seeded_bucket();
    let uri = format!("s3://{b}/data/run.1");

    // recursive: run.1, run.1.old, run.1/child -- NOT run_1, prerun.1, run.2
    assert_lists(
        port,
        &uri,
        true,
        uris(&b, &["data/run.1", "data/run.1.old", "data/run.1/child"]),
    )
    .await;

    // non-recursive: the two objects and the one directory `data/run.1/`
    assert_lists(
        port,
        &uri,
        false,
        uris(&b, &["data/run.1", "data/run.1.old", "data/run.1/"]),
    )
    .await;
}

#[tokio::test]
async fn the_whole_key_part_of_the_uri_is_what_the_server_is_asked_for() {
    let (port, state, b) = seeded_bucket();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    store
        .list(&format!("s3://{b}/data/run.1"), false)
        .await
        .expect("list");
    S3ObjectStore::new()
        .list(&format!("s3://{b}/data/run.1"), false)
        .await
        .expect("list");

    let lists: Vec<_> = state
        .requests_containing(&b)
        .into_iter()
        .filter(|r| r.method == "GET" && r.query.contains("list-type=2"))
        .collect();
    assert_eq!(lists.len(), 2, "one listing request per call: {lists:?}");
    for r in &lists {
        assert!(
            r.query.split('&').any(|kv| kv == "prefix=data%2Frun.1"),
            "the request prefix must be the whole literal key part `data/run.1`, got {:?}",
            r.query
        );
        assert!(
            r.query.split('&').any(|kv| kv == "delimiter=%2F"),
            "non-recursive uses the '/' delimiter: {:?}",
            r.query
        );
    }
}

// ----------------------------------------------------------------------------
// Characters that are regex syntax are ordinary characters of the prefix
// ----------------------------------------------------------------------------

#[tokio::test]
async fn regex_metacharacters_in_the_uri_are_literal_and_never_fail() {
    let (port, _state, b) = seeded_bucket();
    let cases: &[(&str, &[&str])] = &[
        ("weird/p(", &["weird/p(", "weird/p(x"]),
        ("weird/q[", &["weird/q[", "weird/q[1]"]),
        ("weird/r+", &["weird/r+", "weird/r+1"]),
        ("weird/s$", &["weird/s$", "weird/s$1"]),
        ("weird/t?", &["weird/t?", "weird/t?1"]),
    ];
    for (prefix, want) in cases {
        let uri = format!("s3://{b}/{prefix}");
        assert_lists(port, &uri, true, uris(&b, want)).await;
        assert_lists(port, &uri, false, uris(&b, want)).await;
    }
}

// ----------------------------------------------------------------------------
// A bare name: siblings and directories that START WITH it, nothing else
// ----------------------------------------------------------------------------

#[tokio::test]
async fn bare_name_non_recursive_returns_foo_foobar_and_foo_directory_only() {
    let (port, _state, b) = seeded_bucket();
    // NOT afoo, NOT data/, other/, weird/, zeta/, dl/ (unrelated top-level folders)
    assert_lists(
        port,
        &format!("s3://{b}/foo"),
        false,
        uris(&b, &["foo", "foobar", "foo/"]),
    )
    .await;
}

#[tokio::test]
async fn bare_name_recursive_returns_every_object_that_starts_with_it() {
    let (port, _state, b) = seeded_bucket();
    assert_lists(
        port,
        &format!("s3://{b}/foo"),
        true,
        uris(&b, &["foo", "foobar", "foo/a", "foo/b/c"]),
    )
    .await;
}

#[tokio::test]
async fn trailing_slash_prefix_is_unchanged() {
    let (port, _state, b) = seeded_bucket();
    assert_lists(
        port,
        &format!("s3://{b}/foo/"),
        false,
        uris(&b, &["foo/a", "foo/b/"]),
    )
    .await;
    assert_lists(
        port,
        &format!("s3://{b}/foo/"),
        true,
        uris(&b, &["foo/a", "foo/b/c"]),
    )
    .await;
}

#[tokio::test]
async fn a_long_listing_pages_through_every_key() {
    let (port, state, b) = seeded_bucket();
    let many: Vec<String> = (0..2500).map(|i| format!("pg/k{i:05}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    state.seed_bucket(&b, &refs);
    let want: Vec<String> = many.iter().map(|k| format!("s3://{b}/{k}")).collect();
    let uri = format!("s3://{b}/pg/k");
    for (which, got) in list_both_ways(port, &uri, true).await {
        assert_eq!(got.len(), want.len(), "{which} must follow every page");
        assert_eq!(got, want, "{which}");
    }
}

// ----------------------------------------------------------------------------
// Helpers behind s3_copy::download_objects and S3BytesDataset (s3_utils::list_objects)
// ----------------------------------------------------------------------------

/// `s3_copy` and `S3BytesDataset` call the synchronous helpers from their body,
/// which block on the global runtime: drive them from a separate runtime.
fn on_fresh_runtime<F, T>(f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(f)
}

fn downloaded_names(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

#[test]
fn s3_bytes_dataset_lists_the_literal_prefix_recursively() {
    use s3dlio::data_loader::s3_bytes::S3BytesDataset;
    let (_port, _state, b) = seeded_bucket();
    let ds = S3BytesDataset::from_prefix(&format!("s3://{b}/data/run.1")).expect("from_prefix");
    let mut keys = ds.keys().clone();
    keys.sort();
    assert_eq!(
        keys,
        vec!["data/run.1", "data/run.1.old", "data/run.1/child"],
        "S3BytesDataset::from_prefix takes the URI as a literal key prefix"
    );

    // a '(' in the prefix used to fail with "Invalid regex pattern"
    let ds = S3BytesDataset::from_prefix(&format!("s3://{b}/weird/p(")).expect("from_prefix");
    let mut keys = ds.keys().clone();
    keys.sort();
    assert_eq!(keys, vec!["weird/p(", "weird/p(x"]);
}

#[test]
fn download_a_prefix_directory_downloads_every_object_under_it() {
    let (_port, _state, b) = seeded_bucket();
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{b}/dl/"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    // every object under dl/ (recursive), by basename
    assert_eq!(
        downloaded_names(out.path()),
        vec!["a.npz", "b.npz", "c.txt", "d.npz"]
    );
}

#[test]
fn download_non_recursive_takes_the_direct_children_only() {
    let (_port, _state, b) = seeded_bucket();
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{b}/dl/"),
        out.path(),
        4,
        false,
        None,
    ))
    .expect("download");
    assert_eq!(
        downloaded_names(out.path()),
        vec!["a.npz", "b.npz", "c.txt"]
    );
}

#[test]
fn download_one_key_downloads_that_object_and_not_regex_look_alikes() {
    let (_port, _state, b) = seeded_bucket();
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{b}/dl/c.txt"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(downloaded_names(out.path()), vec!["c.txt"]);

    // `data/run.1` is a literal prefix: run.1 and run.1.old and run.1/child, not run_1 / prerun.1
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{b}/data/run.1"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(
        downloaded_names(out.path()),
        vec!["child", "run.1", "run.1.old"]
    );
}

#[test]
fn download_with_an_explicit_pattern_keeps_the_glob_regex_selection() {
    let (_port, _state, b) = seeded_bucket();

    // prefix `dl/` + pattern: only the .npz objects, recursively
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
        &format!("s3://{b}/dl/"),
        Some(r"\.npz$"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(
        downloaded_names(out.path()),
        vec!["a.npz", "b.npz", "d.npz"]
    );

    // the pattern sees the key relative to the listing base (`sub/d.npz`)
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
        &format!("s3://{b}/dl/"),
        Some(r"^sub/"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(downloaded_names(out.path()), vec!["d.npz"]);

    // a pattern nothing matches is the same "no objects" error as an empty prefix
    let out = tempfile::tempdir().expect("tempdir");
    let err = on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
        &format!("s3://{b}/dl/"),
        Some(r"\.parquet$"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect_err("nothing matches");
    assert!(format!("{err:#}").contains("No objects matched"), "{err:#}");

    // an invalid pattern is an error, raised before any download
    let out = tempfile::tempdir().expect("tempdir");
    let err = on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
        &format!("s3://{b}/dl/"),
        Some("("),
        out.path(),
        4,
        true,
        None,
    ))
    .expect_err("invalid pattern");
    assert!(
        format!("{err:#}").contains("Invalid regex pattern"),
        "{err:#}"
    );
}
