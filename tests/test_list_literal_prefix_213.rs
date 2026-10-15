// tests/test_list_literal_prefix_213.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #213: the key part of an `s3://` URI selects the
// object with exactly that name (and what is under `name/`), a directory (`name/`), or
// an explicit `*` / `**` glob -- it is never a regular expression and never a bare
// string prefix.
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
// The contract (architect ruling, #213): `s3://b/K` with no `*` selects the object
// `K` and, when descending, everything under `K/` -- NEVER `K.bak`, `K2`, `Kfoo`, or
// `K.d/`. A regular expression only ever comes from an explicit pattern (`pattern=` /
// `--pattern`, or `s3_copy::download_objects_matching`). The wildcard rule has its own
// tests in `test_s3_selection_213.rs`.
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
            "{which}({uri}, recursive={recursive}): a URI selects the exact object and its directory"
        );
    }
}

// ----------------------------------------------------------------------------
// The reviewer's example: literal PREFIX, not regex, not exact
// ----------------------------------------------------------------------------

#[tokio::test]
async fn run_dot_1_lists_the_object_and_its_directory_and_never_a_sibling() {
    let (port, _state, b) = seeded_bucket();
    let uri = format!("s3://{b}/data/run.1");

    // recursive: run.1 and run.1/child -- NOT run.1.old, run_1, prerun.1, run.2
    assert_lists(
        port,
        &uri,
        true,
        uris(&b, &["data/run.1", "data/run.1/child"]),
    )
    .await;

    // non-recursive: the object `data/run.1` and the one directory `data/run.1/`
    assert_lists(port, &uri, false, uris(&b, &["data/run.1", "data/run.1/"])).await;
}

#[tokio::test]
async fn an_exact_key_is_two_probes_that_cannot_return_siblings() {
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
    assert_eq!(
        lists.len(),
        4,
        "two probe requests per non-recursive call: {lists:?}"
    );
    for r in &lists {
        assert!(
            r.query.split('&').any(|kv| kv == "max-keys=1"),
            "every request is a max-keys=1 probe, so no sibling can be paged through: {:?}",
            r.query
        );
        assert!(
            !r.query.split('&').any(|kv| kv.starts_with("delimiter=")),
            "a probe needs no delimiter: {:?}",
            r.query
        );
    }
    let prefixes: std::collections::BTreeSet<String> = lists
        .iter()
        .filter_map(|r| {
            r.query
                .split('&')
                .find_map(|kv| kv.strip_prefix("prefix=").map(str::to_string))
        })
        .collect();
    assert_eq!(
        prefixes,
        ["data%2Frun.1".to_string(), "data%2Frun.1%2F".to_string()]
            .into_iter()
            .collect(),
        "the object `data/run.1` and the directory `data/run.1/`, nothing else"
    );
}

// ----------------------------------------------------------------------------
// Characters that are regex syntax are ordinary characters of the prefix
// ----------------------------------------------------------------------------

#[tokio::test]
async fn regex_metacharacters_in_the_uri_are_literal_and_never_fail() {
    let (port, _state, b) = seeded_bucket();
    let cases: &[(&str, &[&str])] = &[
        ("weird/p(", &["weird/p("]),
        ("weird/q[", &["weird/q["]),
        ("weird/r+", &["weird/r+"]),
        ("weird/s$", &["weird/s$"]),
        ("weird/t?", &["weird/t?"]),
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
async fn bare_name_non_recursive_returns_foo_and_the_foo_directory_only() {
    let (port, _state, b) = seeded_bucket();
    // NOT foobar, afoo, data/, other/, weird/, zeta/, dl/
    assert_lists(
        port,
        &format!("s3://{b}/foo"),
        false,
        uris(&b, &["foo", "foo/"]),
    )
    .await;
}

#[tokio::test]
async fn bare_name_recursive_returns_the_object_and_everything_under_it() {
    let (port, _state, b) = seeded_bucket();
    assert_lists(
        port,
        &format!("s3://{b}/foo"),
        true,
        uris(&b, &["foo", "foo/a", "foo/b/c"]),
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
    let uri = format!("s3://{b}/pg/");
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
fn s3_bytes_dataset_lists_the_exact_object_and_its_directory_recursively() {
    use s3dlio::data_loader::s3_bytes::S3BytesDataset;
    let (_port, _state, b) = seeded_bucket();
    let ds = S3BytesDataset::from_prefix(&format!("s3://{b}/data/run.1")).expect("from_prefix");
    let mut keys = ds.keys().clone();
    keys.sort();
    assert_eq!(
        keys,
        vec!["data/run.1", "data/run.1/child"],
        "S3BytesDataset::from_prefix selects the exact object and what is under it"
    );

    // a '(' in the prefix used to fail with "Invalid regex pattern"
    let ds = S3BytesDataset::from_prefix(&format!("s3://{b}/weird/p(")).expect("from_prefix");
    let mut keys = ds.keys().clone();
    keys.sort();
    assert_eq!(keys, vec!["weird/p("]);
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
fn download_one_key_downloads_that_object_and_its_directory_and_no_sibling() {
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

    // `data/run.1` is the object and its directory: run.1 and run.1/child, not run.1.old / run_1 / prerun.1
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{b}/data/run.1"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(downloaded_names(out.path()), vec!["child", "run.1"]);
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

// ----------------------------------------------------------------------------
// Host-form URIs (`s3://host:port/bucket/key`) and an explicit pattern
//
// `S3ObjectStore::list` returns `s3://{bucket}/{key}` WITHOUT the host:port, so the
// pattern base must be worked out from the KEY parts, not by comparing whole URIs.
// Python `list(pattern=)`, `s3-cli ls -p` and `download_objects_matching` all use the
// same rule and must agree.
// ----------------------------------------------------------------------------

#[tokio::test]
async fn host_form_uri_with_a_pattern_selects_relative_to_the_listing_base() {
    use s3dlio::list_pattern::ListPattern;
    let (port, _state, b) = seeded_bucket();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");

    // What Python `list(uri, recursive=True, pattern=...)` does: list, then retain.
    for uri in [
        format!("s3://127.0.0.1:{port}/{b}/dl/"),
        format!("s3://127.0.0.1:{port}/{b}/dl"),
    ] {
        let mut keys = store.list(&uri, true).await.expect("list");
        // the listing itself is unchanged: bucket-form entries
        assert_eq!(
            keys,
            uris(&b, &["dl/a.npz", "dl/b.npz", "dl/c.txt", "dl/sub/d.npz"]),
            "list({uri})"
        );
        ListPattern::new("^sub/")
            .expect("pattern")
            .retain(&uri, &mut keys);
        assert_eq!(
            keys,
            uris(&b, &["dl/sub/d.npz"]),
            "pattern '^sub/' on {uri} is matched against the key relative to the listing base"
        );
    }

    // a non-slash key of a host-form URI: the object is relative to the parent directory,
    // what is under it to the directory itself
    let uri = format!("s3://127.0.0.1:{port}/{b}/data/run.1");
    let mut keys = store.list(&uri, true).await.expect("list");
    ListPattern::new("^child$")
        .expect("pattern")
        .retain(&uri, &mut keys);
    assert_eq!(keys, uris(&b, &["data/run.1/child"]), "{uri}");
}

#[test]
fn host_form_uri_download_matching_agrees_with_list_on_the_pattern() {
    let (port, _state, b) = seeded_bucket();
    let out = tempfile::tempdir().expect("tempdir");
    on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
        &format!("s3://127.0.0.1:{port}/{b}/dl/"),
        Some(r"^sub/"),
        out.path(),
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(downloaded_names(out.path()), vec!["d.npz"]);
}

// ----------------------------------------------------------------------------
// Edge shapes of the prefix: empty, a space, non-ASCII, non-slash prefix + pattern
// ----------------------------------------------------------------------------

#[tokio::test]
async fn empty_prefix_lists_the_whole_bucket_top_level_or_recursively() {
    let (port, _state, b) = seeded_bucket();
    // `s3://b/`: non-recursive = the top-level objects plus one entry per top-level folder
    assert_lists(
        port,
        &format!("s3://{b}/"),
        false,
        uris(
            &b,
            &[
                "afoo", "data/", "dl/", "foo", "foo/", "foobar", "other/", "weird/", "zeta/",
            ],
        ),
    )
    .await;
    // recursive = every key of the bucket
    assert_lists(port, &format!("s3://{b}/"), true, uris(&b, CATALOG)).await;
}

#[tokio::test]
async fn a_space_and_a_non_ascii_character_in_a_key_are_literal() {
    let (port, state, b) = seeded_bucket();
    let keys: &[&str] = &[
        "sp ace/a b.txt",
        "sp ace/c.txt",
        "sp acer",
        "sp_ace/x",
        "uni/caf\u{e9}-\u{65e5}\u{672c}.dat",
        "uni/caf\u{e9}.dat",
        "uni/cafe.dat",
    ];
    state.seed_bucket(&b, keys);
    assert_lists(
        port,
        &format!("s3://{b}/sp ace"),
        true,
        uris(&b, &["sp ace/a b.txt", "sp ace/c.txt"]),
    )
    .await;
    assert_lists(
        port,
        &format!("s3://{b}/sp ace/"),
        false,
        uris(&b, &["sp ace/a b.txt", "sp ace/c.txt"]),
    )
    .await;
    // `uni/caf\u{e9}` names no object and no directory (it is not a prefix match)
    assert_lists(port, &format!("s3://{b}/uni/caf\u{e9}"), true, vec![]).await;
    // a non-ASCII key is selected by its exact name only
    assert_lists(
        port,
        &format!("s3://{b}/uni/caf\u{e9}-\u{65e5}\u{672c}.dat"),
        false,
        uris(&b, &["uni/caf\u{e9}-\u{65e5}\u{672c}.dat"]),
    )
    .await;
    assert_lists(
        port,
        &format!("s3://{b}/uni/caf\u{e9}.dat"),
        true,
        uris(&b, &["uni/caf\u{e9}.dat"]),
    )
    .await;
}

#[test]
fn download_matching_with_a_non_slash_prefix_and_a_pattern() {
    let (_port, _state, b) = seeded_bucket();
    // key `data/run.1` (no trailing slash): the selection is the object `run.1` and
    // `run.1/child`; the pattern is matched against the entry relative to the listing base.
    let cases: &[(&str, &[&str])] = &[
        // the object lies beside its siblings: relative to the parent `data/`
        (r"^run\.1$", &["run.1"]),
        // `run.1/child` lies under the directory `run.1/`: relative to it
        (r"^child$", &["child"]),
        // nothing outside the selection is ever offered to the pattern
        (r"run_1|prerun|\.old", &[]),
    ];
    for (pattern, want) in cases {
        let out = tempfile::tempdir().expect("tempdir");
        let res = on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
            &format!("s3://{b}/data/run.1"),
            Some(pattern),
            out.path(),
            4,
            true,
            None,
        ));
        if want.is_empty() {
            assert!(res.is_err(), "pattern {pattern}: nothing may match");
        } else {
            res.unwrap_or_else(|e| panic!("pattern {pattern}: {e:#}"));
        }
        assert_eq!(downloaded_names(out.path()), *want, "pattern {pattern}");
    }
}
