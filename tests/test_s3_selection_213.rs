// tests/test_s3_selection_213.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Issue #213: the s3:// selection rule. Selecting `file.txt` never also selects
// `file.txt.bak`; only the explicit glob `file.txt*` does.
//
// The rule for an `s3://` URI whose key part is `K` (src/s3_selection.rs):
//
//   1. no `*`, not ending in `/`: the object whose key is exactly `K`, plus -- for
//      operations that descend -- every key under `K/`. Non-recursive: the object `K`
//      and the directory entry `K/`. NEVER `K.bak`, `K2`, `Kfoo`, `K.d/`.
//   2. `K` ending in `/`, or empty: everything under it.
//   3. `*` in `K`: an explicit glob over the FULL key. `*` = any run except `/`, `**` =
//      any run including `/`; every other character is literal. The S3 request prefix is
//      the text before the first `*`.
//   4. an explicit `--pattern` / `pattern=` regex applies AFTER the selection.
//
// These tests drive every selection path of the library against the in-process mock
// S3 server with a seeded catalog (prefix, delimiter, max-keys and paging honoured, and
// DeleteObjects removing keys): `S3ObjectStore::list` / `list_stream` (per-endpoint and
// process-wide client), `s3_copy::download_objects[_matching]`,
// `S3BytesDataset::from_prefix`, and the recursive delete of `delete_contract`. The CLI
// binary and the Python module have their own tests.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, MockS3State};
use s3dlio::delete_contract::{delete_objects_under_literal_prefix, DeleteMode};
use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use std::sync::atomic::{AtomicU64, Ordering};

/// Decoys that a bare-prefix or regex reading would let through, plus a directory and wildcard fixtures.
const CATALOG: &[&str] = &[
    "file.txt",
    "file.txt.bak",
    "file.txt2",
    "file.txt.d/x",
    "dl/a.npz",
    "dl/b.txt",
    "dl/sub/c.npz",
    "dl2/x",
    "run.1",
    "run.1/child",
    "run.1.old",
    "run.1.d/x",
    "prerun.1",
    "run_1",
    "data/top.npz",
    "data/top.txt",
    "data/sub/deep/z.npz",
    "g*/q",
    "gx/q",
];

fn seeded() -> (u16, MockS3State, String) {
    static N: AtomicU64 = AtomicU64::new(0);
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("sel213-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, CATALOG);
    (port, state, bucket)
}

fn uris(bucket: &str, keys: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = keys.iter().map(|k| format!("s3://{bucket}/{k}")).collect();
    v.sort();
    v
}

fn keys_of(keys: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = keys.iter().map(|k| (*k).to_string()).collect();
    v.sort();
    v
}

/// `list` through all four S3 code paths: per-endpoint `list` / `list_stream`, and the
/// process-wide client's `list` / `list_stream`.
async fn list_all_ways(port: u16, uri: &str, recursive: bool) -> Vec<(&'static str, Vec<String>)> {
    use futures::StreamExt;
    let per_endpoint = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let global = S3ObjectStore::new();
    let a = per_endpoint
        .list(uri, recursive)
        .await
        .unwrap_or_else(|e| panic!("per-endpoint list({uri}) failed: {e:#}"));
    let b = global
        .list(uri, recursive)
        .await
        .unwrap_or_else(|e| panic!("global list({uri}) failed: {e:#}"));
    let mut c = Vec::new();
    let mut s = per_endpoint.list_stream(uri, recursive);
    while let Some(item) = s.next().await {
        c.push(item.unwrap_or_else(|e| panic!("per-endpoint list_stream({uri}) failed: {e:#}")));
    }
    c.sort();
    let mut d = Vec::new();
    let mut s = global.list_stream(uri, recursive);
    while let Some(item) = s.next().await {
        d.push(item.unwrap_or_else(|e| panic!("global list_stream({uri}) failed: {e:#}")));
    }
    d.sort();
    vec![
        ("per-endpoint list", a),
        ("global list", b),
        ("per-endpoint list_stream", c),
        ("global list_stream", d),
    ]
}

async fn assert_selects(port: u16, uri: &str, recursive: bool, want: Vec<String>) {
    for (which, got) in list_all_ways(port, uri, recursive).await {
        assert_eq!(got, want, "{which}({uri}, recursive={recursive})");
    }
}

/// The ListObjectsV2 requests the mock saw for `bucket`.
fn list_requests(
    state: &MockS3State,
    bucket: &str,
) -> Vec<common::mock_s3_server::RecordedRequest> {
    state
        .requests_containing(bucket)
        .into_iter()
        .filter(|r| r.method == "GET" && r.query.contains("list-type=2"))
        .collect()
}

fn param(query: &str, name: &str) -> Option<String> {
    let want = format!("{name}=");
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix(want.as_str()).map(str::to_string))
}

// ----------------------------------------------------------------------------
// Rule 1: an exact name selects the object and its directory, never a sibling
// ----------------------------------------------------------------------------

#[tokio::test]
async fn file_txt_selects_only_file_txt() {
    let (port, _s, b) = seeded();
    let uri = format!("s3://{b}/file.txt");
    // not file.txt.bak, file.txt2, file.txt.d/x
    assert_selects(port, &uri, true, uris(&b, &["file.txt"])).await;
    assert_selects(port, &uri, false, uris(&b, &["file.txt"])).await;
}

#[tokio::test]
async fn run_dot_1_selects_the_object_and_its_directory_only() {
    let (port, _s, b) = seeded();
    let uri = format!("s3://{b}/run.1");
    // never run.1.old, run.1.d/x, prerun.1, run_1
    assert_selects(port, &uri, true, uris(&b, &["run.1", "run.1/child"])).await;
    assert_selects(port, &uri, false, uris(&b, &["run.1", "run.1/"])).await;
}

#[tokio::test]
async fn a_directory_name_selects_the_directory_and_never_dl2() {
    let (port, _s, b) = seeded();
    let uri = format!("s3://{b}/dl");
    assert_selects(
        port,
        &uri,
        true,
        uris(&b, &["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"]),
    )
    .await;
    // non-recursive: no object `dl`, one directory entry
    assert_selects(port, &uri, false, uris(&b, &["dl/"])).await;
}

#[tokio::test]
async fn a_name_that_is_neither_an_object_nor_a_directory_selects_nothing() {
    let (port, _s, b) = seeded();
    // `file.tx` and `ru` are prefixes of keys, but name neither an object nor a directory
    for key in ["file.tx", "ru", "file.txt.ba", "nothing"] {
        let uri = format!("s3://{b}/{key}");
        assert_selects(port, &uri, true, vec![]).await;
        assert_selects(port, &uri, false, vec![]).await;
    }
}

#[tokio::test]
async fn regex_characters_are_literal_and_run_dot_1_never_matches_run_1() {
    let (port, _s, b) = seeded();
    for key in ["run.1", "ru.1", "run[.]1", "run.1$", "(run.1)"] {
        let got = list_all_ways(port, &format!("s3://{b}/{key}"), true).await;
        for (which, keys) in got {
            assert!(
                !keys
                    .iter()
                    .any(|k| k.ends_with("/run_1") || k.ends_with("/prerun.1")),
                "{which}({key}): {keys:?}"
            );
        }
    }
    assert_selects(port, &format!("s3://{b}/ru.1"), true, vec![]).await;
}

// ----------------------------------------------------------------------------
// Rule 2: a trailing '/' and the bucket root are unchanged
// ----------------------------------------------------------------------------

#[tokio::test]
async fn trailing_slash_and_the_bucket_root_are_unchanged() {
    let (port, _s, b) = seeded();
    assert_selects(
        port,
        &format!("s3://{b}/dl/"),
        false,
        uris(&b, &["dl/a.npz", "dl/b.txt", "dl/sub/"]),
    )
    .await;
    assert_selects(
        port,
        &format!("s3://{b}/dl/"),
        true,
        uris(&b, &["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"]),
    )
    .await;
    assert_selects(port, &format!("s3://{b}/"), true, uris(&b, CATALOG)).await;
    assert_selects(
        port,
        &format!("s3://{b}/"),
        false,
        uris(
            &b,
            &[
                "data/",
                "dl/",
                "dl2/",
                "file.txt",
                "file.txt.bak",
                "file.txt.d/",
                "file.txt2",
                "g*/",
                "gx/",
                "prerun.1",
                "run.1",
                "run.1.d/",
                "run.1.old",
                "run.1/",
                "run_1",
            ],
        ),
    )
    .await;
}

// ----------------------------------------------------------------------------
// Rule 3: `*` is an explicit glob over the full key
// ----------------------------------------------------------------------------

#[tokio::test]
async fn star_selects_the_same_directory_level() {
    let (port, _s, b) = seeded();
    let want = uris(&b, &["file.txt", "file.txt.bak", "file.txt2"]);
    // not file.txt.d/x: `*` does not cross '/'; -r makes no difference for a wildcard
    assert_selects(port, &format!("s3://{b}/file.txt*"), true, want.clone()).await;
    assert_selects(port, &format!("s3://{b}/file.txt*"), false, want).await;
}

#[tokio::test]
async fn single_star_does_not_cross_slash_but_double_star_does() {
    let (port, _s, b) = seeded();
    for recursive in [true, false] {
        assert_selects(
            port,
            &format!("s3://{b}/data/*.npz"),
            recursive,
            uris(&b, &["data/top.npz"]),
        )
        .await;
        assert_selects(
            port,
            &format!("s3://{b}/data/**.npz"),
            recursive,
            uris(&b, &["data/sub/deep/z.npz", "data/top.npz"]),
        )
        .await;
        assert_selects(
            port,
            &format!("s3://{b}/data/**"),
            recursive,
            uris(&b, &["data/sub/deep/z.npz", "data/top.npz", "data/top.txt"]),
        )
        .await;
    }
    // a `*` in a directory position matches one level
    assert_selects(
        port,
        &format!("s3://{b}/d*/b.txt"),
        true,
        uris(&b, &["dl/b.txt"]),
    )
    .await;
}

#[tokio::test]
async fn other_characters_in_a_glob_are_literal() {
    let (port, _s, b) = seeded();
    // `run.1*` is a glob whose `.` is a dot: run_1 and prerun.1 are not selected
    assert_selects(
        port,
        &format!("s3://{b}/run.1*"),
        true,
        uris(&b, &["run.1", "run.1.old"]),
    )
    .await;
    // `?` is not a wildcard
    assert_selects(port, &format!("s3://{b}/file.tx?*"), true, vec![]).await;
    // `g*/q` is a glob: its `*` matches both the key `g*/q` and `gx/q`
    assert_selects(
        port,
        &format!("s3://{b}/g*/q"),
        true,
        uris(&b, &["g*/q", "gx/q"]),
    )
    .await;
}

#[tokio::test]
async fn a_wildcard_key_ending_in_slash_is_refused_and_lists_nothing() {
    let (port, state, b) = seeded();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let err = store
        .list(&format!("s3://{b}/data/*/"), true)
        .await
        .expect_err("a wildcard key ending in '/' is not defined");
    assert!(format!("{err:#}").contains("not defined"), "{err:#}");
    assert!(
        list_requests(&state, &b).is_empty(),
        "nothing may be listed"
    );
}

// ----------------------------------------------------------------------------
// Requests: no sibling is ever paged through, no HEAD
// ----------------------------------------------------------------------------

#[tokio::test]
async fn request_counts_per_operation() {
    let (port, state, b) = seeded();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");

    // (uri, recursive, expected request count, expected prefixes in order)
    type Shape = &'static [(&'static str, bool)];
    let cases: &[(&str, bool, usize, Shape)] = &[
        // exact key, descending: the existence probe, then the pages of `K/`
        (
            "file.txt",
            true,
            2,
            &[("file.txt", true), ("file.txt%2F", false)],
        ),
        // exact key, not descending: two probes
        (
            "file.txt",
            false,
            2,
            &[("file.txt", true), ("file.txt%2F", true)],
        ),
        // directory: the pages of the listing
        ("dl/", true, 1, &[("dl%2F", false)]),
        // wildcard: one listing of the text before the first `*`
        ("file.txt*", true, 1, &[("file.txt", false)]),
        ("data/**.npz", true, 1, &[("data%2F", false)]),
    ];
    for (key, recursive, count, shape) in cases {
        let before = list_requests(&state, &b).len();
        store
            .list(&format!("s3://{b}/{key}"), *recursive)
            .await
            .expect("list");
        let all = list_requests(&state, &b);
        let mine = &all[before..];
        assert_eq!(mine.len(), *count, "{key} recursive={recursive}: {mine:?}");
        for (r, (prefix, probe)) in mine.iter().zip(shape.iter()) {
            assert_eq!(param(&r.query, "prefix").as_deref(), Some(*prefix), "{r:?}");
            assert_eq!(
                param(&r.query, "max-keys").as_deref(),
                if *probe { Some("1") } else { None },
                "{key}: {r:?}"
            );
        }
    }
    assert!(
        !state
            .requests_containing(&b)
            .iter()
            .any(|r| r.method == "HEAD"),
        "no HEAD is ever sent"
    );

    // a single-level glob lists with the '/' delimiter, so the subtree is not read
    let before = list_requests(&state, &b).len();
    store
        .list(&format!("s3://{b}/data/*.npz"), true)
        .await
        .expect("list");
    let all = list_requests(&state, &b);
    assert_eq!(all.len() - before, 1);
    assert_eq!(
        param(&all[before].query, "delimiter").as_deref(),
        Some("%2F"),
        "{:?}",
        all[before]
    );
}

#[tokio::test]
async fn an_exact_key_with_ten_thousand_siblings_still_costs_two_requests() {
    let (port, state, b) = seeded();
    let mut keys: Vec<String> = (0..10_000).map(|i| format!("big/k.{i:05}")).collect();
    keys.push("big/k".to_string());
    keys.push("big/k/child".to_string());
    let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    state.seed_bucket(&b, &refs);
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let got = store
        .list(&format!("s3://{b}/big/k"), true)
        .await
        .expect("list");
    assert_eq!(got, uris(&b, &["big/k", "big/k/child"]));
    assert_eq!(list_requests(&state, &b).len(), 2);
}

#[tokio::test]
async fn a_long_wildcard_listing_pages_through_every_match() {
    let (port, state, b) = seeded();
    let many: Vec<String> = (0..2500).map(|i| format!("pg/k{i:05}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    state.seed_bucket(&b, &refs);
    let want: Vec<String> = many.iter().map(|k| format!("s3://{b}/{k}")).collect();
    for (which, got) in list_all_ways(port, &format!("s3://{b}/pg/k*"), true).await {
        assert_eq!(got.len(), want.len(), "{which} must follow every page");
        assert_eq!(got, want, "{which}");
    }
    // an exact directory pages too
    let nested: Vec<String> = (0..2500).map(|i| format!("pg/k/{i:05}")).collect();
    let refs: Vec<&str> = nested.iter().map(String::as_str).collect();
    state.seed_bucket(&b, &refs);
    for (which, got) in list_all_ways(port, &format!("s3://{b}/pg/k"), true).await {
        assert_eq!(got.len(), nested.len(), "{which} must follow every page");
    }
}

// ----------------------------------------------------------------------------
// download, S3BytesDataset (blocking helpers: drive them from a separate runtime)
// ----------------------------------------------------------------------------

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

fn download(uri: &str, pattern: Option<&str>, recursive: bool) -> Vec<String> {
    let out = tempfile::tempdir().expect("tempdir");
    let res = on_fresh_runtime(s3dlio::s3_copy::download_objects_matching(
        uri,
        pattern,
        out.path(),
        4,
        recursive,
        None,
    ));
    if let Err(e) = res {
        assert!(
            format!("{e:#}").contains("No objects matched"),
            "unexpected failure for {uri}: {e:#}"
        );
    }
    downloaded_names(out.path())
}

#[test]
fn download_of_file_txt_takes_only_file_txt() {
    let (_p, state, b) = seeded();
    for recursive in [true, false] {
        assert_eq!(
            download(&format!("s3://{b}/file.txt"), None, recursive),
            keys_of(&["file.txt"]),
            "recursive={recursive}"
        );
    }
    // the only object GETs are for file.txt itself
    let gets: std::collections::BTreeSet<String> = state
        .requests_containing(&b)
        .into_iter()
        .filter(|r| r.method == "GET" && !r.query.contains("list-type"))
        .map(|r| r.path)
        .collect();
    assert_eq!(
        gets,
        [format!("/{b}/file.txt")].into_iter().collect(),
        "no sibling was fetched"
    );
}

#[test]
fn download_of_run_dot_1_takes_the_object_and_its_directory() {
    let (_p, _s, b) = seeded();
    assert_eq!(
        download(&format!("s3://{b}/run.1"), None, true),
        keys_of(&["child", "run.1"])
    );
    // non-recursive: the object (the `run.1/` entry is a directory and is skipped)
    assert_eq!(
        download(&format!("s3://{b}/run.1"), None, false),
        keys_of(&["run.1"])
    );
}

#[test]
fn download_of_a_wildcard_takes_the_glob_matches() {
    let (_p, _s, b) = seeded();
    assert_eq!(
        download(&format!("s3://{b}/file.txt*"), None, true),
        keys_of(&["file.txt", "file.txt.bak", "file.txt2"])
    );
    assert_eq!(
        download(&format!("s3://{b}/data/*.npz"), None, true),
        keys_of(&["top.npz"])
    );
    assert_eq!(
        download(&format!("s3://{b}/data/**.npz"), None, true),
        keys_of(&["top.npz", "z.npz"])
    );
}

#[test]
fn download_of_a_directory_name_does_not_cross_into_dl2() {
    let (_p, _s, b) = seeded();
    assert_eq!(
        download(&format!("s3://{b}/dl"), None, true),
        keys_of(&["a.npz", "b.txt", "c.npz"])
    );
}

#[test]
fn an_explicit_pattern_applies_after_the_selection() {
    let (_p, _s, b) = seeded();
    // `.bak` is only reachable through the wildcard; the pattern then narrows it
    assert_eq!(
        download(&format!("s3://{b}/file.txt*"), Some(r"\.bak$"), true),
        keys_of(&["file.txt.bak"])
    );
    // the exact name never offers `file.txt.bak` to the pattern
    assert!(download(&format!("s3://{b}/file.txt"), Some(r"\.bak$"), true).is_empty());
    assert_eq!(
        download(&format!("s3://{b}/dl/"), Some(r"\.npz$"), true),
        keys_of(&["a.npz", "c.npz"])
    );
    assert_eq!(
        download(&format!("s3://{b}/dl"), Some(r"^sub/"), true),
        keys_of(&["c.npz"])
    );
}

#[test]
fn s3_bytes_dataset_follows_the_same_rule() {
    use s3dlio::data_loader::s3_bytes::S3BytesDataset;
    let (_p, _s, b) = seeded();
    let keys = |uri: String| {
        let mut k = S3BytesDataset::from_prefix(&uri)
            .expect("from_prefix")
            .keys()
            .clone();
        k.sort();
        k
    };
    assert_eq!(keys(format!("s3://{b}/file.txt")), vec!["file.txt"]);
    assert_eq!(
        keys(format!("s3://{b}/file.txt*")),
        vec!["file.txt", "file.txt.bak", "file.txt2"]
    );
    assert_eq!(
        keys(format!("s3://{b}/run.1")),
        vec!["run.1", "run.1/child"]
    );
    assert_eq!(keys(format!("s3://{b}/dl2")), vec!["dl2/x"]);
}

// ----------------------------------------------------------------------------
// delete -r (the library behind `s3-cli delete -r` and Python delete(recursive=True))
// ----------------------------------------------------------------------------

async fn delete_recursively(port: u16, uri: &str) {
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    delete_objects_under_literal_prefix(&store, uri, None, DeleteMode::Batched, None)
        .await
        .into_result(uri, None)
        .unwrap_or_else(|e| panic!("delete -r {uri}: {e:#}"));
}

fn without(removed: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = CATALOG
        .iter()
        .filter(|k| !removed.contains(k))
        .map(|k| (*k).to_string())
        .collect();
    v.sort();
    v
}

#[tokio::test]
async fn delete_r_of_file_txt_leaves_file_txt_bak() {
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://{b}/file.txt")).await;
    assert_eq!(
        state.remaining_keys(&b),
        without(&["file.txt"]),
        "only file.txt may be gone: file.txt.bak, file.txt2 and file.txt.d/x survive"
    );
}

#[tokio::test]
async fn delete_r_of_run_dot_1_takes_the_object_and_its_directory_only() {
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://{b}/run.1")).await;
    assert_eq!(state.remaining_keys(&b), without(&["run.1", "run.1/child"]));
}

#[tokio::test]
async fn delete_r_of_a_directory_name_does_not_cross_into_dl2() {
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://{b}/dl")).await;
    assert_eq!(
        state.remaining_keys(&b),
        without(&["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );
}

#[tokio::test]
async fn delete_r_trailing_slash_is_unchanged() {
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://{b}/dl/")).await;
    assert_eq!(
        state.remaining_keys(&b),
        without(&["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );
}

#[tokio::test]
async fn delete_r_of_a_wildcard_is_an_explicit_glob() {
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://{b}/file.txt*")).await;
    assert_eq!(
        state.remaining_keys(&b),
        without(&["file.txt", "file.txt.bak", "file.txt2"]),
        "file.txt.d/x is another directory level"
    );
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://{b}/data/**.npz")).await;
    assert_eq!(
        state.remaining_keys(&b),
        without(&["data/top.npz", "data/sub/deep/z.npz"])
    );
}

#[tokio::test]
async fn delete_r_of_a_wildcard_key_ending_in_slash_deletes_nothing() {
    let (port, state, b) = seeded();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let uri = format!("s3://{b}/data/*/");
    let report =
        delete_objects_under_literal_prefix(&store, &uri, None, DeleteMode::Batched, None).await;
    let err = report.into_result(&uri, None).expect_err("must be refused");
    assert!(format!("{err:#}").contains("not defined"), "{err:#}");
    assert_eq!(state.remaining_keys(&b), without(&[]));
}

/// `ObjectStore::delete_prefix` keeps its own contract: it appends '/' and reads the
/// prefix literally (a `*` in it is a key character, not a glob).
#[tokio::test]
async fn delete_prefix_stays_a_literal_directory_prefix() {
    let (port, state, b) = seeded();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    store
        .delete_prefix(&format!("s3://{b}/file.txt"))
        .await
        .expect("delete_prefix");
    // nothing is under `file.txt/`, and `file.txt` itself and its siblings are untouched
    assert_eq!(state.remaining_keys(&b), without(&[]));

    // a `*` in the prefix is literal here: only `g*/q` goes, not `gx/q`
    let (port, state, b) = seeded();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    store
        .delete_prefix(&format!("s3://{b}/g*"))
        .await
        .expect("delete_prefix");
    assert_eq!(state.remaining_keys(&b), without(&["g*/q"]));
    // and so does the process-wide client
    let (_port, state, b) = seeded();
    S3ObjectStore::new()
        .delete_prefix(&format!("s3://{b}/g*"))
        .await
        .expect("delete_prefix");
    assert_eq!(state.remaining_keys(&b), without(&["g*/q"]));
}

// ----------------------------------------------------------------------------
// Review follow-ups (F2, F10): pagination end, host-form globs, leading-'/' keys
// ----------------------------------------------------------------------------

/// A bucket whose name carries one of the mock's behaviour markers, seeded with `keys`.
fn seeded_marked(marker: &str, keys: &[&str]) -> (u16, MockS3State, String) {
    static N: AtomicU64 = AtomicU64::new(0);
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("sel213-{marker}-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, keys);
    (port, state, bucket)
}

/// A server that sends a (constant) continuation token on its LAST page, with
/// `IsTruncated=false`, must not be paged again: the listing stops on `IsTruncated=false`.
/// (Without the stop the same token is sent back and the page repeats for ever.)
#[tokio::test]
async fn a_last_page_with_a_token_and_is_truncated_false_ends_the_listing() {
    use common::mock_s3_server::LAST_PAGE_TOKEN_MARKER;
    let (port, state, b) = seeded_marked(LAST_PAGE_TOKEN_MARKER, CATALOG);
    for uri in [
        format!("s3://{b}/dl/"),
        format!("s3://{b}/dl"),
        format!("s3://{b}/data/**.npz"),
    ] {
        let listed = tokio::time::timeout(
            std::time::Duration::from_secs(20),
            list_all_ways(port, &uri, true),
        )
        .await
        .unwrap_or_else(|_| panic!("list({uri}) did not terminate on IsTruncated=false"));
        for (which, got) in listed {
            let mut sorted = got.clone();
            sorted.dedup();
            assert_eq!(got, sorted, "{which}({uri}) repeated a page: {got:?}");
        }
    }
    assert!(
        list_requests(&state, &b)
            .iter()
            .all(|r| !r.query.contains("continuation-token")),
        "a page with IsTruncated=false must not be followed"
    );
    // the delete path (strict) stops too, and deletes exactly the directory
    tokio::time::timeout(
        std::time::Duration::from_secs(20),
        delete_recursively(port, &format!("s3://{b}/dl/")),
    )
    .await
    .expect("delete -r terminated");
    assert_eq!(
        state.remaining_keys(&b),
        without(&["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );
}

/// The strict error stays: a TRUNCATED page without a token is an incomplete listing, and
/// a delete must not run on it (exact-name children listing and directory listing alike).
#[tokio::test]
async fn a_truncated_page_without_a_token_fails_a_delete_listing() {
    use common::mock_s3_server::TRUNCATED_NO_TOKEN_MARKER;
    let many: Vec<String> = (0..1500).map(|i| format!("big/k{i:05}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    for key in ["big/", "big"] {
        let (port, state, b) = seeded_marked(TRUNCATED_NO_TOKEN_MARKER, &refs);
        let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
            .await
            .expect("for_endpoint");
        let uri = format!("s3://{b}/{key}");
        let report =
            delete_objects_under_literal_prefix(&store, &uri, None, DeleteMode::Batched, None)
                .await;
        let err = report
            .into_result(&uri, None)
            .expect_err("an incomplete listing must fail the delete");
        assert!(format!("{err:#}").contains("truncated"), "{key}: {err:#}");
        assert!(
            state.remaining_keys(&b).len() >= many.len() - 1000,
            "{key}: at most the first page may have been deleted"
        );
    }
}

/// A host-form URI (`s3://host:port/bucket/key`) with a glob selects exactly what the
/// bucket form selects, and the entries come back in bucket form.
#[tokio::test]
async fn a_host_form_uri_with_a_glob_selects_like_the_bucket_form() {
    use s3dlio::list_pattern::ListPattern;
    let (port, _s, b) = seeded();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint");
    let host = format!("s3://127.0.0.1:{port}/{b}");
    for recursive in [true, false] {
        assert_eq!(
            store
                .list(&format!("{host}/dl/*.npz"), recursive)
                .await
                .unwrap(),
            uris(&b, &["dl/a.npz"]),
            "recursive={recursive}"
        );
        assert_eq!(
            store
                .list(&format!("{host}/file.txt*"), recursive)
                .await
                .unwrap(),
            uris(&b, &["file.txt", "file.txt.bak", "file.txt2"]),
            "recursive={recursive}"
        );
        assert_eq!(
            store
                .list(&format!("{host}/data/**.npz"), recursive)
                .await
                .unwrap(),
            uris(&b, &["data/sub/deep/z.npz", "data/top.npz"]),
            "recursive={recursive}"
        );
    }
    // the explicit pattern is relative to the glob's directory, host in the URI or not
    let uri = format!("{host}/data/**.npz");
    let mut keys = store.list(&uri, true).await.unwrap();
    ListPattern::new("^sub/").unwrap().retain(&uri, &mut keys);
    assert_eq!(keys, uris(&b, &["data/sub/deep/z.npz"]));
    // and the host-form delete takes the glob matches only
    let (port, state, b) = seeded();
    delete_recursively(port, &format!("s3://127.0.0.1:{port}/{b}/dl/*.npz")).await;
    assert_eq!(state.remaining_keys(&b), without(&["dl/a.npz"]));
}

/// A key that starts with '/' (`s3://b//x`): the key part is `/x...`, and a glob in it
/// works the same way (`*` does not cross the following '/').
#[tokio::test]
async fn a_leading_slash_key_with_a_glob_selects_by_the_rule() {
    let keys = ["/x", "/x1", "/x2/y", "x3", "/other"];
    let (port, state, b) = seeded_marked("lead", &keys);
    assert_selects(
        port,
        &format!("s3://{b}//x*"),
        true,
        uris(&b, &["/x", "/x1"]),
    )
    .await;
    assert_selects(
        port,
        &format!("s3://{b}//x**"),
        true,
        uris(&b, &["/x", "/x1", "/x2/y"]),
    )
    .await;
    // an exact leading-slash name selects only itself
    assert_selects(port, &format!("s3://{b}//x"), true, uris(&b, &["/x"])).await;
    delete_recursively(port, &format!("s3://{b}//x*")).await;
    let mut want: Vec<String> = vec!["/other".into(), "/x2/y".into(), "x3".into()];
    want.sort();
    assert_eq!(state.remaining_keys(&b), want);
}

// ----------------------------------------------------------------------------
// Dataset constructors: a selection of nothing is an error, never an empty dataset
// ----------------------------------------------------------------------------

fn assert_empty_selection_message(msg: &str, uri: &str) {
    assert!(msg.contains(uri), "the message must name the URI: {msg}");
    assert!(
        msg.contains("exact object name") && msg.contains("directory"),
        "the message must say what a URI selects: {msg}"
    );
    assert!(
        msg.contains("'*' selects by prefix")
            && msg.contains(&format!("{}*", uri.trim_end_matches('/'))),
        "the message must show the `*` form: {msg}"
    );
}

#[test]
fn s3_bytes_dataset_refuses_a_bare_partial_name() {
    use s3dlio::data_loader::s3_bytes::S3BytesDataset;
    let (_p, _s, b) = seeded();
    for key in ["file.tx", "ru", "nothing"] {
        let uri = format!("s3://{b}/{key}");
        let err = match S3BytesDataset::from_prefix(&uri) {
            Ok(ds) => panic!("{uri} selects nothing but gave {} keys", ds.keys().len()),
            Err(e) => e.to_string(),
        };
        assert_empty_selection_message(&err, &uri);
    }
    // the same names with a `*` are a prefix selection and work
    let ds = S3BytesDataset::from_prefix(&format!("s3://{b}/file.tx*")).expect("glob");
    assert_eq!(ds.keys().len(), 3); // file.txt, file.txt.bak, file.txt2
}

#[test]
fn multi_backend_dataset_refuses_a_bare_partial_name() {
    use s3dlio::data_loader::async_pool_dataloader::MultiBackendDataset;
    let (_p, _s, b) = seeded();
    for key in ["file.tx", "dl2/", "nothing/"] {
        let uri = format!("s3://{b}/{key}");
        let res = on_fresh_runtime(MultiBackendDataset::from_prefix(&uri));
        match (key, res) {
            ("dl2/", Ok(ds)) => assert_eq!(ds.len(), 1, "a real directory still works"),
            ("dl2/", Err(e)) => panic!("{uri}: {e:#}"),
            (_, Ok(ds)) => panic!("{uri} selects nothing but gave {} uris", ds.len()),
            (_, Err(e)) => assert_empty_selection_message(&format!("{e:#}"), &uri),
        }
    }
    let ds = on_fresh_runtime(MultiBackendDataset::from_prefix(&format!(
        "s3://{b}/file.tx*"
    )))
    .expect("glob");
    assert_eq!(ds.len(), 3);
}
