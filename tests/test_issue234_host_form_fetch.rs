// tests/test_issue234_host_form_fetch.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #234: `s3-cli download`, `s3-cli get` and the library's
// backend-aware download (`generic_download_objects[_matching]_with_summary`) with a
// HOST-FORM URI (`s3://host:port/bucket/prefix/`) listed on the named server but then
// fetched every listed object through a store built from the bucket-form URI the listing
// returned (`s3://bucket/key`), that is, through the process-wide client: the server
// `AWS_ENDPOINT_URL` names, or AWS.
//
//   A = the process-wide client's server (`AWS_ENDPOINT_URL`); it holds the SAME bucket and
//       keys as B with DIFFERENT bytes
//   B = the server the host-form URI names
//
// Every object must be fetched from B: B's bytes come back and A sees no request at all.
// Each test uses a bucket name of its own, so the requests A recorded can be told apart.

#[path = "common/catalog_mock.rs"]
mod catalog_mock;

use catalog_mock::CatalogMock;
use s3dlio::list_pattern::ListPattern;
use s3dlio::object_store::{
    generic_download_objects_matching_with_summary, generic_download_objects_with_summary,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};

static GLOBAL_A: OnceLock<CatalogMock> = OnceLock::new();
static ENV: Once = Once::new();
static N: AtomicU64 = AtomicU64::new(0);

/// Server A, started once; the process environment points the global client at it.
fn server_a() -> &'static CatalogMock {
    let a = GLOBAL_A.get_or_init(|| CatalogMock::start(&[]));
    ENV.call_once(|| {
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ENDPOINT_URL", a.url());
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "1");
    });
    a
}

const KEYS: &[&str] = &["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"];

/// A fresh bucket name, seeded on A with `A-bytes:<key>` and returned with a B holding
/// `B-bytes:<key>` under the same bucket and keys.
fn two_servers(tag: &str) -> (&'static CatalogMock, CatalogMock, String) {
    let a = server_a();
    let bucket = format!("t234-{tag}-{}", N.fetch_add(1, Ordering::SeqCst));
    let b = CatalogMock::start(&[]);
    for key in KEYS {
        a.seed(&bucket, key, format!("A-bytes:{key}").as_bytes());
        b.seed(&bucket, key, format!("B-bytes:{key}").as_bytes());
    }
    (a, b, bucket)
}

fn fresh_dir(tag: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "t234-{}-{}-{tag}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn files_in(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

/// Object GETs (no listing) the mock saw for `bucket`.
fn object_gets(m: &CatalogMock, bucket: &str) -> BTreeSet<String> {
    let prefix = format!("/{bucket}/");
    m.requests_for(bucket)
        .into_iter()
        .filter(|r| r.method == "GET" && !r.query.contains("list-type"))
        .filter_map(|r| r.path.strip_prefix(&prefix).map(str::to_string))
        .collect()
}

fn text(out: &Output) -> String {
    format!(
        "status={:?}\nstdout={}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn s3_cli(a: &CatalogMock, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .args(args)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_ENDPOINT_URL", a.url())
        .env("AWS_ACCESS_KEY_ID", "test-key")
        .env("AWS_SECRET_ACCESS_KEY", "test-secret")
        .env("AWS_REGION", "us-east-1")
        .env("S3DLIO_MAX_RETRY_ATTEMPTS", "1")
        .output()
        .expect("run s3-cli")
}

fn assert_b_bytes(dest: &Path, expected: &[(&str, &str)]) {
    for (file, key) in expected {
        assert_eq!(
            std::fs::read(dest.join(file)).unwrap_or_else(|e| panic!("read {file}: {e}")),
            format!("B-bytes:{key}").into_bytes(),
            "{file} must hold the bytes of the server the URI names"
        );
    }
}

// ---------------------------------------------------------------------------
// The library function
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn library_download_fetches_from_the_server_that_listed() {
    let (a, b, bucket) = two_servers("lib");
    let dest = fresh_dir("lib");
    let uri = format!("s3://{}/{bucket}/dl/", b.authority());

    let summary = generic_download_objects_with_summary(&uri, &dest, 4, true, None)
        .await
        .expect("download");

    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (3, 3, 0)
    );
    assert_eq!(
        files_in(&dest),
        set(&["a.npz", "b.txt", "c.npz"]),
        "flat destination"
    );
    assert_b_bytes(
        &dest,
        &[
            ("a.npz", "dl/a.npz"),
            ("b.txt", "dl/b.txt"),
            ("c.npz", "dl/sub/c.npz"),
        ],
    );
    assert!(
        a.requests_for(&bucket).is_empty(),
        "the process-wide server must see no request: {:?}",
        a.requests_for(&bucket)
    );
    assert_eq!(
        object_gets(&b, &bucket),
        set(&["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );
    let _ = std::fs::remove_dir_all(&dest);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn library_download_matching_fetches_from_the_server_that_listed() {
    let (a, b, bucket) = two_servers("libpat");
    let dest = fresh_dir("libpat");
    let uri = format!("s3://{}/{bucket}/dl/", b.authority());
    let pattern = ListPattern::new(r"\.npz$").expect("regex");

    let summary =
        generic_download_objects_matching_with_summary(&uri, &dest, 4, true, Some(&pattern), None)
            .await
            .expect("download");

    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (2, 2, 0)
    );
    assert_eq!(files_in(&dest), set(&["a.npz", "c.npz"]));
    assert_b_bytes(&dest, &[("a.npz", "dl/a.npz"), ("c.npz", "dl/sub/c.npz")]);
    assert!(
        a.requests_for(&bucket).is_empty(),
        "the process-wide server must see no request: {:?}",
        a.requests_for(&bucket)
    );
    let _ = std::fs::remove_dir_all(&dest);
}

// ---------------------------------------------------------------------------
// The s3-cli binary
// ---------------------------------------------------------------------------

#[test]
fn cli_download_fetches_from_the_server_that_listed() {
    let (a, b, bucket) = two_servers("clidl");
    let dest = fresh_dir("clidl");
    let uri = format!("s3://{}/{bucket}/dl/", b.authority());

    let out = s3_cli(a, &["download", "-r", &uri, dest.to_str().unwrap()]);

    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        files_in(&dest),
        set(&["a.npz", "b.txt", "c.npz"]),
        "{}",
        text(&out)
    );
    assert_b_bytes(
        &dest,
        &[
            ("a.npz", "dl/a.npz"),
            ("b.txt", "dl/b.txt"),
            ("c.npz", "dl/sub/c.npz"),
        ],
    );
    assert!(
        a.requests_for(&bucket).is_empty(),
        "the process-wide server must see no request: {:?}\n{}",
        a.requests_for(&bucket),
        text(&out)
    );
    let _ = std::fs::remove_dir_all(&dest);
}

#[test]
fn cli_download_with_pattern_fetches_from_the_server_that_listed() {
    let (a, b, bucket) = two_servers("clipat");
    let dest = fresh_dir("clipat");
    let uri = format!("s3://{}/{bucket}/dl/", b.authority());

    let out = s3_cli(
        a,
        &[
            "download",
            "-r",
            "-p",
            r"\.npz$",
            &uri,
            dest.to_str().unwrap(),
        ],
    );

    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(files_in(&dest), set(&["a.npz", "c.npz"]), "{}", text(&out));
    assert_b_bytes(&dest, &[("a.npz", "dl/a.npz"), ("c.npz", "dl/sub/c.npz")]);
    assert!(
        a.requests_for(&bucket).is_empty(),
        "{:?}",
        a.requests_for(&bucket)
    );
    let _ = std::fs::remove_dir_all(&dest);
}

#[test]
fn cli_get_fetches_from_the_server_that_listed() {
    let (a, b, bucket) = two_servers("cliget");
    let uri = format!("s3://{}/{bucket}/dl/", b.authority());

    let out = s3_cli(a, &["get", "-r", &uri]);

    assert!(out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stdout)
            .contains("GET summary: attempted=3, succeeded=3, failed=0"),
        "{}",
        text(&out)
    );
    assert!(
        a.requests_for(&bucket).is_empty(),
        "the process-wide server must see no request: {:?}\n{}",
        a.requests_for(&bucket),
        text(&out)
    );
    assert_eq!(
        object_gets(&b, &bucket),
        set(&["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"]),
        "{}",
        text(&out)
    );
}
