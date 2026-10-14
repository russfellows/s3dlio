// tests/s3_cli_download_pattern_213.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Issue #213, reviewer ruling D3/D4, through the real `s3-cli` binary.
//
// `s3-cli download` and `s3-cli get` treat an `s3://` URI as a LITERAL key prefix
// (`s3://b/file.txt` also selects `file.txt.bak`). The regex they used to apply
// implicitly is now available only through the explicit `-p/--pattern` option, which
// reuses `s3dlio::list_pattern::ListPattern` inside the existing backend-aware paths
// (matched against the entry relative to the listing base).
//
// The S3 server is the shared in-process mock with a seeded catalog that honours
// prefix and delimiter like S3 (tests/common/mock_s3_server.rs).

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, MockS3State};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

const CATALOG: &[&str] = &[
    "file.txt",
    "file.txt.bak",
    "other.txt",
    "dl/a.npz",
    "dl/b.txt",
    "dl/sub/c.npz",
    "dl2/d.npz",
    "dl2/e.txt",
];

static N: AtomicU64 = AtomicU64::new(0);

fn seeded() -> (u16, MockS3State, String) {
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("dlpat213-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, CATALOG);
    (port, state, bucket)
}

/// A fresh, not-yet-existing directory path under the cargo test tmp dir.
fn fresh_dir(tag: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "dlpat213-{}-{}-{tag}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn s3_cli(port: u16, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .args(args)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"))
        .env("AWS_ACCESS_KEY_ID", "test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        .env("AWS_REGION", "us-east-1")
        .output()
        .expect("run s3-cli")
}

fn text(out: &Output) -> String {
    format!(
        "status={:?}\nstdout={}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Names of the files directly inside `dir` (download writes a flat directory).
fn files_in(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}"))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

/// Keys of the object GETs (everything but ListObjectsV2) the mock saw for `bucket`.
fn object_gets(state: &MockS3State, bucket: &str) -> BTreeSet<String> {
    let prefix = format!("/{bucket}/");
    state
        .requests_containing(bucket)
        .into_iter()
        .filter(|r| r.method == "GET" && !r.query.contains("list-type"))
        .filter_map(|r| r.path.strip_prefix(&prefix).map(str::to_string))
        .collect()
}

// ---------------------------------------------------------------------------
// download -p
// ---------------------------------------------------------------------------

#[test]
fn download_pattern_selects_only_matching_objects() {
    let (port, _s, b) = seeded();
    let dest = fresh_dir("sel");
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            "-p",
            r"\.npz$",
            &format!("s3://{b}/dl/"),
            dest.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", text(&out));
    // dl/b.txt is a decoy under the prefix; dl2/ is outside the `dl/` directory.
    assert_eq!(files_in(&dest), set(&["a.npz", "c.npz"]), "{}", text(&out));
    assert_eq!(std::fs::read(dest.join("a.npz")).unwrap(), b"mock");
    let _ = std::fs::remove_dir_all(&dest);
}

#[test]
fn download_pattern_is_matched_relative_to_the_listing_base() {
    let (port, _s, b) = seeded();
    let dest = fresh_dir("rel");
    // `^sub/` only matches because the pattern sees `sub/c.npz`, not the whole URI.
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            "--pattern",
            "^sub/",
            &format!("s3://{b}/dl/"),
            dest.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(files_in(&dest), set(&["c.npz"]), "{}", text(&out));
    let _ = std::fs::remove_dir_all(&dest);
}

#[test]
fn download_pattern_composes_with_a_literal_prefix_without_slash() {
    let (port, _s, b) = seeded();
    let dest = fresh_dir("comp");
    // `s3://b/dl` is a prefix: it covers dl/ AND dl2/. The pattern then narrows it.
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            "-p",
            r"d\.npz$",
            &format!("s3://{b}/dl"),
            dest.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(files_in(&dest), set(&["d.npz"]), "{}", text(&out));
    let _ = std::fs::remove_dir_all(&dest);
}

#[test]
fn download_invalid_pattern_sends_nothing_and_creates_nothing() {
    let (port, state, b) = seeded();
    let dest = fresh_dir("bad");
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            "-p",
            "(",
            &format!("s3://{b}/dl/"),
            dest.to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Invalid regex pattern"),
        "{}",
        text(&out)
    );
    assert!(
        state.requests_containing(&b).is_empty(),
        "an invalid --pattern must send no request: {:?}",
        state.requests_containing(&b)
    );
    assert!(!dest.exists(), "no output directory may be created");
}

#[test]
fn download_pattern_matching_nothing_fails_without_creating_the_directory() {
    let (port, _s, b) = seeded();
    let dest = fresh_dir("none");
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            "-p",
            r"\.nomatch$",
            &format!("s3://{b}/dl/"),
            dest.to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("No objects matched"),
        "{}",
        text(&out)
    );
    assert!(!dest.exists(), "no output directory may be created");
}

#[test]
fn download_without_pattern_is_a_literal_prefix() {
    let (port, _s, b) = seeded();
    let dest = fresh_dir("prefix");
    // Not exact-match: `file.txt` selects `file.txt.bak` too, but never `other.txt`.
    let out = s3_cli(
        port,
        &[
            "download",
            &format!("s3://{b}/file.txt"),
            dest.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        files_in(&dest),
        set(&["file.txt", "file.txt.bak"]),
        "{}",
        text(&out)
    );
    let _ = std::fs::remove_dir_all(&dest);
}

#[test]
fn download_without_pattern_directory_prefix_does_not_cross_into_dl2() {
    let (port, _s, b) = seeded();
    let with_slash = fresh_dir("slash");
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            &format!("s3://{b}/dl/"),
            with_slash.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        files_in(&with_slash),
        set(&["a.npz", "b.txt", "c.npz"]),
        "{}",
        text(&out)
    );
    let _ = std::fs::remove_dir_all(&with_slash);

    let bare = fresh_dir("bare");
    let out = s3_cli(
        port,
        &[
            "download",
            "-r",
            &format!("s3://{b}/dl"),
            bare.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        files_in(&bare),
        set(&["a.npz", "b.txt", "c.npz", "d.npz", "e.txt"]),
        "s3://b/dl is a prefix and also selects dl2/...: {}",
        text(&out)
    );
    let _ = std::fs::remove_dir_all(&bare);
}

// ---------------------------------------------------------------------------
// get -p
// ---------------------------------------------------------------------------

#[test]
fn get_pattern_selects_only_matching_objects() {
    let (port, state, b) = seeded();
    let out = s3_cli(port, &["get", "-p", r"\.npz$", &format!("s3://{b}/dl/")]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        object_gets(&state, &b),
        set(&["dl/a.npz", "dl/sub/c.npz"]),
        "{}",
        text(&out)
    );
}

#[test]
fn get_pattern_is_matched_relative_to_the_listing_base() {
    let (port, state, b) = seeded();
    let out = s3_cli(
        port,
        &["get", "--pattern", "^sub/", &format!("s3://{b}/dl/")],
    );
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        object_gets(&state, &b),
        set(&["dl/sub/c.npz"]),
        "{}",
        text(&out)
    );
}

#[test]
fn get_invalid_pattern_sends_nothing() {
    let (port, state, b) = seeded();
    let out = s3_cli(port, &["get", "-p", "(", &format!("s3://{b}/dl/")]);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Invalid regex pattern"),
        "{}",
        text(&out)
    );
    assert!(
        state.requests_containing(&b).is_empty(),
        "an invalid --pattern must send no request: {:?}",
        state.requests_containing(&b)
    );
}

#[test]
fn get_pattern_with_a_keylist_is_refused_before_any_request() {
    let (port, state, b) = seeded();
    let keylist = fresh_dir("keylist").with_extension("txt");
    std::fs::write(&keylist, format!("s3://{b}/dl/a.npz\n")).unwrap();
    let out = s3_cli(
        port,
        &[
            "get",
            "-p",
            r"\.npz$",
            "--keylist",
            keylist.to_str().unwrap(),
        ],
    );
    let _ = std::fs::remove_file(&keylist);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--keylist"),
        "{}",
        text(&out)
    );
    assert!(state.requests_containing(&b).is_empty());
}

#[test]
fn get_without_pattern_is_a_literal_prefix() {
    let (port, state, b) = seeded();
    let out = s3_cli(port, &["get", &format!("s3://{b}/file.txt")]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(
        object_gets(&state, &b),
        set(&["file.txt", "file.txt.bak"]),
        "{}",
        text(&out)
    );
}

// ---------------------------------------------------------------------------
// backend-aware: file://
// ---------------------------------------------------------------------------

#[test]
fn download_pattern_works_on_the_file_backend() {
    let src = fresh_dir("fsrc");
    std::fs::create_dir_all(src.join("sub")).unwrap();
    for name in ["a.npz", "b.txt", "sub/c.npz"] {
        std::fs::write(src.join(name), name.as_bytes()).unwrap();
    }
    let dest = fresh_dir("fdest");
    // No S3 endpoint is needed: the file:// backend is listed and read directly.
    let out = Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .args([
            "download",
            "-r",
            "-p",
            r"\.npz$",
            &format!("file://{}/", src.display()),
            dest.to_str().unwrap(),
        ])
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .output()
        .expect("run s3-cli");
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(files_in(&dest), set(&["a.npz", "c.npz"]), "{}", text(&out));
    assert_eq!(std::fs::read(dest.join("c.npz")).unwrap(), b"sub/c.npz");
    let _ = std::fs::remove_dir_all(&src);
    let _ = std::fs::remove_dir_all(&dest);

    // An invalid pattern on file:// is also refused before the directory is created.
    let dest = fresh_dir("fbad");
    let out = Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .args(["download", "-r", "-p", "(", "file:///nonexistent-213/"])
        .arg(&dest)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .output()
        .expect("run s3-cli");
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Invalid regex pattern"),
        "{}",
        text(&out)
    );
    assert!(!dest.exists());
}
