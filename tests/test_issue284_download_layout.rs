// tests/test_issue284_download_layout.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Issue #284: a recursive download keeps each object's key path relative to the listing
// base (`<dest>/<key relative to the base>`), never writes outside `dest`, and reports
// two objects that map to one local path instead of silently overwriting one.
//
// The sources are `file://` trees in temp directories. A Linux file name may hold a
// backslash, which s3dlio reads as a path separator on every platform, so
// `..\evil` and `a\b` are real, listable "keys" that exercise the traversal and collision
// rules; keys a file system cannot hold (`a//b`, `/abs`, NUL) are unit-tested on the mapping
// function in `src/download_layout.rs`.

use s3dlio::{generic_download_objects, generic_download_objects_with_summary};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn write(path: &Path, bytes: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("mkdir");
    }
    std::fs::write(path, bytes).expect("write");
}

/// Every regular file below `root`, as `/`-joined paths relative to `root`.
fn tree(root: &Path) -> BTreeSet<String> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).expect("read_dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if entry.file_type().expect("type").is_dir() {
                walk(&path, root, out);
            } else {
                let rel = path.strip_prefix(root).expect("under root");
                let parts: Vec<_> = rel.iter().map(|c| c.to_string_lossy()).collect();
                out.insert(parts.join("/"));
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(root, root, &mut out);
    out
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

fn uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

/// `<tmp>/src` (source tree), `<tmp>/work/dest` (destination); `work` holds nothing else, so
/// anything written outside `dest` shows up as an extra entry of `work` or of `tmp`.
struct Dirs {
    _tmp: tempfile::TempDir,
    src: PathBuf,
    work: PathBuf,
    dest: PathBuf,
}

fn dirs() -> Dirs {
    let tmp = tempfile::tempdir().expect("tempdir");
    let src = tmp.path().join("src");
    let work = tmp.path().join("work");
    let dest = work.join("dest");
    std::fs::create_dir_all(&src).expect("mkdir src");
    std::fs::create_dir_all(&work).expect("mkdir work");
    Dirs {
        _tmp: tmp,
        src,
        work,
        dest,
    }
}

fn entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn nested_keys_keep_their_path_and_same_names_both_land() {
    let d = dirs();
    write(&d.src.join("a/x.bin"), b"from-a");
    write(&d.src.join("b/x.bin"), b"from-b");
    write(&d.src.join("top.bin"), b"top");
    write(&d.src.join("deep/1/2/3/4/y.bin"), b"deep");

    let summary =
        generic_download_objects_with_summary(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
            .await
            .expect("download");

    assert_eq!(
        tree(&d.dest),
        set(&["a/x.bin", "b/x.bin", "top.bin", "deep/1/2/3/4/y.bin"])
    );
    assert_eq!(std::fs::read(d.dest.join("a/x.bin")).unwrap(), b"from-a");
    assert_eq!(std::fs::read(d.dest.join("b/x.bin")).unwrap(), b"from-b");
    assert_eq!(
        std::fs::read(d.dest.join("deep/1/2/3/4/y.bin")).unwrap(),
        b"deep"
    );
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (4, 4, 0)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_directory_uri_without_a_slash_selects_the_same_layout() {
    let d = dirs();
    write(&d.src.join("a/x.bin"), b"1");
    write(&d.src.join("b/x.bin"), b"2");

    let summary = generic_download_objects_with_summary(&uri(&d.src), &d.dest, 4, true, None)
        .await
        .expect("download");

    assert_eq!(tree(&d.dest), set(&["a/x.bin", "b/x.bin"]));
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (2, 2, 0)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_single_exact_object_goes_to_dest_slash_its_name() {
    let d = dirs();
    write(&d.src.join("sub/only.bin"), b"only");
    write(&d.src.join("sub/other.bin"), b"other");

    let summary = generic_download_objects_with_summary(
        &uri(&d.src.join("sub/only.bin")),
        &d.dest,
        4,
        false,
        None,
    )
    .await
    .expect("download");

    assert_eq!(tree(&d.dest), set(&["only.bin"]));
    assert_eq!(std::fs::read(d.dest.join("only.bin")).unwrap(), b"only");
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (1, 1, 0)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn traversal_keys_are_refused_and_nothing_is_written_outside_dest() {
    let d = dirs();
    write(&d.src.join("ok.bin"), b"ok");
    write(&d.src.join("sub/fine.bin"), b"fine");
    // Backslash-separated traversal: legal file names on Linux, path separators to s3dlio.
    write(&d.src.join("..\\evil"), b"evil1");
    write(&d.src.join("sub/..\\..\\evil2"), b"evil2");
    write(&d.src.join("a\\..\\..\\evil3"), b"evil3");

    let before = entries(d.work.parent().expect("tmp"));
    let summary =
        generic_download_objects_with_summary(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
            .await
            .expect("the summary reports per-object failures");

    assert_eq!(tree(&d.dest), set(&["ok.bin", "sub/fine.bin"]));
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (5, 2, 3)
    );
    assert_eq!(entries(&d.work), set(&["dest"]), "nothing next to dest");
    assert_eq!(
        entries(d.work.parent().expect("tmp")),
        before,
        "nothing above work"
    );
    for name in ["evil", "evil1", "evil2", "evil3"] {
        assert!(!d.work.join(name).exists(), "{name} escaped dest");
    }

    // The all-or-error wrapper reports the failures too.
    let err = generic_download_objects(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
        .await
        .expect_err("failures are an error");
    assert!(format!("{err:#}").contains("failed=3"), "{err:#}");
}

#[tokio::test(flavor = "multi_thread")]
async fn two_objects_for_one_local_path_are_reported_not_overwritten() {
    let d = dirs();
    // `a\b` is the local path a/b, the same as the real file a/b.
    write(&d.src.join("a/b"), b"slash");
    write(&d.src.join("a\\b"), b"backslash");
    write(&d.src.join("c.bin"), b"c");

    let summary =
        generic_download_objects_with_summary(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
            .await
            .expect("download");

    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (3, 2, 1)
    );
    assert_eq!(tree(&d.dest), set(&["a/b", "c.bin"]));
    let got = std::fs::read(d.dest.join("a/b")).unwrap();
    assert!(got == b"slash" || got == b"backslash", "{got:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_and_a_directory_for_one_local_path_are_reported() {
    let d = dirs();
    write(&d.src.join("a"), b"file-a");
    write(&d.src.join("a\\b"), b"would-be-a/b");

    let summary =
        generic_download_objects_with_summary(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
            .await
            .expect("download");

    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (2, 1, 1)
    );
    let files = tree(&d.dest);
    assert!(
        files == set(&["a"]) || files == set(&["a/b"]),
        "exactly one of the two lands: {files:?}"
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn a_symlink_in_dest_is_not_followed_out_of_dest() {
    let d = dirs();
    write(&d.src.join("sub/x.bin"), b"payload");
    write(&d.src.join("ok.bin"), b"ok");
    let outside = d.work.join("outside");
    std::fs::create_dir_all(&outside).expect("mkdir");
    std::fs::create_dir_all(&d.dest).expect("mkdir");
    std::os::unix::fs::symlink(&outside, d.dest.join("sub")).expect("symlink");

    let summary =
        generic_download_objects_with_summary(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
            .await
            .expect("download");

    assert!(!outside.join("x.bin").exists(), "the symlink was followed");
    assert!(entries(&outside).is_empty());
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (2, 1, 1)
    );
    assert_eq!(std::fs::read(d.dest.join("ok.bin")).unwrap(), b"ok");
}

// ---------------------------------------------------------------------------
// The real binary
// ---------------------------------------------------------------------------

fn s3_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .args(args)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
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

#[test]
fn cli_download_recursive_keeps_nested_layout() {
    let d = dirs();
    write(&d.src.join("a/x.bin"), b"A");
    write(&d.src.join("b/x.bin"), b"B");
    write(&d.src.join("c/d/z.bin"), b"Z");

    let out = s3_cli(&[
        "download",
        "-r",
        &format!("{}/", uri(&d.src)),
        d.dest.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(tree(&d.dest), set(&["a/x.bin", "b/x.bin", "c/d/z.bin"]));
    assert_eq!(std::fs::read(d.dest.join("b/x.bin")).unwrap(), b"B");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("attempted=3, succeeded=3, failed=0"),
        "{}",
        text(&out)
    );
}

#[test]
fn cli_download_refuses_traversal_and_exits_nonzero() {
    let d = dirs();
    write(&d.src.join("ok.bin"), b"ok");
    write(&d.src.join("..\\evil"), b"evil");

    let out = s3_cli(&[
        "download",
        "-r",
        &format!("{}/", uri(&d.src)),
        d.dest.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("attempted=2, succeeded=1, failed=1"),
        "{}",
        text(&out)
    );
    assert_eq!(tree(&d.dest), set(&["ok.bin"]));
    assert!(!d.work.join("evil").exists());
}

#[test]
fn cli_cp_download_recursive_uses_the_same_layout() {
    // `cp` shares the helper, so a recursive cp download keeps nested keys too.
    let d = dirs();
    write(&d.src.join("a/x.bin"), b"A");
    write(&d.src.join("b/x.bin"), b"B");

    let out = s3_cli(&[
        "cp",
        "-r",
        &format!("{}/", uri(&d.src)),
        d.dest.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(tree(&d.dest), set(&["a/x.bin", "b/x.bin"]));
}

// ---------------------------------------------------------------------------
// Hostile S3 keys (an S3 key may be `../x` or `a//b`; a file system cannot list those)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
mod common;

use common::mock_s3_server::ensure_mock_s3_server;
use std::sync::atomic::{AtomicU64, Ordering};

static N: AtomicU64 = AtomicU64::new(0);

fn s3_cli_on(port: u16, args: &[&str]) -> Output {
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

#[test]
fn cli_download_of_hostile_s3_keys_refuses_them_before_fetching() {
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(
        &bucket,
        &[
            "dl/ok.bin",
            "dl/sub/y.bin",
            "dl/a/x",
            "dl/b/x",
            "dl/../evil",
            "dl/a/../../evil2",
            "dl/a//b",
            "dl/./dot",
        ],
    );
    let d = dirs();

    let out = s3_cli_on(
        port,
        &[
            "download",
            "-r",
            &format!("s3://{bucket}/dl/"),
            d.dest.to_str().unwrap(),
        ],
    );

    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("attempted=8, succeeded=4, failed=4"),
        "{}",
        text(&out)
    );
    assert_eq!(tree(&d.dest), set(&["ok.bin", "sub/y.bin", "a/x", "b/x"]));
    assert_eq!(entries(&d.work), set(&["dest"]), "nothing next to dest");
    let fetched: BTreeSet<String> = state
        .requests_for_bucket(&bucket)
        .into_iter()
        .filter(|r| r.method == "GET" && !r.query.contains("list-type"))
        .map(|r| r.path)
        .collect();
    for path in &fetched {
        assert!(
            !path.contains("evil") && !path.contains("//b") && !path.contains("/./"),
            "a refused key was fetched: {path}"
        );
    }
}

// ---------------------------------------------------------------------------
// Review round: one base per listing, colons in keys, error surface, s3_copy fail-fast
// ---------------------------------------------------------------------------

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

#[test]
fn an_exact_object_that_is_also_a_directory_gives_the_directory_the_path() {
    // `s3://b/dl` selects the object `dl` (a zero-byte folder marker) AND the directory
    // `dl/`. One base (the parent) for the whole listing, and the directory's contents win:
    // `dl/a` -> `dest/dl/a`, `dl/b/c` -> `dest/dl/b/c`; only the single object `dl` is
    // refused (one failed object), never the whole directory and never `dest/a`.
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl", "dl/a", "dl/b/c"]);
    let d = dirs();

    let summary = on_fresh_runtime(generic_download_objects_with_summary(
        &format!("s3://{bucket}/dl"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(tree(&d.dest), set(&["dl/a", "dl/b/c"]));
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (3, 2, 1)
    );

    // The CLI reports the one failure, names the alternatives, and exits non-zero.
    let d = dirs();
    let out = s3_cli_on(
        port,
        &[
            "download",
            "-r",
            &format!("s3://{bucket}/dl"),
            d.dest.to_str().unwrap(),
        ],
    );
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("attempted=3, succeeded=2, failed=1"),
        "{}",
        text(&out)
    );
    assert_eq!(tree(&d.dest), set(&["dl/a", "dl/b/c"]));

    // The error of the library call names the object, the directory and the way out.
    let d = dirs();
    let err = on_fresh_runtime(generic_download_objects(
        &format!("s3://{bucket}/dl"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect_err("one object failed");
    let msg = format!("{err:#}");
    assert!(msg.contains("failed=1"), "{msg}");
    assert!(msg.contains("object 'dl' and directory 'dl/'"), "{msg}");
    assert!(msg.contains(&format!("s3://{bucket}/dl/")), "{msg}");
    assert_eq!(tree(&d.dest), set(&["dl/a", "dl/b/c"]));

    // The directory alone, and the object alone, both work as the message advises.
    let d = dirs();
    on_fresh_runtime(generic_download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect("the directory alone");
    assert_eq!(tree(&d.dest), set(&["a", "b/c"]));

    // Without a marker object the directory is the base, as before.
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["run.1/child", "run.1/z/w"]);
    let d = dirs();
    on_fresh_runtime(generic_download_objects_with_summary(
        &format!("s3://{bucket}/run.1"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(tree(&d.dest), set(&["child", "z/w"]));
}

#[test]
fn s3_copy_refuses_an_object_and_directory_of_one_name_naming_the_alternative() {
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl", "dl/a", "dl/b/c"]);
    let d = dirs();

    let err = on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{bucket}/dl"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect_err("the fail-fast helper refuses the whole call");
    let msg = format!("{err:#}");
    assert!(msg.contains("Download refused"), "{msg}");
    assert!(msg.contains("object 'dl' and directory 'dl/'"), "{msg}");
    assert!(
        msg.contains(&format!("'s3://{bucket}/dl/'")),
        "names the trailing-slash alternative: {msg}"
    );
    assert!(!d.dest.exists(), "dest_dir must not be created");
    let gets: Vec<_> = state
        .requests_for_bucket(&bucket)
        .into_iter()
        .filter(|r| r.method == "GET" && !r.query.contains("list-type"))
        .collect();
    assert!(gets.is_empty(), "nothing may be fetched: {gets:?}");

    // The trailing-slash spelling downloads the directory.
    on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect("the directory alone");
    assert_eq!(tree(&d.dest), set(&["a", "b/c"]));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hostile_key_cannot_forge_lines_in_the_error() {
    // A key holding a newline and an ESC sequence (both legal in a Linux file name) that is
    // refused (a backslash traversal) must reach the error text escaped.
    let d = dirs();
    write(&d.src.join("ok.bin"), b"ok");
    write(
        &d.src.join("..\\evil\nFORGED: Download completed\u{1b}[2J"),
        b"evil",
    );

    let err = generic_download_objects(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
        .await
        .expect_err("the traversal key fails");
    let msg = format!("{err:#}");
    assert!(msg.contains("failed=1"), "{msg}");
    assert!(
        !msg.contains('\n') && !msg.contains('\u{1b}'),
        "a control character reached the message: {msg:?}"
    );
    assert!(msg.contains("\\n") && msg.contains("\\u{1b}"), "{msg}");
    assert_eq!(tree(&d.dest), set(&["ok.bin"]));
}

#[test]
fn keys_with_a_colon_download_like_any_other_key() {
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["x:y", "v:1/x", "dl/a:b", "dl/v:1/x"]);

    let d = dirs();
    let summary = on_fresh_runtime(generic_download_objects_with_summary(
        &format!("s3://{bucket}/x:y"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(tree(&d.dest), set(&["x:y"]));
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (1, 1, 0)
    );

    let d = dirs();
    let summary = on_fresh_runtime(generic_download_objects_with_summary(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect("download");
    assert_eq!(tree(&d.dest), set(&["a:b", "v:1/x"]));
    assert_eq!(
        (summary.attempted, summary.succeeded, summary.failed),
        (2, 2, 0)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_error_names_the_first_three_failures_not_only_the_log() {
    let d = dirs();
    write(&d.src.join("ok.bin"), b"ok");
    for n in 1..=5 {
        write(&d.src.join(format!("..\\e{n}")), b"evil");
    }

    let err = generic_download_objects(&format!("{}/", uri(&d.src)), &d.dest, 4, true, None)
        .await
        .expect_err("failures are an error");
    let msg = format!("{err:#}");
    assert!(msg.contains("failed=5"), "{msg}");
    assert_eq!(
        msg.matches("..\\e").count(),
        3,
        "exactly the first three failures are listed: {msg}"
    );
    assert!(
        msg.contains("refused"),
        "the reason is in the message: {msg}"
    );
    assert!(msg.contains("2 more"), "the rest is counted: {msg}");
    assert_eq!(tree(&d.dest), set(&["ok.bin"]));
}

#[test]
fn s3_copy_refuses_the_whole_download_when_one_key_is_unsafe() {
    // The library helper returns `Result<()>`, so it is fail-fast: one bad key, nothing
    // fetched, no destination directory created (the CLI and Python paths refuse per object).
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(
        &bucket,
        &["dl/ok.bin", "dl/sub/fine.bin", "dl/a/../../evil"],
    );
    let d = dirs();

    let err = on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect_err("one unsafe key refuses the whole download");
    let msg = format!("{err:#}");
    assert!(msg.contains("Download refused"), "{msg}");
    assert!(msg.contains("evil"), "the refused key is named: {msg}");
    assert!(!d.dest.exists(), "dest_dir must not be created");
    assert!(entries(&d.work).is_empty(), "nothing next to it either");
    let gets: Vec<_> = state
        .requests_for_bucket(&bucket)
        .into_iter()
        .filter(|r| r.method == "GET" && !r.query.contains("list-type"))
        .collect();
    assert!(gets.is_empty(), "nothing may be fetched: {gets:?}");
}

// ---------------------------------------------------------------------------
// Source spellings whose stores return a different canonical entry form (azure fake)
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-azure")]
#[path = "common/azure_fake.rs"]
mod azure_fake;

#[cfg(feature = "backend-azure")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn azure_https_form_sources_map_to_the_canonical_az_entries() {
    let _guard = azure_fake::ENV_LOCK.lock().await;
    azure_fake::clear_azure_env();
    let (state, port) = azure_fake::start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    azure_fake::set_blob(&state, "dl/a/x.bin", b"A".to_vec());
    azure_fake::set_blob(&state, "dl/b/x.bin", b"B".to_vec());
    azure_fake::set_blob(&state, "dl/top.bin", b"T".to_vec());

    for src in [
        "az://acct/ctr/dl/",
        "az://acct/ctr/dl",
        "https://acct.blob.core.windows.net/ctr/dl/",
        "https://acct.blob.core.windows.net/ctr/dl",
        "http://acct.blob.core.windows.net/ctr/dl/",
    ] {
        let d = dirs();
        let summary = generic_download_objects_with_summary(src, &d.dest, 4, true, None)
            .await
            .unwrap_or_else(|e| panic!("{src}: {e:#}"));
        assert_eq!(
            tree(&d.dest),
            set(&["a/x.bin", "b/x.bin", "top.bin"]),
            "{src}"
        );
        assert_eq!(
            (summary.attempted, summary.succeeded, summary.failed),
            (3, 3, 0),
            "{src}"
        );
    }

    // A single exact blob, in the https spelling.
    let d = dirs();
    let summary = generic_download_objects_with_summary(
        "https://acct.blob.core.windows.net/ctr/dl/top.bin",
        &d.dest,
        4,
        false,
        None,
    )
    .await
    .expect("download");
    assert_eq!(tree(&d.dest), set(&["top.bin"]));
    assert_eq!(summary.failed, 0);
    azure_fake::clear_azure_env();
}

// ---------------------------------------------------------------------------
// Final review: control characters of a remote key through the cause chain
// ---------------------------------------------------------------------------

/// A listed key holding a TAB, U+0085, U+2028 and a bidi override whose GET answers 404
/// (the key also carries the mock's `get-404` marker). The store's own error quotes the raw
/// key, so every place that renders a download failure must escape the final text.
const HOSTILE_404_KEY: &str = "dl/get-404\tx\u{85}y\u{2028}z\u{202e}w";

fn assert_no_raw_control(text: &str) {
    assert!(
        !text
            .chars()
            .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}' | '\u{202e}')),
        "a raw control character reached the text: {text:?}"
    );
}

#[test]
fn a_failing_get_of_a_hostile_key_reaches_the_error_escaped() {
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl/ok.bin", HOSTILE_404_KEY]);
    let d = dirs();

    let err = on_fresh_runtime(generic_download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect_err("the 404 key fails");
    let msg = format!("{err:#}");
    assert!(msg.contains("failed=1"), "{msg}");
    assert!(msg.contains("get-404"), "the reason names the key: {msg}");
    assert_no_raw_control(&msg);
    assert!(
        msg.contains("\\t") && msg.contains("\\u{85}") && msg.contains("\\u{2028}"),
        "{msg}"
    );
}

#[test]
fn s3_copy_error_for_a_failing_get_of_a_hostile_key_is_escaped() {
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl/ok.bin", HOSTILE_404_KEY]);
    let d = dirs();

    let err = on_fresh_runtime(s3dlio::s3_copy::download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ))
    .expect_err("the 404 key fails");
    let msg = format!("{err:#}");
    assert!(msg.contains("get-404"), "the error names the key: {msg}");
    assert_no_raw_control(&msg);
    assert!(msg.contains("\\t") && msg.contains("\\u{85}"), "{msg}");
}

// ---------------------------------------------------------------------------
// Review F-F1: credentials inside an underlying (provider) error message
// ---------------------------------------------------------------------------

/// Everything secret in `common::mock_s3_server::LEAKY_ERROR_MESSAGE`.
const LEAKY_SECRETS: [&str; 5] = ["SECRETPW1", "admin:", "SIGONE1", "SIGTWO2", "CREDONE1"];

/// A listed key (safe, but with a tab) whose GET answers `403` with a message quoting two
/// credential-bearing URLs and a bidi override.
const LEAKY_KEY: &str = "dl/get-leaky\tkey";

fn assert_no_leaked_secret(what: &str, text: &str) {
    for secret in LEAKY_SECRETS {
        assert!(!text.contains(secret), "{what} leaked `{secret}`: {text:?}");
    }
}

/// The message must still be useful: the host and the non-secret path survive, the escape of
/// the bidi override is visible, and no raw control / bidi character remains.
fn assert_leaky_text_is_sanitised_but_useful(what: &str, text: &str) {
    assert_no_leaked_secret(what, text);
    assert!(text.contains("get-leaky"), "{what} names the key: {text:?}");
    assert!(
        text.contains("creds.example.invalid"),
        "{what} lost the host: {text:?}"
    );
    assert!(
        text.contains("s3://other-bucket/k"),
        "{what} lost the second URI: {text:?}"
    );
    assert_no_raw_control(text);
    assert!(
        text.contains("\\u{202e}") && text.contains("\\t"),
        "{what} lost the escapes: {text:?}"
    );
}

/// A `tracing` writer that keeps what the WARN / ERROR lines say.
#[derive(Clone, Default)]
struct CapturedLog(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("log buffer poisoned"))?
            .extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
    type Writer = CapturedLog;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl CapturedLog {
    fn text(&self) -> String {
        let bytes = self.0.lock().map(|b| b.clone()).unwrap_or_default();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Run `f` on a fresh runtime of the calling thread with the WARN-and-up output captured
/// (the lines these functions log from their own task are emitted on this thread).
fn capturing_warnings<F, T>(f: F) -> (T, String)
where
    F: std::future::Future<Output = T>,
{
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(log.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::WARN)
        .finish();
    let out = tracing::subscriber::with_default(subscriber, || on_fresh_runtime(f));
    (out, log.text())
}

#[test]
fn a_provider_error_with_credential_urls_is_sanitised_in_the_generic_download() {
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl/ok.bin", LEAKY_KEY]);
    let d = dirs();

    let src = format!("s3://{bucket}/dl/");
    let (result, warned) =
        capturing_warnings(generic_download_objects(&src, &d.dest, 4, true, None));
    let err = result.expect_err("the leaky key fails");
    let msg = format!("{err:#}");
    assert!(msg.contains("failed=1"), "{msg}");
    assert!(
        d.dest.join("ok.bin").exists(),
        "the healthy object still downloaded"
    );
    assert_leaky_text_is_sanitised_but_useful("the returned error", &msg);
    assert!(
        warned.contains("get-leaky"),
        "no WARN was captured: {warned:?}"
    );
    assert_no_leaked_secret("the WARN log", &warned);
}

#[test]
fn a_provider_error_with_credential_urls_is_sanitised_in_s3_copy() {
    let (_port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl/ok.bin", LEAKY_KEY]);
    let d = dirs();

    let src = format!("s3://{bucket}/dl/");
    let (result, logged) = capturing_warnings(s3dlio::s3_copy::download_objects(
        &src, &d.dest, 4, true, None,
    ));
    let err = result.expect_err("the leaky key fails");
    let msg = format!("{err:#}");
    assert_leaky_text_is_sanitised_but_useful("the returned error", &msg);
    // The failure reaches the ERROR log, and that line is clean too.
    assert!(
        logged.contains("Download failed") && logged.contains("get-leaky"),
        "no ERROR line was captured: {logged:?}"
    );
    assert_no_leaked_secret("the logged error", &logged);
}

/// A bucket whose LIST answers `403` with a message quoting credential-bearing URLs.
fn list_leaky_bucket() -> String {
    let (_port, _state) = ensure_mock_s3_server();
    format!(
        "dl284-{}-{}",
        common::mock_s3_server::LIST_LEAKY_ERROR_MARKER,
        N.fetch_add(1, Ordering::SeqCst)
    )
}

#[test]
fn a_failing_list_with_credential_urls_is_sanitised_in_the_generic_download() {
    let bucket = list_leaky_bucket();
    let d = dirs();
    let (result, logged) = capturing_warnings(generic_download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ));
    let err = result.expect_err("the LIST fails");
    let msg = format!("{err:#}");
    assert_no_leaked_secret("the returned error", &msg);
    assert!(msg.contains("creds.example.invalid"), "{msg}");
    assert_no_raw_control(&msg);
    assert!(msg.contains("\\u{202e}"), "{msg}");
    assert_no_leaked_secret("the log", &logged);
}

#[test]
fn a_failing_list_with_credential_urls_is_sanitised_in_s3_copy() {
    let bucket = list_leaky_bucket();
    let d = dirs();
    let (result, logged) = capturing_warnings(s3dlio::s3_copy::download_objects(
        &format!("s3://{bucket}/dl/"),
        &d.dest,
        4,
        true,
        None,
    ));
    let err = result.expect_err("the LIST fails");
    let msg = format!("{err:#}");
    assert_no_leaked_secret("the returned error", &msg);
    assert!(msg.contains("creds.example.invalid"), "{msg}");
    assert_no_raw_control(&msg);
    assert!(msg.contains("\\u{202e}"), "{msg}");
    assert_no_leaked_secret("the log", &logged);
}

/// Review B1: `s3-cli download` lists the source itself before the shared helper runs; a
/// listing failure quoting credential-bearing URLs must reach stderr clean (the raw cause
/// chain used to be printed by `main`).
#[test]
fn cli_download_with_a_failing_list_prints_no_credentials_or_raw_bidi() {
    let bucket = list_leaky_bucket();
    let d = dirs();
    let (port, _state) = ensure_mock_s3_server();
    for recursive in [false, true] {
        let src = format!("s3://{bucket}/dl/");
        let dest = d.dest.display().to_string();
        let mut args = vec!["download"];
        if recursive {
            args.push("-r");
        }
        args.extend([src.as_str(), dest.as_str()]);
        let out = s3_cli_on(port, &args);
        assert!(
            !out.status.success(),
            "the LIST failure must fail the command"
        );
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        for (what, text) in [("stderr", &stderr), ("stdout", &stdout)] {
            assert_no_leaked_secret(what, text);
            assert!(
                !text.contains('\u{202e}'),
                "{what} holds a raw U+202E: {text:?}"
            );
        }
        assert!(
            stderr.contains("creds.example.invalid"),
            "the message is still useful: {stderr:?}"
        );
        assert!(!d.dest.exists(), "nothing is created for a failed listing");
    }
}

/// The same through `s3-cli cp <uri> <dir>` (it goes through the shared helper).
#[test]
fn cli_cp_download_with_a_failing_list_prints_no_credentials_or_raw_bidi() {
    let bucket = list_leaky_bucket();
    let d = dirs();
    let (port, _state) = ensure_mock_s3_server();
    let src = format!("s3://{bucket}/dl/");
    let dest = d.dest.display().to_string();
    let out = s3_cli_on(port, &["cp", "-r", src.as_str(), dest.as_str()]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_no_leaked_secret("stderr", &stderr);
    assert!(!stderr.contains('\u{202e}'), "raw U+202E: {stderr:?}");
}

/// An upload whose failure quotes credential-bearing URLs: the returned error and the ERROR
/// log line are clean, and the failure reaches the log at all (the early `??` return used to
/// skip it).
#[test]
fn s3_copy_upload_failure_is_sanitised_and_logged() {
    let (_port, _state) = ensure_mock_s3_server();
    let bucket = format!("up284-{}", N.fetch_add(1, Ordering::SeqCst));
    let d = dirs();
    let file = d.src.join(format!(
        "{}.bin",
        common::mock_s3_server::PUT_LEAKY_ERROR_MARKER
    ));
    write(&file, b"payload");

    let (result, logged) = capturing_warnings(s3dlio::s3_copy::upload_files(
        &format!("s3://{bucket}/up/"),
        &[file],
        2,
        false,
        None,
    ));
    let err = result.expect_err("the PUT fails");
    let msg = format!("{err:#}");
    assert_leaky_text_is_sanitised_but_useful_for_upload("the returned error", &msg);
    assert!(
        logged.contains("Upload failed"),
        "no ERROR line was captured: {logged:?}"
    );
    assert_no_leaked_secret("the logged error", &logged);
}

fn assert_leaky_text_is_sanitised_but_useful_for_upload(what: &str, text: &str) {
    assert_no_leaked_secret(what, text);
    assert!(
        text.contains("creds.example.invalid"),
        "{what} lost the host: {text:?}"
    );
    assert!(
        text.contains("put-leaky"),
        "{what} does not name the file: {text:?}"
    );
    assert_no_raw_control(text);
    assert!(
        text.contains("\\u{202e}"),
        "{what} lost the escape: {text:?}"
    );
}

/// A per-object GET failure quoting credential-bearing URLs reaches the CLI's stderr clean.
#[test]
fn cli_download_with_a_failing_get_prints_no_credentials_or_raw_bidi() {
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("dl284-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, &["dl/ok.bin", LEAKY_KEY]);
    let d = dirs();
    let src = format!("s3://{bucket}/dl/");
    let dest = d.dest.display().to_string();
    let out = s3_cli_on(port, &["download", "-r", src.as_str(), dest.as_str()]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    for (what, text) in [("stderr", &stderr), ("stdout", &stdout)] {
        assert_no_leaked_secret(what, text);
        assert!(!text.contains('\u{202e}'), "{what}: raw U+202E in {text:?}");
    }
    assert!(d.dest.join("ok.bin").exists());
}
