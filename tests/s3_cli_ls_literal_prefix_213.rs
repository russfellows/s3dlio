// tests/s3_cli_ls_literal_prefix_213.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #213 through the real `s3-cli` binary:
// `s3-cli ls <uri>` lists the URI as a LITERAL key prefix (never a regex), and
// the regex feature stays, but only through the explicit `-p/--pattern` option.
//
// The server is the shared in-process mock with a seeded catalog that honours
// prefix and delimiter like S3 does (tests/common/mock_s3_server.rs).

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, MockS3State};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const CATALOG: &[&str] = &[
    "data/run.1",
    "data/run.1.old",
    "data/run.1/child",
    "data/run_1",
    "data/prerun.1",
    "foo",
    "foobar",
    "foo/a",
    "afoo",
    "other/x",
    "weird/p(",
    "weird/p(x",
    "weird/pp",
    "dl/a.npz",
    "dl/b.txt",
    "dl/sub/c.npz",
];

fn seeded() -> (u16, MockS3State, String) {
    static N: AtomicU64 = AtomicU64::new(0);
    let (port, state) = ensure_mock_s3_server();
    let bucket = format!("cli213-{}", N.fetch_add(1, Ordering::SeqCst));
    state.seed_bucket(&bucket, CATALOG);
    (port, state, bucket)
}

struct Ls {
    ok: bool,
    uris: Vec<String>,
    stdout: String,
    stderr: String,
}

/// Run `s3-cli ls <args...>` against the mock; `uris` is every `s3://` line, sorted.
fn s3_cli_ls(port: u16, args: &[&str]) -> Ls {
    let out = Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .arg("ls")
        .args(args)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"))
        .env("AWS_ACCESS_KEY_ID", "test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        .env("AWS_REGION", "us-east-1")
        .output()
        .expect("run s3-cli");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut uris: Vec<String> = stdout
        .lines()
        .filter(|l| l.starts_with("s3://"))
        .map(str::to_string)
        .collect();
    uris.sort();
    Ls {
        ok: out.status.success(),
        uris,
        stdout,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn want(bucket: &str, keys: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = keys.iter().map(|k| format!("s3://{bucket}/{k}")).collect();
    v.sort();
    v
}

fn assert_ls(r: &Ls, bucket: &str, keys: &[&str], what: &str) {
    assert!(
        r.ok,
        "{what}: s3-cli ls failed\nstdout={}\nstderr={}",
        r.stdout, r.stderr
    );
    assert_eq!(r.uris, want(bucket, keys), "{what}\nstdout={}", r.stdout);
}

#[test]
fn ls_run_dot_1_is_a_literal_prefix() {
    let (port, _s, b) = seeded();
    let uri = format!("s3://{b}/data/run.1");
    // not run_1, not prerun.1
    assert_ls(
        &s3_cli_ls(port, &["-r", &uri]),
        &b,
        &["data/run.1", "data/run.1.old", "data/run.1/child"],
        "ls -r",
    );
    assert_ls(
        &s3_cli_ls(port, &[&uri]),
        &b,
        &["data/run.1", "data/run.1.old", "data/run.1/"],
        "ls (non-recursive)",
    );
}

#[test]
fn ls_bare_name_non_recursive_has_no_unrelated_folders() {
    let (port, _s, b) = seeded();
    assert_ls(
        &s3_cli_ls(port, &[&format!("s3://{b}/foo")]),
        &b,
        &["foo", "foobar", "foo/"],
        "ls s3://b/foo (not afoo, not data/ other/ weird/ dl/)",
    );
}

#[test]
fn ls_regex_metacharacters_in_the_uri_are_literal() {
    let (port, _s, b) = seeded();
    assert_ls(
        &s3_cli_ls(port, &["-r", &format!("s3://{b}/weird/p(")]),
        &b,
        &["weird/p(", "weird/p(x"],
        "ls -r s3://b/weird/p(",
    );
}

#[test]
fn ls_explicit_pattern_still_filters_relative_to_the_listing_base() {
    let (port, _s, b) = seeded();
    // trailing slash: the pattern sees the key relative to `dl/`
    assert_ls(
        &s3_cli_ls(port, &["-r", "-p", r"\.npz$", &format!("s3://{b}/dl/")]),
        &b,
        &["dl/a.npz", "dl/sub/c.npz"],
        "ls -r -p '\\.npz$' s3://b/dl/",
    );
    assert_ls(
        &s3_cli_ls(port, &["-r", "-p", "^sub/", &format!("s3://{b}/dl/")]),
        &b,
        &["dl/sub/c.npz"],
        "ls -r -p '^sub/' s3://b/dl/  (relative path)",
    );
    // no trailing slash: same base, because `dl/a.npz` lies under `dl/`
    assert_ls(
        &s3_cli_ls(port, &["-r", "-p", "^a", &format!("s3://{b}/dl")]),
        &b,
        &["dl/a.npz"],
        "ls -r -p '^a' s3://b/dl",
    );
    // a pattern composes with a literal prefix
    assert_ls(
        &s3_cli_ls(port, &["-r", "-p", r"1$", &format!("s3://{b}/data/run.1")]),
        &b,
        &["data/run.1"],
        "ls -r -p '1$' s3://b/data/run.1",
    );
}

#[test]
fn ls_invalid_pattern_is_an_error_not_a_listing() {
    let (port, _s, b) = seeded();
    let r = s3_cli_ls(port, &["-r", "-p", "(", &format!("s3://{b}/dl/")]);
    assert!(!r.ok, "an invalid --pattern must fail: {}", r.stdout);
    assert!(
        r.stderr.contains("Invalid regex pattern"),
        "stderr={}",
        r.stderr
    );
}

#[test]
fn ls_pattern_with_a_host_form_uri_is_relative_to_the_listing_base() {
    let (port, _s, b) = seeded();
    // The CLI prints bucket-form URIs (`s3://bucket/key`) for a host-form argument; the
    // pattern is still matched against the key relative to the listing base.
    for uri in [
        format!("s3://127.0.0.1:{port}/{b}/dl/"),
        format!("s3://127.0.0.1:{port}/{b}/dl"),
    ] {
        assert_ls(
            &s3_cli_ls(port, &["-r", "-p", "^sub/", &uri]),
            &b,
            &["dl/sub/c.npz"],
            &format!("ls -r -p '^sub/' {uri}"),
        );
    }
}
