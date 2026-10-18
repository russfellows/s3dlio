// tests/test_cli_error_redaction.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Every `s3-cli` command's final error is printed by `main` through
// `s3dlio::sanitized_error`: credentials in any `scheme://` URI of the provider's message
// (userinfo, signed queries) are removed and control / bidi characters are escaped, for
// `ls`, `get`, `stat`, `put`, ... and not only the download paths. The exit status stays
// non-zero and the line keeps the `Error: ...` form.
//
// The server is the shared in-process mock; its "leaky" markers answer `403` with a
// message quoting two credential-bearing URLs and a U+202E bidi override.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{
    ensure_mock_s3_server, GET_LEAKY_ERROR_MARKER, LIST_LEAKY_ERROR_MARKER, PUT_LEAKY_ERROR_MARKER,
};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static N: AtomicU64 = AtomicU64::new(0);

/// Everything secret in `common::mock_s3_server::LEAKY_ERROR_MESSAGE`.
const LEAKY_SECRETS: [&str; 5] = ["SECRETPW1", "admin:", "SIGONE1", "SIGTWO2", "CREDONE1"];

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

fn bucket_with(marker: &str) -> String {
    format!("cli-redact-{marker}-{}", N.fetch_add(1, Ordering::SeqCst))
}

/// `text` without ANSI colour sequences (`ESC [ ... m`) that the log subscriber adds.
fn strip_ansi(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The command failed, and none of the given streams holds a secret or a raw control / bidi
/// character.
fn assert_failed_clean(what: &str, out: &Output, streams: &[&str]) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        !out.status.success(),
        "{what}: must fail (stdout {stdout:?}, stderr {stderr:?})"
    );
    let mut shown = String::new();
    for stream in streams {
        let text = strip_ansi(if *stream == "stderr" {
            &stderr
        } else {
            &stdout
        });
        for secret in LEAKY_SECRETS {
            assert!(
                !text.contains(secret),
                "{what}: {stream} leaked `{secret}`: {text:?}"
            );
        }
        assert!(
            !text
                .chars()
                .any(|c| (c.is_control() && c != '\n' && c != '\t')
                    || matches!(c, '\u{202e}' | '\u{2028}' | '\u{2029}')),
            "{what}: {stream} holds a raw control or bidi character: {text:?}"
        );
        shown.push_str(&text);
    }
    shown
}

/// The message is still useful: the harmless host, the second URI without its signature, and
/// the escaped bidi override survive.
fn assert_still_useful(what: &str, text: &str) {
    assert!(
        text.contains("creds.example.invalid"),
        "{what}: lost the host: {text:?}"
    );
    assert!(
        text.contains("s3://other-bucket/k"),
        "{what}: lost the second URI: {text:?}"
    );
    assert!(
        text.contains("\\u{202e}"),
        "{what}: the bidi override is not shown escaped: {text:?}"
    );
}

#[test]
fn ls_error_is_redacted() {
    let (port, _state) = ensure_mock_s3_server();
    let bucket = bucket_with(LIST_LEAKY_ERROR_MARKER);
    for sub in ["ls", "list"] {
        let uri = format!("s3://{bucket}/dl/");
        let out = s3_cli_on(port, &[sub, uri.as_str()]);
        let shown = assert_failed_clean(sub, &out, &["stderr", "stdout"]);
        assert!(shown.contains("Error:"), "{sub}: `Error:` line: {shown:?}");
        assert_still_useful(sub, &shown);
    }
}

#[test]
fn ls_recursive_error_is_redacted() {
    let (port, _state) = ensure_mock_s3_server();
    let bucket = bucket_with(LIST_LEAKY_ERROR_MARKER);
    let uri = format!("s3://{bucket}/dl/");
    let out = s3_cli_on(port, &["ls", "-r", uri.as_str()]);
    let shown = assert_failed_clean("ls -r", &out, &["stderr", "stdout"]);
    assert_still_useful("ls -r", &shown);
}

#[test]
fn get_error_is_redacted() {
    let (port, state) = ensure_mock_s3_server();
    let bucket = bucket_with("b");
    let key = format!("dl/{GET_LEAKY_ERROR_MARKER}-key");
    state.seed_bucket(&bucket, &[key.as_str()]);
    let uri = format!("s3://{bucket}/{key}");
    let out = s3_cli_on(port, &["get", uri.as_str()]);
    // The per-object cause is a `GET failed` log line; the command ends with an `Error:` line.
    let shown = assert_failed_clean("get", &out, &["stderr", "stdout"]);
    assert!(shown.contains("GET failed"), "get: log line: {shown:?}");
    assert_still_useful("get", &shown);
}

#[test]
fn put_error_is_redacted() {
    let (port, _state) = ensure_mock_s3_server();
    let bucket = bucket_with("b");
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join(format!("{PUT_LEAKY_ERROR_MARKER}.bin"));
    std::fs::write(&file, b"payload").expect("write");
    let file = file.display().to_string();
    let dest = format!("s3://{bucket}/up/");
    for sub in ["upload", "cp"] {
        let out = s3_cli_on(port, &[sub, file.as_str(), dest.as_str()]);
        // Both streams: the final `Error:` line (stderr) and the library's `Upload task failed`
        // log line (stdout) must be clean.
        let shown = assert_failed_clean(sub, &out, &["stderr", "stdout"]);
        assert!(shown.contains("Error:"), "{sub}: `Error:` line: {shown:?}");
    }
}
