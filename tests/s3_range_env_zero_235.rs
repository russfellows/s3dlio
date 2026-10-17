// tests/s3_range_env_zero_235.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression tests for s3dlio issue #235.
//
// Bug: the zero-limit checks of #211 validate caller ARGUMENTS only. A zero taken from the
// environment was not validated:
//   * S3DLIO_CHUNK_SIZE=0 made the concurrent range GET loop forever building ranges
//     (unbounded memory);
//   * S3DLIO_RANGE_CONCURRENCY=0 created a semaphore with no permits, so every range waited
//     forever.
//
// Contract under test: a zero from either variable is rejected with an error that names the
// variable, before any range is allocated or any semaphore is awaited; valid settings still
// download the object unchanged.
//
// Mechanism: each case runs in its OWN child process (this test binary re-executed on one
// `--exact` test), against the in-process mock S3 server, under a hard wall-clock timeout and
// an address-space limit. The unfixed code either spins allocating ranges (the limit turns
// that into an abort instead of exhausting the host) or hangs on the empty semaphore (the
// timeout turns that into a failure); neither can freeze or exhaust the test run.

// The address-space limit (`ulimit -v`) is a Linux mechanism, so the file is Linux-only.
#![cfg(target_os = "linux")]

// Only the mock server file is included, not all of tests/common.  This binary uses part of
// the mock's API (a few of its request-recording fields and helper methods are unused here),
// so the expectation is on this one module and fails if the warnings ever stop.
#[path = "common/mock_s3_server.rs"]
#[expect(
    dead_code,
    reason = "this binary uses only part of the shared mock server API"
)]
mod mock_s3_server;

use mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, SIZED_OBJECT_BODY, SIZED_OBJECT_MARKER,
};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const BUCKET: &str = "mock-bucket";

/// Environment variable that marks a re-executed child and selects what it does.
const CHILD_CASE: &str = "S3DLIO_TEST_235_CASE";
/// What the child must observe: `reject:<VARIABLE>` or `ok`.
const CHILD_EXPECT: &str = "S3DLIO_TEST_235_EXPECT";
/// Printed by the child when its assertions held.
const SENTINEL: &str = "S3DLIO_235_CHILD_VERIFIED";

/// Hard wall-clock limit for one child.
const HARD_TIMEOUT: Duration = Duration::from_secs(45);
/// Address-space limit (KiB) for one child: generous for the runtime and SDK threads, small
/// enough that an unbounded range vector aborts quickly rather than exhausting the host.
const CHILD_ADDRESS_SPACE_KIB: u64 = 3 * 1024 * 1024;

// ─────────────────────────────── child side ───────────────────────────────

/// Runs only when re-executed by [`run_case`]; a plain `cargo test` finds no marker and returns.
#[test]
fn child_run_case() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    let expect = std::env::var(CHILD_EXPECT).expect("the parent sets the expectation");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let state = ensure_mock_s3_server().1;
    let key = unique_test_key(SIZED_OBJECT_MARKER);
    let uri = format!("s3://{BUCKET}/{key}");

    let result = rt.block_on(async {
        match case.as_str() {
            "concurrent_range" => {
                s3dlio::s3_utils::get_object_concurrent_range_async(&uri, 0, None, None, None).await
            }
            // Threshold 0 sends the (16-byte) object through the known-size range path.
            "uri_optimized" => s3dlio::s3_utils::get_object_uri_optimized_async(&uri).await,
            other => panic!("unknown case {other}"),
        }
    });

    match expect.strip_prefix("reject:") {
        Some(variable) => {
            let err = result.expect_err("a zero setting must be rejected, not downloaded");
            let msg = format!("{err:#}");
            assert!(msg.contains(variable), "error must name {variable}: {msg}");
            assert!(
                msg.contains(">= 1"),
                "error must give the valid range: {msg}"
            );
            // Rejected before any range was requested.
            let gets = state.counts_for(&key_path(BUCKET, &key)).get_calls;
            assert_eq!(gets, 0, "no GET may be sent for a rejected setting");
        }
        None => {
            assert_eq!(expect, "ok");
            let body = result.expect("valid settings must still download the object");
            assert_eq!(&body[..], &SIZED_OBJECT_BODY[..], "bytes must be unchanged");
        }
    }
    println!("{SENTINEL}");
}

// ─────────────────────────────── parent side ───────────────────────────────

struct ChildOutcome {
    timed_out: bool,
    /// Set when the child was still alive after the whole SIGTERM ladder.
    survivor_pid: Option<u32>,
    status: Option<std::process::ExitStatus>,
    output: String,
}

/// The most output kept from one child stream; the rest is read and discarded.
const KEEP_OUTPUT_BYTES: usize = 1 << 20;
/// How long to wait for a reader thread to see end-of-file after the child exited.
const READER_GRACE: Duration = Duration::from_secs(5);

/// Drain `stream` on its own thread, so a chatty child can never fill a pipe and block.
/// The kept text (at most [`KEEP_OUTPUT_BYTES`]) is sent when the stream ends.
fn drain<R: Read + Send + 'static>(mut stream: R) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let room = KEEP_OUTPUT_BYTES.saturating_sub(kept.len());
                    kept.extend_from_slice(&buf[..n.min(room)]);
                }
            }
        }
        // The receiver may be gone (the parent gave up); nothing else to do then.
        tx.send(String::from_utf8_lossy(&kept).into_owned()).ok();
    });
    rx
}

/// A two-step SIGTERM ladder with the default disposition (the child installs no handler).
/// Returns whether the child exited.  There is no SIGKILL: a child that outlives the ladder
/// is reported to the caller, which must not wait on it.
fn terminate(child: &mut std::process::Child) -> bool {
    for _ in 0..2 {
        Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .ok();
        let until = Instant::now() + Duration::from_secs(3);
        while Instant::now() < until {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    false
}

/// Variables from a developer or CI shell that would change which server the child talks
/// to, which credentials or profile it resolves, or how it retries and times out.  Every
/// endpoint, profile and transport variable `s3_client.rs` and the AWS SDK honour is here.
const SCRUBBED_ENV: &[&str] = &[
    "S3DLIO_CHUNK_SIZE",
    "S3DLIO_RANGE_CONCURRENCY",
    "S3DLIO_RANGE_THRESHOLD_MB",
    "S3DLIO_SKIP_HEAD",
    "S3DLIO_FOLLOW_REDIRECTS",
    "S3DLIO_OPERATION_TIMEOUT_SECS",
    "S3DLIO_CONNECT_TIMEOUT_SECS",
    "S3DLIO_MAX_RETRY_ATTEMPTS",
    "S3DLIO_EXTRA_TRANSIENT_CODES",
    "S3DLIO_RT_THREADS",
    "S3DLIO_RT_THREADS_UNSAFE",
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_SESSION_TOKEN",
    "AWS_CA_BUNDLE",
    "AWS_DEFAULT_REGION",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
];

fn run_case(case: &str, expect: &str, env: &[(&str, &str)]) -> ChildOutcome {
    let exe = std::env::current_exe().expect("test executable path");
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(format!(
            "ulimit -v {CHILD_ADDRESS_SPACE_KIB} && exec \"$0\" \"$@\""
        ))
        .arg(exe)
        .args([
            "--exact",
            "child_run_case",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_CASE, case)
        .env(CHILD_EXPECT, expect)
        // Fewer malloc arenas keep the address-space limit about real memory.
        .env("MALLOC_ARENA_MAX", "2")
        .env("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "1")
        .env("S3DLIO_LOAD_DOTENV", "0")
        // No instance-metadata lookups if credentials were somehow missing.
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Values inherited from a developer shell must not change the outcome.
    for name in SCRUBBED_ENV {
        cmd.env_remove(name);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn the child test process");
    let stdout = child.stdout.take().map(drain);
    let stderr = child.stderr.take().map(drain);

    let deadline = Instant::now() + HARD_TIMEOUT;
    let mut timed_out = false;
    let mut survivor_pid = None;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                if !terminate(&mut child) {
                    survivor_pid = Some(child.id());
                }
                break child.try_wait().ok().flatten();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => break None,
        }
    };

    // A surviving child still holds its pipes open: waiting for end-of-file would block.
    let collect = |rx: Option<mpsc::Receiver<String>>| match (rx, survivor_pid) {
        (Some(rx), None) => rx
            .recv_timeout(READER_GRACE)
            .unwrap_or_else(|_| "(output incomplete)".to_string()),
        (Some(_), Some(_)) => "(child still running; output not collected)".to_string(),
        (None, _) => "(no output stream)".to_string(),
    };
    let output = format!("{}\n{}", collect(stdout), collect(stderr));
    ChildOutcome {
        timed_out,
        survivor_pid,
        status,
        output,
    }
}

fn assert_verified(what: &str, o: ChildOutcome) {
    if let Some(pid) = o.survivor_pid {
        panic!(
            "{what}: the child (pid {pid}) is STILL RUNNING after the SIGTERM ladder; it was not \
             killed and not waited for, and needs to be stopped by hand"
        );
    }
    assert!(
        !o.timed_out,
        "{what}: HUNG past {HARD_TIMEOUT:?} (the setting was not validated)\n{}",
        o.output
    );
    assert!(
        o.status.is_some_and(|s| s.success()) && o.output.contains(SENTINEL),
        "{what}: the child did not verify the contract (exit {:?}; a crash here is the \
         unbounded range vector hitting the address-space limit)\n{}",
        o.status,
        o.output
    );
}

#[test]
fn chunk_size_env_zero_is_rejected_by_name_on_the_concurrent_range_path() {
    assert_verified(
        "S3DLIO_CHUNK_SIZE=0, get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_CHUNK_SIZE",
            &[("S3DLIO_CHUNK_SIZE", "0")],
        ),
    );
}

#[test]
fn range_concurrency_env_zero_is_rejected_by_name_on_the_concurrent_range_path() {
    // A valid chunk size of 4 bytes splits the 16-byte object into four ranges, so the
    // concurrency limit is reached.
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0, get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

#[test]
fn chunk_size_env_zero_is_rejected_by_name_on_the_known_size_path() {
    assert_verified(
        "S3DLIO_CHUNK_SIZE=0, get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_CHUNK_SIZE",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "0"),
            ],
        ),
    );
}

#[test]
fn range_concurrency_env_zero_is_rejected_by_name_on_the_known_size_path() {
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0, get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

#[test]
fn valid_settings_still_download_the_object_unchanged() {
    assert_verified(
        "valid S3DLIO_CHUNK_SIZE=4 / S3DLIO_RANGE_CONCURRENCY=2, get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "ok",
            &[
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
    assert_verified(
        "valid settings, get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "ok",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
}

/// The object is 16 bytes and the default chunk is 1 MiB, so the object fits one chunk and
/// the concurrency setting is never needed.  It is still an invalid setting: whether it is
/// rejected must not depend on the object's size.
#[test]
fn range_concurrency_env_zero_is_rejected_for_a_small_object_on_the_concurrent_range_path() {
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0 (small object), get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[("S3DLIO_RANGE_CONCURRENCY", "0")],
        ),
    );
}

#[test]
fn range_concurrency_env_zero_is_rejected_for_a_small_object_on_the_known_size_path() {
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0 (small object), get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

/// A concurrency above what a semaphore can hold would panic in `Semaphore::new`; it is
/// rejected by name instead, for a small object and for one that is split into ranges.
#[test]
fn range_concurrency_env_above_the_semaphore_limit_is_rejected_by_name() {
    let too_many = usize::MAX.to_string();
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=usize::MAX (small object), get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[("S3DLIO_RANGE_CONCURRENCY", &too_many)],
        ),
    );
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=usize::MAX (split object), get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", &too_many),
            ],
        ),
    );
}
