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

#[allow(dead_code)] // the shared mock module is imported whole; this binary uses part of it
mod common;

use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, SIZED_OBJECT_BODY, SIZED_OBJECT_MARKER,
};
use std::process::{Command, Stdio};
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
    let (_port, state) = ensure_mock_s3_server();
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
    status: Option<std::process::ExitStatus>,
    output: String,
}

fn terminate(child: &mut std::process::Child) {
    // A two-step ladder with the default SIGTERM disposition; the child installs no handler.
    for _ in 0..2 {
        let _ = Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status();
        let until = Instant::now() + Duration::from_secs(3);
        while Instant::now() < until {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

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
        // Values inherited from a developer shell must not change the outcome.
        .env_remove("S3DLIO_CHUNK_SIZE")
        .env_remove("S3DLIO_RANGE_CONCURRENCY")
        .env_remove("S3DLIO_RANGE_THRESHOLD_MB")
        .env("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "1")
        .env_remove("S3DLIO_SKIP_HEAD")
        .env("S3DLIO_LOAD_DOTENV", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn the child test process");

    let deadline = Instant::now() + HARD_TIMEOUT;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                terminate(&mut child);
                break child.try_wait().ok().flatten();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => break None,
        }
    };
    let out = child.wait_with_output();
    let output = match out {
        Ok(o) => format!(
            "{}\n{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => format!("(output unavailable: {e})"),
    };
    ChildOutcome {
        timed_out,
        status,
        output,
    }
}

fn assert_verified(what: &str, o: ChildOutcome) {
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
