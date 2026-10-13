// tests/oplog_lifecycle_diagnostics_once.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle test for s3dlio issue #170 (stderr diagnostics).
//
// Scenario: a Logger captured before finalize is called 100 times after it.
// Every call returns Closed, and stderr carries EXACTLY ONE "op-log finalized"
// line and no "channel full" / "writer has stopped" line.
//
// stderr is only observable from outside the process, so the test re-runs
// itself: the parent spawns this same test binary with a marker env var and
// `--nocapture`, the child runs the scenario and prints a marker, and the
// parent asserts that the child really ran (marker + "1 passed") before it
// inspects the child's stderr. Own process (the op-log is process-global).

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use std::process::Command;
use std::time::SystemTime;

const CHILD_ENV: &str = "S3DLIO_TEST_DIAGNOSTICS_ONCE_CHILD";
const RAN_MARKER: &str = "CHILD-SCENARIO-RAN";
const TEST_NAME: &str = "diagnostics_are_printed_once_per_closed_logger";

fn entry(tag: &str) -> LogEntry {
    let now = SystemTime::now();
    LogEntry {
        idx: 0,
        thread_id: 0,
        operation: "GET".to_string(),
        client_id: String::new(),
        num_objects: 1,
        bytes: 1,
        endpoint: "file://".to_string(),
        file: tag.to_string(),
        error: None,
        start_time: now,
        first_byte_time: None,
        end_time: now,
    }
}

/// The scenario, run inside the child process.
fn child_scenario() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");
    init_op_logger(path.to_str().unwrap()).unwrap();
    let captured = global_logger().expect("active after init");
    assert_eq!(captured.try_log(entry("pre")), LogOutcome::Accepted);
    finalize_op_logger();
    for i in 0..100 {
        assert_eq!(
            captured.try_log(entry(&format!("post-{i}"))),
            LogOutcome::Closed
        );
    }
    println!("{RAN_MARKER}: 100 post-finalize calls all returned Closed");
}

#[test]
fn diagnostics_are_printed_once_per_closed_logger() {
    if std::env::var_os(CHILD_ENV).is_some() {
        child_scenario();
        return;
    }

    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env_remove("S3DLIO_OPLOG_BUF")
        .env_remove("S3DLIO_OPLOG_LOSSLESS")
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // The child must really have run the scenario (not filtered out / vacuous).
    assert!(
        out.status.success(),
        "child failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        out.status
    );
    assert!(
        stdout.contains(RAN_MARKER),
        "child did not run the scenario\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("1 passed"), "child stdout: {stdout}");

    let finalized = stderr
        .lines()
        .filter(|l| l.contains("op-log finalized"))
        .count();
    assert_eq!(
        finalized, 1,
        "exactly one finalized diagnostic\nstderr={stderr}"
    );
    for unwanted in ["channel full", "writer has stopped"] {
        assert_eq!(
            stderr.lines().filter(|l| l.contains(unwanted)).count(),
            0,
            "unexpected '{unwanted}' on stderr\nstderr={stderr}"
        );
    }
}
