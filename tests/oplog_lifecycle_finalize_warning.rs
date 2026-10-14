// tests/oplog_lifecycle_finalize_warning.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle test for s3dlio issue #170 (loss detection at finalize).
//
// Scenario: the op-log file hits a REAL write error while entries are queued.
// The child process caps its own file size (RLIMIT_FSIZE, with SIGXFSZ ignored
// so the kernel returns EFBIG instead of killing the process), logs far more
// than fits, and then four threads call `finalize_op_logger()` at once. Then:
//
// - the stats show the loss (`written < accepted`, `writer_error` set,
//   `is_complete()` false) even though no call ever failed after the writer
//   stopped being noticed by `writer_failed` alone;
// - finalize prints EXACTLY ONE "op-log is INCOMPLETE" warning on stderr, with
//   the counts and the writer's error, even under concurrent finalize;
// - the warning is tailored: in lossless mode it gives no queue-full advice.
//
// stderr is only observable from outside, so the test re-runs itself (as
// oplog_lifecycle_diagnostics_once does): the parent spawns this binary with a
// marker env var, the child runs the scenario and prints a marker, and the
// parent asserts that the child really ran before it inspects stderr. Own
// process (the op-log is process-global, so is the rlimit).

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
use std::process::Command;
use std::time::SystemTime;

const CHILD_ENV: &str = "S3DLIO_TEST_FINALIZE_WARNING_CHILD";
const RAN_MARKER: &str = "CHILD-SCENARIO-RAN";
const TEST_NAME: &str = "finalize_warns_once_when_the_writer_loses_queued_entries";
/// The child may not grow any file beyond this many bytes.
const FILE_LIMIT_BYTES: u64 = 64 * 1024;
const ENTRIES: u64 = 200;

/// ~100 KB of hex digits in the `file` column: hard to compress, so a few
/// entries exceed the file limit.
fn big_entry(seed: u64) -> LogEntry {
    let mut x = seed | 1;
    let mut file = String::with_capacity(100_000);
    while file.len() < 100_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        file.push_str(&format!("{x:016x}"));
    }
    let now = SystemTime::now();
    LogEntry {
        idx: 0,
        thread_id: 0,
        operation: "GET".to_string(),
        client_id: String::new(),
        num_objects: 1,
        bytes: 1,
        endpoint: "file://".to_string(),
        file,
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
    let logger = global_logger().expect("active after init");

    // From now on writes past the limit fail with EFBIG instead of killing us.
    // SAFETY: plain syscalls with valid arguments.
    unsafe {
        assert_ne!(libc::signal(libc::SIGXFSZ, libc::SIG_IGN), libc::SIG_ERR);
        let limit = libc::rlimit {
            rlim_cur: FILE_LIMIT_BYTES,
            rlim_max: FILE_LIMIT_BYTES,
        };
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &limit), 0);
    }

    let mut accepted_by_caller = 0_u64;
    for i in 0..ENTRIES {
        match logger.try_log(big_entry(i)) {
            LogOutcome::Accepted => accepted_by_caller += 1,
            // The writer may already have stopped: refused, counted as writer_failed.
            LogOutcome::WriterFailed => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    // Four threads finalize at once: only one may warn.
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(finalize_op_logger))
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let st = op_log_stats().expect("stats are kept after finalize");
    assert_eq!(st.accepted, accepted_by_caller);
    assert!(
        st.written < st.accepted,
        "the entry that hit the limit was accepted but not written: {st:?}"
    );
    assert!(!st.is_complete(), "{st:?}");
    let error = st.writer_error.as_deref().unwrap_or_default();
    assert!(error.contains("write failed"), "{st:?}");
    assert_eq!(st.dropped_full, 0, "queue never fills: {st:?}");
    println!(
        "{RAN_MARKER}: accepted={} written={} writer_failed={}",
        st.accepted, st.written, st.writer_failed
    );
}

fn run_child(lossless: &str) -> (String, String) {
    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env("S3DLIO_OPLOG_LOSSLESS", lossless)
        .env_remove("S3DLIO_OPLOG_BUF")
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
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
    (stdout, stderr)
}

#[test]
fn finalize_warns_once_when_the_writer_loses_queued_entries() {
    if std::env::var_os(CHILD_ENV).is_some() {
        child_scenario();
        return;
    }

    for lossless in ["0", "1"] {
        let (_stdout, stderr) = run_child(lossless);
        let warnings = stderr
            .lines()
            .filter(|l| l.contains("op-log is INCOMPLETE"))
            .count();
        assert_eq!(
            warnings, 1,
            "lossless={lossless}: exactly one warning under concurrent finalize\nstderr={stderr}"
        );
        assert!(
            stderr.contains("accepted entries were never written")
                && stderr.contains("the writer failed: write failed"),
            "lossless={lossless}: warning must carry the counts and the error\nstderr={stderr}"
        );
        assert!(
            stderr.contains("Check the op-log destination"),
            "lossless={lossless}: {stderr}"
        );
        // Tailored: nothing was dropped for want of queue space, so there is no
        // queue advice in either mode (and never LOSSLESS advice in lossless mode).
        assert!(
            !stderr.contains("S3DLIO_OPLOG_LOSSLESS=1") && !stderr.contains("S3DLIO_OPLOG_BUF"),
            "lossless={lossless}: no queue-full advice expected\nstderr={stderr}"
        );
        assert!(
            !stderr.contains("were dropped because the queue was full"),
            "lossless={lossless}: {stderr}"
        );
    }
}
