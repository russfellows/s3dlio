// tests/oplog_lifecycle_reinit_rejected.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #170.
//
// The op-log is process-global (Uninitialized -> Active -> Finalized), so each
// lifecycle scenario lives in its OWN integration-test file: every file under
// tests/ is a separate binary and therefore a separate process, so test
// ordering cannot hide a defect.

// Scenario: init after finalize is an error and performs NO filesystem action:
// it neither creates a new file nor truncates an existing one (including the
// finished log of the first session).

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use std::path::Path;
use std::time::SystemTime;

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

/// Decompress an op-log and return the `file` column of every data row.
fn logged_tags(path: &Path) -> Vec<String> {
    let raw = std::fs::read(path).expect("read op-log");
    let text = zstd::decode_all(raw.as_slice()).expect("op-log must be a complete zstd stream");
    let text = String::from_utf8(text).expect("op-log must be UTF-8");
    let mut lines = text.lines();
    assert!(
        lines
            .next()
            .is_some_and(|h| h.starts_with("idx\tthread\top\t")),
        "op-log must start with the header row"
    );
    lines
        .map(|l| l.split('\t').nth(7).expect("file column").to_string())
        .collect()
}

#[test]
fn init_after_finalize_errors_and_touches_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.tsv.zst");
    init_op_logger(first.to_str().unwrap()).unwrap();
    let logger = global_logger().unwrap();
    for i in 0..5 {
        assert_eq!(logger.log(entry(&format!("e{i}"))), LogOutcome::Accepted);
    }
    finalize_op_logger();
    let first_bytes = std::fs::read(&first).unwrap();
    assert_eq!(logged_tags(&first).len(), 5);

    // Pre-existing file with known bytes: must survive unchanged.
    let existing = dir.path().join("existing.tsv.zst");
    let known = b"known bytes that must not be truncated".to_vec();
    std::fs::write(&existing, &known).unwrap();
    // A path that does not exist yet: must not be created.
    let fresh = dir.path().join("fresh.tsv.zst");

    for requested in [&existing, &fresh, &first] {
        let err =
            init_op_logger(requested.to_str().unwrap()).expect_err("init after finalize must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("finalized") && msg.contains("one op-log session"),
            "error must say the op-log was already finalized and a process supports one session; got: {msg}"
        );
    }

    assert_eq!(
        std::fs::read(&existing).unwrap(),
        known,
        "existing file was modified"
    );
    assert!(
        !fresh.exists(),
        "init after finalize created the requested file"
    );
    assert_eq!(
        std::fs::read(&first).unwrap(),
        first_bytes,
        "re-init truncated or rewrote the finished first-session log"
    );
    assert!(
        global_logger().is_none(),
        "failed init must not revive the logger"
    );
}
