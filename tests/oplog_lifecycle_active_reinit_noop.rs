// tests/oplog_lifecycle_active_reinit_noop.rs
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

// Scenario: init while Active is a documented no-op returning Ok. It does NOT
// create or truncate the second path and the session keeps writing to the first.

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
fn init_while_active_is_a_noop() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.tsv.zst");
    let second = dir.path().join("second.tsv.zst");
    init_op_logger(first.to_str().unwrap()).unwrap();
    init_op_logger(second.to_str().unwrap()).expect("init while Active returns Ok");
    assert!(
        !second.exists(),
        "init while Active must not create the new path"
    );

    let logger = global_logger().unwrap();
    assert_eq!(logger.log(entry("only")), LogOutcome::Accepted);
    finalize_op_logger();
    assert_eq!(logged_tags(&first), vec!["only".to_string()]);
    assert!(!second.exists());
}
