// tests/oplog_lifecycle_failed_init_recoverable.rs
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

// Scenario: a failed init (bad path, unsupported scheme) leaves the op-log
// Uninitialized so a corrected init succeeds.

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
fn failed_init_can_be_retried() {
    let dir = tempfile::tempdir().unwrap();
    let missing_dir = dir.path().join("no-such-dir").join("ops.tsv.zst");
    assert!(init_op_logger(missing_dir.to_str().unwrap()).is_err());
    assert!(init_op_logger("s3://bucket/ops.tsv.zst").is_err());
    assert!(
        global_logger().is_none(),
        "failed init must leave the op-log inactive"
    );

    let good = dir.path().join("ops.tsv.zst");
    init_op_logger(good.to_str().unwrap()).expect("corrected init must succeed");
    let logger = global_logger().expect("active after corrected init");
    assert_eq!(logger.log(entry("ok")), LogOutcome::Accepted);
    finalize_op_logger();
    assert_eq!(logged_tags(&good), vec!["ok".to_string()]);
}
