// tests/oplog_lifecycle_finalize_idempotent.rs
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

// Scenario: finalize before init is a no-op that leaves the op-log
// initialisable; finalize repeated (sequentially and from several threads at
// once) never hangs or errors and the log stays complete.

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
fn finalize_is_idempotent_and_safe_before_init() {
    // Uninitialized: no-op, and it does not poison a later init.
    finalize_op_logger();
    finalize_op_logger();
    assert!(global_logger().is_none());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");
    init_op_logger(path.to_str().unwrap())
        .expect("finalize before init must not prevent a later init");
    let logger = global_logger().unwrap();
    for i in 0..50 {
        assert_eq!(logger.log(entry(&format!("e{i}"))), LogOutcome::Accepted);
    }

    // Concurrent finalizers: each must return only after the log is complete.
    let path = std::sync::Arc::new(path);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let path = std::sync::Arc::clone(&path);
            std::thread::spawn(move || {
                finalize_op_logger();
                logged_tags(&path).len()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(
            h.join().unwrap(),
            50,
            "a finalize returned before the log was complete"
        );
    }

    // Further sequential finalizes.
    finalize_op_logger();
    finalize_op_logger();
    assert_eq!(logged_tags(&path).len(), 50);
    assert!(global_logger().is_none());
}
