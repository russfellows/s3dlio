// tests/oplog_lifecycle_finalize_inactive.rs
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

// Scenario: after finalize the op-log is inactive. global_logger() is None
// (new stores are unlogged), a Logger captured earlier reports Closed -- never
// "channel full" -- and the clock-offset accessors report "not initialized".

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{
    finalize_op_logger, get_clock_offset, global_logger, init_op_logger, set_clock_offset,
};
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
fn finalize_makes_the_oplog_inactive_and_closes_captured_loggers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");
    init_op_logger(path.to_str().unwrap()).unwrap();

    let captured = global_logger().expect("logger active after init");
    for i in 0..10 {
        assert_eq!(
            captured.log(entry(&format!("pre-{i}"))),
            LogOutcome::Accepted
        );
    }
    assert!(set_clock_offset(0).is_ok());

    finalize_op_logger();

    assert!(
        global_logger().is_none(),
        "global_logger() must be None after finalize so new stores are unlogged"
    );
    assert!(
        set_clock_offset(1).is_err(),
        "no active logger to configure"
    );
    assert!(get_clock_offset().is_err(), "no active logger to query");

    // A store created before finalize still holds a Logger clone: it must be
    // told the logger is closed, not that a channel is full, and nothing it
    // submits may reach the file.
    for i in 0..100 {
        assert_eq!(
            captured.log(entry(&format!("post-{i}"))),
            LogOutcome::Closed,
            "logging to a finalized Logger must report Closed"
        );
    }

    let tags = logged_tags(&path);
    assert_eq!(tags.len(), 10, "only the 10 pre-finalize entries: {tags:?}");
    assert!(tags.iter().all(|t| t.starts_with("pre-")));
}
