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

// Scenario (issue #294, round 2): init while Active.
//  - with the SAME file: a documented no-op returning Ok. Nothing is created or
//    truncated; rows logged before and after the call land in the one file.
//  - with a DIFFERENT file: an error that names the file being written and the
//    requested path and says to finalize first. The requested path is not
//    created or truncated and the session carries on unaffected.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
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

const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

/// Decompress an op-log (it must be a complete zstd stream), check the full
/// header line, 13 columns and sequential idx on every row, and return the
/// `file` column of every data row.
fn logged_tags(path: &Path) -> Vec<String> {
    let raw = std::fs::read(path).expect("read op-log");
    let text = zstd::decode_all(raw.as_slice()).expect("op-log must be a complete zstd stream");
    let text = String::from_utf8(text).expect("op-log must be UTF-8");
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(HEADER_LINE),
        "op-log must start with the exact header row"
    );
    lines
        .enumerate()
        .map(|(i, l)| {
            let cols: Vec<&str> = l.split('\t').collect();
            assert_eq!(cols.len(), 13, "row {i} must have 13 columns: {l:?}");
            assert_eq!(cols[0], i.to_string(), "row {i} must carry idx {i}");
            cols[7].to_string()
        })
        .collect()
}

#[test]
fn init_while_active_same_file_is_a_noop_and_different_file_is_an_error() {
    // Do not depend on the caller's environment: the default queue (8192) and
    // non-lossless mode are what this scenario is written for.
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.tsv.zst");
    let second = dir.path().join("second.tsv.zst");
    init_op_logger(first.to_str().unwrap()).unwrap();
    let logger = global_logger().unwrap();
    assert_eq!(logger.try_log(entry("before")), LogOutcome::Accepted);

    // Same file: no-op, no truncation, same session (the stats keep counting).
    init_op_logger(first.to_str().unwrap()).expect("init with the active file is a no-op");
    assert_eq!(logger.try_log(entry("after")), LogOutcome::Accepted);
    assert_eq!(op_log_stats().unwrap().accepted, 2, "same session");

    // Different file: an error that names both files and says what to do.
    let err = init_op_logger(second.to_str().unwrap())
        .expect_err("init with a different file while Active must fail");
    assert_eq!(err.kind(), std::io::ErrorKind::ResourceBusy);
    let msg = err.to_string();
    assert!(
        msg.contains(first.to_str().unwrap()),
        "must name the file being written: {msg}"
    );
    assert!(
        msg.contains(second.to_str().unwrap()),
        "must name the requested file: {msg}"
    );
    assert!(
        msg.contains("finalize"),
        "must say to finalize first: {msg}"
    );
    assert!(!second.exists(), "the requested path must not be created");

    // The session is unaffected and still the same one.
    assert!(global_logger().is_some());
    assert_eq!(logger.try_log(entry("last")), LogOutcome::Accepted);
    assert_eq!(op_log_stats().unwrap().accepted, 3, "same session");
    drop(logger);
    finalize_op_logger();
    assert_eq!(
        logged_tags(&first),
        vec![
            "before".to_string(),
            "after".to_string(),
            "last".to_string()
        ]
    );
    assert!(!second.exists());
}
