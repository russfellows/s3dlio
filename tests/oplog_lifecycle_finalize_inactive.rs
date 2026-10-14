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
// "channel full" -- and the clock-offset accessors say the op-log was FINALIZED
// (not "not initialized", which is what they say before any init).

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{
    finalize_op_logger, get_clock_offset, global_logger, init_op_logger, set_clock_offset,
};
use std::io::ErrorKind;
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
fn finalize_makes_the_oplog_inactive_and_closes_captured_loggers() {
    // Do not depend on the caller's environment: the default queue (8192) and
    // non-lossless mode are what this scenario is written for.
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");

    // Before any init the accessors say the logger was never initialised.
    for err in [
        set_clock_offset(1).expect_err("no logger before init"),
        get_clock_offset().expect_err("no logger before init"),
    ] {
        assert_eq!(err.kind(), ErrorKind::NotFound, "{err}");
        assert!(err.to_string().contains("not initialized"), "{err}");
    }

    init_op_logger(path.to_str().unwrap()).unwrap();

    let captured = global_logger().expect("logger active after init");
    for i in 0..10 {
        assert_eq!(
            captured.try_log(entry(&format!("pre-{i}"))),
            LogOutcome::Accepted
        );
    }
    assert!(set_clock_offset(0).is_ok());

    finalize_op_logger();

    assert!(
        global_logger().is_none(),
        "global_logger() must be None after finalize so new stores are unlogged"
    );
    // After finalize the message must say so -- not claim "not initialized",
    // which would send the caller to call init_op_logger() again (it cannot work).
    for err in [
        set_clock_offset(1).expect_err("no active logger to configure"),
        get_clock_offset().expect_err("no active logger to query"),
    ] {
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
        let msg = err.to_string();
        assert!(msg.contains("finalized"), "must say finalized: {msg}");
        assert!(
            !msg.contains("not initialized"),
            "must not say not initialized: {msg}"
        );
    }

    // A store created before finalize still holds a Logger clone: it must be
    // told the logger is closed, not that a channel is full, and nothing it
    // submits may reach the file.
    for i in 0..100 {
        assert_eq!(
            captured.try_log(entry(&format!("post-{i}"))),
            LogOutcome::Closed,
            "logging to a finalized Logger must report Closed"
        );
    }

    let tags = logged_tags(&path);
    assert_eq!(tags.len(), 10, "only the 10 pre-finalize entries: {tags:?}");
    assert!(tags.iter().all(|t| t.starts_with("pre-")));
}
