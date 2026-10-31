// tests/oplog_lifecycle_reinit_after_finalize.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294 (it replaces the #170
// test that asserted the refusal).
//
// The op-log is process-global (Uninitialized -> Active -> Finalized -> Active
// ...), so each lifecycle scenario lives in its OWN integration-test file: every
// file under tests/ is a separate binary and therefore a separate process, so
// test ordering cannot hide a defect.

// Scenario: init after finalize starts a NEW session. Session 2 writes a
// complete, valid op-log of its own (header + only its own rows), the finished
// session-1 log is untouched, op_log_stats() and the clock offset are session
// 2's, the process client ID carries over, init while Active is still a no-op,
// and a failed re-init leaves the op-log Finalized so a corrected call works.

use s3dlio::s3_logger::{get_client_id, set_client_id, LogEntry, LogOutcome};
use s3dlio::{
    finalize_op_logger, get_clock_offset, global_logger, init_op_logger, op_log_stats,
    set_clock_offset,
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
fn init_after_finalize_starts_a_fresh_session() {
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
    let third = dir.path().join("third.tsv.zst");

    set_client_id("agent-294").unwrap();
    init_op_logger(first.to_str().unwrap()).unwrap();
    set_clock_offset(12_345).unwrap();
    let logger = global_logger().unwrap();
    for i in 0..5 {
        assert_eq!(
            logger.try_log(entry(&format!("s1-{i}"))),
            LogOutcome::Accepted
        );
    }
    finalize_op_logger();
    assert!(global_logger().is_none());
    let first_bytes = std::fs::read(&first).unwrap();
    assert_eq!(logged_tags(&first).len(), 5);
    assert_eq!(op_log_stats().unwrap().accepted, 5);

    // The restart.
    init_op_logger(second.to_str().unwrap()).expect("init after finalize must start a new session");
    assert!(global_logger().is_some(), "new session must be Active");
    assert_eq!(
        get_clock_offset().unwrap(),
        0,
        "a new session starts with a fresh clock offset"
    );
    assert_eq!(get_client_id(), "agent-294", "client ID carries over");
    let fresh = op_log_stats().expect("stats of the new session");
    assert_eq!(fresh.accepted, 0, "stats are the new session's: {fresh:?}");

    // Init while Active with another file is an error: the new path is not created.
    init_op_logger(third.to_str().unwrap()).expect_err("init with another file while Active");
    assert!(!third.exists(), "init while Active created a file");
    init_op_logger(second.to_str().unwrap()).expect("the active file again: a no-op");

    let logger2 = global_logger().unwrap();
    for i in 0..3 {
        assert_eq!(
            logger2.try_log(entry(&format!("s2-{i}"))),
            LogOutcome::Accepted
        );
    }
    finalize_op_logger();
    let stats2 = op_log_stats().unwrap();
    assert_eq!(stats2.accepted, 3, "{stats2:?}");
    assert!(stats2.is_complete(), "{stats2:?}");

    assert_eq!(logged_tags(&second), ["s2-0", "s2-1", "s2-2"]);
    assert_eq!(
        std::fs::read(&first).unwrap(),
        first_bytes,
        "session 2 modified the finished session-1 log"
    );

    // A failed re-init (missing parent directory) errors, creates nothing and
    // leaves the op-log Finalized; a corrected call then succeeds.
    let bad = dir.path().join("no-such-dir").join("x.tsv.zst");
    init_op_logger(bad.to_str().unwrap()).expect_err("bad path must fail");
    assert!(global_logger().is_none(), "failed init must not activate");
    init_op_logger(third.to_str().unwrap()).unwrap();
    assert_eq!(
        global_logger().unwrap().try_log(entry("s3-0")),
        LogOutcome::Accepted
    );
    finalize_op_logger();
    assert_eq!(logged_tags(&third), ["s3-0"]);
    assert_eq!(logged_tags(&second), ["s2-0", "s2-1", "s2-2"]);
}
