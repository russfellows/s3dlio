// tests/oplog_lifecycle_reinit_same_path.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294.
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: re-init after finalize with the SAME path. init always creates the
// file with create-and-truncate (as for a first init with a pre-existing file),
// so the second session REPLACES the first session's log: the file ends up
// with the second session's rows only. Also covers a pre-existing unrelated
// file as the new session's destination (truncated, like a first init).

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

fn session(path: &Path, tags: &[&str]) {
    init_op_logger(path.to_str().unwrap()).unwrap();
    let logger = global_logger().unwrap();
    for t in tags {
        assert_eq!(logger.try_log(entry(t)), LogOutcome::Accepted);
    }
    drop(logger);
    finalize_op_logger();
}

#[test]
fn reinit_with_the_same_path_replaces_the_log() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");

    session(&path, &["a0", "a1", "a2", "a3", "a4"]);
    assert_eq!(logged_tags(&path).len(), 5);

    session(&path, &["b0", "b1"]);
    assert_eq!(logged_tags(&path), ["b0", "b1"], "truncated and rewritten");

    // Third session (still the same path): the cycle repeats indefinitely.
    session(&path, &["c0"]);
    assert_eq!(logged_tags(&path), ["c0"]);

    // A pre-existing unrelated file is truncated, as for a first init.
    let existing = dir.path().join("existing.tsv.zst");
    std::fs::write(&existing, b"junk that is not zstd").unwrap();
    session(&existing, &["d0", "d1"]);
    assert_eq!(logged_tags(&existing), ["d0", "d1"]);
}
