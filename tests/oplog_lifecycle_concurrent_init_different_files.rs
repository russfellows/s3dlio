// tests/oplog_lifecycle_concurrent_init_different_files.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294 (round 2).
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: 16 threads call init_op_logger at the same moment (a Barrier). With 16
// DIFFERENT files: exactly one wins and creates its file; the other 15 get the
// error naming the winner's file and their own; only one file exists.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use std::path::Path;
use std::sync::{Arc, Barrier};
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
fn sixteen_concurrent_inits_with_different_files_have_one_winner() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let barrier = Arc::new(Barrier::new(16));
    let handles: Vec<_> = (0..16)
        .map(|i| {
            let (barrier, path) = (
                Arc::clone(&barrier),
                dir.path().join(format!("f{i}.tsv.zst")),
            );
            std::thread::spawn(move || {
                barrier.wait();
                (path.clone(), init_op_logger(path.to_str().unwrap()))
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let winners: Vec<_> = results.iter().filter(|(_, r)| r.is_ok()).collect();
    assert_eq!(winners.len(), 1, "exactly one init wins");
    let winner = &winners[0].0;
    for (path, r) in &results {
        if path == winner {
            continue;
        }
        let err = r.as_ref().expect_err("every loser gets the error");
        assert_eq!(err.kind(), std::io::ErrorKind::ResourceBusy);
        let msg = err.to_string();
        assert!(
            msg.contains(winner.to_str().unwrap()),
            "names the winner: {msg}"
        );
        assert!(
            msg.contains(path.to_str().unwrap()),
            "names the loser: {msg}"
        );
        assert!(!path.exists(), "a losing path must not be created");
    }
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "one file"
    );
    let logger = global_logger().unwrap();
    assert_eq!(logger.try_log(entry("w")), LogOutcome::Accepted);
    drop(logger);
    finalize_op_logger();
    assert_eq!(logged_tags(winner), vec!["w".to_string()]);
}
