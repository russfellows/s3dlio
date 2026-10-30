// tests/oplog_lifecycle_cycles_no_leaks.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294 (round 2).
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: 500 init/finalize cycles, mixing the same path (replace) and
// different paths, leave no extra file descriptors and no extra OS threads
// behind (writer threads are joined/exit, files closed), and every file is a
// complete zstd stream with its rows.

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

fn count(dir: &str) -> usize {
    std::fs::read_dir(dir).unwrap().count()
}

#[test]
fn five_hundred_cycles_leak_neither_fds_nor_threads() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let cycle = |i: usize| {
        // Pattern: A, A, B, C, B, ... : repeats of the same path and new paths.
        let name = match i % 4 {
            0 | 1 => "same.tsv.zst".to_string(),
            2 => format!("diff{i}.tsv.zst"),
            _ => "other.tsv.zst".to_string(),
        };
        let path = dir.path().join(name);
        init_op_logger(path.to_str().unwrap()).unwrap();
        let logger = global_logger().unwrap();
        assert_eq!(
            logger.try_log(entry(&format!("c{i}"))),
            LogOutcome::Accepted
        );
        drop(logger);
        finalize_op_logger();
        assert_eq!(logged_tags(&path), vec![format!("c{i}")], "cycle {i}");
    };
    // Warm-up: the first cycle may start lazily created process-wide threads/fds
    // (e.g. the tracing or dotenv machinery); the baseline is taken after it.
    cycle(0);
    let (fds, threads) = (count("/proc/self/fd"), count("/proc/self/task"));
    for i in 1..=500 {
        cycle(i);
    }
    // The writer thread signals completion (the "done" message) just before it
    // exits, so allow it a moment to disappear from /proc.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while count("/proc/self/task") > threads && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(count("/proc/self/fd"), fds, "file descriptors leaked");
    assert_eq!(count("/proc/self/task"), threads, "OS threads leaked");
}
