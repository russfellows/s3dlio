// tests/oplog_lifecycle_reinit_same_path_during_drain.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294.
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: finalize runs on one thread while another thread re-inits with the
// SAME path. Finalize publishes "Finalized" BEFORE its writer has drained, so a
// re-init that did not wait would truncate the file while session 1's writer
// still holds it open and keeps appending: the result would be a corrupt or
// mixed stream. Assertion: after session 2 is finalized the file is a complete
// zstd stream holding exactly session 2's rows.

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

/// A bulky, poorly compressible tag so the writer has real work to drain.
fn bulky(session: &str, i: usize) -> String {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let noise: String = (0..400)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            char::from(b'a' + (x % 26) as u8)
        })
        .collect();
    format!("{session}-{i}-{noise}")
}

#[test]
fn reinit_with_the_same_path_waits_for_the_old_writer() {
    // Lossless: every row is kept, so row counts are exact.
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::set_var("S3DLIO_OPLOG_LOSSLESS", "1");
    }
    const ROUNDS: usize = 15;
    const ROWS: usize = 6000;
    let dir = tempfile::tempdir().unwrap();

    for round in 0..ROUNDS {
        let path = dir.path().join(format!("r{round}.tsv.zst"));
        init_op_logger(path.to_str().unwrap()).unwrap();
        let logger = global_logger().unwrap();
        for i in 0..ROWS {
            assert_eq!(logger.try_log(entry(&bulky("s1", i))), LogOutcome::Accepted);
        }
        drop(logger);

        let finalizer = std::thread::spawn(finalize_op_logger);
        // Finalize has swapped the state as soon as there is no active logger;
        // its writer may still be draining into `path`.
        while global_logger().is_some() {
            std::thread::yield_now();
        }
        init_op_logger(path.to_str().unwrap()).expect("re-init with the same path");

        let logger2 = global_logger().expect("session 2 is Active after init");
        for i in 0..10 {
            assert_eq!(
                logger2.try_log(entry(&format!("s2-{i}"))),
                LogOutcome::Accepted
            );
        }
        drop(logger2);
        finalizer.join().unwrap();
        finalize_op_logger();

        let tags = logged_tags(&path);
        assert_eq!(tags.len(), 10, "round {round}: {} rows", tags.len());
        assert!(tags.iter().all(|t| t.starts_with("s2-")), "round {round}");
    }
}
