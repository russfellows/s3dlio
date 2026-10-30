// tests/oplog_lifecycle_reinit_waits_for_drain.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294.
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: finalize runs on one thread while another thread re-inits. Finalize
// publishes "Finalized" BEFORE its writer has drained, so a re-init that does
// not wait would open session 2 while session 1's writer is still running.
// Assertions: when init returns, the session-1 file is ALREADY a complete zstd
// stream with exactly its own rows (init waited for the drain, without joining
// the finalizing thread), the session-2 file holds only session-2 rows, and
// when many threads re-init at once exactly one file is created.

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
fn reinit_waits_for_the_previous_session_to_drain() {
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
        let p1 = dir.path().join(format!("r{round}-s1.tsv.zst"));
        let p2 = dir.path().join(format!("r{round}-s2.tsv.zst"));
        // From round 1 on this init is itself a re-init from Finalized.
        init_op_logger(p1.to_str().unwrap()).unwrap();
        let logger = global_logger().unwrap();
        for i in 0..ROWS {
            assert_eq!(logger.try_log(entry(&bulky("s1", i))), LogOutcome::Accepted);
        }
        drop(logger);

        let finalizer = std::thread::spawn(finalize_op_logger);
        // Finalize has swapped the state as soon as there is no active logger;
        // its writer may still be draining.
        while global_logger().is_some() {
            std::thread::yield_now();
        }
        init_op_logger(p2.to_str().unwrap()).expect("re-init during/after finalize");

        // init returned: the previous session must already be fully drained,
        // although we have not joined the finalizing thread.
        let tags1 = logged_tags(&p1);
        assert_eq!(tags1.len(), ROWS, "round {round}: session 1 incomplete");
        assert!(tags1.iter().all(|t| t.starts_with("s1-")));

        let logger2 = global_logger().expect("session 2 is Active after init");
        for i in 0..10 {
            assert_eq!(
                logger2.try_log(entry(&format!("s2-{i}"))),
                LogOutcome::Accepted
            );
        }
        drop(logger2);
        finalizer.join().unwrap();
        // The finalizer must not have finalized session 2: it was Active until now.
        assert!(
            global_logger().is_some(),
            "round {round}: a finalize begun before the re-init closed session 2"
        );
        finalize_op_logger();
        let tags2 = logged_tags(&p2);
        assert_eq!(tags2.len(), 10, "round {round}: {tags2:?}");
        assert!(tags2.iter().all(|t| t.starts_with("s2-")), "{tags2:?}");
        assert_eq!(logged_tags(&p1).len(), ROWS, "session 1 changed");
    }

    // Many threads re-init at once from Finalized: exactly one file is created
    // (the losers see Active and no-op), and the session works.
    const THREADS: usize = 8;
    let paths: Vec<_> = (0..THREADS)
        .map(|t| dir.path().join(format!("concurrent-{t}.tsv.zst")))
        .collect();
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = paths
        .iter()
        .map(|p| {
            let (p, barrier) = (p.clone(), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                init_op_logger(p.to_str().unwrap()).unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let created: Vec<_> = paths.iter().filter(|p| p.exists()).collect();
    assert_eq!(created.len(), 1, "exactly one init wins: {created:?}");
    assert_eq!(
        global_logger().unwrap().try_log(entry("winner")),
        LogOutcome::Accepted
    );
    finalize_op_logger();
    assert_eq!(logged_tags(created[0]), ["winner"]);
}
