// tests/oplog_lifecycle_stats.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #170 (drop detection).
//
// The op-log is process-global (Uninitialized -> Active -> Finalized), so this
// scenario lives in its OWN integration-test file: every file under tests/ is a
// separate binary and therefore a separate process.
//
// Scenario: `op_log_stats()` is None before init, reports live counts while
// Active (a tiny queue and a flood of entries make the default lossy mode drop,
// and the drops are counted), and keeps reporting the same final counts after
// finalize, where `accepted` equals the rows in the file and calls on a
// captured Logger are counted as `rejected_closed`.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
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

#[test]
fn op_log_stats_before_during_and_after_finalize() {
    // A one-slot queue in the default (lossy) mode, whatever the caller's env.
    #[allow(deprecated)]
    {
        std::env::set_var("S3DLIO_OPLOG_BUF", "1");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");

    assert!(op_log_stats().is_none(), "None before init");

    init_op_logger(path.to_str().unwrap()).unwrap();
    let st = op_log_stats().expect("Some while Active");
    assert_eq!(
        (
            st.accepted,
            st.dropped_full,
            st.rejected_closed,
            st.writer_failed
        ),
        (0, 0, 0, 0)
    );
    assert!(!st.lossless);

    // Flood the one-slot queue from four threads and tally the outcomes.
    let logger = global_logger().expect("active");
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let logger = logger.clone();
            std::thread::spawn(move || {
                let (mut accepted, mut full) = (0_u64, 0_u64);
                for n in 0..20_000 {
                    match logger.try_log(entry(&format!("t{t}-{n}"))) {
                        LogOutcome::Accepted => accepted += 1,
                        LogOutcome::Full => full += 1,
                        other => panic!("unexpected {other:?}"),
                    }
                }
                (accepted, full)
            })
        })
        .collect();
    let (mut accepted, mut full) = (0_u64, 0_u64);
    for h in handles {
        let (a, f) = h.join().unwrap();
        accepted += a;
        full += f;
    }
    assert!(full > 0, "a one-slot queue under four producers must drop");
    assert_eq!(accepted + full, 80_000);

    let during = op_log_stats().expect("Some while Active");
    assert_eq!(during.accepted, accepted);
    assert_eq!(during.dropped_full, full);
    assert_eq!((during.rejected_closed, during.writer_failed), (0, 0));

    finalize_op_logger();
    assert!(global_logger().is_none(), "no active logger after finalize");

    let after = op_log_stats().expect("Some after finalize (stats are kept)");
    assert_eq!(
        (
            after.accepted,
            after.dropped_full,
            after.rejected_closed,
            after.writer_failed
        ),
        (
            during.accepted,
            during.dropped_full,
            during.rejected_closed,
            during.writer_failed
        ),
        "finalize must not change the outcome counts"
    );
    // The writer may still have been draining at `during`; after finalize it has
    // written every accepted row, cleanly. Dropped entries make it incomplete.
    assert_eq!(after.written, after.accepted);
    assert_eq!(after.writer_error, None);
    assert!(!after.is_complete(), "queue-full drops are not complete");

    for _ in 0..5 {
        assert_eq!(logger.try_log(entry("late")), LogOutcome::Closed);
    }
    let late = op_log_stats().expect("still Some");
    assert_eq!(late.rejected_closed, 5);
    assert_eq!(late.accepted, accepted);
    assert_eq!(late.written, accepted);

    let text = zstd::decode_all(std::fs::read(&path).unwrap().as_slice()).unwrap();
    let rows = text.split(|b| *b == b'\n').count() - 2; // header and trailing empty
    assert_eq!(
        rows as u64, late.accepted,
        "accepted == data rows in the file"
    );
}
