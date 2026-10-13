// tests/oplog_lifecycle_shutdown_race_lossless.rs
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

// Scenario (LOSSLESS blocking-send mode, tiny channel so senders block on a
// full queue while finalize runs; finalize must not deadlock): several threads log continuously while finalize runs. Every entry
// whose log() was Accepted must be in the file; nothing else may be (entries
// that were rejected as Closed or dropped as Full are never written); once a
// thread has seen Closed it never sees Accepted or Full again.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

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
fn lossless_accepted_entries_are_exactly_the_written_entries_under_race() {
    const THREADS: usize = 8;
    // Set before the first Logger is built; nothing else is running yet.
    #[allow(deprecated)]
    {
        std::env::set_var("S3DLIO_OPLOG_LOSSLESS", "1");
        std::env::set_var("S3DLIO_OPLOG_BUF", "4");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");
    init_op_logger(path.to_str().unwrap()).unwrap();

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS + 1));
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let logger = global_logger().unwrap();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let deadline = Instant::now() + Duration::from_secs(20);
                let mut accepted = Vec::new();
                let (mut full, mut saw_closed) = (0_usize, false);
                for n in 0_u64.. {
                    let tag = format!("t{t}-{n}");
                    match logger.log(entry(&tag)) {
                        LogOutcome::Accepted => {
                            assert!(!saw_closed, "Accepted after Closed");
                            accepted.push(tag);
                        }
                        LogOutcome::Full => {
                            assert!(!saw_closed, "Full after Closed");
                            full += 1;
                        }
                        LogOutcome::Closed => {
                            saw_closed = true;
                            break;
                        }
                        LogOutcome::WriterFailed => panic!("writer thread failed"),
                    }
                    if Instant::now() > deadline {
                        break;
                    }
                }
                (accepted, full, saw_closed)
            })
        })
        .collect();

    barrier.wait();
    std::thread::sleep(Duration::from_millis(100));
    finalize_op_logger();

    let mut accepted_all = BTreeSet::new();
    let mut accepted_count = 0;
    for h in handles {
        let (accepted, full, saw_closed) = h.join().unwrap();
        assert_eq!(full, 0, "LOSSLESS mode must never report Full");
        assert!(
            saw_closed,
            "a logging thread never observed Closed after finalize"
        );
        accepted_count += accepted.len();
        accepted_all.extend(accepted);
    }
    assert!(accepted_count > 0, "race test logged nothing");
    assert_eq!(accepted_all.len(), accepted_count, "duplicate tags");

    let tags = logged_tags(&path);
    let written: BTreeSet<String> = tags.iter().cloned().collect();
    assert_eq!(written.len(), tags.len(), "an entry was written twice");
    let lost: Vec<_> = accepted_all.difference(&written).take(5).collect();
    assert!(
        lost.is_empty(),
        "accepted entries missing from the file ({} total accepted): {lost:?}",
        accepted_count
    );
    let phantom: Vec<_> = written.difference(&accepted_all).take(5).collect();
    assert!(
        phantom.is_empty(),
        "file holds entries that were not accepted: {phantom:?}"
    );
}
