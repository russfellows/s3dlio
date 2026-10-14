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
// Active (a tiny queue and a stalled writer make the default lossy mode drop,
// and the drops are counted), and keeps reporting the same final counts after
// finalize, where `accepted` equals the rows in the file and calls on a
// captured Logger are counted as `rejected_closed`.
//
// The stall is deterministic, not a scheduling race: the op-log "file" is a
// FIFO whose reader does not read until told to. The first (400 KB, barely
// compressible) entry produces far more output than the pipe holds, so the writer
// thread blocks inside `write(2)` and cannot take another entry off the one-slot
// queue; after at most one more entry is queued, every further call is `Full`.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
use std::io::Read;
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

/// ~400 KB of hex digits in the `file` column: far more compressed output than
/// a pipe holds.
fn big_entry(seed: u64) -> LogEntry {
    let mut x = seed | 1;
    let mut file = String::with_capacity(400_000);
    while file.len() < 400_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        file.push_str(&format!("{x:016x}"));
    }
    entry(&file)
}

fn make_fifo(path: &std::path::Path) {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: `c_path` is a valid NUL-terminated path that outlives the call.
    let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
    assert_eq!(rc, 0, "mkfifo: {}", std::io::Error::last_os_error());
}

#[test]
fn op_log_stats_before_during_and_after_finalize() {
    // A one-slot queue in the default (lossy) mode, whatever the caller's env, and
    // a tiny file buffer so the writer's output reaches the pipe at once.
    #[allow(deprecated)]
    {
        std::env::set_var("S3DLIO_OPLOG_BUF", "1");
        std::env::set_var("S3DLIO_OPLOG_WBUFCAP", "4096");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.fifo");
    make_fifo(&path);

    // The reader holds the read end open (so creating the "file" succeeds) but
    // reads nothing until told to, or until the test dies.
    let (start_reading, wait_to_read) = std::sync::mpsc::channel::<()>();
    let fifo = path.clone();
    let reader = std::thread::spawn(move || {
        let mut read_end = std::fs::File::open(fifo).unwrap();
        let _ = wait_to_read.recv();
        let mut bytes = Vec::new();
        read_end.read_to_end(&mut bytes).unwrap();
        bytes
    });

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

    // Stall the writer: it takes this entry and blocks writing it to the pipe.
    let logger = global_logger().expect("active");
    assert_eq!(logger.try_log(big_entry(1)), LogOutcome::Accepted);

    // Flood the one-slot queue from four threads and tally the outcomes.
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
    let (mut accepted, mut full) = (1_u64, 0_u64); // the big entry counts
    for h in handles {
        let (a, f) = h.join().unwrap();
        accepted += a;
        full += f;
    }
    // The writer is stalled and the queue holds one entry, so at most one more
    // was accepted and every other call was dropped.
    assert!(
        accepted <= 2,
        "a stalled writer accepted {accepted} entries"
    );
    assert!(full >= 79_999, "only {full} of 80000 calls were dropped");
    assert_eq!(accepted + full, 80_001);

    let during = op_log_stats().expect("Some while Active");
    assert_eq!(during.accepted, accepted);
    assert_eq!(during.dropped_full, full);
    assert_eq!((during.rejected_closed, during.writer_failed), (0, 0));

    // Let the writer's output drain, then finalize.
    start_reading.send(()).unwrap();
    finalize_op_logger();
    let written_bytes = reader.join().unwrap();
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

    let text = zstd::decode_all(written_bytes.as_slice()).unwrap();
    let rows = text.split(|b| *b == b'\n').count() - 2; // header and trailing empty
    assert_eq!(
        rows as u64, late.accepted,
        "accepted == data rows in the file"
    );
}
