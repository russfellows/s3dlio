// tests/oplog_lifecycle_log_returns_unit.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Public-API compatibility test for the op-log (issue #170 follow-up):
// `Logger::log` returned `()` before the lifecycle work, and downstream code
// may rely on that (for example a closure passed to `for_each`, or a
// `fn(&Logger, LogEntry)` pointer). The outcome is available from the separate
// `Logger::try_log`.
//
// This is mostly a COMPILE-LEVEL test: it fails to build if `log` ever returns
// anything but `()`. Own process (the op-log is process-global).

use s3dlio::s3_logger::{LogEntry, LogOutcome, Logger};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
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
fn log_returns_unit_and_try_log_returns_the_outcome() {
    // Do not depend on the caller's environment.
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");
    init_op_logger(path.to_str().unwrap()).unwrap();
    let logger = global_logger().expect("active after init");

    // Type-level: `log` is exactly `fn(&Logger, LogEntry)`, returning `()`.
    let as_fn_pointer: fn(&Logger, LogEntry) = Logger::log;
    as_fn_pointer(&logger, entry("via-fn-pointer"));

    // A unit-returning closure body (the pre-lifecycle API allowed this).
    (0..3)
        .map(|i| entry(&format!("via-for-each-{i}")))
        .for_each(|e| logger.log(e));

    // The outcome is available from try_log.
    let outcome: LogOutcome = logger.try_log(entry("via-try-log"));
    assert_eq!(outcome, LogOutcome::Accepted);

    finalize_op_logger();

    // `log` and `try_log` after finalize: log() is silent, try_log() says why.
    logger.log(entry("late"));
    assert_eq!(logger.try_log(entry("late")), LogOutcome::Closed);

    let text = zstd::decode_all(std::fs::read(&path).unwrap().as_slice()).unwrap();
    let text = String::from_utf8(text).unwrap();
    let tags: Vec<&str> = text
        .lines()
        .skip(1)
        .map(|l| l.split('\t').nth(7).unwrap())
        .collect();
    assert_eq!(
        tags,
        [
            "via-fn-pointer",
            "via-for-each-0",
            "via-for-each-1",
            "via-for-each-2",
            "via-try-log"
        ]
    );
}
