// tests/oplog_lifecycle_fork_child_logger_closed.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log fork safety (issue #294, round 3, finding S3): the fork CHILD handler
// must close the inherited logger's gate.
//
// A `Logger` captured before `fork` (a store keeps one) is a handle to a writer
// thread that exists only in the parent. In the child, with a queue of ONE entry
// and lossless mode, the second `try_log` would block forever on the full queue.
// The child handler closes the gate of the inherited session, so every clone
// rejects at once (`LogOutcome::Closed`) and nothing blocks.
//
// Mutation record: with the gate-close removed from the child handler, or with
// the handler registration (`fork_safety::register`) removed, the child blocks
// on its second `try_log` and the 20 s alarm kills it (this test then fails).
//
// One scenario per file: the op-log is process-global.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
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
fn inherited_logger_clone_rejects_everything_in_the_child_without_blocking() {
    // A queue of one entry and lossless mode: the child blocks on its second
    // `try_log` unless the gate is closed.
    #[allow(deprecated)]
    {
        std::env::set_var("S3DLIO_OPLOG_BUF", "1");
        std::env::set_var("S3DLIO_OPLOG_LOSSLESS", "1");
    }
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("parent.tsv.zst");
    init_op_logger(log.to_str().unwrap()).unwrap();
    let logger = global_logger().expect("Active");

    // SAFETY: the child only calls `try_log` and `_exit`; the parent does not
    // log during the fork.
    // The child's stderr goes to a file, to check the one-line note it prints.
    let child_stderr = dir.path().join("child_stderr.txt");
    let sink = std::fs::File::create(&child_stderr).unwrap();
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        unsafe {
            libc::dup2(std::os::unix::io::AsRawFd::as_raw_fd(&sink), 2);
            libc::alarm(20);
        }
        let closed = (0..1000)
            .filter(|_| logger.try_log(entry("child")) == LogOutcome::Closed)
            .count();
        unsafe { libc::_exit(if closed == 1000 { 0 } else { 3 }) };
    }

    let mut status = 0;
    let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
    assert_eq!(waited, pid);
    assert!(
        !libc::WIFSIGNALED(status),
        "the child was killed by signal {} (SIGALRM = it HUNG on the inherited logger)",
        libc::WTERMSIG(status)
    );
    assert!(libc::WIFEXITED(status));
    assert_eq!(
        libc::WEXITSTATUS(status),
        0,
        "every try_log in the child must be LogOutcome::Closed"
    );

    // (d) The child's note says the process was FORKED, not that the op-log was
    // finalized (nobody finalized anything in the child).
    let note = std::fs::read_to_string(&child_stderr).unwrap();
    assert!(note.contains("was forked"), "child note: {note:?}");
    assert!(!note.contains("op-log finalized"), "child note: {note:?}");
    assert_eq!(
        note.lines().count(),
        1,
        "the note is printed once: {note:?}"
    );

    // The parent's session is unaffected: its clone still logs and the file is whole.
    assert_eq!(logger.try_log(entry("parent")), LogOutcome::Accepted);
    finalize_op_logger();
    let text = zstd::decode_all(std::fs::read(&log).unwrap().as_slice())
        .expect("the parent's log must be a complete zstd stream");
    let text = String::from_utf8(text).unwrap();
    let rows: Vec<&str> = text.lines().skip(1).collect();
    assert_eq!(rows.len(), 1, "only the parent's entry: {rows:?}");
    assert!(rows[0].contains("\tparent\t"));
}
