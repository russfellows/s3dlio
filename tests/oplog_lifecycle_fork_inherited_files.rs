// tests/oplog_lifecycle_fork_inherited_files.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log fork safety (issue #294, round 3, findings S2 and N5): a process created
// by `fork` must never truncate the op-log file of ANY ancestor, however its own
// sessions start and finish.
//
// Round 2 remembered the parent's file only while the child had not started a
// session of its own, so `child: init(c); finalize; init(a)` truncated the
// parent's live log, and a grandchild forgot its grandparent's file. The set of
// inherited files is now process-lifetime and append-only.
//
// Also pinned here: the refusal message is accurate whether the ancestor was
// still writing the file or had already finalized it (N5); a child may still use
// every unrelated file; and the parent's files are byte-identical at the end.
//
// One scenario per file: the op-log is process-global.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
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

fn tags(path: &Path) -> Vec<String> {
    let raw = std::fs::read(path).expect("read op-log");
    let text = zstd::decode_all(raw.as_slice()).expect("complete zstd stream");
    String::from_utf8(text)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split('\t').nth(7).unwrap().to_string())
        .collect()
}

/// Run `body` in a forked child; return its exit status (0 = every check held).
/// A child that hangs is killed by SIGALRM after 30 s and reported.
fn in_child(body: impl FnOnce() -> Result<(), String>) -> i32 {
    // SAFETY: the child runs `body` and `_exit`s.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        unsafe { libc::alarm(30) };
        let code = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
            Ok(Ok(())) => 0,
            Ok(Err(msg)) => {
                eprintln!("CHILD CHECK FAILED: {msg}");
                1
            }
            Err(_) => 2,
        };
        unsafe { libc::_exit(code) };
    }
    let mut status = 0;
    assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
    if libc::WIFSIGNALED(status) {
        return 100 + libc::WTERMSIG(status);
    }
    libc::WEXITSTATUS(status)
}

fn must_refuse(path: &Path, expect: &[&str]) -> Result<(), String> {
    match init_op_logger(path.to_str().unwrap()) {
        Ok(()) => Err(format!("init of {} must be refused", path.display())),
        Err(e) => {
            let m = e.to_string();
            if e.kind() != std::io::ErrorKind::ResourceBusy {
                return Err(format!("wrong kind {:?}: {m}", e.kind()));
            }
            for want in expect {
                if !m.contains(want) {
                    return Err(format!("message lacks {want:?}: {m}"));
                }
            }
            Ok(())
        }
    }
}

#[test]
fn a_forked_process_never_truncates_any_ancestors_op_log() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    let (a, c, d, e, g) = (p("a"), p("c"), p("d"), p("e"), p("g"));

    init_op_logger(a.to_str().unwrap()).unwrap();
    let logger = global_logger().unwrap();
    assert_eq!(logger.try_log(entry("parent-1")), LogOutcome::Accepted);

    // --- Child of an ACTIVE parent: the S2 repro, and the grandchild. --------
    let (a1, c1, d1, e1) = (a.clone(), c.clone(), d.clone(), e.clone());
    let status = in_child(move || {
        if op_log_stats().is_some() {
            return Err("stats must be None in the child".into());
        }
        finalize_op_logger(); // no-op, must not wait for a writer
        must_refuse(&a1, &["PARENT", "still writing"])?;
        init_op_logger(c1.to_str().unwrap()).map_err(|e| e.to_string())?;
        finalize_op_logger();
        // S2: the child's own session came and went; the parent's file stays protected.
        must_refuse(&a1, &["PARENT", "still writing"])?;
        // The child's own file is fair game again (it is not inherited).
        init_op_logger(c1.to_str().unwrap()).map_err(|e| e.to_string())?;
        // Grandchild of the child (child Active on c): a AND c are inherited.
        let (a2, c2, d2, e2) = (a1.clone(), c1.clone(), d1.clone(), e1.clone());
        let st = in_child(move || {
            must_refuse(&a2, &["PARENT"])?; // the grandparent's file
            must_refuse(&c2, &["PARENT"])?; // the parent's live file
            init_op_logger(d2.to_str().unwrap()).map_err(|e| e.to_string())?; // unrelated: fine
            finalize_op_logger();
            must_refuse(&a2, &["PARENT"])?;
            must_refuse(&c2, &["PARENT"])?;
            init_op_logger(e2.to_str().unwrap()).map_err(|e| e.to_string())?;
            finalize_op_logger();
            Ok(())
        });
        if st != 0 {
            return Err(format!("grandchild exited {st}"));
        }
        // Back in the child: its own session on c continues, so any other file
        // (the parent's a included) is the ordinary Active error.
        must_refuse(&a1, &["already active"])?;
        match init_op_logger(d1.to_str().unwrap()) {
            Ok(()) => return Err("another file while the child is Active must fail".into()),
            Err(e) if e.to_string().contains("already active") => {}
            Err(e) => return Err(format!("expected the ordinary Active error: {e}")),
        }
        finalize_op_logger();
        // Finalized again: now it is the inherited-file refusal, not the Active one.
        must_refuse(&a1, &["PARENT", "still writing"])?;
        Ok(())
    });
    assert_eq!(
        status, 0,
        "child scenario failed (see CHILD CHECK FAILED above)"
    );

    // The parent's session was never disturbed by any of that.
    assert_eq!(logger.try_log(entry("parent-2")), LogOutcome::Accepted);
    assert_eq!(op_log_stats().unwrap().accepted, 2);
    finalize_op_logger();
    assert_eq!(tags(&a), vec!["parent-1", "parent-2"]);

    // --- Child of a FINALIZED parent (N5): refused, with an accurate message. -
    let before = std::fs::read(&a).unwrap();
    let a3 = a.clone();
    let g3 = g.clone();
    let status = in_child(move || {
        if op_log_stats().is_some() {
            return Err("stats must be None in the child of a finalized parent".into());
        }
        // The parent had FINISHED this log: it must not be described as still being written.
        match init_op_logger(a3.to_str().unwrap()) {
            Ok(()) => return Err("must be refused".into()),
            Err(e) => {
                let m = e.to_string();
                if m.contains("still writing") || !m.contains("PARENT") || !m.contains("finished") {
                    return Err(format!("inaccurate message for a finalized parent: {m}"));
                }
            }
        }
        init_op_logger(g3.to_str().unwrap()).map_err(|e| e.to_string())?;
        finalize_op_logger();
        must_refuse(&a3, &["PARENT", "finished"])?;
        Ok(())
    });
    assert_eq!(status, 0, "finalized-parent scenario failed");
    assert_eq!(
        std::fs::read(&a).unwrap(),
        before,
        "the parent's finished log must be byte-identical"
    );
    assert_eq!(tags(&g), Vec::<String>::new());
    assert_eq!(tags(&c), Vec::<String>::new());
    assert_eq!(tags(&e), Vec::<String>::new());
    assert!(!d.exists() || tags(&d).is_empty());
}
