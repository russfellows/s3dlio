// tests/oplog_lifecycle_fork_inode_reuse.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log fork safety (issue #294, round 4, B1): a FINISHED ancestor log is not
// protected, so a deleted finished log whose inode number the filesystem reuses
// can never make a child's request for a NEW path fail.
//
// Round 3 compared the (device, inode) of the file a child opened with the files
// of every ancestor session, finalized ones included. A finalized session's
// descriptor is closed, so after `rm a` the inode number is reused at once on
// ext4: `parent: init a; finalize; rm a; fork; child: init NEW b0` was refused
// with ResourceBusy, the message said "nothing was created", and b0 was left
// behind as an empty file.
//
// Also pinned: a refusal (a file an ancestor is still WRITING) leaves that file
// byte-identical, and a path that did not exist is never refused (a file created
// by this very request cannot be an ancestor's live file) and is a valid log.
//
// The scratch directory is under the crate's `target/` (a real filesystem; /tmp
// can be tmpfs, which never reuses inode numbers).
//
// One scenario per file: the op-log is process-global.

use s3dlio::{finalize_op_logger, init_op_logger};
use std::os::unix::fs::MetadataExt;

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

fn is_complete_empty_log(path: &std::path::Path) -> bool {
    let Ok(raw) = std::fs::read(path) else {
        return false;
    };
    match zstd::decode_all(raw.as_slice()) {
        Ok(text) => {
            text.starts_with(b"idx\tthread\t") && text.iter().filter(|&&b| b == b'\n').count() == 1
        }
        Err(_) => false,
    }
}

#[test]
fn a_reused_inode_number_of_a_deleted_finished_log_never_refuses_a_new_path() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/target");
    std::fs::create_dir_all(base).unwrap();
    let dir = tempfile::Builder::new()
        .prefix("oplog-inode-reuse-")
        .tempdir_in(base)
        .unwrap();
    let p = |n: &str| dir.path().join(n);
    let (a, b0, live, new_in_active_child) = (p("a"), p("b0"), p("live"), p("fresh"));

    // The parent finishes a log, then deletes it: its inode number is free.
    init_op_logger(a.to_str().unwrap()).unwrap();
    finalize_op_logger();
    let a_ino = std::fs::metadata(&a).unwrap().ino();
    std::fs::remove_file(&a).unwrap();

    let (b0c, a_c) = (b0.clone(), a.clone());
    let status = in_child(move || {
        init_op_logger(b0c.to_str().unwrap()).map_err(|e| format!("new path refused: {e}"))?;
        let reused = std::fs::metadata(&b0c).map_err(|e| e.to_string())?.ino() == a_ino;
        eprintln!("inode number reused by the new file: {reused}");
        finalize_op_logger();
        if !is_complete_empty_log(&b0c) {
            return Err("b0 is not a valid op-log".into());
        }
        // The deleted finished log's own old path is fine too.
        init_op_logger(a_c.to_str().unwrap()).map_err(|e| format!("old path refused: {e}"))?;
        finalize_op_logger();
        Ok(())
    });
    assert_eq!(status, 0, "child scenario failed (see CHILD CHECK FAILED)");
    assert!(is_complete_empty_log(&b0));

    // A file the parent is still WRITING stays protected; a refusal leaves it
    // byte-identical and creates nothing; a brand-new path is never refused.
    init_op_logger(live.to_str().unwrap()).unwrap();
    let before = std::fs::read(&live).unwrap();
    let (livec, freshc) = (live.clone(), new_in_active_child.clone());
    let status = in_child(move || {
        match init_op_logger(livec.to_str().unwrap()) {
            Ok(()) => return Err("the parent's live file must be refused".into()),
            Err(e) if e.kind() == std::io::ErrorKind::ResourceBusy => {}
            Err(e) => return Err(format!("wrong error: {e}")),
        }
        init_op_logger(freshc.to_str().unwrap()).map_err(|e| format!("fresh refused: {e}"))?;
        finalize_op_logger();
        if !is_complete_empty_log(&freshc) {
            return Err("fresh is not a valid op-log".into());
        }
        Ok(())
    });
    assert_eq!(status, 0, "live-file scenario failed");
    assert_eq!(std::fs::read(&live).unwrap(), before, "live file untouched");
    finalize_op_logger();
}
