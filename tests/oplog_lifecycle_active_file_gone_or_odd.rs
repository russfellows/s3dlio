// tests/oplog_lifecycle_active_file_gone_or_odd.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle (issue #294, round 3, finding N7): `init_op_logger` while a
// session is Active is an error for any file but the one being written, and the
// message must be ACCURATE about why:
//
//  * the active file was renamed, deleted or replaced after the session started:
//    "already active and writing '<path>'" read as a contradiction, because the
//    path no longer names the file the session writes. The message now says the
//    session writes a file that was renamed or deleted (and, on Linux, what the
//    system now calls it). Identity stays authoritative: the session keeps writing
//    the original inode, so a request for the old path is still an ERROR and is
//    never created.
//  * the request is a directory, a device (/dev/null), a FIFO or a dangling
//    symlink: the message says what the request is. Nothing is created.
//
// One scenario per file: the op-log is process-global. Unix only.
#![cfg(unix)]

use s3dlio::{finalize_op_logger, init_op_logger, op_log_stats};
use std::path::Path;

fn refusal(path: &Path) -> String {
    let e = init_op_logger(path.to_str().unwrap()).expect_err("must be refused");
    assert_eq!(e.kind(), std::io::ErrorKind::ResourceBusy, "{e}");
    e.to_string()
}

#[test]
fn active_session_reports_gone_active_files_and_odd_requests_accurately() {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    let a = p("a.tsv.zst");
    init_op_logger(a.to_str().unwrap()).unwrap();

    // Normal different file: the ordinary message, no "gone" wording.
    let m = refusal(&p("b.tsv.zst"));
    assert!(m.contains("already active and writing"), "{m}");
    assert!(!m.contains("renamed or deleted"), "{m}");

    // --- Renamed -------------------------------------------------------------
    let a1 = p("a.tsv.zst.1");
    std::fs::rename(&a, &a1).unwrap();
    let m = refusal(&a);
    assert!(
        m.contains("renamed or deleted"),
        "must say it was renamed: {m}"
    );
    assert!(
        m.contains(a1.to_str().unwrap()),
        "must show what the system now calls the open file (Linux /proc/self/fd): {m}"
    );
    assert!(
        !a.exists(),
        "the old path must not be re-created: it is an error"
    );
    // A third spelling of the old name is still an error too.
    let _ = refusal(&p("./a.tsv.zst"));
    assert!(!a.exists());

    // --- A different file now sits at the old path -----------------------------
    std::fs::write(&a, b"somebody else's file").unwrap();
    let m = refusal(&a);
    assert!(
        m.contains("renamed or deleted") && m.contains("different file"),
        "a different file now has the name: {m}"
    );
    assert_eq!(
        std::fs::read(&a).unwrap(),
        b"somebody else's file",
        "not truncated"
    );

    // --- Deleted -----------------------------------------------------------------
    std::fs::remove_file(&a).unwrap();
    std::fs::remove_file(&a1).unwrap();
    let m = refusal(&a);
    assert!(
        m.contains("renamed or deleted"),
        "must say it was deleted: {m}"
    );
    assert!(
        m.contains("(deleted)"),
        "the kernel's name for the unlinked file: {m}"
    );
    assert!(!a.exists());
    // The session itself is fine and still the one session.
    assert!(op_log_stats().is_some());

    // --- Odd requests: each is an error, each is described, nothing created ----
    let sub = p("a_directory");
    std::fs::create_dir(&sub).unwrap();
    let m = refusal(&sub);
    assert!(m.contains("is a directory"), "{m}");
    assert!(std::fs::read_dir(&sub).unwrap().next().is_none(), "{m}");

    let m = refusal(Path::new("/dev/null"));
    assert!(m.contains("character device"), "{m}");
    assert!(Path::new("/dev/null").exists());

    let fifo = p("a_fifo");
    let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    let m = refusal(&fifo);
    assert!(m.contains("named pipe"), "{m}");

    let dangling = p("dangling");
    std::os::unix::fs::symlink(p("no_such_target"), &dangling).unwrap();
    let m = refusal(&dangling);
    assert!(
        m.contains("symbolic link") && m.contains("does not exist"),
        "{m}"
    );
    assert!(
        !p("no_such_target").exists(),
        "the symlink target must not be created"
    );

    // Every message still names the request and says to finalize first.
    assert!(
        m.contains(dangling.to_str().unwrap()) && m.contains("finalize"),
        "{m}"
    );

    finalize_op_logger();
}
