// tests/oplog_lifecycle_fork_from_tls_destructor.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log fork safety (issue #294, round 3, finding N8b): a `fork` issued from a
// thread-local DESTRUCTOR runs s3dlio's atfork handlers after the handlers'
// own thread-local (the held lifecycle guard) has been destroyed. They must not
// panic: a panic cannot unwind out of an `extern "C"` handler, so it would ABORT
// the whole process. They now use `LocalKey::try_with` and simply do nothing.
//
// Ordering: thread-locals are destroyed in REVERSE order of first use. The
// destructor thread-local below is touched first, then a fork (which touches the
// handlers' thread-local) runs, so at thread exit the handlers' one is destroyed
// first and ours (whose destructor forks) second.
//
// Mutation record: with `HELD.with(..)` instead of `try_with` in the handlers,
// the process aborts (SIGABRT) and this test fails.
//
// Unix only; one scenario per file.
#![cfg(unix)]

use s3dlio::init_op_logger;
use std::cell::Cell;
use std::sync::atomic::{AtomicI32, Ordering};

static CHILD_STATUS: AtomicI32 = AtomicI32::new(-1);

struct ForksOnDrop(Cell<bool>);

impl Drop for ForksOnDrop {
    fn drop(&mut self) {
        // SAFETY: the child only `_exit`s.
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            unsafe { libc::_exit(0) };
        }
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        CHILD_STATUS.store(
            if libc::WIFEXITED(status) {
                libc::WEXITSTATUS(status)
            } else {
                1000 + libc::WTERMSIG(status)
            },
            Ordering::SeqCst,
        );
    }
}

thread_local! {
    static FORKS_ON_DROP: ForksOnDrop = const { ForksOnDrop(Cell::new(false)) };
}

#[test]
fn fork_from_a_thread_local_destructor_does_not_abort() {
    let dir = tempfile::tempdir().unwrap();
    init_op_logger(dir.path().join("a.tsv.zst").to_str().unwrap()).unwrap();

    std::thread::spawn(|| {
        FORKS_ON_DROP.with(|f| f.0.set(true)); // registers our destructor FIRST
                                               // An ordinary fork now touches the handlers' thread-local, registering
                                               // ITS destructor second (so it is destroyed first at thread exit).
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            unsafe { libc::_exit(0) };
        }
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
    })
    .join()
    .expect("the thread (and its destructor's fork) must not panic or abort");
    assert_eq!(
        CHILD_STATUS.load(Ordering::SeqCst),
        0,
        "the fork in the destructor must have happened and its child exited cleanly"
    );
}
