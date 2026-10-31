// tests/fork_oplog_idle_297.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! #297: `global_logger()` locks the op-log lifecycle state on EVERY store lookup. Without the
//! fork handlers (installed by `init_op_logger` only, before this fix) a parent thread inside
//! `global_logger()` at the instant of a fork left that lock held in the child, and the first
//! s3dlio call of the child waited for it for ever, in a process that never started an op-log
//! session. Own test binary, like the other fork tests.
#![cfg(unix)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn child_finishes(deadline_ms: u64) -> bool {
    // SAFETY: the child runs a few lock/unlock pairs and `_exit`s.
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        for _ in 0..50 {
            let _ = s3dlio::s3_logger::global_logger();
        }
        unsafe { libc::_exit(0) };
    }
    let mut status = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(deadline_ms);
    loop {
        if unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) } == pid {
            return libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
        }
        if std::time::Instant::now() > deadline {
            unsafe { libc::kill(pid, libc::SIGKILL) };
            unsafe { libc::waitpid(pid, &mut status, 0) };
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn a_child_forked_while_parent_threads_call_global_logger_does_not_hang() {
    let stop = Arc::new(AtomicBool::new(false));
    let hammers: Vec<_> = (0..4)
        .map(|_| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _ = s3dlio::s3_logger::global_logger();
                }
            })
        })
        .collect();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut hung = 0;
    for _ in 0..300 {
        if !child_finishes(2000) {
            hung += 1;
            break;
        }
    }
    stop.store(true, Ordering::Relaxed);
    for h in hammers {
        let _ = h.join();
    }
    assert_eq!(
        hung, 0,
        "a fork child waited for the op-log state lock of its parent"
    );
}
