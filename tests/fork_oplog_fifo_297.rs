// tests/fork_oplog_fifo_297.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! #297: a thread blocked opening the op-log file (a FIFO without a reader here; a dead NFS
//! server in real life) must not stall `fork()` or any other op-log call. Before, the lifecycle
//! lock was held across the open, and the `pthread_atfork` prepare handler (registered by
//! `global_logger()` at the first store lookup) waited for it: every fork of the process hung.
//! Own test binary, like the other fork tests.
#![cfg(unix)]

use std::sync::mpsc;
use std::time::Duration;

fn fork_child_that_uses_the_oplog(other: &str) -> i32 {
    // SAFETY: the child calls the op-log functions and `_exit`s.
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        let ok = s3dlio::s3_logger::global_logger().is_none()
            && s3dlio::s3_logger::init_op_logger(other).is_ok();
        s3dlio::s3_logger::finalize_op_logger();
        unsafe { libc::_exit(if ok { 0 } else { 3 }) };
    }
    let mut status = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        -1
    }
}

#[test]
fn a_blocked_init_does_not_stall_fork_or_other_oplog_calls() {
    let dir = std::env::temp_dir().join(format!("fork_fifo_297_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fifo = dir.join("log.fifo");
    let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    let other = dir.join("child.tsv.zst");
    let other = other.to_str().unwrap().to_string();

    // Registers the fork handlers the way a store lookup does.
    let _ = s3dlio::s3_logger::global_logger();

    // This thread blocks in open(2) of the FIFO (no reader).
    let fifo_s = fifo.to_str().unwrap().to_string();
    let starter = std::thread::spawn(move || s3dlio::s3_logger::init_op_logger(&fifo_s));
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !starter.is_finished(),
        "the init must be blocked opening the FIFO"
    );

    // Everything else proceeds: the state calls and a fork whose child uses the op-log.
    let (tx, rx) = mpsc::channel();
    let other2 = other.clone();
    let forker = std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let _ = s3dlio::s3_logger::global_logger();
        let _ = s3dlio::s3_logger::op_log_stats();
        let status = fork_child_that_uses_the_oplog(&other2);
        let _ = tx.send((status, t0.elapsed()));
    });
    let got = rx.recv_timeout(Duration::from_secs(10));

    // Release the blocked open (a reader), whatever happened above.
    let _reader = std::fs::File::open(&fifo);
    let _ = starter.join();
    let _ = forker.join();
    let (status, took) = got.expect("fork() or an op-log call waited for the blocked init");
    assert_eq!(
        status, 0,
        "the child must finish its op-log calls (status {status})"
    );
    assert!(took < Duration::from_secs(5), "took {took:?}");
    s3dlio::s3_logger::finalize_op_logger();
    let _ = std::fs::remove_dir_all(&dir);
}
