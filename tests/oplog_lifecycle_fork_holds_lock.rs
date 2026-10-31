// tests/oplog_lifecycle_fork_holds_lock.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log fork safety (issue #294, round 3, finding S4): s3dlio's `pthread_atfork`
// PREPARE handler must hold the lifecycle lock (and the client-id lock) for the
// whole duration of `fork`, so no other thread owns either at the instant of the
// fork (the child would inherit a locked mutex whose owner does not exist).
//
// How the test observes that. glibc runs PREPARE handlers in REVERSE order of
// registration (last registered first), and parent/child handlers in
// registration order. s3dlio registers its handlers on the first
// `init_op_logger`. This test registers its OWN handler BEFORE that, so its
// prepare runs AFTER s3dlio's prepare, i.e. while s3dlio's prepare already holds
// the locks. The test's prepare lets two probe threads call
// `op_log_stats()` and `set_client_id()` and then sleeps 500 ms; if the locks are
// held, both probes are still blocked when it wakes. After `fork` returns in the
// parent, both must complete. (If glibc ordered handlers the other way, the
// probes would finish inside the 500 ms and the test would fail, so it validates
// its own premise.)
//
// Mutation record: fails when prepare does nothing, when it locks and drops at
// once, and when `fork_safety::register` is removed (probes finish in prepare).
// The client-id half fails when only the lifecycle lock is held.

use s3dlio::{init_op_logger, op_log_stats, set_client_id};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static GO: AtomicBool = AtomicBool::new(false);
static STATS_DONE: AtomicBool = AtomicBool::new(false);
static CLIENT_DONE: AtomicBool = AtomicBool::new(false);
static STATS_DONE_IN_PREPARE: AtomicBool = AtomicBool::new(true);
static CLIENT_DONE_IN_PREPARE: AtomicBool = AtomicBool::new(true);

extern "C" fn test_prepare() {
    GO.store(true, Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(500));
    STATS_DONE_IN_PREPARE.store(STATS_DONE.load(Ordering::SeqCst), Ordering::SeqCst);
    CLIENT_DONE_IN_PREPARE.store(CLIENT_DONE.load(Ordering::SeqCst), Ordering::SeqCst);
}

fn wait_for(flag: &AtomicBool, what: &str) {
    let end = Instant::now() + Duration::from_secs(10);
    while !flag.load(Ordering::SeqCst) {
        assert!(
            Instant::now() < end,
            "{what} still blocked after fork returned"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn prepare_holds_the_lifecycle_and_client_id_locks_across_fork() {
    // 1. Register the test's handler FIRST (so it runs after s3dlio's prepare).
    // SAFETY: a plain `extern "C" fn()` that lives for the whole process.
    let rc = unsafe { libc::pthread_atfork(Some(test_prepare), None, None) };
    assert_eq!(rc, 0);
    // 2. s3dlio registers its handlers on the first init; create the client-id lock.
    let dir = tempfile::tempdir().unwrap();
    init_op_logger(dir.path().join("a.tsv.zst").to_str().unwrap()).unwrap();
    set_client_id("before").unwrap();

    let probes = [
        std::thread::spawn(|| {
            while !GO.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            let _ = op_log_stats();
            STATS_DONE.store(true, Ordering::SeqCst);
        }),
        std::thread::spawn(|| {
            while !GO.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            set_client_id("probe").unwrap();
            CLIENT_DONE.store(true, Ordering::SeqCst);
        }),
    ];

    // SAFETY: the child does nothing but `_exit`.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);
    if pid == 0 {
        unsafe { libc::_exit(0) };
    }
    let mut status = 0;
    assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);

    assert!(
        !STATS_DONE_IN_PREPARE.load(Ordering::SeqCst),
        "op_log_stats() completed while fork was preparing: the lifecycle lock was NOT held"
    );
    assert!(
        !CLIENT_DONE_IN_PREPARE.load(Ordering::SeqCst),
        "set_client_id() completed while fork was preparing: the client-id lock was NOT held"
    );
    wait_for(&STATS_DONE, "op_log_stats()");
    wait_for(&CLIENT_DONE, "set_client_id()");
    for p in probes {
        p.join().unwrap();
    }
}
