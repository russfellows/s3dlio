// tests/oplog_lifecycle_spawn_failure.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log test for s3dlio issue #170: when the OS refuses to start the writer
// thread, init_op_logger must return an error instead of panicking, and the
// op-log must stay Uninitialized.
//
// The refusal is real, not simulated: this process lowers its OWN
// RLIMIT_NPROC to 1, so every further thread/process creation by this process
// fails with EAGAIN (the limit is per process; other processes are unaffected).
// The limit does not apply to root / CAP_SYS_RESOURCE; the test then FAILS
// loudly at its precondition rather than passing vacuously. Own process.

use s3dlio::{global_logger, init_op_logger};

#[test]
fn init_returns_an_error_when_the_os_refuses_a_thread() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ops.tsv.zst");

    let limit = libc::rlimit {
        rlim_cur: 1,
        rlim_max: 1,
    };
    // SAFETY: plain syscall with a valid pointer to a local struct.
    let rc = unsafe { libc::setrlimit(libc::RLIMIT_NPROC, &limit) };
    assert_eq!(rc, 0, "setrlimit(RLIMIT_NPROC) failed");

    // Precondition: thread creation really is refused now.
    let probe = std::thread::Builder::new().spawn(|| {});
    assert!(
        probe.is_err(),
        "cannot exercise the failure: this process can still start threads \
         (running as root / with CAP_SYS_RESOURCE?)"
    );

    let err = init_op_logger(path.to_str().unwrap())
        .expect_err("init must fail when the writer thread cannot start");
    assert!(
        err.to_string().contains("writer thread"),
        "error should say the writer thread could not start: {err}"
    );
    assert!(
        global_logger().is_none(),
        "a failed init must leave the op-log Uninitialized"
    );
}
