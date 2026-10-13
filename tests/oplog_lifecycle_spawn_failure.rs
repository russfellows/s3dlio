// tests/oplog_lifecycle_spawn_failure.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log test for s3dlio issue #170: when the OS refuses to start the writer
// thread, init_op_logger must return an error instead of panicking, and the
// op-log must stay Uninitialized.
//
// The refusal is real, not simulated. This process lowers its OWN RLIMIT_NPROC
// to 1. The kernel does not compare that value with the number of threads of
// this process: RLIMIT_NPROC counts tasks (threads and processes) per REAL USER
// ID, system-wide, and a clone() by a task whose limit value is exceeded by its
// user's task count fails with EAGAIN. This process is itself a task of its user,
// so the count is always >= 1 and every further thread creation fails. Other
// processes keep their own (unchanged) limit values and are unaffected.
//
// PRECONDITION: the limit is not enforced for root or a process holding
// CAP_SYS_RESOURCE / CAP_SYS_ADMIN. There the scenario cannot be produced. The
// test then joins its probe thread, prints
//   SPAWN-FAILURE SCENARIO NOT EXERCISED: <reason>
// and returns, unless S3DLIO_REQUIRE_SPAWN_FAILURE_TEST=1 is set, in which case
// it FAILS, so CI (which runs as an unprivileged user) cannot silently skip it.
// On an unprivileged run the precondition (the probe thread fails to start) is
// asserted before init's behaviour is. Own process (the limit is process-wide
// and so is the op-log).

use s3dlio::{global_logger, init_op_logger};

const REQUIRE_ENV: &str = "S3DLIO_REQUIRE_SPAWN_FAILURE_TEST";

/// What to do with the result of the probe thread.
#[derive(Debug, PartialEq, Eq)]
enum Probe {
    /// The thread could not start: the scenario is real; assert init's behaviour.
    Exercise,
    /// The thread started and the environment does not require the scenario.
    NotExercised,
    /// The thread started and the environment requires the scenario: fail.
    FailBecauseRequired,
}

fn decide(probe_thread_started: bool, required: bool) -> Probe {
    match (probe_thread_started, required) {
        (false, _) => Probe::Exercise,
        (true, false) => Probe::NotExercised,
        (true, true) => Probe::FailBecauseRequired,
    }
}

#[test]
fn init_returns_an_error_when_the_os_refuses_a_thread() {
    // The decision table, checked before the limit is lowered (afterwards this
    // process cannot start threads).
    assert_eq!(decide(false, false), Probe::Exercise);
    assert_eq!(decide(false, true), Probe::Exercise);
    assert_eq!(decide(true, false), Probe::NotExercised);
    assert_eq!(decide(true, true), Probe::FailBecauseRequired);

    let required = std::env::var(REQUIRE_ENV).as_deref() == Ok("1");
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
    let probe_thread_started = probe.is_ok();
    if let Ok(handle) = probe {
        handle.join().expect("probe thread panicked");
    }
    match decide(probe_thread_started, required) {
        Probe::Exercise => {}
        Probe::NotExercised => {
            println!(
                "SPAWN-FAILURE SCENARIO NOT EXERCISED: with RLIMIT_NPROC=1 this process can \
                 still start threads (running as root or with CAP_SYS_RESOURCE / CAP_SYS_ADMIN, \
                 which are exempt from the limit); {REQUIRE_ENV}=1 makes this a failure"
            );
            return;
        }
        Probe::FailBecauseRequired => {
            println!(
                "SPAWN-FAILURE SCENARIO NOT EXERCISED: with RLIMIT_NPROC=1 this process can \
                 still start threads (running as root or with CAP_SYS_RESOURCE / CAP_SYS_ADMIN)"
            );
            panic!(
                "{REQUIRE_ENV}=1 but the spawn-failure scenario could not be produced: run \
                 this test as an unprivileged user"
            );
        }
    }

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
