// src/test_env.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The ONE crate-wide guard for unit tests that read or write process
//! environment variables (issue #216).
//!
//! Environment variables and the process-wide atomics derived from them
//! (`s3_client::RT_THREADS_LIMIT`, `CONCURRENCY_HINT`) are shared by every
//! unit test in the `--lib` binary, and cargo runs those tests on parallel
//! threads.  A lock that is private to one test module cannot protect a
//! variable that tests in OTHER modules also set or read: before this guard,
//! `constants::mpi_thread_budget_tests`, `constants::effective_thread_budget_tests`
//! and `s3_client::mpi_aware_runtime_sizing_tests` each held their own lock
//! while all three set `OMPI_COMM_WORLD_SIZE`, `PMI_SIZE` and `WORLD_SIZE`
//! (and `hardware`, `python_aiml_api` and `s3_client` raced on
//! `S3DLIO_RT_THREADS`; `retry` and `constants::max_retry_tests` raced on
//! `S3DLIO_MAX_RETRY_ATTEMPTS`).
//!
//! Every test that sets, removes, or asserts on one of those variables takes
//! this guard for its whole body.  It is a `tokio::sync::Mutex` so that it
//! (a) never poisons -- one failed assertion does not turn every later
//! test into a "lock poisoned" failure -- and (b) can be held across
//! `.await` in async tests without tripping `clippy::await_holding_lock`.
//!
//! Do not take it twice in one test (it is not re-entrant), and do not call
//! [`lock`] from inside an async context (use [`lock_async`]).

use tokio::sync::{Mutex, MutexGuard};

static ENV_TEST_LOCK: Mutex<()> = Mutex::const_new(());

/// Take the guard from a synchronous `#[test]`.
pub(crate) fn lock() -> MutexGuard<'static, ()> {
    ENV_TEST_LOCK.blocking_lock()
}

/// Take the guard from an async test.
pub(crate) async fn lock_async() -> MutexGuard<'static, ()> {
    ENV_TEST_LOCK.lock().await
}

// ---------------------------------------------------------------------------
// Panic-safe restore
// ---------------------------------------------------------------------------
//
// The lock above never poisons, so a failed assertion inside a test body must
// not leave the variables it changed behind for the next test.  Restoring
// after `body()` returns is not enough -- a panic skips it.  These guards
// restore in `Drop`, which also runs while unwinding.
//
// Declare them AFTER the lock guard: locals drop in reverse order, so the
// restore then runs while the lock is still held.

/// Restores process environment variables to the values they had when the
/// guard was created, on drop (including during a panic unwind).
pub(crate) struct EnvRestore {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvRestore {
    /// Record the current value (or absence) of each of `vars`.  Create it
    /// BEFORE changing any of them.
    pub(crate) fn capture(vars: &[&'static str]) -> Self {
        Self {
            saved: vars.iter().map(|v| (*v, std::env::var_os(v))).collect(),
        }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
    }
}

/// Restores a process-wide `AtomicUsize` (for example `RT_THREADS_LIMIT`) to
/// the value it had when the guard was created, on drop (including during a
/// panic unwind).
pub(crate) struct AtomicRestore {
    target: &'static std::sync::atomic::AtomicUsize,
    prev: usize,
}

impl AtomicRestore {
    /// Record the current value of `target`.  Create it BEFORE changing it.
    pub(crate) fn capture(target: &'static std::sync::atomic::AtomicUsize) -> Self {
        Self {
            target,
            prev: target.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

impl Drop for AtomicRestore {
    fn drop(&mut self) {
        self.target
            .store(self.prev, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod restore_guard_tests {
    use super::{lock, AtomicRestore, EnvRestore};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const SET_VAR: &str = "S3DLIO_TEST_ENV_RESTORE_PROBE_SET";
    const UNSET_VAR: &str = "S3DLIO_TEST_ENV_RESTORE_PROBE_UNSET";
    static PROBE_ATOMIC: AtomicUsize = AtomicUsize::new(7);

    /// A panic inside the body must still put the variables and the atomic
    /// back (#216): this is the case a restore loop after `body()` misses.
    #[test]
    fn restore_runs_when_the_body_panics() {
        let _guard = lock();
        std::env::set_var(SET_VAR, "original");
        std::env::remove_var(UNSET_VAR);
        PROBE_ATOMIC.store(7, Ordering::Relaxed);

        let outcome = std::panic::catch_unwind(|| {
            let _env = EnvRestore::capture(&[SET_VAR, UNSET_VAR]);
            let _atomic = AtomicRestore::capture(&PROBE_ATOMIC);
            std::env::set_var(SET_VAR, "changed");
            std::env::set_var(UNSET_VAR, "now-set");
            PROBE_ATOMIC.store(99, Ordering::Relaxed);
            panic!("simulated failed assertion");
        });

        assert!(outcome.is_err(), "the body was expected to panic");
        assert_eq!(std::env::var(SET_VAR).as_deref(), Ok("original"));
        assert!(std::env::var_os(UNSET_VAR).is_none(), "must be unset again");
        assert_eq!(PROBE_ATOMIC.load(Ordering::Relaxed), 7);
        std::env::remove_var(SET_VAR);
    }

    #[test]
    fn restore_runs_on_normal_return() {
        let _guard = lock();
        std::env::remove_var(UNSET_VAR);
        {
            let _env = EnvRestore::capture(&[UNSET_VAR]);
            std::env::set_var(UNSET_VAR, "x");
        }
        assert!(std::env::var_os(UNSET_VAR).is_none());
    }
}
