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
