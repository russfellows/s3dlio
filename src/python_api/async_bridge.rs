// src/python_api/async_bridge.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2026 Russ Fellows <russ.fellows@gmail.com>

//! Exit-safe bridge from Rust futures to Python awaitables (issue #192).
//!
//! # The bug this module closes
//!
//! `pyo3_async_runtimes::tokio::future_into_py` finishes an awaitable on a
//! tokio *blocking-pool* thread: `Python::attach` -> `set_result` ->
//! `loop.call_soon_threadsafe`.  `call_soon_threadsafe` writes the event loop's
//! self-pipe (waking the main thread) and then **releases the GIL** around that
//! write.  The main thread can immediately return from `asyncio.run`, reach the
//! end of the script and start `Py_FinalizeEx` while the delivery thread is
//! still inside its `Python::attach` scope.  When that thread retakes the GIL,
//! CPython sees the runtime finalizing and force-exits the thread
//! (`PyThread_exit_thread`, a `pthread_exit` unwind).  The unwind drops pyo3
//! objects with pyo3's `GIL_COUNT` still 1 but a NULL CPython thread state, so
//! `Py_DECREF` deallocates a `contextvars.Context` with no thread state:
//! SIGSEGV.  See <https://github.com/russfellows/s3dlio/issues/192>.
//!
//! # The mechanism
//!
//! `atexit` callbacks run **before** CPython marks the runtime as finalizing,
//! so a delivery that is still in flight can complete normally if the exiting
//! thread simply releases the GIL and waits for it.  This module provides the
//! exact in-flight signal that wait needs:
//!
//! * [`S3dlioRuntime`] is a thin `pyo3_async_runtimes::generic::Runtime` that is
//!   identical to the stock tokio runtime adapter except that its
//!   `spawn_blocking` -- the call that carries the *whole* result delivery --
//!   takes a [`DeliveryTicket`] before the delivery is queued and drops it only
//!   after the delivery closure (including the `Python::attach` scope and every
//!   Python object it owned) has fully returned.  The ticket is held by a plain
//!   RAII guard, so it is released exactly once on every path: normal return,
//!   unwinding, or the closure being dropped unrun at runtime shutdown.
//! * [`future_into_py`] is a drop-in replacement for
//!   `pyo3_async_runtimes::tokio::future_into_py` built on that runtime; every
//!   s3dlio call site uses it.
//! * [`register_exit_drain`] registers the `atexit` callback that releases the
//!   GIL (`Python::detach`) and waits, bounded by
//!   [`crate::constants::ASYNC_EXIT_DRAIN_TIMEOUT`], until no delivery is in
//!   flight.
//!
//! A future that has not yet completed on the Rust side holds no ticket and is
//! never waited for, so a task that never completes cannot delay or hang exit.
//!
//! Known limit: a delivery that *starts* after the drain returned (a task the
//! script abandoned and that happens to finish during interpreter teardown) is
//! outside what an `atexit` hook can cover; that is an upstream pyo3 /
//! pyo3-async-runtimes concern, not something s3dlio can close from here.

use std::cell::OnceCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::IntoPyObject;
use pyo3_async_runtimes::generic::{self, ContextExt, Runtime};
use pyo3_async_runtimes::TaskLocals;

/// Count of result deliveries that have been handed to the blocking pool and
/// have not yet fully returned.
///
/// Panic-safe bookkeeping: the count only ever changes through
/// [`DeliveryTicket`] construction and `Drop`, and a poisoned lock is
/// recovered (the protected value is a plain integer, so poisoning cannot leave
/// it inconsistent).
pub(crate) struct DeliveryTracker {
    in_flight: Mutex<usize>,
    idle: Condvar,
}

impl DeliveryTracker {
    pub(crate) const fn new() -> Self {
        Self {
            in_flight: Mutex::new(0),
            idle: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, usize> {
        self.in_flight
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Register one in-flight delivery.  The returned ticket releases it on drop.
    pub(crate) fn enter(&'static self) -> DeliveryTicket {
        {
            let mut n = self.lock();
            *n = n.saturating_add(1);
        }
        DeliveryTicket { tracker: self }
    }

    /// Deliveries currently in flight.
    pub(crate) fn in_flight(&self) -> usize {
        *self.lock()
    }

    /// Block the calling thread until no delivery is in flight or `timeout`
    /// elapses.  Returns `true` when drained, `false` on timeout.
    pub(crate) fn wait_idle(&self, timeout: Duration) -> bool {
        let guard = self.lock();
        let guard = self
            .idle
            .wait_timeout_while(guard, timeout, |n| *n > 0)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
        *guard == 0
    }
}

/// RAII proof that one result delivery is in flight.  Dropping it (on any path)
/// releases exactly one count and wakes the drain waiter when the count hits 0.
pub(crate) struct DeliveryTicket {
    tracker: &'static DeliveryTracker,
}

impl Drop for DeliveryTicket {
    fn drop(&mut self) {
        let mut n = self.tracker.lock();
        *n = n.saturating_sub(1);
        if *n == 0 {
            self.tracker.idle.notify_all();
        }
    }
}

/// Process-wide tracker for deliveries made through [`S3dlioRuntime`].
static DELIVERIES: DeliveryTracker = DeliveryTracker::new();

tokio::task_local! {
    static TASK_LOCALS: OnceCell<TaskLocals>;
}

/// `pyo3_async_runtimes` runtime adapter over the shared pyo3-async-runtimes
/// tokio runtime (the one `s3_client::configure_thread_pools` sizes), with
/// delivery accounting on its blocking path.
struct S3dlioRuntime;

impl Runtime for S3dlioRuntime {
    type JoinError = tokio::task::JoinError;
    type JoinHandle = tokio::task::JoinHandle<()>;

    fn spawn<F>(fut: F) -> Self::JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        pyo3_async_runtimes::tokio::get_runtime().spawn(fut)
    }

    fn spawn_blocking<F>(f: F) -> Self::JoinHandle
    where
        F: FnOnce() + Send + 'static,
    {
        // Counted BEFORE the delivery is queued, released only after `f` (the
        // whole Python::attach scope, drops included) has returned.
        let ticket = DELIVERIES.enter();
        pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || {
            f();
            drop(ticket);
        })
    }
}

impl ContextExt for S3dlioRuntime {
    fn scope<F, R>(locals: TaskLocals, fut: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: Future<Output = R> + Send + 'static,
    {
        let cell = OnceCell::new();
        // A fresh cell is empty, so this `set` cannot fail.
        let _ = cell.set(locals);
        Box::pin(TASK_LOCALS.scope(cell, fut))
    }

    fn get_task_locals() -> Option<TaskLocals> {
        TASK_LOCALS
            .try_with(|cell| cell.get().cloned())
            .unwrap_or_default()
    }
}

/// Drop-in replacement for `pyo3_async_runtimes::tokio::future_into_py` whose
/// result deliveries are visible to the exit drain (see the module docs).
pub(crate) fn future_into_py<F, T>(py: Python<'_>, fut: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    generic::future_into_py::<S3dlioRuntime, F, T>(py, fut)
}

/// `atexit` callback: release the GIL and wait (bounded) for in-flight result
/// deliveries to finish before the interpreter starts finalizing.
#[pyfunction]
fn drain_async_deliveries_at_exit(py: Python<'_>) {
    let timeout = crate::constants::ASYNC_EXIT_DRAIN_TIMEOUT;
    if !py.detach(|| DELIVERIES.wait_idle(timeout)) {
        tracing::warn!(
            in_flight = DELIVERIES.in_flight(),
            timeout = ?timeout,
            "s3dlio: async result deliveries still in flight at interpreter exit; \
             giving up the wait (issue #192 drain timed out)"
        );
    }
}

/// Register the exit drain with Python's `atexit` (runs before finalization).
pub(crate) fn register_exit_drain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let callback = wrap_pyfunction!(drain_async_deliveries_at_exit, m)?;
    m.py()
        .import("atexit")?
        .call_method1("register", (callback,))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Instant;

    /// A fresh `'static` tracker per test (leaked: a few bytes, test-only).
    fn tracker() -> &'static DeliveryTracker {
        Box::leak(Box::new(DeliveryTracker::new()))
    }

    #[test]
    fn idle_tracker_drains_immediately() {
        let t = tracker();
        let started = Instant::now();
        assert!(t.wait_idle(Duration::from_secs(5)));
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(t.in_flight(), 0);
    }

    #[test]
    fn held_ticket_times_out_and_is_bounded() {
        let t = tracker();
        let ticket = t.enter();
        assert_eq!(t.in_flight(), 1);
        let started = Instant::now();
        assert!(!t.wait_idle(Duration::from_millis(150)));
        let waited = started.elapsed();
        assert!(waited >= Duration::from_millis(140), "waited {waited:?}");
        assert!(waited < Duration::from_secs(3), "waited {waited:?}");
        drop(ticket);
        assert_eq!(t.in_flight(), 0);
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    #[test]
    fn release_from_another_thread_wakes_waiter_early() {
        let t = tracker();
        let ticket = t.enter();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(ticket);
        });
        let started = Instant::now();
        assert!(t.wait_idle(Duration::from_secs(30)));
        assert!(started.elapsed() < Duration::from_secs(10));
        releaser.join().expect("releaser thread");
        assert_eq!(t.in_flight(), 0);
    }

    #[test]
    fn every_ticket_counts_exactly_once() {
        let t = tracker();
        let tickets: Vec<_> = (0..5).map(|_| t.enter()).collect();
        assert_eq!(t.in_flight(), 5);
        for (i, ticket) in tickets.into_iter().enumerate() {
            drop(ticket);
            assert_eq!(t.in_flight(), 4 - i);
        }
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    #[test]
    fn ticket_is_released_when_the_holder_panics() {
        let t = tracker();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ticket = t.enter();
            assert_eq!(t.in_flight(), 1);
            std::panic::panic_any("simulated delivery panic");
        }));
        assert!(outcome.is_err());
        assert_eq!(t.in_flight(), 0);
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    #[test]
    fn poisoned_lock_is_recovered() {
        let t = tracker();
        let poisoner = std::thread::spawn(move || {
            let _guard = t.in_flight.lock().expect("lock");
            std::panic::panic_any("poison the lock");
        });
        assert!(poisoner.join().is_err());
        assert!(t.in_flight.is_poisoned());
        let ticket = t.enter();
        assert_eq!(t.in_flight(), 1);
        drop(ticket);
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    /// The blocking path of the real runtime adapter: the global ticket is
    /// taken before the closure is queued and is still held while the closure
    /// body runs, and released only after it has returned.
    #[test]
    fn runtime_spawn_blocking_holds_ticket_until_closure_returns() {
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let finished = Arc::new(AtomicBool::new(false));
        let finished_in_task = Arc::clone(&finished);

        let before = DELIVERIES.in_flight();
        let handle = <S3dlioRuntime as Runtime>::spawn_blocking(move || {
            let _ = started_tx.send(());
            let _ = release_rx.recv();
            finished_in_task.store(true, Ordering::SeqCst);
        });
        assert_eq!(DELIVERIES.in_flight(), before + 1, "counted at enqueue");

        started_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("closure started");
        assert_eq!(DELIVERIES.in_flight(), before + 1, "held while running");
        assert!(!finished.load(Ordering::SeqCst));

        release_tx.send(()).expect("release");
        pyo3_async_runtimes::tokio::get_runtime()
            .block_on(handle)
            .expect("join delivery");
        assert!(finished.load(Ordering::SeqCst));
        assert!(DELIVERIES.wait_idle(Duration::from_secs(10)));
        assert_eq!(DELIVERIES.in_flight(), before);
    }
}
