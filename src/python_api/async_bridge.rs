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
//! * [`register_exit_drain`] registers the `atexit` callback.  The callback
//!   first **closes the delivery gate** (`DeliveryTracker::close`), then
//!   releases the GIL (`Python::detach`) and waits, bounded by
//!   [`crate::constants::ASYNC_EXIT_DRAIN_TIMEOUT`], until no delivery is in
//!   flight.
//!
//! # The closed gate
//!
//! Waiting alone leaves a hole: a delivery whose future completes *after* the
//! drain returned would still enter `Python::attach` and could be inside it
//! when finalization starts.  The gate closes that hole.  The count and the
//! `closed` flag live under one mutex, so `enter` has exactly two outcomes:
//!
//! * it ran before `close`: the delivery is counted, the drain waits for it,
//!   and it completes normally (the drain released the GIL for it);
//! * it ran after `close`: it is refused, and [`S3dlioRuntime::spawn_blocking`]
//!   **does not call the delivery closure at all**.  The closure is dropped
//!   unrun, on the calling tokio worker thread.  That thread is not attached
//!   to Python, so the `Py<..>` objects the closure captured (the awaitable,
//!   the event loop, the `contextvars.Context`) take pyo3's deferred-decref
//!   path: `Py::drop` checks `thread_is_attached()`, and when false hands the
//!   pointer to `register_decref`, which only pushes it onto a Rust-side
//!   `Mutex<Vec>` and never calls into CPython.  Leaking those few references
//!   during interpreter teardown is harmless.  The Python awaitable is never
//!   resolved (see the known limits below for what that means to a waiter).
//!
//! What the gate guarantees, exactly: after `close()`, no *new* delivery is
//! admitted (a delivery closure scheduled after the close is dropped unrun).
//! Deliveries admitted *before* the close may still attach to Python while the
//! drain waits for them.  If the drain succeeds (within the bound), every
//! admitted delivery has finished before interpreter finalization begins, so
//! no s3dlio delivery can be inside an attach scope when `Py_FinalizeEx`
//! starts.  Attachment paths outside [`future_into_py`] delivery (any other
//! `Python::attach` in s3dlio or in user code) are not covered by this gate.
//! Ticket accounting stays exactly-once: a closure that ran, panicked or was
//! dropped unrun by the runtime releases its single ticket through the RAII
//! guard; a refused delivery never held one.
//!
//! A future that has not yet completed on the Rust side holds no ticket and is
//! never waited for, so a task that never completes cannot delay or hang exit.
//!
//! # Known limits
//!
//! * The drain wait is bounded.  A delivery that is still in flight when
//!   [`crate::constants::ASYNC_EXIT_DRAIN_TIMEOUT`] expires (a wedged
//!   `call_soon_threadsafe`, for example) was admitted before the close and
//!   can therefore still be inside `Python::attach` at finalization; the
//!   timeout is logged.  That is the same crash as before, a residual risk
//!   reachable only when a single delivery fails to finish within the timeout.
//! * The gate covers deliveries made through [`future_into_py`].  Any other
//!   `Python::attach` (elsewhere in s3dlio, or in user code) on a non-Python
//!   thread during teardown is outside this module and not covered.
//! * The gate never reopens: it closes at `atexit` and the process is ending.
//!   An awaitable whose delivery arrives after the close is **never
//!   resolved**.  `atexit` is LIFO, so a handler registered *before*
//!   `import s3dlio` runs after s3dlio's drain; if such a handler awaits an
//!   s3dlio awaitable whose delivery arrives after the close, that await never
//!   completes and the process can hang.  Handlers registered after the import
//!   run before the drain and are unaffected.
//! * This is a targeted shutdown fix for the observed crash, not a guarantee
//!   covering every teardown pattern.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObject;
use pyo3_async_runtimes::generic::Runtime;

/// Count of result deliveries that have been handed to the blocking pool and
/// have not yet fully returned.
///
/// Panic-safe bookkeeping: the count only ever changes through
/// [`DeliveryTicket`] construction and `Drop`, and a poisoned lock is
/// recovered (the protected value is a plain integer, so poisoning cannot leave
/// it inconsistent).
pub(crate) struct DeliveryTracker {
    state: Mutex<TrackerState>,
    idle: Condvar,
}

/// The count and the closed flag live under ONE mutex so that "is the gate
/// closed?" and "register one more delivery" are a single atomic decision.
struct TrackerState {
    in_flight: usize,
    closed: bool,
}

impl DeliveryTracker {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(TrackerState {
                in_flight: 0,
                closed: false,
            }),
            idle: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, TrackerState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Register one in-flight delivery.  The returned ticket releases it on
    /// drop.  Returns `None` once the gate is closed; nothing is counted then.
    ///
    /// The closed check and the increment happen under one lock hold, so every
    /// delivery is either refused or visible to the drain's wait: there is no
    /// window in which one is admitted but not counted.
    pub(crate) fn enter(&'static self) -> Option<DeliveryTicket> {
        let mut st = self.lock();
        if st.closed {
            return None;
        }
        st.in_flight = st.in_flight.saturating_add(1);
        drop(st);
        Some(DeliveryTicket { tracker: self })
    }

    /// Close the gate: from now on [`enter`](Self::enter) refuses.  Deliveries
    /// already counted are unaffected (they are waited for, not cancelled).
    /// Permanent for the life of the process and idempotent.
    pub(crate) fn close(&self) {
        self.lock().closed = true;
    }

    /// Whether the gate has been closed.
    #[cfg(test)]
    pub(crate) fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// Deliveries currently in flight.
    pub(crate) fn in_flight(&self) -> usize {
        self.lock().in_flight
    }

    /// Block the calling thread until no delivery is in flight or `timeout`
    /// elapses.  Returns `true` when drained, `false` on timeout.
    pub(crate) fn wait_idle(&self, timeout: Duration) -> bool {
        let guard = self.lock();
        let guard = self
            .idle
            .wait_timeout_while(guard, timeout, |st| st.in_flight > 0)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
        guard.in_flight == 0
    }
}

/// RAII proof that one result delivery is in flight.  Dropping it (on any path)
/// releases exactly one count and wakes the drain waiter when the count hits 0.
pub(crate) struct DeliveryTicket {
    tracker: &'static DeliveryTracker,
}

impl Drop for DeliveryTicket {
    fn drop(&mut self) {
        let mut st = self.tracker.lock();
        st.in_flight = st.in_flight.saturating_sub(1);
        if st.in_flight == 0 {
            self.tracker.idle.notify_all();
        }
    }
}

/// Process-wide tracker for deliveries made through [`S3dlioRuntime`].
///
/// Fork-aware (#297): the count belongs to the threads and tasks of the process that made it,
/// and its mutex may have been held by one of them at a fork. A fork child starts with a fresh,
/// open tracker (the inherited one is leaked, never used).
static DELIVERIES_CELL: crate::fork_guard::ForkCell<DeliveryTracker> =
    crate::fork_guard::ForkCell::new();

fn deliveries() -> &'static DeliveryTracker {
    DELIVERIES_CELL.get_or_init(DeliveryTracker::new)
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
        crate::s3_client::pyo3_runtime().spawn(fut)
    }

    fn spawn_blocking<F>(f: F) -> Self::JoinHandle
    where
        F: FnOnce() + Send + 'static,
    {
        spawn_gated(deliveries(), f)
    }
}

/// The gated `spawn_blocking` behind [`S3dlioRuntime`], parameterised on the
/// tracker so tests can drive it without touching the process-wide gate.
///
/// * Gate open: the ticket is taken BEFORE the delivery is queued and released
///   only after `f` (the whole `Python::attach` scope, drops included) has
///   returned -- or, if `f` panics or the queued closure is dropped unrun at
///   runtime shutdown, when the closure's captured ticket is dropped.
/// * Gate closed: `f` is **not run**.  It is dropped right here, on the
///   calling thread, which is a tokio worker that is not attached to Python:
///   the `Py<..>` objects `f` captured are therefore dropped without the GIL,
///   which pyo3 defers instead of touching CPython (see the module docs).  No
///   ticket exists on this path, so nothing is counted and nothing can be
///   double-released.
fn spawn_gated<F>(tracker: &'static DeliveryTracker, f: F) -> tokio::task::JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    let runtime = crate::s3_client::pyo3_runtime();
    match tracker.enter() {
        Some(ticket) => runtime.spawn_blocking(move || {
            // Release buffer exports queued by a finished `put_bytes_async(borrow=True)` BEFORE
            // the result reaches Python, so the caller can resize its buffer right after the
            // `await`. This thread is a blocking-pool thread that attaches to Python anyway.
            // Drain and deliver in ONE attach scope (the delivery closure re-enters it cheaply).
            if super::put_payload::exports_pending() {
                let mut f = Some(f);
                let _ = Python::try_attach(|py| {
                    super::put_payload::drain_pending_exports(py);
                    if let Some(f) = f.take() {
                        f();
                    }
                });
                if let Some(f) = f {
                    f(); // attaching was refused; deliver as before
                }
            } else {
                f();
            }
            drop(ticket);
        }),
        None => {
            tracing::debug!(
                "s3dlio: async result delivery dropped: interpreter is exiting \
                 (issue #192 delivery gate closed)"
            );
            drop(f);
            runtime.spawn(async {})
        }
    }
}

/// A Python reference that is NEVER dropped by pyo3's off-GIL path.
///
/// Dropping a `Py<..>` on a thread without the GIL makes pyo3 push the pointer onto its global
/// reference pool, a `Mutex` that is locked by every such drop. A parent thread that holds that
/// mutex at a `fork()` leaves it locked for ever in the child, and the child's first Python call
/// waits for it (#297). Every Python reference that lives in a task or closure of this module is
/// therefore wrapped here: it is consumed with the GIL held ([`GilRef::into_bound`]), and if it is
/// dropped anywhere else (a task dropped at runtime shutdown, a delivery refused by the closed
/// gate) it is queued for a GIL holder to decref ([`super::put_payload::defer_decref`]).
struct GilRef(Option<Py<PyAny>>);

impl GilRef {
    fn new(obj: Py<PyAny>) -> Self {
        Self(Some(obj))
    }

    /// Take the reference; the `py` token proves the GIL is held.
    fn into_bound(mut self, py: Python<'_>) -> Bound<'_, PyAny> {
        self.0
            .take()
            .expect("a GilRef is consumed once")
            .into_bound(py)
    }
}

impl Drop for GilRef {
    fn drop(&mut self) {
        if let Some(obj) = self.0.take() {
            super::put_payload::ASYNC_GILREF_DEFERRED
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            super::put_payload::defer_decref(obj.into_ptr());
        }
    }
}

/// `add_done_callback` target: a cancelled Python future cancels the Rust future.
#[pyclass]
struct DoneCallback {
    cancel_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

#[pymethods]
impl DoneCallback {
    fn __call__(&mut self, fut: &Bound<'_, PyAny>) -> PyResult<()> {
        if future_cancelled(fut).unwrap_or(false) {
            if let Some(tx) = self.cancel_tx.take() {
                let _ = tx.send(());
            }
        }
        Ok(())
    }
}

/// Runs on the event loop thread: completes the future unless it was cancelled meanwhile.
#[pyclass]
struct CheckedCompletor;

#[pymethods]
impl CheckedCompletor {
    fn __call__(
        &self,
        future: &Bound<'_, PyAny>,
        complete: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if future_cancelled(future)? {
            return Ok(());
        }
        complete.call1((value,))?;
        Ok(())
    }
}

fn future_cancelled(future: &Bound<'_, PyAny>) -> PyResult<bool> {
    future
        .getattr(pyo3::intern!(future.py(), "cancelled"))?
        .call0()?
        .is_truthy()
}

/// Hand the outcome to the event loop thread (`loop.call_soon_threadsafe`).
fn set_result(
    event_loop: &Bound<'_, PyAny>,
    future: &Bound<'_, PyAny>,
    result: PyResult<Py<PyAny>>,
) -> PyResult<()> {
    let py = event_loop.py();
    let none = py.None().into_bound(py);
    let (complete, val) = match result {
        Ok(val) => (
            future.getattr(pyo3::intern!(py, "set_result"))?,
            val.into_bound(py),
        ),
        Err(err) => (
            future.getattr(pyo3::intern!(py, "set_exception"))?,
            err.into_bound_py_any(py)?,
        ),
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item(pyo3::intern!(py, "context"), &none)?;
    event_loop.call_method(
        pyo3::intern!(py, "call_soon_threadsafe"),
        (CheckedCompletor, future, complete, val),
        Some(&kwargs),
    )?;
    Ok(())
}

fn panic_message(any: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = any.downcast_ref::<&str>() {
        s
    } else if let Some(s) = any.downcast_ref::<String>() {
        s.as_str()
    } else {
        "unknown error"
    }
}

/// What the Rust side of an awaitable ended with.
enum BridgeOutcome<T> {
    Value(PyResult<T>),
    /// The Python future was cancelled first: nothing to deliver.
    Cancelled,
    Panicked(String),
}

/// Replacement for `pyo3_async_runtimes::tokio::future_into_py` whose result deliveries are
/// visible to the exit drain (see the module docs) and which never drops a Python reference
/// without the GIL (#297).
///
/// The upstream function keeps a spare `Py` of the awaitable (for the panic path) in the task
/// that spawns the delivery; that task ends on a Tokio worker, which drops the reference without
/// the GIL on EVERY call. Here the awaitable and the event loop each have exactly one owner at
/// any time (the task, then the delivery closure) and both are consumed under the GIL; a Rust
/// panic is caught inside the task and delivered the same way, so no second reference is needed.
pub(crate) fn future_into_py<F, T>(py: Python<'_>, fut: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    // Test builds only: lets a test learn when the caller-supplied future body has fully
    // finished (see `body_finished_hook`). Absent from the shipped extension.
    #[cfg(test)]
    let fut = body_finished_hook::observe(fut);

    let event_loop = pyo3_async_runtimes::get_running_loop(py)?;
    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();
    let py_fut = event_loop.call_method0(pyo3::intern!(py, "create_future"))?;
    py_fut.call_method1(
        pyo3::intern!(py, "add_done_callback"),
        (DoneCallback {
            cancel_tx: Some(cancel_tx),
        },),
    )?;
    let loop_ref = GilRef::new(event_loop.unbind());
    let fut_ref = GilRef::new(py_fut.clone().unbind());

    S3dlioRuntime::spawn(async move {
        use futures::FutureExt;
        let body = async move {
            tokio::pin!(fut);
            let mut cancel_rx = Some(cancel_rx);
            std::future::poll_fn(|cx| {
                if let std::task::Poll::Ready(v) = fut.as_mut().poll(cx) {
                    return std::task::Poll::Ready(BridgeOutcome::Value(v));
                }
                if let Some(rx) = cancel_rx.as_mut() {
                    match Pin::new(rx).poll(cx) {
                        std::task::Poll::Ready(Ok(())) => {
                            return std::task::Poll::Ready(BridgeOutcome::Cancelled)
                        }
                        // The callback was dropped without a cancellation: keep waiting.
                        std::task::Poll::Ready(Err(_)) => cancel_rx = None,
                        std::task::Poll::Pending => {}
                    }
                }
                std::task::Poll::Pending
            })
            .await
        };
        let outcome = match std::panic::AssertUnwindSafe(body).catch_unwind().await {
            Ok(o) => o,
            Err(payload) => BridgeOutcome::Panicked(format!(
                "rust future panicked: {}",
                panic_message(&*payload)
            )),
        };
        // Not on this worker: the delivery needs the GIL (see the module docs).
        S3dlioRuntime::spawn_blocking(move || {
            Python::attach(move |py| {
                let event_loop = loop_ref.into_bound(py);
                let future = fut_ref.into_bound(py);
                let result = match outcome {
                    BridgeOutcome::Cancelled => return,
                    BridgeOutcome::Value(r) => r.and_then(|v| v.into_py_any(py)),
                    BridgeOutcome::Panicked(msg) => {
                        Err(pyo3_async_runtimes::err::RustPanic::new_err(msg))
                    }
                };
                if future_cancelled(&future).unwrap_or(false) {
                    return;
                }
                if let Err(e) = set_result(&event_loop, &future, result) {
                    e.print_and_set_sys_last_vars(py);
                }
            });
        });
    });
    Ok(py_fut)
}

/// Test-only observation of "the future body handed to [`future_into_py`] has returned".
///
/// The flag is registered per THREAD and consumed by the next `future_into_py` call made on
/// that thread, so concurrently running tests cannot see each other's futures. It is set the
/// moment the body's last statement has returned, i.e. before the result is converted and
/// delivered. A body that still needs the GIL (an ungated `Python::attach` after its I/O,
/// issue #233 / C-F2) cannot finish while the GIL is held elsewhere; a body that returns a
/// Rust value can.
#[cfg(test)]
pub(crate) mod body_finished_hook {
    use std::cell::RefCell;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    thread_local! {
        static NEXT: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
    }

    /// Arm the hook: the next `future_into_py` call on THIS thread sets `flag` when its body
    /// has returned.
    pub(crate) fn arm(flag: Arc<AtomicBool>) {
        NEXT.with(|slot| *slot.borrow_mut() = Some(flag));
    }

    pub(super) fn observe<F: Future>(fut: F) -> impl Future<Output = F::Output> {
        let flag = NEXT.with(|slot| slot.borrow_mut().take());
        async move {
            let out = fut.await;
            if let Some(flag) = flag {
                flag.store(true, Ordering::SeqCst);
            }
            out
        }
    }
}

/// `atexit` callback: release the GIL and wait (bounded) for in-flight result
/// deliveries to finish before the interpreter starts finalizing.
#[pyfunction]
fn drain_async_deliveries_at_exit(py: Python<'_>) {
    let timeout = crate::constants::ASYNC_EXIT_DRAIN_TIMEOUT;
    // Silent: a child that never used s3dlio must not warn merely because it exits.
    let tracker = DELIVERIES_CELL.get_or_init_silent(DeliveryTracker::new);
    tracker.close();
    let drained = py.detach(|| tracker.wait_idle(timeout));
    // Close the buffer-export releaser's attach gate, wait (GIL detached) for any attach scope it
    // is already in, release what is queued and stop releasing for good (issue #238): nothing
    // may attach to the interpreter once it starts finalizing.
    super::put_payload::begin_shutdown(py);
    if !drained {
        tracing::warn!(
            in_flight = tracker.in_flight(),
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
        let ticket = t.enter().expect("gate open");
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
        let ticket = t.enter().expect("gate open");
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
        let tickets: Vec<_> = (0..5).filter_map(|_| t.enter()).collect();
        assert_eq!(tickets.len(), 5);
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
            let _ticket = t.enter().expect("gate open");
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
            let _guard = t.state.lock().expect("lock");
            std::panic::panic_any("poison the lock");
        });
        assert!(poisoner.join().is_err());
        assert!(t.state.is_poisoned());
        let ticket = t.enter().expect("gate open");
        assert_eq!(t.in_flight(), 1);
        drop(ticket);
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    // ---- closed gate (issue #192: late deliveries must never enter Python) ----

    #[test]
    fn closed_gate_refuses_new_tickets_and_counts_nothing() {
        let t = tracker();
        assert!(!t.is_closed());
        t.close();
        assert!(t.is_closed());
        assert!(t.enter().is_none());
        assert!(t.enter().is_none());
        assert_eq!(t.in_flight(), 0);
        assert!(t.wait_idle(Duration::from_millis(10)));
        // Closing twice is harmless.
        t.close();
        assert!(t.is_closed());
    }

    #[test]
    fn close_then_wait_still_waits_for_deliveries_already_in_flight() {
        let t = tracker();
        let ticket = t.enter().expect("gate open");
        t.close();
        // A delivery that entered before the close is still counted, still
        // waited for, and not cancelled by the close.
        assert_eq!(t.in_flight(), 1);
        assert!(!t.wait_idle(Duration::from_millis(100)));
        assert_eq!(t.in_flight(), 1);
        // New work is refused while the old delivery is still running.
        assert!(t.enter().is_none());

        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(ticket);
        });
        let started = Instant::now();
        assert!(t.wait_idle(Duration::from_secs(30)));
        assert!(started.elapsed() < Duration::from_secs(10));
        releaser.join().expect("releaser thread");
        assert_eq!(t.in_flight(), 0);
        assert!(t.is_closed());
    }

    /// Sets a flag when dropped: proves a closure was dropped (not leaked).
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn spawn_after_close_drops_the_closure_unrun_and_leaves_the_count_alone() {
        let t = tracker();
        t.close();
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_in_task = Arc::clone(&ran);
        let guard = DropFlag(Arc::clone(&dropped));

        let handle = spawn_gated(t, move || {
            let _guard = guard;
            ran_in_task.store(true, Ordering::SeqCst);
        });
        // `f` is dropped synchronously, before the handle is even awaited.
        assert!(dropped.load(Ordering::SeqCst), "closure must be dropped");
        crate::s3_client::pyo3_runtime()
            .block_on(handle)
            .expect("join no-op handle");
        assert!(!ran.load(Ordering::SeqCst), "closure must NOT run");
        assert_eq!(t.in_flight(), 0, "nothing counted, nothing released");
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    #[test]
    fn spawn_before_close_still_runs_and_releases_exactly_once() {
        let t = tracker();
        let ran = Arc::new(AtomicBool::new(false));
        let ran_in_task = Arc::clone(&ran);
        let handle = spawn_gated(t, move || ran_in_task.store(true, Ordering::SeqCst));
        crate::s3_client::pyo3_runtime()
            .block_on(handle)
            .expect("join delivery");
        assert!(ran.load(Ordering::SeqCst));
        assert_eq!(t.in_flight(), 0);
        t.close();
        assert_eq!(t.in_flight(), 0);
    }

    #[test]
    fn panicking_delivery_releases_its_ticket_exactly_once() {
        let t = tracker();
        let handle = spawn_gated(t, || std::panic::panic_any("simulated delivery panic"));
        let joined = crate::s3_client::pyo3_runtime().block_on(handle);
        assert!(joined.is_err_and(|e| e.is_panic()));
        assert_eq!(t.in_flight(), 0);
        assert!(t.wait_idle(Duration::from_millis(10)));
    }

    /// `enter()` racing with `close()`: every iteration must end with a count
    /// of 0, and "the enter succeeded" must coincide exactly with "the ticket
    /// was already counted when the close finished" -- there is no window in
    /// which a delivery is admitted but invisible to the drain.
    #[test]
    fn enter_racing_with_close_never_leaves_the_count_wrong() {
        for iteration in 0..2000 {
            let t = tracker();
            let start = Arc::new(std::sync::Barrier::new(2));
            let (release_tx, release_rx) = mpsc::channel::<()>();

            let enterer = {
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    match t.enter() {
                        Some(ticket) => {
                            // Hold the ticket until the closer has looked.
                            let _ = release_rx.recv();
                            drop(ticket);
                            true
                        }
                        None => false,
                    }
                })
            };
            let closer = {
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    t.close();
                    let seen = t.in_flight();
                    let _ = release_tx.send(());
                    seen
                })
            };

            let admitted = enterer.join().expect("enterer thread");
            let seen_after_close = closer.join().expect("closer thread");
            assert_eq!(
                seen_after_close,
                usize::from(admitted),
                "iteration {iteration}: admitted={admitted} but the drain saw {seen_after_close}"
            );
            assert_eq!(t.in_flight(), 0, "iteration {iteration}: leaked count");
            assert!(t.is_closed());
            assert!(t.enter().is_none());
        }
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

        let before = deliveries().in_flight();
        let handle = <S3dlioRuntime as Runtime>::spawn_blocking(move || {
            let _ = started_tx.send(());
            let _ = release_rx.recv();
            finished_in_task.store(true, Ordering::SeqCst);
        });
        assert_eq!(deliveries().in_flight(), before + 1, "counted at enqueue");

        started_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("closure started");
        assert_eq!(deliveries().in_flight(), before + 1, "held while running");
        assert!(!finished.load(Ordering::SeqCst));

        release_tx.send(()).expect("release");
        crate::s3_client::pyo3_runtime()
            .block_on(handle)
            .expect("join delivery");
        assert!(finished.load(Ordering::SeqCst));
        assert!(deliveries().wait_idle(Duration::from_secs(10)));
        assert_eq!(deliveries().in_flight(), before);
    }
}

#[cfg(test)]
mod gilref_tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::sync::atomic::Ordering;

    /// A `GilRef` that is NOT consumed under the GIL (a task dropped at runtime shutdown, a
    /// delivery refused by the closed gate) and is dropped on a thread without the GIL does not
    /// touch Python (no decref there, nothing registered in pyo3's reference pool): it is queued
    /// for a GIL holder and counted; a consumed one is not counted (#297).
    #[test]
    fn an_unconsumed_gilref_is_queued_not_dropped_and_a_consumed_one_is_not_counted() {
        let _serial = super::super::put_payload::test_serial_guard();
        Python::initialize();
        Python::attach(|py| {
            let obj: Py<PyAny> = PyDict::new(py).into_any().unbind();
            let refcnt = || unsafe { pyo3::ffi::Py_REFCNT(obj.as_ptr()) };
            assert_eq!(refcnt(), 1);

            // Consumed under the GIL: no fallback, count restored by the Bound's own drop.
            let counted = super::super::put_payload::ASYNC_GILREF_DEFERRED.load(Ordering::Relaxed);
            let consumed = GilRef::new(obj.clone_ref(py));
            assert_eq!(refcnt(), 2);
            drop(consumed.into_bound(py));
            assert_eq!(refcnt(), 1);
            assert_eq!(
                super::super::put_payload::ASYNC_GILREF_DEFERRED.load(Ordering::Relaxed),
                counted
            );

            // Dropped un-consumed on a thread that does not hold the GIL (we hold it here for
            // the whole test, so nothing can drain the queue meanwhile).
            let stray = GilRef::new(obj.clone_ref(py));
            assert_eq!(refcnt(), 2);
            std::thread::spawn(move || drop(stray))
                .join()
                .expect("dropping thread");
            assert!(
                super::super::put_payload::ASYNC_GILREF_DEFERRED.load(Ordering::Relaxed) > counted,
                "the fallback is counted"
            );
            assert_eq!(refcnt(), 2, "not released by the dropping thread");
            super::super::put_payload::drain_pending_exports(py);
            assert_eq!(refcnt(), 1, "released exactly once by the GIL holder");
        });
    }
}
