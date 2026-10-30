// src/python_api/put_payload.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Conversion of the `data` argument of `put_bytes()` / `put_bytes_async()`
//! into the `Bytes` handed to the object store (issues #238, round 2).
//!
//! # What is copied, and when
//!
//! The goal is zero copies. The store receives a `Bytes` that points at the caller's memory
//! for every C-contiguous buffer, whatever its item type (`uint8`, `float32`, `bool`,
//! `array('b')`, ctypes `'<B'`, a multi-dimensional C-contiguous array, ...).
//!
//! | `data`                                              | `put_bytes`, `put_many`  | `put_bytes_async` / `put_many_async` / `MultiEndpointStore` puts, default (`borrow=True`) | same with `borrow=False` (the opt-in snapshot) |
//! |-----------------------------------------------------|--------------------------|----------------------------|--------------------------------|
//! | `s3dlio.BytesView`                                  | 0 copies                 | 0 copies                   | 0 copies                       |
//! | exact `bytes`                                       | 0 copies                 | 0 copies                   | 0 copies                       |
//! | an immutable source behind a buffer export (`memoryview(bytes)`, a `bytes` subclass) | 0 copies | 0 copies | 0 copies (kept alive by a reference, read in place) |
//! | any other C-contiguous buffer (bytearray, memoryview, numpy, `array.array`, ctypes, mmap, ...) | 0 copies | 0 copies | 1 snapshot copy (GIL released above 256 KiB for `put_bytes_async`, held for the batch/endpoint puts) |
//! | non-contiguous (strided, reversed, transposed) view | 1 gather copy, GIL held  | same                       | same                           |
//! | not a buffer, only `__bytes__()`                    | 1 copy, made by Python   | same                       | same                           |
//!
//! "0 copies" means the native code reads the caller's own memory. For a buffer export
//! the `Py_buffer` is held as the owner of the `Bytes` ([`Bytes::from_owner`]) and read with
//! the GIL RELEASED, so other Python threads keep running during a large PUT (issue #238:
//! copying 512 MiB with the GIL held stalled every other thread for ~0.3 s).
//! The gather and `__bytes__` copies are made by Python itself, so the resulting `bytes`
//! object is wrapped as the owner: one copy, not two.
//!
//! # Contract with the caller
//!
//! While the export is held (for `put_bytes` and `put_many`: the whole call; for
//! `put_bytes_async`, `put_many_async` and the `MultiEndpointStore` puts: until the awaitable
//! completes, which is the DEFAULT (`borrow=True`); only during the snapshot copy with
//! `borrow=False`):
//!
//! * the exporter cannot be resized or freed: `bytearray.extend()/clear()` from another
//!   thread raises `BufferError`;
//! * the caller must NOT modify the bytes (another thread writing to the bytearray, numpy
//!   array, mmap or ctypes memory). Nothing prevents it, and it is a DATA RACE: the Rust side
//!   reads the memory through shared references, so concurrent writes are undefined behaviour
//!   in the Rust sense and the bytes uploaded are undefined. The memory itself is not freed or
//!   resized meanwhile (the export pins it), but that is not a safety net for writers. The same
//!   applies to the snapshot copy of `put_bytes_async` (made with the GIL released above
//!   256 KiB).
//!
//! # The completion barrier
//!
//! A put does not complete - normally, with an error, OR by cancellation - until every native
//! reader of the borrowed memory is done and the export has really been released, on every
//! backend. The per-call [`Latch`] is signalled by the export owner's `Drop`, which runs when
//! the LAST clone of the `Bytes` dies wherever that happens (Tokio tasks, SDK/hyper tasks,
//! `spawn_blocking` writers that a cancelled future cannot stop), and again when
//! `PyBuffer_Release` has really run. The two events are separate: an export can be dropped but
//! still owned by a drainer that is mid-release (its release callback can run Python code and
//! yield the GIL), so completion waits for `released`, notified to blocking and async waiters
//! alike, and re-checks it after every drain.
//!
//! * `put_bytes` checks the latch after the put returned (usually it is already released: then
//!   it only drains, holding the GIL) and otherwise waits with the GIL released, first for the
//!   owner's drop and then, after draining itself, for the release. It is bounded by
//!   `export_quiesce_timeout()` (30 s, `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`) and not interruptible
//!   by signals. If a backend never drops its clone the buffer stays pinned (never released while
//!   it can be read), the missed deadline is logged and counted, and the call raises
//!   `ExportStillReferencedError` in its own stack, where no Task can rewrite it.
//! * The awaitable of `put_bytes_async(borrow=True)` is a `Future` subclass
//!   (`GuardedFuture`, see `python_core_api.rs`), NOT a Task running a coroutine: a coroutine
//!   body never runs if its Task is cancelled before the first step. It NEVER reaches a terminal
//!   state - result, exception, cancelled - while a native reader remains or before the export is
//!   released: `cancel()` cancels the put but its own terminal state waits for the barrier, so
//!   `asyncio.timeout(0)`, `wait_for(.., 0)` and repeated `Task.cancel()` wait too. A returned
//!   exception could not carry "do not reuse" past Task cancellation (a Task whose waiter refused
//!   cancellation replaces the waiter's exception by `CancelledError`), so at every missed
//!   deadline it only warns, counts and waits again; a failure of the barrier machinery itself
//!   falls back to polling (and, if that cannot be scheduled, to a blocking wait).
//!
//! # Releasing the export never blocks a runtime thread
//!
//! `PyBuffer_Release` needs the GIL. The last clone of the `Bytes` is dropped wherever the
//! store finishes with it, typically a Tokio worker, and a Tokio worker must never wait for
//! the GIL. So [`ExportOwner`]'s `Drop` NEVER touches Python and never guesses whether the
//! GIL is held (`PyGILState_Check` can answer "held" for a thread without it once a
//! sub-interpreter exists): it always queues the export. Whoever holds the GIL releases it,
//! explicitly, with a `py` token:
//!
//! * the thread that waits for the call (`put_bytes` after it re-takes the GIL; the event-loop
//!   thread in `GuardedFuture` via `latch_ready`; the result-delivery path of
//!   `put_bytes_async`, which runs inside the #192 delivery gate) drains the queue before the
//!   call returns or the awaitable resolves;
//! * the streaming writers hold the buffer for the one call ([`Held`]) and release it directly
//!   with their GIL token (no queue); and
//! * a dedicated thread (`s3dlio-export-release`), after a short grace period
//!   ([`EXPORT_RELEASE_GRACE`]) that normally lets the callers above get there first, so it
//!   finds nothing to do and never contends for the GIL on the hot path.
//!
//! Exact `bytes` are held as a `Py<PyBytes>`: dropping it off the GIL defers the decref
//! (pyo3's reference pool), also without blocking.
//!
//! # Interpreter shutdown
//!
//! The atexit callback ([`begin_shutdown`]) closes the *autonomous attach gate* ([`AttachGate`]),
//! waits - with the GIL DETACHED - until no autonomous attach scope is in flight, and only then
//! drains the queue with the GIL and returns, so finalization never starts while the releaser
//! thread is still on its way into (or inside) CPython. The gate is a Dekker-style handshake on
//! two sequentially consistent atomics:
//!
//! * the releaser thread increments the in-flight counter, THEN checks the shutdown flag; when it
//!   is set it decrements and exits, otherwise it calls `Python::try_attach` and decrements after
//!   `try_attach` has returned (an RAII guard, so on every path including unwinding);
//! * the atexit path stores the shutdown flag, THEN waits for the counter to reach zero.
//!
//! One of the two always sees the other: either the releaser sees the flag and never attaches, or
//! atexit sees a non-zero counter and waits for that attach scope to finish. (A second flag check
//! inside the attach closure, as before, could not close this: it runs after the attachment.) The
//! wait must not hold the GIL - the releaser may be waiting for it - and is bounded by
//! `RELEASER_ATTACH_DRAIN_TIMEOUT` (30 s, generous because a draining releaser can run an
//! exporter's Python `__release_buffer__`); if it is exceeded the event is logged and counted
//! (`shutdown_attach_wait_timeouts`) and exit continues, so a wedged exporter can never hang the
//! process forever. That timeout is the only remaining way for the releaser to still be attached
//! at finalization.
//!
//! From then on the releaser never attaches again (and is not even started if the queue is
//! created later), on every Python version. An export dropped after that is still QUEUED (never
//! forgotten) so a thread that holds a real GIL token - a `put_bytes` finishing during exit -
//! releases it; what nobody drains is abandoned with the process. (pyo3's own `Py_IsFinalizing`
//! check inside `try_attach` exists only when built for Python 3.13 or newer, so it cannot be
//! relied on for 3.12; the gate does not depend on it.)
//!
//! # Other buffer details
//!
//! Buffers whose item format contains `O` (Python objects) or `P` (pointers) are refused with a
//! `TypeError`: their bytes are addresses, not data. A structured (record) dtype with padding is
//! uploaded as its raw memory, padding bytes included, exactly as it sits in memory.
//! The default `put_bytes_async` snapshot is copied with the GIL held below
//! [`SNAPSHOT_GIL_DETACH_THRESHOLD`] (releasing and re-taking it costs more than the copy) and
//! with the GIL released above it, where a concurrent writer (another Python thread, a C
//! extension, a device) is a DATA RACE with the copy: the snapshot is then undefined and the
//! behaviour is undefined in the Rust sense; multi-threaded callers that mutate shared buffers
//! must serialise the call with their writers. An allocation failure is a `MemoryError`.
//! Every writer (`put_bytes`, `put_bytes_async`, `write_chunk`, `write_owned_bytes`,
//! `MultipartUploadWriter.write`) uses the same STRICT conversion (issue #261): only
//! buffer-protocol objects and objects with `__bytes__()` are accepted. An int or a bool (which
//! `bytes(5)` would turn into five zero bytes) and an arbitrary iterable are a `TypeError`.
//! They BORROW the buffer for the one call (no thread may modify it until the call returns; the
//! export is released before it returns). `write_owned_bytes(copy=True)` and
//! `MultipartUploadWriter.write(copy=True)` instead take one snapshot with the GIL held
//! ([`snapshot_for_call`]); exact `bytes` and `BytesView` are immutable and never copied.

use super::python_core_api::PyBytesView;
use bytes::Bytes;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use crate::constants::{
    export_quiesce_timeout, EXPORT_RELEASE_GRACE, RELEASER_ATTACH_DRAIN_TIMEOUT,
    SNAPSHOT_GIL_DETACH_THRESHOLD,
};

/// How long the store reads the payload relative to the Python call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayloadUse {
    /// The call blocks until the store is done (`put_bytes`): read the caller's memory in place.
    Blocking,
    /// `put_bytes_async(borrow=False)`, the opt-in snapshot: the store keeps reading after the
    /// call returns, so MUTABLE memory is copied once, now. An immutable source (exact `bytes`,
    /// `BytesView`, `memoryview(bytes)`, a `bytes` subclass) is never copied.
    Snapshot,
    /// `put_bytes_async` (DEFAULT, `borrow=True`), `put_many_async`, the `MultiEndpointStore`
    /// puts: the caller promised not to modify or free the buffer until the awaitable completes,
    /// so it is read in place like `Blocking`.
    Borrowed,
    /// The streaming writers' `copy=True`: a mutable exporter is copied once, with the GIL held
    /// for the whole copy (no pure-Python thread can interleave; a writer that RELEASES the GIL -
    /// large NumPy operations, `readinto`/`recv_into`, CUDA/torch copies - still can and can tear
    /// the snapshot), and the export is released before the write starts. Exact `bytes` and
    /// `BytesView` are immutable and still not copied.
    SnapshotUnderGil,
}

/// The completion barrier of ONE call that holds a buffer export.
///
/// A put must not complete (normally, with an error, or by cancellation) until every native
/// reader of the borrowed memory is done and the export has really been released. The readers are
/// the clones of the `Bytes` built over the export, wherever they live (Tokio tasks, hyper/SDK
/// tasks, `spawn_blocking` file writers that a cancelled future cannot stop). The owner's `Drop`
/// runs when the LAST clone dies, so it is the exact "all readers are done" event:
///
/// * `dropped`: the owner has been dropped (no native reader remains); the export is then queued
///   for release (or released explicitly with a `py` token);
/// * `released`: `PyBuffer_Release` has run.
///
/// The Python-facing completion waits for `dropped`, releases the queue with a real `py` token,
/// re-checks `released` (another drainer may own the release) and waits for it, and only then
/// resolves, raises, or ends cancelled. Each wait is bounded by `export_quiesce_timeout()`; what
/// happens at the bound differs by caller (see "The completion barrier").
pub(crate) struct Latch {
    state: Mutex<LatchState>,
    cv: Condvar,
    /// Wakes the async waiter when `dropped` is set.
    notify: tokio::sync::Notify,
    /// Wakes the async waiter when `released` is set.
    notify_released: tokio::sync::Notify,
    /// Set once the short "re-drain" slice of [`quiesce_async`] has been used.
    redrain_slice_used: AtomicBool,
}

struct LatchState {
    dropped: bool,
    released: bool,
    /// Owners that have not yet been dropped / exports not yet released. A single-export latch
    /// ([`Latch::new`]) starts at 1 each; a batch latch ([`Latch::new_batch`]) counts up as
    /// exports are added and reports `dropped`/`released` only once the batch is sealed and every
    /// counted export has been dropped/released.
    drops_left: usize,
    releases_left: usize,
    /// A batch latch that is still being filled (see [`Latch::batch_seal`]).
    batch_open: bool,
    /// Exports counted on a batch latch so far.
    added: usize,
}

impl LatchState {
    fn single() -> Self {
        Self {
            dropped: false,
            released: false,
            drops_left: 1,
            releases_left: 1,
            batch_open: false,
            added: 0,
        }
    }

    /// Set the flags whose counts have reached zero (once the batch, if any, is sealed).
    fn settle(&mut self) {
        if !self.batch_open {
            if self.drops_left == 0 {
                self.dropped = true;
            }
            if self.releases_left == 0 {
                self.released = true;
            }
        }
    }
}

/// Times a wait for native readers hit its deadline (a backend kept a clone past the timeout).
static QUIESCE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
/// Failures of the barrier machinery itself (not a backend that holds on).
static BARRIER_INTERNAL_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Count an internal failure of the barrier (the buffer stays pinned).
pub(crate) fn note_internal_error() {
    BARRIER_INTERNAL_ERRORS.fetch_add(1, Ordering::Relaxed);
}

/// Record that a backend did not drop its clone by the deadline: the export STAYS PINNED (memory
/// that is still being read is never released). Counted and logged at EVERY missed deadline (one
/// line per `export_quiesce_timeout()` per waiting call, so the rate is bounded by the timeout).
fn note_quiesce_timeout() {
    let n = QUIESCE_TIMEOUTS.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::warn!("{}", deadline_warning_text(n));
}

/// The text of the warning for the `n`-th missed deadline of this process (it names the timeout).
pub(crate) fn deadline_warning_text(n: u64) -> String {
    format!(
        "s3dlio: a storage backend has kept reading a borrowed Python buffer for more than {:?} \
         after the put finished (missed deadline #{n} in this process); the buffer stays pinned \
         and a put_bytes_async awaitable keeps waiting - it is never completed while a native \
         reader remains (put_bytes raises ExportStillReferencedError). \
         See s3dlio._put_export_stats()",
        export_quiesce_timeout()
    )
}

impl Latch {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(LatchState::single()),
            cv: Condvar::new(),
            notify: tokio::sync::Notify::new(),
            notify_released: tokio::sync::Notify::new(),
            redrain_slice_used: AtomicBool::new(false),
        })
    }

    /// A latch for ONE batch of exports (`put_many`, `put_many_async`, ...): the completion barrier
    /// of every export added with [`batch_add`](Self::batch_add) at once. It reads as dropped /
    /// released only after [`batch_seal`](Self::batch_seal) and when EVERY counted export has been
    /// dropped / released (an empty batch is ready as soon as it is sealed).
    pub(crate) fn new_batch() -> Arc<Self> {
        let latch = Self::new();
        {
            let mut state = latch.lock();
            state.drops_left = 0;
            state.releases_left = 0;
            state.batch_open = true;
        }
        latch
    }

    /// Count one more export of a batch latch. Call BEFORE the export's owner exists.
    pub(crate) fn batch_add(&self) {
        let mut state = self.lock();
        state.drops_left += 1;
        state.releases_left += 1;
        state.added += 1;
    }

    /// How many exports were counted on this batch latch (0: the batch holds none, so no
    /// barrier is needed).
    pub(crate) fn batch_len(&self) -> usize {
        self.lock().added
    }

    /// No more exports will be added: the latch may now report completion.
    pub(crate) fn batch_seal(&self) {
        let (dropped, released) = {
            let mut state = self.lock();
            state.batch_open = false;
            state.settle();
            (state.dropped, state.released)
        };
        self.cv.notify_all();
        if dropped {
            self.notify.notify_one();
        }
        if released {
            self.notify_released.notify_one();
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LatchState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn mark_dropped(&self) {
        let dropped = {
            let mut state = self.lock();
            state.drops_left = state.drops_left.saturating_sub(1);
            state.settle();
            state.dropped
        };
        self.cv.notify_all();
        if dropped {
            self.notify.notify_one();
        }
    }

    /// `PyBuffer_Release` has run: wake every blocking and async waiter.
    pub(crate) fn mark_released(&self) {
        let released = {
            let mut state = self.lock();
            state.releases_left = state.releases_left.saturating_sub(1);
            state.settle();
            state.released
        };
        self.cv.notify_all();
        if released {
            self.notify_released.notify_one();
        }
    }

    pub(crate) fn is_dropped(&self) -> bool {
        self.lock().dropped
    }

    pub(crate) fn is_released(&self) -> bool {
        self.lock().released
    }

    /// Block until the owner has been dropped; `false` on timeout. Call without the GIL.
    pub(crate) fn wait_dropped_blocking(&self, timeout: Duration) -> bool {
        let guard = self.lock();
        let (guard, _) = self
            .cv
            .wait_timeout_while(guard, timeout, |s| !s.dropped)
            .unwrap_or_else(PoisonError::into_inner);
        guard.dropped
    }

    /// Block until the export has really been released; `false` on timeout. Call WITHOUT the
    /// GIL: the thread that owns the release needs it.
    pub(crate) fn wait_released_blocking(&self, timeout: Duration) -> bool {
        let guard = self.lock();
        let (guard, _) = self
            .cv
            .wait_timeout_while(guard, timeout, |s| !s.released)
            .unwrap_or_else(PoisonError::into_inner);
        guard.released
    }

    /// Async variant of [`wait_dropped_blocking`](Self::wait_dropped_blocking).
    pub(crate) async fn wait_dropped(&self, timeout: Duration) -> bool {
        self.wait_flag(&self.notify, timeout, Self::is_dropped)
            .await
    }

    /// Async variant of [`wait_released_blocking`](Self::wait_released_blocking). It never
    /// attaches to Python: the release is done by whoever holds the GIL.
    pub(crate) async fn wait_released(&self, timeout: Duration) -> bool {
        self.wait_flag(&self.notify_released, timeout, Self::is_released)
            .await
    }

    async fn wait_flag(
        &self,
        notify: &tokio::sync::Notify,
        timeout: Duration,
        is_set: fn(&Self) -> bool,
    ) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if is_set(self) {
                return true;
            }
            if tokio::time::timeout_at(deadline, notify.notified())
                .await
                .is_err()
            {
                return is_set(self);
            }
        }
    }
}

/// The wait for native readers gave up: a backend still references the buffer. Maps to the
/// Python `ExportStillReferencedError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuiesceTimedOut;

/// `true` when every native reader is done AND the export has really been released. The queued
/// export is released with this thread's GIL token (`py`) when this thread finds it in the queue.
/// Never waits. Used on the event-loop thread by the awaitable of `put_bytes_async(borrow=True)`
/// and as the fast path of the blocking wait.
pub(crate) fn latch_ready(py: Python<'_>, latch: &Latch) -> bool {
    if latch.is_dropped() {
        drain_pending_exports(py);
        // Ready only when the export has REALLY been released: a dropped owner whose export is
        // not in the queue that was drained - another drainer's private batch whose release
        // callback is still running, or an abandoned export - is not ready.
        latch.is_released()
    } else {
        false
    }
}

/// Wait until every native reader of the export is done and the export has REALLY been released.
/// Used by the blocking `put_bytes` before it returns, however the put ended.
///
/// When the latch is already dropped and released (the usual case: file/direct stores and
/// borrowed-slice writers drop their last clone before returning) this only drains, holding the
/// GIL: no detach, no wait. Otherwise it waits with the GIL RELEASED: first for the owner's drop,
/// then (after draining the queue itself) for the release, which another drainer may own - its
/// release callback can need the GIL, so waiting for it while holding the GIL would deadlock.
/// One deadline, `export_quiesce_timeout()`, covers both waits (this wait is NOT interruptible by
/// signals/KeyboardInterrupt: the buffer must not be released while it can still be read). On
/// timeout the export stays pinned, the timeout is counted and `Err(QuiesceTimedOut)` is
/// returned; the caller raises it in its own stack, where no Task/Future can rewrite it.
pub(crate) fn quiesce_blocking(py: Python<'_>, latch: &Latch) -> Result<(), QuiesceTimedOut> {
    if latch_ready(py, latch) {
        return Ok(());
    }
    let deadline = std::time::Instant::now() + export_quiesce_timeout();
    let remaining = || deadline.saturating_duration_since(std::time::Instant::now());
    if !py.detach(|| latch.wait_dropped_blocking(remaining())) {
        note_quiesce_timeout();
        return Err(QuiesceTimedOut);
    }
    // Dropped: release what is queued with this thread's token, then re-check actual release.
    if latch_ready(py, latch) {
        return Ok(());
    }
    // The export is owned by a drainer that is mid-release (or sits in no drained queue).
    if py.detach(|| latch.wait_released_blocking(remaining())) {
        Ok(())
    } else {
        note_quiesce_timeout();
        Err(QuiesceTimedOut)
    }
}

/// How long the first "already dropped" step of [`quiesce_async`] waits before returning so the
/// caller re-drains: two release grace periods (50 ms).
const REDRAIN_SLICE: Duration = Duration::from_millis(50);

/// Async counterpart for the awaitable of `put_bytes_async(borrow=True)`. One step of the guard's
/// loop, bounded by `export_quiesce_timeout()`: while the owner has not been dropped it waits for
/// the drop; once dropped (the caller has drained the queue and the export is still not released:
/// another drainer is mid-release) it waits for the actual release. Returns `true` on TIMEOUT
/// (counted and logged; the caller keeps waiting, it never completes on a timeout).
/// It never attaches to Python: the queued export is released by the result-delivery path of
/// this very awaitable (which runs inside the #192 delivery gate and drains first, and is skipped
/// once the interpreter is exiting) or by the other drainer. Runs even when the awaitable was
/// cancelled. The caller re-checks [`latch_ready`] after every step.
pub(crate) async fn quiesce_async(latch: Arc<Latch>) -> bool {
    let timeout = export_quiesce_timeout();
    let ok = if latch.is_dropped() {
        // The owner was already dropped when this step started. Either another drainer is
        // mid-release, or the last clone died after the caller's `latch_ready()` and the export
        // sits undrained in the public queue (nobody on the event-loop thread would drain it).
        // The FIRST such step therefore waits only one short slice and returns, so that the
        // delivery of this very future drains the queue and the caller re-checks - no dependence
        // on the releaser thread. Later steps wait the full deadline for a drainer mid-release.
        if !latch.redrain_slice_used.swap(true, Ordering::AcqRel) {
            latch.wait_released(REDRAIN_SLICE.min(timeout)).await;
            return false;
        }
        latch.wait_released(timeout).await
    } else {
        latch.wait_dropped(timeout).await
    };
    if ok {
        false
    } else {
        note_quiesce_timeout();
        true
    }
}

/// Counters for the export barrier (hidden Python function `s3dlio._put_export_stats()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExportStats {
    /// Waits for native readers that timed out (export left pinned).
    pub(crate) quiesce_timeouts: u64,
    /// Exports waiting in the release queue right now.
    pub(crate) queued: usize,
    /// Internal failures of the barrier machinery (buffer left pinned).
    pub(crate) internal_errors: u64,
    /// Interpreter exits that had to wait for the releaser thread's attach scope (see
    /// [`AttachGate`]).
    pub(crate) shutdown_attach_waits: u64,
    /// Of those, the waits that hit `RELEASER_ATTACH_DRAIN_TIMEOUT`.
    pub(crate) shutdown_attach_wait_timeouts: u64,
    /// Times the releaser thread attached to Python and drained.
    pub(crate) releaser_attaches: usize,
    /// Test hook: times the releaser parked at the armed point.
    pub(crate) test_releaser_parked: u64,
}

pub(crate) fn export_stats() -> ExportStats {
    ExportStats {
        quiesce_timeouts: QUIESCE_TIMEOUTS.load(Ordering::Relaxed),
        queued: QUEUE.get().map_or(0, |q| q.count.load(Ordering::Acquire)),
        internal_errors: BARRIER_INTERNAL_ERRORS.load(Ordering::Relaxed),
        shutdown_attach_waits: gate().waits.load(Ordering::Relaxed),
        shutdown_attach_wait_timeouts: gate().wait_timeouts.load(Ordering::Relaxed),
        releaser_attaches: QUEUE
            .get()
            .map_or(0, |q| q.attaches.load(Ordering::Relaxed)),
        test_releaser_parked: TEST_RELEASER_PARKED.load(Ordering::SeqCst),
    }
}

/// Test hook (hidden `s3dlio._put_test_retain_next_clone`): when armed, the next payload that
/// holds a buffer export (`Blocking`/`Borrowed`) also keeps one clone of its `Bytes` in
/// [`TEST_RETAINED`], standing in for a storage backend that keeps reading after the put ended.
/// [`release_retained_for_test`] drops them in the order they were taken.
static TEST_RETAIN_NEXT: AtomicBool = AtomicBool::new(false);
static TEST_RETAINED: Mutex<Vec<Bytes>> = Mutex::new(Vec::new());

pub(crate) fn arm_retain_for_test() {
    TEST_RETAIN_NEXT.store(true, Ordering::SeqCst);
}

/// How many clones are retained right now.
pub(crate) fn retained_for_test() -> usize {
    TEST_RETAINED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len()
}

/// Drop every retained clone, in order, and disarm a pending [`arm_retain_for_test`] (so a test
/// that armed the hook and failed before any put consumed it cannot make the NEXT test's put
/// retain a clone forever). Never touches Python (an `ExportOwner` drop only queues), so it is
/// safe to call with the GIL held - which also keeps every drainer out until all of them are
/// queued. Returns how many were dropped.
pub(crate) fn release_retained_for_test() -> usize {
    TEST_RETAIN_NEXT.store(false, Ordering::SeqCst);
    let retained =
        std::mem::take(&mut *TEST_RETAINED.lock().unwrap_or_else(PoisonError::into_inner));
    let n = retained.len();
    drop(retained);
    n
}

/// Test hook (hidden `s3dlio._put_test_queue_export`): hold `data` as a blocking payload and drop
/// it at once on this GIL-holding thread, so the export sits in the release queue with nobody
/// draining it. When `park_ms > 0` the releaser thread then parks once, between entering the
/// attach gate and `Python::try_attach`, for that long.
pub(crate) fn test_queue_undrained_export(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    park_ms: u64,
) -> PyResult<()> {
    let payload = put_payload(py, data, PayloadUse::Blocking)?;
    TEST_PARK_RELEASER_MS.store(park_ms, Ordering::SeqCst);
    drop(payload);
    Ok(())
}

/// A raw `Py_buffer` export (`PyBUF_SIMPLE`: any C-contiguous exporter, any item type).
struct RawExport {
    view: Box<ffi::Py_buffer>,
    /// Marked "released" when `PyBuffer_Release` has really run (set by `ExportOwner::new`).
    latch: Option<Arc<Latch>>,
}

// SAFETY: `Send`: the `Py_buffer` is a plain description of memory that the export pins for as
// long as it is held; it is released by value, and only while holding the GIL (a `py` token, or
// the queue drained by a GIL holder), so which thread owns the struct does not matter.
// `Sync`: through `&RawExport` the fields (`buf`, `len`, `format`, ...) are only ever READ - they
// are fixed for the life of the export - and the read-only payload path hands out only shared
// `&[u8]` views of the memory. The writable path (`WritableExport`) reads the `buf` POINTER VALUE
// through a shared `&self.inner.export` in `as_mut_slice` - never the bytes behind it - and builds
// the `&mut [u8]` only inside `&mut WritableExport`, which is exclusive, and only while the caller
// honours the documented exclusive-use contract (nothing else reads or writes the destination
// until the call is terminal).
unsafe impl Send for RawExport {}
unsafe impl Sync for RawExport {}

/// Whether a PEP 3118 / `struct`-module format string describes Python objects or pointers, which
/// must never be uploaded as bytes (the bytes are memory addresses). The pointer-bearing codes are
/// `O` (object), `P` (`void *`), `&` (pointer to the following type: `ctypes.POINTER(T)` is
/// `&<T>`), `z` (`char *`), `Z` (`wchar_t *`) and `X` (function pointer, `X{}`), wherever they
/// appear: after byte-order prefixes and counts, and inside `T{...}`. `:name:` field names are
/// skipped. `Zf` / `Zd` / `Zg` are complex numbers (NumPy), not pointers. A lowercase `p` is a
/// Pascal string (a length byte plus characters), not a pointer, and is accepted.
///
/// Known limits (the format string alone cannot tell, so these buffers are NOT refused):
/// * A `:` inside a ctypes structure FIELD NAME hides what follows it. The PEP 3118 name syntax
///   (`:name:`) has no escape for a colon, so the scan cannot tell where such a name ends and a
///   pointer field after it can be missed.
/// * A ctypes `Union` reports the format `B` (plain bytes) whatever it contains, so a pointer
///   inside a union cannot be detected from the format.
fn format_has_pointers(fmt: &str) -> bool {
    let mut in_name = false;
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ':' => in_name = !in_name,
            _ if in_name => {}
            'Z' if matches!(chars.peek(), Some('f' | 'd' | 'g')) => {
                chars.next(); // the complex-number code, not a wide-string pointer
            }
            'O' | 'P' | '&' | 'z' | 'Z' | 'X' => return true,
            _ => {}
        }
    }
    false
}

/// The PUT refusal of a buffer of Python objects or pointers.
fn pointer_format_error(fmt: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "cannot upload a buffer of Python objects or pointers (buffer format {fmt:?}): \
         its bytes are memory addresses, not data"
    ))
}

/// The same refusal for a READ destination (`get_into_async`): nothing is uploaded there, so the
/// wording is about the destination (issue #277).
fn pointer_destination_error(fmt: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "a buffer of Python objects or pointers (buffer format {fmt:?}) cannot be used as a \
         read destination: object data would overwrite its memory addresses"
    ))
}

impl RawExport {
    fn get(data: &Bound<'_, PyAny>, flags: std::ffi::c_int) -> PyResult<Self> {
        // SAFETY: an all-zero Py_buffer is a valid "empty" value for PyObject_GetBuffer to fill.
        let mut view: Box<ffi::Py_buffer> = Box::new(unsafe { std::mem::zeroed() });
        // SAFETY: GIL held (we have a Bound), `view` is a valid out-pointer.
        if unsafe { ffi::PyObject_GetBuffer(data.as_ptr(), &mut *view, flags) } == -1 {
            return Err(PyErr::fetch(data.py()));
        }
        Ok(Self { view, latch: None })
    }

    /// Request a flat view of `data`. `Ok(None)`: the object has no flat (C-contiguous) view
    /// (non-contiguous exporter, or no buffer protocol). `Err`: a buffer of objects or pointers.
    fn acquire(data: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let py = data.py();
        // Ask for the item format too, to refuse arrays of objects/pointers; an exporter that
        // rejects the format request is retried without it (then no format check is possible).
        let export = match Self::get(data, ffi::PyBUF_SIMPLE | ffi::PyBUF_FORMAT)
            .or_else(|_| Self::get(data, ffi::PyBUF_SIMPLE))
        {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };
        if !export.view.format.is_null() {
            // SAFETY: a non-null `format` is a NUL-terminated string owned by the exporter.
            let fmt = unsafe { std::ffi::CStr::from_ptr(export.view.format) }
                .to_string_lossy()
                .into_owned();
            if format_has_pointers(&fmt) {
                export.release(py);
                return Err(pointer_format_error(&fmt));
            }
        }
        Ok(Some(export))
    }

    fn len(&self) -> usize {
        usize::try_from(self.view.len).unwrap_or(0)
    }

    /// Release the export; the `py` token proves the GIL is held.
    fn release(mut self, _py: Python<'_>) {
        // SAFETY: `_py` proves this thread holds the GIL.
        unsafe { ffi::PyBuffer_Release(&mut *self.view) };
        if let Some(latch) = &self.latch {
            latch.mark_released();
        }
    }
}

/// Queue of exports waiting for a thread that holds the GIL to release them.
struct ReleaseQueue {
    pending: Mutex<Vec<RawExport>>,
    wake: Condvar,
    count: AtomicUsize,
    /// The autonomous attach gate (closed at interpreter exit): the releaser enters it before
    /// every attach and never attaches once it is closed.
    gate: &'static AttachGate,
    /// How long the releaser waits after a wake-up before attaching, so a `put_bytes` caller
    /// or the async delivery path (which drain the queue themselves) normally gets there first
    /// and the releaser finds nothing to do (no GIL traffic).
    grace: Duration,
    /// Times the releaser took the GIL (diagnostics / tests).
    attaches: AtomicUsize,
}

/// The process-wide gate, closed once by the atexit callback; see [`begin_shutdown`].
///
/// Fork-aware (#297): the in-flight count belongs to threads of the process that made it, and the
/// gate's lock may have been held by one of them at a fork. A fork child starts with a fresh,
/// open gate (the inherited one is leaked, never used).
static GATE_CELL: crate::fork_guard::ForkCell<AttachGate> = crate::fork_guard::ForkCell::new();

fn gate() -> &'static AttachGate {
    GATE_CELL.get_or_init(AttachGate::new)
}

/// Gate for the releaser thread's autonomous attachments to the interpreter.
///
/// `enter` and `close_and_wait` are the two halves of a handshake (see the module docs,
/// "Interpreter shutdown"): the in-flight counter is incremented BEFORE the flag is read, and the
/// flag is stored BEFORE the counter is read, all `SeqCst`, so at least one side sees the other.
pub(crate) struct AttachGate {
    closed: AtomicBool,
    in_flight: AtomicUsize,
    idle: Mutex<()>,
    idle_cv: Condvar,
    /// Closings that found an attach scope in flight and had to wait for it.
    waits: AtomicU64,
    /// Closings whose wait hit the bound (an attach scope was still in flight).
    wait_timeouts: AtomicU64,
}

/// RAII proof that one autonomous attach scope is in flight; dropping it (on any path) leaves
/// the scope and wakes a waiting `close_and_wait`.
pub(crate) struct AttachScope {
    gate: &'static AttachGate,
}

impl AttachGate {
    pub(crate) const fn new() -> Self {
        Self {
            closed: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            idle: Mutex::new(()),
            idle_cv: Condvar::new(),
            waits: AtomicU64::new(0),
            wait_timeouts: AtomicU64::new(0),
        }
    }

    /// Count one attach scope, THEN check the flag: `None` (nothing counted) once closed.
    pub(crate) fn enter(&'static self) -> Option<AttachScope> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let scope = AttachScope { gate: self }; // decrements on every path, including the refusal
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        Some(scope)
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Store the flag, THEN wait until no scope is in flight or `timeout` elapses. Call WITHOUT
    /// the GIL: an in-flight scope may be waiting for it. `true` when drained.
    pub(crate) fn close_and_wait(&self, timeout: Duration) -> bool {
        self.closed.store(true, Ordering::SeqCst);
        if self.in_flight.load(Ordering::SeqCst) == 0 {
            return true;
        }
        self.waits.fetch_add(1, Ordering::Relaxed);
        let guard = self.idle.lock().unwrap_or_else(PoisonError::into_inner);
        let (_guard, _) = self
            .idle_cv
            .wait_timeout_while(guard, timeout, |_| {
                self.in_flight.load(Ordering::SeqCst) > 0
            })
            .unwrap_or_else(PoisonError::into_inner);
        let drained = self.in_flight.load(Ordering::SeqCst) == 0;
        if !drained {
            self.wait_timeouts.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                "s3dlio: the buffer-export releaser was still attached to the interpreter \
                 {timeout:?} after shutdown began (an exporter's __release_buffer__ may be \
                 stuck); giving up the wait so the process can exit"
            );
        }
        drained
    }
}

impl Drop for AttachScope {
    fn drop(&mut self) {
        if self.gate.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            // Take the lock so a waiter between its check and its wait cannot miss the wake-up.
            drop(
                self.gate
                    .idle
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner),
            );
            self.gate.idle_cv.notify_all();
        }
    }
}

/// Test-only hook: park the releaser (in milliseconds, once) between its gate entry and
/// `Python::try_attach`, the interval of the Python 3.12 shutdown race. Armed by the hidden
/// `s3dlio._put_test_queue_export()`.
static TEST_PARK_RELEASER_MS: AtomicU64 = AtomicU64::new(0);
/// Times the releaser parked because of [`TEST_PARK_RELEASER_MS`].
static TEST_RELEASER_PARKED: AtomicU64 = AtomicU64::new(0);

fn park_releaser_for_test() {
    if TEST_PARK_RELEASER_MS.load(Ordering::Relaxed) == 0 {
        return; // production: a plain load, no read-modify-write per wake-up
    }
    let ms = TEST_PARK_RELEASER_MS.swap(0, Ordering::SeqCst);
    if ms > 0 {
        TEST_RELEASER_PARKED.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(ms));
    }
}

impl ReleaseQueue {
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<RawExport>> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn push(&self, export: RawExport) {
        let mut pending = self.lock();
        pending.push(export);
        self.count.store(pending.len(), Ordering::Release);
        drop(pending);
        self.wake.notify_one();
    }

    /// Release every queued export. The caller holds the GIL (it has a `Python` token); the
    /// releases run outside the queue lock (a release can run Python code that drops another
    /// export). The two `Vec`s swap, so neither loses its capacity: no reallocation per push.
    fn drain(&self, py: Python<'_>) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        thread_local! {
            static SCRATCH: RefCell<Vec<RawExport>> = const { RefCell::new(Vec::new()) };
        }
        let mut batch = SCRATCH.with(|s| std::mem::take(&mut *s.borrow_mut()));
        {
            let mut pending = self.lock();
            std::mem::swap(&mut *pending, &mut batch);
            self.count.store(0, Ordering::Release);
        }
        for export in batch.drain(..) {
            export.release(py);
        }
        SCRATCH.with(|s| *s.borrow_mut() = batch);
    }

    /// Dedicated releaser: wait for work, give the callers a grace period to drain it
    /// themselves, then release what is left while holding the GIL. Exports stay in the queue
    /// until the GIL is held, so a thread that drains while it holds the GIL sees every export
    /// that is not yet released. Every attachment is made inside an [`AttachScope`] of the gate,
    /// so the atexit path can close the gate and wait for it with the GIL detached; once the gate
    /// is closed the releaser never attaches.
    fn run(&'static self) {
        loop {
            {
                let mut pending = self.lock();
                while pending.is_empty() {
                    pending = self
                        .wake
                        .wait(pending)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
            std::thread::sleep(self.grace);
            // Counter first, flag second (see `AttachGate`): after this line either the gate is
            // known closed, or a closing atexit will wait for this scope.
            let Some(scope) = self.gate.enter() else {
                return;
            };
            if self.count.load(Ordering::Acquire) == 0 {
                continue; // a caller drained it during the grace period
            }
            park_releaser_for_test();
            let attached = Python::try_attach(|py| {
                self.attaches.fetch_add(1, Ordering::Relaxed);
                self.drain(py);
            });
            drop(scope); // after try_attach has returned
            if attached.is_none() || self.gate.is_closed() {
                return; // interpreter gone or exiting: abandon the rest with the process
            }
        }
    }

    /// A queue with its releaser thread.
    fn start(
        grace: Duration,
        gate: &'static AttachGate,
        spawn_releaser: bool,
    ) -> &'static ReleaseQueue {
        let queue: &'static ReleaseQueue = Box::leak(Box::new(ReleaseQueue {
            pending: Mutex::new(Vec::new()),
            wake: Condvar::new(),
            count: AtomicUsize::new(0),
            gate,
            grace,
            attaches: AtomicUsize::new(0),
        }));
        if !spawn_releaser {
            return queue;
        }
        if let Err(e) = std::thread::Builder::new()
            .name("s3dlio-export-release".into())
            .spawn(move || queue.run())
        {
            // Exports are then released by the next put_bytes/awaitable that drains.
            tracing::warn!("s3dlio: cannot start the buffer-export releaser thread: {e}");
        }
        queue
    }
}

/// The process-wide queue and its releaser thread (started on the first export that had to be
/// handed off).
static QUEUE: crate::fork_guard::ForkCell<&'static ReleaseQueue> =
    crate::fork_guard::ForkCell::new();

fn global_queue() -> &'static ReleaseQueue {
    QUEUE.get_or_init(|| {
        // After shutdown no releaser thread is started (it could only attach to a dying interpreter).
        ReleaseQueue::start(EXPORT_RELEASE_GRACE, gate(), !gate().is_closed())
    })
}

/// Release every queued export. The caller holds the GIL (it has a `Python` token).
pub(crate) fn drain_pending_exports(py: Python<'_>) {
    if let Some(queue) = QUEUE.get() {
        queue.drain(py);
    }
}

/// True when exports are waiting (cheap check for delivery paths).
pub(crate) fn exports_pending() -> bool {
    QUEUE
        .get()
        .is_some_and(|q| q.count.load(Ordering::Acquire) > 0)
}

/// Interpreter exit (called from the atexit callback, GIL held): close the autonomous attach
/// gate, wait for any attach scope of the releaser thread to finish WITH THE GIL DETACHED
/// (bounded by `RELEASER_ATTACH_DRAIN_TIMEOUT`), release what is queued, and stop releasing for
/// good. After this the releaser never attaches again. This replaces relying on pyo3's
/// `Py_IsFinalizing` check, which only exists when built for Python 3.13 or newer.
pub(crate) fn begin_shutdown(py: Python<'_>) {
    // Silent: a child that never used s3dlio must not warn merely because it exits.
    let gate = GATE_CELL.get_or_init_silent(AttachGate::new);
    py.detach(|| gate.close_and_wait(RELEASER_ATTACH_DRAIN_TIMEOUT));
    drain_pending_exports(py);
}

/// Owns a buffer export for as long as the `Bytes` built from it lives.
///
/// Dropping it never touches Python: the export is queued and released by a thread that holds
/// the GIL (whoever drains next, or the releaser thread). The GIL's state is never inferred
/// (`PyGILState_Check` can report "held" for a thread without the GIL once a sub-interpreter
/// exists); the paths that do hold it release explicitly with [`ExportOwner::release`].
struct ExportOwner {
    export: Option<RawExport>,
    /// `None`: the process-wide queue (tests inject their own).
    queue: Option<&'static ReleaseQueue>,
}

impl ExportOwner {
    fn new(mut export: RawExport, latch: Arc<Latch>) -> Self {
        export.latch = Some(latch);
        Self {
            export: Some(export),
            queue: None,
        }
    }

    /// Same, queueing into `queue` instead of the process-wide one (tests).
    #[cfg(test)]
    fn with_queue(export: RawExport, queue: &'static ReleaseQueue) -> (Self, Arc<Latch>) {
        let latch = Latch::new();
        let mut owner = Self::new(export, latch.clone());
        owner.queue = Some(queue);
        (owner, latch)
    }

    /// Release now; the `py` token proves the GIL is held. Marks the call's latch dropped too.
    fn release(mut self, py: Python<'_>) {
        if let Some(export) = self.export.take() {
            let latch = export.latch.clone();
            export.release(py);
            if let Some(latch) = latch {
                latch.mark_dropped();
            }
        }
    }
}

impl AsRef<[u8]> for ExportOwner {
    fn as_ref(&self) -> &[u8] {
        match &self.export {
            // SAFETY: the export is live (released only by `release`/the queue), non-empty and
            // contiguous; `buf`/`len` describe `len` readable bytes. Documented caller
            // obligation: no concurrent modification of those bytes.
            Some(e) if e.len() > 0 => unsafe {
                std::slice::from_raw_parts(e.view.buf as *const u8, e.len())
            },
            _ => &[],
        }
    }
}

impl Drop for ExportOwner {
    fn drop(&mut self) {
        let Some(export) = self.export.take() else {
            return;
        };
        // The last native reader is gone: this IS the "all readers done" event. The latch is
        // signalled AFTER the export is queued, so a waiter that wakes and drains finds it.
        let latch = export.latch.clone();
        let signal = |latch: &Option<Arc<Latch>>| {
            if let Some(l) = latch {
                l.mark_dropped();
            }
        };
        // SAFETY: always safe to call.
        if unsafe { ffi::Py_IsInitialized() } == 0 {
            std::mem::forget(export); // the interpreter is gone: nothing to release into
            signal(&latch);
            return;
        }
        // After shutdown the export is STILL queued (never forgotten): a thread that holds a real
        // GIL token - a `put_bytes` finishing during exit - can still drain and release it. Only
        // the autonomous releaser thread is off (it never attaches once the flag is set, and is
        // not even started).
        let queue = match self.queue {
            Some(q) => q,
            None => global_queue(),
        };
        queue.push(export);
        signal(&latch);
    }
}

/// Owner of a WRITABLE buffer export (the destination of `get_into_async`).
///
/// It wraps an [`ExportOwner`] rather than duplicating it, so the release and drop behaviour is
/// the very code the PUT path uses: `Drop` queues the export for a GIL holder to release and then
/// marks the latch dropped (nothing is queued, and the export is forgotten, on a finalized
/// interpreter); [`ExportOwnerMut::release`] releases directly with a `py` token, no queue.
/// `len` is the number of bytes the writer may use (`expected_len`, never more than the
/// exporter's length).
struct ExportOwnerMut {
    inner: ExportOwner,
    len: usize,
}

impl ExportOwnerMut {
    /// Take ownership of `export` (attaching `latch` to it) and validate it with
    /// [`check_writable_view`] for `len` bytes, so no `ExportOwnerMut` can exist for a view that
    /// fails those checks (writable, itemsize 1, plain byte format, no suboffsets, C-contiguous,
    /// `len <= export.len()`, a non-null `buf` when `len > 0`), which is what
    /// [`as_mut_slice`](Self::as_mut_slice) relies on.
    ///
    /// On a refused check the export is released here, with the `py` token (latch marked released,
    /// then dropped), and the error is returned: nothing stays held.
    fn new(py: Python<'_>, export: RawExport, latch: Arc<Latch>, len: usize) -> PyResult<Self> {
        let inner = ExportOwner::new(export, latch);
        let checked = match &inner.export {
            Some(e) => check_writable_view(&e.view, len),
            // `ExportOwner::new` always holds the export it was given.
            None => Err(pyo3::exceptions::PyBufferError::new_err(
                "internal error: the destination export is missing",
            )),
        };
        match checked {
            Ok(()) => Ok(Self { inner, len }),
            Err(e) => {
                inner.release(py);
                Err(e)
            }
        }
    }

    /// Exactly `len` bytes of the exporter's memory; `&mut []` when `len == 0`, so no slice is
    /// ever built from a pointer that may be null.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        match &self.inner.export {
            // SAFETY: the export is live (released only by `release`/the queue, both of which
            // consume or drop `self`), so `buf` points at the exporter's memory, which the export
            // pins. `new` ran `check_writable_view` (no `ExportOwnerMut` exists otherwise), which
            // gives a non-null `buf`, `len <= view.len` and a contiguous view, so `len` bytes are
            // in bounds. `&mut self` makes this the
            // only reference to those bytes derived from this owner; the documented caller
            // obligation (nothing else reads, writes, resizes or reuses the destination until the
            // call is terminal) covers the Python side.
            Some(e) => unsafe { std::slice::from_raw_parts_mut(e.view.buf.cast::<u8>(), self.len) },
            None => &mut [],
        }
    }

    /// Release now; the `py` token proves the GIL is held. Marks the latch released, then dropped.
    fn release(self, py: Python<'_>) {
        self.inner.release(py);
    }
}

/// A writable, C-contiguous byte destination held for one `get_into` call. Native code writes
/// the object into [`as_mut_slice`](Self::as_mut_slice); the export is released either directly
/// ([`release`](Self::release), with a GIL token, for failures before ownership transfers) or by
/// dropping it (the path once native code owns it: queued, then released by a GIL holder).
///
/// The destination is under an exclusive-use contract (plan §4.2 item 4): from acquisition until
/// the call is terminal, nothing else reads, writes, resizes or reuses that memory. In
/// particular, two concurrent receives into the same or overlapping memory are forbidden, since
/// each would hold a `&mut [u8]` over bytes the other also writes (aliasing mutable references).
pub(crate) struct WritableExport {
    owner: ExportOwnerMut,
}

// Note (no unsafe code here): `WritableExport` moves into a Tokio task and is used on whatever
// worker runs it. Its only non-`Send` content would be the raw `Py_buffer` pointers inside
// `RawExport`, which are `Send` (see the safety comment there): the export is merely a pin on the
// exporter's memory. The bytes behind `as_mut_slice` are written from that worker WITHOUT the
// GIL, which is sound under the exclusive-use contract (no Python thread reads or writes the
// destination until the call is terminal), and `as_mut_slice` needs `&mut self`, so two Rust
// threads cannot obtain the slice at once. Dropping the value on any thread never touches Python
// (it queues the export). The auto trait impls already hold; this is a compile-time assertion
// that they keep holding.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<WritableExport>();
};

impl WritableExport {
    /// Exactly `expected_len` bytes (never the exporter's full length); empty when 0.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.owner.as_mut_slice()
    }

    /// Release the export directly (no queue); the `py` token proves the GIL is held. Marks the
    /// call's latch released and dropped.
    pub(crate) fn release(self, py: Python<'_>) {
        self.owner.release(py);
    }
}

/// The checks of A4 on a freshly acquired writable `Py_buffer`, as pure functions of the view so
/// every refusal is one place: not read-only, one-byte items in a plain byte format (none of the
/// pointer-bearing formats; one optional byte-order prefix), `expected_len` within the exporter's
/// length (checked conversion), a non-null pointer when there is data, and (defense behind the
/// `PyBUF_C_CONTIGUOUS` request) no suboffsets and a really C-contiguous layout.
fn check_writable_view(view: &ffi::Py_buffer, expected_len: usize) -> PyResult<()> {
    use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
    if view.readonly != 0 {
        return Err(PyBufferError::new_err(
            "the destination buffer is read-only; get_into needs a writable buffer",
        ));
    }
    // The pointer-bearing formats are named before the item size is checked: an array of
    // pointers (`ctypes.c_void_p * n`) has 8-byte items, and "pointers" is the useful reason.
    if !view.format.is_null() {
        // SAFETY: a non-null `format` is a NUL-terminated string owned by the exporter, valid
        // while the export is held (the caller holds it).
        let fmt = unsafe { std::ffi::CStr::from_ptr(view.format) }.to_string_lossy();
        if format_has_pointers(&fmt) {
            return Err(pointer_destination_error(&fmt));
        }
    }
    if view.itemsize != 1 {
        return Err(PyTypeError::new_err(format!(
            "the destination buffer must have 1-byte items, got itemsize {}",
            view.itemsize
        )));
    }
    if !view.format.is_null() {
        // SAFETY: as above.
        let fmt = unsafe { std::ffi::CStr::from_ptr(view.format) }.to_string_lossy();
        // At most ONE leading byte-order/alignment character (`@ = < > !`), then exactly one byte
        // code: what a ctypes byte array reports (`<B`), and what the borrowed PUT path accepts.
        let code = fmt
            .strip_prefix(['@', '=', '<', '>', '!'])
            .unwrap_or(fmt.as_ref());
        if !matches!(code, "B" | "b" | "c") {
            return Err(PyTypeError::new_err(format!(
                "the destination buffer must be a plain byte buffer (format 'B', 'b' or 'c', \
                 optionally with one byte-order prefix), got format {fmt:?}"
            )));
        }
    }
    let len = usize::try_from(view.len).map_err(|_| {
        PyValueError::new_err(format!(
            "the destination buffer reports length {}",
            view.len
        ))
    })?;
    if expected_len > len {
        return Err(PyValueError::new_err(format!(
            "the destination buffer holds {len} bytes but {expected_len} are expected"
        )));
    }
    if len > 0 && view.buf.is_null() {
        return Err(PyBufferError::new_err(
            "the destination buffer has no memory address",
        ));
    }
    // Defense in depth behind the `PyBUF_C_CONTIGUOUS` request: an exporter that returned a
    // view anyway must really be one flat run of `len` bytes, or the slice built from `buf`
    // would cover memory the view does not describe.
    if !view.suboffsets.is_null() {
        return Err(PyBufferError::new_err(
            "the destination buffer uses suboffsets (indirect addressing); \
             get_into needs a flat C-contiguous buffer",
        ));
    }
    if view.ndim > 1 || !view.strides.is_null() {
        // SAFETY: `view` is a valid, fully initialised `Py_buffer` (a shared reference); when
        // `ndim > 1` or `strides` is set, `shape`/`strides` are the exporter's arrays of `ndim`
        // entries (the buffer-protocol contract), which `PyBuffer_IsContiguous` only reads. It
        // does not touch Python state and needs no particular thread state beyond the caller's.
        let contiguous = unsafe { ffi::PyBuffer_IsContiguous(view, b'C' as std::ffi::c_char) };
        if contiguous == 0 {
            return Err(PyBufferError::new_err(
                "the destination buffer is not C-contiguous; get_into needs a flat buffer",
            ));
        }
    }
    Ok(())
}

/// Hold `dst` as the writable destination of one `get_into` call (A4), on the CALLER'S `latch`
/// (the one the completion guard waits on).
///
/// The export is requested with `PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS`, with no
/// read-only fallback. Any refusal leaves nothing held: a held export is released first (with
/// `py`), and the latch is marked released and dropped, because no native writer will ever exist
/// for it. Then the error is raised.
///
/// The `latch` is SINGLE-USE per call: every path here ends with it marked released and dropped
/// (success: when the returned export is released or dropped). That includes the
/// `PyObject_GetBuffer` failure path, which marks both at once because nothing is held and no
/// owner will ever do it. Pass a fresh latch for every call; a latch must not be reused.
pub(crate) fn acquire_writable(
    py: Python<'_>,
    dst: &Bound<'_, PyAny>,
    expected_len: usize,
    latch: &Arc<Latch>,
) -> PyResult<WritableExport> {
    let flags = ffi::PyBUF_WRITABLE | ffi::PyBUF_FORMAT | ffi::PyBUF_C_CONTIGUOUS;
    let export = match RawExport::get(dst, flags) {
        Ok(export) => export,
        Err(e) => {
            // Nothing was exported: there is nothing for the latch to wait for.
            latch.mark_released();
            latch.mark_dropped();
            return Err(e);
        }
    };
    // From here the latch is attached to the export, so releasing it marks `released`. `new`
    // runs the view checks and, on a refusal, releases the export (released, then dropped).
    let owner = ExportOwnerMut::new(py, export, latch.clone(), expected_len)?;
    Ok(WritableExport { owner })
}

/// Exact `bytes` (or the `bytes` Python made for a gather / `__bytes__`) as the owner of a `Bytes`.
struct PyBytesOwner {
    _keep: Py<PyBytes>,
    ptr: *const u8,
    len: usize,
}

// SAFETY: `bytes` objects are immutable and `_keep` pins the allocation; dropping a `Py`
// off the GIL defers the decref (pyo3 reference pool), it never waits for the GIL.
unsafe impl Send for PyBytesOwner {}

impl AsRef<[u8]> for PyBytesOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: `ptr`/`len` were read from the live, immutable bytes object kept in `_keep`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

fn bytes_from_pybytes(b: &Bound<'_, PyBytes>) -> Bytes {
    let slice = b.as_bytes();
    if slice.is_empty() {
        return Bytes::new();
    }
    Bytes::from_owner(PyBytesOwner {
        ptr: slice.as_ptr(),
        len: slice.len(),
        _keep: b.clone().unbind(),
    })
}

/// One copy of `src` for `put_bytes_async` (default): with the GIL held below
/// [`SNAPSHOT_GIL_DETACH_THRESHOLD`] (releasing and re-taking the GIL costs more than the copy),
/// released above it so other threads keep running; with `may_detach == false` (the streaming
/// writers' `copy=True`) the GIL is held for the whole copy. An allocation failure is a
/// `MemoryError`, not an abort.
fn snapshot_copy(py: Python<'_>, src: &[u8], may_detach: bool) -> PyResult<Bytes> {
    let result = if !may_detach || src.len() < SNAPSHOT_GIL_DETACH_THRESHOLD {
        try_snapshot(src, src.len())
    } else {
        py.detach(|| try_snapshot(src, src.len()))
    };
    result.map_err(|_| snapshot_allocation_error(src.len()))
}

/// The `MemoryError` of a failed snapshot allocation. Every caller that snapshots a payload
/// (`put_bytes_async`, `put_many`, `MultiEndpointStore`, the streaming writers) shares it, so the
/// wording names no single API.
fn snapshot_allocation_error(len: usize) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyMemoryError, _>(format!(
        "cannot allocate {len} bytes for the payload snapshot"
    ))
}

/// Copy `src` into a fresh buffer reserved fallibly (`reserve` is `src.len()`; a parameter so
/// a test can force the failure).
fn try_snapshot(src: &[u8], reserve: usize) -> Result<Bytes, std::collections::TryReserveError> {
    let mut v = Vec::new();
    v.try_reserve_exact(reserve)?;
    v.extend_from_slice(src);
    Ok(Bytes::from(v))
}

/// The payload of a put: the `Bytes` for the store and, when it is backed by a buffer export
/// that the call must wait for, the completion barrier.
pub(crate) struct Payload {
    pub(crate) bytes: Bytes,
    /// `Some` only for payloads backed by a held `Py_buffer` export (`Blocking`/`Borrowed`
    /// buffers other than exact `bytes`); immutable payloads need no barrier.
    pub(crate) latch: Option<Arc<Latch>>,
}

/// What a conversion produced: plain `Bytes`, or a held buffer export (not yet wrapped).
enum Parts {
    Plain(Bytes),
    Export(ExportOwner, Arc<Latch>),
}

/// A payload held for the duration of ONE call that reads it through a borrowed slice and keeps
/// no clone (the streaming writers): on a buffer export, [`Held::release`] releases it
/// directly with the caller's GIL token, without going through the release queue.
pub(crate) struct Held(Parts);

impl Held {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Parts::Plain(b) => b.as_ref(),
            Parts::Export(owner, _) => owner.as_ref(),
        }
    }

    /// Release the hold; the `py` token proves the GIL is held.
    pub(crate) fn release(self, py: Python<'_>) {
        if let Parts::Export(owner, _) = self.0 {
            owner.release(py);
        }
    }
}

/// Hold `data` for one blocking call that borrows it as a slice (the streaming writers). Strict:
/// only buffer-protocol objects and objects with `__bytes__()` are accepted - NOT an int (which
/// `bytes(5)` would turn into five zero bytes) or an arbitrary iterable.
pub(crate) fn hold_for_call(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Held> {
    payload_parts(py, data, PayloadUse::Blocking).map(Held)
}

/// The streaming writers' `copy=True`: `data` as `Bytes` that nothing else can modify. A mutable
/// exporter is copied once with the GIL held; exact `bytes` and `BytesView` are returned as they
/// are (immutable, zero copies). Strict like [`hold_for_call`].
pub(crate) fn snapshot_for_call(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Bytes> {
    match payload_parts(py, data, PayloadUse::SnapshotUnderGil)? {
        Parts::Plain(bytes) => Ok(bytes),
        // Not produced for SnapshotUnderGil; copy defensively rather than assume.
        Parts::Export(owner, _) => {
            let copy = snapshot_copy(py, owner.as_ref(), false);
            owner.release(py);
            copy
        }
    }
}

/// Convert `data` to the `Bytes` that is written to the store (see the module docs for
/// exactly what is copied).
#[cfg(test)]
pub(crate) fn put_payload_from_py(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
) -> PyResult<Bytes> {
    put_payload(py, data, usage).map(|p| p.bytes)
}

/// One item of a batch (`put_many`, `put_many_async`): like [`put_payload`], but a held export
/// is counted on the shared `batch` latch ([`Latch::new_batch`]) instead of getting its own, so
/// ONE barrier covers the whole batch. A snapshot (`PayloadUse::Snapshot*`) or an immutable
/// source yields plain `Bytes` and adds nothing to the latch. Strict, like [`put_payload`].
pub(crate) fn put_payload_in_batch(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
    batch: &Arc<Latch>,
) -> PyResult<Bytes> {
    Ok(match payload_parts_in(py, data, usage, Some(batch))? {
        Parts::Plain(bytes) => bytes,
        Parts::Export(owner, _) => Bytes::from_owner(owner),
    })
}

/// Like [`put_payload_from_py`] but also returns the completion barrier of a held export.
/// Strict (issue #261): an int, a bool or an arbitrary iterable is a `TypeError`.
pub(crate) fn put_payload(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
) -> PyResult<Payload> {
    Ok(match payload_parts(py, data, usage)? {
        Parts::Plain(bytes) => Payload { bytes, latch: None },
        Parts::Export(owner, latch) => {
            let bytes = Bytes::from_owner(owner);
            // Relaxed load first: the production borrowed-export path does no read-modify-write.
            if TEST_RETAIN_NEXT.load(Ordering::Relaxed)
                && TEST_RETAIN_NEXT.swap(false, Ordering::SeqCst)
            {
                TEST_RETAINED
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(bytes.clone());
            }
            Payload {
                bytes,
                latch: Some(latch),
            }
        }
    })
}

/// The immutable `bytes` object (or `bytes` subclass instance) behind `data`, looking through
/// `memoryview` wrappers (`memoryview(b)`, `memoryview(memoryview(b))`, slices of them), or `None`
/// when `data` is anything else. A buffer's own `readonly` flag is NOT enough: a read-only
/// `memoryview(bytearray).toreadonly()`, a read-only mmap or a non-writeable NumPy view can still
/// change through another handle, so only a `bytes` base proves the memory can never change.
fn immutable_bytes_base<'py>(data: &Bound<'py, PyAny>) -> Option<Bound<'py, PyBytes>> {
    use pyo3::types::PyMemoryView;
    let mut current = data.clone();
    // Bounded: a memoryview of a memoryview of ... is a short chain.
    for _ in 0..16 {
        if let Ok(b) = current.cast::<PyBytes>() {
            return Some(b.clone());
        }
        if current.is_instance_of::<PyMemoryView>() {
            let next = current.getattr("obj").ok()?;
            if next.is_none() {
                return None; // a released memoryview
            }
            current = next;
            continue;
        }
        return None;
    }
    None
}

/// `Bytes` over an export of an IMMUTABLE source, with no copy and no held export: the
/// `bytes` object behind the export is kept alive by a reference instead (see
/// [`immutable_bytes_base`]). `None` unless the export is read-only, its source is a `bytes`
/// object and the exported memory lies inside that object's data; the caller then falls back to
/// the snapshot copy of a mutable source.
fn immutable_export_bytes(data: &Bound<'_, PyAny>, export: &RawExport) -> Option<Bytes> {
    if export.view.readonly == 0 || export.view.buf.is_null() {
        return None;
    }
    let base = immutable_bytes_base(data)?;
    let base_data = base.as_bytes();
    let start = base_data.as_ptr() as usize;
    let end = start.checked_add(base_data.len())?;
    let ptr = export.view.buf as usize;
    let len = export.len();
    if ptr < start || ptr.checked_add(len)? > end {
        return None;
    }
    Some(Bytes::from_owner(PyBytesOwner {
        ptr: ptr as *const u8,
        len,
        _keep: base.unbind(),
    }))
}

fn payload_parts(py: Python<'_>, data: &Bound<'_, PyAny>, usage: PayloadUse) -> PyResult<Parts> {
    payload_parts_in(py, data, usage, None)
}

/// [`payload_parts`], with the export (if any) counted on the shared `batch` latch instead of
/// getting a latch of its own.
fn payload_parts_in(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
    batch: Option<&Arc<Latch>>,
) -> PyResult<Parts> {
    let plain = Parts::Plain;
    // 1. s3dlio's own BytesView: refcount clone. MUST come first: BytesView also implements
    //    the buffer protocol, which used to shadow this branch (issue #238).
    if let Ok(bytes_view) = data.cast::<PyBytesView>() {
        return Ok(plain(bytes_view.try_borrow()?.bytes.clone()));
    }

    // 2. Exact `bytes`: immutable, kept alive by a reference (no export needed).
    if let Ok(exact) = data.cast_exact::<PyBytes>() {
        return Ok(plain(bytes_from_pybytes(exact)));
    }

    // 3. Any C-contiguous buffer, whatever its item type: hold the export. (Arrays of
    //    Python objects or pointers are refused with a TypeError.)
    if let Some(export) = RawExport::acquire(data)? {
        if export.len() == 0 {
            export.release(py);
            return Ok(plain(Bytes::new()));
        }
        let snapshotting = matches!(usage, PayloadUse::Snapshot | PayloadUse::SnapshotUnderGil);
        // A BORROWED export of an immutable source (the streaming writers, `borrow=True`) needs
        // no hold either: the `bytes` is kept alive by a reference, so there is no export to
        // guard or wait for, and the same view may be passed again while a part is in flight.
        if snapshotting || usage == PayloadUse::Borrowed {
            // Rule #1: an IMMUTABLE source (a `bytes` subclass, `memoryview(bytes)`, ...) cannot
            // change after this call, so a snapshot of it would be a pointless copy: the source
            // is kept alive by a reference and read in place, whatever the usage. Only a source
            // that can change (bytearray, NumPy, mmap, a read-only view of a mutable object, ...)
            // is copied; that copy is the documented opt-in of `borrow=False` / `copy=True`.
            if let Some(bytes) = immutable_export_bytes(data, &export) {
                export.release(py);
                return Ok(plain(bytes));
            }
        }
        let latch = match batch {
            Some(shared) if !snapshotting => {
                shared.batch_add();
                Arc::clone(shared)
            }
            _ => Latch::new(),
        };
        let owner = ExportOwner::new(export, latch.clone());
        return match usage {
            PayloadUse::Blocking | PayloadUse::Borrowed => Ok(Parts::Export(owner, latch)),
            PayloadUse::Snapshot | PayloadUse::SnapshotUnderGil => {
                // Copy now (the caller may reuse the buffer as soon as we return). `owner`
                // keeps the export, so the memory cannot be freed or resized meanwhile; it
                // is released explicitly right after (this thread holds the GIL again).
                let detach = usage == PayloadUse::Snapshot;
                let copy = snapshot_copy(py, owner.as_ref(), detach);
                owner.release(py);
                copy.map(plain)
            }
        };
    }

    // 4. A buffer that is not C-contiguous (strided, reversed, transposed): Python gathers
    //    it in logical order (one copy, GIL held so a concurrent change cannot tear it) and
    //    the resulting `bytes` is wrapped, not copied again.
    if let Ok(mv) = py
        .import("builtins")
        .and_then(|b| b.call_method1("memoryview", (data,)))
    {
        if let Ok(fmt) = mv.getattr("format").and_then(|f| f.extract::<String>()) {
            if format_has_pointers(&fmt) {
                return Err(pointer_format_error(&fmt));
            }
        }
        if let Ok(gathered) = mv.call_method0("tobytes") {
            if let Ok(b) = gathered.cast_exact::<PyBytes>() {
                return Ok(plain(bytes_from_pybytes(b)));
            }
        }
    }

    // 5. Last resort: `__bytes__()` only; Python's fresh `bytes` is wrapped (one copy). There is
    //    NO `bytes(obj)` fallback (issue #261): it would turn an int or a bool into that many
    //    zero bytes and accept any iterable of ints, which every writer must refuse.
    match data.call_method0("__bytes__") {
        Ok(bytes_obj) => match bytes_obj.cast_exact::<PyBytes>() {
            Ok(b) => Ok(plain(bytes_from_pybytes(b))),
            Err(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "__bytes__() returned non-bytes object: {}",
                bytes_obj.get_type().name()?
            ))),
        },
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Cannot convert data to bytes: {}. Expected bytes-like object (bytes, bytearray, memoryview, BytesView, numpy array, or object with __bytes__() method)",
            e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::ffi::c_str;
    use pyo3::types::{PyByteArray, PyDict};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn init() {
        Python::initialize();
    }

    const BOTH: [PayloadUse; 3] = [
        PayloadUse::Blocking,
        PayloadUse::Snapshot,
        PayloadUse::Borrowed,
    ];

    fn exported_ptr(obj: &Bound<'_, PyAny>) -> *const u8 {
        let e = RawExport::acquire(obj).unwrap().expect("buffer");
        let p = e.view.buf as *const u8;
        e.release(obj.py());
        p
    }

    /// Run `src` in a fresh namespace and return it.
    fn ns<'py>(py: Python<'py>, src: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(src, None, Some(&locals)).unwrap();
        locals
    }

    fn resizable(obj: &Bound<'_, PyAny>) -> bool {
        obj.call_method1("extend", (b"x".as_slice(),)).is_ok()
    }

    /// #238: a BytesView is reused as-is (same memory), in every mode.
    #[test]
    fn bytesview_is_reused_without_copy() {
        init();
        Python::attach(|py| {
            let original = Bytes::from(vec![7u8; 1 << 20]);
            let original_ptr = original.as_ptr();
            let view = Bound::new(py, PyBytesView::new(original)).unwrap();
            for mode in BOTH {
                let got = put_payload_from_py(py, view.as_any(), mode).unwrap();
                assert_eq!(
                    got.as_ptr(),
                    original_ptr,
                    "BytesView was copied ({mode:?})"
                );
                assert_eq!(got.len(), 1 << 20);
            }
        });
    }

    /// A BytesView that is mutably borrowed is an error, never a panic.
    #[test]
    fn mutably_borrowed_bytesview_is_an_error_not_a_panic() {
        init();
        Python::attach(|py| {
            let view = Bound::new(py, PyBytesView::new(Bytes::from_static(b"abc"))).unwrap();
            let _guard = view.borrow_mut();
            assert!(put_payload_from_py(py, view.as_any(), PayloadUse::Blocking).is_err());
        });
    }

    /// A blocking (or borrowed) put reads any contiguous exporter in place.
    #[test]
    fn contiguous_exporters_are_not_copied() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 1 << 20]);
            let src = exported_ptr(ba.as_any());
            for mode in [PayloadUse::Blocking, PayloadUse::Borrowed] {
                let got = put_payload_from_py(py, ba.as_any(), mode).unwrap();
                assert_eq!(got.as_ptr(), src, "bytearray was copied ({mode:?})");
                assert!(got.iter().all(|&b| b == 3));
            }
            let mv = ns(py, c_str!("ba = bytearray(1 << 20); mv = memoryview(ba)"))
                .get_item("mv")
                .unwrap()
                .unwrap();
            let got = put_payload_from_py(py, &mv, PayloadUse::Blocking).unwrap();
            assert_eq!(got.as_ptr(), exported_ptr(&mv), "memoryview was copied");
        });
    }

    /// Round 2, item 1: ANY item type, not just uint8/'B'.
    #[test]
    fn every_item_type_is_zero_copy() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "import array, ctypes\n\
                     objs = {\n\
                       'array_f': array.array('f', range(4096)),\n\
                       'array_b': array.array('b', [1, -2] * 2048),\n\
                       'array_q': array.array('q', range(1024)),\n\
                       'mv_c': memoryview(bytearray(b'x' * 4096)).cast('c'),\n\
                       'mv_h': memoryview(bytearray(4096)).cast('H'),\n\
                       'mv_3d': memoryview(bytearray(24)).cast('B', (2, 3, 4)),\n\
                       'ctypes_ubyte_mv': memoryview((ctypes.c_ubyte * 8192)()),\n\
                       'ctypes_float': (ctypes.c_float * 1024)(),\n\
                     }"
                ),
            );
            let objs = l.get_item("objs").unwrap().unwrap();
            let objs = objs.cast::<PyDict>().unwrap();
            for (name, obj) in objs.iter() {
                let src = exported_ptr(&obj);
                for mode in [PayloadUse::Blocking, PayloadUse::Borrowed] {
                    let got = put_payload_from_py(py, &obj, mode).unwrap();
                    assert_eq!(got.as_ptr(), src, "{name} was copied ({mode:?})");
                    let expected: usize = obj
                        .call_method0("__len__")
                        .and_then(|_| {
                            obj.getattr("nbytes").or_else(|_| {
                                py.import("builtins")?
                                    .call_method1("memoryview", (&obj,))?
                                    .getattr("nbytes")
                            })
                        })
                        .unwrap()
                        .extract()
                        .unwrap();
                    assert_eq!(got.len(), expected, "{name} length in bytes");
                }
            }
        });
    }

    /// The borrowed buffer stays exported (cannot be resized) while the Bytes is alive and is
    /// released, inline because this thread holds the GIL, when the last clone is dropped.
    #[test]
    fn borrowed_buffer_is_held_until_bytes_dropped() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Blocking).unwrap();
            let clone = got.clone();
            let err = ba
                .call_method1("extend", (b"x".as_slice(),))
                .expect_err("resize must fail while exported");
            assert!(err.is_instance_of::<pyo3::exceptions::PyBufferError>(py));
            drop(got);
            assert!(!resizable(ba.as_any()), "still held by the clone");
            drop(clone);
            // Dropping only queues the export; a thread holding the GIL releases it explicitly.
            drain_pending_exports(py);
            assert!(resizable(ba.as_any()), "released after the last clone");
        });
    }

    /// Round 2, item 2: dropping the last clone on a thread that does not hold the GIL must
    /// NOT wait for the GIL (a Tokio worker would stall); the export is released by whoever
    /// next holds the GIL.
    #[test]
    fn dropping_the_last_clone_without_the_gil_never_blocks() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Blocking).unwrap();
            let (tx, rx) = mpsc::channel();
            let t = std::thread::spawn(move || {
                drop(got); // no GIL on this thread; the main thread holds it throughout
                tx.send(()).ok();
            });
            // The main thread keeps the GIL while it waits: the drop must finish anyway.
            rx.recv_timeout(Duration::from_secs(5))
                .expect("dropping the export blocked on the GIL");
            t.join().unwrap();
            // Queued, not lost: a GIL holder drains it.
            drain_pending_exports(py);
            assert!(resizable(ba.as_any()), "export not released after drain");
        });
    }

    /// The dedicated releaser thread frees a queued export without any help from the caller.
    #[test]
    fn releaser_thread_frees_queued_exports() {
        init();
        let ba = Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Blocking).unwrap();
            py.detach(move || drop(got)); // dropped without the GIL
            ba.unbind()
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let ok = Python::attach(|py| resizable(ba.bind(py).as_any()));
            if ok {
                return;
            }
            assert!(Instant::now() < deadline, "export never released");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Exact `bytes` is immutable: never copied, in any mode, and not exported.
    #[test]
    fn exact_bytes_is_never_copied() {
        init();
        Python::attach(|py| {
            let b = PyBytes::new(py, &[9u8; 4096]);
            let src = b.as_bytes().as_ptr();
            for mode in BOTH {
                let got = put_payload_from_py(py, b.as_any(), mode).unwrap();
                assert_eq!(got.as_ptr(), src, "bytes was copied ({mode:?})");
                // dropping off the GIL defers the decref instead of blocking
                py.detach(move || drop(got));
            }
        });
    }

    /// The default `put_bytes_async` is a snapshot: later changes and resizes by the caller
    /// are neither blocked by nor visible to the pending write.
    #[test]
    fn snapshot_of_mutable_exporter_is_independent() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[5u8; 4096]);
            let src = exported_ptr(ba.as_any());
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Snapshot).unwrap();
            assert_ne!(got.as_ptr(), src);
            assert!(resizable(ba.as_any()), "no export may be held");
            ba.call_method1("__setitem__", (0, 6)).unwrap();
            assert_eq!(got.len(), 4096);
            assert!(got.iter().all(|&b| b == 5));
        });
    }

    /// Stage G (W9, Rule #1): a snapshot never copies an IMMUTABLE source. `memoryview(bytes)`
    /// (also nested, also a slice of it) and a `bytes` subclass yield `Bytes` over the very same
    /// memory in both snapshot modes, with no export held (the bytes object is kept alive by a
    /// reference instead), and the bytes stay valid after the Python names are dropped.
    #[test]
    fn snapshot_does_not_copy_an_immutable_source() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "class Sub(bytes):\n    pass\n\
                     base = bytes(range(256)) * 16\n\
                     sub = Sub(base)\n\
                     mv = memoryview(base)\n\
                     mv2 = memoryview(mv)\n\
                     piece = memoryview(base)[100:3000]\n\
                     mv_sub = memoryview(sub)"
                ),
            );
            for name in ["sub", "mv", "mv2", "piece", "mv_sub"] {
                let obj = l.get_item(name).unwrap().unwrap();
                let src = exported_ptr(&obj);
                for mode in [PayloadUse::Snapshot, PayloadUse::SnapshotUnderGil] {
                    let got = put_payload_from_py(py, &obj, mode).unwrap();
                    assert_eq!(got.as_ptr(), src, "{name} was copied ({mode:?})");
                    let full = (name == "sub" || name == "mv" || name == "mv2" || name == "mv_sub")
                        .then_some(4096);
                    assert_eq!(got.len(), full.unwrap_or(2900), "{name} length");
                    py.detach(move || drop(got)); // dropped off the GIL: never blocks
                }
            }
            // The Bytes outlive every Python name for the source.
            let got = put_payload_from_py(
                py,
                &l.get_item("mv").unwrap().unwrap(),
                PayloadUse::Snapshot,
            )
            .unwrap();
            py.run(c_str!("mv.release(); mv2.release()"), None, Some(&l))
                .unwrap();
            l.del_item("mv").unwrap();
            l.del_item("mv2").unwrap();
            l.del_item("base").unwrap();
            l.del_item("sub").unwrap();
            l.del_item("mv_sub").unwrap();
            l.del_item("piece").unwrap();
            assert_eq!(got.len(), 4096);
            assert!(got.iter().enumerate().all(|(i, &b)| b == (i % 256) as u8));
        });
    }

    /// The `readonly` flag alone does not make a source immutable: a read-only VIEW of a
    /// bytearray (or of anything else that can change) keeps the documented snapshot copy.
    #[test]
    fn snapshot_still_copies_a_readonly_view_of_a_mutable_source() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "ba = bytearray(4096)\n\
                     ro = memoryview(ba).toreadonly()"
                ),
            );
            let ro = l.get_item("ro").unwrap().unwrap();
            let src = exported_ptr(&ro);
            let ba = l.get_item("ba").unwrap().unwrap();
            for mode in [PayloadUse::Snapshot, PayloadUse::SnapshotUnderGil] {
                let got = put_payload_from_py(py, &ro, mode).unwrap();
                assert_ne!(got.as_ptr(), src, "a mutable source must be snapshotted");
                ba.call_method1("__setitem__", (0, 7)).unwrap();
                assert_eq!(got[0], 0, "the snapshot must not see later changes");
                ba.call_method1("__setitem__", (0, 0)).unwrap();
            }
        });
    }

    /// Stage G (W8): ONE batch latch is the barrier of every export of a `put_many` batch. It
    /// reads as dropped/released only when the batch is sealed and EVERY export is dropped /
    /// released; an item that is snapshotted or immutable adds nothing; the exports of a failed
    /// batch are all released.
    #[test]
    fn batch_latch_covers_every_export_of_the_batch() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "a = bytearray(b'a' * 4096)\n\
                     b = bytearray(b'b' * 4096)\n\
                     frozen = b'immutable'"
                ),
            );
            let a = l.get_item("a").unwrap().unwrap();
            let b = l.get_item("b").unwrap().unwrap();
            let frozen = l.get_item("frozen").unwrap().unwrap();
            let batch = Latch::new_batch();
            let pa = put_payload_in_batch(py, &a, PayloadUse::Blocking, &batch).unwrap();
            let pb = put_payload_in_batch(py, &b, PayloadUse::Blocking, &batch).unwrap();
            let pf = put_payload_in_batch(py, &frozen, PayloadUse::Blocking, &batch).unwrap();
            assert_eq!(pa.as_ptr(), exported_ptr(&a), "item a was copied");
            assert_eq!(pb.as_ptr(), exported_ptr(&b), "item b was copied");
            assert_eq!(batch.batch_len(), 2, "bytes adds no export to the batch");
            // Not sealed yet: nothing is complete, whatever has been dropped.
            assert_eq!(latch_state(&batch), (false, false));
            batch.batch_seal();
            assert_eq!(latch_state(&batch), (false, false), "both exports are held");
            assert!(!resizable(&a) && !resizable(&b), "both buffers are locked");
            drop(pa);
            assert!(!latch_ready(py, &batch), "one export is still held");
            assert!(!resizable(&b));
            drop(pb);
            assert!(latch_ready(py, &batch), "all exports dropped and released");
            assert_eq!(latch_state(&batch), (true, true));
            assert!(resizable(&a) && resizable(&b), "both released");
            drop(pf);
        });
    }

    /// A sealed batch with no export is ready at once; a snapshot item adds nothing.
    #[test]
    fn empty_batch_latch_is_ready_when_sealed() {
        init();
        Python::attach(|py| {
            let batch = Latch::new_batch();
            assert!(!latch_ready(py, &batch), "an open batch is never ready");
            let ba = PyByteArray::new(py, &[1u8; 64]);
            let got = put_payload_in_batch(py, ba.as_any(), PayloadUse::SnapshotUnderGil, &batch)
                .unwrap();
            assert_ne!(got.as_ptr(), exported_ptr(ba.as_any()));
            batch.batch_seal();
            assert_eq!(batch.batch_len(), 0);
            assert!(latch_ready(py, &batch));
        });
    }

    /// A non-contiguous view is gathered in logical order with ONE copy (the `bytes` Python
    /// made is the owner), for any item type.
    #[test]
    fn non_contiguous_view_is_gathered_in_logical_order() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "import array\n\
                     mv = memoryview(bytearray(range(16)))[::2]\n\
                     mvh = memoryview(array.array('H', range(8)))[::2]"
                ),
            );
            let mv = l.get_item("mv").unwrap().unwrap();
            let mvh = l.get_item("mvh").unwrap().unwrap();
            for mode in BOTH {
                let got = put_payload_from_py(py, &mv, mode).unwrap();
                assert_eq!(got.as_ref(), &[0u8, 2, 4, 6, 8, 10, 12, 14]);
                let got = put_payload_from_py(py, &mvh, mode).unwrap();
                let expect: Vec<u8> = [0u16, 2, 4, 6]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect();
                assert_eq!(got.as_ref(), expect.as_slice());
            }
        });
    }

    /// Last resort: `__bytes__()` results are wrapped, not copied a second time.
    #[test]
    fn dunder_bytes_result_is_wrapped_not_copied() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "class C:\n    def __init__(self):\n        self.b = bytes(range(200))\n    def __bytes__(self):\n        return self.b\nc = C()\nbad = object()"
                ),
            );
            let c = l.get_item("c").unwrap().unwrap();
            let inner = c.getattr("b").unwrap();
            let want = inner.cast::<PyBytes>().unwrap().as_bytes().as_ptr();
            let got = put_payload_from_py(py, &c, PayloadUse::Blocking).unwrap();
            assert_eq!(got.as_ptr(), want, "__bytes__ result was copied again");
            let bad = l.get_item("bad").unwrap().unwrap();
            assert!(put_payload_from_py(py, &bad, PayloadUse::Blocking).is_err());
        });
    }

    #[test]
    fn empty_buffer_is_empty_and_releases_its_export() {
        init();
        Python::attach(|py| {
            for mode in BOTH {
                let empty = PyByteArray::new(py, &[]);
                assert!(put_payload_from_py(py, empty.as_any(), mode)
                    .unwrap()
                    .is_empty());
                assert!(resizable(empty.as_any()), "{mode:?}");
            }
        });
    }

    // ---- round 3 -----------------------------------------------------------------------

    /// Dropping an export must never infer GIL ownership: it queues, and only an explicit
    /// drain (with a `py` token) releases it. A thread that does hold the GIL gets no special
    /// inline path, so a wrong guess about the GIL cannot release without it.
    #[test]
    fn dropping_with_the_gil_held_still_only_queues() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Blocking).unwrap();
            drop(got); // this thread holds the GIL
                       // Released only when something drains; the grace period of the releaser thread
                       // cannot run before this thread lets go of the GIL.
            assert!(
                !resizable(ba.as_any()),
                "dropped export was released without a drain"
            );
            drain_pending_exports(py);
            assert!(resizable(ba.as_any()));
        });
    }

    fn leak_gate(closed: bool) -> &'static AttachGate {
        let gate: &'static AttachGate = Box::leak(Box::new(AttachGate::new()));
        if closed {
            assert!(gate.close_and_wait(Duration::from_secs(1)));
        }
        gate
    }

    /// The attach gate: a closed gate refuses and counts nothing; a scope in flight is waited for
    /// by the closer (and its wait is counted); a scope that never ends makes the wait give up
    /// at its bound (counted) instead of hanging.
    #[test]
    fn attach_gate_refuses_after_close_and_closing_waits_for_a_scope_in_flight() {
        let gate = leak_gate(false);
        assert!(gate.close_and_wait(Duration::from_millis(10)), "idle gate");
        assert_eq!(gate.waits.load(Ordering::Relaxed), 0);
        assert!(gate.enter().is_none(), "closed gate admitted a scope");
        assert_eq!(gate.in_flight.load(Ordering::SeqCst), 0, "refusal counted");

        let gate = leak_gate(false);
        let scope = gate.enter().expect("open gate");
        let started = Instant::now();
        let closer = std::thread::spawn(move || gate.close_and_wait(Duration::from_secs(30)));
        std::thread::sleep(Duration::from_millis(150));
        assert!(!closer.is_finished(), "closer did not wait for the scope");
        assert!(gate.enter().is_none(), "closing gate admitted a new scope");
        drop(scope);
        assert!(
            closer.join().unwrap(),
            "closer gave up although the scope ended"
        );
        assert!(started.elapsed() >= Duration::from_millis(140));
        assert_eq!(gate.waits.load(Ordering::Relaxed), 1);
        assert_eq!(gate.wait_timeouts.load(Ordering::Relaxed), 0);

        let gate = leak_gate(false);
        let _stuck = gate.enter().expect("open gate");
        assert!(
            !gate.close_and_wait(Duration::from_millis(50)),
            "must be bounded"
        );
        assert_eq!(gate.wait_timeouts.load(Ordering::Relaxed), 1);
    }

    /// Round 3 / item 3, round 5: once shutdown is flagged the releaser thread never attaches,
    /// but an export is still QUEUED (never forgotten) so a thread that holds a real GIL token -
    /// a `put_bytes` finishing during exit - can release it.
    #[test]
    fn after_shutdown_the_releaser_never_attaches_but_a_token_holder_still_releases() {
        init();
        let flag = leak_gate(true);
        let queue = ReleaseQueue::start(Duration::from_millis(5), flag, true);
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let (owner, latch) = ExportOwner::with_queue(export, queue);
            let bytes = Bytes::from_owner(owner);
            py.detach(move || drop(bytes));
            std::thread::sleep(Duration::from_millis(100));
            assert_eq!(queue.count.load(Ordering::Acquire), 1, "must stay queued");
            assert_eq!(
                queue.attaches.load(Ordering::Relaxed),
                0,
                "releaser attached"
            );
            assert!(
                !resizable(ba.as_any()),
                "nobody with a token has drained yet"
            );
            queue.drain(py); // a thread holding the GIL token releases it
            assert!(latch.is_released());
            assert!(resizable(ba.as_any()));
        });
    }

    /// The global queue is not even given a releaser thread once shutdown has begun.
    #[test]
    fn queue_started_after_shutdown_has_no_releaser_thread() {
        init();
        let flag = leak_gate(true);
        let queue = ReleaseQueue::start(Duration::from_millis(1), flag, false);
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let (owner, _l) = ExportOwner::with_queue(export, queue);
            py.detach(move || drop(Bytes::from_owner(owner)));
            std::thread::sleep(Duration::from_millis(60));
            assert_eq!(queue.attaches.load(Ordering::Relaxed), 0);
            assert_eq!(queue.count.load(Ordering::Acquire), 1);
            queue.drain(py);
            assert!(resizable(ba.as_any()));
        });
    }

    /// Item 4: a caller that drains within the grace period leaves the releaser nothing to do:
    /// it never takes the GIL.
    #[test]
    fn caller_draining_within_the_grace_period_means_no_releaser_gil_traffic() {
        init();
        let queue = ReleaseQueue::start(Duration::from_millis(300), leak_gate(false), true);
        Python::attach(|py| {
            for _ in 0..20 {
                let ba = PyByteArray::new(py, &[1u8; 4096]);
                let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
                let (owner, _latch) = ExportOwner::with_queue(export, queue);
                let bytes = Bytes::from_owner(owner);
                py.detach(move || drop(bytes));
                queue.drain(py);
                assert!(resizable(ba.as_any()));
            }
        });
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(
            queue.attaches.load(Ordering::Relaxed),
            0,
            "the releaser took the GIL although every export was drained by its caller"
        );
    }

    /// And with nobody draining, the releaser does release (after the grace period).
    #[test]
    fn releaser_releases_after_the_grace_period_when_nobody_drains() {
        init();
        let queue = ReleaseQueue::start(Duration::from_millis(20), leak_gate(false), true);
        let ba = Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let (owner, _latch) = ExportOwner::with_queue(export, queue);
            let bytes = Bytes::from_owner(owner);
            py.detach(move || drop(bytes));
            ba.unbind()
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !Python::attach(|py| resizable(ba.bind(py).as_any())) {
            assert!(Instant::now() < deadline, "never released");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(queue.attaches.load(Ordering::Relaxed) >= 1);
    }

    /// Item 4: the queue keeps its capacity across drains (no reallocation per push).
    #[test]
    fn queue_keeps_its_capacity_across_drains() {
        init();
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_gate(false), true);
        Python::attach(|py| {
            for round in 0..4 {
                for _ in 0..8 {
                    let ba = PyByteArray::new(py, &[1u8; 64]);
                    let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
                    queue.push(export);
                    std::mem::forget(ba); // keep the exporter alive until the drain
                }
                queue.drain(py);
                if round > 0 {
                    assert!(
                        queue.lock().capacity() >= 8,
                        "queue capacity lost after drain {round}"
                    );
                }
            }
        });
    }

    /// Item 7: arrays of objects or pointers are refused, struct field names are not mistaken
    /// for formats.
    #[test]
    fn buffers_of_objects_or_pointers_are_refused() {
        init();
        assert!(format_has_pointers("O"));
        assert!(format_has_pointers("<P"));
        assert!(format_has_pointers("T{<i:x:<P:p:}"));
        assert!(!format_has_pointers("T{<i:Pos:<B:Obj:}"));
        assert!(!format_has_pointers("<f"));
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "import ctypes\nptrs = (ctypes.c_void_p * 8)()\nobjs = (ctypes.py_object * 2)()\nok = (ctypes.c_double * 8)()"
                ),
            );
            for name in ["ptrs", "objs"] {
                let obj = l.get_item(name).unwrap().unwrap();
                for mode in BOTH {
                    let err = put_payload_from_py(py, &obj, mode).expect_err(name);
                    assert!(
                        err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                        "{name}"
                    );
                }
                assert!(resizable_len_check(&obj), "{name} export leaked");
            }
            let ok = l.get_item("ok").unwrap().unwrap();
            assert!(put_payload_from_py(py, &ok, PayloadUse::Blocking).is_ok());
        });
    }

    /// The export of a refused buffer is released (here: a memoryview over a pointer array
    /// can be created and released again without error).
    fn resizable_len_check(obj: &Bound<'_, PyAny>) -> bool {
        obj.py()
            .import("builtins")
            .and_then(|b| b.call_method1("memoryview", (obj,)))
            .and_then(|mv| mv.call_method0("release"))
            .is_ok()
    }

    /// #282: EVERY pointer-bearing format code is refused, not only `O` and `P`. The formats are
    /// the ones CPython reports for real ctypes arrays: `POINTER(T)` is `&<T>`, `c_char_p` is
    /// `z`, `c_wchar_p` is `Z`, a function pointer is `X{}`; a structure with a pointer field is
    /// `T{...}`. Every put conversion refuses them in every mode, and nothing stays exported;
    /// a read destination refuses them with the destination wording.
    #[test]
    fn every_pointer_bearing_ctypes_buffer_is_refused() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "import ctypes\n\
                     S = lambda base, name, *fields: type(name, (base,), {'_fields_': list(fields)})\n\
                     WithPointer = S(ctypes.Structure, 'WithPointer', ('n', ctypes.c_int), ('p', ctypes.POINTER(ctypes.c_int)))\n\
                     WithText = S(ctypes.Structure, 'WithText', ('n', ctypes.c_int), ('s', ctypes.c_char_p))\n\
                     Plain = S(ctypes.Structure, 'Plain', ('n', ctypes.c_int), ('x', ctypes.c_double))\n\
                     bad = {\n\
                       'POINTER': (ctypes.POINTER(ctypes.c_int) * 4)(),\n\
                       'c_char_p': (ctypes.c_char_p * 4)(),\n\
                       'c_wchar_p': (ctypes.c_wchar_p * 4)(),\n\
                       'py_object': (ctypes.py_object * 2)(),\n\
                       'c_void_p': (ctypes.c_void_p * 4)(),\n\
                       'function_pointer': (ctypes.CFUNCTYPE(None) * 2)(),\n\
                       'struct_with_pointer': (WithPointer * 2)(),\n\
                       'struct_with_char_p': (WithText * 2)(),\n\
                     }\n\
                     good = {\n\
                       'c_ubyte': (ctypes.c_ubyte * 8)(),\n\
                       'c_byte': (ctypes.c_byte * 8)(),\n\
                       'c_char': (ctypes.c_char * 8)(),\n\
                       'c_short': (ctypes.c_short * 8)(),\n\
                       'c_uint': (ctypes.c_uint * 8)(),\n\
                       'c_float': (ctypes.c_float * 8)(),\n\
                       'c_double': (ctypes.c_double * 8)(),\n\
                       'c_bool': (ctypes.c_bool * 8)(),\n\
                       'c_wchar': (ctypes.c_wchar * 8)(),\n\
                       'plain_struct': (Plain * 2)(),\n\
                     }"
                ),
            );
            let bad = l.get_item("bad").unwrap().unwrap();
            let bad = bad.cast::<PyDict>().unwrap();
            for (name, obj) in bad.iter() {
                let name: String = name.extract().unwrap();
                for mode in BOTH {
                    let err = put_payload_from_py(py, &obj, mode)
                        .err()
                        .unwrap_or_else(|| panic!("{name} ({mode:?}) was accepted for upload"));
                    assert!(
                        err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                        "{name} ({mode:?}): {err}"
                    );
                    assert!(
                        err.to_string().contains("cannot upload a buffer"),
                        "{name} ({mode:?}): {err}"
                    );
                }
                assert!(
                    snapshot_for_call(py, &obj).is_err(),
                    "{name}: snapshot_for_call accepted it"
                );
                assert!(
                    hold_for_call(py, &obj).is_err(),
                    "{name}: hold_for_call accepted it"
                );
                assert!(resizable_len_check(&obj), "{name}: export leaked");

                let latch = Latch::new();
                let err = acquire_writable(py, &obj, 4, &latch)
                    .err()
                    .unwrap_or_else(|| panic!("{name}: accepted as a read destination"));
                assert!(
                    err.to_string().contains("read destination"),
                    "{name}: {err}"
                );
            }
            let good = l.get_item("good").unwrap().unwrap();
            let good = good.cast::<PyDict>().unwrap();
            for (name, obj) in good.iter() {
                for mode in BOTH {
                    put_payload_from_py(py, &obj, mode)
                        .unwrap_or_else(|e| panic!("{name} ({mode:?}) was refused: {e}"));
                }
                drain_pending_exports(py);
            }
        });
    }

    /// #282: the format scan, on format strings: pointer codes anywhere (also after byte-order
    /// prefixes and inside `T{...}`), field names skipped, complex `Zf`/`Zd`/`Zg` and plain
    /// codes untouched.
    #[test]
    fn format_scan_names_every_pointer_code_and_nothing_else() {
        for fmt in [
            "O",
            "<O",
            "P",
            "<P",
            "&<i",
            "&i",
            "<z",
            "z",
            "<Z",
            "Z",
            "X{}",
            "T{<i:a:4x<P:p:}",
            "T{<i:a:4x&<i:p:}",
            "T{<i:a:<z:s:}",
            "T{<i:a:<Z:s:}",
            "T{<i:a:X{}:f:}",
            "3<P",
            "T{<Zd:c:<P:p:}",
            "4P",
            "3<P",
            "(2,3)P",
            "T{T{<i:a:<P:p:}:b:}",
        ] {
            assert!(format_has_pointers(fmt), "{fmt:?} must be refused");
        }
        for fmt in [
            "B",
            "b",
            "c",
            "<B",
            "<H",
            "<I",
            "<q",
            "<f",
            "<d",
            "<?",
            "<u",
            "<e",
            "<Zf",
            "Zd",
            "<Zg",
            "10s",
            "T{<i:a:4x<d:x:}",
            "T{<i:Pos:<B:Obj:}",
            "T{<i:zed:<Zd:Zeta:}",
            "2x",
            "T{<i:X:}",
            // lowercase `p` is a Pascal string (length byte + characters), not a pointer
            "p",
            "<p",
            "10p",
            "T{<i:a:8p:s:}",
        ] {
            assert!(!format_has_pointers(fmt), "{fmt:?} must stay valid");
        }
    }

    /// #277: a pointer-format buffer is refused as a READ destination with wording about a
    /// destination, not an upload; the PUT wording is unchanged. Both are `TypeError`s and nothing
    /// stays exported.
    #[test]
    fn pointer_format_refusal_wording_depends_on_the_direction() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!("import ctypes\nptrs = (ctypes.c_void_p * 4)()\nobjs = (ctypes.py_object * 2)()"),
            );
            for name in ["ptrs", "objs"] {
                let obj = l.get_item(name).unwrap().unwrap();

                let latch = Latch::new();
                let err = acquire_writable(py, &obj, 4, &latch)
                    .err()
                    .unwrap_or_else(|| panic!("{name}: a pointer destination must be refused"));
                assert!(
                    err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                    "{name}: {err}"
                );
                let text = err.to_string();
                assert!(
                    !text.contains("upload"),
                    "{name}: a read destination refusal talks about uploading: {text}"
                );
                assert!(
                    text.contains("read destination"),
                    "{name}: the refusal does not say it is a read destination: {text}"
                );
                assert!(resizable_len_check(&obj), "{name}: export leaked");

                let err = put_payload_from_py(py, &obj, PayloadUse::Blocking)
                    .expect_err("a pointer buffer must not be uploaded");
                assert!(
                    err.to_string().contains("cannot upload a buffer"),
                    "{name}: the PUT wording changed: {err}"
                );
            }
        });
    }

    /// #278: `release_retained_for_test` also disarms a pending `arm_retain_for_test`, so a test
    /// that armed the hook and failed before any put consumed it cannot make the NEXT test's put
    /// retain a clone forever.
    #[test]
    fn release_retained_disarms_a_pending_arming() {
        init();
        Python::attach(|py| {
            let l = ns(py, c_str!("buf = bytearray(b'abcdefgh')"));
            let buf = l.get_item("buf").unwrap().unwrap();

            arm_retain_for_test();
            release_retained_for_test();
            let payload = put_payload(py, &buf, PayloadUse::Blocking).expect("payload");
            let retained = retained_for_test();

            // Leave nothing behind whatever the outcome: drop any clone, then the payload.
            release_retained_for_test();
            drop(payload);
            drain_pending_exports(py);

            assert_eq!(
                retained, 0,
                "a pending arming survived release_retained_for_test: the next put retained a clone"
            );
            assert!(resizable(&buf), "export leaked");
        });
    }

    /// Item 7: a failed snapshot allocation is a MemoryError, not an abort.
    #[test]
    fn snapshot_allocation_failure_is_a_memory_error() {
        assert!(try_snapshot(&[1, 2, 3], usize::MAX).is_err());
        assert_eq!(try_snapshot(&[1, 2, 3], 3).unwrap().as_ref(), &[1, 2, 3]);
    }

    /// The snapshot `MemoryError` is shared by `put_many` / `MultiEndpointStore` / the writers, so
    /// its wording must not name `put_bytes_async`.
    #[test]
    fn snapshot_memory_error_wording_is_generic() {
        init();
        Python::attach(|py| {
            let e = snapshot_allocation_error(12345);
            assert!(e.is_instance_of::<pyo3::exceptions::PyMemoryError>(py));
            let text = e.value(py).str().unwrap().to_string();
            assert!(text.contains("12345 bytes"), "{text}");
            assert!(text.contains("payload snapshot"), "{text}");
            assert!(!text.contains("put_bytes_async"), "{text}");
        });
    }

    /// Item 5: a small snapshot is copied with the GIL held (no detach); big ones detach.
    /// Both are correct and independent of the source.
    #[test]
    fn snapshot_sizes_around_the_detach_threshold() {
        init();
        Python::attach(|py| {
            for len in [
                1,
                SNAPSHOT_GIL_DETACH_THRESHOLD - 1,
                SNAPSHOT_GIL_DETACH_THRESHOLD,
                SNAPSHOT_GIL_DETACH_THRESHOLD + 1,
            ] {
                let ba = PyByteArray::new(py, &vec![7u8; len]);
                let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Snapshot).unwrap();
                assert_eq!(got.len(), len);
                assert!(got.iter().all(|&b| b == 7));
                assert!(resizable(ba.as_any()), "export leaked at {len}");
            }
        });
    }

    // ---- round 4: the completion barrier --------------------------------------------------

    /// A held export backed by a fresh latch, wrapped as the owner of a `Bytes`.
    fn borrowed(py: Python<'_>, ba: &Bound<'_, PyByteArray>) -> Payload {
        put_payload(py, ba.as_any(), PayloadUse::Borrowed).unwrap()
    }

    /// The owner's drop (the last clone dying ANYWHERE) is the "all readers done" event, and the
    /// export is released only by a thread holding a real GIL token.
    #[test]
    fn last_clone_dropping_signals_the_latch_and_a_token_holder_releases() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let p = borrowed(py, &ba);
            let latch = p.latch.clone().unwrap();
            let clone = p.bytes.clone();
            drop(p);
            assert!(!latch.is_dropped(), "a live clone is still a native reader");
            py.detach(move || drop(clone)); // last clone dies on a non-GIL thread
            assert!(latch.is_dropped());
            assert!(!latch.is_released(), "release needs a py token");
            assert!(!resizable(ba.as_any()));
            drain_pending_exports(py);
            assert!(latch.is_released());
            assert!(resizable(ba.as_any()));
        });
    }

    /// Retained backend clone: a "store" keeps a clone for 300 ms past the put. The blocking
    /// completion does not return before that clone is gone, and the export is then released.
    #[test]
    fn quiesce_blocks_while_a_backend_retains_a_clone_then_releases() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let p = borrowed(py, &ba);
            let latch = p.latch.clone().unwrap();
            let retained = p.bytes.clone();
            let started = Instant::now();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                drop(retained);
            });
            drop(p); // the "put" itself returned; only the retained clone remains
            quiesce_blocking(py, &latch).expect("not timed out");
            assert!(
                started.elapsed() >= Duration::from_millis(290),
                "returned too early"
            );
            t.join().unwrap();
            assert!(latch.is_released());
            assert!(
                resizable(ba.as_any()),
                "export not released after the barrier"
            );
        });
    }

    /// A backend that never lets go: the wait is bounded, the buffer STAYS pinned (a resize
    /// still fails), the timeout is counted, and the export is released once the clone dies.
    #[test]
    fn quiesce_timeout_keeps_the_buffer_pinned_and_counts() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let p = borrowed(py, &ba);
            let latch = p.latch.clone().unwrap();
            let retained = p.bytes.clone();
            drop(p);
            let before = export_stats().quiesce_timeouts;
            let gave_up = py.detach(|| !latch.wait_dropped_blocking(Duration::from_millis(50)));
            assert!(gave_up);
            note_quiesce_timeout();
            assert_eq!(export_stats().quiesce_timeouts, before + 1);
            assert!(
                !resizable(ba.as_any()),
                "memory still being read was released"
            );
            py.detach(move || drop(retained));
            drain_pending_exports(py);
            assert!(resizable(ba.as_any()));
        });
    }

    /// Error / cancellation before any native work: the Bytes is dropped unread (the future was
    /// dropped before its first poll). The barrier is satisfied immediately.
    #[test]
    fn dropping_the_payload_unread_satisfies_the_barrier() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let p = borrowed(py, &ba);
            let latch = p.latch.clone().unwrap();
            drop(p);
            quiesce_blocking(py, &latch).expect("not timed out");
            assert!(latch.is_released());
            assert!(resizable(ba.as_any()));
        });
    }

    /// The async wait (cancellation path) returns exactly when the last clone dies.
    #[tokio::test]
    async fn async_wait_follows_the_last_clone() {
        init();
        let (latch, clone, ba) = Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let p = borrowed(py, &ba);
            (p.latch.clone().unwrap(), p.bytes.clone(), ba.unbind())
        });
        assert!(!latch.wait_dropped(Duration::from_millis(40)).await);
        let l2 = latch.clone();
        let waiter = tokio::spawn(async move { l2.wait_dropped(Duration::from_secs(10)).await });
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(clone); // may run on any thread
        assert!(waiter.await.unwrap());
        // The async wait never attaches to Python; the delivery path drains. Here: do it by hand.
        assert!(
            !quiesce_async(latch.clone()).await,
            "must not report a timeout"
        );
        Python::attach(|py| {
            assert!(latch_ready(py, &latch));
            assert!(latch.is_released());
            assert!(resizable(ba.bind(py).as_any()));
        });
    }

    /// Subinterpreter hosts: after any subinterpreter was created (which makes
    /// `PyGILState_Check` answer "held" on threads without the GIL) a detached drop must still
    /// only queue. Skipped (with a message) where `_interpreters` is unavailable.
    #[test]
    fn detached_drop_after_a_subinterpreter_was_created_only_queues() {
        init();
        Python::attach(|py| {
            let created = py.run(
                c_str!(
                    "try:\n    import _interpreters as _i\nexcept ImportError:\n    import _xxsubinterpreters as _i\nx = _i.create()\n_i.destroy(x)"
                ),
                None,
                None,
            );
            if let Err(e) = created {
                eprintln!("SKIPPED: no subinterpreter module here ({e})");
                return;
            }
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let p = borrowed(py, &ba);
            let latch = p.latch.clone().unwrap();
            py.detach(move || drop(p)); // a thread WITHOUT the GIL, PyGILState_Check() == 1 now
            assert!(latch.is_dropped());
            assert!(!latch.is_released(), "released without a py token");
            drain_pending_exports(py);
            assert!(latch.is_released());
        });
    }

    // ---- round 5 --------------------------------------------------------------------------

    /// The writers' conversion is STRICT: an int (which `bytes(5)` would turn into five zero
    /// bytes) and an arbitrary iterable are refused; `__bytes__()` objects and buffers are fine.
    #[test]
    fn strict_hold_refuses_ints_and_iterables_but_accepts_bytes_likes() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!(
                    "class B:\n    def __bytes__(self):\n        return b'xyz'\nobjs = {'int': 5, 'list': [1, 2, 3], 'gen': (i for i in range(3)), 'str': 'abc', 'none': None}\nb = B()"
                ),
            );
            let objs = l.get_item("objs").unwrap().unwrap();
            let objs = objs.cast::<PyDict>().unwrap();
            for (name, obj) in objs.iter() {
                let e = hold_for_call(py, &obj)
                    .err()
                    .unwrap_or_else(|| panic!("{name} accepted"));
                assert!(
                    e.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                    "{name}"
                );
            }
            let b = l.get_item("b").unwrap().unwrap();
            assert_eq!(hold_for_call(py, &b).unwrap().as_slice(), b"xyz");
            let ba = PyByteArray::new(py, &[7u8; 64]);
            let held = hold_for_call(py, ba.as_any()).unwrap();
            assert_eq!(held.as_slice(), &[7u8; 64]);
            assert!(!resizable(ba.as_any()), "held");
            held.release(py);
            assert!(
                resizable(ba.as_any()),
                "released directly with the py token"
            );
            // put_bytes uses the same strict conversion (issue #261): no bytes(obj) fallback
            for (name, obj) in objs.iter() {
                let e = put_payload_from_py(py, &obj, PayloadUse::Blocking)
                    .err()
                    .unwrap_or_else(|| panic!("{name} accepted by put_bytes"));
                assert!(
                    e.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                    "{name}"
                );
            }
        });
    }

    /// The direct release of a held buffer does not touch the release queue.
    #[test]
    fn direct_release_does_not_use_the_queue() {
        init();
        Python::attach(|py| {
            let before = export_stats().queued;
            let ba = PyByteArray::new(py, &[7u8; 64]);
            hold_for_call(py, ba.as_any()).unwrap().release(py);
            assert_eq!(export_stats().queued, before);
        });
    }

    /// `latch_ready` is a non-waiting check that releases with the held GIL.
    #[test]
    fn latch_ready_is_false_while_a_clone_lives_and_true_after() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let p = borrowed(py, &ba);
            let latch = p.latch.clone().unwrap();
            assert!(!latch_ready(py, &latch));
            drop(p);
            assert!(latch_ready(py, &latch));
            assert!(latch.is_released());
            assert!(resizable(ba.as_any()));
        });
    }

    /// Round 6, item 5: `latch_ready` needs the export to be really RELEASED, not just the owner
    /// dropped. Here the owner is dropped into a queue that the global drain does not touch.
    #[test]
    fn latch_ready_requires_released_as_well_as_dropped() {
        init();
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_gate(false), true);
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let (owner, latch) = ExportOwner::with_queue(export, queue);
            let bytes = Bytes::from_owner(owner);
            py.detach(move || drop(bytes));
            assert!(latch.is_dropped());
            assert!(
                !latch_ready(py, &latch),
                "ready although the export is still held"
            );
            queue.drain(py); // the queue that actually holds it
            assert!(latch_ready(py, &latch));
            assert!(resizable(ba.as_any()));
        });
    }

    /// A dropped export that another drainer still owns (its release has not run yet): the owner
    /// is dropped, the public queue is empty, `released` is false. R3: the blocking wait must not
    /// return before the actual release, and must wait without the GIL (the other drainer's
    /// release needs it).
    fn dropped_but_unreleased(
        queue: &'static ReleaseQueue,
    ) -> (Arc<Latch>, Py<pyo3::types::PyByteArray>) {
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let (owner, latch) = ExportOwner::with_queue(export, queue);
            let bytes = Bytes::from_owner(owner);
            py.detach(move || drop(bytes)); // dropped; queued in the test queue only
            assert!(latch.is_dropped() && !latch.is_released());
            (latch, ba.unbind())
        })
    }

    #[test]
    fn blocking_quiesce_waits_for_the_actual_release_owned_by_another_drainer() {
        init();
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_gate(false), true);
        let (latch, ba) = dropped_but_unreleased(queue);
        let other_drainer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            Python::attach(|py| queue.drain(py)); // needs the GIL: the waiter must not hold it
        });
        Python::attach(|py| {
            let started = Instant::now();
            quiesce_blocking(py, &latch).expect("the other drainer releases it");
            assert!(
                started.elapsed() >= Duration::from_millis(290),
                "returned before the export was actually released"
            );
            assert!(latch.is_released());
            assert!(resizable(ba.bind(py).as_any()));
        });
        other_drainer.join().unwrap();
    }

    /// P3-2: the last clone died after the caller's `latch_ready()` and the export sits
    /// undrained in the public queue. The first step must come back after a short slice (not the
    /// 30 s deadline, no timeout reported, nothing released by it) so that the delivery drains
    /// and the guard re-checks; it must not depend on the releaser thread.
    #[tokio::test]
    async fn first_already_dropped_step_returns_after_a_short_slice_without_a_timeout() {
        init();
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_gate(false), true);
        let (latch, ba) = dropped_but_unreleased(queue);
        let before = export_stats().quiesce_timeouts;
        let started = Instant::now();
        assert!(
            !quiesce_async(latch.clone()).await,
            "a slice is not a timeout"
        );
        let took = started.elapsed();
        assert!(took >= REDRAIN_SLICE, "returned before the slice: {took:?}");
        assert!(
            took < Duration::from_secs(5),
            "waited the deadline: {took:?}"
        );
        assert!(!latch.is_released(), "nothing drained yet");
        assert_eq!(
            export_stats().quiesce_timeouts,
            before,
            "no timeout counted"
        );
        // What the delivery of the quiesce future does next, then the guard's re-check:
        Python::attach(|py| {
            queue.drain(py);
            assert!(latch.is_released());
            assert!(resizable(ba.bind(py).as_any()));
        });
    }

    #[tokio::test]
    async fn async_quiesce_waits_for_the_actual_release_after_the_owner_dropped() {
        init();
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_gate(false), true);
        let (latch, ba) = dropped_but_unreleased(queue);
        let other_drainer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            Python::attach(|py| queue.drain(py));
        });
        let started = Instant::now();
        // Steps as the guard runs them: re-check after each step (the first "already dropped"
        // step is only a short re-drain slice, the next one waits for the other drainer).
        while !latch.is_released() {
            assert!(
                !quiesce_async(latch.clone()).await,
                "must not report a timeout"
            );
        }
        assert!(
            started.elapsed() >= Duration::from_millis(290),
            "completed before the export was actually released"
        );
        assert!(latch.is_released());
        other_drainer.join().unwrap();
        Python::attach(|py| assert!(resizable(ba.bind(py).as_any())));
    }

    /// The writers' `copy=True`: a mutable exporter is copied once and independent of the source
    /// (the export is released at once, later changes are invisible); exact `bytes` and
    /// `BytesView` are still not copied.
    #[test]
    fn snapshot_for_call_copies_mutable_exporters_but_never_immutable_bytes() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &vec![5u8; 1 << 20]);
            let src = exported_ptr(ba.as_any());
            let got = snapshot_for_call(py, ba.as_any()).unwrap();
            assert_ne!(got.as_ptr(), src, "a mutable exporter must be copied");
            assert!(
                resizable(ba.as_any()),
                "the export must be released at once"
            );
            ba.call_method1("__setitem__", (0, 6)).unwrap();
            assert_eq!(got.len(), 1 << 20);
            assert!(got.iter().all(|&b| b == 5), "snapshot saw a later change");

            let b = PyBytes::new(py, &[9u8; 4096]);
            let got = snapshot_for_call(py, b.as_any()).unwrap();
            assert_eq!(
                got.as_ptr(),
                b.as_bytes().as_ptr(),
                "exact bytes was copied"
            );

            let original = Bytes::from(vec![7u8; 4096]);
            let ptr = original.as_ptr();
            let view = Bound::new(py, PyBytesView::new(original)).unwrap();
            let got = snapshot_for_call(py, view.as_any()).unwrap();
            assert_eq!(got.as_ptr(), ptr, "BytesView was copied");

            // strict: an int is refused, nothing is written
            assert!(snapshot_for_call(py, &5i32.into_pyobject(py).unwrap().into_any()).is_err());
        });
    }

    // ---- writable export (A4, #264) -------------------------------------------------------

    fn latch_state(latch: &Latch) -> (bool, bool) {
        (latch.is_dropped(), latch.is_released())
    }

    /// Objects that must be accepted as a writable destination: name, object, its length in bytes.
    fn writable_objects<'py>(py: Python<'py>) -> Vec<(&'static str, Bound<'py, PyAny>, usize)> {
        let l = ns(
            py,
            c_str!(
                "import array, ctypes, mmap\n\
                 objs = [\n\
                   ('bytearray', bytearray(range(64)), 64),\n\
                   ('memoryview', memoryview(bytearray(64)), 64),\n\
                   ('array_B', array.array('B', range(32)), 32),\n\
                   ('array_b', array.array('b', [1, -2] * 16), 32),\n\
                   ('mv_c', memoryview(bytearray(b'x' * 48)).cast('c'), 48),\n\
                   ('mv_3d', memoryview(bytearray(24)).cast('B', (2, 3, 4)), 24),\n\
                   ('mmap', mmap.mmap(-1, 4096), 4096),\n\
                   ('ctypes_ubyte', (ctypes.c_ubyte * 16)(), 16),\n\
                   ('ctypes_byte', (ctypes.c_byte * 16)(), 16),\n\
                   ('ctypes_char', (ctypes.c_char * 16)(), 16),\n\
                 ]"
            ),
        );
        let objs = l.get_item("objs").unwrap().unwrap();
        objs.try_iter()
            .unwrap()
            .map(|t| {
                let t = t.unwrap();
                let name: String = t.get_item(0).unwrap().extract().unwrap();
                let name: &'static str = Box::leak(name.into_boxed_str());
                (
                    name,
                    t.get_item(1).unwrap(),
                    t.get_item(2).unwrap().extract().unwrap(),
                )
            })
            .collect()
    }

    /// Accepted: the slice IS the exporter's memory (pointer identity), exactly `expected_len`
    /// long; writes through it are visible in the Python object after `release`; the caller's
    /// latch is the one that is marked.
    #[test]
    fn writable_export_accepts_writable_byte_buffers_and_writes_in_place() {
        init();
        Python::attach(|py| {
            let count = writable_objects(py).len();
            for idx in 0..count {
                // A fresh object for every length, so an earlier fill cannot mask a later one.
                for pick in [|n: usize| n, |n: usize| n - 1, |n: usize| n / 2] {
                    let (name, obj, nbytes) = writable_objects(py).swap_remove(idx);
                    let expected = pick(nbytes);
                    let before = bytes_of(py, &obj);
                    let src = exported_ptr(&obj);
                    let latch = Latch::new();
                    let mut ex = acquire_writable(py, &obj, expected, &latch)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                    assert_eq!(latch_state(&latch), (false, false), "{name}: held");
                    let slice = ex.as_mut_slice();
                    assert_eq!(slice.as_ptr(), src, "{name}: not the exporter's memory");
                    assert_eq!(slice.len(), expected, "{name}: length");
                    slice.fill(0xA5);
                    ex.release(py);
                    assert_eq!(latch_state(&latch), (true, true), "{name}: after release");
                    let after = bytes_of(py, &obj);
                    assert!(
                        after[..expected].iter().all(|&b| b == 0xA5),
                        "{name}: the write is not visible in the Python object"
                    );
                    assert_eq!(
                        after[expected..],
                        before[expected..],
                        "{name}: bytes beyond expected_len changed"
                    );
                }
            }
        });
    }

    fn bytes_of(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Vec<u8> {
        py.import("builtins")
            .unwrap()
            .call_method1("bytes", (obj,))
            .unwrap()
            .extract()
            .unwrap()
    }

    /// `expected_len` beyond the exporter's length is refused (ValueError), nothing stays held.
    #[test]
    fn writable_export_refuses_expected_len_beyond_the_buffer() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[0u8; 16]);
            let latch = Latch::new();
            let err = acquire_writable(py, ba.as_any(), 17, &latch)
                .err()
                .expect("17 > 16 must be refused");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert_eq!(latch_state(&latch), (true, true), "export not released");
            assert!(resizable(ba.as_any()), "export leaked");
            // exact length is fine
            let latch = Latch::new();
            let ex = acquire_writable(py, ba.as_any(), 16, &latch).unwrap();
            ex.release(py);
        });
    }

    /// Read-only, non-contiguous, wrong-itemsize/format and pointer-bearing exporters are all
    /// refused with the right exception class, and nothing stays exported.
    #[test]
    fn writable_export_refuses_unsuitable_exporters_and_releases() {
        init();
        Python::attach(|py| {
            use pyo3::exceptions::{PyBufferError, PyTypeError};
            // Third element of every case: a probe that raises (BufferError) if a hold on the
            // exporter is still outstanding - releasing a memoryview, or resizing an array, fails
            // while an export of it exists. The probes bind their target as a default argument
            // because `ns` runs the source with separate locals (no closure over them).
            let l = ns(
                py,
                c_str!(
                    "import array, ctypes\n\
                     base = bytearray(range(32))\n\
                     mv_ro = memoryview(b'abcdefgh')\n\
                     mv_rob = memoryview(base).toreadonly()\n\
                     mv_str = memoryview(base)[::2]\n\
                     mv_rev = memoryview(base)[::-1]\n\
                     a_i = array.array('i', range(8))\n\
                     a_H = array.array('H', range(8))\n\
                     a_d = array.array('d', [0.0] * 4)\n\
                     mv_bool = memoryview(bytearray(8)).cast('?')\n\
                     nop = lambda: None\n\
                     cases = {\n\
                       'bytes': (b'abcdefgh', 'buffer', nop),\n\
                       'ro_memoryview': (mv_ro, 'buffer', lambda o=mv_ro: o.release()),\n\
                       'ro_view_of_bytearray': (mv_rob, 'buffer', lambda o=mv_rob: o.release()),\n\
                       'strided': (mv_str, 'buffer', lambda o=mv_str: o.release()),\n\
                       'reversed': (mv_rev, 'buffer', lambda o=mv_rev: o.release()),\n\
                       'array_i': (a_i, 'type', lambda o=a_i: o.append(1)),\n\
                       'array_H': (a_H, 'type', lambda o=a_H: o.append(1)),\n\
                       'array_d': (a_d, 'type', lambda o=a_d: o.append(1.0)),\n\
                       'bool_bytes': (mv_bool, 'type', lambda o=mv_bool: o.release()),\n\
                       'c_void_p': ((ctypes.c_void_p * 4)(), 'type', nop),\n\
                       'py_object': ((ctypes.py_object * 2)(), 'type', nop),\n\
                       'not_a_buffer': (object(), 'type', nop),\n\
                       'int': (5, 'type', nop),\n\
                     }"
                ),
            );
            let cases = l.get_item("cases").unwrap().unwrap();
            let cases = cases.cast::<PyDict>().unwrap();
            let getrefcount = py.import("sys").unwrap().getattr("getrefcount").unwrap();
            let refcount = |obj: &Bound<'_, PyAny>| -> usize {
                getrefcount.call1((obj,)).unwrap().extract().unwrap()
            };
            for (name, entry) in cases.iter() {
                let name: String = name.extract().unwrap();
                let obj = entry.get_item(0).unwrap();
                let kind: String = entry.get_item(1).unwrap().extract().unwrap();
                let probe = entry.get_item(2).unwrap();
                let refs_before = refcount(&obj);
                let latch = Latch::new();
                let err = acquire_writable(py, &obj, 4, &latch)
                    .err()
                    .unwrap_or_else(|| panic!("{name} must be refused"));
                // The exporter's own refusal of a writable / contiguous request is a
                // BufferError; our own format/itemsize refusals are TypeErrors.
                let ok = if kind == "buffer" {
                    err.is_instance_of::<PyBufferError>(py)
                } else {
                    err.is_instance_of::<PyTypeError>(py)
                };
                assert!(ok, "{name}: wrong exception class: {err}");
                assert_eq!(latch_state(&latch), (true, true), "{name}: latch");
                // Nothing leaked, for EVERY refused exporter: the reference a held `Py_buffer`
                // keeps on its exporter is gone, and the exporter is still releasable/resizable.
                assert_eq!(
                    refcount(&obj),
                    refs_before,
                    "{name}: a refused export kept a reference to the exporter"
                );
                probe
                    .call0()
                    .unwrap_or_else(|e| panic!("{name}: a refused export leaked a hold: {e}"));
            }
            // The bytearray behind several of the memoryviews is resizable again as well.
            let base = l.get_item("base").unwrap().unwrap();
            assert!(
                resizable(&base),
                "a refused export leaked a hold on the bytearray"
            );
        });
    }

    /// A refusal after a successful `PyObject_GetBuffer` (wrong format) releases the export that
    /// was taken: the plain `bytearray` behind it is resizable again at once.
    #[test]
    fn writable_export_refusal_after_acquisition_releases_the_export() {
        init();
        Python::attach(|py| {
            let l = ns(
                py,
                c_str!("ba = bytearray(8)\nmv = memoryview(ba).cast('?')"),
            );
            let ba = l.get_item("ba").unwrap().unwrap();
            let mv = l.get_item("mv").unwrap().unwrap();
            let latch = Latch::new();
            // The memoryview holds one export of `ba` itself; ours must add and drop another.
            assert!(acquire_writable(py, &mv, 4, &latch).is_err());
            assert_eq!(latch_state(&latch), (true, true));
            mv.call_method0("release").unwrap();
            assert!(resizable(&ba), "the refused export was not released");
        });
    }

    /// Zero length: accepted, an empty slice, released cleanly. The slice is not built from the
    /// exporter's pointer (which may be null) when the length is 0.
    #[test]
    fn writable_export_zero_length_gives_an_empty_slice() {
        init();
        Python::attach(|py| {
            // an empty exporter
            let empty = PyByteArray::new(py, &[]);
            let latch = Latch::new();
            let mut ex = acquire_writable(py, empty.as_any(), 0, &latch).unwrap();
            assert!(ex.as_mut_slice().is_empty());
            assert!(!resizable(empty.as_any()), "an empty export is still held");
            ex.release(py);
            assert_eq!(latch_state(&latch), (true, true));
            assert!(resizable(empty.as_any()));

            // expected_len 0 on a non-empty exporter: still empty, memory untouched
            let ba = PyByteArray::new(py, &[7u8; 16]);
            let latch = Latch::new();
            let mut ex = acquire_writable(py, ba.as_any(), 0, &latch).unwrap();
            assert_eq!(ex.as_mut_slice().len(), 0);
            ex.release(py);
            let seen: Vec<u8> = ba.to_vec();
            assert_eq!(seen, vec![7u8; 16]);

            // A view whose address is NULL with length 0 (what some exporters report): the
            // checks accept it and no slice is built from the null pointer.
            // SAFETY: an all-zero Py_buffer is a valid empty value; `obj` is NULL, which
            // `PyBuffer_Release` treats as nothing to release.
            let mut view: Box<ffi::Py_buffer> = Box::new(unsafe { std::mem::zeroed() });
            view.itemsize = 1;
            assert!(view.buf.is_null() && view.len == 0);
            check_writable_view(&view, 0).expect("null buf with len 0 is valid");
            let latch = Latch::new();
            let raw = RawExport { view, latch: None };
            let mut owner = ExportOwnerMut::new(py, raw, latch.clone(), 0)
                .expect("the constructor accepts a null buf with len 0");
            assert!(owner.as_mut_slice().is_empty());
            owner.release(py);
            assert_eq!(latch_state(&latch), (true, true));
        });
    }

    /// Data with a null address is refused by the checks; a negative reported length too.
    #[test]
    fn writable_view_checks_refuse_null_data_and_bad_lengths() {
        init();
        // SAFETY: an all-zero Py_buffer is a valid empty value.
        let mut view: Box<ffi::Py_buffer> = Box::new(unsafe { std::mem::zeroed() });
        view.itemsize = 1;
        view.len = 8;
        assert!(check_writable_view(&view, 4).is_err(), "null buf, len 8");
        view.len = -1;
        assert!(check_writable_view(&view, 0).is_err(), "negative len");
        view.len = 0;
        view.readonly = 1;
        assert!(check_writable_view(&view, 0).is_err(), "read-only");
    }

    /// Refusals AFTER a view was handed over, driven through hand-built `Py_buffer`s (so the
    /// exporter cannot pre-empt them): read-only, suboffsets, and non-contiguous shapes are all
    /// refused with a BufferError, a flat view and a C-contiguous multi-dimensional one are
    /// accepted (so the contiguity defense does not over-refuse), and the constructor refuses the
    /// same views itself, releasing the export (latch released, then dropped): no
    /// `ExportOwnerMut` exists for a view that fails the checks.
    #[test]
    fn writable_view_checks_refuse_readonly_suboffsets_and_non_contiguous_views() {
        use pyo3::exceptions::PyBufferError;
        init();
        let mut mem = [0u8; 4];
        let mut shape1: [ffi::Py_ssize_t; 1] = [4];
        let mut stride_gap: [ffi::Py_ssize_t; 1] = [2];
        let mut shape2: [ffi::Py_ssize_t; 2] = [2, 2];
        let mut strides_c: [ffi::Py_ssize_t; 2] = [2, 1];
        let mut strides_f: [ffi::Py_ssize_t; 2] = [1, 2];
        let mut suboffsets: [ffi::Py_ssize_t; 1] = [0];
        // A valid flat 4-byte view of `mem`; the tests below change one property each. The arrays
        // above outlive every view built here.
        let mut flat = |mem: &mut [u8; 4]| -> Box<ffi::Py_buffer> {
            // SAFETY: an all-zero Py_buffer is a valid empty value; `obj` stays NULL, which
            // `PyBuffer_Release` treats as nothing to release.
            let mut view: Box<ffi::Py_buffer> = Box::new(unsafe { std::mem::zeroed() });
            view.buf = mem.as_mut_ptr().cast();
            view.len = 4;
            view.itemsize = 1;
            view.ndim = 1;
            view.shape = shape1.as_mut_ptr();
            view
        };
        let refused = |view: &ffi::Py_buffer, what: &str| {
            Python::attach(|py| {
                let err = check_writable_view(view, 4)
                    .err()
                    .unwrap_or_else(|| panic!("{what} must be refused"));
                assert!(err.is_instance_of::<PyBufferError>(py), "{what}: {err}");
            });
        };

        let ok = flat(&mut mem);
        check_writable_view(&ok, 4).expect("a flat writable view is valid");

        let mut ro = flat(&mut mem);
        ro.readonly = 1;
        refused(&ro, "read-only with data");

        let mut sub = flat(&mut mem);
        sub.suboffsets = suboffsets.as_mut_ptr();
        refused(&sub, "suboffsets");

        let mut gap = flat(&mut mem);
        gap.strides = stride_gap.as_mut_ptr();
        refused(&gap, "stride 2 over 1-byte items");

        let mut fortran = flat(&mut mem);
        fortran.ndim = 2;
        fortran.shape = shape2.as_mut_ptr();
        fortran.strides = strides_f.as_mut_ptr();
        refused(&fortran, "Fortran-ordered 2-D");

        let mut c2 = flat(&mut mem);
        c2.ndim = 2;
        c2.shape = shape2.as_mut_ptr();
        c2.strides = strides_c.as_mut_ptr();
        check_writable_view(&c2, 4).expect("a C-contiguous 2-D view is valid");

        // The constructor runs the same checks and releases what it was given on a refusal.
        for (what, view) in [("read-only", ro), ("suboffsets", sub), ("Fortran", fortran)] {
            Python::attach(|py| {
                let latch = Latch::new();
                let raw = RawExport { view, latch: None };
                let err = ExportOwnerMut::new(py, raw, latch.clone(), 4)
                    .err()
                    .unwrap_or_else(|| panic!("{what}: the constructor must refuse"));
                assert!(err.is_instance_of::<PyBufferError>(py), "{what}: {err}");
                assert_eq!(latch_state(&latch), (true, true), "{what}: latch");
            });
        }
        let latch = Latch::new();
        Python::attach(|py| {
            let raw = RawExport {
                view: c2,
                latch: None,
            };
            let mut owner = ExportOwnerMut::new(py, raw, latch.clone(), 4)
                .expect("the constructor accepts a C-contiguous 2-D view");
            assert_eq!(owner.as_mut_slice().len(), 4);
            assert_eq!(latch_state(&latch), (false, false), "held");
            owner.release(py);
        });
        assert_eq!(latch_state(&latch), (true, true));
    }

    /// The format rule of A4 through hand-built `Py_buffer`s: `B`, `b` and `c`, each with at most
    /// ONE leading byte-order/alignment character (`@ = < > !`), are accepted (what a ctypes byte
    /// array reports, matching the borrowed PUT path); a bare prefix, two prefixes, and every other
    /// format are refused with a TypeError.
    #[test]
    fn writable_view_checks_format_accepts_one_byte_order_prefix_with_byte_codes() {
        use pyo3::exceptions::PyTypeError;
        init();
        let mut mem = [0u8; 4];
        let mut shape1: [ffi::Py_ssize_t; 1] = [4];
        let mut view_with = |format: &'static std::ffi::CStr| -> Box<ffi::Py_buffer> {
            // SAFETY: an all-zero Py_buffer is a valid empty value; `obj` stays NULL.
            let mut view: Box<ffi::Py_buffer> = Box::new(unsafe { std::mem::zeroed() });
            view.buf = mem.as_mut_ptr().cast();
            view.len = 4;
            view.itemsize = 1;
            view.ndim = 1;
            view.shape = shape1.as_mut_ptr();
            view.format = format.as_ptr().cast_mut();
            view
        };
        for fmt in [
            c"B", c"b", c"c", c"@B", c"@b", c"@c", c"=B", c"=b", c"=c", c"<B", c"<b", c"<c", c">B",
            c">b", c">c", c"!B", c"!b", c"!c",
        ] {
            let view = view_with(fmt);
            check_writable_view(&view, 4)
                .unwrap_or_else(|e| panic!("format {fmt:?} must be accepted: {e}"));
        }
        for fmt in [
            c"?", c"<?", c"x", c"e", c"i", c"H", c"<<B", c"@<B", c"BB", c"<BB", c"<", c"@", c"!",
            c"", c"B<", c"<H", c"s", c"<s", c"2B",
        ] {
            let view = view_with(fmt);
            Python::attach(|py| {
                let err = check_writable_view(&view, 4)
                    .err()
                    .unwrap_or_else(|| panic!("format {fmt:?} must be refused"));
                assert!(err.is_instance_of::<PyTypeError>(py), "{fmt:?}: {err}");
            });
        }
        // A pointer-bearing format keeps its own refusal (also a TypeError), with or without a prefix.
        for fmt in [c"O", c"<P"] {
            let view = view_with(fmt);
            assert!(check_writable_view(&view, 4).is_err(), "{fmt:?}");
        }
    }

    /// `release(py)` is direct: latch released and dropped at once, no queue, and the exporter is
    /// resizable immediately (a live export would make `extend` raise BufferError).
    #[test]
    fn writable_export_release_is_direct_and_immediate() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let latch = Latch::new();
            let ex = acquire_writable(py, ba.as_any(), 4096, &latch).unwrap();
            let err = ba
                .call_method1("extend", (b"x".as_slice(),))
                .expect_err("resize must fail while the destination is exported");
            assert!(err.is_instance_of::<pyo3::exceptions::PyBufferError>(py));
            ex.release(py);
            assert_eq!(latch_state(&latch), (true, true));
            assert!(resizable(ba.as_any()), "release(py) left the export held");
        });
    }

    /// The Drop path (no GIL token): dropping only marks the latch dropped and queues the
    /// export; the export stays held until a GIL holder drains, which releases it and marks the
    /// latch released.
    #[test]
    fn writable_export_drop_queues_then_drain_releases() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let latch = Latch::new();
            let ex = acquire_writable(py, ba.as_any(), 4096, &latch).unwrap();
            drop(ex); // this thread holds the GIL throughout, so no drainer can run
            assert_eq!(
                latch_state(&latch),
                (true, false),
                "drop must mark dropped, not released"
            );
            assert!(
                !resizable(ba.as_any()),
                "dropped export was released without a drain"
            );
            assert!(latch_ready(py, &latch), "drain must release it");
            assert_eq!(latch_state(&latch), (true, true));
            assert!(resizable(ba.as_any()));
        });
    }

    /// Dropped on a thread without the GIL (as a Tokio worker would): never blocks, and the
    /// releaser thread (or any drainer) releases it afterwards.
    #[test]
    fn writable_export_dropped_without_the_gil_is_released_by_the_releaser() {
        init();
        let (ba, latch) = Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let latch = Latch::new();
            let mut ex = acquire_writable(py, ba.as_any(), 4096, &latch).unwrap();
            let (tx, rx) = mpsc::channel();
            let t = std::thread::spawn(move || {
                ex.as_mut_slice().fill(9); // writing without the GIL
                drop(ex);
                tx.send(()).ok();
            });
            // The GIL is held here while the other thread drops the export.
            rx.recv_timeout(Duration::from_secs(5))
                .expect("dropping the writable export blocked on the GIL");
            t.join().unwrap();
            assert!(latch.is_dropped());
            (ba.unbind(), latch)
        });
        assert!(
            latch.wait_released_blocking(Duration::from_secs(10)),
            "nobody released the dropped export"
        );
        Python::attach(|py| {
            let ba = ba.bind(py);
            let v = ba.to_vec(); // before `resizable`, which appends a byte
            assert!(v.iter().all(|&b| b == 9), "writes lost");
            assert!(resizable(ba.as_any()));
        });
    }
}
