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
//! | `data`                                              | `put_bytes`              | `put_bytes_async`, default | `put_bytes_async(borrow=True)` |
//! |-----------------------------------------------------|--------------------------|----------------------------|--------------------------------|
//! | `s3dlio.BytesView`                                  | 0 copies                 | 0 copies                   | 0 copies                       |
//! | exact `bytes`                                       | 0 copies                 | 0 copies                   | 0 copies                       |
//! | any other C-contiguous buffer (bytearray, memoryview, numpy, `array.array`, ctypes, mmap, ...) | 0 copies | 1 snapshot copy (GIL released above 256 KiB) | 0 copies |
//! | non-contiguous (strided, reversed, transposed) view | 1 gather copy, GIL held  | same                       | same                           |
//! | not a buffer, only `__bytes__()` / `bytes(obj)`     | 1 copy, made by Python   | same                       | same                           |
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
//! While the export is held (for `put_bytes`: the whole call; for `put_bytes_async`: until
//! the awaitable completes with `borrow=True`, or only during the snapshot copy otherwise):
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
//! `spawn_blocking` writers that a cancelled future cannot stop).
//!
//! * `put_bytes` checks the latch after the put returned (usually it is already dropped: then
//!   it only drains, holding the GIL) and otherwise waits with the GIL released.
//! * The awaitable of `put_bytes_async(borrow=True)` is a `Future` subclass
//!   (`GuardedFuture`, see `python_core_api.rs`), NOT a Task running a coroutine: a coroutine
//!   body never runs if its Task is cancelled before the first step. Its `cancel()` cancels
//!   the put but its own terminal state is deferred until the latch is dropped and the export
//!   released, so `asyncio.timeout(0)` and `wait_for(.., 0)` wait too.
//! * The wait is bounded by `export_quiesce_timeout()` (30 s, `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`)
//!   and not interruptible by signals. If a backend never drops its clone the buffer stays pinned
//!   (never released while it can be read), the event is logged once and counted, and the call
//!   raises `ExportStillReferencedError`.
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
//! The atexit callback ([`begin_shutdown`]) sets a flag and drains the queue with the GIL.
//! From then on the autonomous releaser thread never attaches again (and is not even started
//! if the queue is created later), on every Python version. An export dropped after that is
//! still QUEUED (never forgotten) so a thread that holds a real GIL token - a `put_bytes`
//! finishing during exit - releases it; what nobody drains is abandoned with the process.
//! (pyo3's own `Py_IsFinalizing` check inside `try_attach` exists only when built for Python
//! 3.13 or newer, so it cannot be relied on for 3.12.) A releaser already blocked inside
//! `try_attach` when the interpreter starts finalizing is the one residual race on Python
//! <= 3.12; CPython then ends that thread.
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
//! The streaming writers (`write_chunk`, `write_owned_bytes`, `MultipartUploadWriter.write`)
//! use the same conversion but are STRICT: only buffer-protocol objects and objects with
//! `__bytes__()` are accepted (`put_bytes` keeps its `bytes(obj)` fallback).

use super::python_core_api::PyBytesView;
use bytes::Bytes;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use crate::constants::{
    export_quiesce_timeout, EXPORT_RELEASE_GRACE, SNAPSHOT_GIL_DETACH_THRESHOLD,
};

/// How long the store reads the payload relative to the Python call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayloadUse {
    /// The call blocks until the store is done (`put_bytes`): read the caller's memory in place.
    Blocking,
    /// `put_bytes_async` default: the store keeps reading after the call returns, so mutable
    /// memory is copied once, now (a snapshot).
    Snapshot,
    /// `put_bytes_async(borrow=True)`: the caller promised not to modify or free the buffer
    /// until the awaitable completes, so it is read in place like `Blocking`.
    Borrowed,
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
/// The Python-facing completion waits for `dropped` (bounded by `EXPORT_QUIESCE_TIMEOUT`), then
/// releases the queue with a real `py` token, and only then resolves, raises, or raises
/// `CancelledError`.
pub(crate) struct Latch {
    state: Mutex<LatchState>,
    cv: Condvar,
    notify: tokio::sync::Notify,
}

#[derive(Default)]
struct LatchState {
    dropped: bool,
    released: bool,
}

/// Times a wait for native readers gave up (a backend kept a clone past the timeout).
static QUIESCE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);
static QUIESCE_WARNED: AtomicBool = AtomicBool::new(false);
/// Failures of the barrier machinery itself (not a backend that holds on).
static BARRIER_INTERNAL_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Count an internal failure of the barrier (the buffer stays pinned).
pub(crate) fn note_internal_error() {
    BARRIER_INTERNAL_ERRORS.fetch_add(1, Ordering::Relaxed);
}

/// Record that a backend did not drop its clone in time: the export STAYS PINNED (memory that is
/// still being read is never released); log once, count always.
fn note_quiesce_timeout() {
    QUIESCE_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
    if !QUIESCE_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "s3dlio: a storage backend kept reading a borrowed Python buffer for more than {:?} \
             after the put finished; the buffer stays pinned (it is released when the backend \
             lets go). Further occurrences are only counted (see s3dlio._put_export_stats())",
            export_quiesce_timeout()
        );
    }
}

impl Latch {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(LatchState::default()),
            cv: Condvar::new(),
            notify: tokio::sync::Notify::new(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LatchState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn mark_dropped(&self) {
        self.lock().dropped = true;
        self.cv.notify_all();
        self.notify.notify_one();
    }

    pub(crate) fn mark_released(&self) {
        self.lock().released = true;
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

    /// Async variant of [`wait_dropped_blocking`](Self::wait_dropped_blocking).
    pub(crate) async fn wait_dropped(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.is_dropped() {
                return true;
            }
            if tokio::time::timeout_at(deadline, self.notify.notified())
                .await
                .is_err()
            {
                return self.is_dropped();
            }
        }
    }
}

/// The wait for native readers gave up: a backend still references the buffer. Maps to the
/// Python `ExportStillReferencedError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuiesceTimedOut;

/// `true` when every native reader is done (the owner was dropped); the queued export is then
/// released with this thread's GIL token (`py`). Never waits. Used on the event-loop thread by
/// the awaitable of `put_bytes_async(borrow=True)` and as the fast path of the blocking wait.
pub(crate) fn latch_ready(py: Python<'_>, latch: &Latch) -> bool {
    if latch.is_dropped() {
        drain_pending_exports(py);
        // Ready only when the export has REALLY been released: a dropped owner whose export is
        // not in the queue that was drained (or was abandoned) falls through to the quiesce path.
        latch.is_released()
    } else {
        false
    }
}

/// Wait until every native reader of the export is done, then release the queue with the real
/// `py` token. Used by the blocking `put_bytes` before it returns, however the put ended. When
/// the latch is already dropped (the usual case: file/direct stores and borrowed-slice writers
/// drop their last clone before returning) this only drains, holding the GIL: no detach, no
/// wait. Otherwise it waits with the GIL released, bounded by `export_quiesce_timeout()` (this
/// wait is NOT interruptible by signals/KeyboardInterrupt: the buffer must not be released
/// while it can still be read). On timeout the export stays pinned, the timeout is counted
/// and `Err(QuiesceTimedOut)` is returned.
pub(crate) fn quiesce_blocking(py: Python<'_>, latch: &Latch) -> Result<(), QuiesceTimedOut> {
    if latch_ready(py, latch) {
        return Ok(());
    }
    let timeout = export_quiesce_timeout();
    if py.detach(|| latch.wait_dropped_blocking(timeout)) {
        drain_pending_exports(py);
        Ok(())
    } else {
        note_quiesce_timeout();
        Err(QuiesceTimedOut)
    }
}

/// Async counterpart for the awaitable of `put_bytes_async(borrow=True)`: wait for the owner's
/// drop. Returns `true` on TIMEOUT (a backend still references the buffer; it stays pinned).
/// It never attaches to Python: the queued export is released by the result-delivery path of
/// the awaitable (which runs inside the #192 delivery gate, drains first, and is skipped once
/// the interpreter is exiting). Runs even when the awaitable was cancelled.
pub(crate) async fn quiesce_async(latch: Arc<Latch>) -> bool {
    if latch.wait_dropped(export_quiesce_timeout()).await {
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
}

pub(crate) fn export_stats() -> ExportStats {
    ExportStats {
        quiesce_timeouts: QUIESCE_TIMEOUTS.load(Ordering::Relaxed),
        queued: QUEUE.get().map_or(0, |q| q.count.load(Ordering::Acquire)),
        internal_errors: BARRIER_INTERNAL_ERRORS.load(Ordering::Relaxed),
    }
}

/// A raw `Py_buffer` export (`PyBUF_SIMPLE`: any C-contiguous exporter, any item type).
struct RawExport {
    view: Box<ffi::Py_buffer>,
    /// Marked "released" when `PyBuffer_Release` has really run (set by `ExportOwner::new`).
    latch: Option<Arc<Latch>>,
}

// SAFETY: the struct is only read from other threads (its `buf`/`len` are fixed for the life
// of the export) and released by value under the GIL.
unsafe impl Send for RawExport {}
unsafe impl Sync for RawExport {}

/// Whether a `struct`-module format string describes Python objects or pointers (`O`, `P`),
/// which must never be uploaded as bytes. `:name:` field names are skipped.
fn format_has_pointers(fmt: &str) -> bool {
    let mut in_name = false;
    for c in fmt.chars() {
        match c {
            ':' => in_name = !in_name,
            'O' | 'P' if !in_name => return true,
            _ => {}
        }
    }
    false
}

fn pointer_format_error(fmt: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "cannot upload a buffer of Python objects or pointers (buffer format {fmt:?}): \
         its bytes are memory addresses, not data"
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
    /// Set at interpreter exit: no export is released by the releaser, queued, or kept.
    shutdown: &'static AtomicBool,
    /// How long the releaser waits after a wake-up before attaching, so a `put_bytes` caller
    /// or the async delivery path (which drain the queue themselves) normally gets there first
    /// and the releaser finds nothing to do (no GIL traffic).
    grace: Duration,
    /// Times the releaser took the GIL (diagnostics / tests).
    attaches: AtomicUsize,
}

/// Set once by the atexit callback; see [`begin_shutdown`].
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

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
    /// that is not yet released. It never attaches once `shutdown` is set.
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
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }
            if self.count.load(Ordering::Acquire) == 0 {
                continue; // a caller drained it during the grace period
            }
            let attached = Python::try_attach(|py| {
                if !self.shutdown.load(Ordering::Acquire) {
                    self.attaches.fetch_add(1, Ordering::Relaxed);
                    self.drain(py);
                }
            });
            if attached.is_none() || self.shutdown.load(Ordering::Acquire) {
                return; // interpreter gone or exiting: abandon the rest with the process
            }
        }
    }

    /// A queue with its releaser thread.
    fn start(
        grace: Duration,
        shutdown: &'static AtomicBool,
        spawn_releaser: bool,
    ) -> &'static ReleaseQueue {
        let queue: &'static ReleaseQueue = Box::leak(Box::new(ReleaseQueue {
            pending: Mutex::new(Vec::new()),
            wake: Condvar::new(),
            count: AtomicUsize::new(0),
            shutdown,
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
static QUEUE: OnceLock<&'static ReleaseQueue> = OnceLock::new();

fn global_queue() -> &'static ReleaseQueue {
    QUEUE.get_or_init(|| {
        // After shutdown no releaser thread is started (it could only attach to a dying interpreter).
        ReleaseQueue::start(
            EXPORT_RELEASE_GRACE,
            &SHUTDOWN,
            !SHUTDOWN.load(Ordering::Acquire),
        )
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

/// Interpreter exit (called from the atexit callback, GIL held): release what is queued, then
/// stop releasing and queueing for good. After this, dropping an export forgets it (the
/// interpreter is about to finalize; nothing may attach to it any more) and the releaser
/// thread never attaches again. This replaces relying on pyo3's `Py_IsFinalizing` check, which
/// only exists when built for Python 3.13 or newer.
pub(crate) fn begin_shutdown(py: Python<'_>) {
    SHUTDOWN.store(true, Ordering::Release);
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
/// released above it so other threads keep running. An allocation failure is a `MemoryError`,
/// not an abort.
fn snapshot_copy(py: Python<'_>, src: &[u8]) -> PyResult<Bytes> {
    let result = if src.len() < SNAPSHOT_GIL_DETACH_THRESHOLD {
        try_snapshot(src, src.len())
    } else {
        py.detach(|| try_snapshot(src, src.len()))
    };
    result.map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyMemoryError, _>(format!(
            "cannot allocate {} bytes for the put_bytes_async snapshot",
            src.len()
        ))
    })
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

    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
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
    payload_parts(py, data, PayloadUse::Blocking, true).map(Held)
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

/// Like [`put_payload_from_py`] but also returns the completion barrier of a held export.
/// (`put_bytes` keeps accepting anything `bytes(obj)` accepts, as before.)
pub(crate) fn put_payload(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
) -> PyResult<Payload> {
    Ok(match payload_parts(py, data, usage, false)? {
        Parts::Plain(bytes) => Payload { bytes, latch: None },
        Parts::Export(owner, latch) => Payload {
            bytes: Bytes::from_owner(owner),
            latch: Some(latch),
        },
    })
}

fn payload_parts(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
    strict: bool,
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
        let latch = Latch::new();
        let owner = ExportOwner::new(export, latch.clone());
        return match usage {
            PayloadUse::Blocking | PayloadUse::Borrowed => Ok(Parts::Export(owner, latch)),
            PayloadUse::Snapshot => {
                // Copy now (the caller may reuse the buffer as soon as we return). `owner`
                // keeps the export, so the memory cannot be freed or resized meanwhile; it
                // is released explicitly right after (this thread holds the GIL again).
                let copy = snapshot_copy(py, owner.as_ref());
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

    // 5. Last resort: `__bytes__()`, and (unless `strict`) `bytes(obj)`; Python's fresh `bytes`
    //    is wrapped (one copy). `bytes(obj)` would turn an int into that many zero bytes and
    //    accept any iterable of ints, which the streaming writers must refuse.
    let bytes_result = data.call_method0("__bytes__").or_else(|e| {
        if strict {
            Err(e)
        } else {
            py.get_type::<PyBytes>().call1((data,))
        }
    });
    match bytes_result {
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

    fn leak_flag(v: bool) -> &'static AtomicBool {
        Box::leak(Box::new(AtomicBool::new(v)))
    }

    /// Round 3 / item 3, round 5: once shutdown is flagged the releaser thread never attaches,
    /// but an export is still QUEUED (never forgotten) so a thread that holds a real GIL token -
    /// a `put_bytes` finishing during exit - can release it.
    #[test]
    fn after_shutdown_the_releaser_never_attaches_but_a_token_holder_still_releases() {
        init();
        let flag = leak_flag(true);
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
        let flag = leak_flag(true);
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
        let queue = ReleaseQueue::start(Duration::from_millis(300), leak_flag(false), true);
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
        let queue = ReleaseQueue::start(Duration::from_millis(20), leak_flag(false), true);
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
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_flag(false), true);
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

    /// Item 7: a failed snapshot allocation is a MemoryError, not an abort.
    #[test]
    fn snapshot_allocation_failure_is_a_memory_error() {
        assert!(try_snapshot(&[1, 2, 3], usize::MAX).is_err());
        assert_eq!(try_snapshot(&[1, 2, 3], 3).unwrap().as_ref(), &[1, 2, 3]);
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
            // put_bytes keeps its liberal bytes(obj) fallback (unchanged behaviour)
            assert_eq!(
                put_payload_from_py(
                    py,
                    &objs.get_item("list").unwrap().unwrap(),
                    PayloadUse::Blocking
                )
                .unwrap()
                .as_ref(),
                &[1u8, 2, 3]
            );
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
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_flag(false), true);
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
}
