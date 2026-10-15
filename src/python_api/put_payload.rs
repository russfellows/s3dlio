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
//!   array, mmap or ctypes memory). Nothing prevents it, and the stored object would then
//!   mix old and new bytes. This is memory-safe but not byte-correct. It applies to the
//!   snapshot copy of `put_bytes_async` too (it happens with the GIL released).
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
//! * the thread that waits for the call (`put_bytes` after it re-takes the GIL, and the
//!   result-delivery path of `put_bytes_async`, which already runs on a blocking-pool thread
//!   that attaches to Python) drains the queue before the call returns or the awaitable
//!   resolves; and
//! * a dedicated thread (`s3dlio-export-release`), after a short grace period
//!   ([`EXPORT_RELEASE_GRACE`]) that normally lets the callers above get there first, so it
//!   finds nothing to do and never contends for the GIL on the hot path.
//!
//! Hence, once the store has dropped its clones of the `Bytes`, the export is released
//! before `put_bytes` returns or the awaitable completes and a `resize` of the source works.
//! That is guaranteed for `file://` and `direct://` (their `put` has dropped everything when
//! it returns); for `s3://`, `gs://`, `az://` it is best effort (an SDK may keep a clone of
//! the request body for an instant). If an awaitable is CANCELLED, a request that is already
//! running (a blocking file write, an in-flight HTTP body) is not stopped: it keeps the
//! export, and so the buffer pinned, until it finishes.
//!
//! Exact `bytes` are held as a `Py<PyBytes>`: dropping it off the GIL defers the decref
//! (pyo3's reference pool), also without blocking.
//!
//! # Interpreter shutdown
//!
//! The atexit callback ([`begin_shutdown`]) sets a flag and drains the queue with the GIL.
//! From then on dropping an export forgets it (never queued) and the releaser thread never
//! attaches again, on every Python version. (pyo3's own `Py_IsFinalizing` check inside
//! `try_attach` exists only when built for Python 3.13 or newer, so it cannot be relied on for
//! 3.12.) A releaser already blocked inside `try_attach` when the interpreter starts
//! finalizing is the one residual race on Python <= 3.12; CPython then ends that thread.
//!
//! # Other buffer details
//!
//! Buffers whose item format contains `O` (Python objects) or `P` (pointers) are refused with a
//! `TypeError`: their bytes are addresses, not data. A structured (record) dtype with padding is
//! uploaded as its raw memory, padding bytes included, exactly as it sits in memory.
//! The default `put_bytes_async` snapshot is copied with the GIL held below
//! [`SNAPSHOT_GIL_DETACH_THRESHOLD`] (releasing and re-taking it costs more than the copy) and
//! with the GIL released above it; an allocation failure is a `MemoryError`.

use super::python_core_api::PyBytesView;
use bytes::Bytes;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use crate::constants::{EXPORT_RELEASE_GRACE, SNAPSHOT_GIL_DETACH_THRESHOLD};

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

/// A raw `Py_buffer` export (`PyBUF_SIMPLE`: any C-contiguous exporter, any item type).
struct RawExport(Box<ffi::Py_buffer>);

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
        Ok(Self(view))
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
        if !export.0.format.is_null() {
            // SAFETY: a non-null `format` is a NUL-terminated string owned by the exporter.
            let fmt = unsafe { std::ffi::CStr::from_ptr(export.0.format) }
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
        usize::try_from(self.0.len).unwrap_or(0)
    }

    /// Release the export; the `py` token proves the GIL is held.
    fn release(mut self, _py: Python<'_>) {
        // SAFETY: `_py` proves this thread holds the GIL.
        unsafe { ffi::PyBuffer_Release(&mut *self.0) };
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
    fn start(grace: Duration, shutdown: &'static AtomicBool) -> &'static ReleaseQueue {
        let queue: &'static ReleaseQueue = Box::leak(Box::new(ReleaseQueue {
            pending: Mutex::new(Vec::new()),
            wake: Condvar::new(),
            count: AtomicUsize::new(0),
            shutdown,
            grace,
            attaches: AtomicUsize::new(0),
        }));
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
    QUEUE.get_or_init(|| ReleaseQueue::start(EXPORT_RELEASE_GRACE, &SHUTDOWN))
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
    fn new(export: RawExport) -> Self {
        Self {
            export: Some(export),
            queue: None,
        }
    }

    /// Release now; the `py` token proves the GIL is held.
    fn release(mut self, py: Python<'_>) {
        if let Some(export) = self.export.take() {
            export.release(py);
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
                std::slice::from_raw_parts(e.0.buf as *const u8, e.len())
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
        let queue = match self.queue {
            Some(q) => q,
            None => {
                // Do not even start the releaser thread once the interpreter is exiting.
                if SHUTDOWN.load(Ordering::Acquire) {
                    std::mem::forget(export);
                    return;
                }
                global_queue()
            }
        };
        // SAFETY: always safe to call.
        if queue.shutdown.load(Ordering::Acquire) || unsafe { ffi::Py_IsInitialized() } == 0 {
            std::mem::forget(export); // nothing may attach any more: abandon with the process
            return;
        }
        queue.push(export);
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

/// Convert `data` to the `Bytes` that is written to the store (see the module docs for
/// exactly what is copied).
pub(crate) fn put_payload_from_py(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
) -> PyResult<Bytes> {
    // 1. s3dlio's own BytesView: refcount clone. MUST come first: BytesView also implements
    //    the buffer protocol, which used to shadow this branch (issue #238).
    if let Ok(bytes_view) = data.cast::<PyBytesView>() {
        return Ok(bytes_view.try_borrow()?.bytes.clone());
    }

    // 2. Exact `bytes`: immutable, kept alive by a reference (no export needed).
    if let Ok(exact) = data.cast_exact::<PyBytes>() {
        return Ok(bytes_from_pybytes(exact));
    }

    // 3. Any C-contiguous buffer, whatever its item type: hold the export. (Arrays of
    //    Python objects or pointers are refused with a TypeError.)
    if let Some(export) = RawExport::acquire(data)? {
        if export.len() == 0 {
            export.release(py);
            return Ok(Bytes::new());
        }
        let owner = ExportOwner::new(export);
        return match usage {
            PayloadUse::Blocking | PayloadUse::Borrowed => Ok(Bytes::from_owner(owner)),
            PayloadUse::Snapshot => {
                // Copy now (the caller may reuse the buffer as soon as we return). `owner`
                // keeps the export, so the memory cannot be freed or resized meanwhile; it
                // is released explicitly right after (this thread holds the GIL again).
                let copy = snapshot_copy(py, owner.as_ref());
                owner.release(py);
                copy
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
                return Ok(bytes_from_pybytes(b));
            }
        }
    }

    // 5. Last resort: `__bytes__()` or `bytes(obj)`; Python's fresh `bytes` is wrapped (one copy).
    let bytes_result = data.call_method0("__bytes__").or_else(|_| {
        let bytes_type = py.get_type::<PyBytes>();
        bytes_type.call1((data,))
    });
    match bytes_result {
        Ok(bytes_obj) => match bytes_obj.cast_exact::<PyBytes>() {
            Ok(b) => Ok(bytes_from_pybytes(b)),
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
        let p = e.0.buf as *const u8;
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

    /// Round 3 / item 3: once shutdown is flagged, an export is forgotten (never queued,
    /// never released) and the releaser never attaches.
    #[test]
    fn after_shutdown_exports_are_forgotten_and_the_releaser_never_attaches() {
        init();
        let flag = leak_flag(true);
        let queue = ReleaseQueue::start(Duration::from_millis(5), flag);
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let owner = ExportOwner {
                export: Some(export),
                queue: Some(queue),
            };
            let bytes = Bytes::from_owner(owner);
            py.detach(move || drop(bytes));
            std::thread::sleep(Duration::from_millis(100));
            assert_eq!(
                queue.count.load(Ordering::Acquire),
                0,
                "queued after shutdown"
            );
            assert_eq!(
                queue.attaches.load(Ordering::Relaxed),
                0,
                "releaser attached"
            );
            assert!(
                !resizable(ba.as_any()),
                "forgotten export must stay exported"
            );
        });
    }

    /// Item 4: a caller that drains within the grace period leaves the releaser nothing to do:
    /// it never takes the GIL.
    #[test]
    fn caller_draining_within_the_grace_period_means_no_releaser_gil_traffic() {
        init();
        let queue = ReleaseQueue::start(Duration::from_millis(300), leak_flag(false));
        Python::attach(|py| {
            for _ in 0..20 {
                let ba = PyByteArray::new(py, &[1u8; 4096]);
                let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
                let owner = ExportOwner {
                    export: Some(export),
                    queue: Some(queue),
                };
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
        let queue = ReleaseQueue::start(Duration::from_millis(20), leak_flag(false));
        let ba = Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let export = RawExport::acquire(ba.as_any()).unwrap().unwrap();
            let owner = ExportOwner {
                export: Some(export),
                queue: Some(queue),
            };
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
        let queue = ReleaseQueue::start(Duration::from_secs(3600), leak_flag(false));
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
}
