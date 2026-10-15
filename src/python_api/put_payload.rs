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
//! | any other C-contiguous buffer (bytearray, memoryview, numpy, `array.array`, ctypes, mmap, ...) | 0 copies | 1 snapshot copy (GIL released) | 0 copies |
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
//! the GIL. So [`ExportOwner`]'s `Drop` releases inline only when the dropping thread
//! already holds the GIL; otherwise it hands the export to a queue that
//!
//! * a dedicated thread (`s3dlio-export-release`) empties as soon as it can take the GIL,
//!   and
//! * the thread that waits for the call (`put_bytes` after it re-takes the GIL, and the
//!   result-delivery path of `put_bytes_async`, which already runs on a blocking-pool thread
//!   that attaches to Python) empties itself before the call returns or the awaitable
//!   resolves.
//!
//! Hence by the time `put_bytes` returns or the awaitable completes the export is released
//! (given the store dropped its clones, which it does before `put` returns) and a `resize`
//! of the source works.
//!
//! Exact `bytes` are held as a `Py<PyBytes>`: dropping it off the GIL defers the decref
//! (pyo3's reference pool), also without blocking.
//!
//! Interpreter shutdown: the releaser only attaches with `Python::try_attach`. That refuses
//! to attach while the interpreter is finalizing only when pyo3 is built for Python 3.13 or
//! newer (the `Py_IsFinalizing` guard is `Py_3_13`-only); on 3.12 a release racing with
//! shutdown is best effort. Exports still queued at exit are abandoned with the process.

use super::python_core_api::PyBytesView;
use bytes::Bytes;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

/// How long the store reads the payload relative to the Python call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayloadUse {
    /// The call blocks until the store is done (`put_bytes`): read the caller's memory in place.
    Blocking,
    /// `put_bytes_async` default: the store keeps reading after the call returns, so mutable
    /// memory is copied once, now (a snapshot), with the GIL released.
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

impl RawExport {
    /// Request a flat view of `data`; fails (BufferError/TypeError) for non-contiguous
    /// exporters and for objects without the buffer protocol.
    fn acquire(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // SAFETY: an all-zero Py_buffer is a valid "empty" value for PyObject_GetBuffer to fill.
        let mut view: Box<ffi::Py_buffer> = Box::new(unsafe { std::mem::zeroed() });
        // SAFETY: GIL held (we have a Bound), `view` is a valid out-pointer.
        if unsafe { ffi::PyObject_GetBuffer(data.as_ptr(), &mut *view, ffi::PyBUF_SIMPLE) } == -1 {
            return Err(PyErr::fetch(data.py()));
        }
        Ok(Self(view))
    }

    fn len(&self) -> usize {
        usize::try_from(self.0.len).unwrap_or(0)
    }

    /// Release the export.
    ///
    /// # Safety
    /// The calling thread must hold the GIL.
    unsafe fn release(mut self) {
        ffi::PyBuffer_Release(&mut *self.0);
    }
}

/// Queue of exports waiting for a thread that holds the GIL to release them.
struct ReleaseQueue {
    pending: Mutex<Vec<RawExport>>,
    wake: Condvar,
    count: AtomicUsize,
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

    /// Dedicated releaser: wait for work, then release it while holding the GIL. Items stay
    /// in the queue until the GIL is held, so a thread that drains while it holds the GIL
    /// sees every export that is not yet released.
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
            if Python::try_attach(drain_pending_exports).is_none() {
                return; // interpreter gone or finalizing: abandon the rest with the process
            }
        }
    }
}

/// The queue and its releaser thread (started on first use, i.e. the first export that
/// had to be handed off).
static QUEUE: OnceLock<&'static ReleaseQueue> = OnceLock::new();

fn release_queue() -> &'static ReleaseQueue {
    QUEUE.get_or_init(|| {
        let queue: &'static ReleaseQueue = Box::leak(Box::new(ReleaseQueue {
            pending: Mutex::new(Vec::new()),
            wake: Condvar::new(),
            count: AtomicUsize::new(0),
        }));
        if let Err(e) = std::thread::Builder::new()
            .name("s3dlio-export-release".into())
            .spawn(move || queue.run())
        {
            // Exports are then released by the next put_bytes/awaitable that drains.
            tracing::warn!("s3dlio: cannot start the buffer-export releaser thread: {e}");
        }
        queue
    })
}

/// Release every queued export. The caller holds the GIL (it has a `Python` token).
pub(crate) fn drain_pending_exports(_py: Python<'_>) {
    let Some(queue) = QUEUE.get() else {
        return;
    };
    if queue.count.load(Ordering::Acquire) == 0 {
        return;
    }
    let batch = {
        let mut pending = queue.lock();
        queue.count.store(0, Ordering::Release);
        std::mem::take(&mut *pending)
    };
    for export in batch {
        // SAFETY: `_py` proves the GIL is held.
        unsafe { export.release() };
    }
}

/// True when exports are waiting (cheap check for delivery paths).
pub(crate) fn exports_pending() -> bool {
    QUEUE
        .get()
        .is_some_and(|q| q.count.load(Ordering::Acquire) > 0)
}

/// Owns a buffer export for as long as the `Bytes` built from it lives.
struct ExportOwner {
    export: Option<RawExport>,
}

impl AsRef<[u8]> for ExportOwner {
    fn as_ref(&self) -> &[u8] {
        match &self.export {
            // SAFETY: the export is live (released only in `drop`), non-empty and contiguous;
            // `buf`/`len` describe `len` readable bytes. Documented caller obligation: no
            // concurrent modification of those bytes.
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
        // SAFETY: both calls are safe to make from any thread at any time.
        if unsafe { ffi::Py_IsInitialized() } == 0 {
            std::mem::forget(export); // interpreter gone: nothing to release into
            return;
        }
        if unsafe { ffi::PyGILState_Check() } == 1 {
            // SAFETY: this thread holds the GIL.
            unsafe { export.release() };
        } else {
            release_queue().push(export);
        }
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

    // 3. Any C-contiguous buffer, whatever its item type: hold the export.
    if let Ok(export) = RawExport::acquire(data) {
        if export.len() == 0 {
            // SAFETY: we hold the GIL (`py`).
            unsafe { export.release() };
            return Ok(Bytes::new());
        }
        let owner = ExportOwner {
            export: Some(export),
        };
        return Ok(match usage {
            PayloadUse::Blocking | PayloadUse::Borrowed => Bytes::from_owner(owner),
            PayloadUse::Snapshot => {
                // Copy now (the caller may reuse the buffer as soon as we return), with the
                // GIL released so other threads keep running. `owner` keeps the export, so
                // the memory cannot be freed or resized meanwhile; it is released right
                // after (this thread holds the GIL again, so inline).
                py.detach(|| Bytes::copy_from_slice(owner.as_ref()))
            }
        });
    }

    // 4. A buffer that is not C-contiguous (strided, reversed, transposed): Python gathers
    //    it in logical order (one copy, GIL held so a concurrent change cannot tear it) and
    //    the resulting `bytes` is wrapped, not copied again.
    if let Ok(gathered) = py
        .import("builtins")
        .and_then(|b| b.call_method1("memoryview", (data,)))
        .and_then(|mv| mv.call_method0("tobytes"))
    {
        if let Ok(b) = gathered.cast_exact::<PyBytes>() {
            return Ok(bytes_from_pybytes(b));
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
        let e = RawExport::acquire(obj).expect("buffer");
        let p = e.0.buf as *const u8;
        // SAFETY: the test thread holds the GIL.
        unsafe { e.release() };
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
}
