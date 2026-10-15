// src/python_api/put_payload.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Conversion of the `data` argument of `put_bytes()` / `put_bytes_async()`
//! into the `Bytes` handed to the object store (issue #238).
//!
//! # What is copied, and when
//!
//! | `data`                                   | `put_bytes` (blocking)        | `put_bytes_async` (deferred)             |
//! |------------------------------------------|-------------------------------|------------------------------------------|
//! | `s3dlio.BytesView`                       | no copy (reference-count bump)| no copy                                  |
//! | exact `bytes`                            | no copy (buffer held)         | no copy (immutable, buffer held)         |
//! | contiguous bytearray / memoryview / numpy / mmap ... | no copy (buffer held) | one copy, made with the GIL released     |
//! | non-contiguous (strided) view            | one gathering copy, GIL held  | same                                     |
//! | only `__bytes__()` / `bytes(obj)`        | one copy, GIL held            | same                                     |
//!
//! "Buffer held" means the `Py_buffer` export taken from the caller's object is wrapped
//! as the owner of the `Bytes` ([`Bytes::from_owner`]) and is released only when the last
//! clone of that `Bytes` is dropped. While it is held the exporter cannot be resized or
//! freed (a `bytearray` raises `BufferError` on `extend`), so the pointer stays valid for
//! the whole native access.
//!
//! # Safety argument and the one caller obligation
//!
//! The native code reads the caller's memory with the GIL RELEASED, so other Python
//! threads run (that is the point of #238: copying 512 MiB with the GIL held stalled
//! every other thread for ~0.3 s). Two things follow:
//!
//! * Memory safety: the export pins the allocation (see above), and the native side only
//!   reads. Dropping the owner re-attaches to Python to release the export (`PyBuffer`'s
//!   `Drop` uses `try_attach`), which is correct from a Tokio thread, from the calling
//!   thread inside `py.detach`, and during interpreter shutdown (the release is then
//!   skipped and the export leaks with the dying interpreter).
//! * Content: nothing stops another Python thread (or native code sharing the memory)
//!   from WRITING the bytes while they are being uploaded. The upload is then torn but
//!   memory-safe. Callers must not modify a mutable buffer (bytearray, writable
//!   memoryview, numpy array, mmap) during a blocking `put_bytes()` call. For
//!   `put_bytes_async()` that obligation is avoided by taking a snapshot at call time
//!   (the caller may reuse the buffer as soon as the call returns), at the price of one
//!   copy, made without the GIL.

use super::python_core_api::PyBytesView;
use bytes::Bytes;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods};

/// Whether the store reads the payload only while the Python call is running, or
/// keeps reading after `put_bytes_async()` has returned its awaitable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayloadUse {
    /// The call blocks until the store is done (`put_bytes`).
    Blocking,
    /// The store keeps reading after the call returns (`put_bytes_async`).
    Deferred,
}

/// A live `Py_buffer` export of a C-contiguous, non-empty exporter.
///
/// `PyBuffer<u8>` is `Send + Sync`; dropping it releases the export (see module docs).
struct PyBufferOwner {
    buffer: PyBuffer<u8>,
}

impl PyBufferOwner {
    /// Take the export of `data`. `Ok(None)` means the buffer is empty or not
    /// C-contiguous, so its raw memory is not the logical byte sequence.
    fn contiguous(buffer: PyBuffer<u8>) -> Option<Self> {
        if buffer.len_bytes() == 0 || !buffer.is_c_contiguous() {
            return None;
        }
        Some(Self { buffer })
    }
}

impl AsRef<[u8]> for PyBufferOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: `contiguous()` guarantees a non-empty, C-contiguous export of u8, and
        // `self.buffer` keeps the export (and so the allocation) alive for as long as
        // `self` exists. The slice is only read. See the module docs for the (documented)
        // caller obligation not to mutate the memory concurrently.
        unsafe {
            std::slice::from_raw_parts(self.buffer.buf_ptr() as *const u8, self.buffer.len_bytes())
        }
    }
}

fn buffer_error(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyBufferError, _>(format!("Buffer copy failed: {}", e))
}

/// Convert `data` to the `Bytes` that is written to the store (see the module docs for
/// exactly what is copied).
pub(crate) fn put_payload_from_py(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    usage: PayloadUse,
) -> PyResult<Bytes> {
    // 1. s3dlio's own BytesView: Arc/refcount clone. This MUST come first: BytesView also
    //    implements the buffer protocol, which used to shadow this branch (issue #238).
    if let Ok(bytes_view) = data.cast::<PyBytesView>() {
        return Ok(bytes_view.borrow().bytes.clone());
    }

    // 2. Anything that exports a buffer (bytes, bytearray, memoryview, numpy, mmap, ...).
    if let Ok(buffer) = PyBuffer::<u8>::get(data) {
        if buffer.len_bytes() == 0 {
            return Ok(Bytes::new());
        }
        let Some(owner) = PyBufferOwner::contiguous(buffer) else {
            // Strided or reversed view: the raw memory is not the logical byte sequence,
            // so gather it in logical order (a copy is unavoidable). Done with the GIL
            // held so a concurrent change cannot tear the gather.
            let buffer = PyBuffer::<u8>::get(data)?;
            let vec = buffer.to_vec(py).map_err(buffer_error)?;
            return Ok(Bytes::from(vec));
        };
        let immutable = data.is_exact_instance_of::<PyBytes>();
        return Ok(match usage {
            PayloadUse::Blocking => Bytes::from_owner(owner),
            PayloadUse::Deferred if immutable => Bytes::from_owner(owner),
            PayloadUse::Deferred => {
                // Snapshot now (the caller may reuse the buffer as soon as we return),
                // with the GIL released so other threads keep running during the copy.
                // `owner` keeps the export, so the memory cannot be freed or resized
                // while it is read, and it is released right after the copy.
                py.detach(|| Bytes::copy_from_slice(owner.as_ref()))
            }
        });
    }

    // 3. Last resort: `__bytes__()` or `bytes(obj)`; the result is a fresh `bytes`.
    let bytes_result = data.call_method0("__bytes__").or_else(|_| {
        let bytes_type = py.get_type::<PyBytes>();
        bytes_type.call1((data,))
    });
    match bytes_result {
        Ok(bytes_obj) => match bytes_obj.cast::<PyBytes>() {
            Ok(py_bytes) => Ok(Bytes::copy_from_slice(py_bytes.as_bytes())),
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

    fn init() {
        Python::initialize();
    }

    const BOTH: [PayloadUse; 2] = [PayloadUse::Blocking, PayloadUse::Deferred];

    fn exported_ptr(obj: &Bound<'_, PyAny>) -> *const u8 {
        PyBuffer::<u8>::get(obj).expect("buffer").buf_ptr() as *const u8
    }

    /// #238: a BytesView is reused as-is (same memory), in both modes.
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

    /// #238: a blocking put of any contiguous exporter reads the caller's memory in place.
    #[test]
    fn blocking_put_of_mutable_exporter_does_not_copy() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[3u8; 1 << 20]);
            let src = exported_ptr(ba.as_any());
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Blocking).unwrap();
            assert_eq!(got.as_ptr(), src, "bytearray was copied");
            assert!(got.iter().all(|&b| b == 3));

            // memoryview over the same bytearray: same memory too
            let mv = py
                .import("builtins")
                .unwrap()
                .call_method1("memoryview", (&ba,))
                .unwrap();
            let got = put_payload_from_py(py, &mv, PayloadUse::Blocking).unwrap();
            assert_eq!(got.as_ptr(), src, "memoryview was copied");
        });
    }

    /// #238: the borrowed buffer stays exported (cannot be resized or freed) while the
    /// Bytes is alive, and is released when the last clone is dropped.
    #[test]
    fn borrowed_buffer_is_held_until_bytes_dropped() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[1u8; 4096]);
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Blocking).unwrap();
            let clone = got.clone();
            let resize = || ba.call_method1("extend", (b"x".as_slice(),));
            let err = resize().expect_err("resize must fail while exported");
            assert!(err.is_instance_of::<pyo3::exceptions::PyBufferError>(py));
            drop(got);
            assert!(resize().is_err(), "still held by the clone");
            drop(clone);
            resize().expect("released after the last clone is dropped");
        });
    }

    /// #238: exact `bytes` is immutable, so even a deferred put reads it in place.
    #[test]
    fn exact_bytes_is_never_copied() {
        init();
        Python::attach(|py| {
            let b = PyBytes::new(py, &[9u8; 4096]);
            let src = b.as_bytes().as_ptr();
            for mode in BOTH {
                let got = put_payload_from_py(py, b.as_any(), mode).unwrap();
                assert_eq!(got.as_ptr(), src, "bytes was copied ({mode:?})");
            }
        });
    }

    /// A deferred put of mutable memory is a snapshot: later changes and resizes by the
    /// caller must not affect or be blocked by the pending write.
    #[test]
    fn deferred_put_of_mutable_exporter_is_an_independent_snapshot() {
        init();
        Python::attach(|py| {
            let ba = PyByteArray::new(py, &[5u8; 4096]);
            let src = exported_ptr(ba.as_any());
            let got = put_payload_from_py(py, ba.as_any(), PayloadUse::Deferred).unwrap();
            assert_ne!(got.as_ptr(), src);
            ba.call_method1("extend", (b"tail".as_slice(),))
                .expect("no export may be held");
            ba.call_method1("__setitem__", (0, 6)).unwrap();
            assert_eq!(got.len(), 4096);
            assert!(got.iter().all(|&b| b == 5));
        });
    }

    /// A strided view is gathered in logical order (a copy is unavoidable).
    #[test]
    fn non_contiguous_view_is_gathered_in_logical_order() {
        init();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c_str!("ba = bytearray(range(16)); mv = memoryview(ba)[::2]"),
                None,
                Some(&locals),
            )
            .unwrap();
            let mv = locals.get_item("mv").unwrap().unwrap();
            for mode in BOTH {
                let got = put_payload_from_py(py, &mv, mode).unwrap();
                assert_eq!(got.as_ref(), &[0u8, 2, 4, 6, 8, 10, 12, 14]);
            }
        });
    }

    #[test]
    fn empty_buffer_and_dunder_bytes_fallback() {
        init();
        Python::attach(|py| {
            let empty = PyByteArray::new(py, &[]);
            for mode in BOTH {
                assert!(put_payload_from_py(py, empty.as_any(), mode)
                    .unwrap()
                    .is_empty());
            }
            let locals = PyDict::new(py);
            py.run(
                c_str!("class C:\n    def __bytes__(self):\n        return b'abc'\nc = C()\nbad = object()"),
                None,
                Some(&locals),
            )
            .unwrap();
            let c = locals.get_item("c").unwrap().unwrap();
            assert_eq!(
                put_payload_from_py(py, &c, PayloadUse::Blocking)
                    .unwrap()
                    .as_ref(),
                b"abc"
            );
            let bad = locals.get_item("bad").unwrap().unwrap();
            assert!(put_payload_from_py(py, &bad, PayloadUse::Blocking).is_err());
        });
    }
}
