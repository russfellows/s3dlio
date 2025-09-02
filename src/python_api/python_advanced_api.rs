// src/python_api/python_advanced_api.rs
//
// Copyright 2025
// Signal65 / Futurum Group.
//
// Contains advanced features like the streaming multipart uploader.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::ffi;
use std::os::raw::c_char;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods, PyDict, PyDictMethods};
use pyo3::{PyObject, Bound};
use pyo3::exceptions::PyRuntimeError;

// Project crates
use crate::multipart::{
    MultipartUploadConfig,
    MultipartUploadSink,
};

// ---------------------------------------------------------------------------
// Multipart upload code
// ---------------------------------------------------------------------------
/// Streaming S3 multipart uploader.
///
/// - Fully streaming (no temp files by default)
/// - Concurrent UploadPart with bounded in-flight concurrency
/// - **Zero-copy** path via `reserve(size)` → fill memoryview → `commit(nbytes)`
///
/// Prefer the `reserve()` / `commit()` path for maximum throughput.
/// Use `write()` as a convenient general path for any bytes-like object.
#[pyclass(name = "MultipartUploadWriter", module = "s3dlio")]
pub struct PyMultipartUploadWriter {
    inner: Option<MultipartUploadSink>,
    pending_buf: Option<Vec<u8>>, // Rust-owned buffer between reserve() and commit()
}

#[pymethods]
impl PyMultipartUploadWriter {
    /// Create a writer for `bucket` + `key`.
    ///
    /// Args:
    ///     bucket: S3 bucket name (string)
    ///     key: object key (string)
    ///     part_size: target part size in bytes (>= 5 MiB). Default 16 MiB.
    ///     max_in_flight: max concurrent part uploads. Default 16.
    ///     content_type: optional Content-Type for the object.
    ///     abort_on_drop: auto-abort if the writer is dropped without close(). Default True.
    #[new]
    #[pyo3(signature = (bucket, key, part_size=None, max_in_flight=None, content_type=None, abort_on_drop=None))]
    fn new(
        bucket: &str,
        key: &str,
        part_size: Option<usize>,
        max_in_flight: Option<usize>,
        content_type: Option<String>,
        abort_on_drop: Option<bool>,
    ) -> PyResult<Self> {
        let mut cfg = MultipartUploadConfig::default();
        if let Some(ps) = part_size { cfg.part_size = ps; }
        if let Some(mif) = max_in_flight { cfg.max_in_flight = mif; }
        if let Some(ct) = content_type { cfg.content_type = Some(ct); }
        if let Some(aod) = abort_on_drop { cfg.abort_on_drop = aod; }

        let inner = MultipartUploadSink::new(bucket, key, cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Multipart init failed: {e}")))?;
        Ok(Self { inner: Some(inner), pending_buf: None })
    }

    /// Create a writer from an `s3://bucket/key` URI.
    ///
    /// Args:
    ///     uri: e.g. "s3://my-bucket/dir/obj.bin"
    ///     part_size, max_in_flight, content_type, abort_on_drop: see `__init__`.
    #[staticmethod]
    #[pyo3(signature = (uri, part_size=None, max_in_flight=None, content_type=None, abort_on_drop=None))]
    fn from_uri(
        uri: &str,
        part_size: Option<usize>,
        max_in_flight: Option<usize>,
        content_type: Option<String>,
        abort_on_drop: Option<bool>,
    ) -> PyResult<Self> {
        let mut cfg = MultipartUploadConfig::default();
        if let Some(ps) = part_size { cfg.part_size = ps; }
        if let Some(mif) = max_in_flight { cfg.max_in_flight = mif; }
        if let Some(ct) = content_type { cfg.content_type = Some(ct); }
        if let Some(aod) = abort_on_drop { cfg.abort_on_drop = aod; }

        let inner = MultipartUploadSink::from_uri(uri, cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Multipart init failed: {e}")))?;
        Ok(Self { inner: Some(inner), pending_buf: None })
    }

    /// Write a bytes-like object (bytes, bytearray, memoryview, NumPy buffer).
    ///
    /// For large buffers (>= 8 MiB), uses an owned fast path internally to avoid
    /// extra copies. For maximum performance, prefer `reserve()`/`commit()`.
    ///
    /// Returns:
    ///     int: number of bytes accepted.
    #[pyo3(text_signature = "(self, data, /)")]
    fn write<'py>(&mut self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<usize> {
        let inner = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;

        const OWNED_THRESHOLD: usize = 8 * 1024 * 1024; // 8 MiB

        // 1) Fast path for any buffer-protocol object (NumPy, memoryview, bytearray, etc.)
        if let Ok(buf) = data.extract::<pyo3::buffer::PyBuffer<u8>>() {
            let len = buf.len_bytes();
            let mut vec = Vec::<u8>::with_capacity(len);
            unsafe { vec.set_len(len); }
            buf.copy_to_slice(py, &mut vec[..])
                .map_err(|e| PyRuntimeError::new_err(format!("buffer copy failed: {e}")))?;

            let res = if len >= OWNED_THRESHOLD {
                py.allow_threads(|| inner.write_owned_blocking(vec))
            } else {
                py.allow_threads(|| inner.write_blocking(&vec))
            };
            return res
                .map(|_| len)
                .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")));
        }

        // 2) Fallback for plain bytes objects
        if let Ok(bytes) = data.downcast::<PyBytes>() {
            let slice = bytes.as_bytes(); // needs PyBytesMethods import
            let len = slice.len();
            if len >= OWNED_THRESHOLD {
                let vec = slice.to_vec();
                let res = py.allow_threads(|| inner.write_owned_blocking(vec));
                return res
                    .map(|_| len)
                    .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")));
            } else {
                let res = py.allow_threads(|| inner.write_blocking(slice));
                return res
                    .map(|_| len)
                    .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")));
            }
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "write() expects bytes-like (bytes, bytearray, memoryview, NumPy buffer)",
        ))
    }

    /// Reserve a Rust-owned buffer and return a writable Python `memoryview`.
    ///
    /// Fill the memoryview in Python, then call `commit(nbytes)`. Do not reuse
    /// the memoryview after `commit()`. Only one pending buffer is allowed.
    ///
    /// Args:
    ///     size: number of bytes to reserve
    ///
    /// Returns:
    ///     memoryview: writable view into the reserved buffer
    #[pyo3(text_signature = "(self, size, /)")]
    fn reserve(&mut self, py: Python<'_>, size: usize) -> PyResult<PyObject> {
        // Disallow overlapping reserves
        if self.pending_buf.is_some() {
            return Err(PyRuntimeError::new_err(
                "reserve() called while a previous buffer is pending; call commit() first",
            ));
        }

        // reserve(): treat size==0 as no-op (avoids odd memoryviews)
        if size == 0 {
            self.pending_buf = Some(Vec::new());
            // Return a 0-length memoryview (still legal)
            let ptr = std::ptr::null_mut::<std::os::raw::c_char>();
            let mv_ptr = unsafe { pyo3::ffi::PyMemoryView_FromMemory(ptr, 0, pyo3::ffi::PyBUF_WRITE) };
            if mv_ptr.is_null() { return Err(PyErr::fetch(py)); }
            return Ok(unsafe { PyObject::from_owned_ptr(py, mv_ptr) });
        }

        // Allocate Rust-owned buffer and keep it alive in self.pending_buf so the
        // memoryview's pointer stays valid until commit() (or close/abort).
        let mut buf = vec![0u8; size];

        // Create a writable memoryview that points directly at our Vec's memory.
        // Safety:
        // - We pass a valid pointer/length for the Vec.
        // - We keep the Vec alive (self.pending_buf = Some(buf)) so the memory doesn't move.
        // - We won't reallocate this Vec (we never push; commit() only shrinks len).
        let ptr = buf.as_mut_ptr() as *mut c_char;
        let len = buf.len() as ffi::Py_ssize_t;
        let flags = ffi::PyBUF_WRITE; // writable view

        let mv_ptr = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, flags) };
        if mv_ptr.is_null() {
            // Convert the current Python error (if any) into a PyErr.
            return Err(PyErr::fetch(py));
        }

        // Convert the owned PyObject* into a safe PyObject handle.
        let mv = unsafe { PyObject::from_owned_ptr(py, mv_ptr) };

        // Stash the Vec; this guarantees the pointer remains valid until commit().
        self.pending_buf = Some(buf);
        Ok(mv)
    }



    /// Commit the previously reserved buffer.
    ///
    /// Args:
    ///     nbytes: number of bytes actually written into the reserved buffer
    ///
    /// Errors if `nbytes` exceeds the reserved size or if no buffer is pending.
    #[pyo3(text_signature = "(self, nbytes, /)")]
    fn commit(&mut self, py: Python<'_>, nbytes: usize) -> PyResult<()> {
        let inner = self.inner.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        let mut buf = self.pending_buf.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no pending buffer; call reserve() first"))?;
        if nbytes > buf.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("commit(nbytes={nbytes}) exceeds reserved size {}", buf.len()),
            ));
        }
        unsafe { buf.set_len(nbytes); }
        py.allow_threads(|| inner.write_owned_blocking(buf))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("commit failed: {e}")))
    }

    /// Flush any buffered bytes as a (possibly short) part without finishing.
    ///
    /// Typically not required; `close()` will upload any tail automatically.
    #[pyo3(text_signature = "(self)")]
    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        py.allow_threads(|| inner.flush_blocking())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("flush failed: {e}")))
    }

    /// Complete the multipart upload and close the writer.
    ///
    /// Returns:
    ///     dict: {
    ///        'etag': str or None,
    ///        'total_bytes': int,
    ///        'parts': int,
    ///        'started_at': float (unix seconds),
    ///        'completed_at': float (unix seconds)
    ///     }
    #[pyo3(text_signature = "(self)")]
    fn close(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already closed"))?;
        let info = py.allow_threads(|| { let mut owned = inner; owned.finish_blocking() })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("finish failed: {e}")))?;
        let dict = PyDict::new(py);
        if let Some(etag) = info.e_tag { dict.set_item("etag", etag).ok(); }
        dict.set_item("total_bytes", info.total_bytes).ok();
        dict.set_item("parts", info.parts).ok();
        let started = info.started_at.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        let completed = info.completed_at.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        dict.set_item("started_at", started).ok();
        dict.set_item("completed_at", completed).ok();
        Ok(dict.into())
    }

    /// Abort the multipart upload and close the writer.
    ///
    /// Best effort: outstanding tasks are joined and the MPU is aborted.
    /// After abort, the writer is unusable.
    #[pyo3(text_signature = "(self)")]
    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut inner) = self.inner.take() {
            py.allow_threads(|| inner.abort_blocking())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("abort failed: {e}")))?;
        }
        Ok(())
    }

    /// Context manager enter: returns the writer.
    #[pyo3(text_signature = "(self)")]
    fn __enter__<'py>(slf: PyRefMut<'py, Self>) -> PyResult<PyRefMut<'py, Self>> { Ok(slf) }

    /// Context manager exit: closes on success; aborts on exception.
    #[pyo3(text_signature = "(self, exc_type, exc, tb)")]
    fn __exit__(&mut self, py: Python<'_>, _t: PyObject, _v: PyObject, _tb: PyObject) -> PyResult<()> {
        if self.inner.is_some() {
            if let Some(mut inner) = self.inner.take() {
                if let Err(_e) = py.allow_threads(|| inner.finish_blocking()) {
                    let _ = py.allow_threads(|| inner.abort_blocking());
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module registration function for advanced API
// ---------------------------------------------------------------------------
pub fn register_advanced_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Multipart upload writer
    m.add_class::<PyMultipartUploadWriter>()?;
    
    // TODO: Add checkpoint system when re-enabled
    // #[cfg(feature = "extension-module")]
    // {
    //     m.add_class::<PyCheckpointStore>()?;
    //     m.add_class::<PyCheckpointWriter>()?;
    //     m.add_class::<PyCheckpointReader>()?;
    // }
    
    Ok(())
}

