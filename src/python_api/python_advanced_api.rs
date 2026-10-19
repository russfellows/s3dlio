// src/python_api/python_advanced_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};
use pyo3::Bound;
use std::os::raw::c_char;

// Project crates
use crate::multipart::{MultipartUploadConfig, MultipartUploadSink};
use crate::python_api::put_payload::{
    drain_pending_exports, put_payload, quiesce_blocking, snapshot_for_call, Latch, PayloadUse,
};
use crate::python_api::python_core_api::{
    py_err, require_max_in_flight, ExportStillReferencedError, PyBytesView,
};
use std::collections::BTreeMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Multipart upload code
// ---------------------------------------------------------------------------

/// The buffer exports a `MultipartUploadWriter` still holds on behalf of `write()` (zero-copy
/// default): `[start, end)` memory ranges, disjoint, each with its completion latch.
///
/// Both operations are O(log n) (ordered map keyed by `start`); released entries are removed
/// lazily - whenever one is found in the way, and in a batch each time the map has doubled since
/// the last sweep - so the per-write bookkeeping is O(1) amortised, not a scan of the list.
struct HeldExports {
    ranges: BTreeMap<usize, HeldRange>,
    /// Sweep released entries when `ranges` reaches this size.
    sweep_at: usize,
}

struct HeldRange {
    end: usize,
    latch: Arc<Latch>,
}

const HELD_SWEEP_MIN: usize = 64;

impl HeldExports {
    fn new() -> Self {
        Self {
            ranges: BTreeMap::new(),
            sweep_at: HELD_SWEEP_MIN,
        }
    }

    /// The start of a held, not yet released range that overlaps `[start, end)`, removing any
    /// released ranges met on the way.
    fn find_overlap(&mut self, py: Python<'_>, start: usize, end: usize) -> Option<(usize, usize)> {
        loop {
            let candidate = self
                .ranges
                .range(..=start)
                .next_back()
                .filter(|(_, r)| r.end > start)
                .or_else(|| self.ranges.range(start..end).next())
                .map(|(s, r)| (*s, r.end));
            let (s, e) = candidate?;
            let released = match self.ranges.get(&s) {
                Some(r) => {
                    // A dropped owner sits in the release queue until a thread with the GIL
                    // drains it; do that before judging the range still held.
                    if r.latch.is_dropped() {
                        drain_pending_exports(py);
                    }
                    r.latch.is_released()
                }
                None => true,
            };
            if !released {
                return Some((s, e));
            }
            self.ranges.remove(&s);
        }
    }

    /// `Ok` when `[start, end)` overlaps no export the writer still holds; otherwise the
    /// `BufferError` message.
    fn check_free(&mut self, py: Python<'_>, start: usize, end: usize) -> Result<(), String> {
        match self.find_overlap(py, start, end) {
            None => Ok(()),
            Some(_) => Err(
                "buffer reused while still borrowed; pass copy=True or use a fresh buffer \
                 (this write's memory overlaps an earlier write() that has not been uploaded \
                 yet)"
                    .to_string(),
            ),
        }
    }

    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn insert(&mut self, py: Python<'_>, start: usize, end: usize, latch: Arc<Latch>) {
        if self.ranges.len() >= self.sweep_at {
            drain_pending_exports(py);
            self.ranges.retain(|_, r| !r.latch.is_released());
            self.sweep_at = (self.ranges.len() * 2).max(HELD_SWEEP_MIN);
        }
        self.ranges.insert(start, HeldRange { end, latch });
    }

    fn drain(&mut self) -> Vec<Arc<Latch>> {
        std::mem::take(&mut self.ranges)
            .into_values()
            .map(|r| r.latch)
            .collect()
    }
}

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
    // Self-referencing strong handle, held only while `pending_buf` is `Some`
    // (i.e. between reserve() and commit()/abort()/close()/__exit__()).
    //
    // reserve() hands Python a memoryview built via `PyMemoryView_FromMemory`,
    // which — unlike `PyMemoryView_FromObject` routed through the buffer
    // protocol (see `PyBytesView::__getbuffer__`'s doc comment for the same
    // class of bug, already fixed there once) — never sets `view.obj` and so
    // carries **no reference** back to this writer. Without this field,
    // dropping the last Python reference to the writer while its memoryview
    // is still held frees `pending_buf`'s heap allocation out from under a
    // live memoryview: a use-after-free. Holding an extra `Py<Self>` here
    // keeps the writer (and thus `pending_buf`) alive for the whole
    // reserve()..commit() window regardless of what the caller does with the
    // writer's own Python reference. Cleared via `clear_pending_reservation`
    // wherever `pending_buf` is cleared, so it never outlives the reservation
    // it protects.
    //
    // Trade-off: if a caller calls reserve() and then abandons the writer
    // without ever calling commit()/abort()/close()/using it as a context
    // manager, this self-reference is never cleared and the writer leaks for
    // the life of the process (PyMultipartUploadWriter does not implement
    // PyO3's GC protocol, so Python's cyclic collector cannot break this
    // cycle). A leak is a strict improvement over the use-after-free it
    // replaces, and the existing docstring already documents reserve() as
    // requiring a paired commit() call.
    self_pin_while_pending: Option<Py<PyMultipartUploadWriter>>,
    /// Completion barriers of the borrowed buffer exports behind `write()` (zero-copy default):
    /// the parts still reference the caller's memory until they are uploaded, so `close()` /
    /// `abort()` / `__exit__` wait on every latch (bounded) before they return.
    held: HeldExports,
}

impl PyMultipartUploadWriter {
    /// Clear any pending reserve() buffer and the self-pin that keeps this
    /// writer alive while a reservation is outstanding. Idempotent. Must be
    /// called wherever `pending_buf` is consumed or the writer is closed, so
    /// `self_pin_while_pending` never leaks past the reservation it guards.
    fn clear_pending_reservation(&mut self) {
        self.pending_buf = None;
        self.self_pin_while_pending = None;
    }

    /// Wait until every buffer export borrowed by `write()` has really been released, i.e. no
    /// upload task or queued part references the caller's memory any more. Bounded by
    /// `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` per export; on a timeout the buffers stay pinned and
    /// `ExportStillReferencedError` is returned (the caller raises it in its own stack).
    /// `outcome` says what the upload itself did, for the message. The sink MUST already be
    /// dropped or finished: it is what drops the parts that still queue slices of the buffer.
    fn release_held_exports(&mut self, py: Python<'_>, outcome: &str) -> PyResult<()> {
        let mut timed_out = false;
        for latch in self.held.drain() {
            if quiesce_blocking(py, &latch).is_err() {
                timed_out = true;
            }
        }
        if timed_out {
            return Err(ExportStillReferencedError::new_err(format!(
                "a storage backend still references a buffer passed to write() {:?} after the \
                 writer finished ({outcome}); the buffer stays pinned - do not modify, resize \
                 or free it (S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS)",
                crate::constants::export_quiesce_timeout()
            )));
        }
        Ok(())
    }
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
        py: Python<'_>,
        bucket: &str,
        key: &str,
        part_size: Option<usize>,
        max_in_flight: Option<usize>,
        content_type: Option<String>,
        abort_on_drop: Option<bool>,
    ) -> PyResult<Self> {
        // Issue #211: a limit of 0 is a ValueError, raised before anything else.
        if let Some(mif) = max_in_flight {
            require_max_in_flight("max_in_flight", mif)?;
        }
        let mut cfg = MultipartUploadConfig::default();
        if let Some(ps) = part_size {
            cfg.part_size = ps;
        }
        if let Some(mif) = max_in_flight {
            cfg.max_in_flight = Some(mif);
        }
        if let Some(ct) = content_type {
            cfg.content_type = Some(ct);
        }
        if let Some(aod) = abort_on_drop {
            cfg.abort_on_drop = aod;
        }

        // Release the GIL for the blocking CreateMultipartUpload S3 call.
        let bucket_s = bucket.to_string();
        let key_s = key.to_string();
        let inner = py
            .detach(move || MultipartUploadSink::new(&bucket_s, &key_s, cfg))
            .map_err(|e| py_err(e.context("Multipart init failed")))?;
        Ok(Self {
            inner: Some(inner),
            pending_buf: None,
            self_pin_while_pending: None,
            held: HeldExports::new(),
        })
    }

    /// Create a writer from an `s3://bucket/key` URI.
    ///
    /// Args:
    ///     uri: e.g. "s3://my-bucket/dir/obj.bin"
    ///     part_size, max_in_flight, content_type, abort_on_drop: see `__init__`.
    #[staticmethod]
    #[pyo3(signature = (uri, part_size=None, max_in_flight=None, content_type=None, abort_on_drop=None))]
    fn from_uri(
        py: Python<'_>,
        uri: &str,
        part_size: Option<usize>,
        max_in_flight: Option<usize>,
        content_type: Option<String>,
        abort_on_drop: Option<bool>,
    ) -> PyResult<Self> {
        // Issue #211: a limit of 0 is a ValueError, raised before anything else.
        if let Some(mif) = max_in_flight {
            require_max_in_flight("max_in_flight", mif)?;
        }
        let mut cfg = MultipartUploadConfig::default();
        if let Some(ps) = part_size {
            cfg.part_size = ps;
        }
        if let Some(mif) = max_in_flight {
            cfg.max_in_flight = Some(mif);
        }
        if let Some(ct) = content_type {
            cfg.content_type = Some(ct);
        }
        if let Some(aod) = abort_on_drop {
            cfg.abort_on_drop = aod;
        }

        // Release the GIL for the blocking CreateMultipartUpload S3 call.
        let uri_s = uri.to_string();
        let inner = py
            .detach(move || MultipartUploadSink::from_uri(&uri_s, cfg))
            .map_err(|e| py_err(e.context("Multipart init failed")))?;
        Ok(Self {
            inner: Some(inner),
            pending_buf: None,
            self_pin_while_pending: None,
            held: HeldExports::new(),
        })
    }

    /// Write a bytes-like object (bytes, bytearray, memoryview, NumPy buffer).
    ///
    /// ZERO-COPY by default for a large write (threshold below): the buffer is NOT copied.
    /// Its Python buffer export is held (so it cannot be resized or freed:
    /// ``bytearray.extend()`` raises ``BufferError``) and the parts are uploaded straight out
    /// of your memory, asynchronously, after this call returns. The export is kept until every
    /// part that uses it has been uploaded; ``close()`` / ``abort()`` (and leaving a ``with``
    /// block) wait for that before returning.
    ///
    /// **You must not modify the buffer until ``close()`` (or ``abort()``) has returned.** The
    /// bytes are read by background upload tasks at any time until then, so changing them is a
    /// data race and the object contains undefined bytes; allocate a fresh buffer for each
    /// ``write()`` (or pass ``copy=True``). Reusing a buffer that the writer still holds (the
    /// ``readinto(buf); write(buf)`` loop) is refused: ``write()`` raises ``BufferError``
    /// ("buffer reused while still borrowed; pass copy=True or use a fresh buffer") when the
    /// new buffer overlaps memory of an earlier write that has not been uploaded yet; the
    /// writer stays usable and can be aborted. If a storage backend still references the buffer
    /// long after the writer finished (``S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS``, 30 s),
    /// ``close()``/``abort()`` raise ``s3dlio.ExportStillReferencedError`` and the buffer
    /// stays pinned: do not modify, resize or free it.
    ///
    /// Which writes are borrowed. SIGNED uploads (the default): a write of at least ``part_size``
    /// bytes is borrowed; a smaller write is copied ONCE into the part buffer and nothing is
    /// borrowed (no ``BufferError`` lock, the buffer may be reused at once). A part is made of
    /// several such writes and must be one contiguous request body, so a sub-part write is
    /// copied into it anyway; borrowing it would not avoid that copy. Measured (16 MiB parts,
    /// 128 MiB per case, best of 3, write loop + ``close()``, borrow/copy time): 1.07-1.25 at
    /// 1 KiB and below, 0.87-1.04 from 2 KiB up (typically ~0.93, about the run-to-run noise).
    /// UNSIGNED uploads (``S3DLIO_UNSIGNED_PAYLOAD``): a part that spans writes is sent as a
    /// multi-chunk body, so borrowing a sub-part write IS zero-copy end to end; writes of 2 KiB
    /// or more are borrowed, smaller ones copied. Measured the same way: 0.72-0.93 from 2 KiB
    /// to 32 KiB, parity (0.94-1.10) from 64 KiB up, 1.05-1.19 at 1 KiB and below.
    /// Two notes on the unsigned multi-chunk bodies: they are streaming bodies, so reqwest's own
    /// redirect policy cannot replay one on a 307/308 (``S3DLIO_FOLLOW_REDIRECTS`` still works);
    /// and the 2 KiB threshold was measured over plain HTTP/1.1, TLS/h2 may favour larger writes.
    ///
    /// BEHAVIOUR CHANGE in 0.10.0: the buffer is no longer copied by the writer. It is borrowed
    /// until ``close()`` / ``abort()`` returns; ``copy=True`` is the explicit independent
    /// snapshot.
    ///
    /// Args:
    ///     data: bytes-like object (or an object with ``__bytes__()``).
    ///     copy: keyword-only, default False. With ``copy=True`` a mutable buffer is first
    ///         copied once with the GIL held and the export released, so you may modify or reuse
    ///         it as soon as the call has started. The snapshot is atomic only against writers
    ///         that HOLD the GIL (pure-Python threads); a writer that releases the GIL (large
    ///         NumPy operations, ``readinto``/``recv_into``, CUDA or torch copies) can still tear
    ///         it and must be serialised by the caller. Exact ``bytes`` and ``BytesView`` are
    ///         immutable, shared and never copied by ``write()``.
    ///
    /// Data is sliced into parts without copying. The exceptions, each one bounded copy: a
    /// write below the threshold (above); and, for a SIGNED upload, a part that straddles
    /// two ``write()`` calls (the unfinished tail of the previous write plus the head of this one),
    /// which must be one contiguous request body and is gathered with ONE copy of at most
    /// ``part_size`` bytes. With ``S3DLIO_UNSIGNED_PAYLOAD`` on, a straddling part is sent as a
    /// multi-chunk body and not copied. Writes that are multiples of ``part_size`` never copy.
    /// For maximum performance, prefer `reserve()`/`commit()`.
    ///
    /// Returns:
    ///     int: number of bytes accepted.
    #[pyo3(signature = (data, /, *, copy = false), text_signature = "(self, data, /, *, copy=False)")]
    fn write<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        copy: bool,
    ) -> PyResult<usize> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;
        let small_below = inner.coalesce_below();

        // 0) Our own BytesView: Arc clone, NO copy; the sink copies it once only when it is
        //    below the small-write threshold.
        if let Ok(bv) = data.extract::<PyRef<PyBytesView>>() {
            let bytes = bv.bytes.clone(); // just atomic refcount++, no data movement
            let len = bytes.len();
            return py
                .detach(|| inner.write_bytes_blocking(bytes))
                .map(|_| len)
                .map_err(|e| py_err(e.context("write failed")));
        }

        // 1) Any other bytes-like object (bytes, bytearray, memoryview, NumPy of any dtype, ...):
        //    the same strict conversion as put_bytes().
        let not_bytes_like = |e: PyErr| {
            if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "write() expects bytes-like (bytes, bytearray, memoryview, NumPy buffer) or an object with __bytes__(): {e}"
                ))
            } else {
                e
            }
        };
        if copy {
            // `copy=True`: one snapshot with the GIL held (export released), then the owned
            // bytes are handed to the sink without another copy of the caller's memory.
            let snapshot = snapshot_for_call(py, data).map_err(not_bytes_like)?;
            let len = snapshot.len();
            return py
                .detach(|| inner.write_bytes_blocking(snapshot))
                .map(|_| len)
                .map_err(|e| py_err(e.context("write failed")));
        }

        // Default: BORROW the buffer. `Bytes::from_owner` over the held export lets the sink
        // slice it into parts and upload them after this call returns; the export is released
        // when the last part clone is dropped, and `close()`/`abort()` wait for that through
        // the latch (the same barrier `put_bytes(borrow=True)` uses). Exact `bytes` and other
        // immutable payloads have no latch: nothing to wait for.
        let payload = put_payload(py, data, PayloadUse::Borrowed).map_err(not_bytes_like)?;
        let len = payload.bytes.len();
        let Some(latch) = payload.latch else {
            // Immutable payload (exact bytes, empty): nothing to hold.
            return py
                .detach(|| inner.write_bytes_blocking(payload.bytes))
                .map(|_| len)
                .map_err(|e| py_err(e.context("write failed")));
        };
        // Refuse a buffer that overlaps memory the writer still holds (the `readinto(buf);
        // write(buf)` loop, including its short final chunk): the earlier parts would be
        // rewritten under the uploader. Checked BEFORE anything is handed to the sink, for
        // EVERY write - also one below the coalescing threshold - whenever something is held
        // (an O(log n) lookup; when nothing is held nothing is checked).
        let start = payload.bytes.as_ptr() as usize;
        let end = start.saturating_add(len);
        if !self.held.is_empty() {
            if let Err(msg) = self.held.check_free(py, start, end) {
                drop(payload.bytes);
                drain_pending_exports(py);
                return Err(pyo3::exceptions::PyBufferError::new_err(msg));
            }
        }
        if len < small_below {
            // Small write: ONE copy into the part buffer, nothing borrowed. The export is
            // dropped (queued) and released right here with the GIL token.
            let res = py.detach(|| inner.write_blocking(&payload.bytes));
            drop(payload.bytes);
            drain_pending_exports(py);
            return res
                .map(|_| len)
                .map_err(|e| py_err(e.context("write failed")));
        }
        // Large borrowed write (the overlap check above has passed).
        let res = py.detach(|| inner.write_bytes_blocking(payload.bytes));
        // Tracked even when the write failed: part slices may already be queued.
        self.held.insert(py, start, end, latch);
        res.map(|_| len)
            .map_err(|e| py_err(e.context("write failed")))
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
    fn reserve(mut slf: PyRefMut<'_, Self>, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
        // Disallow overlapping reserves
        if slf.pending_buf.is_some() {
            return Err(PyRuntimeError::new_err(
                "reserve() called while a previous buffer is pending; call commit() first",
            ));
        }

        // Pin ourselves alive for the whole reserve()..commit() window — see
        // `self_pin_while_pending`'s field doc for why this is required for
        // memory safety, not just cleanliness. Built from a borrowed pointer
        // (Py_INCREF under the hood), so this is zero-copy: no data is
        // duplicated, only a Python reference count changes.
        let self_pin: Py<Self> = unsafe { Py::from_borrowed_ptr(py, slf.as_ptr()) };

        // reserve(): treat size==0 as no-op (avoids odd memoryviews)
        if size == 0 {
            slf.pending_buf = Some(Vec::new());
            slf.self_pin_while_pending = Some(self_pin);
            // Return a 0-length memoryview (still legal)
            let ptr = std::ptr::null_mut::<std::os::raw::c_char>();
            let mv_ptr =
                unsafe { pyo3::ffi::PyMemoryView_FromMemory(ptr, 0, pyo3::ffi::PyBUF_WRITE) };
            if mv_ptr.is_null() {
                slf.clear_pending_reservation();
                return Err(PyErr::fetch(py));
            }
            return Ok(unsafe { Py::<PyAny>::from_owned_ptr(py, mv_ptr) });
        }

        // Allocate Rust-owned buffer and keep it alive in self.pending_buf so the
        // memoryview's pointer stays valid until commit() (or close/abort).
        let mut buf = vec![0u8; size];

        // Create a writable memoryview that points directly at our Vec's memory.
        // Safety:
        // - We pass a valid pointer/length for the Vec.
        // - We keep the Vec alive (self.pending_buf = Some(buf)) so the memory doesn't move.
        // - We won't reallocate this Vec (we never push; commit() only shrinks len).
        // - We keep `self` (the writer) alive via `self_pin_while_pending` so the
        //   above holds even if the caller drops every other Python reference
        //   to the writer before calling commit() — see the field doc comment.
        let ptr = buf.as_mut_ptr() as *mut c_char;
        let len = buf.len() as ffi::Py_ssize_t;
        let flags = ffi::PyBUF_WRITE; // writable view

        let mv_ptr = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, flags) };
        if mv_ptr.is_null() {
            // Convert the current Python error (if any) into a PyErr.
            return Err(PyErr::fetch(py));
        }

        // Convert the owned PyObject* into a safe PyObject handle.
        let mv = unsafe { Py::<PyAny>::from_owned_ptr(py, mv_ptr) };

        // Stash the Vec and the self-pin; this guarantees both the pointer
        // and the writer object itself remain valid until commit().
        slf.pending_buf = Some(buf);
        slf.self_pin_while_pending = Some(self_pin);
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
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        let mut buf = self.pending_buf.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no pending buffer; call reserve() first")
        })?;
        // The reservation window is over now that pending_buf is taken —
        // release the self-pin (see `self_pin_while_pending`'s field doc).
        self.self_pin_while_pending = None;
        if nbytes > buf.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "commit(nbytes={nbytes}) exceeds reserved size {}",
                buf.len()
            )));
        }
        unsafe {
            buf.set_len(nbytes);
        }
        // A commit much smaller than the reservation must not pin the whole reserved capacity
        // for as long as its part is in flight (`Bytes::from(Vec)` keeps the allocation):
        // shrink it first. This is a realloc of the allocation (the kernel remaps big blocks), not
        // a copy of the payload. Commits below the small-write threshold are copied once into
        // the part buffer by the sink, and this buffer is freed.
        if buf.capacity() > buf.len() {
            buf.shrink_to_fit();
        }
        py.detach(|| inner.write_owned_blocking(buf))
            .map_err(|e| py_err(e.context("commit failed")))
    }

    /// Flush any buffered bytes as a (possibly short) part without finishing.
    ///
    /// Typically not required; `close()` will upload any tail automatically.
    #[pyo3(text_signature = "(self)")]
    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        py.detach(|| inner.flush_blocking())
            .map_err(|e| py_err(e.context("flush failed")))
    }

    /// Complete the multipart upload and close the writer.
    ///
    /// Waits for every part, including parts that read buffers you passed to ``write()``, and
    /// releases those buffers before it returns. If a storage backend still references one of
    /// them after ``S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`` (30 s), ``s3dlio.ExportStillReferencedError``
    /// is raised INSTEAD of the upload result: it takes precedence over an upload failure too,
    /// which is then only quoted in its message ("the upload itself failed: ..."). The buffers
    /// stay pinned; do not modify, resize or free them.
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
    fn close(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Release any outstanding reserve() buffer and its self-pin (see
        // `self_pin_while_pending`'s field doc) — otherwise a reserve()
        // that was never commit()'d would leak the writer permanently.
        self.clear_pending_reservation();
        let inner = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already closed"))?;
        let finished = py.detach(|| {
            // `owned` (the sink, with any parts it still queues) is dropped before the closure
            // returns, so no clone of a borrowed buffer survives it except in-flight uploads.
            let mut owned = inner;
            owned.finish_blocking()
        });
        // Zero-copy writes: wait until no part references the caller's buffers any more, and
        // release the exports, however the upload ended.
        let outcome = match &finished {
            Ok(_) => "the upload itself succeeded".to_string(),
            Err(e) => format!("the upload itself failed: {e}"),
        };
        self.release_held_exports(py, &outcome)?;
        let info = finished.map_err(|e| py_err(e.context("finish failed")))?;
        let dict = PyDict::new(py);
        // Always set the key, even when there is no etag (info.e_tag is
        // None) -- PyO3 converts Option<String> to None/str automatically.
        // The docstring above promises 'etag': str or None; omitting the
        // key entirely in the None case (as the old `if let` guard did)
        // made result['etag'] raise KeyError instead of returning None
        // (design doc §4.3).
        dict.set_item("etag", info.e_tag).ok();
        dict.set_item("total_bytes", info.total_bytes).ok();
        dict.set_item("parts", info.parts).ok();
        let started = info
            .started_at
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let completed = info
            .completed_at
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        dict.set_item("started_at", started).ok();
        dict.set_item("completed_at", completed).ok();
        Ok(dict.into())
    }

    /// Abort the multipart upload and close the writer.
    ///
    /// Best effort: outstanding upload tasks are cancelled and the MPU is aborted. Buffers
    /// passed to `write()` are released before this returns (see `write()`).
    /// After abort, the writer is unusable.
    #[pyo3(text_signature = "(self)")]
    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        // Release any outstanding reserve() buffer and its self-pin (see
        // `self_pin_while_pending`'s field doc) — otherwise a reserve()
        // that was never commit()'d would leak the writer permanently.
        self.clear_pending_reservation();
        let mut aborted: PyResult<()> = Ok(());
        if let Some(mut inner) = self.inner.take() {
            aborted = py
                .detach(|| inner.abort_blocking())
                .map_err(|e| py_err(e.context("abort failed")));
            // Drop the sink (and the parts it still queues) before waiting for the buffers.
            drop(inner);
        }
        // Zero-copy writes: the in-flight uploads were cancelled; wait until no part references
        // the caller's buffers any more (bounded) and release the exports.
        self.release_held_exports(py, "the upload was aborted")?;
        aborted
    }

    /// Context manager enter: returns the writer.
    #[pyo3(text_signature = "(self)")]
    fn __enter__<'py>(slf: PyRefMut<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        Ok(slf)
    }

    /// Context manager exit: closes on success; aborts and raises on error.
    ///
    /// If an exception propagated from inside the `with` block (_t is not None),
    /// the upload is aborted and the original exception is re-raised unchanged.
    /// If no exception occurred, `finish_blocking()` is called; any error it
    /// returns is surfaced as a new `RuntimeError`.
    ///
    /// Like ``close()``, it waits for and releases the buffers passed to ``write()``; if that
    /// wait times out (``S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS``) ``s3dlio.ExportStillReferencedError``
    /// is raised and takes precedence over both the in-flight exception and an upload failure
    /// (the upload error is quoted in its message).
    ///
    /// Audit #151 bug 1.4 (C4): both `let _ = ...abort_blocking()` calls
    /// below look like they discard a failure signal, but they don't —
    /// `abort_blocking()` is infallible by contract (bug A3, this session):
    /// it always cancels the coordinator and always returns `Ok(())`, even
    /// when the underlying `AbortMultipartUpload` request itself fails.
    /// That failure is not lost — `cancel_coordinator_and_maybe_abort`
    /// (shared by `abort_blocking()` and `Drop`) emits a `tracing::warn!`
    /// naming the bucket/key/upload_id right where the failure occurs, so
    /// there is nothing left for `__exit__` to swallow. See
    /// `abort_blocking_returns_ok_and_never_completes_even_when_s3_abort_fails`
    /// in tests/test_multipart_abort_blocking.rs, which proves this
    /// invariant directly against the mock server's induced-failure mode.
    #[pyo3(text_signature = "(self, exc_type, exc, tb)")]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _t: Py<PyAny>,
        _v: Py<PyAny>,
        _tb: Py<PyAny>,
    ) -> PyResult<()> {
        // Release any outstanding reserve() buffer and its self-pin (see
        // `self_pin_while_pending`'s field doc) — otherwise a reserve()
        // that was never commit()'d would leak the writer permanently.
        self.clear_pending_reservation();
        let mut result: PyResult<()> = Ok(());
        let mut outcome = "the writer was closed".to_string();
        if let Some(mut inner) = self.inner.take() {
            if !_t.bind(py).is_none() {
                // An exception propagated from inside the `with` block — abort the
                // upload and let the original exception propagate unchanged.
                // Safe to discard: see the infallibility note above.
                let _ = py.detach(|| inner.abort_blocking());
                outcome = "the upload was aborted".to_string();
            } else if let Err(e) = py.detach(|| inner.finish_blocking()) {
                // Best-effort cleanup after a failed finish; safe to
                // discard for the same reason as above.
                let _ = py.detach(|| inner.abort_blocking());
                outcome = format!("the upload itself failed: {e}");
                result = Err(py_err(e.context("multipart upload failed")));
            }
            // Drop the sink (and the parts it still queues) before waiting for the buffers.
            drop(inner);
        }
        // Zero-copy writes: no part may reference the caller's buffers when we return. A
        // timeout is raised (the buffers stay pinned); otherwise the upload's own result stands.
        self.release_held_exports(py, &outcome)?;
        result
    }
}

/// Hidden test/benchmark hook: the size below which `MultipartUploadWriter.write()` copies a
/// write once instead of borrowing it. With an argument, sets it (for writers created after the
/// call) and returns the previous value; without, returns the current value.
#[pyfunction]
#[pyo3(signature = (bytes=None))]
fn _mpu_coalesce_below(bytes: Option<usize>) -> usize {
    match bytes {
        Some(n) => crate::multipart::set_coalesce_below(n),
        None => crate::multipart::coalesce_below(),
    }
}

// ---------------------------------------------------------------------------
// Module registration function for advanced API
// ---------------------------------------------------------------------------
pub fn register_advanced_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Multipart upload writer
    m.add_class::<PyMultipartUploadWriter>()?;
    m.add_function(wrap_pyfunction!(_mpu_coalesce_below, m)?)?;

    // TODO: Add checkpoint system when re-enabled
    // #[cfg(feature = "extension-module")]
    // {
    //     m.add_class::<PyCheckpointStore>()?;
    //     m.add_class::<PyCheckpointWriter>()?;
    //     m.add_class::<PyCheckpointReader>()?;
    // }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests: the reuse guard and the commit shrink, driven through the real `write()` /
// `commit()` methods on a writer whose sink has no S3 behind it.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod writer_tests {
    use super::*;
    use crate::multipart::WriterTestGuard;
    use pyo3::types::PyByteArray;

    const MIB: usize = 1 << 20;

    fn writer(
        py: Python<'_>,
        part_size: usize,
        coalesce_below: usize,
    ) -> (Bound<'_, PyMultipartUploadWriter>, WriterTestGuard) {
        let (sink, guard) = MultipartUploadSink::for_writer_test(part_size, coalesce_below);
        let w = PyMultipartUploadWriter {
            inner: Some(sink),
            pending_buf: None,
            self_pin_while_pending: None,
            held: HeldExports::new(),
        };
        (Bound::new(py, w).expect("writer object"), guard)
    }

    fn write(w: &Bound<'_, PyMultipartUploadWriter>, data: &Bound<'_, PyAny>) -> PyResult<usize> {
        w.call_method1("write", (data,))?.extract()
    }

    fn view_of<'py>(data: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        data.py()
            .import("builtins")
            .and_then(|b| b.call_method1("memoryview", (data,)))
            .expect("memoryview")
    }

    fn slice_of<'py>(view: &Bound<'py, PyAny>, from: usize, to: usize) -> Bound<'py, PyAny> {
        view.get_item(pyo3::types::PySlice::new(
            view.py(),
            from as isize,
            to as isize,
            1,
        ))
        .expect("slice")
    }

    /// The `readinto(buf); write(buf)` loop ends with a SHORT chunk (below the coalescing
    /// threshold): it overwrites memory the writer still reads, so it must be refused like any
    /// other reuse. (It used to return before the overlap check.)
    #[test]
    fn a_short_final_chunk_over_a_still_borrowed_buffer_is_refused() {
        Python::initialize();
        Python::attach(|py| {
            let (w, _guard) = writer(py, MIB, 4096);
            let buf = PyByteArray::new(py, &[1u8; 8192]);
            assert_eq!(write(&w, buf.as_any()).expect("first write"), 8192);
            // readinto(buf) refilled the buffer; the last chunk is only 100 bytes long.
            let short = slice_of(&view_of(buf.as_any()), 0, 100);
            let err = write(&w, &short).expect_err("the short chunk reuses borrowed memory");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyBufferError>(py),
                "{err}"
            );
            // A short write of unrelated memory is still fine (copied).
            let other = PyByteArray::new(py, &[2u8; 100]);
            assert_eq!(write(&w, other.as_any()).expect("distinct buffer"), 100);
        });
    }

    /// `memoryview(bytes)` is immutable memory: writing the same view twice is not a reuse,
    /// and nothing is held for it.
    #[test]
    fn a_memoryview_of_bytes_is_not_a_reuse() {
        Python::initialize();
        Python::attach(|py| {
            let (w, _guard) = writer(py, MIB, 4096);
            let data = pyo3::types::PyBytes::new(py, &[3u8; 8192]);
            let view = view_of(data.as_any());
            assert_eq!(write(&w, &view).expect("first"), 8192);
            assert_eq!(write(&w, &view).expect("same immutable view again"), 8192);
            assert!(
                w.borrow().held.is_empty(),
                "an immutable view needs no export and no latch"
            );
        });
    }

    /// `commit(n)` of a small part of a large reservation must not pin the reservation: the
    /// unused capacity is given back (a realloc, not a copy of the payload). Without it each
    /// queued part is a slice that keeps its whole 64 MiB allocation alive.
    #[test]
    fn commit_does_not_pin_the_whole_reservation() {
        Python::initialize();
        Python::attach(|py| {
            // Commits of exactly one part (2 MiB): each becomes a part that is a zero-copy slice
            // of its Vec, so the Vec stays alive as long as the part does.
            let (w, mut guard) = writer(py, 2 * MIB, MIB);
            for _ in 0..4 {
                let mv = w.call_method1("reserve", (64 * MIB,)).expect("reserve");
                w.call_method1("commit", (2 * MIB,)).expect("commit");
                drop(mv);
            }
            let parts = guard.part_capacities();
            assert_eq!(parts.len(), 4);
            for (len, capacity) in parts {
                assert_eq!(len, 2 * MIB);
                assert!(
                    capacity <= 4 * MIB,
                    "a {len}-byte part pins a {} MiB allocation",
                    capacity >> 20
                );
            }
        });
    }
}
