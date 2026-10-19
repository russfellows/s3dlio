// src/python_api/python_advanced_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

#![allow(unsafe_op_in_unsafe_fn)]

use bytes::Bytes;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods};
use pyo3::Bound;
use std::os::raw::c_char;
use std::ptr::NonNull;

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

// ---------------------------------------------------------------------------
// reserve(): the owned allocation and its buffer exporter
// ---------------------------------------------------------------------------

/// The zero-filled allocation behind a `reserve()`: a fixed block of bytes that is never moved,
/// resized or reused, and is freed exactly once, when the LAST owner is gone.
///
/// Owners are `Arc` clones: the writer (while the reservation is pending), the
/// [`PyReservedBuffer`] exporter that every memoryview points at, and the zero-copy `Bytes` that
/// `commit()` hands to the upload. Nothing else frees or reallocates it, which is what lets a
/// view outlive `commit()`, `close()`, `abort()` and the writer itself.
///
/// It is a raw allocation rather than a `Vec` so that nothing (a `shrink_to_fit`, a `push`) can
/// ever relocate it while a Python view or an in-flight part still points into it, and so that
/// an allocation failure is an error to report, not an abort.
pub(crate) struct ReservedBlock {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: a plain heap allocation of bytes with no thread affinity; it stays allocated (never
// moved or freed) while any owner lives. Sharing it across threads is sound only under the
// documented contract of `reserve()`, the same as for a borrowed `write()`: the caller does not
// write through a retained view while the committed payload is in flight. Keeping the allocation
// alive does not make such a concurrent write harmless (it is a data race on the bytes the
// upload reads), and a payload checksum is not guaranteed to detect it.
unsafe impl Send for ReservedBlock {}
unsafe impl Sync for ReservedBlock {}

impl ReservedBlock {
    /// A zero-filled block of `size` bytes, or `None` when the size cannot be allocated.
    fn zeroed(size: usize) -> Option<Self> {
        if size == 0 {
            // Nothing to allocate (and a zero-size allocation is not allowed): a dangling,
            // well-aligned pointer is valid for a zero-length view.
            return Some(Self {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        let layout = std::alloc::Layout::array::<u8>(size).ok()?;
        // SAFETY: `layout` has a non-zero size.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        NonNull::new(raw).map(|ptr| Self { ptr, len: size })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for ReservedBlock {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // The layout was valid when the block was allocated, so it is valid again here.
        if let Ok(layout) = std::alloc::Layout::array::<u8>(self.len) {
            // SAFETY: `ptr` came from `alloc_zeroed` with this exact layout and is freed once.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

/// `Bytes` owner that shares a [`ReservedBlock`]: how `commit()` gives the upload the reserved
/// memory without copying it. Dropping the last `Bytes` slice drops this reference.
struct ReservedOwner(Arc<ReservedBlock>);

impl AsRef<[u8]> for ReservedOwner {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the block is `len` initialised (zero-filled) bytes that stay allocated while
        // this `Arc` lives. The shared slice is sound only while nobody writes to those bytes:
        // `reserve()`'s contract forbids writing through a view after `commit()` (see the
        // `Send`/`Sync` note on `ReservedBlock`).
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), self.0.len()) }
    }
}

/// The owner behind the memoryview that `reserve()` returns (``mv.obj``).
///
/// A memoryview built from this object by `PyMemoryView_FromObject` holds a reference to it for
/// as long as the view, any slice or sub-view of it, or any object exporting from it exists, and
/// it holds the allocation: the memory therefore cannot be freed under a retained view. It is
/// writable, and never moves.
#[pyclass(name = "ReservedBuffer", module = "s3dlio")]
pub struct PyReservedBuffer {
    block: Arc<ReservedBlock>,
    /// `shape[0]` for the buffer protocol; lives in this heap object so the pointer handed to
    /// the view stays valid as long as the view holds a reference to this object.
    shape_len: isize,
}

impl PyReservedBuffer {
    fn new(block: Arc<ReservedBlock>) -> Self {
        // A `Vec`-sized allocation never exceeds `isize::MAX` (`Layout::array` rejects it).
        let shape_len = block.len() as isize;
        Self { block, shape_len }
    }
}

#[pymethods]
impl PyReservedBuffer {
    fn __len__(&self) -> usize {
        self.block.len()
    }

    fn __repr__(&self) -> String {
        format!("ReservedBuffer({} bytes)", self.block.len())
    }

    /// Writable buffer protocol over the reservation. Every request succeeds (the buffer is
    /// always writable and C-contiguous); the view keeps this object, and so the allocation,
    /// alive by holding a reference to it in `view.obj`.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("null Py_buffer"));
        }
        // Unit stride: 1 byte per element, a 'static address.
        static UNIT_STRIDE: isize = 1;
        // SAFETY: `view` is the valid `Py_buffer` CPython passed in; every pointer stored in
        // it stays valid while `view.obj` (this object) is alive, and CPython keeps it alive.
        unsafe {
            (*view).buf = slf.block.as_ptr().cast::<std::os::raw::c_void>();
            (*view).len = slf.shape_len;
            (*view).readonly = 0;
            (*view).itemsize = 1;
            (*view).ndim = 1;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) != 0 {
                c"B".as_ptr() as *mut c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).shape = if (flags & ffi::PyBUF_ND) != 0 {
                &slf.shape_len as *const isize as *mut isize
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) != 0 {
                &UNIT_STRIDE as *const isize as *mut isize
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            (*view).obj = slf.as_ptr();
            ffi::Py_INCREF((*view).obj);
        }
        Ok(())
    }

    /// Nothing to release: CPython drops the reference taken in `__getbuffer__`, and the
    /// allocation is freed only when no owner is left.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
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
    /// The reservation between `reserve()` and `commit()`: the writer's OWN reference to the
    /// allocation behind the memoryview that `reserve()` returned. The memoryview's exporter
    /// ([`PyReservedBuffer`]) holds another one, and `commit()` hands the upload a third (a
    /// zero-copy `Bytes`), so the memory lives until the writer, every Python view and every
    /// part in flight are all done with it. Dropping this field (`close`, `abort`, `__exit__`,
    /// deleting the writer) therefore never frees memory a view can still reach.
    pending_reservation: Option<Arc<ReservedBlock>>,
    /// Completion barriers of the borrowed buffer exports behind `write()` (zero-copy default):
    /// the parts still reference the caller's memory until they are uploaded, so `close()` /
    /// `abort()` / `__exit__` wait on every latch (bounded) before they return.
    held: HeldExports,
}

impl PyMultipartUploadWriter {
    /// Drop the writer's reference to a pending `reserve()` allocation. Idempotent. Only the
    /// WRITER's reference goes: a memoryview (or slice of one) the caller still holds keeps the
    /// memory alive through its exporter, so this can never leave a view dangling.
    fn clear_pending_reservation(&mut self) {
        self.pending_reservation = None;
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
            pending_reservation: None,
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
            pending_reservation: None,
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
    /// Fill the memoryview in Python (``mv[:n] = ...``, ``f.readinto(mv)``,
    /// ``np.frombuffer(mv)``), then call `commit(nbytes)`. Only one pending buffer is allowed.
    ///
    /// The memory is owned by an exporter object (``mv.obj``, a ``s3dlio.ReservedBuffer``) that
    /// every view and every slice of it keeps alive. It is never moved or resized, and it is
    /// freed only when the writer, every view and the upload itself are all done with it:
    /// ``close()``, ``abort()``, leaving a ``with`` block, deleting the writer, and
    /// ``commit()`` never invalidate a view you still hold; reading it stays safe.
    ///
    /// After ``commit()`` the bytes you committed are being uploaded straight out of this
    /// memory (zero copy). Do not write through a retained view after ``commit()``: that
    /// changes the payload in flight while the upload reads it: the stored object may contain
    /// a mix of old and new bytes, and a signed part may (but is not guaranteed to) fail its
    /// payload checksum. The same contract applies to a borrowed ``write()``.
    /// A SHORT commit keeps the whole reservation allocated until its part is uploaded and
    /// every view is gone (the allocation is not shrunk, which would be a realloc that may
    /// move it): reserve what you will fill.
    ///
    /// Args:
    ///     size: number of bytes to reserve (zero-filled; ``MemoryError`` if unavailable)
    ///
    /// Returns:
    ///     memoryview: writable view into the reserved buffer
    #[pyo3(text_signature = "(self, size, /)")]
    fn reserve(&mut self, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
        // Disallow overlapping reserves
        if self.pending_reservation.is_some() {
            return Err(PyRuntimeError::new_err(
                "reserve() called while a previous buffer is pending; call commit() first",
            ));
        }
        let block = ReservedBlock::zeroed(size).map(Arc::new).ok_or_else(|| {
            pyo3::exceptions::PyMemoryError::new_err(format!("cannot reserve {size} bytes"))
        })?;
        let exporter = Py::new(py, PyReservedBuffer::new(Arc::clone(&block)))?;
        // `PyMemoryView_FromObject` goes through the buffer protocol, so the memoryview (and
        // every slice or sub-view derived from it) holds a reference to `exporter` in
        // `view.obj`, and the exporter holds the allocation. Nothing here can outlive that.
        // SAFETY: `exporter` is a live, exclusively typed Python object that implements the
        // buffer protocol; the returned pointer is a new reference or null with an error set.
        let raw = unsafe { ffi::PyMemoryView_FromObject(exporter.as_ptr()) };
        if raw.is_null() {
            return Err(PyErr::fetch(py));
        }
        // SAFETY: `raw` is a non-null owned reference.
        let mv = unsafe { Py::<PyAny>::from_owned_ptr(py, raw) };
        self.pending_reservation = Some(block);
        Ok(mv)
    }

    /// Commit the previously reserved buffer.
    ///
    /// The first ``nbytes`` bytes become the next piece of the object. They are uploaded
    /// straight out of the reserved memory, with no copy (a commit below the small-write
    /// threshold is copied once into the part buffer instead, like any small write). The
    /// reservation stays allocated until the upload and every memoryview of it are gone; see
    /// ``reserve()`` for what that means for a short commit and for writing through a view
    /// after this call.
    ///
    /// Args:
    ///     nbytes: number of bytes actually written into the reserved buffer
    ///
    /// Errors if no buffer is pending (``RuntimeError``) or if `nbytes` exceeds the reserved
    /// size (``ValueError``; the reservation then stays pending and can still be committed
    /// with a valid length).
    #[pyo3(text_signature = "(self, nbytes, /)")]
    fn commit(&mut self, py: Python<'_>, nbytes: usize) -> PyResult<()> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        let block = self.pending_reservation.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("no pending buffer; call reserve() first")
        })?;
        // Validate BEFORE consuming anything: a bad length leaves the reservation pending.
        if nbytes > block.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "commit(nbytes={nbytes}) exceeds reserved size {}",
                block.len()
            )));
        }
        // The upload shares ownership of the allocation (an `Arc` clone, no copy of the
        // payload) and is sliced to what was filled.
        let bytes = Bytes::from_owner(ReservedOwner(Arc::clone(block))).slice(..nbytes);
        // The reservation is consumed from here on, whatever the sink does with it.
        self.pending_reservation = None;
        py.detach(|| inner.write_bytes_blocking(bytes))
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
        // Drop the writer's reference to an outstanding reserve() allocation. A memoryview
        // the caller still holds keeps the memory alive through its exporter.
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
        // Drop the writer's reference to an outstanding reserve() allocation. A memoryview
        // the caller still holds keeps the memory alive through its exporter.
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
        // Drop the writer's reference to an outstanding reserve() allocation. A memoryview
        // the caller still holds keeps the memory alive through its exporter.
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
    m.add_class::<PyReservedBuffer>()?;
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
// Unit tests: the reuse guard and the ownership of `reserve()` memory, driven through the real
// `write()` / `reserve()` / `commit()` methods on a writer whose sink has no S3 behind it.
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
            pending_reservation: None,
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

    // -- reserve() / commit(): who owns the memory a view points at (stage G review, G-F1) ----
    //
    // Every check below is about OWNERSHIP, read from the `Arc` of the allocation through a
    // `Weak` (which does not perturb it): no test relies on a use of freed memory happening to
    // crash.

    /// A look at the allocation behind a reservation that does not keep it alive itself.
    struct Probe {
        weak: std::sync::Weak<ReservedBlock>,
        ptr: usize,
    }

    impl Probe {
        /// Taken while the writer holds the reservation.
        fn of(w: &Bound<'_, PyMultipartUploadWriter>) -> Self {
            let borrowed = w.borrow();
            let block = borrowed
                .pending_reservation
                .as_ref()
                .expect("a pending reservation");
            Self {
                weak: Arc::downgrade(block),
                ptr: block.as_ptr() as usize,
            }
        }

        /// Owners right now: the writer while pending, each exporter, each queued slice.
        fn owners(&self) -> usize {
            self.weak.strong_count()
        }

        fn freed(&self) -> bool {
            self.weak.upgrade().is_none()
        }
    }

    fn reserve<'py>(w: &Bound<'py, PyMultipartUploadWriter>, size: usize) -> Bound<'py, PyAny> {
        w.call_method1("reserve", (size,)).expect("reserve")
    }

    fn fill(mv: &Bound<'_, PyAny>, at: usize, data: &[u8]) {
        let py = mv.py();
        mv.set_item(
            pyo3::types::PySlice::new(py, at as isize, (at + data.len()) as isize, 1),
            pyo3::types::PyBytes::new(py, data),
        )
        .expect("write through the view");
    }

    fn read(mv: &Bound<'_, PyAny>, from: usize, to: usize) -> Vec<u8> {
        slice_of(mv, from, to)
            .call_method0("tobytes")
            .and_then(|b| b.extract())
            .expect("read through the view")
    }

    fn pending(w: &Bound<'_, PyMultipartUploadWriter>) -> bool {
        w.borrow().pending_reservation.is_some()
    }

    /// `reserve()` still returns a writable `memoryview`; its `.obj` is the exporter that owns
    /// the memory, and a slice of it shares that exporter.
    #[test]
    fn reserve_returns_a_memoryview_owned_by_the_exporter() {
        Python::initialize();
        Python::attach(|py| {
            let (w, _guard) = writer(py, 2 * MIB, 0);
            let mv = reserve(&w, 4096);
            let probe = Probe::of(&w);
            assert_eq!(probe.owners(), 2, "the writer and the exporter");
            let type_name = mv.get_type().name().expect("type name").to_string();
            assert_eq!(type_name, "memoryview");
            assert!(!mv.getattr("readonly").unwrap().is_truthy().unwrap());
            assert_eq!(mv.len().unwrap(), 4096);
            let exporter = mv.getattr("obj").expect("memoryview.obj");
            assert!(exporter.is_instance_of::<PyReservedBuffer>());
            assert!(
                slice_of(&mv, 10, 20).getattr("obj").unwrap().is(&exporter),
                "a slice shares the exporter"
            );
            assert_eq!(
                probe.owners(),
                2,
                "views add references to the exporter only"
            );
            // The ordinary uses of the view keep working.
            fill(&mv, 0, b"abcd");
            assert_eq!(read(&mv, 0, 4), b"abcd");
            assert_eq!(
                read(&mv, 4, 8),
                [0u8; 4],
                "a reservation starts zero-filled"
            );
        });
    }

    /// A successful commit gives the upload the very same allocation (no copy); the memory
    /// stays until the upload's slice AND the view are gone, in either order.
    #[test]
    fn successful_commit_shares_the_allocation_with_the_upload() {
        Python::initialize();
        Python::attach(|py| {
            let (w, mut guard) = writer(py, 2 * MIB, 0);
            let mv = reserve(&w, 2 * MIB);
            let probe = Probe::of(&w);
            fill(&mv, 0, b"abcdefgh");
            w.call_method1("commit", (2 * MIB,)).expect("commit");
            assert!(!pending(&w), "the reservation was consumed");
            let chunks = guard.take_chunks();
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].as_ptr() as usize, probe.ptr, "zero-copy");
            assert_eq!(&chunks[0][..8], b"abcdefgh");
            assert_eq!(probe.owners(), 2, "the exporter and the queued part");
            // The view still reads and the memory is still there after the part is gone.
            drop(chunks);
            assert_eq!(probe.owners(), 1);
            assert_eq!(read(&mv, 0, 8), b"abcdefgh");
            let exporter = mv.getattr("obj").unwrap();
            drop(mv);
            assert!(!probe.freed(), "the exporter object is still referenced");
            drop(exporter);
            assert!(probe.freed(), "freed once every owner is gone");
        });
    }

    /// A short commit slices the same allocation (it is NOT shrunk or moved) and pins the whole
    /// reservation while its part is in flight; the view stays valid throughout.
    #[test]
    fn short_commit_keeps_the_whole_allocation_in_place() {
        Python::initialize();
        Python::attach(|py| {
            let (w, mut guard) = writer(py, 2 * MIB, 0);
            let mv = reserve(&w, 8 * MIB);
            let probe = Probe::of(&w);
            fill(&mv, 0, b"short");
            w.call_method1("commit", (5,)).expect("short commit");
            assert!(!pending(&w));
            // The 5 bytes wait in the sink for the rest of their part until flush().
            assert_eq!(
                probe.owners(),
                2,
                "the exporter and the sink's pending slice"
            );
            w.call_method0("flush").expect("flush");
            let chunks = guard.take_chunks();
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].len(), 5);
            assert_eq!(
                chunks[0].as_ptr() as usize,
                probe.ptr,
                "not moved, not copied"
            );
            assert_eq!(&chunks[0][..], b"short");
            assert_eq!(probe.owners(), 2);
            assert_eq!(read(&mv, 0, 5), b"short", "the view is still valid");
            drop(chunks);
            assert_eq!(probe.owners(), 1);
            drop(mv);
            assert!(probe.freed());
        });
    }

    /// An invalid length is a `ValueError` that consumes nothing: the reservation stays
    /// pending, its memory untouched, and a valid commit still works.
    #[test]
    fn invalid_length_commit_leaves_the_reservation_pending() {
        Python::initialize();
        Python::attach(|py| {
            let (w, mut guard) = writer(py, MIB, 0);
            let mv = reserve(&w, MIB);
            let probe = Probe::of(&w);
            fill(&mv, 0, b"keep");
            let err = w
                .call_method1("commit", (MIB + 1,))
                .expect_err("longer than the reservation");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "{err}"
            );
            assert!(pending(&w), "still pending");
            assert_eq!(probe.owners(), 2, "writer and exporter still own it");
            assert_eq!(read(&mv, 0, 4), b"keep");
            // A second reserve() is still refused, and the valid commit goes through.
            assert!(w.call_method1("reserve", (16,)).is_err());
            w.call_method1("commit", (MIB,)).expect("valid commit");
            let chunks = guard.take_chunks();
            assert_eq!(chunks.len(), 1);
            assert_eq!(chunks[0].as_ptr() as usize, probe.ptr);
            assert_eq!(&chunks[0][..4], b"keep");
        });
    }

    /// Whichever way the writer lets go of a pending reservation, a view the caller holds (and
    /// a slice that outlives the original memoryview) stays backed by live, correct memory.
    fn views_survive(py: Python<'_>, release: impl FnOnce(Bound<'_, PyMultipartUploadWriter>)) {
        let (w, _guard) = writer(py, MIB, 0);
        let mv = reserve(&w, 4096);
        let probe = Probe::of(&w);
        fill(&mv, 100, b"still here");
        let tail = slice_of(&mv, 100, 110);
        release(w);
        assert_eq!(probe.owners(), 1, "only the exporter is left");
        assert_eq!(read(&mv, 100, 110), b"still here");
        // The original view goes; the derived slice alone keeps the memory.
        drop(mv);
        assert!(
            !probe.freed(),
            "a derived slice outlives the original memoryview"
        );
        assert_eq!(
            tail.call_method0("tobytes")
                .unwrap()
                .extract::<Vec<u8>>()
                .unwrap(),
            b"still here"
        );
        fill(&tail, 0, b"STILL");
        assert_eq!(read(&tail, 0, 10), b"STILL here");
        drop(tail);
        assert!(probe.freed(), "freed with the last view");
    }

    #[test]
    fn close_with_a_pending_reservation_leaves_views_backed() {
        Python::initialize();
        Python::attach(|py| {
            views_survive(py, |w| {
                // The test sink has no coordinator, so the upload itself reports an error;
                // the reservation is released either way.
                assert!(w.call_method0("close").is_err());
                assert!(!pending(&w));
            })
        });
    }

    #[test]
    fn abort_with_a_pending_reservation_leaves_views_backed() {
        Python::initialize();
        Python::attach(|py| {
            views_survive(py, |w| {
                w.call_method0("abort").expect("abort");
                assert!(!pending(&w));
            })
        });
    }

    #[test]
    fn context_exit_with_a_pending_reservation_leaves_views_backed() {
        Python::initialize();
        Python::attach(|py| {
            // Leaving the `with` block on an exception aborts.
            views_survive(py, |w| {
                let exc = py
                    .get_type::<pyo3::exceptions::PyValueError>()
                    .call1(("boom",))
                    .expect("exception");
                let exc_type = exc.get_type();
                w.call_method1("__exit__", (exc_type, exc, py.None()))
                    .expect("exit after an exception aborts");
                assert!(!pending(&w));
            });
            // Leaving it normally closes (the test sink then reports its missing coordinator).
            views_survive(py, |w| {
                assert!(w
                    .call_method1("__exit__", (py.None(), py.None(), py.None()))
                    .is_err());
                assert!(!pending(&w));
            });
        });
    }

    #[test]
    fn deleting_the_writer_with_a_pending_reservation_leaves_views_backed() {
        Python::initialize();
        Python::attach(|py| {
            views_survive(py, |w| drop(w));
        });
    }

    /// After a commit the view and the writer can go in any order: the upload's slice, the view
    /// and the writer each hold the memory, and it is freed with the last of them.
    #[test]
    fn views_and_parts_outlive_the_writer_after_a_commit() {
        Python::initialize();
        Python::attach(|py| {
            let (w, mut guard) = writer(py, MIB, 0);
            let mv = reserve(&w, MIB);
            let probe = Probe::of(&w);
            fill(&mv, 0, b"payload");
            w.call_method1("commit", (MIB,)).expect("commit");
            let tail = slice_of(&mv, 0, 7);
            drop(w);
            drop(mv);
            let chunks = guard.take_chunks();
            assert_eq!(&chunks[0][..7], b"payload");
            assert_eq!(
                probe.owners(),
                2,
                "the queued part and the exporter (via the slice)"
            );
            drop(tail);
            assert_eq!(probe.owners(), 1, "the part alone");
            assert_eq!(&chunks[0][..7], b"payload", "the part's memory is intact");
            drop(chunks);
            assert!(probe.freed());
        });
    }

    /// `reserve(0)` works: an empty view, and `commit(0)` (but not `commit(1)`).
    #[test]
    fn a_zero_byte_reservation_works() {
        Python::initialize();
        Python::attach(|py| {
            let (w, _guard) = writer(py, MIB, 0);
            let mv = reserve(&w, 0);
            assert_eq!(mv.len().unwrap(), 0);
            assert!(mv
                .getattr("obj")
                .unwrap()
                .is_instance_of::<PyReservedBuffer>());
            assert_eq!(read(&mv, 0, 0), b"");
            let err = w.call_method1("commit", (1,)).expect_err("1 > 0");
            assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(pending(&w));
            w.call_method1("commit", (0,)).expect("commit(0)");
            assert!(!pending(&w));
        });
    }

    /// A reservation that cannot be allocated is a `MemoryError`, not an abort.
    #[test]
    fn an_impossible_reservation_is_a_memory_error() {
        Python::initialize();
        Python::attach(|py| {
            let (w, _guard) = writer(py, MIB, 0);
            let err = w
                .call_method1("reserve", (usize::MAX / 2,))
                .expect_err("cannot allocate");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyMemoryError>(py),
                "{err}"
            );
            assert!(!pending(&w));
        });
    }
}
