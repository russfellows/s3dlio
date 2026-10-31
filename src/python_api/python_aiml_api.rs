// src/python_api/python_aiml_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

#![allow(unsafe_op_in_unsafe_fn)]

use super::async_bridge::future_into_py;
use bytes::Bytes;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{
    PyBufferError, PyRuntimeError, PyStopAsyncIteration, PyStopIteration, PyTypeError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyDictMethods, PyList};
use pyo3::Bound;

use futures_util::StreamExt;
use std::str::FromStr;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::task::JoinSet;

use numpy::ndarray as nd_np;
use numpy::PyArrayDyn;

use std::sync::Arc;

// Project crates
use crate::api::dataset_for_uri_with_options;
#[cfg(feature = "extension-module")]
use crate::checkpoint::{CheckpointConfig, CheckpointInfo, CheckpointStore, Strategy};
use crate::data_loader::{
    dataset::{blocking_recv_locked, require_known_length, Dataset, DatasetError},
    options::{LoaderOptions, MemoryFormat, MultiprocessingContext, ReaderMode, SamplerType},
    parallel_fetch::DropCancel,
    s3_bytes::S3BytesDataset,
    DataLoader,
};
use crate::object_store::store_for_uri_with_helper_rows;
#[cfg(feature = "extension-module")]
use crate::python_api::put_payload::{put_payload, quiesce_blocking, PayloadUse};
#[cfg(feature = "extension-module")]
use crate::python_api::python_core_api::ExportStillReferencedError;
use crate::python_api::python_core_api::PyBytesView;
use crate::s3_utils::get_object_uri;
use futures::stream::FuturesUnordered;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

// Import io_uring-style submit from s3_client (never calls block_on)
use crate::s3_client::run_on_global_rt;

// Error-to-PyErr conversion, shared with python_core_api.rs — see
// error_chain_message() there for why this preserves the anyhow cause
// chain instead of just the outermost context label (storage#755 /
// s3dlio#161).
use crate::python_api::python_core_api::py_err;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyS3Dataset {
    inner: S3BytesDataset,
}

#[pymethods]
impl PyS3Dataset {
    /// NEW: optional opts (reader mode)
    #[new]
    #[pyo3(signature = (uri, opts=None))]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let o = opts_from_dict(opts);
        let ds = S3BytesDataset::from_prefix_with_opts(uri, &o)
            //.map_err(|e| PyRuntimeError::new_err(e))?;
            .map_err(py_err)?;
        Ok(Self { inner: ds })
    }
    /// Optional: expose keys for debugging
    fn keys(&self) -> Vec<String> {
        self.inner.keys().clone()
    }
}

// Generic dataset that works with all URI schemes
#[pyclass(from_py_object)]
pub struct PyDataset {
    inner: Arc<dyn Dataset<Item = Bytes>>,
}

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (uri, opts=None))]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let o = opts_from_dict(opts);
        let dataset = dataset_for_uri_with_options(uri, &o).map_err(py_err)?;
        Ok(Self {
            inner: Arc::from(dataset),
        })
    }

    /// Get the number of items in the dataset.
    ///
    /// Raises `TypeError` for iterable-style datasets whose length is not
    /// known up front (matches Python convention: a `__len__` that can't
    /// answer should raise, not lie with 0 — design doc
    /// docs/DESIGN_TIER4_FFI_HARDENING.md item 2).
    fn __len__(&self) -> PyResult<usize> {
        require_known_length(self.inner.len()).map_err(PyTypeError::new_err)
    }

    /// Get an item by index (for map-style datasets)
    /// Returns a PyBytesView for zero-copy access
    fn get_item<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Py<PyAny>> {
        let inner = Arc::clone(&self.inner);
        // The dataset captured the op-log logger when it was constructed and scopes every fetch
        // with it (#294), so the awaitable needs nothing here.
        let bound_result = future_into_py(py, async move {
            let data = inner.get(index).await.map_err(py_err)?;
            // Return PyBytesView for zero-copy access; the Python object is created inside the
            // exit-safe delivery (issue #302), not on this tokio worker.
            Ok(PyBytesView::new(data))
        })?;
        Ok(bound_result.unbind())
    }

    /// Return list of keys if supported (for map-style datasets)
    fn keys(&self) -> PyResult<Vec<String>> {
        self.inner
            .keys()
            .ok_or_else(|| PyRuntimeError::new_err("keys() not supported for this dataset type"))
    }

    /// The URI that fetches each item, in the order of `keys()` (issue #223).
    ///
    /// Built by the dataset from the root it parsed, so Python never rebuilds a URI
    /// from the dataset URI and a listed key.
    fn uris(&self) -> PyResult<Vec<String>> {
        self.inner
            .uris()
            .ok_or_else(|| PyRuntimeError::new_err("uris() not supported for this dataset type"))
    }

    /// Create a dataset from a pre-built list of URIs — no listing occurs.
    ///
    /// This is the general-purpose alternative to ``PyDataset(uri)`` (which
    /// lists all objects under a prefix URI).  Use it whenever you already
    /// have a manifest file, a per-worker shard, or any other pre-built URI
    /// collection.
    ///
    /// All URIs must use the same scheme (``s3://``, ``file://``, ``az://``,
    /// …).  The backend is inferred from the first entry.
    ///
    /// Parameters
    /// ----------
    /// uris : list[str]
    ///     Full URIs of every object in the dataset.  Must be non-empty.
    /// opts : dict or None
    ///     Accepted for signature compatibility and ignored; dataset options are
    ///     set on the loader.  (Before 0.10.0 this keyword was named ``_opts``.)
    ///
    /// Returns
    /// -------
    /// PyDataset
    ///     A map-style dataset whose ``len()`` equals ``len(uris)``.
    ///     Pass it to ``PyBytesAsyncDataLoader`` for sliding-window I/O.
    ///
    /// Examples
    /// --------
    /// >>> uris = ["s3://bucket/shard/a.jpg", "s3://bucket/shard/b.jpg"]
    /// >>> ds = s3dlio.PyDataset.from_uris(uris)
    /// >>> loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 64})
    /// >>> for item in loader:          # GIL released while waiting
    /// ...     process(len(item))       # bytes per object
    #[staticmethod]
    #[pyo3(signature = (uris, opts=None))]
    // Accepted and ignored: `api::dataset_from_uris` takes no options; the name is the documented Python keyword.
    #[allow(unused_variables)]
    fn from_uris(uris: Vec<String>, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let ds = crate::api::dataset_from_uris(uris).map_err(py_err)?;
        Ok(Self {
            inner: Arc::from(ds),
        })
    }
}

impl Clone for PyDataset {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// Generic async loader that works with PyDataset
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyBytesAsyncDataLoader {
    dataset: PyDataset,
    opts: LoaderOptions,
}

#[pymethods]
impl PyBytesAsyncDataLoader {
    #[new]
    #[pyo3(signature = (dataset, opts=None))]
    fn new(dataset: PyDataset, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            dataset,
            opts: opts_from_dict(opts),
        })
    }

    fn __aiter__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Py<PyBytesAsyncDataLoaderIter>> {
        PyBytesAsyncDataLoaderIter::spawn_stream(py, slf.dataset.clone(), slf.opts.clone())
    }

    /// Synchronous iterator — `for item in loader:` works in plain Python (no asyncio).
    ///
    /// The producer runs as a Tokio task using `buffer_unordered(prefetch)` to drive
    /// up to `prefetch` concurrent dataset fetches in flight simultaneously.  Tokio's
    /// own work-stealing thread pool determines the actual parallelism — no manual
    /// thread-count tuning required; it adapts automatically from 8-core laptops to
    /// 128-core servers.
    ///
    /// The bounded channel (capacity = `prefetch`) is the backpressure mechanism:
    /// when Python is slow the producer blocks on `tx.send().await`; when Python is
    /// fast the channel stays ahead.  No Semaphore, no JoinSet, no guessing.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyBytesDataLoaderSyncIter>> {
        let prefetch = slf.opts.prefetch.max(1);
        let dataset = slf.dataset.clone();

        // Fail loudly rather than silently yielding zero items -- see
        // docs/DESIGN_TIER4_FFI_HARDENING.md item 2. There is currently no
        // Python-exposed streaming path for unknown-length datasets.
        require_known_length(dataset.inner.len()).map_err(PyTypeError::new_err)?;

        // Activate process-wide HEAD-skip latch if requested.
        if slf.opts.skip_head {
            crate::s3_utils::set_skip_head(true);
        }

        // Bounded channel: capacity = prefetch depth.
        // Producer backs off automatically when Python consumer is slow.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, DatasetError>>(prefetch);

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            if let Some(len) = dataset.inner.len() {
                // Task-level parallelism (issue #148 site 1.1): each fetch is
                // spawned onto tokio's scheduler so requests can be polled
                // across worker threads instead of funneled through this one
                // producer task. `DropCancel` on our stack + `select!` inside
                // each spawned task means: producer exit for ANY reason
                // (normal end, Python dropped the iterator so tx.send fails,
                // producer panic) cancels every in-flight fetch, dropping
                // their I/O. Same pattern as `AsyncPoolDataLoader` site 3.1a.
                let cancel = CancellationToken::new();
                let _drop_cancel = DropCancel(cancel.clone());

                use std::time::Instant;

                let t_start = Instant::now();
                let mut total_bytes: u64 = 0;
                let mut count: usize = 0;
                let mut t_last_log = Instant::now();

                tracing::info!(
                    total_items = len,
                    prefetch,
                    "[s3dlio] loader starting: {} items, prefetch depth {}",
                    len,
                    prefetch
                );

                let spawn_fetch = |idx: usize| -> JoinHandle<Result<Bytes, DatasetError>> {
                    let ds = dataset.clone();
                    let token = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            _ = token.cancelled() => {
                                Err(DatasetError::from("fetch cancelled"))
                            }
                            r = ds.inner.get(idx) => r,
                        }
                    })
                };

                let mut pending: FuturesUnordered<JoinHandle<Result<Bytes, DatasetError>>> =
                    FuturesUnordered::new();
                let mut next_idx = 0usize;

                // Fill initial window.
                while next_idx < len && pending.len() < prefetch {
                    pending.push(spawn_fetch(next_idx));
                    next_idx += 1;
                }

                while let Some(join_res) = pending.next().await {
                    let result: Result<Bytes, DatasetError> = match join_res {
                        Ok(r) => r,
                        Err(join_err) if join_err.is_panic() => Err(DatasetError::from(format!(
                            "fetch task panicked: {}",
                            join_err
                        ))),
                        // Cancelled during shutdown — skip cleanly.
                        Err(_) => continue,
                    };

                    if let Ok(ref bytes) = result {
                        total_bytes += bytes.len() as u64;
                        count += 1;

                        // Log progress every 20 items OR every 5 seconds
                        if count.is_multiple_of(20) || t_last_log.elapsed().as_secs() >= 5 {
                            let elapsed = t_start.elapsed().as_secs_f64();
                            tracing::info!(
                                count,
                                total_items = len,
                                total_bytes,
                                elapsed_s = elapsed,
                                "[s3dlio] progress: {}/{} items | {:.1} MiB | {:.0} MB/s",
                                count,
                                len,
                                total_bytes as f64 / 1024.0 / 1024.0,
                                total_bytes as f64 / elapsed / 1024.0 / 1024.0
                            );
                            t_last_log = Instant::now();
                        }
                    }
                    // blocks when channel full → natural backpressure
                    if tx.send(result).await.is_err() {
                        break; // Python consumer dropped the iterator
                    }
                    // Refill the sliding window.
                    if next_idx < len {
                        pending.push(spawn_fetch(next_idx));
                        next_idx += 1;
                    }
                }

                let elapsed = t_start.elapsed().as_secs_f64();
                tracing::info!(
                    count,
                    total_bytes,
                    elapsed_s = elapsed,
                    "[s3dlio] loader done: {}/{} items | {:.1} MiB | {:.0} MB/s",
                    count,
                    len,
                    total_bytes as f64 / 1024.0 / 1024.0,
                    if elapsed > 0.0 {
                        total_bytes as f64 / elapsed / 1024.0 / 1024.0
                    } else {
                        0.0
                    }
                );
            }
        });

        Py::new(
            slf.py(),
            PyBytesDataLoaderSyncIter {
                rx: parking_lot::Mutex::new(rx),
            },
        )
    }

    /// Sliding-window iterator that yields `PyObjectItem` in network-completion order.
    ///
    /// Unlike `__iter__()` (which returns bare `PyBytesView`), `items()` attaches the
    /// full URI to every result so Python always knows *which* object arrived — even
    /// when completions are out of submission order.
    ///
    /// Concurrency model
    /// -----------------
    /// Tokio keeps exactly ``prefetch`` GET requests in flight at all times via
    /// `buffer_unordered`.  Each item Python consumes frees one slot and immediately
    /// triggers one new GET — a true sliding window with no chunk boundaries, no
    /// thundering-herd spikes, and no idle periods between chunks.
    ///
    /// Backpressure
    /// ------------
    /// The mpsc channel capacity equals ``prefetch``.  If Python is slow the producer
    /// blocks on ``tx.send().await``; if Python is fast the channel stays full.  No
    /// semaphores, no polling, no explicit rate-limiting required.
    ///
    /// Example
    /// -------
    /// ```python
    /// uris = ["s3://bucket/a.jpg", "s3://bucket/b.jpg"]
    /// ds = s3dlio.PyDataset.from_uris(uris)
    /// loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 64})
    /// for item in loader.items():
    ///     process(item.uri, len(item))
    /// ```
    fn items(slf: PyRef<'_, Self>) -> PyResult<Py<PyObjectDataLoaderSyncIter>> {
        use std::sync::Arc;
        use std::time::Instant;

        let prefetch = slf.opts.prefetch.max(1);
        let dataset = slf.dataset.clone();

        // Fail loudly rather than silently yielding zero items -- see
        // docs/DESIGN_TIER4_FFI_HARDENING.md item 2. There is currently no
        // Python-exposed streaming path for unknown-length datasets.
        require_known_length(dataset.inner.len()).map_err(PyTypeError::new_err)?;

        // Activate process-wide HEAD-skip latch if requested.
        if slf.opts.skip_head {
            crate::s3_utils::set_skip_head(true);
        }

        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<(String, Bytes), DatasetError>>(prefetch);

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            if let Some(len) = dataset.inner.len() {
                // Task-level parallelism (issue #148 site 1.1 — items()
                // variant). See the __iter__ version above for the full
                // rationale; the only difference here is that each fetched
                // item is paired with its URI/key so completions arrive as
                // (uri, bytes).
                let cancel = CancellationToken::new();
                let _drop_cancel = DropCancel(cancel.clone());

                // Snapshot keys once — attaches URI to each completed item.
                // Fallback to "item:<idx>" for datasets that don't expose keys().
                let keys: Arc<Option<Vec<String>>> = Arc::new(dataset.inner.keys());

                let t_start = Instant::now();
                let mut total_bytes: u64 = 0;
                let mut count: usize = 0;

                tracing::info!(
                    total_items = len,
                    prefetch,
                    "[s3dlio] items() starting: {} items, {} in-flight",
                    len,
                    prefetch,
                );

                let spawn_fetch =
                    |idx: usize| -> JoinHandle<Result<(String, Bytes), DatasetError>> {
                        let ds = dataset.clone();
                        let k = Arc::clone(&keys);
                        let token = cancel.clone();
                        tokio::spawn(async move {
                            let uri = k
                                .as_ref()
                                .as_ref()
                                .and_then(|v| v.get(idx).cloned())
                                .unwrap_or_else(|| format!("item:{}", idx));
                            tokio::select! {
                                _ = token.cancelled() => {
                                    Err(DatasetError::from("fetch cancelled"))
                                }
                                r = ds.inner.get(idx) => r.map(|b| (uri, b)),
                            }
                        })
                    };

                let mut pending: FuturesUnordered<
                    JoinHandle<Result<(String, Bytes), DatasetError>>,
                > = FuturesUnordered::new();
                let mut next_idx = 0usize;

                while next_idx < len && pending.len() < prefetch {
                    pending.push(spawn_fetch(next_idx));
                    next_idx += 1;
                }

                while let Some(join_res) = pending.next().await {
                    let result: Result<(String, Bytes), DatasetError> = match join_res {
                        Ok(r) => r,
                        Err(join_err) if join_err.is_panic() => Err(DatasetError::from(format!(
                            "fetch task panicked: {}",
                            join_err
                        ))),
                        Err(_) => continue,
                    };

                    if let Ok((_, ref bytes)) = result {
                        total_bytes += bytes.len() as u64;
                        count += 1;
                        if count.is_multiple_of(500) {
                            let elapsed = t_start.elapsed().as_secs_f64();
                            tracing::info!(
                                "[s3dlio] items() progress: {}/{} | {:.0} MB/s",
                                count,
                                len,
                                total_bytes as f64 / elapsed.max(0.001) / 1_048_576.0,
                            );
                        }
                    }
                    if tx.send(result).await.is_err() {
                        break; // Python dropped the iterator
                    }
                    if next_idx < len {
                        pending.push(spawn_fetch(next_idx));
                        next_idx += 1;
                    }
                }

                let elapsed = t_start.elapsed().as_secs_f64();
                tracing::info!(
                    "[s3dlio] items() done: {}/{} | {:.1} MiB | {:.0} MB/s",
                    count,
                    len,
                    total_bytes as f64 / 1_048_576.0,
                    total_bytes as f64 / elapsed.max(0.001) / 1_048_576.0,
                );
            }
            // Note: iterable-style datasets (as_stream) have no per-item URI;
            // use __iter__() for those. items() is for map-style datasets where
            // every item has a stable, known URI.
        });

        Py::new(
            slf.py(),
            PyObjectDataLoaderSyncIter {
                rx: parking_lot::Mutex::new(rx),
            },
        )
    }
}

/// A single object returned by `PyBytesAsyncDataLoader.items()`.
///
/// Carries both the URI and the raw bytes so Python never has to maintain a
/// parallel index to know *which* object arrived.  Because s3dlio's Tokio
/// runtime delivers items in network-completion order, the URI is the only
/// reliable way to identify each item on the Python side.
///
/// Attributes (Python)
/// -------------------
/// ``item.uri``        — full URI of the object (``s3://…``, ``file://…``, …)
/// ``item.byte_count`` — number of bytes transferred (same as ``len(item)``)
/// ``memoryview(item)`` / ``np.frombuffer(item)`` — read the object in place (zero copy)
/// ``item.data``       — the object as a read-only ``BytesView`` (zero copy)
/// ``bytes(item)``     — raw bytes; creates a Python ``bytes`` copy (documented compatibility copy)
/// ``len(item)``       — alias for ``byte_count``
#[pyclass]
pub struct PyObjectItem {
    #[pyo3(get)]
    pub uri: String,
    data: Bytes,
    /// `data.len()` cached in the heap object: the buffer protocol hands out a pointer to it as
    /// `shape`, which must stay valid as long as any memoryview of this item lives.
    shape_len: isize,
}

impl PyObjectItem {
    fn new(uri: String, data: Bytes) -> Self {
        let shape_len = data.len() as isize;
        Self {
            uri,
            data,
            shape_len,
        }
    }
}

#[pymethods]
impl PyObjectItem {
    #[getter]
    fn byte_count(&self) -> usize {
        self.data.len()
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// The object as a read-only `BytesView`: zero copies (it shares this item's buffer).
    #[getter]
    fn data(&self) -> PyBytesView {
        PyBytesView::new(self.data.clone())
    }

    /// Return the raw bytes as a Python ``bytes`` object. This is the COPY: use `memoryview(item)`,
    /// `np.frombuffer(item)` or `item.data` to read the object in place.
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    /// Read-only buffer protocol: `memoryview(item)` / `np.frombuffer(item, dtype=...)` read the
    /// fetched object in place, with no copy.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        fill_read_only_bytes_buffer(view, flags, slf.as_ptr(), &slf.data, &slf.shape_len)
    }

    /// Nothing to release: CPython drops the reference taken in `__getbuffer__`.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    fn __repr__(&self) -> String {
        format!(
            "PyObjectItem(uri={:?}, byte_count={})",
            self.uri,
            self.data.len()
        )
    }
}

/// Fill `view` as a read-only, C-contiguous 1-D `uint8` buffer over `bytes`, kept alive by `obj`
/// (the Python object that owns `bytes` and `shape_len`; this takes a reference to it).
///
/// Mirrors `BytesView.__getbuffer__`. `shape_len` must live in `obj`'s heap object and equal
/// `bytes.len()`, so the `shape` pointer outlives the view. A writable request is a `BufferError`.
///
/// # Safety
/// `view` must be the valid `Py_buffer` CPython passed to `__getbuffer__`, and `obj` the object
/// whose `__getbuffer__` is running, with `bytes`/`shape_len` fields of that object.
unsafe fn fill_read_only_bytes_buffer(
    view: *mut ffi::Py_buffer,
    flags: std::os::raw::c_int,
    obj: *mut ffi::PyObject,
    bytes: &Bytes,
    shape_len: &isize,
) -> PyResult<()> {
    if view.is_null() {
        return Err(PyBufferError::new_err("null Py_buffer"));
    }
    if (flags & ffi::PyBUF_WRITABLE) != 0 {
        return Err(PyBufferError::new_err(
            "this object is read-only and does not support writable buffers",
        ));
    }
    // Unit stride: 1 byte per element, a 'static address.
    static UNIT_STRIDE: isize = 1;
    unsafe {
        (*view).buf = bytes.as_ptr() as *mut std::os::raw::c_void;
        (*view).len = bytes.len() as isize;
        (*view).readonly = 1;
        (*view).itemsize = 1;
        (*view).ndim = 1;
        (*view).format = if (flags & ffi::PyBUF_FORMAT) != 0 {
            c"B".as_ptr() as *mut std::os::raw::c_char
        } else {
            std::ptr::null_mut()
        };
        (*view).shape = if (flags & ffi::PyBUF_ND) != 0 {
            shape_len as *const isize as *mut isize
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
        // Keep the owning object (and so the `Bytes`) alive for the lifetime of the view.
        (*view).obj = obj;
        ffi::Py_INCREF(obj);
    }
    Ok(())
}

/// Synchronous iterator returned by `PyBytesAsyncDataLoader.items()`.
///
/// Yields `PyObjectItem` values in network-completion order — whichever GET
/// finishes first is delivered first, eliminating tail-latency stalls.
///
/// The GIL is released on every `__next__()` call while waiting for the
/// Tokio channel; other Python threads (including other DataLoader workers)
/// run freely during that wait.
#[pyclass]
pub struct PyObjectDataLoaderSyncIter {
    rx: parking_lot::Mutex<tokio::sync::mpsc::Receiver<Result<(String, Bytes), DatasetError>>>,
}

#[pymethods]
impl PyObjectDataLoaderSyncIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyObjectItem>> {
        // Release GIL while the channel is empty (Tokio GETs in flight).
        let result = py.detach(|| blocking_recv_locked(&self.rx));

        match result {
            Some(Ok((uri, data))) => Py::new(py, PyObjectItem::new(uri, data)),
            Some(Err(e)) => Err(py_err(e)),
            None => Err(PyStopIteration::new_err("end of dataset")),
        }
    }

    /// Collect up to `n` completed items from the Rust/Tokio sliding window and
    /// return them as a Python **list** of `PyObjectItem`.
    ///
    /// # GIL behaviour
    ///
    /// The GIL is released **once** for the entire batch.  Inside the
    /// `py.detach()` block, `blocking_recv()` is called up to `n` times
    /// without ever reacquiring the GIL.  After all `n` items (or end-of-stream)
    /// are collected into a plain Rust `Vec`, the GIL is reacquired exactly
    /// **once** to wrap each `(uri, Bytes)` pair into a `PyObjectItem`.
    ///
    /// Contrast with calling `__next__()` n times: that would be n GIL
    /// release + reacquire cycles.  Here there is exactly 1, regardless of n.
    ///
    /// # Return value
    ///
    /// Returns an empty list when the stream is exhausted.  The caller should
    /// treat an empty return as the stop signal:
    ///
    /// ```python
    /// item_iter = loader.items()
    /// while batch := item_iter.collect_batch(16):
    ///     for item in batch:
    ///         process(item)
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates any `DatasetError` from a failed GET as a Python
    /// `RuntimeError`, aborting the batch.
    #[pyo3(signature = (n))]
    fn collect_batch(&self, py: Python<'_>, n: usize) -> PyResult<Vec<Py<PyObjectItem>>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        // Release GIL for the entire drain loop — no Python objects created here.
        let raw: Vec<Result<(String, Bytes), DatasetError>> = py.detach(|| {
            let mut rx = self.rx.lock();
            let mut buf = Vec::with_capacity(n);
            for _ in 0..n {
                match rx.blocking_recv() {
                    Some(item) => buf.push(item),
                    None => break, // channel closed — stream exhausted
                }
            }
            buf
        });

        // GIL reacquired exactly once here.  Build Python objects from the
        // collected raw tuples.
        let mut items = Vec::with_capacity(raw.len());
        for result in raw {
            match result {
                Ok((uri, data)) => items.push(Py::new(py, PyObjectItem::new(uri, data))?),
                Err(e) => return Err(py_err(e)),
            }
        }
        Ok(items)
    }
}

/// Synchronous iterator returned by `PyBytesAsyncDataLoader.__iter__()`.
///
/// Each `__next__()` call releases the GIL (`py.allow_threads`) while waiting
/// for the next item from the Tokio producer, then re-acquires it to wrap the
/// result in a `PyBytesView`.  Other Python threads run freely while waiting.
///
/// Each item is one row-group's bytes (when using the parquet format) or one
/// object's bytes (when using the default S3 bytes format).
#[pyclass]
pub struct PyBytesDataLoaderSyncIter {
    rx: parking_lot::Mutex<tokio::sync::mpsc::Receiver<Result<Bytes, DatasetError>>>,
}

#[pymethods]
impl PyBytesDataLoaderSyncIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Release the GIL while the Tokio channel is empty (network I/O in flight).
        // When the next item arrives the GIL is re-acquired automatically and `py`
        // is valid again — no Python::attach / Python::with_gil needed.
        let result = py.detach(|| blocking_recv_locked(&self.rx));

        match result {
            Some(Ok(item)) => Py::new(py, PyBytesView::new(item))?.into_py_any(py),
            Some(Err(e)) => Err(py_err(e)),
            None => Err(PyStopIteration::new_err("end of dataset")),
        }
    }
}

#[pyclass]
#[allow(clippy::type_complexity)]
pub struct PyBytesAsyncDataLoaderIter {
    rx: Arc<Mutex<mpsc::Receiver<Result<Vec<Bytes>, DatasetError>>>>,
}

impl PyBytesAsyncDataLoaderIter {
    fn spawn_stream(py: Python<'_>, dataset: PyDataset, opts: LoaderOptions) -> PyResult<Py<Self>> {
        // Fail loudly rather than silently yielding zero items -- see
        // docs/DESIGN_TIER4_FFI_HARDENING.md item 2. There is currently no
        // Python-exposed streaming path for unknown-length datasets.
        require_known_length(dataset.inner.len()).map_err(PyTypeError::new_err)?;

        let (tx, rx) = mpsc::channel::<Result<Vec<Bytes>, DatasetError>>(opts.prefetch.max(1));

        // Extract concurrency setting for concurrent batch fetching
        // Default to 8 workers if num_workers is 0
        let num_workers = if opts.num_workers > 0 {
            opts.num_workers
        } else {
            8
        };

        // Spawn the producer task with concurrent batch fetching
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let batch_size = opts.batch_size.max(1);
            let semaphore = Arc::new(Semaphore::new(num_workers));

            if let Some(len) = dataset.inner.len() {
                let mut i = 0;
                while i < len {
                    // Collect indices for this batch
                    let batch_indices: Vec<usize> = (0..batch_size)
                        .filter_map(|j| {
                            let idx = i + j;
                            if idx < len {
                                Some(idx)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if batch_indices.is_empty() {
                        break;
                    }

                    // Fetch batch items CONCURRENTLY using JoinSet
                    let mut join_set = JoinSet::new();
                    for (order, idx) in batch_indices.iter().enumerate() {
                        let dataset_clone = dataset.clone();
                        let sem = Arc::clone(&semaphore);
                        let idx = *idx;

                        join_set.spawn(async move {
                            let _permit = sem.acquire().await.unwrap();
                            let data = dataset_clone.inner.get(idx).await?;
                            // Now returns Bytes directly - zero-copy!
                            Ok::<(usize, Bytes), DatasetError>((order, data))
                        });
                    }

                    // Collect results preserving order
                    let mut batch_results: Vec<Option<Bytes>> = vec![None; batch_indices.len()];
                    while let Some(result) = join_set.join_next().await {
                        match result {
                            Ok(Ok((order, data))) => {
                                batch_results[order] = Some(data);
                            }
                            Ok(Err(e)) => {
                                let _ = tx.send(Err(e)).await;
                                return;
                            }
                            Err(join_err) => {
                                let _ =
                                    tx.send(Err(DatasetError::from(join_err.to_string()))).await;
                                return;
                            }
                        }
                    }

                    // Convert to batch — flatten() drops None results
                    let batch: Vec<Bytes> = batch_results.into_iter().flatten().collect();

                    if !batch.is_empty() && tx.send(Ok(batch)).await.is_err() {
                        break;
                    }
                    i += batch_size;
                }
            }
        });

        Py::new(
            py,
            Self {
                rx: Arc::new(Mutex::new(rx)),
            },
        )
    }
}

#[pymethods]
impl PyBytesAsyncDataLoaderIter {
    fn __anext__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let rx = Arc::clone(&slf.rx);
        let bound_result = future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                // A list of PyBytesView (zero-copy), built inside the exit-safe delivery (issue #302).
                Some(Ok(batch)) => Ok(batch.into_iter().map(PyBytesView::new).collect::<Vec<_>>()),
                Some(Err(e)) => Err(py_err(e)),
                None => Err(PyStopAsyncIteration::new_err("StopAsyncIteration")),
            }
        })?;
        Ok(bound_result.unbind())
    }
}

// async loader over S3BytesDataset
#[pyclass]
#[allow(clippy::type_complexity)]
pub struct PyS3AsyncDataLoader {
    rx: Arc<Mutex<mpsc::Receiver<Result<Vec<Bytes>, DatasetError>>>>,
}

// AsyncDataLoader function
#[pymethods]
impl PyS3AsyncDataLoader {
    #[new]
    #[pyo3(signature = (uri, opts=None))]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        // Parse options once, and apply safe defaults for range mode tuning.
        let o = opts_from_dict(opts)
            .part_size(8 * 1024 * 1024)
            .max_inflight_parts(4);

        // Build the dataset with the SAME options (reader_mode, part_size, etc.)
        let ds = S3BytesDataset::from_prefix_with_opts(uri, &o).map_err(py_err)?;

        // Use the SAME options for DataLoader and for channel capacity.
        let (tx, rx) = mpsc::channel::<Result<Vec<Bytes>, DatasetError>>(o.prefetch.max(1));

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let loader = DataLoader::new(ds, o);
            let mut stream = loader.stream();
            while let Some(batch) = stream.next().await {
                if tx.send(batch).await.is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            rx: Arc::new(Mutex::new(rx)),
        })
    }

    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let bound_result = future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                // A list of PyBytesView (zero-copy), built inside the exit-safe delivery (issue #302).
                Some(Ok(batch)) => Ok(batch.into_iter().map(PyBytesView::new).collect::<Vec<_>>()),
                Some(Err(e)) => Err(py_err(e)),
                None => Err(PyStopAsyncIteration::new_err("end of stream")),
            }
        })?;
        Ok(bound_result.unbind())
    }
}

// ---------------------------------------------------------------------------
// NPZ reader
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (uri, array_name = None))]
pub fn read_npz(py: Python<'_>, uri: &str, array_name: Option<&str>) -> PyResult<Py<PyAny>> {
    use crate::data_formats::npz::{list_npz_arrays, read_npz_array};

    let bytes = get_object_uri(uri).map_err(py_err)?;

    // Determine array name
    let name = if let Some(n) = array_name {
        n.to_string()
    } else {
        // Get first array name from archive
        let names = list_npz_arrays(&bytes).map_err(py_err)?;
        names
            .into_iter()
            .next()
            .ok_or_else(|| PyRuntimeError::new_err("NPZ file is empty"))?
    };

    // Read array using our custom implementation. RULE #1: a STORED (uncompressed) entry is
    // sliced out of the fetched object in place (`npz_entry`), never copied. The ONE copy is the
    // format decode: the little-endian f32 payload is decoded into the owned, aligned `Vec<f32>`
    // below, which the numpy array then takes over without another copy (the entry's bytes are
    // only 1-byte aligned inside the archive, so numpy cannot view them in place).
    let arr: ndarray::ArrayD<f32> = read_npz_array(&bytes, &name).map_err(py_err)?;

    let shape = arr.shape().to_vec();
    // Moves the `Vec<f32>` out of the ndarray (no copy) and into the numpy array.
    let (data, _offset) = arr.into_raw_vec_and_offset();
    let arr_np = nd_np::ArrayD::from_shape_vec(nd_np::IxDyn(&shape), data).map_err(py_err)?;
    let py_arr = PyArrayDyn::<f32>::from_owned_array(py, arr_np);
    py_arr.into_py_any(py)
}

// ---------------------------------------------------------------------------
// LoaderOptions Python wrapper with AI/ML realism knobs
// ---------------------------------------------------------------------------
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyLoaderOptions {
    inner: LoaderOptions,
}

#[pymethods]
impl PyLoaderOptions {
    #[new]
    fn new() -> Self {
        Self {
            inner: LoaderOptions::default(),
        }
    }

    // Basic options
    fn with_batch_size(&mut self, size: usize) -> Self {
        Self {
            inner: self.inner.clone().with_batch_size(size),
        }
    }

    fn drop_last(&mut self, yes: bool) -> Self {
        Self {
            inner: self.inner.clone().drop_last(yes),
        }
    }

    fn shuffle(&mut self, on: bool, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().shuffle(on, seed),
        }
    }

    fn num_workers(&mut self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().num_workers(n),
        }
    }

    fn prefetch(&mut self, n: usize) -> Self {
        Self {
            inner: self.inner.clone().prefetch(n),
        }
    }

    fn auto_tune(&mut self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().auto_tune(on),
        }
    }

    // AI/ML realism knobs
    fn pin_memory(&mut self, pin: bool) -> Self {
        Self {
            inner: self.inner.clone().pin_memory(pin),
        }
    }

    fn persistent_workers(&mut self, persistent: bool) -> Self {
        Self {
            inner: self.inner.clone().persistent_workers(persistent),
        }
    }

    fn with_timeout(&mut self, seconds: f64) -> Self {
        Self {
            inner: self.inner.clone().with_timeout(seconds),
        }
    }

    fn use_spawn(&mut self) -> Self {
        Self {
            inner: self.inner.clone().use_spawn(),
        }
    }

    fn use_fork(&mut self) -> Self {
        Self {
            inner: self.inner.clone().use_fork(),
        }
    }

    fn use_forkserver(&mut self) -> Self {
        Self {
            inner: self.inner.clone().use_forkserver(),
        }
    }

    fn random_sampling(&mut self, replacement: bool) -> Self {
        Self {
            inner: self.inner.clone().random_sampling(replacement),
        }
    }

    fn distributed_sampling(&mut self, rank: usize, world_size: usize) -> Self {
        Self {
            inner: self.inner.clone().distributed_sampling(rank, world_size),
        }
    }

    fn channels_last(&mut self) -> Self {
        Self {
            inner: self.inner.clone().channels_last(),
        }
    }

    fn channels_first(&mut self) -> Self {
        Self {
            inner: self.inner.clone().channels_first(),
        }
    }

    fn non_blocking(&mut self, non_blocking: bool) -> Self {
        Self {
            inner: self.inner.clone().non_blocking(non_blocking),
        }
    }

    fn with_generator_seed(&mut self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().with_generator_seed(seed),
        }
    }

    fn enable_transforms(&mut self, enable: bool) -> Self {
        Self {
            inner: self.inner.clone().enable_transforms(enable),
        }
    }

    fn collate_buffer_size(&mut self, size: usize) -> Self {
        Self {
            inner: self.inner.clone().collate_buffer_size(size),
        }
    }

    // Convenience presets
    fn gpu_optimized(&mut self) -> Self {
        Self {
            inner: self.inner.clone().gpu_optimized(),
        }
    }

    fn distributed_optimized(&mut self, rank: usize, world_size: usize) -> Self {
        Self {
            inner: self.inner.clone().distributed_optimized(rank, world_size),
        }
    }

    fn debug_mode(&mut self) -> Self {
        Self {
            inner: self.inner.clone().debug_mode(),
        }
    }

    // Property accessors.
    //
    // Python cannot hold a method and a property with one name, and the builder
    // methods (`pin_memory(True)`, `num_workers(8)`, ...) must keep working, so the
    // seven values shadowed by a builder are exposed as `<name>_value` (#283).
    // `batch_size`, `seed`, `timeout_seconds`, `shard_rank` and `shard_world_size`
    // have no same-named builder and keep their plain property names.
    #[getter]
    fn get_batch_size(&self) -> usize {
        self.inner.batch_size
    }

    #[getter]
    fn get_drop_last_value(&self) -> bool {
        self.inner.drop_last
    }

    #[getter]
    fn get_shuffle_value(&self) -> bool {
        self.inner.shuffle
    }

    #[getter]
    fn get_seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn get_num_workers_value(&self) -> usize {
        self.inner.num_workers
    }

    #[getter]
    fn get_prefetch_value(&self) -> usize {
        self.inner.prefetch
    }

    #[getter]
    fn get_pin_memory_value(&self) -> bool {
        self.inner.pin_memory
    }

    #[getter]
    fn get_persistent_workers_value(&self) -> bool {
        self.inner.persistent_workers
    }

    #[getter]
    fn get_timeout_seconds(&self) -> Option<f64> {
        self.inner.timeout_seconds
    }

    #[getter]
    fn get_non_blocking_value(&self) -> bool {
        self.inner.non_blocking
    }

    #[getter]
    fn get_shard_rank(&self) -> usize {
        self.inner.shard_rank
    }

    #[getter]
    fn get_shard_world_size(&self) -> usize {
        self.inner.shard_world_size
    }

    /// Return every option that `PyLoaderOptions` can set as a plain dict keyed by
    /// option name (#283).
    ///
    /// Options that cannot be set from Python (for example `loading_mode`,
    /// `reader_mode`, `part_size`, `max_inflight_parts`, `worker_id`) are not
    /// included. Purely additive: the builder methods are unchanged. Enum-valued
    /// options are reported by lower-case name. `sampler_type` is one of
    /// `"sequential"`, `"random"`, `"weighted_random"`, `"distributed_random"`;
    /// its parameters are reported losslessly in `sampler_replacement` (`bool`
    /// for `"random"`, else `None`) and `sampler_rank` / `sampler_world_size`
    /// (`int` for `"distributed_random"`, else `None`).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let o = &self.inner;
        let d = PyDict::new(py);
        d.set_item("batch_size", o.batch_size)?;
        d.set_item("drop_last", o.drop_last)?;
        d.set_item("shuffle", o.shuffle)?;
        d.set_item("seed", o.seed)?;
        d.set_item("num_workers", o.num_workers)?;
        d.set_item("prefetch", o.prefetch)?;
        d.set_item("auto_tune", o.auto_tune)?;
        d.set_item("pin_memory", o.pin_memory)?;
        d.set_item("persistent_workers", o.persistent_workers)?;
        d.set_item("timeout_seconds", o.timeout_seconds)?;
        d.set_item("non_blocking", o.non_blocking)?;
        d.set_item("shard_rank", o.shard_rank)?;
        d.set_item("shard_world_size", o.shard_world_size)?;
        d.set_item("generator_seed", o.generator_seed)?;
        d.set_item("enable_transforms", o.enable_transforms)?;
        d.set_item("collate_buffer_size", o.collate_buffer_size)?;
        d.set_item(
            "multiprocessing_context",
            match o.multiprocessing_context {
                MultiprocessingContext::Spawn => "spawn",
                MultiprocessingContext::Fork => "fork",
                MultiprocessingContext::ForkServer => "forkserver",
            },
        )?;
        d.set_item(
            "memory_format",
            match o.memory_format {
                MemoryFormat::ChannelsLast => "channels_last",
                MemoryFormat::ChannelsFirst => "channels_first",
                MemoryFormat::Auto => "auto",
            },
        )?;
        d.set_item(
            "sampler_type",
            match o.sampler_type {
                SamplerType::Sequential => "sequential",
                SamplerType::Random { .. } => "random",
                SamplerType::WeightedRandom { .. } => "weighted_random",
                SamplerType::DistributedRandom { .. } => "distributed_random",
            },
        )?;
        let replacement = match o.sampler_type {
            SamplerType::Random { replacement } => Some(replacement),
            _ => None,
        };
        let (rank, world_size) = match o.sampler_type {
            SamplerType::DistributedRandom { rank, world_size } => (Some(rank), Some(world_size)),
            _ => (None, None),
        };
        d.set_item("sampler_replacement", replacement)?;
        d.set_item("sampler_rank", rank)?;
        d.set_item("sampler_world_size", world_size)?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// DataLoader bindings
// ---------------------------------------------------------------------------
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyVecDataset {
    inner: Vec<i32>,
}

#[pymethods]
impl PyVecDataset {
    #[new]
    fn new(obj: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: obj.extract()?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<i32> {
        self.inner
            .get(idx)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err(format!("Index {} out of range", idx)))
    }
}

#[async_trait::async_trait]
impl Dataset for PyVecDataset {
    type Item = i32;
    fn len(&self) -> Option<usize> {
        Some(self.inner.len())
    }
    async fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        self.inner
            .get(idx)
            .copied()
            .ok_or(DatasetError::IndexOutOfRange(idx))
    }
}

/// Convert an optional Python dict of options into LoaderOptions.
fn opts_from_dict(d: Option<Bound<'_, PyDict>>) -> LoaderOptions {
    let def = LoaderOptions::default();

    if let Some(d) = d {
        // Helper closures with better error handling
        let g_usize = |k: &str, dv: usize| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<usize>().unwrap_or_else(|_| {
                eprintln!(
                    "Warning: Invalid value for '{}', expected usize, using default {}",
                    k, dv
                );
                dv
            }),
            _ => dv,
        };
        let g_bool = |k: &str, dv: bool| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<bool>().unwrap_or_else(|_| {
                eprintln!(
                    "Warning: Invalid value for '{}', expected bool, using default {}",
                    k, dv
                );
                dv
            }),
            _ => dv,
        };
        let g_u64 = |k: &str, dv: u64| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<u64>().unwrap_or_else(|_| {
                eprintln!(
                    "Warning: Invalid value for '{}', expected u64, using default {}",
                    k, dv
                );
                dv
            }),
            _ => dv,
        };

        // Reader mode: accept "sequential" or "range" (case-insensitive).
        let reader_mode = match d.get_item("reader_mode") {
            Ok(Some(val)) => {
                if let Ok(s) = val.extract::<&str>() {
                    match s.to_ascii_lowercase().as_str() {
                        "range" => ReaderMode::Range,
                        "sequential" => ReaderMode::Sequential,
                        _ => def.reader_mode,
                    }
                } else {
                    def.reader_mode
                }
            }
            _ => def.reader_mode,
        };

        // Check for unknown keys and warn about them
        let known_keys = [
            "batch_size",
            "drop_last",
            "shuffle",
            "seed",
            "num_workers",
            "prefetch",
            "auto_tune",
            "reader_mode",
            "part_size",
            "max_inflight_parts",
            "shard_rank",
            "shard_world_size",
            "worker_id",
            "num_workers_pytorch",
            // Format-routing keys consumed by create_async_loader / create_dataset
            "format",
            "columns",
            "footer_cap",
            "decode", // "raw" (default) | "arrow" (Rust Arrow IPC) | "none" (raw, no Python decode)
            "max_threads", // limit pyo3 Tokio runtime threads (MPI: set to num_cpus / world_size)
            "skip_head", // skip HEAD requests; sets process-wide latch (see s3_utils::set_skip_head)
        ];
        for key in d.keys() {
            if let Ok(key_str) = key.extract::<&str>() {
                if !known_keys.contains(&key_str) {
                    eprintln!(
                        "Warning: Unknown option '{}' ignored. Valid options: {:?}",
                        key_str, known_keys
                    );
                }
            }
        }

        LoaderOptions {
            // existing fields
            batch_size: g_usize("batch_size", def.batch_size),
            drop_last: g_bool("drop_last", def.drop_last),
            shuffle: g_bool("shuffle", def.shuffle),
            seed: g_u64("seed", def.seed),
            num_workers: g_usize("num_workers", def.num_workers),
            prefetch: g_usize("prefetch", def.prefetch),
            auto_tune: g_bool("auto_tune", def.auto_tune),
            skip_head: g_bool("skip_head", def.skip_head),

            // new fields
            loading_mode: def.loading_mode, // Use default loading mode
            reader_mode,
            part_size: g_usize("part_size", def.part_size),
            max_inflight_parts: g_usize("max_inflight_parts", def.max_inflight_parts),

            shard_rank: g_usize("shard_rank", def.shard_rank),
            shard_world_size: g_usize("shard_world_size", def.shard_world_size),
            worker_id: g_usize("worker_id", def.worker_id),
            num_workers_pytorch: g_usize("num_workers_pytorch", def.num_workers_pytorch),

            // AI/ML realism knobs (use defaults for now)
            pin_memory: def.pin_memory,
            persistent_workers: def.persistent_workers,
            timeout_seconds: def.timeout_seconds,
            multiprocessing_context: def.multiprocessing_context,
            sampler_type: def.sampler_type,
            memory_format: def.memory_format,
            non_blocking: def.non_blocking,
            generator_seed: def.generator_seed,
            enable_transforms: def.enable_transforms,
            collate_buffer_size: def.collate_buffer_size,
            page_cache_mode: def.page_cache_mode, // Use default (Auto)
            adaptive: def.adaptive,               // Use default (None/disabled)
            cancellation_token: def.cancellation_token, // Use default (None)
        }
    } else {
        def
    }
}

#[pyclass]
pub struct PyAsyncDataLoader {
    dataset: PyVecDataset,
    opts: LoaderOptions,
}

/// This is the Py Wrapper for the Async Data Loader
#[pymethods]
impl PyAsyncDataLoader {
    #[new]
    #[pyo3(signature = (dataset, opts=None))]
    fn new(dataset: PyVecDataset, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            dataset,
            opts: opts_from_dict(opts),
        })
    }

    fn __aiter__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Py<PyAsyncDataLoaderIter>> {
        PyAsyncDataLoaderIter::spawn_stream(py, slf.dataset.clone(), slf.opts.clone())
    }
}

#[pyclass]
#[allow(clippy::type_complexity)]
pub struct PyAsyncDataLoaderIter {
    rx: Arc<Mutex<mpsc::Receiver<Result<Vec<i32>, DatasetError>>>>,
}

/// This is the actual Async Data Loader
impl PyAsyncDataLoaderIter {
    fn spawn_stream(
        py: Python<'_>,
        dataset: PyVecDataset,
        opts: LoaderOptions,
    ) -> PyResult<Py<Self>> {
        let (tx, rx) = mpsc::channel::<Result<Vec<i32>, DatasetError>>(opts.prefetch.max(1));

        //
        //Spawn the producer task using the runtime managed by
        // `pyo3_async_runtimes`; this ensures a reactor is active.
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let loader = DataLoader::new(dataset, opts);
            let mut stream = loader.stream();
            while let Some(batch) = stream.next().await {
                if tx.send(batch).await.is_err() {
                    break;
                }
            }
        });

        Py::new(
            py,
            Self {
                rx: Arc::new(Mutex::new(rx)),
            },
        )
    }
}

#[pymethods]
impl PyAsyncDataLoaderIter {
    fn __anext__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let rx = slf.rx.clone();
        let bound_result = future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                // Converted inside the exit-safe delivery (issue #302).
                Some(Ok(batch)) => Ok(batch),
                Some(Err(e)) => Err(py_err(e)),
                None => Err(PyStopAsyncIteration::new_err("end of loader")),
            }
        })?;
        Ok(bound_result.unbind())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint Python bindings
// ---------------------------------------------------------------------------

/// Build the dedicated Tokio multi-thread runtime backing a
/// `PyCheckpointStore`. Checkpointing intentionally uses its own runtime
/// rather than s3dlio's shared global one (so long checkpoint I/O can't
/// starve the data-loading pipeline's runtime, or vice versa); this
/// function is the one place that decides how many worker threads it gets.
///
/// Sized via [`crate::constants::effective_thread_budget`] -- the same
/// MPI-aware, `S3DLIO_RT_THREADS`-overridable budget used by
/// [`crate::s3_client::configure_thread_pools`] -- instead of Tokio's own
/// unconstrained `num_cpus` default, so N MPI ranks checkpointing
/// concurrently on one host split the available cores instead of each
/// independently claiming all of them.
#[cfg(feature = "extension-module")]
fn build_checkpoint_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    crate::runtime_support::marked_multi_thread_builder()
        .worker_threads(crate::constants::effective_thread_budget())
        .enable_all()
        .thread_name("s3dlio-checkpoint")
        .build()
}

/// Run a blocking checkpoint save over `data`, read IN PLACE (RULE #1, zero copy).
///
/// `data` may be `bytes`, a `BytesView` or any buffer-protocol object (bytearray, memoryview,
/// numpy array, ...). The payload becomes the `Bytes` the writer hands to the object store
/// (no `to_vec`, no `copy_from_slice`): for a buffer export the `Py_buffer` is the `Bytes`'
/// owner, read with the GIL released, exactly like `put_bytes`. The save blocks until the object
/// is written, so holding the export for the call is correct; the call does not return - with
/// success or an error - until every native reader is done and the export is released
/// (`put_payload.rs`, "The completion barrier"), else it raises `ExportStillReferencedError`.
/// While it runs the caller must not modify the buffer from another thread.
///
/// An unsupported type is a `TypeError` naming what is accepted.
#[cfg(feature = "extension-module")]
fn with_blocking_payload<T, F>(py: Python<'_>, data: &Bound<'_, PyAny>, run: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce(Bytes) -> PyResult<T> + Send,
{
    let payload = put_payload(py, data, PayloadUse::Blocking)?;
    let bytes = payload.bytes;
    let result = py.detach(move || run(bytes));
    if let Some(latch) = &payload.latch {
        if quiesce_blocking(py, latch).is_err() {
            let outcome = match &result {
                Ok(_) => "the save itself succeeded".to_string(),
                Err(e) => format!("the save itself failed: {e}"),
            };
            return Err(ExportStillReferencedError::new_err(format!(
                "a storage backend still references the buffer {:?} after the save finished \
                 ({outcome}); the buffer stays pinned - do not modify, resize or free it \
                 (S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS)",
                crate::constants::export_quiesce_timeout()
            )));
        }
    }
    result
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCheckpointStore {
    inner: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCheckpointStore {
    #[new]
    #[pyo3(signature = (uri, strategy=None, multipart_threshold=None, compression_level=None))]
    fn new(
        uri: String,
        strategy: Option<String>,
        multipart_threshold: Option<usize>,
        compression_level: Option<i32>,
    ) -> PyResult<Self> {
        let mut config = CheckpointConfig::new();

        if let Some(strat_str) = strategy {
            let strat = Strategy::from_str(&strat_str)
                .map_err(|e| py_err(format!("Invalid strategy: {}", e)))?;
            config = config.with_strategy(strat);
        }

        if let Some(threshold) = multipart_threshold {
            config = config.with_multipart_threshold(threshold);
        }

        // Add compression configuration
        if let Some(level) = compression_level {
            use crate::object_store::CompressionConfig;
            let compression_config = CompressionConfig::zstd_level(level);
            config = config.with_compression(compression_config);
        }

        let store = CheckpointStore::open_with_config(&uri, config)
            .map_err(|e| py_err(e.context("Failed to open checkpoint store")))?;

        let runtime = Arc::new(
            build_checkpoint_runtime()
                .map_err(|e| py_err(format!("Failed to create runtime: {}", e)))?,
        );

        Ok(Self {
            inner: Arc::new(Mutex::new(store)),
            runtime,
        })
    }

    /// Save a single-rank checkpoint
    #[pyo3(
        signature = (step, epoch, framework, data, user_meta=None),
        text_signature = "(self, step, epoch, framework, data, user_meta=None)"
    )]
    fn save(
        &self,
        py: Python<'_>,
        step: u64,
        epoch: u64,
        framework: String,
        data: &Bound<'_, PyAny>,
        user_meta: Option<Py<PyAny>>,
    ) -> PyResult<String> {
        let user_meta = user_meta.map(|_obj| {
            // For now, just use null for user metadata
            // TODO: Implement proper Python->JSON conversion
            serde_json::Value::Null
        });

        let inner = self.inner.clone();

        // `data` is read in place (any buffer-protocol object), not copied: see
        // `with_blocking_payload`.
        with_blocking_payload(py, data, |data| {
            self.runtime
                .block_on(async {
                    let store = inner.lock().await;
                    store.save(step, epoch, &framework, data, user_meta).await
                })
                .map_err(|e| py_err(e.context("Save failed")))
        })
    }

    /// Load the latest checkpoint
    #[pyo3(text_signature = "(self)")]
    fn load_latest(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let inner = self.inner.clone();

        py.detach(|| {
            self.runtime.block_on(async {
                let store = inner.lock().await;
                store.load_latest().await
            })
        })
        .map_err(|e| py_err(e.context("Load failed")))
        .map(|opt_data| {
            opt_data
                // exit-gate: allowed (sync pymethod: runs on the calling Python thread, which is already attached)
                .map(|data| Python::attach(|py| PyBytesView::new(data).into_py_any(py).unwrap()))
        })
    }

    /// List available checkpoints
    #[pyo3(text_signature = "(self)")]
    fn list_checkpoints(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let inner = self.inner.clone();

        let infos = py
            .detach(|| {
                self.runtime.block_on(async {
                    let store = inner.lock().await;
                    store.list_checkpoints().await
                })
            })
            .map_err(|e| py_err(e.context("List failed")))?;

        Ok(infos
            .into_iter()
            .map(|info| PyCheckpointInfo { inner: info }.into_py_any(py).unwrap())
            .collect())
    }

    /// Delete a checkpoint
    #[pyo3(text_signature = "(self, step)")]
    fn delete_checkpoint(&self, py: Python<'_>, step: u64) -> PyResult<bool> {
        let inner = self.inner.clone();

        py.detach(|| {
            self.runtime.block_on(async {
                let store = inner.lock().await;
                store.delete_checkpoint(step).await
            })
        })
        .map_err(|e| py_err(e.context("Delete failed")))
    }

    /// Get a writer for distributed checkpointing
    #[pyo3(text_signature = "(self, world_size, rank)")]
    fn writer(&self, world_size: u32, rank: u32) -> PyResult<PyCheckpointWriter> {
        Ok(PyCheckpointWriter {
            store: self.inner.clone(),
            runtime: self.runtime.clone(),
            world_size,
            rank,
        })
    }

    /// Get a reader for checkpoint loading
    #[pyo3(text_signature = "(self)")]
    fn reader(&self) -> PyResult<PyCheckpointReader> {
        Ok(PyCheckpointReader {
            store: self.inner.clone(),
            runtime: self.runtime.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Shard metadata <-> Python dict conversions
//
// Every Python-facing shard-metadata dict goes through these helpers, so a field of
// `ShardMeta` cannot be kept by one wrapper and dropped by another (final review F2: the
// optional writer `checksum` was dropped at every hop). The dict keys are `rank`, `key`, `size`
// and, when present, `etag` and `checksum`; a missing key and a Python `None` both mean "absent".
// ---------------------------------------------------------------------------

/// Dict key of the optional `crc32c:<hex>` checksum of the shard's logical (uncompressed) bytes.
#[cfg(feature = "extension-module")]
const SHARD_CHECKSUM_KEY: &str = "checksum";

/// Put `rank`, `key`, `size` and the optional `etag` / `checksum` of `shard` into `dict`.
#[cfg(feature = "extension-module")]
fn set_shard_meta_items(
    dict: &Bound<'_, PyDict>,
    shard: &crate::checkpoint::ShardMeta,
) -> PyResult<()> {
    dict.set_item("rank", shard.rank)?;
    dict.set_item("key", &shard.key)?;
    dict.set_item("size", shard.size)?;
    if let Some(etag) = &shard.etag {
        dict.set_item("etag", etag)?;
    }
    if let Some(checksum) = &shard.checksum {
        dict.set_item(SHARD_CHECKSUM_KEY, checksum)?;
    }
    Ok(())
}

/// The optional checksum of a shard dict: absent, `None` and `""` are `None`; anything but a
/// string is a `TypeError`; a string that is not a supported `crc32c:<hex>` form is an error here,
/// so a bad value fails when the checkpoint is finalized, not later when it is restored.
#[cfg(feature = "extension-module")]
fn shard_checksum_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    match dict.get_item(SHARD_CHECKSUM_KEY)? {
        Some(value) if !value.is_none() => {
            let checksum = value.extract::<String>()?;
            if checksum.is_empty() {
                return Ok(None);
            }
            crate::checksum::parse_expected_crc32c(&checksum).map_err(py_err)?;
            Ok(Some(checksum))
        }
        _ => Ok(None),
    }
}

/// Rebuild a `ShardMeta` from a shard dict (the inverse of [`set_shard_meta_items`]).
#[cfg(feature = "extension-module")]
fn shard_meta_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<crate::checkpoint::ShardMeta> {
    let rank = dict
        .get_item("rank")?
        .ok_or_else(|| PyRuntimeError::new_err("Missing rank in shard meta"))?
        .extract::<u32>()?;
    let key = dict
        .get_item("key")?
        .ok_or_else(|| PyRuntimeError::new_err("Missing key in shard meta"))?
        .extract::<String>()?;
    let size = dict
        .get_item("size")?
        .ok_or_else(|| PyRuntimeError::new_err("Missing size in shard meta"))?
        .extract::<u64>()?;
    let etag = dict
        .get_item("etag")?
        .map(|e| e.extract::<String>())
        .transpose()?;
    Ok(crate::checkpoint::ShardMeta {
        rank,
        key,
        size,
        etag,
        checksum: shard_checksum_from_dict(dict)?,
    })
}

/// A manifest as a Python dict; every shard goes through [`set_shard_meta_items`].
#[cfg(feature = "extension-module")]
fn manifest_to_pydict<'py>(
    py: Python<'py>,
    manifest: &crate::checkpoint::Manifest,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("format_version", manifest.format_version)?;
    dict.set_item("framework", &manifest.framework)?;
    dict.set_item("global_step", manifest.global_step)?;
    dict.set_item("epoch", manifest.epoch)?;
    dict.set_item("wall_time", &manifest.wall_time)?;
    dict.set_item("world_size", manifest.world_size)?;
    dict.set_item("status", &manifest.status)?;

    let shards_list = PyList::empty(py);
    for shard in &manifest.shards {
        let shard_dict = PyDict::new(py);
        set_shard_meta_items(&shard_dict, shard)?;
        shards_list.append(shard_dict)?;
    }
    dict.set_item("shards", shards_list)?;
    Ok(dict)
}

/// Read a shard through the validating reader when it carries a checksum, and through the
/// plain reader when it does not (a manifest written before checksums were kept stays readable).
#[cfg(feature = "extension-module")]
async fn read_shard_checked(
    reader: &crate::checkpoint::Reader<'_>,
    key: &str,
    checksum: Option<&str>,
) -> anyhow::Result<Bytes> {
    match checksum {
        Some(_) => reader.read_shard_with_validation(key, checksum).await,
        None => reader.read_shard(key).await,
    }
}

/// Streaming checkpoint writer for zero-copy operations
#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCheckpointStream {
    store: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
    step: u64,
    epoch: u64,
    framework: String,
    world_size: u32,
    rank: u32,
    writer: Option<Box<dyn crate::object_store::ObjectWriter + Send>>,
    key: Option<String>,
    finalized: bool,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCheckpointStream {
    /// Write a chunk of data to the stream. `data` may be bytes, bytearray, memoryview, a numpy
    /// array or a BytesView; it is read in place and must not be modified until the call returns.
    #[pyo3(text_signature = "(self, data)")]
    fn write_chunk(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<usize> {
        if self.finalized {
            return Err(PyRuntimeError::new_err("Stream has been finalized"));
        }

        let store = self.store.clone();
        let world_size = self.world_size;
        let rank = self.rank;
        let step = self.step;
        let _epoch = self.epoch; // May be used in future for enhanced metadata
        let _framework = self.framework.clone(); // May be used in future for enhanced metadata
        let runtime = self.runtime.clone();
        let writer_slot = &mut self.writer;
        let key_slot = &mut self.key;

        // RULE #1: `data` (bytes, bytearray, memoryview, numpy, BytesView, ...) is read in
        // place: the export is held for this blocking call (`with_blocking_payload`) and the
        // writer reads it as a slice, so this layer makes no copy.
        //
        // The payload is converted BEFORE the lazy writer is created: an unsupported payload
        // raises its TypeError without creating (and truncating) a shard, so a later
        // `finalize()` still reports "No data written to stream".
        with_blocking_payload(py, data, |chunk| {
            // Lazy initialization of the writer (first valid chunk only).
            if writer_slot.is_none() {
                let (writer, key) = runtime
                    .block_on(async {
                        let store_guard = store.lock().await;
                        let ckpt_writer = store_guard.writer(world_size, rank);
                        let layout = crate::checkpoint::paths::KeyLayout::new(
                            "checkpoints".to_string(),
                            step,
                        );
                        ckpt_writer.get_shard_writer(&layout).await
                    })
                    .map_err(|e| py_err(e.context("Failed to create writer")))?;
                *writer_slot = Some(writer);
                *key_slot = Some(key);
            }

            match writer_slot.as_mut() {
                Some(writer) => {
                    runtime
                        .block_on(async { writer.write_chunk(&chunk).await })
                        .map_err(|e| py_err(e.context("Write failed")))?;
                    Ok(chunk.len())
                }
                None => Err(PyRuntimeError::new_err("Writer initialization failed")),
            }
        })
    }

    /// Get the number of bytes written so far
    #[pyo3(text_signature = "(self)")]
    fn bytes_written(&self) -> u64 {
        self.writer.as_ref().map_or(0, |w| w.bytes_written())
    }

    /// Finalize the stream and return shard metadata
    #[pyo3(text_signature = "(self)")]
    fn finalize(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.finalized {
            return Err(PyRuntimeError::new_err("Stream already finalized"));
        }

        if let (Some(writer), Some(key)) = (self.writer.take(), self.key.take()) {
            let store = self.store.clone();
            let world_size = self.world_size;
            let rank = self.rank;
            let step = self.step;
            let epoch = self.epoch;

            let shard_meta = py
                .detach(|| {
                    self.runtime.block_on(async {
                        // Finalize (upload) WITHOUT the store lock, so concurrent streams on
                        // one store are not serialised for the whole upload.
                        let checksum = writer.checksum();
                        writer.finalize().await?;

                        // Record the shard under the name it was stored as (a compressing
                        // store adds `.zst`: R10b).
                        let store_guard = store.lock().await;
                        let ckpt_writer = store_guard.writer(world_size, rank);
                        let layout = crate::checkpoint::paths::KeyLayout::new(
                            "checkpoints".to_string(),
                            step,
                        );
                        ckpt_writer
                            .finalize_shard_meta_with_checksum(&layout, key, checksum)
                            .await
                    })
                })
                .map_err(|e| py_err(e.context("Finalize failed")))?;

            self.finalized = true;

            // Return metadata as Python dict
            let dict = PyDict::new(py);
            dict.set_item("step", step)?;
            dict.set_item("epoch", epoch)?;
            set_shard_meta_items(&dict, &shard_meta)?;
            Ok(dict.into())
        } else {
            Err(PyRuntimeError::new_err("No data written to stream"))
        }
    }

    /// Cancel the stream and clean up
    #[pyo3(text_signature = "(self)")]
    fn cancel(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(writer) = self.writer.take() {
            py.detach(|| self.runtime.block_on(async { writer.cancel().await }))
                .map_err(|e| py_err(e.context("Cancel failed")))?;
        }

        self.finalized = true;
        self.key = None;
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCheckpointWriter {
    store: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
    world_size: u32,
    rank: u32,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCheckpointWriter {
    /// Save a distributed shard
    #[pyo3(text_signature = "(self, step, epoch, framework, data)")]
    fn save_distributed_shard(
        &self,
        py: Python<'_>,
        step: u64,
        epoch: u64,
        framework: String,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let store = self.store.clone();
        let world_size = self.world_size;
        let rank = self.rank;

        // `data` is read in place (any buffer-protocol object), not copied: see
        // `with_blocking_payload`.
        with_blocking_payload(py, data, |data| {
            self.runtime
                .block_on(async {
                    let store_guard = store.lock().await;
                    let writer = store_guard.writer(world_size, rank);
                    writer
                        .save_distributed_shard(step, epoch, &framework, data)
                        .await
                })
                .map_err(|e| py_err(e.context("Save shard failed")))
        })
        .and_then(|(layout, shard_meta)| {
            let dict = PyDict::new(py);
            dict.set_item("step", layout.step)?;
            dict.set_item("timestamp", &layout.ts)?;
            set_shard_meta_items(&dict, &shard_meta)?;
            Ok(dict.into())
        })
    }

    /// Get a streaming writer for a distributed shard (zero-copy version)
    /// Returns a PyCheckpointStream that can be written to incrementally
    #[pyo3(text_signature = "(self, step, epoch, framework)")]
    fn get_distributed_shard_stream(
        &self,
        _py: Python<'_>, // Python context, may be used for future enhancements
        step: u64,
        epoch: u64,
        framework: String,
    ) -> PyResult<PyCheckpointStream> {
        let store = self.store.clone();
        let world_size = self.world_size;
        let rank = self.rank;

        // Note: We'll need to handle the async creation in a different way
        // For now, return a PyCheckpointStream that handles lazy initialization
        Ok(PyCheckpointStream {
            store,
            runtime: self.runtime.clone(),
            step,
            epoch,
            framework,
            world_size,
            rank,
            writer: None,
            key: None,
            finalized: false,
        })
    }

    /// Finalize distributed checkpoint (rank 0 only)
    #[pyo3(
        signature = (step, epoch, framework, shard_metas, user_meta=None),
        text_signature = "(self, step, epoch, framework, shard_metas, user_meta=None)"
    )]
    fn finalize_distributed_checkpoint(
        &self,
        py: Python<'_>,
        step: u64,
        epoch: u64,
        framework: String,
        shard_metas: Vec<Py<PyAny>>,
        user_meta: Option<Py<PyAny>>,
    ) -> PyResult<String> {
        let user_meta = user_meta.map(|_obj| {
            // For now, just use null for user metadata
            // TODO: Implement proper Python->JSON conversion
            serde_json::Value::Null
        });

        // Convert shard metas from Python objects
        let mut shards = Vec::new();
        for shard_obj in shard_metas {
            let dict = shard_obj
                .cast_bound::<PyDict>(py)
                .map_err(|_| PyRuntimeError::new_err("Shard meta must be a dict"))?;

            shards.push(shard_meta_from_dict(dict)?);
        }

        let store = self.store.clone();
        let world_size = self.world_size;
        let rank = self.rank;

        py.detach(|| {
            self.runtime.block_on(async {
                use crate::checkpoint::KeyLayout;
                let layout = KeyLayout::new(store.lock().await.uri.clone(), step);
                let store_guard = store.lock().await;
                let writer = store_guard.writer(world_size, rank);
                writer
                    .finalize_distributed_checkpoint(&layout, &framework, epoch, shards, user_meta)
                    .await
            })
        })
        .map_err(|e| py_err(e.context("Finalize failed")))
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCheckpointReader {
    store: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCheckpointReader {
    /// Load latest manifest
    #[pyo3(text_signature = "(self)")]
    fn load_latest_manifest(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let store = self.store.clone();

        py.detach(|| {
            self.runtime.block_on(async {
                let store_guard = store.lock().await;
                let reader = store_guard.reader();
                reader.load_latest_manifest().await
            })
        })
        .map_err(|e| py_err(e.context("Load manifest failed")))
        .and_then(|opt_manifest| {
            opt_manifest
                .map(|manifest| manifest_to_pydict(py, &manifest).map(|d| d.into_any().unbind()))
                .transpose()
        })
    }

    /// Read shard by rank
    #[pyo3(text_signature = "(self, manifest, rank)")]
    fn read_shard_by_rank(
        &self,
        py: Python<'_>,
        manifest: Py<PyAny>,
        rank: u32,
    ) -> PyResult<Py<PyAny>> {
        // Parse manifest from Python dict
        let manifest_dict = manifest
            .cast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Manifest must be a dict"))?;

        let shards_item = manifest_dict
            .get_item("shards")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing shards in manifest"))?;
        let shards_list = shards_item.cast::<PyList>()?;

        let mut target = None;
        for shard_obj in shards_list.iter() {
            let shard_dict = shard_obj.cast::<PyDict>()?;
            let shard_rank = shard_dict
                .get_item("rank")?
                .ok_or_else(|| PyRuntimeError::new_err("Missing rank in shard"))?
                .extract::<u32>()?;
            if shard_rank == rank {
                let key = shard_dict
                    .get_item("key")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing key in shard"))?
                    .extract::<String>()?;
                target = Some((key, shard_checksum_from_dict(shard_dict)?));
                break;
            }
        }

        let (key, checksum) =
            target.ok_or_else(|| PyRuntimeError::new_err(format!("Rank {} not found", rank)))?;

        let store = self.store.clone();

        let data = py
            .detach(|| {
                self.runtime.block_on(async {
                    let store_guard = store.lock().await;
                    let reader = store_guard.reader();
                    read_shard_checked(&reader, &key, checksum.as_deref()).await
                })
            })
            .map_err(|e| py_err(e.context("Read shard failed")))?;

        // Return zero-copy BytesView as PyObject
        PyBytesView::new(data).into_py_any(py)
    }
}

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCheckpointInfo {
    inner: CheckpointInfo,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyCheckpointInfo {
    #[getter]
    fn step(&self) -> u64 {
        self.inner.step
    }

    #[getter]
    fn epoch(&self) -> u64 {
        self.inner.epoch
    }

    #[getter]
    fn timestamp(&self) -> &str {
        &self.inner.timestamp
    }

    #[getter]
    fn status(&self) -> &str {
        &self.inner.status
    }

    #[getter]
    fn framework(&self) -> &str {
        &self.inner.framework
    }

    #[getter]
    fn world_size(&self) -> u32 {
        self.inner.world_size
    }

    #[getter]
    fn total_size(&self) -> u64 {
        self.inner.total_size
    }

    #[getter]
    fn size_mb(&self) -> f64 {
        self.inner.size_mb()
    }

    #[getter]
    fn size_gb(&self) -> f64 {
        self.inner.size_gb()
    }

    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    #[getter]
    fn is_single_rank(&self) -> bool {
        self.inner.is_single_rank()
    }
}

// Convenience functions
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(
    signature = (uri, step, epoch, framework, data, user_meta=None),
    text_signature = "(uri, step, epoch, framework, data, user_meta=None)"
)]
fn save_checkpoint(
    py: Python<'_>,
    uri: String,
    step: u64,
    epoch: u64,
    framework: String,
    data: &Bound<'_, PyAny>,
    user_meta: Option<Py<PyAny>>,
) -> PyResult<String> {
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    store.save(py, step, epoch, framework, data, user_meta)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(text_signature = "(uri)")]
fn load_checkpoint(py: Python<'_>, uri: String) -> PyResult<Option<Py<PyAny>>> {
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    store.load_latest(py)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(text_signature = "(uri, step, epoch, framework, data, world_size, rank)")]
#[allow(clippy::too_many_arguments)]
fn save_distributed_shard(
    py: Python<'_>,
    uri: String,
    step: u64,
    epoch: u64,
    framework: String,
    data: &Bound<'_, PyAny>,
    world_size: u32,
    rank: u32,
) -> PyResult<Py<PyAny>> {
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    let writer = store.writer(world_size, rank)?;
    writer.save_distributed_shard(py, step, epoch, framework, data)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(
    signature = (uri, step, epoch, framework, shard_metas, world_size, rank, user_meta=None),
    text_signature = "(uri, step, epoch, framework, shard_metas, world_size, rank, user_meta=None)"
)]
#[allow(clippy::too_many_arguments)]
fn finalize_distributed_checkpoint(
    py: Python<'_>,
    uri: String,
    step: u64,
    epoch: u64,
    framework: String,
    shard_metas: Vec<Py<PyAny>>,
    world_size: u32,
    rank: u32,
    user_meta: Option<Py<PyAny>>,
) -> PyResult<String> {
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    let writer = store.writer(world_size, rank)?;
    writer.finalize_distributed_checkpoint(py, step, epoch, framework, shard_metas, user_meta)
}

// ---------------------------------------------------------------------------
// Phase 3 Priority 4: Rich Python-Rust Data Exchange Enhancements
// ---------------------------------------------------------------------------

/// Fetch one checkpoint shard (the shared body of [`load_checkpoint_with_validation`] and
/// [`load_checkpoint_with_validation_view`]); the GIL is released while it runs.
fn load_checkpoint_shard(py: Python<'_>, uri: &str, validate_integrity: bool) -> PyResult<Bytes> {
    let store = store_for_uri_with_helper_rows(uri).map_err(py_err)?;
    let uri_owned = uri.to_owned();

    py.detach(|| {
        // Submit to global runtime (io_uring pattern — never calls block_on)
        run_on_global_rt(async move {
            if validate_integrity {
                // Verify against the checksum the checkpoint manifest recorded (issue #287).
                let (base, shard_rel_key) = crate::checkpoint::reader::split_shard_uri(&uri_owned)?;
                crate::checkpoint::Reader::new(&*store, base.to_string())
                    .read_shard_against_manifest(shard_rel_key)
                    .await
            } else {
                // `get` returns refcounted `Bytes`: handed on as they are.
                store.get(&uri_owned).await
            }
        })
    })
    .map_err(py_err)
}

/// Read one checkpoint shard by URI.
///
/// With `validate_integrity=True` (default) the bytes are verified against the checksum
/// the checkpoint's manifest recorded when the shard was written: a mismatch, a shard no
/// manifest lists, or a manifest that recorded no checksum for it raises an error (the
/// check is never skipped silently). A `.zst` shard is decoded first and the decoded bytes
/// are returned. With `validate_integrity=False` the object is read as stored, unverified.
///
/// Returns Python `bytes`: this is a compatibility API, and building a `bytes` from the
/// fetched buffer is ONE copy (documented; RULE #1 exception "an API that explicitly returns
/// `bytes`"). Use [`load_checkpoint_with_validation_view`] for the same result with zero
/// copies.
#[pyfunction]
#[pyo3(signature = (uri, validate_integrity=true))]
pub fn load_checkpoint_with_validation(
    py: Python<'_>,
    uri: &str,
    validate_integrity: bool,
) -> PyResult<Py<PyAny>> {
    let data = load_checkpoint_shard(py, uri, validate_integrity)?;
    // Compatibility API returns `bytes`: this single copy is unavoidable (a Python `bytes` owns
    // its memory). It is built straight from the fetched `Bytes`, with no intermediate `Vec`.
    Ok(PyBytes::new(py, &data).into())
}

/// Like `load_checkpoint_with_validation`, but returns a `BytesView` over the fetched buffer:
/// ZERO copies. Same arguments, same validation, same errors.
///
/// `memoryview(view)` / `np.frombuffer(view, dtype=...)` read the checkpoint in place; the
/// buffer lives as long as any view or memoryview of it does.
#[pyfunction]
#[pyo3(signature = (uri, validate_integrity=true))]
pub fn load_checkpoint_with_validation_view(
    py: Python<'_>,
    uri: &str,
    validate_integrity: bool,
) -> PyResult<Py<PyAny>> {
    let data = load_checkpoint_shard(py, uri, validate_integrity)?;
    PyBytesView::new(data).into_py_any(py)
}

/// Save NumPy array with compression and validation
/// Currently disabled due to threading issues - TODO: Fix in future version
// #[pyfunction]
// #[pyo3(signature = (uri, array, compress=true, validate=true))]
#[allow(dead_code)]
pub fn save_numpy_array(
    _py: Python<'_>,
    _uri: &str,
    _array: Bound<'_, PyAny>,
    _compress: bool,
    _validate: bool,
) -> PyResult<String> {
    Err(PyRuntimeError::new_err(
        "save_numpy_array is temporarily disabled",
    ))
}

/// Load NumPy array with validation
/// Currently disabled due to threading issues - TODO: Fix in future version
// #[pyfunction]
// #[pyo3(signature = (uri, shape, dtype="f32", validate_checksum=None))]
#[allow(dead_code)]
pub fn load_numpy_array(
    _py: Python<'_>,
    _uri: &str,
    _shape: Vec<usize>,
    _dtype: &str,
    _validate_checksum: Option<&str>,
) -> PyResult<Py<PyAny>> {
    Err(PyRuntimeError::new_err(
        "load_numpy_array is temporarily disabled",
    ))
}

/// Create a dataset from any supported URI scheme
#[pyfunction]
#[pyo3(signature = (uri, opts=None))]
pub fn create_dataset(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<PyDataset> {
    PyDataset::new(uri, opts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parquet streaming loader — one dataset for ALL files, items in arrival order
// ─────────────────────────────────────────────────────────────────────────────

/// One row group returned from `PyParquetStreamLoader`.
///
/// Python attributes:
///   `item.file_uri`   → full S3/file URI of the containing Parquet file
///   `item.rg_idx`     → row-group index within that file
///   `item.byte_count` → compressed byte count (length of the raw data)
///   `memoryview(item)` / `np.frombuffer(item)` → the raw compressed column-chunk bytes read in
///                       place (zero copy)
///   `item.data`       → the same bytes as a read-only `BytesView` (zero copy)
///   `bytes(item)`     → raw compressed column-chunk bytes as Python `bytes` (a copy, kept for
///                       compatibility)
///   `len(item)`       → same as `byte_count`
#[pyclass]
pub struct PyParquetItem {
    #[pyo3(get)]
    pub file_uri: String,
    #[pyo3(get)]
    pub rg_idx: usize,
    data: Bytes,
    /// `data.len()` cached in the heap object for the buffer protocol's `shape` pointer.
    shape_len: isize,
}

impl PyParquetItem {
    fn new(file_uri: String, rg_idx: usize, data: Bytes) -> Self {
        let shape_len = data.len() as isize;
        Self {
            file_uri,
            rg_idx,
            data,
            shape_len,
        }
    }
}

#[pymethods]
impl PyParquetItem {
    #[getter]
    fn byte_count(&self) -> usize {
        self.data.len()
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// The row group as a read-only `BytesView`: zero copies (it shares this item's buffer).
    #[getter]
    fn data(&self) -> PyBytesView {
        PyBytesView::new(self.data.clone())
    }

    /// Return the raw compressed bytes as Python `bytes`. This is the COPY (kept for
    /// compatibility): use `memoryview(item)`, `np.frombuffer(item)` or `item.data` to read the
    /// bytes in place.
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    /// Read-only buffer protocol: `memoryview(item)` reads the row group in place, no copy.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        fill_read_only_bytes_buffer(view, flags, slf.as_ptr(), &slf.data, &slf.shape_len)
    }

    /// Nothing to release: CPython drops the reference taken in `__getbuffer__`.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    fn __repr__(&self) -> String {
        format!(
            "PyParquetItem(file_uri={:?}, rg_idx={}, byte_count={})",
            self.file_uri,
            self.rg_idx,
            self.data.len()
        )
    }
}

/// Synchronous iterator returned by `PyParquetStreamLoader.__iter__()`.
///
/// Items arrive in **network completion order**, not file/RG index order.
/// The GIL is released while waiting for the next item.
///
/// Each item is a `PyParquetItem` carrying `file_uri`, `rg_idx`, and the raw
/// bytes — enough for the Python consumer to cache by `(file_uri, rg_idx)`.
#[pyclass]
pub struct ParquetStreamIter {
    rx: parking_lot::Mutex<tokio::sync::mpsc::Receiver<Result<(usize, Bytes), DatasetError>>>,
    /// global_rg_idx → (file_uri_idx, rg_idx_in_file)
    rg_info: Arc<Vec<(usize, usize)>>,
    file_uris: Arc<Vec<String>>,
}

#[pymethods]
impl ParquetStreamIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Release the GIL while waiting — other Python threads continue freely.
        let result = py.detach(|| blocking_recv_locked(&self.rx));

        match result {
            Some(Ok((global_idx, bytes))) => {
                let (file_uri_idx, rg_idx) = self.rg_info[global_idx];
                let file_uri = self.file_uris[file_uri_idx].clone();
                let item = PyParquetItem::new(file_uri, rg_idx, bytes);
                Py::new(py, item)?.into_py_any(py)
            }
            Some(Err(e)) => Err(py_err(e)),
            None => Err(PyStopIteration::new_err("parquet stream exhausted")),
        }
    }
}

/// Parquet prefetch stream loader.
///
/// Creates **one** `ParquetRowGroupDataset` covering ALL files under `prefix`.
/// `__iter__()` spawns a single Tokio producer task that keeps `concurrency`
/// row-group range-GETs in flight simultaneously via `buffer_unordered`.
///
/// Items are delivered to Python in network completion order through a bounded
/// `mpsc::channel(concurrency)`.  The bounded channel is the sole backpressure
/// mechanism — no `Semaphore`, no `JoinSet`, no manual worker counting.
///
/// Memory at any instant: ≤ `concurrency` row groups held in the channel
/// (each ~1–8 MiB depending on dataset).
///
/// This is the "all files, all row groups, prefetch N ahead" design:
///   producer: `buffer_unordered(concurrency)` over global_rg_idx 0..total
///   consumer: Python pops items, caches by (file_uri, rg_idx), serves on demand
#[pyclass]
pub struct PyParquetStreamLoader {
    dataset: Arc<crate::data_loader::ParquetRowGroupDataset>,
    /// How many row-group GETs to keep in flight simultaneously.
    /// Also the bounded channel capacity (= always 1 full "batch" ahead minimum).
    concurrency: usize,
}

#[pymethods]
impl PyParquetStreamLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<ParquetStreamIter>> {
        let concurrency = slf.concurrency;
        let dataset = Arc::clone(&slf.dataset);
        let total = dataset.len().unwrap_or(0);

        let rg_info = Arc::new(dataset.rg_info_vec());
        let file_uris = dataset.file_uris_arc();

        // Bounded channel: capacity = concurrency.
        // Producer blocks here (not in a Semaphore) when Python is slow.
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<(usize, Bytes), DatasetError>>(concurrency);

        tracing::info!(
            total_items = total,
            concurrency,
            "[s3dlio] ParquetStreamLoader.__iter__: {} row-groups, {} concurrent GETs",
            total,
            concurrency
        );

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            // Task-level parallelism (issue #148 site 1.1 — Parquet stream
            // variant). Same DropCancel + spawn-each + select! pattern as
            // the other two sites in this file; item type here is
            // (row_group_idx, bytes) instead of Bytes / (uri, bytes).
            let cancel = CancellationToken::new();
            let _drop_cancel = DropCancel(cancel.clone());

            use std::time::Instant;

            let t_start = Instant::now();
            let mut count: usize = 0;
            let mut total_bytes: u64 = 0;
            let mut t_last_log = Instant::now();

            let spawn_fetch = |idx: usize| -> JoinHandle<Result<(usize, Bytes), DatasetError>> {
                let ds = Arc::clone(&dataset);
                let token = cancel.clone();
                tokio::spawn(async move {
                    tokio::select! {
                        _ = token.cancelled() => {
                            Err(DatasetError::from("fetch cancelled"))
                        }
                        r = ds.get(idx) => r.map(|b| (idx, b)),
                    }
                })
            };

            let mut pending: FuturesUnordered<JoinHandle<Result<(usize, Bytes), DatasetError>>> =
                FuturesUnordered::new();
            let mut next_idx = 0usize;

            while next_idx < total && pending.len() < concurrency {
                pending.push(spawn_fetch(next_idx));
                next_idx += 1;
            }

            while let Some(join_res) = pending.next().await {
                let result: Result<(usize, Bytes), DatasetError> = match join_res {
                    Ok(r) => r,
                    Err(join_err) if join_err.is_panic() => Err(DatasetError::from(format!(
                        "fetch task panicked: {}",
                        join_err
                    ))),
                    Err(_) => continue,
                };

                if let Ok((_, ref b)) = result {
                    total_bytes += b.len() as u64;
                    count += 1;
                    if count.is_multiple_of(50) || t_last_log.elapsed().as_secs() >= 5 {
                        let elapsed = t_start.elapsed().as_secs_f64();
                        tracing::info!(
                            count,
                            total_items = total,
                            "[s3dlio] parquet stream: {}/{} RGs | {:.0} MB/s",
                            count,
                            total,
                            total_bytes as f64 / elapsed / 1024.0 / 1024.0
                        );
                        t_last_log = Instant::now();
                    }
                }
                // Bounded send — blocks when channel full (Python not consuming).
                if tx.send(result).await.is_err() {
                    tracing::debug!("[s3dlio] parquet stream: consumer dropped iterator");
                    break;
                }
                if next_idx < total {
                    pending.push(spawn_fetch(next_idx));
                    next_idx += 1;
                }
            }

            let elapsed = t_start.elapsed().as_secs_f64();
            tracing::info!(
                "[s3dlio] parquet stream done: {}/{} RGs | {:.0} MB/s",
                count,
                total,
                if elapsed > 0.0 {
                    total_bytes as f64 / elapsed / 1024.0 / 1024.0
                } else {
                    0.0
                }
            );
        });

        Py::new(
            slf.py(),
            ParquetStreamIter {
                rx: parking_lot::Mutex::new(rx),
                rg_info,
                file_uris,
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Per-row-group fetch + optional Rust Arrow decode (index-based hot path)
// ---------------------------------------------------------------------------

/// Generate a complete Parquet file in memory and return it as a `PyBytesView`.
///
/// All columns are Float32. Data is generated using `dgen_data::RollingPool`
/// (Xoshiro256++ streaming) — one pool instantiated per call, reused across
/// every column of every row group without re-seeding.
///
/// ## Arguments
/// * `num_cols`       — number of Float32 columns
/// * `rows_per_rg`    — rows per row group
/// * `num_row_groups` — number of row groups
///
/// ## Example
/// ```python
/// import s3dlio
/// bv = s3dlio.generate_parquet_bytes(200, 8192, 123)
/// with open("/mnt/test/train/img_00_of_64.parquet", "wb") as f:
///     f.write(memoryview(bv))
/// ```
#[pyfunction]
#[pyo3(signature = (num_cols, rows_per_rg, num_row_groups))]
pub fn generate_parquet_bytes(
    py: Python<'_>,
    num_cols: usize,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> PyResult<crate::python_api::python_core_api::PyBytesView> {
    use crate::data_formats::parquet_gen;
    use crate::python_api::python_core_api::PyBytesView;

    let bytes = py
        .detach(|| parquet_gen::generate_parquet_bytes(num_cols, rows_per_rg, num_row_groups))
        .map_err(py_err)?;

    Ok(PyBytesView::new(bytes))
}

/// Generate a complete Parquet file and write it directly to any URI.
///
/// Everything — data generation, Parquet serialization, and the store write —
/// happens inside a single `py.detach()` block.  The GIL is released exactly
/// once for the entire operation; no data crosses the Python/Rust boundary
/// mid-file.
///
/// Supports `file://`, `direct://`, `s3://`, `az://`, `gs://`.
///
/// ## Arguments
/// * `uri`            — destination URI (any supported scheme)
/// * `num_cols`       — number of Float32 columns
/// * `rows_per_rg`    — rows per row group
/// * `num_row_groups` — number of row groups
///
/// ## Example
/// ```python
/// import s3dlio
/// s3dlio.generate_and_write_parquet(
///     "file:///mnt/test/train/img_00_of_64.parquet", 200, 8192, 123)
/// ```
#[pyfunction]
#[pyo3(signature = (uri, num_cols, rows_per_rg, num_row_groups))]
pub fn generate_and_write_parquet(
    py: Python<'_>,
    uri: &str,
    num_cols: usize,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> PyResult<()> {
    use crate::data_formats::parquet_gen;
    use crate::s3_client::run_on_global_rt;

    let uri_owned = uri.to_owned();
    py.detach(|| {
        run_on_global_rt(async move {
            parquet_gen::generate_and_write_parquet(
                &uri_owned,
                num_cols,
                rows_per_rg,
                num_row_groups,
            )
            .await
        })
    })
    .map_err(py_err)
}

/// Generate a Parquet file in memory with a named, schema-flexible column layout.
///
/// `columns` is a list of `(name, num_float32_per_row)` tuples:
/// - `num_float32_per_row == 1` → plain `Float32` scalar column
/// - `num_float32_per_row > 1`  → `FixedSizeList<Float32>[N]` embedding column
///
/// All data is generated in Rust using the Xoshiro256++ RollingPool — zero
/// Python involvement, zero numpy.
///
/// ## Example (Flux schema)
/// ```python
/// import s3dlio
/// bv = s3dlio.generate_parquet_bytes_schema(
///     columns=[("t5_encodings", 524328), ("clip_encodings", 409),
///              ("mean", 8232), ("logvar", 8232), ("timestamp", 7)],
///     rows_per_rg=48, num_row_groups=6)
/// with open("/mnt/test/train.parquet", "wb") as f:
///     f.write(memoryview(bv))
/// ```
#[pyfunction]
#[pyo3(signature = (columns, rows_per_rg, num_row_groups))]
pub fn generate_parquet_bytes_schema(
    py: Python<'_>,
    columns: Vec<(String, usize)>,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> PyResult<crate::python_api::python_core_api::PyBytesView> {
    use crate::data_formats::parquet_gen;
    use crate::python_api::python_core_api::PyBytesView;

    let bytes = py
        .detach(|| {
            parquet_gen::generate_parquet_bytes_schema(&columns, rows_per_rg, num_row_groups)
        })
        .map_err(py_err)?;

    Ok(PyBytesView::new(bytes))
}

/// Generate a named-schema Parquet file and write it to any URI in one call.
///
/// `columns` is a list of `(name, num_float32_per_row)` tuples (see
/// [`generate_parquet_bytes_schema`] for schema rules).  The entire pipeline —
/// data generation, Parquet serialization, and store write — runs off the GIL.
///
/// Supports `file://`, `direct://`, `s3://`, `az://`, `gs://`.
///
/// ## Example (Flux schema, local filesystem)
/// ```python
/// import s3dlio
/// s3dlio.generate_and_write_parquet_schema(
///     "file:///mnt/test/data/flux/train/train_0000.parquet",
///     columns=[("t5_encodings", 524328), ("clip_encodings", 409),
///              ("mean", 8232), ("logvar", 8232), ("timestamp", 7)],
///     rows_per_rg=48, num_row_groups=6)
/// ```
#[pyfunction]
#[pyo3(signature = (uri, columns, rows_per_rg, num_row_groups))]
pub fn generate_and_write_parquet_schema(
    py: Python<'_>,
    uri: &str,
    columns: Vec<(String, usize)>,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> PyResult<()> {
    use crate::data_formats::parquet_gen;
    use crate::s3_client::run_on_global_rt;

    let uri_owned = uri.to_owned();
    py.detach(|| {
        run_on_global_rt(async move {
            parquet_gen::generate_and_write_parquet_schema(
                &uri_owned,
                columns,
                rows_per_rg,
                num_row_groups,
            )
            .await
        })
    })
    .map_err(py_err)
}

/// Generate a Parquet file with a named schema and stream it to an S3 URI,
/// pipelining row-group generation with multipart upload.
///
/// Unlike ``generate_and_write_parquet_schema`` (which builds the entire file
/// in memory before uploading), this function opens the multipart upload
/// immediately and sends each completed row group as a part while the next
/// row group is being generated.  Peak memory is ~2× one row group rather
/// than the full file size.
///
/// ## Parameters
/// - ``uri``: S3 URI, e.g. ``"s3://bucket/path/file.parquet"``
/// - ``columns``: list of ``(name, num_float32_per_row)`` tuples
/// - ``rows_per_rg``: rows per row group
/// - ``num_row_groups``: number of row groups
///
/// ## Example
/// ```python
/// import s3dlio
/// s3dlio.generate_and_write_parquet_schema_streaming(
///     "s3://mlp-flux/data/flux/train/train_0000.parquet",
///     columns=[("t5_encodings", 524328), ("clip_encodings", 409),
///              ("mean", 8232), ("logvar", 8232), ("timestamp", 7)],
///     rows_per_rg=48, num_row_groups=6)
/// ```
#[pyfunction]
#[pyo3(signature = (uri, columns, rows_per_rg, num_row_groups))]
pub fn generate_and_write_parquet_schema_streaming(
    py: Python<'_>,
    uri: &str,
    columns: Vec<(String, usize)>,
    rows_per_rg: usize,
    num_row_groups: usize,
) -> PyResult<()> {
    use crate::data_formats::parquet_gen;
    use crate::s3_client::run_on_global_rt;

    let uri_owned = uri.to_owned();
    py.detach(|| {
        run_on_global_rt(async move {
            parquet_gen::generate_and_write_parquet_schema_streaming(
                &uri_owned,
                columns,
                rows_per_rg,
                num_row_groups,
            )
            .await
        })
    })
    .map_err(py_err)
}

/// Fetch a single Parquet row group and optionally decode it to Arrow IPC.
///
/// This is the per-sample hot-path function for index-based access patterns
/// (e.g. random-access training, PyTorch `DataLoader` with shuffle).  It is
/// the Arrow-decode counterpart of `s3dlio.get_range()`:
///
/// * `s3dlio.get_range(uri, offset, length)` — raw bytes, no decode
/// * `s3dlio.parquet_get_rg(uri, rg_idx)` — single RG, optional Rust Arrow decode
///
/// ## Usage with `PyParquetIndex`
///
/// ```python
/// import s3dlio, pyarrow as pa
///
/// idx = s3dlio.PyParquetIndex()
/// idx.ensure_indexed(uris, epoch=1)
///
/// # Per-sample lookup → fetch → decode in one call
/// rg_idx, offset, length = idx.rg_lookup(uri, sample_row)
/// bv = s3dlio.parquet_get_rg(uri, rg_idx)                    # Rust Arrow IPC
/// batch = pa.ipc.open_stream(pa.py_buffer(bytes(bv))).read_next_batch()
/// tensor = torch.from_numpy(batch.column(0).to_pydict())
/// ```
///
/// ## Arguments
/// * `uri`         — full URI of the `.parquet` file (any supported scheme).
/// * `rg_idx`      — 0-based row-group index within the file.
/// * `col_indices` — column indices to decode; `None` = all columns.
/// * `footer_cap`  — footer fetch size in bytes (default 4 MiB).
/// * `decode`      — `"arrow"` (default) returns Arrow IPC bytes;
///                   `"raw"` returns raw compressed Parquet column-chunk bytes.
///
/// ## Thread safety
///
/// The footer is fetched once and cached in the process-global
/// `parquet_file_cache` — subsequent calls for the same file are zero-cost at
/// the metadata level.  The function releases the GIL while waiting for I/O.
#[pyfunction]
#[pyo3(signature = (uri, rg_idx, col_indices=None, footer_cap=4194304, decode="arrow"))]
pub fn parquet_get_rg(
    py: Python<'_>,
    uri: &str,
    rg_idx: usize,
    col_indices: Option<Vec<usize>>,
    footer_cap: usize,
    decode: &str,
) -> PyResult<crate::python_api::python_core_api::PyBytesView> {
    use crate::data_loader::{parquet_file_cache, parquet_rg};
    use crate::object_store::store_for_uri;
    use crate::python_api::python_core_api::PyBytesView;
    use crate::s3_client::run_on_global_rt;

    let uri_owned = uri.to_owned();
    let do_arrow = decode.eq_ignore_ascii_case("arrow");

    let bytes = py
        .detach(|| {
            run_on_global_rt(async move {
                // ── 1. Footer metadata (cache hit on epoch 2+) ───────────────────
                let cached =
                    parquet_file_cache::get_or_fetch(&uri_owned, footer_cap as u64).await?;
                let meta = Arc::clone(&cached.parquet_meta);

                if rg_idx >= meta.num_row_groups() {
                    anyhow::bail!(
                        "rg_idx {} out of range for '{}' ({} row groups)",
                        rg_idx,
                        uri_owned,
                        meta.num_row_groups()
                    );
                }

                if do_arrow {
                    // ── 2a. Arrow IPC decode path ─────────────────────────────────
                    use arrow_array::RecordBatch;
                    use arrow_ipc::writer::StreamWriter;
                    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
                    use parquet::arrow::ProjectionMask;

                    let reader = parquet_rg::S3AsyncFileReader::new(uri_owned, meta);
                    let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
                        .await
                        .map_err(|e| anyhow::anyhow!("parquet reader init: {}", e))?;

                    builder = builder.with_row_groups(vec![rg_idx]);
                    // Return all rows in the row group in a single RecordBatch.
                    builder = builder.with_batch_size(usize::MAX);

                    if let Some(ref cols) = col_indices {
                        let mask =
                            ProjectionMask::leaves(builder.parquet_schema(), cols.iter().copied());
                        builder = builder.with_projection(mask);
                    }

                    let mut stream = builder
                        .build()
                        .map_err(|e| anyhow::anyhow!("parquet stream build: {}", e))?;

                    use futures_util::StreamExt as _;
                    let batch: RecordBatch = stream
                        .next()
                        .await
                        .ok_or_else(|| anyhow::anyhow!("parquet stream returned no batch"))?
                        .map_err(|e| anyhow::anyhow!("parquet stream read: {}", e))?;

                    let mut buf: Vec<u8> = Vec::with_capacity(batch.get_array_memory_size() + 512);
                    {
                        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
                            .map_err(|e| anyhow::anyhow!("ipc writer init: {}", e))?;
                        writer
                            .write(&batch)
                            .map_err(|e| anyhow::anyhow!("ipc write: {}", e))?;
                        writer
                            .finish()
                            .map_err(|e| anyhow::anyhow!("ipc finish: {}", e))?;
                    }
                    return Ok(bytes::Bytes::from(buf));
                }

                // ── 2b. Raw bytes path ────────────────────────────────────────────
                let rg_meta = meta.row_group(rg_idx);
                let (offset, length) = parquet_rg::rg_byte_extent(rg_meta, col_indices.as_deref())
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let store = store_for_uri(&uri_owned)?;
                store.get_range(&uri_owned, offset, Some(length)).await
            })
        })
        .map_err(py_err)?;

    Ok(PyBytesView::new(bytes))
}

/// Create an async data loader from any supported URI scheme (convenience function)
///
/// Extra `opts` keys handled by this function (in addition to standard [`LoaderOptions`] keys):
///
/// | Key | Type | Description |
/// |-----|------|-------------|
/// | `"format"` | `str` | `"parquet"` — enable row-group mode (requires `parquet` feature) |
/// | `"columns"` | `list[int]` | Column indices to include per row group (`None` = all) |
/// | `"footer_cap"` | `int` | Bytes to fetch from file tail for footer parsing (default 4 MiB) |
#[pyfunction]
#[pyo3(signature = (uri, opts=None))]
pub fn create_async_loader(
    py: Python<'_>,
    uri: &str,
    opts: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // Check for parquet format before falling through to the generic path.
    if let Some(ref d) = opts {
        let format_str = d
            .get_item("format")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok());

        if format_str.as_deref() == Some("parquet") {
            use crate::data_loader::ParquetDecodeMode;
            use crate::data_loader::ParquetRowGroupDataset;
            use crate::data_loader::DEFAULT_FOOTER_CAP;

            // Extract column indices: list[int] or None
            let col_indices: Option<Vec<usize>> = d
                .get_item("columns")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<Vec<usize>>().ok());

            // Extract footer_cap (bytes from file tail for footer parsing)
            let footer_cap: usize = d
                .get_item("footer_cap")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(DEFAULT_FOOTER_CAP);

            // Decode mode: "raw" (default) or "arrow" (Rust-side Arrow IPC decode).
            // "none" is treated identically to "raw" at the Rust layer; the Python
            // caller (dlio_benchmark) is responsible for not invoking PyArrow on the
            // raw bytes in that case.
            let decode_mode = {
                let s = d
                    .get_item("decode")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok());
                match s.as_deref() {
                    Some("arrow") => ParquetDecodeMode::ArrowIpc,
                    _ => ParquetDecodeMode::Raw, // "raw", "none", or not set
                }
            };

            let pq_dataset =
                ParquetRowGroupDataset::new(uri, col_indices.as_deref(), footer_cap, decode_mode)
                    .map_err(py_err)?;

            // max_threads: configure the pyo3 Tokio runtime thread count
            // BEFORE the first spawn.  Only takes effect if the runtime has
            // not yet been initialized (OnceLock — first call wins).
            if let Some(n) = d
                .get_item("max_threads")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
            {
                configure_tokio_threads(n);
            }

            // Concurrency: how many row-group GETs to keep in flight.
            // "prefetch" key sets this; default 32 (matches old Python 32-thread pool).
            let concurrency: usize = d
                .get_item("prefetch")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(32)
                .max(1);

            return Py::new(
                py,
                PyParquetStreamLoader {
                    dataset: Arc::new(pq_dataset),
                    concurrency,
                },
            )?
            .into_py_any(py);
        }
    }

    // Generic path: S3BytesDataset (or any other URI-scheme dataset)
    let dataset = create_dataset(uri, opts.clone())?;

    // For async loaders, default to batch_size = 1 for intuitive individual item iteration
    let mut loader_opts = opts_from_dict(opts);
    if loader_opts.batch_size == LoaderOptions::default().batch_size {
        // User didn't specify batch_size, default to 1 for async loaders
        loader_opts.batch_size = 1;
    }

    Py::new(
        py,
        PyBytesAsyncDataLoader {
            dataset,
            opts: loader_opts,
        },
    )?
    .into_py_any(py)
}

// ---------------------------------------------------------------------------
// Compatibility shims for existing code
// ---------------------------------------------------------------------------

/// Compatibility wrapper - creates a PyDataset but only for S3 URIs
/// DEPRECATED: Use PyDataset or create_dataset() instead
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyS3DatasetCompat {
    inner: PyDataset,
}

#[pymethods]
impl PyS3DatasetCompat {
    #[new]
    #[pyo3(signature = (uri, opts=None))]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        use crate::object_store::infer_scheme;

        if !matches!(infer_scheme(uri), crate::object_store::Scheme::S3) {
            return Err(PyRuntimeError::new_err(
                "PyS3Dataset only supports S3 URIs. Use PyDataset for other schemes.",
            ));
        }

        eprintln!("Warning: PyS3Dataset is deprecated. Use PyDataset or create_dataset() instead.");
        let dataset = PyDataset::new(uri, opts)?;
        Ok(Self { inner: dataset })
    }

    fn keys(&self) -> PyResult<Vec<String>> {
        self.inner.keys()
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner.__len__()
    }

    fn get_item<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Py<PyAny>> {
        self.inner.get_item(py, index)
    }
}

/// Compatibility wrapper for PyS3AsyncDataLoader
/// DEPRECATED: Use PyBytesAsyncDataLoader with create_dataset() instead
#[pyclass]
pub struct PyS3AsyncDataLoaderCompat {
    inner: PyBytesAsyncDataLoader,
}

#[pymethods]
impl PyS3AsyncDataLoaderCompat {
    #[new]
    #[pyo3(signature = (uri, opts=None))]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        use crate::object_store::infer_scheme;

        if !matches!(infer_scheme(uri), crate::object_store::Scheme::S3) {
            return Err(PyRuntimeError::new_err(
                "PyS3AsyncDataLoader only supports S3 URIs. Use create_async_loader() for other schemes."
            ));
        }

        eprintln!("Warning: PyS3AsyncDataLoader is deprecated. Use create_async_loader() instead.");
        // Build a PyBytesAsyncDataLoader directly (the compat type wraps that struct).
        // We can't call create_async_loader() here because its return type is now
        // PyObject (to accommodate both PyBytesAsyncDataLoader and PyParquetStreamLoader).
        let dataset = create_dataset(uri, opts.clone())?;
        let mut loader_opts = opts_from_dict(opts);
        if loader_opts.batch_size == LoaderOptions::default().batch_size {
            loader_opts.batch_size = 1;
        }
        Ok(Self {
            inner: PyBytesAsyncDataLoader {
                dataset,
                opts: loader_opts,
            },
        })
    }

    fn __aiter__(&self) -> PyResult<PyBytesAsyncDataLoader> {
        // Show deprecation warning but still work for compatibility
        eprintln!("Warning: PyS3AsyncDataLoader is deprecated. Use create_async_loader() instead: loader = s3dlio.create_async_loader(uri, opts)");
        Ok(self.inner.clone())
    }
}

// ---------------------------------------------------------------------------
// TFRecord Index Generation (NVIDIA DALI Compatible)
// ---------------------------------------------------------------------------

/// Generate NVIDIA DALI-compatible index file for a TFRecord file
///
/// Creates a text-format index file with "{offset} {size}\n" format that can be used
/// by NVIDIA DALI's fn.readers.tfrecord(index_path=...) and TensorFlow tooling.
///
/// Args:
///     tfrecord_path: Path to the TFRecord file (input)
///     index_path: Path to the index file (output, typically .idx extension).
///                 If None, defaults to tfrecord_path + ".idx"
///
/// Returns:
///     Number of records indexed
///
/// Example:
///     >>> import s3dlio
///     >>> num_records = s3dlio.create_tfrecord_index("train.tfrecord", "train.tfrecord.idx")
///     >>> print(f"Indexed {num_records} records")
#[pyfunction]
#[pyo3(signature = (tfrecord_path, index_path=None))]
pub fn create_tfrecord_index(tfrecord_path: &str, index_path: Option<&str>) -> PyResult<usize> {
    use crate::tfrecord_index::write_index_for_tfrecord_file;
    use std::path::Path;

    // Determine output path
    let output_path = match index_path {
        Some(p) => p.to_string(),
        None => format!("{}.idx", tfrecord_path),
    };

    // Create index
    let num_records =
        write_index_for_tfrecord_file(Path::new(tfrecord_path), Path::new(&output_path))
            .map_err(py_err)?;

    Ok(num_records)
}

/// Generate DALI-compatible index text from TFRecord bytes (in-memory)
///
/// Parses TFRecord data from bytes and returns index text in NVIDIA DALI format.
/// Useful for processing TFRecord data already loaded into memory.
///
/// Args:
///     tfrecord_bytes: Raw TFRecord file data as bytes
///
/// Returns:
///     Index text string in "{offset} {size}\n" format
///
/// Example:
///     >>> import s3dlio
///     >>> with open("train.tfrecord", "rb") as f:
///     ...     data = f.read()
///     >>> index_text = s3dlio.index_tfrecord_bytes(data)
///     >>> with open("train.tfrecord.idx", "w") as f:
///     ...     f.write(index_text)
#[pyfunction]
pub fn index_tfrecord_bytes(tfrecord_bytes: &[u8]) -> PyResult<String> {
    use crate::tfrecord_index::index_text_from_bytes;

    let index_text = index_text_from_bytes(tfrecord_bytes).map_err(py_err)?;
    Ok(index_text)
}

/// Get index entries from TFRecord bytes (returns list of (offset, size) tuples)
///
/// Parses TFRecord data and returns structured index information as Python tuples.
/// Each tuple contains (offset: int, size: int) for one TFRecord entry.
///
/// Args:
///     tfrecord_bytes: Raw TFRecord file data as bytes
///
/// Returns:
///     List of (offset, size) tuples
///
/// Example:
///     >>> import s3dlio
///     >>> with open("train.tfrecord", "rb") as f:
///     ...     data = f.read()
///     >>> entries = s3dlio.get_tfrecord_index_entries(data)
///     >>> for offset, size in entries:
///     ...     print(f"Record at offset {offset}, size {size}")
#[pyfunction]
pub fn get_tfrecord_index_entries(tfrecord_bytes: &[u8]) -> PyResult<Vec<(u64, u64)>> {
    use crate::tfrecord_index::index_entries_from_bytes;

    let entries = index_entries_from_bytes(tfrecord_bytes).map_err(py_err)?;
    let tuples: Vec<(u64, u64)> = entries.into_iter().map(|e| (e.offset, e.size)).collect();

    Ok(tuples)
}

/// Read and parse a DALI-compatible TFRecord index file
///
/// Reads an index file in NVIDIA DALI format and parses it into
/// structured index entries as Python tuples.
///
/// Args:
///     index_path: Path to the index file (.idx)
///
/// Returns:
///     List of (offset, size) tuples
///
/// Example:
///     >>> import s3dlio
///     >>> entries = s3dlio.read_tfrecord_index("train.tfrecord.idx")
///     >>> print(f"Found {len(entries)} records")
///     >>> # Use entries for random access
///     >>> with open("train.tfrecord", "rb") as f:
///     ...     for offset, size in entries:
///     ...         f.seek(offset)
///     ...         record_bytes = f.read(size)
///     ...         # Process record...
#[pyfunction]
pub fn read_tfrecord_index(index_path: &str) -> PyResult<Vec<(u64, u64)>> {
    use crate::tfrecord_index::read_index_file;
    use std::path::Path;

    let entries = read_index_file(Path::new(index_path)).map_err(py_err)?;
    let tuples: Vec<(u64, u64)> = entries.into_iter().map(|e| (e.offset, e.size)).collect();

    Ok(tuples)
}

// ---------------------------------------------------------------------------
// Parquet metadata cache helpers
// ---------------------------------------------------------------------------

/// Return cumulative row-group row offsets for a single Parquet file.
///
/// The result is a list of length `num_row_groups + 1`:
/// - `offsets[i]`    = first sample row in row group `i`
/// - `offsets[-1]`   = total rows in the file
///
/// The metadata is fetched from s3 on the first call and cached for the
/// process lifetime (zero network I/O on subsequent calls).
///
/// This replaces any Python-side footer read (e.g. `_build_rg_offsets` with
/// PyArrow + `_S3RangeFile`) with a single Rust call that hits the in-memory
/// cache after the first `create_async_loader` for the same file.
///
/// Example::
///
///     import s3dlio
///     offsets = s3dlio.parquet_rg_row_offsets("s3://bucket/path/file.parquet")
///     # offsets == [0, 8192, 16384, ..., 1000000]
#[pyfunction]
pub fn parquet_rg_row_offsets(py: Python<'_>, uri: &str) -> PyResult<Vec<i64>> {
    use crate::data_loader::parquet_file_cache;
    use crate::data_loader::parquet_rg::DEFAULT_FOOTER_CAP;
    use crate::s3_client::run_on_global_rt;

    let uri_owned = uri.to_owned();
    // GIL released while the footer is fetched (issue #201).
    py.detach(|| {
        let cached = run_on_global_rt(async move {
            parquet_file_cache::get_or_fetch(&uri_owned, DEFAULT_FOOTER_CAP as u64).await
        })
        .map_err(py_err)?;

        Ok(cached.rg_row_offsets.clone())
    })
}

/// Evict a single file from the Parquet metadata cache.
///
/// Useful after a file has been rewritten and you want the next
/// `parquet_rg_row_offsets` / `create_async_loader` call to re-fetch it.
#[pyfunction]
pub fn parquet_cache_invalidate(uri: &str) {
    use crate::data_loader::parquet_file_cache;
    parquet_file_cache::invalidate(uri);
}

/// Clear the entire Parquet metadata cache.
#[pyfunction]
pub fn parquet_cache_clear() {
    use crate::data_loader::parquet_file_cache;
    parquet_file_cache::clear();
}

/// Return the number of files currently held in the Parquet metadata cache.
#[pyfunction]
pub fn parquet_cache_len() -> usize {
    use crate::data_loader::parquet_file_cache;
    parquet_file_cache::len()
}

// ---------------------------------------------------------------------------
// PyParquetIndex — epoch-aware Parquet row-group byte-range index
// ---------------------------------------------------------------------------

/// Epoch-aware Parquet row-group byte-range index.
///
/// `PyParquetIndex` caches `(file_uri, rg_idx) → (byte_offset, byte_length)`
/// for every row group across all indexed files.  Footers are fetched lazily
/// in configurable batches and dropped after extracting byte ranges.  Entries
/// are retained across epochs; `last_epoch` tracks per-epoch fetch status so
/// row groups are not re-fetched within the same epoch.
///
/// ## Usage
///
/// ```python
/// import s3dlio
///
/// idx = s3dlio.PyParquetIndex()          # col_indices=None (all), footer_cap=4 MiB
///
/// # Index a batch of files (fetches footers concurrently, batch_size at a time)
/// uris = ["s3://bucket/data/file0.parquet", "s3://bucket/data/file1.parquet"]
/// idx.ensure_indexed(uris, epoch=1, batch_size=16)
///
/// # Fast O(1) lookup: sample index → RG index + byte range
/// rg_idx, offset, length = idx.rg_lookup("s3://bucket/data/file0.parquet", sample_idx=42)
///
/// # Issue the range GET directly
/// data = s3dlio.get_range("s3://bucket/data/file0.parquet", offset, length)
///
/// # Mark as fetched this epoch (avoids duplicate GETs)
/// idx.mark_fetched("s3://bucket/data/file0.parquet", rg_idx, epoch=1)
///
/// # Next epoch: entries already present, just check last_epoch
/// idx.ensure_indexed(uris, epoch=2, batch_size=16)  # no-op: footers cached
/// rg_idx, offset, length = idx.rg_lookup(...)       # still O(1)
/// ```
#[pyclass]
pub struct PyParquetIndex {
    inner: std::sync::Arc<crate::data_loader::parquet_index::ParquetIndex>,
}

#[pymethods]
impl PyParquetIndex {
    /// Create a new empty index.
    ///
    /// Args:
    ///     col_indices: List of column indices to include in byte-range
    ///         computation, or ``None`` to include all columns.  Must be
    ///         consistent across all files.
    ///     footer_cap: Bytes fetched from each file tail for footer parsing.
    ///         Default 4 MiB covers DLRM (~2.66 MiB footers) and most workloads.
    #[new]
    #[pyo3(signature = (col_indices=None, footer_cap=4194304))]
    fn new(col_indices: Option<Vec<usize>>, footer_cap: u64) -> Self {
        Self {
            inner: std::sync::Arc::new(crate::data_loader::parquet_index::ParquetIndex::new(
                col_indices,
                footer_cap,
            )),
        }
    }

    /// Ensure all URIs in `uris` have their row groups indexed.
    ///
    /// Already-indexed URIs are skipped.  Footers are fetched concurrently in
    /// batches of `batch_size` (default 16).  Blocks until the batch completes,
    /// releasing the GIL during network I/O.
    ///
    /// Args:
    ///     uris: S3/file URIs to index.
    ///     epoch: Current epoch number (used to initialise ``last_epoch``).
    ///     batch_size: Concurrent footer fetches per batch (default 16).
    #[pyo3(signature = (uris, epoch=1, batch_size=16))]
    fn ensure_indexed(
        &self,
        py: Python<'_>,
        uris: Vec<String>,
        epoch: u32,
        batch_size: usize,
    ) -> PyResult<()> {
        let inner = std::sync::Arc::clone(&self.inner);
        py.detach(|| {
            inner
                .ensure_indexed(&uris, epoch, batch_size)
                .map_err(py_err)
        })
    }

    /// Find which row group contains `sample_idx` (0-based within the file).
    ///
    /// Returns the row-group index.  Raises ``RuntimeError`` if the URI is not
    /// yet indexed.
    fn rg_for_sample(&self, uri: &str, sample_idx: i64) -> PyResult<u32> {
        self.inner.rg_for_sample(uri, sample_idx).map_err(py_err)
    }

    /// Return ``(byte_offset, byte_length)`` for ``(uri, rg_idx)``.
    fn rg_range(&self, uri: &str, rg_idx: u32) -> PyResult<(u64, u64)> {
        self.inner.rg_range(uri, rg_idx).map_err(py_err)
    }

    /// Combined lookup: sample → ``(rg_idx, byte_offset, byte_length)``.
    ///
    /// Combines ``rg_for_sample`` + ``rg_range`` into one PyO3 call, halving
    /// FFI overhead on the per-sample hot path.
    fn rg_lookup(&self, uri: &str, sample_idx: i64) -> PyResult<(u32, u64, u64)> {
        self.inner.rg_lookup(uri, sample_idx).map_err(py_err)
    }

    /// ``True`` if ``(uri, rg_idx)`` was fetched during epoch ``epoch``.
    fn was_fetched(&self, uri: &str, rg_idx: u32, epoch: u32) -> bool {
        self.inner.was_fetched(uri, rg_idx, epoch)
    }

    /// Record that ``(uri, rg_idx)`` was fetched during epoch ``epoch``.
    fn mark_fetched(&self, uri: &str, rg_idx: u32, epoch: u32) {
        self.inner.mark_fetched(uri, rg_idx, epoch);
    }

    /// Combined ``was_fetched`` + ``mark_fetched`` in one DashMap access.
    ///
    /// Returns ``True`` if already fetched this epoch (caller skips GET).
    /// If not fetched, atomically sets ``last_epoch = epoch`` and returns
    /// ``False`` (caller should issue the GET).
    fn check_and_mark(&self, uri: &str, rg_idx: u32, epoch: u32) -> bool {
        self.inner.check_and_mark(uri, rg_idx, epoch)
    }

    /// ``True`` if ``uri`` is already fully indexed.
    fn is_indexed(&self, uri: &str) -> bool {
        self.inner.is_indexed(uri)
    }

    /// Number of row groups in ``uri``, or ``None`` if not indexed.
    fn file_rg_count(&self, uri: &str) -> Option<u32> {
        self.inner.file_rg_count(uri)
    }

    /// Number of row-group entries in the index (across all files).
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of files whose row groups are indexed.
    fn file_count(&self) -> usize {
        self.inner.file_count()
    }

    /// Remove all entries for ``uri`` (e.g. after a file is rewritten).
    fn invalidate_uri(&self, uri: &str) {
        self.inner.invalidate_uri(uri);
    }

    /// Clear all row-group entries (keeps URI interning table).
    fn clear(&self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "PyParquetIndex(files={}, rgs={}, col_indices={:?}, footer_cap={})",
            self.inner.file_count(),
            self.inner.len(),
            self.inner.col_indices,
            self.inner.footer_cap,
        )
    }
}

// ---------------------------------------------------------------------------
// Tokio thread-count configuration for the pyo3 async runtime
// ---------------------------------------------------------------------------

/// Configure the number of Tokio worker threads used by `create_async_loader`.
///
/// Controls the **`pyo3_async_runtimes`** Tokio runtime — the one that drives
/// the `buffer_unordered` streaming pipeline and Rust Arrow IPC decode.
///
/// **Must be called before the first `create_async_loader` call** in each
/// process.  Once the runtime is first used it is permanently fixed;
/// subsequent calls are silently ignored.
///
/// ## Default (n=0): auto-detect from MPI environment variables
///
/// When called with no arguments (or `n=0`), reads standard MPI env vars to
/// determine world size and divides available CPUs accordingly:
///
/// ```python
/// import s3dlio
/// s3dlio.configure_tokio_threads()   # auto: reads OMPI_COMM_WORLD_SIZE etc.
/// ```
///
/// Checked env vars (first one found wins):
///   - `OMPI_COMM_WORLD_SIZE` — Open MPI
///   - `PMI_SIZE`             — MPICH / Slurm PMI
///   - `WORLD_SIZE`           — PyTorch distributed / torchrun
///
/// ## Explicit thread count
///
/// ```python
/// s3dlio.configure_tokio_threads(4)  # force 4 Tokio threads per process
/// ```
///
/// ## Arguments
/// * `n` — worker thread count.  `0` (default) = auto from MPI env vars.
///
/// Since v0.9.113 this also sizes Rayon's global thread pool (used by the
/// NPZ/data-gen `par_chunks_mut` hot paths) with the same budget — see
/// [`crate::s3_client::configure_thread_pools`], which this function now
/// delegates to. Note that as of v0.9.113 `s3dlio` also calls this
/// automatically (with `n=0`, i.e. auto-detect) once at import time, so
/// explicitly calling it is only needed to override the auto-detected
/// budget.
#[pyfunction]
#[pyo3(signature = (n=0))]
pub fn configure_tokio_threads(n: usize) {
    crate::s3_client::configure_thread_pools(n);
}

// ---------------------------------------------------------------------------
// Module registration function for AI/ML API
// ---------------------------------------------------------------------------
pub fn register_aiml_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // NumPy functions (temporarily disabled due to PyO3/numpy compatibility issues)
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;
    // TODO: Re-enable when PyO3/numpy issues are resolved:
    // m.add_function(wrap_pyfunction!(save_numpy_array, m)?)?;
    // m.add_function(wrap_pyfunction!(load_numpy_array, m)?)?;

    // Data loader classes
    m.add_class::<PyLoaderOptions>()?;
    m.add_class::<PyS3Dataset>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyVecDataset>()?;
    m.add_class::<PyAsyncDataLoader>()?;
    m.add_class::<PyAsyncDataLoaderIter>()?;
    m.add_class::<PyBytesAsyncDataLoader>()?;
    m.add_class::<PyBytesAsyncDataLoaderIter>()?;
    m.add_class::<PyBytesDataLoaderSyncIter>()?;
    m.add_class::<PyObjectItem>()?;
    m.add_class::<PyObjectDataLoaderSyncIter>()?;
    m.add_class::<PyParquetItem>()?;
    m.add_class::<ParquetStreamIter>()?;
    m.add_class::<PyParquetStreamLoader>()?;
    m.add_class::<PyParquetIndex>()?;

    // Compatibility classes (deprecated)
    m.add_class::<PyS3DatasetCompat>()?;
    m.add_class::<PyS3AsyncDataLoaderCompat>()?;

    // Dataset factory functions
    m.add_function(wrap_pyfunction!(create_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_loader, m)?)?;
    m.add_function(wrap_pyfunction!(configure_tokio_threads, m)?)?;
    // Per-RG fetch + optional Rust Arrow decode (index-based hot path)
    m.add_function(wrap_pyfunction!(parquet_get_rg, m)?)?;
    // Fast Parquet file generation (all-Float32, random data via dgen_data)
    m.add_function(wrap_pyfunction!(generate_parquet_bytes, m)?)?;
    // Generate + write entire file in one GIL-release (high-level API)
    m.add_function(wrap_pyfunction!(generate_and_write_parquet, m)?)?;
    // Schema-flexible generation: named columns, per-column float32 count
    m.add_function(wrap_pyfunction!(generate_parquet_bytes_schema, m)?)?;
    m.add_function(wrap_pyfunction!(generate_and_write_parquet_schema, m)?)?;
    m.add_function(wrap_pyfunction!(
        generate_and_write_parquet_schema_streaming,
        m
    )?)?;

    // Parquet metadata cache helpers
    m.add_function(wrap_pyfunction!(parquet_rg_row_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_cache_invalidate, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_cache_clear, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_cache_len, m)?)?;

    // TFRecord index generation (NVIDIA DALI compatible)
    m.add_function(wrap_pyfunction!(create_tfrecord_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_tfrecord_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(get_tfrecord_index_entries, m)?)?;
    m.add_function(wrap_pyfunction!(read_tfrecord_index, m)?)?;

    // Add compatibility aliases (these create the deprecated names that existing code expects)
    // The user should migrate to PyDataset and create_async_loader() instead
    m.add(
        "PyS3AsyncDataLoader",
        m.getattr("PyS3AsyncDataLoaderCompat")?,
    )?;

    // Checkpoint system (if enabled)
    #[cfg(feature = "extension-module")]
    {
        m.add_class::<PyCheckpointStore>()?;
        m.add_class::<PyCheckpointWriter>()?;
        m.add_class::<PyCheckpointReader>()?;
        m.add_class::<PyCheckpointInfo>()?;
        m.add_function(wrap_pyfunction!(save_checkpoint, m)?)?;
        m.add_function(wrap_pyfunction!(load_checkpoint, m)?)?;
        m.add_function(wrap_pyfunction!(save_distributed_shard, m)?)?;
        m.add_function(wrap_pyfunction!(finalize_distributed_checkpoint, m)?)?;
        m.add_function(wrap_pyfunction!(load_checkpoint_with_validation, m)?)?;
        m.add_function(wrap_pyfunction!(load_checkpoint_with_validation_view, m)?)?;
    }

    Ok(())
}

#[cfg(all(test, feature = "extension-module"))]
mod checkpoint_runtime_sizing_tests {
    use super::*;
    use std::time::{Duration, Instant};

    // build_checkpoint_runtime() reads S3DLIO_RT_THREADS; serialize against
    // any other test in this binary that touches the same env var via the
    // crate-wide guard `crate::test_env` (issue #216).

    /// Reproduces the CPU-oversubscription bug found while chasing the DLIO
    /// unet3d npz-datagen slowdown, applied to checkpointing specifically:
    /// `PyCheckpointStore` builds its own dedicated Tokio runtime (separate
    /// from s3dlio's shared global one), and it used to do so via a bare
    /// `tokio::runtime::Runtime::new()` -- Tokio's own unconstrained
    /// `num_cpus`-worker default, completely bypassing
    /// `configure_thread_pools()` / `effective_thread_budget()`. Under
    /// `mpirun -n N`, N ranks checkpointing concurrently on one host would
    /// each independently claim the full core count.
    ///
    /// This proves the checkpoint runtime is actually CAPPED to the
    /// requested worker count (2, via `S3DLIO_RT_THREADS=2`) rather than
    /// defaulting to the host's full core count: 3 synchronous (non-
    /// yielding) blocking tasks on a 2-worker runtime must run in two
    /// waves, taking measurably longer than one wave would. An
    /// unconstrained runtime (e.g. this host's `num_cpus` >> 3) would run
    /// all 3 concurrently in a single wave and finish much faster.
    #[test]
    fn build_checkpoint_runtime_honors_explicit_thread_budget() {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(&["S3DLIO_RT_THREADS"]);
        std::env::set_var("S3DLIO_RT_THREADS", "2");

        let rt = build_checkpoint_runtime().expect("build checkpoint runtime");
        const SLEEP: Duration = Duration::from_millis(150);
        const TASKS: usize = 3;

        let start = Instant::now();
        rt.block_on(async {
            let mut set = tokio::task::JoinSet::new();
            for _ in 0..TASKS {
                // A synchronous, non-yielding sleep genuinely occupies an
                // OS worker thread -- unlike tokio::time::sleep().await,
                // which would free the thread for other tasks and defeat
                // the whole point of this test.
                set.spawn(async move { std::thread::sleep(SLEEP) });
            }
            while set.join_next().await.is_some() {}
        });
        let elapsed = start.elapsed();

        // 2 workers, 3 tasks of 150ms each -> 2 waves -> >= ~300ms. An
        // unconstrained pool (this host's real core count) would run all 3
        // concurrently in ~150ms. 250ms sits safely between the two.
        assert!(
            elapsed >= Duration::from_millis(250),
            "checkpoint runtime finished 3x150ms blocking tasks in {elapsed:?} -- \
             too fast for a 2-worker-thread pool, meaning S3DLIO_RT_THREADS=2 \
             was not applied and the runtime is still using an unconstrained \
             (full-core-count) default"
        );
    }
}

/// Final review F2: the shard-metadata <-> dict conversions keep the optional checksum in both
/// directions (the Python regression `python/tests/test_checkpoint_checksum_f2.py` covers the
/// whole stream -> manifest -> read path; these pin the conversion functions themselves).
#[cfg(all(test, feature = "extension-module"))]
mod shard_meta_conversion_tests {
    use super::*;
    use crate::checkpoint::{Manifest, ShardMeta};

    fn init() {
        Python::initialize();
    }

    fn shard(checksum: Option<&str>) -> ShardMeta {
        ShardMeta {
            rank: 2,
            key: "shards/ckpt-3-x/rank-2.bin.zst".to_string(),
            size: 41,
            etag: Some("\"etag-1\"".to_string()),
            checksum: checksum.map(str::to_string),
        }
    }

    #[test]
    fn checksum_survives_meta_to_dict_to_meta() {
        init();
        Python::attach(|py| {
            let original = shard(Some("crc32c:e3069283"));
            let dict = PyDict::new(py);
            set_shard_meta_items(&dict, &original).unwrap();
            let c: String = dict
                .get_item("checksum")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(c, "crc32c:e3069283");

            let back = shard_meta_from_dict(&dict).unwrap();
            assert_eq!(back.rank, original.rank);
            assert_eq!(back.key, original.key);
            assert_eq!(back.size, original.size);
            assert_eq!(back.etag, original.etag);
            assert_eq!(back.checksum.as_deref(), Some("crc32c:e3069283"));
        });
    }

    #[test]
    fn absent_checksum_stays_absent_in_both_directions() {
        init();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            set_shard_meta_items(&dict, &shard(None)).unwrap();
            assert!(dict.get_item("checksum").unwrap().is_none());
            assert!(shard_meta_from_dict(&dict).unwrap().checksum.is_none());
            assert!(shard_checksum_from_dict(&dict).unwrap().is_none());

            // A Python None is the same as absent.
            dict.set_item("checksum", py.None()).unwrap();
            assert!(shard_meta_from_dict(&dict).unwrap().checksum.is_none());
        });
    }

    #[test]
    fn non_string_checksum_is_a_type_error() {
        init();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            set_shard_meta_items(&dict, &shard(None)).unwrap();
            dict.set_item("checksum", 12345u32).unwrap();
            let err = shard_meta_from_dict(&dict).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
        });
    }

    #[test]
    fn malformed_checksum_is_refused_and_empty_means_absent() {
        init();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            set_shard_meta_items(&dict, &shard(None)).unwrap();
            for bad in [
                "md5:abc",
                "zzz",
                "crc32c:123456789",
                "0xe3069283",
                "crc32c:",
            ] {
                dict.set_item("checksum", bad).unwrap();
                assert!(
                    shard_meta_from_dict(&dict).is_err(),
                    "{bad} must be refused"
                );
            }
            dict.set_item("checksum", "").unwrap();
            assert!(shard_meta_from_dict(&dict).unwrap().checksum.is_none());
            // Both accepted spellings pass through unchanged.
            for ok in ["crc32c:e3069283", "e3069283"] {
                dict.set_item("checksum", ok).unwrap();
                assert_eq!(
                    shard_meta_from_dict(&dict).unwrap().checksum.as_deref(),
                    Some(ok)
                );
            }
        });
    }

    #[test]
    fn manifest_dict_carries_each_shards_checksum() {
        init();
        Python::attach(|py| {
            let mut manifest = Manifest::new("test".to_string(), 3, 1, 2);
            manifest.add_shard(shard(Some("crc32c:e3069283")));
            manifest.add_shard(ShardMeta {
                rank: 3,
                ..shard(None)
            });
            let dict = manifest_to_pydict(py, &manifest).unwrap();
            let shards = dict.get_item("shards").unwrap().unwrap();
            let shards = shards.cast::<PyList>().unwrap();
            let with = shards.get_item(0).unwrap();
            let without = shards.get_item(1).unwrap();
            assert_eq!(
                shard_checksum_from_dict(with.cast::<PyDict>().unwrap())
                    .unwrap()
                    .as_deref(),
                Some("crc32c:e3069283")
            );
            assert!(shard_checksum_from_dict(without.cast::<PyDict>().unwrap())
                .unwrap()
                .is_none());
        });
    }
}

/// #285: every constructor documented as taking an optional `opts` must be
/// callable with only a URI (pyo3 0.27 makes a bare `Option<T>` parameter
/// required unless a `#[pyo3(signature = ...)]` gives it a default).
#[cfg(all(test, feature = "extension-module"))]
mod optional_opts_constructor_tests {
    use super::*;
    use pyo3::types::PyTuple;

    /// A URI that never touches the network. The S3-only constructors reject
    /// it with their documented RuntimeError; the generic ones may succeed or
    /// fail on the missing directory. None may fail on a missing argument.
    const URI: &str = "file:///nonexistent-s3dlio-285-dir/";

    fn assert_opts_optional(py: Python<'_>, name: &str, cls: Bound<'_, pyo3::types::PyType>) {
        let args = PyTuple::new(py, [URI]).unwrap();
        if let Err(e) = cls.call1(args) {
            let msg = e.to_string();
            assert!(
                !(e.is_instance_of::<PyTypeError>(py) && msg.contains("opts")),
                "{name}(uri) must not require 'opts', got: {msg}"
            );
        }
    }

    #[test]
    fn constructors_accept_a_uri_alone() {
        Python::initialize();
        Python::attach(|py| {
            assert_opts_optional(py, "PyS3Dataset", py.get_type::<PyS3Dataset>());
            assert_opts_optional(py, "PyDataset", py.get_type::<PyDataset>());
            assert_opts_optional(
                py,
                "PyS3AsyncDataLoader",
                py.get_type::<PyS3AsyncDataLoader>(),
            );
            assert_opts_optional(py, "PyS3DatasetCompat", py.get_type::<PyS3DatasetCompat>());
            assert_opts_optional(
                py,
                "PyS3AsyncDataLoaderCompat",
                py.get_type::<PyS3AsyncDataLoaderCompat>(),
            );
        });
    }
}

/// #286: every checkpoint entry point whose published signature advertises
/// `user_meta=None` must be callable without it, and #287: the integrity
/// check of `load_checkpoint_with_validation` must compare against the
/// checksum the checkpoint recorded when it was written.
#[cfg(all(test, feature = "extension-module"))]
mod checkpoint_286_287_tests {
    use super::*;
    use pyo3::types::PyTuple;

    fn shard_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for entry in rd.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else {
                        out.push(p);
                    }
                }
            }
        }
        let mut all = Vec::new();
        walk(&root.join("shards"), &mut all);
        all
    }

    fn module<'py>(py: Python<'py>) -> Bound<'py, PyModule> {
        let m = PyModule::new(py, "ckpt_286").unwrap();
        m.add_function(wrap_pyfunction!(save_checkpoint, &m).unwrap())
            .unwrap();
        m.add_function(wrap_pyfunction!(save_distributed_shard, &m).unwrap())
            .unwrap();
        m.add_function(wrap_pyfunction!(finalize_distributed_checkpoint, &m).unwrap())
            .unwrap();
        m.add_function(wrap_pyfunction!(load_checkpoint_with_validation, &m).unwrap())
            .unwrap();
        m
    }

    #[test]
    fn user_meta_is_optional_everywhere_it_is_advertised() {
        Python::initialize();
        Python::attach(|py| {
            let m = module(py);

            // Module function save_checkpoint(uri, step, epoch, framework, data)
            let dir = tempfile::tempdir().unwrap();
            let uri = format!("file://{}", dir.path().display());
            let args = (uri.clone(), 1u64, 1u64, "test", PyBytes::new(py, b"abc"));
            m.getattr("save_checkpoint")
                .unwrap()
                .call1(args)
                .expect("save_checkpoint must accept a call without user_meta");

            // PyCheckpointStore.save(step, epoch, framework, data)
            let dir2 = tempfile::tempdir().unwrap();
            let uri2 = format!("file://{}", dir2.path().display());
            let store = Py::new(
                py,
                PyCheckpointStore::new(uri2.clone(), None, None, None).unwrap(),
            )
            .unwrap();
            store
                .bind(py)
                .call_method1("save", (2u64, 1u64, "test", PyBytes::new(py, b"abc")))
                .expect("PyCheckpointStore.save must accept a call without user_meta");

            // PyCheckpointWriter.finalize_distributed_checkpoint(step, epoch, framework, metas)
            let dir3 = tempfile::tempdir().unwrap();
            let uri3 = format!("file://{}", dir3.path().display());
            let store3 = PyCheckpointStore::new(uri3.clone(), None, None, None).unwrap();
            let writer = Py::new(py, store3.writer(1, 0).unwrap()).unwrap();
            let meta = writer
                .bind(py)
                .call_method1(
                    "save_distributed_shard",
                    (3u64, 1u64, "test", PyBytes::new(py, b"abc")),
                )
                .unwrap();
            let metas = vec![meta];
            writer
                .bind(py)
                .call_method1(
                    "finalize_distributed_checkpoint",
                    (3u64, 1u64, "test", metas.clone()),
                )
                .expect("PyCheckpointWriter.finalize_distributed_checkpoint must accept a call without user_meta");

            // Module function finalize_distributed_checkpoint(uri, step, epoch, framework,
            // shard_metas, world_size, rank)
            let dir4 = tempfile::tempdir().unwrap();
            let uri4 = format!("file://{}", dir4.path().display());
            let shard_args = PyTuple::new(
                py,
                [
                    uri4.clone().into_pyobject(py).unwrap().into_any(),
                    4u64.into_pyobject(py).unwrap().into_any(),
                    1u64.into_pyobject(py).unwrap().into_any(),
                    "test".into_pyobject(py).unwrap().into_any(),
                    PyBytes::new(py, b"abc").into_any(),
                    1u32.into_pyobject(py).unwrap().into_any(),
                    0u32.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let shard_meta = m
                .getattr("save_distributed_shard")
                .unwrap()
                .call1(shard_args)
                .unwrap();
            m.getattr("finalize_distributed_checkpoint")
                .unwrap()
                .call1((uri4, 4u64, 1u64, "test", vec![shard_meta], 1u32, 0u32))
                .expect("finalize_distributed_checkpoint must accept a call without user_meta");
        });
    }

    #[test]
    fn validated_load_round_trips_and_detects_corruption() {
        Python::initialize();
        Python::attach(|py| {
            let m = module(py);
            let dir = tempfile::tempdir().unwrap();
            let uri = format!("file://{}", dir.path().display());
            let payload = b"checkpoint payload for 287 round trip".to_vec();
            m.getattr("save_checkpoint")
                .unwrap()
                .call1((
                    uri.clone(),
                    7u64,
                    1u64,
                    "test",
                    PyBytes::new(py, &payload),
                    py.None(),
                ))
                .unwrap();

            let files = shard_files(dir.path());
            assert_eq!(files.len(), 1, "one shard expected, got {files:?}");
            let rel = files[0].strip_prefix(dir.path()).unwrap();
            let shard_uri = format!("{}/{}", uri, rel.display());

            // Clean round trip passes and returns the stored bytes.
            let got = load_checkpoint_with_validation(py, &shard_uri, true).unwrap();
            let got: Vec<u8> = got.extract(py).unwrap();
            assert_eq!(got, payload);

            // Corrupt the stored bytes (same length): validate_integrity=True must raise.
            let mut bad = payload.clone();
            bad[0] ^= 0xff;
            std::fs::write(&files[0], &bad).unwrap();
            let err = load_checkpoint_with_validation(py, &shard_uri, true)
                .expect_err("corrupted shard must fail integrity validation");
            assert!(
                err.to_string().contains("Integrity validation failed"),
                "unexpected error: {err}"
            );

            // validate_integrity=False keeps returning the stored bytes.
            let raw = load_checkpoint_with_validation(py, &shard_uri, false).unwrap();
            let raw: Vec<u8> = raw.extract(py).unwrap();
            assert_eq!(raw, bad);
        });
    }
}

/// R8 (RULE #1): `PyObjectItem` / `PyParquetItem` expose the buffer protocol and `.data`
/// (a `BytesView`), so reading them in place is zero copy; `bytes(item)` stays the one copy.
#[cfg(all(test, feature = "extension-module"))]
mod item_buffer_protocol_tests {
    use super::*;
    use pyo3::buffer::PyBuffer;

    fn payload() -> Bytes {
        Bytes::from((0..4096u32).map(|i| (i % 253) as u8).collect::<Vec<u8>>())
    }

    /// Address of the first byte `obj` exposes through the buffer protocol.
    fn buffer_address(obj: &Bound<'_, PyAny>) -> usize {
        let buf = PyBuffer::<u8>::get(obj).expect("buffer protocol");
        assert!(buf.readonly(), "the item's buffer must be read-only");
        buf.buf_ptr() as usize
    }

    fn check_in_place(py: Python<'_>, item: Bound<'_, PyAny>, data: &Bytes) {
        let want = data.as_ptr() as usize;

        // memoryview(item): zero copy, read-only, repeatable, same address every time.
        let builtins = py.import("builtins").expect("builtins");
        let mv1 = builtins
            .call_method1("memoryview", (&item,))
            .expect("memoryview");
        let mv2 = builtins
            .call_method1("memoryview", (&item,))
            .expect("memoryview");
        assert_eq!(
            buffer_address(&mv1),
            want,
            "memoryview(item) copied the data"
        );
        assert_eq!(buffer_address(&mv2), want);
        assert_eq!(mv1.len().expect("len"), data.len());
        let readonly: bool = mv1
            .getattr("readonly")
            .expect("readonly")
            .extract()
            .expect("bool");
        assert!(readonly);

        // item.data: a BytesView over the same allocation.
        let view = item.getattr("data").expect("data property");
        assert_eq!(buffer_address(&view), want, "item.data copied the data");
        let view = view.cast::<PyBytesView>().expect("BytesView").borrow();
        assert_eq!(view.bytes.as_ptr() as usize, want);
        drop(view);

        // The memoryview keeps the item (and so the Bytes) alive after the last name is dropped.
        drop(item);
        assert_eq!(buffer_address(&mv1), want);
        let first: u8 = mv1.get_item(0).expect("item").extract().expect("u8");
        assert_eq!(first, data[0]);
    }

    #[test]
    fn object_item_reads_in_place() {
        Python::initialize();
        Python::attach(|py| {
            let data = payload();
            let item = Py::new(py, PyObjectItem::new("file:///x".to_string(), data.clone()))
                .expect("item")
                .into_bound(py)
                .into_any();
            // `bytes(item)` still works and IS the copy (a different address).
            let copied = py
                .import("builtins")
                .expect("builtins")
                .call_method1("bytes", (&item,))
                .expect("bytes(item)");
            let copied = copied.cast::<PyBytes>().expect("bytes").as_bytes().to_vec();
            assert_eq!(copied, data.to_vec());
            check_in_place(py, item, &data);
        });
    }

    #[test]
    fn parquet_item_reads_in_place() {
        Python::initialize();
        Python::attach(|py| {
            let data = payload();
            let item = Py::new(
                py,
                PyParquetItem::new("file:///x.parquet".to_string(), 3, data.clone()),
            )
            .expect("item")
            .into_bound(py)
            .into_any();
            check_in_place(py, item, &data);
        });
    }

    #[test]
    fn writable_buffer_request_is_refused() {
        Python::initialize();
        Python::attach(|py| {
            let item = Py::new(py, PyObjectItem::new("file:///x".to_string(), payload()))
                .expect("item")
                .into_bound(py);
            let mut view = std::mem::MaybeUninit::<ffi::Py_buffer>::zeroed();
            // SAFETY: `item` is a live object and `view` is writable storage for a Py_buffer;
            // on failure CPython leaves it untouched and sets BufferError.
            let rc = unsafe {
                ffi::PyObject_GetBuffer(item.as_ptr(), view.as_mut_ptr(), ffi::PyBUF_WRITABLE)
            };
            assert_eq!(rc, -1, "a writable buffer must be refused");
            let err = PyErr::fetch(py);
            assert!(err.is_instance_of::<PyBufferError>(py), "{err}");
        });
    }

    #[test]
    fn empty_item_is_a_valid_empty_buffer() {
        Python::initialize();
        Python::attach(|py| {
            let item = Py::new(py, PyObjectItem::new("file:///e".to_string(), Bytes::new()))
                .expect("item")
                .into_bound(py)
                .into_any();
            let mv = py
                .import("builtins")
                .expect("builtins")
                .call_method1("memoryview", (&item,))
                .expect("memoryview");
            assert_eq!(mv.len().expect("len"), 0);
        });
    }
}

/// A `write_chunk` with an unsupported payload must not create (and truncate) the shard: the
/// payload is converted before the lazy writer is created.
#[cfg(all(test, feature = "extension-module"))]
mod stream_write_chunk_tests {
    use super::*;

    fn files_below(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    found.extend(files_below(&path));
                } else {
                    found.push(path);
                }
            }
        }
        found
    }

    #[test]
    fn unsupported_payload_creates_no_shard_and_finalize_still_fails() {
        Python::initialize();
        Python::attach(|py| {
            let dir = tempfile::tempdir().expect("tempdir");
            let uri = format!("file://{}", dir.path().display());
            let store = PyCheckpointStore::new(uri, None, None, None).expect("store");
            let writer = store.writer(1, 0).expect("writer");
            let mut stream = writer
                .get_distributed_shard_stream(py, 5, 1, "test".to_string())
                .expect("stream");

            let bad = 123i32.into_pyobject(py).expect("int").into_any();
            let err = stream
                .write_chunk(py, &bad)
                .expect_err("an int is not a payload");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            assert!(
                files_below(dir.path()).is_empty(),
                "a rejected payload created a shard file: {:?}",
                files_below(dir.path())
            );

            let err = stream
                .finalize(py)
                .expect_err("nothing was written, finalize must say so");
            assert!(err.to_string().contains("No data written"), "{err}");

            // A valid chunk afterwards still works on a fresh stream.
            let mut stream = writer
                .get_distributed_shard_stream(py, 6, 1, "test".to_string())
                .expect("stream");
            let ok = PyBytes::new(py, b"abc").into_any();
            assert_eq!(stream.write_chunk(py, &ok).expect("write"), 3);
        });
    }
}

#[cfg(test)]
mod loader_options_tests {
    use super::PyLoaderOptions;
    use pyo3::types::{PyAnyMethods, PyDictMethods};

    /// #283: the seven values shadowed by same-named builder methods are readable
    /// through the `*_value` getters, and every builder still returns the options
    /// object so calls chain.
    #[test]
    fn value_getters_return_values_set_by_builders() {
        let opts = PyLoaderOptions::new()
            .pin_memory(true)
            .num_workers(8)
            .shuffle(true, 42)
            .drop_last(true)
            .prefetch(4)
            .persistent_workers(true)
            .non_blocking(true);

        assert!(opts.get_pin_memory_value());
        assert_eq!(opts.get_num_workers_value(), 8);
        assert!(opts.get_shuffle_value());
        assert!(opts.get_drop_last_value());
        assert_eq!(opts.get_prefetch_value(), 4);
        assert!(opts.get_persistent_workers_value());
        assert!(opts.get_non_blocking_value());
        // The seed given to `shuffle` is still readable through its working getter.
        assert_eq!(opts.get_seed(), 42);
    }

    /// Each `_value` getter tracks the builder in both directions (set then unset).
    #[test]
    fn value_getters_follow_builder_changes() {
        let on = PyLoaderOptions::new()
            .pin_memory(true)
            .persistent_workers(true)
            .non_blocking(true)
            .drop_last(true)
            .shuffle(true, 7);
        let off = on
            .clone()
            .pin_memory(false)
            .persistent_workers(false)
            .non_blocking(false)
            .drop_last(false)
            .shuffle(false, 7)
            .num_workers(0)
            .prefetch(1);

        assert!(on.get_pin_memory_value());
        assert!(on.get_persistent_workers_value());
        assert!(on.get_non_blocking_value());
        assert!(on.get_drop_last_value());
        assert!(on.get_shuffle_value());
        assert!(!off.get_pin_memory_value());
        assert!(!off.get_persistent_workers_value());
        assert!(!off.get_non_blocking_value());
        assert!(!off.get_drop_last_value());
        assert!(!off.get_shuffle_value());
        assert_eq!(off.get_num_workers_value(), 0);
        assert_eq!(off.get_prefetch_value(), 1);
    }

    /// The presets (used by DLIO-style configs) set the shadowed values too.
    #[test]
    fn presets_are_visible_through_value_getters() {
        let gpu = PyLoaderOptions::new().gpu_optimized();
        assert!(gpu.get_pin_memory_value());
        assert!(gpu.get_persistent_workers_value());
        assert!(gpu.get_non_blocking_value());

        let debug = PyLoaderOptions::new().debug_mode();
        assert_eq!(debug.get_num_workers_value(), 0);
        assert!(!debug.get_persistent_workers_value());

        let dist = PyLoaderOptions::new().distributed_optimized(2, 8);
        assert!(dist.get_pin_memory_value());
        assert_eq!(dist.get_shard_rank(), 2);
        assert_eq!(dist.get_shard_world_size(), 8);
    }

    /// `to_dict()` reports every option, matching the getters, after a builder chain.
    #[test]
    fn to_dict_matches_getters_after_builder_chain() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let opts = PyLoaderOptions::new()
                .with_batch_size(64)
                .num_workers(6)
                .prefetch(3)
                .shuffle(true, 99)
                .drop_last(true)
                .pin_memory(true)
                .persistent_workers(true)
                .non_blocking(true)
                .with_timeout(12.5)
                .distributed_optimized(3, 8)
                .channels_last()
                .use_forkserver();
            let d = opts.to_dict(py).expect("to_dict");
            fn get<'py, T: pyo3::conversion::FromPyObjectOwned<'py>>(
                d: &pyo3::Bound<'py, pyo3::types::PyDict>,
                k: &str,
            ) -> T {
                d.get_item(k)
                    .expect("get_item")
                    .unwrap_or_else(|| panic!("missing key {k}"))
                    .extract::<T>()
                    .unwrap_or_else(|_| panic!("extract {k}"))
            }
            assert_eq!(get::<usize>(&d, "batch_size"), opts.get_batch_size());
            assert_eq!(get::<bool>(&d, "drop_last"), opts.get_drop_last_value());
            assert_eq!(get::<bool>(&d, "shuffle"), opts.get_shuffle_value());
            assert_eq!(get::<u64>(&d, "seed"), opts.get_seed());
            assert_eq!(
                get::<usize>(&d, "num_workers"),
                opts.get_num_workers_value()
            );
            assert_eq!(get::<usize>(&d, "prefetch"), opts.get_prefetch_value());
            assert_eq!(get::<bool>(&d, "pin_memory"), opts.get_pin_memory_value());
            assert_eq!(
                get::<bool>(&d, "persistent_workers"),
                opts.get_persistent_workers_value()
            );
            assert_eq!(
                get::<bool>(&d, "non_blocking"),
                opts.get_non_blocking_value()
            );
            assert_eq!(
                get::<Option<f64>>(&d, "timeout_seconds"),
                opts.get_timeout_seconds()
            );
            assert_eq!(get::<usize>(&d, "shard_rank"), opts.get_shard_rank());
            assert_eq!(
                get::<usize>(&d, "shard_world_size"),
                opts.get_shard_world_size()
            );
            assert_eq!(get::<usize>(&d, "shard_rank"), 3);
            assert_eq!(get::<usize>(&d, "shard_world_size"), 8);
            assert_eq!(get::<String>(&d, "memory_format"), "channels_last");
            assert_eq!(get::<String>(&d, "multiprocessing_context"), "forkserver");
            assert_eq!(get::<usize>(&d, "batch_size"), 64);
            assert_eq!(get::<Option<f64>>(&d, "timeout_seconds"), Some(12.5));
            assert!(get::<bool>(&d, "auto_tune") == opts.inner.auto_tune);
        });
    }

    /// `to_dict()` exposes sampler parameters losslessly via the
    /// `sampler_replacement`, `sampler_rank` and `sampler_world_size` keys.
    #[test]
    fn to_dict_reports_sampler_parameters() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            fn sampler(
                opts: &PyLoaderOptions,
                py: pyo3::Python<'_>,
            ) -> (String, Option<bool>, Option<usize>, Option<usize>) {
                let d = opts.to_dict(py).expect("to_dict");
                fn item<'py, T: pyo3::conversion::FromPyObjectOwned<'py>>(
                    d: &pyo3::Bound<'py, pyo3::types::PyDict>,
                    k: &str,
                ) -> T {
                    d.get_item(k)
                        .expect("get_item")
                        .unwrap_or_else(|| panic!("missing key {k}"))
                        .extract::<T>()
                        .unwrap_or_else(|_| panic!("extract {k}"))
                }
                (
                    item(&d, "sampler_type"),
                    item(&d, "sampler_replacement"),
                    item(&d, "sampler_rank"),
                    item(&d, "sampler_world_size"),
                )
            }

            let mut base = PyLoaderOptions::new();
            assert_eq!(
                sampler(&base, py),
                ("sequential".to_string(), None, None, None)
            );
            assert_eq!(
                sampler(&base.random_sampling(true), py),
                ("random".to_string(), Some(true), None, None)
            );
            assert_eq!(
                sampler(&base.random_sampling(false), py),
                ("random".to_string(), Some(false), None, None)
            );
            assert_eq!(
                sampler(&base.distributed_sampling(1, 4), py),
                ("distributed_random".to_string(), None, Some(1), Some(4))
            );
            // Builders return a new object and leave the receiver unchanged.
            assert_eq!(
                sampler(&base, py),
                ("sequential".to_string(), None, None, None)
            );
        });
    }
}

#[cfg(all(test, feature = "extension-module"))]
mod checkpoint_runtime_marker_tests {
    /// The checkpoint runtime hands workers off around file I/O (#295).
    #[test]
    fn checkpoint_runtime_workers_are_marked() {
        let rt = super::build_checkpoint_runtime().unwrap();
        assert!(rt.block_on(async {
            tokio::spawn(async { crate::runtime_support::would_hand_off() })
                .await
                .unwrap()
        }));
    }
}
