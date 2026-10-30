// src/python_api/python_core_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use super::async_bridge::future_into_py;
use super::put_payload::{
    acquire_writable, deadline_warning_text, export_stats, hold_for_call, latch_ready,
    note_internal_error, put_payload, put_payload_in_batch, quiesce_async, quiesce_blocking,
    snapshot_for_call, Latch, PayloadUse, WritableExport,
};
use bytes::Bytes;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::create_exception;
use pyo3::exceptions::{PyFileNotFoundError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::Bound;

use tokio::sync::oneshot;
use tokio::task;

use std::future::Future;
use std::path::PathBuf;

use tracing::{debug, warn, Level};

// Project crates
use crate::config::{Config, DataGenAlgorithm, DataGenMode, ObjectType};
use crate::delete_batch_plan::parse_s3_uri_for_global_client;
use crate::list_containers::list_containers as list_containers_rs;
use crate::object_error::{object_error, ObjectErrorKind};
use crate::object_store::{
    get_many_with_stores, store_for_uri_with_logger, store_routing_endpoint_for_uri,
};
use crate::redact::redact_uri;
use crate::s3_client::run_on_global_rt;
use crate::s3_logger::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
use crate::s3_utils::{
    create_bucket_for_uri, delete_bucket_for_uri, get_objects_parallel, parse_s3_uri,
    put_objects_with_random_data_and_type, s3_bucket_target, stat_object_many_async,
    DEFAULT_OBJECT_SIZE,
};
use crate::{generic_download_objects, generic_upload_files};

// Phase 2 streaming functionality imports
use crate::file_store::FileSystemObjectStore;
use crate::file_store_direct::ConfigurableFileSystemObjectStore;
#[cfg(feature = "backend-azure")]
use crate::object_store::AzureObjectStore;
use crate::object_store::{
    CompressionConfig, HeadMetadata, ObjectStore, ObjectWriter, ReceiveProgress, S3ObjectStore,
    WriterOptions,
};

// ---------------------------------------------------------------------------
// Client Caching + io_uring-style Submit for Python API
// ---------------------------------------------------------------------------
//
// Architecture: Instead of calling block_on() (which panics if called from
// within a Tokio runtime), we use the io_uring pattern:
//
//   Python thread → handle.spawn(async work) → channel.recv()
//
// This is exactly what `run_on_global_rt()` in s3_client.rs does:
//   1. SUBMIT: spawn the async future onto the global runtime
//   2. PROCESS: runtime worker threads handle the I/O
//   3. COMPLETE: result flows back through std::sync::mpsc channel
//
// The calling thread blocks on channel recv (NOT on block_on), so it works
// from ANY context — plain OS threads, Python ThreadPoolExecutor, or even
// inside another runtime.
//
// Zero-copy: Bytes (Arc-based ref-counted) flows through channels with no
// data copy — just moving the pointer.
// ---------------------------------------------------------------------------

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::Arc;

/// Cache key: uniquely identifies an ObjectStore configuration
///
/// Note: We DON'T include bucket in the key because AWS clients work across
/// all buckets in a region/endpoint. This maximizes cache hit rate.
///
/// `route` is the server the factory would pick for the URI: it is taken from
/// [`store_routing_endpoint_for_uri`], the same function
/// `store_for_uri_with_logger` calls, so a host-form URI
/// (`s3://host:port/bucket/key`) is cached per server and a bucket URI
/// (`None`) is cached per `AWS_ENDPOINT_URL` (issue #171).  The cache never
/// calls the factory on a hit, so anything the factory routes on must be in
/// the key.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    scheme: String,        // "s3", "az", "gs", "file", "direct"
    endpoint: String,      // AWS_ENDPOINT_URL or default
    region: String,        // AWS_REGION or default
    route: Option<String>, // per-endpoint URL the factory routes this URI to, if any
}

impl StoreKey {
    /// Errors when the routing decision does (an invalid server in the URI,
    /// issue #178): never mapped to "no route", which would alias the URI onto
    /// the global-client cache entry.
    fn from_uri(uri: &str) -> anyhow::Result<Self> {
        let scheme = uri.split("://").next().unwrap_or("s3");
        let endpoint = crate::s3_client::s3_endpoint_url_from_env().unwrap_or_default();
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        Ok(StoreKey {
            scheme: scheme.to_string(),
            endpoint,
            region,
            route: store_routing_endpoint_for_uri(uri)?,
        })
    }
}

/// One cache slot: the store for one [`StoreKey`], built at most once.
///
/// `store` is read lock-free on every call after the first. `build` serialises
/// the callers that find the slot empty, so one of them builds and the rest
/// wait for its result instead of each building a store (and an S3 client with
/// its own connection pool) that the others would then discard (issue #204).
/// A failed build leaves the slot empty, so the next caller retries.
struct StoreSlot {
    store: std::sync::OnceLock<Arc<dyn ObjectStore>>,
    build: std::sync::Mutex<()>,
}

/// Take a slot's build lock. Uncontended, this is a plain `try_lock`. When another caller is
/// building and this one is on a multi-thread Tokio runtime (the async entry points run on
/// the pyo3 runtime), the wait is wrapped in `block_in_place` so the runtime can run other
/// tasks on a replacement worker instead of losing this one for the whole cold start. Any
/// other thread just waits. A poisoned lock (a build that panicked) guards no data and is
/// recovered.
fn lock_build(build: &std::sync::Mutex<()>) -> std::sync::MutexGuard<'_, ()> {
    match build.try_lock() {
        Ok(guard) => return guard,
        Err(std::sync::TryLockError::Poisoned(poisoned)) => return poisoned.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {}
    }
    let on_multi_thread_runtime = tokio::runtime::Handle::try_current()
        .map(|handle| handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
        .unwrap_or(false);
    if on_multi_thread_runtime {
        tokio::task::block_in_place(|| build.lock().unwrap_or_else(|e| e.into_inner()))
    } else {
        build.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl StoreSlot {
    fn new() -> Self {
        Self {
            store: std::sync::OnceLock::new(),
            build: std::sync::Mutex::new(()),
        }
    }

    /// The slot's store, building it from `uri` if no caller has yet.
    ///
    /// Callers waiting on `build` are only those racing on the SAME key's first
    /// miss; every other key, and every later call on this one, never touches the
    /// lock. A poisoned lock (a build that panicked) guards no data, so it is
    /// recovered and the build retried.
    fn get_or_build(&self, uri: &str) -> anyhow::Result<Arc<dyn ObjectStore>> {
        if let Some(store) = self.store.get() {
            return Ok(Arc::clone(store));
        }
        let _building = lock_build(&self.build);
        if let Some(store) = self.store.get() {
            return Ok(Arc::clone(store)); // another caller built it while this one waited
        }
        // store_for_uri_with_logger is SYNC, no runtime needed. The store itself is
        // Send + Sync and uses the global runtime's I/O reactor when its async methods
        // are called. Built without a logger: the cache must stay logger-free.
        #[cfg(test)]
        {
            STORE_BUILD_LOG
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(uri.to_owned());
            STORE_BUILD_THREADS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((uri.to_owned(), std::thread::current().id()));
        }
        let built: Arc<dyn ObjectStore> = Arc::from(store_for_uri_with_logger(uri, None)?);
        Ok(Arc::clone(self.store.get_or_init(|| built)))
    }
}

/// Global cache of ObjectStore instances, one [`StoreSlot`] per [`StoreKey`].
///
/// DashMap provides lock-free concurrent access (better than RwLock<HashMap>)
/// for read-heavy workloads. Expected cache hit rate: >99% in typical workloads.
/// A slot is inserted with the map's entry API, so racing callers share one slot
/// per key and the slot makes the build happen once.
///
/// Performance: <100ns per lookup (DashMap sharded locking)
static STORE_CACHE: Lazy<DashMap<StoreKey, Arc<StoreSlot>>> = Lazy::new(DashMap::new);

/// Test hook: the URI of every store `get_or_create_store` built (not served from the
/// cache), in order.  Lets a test count the builds for one URI.
#[cfg(test)]
static STORE_BUILD_LOG: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Test hook: the URI and the BUILDING THREAD of every store build, in order (the same builds
/// as [`STORE_BUILD_LOG`]). Lets a test prove a store is built on the calling thread (#173).
#[cfg(test)]
static STORE_BUILD_THREADS: std::sync::Mutex<Vec<(String, std::thread::ThreadId)>> =
    std::sync::Mutex::new(Vec::new());

/// Get or create a cached ObjectStore instance.
///
/// store_for_uri_with_logger() is synchronous — no runtime needed for creation.
/// The store is cached by (scheme, endpoint, region, per-endpoint route), for a
/// cache hit rate above 99%; see [`StoreKey`]. Every Python entry point that needs
/// a store, sync or async, gets it here (issue #204), so a host-form server's S3
/// client and connection pool are built once and shared by all calls to it.
///
/// A cache miss is built ONCE per key however many callers race on it: the
/// others wait for the builder (see [`StoreSlot`]).
///
/// The cache holds only the UNLOGGED store, never a logger. `logger` is the
/// caller's snapshot of `global_logger()` taken for THIS call, and a call that
/// has one gets the cached store wrapped in a `LoggedObjectStore` for that
/// logger. So whether a call is recorded depends only on whether the op-log is
/// active when the call is made, never on which call populated the cache first
/// or on a logger that has since been finalized (issue #174).
///
/// Why this is race-free: there is no logger state in the cache to invalidate,
/// so `init_op_log` / `finalize_op_log` have nothing to clear and no window in
/// which a concurrent call can pick up a stale or half-updated entry. A call
/// racing with init or finalize is either wholly unlogged or wholly logged on
/// the logger it snapshotted; if that logger has been finalized meanwhile its
/// entry is refused and counted (`rejected_closed`), as for any other store.
/// The wrapper is one `Arc` allocation per call while the op-log is active and
/// nothing while it is not.
fn get_or_create_store(
    uri: &str,
    logger: Option<crate::s3_logger::Logger>,
) -> anyhow::Result<Arc<dyn ObjectStore>> {
    // Routing errors propagate BEFORE the cache is consulted or populated.
    let key = StoreKey::from_uri(uri)?;

    // Fast path: the slot exists (cache hit >99% in typical workloads).
    let existing = STORE_CACHE.get(&key).map(|slot| Arc::clone(slot.value()));
    let slot = match existing {
        Some(slot) => slot,
        None => Arc::clone(
            STORE_CACHE
                .entry(key.clone())
                .or_insert_with(|| Arc::new(StoreSlot::new()))
                .value(),
        ),
    };
    let store = slot.get_or_build(uri)?;

    Ok(match logger {
        Some(logger) => Arc::new(
            crate::object_store_logger::LoggedObjectStore::with_container_prefix(
                store,
                logger,
                crate::object_store::container_uri_prefix(uri),
            ),
        ),
        None => store,
    })
}

/// `get_many` for `az://` and `gs://`: read every URI through `get_many_with_stores`, with the
/// stores taken from the process-wide cache ([`get_or_create_store`], issue #204: one store,
/// and so one Azure/GCS client and connection pool, per [`StoreKey`]) and, when `logger` is
/// `Some`, wrapped to write op-log rows exactly as `get()` does (issue #174).
///
/// `logger` is the caller's snapshot of `global_logger()`, taken on the calling thread
/// BEFORE `submit_io` like every other entry point (#173/#174). The factory below runs on
/// global-runtime workers, so it must never reach `run_on_global_rt`: `get_or_create_store`
/// does that only for host-form `s3://` URIs, which are not az/gs.
async fn get_many_cached_stores(
    uris: Vec<String>,
    max_in_flight: usize,
    logger: Option<crate::s3_logger::Logger>,
) -> anyhow::Result<Vec<(String, Bytes)>> {
    get_many_with_stores(uris, max_in_flight, move |uri| {
        get_or_create_store(uri, logger.clone())
    })
    .await
}

/// Submit async work to the global runtime (io_uring pattern).
///
/// Uses run_on_global_rt from s3_client.rs which:
/// - Spawns the future onto the dedicated runtime thread pool
/// - Returns the result via std::sync::mpsc channel
/// - NEVER calls block_on() — works from any thread context
/// - Zero-copy: Bytes (Arc) moves through the channel, no data copied
fn submit_io<F, T>(fut: F) -> PyResult<T>
where
    F: std::future::Future<Output = anyhow::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    run_on_global_rt(fut).map_err(py_err)
}

// ---------------------------------------------------------------------------
// PUT with HEAD verification and automatic retry (all backends)
// ---------------------------------------------------------------------------

/// Parse `S3DLIO_PUT_MAX_RETRIES` per its documented contract: "non-numeric
/// or < 1 values are silently treated as 3 (the default)".
///
/// audit #157 bug 7.4 (F2): the previous inline parse was
/// `.and_then(|v| v.parse().ok()).unwrap_or(3)`, which only caught
/// non-numeric values. `0` parses as a perfectly valid `u32`, so it sailed
/// through untouched -- and `for attempt in 1..=0` never iterates, so
/// `put_bytes`/`put_bytes_async` returned the placeholder "no attempts
/// made" error without ever issuing a PUT, even against a healthy backend.
/// Extracted as a pure function so the clamp is directly unit-testable
/// without needing a live or mocked `ObjectStore`.
fn parse_put_max_retries(raw: Option<&str>) -> u32 {
    raw.and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(3)
}

/// PUT `data` to `uri` via `store`.
///
/// By default this is a single PUT — the same cost as every other S3 client
/// library (no extra round-trip), relying on `S3DLIO_MAX_RETRY_ATTEMPTS` (SDK
/// layer) to ride out transient network failures.
///
/// Set `S3DLIO_PUT_VERIFY=true` to opt into HEAD-after-PUT verification: after
/// the PUT, a HEAD request confirms the stored byte count matches what was
/// sent, and retries up to `S3DLIO_PUT_MAX_RETRIES` times (default 3) on
/// mismatch or transient PUT failure.  On size mismatch the truncated object
/// is deleted before retrying, mirroring the same cleanup logic used by the
/// multipart coordinator_task.  This protects against backends that return a
/// successful PUT response for an object that is not actually fully/durably
/// stored (see mlcommons/storage#593) — at the cost of one extra round-trip
/// per object, so it is opt-in rather than the default.
///
/// **Always skipped for `file://` and `direct://`**: local writes have
/// OS-level durability guarantees so the extra round-trip is never useful.
///
/// Env vars (read at call time, no caching needed):
///   `S3DLIO_PUT_VERIFY`         — enable HEAD verification + retry, default false
///   `S3DLIO_PUT_MAX_RETRIES`    — total attempts (only used when verify is on), default 3
///   `S3DLIO_PUT_RETRY_DELAY_MS` — delay between attempts in ms (only used when verify is on), default 1000
async fn put_verified_with_retry(
    store: &dyn ObjectStore,
    uri: &str,
    data: Bytes,
) -> anyhow::Result<()> {
    // Local stores have strong write durability — skip the extra HEAD call.
    let scheme = uri.split("://").next().unwrap_or("");
    if scheme == "file" || scheme == "direct" {
        return store.put(uri, data).await;
    }

    let verify_enabled = std::env::var("S3DLIO_PUT_VERIFY")
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    if !verify_enabled {
        // Default path: single PUT, no HEAD — matches the cost of every other
        // S3 client library. Transient failures are covered by the SDK's own
        // retry layer (S3DLIO_MAX_RETRY_ATTEMPTS).
        return store.put(uri, data).await;
    }

    let expected_len = data.len() as u64;
    let max_attempts: u32 =
        parse_put_max_retries(std::env::var("S3DLIO_PUT_MAX_RETRIES").ok().as_deref());
    let retry_delay_ms: u64 = std::env::var("S3DLIO_PUT_RETRY_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000);

    let mut last_err = anyhow::anyhow!("no attempts made");

    for attempt in 1..=max_attempts {
        debug!(
            "s3dlio put_bytes: attempt={}/{} uri={} bytes={}",
            attempt, max_attempts, uri, expected_len
        );

        // PUT — Bytes is Arc-backed; clone is a ref-count bump, not a data copy.
        if let Err(e) = store.put(uri, data.clone()).await {
            last_err = e;
            warn!(
                "s3dlio put_bytes: PUT attempt {}/{} failed for {}: {} — {}",
                attempt,
                max_attempts,
                uri,
                last_err,
                if attempt < max_attempts {
                    "retrying"
                } else {
                    "giving up"
                }
            );
            if attempt < max_attempts {
                tokio::time::sleep(std::time::Duration::from_millis(retry_delay_ms)).await;
            }
            continue;
        }

        // HEAD verification — works on every backend (S3, Azure, GCS, file, direct).
        match store.stat(uri).await {
            Err(e) => {
                // Stat failed after a successful PUT — warn but return Ok rather
                // than deleting an object that may be perfectly intact.
                warn!(
                    "s3dlio put_bytes: stat failed for {} after successful PUT: {} — assuming data intact",
                    uri, e
                );
                return Ok(());
            }
            Ok(meta) => {
                if meta.size == expected_len {
                    debug!(
                        "s3dlio put_bytes: verified {} bytes stored at {}",
                        expected_len, uri
                    );
                    return Ok(());
                }

                // Truncated object — mirrors multipart coordinator_task cleanup.
                warn!(
                    "s3dlio put_bytes: size mismatch at {} — wrote {} bytes but HEAD reports {} bytes; \
                     set RUST_LOG=s3dlio=debug for per-attempt trace",
                    uri, expected_len, meta.size
                );
                let cleanup_ok = store.delete(uri).await.is_ok();
                if cleanup_ok {
                    warn!(
                        "s3dlio put_bytes: deleted truncated object at {} — {}",
                        uri,
                        if attempt < max_attempts {
                            "will retry"
                        } else {
                            "caller must retry upload"
                        }
                    );
                } else {
                    warn!(
                        "s3dlio put_bytes: failed to delete truncated object at {} — delete it manually before retrying",
                        uri
                    );
                }
                last_err = anyhow::anyhow!(
                    "PUT size mismatch for {}: wrote {} bytes but HEAD reports {} bytes. {}",
                    uri,
                    expected_len,
                    meta.size,
                    if cleanup_ok {
                        "Truncated object has been deleted."
                    } else {
                        "WARNING: truncated object could not be deleted — remove it manually before retrying."
                    }
                );
                if attempt < max_attempts {
                    tokio::time::sleep(std::time::Duration::from_millis(retry_delay_ms)).await;
                }
            }
        }
    }

    Err(last_err)
}

// ---------------------------------------------------------------------------
// Zero-Copy Buffer Support
// ---------------------------------------------------------------------------

/// A Python-visible wrapper around Rust Bytes that exposes buffer protocol
/// This allows Python code to get a memoryview without copying data
///
/// Implements the Python buffer protocol via __getbuffer__ and __releasebuffer__
/// so that `memoryview(bytes_view)` works directly with zero-copy access.
#[pyclass(name = "BytesView")]
pub struct PyBytesView {
    /// The underlying Bytes (reference-counted, cheap to clone).
    /// `pub` so sibling modules (e.g. python_advanced_api) can Arc-clone with zero data copy.
    pub bytes: Bytes,
    /// Cached length stored in the heap object for a stable shape[0] pointer.
    /// The Python buffer protocol requires shape/strides pointers to remain
    /// valid for the full lifetime of the Py_buffer view.  Storing `len_isize`
    /// here (in the heap-allocated PyObject) guarantees stability: CPython
    /// keeps this object alive (via Py_INCREF on view.obj) until the last
    /// memoryview referencing it is garbage-collected.
    shape_len: isize,
}

#[pymethods]
impl PyBytesView {
    /// Get the length of the data
    fn __len__(&self) -> usize {
        self.bytes.len()
    }

    /// Support bytes() conversion - returns a copy
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }

    /// Implement Python buffer protocol for zero-copy access.
    ///
    /// When Python evaluates `memoryview(bytes_view_instance)` it calls
    /// `__getbuffer__` on this object.  We fill in the `Py_buffer` descriptor
    /// so that the resulting memoryview points directly at the Arc-backed
    /// `Bytes` buffer with **no data copy**.
    ///
    /// Lifecycle safety:
    /// - `(*view).obj` is set to this Python object and `Py_INCREF`'d, so
    ///   CPython keeps the `PyBytesView` (and thus the `Bytes` Arc) alive for
    ///   as long as any consumer holds the memoryview.
    /// - `shape` points to `slf.shape_len` which lives in the heap-allocated
    ///   PyObject — stable for the object's lifetime.
    /// - `strides` uses a `'static` unit stride (byte arrays always have
    ///   stride = 1).
    ///
    /// The buffer is read-only; requesting a writable buffer will raise
    /// `BufferError`.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if (flags & ffi::PyBUF_WRITABLE) != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "BytesView is read-only and does not support writable buffers",
            ));
        }

        // Unit stride: 1 byte per element — constant, 'static lifetime.
        static UNIT_STRIDE: isize = 1;

        let bytes = &slf.bytes;
        unsafe {
            (*view).buf = bytes.as_ptr() as *mut std::os::raw::c_void;
            (*view).len = bytes.len() as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).ndim = 1;

            // Format: "B" = unsigned byte.
            (*view).format = if (flags & ffi::PyBUF_FORMAT) != 0 {
                c"B".as_ptr() as *mut std::os::raw::c_char
            } else {
                std::ptr::null_mut()
            };

            // shape[0] = number of elements.  Points into the PyBytesView
            // heap object (not into the Py_buffer struct) for lifetime safety.
            (*view).shape = if (flags & ffi::PyBUF_ND) != 0 {
                &slf.shape_len as *const isize as *mut isize
            } else {
                std::ptr::null_mut()
            };

            // strides[0] = 1 (contiguous byte array).  'static pointer.
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) != 0 {
                &UNIT_STRIDE as *const isize as *mut isize
            } else {
                std::ptr::null_mut()
            };

            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();

            // Keep this PyBytesView alive for the duration of the buffer view.
            (*view).obj = slf.as_ptr();
            ffi::Py_INCREF((*view).obj);
        }
        Ok(())
    }

    /// Release the buffer.
    ///
    /// PyO3 / CPython will `Py_DECREF(view.obj)` after this call, decrementing
    /// the reference count on this `PyBytesView`.  When the count reaches zero
    /// the `PyBytesView` is dropped, which decrements the `Bytes` Arc.
    /// Nothing additional to do here.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    /// Return a **zero-copy** read-only `memoryview` over the underlying buffer.
    ///
    /// Uses `PyMemoryView_FromObject`, which routes through the buffer protocol
    /// (`__getbuffer__` above) and correctly increments the reference count on
    /// this `PyBytesView` object.  The resulting memoryview keeps the `Bytes`
    /// Arc alive for as long as any Python consumer holds it.
    ///
    /// The previous implementation used `PyMemoryView_FromMemory`, which does
    /// **not** link the memoryview's lifetime to this object — a potential
    /// use-after-free.  This implementation is safe.
    ///
    /// # Python usage
    /// ```python
    /// bv  = s3dlio.get("gs://bucket/obj")   # returns BytesView
    /// mv  = bv.memoryview()                  # zero-copy memoryview
    /// arr = np.frombuffer(mv, dtype=np.uint8) # numpy reads directly — no copy
    /// t   = torch.frombuffer(mv, dtype=torch.uint8) # torch too
    /// ```
    fn memoryview<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        // PyMemoryView_FromObject calls PyObject_GetBuffer → our __getbuffer__.
        // CPython will Py_INCREF(view.obj = slf) keeping the Arc alive.
        let raw = unsafe { ffi::PyMemoryView_FromObject(slf.as_ptr()) };
        if raw.is_null() {
            Err(PyErr::fetch(py))
        } else {
            Ok(unsafe { Bound::from_owned_ptr(py, raw) })
        }
    }

    /// Convert to Python bytes (copy)
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }

    /// Python repr
    fn __repr__(&self) -> String {
        format!("BytesView({} bytes)", self.bytes.len())
    }
}

impl PyBytesView {
    /// Create from Rust `Bytes`.
    ///
    /// `shape_len` is cached here (not in the `Py_buffer` struct) so that the
    /// pointer we hand to Python's buffer protocol remains stable for the full
    /// lifetime of any memoryview that references this object.
    pub fn new(bytes: Bytes) -> Self {
        let shape_len = bytes.len() as isize;
        Self { bytes, shape_len }
    }
}

/// Format an error for the Python-visible message, preserving the full
/// `anyhow` cause chain rather than just the outermost context label.
///
/// Plain `Display` (`{}`) on an `anyhow::Error` — or anything transparently
/// wrapping one, like `DatasetError::Backend` — shows only the outermost
/// `.context(...)` message. The alternate-flag `{:#}` walks the full chain
/// (storage#755 / s3dlio#161: a chained SDK/timeout/TLS error was reduced to
/// the bare label "concurrent range chunk failed" by the time it reached
/// Python).
pub(crate) fn error_chain_message<E: std::fmt::Display>(error: &E) -> String {
    format!("{:#}", error)
}

pub fn py_err<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error_chain_message(&error))
}

/// Issue #211: reject a caller-supplied concurrency limit of 0 at once with a
/// `ValueError` that names the parameter and the valid range.  A limit of 0
/// would build a semaphore with no permits and hang the call forever.  Call
/// this FIRST in every entry point that takes such a limit; it never
/// substitutes another value.
pub(crate) fn require_max_in_flight(name: &str, value: usize) -> PyResult<()> {
    crate::concurrency::require_concurrency_at_least_one(name, value)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Multi-process GET for maximum throughput (Python API)
///
/// Args:
///     uri (str): S3 URI prefix (e.g., "s3://bucket/prefix/")
///     procs (int): Number of worker processes to spawn (default: 4)
///     jobs (int): Concurrent operations per worker process (default: 64)  
///     num (int): Number of objects to download (for benchmarking, default: 1000)
///     template (str): Object name template with {} placeholder (default: "object_{}.dat")
///
/// Returns:
///     dict: Performance statistics with keys:
///         - total_objects (int): Total objects processed
///         - total_bytes (int): Total bytes transferred  
///         - duration_seconds (float): Time taken
///         - throughput_mb_s (float): Overall throughput in MB/s
///         - ops_per_sec (float): Operations per second
///         - workers (list): Per-worker performance stats
/// Find the s3-cli binary for use as worker processes
fn find_s3_cli_binary() -> Result<String, anyhow::Error> {
    use std::path::Path;

    // Try common development locations
    let candidates = vec![
        "./target/release/s3-cli",
        "./target/debug/s3-cli",
        "../target/release/s3-cli",
        "../target/debug/s3-cli",
    ];

    for candidate in candidates {
        if Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }

    // Try finding s3-cli in PATH
    if let Ok(output) = std::process::Command::new("which").arg("s3-cli").output() {
        if output.status.success() {
            let path = String::from_utf8(output.stdout)?;
            let path = path.trim();
            if !path.is_empty() {
                return Ok(path.to_string());
            }
        }
    }

    anyhow::bail!("Could not find s3-cli binary. Please build it with 'cargo build --bin s3-cli' or ensure it's in PATH")
}

#[pyfunction]
#[pyo3(signature = (uri, procs = 4, jobs = 64, num = 1000, template = "object_{}.dat"))]
pub fn mp_get(
    uri: &str,
    procs: usize,
    jobs: usize,
    num: usize,
    template: &str,
) -> PyResult<Py<PyAny>> {
    require_max_in_flight("procs", procs)?;
    require_max_in_flight("jobs", jobs)?;
    use crate::mp::{run_get_shards, MpGetConfigBuilder};
    use std::io::Write;
    use tempfile::NamedTempFile;

    Python::attach(|py| {
        // Parse S3 URI
        let (bucket, key_prefix) = parse_s3_uri_for_global_client(uri).map_err(py_err)?;

        // Generate object keys based on template and number
        let mut keys = Vec::new();
        for i in 0..num {
            let key = if template.contains("{}") {
                template.replace("{}", &i.to_string())
            } else {
                format!("{}{}", template, i)
            };
            keys.push(format!("s3://{}/{}{}", bucket, key_prefix, key));
        }

        // Create temporary keylist file
        let mut keylist_file = NamedTempFile::new().map_err(py_err)?;
        for key in &keys {
            writeln!(keylist_file, "{}", key).map_err(py_err)?;
        }
        let keylist_path = keylist_file.path().to_path_buf();

        // Find the s3-cli binary for worker processes
        let s3_cli_path = find_s3_cli_binary().map_err(py_err)?;

        // Configure multi-process GET
        let config = MpGetConfigBuilder::new()
            .procs(procs)
            .concurrent_per_proc(jobs)
            .keylist(keylist_path)
            .worker_cmd(s3_cli_path)
            .build();

        // Run the multi-process operation
        let result = run_get_shards(&config).map_err(py_err)?;

        // Build Python dictionary result
        let dict = PyDict::new(py);
        // New schema
        dict.set_item("attempted", result.attempted)?;
        dict.set_item("succeeded", result.succeeded)?;
        dict.set_item("failed", result.failed)?;
        dict.set_item("total_bytes", result.total_bytes)?;
        dict.set_item("bytes_source", format!("{:?}", result.bytes_source))?;
        dict.set_item("duration_seconds", result.elapsed_seconds)?;
        dict.set_item("throughput_mb_s", result.throughput_mbps())?;
        dict.set_item("ops_per_sec", result.ops_per_second())?;

        // Backward compatibility key: historically this represented successful
        // operations in mp_get summaries.
        dict.set_item("total_objects", result.succeeded)?;

        // Add per-worker stats
        let workers = PyList::empty(py);
        for worker in &result.per_worker {
            let worker_dict = PyDict::new(py);
            worker_dict.set_item("worker_id", worker.worker_id)?;
            worker_dict.set_item("attempted", worker.attempted)?;
            worker_dict.set_item("succeeded", worker.succeeded)?;
            worker_dict.set_item("failed", worker.failed)?;
            worker_dict.set_item("bytes", worker.bytes)?;
            // Backward compatibility key
            worker_dict.set_item("objects", worker.succeeded)?;

            let worker_throughput = if result.elapsed_seconds > 0.0 {
                worker
                    .bytes
                    .map(|bytes| (bytes as f64 / (1024.0 * 1024.0)) / result.elapsed_seconds)
            } else {
                None
            };
            worker_dict.set_item("throughput_mb_s", worker_throughput)?;
            workers.append(worker_dict)?;
        }
        dict.set_item("workers", workers)?;

        dict.into_py_any(py)
    })
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------
#[pyfunction]
pub fn init_logging(level: &str) -> PyResult<()> {
    let filter = match level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    // Initialize tracing subscriber with the specified level
    let _ = tracing_subscriber::fmt()
        .with_max_level(filter)
        .with_target(false)
        .try_init();

    // NOTE: tracing_log::LogTracer::init() intentionally omitted here.
    // The log→tracing bridge causes deadlocks during async OnceCell
    // initialization when debug/trace logging is active (AWS SDK uses
    // the `log` crate internally, and the bridge re-enters tracing).
    // See: https://github.com/russfellows/s3dlio/issues/105

    Ok(())
}

/// Enable operation logging to a local zstd-compressed TSV file.
///
/// One op-log session runs at a time. After finalize_op_log() a new session can be
/// started with another init_op_log() call (issue #294), any number of times.
///
/// Args:
///     path: Local filesystem path (e.g. "/tmp/ops.tsv.zst") or a file://
///         destination (e.g. "file:///tmp/ops.tsv.zst"). The parent directory
///         must exist. file:// follows s3dlio's file:// convention (the same
///         as the file:// object store); it is not full RFC 8089: only the
///         lowercase prefix "file://" is recognised, it is removed, and the
///         rest is the path, used literally ("file://localhost/x" is the
///         relative path "localhost/x"; "%20" is not decoded; "file://" alone
///         is an error). Any other scheme, including "FILE://", "direct://"
///         and s3://, is rejected as unsupported.
///
/// Behaviour:
///     - First call: creates the file and starts logging. A call that fails
///       (bad path, unsupported scheme) leaves logging uninitialized, so a
///       corrected call can succeed.
///     - Call while logging is already active with the SAME file (any
///       spelling: "x", "./x", "/abs/x", "file:///abs/x", "dir/../x", a symlink
///       or hard link to it): returns without effect; nothing is created or
///       truncated and the session continues.
///     - Call while logging is already active with a DIFFERENT file: raises
///       RuntimeError naming the file being written and the requested path and
///       saying to call finalize_op_log() first. The requested path is not
///       created or truncated and the session carries on unaffected. A path
///       that cannot be used (unsupported scheme, empty "file://") raises too.
///       Exactly one of several threads calling at once wins (same file: all
///       return, one creates it; different files: the others raise).
///     - In a child process created by os.fork() the parent's session does not
///       exist (its writer thread is not copied): is_op_log_active() is False
///       there and the child may start its OWN session, but not on the
///       file an ancestor is still writing (RuntimeError: it would truncate a
///       live log); a log the parent had already finalized may be initialised.
///     - Call after finalize_op_log(): starts a NEW session. If the previous
///       session's writer is still draining (finalize running on another
///       thread), it waits for that first. The file is created like a first
///       init (create-and-truncate: the same path REPLACES the previous log, a
///       different path leaves it untouched); op_log_stats() restarts from zero
///       for the new session and the process client ID carries over. A failed
///       call leaves logging finalized, so a corrected call can succeed.
///
/// The module-level functions (get, put_bytes, stat, list, delete, get_async,
/// ...) look the op-log up on every call: a call made while logging is active is
/// recorded once, and a call made before init_op_log() or after
/// finalize_op_log() is not, whichever call first used a given scheme. Objects
/// you construct yourself (stores, readers, writers) capture the logger when
/// they are constructed: those created before init_op_log() are never logged;
/// those created before finalize_op_log() stop recording when it runs, and stay
/// silent after a later init_op_log() (they never write into the new session).
#[pyfunction]
pub fn init_op_log(py: Python<'_>, path: &str) -> PyResult<()> {
    // A re-init may wait for the previous session to drain (#294): release the GIL
    // meanwhile, as finalize_op_log does.
    py.detach(|| init_op_logger(path)).map_err(py_err)
}

/// Finish operation logging: write every accepted record (unless the writer
/// fails: an I/O error or a panic in the writer thread, see op_log_stats()),
/// close the zstd stream and wait until the writer has finished and signalled
/// completion. Call it before the process exits: nothing does it for you (no
/// atexit hook), and without it queued records and the zstd frame end are lost. It
/// ends the current session; init_op_log() can start a new one (see init_op_log).
/// Safe to call repeatedly, or before init_op_log() (no effect).
/// After it returns is_op_log_active() is False and operations on objects
/// created earlier are no longer recorded. The GIL is released while the
/// pending records are drained, so other Python threads keep running.
#[pyfunction]
pub fn finalize_op_log(py: Python<'_>) -> PyResult<()> {
    py.detach(finalize_op_logger);
    Ok(())
}

/// True only between a successful init_op_log() and the next finalize_op_log()
/// (and again after a re-init).
#[pyfunction]
pub fn is_op_log_active() -> PyResult<bool> {
    Ok(global_logger().is_some())
}

/// Counts of what happened to the entries submitted to the op-log between
/// init_op_log() and finalize_op_log(), so you can detect most ways records are
/// lost.
///
/// The counts are those of the CURRENT session: a new init_op_log() after
/// finalize_op_log() starts again from zero (read them before re-initialising).
///
/// Returns None before the first init_op_log(); otherwise a dict (available while
/// logging is active AND after finalize_op_log()):
///
///     accepted         entries queued for the writer
///     written          rows the writer handed to the compressor without error
///                      (an upper bound of the rows in the file)
///     dropped_full     entries dropped because the queue was full (default
///                      mode only)
///     rejected_closed  calls that arrived after finalize began (not in the file)
///     writer_failed    calls refused because the writer had already stopped
///     writer_error     None, or the first error the writer met (a failed write,
///                      a failed finish/flush of the compressed stream, or a
///                      writer panic, which is a defect, not an I/O error) as text
///     lossless         True if S3DLIO_OPLOG_LOSSLESS=1 (callers wait, never drop)
///     rejected_unformattable
///                      accepted rows the writer did not write because a
///                      timestamp (start, first byte or end, after the clock
///                      offset is applied) is before 1970-01-01 or after
///                      9999-12-31 and cannot be written as RFC 3339. The row is
///                      absent from the file (its idx is skipped), later rows are
///                      written, and the trace is incomplete.
///     rejected_unsafe_field
///                      accepted records the writer did not write because the
///                      operation name, client id, endpoint or file holds a TAB,
///                      LF or CR, which the tab-separated row cannot carry. The
///                      whole record is absent from the file (its idx is skipped),
///                      later records are written, and the trace is incomplete.
///                      The record is lost, not encoded.
///     normalized_error_text
///                      written rows whose error text held a TAB, LF or CR, each
///                      replaced by one space. The operation is recorded, so this
///                      does NOT make the log incomplete; it discloses a changed
///                      message.
///
/// After finalize_op_log() no counted loss happened exactly when
/// accepted == written and dropped_full == 0 and writer_failed == 0 and
/// writer_error is None and rejected_unformattable == 0 and
/// rejected_unsafe_field == 0. "written" counts rows the zstd compressor
/// accepted; the bytes may still be buffered when it is counted, and a later
/// write, finish or flush can still fail (that is writer_error). A successful finalize_op_log()
/// with that condition true means every accepted row reached the compressor and
/// the stream was finished and flushed to the file without error. That is
/// completion, not durability: there is no fsync, and a close(2) error cannot be
/// observed. (accepted > written means entries were queued but never written,
/// e.g. the disk filled up; writer_failed alone cannot show that.)
/// rejected_closed is not part of "complete" -- those calls were never going to
/// be in the file -- but finalize only after your worker threads have stopped, and
/// check rejected_closed == 0 as well. While logging is active accepted > written
/// is normal (the queue is draining), so judge completeness after finalize.
///
/// NOT covered: process exit without finalize_op_log() (no atexit hook: queued
/// records and the zstd frame end are lost and these stats are never read);
/// objects created before init_op_log() (unlogged, uncounted); errors reported
/// only when the file is closed (e.g. NFS or quota errors at close(2)).
///
/// To avoid queue-full drops set S3DLIO_OPLOG_LOSSLESS=1 and/or raise
/// S3DLIO_OPLOG_BUF. finalize_op_log() also prints one warning on stderr if
/// anything was lost, stating the exact number of queue-full drops. In the
/// default mode the first queue-full drop of the session prints one stderr note
/// at once; later drops are only counted.
#[pyfunction]
#[pyo3(name = "op_log_stats")]
pub fn op_log_stats_py(py: Python<'_>) -> PyResult<Option<Bound<'_, PyDict>>> {
    let Some(stats) = op_log_stats() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("accepted", stats.accepted)?;
    dict.set_item("dropped_full", stats.dropped_full)?;
    dict.set_item("rejected_closed", stats.rejected_closed)?;
    dict.set_item("writer_failed", stats.writer_failed)?;
    dict.set_item("lossless", stats.lossless)?;
    dict.set_item("written", stats.written)?;
    dict.set_item("writer_error", stats.writer_error)?;
    dict.set_item("rejected_unformattable", stats.rejected_unformattable)?;
    dict.set_item("rejected_unsafe_field", stats.rejected_unsafe_field)?;
    dict.set_item("normalized_error_text", stats.normalized_error_text)?;
    Ok(Some(dict))
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
fn build_uri_list(prefix: &str, template: &str, num: usize) -> PyResult<Vec<String>> {
    // Extract scheme and parse URI generically (works for s3://, az://, file://, etc.)
    let scheme_end = prefix
        .find("://")
        .ok_or_else(|| anyhow::anyhow!("URI must contain scheme (e.g., s3://, az://, file://)"))
        .map_err(py_err)?;

    let scheme = &prefix[..scheme_end + 3]; // Include "://"
    let remainder = &prefix[scheme_end + 3..];

    // Split into bucket/container and key prefix
    let (bucket, mut key_prefix) = if let Some(slash_pos) = remainder.find('/') {
        (
            remainder[..slash_pos].to_string(),
            remainder[slash_pos + 1..].to_string(),
        )
    } else {
        (remainder.to_string(), String::new())
    };

    if !key_prefix.is_empty() && !key_prefix.ends_with('/') {
        key_prefix.push('/');
    }

    let uris = (0..num)
        .map(|i| {
            let name =
                template
                    .replacen("{}", &i.to_string(), 1)
                    .replacen("{}", &num.to_string(), 1);
            format!("{}{}/{}{}", scheme, bucket, key_prefix, name)
        })
        .collect();
    Ok(uris)
}

/// Create the bucket a `put(prefix, ..., should_create_bucket=True)` call writes
/// into: for an `s3://` prefix, the bucket it names on the server it names
/// (`s3://host:port/bkt/pfx/` is bucket `bkt` on `host:port`, issue #199). Other
/// schemes have no S3 bucket to create.
///
/// `Err` only for a prefix that cannot name a bucket (a portless IP address, the
/// #178 error, or no bucket at all); a failed *creation* (the bucket may exist, or
/// the credentials may only allow object writes) is logged and the put proceeds.
fn create_put_bucket(prefix: &str) -> anyhow::Result<()> {
    if !prefix.starts_with("s3://") {
        return Ok(());
    }
    let target = s3_bucket_target(prefix)?;
    if let Err(e) = create_bucket_for_uri(prefix) {
        warn!("Failed to create bucket {}: {}", target.bucket, e);
    }
    Ok(())
}

fn str_to_obj(s: &str) -> ObjectType {
    ObjectType::from(s)
}

fn str_to_data_gen_algorithm(s: &str) -> DataGenAlgorithm {
    match s.to_lowercase().as_str() {
        "random" => DataGenAlgorithm::Random,
        _ => DataGenAlgorithm::Random, // Default to random (prand deprecated)
    }
}

fn str_to_data_gen_mode(s: &str) -> DataGenMode {
    match s.to_lowercase().as_str() {
        "streaming" => DataGenMode::Streaming,
        "single-pass" | "single_pass" | "singlepass" => DataGenMode::SinglePass,
        _ => DataGenMode::Streaming, // Default to streaming
    }
}

// ------------------------------------------------------------------------
// URI Parsing utilities
// ------------------------------------------------------------------------

/// Parse S3 URI with optional endpoint support
///
/// Returns a dictionary with keys: 'endpoint' (optional), 'bucket', 'key'
///
/// Examples:
///     >>> parse_s3_uri_full("s3://mybucket/data.bin")
///     {'endpoint': None, 'bucket': 'mybucket', 'key': 'data.bin'}
///     
///     >>> parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")
///     {'endpoint': '192.168.100.1:9001', 'bucket': 'mybucket', 'key': 'data.bin'}
#[pyfunction]
fn parse_s3_uri_full(py: Python<'_>, uri: &str) -> PyResult<Py<PyAny>> {
    use crate::s3_utils::parse_s3_uri_full as parse_full;

    let components = parse_full(uri).map_err(py_err)?;

    // Create Python dictionary
    let dict = pyo3::types::PyDict::new(py);

    // Add endpoint (None if not present)
    if let Some(endpoint) = components.endpoint {
        dict.set_item("endpoint", endpoint)?;
    } else {
        dict.set_item("endpoint", py.None())?;
    }

    dict.set_item("bucket", components.bucket)?;
    dict.set_item("key", components.key)?;

    Ok(dict.into())
}

// - create-bucket command
#[pyfunction]
pub fn create_bucket(py: Python<'_>, bucket_name: &str) -> PyResult<()> {
    // `bkt`, `s3://bkt`, `s3://bkt/` -> bucket `bkt` on the global endpoint;
    // `s3://host:port/bkt` -> bucket `bkt` on that server (issue #199).
    let bucket_name = bucket_name.to_string(); // Own it to move across the thread boundary

    // Release the GIL to allow other Python threads to run
    py.detach(move || create_bucket_for_uri(&bucket_name).map_err(py_err))
}

// - delete-bucket command
#[pyfunction]
pub fn delete_bucket(py: Python<'_>, bucket_name: &str) -> PyResult<()> {
    // Same URI forms as `create_bucket` (issue #199).
    let bucket_name = bucket_name.to_string(); // Own it to move across the thread boundary

    // Release the GIL for the blocking S3 call
    py.detach(move || delete_bucket_for_uri(&bucket_name).map_err(py_err))
}

// ---------------------------------------------------------------------------
// S3 helpers (sync + async)
// ---------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (
    prefix, num, template = None,
    max_in_flight = 64, size = None,
    should_create_bucket = false, object_type = "zeros",
    dedup_factor = 1, compress_factor = 1,
    data_gen_algorithm = "random", data_gen_mode = "streaming", chunk_size = 262144
))]
#[allow(clippy::too_many_arguments)]
pub fn put(
    py: Python<'_>,
    prefix: &str,
    num: usize,
    template: Option<&str>,
    max_in_flight: usize,
    size: Option<usize>,
    should_create_bucket: bool,
    object_type: &str,
    dedup_factor: usize,
    compress_factor: usize,
    data_gen_algorithm: &str,
    data_gen_mode: &str,
    chunk_size: usize,
) -> PyResult<()> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let template = template.unwrap_or("object-{}");
    let uris = build_uri_list(prefix, template, num)?;
    // `num == 0` uploads nothing; keep the derived job count >= 1 so the
    // (already validated) limit is not mistaken for a caller-supplied 0.
    let jobs = max_in_flight.min(num.max(1));
    let obj = str_to_obj(object_type);
    let algorithm = str_to_data_gen_algorithm(data_gen_algorithm);
    let mode = str_to_data_gen_mode(data_gen_mode);
    let config = Config::new_with_defaults(obj, 1, sz, dedup_factor, compress_factor)
        .with_data_gen_algorithm(algorithm)
        .with_data_gen_mode(mode)
        .with_chunk_size(chunk_size);
    py.detach(|| {
        if should_create_bucket {
            create_put_bucket(prefix).map_err(py_err)?;
        }
        put_objects_with_random_data_and_type(&uris, sz, jobs, config).map_err(py_err)
    })
}

#[pyfunction(name = "put_async")]
#[pyo3(signature = (
    prefix, num, template = None,
    max_in_flight = 64, size = None,
    should_create_bucket = false, object_type = "zeros",
    dedup_factor = 1, compress_factor = 1,
    data_gen_algorithm = "random", data_gen_mode = "streaming", chunk_size = 262144
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_async_py<'p>(
    py: Python<'p>,
    prefix: &'p str,
    num: usize,
    template: Option<&'p str>,
    max_in_flight: usize,
    size: Option<usize>,
    should_create_bucket: bool,
    object_type: &'p str,
    dedup_factor: usize,
    compress_factor: usize,
    data_gen_algorithm: &'p str,
    data_gen_mode: &'p str,
    chunk_size: usize,
) -> PyResult<Bound<'p, PyAny>> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let template = template.unwrap_or("object-{}");
    let uris = build_uri_list(prefix, template, num)?;
    // `num == 0` uploads nothing; keep the derived job count >= 1 so the
    // (already validated) limit is not mistaken for a caller-supplied 0.
    let jobs = max_in_flight.min(num.max(1));
    let obj = str_to_obj(object_type);
    let algorithm = str_to_data_gen_algorithm(data_gen_algorithm);
    let mode = str_to_data_gen_mode(data_gen_mode);
    let config = Config::new_with_defaults(obj, 1, sz, dedup_factor, compress_factor)
        .with_data_gen_algorithm(algorithm)
        .with_data_gen_mode(mode)
        .with_chunk_size(chunk_size);

    let prefix = prefix.to_string();
    future_into_py(py, async move {
        task::spawn_blocking(move || {
            if should_create_bucket {
                create_put_bucket(&prefix).map_err(py_err)?;
            }
            put_objects_with_random_data_and_type(&uris, sz, jobs, config).map_err(py_err)
        })
        .await
        .map_err(py_err)??;
        Python::attach(|py| Ok(py.None()))
    })
}

/// Configure S3-compatible connection parameters from Python code.
///
/// This is the Python equivalent of the CLI's `--endpoint-url`, `--region`,
/// and `--ca-bundle` global flags.  It applies to **all** S3 operations:
/// `list`, `get`, `put`, `stat`, `delete`, `upload`, `download`, etc.
///
/// Call this **before the first S3 operation** in your script.  The
/// configuration is process-wide: the underlying AWS SDK client is a
/// global singleton that is initialised on first use and cannot be
/// reconfigured afterwards.  Calling `configure_s3()` after the first
/// S3 call has no effect on the already-initialised client.
///
/// This function also clears the internal store cache so that any
/// previously created store objects are discarded and rebuilt with the
/// new settings on the next operation.
///
/// Parameters
/// ----------
/// endpoint_url : str or None
///     Full URL of the S3-compatible endpoint, e.g. ``"https://minio.corp:9000"``.
///     Sets ``AWS_ENDPOINT_URL``.
/// region : str or None
///     AWS region name, e.g. ``"us-east-1"``.  Sets ``AWS_DEFAULT_REGION``.
/// ca_bundle : str or None
///     Filesystem path to a PEM CA certificate bundle for TLS verification.
///     Sets ``AWS_CA_BUNDLE``.
///
/// Example
/// -------
/// ```python
/// import s3dlio
/// s3dlio.configure_s3(
///     endpoint_url="https://172.16.1.40:9000",
///     region="us-east-1",
///     ca_bundle="/etc/ssl/certs/my-ca.pem",
/// )
/// # Now all operations target the custom endpoint:
/// keys   = s3dlio.list("s3://my-bucket/", recursive=False)
/// data   = s3dlio.get("s3://my-bucket/my-file.bin")
/// s3dlio.put("s3://my-bucket/out.bin", data)
/// info   = s3dlio.stat("s3://my-bucket/my-file.bin")
/// s3dlio.delete("s3://my-bucket/my-file.bin")
/// ```
#[pyfunction]
#[pyo3(signature = (endpoint_url = None, region = None, ca_bundle = None))]
pub fn configure_s3(
    endpoint_url: Option<&str>,
    region: Option<&str>,
    ca_bundle: Option<&str>,
) -> PyResult<()> {
    // The GIL does not make this single-threaded (s3dlio's runtimes and native
    // threads may be running); see the limitations on `set_s3_endpoint_override`.
    // Call `configure_s3` at start-up, before the first S3 operation.
    if let Some(url) = endpoint_url {
        crate::s3_client::set_s3_endpoint_override(url);
    }
    if let Some(r) = region {
        unsafe { std::env::set_var("AWS_DEFAULT_REGION", r) };
    }
    if let Some(bundle) = ca_bundle {
        unsafe { std::env::set_var("AWS_CA_BUNDLE", bundle) };
    }
    // Clear the store cache so subsequent operations create fresh stores
    // with the updated configuration rather than reusing stale ones.
    STORE_CACHE.clear();
    Ok(())
}

// --- `list` function
//
// On s3:// `uri` is never a regular expression and never a bare prefix (issue #213, rule in
// `crate::s3_selection`): `s3://b/K` lists the object `K` (and, recursive, what is under `K/`)
// and NEVER `K.bak`; a trailing `/` lists a directory; `*` / `**` make an explicit glob over the
// full key. `.`, `(`, `[`, `+`, `$` and `?` in it are ordinary characters. `pattern` is the one explicit way to
// select by regular expression; it is matched against each listed entry relative to the
// listing base (`crate::list_pattern`), the same rule `s3-cli ls --pattern` uses.
#[pyfunction]
#[pyo3(signature = (uri, recursive = false, pattern = None))]
pub fn list(
    py: Python<'_>,
    uri: &str,
    recursive: bool,
    pattern: Option<&str>,
) -> PyResult<Vec<String>> {
    // Compile an explicit pattern first: an invalid one fails before any request is made.
    let filter = pattern
        .map(crate::list_pattern::ListPattern::new)
        .transpose()
        .map_err(py_err)?;

    // Universal list using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let uri_owned = uri.to_owned();

    // The GIL is released for the whole blocking part -- store creation (credential
    // loading) and the network listing -- so other Python threads keep running (issue #201).
    let mut keys = py.detach(|| {
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move { store.list(&uri_owned, recursive).await })
    })?;

    // Apply the explicit client-side regex against the path relative to the listing base.
    if let Some(filter) = &filter {
        filter.retain(uri, &mut keys);
    }

    Ok(keys)
}

// New stat calls, almost the same ????
//
#[pyfunction]
pub fn stat(py: Python<'_>, uri: &str) -> PyResult<Py<PyAny>> {
    // Universal stat using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let uri_owned = uri.to_owned();

    // GIL released across store creation and the network request (issue #201).
    let os = py.detach(|| {
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move { store.stat(&uri_owned).await })
    })?;

    let d = stat_to_pydict(py, os)?;
    Ok(d.unbind().into())
}

#[pyfunction]
fn stat_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    // Through the shared store cache (issue #204), built with the GIL released: a
    // miss may create a client (credential loading) on this caller thread.
    let store = py
        .detach(|| get_or_create_store(&uri, global_logger()))
        .map_err(py_err)?;

    future_into_py(py, async move {
        let os = store.stat(&uri).await.map_err(py_err)?;
        Python::attach(|py| {
            let obj: Py<PyAny> = stat_to_pydict(py, os)?.unbind().into();
            Ok::<Py<PyAny>, PyErr>(obj)
        })
    })
}

#[pyfunction]
fn stat_many_async<'py>(py: Python<'py>, uris: Vec<String>) -> PyResult<pyo3::Bound<'py, PyAny>> {
    future_into_py(py, async move {
        let metas = stat_object_many_async(uris).await.map_err(py_err)?;
        Python::attach(|py| {
            let list = PyList::empty(py);
            for os in metas {
                // append accepts any IntoPyObject<'py>; Bound<'py, PyDict> works directly
                list.append(stat_to_pydict(py, os)?)?;
            }
            Ok::<Py<PyAny>, PyErr>(list.unbind().into())
        })
    })
}

/// Helper function to map Rust to Python for stat call
fn stat_to_pydict<'py>(
    py: Python<'py>,
    os: crate::s3_utils::ObjectStat,
) -> PyResult<pyo3::Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("size", os.size)?;
    d.set_item("last_modified", os.last_modified)?;
    d.set_item("etag", os.e_tag)?;
    d.set_item("content_type", os.content_type)?;
    d.set_item("content_language", os.content_language)?;
    d.set_item("content_encoding", os.content_encoding)?;
    d.set_item("cache_control", os.cache_control)?;
    d.set_item("content_disposition", os.content_disposition)?;
    d.set_item("expires", os.expires)?;
    d.set_item("storage_class", os.storage_class)?;
    d.set_item("server_side_encryption", os.server_side_encryption)?;
    d.set_item("ssekms_key_id", os.ssekms_key_id)?;
    d.set_item("sse_customer_algorithm", os.sse_customer_algorithm)?;
    d.set_item("version_id", os.version_id)?;
    d.set_item("replication_status", os.replication_status)?;
    d.set_item("metadata", os.metadata)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// exists() - Check if object exists (DLIO benchmark compatibility)
// ---------------------------------------------------------------------------

/// Check if an object exists at the given URI.
/// Returns True if the object exists, False otherwise.
/// Works with all backends: S3, GCS, Azure, file://, direct://
///
/// **Important**: this function returns `False` both when the object
/// genuinely does not exist *and* when existence could not be determined
/// for any other reason (a malformed URI, invalid/missing credentials, a
/// network partition, a TLS handshake failure, an unreachable endpoint,
/// etc.) — every failure collapses to `False`, by design, the same way
/// Python's own `os.path.exists()` / `pathlib.Path.exists()` swallow
/// `PermissionError`. If you need to tell "definitely not there" apart
/// from "could not check," use [`stat()`] instead — it raises on *any*
/// failure (including not-found), but with the full underlying cause in
/// the exception message, so the caller can distinguish "not found" from
/// a real connectivity/auth/permissions problem by inspecting it.
#[pyfunction]
pub fn exists(py: Python<'_>, uri: &str) -> PyResult<bool> {
    let uri_owned = uri.to_owned();
    py.detach(|| {
        let logger = global_logger();
        // Build the store on the caller thread BEFORE submit_io (issue #173),
        // as `stat`/`list` do: a host-form S3 URI creates its client via
        // `run_on_global_rt`, which must not run on a global-runtime worker.
        let store = match get_or_create_store(&uri_owned, logger) {
            Ok(s) => s,
            Err(_) => return Ok(false), // URI parsing error → doesn't exist
        };

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            match store.stat(&uri_owned).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        })
    })
}

/// Async version of exists()
/// Works with all backends: S3, GCS, Azure, file://, direct://
///
/// Same `False`-on-any-failure caveat as [`exists()`] applies to the
/// `stat()` check itself — see that docstring. Note the one asymmetry
/// between the two: this async version *does* raise on a failure to open
/// the store for the given URI (e.g. an unparseable URI), before it ever
/// gets to the `stat()` call; the sync `exists()` swallows that same
/// failure into `False` too. If you need fully consistent behavior
/// between the sync and async paths, use [`stat()`]/`stat_async()`
/// instead, which raise uniformly on every failure.
#[pyfunction]
fn exists_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    // Through the shared store cache (issue #204); see `stat_async`.
    let store = py
        .detach(|| get_or_create_store(&uri, global_logger()))
        .map_err(py_err)?;

    future_into_py(py, async move {
        match store.stat(&uri).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    })
}

// ---------------------------------------------------------------------------
// put_bytes() - Put from a Python bytes-like object to any backend
// ---------------------------------------------------------------------------

/// Put bytes data to any storage backend, without copying it where possible.
///
/// What is copied (details and the safety argument: `put_payload.rs`):
/// * `s3dlio.BytesView`, exact `bytes`: 0 copies.
/// * Any other C-contiguous buffer, whatever its item type (bytearray, memoryview, numpy
///   uint8/float32/bool/..., `array.array`, ctypes arrays, mmap): 0 copies. The buffer is
///   held and read in place with the GIL released, so other Python threads keep running
///   during a large PUT.
/// * A non-contiguous (strided, reversed, transposed) view: one gather copy with the GIL
///   held. An object that only provides `__bytes__()`: one copy, made by Python.
///
/// The native stores then write that memory without another copy on `s3://`, `gs://` (official client),
/// `az://` and `file://`. `direct://` (O_DIRECT) writes a page-aligned source (address
/// multiple of 4096) straight from it; any other source is copied once into an aligned
/// bounce buffer from the shared per-NUMA-node pool (16 MiB buffers by default, 4 ping-pong pairs per node).
///
/// The call blocks until the object is written. While it runs the source is exported:
/// * resizing it from another thread (`bytearray.extend()`/`clear()`) raises `BufferError`;
/// * you must not modify its bytes from another thread: that is a DATA RACE (the bytes
///   uploaded are undefined; the buffer is never freed or resized meanwhile, but that does not
///   make a concurrent write safe).
///
/// The call does not return - normally or with an error - until every native reader of the
/// buffer is done and the export has been released. The wait is bounded
/// (`S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`, 30 s) and not interruptible by signals; if a backend
/// still holds the payload then, the call raises `s3dlio.ExportStillReferencedError` directly in
/// the calling thread (no Future or Task can rewrite it), the buffer stays pinned and must not
/// be reused, and `s3dlio._put_export_stats()["quiesce_timeouts"]` counts it.
///
/// Accepted payloads (issue #261): `bytes`, any buffer-protocol object, and any object with
/// `__bytes__()`. An int, a bool or an arbitrary iterable is refused with a `TypeError`
/// before anything is submitted (`bytes(5)` used to write five zero bytes).
///
/// # Arguments
/// * `uri` - Full URI including scheme (s3://, az://, gs://, file://, direct://)
/// * `data` - Python bytes, a buffer-protocol object (bytearray, memoryview, array.array, numpy
///   array, ...) or an object with `__bytes__()`. An int, a bool or an arbitrary iterable is
///   refused (see "Accepted payloads").
///
/// # Examples
/// ```python
/// import s3dlio
///
/// # Put bytes to S3
/// s3dlio.put_bytes("s3://bucket/key.bin", b"hello world")
///
/// # Put bytes to local file
/// s3dlio.put_bytes("file:///tmp/test.bin", b"hello world")
///
/// # Put bytes to Azure
/// s3dlio.put_bytes("az://myaccount/container/blob.bin", b"hello world")
///
/// # Put bytearray or any buffer-like object
/// data = bytearray(b"hello world")
/// s3dlio.put_bytes("file:///tmp/test.bin", data)
///
/// # Put dgen_py.BytesView or numpy array
/// import dgen_py
/// data = dgen_py.generate_buffer(1024)
/// s3dlio.put_bytes("s3://bucket/data.bin", data)
/// ```
#[pyfunction]
pub fn put_bytes(py: Python<'_>, uri: &str, data: &Bound<'_, PyAny>) -> PyResult<()> {
    let uri = uri.to_owned();

    let payload = put_payload(py, data, PayloadUse::Blocking)?;
    let data_owned = payload.bytes;

    let result = py.detach(move || {
        // Build the store on the caller thread BEFORE submit_io (issue #173).
        let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move { put_verified_with_retry(store.as_ref(), &uri, data_owned).await })
    });
    // The put is over (it succeeded, failed, or the store gave up), but a backend may still hold
    // a clone of the payload (an SDK/hyper task, a detached writer). Wait until the LAST clone is
    // gone (the owner's drop), then release the export with this thread's real GIL token: the
    // call does not return while any native reader can still touch the caller's buffer. When the
    // store already dropped everything (the usual case) this is just a drain under the GIL.
    // Bounded (see `export_quiesce_timeout`); on timeout the buffer stays pinned and the call
    // raises ExportStillReferencedError. The wait is not interruptible by signals.
    if let Some(latch) = &payload.latch {
        if quiesce_blocking(py, latch).is_err() {
            return Err(still_referenced_error(&put_outcome_text(&result)));
        }
    }
    result
}

create_exception!(
    _pymod,
    ExportStillReferencedError,
    PyRuntimeError,
    "A storage backend still references a borrowed Python buffer long after the put finished \
     (S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS). The data may or may not have been written, and the \
     buffer stays pinned: do not modify, resize or free it."
);

fn still_referenced_error(outcome: &str) -> PyErr {
    ExportStillReferencedError::new_err(still_referenced_message(outcome))
}

/// The message of a barrier timeout: names the timeout value and what the put itself did.
fn still_referenced_message(outcome: &str) -> String {
    let timeout = crate::constants::export_quiesce_timeout();
    format!(
        "a storage backend still references the buffer {timeout:?} after the put finished \
         ({outcome}); the buffer stays pinned - do not modify, resize or free it \
         (S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS)"
    )
}

/// The message of an INTERNAL failure of the barrier machinery (not a backend that holds on):
/// the awaitable falls back to polling and stays pending until the export is released.
fn barrier_internal_message(detail: &str) -> String {
    format!(
        "internal error in the buffer completion barrier ({detail}); the buffer may still be \
         referenced and stays pinned; the awaitable keeps waiting by polling - do not modify, \
         resize or free the buffer until it completes"
    )
}

/// What the blocking `put_bytes` did, for error messages.
fn put_outcome_text(result: &PyResult<()>) -> String {
    match result {
        Ok(()) => "the put itself succeeded".to_string(),
        Err(e) => format!("the put itself failed: {e}"),
    }
}

/// Async version of put_bytes() - accepts any bytes-like object
///
/// Must be called with a running event loop, and awaited.
///
/// What is copied:
/// * `s3dlio.BytesView`, exact `bytes`: 0 copies. Also any other IMMUTABLE source whose
///   `bytes` base is known (a `bytes` subclass, `memoryview(bytes)`): 0 copies, in either mode.
/// * Any other C-contiguous buffer (bytearray, memoryview, numpy of any dtype,
///   `array.array`, ctypes, mmap), DEFAULT (`borrow=True`): 0 copies. The buffer is held and
///   read in place until the awaitable completes. By default the caller PROMISES not to modify,
///   resize or free the buffer until the awaitable has completed (successfully or not); until
///   then `bytearray.extend()/clear()` raise `BufferError` (the buffer is locked against
///   resizing) and any write to the bytes is a data race (the bytes uploaded are undefined). The
///   awaitable is a `Future` that NEVER reaches
///   a terminal state - result, exception or cancelled - while any native reader of the buffer
///   remains or before the export has actually been released, under the supported consumer
///   operations: `await`, `cancel()`, `Task.cancel()` (also repeated), `asyncio.timeout()` and
///   `wait_for()` leave the awaiting task pending until the future ends; a cancellation or timeout
///   is delivered only after that point. The future belongs to s3dlio: forcing it to complete with
///   `set_result` / `set_exception` (or the base-class methods) is unsupported and ends it at once.
///   So reusing the buffer after the `await` finished - with a result, an error, `CancelledError`
///   or `TimeoutError` - is safe. Cancelling therefore waits for a running native write to finish.
///   Every `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (default 30 s) that a reader remains, a
///   `RuntimeWarning` naming the timeout is emitted, `s3dlio._put_export_stats()["quiesce_timeouts"]`
///   is incremented and the wait continues (it does not complete). If the barrier machinery itself
///   fails, it polls instead; it still completes only after the release. (Await the future itself:
///   `asyncio.shield(fut)` bypasses the guard - the shield is cancelled at once while the put
///   still reads the buffer.)
/// * Same, with `borrow=False` (the opt-in snapshot): a mutable buffer is copied ONCE, before
///   this function returns, with the GIL released above 256 KiB. The caller may then reuse,
///   modify or resize the buffer immediately. The only requirement is not to modify it from
///   another thread DURING that copy. This copy is the documented price of not borrowing.
/// * A non-contiguous view: one gather copy with the GIL held; `__bytes__()`-only object:
///   one copy. (`borrow` has no effect on these.)
///
/// Accepted payloads (issue #261): `bytes`, any buffer-protocol object, and any object with
/// `__bytes__()`. An int, a bool or an arbitrary iterable is refused with a `TypeError`
/// before anything is submitted (`bytes(5)` used to write five zero bytes).
#[pyfunction]
#[pyo3(signature = (uri, data, *, borrow = true))]
fn put_bytes_async<'py>(
    py: Python<'py>,
    uri: &str,
    data: &Bound<'_, PyAny>,
    borrow: bool,
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();

    let usage = if borrow {
        PayloadUse::Borrowed
    } else {
        PayloadUse::Snapshot
    };
    let payload = put_payload(py, data, usage)?;
    let latch = payload.latch;
    let data_owned = payload.bytes;

    guarded_put(py, latch, move || async move {
        let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;
        put_verified_with_retry(store.as_ref(), &uri, data_owned)
            .await
            .map_err(py_err)?;
        // Python `None` (the Rust unit would reach Python as `()`; issue #271). `Option<()>`
        // converts to `None` INSIDE the exit-safe delivery (#192): this body never attaches.
        Ok(None::<()>)
    })
}

/// The awaitable of an asynchronous put whose payload may be borrowed from the caller.
///
/// A payload that borrows the caller's memory may still be read by native code (a blocking
/// file write that a cancelled future cannot stop, a backend task) after the Python future
/// has become terminal. So when `latch` is `Some` the awaitable is a `GuardedFuture` whose
/// cancel() cancels the put but whose own completion - result, error OR cancellation - is
/// delayed until every native reader is done and the export has been released with a real GIL
/// token. `make` builds the put future; it owns the payload, and is dropped unrun if the guard
/// cannot be prepared. Everything that can fail is prepared BEFORE the native put is spawned.
fn guarded_put<'py, F, Fut>(
    py: Python<'py>,
    latch: Option<std::sync::Arc<Latch>>,
    make: F,
) -> PyResult<pyo3::Bound<'py, PyAny>>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = PyResult<Option<()>>> + Send + 'static,
{
    let guard = match &latch {
        None => None,
        Some(latch) => match prepare_guard(py, latch.clone()) {
            Ok(g) => Some(g),
            Err(e) => {
                drop(make); // nothing was spawned: this releases the (queued) exports...
                if quiesce_blocking(py, latch).is_err() {
                    return Err(still_referenced_error(&format!(
                        "the put could not be started: {e}"
                    )));
                }
                return Err(e);
            }
        },
    };

    let inner = future_into_py(py, make());
    // On EVERY error path from here the barrier is kept: the blocking wait + drain below.
    let fail = |py: Python<'py>, latch: &Option<std::sync::Arc<Latch>>, e: PyErr| -> PyErr {
        if let Some(latch) = latch {
            if quiesce_blocking(py, latch).is_err() {
                return still_referenced_error(&format!("the put could not be started: {e}"));
            }
        }
        e
    };
    let inner = match inner {
        Ok(i) => i,
        Err(e) => return Err(fail(py, &latch, e)),
    };
    match (guard, &latch) {
        (Some(g), Some(latch)) => wrap_in_guard(py, g, inner, latch),
        _ => Ok(inner),
    }
}

/// Wrap the already-spawned put future `inner` in the `GuardedFuture`.
///
/// If the wrapper cannot be built (the Python class call fails: out of memory) the put is already
/// running. `inner.cancel()` is requested, but it only schedules the cancellation on the event
/// loop, and the loop thread is then blocked in the wait below - so the cancellation takes effect
/// only AFTER that wait and the put keeps running meanwhile. The call falls back to the blocking
/// wait for the native readers before it raises. **That fallback blocks the event-loop thread**
/// for as long as the running put needs, bounded by the quiesce timeout
/// (`S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`); on expiry it raises `ExportStillReferencedError` even
/// though the put may still be running (the buffer stays pinned). It can only happen on that
/// OOM-type failure.
fn wrap_in_guard<'py>(
    py: Python<'py>,
    guard: PreparedGuard<'py>,
    inner: Bound<'py, PyAny>,
    latch: &std::sync::Arc<Latch>,
) -> PyResult<Bound<'py, PyAny>> {
    match guard.instantiate(py, inner.clone()) {
        Ok(f) => Ok(f),
        Err(e) => {
            let _ = inner.call_method0("cancel");
            if quiesce_blocking(py, latch).is_err() {
                return Err(still_referenced_error(&format!(
                    "the put could not be started: {e}"
                )));
            }
            Err(e)
        }
    }
}

/// The Python helper that makes an awaitable's completion wait for the export barrier.
///
/// Not a coroutine: a coroutine body never runs if its Task is cancelled before the first step,
/// which would let the caller see `CancelledError` while a native write still reads the buffer.
/// A `Future` subclass has no such window: `cancel()` is called on it directly, cancels the put,
/// and defers the future's own terminal state.
///
/// THE CONTRACT: this future never reaches a terminal state (result, exception, cancelled) while
/// a native reader remains or before the export has really been released - under the supported
/// consumer operations (`await`, `cancel()`, `Task.cancel()` once or repeatedly, `asyncio.timeout`,
/// `wait_for`) and whatever happens to the barrier machinery. Producer-side calls
/// (`set_result`, `set_exception`, base-class `asyncio.Future` methods) are unsupported: they end
/// the future at once. At every quiesce deadline it warns and counts, then waits again. If
/// the machinery itself fails (the quiesce future cannot be created, `add_done_callback` raises,
/// the waiter fails, `latch_ready` raises) it falls back to polling with `loop.call_later`, and
/// if even that cannot be scheduled to a blocking wait on the event-loop thread. A failure is
/// reported through `event('internal', ...)` and counted, never turned into a completion.
const PUT_GUARD_PY: &std::ffi::CStr = c_str!(
    "import asyncio
import warnings

# Failures of the machinery we recover from; KeyboardInterrupt/SystemExit are left alone.
_RECOVERABLE = (Exception, asyncio.CancelledError)

class GuardedFuture(asyncio.Future):
    def __init__(self, inner, latch_ready, quiesce, event, block, poll_interval):
        super().__init__(loop=inner.get_loop())
        self._inner = inner
        self._latch_ready = latch_ready
        self._quiesce = quiesce
        self._event = event
        self._block = block
        self._poll_interval = poll_interval
        self._outcome = None
        self._cancel_msg = None
        self._finishing = False
        inner.add_done_callback(self._inner_done)

    def cancel(self, msg=None):
        # Request cancellation of the put; this future stays pending until no native reader of
        # the borrowed buffer remains and the export is released, then ends cancelled (see
        # _complete). When the put is already over (inner done: its outcome is fixed and only
        # the barrier is pending) return False: a Task awaiting this future then records the
        # cancellation itself and raises CancelledError right after the barrier completes,
        # instead of the outcome being lost.
        if self.done() or self._finishing:
            return False
        if not self._inner.cancel():
            return False
        if self._cancel_msg is None:
            self._cancel_msg = msg
        return True

    def _inner_done(self, inner):
        if self._finishing:
            return
        self._finishing = True
        try:
            if inner.cancelled():
                outcome = (2, None)
            else:
                exc = inner.exception()
                outcome = (1, exc) if exc is not None else (0, inner.result())
        except _RECOVERABLE as e:
            outcome = (1, e)
        self._outcome = outcome
        self._await_release()

    def _await_release(self):
        # Completes ONLY when latch_ready() returned True (owner dropped AND export released).
        if self.done():
            return  # finished by someone else: stop waiting and polling
        try:
            if self._latch_ready():
                self._complete()
                return
        except _RECOVERABLE as e:
            self._fallback('latch_ready', e)
            return
        try:
            waiter = self._quiesce()
            waiter.add_done_callback(self._quiesced)
        except _RECOVERABLE as e:
            self._fallback('quiesce', e)

    def _quiesced(self, waiter):
        try:
            timed_out = bool(waiter.result())
        except _RECOVERABLE as e:
            self._fallback('waiter', e)
            return
        if timed_out:
            # Deadline: warn, count (the wait counted it), and wait again. NEVER complete here.
            try:
                warnings.warn(self._event('timeout', ''), RuntimeWarning, stacklevel=1)
            except _RECOVERABLE:
                pass
        self._await_release()

    def _fallback(self, where, e):
        try:
            self._event('internal', '%s: %r' % (where, e))
        except _RECOVERABLE:
            pass
        self._poll()

    def _poll(self):
        if self.done():
            return  # finished by someone else: stop polling
        try:
            if self._latch_ready():
                self._complete()
                return
        except _RECOVERABLE:
            pass  # keep polling; the first failure was reported by _fallback
        try:
            self.get_loop().call_later(self._poll_interval, self._poll)
        except _RECOVERABLE as e:
            try:
                self._event('internal', 'call_later: %r' % (e,))
                self._block()  # last resort: blocks this thread until the export is released
            except _RECOVERABLE:
                return  # nothing safe is left to do: stay pending (ownership first)
            self._complete()

    def _complete(self):
        if self.done():
            return
        kind, value = self._outcome
        if kind == 0:
            asyncio.Future.set_result(self, value)
        elif kind == 1:
            asyncio.Future.set_exception(self, value)
        else:
            asyncio.Future.cancel(self, self._cancel_msg)
"
);

/// Everything the guard needs that can fail to build; made before the put is spawned.
struct PreparedGuard<'py> {
    class: Bound<'py, PyAny>,
    latch_ready: Bound<'py, PyAny>,
    quiesce: Bound<'py, PyAny>,
    event: Bound<'py, PyAny>,
    block: Bound<'py, PyAny>,
}

impl<'py> PreparedGuard<'py> {
    /// Wrap the `inner` put future (cannot fail for lack of resources prepared above).
    fn instantiate(self, py: Python<'py>, inner: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let _ = py;
        self.class.call1((
            inner,
            self.latch_ready,
            self.quiesce,
            self.event,
            self.block,
            crate::constants::EXPORT_BARRIER_POLL_INTERVAL.as_secs_f64(),
        ))
    }
}

fn prepare_guard<'py>(
    py: Python<'py>,
    latch: std::sync::Arc<Latch>,
) -> PyResult<PreparedGuard<'py>> {
    static GUARD: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();
    let class = match GUARD.get() {
        Some(g) => g.bind(py).clone(),
        None => {
            let module = PyModule::from_code(
                py,
                PUT_GUARD_PY,
                c_str!("s3dlio_put_guard.py"),
                c_str!("_s3dlio_put_guard"),
            )?;
            let f = module.getattr("GuardedFuture")?.unbind();
            GUARD.get_or_init(|| f).bind(py).clone()
        }
    };
    let l1 = latch.clone();
    let l2 = latch.clone();
    let latch_ready_fn = pyo3::types::PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, pyo3::types::PyTuple>,
              _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>|
              -> PyResult<bool> { Ok(latch_ready(args.py(), &l1)) },
    )?;
    let quiesce_fn = pyo3::types::PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, pyo3::types::PyTuple>,
              _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>|
              -> PyResult<Py<PyAny>> {
            let latch = latch.clone();
            // resolves to True when the wait hit its deadline
            let fut = future_into_py(args.py(), async move { Ok(quiesce_async(latch).await) })?;
            Ok(fut.unbind())
        },
    )?;
    // Reports a barrier event: ('timeout', _) -> the warning text of the missed deadline (the
    // wait already counted it); ('internal', detail) -> counts and logs the machinery failure.
    let event_fn = pyo3::types::PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, pyo3::types::PyTuple>,
              _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>|
              -> PyResult<String> {
            let kind: String = args.get_item(0)?.extract()?;
            let detail: String = args.get_item(1)?.extract()?;
            if kind == "internal" {
                note_internal_error();
                let msg = barrier_internal_message(&detail);
                tracing::warn!("s3dlio: {msg}");
                Ok(msg)
            } else {
                Ok(deadline_warning_text(export_stats().quiesce_timeouts))
            }
        },
    )?;
    // Last resort when even a poll cannot be scheduled: wait on THIS thread (GIL released) until
    // the export is really released, however long a native reader needs.
    let block_fn = pyo3::types::PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, pyo3::types::PyTuple>,
              _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>|
              -> PyResult<()> {
            while quiesce_blocking(args.py(), &l2).is_err() {
                // each missed deadline is logged and counted by the wait; keep waiting
            }
            Ok(())
        },
    )?;
    Ok(PreparedGuard {
        class,
        latch_ready: latch_ready_fn.into_any(),
        quiesce: quiesce_fn.into_any(),
        event: event_fn.into_any(),
        block: block_fn.into_any(),
    })
}

/// Counters of the buffer-export barrier (diagnostics and tests):
/// `{"quiesce_timeouts", "queued", "internal_errors", "shutdown_attach_waits",
/// "shutdown_attach_wait_timeouts", "releaser_attaches", "test_releaser_parked"}`.
#[pyfunction]
fn _put_export_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let s = export_stats();
    let d = PyDict::new(py);
    d.set_item("quiesce_timeouts", s.quiesce_timeouts)?;
    d.set_item("queued", s.queued)?;
    d.set_item("internal_errors", s.internal_errors)?;
    d.set_item("shutdown_attach_waits", s.shutdown_attach_waits)?;
    d.set_item(
        "shutdown_attach_wait_timeouts",
        s.shutdown_attach_wait_timeouts,
    )?;
    d.set_item("releaser_attaches", s.releaser_attaches)?;
    d.set_item("test_releaser_parked", s.test_releaser_parked)?;
    Ok(d)
}

/// TEST-ONLY hidden hook (not part of the API; never call it in production): queue an undrained export of `data` and, with `park_ms > 0`, park the
/// autonomous releaser once between its attach-gate entry and `Python::try_attach`.
#[pyfunction]
#[pyo3(signature = (data, park_ms = 0))]
fn _put_test_queue_export(py: Python<'_>, data: &Bound<'_, PyAny>, park_ms: u64) -> PyResult<()> {
    super::put_payload::test_queue_undrained_export(py, data, park_ms)
}

/// TEST-ONLY hidden hook (not part of the API; never call it in production): the next borrowed/blocking payload that holds a buffer export keeps one clone
/// of its bytes, like a backend that keeps reading after the put ended, until
/// `_put_test_release_retained()`.
#[pyfunction]
fn _put_test_retain_next_clone() {
    super::put_payload::arm_retain_for_test();
}

/// TEST-ONLY hidden hook (not part of the API; never call it in production): number of clones retained by `_put_test_retain_next_clone()` right now.
#[pyfunction]
fn _put_test_retained() -> usize {
    super::put_payload::retained_for_test()
}

/// TEST-ONLY hidden hook (not part of the API; never call it in production): drops every retained clone (in order, with the GIL held) and disarms a pending arming; returns how many were dropped.
#[pyfunction]
fn _put_test_release_retained() -> usize {
    super::put_payload::release_retained_for_test()
}

// ---------------------------------------------------------------------------
// Caller-buffer GET, typed HEAD and pure capabilities (#264, plan §4.5 A5)
// ---------------------------------------------------------------------------

create_exception!(
    _pymod,
    ObjectNotFoundError,
    PyFileNotFoundError,
    "The object does not exist (get_into_async on a missing object; head_async reports \
     absence as None instead)."
);

create_exception!(
    _pymod,
    LengthMismatchError,
    PyValueError,
    "The object's size is not the expected length. `.expected` is the length the caller \
     asked for; `.actual` is the object's size, or None when it is not known. Nothing in the \
     destination buffer is valid after this error."
);

create_exception!(
    _pymod,
    UnsupportedOperationError,
    PyNotImplementedError,
    "This build or provider does not support the operation (see object_capabilities). \
     Nothing was fetched and no fallback was attempted."
);

/// The result of `head_async` for an object that exists: its size and opaque identity.
///
/// Frozen: the fields cannot be assigned, and equal values hash equal.
#[pyclass(name = "ObjectMetadata", frozen, eq, hash)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PyHeadMetadata {
    /// The object size in bytes (never negative).
    #[pyo3(get)]
    size: u64,
    /// The entity tag exactly as the provider reports it (opaque, not assumed to be an MD5).
    #[pyo3(get)]
    etag: Option<String>,
    /// The provider's version identifier (S3 version id, GCS generation), when it reports one.
    #[pyo3(get)]
    version_id: Option<String>,
}

#[pymethods]
impl PyHeadMetadata {
    /// Python-style: `ObjectMetadata(size=5, etag='e', version_id=None)`. The strings are
    /// rendered by Python's own `repr`, so quotes and backslashes are escaped the Python way.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let optional = |value: &Option<String>| -> PyResult<String> {
            match value {
                None => Ok("None".to_owned()),
                Some(text) => Ok(pyo3::types::PyString::new(py, text).repr()?.to_string()),
            }
        };
        Ok(format!(
            "ObjectMetadata(size={}, etag={}, version_id={})",
            self.size,
            optional(&self.etag)?,
            optional(&self.version_id)?
        ))
    }
}

impl From<HeadMetadata> for PyHeadMetadata {
    fn from(meta: HeadMetadata) -> Self {
        Self {
            size: meta.size,
            etag: meta.etag,
            version_id: meta.version_id,
        }
    }
}

/// The Python exception for a failed object operation (A2, A5 "Results and errors"): the typed
/// classification when the error carries one, otherwise the generic `RuntimeError` of [`py_err`].
/// The message is the whole error chain either way.
///
/// Needs the GIL (`LengthMismatchError` gets `.expected` / `.actual` set), so it runs where the
/// result is delivered to Python (see [`Outcome`]), never on a runtime worker.
fn object_py_err(py: Python<'_>, err: &anyhow::Error) -> PyErr {
    let message = error_chain_message(err);
    let Some(typed) = object_error(err) else {
        return PyRuntimeError::new_err(message);
    };
    match typed.kind {
        ObjectErrorKind::NotFound => ObjectNotFoundError::new_err(message),
        ObjectErrorKind::LengthMismatch { expected, actual } => {
            let exception = LengthMismatchError::new_err(message);
            let value = exception.value(py);
            // Setting a plain attribute on a fresh exception does not fail in practice. If it
            // ever did, the classification matters more than the attributes: keep the type.
            let set = value
                .setattr("expected", expected)
                .and_then(|()| value.setattr("actual", actual));
            if let Err(e) = set {
                warn!("s3dlio: could not set LengthMismatchError.expected/actual: {e}");
            }
            exception
        }
        ObjectErrorKind::Unsupported => UnsupportedOperationError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// The value a caller-buffer GET or typed HEAD future resolves to. A failure travels as DATA
/// until the result is delivered, and only then becomes the typed Python exception
/// ([`object_py_err`]): `IntoPyObject` runs with the GIL held, on the delivery path, so the
/// conversion never makes a Tokio worker wait for the GIL.
struct Outcome<T>(anyhow::Result<T>);

impl<'py, T> IntoPyObject<'py> for Outcome<T>
where
    T: IntoPyObject<'py>,
{
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            Ok(value) => value.into_bound_py_any(py),
            Err(err) => Err(object_py_err(py, &err)),
        }
    }
}

/// Which operation a capability check is for.
#[derive(Clone, Copy)]
enum Operation {
    GetInto,
    Head,
}

/// Step 1 of A5: pure capability check. `ValueError` for a malformed URI,
/// `UnsupportedOperationError` (synchronously) when this build or provider lacks the operation.
fn require_capability(uri: &str, operation: Operation) -> PyResult<()> {
    let caps = crate::capabilities::object_capabilities(uri)
        .map_err(|e| PyValueError::new_err(error_chain_message(&e)))?;
    let (supported, name) = match operation {
        Operation::GetInto => (caps.get_into, "get_into_async"),
        Operation::Head => (caps.head, "head_async"),
    };
    if supported {
        Ok(())
    } else {
        Err(UnsupportedOperationError::new_err(format!(
            "{name} is not supported for {} URIs in this build (see object_capabilities)",
            caps.provider
        )))
    }
}

/// `expected_len` as a `usize`: an integer (not a bool), non-negative, within `usize`.
///
/// Anything with `__index__` is accepted, like `operator.index`. A failure of that conversion is
/// never replaced by a message that hides it: a non-integer is a `TypeError` and a value beyond
/// the machine range a `ValueError`, each with the original exception as `__cause__`, and any
/// other exception raised by `__index__` propagates unchanged.
fn checked_expected_len(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    use pyo3::exceptions::{PyOverflowError, PyTypeError};
    let py = value.py();
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err(
            "expected_len must be an integer, not a bool",
        ));
    }
    let n: i128 = match value.extract() {
        Ok(n) => n,
        Err(e) if e.is_instance_of::<PyOverflowError>(py) => {
            // An `int` (or an `__index__` result) too large even for i128.
            let wrapped = PyValueError::new_err("expected_len is too large");
            wrapped.set_cause(py, Some(e));
            return Err(wrapped);
        }
        Err(e) if e.is_instance_of::<PyTypeError>(py) => {
            let wrapped = PyTypeError::new_err("expected_len must be an integer");
            wrapped.set_cause(py, Some(e));
            return Err(wrapped);
        }
        Err(e) => return Err(e),
    };
    if n < 0 {
        return Err(PyValueError::new_err(format!(
            "expected_len must not be negative, got {n}"
        )));
    }
    usize::try_from(n).map_err(|_| PyValueError::new_err("expected_len is too large"))
}

/// The two steps of starting an awaitable that tests need to fail on demand. The live
/// implementation is [`LiveSeams`]; tests substitute failing ones.
trait StartSeams {
    /// Hand the native future to the Python event loop (`future_into_py`). On failure the future
    /// is dropped without having run.
    fn bridge<'py, F, T>(&self, py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
    where
        F: Future<Output = PyResult<T>> + Send + 'static,
        T: for<'a> IntoPyObject<'a> + Send + 'static;

    /// Wrap the bridged future in the guarded completion (`PreparedGuard::instantiate`).
    fn instantiate<'py>(
        &self,
        py: Python<'py>,
        guard: PreparedGuard<'py>,
        inner: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>>;
}

/// The real bridge and guard constructor.
struct LiveSeams;

impl StartSeams for LiveSeams {
    fn bridge<'py, F, T>(&self, py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
    where
        F: Future<Output = PyResult<T>> + Send + 'static,
        T: for<'a> IntoPyObject<'a> + Send + 'static,
    {
        future_into_py(py, fut)
    }

    fn instantiate<'py>(
        &self,
        py: Python<'py>,
        guard: PreparedGuard<'py>,
        inner: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        guard.instantiate(py, inner)
    }
}

/// Request cancellation of `future`, ignoring any failure to do so. Used only on the failure
/// paths of starting an awaitable, where the original error is what the caller must see and
/// nothing may wait on the outcome of the cancellation.
fn cancel_ignoring_errors(future: &Bound<'_, PyAny>) {
    if let Err(e) = future.call_method0("cancel") {
        debug!("s3dlio: ignoring an error cancelling an awaitable that failed to start: {e}");
    }
}

/// Read the object at `uri` into `dst`, which must be a writable, C-contiguous byte buffer
/// (`bytearray`, `memoryview` of bytes, a NumPy `uint8` array, an `mmap`, ...). Returns an
/// awaitable that resolves to `expected_len` once exactly `expected_len` bytes have been
/// written into `dst[:expected_len]`.
///
/// `expected_len` is the object's size: a different size fails with `LengthMismatchError`
/// (`.expected`, `.actual`) and nothing is written past `expected_len`. A missing object raises
/// `ObjectNotFoundError`; a build or provider without the operation raises
/// `UnsupportedOperationError` immediately (see `object_capabilities`). No fallback download
/// plus copy is ever made.
///
/// While the awaitable is pending, native code has exclusive use of `dst[:expected_len]`: do not
/// read, modify, resize or reuse the buffer until the awaitable has finished (result, exception
/// or cancelled). The export blocks resizing but cannot stop a racing writer. The awaitable
/// never finishes while a native writer remains or before the export has been released, so the
/// buffer is safe to reuse right after the `await`, including after an error or a cancellation.
/// After a failure the contents of `dst[:expected_len]` are unspecified. A failure before the
/// transfer starts (for example, no running event loop) releases the buffer before it raises.
///
/// Concurrent receives: two calls that write the same or overlapping memory at the same time are
/// forbidden (each would hold a mutable view over bytes the other also writes). Give every
/// in-flight call its own destination, or non-overlapping slices of one buffer.
///
/// The awaitable is a `Future` that belongs to s3dlio. Await it directly:
/// `asyncio.shield(fut)` bypasses the guard (the shield is cancelled at once while the native
/// read still writes into `dst`), and forcing the future to complete with `set_result` /
/// `set_exception` (or the base-class methods) is unsupported and ends it at once, so `dst`
/// would then be reused while a native writer may still run. `cancel()`, `Task.cancel()`
/// (also repeated), `asyncio.timeout()` and `wait_for()` leave the awaiting task pending until
/// the native read has stopped and the export is released; the cancellation or timeout is
/// delivered only after that point.
///
/// Every `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (default 30 s) that a native writer remains after
/// the call finished, a `RuntimeWarning` naming the timeout is emitted,
/// `s3dlio._put_export_stats()["quiesce_timeouts"]` is incremented and the wait continues: the
/// awaitable never completes while a native writer remains. A call still running when the
/// interpreter exits is not waited for.
#[pyfunction]
#[pyo3(signature = (uri, dst, *, expected_len))]
fn get_into_async<'py>(
    py: Python<'py>,
    uri: &str,
    dst: &Bound<'py, PyAny>,
    expected_len: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let uri = uri.to_owned();

    // Step 1: pure capability check; unsupported raises here, synchronously.
    require_capability(&uri, Operation::GetInto)?;
    let expected_len = checked_expected_len(expected_len)?;

    // Step 2: build the store on the caller thread (#173).
    let store = py
        .detach(|| get_or_create_store(&uri, global_logger()))
        .map_err(py_err)?;

    // Step 3 (the latch; the guard is prepared from it in `get_into_with_store`). The guard and
    // the export always share this one latch.
    let latch = Latch::new();
    get_into_with_store(py, &LiveSeams, store, uri, dst, expected_len, latch)
}

/// Steps 3 to 7 of A5 for a store that is already built: prepare the guard on `latch` (3),
/// acquire the export on the same latch (4), bridge a task that waits for the export before it
/// touches the store (5), instantiate the guard (6; never `wrap_in_guard`, whose error path
/// waits on the latch) and hand the export to the task (7).
///
/// A failure in 3 or 4 holds no export and raises directly. A failure in 5 to 7, before the send
/// succeeded, releases the export directly with this GIL token (no queue, no wait), cancels what
/// exists (ignoring cancel errors) and raises the ORIGINAL error: the buffer is reusable at once.
/// Once the send succeeded nothing fallible remains.
fn get_into_with_store<'py, S: StartSeams>(
    py: Python<'py>,
    seams: &S,
    store: Arc<dyn ObjectStore>,
    uri: String,
    dst: &Bound<'py, PyAny>,
    expected_len: usize,
    latch: Arc<Latch>,
) -> PyResult<Bound<'py, PyAny>> {
    let guard = prepare_guard(py, latch.clone())?;
    let owner = acquire_writable(py, dst, expected_len, &latch)?;

    let (tx, rx) = oneshot::channel::<WritableExport>();
    let task = async move {
        // The export arrives only after the guard exists: no native write before that.
        let mut owner = match rx.await {
            Ok(owner) => owner,
            Err(_) => {
                return Ok::<_, PyErr>(Outcome(Err(anyhow::anyhow!(
                    "the destination buffer was never handed to the read"
                ))));
            }
        };
        // `owner` is declared before the `get_into` future, so it drops after every writer.
        let progress = ReceiveProgress::new();
        let result = store.get_into(&uri, owner.as_mut_slice(), &progress).await;
        Ok(Outcome(result.map(|()| expected_len)))
    };

    let inner = match seams.bridge(py, task) {
        Ok(inner) => inner,
        Err(e) => {
            drop(tx);
            owner.release(py);
            return Err(e);
        }
    };
    let guarded = match seams.instantiate(py, guard, inner.clone()) {
        Ok(guarded) => guarded,
        Err(e) => {
            // Order matters: release the export, cancel the bridged future, and only THEN close
            // the start channel. Closing it first lets the task finish with "never handed" before
            // the cancel lands, leaving an exception nobody retrieves (logged at GC).
            owner.release(py);
            cancel_ignoring_errors(&inner);
            drop(tx);
            return Err(e);
        }
    };
    match tx.send(owner) {
        Ok(()) => Ok(guarded),
        Err(owner) => {
            owner.release(py);
            cancel_ignoring_errors(&guarded);
            Err(PyRuntimeError::new_err(
                "get_into_async could not hand the destination buffer to its task",
            ))
        }
    }
}

/// Marks a `head_async` call's latch when it is dropped: the point at which no native request
/// of the call can still be running. There is no buffer, so both latch signals are set together.
///
/// It is moved into the future before the request exists and is the FIRST local of the future
/// body, so it drops after any request future, wherever the future is dropped: unpolled, on a
/// closed start channel, on cancellation, on a panic or on completion. `Drop` takes no GIL.
struct HeadQuiescence {
    latch: Arc<Latch>,
}

impl Drop for HeadQuiescence {
    fn drop(&mut self) {
        self.latch.mark_dropped();
        self.latch.mark_released();
        #[cfg(test)]
        head_token_tests::note_completion(&self.latch);
    }
}

/// Look up the object at `uri` without reading it. Returns an awaitable that resolves to an
/// `ObjectMetadata` (`size`, `etag`, `version_id`), or to `None` when the object does not
/// exist. S3's bodyless HEAD 404 and GCS's `NotFound` cannot tell an absent bucket from an
/// absent object, so on `s3://` and `gs://` a missing bucket also resolves to `None`; no extra
/// probe is made to tell them apart. A missing Azure container is distinguishable and raises.
/// A 403 and every other failure raises. A build or provider without the operation raises
/// `UnsupportedOperationError` immediately.
///
/// Like `get_into_async` the awaitable never finishes while a native request of the call
/// remains, so callers can bound outstanding native work; there is no buffer to release (and so
/// no destination to protect: the exclusive-use rules of `get_into_async` do not apply here).
///
/// The awaitable is a `Future` that belongs to s3dlio. Await it directly: `asyncio.shield(fut)`
/// bypasses the guard (the shield is cancelled at once while the request still runs), and
/// forcing the future to complete with `set_result` / `set_exception` (or the base-class
/// methods) is unsupported and ends it at once. `cancel()`, `Task.cancel()` (also repeated),
/// `asyncio.timeout()` and `wait_for()` leave the awaiting task pending until the request has
/// stopped; the cancellation or timeout is delivered only after that point.
///
/// Every `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` (default 30 s) that a native request remains after
/// the call finished, a `RuntimeWarning` naming the timeout is emitted,
/// `s3dlio._put_export_stats()["quiesce_timeouts"]` is incremented and the wait continues. A
/// call still running when the interpreter exits is not waited for.
#[pyfunction]
fn head_async<'py>(py: Python<'py>, uri: &str) -> PyResult<Bound<'py, PyAny>> {
    let uri = uri.to_owned();

    // Capability `head` (not `get_into`), then the store on the caller thread.
    require_capability(&uri, Operation::Head)?;
    let store = py
        .detach(|| get_or_create_store(&uri, global_logger()))
        .map_err(py_err)?;

    let latch = Latch::new();
    head_with_store(py, &LiveSeams, store, uri, latch)
}

/// The HEAD half of A5 for a store that is already built.
///
/// The start channel carries a unit permit. The [`HeadQuiescence`] token holds the latch and is
/// built before the future. A failure before the permit is sent drops `tx`, cancels what exists
/// (ignoring cancel errors) and raises the original error: the token marks the latch when the
/// future drops, and nothing waits.
fn head_with_store<'py, S: StartSeams>(
    py: Python<'py>,
    seams: &S,
    store: Arc<dyn ObjectStore>,
    uri: String,
    latch: Arc<Latch>,
) -> PyResult<Bound<'py, PyAny>> {
    let guard = prepare_guard(py, latch.clone())?;

    let (tx, rx) = oneshot::channel::<()>();
    let token = HeadQuiescence { latch };
    let task = async move {
        // First local, so it is dropped last: after any request future, on every path (finish,
        // cancellation, panic unwind). The explicit `drop` below is the normal-completion
        // spelling of the same moment.
        let quiescence = token;
        let outcome = async {
            if rx.await.is_err() {
                return Outcome(Err(anyhow::anyhow!("the HEAD request was never started")));
            }
            let result = store.head(&uri).await;
            Outcome(result.map(|found| found.map(PyHeadMetadata::from)))
        }
        .await;
        drop(quiescence);
        Ok::<_, PyErr>(outcome)
    };

    let inner = match seams.bridge(py, task) {
        Ok(inner) => inner,
        Err(e) => {
            drop(tx);
            return Err(e);
        }
    };
    let guarded = match seams.instantiate(py, guard, inner.clone()) {
        Ok(guarded) => guarded,
        Err(e) => {
            // Cancel first, then close the start channel (see `get_into_with_store`).
            cancel_ignoring_errors(&inner);
            drop(tx);
            return Err(e);
        }
    };
    match tx.send(()) {
        Ok(()) => Ok(guarded),
        Err(()) => {
            cancel_ignoring_errors(&guarded);
            Err(PyRuntimeError::new_err(
                "head_async could not start its request",
            ))
        }
    }
}

/// What this build can do for `uri`, from the scheme alone: no store, credentials, client or
/// network. Returns `{"provider": str, "get_into": bool, "head": bool}`; `provider` is `"s3"`,
/// `"az"`, `"gs"`, `"file"`, `"direct"` or `"unknown"`. `get_into` and `head` say whether
/// `get_into_async` and `head_async` are available for the URI in this build.
///
/// `ValueError` for a URI that is malformed for a known object scheme. A true answer is
/// structural: a store can still refuse a URI that passes here (for example an Azure account
/// name that depends on the configured endpoint). Rapid versus standard GCS routing is resolved
/// during real I/O, never here.
#[pyfunction]
#[pyo3(name = "object_capabilities")]
fn object_capabilities_py<'py>(py: Python<'py>, uri: &str) -> PyResult<Bound<'py, PyDict>> {
    let caps = crate::capabilities::object_capabilities(uri)
        .map_err(|e| PyValueError::new_err(error_chain_message(&e)))?;
    let result = PyDict::new(py);
    result.set_item("provider", caps.provider)?;
    result.set_item("get_into", caps.get_into)?;
    result.set_item("head", caps.head)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// mkdir() - Create directory/prefix for any backend
// ---------------------------------------------------------------------------

/// Create a directory (file://, direct://). A no-op on object stores (s3://, az://, gs://).
///
/// # Backend Behavior
/// - `file://`, `direct://`: Creates the directory and any missing parents using
///   create_dir_all(); an existing directory is success. `direct://` behaves exactly
///   like `file://` (O_DIRECT does not apply to directory operations).
/// - `s3://`, `az://`, `gs://`: **does nothing and returns `None`.** An object store has
///   no directories: a prefix exists as soon as an object is written under it, so there
///   is nothing to create. The URI is validated, no request is sent and no marker
///   object is written. To remove the objects under a prefix use
///   `delete(uri, recursive=True)`.
///
/// Never recorded in the op-log.
///
/// # Arguments
/// * `uri` - Full URI including scheme and trailing slash recommended for clarity
///
/// # Examples
/// ```python
/// import s3dlio
///
/// # Create local directory
/// s3dlio.mkdir("file:///tmp/mydata/subdir/")
///
/// # On object stores (s3://, az://, gs://) this is a no-op that succeeds:
/// # nothing is created, a prefix exists once an object is written under it
/// s3dlio.mkdir("s3://bucket/prefix/")
/// ```
#[pyfunction]
pub fn mkdir(py: Python<'_>, uri: &str) -> PyResult<()> {
    let uri = uri.to_owned();

    py.detach(move || {
        // Build the store on the caller thread BEFORE submit_io (issue #173).
        let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move { store.mkdir(&uri).await })
    })
}

/// Async version of mkdir()
#[pyfunction]
fn mkdir_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();

    future_into_py(py, async move {
        let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;
        store.mkdir(&uri).await.map_err(py_err)?;
        // Python `None` (the Rust unit would reach Python as `()`, an empty tuple; issue #233).
        // `Option<()>` converts to `None` INSIDE the exit-safe delivery (issue #192). This body
        // must never attach to Python itself: it runs on a Tokio worker before the delivery gate
        // takes its ticket, so a late completion could attach during interpreter finalization.
        Ok(None::<()>)
    })
}

#[pyfunction(name = "get_many_stats")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub fn get_many_stats(
    py: Python<'_>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<(usize, usize)> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    py.detach(|| {
        let res = get_objects_parallel(&uris, max_in_flight).map_err(py_err)?;
        let total: usize = res.iter().map(|(_, b)| b.len()).sum();
        Ok((res.len(), total))
    })
}

#[pyfunction(name = "get_many_stats_async")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub fn get_many_stats_async<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Bound<'p, PyAny>> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    future_into_py(py, async move {
        let (count, total) = task::spawn_blocking(move || {
            let res = get_objects_parallel(&uris, max_in_flight).map_err(py_err)?;
            let bytes = res.iter().map(|(_, b)| b.len()).sum::<usize>();
            Ok::<(usize, usize), PyErr>((res.len(), bytes))
        })
        .await
        .map_err(py_err)??;

        Python::attach(|py| (count, total).into_py_any(py))
    })
}

#[pyfunction(name = "get_many_async")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub(crate) fn get_many_async_py<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Bound<'p, PyAny>> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    future_into_py(py, async move {
        // get_objects_parallel returns Vec<(String, Bytes)> - keep as Bytes for zero-copy
        let pairs: Vec<(String, Bytes)> = task::spawn_blocking(move || {
            get_objects_parallel(&uris, max_in_flight).map_err(py_err)
        })
        .await
        .map_err(py_err)??;

        Python::attach(|_py| {
            let out = pairs
                .into_iter()
                .map(|(u, b)| (u, PyBytesView::new(b)))
                .collect::<Vec<_>>();
            Ok(out)
        })
    })
}

#[pyfunction]
pub fn get(py: Python<'_>, uri: &str) -> PyResult<PyBytesView> {
    // Use universal ObjectStore API for all backends (with op-log support)
    let uri_owned = uri.to_owned();
    py.detach(|| {
        // Build the store on the caller thread BEFORE submit_io (issue #173):
        // for a host-form URI this calls `run_on_global_rt`, which must not
        // be reached from a global-runtime worker.
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let bytes = store.get(&uri_owned).await?;
            Ok(PyBytesView::new(bytes))
        })
    })
}

// --- `get_range` function (universal backend support)
#[pyfunction(name = "get_range")]
#[pyo3(signature = (uri, offset, length = None))]
pub fn get_range_py(
    py: Python<'_>,
    uri: &str,
    offset: u64,
    length: Option<u64>,
) -> PyResult<PyBytesView> {
    // Use universal ObjectStore interface for range requests
    let uri_owned = uri.to_owned();
    py.detach(|| {
        // Build the store on the caller thread BEFORE submit_io (issue #173).
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let bytes = store.get_range(&uri_owned, offset, length).await?;
            Ok(PyBytesView::new(bytes))
        })
    })
}

/// Async variant of `get` — returns a coroutine yielding a `BytesView`.
///
/// The returned `BytesView` wraps the Arc-backed `Bytes` from the storage
/// backend with **no copy**.  Use `await`'d result's `.memoryview()` method
/// for fully zero-copy access from numpy / torch.
///
/// # Python usage
/// ```python
/// import asyncio, s3dlio
/// async def read():
///     bv = await s3dlio.get_async("gs://bucket/object")
///     mv = bv.memoryview()               # zero-copy
///     arr = np.frombuffer(mv, dtype=np.uint8)
/// asyncio.run(read())
/// ```
#[pyfunction(name = "get_async")]
#[pyo3(signature = (uri))]
pub fn get_async_py<'p>(py: Python<'p>, uri: &str) -> PyResult<Bound<'p, PyAny>> {
    let uri_owned = uri.to_owned();
    future_into_py(py, async move {
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;
        let bytes = store.get(&uri_owned).await.map_err(py_err)?;
        Python::attach(|py| Ok(Py::new(py, PyBytesView::new(bytes))?.into_any()))
    })
}

/// Async variant of `get_range` — returns a coroutine yielding a `BytesView`.
#[pyfunction(name = "get_range_async")]
#[pyo3(signature = (uri, offset, length = None))]
pub fn get_range_async_py<'p>(
    py: Python<'p>,
    uri: &str,
    offset: u64,
    length: Option<u64>,
) -> PyResult<Bound<'p, PyAny>> {
    let uri_owned = uri.to_owned();
    future_into_py(py, async move {
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;
        let bytes = store
            .get_range(&uri_owned, offset, length)
            .await
            .map_err(py_err)?;
        Python::attach(|py| Ok(Py::new(py, PyBytesView::new(bytes))?.into_any()))
    })
}

// --- `download` function
#[pyfunction]
#[pyo3(signature = (src_uri, dest_dir, max_in_flight = 64, recursive = true))]
pub fn download(
    py: Python<'_>,
    src_uri: &str,
    dest_dir: &str,
    max_in_flight: usize,
    recursive: bool,
) -> PyResult<()> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    let dir = PathBuf::from(dest_dir);
    let src_uri_owned = src_uri.to_owned();
    // The `download_objects` function in `s3_copy.rs` needs a `recursive` flag.
    // We'll assume for now it has been added.
    py.detach(move || {
        // Use the new generic download function that works with all backends
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            generic_download_objects(&src_uri_owned, &dir, max_in_flight, recursive, None).await
        })
    })
}

// --- `get_many` function (universal backend support)
#[pyfunction]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub fn get_many(
    py: Python<'_>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Vec<(String, PyBytesView)>> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    use crate::object_store::{infer_scheme, Scheme};
    use futures::stream::{FuturesUnordered, StreamExt};
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    py.detach(|| {
        // Check all URIs are the same scheme
        let schemes: Vec<_> = uris.iter().map(|u| infer_scheme(u)).collect();
        let first_scheme = schemes
            .first()
            .ok_or_else(|| PyRuntimeError::new_err("get_many requires at least one URI"))?;

        // Verify all URIs use the same scheme
        if !schemes.iter().all(|s| s == first_scheme) {
            return Err(PyRuntimeError::new_err(
                "get_many requires all URIs to use the same backend scheme",
            ));
        }

        // Route to appropriate backend
        match first_scheme {
            Scheme::S3 => {
                // Use existing optimized S3 implementation
                let res = get_objects_parallel(&uris, max_in_flight).map_err(py_err)?;
                Ok(res
                    .into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::Direct => {
                // Parallel O_DIRECT reads — bypass page cache via ConfigurableFileSystemObjectStore
                let uris_clone = uris.clone();
                let res = submit_io(async move {
                    let store = Arc::new(ConfigurableFileSystemObjectStore::with_direct_io());
                    let sem = Arc::new(Semaphore::new(clamp_in_flight(max_in_flight)));
                    let mut futs = FuturesUnordered::new();

                    for uri in uris_clone.iter().cloned() {
                        let store = Arc::clone(&store);
                        let sem = Arc::clone(&sem);
                        futs.push(tokio::spawn(async move {
                            let _permit = sem.acquire_owned().await.unwrap();
                            let bytes = store.get(&uri).await?;
                            Ok::<_, anyhow::Error>((uri, bytes))
                        }));
                    }

                    let mut out = Vec::new();
                    while let Some(res) = futs.next().await {
                        let (uri, bytes) =
                            res.map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;
                        out.push((uri, bytes));
                    }

                    // Maintain input order
                    out.sort_by_key(|(u, _)| uris_clone.iter().position(|x| x == u).unwrap());
                    Ok::<_, anyhow::Error>(out)
                })?;

                Ok(res
                    .into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::File => {
                // Parallel buffered file reads
                let uris_clone = uris.clone();
                let res = submit_io(async move {
                    let sem = Arc::new(Semaphore::new(clamp_in_flight(max_in_flight)));
                    let mut futs = FuturesUnordered::new();

                    for uri in uris_clone.iter().cloned() {
                        let sem = Arc::clone(&sem);
                        futs.push(tokio::spawn(async move {
                            let _permit = sem.acquire_owned().await.unwrap();
                            let path = uri.strip_prefix("file://").unwrap_or(&uri);
                            let file_bytes = tokio::fs::read(path).await?;
                            Ok::<_, std::io::Error>((uri, Bytes::from(file_bytes)))
                        }));
                    }

                    let mut out = Vec::new();
                    while let Some(res) = futs.next().await {
                        let (uri, bytes) = res
                            .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
                            .map_err(|e| anyhow::anyhow!("File read error: {}", e))?;
                        out.push((uri, bytes));
                    }

                    // Maintain input order
                    out.sort_by_key(|(u, _)| uris_clone.iter().position(|x| x == u).unwrap());
                    Ok::<_, anyhow::Error>(out)
                })?;

                Ok(res
                    .into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::Azure | Scheme::Gcs => {
                // Every Azure/GCS store's `get` takes the FULL object URI (#242), so each read
                // is handed the URI as written. Submit to global runtime (io_uring pattern —
                // never calls block_on).
                let logger = global_logger();
                let res = submit_io(get_many_cached_stores(uris.clone(), max_in_flight, logger))?;
                Ok(res
                    .into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::Unknown => Err(PyRuntimeError::new_err(format!(
                "Unsupported URI scheme for get_many: {}",
                uris.first().unwrap_or(&String::new())
            ))),
        }
    })
}

// --- `delete` function
//
/// Delete an object, or with ``recursive=True`` every object under a prefix
/// (delete contract A, issue #212; the same contract as ``s3-cli delete``).
///
/// ``delete(uri)`` sends ONE delete request for exactly the key named by
/// ``uri`` and returns ``None``:
///
/// * There is no lookup first. Deleting an object that is already missing
///   succeeds (S3's DELETE succeeds for a missing key; on other backends only
///   the backend's own typed not-found error is treated as success). Success
///   means the delete request succeeded, not that an existing object was
///   removed: S3 does not say whether the key existed, and on a versioned
///   bucket a delete adds a delete marker and erases no earlier version.
/// * Any actual failure (permission denied, server error, network error, ...)
///   raises ``RuntimeError`` naming the key; it is never read as "already
///   absent".
/// * ``/``, ``*`` and ``?`` are ordinary key characters: ``s3://b/dir/`` names
///   the object ``dir/`` (not "everything under dir"), and ``s3://b/a*`` names
///   the key ``a*``. Nothing is listed, globbed or expanded, no parent is
///   consulted, and no ``/`` is added.
///
/// ``delete(uri, recursive=True)`` deletes the real objects the ``s3://`` URI
/// selects (issue #213): listed without a delimiter, objects only (a
/// "directory" is never a deletion target). ``foo`` selects the object ``foo``
/// and everything under ``foo/`` and NEVER ``foobar`` or ``foo.bak``; ``foo/``
/// selects everything under it; ``*`` makes an explicit glob over the full key
/// (``*`` stays in one directory level, ``**`` crosses ``/``), so ``foo*`` selects
/// ``foo``, ``foo.bak`` and ``foobar``. A selection with no objects is a no-op. Each selected object gets
/// its own delete request (several in flight at once), so the op-log records
/// one ``LIST`` row and one ``DELETE`` row per object. If a delete request or
/// the listing fails, no further delete request is sent (requests already in
/// flight finish; keys listed but not yet sent are not sent) and
/// ``RuntimeError`` names what was and was not attempted.
///
/// Other backends follow the same contract through the same store methods; on
/// a backend without a literal-prefix listing, ``recursive=True`` requires a
/// URI ending in ``/`` (otherwise ``RuntimeError``, nothing deleted).
///
/// ``recursive=True`` needs list permission in addition to delete. This
/// replaced an earlier implicit expansion of a trailing ``/``, ``*`` or ``?``
/// (issue #212).
///
/// Raises:
///     RuntimeError: any failure. For ``recursive=True``, nothing is deleted
///         when the first listing page fails.
#[pyfunction]
#[pyo3(signature = (uri, recursive = false))]
pub fn delete(py: Python<'_>, uri: &str, recursive: bool) -> PyResult<()> {
    // Use universal ObjectStore API for deletion (with op-log support)
    let uri_owned = uri.to_owned();
    py.detach(|| {
        // Build the store on the caller thread BEFORE submit_io (issue #173).
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;

        // Submit to global runtime (io_uring pattern — never calls block_on).
        // The selection and deletion are the library's `delete_contract`,
        // shared with `s3-cli delete`.
        submit_io(async move {
            if recursive {
                crate::delete_contract::delete_objects_under_literal_prefix(
                    store.as_ref(),
                    &uri_owned,
                    None,
                    // One DELETE per object (as before #212): the op-log
                    // records LIST + one DELETE row per object. Batching is
                    // deferred.
                    crate::delete_contract::DeleteMode::PerObject,
                    None,
                )
                .await
                .into_result(&uri_owned, None)
            } else {
                crate::delete_contract::delete_exact_object(store.as_ref(), &uri_owned).await
            }
        })
    })
}

#[pyfunction]
#[pyo3(signature = (src_patterns, dest_prefix,
                   max_in_flight = 32, create_bucket = false))]
pub fn upload(
    py: Python<'_>,
    src_patterns: Vec<String>,
    dest_prefix: &str,
    max_in_flight: usize,
    create_bucket: bool,
) -> PyResult<()> {
    require_max_in_flight("max_in_flight", max_in_flight)?;
    let paths: Vec<PathBuf> = src_patterns.into_iter().map(PathBuf::from).collect();
    let dest_prefix_owned = dest_prefix.to_owned();
    py.detach(|| {
        // Get logger if op-log is active
        let logger = global_logger();

        // The bucket-creation store (only when explicitly requested) is built
        // on the caller thread BEFORE submit_io (issue #173): a host-form S3
        // URI creates its client via `run_on_global_rt`, which must not be
        // reached from a global-runtime worker.  Its failure is still only
        // warned about, inside the future, exactly as before.
        let bucket_store =
            create_bucket.then(|| store_for_uri_with_logger(&dest_prefix_owned, logger));

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            // Handle bucket creation ONLY if explicitly requested
            if let Some(bucket_store) = bucket_store {
                match bucket_store {
                    Ok(store) => {
                        // Extract bucket/container name from URI
                        if dest_prefix_owned.starts_with("s3://") {
                            match parse_s3_uri(&dest_prefix_owned) {
                                Ok((bucket, _)) => {
                                    if let Err(e) = store.create_container(&bucket).await {
                                        warn!("Failed to create bucket {}: {}", bucket, e);
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to parse S3 URI {}: {}", dest_prefix_owned, e)
                                }
                            }
                        } else if dest_prefix_owned.starts_with("az://")
                            || dest_prefix_owned.starts_with("azure://")
                        {
                            // For Azure, extract container name
                            let parts: Vec<&str> = dest_prefix_owned
                                .trim_start_matches("az://")
                                .trim_start_matches("azure://")
                                .split('/')
                                .collect();
                            if let Some(container) = parts.first() {
                                if let Err(e) = store.create_container(container).await {
                                    warn!("Failed to create container {}: {}", container, e);
                                }
                            }
                        }
                        // File backends don't need bucket creation, directories are created automatically
                    }
                    Err(e) => warn!(
                        "Failed to open store for bucket creation at {}: {}",
                        dest_prefix_owned, e
                    ),
                }
            }

            // Use generic upload that works with all backends
            generic_upload_files(&dest_prefix_owned, &paths, max_in_flight, None).await
        })
    })
}

// ---------------------------------------------------------------------------
// put_many() - Batch put for maximum throughput (Python API)
// ---------------------------------------------------------------------------

/// The payloads of a `put_many` / `put_many_async` batch as `Bytes`, converted exactly like
/// `put_bytes` payloads (issue #272): `bytes`, any buffer-protocol object and any object with
/// `__bytes__()` are accepted; an int, a bool or an arbitrary iterable (a list of ints) is a
/// `TypeError`. Every item is converted before the caller sends anything, so one bad item means no
/// request at all.
///
/// Rule #1 (zero copy): with `PayloadUse::Blocking` (`put_many`) or `Borrowed`
/// (`put_many_async(borrow=True)`) a buffer export is HELD, not copied, and counted on ONE shared
/// batch latch; the caller waits on that barrier before it returns (or resolves the awaitable).
/// With `SnapshotUnderGil` (`put_many_async(borrow=False)`, the opt-in) a mutable buffer is copied
/// once with the GIL held. Exact `bytes`, `BytesView` and other immutable sources are never copied.
struct BatchPayloads {
    items: Vec<(String, Bytes)>,
    /// `Some` only when at least one item holds a buffer export.
    latch: Option<Arc<Latch>>,
}

fn batch_payloads(
    py: Python<'_>,
    items: Vec<(String, Bound<'_, PyAny>)>,
    usage: PayloadUse,
) -> PyResult<BatchPayloads> {
    let batch = Latch::new_batch();
    let mut converted: Vec<(String, Bytes)> = Vec::with_capacity(items.len());
    for (index, (uri, data)) in items.into_iter().enumerate() {
        match put_payload_in_batch(py, &data, usage, &batch) {
            Ok(bytes) => converted.push((uri, bytes)),
            Err(e) => {
                let e = error_with_context(py, e, &format!("item {index} ({})", redact_uri(&uri)));
                // Nothing is sent. Release the exports already taken: dropping their `Bytes`
                // queues them, and the barrier drains and waits for the release.
                drop(converted);
                batch.batch_seal();
                if quiesce_blocking(py, &batch).is_err() {
                    return Err(still_referenced_error(&format!(
                        "the batch was rejected: {e}"
                    )));
                }
                return Err(e);
            }
        }
    }
    batch.batch_seal();
    let latch = (batch.batch_len() > 0).then_some(batch);
    Ok(BatchPayloads {
        items: converted,
        latch,
    })
}

/// `e` with `context` in front of its message, for a failure in one item of a batch.
///
/// Only an exact builtin class that a single string rebuilds faithfully (`TypeError`,
/// `ValueError`, `MemoryError`, `BufferError`, `OverflowError`, ... see [`REBUILDABLE`]), with
/// exactly one `str` argument and no extra state, is rebuilt as the same class with the message
/// `"{context}: {original}"`, and `e` is kept as its `__cause__`. Everything else is returned
/// UNCHANGED, the same object, with no constructor of its class ever called: a subclass (its
/// `__str__` and `__init__` are its own), an exception with fields (`ImportError(name=...)`,
/// `AttributeError(name=..., obj=...)`, `StopIteration`), `KeyError` (whose `str` is a `repr`),
/// `OSError(errno, msg)`, `UnicodeDecodeError`, `KeyboardInterrupt`, `SystemExit`. The context
/// then goes onto the original as a note (`add_note`, Python 3.11+; skipped on older versions).
fn error_with_context(py: Python<'_>, e: PyErr, context: &str) -> PyErr {
    match rebuilt_with_context(py, &e, context) {
        Some(wrapped) => {
            wrapped.set_cause(py, Some(e));
            wrapped
        }
        None => {
            let value = e.value(py);
            if value.hasattr("add_note").unwrap_or(false) {
                // A note is a convenience: if it cannot be added the error still stands.
                let _ = value.call_method1("add_note", (context,));
            }
            e
        }
    }
}

/// Builtin exception classes whose whole state is one message.
const REBUILDABLE: &[&str] = &[
    "Exception",
    "TypeError",
    "ValueError",
    "MemoryError",
    "BufferError",
    "OverflowError",
    "ArithmeticError",
    "ZeroDivisionError",
    "FloatingPointError",
    "RuntimeError",
    "NotImplementedError",
    "RecursionError",
    "LookupError",
    "IndexError",
    "EOFError",
    "AssertionError",
    "ReferenceError",
    "SystemError",
];

/// The same-class error with `context` in front, or `None` when `e` is not a plain
/// single-message builtin exception (see [`error_with_context`]).
fn rebuilt_with_context(py: Python<'_>, e: &PyErr, context: &str) -> Option<PyErr> {
    use pyo3::exceptions::PyBaseException;
    use pyo3::types::{PyString, PyTuple};
    let value = e.value(py);
    let class = e.get_type(py);
    // An exact builtin class: `type(e).__module__ == "builtins"` and `type(e)` is the very
    // object `builtins.<name>` (a look-alike class that only claims the module is not).
    let name = class.name().ok()?.to_string();
    let builtin = py.import("builtins").ok()?.getattr(name.as_str()).ok()?;
    if !REBUILDABLE.contains(&name.as_str())
        || !builtin.is(&class)
        || class.getattr("__module__").ok()?.extract::<String>().ok()? != "builtins"
    {
        return None;
    }
    let args = value.getattr("args").ok()?.cast_into::<PyTuple>().ok()?;
    if args.len() != 1 || !args.get_item(0).ok()?.is_instance_of::<PyString>() {
        return None;
    }
    // Extra state (a custom attribute, an earlier note) would be lost by a rebuild.
    let has_state = value
        .getattr("__dict__")
        .ok()
        .and_then(|d| d.len().ok())
        .is_none_or(|n| n != 0);
    if has_state {
        return None;
    }
    let message = value.str().ok()?.to_string();
    let rebuilt = class.call1((format!("{context}: {message}"),)).ok()?;
    if !rebuilt.get_type().is(&class) || !rebuilt.is_instance_of::<PyBaseException>() {
        return None;
    }
    // The rebuilt error must really carry the context.
    if !rebuilt.str().ok()?.to_string().contains(context) {
        return None;
    }
    Some(PyErr::from_value(rebuilt))
}

/// One `put_many` item with its store already resolved (or the resolution
/// error, surfaced from that item's task): (uri, data, store).
type PreparedPut = (String, Bytes, anyhow::Result<Arc<dyn ObjectStore>>);

/// The largest `max_in_flight` a batch can use: tokio's semaphore panics above its own maximum,
/// so a larger request (for example `2**62`) is clamped to it, which means "no limit".
fn clamp_in_flight(max_in_flight: usize) -> usize {
    max_in_flight.min(tokio::sync::Semaphore::MAX_PERMITS)
}

/// Run the puts of a batch with at most `max_in_flight` of them running at once.
///
/// **Ordered admission.** Every item's request for a permit is registered with the (FIFO) semaphore
/// IN LIST ORDER, before any item runs: the batch polls item `i`'s `acquire` once, in order, and
/// only then spawns item `i`'s task, which owns that already-queued request. Permits are therefore
/// granted to item `i` before item `i + 1`, whichever thread later runs the tasks. (Spawning every
/// task and letting each one ask for a permit when it first runs does not keep that order:
/// tokio's scheduling of tasks spawned from a worker thread is not FIFO.) `max_in_flight` above
/// tokio's maximum is clamped ([`clamp_in_flight`]).
///
/// **Why the requests are not made by the batch task itself.** The obvious ordered design - the
/// batch task takes a permit, spawns item `i`, takes the next - was measured 24% slower than the
/// unordered one for large objects (`put_many` of 16 x 256 MiB, 4 in flight: 2.3 against 3.1 GB/s
/// on loopback). A signed S3 PUT hashes the whole body (SigV4 payload SHA-256, about 160 ms for
/// 256 MiB) inside ONE poll of the request future, so a worker is blocked for that long. When
/// several items finish together, the single batch task is woken once and then spawns all the
/// replacements from ONE worker: they sit in that worker's queue and LIFO slot (a LIFO slot is
/// never stolen) behind each other's hashing, and the other workers stay idle. Here each item
/// task is woken by its OWN permit, released by a different finishing item on a different
/// worker, so the replacements start on the workers that just finished, side by side.
///
/// **Failure.** A failing item closes the semaphore BEFORE it gives its permit back, so no item
/// that has not been admitted yet can start after that failure: the rule is exact (a request
/// still queued fails with "closed", and an item that holds a permit but has not begun checks the
/// semaphore first). Items already running keep running only until the batch notices: it then
/// aborts every task that has not finished, so each payload (which may be a held buffer export)
/// is dropped at once, and returns the FIRST failure once the aborted tasks are reaped. Only a
/// write that already runs in a blocking thread cannot be stopped, and only those are waited for
/// by the completion barrier of the call. Items that finished before the failure stay done (the
/// batch is not transactional), and the FIFO admission above orders STARTS only: with more than
/// one permit, items finish in any order. The failure returned is the item's own error, not a
/// barrier error, whenever the drain finishes within `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`.
///
/// A `put` that PANICS is a failure too: the item's task closes the semaphore from a drop guard
/// while it unwinds, before its permit is released, and the batch returns the panic (as `join_failed`
/// makes it) once it reaps the task.
///
/// **Cancellation.** Dropping the returned future (cancelling the awaitable of
/// `put_many_async`) closes the semaphore first and then aborts the tasks (`JoinSet` aborts on
/// drop): nothing further is admitted.
async fn run_bounded_puts<E, Fut>(
    puts: Vec<Fut>,
    max_in_flight: usize,
    join_failed: fn(tokio::task::JoinError) -> E,
) -> Result<(), E>
where
    E: Send + 'static,
    Fut: Future<Output = Result<(), E>> + Send + 'static,
{
    use futures::FutureExt;
    use std::pin::Pin;
    use tokio::sync::{AcquireError, OwnedSemaphorePermit};

    /// Closes the semaphore when dropped, so that a dropped batch (cancellation) stops handing
    /// permits out BEFORE its `JoinSet` aborts the tasks: otherwise a permit released by the
    /// first aborted task could let the next, not yet aborted, task start its write.
    struct CloseOnDrop(Arc<tokio::sync::Semaphore>);
    impl Drop for CloseOnDrop {
        fn drop(&mut self) {
            self.0.close();
        }
    }

    /// Closes the semaphore when dropped WHILE UNWINDING (a `put` that panicked); a normal drop
    /// does nothing, so a successful or failed-with-`Err` item is handled by its own code.
    struct ClosePanicking(Arc<tokio::sync::Semaphore>);
    impl Drop for ClosePanicking {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0.close();
            }
        }
    }

    type Granted = Result<OwnedSemaphorePermit, AcquireError>;

    /// An item's permit request: already granted when it was registered, or queued (and so
    /// already in its place in the semaphore's FIFO) until a permit is released.
    enum Admission {
        Granted(Granted),
        Queued(Pin<Box<dyn Future<Output = Granted> + Send>>),
    }

    let sem = Arc::new(tokio::sync::Semaphore::new(clamp_in_flight(max_in_flight)));
    let mut tasks = tokio::task::JoinSet::new();
    // Declared AFTER `tasks`, so on a drop (cancellation) it runs FIRST: the semaphore is closed
    // before the JoinSet aborts its tasks.
    let stop = CloseOnDrop(Arc::clone(&sem));

    for put in puts {
        if sem.is_closed() {
            // A failure already ended the batch: nothing more is admitted (the rest of `puts`
            // is dropped here, releasing its payloads).
            break;
        }
        // Register this item's permit request NOW, in list order: the first poll of `acquire`
        // takes a free permit or joins the semaphore's FIFO queue. The no-op waker of this
        // single poll is replaced by the task's own waker when the task polls it again.
        // `unconstrained`: tokio's cooperative budget (128 operations per poll of THIS task)
        // would otherwise make a poll after the budget is spent return `Pending` WITHOUT joining
        // the queue, which breaks the order for a limit above 128.
        let mut acquire: Pin<Box<dyn Future<Output = Granted> + Send>> =
            Box::pin(tokio::task::unconstrained(Arc::clone(&sem).acquire_owned()));
        let admission = match acquire.as_mut().now_or_never() {
            Some(granted) => Admission::Granted(granted),
            None => Admission::Queued(acquire),
        };
        let task_sem = Arc::clone(&sem);
        tasks.spawn(async move {
            let granted = match admission {
                Admission::Granted(granted) => granted,
                Admission::Queued(queued) => queued.await,
            };
            // `Err`: the semaphore was closed while this item waited - a failure ended the
            // batch, this item does not start, and it is not a failure of its own (the real
            // one is reported).
            let Ok(permit) = granted else {
                return Ok(());
            };
            if task_sem.is_closed() {
                // The same, for a permit granted just before the close.
                return Ok(());
            }
            // Declared AFTER `permit`, so it drops BEFORE it: if `put` panics, the unwinding
            // closes the semaphore first and only then gives the permit back. Without it the
            // permit would go to the next queued item, which could start before the batch has
            // reaped this task's `JoinError` and closed the semaphore.
            let _close_on_panic = ClosePanicking(Arc::clone(&task_sem));
            let outcome = put.await;
            if outcome.is_err() {
                // Closed BEFORE the permit is given back: no later item can start.
                task_sem.close();
            }
            drop(permit);
            outcome
        });
    }

    // Everything is admitted (or admission stopped on a closed semaphore): collect the results,
    // and stop at the first failure.
    let mut first_failure: Option<E> = None;
    while let Some(joined) = tasks.join_next().await {
        if let Err(e) = joined.unwrap_or_else(|e| Err(join_failed(e))) {
            first_failure = Some(e);
            break;
        }
    }
    if first_failure.is_some() {
        sem.close();
        tasks.abort_all();
    }
    // Reap the aborted tasks: their futures (and so their payloads) are dropped before we return.
    while tasks.join_next().await.is_some() {}
    drop(stop);
    match first_failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Put multiple objects in parallel from Python.
///
/// Accepts a list of (uri, data) tuples where data is Python bytes.
/// All put operations execute concurrently inside Rust's async runtime
/// using FuturesUnordered + Semaphore for backpressure — similar to
/// how get_objects_parallel works for reads.
///
/// # Arguments
/// * `items` - List of (uri, data) tuples: uri is a string, data is `bytes`, any buffer-protocol
///   object (bytearray, memoryview, numpy array, ...) or an object with `__bytes__()` - the same
///   payloads `put_bytes` accepts. An int, a bool or an arbitrary iterable is refused with a
///   `TypeError` before any request is sent (issue #272)
/// * `max_in_flight` - Maximum concurrent put operations (default: 64); must be
///   an integer >= 1, otherwise `ValueError` is raised at once (issue #211). A value above the
///   largest the runtime supports (about 2**61) is clamped to it, which means "no limit".
///
/// # Zero-copy path
/// No payload is copied. The call blocks until every object is written, so each buffer is
/// HELD (exported) and read in place, exactly like `put_bytes`: while the call runs, a
/// `bytearray` cannot be resized (`BufferError`) and no thread may modify any of the buffers (a
/// data race otherwise). The call does not return - normally or with an error - until every
/// native reader of every buffer is done and the exports are released (bounded by
/// `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`; on expiry `ExportStillReferencedError` is raised and the
/// buffers stay pinned), so every buffer is yours again as soon as the call returns. Exact
/// `bytes` and `BytesView` payloads are immutable and are never exported.
///
/// # Failure and partial success
/// Items are ADMITTED in list order (first in, first out), so the first `max_in_flight` items
/// start first; with `max_in_flight` above 1 that is the order they start in, not the order
/// they finish in. The first failed item ends the batch: no item that has not been started is
/// started after that failure (the failing item closes the batch before it frees its slot), the
/// items still queued are dropped (releasing their buffers at once), every item still in flight
/// is ABORTED (its payload is dropped at once, so its request may never complete), and that
/// item's own error is raised. Only a write that already runs on a blocking thread cannot be
/// stopped: the call waits for it, and for any other native reader of your buffers, before it
/// returns (it never returns while a write still reads one of your buffers).
///
/// The batch is NOT transactional. Objects whose put completed before the failure stay stored;
/// an item that was aborted in flight may be absent, complete or (on a store whose put is not
/// atomic, such as `file://`) partial, and you cannot tell which from the error; items that
/// never started were not written. After an error, re-check or re-put every object that is not
/// known to be complete.
///
/// # Examples
/// ```python
/// import s3dlio
///
/// items = [
///     ("s3://bucket/obj1.bin", b"data1"),
///     ("s3://bucket/obj2.bin", b"data2"),
///     ("s3://bucket/obj3.bin", b"data3"),
/// ]
/// s3dlio.put_many(items, max_in_flight=64)
/// ```
#[pyfunction]
#[pyo3(signature = (items, max_in_flight = 64))]
pub fn put_many(
    py: Python<'_>,
    items: Vec<(String, Bound<'_, PyAny>)>,
    max_in_flight: usize,
) -> PyResult<()> {
    require_max_in_flight("max_in_flight", max_in_flight)?;

    // Every payload is converted (strictly, like `put_bytes`) on the Python thread BEFORE any
    // request is sent. A buffer is held (no copy), like `put_bytes`: this call blocks until the
    // objects are written, so the exports outlive every reader (the barrier below).
    let BatchPayloads {
        items: owned_items,
        latch,
    } = batch_payloads(py, items, PayloadUse::Blocking)?;

    let result = py.detach(move || {
        // Build every store on the caller thread BEFORE submit_io (issue
        // #173): a host-form S3 URI creates its client via
        // `run_on_global_rt`, which must not be reached from a global-runtime
        // worker.  Stores are cached by (scheme, endpoint, region), so after
        // the first URI of a kind these are cache hits.  A creation failure
        // is carried per item and surfaces from that item's task, so the
        // error semantics (other items still run; first error returned) are
        // unchanged.
        let logger = global_logger();
        let prepared: Vec<PreparedPut> = owned_items
            .into_iter()
            .map(|(uri, data)| {
                let store = get_or_create_store(&uri, logger.clone());
                (uri, data, store)
            })
            .collect();

        // Submit entire batch to global runtime (io_uring pattern)
        submit_io(async move {
            let puts = prepared.into_iter().map(|(uri, data, store)| async move {
                let store = store?;
                store.put(&uri, data).await
            });
            run_bounded_puts(puts.collect(), max_in_flight, |e| {
                anyhow::anyhow!("Task join error: {e}")
            })
            .await
        })
    });
    // The batch is over (it succeeded, failed, or gave up at the first error), but a spawned
    // item may still hold a clone of its payload. Wait until the LAST clone of EVERY held export
    // is gone, then release them with this thread's real GIL token (see `put_bytes`).
    if let Some(latch) = &latch {
        if quiesce_blocking(py, latch).is_err() {
            return Err(still_referenced_error(&put_outcome_text(&result)));
        }
    }
    result
}

/// Async version of put_many()
///
/// `borrow=True` (the DEFAULT) reads every mutable buffer in place, with no copy: the buffers
/// are held until the awaitable completes. By default you PROMISE not to modify, resize or free
/// any of them until it has completed (successfully or not); until then a `bytearray` raises
/// `BufferError` on resize and any write to the bytes is a data race (the bytes uploaded are
/// undefined). Like `put_bytes_async(borrow=True)` the awaitable never completes - result,
/// error or cancelled - while a native reader remains or before every export is released.
/// `borrow=False` is the opt-in snapshot: each mutable buffer is copied ONCE now with the GIL
/// held, so you may reuse the buffers immediately. Exact `bytes`, `BytesView` and other
/// immutable payloads are never copied in either mode.
///
/// Items are ADMITTED in list order (first in, first out), so the first `max_in_flight` items
/// start first; with `max_in_flight` above 1 that is the order they start in, not the order
/// they finish in. The first failed item ends the batch: no item that has not been started is
/// started after that failure, queued items are dropped at once (releasing their buffers),
/// every item still in flight is ABORTED (its payload is dropped at once, so its request may
/// never complete), and that item's own error is raised. Only a write that already runs on a
/// blocking thread cannot be stopped: the awaitable completes only after it (and any other
/// native reader of your buffers) is done. The batch is NOT transactional: objects whose put
/// completed before the failure stay stored; an item aborted in flight may be absent, complete
/// or (on a store whose put is not atomic, such as `file://`) partial; items that never started
/// were not written. Cancelling the awaitable stops the batch the same way: no item that has
/// not started is uploaded, items in flight are aborted, and the cancellation completes once
/// the native readers that survive cancellation are done.
#[pyfunction]
#[pyo3(signature = (items, max_in_flight = 64, *, borrow = true))]
pub fn put_many_async<'py>(
    py: Python<'py>,
    items: Vec<(String, Bound<'py, PyAny>)>,
    max_in_flight: usize,
    borrow: bool,
) -> PyResult<Bound<'py, PyAny>> {
    require_max_in_flight("max_in_flight", max_in_flight)?;

    let usage = if borrow {
        PayloadUse::Borrowed
    } else {
        PayloadUse::SnapshotUnderGil
    };
    let BatchPayloads {
        items: owned_items,
        latch,
    } = batch_payloads(py, items, usage)?;

    guarded_put(py, latch, move || async move {
        let puts = owned_items.into_iter().map(|(uri, data)| async move {
            let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;
            store.put(&uri, data).await.map_err(py_err)?;
            Ok::<(), PyErr>(())
        });
        run_bounded_puts(puts.collect(), max_in_flight, py_err).await?;
        // Python `None`, converted inside the exit-safe delivery (issue #271; see `mkdir_async`).
        Ok(None::<()>)
    })
}
// ---------------------------------------------------------------------------
// Phase 2 Streaming API Classes
// ---------------------------------------------------------------------------

/// Python wrapper for WriterOptions
#[pyclass]
pub struct PyWriterOptions {
    inner: WriterOptions,
}

#[pymethods]
impl PyWriterOptions {
    #[new]
    fn new() -> Self {
        Self {
            inner: WriterOptions::new(),
        }
    }

    /// Set compression for the writer
    #[pyo3(signature = (compression_type, level = None))]
    fn with_compression(&mut self, compression_type: &str, level: Option<i32>) -> PyResult<()> {
        let compression = match compression_type.to_lowercase().as_str() {
            "zstd" => {
                let level = level.unwrap_or(3); // Default zstd compression level
                CompressionConfig::Zstd { level }
            }
            "none" => CompressionConfig::None,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported compression type: {}",
                    compression_type
                )))
            }
        };

        self.inner = self.inner.clone().with_compression(compression);
        Ok(())
    }

    /// Set buffer size for the writer
    fn with_buffer_size(&mut self, size: usize) {
        self.inner = self.inner.clone().with_buffer_size(size);
    }
}

/// Snapshot a writer's (bytes_written, compressed_bytes, checksum) via
/// shared reference, before any consuming call (e.g. `finalize()`, which
/// takes `self: Box<Self>`). Safe to call at this point because
/// `ObjectWriter::checksum()` is documented and implemented as an
/// incremental "written so far" rolling-hash value, not something only
/// computed inside `finalize()` — see docs/DESIGN_FFI_BOUNDARY_HARDENING.md
/// §4.1 for the verified evidence. If a future `ObjectWriter` impl ever
/// violated that (computed checksum only during `finalize()`), this
/// snapshot would silently capture `None` — `checksum_survives_finalize_tests`
/// below guards the current contract.
fn snapshot_final_writer_stats(writer: &dyn ObjectWriter) -> (u64, u64, Option<String>) {
    (
        writer.bytes_written(),
        writer.compressed_bytes(),
        writer.checksum(),
    )
}

/// (bytes_written, compressed_bytes, checksum) of a writer.
type WriterStats = (u64, u64, Option<String>);

/// Lock a mutex whose data stays consistent even if a holder panicked.
fn lock_ignoring_poison<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Python wrapper for ObjectWriter
///
/// `inner` is locked for the whole of a write or of `finalize()`, so calls on one writer
/// from several threads run one after another instead of raising. `stats` holds the
/// numbers as of the last COMPLETED call and is never held across I/O, so the accessors
/// (`bytes_written`, `checksum`, `compressed_bytes`) answer immediately, even while a write
/// is in flight on another thread (issue #201: the GIL is released during writes).
#[pyclass]
pub struct PyObjectWriter {
    inner: std::sync::Mutex<Option<Box<dyn ObjectWriter>>>,
    stats: std::sync::Mutex<WriterStats>,
}

impl PyObjectWriter {
    fn new(writer: Box<dyn ObjectWriter>) -> Self {
        let stats = snapshot_final_writer_stats(writer.as_ref());
        Self {
            inner: std::sync::Mutex::new(Some(writer)),
            stats: std::sync::Mutex::new(stats),
        }
    }

    fn finalized_error() -> PyErr {
        PyRuntimeError::new_err("Writer has been finalized")
    }

    fn publish_stats(&self, writer: &dyn ObjectWriter) {
        *lock_ignoring_poison(&self.stats) = snapshot_final_writer_stats(writer);
    }

    /// Write `data` through the locked writer and publish the new stats. Blocks the
    /// calling thread; the caller decides whether it holds the GIL.
    fn write_locked(
        &self,
        writer: &mut Option<Box<dyn ObjectWriter>>,
        data: &[u8],
    ) -> PyResult<()> {
        let writer = writer.as_mut().ok_or_else(Self::finalized_error)?;
        let result = pyo3_async_runtimes::tokio::get_runtime().block_on(async {
            writer
                .write_chunk(data)
                .await
                .map_err(|e| py_err(e.context("Failed to write chunk")))
        });
        self.publish_stats(writer.as_ref());
        result
    }

    /// Write `data` with the GIL released. `data` must stay valid and unchanged for the
    /// call without the GIL: an exported buffer the caller promised not to modify, or an
    /// immutable `bytes`.
    fn write_detached(&self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        py.detach(|| {
            let mut guard = lock_ignoring_poison(&self.inner);
            self.write_locked(&mut guard, data)
        })
    }
}

#[pymethods]
impl PyObjectWriter {
    /// Write a chunk of bytes to the stream, WITHOUT copying it.
    ///
    /// Accepts any Python object supporting the buffer protocol:
    /// - bytes, bytearray, memoryview
    /// - NumPy arrays (any dtype), PyTorch tensors, ctypes arrays, mmap
    /// - Any object with __buffer__ method
    ///
    /// CONTRACT - the call BORROWS your buffer: no copy is made by this Python layer for a
    /// C-contiguous buffer (a non-contiguous view is gathered once by Python, an object that
    /// only has `__bytes__()` is converted once), and the GIL is RELEASED for the whole blocking
    /// write. While the call runs the buffer is exported (resizing it from another thread raises
    /// `BufferError`) and NO thread may modify its bytes until the call returns: a concurrent
    /// write is a data race and the data written is undefined. The export is released before
    /// the call returns, so the buffer is yours again right after. Exact `bytes` and
    /// `s3dlio.BytesView` are immutable, so they can never race. (Before 0.9.128 the GIL was
    /// held for mutable buffers, which made other Python threads wait for the whole write.)
    ///
    /// Whether the writer then copies the data itself depends on the backend: the S3 and Azure
    /// writers buffer the whole object, so each chunk is copied once into that buffer (a
    /// documented exception; use `put_bytes` to upload a buffer with no copy), while the
    /// `file://` and `direct://` writers write straight from your buffer.
    ///
    /// Calls on one writer from several threads run one after another.
    fn write_chunk(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        // Strict: bytes-like or `__bytes__()` only (never an int or an arbitrary iterable).
        let held = hold_for_call(py, data)?;
        let result = self.write_detached(py, held.as_slice());
        // The write took a borrowed slice (nothing keeps a clone), so the export is free now:
        // release it directly with this thread's GIL token (no queue, no releaser wake-up).
        held.release(py);
        result
    }

    /// Write bytes the caller owns. BEHAVIOUR CHANGE in 0.9.128: this now BORROWS the buffer
    /// exactly like `write_chunk()`; in 0.9.126 and earlier it copied a mutable buffer (under
    /// the GIL) before writing. Pass `copy=True` to get the snapshot back.
    ///
    /// * `copy=False` (default): no copy; the buffer is exported and read in place with the GIL
    ///   released. No thread may modify the buffer until the call returns (a data race
    ///   otherwise; the data written is undefined); it is released before the call returns.
    /// * `copy=True`: a mutable buffer (bytearray, memoryview, NumPy, ...) is first copied ONCE
    ///   with the GIL held, the export is released, and the write then runs on the copy, so you
    ///   may modify or reuse your buffer as soon as the call has started. The snapshot is atomic
    ///   only against writers that HOLD the GIL (pure-Python code in other threads). A writer
    ///   that releases the GIL (a large NumPy operation, `readinto`/`recv_into`, a CUDA or torch
    ///   copy into pinned memory) can still tear it: serialise such writers yourself. Exact
    ///   `bytes` and `s3dlio.BytesView` are immutable: they are never copied, in either mode.
    ///
    /// The writer is put back after a failed write, so a failed `write_owned_bytes()`
    /// leaves it usable for a retry or a clean `finalize()`.
    #[pyo3(signature = (data, *, copy = false))]
    fn write_owned_bytes(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<()> {
        if copy {
            // Snapshot first (GIL held, export released), then write the private copy.
            let snapshot = snapshot_for_call(py, data)?;
            return self.write_detached(py, &snapshot);
        }
        self.write_chunk(py, data)
    }

    /// Finalize the writer and complete the upload
    fn finalize(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        py.detach(|| {
            let mut guard = lock_ignoring_poison(&self.inner);
            let writer = guard
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("Writer already finalized"))?;
            let stats = submit_io(async move {
                let stats = snapshot_final_writer_stats(writer.as_ref());
                writer.finalize().await?;
                Ok::<WriterStats, anyhow::Error>(stats)
            })?;
            let result = (stats.0, stats.1);
            *lock_ignoring_poison(&self.stats) = stats;
            Ok(result)
        })
    }

    /// Get the number of bytes written so far (as of the last completed write; it does
    /// not wait for a write in flight on another thread)
    fn bytes_written(&self) -> PyResult<u64> {
        Ok(lock_ignoring_poison(&self.stats).0)
    }

    /// Get the checksum of the data written (if available; see `bytes_written` for when
    /// it is read)
    fn checksum(&self) -> Option<String> {
        lock_ignoring_poison(&self.stats).2.clone()
    }

    /// Get the number of compressed bytes (if compression is enabled; see
    /// `bytes_written` for when it is read)
    fn compressed_bytes(&self) -> PyResult<u64> {
        Ok(lock_ignoring_poison(&self.stats).1)
    }
}

/// Create a streaming writer for S3
#[pyfunction]
#[pyo3(signature = (uri, options = None))]
pub fn create_s3_writer(
    py: Python<'_>,
    uri: String,
    options: Option<&PyWriterOptions>,
) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

    py.detach(|| {
        submit_io(async move {
            let store = S3ObjectStore::new();
            let writer = store
                .create_writer(&uri, opts)
                .await
                .map_err(|e| e.context("Failed to create S3 writer"))?;

            Ok(PyObjectWriter::new(writer))
        })
    })
}

/// Create a streaming writer for Azure Blob Storage
#[pyfunction]
#[cfg(feature = "backend-azure")]
#[pyo3(signature = (uri, options = None))]
pub fn create_azure_writer(
    py: Python<'_>,
    uri: String,
    options: Option<&PyWriterOptions>,
) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

    py.detach(|| {
        submit_io(async move {
            let store = AzureObjectStore::new();
            let writer = store
                .create_writer(&uri, opts)
                .await
                .map_err(|e| e.context("Failed to create Azure writer"))?;

            Ok(PyObjectWriter::new(writer))
        })
    })
}

/// Create a streaming writer for Azure Blob Storage
#[pyfunction]
#[cfg(not(feature = "backend-azure"))]
#[pyo3(signature = (uri, options = None))]
// Accepted and ignored: this build has no Azure backend; the names must match the real build's Python keywords.
#[allow(unused_variables)]
pub fn create_azure_writer(
    uri: String,
    options: Option<&PyWriterOptions>,
) -> PyResult<PyObjectWriter> {
    Err(PyRuntimeError::new_err(
        "Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'."
    ))
}

/// Create a streaming writer for filesystem
#[pyfunction]
#[pyo3(signature = (uri, options = None))]
pub fn create_filesystem_writer(
    py: Python<'_>,
    uri: String,
    options: Option<&PyWriterOptions>,
) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

    py.detach(|| {
        submit_io(async move {
            let store = FileSystemObjectStore::new();
            let writer = store
                .create_writer(&uri, opts)
                .await
                .map_err(|e| e.context("Failed to create filesystem writer"))?;

            Ok(PyObjectWriter::new(writer))
        })
    })
}

/// Create a streaming writer for direct I/O filesystem
#[pyfunction]
#[pyo3(signature = (uri, options = None))]
pub fn create_direct_filesystem_writer(
    py: Python<'_>,
    uri: String,
    options: Option<&PyWriterOptions>,
) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

    py.detach(|| {
        submit_io(async move {
            let store = ConfigurableFileSystemObjectStore::new(Default::default());
            let writer = store
                .create_writer(&uri, opts)
                .await
                .map_err(|e| e.context("Failed to create direct filesystem writer"))?;

            Ok(PyObjectWriter::new(writer))
        })
    })
}

// ---------------------------------------------------------------------------
// Multi-Endpoint Support (v0.9.14+)
// ---------------------------------------------------------------------------

use crate::multi_endpoint::{LoadBalanceStrategy, MultiEndpointStore};
use crate::uri_utils;

/// The payload of a `MultiEndpointStore` put: a held export when `borrow` (zero copy, completion
/// barrier in `Payload::latch`), otherwise the opt-in snapshot of a mutable buffer (taken with
/// the GIL held, like the streaming writers' `copy=True`). Immutable sources are never copied.
fn endpoint_payload(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    borrow: bool,
    uri: &str,
) -> PyResult<super::put_payload::Payload> {
    let usage = if borrow {
        PayloadUse::Borrowed
    } else {
        PayloadUse::SnapshotUnderGil
    };
    put_payload(py, data, usage)
        .map_err(|e| error_with_context(py, e, &format!("payload for {}", redact_uri(uri))))
}

/// Python wrapper for MultiEndpointStore
#[pyclass(name = "MultiEndpointStore")]
struct PyMultiEndpointStore {
    store: Arc<MultiEndpointStore>,
}

#[pymethods]
impl PyMultiEndpointStore {
    /// Get an object from the multi-endpoint store (zero-copy via BytesView)
    fn get<'py>(&self, py: Python<'py>, uri: &str) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            let data = store
                .get(&uri)
                .await
                .map_err(|e| py_err(e.context("Get failed")))?;
            // Return zero-copy BytesView wrapper (maintains Arc-counted Bytes)
            Python::attach(|py| Ok(Py::new(py, PyBytesView::new(data))?.into_any()))
        })
    }

    /// Get a byte range from an object (zero-copy via BytesView)
    #[pyo3(signature = (uri, offset, length = None))]
    fn get_range<'py>(
        &self,
        py: Python<'py>,
        uri: &str,
        offset: u64,
        length: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            let data = store
                .get_range(&uri, offset, length)
                .await
                .map_err(|e| py_err(e.context("Get range failed")))?;
            // Return zero-copy BytesView wrapper (maintains Arc-counted Bytes)
            Python::attach(|py| Ok(Py::new(py, PyBytesView::new(data))?.into_any()))
        })
    }

    /// Put an object to the multi-endpoint store.
    ///
    /// `data` is converted like `put_bytes` payloads (issue #281): `bytes`, any buffer-protocol
    /// object or an object with `__bytes__()`; an int, a bool or an arbitrary iterable is a
    /// `TypeError` before any request.
    ///
    /// `borrow=True` (the DEFAULT): a mutable buffer is read in place, with no copy, and held
    /// until the awaitable completes. By default you PROMISE not to modify, resize or free it
    /// until the awaitable has completed (a `bytearray` raises `BufferError` on resize
    /// meanwhile; a write to the bytes is a data race). Like `put_bytes_async(borrow=True)` the
    /// awaitable never completes while a native reader remains. `borrow=False` is the opt-in
    /// snapshot: a mutable buffer is copied once, now, with the GIL held. Exact `bytes`,
    /// `BytesView` and other immutable payloads are never copied.
    #[pyo3(signature = (uri, data, *, borrow = true))]
    fn put<'py>(
        &self,
        py: Python<'py>,
        uri: &str,
        data: &Bound<'py, PyAny>,
        borrow: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let payload = endpoint_payload(py, data, borrow, &uri)?;
        let store = self.store.clone();

        guarded_put(py, payload.latch, move || async move {
            store
                .put(&uri, payload.bytes)
                .await
                .map_err(|e| py_err(e.context("Put failed")))?;
            // Python `None`, converted inside the exit-safe delivery (issue #271).
            Ok(None::<()>)
        })
    }

    /// List the objects a URI selects. On `s3://` the key part is an exact name (the object, and
    /// with `recursive` the keys under `name/`), a directory (`name/`) or a `*` / `**` glob --
    /// never a bare key prefix (`s3://b/foo` does not select `foobar`; use `s3://b/foo*`).
    fn list<'py>(
        &self,
        py: Python<'py>,
        prefix: &str,
        recursive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prefix = prefix.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            let objects = store
                .list(&prefix, recursive)
                .await
                .map_err(|e| py_err(e.context("List failed")))?;
            // Return Vec<String> - PyO3 automatically converts to Python list
            Ok(objects)
        })
    }

    /// Delete an object
    fn delete<'py>(&self, py: Python<'py>, uri: &str) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .delete(&uri)
                .await
                .map_err(|e| py_err(e.context("Delete failed")))?;
            // Python `None`, converted inside the exit-safe delivery (issue #271).
            Ok(None::<()>)
        })
    }

    /// List objects from ALL endpoints and merge results (deduplicated).
    ///
    /// Unlike ``list()``, which only queries one load-balanced endpoint, this
    /// queries every configured endpoint in parallel. For true-replication
    /// deployments (see the ``MultiEndpointStore`` replication-assumption note in
    /// the Rust docs / issue #162) this is the reliable way to discover objects
    /// regardless of round-robin state.
    fn list_all_endpoints<'py>(
        &self,
        py: Python<'py>,
        prefix: &str,
        recursive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prefix = prefix.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            let objects = store
                .list_all_endpoints(&prefix, recursive)
                .await
                .map_err(|e| py_err(e.context("List-all-endpoints failed")))?;
            Ok(objects)
        })
    }

    /// Get an object from a specific configured endpoint, bypassing load
    /// balancing entirely. Use this for sharded (non-replicated) deployments --
    /// see the replication-assumption note on ``MultiEndpointStore`` / issue #162.
    ///
    /// Raises if ``index`` is out of range for the configured endpoint count.
    fn get_from_endpoint<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        uri: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            let data = store
                .get_from_endpoint(index, &uri)
                .await
                .map_err(|e| py_err(e.context("Get from endpoint failed")))?;
            Python::attach(|py| Ok(Py::new(py, PyBytesView::new(data))?.into_any()))
        })
    }

    /// Put an object to a specific configured endpoint, bypassing load
    /// balancing entirely. See ``get_from_endpoint``.
    ///
    /// Raises if ``index`` is out of range for the configured endpoint count.
    ///
    /// `borrow` is as for ``put``: `True` (the default) reads a mutable buffer in place until
    /// the awaitable completes (do not modify, resize or free it meanwhile); `False` copies it
    /// once now.
    #[pyo3(signature = (index, uri, data, *, borrow = true))]
    fn put_to_endpoint<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        uri: &str,
        data: &Bound<'py, PyAny>,
        borrow: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let payload = endpoint_payload(py, data, borrow, &uri)?;
        let store = self.store.clone();

        guarded_put(py, payload.latch, move || async move {
            store
                .put_to_endpoint(index, &uri, payload.bytes)
                .await
                .map_err(|e| py_err(e.context("Put to endpoint failed")))?;
            // Python `None`, converted inside the exit-safe delivery (issue #271).
            Ok(None::<()>)
        })
    }

    /// Delete an object from a specific configured endpoint, bypassing load
    /// balancing entirely. See ``get_from_endpoint``.
    ///
    /// Raises if ``index`` is out of range for the configured endpoint count.
    fn delete_from_endpoint<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        uri: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .delete_from_endpoint(index, &uri)
                .await
                .map_err(|e| py_err(e.context("Delete from endpoint failed")))?;
            // Python `None`, converted inside the exit-safe delivery (issue #271).
            Ok(None::<()>)
        })
    }

    /// Put the same object to every configured endpoint (fan-out write), for
    /// true-replication deployments that want a guaranteed-consistent write
    /// instead of relying on independent round-robin ``put()`` calls to
    /// eventually cover every endpoint. Mirrors ``list_all_endpoints`` for
    /// the write side. Raises if the write failed on any endpoint (not
    /// transactional -- some endpoints may already hold the write).
    ///
    /// `borrow` is as for ``put``: `True` (the default) reads a mutable buffer in place, once
    /// for ALL endpoints, until the awaitable completes (do not modify, resize or free it
    /// meanwhile); `False` copies it once now.
    #[pyo3(signature = (uri, data, *, borrow = true))]
    fn put_all_endpoints<'py>(
        &self,
        py: Python<'py>,
        uri: &str,
        data: &Bound<'py, PyAny>,
        borrow: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let payload = endpoint_payload(py, data, borrow, &uri)?;
        let store = self.store.clone();

        guarded_put(py, payload.latch, move || async move {
            store
                .put_all_endpoints(&uri, payload.bytes)
                .await
                .map_err(|e| py_err(e.context("Put-all-endpoints failed")))?;
            // Python `None`, converted inside the exit-safe delivery (issue #271).
            Ok(None::<()>)
        })
    }

    /// Get per-endpoint statistics
    fn get_endpoint_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.store.get_all_stats();
        let list = PyList::empty(py);

        for (uri, stat) in stats {
            let dict = PyDict::new(py);
            dict.set_item("uri", uri)?;
            dict.set_item("total_requests", stat.total_requests)?;
            dict.set_item("bytes_read", stat.bytes_read)?;
            dict.set_item("bytes_written", stat.bytes_written)?;
            dict.set_item("error_count", stat.error_count)?;
            dict.set_item("active_requests", stat.active_requests)?;
            list.append(dict)?;
        }

        Ok(list.into())
    }

    /// Get total aggregated statistics across all endpoints
    fn get_total_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let total = self.store.get_total_stats();
        let dict = PyDict::new(py);

        dict.set_item("total_requests", total.total_requests)?;
        dict.set_item("bytes_read", total.bytes_read)?;
        dict.set_item("bytes_written", total.bytes_written)?;
        dict.set_item("error_count", total.error_count)?;
        dict.set_item("active_requests", total.active_requests)?;

        Ok(dict.into())
    }

    /// Get the number of configured endpoints
    fn endpoint_count(&self) -> usize {
        self.store.endpoint_count()
    }

    /// Get the load balancing strategy
    fn strategy(&self) -> String {
        match self.store.strategy() {
            LoadBalanceStrategy::RoundRobin => "round_robin".to_string(),
            LoadBalanceStrategy::LeastConnections => "least_connections".to_string(),
        }
    }
}

/// Create a multi-endpoint store from a list of URIs
///
/// Args:
///     uris: List of storage URIs (must all use the same scheme: s3://, az://, gs://, file://, or direct://)
///     strategy: Load balancing strategy - use "round_robin" or "least_connections" (default: "round_robin")
///
/// Returns:
///     MultiEndpointStore instance
///
/// Example:
///     >>> import asyncio
///     >>> store = s3dlio.create_multi_endpoint_store(
///     ...     uris=["s3://host1:9000/bucket/", "s3://host2:9000/bucket/"],
///     ...     strategy="least_connections"
///     ... )
///     >>> # All methods are async and require await
///     >>> data = asyncio.run(store.get("s3://host1:9000/bucket/object.dat"))
#[pyfunction]
#[pyo3(signature = (uris, strategy = None))]
fn create_multi_endpoint_store(
    uris: Vec<String>,
    strategy: Option<&str>,
) -> PyResult<PyMultiEndpointStore> {
    let strategy = match strategy.unwrap_or("round_robin") {
        "round_robin" => LoadBalanceStrategy::RoundRobin,
        "least_connections" => LoadBalanceStrategy::LeastConnections,
        s => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid strategy '{}'. Use 'round_robin' or 'least_connections'",
                s
            )))
        }
    };

    let store = MultiEndpointStore::new(uris, strategy, None)
        .map_err(|e| py_err(e.context("Failed to create multi-endpoint store")))?;

    Ok(PyMultiEndpointStore {
        store: Arc::new(store),
    })
}

/// Create a multi-endpoint store from a URI template with range expansion
///
/// Args:
///     uri_template: URI template with {start...end} range syntax (e.g., "s3://bucket-{1...10}/data")
///     strategy: Load balancing strategy - use "round_robin" or "least_connections" (default: "round_robin")
///
/// Returns:
///     MultiEndpointStore instance
///
/// Example:
///     >>> import asyncio
///     >>> store = s3dlio.create_multi_endpoint_store_from_template(
///     ...     "s3://10.0.0.{1...8}:9000/bucket/",
///     ...     strategy="least_connections"
///     ... )
///     >>> # Creates store with 8 endpoints: 10.0.0.1 through 10.0.0.8
///     >>> # All methods are async and require await
///     >>> data = asyncio.run(store.get("s3://10.0.0.1:9000/bucket/object.dat"))
#[pyfunction]
#[pyo3(signature = (uri_template, strategy = None))]
fn create_multi_endpoint_store_from_template(
    uri_template: &str,
    strategy: Option<&str>,
) -> PyResult<PyMultiEndpointStore> {
    let uris = uri_utils::expand_uri_template(uri_template)
        .map_err(|e| py_err(e.context("Template expansion failed")))?;

    create_multi_endpoint_store(uris, strategy)
}

/// Create a multi-endpoint store by loading URIs from a file
///
/// Args:
///     file_path: Path to file containing URIs (one per line, # for comments, blank lines ignored)
///     strategy: Load balancing strategy - use "round_robin" or "least_connections" (default: "round_robin")
///
/// Returns:
///     MultiEndpointStore instance
///
/// Example:
///     >>> import asyncio
///     >>> # endpoints.txt contains:
///     >>> # s3://host1:9000/bucket/
///     >>> # s3://host2:9000/bucket/
///     >>> # s3://host3:9000/bucket/
///     >>> store = s3dlio.create_multi_endpoint_store_from_file(
///     ...     "endpoints.txt",
///     ...     strategy="round_robin"
///     ... )
///     >>> # All methods are async and require await
///     >>> data = asyncio.run(store.get("s3://host1:9000/bucket/object.dat"))
#[pyfunction]
#[pyo3(signature = (file_path, strategy = None))]
fn create_multi_endpoint_store_from_file(
    file_path: &str,
    strategy: Option<&str>,
) -> PyResult<PyMultiEndpointStore> {
    let path = std::path::Path::new(file_path);
    let uris = uri_utils::load_uris_from_file(path)
        .map_err(|e| py_err(e.context("Failed to load URIs from file")))?;

    create_multi_endpoint_store(uris, strategy)
}

/// Create a multi-endpoint store from the ``S3_ENDPOINT_URIS`` environment variable
///
/// Reads a comma-separated list of storage URIs from the ``S3_ENDPOINT_URIS``
/// environment variable and creates a :class:`MultiEndpointStore`.  All URIs
/// must use the same scheme.  The list must contain between 1 and
/// ``MAX_ENDPOINTS`` (32) entries.
///
/// Args:
///     strategy: Load balancing strategy - use "round_robin" or "least_connections" (default: "round_robin")
///
/// Returns:
///     MultiEndpointStore instance
///
/// Raises:
///     RuntimeError: If ``S3_ENDPOINT_URIS`` is unset, empty, exceeds MAX_ENDPOINTS, or contains mixed schemes
///
/// Example:
///     >>> import os, asyncio
///     >>> os.environ['S3_ENDPOINT_URIS'] = 's3://host1:9000/bucket/,s3://host2:9000/bucket/'
///     >>> store = s3dlio.create_multi_endpoint_store_from_env(strategy="round_robin")
#[pyfunction]
#[pyo3(signature = (strategy = None))]
fn create_multi_endpoint_store_from_env(strategy: Option<&str>) -> PyResult<PyMultiEndpointStore> {
    let lb_strategy = match strategy.unwrap_or("round_robin") {
        "round_robin" => LoadBalanceStrategy::RoundRobin,
        "least_connections" => LoadBalanceStrategy::LeastConnections,
        s => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid strategy '{}'. Use 'round_robin' or 'least_connections'",
                s
            )))
        }
    };
    let store = MultiEndpointStore::from_env(lb_strategy, None).map_err(|e| {
        py_err(e.context("Failed to create multi-endpoint store from S3_ENDPOINT_URIS"))
    })?;
    Ok(PyMultiEndpointStore {
        store: Arc::new(store),
    })
}

// ---------------------------------------------------------------------------
// GCS tuning — programmatic setters, getters, and bucket RAPID query
// ---------------------------------------------------------------------------

/// Set the number of gRPC subchannels for GCS.
///
/// Must be called before the first ``gs://`` operation.  Setting this to
/// your workload's concurrency (e.g. ``num_jobs``) gives each concurrent
/// stream an uncontested HTTP/2 flow-control window, which is critical for
/// RAPID / Hyperdisk ML throughput.
///
/// The ``S3DLIO_GCS_GRPC_CHANNELS`` environment variable, if set, takes
/// precedence over this call.
#[pyfunction]
#[cfg(feature = "backend-gcs")]
fn gcs_set_channel_count(n: usize) {
    crate::google_gcs_client::set_gcs_channel_count(n);
}

#[pyfunction]
#[cfg(not(feature = "backend-gcs"))]
// Accepted and ignored: this build has no GCS backend; the name must match the real build's Python keyword.
#[allow(unused_variables)]
fn gcs_set_channel_count(n: usize) {}

/// Configure GCS RAPID (Hyperdisk ML / zonal) mode before the first ``gs://``
/// operation.
///
/// Parameters
/// ----------
/// force : bool or None
///     * ``True``  — force RAPID on for every bucket
///     * ``False`` — force RAPID off for every bucket
///     * ``None``  — auto-detect per bucket (default; recommended)
///
/// The ``S3DLIO_GCS_RAPID`` environment variable still takes precedence if set.
#[pyfunction]
#[cfg(feature = "backend-gcs")]
#[pyo3(signature = (force = None))]
fn gcs_set_rapid_mode(force: Option<bool>) {
    crate::google_gcs_client::set_gcs_rapid_mode(force);
}

#[pyfunction]
#[cfg(not(feature = "backend-gcs"))]
#[pyo3(signature = (force = None))]
// Accepted and ignored: this build has no GCS backend; the name must match the real build's Python keyword.
#[allow(unused_variables)]
fn gcs_set_rapid_mode(force: Option<bool>) {}

/// Return the programmatically-configured GCS subchannel count.
///
/// Returns ``0`` if :func:`gcs_set_channel_count` has not been called
/// (auto-detect will be used on first client initialization).
#[pyfunction]
#[cfg(feature = "backend-gcs")]
fn gcs_get_channel_count() -> usize {
    crate::google_gcs_client::get_gcs_channel_count()
}

#[pyfunction]
#[cfg(not(feature = "backend-gcs"))]
fn gcs_get_channel_count() -> usize {
    0
}

/// Return the current effective RAPID mode setting.
///
/// Resolution includes the ``S3DLIO_GCS_RAPID`` environment variable:
///
/// * ``True``  — RAPID forced on
/// * ``False`` — RAPID forced off
/// * ``None``  — auto-detect per bucket (default)
#[pyfunction]
#[cfg(feature = "backend-gcs")]
fn gcs_get_rapid_mode() -> Option<bool> {
    crate::google_gcs_client::get_gcs_rapid_mode()
}

#[pyfunction]
#[cfg(not(feature = "backend-gcs"))]
fn gcs_get_rapid_mode() -> Option<bool> {
    None
}

/// Query whether a GCS bucket or ``gs://`` URI is a RAPID (Hyperdisk ML) bucket.
///
/// Accepts either a bare bucket name (``"my-bucket"``) or a full URI
/// (``"gs://my-bucket/prefix/"``).  The result is determined once via the
/// ``GetStorageLayout`` API and cached for the lifetime of the process, so
/// repeated calls for the same bucket are essentially free.
///
/// Returns ``False`` on authentication / network errors (a warning is logged).
///
/// Example
/// -------
/// >>> import s3dlio
/// >>> s3dlio.gcs_query_rapid_bucket("gs://my-rapid-bucket/")
/// True
#[pyfunction]
#[cfg(feature = "backend-gcs")]
fn gcs_query_rapid_bucket(py: Python<'_>, bucket_or_uri: &str) -> PyResult<bool> {
    let owned = bucket_or_uri.to_owned();
    // GIL released while the bucket is queried (issue #201).
    py.detach(|| {
        submit_io(async move {
            Ok::<bool, anyhow::Error>(
                crate::google_gcs_client::query_gcs_rapid_bucket(&owned).await,
            )
        })
    })
}

#[pyfunction]
#[cfg(not(feature = "backend-gcs"))]
// Accepted and ignored: this build has no GCS backend; the name must match the real build's Python keyword.
#[allow(unused_variables)]
fn gcs_query_rapid_bucket(bucket_or_uri: &str) -> PyResult<bool> {
    Ok(false)
}

// ---------------------------------------------------------------------------
// Module registration function for core API
// ---------------------------------------------------------------------------

/// List top-level containers (buckets, containers, or directories) for the
/// storage backend identified by `uri`.
///
/// Args:
///     uri: Storage URI identifying the backend.  Examples:
///          ``"s3://"`` — all S3 buckets visible to current credentials
///          ``"gs://"`` — all GCS buckets in GOOGLE_CLOUD_PROJECT
///          ``"az://mystorageaccount"`` — all containers for that account
///          ``"file:///path/to/dir"`` — top-level sub-directories
///
/// Returns:
///     list[dict]: Each element has keys ``name``, ``uri``, and
///     ``creation_date`` (a string or ``None``).
///
/// Raises:
///     RuntimeError: On any storage or credentials error.
#[pyfunction]
#[pyo3(text_signature = "(uri, /)")]
pub fn list_containers(py: Python<'_>, uri: &str) -> PyResult<Py<PyAny>> {
    let uri_owned = uri.to_string();
    let containers = py.detach(move || list_containers_rs(&uri_owned).map_err(py_err))?;

    let result = PyList::empty(py);
    for c in containers {
        let d = PyDict::new(py);
        d.set_item("name", &c.name)?;
        d.set_item("uri", &c.uri)?;
        match c.creation_date {
            Some(ref date) => d.set_item("creation_date", date)?,
            None => d.set_item("creation_date", py.None())?,
        }
        result.append(d)?;
    }
    result.into_py_any(py)
}

pub fn register_core_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Logging
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(init_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(is_op_log_active, m)?)?;
    m.add_function(wrap_pyfunction!(op_log_stats_py, m)?)?;

    // URI parsing utilities
    m.add_function(wrap_pyfunction!(parse_s3_uri_full, m)?)?;

    // S3 connection configuration (equivalent to CLI --endpoint-url / --region / --ca-bundle)
    m.add_function(wrap_pyfunction!(configure_s3, m)?)?;

    // Bucket / container management
    m.add_function(wrap_pyfunction!(create_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(delete_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(list_containers, m)?)?;

    // Universal storage operations (preferred)
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(stat, m)?)?;
    m.add_function(wrap_pyfunction!(stat_async, m)?)?;
    m.add_function(wrap_pyfunction!(stat_many_async, m)?)?;
    m.add_function(wrap_pyfunction!(exists, m)?)?;
    m.add_function(wrap_pyfunction!(exists_async, m)?)?;
    m.add_function(wrap_pyfunction!(put_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(put_bytes_async, m)?)?;
    m.add_function(wrap_pyfunction!(_put_export_stats, m)?)?;
    m.add_function(wrap_pyfunction!(_put_test_queue_export, m)?)?;
    m.add_function(wrap_pyfunction!(_put_test_retain_next_clone, m)?)?;
    m.add_function(wrap_pyfunction!(_put_test_retained, m)?)?;
    m.add_function(wrap_pyfunction!(_put_test_release_retained, m)?)?;
    m.add(
        "ExportStillReferencedError",
        m.py().get_type::<ExportStillReferencedError>(),
    )?;
    // Caller-buffer GET, typed HEAD and pure capabilities (#264)
    m.add_function(wrap_pyfunction!(object_capabilities_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_into_async, m)?)?;
    m.add_function(wrap_pyfunction!(head_async, m)?)?;
    m.add_class::<PyHeadMetadata>()?;
    m.add(
        "ObjectNotFoundError",
        m.py().get_type::<ObjectNotFoundError>(),
    )?;
    m.add(
        "LengthMismatchError",
        m.py().get_type::<LengthMismatchError>(),
    )?;
    m.add(
        "UnsupportedOperationError",
        m.py().get_type::<UnsupportedOperationError>(),
    )?;
    m.add_function(wrap_pyfunction!(mkdir, m)?)?;
    m.add_function(wrap_pyfunction!(mkdir_async, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_range_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(put_many, m)?)?;
    m.add_function(wrap_pyfunction!(put_many_async, m)?)?;
    m.add_function(wrap_pyfunction!(mp_get, m)?)?;

    // Phase 2 Streaming API classes and functions
    m.add_class::<PyWriterOptions>()?;
    m.add_class::<PyObjectWriter>()?;
    m.add_class::<PyBytesView>()?; // Zero-copy buffer wrapper
    m.add_function(wrap_pyfunction!(create_s3_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_azure_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_filesystem_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_direct_filesystem_writer, m)?)?;

    // Multi-endpoint support (v0.9.14+)
    m.add_class::<PyMultiEndpointStore>()?;
    m.add_function(wrap_pyfunction!(create_multi_endpoint_store, m)?)?;
    m.add_function(wrap_pyfunction!(
        create_multi_endpoint_store_from_template,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(create_multi_endpoint_store_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(create_multi_endpoint_store_from_env, m)?)?;

    // GCS tuning (programmatic setters / getters / bucket RAPID query)
    m.add_function(wrap_pyfunction!(gcs_set_channel_count, m)?)?;
    m.add_function(wrap_pyfunction!(gcs_set_rapid_mode, m)?)?;
    m.add_function(wrap_pyfunction!(gcs_get_channel_count, m)?)?;
    m.add_function(wrap_pyfunction!(gcs_get_rapid_mode, m)?)?;
    m.add_function(wrap_pyfunction!(gcs_query_rapid_bucket, m)?)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — put_verified_with_retry opt-in verification (storage#593, v0.9.106)
// ---------------------------------------------------------------------------

// This module lives inside `python_api`, which is gated behind the
// `extension-module` feature (not in the default feature set). Run with:
//   cargo test --lib --features extension-module -- --ignored --test-threads=1 put_verify_tests
// `--test-threads=1` is required: both tests mutate the shared process-wide
// `S3DLIO_PUT_VERIFY` env var and would otherwise race if run concurrently.
// ---------------------------------------------------------------------------
// Tests — error chain preserved across the PyO3 boundary
// (storage#755 / s3dlio#161, v0.9.112)
// ---------------------------------------------------------------------------
//
// A user's DLIO training run against S3 died with only
// `RuntimeError: concurrent range chunk failed` — the real cause (a chained
// `anyhow::Error`, e.g. an SDK/timeout/TLS failure under `.context(...)`) was
// discarded at the PyO3 boundary. Root cause: `py_err()` formatted the error
// with plain `Display` (`{}`), which for `anyhow::Error` (and anything
// wrapping it, like `DatasetError::Backend`) prints only the outermost
// context message. `{:#}` (alternate Display) prints the full cause chain.
#[cfg(test)]
mod error_chain_preservation_tests {
    use super::*;

    fn chained_error() -> anyhow::Error {
        // Mirrors the real shape from src/s3_utils.rs:
        // an SDK-level cause wrapped by a higher-level `.context(...)` label.
        let inner = anyhow::anyhow!("HEAD failed: connection reset by peer");
        inner.context("concurrent range chunk failed")
    }

    #[test]
    fn error_chain_message_preserves_the_full_cause_chain() {
        let msg = error_chain_message(&chained_error());
        assert!(
            msg.contains("connection reset by peer"),
            "error message reaching Python must include the underlying cause, \
             not just the outer context label — got: {msg:?}"
        );
        assert!(
            msg.contains("concurrent range chunk failed"),
            "the outer context label should still be present too — got: {msg:?}"
        );
    }

    #[test]
    fn error_chain_message_preserves_chain_through_dataset_error_wrapper() {
        // `DatasetError::Backend` is `#[error(transparent)]` over `anyhow::Error`
        // — this is the exact wrapper collect_batch()/`__next__()` see.
        use crate::data_loader::dataset::DatasetError;
        let wrapped: DatasetError = chained_error().into();
        let msg = error_chain_message(&wrapped);
        assert!(
            msg.contains("connection reset by peer"),
            "chain must survive the DatasetError wrapper too — got: {msg:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — PyObjectWriter::checksum() survives finalize() (design doc §4.1)
// ---------------------------------------------------------------------------
//
// `finalize()` used to snapshot only (bytes_written, compressed_bytes) into
// `finalized_stats` before consuming the writer, so `checksum()` always
// returned `None` afterward even though the real, fully-computed checksum
// was briefly reachable right before consumption. `ObjectWriter::checksum()`
// is documented and implemented (see docs/DESIGN_FFI_BOUNDARY_HARDENING.md
// §4.1) as an incremental "written so far" rolling-hash value, so it's safe
// to snapshot it alongside the other two stats at the same point.
#[cfg(test)]
mod checksum_survives_finalize_tests {
    use super::*;
    use async_trait::async_trait;

    /// Minimal `ObjectWriter` double: a fixed checksum, available at any
    /// point (mirrors every real implementation's rolling-hasher behavior).
    struct StubWriter {
        checksum: String,
    }

    #[async_trait]
    impl ObjectWriter for StubWriter {
        async fn write_chunk(&mut self, _chunk: &[u8]) -> anyhow::Result<()> {
            Ok(())
        }
        async fn finalize(self: Box<Self>) -> anyhow::Result<()> {
            Ok(())
        }
        fn bytes_written(&self) -> u64 {
            42
        }
        fn checksum(&self) -> Option<String> {
            Some(self.checksum.clone())
        }
    }

    #[test]
    fn snapshot_captures_checksum_before_finalize_consumes_the_writer() {
        let writer: Box<dyn ObjectWriter> = Box::new(StubWriter {
            checksum: "crc32c:deadbeef".to_string(),
        });
        let (bytes_written, _compressed_bytes, checksum) =
            snapshot_final_writer_stats(writer.as_ref());
        assert_eq!(bytes_written, 42);
        assert_eq!(
            checksum,
            Some("crc32c:deadbeef".to_string()),
            "checksum must be captured in the pre-finalize snapshot, not \
             lost once the writer is consumed by finalize()"
        );
    }
}

#[cfg(test)]
mod parse_put_max_retries_tests {
    use super::*;

    // RED-then-GREEN regression tests for s3dlio issue #157 bug 7.4 (F2).
    // No live/mocked ObjectStore needed -- the clamp is a pure function.

    #[test]
    fn zero_is_clamped_to_the_documented_default_of_3() {
        assert_eq!(
            parse_put_max_retries(Some("0")),
            3,
            "S3DLIO_PUT_MAX_RETRIES=0 must fall back to the documented \
             default of 3, not produce a 0-iteration retry loop"
        );
    }

    #[test]
    fn non_numeric_is_clamped_to_the_documented_default_of_3() {
        assert_eq!(parse_put_max_retries(Some("not-a-number")), 3);
    }

    #[test]
    fn unset_uses_the_documented_default_of_3() {
        assert_eq!(parse_put_max_retries(None), 3);
    }

    #[test]
    fn valid_positive_value_passes_through_unchanged() {
        assert_eq!(parse_put_max_retries(Some("7")), 7);
        assert_eq!(parse_put_max_retries(Some("1")), 1);
    }
}

#[cfg(test)]
mod put_verify_tests {
    use super::*;

    fn bucket() -> String {
        std::env::var("BUCKET").unwrap_or_else(|_| "mlp-s3dlio".to_string())
    }

    fn unique_key(tag: &str) -> String {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("test-put-verify-{tag}-{ts}.bin")
    }

    /// Default behavior (`S3DLIO_PUT_VERIFY` unset): a single PUT lands the
    /// object correctly with no internal HEAD call.  Confirmed here via an
    /// independent stat from the test itself, not via internal logic.
    ///
    /// Run with: `cargo test --lib -- --ignored test_put_verify_default_off`
    #[test]
    #[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
    fn test_put_verify_default_off() -> anyhow::Result<()> {
        dotenvy::dotenv().ok();
        std::env::remove_var("S3DLIO_PUT_VERIFY"); // explicit: exercise the default
        let bucket = bucket();
        let key = unique_key("default-off");
        let uri = format!("s3://{bucket}/{key}");
        let data = Bytes::from(vec![0x42u8; 4096]);
        let expected = data.len() as u64;

        run_on_global_rt(async move {
            let store = store_for_uri_with_logger(&uri, None)?;
            put_verified_with_retry(store.as_ref(), &uri, data).await?;
            let meta = store.stat(&uri).await?;
            anyhow::ensure!(
                meta.size == expected,
                "object must land correctly even without internal verification: \
                 expected {} got {}",
                expected,
                meta.size
            );
            store.delete(&uri).await?;
            Ok(())
        })
    }

    /// Opt-in behavior (`S3DLIO_PUT_VERIFY=true`): the HEAD-verify-retry path
    /// runs and confirms the stored size, same correctness guarantee as
    /// before v0.9.106 made this opt-in.
    ///
    /// Run with: `cargo test --lib -- --ignored test_put_verify_opt_in`
    #[test]
    #[ignore = "requires live S3 endpoint (set AWS_* + BUCKET env vars or use .env file)"]
    fn test_put_verify_opt_in() -> anyhow::Result<()> {
        dotenvy::dotenv().ok();
        std::env::set_var("S3DLIO_PUT_VERIFY", "true");
        let bucket = bucket();
        let key = unique_key("opt-in");
        let uri = format!("s3://{bucket}/{key}");
        let data = Bytes::from(vec![0x99u8; 4096]);
        let expected = data.len() as u64;

        let result = run_on_global_rt(async move {
            let store = store_for_uri_with_logger(&uri, None)?;
            put_verified_with_retry(store.as_ref(), &uri, data).await?;
            let meta = store.stat(&uri).await?;
            anyhow::ensure!(
                meta.size == expected,
                "HEAD-verified size mismatch: expected {} got {}",
                expected,
                meta.size
            );
            store.delete(&uri).await?;
            Ok(())
        });

        std::env::remove_var("S3DLIO_PUT_VERIFY");
        result
    }
}

// ---------------------------------------------------------------------------
// Tests -- issue #173: sync entry points must build host-form stores on the
// caller thread, and `get_or_create_store` must also be safe when it IS called
// from a runtime worker.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod issue173_store_creation_tests {
    use super::*;
    use crate::s3_client::issue173_worker_reentrancy_tests::{run_with_watchdog, WATCHDOG};

    /// Host-form URI: the authority carries `host:port`, so
    /// `store_for_uri_with_logger` takes the `run_on_global_rt(for_endpoint)`
    /// path.  Store creation does not connect, so the (closed) port 9 is never
    /// contacted.
    const HOST_FORM_URI: &str = "s3://127.0.0.1:9/bucket/key";

    fn prepare_environment() {
        // for_endpoint requires credentials to exist; none are ever used.
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        // Force the slow (creating) path of get_or_create_store.
        STORE_CACHE.remove(&StoreKey::from_uri(HOST_FORM_URI).expect("valid URI"));
    }

    /// The exact shape of the old `get`/`get_range`/`put_bytes` bodies:
    /// `get_or_create_store` inside the future handed to `submit_io`, i.e.
    /// executing on a global-runtime worker.
    #[test]
    fn get_or_create_store_inside_submit_io_completes_for_host_form_uri() {
        prepare_environment();
        let got = run_with_watchdog("py-submit-io", || {
            submit_io(async move {
                get_or_create_store(HOST_FORM_URI, None)
                    .map(|_| ())
                    .map_err(|e| e.context("get_or_create_store"))
            })
        });
        match got {
            Some(r) => assert!(r.is_ok(), "store creation failed: {r:?}"),
            None => panic!(
                "RED: get_or_create_store(host-form) inside submit_io did not \
                 return within {WATCHDOG:?} (issue #173 deadlock)"
            ),
        }
    }

    /// The shape of the fixed entry points: create on the caller thread
    /// (no runtime worker involved), then hand the store to `submit_io`.
    #[test]
    fn store_created_on_caller_thread_before_submit_io_completes() {
        prepare_environment();
        let got = run_with_watchdog("py-caller-thread", || {
            let store = get_or_create_store(HOST_FORM_URI, None)?;
            submit_io(async move {
                // Touch the store on the worker, as the entry points do.
                let _ = store.as_ref();
                Ok::<(), anyhow::Error>(())
            })
            .map_err(|e| anyhow::anyhow!("submit_io failed: {e}"))
        });
        match got {
            Some(r) => assert!(r.is_ok(), "caller-thread creation failed: {r:?}"),
            None => panic!(
                "caller-thread store creation + submit_io did not return within {WATCHDOG:?}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- issue #171: the store cache must key on the server a URI names.
//
// Before the fix `StoreKey` was (scheme, AWS_ENDPOINT_URL, region), so a
// cache hit for `s3://hostB:9000/b/k` returned the store built for hostA (or
// for the global AWS_ENDPOINT_URL client) and the request went to the wrong
// server.  These tests need no network: store creation never connects.
//
// STORE_CACHE is process-global, so these tests serialise on TEST_LOCK and never
// remove entries (since #204 racing misses on one key build one store; the
// serialising keeps the pointer-identity assertions independent of other tests).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod issue171_store_cache_routing_tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn serialise() -> MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const GLOBAL_BUCKET_URI: &str = "s3://plain-bucket-171/key";
    const GLOBAL_BUCKET_URI_2: &str = "s3://another-plain-bucket-171/key";

    fn host_form(port: u16, bucket: &str) -> String {
        format!("s3://127.0.0.1:{port}/{bucket}/some/key")
    }

    fn prepare_environment() {
        // for_endpoint requires credentials to exist; none are ever used.
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
    }

    #[test]
    fn host_form_uris_on_different_servers_have_different_keys() {
        let _guard = serialise();
        assert!(
            StoreKey::from_uri(&host_form(19, "b")).expect("valid URI")
                != StoreKey::from_uri(&host_form(20, "b")).expect("valid URI"),
            "RED: two host-form URIs on different host:port share one StoreKey"
        );
    }

    #[test]
    fn host_form_and_bucket_form_uris_have_different_keys() {
        let _guard = serialise();
        assert!(
            StoreKey::from_uri(&host_form(19, "b")).expect("valid URI")
                != StoreKey::from_uri(GLOBAL_BUCKET_URI).expect("valid URI"),
            "RED: a host-form URI and a bucket URI share one StoreKey"
        );
    }

    #[test]
    fn same_server_different_buckets_share_a_key() {
        let _guard = serialise();
        assert!(
            StoreKey::from_uri(&host_form(19, "bucket1")).expect("valid URI")
                == StoreKey::from_uri(&host_form(19, "bucket2")).expect("valid URI"),
            "buckets on one server must share a StoreKey or the cache never hits"
        );
        assert!(
            StoreKey::from_uri(GLOBAL_BUCKET_URI).expect("valid URI")
                == StoreKey::from_uri(GLOBAL_BUCKET_URI_2).expect("valid URI"),
            "bucket URIs on the global endpoint must share a StoreKey"
        );
    }

    #[test]
    fn different_servers_get_distinct_stores_from_the_cache() {
        let _guard = serialise();
        prepare_environment();
        let a = get_or_create_store(&host_form(31, "bucket1"), None).expect("store for server A");
        // Warm-cache hit for A, then ask for B: the old key returned A's store.
        let b = get_or_create_store(&host_form(32, "bucket1"), None).expect("store for server B");
        assert!(
            !Arc::ptr_eq(&a, &b),
            "RED: get_or_create_store returned server A's store for a server B URI"
        );
    }

    #[test]
    fn host_form_and_bucket_form_get_distinct_stores_from_the_cache() {
        let _guard = serialise();
        prepare_environment();
        let host = get_or_create_store(&host_form(33, "bucket1"), None).expect("host-form store");
        let global = get_or_create_store(GLOBAL_BUCKET_URI, None).expect("global store");
        assert!(
            !Arc::ptr_eq(&global, &host),
            "RED: a host-form URI was served by the global-endpoint store"
        );
    }

    #[test]
    fn same_server_different_buckets_reuse_one_cached_store() {
        let _guard = serialise();
        prepare_environment();
        let first = get_or_create_store(&host_form(34, "bucket1"), None).expect("bucket1 store");
        let second = get_or_create_store(&host_form(34, "bucket2"), None).expect("bucket2 store");
        assert!(
            Arc::ptr_eq(&first, &second),
            "two buckets on one server must hit the same cached store"
        );
        let g1 = get_or_create_store(GLOBAL_BUCKET_URI, None).expect("global store 1");
        let g2 = get_or_create_store(GLOBAL_BUCKET_URI_2, None).expect("global store 2");
        assert!(
            Arc::ptr_eq(&g1, &g2),
            "two buckets on the global endpoint must hit the same cached store"
        );
    }

    // -----------------------------------------------------------------------
    // #171 x #178: a routing error must reach the caller BEFORE the cache is
    // consulted or populated. If `StoreKey::from_uri` mapped the #178 error to
    // "no route" (`None`), an invalid server URI would hash to the global-client
    // key and be served by the store cached for a perfectly valid bucket URI.
    // -----------------------------------------------------------------------

    /// How many cache entries sit under the global-client key (route `None`).
    fn global_route_entries() -> usize {
        let want = StoreKey::from_uri(GLOBAL_BUCKET_URI).expect("valid URI");
        STORE_CACHE.iter().filter(|e| *e.key() == want).count()
    }

    #[test]
    fn invalid_server_uri_errors_instead_of_hitting_a_warm_cache() {
        let _guard = serialise();
        prepare_environment();
        let warm_global = get_or_create_store(GLOBAL_BUCKET_URI, None).expect("warm global store");
        let warm_host =
            get_or_create_store(&host_form(35, "bucket1"), None).expect("warm host-form");
        let entries_before = global_route_entries();
        assert_eq!(
            entries_before, 1,
            "exactly one global-route entry is cached"
        );

        for bad in [
            "s3://10.0.0.5/bkt/key",        // portless IP literal
            "s3://node%x:9000/bkt/key",     // invalid host character
            "s3://127.0.0.1:99999/bkt/key", // port out of range
            "s3://[::1]/bkt/key",           // IPv6 without a port
        ] {
            assert!(
                StoreKey::from_uri(bad).is_err(),
                "RED: StoreKey::from_uri({bad}) must be an error, not a key"
            );
            match get_or_create_store(bad, None) {
                Ok(store) => panic!(
                    "RED: get_or_create_store({bad}) returned a store (global cached store: {}, \
                     host-form cached store: {})",
                    Arc::ptr_eq(&store, &warm_global),
                    Arc::ptr_eq(&store, &warm_host)
                ),
                Err(e) => assert!(
                    format!("{e:#}").contains(bad),
                    "the error must name the URI: {e:#}"
                ),
            }
        }

        assert_eq!(
            global_route_entries(),
            entries_before,
            "a failed lookup must not populate the cache"
        );
        // The warm entries are untouched and still served for valid URIs.
        let again = get_or_create_store(GLOBAL_BUCKET_URI_2, None).expect("global store");
        assert!(Arc::ptr_eq(&again, &warm_global));
        let host_again = get_or_create_store(&host_form(35, "bucket2"), None).expect("host-form");
        assert!(Arc::ptr_eq(&host_again, &warm_host));
    }
}

#[cfg(test)]
mod store_cache_logger_tests {
    use super::*;
    use crate::s3_logger::Logger;

    /// Issue #174: the process-wide store cache must not decide whether a call
    /// is recorded. Whoever builds the cached entry first (with or without a
    /// logger), each call is recorded exactly when ITS caller passes a logger
    /// (the caller passes `global_logger()`, i.e. "the op-log is active now"),
    /// and recorded on that caller's logger only.
    #[test]
    fn a_call_is_recorded_exactly_when_its_caller_passes_a_logger() {
        let dir = tempfile::tempdir().unwrap();
        let path = |name: &str| dir.path().join(name).to_str().unwrap().to_string();
        let first = Logger::new(&path("first.tsv.zst")).unwrap();
        let second = Logger::new(&path("second.tsv.zst")).unwrap();
        let file = path("data.bin");
        std::fs::write(&file, b"data").unwrap();
        let uri = format!("file://{file}");
        let accepted = |l: &Logger| l.stats().accepted;

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            // 1. Cache miss, built by a call that has a logger: recorded on it.
            let store = get_or_create_store(&uri, Some(first.clone())).unwrap();
            store.get(&uri).await.unwrap();
            assert_eq!(accepted(&first), 1, "logged call on a cache miss");

            // 2. Cache hit, caller has no logger (op-log inactive): not recorded,
            //    even though the first caller's logger built the cached store.
            let store = get_or_create_store(&uri, None).unwrap();
            store.get(&uri).await.unwrap();
            store.stat(&uri).await.unwrap();
            assert_eq!(accepted(&first), 1, "unlogged call on a cache hit");

            // 3. Cache hit, caller has a different logger: recorded there only.
            let store = get_or_create_store(&uri, Some(second.clone())).unwrap();
            store.get(&uri).await.unwrap();
            assert_eq!(accepted(&second), 1, "logged call on a cache hit");
            assert_eq!(accepted(&first), 1, "the first logger is not used again");
        });
    }

    /// Issue #242 review: `get_many` on az:// / gs:// must write op-log rows like `get()`
    /// does. The helper is scheme-agnostic, so the test reads `file://` objects through it
    /// (Azure and GCS need a service); the az/gs arm of `get_many` calls this same helper
    /// (see `get_many_routes_remote_schemes_through_the_cached_store_helper`).
    #[test]
    fn get_many_remote_helper_writes_op_log_rows_and_reuses_the_cached_store() {
        let dir = tempfile::tempdir().unwrap();
        let log = Logger::new(dir.path().join("rows.tsv.zst").to_str().unwrap()).unwrap();
        let uris: Vec<String> = (0..6)
            .map(|i| {
                let f = dir.path().join(format!("obj{i}.bin"));
                std::fs::write(&f, format!("data{i}")).unwrap();
                format!("file://{}", f.display())
            })
            .collect();

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let got = rt
            .block_on(get_many_cached_stores(uris.clone(), 3, Some(log.clone())))
            .unwrap();
        assert_eq!(got.len(), uris.len());
        assert_eq!(
            log.stats().accepted,
            uris.len() as u64,
            "RED: get_many wrote no op-log row per object"
        );

        // Every URI under one store key is served by the SAME cached store (not a store built
        // per URI), whether or not another test warmed the key first.
        let first = get_or_create_store(&uris[0], None).unwrap();
        for uri in &uris[1..] {
            let other = get_or_create_store(uri, None).unwrap();
            assert!(
                Arc::ptr_eq(&first, &other),
                "RED: {uri} got a different store from the cache than {}",
                uris[0]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- issue #199: `put(..., should_create_bucket=True)` on a host-form prefix.
//
// Before the fix `build_uri_list` took the FIRST segment as the bucket, so
// `s3://host:9000/bkt/pfx/` asked the global endpoint to create a bucket named
// `host:9000`. The tests use a tiny in-process mock that records each request.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod issue199_put_bucket_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;

    /// Mock S3 server: answers every request `200` and records `"METHOD /path"`.
    fn start_mock() -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let port = listener.local_addr().expect("mock addr").port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_in_thread = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let seen = Arc::clone(&seen_in_thread);
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    let Ok(n) = stream.read(&mut buf) else { return };
                    let head = String::from_utf8_lossy(&buf[..n]).to_string();
                    let mut parts = head.lines().next().unwrap_or("").split_whitespace();
                    let method = parts.next().unwrap_or("");
                    let path = parts.next().unwrap_or("").trim_end_matches('/');
                    if let Ok(mut r) = seen.lock() {
                        r.push(format!("{method} {path}"));
                    }
                    let _ = stream.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                });
            }
        });
        (port, seen)
    }

    /// `create_s3_client_for_endpoint` refuses to build a client without credentials.
    fn ensure_test_credentials() {
        if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
            unsafe { std::env::set_var("AWS_ACCESS_KEY_ID", "test-access-key") };
        }
        if std::env::var("AWS_SECRET_ACCESS_KEY").is_err() {
            unsafe { std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret-key") };
        }
    }

    #[test]
    fn host_form_prefix_creates_the_bucket_on_the_named_server() {
        ensure_test_credentials();
        let (port, seen) = start_mock();

        create_put_bucket(&format!("s3://127.0.0.1:{port}/bkt199py/pfx/"))
            .expect("a host-form prefix names a bucket");

        let seen = seen.lock().map(|r| r.clone()).unwrap_or_default();
        assert_eq!(seen, vec!["PUT /bkt199py".to_string()]);
    }

    #[test]
    fn portless_ip_prefix_is_the_178_error() {
        let err = create_put_bucket("s3://10.0.0.5/bkt/pfx/").expect_err("portless IP");
        assert!(
            format!("{err:#}").contains("IP address with no port"),
            "got: {err:#}"
        );
    }

    #[test]
    fn non_s3_prefix_has_no_bucket_to_create() {
        create_put_bucket("file:///tmp/s3dlio-199-no-bucket/").expect("nothing to create");
    }

    #[test]
    fn object_uris_keep_the_server_and_the_bucket() {
        let uris = build_uri_list("s3://127.0.0.1:9000/bkt/pfx/", "object-{}", 2)
            .expect("build the object URIs");
        assert_eq!(
            uris,
            vec![
                "s3://127.0.0.1:9000/bkt/pfx/object-0".to_string(),
                "s3://127.0.0.1:9000/bkt/pfx/object-1".to_string(),
            ]
        );
    }
}

// ---------------------------------------------------------------------------
// Tests -- issue #204: the async entry points share the store cache, and a
// cache miss is built once per key however many callers race on it.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod issue204_store_cache_tests {
    use super::*;

    fn prepare_environment() {
        // for_endpoint requires credentials to exist; none are ever used.
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
    }

    /// How many stores `get_or_create_store` has built for `uri`.
    fn builds_for(uri: &str) -> usize {
        STORE_BUILD_LOG
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|built| built.as_str() == uri)
            .count()
    }

    /// Every caller racing on one cold key gets the SAME store and exactly one
    /// store is built.  The port is unique to this test (the cache is
    /// process-global and other tests build their own keys); store creation
    /// never connects.
    #[test]
    fn racing_misses_on_one_key_build_one_store() {
        prepare_environment();
        const CALLERS: usize = 8;
        let uri = "s3://127.0.0.1:59204/bucket-204/some/key";
        assert_eq!(builds_for(uri), 0, "the key must start cold");

        let start = Arc::new(std::sync::Barrier::new(CALLERS));
        let handles: Vec<_> = (0..CALLERS)
            .map(|_| {
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    get_or_create_store(uri, None).map_err(|e| format!("{e:#}"))
                })
            })
            .collect();
        let stores: Vec<Arc<dyn ObjectStore>> = handles
            .into_iter()
            .map(|h| h.join().expect("caller thread").expect("store"))
            .collect();

        assert_eq!(
            builds_for(uri),
            1,
            "RED: {CALLERS} callers racing on one cold key built {} stores",
            builds_for(uri)
        );
        for store in &stores[1..] {
            assert!(
                Arc::ptr_eq(&stores[0], store),
                "RED: racing callers were handed different stores for one key"
            );
        }
    }

    /// The async entry points call `get_or_create_store` from pyo3-runtime worker threads.
    /// Racing misses there wait for the builder inside `block_in_place`: they complete,
    /// and still build one store.
    #[test]
    fn racing_misses_on_a_multi_thread_runtime_complete_and_build_once() {
        use crate::s3_client::issue173_worker_reentrancy_tests::{run_with_watchdog, WATCHDOG};
        prepare_environment();
        let uri = "s3://127.0.0.1:59206/bucket-204/some/key";
        let done = run_with_watchdog("racing-misses-runtime", move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async move {
                let tasks: Vec<_> = (0..8)
                    .map(|_| {
                        tokio::spawn(async move {
                            get_or_create_store(uri, None)
                                .map(|_| ())
                                .map_err(|e| format!("{e:#}"))
                        })
                    })
                    .collect();
                for task in tasks {
                    task.await.expect("task").expect("store");
                }
            });
        });
        assert!(
            done.is_some(),
            "racing misses on a multi-thread runtime did not finish within {WATCHDOG:?}"
        );
        assert_eq!(builds_for(uri), 1);
    }

    /// Sequential calls after the first are cache hits: no further build.
    #[test]
    fn a_warm_key_is_never_rebuilt() {
        prepare_environment();
        let uri = "s3://127.0.0.1:59205/bucket-204/some/key";
        for _ in 0..3 {
            get_or_create_store(uri, None).expect("store");
        }
        assert_eq!(
            builds_for(uri),
            1,
            "three calls on one key built more than once"
        );
    }

    /// `get_many` itself must route az:// / gs:// through the cached-store helper, with the
    /// op-log logger snapshot taken on the caller thread before `submit_io` (#173/#174).
    #[test]
    fn get_many_routes_remote_schemes_through_the_cached_store_helper() {
        let body = entry_point_source("get_many");
        assert!(
            !body.contains("store_for_uri(") && !body.contains("store_for_uri_with_logger("),
            "RED: get_many builds a store per URI with the factory"
        );
        let logger = body
            .find("global_logger()")
            .expect("RED: get_many never takes the op-log logger snapshot");
        let helper = body
            .find("get_many_cached_stores(")
            .expect("RED: get_many does not call get_many_cached_stores");
        assert!(
            logger < helper,
            "RED: the logger snapshot must be taken before get_many_cached_stores is called"
        );
    }

    /// Source of the entry point `name`: from its `fn` line to the closing brace at
    /// column 0.
    fn entry_point_source(name: &str) -> String {
        let src = include_str!("python_core_api.rs");
        let start = ["<'py>(", "<'p>(", "("]
            .iter()
            .find_map(|generics| src.find(&format!("fn {name}{generics}")))
            .unwrap_or_else(|| panic!("entry point {name} not found"));
        let rest = &src[start..];
        let end = rest.find("\n}\n").expect("end of function");
        rest[..end].to_string()
    }

    /// Structural guard for the entry points whose store build is not observable on
    /// the wire (`mkdir_async` makes no S3 request): none may build its own store
    /// with the factory; each must go through `get_or_create_store` (the #171
    /// route-aware key, the #174 per-call logger wrapping).
    #[test]
    fn async_entry_points_use_the_store_cache() {
        for name in [
            "stat_async",
            "exists_async",
            "put_bytes_async",
            "mkdir_async",
            "put_many_async",
            // The caller-buffer GET and typed HEAD (#264): the store is built on the caller thread.
            "get_into_async",
            "head_async",
            // `get_many` for az:// / gs:// goes through this helper (#242 review).
            "get_many_cached_stores",
        ] {
            let body = entry_point_source(name);
            assert!(
                !body.contains("store_for_uri_with_logger("),
                "RED: {name} builds a store per call with store_for_uri_with_logger"
            );
            assert!(
                body.contains("get_or_create_store("),
                "RED: {name} does not use get_or_create_store"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- PyObjectWriter::write_chunk / write_owned_bytes (issue #254).
//
// Superseded design: write_chunk used to keep the GIL for every mutable buffer and took the
// writer lock BEFORE building a slice over the caller's memory (waiting for the lock could
// release the GIL). It now converts the argument like put_bytes() (a held buffer export, no
// copy by the Python layer) and writes with the GIL RELEASED under the documented "do not
// modify during the call" contract. The behaviour (RSS, GIL stall, round trips) is tested in
// python/tests/test_python_write_paths_254.py; this pins the structure: no private raw-slice
// path and no GIL-held write is left in write_chunk.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod write_chunk_uses_the_shared_payload_conversion_tests {
    #[test]
    fn write_chunk_converts_with_put_payload_and_writes_detached() {
        let src = include_str!("python_core_api.rs");
        let start = src
            .find("fn write_chunk(&self, py: Python<'_>")
            .expect("write_chunk not found");
        let rest = &src[start..];
        let end = rest.find("\n    }\n").expect("end of write_chunk");
        let body = &rest[..end];
        assert!(
            body.contains("hold_for_call("),
            "write_chunk must use the strict shared conversion"
        );
        assert!(
            body.contains("write_detached("),
            "the write must run with the GIL released"
        );
        assert!(
            !body.contains("from_raw_parts(") && !body.contains("write_locked("),
            "write_chunk must not build its own slice or write with the GIL held"
        );
        assert!(
            body.contains("held.release(py)"),
            "the export must be released directly with the GIL token before returning"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests -- the borrowed-put guard (round 6).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod borrowed_put_guard_tests {
    use super::*;
    use pyo3::types::PyDict;

    /// A guard whose class constructor raises, over a plain asyncio future standing in for the
    /// spawned put.
    fn failing_guard<'py>(py: Python<'py>) -> (PreparedGuard<'py>, Bound<'py, PyAny>) {
        let locals = PyDict::new(py);
        py.run(
            c_str!(
                "import asyncio\nloop = asyncio.new_event_loop()\ninner = loop.create_future()\nclass Boom:\n    def __init__(self, *a):\n        raise RuntimeError('boom')\nnoop = lambda *a: None"
            ),
            None,
            Some(&locals),
        )
        .unwrap();
        let get = |k: &str| locals.get_item(k).unwrap().unwrap();
        (
            PreparedGuard {
                class: get("Boom"),
                latch_ready: get("noop"),
                quiesce: get("noop"),
                event: get("noop"),
                block: get("noop"),
            },
            get("inner"),
        )
    }

    /// Round 6, item 4: when the wrapper cannot be built after the put was spawned, the put is
    /// CANCELLED before the (blocking) fallback wait, and the original error is raised.
    #[test]
    fn constructor_failure_cancels_the_put_then_waits_and_raises_the_original_error() {
        Python::initialize();
        Python::attach(|py| {
            let (guard, inner) = failing_guard(py);
            let latch = Latch::new();
            latch.mark_dropped();
            latch.mark_released(); // no native reader: the fallback wait returns at once
            let err = wrap_in_guard(py, guard, inner.clone(), &latch).expect_err("must fail");
            assert!(err.to_string().contains("boom"), "{err}");
            let cancelled: bool = inner.call_method0("cancelled").unwrap().extract().unwrap();
            assert!(cancelled, "the already-spawned put was not cancelled");
        });
    }

    #[test]
    fn barrier_error_messages_name_the_timeout_and_the_put_outcome() {
        let m = still_referenced_message("the put itself failed: disk full");
        assert!(m.contains("30s"), "{m}");
        assert!(m.contains("the put itself failed: disk full"), "{m}");
        assert!(m.contains("do not modify"), "{m}");
        let i = barrier_internal_message("RuntimeError('x')");
        assert!(i.starts_with("internal error"), "{i}");
        assert!(
            i.contains("RuntimeError('x')") && i.contains("keeps waiting"),
            "{i}"
        );
        assert!(!i.contains("still references"), "{i}");
        let w = deadline_warning_text(3);
        assert!(
            w.contains("30s") && w.contains("#3") && w.contains("never completed"),
            "{w}"
        );
        let before = export_stats().internal_errors;
        note_internal_error();
        assert_eq!(export_stats().internal_errors, before + 1);
    }
}

// ---------------------------------------------------------------------------
// Tests -- caller-buffer GET, typed HEAD and pure capabilities (#264, stage B3b).
// ---------------------------------------------------------------------------

/// Counts the completions of `HeadQuiescence` tokens, per latch (test hook, see its `Drop`).
#[cfg(test)]
mod head_token_tests {
    use super::*;

    static COMPLETIONS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    pub(super) fn note_completion(latch: &Arc<Latch>) {
        COMPLETIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::as_ptr(latch) as usize);
    }

    /// Completions recorded for `latch` so far. An allocator may reuse a freed latch's address,
    /// so a test compares against the value it read before the call.
    pub(super) fn completions(latch: &Arc<Latch>) -> usize {
        let key = Arc::as_ptr(latch) as usize;
        COMPLETIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|seen| **seen == key)
            .count()
    }
}

#[cfg(test)]
mod caller_buffer_tests {
    use super::*;
    use crate::object_error::attach_object_error;
    use async_trait::async_trait;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// The URI handed to the internal functions; the fake store ignores it.
    const URI: &str = "s3://bucket/key";
    /// The canary byte every test buffer starts filled with.
    const CANARY: u8 = 0xCC;
    /// Far below the 30 s quiesce timeout: a test that took this long waited on a barrier.
    const NO_BARRIER_WAIT: Duration = Duration::from_secs(10);

    // ---- the Python side of the harness ----

    /// `Exporter` is a writable byte buffer that counts its exports and releases (PEP 688), so a
    /// test can assert "exactly one release". `run` drives one scenario on a fresh event loop
    /// with a hard timeout, so a broken barrier fails instead of hanging.
    const DRIVER_PY: &std::ffi::CStr = c_str!(
        "import asyncio
import ctypes

class Exporter:
    def __init__(self, n, fill=204):
        self.ba = bytearray([fill]) * n
        self.exports = 0
        self.releases = 0
    def __buffer__(self, flags):
        self.exports += 1
        return memoryview(self.ba)
    def __release_buffer__(self, view):
        self.releases += 1
        view.release()
    def address(self):
        c = (ctypes.c_char * len(self.ba)).from_buffer(self.ba)
        a = ctypes.addressof(c)
        del c
        return a
    def resizable(self):
        try:
            self.ba.append(0)
            self.ba.pop()
        except BufferError:
            return False
        return True

class Indexable:
    def __init__(self, value):
        self.value = value
    def __index__(self):
        return self.value

class BrokenIndex:
    def __index__(self):
        raise KeyError('index exploded')

def run(main, *args):
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(main(*args))
        timer = loop.call_later(20, loop.stop)
        try:
            return loop.run_until_complete(task)
        finally:
            timer.cancel()
    finally:
        loop.close()

async def scn_await(start):
    return await start()

async def scn_error(start):
    try:
        await start()
    except BaseException as e:
        return e
    raise AssertionError('expected an error')

async def scn_cancel_now(start):
    fut = start()
    cancel = fut.cancel()
    try:
        await fut
    except asyncio.CancelledError:
        return (cancel, 'cancelled')
    return (cancel, 'finished')

async def scn_cancel_when(start, entered):
    fut = start()
    while not entered():
        await asyncio.sleep(0.002)
    cancel = fut.cancel()
    try:
        await fut
    except asyncio.CancelledError:
        return (cancel, 'cancelled')
    return (cancel, 'finished')

async def scn_cancel_after(start):
    fut = start()
    value = await fut
    return (value, fut.cancel(), fut.done(), fut.cancelled())

async def scn_blocked_observe(start, entered, observe, release):
    fut = start()
    while not entered():
        await asyncio.sleep(0.002)
    await asyncio.sleep(0.05)
    unmarked = observe()
    pending = not fut.done()
    release()
    value = await fut
    return (unmarked, pending, value)

async def scn_error_then_settle(start):
    import gc
    loop = asyncio.get_running_loop()
    unretrieved = []
    loop.set_exception_handler(lambda lp, ctx: unretrieved.append(ctx.get('message', '')))
    error = None
    try:
        start()
    except BaseException as e:
        error = e
    for _ in range(3):
        await asyncio.sleep(0.02)
        gc.collect()
    return (error, unretrieved)
"
    );

    struct Driver<'py> {
        module: Bound<'py, PyModule>,
    }

    impl<'py> Driver<'py> {
        fn new(py: Python<'py>) -> Self {
            let version = py.version_info();
            assert!(
                (version.major, version.minor) >= (3, 12),
                "these tests need CPython >= 3.12 for the PEP 688 __buffer__ exporter, \
                 found {}.{}",
                version.major,
                version.minor
            );
            let module = PyModule::from_code(
                py,
                DRIVER_PY,
                c_str!("s3dlio_caller_buffer_driver.py"),
                c_str!("_s3dlio_caller_buffer_driver"),
            )
            .expect("driver module");
            Self { module }
        }

        fn exporter(&self, n: usize) -> Bound<'py, PyAny> {
            self.module
                .getattr("Exporter")
                .and_then(|class| class.call1((n, CANARY)))
                .expect("Exporter")
        }

        /// Run `scenario(start, *extra)` to completion on a fresh event loop.
        fn run(
            &self,
            scenario: &str,
            start: Bound<'py, PyAny>,
            extra: Vec<Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let py = self.module.py();
            let mut args = vec![self.module.getattr(scenario)?, start];
            args.extend(extra);
            self.module
                .getattr("run")?
                .call1(pyo3::types::PyTuple::new(py, args)?)
        }
    }

    fn attr_usize(obj: &Bound<'_, PyAny>, name: &str) -> usize {
        obj.getattr(name).unwrap().extract().unwrap()
    }

    fn method_bool(obj: &Bound<'_, PyAny>, name: &str) -> bool {
        obj.call_method0(name).unwrap().extract().unwrap()
    }

    fn contents(exporter: &Bound<'_, PyAny>) -> Vec<u8> {
        exporter.getattr("ba").unwrap().extract().unwrap()
    }

    fn latch_state(latch: &Latch) -> (bool, bool) {
        (latch.is_dropped(), latch.is_released())
    }

    /// Wait (GIL released) until `ready` holds, at most `NO_BARRIER_WAIT`.
    fn wait_until(py: Python<'_>, ready: impl Fn() -> bool + Send + Sync) -> bool {
        py.detach(|| {
            let deadline = Instant::now() + NO_BARRIER_WAIT;
            while Instant::now() < deadline {
                if ready() {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            ready()
        })
    }

    /// Let tasks that would start native work, if they could, run (GIL released).
    fn let_runtime_settle(py: Python<'_>) {
        py.detach(|| std::thread::sleep(Duration::from_millis(60)));
    }

    // ---- the drop probe ----

    /// What a [`DropProbe`] reads off the call's latch (and, for HEAD, the quiescence token).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ProbeSample {
        latch_dropped: bool,
        latch_released: bool,
        /// HEAD: completions of the call's `HeadQuiescence` token since the probe was made.
        token_completions: usize,
    }

    /// Nothing marked yet: the state the latch and the token must be in while the native future
    /// is alive and while its `Drop` runs.
    const UNMARKED: ProbeSample = ProbeSample {
        latch_dropped: false,
        latch_released: false,
        token_completions: 0,
    };

    /// Watches the instant the native future of a fake store call is dropped. The future the
    /// store returns (a [`ProbedRequest`]) owns a [`NativeFutureGuard`] as a field, so its `Drop`
    /// runs exactly when that future is dropped, not when the request body ends: after a
    /// completion, on cancellation and on a panic unwind. The probe reads the call's latch (and HEAD token) at that moment, and again after
    /// a deliberately slow `Drop`, so a test can prove that nothing was marked before the native
    /// future was really gone.
    struct DropProbe {
        latch: Arc<Latch>,
        /// `Drop` sleeps, widening any window in which the latch could be marked too early.
        slow: bool,
        token_baseline: usize,
        /// The request future's completion (its final poll returned `Ready`), when it completed.
        at_completion: Mutex<Option<ProbeSample>>,
        at_drop: Mutex<Option<ProbeSample>>,
        after_slow_drop: Mutex<Option<ProbeSample>>,
        /// `Drop` ran to its end.
        finished: AtomicBool,
    }

    impl DropProbe {
        fn new(latch: &Arc<Latch>, slow: bool) -> Arc<Self> {
            Arc::new(Self {
                latch: Arc::clone(latch),
                slow,
                token_baseline: head_token_tests::completions(latch),
                at_completion: Mutex::new(None),
                at_drop: Mutex::new(None),
                after_slow_drop: Mutex::new(None),
                finished: AtomicBool::new(false),
            })
        }

        fn sample(&self) -> ProbeSample {
            ProbeSample {
                latch_dropped: self.latch.is_dropped(),
                latch_released: self.latch.is_released(),
                token_completions: head_token_tests::completions(&self.latch) - self.token_baseline,
            }
        }

        /// The request future just completed: record the latch and token as they stand.
        fn note_completion(&self) {
            *self.at_completion.lock().unwrap_or_else(|e| e.into_inner()) = Some(self.sample());
        }

        /// The request future completed (it did not panic, and was not dropped unfinished).
        fn completed(&self) -> bool {
            self.at_completion
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
        }

        fn enter(self: &Arc<Self>) -> NativeFutureGuard {
            NativeFutureGuard(Arc::clone(self))
        }

        fn finished(&self) -> bool {
            self.finished.load(Ordering::SeqCst)
        }

        /// The native future was dropped, and neither the latch nor the HEAD token was marked
        /// when it was (nor, for a slow `Drop`, while that `Drop` was still running).
        fn assert_dropped_before_any_marking(&self, what: &str) {
            let at = *self.at_drop.lock().unwrap_or_else(|e| e.into_inner());
            let at = at.unwrap_or_else(|| panic!("{what}: the native future was never dropped"));
            assert_eq!(
                at, UNMARKED,
                "{what}: marked before the native future was dropped"
            );
            // A request that completed was unmarked at that moment too: the latch is marked by
            // the owner of the export, after the request future is gone, never by completion.
            let done = *self.at_completion.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(done) = done {
                assert_eq!(
                    done, UNMARKED,
                    "{what}: marked when the request completed, before it was dropped"
                );
            }
            if self.slow {
                let after = *self
                    .after_slow_drop
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let after = after.unwrap_or_else(|| panic!("{what}: the slow Drop never finished"));
                assert_eq!(
                    after, UNMARKED,
                    "{what}: marked while the native future's Drop was still running"
                );
            }
            assert!(
                self.finished(),
                "{what}: the native future's Drop did not finish"
            );
        }
    }

    /// The local of the fake store's native future that reports its drop to a [`DropProbe`].
    struct NativeFutureGuard(Arc<DropProbe>);

    impl Drop for NativeFutureGuard {
        fn drop(&mut self) {
            let probe = &self.0;
            *probe.at_drop.lock().unwrap_or_else(|e| e.into_inner()) = Some(probe.sample());
            if probe.slow {
                std::thread::sleep(Duration::from_millis(80));
                *probe
                    .after_slow_drop
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(probe.sample());
            }
            probe.finished.store(true, Ordering::SeqCst);
        }
    }

    // ---- the deterministic fake store ----

    #[derive(Clone, Copy)]
    enum Failure {
        NotFound,
        Length,
        LengthUnknown,
        Unsupported,
        Forbidden,
        VersionChanged,
        Plain,
    }

    fn failure_error(failure: Failure) -> anyhow::Error {
        let source = anyhow::anyhow!("sdk source");
        match failure {
            Failure::NotFound => attach_object_error(
                source,
                ObjectErrorKind::NotFound,
                "s3",
                "status 404, code NoSuchKey",
            ),
            Failure::Length => attach_object_error(
                source,
                ObjectErrorKind::LengthMismatch {
                    expected: 8,
                    actual: Some(5),
                },
                "s3",
                "",
            ),
            Failure::LengthUnknown => attach_object_error(
                source,
                ObjectErrorKind::LengthMismatch {
                    expected: 8,
                    actual: None,
                },
                "s3",
                "",
            ),
            Failure::Unsupported => {
                attach_object_error(source, ObjectErrorKind::Unsupported, "s3", "no")
            }
            Failure::Forbidden => {
                attach_object_error(source, ObjectErrorKind::Forbidden, "s3", "status 403")
            }
            Failure::VersionChanged => {
                attach_object_error(source, ObjectErrorKind::VersionChanged, "s3", "status 412")
            }
            Failure::Plain => anyhow::anyhow!("plain failure"),
        }
        .context("caller-buffer operation failed")
    }

    /// A store whose `get_into` and `head` are scripted. It records what it was given.
    struct FakeStore {
        /// Written to the front of the destination by a successful (or half of it by a failed)
        /// `get_into`.
        fill: Vec<u8>,
        /// Hold inside the native call until the test adds a permit to `gate`.
        block: bool,
        gate: tokio::sync::Semaphore,
        panic: bool,
        failure: Option<Failure>,
        /// What `head` returns on success: `None` is an absent object.
        head_result: Option<HeadMetadata>,
        get_calls: AtomicUsize,
        head_calls: AtomicUsize,
        entered: AtomicBool,
        /// `(address, length)` of the destination of every `get_into` call.
        seen: Mutex<Vec<(usize, usize)>>,
        /// `(releases, resizable)` of the destination observed INSIDE the start call, at the moment
        /// it returned an error (see `get_start`): proves the release was direct, not queued.
        error_probe: Mutex<Option<(usize, bool)>>,
        /// Reports the drop of the native future (see [`DropProbe`]).
        drop_probe: Option<Arc<DropProbe>>,
        /// Negative control only: finished requests park their guard here (see
        /// [`FakeStore::keeping_finished_requests`]).
        kept: Option<Arc<Mutex<Vec<NativeFutureGuard>>>>,
    }

    impl FakeStore {
        fn new() -> Self {
            Self {
                fill: vec![0x5A; 8],
                block: false,
                gate: tokio::sync::Semaphore::new(0),
                panic: false,
                failure: None,
                head_result: Some(HeadMetadata {
                    size: 8,
                    etag: Some("\"etag-1\"".to_string()),
                    version_id: Some("gen-7".to_string()),
                }),
                get_calls: AtomicUsize::new(0),
                head_calls: AtomicUsize::new(0),
                entered: AtomicBool::new(false),
                seen: Mutex::new(Vec::new()),
                error_probe: Mutex::new(None),
                drop_probe: None,
                kept: None,
            }
        }

        fn probing(mut self, probe: &Arc<DropProbe>) -> Self {
            self.drop_probe = Some(Arc::clone(probe));
            self
        }

        /// Negative control: a finished request hands its guard to the store instead of
        /// destroying it with the request future, as a caller that keeps the completed future
        /// alive would. The guard is destroyed only by [`FakeStore::release_kept`].
        fn keeping_finished_requests(mut self) -> Self {
            self.kept = Some(Arc::new(Mutex::new(Vec::new())));
            self
        }

        /// Destroy the guards [`FakeStore::keeping_finished_requests`] parked.
        fn release_kept(&self) {
            if let Some(kept) = &self.kept {
                kept.lock().unwrap_or_else(|e| e.into_inner()).clear();
            }
        }

        /// Wrap a scripted request in the hand-written future that owns the probe guard.
        fn probed<'a, T>(
            &self,
            request: impl std::future::Future<Output = anyhow::Result<T>> + Send + 'a,
        ) -> ProbedRequest<'a, T> {
            ProbedRequest {
                request: Box::pin(request),
                guard: self.drop_probe.as_ref().map(DropProbe::enter),
                keep: self.kept.clone(),
            }
        }

        fn filling(mut self, fill: Vec<u8>) -> Self {
            self.fill = fill;
            self
        }

        fn blocking(mut self) -> Self {
            self.block = true;
            self
        }

        fn panicking(mut self) -> Self {
            self.panic = true;
            self
        }

        fn failing(mut self, failure: Failure) -> Self {
            self.failure = Some(failure);
            self
        }

        fn absent(mut self) -> Self {
            self.head_result = None;
            self
        }

        fn get_calls(&self) -> usize {
            self.get_calls.load(Ordering::SeqCst)
        }

        fn head_calls(&self) -> usize {
            self.head_calls.load(Ordering::SeqCst)
        }

        fn seen(&self) -> Vec<(usize, usize)> {
            self.seen.lock().unwrap().clone()
        }

        fn error_probe(&self) -> Option<(usize, bool)> {
            *self.error_probe.lock().unwrap()
        }
    }

    /// The request future the fake store returns, written by hand. It OWNS the probe guard as a
    /// field, so the guard is destroyed exactly when this future is dropped (which is what the
    /// caller of `get_into` / `head` does with the future it was handed) and not when the
    /// request's body finishes: a guard that was a local of an `async fn` body would be destroyed
    /// in the final poll whether the body dropped it explicitly or let it go out of scope, and
    /// the test could not tell the two apart, nor see a caller that keeps the finished future.
    struct ProbedRequest<'a, T> {
        /// The scripted request itself.
        request: Pin<Box<dyn std::future::Future<Output = anyhow::Result<T>> + Send + 'a>>,
        /// Reports the moment this future is dropped (see [`DropProbe`]).
        guard: Option<NativeFutureGuard>,
        /// Where a completed request moves its guard instead of leaving it to this future's
        /// `Drop`. Only the negative control uses it (see [`FakeStore::keeping_finished_requests`]).
        keep: Option<Arc<Mutex<Vec<NativeFutureGuard>>>>,
    }

    impl<T> std::future::Future for ProbedRequest<'_, T> {
        type Output = anyhow::Result<T>;

        fn poll(
            mut self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            let this = &mut *self;
            let polled = this.request.as_mut().poll(cx);
            if polled.is_ready() {
                // The request finished; this future is still alive, and so is its guard.
                if let Some(guard) = &this.guard {
                    guard.0.note_completion();
                }
                assert!(
                    this.guard.as_ref().is_none_or(|guard| !guard.0.finished()),
                    "the native future guard was dropped before the call returned"
                );
                // Only the negative control takes the guard out; otherwise it stays a field and
                // is destroyed with this future.
                if let Some(keep) = &this.keep {
                    if let Some(guard) = this.guard.take() {
                        keep.lock().unwrap_or_else(|e| e.into_inner()).push(guard);
                    }
                }
            }
            polled
        }
    }

    // `get_into` and `head` return a hand-written future, so they are written in the desugared
    // form `#[async_trait]` expands an `async fn` to (it leaves a method that is not `async`
    // alone).
    #[async_trait]
    impl ObjectStore for FakeStore {
        fn get_into<'life0, 'life1, 'life2, 'life3, 'async_trait>(
            &'life0 self,
            _: &'life1 str,
            dst: &'life2 mut [u8],
            progress: &'life3 ReceiveProgress,
        ) -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'async_trait>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            'life2: 'async_trait,
            'life3: 'async_trait,
            Self: 'async_trait,
        {
            let request = async move {
                self.get_calls.fetch_add(1, Ordering::SeqCst);
                self.seen
                    .lock()
                    .unwrap()
                    .push((dst.as_ptr() as usize, dst.len()));
                self.entered.store(true, Ordering::SeqCst);
                if self.panic {
                    panic!("fake store panics inside the native future");
                }
                if self.block {
                    // Held until the test adds a permit; a dropped future just stops here.
                    let permit = self.gate.acquire().await?;
                    drop(permit);
                }
                let n = self.fill.len().min(dst.len());
                // A failure first writes half of what it would have: partial progress.
                let written = if self.failure.is_some() { n / 2 } else { n };
                dst[..written].copy_from_slice(&self.fill[..written]);
                crate::receive_progress::SpanMark::new(progress).record_copy(0, written as u64)?;
                match self.failure {
                    Some(failure) => Err(failure_error(failure)),
                    None => Ok(()),
                }
            };
            Box::pin(self.probed(request))
        }

        fn head<'life0, 'life1, 'async_trait>(
            &'life0 self,
            _: &'life1 str,
        ) -> Pin<
            Box<
                dyn std::future::Future<Output = anyhow::Result<Option<HeadMetadata>>>
                    + Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            let request = async move {
                self.head_calls.fetch_add(1, Ordering::SeqCst);
                self.entered.store(true, Ordering::SeqCst);
                if self.panic {
                    panic!("fake store panics inside the native future");
                }
                if self.block {
                    let permit = self.gate.acquire().await?;
                    drop(permit);
                }
                match self.failure {
                    Some(failure) => Err(failure_error(failure)),
                    None => Ok(self.head_result.clone()),
                }
            };
            Box::pin(self.probed(request))
        }

        async fn get(&self, _: &str) -> anyhow::Result<Bytes> {
            anyhow::bail!("not scripted")
        }
        async fn get_range(&self, _: &str, _: u64, _: Option<u64>) -> anyhow::Result<Bytes> {
            anyhow::bail!("not scripted")
        }
        async fn put(&self, _: &str, _: Bytes) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn put_multipart(&self, _: &str, _: Bytes, _: Option<usize>) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn list(&self, _: &str, _: bool) -> anyhow::Result<Vec<String>> {
            anyhow::bail!("not scripted")
        }
        fn list_stream<'a>(
            &'a self,
            _: &'a str,
            _: bool,
        ) -> Pin<Box<dyn futures::Stream<Item = anyhow::Result<String>> + Send + 'a>> {
            Box::pin(futures::stream::empty())
        }
        async fn stat(&self, _: &str) -> anyhow::Result<crate::object_store::ObjectMetadata> {
            anyhow::bail!("not scripted")
        }
        async fn delete(&self, _: &str) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn delete_batch(&self, _: &[String]) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn delete_prefix(&self, _: &str) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn create_container(&self, _: &str) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn delete_container(&self, _: &str) -> anyhow::Result<()> {
            anyhow::bail!("not scripted")
        }
        async fn get_writer(&self, _: &str) -> anyhow::Result<Box<dyn ObjectWriter>> {
            anyhow::bail!("not scripted")
        }
    }

    // ---- failure injection ----

    #[derive(Clone, Default)]
    enum BridgeMode {
        /// The real `future_into_py`.
        #[default]
        Live,
        /// `future_into_py` fails (the future is dropped unrun).
        Fail,
        /// The bridge "succeeds" but the Rust future is gone: the start receiver is closed.
        DropFuture,
        /// The real bridge, but the wrapped future is not polled until the test adds a permit to
        /// the semaphore. It models a task that is DROPPED UNPOLLED (the Rust future never ran a
        /// single step), the one case in which the export is still in the start channel. It is
        /// NOT what the live bridge does on a Python cancel: pyo3-async-runtimes polls the task
        /// once before it looks at the cancellation, so a live cancel can still start the call.
        Gated(Arc<tokio::sync::Semaphore>),
    }

    #[derive(Clone, Default)]
    struct TestSeams {
        bridge: BridgeMode,
        fail_instantiate: bool,
        /// Run just before the guard is instantiated, i.e. before the start permit or the
        /// export is sent.
        probe: Option<Arc<dyn Fn() + Send + Sync>>,
    }

    impl StartSeams for TestSeams {
        fn bridge<'py, F, T>(&self, py: Python<'py>, fut: F) -> PyResult<Bound<'py, PyAny>>
        where
            F: Future<Output = PyResult<T>> + Send + 'static,
            T: for<'a> IntoPyObject<'a> + Send + 'static,
        {
            match &self.bridge {
                BridgeMode::Live => future_into_py(py, fut),
                BridgeMode::Fail => {
                    drop(fut);
                    Err(PyRuntimeError::new_err("injected bridge failure"))
                }
                BridgeMode::DropFuture => {
                    drop(fut);
                    py.import("asyncio")?
                        .call_method0("get_running_loop")?
                        .call_method0("create_future")
                }
                BridgeMode::Gated(gate) => {
                    let gate = Arc::clone(gate);
                    future_into_py(py, async move {
                        let permit = gate
                            .acquire()
                            .await
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        let result = fut.await;
                        drop(permit);
                        result
                    })
                }
            }
        }

        fn instantiate<'py>(
            &self,
            py: Python<'py>,
            guard: PreparedGuard<'py>,
            inner: Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            if let Some(probe) = &self.probe {
                probe();
            }
            if self.fail_instantiate {
                drop(guard);
                return Err(PyRuntimeError::new_err("injected guard failure"));
            }
            guard.instantiate(py, inner)
        }
    }

    impl TestSeams {
        fn bridge(mut self, mode: BridgeMode) -> Self {
            self.bridge = mode;
            self
        }
        fn failing_guard(mut self) -> Self {
            self.fail_instantiate = true;
            self
        }
        /// Assert, just before the start permit/export is sent, that the store was never
        /// called (after giving a premature task time to call it).
        fn assert_store_untouched_before_start(mut self, store: &Arc<FakeStore>) -> Self {
            let store = Arc::clone(store);
            self.probe = Some(Arc::new(move || {
                std::thread::sleep(Duration::from_millis(60));
                assert_eq!(
                    store.get_calls(),
                    0,
                    "native GET started before the transfer"
                );
                assert_eq!(
                    store.head_calls(),
                    0,
                    "native HEAD started before the permit"
                );
            }));
            self
        }
    }

    /// The Python callable `start()` of one internal GET call.
    fn get_start<'py>(
        py: Python<'py>,
        seams: TestSeams,
        store: &Arc<FakeStore>,
        dst: &Bound<'py, PyAny>,
        expected_len: usize,
        latch: &Arc<Latch>,
    ) -> Bound<'py, PyAny> {
        let store = Arc::clone(store);
        let dst = dst.clone().unbind();
        let latch = Arc::clone(latch);
        pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, pyo3::types::PyTuple>,
                  _: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let py = args.py();
                let store_dyn: Arc<dyn ObjectStore> = store.clone();
                let result = get_into_with_store(
                    py,
                    &seams,
                    store_dyn,
                    URI.to_owned(),
                    dst.bind(py),
                    expected_len,
                    latch.clone(),
                );
                if result.is_err() {
                    let dst = dst.bind(py);
                    *store.error_probe.lock().unwrap() =
                        Some((attr_usize(dst, "releases"), method_bool(dst, "resizable")));
                }
                result.map(Bound::unbind)
            },
        )
        .expect("start closure")
        .into_any()
    }

    /// The Python callable `start()` of one internal HEAD call.
    fn head_start<'py>(
        py: Python<'py>,
        seams: TestSeams,
        store: &Arc<FakeStore>,
        latch: &Arc<Latch>,
    ) -> Bound<'py, PyAny> {
        let store = Arc::clone(store);
        let latch = Arc::clone(latch);
        pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, pyo3::types::PyTuple>,
                  _: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let store: Arc<dyn ObjectStore> = store.clone();
                head_with_store(args.py(), &seams, store, URI.to_owned(), latch.clone())
                    .map(Bound::unbind)
            },
        )
        .expect("start closure")
        .into_any()
    }

    /// The Python callable `entered()`: has the fake store been called?
    fn entered_fn<'py>(py: Python<'py>, store: &Arc<FakeStore>) -> Bound<'py, PyAny> {
        let store = Arc::clone(store);
        pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            move |_: &Bound<'_, pyo3::types::PyTuple>,
                  _: Option<&Bound<'_, PyDict>>|
                  -> PyResult<bool> { Ok(store.entered.load(Ordering::SeqCst)) },
        )
        .expect("entered closure")
        .into_any()
    }

    /// A Python callable `() -> bool` over a Rust closure.
    fn bool_fn<'py>(py: Python<'py>, f: impl Fn() -> bool + Send + 'static) -> Bound<'py, PyAny> {
        pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            move |_: &Bound<'_, pyo3::types::PyTuple>,
                  _: Option<&Bound<'_, PyDict>>|
                  -> PyResult<bool> { Ok(f()) },
        )
        .expect("bool closure")
        .into_any()
    }

    /// A Python callable `()` that runs a Rust closure and returns `None`.
    fn action_fn<'py>(py: Python<'py>, f: impl Fn() + Send + 'static) -> Bound<'py, PyAny> {
        pyo3::types::PyCFunction::new_closure(
            py,
            None,
            None,
            move |_: &Bound<'_, pyo3::types::PyTuple>,
                  _: Option<&Bound<'_, PyDict>>|
                  -> PyResult<()> {
                f();
                Ok(())
            },
        )
        .expect("action closure")
        .into_any()
    }

    fn init() {
        Python::initialize();
    }

    fn is_instance<T: pyo3::type_object::PyTypeInfo>(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
    ) -> bool {
        obj.is_instance(&py.get_type::<T>()).unwrap()
    }

    // =======================================================================
    // Pure capabilities
    // =======================================================================

    #[test]
    fn object_capabilities_returns_the_dict() {
        init();
        Python::attach(|py| {
            let native = cfg!(all(
                feature = "native-backends",
                not(feature = "arrow-backend")
            ));
            let get = |uri: &str| {
                let d = object_capabilities_py(py, uri).unwrap();
                let provider: String = d.get_item("provider").unwrap().unwrap().extract().unwrap();
                let get_into: bool = d.get_item("get_into").unwrap().unwrap().extract().unwrap();
                let head: bool = d.get_item("head").unwrap().unwrap().extract().unwrap();
                assert_eq!(d.len(), 3, "exactly provider, get_into, head");
                (provider, get_into, head)
            };
            assert_eq!(get("s3://bucket/key"), ("s3".to_string(), native, native));
            assert_eq!(get("file:///tmp/x"), ("file".to_string(), false, false));
            assert_eq!(get("direct:///tmp/x"), ("direct".to_string(), false, false));
            assert_eq!(get("ftp://host/x"), ("unknown".to_string(), false, false));
        });
    }

    #[test]
    fn object_capabilities_malformed_uri_is_a_value_error() {
        init();
        Python::attach(|py| {
            // An empty Azure container (#224) is malformed in every environment.
            let err = object_capabilities_py(py, "az://account//key").expect_err("malformed");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = object_capabilities_py(py, "s3://").expect_err("malformed");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
        });
    }

    // =======================================================================
    // Synchronous refusals: nothing is exported, no store is built
    // =======================================================================

    fn store_builds_for(uri: &str) -> usize {
        STORE_BUILD_LOG
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|built| built.as_str() == uri)
            .count()
    }

    #[test]
    fn unsupported_provider_raises_synchronously_before_any_export_or_store() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let n = 4i32.into_pyobject(py).unwrap().into_any();
            for uri in [
                "file:///tmp/s3dlio-b3b-unsupported",
                "direct:///tmp/s3dlio-b3b-x",
            ] {
                let ex = driver.exporter(16);
                let err = get_into_async(py, uri, &ex, &n).expect_err("unsupported");
                assert!(err.is_instance_of::<UnsupportedOperationError>(py), "{err}");
                assert!(err.is_instance_of::<PyNotImplementedError>(py), "{err}");
                assert_eq!(
                    attr_usize(&ex, "exports"),
                    0,
                    "{uri}: the buffer was exported"
                );
                assert_eq!(store_builds_for(uri), 0, "{uri}: a store was built");

                let err = head_async(py, uri).expect_err("unsupported");
                assert!(err.is_instance_of::<UnsupportedOperationError>(py), "{err}");
                assert_eq!(store_builds_for(uri), 0, "{uri}: a store was built");
            }
        });
    }

    #[test]
    fn malformed_uri_is_a_value_error_before_any_export() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let n = 4i32.into_pyobject(py).unwrap().into_any();
            let err = get_into_async(py, "az://account//key", &ex, &n).expect_err("malformed");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert_eq!(attr_usize(&ex, "exports"), 0);
            let err = head_async(py, "az://account//key").expect_err("malformed");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
        });
    }

    #[test]
    fn expected_len_is_validated_before_the_buffer_is_exported() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            // A unique host-form URI: a store built for it would show in the build log.
            let uri = "s3://127.0.0.1:59265/bucket/expected-len";
            let call = |value: Bound<'_, PyAny>| get_into_async(py, uri, &ex, &value).map(|_| ());
            let int = |v: i64| v.into_pyobject(py).unwrap().into_any();

            let err = call(int(-1)).expect_err("negative");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err =
                call(py.eval(c_str!("2 ** 200"), None, None).unwrap()).expect_err("too large");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err =
                call(true.into_pyobject(py).unwrap().to_owned().into_any()).expect_err("bool");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "{err}"
            );
            let err = call("4".into_pyobject(py).unwrap().into_any()).expect_err("str");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "{err}"
            );
            let err = call(1.5f64.into_pyobject(py).unwrap().into_any()).expect_err("float");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyTypeError>(py),
                "{err}"
            );

            // Every refusal above came before the buffer was exported and before any store was
            // built.
            assert_eq!(
                attr_usize(&ex, "exports"),
                0,
                "a refused length exported the buffer"
            );
            assert_eq!(store_builds_for(uri), 0, "a refused length built a store");
            assert!(method_bool(&ex, "resizable"));
        });
    }

    /// The conversion itself, over objects that go through `__index__`.
    #[test]
    fn expected_len_goes_through_index_and_keeps_the_original_failure() {
        init();
        Python::attach(|py| {
            use pyo3::exceptions::{PyKeyError, PyOverflowError, PyTypeError};
            let driver = Driver::new(py);
            fn indexable<'py>(driver: &Driver<'py>, value: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
                driver
                    .module
                    .getattr("Indexable")
                    .and_then(|class| class.call1((value,)))
                    .expect("Indexable")
            }
            let int = |v: i64| v.into_pyobject(py).unwrap().into_any();
            let huge = || py.eval(c_str!("2 ** 200"), None, None).unwrap();

            // An object with `__index__` is an integer.
            assert_eq!(
                checked_expected_len(&indexable(&driver, int(4))).unwrap(),
                4
            );

            // A float is a TypeError, with the interpreter's own error as the cause.
            let err = checked_expected_len(&1.5f64.into_pyobject(py).unwrap().into_any())
                .expect_err("float");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            let cause = err.cause(py).expect("the cause of a float is kept");
            assert!(cause.is_instance_of::<PyTypeError>(py), "{cause}");

            // An `__index__` beyond the machine range is a ValueError (not a TypeError), with the
            // OverflowError as the cause.
            let err =
                checked_expected_len(&indexable(&driver, huge())).expect_err("huge __index__");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert!(err.to_string().contains("too large"), "{err}");
            let cause = err.cause(py).expect("the cause of a huge value is kept");
            assert!(cause.is_instance_of::<PyOverflowError>(py), "{cause}");
            let err = checked_expected_len(&huge()).expect_err("huge int");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");

            // A negative `__index__` is a ValueError.
            let err =
                checked_expected_len(&indexable(&driver, int(-3))).expect_err("negative __index__");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");

            // An `__index__` that raises: that exception is what the caller sees, with its message,
            // not a generic replacement.
            let broken = driver
                .module
                .getattr("BrokenIndex")
                .and_then(|class| class.call0())
                .expect("BrokenIndex");
            let err = checked_expected_len(&broken).expect_err("raising __index__");
            assert!(err.is_instance_of::<PyKeyError>(py), "{err}");
            assert!(err.to_string().contains("index exploded"), "{err}");
        });
    }

    /// An `expected_len` longer than the buffer is refused by the export (which releases it),
    /// through the real internal path with a fake store: no store is built, no native call made.
    #[test]
    fn expected_len_longer_than_the_buffer_is_refused_and_released() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let store_dyn: Arc<dyn ObjectStore> = store.clone();
            let err = get_into_with_store(
                py,
                &LiveSeams,
                store_dyn,
                URI.to_owned(),
                &ex,
                17,
                latch.clone(),
            )
            .expect_err("longer than the buffer");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert_eq!(attr_usize(&ex, "exports"), 1);
            assert_eq!(
                attr_usize(&ex, "releases"),
                1,
                "the refused export is released"
            );
            assert!(method_bool(&ex, "resizable"));
            assert_eq!(latch_state(&latch), (true, true));
            let_runtime_settle(py);
            assert_eq!(store.get_calls(), 0);
        });
    }

    // =======================================================================
    // GET: ordinary completion, pointer identity, canaries, one release
    // =======================================================================

    #[test]
    fn get_completes_into_the_callers_memory_with_one_export_and_one_release() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let address: usize = ex.call_method0("address").unwrap().extract().unwrap();
            let store = Arc::new(FakeStore::new().filling(vec![0x5A; 8]));
            let latch = Latch::new();
            let seams = TestSeams::default().assert_store_untouched_before_start(&store);
            let start = get_start(py, seams, &store, &ex, 8, &latch);

            let value = driver.run("scn_await", start, vec![]).unwrap();
            assert_eq!(value.extract::<usize>().unwrap(), 8, "returns expected_len");

            // Pointer identity and length: the store wrote into the caller's allocation.
            assert_eq!(store.seen(), vec![(address, 8)]);
            // The data landed, and nothing past expected_len was touched.
            let bytes = contents(&ex);
            assert_eq!(&bytes[..8], &[0x5A; 8]);
            assert!(bytes[8..].iter().all(|&b| b == CANARY), "canaries damaged");
            // Exactly one export, released exactly once, on the call's one latch.
            assert_eq!(attr_usize(&ex, "exports"), 1);
            assert_eq!(attr_usize(&ex, "releases"), 1);
            assert_eq!(latch_state(&latch), (true, true));
            assert!(method_bool(&ex, "resizable"));
            assert_eq!(store.get_calls(), 1);
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    #[test]
    fn get_into_an_empty_span_passes_an_empty_slice_and_writes_nothing() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let start = get_start(py, TestSeams::default(), &store, &ex, 0, &latch);
            let value = driver.run("scn_await", start, vec![]).unwrap();
            assert_eq!(value.extract::<usize>().unwrap(), 0);
            let seen = store.seen();
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0].1, 0, "an empty destination");
            assert!(contents(&ex).iter().all(|&b| b == CANARY));
            assert_eq!(attr_usize(&ex, "releases"), 1);

            // A zero-length exporter has no (or a dangling) pointer: still fine.
            let empty = driver.exporter(0);
            let latch = Latch::new();
            let start = get_start(py, TestSeams::default(), &store, &empty, 0, &latch);
            driver.run("scn_await", start, vec![]).unwrap();
            assert_eq!(attr_usize(&empty, "releases"), 1);
        });
    }

    // =======================================================================
    // GET: pre-transfer failures (the start gate)
    // =======================================================================

    /// The invariants every pre-transfer failure shares: the ORIGINAL error, no native call,
    /// the buffer untouched, released exactly once and directly, reusable at once, the latch
    /// signalled, and no barrier wait.
    fn assert_get_pre_transfer_failure(
        py: Python<'_>,
        ex: &Bound<'_, PyAny>,
        store: &FakeStore,
        latch: &Latch,
        error: &Bound<'_, PyAny>,
        original: &str,
        started: Instant,
    ) {
        let text = error.str().unwrap().to_string();
        assert!(
            text.contains(original),
            "the original error was lost: {text}"
        );
        assert!(
            !is_instance::<ExportStillReferencedError>(py, error),
            "a quiescence timeout was raised: {text}"
        );
        assert_eq!(
            store.error_probe(),
            Some((1, true)),
            "the export was not released directly before the error was raised"
        );
        assert_eq!(attr_usize(ex, "exports"), 1);
        assert_eq!(attr_usize(ex, "releases"), 1, "exactly one release");
        assert_eq!(latch_state(latch), (true, true));
        assert!(
            method_bool(ex, "resizable"),
            "the buffer is not reusable right after the failure"
        );
        let_runtime_settle(py);
        assert_eq!(store.get_calls(), 0, "a native read started");
        assert!(
            contents(ex).iter().all(|&b| b == CANARY),
            "the destination was written"
        );
        assert!(started.elapsed() < NO_BARRIER_WAIT, "waited on a barrier");
    }

    #[test]
    fn get_guard_constructor_failure_releases_directly_and_raises_the_original_error() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let seams = TestSeams::default().failing_guard();
            let start = get_start(py, seams, &store, &ex, 8, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert_get_pre_transfer_failure(
                py,
                &ex,
                &store,
                &latch,
                &error,
                "injected guard failure",
                started,
            );
        });
    }

    #[test]
    fn get_bridge_failure_releases_directly_and_raises_the_original_error() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let seams = TestSeams::default().bridge(BridgeMode::Fail);
            let start = get_start(py, seams, &store, &ex, 8, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert_get_pre_transfer_failure(
                py,
                &ex,
                &store,
                &latch,
                &error,
                "injected bridge failure",
                started,
            );
        });
    }

    /// The REAL bridge fails when there is no running event loop: the same pre-transfer path,
    /// not injected.
    #[test]
    fn get_real_bridge_failure_without_an_event_loop_leaves_the_buffer_reusable() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let store_dyn: Arc<dyn ObjectStore> = store.clone();
            let err = get_into_with_store(
                py,
                &LiveSeams,
                store_dyn,
                URI.to_owned(),
                &ex,
                8,
                latch.clone(),
            )
            .expect_err("no running event loop");
            // Observed right after the call returned, as `get_start` does.
            *store.error_probe.lock().unwrap() =
                Some((attr_usize(&ex, "releases"), method_bool(&ex, "resizable")));
            let error = err.value(py).clone().into_any();
            assert_get_pre_transfer_failure(
                py,
                &ex,
                &store,
                &latch,
                &error,
                "no running event loop",
                started,
            );
        });
    }

    #[test]
    fn get_closed_start_receiver_releases_directly_and_cancels() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let seams = TestSeams::default().bridge(BridgeMode::DropFuture);
            let start = get_start(py, seams, &store, &ex, 8, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert_get_pre_transfer_failure(
                py,
                &ex,
                &store,
                &latch,
                &error,
                "could not hand the destination",
                started,
            );
        });
    }

    #[test]
    fn get_makes_no_native_call_before_the_export_is_transferred() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            // The probe runs before the send and asserts the (already spawned, live) task has
            // not called the store after 60 ms; the guard failure then keeps the send from ever
            // happening, so the store must never be called afterwards either.
            let seams = TestSeams::default()
                .assert_store_untouched_before_start(&store)
                .failing_guard();
            let start = get_start(py, seams, &store, &ex, 8, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert!(error
                .str()
                .unwrap()
                .to_string()
                .contains("injected guard failure"));
            let_runtime_settle(py);
            assert_eq!(store.get_calls(), 0);
            assert!(
                contents(&ex).iter().all(|&b| b == CANARY),
                "a native write happened"
            );
        });
    }

    // =======================================================================
    // GET: cancellation, panic
    // =======================================================================

    #[test]
    fn get_task_dropped_unpolled_never_starts_a_native_read() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            // The task is never polled (the gate never opens): it is dropped unpolled with the
            // export still in the start channel.
            let gate = Arc::new(tokio::sync::Semaphore::new(0));
            let seams = TestSeams::default().bridge(BridgeMode::Gated(gate));
            let start = get_start(py, seams, &store, &ex, 8, &latch);
            let outcome = driver.run("scn_cancel_now", start, vec![]).unwrap();
            let (cancel, state): (bool, String) = outcome.extract().unwrap();
            assert!(cancel, "cancel() is True while the work is pending");
            assert_eq!(state, "cancelled");
            let_runtime_settle(py);
            assert_eq!(
                store.get_calls(),
                0,
                "a native read started after the cancel"
            );
            // The export was transferred into the channel and dropped with it: queued, then
            // released by the guard's barrier, exactly once.
            assert_eq!(attr_usize(&ex, "exports"), 1);
            assert_eq!(attr_usize(&ex, "releases"), 1);
            assert_eq!(latch_state(&latch), (true, true));
            assert!(method_bool(&ex, "resizable"));
            assert!(contents(&ex).iter().all(|&b| b == CANARY));
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    /// With the live bridge a cancel right after the call may or may not beat the task's first
    /// poll; the invariants hold either way.
    #[test]
    fn get_cancel_immediately_after_the_call_always_ends_cancelled_and_released() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for _ in 0..20 {
                let ex = driver.exporter(16);
                let store = Arc::new(FakeStore::new());
                let latch = Latch::new();
                let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
                let outcome = driver.run("scn_cancel_now", start, vec![]).unwrap();
                let (cancel, state): (bool, String) = outcome.extract().unwrap();
                assert!(cancel);
                assert_eq!(state, "cancelled");
                assert_eq!(attr_usize(&ex, "releases"), 1);
                assert_eq!(latch_state(&latch), (true, true));
                assert!(method_bool(&ex, "resizable"));
            }
        });
    }

    #[test]
    fn get_cancel_during_the_native_read_ends_cancelled_after_the_writers_stopped() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new().blocking());
            let latch = Latch::new();
            let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
            let entered = entered_fn(py, &store);
            let outcome = driver.run("scn_cancel_when", start, vec![entered]).unwrap();
            let (cancel, state): (bool, String) = outcome.extract().unwrap();
            assert!(cancel, "cancel() is True while the read is pending");
            assert_eq!(state, "cancelled");
            assert_eq!(store.get_calls(), 1);
            assert_eq!(attr_usize(&ex, "releases"), 1);
            assert_eq!(latch_state(&latch), (true, true));
            assert!(method_bool(&ex, "resizable"));
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    #[test]
    fn get_cancel_is_false_once_the_outcome_is_fixed() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
            let outcome = driver.run("scn_cancel_after", start, vec![]).unwrap();
            let (value, cancel, done, cancelled): (usize, bool, bool, bool) =
                outcome.extract().unwrap();
            assert_eq!(value, 8);
            assert!(!cancel, "cancel() on a finished awaitable is False");
            assert!(done && !cancelled, "the result stands");
        });
    }

    #[test]
    fn get_panic_inside_the_native_future_is_delivered_and_the_buffer_released() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let store = Arc::new(FakeStore::new().panicking());
            let latch = Latch::new();
            let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            let text = error.str().unwrap().to_string();
            assert!(text.contains("panicked"), "{text}");
            assert!(
                !is_instance::<ExportStillReferencedError>(py, &error),
                "{text}"
            );
            assert_eq!(store.get_calls(), 1);
            assert_eq!(attr_usize(&ex, "releases"), 1);
            assert_eq!(latch_state(&latch), (true, true));
            assert!(method_bool(&ex, "resizable"));
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    // =======================================================================
    // GET: the native future is dropped BEFORE the latch is marked or the export released
    // =======================================================================

    /// How a probed call ends.
    #[derive(Clone, Copy, Debug)]
    enum Ending {
        Completion,
        CancelDuringTheNativeCall,
        Panic,
    }

    const ENDINGS: [Ending; 3] = [
        Ending::Completion,
        Ending::CancelDuringTheNativeCall,
        Ending::Panic,
    ];

    /// One live GET against a store whose native future carries a [`DropProbe`]; returns once the
    /// awaitable has ended, with the probe, the call's latch and the destination.
    fn run_get_with_probe<'py>(
        py: Python<'py>,
        driver: &Driver<'py>,
        ending: Ending,
        slow: bool,
    ) -> (Arc<DropProbe>, Arc<Latch>, Bound<'py, PyAny>) {
        let ex = driver.exporter(16);
        let latch = Latch::new();
        let probe = DropProbe::new(&latch, slow);
        let fake = FakeStore::new().probing(&probe);
        let store = Arc::new(match ending {
            Ending::Completion => fake,
            Ending::CancelDuringTheNativeCall => fake.blocking(),
            Ending::Panic => fake.panicking(),
        });
        let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
        match ending {
            Ending::Completion => {
                let value = driver.run("scn_await", start, vec![]).unwrap();
                assert_eq!(value.extract::<usize>().unwrap(), 8);
            }
            Ending::CancelDuringTheNativeCall => {
                let entered = entered_fn(py, &store);
                let outcome = driver.run("scn_cancel_when", start, vec![entered]).unwrap();
                let (cancel, state): (bool, String) = outcome.extract().unwrap();
                assert!(cancel, "cancel() is True while the read is pending");
                assert_eq!(state, "cancelled");
            }
            Ending::Panic => {
                let error = driver.run("scn_error", start, vec![]).unwrap();
                assert!(error.str().unwrap().to_string().contains("panicked"));
            }
        }
        (probe, latch, ex)
    }

    /// The load-bearing order of the GET task: the export (whose drop marks the latch) outlives
    /// the native `get_into` future on every path. The fake store's future reports the latch it
    /// sees at the moment it is dropped, and again after a slow `Drop`.
    #[test]
    fn get_native_future_is_dropped_before_the_latch_is_marked() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for ending in ENDINGS {
                for slow in [false, true] {
                    let what = format!("GET {ending:?}, slow drop {slow}");
                    let (probe, latch, ex) = run_get_with_probe(py, &driver, ending, slow);
                    probe.assert_dropped_before_any_marking(&what);
                    // Marked afterwards, and the export released exactly once.
                    assert_eq!(latch_state(&latch), (true, true), "{what}");
                    assert_eq!(attr_usize(&ex, "exports"), 1, "{what}");
                    assert_eq!(attr_usize(&ex, "releases"), 1, "{what}");
                    assert!(method_bool(&ex, "resizable"), "{what}");
                }
            }
        });
    }

    /// Normal completion, with the native future destroyed IMPLICITLY: the guard is a FIELD of the
    /// request future the fake returns, so it is destroyed only when that future is dropped. The
    /// probe also samples the latch when the request completes (it must be unmarked then) and
    /// the future asserts the guard is still alive at that point. Repeated, with a plain and a slow `Drop`, so the
    /// order is not one lucky schedule: the guard is gone before the latch is marked, the export
    /// is released once, and the call returned its value.
    #[test]
    fn get_completed_request_future_is_destroyed_implicitly_before_the_latch_is_marked() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for round in 0..10 {
                for slow in [false, true] {
                    let what = format!("GET completion round {round}, slow drop {slow}");
                    let (probe, latch, ex) =
                        run_get_with_probe(py, &driver, Ending::Completion, slow);
                    assert!(probe.completed(), "{what}: the request never completed");
                    assert!(probe.finished(), "{what}: the guard never ran its Drop");
                    probe.assert_dropped_before_any_marking(&what);
                    assert_eq!(latch_state(&latch), (true, true), "{what}");
                    assert_eq!(attr_usize(&ex, "releases"), 1, "{what}");
                }
            }
        });
    }

    /// Negative control for the implicit-destruction tests: a caller that KEEPS the finished
    /// request past the latch mark is caught. The fake parks the guard instead of destroying it
    /// with the request future; it is destroyed only after the call ended and the latch was
    /// marked, and the probe must report that.
    #[test]
    #[should_panic(expected = "marked before the native future was dropped")]
    fn get_a_finished_request_kept_past_the_latch_mark_is_detected() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let latch = Latch::new();
            let probe = DropProbe::new(&latch, false);
            let store = Arc::new(FakeStore::new().probing(&probe).keeping_finished_requests());
            let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
            driver.run("scn_await", start, vec![]).unwrap();
            assert_eq!(latch_state(&latch), (true, true));
            assert!(probe.completed() && !probe.finished());
            store.release_kept();
            probe.assert_dropped_before_any_marking("GET kept past the latch mark");
        });
    }

    /// While a blocked native request is in flight nothing of the call is marked, and the
    /// awaitable stays pending.
    #[test]
    fn get_latch_stays_unmarked_while_a_blocked_request_is_in_flight() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let latch = Latch::new();
            let probe = DropProbe::new(&latch, false);
            let store = Arc::new(FakeStore::new().blocking().probing(&probe));
            let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
            let entered = entered_fn(py, &store);
            let observed = Arc::clone(&probe);
            let observe = bool_fn(py, move || {
                observed.sample() == UNMARKED && !observed.finished()
            });
            let gate_store = Arc::clone(&store);
            let release = action_fn(py, move || gate_store.gate.add_permits(1));
            let outcome = driver
                .run(
                    "scn_blocked_observe",
                    start,
                    vec![entered, observe, release],
                )
                .unwrap();
            let (unmarked, pending, value): (bool, bool, usize) = outcome.extract().unwrap();
            assert!(
                unmarked,
                "the latch was marked while the request was in flight"
            );
            assert!(
                pending,
                "the awaitable finished while the request was in flight"
            );
            assert_eq!(value, 8);
            probe.assert_dropped_before_any_marking("GET blocked then released");
            assert_eq!(latch_state(&latch), (true, true));
            assert_eq!(attr_usize(&ex, "releases"), 1);
        });
    }

    /// The LIVE path: pyo3-async-runtimes polls the task once before it checks for a cancel, so
    /// a Python cancel right after the call may or may not start the request. Either way the
    /// awaitable ends cancelled, the export is released exactly once, and only after any native
    /// future was dropped. (That the request is never started is true only for a task dropped
    /// unpolled; see `get_task_dropped_unpolled_never_starts_a_native_read`.)
    #[test]
    fn get_live_cancel_right_after_the_call_releases_once_after_any_native_drop() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for round in 0..20 {
                let ex = driver.exporter(16);
                let latch = Latch::new();
                let probe = DropProbe::new(&latch, round % 2 == 1);
                let store = Arc::new(FakeStore::new().blocking().probing(&probe));
                let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
                let outcome = driver.run("scn_cancel_now", start, vec![]).unwrap();
                let (cancel, state): (bool, String) = outcome.extract().unwrap();
                assert!(cancel, "round {round}");
                assert_eq!(state, "cancelled", "round {round}");
                assert_eq!(attr_usize(&ex, "exports"), 1, "round {round}");
                assert_eq!(attr_usize(&ex, "releases"), 1, "round {round}");
                assert_eq!(latch_state(&latch), (true, true), "round {round}");
                assert!(method_bool(&ex, "resizable"), "round {round}");
                if store.get_calls() == 1 {
                    // The request did start (the task was polled before the cancel landed): its
                    // future was dropped before the latch was marked.
                    probe.assert_dropped_before_any_marking(&format!("live round {round}"));
                } else {
                    assert_eq!(store.get_calls(), 0, "round {round}");
                    assert!(!probe.finished(), "round {round}: a native future ran");
                }
            }
        });
    }

    /// A guard that cannot be built after the task was bridged must not leave an exception for
    /// nobody to retrieve: the bridged future is cancelled before the start channel is closed.
    /// (A race detector: the old order lost the race only sometimes, so it runs many rounds.)
    #[test]
    fn get_instantiate_failure_leaves_no_unretrieved_exception() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for round in 0..25 {
                let ex = driver.exporter(16);
                let store = Arc::new(FakeStore::new());
                let latch = Latch::new();
                let seams = TestSeams::default().failing_guard();
                let start = get_start(py, seams, &store, &ex, 8, &latch);
                let outcome = driver.run("scn_error_then_settle", start, vec![]).unwrap();
                let (error, unretrieved): (Bound<'_, PyAny>, Vec<String>) =
                    outcome.extract().unwrap();
                assert!(error
                    .str()
                    .unwrap()
                    .to_string()
                    .contains("injected guard failure"));
                assert!(
                    unretrieved.is_empty(),
                    "round {round}: the loop logged {unretrieved:?}"
                );
                assert_eq!(attr_usize(&ex, "releases"), 1, "round {round}");
                assert_eq!(store.get_calls(), 0, "round {round}");
            }
        });
    }

    // =======================================================================
    // GET and HEAD: error mapping, end to end
    // =======================================================================

    /// One row of [`MAPPING`]: `(failure, expected class name, expected, actual)`.
    type MappingRow = (Failure, &'static str, Option<u64>, Option<u64>);

    const MAPPING: &[MappingRow] = &[
        (Failure::NotFound, "ObjectNotFoundError", None, None),
        (Failure::Length, "LengthMismatchError", Some(8), Some(5)),
        (Failure::LengthUnknown, "LengthMismatchError", Some(8), None),
        (
            Failure::Unsupported,
            "UnsupportedOperationError",
            None,
            None,
        ),
        (Failure::Forbidden, "RuntimeError", None, None),
        (Failure::VersionChanged, "RuntimeError", None, None),
        (Failure::Plain, "RuntimeError", None, None),
    ];

    fn assert_mapped(
        py: Python<'_>,
        error: &Bound<'_, PyAny>,
        class: &str,
        expected: Option<u64>,
        actual: Option<u64>,
    ) {
        let name: String = error.get_type().name().unwrap().to_string();
        assert_eq!(name, class, "{error}");
        match class {
            "ObjectNotFoundError" => assert!(is_instance::<PyFileNotFoundError>(py, error)),
            "LengthMismatchError" => {
                assert!(is_instance::<PyValueError>(py, error));
                let got_expected: Option<u64> =
                    error.getattr("expected").unwrap().extract().unwrap();
                let got_actual: Option<u64> = error.getattr("actual").unwrap().extract().unwrap();
                assert_eq!(got_expected, expected, "{error}");
                assert_eq!(got_actual, actual, "{error}");
            }
            "UnsupportedOperationError" => {
                assert!(is_instance::<PyNotImplementedError>(py, error))
            }
            _ => {
                assert!(!is_instance::<ObjectNotFoundError>(py, error));
                assert!(!is_instance::<LengthMismatchError>(py, error));
                assert!(!is_instance::<UnsupportedOperationError>(py, error));
            }
        }
        // The message is the whole chain: the outer context is kept.
        let text = error.str().unwrap().to_string();
        assert!(text.contains("caller-buffer operation failed"), "{text}");
    }

    #[test]
    fn get_failures_map_to_the_typed_python_exceptions() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for &(failure, class, expected, actual) in MAPPING {
                let ex = driver.exporter(16);
                let store = Arc::new(FakeStore::new().failing(failure));
                let latch = Latch::new();
                let start = get_start(py, TestSeams::default(), &store, &ex, 8, &latch);
                let error = driver.run("scn_error", start, vec![]).unwrap();
                assert_mapped(py, &error, class, expected, actual);
                // After a failure the buffer is still released and reusable, and the partial
                // write stays inside expected_len.
                assert_eq!(attr_usize(&ex, "releases"), 1, "{class}");
                assert!(method_bool(&ex, "resizable"), "{class}");
                assert!(contents(&ex)[8..16].iter().all(|&b| b == CANARY), "{class}");
            }
        });
    }

    #[test]
    fn head_failures_map_to_the_typed_python_exceptions() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for &(failure, class, expected, actual) in MAPPING {
                let store = Arc::new(FakeStore::new().failing(failure));
                let latch = Latch::new();
                let start = head_start(py, TestSeams::default(), &store, &latch);
                let error = driver.run("scn_error", start, vec![]).unwrap();
                assert_mapped(py, &error, class, expected, actual);
                assert_eq!(latch_state(&latch), (true, true), "{class}");
            }
        });
    }

    /// The mapping itself, over every kind and through nested string contexts.
    #[test]
    fn object_py_err_maps_every_kind() {
        init();
        Python::attach(|py| {
            use anyhow::Context;
            use ObjectErrorKind::*;
            let kinds = [
                (NotFound, "ObjectNotFoundError"),
                (
                    LengthMismatch {
                        expected: 3,
                        actual: Some(4),
                    },
                    "LengthMismatchError",
                ),
                (Unsupported, "UnsupportedOperationError"),
                (VersionChanged, "RuntimeError"),
                (Forbidden, "RuntimeError"),
                (Throttled, "RuntimeError"),
                (Transport, "RuntimeError"),
                (Server, "RuntimeError"),
                (Other, "RuntimeError"),
            ];
            for (kind, class) in kinds {
                let typed = attach_object_error(anyhow::anyhow!("source"), kind, "gs", "msg");
                let nested = Err::<(), _>(typed)
                    .context("inner string context")
                    .context("outer string context")
                    .unwrap_err();
                let err = object_py_err(py, &nested);
                assert_eq!(err.get_type(py).name().unwrap().to_string(), class);
                let text = err.value(py).str().unwrap().to_string();
                assert!(text.contains("outer string context"), "{text}");
                assert!(text.contains("source"), "the whole chain is kept: {text}");
            }
            // The outermost classification wins: VersionChanged over a NotFound is not absence.
            let inner = attach_object_error(anyhow::anyhow!("s"), NotFound, "s3", "404");
            let outer = attach_object_error(inner, VersionChanged, "s3", "pinned");
            assert_eq!(
                object_py_err(py, &outer)
                    .get_type(py)
                    .name()
                    .unwrap()
                    .to_string(),
                "RuntimeError"
            );
            // No ObjectError at all: today's behaviour.
            assert!(
                object_py_err(py, &anyhow::anyhow!("plain")).is_instance_of::<PyRuntimeError>(py)
            );
        });
    }

    // =======================================================================
    // HEAD
    // =======================================================================

    /// The invariants of every HEAD that never started a request or ended without finishing it:
    /// exactly one quiescence-token completion, both terminal latch signals, no buffer.
    fn assert_head_quiesced(py: Python<'_>, latch: &Arc<Latch>, before: usize) {
        let l = Arc::clone(latch);
        assert!(
            wait_until(py, move || l.is_dropped() && l.is_released()),
            "the quiescence token never completed"
        );
        // Let a second (wrong) completion show up, then count.
        let_runtime_settle(py);
        assert_eq!(
            head_token_tests::completions(latch) - before,
            1,
            "exactly one quiescence-token completion"
        );
        assert_eq!(latch_state(latch), (true, true));
    }

    #[test]
    fn head_completes_with_metadata_one_token_completion_and_no_buffer() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let seams = TestSeams::default().assert_store_untouched_before_start(&store);
            let start = head_start(py, seams, &store, &latch);
            let meta = driver.run("scn_await", start, vec![]).unwrap();
            assert_eq!(
                meta.get_type().name().unwrap().to_string(),
                "ObjectMetadata"
            );
            assert_eq!(attr_usize(&meta, "size"), 8);
            assert_eq!(
                meta.getattr("etag")
                    .unwrap()
                    .extract::<Option<String>>()
                    .unwrap(),
                Some("\"etag-1\"".to_string())
            );
            assert_eq!(
                meta.getattr("version_id")
                    .unwrap()
                    .extract::<Option<String>>()
                    .unwrap(),
                Some("gen-7".to_string())
            );
            assert_eq!(store.head_calls(), 1);
            assert_eq!(store.get_calls(), 0);
            assert_head_quiesced(py, &latch, before);
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    #[test]
    fn head_of_an_absent_object_is_none() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new().absent());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let start = head_start(py, TestSeams::default(), &store, &latch);
            let result = driver.run("scn_await", start, vec![]).unwrap();
            assert!(result.is_none());
            assert_head_quiesced(py, &latch, before);
        });
    }

    fn assert_head_pre_start_failure(
        py: Python<'_>,
        store: &FakeStore,
        latch: &Arc<Latch>,
        before: usize,
        error: &Bound<'_, PyAny>,
        original: &str,
        started: Instant,
    ) {
        let text = error.str().unwrap().to_string();
        assert!(
            text.contains(original),
            "the original error was lost: {text}"
        );
        assert!(
            !is_instance::<ExportStillReferencedError>(py, error),
            "{text}"
        );
        assert_head_quiesced(py, latch, before);
        assert_eq!(store.head_calls(), 0, "a native HEAD started");
        assert!(started.elapsed() < NO_BARRIER_WAIT, "waited on a barrier");
    }

    #[test]
    fn head_guard_constructor_failure_cancels_and_the_token_marks_the_latch() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let seams = TestSeams::default().failing_guard();
            let start = head_start(py, seams, &store, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert_head_pre_start_failure(
                py,
                &store,
                &latch,
                before,
                &error,
                "injected guard failure",
                started,
            );
        });
    }

    #[test]
    fn head_bridge_failure_drops_the_future_and_the_token_marks_the_latch() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let seams = TestSeams::default().bridge(BridgeMode::Fail);
            let start = head_start(py, seams, &store, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert_head_pre_start_failure(
                py,
                &store,
                &latch,
                before,
                &error,
                "injected bridge failure",
                started,
            );
        });
    }

    #[test]
    fn head_real_bridge_failure_without_an_event_loop() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let store_dyn: Arc<dyn ObjectStore> = store.clone();
            let err = head_with_store(py, &LiveSeams, store_dyn, URI.to_owned(), latch.clone())
                .expect_err("no running event loop");
            let error = err.value(py).clone().into_any();
            assert_head_pre_start_failure(
                py,
                &store,
                &latch,
                before,
                &error,
                "no running event loop",
                started,
            );
        });
    }

    #[test]
    fn head_closed_start_receiver_cancels_and_the_token_marks_the_latch() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let seams = TestSeams::default().bridge(BridgeMode::DropFuture);
            let start = head_start(py, seams, &store, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            assert_head_pre_start_failure(
                py,
                &store,
                &latch,
                before,
                &error,
                "could not start its request",
                started,
            );
        });
    }

    #[test]
    fn head_task_never_polled_never_starts_a_request() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let gate = Arc::new(tokio::sync::Semaphore::new(0));
            let seams = TestSeams::default().bridge(BridgeMode::Gated(gate));
            let start = head_start(py, seams, &store, &latch);
            let outcome = driver.run("scn_cancel_now", start, vec![]).unwrap();
            let (cancel, state): (bool, String) = outcome.extract().unwrap();
            assert!(cancel);
            assert_eq!(state, "cancelled");
            assert_head_quiesced(py, &latch, before);
            assert_eq!(store.head_calls(), 0, "a request started after the cancel");
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    #[test]
    fn head_cancel_during_the_request_ends_cancelled_after_the_token_completed() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new().blocking());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let start = head_start(py, TestSeams::default(), &store, &latch);
            let entered = entered_fn(py, &store);
            let outcome = driver.run("scn_cancel_when", start, vec![entered]).unwrap();
            let (cancel, state): (bool, String) = outcome.extract().unwrap();
            assert!(cancel, "cancel() is True while the request is pending");
            assert_eq!(state, "cancelled");
            assert_eq!(store.head_calls(), 1);
            assert_head_quiesced(py, &latch, before);
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    #[test]
    fn head_cancel_is_false_once_the_outcome_is_fixed() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new());
            let latch = Latch::new();
            let start = head_start(py, TestSeams::default(), &store, &latch);
            let outcome = driver.run("scn_cancel_after", start, vec![]).unwrap();
            let (value, cancel, done, cancelled): (Bound<'_, PyAny>, bool, bool, bool) =
                outcome.extract().unwrap();
            assert_eq!(attr_usize(&value, "size"), 8);
            assert!(!cancel);
            assert!(done && !cancelled);
        });
    }

    #[test]
    fn head_panic_inside_the_native_future_is_delivered_and_the_token_completes() {
        init();
        Python::attach(|py| {
            let started = Instant::now();
            let driver = Driver::new(py);
            let store = Arc::new(FakeStore::new().panicking());
            let latch = Latch::new();
            let before = head_token_tests::completions(&latch);
            let start = head_start(py, TestSeams::default(), &store, &latch);
            let error = driver.run("scn_error", start, vec![]).unwrap();
            let text = error.str().unwrap().to_string();
            assert!(text.contains("panicked"), "{text}");
            assert!(
                !is_instance::<ExportStillReferencedError>(py, &error),
                "{text}"
            );
            assert_head_quiesced(py, &latch, before);
            assert!(started.elapsed() < NO_BARRIER_WAIT);
        });
    }

    // =======================================================================
    // HEAD: the native future is dropped BEFORE the quiescence token completes
    // =======================================================================

    /// One live HEAD against a store whose native future carries a [`DropProbe`].
    fn run_head_with_probe<'py>(
        py: Python<'py>,
        driver: &Driver<'py>,
        ending: Ending,
        slow: bool,
    ) -> (Arc<DropProbe>, Arc<Latch>) {
        let latch = Latch::new();
        let probe = DropProbe::new(&latch, slow);
        let fake = FakeStore::new().probing(&probe);
        let store = Arc::new(match ending {
            Ending::Completion => fake,
            Ending::CancelDuringTheNativeCall => fake.blocking(),
            Ending::Panic => fake.panicking(),
        });
        let start = head_start(py, TestSeams::default(), &store, &latch);
        match ending {
            Ending::Completion => {
                let meta = driver.run("scn_await", start, vec![]).unwrap();
                assert_eq!(attr_usize(&meta, "size"), 8);
            }
            Ending::CancelDuringTheNativeCall => {
                let entered = entered_fn(py, &store);
                let outcome = driver.run("scn_cancel_when", start, vec![entered]).unwrap();
                let (cancel, state): (bool, String) = outcome.extract().unwrap();
                assert!(cancel, "cancel() is True while the request is pending");
                assert_eq!(state, "cancelled");
            }
            Ending::Panic => {
                let error = driver.run("scn_error", start, vec![]).unwrap();
                assert!(error.str().unwrap().to_string().contains("panicked"));
            }
        }
        (probe, latch)
    }

    /// The load-bearing order of the HEAD task: the `HeadQuiescence` token outlives the native
    /// `head` future on every path, so neither latch signal nor the token's completion is seen
    /// while that future is alive or while its `Drop` runs.
    #[test]
    fn head_native_future_is_dropped_before_the_token_completes() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for ending in ENDINGS {
                for slow in [false, true] {
                    let what = format!("HEAD {ending:?}, slow drop {slow}");
                    let (probe, latch) = run_head_with_probe(py, &driver, ending, slow);
                    probe.assert_dropped_before_any_marking(&what);
                    // Completed exactly once afterwards, with both latch signals.
                    assert_head_quiesced(py, &latch, probe.token_baseline);
                }
            }
        });
    }

    /// HEAD twin of `get_completed_request_future_is_destroyed_implicitly_before_the_latch_is_marked`:
    /// the fake `head` leaves its guard to be destroyed implicitly, and the token completes
    /// exactly once, after it.
    #[test]
    fn head_completed_request_future_is_destroyed_implicitly_before_the_token_completes() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for round in 0..10 {
                for slow in [false, true] {
                    let what = format!("HEAD completion round {round}, slow drop {slow}");
                    let (probe, latch) = run_head_with_probe(py, &driver, Ending::Completion, slow);
                    assert!(probe.completed(), "{what}: the request never completed");
                    assert!(probe.finished(), "{what}: the guard never ran its Drop");
                    probe.assert_dropped_before_any_marking(&what);
                    assert_head_quiesced(py, &latch, probe.token_baseline);
                }
            }
        });
    }

    /// HEAD twin of `get_a_finished_request_kept_past_the_latch_mark_is_detected`.
    #[test]
    #[should_panic(expected = "marked before the native future was dropped")]
    fn head_a_finished_request_kept_past_the_token_completion_is_detected() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let latch = Latch::new();
            let probe = DropProbe::new(&latch, false);
            let store = Arc::new(FakeStore::new().probing(&probe).keeping_finished_requests());
            let start = head_start(py, TestSeams::default(), &store, &latch);
            driver.run("scn_await", start, vec![]).unwrap();
            assert_head_quiesced(py, &latch, probe.token_baseline);
            assert!(probe.completed() && !probe.finished());
            store.release_kept();
            probe.assert_dropped_before_any_marking("HEAD kept past the token completion");
        });
    }
    #[test]
    fn head_latch_stays_unmarked_while_a_blocked_request_is_in_flight() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            let latch = Latch::new();
            let probe = DropProbe::new(&latch, false);
            let store = Arc::new(FakeStore::new().blocking().probing(&probe));
            let start = head_start(py, TestSeams::default(), &store, &latch);
            let entered = entered_fn(py, &store);
            let observed = Arc::clone(&probe);
            let observe = bool_fn(py, move || {
                observed.sample() == UNMARKED && !observed.finished()
            });
            let gate_store = Arc::clone(&store);
            let release = action_fn(py, move || gate_store.gate.add_permits(1));
            let outcome = driver
                .run(
                    "scn_blocked_observe",
                    start,
                    vec![entered, observe, release],
                )
                .unwrap();
            let (unmarked, pending, meta): (bool, bool, Bound<'_, PyAny>) =
                outcome.extract().unwrap();
            assert!(
                unmarked,
                "the token or latch was marked while the request was in flight"
            );
            assert!(
                pending,
                "the awaitable finished while the request was in flight"
            );
            assert_eq!(attr_usize(&meta, "size"), 8);
            probe.assert_dropped_before_any_marking("HEAD blocked then released");
            assert_head_quiesced(py, &latch, probe.token_baseline);
        });
    }

    /// The LIVE path (see `get_live_cancel_right_after_the_call_releases_once_after_any_native_drop`):
    /// a cancel right after the call may or may not start the request; the awaitable ends
    /// cancelled and the token completes once, after any native future was dropped.
    #[test]
    fn head_live_cancel_right_after_the_call_completes_the_token_once_after_any_native_drop() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for round in 0..20 {
                let latch = Latch::new();
                let probe = DropProbe::new(&latch, round % 2 == 1);
                let store = Arc::new(FakeStore::new().blocking().probing(&probe));
                let start = head_start(py, TestSeams::default(), &store, &latch);
                let outcome = driver.run("scn_cancel_now", start, vec![]).unwrap();
                let (cancel, state): (bool, String) = outcome.extract().unwrap();
                assert!(cancel, "round {round}");
                assert_eq!(state, "cancelled", "round {round}");
                assert_head_quiesced(py, &latch, probe.token_baseline);
                if store.head_calls() == 1 {
                    probe.assert_dropped_before_any_marking(&format!("live round {round}"));
                } else {
                    assert_eq!(store.head_calls(), 0, "round {round}");
                    assert!(!probe.finished(), "round {round}: a native future ran");
                }
            }
        });
    }

    #[test]
    fn head_instantiate_failure_leaves_no_unretrieved_exception() {
        init();
        Python::attach(|py| {
            let driver = Driver::new(py);
            for round in 0..25 {
                let store = Arc::new(FakeStore::new());
                let latch = Latch::new();
                let before = head_token_tests::completions(&latch);
                let seams = TestSeams::default().failing_guard();
                let start = head_start(py, seams, &store, &latch);
                let outcome = driver.run("scn_error_then_settle", start, vec![]).unwrap();
                let (error, unretrieved): (Bound<'_, PyAny>, Vec<String>) =
                    outcome.extract().unwrap();
                assert!(error
                    .str()
                    .unwrap()
                    .to_string()
                    .contains("injected guard failure"));
                assert!(
                    unretrieved.is_empty(),
                    "round {round}: the loop logged {unretrieved:?}"
                );
                assert_head_quiesced(py, &latch, before);
                assert_eq!(store.head_calls(), 0, "round {round}");
            }
        });
    }

    // =======================================================================
    // ObjectMetadata
    // =======================================================================

    #[test]
    fn object_metadata_is_frozen_hashable_and_has_a_repr() {
        init();
        Python::attach(|py| {
            let make = |size: u64, etag: Option<&str>, version: Option<&str>| {
                Py::new(
                    py,
                    PyHeadMetadata {
                        size,
                        etag: etag.map(str::to_owned),
                        version_id: version.map(str::to_owned),
                    },
                )
                .unwrap()
                .into_bound(py)
                .into_any()
            };
            let a = make(5, Some("e"), None);
            let b = make(5, Some("e"), None);
            let c = make(6, Some("e"), None);
            assert!(a.eq(&b).unwrap());
            assert!(!a.eq(&c).unwrap());
            assert_eq!(a.hash().unwrap(), b.hash().unwrap());
            assert_eq!(attr_usize(&a, "size"), 5);
            assert!(a.getattr("version_id").unwrap().is_none());
            // Python's own repr for the strings, never Rust's `Some("e")`.
            assert_eq!(
                a.repr().unwrap().to_string(),
                "ObjectMetadata(size=5, etag='e', version_id=None)"
            );
            assert_eq!(
                make(0, None, Some("v1")).repr().unwrap().to_string(),
                "ObjectMetadata(size=0, etag=None, version_id='v1')"
            );
            // Quotes and backslashes are escaped the Python way.
            assert_eq!(
                make(5, Some("a'b\"c\\d"), None).repr().unwrap().to_string(),
                r#"ObjectMetadata(size=5, etag='a\'b"c\\d', version_id=None)"#
            );
            assert_eq!(
                make(5, Some("it's"), Some("say \"x\""))
                    .repr()
                    .unwrap()
                    .to_string(),
                r#"ObjectMetadata(size=5, etag="it's", version_id='say "x"')"#
            );
            let err = a.setattr("size", 9).expect_err("frozen");
            assert!(
                err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py),
                "{err}"
            );
            let set = pyo3::types::PySet::new(py, [a.clone(), b, c]).unwrap();
            assert_eq!(set.len(), 2);
        });
    }

    // =======================================================================
    // The Python surface, through the real module registration and pyo3 argument parsing
    // =======================================================================

    const SURFACE_PY: &std::ffi::CStr = c_str!(
        "def positional_expected_len_is_refused(core):
    try:
        core.get_into_async('file:///x', bytearray(1), 5)
    except TypeError as e:
        return str(e)
    raise AssertionError('a positional expected_len was accepted')

def expected_len_is_required(core):
    try:
        core.get_into_async('file:///x', bytearray(1))
    except TypeError as e:
        return str(e)
    raise AssertionError('a missing expected_len was accepted')

def exception_classes(core):
    assert issubclass(core.ObjectNotFoundError, FileNotFoundError)
    assert issubclass(core.ObjectNotFoundError, OSError)
    assert core.ObjectNotFoundError is not FileNotFoundError
    assert issubclass(core.LengthMismatchError, ValueError)
    assert issubclass(core.UnsupportedOperationError, NotImplementedError)
    assert not issubclass(core.UnsupportedOperationError, ValueError)
    return True

def metadata_is_immutable_and_hashable(a, b, c):
    assert type(a).__name__ == 'ObjectMetadata'
    assert a == b and a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2
    for name, value in (('size', 9), ('etag', 'x'), ('version_id', 'y')):
        try:
            setattr(a, name, value)
        except AttributeError:
            pass
        else:
            raise AssertionError('%s is assignable' % name)
    try:
        a.extra = 1
    except AttributeError:
        pass
    else:
        raise AssertionError('a new attribute can be added')
    return repr(a)

def capabilities_dict(core):
    d = core.object_capabilities('file:///x')
    assert type(d) is dict, type(d)
    assert d == {'provider': 'file', 'get_into': False, 'head': False}, d
    return True

def unsupported_is_raised_synchronously(core):
    buf = bytearray(1)
    try:
        core.get_into_async('file:///x', buf, expected_len=1)
    except core.UnsupportedOperationError as e:
        assert isinstance(e, NotImplementedError)
    else:
        raise AssertionError('get_into_async did not raise UnsupportedOperationError')
    buf.append(0)  # nothing was exported
    try:
        core.head_async('file:///x')
    except core.UnsupportedOperationError:
        pass
    else:
        raise AssertionError('head_async did not raise UnsupportedOperationError')
    return True
"
    );

    /// A fresh module registered exactly as the extension module registers `_core`, and the
    /// checks above, so a test calls the functions through Python argument parsing.
    fn surface<'py>(py: Python<'py>) -> (Bound<'py, PyModule>, Bound<'py, PyModule>) {
        let core = PyModule::new(py, "s3dlio_surface_test").expect("module");
        register_core_functions(&core).expect("register_core_functions");
        let checks = PyModule::from_code(
            py,
            SURFACE_PY,
            c_str!("s3dlio_surface_checks.py"),
            c_str!("_s3dlio_surface_checks"),
        )
        .expect("checks module");
        (core, checks)
    }

    #[test]
    fn python_surface_get_into_async_takes_expected_len_by_keyword_only() {
        init();
        Python::attach(|py| {
            let (core, checks) = surface(py);
            let message: String = checks
                .getattr("positional_expected_len_is_refused")
                .and_then(|f| f.call1((&core,)))
                .expect("a positional expected_len is a TypeError")
                .extract()
                .unwrap();
            assert!(message.contains("positional"), "{message}");
            let message: String = checks
                .getattr("expected_len_is_required")
                .and_then(|f| f.call1((&core,)))
                .expect("a missing expected_len is a TypeError")
                .extract()
                .unwrap();
            assert!(message.contains("expected_len"), "{message}");
        });
    }

    #[test]
    fn python_surface_exception_classes_derive_from_the_builtins() {
        init();
        Python::attach(|py| {
            let (core, checks) = surface(py);
            checks
                .getattr("exception_classes")
                .and_then(|f| f.call1((&core,)))
                .expect("exception hierarchy");
        });
    }

    #[test]
    fn python_surface_object_metadata_is_immutable_and_hashable() {
        init();
        Python::attach(|py| {
            let make = |size: u64| {
                Py::new(
                    py,
                    PyHeadMetadata {
                        size,
                        etag: Some("e".to_owned()),
                        version_id: None,
                    },
                )
                .unwrap()
            };
            // The class is registered under the name Python sees.
            let (core, checks) = surface(py);
            assert!(core.getattr("ObjectMetadata").is_ok());
            let repr: String = checks
                .getattr("metadata_is_immutable_and_hashable")
                .and_then(|f| f.call1((make(5), make(5), make(6))))
                .expect("ObjectMetadata behaves as a frozen value")
                .extract()
                .unwrap();
            assert_eq!(repr, "ObjectMetadata(size=5, etag='e', version_id=None)");
        });
    }

    #[test]
    fn python_surface_object_capabilities_returns_a_plain_dict() {
        init();
        Python::attach(|py| {
            let (core, checks) = surface(py);
            checks
                .getattr("capabilities_dict")
                .and_then(|f| f.call1((&core,)))
                .expect("object_capabilities('file:///x')");
        });
    }

    #[test]
    fn python_surface_unsupported_operation_is_raised_synchronously() {
        init();
        Python::attach(|py| {
            let (core, checks) = surface(py);
            checks
                .getattr("unsupported_is_raised_synchronously")
                .and_then(|f| f.call1((&core,)))
                .expect("a file:// URI raises UnsupportedOperationError at the call");
        });
    }

    // =======================================================================
    // The store is built once per URI, on the calling thread
    // =======================================================================

    /// The thread of every build of a store for exactly `uri`.
    #[cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]
    fn store_build_threads_for(uri: &str) -> Vec<std::thread::ThreadId> {
        STORE_BUILD_THREADS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|(built, _)| built.as_str() == uri)
            .map(|(_, thread)| *thread)
            .collect()
    }

    /// The real `get_into_async` / `head_async`, called twice each on one fresh URI with no
    /// running event loop (so the bridge fails right after the store was obtained): one store is
    /// built for the URI, and it is built on the calling thread, not on a runtime worker.
    #[cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]
    #[test]
    fn get_into_async_and_head_async_build_one_store_on_the_calling_thread() {
        init();
        // for_endpoint requires credentials to exist; none are ever used (store creation never
        // connects).
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        let uri = "s3://127.0.0.1:59264/bucket-b3b/store-cache";
        Python::attach(|py| {
            let driver = Driver::new(py);
            let ex = driver.exporter(16);
            let n = 8i32.into_pyobject(py).unwrap().into_any();
            assert_eq!(store_builds_for(uri), 0, "the URI must start cold");
            for call in 0..2 {
                let err = get_into_async(py, uri, &ex, &n).expect_err("no running event loop");
                assert!(err.to_string().contains("no running event loop"), "{err}");
                assert_eq!(store_builds_for(uri), 1, "get_into_async call {call}");
            }
            for call in 0..2 {
                let err = head_async(py, uri).expect_err("no running event loop");
                assert!(err.to_string().contains("no running event loop"), "{err}");
                assert_eq!(store_builds_for(uri), 1, "head_async call {call}");
            }
            assert_eq!(
                store_build_threads_for(uri),
                vec![std::thread::current().id()],
                "the store was not built on the calling thread"
            );
            // The failed starts released the buffer.
            assert_eq!(attr_usize(&ex, "exports"), attr_usize(&ex, "releases"));
            assert!(method_bool(&ex, "resizable"));
        });
    }

    // =======================================================================
    // Structure: the A5 order, the one latch, no wait on the pre-transfer path
    // =======================================================================

    /// Source of the function `name`: from its `fn` line to the closing brace at column 0.
    fn function_source(name: &str) -> String {
        let src = include_str!("python_core_api.rs");
        let start = ["<'py, S: StartSeams>(", "<'py>(", "("]
            .iter()
            .find_map(|generics| src.find(&format!("\nfn {name}{generics}")))
            .unwrap_or_else(|| panic!("function {name} not found"));
        let rest = &src[start + 1..];
        let end = rest.find("\n}\n").expect("end of function");
        rest[..end].to_string()
    }

    fn assert_in_order(body: &str, label: &str, needles: &[&str]) {
        let mut at = 0;
        for needle in needles {
            let found = body[at..]
                .find(needle)
                .unwrap_or_else(|| panic!("{label}: `{needle}` missing or out of order"));
            at += found + needle.len();
        }
    }

    #[test]
    fn get_into_async_follows_the_a5_step_order() {
        let entry = function_source("get_into_async");
        assert_in_order(
            &entry,
            "get_into_async",
            &[
                "require_capability(",
                "checked_expected_len(",
                ".detach(",
                "get_or_create_store(",
                "Latch::new()",
                "get_into_with_store(",
            ],
        );
        assert_eq!(
            entry.matches("Latch::new()").count(),
            1,
            "one latch per call"
        );
        let inner = function_source("get_into_with_store");
        assert_in_order(
            &inner,
            "get_into_with_store",
            &[
                "prepare_guard(py, latch.clone())",
                "acquire_writable(py, dst, expected_len, &latch)",
                "oneshot::channel::<WritableExport>()",
                "seams.bridge(",
                "seams.instantiate(",
                "tx.send(owner)",
            ],
        );
        assert!(
            !inner.contains("Latch::new()"),
            "the guard and the export share the caller's latch"
        );
        for body in [&entry, &inner] {
            assert!(
                !body.contains("wrap_in_guard("),
                "its error path waits on the latch"
            );
            assert!(
                !body.contains("quiesce_blocking("),
                "nothing may wait on the pre-transfer path"
            );
            assert!(!body.contains("quiesce_async("));
        }
        // The task binds the export before calling the store.
        assert_in_order(
            &inner,
            "task",
            &[
                "rx.await",
                "store.get_into(&uri, owner.as_mut_slice(), &progress)",
            ],
        );
        // Every pre-transfer failure releases directly.
        assert!(inner.matches("owner.release(py)").count() >= 3);
        // A guard failure releases, then cancels the bridged future, and only then closes the
        // start channel (else the task can finish with an exception nobody retrieves).
        assert_in_order(
            &inner,
            "instantiate failure",
            &[
                "seams.instantiate(",
                "owner.release(py);",
                "cancel_ignoring_errors(&inner);",
                "drop(tx);",
            ],
        );
    }

    #[test]
    fn head_async_follows_the_a5_step_order() {
        let entry = function_source("head_async");
        assert_in_order(
            &entry,
            "head_async",
            &[
                "require_capability(&uri, Operation::Head)",
                ".detach(",
                "get_or_create_store(",
                "Latch::new()",
                "head_with_store(",
            ],
        );
        assert_eq!(entry.matches("Latch::new()").count(), 1);
        let inner = function_source("head_with_store");
        assert_in_order(
            &inner,
            "head_with_store",
            &[
                "prepare_guard(py, latch.clone())",
                "oneshot::channel::<()>()",
                "HeadQuiescence { latch }",
                "seams.bridge(",
                "seams.instantiate(",
                "tx.send(())",
            ],
        );
        // The token is the first local of the future body.
        assert_in_order(
            &inner,
            "future body",
            &[
                "async move {",
                "let quiescence = token;",
                "rx.await",
                "store.head(",
                "drop(quiescence);",
            ],
        );
        // A guard failure cancels the bridged future BEFORE it closes the start channel.
        assert_in_order(
            &inner,
            "instantiate failure",
            &[
                "seams.instantiate(",
                "cancel_ignoring_errors(&inner);",
                "drop(tx);",
            ],
        );
        for body in [&entry, &inner] {
            assert!(!body.contains("wrap_in_guard("));
            assert!(!body.contains("quiesce_blocking("));
            assert!(
                !body.contains("acquire_writable("),
                "HEAD exports no buffer"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- #261 (PUT payload conversion is strict) and #233 (`mkdir_async` returns None).
//
// Embedded interpreter, local `file://` store only. A rejected payload must be refused
// BEFORE anything is submitted, so the target object must not exist afterwards.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod put_strict_payload_and_mkdir_tests {
    use super::*;
    use pyo3::ffi::c_str;
    use pyo3::types::{PyBytes, PyCFunction, PyDict, PyList, PyModule, PyTuple};

    const SCENARIO_PY: &std::ffi::CStr = c_str!(
        "
import array
import asyncio
import inspect

class B:
    def __bytes__(self):
        return b'xyz'

bad = {
    'int': 5,
    'bool': True,
    'false': False,
    'list': [1, 2, 3],
    'tuple': (1, 2, 3),
    'generator': (i for i in range(3)),
    'range': range(4),
}
good = {
    'bytes': (b'abc', b'abc'),
    'bytearray': (bytearray(b'abc'), b'abc'),
    'memoryview': (memoryview(b'abc'), b'abc'),
    'array_bytes': (array.array('B', b'abc'), b'abc'),
    'array_typed': (array.array('H', [1, 2, 515]), array.array('H', [1, 2, 515]).tobytes()),
    'strided_memoryview': (memoryview(b'abcdef')[::2], b'ace'),
    'dunder_bytes': (B(), b'xyz'),
}

async def _main(start):
    try:
        return await start()
    except BaseException as e:
        return e

def run(start):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_main(start))
    finally:
        loop.close()

def start_of(fn, args):
    # A start callable for `run`: calls fn(*args) and awaits the result if it is awaitable (the
    # sync batch API returns its result directly).
    async def go():
        result = fn(*args)
        if inspect.isawaitable(result):
            result = await result
        return result
    return go
"
    );

    fn init() {
        Python::initialize();
    }

    fn scenario(py: Python<'_>) -> Bound<'_, PyModule> {
        PyModule::from_code(
            py,
            SCENARIO_PY,
            c_str!("s3dlio_put_strict_scenario.py"),
            c_str!("_s3dlio_put_strict_scenario"),
        )
        .expect("scenario module")
    }

    /// The Python callable `start()` that calls `put_bytes_async(uri, data, borrow=borrow)`.
    fn put_async_start<'py>(
        py: Python<'py>,
        uri: String,
        data: Py<PyAny>,
        borrow: bool,
    ) -> Bound<'py, PyAny> {
        PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>,
                  _: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let py = args.py();
                put_bytes_async(py, &uri, data.bind(py), borrow).map(Bound::unbind)
            },
        )
        .expect("start closure")
        .into_any()
    }

    fn uri_in(dir: &std::path::Path, name: &str) -> (String, std::path::PathBuf) {
        let path = dir.join(name);
        (format!("file://{}", path.display()), path)
    }

    fn is_type_error(py: Python<'_>, e: &PyErr) -> bool {
        e.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
    }

    /// #261: `put_bytes` refuses an int, a bool and arbitrary iterables with a `TypeError`
    /// before it submits anything (`bytes(5)` used to write five zero bytes).
    #[test]
    fn put_bytes_refuses_ints_bools_and_iterables_before_submission() {
        init();
        Python::attach(|py| {
            let dir = tempfile::tempdir().expect("tempdir");
            let bad = scenario(py).getattr("bad").unwrap();
            let bad = bad.cast::<PyDict>().unwrap();
            for (name, obj) in bad.iter() {
                let (uri, path) = uri_in(dir.path(), &format!("sync-{name}"));
                let outcome = put_bytes(py, &uri, &obj);
                match outcome {
                    Err(e) => assert!(is_type_error(py, &e), "{name}: {e}"),
                    Ok(()) => panic!("{name}: put_bytes accepted the payload"),
                }
                assert!(!path.exists(), "{name}: an object was written");
            }
        });
    }

    /// #261: `put_bytes_async` refuses the same payloads synchronously, before the awaitable
    /// exists, with and without `borrow`.
    #[test]
    fn put_bytes_async_refuses_ints_bools_and_iterables_before_submission() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            let bad = module.getattr("bad").unwrap();
            let bad = bad.cast::<PyDict>().unwrap();
            for borrow in [false, true] {
                for (name, obj) in bad.iter() {
                    let (uri, path) = uri_in(dir.path(), &format!("async-{borrow}-{name}"));
                    let start = put_async_start(py, uri, obj.unbind(), borrow);
                    let outcome = module.getattr("run").unwrap().call1((start,)).unwrap();
                    let is_type_error = outcome
                        .is_instance(&py.get_type::<pyo3::exceptions::PyTypeError>())
                        .unwrap();
                    assert!(
                        is_type_error,
                        "{name} (borrow={borrow}): expected a TypeError, got {outcome:?}"
                    );
                    assert!(
                        !path.exists(),
                        "{name} (borrow={borrow}): an object was written"
                    );
                }
            }
        });
    }

    /// #261: bytes, buffers and objects with `__bytes__()` still work through both entry
    /// points, and the right bytes are written.
    #[test]
    fn put_bytes_still_accepts_bytes_buffers_and_dunder_bytes() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            let good = module.getattr("good").unwrap();
            let good = good.cast::<PyDict>().unwrap();
            for (name, pair) in good.iter() {
                let (obj, expected): (Bound<'_, PyAny>, Vec<u8>) = pair.extract().unwrap();

                let (uri, path) = uri_in(dir.path(), &format!("sync-{name}"));
                put_bytes(py, &uri, &obj).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(std::fs::read(&path).unwrap(), expected, "{name}: sync");

                for borrow in [false, true] {
                    let (uri, path) = uri_in(dir.path(), &format!("async-{borrow}-{name}"));
                    let start = put_async_start(py, uri, obj.clone().unbind(), borrow);
                    let outcome = module.getattr("run").unwrap().call1((start,)).unwrap();
                    assert!(
                        !outcome.is_instance_of::<pyo3::exceptions::PyBaseException>(),
                        "{name} (borrow={borrow}): {outcome:?}"
                    );
                    assert_eq!(
                        std::fs::read(&path).unwrap(),
                        expected,
                        "{name}: async, borrow={borrow}"
                    );
                }
            }
        });
    }

    /// #233: `mkdir_async` resolves to Python `None` (it used to resolve to `()`, an empty
    /// tuple), and the directory exists afterwards.
    #[test]
    fn mkdir_async_returns_python_none() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            let target = dir.path().join("made").join("deeper");
            let uri = format!("file://{}/", target.display());
            let start = PyCFunction::new_closure(
                py,
                None,
                None,
                move |args: &Bound<'_, PyTuple>,
                      _: Option<&Bound<'_, PyDict>>|
                      -> PyResult<Py<PyAny>> {
                    mkdir_async(args.py(), &uri).map(Bound::unbind)
                },
            )
            .expect("start closure")
            .into_any();
            let outcome = module.getattr("run").unwrap().call1((start,)).unwrap();
            assert!(outcome.is_none(), "mkdir_async resolved to {outcome:?}");
            assert!(target.is_dir(), "the directory was not made");
        });
    }

    /// #271: `put_bytes_async` resolves to Python `None` (it used to resolve to `()`), also with
    /// `borrow=True` (the `GuardedFuture` passes the result through), and the bytes are written.
    #[test]
    fn put_bytes_async_returns_python_none() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            for borrow in [false, true] {
                let (uri, path) = uri_in(dir.path(), &format!("none-{borrow}"));
                let data = pyo3::types::PyByteArray::new(py, b"payload").into_any();
                let start = put_async_start(py, uri, data.unbind(), borrow);
                let outcome = module.getattr("run").unwrap().call1((start,)).unwrap();
                assert!(
                    outcome.is_none(),
                    "put_bytes_async (borrow={borrow}) resolved to {outcome:?}"
                );
                assert_eq!(std::fs::read(&path).unwrap(), b"payload");
            }
        });
    }

    /// The module-level `#[pyfunction]` `name` as a Python callable (its real argument extraction).
    fn batch_function<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyAny> {
        let function = match name {
            "put_many" => wrap_pyfunction!(put_many, py),
            "put_many_async" => wrap_pyfunction!(put_many_async, py),
            other => panic!("no such function in this test: {other}"),
        };
        function.expect("wrap pyfunction").into_any()
    }

    /// Run `callable(*args)` (a Python call, awaited if it returns an awaitable) on a fresh event
    /// loop; returns its result, or the exception it raised.
    fn run_call<'py, A: IntoPyObject<'py>>(
        module: &Bound<'py, PyModule>,
        callable: &Bound<'py, PyAny>,
        args: A,
    ) -> Bound<'py, PyAny> {
        let start = module
            .getattr("start_of")
            .unwrap()
            .call1((callable, args))
            .unwrap();
        module.getattr("run").unwrap().call1((start,)).unwrap()
    }

    /// A Python list of `(uri, payload)` tuples.
    fn items_list<'py>(
        py: Python<'py>,
        pairs: Vec<(String, Bound<'py, PyAny>)>,
    ) -> Bound<'py, PyAny> {
        PyList::new(py, pairs).expect("items list").into_any()
    }

    /// #271: `put_many_async` resolves to Python `None`, and `put_many` returns `None`.
    #[test]
    fn put_many_and_put_many_async_return_python_none() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            for name in ["put_many", "put_many_async"] {
                let (uri_a, path_a) = uri_in(dir.path(), &format!("{name}-a"));
                let (uri_b, path_b) = uri_in(dir.path(), &format!("{name}-b"));
                let items = items_list(
                    py,
                    vec![
                        (uri_a, PyBytes::new(py, b"aa").into_any()),
                        (uri_b, PyBytes::new(py, b"bb").into_any()),
                    ],
                );
                let outcome = run_call(&module, &batch_function(py, name), (items,));
                assert!(outcome.is_none(), "{name} resolved to {outcome:?}");
                assert_eq!(std::fs::read(path_a).unwrap(), b"aa");
                assert_eq!(std::fs::read(path_b).unwrap(), b"bb");
            }
        });
    }

    /// #272: the batch PUT APIs refuse an int, a bool and arbitrary iterables as a payload with a
    /// `TypeError`, BEFORE any request is sent: an earlier, valid item of the same batch is not
    /// written either.
    #[test]
    fn put_many_refuses_ints_bools_and_iterables_before_submission() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            let bad = module.getattr("bad").unwrap();
            let bad = bad.cast::<PyDict>().unwrap();
            for name in ["put_many", "put_many_async"] {
                for (kind, obj) in bad.iter() {
                    let (good_uri, good_path) = uri_in(dir.path(), &format!("{name}-{kind}-good"));
                    let (bad_uri, bad_path) = uri_in(dir.path(), &format!("{name}-{kind}-bad"));
                    let items = items_list(
                        py,
                        vec![
                            (good_uri, PyBytes::new(py, b"ok").into_any()),
                            (bad_uri, obj),
                        ],
                    );
                    let outcome = run_call(&module, &batch_function(py, name), (items,));
                    let is_type_error = outcome
                        .is_instance(&py.get_type::<pyo3::exceptions::PyTypeError>())
                        .unwrap();
                    assert!(
                        is_type_error,
                        "{name} {kind}: expected a TypeError, got {outcome:?}"
                    );
                    assert!(
                        !bad_path.exists(),
                        "{name} {kind}: the bad item was written"
                    );
                    assert!(
                        !good_path.exists(),
                        "{name} {kind}: the valid item was written: a request was sent"
                    );
                }
            }
        });
    }

    /// #272: bytes, buffer-protocol objects and `__bytes__` objects are still accepted by the batch
    /// PUT APIs and the right bytes are written (a strided view is gathered in logical order).
    #[test]
    fn put_many_accepts_bytes_buffers_and_dunder_bytes() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            let good = module.getattr("good").unwrap();
            let good = good.cast::<PyDict>().unwrap();
            for name in ["put_many", "put_many_async"] {
                for (kind, pair) in good.iter() {
                    let (obj, expected): (Bound<'_, PyAny>, Vec<u8>) = pair.extract().unwrap();
                    let (uri, path) = uri_in(dir.path(), &format!("{name}-{kind}"));
                    let items = items_list(py, vec![(uri, obj)]);
                    let outcome = run_call(&module, &batch_function(py, name), (items,));
                    assert!(outcome.is_none(), "{name} {kind}: {outcome:?}");
                    assert_eq!(std::fs::read(&path).unwrap(), expected, "{name} {kind}");
                }
            }
        });
    }

    /// The message text of the exception object `outcome`.
    fn error_text(outcome: &Bound<'_, PyAny>) -> String {
        outcome.str().map(|s| s.to_string()).unwrap_or_default()
    }

    /// #282 follow-up: a batch conversion error names the failing item (index and URI), keeps its
    /// exception class and the original message after the prefix; a `MultiEndpointStore` write
    /// names the URI.
    #[test]
    fn payload_errors_name_the_failing_item() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let dir = tempfile::tempdir().expect("tempdir");
            let type_error = py.get_type::<pyo3::exceptions::PyTypeError>();
            for name in ["put_many", "put_many_async"] {
                let (ok_uri, _) = uri_in(dir.path(), "named-ok");
                let (bad_uri, _) = uri_in(dir.path(), "named-bad");
                let int_payload = 42i32.into_pyobject(py).unwrap().into_any();
                let items = items_list(
                    py,
                    vec![
                        (ok_uri, PyBytes::new(py, b"ok").into_any()),
                        (bad_uri.clone(), int_payload),
                    ],
                );
                let outcome = run_call(&module, &batch_function(py, name), (items,));
                assert!(
                    outcome.is_instance(&type_error).unwrap(),
                    "{name}: {outcome:?}"
                );
                let text = error_text(&outcome);
                assert!(text.contains("item 1"), "{name}: no item index: {text}");
                assert!(text.contains(&bad_uri), "{name}: no URI: {text}");
                assert!(
                    text.contains("Cannot convert data to bytes"),
                    "{name}: the original message is gone: {text}"
                );
            }

            let ep0 = tempfile::tempdir().expect("tempdir");
            let store = create_multi_endpoint_store(
                vec![format!("file://{}/", ep0.path().display())],
                None,
            )
            .expect("store");
            let store = Bound::new(py, store).expect("store object").into_any();
            let uri = format!("file://{}/named", ep0.path().display());
            let outcome = run_call(
                &module,
                &store.getattr("put").unwrap(),
                (uri.clone(), 42i32.into_pyobject(py).unwrap()),
            );
            assert!(outcome.is_instance(&type_error).unwrap(), "{outcome:?}");
            let text = error_text(&outcome);
            assert!(text.contains(&uri), "no URI: {text}");
            assert!(text.contains("Cannot convert data to bytes"), "{text}");
        });
    }

    /// #271: the `MultiEndpointStore` mutators resolve to Python `None` (they used to resolve to
    /// `()`), on two `file://` endpoints; each one really did its work.
    #[test]
    fn multi_endpoint_store_mutators_return_python_none() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let ep0 = tempfile::tempdir().expect("tempdir");
            let ep1 = tempfile::tempdir().expect("tempdir");
            let store = create_multi_endpoint_store(
                vec![
                    format!("file://{}/", ep0.path().display()),
                    format!("file://{}/", ep1.path().display()),
                ],
                None,
            )
            .expect("store");
            let store = Bound::new(py, store).expect("store object").into_any();
            let uri_on = |ep: &tempfile::TempDir, name: &str| {
                format!("file://{}/{name}", ep.path().display())
            };
            let method = |name: &str| store.getattr(name).expect("method");
            let payload = |bytes: &[u8]| PyBytes::new(py, bytes);

            let outcome = run_call(
                &module,
                &method("put"),
                (uri_on(&ep0, "balanced"), payload(b"x")),
            );
            assert!(outcome.is_none(), "put resolved to {outcome:?}");

            let outcome = run_call(
                &module,
                &method("put_to_endpoint"),
                (1usize, uri_on(&ep1, "direct"), payload(b"direct")),
            );
            assert!(outcome.is_none(), "put_to_endpoint resolved to {outcome:?}");
            assert_eq!(std::fs::read(ep1.path().join("direct")).unwrap(), b"direct");

            let outcome = run_call(
                &module,
                &method("put_all_endpoints"),
                (uri_on(&ep0, "everywhere"), payload(b"all")),
            );
            assert!(
                outcome.is_none(),
                "put_all_endpoints resolved to {outcome:?}"
            );
            for ep in [&ep0, &ep1] {
                assert_eq!(std::fs::read(ep.path().join("everywhere")).unwrap(), b"all");
            }

            let outcome = run_call(
                &module,
                &method("delete_from_endpoint"),
                (1usize, uri_on(&ep1, "direct")),
            );
            assert!(
                outcome.is_none(),
                "delete_from_endpoint resolved to {outcome:?}"
            );
            assert!(!ep1.path().join("direct").exists(), "not deleted");

            let outcome = run_call(&module, &method("delete"), (uri_on(&ep0, "everywhere"),));
            assert!(outcome.is_none(), "delete resolved to {outcome:?}");
        });
    }

    /// #281: `MultiEndpointStore.put`, `put_to_endpoint` and `put_all_endpoints` convert their
    /// payload like `put_bytes`: bytes, buffers and `__bytes__` objects are written; an int, a
    /// bool and arbitrary iterables are a `TypeError` and nothing is written on any endpoint.
    #[test]
    fn multi_endpoint_store_writes_use_the_strict_payload_conversion() {
        init();
        Python::attach(|py| {
            let module = scenario(py);
            let ep0 = tempfile::tempdir().expect("tempdir");
            let ep1 = tempfile::tempdir().expect("tempdir");
            let store = create_multi_endpoint_store(
                vec![
                    format!("file://{}/", ep0.path().display()),
                    format!("file://{}/", ep1.path().display()),
                ],
                None,
            )
            .expect("store");
            let store = Bound::new(py, store).expect("store object").into_any();
            let uri_on = |ep: &tempfile::TempDir, name: &str| {
                format!("file://{}/{name}", ep.path().display())
            };
            // Every write method as (name, args builder over (object name, payload)).
            let call = |method: &str, name: &str, payload: Py<PyAny>| {
                let function = store.getattr(method).expect("method");
                match method {
                    "put" => run_call(&module, &function, (uri_on(&ep0, name), payload)),
                    "put_to_endpoint" => {
                        run_call(&module, &function, (1usize, uri_on(&ep1, name), payload))
                    }
                    _ => run_call(&module, &function, (uri_on(&ep0, name), payload)),
                }
            };

            let bad = module.getattr("bad").unwrap();
            let bad = bad.cast::<PyDict>().unwrap();
            for method in ["put", "put_to_endpoint", "put_all_endpoints"] {
                for (kind, obj) in bad.iter() {
                    let name = format!("{method}-{kind}");
                    let outcome = call(method, &name, obj.unbind());
                    let is_type_error = outcome
                        .is_instance(&py.get_type::<pyo3::exceptions::PyTypeError>())
                        .unwrap();
                    assert!(
                        is_type_error,
                        "{method} {kind}: expected a TypeError, got {outcome:?}"
                    );
                    for ep in [&ep0, &ep1] {
                        assert!(
                            !ep.path().join(&name).exists(),
                            "{method} {kind}: an object was written"
                        );
                    }
                }
            }

            let good = module.getattr("good").unwrap();
            let good = good.cast::<PyDict>().unwrap();
            for method in ["put", "put_to_endpoint", "put_all_endpoints"] {
                for (kind, pair) in good.iter() {
                    let (obj, expected): (Bound<'_, PyAny>, Vec<u8>) = pair.extract().unwrap();
                    let name = format!("{method}-{kind}");
                    let outcome = call(method, &name, obj.unbind());
                    assert!(outcome.is_none(), "{method} {kind}: {outcome:?}");
                    // `put` is load balanced: it lands on exactly one of the endpoints.
                    let written = [&ep0, &ep1]
                        .into_iter()
                        .filter_map(|ep| std::fs::read(ep.path().join(&name)).ok())
                        .collect::<Vec<_>>();
                    assert!(!written.is_empty(), "{method} {kind}: nothing was written");
                    assert!(
                        written.iter().all(|bytes| *bytes == expected),
                        "{method} {kind}: wrong bytes {written:?}"
                    );
                    if method == "put_all_endpoints" {
                        assert_eq!(
                            std::fs::read(ep1.path().join(&name)).unwrap(),
                            expected,
                            "{method} {kind} (second endpoint)"
                        );
                    }
                }
            }
        });
    }

    /// The Python side of the C-F2 regression: one native call starts `mkdir_async` and polls the
    /// body WHILE HOLDING the GIL, and returns `(finished, future)`; Python only awaits the
    /// future.  No Python bytecode runs between the start and the poll, so CPython's switch
    /// interval cannot hand the GIL to the worker in between.
    const GIL_HELD_PY: &std::ffi::CStr = c_str!(
        "
import asyncio

async def _main(start):
    finished, fut = start()
    return (finished, await fut)

def run(start):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_main(start))
    finally:
        loop.close()
"
    );

    /// C-F2 (late completion / shutdown): the `mkdir_async` future body must finish its work
    /// WITHOUT the GIL. After the interpreter is unavailable (here: the GIL is held by this
    /// thread, as it is for a delivery that completes after the shutdown drain closed the gate),
    /// a body that ends in its own `Python::attach` (the old code) is stuck on a Tokio worker,
    /// outside the exit-safe delivery gate. A body that returns a Rust value finishes, and only
    /// the gated delivery converts it. The wait is bounded; the result is still `None` once the
    /// GIL is released.
    #[test]
    fn mkdir_async_body_finishes_without_the_gil_so_conversion_stays_in_the_gated_delivery() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        init();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                GIL_HELD_PY,
                c_str!("s3dlio_mkdir_gil_held.py"),
                c_str!("_s3dlio_mkdir_gil_held"),
            )
            .expect("gil-held module");
            let dir = tempfile::tempdir().expect("tempdir");
            let target = dir.path().join("late").join("completion");
            let uri = format!("file://{}/", target.display());
            let finished = Arc::new(AtomicBool::new(false));

            // Arm, start and poll inside ONE native call that keeps the GIL throughout: no Python
            // bytecode runs between `mkdir_async` and the poll, so CPython's switch interval
            // cannot give the worker the GIL in between (which would let the old body finish).
            let watched = Arc::clone(&finished);
            let start = PyCFunction::new_closure(
                py,
                None,
                None,
                move |args: &Bound<'_, PyTuple>,
                      _: Option<&Bound<'_, PyDict>>|
                      -> PyResult<Py<PyAny>> {
                    let py = args.py();
                    // The hook is per thread: it is armed on the thread that calls mkdir_async.
                    crate::python_api::async_bridge::body_finished_hook::arm(Arc::clone(&watched));
                    let future = mkdir_async(py, &uri)?;
                    // Polled with the GIL held: it never detaches, which is the point.
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while !watched.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    let done = watched.load(Ordering::SeqCst);
                    Ok(
                        PyTuple::new(py, [done.into_pyobject(py)?.to_owned().into_any(), future])?
                            .into_any()
                            .unbind(),
                    )
                },
            )
            .expect("start closure");

            let outcome = module.getattr("run").unwrap().call1((start,)).unwrap();
            let (body_done_without_gil, value): (bool, Bound<'_, PyAny>) =
                outcome.extract().expect("(bool, value)");
            assert!(
                body_done_without_gil,
                "the mkdir_async future body did not finish while the GIL was unavailable: it \
                 attaches to Python itself, outside the exit-safe delivery gate (C-F2, #192)"
            );
            assert!(value.is_none(), "mkdir_async resolved to {value:?}");
            assert!(target.is_dir(), "the directory was not made");
        });
    }
}

// ---------------------------------------------------------------------------
// Tests -- `error_with_context` (batch / multi-endpoint payload errors).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod error_context_tests {
    use super::*;
    use pyo3::exceptions::{PyMemoryError, PyTypeError};

    const CLASSES_PY: &std::ffi::CStr = c_str!(
        "
class TwoArg(Exception):
    def __init__(self, a, b):
        super().__init__(a, b)

class KwOnly(Exception):
    def __init__(self, *, code):
        super().__init__('kw')
        self.code = code

class WithAttr(Exception):
    pass

def with_attr():
    e = WithAttr('plain message')
    e.code = 7
    return e

def make(cls, *args):
    return cls(*args)

def make_kw():
    return KwOnly(code=3)

class CustomStr(ValueError):
    def __str__(self):
        return 'custom text'

class IgnoresArgs(Exception):
    def __init__(self, *a):
        super().__init__('fixed text')

class Counting(Exception):
    created = 0
    def __init__(self, *a):
        type(self).created += 1
        super().__init__(*a)

class NoteFails(ValueError):
    def add_note(self, note):
        raise RuntimeError('no notes here')

class TypeError(Exception):
    pass

def import_error():
    return ImportError('cannot import x', name='pkg.x', path='/p/x.py')

def attribute_error():
    class Owner:
        pass
    owner = Owner()
    return AttributeError('no attribute q', name='q', obj=owner), owner
"
    );

    fn init() {
        Python::initialize();
    }

    fn classes(py: Python<'_>) -> Bound<'_, pyo3::types::PyModule> {
        pyo3::types::PyModule::from_code(
            py,
            CLASSES_PY,
            c_str!("s3dlio_error_context_classes.py"),
            c_str!("_s3dlio_error_context_classes"),
        )
        .expect("classes module")
    }

    fn text(py: Python<'_>, e: &PyErr) -> String {
        e.value(py).str().expect("str(e)").to_string()
    }

    fn same_object(py: Python<'_>, a: &PyErr, b: &PyErr) -> bool {
        a.value(py).is(b.value(py))
    }

    /// A plain single-message exception is rebuilt as the SAME class with the context in front,
    /// and the original object is the `__cause__`.
    #[test]
    fn single_message_classes_are_preserved_with_the_original_as_cause() {
        init();
        Python::attach(|py| {
            for original in [
                PyErr::new::<PyTypeError, _>("Cannot convert data to bytes"),
                PyErr::new::<PyMemoryError, _>("cannot allocate 5 bytes"),
            ] {
                let class_name = original.get_type(py).name().unwrap().to_string();
                let message = text(py, &original);
                let keep = original.clone_ref(py);
                let wrapped = error_with_context(py, original, "item 1 (file:///x)");
                assert_eq!(
                    wrapped.get_type(py).name().unwrap().to_string(),
                    class_name,
                    "the class changed"
                );
                assert_eq!(text(py, &wrapped), format!("item 1 (file:///x): {message}"));
                let cause = wrapped.cause(py).expect("the original is the __cause__");
                assert!(
                    same_object(py, &cause, &keep),
                    "__cause__ is not the original"
                );
                assert!(!same_object(py, &wrapped, &keep));
            }
        });
    }

    /// Exceptions that a one-string constructor cannot rebuild come back UNCHANGED (the same
    /// object, same class, same arguments), never as a different class.
    #[test]
    fn exceptions_that_cannot_be_rebuilt_come_back_unchanged() {
        init();
        Python::attach(|py| {
            let module = classes(py);
            let make = module.getattr("make").unwrap();
            let builtins = py.import("builtins").unwrap();
            let cases: Vec<(&str, Bound<'_, PyAny>)> = vec![
                (
                    "TwoArg",
                    make.call1((module.getattr("TwoArg").unwrap(), 1, 2))
                        .unwrap(),
                ),
                (
                    "KwOnly",
                    module.getattr("make_kw").unwrap().call0().unwrap(),
                ),
                (
                    "UnicodeDecodeError",
                    make.call1((
                        builtins.getattr("UnicodeDecodeError").unwrap(),
                        "utf-8",
                        PyBytes::new(py, b"\xff"),
                        0,
                        1,
                        "invalid start byte",
                    ))
                    .unwrap(),
                ),
                (
                    "OSError",
                    make.call1((builtins.getattr("OSError").unwrap(), 2, "No such file"))
                        .unwrap(),
                ),
                (
                    "WithAttr",
                    module.getattr("with_attr").unwrap().call0().unwrap(),
                ),
                (
                    "KeyboardInterrupt",
                    make.call1((builtins.getattr("KeyboardInterrupt").unwrap(),))
                        .unwrap(),
                ),
            ];
            for (name, instance) in cases {
                // `OSError(2, ...)` is built as its errno subclass, so compare with the instance.
                let class_name = instance.get_type().name().unwrap().to_string();
                let original = PyErr::from_value(instance.clone());
                let wrapped = error_with_context(py, original, "item 1 (file:///x)");
                assert!(
                    wrapped.value(py).is(&instance),
                    "{name}: a different exception object was returned"
                );
                assert_eq!(
                    wrapped.get_type(py).name().unwrap().to_string(),
                    class_name,
                    "{name}: the class changed"
                );
                assert!(
                    !wrapped.is_instance_of::<PyTypeError>(py),
                    "{name}: turned into a TypeError"
                );
                assert!(
                    wrapped.cause(py).is_none(),
                    "{name}: an unchanged error must not get a __cause__"
                );
            }
        });
    }

    /// The unchanged original still tells which item failed: a note (Python >= 3.11) with the
    /// context, and no class name twice.
    #[test]
    fn an_unchanged_original_carries_the_context_as_a_note() {
        init();
        Python::attach(|py| {
            let module = classes(py);
            let instance = module
                .getattr("make")
                .unwrap()
                .call1((module.getattr("TwoArg").unwrap(), 1, 2))
                .unwrap();
            let before = instance.str().unwrap().to_string();
            let wrapped = error_with_context(
                py,
                PyErr::from_value(instance.clone()),
                "item 4 (file:///x)",
            );
            assert_eq!(text(py, &wrapped), before, "the message must not change");
            if instance.hasattr("add_note").unwrap() {
                let notes: Vec<String> = instance.getattr("__notes__").unwrap().extract().unwrap();
                assert_eq!(notes, vec!["item 4 (file:///x)".to_string()]);
                assert!(
                    !notes[0].contains("TwoArg"),
                    "class name repeated in the note"
                );
            }
        });
    }

    /// `e` came back unchanged: the same object, no `__cause__`, and (when Python has
    /// `add_note`) the context as its last note.
    fn assert_unchanged_with_note(py: Python<'_>, name: &str, instance: &Bound<'_, PyAny>) {
        let before = instance.str().unwrap().to_string();
        let args_before = instance.getattr("args").unwrap().to_string();
        let wrapped = error_with_context(
            py,
            PyErr::from_value(instance.clone()),
            "item 2 (file:///y)",
        );
        assert!(
            wrapped.value(py).is(instance),
            "{name}: a different exception object was returned"
        );
        assert!(wrapped.cause(py).is_none(), "{name}: got a __cause__");
        assert_eq!(text(py, &wrapped), before, "{name}: the message changed");
        assert_eq!(
            instance.getattr("args").unwrap().to_string(),
            args_before,
            "{name}: args changed"
        );
        if instance.hasattr("add_note").unwrap() {
            let notes: Vec<String> = instance.getattr("__notes__").unwrap().extract().unwrap();
            assert_eq!(notes.last().map(String::as_str), Some("item 2 (file:///y)"));
        }
    }

    /// A subclass, even of a builtin, is never rebuilt: its `__str__` and `__init__` are its own.
    #[test]
    fn subclasses_with_their_own_str_or_init_are_returned_unchanged() {
        init();
        Python::attach(|py| {
            let module = classes(py);
            let make = module.getattr("make").unwrap();
            let class = |name: &str| module.getattr(name).unwrap();
            for name in ["CustomStr", "IgnoresArgs", "TypeError"] {
                let instance = make.call1((class(name), "original")).unwrap();
                assert_unchanged_with_note(py, name, &instance);
            }
            // The `__str__` of a subclass is still its own after the call.
            let custom = make.call1((class("CustomStr"), "original")).unwrap();
            assert_unchanged_with_note(py, "CustomStr", &custom);
            assert_eq!(custom.str().unwrap().to_string(), "custom text");
            let ignores = make.call1((class("IgnoresArgs"), "original")).unwrap();
            assert_unchanged_with_note(py, "IgnoresArgs", &ignores);
            assert_eq!(ignores.str().unwrap().to_string(), "fixed text");
        });
    }

    /// The constructor of a class that is not rebuilt must not run a second time.
    #[test]
    fn a_side_effect_in_init_does_not_run_a_second_time() {
        init();
        Python::attach(|py| {
            let module = classes(py);
            let class = module.getattr("Counting").unwrap();
            let instance = module
                .getattr("make")
                .unwrap()
                .call1((class.clone(), "boom"))
                .unwrap();
            assert_eq!(
                class.getattr("created").unwrap().extract::<u32>().unwrap(),
                1
            );
            assert_unchanged_with_note(py, "Counting", &instance);
            assert_eq!(
                class.getattr("created").unwrap().extract::<u32>().unwrap(),
                1,
                "__init__ ran again"
            );
        });
    }

    /// Exceptions with fields keep them: the same object comes back, with a note.
    #[test]
    fn exceptions_with_fields_keep_their_fields() {
        init();
        Python::attach(|py| {
            let module = classes(py);
            let import_error = module.getattr("import_error").unwrap().call0().unwrap();
            assert_unchanged_with_note(py, "ImportError", &import_error);
            assert_eq!(import_error.getattr("name").unwrap().to_string(), "pkg.x");
            assert_eq!(import_error.getattr("path").unwrap().to_string(), "/p/x.py");

            let pair = module.getattr("attribute_error").unwrap().call0().unwrap();
            let attribute_error = pair.get_item(0).unwrap();
            let owner = pair.get_item(1).unwrap();
            assert_unchanged_with_note(py, "AttributeError", &attribute_error);
            assert_eq!(attribute_error.getattr("name").unwrap().to_string(), "q");
            assert!(attribute_error.getattr("obj").unwrap().is(&owner));

            let builtins = py.import("builtins").unwrap();
            let make = module.getattr("make").unwrap();
            let key_error = make
                .call1((builtins.getattr("KeyError").unwrap(), "k"))
                .unwrap();
            assert_unchanged_with_note(py, "KeyError", &key_error);
            let stop = make
                .call1((builtins.getattr("StopIteration").unwrap(), 3))
                .unwrap();
            assert_unchanged_with_note(py, "StopIteration", &stop);
            assert_eq!(stop.getattr("value").unwrap().to_string(), "3");
        });
    }

    /// Every plain builtin class on the rebuild list comes back as the same class, with the
    /// context in front and the original as `__cause__`.
    #[test]
    fn plain_builtins_are_rebuilt_with_the_prefix_and_a_cause() {
        init();
        Python::attach(|py| {
            let builtins = py.import("builtins").unwrap();
            for name in [
                "TypeError",
                "ValueError",
                "BufferError",
                "OverflowError",
                "MemoryError",
            ] {
                let class = builtins.getattr(name).unwrap();
                let instance = class.call1(("original text",)).unwrap();
                let wrapped = error_with_context(
                    py,
                    PyErr::from_value(instance.clone()),
                    "item 7 (s3://b/k)",
                );
                assert!(wrapped.get_type(py).is(&class), "{name}: class changed");
                assert_eq!(text(py, &wrapped), "item 7 (s3://b/k): original text");
                assert!(
                    wrapped.cause(py).unwrap().value(py).is(&instance),
                    "{name}: __cause__ is not the original"
                );
            }
        });
    }

    /// A failing `add_note` is ignored: the original still comes back.
    #[test]
    fn a_failing_add_note_is_ignored() {
        init();
        Python::attach(|py| {
            let module = classes(py);
            let instance = module
                .getattr("make")
                .unwrap()
                .call1((module.getattr("NoteFails").unwrap(), "original"))
                .unwrap();
            let wrapped = error_with_context(
                py,
                PyErr::from_value(instance.clone()),
                "item 1 (file:///x)",
            );
            assert!(wrapped.value(py).is(&instance));
            assert!(
                PyErr::take(py).is_none(),
                "the failed add_note left an exception set"
            );
        });
    }

    /// End to end: a `?` in an `s3://` key stays in the message; an https SAS query does not.
    #[test]
    fn batch_error_keeps_a_question_mark_key_and_drops_a_sas_query() {
        init();
        Python::attach(|py| {
            for (uri, shown, hidden) in [
                ("s3://b/what?.csv", "s3://b/what?.csv", ""),
                (
                    "https://acct.blob.core.windows.net/c/k?sv=1&sig=TOPSECRET",
                    "https://acct.blob.core.windows.net/c/k",
                    "TOPSECRET",
                ),
            ] {
                let bad = 42i32.into_pyobject(py).unwrap().into_any();
                let err = batch_payloads(py, vec![(uri.to_string(), bad)], PayloadUse::Blocking)
                    .err()
                    .expect("an int payload must be refused");
                let message = text(py, &err);
                assert!(message.contains(shown), "{message}");
                assert!(hidden.is_empty() || !message.contains(hidden), "{message}");
            }
        });
    }

    /// End to end: a bad item in a batch names the item and its (redacted) URI, keeps the class
    /// and the `__cause__`, and the secret never appears.
    #[test]
    fn batch_error_redacts_the_uri_and_keeps_the_cause() {
        init();
        Python::attach(|py| {
            let secret_uri = "https://acct.blob.core.windows.net/c/o?sv=1&sig=TOPSECRET";
            let good = PyBytes::new(py, b"ok").into_any();
            let bad = 42i32.into_pyobject(py).unwrap().into_any();
            let err = batch_payloads(
                py,
                vec![
                    ("file:///ok".to_string(), good),
                    (secret_uri.to_string(), bad),
                ],
                PayloadUse::Blocking,
            )
            .err()
            .expect("an int payload must be refused");
            let message = text(py, &err);
            assert!(err.is_instance_of::<PyTypeError>(py), "{message}");
            assert!(
                !message.contains("TOPSECRET") && !message.contains("sig="),
                "{message}"
            );
            assert!(message.contains("item 1"), "{message}");
            assert!(
                message.contains("https://acct.blob.core.windows.net/c/o"),
                "{message}"
            );
            assert!(
                message.contains("Cannot convert data to bytes"),
                "{message}"
            );
            assert!(
                err.cause(py).is_some(),
                "the original TypeError is the __cause__"
            );
        });
    }
}

/// Issue #285: every `Option<...>` parameter the docstrings call optional must be callable
/// when omitted. Each function / method is checked on its own, once without the parameter and
/// once with it passed by keyword; all failures are collected and reported together.
#[cfg(test)]
mod omitted_optional_parameter_tests {
    use super::*;
    use pyo3::types::PyModule;

    fn init() {
        Python::initialize();
    }

    /// `None` when the call got past argument parsing (any result, any other error), else why not.
    fn argument_problem(result: PyResult<Bound<'_, PyAny>>, py: Python<'_>) -> Option<String> {
        let err = result.err()?;
        if !err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
            return None;
        }
        let message = err.value(py).str().ok()?.to_string();
        let bad = message.contains("required positional argument")
            || message.contains("unexpected keyword argument");
        bad.then_some(message)
    }

    fn call<'py>(
        target: &Bound<'py, PyAny>,
        args: Vec<Bound<'py, PyAny>>,
        kwargs: Vec<(&str, Bound<'py, PyAny>)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = target.py();
        let args = pyo3::types::PyTuple::new(py, args)?;
        let dict = pyo3::types::PyDict::new(py);
        for (name, value) in kwargs {
            dict.set_item(name, value)?;
        }
        target.call(args, Some(&dict))
    }

    #[test]
    fn omitted_optional_parameters_are_defaults() {
        init();
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().to_str().expect("utf-8 temp path").to_string();
        Python::attach(|py| {
            let m = PyModule::new(py, "s3dlio_285_probe").expect("module");
            register_core_functions(&m).expect("register");
            let get = |name: &str| m.getattr(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            let s = |text: &str| text.into_pyobject(py).expect("str").into_any();
            let options = Bound::new(py, PyWriterOptions::new())
                .expect("options")
                .into_any();
            let uris_file = format!("{base}/endpoints.txt");
            std::fs::write(&uris_file, format!("file://{base}/a/\nfile://{base}/b/\n"))
                .expect("write endpoints file");
            let two = || {
                pyo3::types::PyList::new(
                    py,
                    [format!("file://{base}/a/"), format!("file://{base}/b/")],
                )
                .expect("list")
                .into_any()
            };

            let failures: std::cell::RefCell<Vec<String>> = std::cell::RefCell::default();
            let check = |label: &str, result: PyResult<Bound<'_, PyAny>>| {
                if let Some(why) = argument_problem(result, py) {
                    failures.borrow_mut().push(format!("{label}: {why}"));
                }
            };

            // Writer factories: `options` omitted, then by keyword. The S3 and Azure ones get a
            // URI of the wrong scheme so that nothing leaves the process.
            for (name, uri) in [
                ("create_s3_writer", "file:///not-an-s3-uri".to_string()),
                (
                    "create_azure_writer",
                    "file:///not-an-azure-uri".to_string(),
                ),
                ("create_filesystem_writer", format!("file://{base}/w1.bin")),
                (
                    "create_direct_filesystem_writer",
                    format!("file://{base}/w2.bin"),
                ),
            ] {
                let f = get(name);
                check(&format!("{name}(uri)"), call(&f, vec![s(&uri)], vec![]));
                check(
                    &format!("{name}(uri, options=...)"),
                    call(&f, vec![s(&uri)], vec![("options", options.clone())]),
                );
                check(
                    &format!("{name}(uri, options=None)"),
                    call(
                        &f,
                        vec![s(&uri)],
                        vec![("options", py.None().into_bound(py))],
                    ),
                );
            }

            // Multi-endpoint factories: `strategy` omitted, then by keyword.
            let f = get("create_multi_endpoint_store");
            check(
                "create_multi_endpoint_store(uris)",
                call(&f, vec![two()], vec![]),
            );
            check(
                "create_multi_endpoint_store(uris, strategy=...)",
                call(&f, vec![two()], vec![("strategy", s("least_connections"))]),
            );
            let f = get("create_multi_endpoint_store_from_template");
            let template = format!("file://{base}/t{{1...2}}/");
            check(
                "create_multi_endpoint_store_from_template(template)",
                call(&f, vec![s(&template)], vec![]),
            );
            check(
                "create_multi_endpoint_store_from_template(template, strategy=...)",
                call(&f, vec![s(&template)], vec![("strategy", s("round_robin"))]),
            );
            let f = get("create_multi_endpoint_store_from_file");
            check(
                "create_multi_endpoint_store_from_file(path)",
                call(&f, vec![s(&uris_file)], vec![]),
            );
            check(
                "create_multi_endpoint_store_from_file(path, strategy=...)",
                call(
                    &f,
                    vec![s(&uris_file)],
                    vec![("strategy", s("round_robin"))],
                ),
            );
            // S3_ENDPOINT_URIS is not touched here (other tests run in this process): unset or
            // set, the call must get past argument parsing.
            let f = get("create_multi_endpoint_store_from_env");
            check(
                "create_multi_endpoint_store_from_env()",
                call(&f, vec![], vec![]),
            );
            check(
                "create_multi_endpoint_store_from_env(strategy=...)",
                call(&f, vec![], vec![("strategy", s("round_robin"))]),
            );

            // `PyWriterOptions.with_compression(level)`.
            let w = options
                .getattr("with_compression")
                .expect("with_compression");
            check(
                "with_compression('none')",
                call(&w, vec![s("none")], vec![]),
            );
            check(
                "with_compression('zstd')",
                call(&w, vec![s("zstd")], vec![]),
            );
            check(
                "with_compression('zstd', level=5)",
                call(
                    &w,
                    vec![s("zstd")],
                    vec![("level", 5i32.into_pyobject(py).unwrap().into_any())],
                ),
            );

            // `MultiEndpointStore.get_range(length)`.
            match call(
                &get("create_multi_endpoint_store"),
                vec![two()],
                vec![("strategy", s("round_robin"))],
            ) {
                Ok(store) => {
                    let g = store.getattr("get_range").expect("get_range");
                    let uri = s(&format!("file://{base}/a/missing.bin"));
                    let zero = 0u64.into_pyobject(py).unwrap().into_any();
                    check(
                        "get_range(uri, offset)",
                        call(&g, vec![uri.clone(), zero.clone()], vec![]),
                    );
                    check(
                        "get_range(uri, offset, length=4)",
                        call(
                            &g,
                            vec![uri, zero],
                            vec![("length", 4u64.into_pyobject(py).unwrap().into_any())],
                        ),
                    );
                }
                Err(e) => failures
                    .borrow_mut()
                    .push(format!("no store for get_range: {e}")),
            }

            // `gcs_set_rapid_mode(force)`; None means auto-detect, i.e. no change.
            let f = get("gcs_set_rapid_mode");
            check("gcs_set_rapid_mode()", call(&f, vec![], vec![]));
            check(
                "gcs_set_rapid_mode(force=None)",
                call(&f, vec![], vec![("force", py.None().into_bound(py))]),
            );

            let failures = failures.into_inner();
            assert!(
                failures.is_empty(),
                "{} call(s) failed argument parsing:\n{}",
                failures.len(),
                failures.join("\n")
            );
        });
    }
}

/// Python sees the parameter NAMES of a `#[pyfunction]`, so the no-backend stubs must expose the
/// same names as the real functions of a build with the backend (a keyword call such as
/// `gcs_set_channel_count(n=4)` must not depend on how the wheel was built). The expected text
/// signatures below are the contract of every build.
#[cfg(test)]
mod stub_signature_tests {
    use super::*;
    use pyo3::types::PyModule;

    fn init() {
        Python::initialize();
    }

    fn text_signature(function: &Bound<'_, PyAny>) -> String {
        function
            .getattr("__text_signature__")
            .and_then(|s| s.extract::<String>())
            .unwrap_or_else(|e| panic!("no __text_signature__: {e}"))
    }

    #[test]
    fn stubs_and_real_functions_expose_the_same_parameter_names() {
        init();
        Python::attach(|py| {
            let m = PyModule::new(py, "s3dlio_stub_signature_probe").expect("module");
            register_core_functions(&m).expect("register");
            let mut wrong = Vec::new();
            for (name, want) in [
                ("gcs_set_channel_count", "(n)"),
                ("gcs_set_rapid_mode", "(force=None)"),
                ("gcs_query_rapid_bucket", "(bucket_or_uri)"),
                ("create_azure_writer", "(uri, options=None)"),
            ] {
                let got =
                    text_signature(&m.getattr(name).unwrap_or_else(|e| panic!("{name}: {e}")));
                if got != want {
                    wrong.push(format!("{name}: {got} (want {want})"));
                }
            }
            let dataset = py
                .get_type::<crate::python_api::python_aiml_api::PyDataset>()
                .getattr("from_uris")
                .expect("PyDataset.from_uris");
            let got = text_signature(&dataset);
            if got != "(uris, opts=None)" {
                wrong.push(format!(
                    "PyDataset.from_uris: {got} (want (uris, opts=None))"
                ));
            }
            assert!(
                wrong.is_empty(),
                "signature mismatch:\n{}",
                wrong.join("\n")
            );
        });
    }

    /// In a build without the GCS / Azure backend the stubs take the real keywords too.
    #[cfg(not(any(feature = "backend-gcs", feature = "backend-azure")))]
    #[test]
    fn stubs_accept_the_real_keywords() {
        init();
        Python::attach(|py| {
            let m = PyModule::new(py, "s3dlio_stub_keyword_probe").expect("module");
            register_core_functions(&m).expect("register");
            let kw = |name: &str, value: Bound<'_, PyAny>| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item(name, value).expect("kwarg");
                d
            };
            let n = 4usize.into_pyobject(py).expect("int").into_any();
            m.getattr("gcs_set_channel_count")
                .and_then(|f| f.call((), Some(&kw("n", n))))
                .expect("gcs_set_channel_count(n=4)");
            let b = "my-bucket".into_pyobject(py).expect("str").into_any();
            let got = m
                .getattr("gcs_query_rapid_bucket")
                .and_then(|f| f.call((), Some(&kw("bucket_or_uri", b))))
                .expect("gcs_query_rapid_bucket(bucket_or_uri=...)");
            assert!(!got.is_truthy().expect("bool"));
        });
    }

    /// `with_compression("zstd")` without a level uses zstd level 3, an explicit level is kept.
    #[test]
    fn with_compression_default_zstd_level_is_three() {
        let mut options = PyWriterOptions::new();
        options.with_compression("zstd", None).expect("zstd");
        assert_eq!(
            options.inner.compression,
            Some(CompressionConfig::Zstd { level: 3 })
        );
        options.with_compression("zstd", Some(7)).expect("zstd 7");
        assert_eq!(
            options.inner.compression,
            Some(CompressionConfig::Zstd { level: 7 })
        );
        options.with_compression("none", None).expect("none");
        assert_eq!(options.inner.compression, Some(CompressionConfig::None));
    }
}

// ---------------------------------------------------------------------------
// Tests -- run_bounded_puts: ordered admission; a failed or cancelled batch stops the rest.
//
// The production context matters: `put_many_async` runs the batch on a runtime WORKER thread, and
// spawning many tasks from a worker without yielding is what scrambles tokio's scheduling order
// (LIFO slot, stealing). Every test therefore runs the batch inside a spawned task on a
// multi-thread runtime, and repeats it.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod bounded_puts_tests {
    use super::run_bounded_puts;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    type Put = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>;

    fn joined(e: tokio::task::JoinError) -> String {
        format!("join: {e}")
    }

    /// Run the batch the way production does: inside a spawned task (a runtime worker).
    async fn run_in_worker(puts: Vec<Put>, max_in_flight: usize) -> Result<(), String> {
        tokio::spawn(run_bounded_puts(puts, max_in_flight, joined))
            .await
            .unwrap_or_else(|e| Err(format!("batch task: {e}")))
    }

    /// A put that records its index when it starts, "holds a payload" (an `Arc` clone) and takes
    /// `ms` to finish.
    fn put(order: &Arc<Mutex<Vec<usize>>>, payload: &Arc<()>, index: usize, ms: u64) -> Put {
        let order = Arc::clone(order);
        let payload = Arc::clone(payload);
        Box::pin(async move {
            order
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(index);
            tokio::time::sleep(Duration::from_millis(ms)).await;
            drop(payload);
            Ok(())
        })
    }

    fn started(order: &Arc<Mutex<Vec<usize>>>) -> Vec<usize> {
        order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_puts_run_and_succeed() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let payload = Arc::new(());
        let puts: Vec<Put> = (0..12).map(|i| put(&order, &payload, i, 5)).collect();
        assert_eq!(run_in_worker(puts, 3).await, Ok(()));
        assert_eq!(started(&order).len(), 12);
        assert_eq!(Arc::strong_count(&payload), 1);
    }

    /// Admission is in input order: with one permit item i starts before item i + 1, always,
    /// even when the batch is spawned from a worker thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn puts_start_in_input_order() {
        for round in 0..20 {
            let order = Arc::new(Mutex::new(Vec::new()));
            let payload = Arc::new(());
            let puts: Vec<Put> = (0..150).map(|i| put(&order, &payload, i, 0)).collect();
            assert_eq!(run_in_worker(puts, 1).await, Ok(()));
            assert_eq!(
                started(&order),
                (0..150).collect::<Vec<_>>(),
                "round {round}: items started out of input order"
            );
        }
    }

    /// The ordered registration of the permit requests must not depend on tokio's cooperative
    /// task budget (128 operations per poll of the batch task): with a limit above it, the
    /// requests made after the budget is spent must still be queued in list order. With the
    /// permits all held, each release must go to the NEXT item of the list.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn admission_order_holds_for_a_limit_above_the_task_budget() {
        const LIMIT: usize = 200;
        const ITEMS: usize = 400;
        const STEPS: usize = 60;

        async fn until_started(order: &Arc<Mutex<Vec<usize>>>, count: usize) {
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while started(order).len() < count {
                assert!(
                    std::time::Instant::now() < deadline,
                    "only {} of {count} items started",
                    started(order).len()
                );
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }

        let order = Arc::new(Mutex::new(Vec::new()));
        let gates: Vec<Arc<tokio::sync::Notify>> = (0..LIMIT)
            .map(|_| Arc::new(tokio::sync::Notify::new()))
            .collect();
        let puts: Vec<Put> = (0..ITEMS)
            .map(|i| {
                let order = Arc::clone(&order);
                let gate = gates.get(i).cloned();
                Box::pin(async move {
                    order
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(i);
                    if let Some(gate) = gate {
                        gate.notified().await;
                    }
                    Ok(())
                }) as Put
            })
            .collect();
        let batch = tokio::spawn(run_bounded_puts(puts, LIMIT, joined));

        // The first LIMIT items fill every permit and hold them.
        until_started(&order, LIMIT).await;
        let mut first_wave = started(&order);
        first_wave.sort_unstable();
        assert_eq!(first_wave, (0..LIMIT).collect::<Vec<_>>());

        // Each release goes to the next item in the list.
        for step in 0..STEPS {
            gates[step].notify_one();
            until_started(&order, LIMIT + step + 1).await;
            assert_eq!(
                started(&order)[LIMIT + step],
                LIMIT + step,
                "release {step} did not admit the next item of the list"
            );
        }
        for gate in &gates[STEPS..] {
            gate.notify_one();
        }
        assert_eq!(
            batch
                .await
                .unwrap_or_else(|e| Err(format!("batch task: {e}"))),
            Ok(())
        );
        assert_eq!(started(&order).len(), ITEMS);
    }

    /// The first failure is returned, no later item starts, and every payload is dropped before
    /// the call returns (queued items are never started, running ones are aborted).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn first_failure_stops_the_rest_and_drops_their_payloads() {
        for round in 0..20 {
            let order = Arc::new(Mutex::new(Vec::new()));
            let payload = Arc::new(());
            let mut puts: Vec<Put> = Vec::new();
            let failing = Arc::clone(&payload);
            puts.push(Box::pin(async move {
                drop(failing);
                Err("the item's own error".to_string())
            }));
            for i in 1..200 {
                puts.push(put(&order, &payload, i, 200));
            }
            let t0 = std::time::Instant::now();
            let result = run_in_worker(puts, 1).await;
            assert_eq!(
                result,
                Err("the item's own error".to_string()),
                "round {round}"
            );
            assert!(
                started(&order).is_empty(),
                "round {round}: items {:?} started after the failure",
                started(&order)
            );
            assert!(
                t0.elapsed() < Duration::from_millis(1500),
                "round {round}: the batch drained the rest instead of stopping"
            );
            assert_eq!(
                Arc::strong_count(&payload),
                1,
                "round {round}: a payload is still held after the batch returned"
            );
        }
    }

    /// An item whose put PANICS fails the batch like an error does: the panic is returned as that
    /// item's failure (through `join_failed`) and no item that had not started yet starts after
    /// it. The unwinding task gives its permit back without any code of its own running, so the
    /// semaphore must be closed by a drop guard BEFORE the permit is released: otherwise the next
    /// queued item can be granted that permit and start before the batch reaps the `JoinError`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_panicking_put_stops_the_rest_before_any_later_item_starts() {
        for round in 0..200 {
            let order = Arc::new(Mutex::new(Vec::new()));
            let payload = Arc::new(());
            let mut puts: Vec<Put> = Vec::new();
            let panicking = Arc::clone(&payload);
            puts.push(Box::pin(async move {
                // Let the batch register every later item's permit request first.
                tokio::time::sleep(Duration::from_millis(2)).await;
                drop(panicking);
                let () = Err::<(), String>("boom".to_string()).expect("item 0 panics on purpose");
                Ok(())
            }));
            for i in 1..8 {
                puts.push(put(&order, &payload, i, 10_000));
            }
            let result = run_in_worker(puts, 1).await;
            match &result {
                Err(e) if e.starts_with("join: ") && e.contains("panic") => {}
                other => panic!("round {round}: the panic was not item 0's failure: {other:?}"),
            }
            assert!(
                started(&order).is_empty(),
                "round {round}: items {:?} started after the panic",
                started(&order)
            );
            assert_eq!(Arc::strong_count(&payload), 1, "round {round}");
        }
    }

    /// With several permits, items already running when a sibling fails are aborted; items not
    /// yet admitted never start.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn failure_with_several_permits_starts_nothing_new() {
        for round in 0..20 {
            let order = Arc::new(Mutex::new(Vec::new()));
            let payload = Arc::new(());
            let mut puts: Vec<Put> = Vec::new();
            for i in 0..3 {
                puts.push(put(&order, &payload, i, 10_000));
            }
            puts.push(Box::pin(async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Err("late failure".to_string())
            }));
            for i in 4..100 {
                puts.push(put(&order, &payload, i, 10_000));
            }
            let result = run_in_worker(puts, 4).await;
            assert_eq!(result, Err("late failure".to_string()), "round {round}");
            let mut begun = started(&order);
            begun.sort_unstable();
            assert_eq!(
                begun,
                vec![0, 1, 2],
                "round {round}: an item started after the failure"
            );
            assert_eq!(Arc::strong_count(&payload), 1, "round {round}");
        }
    }

    /// Dropping the batch (the cancellation of `put_many_async`) starts no further item and
    /// drops every payload.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropping_the_batch_stops_the_rest() {
        for round in 0..10 {
            let order = Arc::new(Mutex::new(Vec::new()));
            let payload = Arc::new(());
            let puts: Vec<Put> = (0..30).map(|i| put(&order, &payload, i, 10_000)).collect();
            let batch = tokio::spawn(run_bounded_puts(puts, 1, joined));
            while started(&order).is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            batch.abort();
            let _ = batch.await;
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert_eq!(
                started(&order),
                vec![0],
                "round {round}: an item started after the batch was dropped"
            );
            assert_eq!(
                Arc::strong_count(&payload),
                1,
                "round {round}: a payload outlived the batch"
            );
        }
    }

    /// A huge `max_in_flight` is clamped to the semaphore's maximum instead of panicking.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_huge_limit_is_clamped_not_a_panic() {
        for limit in [usize::MAX, 1usize << 62, (1usize << 61) + 1] {
            let counter = Arc::new(AtomicUsize::new(0));
            let puts: Vec<Put> = (0..5)
                .map(|_| {
                    let counter = Arc::clone(&counter);
                    Box::pin(async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }) as Put
                })
                .collect();
            assert_eq!(run_in_worker(puts, limit).await, Ok(()), "limit {limit}");
            assert_eq!(counter.load(Ordering::SeqCst), 5);
        }
    }
}
