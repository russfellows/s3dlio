// src/python_api/python_core_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use super::async_bridge::future_into_py;
use bytes::Bytes;
use pyo3::buffer::PyBuffer;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::Bound;

use tokio::task;

use std::path::PathBuf;

use tracing::{debug, warn, Level};

// Project crates
use crate::config::{Config, DataGenAlgorithm, DataGenMode, ObjectType};
use crate::delete_batch_plan::parse_s3_uri_for_global_client;
use crate::list_containers::list_containers as list_containers_rs;
use crate::object_store::{
    store_for_uri, store_for_uri_with_logger, store_routing_endpoint_for_uri,
};
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
    CompressionConfig, ObjectStore, ObjectWriter, S3ObjectStore, WriterOptions,
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
        STORE_BUILD_LOG
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(uri.to_owned());
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
/// A process supports exactly ONE op-log session (init -> finalize, no restart).
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
///     - Call while logging is already active: returns without effect; the new
///       path is not created or truncated.
///     - Call after finalize_op_log(): raises RuntimeError and touches no file.
///
/// The module-level functions (get, put_bytes, stat, list, delete, get_async,
/// ...) look the op-log up on every call: a call made while logging is active is
/// recorded once, and a call made before init_op_log() or after
/// finalize_op_log() is not, whichever call first used a given scheme. Objects
/// you construct yourself (stores, readers, writers) capture the logger when
/// they are constructed: those created before init_op_log() are never logged;
/// those created before finalize_op_log() stop recording when it runs.
#[pyfunction]
pub fn init_op_log(path: &str) -> PyResult<()> {
    init_op_logger(path).map_err(py_err)
}

/// Finish operation logging: write every accepted record (unless the writer
/// fails: an I/O error or a panic in the writer thread, see op_log_stats()),
/// close the zstd stream and wait until the writer has finished and signalled
/// completion. Call it before the process exits: nothing does it for you (no
/// atexit hook), and without it queued records and the zstd frame end are lost. Terminal for this process (see
/// init_op_log). Safe to call repeatedly, or before init_op_log() (no effect).
/// After it returns is_op_log_active() is False and operations on objects
/// created earlier are no longer recorded. The GIL is released while the
/// pending records are drained, so other Python threads keep running.
#[pyfunction]
pub fn finalize_op_log(py: Python<'_>) -> PyResult<()> {
    py.detach(finalize_op_logger);
    Ok(())
}

/// True only between a successful init_op_log() and finalize_op_log().
#[pyfunction]
pub fn is_op_log_active() -> PyResult<bool> {
    Ok(global_logger().is_some())
}

/// Counts of what happened to the entries submitted to the op-log between
/// init_op_log() and finalize_op_log(), so you can detect most ways records are
/// lost.
///
/// Returns None before init_op_log(); otherwise a dict (available while
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
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let template = template.unwrap_or("object-{}");
    let uris = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
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
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let template = template.unwrap_or("object-{}");
    let uris = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
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
#[pyfunction]
#[pyo3(signature = (uri, recursive = false, pattern = None))]
pub fn list(
    py: Python<'_>,
    uri: &str,
    recursive: bool,
    pattern: Option<&str>,
) -> PyResult<Vec<String>> {
    // Universal list using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let uri_owned = uri.to_owned();

    // The GIL is released for the whole blocking part -- store creation (credential
    // loading) and the network listing -- so other Python threads keep running (issue #201).
    let mut keys = py.detach(|| {
        let store = get_or_create_store(&uri_owned, global_logger()).map_err(py_err)?;
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move { store.list(&uri_owned, recursive).await })
    })?;

    // Apply client-side regex filtering against the relative path (after the listing prefix).
    // Pattern sees only the key portion after the URI prefix, e.g. "llama3-8b/" not the full URI.
    if let Some(pat) = pattern {
        use regex::Regex;
        let re = Regex::new(pat).map_err(py_err)?;
        let base = if uri.ends_with('/') {
            uri.to_owned()
        } else {
            format!("{}/", uri)
        };
        keys.retain(|k| {
            let relative = k.strip_prefix(base.as_str()).unwrap_or(k.as_str());
            re.is_match(relative)
        });
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
// put_bytes() - Zero-copy put from Python bytes to any backend
// ---------------------------------------------------------------------------

/// Put bytes data to any storage backend with zero-copy from Python.
///
/// This function accepts Python bytes/bytearray and writes them to the specified
/// URI without copying the data on the Rust side. The data is borrowed directly
/// from Python's memory buffer.
///
/// # Arguments
/// * `uri` - Full URI including scheme (s3://, az://, gs://, file://, direct://)
/// * `data` - Python bytes, bytearray, or any object supporting the buffer protocol
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
/// s3dlio.put_bytes("az://container/blob.bin", b"hello world")
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

    // Use same conversion logic as put_bytes_async for consistency
    let data_owned = if let Ok(buffer) = PyBuffer::<u8>::get(data) {
        // Buffer protocol (dgen_py.BytesView, numpy, bytearray, memoryview, etc.)
        let len = buffer.len_bytes();
        let mut vec = vec![0u8; len];
        buffer.copy_to_slice(py, &mut vec[..]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyBufferError, _>(format!("Buffer copy failed: {}", e))
        })?;

        Bytes::from(vec)
    } else if let Ok(bytes_view) = data.extract::<PyRef<PyBytesView>>() {
        // s3dlio's own BytesView - zero-copy Arc clone (optimal)
        bytes_view.bytes.clone()
    } else if let Ok(py_bytes) = data.cast::<PyBytes>() {
        // PyBytes - direct copy
        Bytes::copy_from_slice(py_bytes.as_bytes())
    } else {
        // Last resort: Try __bytes__() method or bytes() conversion
        let bytes_result = data.call_method0("__bytes__").or_else(|_| {
            let bytes_type = py.get_type::<PyBytes>();
            bytes_type.call1((data,))
        });

        match bytes_result {
            Ok(bytes_obj) => {
                if let Ok(py_bytes) = bytes_obj.cast::<PyBytes>() {
                    Bytes::copy_from_slice(py_bytes.as_bytes())
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "__bytes__() returned non-bytes object: {}",
                        bytes_obj.get_type().name()?
                    )));
                }
            }
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "Cannot convert data to bytes: {}. Expected bytes-like object",
                    e
                )));
            }
        }
    };

    py.detach(move || {
        // Build the store on the caller thread BEFORE submit_io (issue #173).
        let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;

        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move { put_verified_with_retry(store.as_ref(), &uri, data_owned).await })
    })
}

/// Async version of put_bytes() - accepts any bytes-like object
///
/// Accepts (in order of optimization):
/// 1. Buffer protocol objects (dgen_py.BytesView, numpy arrays, bytearray, memoryview) - optimized copy
/// 2. s3dlio's own BytesView - zero-copy Arc clone
/// 3. Python bytes - direct copy
/// 4. Any object with __bytes__() method - converted via bytes()
///
/// Design philosophy: Be liberal in what you accept, optimize based on what you get.
/// Never reject valid data - match the permissiveness of minio/boto3/s3torch.
#[pyfunction]
fn put_bytes_async<'py>(
    py: Python<'py>,
    uri: &str,
    data: &Bound<'_, PyAny>,
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();

    // Try optimal paths first, fall back gracefully
    let data_owned = if let Ok(buffer) = PyBuffer::<u8>::get(data) {
        // Buffer protocol (dgen_py.BytesView, numpy, bytearray, memoryview, etc.)
        // This is the most common path for external buffer objects
        let len = buffer.len_bytes();
        let mut vec = vec![0u8; len];
        buffer.copy_to_slice(py, &mut vec[..]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyBufferError, _>(format!("Buffer copy failed: {}", e))
        })?;

        Bytes::from(vec)
    } else if let Ok(bytes_view) = data.extract::<PyRef<PyBytesView>>() {
        // s3dlio's own BytesView - zero-copy Arc clone (optimal)
        bytes_view.bytes.clone()
    } else if let Ok(py_bytes) = data.cast::<PyBytes>() {
        // PyBytes - direct copy
        Bytes::copy_from_slice(py_bytes.as_bytes())
    } else {
        // Last resort: Try __bytes__() method or bytes() conversion
        // This handles any custom objects that can be converted to bytes
        let bytes_result = data.call_method0("__bytes__").or_else(|_| {
            // If __bytes__() doesn't exist, try bytes(data)
            let bytes_type = py.get_type::<PyBytes>();
            bytes_type.call1((data,))
        });

        match bytes_result {
            Ok(bytes_obj) => {
                if let Ok(py_bytes) = bytes_obj.cast::<PyBytes>() {
                    Bytes::copy_from_slice(py_bytes.as_bytes())
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "__bytes__() returned non-bytes object: {}",
                        bytes_obj.get_type().name()?
                    )));
                }
            }
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    format!("Cannot convert data to bytes: {}. Expected bytes-like object (bytes, bytearray, memoryview, BytesView, numpy array, or object with __bytes__() method)", e)
                ));
            }
        }
    };

    future_into_py(py, async move {
        let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;
        put_verified_with_retry(store.as_ref(), &uri, data_owned)
            .await
            .map_err(py_err)
    })
}

// ---------------------------------------------------------------------------
// mkdir() - Create directory/prefix for any backend
// ---------------------------------------------------------------------------

/// Create a directory (file://, direct://) or prefix marker (s3://, az://, gs://).
///
/// # Backend Behavior
/// - `file://`, `direct://`: Creates actual directory using create_dir_all()
/// - `s3://`, `az://`, `gs://`: Creates empty marker object (no-op for some backends)
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
/// # Create S3 prefix marker
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
        store.mkdir(&uri).await.map_err(py_err)
    })
}

#[pyfunction(name = "get_many_stats")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub fn get_many_stats(
    py: Python<'_>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<(usize, usize)> {
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
                    let sem = Arc::new(Semaphore::new(max_in_flight));
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
                    let sem = Arc::new(Semaphore::new(max_in_flight));
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
                // Use store_for_uri for Azure and GCS
                // Submit to global runtime (io_uring pattern — never calls block_on)
                let uris_clone = uris.clone();
                let res = submit_io(async move {
                    let sem = Arc::new(Semaphore::new(max_in_flight));
                    let mut futs = FuturesUnordered::new();

                    for uri in uris_clone.iter().cloned() {
                        let sem = Arc::clone(&sem);
                        futs.push(tokio::spawn(async move {
                            let _permit = sem.acquire_owned().await.unwrap();

                            // Use store_for_uri to get appropriate backend
                            let store = store_for_uri(&uri)?;

                            // Extract the key from the URI
                            let key = if let Some(idx) = uri.find("://") {
                                if let Some(slash_idx) = uri[idx + 3..].find('/') {
                                    &uri[idx + 3 + slash_idx + 1..]
                                } else {
                                    ""
                                }
                            } else {
                                &uri
                            };

                            if key.is_empty() {
                                anyhow::bail!("Cannot GET: no key specified in URI: {}", uri);
                            }

                            let bytes = store.get(key).await?;
                            Ok::<_, anyhow::Error>((uri, bytes))
                        }));
                    }

                    let mut out = Vec::new();
                    while let Some(res) = futs.next().await {
                        let (uri, bytes) = res
                            .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
                            .map_err(|e| anyhow::anyhow!("Get error: {}", e))?;
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
/// ``delete(uri, recursive=True)`` deletes the real objects whose key starts
/// with the LITERAL prefix given: listed without a delimiter, objects only (a
/// "directory" is never a deletion target), no ``/`` appended, so the prefix
/// ``foo`` also matches ``foobar``; add the trailing ``/`` yourself to mean a
/// directory. A prefix with no objects is a no-op. Each selected object gets
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

/// One `put_many` item with its store already resolved (or the resolution
/// error, surfaced from that item's task): (uri, data, store).
type PreparedPut = (String, Bytes, anyhow::Result<Arc<dyn ObjectStore>>);

/// Put multiple objects in parallel from Python.
///
/// Accepts a list of (uri, data) tuples where data is Python bytes.
/// All put operations execute concurrently inside Rust's async runtime
/// using FuturesUnordered + Semaphore for backpressure — similar to
/// how get_objects_parallel works for reads.
///
/// # Arguments
/// * `items` - List of (uri, data) tuples: uri is a string, data is bytes
/// * `max_in_flight` - Maximum concurrent put operations (default: 64)
///
/// # Zero-copy path
/// Each Python bytes is copied once into Bytes (unavoidable — Python GIL →
/// Rust ownership boundary). After that, the Bytes (Arc-based) flows through
/// the async pipeline with zero additional copies.
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
    items: Vec<(String, Vec<u8>)>,
    max_in_flight: usize,
) -> PyResult<()> {
    use futures::stream::{FuturesUnordered, StreamExt};
    use tokio::sync::Semaphore;

    // Convert Vec<u8> → Bytes on the Python thread (one copy per item,
    // after this everything is zero-copy via Arc)
    let owned_items: Vec<(String, Bytes)> = items
        .into_iter()
        .map(|(uri, data)| (uri, Bytes::from(data)))
        .collect();

    py.detach(move || {
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
            let sem = Arc::new(Semaphore::new(max_in_flight));
            let mut futs = FuturesUnordered::new();

            for (uri, data, store) in prepared {
                let sem = Arc::clone(&sem);
                futs.push(tokio::spawn(async move {
                    let _permit = sem.acquire_owned().await.unwrap();
                    let store = store?;
                    store.put(&uri, data).await?;
                    Ok::<_, anyhow::Error>(())
                }));
            }

            // Drain all futures, propagate first error
            while let Some(res) = futs.next().await {
                res.map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;
            }
            Ok(())
        })
    })
}

/// Async version of put_many()
#[pyfunction]
#[pyo3(signature = (items, max_in_flight = 64))]
pub fn put_many_async<'py>(
    py: Python<'py>,
    items: Vec<(String, Vec<u8>)>,
    max_in_flight: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use futures::stream::{FuturesUnordered, StreamExt};
    use tokio::sync::Semaphore;

    // Convert Vec<u8> → Bytes
    let owned_items: Vec<(String, Bytes)> = items
        .into_iter()
        .map(|(uri, data)| (uri, Bytes::from(data)))
        .collect();

    future_into_py(py, async move {
        let sem = Arc::new(Semaphore::new(max_in_flight));
        let mut futs = FuturesUnordered::new();

        for (uri, data) in owned_items {
            let sem = Arc::clone(&sem);
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let store = get_or_create_store(&uri, global_logger()).map_err(py_err)?;
                store.put(&uri, data).await.map_err(py_err)?;
                Ok::<_, PyErr>(())
            }));
        }

        while let Some(res) = futs.next().await {
            res.map_err(py_err)??;
        }
        Ok(())
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

    /// Lock `inner` from a thread that holds the GIL, without ever waiting for the lock
    /// while holding it: a thread writing with the GIL released needs the GIL back to
    /// return, so waiting here with the GIL would deadlock.
    fn lock_with_gil(
        &self,
        py: Python<'_>,
    ) -> std::sync::MutexGuard<'_, Option<Box<dyn ObjectWriter>>> {
        loop {
            match self.inner.try_lock() {
                Ok(guard) => return guard,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => return poisoned.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    // Wait for the holder with the GIL released, then try again.
                    py.detach(|| drop(lock_ignoring_poison(&self.inner)));
                }
            }
        }
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
    /// call without the GIL: immutable `bytes` or an owned copy.
    fn write_detached(&self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        py.detach(|| {
            let mut guard = lock_ignoring_poison(&self.inner);
            self.write_locked(&mut guard, data)
        })
    }

    /// `write_owned_bytes` for data already copied out of Python, GIL released.
    fn write_owned_detached(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<()> {
        py.detach(|| {
            let mut guard = lock_ignoring_poison(&self.inner);
            let writer = guard.take().ok_or_else(Self::finalized_error)?;
            // The writer is moved into the 'static future for `submit_io` and handed
            // back, so a failed write leaves it usable for a retry or `finalize()`.
            let (writer, result) = submit_io(async move {
                let mut writer = writer;
                let r = writer.write_owned_bytes(data).await;
                Ok::<_, anyhow::Error>((writer, r))
            })?;
            self.publish_stats(writer.as_ref());
            *guard = Some(writer);
            result.map_err(|e| py_err(e.context("Failed to write owned bytes")))
        })
    }
}

#[pymethods]
impl PyObjectWriter {
    /// Write a chunk of bytes to the stream (zero-copy via buffer protocol)
    ///
    /// Accepts any Python object supporting the buffer protocol:
    /// - bytes, bytearray, memoryview
    /// - NumPy arrays, PyTorch tensors
    /// - Any object with __buffer__ method
    ///
    /// No copy is made, except for a non-contiguous view (a strided or reversed
    /// memoryview, a transposed array), which is copied once into contiguous memory.
    ///
    /// GIL: it is released for the blocking write (issue #201) only when `data` is an
    /// exact `bytes` object, which no thread can change, or after the copy of a
    /// non-contiguous view. For every other exporter (bytearray, a writable memoryview,
    /// numpy, mmap, ctypes, ...) the GIL is kept for the write, as before #201, because
    /// another thread could otherwise change the memory while it is being read. To keep
    /// other threads running during a large write of mutable data, pass `bytes(data)`.
    ///
    /// Calls on one writer from several threads run one after another.
    fn write_chunk(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        // An exact `bytes` object is immutable and `data` keeps it alive: zero-copy and
        // safe to read without the GIL.
        if data.is_exact_instance_of::<PyBytes>() {
            if let Ok(bytes) = data.cast::<PyBytes>() {
                return self.write_detached(py, bytes.as_bytes());
            }
        }

        if let Ok(buffer) = PyBuffer::<u8>::get(data) {
            if !buffer.is_c_contiguous() {
                // The buffer pointer and byte length do not describe the view's bytes
                // (strides). Copy them in logical order under the GIL, then write the
                // owned copy without it.
                let owned = buffer.to_vec(py).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyBufferError, _>(format!(
                        "Buffer copy failed: {}",
                        e
                    ))
                })?;
                drop(buffer);
                return self.write_detached(py, &owned);
            }

            // Take the writer lock FIRST. Waiting for it may release the GIL, and a
            // reference over the caller's memory must not exist while another Python
            // thread can run and change that memory.
            let mut guard = self.lock_with_gil(py);
            let result = if buffer.len_bytes() == 0 {
                self.write_locked(&mut guard, &[])
            } else {
                // SAFETY: the buffer is C-contiguous and `buffer` is a live Py_buffer
                // export, released only after the write has returned, so the exporter
                // cannot free or resize the memory meanwhile. The slice is created
                // only now, after the lock was acquired, and the GIL is HELD from
                // here through the end of the write (`write_locked` does not release
                // it), so no Python thread can change the contents while the slice
                // exists. Non-Python code that shares the memory (a mmap or ctypes
                // buffer written by native code) is the caller's responsibility, as
                // in any buffer-protocol consumer. The write only reads from the slice.
                let slice = unsafe {
                    std::slice::from_raw_parts(buffer.buf_ptr() as *const u8, buffer.len_bytes())
                };
                self.write_locked(&mut guard, slice)
            };
            drop(guard);
            // Release the export only now.
            drop(buffer);
            return result;
        }

        Err(PyRuntimeError::new_err(
            "write_chunk requires bytes-like object (bytes, bytearray, memoryview, numpy array, etc.)"
        ))
    }

    /// Write owned bytes (copies the input out of Python, then writes with the GIL released)
    ///
    /// This method copies data but takes ownership, useful when the caller
    /// doesn't need the buffer anymore, and it releases the GIL for the whole write
    /// whatever the type of `data` (the copy is made under the GIL, so a concurrent
    /// change to a mutable buffer cannot tear it). For a zero-copy write of `bytes`
    /// use write_chunk().
    ///
    /// Implementation note: the writer is temporarily moved out of the lock into the
    /// `'static` future `submit_io` needs, and always put back afterwards whether or not
    /// the write succeeded, so a failed `write_owned_bytes()` leaves the writer usable
    /// for a retry or a clean `finalize()`.
    fn write_owned_bytes(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try buffer protocol
        if let Ok(buffer) = PyBuffer::<u8>::get(data) {
            let len = buffer.len_bytes();
            let mut vec = vec![0u8; len];
            buffer
                .copy_to_slice(py, &mut vec[..])
                .map_err(|e| PyRuntimeError::new_err(format!("Buffer copy failed: {}", e)))?;
            // The copy is made: release the export before the GIL is released.
            drop(buffer);
            return self.write_owned_detached(py, vec);
        }

        // Fallback for PyBytes
        if let Ok(bytes) = data.cast::<PyBytes>() {
            return self.write_owned_detached(py, bytes.as_bytes().to_vec());
        }

        Err(PyRuntimeError::new_err(
            "write_owned_bytes requires bytes-like object",
        ))
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
pub fn create_azure_writer(
    _py: Python<'_>,
    _uri: String,
    _options: Option<&PyWriterOptions>,
) -> PyResult<PyObjectWriter> {
    Err(PyRuntimeError::new_err(
        "Azure backend is not enabled in this build. Rebuild with feature 'backend-azure' or 'full-backends'."
    ))
}

/// Create a streaming writer for filesystem
#[pyfunction]
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

    /// Put an object to the multi-endpoint store
    fn put<'py>(&self, py: Python<'py>, uri: &str, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let data = Bytes::copy_from_slice(data);
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .put(&uri, data)
                .await
                .map_err(|e| py_err(e.context("Put failed")))?;
            Ok(())
        })
    }

    /// List objects under a prefix
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
            Ok(())
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
    fn put_to_endpoint<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        uri: &str,
        data: &[u8],
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let data = Bytes::copy_from_slice(data);
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .put_to_endpoint(index, &uri, data)
                .await
                .map_err(|e| py_err(e.context("Put to endpoint failed")))?;
            Ok(())
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
            Ok(())
        })
    }

    /// Put the same object to every configured endpoint (fan-out write), for
    /// true-replication deployments that want a guaranteed-consistent write
    /// instead of relying on independent round-robin ``put()`` calls to
    /// eventually cover every endpoint. Mirrors ``list_all_endpoints`` for
    /// the write side. Raises if the write failed on any endpoint (not
    /// transactional -- some endpoints may already hold the write).
    fn put_all_endpoints<'py>(
        &self,
        py: Python<'py>,
        uri: &str,
        data: &[u8],
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let data = Bytes::copy_from_slice(data);
        let store = self.store.clone();

        future_into_py(py, async move {
            store
                .put_all_endpoints(&uri, data)
                .await
                .map_err(|e| py_err(e.context("Put-all-endpoints failed")))?;
            Ok(())
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
fn gcs_set_channel_count(_n: usize) {}

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
fn gcs_set_rapid_mode(force: Option<bool>) {
    crate::google_gcs_client::set_gcs_rapid_mode(force);
}

#[pyfunction]
#[cfg(not(feature = "backend-gcs"))]
fn gcs_set_rapid_mode(_force: Option<bool>) {}

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
fn gcs_query_rapid_bucket(_bucket_or_uri: &str) -> PyResult<bool> {
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
// Tests -- PyObjectWriter::write_chunk must not create a reference over a mutable
// exporter's memory before it holds the writer lock: waiting for the lock releases
// the GIL, and another Python thread may then change that memory under the reference.
//
// A behavioural RED is not possible: the read itself happens with the GIL held, so
// the only difference is a live `&[u8]` over memory that changes during the wait
// (undefined behaviour in the abstract machine, no observable torn data). The order
// of the two steps is therefore pinned structurally.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod write_chunk_slice_after_lock_tests {
    #[test]
    fn the_slice_over_a_mutable_buffer_is_created_after_the_writer_lock_is_taken() {
        let src = include_str!("python_core_api.rs");
        let start = src
            .find("fn write_chunk(&self, py: Python<'_>")
            .expect("write_chunk not found");
        let rest = &src[start..];
        let end = rest.find("\n    }\n").expect("end of write_chunk");
        let body = &rest[..end];

        assert_eq!(
            body.matches("self.lock_with_gil(py)").count(),
            1,
            "RED: write_chunk takes the writer lock in more than one place, so a slice can be \
             built on a path that has not taken it yet"
        );
        let lock = body
            .find("self.lock_with_gil(py)")
            .expect("RED: write_chunk's GIL-held path does not take the writer lock");
        let slice = body
            .find("from_raw_parts(")
            .expect("RED: write_chunk's GIL-held path builds no slice");
        assert!(
            lock < slice,
            "RED: the slice over the caller's buffer is created before the writer lock is \
             acquired (acquiring it can release the GIL)"
        );
    }
}
