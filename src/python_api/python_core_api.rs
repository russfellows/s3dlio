// src/python_api/python_core_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::Bound;
use pyo3::exceptions::PyRuntimeError;
use pyo3::conversion::IntoPyObjectExt;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3::ffi;
use pyo3::buffer::PyBuffer;
use bytes::Bytes;

use tokio::task;

use std::path::PathBuf;

use tracing::{Level, warn};
use tracing_subscriber;

// Project crates
use crate::config::{ObjectType, DataGenMode, DataGenAlgorithm, Config};
use crate::s3_utils::{
    get_objects_parallel,
    parse_s3_uri, put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE,
    create_bucket as create_bucket_rs, delete_bucket as delete_bucket_rs,
    stat_object_many_async,
};
use crate::{generic_upload_files, generic_download_objects};
use crate::s3_logger::{finalize_op_logger, init_op_logger, global_logger};
use crate::object_store::{store_for_uri_with_logger, store_for_uri};
use crate::s3_client::run_on_global_rt;

// Phase 2 streaming functionality imports
use crate::object_store::{
    ObjectStore, ObjectWriter, WriterOptions, CompressionConfig,
    S3ObjectStore, AzureObjectStore
};
use crate::file_store::FileSystemObjectStore;
use crate::file_store_direct::ConfigurableFileSystemObjectStore;

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

use once_cell::sync::Lazy;
use dashmap::DashMap;
use std::sync::Arc;

/// Cache key: uniquely identifies an ObjectStore configuration
/// 
/// Note: We DON'T include bucket in the key because AWS clients work across
/// all buckets in a region/endpoint. This maximizes cache hit rate.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    scheme: String,      // "s3", "az", "gs", "file", "direct"
    endpoint: String,    // AWS_ENDPOINT_URL or default
    region: String,      // AWS_REGION or default
}

impl StoreKey {
    fn from_uri(uri: &str) -> Self {
        let scheme = uri.split("://").next().unwrap_or("s3");
        let endpoint = std::env::var("AWS_ENDPOINT_URL").unwrap_or_default();
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        
        StoreKey {
            scheme: scheme.to_string(),
            endpoint,
            region,
        }
    }
}

/// Global cache of ObjectStore instances
/// 
/// DashMap provides lock-free concurrent access (better than RwLock<HashMap>)
/// for read-heavy workloads. Expected cache hit rate: >99% in typical workloads.
/// 
/// Performance: <100ns per lookup (DashMap sharded locking)
static STORE_CACHE: Lazy<DashMap<StoreKey, Arc<dyn ObjectStore>>> = Lazy::new(DashMap::new);

/// Get or create a cached ObjectStore instance.
///
/// store_for_uri_with_logger() is synchronous — no runtime needed for creation.
/// The store is cached by (scheme, endpoint, region) for >99% cache hit rate.
fn get_or_create_store(uri: &str, logger: Option<crate::s3_logger::Logger>) -> anyhow::Result<Arc<dyn ObjectStore>> {
    let key = StoreKey::from_uri(uri);
    
    // Fast path: Store already exists (cache hit >99% in typical workloads)
    if let Some(store) = STORE_CACHE.get(&key) {
        return Ok(store.clone());
    }
    
    // Slow path: Create new store — store_for_uri_with_logger is SYNC,
    // no runtime needed. The store itself is Send + Sync and will use
    // the global runtime's I/O reactor when async methods are called.
    let store_box: Box<dyn ObjectStore> = store_for_uri_with_logger(uri, logger)?;
    
    let store_arc: Arc<dyn ObjectStore> = Arc::from(store_box);
    STORE_CACHE.insert(key, store_arc.clone());
    
    Ok(store_arc)
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
// Zero-Copy Buffer Support
// ---------------------------------------------------------------------------

/// A Python-visible wrapper around Rust Bytes that exposes buffer protocol
/// This allows Python code to get a memoryview without copying data
/// 
/// Implements the Python buffer protocol via __getbuffer__ and __releasebuffer__
/// so that `memoryview(bytes_view)` works directly with zero-copy access.
#[pyclass(name = "BytesView")]
pub struct PyBytesView {
    /// The underlying Bytes (reference-counted, cheap to clone)
    bytes: Bytes,
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
    /// This allows `memoryview(bytes_view)` to work directly.
    /// 
    /// The buffer is read-only; requesting a writable buffer will raise BufferError.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        // Check for writable request - we only support read-only buffers
        if (flags & ffi::PyBUF_WRITABLE) != 0 {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "BytesView is read-only and does not support writable buffers"
            ));
        }
        
        let bytes = &slf.bytes;
        
        // Fill in the Py_buffer struct
        // Safety: view is a valid pointer provided by Python
        unsafe {
            (*view).buf = bytes.as_ptr() as *mut std::os::raw::c_void;
            (*view).len = bytes.len() as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            
            // Format string: "B" = unsigned byte (matches u8)
            (*view).format = if (flags & ffi::PyBUF_FORMAT) != 0 {
                // Static string that lives for the duration of the program
                b"B\0".as_ptr() as *mut std::os::raw::c_char
            } else {
                std::ptr::null_mut()
            };
            
            (*view).ndim = 1;
            
            // Shape: pointer to the length (1D array of len elements)
            (*view).shape = if (flags & ffi::PyBUF_ND) != 0 {
                // Point to our len field - this is a 1D array
                &(*view).len as *const isize as *mut isize
            } else {
                std::ptr::null_mut()
            };
            
            // Strides: 1 byte per element
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) != 0 {
                &(*view).itemsize as *const isize as *mut isize
            } else {
                std::ptr::null_mut()
            };
            
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            
            // CRITICAL: Store a reference to the PyBytesView object
            // This prevents the Bytes data from being deallocated while the buffer is in use
            (*view).obj = slf.as_ptr() as *mut ffi::PyObject;
            ffi::Py_INCREF((*view).obj);
        }
        
        Ok(())
    }
    
    /// Release the buffer - called when the memoryview is garbage collected.
    /// We don't need to do anything here since the Bytes is reference-counted.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // Nothing to do - the Py_DECREF on view.obj will be handled by Python
        // and will eventually drop the PyBytesView (and thus the Bytes) when refcount hits 0
    }
    
    /// Get a memoryview (TRUE zero-copy readonly access)
    /// Uses PyMemoryView_FromMemory from the Python C API to create a memoryview
    /// that directly references the Rust Bytes buffer without copying.
    fn memoryview(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ptr = self.bytes.as_ptr() as *mut i8;
        let len = self.bytes.len() as isize;
        
        // PyBUF_READ = 0x100 (read-only buffer)
        let raw_mv = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, ffi::PyBUF_READ) };
        if raw_mv.is_null() {
            // If something went wrong, fetch and return the Python exception
            Err(PyErr::fetch(py))
        } else {
            // Convert the raw pointer into a safe Py<PyAny> object
            Ok(unsafe { Py::from_owned_ptr(py, raw_mv) })
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
    /// Create from Rust Bytes
    pub fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }
}

pub fn py_err<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(format!("{}", error))
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
    use crate::mp::{MpGetConfigBuilder, run_get_shards};
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    Python::attach(|py| {
        // Parse S3 URI
        let (bucket, key_prefix) = parse_s3_uri(uri).map_err(py_err)?;
        
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
        dict.set_item("total_objects", result.total_objects)?;
        dict.set_item("total_bytes", result.total_bytes)?;
        dict.set_item("duration_seconds", result.elapsed_seconds)?;
        dict.set_item("throughput_mb_s", result.throughput_mbps())?;
        dict.set_item("ops_per_sec", result.ops_per_second())?;
        
        // Add per-worker stats
        let workers = PyList::empty(py);
        for worker in &result.per_worker {
            let worker_dict = PyDict::new(py);
            worker_dict.set_item("worker_id", worker.worker_id)?;
            worker_dict.set_item("objects", worker.objects)?;
            worker_dict.set_item("bytes", worker.bytes)?;
            let worker_throughput = if result.elapsed_seconds > 0.0 {
                (worker.bytes as f64 / (1024.0 * 1024.0)) / result.elapsed_seconds
            } else {
                0.0
            };
            worker_dict.set_item("throughput_mb_s", worker_throughput)?;
            workers.append(worker_dict)?;
        }
        dict.set_item("workers", workers)?;
        
        Ok(dict.into_py_any(py)?.into())
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
        "warn"  => Level::WARN,
        "error" => Level::ERROR,
        _       => Level::INFO,
    };
    
    // Initialize tracing subscriber with the specified level
    let _ = tracing_subscriber::fmt()
        .with_max_level(filter)
        .with_target(false)
        .try_init();
    
    // Initialize tracing-log bridge to capture log crate messages
    tracing_log::LogTracer::init().ok();
    
    Ok(())
}

#[pyfunction]
pub fn init_op_log(path: &str) -> PyResult<()> { init_op_logger(path).map_err(py_err) }

#[pyfunction]
pub fn finalize_op_log() -> PyResult<()> { finalize_op_logger(); Ok(()) }

#[pyfunction]
pub fn is_op_log_active() -> PyResult<bool> {
    Ok(global_logger().is_some())
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
fn build_uri_list(prefix: &str, template: &str, num: usize) -> PyResult<(String, Vec<String>)> {
    // Extract scheme and parse URI generically (works for s3://, az://, file://, etc.)
    let scheme_end = prefix.find("://").ok_or_else(|| {
        anyhow::anyhow!("URI must contain scheme (e.g., s3://, az://, file://)")
    }).map_err(py_err)?;
    
    let scheme = &prefix[..scheme_end + 3]; // Include "://"
    let remainder = &prefix[scheme_end + 3..];
    
    // Split into bucket/container and key prefix
    let (bucket, mut key_prefix) = if let Some(slash_pos) = remainder.find('/') {
        (remainder[..slash_pos].to_string(), remainder[slash_pos + 1..].to_string())
    } else {
        (remainder.to_string(), String::new())
    };
    
    if !key_prefix.is_empty() && !key_prefix.ends_with('/') { 
        key_prefix.push('/'); 
    }
    
    let uris = (0..num).map(|i| {
        let name = template
            .replacen("{}", &i.to_string(), 1)
            .replacen("{}", &num.to_string(), 1);
        format!("{}{}/{}{}", scheme, bucket, key_prefix, name)
    }).collect();
    Ok((bucket, uris))
}
fn str_to_obj(s: &str) -> ObjectType { ObjectType::from(s) }

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
    // --- Add consistent prefix handling, just like in the CLI ---
    let final_bucket_name = bucket_name
        .strip_prefix("s3://")
        .unwrap_or(bucket_name)
        .trim_end_matches('/')
        .to_string(); // Own the string to move it across the thread boundary

    // Release the GIL to allow other Python threads to run
    py.detach(move || {
        create_bucket_rs(&final_bucket_name).map_err(py_err)
    })
}

// - delete-bucket command
#[pyfunction]
pub fn delete_bucket(py: Python<'_>, bucket_name: &str) -> PyResult<()> {
    // Clean the input string to get a valid bucket name
    let final_bucket_name = bucket_name
        .strip_prefix("s3://")
        .unwrap_or(bucket_name)
        .trim_end_matches('/')
        .to_string(); // Own the string to move across the thread boundary

    // Release the GIL for the blocking S3 call
    py.detach(move || {
        delete_bucket_rs(&final_bucket_name).map_err(py_err)
    })
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
pub fn put(
    py: Python<'_>,
    prefix: &str, num: usize, template: Option<&str>,
    max_in_flight: usize, size: Option<usize>,
    should_create_bucket: bool, object_type: &str,
    dedup_factor: usize, compress_factor: usize,
    data_gen_algorithm: &str, data_gen_mode: &str, chunk_size: usize,
) -> PyResult<()> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let template = template.unwrap_or("object-{}");
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);
    let algorithm = str_to_data_gen_algorithm(data_gen_algorithm);
    let mode = str_to_data_gen_mode(data_gen_mode);
    let config = Config::new_with_defaults(obj, 1, sz, dedup_factor, compress_factor)
        .with_data_gen_algorithm(algorithm)
        .with_data_gen_mode(mode)
        .with_chunk_size(chunk_size);
    py.detach(|| {
        if should_create_bucket { let _ = create_bucket_rs(&bucket); }
        put_objects_with_random_data_and_type(&uris, sz, jobs, config)
            .map_err(py_err)
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
pub (crate) fn put_async_py<'p>(
    py: Python<'p>,
    prefix: &'p str, num: usize, template: Option<&'p str>,
    max_in_flight: usize, size: Option<usize>,
    should_create_bucket: bool, object_type: &'p str,
    dedup_factor: usize, compress_factor: usize,
    data_gen_algorithm: &'p str, data_gen_mode: &'p str, chunk_size: usize,
) -> PyResult<Bound<'p, PyAny>> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let template = template.unwrap_or("object-{}");
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);
    let algorithm = str_to_data_gen_algorithm(data_gen_algorithm);
    let mode = str_to_data_gen_mode(data_gen_mode);
    let config = Config::new_with_defaults(obj, 1, sz, dedup_factor, compress_factor)
        .with_data_gen_algorithm(algorithm)
        .with_data_gen_mode(mode)
        .with_chunk_size(chunk_size);

    future_into_py(py, async move {
        task::spawn_blocking(move || {
            if should_create_bucket { let _ = create_bucket_rs(&bucket); }
            put_objects_with_random_data_and_type(&uris, sz, jobs, config)
                .map_err(py_err)
        })
        .await
        .map_err(py_err)??;
        Python::attach(|py| Ok(py.None()))
    })
}


// --- `list` function
#[pyfunction]
#[pyo3(signature = (uri, recursive = false, pattern = None))]
pub fn list(uri: &str, recursive: bool, pattern: Option<&str>) -> PyResult<Vec<String>> {
    // Universal list using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let logger = global_logger();
    let store = get_or_create_store(uri, logger).map_err(py_err)?;
    let uri_owned = uri.to_owned();
    
    // Submit to global runtime (io_uring pattern — never calls block_on)
    let mut keys = submit_io(async move {
        store.list(&uri_owned, recursive).await
    })?;
    
    // Apply client-side regex filtering if pattern provided
    if let Some(pat) = pattern {
        use regex::Regex;
        let re = Regex::new(pat).map_err(py_err)?;
        keys.retain(|k| re.is_match(k));
    }
    
    Ok(keys)
}


// New stat calls, almost the same ????
//
#[pyfunction]
pub fn stat(py: Python<'_>, uri: &str) -> PyResult<Py<PyAny>> {
    // Universal stat using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let logger = global_logger();
    let store = get_or_create_store(uri, logger).map_err(py_err)?;
    let uri_owned = uri.to_owned();
    
    // Submit to global runtime (io_uring pattern — never calls block_on)
    let os = submit_io(async move {
        store.stat(&uri_owned).await
    })?;
    
    let d = stat_to_pydict(py, os)?;
    Ok(d.unbind().into())
}

#[pyfunction]
fn stat_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    let logger = global_logger();
    let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
    
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
fn stat_to_pydict<'py>(py: Python<'py>, os: crate::s3_utils::ObjectStat)
    -> PyResult<pyo3::Bound<'py, PyDict>>
{
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
#[pyfunction]
pub fn exists(py: Python<'_>, uri: &str) -> PyResult<bool> {
    let uri_owned = uri.to_owned();
    py.detach(|| {
        let logger = global_logger();
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let store = match get_or_create_store(&uri_owned, logger) {
                Ok(s) => s,
                Err(_) => return Ok(false), // URI parsing error → doesn't exist
            };
            
            match store.stat(&uri_owned).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        })
    })
}

/// Async version of exists()
/// Works with all backends: S3, GCS, Azure, file://, direct://
#[pyfunction]
fn exists_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    let logger = global_logger();
    let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
    
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
/// # Put bytearray (also zero-copy)
/// data = bytearray(b"hello world")
/// s3dlio.put_bytes("file:///tmp/test.bin", data)
/// ```
#[pyfunction]
pub fn put_bytes(py: Python<'_>, uri: &str, data: &Bound<'_, PyBytes>) -> PyResult<()> {
    // PyBytes.as_bytes() gives us a &[u8] view without copying
    let data_slice: &[u8] = data.as_bytes();
    let uri = uri.to_owned();
    let data_owned = Bytes::copy_from_slice(data_slice);
    
    py.detach(move || {
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let logger = global_logger();
            
            // Get cached store (or create if first time)
            let store = get_or_create_store(&uri, logger)?;
            
            // Execute operation using cached client — zero-copy Bytes
            store.put(&uri, data_owned).await
        })
    })
}

/// Async version of put_bytes() - accepts BytesView (zero-copy via Arc clone) or PyBytes (copied)
#[pyfunction]
fn put_bytes_async<'py>(py: Python<'py>, uri: &str, data: &Bound<'_, PyAny>) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    
    // Try to extract BytesView first (zero-copy via Arc clone)
    let data_owned = if let Ok(bytes_view) = data.extract::<PyRef<PyBytesView>>() {
        // Zero-copy: Clone the Arc-based Bytes (cheap pointer increment, no data copy)
        bytes_view.bytes.clone()
    } else if let Ok(py_bytes) = data.cast::<PyBytes>() {
        // Fallback: Copy data from PyBytes
        // We need to copy because PyBytes can't be sent across threads
        Bytes::copy_from_slice(py_bytes.as_bytes())
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "data must be BytesView or bytes"
        ));
    };
    
    future_into_py(py, async move {
        let logger = global_logger();
        let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
        store.put(&uri, data_owned).await.map_err(py_err)
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
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let logger = global_logger();
            let store = get_or_create_store(&uri, logger)?;
            store.mkdir(&uri).await
        })
    })
}

/// Async version of mkdir()
#[pyfunction]
fn mkdir_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    
    future_into_py(py, async move {
        let logger = global_logger();
        let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
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
pub (crate) fn get_many_async_py<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Bound<'p, PyAny>> {
    future_into_py(py, async move {
        // get_objects_parallel returns Vec<(String, Bytes)> - keep as Bytes for zero-copy
        let pairs: Vec<(String, Bytes)> = task::spawn_blocking(move || {
            get_objects_parallel(&uris, max_in_flight)
                .map_err(py_err)
        })
        .await
        .map_err(py_err)??;

        Python::attach(|_py| {
            let out = pairs.into_iter()
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
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let logger = global_logger();
            let store = get_or_create_store(&uri_owned, logger)?;
            let bytes = store.get(&uri_owned).await?;
            Ok(PyBytesView::new(bytes))
        })
    })
}

// --- `get_range` function (universal backend support)
#[pyfunction(name = "get_range")]
#[pyo3(signature = (uri, offset, length = None))]
pub fn get_range_py(py: Python<'_>, uri: &str, offset: u64, length: Option<u64>) -> PyResult<PyBytesView> {
    // Use universal ObjectStore interface for range requests
    let uri_owned = uri.to_owned();
    py.detach(|| {
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let logger = global_logger();
            let store = get_or_create_store(&uri_owned, logger)?;
            let bytes = store.get_range(&uri_owned, offset, length).await?;
            Ok(PyBytesView::new(bytes))
        })
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
    use tokio::sync::Semaphore;
    use futures::stream::{FuturesUnordered, StreamExt};
    use std::sync::Arc;
    
    py.detach(|| {
        // Check all URIs are the same scheme
        let schemes: Vec<_> = uris.iter().map(|u| infer_scheme(u)).collect();
        let first_scheme = schemes.first().ok_or_else(|| {
            PyRuntimeError::new_err("get_many requires at least one URI")
        })?;
        
        // Verify all URIs use the same scheme
        if !schemes.iter().all(|s| s == first_scheme) {
            return Err(PyRuntimeError::new_err(
                "get_many requires all URIs to use the same backend scheme"
            ));
        }
        
        // Route to appropriate backend
        match first_scheme {
            Scheme::S3 => {
                // Use existing optimized S3 implementation
                let res = get_objects_parallel(&uris, max_in_flight).map_err(py_err)?;
                Ok(res.into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::File | Scheme::Direct => {
                // Parallel file reads
                // Submit to global runtime (io_uring pattern — never calls block_on)
                let uris_clone = uris.clone();
                let res = submit_io(async move {
                    let sem = Arc::new(Semaphore::new(max_in_flight));
                    let mut futs = FuturesUnordered::new();
                    
                    for uri in uris_clone.iter().cloned() {
                        let sem = Arc::clone(&sem);
                        futs.push(tokio::spawn(async move {
                            let _permit = sem.acquire_owned().await.unwrap();
                            let path = uri.strip_prefix("file://")
                                .or_else(|| uri.strip_prefix("direct://"))
                                .unwrap_or(&uri);
                            let file_bytes = tokio::fs::read(path).await?;
                            Ok::<_, std::io::Error>((uri, Bytes::from(file_bytes)))
                        }));
                    }
                    
                    let mut out = Vec::new();
                    while let Some(res) = futs.next().await {
                        let (uri, bytes) = res.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
                            .map_err(|e| anyhow::anyhow!("File read error: {}", e))?;
                        out.push((uri, bytes));
                    }
                    
                    // Maintain input order
                    out.sort_by_key(|(u, _)| uris_clone.iter().position(|x| x == u).unwrap());
                    Ok::<_, anyhow::Error>(out)
                })?;
                
                Ok(res.into_iter()
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
                                if let Some(slash_idx) = uri[idx+3..].find('/') {
                                    &uri[idx+3+slash_idx+1..]
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
                        let (uri, bytes) = res.map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
                            .map_err(|e| anyhow::anyhow!("Get error: {}", e))?;
                        out.push((uri, bytes));
                    }
                    
                    // Maintain input order
                    out.sort_by_key(|(u, _)| uris_clone.iter().position(|x| x == u).unwrap());
                    Ok::<_, anyhow::Error>(out)
                })?;
                
                Ok(res.into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::Unknown => {
                Err(PyRuntimeError::new_err(format!(
                    "Unsupported URI scheme for get_many: {}", 
                    uris.first().unwrap_or(&String::new())
                )))
            }
        }
    })
}

// --- `delete` function
#[pyfunction]
#[pyo3(signature = (uri, recursive = false))]
pub fn delete(py: Python<'_>, uri: &str, recursive: bool) -> PyResult<()> {
    // Use universal ObjectStore API for deletion (with op-log support)
    let uri_owned = uri.to_owned();
    py.detach(|| {
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            let logger = global_logger();
            let store = get_or_create_store(&uri_owned, logger)?;
            
            // Check if URI contains wildcards or ends with / (pattern/directory)
            let has_pattern = uri_owned.contains('*') || uri_owned.contains('?') || uri_owned.ends_with('/');
            
            if has_pattern || recursive {
                // Need to list objects first, then delete them
                let list_results = store.list(&uri_owned, recursive).await?;
                
                if list_results.is_empty() {
                    // No objects matched - this is OK
                    return Ok(());
                }
                
                // Delete each object (list returns full URIs)
                for obj_uri in list_results {
                    store.delete(&obj_uri).await?;
                }
                Ok(())
            } else {
                // Simple single object deletion
                store.delete(&uri_owned).await
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
        // Submit to global runtime (io_uring pattern — never calls block_on)
        submit_io(async move {
            // Get logger if op-log is active
            let logger = global_logger();
            
            // Handle bucket creation ONLY if explicitly requested
            if create_bucket {
                if let Ok(store) = store_for_uri_with_logger(&dest_prefix_owned, logger.clone()) {
                    // Extract bucket/container name from URI
                    if dest_prefix_owned.starts_with("s3://") {
                        if let Ok((bucket, _)) = parse_s3_uri(&dest_prefix_owned) {
                            if let Err(e) = store.create_container(&bucket).await {
                                warn!("Failed to create bucket {}: {}", bucket, e);
                            }
                        }
                    } else if dest_prefix_owned.starts_with("az://") || dest_prefix_owned.starts_with("azure://") {
                        // For Azure, extract container name
                        let parts: Vec<&str> = dest_prefix_owned.trim_start_matches("az://").trim_start_matches("azure://").split('/').collect();
                        if let Some(container) = parts.get(0) {
                            if let Err(e) = store.create_container(container).await {
                                warn!("Failed to create container {}: {}", container, e);
                            }
                        }
                    }
                    // File backends don't need bucket creation, directories are created automatically
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
    use tokio::sync::Semaphore;
    use futures::stream::{FuturesUnordered, StreamExt};

    // Convert Vec<u8> → Bytes on the Python thread (one copy per item,
    // after this everything is zero-copy via Arc)
    let owned_items: Vec<(String, Bytes)> = items
        .into_iter()
        .map(|(uri, data)| (uri, Bytes::from(data)))
        .collect();

    py.detach(move || {
        // Submit entire batch to global runtime (io_uring pattern)
        submit_io(async move {
            let sem = Arc::new(Semaphore::new(max_in_flight));
            let mut futs = FuturesUnordered::new();
            let logger = global_logger();

            for (uri, data) in owned_items {
                let sem = Arc::clone(&sem);
                let logger = logger.clone();
                futs.push(tokio::spawn(async move {
                    let _permit = sem.acquire_owned().await.unwrap();
                    let store = get_or_create_store(&uri, logger)?;
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
    use tokio::sync::Semaphore;
    use futures::stream::{FuturesUnordered, StreamExt};

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
                let store = store_for_uri_with_logger(&uri, global_logger()).map_err(py_err)?;
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
            },
            "none" => CompressionConfig::None,
            _ => return Err(PyRuntimeError::new_err(format!("Unsupported compression type: {}", compression_type))),
        };
        
        self.inner = self.inner.clone().with_compression(compression);
        Ok(())
    }
    
    /// Set buffer size for the writer
    fn with_buffer_size(&mut self, size: usize) {
        self.inner = self.inner.clone().with_buffer_size(size);
    }
}

/// Python wrapper for ObjectWriter
#[pyclass]
pub struct PyObjectWriter {
    finalized_stats: Option<(u64, u64)>, // (bytes_written, compressed_bytes)
    inner: Option<Box<dyn ObjectWriter>>,
}

#[pymethods]
impl PyObjectWriter {
    /// Write a chunk of bytes to the stream (ZERO-COPY via buffer protocol)
    /// 
    /// Accepts any Python object supporting the buffer protocol:
    /// - bytes, bytearray, memoryview
    /// - NumPy arrays, PyTorch tensors
    /// - Any object with __buffer__ method
    /// 
    /// This method avoids copying data from Python to Rust by using PyBuffer
    /// to get a direct view of Python's memory during the synchronous write.
    fn write_chunk(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let writer = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer has been finalized"))?;
        
        // Try buffer protocol first (zero-copy path)
        if let Ok(buffer) = PyBuffer::<u8>::get(data) {
            // Get readonly slice - no copy!
            let slice = unsafe {
                // SAFETY: We hold the buffer for the entire duration of block_on,
                // so the memory remains valid. The GIL is held during block_on.
                std::slice::from_raw_parts(
                    buffer.buf_ptr() as *const u8,
                    buffer.len_bytes()
                )
            };
            
            // Synchronous write while buffer is alive
            return pyo3_async_runtimes::tokio::get_runtime().block_on(async {
                writer.write_chunk(slice).await
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write chunk: {}", e)))
            });
        }
        
        // Fallback for PyBytes (shouldn't happen, but safe)
        if let Ok(bytes) = data.cast::<PyBytes>() {
            let slice = bytes.as_bytes();
            return pyo3_async_runtimes::tokio::get_runtime().block_on(async {
                writer.write_chunk(slice).await
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write chunk: {}", e)))
            });
        }
        
        Err(PyRuntimeError::new_err(
            "write_chunk requires bytes-like object (bytes, bytearray, memoryview, numpy array, etc.)"
        ))
    }
    
    /// Write owned bytes (converts buffer protocol object to owned Vec for async)
    /// 
    /// This method copies data but takes ownership, useful when the caller
    /// doesn't need the buffer anymore. For true zero-copy, use write_chunk().
    fn write_owned_bytes(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let writer = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer has been finalized"))?;
        
        // Try buffer protocol
        if let Ok(buffer) = PyBuffer::<u8>::get(data) {
            let len = buffer.len_bytes();
            let mut vec = Vec::<u8>::with_capacity(len);
            unsafe { vec.set_len(len); }
            
            buffer.copy_to_slice(data.py(), &mut vec[..])
                .map_err(|e| PyRuntimeError::new_err(format!("Buffer copy failed: {}", e)))?;
            
            return pyo3_async_runtimes::tokio::get_runtime().block_on(async {
                writer.write_owned_bytes(vec).await
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write owned bytes: {}", e)))
            });
        }
        
        // Fallback for PyBytes
        if let Ok(bytes) = data.cast::<PyBytes>() {
            let vec = bytes.as_bytes().to_vec();
            return pyo3_async_runtimes::tokio::get_runtime().block_on(async {
                writer.write_owned_bytes(vec).await
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write owned bytes: {}", e)))
            });
        }
        
        Err(PyRuntimeError::new_err(
            "write_owned_bytes requires bytes-like object"
        ))
    }
    
    /// Finalize the writer and complete the upload
    fn finalize(&mut self, py: Python<'_>) -> PyResult<(u64, u64)> {
        if let Some(writer) = self.inner.take() {
            py.detach(|| {
                pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
                    let stats = (writer.bytes_written(), writer.compressed_bytes());
                    writer.finalize().await.map_err(py_err)?;
                    Ok::<(u64, u64), PyErr>(stats)
                })
            }).map(|stats| {
                self.finalized_stats = Some(stats);
                stats
            })
        } else {
            Err(PyRuntimeError::new_err("Writer already finalized"))
        }
    }
    
    /// Get the number of bytes written so far
    fn bytes_written(&self) -> PyResult<u64> {
        if let Some((bytes_written, _)) = self.finalized_stats {
            Ok(bytes_written)
        } else if let Some(writer) = &self.inner {
            Ok(writer.bytes_written())
        } else {
            Err(PyRuntimeError::new_err("Writer has been finalized"))
        }
    }
    
    /// Get the checksum of the data written (if available)
    fn checksum(&self) -> Option<String> {
        self.inner.as_ref().and_then(|w| w.checksum())
    }
    
    /// Get the number of compressed bytes (if compression is enabled)
    fn compressed_bytes(&self) -> PyResult<u64> {
        if let Some((_, compressed)) = self.finalized_stats {
            Ok(compressed)
        } else if let Some(writer) = &self.inner {
            Ok(writer.compressed_bytes())
        } else {
            Err(PyRuntimeError::new_err("Writer has been finalized"))
        }
    }
}

/// Create a streaming writer for S3
#[pyfunction]
pub fn create_s3_writer(py: Python<'_>, uri: String, options: Option<&PyWriterOptions>) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_else(WriterOptions::new);
    
    py.detach(|| {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
            let store = S3ObjectStore::new();
            let writer = store.create_writer(&uri, opts).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create S3 writer: {}", e)))?;
            
            Ok(PyObjectWriter { inner: Some(writer), finalized_stats: None })
        })
    })
}

/// Create a streaming writer for Azure Blob Storage
#[pyfunction]
pub fn create_azure_writer(py: Python<'_>, uri: String, options: Option<&PyWriterOptions>) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_else(WriterOptions::new);
    
    py.detach(|| {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
            let store = AzureObjectStore::new();
            let writer = store.create_writer(&uri, opts).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Azure writer: {}", e)))?;
            
            Ok(PyObjectWriter { inner: Some(writer), finalized_stats: None })
        })
    })
}

/// Create a streaming writer for filesystem
#[pyfunction]
pub fn create_filesystem_writer(py: Python<'_>, uri: String, options: Option<&PyWriterOptions>) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_else(WriterOptions::new);
    
    py.detach(|| {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
            let store = FileSystemObjectStore::new();
            let writer = store.create_writer(&uri, opts).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create filesystem writer: {}", e)))?;
            
            Ok(PyObjectWriter { inner: Some(writer), finalized_stats: None })
        })
    })
}

/// Create a streaming writer for direct I/O filesystem
#[pyfunction]
pub fn create_direct_filesystem_writer(py: Python<'_>, uri: String, options: Option<&PyWriterOptions>) -> PyResult<PyObjectWriter> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_else(WriterOptions::new);
    
    py.detach(|| {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
            let store = ConfigurableFileSystemObjectStore::new(Default::default());
            let writer = store.create_writer(&uri, opts).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create direct filesystem writer: {}", e)))?;
            
            Ok(PyObjectWriter { inner: Some(writer), finalized_stats: None })
        })
    })
}

// ---------------------------------------------------------------------------
// Multi-Endpoint Support (v0.9.14+)
// ---------------------------------------------------------------------------

use crate::multi_endpoint::{MultiEndpointStore, LoadBalanceStrategy};
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
            let data = store.get(&uri).await
                .map_err(|e| PyRuntimeError::new_err(format!("Get failed: {}", e)))?;
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
        length: Option<u64>
    ) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();
        
        future_into_py(py, async move {
            let data = store.get_range(&uri, offset, length).await
                .map_err(|e| PyRuntimeError::new_err(format!("Get range failed: {}", e)))?;
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
            store.put(&uri, data).await
                .map_err(|e| PyRuntimeError::new_err(format!("Put failed: {}", e)))?;
            Ok(())
        })
    }
    
    /// List objects under a prefix
    fn list<'py>(
        &self,
        py: Python<'py>,
        prefix: &str,
        recursive: bool
    ) -> PyResult<Bound<'py, PyAny>> {
        let prefix = prefix.to_string();
        let store = self.store.clone();
        
        future_into_py(py, async move {
            let objects = store.list(&prefix, recursive).await
                .map_err(|e| PyRuntimeError::new_err(format!("List failed: {}", e)))?;
            // Return Vec<String> - PyO3 automatically converts to Python list
            Ok(objects)
        })
    }
    
    /// Delete an object
    fn delete<'py>(&self, py: Python<'py>, uri: &str) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let store = self.store.clone();
        
        future_into_py(py, async move {
            store.delete(&uri).await
                .map_err(|e| PyRuntimeError::new_err(format!("Delete failed: {}", e)))?;
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
    strategy: Option<&str>
) -> PyResult<PyMultiEndpointStore> {
    let strategy = match strategy.unwrap_or("round_robin") {
        "round_robin" => LoadBalanceStrategy::RoundRobin,
        "least_connections" => LoadBalanceStrategy::LeastConnections,
        s => return Err(PyRuntimeError::new_err(format!(
            "Invalid strategy '{}'. Use 'round_robin' or 'least_connections'", s
        ))),
    };
    
    let store = MultiEndpointStore::new(uris, strategy, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create multi-endpoint store: {}", e)))?;
    
    Ok(PyMultiEndpointStore { store: Arc::new(store) })
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
    strategy: Option<&str>
) -> PyResult<PyMultiEndpointStore> {
    let uris = uri_utils::expand_uri_template(uri_template)
        .map_err(|e| PyRuntimeError::new_err(format!("Template expansion failed: {}", e)))?;
    
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
    strategy: Option<&str>
) -> PyResult<PyMultiEndpointStore> {
    let path = std::path::Path::new(file_path);
    let uris = uri_utils::load_uris_from_file(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load URIs from file: {}", e)))?;
    
    create_multi_endpoint_store(uris, strategy)
}

// ---------------------------------------------------------------------------
// Module registration function for core API
// ---------------------------------------------------------------------------
pub fn register_core_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Logging
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(init_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(is_op_log_active, m)?)?;
    
    // URI parsing utilities
    m.add_function(wrap_pyfunction!(parse_s3_uri_full, m)?)?;
    
    // Bucket management (S3-specific but kept for convenience)
    m.add_function(wrap_pyfunction!(create_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(delete_bucket, m)?)?;
    
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
    m.add_function(wrap_pyfunction!(get_range_py, m)?)?;
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
    m.add_class::<PyBytesView>()?;  // Zero-copy buffer wrapper
    m.add_function(wrap_pyfunction!(create_s3_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_azure_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_filesystem_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_direct_filesystem_writer, m)?)?;
    
    // Multi-endpoint support (v0.9.14+)
    m.add_class::<PyMultiEndpointStore>()?;
    m.add_function(wrap_pyfunction!(create_multi_endpoint_store, m)?)?;
    m.add_function(wrap_pyfunction!(create_multi_endpoint_store_from_template, m)?)?;
    m.add_function(wrap_pyfunction!(create_multi_endpoint_store_from_file, m)?)?;
    
    Ok(())
}

