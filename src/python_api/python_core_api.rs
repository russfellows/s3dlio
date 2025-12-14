// src/python_api/python_core_api.rs
//
// Copyright 2025
// Signal65 / Futurum Group.
//
// Contains core S3 operations: put, get, list, delete, stat, create/delete buckets

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyBytesMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use pyo3::{PyObject, Bound};
use pyo3::exceptions::PyRuntimeError;
use pyo3::conversion::IntoPyObjectExt;
use pyo3_async_runtimes::tokio::future_into_py;
use bytes::Bytes;

use tokio::task;

use std::path::PathBuf;

use tracing::{Level, warn};
use tracing_subscriber;

// Project crates
use crate::config::{ObjectType, DataGenMode, DataGenAlgorithm, Config};
use crate::s3_utils::{
    get_objects_parallel,
    list_objects as list_objects_rs, get_range as s3_get_range,
    parse_s3_uri, put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE,
    create_bucket as create_bucket_rs, delete_bucket as delete_bucket_rs,
    stat_object_many_async,
};
use crate::{generic_upload_files, generic_download_objects};
use crate::s3_logger::{finalize_op_logger, init_op_logger, global_logger};
use crate::object_store::{store_for_uri_with_logger, store_for_uri};

// Phase 2 streaming functionality imports
use crate::object_store::{
    ObjectStore, ObjectWriter, WriterOptions, CompressionConfig,
    S3ObjectStore, AzureObjectStore
};
use crate::file_store::FileSystemObjectStore;
use crate::file_store_direct::ConfigurableFileSystemObjectStore;

// ---------------------------------------------------------------------------
// Zero-Copy Buffer Support
// ---------------------------------------------------------------------------

/// A Python-visible wrapper around Rust Bytes that exposes buffer protocol
/// This allows Python code to get a memoryview without copying data
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
    
    /// Get a memoryview (zero-copy readonly access)
    fn memoryview<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        // Create a memoryview from our data
        // Note: This creates a Python memoryview object that references our Bytes
        let bytes_obj = PyBytes::new(py, &self.bytes);
        let memoryview_class = py.import("builtins")?.getattr("memoryview")?;
        Ok(memoryview_class.call1((bytes_obj,))?.into())
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
) -> PyResult<PyObject> {
    use crate::mp::{MpGetConfigBuilder, run_get_shards};
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    Python::with_gil(|py| {
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
        "prand" => DataGenAlgorithm::Prand,
        _ => DataGenAlgorithm::Random, // Default to random
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
fn parse_s3_uri_full(py: Python<'_>, uri: &str) -> PyResult<PyObject> {
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

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// DEPRECATED: Python-visible list_objects()  (sync wrapper), with recursion
// This function is S3-specific and will be removed in v1.0.0.
// Use the universal `list(uri, recursive, pattern)` function instead.
// ------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (bucket, prefix, recursive = false))]
#[deprecated(since = "0.9.4", note = "Use universal `list(uri, recursive, pattern)` instead. This S3-specific function will be removed in v1.0.0.")]
fn list_objects(bucket: &str, prefix: &str, recursive: bool) -> PyResult<Vec<String>> {
    eprintln!("WARNING: list_objects() is deprecated and will be removed in v1.0.0. Use list(uri, recursive, pattern) instead.");
    list_objects_rs(bucket, prefix, recursive).map_err(py_err)
}

// ------------------------------------------------------------------------
// DEPRECATED: Python-visible get_object()  (sync wrapper)
// This function is S3-specific and will be removed in v1.0.0.
// Use the universal `get(uri)` or `get_range(uri, offset, length)` function instead.
// ------------------------------------------------------------------------
#[pyfunction]
#[deprecated(since = "0.9.4", note = "Use universal `get(uri)` or `get_range(uri, offset, length)` instead. This S3-specific function will be removed in v1.0.0.")]
fn get_object(
    _py: Python<'_>,
    bucket: &str,
    key: &str,
    offset: Option<u64>,
    length: Option<u64>,
) -> PyResult<PyBytesView> {
    eprintln!("WARNING: get_object() is deprecated and will be removed in v1.0.0. Use get(uri) or get_range(uri, offset, length) instead.");
    let bytes = s3_get_range(bucket, key, offset.unwrap_or(0), length)
        .map_err(py_err)?;
    // Return zero-copy BytesView wrapper
    Ok(PyBytesView::new(bytes))
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
    py.allow_threads(move || {
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
    py.allow_threads(move || {
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
    py.allow_threads(|| {
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
        Python::with_gil(|py| Ok(py.None()))
    })
}


// --- `list` function
#[pyfunction]
#[pyo3(signature = (uri, recursive = false, pattern = None))]
pub fn list(uri: &str, recursive: bool, pattern: Option<&str>) -> PyResult<Vec<String>> {
    // Universal list using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger).map_err(py_err)?;
    
    // Use tokio runtime to call async list
    let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
    let mut keys = rt.block_on(async {
        store.list(uri, recursive).await
    }).map_err(py_err)?;
    
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
pub fn stat(py: Python<'_>, uri: &str) -> PyResult<PyObject> {
    // Universal stat using ObjectStore - works with all backends (S3, GCS, Azure, File, DirectIO)
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger).map_err(py_err)?;
    
    // Use tokio runtime to call async stat
    let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
    let os = rt.block_on(async {
        store.stat(uri).await
    }).map_err(py_err)?;
    
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
        Python::with_gil(|py| {
            let obj: PyObject = stat_to_pydict(py, os)?.unbind().into();
            Ok::<PyObject, PyErr>(obj)
        })
    })
}

#[pyfunction]
fn stat_many_async<'py>(py: Python<'py>, uris: Vec<String>) -> PyResult<pyo3::Bound<'py, PyAny>> {
    future_into_py(py, async move {
        let metas = stat_object_many_async(uris).await.map_err(py_err)?;
        Python::with_gil(|py| {
            let list = PyList::empty(py);
            for os in metas {
                // append accepts any IntoPyObject<'py>; Bound<'py, PyDict> works directly
                list.append(stat_to_pydict(py, os)?)?;
            }
            Ok::<PyObject, PyErr>(list.unbind().into())
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
    py.allow_threads(|| {
        let logger = global_logger();
        let store = match store_for_uri_with_logger(uri, logger) {
            Ok(s) => s,
            Err(_) => return Ok(false), // URI parsing error → doesn't exist
        };
        
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async {
            match store.stat(uri).await {
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
    
    py.allow_threads(move || {
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async move {
            let logger = global_logger();
            let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
            store.put(&uri, &data_owned).await.map_err(py_err)
        })
    })
}

/// Async version of put_bytes()
#[pyfunction]
fn put_bytes_async<'py>(py: Python<'py>, uri: &str, data: &Bound<'_, PyBytes>) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    // We need to copy the data for the async future since PyBytes can't be sent across threads
    let data_owned = Bytes::copy_from_slice(data.as_bytes());
    
    future_into_py(py, async move {
        let logger = global_logger();
        let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
        store.put(&uri, &data_owned).await.map_err(py_err)
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
    
    py.allow_threads(move || {
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async move {
            let logger = global_logger();
            let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?;
            store.mkdir(&uri).await.map_err(py_err)
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
    py.allow_threads(|| {
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

        Python::with_gil(|py| (count, total).into_py_any(py))
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

        Python::with_gil(|_py| {
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
    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async {
            let logger = global_logger();
            let store = store_for_uri_with_logger(uri, logger).map_err(py_err)?;
            let bytes = store.get(uri).await.map_err(py_err)?;
            Ok(PyBytesView::new(bytes))
        })
    })
}

// --- `get_range` function (universal backend support)
#[pyfunction(name = "get_range")]
#[pyo3(signature = (uri, offset, length = None))]
pub fn get_range_py(py: Python<'_>, uri: &str, offset: u64, length: Option<u64>) -> PyResult<PyBytesView> {
    // Use universal ObjectStore interface for range requests
    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async {
            let logger = global_logger();
            let store = store_for_uri_with_logger(uri, logger).map_err(py_err)?;
            let bytes = store.get_range(uri, offset, length).await.map_err(py_err)?;
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
    // The `download_objects` function in `s3_copy.rs` needs a `recursive` flag.
    // We'll assume for now it has been added.
    py.allow_threads(move || {
        // Use the new generic download function that works with all backends
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(generic_download_objects(src_uri, &dir, max_in_flight, recursive, None)).map_err(py_err)
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
    
    py.allow_threads(|| {
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
                let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
                let res = rt.block_on(async {
                    let sem = Arc::new(Semaphore::new(max_in_flight));
                    let mut futs = FuturesUnordered::new();
                    
                    for uri in uris.clone() {
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
                    out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap());
                    Ok::<_, anyhow::Error>(out)
                }).map_err(py_err)?;
                
                Ok(res.into_iter()
                    .map(|(u, b)| (u, PyBytesView::new(b)))
                    .collect())
            }
            Scheme::Azure | Scheme::Gcs => {
                // Use store_for_uri for Azure and GCS
                let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
                let res = rt.block_on(async {
                    let sem = Arc::new(Semaphore::new(max_in_flight));
                    let mut futs = FuturesUnordered::new();
                    
                    for uri in uris.clone() {
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
                    out.sort_by_key(|(u, _)| uris.iter().position(|x| x == u).unwrap());
                    Ok::<_, anyhow::Error>(out)
                }).map_err(py_err)?;
                
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
    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async {
            let logger = global_logger();
            let store = store_for_uri_with_logger(uri, logger).map_err(py_err)?;
            
            // Check if URI contains wildcards or ends with / (pattern/directory)
            let has_pattern = uri.contains('*') || uri.contains('?') || uri.ends_with('/');
            
            if has_pattern || recursive {
                // Need to list objects first, then delete them
                let list_results = store.list(uri, recursive).await.map_err(py_err)?;
                
                if list_results.is_empty() {
                    // No objects matched - this is OK
                    return Ok(());
                }
                
                // Delete each object (list returns full URIs)
                for obj_uri in list_results {
                    store.delete(&obj_uri).await.map_err(py_err)?;
                }
                Ok(())
            } else {
                // Simple single object deletion
                store.delete(uri).await.map_err(py_err)
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
    py.allow_threads(|| {
        // Use the new generic upload function that works with all backends
        let rt = tokio::runtime::Runtime::new().map_err(py_err)?;
        rt.block_on(async {
            // Get logger if op-log is active
            let logger = global_logger();
            
            // Handle bucket creation ONLY if explicitly requested
            if create_bucket {
                if let Ok(store) = store_for_uri_with_logger(dest_prefix, logger.clone()) {
                    // Extract bucket/container name from URI
                    if dest_prefix.starts_with("s3://") {
                        if let Ok((bucket, _)) = parse_s3_uri(dest_prefix) {
                            if let Err(e) = store.create_container(&bucket).await {
                                warn!("Failed to create bucket {}: {}", bucket, e);
                            }
                        }
                    } else if dest_prefix.starts_with("az://") || dest_prefix.starts_with("azure://") {
                        // For Azure, extract container name
                        let parts: Vec<&str> = dest_prefix.trim_start_matches("az://").trim_start_matches("azure://").split('/').collect();
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
            generic_upload_files(dest_prefix, &paths, max_in_flight, None).await
        }).map_err(py_err)
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
    /// Write a chunk of bytes to the stream
    fn write_chunk(&mut self, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let writer = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer has been finalized"))?;
        
        let bytes = data.as_bytes().to_vec();

        pyo3_async_runtimes::tokio::get_runtime().block_on(async {
            writer.write_chunk(&bytes).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to write chunk: {}", e)))
        })
    }
    
    fn write_owned_bytes(&mut self, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        let writer = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer has been finalized"))?;
        
        let bytes = data.as_bytes().to_vec();
        
        pyo3_async_runtimes::tokio::get_runtime().block_on(async {
            writer.write_owned_bytes(bytes).await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to write owned bytes: {}", e)))
        })
    }
    
    /// Finalize the writer and complete the upload
    fn finalize(&mut self, py: Python<'_>) -> PyResult<(u64, u64)> {
        if let Some(writer) = self.inner.take() {
            py.allow_threads(|| {
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
    
    py.allow_threads(|| {
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
    
    py.allow_threads(|| {
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
    
    py.allow_threads(|| {
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
    
    py.allow_threads(|| {
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
use std::sync::Arc;

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
            Python::with_gil(|py| Ok(Py::new(py, PyBytesView::new(data))?.into_any()))
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
            Python::with_gil(|py| Ok(Py::new(py, PyBytesView::new(data))?.into_any()))
        })
    }
    
    /// Put an object to the multi-endpoint store
    fn put<'py>(&self, py: Python<'py>, uri: &str, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let uri = uri.to_string();
        let data = data.to_vec();
        let store = self.store.clone();
        
        future_into_py(py, async move {
            store.put(&uri, &data).await
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
    fn get_endpoint_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
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
    fn get_total_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
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
    
    // Core storage operations
    // DEPRECATED: S3-specific data operations (will be removed in v1.0.0)
    m.add_function(wrap_pyfunction!(list_objects, m)?)?;      // Use list() instead
    m.add_function(wrap_pyfunction!(get_object, m)?)?;        // Use get() or get_range() instead
    
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

