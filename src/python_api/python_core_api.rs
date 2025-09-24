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

use tokio::task;

use std::path::PathBuf;

use log::LevelFilter;
use env_logger;

// Project crates
use crate::config::ObjectType;
use crate::s3_utils::{
    delete_objects, get_object_uri, get_objects_parallel,
    list_objects as list_objects_rs, get_range,
    parse_s3_uri, put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE,
    create_bucket as create_bucket_rs, delete_bucket as delete_bucket_rs,
    stat_object_uri,
    stat_object_uri_async,
    stat_object_many_async,
};
use crate::s3_copy::{download_objects, upload_files};
use crate::s3_logger::{finalize_op_logger, init_op_logger};

// Phase 2 streaming functionality imports
use crate::object_store::{
    ObjectStore, ObjectWriter, WriterOptions, CompressionConfig,
    S3ObjectStore, AzureObjectStore
};
use crate::file_store::FileSystemObjectStore;
use crate::file_store_direct::ConfigurableFileSystemObjectStore;

// ---------------------------------------------------------------------------
// Core API functions
// ---------------------------------------------------------------------------

/// Convert various errors to PyErr
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
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "warn"  => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _       => LevelFilter::Info,
    };
    let _ = env_logger::builder().filter_level(filter).is_test(false).try_init();
    Ok(())
}

#[pyfunction]
pub fn init_op_log(path: &str) -> PyResult<()> { init_op_logger(path).map_err(py_err) }

#[pyfunction]
pub fn finalize_op_log() -> PyResult<()> { finalize_op_logger(); Ok(()) }

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
fn build_uri_list(prefix: &str, template: &str, num: usize) -> PyResult<(String, Vec<String>)> {
    let (bucket, mut key_prefix) = parse_s3_uri(prefix).map_err(py_err)?;
    if !key_prefix.ends_with('/') { key_prefix.push('/'); }
    let uris = (0..num).map(|i| {
        let name = template
            .replacen("{}", &i.to_string(), 1)
            .replacen("{}", &num.to_string(), 1);
        format!("s3://{}/{}{}", bucket, key_prefix, name)
    }).collect();
    Ok((bucket, uris))
}
fn str_to_obj(s: &str) -> ObjectType { ObjectType::from(s) }

// ------------------------------------------------------------------------
// NEW: Python-visible list_objects()  (sync wrapper), with recursion
// ------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (bucket, prefix, recursive = false))]
fn list_objects(bucket: &str, prefix: &str, recursive: bool) -> PyResult<Vec<String>> {
    list_objects_rs(bucket, prefix, recursive).map_err(py_err)
}

// ------------------------------------------------------------------------
// NEW: Python-visible get_object()  (sync wrapper)
// ------------------------------------------------------------------------
#[pyfunction]
fn get_object(
    py: Python<'_>,
    bucket: &str,
    key: &str,
    offset: Option<u64>,
    length: Option<u64>,
) -> PyResult<Py<PyBytes>> {
    let bytes = get_range(bucket, key, offset.unwrap_or(0), length)
        .map_err(py_err)?;
    // Owned PyBytes until zero-copy lands
    Ok(PyBytes::new(py, &bytes[..]).unbind())
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
    prefix, num, template,
    max_in_flight = 64, size = None,
    should_create_bucket = false, object_type = "zeros",
    dedup_factor = 1, compress_factor = 1
))]
pub fn put(
    py: Python<'_>,
    prefix: &str, num: usize, template: &str,
    max_in_flight: usize, size: Option<usize>,
    should_create_bucket: bool, object_type: &str,
    dedup_factor: usize, compress_factor: usize,
) -> PyResult<()> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);
    py.allow_threads(|| {
        if should_create_bucket { let _ = create_bucket_rs(&bucket); }
        put_objects_with_random_data_and_type(&uris, sz, jobs, obj, dedup_factor, compress_factor)
            .map_err(py_err)
    })
}

#[pyfunction(name = "put_async")]
#[pyo3(signature = (
    prefix, num, template,
    max_in_flight = 64, size = None,
    should_create_bucket = false, object_type = "zeros",
    dedup_factor = 1, compress_factor = 1
))]
pub (crate) fn put_async_py<'p>(
    py: Python<'p>,
    prefix: &'p str, num: usize, template: &'p str,
    max_in_flight: usize, size: Option<usize>,
    should_create_bucket: bool, object_type: &'p str,
    dedup_factor: usize, compress_factor: usize,
) -> PyResult<Bound<'p, PyAny>> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);

    future_into_py(py, async move {
        task::spawn_blocking(move || {
            if should_create_bucket { let _ = create_bucket_rs(&bucket); }
            put_objects_with_random_data_and_type(&uris, sz, jobs, obj, dedup_factor, compress_factor)
                .map_err(py_err)
        })
        .await
        .map_err(py_err)??;
        Python::with_gil(|py| Ok(py.None()))
    })
}


// --- `list` function
#[pyfunction]
#[pyo3(signature = (uri, recursive = false))]
pub fn list(uri: &str, recursive: bool) -> PyResult<Vec<String>> {
    use crate::object_store::{infer_scheme, Scheme};
    
    match infer_scheme(uri) {
        Scheme::S3 => {
            let (bucket, mut path) = parse_s3_uri(uri).map_err(py_err)?;
            
            // If the path ends with a slash, automatically append a wildcard to search inside it.
            if path.ends_with('/') {
                path.push_str(".*");
            }
            
            crate::s3_utils::list_objects(&bucket, &path, recursive).map_err(py_err)
        }
        Scheme::File | Scheme::Direct => {
            // For file systems, we'll need to implement file listing
            // TODO: Implement file system listing
            Err(PyRuntimeError::new_err("File system listing not yet implemented"))
        }
        Scheme::Azure => {
            // TODO: Implement Azure listing
            Err(PyRuntimeError::new_err("Azure listing not yet implemented"))
        }
        Scheme::Unknown => {
            Err(PyRuntimeError::new_err(format!("Unsupported URI scheme: {}", uri)))
        }
    }
}


// New stat calls, almost the same ????
//
#[pyfunction]
pub fn stat(py: Python<'_>, uri: &str) -> PyResult<PyObject> {
    use crate::object_store::{infer_scheme, Scheme};
    
    match infer_scheme(uri) {
        Scheme::S3 => {
            let os = stat_object_uri(uri).map_err(py_err)?;
            let d = stat_to_pydict(py, os)?;
            Ok(d.unbind().into())
        }
        Scheme::File | Scheme::Direct => {
            // For file systems, get file metadata
            use std::fs;
            let path = uri.strip_prefix("file://").unwrap_or(uri);
            let metadata = fs::metadata(path).map_err(py_err)?;
            
            // Create a simple stat dict for files
            let dict = PyDict::new(py);
            dict.set_item("size", metadata.len())?;
            dict.set_item("key", path)?;
            if let Ok(modified) = metadata.modified() {
                if let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH) {
                    dict.set_item("last_modified", since_epoch.as_secs())?;
                }
            }
            Ok(dict.unbind().into())
        }
        Scheme::Azure => {
            // TODO: Implement Azure stat
            Err(PyRuntimeError::new_err("Azure stat not yet implemented"))
        }
        Scheme::Unknown => {
            Err(PyRuntimeError::new_err(format!("Unsupported URI scheme: {}", uri)))
        }
    }
}

#[pyfunction]
fn stat_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    future_into_py(py, async move {
        let os = stat_object_uri_async(&uri).await.map_err(py_err)?;
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
        let pairs: Vec<(String, Vec<u8>)> = task::spawn_blocking(move || {
            get_objects_parallel(&uris, max_in_flight).map_err(py_err)
        })
        .await
        .map_err(py_err)??;

        Python::with_gil(|py| {
            let out = pairs.into_iter()
                .map(|(u, b)| (u, PyBytes::new(py, &b[..]).unbind()))
                .collect::<Vec<_>>();
            Ok(out)
        })
    })
}

#[pyfunction]
pub fn get(py: Python<'_>, uri: &str) -> PyResult<Py<PyBytes>> {
    use crate::object_store::{infer_scheme, Scheme};
    
    match infer_scheme(uri) {
        Scheme::S3 => {
            let bytes = get_object_uri(uri).map_err(py_err)?;
            Ok(PyBytes::new(py, &bytes[..]).unbind())
        }
        Scheme::File | Scheme::Direct => {
            // For file systems, read the file directly
            use std::fs;
            let path = uri.strip_prefix("file://").unwrap_or(uri);
            let bytes = fs::read(path).map_err(py_err)?;
            Ok(PyBytes::new(py, &bytes[..]).unbind())
        }
        Scheme::Azure => {
            // TODO: Implement Azure get
            Err(PyRuntimeError::new_err("Azure get not yet implemented"))
        }
        Scheme::Unknown => {
            Err(PyRuntimeError::new_err(format!("Unsupported URI scheme: {}", uri)))
        }
    }
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
        // NOTE: This assumes `download_objects` will be updated to accept a `recursive` flag.
        // Let's pretend the signature is: `download_objects(src_uri, &dir, max_in_flight, recursive)`
        // For now, we call the existing function which is implicitly recursive.
        download_objects(src_uri, &dir, max_in_flight, recursive).map_err(py_err)
    })
}


// --- `get_many` function
#[pyfunction]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub fn get_many(
    py: Python<'_>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Vec<(String, Py<PyBytes>)>> {
    py.allow_threads(|| {
        let res = get_objects_parallel(&uris, max_in_flight).map_err(py_err)?;
        Python::with_gil(|py| {
            Ok(res
                .into_iter()
                .map(|(u, b)| (u, PyBytes::new(py, &b[..]).unbind()))
                .collect())
        })
    })
}

// --- `delete` function
#[pyfunction]
#[pyo3(signature = (uri, recursive = false))]
pub fn delete(uri: &str, recursive: bool) -> PyResult<()> {
    let (bucket, mut key_pattern) = parse_s3_uri(uri).map_err(py_err)?;

    /*
     * Old
     *
    // If the pattern isn't a glob/regex and not a prefix, it's a single key.
    // Otherwise, use list_objects to find all matching keys.
    let keys_to_delete = if !key_pattern.contains('*') && !key_pattern.contains('?') && !key_pattern.ends_with('/') {
        vec![key_pattern]
    } else {
        crate::s3_utils::list_objects(&bucket, &key_pattern, recursive).map_err(py_err)?
    };
    */

    // If the path ends with a slash, automatically append a wildcard to search inside it.
    if key_pattern.ends_with('/') {
        key_pattern.push_str(".*");
    }

    let keys_to_delete =
        crate::s3_utils::list_objects(&bucket, &key_pattern, recursive).map_err(py_err)?;

    if keys_to_delete.is_empty() {
        // Return Ok instead of an error if nothing matched, this is common practice
        return Ok(());
    }

    delete_objects(&bucket, &keys_to_delete).map_err(py_err)
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
    py.allow_threads(|| upload_files(dest_prefix, &paths, max_in_flight, create_bucket).map_err(py_err))
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
            let store = S3ObjectStore;
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
            let store = AzureObjectStore;
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
            let store = FileSystemObjectStore;
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
// Module registration function for core API
// ---------------------------------------------------------------------------
pub fn register_core_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Logging
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(init_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_op_log, m)?)?;
    
    // Core storage operations
    m.add_function(wrap_pyfunction!(list_objects, m)?)?;
    m.add_function(wrap_pyfunction!(get_object, m)?)?;
    m.add_function(wrap_pyfunction!(create_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(delete_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(stat, m)?)?;
    m.add_function(wrap_pyfunction!(stat_async, m)?)?;
    m.add_function(wrap_pyfunction!(stat_many_async, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(mp_get, m)?)?;
    
    // Phase 2 Streaming API classes and functions
    m.add_class::<PyWriterOptions>()?;
    m.add_class::<PyObjectWriter>()?;
    m.add_function(wrap_pyfunction!(create_s3_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_azure_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_filesystem_writer, m)?)?;
    m.add_function(wrap_pyfunction!(create_direct_filesystem_writer, m)?)?;
    
    Ok(())
}

