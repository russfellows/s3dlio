// src/python_api.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
//! PyO3 bindings — sync *and* async wrappers around the Rust S3 helpers.
//!
//! * **Sync** paths call `py.allow_threads` so the GIL is released while Rust blocks.
//! * **Async** paths return `asyncio` futures via **pyo3-async-runtimes + Tokio**.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
//use pyo3::types::{PyBytes, PyAny};
use pyo3::types::{PyBytes, PyAny, PyDict};
use pyo3::exceptions::PyRuntimeError;
use pyo3::conversion::IntoPyObjectExt;
use numpy::ndarray as nd_np;
use pyo3_async_runtimes::tokio as aio;
use tokio::task;

use std::path::PathBuf;
use log::LevelFilter;
use env_logger;

use crate::config::ObjectType;

use crate::s3_utils::{
    create_bucket, delete_objects, get_object_uri, get_objects_parallel, list_objects,
    stat_object_uri, parse_s3_uri, put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE,
};

use crate::s3_copy::{upload_files, download_objects};
use crate::s3_logger::{init_op_logger, finalize_op_logger};

// ---------------------------------------------------------------------------
// Logging init (sync) --------------------------------------------------------
// ---------------------------------------------------------------------------

/// Initialise Rust-side logging once.
/// Safe to call multiple times; subsequent calls are ignored.
#[pyfunction]
pub fn init_logging(level: &str) -> PyResult<()> {
    let filter = match level.to_lowercase().as_str() {
        "trace" => LevelFilter::Trace,
        "debug" => LevelFilter::Debug,
        "warn"  => LevelFilter::Warn,
        "error" => LevelFilter::Error,
        _       => LevelFilter::Info,
    };

    let _ = env_logger::builder()
        .filter_level(filter)
        .is_test(false)
        .try_init();

    Ok(())
}

// Error helper ----------------------------------------------------------------
fn py_err<E: std::fmt::Display>(e: E) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

// ---------------------------------------------------------------------------
// OP-LOG INITIALISER / FINALISER (sync)
// ---------------------------------------------------------------------------
/// Start operation logging to a warp-replay compatible `.tsv.zst` file.
/// Must be called **before** the S3 calls you want logged.
#[pyfunction]
pub fn init_op_log(path: &str) -> PyResult<()> {
    init_op_logger(path).map_err(py_err)
}

/// Flush and close the op-log. Safe to call multiple times.
#[pyfunction]
pub fn finalize_op_log() -> PyResult<()> {
    finalize_op_logger();
    Ok(())
}


// ---------------------------------------------------------------------------
// Helper ----------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// PUT (sync) -----------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (
    prefix,         // destination prefix (s3://bucket/prefix/)
    num,            // how many objects
    template,       // file name template e.g. "object_{}_of_{}.dat"
    max_in_flight = 64,
    size = None,
    should_create_bucket = false,
    object_type = "zeros",
    dedup_factor = 1,
    compress_factor = 1,
))]
pub fn put(
    py: Python<'_>,
    prefix: &str,
    num: usize,
    template: &str,
    max_in_flight: usize,
    size: Option<usize>,
    should_create_bucket: bool,
    object_type: &str,
    dedup_factor: usize,
    compress_factor: usize,
) -> PyResult<()> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);
    py.allow_threads(|| {
        if should_create_bucket { let _ = create_bucket(&bucket); }
        put_objects_with_random_data_and_type(&uris, sz, jobs, obj, dedup_factor, compress_factor)
            .map_err(py_err)
    })
}

// ---------------------------------------------------------------------------
// PUT (async) ----------------------------------------------------------------
#[pyfunction(name = "put_async_py")]
#[pyo3(signature = (
    prefix,         // destination prefix (s3://bucket/prefix/)
    num,            // how many objects
    template,       // file name template e.g. "object_{}_of_{}.dat"
    max_in_flight = 64,
    size = None,
    should_create_bucket = false,
    object_type = "zeros",
    dedup_factor = 1,
    compress_factor = 1,
))]
pub(crate) fn put_async_py<'p>(
    py: Python<'p>,
    prefix: &'p str,
    num: usize,
    template: &'p str,
    max_in_flight: usize,
    size: Option<usize>,
    should_create_bucket: bool,
    object_type: &'p str,
    dedup_factor: usize,
    compress_factor: usize,
) -> PyResult<Bound<'p, PyAny>> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);

    aio::future_into_py(py, async move {
        // Run the blocking S3 workload on a dedicated thread
        tokio::task::spawn_blocking(move || {
            if should_create_bucket { let _ = create_bucket(&bucket); }
            put_objects_with_random_data_and_type(&uris, sz, jobs, obj, dedup_factor, compress_factor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))??;

        // Resolve to Python None on success
        Python::with_gil(|py| Ok(py.None()))
    })
}

// ---------------------------------------------------------------------------
// LIST (sync) -----------------------------------------------------------------
#[pyfunction]
pub fn list(uri: &str) -> PyResult<Vec<String>> {
    let (bucket, prefix) = parse_s3_uri(uri).map_err(py_err)?;
    list_objects(&bucket, &prefix).map_err(py_err)
}

// ---------------------------------------------------------------------------
// STAT (sync) ----------------------------------------------------------------
#[pyfunction]
pub fn stat(py: Python, uri: &str) -> PyResult<PyObject> {
    let os = stat_object_uri(uri).map_err(py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("size", os.size)?;
    dict.set_item("last_modified", os.last_modified)?;
    dict.set_item("etag", os.e_tag)?;
    dict.set_item("content_type", os.content_type)?;
    dict.set_item("content_language", os.content_language)?;
    dict.set_item("storage_class", os.storage_class)?;
    dict.set_item("metadata", os.metadata)?;
    dict.set_item("version_id", os.version_id)?;
    dict.set_item("replication_status", os.replication_status)?;
    dict.set_item("server_side_encryption", os.server_side_encryption)?;
    dict.set_item("ssekms_key_id", os.ssekms_key_id)?;
    dict.into_py_any(py)
}

// ---------------------------------------------------------------------------
// GET‑many **stats** (sync)  — no payload copies
// ---------------------------------------------------------------------------
#[pyfunction(name = "get_many_stats")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub fn get_many_stats(
    py: Python<'_>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<(usize, usize)> {
    py.allow_threads(|| {
        let res = get_objects_parallel(&uris, max_in_flight).map_err(py_err)?;
        let n = res.len();
        let total: usize = res.iter().map(|(_, b)| b.len()).sum();
        Ok((n, total))
    })
}

// ---------------------------------------------------------------------------
// GET many (async) ------------------------------------------------------------
#[pyfunction(name = "get_many_stats_async")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub(crate) fn get_many_stats_async<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Bound<'p, PyAny>> {
    // New async
    aio::future_into_py(py, async move {
        // Run blocking listing work on a dedicated thread
        let (n, total) = tokio::task::spawn_blocking(move || {
            let res = get_objects_parallel(&uris, max_in_flight)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let total = res.iter().map(|(_, b)| b.len()).sum::<usize>();
            Ok::<(usize, usize), PyErr>((res.len(), total))
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))??;

        // Return a Python tuple
        Python::with_gil(|py| (n, total).into_py_any(py))
    })
}

// ---------------------------------------------------------------------------
// GET many (async) — return (uri, bytes) -------------------------------------
#[pyfunction(name = "get_many_async_py")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub(crate) fn get_many_async_py<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<Bound<'p, PyAny>> {
    // New async 
    aio::future_into_py(py, async move {
        // off‑load the whole download into a blocking thread
        let pairs: Vec<(String, Vec<u8>)> = task::spawn_blocking(move || {
            get_objects_parallel(&uris, max_in_flight)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))??;
        // now safely convert bytes → PyBytes under the GIL
        Python::with_gil(|py| {
            let pyres: Vec<(String, Py<PyBytes>)> = pairs.into_iter()
                .map(|(u, b)| (u, PyBytes::new(py, &b).unbind()))
                .collect::<Vec<_>>();
            Ok(pyres)
        })
    })
}

// ---------------------------------------------------------------------------
// GET (sync) -------------------------------------------------------------
#[pyfunction]
pub fn get(py: Python<'_>, uri: &str) -> PyResult<Py<PyBytes>> {
    let bytes = get_object_uri(uri).map_err(py_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

// ---------------------------------------------------------------------------
// DOWNLOAD (sync) -----------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (
    src_uri,        // single key, prefix, or glob
    dest_dir,       // local directory
    max_in_flight = 64
))]
pub fn download(
    py: Python<'_>,
    src_uri: &str,
    dest_dir: &str,
    max_in_flight: usize,
) -> PyResult<()> {
    let dir = PathBuf::from(dest_dir);
    py.allow_threads(|| {
        download_objects(src_uri, &dir, max_in_flight)
            .map_err(py_err)
    })
}

// ---------------------------------------------------------------------------
// GET‑many (sync) ---------------------------------------------------------
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
            Ok(res.into_iter().map(|(u,b)| {
                let by: Py<PyBytes> = PyBytes::new(py,&b).unbind();
                (u, by)
            }).collect())
        })
    })
}

// ---------------------------------------------------------------------------
// DELETE (sync) --------------------------------------------------------------
#[pyfunction]
pub fn delete(uri: &str) -> PyResult<()> {
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    let list = if key.ends_with('/') || key.is_empty() {
        // delete everything under a prefix
        list_objects(&bucket, &key).map_err(py_err)?
    } else {
        // delete a single object
        vec![key]
    };
    delete_objects(&bucket, &list).map_err(py_err)
}

/*
 * Poorly modified - remove ?
 *
pub fn delete(uri: &str) -> PyResult<usize> {
    let (bucket, prefix) = parse_s3_uri(uri).map_err(py_err)?;
    delete_objects(&bucket, &prefix).map_err(py_err)
}
*/


// ---------------------------------------------------------------------------
// UPLOAD (sync) --------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (
    src_patterns,     // list of local paths or glob patterns
    dest_prefix,      // S3 URI (must end with `/` or empty)
    max_in_flight = 32,
    create_bucket = false
))]
pub fn upload(
    py: Python<'_>,
    src_patterns: Vec<String>,
    dest_prefix: &str,
    max_in_flight: usize,
    create_bucket: bool,
) -> PyResult<()> {
    // expand to Vec<PathBuf>
    let paths: Vec<PathBuf> = src_patterns.into_iter()
        .map(PathBuf::from)
        .collect();
    py.allow_threads(|| {
        upload_files(dest_prefix, &paths, max_in_flight, create_bucket)
            .map_err(py_err)
    })
}

// ---------------------------------------------------------------------------
// READ NPZ (sync) -> numpy array --------------------------------------------
use numpy::PyArrayDyn;
use std::io::Cursor;

#[pyfunction]
#[pyo3(signature = (uri, array_name = None))]
pub fn read_npz(py: Python<'_>, uri: &str, array_name: Option<&str>) -> PyResult<Py<PyAny>> {
    let bytes = get_object_uri(uri).map_err(py_err)?;
    let cursor = Cursor::new(bytes);
    let mut npz = ndarray_npy::NpzReader::new(cursor).map_err(py_err)?;
    let name = array_name
        .map(|s| s.to_owned())
        .or_else(|| npz.names().ok().and_then(|mut v| v.pop()))
        .ok_or_else(|| PyRuntimeError::new_err("NPZ file is empty"))?;
    let arr: ndarray::ArrayD<f32> = npz.by_name(&name).map_err(py_err)?;
    let shape: Vec<usize> = arr.shape().to_vec();
    let data = arr.into_raw_vec();
    let arr_np = nd_np::ArrayD::from_shape_vec(nd_np::IxDyn(&shape), data).map_err(py_err)?;
    let py_arr = PyArrayDyn::<f32>::from_owned_array(py, arr_np);
    py_arr.into_py_any(py)
}

// ---------------------------------------------------------------------------
// (Sub)module exposed to lib.rs -----------------------------------------
/// This allows re-exporting these functions from the crate-level `#[pymodule]`.
#[pymodule]
pub fn _pymod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(init_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_op_log, m)?)?;
    Ok(())
}

