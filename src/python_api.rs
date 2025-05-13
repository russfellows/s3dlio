// src/python_api.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
//! PyO3 bindings — sync *and* async wrappers around the Rust S3 helpers.
//!
//! * **Sync** paths call `py.allow_threads` so the GIL is released while Rust blocks.
//! * **Async** paths return `asyncio` futures via **pyo3‑asyncio + Tokio**.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyAny};
use pyo3::exceptions::PyRuntimeError;
use pyo3_asyncio::tokio as aio;
use tokio::task;

use log::LevelFilter;
use env_logger;

use crate::s3_utils::{
    create_bucket, delete_objects, get_object_uri, get_objects_parallel, list_objects,
    parse_s3_uri, put_objects_with_random_data_and_type, ObjectType, DEFAULT_OBJECT_SIZE,
};

use crate::s3_copy::{upload_files, download_objects};

// ---------------------------------------------------------------------------
// Error helper ----------------------------------------------------------------
fn py_err<E: std::fmt::Display>(e: E) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

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
// PUT (sync) ------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (
    prefix,
    num = 1,
    template = "object_{}_of_{}.dat", // two slots: {}→i, {}→num 
    max_in_flight = 32,
    size = None,
    should_create_bucket = false,
    object_type = "RAW"
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
) -> PyResult<()> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);
    py.allow_threads(|| {
        if should_create_bucket { let _ = create_bucket(&bucket); }
        put_objects_with_random_data_and_type(&uris, sz, jobs, obj).map_err(py_err)
    })
}

// ---------------------------------------------------------------------------
// PUT (async) -----------------------------------------------------------------
#[pyfunction(name = "put_async")]
#[pyo3(signature = (
    prefix,
    num = 1,
    template = "object_{}_of_{}.dat",
    max_in_flight = 32,
    size = None,
    should_create_bucket = false,
    object_type = "RAW"
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
) -> PyResult<&'p PyAny> {
    let sz = size.unwrap_or(DEFAULT_OBJECT_SIZE);
    let (bucket, uris) = build_uri_list(prefix, template, num)?;
    let jobs = max_in_flight.min(num);
    let obj = str_to_obj(object_type);

    // New Async IO, offloaded to spawn blocking rather than our "block_on" inside a tokio worker.
    aio::future_into_py(py, async move {
        let res: Result<(), PyErr> = task::spawn_blocking(move || {
            if should_create_bucket { let _ = create_bucket(&bucket); }
            put_objects_with_random_data_and_type(&uris, sz, jobs, obj)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        res
    })

    /*
     * Old async IO
    aio::future_into_py(py, async move {
        if should_create_bucket { let _ = create_bucket(&bucket); }
        put_objects_with_random_data_and_type(&uris, sz, jobs, obj)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
    */
}

// ---------------------------------------------------------------------------
// LIST (sync) -----------------------------------------------------------------
#[pyfunction]
pub fn list(uri: &str) -> PyResult<Vec<String>> {
    let (bucket, prefix) = parse_s3_uri(uri).map_err(py_err)?;
    list_objects(&bucket, &prefix).map_err(py_err)
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
// GET many (sync) -------------------------------------------------------------
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
                let by: Py<PyBytes> = PyBytes::new(py,&b).into_py(py);
                (u, by)
            }).collect())
        })
    })
}

// ---------------------------------------------------------------------------
// GET (sync) -------------------------------------------------------------
#[pyfunction]
pub fn get(py: Python<'_>, uri: &str) -> PyResult<Py<PyBytes>> {
    let bytes = get_object_uri(uri).map_err(py_err)?;
    Ok(PyBytes::new(py, &bytes).into_py(py))
}

// ---------------------------------------------------------------------------
// GET‑many **stats** (async)
// ---------------------------------------------------------------------------
#[pyfunction(name = "get_many_stats_async")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub(crate) fn get_many_stats_async<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<&'p PyAny> {
    // New async
    aio::future_into_py(py, async move {
        // spawn the blocking S3 work
        let (n, total) = task::spawn_blocking(move || {
            let res = get_objects_parallel(&uris, max_in_flight)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let total = res.iter().map(|(_, b)| b.len()).sum::<usize>();

            //Ok((res.len(), total))
            Ok::<(usize, usize), PyErr>((res.len(), total))

        })
        .await
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))??;
        Ok((n, total))
    })
    
    /*
     * Old async
    aio::future_into_py(py, async move {
        let res = get_objects_parallel(&uris, max_in_flight)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let total: usize = res.iter().map(|(_, b)| b.len()).sum();
        Ok((res.len(), total))
    })
    */
}

// ---------------------------------------------------------------------------
// GET many (async) ------------------------------------------------------------
#[pyfunction(name = "get_many_async")]
#[pyo3(signature = (uris, max_in_flight = 64))]
pub(crate) fn get_many_async_py<'p>(
    py: Python<'p>,
    uris: Vec<String>,
    max_in_flight: usize,
) -> PyResult<&'p PyAny> {
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
                .map(|(u, b)| (u, PyBytes::new(py, &b).into_py(py)))
                .collect::<Vec<_>>();
            Ok(pyres)
        })
    })

    /*
     * Old async
    aio::future_into_py(py, async move {
        let res = get_objects_parallel(&uris, max_in_flight)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Python::with_gil(|py| {
            Ok(res.into_iter().map(|(u,b)| {
                let by: Py<PyBytes> = PyBytes::new(py,&b).into_py(py);
                (u, by)
            }).collect::<Vec<_>>())
        })
    })
    */
}

// ---------------------------------------------------------------------------
// DELETE (sync) ---------------------------------------------------------------
#[pyfunction]
pub fn delete(uri: &str) -> PyResult<()> {
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    let list = if key.ends_with('/') || key.is_empty() {
        list_objects(&bucket, &key).map_err(py_err)?
    } else { vec![key] };
    Python::with_gil(|py| py.allow_threads(|| delete_objects(&bucket, &list)).map_err(py_err))
}

// ---------------------------------------------------------------------------
// read_npz helper -------------------------------------------------------------
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
    Ok(PyArrayDyn::<f32>::from_owned_array(py, arr).into_py(py))
}

// ---------------------------------------------------------------------------
// LOGGING INITIALISER  (sync) -------------------------------------------------

/// Initialise Rust‑side logging *once*
///
/// Examples (Python):
/// ```python
/// import s3dlio
/// s3dlio.init_logging("info")   # or "debug", "warn"
/// ```
///
/// Subsequent calls are ignored, so it’s safe to invoke from multiple threads.
#[pyfunction]
#[pyo3(signature = (level = "info"))]   // default = "info"
pub fn init_logging(level: &str) -> PyResult<()> {
    let lvl = match level.to_ascii_lowercase().as_str() {
        "error" => LevelFilter::Error,
        "warn"  | "warning" => LevelFilter::Warn,
        "info"  => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        _ => LevelFilter::Info,
    };

    let _ = env_logger::builder()
        .filter_level(lvl)
        .try_init();         // ignores AlreadyInit
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal module exposed to lib.rs -----------------------------------------
#[pymodule]
pub fn _pymod(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    Ok(())
}

