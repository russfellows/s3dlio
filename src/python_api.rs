// src/python_api.rs
// 
// Copyright 2025
// Signal65 / Futurum Group.
//

//
// PyO3 bindings — sync *and* async wrappers around the Rust S3 helpers.
// * Sync paths call `py.allow_threads`, releasing the GIL while Rust blocks.
// * Async paths return real `asyncio` futures via pyo3‑async‑runtimes + Tokio.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyBytes, PyDict, PyDictMethods};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::conversion::IntoPyObjectExt;

use pyo3_async_runtimes::tokio::future_into_py; // only for bridge → Python
use tokio::{
//    spawn,
    task,
    sync::{mpsc, Mutex},
};

use numpy::ndarray as nd_np;

use std::{
    io::Cursor,
    path::PathBuf,
    sync::Arc,
};

use log::LevelFilter;
use env_logger;

// ---------------------------------------------------------------------------
// Project crates
// ---------------------------------------------------------------------------
use crate::config::ObjectType;
use crate::s3_utils::{
    create_bucket, delete_objects, get_object_uri, get_objects_parallel, list_objects,
    parse_s3_uri, put_objects_with_random_data_and_type, stat_object_uri, DEFAULT_OBJECT_SIZE,
};
use crate::s3_copy::{download_objects, upload_files};
use crate::s3_logger::{finalize_op_logger, init_op_logger};

use crate::data_loader::{
    DataLoader,
    dataset::{Dataset, DatasetError},
    options::LoaderOptions,
};

use futures_util::StreamExt;

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

fn py_err<E: std::fmt::Display>(e: E) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

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
        if should_create_bucket { let _ = create_bucket(&bucket); }
        put_objects_with_random_data_and_type(&uris, sz, jobs, obj, dedup_factor, compress_factor)
            .map_err(py_err)
    })
}

// Old, the python visible name has an _py at the end
//#[pyfunction(name = "put_async_py")]
// New, now it does not.
#[pyfunction(name = "put_async")]
#[pyo3(signature = (
    prefix, num, template,
    max_in_flight = 64, size = None,
    should_create_bucket = false, object_type = "zeros",
    dedup_factor = 1, compress_factor = 1
))]
// Old, we knew this was a crate, and now we don't ??? 
//pub fn put_async_py<'p>(
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
            if should_create_bucket { let _ = create_bucket(&bucket); }
            put_objects_with_random_data_and_type(&uris, sz, jobs, obj, dedup_factor, compress_factor)
                .map_err(py_err)
        })
        .await
        .map_err(py_err)??;
        Python::with_gil(|py| Ok(py.None()))
    })
}

#[pyfunction]
pub fn list(uri: &str) -> PyResult<Vec<String>> {
    let (bucket, prefix) = parse_s3_uri(uri).map_err(py_err)?;
    list_objects(&bucket, &prefix).map_err(py_err)
}

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

// Old, had a _py
//#[pyfunction(name = "get_many_async_py")]
#[pyfunction(name = "get_many_async")]
#[pyo3(signature = (uris, max_in_flight = 64))]
//pub fn get_many_async_py<'p>(
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
    let bytes = get_object_uri(uri).map_err(py_err)?;
    Ok(PyBytes::new(py, &bytes[..]).unbind())
}

#[pyfunction]
#[pyo3(signature = (src_uri, dest_dir, max_in_flight = 64))]
pub fn download(
    py: Python<'_>,
    src_uri: &str,
    dest_dir: &str,
    max_in_flight: usize,
) -> PyResult<()> {
    let dir = PathBuf::from(dest_dir);
    py.allow_threads(|| download_objects(src_uri, &dir, max_in_flight).map_err(py_err))
}

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

#[pyfunction]
pub fn delete(uri: &str) -> PyResult<()> {
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    let targets = if key.ends_with('/') || key.is_empty() {
        list_objects(&bucket, &key).map_err(py_err)?
    } else {
        vec![key]
    };
    delete_objects(&bucket, &targets).map_err(py_err)
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
// NPZ reader
// ---------------------------------------------------------------------------
use numpy::PyArrayDyn;

#[pyfunction]
#[pyo3(signature = (uri, array_name = None))]
pub fn read_npz(py: Python<'_>, uri: &str, array_name: Option<&str>) -> PyResult<Py<PyAny>> {
    let bytes = get_object_uri(uri).map_err(py_err)?;
    let mut npz = ndarray_npy::NpzReader::new(Cursor::new(bytes)).map_err(py_err)?;
    let name = array_name
        .map(|s| s.to_owned())
        .or_else(|| npz.names().ok().and_then(|mut v| v.pop()))
        .ok_or_else(|| PyRuntimeError::new_err("NPZ file is empty"))?;
    let arr: ndarray::ArrayD<f32> = npz.by_name(&name).map_err(py_err)?;
    let shape = arr.shape().to_vec();
    let data = arr.into_raw_vec();
    let arr_np = nd_np::ArrayD::from_shape_vec(nd_np::IxDyn(&shape), data).map_err(py_err)?;
    let py_arr = PyArrayDyn::<f32>::from_owned_array(py, arr_np);
    py_arr.into_py_any(py)
}

// ---------------------------------------------------------------------------
// DataLoader bindings
// ---------------------------------------------------------------------------
#[pyclass]
#[derive(Clone)]
pub struct PyVecDataset { inner: Vec<i32> }

#[pymethods]
impl PyVecDataset {
    #[new]
    fn new(obj: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self { inner: obj.extract()? })
    }
}

#[async_trait::async_trait]
impl Dataset for PyVecDataset {
    type Item = i32;
    fn len(&self) -> Option<usize> { Some(self.inner.len()) }
    async fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        self.inner.get(idx).copied().ok_or(DatasetError::IndexOutOfRange(idx))
    }
}


/// Convert an optional Python dict of options into LoaderOptions.
fn opts_from_dict(d: Option<Bound<'_, PyDict>>) -> LoaderOptions {
    if let Some(d) = d {
        // Helper closures that cope with the new Result‑of‑Option signature.
        let g_usize = |k: &str, def: usize| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<usize>().unwrap_or(def),
            _             => def,
        };
        let g_bool = |k: &str, def: bool| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<bool>().unwrap_or(def),
            _             => def,
        };
        let g_u64 = |k: &str, def: u64| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<u64>().unwrap_or(def),
            _             => def,
        };

        LoaderOptions {
            batch_size:  g_usize("batch_size", 32),
            drop_last:   g_bool ("drop_last",  false),
            shuffle:     g_bool ("shuffle",    false),
            seed:        g_u64  ("seed",       0),
            num_workers: g_usize("num_workers", 0),
            prefetch:    g_usize("prefetch",   0),
            auto_tune:   g_bool ("auto_tune",  false),
        }
    } else {
        LoaderOptions::default()
    }
}


#[pyclass]
pub struct PyAsyncDataLoader { dataset: PyVecDataset, opts: LoaderOptions }

/// This is the Py Wrapper for the Async Data Loader
#[pymethods]
impl PyAsyncDataLoader {
    #[new]
    fn new(dataset: PyVecDataset, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self { dataset, opts: opts_from_dict(opts) })
    }

    fn __aiter__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Py<PyAsyncDataLoaderIter>> {
        PyAsyncDataLoaderIter::spawn_stream(py, slf.dataset.clone(), slf.opts.clone())
    }
}

#[pyclass]
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
        pyo3_async_runtimes::tokio::get_runtime()
            .spawn(async move {
            let loader = DataLoader::new(dataset, opts);
            let mut stream = loader.stream();
            while let Some(batch) = stream.next().await {
                if tx.send(batch).await.is_err() { break; }
            }
        });

        Py::new(py, Self { rx: Arc::new(Mutex::new(rx)) })
    }
}

#[pymethods]
impl PyAsyncDataLoaderIter {
    fn __anext__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rx = slf.rx.clone();
        future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                Some(Ok(batch)) => Python::with_gil(|py| batch.into_py_any(py)),
                Some(Err(e))    => Err(PyRuntimeError::new_err(format!("{:?}", e))),
                None            => Err(PyStopAsyncIteration::new_err("end of loader")),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Sub‑module exposed at crate‑root (`lib.rs` re‑exports it as needed)
// ---------------------------------------------------------------------------
#[pymodule]
pub fn _pymod(m: &Bound<PyModule>) -> PyResult<()> {
    // S3 helpers
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(stat, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;

    // logging
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(init_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_op_log, m)?)?;

    // dataloader bindings
    m.add_class::<PyVecDataset>()?;
    m.add_class::<PyAsyncDataLoader>()?;
    m.add_class::<PyAsyncDataLoaderIter>()?;
    Ok(())
}

