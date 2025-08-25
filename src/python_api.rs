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
use pyo3::ffi;
use std::os::raw::c_char;
use pyo3::types::{PyAny, PyAnyMethods, 
    PyBytes, PyBytesMethods, 
    PyDict, PyDictMethods, 
    PyList, PyListMethods};
use pyo3::{PyObject, Bound};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::conversion::IntoPyObjectExt;
use pyo3_async_runtimes::tokio::future_into_py; // only for bridge → Python
                                                
use futures_util::StreamExt;
use tokio::{
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
    delete_objects, get_object_uri, get_objects_parallel,
    list_objects as list_objects_rs, get_range,
    parse_s3_uri, put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE,
    create_bucket as create_bucket_rs, delete_bucket as delete_bucket_rs,
    stat_object_uri,                    // ← sync
    stat_object_uri_async,              // ← async
    stat_object_many_async,             // ← async-many
};
use crate::s3_copy::{download_objects, upload_files};
use crate::s3_logger::{finalize_op_logger, init_op_logger};

use crate::data_loader::{
    DataLoader,
    dataset::{Dataset, DatasetError},
    options::{LoaderOptions, ReaderMode},
    s3_bytes::S3BytesDataset, //Only if required
};

use crate::multipart::{
    MultipartUploadConfig,
    MultipartUploadSink,
};



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

// New Code

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

#[pyclass]
#[derive(Clone)]
pub struct PyS3Dataset { inner: S3BytesDataset }

#[pymethods]
impl PyS3Dataset {
    /// NEW: optional opts (reader mode)
    #[new]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let o = opts_from_dict(opts);
        let ds = S3BytesDataset::from_prefix_with_opts(uri, &o)
            //.map_err(|e| PyRuntimeError::new_err(e))?;
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: ds })
    }
    /// Optional: expose keys for debugging
    fn keys(&self) -> Vec<String> { self.inner.keys().clone() }
}

// async loader over S3BytesDataset
#[pyclass]
pub struct PyS3AsyncDataLoader {
    rx: Arc<Mutex<mpsc::Receiver<Result<Vec<Vec<u8>>, DatasetError>>>>,
}


// AsyncDataLoader function
#[pymethods]
impl PyS3AsyncDataLoader {
    #[new]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        // Parse options once, and apply safe defaults for range mode tuning.
        let o = opts_from_dict(opts)
            .part_size(8 * 1024 * 1024)
            .max_inflight_parts(4);

        // Build the dataset with the SAME options (reader_mode, part_size, etc.)
        let ds = S3BytesDataset::from_prefix_with_opts(uri, &o)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Use the SAME options for DataLoader and for channel capacity.
        let (tx, rx) =
            mpsc::channel::<Result<Vec<Vec<u8>>, DatasetError>>(o.prefetch.max(1));

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let loader = DataLoader::new(ds, o);
            let mut stream = loader.stream();
            while let Some(batch) = stream.next().await {
                if tx.send(batch).await.is_err() { break; }
            }
        });

        Ok(Self { rx: Arc::new(Mutex::new(rx)) })
    }

    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = self.rx.clone();
        future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                Some(Ok(batch)) => Python::with_gil(|py| {
                    let out: Vec<Py<PyBytes>> = batch.into_iter()
                        .map(|b| PyBytes::new(py, &b).unbind())
                        .collect();
                    Ok(out.into_py_any(py)?)
                }),
                Some(Err(e)) => Err(PyRuntimeError::new_err(format!("{:?}", e))),
                None => Err(PyStopAsyncIteration::new_err("end of stream")),
            }
        })
    }
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
    let (bucket, mut path) = parse_s3_uri(uri).map_err(py_err)?;

    // If the path ends with a slash, automatically append a wildcard to search inside it.
    if path.ends_with('/') {
        path.push_str(".*");
    }

    crate::s3_utils::list_objects(&bucket, &path, recursive).map_err(py_err)
}


// New stat calls, almost the same ???? 
//
#[pyfunction]
pub fn stat(py: Python<'_>, uri: &str) -> PyResult<PyObject> {
    let os = stat_object_uri(uri).map_err(py_err)?;
    let d = stat_to_pydict(py, os)?;         // Bound<'py, PyDict>
    Ok(d.unbind().into())                    // -> PyObject (owned)
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
    let bytes = get_object_uri(uri).map_err(py_err)?;
    Ok(PyBytes::new(py, &bytes[..]).unbind())
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
    let def = LoaderOptions::default();

    if let Some(d) = d {
        // Helper closures that cope with the Result-of-Option signature.
        let g_usize = |k: &str, dv: usize| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<usize>().unwrap_or(dv),
            _             => dv,
        };
        let g_bool = |k: &str, dv: bool| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<bool>().unwrap_or(dv),
            _             => dv,
        };
        let g_u64 = |k: &str, dv: u64| match d.get_item(k) {
            Ok(Some(val)) => val.extract::<u64>().unwrap_or(dv),
            _             => dv,
        };

        // Reader mode: accept "sequential" or "range" (case-insensitive).
        let reader_mode = match d.get_item("reader_mode") {
            Ok(Some(val)) => {
                if let Ok(s) = val.extract::<&str>() {
                    match s.to_ascii_lowercase().as_str() {
                        "range"      => ReaderMode::Range,
                        "sequential" => ReaderMode::Sequential,
                        _            => def.reader_mode,
                    }
                } else {
                    def.reader_mode
                }
            }
            _ => def.reader_mode,
        };

        LoaderOptions {
            // existing fields
            batch_size:           g_usize("batch_size",  def.batch_size),
            drop_last:            g_bool ("drop_last",   def.drop_last),
            shuffle:              g_bool ("shuffle",     def.shuffle),
            seed:                 g_u64  ("seed",        def.seed),
            num_workers:          g_usize("num_workers", def.num_workers),
            prefetch:             g_usize("prefetch",    def.prefetch),
            auto_tune:            g_bool ("auto_tune",   def.auto_tune),

            // new fields
            loading_mode:         def.loading_mode,  // Use default loading mode
            reader_mode,
            part_size:            g_usize("part_size",           def.part_size),
            max_inflight_parts:   g_usize("max_inflight_parts",  def.max_inflight_parts),

            shard_rank:           g_usize("shard_rank",          def.shard_rank),
            shard_world_size:     g_usize("shard_world_size",    def.shard_world_size),
            worker_id:            g_usize("worker_id",           def.worker_id),
            num_workers_pytorch:  g_usize("num_workers_pytorch", def.num_workers_pytorch),
        }
    } else {
        def
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
// Multipart upload code 
// ---------------------------------------------------------------------------
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
    pending_buf: Option<Vec<u8>>, // Rust-owned buffer between reserve() and commit()
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
        bucket: &str,
        key: &str,
        part_size: Option<usize>,
        max_in_flight: Option<usize>,
        content_type: Option<String>,
        abort_on_drop: Option<bool>,
    ) -> PyResult<Self> {
        let mut cfg = MultipartUploadConfig::default();
        if let Some(ps) = part_size { cfg.part_size = ps; }
        if let Some(mif) = max_in_flight { cfg.max_in_flight = mif; }
        if let Some(ct) = content_type { cfg.content_type = Some(ct); }
        if let Some(aod) = abort_on_drop { cfg.abort_on_drop = aod; }

        let inner = MultipartUploadSink::new(bucket, key, cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Multipart init failed: {e}")))?;
        Ok(Self { inner: Some(inner), pending_buf: None })
    }

    /// Create a writer from an `s3://bucket/key` URI.
    ///
    /// Args:
    ///     uri: e.g. "s3://my-bucket/dir/obj.bin"
    ///     part_size, max_in_flight, content_type, abort_on_drop: see `__init__`.
    #[staticmethod]
    #[pyo3(signature = (uri, part_size=None, max_in_flight=None, content_type=None, abort_on_drop=None))]
    fn from_uri(
        uri: &str,
        part_size: Option<usize>,
        max_in_flight: Option<usize>,
        content_type: Option<String>,
        abort_on_drop: Option<bool>,
    ) -> PyResult<Self> {
        let mut cfg = MultipartUploadConfig::default();
        if let Some(ps) = part_size { cfg.part_size = ps; }
        if let Some(mif) = max_in_flight { cfg.max_in_flight = mif; }
        if let Some(ct) = content_type { cfg.content_type = Some(ct); }
        if let Some(aod) = abort_on_drop { cfg.abort_on_drop = aod; }

        let inner = MultipartUploadSink::from_uri(uri, cfg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Multipart init failed: {e}")))?;
        Ok(Self { inner: Some(inner), pending_buf: None })
    }

    /// Write a bytes-like object (bytes, bytearray, memoryview, NumPy buffer).
    ///
    /// For large buffers (>= 8 MiB), uses an owned fast path internally to avoid
    /// extra copies. For maximum performance, prefer `reserve()`/`commit()`.
    ///
    /// Returns:
    ///     int: number of bytes accepted.
    #[pyo3(text_signature = "(self, data, /)")]
    fn write<'py>(&mut self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<usize> {
        let inner = self.inner.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("writer is closed"))?;

        const OWNED_THRESHOLD: usize = 8 * 1024 * 1024; // 8 MiB

        // 1) Fast path for any buffer-protocol object (NumPy, memoryview, bytearray, etc.)
        if let Ok(buf) = data.extract::<pyo3::buffer::PyBuffer<u8>>() {
            let len = buf.len_bytes();
            let mut vec = Vec::<u8>::with_capacity(len);
            unsafe { vec.set_len(len); }
            buf.copy_to_slice(py, &mut vec[..])
                .map_err(|e| PyRuntimeError::new_err(format!("buffer copy failed: {e}")))?;
    
            let res = if len >= OWNED_THRESHOLD {
                py.allow_threads(|| inner.write_owned_blocking(vec))
            } else {
                py.allow_threads(|| inner.write_blocking(&vec))
            };
            return res
                .map(|_| len)
                .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")));
        }
    
        // 2) Fallback for plain bytes objects
        if let Ok(bytes) = data.downcast::<PyBytes>() {
            let slice = bytes.as_bytes(); // needs PyBytesMethods import
            let len = slice.len();
            if len >= OWNED_THRESHOLD {
                let vec = slice.to_vec();
                let res = py.allow_threads(|| inner.write_owned_blocking(vec));
                return res
                    .map(|_| len)
                    .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")));
            } else {
                let res = py.allow_threads(|| inner.write_blocking(slice));
                return res
                    .map(|_| len)
                    .map_err(|e| PyRuntimeError::new_err(format!("write failed: {e}")));
            }
        }
    
        Err(pyo3::exceptions::PyTypeError::new_err(
            "write() expects bytes-like (bytes, bytearray, memoryview, NumPy buffer)",
        ))
    }

    /// Reserve a Rust-owned buffer and return a writable Python `memoryview`.
    ///
    /// Fill the memoryview in Python, then call `commit(nbytes)`. Do not reuse
    /// the memoryview after `commit()`. Only one pending buffer is allowed.
    ///
    /// Args:
    ///     size: number of bytes to reserve
    ///
    /// Returns:
    ///     memoryview: writable view into the reserved buffer
    #[pyo3(text_signature = "(self, size, /)")]
    fn reserve(&mut self, py: Python<'_>, size: usize) -> PyResult<PyObject> {
        // Disallow overlapping reserves
        if self.pending_buf.is_some() {
            return Err(PyRuntimeError::new_err(
                "reserve() called while a previous buffer is pending; call commit() first",
            ));
        }

        // reserve(): treat size==0 as no-op (avoids odd memoryviews)
        if size == 0 {
            self.pending_buf = Some(Vec::new());
            // Return a 0-length memoryview (still legal)
            let ptr = std::ptr::null_mut::<std::os::raw::c_char>();
            let mv_ptr = unsafe { pyo3::ffi::PyMemoryView_FromMemory(ptr, 0, pyo3::ffi::PyBUF_WRITE) };
            if mv_ptr.is_null() { return Err(PyErr::fetch(py)); }
            return Ok(unsafe { PyObject::from_owned_ptr(py, mv_ptr) });
        }
    
        // Allocate Rust-owned buffer and keep it alive in self.pending_buf so the
        // memoryview's pointer stays valid until commit() (or close/abort).
        let mut buf = vec![0u8; size];
    
        // Create a writable memoryview that points directly at our Vec's memory.
        // Safety:
        // - We pass a valid pointer/length for the Vec.
        // - We keep the Vec alive (self.pending_buf = Some(buf)) so the memory doesn't move.
        // - We won't reallocate this Vec (we never push; commit() only shrinks len).
        let ptr = buf.as_mut_ptr() as *mut c_char;
        let len = buf.len() as ffi::Py_ssize_t;
        let flags = ffi::PyBUF_WRITE; // writable view
    
        let mv_ptr = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, flags) };
        if mv_ptr.is_null() {
            // Convert the current Python error (if any) into a PyErr.
            return Err(PyErr::fetch(py));
        }
    
        // Convert the owned PyObject* into a safe PyObject handle.
        let mv = unsafe { PyObject::from_owned_ptr(py, mv_ptr) };
    
        // Stash the Vec; this guarantees the pointer remains valid until commit().
        self.pending_buf = Some(buf);
        Ok(mv)
    }



    /// Commit the previously reserved buffer.
    ///
    /// Args:
    ///     nbytes: number of bytes actually written into the reserved buffer
    ///
    /// Errors if `nbytes` exceeds the reserved size or if no buffer is pending.
    #[pyo3(text_signature = "(self, nbytes, /)")]
    fn commit(&mut self, py: Python<'_>, nbytes: usize) -> PyResult<()> {
        let inner = self.inner.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        let mut buf = self.pending_buf.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no pending buffer; call reserve() first"))?;
        if nbytes > buf.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("commit(nbytes={nbytes}) exceeds reserved size {}", buf.len()),
            ));
        }
        unsafe { buf.set_len(nbytes); }
        py.allow_threads(|| inner.write_owned_blocking(buf))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("commit failed: {e}")))
    }

    /// Flush any buffered bytes as a (possibly short) part without finishing.
    ///
    /// Typically not required; `close()` will upload any tail automatically.
    #[pyo3(text_signature = "(self)")]
    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer is closed"))?;
        py.allow_threads(|| inner.flush_blocking())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("flush failed: {e}")))
    }

    /// Complete the multipart upload and close the writer.
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
    fn close(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.inner.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already closed"))?;
        let info = py.allow_threads(|| { let mut owned = inner; owned.finish_blocking() })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("finish failed: {e}")))?;
        let dict = PyDict::new(py);
        if let Some(etag) = info.e_tag { dict.set_item("etag", etag).ok(); }
        dict.set_item("total_bytes", info.total_bytes).ok();
        dict.set_item("parts", info.parts).ok();
        let started = info.started_at.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        let completed = info.completed_at.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        dict.set_item("started_at", started).ok();
        dict.set_item("completed_at", completed).ok();
        Ok(dict.into())
    }

    /// Abort the multipart upload and close the writer.
    ///
    /// Best effort: outstanding tasks are joined and the MPU is aborted.
    /// After abort, the writer is unusable.
    #[pyo3(text_signature = "(self)")]
    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut inner) = self.inner.take() {
            py.allow_threads(|| inner.abort_blocking())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("abort failed: {e}")))?;
        }
        Ok(())
    }

    /// Context manager enter: returns the writer.
    #[pyo3(text_signature = "(self)")]
    fn __enter__<'py>(slf: PyRefMut<'py, Self>) -> PyResult<PyRefMut<'py, Self>> { Ok(slf) }

    /// Context manager exit: closes on success; aborts on exception.
    #[pyo3(text_signature = "(self, exc_type, exc, tb)")]
    fn __exit__(&mut self, py: Python<'_>, _t: PyObject, _v: PyObject, _tb: PyObject) -> PyResult<()> {
        if self.inner.is_some() {
            if let Some(mut inner) = self.inner.take() {
                if let Err(_e) = py.allow_threads(|| inner.finish_blocking()) {
                    let _ = py.allow_threads(|| inner.abort_blocking());
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "extension-module")]
use crate::checkpoint::{CheckpointStore, CheckpointConfig, Strategy, CheckpointInfo};

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyCheckpointStore {
    inner: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyCheckpointStore {
    #[new]
    #[pyo3(text_signature = "(uri, strategy=None, multipart_threshold=None)")]
    fn new(
        uri: String,
        strategy: Option<String>,
        multipart_threshold: Option<usize>,
    ) -> PyResult<Self> {
        let mut config = CheckpointConfig::new();
        
        if let Some(strat_str) = strategy {
            let strat = Strategy::from_str(&strat_str)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid strategy: {}", e)))?;
            config = config.with_strategy(strat);
        }
        
        if let Some(threshold) = multipart_threshold {
            config = config.with_multipart_threshold(threshold);
        }
        
        let store = CheckpointStore::open_with_config(&uri, config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open checkpoint store: {}", e)))?;
        
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?
        );
        
        Ok(Self {
            inner: Arc::new(Mutex::new(store)),
            runtime,
        })
    }

    /// Save a single-rank checkpoint
    #[pyo3(text_signature = "(self, step, epoch, framework, data, user_meta=None)")]
    fn save(
        &self,
        py: Python<'_>,
        step: u64,
        epoch: u64,
        framework: String,
        data: &[u8],
        user_meta: Option<PyObject>,
    ) -> PyResult<String> {
        let user_meta = user_meta.map(|_obj| {
            // For now, just use null for user metadata
            // TODO: Implement proper Python->JSON conversion
            serde_json::Value::Null
        });
        
        let inner = self.inner.clone();
        let data = data.to_vec();
        
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let store = inner.lock().await;
                store.save(step, epoch, &framework, &data, user_meta).await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Save failed: {}", e)))
    }

    /// Load the latest checkpoint
    #[pyo3(text_signature = "(self)")]
    fn load_latest(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let inner = self.inner.clone();
        
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let store = inner.lock().await;
                store.load_latest().await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Load failed: {}", e)))
        .map(|opt_data| opt_data.map(|data| PyBytes::new(py, &data).into()))
    }

    /// List available checkpoints
    #[pyo3(text_signature = "(self)")]
    fn list_checkpoints(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let inner = self.inner.clone();
        
        let infos = py.allow_threads(|| {
            self.runtime.block_on(async {
                let store = inner.lock().await;
                store.list_checkpoints().await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("List failed: {}", e)))?;
        
        Ok(infos.into_iter().map(|info| {
            PyCheckpointInfo { inner: info }.into_py_any(py).unwrap()
        }).collect())
    }

    /// Delete a checkpoint
    #[pyo3(text_signature = "(self, step)")]
    fn delete_checkpoint(&self, py: Python<'_>, step: u64) -> PyResult<bool> {
        let inner = self.inner.clone();
        
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let store = inner.lock().await;
                store.delete_checkpoint(step).await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Delete failed: {}", e)))
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

#[pyclass]
pub struct PyCheckpointWriter {
    store: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
    world_size: u32,
    rank: u32,
}

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
        data: &[u8],
    ) -> PyResult<PyObject> {
        let store = self.store.clone();
        let data = data.to_vec();
        let world_size = self.world_size;
        let rank = self.rank;
        
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let store_guard = store.lock().await;
                let writer = store_guard.writer(world_size, rank);
                writer.save_distributed_shard(step, epoch, &framework, &data).await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Save shard failed: {}", e)))
        .map(|(layout, shard_meta)| {
            let dict = PyDict::new(py);
            dict.set_item("step", layout.step).ok();
            dict.set_item("timestamp", &layout.ts).ok();
            dict.set_item("rank", shard_meta.rank).ok();
            dict.set_item("key", &shard_meta.key).ok();
            dict.set_item("size", shard_meta.size).ok();
            if let Some(etag) = &shard_meta.etag {
                dict.set_item("etag", etag).ok();
            }
            dict.into()
        })
    }

    /// Finalize distributed checkpoint (rank 0 only)
    #[pyo3(text_signature = "(self, step, epoch, framework, shard_metas, user_meta=None)")]
    fn finalize_distributed_checkpoint(
        &self,
        py: Python<'_>,
        step: u64,
        epoch: u64,
        framework: String,
        shard_metas: Vec<PyObject>,
        user_meta: Option<PyObject>,
    ) -> PyResult<String> {
        let user_meta = user_meta.map(|_obj| {
            // For now, just use null for user metadata
            // TODO: Implement proper Python->JSON conversion
            serde_json::Value::Null
        });
        
        // Convert shard metas from Python objects
        let mut shards = Vec::new();
        for shard_obj in shard_metas {
            let dict = shard_obj.downcast_bound::<PyDict>(py)
                .map_err(|_| PyRuntimeError::new_err("Shard meta must be a dict"))?;
            
            let rank = dict.get_item("rank")?
                .ok_or_else(|| PyRuntimeError::new_err("Missing rank in shard meta"))?
                .extract::<u32>()?;
            let key = dict.get_item("key")?
                .ok_or_else(|| PyRuntimeError::new_err("Missing key in shard meta"))?
                .extract::<String>()?;
            let size = dict.get_item("size")?
                .ok_or_else(|| PyRuntimeError::new_err("Missing size in shard meta"))?
                .extract::<u64>()?;
            let etag = dict.get_item("etag")?.map(|e| e.extract::<String>()).transpose()?;
            
            shards.push(crate::checkpoint::ShardMeta {
                rank,
                key,
                size,
                etag,
                checksum: None,
            });
        }
        
        let store = self.store.clone();
        let world_size = self.world_size;
        let rank = self.rank;
        
        py.allow_threads(|| {
            self.runtime.block_on(async {
                use crate::checkpoint::KeyLayout;
                let layout = KeyLayout::new(
                    store.lock().await.uri.clone(),
                    step
                );
                let store_guard = store.lock().await;
                let writer = store_guard.writer(world_size, rank);
                writer.finalize_distributed_checkpoint(
                    &layout,
                    &framework,
                    epoch,
                    shards,
                    user_meta,
                ).await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Finalize failed: {}", e)))
    }
}

#[pyclass]
pub struct PyCheckpointReader {
    store: Arc<Mutex<CheckpointStore>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyCheckpointReader {
    /// Load latest manifest
    #[pyo3(text_signature = "(self)")]
    fn load_latest_manifest(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let store = self.store.clone();
        
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let store_guard = store.lock().await;
                let reader = store_guard.reader();
                reader.load_latest_manifest().await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Load manifest failed: {}", e)))
        .map(|opt_manifest| opt_manifest.map(|manifest| {
            let dict = PyDict::new(py);
            dict.set_item("format_version", manifest.format_version).ok();
            dict.set_item("framework", &manifest.framework).ok();
            dict.set_item("global_step", manifest.global_step).ok();
            dict.set_item("epoch", manifest.epoch).ok();
            dict.set_item("wall_time", &manifest.wall_time).ok();
            dict.set_item("world_size", manifest.world_size).ok();
            dict.set_item("status", &manifest.status).ok();
            
            let shards_list = PyList::new(py, manifest.shards.iter().map(|shard| {
                let shard_dict = PyDict::new(py);
                shard_dict.set_item("rank", shard.rank).ok();
                shard_dict.set_item("key", &shard.key).ok();
                shard_dict.set_item("size", shard.size).ok();
                if let Some(etag) = &shard.etag {
                    shard_dict.set_item("etag", etag).ok();
                }
                shard_dict
            })).unwrap();
            dict.set_item("shards", shards_list).ok();
            
            dict.into()
        }))
    }

    /// Read shard by rank
    #[pyo3(text_signature = "(self, manifest, rank)")]
    fn read_shard_by_rank(
        &self,
        py: Python<'_>,
        manifest: PyObject,
        rank: u32,
    ) -> PyResult<PyObject> {
        // Parse manifest from Python dict
        let manifest_dict = manifest.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Manifest must be a dict"))?;
        
        let shards_item = manifest_dict.get_item("shards")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing shards in manifest"))?;
        let shards_list = shards_item.downcast::<PyList>()?;
        
        let mut target_key = None;
        for shard_obj in shards_list.iter() {
            let shard_dict = shard_obj.downcast::<PyDict>()?;
            let shard_rank = shard_dict.get_item("rank")?
                .ok_or_else(|| PyRuntimeError::new_err("Missing rank in shard"))?
                .extract::<u32>()?;
            if shard_rank == rank {
                target_key = Some(shard_dict.get_item("key")?
                    .ok_or_else(|| PyRuntimeError::new_err("Missing key in shard"))?
                    .extract::<String>()?);
                break;
            }
        }
        
        let key = target_key.ok_or_else(|| PyRuntimeError::new_err(format!("Rank {} not found", rank)))?;
        
        let store = self.store.clone();
        
        let data = py.allow_threads(|| {
            self.runtime.block_on(async {
                let store_guard = store.lock().await;
                let reader = store_guard.reader();
                reader.read_shard(&key).await
            })
        }).map_err(|e| PyRuntimeError::new_err(format!("Read shard failed: {}", e)))?;
        
        Ok(PyBytes::new(py, &data).into())
    }
}

#[pyclass]
pub struct PyCheckpointInfo {
    inner: CheckpointInfo,
}

#[pymethods]
impl PyCheckpointInfo {
    #[getter]
    fn step(&self) -> u64 { self.inner.step }
    
    #[getter]
    fn epoch(&self) -> u64 { self.inner.epoch }
    
    #[getter]
    fn timestamp(&self) -> &str { &self.inner.timestamp }
    
    #[getter]
    fn status(&self) -> &str { &self.inner.status }
    
    #[getter]
    fn framework(&self) -> &str { &self.inner.framework }
    
    #[getter]
    fn world_size(&self) -> u32 { self.inner.world_size }
    
    #[getter]
    fn total_size(&self) -> u64 { self.inner.total_size }
    
    #[getter]
    fn size_mb(&self) -> f64 { self.inner.size_mb() }
    
    #[getter]
    fn size_gb(&self) -> f64 { self.inner.size_gb() }
    
    #[getter]
    fn is_complete(&self) -> bool { self.inner.is_complete() }
    
    #[getter]
    fn is_single_rank(&self) -> bool { self.inner.is_single_rank() }
}

// Convenience functions
#[pyfunction]
#[pyo3(text_signature = "(uri, step, epoch, framework, data, user_meta=None)")]
fn save_checkpoint(
    py: Python<'_>,
    uri: String,
    step: u64,
    epoch: u64,
    framework: String,
    data: &[u8],
    user_meta: Option<PyObject>,
) -> PyResult<String> {
    let store = PyCheckpointStore::new(uri, None, None)?;
    store.save(py, step, epoch, framework, data, user_meta)
}

#[pyfunction]
#[pyo3(text_signature = "(uri)")]
fn load_checkpoint(py: Python<'_>, uri: String) -> PyResult<Option<PyObject>> {
    let store = PyCheckpointStore::new(uri, None, None)?;
    store.load_latest(py)
}

#[pyfunction]
#[pyo3(text_signature = "(uri, step, epoch, framework, data, world_size, rank)")]
fn save_distributed_shard(
    py: Python<'_>,
    uri: String,
    step: u64,
    epoch: u64,
    framework: String,
    data: &[u8],
    world_size: u32,
    rank: u32,
) -> PyResult<PyObject> {
    let store = PyCheckpointStore::new(uri, None, None)?;
    let writer = store.writer(world_size, rank)?;
    writer.save_distributed_shard(py, step, epoch, framework, data)
}

#[pyfunction]
#[pyo3(text_signature = "(uri, step, epoch, framework, shard_metas, world_size, rank, user_meta=None)")]
fn finalize_distributed_checkpoint(
    py: Python<'_>,
    uri: String,
    step: u64,
    epoch: u64,
    framework: String,
    shard_metas: Vec<PyObject>,
    world_size: u32,
    rank: u32,
    user_meta: Option<PyObject>,
) -> PyResult<String> {
    let store = PyCheckpointStore::new(uri, None, None)?;
    let writer = store.writer(world_size, rank)?;
    writer.finalize_distributed_checkpoint(py, step, epoch, framework, shard_metas, user_meta)
}

// ---------------------------------------------------------------------------
// Sub‑module exposed at crate‑root (`lib.rs` re‑exports it as needed)
// ---------------------------------------------------------------------------
#[pymodule]
pub fn _pymod(m: &Bound<PyModule>) -> PyResult<()> {
    // S3 helpers
    m.add_function(wrap_pyfunction!(create_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(delete_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(list_objects, m)?)?;
    m.add_function(wrap_pyfunction!(stat, m)?)?;
    m.add_function(wrap_pyfunction!(stat_async, m)?)?;
    m.add_function(wrap_pyfunction!(stat_many_async, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_object, m)?)?;
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
    m.add_class::<PyS3Dataset>()?;
    m.add_class::<PyS3AsyncDataLoader>()?;

    // Checkpoint writer
    m.add_class::<PyMultipartUploadWriter>()?;

    // Checkpoint system
    m.add_class::<PyCheckpointStore>()?;
    m.add_class::<PyCheckpointWriter>()?;
    m.add_class::<PyCheckpointReader>()?;
    m.add_class::<PyCheckpointInfo>()?;
    m.add_function(wrap_pyfunction!(save_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(save_distributed_shard, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_distributed_checkpoint, m)?)?;

    Ok(())
}

