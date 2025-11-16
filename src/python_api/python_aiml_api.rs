// src/python_api/python_aiml_api.rs
//
// Copyright 2025
// Signal65 / Futurum Group.
//
// Contains AI/ML features: NumPy integration, DataLoaders, and Checkpointing.

#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::{PyObject, Bound};
use pyo3::types::{PyAny, PyBytes, PyDict, PyDictMethods, PyList};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::conversion::IntoPyObjectExt;
use pyo3_async_runtimes::tokio::future_into_py;

use futures_util::StreamExt;
use tokio::sync::{mpsc, Mutex, Semaphore};
use tokio::task::JoinSet;

use numpy::ndarray as nd_np;
use numpy::PyArrayDyn;

use std::sync::Arc;

// Project crates
use crate::object_store::store_for_uri;
use crate::s3_utils::get_object_uri;
use crate::python_api::python_core_api::PyBytesView;
use crate::data_loader::{
    DataLoader,
    dataset::{Dataset, DatasetError},
    options::{LoaderOptions, ReaderMode},
    s3_bytes::S3BytesDataset,
};
use crate::api::dataset_for_uri_with_options;
#[cfg(feature = "extension-module")]
use crate::checkpoint::{CheckpointStore, CheckpointConfig, Strategy, CheckpointInfo};

// Helper function to convert errors
fn py_err<E: std::fmt::Display>(e: E) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

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

// Generic dataset that works with all URI schemes
#[pyclass]
pub struct PyDataset {
    inner: Arc<dyn Dataset<Item = Vec<u8>>>,
}

#[pymethods]
impl PyDataset {
    #[new]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        let o = opts_from_dict(opts);
        let dataset = dataset_for_uri_with_options(uri, &o)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::from(dataset) })
    }

    /// Get the number of items in the dataset
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len().unwrap_or(0))
    }

    /// Get an item by index (for map-style datasets) 
    fn get_item<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Py<PyAny>> {
        let inner = Arc::clone(&self.inner);
        let bound_result = future_into_py(py, async move {
            let data = inner.get(index).await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Python::with_gil(|py| {
                // Bound<'py, PyBytes> -> Bound<'py, PyAny> -> Py<PyAny>
                let obj: Py<PyAny> = PyBytes::new(py, &data).into_any().unbind();
                Ok(obj)
            })
        })?;
        Ok(bound_result.unbind())
    }

    /// Return list of keys if supported (for debugging S3 datasets)
    fn keys(&self) -> PyResult<Vec<String>> {
        // This is a bit of a hack - we try to downcast to S3BytesDataset
        // TODO: Make this more elegant with a proper trait
        Err(PyRuntimeError::new_err("keys() not supported for generic datasets yet"))
    }
}

impl Clone for PyDataset {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

// Generic async loader that works with PyDataset
#[pyclass]
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
        Ok(Self { dataset, opts: opts_from_dict(opts) })
    }

    fn __aiter__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Py<PyBytesAsyncDataLoaderIter>> {
        PyBytesAsyncDataLoaderIter::spawn_stream(py, slf.dataset.clone(), slf.opts.clone())
    }
}

#[pyclass]
pub struct PyBytesAsyncDataLoaderIter {
    rx: Arc<Mutex<mpsc::Receiver<Result<Vec<Vec<u8>>, DatasetError>>>>,
}

impl PyBytesAsyncDataLoaderIter {
    fn spawn_stream(
        py: Python<'_>,
        dataset: PyDataset,
        opts: LoaderOptions,
    ) -> PyResult<Py<Self>> {
        let (tx, rx) = mpsc::channel::<Result<Vec<Vec<u8>>, DatasetError>>(opts.prefetch.max(1));

        // Extract concurrency setting for concurrent batch fetching
        // Default to 8 workers if num_workers is 0
        let num_workers = if opts.num_workers > 0 { opts.num_workers } else { 8 };

        // Spawn the producer task with concurrent batch fetching
        pyo3_async_runtimes::tokio::get_runtime()
            .spawn(async move {
                let batch_size = opts.batch_size.max(1);
                let semaphore = Arc::new(Semaphore::new(num_workers));
                
                if let Some(len) = dataset.inner.len() {
                    let mut i = 0;
                    while i < len {
                        // Collect indices for this batch
                        let batch_indices: Vec<usize> = (0..batch_size)
                            .filter_map(|j| {
                                let idx = i + j;
                                if idx < len { Some(idx) } else { None }
                            })
                            .collect();
                        
                        if batch_indices.is_empty() { break; }
                        
                        // Fetch batch items CONCURRENTLY using JoinSet
                        let mut join_set = JoinSet::new();
                        for (order, idx) in batch_indices.iter().enumerate() {
                            let dataset_clone = dataset.clone();
                            let sem = Arc::clone(&semaphore);
                            let idx = *idx;
                            
                            join_set.spawn(async move {
                                let _permit = sem.acquire().await.unwrap();
                                let data = dataset_clone.inner.get(idx).await?;
                                Ok::<(usize, Vec<u8>), DatasetError>((order, data))
                            });
                        }
                        
                        // Collect results preserving order
                        let mut batch_results: Vec<Option<Vec<u8>>> = vec![None; batch_indices.len()];
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
                                    let _ = tx.send(Err(DatasetError::from(join_err.to_string()))).await;
                                    return;
                                }
                            }
                        }
                        
                        // Convert to batch (filter_map skips None, but all should be Some)
                        let batch: Vec<Vec<u8>> = batch_results.into_iter()
                            .filter_map(|x| x)
                            .collect();
                        
                        if !batch.is_empty() {
                            if tx.send(Ok(batch)).await.is_err() { break; }
                        }
                        i += batch_size;
                    }
                }
            });

        Py::new(py, Self { rx: Arc::new(Mutex::new(rx)) })
    }
}

#[pymethods]
impl PyBytesAsyncDataLoaderIter {
    fn __anext__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Py<PyAny>> {
        let rx = Arc::clone(&slf.rx);
        let bound_result = future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                Some(Ok(batch)) => {
                    Python::with_gil(|py| {
                        // Always return list[bytes] for consistent type contract
                        // This ensures PyTorch/JAX/TF pipelines get stable types
                        let py_list = PyList::empty(py);
                        for item in batch {
                            py_list.append(PyBytes::new(py, &item))?;
                        }
                        Ok(py_list.into_any().unbind())
                    })
                }
                Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
                None => Err(PyStopAsyncIteration::new_err("StopAsyncIteration")),
            }
        })?;
        Ok(bound_result.unbind())
    }
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

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let bound_result = future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                Some(Ok(batch)) => Python::with_gil(|py| {
                    let out: Vec<Py<PyBytes>> = batch.into_iter()
                        .map(|b| PyBytes::new(py, &b).unbind())
                        .collect();
                    // Vec<Py<PyBytes>> -> Py<PyAny> via IntoPyObjectExt
                    Ok(out.into_py_any(py)?)
                }),
                Some(Err(e)) => Err(PyRuntimeError::new_err(format!("{:?}", e))),
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
    use crate::data_formats::npz::{read_npz_array, list_npz_arrays};
    
    let bytes = get_object_uri(uri).map_err(py_err)?;
    
    // Determine array name
    let name = if let Some(n) = array_name {
        n.to_string()
    } else {
        // Get first array name from archive
        let names = list_npz_arrays(&bytes).map_err(py_err)?;
        names.into_iter().next()
            .ok_or_else(|| PyRuntimeError::new_err("NPZ file is empty"))?
    };
    
    // Read array using our custom implementation
    let arr: ndarray::ArrayD<f32> = read_npz_array(&bytes, &name).map_err(py_err)?;
    
    let shape = arr.shape().to_vec();
    let (data, _offset) = arr.into_raw_vec_and_offset();
    let arr_np = nd_np::ArrayD::from_shape_vec(nd_np::IxDyn(&shape), data).map_err(py_err)?;
    let py_arr = PyArrayDyn::<f32>::from_owned_array(py, arr_np);
    py_arr.into_py_any(py)
}

// ---------------------------------------------------------------------------
// LoaderOptions Python wrapper with AI/ML realism knobs
// ---------------------------------------------------------------------------
#[pyclass]
#[derive(Clone)]
pub struct PyLoaderOptions {
    inner: LoaderOptions,
}

#[pymethods]
impl PyLoaderOptions {
    #[new]
    fn new() -> Self {
        Self { inner: LoaderOptions::default() }
    }

    // Basic options
    fn with_batch_size(&mut self, size: usize) -> Self {
        Self { inner: self.inner.clone().with_batch_size(size) }
    }

    fn drop_last(&mut self, yes: bool) -> Self {
        Self { inner: self.inner.clone().drop_last(yes) }
    }

    fn shuffle(&mut self, on: bool, seed: u64) -> Self {
        Self { inner: self.inner.clone().shuffle(on, seed) }
    }

    fn num_workers(&mut self, n: usize) -> Self {
        Self { inner: self.inner.clone().num_workers(n) }
    }

    fn prefetch(&mut self, n: usize) -> Self {
        Self { inner: self.inner.clone().prefetch(n) }
    }

    fn auto_tune(&mut self, on: bool) -> Self {
        Self { inner: self.inner.clone().auto_tune(on) }
    }

    // AI/ML realism knobs
    fn pin_memory(&mut self, pin: bool) -> Self {
        Self { inner: self.inner.clone().pin_memory(pin) }
    }

    fn persistent_workers(&mut self, persistent: bool) -> Self {
        Self { inner: self.inner.clone().persistent_workers(persistent) }
    }

    fn with_timeout(&mut self, seconds: f64) -> Self {
        Self { inner: self.inner.clone().with_timeout(seconds) }
    }

    fn use_spawn(&mut self) -> Self {
        Self { inner: self.inner.clone().use_spawn() }
    }

    fn use_fork(&mut self) -> Self {
        Self { inner: self.inner.clone().use_fork() }
    }

    fn use_forkserver(&mut self) -> Self {
        Self { inner: self.inner.clone().use_forkserver() }
    }

    fn random_sampling(&mut self, replacement: bool) -> Self {
        Self { inner: self.inner.clone().random_sampling(replacement) }
    }

    fn distributed_sampling(&mut self, rank: usize, world_size: usize) -> Self {
        Self { inner: self.inner.clone().distributed_sampling(rank, world_size) }
    }

    fn channels_last(&mut self) -> Self {
        Self { inner: self.inner.clone().channels_last() }
    }

    fn channels_first(&mut self) -> Self {
        Self { inner: self.inner.clone().channels_first() }
    }

    fn non_blocking(&mut self, non_blocking: bool) -> Self {
        Self { inner: self.inner.clone().non_blocking(non_blocking) }
    }

    fn with_generator_seed(&mut self, seed: u64) -> Self {
        Self { inner: self.inner.clone().with_generator_seed(seed) }
    }

    fn enable_transforms(&mut self, enable: bool) -> Self {
        Self { inner: self.inner.clone().enable_transforms(enable) }
    }

    fn collate_buffer_size(&mut self, size: usize) -> Self {
        Self { inner: self.inner.clone().collate_buffer_size(size) }
    }

    // Convenience presets
    fn gpu_optimized(&mut self) -> Self {
        Self { inner: self.inner.clone().gpu_optimized() }
    }

    fn distributed_optimized(&mut self, rank: usize, world_size: usize) -> Self {
        Self { inner: self.inner.clone().distributed_optimized(rank, world_size) }
    }

    fn debug_mode(&mut self) -> Self {
        Self { inner: self.inner.clone().debug_mode() }
    }

    // Property accessors (renamed to avoid conflicts)
    #[getter]
    fn get_batch_size(&self) -> usize { self.inner.batch_size }

    #[getter]
    fn get_drop_last(&self) -> bool { self.inner.drop_last }

    #[getter]
    fn get_shuffle(&self) -> bool { self.inner.shuffle }

    #[getter]
    fn get_seed(&self) -> u64 { self.inner.seed }

    #[getter]
    fn get_num_workers(&self) -> usize { self.inner.num_workers }

    #[getter]
    fn get_prefetch(&self) -> usize { self.inner.prefetch }

    #[getter]
    fn get_pin_memory(&self) -> bool { self.inner.pin_memory }

    #[getter]
    fn get_persistent_workers(&self) -> bool { self.inner.persistent_workers }

    #[getter]
    fn get_timeout_seconds(&self) -> Option<f64> { self.inner.timeout_seconds }

    #[getter]
    fn get_non_blocking(&self) -> bool { self.inner.non_blocking }

    #[getter]
    fn get_shard_rank(&self) -> usize { self.inner.shard_rank }

    #[getter]
    fn get_shard_world_size(&self) -> usize { self.inner.shard_world_size }
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
    
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    fn __getitem__(&self, idx: usize) -> PyResult<i32> {
        self.inner.get(idx).copied()
            .ok_or_else(|| PyRuntimeError::new_err(format!("Index {} out of range", idx)))
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
        // Helper closures with better error handling
        let g_usize = |k: &str, dv: usize| match d.get_item(k) {
            Ok(Some(val)) => {
                val.extract::<usize>().unwrap_or_else(|_| {
                    eprintln!("Warning: Invalid value for '{}', expected usize, using default {}", k, dv);
                    dv
                })
            }
            _ => dv,
        };
        let g_bool = |k: &str, dv: bool| match d.get_item(k) {
            Ok(Some(val)) => {
                val.extract::<bool>().unwrap_or_else(|_| {
                    eprintln!("Warning: Invalid value for '{}', expected bool, using default {}", k, dv);
                    dv
                })
            }
            _ => dv,
        };
        let g_u64 = |k: &str, dv: u64| match d.get_item(k) {
            Ok(Some(val)) => {
                val.extract::<u64>().unwrap_or_else(|_| {
                    eprintln!("Warning: Invalid value for '{}', expected u64, using default {}", k, dv);
                    dv
                })
            }
            _ => dv,
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

        // Check for unknown keys and warn about them
        let known_keys = [
            "batch_size", "drop_last", "shuffle", "seed", "num_workers", "prefetch", "auto_tune",
            "reader_mode", "part_size", "max_inflight_parts", "shard_rank", "shard_world_size",
            "worker_id", "num_workers_pytorch"
        ];
        for key in d.keys() {
            if let Ok(key_str) = key.extract::<&str>() {
                if !known_keys.contains(&key_str) {
                    eprintln!("Warning: Unknown option '{}' ignored. Valid options: {:?}", key_str, known_keys);
                }
            }
        }

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

            // AI/ML realism knobs (use defaults for now)
            pin_memory:           def.pin_memory,
            persistent_workers:   def.persistent_workers,
            timeout_seconds:      def.timeout_seconds,
            multiprocessing_context: def.multiprocessing_context,
            sampler_type:         def.sampler_type,
            memory_format:        def.memory_format,
            non_blocking:         def.non_blocking,
            generator_seed:       def.generator_seed,
            enable_transforms:    def.enable_transforms,
            collate_buffer_size:  def.collate_buffer_size,
            page_cache_mode:      def.page_cache_mode,  // Use default (Auto)
            adaptive:             def.adaptive,  // Use default (None/disabled)
            cancellation_token:   def.cancellation_token,  // Use default (None)
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
    #[pyo3(signature = (dataset, opts=None))]
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
    ) -> PyResult<Py<PyAny>> {
        let rx = slf.rx.clone();
        let bound_result = future_into_py(py, async move {
            let mut guard = rx.lock().await;
            match guard.recv().await {
                Some(Ok(batch)) => Python::with_gil(|py| batch.into_py_any(py)),
                Some(Err(e))    => Err(PyRuntimeError::new_err(format!("{:?}", e))),
                None            => Err(PyStopAsyncIteration::new_err("end of loader")),
            }
        })?;
        Ok(bound_result.unbind())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint Python bindings
// ---------------------------------------------------------------------------

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
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid strategy: {}", e)))?;
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
        .map(|opt_data| opt_data.map(|data| Python::with_gil(|py| PyBytesView::new(data).into_py_any(py).unwrap())))
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
    /// Write a chunk of data to the stream (zero-copy)
    #[pyo3(text_signature = "(self, data)")]
    fn write_chunk(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<usize> {
        if self.finalized {
            return Err(PyRuntimeError::new_err("Stream has been finalized"));
        }

        // Lazy initialization of the writer
        if self.writer.is_none() {
            let store = self.store.clone();
            let world_size = self.world_size;
            let rank = self.rank;
            let step = self.step;
            let _epoch = self.epoch; // May be used in future for enhanced metadata
            let _framework = self.framework.clone(); // May be used in future for enhanced metadata

            let (writer, key) = py.allow_threads(|| {
                self.runtime.block_on(async {
                    let store_guard = store.lock().await;
                    let ckpt_writer = store_guard.writer(world_size, rank);
                    let layout = crate::checkpoint::paths::KeyLayout::new(
                        "checkpoints".to_string(),
                        step,
                    );
                    ckpt_writer.get_shard_writer(&layout).await
                })
            }).map_err(|e| PyRuntimeError::new_err(format!("Failed to create writer: {}", e)))?;

            self.writer = Some(writer);
            self.key = Some(key);
        }

        // Write the chunk
        if let Some(ref mut writer) = self.writer {
            py.allow_threads(|| {
                self.runtime.block_on(async {
                    writer.write_chunk(data).await
                })
            }).map_err(|e| PyRuntimeError::new_err(format!("Write failed: {}", e)))?;

            Ok(data.len())
        } else {
            Err(PyRuntimeError::new_err("Writer initialization failed"))
        }
    }

    /// Get the number of bytes written so far
    #[pyo3(text_signature = "(self)")]
    fn bytes_written(&self) -> u64 {
        self.writer.as_ref().map_or(0, |w| w.bytes_written())
    }

    /// Finalize the stream and return shard metadata
    #[pyo3(text_signature = "(self)")]
    fn finalize(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if self.finalized {
            return Err(PyRuntimeError::new_err("Stream already finalized"));
        }

        if let (Some(writer), Some(key)) = (self.writer.take(), self.key.take()) {
            let store = self.store.clone();
            let world_size = self.world_size;
            let rank = self.rank;
            let step = self.step;
            let epoch = self.epoch;

            let shard_meta = py.allow_threads(|| {
                self.runtime.block_on(async {
                    // Finalize the writer
                    writer.finalize().await?;

                    // Create shard metadata
                    let store_guard = store.lock().await;
                    let ckpt_writer = store_guard.writer(world_size, rank);
                    let layout = crate::checkpoint::paths::KeyLayout::new(
                        "checkpoints".to_string(),
                        step,
                    );
                    ckpt_writer.finalize_shard_meta(&layout, key).await
                })
            }).map_err(|e| PyRuntimeError::new_err(format!("Finalize failed: {}", e)))?;

            self.finalized = true;

            // Return metadata as Python dict
            let dict = PyDict::new(py);
            dict.set_item("step", step).ok();
            dict.set_item("epoch", epoch).ok();
            dict.set_item("rank", shard_meta.rank).ok();
            dict.set_item("key", &shard_meta.key).ok();
            dict.set_item("size", shard_meta.size).ok();
            if let Some(etag) = &shard_meta.etag {
                dict.set_item("etag", etag).ok();
            }
            Ok(dict.into())
        } else {
            Err(PyRuntimeError::new_err("No data written to stream"))
        }
    }

    /// Cancel the stream and clean up
    #[pyo3(text_signature = "(self)")]
    fn cancel(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(writer) = self.writer.take() {
            py.allow_threads(|| {
                self.runtime.block_on(async {
                    writer.cancel().await
                })
            }).map_err(|e| PyRuntimeError::new_err(format!("Cancel failed: {}", e)))?;
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

        // Return zero-copy BytesView as PyObject
        Ok(PyBytesView::new(data).into_py_any(py)?)
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
#[cfg(feature = "extension-module")]
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
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    store.save(py, step, epoch, framework, data, user_meta)
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(text_signature = "(uri)")]
fn load_checkpoint(py: Python<'_>, uri: String) -> PyResult<Option<PyObject>> {
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    store.load_latest(py)
}

#[cfg(feature = "extension-module")]
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
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    let writer = store.writer(world_size, rank)?;
    writer.save_distributed_shard(py, step, epoch, framework, data)
}

#[cfg(feature = "extension-module")]
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
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    let writer = store.writer(world_size, rank)?;
    writer.finalize_distributed_checkpoint(py, step, epoch, framework, shard_metas, user_meta)
}

// ---------------------------------------------------------------------------
// Phase 3 Priority 4: Rich Python-Rust Data Exchange Enhancements
// ---------------------------------------------------------------------------

/// Enhanced checkpoint loading with integrity validation
#[pyfunction]
#[pyo3(signature = (uri, validate_integrity=true))]
pub fn load_checkpoint_with_validation(
    py: Python<'_>,
    uri: &str,
    validate_integrity: bool,
) -> PyResult<PyObject> {
    let store = store_for_uri(uri).map_err(py_err)?;

    py.allow_threads(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            if validate_integrity {
                // load_checkpoint_with_validation returns Vec<u8> for backward compatibility
                store.load_checkpoint_with_validation(uri, None).await
            } else {
                // get returns Bytes, convert to Vec<u8>
                store.get(uri).await.map(|b| b.to_vec())
            }
        })
    }).map_err(py_err)
    .map(|data| PyBytes::new(py, &data).into())
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
    Err(PyRuntimeError::new_err("save_numpy_array is temporarily disabled"))
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
) -> PyResult<PyObject> {
    Err(PyRuntimeError::new_err("load_numpy_array is temporarily disabled"))
}

/// Create a dataset from any supported URI scheme
#[pyfunction]
#[pyo3(signature = (uri, opts=None))]
pub fn create_dataset(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<PyDataset> {
    PyDataset::new(uri, opts)
}

/// Create an async data loader from any supported URI scheme (convenience function)
#[pyfunction]
#[pyo3(signature = (uri, opts=None))]
pub fn create_async_loader(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<PyBytesAsyncDataLoader> {
    let dataset = create_dataset(uri, opts.as_ref().map(|d| d.clone()))?;
    
    // For async loaders, default to batch_size = 1 for intuitive individual item iteration
    let mut loader_opts = opts_from_dict(opts);
    if loader_opts.batch_size == LoaderOptions::default().batch_size {
        // User didn't specify batch_size, default to 1 for async loaders
        loader_opts.batch_size = 1;
    }
    
    Ok(PyBytesAsyncDataLoader { dataset, opts: loader_opts })
}

// ---------------------------------------------------------------------------
// Compatibility shims for existing code
// ---------------------------------------------------------------------------

/// Compatibility wrapper - creates a PyDataset but only for S3 URIs
/// DEPRECATED: Use PyDataset or create_dataset() instead
#[pyclass]
#[derive(Clone)]
pub struct PyS3DatasetCompat {
    inner: PyDataset,
}

#[pymethods]
impl PyS3DatasetCompat {
    #[new]
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        use crate::object_store::infer_scheme;
        
        if !matches!(infer_scheme(uri), crate::object_store::Scheme::S3) {
            return Err(PyRuntimeError::new_err(
                "PyS3Dataset only supports S3 URIs. Use PyDataset for other schemes."
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
    fn new(uri: &str, opts: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        use crate::object_store::infer_scheme;
        
        if !matches!(infer_scheme(uri), crate::object_store::Scheme::S3) {
            return Err(PyRuntimeError::new_err(
                "PyS3AsyncDataLoader only supports S3 URIs. Use create_async_loader() for other schemes."
            ));
        }
        
        eprintln!("Warning: PyS3AsyncDataLoader is deprecated. Use create_async_loader() instead.");
        let loader = create_async_loader(uri, opts)?;
        Ok(Self { inner: loader })
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
pub fn create_tfrecord_index(
    tfrecord_path: &str,
    index_path: Option<&str>,
) -> PyResult<usize> {
    use crate::tfrecord_index::write_index_for_tfrecord_file;
    use std::path::Path;
    
    // Determine output path
    let output_path = match index_path {
        Some(p) => p.to_string(),
        None => format!("{}.idx", tfrecord_path),
    };
    
    // Create index
    let num_records = write_index_for_tfrecord_file(
        Path::new(tfrecord_path),
        Path::new(&output_path),
    ).map_err(py_err)?;
    
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
    let tuples: Vec<(u64, u64)> = entries
        .into_iter()
        .map(|e| (e.offset, e.size))
        .collect();
    
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
    let tuples: Vec<(u64, u64)> = entries
        .into_iter()
        .map(|e| (e.offset, e.size))
        .collect();
    
    Ok(tuples)
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

    // Compatibility classes (deprecated)
    m.add_class::<PyS3DatasetCompat>()?;
    m.add_class::<PyS3AsyncDataLoaderCompat>()?;

    // Dataset factory functions
    m.add_function(wrap_pyfunction!(create_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(create_async_loader, m)?)?;

    // TFRecord index generation (NVIDIA DALI compatible)
    m.add_function(wrap_pyfunction!(create_tfrecord_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_tfrecord_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(get_tfrecord_index_entries, m)?)?;
    m.add_function(wrap_pyfunction!(read_tfrecord_index, m)?)?;

    // Add compatibility aliases (these create the deprecated names that existing code expects)
    // The user should migrate to PyDataset and create_async_loader() instead
    m.add("PyS3AsyncDataLoader", m.getattr("PyS3AsyncDataLoaderCompat")?)?;
    
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
    }
    
    Ok(())
}

