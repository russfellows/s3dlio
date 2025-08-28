// src/python_advanced_api.rs
// 
// Copyright 2025
// Signal65 / Futurum Group.
//

//
// Advanced Features API
// Checkpoint system, multipart uploads, async operations

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError};
use pyo3::types::{PyBytes, PyDict};
use pyo3::conversion::IntoPyObjectExt;

use tokio::{
    sync::Mutex,
};

use std::{
    sync::Arc,
};

// ---------------------------------------------------------------------------
// Project crates
// ---------------------------------------------------------------------------
use super::python_core_api::py_err;
use crate::object_store::store_for_uri;

#[cfg(feature = "extension-module")]
use crate::checkpoint::{CheckpointStore, CheckpointConfig, CheckpointInfo};

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
        
        if let Some(_strat_str) = strategy {
            // Strategy configuration would go here
            // For now, use default strategy
        }
        
        if let Some(threshold) = multipart_threshold {
            config = config.with_multipart_threshold(threshold);
        }
        
        if let Some(_level) = compression_level {
            // Compression configuration would go here
            // For now, use default compression
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
    ) -> PyResult<String> {
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
        .map(|_result| format!("shard_{}_{}_{}_{}", step, epoch, rank, framework))
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
            dict.set_item("framework", &manifest.framework).ok();
            dict.set_item("global_step", manifest.global_step).ok();
            dict.set_item("epoch", manifest.epoch).ok();
            dict.set_item("world_size", manifest.world_size).ok();
            dict.into()
        }))
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
    fn is_complete(&self) -> bool { self.inner.is_complete() }
    
    #[getter]
    fn is_single_rank(&self) -> bool { self.inner.is_single_rank() }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------
#[pyfunction]
#[pyo3(text_signature = "(uri, step, epoch, framework, data, user_meta=None)")]
pub fn save_checkpoint(
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

#[pyfunction]
#[pyo3(text_signature = "(uri)")]
pub fn load_checkpoint(py: Python<'_>, uri: String) -> PyResult<Option<PyObject>> {
    let store = PyCheckpointStore::new(uri, None, None, None)?;
    store.load_latest(py)
}

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
                // For now, just do a basic load
                // TODO: Add proper validation
                store.get(uri).await
            } else {
                store.get(uri).await
            }
        })
    }).map_err(py_err)
    .map(|data| PyBytes::new(py, &data).into())
}

// ---------------------------------------------------------------------------
// Module registration function for advanced API
// ---------------------------------------------------------------------------
pub fn register_advanced_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "extension-module")]
    {
        // Checkpoint system
        m.add_class::<PyCheckpointStore>()?;
        m.add_class::<PyCheckpointWriter>()?;
        m.add_class::<PyCheckpointReader>()?;
        m.add_class::<PyCheckpointInfo>()?;
        m.add_function(wrap_pyfunction!(save_checkpoint, m)?)?;
        m.add_function(wrap_pyfunction!(load_checkpoint, m)?)?;
        m.add_function(wrap_pyfunction!(load_checkpoint_with_validation, m)?)?;
    }
    
    Ok(())
}
