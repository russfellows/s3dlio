// src/lib.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
//
// Crate root — public re-exports plus the Python module glue.

pub mod data_formats;
pub mod config;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

// Local files to use
pub mod s3_client;
pub mod s3_copy;
pub mod s3_utils;
pub mod s3_logger;
pub mod s3_ops;

// (other modules/elided)
pub mod data_gen;
pub use data_gen::generate_controlled_data;

pub mod data_loader;
// ===== Re-exports expected by tests/test_dataloader.rs at the crate root =====
// Types:
pub use crate::data_loader::dataloader::DataLoader;
pub use crate::data_loader::dataset::{Dataset, DatasetError};
pub use crate::data_loader::options::LoaderOptions;
// Module alias so tests can use `s3dlio::dataset::DynStream`:
pub use crate::data_loader::dataset;  // re-export the whole module as `s3dlio::dataset`

// ===== Re-exports expected by src/bin/cli.rs at the crate root =====
// s3_utils items:
pub use crate::s3_utils::{
    delete_objects,
    get_object_uri,
    get_objects_parallel,
    list_objects,
    parse_s3_uri,
    stat_object_uri,
    put_objects_with_random_data_and_type,
    DEFAULT_OBJECT_SIZE,
};

// types:
pub use crate::config::ObjectType;

// logger helpers:
pub use crate::s3_logger::{init_op_logger, finalize_op_logger};


// Bring in Python wrappers from python_api.rs when building as extension.
#[cfg(feature = "extension-module")]
mod python_api;

#[cfg(feature = "extension-module")]
use python_api::{
    // synchronous
    put,
    list,
    stat,
    get,
    get_many,
    get_many_stats,
    delete,
    read_npz,
    init_logging,
    init_op_log,
    finalize_op_log,
    upload,
    download,

    // async
    put_async_py,
    get_many_async_py,
    get_many_stats_async,
};

// ---------------------------------------------------------------------------
// PyO3 module init -----------------------------------------------------------
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pymodule]
pub fn s3dlio(m: &Bound<PyModule>) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    // sync wrappers
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(stat, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(init_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_op_log, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;

    // async wrappers
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;

    // -- Add async DataLoader classes for Python API --
    m.add_class::<python_api::PyVecDataset>()?;
    m.add_class::<python_api::PyAsyncDataLoader>()?;
    m.add_class::<python_api::PyAsyncDataLoaderIter>()?;

    Ok(())
}

