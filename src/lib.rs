// src/lib.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
//
// Crate root — public re‑exports plus the Python module glue.

pub mod data_formats;
pub mod config;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

// Local files to use
pub mod s3_client;
pub mod s3_copy;
pub mod s3_utils;
pub mod data_gen;

pub mod s3_logger;
pub mod s3_ops;

pub use data_gen::generate_controlled_data;
pub use crate::s3_logger::{init_op_logger, global_logger, finalize_op_logger, Logger};

// ---------------------------------------------------------------------------
// Python bindings -----------------------------------------------------------
// ---------------------------------------------------------------------------
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
    upload,
    download,
    // asynchronous
    put_async_py,
    get_many_async_py,
    get_many_stats_async,
};

// Re‑export Rust helpers for the CLI and downstream Rust crates ------------
pub use s3_utils::{
    parse_s3_uri,
    list_objects,
    get_object_uri,
    stat_object_uri,
    get_objects_parallel,
    delete_objects,
    put_objects_with_random_data_and_type,
    DEFAULT_OBJECT_SIZE,
};

pub use crate::config::ObjectType;

// ---------------------------------------------------------------------------
// PyO3 module init -----------------------------------------------------------
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pymodule]
pub fn s3dlio(_py: Python, m: &PyModule) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;

    // async wrappers
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_stats_async, m)?)?;

    Ok(())
}

