//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
// src/lib.rs
//! Crate root — public re‑exports plus the Python module glue.

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

pub mod s3_utils;
pub mod data_gen;

// ---------------------------------------------------------------------------
// Python bindings -----------------------------------------------------------
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
mod python_api;

#[cfg(feature = "extension-module")]
use python_api::{
    // synchronous
    put, list, get_many, get, delete, read_npz,
    // asynchronous
    put_async_py, get_many_async_py,
};

// Re‑export Rust helpers for the CLI and downstream Rust crates -------------
pub use s3_utils::{
    parse_s3_uri,
    list_objects,
    get_object_uri,
    get_objects_parallel,
    delete_objects,
    put_objects_with_random_data_and_type,
    DEFAULT_OBJECT_SIZE,
    ObjectType,
};

// ---------------------------------------------------------------------------
// PyO3 module init -----------------------------------------------------------
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pymodule]
fn dlio_s3_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    // sync wrappers
    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;

    // async wrappers
    m.add_function(wrap_pyfunction!(put_async_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_many_async_py, m)?)?;

    Ok(())
}

