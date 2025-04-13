// src/lib.rs
//! Public re‑exports **plus** the Python module glue.
//! Crate root – re‑export helpers for the CLI and PyO3 bindings.

#[allow(unused_imports)]
use pyo3::prelude::*;

pub mod s3_utils;


// keep the python module for the wheel build
#[cfg(feature = "extension-module")]
mod python_api;

// Re‑export the PyO3 module so that its initialization function is at the crate root.
#[cfg(feature = "extension-module")]
pub use python_api::*;


pub use s3_utils::{
    parse_s3_uri,
    list_objects,
    get_object,
    get_object_uri,
    get_objects_parallel,
    delete_objects,
    put_object_with_random_data,
    put_objects_with_random_data,
    generate_random_data,
    DEFAULT_OBJECT_SIZE,
};


// ---------------------------------------------------------------------------
// Module definition ----------------------------------------------------------
// ---------------------------------------------------------------------------

#[cfg(feature = "extension-module")]
#[pymodule]
pub fn dlio_s3_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    m.add_function(wrap_pyfunction!(list, m)?)?;
    m.add_function(wrap_pyfunction!(get, m)?)?;
    m.add_function(wrap_pyfunction!(get_many, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?; // optional helper

    Ok(())
}



