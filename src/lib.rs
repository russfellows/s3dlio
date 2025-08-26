// src/lib.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
//
// Crate root — public re-exports plus the Python module glue.

pub mod constants;
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
pub mod object_store;
pub mod file_store;
pub mod file_store_direct;

// (other modules/elided)
pub mod data_gen;
pub use data_gen::generate_controlled_data;

pub mod data_loader;
pub mod checkpoint;

// ===== Re-exports expected by tests/test_dataloader.rs at the crate root =====
// Types:
pub use crate::data_loader::dataloader::DataLoader;
pub use crate::data_loader::dataset::{Dataset, DatasetError};
pub use crate::data_loader::options::LoaderOptions;
// Module alias so tests can use `s3dlio::dataset::DynStream`:
pub use crate::data_loader::dataset;  // re-export the whole module as `s3dlio::dataset`
                                      //

// Re-export the Azure client module (always available now)
pub mod azure_client;

pub use object_store::{
    ObjectStore,
    S3ObjectStore,
    ObjectMetadata,
    infer_scheme,
    store_for_uri,
    Scheme,
};

pub use file_store::FileSystemObjectStore;

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

mod multipart;

// Re-export the multipart public types
pub use crate::multipart::{
    MultipartUploadConfig,
    MultipartUploadSink,
    MultipartCompleteInfo,
};

// types:
pub use crate::config::ObjectType;

// logger helpers:
pub use crate::s3_logger::{init_op_logger, finalize_op_logger};


// Bring in Python wrappers from python_api.rs when building as extension.
#[cfg(feature = "extension-module")]
mod python_api;


// ---------------------------------------------------------------------------
// PyO3 module init -----------------------------------------------------------
// ---------------------------------------------------------------------------
#[cfg(feature = "extension-module")]
#[pymodule]
pub fn _pymod(m: &Bound<PyModule>) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    // Register all functions from modular API
    python_api::register_all_functions(m)?;

    Ok(())
}

