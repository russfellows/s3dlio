// src/lib.rs
//
// Copyright, 2025.  Signal65 / Futurum Group.
//
// Crate root — public re-exports plus the Python module glue.

// Compile-time feature compatibility check
#[cfg(all(feature = "native-backends", feature = "arrow-backend"))]
compile_error!("Enable only one of: native-backends or arrow-backend");

#[cfg(not(any(feature = "native-backends", feature = "arrow-backend")))]
compile_error!("Must enable either 'native-backends' or 'arrow-backend' feature");

// ===== Core Public API =====
// This is the main stable API that external users should use
pub mod api;

// Re-export the main API at the crate root for convenience
pub use api::*;

// ===== Internal Modules (Implementation) =====
// These are public for internal use but may change without notice

// ===== Internal Modules (Implementation) =====
// These are public for internal use but may change without notice

pub mod constants;
pub mod data_formats;
pub mod config;
pub mod progress;
pub mod memory;
pub mod download;
pub mod sharded_client;
pub mod range_engine;
pub mod mp;

// Profiling infrastructure (feature-gated)
pub mod profiling;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

// Internal modules - these may change in future versions
pub mod s3_client;
pub mod s3_copy;
pub mod s3_utils;
pub mod s3_logger;
pub mod s3_ops;
pub mod object_store;

// Arrow backend (optional, feature-gated)
#[cfg(feature = "arrow-backend")]
pub mod object_store_arrow;

pub mod file_store;
pub mod file_store_direct;
pub mod data_gen;
pub mod streaming_writer;
pub mod data_loader;
pub mod checkpoint;
pub mod azure_client;
pub mod concurrency;
#[cfg(feature = "enhanced-http")]
pub mod http;
pub mod performance;
mod multipart;

// ===== Legacy Re-exports for Backward Compatibility =====
// These maintain compatibility with existing code but may be deprecated
pub use data_gen::generate_controlled_data;

// ===== Re-exports expected by tests/test_dataloader.rs at the crate root =====
// Types:
pub use crate::data_loader::dataloader::DataLoader;
pub use crate::data_loader::dataset::{Dataset, DatasetError};
pub use crate::data_loader::options::LoaderOptions;
// Module alias so tests can use `s3dlio::dataset::DynStream`:
pub use crate::data_loader::dataset;  // re-export the whole module as `s3dlio::dataset`

// Common object store types (available for both backends)
pub use object_store::{
    ObjectStore,
    ObjectMetadata,
    infer_scheme,
    Scheme,
};

// Native backend specific exports
#[cfg(feature = "native-backends")]
pub use object_store::{
    S3ObjectStore,
    store_for_uri,
    store_for_uri_with_config,
    direct_io_store_for_uri,
    high_performance_store_for_uri,
    generic_upload_files,
    generic_download_objects,
};

// Arrow backend specific exports
#[cfg(feature = "arrow-backend")]
pub use object_store_arrow::store_for_uri;

pub use file_store::FileSystemObjectStore;

// ===== Re-exports expected by src/bin/cli.rs at the crate root =====
// s3_utils items:
pub use crate::s3_utils::{
    delete_objects,
    get_object_uri,
    get_objects_parallel,
    list_objects,
    list_buckets,
    BucketInfo,
    parse_s3_uri,
    stat_object_uri,
    put_objects_with_random_data_and_type,
    DEFAULT_OBJECT_SIZE,
};

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

