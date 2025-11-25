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

// GCS backend selection - choose one implementation
#[cfg(all(feature = "gcs-community", feature = "gcs-official"))]
compile_error!("Enable only one of: gcs-community or gcs-official");

#[cfg(not(any(feature = "gcs-community", feature = "gcs-official")))]
compile_error!("Must enable either 'gcs-community' or 'gcs-official' feature for GCS support");

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
pub mod object_size_cache;  // v0.9.10: Pre-stat size caching for benchmarking
pub mod download;
pub mod sharded_client;
pub mod range_engine;
pub mod range_engine_generic;  // Universal stream-based range engine (v0.9.2+)
pub mod mp;
pub mod uri_utils;  // v0.9.14: Multi-endpoint URI expansion utilities
pub mod multi_endpoint;  // v0.9.14: Multi-endpoint load balancing with thread/process control

// Performance monitoring
pub mod metrics;

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
pub mod page_cache;
pub mod data_gen;
pub mod data_gen_alt;  // Alternative data generation with ChaCha20
pub mod streaming_writer;
pub mod tfrecord_index;
pub mod object_store_logger;  // Op-log support for all backends
pub mod data_loader;
pub mod checkpoint;
pub mod azure_client;

// Google Cloud Storage client - feature-gated backend selection
#[cfg(feature = "gcs-community")]
pub mod gcs_client;  // Community-maintained gcloud-storage implementation

#[cfg(feature = "gcs-official")]
pub mod google_gcs_client;  // Official Google google-cloud-storage implementation

pub mod concurrency;
#[cfg(feature = "enhanced-http")]
pub mod http;
pub mod performance;
pub mod adaptive_config;  // Optional adaptive tuning for performance
mod multipart;

// ===== Legacy Re-exports for Backward Compatibility =====
// These maintain compatibility with existing code but may be deprecated
pub use data_gen::generate_controlled_data;
pub use data_gen::generate_controlled_data_prand;  // Pseudo-random (fast) method

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
    store_for_uri_with_high_performance_cloud,
    generic_upload_files,
    generic_download_objects,
};

// Arrow backend specific exports
#[cfg(feature = "arrow-backend")]
pub use object_store_arrow::store_for_uri;

pub use file_store::FileSystemObjectStore;

// Multi-endpoint support (v0.9.14+)
pub use multi_endpoint::{
    MultiEndpointStore,
    MultiEndpointStoreConfig,
    EndpointConfig,
    LoadBalanceStrategy,
    EndpointStats,
    EndpointStatsSnapshot,
};

// ===== Re-exports expected by src/bin/cli.rs at the crate root =====
// s3_utils items:
pub use crate::s3_utils::{
    delete_objects,
    get_object_uri,
    get_objects_parallel,
    get_objects_parallel_with_progress,
    list_objects,
    list_buckets,
    BucketInfo,
    parse_s3_uri,
    parse_s3_uri_full,
    S3UriComponents,
    stat_object_uri,
    put_objects_with_random_data_and_type,
    put_objects_with_random_data_and_type_with_progress,
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
pub use crate::s3_logger::{init_op_logger, finalize_op_logger, set_clock_offset, get_clock_offset};


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

