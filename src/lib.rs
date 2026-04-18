// src/lib.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

#[cfg(all(feature = "native-backends", feature = "arrow-backend"))]
compile_error!("Enable only one of: native-backends or arrow-backend");

#[cfg(not(any(feature = "native-backends", feature = "arrow-backend")))]
compile_error!("Must enable either 'native-backends' or 'arrow-backend' feature");

// The 'gcs-community' feature is a legacy opt-in alternative (JSON API, no RAPID)
// selected under the shared `backend-gcs` backend gate.

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
// NOTE: sharded_client.rs and range_engine.rs (the S3-specific variant) were
// removed in v0.9.90.  They were never-completed stubs with no callers.  The
// production range engine is range_engine_generic.rs — see that file for context.
pub mod range_engine_generic;  // Universal stream-based range engine (v0.9.2+)
pub mod mp;
pub mod uri_utils;  // v0.9.14: Multi-endpoint URI expansion utilities
pub mod multi_endpoint;  // v0.9.14: Multi-endpoint load balancing with thread/process control

// Hardware detection and optimization (v0.9.35+: always available, runtime detection)
pub mod hardware;

// Performance monitoring
pub mod metrics;

// Profiling infrastructure (feature-gated)
pub mod profiling;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

// Internal modules - these may change in future versions
pub mod s3_client;
pub(crate) mod redirect_client;  // HTTP 307 redirect following (for AIStore compatibility)
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
#[cfg(feature = "backend-azure")]
pub mod azure_client;

// NUMA topology detection (optional, feature-gated)
#[cfg(feature = "numa")]
pub mod numa;

// Google Cloud Storage client - feature-gated backend selection
#[cfg(feature = "gcs-community")]
pub mod gcs_client;  // Community-maintained gcloud-storage implementation

#[cfg(feature = "backend-gcs")]
pub mod gcs_constants;      // Single source of truth for all GCS/gRPC tuning constants
#[cfg(feature = "backend-gcs")]
pub mod google_gcs_client;  // Official Google google-cloud-storage implementation

/// Pre-configure the number of gRPC subchannels (TCP connections) the GCS client
/// will open.  Call this once, before any GCS operation, to auto-tune throughput
/// to your concurrency level.  Matching channels to concurrent jobs ensures each
/// active stream has an uncontested HTTP/2 flow-control window.
///
/// **Call pattern**: invoke this at startup, before constructing any `ObjectStore`
/// or calling any `get`/`put` function.  The GCS client is a process-wide singleton;
/// this setting is only observed on first initialization.
///
/// Priority inside the GCS client:
///   1. `S3DLIO_GCS_GRPC_CHANNELS` env var  (always wins)
///   2. The value set here
///   3. `max(64, cpu_count)` auto fallback
pub fn set_gcs_channel_count(n: usize) {
    #[cfg(feature = "backend-gcs")]
    google_gcs_client::set_gcs_channel_count(n);
    #[cfg(not(feature = "backend-gcs"))]
    {
        let _ = n;
    }
}

/// Pre-configure RAPID (Hyperdisk ML / zonal GCS) mode before the first GCS
/// operation.  Must be called before any `gs://` I/O.
///
/// - `Some(true)`  — force RAPID on for all buckets
/// - `Some(false)` — force RAPID off for all buckets
/// - `None`        — auto-detect per bucket (default)
///
/// `S3DLIO_GCS_RAPID` env var still takes precedence if set.
pub fn set_gcs_rapid_mode(force: Option<bool>) {
    #[cfg(feature = "backend-gcs")]
    google_gcs_client::set_gcs_rapid_mode(force);
    #[cfg(not(feature = "backend-gcs"))]
    {
        let _ = force;
    }
}

/// Read back the programmatic GCS subchannel count.
///
/// Returns `0` if [`set_gcs_channel_count`] has not been called
/// (`S3DLIO_GCS_GRPC_CHANNELS` env var or auto-detect will be used on first
/// client initialization).
pub fn get_gcs_channel_count() -> usize {
    #[cfg(feature = "backend-gcs")]
    {
        return google_gcs_client::get_gcs_channel_count();
    }
    #[cfg(not(feature = "backend-gcs"))]
    {
        0
    }
}

/// Read back the current effective GCS RAPID mode setting.
///
/// Resolution includes the `S3DLIO_GCS_RAPID` env var (highest priority):
/// - `Some(true)`  — RAPID forced on
/// - `Some(false)` — RAPID forced off
/// - `None`        — auto-detect per bucket (default)
pub fn get_gcs_rapid_mode() -> Option<bool> {
    #[cfg(feature = "backend-gcs")]
    {
        return google_gcs_client::get_gcs_rapid_mode();
    }
    #[cfg(not(feature = "backend-gcs"))]
    {
        None
    }
}

/// Query whether a GCS bucket or `gs://` URI is a RAPID (Hyperdisk ML / zonal) bucket.
///
/// The result is cached for the process lifetime.  Accepts either a plain
/// bucket name or a full `gs://bucket/prefix/` URI.
/// Returns `false` on authentication/network errors (logs a warning).
pub async fn query_gcs_rapid_bucket(bucket_or_uri: &str) -> bool {
    #[cfg(feature = "backend-gcs")]
    {
        return google_gcs_client::query_gcs_rapid_bucket(bucket_or_uri).await;
    }
    #[cfg(not(feature = "backend-gcs"))]
    {
        let _ = bucket_or_uri;
        false
    }
}

pub mod list_containers;    // Backend-agnostic bucket/container listing (s3://, gs://, az://, file://)
pub use list_containers::{list_containers, ContainerInfo};

pub mod concurrency;
pub mod reqwest_client;  // Reqwest-based HTTP client for AWS SDK (h2c + connection pool tuning)
pub mod adaptive_config;  // Optional adaptive tuning for performance
mod multipart;

// ===== Legacy Re-exports for Backward Compatibility =====
// DEPRECATED FUNCTIONS REMOVED - Use fill_controlled_data() or data_gen_alt instead
// Old exports commented out to prevent accidental usage:
// #[allow(deprecated)]
// pub use data_gen::generate_controlled_data;  // REMOVED: Use fill_controlled_data instead
// #[allow(deprecated)]
// pub use data_gen::generate_controlled_data_prand;  // REMOVED: Use fill_controlled_data instead
pub use data_gen::fill_controlled_data;  // Fill-in-place (respects Rayon pool context)

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
    generic_upload_files_with_summary,
    generic_download_objects_with_summary,
    TransferSummary,
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
    Python::initialize();

    // Register all functions from modular API
    python_api::register_all_functions(m)?;

    Ok(())
}

