// src/api.rs
//! # S3DL-IO Public API
//!
//! This module provides the clean, stable public API for the s3dlio library.
//! All items in this module are guaranteed to maintain backward compatibility
//! within major versions.
//!
//! ## Core Concepts
//!
//! - **ObjectStore**: Unified interface for cloud storage operations
//! - **DataLoader**: High-performance data loading for ML/AI workloads  
//! - **Streaming**: Efficient streaming read/write operations
//! - **Checkpointing**: State management for long-running operations
//!
//! ## Quick Start
//!
//! ```rust
//! use s3dlio::api::{ObjectStore, store_for_uri};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a store for any supported URI
//! let store = store_for_uri("s3://mybucket/path/")?;
//!
//! // Read an object
//! let data = store.get("file.txt").await?;
//!
//! // Write an object with streaming
//! let mut writer = store.create_writer("output.txt", Default::default()).await?;
//! writer.write_chunk(b"Hello, world!").await?;
//! writer.finalize().await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;

// Re-export core types from internal modules with clean names
// This creates a stable API facade over our internal implementation

/// Core object storage interface
pub use crate::object_store::ObjectStore;

/// Streaming writer interface for efficient uploads
pub use crate::object_store::ObjectWriter;

/// Configuration for writer behavior
pub use crate::object_store::WriterOptions;

/// Compression options for storage operations
pub use crate::object_store::CompressionConfig;

/// Object metadata information
pub use crate::object_store::ObjectMetadata;

/// FileSystem-specific configuration (page cache, range engine, etc.)
pub use crate::file_store::FileSystemConfig;

/// Page cache behavior hint for file I/O (maps to posix_fadvise on Linux/Unix)
pub use crate::data_loader::options::PageCacheMode;

/// URI scheme detection
pub use crate::object_store::{Scheme, infer_scheme};

// Op-log (trace logging) support
/// Initialize operation logging to a file
pub use crate::s3_logger::{init_op_logger, finalize_op_logger, global_logger, Logger};

/// Set clock offset for distributed op-log synchronization (issue #100)
pub use crate::s3_logger::{set_clock_offset, get_clock_offset};

/// Set client ID for operation logging (v0.9.22)
/// All operations logged after this call will use the specified client_id value
pub use crate::s3_logger::{set_client_id, get_client_id};

/// Wrapper that adds logging to any ObjectStore
pub use crate::object_store_logger::LoggedObjectStore;

// Factory functions for creating stores (backend-dependent)
#[cfg(feature = "native-backends")]
/// Create an object store for the given URI (native AWS/Azure backend)
pub use crate::object_store::store_for_uri;

#[cfg(feature = "native-backends")]
/// Create an object store with optional operation logging
pub use crate::object_store::store_for_uri_with_logger;

#[cfg(feature = "native-backends")]
/// Create an object store with FileSystemConfig (for page cache control, range engine, etc.)
pub use crate::object_store::store_for_uri_with_config;

#[cfg(feature = "native-backends")]
/// Create an object store with configuration and optional logging
pub use crate::object_store::store_for_uri_with_config_and_logger;

#[cfg(feature = "native-backends")]
/// Create a direct I/O optimized object store (native backend only)
pub use crate::object_store::direct_io_store_for_uri;

#[cfg(feature = "native-backends")]
/// Create a direct I/O store with optional logging
pub use crate::object_store::direct_io_store_for_uri_with_logger;

#[cfg(feature = "native-backends")]
/// Create a high-performance object store (native backend only)
pub use crate::object_store::high_performance_store_for_uri;

#[cfg(feature = "native-backends")]
/// Create a high-performance store with optional logging
pub use crate::object_store::high_performance_store_for_uri_with_logger;

#[cfg(feature = "arrow-backend")]
/// Create an object store for the given URI (Apache Arrow backend)
pub use crate::object_store_arrow::store_for_uri;

// Data loading interface
/// Main data loader for ML/AI workloads
pub use crate::data_loader::dataloader::DataLoader;

/// Dataset abstraction for data sources
pub use crate::data_loader::dataset::Dataset;

/// Configuration options for data loading
pub use crate::data_loader::options::LoaderOptions;

/// Data loading modes and reader configurations
pub use crate::data_loader::options::{ReaderMode, LoadingMode};

// Checkpoint system for state management
/// Checkpoint store for saving/loading state
pub use crate::checkpoint::CheckpointStore;

/// Configuration for checkpoint behavior
pub use crate::checkpoint::CheckpointConfig;

/// Information about saved checkpoints
pub use crate::checkpoint::CheckpointInfo;

// Adaptive performance tuning (optional, opt-in)
/// Adaptive configuration for optional auto-tuning
pub use crate::adaptive_config::AdaptiveConfig;

/// Adaptive tuning mode (Enabled/Disabled)
pub use crate::adaptive_config::AdaptiveMode;

/// Workload type hints for adaptive optimization
pub use crate::adaptive_config::WorkloadType;

/// Compute adaptive parameters based on workload
pub use crate::adaptive_config::AdaptiveParams;

// Essential utilities
/// Parse S3 URI into bucket and key components
pub use crate::s3_utils::parse_s3_uri;

/// Get object metadata/statistics
pub use crate::s3_utils::stat_object_uri;

/// Object metadata structure
pub use crate::s3_utils::ObjectStat;

// Error types
/// Main error type for s3dlio operations
pub use anyhow::Error;

/// Dataset-specific errors
pub use crate::data_loader::dataset::DatasetError;

/// S3 bytes dataset
pub use crate::data_loader::s3_bytes::S3BytesDataset;

/// File system bytes dataset
pub use crate::data_loader::fs_bytes::FileSystemBytesDataset;

/// Direct I/O bytes dataset for maximum performance
pub use crate::data_loader::directio_bytes::DirectIOBytesDataset;

/// Convenience type alias for Results
pub type S3dlioResult<T> = Result<T>;

/// Create a dataset from a URI with default options
pub fn dataset_for_uri(uri: &str) -> Result<Box<dyn Dataset<Item = Vec<u8>>>> {
    dataset_for_uri_with_options(uri, &LoaderOptions::default())
}

/// Create a dataset from a URI with custom options
pub fn dataset_for_uri_with_options(uri: &str, opts: &LoaderOptions) -> Result<Box<dyn Dataset<Item = Vec<u8>>>> {
    use crate::object_store::infer_scheme;
    
    match infer_scheme(uri) {
        crate::object_store::Scheme::S3 => {
            let dataset = S3BytesDataset::from_prefix_with_opts(uri, opts)
                .map_err(|e| anyhow::anyhow!("Failed to create S3 dataset: {}", e))?;
            Ok(Box::new(dataset))
        }
        crate::object_store::Scheme::File => {
            let dataset = FileSystemBytesDataset::from_uri_with_opts(uri, opts)
                .map_err(|e| anyhow::anyhow!("Failed to create file system dataset: {}", e))?;
            Ok(Box::new(dataset))
        }
        crate::object_store::Scheme::Azure => {
            // TODO: Implement Azure dataset
            anyhow::bail!("Azure datasets not yet implemented")
        }
        crate::object_store::Scheme::Direct => {
            let dataset = DirectIOBytesDataset::from_uri_with_opts(uri, opts)
                .map_err(|e| anyhow::anyhow!("Failed to create Direct I/O dataset: {}", e))?;
            Ok(Box::new(dataset))
        }
        crate::object_store::Scheme::Gcs => {
            // TODO: Implement GCS dataset
            anyhow::bail!("GCS datasets not yet implemented")
        }
        crate::object_store::Scheme::Unknown => {
            anyhow::bail!("Unable to infer dataset type from URI: {}", uri)
        }
    }
}

/// Advanced API for power users
pub mod advanced;

// Version information
/// Current API version
pub const API_VERSION: &str = "0.9.21";

/// Check if this API version is compatible with the given version
/// Uses semantic versioning: major.minor.patch
/// Compatible if: same major version AND same minor version (patch can differ)
pub fn is_compatible_version(required: &str) -> bool {
    let current_parts: Vec<u32> = API_VERSION.split('.').map(|s| s.parse().unwrap_or(0)).collect();
    let required_parts: Vec<u32> = required.split('.').map(|s| s.parse().unwrap_or(0)).collect();
    
    if current_parts.len() >= 2 && required_parts.len() >= 2 {
        // Major and minor versions must match exactly
        current_parts[0] == required_parts[0] && 
        current_parts[1] == required_parts[1]
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_compatibility() {
        // Current version is 0.9.21, so 0.9.x versions are compatible
        assert!(is_compatible_version("0.9.0"));
        assert!(is_compatible_version("0.9.21"));
        assert!(is_compatible_version("0.9.10"));
        // 0.8.x and earlier are not compatible (different minor version)
        assert!(!is_compatible_version("0.8.0"));
        assert!(!is_compatible_version("0.7.0"));
        // 1.0.x is not compatible (different major version)
        assert!(!is_compatible_version("1.0.0"));
    }
}
