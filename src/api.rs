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

/// URI scheme detection
pub use crate::object_store::{Scheme, infer_scheme};

// Factory functions for creating stores (backend-dependent)
#[cfg(feature = "native-backends")]
#[cfg(feature = "native-backends")]
/// Create an object store for the given URI (native AWS/Azure backend)
pub use crate::object_store::store_for_uri;

#[cfg(feature = "native-backends")]
/// Create a direct I/O optimized object store (native backend only)
pub use crate::object_store::direct_io_store_for_uri;

#[cfg(feature = "native-backends")]
/// Create a high-performance object store (native backend only)
pub use crate::object_store::high_performance_store_for_uri;

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

/// Convenience type alias for Results
pub type S3dlioResult<T> = Result<T>;

/// Advanced API for power users
pub mod advanced;

// Version information
/// Current API version
pub const API_VERSION: &str = "0.8.0";

/// Check if this API version is compatible with the given version
pub fn is_compatible_version(required: &str) -> bool {
    // Simple semantic version compatibility check
    let current_parts: Vec<u32> = API_VERSION.split('.').map(|s| s.parse().unwrap_or(0)).collect();
    let required_parts: Vec<u32> = required.split('.').map(|s| s.parse().unwrap_or(0)).collect();
    
    if current_parts.len() >= 2 && required_parts.len() >= 2 {
        // Major version must match, minor version must be >= required
        current_parts[0] == required_parts[0] && 
        current_parts[1] >= required_parts[1]
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_compatibility() {
        assert!(is_compatible_version("0.8.0"));
        assert!(is_compatible_version("0.7.0"));
        assert!(!is_compatible_version("1.0.0"));
        assert!(!is_compatible_version("0.9.0"));
    }
}
