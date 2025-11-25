// src/api/advanced.rs
//! # Advanced S3DL-IO API
//!
//! This module provides advanced functionality for power users who need
//! fine-grained control over performance and behavior.
//!
//! ## Features
//!
//! - **Async Pool DataLoader**: High-concurrency async data loading
//! - **Direct I/O**: Zero-copy operations for maximum performance
//! - **Multipart Operations**: Advanced upload management
//! - **Custom Samplers**: Control data ordering and sampling
//!
//! ## Performance Considerations
//!
//! The advanced API provides more control but requires deeper understanding
//! of the underlying systems. Use the basic API unless you have specific
//! performance requirements.

// Advanced data loading
/// High-performance async data loader with connection pooling
pub use crate::data_loader::async_pool_dataloader::AsyncPoolDataLoader;

/// Multi-backend dataset for complex data sources
pub use crate::data_loader::async_pool_dataloader::MultiBackendDataset;

/// Configuration for connection pooling
pub use crate::data_loader::async_pool_dataloader::PoolConfig;

/// Unified data loader combining multiple strategies
pub use crate::data_loader::async_pool_dataloader::UnifiedDataLoader;

// Sampling and data ordering
/// Sampler trait for controlling data order
pub use crate::data_loader::sampler::Sampler;

/// Sequential data sampling
pub use crate::data_loader::sampler::SequentialSampler;

/// Randomized data sampling
pub use crate::data_loader::sampler::ShuffleSampler;

// Direct I/O and high-performance storage
/// Create a direct I/O optimized store
pub use crate::object_store::direct_io_store_for_uri;

/// Create store with custom filesystem configuration
pub use crate::object_store::store_for_uri_with_config;

// Multipart upload management
/// Configuration for multipart uploads
pub use crate::multipart::MultipartUploadConfig;

/// Sink for streaming multipart uploads
pub use crate::multipart::MultipartUploadSink;

/// Information about completed multipart uploads
pub use crate::multipart::MultipartCompleteInfo;

// Advanced checkpoint features
/// Checkpoint writer for custom checkpoint creation
pub use crate::checkpoint::writer::Writer as CheckpointWriter;

/// Checkpoint reader for custom checkpoint loading
pub use crate::checkpoint::reader::Reader as CheckpointReader;

/// Checkpoint path strategies
pub use crate::checkpoint::paths::{Strategy as CheckpointStrategy, KeyLayout};

/// Checkpoint manifest management
pub use crate::checkpoint::manifest::{Manifest, ShardMeta};

// Low-level S3 operations
/// Parallel object retrieval
pub use crate::s3_utils::get_objects_parallel;

/// Bucket management
pub use crate::s3_utils::{list_buckets, BucketInfo};

/// Object listing with advanced options
pub use crate::s3_utils::list_objects;

/// Bulk object deletion
pub use crate::s3_utils::delete_objects;

/// Range-based object retrieval
pub use crate::s3_utils::get_range;

/// Concurrent range downloads
pub use crate::s3_utils::get_object_concurrent_range;

// Performance profiling (when enabled)
#[cfg(feature = "profiling")]
pub use crate::profiling::*;

// Data generation utilities
/// Generate test data with controlled characteristics (new algorithm - high randomness)
pub use crate::data_gen::generate_controlled_data;

/// Generate test data with pseudo-random method (old algorithm - high performance)
pub use crate::data_gen::generate_controlled_data_prand;
