// src/data_loader/mod.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Public API surface for the s3dlio data_loader layer.
pub mod async_pool_dataloader;
/// expose the `dataloader` module (file dataloader.rs)
pub mod dataloader;
pub mod dataset;
pub mod directio_bytes;
pub mod fs_bytes;
pub mod options;
pub mod prefetch;
pub mod s3_bytes;
pub mod sampler;
pub mod transform;

// Re‐export the key types at this level:
pub use async_pool_dataloader::{
    AsyncPoolDataLoader, MultiBackendDataset, PoolConfig, UnifiedDataLoader,
};
pub use dataloader::DataLoader;
pub use dataset::{Dataset, DatasetError};
pub use directio_bytes::DirectIOBytesDataset;
pub use fs_bytes::FileSystemBytesDataset;
pub use options::{
    LoaderOptions, LoadingMode, MemoryFormat, MultiprocessingContext, PageCacheMode, ReaderMode,
    SamplerType,
};
pub use s3_bytes::S3BytesDataset;
