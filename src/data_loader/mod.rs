// src/data_loader/mod.rs

//! Public API surface for the s3dlio data_loader layer.
/// expose the `dataloader` module (file dataloader.rs)
pub mod dataloader;
pub mod dataset;
pub mod options;
pub mod sampler;
pub mod prefetch;
pub mod transform;
pub mod s3_bytes;
pub mod fs_bytes;
pub mod directio_bytes;
pub mod async_pool_dataloader;


// Re‐export the key types at this level:
pub use dataloader::DataLoader;
pub use dataset::{Dataset, DatasetError};
pub use options::{LoaderOptions, PageCacheMode, ReaderMode, LoadingMode, MultiprocessingContext, SamplerType, MemoryFormat};
pub use s3_bytes::S3BytesDataset;
pub use fs_bytes::FileSystemBytesDataset;
pub use directio_bytes::DirectIOBytesDataset;
pub use async_pool_dataloader::{AsyncPoolDataLoader, MultiBackendDataset, PoolConfig, UnifiedDataLoader};

