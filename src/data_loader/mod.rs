// src/data_loader/mod.rs

//! Public API surface for the s3dlio data_loader layer.
/// expose the `dataloader` module (file dataloader.rs)
pub mod dataloader;

/// expose the `dataset` module (file dataset.rs)
pub mod dataset;

/// expose the `options` module (file options.rs)
pub mod options;

// Re‐export the key types at this level:
pub use dataset::{Dataset, DatasetError, DynStream};
pub use dataloader::DataLoader;
pub use options::LoaderOptions;

