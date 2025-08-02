// src/data_loader/mod.rs

//! Public API surface for the s3dlio data_loader layer.
/// expose the `dataloader` module (file dataloader.rs)
pub mod dataloader;
pub mod dataset;
pub mod options;
pub mod sampler;
pub mod prefetch;
pub mod transform;


// Re‐export the key types at this level:
pub use dataloader::DataLoader;
//pub use dataset::{Dataset, DatasetError, DynStream};
pub use dataset::{Dataset, DatasetError};
pub use options::LoaderOptions;

