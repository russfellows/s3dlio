//! Core dataset abstractions for s3dlio’s high‑level data‑loader.
//!
//! Stage 1 gives you the minimum surface needed to iterate over samples
//! and (optionally) fetch them at random indices.  No shuffling,
//! prefetching, or multi‑threading is included yet—those arrive in the
//! next stage.

use async_trait::async_trait;
use futures_core::stream::Stream;
use std::pin::Pin;
use thiserror::Error;
use anyhow::{self, Error as AnyError}; // <-- bring Error type in

/// A boxed, pinned, sendable async stream of fallible items.
pub type DynStream<T> =
    Pin<Box<dyn Stream<Item = Result<T, DatasetError>> + Send + 'static>>;

/// Item‑level error type for dataset & loader operations.
#[derive(Error, Debug)] 
pub enum DatasetError {
    #[error("index out of range: {0}")]  // if using thiserror
    IndexOutOfRange(usize),

    #[error("operation not supported for this dataset type")]
    Unsupported,

    // NEW generic backend error
    #[error(transparent)]
    Backend(#[from] AnyError),
}


// Mapping from string to error
impl From<String> for DatasetError {
    fn from(s: String) -> Self {
        DatasetError::Backend(AnyError::msg(s))
    }
}

impl From<&str> for DatasetError {
    fn from(s: &str) -> Self {
        DatasetError::Backend(AnyError::msg(s.to_string()))
    }
}

/// A logical collection of **samples** (e.g. S3 objects, TFRecord
/// examples, rows of an HDF5 dataset).
///
/// Implementors fall into two broad categories:
///
/// * **Map‑style** – support random access through [`Dataset::get`];
///   `len()` normally returns `Some(_)`.
/// * **Iterable** – deliver data solely via `as_stream`; `len()` often
///   returns `None`.
#[async_trait]
pub trait Dataset: Send + Sync + 'static {
    /// Concrete Rust type produced for each sample.  For a raw object
    /// loader this might be `bytes::Bytes`; for parsed examples it could
    /// be `ndarray::Array` or your own struct.
    type Item: Send + 'static;

    /// Total number of samples if known *a priori*; otherwise `None`.
    fn len(&self) -> Option<usize>;

    /// Retrieve a sample by zero‑based index.  Iterable‑only datasets may
    /// return `DatasetError::Unsupported`.
    async fn get(&self, index: usize) -> Result<Self::Item, DatasetError>;

    /// Provide an async stream of samples if the dataset is iterable.
    /// Map‑style datasets can simply keep the default (`None`).
    fn as_stream(&self) -> Option<DynStream<Self::Item>> {
        None
    }

    /// Convenience helper.
    fn is_empty(&self) -> bool {
        self.len().map(|n| n == 0).unwrap_or(false)
    }
}

