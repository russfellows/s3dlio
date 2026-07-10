// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Core dataset abstractions for s3dlio’s high‑level data‑loader.
//!
//! Stage 1 gives you the minimum surface needed to iterate over samples
//! and (optionally) fetch them at random indices.  No shuffling,
//! prefetching, or multi‑threading is included yet—those arrive in the
//! next stage.

use anyhow::{self, Error as AnyError};
use async_trait::async_trait;
use futures_core::stream::Stream;
use std::pin::Pin;
use thiserror::Error; // <-- bring Error type in

/// A boxed, pinned, sendable async stream of fallible items.
pub type DynStream<T> = Pin<Box<dyn Stream<Item = Result<T, DatasetError>> + Send + 'static>>;

/// Item‑level error type for dataset & loader operations.
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("index out of range: {0}")] // if using thiserror
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

    /// Return the keys/identifiers for all items in the dataset.
    /// For object storage, these are typically object keys or file paths.
    /// Returns `None` if the dataset doesn't support key enumeration.
    fn keys(&self) -> Option<Vec<String>> {
        None
    }

    /// Convenience helper.
    fn is_empty(&self) -> bool {
        self.len().map(|n| n == 0).unwrap_or(false)
    }
}

/// Message shared by every FFI entry point that requires a known dataset
/// length before proceeding (`PyDataset::__iter__`, `.items()`,
/// `PyBytesAsyncDataLoaderIter::spawn_stream`) -- design doc
/// docs/DESIGN_TIER4_FFI_HARDENING.md item 2.
pub const UNKNOWN_LENGTH_MSG: &str =
    "this dataset has unknown length -- no streaming iteration path is currently exposed to Python";

/// Guard used at every FFI entry point above: `Ok(n)` when the length is
/// known, `Err(UNKNOWN_LENGTH_MSG)` when it's `None`. Pure/GIL-free so it's
/// unit-testable without the `extension-module` feature (which cannot link
/// for `cargo test`).
pub fn require_known_length(len: Option<usize>) -> Result<usize, &'static str> {
    len.ok_or(UNKNOWN_LENGTH_MSG)
}

#[cfg(test)]
mod tier4_length_guard_tests {
    use super::*;

    #[test]
    fn known_length_passes_through() {
        assert_eq!(require_known_length(Some(42)), Ok(42));
        assert_eq!(require_known_length(Some(0)), Ok(0));
    }

    #[test]
    fn unknown_length_is_rejected_not_defaulted_to_zero() {
        assert_eq!(require_known_length(None), Err(UNKNOWN_LENGTH_MSG));
    }
}

/// Blocking-receive helper shared by every `std::sync::Mutex`-guarded sync
/// iterator (`PyObjectDataLoaderSyncIter`, `PyBytesDataLoaderSyncIter`,
/// `ParquetStreamIter`) -- design doc docs/DESIGN_TIER4_FFI_HARDENING.md
/// item 4, Option C. `parking_lot::Mutex` never poisons, so an unrelated
/// panic elsewhere while the lock happens to be held no longer wedges the
/// iterator for the rest of its lifetime the way `std::sync::Mutex` +
/// `.expect(...)` did. Pure/GIL-free (the caller is always a plain
/// synchronous `py.detach(|| ...)` closure, never an `.await` point) so
/// it's unit-testable without the `extension-module` feature.
pub fn blocking_recv_locked<T>(
    mtx: &parking_lot::Mutex<tokio::sync::mpsc::Receiver<T>>,
) -> Option<T> {
    mtx.lock().blocking_recv()
}

#[cfg(test)]
mod tier4_mutex_poisoning_tests {
    use super::*;

    /// Characterizes the hazard described in item 4: `std::sync::Mutex`
    /// (the type used by all 3 affected iterators before this fix) poisons
    /// itself if any thread panics while holding the lock, and every
    /// subsequent `.lock()` on that same mutex then fails for the rest of
    /// its lifetime -- exactly what the removed `.expect("... poisoned")`
    /// calls would have turned into a permanent `PanicException`.
    #[test]
    fn std_mutex_poisons_and_wedges_after_panic() {
        let mtx = std::sync::Mutex::new(0i32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mtx.lock().unwrap();
            panic!("simulated panic while holding the lock");
        }));
        assert!(
            mtx.lock().is_err(),
            "expected std::sync::Mutex to be poisoned after a panic while held"
        );
    }

    /// The actual replacement helper (`blocking_recv_locked` over
    /// `parking_lot::Mutex`) must survive the identical scenario: a panic
    /// on another thread while the lock is held must not wedge later calls.
    #[test]
    fn blocking_recv_locked_survives_panic_while_held() {
        let (tx, rx) = tokio::sync::mpsc::channel::<i32>(4);
        let mtx = std::sync::Arc::new(parking_lot::Mutex::new(rx));

        let mtx_for_panic = std::sync::Arc::clone(&mtx);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mtx_for_panic.lock();
            panic!("simulated panic while holding the lock");
        }));

        // The mutex must still be usable -- no poisoning, no wedge. Mirrors
        // the real call sites: `blocking_recv_locked` runs on a plain OS
        // thread (inside `py.detach`), never inside the runtime it's
        // receiving from.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            let _ = tx.send(7).await;
        });
        assert_eq!(blocking_recv_locked(&mtx), Some(7));
    }
}
