// src/python_api/test_hooks.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Test-only hook (issue #294): make "the Python call was made, but the runtime has not
//! polled its future yet" a provable state instead of a race.
//!
//! `s3dlio._hold_async_runtime()` occupies EVERY worker thread of the async runtime that polls
//! the futures of the `*_async` calls with a task that blocks, and returns only once all workers
//! are occupied. An awaitable created while the hold is active is spawned but cannot take its
//! first poll until the hold is released, so a test can run `finalize_op_log()` and
//! `init_op_log()` between the call and the first poll, deterministically.
//!
//! It is hidden (underscore name, not in `__all__`, not documented in the user guides) and
//! harmless on its own: the hold releases itself after [`FAILSAFE`] if nobody releases it. It
//! stalls only the caller's own async calls.

use pyo3::prelude::*;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// A hold that was never released ends by itself after this long.
const FAILSAFE: Duration = Duration::from_secs(60);
/// How long `_hold_async_runtime()` waits for every worker to be occupied.
const SATURATE_TIMEOUT: Duration = Duration::from_secs(20);

struct HoldState {
    released: bool,
    occupied: usize,
}

/// The handle of an active hold. `release()` (or leaving the `with` block) lets the occupied
/// workers go.
#[pyclass(name = "_AsyncRuntimeHold", module = "s3dlio")]
pub(crate) struct AsyncRuntimeHold {
    shared: Arc<(Mutex<HoldState>, Condvar)>,
}

#[pymethods]
impl AsyncRuntimeHold {
    /// Let the held workers go. Safe to call more than once.
    fn release(&self) {
        let (lock, cv) = &*self.shared;
        lock.lock().unwrap_or_else(|e| e.into_inner()).released = true;
        cv.notify_all();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.release();
    }
}

impl Drop for AsyncRuntimeHold {
    fn drop(&mut self) {
        self.release();
    }
}

/// Occupy every worker of the async runtime until the returned hold is released.
#[pyfunction]
#[pyo3(name = "_hold_async_runtime")]
fn hold_async_runtime(py: Python<'_>) -> PyResult<AsyncRuntimeHold> {
    let runtime = pyo3_async_runtimes::tokio::get_runtime();
    let workers = runtime.metrics().num_workers();
    let shared = Arc::new((
        Mutex::new(HoldState {
            released: false,
            occupied: 0,
        }),
        Condvar::new(),
    ));
    for _ in 0..workers {
        let shared = Arc::clone(&shared);
        runtime.spawn(async move {
            let (lock, cv) = &*shared;
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.occupied += 1;
            cv.notify_all();
            let deadline = Instant::now() + FAILSAFE;
            // Blocks this worker thread on purpose: that is the hold.
            while !st.released {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                st = cv
                    .wait_timeout(st, left)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        });
    }
    let hold = AsyncRuntimeHold {
        shared: Arc::clone(&shared),
    };
    let saturated = py.detach(|| {
        let (lock, cv) = &*shared;
        let deadline = Instant::now() + SATURATE_TIMEOUT;
        let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
        while st.occupied < workers {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            st = cv
                .wait_timeout(st, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    });
    if !saturated {
        hold.release();
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "_hold_async_runtime: the async runtime workers were not all occupied in time",
        ));
    }
    Ok(hold)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hold_async_runtime, m)?)?;
    m.add_class::<AsyncRuntimeHold>()?;
    Ok(())
}
