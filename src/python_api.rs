// src/python_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use pyo3::prelude::*;

// Import all module APIs

mod async_bridge;
mod put_payload;
mod python_advanced_api;
mod python_aiml_api;
mod python_core_api;
mod python_datagen_api;

// Main module registration function
pub fn register_all_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Issue #192: wait for in-flight async result deliveries before the
    // interpreter finalizes (atexit runs before CPython marks the runtime as
    // finalizing).
    async_bridge::register_exit_drain(m)?;

    // Register functions from all modules
    python_core_api::register_core_functions(m)?;
    python_aiml_api::register_aiml_functions(m)?;
    python_advanced_api::register_advanced_functions(m)?;
    python_datagen_api::register_datagen_functions(m)?;

    Ok(())
}
