// src/python_api.rs
// 
// Copyright 2025
// Signal65 / Futurum Group.
//

//
// Main Python API - imports and re-exports from modular structure
//

use pyo3::prelude::*;

// Import all module APIs


mod python_core_api;
mod python_aiml_api;
mod python_advanced_api;
// NOTE: zero_copy_api.rs contains valuable zero-copy implementations but is disabled
// due to numpy dependency. See zero_copy_api.rs header for enabling instructions.
// mod zero_copy_api;

// Main module registration function
pub fn register_all_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register functions from all modules
    python_core_api::register_core_functions(m)?;
    python_aiml_api::register_aiml_functions(m)?;
    python_advanced_api::register_advanced_functions(m)?;
    
    Ok(())
}
