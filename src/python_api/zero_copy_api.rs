// src/python_api/zero_copy_api.rs
//
// Zero-Copy Python API for s3dlio
// Provides efficient data transfer between Python and Rust without copying
//
// ⚠️  WORK IN PROGRESS - NOT CURRENTLY ENABLED ⚠️
// This module contains advanced zero-copy API implementations but is disabled
// due to numpy dependency requirements. To enable:
// 1. Add numpy dependency to Cargo.toml 
// 2. Uncomment module in src/python_api.rs
// 3. Add registration calls in register_all_functions
//
// Status: Implementation complete, needs dependency integration
//

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyByteArray};
use pyo3::buffer::PyBuffer;
use pyo3::{PyObject, PyResult};
use numpy::{PyArray1, PyReadonlyArray1};

use std::collections::HashMap;
use std::sync::Arc;

// Import our core functionality
use crate::s3_utils::{parse_s3_uri, delete_objects};
use crate::object_store::ObjectStore;
use super::python_core_api::py_err;

// ---------------------------------------------------------------------------
// Zero-Copy Buffer Management
// ---------------------------------------------------------------------------

/// A zero-copy buffer that can be allocated by Rust and filled by Python
#[pyclass(name = "Buffer")]
pub struct PyBuffer {
    data: Vec<u8>,
    capacity: usize,
}

#[pymethods]
impl PyBuffer {
    /// Get the current length of data in the buffer
    fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Get the total capacity of the buffer
    fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get a memory view of the buffer for zero-copy access from Python
    fn memoryview(&self, py: Python) -> PyResult<PyObject> {
        // Create a memoryview that points to our Rust-allocated memory
        let bytes = PyBytes::new(py, &self.data);
        let memoryview = py.import("builtins")?.getattr("memoryview")?;
        memoryview.call1((bytes,))
    }
    
    /// Resize the buffer (this may copy if needed)
    fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, 0);
        if new_size > self.capacity {
            self.capacity = new_size;
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-Copy Put Operations
// ---------------------------------------------------------------------------

/// Put data with zero-copy from Python bytes
#[pyfunction]
pub fn put_bytes(uri: &str, data: &PyBytes) -> PyResult<()> {
    // PyBytes gives us a &[u8] view without copying
    let data_slice: &[u8] = data.as_bytes();
    put_data_internal(uri, data_slice)
}

/// Put data with zero-copy from NumPy array
#[pyfunction]
pub fn put_numpy(uri: &str, array: PyReadonlyArray1<u8>) -> PyResult<()> {
    // PyReadonlyArray1 gives us a zero-copy view of NumPy data
    let data_slice: &[u8] = array.as_slice()?;
    put_data_internal(uri, data_slice)
}

/// Put data with zero-copy from any buffer protocol object
#[pyfunction]
pub fn put_buffer(py: Python, uri: &str, buffer: PyObject) -> PyResult<()> {
    // Use PyO3's buffer protocol for maximum compatibility
    let buf: PyBuffer<u8> = PyBuffer::get(py, &buffer)?;
    
    // Ensure the buffer is contiguous for efficient access
    if !buf.is_c_contiguous() {
        return Err(py_err("Buffer must be C-contiguous for zero-copy operation"));
    }
    
    // Get a slice view without copying
    let data_slice = unsafe {
        std::slice::from_raw_parts(buf.buf_ptr(), buf.len_bytes())
    };
    
    put_data_internal(uri, data_slice)
}

/// Unified put function that accepts multiple Python data types
#[pyfunction]
pub fn put_data(py: Python, uri: &str, data: PyObject) -> PyResult<()> {
    // Try different zero-copy strategies based on the Python object type
    
    // Strategy 1: Python bytes (most common)
    if let Ok(bytes_obj) = data.downcast::<PyBytes>(py) {
        return put_bytes(uri, bytes_obj);
    }
    
    // Strategy 2: NumPy array
    if let Ok(array) = data.extract::<PyReadonlyArray1<u8>>(py) {
        return put_numpy(uri, array);
    }
    
    // Strategy 3: Buffer protocol (bytearray, memoryview, etc.)
    match PyBuffer::<u8>::get(py, &data) {
        Ok(buf) => {
            if !buf.is_c_contiguous() {
                return Err(py_err("Buffer must be C-contiguous for zero-copy operation"));
            }
            
            let data_slice = unsafe {
                std::slice::from_raw_parts(buf.buf_ptr(), buf.len_bytes())
            };
            
            put_data_internal(uri, data_slice)
        }
        Err(_) => {
            Err(py_err("Data must be bytes, numpy array, or support buffer protocol"))
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-Copy Get Operations
// ---------------------------------------------------------------------------

/// Get data with optional pre-allocated buffer for zero-copy
#[pyfunction]
pub fn get_into_buffer(py: Python, uri: &str, buffer: Option<PyObject>) -> PyResult<PyObject> {
    // First, determine the size of the object
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    let size = get_object_size(&bucket, &key).map_err(py_err)?;
    
    let target_buffer = if let Some(buf) = buffer {
        // Use provided buffer
        let pybuf: PyBuffer<u8> = PyBuffer::get(py, &buf)?;
        if pybuf.len_bytes() < size {
            return Err(py_err("Provided buffer is too small"));
        }
        buf
    } else {
        // Allocate a new buffer
        PyByteArray::new(py, &vec![0u8; size]).into()
    };
    
    // Get buffer view for writing
    let mut buf: PyBuffer<u8> = PyBuffer::get(py, &target_buffer)?;
    if !buf.is_c_contiguous() {
        return Err(py_err("Buffer must be C-contiguous"));
    }
    
    // Download directly into the buffer (zero-copy on Rust side)
    let data_slice = unsafe {
        std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut u8, buf.len_bytes())
    };
    
    get_data_into_slice(uri, data_slice)?;
    
    Ok(target_buffer)
}

/// Get data and return as Python bytes (one copy from Rust to Python)
#[pyfunction]
pub fn get_data(py: Python, uri: &str) -> PyResult<PyObject> {
    let data = get_data_internal(uri)?;
    // PyBytes::new creates a copy, but this is unavoidable for owned data
    Ok(PyBytes::new(py, &data).into())
}

/// Pre-allocate a buffer for zero-copy operations
#[pyfunction]
pub fn allocate_buffer(py: Python, size: usize) -> PyResult<PyObject> {
    // Allocate using Python's memory allocator for compatibility
    let buffer = PyByteArray::new(py, &vec![0u8; size]);
    Ok(buffer.into())
}

// ---------------------------------------------------------------------------
// Zero-Copy Streaming Interface
// ---------------------------------------------------------------------------

/// Streaming writer with zero-copy chunk processing
#[pyclass(name = "StreamWriter")]
pub struct PyStreamWriter {
    writer: Box<dyn crate::object_store::StreamWriter + Send>,
}

#[pymethods]
impl PyStreamWriter {
    /// Write a chunk with zero-copy
    pub fn write_chunk(&mut self, py: Python, data: PyObject) -> PyResult<()> {
        // Use the same zero-copy strategies as put_data
        
        if let Ok(bytes_obj) = data.downcast::<PyBytes>(py) {
            let data_slice: &[u8] = bytes_obj.as_bytes();
            return self.writer.write_chunk(data_slice).map_err(py_err);
        }
        
        if let Ok(array) = data.extract::<PyReadonlyArray1<u8>>(py) {
            let data_slice: &[u8] = array.as_slice()?;
            return self.writer.write_chunk(data_slice).map_err(py_err);
        }
        
        match PyBuffer::<u8>::get(py, &data) {
            Ok(buf) => {
                if !buf.is_c_contiguous() {
                    return Err(py_err("Buffer must be C-contiguous"));
                }
                
                let data_slice = unsafe {
                    std::slice::from_raw_parts(buf.buf_ptr(), buf.len_bytes())
                };
                
                self.writer.write_chunk(data_slice).map_err(py_err)
            }
            Err(_) => {
                Err(py_err("Data must be bytes, numpy array, or support buffer protocol"))
            }
        }
    }
    
    /// Finalize the stream and get statistics
    pub fn finalize(&mut self) -> PyResult<HashMap<String, u64>> {
        self.writer.finalize().map_err(py_err)
    }
}

/// Create a streaming writer for large uploads
#[pyfunction]
pub fn create_stream_writer(uri: &str, options: Option<super::python_core_api::PyWriterOptions>) -> PyResult<PyStreamWriter> {
    let writer = create_writer_internal(uri, options)?;
    Ok(PyStreamWriter { writer })
}

// ---------------------------------------------------------------------------
// Internal Implementation Functions
// ---------------------------------------------------------------------------

fn put_data_internal(uri: &str, data: &[u8]) -> PyResult<()> {
    // This borrows the data slice without copying
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    
    // Use our existing S3 put implementation with borrowed data
    crate::s3_ops::put_object_data(&bucket, &key, data).map_err(py_err)
}

fn get_data_internal(uri: &str) -> PyResult<Vec<u8>> {
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    crate::s3_ops::get_object(&bucket, &key).map_err(py_err)
}

fn get_data_into_slice(uri: &str, buffer: &mut [u8]) -> PyResult<usize> {
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    crate::s3_ops::get_object_into_buffer(&bucket, &key, buffer).map_err(py_err)
}

fn get_object_size(bucket: &str, key: &str) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // Implementation to get object size without downloading
    crate::s3_ops::head_object(bucket, key).map(|metadata| metadata.content_length as usize)
}

fn create_writer_internal(uri: &str, _options: Option<super::python_core_api::PyWriterOptions>) -> PyResult<Box<dyn crate::object_store::StreamWriter + Send>> {
    let (bucket, key) = parse_s3_uri(uri).map_err(py_err)?;
    crate::s3_ops::create_multipart_writer(&bucket, &key).map_err(py_err)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register_zero_copy_functions(py: Python, m: &PyModule) -> PyResult<()> {
    println!("Registering StreamWriter...");
    m.add_class::<PyStreamWriter>()?;
    println!("StreamWriter registered successfully.");
    Ok(())
}
