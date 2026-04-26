// src/python_api/python_datagen_api.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Zero-copy Python bindings for data generation using PyO3 buffer protocol
//!
//! This module provides TRUE zero-copy data generation for Python by implementing
//! the buffer protocol. Python code can use memoryview() for zero-copy access.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;

use super::python_core_api::PyBytesView;
use crate::data_gen_alt::{default_data_gen_threads, total_cpus, DataGenerator};

// =============================================================================
// Simple API - Single-call data generation
// =============================================================================

/// Generate random data with controllable deduplication and compression (ZERO-COPY)
///
/// # Arguments
/// * `size` - Total bytes to generate
/// * `dedup` - Deduplication factor (1 = no dedup, 2 = 2:1 ratio)
/// * `compress` - Compression factor (1 = incompressible, 3 = 3:1 ratio)
///
/// # Returns
/// BytesView object supporting buffer protocol for zero-copy access via memoryview()
///
/// # Example
/// ```python
/// import s3dlio
///
/// # Generate 1 MiB incompressible data (uses default 50% of CPUs)
/// data = s3dlio.generate_data(1024 * 1024)
///
/// # Use memoryview for ZERO-COPY access
/// view = memoryview(data)
/// print(f"Generated {len(view)} bytes, zero copies!")
///
/// # Or convert to bytes if needed (this DOES copy)
/// data_bytes = bytes(data)
/// ```
#[pyfunction]
#[pyo3(signature = (size, dedup=1, compress=1))]
fn generate_data(
    py: Python<'_>,
    size: usize,
    dedup: usize,
    compress: usize,
) -> PyResult<Py<PyBytesView>> {
    // Generate data WITHOUT holding GIL (allows parallel Python threads)
    let buffer = py.detach(|| {
        use crate::data_gen_alt::{generate_data as gen_data, GeneratorConfig, NumaMode};

        let config = GeneratorConfig {
            size,
            dedup_factor: dedup,
            compress_factor: compress,
            numa_mode: NumaMode::Auto,
            max_threads: Some(default_data_gen_threads()),
            numa_node: None,
            block_size: None,
            seed: None,
        };

        gen_data(config) // Returns DataBuffer directly - NO copies!
    });

    // Convert to Bytes (zero-copy for Uma/Vec via Bytes::from(vec)) and wrap
    Py::new(py, PyBytesView::new(buffer.into_bytes()))
}

/// Generate random data with custom thread count (ZERO-COPY)
///
/// # Arguments
/// * `size` - Total bytes to generate
/// * `dedup` - Deduplication factor
/// * `compress` - Compression factor
/// * `threads` - Number of threads (None = use default_data_gen_threads)
///
/// # Returns
/// BytesView object supporting buffer protocol for zero-copy access
///
/// # Example
/// ```python
/// import s3dlio
///
/// # Generate with 8 threads
/// data = s3dlio.generate_data_with_threads(1024 * 1024, threads=8)
/// view = memoryview(data)  # Zero-copy access!
/// ```
#[pyfunction]
#[pyo3(signature = (size, dedup=1, compress=1, threads=None))]
fn generate_data_with_threads(
    py: Python<'_>,
    size: usize,
    dedup: usize,
    compress: usize,
    threads: Option<usize>,
) -> PyResult<Py<PyBytesView>> {
    let num_threads = threads.unwrap_or_else(default_data_gen_threads);

    // Generate with custom thread count, WITHOUT holding GIL
    let buffer = py.detach(|| {
        use crate::data_gen_alt::{generate_data, GeneratorConfig, NumaMode};

        let config = GeneratorConfig {
            size,
            dedup_factor: dedup,
            compress_factor: compress,
            numa_mode: NumaMode::Auto,
            max_threads: Some(num_threads),
            numa_node: None,
            block_size: None,
            seed: None,
        };

        generate_data(config) // Returns DataBuffer directly - NO 16GB copy to bytes::Bytes!
    });

    // Convert to Bytes (zero-copy for Uma/Vec via Bytes::from(vec)) and wrap
    Py::new(py, PyBytesView::new(buffer.into_bytes()))
}

/// Generate data directly into existing Python buffer (ZERO-COPY WRITE)
///
/// # Arguments
/// * `buffer` - Pre-allocated Python buffer (bytearray, memoryview, numpy array, etc.)
/// * `dedup` - Deduplication factor
/// * `compress` - Compression factor
/// * `threads` - Number of threads (None = default)
///
/// # Returns
/// Number of bytes written
///
/// # Example
/// ```python
/// import s3dlio
/// import numpy as np
///
/// # Pre-allocate buffer
/// buf = bytearray(1024 * 1024)
/// nbytes = s3dlio.generate_into_buffer(buf)
/// print(f"Wrote {nbytes} bytes directly into buffer")
///
/// # Works with NumPy arrays too
/// arr = np.zeros(1024 * 1024, dtype=np.uint8)
/// s3dlio.generate_into_buffer(arr)
/// ```
#[pyfunction]
#[pyo3(signature = (buffer, dedup=1, compress=1, threads=None))]
fn generate_into_buffer(
    py: Python<'_>,
    buffer: Py<PyAny>,
    dedup: usize,
    compress: usize,
    threads: Option<usize>,
) -> PyResult<usize> {
    // Get buffer via PyBuffer protocol
    let buf: PyBuffer<u8> = PyBuffer::get(buffer.bind(py))?;

    // Ensure buffer is writable and contiguous
    if buf.readonly() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Buffer must be writable",
        ));
    }

    if !buf.is_c_contiguous() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Buffer must be C-contiguous for zero-copy operation",
        ));
    }

    let size = buf.len_bytes();
    let num_threads = threads.unwrap_or_else(default_data_gen_threads);

    // Generate data directly into DataBuffer (NO intermediate bytes::Bytes conversion!)
    let data_buffer = py.detach(|| {
        use crate::data_gen_alt::{generate_data, GeneratorConfig, NumaMode};

        let config = GeneratorConfig {
            size,
            dedup_factor: dedup,
            compress_factor: compress,
            numa_mode: NumaMode::Auto,
            max_threads: Some(num_threads),
            numa_node: None,
            block_size: None,
            seed: None,
        };

        generate_data(config) // Returns DataBuffer directly - NO 16GB copy to bytes::Bytes!
    });

    // Write into buffer (single copy only - can't avoid this since user provided the buffer)
    unsafe {
        let dst_ptr = buf.buf_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(data_buffer.as_ptr(), dst_ptr, size);
    }

    Ok(size)
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get default number of threads for data generation (50% of available CPUs)
///
/// # Returns
/// Number of threads that will be used by default
///
/// # Example
/// ```python
/// import s3dlio
///
/// threads = s3dlio.default_data_gen_threads()
/// print(f"Will use {threads} threads by default")
/// ```
#[pyfunction]
fn py_default_data_gen_threads() -> usize {
    default_data_gen_threads()
}

/// Get total number of CPU cores/threads available
///
/// # Returns
/// Total number of logical CPUs
///
/// # Example
/// ```python
/// import s3dlio
///
/// total = s3dlio.total_cpus()
/// print(f"System has {total} logical CPUs")
/// ```
#[pyfunction]
fn py_total_cpus() -> usize {
    total_cpus()
}

// =============================================================================
// Streaming Generator API
// =============================================================================

/// Streaming data generator for efficient chunk-by-chunk generation
///
/// This class allows you to generate large amounts of data incrementally,
/// reusing the same generator state for optimal performance. Perfect for
/// benchmarking and testing scenarios where you need to fill buffers repeatedly.
///
/// # Example
/// ```python
/// import s3dlio
///
/// # Create generator for 16 GB
/// gen = s3dlio.Generator(size=16 * 1024**3, dedup=1, compress=1)
///
/// # Generate in chunks
/// buf = bytearray(100 * 1024**2)  # 100 MB buffer
/// total = 0
///
/// while not gen.is_complete():
///     nbytes = gen.fill_chunk(buf)
///     if nbytes == 0:
///         break
///     total += nbytes
///     # Use buf[: nbytes]...
///
/// print(f"Generated {total} bytes")
/// ```
#[pyclass(name = "Generator")]
struct PyGenerator {
    inner: DataGenerator,
    chunk_size: usize,
}

#[pymethods]
impl PyGenerator {
    /// Create new streaming generator
    ///
    /// # Arguments
    /// * `size` - Total bytes to generate
    /// * `dedup` - Deduplication factor (1 = no dedup, 2 = 2:1 ratio, etc.)
    /// * `compress` - Compression factor (1 = incompressible, 2 = 2:1 ratio, etc.)
    /// * `threads` - Maximum threads to use (None = use default)
    /// * `chunk_size` - Recommended chunk size for fill_chunk() (default: 32 MB)
    /// * `seed` - Random seed for reproducible data (None = non-deterministic)
    #[new]
    #[pyo3(signature = (size, dedup=1, compress=1, threads=None, chunk_size=None, seed=None))]
    fn new(
        size: usize,
        dedup: usize,
        compress: usize,
        threads: Option<usize>,
        chunk_size: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        use crate::data_gen_alt::{GeneratorConfig, NumaMode};

        let config = GeneratorConfig {
            size,
            dedup_factor: dedup,
            compress_factor: compress,
            numa_mode: NumaMode::Auto,
            max_threads: threads,
            numa_node: None,
            block_size: None,
            seed,
        };

        let chunk_size = chunk_size.unwrap_or_else(DataGenerator::recommended_chunk_size);

        Ok(Self {
            inner: DataGenerator::new(config),
            chunk_size,
        })
    }

    /// Get recommended chunk size for optimal performance
    #[getter]
    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Fill the next chunk of data (ZERO-COPY WRITE)
    ///
    /// # Arguments
    /// * `buffer` - Pre-allocated buffer to fill
    ///
    /// # Returns
    /// Number of bytes written (0 when complete)
    fn fill_chunk(&mut self, py: Python<'_>, buffer: Py<PyAny>) -> PyResult<usize> {
        let buf: PyBuffer<u8> = PyBuffer::get(buffer.bind(py))?;

        if buf.readonly() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be writable",
            ));
        }

        if !buf.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Buffer must be C-contiguous",
            ));
        }

        let size = buf.len_bytes();

        // Generate DIRECTLY into Python buffer without holding GIL
        let written = py.detach(|| unsafe {
            let dst_ptr = buf.buf_ptr() as *mut u8;
            let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, size);
            self.inner.fill_chunk(dst_slice)
        });

        Ok(written)
    }

    /// Check if generation is complete
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Reset generator to start
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// =============================================================================
// NPZ Fast-Build API
// =============================================================================

/// Build a complete NPZ archive with random data in Rust (hardware-accelerated CRC32).
///
/// This replaces `numpy.savez()` for object-store workloads, eliminating the
/// Python-side CRC32 bottleneck (~178 ms for 140 MiB using software `zlib.crc32`).
/// Rust's `crc32fast` uses SSE4.2/NEON and runs at 4–8 GB/s, giving a ~5× speedup.
///
/// The generated archive matches numpy's format and is loadable by `numpy.load()`.
///
/// # Arguments
/// * `shape`       — Python list of ints, e.g. `[6053, 6053, 1]`
/// * `dtype`       — NumPy dtype string (default `"<f4"` = float32 little-endian)
/// * `num_samples` — number of label samples in y.npy (default `1`)
///
/// # Returns
/// A `BytesView` supporting the buffer protocol for zero-copy upload.
///
/// # Example
/// ```python
/// import s3dlio
///
/// # Generate a 140 MiB unet3d-style NPZ (~50 ms vs ~270 ms for numpy.savez)
/// npz = s3dlio.generate_npz_bytes(shape=[6053, 6053, 1])
/// s3dlio.put_bytes("s3://bucket/file.npz", npz)   # zero-copy upload
/// ```
#[pyfunction]
#[pyo3(signature = (shape, dtype="<f4", num_samples=1))]
fn generate_npz_bytes(
    py: Python<'_>,
    shape: Vec<usize>,
    dtype: &str,
    num_samples: usize,
) -> PyResult<Py<PyBytesView>> {
    let dtype_owned = dtype.to_string(); // move into closure

    // Build the NPZ entirely without holding the GIL (parallel generation + CRC32)
    let vec = py
        .detach(|| {
            crate::data_formats::npz::generate_npz_bytes_raw(&shape, &dtype_owned, num_samples)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;

    let bytes = bytes::Bytes::from(vec); // zero-copy: wraps Vec<u8> in Arc
    Py::new(py, PyBytesView::new(bytes))
}

// =============================================================================
// Module Registration
// =============================================================================

pub fn register_datagen_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyGenerator>()?;

    // Register data generation functions
    m.add_function(wrap_pyfunction!(generate_data, m)?)?;
    m.add_function(wrap_pyfunction!(generate_data_with_threads, m)?)?;
    m.add_function(wrap_pyfunction!(generate_into_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(generate_npz_bytes, m)?)?;

    // Register utility functions
    m.add_function(wrap_pyfunction!(py_default_data_gen_threads, m)?)?;
    m.add_function(wrap_pyfunction!(py_total_cpus, m)?)?;

    Ok(())
}
