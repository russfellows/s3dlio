# HowTo ZeroCopy

Note: This is from Chat, describing how to implement Zero Copy, from Rust to Python.  Will mark this as a future enhancement.  

Below is a minimal and tested approach for PyO3 0.20.3 that demonstrates how to return a read-only memoryview referencing a Rust Vec<u8> without copying. You do not implement a PyBufferProtocol trait (it does not exist in user-facing form in pyo3 0.20.x); instead, you call the C API function PyMemoryView_FromMemory and return the resulting memoryview object to Python.

Use this pattern in your S3 “get” code:

You download the object into a Vec<u8>.
You wrap that data in a small #[pyclass] type (RustBuffer).
Python code can call a memoryview() method on that object to obtain a zero‑copy memoryview.
Below is an example file – call it python_api.rs – with a single minimal RustBuffer class plus a #[pyfunction] called get():

```
// src/python_api.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
// We need raw FFI calls for PyMemoryView_FromMemory.
use pyo3::ffi;
use std::ffi::CString;

/// Error helper: convert Rust error text into a Python exception.
fn py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A small class holding a Rust buffer.  
/// The `memoryview()` method returns a Python memoryview referencing it *without copying*.
#[pyclass]
pub struct RustBuffer {
    data: Vec<u8>,
}

#[pymethods]
impl RustBuffer {
    /// A Python constructor (e.g. `buf = RustBuffer(1024)`) that creates `size` zero bytes.
    /// If you only want to create this from Rust, you can remove or hide this constructor.
    #[new]
    fn new(size: usize) -> Self {
        RustBuffer {
            data: vec![0u8; size],
        }
    }

    /// Return a read-only memoryview referencing `self.data` without copying.
    /// The memory stays valid as long as `RustBuffer` is alive in Python.
    fn memoryview(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // We only do a read-only memory here. 
        // For a writable buffer, use self.data.as_mut_ptr() and ffi::PyBUF_WRITE.
        let ptr = self.data.as_ptr() as *mut i8;
        let len = self.data.len() as isize;

        // https://docs.rs/pyo3/0.20.3/pyo3/ffi/fn.PyMemoryView_FromMemory.html
        let raw_mv = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, ffi::PyBUF_READ) };
        if raw_mv.is_null() {
            // If something went wrong (e.g. out of memory), fetch and return the Python exception.
            Err(PyErr::fetch(py))
        } else {
            // Convert the raw pointer into a safe Py<PyAny> object and return it.
            Ok(unsafe { Py::from_owned_ptr(py, raw_mv) })
        }
    }
}

impl RustBuffer {
    /// A **Rust-only** constructor for building a RustBuffer from a Vec<u8>.
    /// (not exposed to Python, so we don't put `#[pymethods]` on it)
    pub fn from_vec(data: Vec<u8>) -> Self {
        RustBuffer { data }
    }
}

/// Example "get" function that returns a RustBuffer referencing the S3 bytes.
#[pyfunction]
pub fn get(uri: &str) -> PyResult<RustBuffer> {
    // 1. parse the S3 uri, get the data from S3 as a Vec<u8>. 
    //    (Pseudo-code for demonstration)
    let bytes = match uri {
        "s3://bucket/test" => vec![1,2,3,4],   // placeholder
        _ => return Err(py_err("Invalid S3 URI or not found")),
    };

    // 2. Wrap in RustBuffer for zero-copy memoryview usage
    Ok(RustBuffer::from_vec(bytes))
}
```

## Explanation

1. No “PyBufferProtocol”
In PyO3 0.20.x there is no stable user‐facing trait to export a buffer to Python. Instead, you directly call PyMemoryView_FromMemory from the raw C API.

2. Memoryview vs. Bytes
memoryview() returns a Py<PyAny> which references your Rust data without copying.
If you want a Python bytes object (which is a copy), you can define a separate method like:

Rust

```
fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
    PyBytes::new(py, &self.data)
}
```

By having both, you can choose from Python whether to do a zero‑copy memoryview or a copy.

3. Rust-Only Constructor
The impl RustBuffer { pub fn from_vec(...) } method is used in your S3 “get” function to create a zero‑copy buffer from the downloaded data.
4. Lifetime Concerns Because RustBuffer holds a Vec<u8>, that memory remains valid so long as RustBuffer is alive in Python. If Python code keeps a memoryview after dropping RustBuffer, that pointer becomes invalid. This is typical for a memoryview.
5. If You Need Writeable Memory Switch ffi::PyBUF_READ to ffi::PyBUF_WRITE.
Use self.data.as_mut_ptr() as *mut i8.
Then Python code can modify the memory in-place.

### Module Initialization (for PyO3 v0.20.3)
You can then expose RustBuffer and the get function in your top-level #[pymodule]. For instance, in src/lib.rs:
Rust

```
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};

mod python_api;  // where the code above lives
use python_api::{RustBuffer, get};

#[pymodule]
fn dlio_s3_rust(py: Python, m: &PyModule) -> PyResult<()> {
    // Register our "get" function
    m.add_function(wrap_pyfunction!(get, py)?)?;
    // Expose the RustBuffer class
    m.add_class::<RustBuffer>()?;
    Ok(())
}
```

(If your original code used older approaches like #[pyfn(m, "get")], that also works. The snippet above uses wrap_pyfunction!. Both styles exist in PyO3 0.20.)

### Python Usage
In Python:

```
import dlio_s3_rust

# Suppose 'get' returns a RustBuffer referencing the S3 data
buf = dlio_s3_rust.get("s3://bucket/test")

# To get a zero-copy memoryview:
mv = buf.memoryview()
print(len(mv))          # the length of the data
print(list(mv[0:4]))    # e.g. [1,2,3,4]
```

No data is copied into Python’s memory unless you explicitly request it (e.g. by converting the memoryview to bytes).

# Conclusion

Using pyo3 0.20.3, the recommended approach to expose zero‑copy memory from Rust to Python is to:

Create a small #[pyclass] that holds your Vec<u8>.
Write a method that calls PyMemoryView_FromMemory (from the PyO3 ffi module).
Return that memoryview object, referencing your Rust buffer with no copying.
This approach is simpler and more direct than any user‑facing “PyBufferProtocol” trait, which does not exist in pyo3 0.20.3. The snippet above shows how to integrate that class and function into your code, so your S3 “get” returns a RustBuffer. Then from Python you can call .memoryview() for a zero‑copy view of the data.
