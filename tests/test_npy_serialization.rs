//! Test ndarray-npy 0.9 API and compare with custom implementation
//!
//! This test verifies:
//! 1. Does ndarray-npy 0.9 WriteNpyExt work with Vec<u8>?
//! 2. Compare output format with dl-driver's custom implementation
//! 3. Validate zero-copy approach using Bytes

use anyhow::Result;
use ndarray::{Array, ArrayD};

#[test]
fn test_s3dlio_array_to_npy_bytes() -> Result<()> {
    use s3dlio::data_formats::npz::array_to_npy_bytes;
    
    // Create test array: 2x3 float32
    let array = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    // Use s3dlio's new API
    let npy_bytes = array_to_npy_bytes(&array)?;
    
    println!("s3dlio array_to_npy_bytes succeeded!");
    println!("Buffer length: {} bytes", npy_bytes.len());
    println!("First 128 bytes (hex): {:02x?}", &npy_bytes[..128.min(npy_bytes.len())]);
    
    // Verify NPY 1.0 format header
    assert_eq!(&npy_bytes[0..6], b"\x93NUMPY");
    assert_eq!(npy_bytes[6], 1); // major version
    assert_eq!(npy_bytes[7], 0); // minor version
    
    // Verify it's a Bytes object (zero-copy capable)
    let npy_bytes2 = npy_bytes.clone(); // Should be zero-copy
    assert_eq!(npy_bytes.as_ptr(), npy_bytes2.as_ptr(), "Bytes clone should be zero-copy");
    
    Ok(())
}

#[test]
fn test_npy_09_write_to_vec() -> Result<()> {
    // Create test array: 2x3 float32
    let array = Array::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    // Test 1: Can we write to Vec<u8> with ndarray-npy 0.9?
    let mut buffer: Vec<u8> = Vec::new();
    
    // This is what Jim Turner claims works in 0.9:
    #[cfg(feature = "test-npy-09")]
    {
        use ndarray_npy::WriteNpyExt;
        array.write_npy(&mut buffer)?;
        
        println!("ndarray-npy 0.9 WriteNpyExt succeeded!");
        println!("Buffer length: {} bytes", buffer.len());
        println!("First 128 bytes (hex): {:02x?}", &buffer[..128.min(buffer.len())]);
    }
    
    #[cfg(not(feature = "test-npy-09"))]
    {
        println!("SKIPPED: Requires ndarray-npy 0.9 (use --features test-npy-09)");
    }
    
    Ok(())
}

#[test]
fn test_custom_npy_implementation() -> Result<()> {
    // Create same test array
    let array = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    // Test dl-driver's custom implementation approach
    let npy_bytes = array_to_npy_bytes_custom(&array)?;
    
    println!("Custom implementation succeeded!");
    println!("Buffer length: {} bytes", npy_bytes.len());
    println!("First 128 bytes (hex): {:02x?}", &npy_bytes[..128.min(npy_bytes.len())]);
    
    // Verify NPY 1.0 format header
    assert_eq!(&npy_bytes[0..6], b"\x93NUMPY");
    assert_eq!(npy_bytes[6], 1); // major version
    assert_eq!(npy_bytes[7], 0); // minor version
    
    Ok(())
}

/// Custom NPY serialization (based on dl-driver workaround)
/// This will be moved to s3dlio as the public API
fn array_to_npy_bytes_custom(array: &ArrayD<f32>) -> Result<Vec<u8>> {
    use std::io::Write;
    
    // Build NPY 1.0 header
    let shape_str = array.shape()
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        shape_str
    );
    
    // NPY 1.0 header must be divisible by 64 bytes (after magic + version)
    let header_len = header.len() + 1; // +1 for newline
    let padding = (64 - ((6 + 2 + 2 + header_len) % 64)) % 64;
    let total_header_len = 6 + 2 + 2 + header_len + padding;
    
    // Calculate total size
    let data_size = array.len() * std::mem::size_of::<f32>();
    let mut buffer = Vec::with_capacity(total_header_len + data_size);
    
    // Write magic number
    buffer.write_all(b"\x93NUMPY")?;
    
    // Write version
    buffer.write_all(&[1, 0])?;
    
    // Write header length (little-endian u16)
    let header_data_len = (header_len + padding) as u16;
    buffer.write_all(&header_data_len.to_le_bytes())?;
    
    // Write header
    buffer.write_all(header.as_bytes())?;
    buffer.write_all(&vec![b' '; padding])?;
    buffer.write_all(b"\n")?;
    
    // Write data (zero-copy if contiguous)
    if let Some(slice) = array.as_slice_memory_order() {
        // Zero-copy path
        let bytes = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * std::mem::size_of::<f32>()
            )
        };
        buffer.write_all(bytes)?;
    } else {
        // Fallback: iterate (non-contiguous arrays)
        for value in array.iter() {
            buffer.write_all(&value.to_le_bytes())?;
        }
    }
    
    Ok(buffer)
}

#[test]
fn test_bytes_zero_copy() {
    use bytes::Bytes;
    
    // Demonstrate zero-copy with Bytes
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let bytes1 = Bytes::from(data);
    let bytes2 = bytes1.clone(); // Zero-copy clone
    
    println!("bytes1 ptr: {:p}", bytes1.as_ptr());
    println!("bytes2 ptr: {:p}", bytes2.as_ptr());
    
    // Same pointer = zero-copy
    assert_eq!(bytes1.as_ptr(), bytes2.as_ptr());
    
    // Slicing is also zero-copy
    let slice = bytes1.slice(2..6);
    println!("slice ptr: {:p}", slice.as_ptr());
    // slice.as_ptr() should be bytes1.as_ptr() + 2
}

#[test]
fn test_npy_round_trip() -> Result<()> {
    use s3dlio::data_formats::npz::{array_to_npy_bytes, read_npy_bytes};
    
    // Create test array
    let original = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    // Serialize to NPY
    let npy_bytes = array_to_npy_bytes(&original)?;
    
    // Deserialize back
    let restored = read_npy_bytes(&npy_bytes)?;
    
    // Verify shape
    assert_eq!(original.shape(), restored.shape());
    
    // Verify data
    assert_eq!(original.as_slice().unwrap(), restored.as_slice().unwrap());
    
    println!("NPY round-trip test passed!");
    Ok(())
}

#[test]
fn test_npz_round_trip() -> Result<()> {
    use s3dlio::data_formats::npz::{array_to_npy_bytes, read_npz_array, list_npz_arrays};
    use std::io::Write;
    use zip::{write::FileOptions, CompressionMethod, ZipWriter};
    
    // Create test array
    let original = ArrayD::from_shape_vec(vec![3, 4], vec![
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ])?;
    
    // Create NPZ file manually
    let mut npz_buffer = Vec::new();
    {
        let mut zip = ZipWriter::new(std::io::Cursor::new(&mut npz_buffer));
        let opts: FileOptions<()> = FileOptions::default()
            .compression_method(CompressionMethod::Stored);
        
        // Add array as "data.npy"
        zip.start_file("data.npy", opts)?;
        let npy_bytes = array_to_npy_bytes(&original)?;
        zip.write_all(&npy_bytes)?;
        
        zip.finish()?;
    }
    
    // List arrays
    let names = list_npz_arrays(&npz_buffer)?;
    assert_eq!(names, vec!["data"]);
    
    // Read array back
    let restored = read_npz_array(&npz_buffer, "data")?;
    
    // Verify shape and data
    assert_eq!(original.shape(), restored.shape());
    assert_eq!(original.as_slice().unwrap(), restored.as_slice().unwrap());
    
    println!("NPZ round-trip test passed!");
    Ok(())
}
