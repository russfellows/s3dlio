use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use ndarray::ArrayD;
use std::io::{Cursor, Read, Write};
use zip::{read::ZipArchive, write::FileOptions, CompressionMethod, ZipWriter};


/// Serialize an ndarray ArrayD<f32> to NPY 1.0 format as Bytes (zero-copy).
///
/// This function creates a complete NPY file in memory following the NumPy .npy format v1.0 spec:
/// - Magic number: \x93NUMPY
/// - Version: 1.0 (major=1, minor=0)
/// - Header: Python dict with dtype, shape, and fortran_order
/// - Data: Array data in C-contiguous layout (little-endian)
///
/// # Zero-Copy Strategy
/// - Returns `Bytes` instead of `Vec<u8>` for efficient cloning/slicing
/// - Uses `as_slice_memory_order()` for zero-copy access when array is contiguous
/// - Falls back to iteration for non-contiguous arrays
///
/// # Format Details
/// The header is aligned to 64-byte boundaries (NPY 1.0 spec requirement).
/// Data is written in little-endian format ('<f4' in NumPy dtype notation).
///
/// # Example
/// ```
/// use s3dlio::data_formats::npz::array_to_npy_bytes;
/// use ndarray::ArrayD;
///
/// let array = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let npy_bytes = array_to_npy_bytes(&array)?;
/// // npy_bytes can now be written to file or sent over network with zero-copy semantics
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn array_to_npy_bytes(array: &ArrayD<f32>) -> Result<Bytes> {
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
    
    // NPY 1.0 header must be divisible by 64 bytes (after magic + version + header_len)
    let header_len = header.len() + 1; // +1 for newline
    let padding = (64 - ((6 + 2 + 2 + header_len) % 64)) % 64;
    
    // Calculate total size
    let data_size = array.len() * std::mem::size_of::<f32>();
    let total_size = 6 + 2 + 2 + header_len + padding + data_size;
    let mut buffer = Vec::with_capacity(total_size);
    
    // Write magic number
    buffer.write_all(b"\x93NUMPY")?;
    
    // Write version (1.0)
    buffer.write_all(&[1, 0])?;
    
    // Write header length (little-endian u16)
    let header_data_len = (header_len + padding) as u16;
    buffer.write_all(&header_data_len.to_le_bytes())?;
    
    // Write header with padding and newline
    buffer.write_all(header.as_bytes())?;
    buffer.write_all(&vec![b' '; padding])?;
    buffer.write_all(b"\n")?;
    
    // Write data (zero-copy if contiguous, iteration if not)
    if let Some(slice) = array.as_slice_memory_order() {
        // Zero-copy path: array is contiguous in memory
        let bytes = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * std::mem::size_of::<f32>()
            )
        };
        buffer.write_all(bytes)?;
    } else {
        // Fallback: array is not contiguous, iterate elements
        for value in array.iter() {
            buffer.write_all(&value.to_le_bytes())?;
        }
    }
    
    Ok(Bytes::from(buffer))
}


fn make_npy_header(shape: usize) -> Vec<u8> {
    let dict = format!(
        "{{'descr': '|u1', 'fortran_order': False, 'shape': ({shape},)}}"
    );
    let mut header = dict.into_bytes();

    // Align to 16‑byte boundary per .npy spec
    const BASE: usize = 10; // magic (6) + version (2) + hdr‑len (2)
    let with_nl = BASE + header.len() + 1;
    let pad = (16 - (with_nl % 16)) % 16;
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');

    // Magic + version = v1.0
    let mut result = b"\x93NUMPY\x01\x00".to_vec();
    let len: u16 = header.len() as u16;
    result.extend(&len.to_le_bytes());
    result.extend(header);
    result
}

/// Build an **entire** .npz archive in memory and return it.
/// 
/// Legacy single-array builder. Use `build_multi_npz()` for multiple arrays.
pub fn build_npz(elements: usize, _element_size: usize, data: &[u8]) -> Result<Bytes> {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut zip = ZipWriter::new(&mut cursor);
        //
        // Old interface
        //let opts = FileOptions::default()
        // New 
        let opts: FileOptions<()> = FileOptions::default()
            .compression_method(CompressionMethod::Stored);

        zip.start_file("data.npy", opts)?;
        zip.write_all(&make_npy_header(elements))?;   // HEADER KEPT
        zip.write_all(data)?;
        zip.finish()?;
    } // <- ZipWriter dropped, borrow ends

    Ok(Bytes::from(cursor.into_inner()))
}

/// Build an NPZ archive with multiple named arrays (zero-copy).
///
/// This function creates a complete NPZ file (ZIP archive) in memory containing
/// multiple named NumPy arrays. This is the standard format used by numpy.savez()
/// and commonly used in PyTorch/TensorFlow data pipelines.
///
/// # Format
/// - NPZ files are ZIP archives containing one or more .npy files
/// - Each array is stored as a separate .npy file inside the archive
/// - Array names become filenames (with .npy extension added automatically)
/// - No compression (CompressionMethod::Stored) for maximum I/O performance
///
/// # Zero-Copy Strategy
/// - Uses `array_to_npy_bytes()` which returns Arc-backed `Bytes`
/// - Multiple arrays share memory allocation when possible
/// - Returns `Bytes` for efficient cloning/slicing
///
/// # PyTorch/NumPy Compatibility
/// - Fully compatible with `numpy.load()` and `numpy.savez()`
/// - PyTorch can load via `torch.from_numpy(np.load('file.npz')['array_name'])`
/// - Common use case: Store training data, labels, metadata in one file
///
/// # Example
/// ```
/// use s3dlio::data_formats::npz::build_multi_npz;
/// use ndarray::ArrayD;
///
/// let images = ArrayD::from_shape_vec(vec![10, 28, 28], vec![0.0f32; 7840])?;
/// let labels = ArrayD::from_shape_vec(vec![10], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0])?;
/// 
/// let arrays = vec![
///     ("images", &images),
///     ("labels", &labels),
/// ];
/// 
/// let npz_bytes = build_multi_npz(arrays)?;
/// // npz_bytes can be written to storage or sent over network
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn build_multi_npz(arrays: Vec<(&str, &ArrayD<f32>)>) -> Result<Bytes> {
    if arrays.is_empty() {
        return Err(anyhow!("Cannot create NPZ with zero arrays"));
    }
    
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut zip = ZipWriter::new(&mut cursor);
        let opts: FileOptions<()> = FileOptions::default()
            .compression_method(CompressionMethod::Stored); // No compression for speed

        for (name, array) in arrays {
            // Ensure .npy extension
            let npy_name = if name.ends_with(".npy") {
                name.to_string()
            } else {
                format!("{}.npy", name)
            };
            
            // Start new file in ZIP
            zip.start_file(&npy_name, opts)
                .with_context(|| format!("Failed to create NPZ entry: {}", npy_name))?;
            
            // Convert array to NPY format (zero-copy)
            let npy_bytes = array_to_npy_bytes(array)
                .with_context(|| format!("Failed to serialize array: {}", name))?;
            
            // Write NPY bytes to ZIP
            zip.write_all(&npy_bytes)
                .with_context(|| format!("Failed to write NPZ entry: {}", npy_name))?;
        }
        
        zip.finish()
            .context("Failed to finalize NPZ archive")?;
    }

    Ok(Bytes::from(cursor.into_inner()))
}

/// Read an NPY file from bytes and parse into ArrayD<f32>.
///
/// This function parses the NPY 1.0 format:
/// - Validates magic number and version
/// - Parses Python dict header for dtype, shape, and fortran_order
/// - Reads array data in little-endian format
///
/// # Example
/// ```no_run
/// use s3dlio::data_formats::npz::read_npy_bytes;
/// use bytes::Bytes;
///
/// let npy_data = Bytes::from(vec![/* NPY file bytes */]);
/// let array = read_npy_bytes(&npy_data)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn read_npy_bytes(data: &[u8]) -> Result<ArrayD<f32>> {
    // Parse NPY header
    if data.len() < 10 {
        return Err(anyhow!("NPY file too short (< 10 bytes)"));
    }
    
    // Check magic number
    if &data[0..6] != b"\x93NUMPY" {
        return Err(anyhow!("Invalid NPY magic number"));
    }
    
    // Check version (we only support 1.0)
    let major = data[6];
    let minor = data[7];
    if major != 1 || minor != 0 {
        return Err(anyhow!("Unsupported NPY version {}.{} (only 1.0 supported)", major, minor));
    }
    
    // Read header length
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header_start = 10;
    let header_end = header_start + header_len;
    
    if data.len() < header_end {
        return Err(anyhow!("NPY file truncated (header claims {} bytes)", header_len));
    }
    
    // Parse header (Python dict as string)
    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .context("NPY header is not valid UTF-8")?;
    
    // Extract shape from header
    // Format: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
    let shape = parse_npy_shape(header_str)?;
    
    // Validate dtype (we only support '<f4' = little-endian float32)
    if !header_str.contains("'<f4'") && !header_str.contains("'<f'") {
        return Err(anyhow!("Unsupported NPY dtype (only '<f4' supported), header: {}", header_str));
    }
    
    // Check fortran_order (we only support C-contiguous)
    if header_str.contains("True") {
        return Err(anyhow!("Fortran-order arrays not supported"));
    }
    
    // Calculate expected data size
    let num_elements: usize = shape.iter().product();
    let data_size = num_elements * std::mem::size_of::<f32>();
    let data_start = header_end;
    let data_end = data_start + data_size;
    
    if data.len() < data_end {
        return Err(anyhow!(
            "NPY file truncated (expected {} data bytes, got {})",
            data_size,
            data.len() - data_start
        ));
    }
    
    // Read data as f32 slice (little-endian)
    let float_data = unsafe {
        std::slice::from_raw_parts(
            data[data_start..data_end].as_ptr() as *const f32,
            num_elements
        )
    };
    
    // Create ndarray
    ArrayD::from_shape_vec(shape, float_data.to_vec())
        .context("Failed to create ndarray from NPY data")
}

/// Parse shape tuple from NPY header string.
/// Example: "{'shape': (2, 3), }" -> vec![2, 3]
fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    // Find 'shape': (...)
    let shape_start = header.find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| anyhow!("NPY header missing 'shape' field"))?;
    
    let after_shape = &header[shape_start..];
    let tuple_start = after_shape.find('(')
        .ok_or_else(|| anyhow!("NPY header 'shape' field malformed (no '(')"))?;
    let tuple_end = after_shape.find(')')
        .ok_or_else(|| anyhow!("NPY header 'shape' field malformed (no ')')"))?;
    
    let tuple_content = &after_shape[tuple_start + 1..tuple_end];
    
    // Handle empty shape (scalar)
    if tuple_content.trim().is_empty() {
        return Ok(vec![]);
    }
    
    // Parse comma-separated dimensions
    let dims: Result<Vec<usize>> = tuple_content
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>()
            .with_context(|| format!("Invalid dimension in shape: '{}'", s)))
        .collect();
    
    dims
}

/// Read an array from an NPZ archive by name.
///
/// NPZ files are ZIP archives containing one or more .npy files.
/// This function extracts a specific array by name.
///
/// # Example
/// ```no_run
/// use s3dlio::data_formats::npz::read_npz_array;
/// use bytes::Bytes;
///
/// let npz_data = Bytes::from(vec![/* NPZ file bytes */]);
/// let array = read_npz_array(&npz_data, "data")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn read_npz_array(data: &[u8], array_name: &str) -> Result<ArrayD<f32>> {
    let cursor = Cursor::new(data);
    let mut archive = ZipArchive::new(cursor)
        .context("Failed to open NPZ file as ZIP archive")?;
    
    // Find the array by name (with or without .npy extension)
    let npy_name = if array_name.ends_with(".npy") {
        array_name.to_string()
    } else {
        format!("{}.npy", array_name)
    };
    
    let mut file = archive.by_name(&npy_name)
        .with_context(|| format!("Array '{}' not found in NPZ archive", array_name))?;
    
    // Read entire .npy file into buffer
    let mut npy_bytes = Vec::new();
    file.read_to_end(&mut npy_bytes)
        .context("Failed to read NPY data from NPZ archive")?;
    
    // Parse NPY format
    read_npy_bytes(&npy_bytes)
}

/// List all array names in an NPZ archive.
///
/// # Example
/// ```no_run
/// use s3dlio::data_formats::npz::list_npz_arrays;
/// use bytes::Bytes;
///
/// let npz_data = Bytes::from(vec![/* NPZ file bytes */]);
/// let names = list_npz_arrays(&npz_data)?;
/// println!("Arrays: {:?}", names);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn list_npz_arrays(data: &[u8]) -> Result<Vec<String>> {
    let cursor = Cursor::new(data);
    let archive = ZipArchive::new(cursor)
        .context("Failed to open NPZ file as ZIP archive")?;
    
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            archive.name_for_index(i)
                .map(|name| {
                    // Strip .npy extension
                    if name.ends_with(".npy") {
                        name[..name.len() - 4].to_string()
                    } else {
                        name.to_string()
                    }
                })
        })
        .collect();
    
    Ok(names)
}


