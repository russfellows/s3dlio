// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

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
    let shape_str = array
        .shape()
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
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
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
    let dict = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': ({shape},)}}");
    let mut header = dict.into_bytes();

    // Align to 16‑byte boundary per .npy spec
    const BASE: usize = 10; // magic (6) + version (2) + hdr‑len (2)
    let with_nl = BASE + header.len() + 1;
    let pad = (16 - (with_nl % 16)) % 16;
    header.extend(std::iter::repeat_n(b' ', pad));
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
        let opts: FileOptions<()> =
            FileOptions::default().compression_method(CompressionMethod::Stored);

        zip.start_file("data.npy", opts)?;
        zip.write_all(&make_npy_header(elements))?; // HEADER KEPT
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
        let opts: FileOptions<()> =
            FileOptions::default().compression_method(CompressionMethod::Stored); // No compression for speed

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

        zip.finish().context("Failed to finalize NPZ archive")?;
    }

    Ok(Bytes::from(cursor.into_inner()))
}

// =============================================================================
// Fast NPZ builder — single allocation, Rayon in-place generation
// =============================================================================

/// Build an NPY 1.0 header for any shape and dtype.
/// The header is padded to a 64-byte boundary (NPY 1.0 spec).
fn build_npy_header_typed(shape: &[usize], dtype_str: &str) -> Vec<u8> {
    let shape_str: String = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape_tuple = if shape.len() == 1 {
        format!("({},)", shape_str)
    } else {
        format!("({})", shape_str)
    };
    let dict =
        format!("{{'descr': '{dtype_str}', 'fortran_order': False, 'shape': {shape_tuple}, }}");
    let header_len = dict.len() + 1; // +1 for newline
    let padding = (64 - ((6 + 2 + 2 + header_len) % 64)) % 64;
    let header_data_len = (header_len + padding) as u16;

    let mut result = Vec::with_capacity(6 + 2 + 2 + header_len + padding);
    result.extend_from_slice(b"\x93NUMPY\x01\x00");
    result.extend_from_slice(&header_data_len.to_le_bytes());
    result.extend_from_slice(dict.as_bytes());
    result.extend(std::iter::repeat_n(b' ', padding));
    result.push(b'\n');
    result
}

/// Infer element size in bytes from an NPY dtype string.
fn dtype_element_size(dtype_str: &str) -> usize {
    dtype_str
        .chars()
        .rev()
        .find(|c| c.is_ascii_digit())
        .and_then(|c| c.to_digit(10))
        .map(|d| d as usize)
        .unwrap_or(4)
}

/// Write a ZIP local file header (30 + name.len() bytes) into dst.
/// CRC32 and data_size may be zero (placeholders) and patched later.
fn write_local_file_header(dst: &mut [u8], name: &[u8], crc32: u32, data_size: u32) {
    dst[0..4].copy_from_slice(b"PK\x03\x04");
    dst[4..6].copy_from_slice(&20u16.to_le_bytes());
    dst[6..8].copy_from_slice(&0u16.to_le_bytes());
    dst[8..10].copy_from_slice(&0u16.to_le_bytes()); // STORED
    dst[10..12].copy_from_slice(&0u16.to_le_bytes());
    dst[12..14].copy_from_slice(&0u16.to_le_bytes());
    dst[14..18].copy_from_slice(&crc32.to_le_bytes());
    dst[18..22].copy_from_slice(&data_size.to_le_bytes());
    dst[22..26].copy_from_slice(&data_size.to_le_bytes());
    dst[26..28].copy_from_slice(&(name.len() as u16).to_le_bytes());
    dst[28..30].copy_from_slice(&0u16.to_le_bytes());
    dst[30..30 + name.len()].copy_from_slice(name);
}

/// Write a ZIP central directory entry (46 + name.len() bytes) into dst.
fn write_central_dir_entry(
    dst: &mut [u8],
    name: &[u8],
    data_size: u32,
    crc32: u32,
    local_offset: u32,
) {
    dst[0..4].copy_from_slice(b"PK\x01\x02");
    dst[4..6].copy_from_slice(&20u16.to_le_bytes());
    dst[6..8].copy_from_slice(&20u16.to_le_bytes());
    dst[8..10].copy_from_slice(&0u16.to_le_bytes());
    dst[10..12].copy_from_slice(&0u16.to_le_bytes()); // STORED
    dst[12..14].copy_from_slice(&0u16.to_le_bytes());
    dst[14..16].copy_from_slice(&0u16.to_le_bytes());
    dst[16..20].copy_from_slice(&crc32.to_le_bytes());
    dst[20..24].copy_from_slice(&data_size.to_le_bytes());
    dst[24..28].copy_from_slice(&data_size.to_le_bytes());
    dst[28..30].copy_from_slice(&(name.len() as u16).to_le_bytes());
    dst[30..32].copy_from_slice(&0u16.to_le_bytes());
    dst[32..34].copy_from_slice(&0u16.to_le_bytes());
    dst[34..36].copy_from_slice(&0u16.to_le_bytes());
    dst[36..38].copy_from_slice(&0u16.to_le_bytes());
    dst[38..42].copy_from_slice(&0u32.to_le_bytes());
    dst[42..46].copy_from_slice(&local_offset.to_le_bytes());
    dst[46..46 + name.len()].copy_from_slice(name);
}

/// Write the ZIP end-of-central-directory record (22 bytes) into dst.
fn write_eocd(dst: &mut [u8], num_entries: u16, cd_size: u32, cd_offset: u32) {
    dst[0..4].copy_from_slice(b"PK\x05\x06");
    dst[4..6].copy_from_slice(&0u16.to_le_bytes());
    dst[6..8].copy_from_slice(&0u16.to_le_bytes());
    dst[8..10].copy_from_slice(&num_entries.to_le_bytes());
    dst[10..12].copy_from_slice(&num_entries.to_le_bytes());
    dst[12..16].copy_from_slice(&cd_size.to_le_bytes());
    dst[16..20].copy_from_slice(&cd_offset.to_le_bytes());
    dst[20..22].copy_from_slice(&0u16.to_le_bytes());
}

/// Build a complete NPZ archive with random data.
///
/// Uses a single Vec<u8> allocation.  Rayon fills the x-data region directly
/// into the output buffer (no intermediate copy), then crc32fast computes the
/// checksum over the already-hot pages.
///
/// # NPZ structure
/// ```text
/// x.npy  -- dtype array with the given shape, filled with random bytes
/// y.npy  -- int64 array of shape (num_samples,) containing zeros (labels)
/// ```
pub fn generate_npz_bytes_raw(
    shape: &[usize],
    dtype_str: &str,
    num_samples: usize,
) -> Result<Vec<u8>> {
    use rand::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rayon::prelude::*;

    let npy_hdr_x = build_npy_header_typed(shape, dtype_str);
    let npy_hdr_y = build_npy_header_typed(&[num_samples], "<i8");

    let elem_size = dtype_element_size(dtype_str);
    let x_data_size: usize = shape.iter().product::<usize>() * elem_size;
    let y_data_size: usize = num_samples * 8;
    let x_npy_size: usize = npy_hdr_x.len() + x_data_size;
    let y_npy_size: usize = npy_hdr_y.len() + y_data_size;

    let name_x: &[u8] = b"x.npy";
    let name_y: &[u8] = b"y.npy";

    // ZIP structural constants
    const LOCAL_BASE: usize = 30;
    const CD_BASE: usize = 46;
    let x_local_hdr = LOCAL_BASE + name_x.len();
    let y_local_hdr = LOCAL_BASE + name_y.len();
    let x_cd_entry = CD_BASE + name_x.len();
    let y_cd_entry = CD_BASE + name_y.len();

    // Exact byte offsets
    let off_x_local: usize = 0;
    let off_x_npy_hdr: usize = off_x_local + x_local_hdr;
    let off_x_data: usize = off_x_npy_hdr + npy_hdr_x.len();

    let off_y_local: usize = off_x_data + x_data_size;
    let off_y_npy_hdr: usize = off_y_local + y_local_hdr;
    let off_y_data: usize = off_y_npy_hdr + npy_hdr_y.len();

    let off_cd: usize = off_y_data + y_data_size;
    let off_cd_y: usize = off_cd + x_cd_entry;
    let off_eocd: usize = off_cd_y + y_cd_entry;
    let total: usize = off_eocd + 22;

    // Single allocation -- zeroed, then fully overwritten before returning.
    let mut buf: Vec<u8> = vec![0u8; total];

    // x.npy local header (placeholder CRC/size, patched below)
    write_local_file_header(&mut buf[off_x_local..], name_x, 0, 0);

    // x.npy NPY header
    buf[off_x_npy_hdr..off_x_npy_hdr + npy_hdr_x.len()].copy_from_slice(&npy_hdr_x);

    // x.npy random data -- Rayon writes directly into output buffer (no copy)
    // Each 2 MiB chunk gets a unique seed; page faults distributed across all cores.
    const CHUNK: usize = 2 * 1024 * 1024;
    buf[off_x_data..off_x_data + x_data_size]
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(i as u64);
            rng.fill_bytes(chunk);
        });

    // CRC32 over x.npy content -- pages already hot, fast L3/RAM read
    let crc_x = crc32fast::hash(&buf[off_x_npy_hdr..off_x_npy_hdr + x_npy_size]);

    // Patch x.npy local header
    let x_npy_size_u32 = x_npy_size as u32;
    buf[off_x_local + 14..off_x_local + 18].copy_from_slice(&crc_x.to_le_bytes());
    buf[off_x_local + 18..off_x_local + 22].copy_from_slice(&x_npy_size_u32.to_le_bytes());
    buf[off_x_local + 22..off_x_local + 26].copy_from_slice(&x_npy_size_u32.to_le_bytes());

    // y.npy local header
    write_local_file_header(&mut buf[off_y_local..], name_y, 0, 0);

    // y.npy NPY header
    buf[off_y_npy_hdr..off_y_npy_hdr + npy_hdr_y.len()].copy_from_slice(&npy_hdr_y);

    // y.npy data (int64 zeros)
    buf[off_y_data..off_y_data + y_data_size].fill(0);

    // CRC32 for y.npy
    let crc_y = crc32fast::hash(&buf[off_y_npy_hdr..off_y_npy_hdr + y_npy_size]);

    // Patch y.npy local header
    let y_npy_size_u32 = y_npy_size as u32;
    buf[off_y_local + 14..off_y_local + 18].copy_from_slice(&crc_y.to_le_bytes());
    buf[off_y_local + 18..off_y_local + 22].copy_from_slice(&y_npy_size_u32.to_le_bytes());
    buf[off_y_local + 22..off_y_local + 26].copy_from_slice(&y_npy_size_u32.to_le_bytes());

    // Central directory
    write_central_dir_entry(
        &mut buf[off_cd..],
        name_x,
        x_npy_size_u32,
        crc_x,
        off_x_local as u32,
    );
    write_central_dir_entry(
        &mut buf[off_cd_y..],
        name_y,
        y_npy_size_u32,
        crc_y,
        off_y_local as u32,
    );

    // EOCD
    let cd_total = (x_cd_entry + y_cd_entry) as u32;
    write_eocd(&mut buf[off_eocd..], 2, cd_total, off_cd as u32);

    debug_assert_eq!(buf.len(), total, "NPZ size mismatch");
    Ok(buf)
}

// =============================================================================

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
        return Err(anyhow!(
            "Unsupported NPY version {}.{} (only 1.0 supported)",
            major,
            minor
        ));
    }

    // Read header length
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header_start = 10;
    let header_end = header_start + header_len;

    if data.len() < header_end {
        return Err(anyhow!(
            "NPY file truncated (header claims {} bytes)",
            header_len
        ));
    }

    // Parse header (Python dict as string)
    let header_str = std::str::from_utf8(&data[header_start..header_end])
        .context("NPY header is not valid UTF-8")?;

    // Extract shape from header
    // Format: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
    let shape = parse_npy_shape(header_str)?;

    // Validate dtype (we only support '<f4' = little-endian float32)
    if !header_str.contains("'<f4'") && !header_str.contains("'<f'") {
        return Err(anyhow!(
            "Unsupported NPY dtype (only '<f4' supported), header: {}",
            header_str
        ));
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
            num_elements,
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
    let shape_start = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| anyhow!("NPY header missing 'shape' field"))?;

    let after_shape = &header[shape_start..];
    let tuple_start = after_shape
        .find('(')
        .ok_or_else(|| anyhow!("NPY header 'shape' field malformed (no '(')"))?;
    let tuple_end = after_shape
        .find(')')
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
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .with_context(|| format!("Invalid dimension in shape: '{}'", s))
        })
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
    let mut archive = ZipArchive::new(cursor).context("Failed to open NPZ file as ZIP archive")?;

    // Find the array by name (with or without .npy extension)
    let npy_name = if array_name.ends_with(".npy") {
        array_name.to_string()
    } else {
        format!("{}.npy", array_name)
    };

    let mut file = archive
        .by_name(&npy_name)
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
    let archive = ZipArchive::new(cursor).context("Failed to open NPZ file as ZIP archive")?;

    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            archive.name_for_index(i).map(|name| {
                // Strip .npy extension
                if let Some(stripped) = name.strip_suffix(".npy") {
                    stripped.to_string()
                } else {
                    name.to_string()
                }
            })
        })
        .collect();

    Ok(names)
}
