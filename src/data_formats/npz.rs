use anyhow::Result;
use bytes::Bytes;
use std::io::{Cursor, Write};
use zip::{write::FileOptions, CompressionMethod, ZipWriter};


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
pub fn build_npz(elements: usize, _element_size: usize, data: &[u8]) -> Result<Bytes> {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    {
        let mut zip = ZipWriter::new(&mut cursor);
        let opts = FileOptions::default()
            .compression_method(CompressionMethod::Stored);

        zip.start_file("data.npy", opts)?;
        zip.write_all(&make_npy_header(elements))?;   // HEADER KEPT
        zip.write_all(data)?;
        zip.finish()?;
    } // <- ZipWriter dropped, borrow ends

    Ok(Bytes::from(cursor.into_inner()))
}

