// tests/object_format_tests.rs

use bytes::Bytes;
use std::convert::TryInto;
use std::io::Cursor;
use zip::ZipArchive;

use s3dlio::data_formats::{
    build_raw, build_npz, build_hdf5, tfrecord::TfRecordWithIndex,
};

use s3dlio::data_formats::tfrecord::build_tfrecord_with_index;

#[test]
fn raw_roundtrip() -> anyhow::Result<()> {
    let data = vec![0u8, 1, 2, 3, 4, 5, 255];
    let out = build_raw(&data)?;
    assert_eq!(out, Bytes::from(data.clone()));
    Ok(())
}

#[test]
fn npz_roundtrip_and_content() -> anyhow::Result<()> {
    // 5 elements of size 1 byte each, filled with 0x42
    let elements = 5;
    let element_size = 1;
    let data = vec![0x42u8; elements * element_size];

    // Build NPZ in memory
    let npz_bytes = build_npz(elements, element_size, &data)?;
    let buf = npz_bytes.to_vec();
    let cursor = Cursor::new(buf);
    let mut zip = ZipArchive::new(cursor)?;
    // exactly one file
    assert_eq!(zip.len(), 1, "Expected 1 entry, found {}", zip.len());

    let mut file = zip.by_index(0)?;
    assert_eq!(file.name(), "data.npy", "NPZ entry has wrong name");

    // Read full content
    let mut content = Vec::new();
    std::io::copy(&mut file, &mut content)?;

    // Check NumPy magic header: \x93NUMPY
    assert!(content.starts_with(b"\x93NUMPY"), "Missing .npy magic");

    // Next two bytes are little-endian header length
    let header_len = u16::from_le_bytes([content[8], content[9]]) as usize;
    // Total header size = magic+version(8) + len-field(2) + header_len
    let total_header = 8 + 2 + header_len;
    let payload = &content[total_header..];
    assert_eq!(payload, data.as_slice(), "Payload mismatch");

    Ok(())
}

#[test]
fn hdf5_signature_and_dataset() -> anyhow::Result<()> {
    let elements = 10;
    let element_size = 2;
    let data: Vec<u8> = (0..elements*element_size).map(|i| (i % 256) as u8).collect();

    let h5_bytes = build_hdf5(elements, element_size, &data)?;
    // HDF5 signature
    const HDF5_SIG: &[u8;8] = b"\x89HDF\r\n\x1A\n";
    assert!(h5_bytes.len() > 8, "HDF5 image too small");
    assert_eq!(&h5_bytes[..8], HDF5_SIG, "Invalid HDF5 signature");

    // The name of the dataset ("data") should be embedded somewhere
    assert!(
        h5_bytes.windows(4).any(|w| w == b"data"),
        "Dataset name 'data' not found in HDF5 image"
    );

    Ok(())
}

#[test]
fn tfrecord_index_and_stream_consistency() -> anyhow::Result<()> {
    let records = 3;
    let rec_size = 4;
    // fill with sequential bytes
    let mut data = Vec::with_capacity(records * rec_size);
    for i in 0..(records * rec_size) {
        data.push((i % 256) as u8);
    }

    let TfRecordWithIndex { data: tf_bytes, index: idx_bytes } =
        build_tfrecord_with_index(records, rec_size, &data)?;

    // Index should be exactly 16 bytes per record (u64 offset + u64 length)
    assert_eq!(idx_bytes.len(), records * 16);

    // Parse first entry
    let offset0 = u64::from_le_bytes(idx_bytes[0..8].try_into().unwrap());
    let length0 = u64::from_le_bytes(idx_bytes[8..16].try_into().unwrap());
    assert_eq!(offset0, 0, "First record offset should be 0");

    // Each record on-wire = 8(len) +4(len-CRC) + data +4(data-CRC)
    let expected_len0 = 8 + 4 + rec_size as u64 + 4;
    assert_eq!(length0, expected_len0, "First record length mismatch");

    // Sum all indexed lengths and compare to the TFRecord stream length
    let total: u64 = (0..records)
        .map(|i| {
            let start = i * 16;
            u64::from_le_bytes(idx_bytes[start+8..start+16].try_into().unwrap())
        })
        .sum();
    assert_eq!(
        tf_bytes.len() as u64,
        total,
        "Total TFRecord stream length != sum of indexed lengths"
    );

    Ok(())
}

