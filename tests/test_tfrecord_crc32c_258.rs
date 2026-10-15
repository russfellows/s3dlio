// tests/test_tfrecord_crc32c_258.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #258: the TFRecord writer computed its masked length-CRC
// and data-CRC with CRC-32 (IEEE). The TFRecord format requires CRC-32C (Castagnoli),
// masked as `((crc >> 15) | (crc << 17)) + 0xa282ead8`; a reader that verifies (TensorFlow,
// TensorBoard) rejects the old files.
//
// The reference vector below was NOT produced by s3dlio. It was written by an independent
// pure-Python, table-free bitwise CRC-32C (polynomial 0x82F63B78) following the TFRecord
// spec, for two records of 4 bytes cut from the bytes 0..8, and then read back and
// CRC-verified by TensorBoard's TensorFlow-stub `PyRecordReader` (tensorboard from PyPI,
// whose reader raises DataLossError on a bad masked CRC). That reader accepts this vector
// and rejects the same bytes with a flipped last byte, and it rejects what s3dlio wrote
// before the fix.

use s3dlio::data_formats::tfrecord::{build_tfrecord, build_tfrecord_with_index};

/// length(8, LE) | masked-crc(length)(4) | data | masked-crc(data)(4), twice.
const REFERENCE: &str = "040000000000000042455204000102033e9dcad7\
                         040000000000000042455204040506079656969\
                         5";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn payload() -> Vec<u8> {
    (0u8..8).collect()
}

#[test]
fn tfrecord_bytes_match_the_independently_computed_reference() {
    let got = build_tfrecord(2, 4, &payload()).unwrap();
    let want: String = REFERENCE.split_whitespace().collect();
    assert_eq!(hex(&got), want);
}

/// A bitwise CRC-32C written for this test (no table, no crate), checked against the
/// standard check value, then used to verify every CRC in the generated stream.
fn crc32c_bitwise(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 == 1 { 0x82F6_3B78 } else { 0 };
        }
    }
    !crc
}

fn masked(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

#[test]
fn every_crc_in_the_stream_is_a_masked_crc32c() {
    assert_eq!(crc32c_bitwise(b"123456789"), 0xe306_9283);
    // Different sizes, including an empty record and one larger than a CRC block.
    for (records, size) in [(1usize, 0usize), (3, 1), (2, 37), (1, 5000)] {
        let data: Vec<u8> = (0..records * size).map(|i| (i * 7 % 251) as u8).collect();
        let tf = build_tfrecord_with_index(records, size, &data).unwrap();
        let mut pos = 0usize;
        for i in 0..records {
            let len_bytes: [u8; 8] = tf.data[pos..pos + 8].try_into().unwrap();
            assert_eq!(u64::from_le_bytes(len_bytes) as usize, size);
            let len_crc = u32::from_le_bytes(tf.data[pos + 8..pos + 12].try_into().unwrap());
            assert_eq!(
                len_crc,
                masked(crc32c_bitwise(&len_bytes)),
                "length CRC, record {i}"
            );
            let body = &tf.data[pos + 12..pos + 12 + size];
            let data_crc = u32::from_le_bytes(
                tf.data[pos + 12 + size..pos + 16 + size]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                data_crc,
                masked(crc32c_bitwise(body)),
                "data CRC, record {i}"
            );
            pos += 16 + size;
        }
        assert_eq!(pos, tf.data.len());
    }
}
