use anyhow::Result;
use bytes::Bytes;
use crc32fast::Hasher;
use std::io::{Cursor, Write};

/* ---------- low‑level helpers, unchanged ---------- */
fn mask_crc(crc: u32) -> u32 {
    (((crc >> 15) | (crc << 17)).wrapping_add(0xa282_ead8)) & 0xFFFF_FFFF
}

fn write_raw_record<W: Write>(w: &mut W, data: &[u8]) -> std::io::Result<usize> {
    // 1 length
    let len = data.len() as u64;
    let len_buf = len.to_le_bytes();
    w.write_all(&len_buf)?;

    // 2 masked len‑CRC
    let mut h = Hasher::new();
    h.update(&len_buf);
    w.write_all(&mask_crc(h.finalize()).to_le_bytes())?;

    // 3 data + 4 masked data‑CRC
    w.write_all(data)?;
    let mut h = Hasher::new();
    h.update(data);
    w.write_all(&mask_crc(h.finalize()).to_le_bytes())?;

    Ok(8 + 4 + data.len() + 4)
}
/* -------------------------------------------------- */

/// Return only the TFRecord stream (`Bytes`).
pub fn build_tfrecord(records: usize, record_size: usize, data: &[u8]) -> Result<Bytes> {
    Ok(build_tfrecord_with_index(records, record_size, data)?.data)
}

/// TFRecord + matching 16‑byte‑per‑record index.
pub struct TfRecordWithIndex {
    pub data:  Bytes,
    pub index: Bytes,
}

/// Build both the record stream *and* its index.
pub fn build_tfrecord_with_index(
    records: usize,
    record_size: usize,
    data: &[u8],
) -> Result<TfRecordWithIndex> {
    let mut tf_buf   = Cursor::new(Vec::<u8>::new());
    let mut idx_buf  = Cursor::new(Vec::<u8>::new());

    let mut offset: u64 = 0;

    for i in 0..records {
        let start = i * record_size;
        let end   = start + record_size;

        // write record, capture its on‑wire length
        let written = write_raw_record(&mut tf_buf, &data[start..end])? as u64;

        // append <offset, length> to index
        idx_buf.write_all(&offset.to_le_bytes())?;
        idx_buf.write_all(&written.to_le_bytes())?;

        offset += written;
    }

    Ok(TfRecordWithIndex {
        data:  Bytes::from(tf_buf.into_inner()),
        index: Bytes::from(idx_buf.into_inner()),
    })
}

