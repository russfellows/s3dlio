// tests/common/oplog_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Reads a finished s3dlio op-log back as rows, for the exactly-once tests.

use std::path::Path;

/// The 13-column header every op-log starts with.
pub const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

pub const COL_OP: usize = 2;
pub const COL_CLIENT_ID: usize = 3;
pub const COL_N_OBJECTS: usize = 4;
pub const COL_BYTES: usize = 5;
pub const COL_ENDPOINT: usize = 6;
pub const COL_FILE: usize = 7;
pub const COL_ERROR: usize = 8;

/// One data row, split into its 13 columns.
pub type Row = Vec<String>;

/// Decode a finalized op-log (zstd TSV) into data rows. Asserts the header and
/// that every row has 13 columns and a sequential `idx`.
pub fn read_rows(path: &Path) -> Vec<Row> {
    let raw = std::fs::read(path).expect("op-log file must exist");
    let text = String::from_utf8(
        zstd::decode_all(raw.as_slice()).expect("op-log must be a complete zstd stream"),
    )
    .expect("op-log must be UTF-8");
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(HEADER_LINE), "header line");
    let rows: Vec<Row> = lines
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    for (i, cols) in rows.iter().enumerate() {
        assert_eq!(cols.len(), 13, "row {i} must have 13 columns: {cols:?}");
        assert_eq!(cols[0], i.to_string(), "row {i} idx");
    }
    rows
}

/// The `op` column of every row, in file order.
pub fn ops(rows: &[Row]) -> Vec<&str> {
    rows.iter().map(|r| r[COL_OP].as_str()).collect()
}
