// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Regression tests for s3dlio issue #176 (reader half): an op name the reader does not model
//! must produce an error that names the file line, the op, and that the op is unsupported for
//! exact replay. Replay behaviour is unchanged: the load still aborts at the first such row.

mod oplog_support;

use oplog_support::{replay, row, tsv};
use s3dlio_oplog::{OpLogReader, OpLogStreamReader};
use std::io::Write;
use tempfile::NamedTempFile;

const COMPOUND_OPS: &[&str] = &[
    "GET_RANGE",
    "DELETE_BATCH",
    "DELETE_PREFIX",
    "CREATE_CONTAINER",
    "DELETE_CONTAINER",
];

fn jsonl(rows: &[&str]) -> NamedTempFile {
    let mut f = NamedTempFile::with_suffix(".jsonl").expect("temp file");
    for r in rows {
        writeln!(f, "{r}").expect("write row");
    }
    f.flush().expect("flush");
    f
}

fn batch_err(f: &NamedTempFile) -> String {
    match OpLogReader::from_file(f.path()) {
        Ok(r) => panic!("expected an error, got {} entries", r.len()),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn batch_reader_error_names_line_and_op_tsv() {
    for name in COMPOUND_OPS {
        // header is line 1, the GET is line 2, the unsupported row is line 3
        let f = tsv(&[
            row(0, "GET", "s3://bkt", "bkt/a"),
            row(1, name, "s3://bkt", "bkt/b"),
        ]);
        let msg = batch_err(&f);
        assert!(msg.contains("line 3"), "{name}: {msg}");
        assert!(
            msg.contains(&format!("operation {name} is not supported for exact replay")),
            "{name}: {msg}"
        );
    }
}

#[test]
fn batch_reader_error_names_line_and_op_jsonl() {
    for name in COMPOUND_OPS {
        let good = r#"{"op": "GET", "file": "a", "start": "2026-01-01T00:00:00Z"}"#;
        let bad = format!(r#"{{"op": "{name}", "file": "b", "start": "2026-01-01T00:00:01Z"}}"#);
        let f = jsonl(&[good, &bad]);
        let msg = batch_err(&f);
        assert!(msg.contains("line 2"), "{name}: {msg}");
        assert!(
            msg.contains(&format!("operation {name} is not supported for exact replay")),
            "{name}: {msg}"
        );
    }
}

#[test]
fn stream_reader_error_names_line_row_and_op() {
    let f = tsv(&[
        row(0, "GET", "s3://bkt", "bkt/a"),
        row(1, "GET_RANGE", "s3://bkt", "bkt/b"),
        row(2, "GET", "s3://bkt", "bkt/c"),
    ]);
    let results: Vec<_> = OpLogStreamReader::from_file(f.path())
        .expect("open")
        .collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok() && results[2].is_ok());
    let msg = format!("{:#}", results[1].as_ref().expect_err("unsupported row"));
    assert!(msg.contains("line 3"), "{msg}");
    assert!(msg.contains("row index 1"), "{msg}");
    assert!(
        msg.contains("operation GET_RANGE is not supported for exact replay"),
        "{msg}"
    );
}

#[test]
fn a_name_that_is_not_a_known_writer_op_is_called_unknown_not_unsupported() {
    let f = tsv(&[
        row(0, "GET", "s3://bkt", "bkt/a"),
        row(1, "FROBNICATE", "s3://bkt", "bkt/b"),
    ]);
    let msg = batch_err(&f);
    assert!(msg.contains("line 3") && msg.contains("FROBNICATE"), "{msg}");
    assert!(msg.contains("unknown operation"), "{msg}");
    assert!(!msg.contains("not supported for exact replay"), "{msg}");
}

#[tokio::test]
async fn replay_still_aborts_at_load_and_the_error_names_the_line() {
    let f = tsv(&[
        row(0, "GET", "s3://bkt", "bkt/a"),
        row(1, "GET_RANGE", "s3://bkt", "bkt/b"),
    ]);
    // continue_on_error does not bypass load errors: the same abort in both modes.
    for continue_on_error in [false, true] {
        let (res, seen) = replay(&f, None, continue_on_error, None).await;
        let err = format!("{:#}", res.expect_err("must fail"));
        assert!(err.contains("line 3") && err.contains("GET_RANGE"), "{err}");
        assert!(seen.is_empty(), "{seen:?}");
    }
}
