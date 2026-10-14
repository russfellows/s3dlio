// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Regression tests for s3dlio issue #187: an unparsable `start` silently became "now".
//! (Op names `OpType` does not model remain a clear error; see the last section.)

mod oplog_support;

use oplog_support::{replay, row, tsv};
use s3dlio_oplog::{OpLogReader, OpLogStreamReader, OpType};
use std::io::Write;
use tempfile::NamedTempFile;

// ----------------------------------------------------------------------------
// #187: `start` handling
// ----------------------------------------------------------------------------

fn read_err(f: &NamedTempFile) -> String {
    match OpLogReader::from_file(f.path()) {
        Ok(r) => panic!("expected an error, got {} entries", r.len()),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn unparsable_start_is_an_error_naming_the_line() {
    for bad in ["5", "1000000000", "", "yesterday", "2026-01-01 00:00:00"] {
        let f = tsv(&[
            row(0, "GET", "s3://bkt", "bkt/a"),
            format!("1\t1\tGET\t\t1\t10\ts3://bkt\tbkt/b\t\t{bad}\t\t\t1000"),
        ]);
        let msg = read_err(&f);
        assert!(
            msg.contains("line 3") && msg.contains("start"),
            "start={bad:?}: {msg}"
        );
    }
}

#[test]
fn stream_reader_reports_only_the_bad_row() {
    let f = tsv(&[
        row(0, "GET", "s3://bkt", "bkt/a"),
        "1\t1\tGET\t\t1\t10\ts3://bkt\tbkt/b\t\tnot-a-time\t\t\t1000".to_string(),
        row(2, "GET", "s3://bkt", "bkt/c"),
    ]);
    let results: Vec<_> = OpLogStreamReader::from_file(f.path())
        .expect("open")
        .collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    assert!(results[2].is_ok());
    let msg = format!("{:#}", results[1].as_ref().expect_err("bad row"));
    assert!(
        msg.contains("line 3") && msg.contains("not-a-time"),
        "{msg}"
    );
}

#[test]
fn valid_rfc3339_start_is_unchanged() {
    let f = tsv(&[
        "0\t1\tGET\t\t1\t10\ts3://bkt\tbkt/a\t\t2026-01-01T02:00:05.250+02:00\t\t\t1000"
            .to_string(),
        "1\t1\tGET\t\t1\t10\ts3://bkt\tbkt/b\t\t2026-01-01T00:00:07Z\t\t\t1000".to_string(),
    ]);
    let r = OpLogReader::from_file(f.path()).expect("valid rows");
    assert_eq!(
        r.entries()[0].start.to_rfc3339(),
        "2026-01-01T00:00:05.250+00:00"
    );
    assert_eq!(
        r.entries()[1].start.to_rfc3339(),
        "2026-01-01T00:00:07+00:00"
    );
}

#[test]
fn log_without_any_start_column_is_untimed_not_an_error() {
    // The minimal dl-driver format has no timing at all: it still loads (rows take the read
    // time, with a one-time warning), unlike a present-but-unparsable `start`.
    let mut f = NamedTempFile::with_suffix(".tsv").expect("temp file");
    writeln!(f, "idx\top\tendpoint\tfile").expect("w");
    writeln!(f, "0\tGET\ts3://bkt\tbkt/a").expect("w");
    f.flush().expect("flush");
    let r = OpLogReader::from_file(f.path()).expect("untimed log loads");
    assert_eq!(r.len(), 1);
    let age = chrono::Utc::now() - r.entries()[0].start;
    assert!(age.num_seconds().abs() < 60, "{age}");
}

#[test]
fn header_without_an_operation_column_is_reported_to_the_caller() {
    // Before the fix the background parser's error was only logged and the file read as empty.
    let mut f = NamedTempFile::with_suffix(".tsv").expect("temp file");
    writeln!(f, "idx\tendpoint\tfile\tstart").expect("w");
    writeln!(f, "0\ts3://bkt\tbkt/a\t2026-01-01T00:00:00Z").expect("w");
    f.flush().expect("flush");
    let msg = read_err(&f);
    assert!(msg.contains("operation"), "{msg}");
}

#[test]
fn t_start_ns_is_read_as_nanoseconds_since_the_epoch() {
    let mut f = NamedTempFile::with_suffix(".tsv").expect("temp file");
    writeln!(f, "operation\tendpoint\tfile\tt_start_ns").expect("w");
    writeln!(f, "GET\ts3://bkt\tbkt/a\t1767225600000000000").expect("w");
    f.flush().expect("flush");
    let r = OpLogReader::from_file(f.path()).expect("t_start_ns rows");
    assert_eq!(
        r.entries()[0].start.to_rfc3339(),
        "2026-01-01T00:00:00+00:00"
    );
}

#[test]
fn jsonl_bad_start_is_an_error_naming_the_line() {
    for body in [
        r#"{"op":"GET","endpoint":"s3://bkt","file":"bkt/b","start":12345}"#,
        r#"{"op":"GET","endpoint":"s3://bkt","file":"bkt/b","start":"soon"}"#,
    ] {
        let mut f = NamedTempFile::with_suffix(".jsonl").expect("temp file");
        writeln!(
            f,
            r#"{{"op":"GET","endpoint":"s3://bkt","file":"bkt/a","start":"2026-01-01T00:00:00Z"}}"#
        )
        .expect("w");
        writeln!(f, "{body}").expect("w");
        f.flush().expect("flush");
        let msg = read_err(&f);
        assert!(
            msg.contains("line 2") && msg.contains("start"),
            "{body}: {msg}"
        );
    }
}

// ----------------------------------------------------------------------------
// #187: op names outside the five OpType variants stay a clear error
// ----------------------------------------------------------------------------

/// Operation names the object-store writer can emit that `OpType` does not model. `OpType` is
/// matched exhaustively by downstream crates, so it keeps exactly its five variants and these
/// names remain what they always were: a clear per-row error, never silent acceptance.
const UNMODELLED_WRITER_OPS: &[&str] = &[
    "GET_RANGE",
    "DELETE_BATCH",
    "DELETE_PREFIX",
    "CREATE_CONTAINER",
    "DELETE_CONTAINER",
];

#[test]
fn unmodelled_op_names_do_not_parse() {
    for name in UNMODELLED_WRITER_OPS {
        let err = name.parse::<OpType>().expect_err(name);
        assert!(
            format!("{err:#}").contains(&format!("Unknown operation type: {name}")),
            "{name}: {err:#}"
        );
    }
    assert_eq!("HEAD".parse::<OpType>().expect("HEAD"), OpType::STAT);
}

#[test]
fn reader_rejects_a_log_containing_an_unmodelled_op_with_a_clear_error() {
    for name in UNMODELLED_WRITER_OPS {
        let f = tsv(&[
            row(0, "GET", "s3://bkt", "bkt/a"),
            row(1, name, "s3://bkt", "bkt/b"),
        ]);
        let msg = read_err(&f);
        assert!(
            msg.contains(&format!("Invalid operation type: {name}"))
                && msg.contains("Unknown operation type"),
            "{name}: {msg}"
        );
    }
}

#[test]
fn stream_reader_reports_only_the_unmodelled_row() {
    let f = tsv(&[
        row(0, "GET", "s3://bkt", "bkt/a"),
        row(1, "DELETE_BATCH", "s3://bkt", "bkt/b"),
        row(2, "GET", "s3://bkt", "bkt/c"),
    ]);
    let results: Vec<_> = OpLogStreamReader::from_file(f.path())
        .expect("open")
        .collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok() && results[2].is_ok());
    let msg = format!("{:#}", results[1].as_ref().expect_err("unmodelled row"));
    assert!(msg.contains("DELETE_BATCH"), "{msg}");
}

#[tokio::test]
async fn replay_of_a_log_with_an_unmodelled_op_fails_and_executes_nothing() {
    let f = tsv(&[
        row(0, "GET", "s3://bkt", "bkt/a"),
        row(1, "GET_RANGE", "s3://bkt", "bkt/b"),
    ]);
    let (res, seen) = replay(&f, None, false, None).await;
    let err = format!("{:#}", res.expect_err("must fail"));
    assert!(err.contains("GET_RANGE"), "{err}");
    assert!(seen.is_empty(), "{seen:?}");
}
