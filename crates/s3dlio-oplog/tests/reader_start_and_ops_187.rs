// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Regression tests for s3dlio issue #187: an unparsable `start` silently became "now",
//! and the op names the writer emits (`DELETE_BATCH`, ...) were rejected by the reader.

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
// #187: op names the writer emits
// ----------------------------------------------------------------------------

/// Every operation name `LoggedObjectStore` / `S3Ops` can write (HEAD is an alias of STAT).
const WRITER_OPS: &[&str] = &[
    "GET",
    "GET_RANGE",
    "PUT",
    "LIST",
    "STAT",
    "DELETE",
    "DELETE_BATCH",
    "DELETE_PREFIX",
    "CREATE_CONTAINER",
    "DELETE_CONTAINER",
];

#[test]
fn every_writer_op_name_parses_and_round_trips() {
    for name in WRITER_OPS {
        let op: OpType = name.parse().unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(op.to_string(), *name);
    }
    assert_eq!("HEAD".parse::<OpType>().expect("HEAD"), OpType::STAT);
}

#[test]
fn reader_loads_a_log_containing_every_writer_op() {
    let rows: Vec<String> = WRITER_OPS
        .iter()
        .enumerate()
        .map(|(i, op)| row(i as u32, op, "s3://bkt", "bkt/k"))
        .collect();
    let f = tsv(&rows);
    let r = OpLogReader::from_file(f.path()).expect("all writer ops must load");
    assert_eq!(r.len(), WRITER_OPS.len());
}

#[tokio::test]
async fn compound_ops_are_unsupported_for_exact_replay_and_never_executed() {
    for name in [
        "DELETE_BATCH",
        "DELETE_PREFIX",
        "GET_RANGE",
        "CREATE_CONTAINER",
        "DELETE_CONTAINER",
    ] {
        let f = tsv(&[
            row(0, "GET", "s3://bkt", "bkt/a"),
            row(1, name, "s3://bkt", "bkt/b"),
            row(2, "GET", "s3://bkt", "bkt/c"),
        ]);

        // Default: the replay fails and says why.
        let (res, _) = replay(&f, None, false, None).await;
        let err = format!("{:#}", res.expect_err(name));
        assert!(
            err.contains("unsupported for exact replay") && err.contains(name),
            "{name}: {err}"
        );

        // continue_on_error: the plain rows run, the compound row is never executed.
        let (res, seen) = replay(&f, None, true, None).await;
        assert!(res.is_ok(), "{name}: {res:?}");
        assert_eq!(
            seen,
            vec!["GET s3://bkt/a".to_string(), "GET s3://bkt/c".to_string()],
            "{name}"
        );

        // The existing opt-in (filter_ops) is the way to leave them out explicitly.
        let (res, seen) = replay(&f, None, false, Some(vec![OpType::GET])).await;
        assert!(res.is_ok(), "{name} filtered: {res:?}");
        assert_eq!(seen.len(), 2, "{name}");
    }
}
