// tests/oplog_lifecycle_same_file_spellings.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294 (round 2).
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: while Active, every spelling of the ACTIVE file is the same file and
// init is a no-op (x, ./x, /abs/x, file:///abs/x, dir/../x, a symlink to it, a
// hard link to it, a symlink chain); any other file is an error and is not
// created. The comparison is by identity ((dev, inode) on Unix), not by text.

use s3dlio::s3_logger::{LogEntry, LogOutcome};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, op_log_stats};
use std::os::unix::fs::symlink;
use std::path::Path;
use std::time::SystemTime;

fn entry(tag: &str) -> LogEntry {
    let now = SystemTime::now();
    LogEntry {
        idx: 0,
        thread_id: 0,
        operation: "GET".to_string(),
        client_id: String::new(),
        num_objects: 1,
        bytes: 1,
        endpoint: "file://".to_string(),
        file: tag.to_string(),
        error: None,
        start_time: now,
        first_byte_time: None,
        end_time: now,
    }
}

const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

/// Decompress an op-log (it must be a complete zstd stream), check the full
/// header line, 13 columns and sequential idx on every row, and return the
/// `file` column of every data row.
fn logged_tags(path: &Path) -> Vec<String> {
    let raw = std::fs::read(path).expect("read op-log");
    let text = zstd::decode_all(raw.as_slice()).expect("op-log must be a complete zstd stream");
    let text = String::from_utf8(text).expect("op-log must be UTF-8");
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(HEADER_LINE),
        "op-log must start with the exact header row"
    );
    lines
        .enumerate()
        .map(|(i, l)| {
            let cols: Vec<&str> = l.split('\t').collect();
            assert_eq!(cols.len(), 13, "row {i} must have 13 columns: {l:?}");
            assert_eq!(cols[0], i.to_string(), "row {i} must carry idx {i}");
            cols[7].to_string()
        })
        .collect()
}

#[test]
fn same_file_in_every_spelling_is_a_noop_and_other_files_are_errors() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    // Canonicalize the base: on some systems the temp dir itself is behind a symlink.
    let base = std::fs::canonicalize(dir.path()).unwrap();
    std::fs::create_dir(base.join("sub")).unwrap();
    std::env::set_current_dir(&base).unwrap();

    let abs = base.join("x");
    init_op_logger("x").unwrap(); // relative spelling creates ./x
    assert!(abs.exists());
    let logger = global_logger().unwrap();
    assert_eq!(logger.try_log(entry("one")), LogOutcome::Accepted);

    symlink(&abs, base.join("link_to_x")).unwrap();
    symlink(base.join("link_to_x"), base.join("sub").join("chain")).unwrap();
    std::fs::hard_link(&abs, base.join("hard_to_x")).unwrap();

    let abs_s = abs.to_str().unwrap().to_string();
    let same: Vec<String> = vec![
        "x".to_string(),
        "./x".to_string(),
        abs_s.clone(),
        format!("file://{abs_s}"),
        "sub/../x".to_string(),
        format!("{}/sub/../x", base.display()),
        "link_to_x".to_string(),
        "sub/chain".to_string(),
        "hard_to_x".to_string(),
        format!("file://{}/link_to_x", base.display()),
    ];
    for spelling in &same {
        init_op_logger(spelling)
            .unwrap_or_else(|e| panic!("{spelling:?} is the active file, must be a no-op: {e}"));
    }
    assert_eq!(logger.try_log(entry("two")), LogOutcome::Accepted);
    assert_eq!(
        op_log_stats().unwrap().accepted,
        2,
        "still the first session"
    );

    // Different files (and unusable destinations): all errors, nothing created.
    std::fs::write(base.join("other"), b"precious").unwrap();
    symlink(base.join("other"), base.join("link_to_other")).unwrap();
    // (spelling, is it an unusable destination rather than merely another file?)
    let different: Vec<(String, bool)> = vec![
        ("y".to_string(), false),
        ("./sub/x".to_string(), false), // same NAME, different directory
        ("x2".to_string(), false),
        ("sub/y".to_string(), false),
        ("no_such_dir/x".to_string(), false),
        ("other".to_string(), false),
        ("link_to_other".to_string(), false),
        (format!("file://{}/y", base.display()), false),
        ("file://".to_string(), true),
        ("s3://bucket/key".to_string(), true),
    ];
    for (spelling, unusable) in &different {
        let err = init_op_logger(spelling)
            .expect_err(&format!("{spelling:?} is not the active file, must fail"));
        if *unusable {
            // Unresolvable while Active is an error too: the usual diagnostics.
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "{spelling:?}: {err}"
            );
        } else {
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::ResourceBusy,
                "{spelling:?}: {err}"
            );
            assert!(
                err.to_string().contains(&abs_s),
                "must name the active file: {err}"
            );
        }
    }
    for gone in ["y", "sub/x", "x2", "sub/y"] {
        assert!(!base.join(gone).exists(), "{gone} must not be created");
    }
    assert_eq!(
        std::fs::read(base.join("other")).unwrap(),
        b"precious",
        "not truncated"
    );

    drop(logger);
    finalize_op_logger();
    assert_eq!(
        logged_tags(&abs),
        vec!["one".to_string(), "two".to_string()]
    );
}
