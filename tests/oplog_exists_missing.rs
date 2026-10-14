// tests/oplog_exists_missing.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #189: `exists()` returning false was recorded as a SUCCESSFUL `HEAD`
// row (empty error), although the same probe done with `stat()` is a failed
// `HEAD` row with the error filled. A reader or replayer could not tell the
// negative probe from a successful HEAD.
//
// The `exists` row now records a missing object the way `stat` records it: op
// `HEAD`, the error column holding the very error the failed HEAD returned (so it
// is backend-specific text, identical to the stat row's for the same URI). A
// present object is still a `HEAD` row without an error.
//
// Own process (op-log + S3 client are process-global); mock S3 server for the S3
// case, a real directory for the file case.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{ensure_mock_s3_server, unique_test_key, HEAD_NOT_FOUND_MARKER};
use common::oplog_rows::{ops, read_rows, COL_BYTES, COL_ERROR, COL_FILE, COL_N_OBJECTS};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};

#[tokio::test]
async fn a_missing_object_probe_is_recorded_like_a_failed_head() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    ensure_mock_s3_server();
    let work = tempfile::tempdir().unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let file_store = store_for_uri_with_logger(
        &format!("file://{}", work.path().display()),
        global_logger(),
    )
    .unwrap();
    let present = format!("file://{}/present.bin", work.path().display());
    let absent = format!("file://{}/absent.bin", work.path().display());
    file_store
        .put(&present, Bytes::from_static(b"abc"))
        .await
        .unwrap();

    let s3_store = store_for_uri_with_logger("s3://mock-bucket/", global_logger()).unwrap();
    let s3_absent = format!(
        "s3://mock-bucket/{}",
        unique_test_key(HEAD_NOT_FOUND_MARKER)
    );

    // Rows 1..: file present, file absent (exists then stat), s3 absent (exists then stat).
    assert!(file_store.exists(&present).await.unwrap());
    assert!(!file_store.exists(&absent).await.unwrap());
    assert!(file_store.stat(&absent).await.is_err());
    assert!(!s3_store.exists(&s3_absent).await.unwrap());
    assert!(s3_store.stat(&s3_absent).await.is_err());

    finalize_op_logger();

    let rows = read_rows(&oplog);
    assert_eq!(
        ops(&rows),
        ["PUT", "HEAD", "HEAD", "HEAD", "HEAD", "HEAD"],
        "rows: {rows:#?}"
    );
    let (present_row, file_exists, file_stat, s3_exists, s3_stat) =
        (&rows[1], &rows[2], &rows[3], &rows[4], &rows[5]);

    // A present object: a HEAD with no error.
    assert_eq!(present_row[COL_ERROR], "");
    assert_eq!(present_row[COL_N_OBJECTS], "1");

    // A missing object: a HEAD whose error column is the failed HEAD's error,
    // exactly as stat() records it for the same URI.
    for (name, exists, stat) in [("file", file_exists, file_stat), ("s3", s3_exists, s3_stat)] {
        assert!(
            !exists[COL_ERROR].is_empty(),
            "{name}: exists()==false must not be a successful HEAD: {exists:?}"
        );
        assert_eq!(exists[COL_ERROR], stat[COL_ERROR], "{name} error text");
        assert_eq!(exists[COL_FILE], stat[COL_FILE], "{name} file");
        assert_eq!(exists[COL_BYTES], stat[COL_BYTES], "{name} bytes");
        assert_eq!(
            exists[COL_N_OBJECTS], stat[COL_N_OBJECTS],
            "{name} n_objects"
        );
    }
}
