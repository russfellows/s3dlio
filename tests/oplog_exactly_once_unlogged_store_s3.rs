// tests/oplog_exactly_once_unlogged_store_s3.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log exactly-once contract (s3dlio #169, ruling D4/O1): a store built
// WITHOUT a logger is silent for every backend, even while an op-log is active.
// Before the fix an unlogged S3 store still wrote the S3 backend's own legacy
// row for every GET and PUT (logged "by accident", S3 only).
//
// Own process; mock S3 server.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{ensure_mock_s3_server, key_path, unique_test_key};
use common::oplog_rows::read_rows;
use s3dlio::{finalize_op_logger, init_op_logger, store_for_uri, store_for_uri_with_logger};

const BUCKET: &str = "mock-bucket";

#[tokio::test]
async fn unlogged_s3_stores_write_no_rows_while_the_op_log_is_active() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (_port, server) = ensure_mock_s3_server();
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let key = unique_test_key("silent");
    let uri = format!("s3://{BUCKET}/{key}");
    for store in [
        store_for_uri(&uri).unwrap(),
        store_for_uri_with_logger(&uri, None).unwrap(),
    ] {
        store.put(&uri, Bytes::from_static(b"data")).await.unwrap();
        store.get(&uri).await.unwrap();
    }

    finalize_op_logger();

    // The calls really reached the server (ground truth) ...
    let counts = server.counts_for(&key_path(BUCKET, &key));
    assert_eq!(counts.put_calls, 2, "{counts:?}");
    assert!(counts.get_calls >= 2, "{counts:?}");
    // ... and none of them is in the op-log.
    let rows = read_rows(&oplog);
    assert!(
        rows.is_empty(),
        "unlogged stores must not record: {rows:#?}"
    );
}
