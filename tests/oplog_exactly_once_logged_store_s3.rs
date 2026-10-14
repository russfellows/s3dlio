// tests/oplog_exactly_once_logged_store_s3.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log exactly-once contract (s3dlio #169, design doc section 8): one call
// through a LOGGED store is ONE row, on the S3 backend too. Before the fix every
// S3 GET and PUT produced two rows: the LoggedObjectStore row and a second row
// written by the S3 backend's own legacy logger (endpoint = AWS_ENDPOINT_URL,
// file = bare key, client_id = the access key id).
//
// Own process (the op-log and the S3 client are process-global). The traffic goes
// to the in-process mock S3 server; its per-key request counters are the ground
// truth for "what actually reached the server".

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, HEAD_NOT_FOUND_MARKER,
};
use common::oplog_rows::{
    ops, read_rows, COL_CLIENT_ID, COL_ENDPOINT, COL_ERROR, COL_FILE, COL_N_OBJECTS,
};
use s3dlio::{
    finalize_op_logger, global_logger, init_op_logger, set_client_id, store_for_uri_with_logger,
};

const BUCKET: &str = "mock-bucket";
const CLIENT_ID: &str = "client-exactly-once";

#[tokio::test]
async fn logged_store_records_each_s3_call_exactly_once() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (_port, server) = ensure_mock_s3_server();
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    set_client_id(CLIENT_ID).unwrap();

    let store =
        store_for_uri_with_logger(&format!("s3://{BUCKET}/"), global_logger()).expect("S3 store");

    let key = unique_test_key("once");
    let missing = unique_test_key(HEAD_NOT_FOUND_MARKER);
    let uri = format!("s3://{BUCKET}/{key}");
    let missing_uri = format!("s3://{BUCKET}/{missing}");

    // Eight calls, one row each.
    store.put(&uri, Bytes::from_static(b"data")).await.unwrap();
    store.get(&uri).await.unwrap();
    store.get_range(&uri, 0, Some(2)).await.unwrap();
    store.stat(&uri).await.unwrap();
    store
        .list(&format!("s3://{BUCKET}/p/"), false)
        .await
        .unwrap();
    store.delete(&uri).await.unwrap();
    store
        .delete_batch(std::slice::from_ref(&uri))
        .await
        .unwrap();
    assert!(store.stat(&missing_uri).await.is_err());
    const CALLS: usize = 8;

    finalize_op_logger();

    // Ground truth 1: what the server received for the PUT/GET key.
    let counts = server.counts_for(&key_path(BUCKET, &key));
    assert_eq!(
        counts.put_calls, 1,
        "server saw exactly one PutObject: {counts:?}"
    );

    // Ground truth 2: exactly one row per call, in order.
    let rows = read_rows(&oplog);
    assert_eq!(
        ops(&rows),
        [
            "PUT",
            "GET",
            "GET_RANGE",
            "HEAD",
            "LIST",
            "DELETE",
            "DELETE_BATCH",
            "HEAD"
        ],
        "one row per call, no extra row from the S3 backend; rows: {rows:#?}"
    );
    assert_eq!(rows.len(), CALLS);

    // Every row is the LoggedObjectStore shape: scheme://bucket, bucket/key,
    // the client id set with set_client_id (not the access key id).
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(
            r[COL_ENDPOINT],
            format!("s3://{BUCKET}"),
            "row {i} endpoint"
        );
        assert!(
            r[COL_FILE].starts_with(&format!("{BUCKET}/")),
            "row {i} file keeps the bucket: {r:?}"
        );
        assert_eq!(r[COL_CLIENT_ID], CLIENT_ID, "row {i} client_id");
    }
    assert_eq!(rows[0][COL_FILE], format!("{BUCKET}/{key}"));
    assert_eq!(rows[6][COL_N_OBJECTS], "1");
    assert_eq!(rows[7][COL_FILE], format!("{BUCKET}/{missing}"));
    assert!(
        !rows[7][COL_ERROR].is_empty(),
        "the failed HEAD carries an error"
    );
}
