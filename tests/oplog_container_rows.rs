// tests/oplog_container_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #182: CREATE_CONTAINER / DELETE_CONTAINER rows were written with the
// bare container name (`newbkt`, `/dir`), so the endpoint column held "unknown"
// and the file column had no scheme-relative form, unlike every other row of the
// same store (`s3://bucket` + `bucket/key`, `file://` + `/path`).
//
// The rows now use the endpoint and file the store's other rows use. Checked on
// a file store, an S3 bucket-form store and an S3 host-form store (own mock
// server address), each compared with a PUT row of the same store.
//
// Own process (op-log + S3 client are process-global); mock S3 server.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{ensure_mock_s3_server, unique_test_key};
use common::oplog_rows::{ops, read_rows, COL_ENDPOINT, COL_FILE};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn container_rows_use_the_stores_endpoint_and_file_derivation() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (port, _server) = ensure_mock_s3_server();
    let work = tempfile::tempdir().unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    let logger = global_logger();

    // 1. file:// store: container = a directory path.
    let dir = format!("{}/sub", work.path().display());
    let file_store =
        store_for_uri_with_logger(&format!("file://{}", work.path().display()), logger.clone())
            .unwrap();
    file_store.create_container(&dir).await.unwrap();
    file_store
        .put(
            &format!("file://{}/o.bin", work.path().display()),
            Bytes::from_static(b"x"),
        )
        .await
        .unwrap();
    file_store.delete_container(&dir).await.unwrap();

    // 2. S3 bucket-form store: container = bucket name.
    let bucket = unique_test_key("newbkt");
    let s3_store = store_for_uri_with_logger("s3://mock-bucket/", logger.clone()).unwrap();
    s3_store.create_container(&bucket).await.unwrap();
    s3_store
        .put(&format!("s3://{bucket}/k"), Bytes::from_static(b"x"))
        .await
        .unwrap();
    s3_store.delete_container(&bucket).await.unwrap();

    // 3. S3 host-form store (per-endpoint client).
    let host = format!("127.0.0.1:{port}");
    let host_store =
        store_for_uri_with_logger(&format!("s3://{host}/mock-bucket/"), logger).unwrap();
    host_store.create_container("hostbkt").await.unwrap();
    host_store
        .put(&format!("s3://{host}/hostbkt/k"), Bytes::from_static(b"x"))
        .await
        .unwrap();
    host_store.delete_container("hostbkt").await.unwrap();

    finalize_op_logger();

    let rows = read_rows(&oplog);
    assert_eq!(
        ops(&rows),
        [
            "CREATE_CONTAINER",
            "PUT",
            "DELETE_CONTAINER",
            "CREATE_CONTAINER",
            "PUT",
            "DELETE_CONTAINER",
            "CREATE_CONTAINER",
            "PUT",
            "DELETE_CONTAINER",
        ],
        "rows: {rows:#?}"
    );
    let work_path = work.path().display().to_string();
    for (i, (endpoint, container, object_prefix)) in [
        ("file://".to_string(), dir.clone(), work_path),
        (format!("s3://{bucket}"), bucket.clone(), bucket.clone()),
        (
            format!("s3://{host}"),
            format!("{host}/hostbkt"),
            format!("{host}/hostbkt"),
        ),
    ]
    .iter()
    .enumerate()
    {
        let (create, put, delete) = (&rows[3 * i], &rows[3 * i + 1], &rows[3 * i + 2]);
        // The PUT row is the store's reference derivation ...
        assert_eq!(&put[COL_ENDPOINT], endpoint, "store {i} PUT endpoint");
        // ... and the container rows must agree with it.
        assert_eq!(&create[COL_ENDPOINT], endpoint, "store {i} CREATE endpoint");
        assert_eq!(&delete[COL_ENDPOINT], endpoint, "store {i} DELETE endpoint");
        assert_eq!(&create[COL_FILE], container, "store {i} CREATE file");
        assert_eq!(&delete[COL_FILE], container, "store {i} DELETE file");
        assert!(
            put[COL_FILE].starts_with(object_prefix.as_str()),
            "store {i}: the store's object paths start with {object_prefix}: {put:?}"
        );
    }
}
