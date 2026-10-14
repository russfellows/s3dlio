// tests/oplog_helper_rows_endpoint_and_routing.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Two integration requirements for the public S3 helper functions (#169, #183, #197):
//
// 1. The row a helper writes records the EFFECTIVE endpoint. With
//    AWS_ENDPOINT_URL = A and AWS_ENDPOINT_URL_S3 = B, a helper get/put reaches B
//    only, and its row's `endpoint` column names B (the representation is
//    unchanged: server URL in `endpoint`, bare key in `file`).
// 2. A helper called with a URI that names another server is rejected by the #197
//    check before any request is sent, and it writes no row.
//
// Own process: the op-log and the S3 client are process-global.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::MockS3;

#[allow(dead_code)] // shared test helpers; this binary uses a subset
mod common;
use common::oplog_rows::{read_rows, COL_ENDPOINT, COL_FILE, COL_OP};

use bytes::Bytes;
use s3dlio::s3_utils::{
    get_object_uri_async, get_object_uri_optimized_async, get_objects_parallel,
    put_object_uri_async,
};
use s3dlio::{finalize_op_logger, init_op_logger};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn helper_rows_name_the_effective_endpoint_and_foreign_servers_are_rejected() {
    let a = MockS3::start(&[]);
    let b = MockS3::start(&[]);
    #[allow(deprecated)]
    {
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ENDPOINT_URL", a.url());
        std::env::set_var("AWS_ENDPOINT_URL_S3", b.url());
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().expect("utf-8 path")).expect("init op-log");

    // (1) bucket-form helper calls: B only, rows name B.
    let uri = "s3://bkt183h/key";
    put_object_uri_async(uri, Bytes::from_static(b"x"))
        .await
        .expect("helper put");
    get_object_uri_async(uri).await.expect("helper get");
    assert_eq!(
        b.requests(),
        vec!["PUT /bkt183h/key".to_string(), "GET /bkt183h/key".to_string()]
    );
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");

    // (2) every public helper entry rejects a URI naming another server (A is not
    // the global server): no request reaches either mock, and no row is written.
    let foreign = format!("s3://127.0.0.1:{}/bkt197h/key", a.port);
    let err = put_object_uri_async(&foreign, Bytes::from_static(b"x"))
        .await
        .expect_err("put naming another server");
    assert!(
        format!("{err:#}").contains(&format!("127.0.0.1:{}", a.port)),
        "got: {err:#}"
    );
    assert!(get_object_uri_async(&foreign).await.is_err());
    assert!(get_object_uri_optimized_async(&foreign).await.is_err());
    let batch = vec![foreign.clone()];
    let joined = std::thread::spawn(move || get_objects_parallel(&batch, 2))
        .join()
        .expect("get_objects_parallel thread");
    assert!(joined.is_err(), "get_objects_parallel naming another server");
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
    assert_eq!(b.requests().len(), 2, "B got a request for the rejected URIs");

    finalize_op_logger();
    let rows = read_rows(&oplog);
    let summary: Vec<(String, String, String)> = rows
        .iter()
        .map(|r| (r[COL_OP].clone(), r[COL_ENDPOINT].clone(), r[COL_FILE].clone()))
        .collect();
    assert_eq!(
        summary,
        vec![
            ("PUT".to_string(), b.url(), "key".to_string()),
            ("GET".to_string(), b.url(), "key".to_string()),
        ],
        "only the two accepted calls write a row, and each names B"
    );
}
