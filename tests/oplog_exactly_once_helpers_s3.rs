// tests/oplog_exactly_once_helpers_s3.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log exactly-once contract (s3dlio #169, design doc section 8 item 4): the
// public S3 helper functions that are user entry points (they do not go through
// a logged ObjectStore) record each call ONCE, in the same row shape the
// LoggedObjectStore writes: endpoint `s3://<bucket>`, file `<bucket>/<key>`,
// client_id from set_client_id() (never the access key id).
//
// Before the fix these helpers wrote the legacy S3 backend row: endpoint =
// AWS_ENDPOINT_URL, file = the bare key, client_id = AWS_ACCESS_KEY_ID, and the
// batch delete was a `DELETE` row per 1000 keys.
//
// Own process (op-log + S3 client are process-global); mock S3 server.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{ensure_mock_s3_server, unique_test_key};
use common::oplog_rows::{
    ops, read_rows, COL_BYTES, COL_CLIENT_ID, COL_ENDPOINT, COL_FILE, COL_N_OBJECTS,
};
use s3dlio::s3_client::run_on_global_rt;
use s3dlio::s3_utils::{
    delete_objects, get_object, get_object_uri, get_object_uri_async,
    get_object_uri_optimized_async, get_objects_parallel, put_object_async, put_object_uri_async,
};
use s3dlio::{finalize_op_logger, init_op_logger, set_client_id};

const BUCKET: &str = "mock-bucket";
const CLIENT_ID: &str = "client-helpers";

#[test]
fn public_s3_helpers_record_each_call_once_in_the_logged_store_shape() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    ensure_mock_s3_server();
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    set_client_id(CLIENT_ID).unwrap();

    let k: Vec<String> = (0..9).map(|i| unique_test_key(&format!("h{i}"))).collect();
    let uri = |i: usize| format!("s3://{BUCKET}/{}", k[i]);

    // Nine calls, one row each, in this order. The two parallel GETs are two
    // objects fetched by ONE get_objects_parallel call: one row per object.
    let (k0, k1, k2, k3, k4) = (k[0].clone(), k[1].clone(), uri(2), uri(3), uri(4));
    run_on_global_rt(async move {
        put_object_async(BUCKET, &k0, Bytes::from_static(b"data")).await?;
        get_object(BUCKET, &k1).await?;
        get_object_uri_async(&k2).await?;
        put_object_uri_async(&k3, Bytes::from_static(b"data")).await?;
        get_object_uri_optimized_async(&k4).await?;
        Ok(())
    })
    .unwrap();
    get_object_uri(&uri(5)).unwrap();
    get_objects_parallel(&[uri(6), uri(7)], 2).unwrap();
    delete_objects(BUCKET, &[k[8].clone(), k[0].clone()]).unwrap();

    finalize_op_logger();

    let rows = read_rows(&oplog);
    assert_eq!(
        ops(&rows),
        [
            "PUT",
            "GET",
            "GET",
            "PUT",
            "GET",
            "GET",
            "GET",
            "GET",
            "DELETE_BATCH"
        ],
        "exactly one row per call (two for the two-object parallel GET); rows: {rows:#?}"
    );

    for (i, r) in rows.iter().enumerate() {
        assert_eq!(
            r[COL_ENDPOINT],
            format!("s3://{BUCKET}"),
            "row {i} endpoint: {r:?}"
        );
        assert_eq!(r[COL_CLIENT_ID], CLIENT_ID, "row {i} client_id: {r:?}");
        assert!(
            r[COL_FILE].starts_with(&format!("{BUCKET}/")),
            "row {i} file keeps the bucket: {r:?}"
        );
    }
    // Data bytes are the payload size (PUT) and the mock's 4-byte body (GET).
    assert_eq!(rows[0][COL_BYTES], "4");
    assert_eq!(rows[1][COL_BYTES], "4");
    // The batch delete is one row for the whole call: n_objects = 2, file = first URI.
    assert_eq!(rows[8][COL_N_OBJECTS], "2");
    assert_eq!(rows[8][COL_FILE], format!("{BUCKET}/{}", k[8]));
}
