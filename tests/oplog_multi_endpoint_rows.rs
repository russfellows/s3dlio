// tests/oplog_multi_endpoint_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log rows of `MultiEndpointStore` (s3dlio #175, #169 W2 round 2).
//
// A MultiEndpointStore is never built with a logger. Before W2, `get` and `put`
// on a BUCKET-form S3 endpoint (`s3://bucket/...`, the process-wide client) still
// produced one row each, written by the S3 backend's own writer (`S3Ops`):
// endpoint = AWS_ENDPOINT_URL, file = the bare key, client_id = the access key id.
// Round 1 removed that writer and with it these rows. This test pins the rows
// back, field for field, and pins that every OTHER operation (list, delete,
// get_range, stat, and everything on a host-form endpoint) is still silent: a row
// for those is a proposal for the architect, not part of this change (see
// docs/design/oplog-record-contract/W2-ownership-audit.md, section 5).
//
// Rows per operation (R = rows, Q = requests):
//
//   endpoint form   operation   R                                  Q
//   --------------  ----------  ---------------------------------  ------------
//   bucket-form     put         1 PUT  (legacy S3Ops shape)         1 PUT
//   bucket-form     get         1 GET  (legacy S3Ops shape)         1 GET
//   bucket-form     rename      2 (GET, PUT; the DELETE is silent)  GET, PUT, DELETE
//   bucket-form     list        0                                  1 list
//   bucket-form     get_range   0                                  1 ranged GET
//   bucket-form     stat        0                                  1 HEAD
//   bucket-form     delete      0                                  1 DELETE
//   host-form       put, get    0                                  1 PUT, 1 GET
//
// Own process (the op-log and the S3 client are process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{ensure_mock_s3_server, key_path, unique_test_key};
use common::oplog_rows::{ops, read_rows, COL_BYTES, COL_ENDPOINT, COL_FILE, COL_N_OBJECTS};
use s3dlio::multi_endpoint::{LoadBalanceStrategy, MultiEndpointStore};
use s3dlio::s3_client::run_on_global_rt;
use s3dlio::{finalize_op_logger, init_op_logger, set_client_id, ObjectStore};

const BUCKET: &str = "mock-bucket";

#[test]
fn multi_endpoint_get_and_put_keep_their_pre_w2_rows_and_nothing_else_is_recorded() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (port, server) = ensure_mock_s3_server();
    let mock = format!("127.0.0.1:{port}");
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    set_client_id("client-multi-endpoint").unwrap();

    let bucket_form = MultiEndpointStore::new(
        vec![format!("s3://{BUCKET}/mep/")],
        LoadBalanceStrategy::RoundRobin,
        None,
    )
    .unwrap();
    let host_form = MultiEndpointStore::new(
        vec![format!("s3://{mock}/{BUCKET}/mep-host/")],
        LoadBalanceStrategy::RoundRobin,
        None,
    )
    .unwrap();

    let (k_put, k_ren, k_ren2) = (
        unique_test_key("mep-put"),
        unique_test_key("mep-ren-src"),
        unique_test_key("mep-ren-dst"),
    );
    let h_key = unique_test_key("mep-host-key");
    let bf = |k: &str| format!("s3://{BUCKET}/mep/{k}");
    let hf = |k: &str| format!("s3://{mock}/{BUCKET}/mep-host/{k}");
    let (u_put, u_ren, u_ren2, u_host) = (bf(&k_put), bf(&k_ren), bf(&k_ren2), hf(&h_key));
    let (bucket_prefix, host_prefix) = (
        format!("s3://{BUCKET}/mep/"),
        format!("s3://{mock}/{BUCKET}/mep-host/"),
    );
    let (bucket_form, host_form) = (
        std::sync::Arc::new(bucket_form),
        std::sync::Arc::new(host_form),
    );
    run_on_global_rt(async move {
        // bucket-form, recorded
        bucket_form.put(&u_put, Bytes::from_static(b"data")).await?;
        bucket_form.get(&u_put).await?;
        // bucket-form, silent
        bucket_form.list(&bucket_prefix, false).await?;
        bucket_form.get_range(&u_put, 0, Some(2)).await?;
        bucket_form.stat(&u_put).await?;
        bucket_form.delete(&u_put).await?;
        // rename = the trait default over the store: GET + PUT recorded (by the
        // endpoint store's get/put), DELETE not.
        bucket_form.rename(&u_ren, &u_ren2).await?;
        // host-form: nothing is recorded.
        host_form.put(&u_host, Bytes::from_static(b"data")).await?;
        host_form.get(&u_host).await?;
        host_form.list(&host_prefix, false).await?;
        host_form.delete(&u_host).await?;
        Ok(())
    })
    .unwrap();

    finalize_op_logger();
    let rows = read_rows(&oplog);

    assert_eq!(
        ops(&rows),
        ["PUT", "GET", "GET", "PUT"],
        "bucket-form put, get, then the rename's GET and PUT; nothing for list, \
         get_range, stat, delete or the host-form endpoint: {rows:#?}"
    );
    // The pre-W2 S3Ops shape: endpoint = AWS_ENDPOINT_URL, file = the bare key,
    // client_id = the access key id (NOT set_client_id), bytes 4 on success.
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r[COL_ENDPOINT], format!("http://{mock}"), "row {i}: {r:?}");
        assert_eq!(r[3], "mock-access-key", "row {i} client_id: {r:?}");
        assert_eq!(r[COL_N_OBJECTS], "1", "row {i}: {r:?}");
        assert_eq!(r[COL_BYTES], "4", "row {i}: {r:?}");
        assert!(r[8].is_empty(), "row {i} error: {r:?}");
    }
    assert_eq!(rows[0][COL_FILE], format!("mep/{k_put}"));
    assert_eq!(rows[1][COL_FILE], format!("mep/{k_put}"));
    assert_eq!(rows[2][COL_FILE], format!("mep/{k_ren}"));
    assert_eq!(rows[3][COL_FILE], format!("mep/{k_ren2}"));
    // first_byte: a GET has one (response header time), a PUT has none.
    assert!(!rows[1][10].is_empty() && !rows[2][10].is_empty());
    assert!(rows[0][10].is_empty() && rows[3][10].is_empty());

    // What reached the server: every operation did, rows or not, all at the one
    // mock server (host:port), never anywhere else.
    let methods = |k: &str| {
        let mut v: Vec<String> = server
            .requests_containing(k)
            .iter()
            .map(|r| {
                assert_eq!(r.host.as_deref(), Some(mock.as_str()), "{r:?}");
                format!(
                    "{} {}",
                    r.method,
                    if r.range.is_some() { "range" } else { "" }
                )
                .trim()
                .to_string()
            })
            .collect();
        v.sort();
        v
    };
    // put: 1 PUT. get: the optimized read's size probe (HEAD) + 1 GET. get_range: 1
    // ranged GET. stat: 1 HEAD. (delete is a DeleteObjects POST naming the key in its
    // body: counted below.)
    assert_eq!(
        methods(&k_put),
        ["GET", "GET range", "HEAD", "HEAD", "PUT"],
        "bucket-form put/get/get_range/stat each sent their requests"
    );
    assert_eq!(
        server
            .counts_for(&key_path(BUCKET, &format!("mep/{k_put}")))
            .batch_delete_calls,
        1,
        "bucket-form delete reached the server"
    );
    // rename: GET (with its size probe) of the source, PUT of the destination, and
    // the source's DELETE (a DeleteObjects POST).
    assert_eq!(methods(&k_ren), ["GET", "HEAD"]);
    assert_eq!(methods(&k_ren2), ["PUT"]);
    assert_eq!(
        server
            .counts_for(&key_path(BUCKET, &format!("mep/{k_ren}")))
            .batch_delete_calls,
        1
    );
    // host-form endpoint: PUT, GET, delete all reached the server, none is a row.
    assert_eq!(methods(&h_key), ["DELETE", "GET", "PUT"]);
}
