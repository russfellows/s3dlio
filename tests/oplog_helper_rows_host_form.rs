// tests/oplog_helper_rows_host_form.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log test (s3dlio #169 W2 round 2 + #197): the row of a host-form `get_many`
// (`get_objects_parallel`) never names a server that was not contacted.
//
// This file pins the POST-#197 contract. `s3://<host>:<port>/<bucket>/<key>` names a
// server. Before #197 the S3 helpers (the engine of Python `get_many`) ignored the
// named server and used the process-wide client, i.e. the `AWS_ENDPOINT_URL`
// server; the W2 round-2 version of this test pinned that the row then named the
// server the request really went to (reviewer item B5) and not the decoy the URI
// named. #197 removed the misrouting: a URI naming another server is rejected
// before any request, so that expectation (an `Ok` from a decoy URI) was superseded.
// B5's intent is kept in two parts:
//   (a) a decoy host-form URI is rejected, the mock receives zero requests and the
//       op-log has no row for it;
//   (b) a host-form URI naming exactly the `AWS_ENDPOINT_URL` server succeeds with
//       the right bytes, the mock sees the GET, and the row's endpoint names that
//       server (server URL in `endpoint`, bare key in `file`).
//
// Own process (the op-log and the S3 client are process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, unique_test_key};
use common::oplog_rows::{read_rows, COL_ENDPOINT, COL_FILE};
use s3dlio::s3_utils::get_objects_parallel;
use s3dlio::{finalize_op_logger, init_op_logger};

const BUCKET: &str = "mock-bucket";
/// A port nothing listens on: the server the URI names.
const DECOY: &str = "127.0.0.1:9";

#[test]
fn host_form_get_many_rows_never_name_a_server_not_contacted() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (port, server) = ensure_mock_s3_server();
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    // (a) The URI names a server that is not the AWS_ENDPOINT_URL one: rejected,
    // nothing is sent, nothing is recorded.
    let decoy_key = unique_test_key("hostform-decoy");
    let err = get_objects_parallel(&[format!("s3://{DECOY}/{BUCKET}/{decoy_key}")], 1)
        .expect_err("a URI naming another server must be rejected (#197)");
    assert!(
        format!("{err:#}").contains(DECOY),
        "the error names the URI's server: {err:#}"
    );
    assert!(
        server.requests_containing(&decoy_key).is_empty(),
        "the mock must receive zero requests for the rejected URI"
    );

    // (b) The URI names exactly the AWS_ENDPOINT_URL server: served, and recorded
    // with that server as the endpoint.
    let key = unique_test_key("hostform");
    let got = get_objects_parallel(&[format!("s3://127.0.0.1:{port}/{BUCKET}/{key}")], 1)
        .expect("a URI naming the global server is served");
    assert_eq!(got.len(), 1);
    assert_eq!(&got[0].1[..], b"mock");
    finalize_op_logger();

    let reqs = server.requests_containing(&key);
    assert!(!reqs.is_empty(), "the GET reached the mock");
    for r in &reqs {
        assert_eq!(
            r.host.as_deref(),
            Some(format!("127.0.0.1:{port}").as_str())
        );
    }
    let rows = read_rows(&oplog);
    assert_eq!(
        rows.len(),
        1,
        "one row for the served call, none for the rejected one: {rows:#?}"
    );
    assert_eq!(
        rows[0][COL_ENDPOINT],
        format!("http://127.0.0.1:{port}"),
        "the row names the server the request went to"
    );
    assert!(
        !rows[0][COL_ENDPOINT].contains(DECOY),
        "and not the decoy server: {:?}",
        rows[0]
    );
    assert_eq!(rows[0][COL_FILE], key);
}
