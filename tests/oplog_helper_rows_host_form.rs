// tests/oplog_helper_rows_host_form.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log golden test (s3dlio #169, W2 round 2): the row of a host-form
// `get_many` (`get_objects_parallel`) names the server that was ACTUALLY contacted.
//
// `s3://<host>:<port>/<bucket>/<key>` names a server, but the S3 helpers (the
// engine of Python `get_many`) ignore it and use the process-wide client, i.e. the
// `AWS_ENDPOINT_URL` server (that misrouting is #171/#197, not fixed here).
// Before W2 the row's endpoint column was `AWS_ENDPOINT_URL`, which is where the
// request really went. Round 1 wrote `s3://<bucket>`, naming neither server: a
// wrong claim about routing. This test pins the pre-W2 column: the endpoint is the
// server the mock saw the request on, and NOT the decoy server the URI named.
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
fn host_form_get_many_row_names_the_server_actually_contacted() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (port, server) = ensure_mock_s3_server();
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let key = unique_test_key("hostform");
    let got = get_objects_parallel(&[format!("s3://{DECOY}/{BUCKET}/{key}")], 1).unwrap();
    assert_eq!(got.len(), 1);
    finalize_op_logger();

    let reqs = server.requests_containing(&key);
    assert!(
        !reqs.is_empty(),
        "the request reached the AWS_ENDPOINT_URL server"
    );
    for r in &reqs {
        assert_eq!(
            r.host.as_deref(),
            Some(format!("127.0.0.1:{port}").as_str())
        );
    }
    let rows = read_rows(&oplog);
    assert_eq!(rows.len(), 1, "{rows:#?}");
    assert_eq!(
        rows[0][COL_ENDPOINT],
        format!("http://127.0.0.1:{port}"),
        "the row names the server the request went to"
    );
    assert!(
        !rows[0][COL_ENDPOINT].contains(DECOY),
        "and not the server the URI named: {:?}",
        rows[0]
    );
    assert_eq!(rows[0][COL_FILE], key);
}
