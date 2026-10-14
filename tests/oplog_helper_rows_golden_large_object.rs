// tests/oplog_helper_rows_golden_large_object.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log golden test (s3dlio #169, W2 round 2), large-object path.
//
// An object at or above the range threshold is fetched by `get_object_uri_optimized_async`
// (and so by `get_objects_parallel`, the engine of Python `get_many`) with
// concurrent ranged GETs through a different code path that never reached the S3
// backend's row writer. Before W2 it wrote NO row. Round 1 made it write one, in a
// shape the helpers do not otherwise use; that is not approved. The pre-W2
// behaviour is restored: no row, only requests. (A row in the helper's own shape
// would need a first_byte and a client id the ranged path does not have; it is
// listed in the audit as a proposal.)
//
// This test passes unchanged on the parent commit (9473d2c), where it was captured.
//
// Own process: the range threshold is read once per process.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, SIZED_OBJECT_MARKER};
use common::oplog_rows::read_rows;
use s3dlio::s3_client::run_on_global_rt;
use s3dlio::s3_utils::{get_object_uri_optimized_async, get_objects_parallel};
use s3dlio::{finalize_op_logger, init_op_logger};

const BUCKET: &str = "mock-bucket";

#[test]
fn large_object_reads_write_no_helper_row() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
        std::env::set_var("S3DLIO_RANGE_THRESHOLD_MB", "0");
        std::env::set_var("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "1");
        std::env::set_var("S3DLIO_SKIP_HEAD", "0");
    }
    let (_port, server) = ensure_mock_s3_server();
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let key = |n: u32| format!("golden-large-{n}/{SIZED_OBJECT_MARKER}/data");
    let uri = |n: u32| format!("s3://{BUCKET}/{}", key(n));
    let u0 = uri(0);
    run_on_global_rt(async move {
        let data = get_object_uri_optimized_async(&u0).await?;
        assert_eq!(data.len(), 16);
        Ok(())
    })
    .unwrap();
    let got = get_objects_parallel(&[uri(1), uri(2)], 2).unwrap();
    assert!(got.iter().all(|(_, d)| d.len() == 16));

    finalize_op_logger();

    // The large-object path really ran: the server saw ranged GETs for the keys.
    for n in 0..3 {
        let ranged = server
            .requests_containing(&key(n))
            .iter()
            .filter(|r| r.method == "GET" && r.range.is_some())
            .count();
        assert!(ranged >= 1, "object {n} was fetched by a ranged GET");
    }
    let rows = read_rows(&oplog);
    assert!(
        rows.is_empty(),
        "no row for a ranged large-object read: {rows:#?}"
    );
}
