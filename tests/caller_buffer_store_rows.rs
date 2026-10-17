// tests/caller_buffer_store_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log rows of the REAL stores' `get_into` and `head`, built the way the Python store cache
// builds them: `store_for_uri_with_logger(uri, global_logger())` (v0.10.0 plan §4.4, §4.5 A6;
// #264). `LoggedObjectStore` writes the ONE logical row per call. The stores underneath must
// add none: not the S3 helper row of the process-wide client (`S3ObjectStore::rows()` is `Skip`
// for a store the cache builds), not one per range, retry or fan-out HEAD, and not one from a
// per-endpoint client or from Azure.
//
// One process, one op-log, one ordered test. The totals are exact: every key is used by one
// call, so a stray helper row shows up as an extra row.

#![cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]

#[path = "common/oplog_rows.rs"]
#[allow(dead_code)] // the row reader serves several test binaries; each uses a subset
mod oplog_rows;
#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

#[cfg(feature = "backend-azure")]
#[path = "common/azure_fake.rs"]
mod azure_fake;
#[cfg(feature = "backend-azure")]
#[path = "caller_buffer_store_support/mod.rs"]
mod support;

use oplog_rows::{read_rows, Row, COL_BYTES, COL_ERROR, COL_FILE, COL_OP, HEADER_LINE};
use s3_get_into_support::mock::{start_isolated_mock_s3_server, MockObject};
use s3_get_into_support::{init, key, pattern, BUCKET};
use s3dlio::object_store::store_for_uri_with_logger;
use s3dlio::receive_progress::ReceiveProgress;
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use std::sync::Once;

const MIB: usize = 1024 * 1024;

/// Every read of 1 MiB or more is a ranged fan-out: a HEAD and several ranges per call.
const KNOBS: &[(&str, &str)] = &[
    ("S3DLIO_RANGE_THRESHOLD_MB", "1"),
    ("S3DLIO_CHUNK_SIZE", "262144"),
    ("S3DLIO_RANGE_CONCURRENCY", "2"),
];

static ENV: Once = Once::new();

/// The process environment, set ONCE before the test touches the network: the op-log knobs the
/// default must not inherit, the S3 mock and its knobs, and the Azure fake with its endpoint.
fn prepare_env() {
    ENV.call_once(|| {
        for name in ["S3DLIO_OPLOG_BUF", "S3DLIO_OPLOG_LOSSLESS"] {
            std::env::remove_var(name);
        }
        init(KNOBS);
        #[cfg(feature = "backend-azure")]
        support::azure();
    });
}

/// The index of the `first_byte` column, taken from the header the shared reader checks.
fn col_first_byte() -> usize {
    HEADER_LINE
        .split('\t')
        .position(|name| name == "first_byte")
        .expect("the op-log header has a first_byte column")
}

/// The rows whose `file` column names `key`.
fn rows_of<'r>(rows: &'r [Row], key: &str) -> Vec<&'r Row> {
    rows.iter().filter(|r| r[COL_FILE].contains(key)).collect()
}

/// The one row of `key`, asserting it is the only one, and its op.
fn only<'r>(rows: &'r [Row], key: &str, op: &str) -> &'r Row {
    let found = rows_of(rows, key);
    assert_eq!(found.len(), 1, "{key}: exactly one row, got {found:#?}");
    assert_eq!(found[0][COL_OP], op, "{key}");
    found[0]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_logged_real_store_writes_exactly_one_row_per_call() {
    prepare_env();
    let (_, global) = init(KNOBS);
    let (endpoint_port, endpoint) = start_isolated_mock_s3_server();

    let dir = tempfile::tempdir().expect("tempdir");
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().expect("utf-8 path")).expect("init the op-log");
    assert!(global_logger().is_some());

    // Each entry: (key, expected op, expected bytes, error expected).
    let mut expected: Vec<(String, &str, u64, bool)> = Vec::new();

    // ---- S3, process-wide client, bucket-form URI ----
    let uri = |k: &str| format!("s3://{BUCKET}/{k}");
    let store = store_for_uri_with_logger(&uri("x"), global_logger()).expect("s3 store");

    let small = key("rows-small");
    global.put_object(BUCKET, &small, MockObject::new(pattern(700, 1)));
    store
        .get_into(&uri(&small), &mut [0u8; 700], &ReceiveProgress::new())
        .await
        .expect("get");
    expected.push((small.clone(), "GET", 700, false));

    let big = key("rows-fanout");
    global.put_object(BUCKET, &big, MockObject::new(pattern(MIB + 17, 2)));
    let mut dst = vec![0u8; MIB + 17];
    store
        .get_into(&uri(&big), &mut dst, &ReceiveProgress::new())
        .await
        .expect("get");
    assert!(
        s3_get_into_support::gets(&global, &big).len() >= 5
            && s3_get_into_support::heads(&global, &big) == 1,
        "the fan-out (a HEAD and ranges) really ran"
    );
    expected.push((big.clone(), "GET", (MIB + 17) as u64, false));

    let absent = key("rows-absent");
    global.put_object(BUCKET, &absent, MockObject::missing());
    store
        .get_into(&uri(&absent), &mut [0u8; 8], &ReceiveProgress::new())
        .await
        .expect_err("absent");
    expected.push((absent.clone(), "GET", 0, true));

    let head_ok = key("rows-head");
    global.put_object(BUCKET, &head_ok, MockObject::new(pattern(321, 3)));
    store
        .head(&uri(&head_ok))
        .await
        .expect("head")
        .expect("present");
    expected.push((head_ok.clone(), "HEAD", 321, false));

    let head_absent = key("rows-head-absent");
    global.put_object(BUCKET, &head_absent, MockObject::missing());
    assert!(store
        .head(&uri(&head_absent))
        .await
        .expect("head")
        .is_none());
    expected.push((head_absent.clone(), "HEAD", 0, false));

    // ---- S3, per-endpoint client, host-form URI ----
    let host_uri = |k: &str| format!("s3://127.0.0.1:{endpoint_port}/{BUCKET}/{k}");
    let host_store = store_for_uri_with_logger(&host_uri("x"), global_logger()).expect("store");

    let host_get = key("rows-host-get");
    endpoint.put_object(BUCKET, &host_get, MockObject::new(pattern(MIB + 3, 4)));
    let mut dst = vec![0u8; MIB + 3];
    host_store
        .get_into(&host_uri(&host_get), &mut dst, &ReceiveProgress::new())
        .await
        .expect("get");
    expected.push((host_get.clone(), "GET", (MIB + 3) as u64, false));

    let host_head = key("rows-host-head");
    endpoint.put_object(BUCKET, &host_head, MockObject::new(pattern(55, 5)));
    host_store
        .head(&host_uri(&host_head))
        .await
        .expect("head")
        .expect("present");
    expected.push((host_head.clone(), "HEAD", 55, false));

    // ---- Azure ----
    #[cfg(feature = "backend-azure")]
    {
        use azure_fake::{pattern as az_pattern, set_blob, uri as az_uri};
        let azure_state = support::azure();
        let store = store_for_uri_with_logger(&az_uri("x"), global_logger()).expect("az store");

        set_blob(azure_state, "rows-az-get", az_pattern(900, 6));
        store
            .get_into(
                &az_uri("rows-az-get"),
                &mut [0u8; 900],
                &ReceiveProgress::new(),
            )
            .await
            .expect("get");
        expected.push(("rows-az-get".into(), "GET", 900, false));

        store
            .get_into(
                &az_uri("rows-az-absent"),
                &mut [0u8; 9],
                &ReceiveProgress::new(),
            )
            .await
            .expect_err("absent");
        expected.push(("rows-az-absent".into(), "GET", 0, true));

        set_blob(azure_state, "rows-az-head", az_pattern(66, 7));
        store
            .head(&az_uri("rows-az-head"))
            .await
            .expect("head")
            .expect("present");
        expected.push(("rows-az-head".into(), "HEAD", 66, false));

        assert!(store
            .head(&az_uri("rows-az-nohead"))
            .await
            .expect("head")
            .is_none());
        expected.push(("rows-az-nohead".into(), "HEAD", 0, false));
    }

    finalize_op_logger();
    let rows = read_rows(&oplog);
    let first_byte = col_first_byte();

    for (key, op, bytes, errored) in &expected {
        let row = only(&rows, key, op);
        assert_eq!(row[COL_BYTES], bytes.to_string(), "{key}");
        assert_eq!(!row[COL_ERROR].is_empty(), *errored, "{key}: {row:?}");
        if *op == "HEAD" {
            assert_eq!(row[first_byte], "", "{key}: a HEAD row has no first byte");
        } else if !*errored {
            assert!(!row[first_byte].is_empty(), "{key}: a GET row has one");
        }
    }
    // Nothing else: no helper row, no row per range, retry or fan-out HEAD.
    assert_eq!(
        rows.len(),
        expected.len(),
        "unexpected extra rows: {rows:#?}"
    );
}
