// tests/s3_get_into_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Helper op-log rows of the S3 caller-buffer GET and typed HEAD (plan §4.5 A6): with a logger
// each call writes EXACTLY ONE row -- `GET` or `HEAD` -- whichever way it ends; the internal
// ranges, retries and the fan-out HEAD write none; without a logger nothing is written.
//
// One process, one op-log, one ordered test: the op-log is process-global. The range threshold
// is 0 and the chunk size 4, so every non-empty read here is a ranged fan-out (a HEAD, four
// ranges and a retry per call) that must still be one row.

#[path = "common/oplog_rows.rs"]
#[allow(dead_code)] // the row reader serves several test binaries; each uses a subset
mod oplog_rows;
#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use oplog_rows::{read_rows, Row, COL_BYTES, COL_ERROR, COL_FILE, COL_OP};
use s3_get_into_support::mock::{MockObject, ObjectFault};
use s3_get_into_support::*;
use s3dlio::receive_progress::ReceiveProgress;
use s3dlio::s3_ops::S3Ops;
use std::time::Duration;

const N: usize = 16;
const COL_FIRST_BYTE: usize = 10;

fn rows_of<'a>(rows: &'a [Row], op: &str, key: &str) -> Vec<&'a Row> {
    rows.iter()
        .filter(|r| r[COL_OP] == op && r[COL_FILE] == key)
        .collect()
}

#[tokio::test]
async fn each_call_is_one_helper_row_and_no_logger_means_no_row() {
    // The op-log knobs of a developer shell must not change the result.
    for name in ["S3DLIO_OPLOG_BUF", "S3DLIO_OPLOG_LOSSLESS"] {
        std::env::remove_var(name);
    }
    let (port, state) = init(&[
        ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
        ("S3DLIO_CHUNK_SIZE", "4"),
        ("S3DLIO_RANGE_CONCURRENCY", "2"),
    ]);
    let dir = tempfile::tempdir().expect("tempdir");
    let oplog = dir.path().join("ops.tsv.zst");
    s3dlio::init_op_logger(oplog.to_str().expect("utf-8 path")).expect("init the op-log");

    let client = client_for(port).await;
    let on = S3Ops::new(
        client.clone(),
        s3dlio::s3_logger::global_logger(),
        "client-rows",
        "endpoint-rows",
    );
    assert!(s3dlio::s3_logger::global_logger().is_some());
    let off = S3Ops::new(client, None, "", "");

    // ---- GET: a fan-out with a broken range and its retry: ONE row ----
    let ok = key("get-ok");
    state.put_object(
        BUCKET,
        &ok,
        MockObject::new(pattern(N, 1)).with_fault(
            Some(8),
            1,
            ObjectFault::TruncateBody { send: 2 },
        ),
    );
    on.get_into(BUCKET, &ok, Dst::new(N).slice(), &ReceiveProgress::new())
        .await
        .expect("get");
    assert!(gets(&state, &ok).len() > 4, "ranges and a retry were sent");

    // ---- the same without a logger: no row ----
    let quiet = key("get-quiet");
    state.put_object(BUCKET, &quiet, MockObject::new(pattern(N, 2)));
    off.get_into(BUCKET, &quiet, Dst::new(N).slice(), &ReceiveProgress::new())
        .await
        .expect("get");

    // ---- a partial failure: bytes that were touched, and the error ----
    let partial = key("get-partial");
    let mut object =
        MockObject::new(pattern(N, 3)).with_fault(None, 1, ObjectFault::TruncateBody { send: 6 });
    object.etag = None; // no ETag: the broken response is not repaired
    state.put_object(BUCKET, &partial, object);
    let progress = ReceiveProgress::new();
    on.get_into(BUCKET, &partial, Dst::new(N).slice(), &progress)
        .await
        .expect_err("partial");
    assert_eq!(progress.filled(), 6);

    // ---- a failure before any byte ----
    let absent = key("get-absent");
    state.put_object(BUCKET, &absent, MockObject::missing());
    on.get_into(
        BUCKET,
        &absent,
        Dst::new(N).slice(),
        &ReceiveProgress::new(),
    )
    .await
    .expect_err("absent");

    // ---- a zero-length read ----
    let zero = key("get-zero");
    state.put_object(BUCKET, &zero, MockObject::new(Vec::new()));
    on.get_into(BUCKET, &zero, Dst::new(0).slice(), &ReceiveProgress::new())
        .await
        .expect("empty");

    // ---- a cancelled read: dropped while a range stalls ----
    let stalled = key("get-cancelled");
    state.put_object(
        BUCKET,
        &stalled,
        MockObject::new(pattern(N, 4)).with_fault(Some(4), 1, ObjectFault::StallBody { send: 2 }),
    );
    let stall_progress = ReceiveProgress::new();
    let mut stall_dst = Dst::new(N);
    let outcome = tokio::time::timeout(
        Duration::from_millis(800),
        on.get_into(BUCKET, &stalled, stall_dst.slice(), &stall_progress),
    )
    .await;
    assert!(outcome.is_err());
    let cancelled_bytes = stall_progress.filled();
    assert!(cancelled_bytes >= 6, "filled {cancelled_bytes}");

    // ---- HEAD: present, absent, refused, cancelled; and no row without a logger ----
    let head_ok = key("head-ok");
    state.put_object(BUCKET, &head_ok, MockObject::new(pattern(N, 5)));
    on.head_typed(BUCKET, &head_ok)
        .await
        .unwrap()
        .expect("present");
    off.head_typed(BUCKET, &head_ok)
        .await
        .unwrap()
        .expect("present");

    let head_absent = key("head-absent");
    state.put_object(BUCKET, &head_absent, MockObject::missing());
    assert!(on.head_typed(BUCKET, &head_absent).await.unwrap().is_none());

    let head_forbidden = key("head-forbidden");
    let mut object = MockObject::new(pattern(N, 6));
    object.head_status = Some(403);
    state.put_object(BUCKET, &head_forbidden, object);
    on.head_typed(BUCKET, &head_forbidden)
        .await
        .expect_err("refused");

    let head_slow = key("head-cancelled");
    let mut object = MockObject::new(pattern(N, 7));
    object.head_delay = Some(Duration::from_secs(30));
    state.put_object(BUCKET, &head_slow, object);
    let outcome = tokio::time::timeout(
        Duration::from_millis(300),
        on.head_typed(BUCKET, &head_slow),
    )
    .await;
    assert!(outcome.is_err(), "the HEAD was still waiting");

    s3dlio::finalize_op_logger();
    let rows = read_rows(&oplog);

    // GET rows: exactly one each, with the outcome of the call.
    let one = |op: &str, key: &str| -> Row {
        let found = rows_of(&rows, op, key);
        assert_eq!(found.len(), 1, "{op} {key}: rows {found:?}");
        found[0].clone()
    };
    let r = one("GET", &ok);
    assert_eq!((r[COL_BYTES].as_str(), r[COL_ERROR].as_str()), ("16", ""));
    assert!(!r[COL_FIRST_BYTE].is_empty(), "a GET row has a first byte");

    assert_eq!(
        rows.iter().filter(|r| r[COL_FILE] == quiet).count(),
        0,
        "a handler without a logger wrote a row"
    );
    assert_eq!(
        rows_of(&rows, "HEAD", &head_ok).len(),
        1,
        "only the logged HEAD wrote a row"
    );

    let r = one("GET", &partial);
    assert_eq!(r[COL_BYTES], "6", "the touched positions, with an error");
    assert!(!r[COL_ERROR].is_empty());

    let r = one("GET", &absent);
    assert_eq!(r[COL_BYTES], "0");
    assert!(!r[COL_ERROR].is_empty());

    let r = one("GET", &zero);
    assert_eq!((r[COL_BYTES].as_str(), r[COL_ERROR].as_str()), ("0", ""));

    let r = one("GET", &stalled);
    assert_eq!(r[COL_ERROR], "cancelled");
    assert_eq!(r[COL_BYTES], cancelled_bytes.to_string());
    assert!(!r[COL_FIRST_BYTE].is_empty());

    // HEAD rows: the size on success, 0 otherwise, and never a first byte.
    let r = one("HEAD", &head_ok);
    assert_eq!((r[COL_BYTES].as_str(), r[COL_ERROR].as_str()), ("16", ""));
    assert_eq!(r[COL_FIRST_BYTE], "", "a HEAD row has no first byte");

    let r = one("HEAD", &head_absent);
    assert_eq!((r[COL_BYTES].as_str(), r[COL_ERROR].as_str()), ("0", ""));
    assert_eq!(r[COL_FIRST_BYTE], "");

    let r = one("HEAD", &head_forbidden);
    assert_eq!(r[COL_BYTES], "0");
    assert!(!r[COL_ERROR].is_empty());
    assert_eq!(r[COL_FIRST_BYTE], "");

    let r = one("HEAD", &head_slow);
    assert_eq!(
        (r[COL_BYTES].as_str(), r[COL_ERROR].as_str()),
        ("0", "cancelled")
    );
    assert_eq!(r[COL_FIRST_BYTE], "");

    // No other rows: nothing for the internal ranges, retries or the fan-out HEAD.
    // Five GET rows and four HEAD rows, and no more.
    assert_eq!(rows.len(), 9, "unexpected extra rows: {rows:?}");
}
