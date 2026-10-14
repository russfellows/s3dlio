// tests/oplog_helper_rows_golden.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log golden test (s3dlio #169, W2 round 2): the public S3 helper functions
// write EXACTLY the row they wrote before W2, in every field, for success and
// for failure.
//
// "Before W2" is the S3 backend's own row writer (`S3Ops::log_op`): endpoint =
// the AWS_ENDPOINT_URL value, file = the bare key, client_id = the access key id
// (never `set_client_id()`), `bytes` = the data size on success and 0 on failure,
// `first_byte` = the measured response-header time for a GET and empty for a PUT,
// and the `error` text as that writer formatted it. The expected rows below were
// CAPTURED FROM THE PARENT COMMIT (9473d2c, before W2) with this very test
// running in capture mode against the same mock-server calls:
//
//     OPLOG_GOLDEN_OUT=<file> cargo test --test oplog_helper_rows_golden
//
// writes the rows the code under test produced, instead of comparing them. The
// fixture `tests/fixtures/oplog_helper_rows_pre_w2.golden` is that capture; this
// test then fails on any difference in the NORMALISED fields it compares, not in
// the raw file byte for byte. (The timestamps are normalized to "where does
// first_byte sit between start and end"; the thread id and the mock server's port
// are dropped.)
//
// Alongside the rows, the test checks what the mock server actually received for
// each call (method, path, host:port, number of attempts), so that "one row" is
// never mistaken for "one request": SDK retries add requests, never rows.
//
// Own process (the op-log and the S3 client are process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{
    ensure_mock_s3_server, GET_FORBIDDEN_MARKER, PUT_FAILS_MARKER, PUT_FORBIDDEN_MARKER,
};
use common::oplog_rows::{read_rows, Row};
use s3dlio::data_loader::dataset::Dataset;
use s3dlio::data_loader::s3_bytes::S3BytesDataset;
use s3dlio::s3_client::run_on_global_rt;
use s3dlio::s3_utils::{
    delete_objects, get_object, get_object_uri, get_object_uri_async,
    get_object_uri_optimized_async, get_objects_parallel, get_range, put_object_async,
    put_object_uri_async,
};
use s3dlio::{finalize_op_logger, init_op_logger, set_client_id};

const BUCKET: &str = "mock-bucket";
const GOLDEN: &str = include_str!("fixtures/oplog_helper_rows_pre_w2.golden");

/// One row reduced to the fields the contract pins. `first_byte` is reduced to
/// where it sits: `none` (empty), `start`, `end`, or `mid` (strictly between).
fn canonical(row: &Row, mock: &str) -> String {
    let scrub = |s: &str| scrub_error(&s.replace(mock, "MOCK"));
    let (start, first_byte, end) = (&row[9], &row[10], &row[11]);
    let fb = if first_byte.is_empty() {
        "none"
    } else if first_byte == start {
        "start"
    } else if first_byte == end {
        "end"
    } else {
        assert!(
            start < first_byte && first_byte < end,
            "first_byte must lie within [start, end]: {row:?}"
        );
        "mid"
    };
    format!(
        "op={} client_id={} n_objects={} bytes={} endpoint={} file={} error=[{}] first_byte={}",
        row[2],
        row[3],
        row[4],
        row[5],
        scrub(&row[6]),
        scrub(&row[7]),
        scrub(&row[8]),
        fb
    )
}

/// Error texts can carry values that differ from run to run (a `Date` response
/// header, the request id the mock fixes anyway): drop the `date: "..."` part.
fn scrub_error(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("\"date\": \"") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 9..];
        match tail.find('"') {
            Some(j) => {
                out.push_str("\"date\": \"<date>\"");
                rest = &tail[j + 1..];
            }
            None => {
                rest = tail;
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

fn put_ok() -> Bytes {
    Bytes::from_static(b"data")
}

#[test]
fn helper_rows_are_byte_identical_to_pre_w2() {
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
    // The helper rows never use this id (they carry the access key id).
    set_client_id("client-golden").unwrap();

    let uri = |key: &str| format!("s3://{BUCKET}/{key}");
    let put403 = format!("golden/{PUT_FORBIDDEN_MARKER}-a");
    let put403u = format!("golden/{PUT_FORBIDDEN_MARKER}-uri");
    let get403 = format!("golden/{GET_FORBIDDEN_MARKER}-a");
    let get403o = format!("golden/{GET_FORBIDDEN_MARKER}-opt");
    let get403p = format!("golden/{GET_FORBIDDEN_MARKER}-par");

    eprintln!("step: put/get helpers");
    // --- PUT / GET helpers, success and failure -------------------------------
    {
        let (put403, put403u, get403, get403o) = (
            put403.clone(),
            put403u.clone(),
            get403.clone(),
            get403o.clone(),
        );
        run_on_global_rt(async move {
            put_object_async(BUCKET, "golden/put-ok", put_ok()).await?;
            assert!(put_object_async(BUCKET, &put403, put_ok()).await.is_err());
            put_object_uri_async(&format!("s3://{BUCKET}/golden/put-uri-ok"), put_ok()).await?;
            assert!(
                put_object_uri_async(&format!("s3://{BUCKET}/{put403u}"), put_ok())
                    .await
                    .is_err()
            );
            get_object(BUCKET, "golden/get-ok").await?;
            assert!(get_object(BUCKET, &get403).await.is_err());
            get_object_uri_async(&format!("s3://{BUCKET}/golden/get-uri-ok")).await?;
            get_object_uri_optimized_async(&format!("s3://{BUCKET}/golden/get-opt-ok")).await?;
            assert!(
                get_object_uri_optimized_async(&format!("s3://{BUCKET}/{get403o}"))
                    .await
                    .is_err()
            );
            Ok(())
        })
        .unwrap();
    }
    eprintln!("step: sync get, get_range, parallel");
    get_object_uri(&uri("golden/get-sync-ok")).unwrap();
    get_range(BUCKET, "golden/get-range-ok", 1, Some(2)).unwrap();
    get_objects_parallel(&[uri("golden/par-1"), uri("golden/par-2")], 2).unwrap();
    assert!(get_objects_parallel(&[uri(&get403p)], 1).is_err());

    eprintln!("step: retried put");
    // --- a retried failure: several requests, ONE row --------------------------
    let put500 = format!("golden/{PUT_FAILS_MARKER}-retried");
    {
        let put500 = put500.clone();
        run_on_global_rt(async move {
            assert!(put_object_async(BUCKET, &put500, put_ok()).await.is_err());
            Ok(())
        })
        .unwrap();
    }

    eprintln!("step: dataset and s3_copy");
    // --- S3BytesDataset (sequential) and s3_copy --------------------------------
    {
        let ds = S3BytesDataset::from_prefix(&uri("golden/ds/")).unwrap();
        assert_eq!(ds.len(), Some(2), "the mock lists two keys under a prefix");
        run_on_global_rt(async move {
            ds.get(0).await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
            ds.get(1).await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
            Ok(())
        })
        .unwrap();
    }
    {
        let up = tempfile::tempdir().unwrap();
        let file = up.path().join("golden-upload.bin");
        std::fs::write(&file, b"upload").unwrap();
        let down = tempfile::tempdir().unwrap();
        // s3_copy calls the synchronous helpers (list, create_bucket) from its async
        // body, which block on the global runtime: drive it from a separate runtime.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            s3dlio::s3_copy::upload_files(
                &format!("s3://{BUCKET}/golden/up/"),
                &[file],
                1,
                false,
                None,
            )
            .await?;
            s3dlio::s3_copy::download_objects(
                &format!("s3://{BUCKET}/golden/down/"),
                down.path(),
                1,
                true,
                None,
            )
            .await?;
            Ok::<(), anyhow::Error>(())
        })
        .unwrap();
    }

    eprintln!("step: delete_objects");
    // --- delete_objects: 0, 1, 1000, 1001 keys and a per-key failure ------------
    let keys = |prefix: &str, n: usize| -> Vec<String> {
        (0..n).map(|i| format!("golden/del/{prefix}-{i}")).collect()
    };
    delete_objects(BUCKET, &[]).unwrap();
    delete_objects(BUCKET, &keys("one", 1)).unwrap();
    delete_objects(BUCKET, &keys("k1000", 1000)).unwrap();
    delete_objects(BUCKET, &keys("k1001", 1001)).unwrap();
    assert!(delete_objects(BUCKET, &["golden/del/delete-fails-one".to_string()]).is_err());

    finalize_op_logger();

    // --- what the server saw ---------------------------------------------------
    // Every request of every call went to the one mock server, addressed by host:port.
    let all = server.requests_containing("golden/");
    assert!(!all.is_empty());
    for r in &all {
        assert_eq!(
            r.host.as_deref(),
            Some(mock.as_str()),
            "request {} {} went to the wrong host",
            r.method,
            r.path
        );
    }
    let attempts = |method: &str, key: &str| {
        all.iter()
            .filter(|r| r.method == method && r.path == format!("/{BUCKET}/{key}"))
            .count()
    };
    assert_eq!(attempts("PUT", "golden/put-ok"), 1);
    assert_eq!(attempts("PUT", &put403), 1, "a 403 is not retried");
    assert!(
        attempts("PUT", &put500) >= 2,
        "the SDK retried the failing PUT ({} attempts)",
        attempts("PUT", &put500)
    );
    // DeleteObjects names its keys in the body, not the path: select by query.
    let delete_posts = server
        .requests_containing("delete")
        .iter()
        .filter(|r| r.method == "POST" && r.query.contains("delete"))
        .count();
    // 0 keys: no request; 1 key: 1; 1000 keys: 1; 1001 keys: 2; failing key: 1.
    assert_eq!(delete_posts, 1 + 1 + 2 + 1);

    // --- the rows ---------------------------------------------------------------
    let rows = read_rows(&oplog);
    let mut lines: Vec<String> = rows.iter().map(|r| canonical(r, &mock)).collect();
    lines.sort();
    let actual = format!("{}\n", lines.join("\n"));

    if let Ok(out) = std::env::var("OPLOG_GOLDEN_OUT") {
        std::fs::write(&out, &actual).unwrap();
        eprintln!("captured {} rows into {out}", lines.len());
        return;
    }
    assert_eq!(
        actual, GOLDEN,
        "helper rows differ from the rows captured from the parent (pre-W2)"
    );

    // The facts the golden pins, stated once more so a regenerated fixture cannot
    // silently change them: one row per call (a retried PUT is ONE row), the
    // delete chunks are `(batch)` rows of 1000 and 1, an empty list writes none.
    let batch: Vec<&Row> = rows.iter().filter(|r| r[7] == "(batch)").collect();
    let n_of = |r: &&Row| r[4].clone();
    let mut batch_sizes: Vec<String> = batch.iter().map(n_of).collect();
    batch_sizes.sort();
    assert_eq!(
        batch_sizes,
        ["1", "1", "1", "1000", "1000"],
        "delete_objects rows: one `(batch)` row per chunk of at most 1000 keys"
    );
    let retried: Vec<&Row> = rows.iter().filter(|r| r[7].contains("put-500")).collect();
    assert_eq!(retried.len(), 1, "a retried PUT is one row: {retried:?}");
    assert!(rows.iter().all(|r| r[3] == "mock-access-key"));
}
