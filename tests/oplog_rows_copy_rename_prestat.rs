// tests/oplog_rows_copy_rename_prestat.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log rows for `copy`, `rename`, `pre_stat_objects` and `pre_stat_and_cache`
// through a LOGGED store (s3dlio #177 / #169 W2 round 2).
//
// Round 1 forwarded these four calls to the inner store so that its own
// implementation runs. That is right when the inner implementation is a native
// backend operation (`rename(2)` on file://), but on S3 (and Azure, GCS,
// multi-endpoint) the inner implementation is the trait DEFAULT: copy = GET + PUT,
// rename = copy + DELETE, pre_stat = one HEAD per object. Those are real requests,
// and forwarding them hid every one of them from the op-log (0 rows).
//
// The contract pinned here, per logical operation (R = rows, Q = requests the
// mock server received):
//
//   backend     operation            R                      Q
//   ----------  -------------------  ---------------------  ---------------------------
//   s3 mock     copy                 GET, PUT               GET src, PUT dst
//   s3 mock     rename               GET, PUT, DELETE       GET src, PUT dst, DELETE src
//   s3 mock     copy, PUT fails      GET, PUT(error)        GET, PUT x attempts (>= 2)
//   s3 mock     pre_stat_objects     HEAD per object        HEAD per object
//   s3 mock     pre_stat_and_cache   HEAD per object        HEAD per object (no extra)
//   s3 mock     get_range_optimized  GET_RANGE              ranged GET only (size cache is full)
//               after pre_stat_and_cache
//   file://     copy                 GET, PUT               read, write
//   file://     rename               none (rename(2))       one rename(2)
//   file://     pre_stat_objects     HEAD per object        one stat per object
//   file://     pre_stat_and_cache   HEAD per object        one stat per object
//
// SDK retries add requests, never rows: the failing PUT is ONE row.
// The op-log never records its own I/O: no row names the log file.
//
// Own process (the op-log and the S3 client are process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, HEAD_NOT_FOUND_MARKER, PUT_FAILS_MARKER,
};
use common::oplog_rows::{ops, read_rows, Row, COL_ERROR, COL_FILE};
use s3dlio::{
    finalize_op_logger, global_logger, init_op_logger, set_client_id, store_for_uri,
    store_for_uri_with_logger,
};

const BUCKET: &str = "mock-bucket";

fn files(rows: &[Row]) -> Vec<String> {
    rows.iter().map(|r| r[COL_FILE].clone()).collect()
}

/// `(op, file)` of each row, sorted: for the phases whose calls run concurrently.
fn sorted_pairs(rows: &[Row]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = rows
        .iter()
        .map(|r| (r[2].clone(), r[COL_FILE].clone()))
        .collect();
    v.sort();
    v
}

#[tokio::test]
async fn copy_rename_and_pre_stat_record_their_real_requests() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (port, server) = ensure_mock_s3_server();
    let mock = format!("127.0.0.1:{port}");
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    set_client_id("client-copy-rename").unwrap();

    // Native-ness is declared by the backend that really overrides the call.
    let s3_plain = store_for_uri(&format!("s3://{BUCKET}/")).unwrap();
    let file_plain = store_for_uri("file:///tmp").unwrap();
    let direct_plain = store_for_uri("direct:///tmp").unwrap();
    assert!(!s3_plain.has_native_copy() && !s3_plain.has_native_rename());
    assert!(!file_plain.has_native_copy() && file_plain.has_native_rename());
    assert!(!direct_plain.has_native_copy() && direct_plain.has_native_rename());

    // ------------------------------------------------------------------ S3 mock
    let s3 = store_for_uri_with_logger(&format!("s3://{BUCKET}/"), global_logger()).unwrap();
    let (a, b, c, d) = (
        unique_test_key("cr-a"),
        unique_test_key("cr-b"),
        unique_test_key("cr-c"),
        unique_test_key("cr-d"),
    );
    let uri = |k: &str| format!("s3://{BUCKET}/{k}");
    let fail_dst = unique_test_key(PUT_FAILS_MARKER);
    let missing = unique_test_key(HEAD_NOT_FOUND_MARKER);

    // phase 1: copy a -> b          rows GET a, PUT b
    s3.copy(&uri(&a), &uri(&b)).await.unwrap();
    // phase 2: rename b -> c        rows GET b, PUT c, DELETE b
    s3.rename(&uri(&b), &uri(&c)).await.unwrap();
    // phase 3: copy whose PUT fails rows GET a, PUT fail_dst (error)
    assert!(s3.copy(&uri(&a), &uri(&fail_dst)).await.is_err());
    // phase 4: pre_stat_objects     rows HEAD a, HEAD c, HEAD missing (error)
    let sizes = s3
        .pre_stat_objects(&[uri(&a), uri(&c), uri(&missing)], 2)
        .await
        .unwrap();
    assert_eq!(sizes.len(), 2, "the missing object is omitted: {sizes:?}");
    // phase 5: pre_stat_and_cache   rows HEAD d, HEAD a
    let cached = s3.pre_stat_and_cache(&[uri(&d), uri(&a)], 2).await.unwrap();
    assert_eq!(cached, 2);
    // phase 6: get_range_optimized without a length asks the size cache for the
    // object size: pre_stat_and_cache filled it, so there is NO new HEAD, one ranged
    // GET, one GET_RANGE row.
    s3.get_range_optimized(&uri(&d), 0, None, None, None)
        .await
        .unwrap();

    // ----------------------------------------------------------------- file://
    let dir = tempfile::tempdir().unwrap();
    let base = format!("file://{}", dir.path().display());
    let fs = store_for_uri_with_logger(&base, global_logger()).unwrap();
    let (fa, fb, fc) = (
        format!("{base}/fa.bin"),
        format!("{base}/fb.bin"),
        format!("{base}/fc.bin"),
    );
    // phase 7: put fa                rows PUT fa
    fs.put(&fa, Bytes::from_static(b"abcd")).await.unwrap();
    // phase 8: copy fa -> fb         rows GET fa, PUT fb
    fs.copy(&fa, &fb).await.unwrap();
    // phase 9: rename fb -> fc       NO rows: native rename(2)
    fs.rename(&fb, &fc).await.unwrap();
    assert!(!dir.path().join("fb.bin").exists());
    assert_eq!(std::fs::read(dir.path().join("fc.bin")).unwrap(), b"abcd");
    // phase 10: pre_stat_objects     rows HEAD fa, HEAD fc
    let sizes = fs
        .pre_stat_objects(&[fa.clone(), fc.clone()], 2)
        .await
        .unwrap();
    assert_eq!(sizes.values().copied().collect::<Vec<_>>(), [4, 4]);
    // phase 11: pre_stat_and_cache   rows HEAD fa, HEAD fc
    assert_eq!(
        fs.pre_stat_and_cache(&[fa.clone(), fc.clone()], 2)
            .await
            .unwrap(),
        2
    );

    finalize_op_logger();
    let rows = read_rows(&oplog);

    // ------------------------------------------------------------------- rows
    let mut at = 0usize;
    let mut take = |n: usize| {
        let s = &rows[at..at + n];
        at += n;
        s
    };
    let key_of = |r: &Row| r[COL_FILE].rsplit('/').next().unwrap().to_string();

    let p1 = take(2);
    assert_eq!(ops(p1), ["GET", "PUT"], "s3 copy: {p1:?}");
    assert_eq!((key_of(&p1[0]), key_of(&p1[1])), (a.clone(), b.clone()));

    let p2 = take(3);
    assert_eq!(ops(p2), ["GET", "PUT", "DELETE"], "s3 rename: {p2:?}");
    assert_eq!(
        (key_of(&p2[0]), key_of(&p2[1]), key_of(&p2[2])),
        (b.clone(), c.clone(), b.clone())
    );

    let p3 = take(2);
    assert_eq!(ops(p3), ["GET", "PUT"], "s3 copy with failing PUT: {p3:?}");
    assert!(p3[0][COL_ERROR].is_empty(), "the GET succeeded: {p3:?}");
    assert!(
        !p3[1][COL_ERROR].is_empty(),
        "the failed PUT row carries the error: {p3:?}"
    );

    let p4 = take(3);
    assert_eq!(
        sorted_pairs(p4),
        {
            let mut v = vec![
                ("HEAD".to_string(), format!("{BUCKET}/{a}")),
                ("HEAD".to_string(), format!("{BUCKET}/{c}")),
                ("HEAD".to_string(), format!("{BUCKET}/{missing}")),
            ];
            v.sort();
            v
        },
        "s3 pre_stat_objects: one HEAD row per object"
    );
    assert_eq!(
        p4.iter().filter(|r| !r[COL_ERROR].is_empty()).count(),
        1,
        "only the missing object's HEAD failed: {p4:?}"
    );

    let p5 = take(2);
    assert_eq!(
        sorted_pairs(p5),
        {
            let mut v = vec![
                ("HEAD".to_string(), format!("{BUCKET}/{d}")),
                ("HEAD".to_string(), format!("{BUCKET}/{a}")),
            ];
            v.sort();
            v
        },
        "s3 pre_stat_and_cache: one HEAD row per object"
    );

    let p6 = take(1);
    assert_eq!(
        ops(p6),
        ["GET_RANGE"],
        "range read of a cached object: {p6:?}"
    );

    let p7 = take(1);
    assert_eq!(ops(p7), ["PUT"]);
    let p8 = take(2);
    assert_eq!(ops(p8), ["GET", "PUT"], "file copy: {p8:?}");
    assert_eq!(
        files(p8),
        [fa.clone(), fb.clone()].map(|u| u.replace("file://", ""))
    );
    // phase 9: native rename, no rows.
    let p10 = take(2);
    let p11 = take(2);
    for (name, p) in [("pre_stat_objects", p10), ("pre_stat_and_cache", p11)] {
        assert_eq!(
            sorted_pairs(p),
            {
                let mut v = vec![
                    ("HEAD".to_string(), fa.replace("file://", "")),
                    ("HEAD".to_string(), fc.replace("file://", "")),
                ];
                v.sort();
                v
            },
            "file {name}: one HEAD row per object"
        );
    }
    assert_eq!(
        at,
        rows.len(),
        "no other row: the op-log must not record its own I/O or anything unexpected: {:#?}",
        &rows[at..]
    );
    assert!(
        rows.iter().all(|r| !r[COL_FILE].ends_with("ops.tsv.zst")),
        "the op-log never records its own I/O"
    );

    // ----------------------------------------------------------- what the server saw
    let reqs = |k: &str| server.requests_containing(k);
    let methods = |k: &str| reqs(k).iter().map(|r| r.method.clone()).collect::<Vec<_>>();
    for r in server.requests_containing("cr-") {
        assert_eq!(r.host.as_deref(), Some(mock.as_str()), "{r:?}");
        assert!(r.path.starts_with(&format!("/{BUCKET}/")), "{r:?}");
    }
    // The S3 read path probes an unknown object's size with a HEAD before its GET
    // (and remembers it process-wide), a request that has never had a row of its
    // own; the counts below include those probes.
    // a: copy source twice (phases 1 and 3): 2 GETs; HEADs: 1 size probe (phase 1;
    // phase 3 finds the size cached), pre_stat_objects (phase 4) and
    // pre_stat_and_cache (phase 5).
    let ma = methods(&a);
    assert_eq!(ma.iter().filter(|m| *m == "GET").count(), 2, "{ma:?}");
    assert_eq!(ma.iter().filter(|m| *m == "HEAD").count(), 3, "{ma:?}");
    // b: written by copy (PUT), read by rename (size probe HEAD + GET), deleted by
    // rename (a DeleteObjects POST naming the key in its body).
    let mut mb = methods(&b);
    mb.sort();
    assert_eq!(mb, ["GET", "HEAD", "PUT"], "{mb:?}");
    assert_eq!(
        server.counts_for(&key_path(BUCKET, &b)).batch_delete_calls,
        1,
        "rename deleted its source"
    );
    // c: the rename destination (PUT) and a HEAD from phase 4.
    let mut mc = methods(&c);
    mc.sort();
    assert_eq!(mc, ["HEAD", "PUT"], "{mc:?}");
    // d: ONE HEAD (pre_stat_and_cache) and ONE ranged GET. The ranged read asked
    // the size cache for the size: no second HEAD, because pre_stat_and_cache filled
    // the inner store's cache from the sizes it had already learned.
    let mut md: Vec<String> = server
        .requests_containing(&d)
        .iter()
        .map(|r| {
            format!(
                "{}{}",
                r.method,
                if r.range.is_some() { " range" } else { "" }
            )
        })
        .collect();
    md.sort();
    assert_eq!(
        md,
        ["GET range", "HEAD"],
        "pre_stat_and_cache must fill the cache without an extra request: {md:?}"
    );
    let counts = server.counts_for(&key_path(BUCKET, &fail_dst));
    assert!(
        counts.put_calls >= 2,
        "the SDK retried the failing PUT: {counts:?}"
    );
    // The missing object was asked for once (HEAD), not retried: 404 is final.
    assert_eq!(server.counts_for(&key_path(BUCKET, &missing)).head_calls, 1);
}
