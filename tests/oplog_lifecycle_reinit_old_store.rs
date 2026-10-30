// tests/oplog_lifecycle_reinit_old_store.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle regression test for s3dlio issue #294.
//
// The op-log is process-global, so each lifecycle scenario lives in its OWN
// integration-test file (a separate binary, hence a separate process).

// Scenario: a REAL store captured in session 1 (store_for_uri_with_logger with
// global_logger()) keeps working after finalize AND after a re-init, but it
// holds session 1's closed Logger, so it never writes a row into session 2's
// file. A store built in session 2 is recorded in session 2's file only.

use bytes::Bytes;
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};

const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

/// (op, file) of every row of a complete op-log.
fn rows(path: &std::path::Path) -> Vec<(String, String)> {
    let raw = std::fs::read(path).unwrap();
    let text = String::from_utf8(zstd::decode_all(raw.as_slice()).unwrap()).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(HEADER_LINE));
    lines
        .enumerate()
        .map(|(i, l)| {
            let c: Vec<&str> = l.split('\t').collect();
            assert_eq!(c.len(), 13, "{l:?}");
            assert_eq!(c[0], i.to_string());
            (c[2].to_string(), c[7].to_string())
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn store_from_session_one_never_writes_into_session_two() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let base = format!("file://{}", dir.path().display());
    let (log1, log2) = (dir.path().join("s1.tsv.zst"), dir.path().join("s2.tsv.zst"));

    init_op_logger(log1.to_str().unwrap()).unwrap();
    let old_store = store_for_uri_with_logger(&base, global_logger()).unwrap();
    old_store
        .put(&format!("{base}/one.bin"), Bytes::from_static(b"one"))
        .await
        .unwrap();
    finalize_op_logger();
    let log1_bytes = std::fs::read(&log1).unwrap();

    init_op_logger(log2.to_str().unwrap()).unwrap();
    let new_store = store_for_uri_with_logger(&base, global_logger()).unwrap();
    // Old store, used while session 2 is Active: works, records nothing anywhere.
    for i in 0..20 {
        old_store
            .put(&format!("{base}/old-{i}.bin"), Bytes::from_static(b"old"))
            .await
            .unwrap();
    }
    assert_eq!(
        &old_store.get(&format!("{base}/old-0.bin")).await.unwrap()[..],
        b"old"
    );
    // New store: recorded in session 2.
    new_store
        .put(&format!("{base}/two.bin"), Bytes::from_static(b"two"))
        .await
        .unwrap();
    finalize_op_logger();

    let r2 = rows(&log2);
    assert_eq!(r2.len(), 1, "only the session-2 store is recorded: {r2:?}");
    assert_eq!(r2[0].0, "PUT");
    assert!(r2[0].1.ends_with("two.bin"), "{r2:?}");
    assert_eq!(
        std::fs::read(&log1).unwrap(),
        log1_bytes,
        "session 1's log changed after it was finalized"
    );
    let r1 = rows(&log1);
    assert_eq!(r1.len(), 1, "{r1:?}");
    assert!(r1[0].1.ends_with("one.bin"), "{r1:?}");
}
