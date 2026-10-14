// tests/oplog_lifecycle_no_self_log_file_uri.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log invariant (maintainer requirement): the op-log NEVER records its own
// I/O, for any destination form. The writer uses std::fs + zstd directly and
// must not go through a logged ObjectStore.
//
// Scenario: the op-log lives at a file:// URI inside <dir>. A file:// workload
// through a LOGGED store reads and writes other files in the SAME <dir>
// (including listing the directory that contains the op-log file and stat-ing
// a workload file). After finalize the op-log has exactly one row per workload
// call and no row's file column names the op-log file. Own process (the op-log
// is process-global).

use bytes::Bytes;
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};

const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

#[tokio::test]
async fn oplog_does_not_record_its_own_io() {
    // Do not depend on the caller's environment (queue size, lossless mode).
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    let destination = format!("file://{}", oplog.display());
    init_op_logger(&destination).unwrap();

    let store =
        store_for_uri_with_logger(&format!("file://{}", dir.path().display()), global_logger())
            .unwrap();
    let uri = |name: &str| format!("file://{}/{name}", dir.path().display());

    // 6 workload calls: 2 puts, 2 gets, 1 list of the op-log's own directory, 1 stat.
    store
        .put(&uri("a.bin"), Bytes::from_static(b"aaaa"))
        .await
        .unwrap();
    store
        .put(&uri("b.bin"), Bytes::from_static(b"bbbbbb"))
        .await
        .unwrap();
    assert_eq!(&store.get(&uri("a.bin")).await.unwrap()[..], b"aaaa");
    assert_eq!(&store.get(&uri("b.bin")).await.unwrap()[..], b"bbbbbb");
    let listed = store
        .list(&format!("file://{}/", dir.path().display()), false)
        .await
        .unwrap();
    assert!(
        listed.len() >= 2,
        "workload files must be listed: {listed:?}"
    );
    store.stat(&uri("a.bin")).await.unwrap();
    const CALLS: usize = 6;

    finalize_op_logger();

    let raw = std::fs::read(&oplog).expect("op-log must exist at the requested local path");
    let text = String::from_utf8(
        zstd::decode_all(raw.as_slice()).expect("op-log must be a complete zstd stream"),
    )
    .unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(HEADER_LINE));
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    for (i, cols) in rows.iter().enumerate() {
        assert_eq!(cols.len(), 13, "row {i}");
        assert_eq!(cols[0], i.to_string(), "row {i} idx");
        assert!(
            !cols[7].contains("ops.tsv.zst"),
            "row {i} names the op-log file: {cols:?}"
        );
    }
    assert_eq!(
        rows.iter().map(|c| c[2]).collect::<Vec<_>>(),
        ["PUT", "PUT", "GET", "GET", "LIST", "HEAD"],
        "exactly one row per workload call, in order"
    );
    assert_eq!(rows.len(), CALLS);
}
