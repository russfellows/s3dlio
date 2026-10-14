// tests/oplog_exactly_once_callers.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log exactly-once contract (s3dlio #169): library entry points that build
// their own store (checkpoint save/load, the multi-backend dataset) used to be
// recorded only by the S3 backend's legacy row, and only on S3. They now ask for
// a logged store explicitly, so with the op-log active every call is ONE row in
// the LoggedObjectStore shape on every backend. Run on `file://` so the test
// needs no server and nothing hides a missing row behind the legacy S3 row.
//
// Own process (the op-log is process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::oplog_rows::{ops, read_rows, COL_CLIENT_ID, COL_ENDPOINT, COL_FILE};
use s3dlio::checkpoint::CheckpointStore;
use s3dlio::data_loader::async_pool_dataloader::MultiBackendDataset;
use s3dlio::{finalize_op_logger, init_op_logger, set_client_id};

#[tokio::test]
async fn library_entry_points_record_each_call_once() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    set_client_id("callers").unwrap();

    // Dataset: two plain files read through the dataset's store.
    std::fs::write(data_dir.join("a.bin"), b"aaaa").unwrap();
    std::fs::write(data_dir.join("b.bin"), b"bbbbbb").unwrap();
    let uris: Vec<String> = ["a.bin", "b.bin"]
        .iter()
        .map(|n| format!("file://{}/{n}", data_dir.display()))
        .collect();
    let dataset = MultiBackendDataset::from_uris(uris.clone()).unwrap();
    dataset.store.get(&uris[0]).await.unwrap();
    dataset.store.get(&uris[1]).await.unwrap();

    // Checkpoint: one save and one load of the latest.
    let ckpt_uri = format!("file://{}/ckpt", dir.path().display());
    std::fs::create_dir(dir.path().join("ckpt")).unwrap();
    let ckpt = CheckpointStore::open(&ckpt_uri).unwrap();
    ckpt.save(1, 0, "test", b"checkpoint-bytes", None)
        .await
        .unwrap();
    let loaded = ckpt.load_latest().await.unwrap().expect("a checkpoint");
    assert_eq!(&loaded[..], b"checkpoint-bytes");

    finalize_op_logger();

    let rows = read_rows(&oplog);
    // Two dataset GETs, then the checkpoint's own calls: save = shard PUT, size
    // check (HEAD), manifest PUT, latest-pointer PUT; load = latest GET, manifest
    // GET, shard GET. One row each.
    assert_eq!(
        ops(&rows),
        ["GET", "GET", "PUT", "HEAD", "PUT", "PUT", "GET", "GET", "GET"],
        "rows: {rows:#?}"
    );
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r[COL_ENDPOINT], "file://", "row {i}");
        assert_eq!(r[COL_CLIENT_ID], "callers", "row {i}");
        assert!(r[COL_FILE].starts_with('/'), "row {i}: {r:?}");
    }
    assert_ne!(rows[0][COL_FILE], rows[1][COL_FILE]);
    assert!(rows[2][COL_FILE].ends_with("/rank-0.bin"));
    assert!(rows[5][COL_FILE].ends_with("/latest.json"));
}
