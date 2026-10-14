// tests/oplog_list_stream_client_id.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #181: the LIST row written when a `list_stream()` stream completes had
// an empty client_id, while the row of `list()` (and every other call) carried
// the id set with `set_client_id()`. In a distributed run the streamed LIST rows
// could not be attributed to a client.
//
// Own process (the op-log is process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::oplog_rows::{
    ops, read_rows, COL_CLIENT_ID, COL_ENDPOINT, COL_ERROR, COL_FILE, COL_N_OBJECTS,
};
use futures::StreamExt;
use s3dlio::{
    finalize_op_logger, global_logger, init_op_logger, set_client_id, store_for_uri_with_logger,
};

#[tokio::test]
async fn streamed_list_rows_carry_the_client_id_like_list_rows() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.bin"), b"a").unwrap();
    std::fs::write(dir.path().join("b.bin"), b"b").unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    set_client_id("client-A").unwrap();

    let store =
        store_for_uri_with_logger(&format!("file://{}", dir.path().display()), global_logger())
            .unwrap();
    let prefix = format!("file://{}/", dir.path().display());

    let mut stream = store.list_stream(&prefix, false);
    let mut streamed = 0;
    while let Some(item) = stream.next().await {
        item.unwrap();
        streamed += 1;
    }
    drop(stream);
    assert_eq!(streamed, 2);
    store.list(&prefix, false).await.unwrap();

    finalize_op_logger();

    let rows = read_rows(&oplog);
    assert_eq!(ops(&rows), ["LIST", "LIST"], "rows: {rows:#?}");
    // The streamed row and the plain row are the same row, apart from timing.
    for col in [
        COL_CLIENT_ID,
        COL_N_OBJECTS,
        COL_ENDPOINT,
        COL_FILE,
        COL_ERROR,
    ] {
        assert_eq!(rows[0][col], rows[1][col], "column {col}: {rows:#?}");
    }
    assert_eq!(rows[0][COL_CLIENT_ID], "client-A");
    assert_eq!(rows[0][COL_N_OBJECTS], "2");
}
