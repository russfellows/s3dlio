// tests/oplog_logged_store_file_transparent.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #177, end to end on a real backend: turning the op-log on must not
// change what the program does. On `file://`, before the fix a logged store
// failed every mkdir/rmdir/update_metadata ("... not implemented for this
// backend"), and turned the atomic rename into read + write + delete (a new
// inode, and three op-log rows).
//
// Own process (the op-log is process-global).

use bytes::Bytes;
use s3dlio::{
    finalize_op_logger, global_logger, init_op_logger, store_for_uri, store_for_uri_with_logger,
};
use std::collections::HashMap;
use std::os::unix::fs::MetadataExt;

#[tokio::test]
async fn logging_does_not_change_file_backend_behaviour() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let base = format!("file://{}", dir.path().display());
    let plain = store_for_uri(&base).unwrap();
    let logged = store_for_uri_with_logger(&base, global_logger()).unwrap();

    // mkdir / rmdir work, exactly as without the op-log.
    plain.mkdir(&format!("{base}/d_plain")).await.unwrap();
    logged.mkdir(&format!("{base}/d_logged")).await.unwrap();
    assert!(dir.path().join("d_logged").is_dir());
    logged
        .rmdir(&format!("{base}/d_logged"), true)
        .await
        .unwrap();
    assert!(!dir.path().join("d_logged").exists());

    // rename is the filesystem's atomic rename: same inode afterwards.
    let a = format!("{base}/a.bin");
    let b = format!("{base}/b.bin");
    logged.put(&a, Bytes::from_static(b"abcd")).await.unwrap();
    let inode_before = std::fs::metadata(dir.path().join("a.bin")).unwrap().ino();
    logged.rename(&a, &b).await.unwrap();
    let inode_after = std::fs::metadata(dir.path().join("b.bin")).unwrap().ino();
    assert_eq!(inode_before, inode_after, "rename must stay atomic");
    assert!(!dir.path().join("a.bin").exists());

    // The same (unsupported) update_metadata error as the unlogged store, not
    // the trait default's.
    let plain_err = plain
        .update_metadata(&b, &HashMap::new())
        .await
        .unwrap_err()
        .to_string();
    let logged_err = logged
        .update_metadata(&b, &HashMap::new())
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(logged_err, plain_err);

    finalize_op_logger();

    // The only call with an op name is the PUT: the forwarded calls write no row.
    let raw = std::fs::read(&oplog).unwrap();
    let text = String::from_utf8(zstd::decode_all(raw.as_slice()).unwrap()).unwrap();
    let ops: Vec<&str> = text
        .lines()
        .skip(1)
        .map(|l| l.split('\t').nth(2).unwrap())
        .collect();
    assert_eq!(ops, ["PUT"], "rows: {text}");
}
