// tests/oplog_mkdir_rmdir_229.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #229 + #228 in the op-log:
//   * `mkdir` writes NO row on any backend (file://, direct://, s3://).
//   * `rmdir` writes ONE `DELETE_PREFIX` row on file:// and direct:// -- a directory
//     really is removed, or the removal attempt fails.
//   * `rmdir` on an object store (s3://) is a no-op that sends no request, deletes
//     nothing and writes NO row: a DELETE_PREFIX row would claim that everything
//     under the prefix was deleted.
//
// One process (the op-log is process-global); the whole log is read back at the end
// so an extra row shows up.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;
#[path = "common/recording_s3.rs"]
mod recording_s3;

use common::oplog_rows::{ops, read_rows, COL_ENDPOINT, COL_ERROR, COL_FILE};
use recording_s3::MockS3;
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};

#[tokio::test]
async fn mkdir_never_and_rmdir_only_on_filesystems_write_rows() {
    let a = MockS3::start(&[]);
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::remove_var("AWS_ENDPOINT_URL_S3");
        std::env::set_var("AWS_ENDPOINT_URL", a.url());
    }
    let work = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("oplog_mkdir_229");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    let root = work.display().to_string();

    let file = store_for_uri_with_logger(&format!("file://{root}"), global_logger()).unwrap();
    let direct = store_for_uri_with_logger(&format!("direct://{root}"), global_logger()).unwrap();
    let s3 = store_for_uri_with_logger("s3://bkt229/dir", global_logger()).unwrap();

    // mkdir: no row anywhere.
    file.mkdir(&format!("file://{root}/f/x")).await.unwrap();
    direct.mkdir(&format!("direct://{root}/d/x")).await.unwrap();
    s3.mkdir("s3://bkt229/dir").await.unwrap();
    s3.mkdir("s3://bkt229/dir/").await.unwrap();

    // rmdir on the filesystems: one DELETE_PREFIX row each.
    file.rmdir(&format!("file://{root}/f"), true).await.unwrap();
    direct
        .rmdir(&format!("direct://{root}/d"), true)
        .await
        .unwrap();
    // A failing removal is recorded too (non-empty, non-recursive).
    std::fs::create_dir_all(format!("{root}/full")).unwrap();
    std::fs::write(format!("{root}/full/keep"), b"k").unwrap();
    let err = direct
        .rmdir(&format!("direct://{root}/full"), false)
        .await
        .expect_err("non-empty non-recursive rmdir must fail")
        .to_string();

    // rmdir on an object store: success, no request, no row.
    s3.rmdir("s3://bkt229/dir", true).await.unwrap();
    s3.rmdir("s3://bkt229/dir/", false).await.unwrap();

    finalize_op_logger();
    let rows = read_rows(&oplog);
    assert_eq!(
        ops(&rows),
        ["DELETE_PREFIX"; 3],
        "one row per filesystem rmdir; none for mkdir or for object-store rmdir: {rows:#?}"
    );
    assert_eq!(&rows[0][COL_ENDPOINT], "file://");
    assert_eq!(&rows[0][COL_FILE], &format!("{root}/f"));
    assert_eq!(&rows[0][COL_ERROR], "");
    assert_eq!(&rows[1][COL_ENDPOINT], "direct://");
    assert_eq!(&rows[1][COL_FILE], &format!("{root}/d"));
    assert_eq!(&rows[1][COL_ERROR], "");
    assert_eq!(&rows[2][COL_ENDPOINT], "direct://");
    assert_eq!(&rows[2][COL_ERROR], &err);
    assert_eq!(
        a.requests(),
        Vec::<String>::new(),
        "the S3 server got a request"
    );
    std::fs::remove_dir_all(&work).unwrap();
}
