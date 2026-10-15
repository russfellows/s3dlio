// tests/oplog_rmdir_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #228: a recursive `rmdir` through s3dlio removed a whole tree and left
// nothing in the op-log. `rmdir` (recursive or not) is now recorded as ONE
// `DELETE_PREFIX` row on the directory URI, shaped exactly like `delete_prefix`'s
// row (bytes 0, count 0, no first byte, the error text on failure), whether the
// removal succeeds or fails. `DELETE_PREFIX` is an op name the format already
// has: no format change. `mkdir` still writes no row.
//
// Every scenario runs on a real `file://` / `direct://` store built with the
// logger, in one process (the op-log is process-global), and the whole log is
// read back at the end so that an extra (double-logged) row would show up.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::oplog_rows::{
    ops, read_rows, COL_BYTES, COL_ENDPOINT, COL_ERROR, COL_FILE, COL_N_OBJECTS,
};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};

/// Column of the `first_byte` timestamp (empty for a metadata-only call).
const COL_FIRST_BYTE: usize = 10;

#[tokio::test]
async fn rmdir_is_recorded_as_one_delete_prefix_row() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let work = tempfile::tempdir().unwrap();
    let log_dir = tempfile::tempdir().unwrap();
    let oplog = log_dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let root = work.path().display().to_string();
    let store = store_for_uri_with_logger(&format!("file://{root}"), global_logger()).unwrap();

    // 1. Recursive rmdir of a tree of files and subdirectories (mkdir: no row).
    let tree = format!("{root}/tree");
    store.mkdir(&format!("file://{tree}/a/b")).await.unwrap();
    std::fs::write(format!("{tree}/top.bin"), b"x").unwrap();
    std::fs::write(format!("{tree}/a/mid.bin"), b"y").unwrap();
    std::fs::write(format!("{tree}/a/b/leaf.bin"), b"z").unwrap();
    store
        .rmdir(&format!("file://{tree}"), true)
        .await
        .expect("recursive rmdir of a tree");
    assert!(!std::path::Path::new(&tree).exists(), "tree must be gone");

    // 2. Non-recursive rmdir of an empty directory.
    let empty = format!("{root}/empty");
    store.mkdir(&format!("file://{empty}")).await.unwrap();
    store
        .rmdir(&format!("file://{empty}"), false)
        .await
        .expect("non-recursive rmdir of an empty directory");
    assert!(!std::path::Path::new(&empty).exists());

    // 3. Non-recursive rmdir of a NON-empty directory fails; the row carries the
    //    error and the call returns it.
    let full = format!("{root}/full");
    store.mkdir(&format!("file://{full}")).await.unwrap();
    std::fs::write(format!("{full}/keep.bin"), b"k").unwrap();
    let err = store
        .rmdir(&format!("file://{full}"), false)
        .await
        .expect_err("non-recursive rmdir of a non-empty directory must fail");
    let err_text = err.to_string();
    assert!(
        std::path::Path::new(&format!("{full}/keep.bin")).exists(),
        "the failed rmdir must leave the directory alone"
    );

    // 4. rmdir of a missing directory: the inner store treats NotFound as success.
    let missing = format!("{root}/never_existed");
    store
        .rmdir(&format!("file://{missing}"), true)
        .await
        .expect("rmdir of a missing directory is success");

    // 5. direct:// recursive rmdir. The direct:// store does not implement
    //    mkdir/rmdir (the trait default bails), so the call fails; the logged
    //    store still writes its one row, with the error. The test follows the
    //    call's outcome, so it stays valid if the backend later implements rmdir.
    let direct_dir = format!("{root}/direct_dir");
    std::fs::create_dir_all(format!("{direct_dir}/sub")).unwrap();
    std::fs::write(format!("{direct_dir}/sub/f.bin"), b"d").unwrap();
    let direct = store_for_uri_with_logger(&format!("direct://{root}"), global_logger()).unwrap();
    let direct_result = direct.rmdir(&format!("direct://{direct_dir}"), true).await;
    let direct_err = direct_result.as_ref().err().map(|e| e.to_string());

    // 6. mkdir on both stores writes no row (guard against over-logging).
    store.mkdir(&format!("file://{root}/m1/m2")).await.unwrap();
    let _ = direct.mkdir(&format!("direct://{root}/m3")).await;

    finalize_op_logger();
    let rows = read_rows(&oplog);
    assert_eq!(
        ops(&rows),
        ["DELETE_PREFIX"; 5],
        "exactly one DELETE_PREFIX row per rmdir call and none for mkdir: {rows:#?}"
    );

    // Every row has the delete_prefix shape: bytes 0, count 0, no first byte.
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r[COL_BYTES], "0", "row {i} bytes");
        assert_eq!(r[COL_N_OBJECTS], "0", "row {i} count");
        assert_eq!(r[COL_FIRST_BYTE], "", "row {i} first_byte");
    }

    // Rows are in call order; the file column is the directory itself.
    let expect = [
        ("file://", tree, ""),
        ("file://", empty, ""),
        ("file://", full, err_text.as_str()),
        ("file://", missing, ""),
        ("direct://", direct_dir, direct_err.as_deref().unwrap_or("")),
    ];
    for (i, (endpoint, dir, error)) in expect.iter().enumerate() {
        assert_eq!(&rows[i][COL_ENDPOINT], endpoint, "row {i} endpoint");
        assert_eq!(&rows[i][COL_FILE], dir, "row {i} file (the directory URI)");
        assert_eq!(&rows[i][COL_ERROR], error, "row {i} error column");
    }
    assert!(!err_text.is_empty());
    // The failed non-recursive rmdir is the one row of the file:// cases with an error.
    assert_eq!(
        rows[..4]
            .iter()
            .filter(|r| !r[COL_ERROR].is_empty())
            .count(),
        1
    );
}
