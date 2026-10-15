// tests/mkdir_rmdir_fs_229.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #229: `mkdir` / `rmdir` worked only on `file://`; `direct://` fell through
// to the trait default and failed with "not implemented". `direct://` is the
// `file://` behaviour (O_DIRECT does not apply to directory operations), so the same
// table runs against both schemes and must give identical results:
//
//   mkdir  = create_dir_all (nested, idempotent)
//   rmdir  non-recursive: an empty directory is removed; a non-empty one is an
//          error and NOTHING is removed
//   rmdir  recursive: the whole tree is removed
//   rmdir  of a missing directory: success

use s3dlio::store_for_uri;
use std::path::{Path, PathBuf};

fn scratch(tag: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("mkdir_rmdir_229_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn table(scheme: &str) {
    let root = scratch(scheme);
    let uri = |p: &Path| format!("{scheme}://{}", p.display());
    let store = store_for_uri(&uri(&root)).unwrap();

    // mkdir creates nested directories, and is idempotent.
    let nested = root.join("a/b/c");
    store.mkdir(&uri(&nested)).await.expect("mkdir nested");
    assert!(nested.is_dir(), "{scheme}: nested directory created");
    store.mkdir(&uri(&nested)).await.expect("mkdir twice");

    // Non-recursive rmdir of an EMPTY directory removes exactly that directory.
    store
        .rmdir(&uri(&nested), false)
        .await
        .expect("rmdir empty, non-recursive");
    assert!(!nested.exists(), "{scheme}: empty dir removed");
    assert!(
        root.join("a/b").is_dir(),
        "{scheme}: its parent is untouched"
    );

    // Non-recursive rmdir of a NON-empty directory is an error; nothing is removed.
    let full = root.join("full");
    store.mkdir(&uri(&full.join("sub"))).await.unwrap();
    std::fs::write(full.join("sub/keep.bin"), b"k").unwrap();
    store
        .rmdir(&uri(&full), false)
        .await
        .expect_err("non-recursive rmdir of a non-empty directory must fail");
    assert!(
        full.join("sub/keep.bin").exists(),
        "{scheme}: nothing removed"
    );

    // Recursive rmdir removes the whole tree.
    store
        .rmdir(&uri(&full), true)
        .await
        .expect("recursive rmdir");
    assert!(!full.exists(), "{scheme}: tree gone");

    // A missing directory is success, recursive or not.
    let missing = root.join("never_existed");
    store
        .rmdir(&uri(&missing), true)
        .await
        .expect("missing, recursive");
    store
        .rmdir(&uri(&missing), false)
        .await
        .expect("missing, non-recursive");

    std::fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn file_scheme_mkdir_rmdir_table() {
    table("file").await;
}

#[tokio::test]
async fn direct_scheme_mkdir_rmdir_table() {
    table("direct").await;
}
