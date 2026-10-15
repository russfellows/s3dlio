// tests/common/odirect.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Scratch directory for direct:// (O_DIRECT) tests. O_DIRECT is unsupported on tmpfs, so the
// directory is S3DLIO_TEST_DIR (a real filesystem) or the cargo target tmp dir; when the chosen
// filesystem refuses O_DIRECT the test is SKIPPED with a clear message (not an unwrap panic).

#![allow(dead_code)]

use std::os::unix::fs::OpenOptionsExt;

/// A scratch directory that supports O_DIRECT, or `None` (after printing why) so the caller
/// can return early.
pub fn odirect_scratch_dir(prefix: &str) -> Option<tempfile::TempDir> {
    let base = std::env::var_os("S3DLIO_TEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_TARGET_TMPDIR").into());
    std::fs::create_dir_all(&base).expect("create the test base directory");
    let dir = tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(base)
        .expect("temp dir");
    let probe = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(dir.path().join("o_direct_probe"));
    if let Err(e) = probe {
        eprintln!(
            "SKIPPED: {} does not support O_DIRECT ({e}); set S3DLIO_TEST_DIR to a directory \
             on a real filesystem (not tmpfs)",
            dir.path().display()
        );
        return None;
    }
    Some(dir)
}
