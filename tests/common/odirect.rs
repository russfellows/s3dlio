// tests/common/odirect.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Scratch directory for direct:// (O_DIRECT) tests. O_DIRECT is unsupported on tmpfs, so the
// directory is S3DLIO_TEST_DIR (a real filesystem) or the cargo target tmp dir; when the chosen
// filesystem refuses O_DIRECT the test is SKIPPED with a clear message (not an unwrap panic),
// UNLESS S3DLIO_REQUIRE_ODIRECT_TESTS=1 (CI sets it): then a skip is a FAILURE, so a runner
// whose scratch directory cannot do O_DIRECT cannot quietly turn the suite into a no-op.

#![allow(dead_code)]

use std::os::unix::fs::OpenOptionsExt;

/// True when a skipped test must fail instead (`S3DLIO_REQUIRE_ODIRECT_TESTS` is 1/true/yes).
pub fn skips_are_failures() -> bool {
    std::env::var("S3DLIO_REQUIRE_ODIRECT_TESTS")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Report that a test is skipped, or panic when skips are failures. Call, then return.
pub fn skip(reason: &str) {
    eprintln!("SKIPPED: {reason}");
    assert!(
        !skips_are_failures(),
        "S3DLIO_REQUIRE_ODIRECT_TESTS is set, so a skipped test is a failure: {reason}"
    );
}

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
        skip(&format!(
            "{} does not support O_DIRECT ({e}); set S3DLIO_TEST_DIR to a directory \
             on a real filesystem (not tmpfs)",
            dir.path().display()
        ));
        return None;
    }
    Some(dir)
}
