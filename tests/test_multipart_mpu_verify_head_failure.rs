// tests/test_multipart_mpu_verify_head_failure.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #151 bug 1.3 (D3).
//
// Bug: with S3DLIO_MPU_PUT_VERIFY=true, `finish()`'s post-Complete HEAD
// verification step propagated ANY HeadObject failure as a hard error via
// `.sdk_context(head_ctx)?` — including a transient failure of the HEAD
// *request itself* (network blip, throttling), which has nothing to do with
// whether the just-completed upload actually landed correctly.
// CompleteMultipartUpload had ALREADY succeeded (with a real ETag) by the
// time HEAD is issued, so treating a failed verification *call* as proof of
// data corruption was wrong — worse, it wasted a fully successful upload the
// caller must now redo.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §6, bug D3): "warn and Ok" -- HEAD request failures are soft. Only a HEAD
// that *succeeds* but reports a different size than what was written is
// treated as real corruption (unchanged: still deletes the object and bails).

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, HEAD_FAILS_MARKER};
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};

const TEST_BUCKET: &str = "mock-bucket";
const TEST_PART_SIZE: usize = 5 * 1024 * 1024; // S3 minimum

/// Serializes the two tests in this file, both of which mutate the
/// process-wide `S3DLIO_MPU_PUT_VERIFY` env var. `cargo test` runs tests
/// within one binary concurrently by default; without this guard, one
/// test's `remove_var` can race another's in-flight `finish_blocking()`
/// (which reads the env var from inside its async coordinator task), so
/// the coordinator observes verification as disabled and never issues a
/// HEAD at all -- a flaky, not a real, test failure. Same pattern as
/// `google_gcs_client.rs`'s `ENV_MUTEX`.
static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn unique_key(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{prefix}-{nanos}-{n}")
}

/// A failed HEAD *request* (not a successful HEAD reporting a mismatched
/// size) must not fail finish() — CompleteMultipartUpload already
/// succeeded, so the upload is presumed correct and `stored_bytes` falls
/// back to `total_bytes`, exactly like the verify-disabled default.
#[test]
fn head_request_failure_does_not_fail_finish() {
    let _guard = ENV_MUTEX.lock().unwrap();
    let (_, state) = ensure_mock_s3_server();
    std::env::set_var("S3DLIO_MPU_PUT_VERIFY", "true");

    let key = unique_key(&format!("mpu-{HEAD_FAILS_MARKER}"));
    let cfg = MultipartUploadConfig {
        part_size: TEST_PART_SIZE,
        abort_on_drop: true,
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(TEST_BUCKET, &key, cfg)
        .expect("CreateMultipartUpload should succeed");

    let data = vec![0xCDu8; TEST_PART_SIZE];
    sink.write_blocking(&data).unwrap();

    let result = sink.finish_blocking();
    std::env::remove_var("S3DLIO_MPU_PUT_VERIFY");

    let info = result.expect(
        "a failed HEAD *request* must not fail finish() -- CompleteMultipartUpload already \
         succeeded, so a flaky verification call is not evidence of data corruption (audit \
         #151 bug 1.3 / D3)",
    );
    assert_eq!(
        info.stored_bytes, info.total_bytes,
        "when HEAD itself fails, stored_bytes must fall back to total_bytes (unverified), \
         matching the verify-disabled default"
    );

    let path = common::mock_s3_server::key_path(TEST_BUCKET, &key);
    let snap = state.counts_for(&path);
    assert!(
        snap.head_calls >= 1,
        "the HEAD request must actually have been attempted, not skipped"
    );
}

/// Sanity check: a HEAD that SUCCEEDS but reports a genuinely different
/// size must still be treated as real corruption -- this fix must not
/// weaken that path. Uses the existing NO_CONTENT_LENGTH_MARKER-style
/// mismatch: HEAD succeeds with Content-Length: 0 (the mock server's
/// unmarked default), which never equals a nonzero total_bytes.
#[test]
fn head_success_with_size_mismatch_still_fails_finish() {
    let _guard = ENV_MUTEX.lock().unwrap();
    let (_, state) = ensure_mock_s3_server();
    std::env::set_var("S3DLIO_MPU_PUT_VERIFY", "true");

    let key = unique_key("mpu-size-mismatch");
    let cfg = MultipartUploadConfig {
        part_size: TEST_PART_SIZE,
        abort_on_drop: true,
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(TEST_BUCKET, &key, cfg)
        .expect("CreateMultipartUpload should succeed");

    let data = vec![0xCDu8; TEST_PART_SIZE];
    sink.write_blocking(&data).unwrap();

    let result = sink.finish_blocking();
    std::env::remove_var("S3DLIO_MPU_PUT_VERIFY");

    assert!(
        result.is_err(),
        "a HEAD that succeeds but reports a mismatched size must still be treated as real \
         corruption -- the D3 fix must only soften a failed HEAD *request*, not a genuine \
         size mismatch"
    );

    let path = common::mock_s3_server::key_path(TEST_BUCKET, &key);
    let snap = state.counts_for(&path);
    assert!(
        snap.head_calls >= 1,
        "HEAD must have been attempted for the mismatch to be detected"
    );
}
