// tests/test_multipart_abort_on_drop_false.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #151 sub-bug 1.1
// (audit finding f12).
//
// Bug: dropping `MultipartUploadSink` without calling `finish()`/`abort()`
// always caused the coordinator task to `CompleteMultipartUpload` with
// whatever parts had been sent so far — silently committing a truncated
// object with a valid ETag — REGARDLESS of the `abort_on_drop` config
// flag. The custom `Drop` impl's own best-effort `AbortMultipartUpload`
// call ran concurrently with (and had no synchronization against) the
// coordinator's unconditional path to Complete once the channel closed,
// because `part_tx` (a struct field) drops before `coordinator` in
// Rust's declaration-order field drop, and dropping a `JoinHandle`
// detaches the task rather than cancelling it.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §3, bug A2): best-effort abort on drop, regardless of `abort_on_drop`.
// Concretely: Drop always cancels the coordinator first (so it can never
// reach Complete), then — only when `abort_on_drop=true` (the default)
// — fires a best-effort `AbortMultipartUpload`. `abort_on_drop=false`
// cancels the coordinator but skips the Abort call, so the operator
// explicitly opts into a dangling upload rather than a silently
// committed partial one.

// tests/common/mod.rs is shared across many test binaries; this one only
// uses mock_s3_server, so rustc sees the GCS-test helpers (TestConfig,
// get_test_config, etc.) as dead code in this specific binary. Matches
// the existing precedent in tests/test_gcs_official.rs.
#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, key_path, unique_test_key};
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
use std::time::Duration;

const TEST_BUCKET: &str = "mock-bucket";
const TEST_PART_SIZE: usize = 5 * 1024 * 1024; // S3 minimum

/// abort_on_drop=true (the default): dropping an unfinished sink must
/// NEVER call CompleteMultipartUpload, no matter what.
#[test]
fn abort_on_drop_true_never_completes_the_upload() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_test_key("mpu-adt-true");
    let path = key_path(TEST_BUCKET, &key);

    let cfg = MultipartUploadConfig {
        part_size: TEST_PART_SIZE,
        abort_on_drop: true,
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(TEST_BUCKET, &key, cfg)
        .expect("CreateMultipartUpload against mock server should succeed");

    // Two full parts, no finish() / abort() call.
    sink.write_blocking(&vec![0xAAu8; TEST_PART_SIZE]).unwrap();
    sink.write_blocking(&vec![0xBBu8; TEST_PART_SIZE]).unwrap();

    drop(sink);

    // Give the detached coordinator task (which, pre-fix, keeps running
    // after part_tx closes) a generous window to reach Complete if it's
    // going to. On a fixed tree nothing should ever increment
    // complete_calls, so waiting out the full timeout is expected there.
    let snap = state.wait_for(&path, Duration::from_millis(400), |c| c.complete_calls > 0);

    assert_eq!(
        snap.complete_calls, 0,
        "CompleteMultipartUpload must never be called for a dropped, unfinished sink \
         — observed {} calls. This is the silent-partial-commit bug (audit f12): dropping \
         part_tx lets the coordinator's recv() loop exit and fall through to Complete \
         regardless of abort_on_drop.",
        snap.complete_calls
    );
    assert!(
        snap.abort_calls >= 1,
        "abort_on_drop=true (default) must fire a best-effort AbortMultipartUpload on drop, \
         observed {} abort calls",
        snap.abort_calls
    );
}

/// abort_on_drop=false: dropping an unfinished sink must still NEVER call
/// CompleteMultipartUpload (that's the actual bug), and per the locked
/// contract must also skip the best-effort Abort — the operator
/// explicitly opted into leaving the upload dangling.
#[test]
fn abort_on_drop_false_never_completes_and_skips_abort() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_test_key("mpu-adt-false");
    let path = key_path(TEST_BUCKET, &key);

    let cfg = MultipartUploadConfig {
        part_size: TEST_PART_SIZE,
        abort_on_drop: false,
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(TEST_BUCKET, &key, cfg)
        .expect("CreateMultipartUpload against mock server should succeed");

    sink.write_blocking(&vec![0xCCu8; TEST_PART_SIZE]).unwrap();
    sink.write_blocking(&vec![0xDDu8; TEST_PART_SIZE]).unwrap();

    drop(sink);

    let snap = state.wait_for(&path, Duration::from_millis(400), |c| c.complete_calls > 0);

    assert_eq!(
        snap.complete_calls, 0,
        "abort_on_drop=false must not silently commit a partial object either — the flag \
         controls only whether the best-effort Abort fires, not whether Complete can slip \
         through. Observed {} Complete calls.",
        snap.complete_calls
    );
    assert_eq!(
        snap.abort_calls, 0,
        "abort_on_drop=false must skip the best-effort AbortMultipartUpload — the operator \
         explicitly chose to leave the upload dangling. Observed {} abort calls.",
        snap.abort_calls
    );
}
