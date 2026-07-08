// tests/test_multipart_abort_blocking.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #151 sub-bug 1.2
// (audit finding f11).
//
// Bug: `abort_blocking()` discarded the `AbortMultipartUpload` SDK
// result via `let _ = ...` — no error, no log, nothing. If that request
// failed transiently, the upload_id remained valid server-side. Worse,
// `abort_blocking()` never stopped the coordinator task: just like the
// Drop path (audit f12 / bug A2), the coordinator's receive loop only
// ended when `part_tx` closed, and unconditionally fell through to
// `CompleteMultipartUpload` regardless of whether `abort_blocking()`
// had been called. A caller that explicitly called `.abort()` (Python
// `writer.abort()` or the `__exit__` exception path) could still end up
// with a silently committed partial object.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §3, bug A3): `abort()`/`abort_blocking()` always cancels the
// coordinator first (so Complete can never fire after this call),
// always attempts a best-effort `AbortMultipartUpload`, and always
// returns `Ok(())` to the caller — a transient failure of the *cleanup*
// request doesn't change the fact that the caller asked to abort. A
// `tracing::warn!` records the failure so an operator has a trail to
// find and manually clean up the dangling upload_id.
//
// This fix landed in the same commit as A2 (docs call out A2/A3 as
// coupled, sharing the coordinator-cancellation mechanism), so RED here
// is verified against the pre-A2/A3 tree, not just "before this test
// file existed" — see the commit message for how that was confirmed.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, ABORT_FAILS_MARKER,
};
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
use std::time::Duration;

const TEST_BUCKET: &str = "mock-bucket";
const TEST_PART_SIZE: usize = 5 * 1024 * 1024; // S3 minimum

/// Baseline: abort_blocking() on a healthy mock server must (a) return
/// Ok(()), (b) fire exactly one AbortMultipartUpload, and (c) never let
/// the coordinator reach CompleteMultipartUpload — INCLUDING after the
/// sink is subsequently dropped, which is what every real caller does
/// immediately afterward (Python's `abort()` / `__exit__` both `.take()`
/// the inner sink into a local that drops at the end of the binding
/// function). In the pre-fix design, `abort_blocking()` alone never
/// touched `part_tx`, so the coordinator stayed parked on `recv()`
/// indefinitely until that subsequent drop closed the channel — at
/// which point it fell through to Complete unconditionally. A check
/// that never drops the sink can't observe that bug at all, so this
/// test deliberately drops it right after calling `abort_blocking()`.
#[test]
fn abort_blocking_succeeds_and_never_completes() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_test_key("mpu-abort-blocking-ok");
    let path = key_path(TEST_BUCKET, &key);

    let cfg = MultipartUploadConfig {
        part_size: TEST_PART_SIZE,
        abort_on_drop: true,
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(TEST_BUCKET, &key, cfg)
        .expect("CreateMultipartUpload against mock server should succeed");

    sink.write_blocking(&vec![0xEEu8; TEST_PART_SIZE]).unwrap();
    sink.write_blocking(&vec![0xFFu8; TEST_PART_SIZE]).unwrap();

    let result = sink.abort_blocking();
    assert!(
        result.is_ok(),
        "abort_blocking() must return Ok(()) on a successful abort, got: {result:?}"
    );

    // Matches real Python usage: the sink is dropped immediately after
    // abort_blocking() returns.
    drop(sink);

    let snap = state.wait_for(&path, Duration::from_millis(400), |c| c.complete_calls > 0);
    assert_eq!(
        snap.complete_calls, 0,
        "CompleteMultipartUpload must never fire after abort_blocking() + drop — observed {} \
         calls. abort_blocking() must cancel the coordinator itself, not rely on a later drop \
         to close the channel (which pre-fix let the coordinator fall through to Complete).",
        snap.complete_calls
    );
    assert_eq!(
        snap.abort_calls, 1,
        "abort_blocking() must fire exactly one AbortMultipartUpload, and the subsequent drop \
         must not fire a second one (idempotent via `finished`) — observed {}",
        snap.abort_calls
    );
}

/// The mock server returns a 500 for AbortMultipartUpload on this key
/// (transient-failure simulation). abort_blocking() must still: (a)
/// return Ok(()) to the caller (a failed cleanup request doesn't
/// invalidate the caller's abort intent), and (b) never let the
/// coordinator reach Complete regardless of whether the S3-side Abort
/// itself succeeded.
#[test]
fn abort_blocking_returns_ok_and_never_completes_even_when_s3_abort_fails() {
    let (_, state) = ensure_mock_s3_server();
    let key = unique_test_key(&format!("mpu-{ABORT_FAILS_MARKER}"));
    let path = key_path(TEST_BUCKET, &key);

    let cfg = MultipartUploadConfig {
        part_size: TEST_PART_SIZE,
        abort_on_drop: true,
        ..Default::default()
    };
    let mut sink = MultipartUploadSink::new(TEST_BUCKET, &key, cfg)
        .expect("CreateMultipartUpload against mock server should succeed");

    sink.write_blocking(&vec![0x11u8; TEST_PART_SIZE]).unwrap();

    let result = sink.abort_blocking();
    assert!(
        result.is_ok(),
        "abort_blocking() must return Ok(()) even when the underlying AbortMultipartUpload \
         request fails — the caller's abort intent stands regardless of cleanup-call outcome. \
         Got: {result:?}"
    );

    // Matches real Python usage: the sink drops right after abort_blocking().
    drop(sink);

    let snap = state.wait_for(&path, Duration::from_millis(400), |c| c.complete_calls > 0);
    assert_eq!(
        snap.complete_calls, 0,
        "CompleteMultipartUpload must never fire, even when the best-effort Abort itself \
         failed server-side — observed {} calls. A failed cleanup request must not leave \
         the coordinator free to race to Complete.",
        snap.complete_calls
    );
    assert!(
        snap.abort_calls >= 1,
        "abort_blocking() must still attempt AbortMultipartUpload even though the mock \
         server will reject it, observed {} attempts",
        snap.abort_calls
    );
    assert!(
        snap.abort_failures_returned >= 1,
        "test setup sanity check: the mock server should have returned at least one \
         failure response for this marked key, observed {}",
        snap.abort_failures_returned
    );
}
