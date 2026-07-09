// tests/test_delete_objects_errors.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #154 sub-bug 4.5
// (audit finding f36).
//
// Bug: S3's `DeleteObjects` batch API returns HTTP 200 with a body
// containing per-object `<Error>` entries for keys that could not be
// deleted (ACL denial, object-lock retention, throttling, etc.) —
// `Deleted=[]`/partial is a valid 200 response. Both s3dlio call sites
// (`S3Ops::delete_objects` in src/s3_ops.rs and the standalone
// `delete_objects_async` in src/s3_utils.rs) matched on `Ok(_)` /
// discarded the response outright and returned `Ok(())` unconditionally
// — a caller had no way to learn that some or all of a batch silently
// failed to delete.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §3, bug A4): keep the `Result<()>` return shape (no API break this
// release). Inspect `DeleteObjectsOutput.errors()`; if non-empty, return
// `Err` naming the failed keys and their S3 error codes. The richer
// `{deleted, errors}` struct return is explicitly deferred to a later
// release.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::{ensure_mock_s3_server, DELETE_FAILS_MARKER};
use s3dlio::s3_client::{aws_s3_client_async, run_on_global_rt};
use s3dlio::s3_ops::S3Ops;
use s3dlio::s3_utils::delete_objects_async;

const TEST_BUCKET: &str = "mock-bucket";

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

// ---------------------------------------------------------------------------
// S3Ops::delete_objects (src/s3_ops.rs)
// ---------------------------------------------------------------------------

#[test]
fn s3ops_delete_objects_returns_ok_when_all_keys_succeed() {
    ensure_mock_s3_server();
    let keys = vec![unique_key("d-ok-a"), unique_key("d-ok-b")];

    let result = run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        let ops = S3Ops::new(client, None, "test-client", "mock-endpoint");
        ops.delete_objects(TEST_BUCKET, keys).await
    });

    assert!(
        result.is_ok(),
        "delete_objects must return Ok(()) when every key succeeds, got: {result:?}"
    );
}

#[test]
fn s3ops_delete_objects_returns_err_when_a_key_partially_fails() {
    ensure_mock_s3_server();
    let failing_key = unique_key(&format!("d-{DELETE_FAILS_MARKER}"));
    let keys = vec![unique_key("d-ok"), failing_key.clone()];

    let result = run_on_global_rt(async move {
        let client = aws_s3_client_async().await?;
        let ops = S3Ops::new(client, None, "test-client", "mock-endpoint");
        ops.delete_objects(TEST_BUCKET, keys).await
    });

    let err = result.expect_err(
        "delete_objects must return Err when the response body's <Error> list is non-empty, \
         even though the HTTP status was 200 — this is the exact silent-partial-failure bug \
         (audit f36): DeleteObjects returning 200 does not mean every key was deleted.",
    );
    let msg = err.to_string();
    assert!(
        msg.contains(&failing_key) || msg.contains("AccessDenied"),
        "error message should name the failed key or S3 error code, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// delete_objects_async (src/s3_utils.rs) — independent implementation,
// same bug class, same fix.
// ---------------------------------------------------------------------------

#[test]
fn delete_objects_async_returns_ok_when_all_keys_succeed() {
    ensure_mock_s3_server();
    let keys = vec![unique_key("da-ok-a"), unique_key("da-ok-b")];

    let result = run_on_global_rt(async move { delete_objects_async(TEST_BUCKET, &keys).await });

    assert!(
        result.is_ok(),
        "delete_objects_async must return Ok(()) when every key succeeds, got: {result:?}"
    );
}

#[test]
fn delete_objects_async_returns_err_when_a_key_partially_fails() {
    ensure_mock_s3_server();
    let failing_key = unique_key(&format!("da-{DELETE_FAILS_MARKER}"));
    let keys = vec![unique_key("da-ok"), failing_key.clone()];

    let result = run_on_global_rt(async move { delete_objects_async(TEST_BUCKET, &keys).await });

    let err = result.expect_err(
        "delete_objects_async must return Err when the response body's <Error> list is \
         non-empty, even though the HTTP status was 200 (audit f36).",
    );
    let msg = err.to_string();
    assert!(
        msg.contains(&failing_key) || msg.contains("AccessDenied"),
        "error message should name the failed key or S3 error code, got: {msg}"
    );
}
