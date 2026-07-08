// tests/test_range_get_zero_length_underflow.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression test for s3dlio issue #152 findings 2.5/2.6/2.7
// (bug B4) — S3 site.
//
// Bug: `S3ObjectStore::get_range()` computed the range header's inclusive
// end as `offset + len - 1` directly. With `length=Some(0)` (a legitimate
// "read zero bytes" request — distinct from `length=None`, which means
// "read to EOF"), `len - 1` underflows: in a debug/test build (overflow
// checks on) this panics; in a release build it silently wraps to
// `u64::MAX`, producing a nonsensical `Range: bytes={offset}-{huge
// number}` header sent to S3.
//
// This is a DIFFERENT code path from bug B2 (issue #152 sub-bug 2.3,
// tests/test_range_length_zero.rs), which fixed the same class of bug in
// `s3_utils::get_object_range_uri_async`. `S3ObjectStore::get_range` (used
// by the generic `ObjectStore` trait, e.g. via `store_for_uri`) builds its
// own range header independently and had the same underflow.
//
// Locked contract (docs/implementation-plans/v0.9.109-audit-fix-plan.md
// §4, bug B4): `length=Some(0)` short-circuits to `Ok(Bytes::new())` with
// no network call, matching B2's precedent. For `length=Some(len>0)`, the
// inclusive end is computed via the shared, overflow-checked
// `range_engine_generic::range_end_inclusive` helper instead of bare
// arithmetic. The same helper is used by the Azure and community-GCS
// `get_range()` implementations (object_store.rs, gcs_client.rs) — see
// that helper's own unit tests in src/range_engine_generic.rs for direct
// coverage of the underflow/overflow arithmetic itself. Azure and GCS
// aren't covered by a live-network integration test here (no credentials
// or emulators available in this environment); their correctness rests on
// sharing this same audited helper, verified to compile via
// `cargo check --features backend-azure,gcs-community`.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::ensure_mock_s3_server;
use s3dlio::object_store::{ObjectStore, S3ObjectStore};

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

#[tokio::test]
async fn get_range_length_zero_returns_empty_bytes_instead_of_underflowing() {
    let (port, _) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let key = unique_key("b4-s3-range-zero");
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = store.get_range(&uri, 100, Some(0)).await.expect(
        "length=Some(0) must succeed with an empty buffer, not panic on \
             `offset + len - 1` underflow (audit #152 finding 2.5 / bug B4)",
    );

    assert_eq!(
        bytes.len(),
        0,
        "length=Some(0) must return zero bytes, not the huge/garbage range \
         a wrapped `offset + 0 - 1` would have requested"
    );
}

/// `offset + len - 1` parses as `(offset + len) - 1`, so the underflow only
/// bites at `offset=0, length=Some(0)` (a genuine u64 underflow panic in a
/// debug/test build); at nonzero offsets it instead silently produces an
/// inverted, malformed range like `bytes=100-99` (end < start) rather than
/// panicking. Both are the same root bug (audit #152 finding 2.5) and both
/// are covered here.
#[tokio::test]
async fn get_range_length_zero_at_offset_zero_does_not_underflow_panic() {
    let (port, _) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let key = unique_key("b4-s3-range-zero-offset-zero");
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = store.get_range(&uri, 0, Some(0)).await.expect(
        "offset=0, length=Some(0) must succeed with an empty buffer, not panic on \
             `(offset + len) - 1` == `0 - 1` underflow (audit #152 finding 2.5 / bug B4)",
    );
    assert_eq!(bytes.len(), 0);
}

/// Sanity check: a real nonzero length still round-trips correctly through
/// the fixed `range_end_inclusive`-based header construction.
#[tokio::test]
async fn get_range_nonzero_length_still_works() {
    let (port, _) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let key = unique_key("b4-s3-range-nonzero");
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = store
        .get_range(&uri, 0, Some(4))
        .await
        .expect("a normal ranged GET should succeed against the mock server");
    assert_eq!(&bytes[..], b"mock");
}

/// Sanity check: the open-ended `length=None` ("read to EOF") case is
/// untouched by this fix — it never computed an inclusive end at all.
#[tokio::test]
async fn get_range_length_none_still_open_ended() {
    let (port, _) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint should succeed against the mock server");

    let key = unique_key("b4-s3-range-none");
    let uri = format!("s3://mock-bucket/{key}");

    let bytes = store
        .get_range(&uri, 0, None)
        .await
        .expect("an open-ended ranged GET should succeed against the mock server");
    assert_eq!(&bytes[..], b"mock");
}
