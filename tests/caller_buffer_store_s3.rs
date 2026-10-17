// tests/caller_buffer_store_s3.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// `ObjectStore::get_into` and `ObjectStore::head` on s3:// URIs, reached the way a caller reaches
// them: through `store_for_uri` (v0.10.0 plan §4.5 A1, A6, A8; #264). The provider's own
// behaviour (ranges, ETag pinning, retries) has its tests in `tests/s3_get_into_*.rs`; this file
// proves the STORE wiring:
//
//   * a bucket-form URI reaches the process-wide client (the `AWS_ENDPOINT_URL` mock) and a
//     host-form URI reaches its OWN per-endpoint client and no other server;
//   * the bytes land in the caller's slice, nothing is written past it, and `progress` counts;
//   * failures are typed (`NotFound`, `LengthMismatch`, `Forbidden`) and HEAD reports absence
//     as `Ok(None)` and nothing else as absence;
//   * a URI naming another server is refused before any request (#197).
//
// Local only: two in-process mock S3 servers. The read threshold is 1 MiB and the chunk 256 KiB
// so a small object is one GET and a larger one is the pinned range fan-out.

#![cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use s3_get_into_support::mock::{start_isolated_mock_s3_server, MockObject};
use s3_get_into_support::*;
use s3dlio::object_error::ObjectErrorKind;
use s3dlio::object_store::{store_for_uri, ObjectStore};
use s3dlio::receive_progress::ReceiveProgress;

const MIB: usize = 1024 * 1024;

fn setup() -> (u16, s3_get_into_support::mock::MockS3State) {
    init(&[
        ("S3DLIO_RANGE_THRESHOLD_MB", "1"),
        ("S3DLIO_CHUNK_SIZE", "262144"),
        ("S3DLIO_RANGE_CONCURRENCY", "2"),
    ])
}

fn store(uri: &str) -> Box<dyn ObjectStore> {
    store_for_uri(uri).unwrap_or_else(|e| panic!("store for {uri}: {e:#}"))
}

// ---- bucket-form URIs: the process-wide client ---------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn global_client_store_reads_into_the_callers_slice() {
    let (_port, state) = setup();
    let key = key("global-small");
    let body = pattern(700, 1);
    state.put_object(BUCKET, &key, MockObject::new(body.clone()));
    let uri = format!("s3://{BUCKET}/{key}");

    let mut dst = Dst::new(700);
    let progress = ReceiveProgress::new();
    store(&uri)
        .get_into(&uri, dst.slice(), &progress)
        .await
        .expect("get_into");

    assert_eq!(
        dst.bytes(),
        &body[..],
        "the bytes are in the caller's slice"
    );
    dst.assert_canaries();
    assert_eq!(progress.filled(), 700);
    assert!(progress.first_byte().is_some());
    assert_eq!(gets(&state, &key).len(), 1, "one GET reached the mock");
    assert_eq!(heads(&state, &key), 0, "a small read needs no HEAD");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn global_client_store_reaches_the_pinned_range_fan_out() {
    let (_port, state) = setup();
    let key = key("global-fanout");
    let body = pattern(MIB + 17, 2);
    state.put_object(BUCKET, &key, MockObject::new(body.clone()));
    let uri = format!("s3://{BUCKET}/{key}");

    let mut dst = Dst::new(body.len());
    let progress = ReceiveProgress::new();
    store(&uri)
        .get_into(&uri, dst.slice(), &progress)
        .await
        .expect("get_into");

    assert!(
        dst.bytes() == &body[..],
        "the bytes are in the caller's slice"
    );
    dst.assert_canaries();
    assert_eq!(progress.filled(), body.len() as u64);
    assert_eq!(heads(&state, &key), 1, "the fan-out sizes the object once");
    let ranged = gets(&state, &key);
    assert!(ranged.len() >= 5, "ranged GETs: {ranged:?}");
    assert!(
        ranged
            .iter()
            .all(|r| r.range.is_some() && r.if_match.is_some()),
        "every range is bounded and pinned to the ETag: {ranged:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_length_that_differs_from_the_object_is_a_typed_length_mismatch() {
    let (_port, state) = setup();
    let key = key("global-length");
    state.put_object(BUCKET, &key, MockObject::new(pattern(100, 3)));
    let uri = format!("s3://{BUCKET}/{key}");
    let store = store(&uri);

    for len in [99usize, 101] {
        let mut dst = Dst::new(len);
        let err = store
            .get_into(&uri, dst.slice(), &ReceiveProgress::new())
            .await
            .expect_err("a wrong expected length is an error");
        assert_eq!(
            kind_of(&err),
            ObjectErrorKind::LengthMismatch {
                expected: len as u64,
                actual: Some(100)
            },
            "{len}"
        );
        dst.assert_canaries();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_absent_object_is_a_typed_not_found_and_leaves_the_slice_alone() {
    let (_port, state) = setup();
    let key = key("global-absent");
    state.put_object(BUCKET, &key, MockObject::missing());
    let uri = format!("s3://{BUCKET}/{key}");

    let mut dst = Dst::new(32);
    let progress = ReceiveProgress::new();
    let err = store(&uri)
        .get_into(&uri, dst.slice(), &progress)
        .await
        .expect_err("absent");
    assert_eq!(kind_of(&err), ObjectErrorKind::NotFound);
    dst.assert_untouched();
    assert_eq!(progress.filled(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_reports_size_and_identity_absence_as_none_and_a_refusal_as_an_error() {
    let (_port, state) = setup();
    let present = key("head-present");
    let body = pattern(321, 4);
    state.put_object(BUCKET, &present, MockObject::new(body.clone()));
    let absent = key("head-absent");
    state.put_object(BUCKET, &absent, MockObject::missing());
    let refused = key("head-refused");
    let mut object = MockObject::new(pattern(8, 5));
    object.head_status = Some(403);
    state.put_object(BUCKET, &refused, object);

    let uri = |k: &str| format!("s3://{BUCKET}/{k}");
    let store = store(&uri(&present));

    let meta = store
        .head(&uri(&present))
        .await
        .expect("head")
        .expect("present");
    assert_eq!(meta.size, 321);
    assert_eq!(
        meta.etag.as_deref(),
        Some(s3_get_into_support::mock::etag_for(&body).as_str())
    );

    assert!(
        store.head(&uri(&absent)).await.expect("head").is_none(),
        "typed absence is Ok(None)"
    );

    let err = store
        .head(&uri(&refused))
        .await
        .expect_err("a refusal is never absence");
    assert_eq!(kind_of(&err), ObjectErrorKind::Forbidden);
}

// ---- host-form URIs: a per-endpoint client and no other server --------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn host_form_store_reaches_its_own_endpoint_and_never_the_global_one() {
    let (_global_port, global) = setup();
    let (port, endpoint) = start_isolated_mock_s3_server();
    let key = key("endpoint");
    let body = pattern(500, 6);
    endpoint.put_object(BUCKET, &key, MockObject::new(body.clone()));
    let big = key.clone() + "-big";
    let big_body = pattern(MIB + 5, 7);
    endpoint.put_object(BUCKET, &big, MockObject::new(big_body.clone()));
    let absent = key.clone() + "-absent";
    endpoint.put_object(BUCKET, &absent, MockObject::missing());

    let uri = |k: &str| format!("s3://127.0.0.1:{port}/{BUCKET}/{k}");
    let store = store(&uri(&key));

    // One GET.
    let mut dst = Dst::new(500);
    let progress = ReceiveProgress::new();
    store
        .get_into(&uri(&key), dst.slice(), &progress)
        .await
        .expect("get_into");
    assert_eq!(dst.bytes(), &body[..]);
    dst.assert_canaries();
    assert_eq!(progress.filled(), 500);
    assert_eq!(gets(&endpoint, &key).len(), 1);

    // The ranged fan-out, on the same per-endpoint client.
    let mut dst = Dst::new(big_body.len());
    store
        .get_into(&uri(&big), dst.slice(), &ReceiveProgress::new())
        .await
        .expect("get_into");
    assert!(dst.bytes() == &big_body[..]);
    dst.assert_canaries();
    assert!(gets(&endpoint, &big).len() >= 5);

    // HEAD: present and absent.
    let meta = store
        .head(&uri(&key))
        .await
        .expect("head")
        .expect("present");
    assert_eq!(meta.size, 500);
    assert!(store.head(&uri(&absent)).await.expect("head").is_none());

    // Typed absence on GET.
    let err = store
        .get_into(&uri(&absent), Dst::new(8).slice(), &ReceiveProgress::new())
        .await
        .expect_err("absent");
    assert_eq!(kind_of(&err), ObjectErrorKind::NotFound);

    // The process-wide (AWS_ENDPOINT_URL) server saw none of it.
    for k in [&key, &big, &absent] {
        assert!(
            global.requests_for(BUCKET, k).is_empty(),
            "the global client was used for {k}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_uri_naming_another_server_is_refused_before_any_request() {
    let (global_port, global) = setup();
    let (port, endpoint) = start_isolated_mock_s3_server();
    let key = key("foreign");
    let body = pattern(64, 8);
    global.put_object(BUCKET, &key, MockObject::new(body.clone()));
    endpoint.put_object(BUCKET, &key, MockObject::new(body));

    // A per-endpoint store given a URI that names the OTHER server.
    let own = format!("s3://127.0.0.1:{port}/{BUCKET}/{key}");
    let other = format!("s3://127.0.0.1:{global_port}/{BUCKET}/{key}");
    let endpoint_store = store(&own);
    let mut dst = Dst::new(64);
    endpoint_store
        .get_into(&other, dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("a foreign server must be refused");
    endpoint_store
        .head(&other)
        .await
        .expect_err("a foreign server must be refused");
    dst.assert_untouched();

    // A store on the process-wide client given a URI that names a decoy server.
    let bucket_form = format!("s3://{BUCKET}/{key}");
    let decoy = format!("s3://127.0.0.1:9/{BUCKET}/{key}");
    let global_store = store(&bucket_form);
    let err = global_store
        .get_into(&decoy, Dst::new(64).slice(), &ReceiveProgress::new())
        .await
        .expect_err("a decoy server must be refused");
    assert!(
        format!("{err:#}").contains("127.0.0.1:9"),
        "the error names the URI's server: {err:#}"
    );
    global_store
        .head(&decoy)
        .await
        .expect_err("a decoy server must be refused");

    assert!(global.requests_for(BUCKET, &key).is_empty());
    assert!(endpoint.requests_for(BUCKET, &key).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bucket_or_empty_key_is_an_error_never_an_object() {
    setup();
    let store = store(&format!("s3://{BUCKET}/x"));
    for uri in [format!("s3://{BUCKET}"), format!("s3://{BUCKET}/")] {
        store
            .get_into(&uri, Dst::new(4).slice(), &ReceiveProgress::new())
            .await
            .expect_err(&uri);
        store.head(&uri).await.expect_err(&uri);
    }
}
