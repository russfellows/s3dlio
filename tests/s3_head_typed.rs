// tests/s3_head_typed.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// S3 typed HEAD (plan §4.3 "S3", §4.5 A8, §9): the size and the opaque identity of an object,
// `Ok(None)` ONLY for typed absence, and an error for every other failure (a refusal, a server
// failure, a missing size). Local mock server only.

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use s3_get_into_support::mock::MockObject;
use s3_get_into_support::*;
use s3dlio::object_error::ObjectErrorKind;

async fn setup() -> (
    s3dlio::s3_ops::S3Ops,
    s3_get_into_support::mock::MockS3State,
) {
    let (port, state) = init(&[]);
    (ops(client_for(port).await), state)
}

#[tokio::test]
async fn an_existing_object_reports_its_size_etag_and_version() {
    let (ops, state) = setup().await;
    let key = key("meta");
    let mut object = MockObject::new(pattern(21, 1));
    object.version_id = Some("v-opaque-1".to_string());
    let etag = object.etag.clone().expect("etag");
    state.put_object(BUCKET, &key, object);

    let meta = ops
        .head_typed(BUCKET, &key)
        .await
        .expect("head")
        .expect("present");

    assert_eq!(meta.size, 21);
    assert_eq!(
        meta.etag.as_deref(),
        Some(etag.as_str()),
        "the ETag exactly as sent"
    );
    assert_eq!(meta.version_id.as_deref(), Some("v-opaque-1"));
}

#[tokio::test]
async fn the_etag_is_opaque_and_may_be_absent() {
    let (ops, state) = setup().await;
    let weak = key("weak");
    let mut object = MockObject::new(pattern(4, 2));
    object.etag = Some("W/\"not-an-md5\"".to_string());
    state.put_object(BUCKET, &weak, object);
    let meta = ops.head_typed(BUCKET, &weak).await.unwrap().unwrap();
    assert_eq!(meta.etag.as_deref(), Some("W/\"not-an-md5\""));

    let none = key("no-etag");
    let mut object = MockObject::new(pattern(4, 3));
    object.etag = None;
    state.put_object(BUCKET, &none, object);
    let meta = ops.head_typed(BUCKET, &none).await.unwrap().unwrap();
    assert_eq!((meta.size, meta.etag, meta.version_id), (4, None, None));
}

#[tokio::test]
async fn an_empty_object_is_present_with_size_zero() {
    let (ops, state) = setup().await;
    let key = key("empty");
    state.put_object(BUCKET, &key, MockObject::new(Vec::new()));
    let meta = ops
        .head_typed(BUCKET, &key)
        .await
        .unwrap()
        .expect("present");
    assert_eq!(meta.size, 0);
}

#[tokio::test]
async fn typed_absence_is_none() {
    let (ops, state) = setup().await;
    let key = key("absent");
    state.put_object(BUCKET, &key, MockObject::missing());
    assert!(ops
        .head_typed(BUCKET, &key)
        .await
        .expect("absence is not an error")
        .is_none());
}

/// A HEAD 404 has no body, so the service's `NoSuchBucket` code never reaches the client: a
/// mistyped bucket looks exactly like a missing key and is `Ok(None)`.  The GET of the same
/// missing bucket carries the XML error code and IS told apart (`Other`, not `NotFound`).
/// This pins the documented limit; it is not a promise that HEAD can detect a missing bucket.
#[tokio::test]
async fn a_head_on_a_missing_bucket_cannot_tell_it_from_a_missing_key() {
    let (ops, state) = setup().await;
    let missing_bucket = "gi-no-such-bucket";
    let key = key("no-bucket");
    // The wire answer for a missing bucket on HEAD: 404 with an empty body.
    state.put_object(missing_bucket, &key, MockObject::missing());
    let outcome = ops
        .head_typed(missing_bucket, &key)
        .await
        .expect("a HEAD 404 is absence, not an error");
    assert!(outcome.is_none(), "{outcome:?}");

    // The same bucket on GET: the 404 body names the bucket.
    let get_key = key.clone() + "-get";
    state.put_object(
        missing_bucket,
        &get_key,
        MockObject::new(pattern(4, 10)).with_fault(
            None,
            1,
            s3_get_into_support::mock::ObjectFault::Status(404, "NoSuchBucket"),
        ),
    );
    let mut dst = [0u8; 4];
    let err = ops
        .get_into(
            missing_bucket,
            &get_key,
            &mut dst,
            &s3dlio::receive_progress::ReceiveProgress::new(),
        )
        .await
        .expect_err("a GET names the missing bucket");
    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
}

/// Everything that is not typed absence is an error, with its own kind.
#[tokio::test]
async fn refusals_and_server_failures_are_errors_never_none() {
    let (ops, state) = setup().await;
    for (tag, status, expected) in [
        ("forbidden", 403u16, ObjectErrorKind::Forbidden),
        ("server", 500, ObjectErrorKind::Server),
        ("throttled", 503, ObjectErrorKind::Throttled),
    ] {
        let key = key(tag);
        let mut object = MockObject::new(pattern(4, 4));
        object.head_status = Some(status);
        state.put_object(BUCKET, &key, object);
        let err = ops.head_typed(BUCKET, &key).await.expect_err("not absence");
        assert_eq!(kind_of(&err), expected, "{tag}");
    }
}

#[tokio::test]
async fn a_head_with_no_content_length_is_an_error_not_a_zero_size() {
    let (ops, state) = setup().await;
    let key = key("no-length");
    let mut object = MockObject::new(pattern(9, 5));
    object.head_content_length = false;
    state.put_object(BUCKET, &key, object);
    let err = ops.head_typed(BUCKET, &key).await.expect_err("no size");
    assert!(format!("{err:#}").contains("Content-Length"), "{err:#}");
    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
}

/// Each call asks the service: nothing is cached, and the answer follows the object.
#[tokio::test]
async fn every_call_takes_its_own_head_and_sees_the_current_object() {
    let (ops, state) = setup().await;
    let key = key("fresh");
    state.put_object(BUCKET, &key, MockObject::new(pattern(4, 6)));
    assert_eq!(ops.head_typed(BUCKET, &key).await.unwrap().unwrap().size, 4);
    let new = pattern(9, 7);
    let etag = s3_get_into_support::mock::etag_for(&new);
    state.overwrite_object(BUCKET, &key, new, Some(etag.clone()));
    let meta = ops.head_typed(BUCKET, &key).await.unwrap().unwrap();
    assert_eq!((meta.size, meta.etag), (9, Some(etag)));
    assert_eq!(heads(&state, &key), 2);
}

#[tokio::test]
async fn the_head_goes_to_the_clients_endpoint_and_never_to_the_global_one() {
    let global = init(&[]).1;
    let (other_port, other) = s3_get_into_support::mock::start_isolated_mock_s3_server();
    let ops = ops(client_for(other_port).await);
    let key = key("routing");
    other.put_object(BUCKET, &key, MockObject::new(pattern(5, 8)));
    global.put_object(BUCKET, &key, MockObject::new(pattern(50, 9)));
    let meta = ops.head_typed(BUCKET, &key).await.unwrap().unwrap();
    assert_eq!(meta.size, 5, "the size of the client's endpoint's object");
    assert!(global.requests_for(BUCKET, &key).is_empty());
}
