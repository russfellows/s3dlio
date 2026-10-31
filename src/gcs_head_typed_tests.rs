// src/gcs_head_typed_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Tests of the GCS object-time conversion (#263) and, below it, the typed HEAD decision logic
// (plan §4.3, §4.5 A8).  Local and deterministic: no network, no credentials.

use super::*;
use google_cloud_storage::model::Object;

/// An object whose update time is parsed from `rfc3339`, built without naming the timestamp
/// type (it lives in a crate this one does not depend on directly).
fn object_updated(rfc3339: &str) -> Object {
    let mut obj = Object::default();
    obj.update_time = Some(rfc3339.try_into().expect("valid timestamp"));
    obj
}

/// A synthetic UTC time is an RFC 3339 string, not a struct dump.
#[test]
fn utc_object_time_is_rfc3339() {
    let obj = object_updated("2025-05-16T09:46:12Z");
    assert_eq!(
        object_time_rfc3339(obj.update_time).as_deref(),
        Some("2025-05-16T09:46:12Z")
    );
}

/// The fraction is kept, down to the nanosecond.
#[test]
fn fractional_object_time_keeps_its_precision() {
    for (input, want) in [
        ("2025-05-16T09:46:12.5Z", "2025-05-16T09:46:12.5Z"),
        ("2025-05-16T09:46:12.123456Z", "2025-05-16T09:46:12.123456Z"),
        (
            "2025-05-16T09:46:12.123456789Z",
            "2025-05-16T09:46:12.123456789Z",
        ),
    ] {
        let obj = object_updated(input);
        assert_eq!(
            object_time_rfc3339(obj.update_time).as_deref(),
            Some(want),
            "{input}"
        );
    }
}

/// The output parses back to the same instant, so nothing but formatting changed.
#[test]
fn rfc3339_output_round_trips_to_the_same_instant() {
    let obj = object_updated("2031-02-03T04:05:06.007008009Z");
    let text = object_time_rfc3339(obj.update_time).expect("present");
    let parsed = chrono::DateTime::parse_from_rfc3339(&text).expect("RFC 3339");
    let want_secs = chrono::NaiveDate::from_ymd_opt(2031, 2, 3)
        .and_then(|d| d.and_hms_opt(4, 5, 6))
        .expect("valid date")
        .and_utc()
        .timestamp();
    assert_eq!(parsed.timestamp(), want_secs);
    assert_eq!(parsed.timestamp_subsec_nanos(), 7_008_009);
}

/// An absent time stays absent; it is never an empty string or a zero time.
#[test]
fn absent_object_time_stays_absent() {
    let obj = Object::default();
    assert_eq!(object_time_rfc3339(obj.update_time), None);
}

/// Both existing metadata sites (control plane and the Rapid descriptor) convert through the
/// helper; neither keeps the Debug-formatting fallback.
#[test]
fn both_metadata_sites_use_the_conversion() {
    let src = include_str!("google_gcs_client.rs");
    let production = src
        .split("#[cfg(test)]")
        .next()
        .expect("source has a production part");
    assert!(
        !production.contains("update_time.map("),
        "a metadata site still maps the update time itself"
    );
    assert!(
        production
            .matches("object_time_rfc3339(obj.update_time)")
            .count()
            >= 2,
        "both metadata sites must convert through object_time_rfc3339"
    );
}

// ---------------------------------------------------------------------------
// Typed HEAD decision logic (plan §4.3 GCS row, §4.5 A8)
// ---------------------------------------------------------------------------

use super::caller_buffer::{head_metadata, head_outcome, metadata_with_generation};
use crate::object_error::{object_error, ObjectErrorKind};
use crate::object_store::HeadMetadata;
use google_cloud_gax::error::rpc::{Code, Status};
use google_cloud_gax::error::Error as Gax;

const KEY: &str = "dir/object";
const WHAT: &str = "gs://bucket/dir/object";
const GENERATION: i64 = 1_700_000_000_000_001;

fn object(size: i64, etag: &str, generation: i64) -> Object {
    Object::default()
        .set_size(size)
        .set_etag(etag)
        .set_generation(generation)
}

fn http(code: u16) -> Gax {
    Gax::http(code, http::HeaderMap::new(), Bytes::from_static(b"body"))
}

fn grpc(code: Code) -> Gax {
    Gax::service(Status::default().set_code(code).set_message("injected"))
}

fn head(result: Result<Object, Gax>) -> Result<Option<HeadMetadata>> {
    head_outcome(result, WHAT, KEY).map(|found| found.map(|(m, g)| head_metadata(m, g)))
}

fn kind(err: &anyhow::Error) -> ObjectErrorKind {
    object_error(err).expect("typed").kind
}

/// A present object reports its size, its ETag and the generation as the version.
#[test]
fn present_object_reports_size_etag_and_generation() {
    let found = head(Ok(object(1234, "CIk=", GENERATION))).expect("present");
    assert_eq!(
        found,
        Some(HeadMetadata {
            size: 1234,
            etag: Some("CIk=".to_string()),
            version_id: Some(GENERATION.to_string()),
        })
    );
}

/// An empty object (the Rapid descriptor case) is present with size 0.  It is not absent and
/// not a failure.
#[test]
fn empty_object_reports_zero_not_absence() {
    let found = head(Ok(object(0, "", GENERATION))).expect("an empty object is present");
    assert_eq!(
        found,
        Some(HeadMetadata {
            size: 0,
            etag: None,
            version_id: Some(GENERATION.to_string()),
        })
    );
}

/// A missing ETag is `None`, and a missing generation gives no version.
#[test]
fn missing_etag_and_generation_are_absent_fields() {
    let found = head(Ok(object(5, "", 0))).expect("present").expect("some");
    assert_eq!(found.etag, None);
    assert_eq!(found.version_id, None);
}

/// Only a typed NotFound is absence, in HTTP and gRPC form.
#[test]
fn only_a_typed_not_found_is_absence() {
    assert_eq!(head(Err(http(404))).expect("absent"), None);
    assert_eq!(head(Err(grpc(Code::NotFound))).expect("absent"), None);
}

/// Every other failure stays a failure.  OUT_OF_RANGE in particular is never absence (it is
/// what a one-byte read of an empty Rapid object reports).
#[test]
fn other_failures_are_never_absence() {
    let cases = [
        (grpc(Code::OutOfRange), ObjectErrorKind::Other),
        (http(403), ObjectErrorKind::Forbidden),
        (grpc(Code::PermissionDenied), ObjectErrorKind::Forbidden),
        (http(503), ObjectErrorKind::Throttled),
        (http(429), ObjectErrorKind::Throttled),
        (http(500), ObjectErrorKind::Server),
        (Gax::io("connection reset"), ObjectErrorKind::Transport),
        (Gax::timeout("deadline"), ObjectErrorKind::Transport),
        (http(401), ObjectErrorKind::Other),
        (Gax::deser("not found"), ObjectErrorKind::Other),
    ];
    for (err, want) in cases {
        let text = format!("{err}");
        let got = head(Err(err)).expect_err(&text);
        assert_eq!(kind(&got), want, "{text}");
    }
}

/// A failed HEAD says so: its context names the HEAD, not a GET.
#[test]
fn a_failed_head_names_the_head_in_its_context() {
    let err = head(Err(http(403))).expect_err("forbidden");
    let message = format!("{err:#}");
    assert!(message.contains("GCS HEAD failed"), "{message}");
    assert!(!message.contains("GET"), "{message}");
}

/// A negative size is an error: it is never read as 0 (an empty object) and never wraps.
#[test]
fn negative_size_is_an_error_not_zero() {
    let err = head(Ok(object(-1, "e", GENERATION))).expect_err("negative size");
    assert_eq!(kind(&err), ObjectErrorKind::Other);
    let err = metadata_with_generation(&object(i64::MIN, "e", GENERATION), KEY)
        .expect_err("negative size");
    assert!(format!("{err:#}").contains("negative size"));
}

/// The crate-internal path carries the generation beside the public struct, which has not
/// gained a field (this literal would stop compiling if it had), and the update time is
/// RFC 3339.
#[test]
fn generation_travels_beside_the_unchanged_public_struct() {
    let mut obj = object(42, "etag", GENERATION);
    obj.update_time = Some("2025-05-16T09:46:12.25Z".try_into().expect("timestamp"));
    let (meta, generation) = metadata_with_generation(&obj, KEY).expect("metadata");
    assert_eq!(generation, GENERATION);
    // An exhaustive pattern: adding a field to the public struct stops this compiling.
    let GcsObjectMetadata {
        size,
        etag,
        updated,
        key,
    } = meta;
    assert_eq!(size, 42);
    assert_eq!(etag.as_deref(), Some("etag"));
    assert_eq!(updated.as_deref(), Some("2025-05-16T09:46:12.25Z"));
    assert_eq!(key, KEY);
}

/// The Rapid metadata route asks the descriptor and requests no data range, so an empty
/// object cannot fail with OUT_OF_RANGE; the control plane's possibly stale size is not
/// used on a Rapid bucket.
#[test]
fn the_rapid_head_route_uses_the_descriptor_without_a_data_range() {
    let src = include_str!("google_gcs_client.rs");
    let at = src
        .find("impl ReadOpener for RapidOpener")
        .expect("Rapid opener");
    let rapid = &src[at..];
    let describe = &rapid[rapid.find("async fn describe").expect("describe")..];
    let describe = &describe[..describe.find("\n        }\n").expect("end of describe")];
    // The request is sent with `send()` (inside the global-runtime context since #303).
    assert!(describe.contains("builder.send()"), "{describe}");
    assert!(!describe.contains("send_and_read"), "{describe}");
    assert!(!describe.contains("ReadRange"), "{describe}");

    let stat = &src[src
        .find("async fn stat_object_with_generation")
        .expect("stat")..];
    let stat = &stat[..stat.find("head_outcome(").expect("outcome")];
    assert!(stat.contains("is_rapid_bucket(bucket)"));
    assert!(stat.contains("opener.describe(None)"));
    assert!(stat.contains("self.control"));
}
