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
