// tests/s3_get_into_fanout.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// S3 caller-buffer GET, the RANGED FAN-OUT route (plan §4.3 "S3" and §9): a fresh HEAD for the
// size and the ETag, then ranges, each sent with `If-Match` on every attempt, validated against
// the `Content-Range` and written straight into disjoint parts of the caller's buffer.
//
// This binary sets the range threshold to 0 and the chunk size to 4 bytes (concurrency 2), so
// a 16-byte object is four ranges. The single-request route is in `s3_get_into_single.rs`.

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use s3_get_into_support::mock::{etag_for, MockObject, ObjectFault};
use s3_get_into_support::*;
use s3dlio::object_error::ObjectErrorKind;
use s3dlio::receive_progress::ReceiveProgress;

const N: usize = 16;

fn setup_env() -> (u16, s3_get_into_support::mock::MockS3State) {
    init(&[
        ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
        ("S3DLIO_CHUNK_SIZE", "4"),
        ("S3DLIO_RANGE_CONCURRENCY", "2"),
    ])
}

async fn setup() -> (
    s3dlio::s3_ops::S3Ops,
    s3_get_into_support::mock::MockS3State,
) {
    let (port, state) = setup_env();
    (ops(client_for(port).await), state)
}

fn ranges(requests: &[s3_get_into_support::mock::RecordedRequest]) -> Vec<String> {
    let mut r: Vec<String> = requests.iter().filter_map(|q| q.range.clone()).collect();
    r.sort();
    r
}

#[tokio::test]
async fn a_fresh_head_then_pinned_ranges_fill_the_buffer() {
    let (ops, state) = setup().await;
    // 18 bytes: four full ranges and a short last one.
    let len = 18usize;
    let key = key("exact");
    let data = pattern(len, 1);
    let object = MockObject::new(data.clone());
    let etag = object.etag.clone().expect("etag");
    state.put_object(BUCKET, &key, object);
    let mut dst = Dst::new(len);
    let progress = ReceiveProgress::new();

    read(&ops, &key, &mut dst, &progress).await.expect("read");

    assert_eq!(dst.bytes(), &data[..]);
    dst.assert_canaries();
    assert_eq!(progress.filled(), len as u64);
    assert!(progress.first_byte().is_some());
    assert_eq!(heads(&state, &key), 1, "one HEAD per call");
    let gets = gets(&state, &key);
    assert_eq!(
        ranges(&gets),
        [
            "bytes=0-3",
            "bytes=12-15",
            "bytes=16-17",
            "bytes=4-7",
            "bytes=8-11"
        ]
    );
    assert!(
        gets.iter()
            .all(|r| r.if_match.as_deref() == Some(etag.as_str())),
        "every range carries the HEAD's ETag: {gets:?}"
    );
}

/// The size cache is warm and STALE (it says 8 bytes; the object is now 16): the fan-out takes
/// a fresh HEAD, so the read still succeeds and sees the real size.
#[tokio::test]
async fn the_head_is_fresh_despite_a_warm_stale_size_cache() {
    let (port, state) = setup_env();
    let key = key("stale-cache");
    let uri = format!("s3://{BUCKET}/{key}");
    state.put_object(BUCKET, &key, MockObject::new(pattern(8, 2)));
    // Warm the process-wide size cache with the 8-byte size (this uses the global client).
    s3dlio::s3_utils::get_object_uri_optimized_async(&uri)
        .await
        .expect("warm the cache");
    s3dlio::s3_utils::get_object_uri_optimized_async(&uri)
        .await
        .expect("served from the cache");
    assert_eq!(heads(&state, &key), 1, "the second call used the cache");

    let new = pattern(N, 3);
    state.overwrite_object(BUCKET, &key, new.clone(), Some(etag_for(&new)));
    let ops = ops(client_for(port).await);
    let mut dst = Dst::new(N);
    read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect("the stale cache must not matter");

    assert_eq!(dst.bytes(), &new[..]);
    assert_eq!(heads(&state, &key), 2, "the call took its own HEAD");
}

#[tokio::test]
async fn a_broken_range_is_retried_with_if_match_and_counted_once() {
    let (ops, state) = setup().await;
    let key = key("retry");
    let data = pattern(N, 4);
    let object =
        MockObject::new(data.clone()).with_fault(Some(8), 1, ObjectFault::TruncateBody { send: 2 });
    let etag = object.etag.clone().expect("etag");
    state.put_object(BUCKET, &key, object);
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    read(&ops, &key, &mut dst, &progress)
        .await
        .expect("retried");

    assert_eq!(
        dst.bytes(),
        &data[..],
        "the retry restarted at the part start"
    );
    assert_eq!(progress.filled(), N as u64, "each position is counted once");
    let gets = gets(&state, &key);
    let eights: Vec<_> = gets
        .iter()
        .filter(|r| r.range.as_deref() == Some("bytes=8-11"))
        .collect();
    assert_eq!(eights.len(), 2, "the broken range was requested twice");
    assert!(
        gets.iter()
            .all(|r| r.if_match.as_deref() == Some(etag.as_str())),
        "If-Match on every range AND every retry"
    );
    assert_eq!(heads(&state, &key), 1, "a retry takes no new HEAD");
}

/// Same-size overwrite while the ranges are in flight: the changed object is refused (412),
/// the read fails, and two versions are never reported as one successful read.
#[tokio::test]
async fn a_same_size_overwrite_during_the_fan_out_is_version_changed() {
    let (ops, state) = setup().await;
    let key = key("overwrite");
    let new = pattern(N, 55);
    let object = MockObject::new(pattern(N, 5)).with_fault(
        Some(8),
        1,
        ObjectFault::OverwriteFirst {
            etag: Some(etag_for(&new)),
            body: new,
        },
    );
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("the object changed");

    assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
    // A 412 is a definite answer: the refused range is not requested again.
    let again = gets(&state, &key)
        .iter()
        .filter(|r| r.range.as_deref() == Some("bytes=8-11"))
        .count();
    assert_eq!(again, 1);
}

/// A server that ignores `If-Match` still cannot slip another version in: the response ETag is
/// compared with the pinned one.
#[tokio::test]
async fn a_server_that_ignores_if_match_is_caught_by_the_response_etag() {
    let (ops, state) = setup().await;
    let key = key("ignores");
    let new = pattern(N, 56);
    let mut object = MockObject::new(pattern(N, 6)).with_fault(
        Some(4),
        1,
        ObjectFault::OverwriteFirst {
            etag: Some(etag_for(&new)),
            body: new,
        },
    );
    object.honour_if_match = false;
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("another version");

    assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
}

#[tokio::test]
async fn a_pinned_not_found_is_version_changed_never_absence() {
    let (ops, state) = setup().await;
    let key = key("vanished");
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 7)).with_fault(
            Some(4),
            1000,
            ObjectFault::Status(404, "NoSuchKey"),
        ),
    );
    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("gone");
    assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
}

/// A HEAD that fails, or that supplies no size or no usable ETag, leaves nothing to pin ranges
/// to: the read is ONE unranged GET (pinned to its own first response if it must retry).
#[tokio::test]
async fn without_a_usable_head_the_read_is_one_unranged_get() {
    let (ops, state) = setup().await;
    type Tweak = fn(&mut MockObject);
    let cases: [(&str, Tweak); 5] = [
        ("head-403", |o| o.head_status = Some(403)),
        ("head-no-etag", |o| o.head_etag = false),
        ("head-no-length", |o| o.head_content_length = false),
        ("no-etag-anywhere", |o| o.etag = None),
        ("weak-etag", |o| o.etag = Some("W/\"abc\"".to_string())),
    ];
    for (tag, tweak) in cases {
        let key = key(tag);
        let data = pattern(N, 8);
        let mut object = MockObject::new(data.clone());
        tweak(&mut object);
        state.put_object(BUCKET, &key, object);
        let mut dst = Dst::new(N);
        let progress = ReceiveProgress::new();

        read(&ops, &key, &mut dst, &progress)
            .await
            .unwrap_or_else(|e| panic!("{tag}: {e:#}"));

        assert_eq!(dst.bytes(), &data[..], "{tag}");
        assert_eq!(progress.filled(), N as u64, "{tag}");
        assert_eq!(heads(&state, &key), 1, "{tag}");
        let gets = gets(&state, &key);
        assert_eq!(gets.len(), 1, "{tag}: one GET");
        assert_eq!(gets[0].range, None, "{tag}: unranged");
    }
}

#[tokio::test]
async fn a_head_size_that_differs_from_the_buffer_is_a_mismatch_before_any_get() {
    let (ops, state) = setup().await;
    for (tag, size) in [("short", 10usize), ("long", 20usize)] {
        let key = key(tag);
        state.put_object(BUCKET, &key, MockObject::new(pattern(size, 9)));
        let mut dst = Dst::new(N);

        let err = read(&ops, &key, &mut dst, &ReceiveProgress::new())
            .await
            .expect_err("mismatch");

        assert_eq!(
            kind_of(&err),
            ObjectErrorKind::LengthMismatch {
                expected: N as u64,
                actual: Some(size as u64)
            },
            "{tag}"
        );
        dst.assert_untouched();
        assert!(gets(&state, &key).is_empty(), "{tag}: no GET was sent");
    }
}

#[tokio::test]
async fn a_head_for_an_absent_key_falls_back_to_a_get_that_reports_not_found() {
    let (ops, state) = setup().await;
    let key = key("absent");
    state.put_object(BUCKET, &key, MockObject::missing());
    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("absent");
    assert_eq!(kind_of(&err), ObjectErrorKind::NotFound);
    assert_eq!(gets(&state, &key).len(), 1);
}

/// A range response is held to the range that was asked for.
#[tokio::test]
async fn the_content_range_of_every_response_is_validated() {
    let (ops, state) = setup().await;
    let cases = [
        (
            "wrong-bounds",
            ObjectFault::ContentRange {
                first: 5,
                last: 8,
                total: 16,
            },
            ObjectErrorKind::Other,
        ),
        (
            "wrong-total",
            ObjectFault::ContentRange {
                first: 4,
                last: 7,
                total: 17,
            },
            ObjectErrorKind::LengthMismatch {
                expected: 16,
                actual: Some(17),
            },
        ),
        (
            "no-content-range",
            ObjectFault::NoContentRange,
            ObjectErrorKind::Other,
        ),
    ];
    for (tag, fault, expected) in cases {
        let key = key(tag);
        state.put_object(
            BUCKET,
            &key,
            MockObject::new(pattern(N, 10)).with_fault(Some(4), 1000, fault),
        );
        let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
            .await
            .expect_err("a bad Content-Range");
        assert_eq!(kind_of(&err), expected, "{tag}");
        // A malformed answer repeats exactly: it is final, not retried.
        assert_eq!(gets_of_range(&state, &key, "bytes=4-7"), 1, "{tag}");
    }
}

// ───────────── retries, protocol violations and the pin (per range part) ─────────────
//
// Every case here faults ONE range (`bytes=4-7`) so the count of GETs for it is exact: the
// other parts are served honestly.  The SDK retries a 5xx on its own, so the counts below use
// faults the SDK cannot see (a body that breaks after the headers, a clean short body, a
// well-formed answer that breaks the protocol): they measure the s3dlio loop and nothing else.

/// The faulted range of the cases below.
const FAULTED: &str = "bytes=4-7";

/// A server that ignores `Range` answers 200 with the whole object and no `Content-Range`.
/// That is the same answer on every attempt, so the part costs ONE request.
#[tokio::test]
async fn a_server_that_ignores_range_is_one_get_for_the_part_not_the_retry_budget() {
    let (ops, state) = setup().await;
    let key = key("ignores-range");
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 14)).with_fault(Some(4), 1000, ObjectFault::IgnoreRange),
    );

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("no Content-Range");

    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
    assert_eq!(gets_of_range(&state, &key, FAULTED), 1);
}

/// The length of the range body is checked against the range asked for.  A Content-Length that
/// disagrees, or a body that runs long, is a protocol violation (final, one GET).  A body that
/// ends clean but short is a delivery failure (`Transport`): retried, with the budget, pinned.
#[tokio::test]
async fn the_range_body_length_is_checked_and_only_a_short_body_is_retried() {
    let (ops, state) = setup().await;
    // (tag, delta, Content-Length sent?, expected kind, GETs for the faulted range, text)
    let cases = [
        (
            "cl-long",
            1isize,
            true,
            ObjectErrorKind::Other,
            1,
            "Content-Length",
        ),
        (
            "cl-short",
            -1,
            true,
            ObjectErrorKind::Other,
            1,
            "Content-Length",
        ),
        (
            "body-long",
            1,
            false,
            ObjectErrorKind::Other,
            1,
            "longer than requested",
        ),
        (
            "body-short",
            -1,
            false,
            ObjectErrorKind::Transport,
            attempts(),
            "shorter than requested",
        ),
    ];
    for (tag, delta, with_length, expected, count, text) in cases {
        let key = key(tag);
        let mut object = MockObject::new(pattern(N, 15)).with_fault(
            Some(4),
            1000,
            ObjectFault::BodyLength { delta },
        );
        object.get_content_length = with_length;
        let etag = object.etag.clone().expect("etag");
        state.put_object(BUCKET, &key, object);

        let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
            .await
            .expect_err(tag);

        assert_eq!(kind_of(&err), expected, "{tag}: {err:#}");
        assert!(format!("{err:#}").contains(text), "{tag}: {err:#}");
        assert_eq!(gets_of_range(&state, &key, FAULTED), count, "{tag}");
        assert!(
            gets(&state, &key)
                .iter()
                .all(|r| r.if_match.as_deref() == Some(etag.as_str())),
            "{tag}: every attempt is pinned"
        );
    }
}

/// A short range body that comes right on the next attempt: the part is retried, the data is
/// whole, and each position is counted once.
#[tokio::test]
async fn a_short_range_body_is_retried_to_success() {
    let (ops, state) = setup().await;
    let key = key("short-once");
    let data = pattern(N, 16);
    let mut object =
        MockObject::new(data.clone()).with_fault(Some(4), 1, ObjectFault::BodyLength { delta: -1 });
    object.get_content_length = false;
    state.put_object(BUCKET, &key, object);
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    read(&ops, &key, &mut dst, &progress)
        .await
        .expect("retried");

    assert_eq!(dst.bytes(), &data[..]);
    assert_eq!(progress.filled(), N as u64);
    assert_eq!(gets_of_range(&state, &key, FAULTED), 2);
}

/// A body that breaks mid-stream on every attempt: exactly the retry budget of GETs for that
/// range, each pinned, then a `Transport` error.
#[tokio::test]
async fn a_persistently_broken_range_makes_exactly_the_retry_budget_of_gets() {
    let (ops, state) = setup().await;
    let key = key("always-broken");
    let object = MockObject::new(pattern(N, 17)).with_fault(
        Some(4),
        1000,
        ObjectFault::TruncateBody { send: 2 },
    );
    let etag = object.etag.clone().expect("etag");
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("every attempt breaks");

    assert_eq!(kind_of(&err), ObjectErrorKind::Transport);
    assert_eq!(gets_of_range(&state, &key, FAULTED), attempts());
    assert!(
        gets(&state, &key)
            .iter()
            .all(|r| r.if_match.as_deref() == Some(etag.as_str())),
        "every attempt of every range is pinned"
    );
    assert_eq!(heads(&state, &key), 1, "no retry takes a new HEAD");
}

/// A pinned range answered without an ETag cannot confirm the version, even when the bytes
/// are honest: the read fails (`Other`) and the range is not asked for again.
#[tokio::test]
async fn a_range_response_with_no_etag_cannot_confirm_the_pin() {
    let (ops, state) = setup().await;
    let key = key("range-no-etag");
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 18)).with_fault(Some(4), 1000, ObjectFault::OmitEtag),
    );

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("unconfirmable");

    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
    assert!(format!("{err:#}").contains("cannot confirm"), "{err:#}");
    assert_eq!(gets_of_range(&state, &key, FAULTED), 1);
}

/// The dangerous shape: the object is replaced between the HEAD and a range, the server
/// ignores `If-Match` AND sends no ETag.  Two versions would be mixed in the buffer; the read
/// must be an error, never `Ok`.
#[tokio::test]
async fn a_replaced_object_answered_without_an_etag_is_an_error_never_mixed_data() {
    let (ops, state) = setup().await;
    let key = key("replaced-no-etag");
    let mut object = MockObject::new(pattern(N, 19)).with_fault(
        Some(4),
        1,
        ObjectFault::OverwriteFirst {
            body: pattern(N, 96),
            etag: None,
        },
    );
    object.honour_if_match = false;
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("another version that cannot be confirmed");

    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
}

#[tokio::test]
async fn a_zero_length_destination_never_sends_a_head_or_a_range() {
    let (ops, state) = setup().await;
    let key = key("zero");
    state.put_object(BUCKET, &key, MockObject::new(Vec::new()));
    let progress = ReceiveProgress::new();
    read(&ops, &key, &mut Dst::new(0), &progress)
        .await
        .expect("empty read");
    assert_eq!(heads(&state, &key), 0);
    let gets = gets(&state, &key);
    assert_eq!(gets.len(), 1);
    assert_eq!(gets[0].range, None);
    assert_eq!(progress.filled(), 0);
}

/// The HEAD and every range go to the client's own endpoint; the process-wide endpoint, which
/// holds other bytes under the same key, sees nothing.
#[tokio::test]
async fn the_head_and_the_ranges_go_to_the_clients_endpoint_only() {
    let global = setup_env().1;
    let (other_port, other) = s3_get_into_support::mock::start_isolated_mock_s3_server();
    let ops = ops(client_for(other_port).await);
    let key = key("routing");
    let on_other = pattern(N, 11);
    other.put_object(BUCKET, &key, MockObject::new(on_other.clone()));
    global.put_object(BUCKET, &key, MockObject::new(pattern(N, 12)));
    let mut dst = Dst::new(N);

    read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect("read");

    assert_eq!(dst.bytes(), &on_other[..]);
    assert_eq!(heads(&other, &key), 1);
    assert_eq!(gets(&other, &key).len(), 4);
    assert!(global.requests_for(BUCKET, &key).is_empty());
}

#[tokio::test]
async fn dropping_a_fan_out_read_stops_every_writer_and_keeps_the_progress() {
    let (ops, state) = setup().await;
    let key = key("stalled");
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 13)).with_fault(Some(4), 1, ObjectFault::StallBody { send: 2 }),
    );
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    // Part 0 is complete (4 bytes) and part 4 delivered 2 before it stalled: 6 positions.
    drive_then_drop(
        read(&ops, &key, &mut dst, &progress),
        &progress,
        6,
        std::time::Duration::from_secs(10),
    )
    .await;

    assert!(progress.filled() >= 6, "filled {}", progress.filled());
    let after_drop = dst.bytes().to_vec();
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert_eq!(
        dst.bytes(),
        &after_drop[..],
        "nothing writes after the drop"
    );
    dst.assert_canaries();
}
