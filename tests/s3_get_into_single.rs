// tests/s3_get_into_single.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// S3 caller-buffer GET, the SINGLE-request route (plan §4.3 "S3" and §9): a read of fewer
// bytes than the range threshold is one unranged GET streamed into the caller's buffer, with
// no HEAD.  This binary keeps the default threshold (32 MiB), so every object here takes that
// route.  The ranged fan-out is in `s3_get_into_fanout.rs`.
//
// Mechanism: `S3Ops::get_into` on a client for the in-process mock server (tests/common/
// mock_s3_server.rs).  The mock answers like a real server (206 with the total in
// Content-Range, 412 on a stale If-Match for ranged and unranged GETs, 416 for a range on an
// empty object, 404 with or without If-Match, quoted hex ETags) and injects faults one
// request at a time.  Every test registers its own object under a unique key.

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use s3_get_into_support::mock::{etag_for, MockObject, ObjectFault};
use s3_get_into_support::*;
use s3dlio::object_error::ObjectErrorKind;
use s3dlio::receive_progress::ReceiveProgress;

const N: usize = 16;

async fn setup() -> (
    s3dlio::s3_ops::S3Ops,
    s3_get_into_support::mock::MockS3State,
) {
    let (port, state) = init(&[]);
    (ops(client_for(port).await), state)
}

#[tokio::test]
async fn an_exact_object_is_streamed_into_the_buffer_with_one_unranged_get() {
    let (ops, state) = setup().await;
    let key = key("exact");
    let data = pattern(N, 1);
    state.put_object(BUCKET, &key, MockObject::new(data.clone()));
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    read(&ops, &key, &mut dst, &progress).await.expect("read");

    assert_eq!(dst.bytes(), &data[..]);
    dst.assert_canaries();
    assert_eq!(progress.filled(), N as u64);
    assert!(progress.first_byte().is_some());
    let gets = gets(&state, &key);
    assert_eq!(gets.len(), 1, "one request");
    assert_eq!(gets[0].range, None, "unranged");
    assert_eq!(
        gets[0].if_match, None,
        "nothing is pinned before a response"
    );
    assert_eq!(heads(&state, &key), 0, "the single route sends no HEAD");
}

#[tokio::test]
async fn a_short_object_is_a_length_mismatch_and_writes_nothing() {
    let (ops, state) = setup().await;
    let key = key("short");
    state.put_object(BUCKET, &key, MockObject::new(pattern(10, 2)));
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    let err = read(&ops, &key, &mut dst, &progress)
        .await
        .expect_err("short");

    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: N as u64,
            actual: Some(10)
        }
    );
    dst.assert_untouched();
    assert_eq!(progress.filled(), 0, "the body was never read");
    assert_eq!(gets(&state, &key).len(), 1, "a length error is not retried");
}

#[tokio::test]
async fn a_long_object_is_a_length_mismatch_and_writes_nothing() {
    let (ops, state) = setup().await;
    let key = key("long");
    state.put_object(BUCKET, &key, MockObject::new(pattern(20, 3)));
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    let err = read(&ops, &key, &mut dst, &progress)
        .await
        .expect_err("long");

    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: N as u64,
            actual: Some(20)
        }
    );
    dst.assert_untouched();
    assert_eq!(progress.filled(), 0);
    assert_eq!(gets(&state, &key).len(), 1);
}

/// With no Content-Length the size is only known from the stream: a body longer than the
/// destination is refused before it is written (never past the end), a shorter one is
/// reported when the stream ends.
#[tokio::test]
async fn without_a_content_length_the_stream_is_still_held_to_the_exact_length() {
    let (ops, state) = setup().await;

    let long = key("nolen-long");
    let mut object = MockObject::new(pattern(20, 4));
    object.get_content_length = false;
    state.put_object(BUCKET, &long, object);
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();
    let err = read(&ops, &long, &mut dst, &progress)
        .await
        .expect_err("long");
    // The stream is not read to its end, so `actual` is a LOWER BOUND: the bytes received
    // when the first chunk past the end arrived (more than the destination, at most the body).
    match kind_of(&err) {
        ObjectErrorKind::LengthMismatch {
            expected,
            actual: Some(at_least),
        } => {
            assert_eq!(expected, N as u64);
            assert!(
                at_least > N as u64 && at_least <= 20,
                "a lower bound past the end, not the declared size: {at_least}"
            );
        }
        other => panic!("expected a length mismatch with a lower bound, got {other:?}"),
    }
    assert!(format!("{err:#}").contains("at least"), "{err:#}");
    dst.assert_untouched();

    let short = key("nolen-short");
    let mut object = MockObject::new(pattern(10, 5));
    object.get_content_length = false;
    state.put_object(BUCKET, &short, object);
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();
    let err = read(&ops, &short, &mut dst, &progress)
        .await
        .expect_err("short");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: N as u64,
            actual: Some(10)
        }
    );
    dst.assert_canaries();
    assert_eq!(progress.filled(), 10, "the positions touched are reported");

    // The exact length with no Content-Length is accepted.
    let exact = key("nolen-exact");
    let data = pattern(N, 6);
    let mut object = MockObject::new(data.clone());
    object.get_content_length = false;
    state.put_object(BUCKET, &exact, object);
    let mut dst = Dst::new(N);
    read(&ops, &exact, &mut dst, &ReceiveProgress::new())
        .await
        .expect("exact");
    assert_eq!(dst.bytes(), &data[..]);
    dst.assert_canaries();
}

#[tokio::test]
async fn a_zero_length_destination_reads_an_empty_object_with_one_unranged_get() {
    let (ops, state) = setup().await;
    let key = key("zero");
    state.put_object(BUCKET, &key, MockObject::new(Vec::new()));
    let mut dst = Dst::new(0);
    let progress = ReceiveProgress::new();

    read(&ops, &key, &mut dst, &progress)
        .await
        .expect("empty read");

    dst.assert_canaries();
    assert_eq!(progress.filled(), 0);
    assert!(progress.first_byte().is_none(), "no byte, no first byte");
    let gets = gets(&state, &key);
    assert_eq!(gets.len(), 1);
    assert_eq!(gets[0].range, None, "`bytes=0--1` is never sent");
    assert_eq!(heads(&state, &key), 0);
}

#[tokio::test]
async fn a_zero_length_destination_over_a_non_empty_object_is_a_mismatch_with_the_body_unread() {
    let (ops, state) = setup().await;
    let key = key("zero-nonempty");
    state.put_object(BUCKET, &key, MockObject::new(pattern(5, 7)));
    let mut dst = Dst::new(0);
    let progress = ReceiveProgress::new();

    let err = read(&ops, &key, &mut dst, &progress)
        .await
        .expect_err("mismatch");

    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: Some(5)
        }
    );
    dst.assert_canaries();
    assert_eq!(progress.filled(), 0);
    assert!(gets(&state, &key).iter().all(|r| r.range.is_none()));

    // An empty destination can only be checked from the size header.
    let nolen = key.clone() + "-nolen";
    let mut object = MockObject::new(Vec::new());
    object.get_content_length = false;
    state.put_object(BUCKET, &nolen, object);
    let err = read(&ops, &nolen, &mut Dst::new(0), &ReceiveProgress::new())
        .await
        .expect_err("no size header");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: None
        }
    );
}

#[tokio::test]
async fn a_zero_length_read_never_sends_a_range_even_for_an_empty_object_that_rejects_ranges() {
    // The service answers any range on an empty object with 416 (observed on a real
    // endpoint); the empty read must not depend on it.
    let (ops, state) = setup().await;
    let key = key("zero-416");
    state.put_object(BUCKET, &key, MockObject::new(Vec::new()));
    read(&ops, &key, &mut Dst::new(0), &ReceiveProgress::new())
        .await
        .expect("empty read");
    assert!(gets(&state, &key).iter().all(|r| r.range.is_none()));
}

#[tokio::test]
async fn an_absent_key_is_not_found_and_not_retried() {
    let (ops, state) = setup().await;
    let key = key("absent");
    state.put_object(BUCKET, &key, MockObject::missing());
    let mut dst = Dst::new(N);

    let err = read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect_err("absent");

    assert_eq!(kind_of(&err), ObjectErrorKind::NotFound);
    dst.assert_untouched();
    assert_eq!(gets(&state, &key).len(), 1);
    // The same for an empty destination.
    let err = read(&ops, &key, &mut Dst::new(0), &ReceiveProgress::new())
        .await
        .expect_err("absent");
    assert_eq!(kind_of(&err), ObjectErrorKind::NotFound);
}

#[tokio::test]
async fn a_refusal_is_forbidden_and_not_retried() {
    let (ops, state) = setup().await;
    let key = key("forbidden");
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 8)).with_fault(
            None,
            100,
            ObjectFault::Status(403, "AccessDenied"),
        ),
    );
    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("forbidden");
    assert_eq!(kind_of(&err), ObjectErrorKind::Forbidden);
    assert_eq!(gets(&state, &key).len(), 1, "a 403 is a definite answer");
}

#[tokio::test]
async fn throttling_and_server_failures_are_typed_and_retried() {
    let (ops, state) = setup().await;
    for (tag, status, code, expected) in [
        ("throttled", 503, "SlowDown", ObjectErrorKind::Throttled),
        ("server", 500, "InternalError", ObjectErrorKind::Server),
    ] {
        let key = key(tag);
        state.put_object(
            BUCKET,
            &key,
            MockObject::new(pattern(N, 9)).with_fault(
                None,
                1000,
                ObjectFault::Status(status, code),
            ),
        );
        let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
            .await
            .expect_err("failure");
        assert_eq!(kind_of(&err), expected, "{tag}");
        assert!(
            gets(&state, &key).len() > 1,
            "{tag}: a transient failure is retried"
        );
    }
}

#[tokio::test]
async fn a_transient_failure_before_the_body_is_retried_to_success() {
    let (ops, state) = setup().await;
    let key = key("flaky");
    let data = pattern(N, 10);
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(data.clone()).with_fault(
            None,
            1,
            ObjectFault::Status(500, "InternalError"),
        ),
    );
    let mut dst = Dst::new(N);
    read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect("retried");
    assert_eq!(dst.bytes(), &data[..]);
    assert!(gets(&state, &key).len() >= 2);
}

/// A body that breaks after 6 bytes: the retry restarts at the span start (the final bytes are
/// the second response's, in full), is pinned to the first response's ETag, and the positions
/// are counted once (16, not 6 + 16).
#[tokio::test]
async fn a_broken_body_is_retried_from_the_start_pinned_to_the_first_etag() {
    let (ops, state) = setup().await;
    let key = key("broken");
    let data = pattern(N, 11);
    let object =
        MockObject::new(data.clone()).with_fault(None, 1, ObjectFault::TruncateBody { send: 6 });
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
        "the cursor restarted at the span start"
    );
    dst.assert_canaries();
    assert_eq!(progress.filled(), N as u64, "each position is counted once");
    let gets = gets(&state, &key);
    assert_eq!(gets.len(), 2);
    assert_eq!(gets[0].if_match, None);
    assert_eq!(
        gets[1].if_match.as_deref(),
        Some(etag.as_str()),
        "the retry is pinned"
    );
}

/// The object is replaced (same size, new ETag) between the broken attempt and its retry: the
/// pinned retry is refused with 412 and the read fails; the two versions are never combined.
#[tokio::test]
async fn an_overwrite_between_attempts_fails_with_version_changed() {
    let (ops, state) = setup().await;
    let key = key("overwritten");
    let new = pattern(N, 99);
    let object = MockObject::new(pattern(N, 12))
        .with_fault(None, 1, ObjectFault::TruncateBody { send: 6 })
        .with_fault(
            None,
            1,
            ObjectFault::OverwriteFirst {
                etag: Some(etag_for(&new)),
                body: new,
            },
        );
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("version changed");

    assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
    assert_eq!(
        gets(&state, &key).len(),
        2,
        "a changed version is not retried"
    );
}

/// Once pinned, a 404 is the pinned version being gone, never absence.
#[tokio::test]
async fn a_not_found_on_a_pinned_retry_is_version_changed() {
    let (ops, state) = setup().await;
    let key = key("vanished");
    let object = MockObject::new(pattern(N, 13))
        .with_fault(None, 1, ObjectFault::TruncateBody { send: 6 })
        .with_fault(None, 1000, ObjectFault::Status(404, "NoSuchKey"));
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("version changed");

    assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
}

/// A server that ignores `If-Match` and answers with another version is caught by comparing
/// the response ETag with the pinned one.
#[tokio::test]
async fn a_server_that_ignores_if_match_is_caught_by_the_response_etag() {
    let (ops, state) = setup().await;
    let key = key("ignores-if-match");
    let new = pattern(N, 98);
    let mut object = MockObject::new(pattern(N, 14))
        .with_fault(None, 1, ObjectFault::TruncateBody { send: 6 })
        .with_fault(
            None,
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
        .expect_err("version changed");

    assert_eq!(kind_of(&err), ObjectErrorKind::VersionChanged);
}

/// No usable ETag (absent, or weak so `If-Match` could never match): the response cannot be
/// pinned, so a partially received request fails instead of being retried from pieces of an
/// unknown version.
#[tokio::test]
async fn a_partial_response_with_no_usable_etag_fails_and_is_not_retried() {
    let (ops, state) = setup().await;
    for (tag, etag) in [
        ("no-etag", None),
        ("weak-etag", Some("W/\"abc\"".to_string())),
    ] {
        let key = key(tag);
        let mut object = MockObject::new(pattern(N, 15)).with_fault(
            None,
            1,
            ObjectFault::TruncateBody { send: 6 },
        );
        object.etag = etag;
        state.put_object(BUCKET, &key, object);
        let mut dst = Dst::new(N);
        let progress = ReceiveProgress::new();

        let err = read(&ops, &key, &mut dst, &progress)
            .await
            .expect_err("partial");

        assert_eq!(kind_of(&err), ObjectErrorKind::Transport, "{tag}");
        assert!(format!("{err:#}").contains("not retried"), "{tag}: {err:#}");
        assert_eq!(gets(&state, &key).len(), 1, "{tag}: no second request");
        assert_eq!(
            progress.filled(),
            6,
            "{tag}: the touched positions are reported"
        );
        dst.assert_canaries();
    }
}

/// The same server, but the body breaks before ANY byte was written: nothing is mixed, so the
/// retry is allowed (unpinned: there is no ETag to send).
#[tokio::test]
async fn with_no_etag_a_failure_before_the_first_byte_is_retried_unpinned() {
    let (ops, state) = setup().await;
    let key = key("no-etag-early");
    let data = pattern(N, 16);
    let mut object =
        MockObject::new(data.clone()).with_fault(None, 1, ObjectFault::TruncateBody { send: 0 });
    object.etag = None;
    state.put_object(BUCKET, &key, object);
    let mut dst = Dst::new(N);

    read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect("retried before any byte was written");

    assert_eq!(dst.bytes(), &data[..]);
    let gets = gets(&state, &key);
    assert_eq!(gets.len(), 2);
    assert!(gets.iter().all(|r| r.if_match.is_none()));
}

/// The request goes to the handler's own client's server and to no other: a different server,
/// reached only through the process-wide client, sees nothing.
#[tokio::test]
async fn the_request_goes_to_the_clients_endpoint_and_never_to_the_global_one() {
    let global = init(&[]).1;
    let (other_port, other) = mock::start_isolated_mock_s3_server();
    let ops = ops(client_for(other_port).await);
    let key = key("routing");
    let on_other = pattern(N, 17);
    other.put_object(BUCKET, &key, MockObject::new(on_other.clone()));
    // The process-wide endpoint holds DIFFERENT bytes under the same key.
    global.put_object(BUCKET, &key, MockObject::new(pattern(N, 18)));
    let mut dst = Dst::new(N);

    read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect("read");

    assert_eq!(
        dst.bytes(),
        &on_other[..],
        "the bytes came from the client's endpoint"
    );
    assert_eq!(other.requests_for(BUCKET, &key).len(), 1);
    assert!(
        global.requests_for(BUCKET, &key).is_empty(),
        "the process-wide endpoint saw a request"
    );
}

/// Dropping the future stops the read: the positions already written stay reported, and the
/// destination is no longer written after the drop.
#[tokio::test]
async fn dropping_the_read_stops_the_writer_and_keeps_the_progress() {
    let (ops, state) = setup().await;
    let key = key("stalled");
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 19)).with_fault(None, 1, ObjectFault::StallBody { send: 6 }),
    );
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();

    drive_then_drop(
        read(&ops, &key, &mut dst, &progress),
        &progress,
        6,
        std::time::Duration::from_secs(10),
    )
    .await;

    assert_eq!(progress.filled(), 6, "the positions written stay reported");
    let after_drop = dst.bytes().to_vec();
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert_eq!(
        dst.bytes(),
        &after_drop[..],
        "nothing writes after the drop"
    );
    dst.assert_canaries();
}

// ───────────── the s3dlio retry loop itself (the SDK does not retry a broken body) ─────────────
//
// The AWS SDK retries a 5xx inside one `send()`, so a 500 fault exercises the SDK's loop (and
// its count is the SDK's budget times s3dlio's).  A body that breaks AFTER the headers is
// invisible to the SDK: only the loop under test retries it, so these counts are exactly its
// budget (`S3DLIO_MAX_RETRY_ATTEMPTS`).

/// A body that breaks after 6 bytes on every attempt: exactly the retry budget of GETs, the
/// first unpinned and every retry pinned, the positions counted once, and the failure typed
/// `Transport` (a delivery failure, not a refusal).
#[tokio::test]
async fn a_persistently_broken_body_makes_exactly_the_retry_budget_of_gets() {
    let (ops, state) = setup().await;
    let key = key("always-broken");
    let object = MockObject::new(pattern(N, 20)).with_fault(
        None,
        1000,
        ObjectFault::TruncateBody { send: 6 },
    );
    let etag = object.etag.clone().expect("etag");
    state.put_object(BUCKET, &key, object);
    let progress = ReceiveProgress::new();

    let err = read(&ops, &key, &mut Dst::new(N), &progress)
        .await
        .expect_err("every attempt breaks");

    assert_eq!(kind_of(&err), ObjectErrorKind::Transport);
    let gets = gets(&state, &key);
    assert_eq!(gets.len(), attempts(), "exactly the retry budget");
    assert_eq!(gets[0].if_match, None);
    assert!(
        gets[1..]
            .iter()
            .all(|r| r.if_match.as_deref() == Some(etag.as_str())),
        "every retry is pinned: {gets:?}"
    );
    assert_eq!(progress.filled(), 6, "each position is counted once");
}

/// A protocol violation is final: a missing or wrong `Content-Range` is answered the same way
/// again, so it costs ONE request, not the retry budget.  A delivery failure (above) is retried.
#[tokio::test]
async fn a_protocol_violation_is_one_get_not_the_retry_budget() {
    let (ops, state) = setup().await;
    let key = key("protocol");
    // A Content-Range on an unranged GET that does not describe the whole object.
    state.put_object(
        BUCKET,
        &key,
        MockObject::new(pattern(N, 21)).with_fault(
            None,
            1000,
            ObjectFault::ContentRange {
                first: 5,
                last: 8,
                total: N as u64,
            },
        ),
    );
    let mut dst = Dst::new(N);

    let err = read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect_err("a wrong Content-Range");

    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
    assert_eq!(gets(&state, &key).len(), 1, "a protocol violation is final");
    dst.assert_untouched();
}

/// Once pinned, a retry reply with no ETag cannot confirm the version: it fails (`Other`),
/// whatever the server did with `If-Match`.  The first response is partial, so without the
/// check the retry would be accepted and the read would report another version as success.
#[tokio::test]
async fn a_retry_response_with_no_etag_cannot_confirm_the_pin() {
    let (ops, state) = setup().await;
    let key = key("retry-no-etag");
    let object = MockObject::new(pattern(N, 22))
        .with_fault(None, 1, ObjectFault::TruncateBody { send: 6 })
        .with_fault(None, 1000, ObjectFault::OmitEtag);
    state.put_object(BUCKET, &key, object);

    let err = read(&ops, &key, &mut Dst::new(N), &ReceiveProgress::new())
        .await
        .expect_err("the retry cannot confirm the version");

    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
    assert!(format!("{err:#}").contains("cannot confirm"), "{err:#}");
    assert_eq!(gets(&state, &key).len(), 2, "not retried again");
}

/// The dangerous shape: the object is replaced between the attempts, the server ignores
/// `If-Match` AND sends no ETag.  The retry must not be read: nothing of the new version is
/// written, and the call is an error, never `Ok` with the other version's bytes.
#[tokio::test]
async fn a_replaced_object_answered_without_an_etag_is_never_read() {
    let (ops, state) = setup().await;
    let key = key("retry-replaced-no-etag");
    let mut object = MockObject::new(pattern(N, 23))
        .with_fault(None, 1, ObjectFault::TruncateBody { send: 6 })
        .with_fault(
            None,
            1,
            ObjectFault::OverwriteFirst {
                body: pattern(N, 97),
                etag: None,
            },
        );
    object.honour_if_match = false;
    state.put_object(BUCKET, &key, object);
    let mut dst = Dst::new(N);

    let err = read(&ops, &key, &mut dst, &ReceiveProgress::new())
        .await
        .expect_err("another version, unconfirmable");

    assert_eq!(kind_of(&err), ObjectErrorKind::Other);
    assert!(
        dst.bytes()[6..].iter().all(|b| *b == FILL),
        "the unconfirmed body was never read into the buffer"
    );
    dst.assert_canaries();
}
/// The trait method that wraps these futures is `Send` (`#[async_trait]`), so the futures of
/// both the single route and the fan-out entry points must be.
#[tokio::test]
async fn the_read_and_head_futures_are_send() {
    fn assert_send<T: Send>(_: &T) {}
    let ops = setup().await.0;
    let mut dst = Dst::new(N);
    let progress = ReceiveProgress::new();
    let get = ops.get_into(BUCKET, "never-polled", dst.slice(), &progress);
    assert_send(&get);
    let head = ops.head_typed(BUCKET, "never-polled");
    assert_send(&head);
}
