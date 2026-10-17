// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `AzureBlob::get_into` and `AzureBlob::head_typed` against a local fake Blob service
//! (plan v0.10.0 §4.3, §4.5 A8, §9 "Providers -> Azure"; #264). No network beyond loopback.
//!
//! Everything runs against the fake in `common/azure_fake.rs` on http (`auto` sends no
//! credential there). The tests share the process environment, so each holds `ENV_LOCK`.
//! Each destination sits inside a buffer with canary bytes on both sides, so a write past
//! either end is caught.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::azure_client::AzureBlob;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::HeadMetadata;
use s3dlio::receive_progress::ReceiveProgress;

const CANARY: u8 = 0xEE;
const PAD: usize = 64;

async fn setup() -> (tokio::sync::MutexGuard<'static, ()>, State, AzureBlob) {
    let guard = ENV_LOCK.lock().await;
    clear_azure_env();
    std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    let client = AzureBlob::with_default_credential("acct", "ctr").expect("client");
    (guard, state, client)
}

/// A buffer with `len` bytes of destination between two canary pads.
fn canary_buf(len: usize) -> Vec<u8> {
    vec![CANARY; len + 2 * PAD]
}

fn canaries_intact(buf: &[u8], len: usize) -> bool {
    buf[..PAD].iter().all(|&b| b == CANARY) && buf[PAD + len..].iter().all(|&b| b == CANARY)
}

fn untouched(window: &[u8]) -> bool {
    window.iter().all(|&b| b == CANARY)
}

fn kind(err: &anyhow::Error) -> ObjectErrorKind {
    object_error(err)
        .unwrap_or_else(|| panic!("an untyped error: {err:#}"))
        .kind
}

fn gets(requests: &[Req]) -> Vec<&Req> {
    requests.iter().filter(|r| r.method == "GET").collect()
}

/// Read `key` into a fresh canary buffer of `len` bytes.
async fn read(
    client: &AzureBlob,
    key: &str,
    len: usize,
) -> (Vec<u8>, ReceiveProgress, anyhow::Result<()>) {
    let mut buf = canary_buf(len);
    let progress = ReceiveProgress::new();
    let result = client
        .get_into(key, &mut buf[PAD..PAD + len], &progress)
        .await;
    (buf, progress, result)
}

// ---- exact reads: one bounded request, nothing past the destination -----------------------------

/// An exact blob lands in the destination with one GET for the bounded range `0..len`, no HEAD,
/// and no write outside the destination. Sizes cross the SDK's 4 MiB default partition.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exact_blob_fills_the_destination_with_one_bounded_request() {
    let (_guard, state, client) = setup().await;
    for len in [1usize, 100, MIB + 13, 4 * MIB, 4 * MIB + 1, 9 * MIB + 7] {
        let key = format!("exact-{len}");
        set_blob(&state, &key, pattern(len, 3));
        let before = mark(&state);
        let (buf, progress, result) = read(&client, &key, len).await;
        result.unwrap_or_else(|e| panic!("{len}: {e:#}"));
        assert!(
            buf[PAD..PAD + len] == pattern(len, 3)[..],
            "{len}: wrong data"
        );
        assert!(canaries_intact(&buf, len), "{len}: wrote outside the dst");
        assert_eq!(progress.filled(), len as u64, "{len}");
        assert!(progress.first_byte().is_some());
        let seen = requests_since(&state, before);
        assert_eq!(count(&seen, "HEAD"), 0, "{len}: {seen:?}");
        let got = gets(&seen);
        assert_eq!(
            got.len(),
            1,
            "{len}: the SDK must not split a bounded range: {seen:?}"
        );
        assert_eq!(
            got[0].range.as_deref(),
            Some(format!("bytes=0-{}", len - 1).as_str()),
            "{len}"
        );
        assert_eq!(got[0].if_match, None, "the first request pins nothing");
    }
}

/// The destination is the caller's memory: a second read into the same allocation overwrites it
/// in place, and a stale value from the first read cannot survive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_destination_is_overwritten_in_place() {
    let (_guard, state, client) = setup().await;
    set_blob(&state, "twice", pattern(1000, 1));
    let mut buf = canary_buf(1000);
    let address = buf[PAD..].as_ptr();
    let progress = ReceiveProgress::new();
    client
        .get_into("twice", &mut buf[PAD..PAD + 1000], &progress)
        .await
        .expect("first");
    assert!(buf[PAD..PAD + 1000] == pattern(1000, 1)[..]);
    set_blob(&state, "twice", pattern(1000, 9));
    let progress = ReceiveProgress::new();
    client
        .get_into("twice", &mut buf[PAD..PAD + 1000], &progress)
        .await
        .expect("second");
    assert!(buf[PAD..PAD + 1000] == pattern(1000, 9)[..]);
    assert_eq!(address, buf[PAD..].as_ptr(), "the allocation never moved");
    assert_eq!(progress.filled(), 1000);
    assert!(canaries_intact(&buf, 1000));
}

/// A service that ignores `Range` answers 200 with the whole blob: `Content-Length` is the size.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_200_reply_to_a_ranged_request_is_read_by_content_length() {
    let (_guard, state, client) = setup().await;
    state.lock().unwrap().ignore_range = true;
    set_blob(&state, "whole", pattern(5000, 4));
    let (buf, progress, result) = read(&client, "whole", 5000).await;
    result.expect("200 with the exact length");
    assert!(buf[PAD..PAD + 5000] == pattern(5000, 4)[..]);
    assert!(canaries_intact(&buf, 5000));
    assert_eq!(progress.filled(), 5000);

    // A 200 carrying a different length is a mismatch decided before any copy.
    set_blob(&state, "whole-big", pattern(9000, 4));
    let (buf, progress, result) = read(&client, "whole-big", 5000).await;
    let err = result.expect_err("9000-byte blob, 5000-byte destination");
    assert_eq!(
        kind(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 5000,
            actual: Some(9000)
        }
    );
    assert!(untouched(&buf[PAD..PAD + 5000]) && canaries_intact(&buf, 5000));
    assert_eq!(progress.filled(), 0);
}

// ---- oversized, undersized and empty blobs -------------------------------------------------------

/// A blob longer or shorter than the destination is a `LengthMismatch` with the real size, and
/// the check comes BEFORE any copy: the destination is left untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oversized_and_undersized_blobs_are_length_mismatches_before_any_copy() {
    let (_guard, state, client) = setup().await;
    for (blob_len, dst_len) in [
        (100usize, 60usize),
        (60, 100),
        (MIB + 1, MIB),
        (MIB, MIB + 1),
        (1, 2),
    ] {
        let key = format!("m-{blob_len}-{dst_len}");
        set_blob(&state, &key, pattern(blob_len, 5));
        let before = mark(&state);
        let (buf, progress, result) = read(&client, &key, dst_len).await;
        let err = result.expect_err(&key);
        assert_eq!(
            kind(&err),
            ObjectErrorKind::LengthMismatch {
                expected: dst_len as u64,
                actual: Some(blob_len as u64)
            },
            "{key}"
        );
        assert!(
            untouched(&buf[PAD..PAD + dst_len]),
            "{key}: bytes were copied before the size check"
        );
        assert!(canaries_intact(&buf, dst_len), "{key}");
        assert_eq!(progress.filled(), 0, "{key}");
        assert!(progress.first_byte().is_none(), "{key}");
        let seen = requests_since(&state, before);
        assert_eq!(
            gets(&seen).len(),
            1,
            "{key}: a size mismatch is never retried: {seen:?}"
        );
    }
}

/// An empty blob read into a non-empty destination: the ranged request gets 416, the SDK retries
/// it unranged and gets 200 with length 0; that is a mismatch (`actual = 0`), not success.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_blob_is_a_mismatch_for_a_non_empty_destination() {
    let (_guard, state, client) = setup().await;
    set_blob(&state, "empty", Vec::new());
    let (buf, progress, result) = read(&client, "empty", 10).await;
    let err = result.expect_err("empty blob, 10-byte destination");
    assert_eq!(
        kind(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: Some(0)
        }
    );
    assert!(untouched(&buf[PAD..PAD + 10]) && canaries_intact(&buf, 10));
    assert_eq!(progress.filled(), 0);
}

/// `expected_len == 0` on an empty blob: a bounded `0..1` request gets 416, the unranged retry
/// gets 200 with length 0, and that is success with nothing counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_length_destination_accepts_an_empty_blob_via_the_416_path() {
    let (_guard, state, client) = setup().await;
    set_blob(&state, "empty", Vec::new());
    let before = mark(&state);
    let progress = ReceiveProgress::new();
    client
        .get_into("empty", &mut [], &progress)
        .await
        .expect("empty blob into an empty destination");
    assert_eq!(progress.filled(), 0);
    assert!(progress.first_byte().is_none());
    let seen = requests_since(&state, before);
    let got = gets(&seen);
    assert_eq!(got.len(), 2, "416, then the unranged retry: {seen:?}");
    assert_eq!(got[0].range.as_deref(), Some("bytes=0-0"));
    assert_eq!(got[1].range, None, "`bytes=0--1` is never sent");
}

/// `expected_len == 0` on a non-empty blob is a `LengthMismatch` carrying the blob's size, found
/// from the one-byte probe (it never downloads the blob).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_length_destination_rejects_a_non_empty_blob() {
    let (_guard, state, client) = setup().await;
    set_blob(&state, "full", pattern(5 * MIB, 2));
    let before = mark(&state);
    let progress = ReceiveProgress::new();
    let err = client
        .get_into("full", &mut [], &progress)
        .await
        .expect_err("non-empty blob into an empty destination");
    assert_eq!(
        kind(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: Some(5 * MIB as u64)
        }
    );
    assert_eq!(progress.filled(), 0);
    let seen = requests_since(&state, before);
    let got = gets(&seen);
    assert_eq!(got.len(), 1, "{seen:?}");
    assert_eq!(
        got[0].range.as_deref(),
        Some("bytes=0-0"),
        "a one-byte probe"
    );
}

/// A reply shorter than the range asked for (not the tail of the blob) cannot be trusted: it is
/// refused before any copy and is not retried.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_range_reply_shorter_than_asked_is_refused() {
    let (_guard, state, client) = setup().await;
    state.lock().unwrap().range_cap = Some(10);
    set_blob(&state, "short-range", pattern(100, 6));
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "short-range", 100).await;
    let err = result.expect_err("10 of 100 bytes");
    assert_eq!(kind(&err), ObjectErrorKind::Other);
    assert!(untouched(&buf[PAD..PAD + 100]) && canaries_intact(&buf, 100));
    assert_eq!(progress.filled(), 0);
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 1, "{seen:?}");
}

// ---- retries keep the first response's ETag --------------------------------------------------------

/// The body is cut after 10 000 bytes; the retry carries `If-Match: <first ETag>`, restarts the
/// cursor at 0 and completes. The count is each position once, not the bytes on the wire.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_after_a_cut_body_is_pinned_to_the_first_etag() {
    let (_guard, state, client) = setup().await;
    let len = 64 * 1024;
    set_blob(&state, "cut", pattern(len, 7));
    set_etag(&state, "cut", "\"etag-A\"");
    push_get_fault(&state, 10_000, AfterFault::Nothing);
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "cut", len).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + len] == pattern(len, 7)[..]);
    assert!(canaries_intact(&buf, len));
    assert_eq!(
        progress.filled(),
        len as u64,
        "10 000 + 65 536 bytes crossed the wire, but each position counts once"
    );
    let seen = requests_since(&state, before);
    let got = gets(&seen);
    assert_eq!(got.len(), 2, "{seen:?}");
    assert_eq!(got[0].if_match, None);
    assert_eq!(
        got[1].if_match.as_deref(),
        Some("\"etag-A\""),
        "the retry must be pinned to the first response's ETag"
    );
}

/// The blob is replaced (new ETag) after the cut: the pinned retry gets 412 and the read fails as
/// `VersionChanged`. Progress keeps the 10 000 positions that were written. Never retried again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_blob_replaced_between_attempts_is_a_version_change() {
    let (_guard, state, client) = setup().await;
    let len = 64 * 1024;
    set_blob(&state, "swap", pattern(len, 7));
    set_etag(&state, "swap", "\"etag-A\"");
    push_get_fault(
        &state,
        10_000,
        AfterFault::Replace(pattern(len, 8), "\"etag-B\"".to_string()),
    );
    let before = mark(&state);
    let (_buf, progress, result) = read(&client, "swap", len).await;
    let err = result.expect_err("the blob changed under the read");
    assert_eq!(kind(&err), ObjectErrorKind::VersionChanged);
    assert_eq!(progress.filled(), 10_000, "partial progress is kept");
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 2, "a version change is final: {seen:?}");
    assert_eq!(gets(&seen)[1].if_match.as_deref(), Some("\"etag-A\""));
}

/// A blob deleted after the cut: the pinned retry's 404 `BlobNotFound` is `VersionChanged`, never
/// absence (`NotFound` comes only from the first request or HEAD).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pinned_404_is_a_version_change_not_absence() {
    let (_guard, state, client) = setup().await;
    let len = 32 * 1024;
    set_blob(&state, "gone", pattern(len, 7));
    push_get_fault(&state, 5_000, AfterFault::Delete);
    let (_buf, progress, result) = read(&client, "gone", len).await;
    let err = result.expect_err("deleted after the first response");
    assert_eq!(kind(&err), ObjectErrorKind::VersionChanged);
    assert!(!s3dlio::delete_contract::is_not_found(&err));
    assert_eq!(progress.filled(), 5_000);
}

/// Without an ETag nothing is pinned: a body cut after bytes were written fails (typed
/// `Transport`) and is NOT retried from pieces of unknown versions. A weak ETag is no ETag.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_usable_etag_a_partial_receive_fails_instead_of_retrying() {
    let (_guard, state, client) = setup().await;
    let len = 32 * 1024;
    set_blob(&state, "noetag", pattern(len, 7));
    state.lock().unwrap().omit_etag = true;
    push_get_fault(&state, 5_000, AfterFault::Nothing);
    let before = mark(&state);
    let (_buf, progress, result) = read(&client, "noetag", len).await;
    let err = result.expect_err("cut body, no ETag");
    assert_eq!(kind(&err), ObjectErrorKind::Transport);
    assert_eq!(progress.filled(), 5_000, "progress survives the failure");
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 1, "no retry from pieces: {seen:?}");

    // The same with a weak ETag.
    state.lock().unwrap().omit_etag = false;
    set_blob(&state, "weak", pattern(len, 7));
    set_etag(&state, "weak", "W/\"weak-1\"");
    push_get_fault(&state, 5_000, AfterFault::Nothing);
    let before = mark(&state);
    let (_buf, progress, result) = read(&client, "weak", len).await;
    assert_eq!(
        kind(&result.expect_err("weak ETag")),
        ObjectErrorKind::Transport
    );
    assert_eq!(progress.filled(), 5_000);
    assert_eq!(gets(&requests_since(&state, before)).len(), 1);
}

/// A single response without an ETag is accepted (nothing was combined).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_single_response_without_an_etag_is_accepted() {
    let (_guard, state, client) = setup().await;
    state.lock().unwrap().omit_etag = true;
    set_blob(&state, "plain", pattern(4096, 1));
    let (buf, progress, result) = read(&client, "plain", 4096).await;
    result.expect("one unpinned response");
    assert!(buf[PAD..PAD + 4096] == pattern(4096, 1)[..]);
    assert_eq!(progress.filled(), 4096);
}

/// Unpinned, but the failure comes before the first byte was written: a retry cannot combine
/// anything, so it is allowed (and not pinned).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_an_etag_a_failure_before_any_byte_may_be_retried() {
    let (_guard, state, client) = setup().await;
    state.lock().unwrap().omit_etag = true;
    set_blob(&state, "early", pattern(8192, 2));
    push_get_fault(&state, 0, AfterFault::Nothing);
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "early", 8192).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + 8192] == pattern(8192, 2)[..]);
    assert_eq!(progress.filled(), 8192);
    let seen = requests_since(&state, before);
    let got = gets(&seen);
    assert_eq!(got.len(), 2, "{seen:?}");
    assert_eq!(got[1].if_match, None, "nothing to pin to");
}

/// A service that ignores `If-Match` and answers the retry from a replaced blob is caught by the
/// ETag on the reply itself: the read fails as `VersionChanged`, it is not completed from pieces
/// of two versions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_reply_with_a_different_etag_is_a_version_change() {
    let (_guard, state, client) = setup().await;
    let len = 32 * 1024;
    state.lock().unwrap().ignore_if_match = true;
    set_blob(&state, "ignored", pattern(len, 7));
    set_etag(&state, "ignored", "\"etag-A\"");
    push_get_fault(
        &state,
        4_000,
        AfterFault::Replace(pattern(len, 8), "\"etag-B\"".to_string()),
    );
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "ignored", len).await;
    let err = result.expect_err("the retry came from another version");
    assert_eq!(kind(&err), ObjectErrorKind::VersionChanged);
    assert_eq!(
        progress.filled(),
        4_000,
        "nothing from the second version was counted"
    );
    assert!(
        buf[PAD + 4_000..PAD + len].iter().all(|&b| b == CANARY),
        "no byte of the second version was written"
    );
    assert_eq!(gets(&requests_since(&state, before)).len(), 2);
}

/// A retry reply that carries no ETag cannot confirm the pinned version: the read fails (typed,
/// not retried) rather than trusting it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_reply_without_an_etag_cannot_confirm_the_pinned_version() {
    let (_guard, state, client) = setup().await;
    let len = 32 * 1024;
    set_blob(&state, "lost-etag", pattern(len, 7));
    set_etag(&state, "lost-etag", "\"etag-A\"");
    push_get_fault(&state, 4_000, AfterFault::OmitEtag);
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "lost-etag", len).await;
    let err = result.expect_err("unconfirmed retry");
    assert_eq!(kind(&err), ObjectErrorKind::Other);
    assert_eq!(progress.filled(), 4_000);
    assert!(buf[PAD + 4_000..PAD + len].iter().all(|&b| b == CANARY));
    assert_eq!(gets(&requests_since(&state, before)).len(), 2);
}

// ---- typed errors -------------------------------------------------------------------------------

/// A missing blob is `NotFound`, with one request; a missing container is `Other`: an error,
/// not absence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn blob_not_found_versus_container_not_found() {
    let (_guard, state, client) = setup().await;
    let before = mark(&state);
    let (_buf, progress, result) = read(&client, "no-such-blob", 10).await;
    let err = result.expect_err("missing blob");
    assert_eq!(kind(&err), ObjectErrorKind::NotFound);
    assert!(s3dlio::delete_contract::is_not_found(&err));
    assert_eq!(progress.filled(), 0);
    assert_eq!(
        gets(&requests_since(&state, before)).len(),
        1,
        "not retried"
    );

    state
        .lock()
        .unwrap()
        .missing_containers
        .insert("acct/ctr".to_string());
    let before = mark(&state);
    let (_buf, _progress, result) = read(&client, "anything", 10).await;
    let err = result.expect_err("missing container");
    assert_eq!(kind(&err), ObjectErrorKind::Other);
    assert!(!s3dlio::delete_contract::is_not_found(&err));
    assert_eq!(
        gets(&requests_since(&state, before)).len(),
        1,
        "not retried"
    );
}

/// 403 is `Forbidden` and never retried; the credential mode is named and a configured SAS token
/// stays out of every rendering of the error even when the service echoes the request URL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forbidden_names_the_credential_mode_and_redacts_the_sas() {
    let (_guard, state, client) = setup().await;
    // A client in `none` mode first (the default for http).
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let before = mark(&state);
    let (_buf, _progress, result) = read(&client, "k", 10).await;
    let err = result.expect_err("403");
    assert_eq!(kind(&err), ObjectErrorKind::Forbidden);
    assert!(
        format!("{err:#}").contains("Azure credential mode none"),
        "{err:#}"
    );
    assert_eq!(
        gets(&requests_since(&state, before)).len(),
        1,
        "not retried"
    );

    // Now SAS mode: the token must not leak.
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var(
        "S3DLIO_AZURE_SAS_TOKEN",
        "sv=2024-11-04&sig=SECRETSIGNATURE123",
    );
    let client = AzureBlob::with_default_credential("acct", "ctr").expect("sas client");
    let (_buf, _progress, result) = read(&client, "k", 10).await;
    let err = result.expect_err("403 with SAS");
    assert_eq!(kind(&err), ObjectErrorKind::Forbidden);
    for text in [format!("{err:#}"), format!("{err:?}"), err.to_string()] {
        assert!(!text.contains("SECRETSIGNATURE123"), "SAS leaked: {text}");
    }
    assert!(
        format!("{err:#}").contains("Azure credential mode sas"),
        "{err:#}"
    );
}

/// The error text of a cut body names no secret and no URL query.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_body_stream_error_is_a_redacted_transport_error() {
    let (_guard, state, client) = setup().await;
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var("S3DLIO_AZURE_SAS_TOKEN", "sv=2024-11-04&sig=BODYSECRET456");
    let client2 = AzureBlob::with_default_credential("acct", "ctr").expect("sas client");
    drop(client);
    set_blob(&state, "cutsas", pattern(20_000, 1));
    state.lock().unwrap().omit_etag = true;
    push_get_fault(&state, 100, AfterFault::Nothing);
    let (_buf, progress, result) = read(&client2, "cutsas", 20_000).await;
    let err = result.expect_err("cut");
    assert_eq!(kind(&err), ObjectErrorKind::Transport);
    assert_eq!(progress.filled(), 100);
    for text in [format!("{err:#}"), format!("{err:?}")] {
        assert!(!text.contains("BODYSECRET456"), "SAS leaked: {text}");
    }
}

// ---- typed HEAD ------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_reports_size_etag_and_version() {
    let (_guard, state, client) = setup().await;
    set_blob(&state, "h", pattern(1234, 1));
    set_etag(&state, "h", "\"etag-h\"");
    set_version_id(&state, "h", "2026-10-17T10:00:00.1234567Z");
    let before = mark(&state);
    let head = client
        .head_typed("h")
        .await
        .expect("head")
        .expect("present");
    assert_eq!(
        head,
        HeadMetadata {
            size: 1234,
            etag: Some("\"etag-h\"".to_string()),
            version_id: Some("2026-10-17T10:00:00.1234567Z".to_string()),
        }
    );
    let seen = requests_since(&state, before);
    assert_eq!((seen.len(), count(&seen, "HEAD")), (1, 1), "{seen:?}");

    // No version id and no ETag: both are None, never invented.
    set_blob(&state, "h2", pattern(7, 1));
    state.lock().unwrap().omit_etag = true;
    let head = client
        .head_typed("h2")
        .await
        .expect("head")
        .expect("present");
    assert_eq!(
        head,
        HeadMetadata {
            size: 7,
            etag: None,
            version_id: None
        }
    );
}

/// An empty blob has size 0 (present), which is different from an absent blob.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_of_an_empty_blob_is_present_with_size_zero() {
    let (_guard, state, client) = setup().await;
    set_blob(&state, "e", Vec::new());
    let head = client
        .head_typed("e")
        .await
        .expect("head")
        .expect("present");
    assert_eq!(head.size, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_absence_is_only_blob_not_found() {
    let (_guard, state, client) = setup().await;
    assert_eq!(
        client.head_typed("absent").await.expect("absent is Ok"),
        None
    );

    // A missing container is an error (Other), not absence.
    state
        .lock()
        .unwrap()
        .missing_containers
        .insert("acct/ctr".to_string());
    let err = client
        .head_typed("absent")
        .await
        .expect_err("missing container");
    assert_eq!(kind(&err), ObjectErrorKind::Other);
    assert!(!s3dlio::delete_contract::is_not_found(&err));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_forbidden_is_a_typed_error_naming_the_mode() {
    let (_guard, state, client) = setup().await;
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let err = client.head_typed("k").await.expect_err("403");
    assert_eq!(kind(&err), ObjectErrorKind::Forbidden);
    assert!(
        format!("{err:#}").contains("Azure credential mode none"),
        "{err:#}"
    );
}
