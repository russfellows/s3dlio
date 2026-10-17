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

/// An exact blob lands in the destination with one GET for the bounded range `0..len + 1` (one
/// byte more than the destination, review C-F1), no HEAD, and no write outside the destination.
/// The reply is `len` bytes because the blob ends there. Sizes cross the SDK's 4 MiB default partition.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exact_blob_fills_the_destination_with_one_bounded_request() {
    let (env_lock, state, client) = setup().await;
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
            Some(format!("bytes=0-{len}").as_str()),
            "{len}"
        );
        assert_eq!(got[0].if_match, None, "the first request pins nothing");
    }
    drop(env_lock);
}

/// The destination is the caller's memory: a second read into the same allocation overwrites it
/// in place, and a stale value from the first read cannot survive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_destination_is_overwritten_in_place() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// A service that ignores `Range` answers 200 with the whole blob: `Content-Length` is the size.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_200_reply_to_a_ranged_request_is_read_by_content_length() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

// ---- oversized, undersized and empty blobs -------------------------------------------------------

/// A blob longer or shorter than the destination is a `LengthMismatch` with the real size, and
/// the check comes BEFORE any copy: the destination is left untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oversized_and_undersized_blobs_are_length_mismatches_before_any_copy() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// An empty blob read into a non-empty destination: the ranged request gets 416, the SDK retries
/// it unranged and gets 200 with length 0; that is a mismatch (`actual = 0`), not success.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_blob_is_a_mismatch_for_a_non_empty_destination() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// `expected_len == 0` on an empty blob: a bounded `0..1` request gets 416, the unranged retry
/// gets 200 with length 0, and that is success with nothing counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_length_destination_accepts_an_empty_blob_via_the_416_path() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// `expected_len == 0` on a non-empty blob is a `LengthMismatch` carrying the blob's size, found
/// from the one-byte probe (it never downloads the blob).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_length_destination_rejects_a_non_empty_blob() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// A reply shorter than the range asked for (not the tail of the blob) cannot be trusted: it is
/// refused before any copy and is not retried.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_range_reply_shorter_than_asked_is_refused() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

// ---- retries keep the first response's ETag --------------------------------------------------------

/// The body is cut after 10 000 bytes; the retry carries `If-Match: <first ETag>`, restarts the
/// cursor at 0 and completes. The count is each position once, not the bytes on the wire.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_after_a_cut_body_is_pinned_to_the_first_etag() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// The blob is replaced (new ETag) after the cut: the pinned retry gets 412 and the read fails as
/// `VersionChanged`. Progress keeps the 10 000 positions that were written. Never retried again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_blob_replaced_between_attempts_is_a_version_change() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    set_blob(&state, "swap", pattern(len, 7));
    set_etag(&state, "swap", "\"etag-A\"");
    push_get_fault(
        &state,
        10_000,
        AfterFault::Replace(pattern(len, 8), "\"etag-B\"".to_string()),
    );
    let before = mark(&state);
    let (_, progress, result) = read(&client, "swap", len).await;
    let err = result.expect_err("the blob changed under the read");
    assert_eq!(kind(&err), ObjectErrorKind::VersionChanged);
    assert_eq!(progress.filled(), 10_000, "partial progress is kept");
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 2, "a version change is final: {seen:?}");
    assert_eq!(gets(&seen)[1].if_match.as_deref(), Some("\"etag-A\""));
    drop(env_lock);
}

/// A blob deleted after the cut: the pinned retry's 404 `BlobNotFound` is `VersionChanged`, never
/// absence (`NotFound` comes only from the first request or HEAD).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pinned_404_is_a_version_change_not_absence() {
    let (env_lock, state, client) = setup().await;
    let len = 32 * 1024;
    set_blob(&state, "gone", pattern(len, 7));
    push_get_fault(&state, 5_000, AfterFault::Delete);
    let (_, progress, result) = read(&client, "gone", len).await;
    let err = result.expect_err("deleted after the first response");
    assert_eq!(kind(&err), ObjectErrorKind::VersionChanged);
    assert!(!s3dlio::delete_contract::is_not_found(&err));
    assert_eq!(progress.filled(), 5_000);
    drop(env_lock);
}

/// Without an ETag nothing is pinned: a body cut after bytes were written fails (typed
/// `Transport`) and is NOT retried from pieces of unknown versions. A weak ETag is no ETag.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_usable_etag_a_partial_receive_fails_instead_of_retrying() {
    let (env_lock, state, client) = setup().await;
    let len = 32 * 1024;
    set_blob(&state, "noetag", pattern(len, 7));
    state.lock().unwrap().omit_etag = true;
    push_get_fault(&state, 5_000, AfterFault::Nothing);
    let before = mark(&state);
    let (_, progress, result) = read(&client, "noetag", len).await;
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
    let (_, progress, result) = read(&client, "weak", len).await;
    assert_eq!(
        kind(&result.expect_err("weak ETag")),
        ObjectErrorKind::Transport
    );
    assert_eq!(progress.filled(), 5_000);
    assert_eq!(gets(&requests_since(&state, before)).len(), 1);
    drop(env_lock);
}

/// A single response without an ETag is accepted (nothing was combined).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_single_response_without_an_etag_is_accepted() {
    let (env_lock, state, client) = setup().await;
    state.lock().unwrap().omit_etag = true;
    set_blob(&state, "plain", pattern(4096, 1));
    let (buf, progress, result) = read(&client, "plain", 4096).await;
    result.expect("one unpinned response");
    assert!(buf[PAD..PAD + 4096] == pattern(4096, 1)[..]);
    assert_eq!(progress.filled(), 4096);
    drop(env_lock);
}

/// Unpinned, but the failure comes before the first byte was written: a retry cannot combine
/// anything, so it is allowed (and not pinned).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_an_etag_a_failure_before_any_byte_may_be_retried() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// A service that ignores `If-Match` and answers the retry from a replaced blob is caught by the
/// ETag on the reply itself: the read fails as `VersionChanged`, it is not completed from pieces
/// of two versions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_reply_with_a_different_etag_is_a_version_change() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// A retry reply that carries no ETag cannot confirm the pinned version: the read fails (typed,
/// not retried) rather than trusting it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retry_reply_without_an_etag_cannot_confirm_the_pinned_version() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

// ---- typed errors -------------------------------------------------------------------------------

/// A missing blob is `NotFound`, with one request; a missing container is `Other`: an error,
/// not absence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn blob_not_found_versus_container_not_found() {
    let (env_lock, state, client) = setup().await;
    let before = mark(&state);
    let (_, progress, result) = read(&client, "no-such-blob", 10).await;
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
    let (_, _, result) = read(&client, "anything", 10).await;
    let err = result.expect_err("missing container");
    assert_eq!(kind(&err), ObjectErrorKind::Other);
    assert!(!s3dlio::delete_contract::is_not_found(&err));
    assert_eq!(
        gets(&requests_since(&state, before)).len(),
        1,
        "not retried"
    );
    drop(env_lock);
}

/// 403 is `Forbidden` and never retried; the credential mode is named and a configured SAS token
/// stays out of every rendering of the error even when the service echoes the request URL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forbidden_names_the_credential_mode_and_redacts_the_sas() {
    let (env_lock, state, client) = setup().await;
    // A client in `none` mode first (the default for http).
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let before = mark(&state);
    let (_, _, result) = read(&client, "k", 10).await;
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
    let (_, _, result) = read(&client, "k", 10).await;
    let err = result.expect_err("403 with SAS");
    assert_eq!(kind(&err), ObjectErrorKind::Forbidden);
    for text in [format!("{err:#}"), format!("{err:?}"), err.to_string()] {
        assert!(!text.contains("SECRETSIGNATURE123"), "SAS leaked: {text}");
    }
    assert!(
        format!("{err:#}").contains("Azure credential mode sas"),
        "{err:#}"
    );
    drop(env_lock);
}

/// The error text of a cut body names no secret and no URL query.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_body_stream_error_is_a_redacted_transport_error() {
    let (env_lock, state, client) = setup().await;
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var("S3DLIO_AZURE_SAS_TOKEN", "sv=2024-11-04&sig=BODYSECRET456");
    let client2 = AzureBlob::with_default_credential("acct", "ctr").expect("sas client");
    drop(client);
    set_blob(&state, "cutsas", pattern(20_000, 1));
    state.lock().unwrap().omit_etag = true;
    push_get_fault(&state, 100, AfterFault::Nothing);
    let (_, progress, result) = read(&client2, "cutsas", 20_000).await;
    let err = result.expect_err("cut");
    assert_eq!(kind(&err), ObjectErrorKind::Transport);
    assert_eq!(progress.filled(), 100);
    for text in [format!("{err:#}"), format!("{err:?}")] {
        assert!(!text.contains("BODYSECRET456"), "SAS leaked: {text}");
    }
    drop(env_lock);
}

// ---- typed HEAD ------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_reports_size_etag_and_version() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// An empty blob has size 0 (present), which is different from an absent blob.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_of_an_empty_blob_is_present_with_size_zero() {
    let (env_lock, state, client) = setup().await;
    set_blob(&state, "e", Vec::new());
    let head = client
        .head_typed("e")
        .await
        .expect("head")
        .expect("present");
    assert_eq!(head.size, 0);
    drop(env_lock);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_absence_is_only_blob_not_found() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_forbidden_is_a_typed_error_naming_the_mode() {
    let (env_lock, state, client) = setup().await;
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
    drop(env_lock);
}

/// The two futures are `Send`: the trait methods that will call them are `#[async_trait]`, and a
/// future that holds the destination must be able to move between threads with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_entry_point_futures_are_send() {
    fn is_send<T: Send>(value: T) -> T {
        value
    }
    let (env_lock, state, client) = setup().await;
    set_blob(&state, "s", pattern(100, 1));
    let mut dst = vec![0u8; 100];
    let progress = ReceiveProgress::new();
    is_send(client.get_into("s", &mut dst, &progress))
        .await
        .expect("get_into");
    assert!(is_send(client.head_typed("s"))
        .await
        .expect("head")
        .is_some());
    drop(env_lock);
}

// ---- retry coverage: statuses, repeated failures, the attempt budget ----------------------------
//
// Two retry layers act on a request. The SDK's own pipeline retries 408, 429, 500, 502, 503 and
// 504 and every connection or I/O failure (default `max_retries` 8, exponential back-off from
// 200 ms, 60 s in all) INSIDE one `get_into` attempt, and the same request, `If-Match` included,
// is sent again. `get_into` then retries what the SDK gave up on (up to
// `S3DLIO_MAX_RETRY_ATTEMPTS` attempts), pinned to the first response's ETag. The client builder
// offers no way to turn the SDK layer off, so the tests that need a deterministic request count
// use statuses the SDK does not retry (501, 507: typed `Server`), and the SDK-retried ones
// (429, 503, a dropped connection) are single faults the SDK absorbs on its first retry.

/// The `If-Match` of every GET after the first.
fn retry_if_matches(requests: &[Req]) -> Vec<Option<&str>> {
    gets(requests)
        .iter()
        .skip(1)
        .map(|r| r.if_match.as_deref())
        .collect()
}

/// A 429 or 503 that arrives on a pinned retry is retried again (by the SDK pipeline, which
/// resends the same request) and the resend still carries the first response's ETag.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_throttle_status_on_a_pinned_retry_keeps_the_first_etag() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    for (status, code) in [(429u16, "ServerBusy"), (503, "ServerBusy")] {
        let key = format!("throttle-{status}");
        set_blob(&state, &key, pattern(len, 7));
        set_etag(&state, &key, "\"etag-A\"");
        push_get_fault(&state, 10_000, AfterFault::Nothing);
        push_get_status(&state, status, Some(code));
        let before = mark(&state);
        let (buf, progress, result) = read(&client, &key, len).await;
        result.unwrap_or_else(|e| panic!("{status}: {e:#}"));
        assert!(buf[PAD..PAD + len] == pattern(len, 7)[..], "{status}");
        assert!(canaries_intact(&buf, len), "{status}");
        assert_eq!(progress.filled(), len as u64, "{status}");
        let seen = requests_since(&state, before);
        assert_eq!(
            gets(&seen).len(),
            3,
            "{status}: the cut, the throttled retry and its resend: {seen:?}"
        );
        assert_eq!(gets(&seen)[0].if_match, None, "{status}");
        assert_eq!(
            retry_if_matches(&seen),
            vec![Some("\"etag-A\""); 2],
            "{status}: every retry carries the first ETag"
        );
    }
    drop(env_lock);
}

/// A 503 before any response has been seen pins nothing: the SDK resends the same unconditional
/// request and the read completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unpinned_503_is_absorbed_by_the_sdk_retry() {
    let (env_lock, state, client) = setup().await;
    set_blob(&state, "busy", pattern(4096, 3));
    push_get_status(&state, 503, Some("ServerBusy"));
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "busy", 4096).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + 4096] == pattern(4096, 3)[..]);
    assert_eq!(progress.filled(), 4096);
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 2, "{seen:?}");
    assert_eq!(retry_if_matches(&seen), vec![None]);
    drop(env_lock);
}

/// A server error the SDK does not retry (501, 507) reaches `get_into` as a typed `Server`
/// failure, which it retries itself, still pinned to the first ETag.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_server_error_after_a_cut_is_retried_by_get_into_with_the_first_etag() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    set_blob(&state, "srv", pattern(len, 7));
    set_etag(&state, "srv", "\"etag-A\"");
    push_get_fault(&state, 10_000, AfterFault::Nothing);
    push_get_status(&state, 507, Some("InsufficientAccountPermissions"));
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "srv", len).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + len] == pattern(len, 7)[..]);
    assert!(canaries_intact(&buf, len));
    assert_eq!(progress.filled(), len as u64);
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 3, "{seen:?}");
    assert_eq!(retry_if_matches(&seen), vec![Some("\"etag-A\""); 2]);
    drop(env_lock);
}

/// Two cuts in a row, then success: EVERY retry carries the FIRST response's ETag (not the
/// previous attempt's, not none), and each position is counted once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_retry_after_repeated_cuts_carries_the_first_etag() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    set_blob(&state, "twice-cut", pattern(len, 7));
    set_etag(&state, "twice-cut", "\"etag-A\"");
    push_get_fault(&state, 5_000, AfterFault::Nothing);
    push_get_fault(&state, 8_000, AfterFault::Nothing);
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "twice-cut", len).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + len] == pattern(len, 7)[..]);
    assert!(canaries_intact(&buf, len));
    assert_eq!(progress.filled(), len as u64);
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 3, "{seen:?}");
    assert_eq!(gets(&seen)[0].if_match, None);
    assert_eq!(retry_if_matches(&seen), vec![Some("\"etag-A\""); 2]);
    drop(env_lock);
}

/// The attempt budget (`S3DLIO_MAX_RETRY_ATTEMPTS`) is spent exactly: a body that is cut every
/// time ends with the typed `Transport` error after exactly that many GETs, all retries pinned,
/// and progress keeps what the attempts wrote.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_cuts_end_with_the_typed_error_after_exactly_the_budget() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    for budget in 1usize..=3 {
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", budget.to_string());
        let key = format!("exhaust-cut-{budget}");
        set_blob(&state, &key, pattern(len, 7));
        set_etag(&state, &key, "\"etag-A\"");
        for _ in 0..=budget {
            push_get_fault(&state, 5_000, AfterFault::Nothing);
        }
        let before = mark(&state);
        let (_, progress, result) = read(&client, &key, len).await;
        let err = result.expect_err("every body is cut");
        assert_eq!(kind(&err), ObjectErrorKind::Transport, "budget {budget}");
        assert_eq!(progress.filled(), 5_000, "budget {budget}");
        let seen = requests_since(&state, before);
        assert_eq!(gets(&seen).len(), budget, "budget {budget}: {seen:?}");
        assert_eq!(
            retry_if_matches(&seen),
            vec![Some("\"etag-A\""); budget - 1],
            "budget {budget}"
        );
        state.lock().unwrap().get_faults.clear();
    }
    drop(env_lock);
}

/// The same for a status: a server error the SDK does not retry, every time, ends with the typed
/// `Server` error after exactly the budget. (A status the SDK DOES retry is not exhausted here:
/// the SDK would send 9 requests per attempt over about 80 s of back-off.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_server_errors_end_with_the_typed_error_after_exactly_the_budget() {
    let (env_lock, state, client) = setup().await;
    set_blob(&state, "always-507", pattern(4096, 7));
    for budget in 1usize..=3 {
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", budget.to_string());
        for _ in 0..=budget {
            push_get_status(&state, 507, Some("InsufficientAccountPermissions"));
        }
        let before = mark(&state);
        let (buf, progress, result) = read(&client, "always-507", 4096).await;
        let err = result.expect_err("every reply is 507");
        assert_eq!(kind(&err), ObjectErrorKind::Server, "budget {budget}");
        assert!(untouched(&buf[PAD..PAD + 4096]) && canaries_intact(&buf, 4096));
        assert_eq!(progress.filled(), 0);
        let seen = requests_since(&state, before);
        assert_eq!(gets(&seen).len(), budget, "budget {budget}: {seen:?}");
        assert!(
            retry_if_matches(&seen).iter().all(Option::is_none),
            "nothing was seen to pin to: {seen:?}"
        );
        state.lock().unwrap().get_faults.clear();
    }
    drop(env_lock);
}

/// A request-level failure (the connection is closed after the request, no reply): the SDK
/// retries it (a connection or I/O error) and the read completes without any bytes counted twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connection_dropped_before_any_reply_is_survived() {
    let (env_lock, state, client) = setup().await;
    set_blob(&state, "dropped", pattern(4096, 5));
    push_get_drop(&state);
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "dropped", 4096).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + 4096] == pattern(4096, 5)[..]);
    assert!(canaries_intact(&buf, 4096));
    assert_eq!(progress.filled(), 4096);
    let seen = requests_since(&state, before);
    assert_eq!(
        gets(&seen).len(),
        2,
        "the dropped request and its resend: {seen:?}"
    );
    assert_eq!(retry_if_matches(&seen), vec![None]);
    drop(env_lock);
}

/// A connection dropped on a pinned retry: the resend still carries the first ETag.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connection_dropped_on_a_pinned_retry_keeps_the_first_etag() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    set_blob(&state, "dropped-pinned", pattern(len, 7));
    set_etag(&state, "dropped-pinned", "\"etag-A\"");
    push_get_fault(&state, 10_000, AfterFault::Nothing);
    push_get_drop(&state);
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "dropped-pinned", len).await;
    result.unwrap_or_else(|e| panic!("{e:#}"));
    assert!(buf[PAD..PAD + len] == pattern(len, 7)[..]);
    assert_eq!(progress.filled(), len as u64);
    let seen = requests_since(&state, before);
    assert_eq!(gets(&seen).len(), 3, "{seen:?}");
    assert_eq!(retry_if_matches(&seen), vec![Some("\"etag-A\""); 2]);
    drop(env_lock);
}

// ---- cancellation through a real SDK body stream ---------------------------------------------------

/// The service sends 10 000 bytes of the body and then stalls with the connection open. Dropping
/// `get_into` (a timeout) stops the read: progress says exactly which positions were written, the
/// bytes are in place, and nothing outside the destination changed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_get_into_on_a_stalled_body_keeps_what_was_written() {
    let (env_lock, state, client) = setup().await;
    let len = 64 * 1024;
    let stalled_at = 10_000;
    set_blob(&state, "stalled", pattern(len, 7));
    push_get_stall(&state, stalled_at);
    let mut buf = canary_buf(len);
    let progress = ReceiveProgress::new();
    let outcome = tokio::time::timeout(
        std::time::Duration::from_millis(1500),
        client.get_into("stalled", &mut buf[PAD..PAD + len], &progress),
    )
    .await;
    assert!(
        outcome.is_err(),
        "the read must still be waiting for the rest"
    );
    assert_eq!(progress.filled(), stalled_at as u64);
    assert!(progress.first_byte().is_some());
    assert!(buf[PAD..PAD + stalled_at] == pattern(len, 7)[..stalled_at]);
    assert!(untouched(&buf[PAD + stalled_at..PAD + len]));
    assert!(canaries_intact(&buf, len));
    drop(env_lock);
}

// ---- a 206 that does not say which part it carries (review C-F1) ------------------------------------

/// REVIEW C-F1. The SDK discards the status, so a 206 stripped of `Content-Range` looks like a
/// whole-blob reply. The request therefore asks for ONE BYTE MORE than the destination
/// (`bytes=0-N`), so a larger blob exposes at least N+1 bytes and its `Content-Length` (N+1) can
/// no longer equal the expected size N. The read is a `LengthMismatch`, refused from the headers
/// before any copy: no progress, an untouched destination, and the extra byte is never written.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_206_without_content_range_for_a_larger_blob_is_a_length_mismatch() {
    let (env_lock, state, client) = setup().await;
    state.lock().unwrap().strip_content_range = true;
    set_blob(&state, "no-range-header", pattern(100, 6));
    let before = mark(&state);
    let (buf, progress, result) = read(&client, "no-range-header", 60).await;
    let err = result.expect_err("a 60-byte read of a 100-byte blob must not succeed");
    assert!(
        matches!(
            kind(&err),
            ObjectErrorKind::LengthMismatch { expected: 60, .. }
        ),
        "{err:#}"
    );
    assert_eq!(progress.filled(), 0, "refused before any copy");
    assert!(progress.first_byte().is_none());
    assert!(
        untouched(&buf[PAD..PAD + 60]),
        "the destination is untouched"
    );
    assert!(
        canaries_intact(&buf, 60),
        "the extra byte was written outside the dst"
    );
    let seen = requests_since(&state, before);
    let got = gets(&seen);
    assert_eq!(got.len(), 1, "a definite answer is not retried: {seen:?}");
    assert_eq!(
        got[0].range.as_deref(),
        Some("bytes=0-60"),
        "N + 1 bytes are requested"
    );
    drop(env_lock);
}

/// The same stripped header on a blob of exactly the expected length still succeeds (the reply
/// is N bytes, which ends the blob), and a blob shorter than expected is a mismatch.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_206_without_content_range_is_judged_by_the_length_it_carries() {
    let (env_lock, state, client) = setup().await;
    state.lock().unwrap().strip_content_range = true;
    set_blob(&state, "exact", pattern(60, 6));
    let (buf, progress, result) = read(&client, "exact", 60).await;
    result.expect("an exact blob ends at N even with the header stripped");
    assert!(buf[PAD..PAD + 60] == pattern(60, 6)[..]);
    assert!(canaries_intact(&buf, 60));
    assert_eq!(progress.filled(), 60);

    set_blob(&state, "short", pattern(40, 6));
    let (buf, progress, result) = read(&client, "short", 60).await;
    let err = result.expect_err("a 40-byte blob into a 60-byte destination");
    assert_eq!(
        kind(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 60,
            actual: Some(40)
        }
    );
    assert!(untouched(&buf[PAD..PAD + 60]) && canaries_intact(&buf, 60));
    assert_eq!(progress.filled(), 0);
    drop(env_lock);
}

// ---- literal keys -----------------------------------------------------------------------------------------

/// A key is a literal name, never a pattern or an already-encoded path: the service receives
/// exactly that name (the fake decodes the request path once), for the read and for HEAD.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn literal_keys_reach_the_service_unchanged() {
    let (env_lock, state, client) = setup().await;
    for key in ["a%2Fb", "100%", "x@y", "k=v", "p/q/r", "name."] {
        set_blob(&state, key, pattern(321, 9));
        let before = mark(&state);
        let (buf, progress, result) = read(&client, key, 321).await;
        result.unwrap_or_else(|e| panic!("{key}: {e:#}"));
        assert!(buf[PAD..PAD + 321] == pattern(321, 9)[..], "{key}");
        assert_eq!(progress.filled(), 321, "{key}");
        let head = client
            .head_typed(key)
            .await
            .unwrap_or_else(|e| panic!("{key}: {e:#}"))
            .unwrap_or_else(|| panic!("{key}: absent"));
        assert_eq!(head.size, 321, "{key}");
        let seen = requests_since(&state, before);
        assert_eq!(seen.len(), 2, "{key}: {seen:?}");
        let expected = format!("acct/ctr/{key}");
        for request in &seen {
            assert_eq!(request.path, expected, "{key}: {request:?}");
        }
        // A near neighbour is a different blob, not the same one under another spelling.
        let (_, _, result) = read(&client, &format!("{key}x"), 321).await;
        assert_eq!(
            kind(&result.expect_err("no such blob")),
            ObjectErrorKind::NotFound,
            "{key}"
        );
    }
    drop(env_lock);
}

// ---- HEAD: an empty ETag header is no ETag ---------------------------------------------------------------

/// An `ETag` header that is present but empty (or only blanks) is reported as `None`, never as
/// `Some("")`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_normalises_an_empty_etag_to_none() {
    let (env_lock, state, client) = setup().await;
    for (key, etag) in [("empty-etag", ""), ("blank-etag", "  ")] {
        set_blob(&state, key, pattern(10, 1));
        set_etag(&state, key, etag);
        let head = client
            .head_typed(key)
            .await
            .expect("head")
            .expect("present");
        assert_eq!(head.etag, None, "{key}");
        assert_eq!(head.size, 10, "{key}");
    }
    drop(env_lock);
}
