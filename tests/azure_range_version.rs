// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! One object version per logical Azure RangeEngine GET (Blue stage G review, item G-F2).
//!
//! The RangeEngine reads a blob as several independent ranged requests. Each range used to be
//! unpinned: replacing the blob with another of the SAME length between two of the ranges made
//! a successful GET whose body was part one version and part the other, and neither the
//! cached-size guard (the length is unchanged) nor the per-range retry pinning (it pins one
//! range to itself) could see it.
//!
//! Every test here replaces the blob on the fake service just before a chosen GET is answered,
//! reads through `AzureObjectStore::get` in explicit `s3dlio` mode, and inspects EVERY request
//! the fake saw. A result is acceptable if it is an error, or the complete bytes of ONE version:
//! a consistently pinned old version and a restart on the new one are both fine, a mix is not.
//! The tests share the process environment, so each holds `ENV_LOCK`.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::object_store::{AzureConfig, AzureDownloadMode, AzureObjectStore, ObjectStore};

const ETAG_A: &str = "\"version-a\"";
const ETAG_B: &str = "\"version-b\"";
/// 1 MiB in 256 KiB ranges: four ranges, all started together.
const LEN: usize = MIB;

async fn setup() -> (tokio::sync::MutexGuard<'static, ()>, State) {
    let guard = ENV_LOCK.lock().await;
    clear_azure_env();
    // The SDK's own retry layer is not what is under test; keep any fault visible.
    std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    (guard, state)
}

/// Explicit s3dlio mode with small ranges, so a 1 MiB blob is read as four ranged requests.
fn engine_store() -> AzureObjectStore {
    let mut config = AzureConfig {
        download_mode: AzureDownloadMode::S3dlio,
        enable_range_engine: false,
        ..AzureConfig::default()
    };
    config.range_engine.min_split_size = 64 * 1024;
    config.range_engine.chunk_size = 256 * 1024;
    AzureObjectStore::with_config(config)
}

fn gets(requests: &[Req]) -> Vec<&Req> {
    requests.iter().filter(|r| r.method == "GET").collect()
}

/// The result is an error or the complete bytes of exactly one of the two versions.
fn assert_not_mixed(result: &anyhow::Result<bytes::Bytes>, a: &[u8], b: &[u8], seen: &[Req]) {
    if let Ok(body) = result {
        assert!(
            body.as_ref() == a || body.as_ref() == b,
            "RED: a successful GET returned bytes of two versions (first mismatch with A at {:?}, \
             with B at {:?}); requests: {seen:?}",
            body.iter().zip(a).position(|(x, y)| x != y),
            body.iter().zip(b).position(|(x, y)| x != y),
        );
    }
}

/// The blob is replaced by another of the same length, with another ETag, as the SECOND GET is
/// answered: the first range saw version A, every other range would see B. No success may mix
/// them, and every request after the first must name version A in `If-Match`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_same_size_replacement_between_ranges_is_never_a_mixed_body() {
    let (_guard, state) = setup().await;
    let store = engine_store();
    let a = pattern(LEN, 1);
    let b = pattern(LEN, 77);
    set_blob(&state, "swap", a.clone());
    set_etag(&state, "swap", ETAG_A);
    replace_before_get(&state, 2, b.clone(), ETAG_B);

    let result = store.get(&uri("swap")).await;
    let seen = requests_since(&state, 0);
    assert_not_mixed(&result, &a, &b, &seen);

    let ranged = gets(&seen);
    assert!(
        ranged.len() >= 2,
        "the blob is read as several ranges: {seen:?}"
    );
    assert_eq!(
        ranged[0].if_match, None,
        "RED: the first range establishes the version and carries no condition: {seen:?}"
    );
    for (i, request) in ranged.iter().enumerate().skip(1) {
        assert_eq!(
            request.if_match.as_deref(),
            Some(ETAG_A),
            "RED: request {} of a logical GET is not pinned to the first response's ETag: {seen:?}",
            i + 1
        );
    }
    // Pinned to A while the service holds B: the siblings are refused, the GET is an error.
    let err = result.expect_err("the pinned version was replaced, so the read must fail");
    assert!(
        format!("{err:#}").contains("version changed"),
        "the error says the version changed: {err:#}"
    );
}

/// An unchanged blob is still read in full, one request per range, the first unconditional and
/// every other one pinned to the ETag the first reported; no extra request is added.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unchanged_blob_costs_one_request_per_range_all_pinned_to_the_first_etag() {
    let (_guard, state) = setup().await;
    let store = engine_store();
    let a = pattern(LEN, 3);
    set_blob(&state, "steady", a.clone());
    set_etag(&state, "steady", ETAG_A);

    let got = store.get(&uri("steady")).await.expect("read");
    assert_eq!(got.as_ref(), a.as_slice());
    let seen = requests_since(&state, 0);
    assert_eq!(count(&seen, "HEAD"), 1, "{seen:?}");
    let ranged = gets(&seen);
    assert_eq!(ranged.len(), 4, "{seen:?}");
    assert_eq!(ranged[0].if_match, None, "{seen:?}");
    assert!(
        ranged[1..]
            .iter()
            .all(|r| r.if_match.as_deref() == Some(ETAG_A)),
        "{seen:?}"
    );

    // The size is now cached (guarded read, no HEAD): the pinning is the same.
    let before = mark(&state);
    let again = store.get(&uri("steady")).await.expect("cached-size read");
    assert_eq!(again.as_ref(), a.as_slice());
    let seen = requests_since(&state, before);
    assert_eq!(count(&seen, "HEAD"), 0, "{seen:?}");
    assert_eq!(gets(&seen).len(), 4, "{seen:?}");
    assert!(
        gets(&seen)[1..]
            .iter()
            .all(|r| r.if_match.as_deref() == Some(ETAG_A)),
        "{seen:?}"
    );
}

/// A blob replaced by another of the same size BEFORE the read (the size is cached) is read
/// completely from the new version: the first range reports the new ETag and the rest follow it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replacement_before_a_cached_size_read_returns_the_whole_new_version() {
    let (_guard, state) = setup().await;
    let store = engine_store();
    let a = pattern(LEN, 5);
    let b = pattern(LEN, 99);
    set_blob(&state, "between-reads", a.clone());
    set_etag(&state, "between-reads", ETAG_A);
    assert_eq!(
        store
            .get(&uri("between-reads"))
            .await
            .expect("first")
            .as_ref(),
        a.as_slice()
    );

    set_blob(&state, "between-reads", b.clone());
    set_etag(&state, "between-reads", ETAG_B);
    let before = mark(&state);
    let got = store
        .get(&uri("between-reads"))
        .await
        .expect("read of the replacement");
    assert_eq!(got.as_ref(), b.as_slice());
    let seen = requests_since(&state, before);
    assert!(
        gets(&seen)[1..]
            .iter()
            .all(|r| r.if_match.as_deref() == Some(ETAG_B)),
        "{seen:?}"
    );
}

/// The same replacement during a read that used a CACHED size: the pinned ranges are refused, the
/// existing fresh-size retry starts the whole read over, and the restart is of the new version
/// from its first byte to its last (a complete restart is as valid as an error).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replacement_during_a_cached_size_read_restarts_on_the_new_version() {
    let (_guard, state) = setup().await;
    let store = engine_store();
    let a = pattern(LEN, 21);
    let b = pattern(LEN, 150);
    set_blob(&state, "restart", a.clone());
    set_etag(&state, "restart", ETAG_A);
    assert_eq!(
        store.get(&uri("restart")).await.expect("first").as_ref(),
        a.as_slice()
    );

    // The second read is guarded by the cached size: its first range sees A, the second GET of
    // that read (the 6th GET overall: the first read made four) sees B.
    let first_read_gets = state.lock().unwrap().gets_seen;
    replace_before_get(&state, first_read_gets + 2, b.clone(), ETAG_B);
    let before = mark(&state);
    let got = store.get(&uri("restart")).await.expect("restarted read");
    assert_eq!(
        got.as_ref(),
        b.as_slice(),
        "RED: the restarted read is not the complete new version"
    );
    let seen = requests_since(&state, before);
    assert_eq!(
        count(&seen, "HEAD"),
        1,
        "one fresh-size HEAD for the restart: {seen:?}"
    );
    // After the HEAD every GET belongs to the restart: the first names no version, the rest B.
    let head_at = seen.iter().position(|r| r.method == "HEAD").unwrap();
    let restart = gets(&seen[head_at..]);
    assert_eq!(restart.len(), 4, "{seen:?}");
    assert_eq!(restart[0].if_match, None, "{seen:?}");
    assert!(
        restart[1..]
            .iter()
            .all(|r| r.if_match.as_deref() == Some(ETAG_B)),
        "{seen:?}"
    );
}

/// A blob smaller than the split size is ONE request: nothing to pin, no condition sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_blob_below_the_split_size_is_one_unconditional_request() {
    let (_guard, state) = setup().await;
    let store = engine_store();
    let a = pattern(10_000, 9);
    set_blob(&state, "small", a.clone());
    set_etag(&state, "small", ETAG_A);
    let got = store.get(&uri("small")).await.expect("read");
    assert_eq!(got.as_ref(), a.as_slice());
    let seen = requests_since(&state, 0);
    assert_eq!(gets(&seen).len(), 1, "{seen:?}");
    assert_eq!(gets(&seen)[0].if_match, None, "{seen:?}");
}

/// Without a strong ETag nothing can pin the ranges, and unpinned ranges are never combined: the
/// blob is read again as ONE response. A replacement as the second GET is answered therefore
/// yields the whole new version (or an error), never a mix, and no sibling range is requested.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_an_etag_the_blob_is_read_as_one_response() {
    let (_guard, state) = setup().await;
    let store = engine_store();
    let a = pattern(LEN, 11);
    let b = pattern(LEN, 123);
    set_blob(&state, "no-etag", a.clone());
    state.lock().unwrap().omit_etag = true;
    replace_before_get(&state, 2, b.clone(), ETAG_B);

    let result = store.get(&uri("no-etag")).await;
    let seen = requests_since(&state, 0);
    assert_not_mixed(&result, &a, &b, &seen);
    let ranged = gets(&seen);
    assert!(
        ranged.iter().all(|r| r.if_match.is_none()),
        "no ETag, no condition: {seen:?}"
    );
    // After the first range found no usable ETag, every later request starts at byte 0: the
    // engine's sibling ranges (offset 256 KiB, 512 KiB, 768 KiB) are never issued.
    for request in &ranged[1..] {
        let range = request.range.as_deref().unwrap_or("bytes=0-");
        assert!(
            range.starts_with("bytes=0-"),
            "RED: a sibling range was requested without any way to pin it: {range} in {seen:?}"
        );
    }
    let body = result.expect("the one-response read succeeds");
    assert_eq!(
        body.as_ref(),
        b.as_slice(),
        "the single response is the new version"
    );
}
