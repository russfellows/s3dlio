// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Azure downloads and the object-size cache (#242 follow-up, Blue review 2026-10-15).
//!
//! `AzureObjectStore` caches the size a `HEAD` reported for 60 s. A blob that is overwritten
//! with a different size inside that window used to make the sequential download plan its
//! requests from the OLD size: a blob cached as 0 or 1 byte and then grown to 1 MiB became about
//! a million one-byte range requests, issued one after another.
//!
//! Everything runs against a local fake Blob service on http (`auto` sends no credential there).
//! The fake answers 403 after 300 requests, so a regression fails in milliseconds instead of
//! hanging. The tests share the process environment, so each holds `ENV_LOCK`.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use bytes::Bytes;
use s3dlio::object_store::{AzureConfig, AzureDownloadMode, AzureObjectStore, ObjectStore};

async fn setup() -> (tokio::sync::MutexGuard<'static, ()>, State) {
    let guard = ENV_LOCK.lock().await;
    clear_azure_env();
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    (guard, state)
}

fn store(mode: AzureDownloadMode, range_engine: bool) -> AzureObjectStore {
    let mut config = AzureConfig {
        download_mode: mode,
        enable_range_engine: range_engine,
        ..AzureConfig::default()
    };
    config.range_engine.min_split_size = 64 * 1024;
    config.range_engine.chunk_size = 256 * 1024;
    AzureObjectStore::with_config(config)
}

/// Read `key` once, then change it on the fake, then read it again inside the size-cache TTL.
async fn stale_then_change(
    state: &State,
    store: &AzureObjectStore,
    key: &str,
    first: usize,
    second: usize,
) {
    set_blob(state, key, pattern(first, 1));
    let got = store
        .get(&uri(key))
        .await
        .unwrap_or_else(|e| panic!("first read of {first} bytes: {e:#}"));
    assert_eq!(got.as_ref(), pattern(first, 1).as_slice());

    set_blob(state, key, pattern(second, 7));
    let before = mark(state);
    let got = store.get(&uri(key)).await.unwrap_or_else(|e| {
        panic!("read after the blob changed from {first} to {second} bytes: {e:#}")
    });
    assert_eq!(
        got.len(),
        second,
        "wrong-length body after the blob changed from {first} to {second}"
    );
    assert_eq!(got.as_ref(), pattern(second, 7).as_slice());
    let seen = requests_since(state, before);
    assert!(
        seen.len() <= 12,
        "RED: re-reading a blob that changed from {first} to {second} bytes made {} requests",
        seen.len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn downloads_do_not_plan_requests_from_a_stale_size() {
    let (_guard, state) = setup().await;

    // Sequential (explicit): the cached size is only a hint. 0 and 1 byte are the worst cases.
    let sequential = store(AzureDownloadMode::Sequential, false);
    stale_then_change(&state, &sequential, "seq-zero", 0, MIB + 12_345).await;
    stale_then_change(&state, &sequential, "seq-one", 1, MIB + 12_345).await;
    // ... and a blob that shrank.
    stale_then_change(&state, &sequential, "seq-shrink", 2 * MIB, 700).await;

    // The default (Auto, Azure SDK engine) does not even ask for the size.
    let default = AzureObjectStore::new();
    stale_then_change(&state, &default, "auto-one", 1, MIB + 12_345).await;
    stale_then_change(&state, &default, "auto-shrink", 2 * MIB, 700).await;

    // s3dlio's RangeEngine plans its ranges from the size: grown and shrunk blobs.
    let engine = store(AzureDownloadMode::Auto, true);
    stale_then_change(&state, &engine, "engine-grow", 128 * 1024, MIB + 12_345).await;
    stale_then_change(&state, &engine, "engine-shrink", MIB, 300 * 1024).await;
    stale_then_change(&state, &engine, "engine-to-small", MIB, 1000).await;

    // Default mode: no HEAD at all, one GET per small blob.
    let n = 5;
    for i in 0..n {
        set_blob(&state, &format!("plain-{i}"), pattern(1000 + i, 3));
    }
    let before = mark(&state);
    for i in 0..n {
        let got = default.get(&uri(&format!("plain-{i}"))).await.unwrap();
        assert_eq!(got.as_ref(), pattern(1000 + i, 3).as_slice());
    }
    let seen = requests_since(&state, before);
    assert_eq!(
        count(&seen, "HEAD"),
        0,
        "RED: the default mode asked for sizes first: {seen:?}"
    );
    assert_eq!(count(&seen, "GET"), n, "{seen:?}");
}

/// The RangeEngine trusts a cached size only with a guard: repeated reads of an unchanged blob
/// cost no HEAD (before the guard every GET started with one), and a changed blob is caught by
/// the guard and read correctly after one fresh HEAD.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn range_engine_reads_cost_no_head_while_the_size_is_unchanged() {
    let (_guard, state) = setup().await;
    let engine = store(AzureDownloadMode::Auto, true);
    let data = pattern(MIB, 2);
    set_blob(&state, "steady", data.clone());

    assert_eq!(
        engine.get(&uri("steady")).await.unwrap().as_ref(),
        data.as_slice()
    );
    let before = mark(&state);
    for _ in 0..4 {
        assert_eq!(
            engine.get(&uri("steady")).await.unwrap().as_ref(),
            data.as_slice()
        );
    }
    let seen = requests_since(&state, before);
    assert_eq!(
        count(&seen, "HEAD"),
        0,
        "RED: {} HEADs for 4 reads of an unchanged blob (one per read before the guard)",
        count(&seen, "HEAD")
    );
    // 4 reads x 4 chunks of 256 KiB.
    assert_eq!(count(&seen, "GET"), 16, "{seen:?}");

    // Grown: the guard (one byte past the planned end of the last range) sees it; one HEAD, one
    // retry, the right bytes.
    let grown = pattern(MIB + 777, 9);
    set_blob(&state, "steady", grown.clone());
    let before = mark(&state);
    assert_eq!(
        engine.get(&uri("steady")).await.unwrap().as_ref(),
        grown.as_slice()
    );
    assert_eq!(count(&requests_since(&state, before), "HEAD"), 1);

    // Shrunk: a range past the end is an error from the engine; same recovery.
    let shrunk = pattern(300 * 1024, 4);
    set_blob(&state, "steady", shrunk.clone());
    let before = mark(&state);
    assert_eq!(
        engine.get(&uri("steady")).await.unwrap().as_ref(),
        shrunk.as_slice()
    );
    assert_eq!(count(&requests_since(&state, before), "HEAD"), 1);

    // A blob that is really gone is an error, not a retry loop.
    state.lock().unwrap().blobs.remove("acct/ctr/steady");
    let before = mark(&state);
    assert!(engine.get(&uri("steady")).await.is_err());
    assert!(requests_since(&state, before).len() <= 8);
}

/// Writes and deletes through a store invalidate exactly the sizes they make stale.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_and_deletes_invalidate_only_the_affected_cached_sizes() {
    let (_guard, state) = setup().await;
    let sequential = store(AzureDownloadMode::Sequential, false);
    for key in ["inv/a", "inv/b", "inv/c", "other/d"] {
        set_blob(&state, key, pattern(500, 1));
        sequential.get(&uri(key)).await.unwrap(); // caches the size
    }

    let heads_reading = |key: &'static str| {
        let sequential = &sequential;
        let state = &state;
        async move {
            let before = mark(state);
            let _ = sequential.get(&uri(key)).await;
            count(&requests_since(state, before), "HEAD")
        }
    };

    // A cache hit costs no HEAD.
    assert_eq!(heads_reading("inv/a").await, 0, "unchanged blob");

    // put: the written URI is invalidated, its neighbours are not.
    sequential
        .put(&uri("inv/a"), Bytes::from(pattern(900, 2)))
        .await
        .unwrap();
    assert_eq!(heads_reading("inv/a").await, 1, "RED: after put");
    assert_eq!(heads_reading("inv/b").await, 0, "neighbour of a put");

    // delete: only the deleted URI.
    sequential.delete(&uri("inv/b")).await.unwrap();
    assert_eq!(
        heads_reading("inv/b").await,
        1,
        "RED: after delete (the deleted blob)"
    );
    assert_eq!(
        heads_reading("inv/c").await,
        0,
        "RED: neighbour of a delete"
    );
    assert_eq!(heads_reading("other/d").await, 0, "RED: another directory");

    // delete_batch: the named URIs.
    sequential.get(&uri("inv/a")).await.unwrap(); // re-cache
    sequential.delete_batch(&[uri("inv/c")]).await.unwrap();
    assert_eq!(
        heads_reading("inv/c").await,
        1,
        "RED: after delete_batch (the deleted blob)"
    );
    assert_eq!(
        heads_reading("inv/a").await,
        0,
        "RED: neighbour of a delete_batch"
    );

    // delete_prefix: the deleted keys, not the whole cache.
    set_blob(&state, "pre/x", pattern(100, 1));
    set_blob(&state, "pre/y", pattern(100, 1));
    sequential.get(&uri("pre/x")).await.unwrap();
    sequential.get(&uri("pre/y")).await.unwrap();
    sequential
        .delete_prefix("az://acct/ctr/pre/")
        .await
        .unwrap();
    assert_eq!(
        heads_reading("pre/x").await,
        1,
        "RED: after delete_prefix (a deleted blob)"
    );
    assert_eq!(
        heads_reading("other/d").await,
        0,
        "RED: outside the deleted prefix"
    );
}

/// A range read is one request however long the range is, and a huge requested length over a
/// small blob neither fails nor reserves that length. An open-ended range is one request up to
/// the default partition size.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn range_reads_are_one_request_and_do_not_reserve_the_requested_length() {
    let (_guard, state) = setup().await;
    let store = AzureObjectStore::new();
    let big = pattern(20 * MIB, 4);
    set_blob(&state, "range-big", big.clone());

    // 8 MiB: the SDK's default 4 MiB partition made this two serial requests.
    let before = mark(&state);
    let got = store
        .get_range(&uri("range-big"), MIB as u64, Some(8 * MIB as u64))
        .await
        .unwrap();
    assert_eq!(got.as_ref(), &big[MIB..9 * MIB]);
    let seen = requests_since(&state, before);
    assert_eq!(
        count(&seen, "GET"),
        1,
        "RED: an 8 MiB range took {} GETs: {seen:?}",
        count(&seen, "GET")
    );

    // Open-ended: the rest of a 20 MiB blob is one request (the default partition is 32 MiB),
    // not five 4 MiB ones.
    let before = mark(&state);
    let rest = store
        .get_range(&uri("range-big"), 2 * MIB as u64, None)
        .await
        .unwrap();
    assert_eq!(rest.as_ref(), &big[2 * MIB..]);
    let seen = requests_since(&state, before);
    assert_eq!(
        count(&seen, "GET"),
        1,
        "RED: an open-ended range took {} GETs: {seen:?}",
        count(&seen, "GET")
    );

    // The client-level API takes an inclusive end: the longest possible range must be an
    // error, not an arithmetic overflow.
    let base = std::env::var("AZURE_STORAGE_ENDPOINT").unwrap();
    let client = s3dlio::azure_client::AzureBlob::with_default_credential_from_url(
        &format!("{base}/acct"),
        "ctr",
    )
    .unwrap();
    let err = client
        .get_range("range-big", 0, Some(u64::MAX))
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("too long"), "{err:#}");
    let err = client
        .get_range("range-big", 10, Some(5))
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("precedes"), "{err:#}");

    // "To the end" idiom: a 1 TiB length over a small blob returns the blob.
    set_blob(&state, "range-small", pattern(1000, 8));
    let got = store
        .get_range(&uri("range-small"), 0, Some(1 << 40))
        .await
        .unwrap();
    assert_eq!(got.as_ref(), pattern(1000, 8).as_slice());
}
