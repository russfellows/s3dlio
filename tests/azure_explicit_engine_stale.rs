// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Explicit `S3DLIO_AZURE_DOWNLOAD_MODE=s3dlio` and a stale cached size (Blue final review F1).
//!
//! The store caches the size a `HEAD` reported. With the s3dlio RangeEngine a blob smaller than
//! `min_split_size` is read as ONE request, and that branch used to accept a response of any
//! length: a blob cached as 1,024 bytes and then grown to 1 MiB came back as 1,025 bytes in a
//! successful GET, and a blob cached as 0 bytes and then grown came back empty with no GET at
//! all. A success never reaches the fresh-size retry, so the caller got a truncated object.
//!
//! Every case here reads a blob (caching its size), changes it on the fake service, reads it
//! again inside the cache TTL, and requires the complete current bytes after exactly one fresh
//! `HEAD`. The tests share the process environment, so each holds `ENV_LOCK`.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::object_store::{AzureConfig, AzureDownloadMode, AzureObjectStore, ObjectStore};

async fn setup() -> (tokio::sync::MutexGuard<'static, ()>, State) {
    let guard = ENV_LOCK.lock().await;
    clear_azure_env();
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    (guard, state)
}

/// Explicit s3dlio mode, so every size is planned by the RangeEngine; objects under 64 KiB take
/// its single-request branch.
fn explicit_engine_store() -> AzureObjectStore {
    let mut config = AzureConfig {
        download_mode: AzureDownloadMode::S3dlio,
        enable_range_engine: false,
        ..AzureConfig::default()
    };
    config.range_engine.min_split_size = 64 * 1024;
    config.range_engine.chunk_size = 256 * 1024;
    AzureObjectStore::with_config(config)
}

/// Read `key` (`first` bytes), change it to `second` bytes, read it again inside the size-cache
/// TTL: the complete new bytes and exactly one fresh HEAD are required.
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
    let seen = requests_since(state, before);
    assert_eq!(
        got.len(),
        second,
        "RED: a blob that changed from {first} to {second} bytes was returned with {} bytes as a success: {seen:?}",
        got.len()
    );
    assert_eq!(got.as_ref(), pattern(second, 7).as_slice());
    assert_eq!(
        count(&seen, "HEAD"),
        1,
        "RED: no fresh-size retry after the blob changed from {first} to {second} bytes: {seen:?}"
    );
    assert!(seen.len() <= 12, "{} requests: {seen:?}", seen.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_zero_that_grew_is_refreshed_not_returned_empty() {
    let (_guard, state) = setup().await;
    let store = explicit_engine_store();
    // Grown to a small blob (single request) and to a large one (concurrent ranges).
    stale_then_change(&state, &store, "zero-to-small", 0, 1000).await;
    stale_then_change(&state, &store, "zero-to-large", 0, MIB + 12_345).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_small_size_that_grew_returns_the_whole_blob() {
    let (_guard, state) = setup().await;
    let store = explicit_engine_store();
    // Blue's trigger: 1,024 bytes cached, 1 MiB now, previously 1,025 bytes as a success.
    stale_then_change(&state, &store, "grow-to-1mib", 1024, MIB).await;
    // Sub-threshold to sub-threshold.
    stale_then_change(&state, &store, "grow-small", 1024, 2000).await;
    // One byte of growth.
    stale_then_change(&state, &store, "grow-one", 1024, 1025).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_small_size_that_shrank_is_refreshed() {
    let (_guard, state) = setup().await;
    let store = explicit_engine_store();
    stale_then_change(&state, &store, "shrink-small", 50_000, 700).await;
    stale_then_change(&state, &store, "shrink-to-one", 2000, 1).await;
    stale_then_change(&state, &store, "shrink-to-zero", 2000, 0).await;
}

/// An unchanged small blob is still read with one GET and no HEAD, and an object that really is
/// empty is a legitimate fresh zero: empty, with no GET.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unchanged_blobs_cost_no_extra_head_and_empty_blobs_stay_empty() {
    let (_guard, state) = setup().await;
    let store = explicit_engine_store();
    set_blob(&state, "steady", pattern(1000, 3));
    assert_eq!(
        store.get(&uri("steady")).await.unwrap().as_ref(),
        pattern(1000, 3).as_slice()
    );
    let before = mark(&state);
    for _ in 0..4 {
        assert_eq!(
            store.get(&uri("steady")).await.unwrap().as_ref(),
            pattern(1000, 3).as_slice()
        );
    }
    let seen = requests_since(&state, before);
    assert_eq!(count(&seen, "HEAD"), 0, "{seen:?}");
    assert_eq!(count(&seen, "GET"), 4, "{seen:?}");

    // A fresh zero (nothing cached) is a legitimate empty read.
    set_blob(&state, "empty", Vec::new());
    let before = mark(&state);
    let got = store.get(&uri("empty")).await.unwrap();
    assert!(got.is_empty());
    assert_eq!(count(&requests_since(&state, before), "GET"), 0);
    // A cached zero that is still zero is empty after the refresh.
    let before = mark(&state);
    let got = store.get(&uri("empty")).await.unwrap();
    assert!(got.is_empty());
    assert_eq!(count(&requests_since(&state, before), "HEAD"), 1);
}
