// tests/test_zero_limits_direct_entry_points_211.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Issue #211 follow-up: a caller limit of 0 is refused by EVERY entry point that takes
// one, including the ones added after #211 was written and the ones a wrapper guard does
// not cover because the caller reaches the shared implementation directly.
//
//   * `s3_copy::download_objects_matching` (the S3 download implementation behind
//     `download_objects`, with an explicit regex) and
//     `generic_download_objects_matching_with_summary` (the backend-aware download behind
//     `s3-cli download|get --pattern`): `max_in_flight == 0` builds a semaphore with no
//     permits, so the first download waits forever.
//   * the concurrent range GET: `get_object_concurrent_range_async` (process-wide client)
//     and the per-endpoint store's `get_range_optimized` (its own client, issue #208) -- a
//     `max_concurrency` of `Some(0)` hangs, a `chunk_size` of `Some(0)` never advances the
//     range-splitting loop (unbounded memory).
//
// Every call is bounded by a timeout, so a regression FAILS ("HUNG") instead of hanging
// the run. For each zero the error must name the parameter and ">= 1", the in-process
// mock S3 must have seen NO request for the test's (unique) bucket, and no destination
// directory may have been created.

#[allow(dead_code)]
mod common;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use common::mock_s3_server::{ensure_mock_s3_server, MockS3State};
use s3dlio::list_pattern::ListPattern;
use s3dlio::object_store::{
    generic_download_objects_matching_with_summary, ObjectStore, S3ObjectStore,
};

const BOUND: Duration = Duration::from_secs(10);

static N: AtomicU64 = AtomicU64::new(0);

/// A bucket name used by exactly one test, so `requests_containing` returns only its traffic.
fn unique_bucket(tag: &str) -> String {
    format!(
        "zero211-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    )
}

/// A directory path that does not exist yet.
fn fresh_dir(tag: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "zero211-{}-{}-{tag}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&p);
    p
}

/// Await `fut`, failing the test (instead of hanging it) when it does not finish in time.
async fn bounded<T>(what: &str, fut: impl std::future::Future<Output = T>) -> T {
    match tokio::time::timeout(BOUND, fut).await {
        Ok(v) => v,
        Err(_) => panic!("{what}: HUNG with a limit of 0 (no error within {BOUND:?})"),
    }
}

fn assert_rejected(what: &str, param: &str, res: anyhow::Result<impl std::fmt::Debug>) {
    let err = res.expect_err(&format!("{what}: a limit of 0 must be an error"));
    let msg = format!("{err:#}");
    assert!(
        msg.contains(param),
        "{what}: message must name `{param}`: {msg}"
    );
    assert!(
        msg.contains(">= 1"),
        "{what}: message must state `>= 1`: {msg}"
    );
}

fn assert_no_traffic(what: &str, state: &MockS3State, bucket: &str) {
    let seen = state.requests_containing(bucket);
    assert!(
        seen.is_empty(),
        "{what}: the mock saw {} request(s) for {bucket}: {seen:?}",
        seen.len()
    );
}

// ----------------------------------------------------------------------------
// A. the shared download implementations
// ----------------------------------------------------------------------------

#[tokio::test]
async fn s3_copy_download_objects_matching_rejects_zero() {
    let (_port, state) = ensure_mock_s3_server();
    for pattern in [None, Some(r"\.npz$")] {
        let bucket = unique_bucket("s3copy");
        let dest = fresh_dir("s3copy");
        let res = bounded(
            "s3_copy::download_objects_matching",
            s3dlio::s3_copy::download_objects_matching(
                &format!("s3://{bucket}/dl/"),
                pattern,
                &dest,
                0,
                true,
                None,
            ),
        )
        .await;
        assert_rejected("s3_copy::download_objects_matching", "max_in_flight", res);
        assert_no_traffic("s3_copy::download_objects_matching", &state, &bucket);
        assert!(
            !dest.exists(),
            "the destination directory must not be created"
        );
    }
}

#[tokio::test]
async fn generic_download_matching_rejects_zero_on_s3() {
    let (_port, state) = ensure_mock_s3_server();
    let pat = ListPattern::new(r"\.npz$").expect("pattern");
    for pattern in [None, Some(&pat)] {
        let bucket = unique_bucket("gen-s3");
        let dest = fresh_dir("gen-s3");
        let res = bounded(
            "generic_download_objects_matching_with_summary",
            generic_download_objects_matching_with_summary(
                &format!("s3://{bucket}/dl/"),
                &dest,
                0,
                true,
                pattern,
                None,
            ),
        )
        .await;
        assert_rejected(
            "generic_download_objects_matching_with_summary",
            "max_in_flight",
            res,
        );
        assert_no_traffic(
            "generic_download_objects_matching_with_summary",
            &state,
            &bucket,
        );
        assert!(
            !dest.exists(),
            "the destination directory must not be created"
        );
    }
}

/// The per-endpoint (host:port) form of the same URI: no request reaches that server.
#[tokio::test]
async fn generic_download_matching_rejects_zero_on_per_endpoint_uri() {
    let (port, state) = ensure_mock_s3_server();
    let bucket = unique_bucket("gen-ep");
    let dest = fresh_dir("gen-ep");
    let res = bounded(
        "generic_download_objects_matching_with_summary (per-endpoint)",
        generic_download_objects_matching_with_summary(
            &format!("s3://127.0.0.1:{port}/{bucket}/dl/"),
            &dest,
            0,
            true,
            None,
            None,
        ),
    )
    .await;
    assert_rejected(
        "generic_download_objects_matching_with_summary",
        "max_in_flight",
        res,
    );
    assert_no_traffic(
        "generic_download_objects_matching_with_summary",
        &state,
        &bucket,
    );
    assert!(
        !dest.exists(),
        "the destination directory must not be created"
    );
}

/// A `file://` source: nothing is listed or created either.
#[tokio::test]
async fn generic_download_matching_rejects_zero_on_file_source() {
    let src = fresh_dir("gen-file-src");
    std::fs::create_dir_all(&src).expect("src dir");
    std::fs::write(src.join("a.npz"), b"x").expect("seed");
    let dest = fresh_dir("gen-file-dest");
    let res = bounded(
        "generic_download_objects_matching_with_summary (file://)",
        generic_download_objects_matching_with_summary(
            &format!("file://{}/", src.display()),
            &dest,
            0,
            true,
            None,
            None,
        ),
    )
    .await;
    assert_rejected(
        "generic_download_objects_matching_with_summary",
        "max_in_flight",
        res,
    );
    assert!(
        !dest.exists(),
        "the destination directory must not be created"
    );
    let _ = std::fs::remove_dir_all(&src);
}

// ----------------------------------------------------------------------------
// B. the concurrent range GET: global wrapper and per-endpoint store
// ----------------------------------------------------------------------------

/// (chunk_size, max_concurrency, the parameter the error must name)
const ZERO_RANGE_LIMITS: [(Option<usize>, Option<usize>, &str); 3] = [
    (None, Some(0), "max_concurrency"),
    (Some(0), None, "chunk_size"),
    (Some(0), Some(0), "max_concurrency"),
];

#[tokio::test]
async fn global_concurrent_range_wrapper_rejects_zero_limits() {
    let (_port, state) = ensure_mock_s3_server();
    for (chunk, conc, param) in ZERO_RANGE_LIMITS {
        let bucket = unique_bucket("glob-range");
        let res = bounded(
            "get_object_concurrent_range_async",
            s3dlio::s3_utils::get_object_concurrent_range_async(
                &format!("s3://{bucket}/obj"),
                0,
                None,
                chunk,
                conc,
            ),
        )
        .await;
        assert_rejected("get_object_concurrent_range_async", param, res);
        assert_no_traffic("get_object_concurrent_range_async", &state, &bucket);
    }
}

/// The per-endpoint store reads through `get_object_concurrent_range_with_client`; a zero must
/// be refused there too, for a transfer above the concurrent threshold (the helper is used)
/// and below it (the plain range GET is used and used to ignore the limits).
#[tokio::test]
async fn per_endpoint_get_range_optimized_rejects_zero_limits() {
    let (port, state) = ensure_mock_s3_server();
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("for_endpoint with the mock's fake credentials");
    // 1 GiB is far above the 32 MiB default threshold; 1 KiB is far below it.
    for length in [Some(1u64 << 30), Some(1024), None] {
        for (chunk, conc, param) in ZERO_RANGE_LIMITS {
            let bucket = unique_bucket("ep-range");
            let uri = format!("s3://127.0.0.1:{port}/{bucket}/obj");
            let res = bounded(
                "S3ObjectStore::get_range_optimized (per-endpoint)",
                store.get_range_optimized(&uri, 0, length, chunk, conc),
            )
            .await;
            assert_rejected(
                &format!("per-endpoint get_range_optimized (length {length:?})"),
                param,
                res,
            );
            assert_no_traffic("per-endpoint get_range_optimized", &state, &bucket);
        }
    }
}

/// The same refusal on a store that holds no client of its own (the process-wide path).
#[tokio::test]
async fn global_store_get_range_optimized_rejects_zero_limits() {
    let (_port, state) = ensure_mock_s3_server();
    let store = S3ObjectStore::new();
    for length in [Some(1u64 << 30), Some(1024), None] {
        for (chunk, conc, param) in ZERO_RANGE_LIMITS {
            let bucket = unique_bucket("gstore-range");
            let uri = format!("s3://{bucket}/obj");
            let res = bounded(
                "S3ObjectStore::get_range_optimized (global)",
                store.get_range_optimized(&uri, 0, length, chunk, conc),
            )
            .await;
            assert_rejected(
                &format!("global get_range_optimized (length {length:?})"),
                param,
                res,
            );
            assert_no_traffic("global get_range_optimized", &state, &bucket);
        }
    }
}
