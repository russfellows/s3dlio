// src/concurrency/mod.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

pub mod scheduler;

pub use scheduler::{
    calculate_optimal_part_size, AdaptiveScheduler, ConcurrencyMode, S3PerformanceProfile,
    SchedulerPermit, Throughput, TransferDirection,
};

/// Reject a caller-supplied concurrency limit of 0 (issue #211).
///
/// A limit of 0 builds a semaphore (or a `buffer_unordered`) with no permits,
/// so the first acquire never completes and the call hangs forever.  Entry
/// points that take such a limit from the caller call this FIRST and return its
/// error; the value is never replaced by another one.  `name` is the parameter
/// as the caller spells it (for example `max_in_flight`).
pub fn require_concurrency_at_least_one(name: &str, value: usize) -> anyhow::Result<()> {
    if value == 0 {
        anyhow::bail!(
            "{name} must be an integer >= 1 (valid range: 1 or more), got 0: \
             a limit of 0 would allow no operation to start"
        );
    }
    Ok(())
}

#[cfg(test)]
mod require_concurrency_tests {
    use super::require_concurrency_at_least_one;

    #[test]
    fn zero_is_rejected_naming_the_parameter_and_range() {
        let err =
            require_concurrency_at_least_one("max_in_flight", 0).expect_err("0 must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max_in_flight"), "{msg}");
        assert!(msg.contains(">= 1"), "{msg}");
        assert!(msg.contains("got 0"), "{msg}");
    }

    #[test]
    fn one_and_larger_are_accepted() {
        for v in [1usize, 2, 64, usize::MAX] {
            require_concurrency_at_least_one("max_in_flight", v).expect("valid value");
        }
    }
}

/// Issue #211: every Rust entry point that takes a caller-supplied concurrency
/// limit rejects 0 immediately.  Each call runs under a time bound so that, if
/// a guard is ever removed, the test FAILS with "hung" instead of freezing the
/// test run.
#[cfg(test)]
mod zero_limit_rejections {
    use std::time::Duration;

    const BOUND: Duration = Duration::from_secs(10);

    /// Run a blocking call on its own thread; `None` means it did not return in time.
    fn run_bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        match rx.recv_timeout(BOUND) {
            Ok(v) => Some(v),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            // The sender is dropped without a value only when `f` panicked.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("the call PANICKED instead of returning an error")
            }
        }
    }

    async fn bounded<T>(fut: impl std::future::Future<Output = T>) -> Option<T> {
        tokio::time::timeout(BOUND, fut).await.ok()
    }

    fn assert_zero_rejected<T: std::fmt::Debug>(what: &str, got: Option<anyhow::Result<T>>) {
        let res = got.unwrap_or_else(|| panic!("{what}: HUNG with a limit of 0 (no error)"));
        let err = res.expect_err(&format!("{what}: a limit of 0 must be an error"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains(">= 1"),
            "{what}: message lacks the range: {msg}"
        );
    }

    #[test]
    fn get_objects_parallel_rejects_zero() {
        let got = run_bounded(|| crate::s3_utils::get_objects_parallel(&[], 0));
        assert_zero_rejected("get_objects_parallel", got);
    }

    #[test]
    fn get_objects_parallel_with_progress_rejects_zero() {
        let got = run_bounded(|| crate::s3_utils::get_objects_parallel_with_progress(&[], 0, None));
        assert_zero_rejected("get_objects_parallel_with_progress", got);
    }

    #[test]
    fn put_objects_parallel_rejects_zero() {
        let got = run_bounded(|| {
            crate::s3_utils::put_objects_parallel_with_progress(
                vec!["s3://b/k".to_string()],
                bytes::Bytes::from_static(b"x"),
                0,
                None,
            )
        });
        assert_zero_rejected("put_objects_parallel_with_progress", got);
    }

    #[test]
    fn put_objects_with_random_data_rejects_zero() {
        let got = run_bounded(|| {
            let config = crate::config::Config::new_with_defaults(
                crate::config::ObjectType::Raw,
                1,
                16,
                1,
                1,
            );
            crate::s3_utils::put_objects_with_random_data_and_type(
                &["s3://b/k".to_string()],
                16,
                0,
                config,
            )
        });
        assert_zero_rejected("put_objects_with_random_data_and_type", got);
    }

    #[tokio::test]
    async fn get_object_concurrent_range_rejects_explicit_zero() {
        let got = bounded(crate::s3_utils::get_object_concurrent_range_async(
            "s3://b/k",
            0,
            None,
            None,
            Some(0),
        ))
        .await;
        assert_zero_rejected("get_object_concurrent_range_async", got);
    }

    #[tokio::test]
    async fn s3_upload_files_rejects_zero() {
        let got = bounded(crate::s3_copy::upload_files(
            "s3://b/p/",
            &["/nonexistent/x"],
            0,
            false,
            None,
        ))
        .await;
        assert_zero_rejected("s3_copy::upload_files", got);
    }

    #[tokio::test]
    async fn s3_download_objects_rejects_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let got = bounded(crate::s3_copy::download_objects(
            "s3://b/p/",
            dir.path(),
            0,
            true,
            None,
        ))
        .await;
        assert_zero_rejected("s3_copy::download_objects", got);
    }

    #[tokio::test]
    async fn generic_upload_files_rejects_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("a.bin");
        std::fs::write(&src, b"x").expect("write");
        let dest = format!("file://{}/dest/", dir.path().display());
        let got = bounded(crate::object_store::generic_upload_files(
            &dest,
            &[src],
            0,
            None,
        ))
        .await;
        assert_zero_rejected("generic_upload_files", got);
    }

    #[tokio::test]
    async fn generic_download_objects_rejects_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.bin"), b"x").expect("write");
        let src = format!("file://{}/", dir.path().display());
        let out = tempfile::tempdir().expect("tempdir");
        let got = bounded(crate::object_store::generic_download_objects(
            &src,
            out.path(),
            0,
            true,
            None,
        ))
        .await;
        assert_zero_rejected("generic_download_objects", got);
    }

    #[tokio::test]
    async fn pre_stat_objects_rejects_zero_and_accepts_one() {
        use crate::object_store::ObjectStore;
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("a.bin");
        std::fs::write(&file, b"x").expect("write");
        let uri = format!("file://{}", file.display());
        let store = crate::file_store::FileSystemObjectStore::new();

        let got = bounded(store.pre_stat_objects(std::slice::from_ref(&uri), 0)).await;
        assert_zero_rejected("pre_stat_objects", got);

        let ok = bounded(store.pre_stat_objects(std::slice::from_ref(&uri), 1))
            .await
            .expect("a limit of 1 must complete")
            .expect("a limit of 1 must succeed");
        assert_eq!(ok.get(&uri), Some(&1u64));
    }

    #[tokio::test]
    async fn range_engine_rejects_zero_max_concurrent_ranges() {
        use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024,
            max_concurrent_ranges: 0,
            min_split_size: 1024,
            ..RangeEngineConfig::default()
        });
        let got = bounded(engine.download(
            4096,
            |_offset, length| async move { Ok(bytes::Bytes::from(vec![0u8; length as usize])) },
            None,
        ))
        .await;
        assert_zero_rejected("RangeEngine::download", got);
    }

    #[test]
    fn mp_get_rejects_zero_concurrent_per_proc() {
        let cfg = crate::mp::MpGetConfigBuilder::new()
            .concurrent_per_proc(0)
            .keylist("/nonexistent/keylist")
            .build();
        let got = run_bounded(move || crate::mp::run_get_shards(&cfg));
        assert_zero_rejected("run_get_shards", got);
    }

    /// A worker count of 0 used to panic with a division by zero while sharding
    /// the keys (needs at least one key).  It must be an error that names
    /// `procs`, raised before any temp directory or process exists.
    #[test]
    fn mp_get_rejects_zero_procs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let keylist = dir.path().join("keys.txt");
        std::fs::write(&keylist, "s3://b/k0\n").expect("write keylist");
        let cfg = crate::mp::MpGetConfigBuilder::new()
            .procs(0)
            .concurrent_per_proc(1)
            .keylist(keylist)
            .build();
        let got = run_bounded(move || crate::mp::run_get_shards(&cfg));
        let msg = got
            .as_ref()
            .and_then(|r| r.as_ref().err())
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(msg.contains("procs"), "error must name procs: {msg}");
        assert_zero_rejected("run_get_shards procs", got);
    }
}
