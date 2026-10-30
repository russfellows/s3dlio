// tests/get_into_panic_hook_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: the file stores must never touch the process's panic machinery. An earlier
// design probed `tokio::task::block_in_place` under `catch_unwind` with a process-wide chained
// panic hook; an application that installed its own hook (a crash reporter, a logger) AFTER the
// first read then saw one false panic per probe in every `LocalSet`.
//
// This binary reads and writes through `file://` in a `LocalSet` on a multi-thread runtime
// (first without any hook of ours), THEN installs the application hook, then does everything
// again in every context, and requires the hook to have been called ZERO times. A control panic
// proves the hook is live. Runs on plain runtimes (no worker marker) and on marked ones.

use s3dlio::object_store::{ObjectStore, ReceiveProgress, WriterOptions};
use s3dlio::FileSystemObjectStore;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

const LEN: usize = 40 * 1024 * 1024 + 5; // three 16 MiB chunks

static APP_HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);

fn pattern() -> Vec<u8> {
    (0..LEN).map(|i| (i % 251) as u8).collect()
}

async fn exercise(dir: &Path) {
    let data = pattern();
    let src = dir.join("src.bin");
    std::fs::write(&src, &data).unwrap();
    let store = FileSystemObjectStore::new();
    let mut dst = vec![0u8; LEN];
    store
        .get_into(
            &format!("file://{}", src.display()),
            &mut dst,
            &ReceiveProgress::new(),
        )
        .await
        .unwrap();
    assert!(dst == data);
    let out = dir.join("out.bin");
    let mut w = store
        .create_writer(
            &format!("file://{}", out.display()),
            WriterOptions::default(),
        )
        .await
        .unwrap();
    w.write_chunk(&data).await.unwrap();
    w.finalize().await.unwrap();
    assert!(std::fs::read(&out).unwrap() == data);
}

fn runtime(marked: bool) -> tokio::runtime::Runtime {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.worker_threads(2).enable_all();
    if marked {
        b.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread);
    }
    b.build().unwrap()
}

/// Every context that used to panic internally.
fn all_contexts(dir: &Path, marked: bool) {
    let rt = runtime(marked);
    // LocalSet::run_until in block_on.
    let local = tokio::task::LocalSet::new();
    rt.block_on(local.run_until(exercise(dir)));
    // LocalSet::block_on.
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, exercise(dir));
    // spawn_local.
    let local = tokio::task::LocalSet::new();
    let d = dir.to_path_buf();
    rt.block_on(local.run_until(async move {
        tokio::task::spawn_local(async move { exercise(&d).await })
            .await
            .unwrap()
    }));
    // A task on a worker.
    let d = dir.to_path_buf();
    rt.block_on(async move {
        tokio::spawn(async move { exercise(&d).await })
            .await
            .unwrap()
    });
}

#[test]
fn an_application_panic_hook_installed_after_the_first_read_sees_no_call() {
    let scratch = tempfile::tempdir().unwrap();
    // First reads, with the default hook still in place.
    all_contexts(scratch.path(), false);
    all_contexts(scratch.path(), true);

    // The application installs its hook now.
    std::panic::set_hook(Box::new(|_| {
        APP_HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
    }));
    for _ in 0..2 {
        all_contexts(scratch.path(), false);
        all_contexts(scratch.path(), true);
    }
    let calls = APP_HOOK_CALLS.load(Ordering::SeqCst);
    assert_eq!(
        calls, 0,
        "the file stores provoked {calls} panic(s) that an application hook saw"
    );

    // Control: the hook is live, a real panic is seen exactly once.
    let r = std::panic::catch_unwind(|| panic!("control"));
    assert!(r.is_err());
    assert_eq!(APP_HOOK_CALLS.load(Ordering::SeqCst), 1);
    let _ = std::panic::take_hook();
}
