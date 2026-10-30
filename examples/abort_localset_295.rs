// examples/abort_localset_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267 regression program: the file stores must work from a tokio `LocalSet` on a
// multi-thread runtime in a binary built with `panic = "abort"` (a downstream crate's release
// profile). A `block_in_place` that panics there (and is expected to be caught) aborts the
// process with SIGABRT instead. Run it through `scripts/test_panic_abort_295.sh`, which builds
// it with the `abort-test` profile (panic = "abort") and checks the exit status.
//
// Exits 0 and prints `OK` when every context works; any failure panics (= aborts here).

use bytes::Bytes;
use s3dlio::object_store::{ObjectStore, ReceiveProgress, WriterOptions};
use s3dlio::FileSystemObjectStore;

const LEN: usize = 40 * 1024 * 1024 + 5; // three 16 MiB read chunks

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

async fn exercise(dir: &std::path::Path) {
    let data = pattern(LEN);
    let src = dir.join("src.bin");
    std::fs::write(&src, &data).unwrap();
    let store = FileSystemObjectStore::new();

    // get_into (blocking_io per chunk)
    let mut dst = vec![0u8; LEN];
    let progress = ReceiveProgress::new();
    store
        .get_into(&format!("file://{}", src.display()), &mut dst, &progress)
        .await
        .unwrap();
    assert!(dst == data, "get_into content");

    // borrowed-slice write (write_borrowed_blocking)
    let out = format!("file://{}", dir.join("out.bin").display());
    let mut w = store
        .create_writer(&out, WriterOptions::default())
        .await
        .unwrap();
    w.write_chunk(&data).await.unwrap();
    w.finalize().await.unwrap();
    assert!(
        std::fs::read(dir.join("out.bin")).unwrap() == data,
        "put content"
    );
    let _ = Bytes::new();
}

fn main() {
    #[cfg(panic = "abort")]
    println!("panic=abort build");
    #[cfg(not(panic = "abort"))]
    println!("panic=unwind build (use scripts/test_panic_abort_295.sh)");

    let base = std::env::var_os("S3DLIO_TEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let tmp = tempfile::Builder::new()
        .prefix("abort-localset-295-")
        .tempdir_in(base)
        .unwrap();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread)
        .build()
        .unwrap();

    // LocalSet driven by the main thread (not a worker).
    let local = tokio::task::LocalSet::new();
    rt.block_on(local.run_until(exercise(tmp.path())));
    // spawn_local inside the LocalSet.
    let local = tokio::task::LocalSet::new();
    let dir = tmp.path().to_path_buf();
    rt.block_on(local.run_until(async move {
        tokio::task::spawn_local(async move { exercise(&dir).await })
            .await
            .unwrap()
    }));
    // A plain spawned task on a worker (the hand-off path itself).
    let dir = tmp.path().to_path_buf();
    rt.block_on(async move {
        tokio::spawn(async move { exercise(&dir).await })
            .await
            .unwrap()
    });
    println!("OK");
}
