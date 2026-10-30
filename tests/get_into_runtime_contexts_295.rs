// tests/get_into_runtime_contexts_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: the local file stores run their blocking reads inside the poll of the returned
// future (see `file_get_into::blocking_io`). That must work, and must never panic or print a
// panic message, in EVERY context a future can be polled in. In particular `block_in_place`
// panics inside a tokio `LocalSet` on a multi-thread runtime; the helper has to notice that
// and run inline instead.
//
// Every context runs the same operations: `file://` get_into, `file://` put through a writer
// (the borrowed-slice write), `direct://` get_into (page-aligned and unaligned destination),
// `direct://` get and put. Every context runs TWICE: on runtimes built with the s3dlio worker
// marker (`runtime_support::mark_tokio_worker_thread`, what s3dlio's own runtimes do) and on
// plain runtimes (an embedder that did not opt in). Each context runs in its own thread under
// a wall-clock timeout, and a panic hook that counts every panic of the process is installed
// first: a caught-and-hidden panic is a failure too (it would print a message).
//
// The helper that decides between `block_in_place` and inline (`runtime_support::blocking_io`)
// provokes no panic to find out, so the count must stay exactly zero.

#[path = "common/odirect.rs"]
mod odirect;

use bytes::Bytes;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use s3dlio::FileSystemObjectStore;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const FILL: u8 = 0xA5;
const LEN: usize = 20 * MIB + 3;

static PANICS: AtomicUsize = AtomicUsize::new(0);
/// Whether the multi-thread runtimes of the running round carry the s3dlio worker marker.
static MARKED: AtomicBool = AtomicBool::new(false);

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

/// The operations every context must be able to run.
async fn exercise(dir: PathBuf) {
    exercise_get_into(dir.clone()).await;
    exercise_put(dir.clone()).await;
    exercise_alloc(dir).await;
}

/// `file://` put through a writer: the borrowed-slice write (`write_borrowed_blocking`), through
/// both store types that serve `file://`.
async fn exercise_put(dir: PathBuf) {
    let data = pattern(LEN);
    let stores: [Box<dyn ObjectStore>; 2] = [
        Box::new(FileSystemObjectStore::new()),
        Box::new(ConfigurableFileSystemObjectStore::new(
            FileSystemConfig::default(),
        )),
    ];
    for (i, store) in stores.iter().enumerate() {
        let path = dir.join(format!("ctx-writer-{i}.bin"));
        let mut w = store
            .create_writer(
                &format!("file://{}", path.display()),
                s3dlio::object_store::WriterOptions::default(),
            )
            .await
            .unwrap();
        w.write_chunk(&data).await.unwrap();
        w.finalize().await.unwrap();
        assert!(
            std::fs::read(&path).unwrap() == data,
            "file:// writer content"
        );
    }
}

/// `get_into` only: needs no tokio runtime at all (the allocating `get` and `put` use
/// `tokio::fs`, which does).
async fn exercise_get_into(dir: PathBuf) {
    let data = pattern(LEN);
    let file_path = dir.join("ctx-file.bin");
    let direct_path = dir.join("ctx-direct.bin");
    std::fs::write(&file_path, &data).unwrap();
    std::fs::write(&direct_path, &data).unwrap();

    let file_uri = format!("file://{}", file_path.display());
    let direct_uri = format!("direct://{}", direct_path.display());

    // file:// get_into, through both store types that serve it.
    let file_stores: [Box<dyn ObjectStore>; 2] = [
        Box::new(FileSystemObjectStore::new()),
        Box::new(ConfigurableFileSystemObjectStore::new(
            FileSystemConfig::default(),
        )),
    ];
    for store in &file_stores {
        let mut dst = vec![0u8; LEN];
        let progress = ReceiveProgress::new();
        store
            .get_into(&file_uri, &mut dst, &progress)
            .await
            .unwrap();
        assert!(dst == data, "file:// get_into content");
        assert_eq!(progress.filled(), LEN as u64);
    }

    let direct = ConfigurableFileSystemObjectStore::with_direct_io();

    // direct:// get_into: an unaligned destination (the bounce pool) and an aligned one.
    for off in [3usize, 0] {
        let mut backing = AlignedBuf::new_zeroed(LEN + 2 * PAGE, PAGE).unwrap();
        backing.as_mut_slice().fill(FILL);
        let progress = ReceiveProgress::new();
        direct
            .get_into(
                &direct_uri,
                &mut backing.as_mut_slice()[off..off + LEN],
                &progress,
            )
            .await
            .unwrap();
        let all = backing.as_slice();
        assert!(
            all[off..off + LEN] == *data,
            "direct:// get_into content (off {off})"
        );
        assert!(all[..off].iter().all(|b| *b == FILL));
        assert!(all[off + LEN..].iter().all(|b| *b == FILL));
        assert_eq!(progress.filled(), LEN as u64);
    }
}

/// The allocating `direct://` `get` and `put` (they share the blocking helper).
async fn exercise_alloc(dir: PathBuf) {
    let data = pattern(LEN);
    let direct_path = dir.join("ctx-direct.bin");
    let direct_uri = format!("direct://{}", direct_path.display());
    let direct = ConfigurableFileSystemObjectStore::with_direct_io();
    let got = direct.get(&direct_uri).await.unwrap();
    assert!(got[..] == data[..], "direct:// get content");
    let put_uri = format!("direct://{}", dir.join("ctx-put.bin").display());
    direct
        .put(&put_uri, Bytes::from(pattern(LEN + 1)))
        .await
        .unwrap();
    assert!(std::fs::read(dir.join("ctx-put.bin")).unwrap() == pattern(LEN + 1));
}

fn mt_runtime(workers: usize) -> tokio::runtime::Runtime {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.worker_threads(workers).enable_all();
    if MARKED.load(Ordering::SeqCst) {
        b.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread);
    }
    b.build().unwrap()
}

fn ct_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

type ContextFn = fn(PathBuf);

fn contexts() -> Vec<(&'static str, ContextFn)> {
    vec![
        ("multi-thread: block_on from the main thread", |d| {
            mt_runtime(2).block_on(exercise(d))
        }),
        ("multi-thread: spawned task on a worker", |d| {
            let rt = mt_runtime(2);
            rt.block_on(async { tokio::spawn(exercise(d)).await.unwrap() })
        }),
        ("multi-thread: LocalSet::run_until in block_on", |d| {
            let rt = mt_runtime(2);
            let local = tokio::task::LocalSet::new();
            rt.block_on(local.run_until(exercise(d)))
        }),
        ("multi-thread: LocalSet::block_on", |d| {
            let rt = mt_runtime(2);
            let local = tokio::task::LocalSet::new();
            local.block_on(&rt, exercise(d))
        }),
        ("multi-thread: spawn_local inside a LocalSet", |d| {
            let rt = mt_runtime(2);
            let local = tokio::task::LocalSet::new();
            rt.block_on(
                local
                    .run_until(async move { tokio::task::spawn_local(exercise(d)).await.unwrap() }),
            )
        }),
        ("multi-thread, 1 worker: LocalSet::run_until", |d| {
            let rt = mt_runtime(1);
            let local = tokio::task::LocalSet::new();
            rt.block_on(local.run_until(exercise(d)))
        }),
        (
            "multi-thread: futures::executor::block_on under rt.enter()",
            |d| {
                let rt = mt_runtime(2);
                let _guard = rt.enter();
                futures::executor::block_on(exercise(d))
            },
        ),
        ("multi-thread: spawn_blocking + Handle::block_on", |d| {
            let rt = mt_runtime(2);
            let handle = rt.handle().clone();
            rt.block_on(async move {
                tokio::task::spawn_blocking(move || handle.block_on(exercise(d)))
                    .await
                    .unwrap()
            })
        }),
        (
            "multi-thread: spawn_blocking + Handle::block_on(LocalSet + spawn_local), inline scope",
            |d| {
                let rt = mt_runtime(2);
                let handle = rt.handle().clone();
                rt.block_on(async move {
                    tokio::task::spawn_blocking(move || {
                        // The documented opt-out for a closure that drives a LocalSet itself.
                        let _inline = s3dlio::runtime_support::inline_blocking_io();
                        let local = tokio::task::LocalSet::new();
                        handle.block_on(local.run_until(async move {
                            tokio::task::spawn_local(exercise(d)).await.unwrap()
                        }))
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        (
            "multi-thread: every pool thread (workers that handed off, then spawn_blocking)",
            |d| {
                let rt = mt_runtime(2);
                let handle = rt.handle().clone();
                rt.block_on(async move {
                    // Many hand-offs first: the core moves between pool threads, so former
                    // workers end up as plain blocking threads.
                    let mut tasks = Vec::new();
                    for i in 0..6 {
                        let d = d.join(format!("w{i}"));
                        std::fs::create_dir_all(&d).unwrap();
                        tasks.push(tokio::spawn(exercise_get_into(d)));
                    }
                    for t in tasks {
                        t.await.unwrap();
                    }
                    // Now drive futures directly with Handle::block_on from blocking threads.
                    let mut blocking = Vec::new();
                    for i in 0..8 {
                        let d = d.join(format!("b{i}"));
                        std::fs::create_dir_all(&d).unwrap();
                        let handle = handle.clone();
                        blocking.push(tokio::task::spawn_blocking(move || {
                            handle.block_on(exercise_get_into(d))
                        }));
                    }
                    for t in blocking {
                        t.await.unwrap();
                    }
                })
            },
        ),
        ("multi-thread: block_in_place + Handle::block_on", |d| {
            let rt = mt_runtime(2);
            rt.block_on(async move {
                tokio::spawn(async move {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(exercise(d))
                    })
                })
                .await
                .unwrap()
            })
        }),
        ("current-thread: block_on", |d| {
            ct_runtime().block_on(exercise(d))
        }),
        ("current-thread: LocalSet::run_until", |d| {
            let rt = ct_runtime();
            let local = tokio::task::LocalSet::new();
            rt.block_on(local.run_until(exercise(d)))
        }),
        ("current-thread: spawn_local", |d| {
            let rt = ct_runtime();
            let local = tokio::task::LocalSet::new();
            rt.block_on(
                local
                    .run_until(async move { tokio::task::spawn_local(exercise(d)).await.unwrap() }),
            )
        }),
        (
            "no runtime: futures::executor::block_on (get_into only)",
            |d| futures::executor::block_on(exercise_get_into(d)),
        ),
    ]
}

#[test]
fn every_polling_context_works_and_prints_no_panic_message() {
    // Installed FIRST: the library may chain a quiet hook over it, but whatever panics it does
    // not hide still reaches this one. A panic that is caught and silently ignored is not
    // counted by anything else, so the count must be exactly zero.
    std::panic::set_hook(Box::new(|info| {
        PANICS.fetch_add(1, Ordering::SeqCst);
        eprintln!("PANIC: {info}");
    }));

    let Some(scratch) = odirect::odirect_scratch_dir("get-into-ctx-295-") else {
        return;
    };
    let mut failures = Vec::new();
    for marked in [true, false] {
        MARKED.store(marked, Ordering::SeqCst);
        let round = if marked { "marked" } else { "unmarked" };
        for (name, run) in contexts() {
            let dir = scratch.path().join(format!(
                "{round}-{}",
                name.replace([':', ' ', ',', '(', ')', '.', '+'], "_")
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let (tx, rx) = mpsc::channel();
            let handle = std::thread::Builder::new()
                .name(name.into())
                .spawn(move || {
                    run(dir);
                    let _ = tx.send(());
                })
                .unwrap();
            match rx.recv_timeout(Duration::from_secs(120)) {
                Ok(()) => {
                    handle.join().unwrap();
                    eprintln!("ok      [{round}] {name}");
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // The thread ended without reporting: it panicked.
                    let _ = handle.join();
                    failures.push(format!("[{round}] {name}: panicked"));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    failures.push(format!(
                        "[{round}] {name}: no result within 120 s (deadlock?)"
                    ));
                }
            }
        }
    }
    let panics = PANICS.load(Ordering::SeqCst);
    assert!(failures.is_empty(), "failing contexts: {failures:#?}");
    assert_eq!(panics, 0, "a panic (even a caught one) printed a message");
}
