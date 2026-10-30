// tests/direct_get_into_tiny_pool_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: many CONCURRENT unaligned direct:// reads through the smallest possible bounce
// pool (one buffer pair per node, 4 MiB buffers), in both busy policies (`wait`: strict FIFO
// for a pair; `fallback`: a private buffer when every pair is leased), on a one-worker
// multi-thread runtime (where `block_in_place` hands the worker off, so reads really overlap)
// and on a current-thread runtime. Every read must come back byte-exact with progress equal to
// its length, and nothing may deadlock.
//
// The pool is configured by environment variables read once per process, so each
// (policy, runtime) pair runs in its OWN child process (this binary re-executed on
// `tiny_pool_child`) under a wall-clock timeout; the parent test fails, and kills the child,
// when it does not finish.

use s3dlio::direct_bounce::{bounce_memory_cap, bounce_stats, config, BusyPolicy};
use s3dlio::file_store_direct::ConfigurableFileSystemObjectStore;
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const CASE_VAR: &str = "S3DLIO_TEST_TINY_POOL_CASE";
const READERS: usize = 24;
const CHILD_TIMEOUT: Duration = Duration::from_secs(240);

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

#[path = "common/odirect.rs"]
mod odirect;

/// The parent: one child process per (busy policy, runtime).
#[test]
fn concurrent_unaligned_reads_through_a_one_pair_pool_neither_fail_nor_deadlock() {
    if odirect::odirect_scratch_dir("tiny-pool-probe-").is_none() {
        return; // the filesystem refuses O_DIRECT: nothing to test (message printed)
    }
    let exe = std::env::current_exe().unwrap();
    let mut failures = Vec::new();
    for policy in ["wait", "fallback"] {
        for runtime in ["mt1", "ct"] {
            let log = tempfile::NamedTempFile::new().unwrap();
            let started = Instant::now();
            let mut child = Command::new(&exe)
                .args([
                    "--exact",
                    "tiny_pool_child",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CASE_VAR, format!("{policy}-{runtime}"))
                .env("S3DLIO_DIRECT_IO_BOUNCE_PAIRS_PER_NODE", "1")
                .env("S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB", "4")
                .env("S3DLIO_DIRECT_IO_BOUNCE_WHEN_BUSY", policy)
                .stdout(Stdio::from(log.reopen().unwrap()))
                .stderr(Stdio::from(log.reopen().unwrap()))
                .spawn()
                .unwrap();
            let status = loop {
                if let Some(s) = child.try_wait().unwrap() {
                    break Some(s);
                }
                if started.elapsed() > CHILD_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(100));
            };
            let output = std::fs::read_to_string(log.path()).unwrap_or_default();
            match status {
                Some(s) if s.success() && output.contains("TINY-POOL-OK") => {
                    let summary = output.lines().find(|l| l.contains("readers,"));
                    eprintln!(
                        "ok {policy}-{runtime} in {:?}: {}",
                        started.elapsed(),
                        summary.unwrap_or("")
                    );
                }
                Some(s) => failures.push(format!("{policy}-{runtime}: {s}\n{output}")),
                None => failures.push(format!(
                    "{policy}-{runtime}: no result within {CHILD_TIMEOUT:?} (deadlock?)\n{output}"
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n----\n"));
}

/// The child: runs one case, selected by the environment, and prints `TINY-POOL-OK`.
#[test]
fn tiny_pool_child() {
    let Ok(case) = std::env::var(CASE_VAR) else {
        return; // run directly (not by the parent): nothing to do
    };
    let (policy, runtime) = case.split_once('-').unwrap();
    assert_eq!(config().pairs_per_node, 1);
    assert_eq!(config().buffer_size, 4 * MIB);
    assert_eq!(
        config().busy,
        if policy == "wait" {
            BusyPolicy::Wait
        } else {
            BusyPolicy::Fallback
        }
    );

    let scratch = odirect::odirect_scratch_dir("tiny-pool-295-").expect("O_DIRECT scratch");
    let sizes = [
        4 * MIB + 1,
        9 * MIB + 4097,
        16 * MIB + 5,
        20 * MIB + 3,
        40 * MIB + 7,
        PAGE + 1,
    ];
    let files: Vec<(String, Arc<Vec<u8>>)> = sizes
        .iter()
        .map(|&len| {
            let data = pattern(len);
            let path = scratch.path().join(format!("f-{len}.bin"));
            std::fs::write(&path, &data).unwrap();
            (format!("direct://{}", path.display()), Arc::new(data))
        })
        .collect();

    let rt = match runtime {
        "mt1" => tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap(),
        _ => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    };
    let store = Arc::new(ConfigurableFileSystemObjectStore::with_direct_io());
    let before = bounce_stats();
    rt.block_on(async {
        let mut tasks = Vec::new();
        for i in 0..READERS {
            let (uri, data) = files[i % files.len()].clone();
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                let len = data.len();
                // A destination 1 byte past a page boundary, in an ordinary allocation.
                let mut backing = vec![0u8; len + 2 * PAGE];
                let start = (PAGE - backing.as_ptr() as usize % PAGE + 1) % PAGE;
                assert_eq!((backing.as_ptr() as usize + start) % PAGE, 1);
                let progress = ReceiveProgress::new();
                store
                    .get_into(&uri, &mut backing[start..start + len], &progress)
                    .await
                    .unwrap_or_else(|e| panic!("reader {i} ({uri}): {e:#}"));
                assert!(
                    backing[start..start + len] == data[..],
                    "reader {i}: content"
                );
                assert_eq!(progress.filled(), len as u64, "reader {i}: progress");
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
    });
    let after = bounce_stats();

    let leased = after.leases[1] - before.leases[1];
    let private = after.fallbacks[1] - before.fallbacks[1];
    assert!(
        leased + private >= READERS as u64,
        "every read used the bounce pool or a private buffer: {leased} + {private}"
    );
    if policy == "wait" {
        assert_eq!(private, 0, "the wait policy never uses a private buffer");
        assert!(
            after.peak_leased_bytes <= bounce_memory_cap() as u64,
            "the pool cap held: {} > {}",
            after.peak_leased_bytes,
            bounce_memory_cap()
        );
    }
    eprintln!(
        "{case}: {READERS} readers, {leased} leases ({} waited), {private} private buffers",
        after.waited[1] - before.waited[1]
    );
    println!("TINY-POOL-OK");
}
