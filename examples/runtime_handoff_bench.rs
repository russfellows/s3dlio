// examples/runtime_handoff_bench.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: what the worker hand-off (`tokio::task::block_in_place`, see
// `s3dlio::runtime_support`) costs and buys for `file://` `get_into`.
//
//   cargo run --release --example runtime_handoff_bench [-- <dir> [size_mib]]
//
// Two measurements, each on a runtime with the worker marker (hand-off) and without (inline):
//   1. one reader, warm page cache, reused destination: GiB/s (median of 7);
//   2. neighbour latency: `workers` reader tasks read continuously while a ticker task asks to
//      be woken every millisecond; the lateness of those wake-ups (p50 / p99 / max) is how long
//      a neighbour waits behind a blocking read chunk.

use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use s3dlio::FileSystemObjectStore;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn runtime(marked: bool, workers: usize) -> tokio::runtime::Runtime {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.worker_threads(workers).enable_all();
    if marked {
        b.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread);
    }
    b.build().unwrap()
}

fn single_reader(marked: bool, uri: &str, len: usize) -> f64 {
    let rt = runtime(marked, 4);
    let uri = uri.to_string();
    // Inside a spawned task, on a worker, as a real caller would be.
    rt.block_on(async move {
        tokio::spawn(async move {
            let store = FileSystemObjectStore::new();
            let mut dst = vec![0u8; len];
            let mut rates = Vec::new();
            for i in 0..8 {
                let t = Instant::now();
                store
                    .get_into(&uri, &mut dst, &ReceiveProgress::new())
                    .await
                    .unwrap();
                if i > 0 {
                    rates.push(len as f64 / (1u64 << 30) as f64 / t.elapsed().as_secs_f64());
                }
            }
            rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
            rates[rates.len() / 2]
        })
        .await
        .unwrap()
    })
}

fn neighbour_latency(marked: bool, uri: &str, len: usize, workers: usize) -> (f64, f64, f64) {
    let rt = runtime(marked, workers);
    let uri = uri.to_string();
    rt.block_on(async move {
        let stop = Arc::new(AtomicBool::new(false));
        let mut readers = Vec::new();
        for _ in 0..workers {
            let (stop, uri) = (Arc::clone(&stop), uri.clone());
            readers.push(tokio::spawn(async move {
                let store = FileSystemObjectStore::new();
                let mut dst = vec![0u8; len];
                while !stop.load(Ordering::Relaxed) {
                    store
                        .get_into(&uri, &mut dst, &ReceiveProgress::new())
                        .await
                        .unwrap();
                }
            }));
        }
        let mut late = Vec::new();
        let end = Instant::now() + Duration::from_secs(4);
        while Instant::now() < end {
            let want = Instant::now() + Duration::from_millis(1);
            tokio::time::sleep_until(want.into()).await;
            late.push(Instant::now().saturating_duration_since(want).as_secs_f64() * 1e3);
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.await.unwrap();
        }
        late.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f64| late[((late.len() - 1) as f64 * q) as usize];
        (at(0.5), at(0.99), *late.last().unwrap())
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| ".".into());
    let mib: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let len = mib << 20;
    let path = std::path::Path::new(&dir).join("handoff-bench.bin");
    std::fs::write(&path, vec![7u8; len]).unwrap();
    let uri = format!("file://{}", path.display());
    println!("file {uri} ({mib} MiB), warm page cache, 4 workers");
    for marked in [true, false] {
        let name = if marked {
            "hand-off (marked)"
        } else {
            "inline (unmarked)"
        };
        let rate = single_reader(marked, &uri, len);
        println!("{name:<20} single reader: {rate:6.2} GiB/s");
    }
    for marked in [true, false] {
        let name = if marked {
            "hand-off (marked)"
        } else {
            "inline (unmarked)"
        };
        let (p50, p99, max) = neighbour_latency(marked, &uri, len, 4);
        println!(
            "{name:<20} neighbour wake-up lateness, 4 readers on 4 workers: p50 {p50:.2} ms, p99 {p99:.2} ms, max {max:.2} ms"
        );
    }
    let _ = std::fs::remove_file(&path);
}
