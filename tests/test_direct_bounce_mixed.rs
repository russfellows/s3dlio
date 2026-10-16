// tests/test_direct_bounce_mixed.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// The shared direct:// bounce pool under MIXED traffic: GETs and PUTs draw ping-pong pairs from
// the same per-node pool (one pair, small buffers, so every request is many chunks and they
// really contend). Checks: content is intact in both directions, both directions were served by
// the pool, nothing deadlocks, and the POOLED memory cap is exactly nodes x pairs x 2 x size.

#[path = "common/odirect.rs"]
mod odirect;

use bytes::Bytes;
use s3dlio::direct_bounce::{
    __bench_configure, bounce_memory_cap, bounce_stats, BounceConfig, BusyPolicy, GetStrategy,
    PutStrategy,
};
use s3dlio::file_store_direct::{
    direct_io_counters, ConfigurableFileSystemObjectStore, FileSystemConfig,
};
use s3dlio::object_store::ObjectStore;
use std::sync::Arc;
use std::time::Duration;

const MIB: usize = 1024 * 1024;
const BUF: usize = 4 * MIB;

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i + seed).wrapping_mul(131) ^ ((i + seed) >> 8).wrapping_mul(7)) as u8)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_gets_and_puts_share_one_pair_without_starving_or_corrupting() {
    let Some(dir) = odirect::odirect_scratch_dir("direct-bounce-mixed-") else {
        return;
    };
    assert!(
        __bench_configure(BounceConfig {
            buffer_size: BUF,
            pairs_per_node: 1,
            busy: BusyPolicy::Wait,
            put: PutStrategy::Pooled,
            get: GetStrategy::Pooled,
        }),
        "this process must configure the pool before its first direct:// transfer"
    );
    let cap = bounce_memory_cap();
    assert_eq!(cap % (2 * BUF), 0, "cap is a whole number of pairs");

    let store = Arc::new(ConfigurableFileSystemObjectStore::new(
        FileSystemConfig::direct_io(),
    ));
    let len = 6 * BUF + 4093; // 7 chunks
    let n_objects = 4;
    for i in 0..n_objects {
        let uri = format!("direct://{}/o{i}.bin", dir.path().display());
        store
            .put(&uri, Bytes::from(pattern(len, i)))
            .await
            .expect("seed put");
    }
    let before = bounce_stats();
    let counters_before = direct_io_counters();

    let work = async {
        let mut tasks = Vec::new();
        for t in 0..8usize {
            let store = store.clone();
            let base = dir.path().display().to_string();
            tasks.push(tokio::spawn(async move {
                for round in 0..6usize {
                    let i = (t + round) % n_objects;
                    let uri = format!("direct://{base}/o{i}.bin");
                    if (t + round) % 2 == 0 {
                        let got = store.get(&uri).await.expect("get");
                        assert!(
                            got.as_ref() == pattern(len, i).as_slice(),
                            "GET content differs (object {i})"
                        );
                    } else {
                        // PUTs go to files of their own: a PUT truncates its target, so a
                        // concurrent GET of the same file would legitimately see a partial one.
                        let target = format!("direct://{base}/w{t}-{i}.bin");
                        store
                            .put(&target, Bytes::from(pattern(len, i)))
                            .await
                            .expect("put");
                    }
                }
            }));
        }
        for t in tasks {
            t.await.expect("task");
        }
    };
    tokio::time::timeout(Duration::from_secs(120), work)
        .await
        .expect("mixed traffic did not finish (deadlock or starvation)");

    let after = bounce_stats();
    let counters = direct_io_counters();
    assert!(
        after.leases[0] > before.leases[0],
        "no PUT went through the pool"
    );
    assert!(
        after.leases[1] > before.leases[1],
        "no GET went through the pool"
    );
    assert_eq!(counters.put_fallbacks, counters_before.put_fallbacks);
    assert_eq!(counters.get_fallbacks, counters_before.get_fallbacks);
    assert!(
        after.pools_created as usize <= cap / (2 * BUF),
        "more node pools than nodes: memory cap exceeded"
    );
    eprintln!(
        "pool stats: puts leased {} (waited {}, {} ms), gets leased {} (waited {}, {} ms)",
        after.leases[0] - before.leases[0],
        after.waited[0] - before.waited[0],
        (after.wait_ns[0] - before.wait_ns[0]) / 1_000_000,
        after.leases[1] - before.leases[1],
        after.waited[1] - before.waited[1],
        (after.wait_ns[1] - before.wait_ns[1]) / 1_000_000,
    );
}
