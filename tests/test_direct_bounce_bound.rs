// tests/test_direct_bounce_bound.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// The HARD bound on direct:// bounce memory under mixed concurrent GETs and PUTs WITH
// CANCELLATIONS, measured independently of throughput:
//   * by the pool's own gauge (`BounceStats::peak_leased_bytes`, updated at every lease), and
//   * independently, by a global allocator that tracks the live bytes of every 4096-aligned
//     allocation of exactly the bounce-buffer size (the pool's buffers; GET destinations and
//     tails have other sizes).
// Both must stay <= `bounce_memory_cap()` = nodes x pairs x 2 x size. After the storm every lease
// must be back (RAII: success, error and cancelled futures alike).

#[path = "common/odirect.rs"]
mod odirect;

use bytes::Bytes;
use s3dlio::direct_bounce::{
    __bench_configure, bounce_memory_cap, bounce_stats, BounceConfig, BusyPolicy, GetStrategy,
    PutStrategy,
};
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::ObjectStore;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIB: usize = 1024 * 1024;
const BUF: usize = 4 * MIB;
const PAIRS: usize = 2;

struct Tracking {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl Tracking {
    fn is_bounce(layout: &Layout) -> bool {
        layout.align() == 4096 && layout.size() == BUF
    }
}

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() && Self::is_bounce(&layout) {
            let now = self.live.fetch_add(BUF, Ordering::SeqCst) + BUF;
            self.peak.fetch_max(now, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if Self::is_bounce(&layout) {
            self.live.fetch_sub(BUF, Ordering::SeqCst);
        }
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOC: Tracking = Tracking {
    live: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i + seed).wrapping_mul(131) ^ ((i + seed) >> 8).wrapping_mul(7)) as u8)
        .collect()
}

fn rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn peak_live_bounce_bytes_never_exceed_the_cap_under_mixed_cancelled_traffic() {
    let Some(dir) = odirect::odirect_scratch_dir("direct-bounce-bound-") else {
        return;
    };
    assert!(__bench_configure(BounceConfig {
        buffer_size: BUF,
        pairs_per_node: PAIRS,
        busy: BusyPolicy::Wait,
        put: PutStrategy::Pooled,
        get: GetStrategy::Pooled,
    }));
    let cap = bounce_memory_cap();
    assert!(cap.is_multiple_of(2 * PAIRS * BUF));

    let store = Arc::new(ConfigurableFileSystemObjectStore::new(FileSystemConfig {
        direct_io: true,
        ..FileSystemConfig::default()
    }));
    // sizes: below one buffer, a few buffers, and many buffers (none is a multiple of 4 MiB)
    let sizes = [100 * 1024 + 7, 3 * MIB + 11, 9 * MIB + 5, 17 * MIB + 1];
    for (i, len) in sizes.iter().enumerate() {
        let uri = format!("direct://{}/o{i}.bin", dir.path().display());
        store
            .put(&uri, Bytes::from(pattern(*len, i)))
            .await
            .expect("seed put");
    }
    let before = bounce_stats();

    let storm = async {
        let mut tasks = Vec::new();
        for t in 0..32u64 {
            let store = store.clone();
            let base = dir.path().display().to_string();
            tasks.push(tokio::spawn(async move {
                let mut state = 0xABCD_0000 ^ t;
                let (mut done, mut cancelled) = (0u32, 0u32);
                for _ in 0..8 {
                    let i = (rng(&mut state) % sizes.len() as u64) as usize;
                    let len = sizes[i];
                    let uri = format!("direct://{base}/o{i}.bin");
                    let is_get = rng(&mut state).is_multiple_of(2);
                    let cancel_after = (rng(&mut state) % 10 < 4)
                        .then(|| Duration::from_millis(rng(&mut state) % 25));
                    let op = async {
                        if is_get {
                            let got = store.get(&uri).await.expect("get");
                            assert!(
                                got.as_ref() == pattern(len, i).as_slice(),
                                "GET content differs"
                            );
                        } else {
                            // PUTs go to files of their own (a PUT truncates its target, so a
                            // concurrent GET of the same file would legitimately see a partial one)
                            let target = format!("direct://{base}/w{t}-{i}.bin");
                            store
                                .put(&target, Bytes::from(pattern(len, i)))
                                .await
                                .expect("put");
                        }
                    };
                    match cancel_after {
                        // dropping the future mid-flight is the cancellation under test
                        Some(d) => match tokio::time::timeout(d, op).await {
                            Ok(()) => done += 1,
                            Err(_) => cancelled += 1,
                        },
                        None => {
                            op.await;
                            done += 1;
                        }
                    }
                }
                (done, cancelled)
            }));
        }
        let mut totals = (0, 0);
        for t in tasks {
            let (d, c) = t.await.expect("task");
            totals = (totals.0 + d, totals.1 + c);
        }
        totals
    };
    let (done, cancelled) = tokio::time::timeout(Duration::from_secs(300), storm)
        .await
        .expect("mixed traffic did not finish (deadlock or starvation)");

    // Cancelled futures leave blocking closures running: wait until every lease is back.
    let deadline = Instant::now() + Duration::from_secs(120);
    while bounce_stats().leased_bytes != 0 {
        assert!(
            Instant::now() < deadline,
            "leases still outstanding after the storm: {:?}",
            bounce_stats()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let after = bounce_stats();
    eprintln!(
        "ops completed {done}, cancelled {cancelled}; peak leased {} B, allocator peak {} B, \
         cap {cap} B; cancelled waits {}; leases put/get {}/{}",
        after.peak_leased_bytes,
        ALLOC.peak.load(Ordering::SeqCst),
        after.cancelled_waits - before.cancelled_waits,
        after.leases[0] - before.leases[0],
        after.leases[1] - before.leases[1],
    );
    assert!(cancelled > 0, "the test must actually cancel operations");
    assert!(
        after.peak_leased_bytes <= cap as u64,
        "pool gauge: peak leased {} > cap {cap}",
        after.peak_leased_bytes
    );
    assert!(
        ALLOC.peak.load(Ordering::SeqCst) <= cap,
        "allocator: peak live bounce bytes {} > cap {cap}",
        ALLOC.peak.load(Ordering::SeqCst)
    );
    assert!(after.allocated_bytes <= cap as u64);
    assert_eq!(after.leased_bytes, 0, "a lease was not returned");
    assert_eq!(after.fallback_live_bytes, 0);
    assert!(after.leases[0] > before.leases[0] && after.leases[1] > before.leases[1]);
}
