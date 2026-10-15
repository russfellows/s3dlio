// tests/test_direct_put_cancel_intact.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// A direct:// PUT that is cancelled while it WAITS for bounce buffers must leave the previous
// object intact: the target is created/truncated only once the buffers are leased (it used to be
// opened with O_TRUNC first, so a cancelled waiter left an EMPTY object over a good one).

#[path = "common/odirect.rs"]
mod odirect;

use bytes::Bytes;
use s3dlio::direct_bounce::{
    __bench_configure, bounce_stats, BounceConfig, BusyPolicy, GetStrategy, PutStrategy,
};
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::ObjectStore;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIB: usize = 1024 * 1024;

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i + seed).wrapping_mul(131) ^ ((i + seed) >> 8).wrapping_mul(7)) as u8)
        .collect()
}

async fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_put_that_waits_for_buffers_leaves_the_old_object_intact() {
    let Some(dir) = odirect::odirect_scratch_dir("direct-put-cancel-intact-") else {
        return;
    };
    assert!(__bench_configure(BounceConfig {
        buffer_size: 4 * MIB,
        pairs_per_node: 1, // one pair per node: a long PUT blocks every other
        busy: BusyPolicy::Wait,
        put: PutStrategy::Pooled,
        get: GetStrategy::Direct,
    }));
    let store = Arc::new(ConfigurableFileSystemObjectStore::new(FileSystemConfig {
        direct_io: true,
        ..FileSystemConfig::default()
    }));
    let base = dir.path().display().to_string();
    let old = pattern(9 * MIB + 5, 1);
    let x = format!("direct://{base}/x.bin");
    store
        .put(&x, Bytes::from(old.clone()))
        .await
        .expect("seed x");
    let fresh_name = format!("direct://{base}/never-written.bin");

    // A: a long PUT that holds the only pair of this node.
    let before = bounce_stats();
    let a_uri = format!("direct://{base}/y.bin");
    let a_data = Bytes::from(pattern(384 * MIB + 3, 2));
    let a = {
        let store = store.clone();
        tokio::spawn(async move { store.put(&a_uri, a_data).await })
    };
    wait_for("A to lease the pair", || {
        bounce_stats().leases[0] > before.leases[0]
    })
    .await;

    // B: PUT of other data over x.bin, and C: a PUT of a NEW object. Both queue behind A.
    let waiters = before.waited[0];
    let b = {
        let store = store.clone();
        let x = x.clone();
        tokio::spawn(async move { store.put(&x, Bytes::from(pattern(5 * MIB, 3))).await })
    };
    let c = {
        let store = store.clone();
        let n = fresh_name.clone();
        tokio::spawn(async move { store.put(&n, Bytes::from(pattern(5 * MIB, 4))).await })
    };
    wait_for("B and C to queue", || {
        bounce_stats().leases[0] == before.leases[0] + 1 && !a.is_finished() && {
            std::thread::sleep(Duration::from_millis(100));
            true
        }
    })
    .await;
    assert!(
        !a.is_finished(),
        "A finished too early for this test to mean anything"
    );
    let _ = waiters;

    // Cancel both waiters.
    b.abort();
    c.abort();
    let _ = b.await;
    let _ = c.await;
    wait_for("the waits to be abandoned", || {
        bounce_stats().cancelled_waits >= before.cancelled_waits + 2
    })
    .await;

    // The old object is untouched, and no object was created for C.
    assert_eq!(
        std::fs::read(dir.path().join("x.bin")).unwrap(),
        old,
        "a cancelled waiter truncated or rewrote the previous object"
    );
    assert!(
        !dir.path().join("never-written.bin").exists(),
        "a cancelled waiter created an (empty) object"
    );

    a.await.expect("A").expect("A's put");
    assert_eq!(std::fs::read(dir.path().join("x.bin")).unwrap(), old);
    assert_eq!(
        std::fs::metadata(dir.path().join("y.bin")).unwrap().len() as usize,
        384 * MIB + 3
    );
}
