// examples/abort_localset_300.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #300 regression program: the synchronous wrappers built on `run_on_global_rt` must work from a
// tokio `LocalSet` on a multi-thread runtime in a binary built with `panic = "abort"`. A
// `block_in_place` that panics there aborts the process with SIGABRT. Run it through
// `scripts/test_panic_abort_300.sh` (profile `abort-test`).
//
// No real S3: the endpoint is an unreachable local port, so `stat_object_uri` returns Err; what is
// checked is that it RETURNS. Prints `OK` and exits 0 when every context works.

use s3dlio::s3_client::run_on_global_rt;

fn exercise() {
    let v = run_on_global_rt(async { Ok::<u32, anyhow::Error>(42) }).unwrap();
    assert_eq!(v, 42);
    let r = s3dlio::s3_utils::stat_object_uri("s3://bucket/key");
    assert!(r.is_err(), "unreachable endpoint must be an Err");
}

fn main() {
    #[cfg(panic = "abort")]
    println!("panic=abort build");
    #[cfg(not(panic = "abort"))]
    println!("panic=unwind build (use scripts/test_panic_abort_300.sh)");

    std::env::set_var("AWS_ACCESS_KEY_ID", "x");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "x");
    std::env::set_var("AWS_REGION", "us-east-1");
    std::env::set_var("AWS_ENDPOINT_URL", "http://127.0.0.1:9");

    for marked in [true, false] {
        let mut b = tokio::runtime::Builder::new_multi_thread();
        b.worker_threads(2).enable_all();
        if marked {
            b.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread);
        }
        let rt = b.build().unwrap();

        // LocalSet driven by the main thread (not a worker): the issue repro.
        let local = tokio::task::LocalSet::new();
        rt.block_on(local.run_until(async { exercise() }));
        // LocalSet::block_on.
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async { exercise() });
        // spawn_local inside the LocalSet.
        let local = tokio::task::LocalSet::new();
        rt.block_on(local.run_until(async {
            tokio::task::spawn_local(async { exercise() })
                .await
                .unwrap()
        }));
        // A plain spawned task on a worker (the hand-off path itself).
        rt.block_on(async { tokio::spawn(async { exercise() }).await.unwrap() });
    }
    println!("OK");
}
