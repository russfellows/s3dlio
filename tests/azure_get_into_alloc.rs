// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `AzureBlob::get_into` takes the SDK's no-remaining-ranges path: the body comes straight from
//! the first response, with no partition buffer and no whole-object allocation (plan v0.10.0
//! §4.3 and §9, "verified by allocation inspection, not only by the options"; #264).
//!
//! How it is measured. A counting global allocator records the LARGEST SINGLE allocation made on
//! a flagged thread. The fake service runs on its own thread with its own runtime, so its
//! (object-sized) allocations are not counted; the client runs on a current-thread runtime, so
//! everything the SDK does for the request (its connection task, any spawned partition task)
//! runs on the flagged thread. The SDK's partitioned path allocates `vec![0u8; partition_size]`
//! for the first partition (`azure_storage_blob` 1.1.0, `partitioned_transfer/download.rs`,
//! `start_initial_download_task_buffer`, line 367), so a get that is not on the no-remaining-ranges path shows an
//! allocation as large as the partition.
//!
//! The positive control reads the same blob through the SDK's partitioned download (a partition
//! of half the blob), and must show the large allocation. Without it, "small" would prove
//! nothing about the instrument.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::azure_client::AzureBlob;
use s3dlio::receive_progress::ReceiveProgress;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
}
static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System` unchanged; the bookkeeping only reads a
// const-initialised thread-local flag and updates an atomic, neither of which allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` come from a matching `alloc` through this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The largest single allocation made on this thread while `work` ran.
async fn largest_allocation<T>(work: impl std::future::Future<Output = T>) -> (usize, T) {
    LARGEST.store(0, Ordering::SeqCst);
    TRACK.with(|t| t.set(true));
    let out = work.await;
    TRACK.with(|t| t.set(false));
    (LARGEST.load(Ordering::SeqCst), out)
}

/// The fake on its own thread and runtime, so its allocations are not the client's.
fn fake_on_its_own_thread() -> (State, u16) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("fake runtime");
        runtime.block_on(async move {
            let started = start().await;
            tx.send(started).expect("hand over the fake");
            std::future::pending::<()>().await;
        });
    });
    rx.recv().expect("fake started")
}

#[test]
fn get_into_makes_no_partition_or_whole_object_allocation() {
    const SIZE: usize = 16 * MIB;
    clear_azure_env();
    std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
    let (state, port) = fake_on_its_own_thread();
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    set_blob(&state, "big", pattern(SIZE, 11));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("client runtime");
    runtime.block_on(async {
        let client = AzureBlob::with_default_credential("acct", "ctr").expect("client");

        // Warm-up: the first request builds connection state; it is not what is measured.
        let mut warm = vec![0u8; 1024];
        set_blob(&state, "warm", pattern(1024, 1));
        client
            .get_into("warm", &mut warm, &ReceiveProgress::new())
            .await
            .expect("warm-up read");

        // Positive control: the SDK's own partitioned download, partition = half the blob.
        let (control_largest, control) = largest_allocation(async {
            client
                .get_with_sdk_concurrency("big", Some(1), Some(SIZE / 2))
                .await
                .expect("partitioned read")
        })
        .await;
        assert_eq!(control.len(), SIZE);
        assert!(
            control_largest >= SIZE / 2,
            "the instrument is blind: the partitioned path's largest allocation was only \
             {control_largest} bytes"
        );
        drop(control);

        // The measured read.
        let mut dst = vec![0u8; SIZE];
        let dst_address = dst.as_ptr();
        let progress = ReceiveProgress::new();
        let before = mark(&state);
        let (largest, result) =
            largest_allocation(client.get_into("big", &mut dst, &progress)).await;
        result.expect("get_into");
        assert_eq!(dst.as_ptr(), dst_address, "the destination never moved");
        assert!(dst == pattern(SIZE, 11), "wrong data");
        assert_eq!(progress.filled(), SIZE as u64);
        assert!(
            largest < SIZE / 8,
            "get_into made an allocation of {largest} bytes for a {SIZE}-byte blob: it is not on \
             the no-remaining-ranges path (a partition or whole-object buffer)"
        );
        let seen = requests_since(&state, before);
        assert_eq!(
            seen.len(),
            1,
            "one bounded request, no partitions: {seen:?}"
        );
        assert_eq!(
            seen[0].range.as_deref(),
            Some(format!("bytes=0-{SIZE}").as_str())
        );
    });
}
