// tests/s3_get_into_allocation.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// "No extra copy" for the S3 caller-buffer GET (plan §4.2 and §9): the bytes are streamed from
// the response into the caller's buffer, with no `get()` plus copy, no zero-filled master
// buffer and no accumulation of a whole part or object.
//
// Mechanism: a counting global allocator records the LARGEST single allocation made on the
// test's own thread while a read runs. The test runtime is a current-thread runtime, so the
// client side of the read (the request future, the transport tasks, the body chunks) all run
// on this thread, while the in-process mock server runs on its own threads and is not counted.
// A streamed read may allocate transport buffers (a few hundred KiB at most) but nothing near
// the size of a part or of the object; a control read through the SDK's own `collect()` is
// shown to allocate the whole object, which proves the counter would catch a copy.
//
// Own process: it installs a global allocator and fixes the range settings.

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use s3_get_into_support::mock::MockObject;
use s3_get_into_support::*;
use s3dlio::receive_progress::ReceiveProgress;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    TRACKING
        .try_with(|tracking| {
            if tracking.get() {
                LARGEST
                    .try_with(|largest| largest.set(largest.get().max(size)))
                    .ok();
            }
        })
        .ok();
}

// SAFETY: every method forwards to the system allocator unchanged, after noting the size; the
// bookkeeping uses const-initialised `Cell`s, which allocate nothing.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const MIB: usize = 1024 * 1024;
/// The most one transport buffer may be: far below a part (4 MiB) or the object.
const TRANSPORT_LIMIT: usize = MIB;

/// Run `fut` and return its result with the largest single allocation made on this thread.
async fn largest_allocation<T>(fut: impl std::future::Future<Output = T>) -> (T, usize) {
    LARGEST.with(|l| l.set(0));
    TRACKING.with(|t| t.set(true));
    let out = fut.await;
    TRACKING.with(|t| t.set(false));
    (out, LARGEST.with(Cell::get))
}

async fn setup() -> (
    s3dlio::s3_ops::S3Ops,
    aws_sdk_s3::Client,
    s3_get_into_support::mock::MockS3State,
) {
    // Below 8 MiB a read is one GET; at or above it, ranges of 4 MiB.
    let (port, state) = init(&[
        ("S3DLIO_RANGE_THRESHOLD_MB", "8"),
        ("S3DLIO_CHUNK_SIZE", "4194304"),
        ("S3DLIO_RANGE_CONCURRENCY", "2"),
    ]);
    let client = client_for(port).await;
    (ops(client.clone()), client, state)
}

#[tokio::test]
async fn the_single_get_streams_into_the_buffer_without_a_whole_object_allocation() {
    let (ops, client, state) = setup().await;
    let n = 3 * MIB;
    let key = key("single");
    let data = pattern(n, 1);
    state.put_object(BUCKET, &key, MockObject::new(data.clone()));
    let mut dst = Dst::new(n);
    let addr = dst.addr();

    let (result, largest) =
        largest_allocation(async { read(&ops, &key, &mut dst, &ReceiveProgress::new()).await })
            .await;

    result.expect("read");
    assert_eq!(
        dst.addr(),
        addr,
        "the bytes landed in the caller's allocation"
    );
    assert!(dst.bytes() == &data[..], "content");
    dst.assert_canaries();
    assert!(
        largest < TRANSPORT_LIMIT,
        "a single {largest}-byte allocation during a {n}-byte read: the body was not streamed"
    );

    // Control: the SDK's own collect() makes the whole object contiguous, and the counter sees it.
    let (collected, control) = largest_allocation(async {
        let resp = client
            .get_object()
            .bucket(BUCKET)
            .key(&key)
            .send()
            .await
            .expect("send");
        resp.body
            .collect()
            .await
            .expect("collect")
            .into_bytes()
            .len()
    })
    .await;
    assert_eq!(collected, n);
    assert!(
        control >= n,
        "the control read allocated at most {control} bytes: the counter cannot see a copy"
    );
}

#[tokio::test]
async fn the_ranged_fan_out_streams_into_disjoint_parts_without_part_or_object_allocations() {
    let (ops, _client, state) = setup().await;
    let n = 24 * MIB;
    let key = key("fanout");
    let data = pattern(n, 2);
    state.put_object(BUCKET, &key, MockObject::new(data.clone()));
    let mut dst = Dst::new(n);
    let addr = dst.addr();
    let progress = ReceiveProgress::new();

    let (result, largest) =
        largest_allocation(async { read(&ops, &key, &mut dst, &progress).await }).await;

    result.expect("read");
    assert_eq!(dst.addr(), addr);
    assert!(dst.bytes() == &data[..], "content");
    dst.assert_canaries();
    assert_eq!(progress.filled(), n as u64);
    assert_eq!(gets(&state, &key).len(), 6, "six 4 MiB ranges");
    assert!(
        largest < TRANSPORT_LIMIT,
        "a single {largest}-byte allocation during a {n}-byte ranged read (parts are 4 MiB)"
    );
}
