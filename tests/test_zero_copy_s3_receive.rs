// tests/test_zero_copy_s3_receive.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Stage G (zero copy everywhere, RULE #1), R2: an S3 `get` / `get_range` through a per-endpoint
// client receives the body into ONE pre-sized buffer (it used to `collect()` every chunk and then
// `into_bytes()` them into a second buffer: 2x the object live at once).
//
// How the read is measured: a counting global allocator keeps the live bytes and their peak of
// EVERY thread of this process except the mock server's own threads
// (`mock::on_mock_server_thread`), while a measured window is open. Since #303 the transport of
// an S3 read (sockets, connection tasks, every body chunk) runs on s3dlio's I/O-owner runtime,
// not on the caller's thread, so counting only the calling thread would miss the chunks; the
// whole process is counted instead. The tests of this binary therefore run one at a time
// (`SERIAL`) and nothing else allocates object-sized buffers while a window is open. A read that
// keeps all chunks and then copies them has a peak of about 2x the object; one that copies each
// chunk into a pre-sized buffer as it arrives peaks at about 1x. A positive control reads the
// same object through the SDK's own `collect().into_bytes()`: it must show the 2x peak, or the
// counter would be blind.
//
// Own process: it installs a global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

use s3_get_into_support::mock::{on_mock_server_thread, MockObject};
use s3_get_into_support::{init, key, pattern, BUCKET};
use s3dlio::object_store::{ObjectStore, S3ObjectStore};

const MIB: usize = 1024 * 1024;

/// A measured window is open.
static TRACKING: AtomicBool = AtomicBool::new(false);
/// Bytes allocated and not yet freed since the window opened (all counted threads).
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// The most `LIVE` reached in the window.
static PEAK: AtomicIsize = AtomicIsize::new(0);

/// One measured test at a time: the count is process-wide.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn counted() -> bool {
    TRACKING.load(Ordering::Relaxed) && !on_mock_server_thread()
}

fn grow(size: usize) {
    if counted() {
        let size = isize::try_from(size).unwrap_or(isize::MAX);
        let now = LIVE.fetch_add(size, Ordering::Relaxed).saturating_add(size);
        PEAK.fetch_max(now, Ordering::Relaxed);
    }
}

fn shrink(size: usize) {
    if counted() {
        LIVE.fetch_sub(
            isize::try_from(size).unwrap_or(isize::MAX),
            Ordering::Relaxed,
        );
    }
}

struct Counting;

// SAFETY: every method forwards to the system allocator unchanged, after noting the size; the
// bookkeeping uses atomics and a const-initialised thread-local, which allocate nothing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrink(layout.size());
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        shrink(layout.size());
        grow(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `fut`, returning the peak of bytes live at once (above the start of the window) on every
/// thread except the mock server's.
async fn measured<T>(fut: impl std::future::Future<Output = T>) -> (T, usize) {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::SeqCst);
    let out = fut.await;
    TRACKING.store(false, Ordering::SeqCst);
    (
        out,
        usize::try_from(PEAK.load(Ordering::Relaxed)).unwrap_or(0),
    )
}

async fn store_and_object(tag: &str, len: usize) -> (S3ObjectStore, String, Vec<u8>) {
    let (port, state) = init(&[]);
    let store = S3ObjectStore::for_endpoint(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("a per-endpoint store for the mock");
    let key = key(tag);
    let data = pattern(len, 7);
    state.put_object(BUCKET, &key, MockObject::new(data.clone()));
    let uri = format!("s3://{BUCKET}/{key}");
    (store, uri, data)
}

#[tokio::test(flavor = "current_thread")]
async fn a_per_endpoint_get_receives_into_one_presized_buffer() {
    let _serial = SERIAL.lock().await;
    let n = 32 * MIB;
    let (store, uri, data) = store_and_object("r2-get", n).await;
    // Warm-up: connection state and the first request are not what is measured.
    let (warm_store, warm_uri, _) = store_and_object("r2-get-warm", 1024).await;
    warm_store.get(&warm_uri).await.expect("warm-up read");

    let (got, peak) = measured(async { store.get(&uri).await.expect("get") }).await;
    assert_eq!(got.len(), n);
    assert!(got[..] == data[..], "wrong content");
    assert!(
        peak < n + n / 2,
        "RED: a {n}-byte get had {peak} bytes live at once (the chunks AND the buffer they were \
         copied into)"
    );

    // Positive control: the SDK's own collect().into_bytes() shows the 2x peak.
    let (port, _) = init(&[]);
    let client = s3_get_into_support::client_for(port).await;
    let key_only = uri
        .trim_start_matches(&format!("s3://{BUCKET}/"))
        .to_string();
    let (len, control) = measured(async {
        let resp = client
            .get_object()
            .bucket(BUCKET)
            .key(&key_only)
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
    assert_eq!(len, n);
    assert!(
        control >= n + n / 2,
        "the control read peaked at only {control} bytes: the counter cannot see the extra copy"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_per_endpoint_get_range_receives_into_one_presized_buffer() {
    let _serial = SERIAL.lock().await;
    let n = 32 * MIB;
    let (store, uri, data) = store_and_object("r2-range", n).await;
    let (warm_store, warm_uri, _) = store_and_object("r2-range-warm", 1024).await;
    warm_store.get(&warm_uri).await.expect("warm-up read");

    let offset = 3 * MIB as u64 + 5;
    let len = 24 * MIB as u64;
    let (got, peak) = measured(async {
        store
            .get_range(&uri, offset, Some(len))
            .await
            .expect("get_range")
    })
    .await;
    let start = offset as usize;
    assert_eq!(got.len(), len as usize);
    assert!(
        got[..] == data[start..start + len as usize],
        "wrong content"
    );
    let bound = len as usize + len as usize / 2;
    assert!(
        peak < bound,
        "RED: a {len}-byte get_range had {peak} bytes live at once"
    );
}
