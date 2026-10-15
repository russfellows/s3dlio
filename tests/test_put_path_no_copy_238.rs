// tests/test_put_path_no_copy_238.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Issue #238: the local backends made a SECOND full copy of the payload on PUT
// (`tokio::fs::write` copies its input into an owned buffer; the direct:// path copied
// it into a Vec before the bounce buffer).  These tests install a peak-tracking global
// allocator and assert that `put()` of a `Bytes` allocates far less than the payload.
//
// The allocator is process-wide, so the tests run one after another under a lock.

use bytes::Bytes;
use s3dlio::file_store::FileSystemObjectStore;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::ObjectStore;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;

struct PeakAlloc {
    live: AtomicUsize,
    peak: AtomicUsize,
}

impl PeakAlloc {
    const fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }
    fn reset(&self) {
        self.peak
            .store(self.live.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    fn peak_above_live_at_reset(&self, live_at_reset: usize) -> usize {
        self.peak
            .load(Ordering::Relaxed)
            .saturating_sub(live_at_reset)
    }
    fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }
    fn grow(&self, delta: usize) {
        let now = self.live.fetch_add(delta, Ordering::Relaxed) + delta;
        self.peak.fetch_max(now, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            self.grow(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.live.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            if new_size >= layout.size() {
                self.grow(new_size - layout.size());
            } else {
                self.live
                    .fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc::new();

static SERIAL: Mutex<()> = Mutex::const_new(());

const MIB: usize = 1024 * 1024;
/// file:// writes straight from the caller's `Bytes`: no allocation proportional to it.
const FILE_PAYLOAD: usize = 64 * MIB;
/// direct:// copies an UNALIGNED source through one bounce buffer (64 MiB by default, see
/// `constants::DEFAULT_DIRECT_IO_BOUNCE_SIZE`), never through a payload-sized copy: the
/// payload is 4x the bounce buffer so a full copy cannot hide under the allowance.
const DIRECT_PAYLOAD: usize = 256 * MIB;
/// Bookkeeping slack on top of the allowed buffers.
const SLACK: usize = 8 * MIB;

fn scratch_dir() -> tempfile::TempDir {
    // O_DIRECT is unsupported on tmpfs, so use the cargo target tmp dir, not /tmp.
    tempfile::Builder::new()
        .prefix("put-no-copy-238-")
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .expect("temp dir")
}

async fn assert_put_does_not_copy(
    store: &dyn ObjectStore,
    scheme: &str,
    payload: usize,
    allowance: usize,
) {
    let _guard = SERIAL.lock().await;
    let dir = scratch_dir();
    let uri = format!("{scheme}://{}/obj.bin", dir.path().display());
    let data = Bytes::from(vec![0xA7u8; payload]);

    let live_before = ALLOC.live();
    ALLOC.reset();
    store.put(&uri, data.clone()).await.expect("put");
    let extra = ALLOC.peak_above_live_at_reset(live_before);
    drop(data);

    let limit = allowance + SLACK;
    assert!(
        extra < limit,
        "{scheme}:// put of {payload} bytes allocated {extra} extra bytes at peak \
         (limit {limit}): the payload was copied"
    );
    let size = std::fs::metadata(dir.path().join("obj.bin")).unwrap().len();
    assert_eq!(size as usize, payload);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_put_makes_no_owned_copy() {
    let store = FileSystemObjectStore::new();
    assert_put_does_not_copy(&store, "file", FILE_PAYLOAD, 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_put_makes_no_payload_sized_copy() {
    let store = ConfigurableFileSystemObjectStore::new(FileSystemConfig::direct_io());
    assert_put_does_not_copy(
        &store,
        "direct",
        DIRECT_PAYLOAD,
        s3dlio::constants::direct_io_bounce_size(),
    )
    .await;
}
