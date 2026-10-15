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

struct AlignedOwner(s3dlio::memory::AlignedBuf);
impl AsRef<[u8]> for AlignedOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

static SERIAL: Mutex<()> = Mutex::const_new(());

const MIB: usize = 1024 * 1024;
/// file:// writes straight from the caller's `Bytes`: no allocation proportional to it.
const FILE_PAYLOAD: usize = 64 * MIB;
/// direct:// copies an UNALIGNED source ONCE, into an aligned bounce buffer (sized to the
/// object, capped at 1 GiB), never twice; a page-aligned source is written in place.
const DIRECT_PAYLOAD: usize = 128 * MIB;
/// Bookkeeping slack on top of the allowed buffers.
const SLACK: usize = 8 * MIB;

fn scratch_dir() -> tempfile::TempDir {
    // O_DIRECT is unsupported on tmpfs: S3DLIO_TEST_DIR (a real filesystem) or the cargo target dir.
    let base = std::env::var_os("S3DLIO_TEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_TARGET_TMPDIR").into());
    std::fs::create_dir_all(&base).unwrap();
    tempfile::Builder::new()
        .prefix("put-no-copy-238-")
        .tempdir_in(base)
        .expect("temp dir")
}

async fn assert_put_does_not_copy(
    store: &dyn ObjectStore,
    scheme: &str,
    payload: usize,
    allowance: usize,
    page_aligned: bool,
) {
    let _guard = SERIAL.lock().await;
    let dir = scratch_dir();
    let uri = format!("{scheme}://{}/obj.bin", dir.path().display());
    let data = if page_aligned {
        let mut buf = s3dlio::memory::AlignedBuf::new(payload, 4096).unwrap();
        buf.as_mut_slice().fill(0xA7);
        Bytes::from_owner(AlignedOwner(buf))
    } else {
        Bytes::from(vec![0xA7u8; payload])
    };
    assert_eq!(data.as_ptr() as usize % 4096 == 0, page_aligned);

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
    assert_put_does_not_copy(&store, "file", FILE_PAYLOAD, 0, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_put_of_an_unaligned_source_makes_exactly_one_copy() {
    let store = ConfigurableFileSystemObjectStore::new(FileSystemConfig::direct_io());
    // One bounce buffer the size of the object (+ slack), not two copies.
    assert_put_does_not_copy(&store, "direct", DIRECT_PAYLOAD, DIRECT_PAYLOAD, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_put_of_a_page_aligned_source_makes_no_copy() {
    let store = ConfigurableFileSystemObjectStore::new(FileSystemConfig::direct_io());
    assert_put_does_not_copy(&store, "direct", DIRECT_PAYLOAD, 0, true).await;
}
