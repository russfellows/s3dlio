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

/// O_DIRECT config WITHOUT the pre-allocating range-read buffer pool: `direct_io()` spawns a task
/// that allocates 32 x 64 MiB in the background, which the peak-memory measurement must not see.
fn direct_config() -> FileSystemConfig {
    FileSystemConfig {
        direct_io: true,
        ..FileSystemConfig::default()
    }
}

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
/// direct:// copies an UNALIGNED source ONCE, through the shared warm bounce pool, never twice
/// and never into an object-sized buffer: the extra memory is the pool (the configured pairs of
/// `direct_io_bounce_size()` buffers per NUMA node). The payload is far larger than the pool so a
/// full copy cannot hide under the allowance.
const DIRECT_PAYLOAD: usize = 512 * MIB;
/// Bookkeeping slack on top of the allowed buffers.
const SLACK: usize = 8 * MIB;

#[path = "common/odirect.rs"]
mod odirect;

fn scratch_dir() -> Option<tempfile::TempDir> {
    odirect::odirect_scratch_dir("put-no-copy-238-")
}

async fn assert_put_does_not_copy(
    store: &dyn ObjectStore,
    scheme: &str,
    payload: usize,
    allowance: usize,
    page_aligned: bool,
) {
    let _guard = SERIAL.lock().await;
    let Some(dir) = scratch_dir() else { return };
    let uri = format!("{scheme}://{}/obj.bin", dir.path().display());
    let data = if page_aligned {
        let mut buf = s3dlio::memory::AlignedBuf::new(payload, 4096).unwrap();
        buf.as_mut_slice().fill(0xA7);
        Bytes::from_owner(AlignedOwner(buf))
    } else {
        Bytes::from(vec![0xA7u8; payload])
    };
    assert_eq!((data.as_ptr() as usize).is_multiple_of(4096), page_aligned);

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
async fn direct_put_of_an_unaligned_source_uses_only_the_bounded_pool() {
    let store = ConfigurableFileSystemObjectStore::new(direct_config());
    // The whole pool (every NUMA node of the host): nodes x pairs x 2 buffers of the configured size.
    let pool = s3dlio::direct_bounce::bounce_memory_cap();
    let before = s3dlio::direct_bounce::bounce_stats();
    assert_put_does_not_copy(&store, "direct", DIRECT_PAYLOAD, pool, false).await;
    let after = s3dlio::direct_bounce::bounce_stats();
    assert_eq!(
        after.implicit_fallbacks, before.implicit_fallbacks,
        "the PUT bypassed the pool with an implicit private buffer: 'only the bounded pool' is false"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn direct_put_of_a_page_aligned_source_makes_no_copy() {
    let store = ConfigurableFileSystemObjectStore::new(direct_config());
    assert_put_does_not_copy(&store, "direct", DIRECT_PAYLOAD, 0, true).await;
}
