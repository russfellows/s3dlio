// tests/test_direct_writer_248_249.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #248: `create_aligned_buffer` allocated with a 4096-aligned Layout and the buffer was later
//       freed through a `Vec` with an alignment-1 layout (undefined behaviour). A global
//       allocator here records the alignment of every over-aligned allocation and counts any
//       deallocation or reallocation that uses a different one.
// #249: the direct:// streaming writer copied every chunk (`chunk.to_vec()`) and fsynced on
//       every flush. Large allocations are counted (a per-chunk copy shows up as one per
//       chunk) and write() system calls come from /proc/self/io. (The fsync count has a
//       unit test in src/file_store_direct.rs next to the counting hook.)
//
// The allocator and /proc counters are process-wide, so the tests run one after another.

use bytes::Bytes;
use s3dlio::file_store_direct::{DirectIOWriter, FileSystemConfig};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::ObjectWriter;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;

const SLOTS: usize = 1 << 14;
/// Marks a freed slot: the probe chain through it continues (a plain 0 would end the chain
/// and hide later entries); `record` may reuse it.
const TOMBSTONE: usize = usize::MAX;
const OVER_ALIGNED: usize = 64;
const BIG: usize = 1024 * 1024;

struct Tracking {
    ptrs: [AtomicUsize; SLOTS],
    aligns: [AtomicUsize; SLOTS],
    mismatches: AtomicUsize,
    big_allocs: AtomicUsize,
    live: AtomicUsize,
    peak: AtomicUsize,
}

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicUsize = AtomicUsize::new(0);

impl Tracking {
    const fn new() -> Self {
        Self {
            ptrs: [ZERO; SLOTS],
            aligns: [ZERO; SLOTS],
            mismatches: AtomicUsize::new(0),
            big_allocs: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn record(&self, ptr: *mut u8, layout: Layout) {
        let now = self.live.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        self.peak.fetch_max(now, Ordering::Relaxed);
        if layout.size() >= BIG {
            self.big_allocs.fetch_add(1, Ordering::Relaxed);
        }
        if layout.align() >= OVER_ALIGNED {
            let start = (ptr as usize >> 6) % SLOTS;
            for i in 0..SLOTS {
                let slot = (start + i) % SLOTS;
                let cur = self.ptrs[slot].load(Ordering::Acquire);
                if (cur == 0 || cur == TOMBSTONE)
                    && self.ptrs[slot]
                        .compare_exchange(cur, ptr as usize, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                {
                    self.aligns[slot].store(layout.align(), Ordering::Release);
                    return;
                }
            }
        }
    }

    fn forget(&self, ptr: *mut u8, layout: Layout) {
        self.live.fetch_sub(layout.size(), Ordering::Relaxed);
        let start = (ptr as usize >> 6) % SLOTS;
        for i in 0..SLOTS {
            let slot = (start + i) % SLOTS;
            let p = self.ptrs[slot].load(Ordering::Acquire);
            if p == ptr as usize {
                if self.aligns[slot].load(Ordering::Acquire) != layout.align() {
                    self.mismatches.fetch_add(1, Ordering::Relaxed);
                }
                self.ptrs[slot].store(TOMBSTONE, Ordering::Release);
                return;
            }
            if p == 0 {
                return; // probe chain ended: not recorded
            }
        }
    }
}

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            self.record(p, layout);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            self.record(p, layout);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.forget(ptr, layout);
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.forget(ptr, layout);
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            self.record(
                p,
                Layout::from_size_align_unchecked(new_size, layout.align()),
            );
        }
        p
    }
}

#[global_allocator]
static ALLOC: Tracking = Tracking::new();

static SERIAL: Mutex<()> = Mutex::const_new(());

const MIB: usize = 1024 * 1024;
const ALIGN: usize = 4096;

#[path = "common/odirect.rs"]
mod odirect;

fn scratch_dir() -> Option<tempfile::TempDir> {
    odirect::odirect_scratch_dir("direct-writer-248-249-")
}

fn syscw() -> u64 {
    let text = std::fs::read_to_string("/proc/self/io").expect("/proc/self/io");
    text.lines()
        .find_map(|l| l.strip_prefix("syscw:"))
        .and_then(|v| v.trim().parse().ok())
        .expect("syscw")
}

/// O_DIRECT config WITHOUT the pre-allocating range-read buffer pool: `FileSystemConfig::direct_io()`
/// spawns a task that allocates 32 x 64 MiB in the background, which these allocation counts must
/// not see.
fn direct_config() -> FileSystemConfig {
    FileSystemConfig {
        direct_io: true,
        ..FileSystemConfig::default()
    }
}

fn aligned_chunk(len: usize, fill: u8) -> AlignedBuf {
    let mut b = AlignedBuf::new(len, ALIGN).unwrap();
    b.as_mut_slice().fill(fill);
    b
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn writer_frees_every_aligned_allocation_with_its_own_layout() {
    let _g = SERIAL.lock().await;
    let Some(dir) = scratch_dir() else { return };
    let path = dir.path().join("w.bin");
    let before = ALLOC.mismatches.load(Ordering::Relaxed);
    {
        let mut w = Box::new(
            DirectIOWriter::new(path.clone(), direct_config())
                .await
                .unwrap(),
        );
        let data = vec![7u8; 5 * MIB + 11];
        for c in data.chunks(MIB + 3) {
            w.write_chunk(c).await.unwrap();
        }
        w.finalize().await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), data);
    }
    // Also a writer that is dropped after cancel.
    {
        let mut w = Box::new(
            DirectIOWriter::new(dir.path().join("c.bin"), direct_config())
                .await
                .unwrap(),
        );
        w.write_chunk(&vec![1u8; 3 * MIB]).await.unwrap();
        w.cancel().await.unwrap();
    }
    let bad = ALLOC.mismatches.load(Ordering::Relaxed) - before;
    assert_eq!(
        bad, 0,
        "{bad} over-aligned allocation(s) were freed with a different layout (#248)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn page_aligned_chunks_are_written_without_any_copy() {
    let _g = SERIAL.lock().await;
    let Some(dir) = scratch_dir() else { return };
    let path = dir.path().join("w.bin");
    let chunk = aligned_chunk(64 * MIB, 0xAB);

    let mut w = Box::new(
        DirectIOWriter::new(path.clone(), direct_config())
            .await
            .unwrap(),
    );
    let big0 = ALLOC.big_allocs.load(Ordering::Relaxed);
    let w0 = syscw();
    for _ in 0..3 {
        w.write_chunk(chunk.as_slice()).await.unwrap();
    }
    let writes = syscw() - w0;
    let big = ALLOC.big_allocs.load(Ordering::Relaxed) - big0;
    w.finalize().await.unwrap();

    assert_eq!(
        big, 0,
        "{big} large allocation(s): a chunk was copied (#249)"
    );
    // One write() per 64 MiB chunk (tokio's file used to split them into 2 MiB pieces).
    assert!(writes < 16, "{writes} write() calls for 3 aligned chunks");
    let meta = std::fs::metadata(&path).unwrap();
    assert_eq!(meta.len() as usize, 192 * MIB);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unaligned_chunks_make_no_per_chunk_copy() {
    let _g = SERIAL.lock().await;
    let Some(dir) = scratch_dir() else { return };
    let path = dir.path().join("w.bin");
    let data = Bytes::from(vec![0x5Au8; 16 * MIB]);
    let mut w = Box::new(
        DirectIOWriter::new(path.clone(), direct_config())
            .await
            .unwrap(),
    );
    let big0 = ALLOC.big_allocs.load(Ordering::Relaxed);
    for _ in 0..8 {
        w.write_chunk(data.as_ref()).await.unwrap(); // 128 MiB in 16 MiB chunks
    }
    let big = ALLOC.big_allocs.load(Ordering::Relaxed) - big0;
    w.finalize().await.unwrap();
    // The staging buffer is the only large allocation; the old code allocated one per chunk.
    assert!(big <= 1, "{big} large allocations for 8 chunks (#249)");
    assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, 128 * MIB);
    let back = std::fs::read(&path).unwrap();
    assert!(back.iter().all(|&b| b == 0x5A));
}
