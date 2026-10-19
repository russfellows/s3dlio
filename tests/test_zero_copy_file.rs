// tests/test_zero_copy_file.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Stage G (Rule #1, zero copy everywhere), file:// and direct:// local stores:
//   W5  `FileSystemWriter::write_chunk` / `write_owned_bytes` (file://) -- tokio's `File::write_all`
//       copied every chunk into its own internal buffer.
//   R6  `file://` `get` / `get_range` -- tokio's `File::read*` copied through its internal buffer.
//   R7  `direct://` `get_range` -- the requested subrange was copied out of the aligned read
//       buffer into a new allocation.
//
// Proof is an allocation bound: a counting global allocator totals the bytes allocated during the
// call.  A read may allocate the destination (object size) and a small constant; a write may
// allocate only a small constant.  The copy tokio makes is up to 2 MiB per call, which these
// sizes put far over the bound.  The allocator is process-wide, so the tests run one after
// another under a lock.

use bytes::Bytes;
use s3dlio::file_store::FileSystemObjectStore;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::ObjectStore;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::Mutex;

#[path = "common/odirect.rs"]
mod odirect;

struct CountingAlloc {
    armed: AtomicBool,
    total: AtomicUsize,
    /// When set, 4096-aligned allocations of exactly `POOL_BUFFER` bytes are not counted: they
    /// are the buffers of a buffer pool, which grows on demand at moments the test cannot pin
    /// down (they are reused, not per-range copies).
    ignore_pool_buffers: AtomicBool,
}

/// The buffer size of the pool in `item5_...`.
const POOL_BUFFER: usize = 2 * 1024 * 1024;

impl CountingAlloc {
    fn add(&self, n: usize) {
        if self.armed.load(Ordering::Relaxed) {
            self.total.fetch_add(n, Ordering::Relaxed);
        }
    }

    fn add_layout(&self, layout: &Layout) {
        if self.ignore_pool_buffers.load(Ordering::Relaxed)
            && layout.align() == 4096
            && layout.size() == POOL_BUFFER
        {
            return;
        }
        self.add(layout.size());
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            self.add_layout(&layout);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            self.add(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            self.add(new_size.saturating_sub(layout.size()));
        }
        p
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc {
    armed: AtomicBool::new(false),
    total: AtomicUsize::new(0),
    ignore_pool_buffers: AtomicBool::new(false),
};

static SERIAL: Mutex<()> = Mutex::const_new(());

const MIB: usize = 1024 * 1024;
/// Allowed allocation beyond the destination: paths, futures, task state.
const SLACK: usize = 256 * 1024;

fn arm() {
    ALLOC.total.store(0, Ordering::SeqCst);
    ALLOC.armed.store(true, Ordering::SeqCst);
}

fn disarm() -> usize {
    ALLOC.armed.store(false, Ordering::SeqCst);
    ALLOC.total.load(Ordering::SeqCst)
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn uri_of(dir: &tempfile::TempDir, name: &str) -> String {
    format!("file://{}/{}", dir.path().display(), name)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn w5_write_chunk_does_not_copy_into_tokio_buffer() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store = FileSystemObjectStore::new();
    let uri = uri_of(&dir, "w5.bin");
    let chunk = pattern(8 * MIB);

    let mut writer = store.get_writer(&uri).await.unwrap();
    arm();
    writer.write_chunk(&chunk).await.unwrap();
    let allocated = disarm();
    writer.finalize().await.unwrap();

    assert!(
        allocated < SLACK,
        "write_chunk of {} bytes allocated {allocated} bytes: the chunk is being copied",
        chunk.len()
    );
    assert_eq!(std::fs::read(dir.path().join("w5.bin")).unwrap(), chunk);
}

#[tokio::test(flavor = "current_thread")]
async fn w5_write_chunk_works_on_current_thread_runtime() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store = FileSystemObjectStore::new();
    let uri = uri_of(&dir, "w5ct.bin");
    let chunk = pattern(3 * MIB + 17);

    let mut writer = store.get_writer(&uri).await.unwrap();
    writer.write_chunk(&chunk).await.unwrap();
    writer.write_chunk(&chunk[..5]).await.unwrap();
    assert_eq!(writer.bytes_written(), (chunk.len() + 5) as u64);
    writer.finalize().await.unwrap();

    let mut want = chunk.clone();
    want.extend_from_slice(&chunk[..5]);
    assert_eq!(std::fs::read(dir.path().join("w5ct.bin")).unwrap(), want);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn w5_write_owned_bytes_does_not_copy_and_keeps_order() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store = FileSystemObjectStore::new();
    let uri = uri_of(&dir, "w5o.bin");
    let first = pattern(8 * MIB);
    let second = vec![0xA5u8; 1000];

    let mut writer = store.get_writer(&uri).await.unwrap();
    let owned = first.clone();
    arm();
    writer.write_owned_bytes(owned).await.unwrap();
    let allocated = disarm();
    writer.write_chunk(&second).await.unwrap();
    writer.finalize().await.unwrap();

    assert!(
        allocated < SLACK,
        "write_owned_bytes of {} bytes allocated {allocated} bytes: the payload is being copied",
        first.len()
    );
    let mut want = first;
    want.extend_from_slice(&second);
    assert_eq!(std::fs::read(dir.path().join("w5o.bin")).unwrap(), want);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r6_file_get_does_not_copy_through_tokio_buffer() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store = FileSystemObjectStore::new();
    let size = 3 * MIB;
    let data = pattern(size);
    std::fs::write(dir.path().join("r6.bin"), &data).unwrap();
    let uri = uri_of(&dir, "r6.bin");

    arm();
    let got = store.get(&uri).await.unwrap();
    let allocated = disarm();

    assert_eq!(got.as_ref(), data.as_slice());
    assert!(
        allocated < size + SLACK,
        "get of {size} bytes allocated {allocated} bytes: more than the destination"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r6_file_get_range_does_not_copy_through_tokio_buffer() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store = FileSystemObjectStore::new();
    let data = pattern(8 * MIB);
    std::fs::write(dir.path().join("r6r.bin"), &data).unwrap();
    let uri = uri_of(&dir, "r6r.bin");
    let (offset, length) = (12345usize, 5 * MIB);

    arm();
    let got = store
        .get_range(&uri, offset as u64, Some(length as u64))
        .await
        .unwrap();
    let allocated = disarm();

    assert_eq!(got.as_ref(), &data[offset..offset + length]);
    assert!(
        allocated < length + SLACK,
        "get_range of {length} bytes allocated {allocated} bytes: more than the destination"
    );

    // Edge cases keep their behaviour: clamped at EOF, empty past EOF, to-the-end.
    let tail = store
        .get_range(&uri, (data.len() - 10) as u64, Some(100))
        .await
        .unwrap();
    assert_eq!(tail.as_ref(), &data[data.len() - 10..]);
    let none = store
        .get_range(&uri, data.len() as u64 + 5, Some(10))
        .await
        .unwrap();
    assert!(none.is_empty());
    let rest = store.get_range(&uri, 1000, None).await.unwrap();
    assert_eq!(rest.as_ref(), &data[1000..]);
}

/// `direct://` with an owned (non-pooled) aligned read buffer: the returned subrange must be a
/// slice of the buffer the kernel filled, not a copy.  The aligned buffer starts on a page
/// boundary and the subrange begins `offset % alignment` bytes into it, so a zero-copy result
/// has `ptr % alignment == offset % alignment`; a malloc'd copy is 16-byte aligned and can never
/// have the odd remainder chosen here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r7_direct_get_range_returns_slice_of_owned_aligned_buffer() {
    let _g = SERIAL.lock().await;
    let Some(dir) = odirect::odirect_scratch_dir("zc-r7-") else {
        return;
    };
    let config = FileSystemConfig {
        direct_io: true,
        buffer_pool: None,
        ..FileSystemConfig::default()
    };
    let alignment = config.alignment;
    let store = ConfigurableFileSystemObjectStore::new(config);
    let data = pattern(4 * MIB);
    std::fs::write(dir.path().join("r7.bin"), &data).unwrap();
    let uri = format!("direct://{}/r7.bin", dir.path().display());
    let (offset, length) = (alignment + 101, MIB);

    arm();
    let got: Bytes = store
        .get_range(&uri, offset as u64, Some(length as u64))
        .await
        .unwrap();
    let allocated = disarm();

    assert_eq!(got.as_ref(), &data[offset..offset + length]);
    assert_eq!(
        got.as_ptr() as usize % alignment,
        offset % alignment,
        "the returned range is not a slice of the page-aligned read buffer: it was copied"
    );
    assert!(
        allocated < length + 2 * alignment + SLACK,
        "get_range of {length} bytes allocated {allocated} bytes: the range was copied"
    );

    // Short range at EOF and past EOF keep their behaviour.
    let tail = store
        .get_range(&uri, (data.len() - 10) as u64, Some(100))
        .await
        .unwrap();
    assert_eq!(tail.as_ref(), &data[data.len() - 10..]);
    let none = store
        .get_range(&uri, data.len() as u64 + 5, Some(10))
        .await
        .unwrap();
    assert!(none.is_empty());
}

/// The `direct://` writer in buffered mode (`direct_io == false`, also the mode a filesystem
/// without O_DIRECT falls back to) had the same tokio copy as the `file://` writer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn w5_direct_writer_buffered_mode_does_not_copy_into_tokio_buffer() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let config = FileSystemConfig {
        direct_io: false,
        ..FileSystemConfig::default()
    };
    let store = ConfigurableFileSystemObjectStore::new(config);
    let uri = format!("direct://{}/w5d.bin", dir.path().display());
    let chunk = pattern(8 * MIB);

    let mut writer = store.get_writer(&uri).await.unwrap();
    arm();
    writer.write_chunk(&chunk).await.unwrap();
    let allocated = disarm();
    writer.finalize().await.unwrap();

    assert!(
        allocated < SLACK,
        "write_chunk of {} bytes allocated {allocated} bytes: the chunk is being copied",
        chunk.len()
    );
    assert_eq!(std::fs::read(dir.path().join("w5d.bin")).unwrap(), chunk);
}

/// The `direct://` range-read fallbacks (buffered mode) also read through tokio's buffer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r6_direct_buffered_get_range_does_not_copy_through_tokio_buffer() {
    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    // The O_DIRECT-disabled store has no aligned fast path: the range is read by the buffered
    // fallback.  (A filesystem without O_DIRECT reaches the same fallback after a failed try.)
    let config = FileSystemConfig {
        direct_io: false,
        ..FileSystemConfig::default()
    };
    let store = ConfigurableFileSystemObjectStore::new(config);
    let data = pattern(8 * MIB);
    std::fs::write(dir.path().join("r6d.bin"), &data).unwrap();
    let uri = format!("direct://{}/r6d.bin", dir.path().display());
    let (offset, length) = (777usize, 5 * MIB);

    arm();
    let got = store
        .get_range(&uri, offset as u64, Some(length as u64))
        .await
        .unwrap();
    let allocated = disarm();

    assert_eq!(got.as_ref(), &data[offset..offset + length]);
    assert!(
        allocated < length + SLACK,
        "get_range of {length} bytes allocated {allocated} bytes: more than the destination"
    );
}

/// R3-style range engine on `file://`: every range is read straight into its segment of the
/// master buffer. The compatibility path allocated a `Bytes` per range and copied it into the
/// master buffer (about 2x the object in allocations).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r6_file_range_engine_reads_straight_into_the_master_buffer() {
    use s3dlio::file_store::FileSystemConfig as FileConfig;
    use s3dlio::range_engine_generic::RangeEngineConfig;

    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let config = FileConfig {
        enable_range_engine: true,
        range_engine: RangeEngineConfig {
            chunk_size: MIB,
            max_concurrent_ranges: 4,
            min_split_size: MIB as u64,
            ..RangeEngineConfig::default()
        },
        ..FileConfig::default()
    };
    let store = FileSystemObjectStore::with_config(config);
    let size = 8 * MIB + 123;
    let data = pattern(size);
    std::fs::write(dir.path().join("re.bin"), &data).unwrap();
    let uri = uri_of(&dir, "re.bin");

    arm();
    let got = store.get(&uri).await.unwrap();
    let allocated = disarm();

    assert_eq!(got.len(), size);
    assert!(
        got.as_ref() == data.as_slice(),
        "range engine returned wrong bytes"
    );
    assert!(
        allocated < size + SLACK,
        "range-engine get of {size} bytes allocated {allocated} bytes: ranges were copied"
    );
}

/// Same for the `direct://` store in buffered mode (`direct_io == false`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r6_direct_buffered_range_engine_reads_straight_into_the_master_buffer() {
    use s3dlio::range_engine_generic::RangeEngineConfig;

    let _g = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let config = FileSystemConfig {
        direct_io: false,
        enable_range_engine: true,
        range_engine: RangeEngineConfig {
            chunk_size: MIB,
            max_concurrent_ranges: 4,
            min_split_size: MIB as u64,
            ..RangeEngineConfig::default()
        },
        ..FileSystemConfig::default()
    };
    let store = ConfigurableFileSystemObjectStore::new(config);
    let size = 8 * MIB + 123;
    let data = pattern(size);
    std::fs::write(dir.path().join("red.bin"), &data).unwrap();
    let uri = format!("direct://{}/red.bin", dir.path().display());

    arm();
    let got = store.get(&uri).await.unwrap();
    let allocated = disarm();

    assert_eq!(got.len(), size);
    assert!(
        got.as_ref() == data.as_slice(),
        "range engine returned wrong bytes"
    );
    assert!(
        allocated < size + SLACK,
        "range-engine get of {size} bytes allocated {allocated} bytes: ranges were copied"
    );
}

/// Fix round, item 3: `get_range(uri, offset, Some(u64::MAX))` is "to the end of the file" on
/// every local store (it was on `direct://` before the buffered fallback was rewritten, which
/// then tried to allocate `usize::MAX` bytes and panicked). A length beyond the end of the file
/// is clamped to it; nothing panics.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn item3_direct_get_range_with_a_huge_length_reads_to_the_end() {
    let _g = SERIAL.lock().await;
    let data = pattern(3 * MIB + 77);
    let offset = 1000usize;

    // buffered direct:// store
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("h.bin"), &data).unwrap();
    let buffered = ConfigurableFileSystemObjectStore::new(FileSystemConfig {
        direct_io: false,
        ..FileSystemConfig::default()
    });
    let uri = format!("direct://{}/h.bin", dir.path().display());
    for length in [
        u64::MAX,
        u64::MAX - 1,
        100 * MIB as u64,
        (data.len() - offset) as u64 + 1,
    ] {
        let got = buffered
            .get_range(&uri, offset as u64, Some(length))
            .await
            .unwrap_or_else(|e| panic!("length {length}: {e:#}"));
        assert_eq!(got.as_ref(), &data[offset..], "length {length}");
    }
    let past = buffered
        .get_range(&uri, data.len() as u64 + 10, Some(u64::MAX))
        .await
        .unwrap();
    assert!(past.is_empty(), "offset past the end of the file");

    // the file:// store
    let fstore = FileSystemObjectStore::new();
    let furi = uri_of(&dir, "h.bin");
    let got = fstore
        .get_range(&furi, offset as u64, Some(u64::MAX))
        .await
        .unwrap();
    assert_eq!(got.as_ref(), &data[offset..]);

    // O_DIRECT direct:// store (skipped on a filesystem without O_DIRECT)
    if let Some(odir) = odirect::odirect_scratch_dir("zc-i3-") {
        std::fs::write(odir.path().join("h.bin"), &data).unwrap();
        let direct = ConfigurableFileSystemObjectStore::new(FileSystemConfig {
            direct_io: true,
            buffer_pool: None,
            ..FileSystemConfig::default()
        });
        let duri = format!("direct://{}/h.bin", odir.path().display());
        let got = direct
            .get_range(&duri, offset as u64, Some(u64::MAX))
            .await
            .unwrap();
        assert_eq!(got.as_ref(), &data[offset..]);
    }
}

/// Fix round, item 5: the `O_DIRECT` range engine with a buffer POOL copied every range twice
/// (pool buffer -> fresh `BytesMut` -> master-buffer window). The single bounce copy (an
/// unaligned destination cannot be read into with O_DIRECT) goes straight into the window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn item5_direct_range_engine_with_a_pool_makes_one_bounce_copy() {
    use s3dlio::memory::BufferPoolConfig;
    use s3dlio::range_engine_generic::RangeEngineConfig;

    let _g = SERIAL.lock().await;
    let Some(dir) = odirect::odirect_scratch_dir("zc-i5-") else {
        return;
    };
    let pool = BufferPoolConfig {
        capacity: 4,
        buffer_size: 2 * MIB,
        alignment: 4096,
    }
    .build()
    .unwrap();
    // Grow the pool before measuring: its buffers are allocated on demand, once. Ranges are read
    // one at a time so that no range ever has to grow the pool while the allocations are counted.
    let mut warm = Vec::new();
    for _ in 0..4 {
        warm.push(pool.take().await.unwrap());
    }
    for b in warm {
        pool.give(b);
    }
    // `BufferPool::new` also prefills `capacity` buffers on the background runtime; let that finish
    // so it cannot land inside the counted window.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let config = FileSystemConfig {
        direct_io: true,
        buffer_pool: Some(pool),
        enable_range_engine: true,
        range_engine: RangeEngineConfig {
            chunk_size: MIB,
            max_concurrent_ranges: 1,
            min_split_size: MIB as u64,
            ..RangeEngineConfig::default()
        },
        ..FileSystemConfig::default()
    };
    let store = ConfigurableFileSystemObjectStore::new(config);
    let size = 8 * MIB + 123;
    let data = pattern(size);
    std::fs::write(dir.path().join("p.bin"), &data).unwrap();
    let uri = format!("direct://{}/p.bin", dir.path().display());

    ALLOC.ignore_pool_buffers.store(true, Ordering::SeqCst);
    arm();
    let got = store.get(&uri).await.unwrap();
    let allocated = disarm();
    ALLOC.ignore_pool_buffers.store(false, Ordering::SeqCst);

    assert_eq!(got.len(), size);
    assert!(got.as_ref() == data.as_slice(), "wrong bytes");
    assert!(
        allocated < size + SLACK,
        "pooled O_DIRECT range-engine get of {size} bytes allocated {allocated} bytes: \
         every range was copied into an intermediate buffer first"
    );
}
