// src/file_store_direct.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::checksum::Crc32c;
use crate::memory::{AlignedAllocError, AlignedBuf, BufferPool, BufferPoolConfig};
use crate::object_store::WriterOptions;
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use anyhow::{bail, Result};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::fs;
#[cfg(not(unix))]
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tracing::{debug, info, trace, warn};

use crate::constants::{
    DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD, DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE,
    DEFAULT_MIN_IO_SIZE, DEFAULT_PAGE_SIZE, MAX_SINGLE_IO_BYTES,
};
use crate::object_store::{CompressionConfig, ObjectMetadata, ObjectStore, ObjectWriter};

/// Configuration options for FileSystem operations
///
/// **DirectIO Performance Characteristics**:
/// - O_DIRECT bypasses page cache - already very fast for sequential reads
/// - Range parallelism provides **limited benefit** compared to network storage
/// - Alignment overhead (page-size rounding) adds complexity
/// - Expected improvement: 20-40% vs 30-50% for network storage
/// - **Recommendation**: Higher thresholds (16-32MB) and lower concurrency (8-16)
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    /// Enable O_DIRECT I/O to bypass page cache (Linux/Unix only)
    pub direct_io: bool,
    /// Buffer alignment for O_DIRECT (typically system page size)
    pub alignment: usize,
    /// Minimum I/O size for O_DIRECT operations
    pub min_io_size: usize,
    /// Enable O_SYNC for synchronous writes
    pub sync_writes: bool,
    /// Enable concurrent range downloads for large files
    /// Default: false (v0.9.6+) - DirectIO rarely benefits from range parallelism
    pub enable_range_engine: bool,
    /// Range engine configuration
    pub range_engine: RangeEngineConfig,
    /// Buffer pool for reusable aligned buffers (v0.9.9+)
    /// Eliminates allocation churn and improves performance by 15-20%
    pub buffer_pool: Option<Arc<BufferPool>>,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        // Get system page size for proper alignment
        let page_size = get_system_page_size();

        Self {
            direct_io: false,
            alignment: page_size,
            min_io_size: page_size,
            sync_writes: false,
            enable_range_engine: false, // Disabled by default due to O_DIRECT seek overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                min_split_size: DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            buffer_pool: None, // No pool by default (v0.9.9+)
        }
    }
}

impl FileSystemConfig {
    /// Create a new configuration with O_DIRECT enabled for AI/ML workloads
    pub fn direct_io() -> Self {
        let page_size = get_system_page_size();

        // Create buffer pool for DirectIO operations (v0.9.9+)
        // Pool size matches range concurrency to avoid blocking
        let buffer_pool = BufferPoolConfig {
            capacity: 32,                  // 32 buffers in pool
            buffer_size: 64 * 1024 * 1024, // 64MB per buffer (matches chunk_size)
            alignment: page_size,
        }
        .build()
        .ok();

        Self {
            direct_io: true,
            alignment: page_size,
            min_io_size: page_size,
            sync_writes: false,
            enable_range_engine: false, // Disabled by default due to O_DIRECT seek overhead (v0.9.6+)
            // DirectIO: higher threshold (16MB), lower concurrency (16)
            range_engine: RangeEngineConfig {
                chunk_size: 64 * 1024 * 1024, // 64MB chunks
                max_concurrent_ranges: 16,    // Lower concurrency for O_DIRECT
                min_split_size: DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            buffer_pool,
        }
    }

    /// Create a configuration optimized for high-performance AI/ML workloads
    pub fn high_performance() -> Self {
        let page_size = get_system_page_size();

        // Create buffer pool for high-performance DirectIO (v0.9.9+)
        let buffer_pool = BufferPoolConfig {
            capacity: 32,
            buffer_size: 64 * 1024 * 1024,
            alignment: page_size,
        }
        .build()
        .ok();

        Self {
            direct_io: true,
            alignment: page_size,
            min_io_size: DEFAULT_MIN_IO_SIZE,
            sync_writes: true,          // Ensure data hits storage
            enable_range_engine: false, // Disabled by default due to O_DIRECT seek overhead (v0.9.6+)
            // High performance: same as direct_io
            range_engine: RangeEngineConfig {
                chunk_size: 64 * 1024 * 1024,
                max_concurrent_ranges: 16,
                min_split_size: DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            buffer_pool,
        }
    }
}

/// Get the system page size for proper O_DIRECT alignment
/// The system page size (the O_DIRECT alignment default).
pub(crate) fn system_page_size() -> usize {
    get_system_page_size()
}

/// Get the system page size for proper O_DIRECT alignment
fn get_system_page_size() -> usize {
    #[cfg(unix)]
    {
        unsafe {
            let page_size = libc::sysconf(libc::_SC_PAGESIZE);
            if page_size > 0 {
                let detected_size = page_size as usize;

                // Validate the detected page size is within reasonable bounds
                // Most systems use 4KB pages, some use 8KB, 16KB, or 64KB
                if (512..=65536).contains(&detected_size) && detected_size.is_power_of_two() {
                    detected_size
                } else {
                    // Invalid page size detected, use safe default
                    DEFAULT_PAGE_SIZE
                }
            } else {
                DEFAULT_PAGE_SIZE // Fallback if sysconf fails
            }
        }
    }

    #[cfg(not(unix))]
    {
        DEFAULT_PAGE_SIZE // Default fallback for non-Unix systems
    }
}

/// Round `n` down to a multiple of `alignment`.
fn round_down(n: usize, alignment: usize) -> usize {
    n / alignment * alignment
}

/// Round `n` up to a multiple of `alignment`.
fn round_up(n: usize, alignment: usize) -> usize {
    n.div_ceil(alignment) * alignment
}

/// One planned I/O system call of an O_DIRECT transfer (issue #244).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectIoStep {
    /// Write `len` bytes straight from the caller's memory at `data[offset..]`, to file
    /// offset `offset`. Both the source address and `len` are alignment multiples.
    FromSource { offset: usize, len: usize },
    /// Copy `data_len` payload bytes (at `offset` in the payload and in the file) through
    /// the aligned bounce buffer, zero-fill it up to `io_len` (a multiple of the alignment)
    /// and transfer `io_len` bytes in one system call.
    ViaBounce {
        offset: usize,
        data_len: usize,
        io_len: usize,
    },
}

/// Plan the bounce-buffer transfers covering `len` bytes: consecutive steps of at most
/// `bounce_size` (rounded down to the alignment, never below one alignment unit); only the
/// last step can be shorter, and its `io_len` is padded up to the alignment.
fn plan_bounce_steps(len: usize, alignment: usize, bounce_size: usize) -> Vec<DirectIoStep> {
    let alignment = alignment.max(1);
    let bounce = round_down(bounce_size, alignment).max(alignment);
    let mut steps = Vec::with_capacity(len.div_ceil(bounce));
    let mut offset = 0;
    while offset < len {
        let data_len = (len - offset).min(bounce);
        steps.push(DirectIoStep::ViaBounce {
            offset,
            data_len,
            io_len: round_up(data_len, alignment),
        });
        offset += data_len;
    }
    steps
}

/// Plan an O_DIRECT write of `len` bytes whose first byte is at address `src_addr`.
///
/// * Source address aligned: the alignment-multiple prefix is written straight from the
///   caller's memory with no copy, as ONE step (split only at `max_single_io`, the kernel's
///   per-call limit); the unaligned tail, if any, is one zero-padded bounce block.
/// * Source address not aligned (O_DIRECT forbids using it): bounce-buffer-sized steps.
///
/// The caller truncates the file back to `len` afterwards, removing the tail padding.
pub(crate) fn plan_direct_write(
    len: usize,
    src_addr: usize,
    alignment: usize,
    bounce_size: usize,
    max_single_io: usize,
) -> Vec<DirectIoStep> {
    let alignment = alignment.max(1);
    if !src_addr.is_multiple_of(alignment) {
        return plan_bounce_steps(len, alignment, bounce_size);
    }
    let max_io = round_down(max_single_io, alignment).max(alignment);
    let aligned_len = round_down(len, alignment);
    let mut steps = Vec::new();
    let mut offset = 0;
    while offset < aligned_len {
        let step_len = (aligned_len - offset).min(max_io);
        steps.push(DirectIoStep::FromSource {
            offset,
            len: step_len,
        });
        offset += step_len;
    }
    if aligned_len < len {
        steps.push(DirectIoStep::ViaBounce {
            offset: aligned_len,
            data_len: len - aligned_len,
            io_len: alignment,
        });
    }
    steps
}

/// Process-wide counters of how `direct://` transfers were actually carried out, so a
/// test (or an operator) can tell a real O_DIRECT transfer from the buffered fallback that
/// silently takes over when O_DIRECT is refused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectIoCounters {
    /// PUTs completed with O_DIRECT writes.
    pub direct_puts: u64,
    /// PUTs where O_DIRECT failed and the object was rewritten with buffered I/O.
    pub put_fallbacks: u64,
    /// GETs completed with O_DIRECT reads.
    pub direct_gets: u64,
    /// GETs where O_DIRECT failed and the object was read with buffered I/O.
    pub get_fallbacks: u64,
}

static DIRECT_PUTS: AtomicU64 = AtomicU64::new(0);
static PUT_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static DIRECT_GETS: AtomicU64 = AtomicU64::new(0);
static GET_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the process-wide [`DirectIoCounters`].
pub fn direct_io_counters() -> DirectIoCounters {
    DirectIoCounters {
        direct_puts: DIRECT_PUTS.load(Ordering::Relaxed),
        put_fallbacks: PUT_FALLBACKS.load(Ordering::Relaxed),
        direct_gets: DIRECT_GETS.load(Ordering::Relaxed),
        get_fallbacks: GET_FALLBACKS.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
thread_local! {
    /// fsync calls made by this thread's direct writers (test hook for issue #249).
    static SYNC_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Writes a direct writer made straight from the caller's memory (test hook).
    static FROM_CALLER_WRITES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// `fsync`, counted in tests.
fn sync_file(file: &std::fs::File) -> std::io::Result<()> {
    #[cfg(test)]
    SYNC_CALLS.with(|c| c.set(c.get() + 1));
    file.sync_all()
}

#[cfg(unix)]
fn write_all_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(not(unix))]
fn write_all_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut f = file;
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(buf)
}

/// Run blocking file I/O from async code without stalling a runtime worker: on a
/// multi-thread runtime the worker hands its queue to another thread first; otherwise the
/// closure just runs inline.
fn blocking_io<R>(f: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

/// Allocation function used for bounce buffers (injectable so tests can simulate failure).
type AllocFn<'a> = &'a dyn Fn(usize, usize) -> Result<AlignedBuf, AlignedAllocError>;

/// PUT `data[base..]` through a private, per-call bounce buffer (the path for small objects, for
/// the benchmark's per-call strategy, and when the shared pool is unavailable). The buffer holds
/// one chunk; when it cannot be allocated the chunk size is halved (down to one alignment unit)
/// instead of aborting, and only then does this return an error (the caller falls back to
/// buffered I/O). Chunks are copied and written one after the other.
#[cfg(unix)]
fn per_call_bounce_put(
    file: &std::fs::File,
    data: &[u8],
    base: usize,
    alignment: usize,
    bounce_size: usize,
    alloc: AllocFn<'_>,
    implicit: bool,
) -> Result<()> {
    let len = data.len() - base;
    let mut bounce_size = bounce_size;
    loop {
        let steps = plan_bounce_steps(len, alignment, bounce_size);
        let need = steps
            .iter()
            .filter_map(|s| match s {
                DirectIoStep::ViaBounce { io_len, .. } => Some(*io_len),
                DirectIoStep::FromSource { .. } => None,
            })
            .max()
            .unwrap_or(0);
        let mut buf = match alloc(need, alignment) {
            Ok(b) => {
                if implicit {
                    // The pool could not serve this PUT: counted, outside the pooled cap.
                    crate::direct_bounce::note_implicit_bounce(
                        crate::direct_bounce::Direction::Put,
                        need,
                    );
                }
                b
            }
            Err(e) => {
                let smaller = round_down(need / 2, alignment);
                if smaller < alignment {
                    return Err(e.into());
                }
                warn!(
                    "cannot allocate a {need} byte O_DIRECT bounce buffer; retrying with {smaller}"
                );
                bounce_size = smaller;
                continue;
            }
        };
        for step in steps {
            if let DirectIoStep::ViaBounce {
                offset,
                data_len,
                io_len,
            } = step
            {
                // SAFETY: the buffer holds at least `io_len` bytes (sized for the largest
                // step); `[0, data_len)` is copied from `data`, `[data_len, io_len)` is
                // zero-filled, so the `io_len` bytes read below are initialised.
                let out = unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(base + offset),
                        buf.as_mut_ptr(),
                        data_len,
                    );
                    std::ptr::write_bytes(buf.as_mut_ptr().add(data_len), 0, io_len - data_len);
                    std::slice::from_raw_parts(buf.as_ptr(), io_len)
                };
                write_all_at(file, out, (base + offset) as u64)?;
            }
        }
        return Ok(());
    }
}

/// Write all of `data` to `file` (opened with O_DIRECT), then truncate the file to
/// `data.len()` when the last block was padded.
///
/// * A page-aligned source is written straight from the caller's memory, as one `write` per at
///   most 2 GiB; the unaligned tail is one zero-padded block.
/// * Any other source goes through the bounce buffers chosen by the process-wide
///   [`crate::direct_bounce`] configuration: the shared warm ping-pong pool by default (the copy
///   of chunk `k+1` overlaps the write of chunk `k`), or a private per-call buffer when the
///   pool is unavailable or cannot satisfy the alignment (counted in
///   [`crate::direct_bounce::BounceStats::implicit_fallbacks`]; availability first, outside the
///   pooled memory cap).
///
/// `write_all_at` resumes after a short write, so a partial write is never lost; if a short
/// write leaves the next offset unaligned the kernel refuses it (EINVAL) and the caller's
/// buffered fallback rewrites the whole file.
#[cfg(unix)]
pub(crate) fn write_all_direct(
    open: &(dyn Fn() -> std::io::Result<std::fs::File> + Sync),
    data: &[u8],
    alignment: usize,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    write_all_direct_with(
        open,
        data,
        alignment,
        crate::direct_bounce::config().put,
        &AlignedBuf::new,
        cancel,
    )
}

/// `open` creates/truncates and opens the target (O_DIRECT). It is called only once the transfer
/// can really start - for a pooled PUT after the bounce buffers have been leased, so a request
/// cancelled while it waits for them never truncates an existing object.
#[cfg(unix)]
fn write_all_direct_with(
    open: &(dyn Fn() -> std::io::Result<std::fs::File> + Sync),
    data: &[u8],
    alignment: usize,
    strategy: crate::direct_bounce::PutStrategy,
    alloc: AllocFn<'_>,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    use crate::direct_bounce::{pooled_put, PutChunk, PutStrategy};
    use std::sync::OnceLock;

    let alignment = alignment.max(1);
    let cap = match strategy {
        PutStrategy::Pooled => crate::direct_bounce::config().buffer_size,
        PutStrategy::PerCall { cap } => cap,
    };
    let steps = plan_direct_write(
        data.len(),
        data.as_ptr() as usize,
        alignment,
        cap,
        MAX_SINGLE_IO_BYTES,
    );

    let mut chunks = Vec::new();
    let mut direct_steps = Vec::new();
    for step in &steps {
        match *step {
            DirectIoStep::FromSource { offset, len } => direct_steps.push((offset, len)),
            DirectIoStep::ViaBounce {
                offset,
                data_len,
                io_len,
            } => chunks.push(PutChunk {
                offset,
                data_len,
                io_len,
            }),
        }
    }

    let file_cell: OnceLock<std::fs::File> = OnceLock::new();
    let ensure_open = || -> std::io::Result<()> {
        if file_cell.get().is_none() {
            let _ = file_cell.set(open()?);
        }
        Ok(())
    };
    let file = || -> std::io::Result<&std::fs::File> {
        file_cell
            .get()
            .ok_or_else(|| std::io::Error::other("direct write before the file was opened"))
    };

    // A page-aligned source only bounces its padded tail block (one page): that never leases
    // pool buffers, it would only make aligned PUTs queue behind unaligned ones.
    let pooled = matches!(strategy, PutStrategy::Pooled)
        && direct_steps.is_empty()
        && !chunks.is_empty()
        && pooled_put(
            &chunks,
            data,
            alignment,
            cancel,
            &ensure_open,
            &|off, bytes| write_all_at(file()?, bytes, off),
        )?;
    if !pooled {
        ensure_open()?;
        for &(offset, len) in &direct_steps {
            write_all_at(file()?, &data[offset..offset + len], offset as u64)?;
        }
        if let Some(first) = chunks.first() {
            // Implicit (counted) only when this PUT wanted the pool and could not get it; the
            // benchmark's `PerCall` strategy and the one-page tail of an aligned source are by
            // design outside the pool.
            let implicit = matches!(strategy, PutStrategy::Pooled) && direct_steps.is_empty();
            per_call_bounce_put(file()?, data, first.offset, alignment, cap, alloc, implicit)?;
        }
    }

    // Remove the zero padding of the last block (none when the length is aligned).
    if !data.len().is_multiple_of(alignment) {
        file()?.set_len(data.len() as u64)?;
    }
    Ok(())
}

/// Buffered (non-O_DIRECT) range read for the `direct://` store: plain `std::fs` positional
/// reads in a blocking task, straight into the destination Vec that becomes the returned `Bytes`
/// (Rule #1: `tokio::fs::File::read*` would copy every byte through tokio's internal buffer).
/// `Some(len)` reads `len` bytes, clamped to the end of the file (so `u64::MAX`, or any length
/// beyond the end, means "to the end of the file", as it does on `file://`; an offset at or past
/// the end is an empty result); `None` reads to the end of the file. The destination is
/// allocated fallibly: an impossible size is an error, never an allocation panic.
async fn read_range_buffered(path: &Path, offset: u64, length: Option<u64>) -> Result<Bytes> {
    let path = path.to_owned();
    let data = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        use std::io::{Read, Seek};
        let mut file = std::fs::File::open(&path)?;
        match length {
            Some(len) => {
                let remaining = file.metadata()?.len().saturating_sub(offset);
                let want = len.min(remaining);
                let want = usize::try_from(want)
                    .map_err(|_| anyhow::anyhow!("range read of {want} bytes is too large"))?;
                let mut buffer = Vec::new();
                buffer.try_reserve_exact(want).map_err(|_| {
                    anyhow::anyhow!("cannot allocate {want} bytes for the range read")
                })?;
                buffer.resize(want, 0);
                crate::file_store::read_exact_at_std(&file, &mut buffer, offset)?;
                Ok(buffer)
            }
            None => {
                let mut buffer = Vec::new();
                file.seek(std::io::SeekFrom::Start(offset))?;
                file.read_to_end(&mut buffer)?;
                Ok(buffer)
            }
        }
    })
    .await??;
    Ok(Bytes::from(data))
}

/// What a `direct://` aligned range read produced: the bytes, or (when the caller supplied a
/// window) the number of bytes copied into the start of that window.
enum RangeRead {
    Bytes(Bytes),
    Filled(usize),
}

/// A page-aligned buffer exposing `len` bytes from `start` (what the kernel filled) as the
/// owner of a `Bytes`, so a `direct://` GET hands back the buffer it read into, with no
/// copy, and the result is itself page-aligned (a later `direct://` PUT of it is one write).
struct AlignedBytes {
    buf: AlignedBuf,
    /// First exposed byte (0 for a whole-object GET; the in-buffer offset of a range GET).
    start: usize,
    len: usize,
}

impl AsRef<[u8]> for AlignedBytes {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: `[start, start + len)` was written by the kernel's read (see
        // `read_all_direct` and the range read); `start + len <= buf.len()`.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr().add(self.start), self.len) }
    }
}

/// `pread` into raw memory, retrying EINTR. Returns the bytes read (0 at EOF).
#[cfg(unix)]
fn pread_raw(
    file: &std::fs::File,
    buf: crate::direct_bounce::SendPtr,
    len: usize,
    offset: u64,
) -> std::io::Result<usize> {
    use std::os::fd::AsRawFd;
    loop {
        // SAFETY: the caller guarantees `buf` is valid for `len` bytes of writes.
        let n = unsafe {
            libc::pread(
                file.as_raw_fd(),
                buf.0 as *mut libc::c_void,
                len,
                offset as libc::off_t,
            )
        };
        if n >= 0 {
            return Ok(n as usize);
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

/// Read `file_size` bytes of `file` (opened with O_DIRECT) into `dest` (any alignment) through
/// the shared bounce pool: the read of chunk `k+1` overlaps the copy of chunk `k`. This is the
/// building block of a caller-supplied-buffer GET (`get_into`).
///
/// Errors when the file is shorter than `file_size` (it shrank), so the caller can fall back to
/// a buffered read.
///
/// # Safety
/// `dest` must be valid for `file_size` bytes of writes and not be used meanwhile.
#[cfg(unix)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "building block of the later public get_into; only tests call it so far"
    )
)]
pub(crate) unsafe fn read_all_direct_into(
    file: &std::fs::File,
    file_size: usize,
    alignment: usize,
    dest: *mut u8,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    use crate::direct_bounce::{pooled_get, SendPtr};

    let alignment = alignment.max(1);
    let dest = SendPtr(dest);
    let read_at = |buf, len, off| pread_raw(file, buf, len, off);
    if pooled_get(file_size, alignment, dest, cancel, &read_at)? {
        return Ok(());
    }
    // The pool could not serve the read: one private bounce buffer (implicit, counted, outside
    // the pooled cap).
    let size = crate::direct_bounce::config()
        .buffer_size
        .min(round_up(file_size, alignment));
    let buf = AlignedBuf::new(size, alignment)?;
    crate::direct_bounce::note_implicit_bounce(crate::direct_bounce::Direction::Get, size);
    let ptr = [SendPtr(buf.as_ptr() as *mut u8)];
    crate::direct_bounce::pingpong_get(file_size, alignment, size, dest, &ptr, &read_at, &|_| {})?;
    Ok(())
}

/// Read `file_size` bytes from `file` (opened with O_DIRECT) and return them as `Bytes` backed by
/// ONE page-aligned buffer (so a later `direct://` PUT of the result is copy-free).
///
/// By default (`S3DLIO_DIRECT_IO_GET_MODE=direct`) the buffer is read into directly, one `pread`
/// per at most [`MAX_SINGLE_IO_BYTES`]: zero copies, but fresh pages are faulted in on every
/// call. With `pooled` the object goes through the shared warm bounce pool instead (one copy into
/// the destination). Which is faster depends on load; see `docs/Environment_Variables.md`.
///
/// Errors when the file is shorter than `file_size` (it shrank), so the caller's buffered
/// fallback re-reads it, and when the aligned allocation fails.
#[cfg(unix)]
pub(crate) fn read_all_direct(
    file: &std::fs::File,
    file_size: usize,
    alignment: usize,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<Bytes> {
    read_all_direct_with(
        file,
        file_size,
        alignment,
        crate::direct_bounce::config().get,
        cancel,
    )
}

#[cfg(unix)]
fn read_all_direct_with(
    file: &std::fs::File,
    file_size: usize,
    alignment: usize,
    strategy: crate::direct_bounce::GetStrategy,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<Bytes> {
    use crate::direct_bounce::{pooled_get, GetStrategy, SendPtr};

    if file_size == 0 {
        return Ok(Bytes::new());
    }
    let alignment = alignment.max(1);
    let capacity = round_up(file_size, alignment);
    let buf = AlignedBuf::new(capacity, alignment)?;
    let dest = SendPtr(buf.as_ptr() as *mut u8);
    let read_at = |b, len, off| pread_raw(file, b, len, off);

    // SAFETY: `dest` is valid for `capacity >= file_size` bytes and exclusively ours.
    let pooled = matches!(strategy, GetStrategy::Pooled)
        && unsafe { pooled_get(file_size, alignment, dest, cancel, &read_at)? };
    if !pooled {
        let max_io = round_down(MAX_SINGLE_IO_BYTES, alignment).max(alignment);
        let mut total = 0usize;
        while total < file_size {
            // Whole alignment units at an aligned offset; the last read may come back short at EOF.
            let want = (capacity - total).min(max_io);
            // SAFETY: `[total, total + want)` lies inside the allocation; the kernel fills it.
            let n = pread_raw(
                file,
                SendPtr(unsafe { dest.0.add(total) }),
                want,
                total as u64,
            )?;
            if n == 0 {
                break; // EOF
            }
            total += n;
        }
        if total < file_size {
            bail!("direct read returned {total} of {file_size} bytes (file shrank)");
        }
    }
    // `AlignedBytes` exposes only `[0, file_size)` (all written by the reads above) through a raw
    // pointer, never through the safe slice accessors, so `buf` need not be marked initialised.
    Ok(Bytes::from_owner(AlignedBytes {
        buf,
        start: 0,
        len: file_size,
    }))
}

/// Sets its flag when dropped: held by the async fn of a `direct://` transfer so that, if the
/// future is dropped (cancelled), a blocking closure still WAITING for bounce buffers gives up
/// before it starts any I/O. A closure that already holds buffers runs to completion (and drops
/// its lease) regardless.
struct CancelOnDrop(Arc<std::sync::atomic::AtomicBool>);

impl CancelOnDrop {
    fn new() -> Self {
        Self(Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }
    fn flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.0)
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Streaming writer for direct I/O filesystem operations
///
/// With `direct_io` the file is opened O_DIRECT and written with large aligned writes:
/// * a chunk whose address is page-aligned (and nothing is staged) is written straight from
///   the caller's memory, no copy;
/// * any other data is copied once into an aligned staging buffer
///   ([`DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE`]) that is written whenever it fills;
/// * the unaligned tail is appended with buffered I/O at `finalize`.
///
/// Data is made durable ONCE, at `finalize` (like the other backends), unless
/// `FileSystemConfig::sync_writes` is set, which asks for an fsync after every write.
///
/// A chunk that starts off a page boundary after earlier data was staged is topped up to the
/// next boundary and the rest is written from the caller's memory when that rest happens to start
/// page-aligned; otherwise it keeps going through the staging buffer (one copy).
///
/// **Partial files.** The file is created when the writer is. If a write fails, or the writer is
/// dropped without `finalize()`, the file stays on disk holding only what had been flushed (a
/// multiple of the staging size; the staged remainder is lost) and is NOT removed. Call
/// `cancel()` to delete it.
pub struct DirectIOWriter {
    /// Buffered mode (`direct_io == false`). A plain `std::fs::File` written straight from the
    /// caller's bytes: `tokio::fs::File::write_all` would copy every chunk into tokio's own buffer.
    file: Option<Arc<std::fs::File>>,
    /// O_DIRECT mode.
    direct: Option<std::fs::File>,
    /// File offset of the next O_DIRECT write.
    direct_offset: u64,
    path: PathBuf,
    config: FileSystemConfig,
    bytes_written: u64,
    finalized: bool,
    /// Aligned staging buffer (allocated on first use) and how much of it is filled.
    staging: Option<AlignedBuf>,
    staged: usize,
    /// Capacity of the staging buffer, a multiple of the alignment.
    staging_cap: usize,
    hasher: Crc32c,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    compressed_bytes: u64,
}

impl DirectIOWriter {
    pub async fn new(path: PathBuf, config: FileSystemConfig) -> Result<Self> {
        Self::new_with_compression(path, config, CompressionConfig::None).await
    }

    pub async fn new_with_compression(
        path: PathBuf,
        config: FileSystemConfig,
        compression: CompressionConfig,
    ) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Append compression extension to path if needed
        // (the one suffix rule, shared with the checkpoint manifest names)
        let final_path = compression.stored_path(&path);

        // Create compressor if needed
        let compressor = match &compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => zstd::Encoder::new(Vec::new(), *level).ok(),
        };

        // Create and open the file for writing with appropriate flags
        let mut file = None;
        let mut direct = None;
        if config.direct_io && cfg!(unix) {
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                let open_path = final_path.clone();
                direct = Some(
                    tokio::task::spawn_blocking(move || {
                        std::fs::OpenOptions::new()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .custom_flags(libc::O_DIRECT)
                            .open(open_path)
                    })
                    .await??,
                );
            }
        } else {
            file = Some(Arc::new(
                fs::File::create(&final_path).await?.into_std().await,
            ));
        }

        let alignment = config.alignment.max(1);
        let staging_cap =
            round_down(DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE, alignment).max(alignment);

        Ok(Self {
            file,
            direct,
            direct_offset: 0,
            path: final_path,
            config,
            bytes_written: 0,
            finalized: false,
            staging: None,
            staged: 0,
            staging_cap,
            hasher: Crc32c::new(),
            compression,
            compressor,
            compressed_bytes: 0,
        })
    }

    /// Write `buf` (alignment-multiple length, and address when it is caller memory) to the
    /// O_DIRECT file at the current offset.
    fn write_direct(&mut self, buf: &[u8]) -> Result<()> {
        let Some(file) = self.direct.as_ref() else {
            bail!("Writer has been finalized or cancelled");
        };
        let offset = self.direct_offset;
        blocking_io(|| write_all_at(file, buf, offset))?;
        self.direct_offset += buf.len() as u64;
        if self.config.sync_writes {
            blocking_io(|| sync_file(file))?;
        }
        Ok(())
    }

    /// Write the first `n` staged bytes (a multiple of the alignment) to the O_DIRECT file.
    fn flush_staging(&mut self, n: usize) -> Result<()> {
        let (Some(file), Some(staging)) = (self.direct.as_ref(), self.staging.as_ref()) else {
            bail!("Writer has been finalized or cancelled");
        };
        // SAFETY: `[0, n)` of the staging buffer was filled by `stage_direct`.
        let buf = unsafe { std::slice::from_raw_parts(staging.as_ptr(), n) };
        let offset = self.direct_offset;
        blocking_io(|| write_all_at(file, buf, offset))?;
        self.direct_offset += n as u64;
        if self.config.sync_writes {
            blocking_io(|| sync_file(file))?;
        }
        Ok(())
    }

    /// Feed `data` to the O_DIRECT stream (see the type docs for the copy rules).
    fn stage_direct(&mut self, mut data: &[u8]) -> Result<()> {
        let alignment = self.config.alignment.max(1);
        let max_io = round_down(MAX_SINGLE_IO_BYTES, alignment).max(alignment);
        while !data.is_empty() {
            // Data is already staged (an earlier chunk ended off a block boundary): top the
            // staging buffer up to the next boundary, write it, and carry on from the caller's
            // memory - but only when the rest of this chunk starts page-aligned, which depends
            // on the caller's buffer (otherwise the chunk keeps going through the staging
            // buffer: one copy).
            if self.staged != 0 {
                let gap = self.staged.next_multiple_of(alignment) - self.staged;
                if data.len() >= gap + alignment
                    && (data.as_ptr() as usize + gap).is_multiple_of(alignment)
                {
                    if gap > 0 {
                        self.copy_into_staging(&data[..gap])?;
                    }
                    let staged = self.staged;
                    self.flush_staging(staged)?;
                    self.staged = 0;
                    data = &data[gap..];
                    continue;
                }
            }
            if self.staged == 0
                && data.len() >= alignment
                && (data.as_ptr() as usize).is_multiple_of(alignment)
            {
                // Straight from the caller's memory: no copy.
                let n = round_down(data.len(), alignment).min(max_io);
                #[cfg(test)]
                FROM_CALLER_WRITES.with(|c| c.set(c.get() + 1));
                self.write_direct(&data[..n])?;
                data = &data[n..];
                continue;
            }
            self.ensure_staging()?;
            let cap = self.staging_cap;
            let k = (cap - self.staged).min(data.len());
            self.copy_into_staging(&data[..k])?;
            data = &data[k..];
            if self.staged == cap {
                self.flush_staging(cap)?;
                self.staged = 0;
            }
        }
        Ok(())
    }

    /// Allocate the staging buffer on first use.
    fn ensure_staging(&mut self) -> Result<()> {
        if self.staging.is_some() {
            return Ok(());
        }
        let alignment = self.config.alignment.max(1);
        // Allocation failure is an error to the caller, never a panic; halve until it fits.
        let mut cap = self.staging_cap;
        let buf = loop {
            match AlignedBuf::new(cap, alignment) {
                Ok(b) => break b,
                Err(e) => {
                    cap = round_down(cap / 2, alignment);
                    if cap < alignment {
                        return Err(e.into());
                    }
                }
            }
        };
        self.staging_cap = buf.len();
        self.staging = Some(buf);
        Ok(())
    }

    /// Append `data` to the staging buffer (which must have room: `staged + len <= cap`).
    fn copy_into_staging(&mut self, data: &[u8]) -> Result<()> {
        self.ensure_staging()?;
        if self.staged + data.len() > self.staging_cap {
            bail!("internal error: staging buffer overflow");
        }
        if let Some(staging) = self.staging.as_mut() {
            // SAFETY: `staged + data.len() <= cap` stays inside the allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    staging.as_mut_ptr().add(self.staged),
                    data.len(),
                );
            }
        }
        self.staged += data.len();
        Ok(())
    }
}

#[async_trait]
impl ObjectWriter for DirectIOWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        if self.file.is_none() && self.direct.is_none() {
            bail!("Writer has been finalized or cancelled");
        }

        // Update checksum for all written data
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;

        // Compression: the encoder's output is the data to write; otherwise the caller's
        // chunk is used as is (no copy).
        let compressed: Vec<u8>;
        let data: &[u8] = if let Some(compressor) = self.compressor.as_mut() {
            compressor.write_all(chunk)?;
            compressed = std::mem::take(compressor.get_mut());
            &compressed
        } else {
            chunk
        };

        if self.direct.is_some() {
            return self.stage_direct(data);
        }
        let Some(file) = self.file.as_ref().map(Arc::clone) else {
            bail!("Writer has been finalized or cancelled");
        };
        // Straight from the caller's slice (Rule #1; see `write_borrowed_blocking`). Trade-off,
        // documented: on a CURRENT-THREAD runtime this write runs inline and blocks that thread
        // for its duration (a borrowed slice cannot be moved to another thread without a copy).
        crate::file_store::write_borrowed_blocking(&file, data)?;
        if self.config.sync_writes {
            tokio::task::spawn_blocking(move || file.sync_all()).await??;
        }
        Ok(())
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.finalized {
            return Ok(());
        }

        // Finalize compression if enabled
        if let Some(compressor) = self.compressor.take() {
            let compressed_data = compressor.finish()?;
            self.compressed_bytes = compressed_data.len() as u64;
            if !compressed_data.is_empty() {
                if self.direct.is_some() {
                    self.stage_direct(&compressed_data)?;
                } else if let Some(file) = self.file.as_ref().map(Arc::clone) {
                    // The compressed tail is an owned Vec: moved into the blocking task.
                    tokio::task::spawn_blocking(move || {
                        crate::file_store::write_all_std(&file, &compressed_data)
                    })
                    .await??;
                }
            }
        } else {
            self.compressed_bytes = self.bytes_written;
        }

        if self.direct.is_some() {
            // Hybrid: the aligned part of what is staged goes out with O_DIRECT, the
            // unaligned remainder (< one alignment unit) is appended with buffered I/O.
            let alignment = self.config.alignment.max(1);
            let aligned = round_down(self.staged, alignment);
            let remainder = self.staged - aligned;
            if aligned > 0 {
                self.flush_staging(aligned)?;
            }
            // The one durability point of a direct stream (unless sync_writes asked for more).
            if let Some(file) = self.direct.take() {
                blocking_io(|| sync_file(&file))?;
            }
            if remainder > 0 {
                let Some(staging) = self.staging.as_ref() else {
                    bail!("internal error: staged bytes without a staging buffer");
                };
                // SAFETY: `[aligned, staged)` was filled by `stage_direct`.
                let tail =
                    unsafe { std::slice::from_raw_parts(staging.as_ptr().add(aligned), remainder) };
                let path = self.path.clone();
                blocking_io(|| -> std::io::Result<()> {
                    use std::io::Write;
                    let mut buffered = std::fs::OpenOptions::new().append(true).open(&path)?;
                    buffered.write_all(tail)?;
                    buffered.flush()?;
                    sync_file(&buffered)
                })?;
            }
        } else if let Some(file) = self.file.take() {
            tokio::task::spawn_blocking(move || file.sync_all()).await??;
        }

        self.finalized = true;
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compression(&self) -> CompressionConfig {
        self.compression
    }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else {
            // Use the staged size if not finalized yet, otherwise use compressed_bytes
            let current_compressed_size = if self.finalized {
                self.compressed_bytes
            } else {
                self.staged as u64
            };
            current_compressed_size as f64 / self.bytes_written as f64
        }
    }

    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        self.staging = None; // Drop any staged data

        if let Some(compressor) = self.compressor.take() {
            drop(compressor); // Clean up compressor
        }

        // The files are closed when dropped.
        self.file = None;
        self.direct = None;

        // Remove the partial file
        if self.path.exists() {
            let _ = fs::remove_file(&self.path).await; // Ignore errors
        }

        Ok(())
    }
}

/// Enhanced FileSystem adapter with O_DIRECT support
pub struct ConfigurableFileSystemObjectStore {
    config: Arc<FileSystemConfig>,
}

impl Clone for ConfigurableFileSystemObjectStore {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
        }
    }
}

impl ConfigurableFileSystemObjectStore {
    pub fn new(config: FileSystemConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    pub fn with_config(config: FileSystemConfig) -> Self {
        Self::new(config)
    }

    pub fn with_direct_io() -> Self {
        Self::new(FileSystemConfig::direct_io())
    }

    pub fn high_performance() -> Self {
        Self::new(FileSystemConfig::high_performance())
    }

    /// Test if the filesystem at the given path supports O_DIRECT
    pub async fn test_direct_io_support(path: &Path) -> bool {
        #[cfg(unix)]
        {
            // Create a test file to check O_DIRECT support
            let test_path = path.join(".test_direct_io");
            let result = tokio::task::spawn_blocking(move || -> bool {
                use std::fs::OpenOptions;
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;

                // Try to create a small file with O_DIRECT
                match OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&test_path)
                {
                    Ok(mut file) => {
                        // Try to write aligned data
                        let page_size = get_system_page_size();
                        let test_data = vec![0u8; page_size];
                        let write_result = file.write_all(&test_data);

                        // Clean up test file
                        let _ = std::fs::remove_file(&test_path);

                        write_result.is_ok()
                    }
                    Err(_) => false,
                }
            })
            .await;

            result.unwrap_or(false)
        }

        #[cfg(not(unix))]
        {
            false // O_DIRECT not supported on non-Unix systems
        }
    }

    /// Convert a URI to a filesystem path (accepts both file:// and direct:// schemes)
    /// The O_DIRECT alignment, never 0 (a 0 in the config would divide by zero).
    fn alignment(&self) -> usize {
        self.config.alignment.max(1)
    }

    fn uri_to_path(uri: &str) -> Result<PathBuf> {
        let path = if let Some(stripped) = uri.strip_prefix("file://") {
            stripped // Strip "file://" prefix
        } else if let Some(stripped) = uri.strip_prefix("direct://") {
            stripped // Strip "direct://" prefix
        } else {
            bail!(
                "FileSystemObjectStore expects file:// or direct:// URI, got: {}",
                uri
            );
        };

        Ok(PathBuf::from(path))
    }

    /// Check if URI uses a valid file scheme (file:// or direct://)
    fn is_valid_file_uri(uri: &str) -> bool {
        uri.starts_with("file://") || uri.starts_with("direct://")
    }

    /// Extract the scheme prefix ("file" or "direct") a URI was addressed
    /// with, for round-tripping into URIs this store produces (e.g. from
    /// `list()`). Defaults to "file" for anything else, matching the
    /// pre-existing hardcoded behavior for callers that don't go through a
    /// recognized scheme.
    fn scheme_of(uri: &str) -> &'static str {
        if uri.starts_with("direct://") {
            "direct"
        } else {
            "file"
        }
    }

    /// Convert a filesystem path back to a URI for list operations.
    ///
    /// audit #156 bug 6.2 (E4): this previously hardcoded "file://"
    /// unconditionally, so `list()` on a `direct://` store returned
    /// `file://` URIs for every entry. Round-tripping one of those URIs
    /// back through `store_for_uri()` would route to the buffered
    /// `file://` store instead of the O_DIRECT store the caller was
    /// actually using, silently losing the direct-I/O behavior. `scheme`
    /// must be the scheme the caller originally addressed this store with
    /// (see `scheme_of`), not a hardcoded constant.
    fn path_to_uri(path: &Path, scheme: &str) -> String {
        if path.is_absolute() {
            format!("{scheme}://{}", path.display())
        } else {
            path.display().to_string()
        }
    }

    /// Create ObjectMetadata from filesystem metadata
    async fn metadata_from_path(path: &Path) -> Result<ObjectMetadata> {
        let metadata = fs::metadata(path).await?;

        // Create ObjectMetadata compatible with S3ObjectStat
        let last_modified = metadata
            .modified()
            .ok()
            .and_then(crate::file_store::modified_time_rfc3339);

        let file_hash = format!(
            "file-{}-{}",
            metadata.len(),
            metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        Ok(ObjectMetadata {
            size: metadata.len(),
            last_modified,
            e_tag: Some(file_hash),
            content_type: None,
            content_language: None,
            content_encoding: None,
            cache_control: None,
            content_disposition: None,
            expires: None,
            storage_class: Some("DIRECT_IO".to_string()),
            server_side_encryption: None,
            ssekms_key_id: None,
            sse_customer_algorithm: None,
            version_id: None,
            replication_status: None,
            metadata: HashMap::new(),
        })
    }

    /// Create OpenOptions with appropriate flags for the configuration
    #[cfg(not(unix))]
    fn create_open_options(&self, read: bool, write: bool) -> fs::OpenOptions {
        let mut options = fs::OpenOptions::new();
        options.read(read).write(write);

        if write {
            options.create(true).truncate(true);
        }

        #[cfg(unix)]
        {
            if self.config.direct_io {
                // O_DIRECT flag (Linux/Unix)
                options.custom_flags(libc::O_DIRECT);
            }
            if self.config.sync_writes && write {
                // O_SYNC flag for synchronous writes
                options.custom_flags(libc::O_SYNC);
            }
        }

        options
    }

    /// Read file with O_DIRECT if configured
    async fn read_file_direct(&self, path: &Path) -> Result<Bytes> {
        if !self.config.direct_io {
            // Fallback to normal tokio read
            let data = fs::read(path).await?;
            return Ok(Bytes::from(data));
        }

        // For O_DIRECT, we need to be careful about alignment and may need to fall back
        match self.try_read_file_direct(path).await {
            Ok(data) => Ok(data),
            Err(e) if crate::direct_bounce::is_cancelled(&e) => Err(e),
            Err(e) => {
                // If O_DIRECT fails, fall back to regular I/O
                GET_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                warn!(
                    "O_DIRECT read failed for {}, falling back to regular I/O: {}",
                    path.display(),
                    e
                );
                let data = fs::read(path).await?;
                Ok(Bytes::from(data))
            }
        }
    }

    /// Try to read file with O_DIRECT, may fail due to alignment or filesystem support
    ///
    /// The object is read straight into one page-aligned buffer (no bounce copy) that is
    /// returned as the `Bytes`, so the result is page-aligned.
    async fn try_read_file_direct(&self, path: &Path) -> Result<Bytes> {
        let metadata = fs::metadata(path).await?;
        let file_size = metadata.len() as usize;

        if file_size == 0 {
            return Ok(Bytes::new());
        }

        // For small files or when O_DIRECT would be inefficient, use regular I/O
        if file_size < self.config.min_io_size {
            return Ok(Bytes::from(fs::read(path).await?));
        }

        #[cfg(unix)]
        {
            let path = path.to_owned();
            let alignment = self.alignment();
            let cancel_guard = CancelOnDrop::new();
            let cancel = cancel_guard.flag();

            // Use spawn_blocking to run sync I/O in thread pool
            tokio::task::spawn_blocking(move || -> Result<Bytes> {
                use std::fs::OpenOptions;
                use std::os::unix::fs::OpenOptionsExt;

                let file = OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)?;

                // One read per at most 2 GiB into one aligned buffer, not one 4 KiB read
                // per system call (issue #244).
                let data = read_all_direct(&file, file_size, alignment, &cancel)?;
                DIRECT_GETS.fetch_add(1, Ordering::Relaxed);
                Ok(data)
            })
            .await?
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix systems
            Ok(Bytes::from(fs::read(path).await?))
        }
    }

    /// Write file with O_DIRECT if configured
    async fn write_file_direct(&self, path: &Path, data: Bytes) -> Result<()> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        if !self.config.direct_io {
            // Fallback to normal tokio write
            return Self::write_file_buffered(path, data).await;
        }

        // Try O_DIRECT write, fall back to regular write if it fails
        match self.try_write_file_direct(path, data.clone()).await {
            Ok(()) => Ok(()),
            Err(e) if crate::direct_bounce::is_cancelled(&e) => Err(e),
            Err(e) => {
                PUT_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                warn!(
                    "O_DIRECT write failed for {}, falling back to regular I/O: {}",
                    path.display(),
                    e
                );
                Self::write_file_buffered(path, data).await
            }
        }
    }

    /// Buffered write of `data` without duplicating it (`tokio::fs::write` would copy
    /// its input into an owned buffer, issue #238).
    async fn write_file_buffered(path: &Path, data: Bytes) -> Result<()> {
        let path = path.to_owned();
        tokio::task::spawn_blocking(move || std::fs::write(&path, data.as_ref())).await??;
        Ok(())
    }

    /// Try to write file with O_DIRECT, may fail due to alignment or filesystem support
    async fn try_write_file_direct(&self, path: &Path, data: Bytes) -> Result<()> {
        if data.is_empty() {
            return Self::write_file_buffered(path, data).await;
        }

        // For small files, use regular I/O
        if data.len() < self.config.min_io_size {
            return Self::write_file_buffered(path, data).await;
        }

        #[cfg(unix)]
        {
            let path = path.to_owned();
            let alignment = self.alignment();
            let sync_writes = self.config.sync_writes;
            let cancel_guard = CancelOnDrop::new();
            let cancel = cancel_guard.flag();

            // Use spawn_blocking for sync I/O
            tokio::task::spawn_blocking(move || -> Result<()> {
                use std::fs::OpenOptions;
                use std::os::unix::fs::OpenOptionsExt;

                let mut flags = libc::O_DIRECT;
                if sync_writes {
                    flags |= libc::O_SYNC;
                }

                let open = || {
                    OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .custom_flags(flags)
                        .open(&path)
                };

                // One write for the whole aligned body when `data` is aligned, otherwise
                // bounce-buffer-sized writes (issue #244); never one 4 KiB block per call. The
                // object is only created/truncated once a transfer can start.
                write_all_direct(&open, data.as_ref(), alignment, &cancel)?;
                DIRECT_PUTS.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .await?
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix systems
            Self::write_file_buffered(path, data).await
        }
    }

    /// Read range with O_DIRECT support
    async fn read_range_direct(
        &self,
        path: &Path,
        offset: u64,
        length: Option<u64>,
    ) -> Result<Bytes> {
        if !self.config.direct_io {
            // Fallback to normal implementation
            return read_range_buffered(path, offset, length).await;
        }

        // Try O_DIRECT first, fall back to regular I/O if it fails
        match self.try_read_range_direct(path, offset, length).await {
            Ok(data) => Ok(data), // Already Bytes now (v0.9.9+)
            Err(e) => {
                warn!(
                    "O_DIRECT range read failed for {}, falling back to regular I/O: {}",
                    path.display(),
                    e
                );
                // Fall back to regular I/O
                read_range_buffered(path, offset, length).await
            }
        }
    }

    /// Try to read file range with O_DIRECT, may fail due to alignment or filesystem support
    ///
    /// **v0.9.9 Enhancement**: Uses buffer pool to eliminate allocation churn
    /// - Borrows aligned buffer from pool (reused across operations)
    /// - Returns only requested subrange (minimal copy vs full buffer copy)
    /// - Returns buffer to pool for future reuse
    async fn try_read_range_direct(
        &self,
        path: &Path,
        offset: u64,
        length: Option<u64>,
    ) -> Result<Bytes> {
        let mut no_window = None;
        match self
            .try_read_range_direct_into(path, offset, length, &mut no_window)
            .await?
        {
            RangeRead::Bytes(bytes) => Ok(bytes),
            RangeRead::Filled(_) => bail!("internal error: a range read filled a window it had no"),
        }
    }

    /// [`try_read_range_direct`](Self::try_read_range_direct), optionally into a caller window.
    ///
    /// With `*window == Some(w)` the requested bytes are copied into the START of `w` (the one
    /// O_DIRECT bounce copy: the aligned read buffer, pooled or owned, to an unaligned
    /// destination) and the result is `Filled(n)`; `w` is put back into `*window` whatever the
    /// outcome, success or error, so the caller can retry. With `None` the result is `Bytes`.
    async fn try_read_range_direct_into(
        &self,
        path: &Path,
        offset: u64,
        length: Option<u64>,
        window: &mut Option<BytesMut>,
    ) -> Result<RangeRead> {
        // O_DIRECT range read requires careful alignment
        let alignment = self.alignment() as u64;
        let aligned_offset = (offset / alignment) * alignment;
        let offset_adjustment = (offset - aligned_offset) as usize;

        // audit #156 bug 6.1 (C6): previously `std::fs::metadata(path)
        // .unwrap()` panicked on any stat failure (most notably the
        // TOCTOU window where a caller's earlier `path.exists()` check
        // passed but the file was unlinked before this call runs), and
        // `metadata.len() - offset` underflowed (u64) whenever `offset`
        // exceeded the file's actual size. Both are now clean errors.
        let read_length = match length {
            Some(len) => len,
            None => {
                let metadata = fs::metadata(path).await.map_err(|e| {
                    anyhow::anyhow!(
                        "failed to stat {} for a to-end-of-file range read: {e}",
                        path.display()
                    )
                })?;
                metadata.len().checked_sub(offset).ok_or_else(|| {
                    anyhow::anyhow!(
                        "range offset {offset} exceeds file size {} for {}",
                        metadata.len(),
                        path.display()
                    )
                })?
            }
        };

        // Checked arithmetic: a huge offset/length must be an error, not a wrapped size.
        let too_big =
            || anyhow::anyhow!("range read of {read_length} bytes at offset {offset} is too large");
        let total_length = usize::try_from(read_length)
            .ok()
            .and_then(|l| offset_adjustment.checked_add(l))
            .ok_or_else(too_big)?;
        let aligned_length = total_length
            .checked_next_multiple_of(self.alignment())
            .ok_or_else(too_big)?;

        // Choose the path by the REQUIRED size, BEFORE taking anything from the pool (v0.9.9+).
        // A range that fits a pooled buffer borrows one: an RAII lease, which (with its permit)
        // goes back to the pool when dropped, on every exit including cancellation. A larger
        // range cannot use a pooled buffer at all: it takes only a private-allocation permit
        // (separate from the pooled permits, so oversized reads are bounded in number but neither
        // hold an unused 64 MiB pool buffer nor starve the reads that fit) and allocates its own
        // buffer. Without a pool, a fresh buffer is allocated as before.
        let mut lease: Option<crate::memory::PooledBuf> = None;
        let mut own: Option<AlignedBuf> = None;
        let mut private_permit: Option<tokio::sync::OwnedSemaphorePermit> = None;
        match &self.config.buffer_pool {
            Some(pool) if aligned_length <= pool.buffer_len() => {
                lease = Some(pool.take().await?);
            }
            Some(pool) => {
                trace!(
                    "Range of {} bytes exceeds the pool buffer size ({}): private allocation",
                    aligned_length,
                    pool.buffer_len()
                );
                private_permit = Some(pool.acquire_private().await?);
                own = Some(AlignedBuf::new(aligned_length, self.alignment())?);
            }
            None => {
                own = Some(AlignedBuf::new(aligned_length, self.alignment())?);
            }
        }
        // The read is done with `pread` straight into the buffer's raw memory by a blocking task
        // that OWNS the buffer (and so the pool permit): the kernel is the only writer, nothing
        // zero-fills a fresh buffer first, and a cancelled future cannot hand the buffer back to
        // the pool while the read is still running.
        #[cfg(unix)]
        {
            use crate::direct_bounce::SendPtr;
            use std::os::unix::fs::OpenOptionsExt;

            let path = path.to_owned();
            let direct = self.config.direct_io;
            let want_len = usize::try_from(read_length).map_err(|_| too_big())?;
            // The caller's window (if any) travels into the blocking task and comes back with
            // its result, so it survives an error and a retry can use it.
            let mut win = window.take();
            let (result, win) =
                tokio::task::spawn_blocking(move || -> (Result<RangeRead>, Option<BytesMut>) {
                    // The private-allocation permit outlives the buffer it admits: the buffer is
                    // dropped explicitly below, before this local (the permit) goes out of scope.
                    let _admission = private_permit;
                    #[cfg(test)]
                    range_read_hold::hold_if_armed(&path);
                    let result = (|| -> Result<RangeRead> {
                        let mut options = std::fs::OpenOptions::new();
                        options.read(true);
                        if direct {
                            options.custom_flags(libc::O_DIRECT);
                        }
                        let file = options.open(&path)?;
                        let ptr = match (own.as_mut(), lease.as_deref_mut()) {
                            (Some(o), _) => o.as_mut_ptr(),
                            (None, Some(l)) => l.as_mut_ptr(),
                            (None, None) => bail!("internal error: no range-read buffer"),
                        };
                        // Loop until the aligned range is filled or EOF: one read() may return far less
                        // than requested (O_DIRECT imposes per-call limits).
                        let mut bytes_read = 0usize;
                        while bytes_read < aligned_length {
                            // SAFETY: `[bytes_read, aligned_length)` lies inside the buffer.
                            let dst = SendPtr(unsafe { ptr.add(bytes_read) });
                            let n = pread_raw(
                                &file,
                                dst,
                                aligned_length - bytes_read,
                                aligned_offset + bytes_read as u64,
                            )?;
                            if n == 0 {
                                break; // EOF
                            }
                            bytes_read += n;
                        }
                        // The requested subrange: only bytes the kernel wrote ([0, bytes_read)).
                        let start = offset_adjustment.min(bytes_read);
                        let end = start.saturating_add(want_len).min(bytes_read);
                        if let Some(w) = win.as_mut() {
                            // The caller's window (the range engine's segment of its master
                            // buffer): the O_DIRECT bounce copy, made ONCE, straight from the
                            // aligned read buffer (pooled or owned) into the window. An unaligned
                            // destination cannot be read into with O_DIRECT, so this copy cannot
                            // be avoided. (No intermediate buffer: it used to be pool -> fresh
                            // BytesMut -> window.)
                            let n = end - start;
                            if n > w.len() {
                                bail!(
                                    "range read of {n} bytes does not fit a window of {}",
                                    w.len()
                                );
                            }
                            // SAFETY: `[start, end)` is inside `[0, bytes_read)`, which pread
                            // initialised; the window is a separate allocation, `n <= w.len()`.
                            w[..n].copy_from_slice(unsafe {
                                std::slice::from_raw_parts(ptr.add(start), n)
                            });
                            return Ok(RangeRead::Filled(n));
                        }
                        if let Some(buf) = own.take() {
                            // Rule #1 (zero copy): an owned (non-pooled) aligned buffer is not going
                            // back to any pool, so the result IS a slice of it (`[start, end)` was
                            // written by pread); it is freed when the last `Bytes` view drops. The
                            // private-allocation permit is released on return as for any GET result
                            // (the caller owns the memory from here on).
                            return Ok(RangeRead::Bytes(Bytes::from_owner(AlignedBytes {
                                buf,
                                start,
                                len: end - start,
                            })));
                        }
                        // Pooled buffer: it must go back to the pool, so the subrange is copied out
                        // (the one unavoidable copy of the bounded-pool design: the pool buffer is
                        // reused for the next read, so handing it out would drain the pool).
                        let mut out = BytesMut::with_capacity(end - start);
                        // SAFETY: `[start, end)` is inside `[0, bytes_read)`, which pread initialised.
                        out.extend_from_slice(unsafe {
                            std::slice::from_raw_parts(ptr.add(start), end - start)
                        });
                        Ok(RangeRead::Bytes(out.freeze()))
                    })();
                    drop(own);
                    drop(lease);
                    (result, win)
                })
                .await?;
            *window = win;
            result
        }

        #[cfg(not(unix))]
        {
            let aligned: &mut AlignedBuf = match (own.as_mut(), lease.as_deref_mut()) {
                (Some(o), _) => o,
                (None, Some(l)) => l,
                (None, None) => bail!("internal error: no range-read buffer"),
            };
            let options = self.create_open_options(true, false);
            let mut file = options.open(path).await?;
            file.seek(std::io::SeekFrom::Start(aligned_offset)).await?;
            let mut bytes_read = 0;
            while bytes_read < aligned_length {
                let n = file
                    .read(&mut aligned.as_mut_slice()[bytes_read..aligned_length])
                    .await?;
                if n == 0 {
                    break;
                }
                bytes_read += n;
            }
            let start = offset_adjustment.min(bytes_read);
            let end = start.saturating_add(read_length as usize).min(bytes_read);
            if let Some(buf) = own.take() {
                // Rule #1: an owned (non-pooled) buffer is handed out as a slice, not copied.
                return Ok(RangeRead::Bytes(Bytes::from_owner(AlignedBytes {
                    buf,
                    start,
                    len: end - start,
                })));
            }
            // Pooled buffer: returns to the pool, so the subrange is copied out (see the unix
            // path above).
            let Some(aligned) = lease.as_deref() else {
                bail!("internal error: no range-read buffer");
            };
            let mut out = BytesMut::with_capacity(end - start);
            out.extend_from_slice(&aligned.as_slice()[start..end]);
            Ok(RangeRead::Bytes(out.freeze()))
        }
    }

    /// The `O_DIRECT` range-engine download (unix): O_DIRECT needs an aligned destination, and
    /// the engine's master buffer is not aligned, so each range is read into an aligned buffer
    /// (pooled or owned) and copied ONCE into its window of the master buffer: the O_DIRECT
    /// bounce copy for unaligned memory, unavoidable and made exactly once (no intermediate
    /// `Bytes`). A failed O_DIRECT read falls back to buffered I/O straight into the window, as
    /// `get_range` falls back.
    #[cfg(unix)]
    async fn o_direct_range_download(
        &self,
        engine: &RangeEngine,
        uri: &str,
        object_size: u64,
    ) -> Result<(Bytes, crate::range_engine_generic::RangeDownloadStats)> {
        let store = self.clone();
        let path = Self::uri_to_path(uri)?;
        let fill_range = move |offset: u64, window: BytesMut| {
            let store = store.clone();
            let path = path.clone();
            async move {
                let want = window.len() as u64;
                let mut slot = Some(window);
                let attempt = store
                    .try_read_range_direct_into(&path, offset, Some(want), &mut slot)
                    .await;
                match attempt {
                    Ok(RangeRead::Filled(n)) => match slot.take() {
                        Some(window) => Ok((window, n)),
                        None => bail!("internal error: the range window was lost"),
                    },
                    Ok(RangeRead::Bytes(_)) => {
                        bail!("internal error: a window read returned bytes")
                    }
                    Err(e) => {
                        warn!(
                            "O_DIRECT range read failed for {}, falling back to regular I/O: {}",
                            path.display(),
                            e
                        );
                        let Some(mut window) = slot.take() else {
                            return Err(e);
                        };
                        tokio::task::spawn_blocking(move || -> Result<(BytesMut, usize)> {
                            let file = std::fs::File::open(&path)?;
                            let read =
                                crate::file_store::read_full_at_std(&file, &mut window, offset)?;
                            Ok((window, read))
                        })
                        .await?
                    }
                }
            }
        };
        engine
            .download_into_master(object_size, false, fill_range, None)
            .await
    }

    /// Non-unix: each range through `get_range`; the engine copies it into its window.
    #[cfg(not(unix))]
    async fn o_direct_range_download(
        &self,
        engine: &RangeEngine,
        uri: &str,
        object_size: u64,
    ) -> Result<(Bytes, crate::range_engine_generic::RangeDownloadStats)> {
        let store = self.clone();
        let uri_owned = uri.to_string();
        let get_range_fn = move |offset: u64, length: u64| {
            let store = store.clone();
            let uri = uri_owned.clone();
            async move { store.get_range(&uri, offset, Some(length)).await }
        };
        engine.download(object_size, get_range_fn, None).await
    }

    /// Get file using RangeEngine for concurrent downloads
    async fn get_with_range_engine(&self, uri: &str, object_size: u64) -> Result<Bytes> {
        debug!(
            "DirectIO using RangeEngine: uri={}, size={} MB, threshold={} MB",
            uri,
            object_size / (1024 * 1024),
            self.config.range_engine.min_split_size / (1024 * 1024)
        );

        let engine = RangeEngine::new(self.config.range_engine.clone());

        let (bytes, stats) = if self.config.direct_io {
            self.o_direct_range_download(&engine, uri, object_size)
                .await?
        } else {
            // Buffered mode (Rule #1): no alignment requirement, so every range is read with a
            // positional read straight into its segment of the master buffer, with no
            // per-range allocation and no copy.
            let path = Self::uri_to_path(uri)?;
            let fill_range = move |offset: u64, mut segment: BytesMut| {
                let path = path.clone();
                async move {
                    tokio::task::spawn_blocking(move || -> Result<(BytesMut, usize)> {
                        let file = std::fs::File::open(&path)?;
                        let read =
                            crate::file_store::read_full_at_std(&file, &mut segment, offset)?;
                        Ok((segment, read))
                    })
                    .await?
                }
            };
            engine
                .download_into_master(object_size, false, fill_range, None)
                .await?
        };

        info!(
            "DirectIO RangeEngine complete: {} MB in {:.2}s ({:.2} MB/s, {} ranges)",
            bytes.len() / (1024 * 1024),
            stats.elapsed_time.as_secs_f64(),
            stats.throughput_mbps(),
            stats.ranges_processed
        );

        Ok(bytes)
    }

    /// Simple file read without RangeEngine (for small files)
    async fn get_simple(&self, uri: &str, _size: u64) -> Result<Bytes> {
        let path = Self::uri_to_path(uri)?;
        trace!("DirectIO using simple read for file: {}", path.display());
        self.read_file_direct(&path).await
    }

    /// Recursively collect files in a directory. `scheme` is the scheme
    /// ("file" or "direct") the caller originally addressed this store
    /// with (audit #156 bug 6.2 / E4) -- see `path_to_uri`.
    async fn collect_files_recursive(
        dir: &Path,
        prefix: &str,
        scheme: &str,
        results: &mut Vec<String>,
    ) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let file_name = entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if entry_path.is_dir() {
                let new_prefix = if prefix.is_empty() {
                    file_name.to_string()
                } else {
                    format!("{}/{}", prefix, file_name)
                };
                Box::pin(Self::collect_files_recursive(
                    &entry_path,
                    &new_prefix,
                    scheme,
                    results,
                ))
                .await?;
            } else {
                let file_uri = Self::path_to_uri(&entry_path, scheme);
                results.push(file_uri);
            }
        }

        Ok(())
    }
}

#[async_trait]
impl ObjectStore for ConfigurableFileSystemObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("ConfigurableFileSystemObjectStore::get uri='{}'", uri);

        if !path.exists() {
            bail!("File not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        // Get file size to determine if we should use RangeEngine
        let metadata = fs::metadata(&path).await?;
        let size = metadata.len();

        // Use RangeEngine for large files if enabled
        if self.config.enable_range_engine && size >= self.config.range_engine.min_split_size {
            return self.get_with_range_engine(uri, size).await;
        }

        // Use simple read for small files or when RangeEngine disabled
        self.get_simple(uri, size).await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!(
            "ConfigurableFileSystemObjectStore::get_range uri='{}', offset={}, length={:?}",
            uri, offset, length
        );

        if !path.exists() {
            bail!("File not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        self.read_range_direct(&path, offset, length).await
    }

    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!(
            "ConfigurableFileSystemObjectStore::put uri='{}', {} bytes",
            uri,
            data.len()
        );

        // Bytes→&[u8] via .as_ref() is zero-copy (just returns pointer to Arc'd buffer)
        self.write_file_direct(&path, data).await
    }

    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        debug!(
            "ConfigurableFileSystemObjectStore::put_multipart uri='{}', {} bytes, part_size={:?}",
            uri,
            data.len(),
            part_size
        );

        // A local file has no parts: "multipart" is the same single write as `put`, with
        // the same large aligned O_DIRECT writes and buffered fallback. (It used to write
        // `part_size` chunks - 4 KiB by default - each zero-padded to the alignment, which
        // also put padding INSIDE the file when `part_size` was not an alignment multiple.)
        self.put(uri, data).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        if !Self::is_valid_file_uri(uri_prefix) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let base_path = Self::uri_to_path(uri_prefix)?;
        let scheme = Self::scheme_of(uri_prefix);
        debug!(
            "ConfigurableFileSystemObjectStore::list prefix='{}', recursive={}",
            uri_prefix, recursive
        );
        let mut results = Vec::new();

        if !base_path.exists() {
            return Ok(results); // Empty list for non-existent paths
        }

        if base_path.is_file() {
            // If the prefix points to a file, return just that file
            results.push(Self::path_to_uri(&base_path, scheme));
            return Ok(results);
        }

        if base_path.is_dir() {
            if recursive {
                Self::collect_files_recursive(&base_path, "", scheme, &mut results).await?;
            } else {
                // Non-recursive: only direct children
                let mut entries = fs::read_dir(&base_path).await?;
                while let Some(entry) = entries.next_entry().await? {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        results.push(Self::path_to_uri(&entry_path, scheme));
                    }
                }
            }
        }

        Ok(results)
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            // For file://, delegate to buffered list() since filesystem operations are fast
            // Streaming doesn't provide significant benefit for local filesystem
            match self.list(uri_prefix, recursive).await {
                Ok(keys) => {
                    for key in keys {
                        yield Ok(key);
                    }
                }
                Err(e) => yield Err(e),
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("ConfigurableFileSystemObjectStore::stat uri='{}'", uri);

        if !path.exists() {
            bail!("File not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        Self::metadata_from_path(&path).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("ConfigurableFileSystemObjectStore::delete uri='{}'", uri);

        if !path.exists() {
            // Already deleted, consider it success
            return Ok(());
        }

        if path.is_file() {
            fs::remove_file(&path).await?;
        } else if path.is_dir() {
            fs::remove_dir_all(&path).await?;
        }

        Ok(())
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        // FileSystemObjectStore: delete files concurrently
        use futures::stream::{self, StreamExt};
        debug!(
            "ConfigurableFileSystemObjectStore::delete_batch {} URIs",
            uris.len()
        );

        let max_concurrency = (uris.len() / 10).clamp(10, 100);
        let uris_owned: Vec<String> = uris.to_vec();

        let deletions = stream::iter(uris_owned)
            .map(|uri| async move { self.delete(&uri).await })
            .buffer_unordered(max_concurrency);

        let results: Vec<Result<()>> = deletions.collect().await;
        for result in results {
            result?;
        }

        Ok(())
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        if !Self::is_valid_file_uri(uri_prefix) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        let base_path = Self::uri_to_path(uri_prefix)?;
        debug!(
            "ConfigurableFileSystemObjectStore::delete_prefix prefix='{}'",
            uri_prefix
        );

        if !base_path.exists() {
            return Ok(()); // Nothing to delete
        }

        if base_path.is_file() {
            fs::remove_file(&base_path).await?;
        } else if base_path.is_dir() {
            // Collect all files under the prefix and delete them
            let files = self.list(uri_prefix, true).await?;
            for file_uri in files {
                self.delete(&file_uri).await?;
            }

            // Try to remove the directory if it's empty
            if (fs::remove_dir(&base_path).await).is_err() {
                // Directory might not be empty due to subdirectories
                // For now, we'll leave non-empty directories
            }
        }

        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);
        fs::create_dir_all(&path).await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);

        if !path.exists() {
            return Ok(()); // Already deleted
        }

        if path.is_dir() {
            fs::remove_dir(&path).await?; // Only removes empty directories
        } else {
            bail!("Path is not a directory: {}", path.display());
        }

        Ok(())
    }

    /// `mkdir` is the `file://` behaviour (`create_dir_all`): O_DIRECT is irrelevant
    /// to a directory operation. Not recorded in the op-log (#229).
    async fn mkdir(&self, uri: &str) -> Result<()> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        crate::file_store::mkdir_path(&Self::uri_to_path(uri)?).await
    }

    /// `rmdir` is the `file://` behaviour: `remove_dir` (empty directories only) or
    /// `remove_dir_all`; a missing directory is success (#229).
    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }
        crate::file_store::rmdir_path(&Self::uri_to_path(uri)?, recursive).await
    }

    /// `rename` is the filesystem's atomic `rename(2)` (below), not copy + delete.
    fn has_native_rename(&self) -> bool {
        true
    }

    async fn rename(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        if !Self::is_valid_file_uri(src_uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI for source");
        }
        if !Self::is_valid_file_uri(dst_uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI for destination");
        }

        let src_path = Self::uri_to_path(src_uri)?;
        let dst_path = Self::uri_to_path(dst_uri)?;

        if !src_path.exists() {
            bail!("Source file not found: {}", src_path.display());
        }

        // Create parent directories for destination if they don't exist
        if let Some(parent) = dst_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Use tokio::fs::rename for atomic filesystem rename operation
        fs::rename(&src_path, &dst_path).await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }

        let path = Self::uri_to_path(uri)?;

        // For direct I/O, we might want a specialized writer, but for now use the same
        // as regular filesystem with potential for future direct I/O optimization
        Ok(Box::new(
            DirectIOWriter::new(path, (*self.config).clone()).await?,
        ))
    }

    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }

        let path = Self::uri_to_path(uri)?;

        // For direct I/O with compression
        Ok(Box::new(
            DirectIOWriter::new_with_compression(path, (*self.config).clone(), compression).await?,
        ))
    }

    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        if !Self::is_valid_file_uri(uri) {
            bail!("FileSystemObjectStore expected file:// or direct:// URI");
        }

        let path = Self::uri_to_path(uri)?;

        // Create DirectIOWriter with optional compression
        if let Some(compression) = options.compression {
            Ok(Box::new(
                DirectIOWriter::new_with_compression(path, (*self.config).clone(), compression)
                    .await?,
            ))
        } else {
            Ok(Box::new(
                DirectIOWriter::new(path, (*self.config).clone()).await?,
            ))
        }
    }
}

impl ConfigurableFileSystemObjectStore {
    #[inline]
    pub fn boxed(config: FileSystemConfig) -> Box<dyn ObjectStore> {
        Box::new(Self::new(config))
    }

    #[inline]
    pub fn boxed_direct_io() -> Box<dyn ObjectStore> {
        Box::new(Self::with_direct_io())
    }

    #[inline]
    pub fn boxed_high_performance() -> Box<dyn ObjectStore> {
        Box::new(Self::high_performance())
    }
}

/// Test hook for the range-read admission tests: reads of files under an armed prefix park inside
/// the blocking read (after admission, before any I/O) until released.
#[cfg(test)]
pub(crate) mod range_read_hold {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Condvar, Mutex, PoisonError};
    use std::time::{Duration, Instant};

    pub(crate) struct Hold {
        prefix: PathBuf,
        state: Mutex<(usize, bool)>, // (reads parked so far, released)
        cv: Condvar,
    }

    /// Arms a hold for its lifetime; dropping it releases every parked read and disarms.
    pub(crate) struct HoldGuard(pub(crate) Arc<Hold>);

    impl std::ops::Deref for HoldGuard {
        type Target = Hold;
        fn deref(&self) -> &Hold {
            &self.0
        }
    }

    impl Drop for HoldGuard {
        fn drop(&mut self) {
            self.0.release();
            ARMED
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|h| !Arc::ptr_eq(h, &self.0));
        }
    }

    static ARMED: Mutex<Vec<Arc<Hold>>> = Mutex::new(Vec::new());

    impl Hold {
        pub(crate) fn arm(prefix: &Path) -> HoldGuard {
            let hold = Arc::new(Hold {
                prefix: prefix.to_owned(),
                state: Mutex::new((0, false)),
                cv: Condvar::new(),
            });
            ARMED
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(hold.clone());
            HoldGuard(hold)
        }

        /// Wait until `n` reads are parked; false on timeout.
        pub(crate) fn wait_parked(&self, n: usize, timeout: Duration) -> bool {
            let deadline = Instant::now() + timeout;
            let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            while st.0 < n {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                st = self
                    .cv
                    .wait_timeout(st, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            true
        }

        pub(crate) fn parked(&self) -> usize {
            self.state.lock().unwrap_or_else(PoisonError::into_inner).0
        }

        pub(crate) fn release(&self) {
            self.state.lock().unwrap_or_else(PoisonError::into_inner).1 = true;
            self.cv.notify_all();
        }
    }

    pub(crate) fn hold_if_armed(path: &Path) {
        let hold = ARMED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|h| path.starts_with(&h.prefix))
            .cloned();
        let Some(hold) = hold else { return };
        let mut st = hold.state.lock().unwrap_or_else(PoisonError::into_inner);
        st.0 += 1;
        hold.cv.notify_all();
        while !st.1 {
            st = hold.cv.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

#[cfg(test)]
mod try_read_range_direct_tests {
    use super::*;

    // RED-then-GREEN regression tests for s3dlio issue #156 bug 6.1 (C6).
    //
    // Bug: `try_read_range_direct`'s `length=None` branch computed the
    // read length as `std::fs::metadata(path).unwrap().len() - offset`.
    // Two independent panics:
    //   (a) `.unwrap()` on the metadata() call — panics if the file
    //       can't be stat'd, most notably the TOCTOU window where a
    //       caller's earlier `path.exists()` check passed but the file
    //       was unlinked before this call runs.
    //   (b) even when metadata() succeeds, `metadata.len() - offset`
    //       underflows (u64) whenever the caller passes an `offset`
    //       larger than the file's actual size.
    //
    // These are white-box unit tests (calling the private method
    // directly) because reliably forcing the TOCTOU race through the
    // public `get_range()` API — whose own `path.exists()` guard runs
    // first — isn't deterministic; see the adequacy note in
    // docs/implementation-plans/v0.9.109-audit-fix-plan.md §11 (C6/6.1b:
    // "TOCTOU race, hard to reproduce reliably"). Calling the vulnerable
    // function directly with a path that's already missing reproduces
    // the same failure mode deterministically. Neither panic requires
    // O_DIRECT to actually be supported by the underlying filesystem —
    // both occur in plain Rust code before any O_DIRECT syscall is
    // attempted, so a tmpfs-backed TempDir is sufficient.

    #[tokio::test]
    async fn missing_file_is_a_clean_error_not_a_panic() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let missing = temp_dir.path().join("does-not-exist.dat");
        let store = ConfigurableFileSystemObjectStore::with_direct_io();

        let result = store.try_read_range_direct(&missing, 0, None).await;
        assert!(
            result.is_err(),
            "a missing file (TOCTOU-equivalent: unlinked before this call) must be a clean \
             Err, not a panic from metadata().unwrap()"
        );
    }

    #[tokio::test]
    async fn offset_past_eof_is_a_clean_error_not_an_underflow_panic() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let path = temp_dir.path().join("small.dat");
        tokio::fs::write(&path, vec![0u8; 100]).await.unwrap();
        let store = ConfigurableFileSystemObjectStore::with_direct_io();

        let result = store.try_read_range_direct(&path, 1_000_000, None).await;
        assert!(
            result.is_err(),
            "offset (1_000_000) exceeding the file's actual size (100 bytes) must be a clean \
             Err, not a `metadata.len() - offset` u64 underflow panic"
        );
    }

    #[tokio::test]
    async fn offset_within_file_size_still_works() {
        // Sanity check: the fix must not break the normal case. Uses the
        // public get_range() API (which has its own fallback-to-regular-
        // I/O path when O_DIRECT isn't usable on the backing filesystem,
        // e.g. tmpfs) rather than calling the private O_DIRECT-only
        // try_read_range_direct() directly, so this test is meaningful
        // regardless of the test environment's O_DIRECT support.
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store = ConfigurableFileSystemObjectStore::with_direct_io();
        let uri = format!("file://{}/normal.dat", temp_dir.path().to_str().unwrap());
        let data = vec![0xABu8; 4096];
        store.put(&uri, Bytes::from(data)).await.unwrap();

        let bytes = store
            .get_range(&uri, 100, None)
            .await
            .expect("a valid in-range offset must still succeed");
        assert_eq!(bytes.len(), 4096 - 100);
    }
}

/// R5: an oversized range read must decide its path by the required size BEFORE taking a pool
/// buffer. It used to take a pooled buffer (and permit) first, find it too small, allocate its own
/// and keep the unused lease for the whole read - enough oversized reads then held every permit
/// and starved the reads that fit.
#[cfg(test)]
mod oversized_range_admission_tests {
    use super::range_read_hold::Hold;
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    const BUF: usize = 64 * 1024;
    const CAP: usize = 2;

    fn scratch() -> tempfile::TempDir {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::TempDir::new_in(base).unwrap()
    }

    fn pattern(len: usize, seed: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 31 + seed) % 251) as u8).collect()
    }

    fn store() -> (
        Arc<ConfigurableFileSystemObjectStore>,
        Arc<crate::memory::BufferPool>,
    ) {
        let pool = BufferPoolConfig {
            capacity: CAP,
            buffer_size: BUF,
            alignment: 4096,
        }
        .build()
        .unwrap();
        let config = FileSystemConfig {
            buffer_pool: Some(pool.clone()),
            ..FileSystemConfig::default() // buffered pread: no O_DIRECT needed for admission
        };
        (
            Arc::new(ConfigurableFileSystemObjectStore::new(config)),
            pool,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn held_oversized_reads_take_no_pool_buffer_and_do_not_starve_small_reads() {
        let dir = scratch();
        let big = dir.path().join("big.dat");
        let small = dir.path().join("small.dat");
        let big_data = pattern(4 * BUF, 1);
        let small_data = pattern(8192, 2);
        std::fs::write(&big, &big_data).unwrap();
        std::fs::write(&small, &small_data).unwrap();
        let (store, pool) = store();
        let hold = Hold::arm(&big);

        // More oversized reads than there are permits: all are held inside the blocking read.
        let want = 3 * BUF + 100; // needs 4 pool buffers' worth: cannot use a pooled buffer
        let mut oversized = Vec::new();
        for _ in 0..(2 * CAP) {
            let (store, big) = (store.clone(), big.clone());
            oversized.push(tokio::spawn(async move {
                store
                    .try_read_range_direct(&big, 0, Some(want as u64))
                    .await
            }));
        }
        let h = hold.0.clone();
        assert!(
            tokio::task::spawn_blocking(move || h.wait_parked(CAP, Duration::from_secs(20)))
                .await
                .unwrap(),
            "the oversized reads never got admitted"
        );
        assert_eq!(
            pool.capacity(),
            CAP,
            "oversized reads hold pooled permits/buffers they do not use"
        );
        assert_eq!(pool.private_available(), 0, "private admission is bounded");
        assert_eq!(
            hold.parked(),
            CAP,
            "more oversized reads than the admission bound"
        );

        // A read that fits is not starved by the held oversized reads.
        let small_read = tokio::time::timeout(
            Duration::from_secs(10),
            store.try_read_range_direct(&small, 0, Some(4096)),
        )
        .await
        .expect("a small read was starved by held oversized reads")
        .unwrap();
        assert_eq!(small_read.as_ref(), &small_data[..4096]);

        hold.release();
        for task in oversized {
            let got = task.await.unwrap().unwrap();
            assert_eq!(got.as_ref(), &big_data[..want]);
        }
        assert_eq!(pool.capacity(), CAP, "pooled permits not all returned");
        assert_eq!(
            pool.private_available(),
            CAP,
            "private permits not all returned"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_read_that_fits_still_borrows_a_pooled_buffer() {
        let dir = scratch();
        let file = dir.path().join("fits.dat");
        let data = pattern(BUF, 3);
        std::fs::write(&file, &data).unwrap();
        let (store, pool) = store();
        let hold = Hold::arm(&file);
        let read = {
            let (store, file) = (store.clone(), file.clone());
            tokio::spawn(async move {
                store
                    .try_read_range_direct(&file, 0, Some(BUF as u64))
                    .await
            })
        };
        let h = hold.0.clone();
        assert!(
            tokio::task::spawn_blocking(move || h.wait_parked(1, Duration::from_secs(20)))
                .await
                .unwrap()
        );
        assert_eq!(
            pool.capacity(),
            CAP - 1,
            "a fitting read must lease a pooled buffer"
        );
        assert_eq!(
            pool.private_available(),
            CAP,
            "and need no private admission"
        );
        hold.release();
        assert_eq!(read.await.unwrap().unwrap().as_ref(), &data[..]);
        assert_eq!(pool.capacity(), CAP);
    }
}

/// R6: when the shared bounce pool cannot serve a PUT (it could not be created, or it cannot give
/// the requested alignment) the PUT still succeeds - availability first - with a private per-call
/// bounce buffer. That memory is outside the pooled cap, so it must be COUNTED
/// (`BounceStats::implicit_fallbacks` / `implicit_fallback_bytes`, with the cause), not silent.
/// One test: the counters are process-wide, so the scenarios run one after the other.
#[cfg(all(test, unix))]
mod implicit_bounce_accounting_tests {
    use super::*;
    use crate::direct_bounce::{bounce_stats, test_seam::NoPool, PutStrategy};
    use std::sync::atomic::AtomicBool;

    fn scratch() -> tempfile::TempDir {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::TempDir::new_in(base).unwrap()
    }

    fn put(path: &Path, data: &[u8], alignment: usize) {
        let open = || {
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
        };
        let cancel = AtomicBool::new(false);
        write_all_direct_with(
            &open,
            data,
            alignment,
            PutStrategy::Pooled,
            &AlignedBuf::new,
            &cancel,
        )
        .expect("the PUT must succeed without the pool (availability first)");
        assert_eq!(std::fs::read(path).unwrap(), data, "content");
    }

    #[test]
    fn implicit_private_bounce_allocations_are_counted_with_their_cause() {
        let cap = crate::direct_bounce::config().buffer_size;
        let dir = scratch();
        let len = 3 * 1024 * 1024 + 5;
        assert!(
            round_up(len, 8192) <= cap,
            "needs a bounce size of at least 4 MiB"
        );
        // Never page-aligned: the source must be bounced.
        let backing: Vec<u8> = (0..len + 1).map(|i| (i * 7 + 3) as u8).collect();
        let data = &backing[1..];

        // 1. A normal PUT is served by the pool: nothing implicit.
        let before = bounce_stats();
        put(&dir.path().join("pooled.bin"), data, 4096);
        let after = bounce_stats();
        assert_eq!(
            after.implicit_fallbacks, before.implicit_fallbacks,
            "pooled PUT counted"
        );
        assert_eq!(
            after.implicit_fallback_bytes,
            before.implicit_fallback_bytes
        );
        assert!(after.leases[0] > before.leases[0], "the pool was not used");

        // 2. The pool cannot be created (injected): one counted private buffer, cause recorded,
        //    (pool lease counters are process-wide and other tests lease concurrently, so the cause
        //    counters, not leases, prove the pool was not used).
        let before = bounce_stats();
        {
            let _no_pool = NoPool::inject();
            put(&dir.path().join("nopool.bin"), data, 4096);
        }
        let after = bounce_stats();
        assert_eq!(
            after.implicit_fallbacks[0],
            before.implicit_fallbacks[0] + 1
        );
        assert_eq!(after.implicit_fallbacks[1], before.implicit_fallbacks[1]);
        assert_eq!(
            after.implicit_fallback_bytes,
            before.implicit_fallback_bytes + round_up(len, 4096) as u64,
            "the bytes of the private buffer"
        );
        assert_eq!(after.pool_unavailable, before.pool_unavailable + 1);
        assert_eq!(after.alignment_mismatches, before.alignment_mismatches);
        assert_eq!(
            after.fallback_live_bytes, 0,
            "an implicit buffer is not a busy fallback"
        );

        // 3. The pool's buffers are page-aligned; a stricter alignment cannot be served by them.
        let before = bounce_stats();
        put(&dir.path().join("align.bin"), data, 8192);
        let after = bounce_stats();
        assert_eq!(
            after.implicit_fallbacks[0],
            before.implicit_fallbacks[0] + 1
        );
        assert_eq!(
            after.implicit_fallback_bytes,
            before.implicit_fallback_bytes + round_up(len, 8192) as u64
        );
        assert_eq!(after.alignment_mismatches, before.alignment_mismatches + 1);
        assert_eq!(after.pool_unavailable, before.pool_unavailable);

        // 4. A page-aligned source writes straight from memory and bounces only its one-page tail:
        //    that is by design (never pooled), not an implicit fallback.
        let mut aligned = AlignedBuf::new(8192 + 100, 4096).unwrap();
        let aligned_len = 8192 + 100;
        aligned.as_mut_slice()[..aligned_len].fill(0x5A);
        let before = bounce_stats();
        put(
            &dir.path().join("aligned.bin"),
            &aligned.as_slice()[..aligned_len],
            4096,
        );
        let after = bounce_stats();
        assert_eq!(
            after.implicit_fallbacks, before.implicit_fallbacks,
            "aligned tail counted"
        );
    }
}

#[cfg(test)]
mod path_to_uri_scheme_tests {
    use super::*;

    // RED-then-GREEN regression tests for s3dlio issue #156 bug 6.2 (E4).
    //
    // Bug: `path_to_uri` hardcoded the "file://" scheme unconditionally,
    // so `list()` on a `direct://`-addressed store returned `file://`
    // URIs for every entry. Round-tripping one of those URIs back through
    // `store_for_uri()` would silently route to the buffered `file://`
    // store instead of the O_DIRECT store the caller was actually using.

    #[test]
    fn scheme_of_direct_uri_is_direct() {
        assert_eq!(
            ConfigurableFileSystemObjectStore::scheme_of("direct:///tmp/x/"),
            "direct"
        );
    }

    #[test]
    fn scheme_of_file_uri_is_file() {
        assert_eq!(
            ConfigurableFileSystemObjectStore::scheme_of("file:///tmp/x/"),
            "file"
        );
    }

    #[test]
    fn path_to_uri_uses_the_given_scheme_not_a_hardcoded_one() {
        let path = Path::new("/tmp/x/data.bin");
        assert_eq!(
            ConfigurableFileSystemObjectStore::path_to_uri(path, "direct"),
            "direct:///tmp/x/data.bin"
        );
        assert_eq!(
            ConfigurableFileSystemObjectStore::path_to_uri(path, "file"),
            "file:///tmp/x/data.bin"
        );
    }

    #[tokio::test]
    async fn list_on_a_direct_scheme_store_returns_direct_scheme_uris() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let store = ConfigurableFileSystemObjectStore::with_direct_io();
        let base_uri = format!("direct://{}/", temp_dir.path().display());

        let file_uri = format!("{base_uri}data.bin");
        store
            .put(&file_uri, Bytes::from(vec![0u8; 4096]))
            .await
            .unwrap();

        let non_recursive = store.list(&base_uri, false).await.unwrap();
        assert!(
            non_recursive.iter().all(|u| u.starts_with("direct://")),
            "non-recursive list() on a direct:// store must return direct:// \
             URIs, got: {non_recursive:?}"
        );

        let recursive = store.list(&base_uri, true).await.unwrap();
        assert!(
            recursive.iter().all(|u| u.starts_with("direct://")),
            "recursive list() on a direct:// store must return direct:// \
             URIs, got: {recursive:?}"
        );
    }
}

#[cfg(test)]
mod direct_io_chunking_tests {
    //! Issue #244: a `direct://` PUT wrote ONE 4 KiB block per `write()` (512 MiB took 45 s).
    //! The planner tests are the structural proof (how many system calls, how large); the
    //! round-trip tests use a REAL O_DIRECT file (no buffered fallback in between).
    use super::*;
    use crate::constants::{DEFAULT_DIRECT_IO_BOUNCE_SIZE, DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE};
    use crate::direct_bounce::{GetStrategy, PutStrategy, NEVER_CANCELLED};

    const A: usize = 4096;
    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;
    const BOUNCE: usize = 64 * MIB;
    const MAX_IO: usize = MAX_SINGLE_IO_BYTES;

    fn covered(steps: &[DirectIoStep], len: usize) {
        // Steps are contiguous from 0, in order, and cover exactly `len` payload bytes.
        let mut next = 0;
        for s in steps {
            let (offset, data_len, io_len) = match *s {
                DirectIoStep::FromSource { offset, len } => (offset, len, len),
                DirectIoStep::ViaBounce {
                    offset,
                    data_len,
                    io_len,
                } => (offset, data_len, io_len),
            };
            assert_eq!(offset, next, "gap or overlap in {steps:?}");
            assert_eq!(offset % A, 0, "every I/O offset must be aligned");
            assert_eq!(io_len % A, 0, "every I/O length must be aligned");
            assert!(data_len > 0 && data_len <= io_len);
            next += data_len;
        }
        assert_eq!(next, len);
    }

    #[test]
    fn aligned_gib_source_is_one_write() {
        let steps = plan_direct_write(GIB, 7 * A, A, BOUNCE, MAX_IO);
        assert_eq!(
            steps,
            vec![DirectIoStep::FromSource {
                offset: 0,
                len: GIB
            }]
        );
    }

    #[test]
    fn aligned_source_is_split_only_at_the_kernel_per_call_limit() {
        let len = 3 * GIB;
        let steps = plan_direct_write(len, 0, A, BOUNCE, MAX_IO);
        assert_eq!(steps.len(), 2, "{steps:?}");
        for s in &steps {
            match s {
                DirectIoStep::FromSource { len, .. } => assert!(*len <= MAX_IO),
                other => panic!("unexpected {other:?}"),
            }
        }
        covered(&steps, len);
    }

    #[test]
    fn aligned_source_with_unaligned_length_has_one_padded_tail_block() {
        let steps = plan_direct_write(GIB + 1, 0, A, BOUNCE, MAX_IO);
        assert_eq!(
            steps,
            vec![
                DirectIoStep::FromSource {
                    offset: 0,
                    len: GIB
                },
                DirectIoStep::ViaBounce {
                    offset: GIB,
                    data_len: 1,
                    io_len: A
                },
            ]
        );
    }

    /// The per-call arm with a 1 GiB cap (kept for comparison): an unaligned object up to 1 GiB
    /// is ONE write; the default pooled chunk is 16 MiB, so 1 GiB is 64 pipelined writes.
    #[test]
    fn per_call_gib_cap_is_one_write_and_the_default_pool_chunk_is_16_mib() {
        for len in [MIB + 3, 512 * MIB, GIB] {
            let steps = plan_direct_write(len, 16, A, GIB, MAX_IO);
            assert_eq!(steps.len(), 1, "{len}: {steps:?}");
        }
        assert_eq!(DEFAULT_DIRECT_IO_BOUNCE_SIZE, 16 * MIB);
        let steps = plan_direct_write(GIB, 16, A, DEFAULT_DIRECT_IO_BOUNCE_SIZE, MAX_IO);
        assert_eq!(steps.len(), 64);
    }

    #[test]
    fn unaligned_source_is_written_in_bounce_sized_writes() {
        let len = 512 * MIB;
        let steps = plan_direct_write(len, 16, A, BOUNCE, MAX_IO);
        assert_eq!(steps.len(), 8, "512 MiB at 64 MiB per write");
        for s in &steps {
            assert!(matches!(s, DirectIoStep::ViaBounce { io_len, .. } if *io_len == BOUNCE));
        }
        covered(&steps, len);
    }

    /// The regression itself: the old loop issued len/4096 writes.
    #[test]
    fn a_512_mib_payload_is_never_written_in_chunks_below_one_mib() {
        for src_addr in [0usize, 16, A, A + 1] {
            let len = 512 * MIB;
            let steps = plan_direct_write(len, src_addr, A, BOUNCE, MAX_IO);
            assert!(steps.len() <= 16, "{} writes for {len} bytes", steps.len());
            let (last, rest) = steps.split_last().unwrap();
            for s in rest.iter().chain(std::iter::once(last)) {
                let io = match s {
                    DirectIoStep::FromSource { len, .. } => *len,
                    DirectIoStep::ViaBounce { io_len, .. } => *io_len,
                };
                assert!(io >= MIB, "{s:?}");
            }
        }
    }

    #[test]
    fn tail_and_degenerate_sizes_are_planned_exactly() {
        for src_addr in [0usize, 16] {
            assert!(plan_direct_write(0, src_addr, A, BOUNCE, MAX_IO).is_empty());
            for len in [1, 4095, 4096, 4097, MIB + 3, 33 * MIB, 64 * MIB + 1] {
                let steps = plan_direct_write(len, src_addr, A, BOUNCE, MAX_IO);
                covered(&steps, len);
                assert!(steps.len() <= 2, "{len} -> {steps:?}");
            }
        }
        // A bounce smaller than the alignment is clamped to one alignment unit.
        let steps = plan_direct_write(3 * A + 5, 1, A, 1, MAX_IO);
        assert_eq!(steps.len(), 4);
        covered(&steps, 3 * A + 5);
    }

    #[test]
    fn planner_covers_every_size_around_the_boundaries() {
        let bounce = 8 * A;
        for src_addr in [0usize, 1] {
            for len in 0..(40 * A) {
                covered(&plan_direct_write(len, src_addr, A, bounce, 16 * A), len);
            }
        }
    }

    // --- real O_DIRECT round trips -------------------------------------------------

    #[cfg(unix)]
    mod real_files {
        use super::*;
        use std::os::unix::fs::OpenOptionsExt;

        /// A scratch dir on a filesystem that supports O_DIRECT (`S3DLIO_TEST_DIR`, else the
        /// cargo target dir). O_DIRECT is unsupported on tmpfs: then the test is SKIPPED with a
        /// message instead of failing in an unwrap.
        fn dir() -> Option<tempfile::TempDir> {
            let base = std::env::var_os("S3DLIO_TEST_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp")
                });
            std::fs::create_dir_all(&base).unwrap();
            let d = tempfile::Builder::new()
                .prefix("direct-chunking-244-")
                .tempdir_in(base)
                .unwrap();
            let probe = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .custom_flags(libc::O_DIRECT)
                .open(d.path().join("probe"));
            if let Err(e) = probe {
                eprintln!(
                    "SKIPPED: {} does not support O_DIRECT ({e}); set S3DLIO_TEST_DIR to a \
                     directory on a real filesystem (not tmpfs)",
                    d.path().display()
                );
                return None;
            }
            Some(d)
        }

        fn pattern(len: usize) -> Vec<u8> {
            (0..len)
                .map(|i| (i.wrapping_mul(131) ^ (i >> 8).wrapping_mul(7)) as u8)
                .collect()
        }

        fn open_direct_write(path: &Path) -> std::fs::File {
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
                .expect("this filesystem must support O_DIRECT (not tmpfs)")
        }

        const SIZES: [usize; 8] = [0, 1, 4095, 4096, 4097, MIB + 3, 33 * MIB, 64 * MIB + 5];

        fn check(len: usize, aligned_src: bool, strategy: PutStrategy) {
            let Some(d) = dir() else { return };
            let path = d.path().join("f.bin");
            let raw = pattern(len);

            // An aligned copy of the payload, or the plain Vec (16-byte aligned heap).
            let mut aligned = AlignedBuf::new(len.max(1), A).unwrap();
            let data: &[u8] = if aligned_src {
                aligned.as_mut_slice()[..len].copy_from_slice(&raw);
                &aligned.as_slice()[..len]
            } else {
                &raw
            };
            if aligned_src {
                assert_eq!(data.as_ptr() as usize % A, 0);
            }

            let file = open_direct_write(&path);
            write_all_direct_with(
                &|| file.try_clone(),
                data,
                A,
                strategy,
                &AlignedBuf::new,
                &NEVER_CANCELLED,
            )
            .expect("O_DIRECT write");
            drop(file);
            assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, len);
            assert_eq!(std::fs::read(&path).unwrap(), raw, "len {len}");

            if len > 0 {
                let rf = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)
                    .unwrap();
                for get in [GetStrategy::Direct, GetStrategy::Pooled] {
                    let back = read_all_direct_with(&rf, len, A, get, &NEVER_CANCELLED)
                        .expect("O_DIRECT read");
                    assert_eq!(
                        back.as_ref(),
                        raw.as_slice(),
                        "read back {get:?}, len {len}"
                    );
                    assert_eq!(
                        back.as_ptr() as usize % A,
                        0,
                        "a direct GET must return page-aligned memory ({get:?}, len {len})"
                    );
                }
            }
        }

        const PER_CALL_1GIB: PutStrategy = PutStrategy::PerCall { cap: GIB };
        const PER_CALL_64MIB: PutStrategy = PutStrategy::PerCall { cap: BOUNCE };

        #[test]
        fn roundtrip_matrix_unaligned_source_pooled() {
            for len in SIZES {
                check(len, false, PutStrategy::Pooled);
            }
        }

        #[test]
        fn roundtrip_matrix_unaligned_source_per_call() {
            for len in SIZES {
                check(len, false, PER_CALL_1GIB);
                check(len, false, PER_CALL_64MIB);
            }
        }

        #[test]
        fn roundtrip_matrix_aligned_source() {
            for len in SIZES {
                check(len, true, PutStrategy::Pooled);
                check(len, true, PER_CALL_64MIB);
            }
        }

        /// Many pooled chunks through a real O_DIRECT file (130 MiB is nine chunks at the 16 MiB default).
        #[test]
        fn pooled_put_of_many_chunks_roundtrips() {
            check(130 * MIB + 7, false, PutStrategy::Pooled);
        }

        #[test]
        fn roundtrip_with_a_small_bounce_buffer_uses_many_steps_correctly() {
            for len in [1, 4097, MIB + 3, 5 * MIB + 17] {
                check(len, false, PutStrategy::PerCall { cap: 64 * A });
                check(len, true, PutStrategy::PerCall { cap: 64 * A });
            }
        }

        /// A caller-supplied destination of any alignment is filled through the shared pool
        /// (the building block of `get_into`), for objects above and below the pool threshold.
        #[test]
        fn read_into_an_unaligned_destination_through_the_pool() {
            let Some(d) = dir() else { return };
            for len in [4097, MIB + 3, 70 * MIB + 11] {
                let raw = pattern(len);
                let path = d.path().join("g.bin");
                std::fs::write(&path, &raw).unwrap();
                let rf = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)
                    .unwrap();
                for off in [0usize, 3, 16, A + 1] {
                    let mut dest = vec![0x77u8; len + off + 8];
                    // SAFETY: dest has room for len bytes at off.
                    unsafe {
                        read_all_direct_into(
                            &rf,
                            len,
                            A,
                            dest.as_mut_ptr().add(off),
                            &NEVER_CANCELLED,
                        )
                    }
                    .unwrap();
                    assert_eq!(&dest[off..off + len], raw.as_slice(), "len {len} off {off}");
                    assert!(dest[..off].iter().all(|&b| b == 0x77));
                    assert!(dest[off + len..].iter().all(|&b| b == 0x77));
                }
            }
        }

        /// A file that is shorter than its size claims (it shrank) is an error in every GET
        /// strategy, so the caller's buffered fallback re-reads it.
        #[test]
        fn a_file_that_shrank_is_an_error_so_the_buffered_fallback_rereads() {
            let Some(d) = dir() else { return };
            let path = d.path().join("s.bin");
            std::fs::write(&path, pattern(8 * MIB)).unwrap();
            let rf = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&path)
                .unwrap();
            for get in [GetStrategy::Direct, GetStrategy::Pooled] {
                let e = read_all_direct_with(&rf, 16 * MIB, A, get, &NEVER_CANCELLED)
                    .expect_err("claimed size is larger than the file");
                assert!(!e.to_string().is_empty(), "{get:?}");
            }
            let mut dest = vec![0u8; 16 * MIB];
            assert!(unsafe {
                read_all_direct_into(&rf, 16 * MIB, A, dest.as_mut_ptr(), &NEVER_CANCELLED)
            }
            .is_err());
        }

        /// Round 2, item 3: a failed bounce allocation retries with half the size, and only
        /// gives up (an error, never a panic) when not even one alignment unit can be had.
        #[test]
        fn bounce_allocation_failure_falls_back_to_a_smaller_buffer_then_errors() {
            let Some(d) = dir() else { return };
            let raw = pattern(5 * MIB + 17);

            // Allocation of anything over 1 MiB "fails": the write must still succeed.
            let path = d.path().join("a.bin");
            let file = open_direct_write(&path);
            let limited = |len: usize, align: usize| {
                if len > MIB {
                    Err(AlignedAllocError { len, align })
                } else {
                    AlignedBuf::new(len, align)
                }
            };
            write_all_direct_with(
                &|| file.try_clone(),
                &raw,
                A,
                PER_CALL_64MIB,
                &limited,
                &NEVER_CANCELLED,
            )
            .expect("smaller bounce");
            drop(file);
            assert_eq!(std::fs::read(&path).unwrap(), raw);

            // Every allocation fails: an error the caller turns into the buffered fallback.
            let path = d.path().join("b.bin");
            let file = open_direct_write(&path);
            let never = |len: usize, align: usize| Err(AlignedAllocError { len, align });
            assert!(write_all_direct_with(
                &|| file.try_clone(),
                &raw,
                A,
                PER_CALL_64MIB,
                &never,
                &NEVER_CANCELLED
            )
            .is_err());
        }

        /// Round 2, item 5 (#249): the stream is made durable once, at finalize, not per flush.
        #[tokio::test]
        async fn streaming_writer_syncs_once_not_per_flush() {
            let Some(d) = dir() else { return };
            let path = d.path().join("w.bin");
            let cfg = FileSystemConfig::direct_io();
            let before = SYNC_CALLS.with(|c| c.get());
            let mut w = Box::new(DirectIOWriter::new(path.clone(), cfg).await.unwrap());
            let raw = pattern(200 * MIB + 17);
            for chunk in raw.chunks(MIB + 3) {
                w.write_chunk(chunk).await.unwrap();
            }
            w.finalize().await.unwrap();
            let syncs = SYNC_CALLS.with(|c| c.get()) - before;
            // EXACTLY two: one for the O_DIRECT file and one for the buffered sub-page tail
            // (a bound like `<= 2` would also pass with zero, i.e. with durability removed).
            assert_eq!(
                syncs, 2,
                "{syncs} fsyncs for an aligned-plus-tail stream (expected exactly 2)"
            );
            assert_eq!(std::fs::read(&path).unwrap(), raw);
        }

        /// A stream whose total length is page-aligned has no buffered tail: EXACTLY one fsync.
        #[tokio::test]
        async fn streaming_writer_aligned_only_stream_syncs_exactly_once() {
            let Some(d) = dir() else { return };
            let path = d.path().join("a.bin");
            let cfg = FileSystemConfig::direct_io();
            let before = SYNC_CALLS.with(|c| c.get());
            let mut w = Box::new(DirectIOWriter::new(path.clone(), cfg).await.unwrap());
            let mut buf = AlignedBuf::new(4 * MIB, A).unwrap();
            buf.as_mut_slice().fill(7);
            for _ in 0..3 {
                w.write_chunk(buf.as_slice()).await.unwrap();
            }
            w.finalize().await.unwrap();
            let syncs = SYNC_CALLS.with(|c| c.get()) - before;
            assert_eq!(
                syncs, 1,
                "{syncs} fsyncs for an aligned-only stream (expected exactly 1)"
            );
            assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, 12 * MIB);
        }

        /// `sync_writes` is the explicit request for per-write durability and is honoured.
        #[tokio::test]
        async fn streaming_writer_honours_sync_writes() {
            let Some(d) = dir() else { return };
            let path = d.path().join("w.bin");
            let cfg = FileSystemConfig {
                sync_writes: true,
                ..FileSystemConfig::direct_io()
            };
            let before = SYNC_CALLS.with(|c| c.get());
            let mut w = Box::new(DirectIOWriter::new(path.clone(), cfg).await.unwrap());
            // Page-aligned chunks are written straight through, one write each.
            let mut buf = AlignedBuf::new(4 * MIB, A).unwrap();
            buf.as_mut_slice().fill(9);
            for _ in 0..3 {
                w.write_chunk(buf.as_slice()).await.unwrap();
            }
            w.finalize().await.unwrap();
            let syncs = SYNC_CALLS.with(|c| c.get()) - before;
            // Three page-aligned writes, each followed by an fsync, plus the one at finalize.
            assert_eq!(
                syncs, 4,
                "sync_writes must fsync after each write and at finalize ({syncs})"
            );
            assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, 12 * MIB);
        }

        /// Round 3, item 8: after an unaligned chunk, a big chunk whose rest starts page-aligned
        /// is topped up to the boundary and written from the caller's memory (no staging copy).
        #[tokio::test]
        async fn streaming_writer_tops_up_then_writes_from_the_callers_memory() {
            let Some(d) = dir() else { return };
            let path = d.path().join("t.bin");
            let cfg = FileSystemConfig {
                direct_io: true,
                ..FileSystemConfig::default()
            };
            let mut w = Box::new(DirectIOWriter::new(path.clone(), cfg).await.unwrap());
            let head = pattern(100); // staged: 100 bytes
            w.write_chunk(&head).await.unwrap();
            // The big chunk starts 100 bytes into a page-aligned buffer, so after a 3996-byte
            // top-up the rest starts exactly on a page boundary.
            let big = pattern(3 * MIB + 500);
            let mut buf = AlignedBuf::new_zeroed(4 * MIB, A).unwrap();
            buf.as_mut_slice()[100..100 + big.len()].copy_from_slice(&big);
            let before = FROM_CALLER_WRITES.with(|c| c.get());
            w.write_chunk(&buf.as_slice()[100..100 + big.len()])
                .await
                .unwrap();
            assert!(
                FROM_CALLER_WRITES.with(|c| c.get()) > before,
                "the rest of the chunk was copied through staging"
            );
            w.finalize().await.unwrap();
            let mut expect = head;
            expect.extend_from_slice(&big);
            assert_eq!(std::fs::read(&path).unwrap(), expect);
        }

        /// A chunk whose rest does NOT start page-aligned keeps going through staging (the
        /// documented limit) and is still correct.
        #[tokio::test]
        async fn streaming_writer_unalignable_chunk_after_staged_data_is_still_correct() {
            let Some(d) = dir() else { return };
            let path = d.path().join("u.bin");
            let cfg = FileSystemConfig {
                direct_io: true,
                ..FileSystemConfig::default()
            };
            let mut w = Box::new(DirectIOWriter::new(path.clone(), cfg).await.unwrap());
            let head = pattern(100);
            w.write_chunk(&head).await.unwrap();
            let big = pattern(2 * MIB + 3);
            let before = FROM_CALLER_WRITES.with(|c| c.get());
            w.write_chunk(&big).await.unwrap(); // Vec memory: 16-byte aligned only
            assert_eq!(FROM_CALLER_WRITES.with(|c| c.get()), before);
            w.finalize().await.unwrap();
            let mut expect = head;
            expect.extend_from_slice(&big);
            assert_eq!(std::fs::read(&path).unwrap(), expect);
        }

        /// Round 3, item 7: documented partial-file behaviour. A writer dropped before
        /// `finalize()` leaves what was flushed (a multiple of the staging size); `cancel()`
        /// removes the file.
        #[tokio::test]
        async fn dropped_writer_leaves_a_partial_file_and_cancel_removes_it() {
            let Some(d) = dir() else { return };
            let cfg = FileSystemConfig {
                direct_io: true,
                ..FileSystemConfig::default()
            };
            let staging = round_down(DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE, A);

            let kept = d.path().join("kept.bin");
            let mut w = Box::new(
                DirectIOWriter::new(kept.clone(), cfg.clone())
                    .await
                    .unwrap(),
            );
            let data = pattern(staging + 6 * MIB);
            w.write_chunk(&data).await.unwrap();
            drop(w); // no finalize
            let on_disk = std::fs::read(&kept).unwrap();
            assert_eq!(
                on_disk.len(),
                staging,
                "only the flushed staging buffers remain"
            );
            assert_eq!(on_disk, data[..staging]);

            let gone = d.path().join("gone.bin");
            let mut w = Box::new(DirectIOWriter::new(gone.clone(), cfg).await.unwrap());
            w.write_chunk(&data).await.unwrap();
            w.cancel().await.unwrap();
            assert!(!gone.exists(), "cancel() must remove the partial file");
        }

        /// Round 3, item 7: a configured alignment of 0 is clamped to 1, never a division by zero.
        #[tokio::test]
        async fn alignment_zero_in_the_config_does_not_panic() {
            let Some(d) = dir() else { return };
            let cfg = FileSystemConfig {
                direct_io: true,
                alignment: 0,
                ..FileSystemConfig::default()
            };
            let store = ConfigurableFileSystemObjectStore::new(cfg);
            let uri = format!("direct://{}/z.bin", d.path().display());
            let raw = pattern(3 * MIB + 5);
            store.put(&uri, Bytes::from(raw.clone())).await.unwrap();
            assert_eq!(store.get(&uri).await.unwrap().as_ref(), raw.as_slice());
            let part = store.get_range(&uri, 1000, Some(5000)).await.unwrap();
            assert_eq!(part.as_ref(), &raw[1000..6000]);
        }

        /// Round 3, item 7: offset + length arithmetic is checked: an absurd range is an error,
        /// not a wrapped size (debug builds used to panic on the overflow).
        #[tokio::test]
        async fn absurd_range_is_an_error_not_an_overflow() {
            let Some(d) = dir() else { return };
            let store = ConfigurableFileSystemObjectStore::new(FileSystemConfig {
                direct_io: true,
                ..FileSystemConfig::default()
            });
            let path = d.path().join("r.bin");
            std::fs::write(&path, pattern(MIB)).unwrap();
            let e = store
                .try_read_range_direct(&path, 4097, Some(u64::MAX))
                .await
                .expect_err("range of u64::MAX bytes");
            assert!(e.to_string().contains("too large"), "{e}");
        }
    }
}
