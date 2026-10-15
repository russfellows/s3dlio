// src/file_store_direct.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::memory::{AlignedAllocError, AlignedBuf, BufferPool, BufferPoolConfig};
use crate::object_store::WriterOptions;
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use anyhow::{bail, Result};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use crc32fast::Hasher;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tracing::{debug, info, trace, warn};

use crate::constants::{
    direct_io_bounce_size, DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD,
    DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE, DEFAULT_MIN_IO_SIZE, DEFAULT_PAGE_SIZE,
    MAX_SINGLE_IO_BYTES,
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
        let buffer_pool = Some(
            BufferPoolConfig {
                capacity: 32,                  // 32 buffers in pool
                buffer_size: 64 * 1024 * 1024, // 64MB per buffer (matches chunk_size)
                alignment: page_size,
            }
            .build(),
        );

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
        let buffer_pool = Some(
            BufferPoolConfig {
                capacity: 32,
                buffer_size: 64 * 1024 * 1024,
                alignment: page_size,
            }
            .build(),
        );

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

/// Plan and allocate the bounce buffer for `len` bytes, halving the bounce size when the
/// allocation fails (down to one alignment unit) instead of aborting.
#[cfg(unix)]
fn plan_with_bounce(
    len: usize,
    src_addr: usize,
    alignment: usize,
    bounce_size: usize,
    alloc: AllocFn<'_>,
) -> Result<(Vec<DirectIoStep>, Option<AlignedBuf>)> {
    let mut bounce_size = bounce_size;
    loop {
        let steps = plan_direct_write(len, src_addr, alignment, bounce_size, MAX_SINGLE_IO_BYTES);
        let need = steps
            .iter()
            .filter_map(|s| match s {
                DirectIoStep::ViaBounce { io_len, .. } => Some(*io_len),
                DirectIoStep::FromSource { .. } => None,
            })
            .max()
            .unwrap_or(0);
        if need == 0 {
            return Ok((steps, None));
        }
        match alloc(need, alignment) {
            Ok(buf) => return Ok((steps, Some(buf))),
            Err(e) => {
                let smaller = round_down(need / 2, alignment);
                if smaller < alignment {
                    return Err(e.into());
                }
                warn!(
                    "cannot allocate a {need} byte O_DIRECT bounce buffer; retrying with {smaller}"
                );
                bounce_size = smaller;
            }
        }
    }
}

/// Write all of `data` to `file` (opened with O_DIRECT) following [`plan_direct_write`],
/// then truncate the file to `data.len()` when the last block was padded.
///
/// `write_all_at` resumes after a short write, so a partial write is never lost; if a short
/// write leaves the next offset unaligned the kernel refuses it (EINVAL) and the caller's
/// buffered fallback rewrites the whole file. A failed bounce-buffer allocation retries with
/// half the size instead of panicking; only when even one alignment unit cannot be had does
/// it return an error (and the caller falls back to buffered I/O).
#[cfg(unix)]
pub(crate) fn write_all_direct(
    file: &std::fs::File,
    data: &[u8],
    alignment: usize,
    bounce_size: usize,
) -> Result<()> {
    write_all_direct_with(file, data, alignment, bounce_size, &AlignedBuf::new)
}

#[cfg(unix)]
fn write_all_direct_with(
    file: &std::fs::File,
    data: &[u8],
    alignment: usize,
    bounce_size: usize,
    alloc: AllocFn<'_>,
) -> Result<()> {
    let (steps, mut bounce) = plan_with_bounce(
        data.len(),
        data.as_ptr() as usize,
        alignment,
        bounce_size,
        alloc,
    )?;

    for step in steps {
        match step {
            DirectIoStep::FromSource { offset, len } => {
                write_all_at(file, &data[offset..offset + len], offset as u64)?;
            }
            DirectIoStep::ViaBounce {
                offset,
                data_len,
                io_len,
            } => {
                let Some(bounce) = bounce.as_mut() else {
                    bail!("internal error: bounce step planned without a bounce buffer");
                };
                // SAFETY: the buffer holds at least `io_len` bytes (it was sized for the
                // largest bounce step); `[0, data_len)` is copied from `data`, `[data_len,
                // io_len)` is zero-filled, so the `io_len` bytes read below are initialised.
                let buf = unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(offset),
                        bounce.as_mut_ptr(),
                        data_len,
                    );
                    std::ptr::write_bytes(bounce.as_mut_ptr().add(data_len), 0, io_len - data_len);
                    std::slice::from_raw_parts(bounce.as_ptr(), io_len)
                };
                write_all_at(file, buf, offset as u64)?;
            }
        }
    }

    // Remove the zero padding of the last block (none when the length is aligned).
    if !data.len().is_multiple_of(alignment.max(1)) {
        file.set_len(data.len() as u64)?;
    }
    Ok(())
}

/// A page-aligned buffer exposing its first `len` bytes (what the kernel filled) as the
/// owner of a `Bytes`, so a `direct://` GET hands back the buffer it read into, with no
/// copy, and the result is itself page-aligned (a later `direct://` PUT of it is one write).
struct AlignedBytes {
    buf: AlignedBuf,
    len: usize,
}

impl AsRef<[u8]> for AlignedBytes {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: `[0, len)` was written by the kernel's read (see `read_all_direct`);
        // `len <= buf.len()`.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr(), self.len) }
    }
}

/// Read `file_size` bytes from `file` (opened with O_DIRECT) straight into ONE page-aligned
/// buffer, one `read` per at most [`MAX_SINGLE_IO_BYTES`], and return that buffer as `Bytes`.
/// An allocation failure is returned as an error so the caller can fall back to buffered I/O.
#[cfg(unix)]
pub(crate) fn read_all_direct(
    file: &std::fs::File,
    file_size: usize,
    alignment: usize,
) -> Result<Bytes> {
    use std::os::unix::fs::FileExt;

    if file_size == 0 {
        return Ok(Bytes::new());
    }
    let capacity = round_up(file_size, alignment);
    let buf = AlignedBuf::new(capacity, alignment)?;
    let max_io = round_down(MAX_SINGLE_IO_BYTES, alignment).max(alignment);

    let mut total = 0usize;
    while total < file_size {
        // Whole alignment units at an aligned offset; the last read may come back short at EOF.
        let want = (capacity - total).min(max_io);
        // SAFETY: `[total, total + want)` lies inside the allocation; the kernel overwrites it.
        let dst =
            unsafe { std::slice::from_raw_parts_mut(buf.as_ptr().add(total) as *mut u8, want) };
        match file.read_at(dst, total as u64) {
            Ok(0) => break, // EOF (file shrank)
            Ok(n) => total += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let len = total.min(file_size);
    Ok(Bytes::from_owner(AlignedBytes { buf, len }))
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
pub struct DirectIOWriter {
    /// Buffered mode (`direct_io == false`).
    file: Option<fs::File>,
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
    hasher: Hasher,
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
        let final_path = if compression.is_enabled() {
            let mut path_with_ext = path.clone();
            let ext = match &compression {
                CompressionConfig::None => "",
                CompressionConfig::Zstd { .. } => ".zst",
            };
            if let Some(existing_ext) = path_with_ext.extension() {
                let new_ext = format!("{}{}", existing_ext.to_str().unwrap_or(""), ext);
                path_with_ext.set_extension(new_ext);
            } else {
                let new_name = format!(
                    "{}{}",
                    path_with_ext
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or(""),
                    ext
                );
                path_with_ext.set_file_name(new_name);
            }
            path_with_ext
        } else {
            path.clone()
        };

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
            file = Some(fs::File::create(&final_path).await?);
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
            hasher: Hasher::new(),
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
            if self.staged == 0
                && data.len() >= alignment
                && (data.as_ptr() as usize).is_multiple_of(alignment)
            {
                // Straight from the caller's memory: no copy.
                let n = round_down(data.len(), alignment).min(max_io);
                self.write_direct(&data[..n])?;
                data = &data[n..];
                continue;
            }
            if self.staging.is_none() {
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
            }
            let cap = self.staging_cap;
            let k = (cap - self.staged).min(data.len());
            if let Some(staging) = self.staging.as_mut() {
                // SAFETY: `staged + k <= cap` stays inside the allocation.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        staging.as_mut_ptr().add(self.staged),
                        k,
                    );
                }
            }
            self.staged += k;
            data = &data[k..];
            if self.staged == cap {
                self.flush_staging(cap)?;
                self.staged = 0;
            }
        }
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
        let Some(file) = self.file.as_mut() else {
            bail!("Writer has been finalized or cancelled");
        };
        file.write_all(data).await?;
        if self.config.sync_writes {
            file.sync_all().await?;
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
                } else if let Some(file) = self.file.as_mut() {
                    file.write_all(&compressed_data).await?;
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
        } else if let Some(mut file) = self.file.take() {
            file.flush().await?;
            file.sync_all().await?;
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
            .and_then(|time| time.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
            .map(|duration| {
                // Convert to RFC 3339 format like S3 uses
                let secs = duration.as_secs();
                let datetime =
                    std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs);
                format!("{:?}", datetime) // Simple formatting for now
            });

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
            let alignment = self.config.alignment;

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
                let data = read_all_direct(&file, file_size, alignment)?;
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
            let alignment = self.config.alignment;
            let sync_writes = self.config.sync_writes;
            let bounce_size = direct_io_bounce_size();

            // Use spawn_blocking for sync I/O
            tokio::task::spawn_blocking(move || -> Result<()> {
                use std::fs::OpenOptions;
                use std::os::unix::fs::OpenOptionsExt;

                let mut flags = libc::O_DIRECT;
                if sync_writes {
                    flags |= libc::O_SYNC;
                }

                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .custom_flags(flags)
                    .open(&path)?;

                // One write for the whole aligned body when `data` is aligned, otherwise
                // bounce-buffer-sized writes (issue #244); never one 4 KiB block per call.
                write_all_direct(&file, data.as_ref(), alignment, bounce_size)?;
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
            let mut file = fs::File::open(path).await?;
            file.seek(std::io::SeekFrom::Start(offset)).await?;

            let read_length = length.unwrap_or(u64::MAX);
            let mut buffer = Vec::new();

            if read_length == u64::MAX {
                file.read_to_end(&mut buffer).await?;
            } else {
                buffer.resize(read_length as usize, 0);
                file.read_exact(&mut buffer).await?;
            }

            return Ok(Bytes::from(buffer));
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
                let mut file = fs::File::open(path).await?;
                file.seek(std::io::SeekFrom::Start(offset)).await?;

                let read_length = length.unwrap_or(u64::MAX);
                let mut buffer = Vec::new();

                if read_length == u64::MAX {
                    file.read_to_end(&mut buffer).await?;
                } else {
                    buffer.resize(read_length as usize, 0);
                    file.read_exact(&mut buffer).await?;
                }

                Ok(Bytes::from(buffer))
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
        // O_DIRECT range read requires careful alignment
        let alignment = self.config.alignment as u64;
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

        let total_length = offset_adjustment + read_length as usize;
        let aligned_length = total_length.div_ceil(self.config.alignment) * self.config.alignment;

        // Borrow aligned buffer from pool if available, otherwise allocate fresh (v0.9.9+)
        let mut aligned: AlignedBuf = if let Some(pool) = &self.config.buffer_pool {
            let b = pool.take().await?;
            // Grow buffer if pooled buffer is too small
            if b.len() < aligned_length {
                trace!(
                    "Pool buffer too small ({} < {}), allocating larger buffer",
                    b.len(),
                    aligned_length
                );
                AlignedBuf::new(aligned_length, self.config.alignment)?
            } else {
                b
            }
        } else {
            // No pool configured, allocate fresh buffer
            AlignedBuf::new(aligned_length, self.config.alignment)?
        };

        let options = self.create_open_options(true, false);
        let mut file = options.open(path).await?;

        file.seek(std::io::SeekFrom::Start(aligned_offset)).await?;
        // Loop until we've filled the aligned buffer or reached EOF.
        // A single read() call may return far fewer bytes than requested,
        // especially for large ranges (O_DIRECT imposes per-call limits).
        let mut bytes_read = 0;
        while bytes_read < aligned_length {
            let n = file
                .read(&mut aligned.as_mut_slice()[bytes_read..aligned_length])
                .await?;
            if n == 0 {
                break;
            } // EOF
            bytes_read += n;
        }

        // Extract the requested subrange (single small copy vs entire buffer copy)
        let start = offset_adjustment.min(bytes_read);
        let end = start.saturating_add(read_length as usize).min(bytes_read);
        let needed = end.saturating_sub(start);

        let mut out = BytesMut::with_capacity(needed);
        out.extend_from_slice(&aligned.as_slice()[start..end]);
        let result = out.freeze();

        // Return aligned buffer to pool for reuse (v0.9.9+)
        // give() is non-blocking (try_send): if the pool channel is full because
        // a fallback-allocated buffer was used while the pre-alloc had already
        // filled the channel, we simply drop the buffer rather than deadlocking.
        if let Some(pool) = &self.config.buffer_pool {
            pool.give(aligned);
        }

        Ok(result)
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
        let store = self.clone();
        let uri_owned = uri.to_string();

        let get_range_fn = move |offset: u64, length: u64| {
            let store = store.clone();
            let uri = uri_owned.clone();
            async move { store.get_range(&uri, offset, Some(length)).await }
        };

        let (bytes, stats) = engine.download(object_size, get_range_fn, None).await?;

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
    use crate::constants::DEFAULT_DIRECT_IO_BOUNCE_SIZE;

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

    /// Round 2, item 3: with the default cap an unaligned object up to 1 GiB is ONE write.
    #[test]
    fn default_bounce_makes_an_unaligned_gib_one_write() {
        assert_eq!(DEFAULT_DIRECT_IO_BOUNCE_SIZE, GIB);
        for len in [MIB + 3, 512 * MIB, GIB] {
            let steps = plan_direct_write(len, 16, A, DEFAULT_DIRECT_IO_BOUNCE_SIZE, MAX_IO);
            assert_eq!(steps.len(), 1, "{len}: {steps:?}");
        }
        let steps = plan_direct_write(GIB + 1, 16, A, DEFAULT_DIRECT_IO_BOUNCE_SIZE, MAX_IO);
        assert_eq!(steps.len(), 2);
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

        /// O_DIRECT is unsupported on tmpfs; the cargo target dir is on a real filesystem.
        fn dir() -> tempfile::TempDir {
            let base = std::env::var_os("S3DLIO_TEST_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp")
                });
            std::fs::create_dir_all(&base).unwrap();
            tempfile::Builder::new()
                .prefix("direct-chunking-244-")
                .tempdir_in(base)
                .unwrap()
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

        fn check(len: usize, aligned_src: bool, bounce: usize) {
            let d = dir();
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
            write_all_direct(&file, data, A, bounce).expect("O_DIRECT write");
            drop(file);
            assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, len);
            assert_eq!(std::fs::read(&path).unwrap(), raw, "len {len}");

            if len > 0 {
                let rf = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)
                    .unwrap();
                let back = read_all_direct(&rf, len, A).expect("O_DIRECT read");
                assert_eq!(back.as_ref(), raw.as_slice(), "read back, len {len}");
                assert_eq!(
                    back.as_ptr() as usize % A,
                    0,
                    "a direct GET must return page-aligned memory (len {len})"
                );
            }
        }

        #[test]
        fn roundtrip_matrix_unaligned_source() {
            for len in SIZES {
                check(len, false, BOUNCE);
            }
        }

        #[test]
        fn roundtrip_matrix_aligned_source() {
            for len in SIZES {
                check(len, true, BOUNCE);
            }
        }

        #[test]
        fn roundtrip_with_a_small_bounce_buffer_uses_many_steps_correctly() {
            for len in [1, 4097, MIB + 3, 5 * MIB + 17] {
                check(len, false, 64 * A);
                check(len, true, 64 * A);
            }
        }

        /// Round 2, item 3: a failed bounce allocation retries with half the size, and only
        /// gives up (an error, never a panic) when not even one alignment unit can be had.
        #[test]
        fn bounce_allocation_failure_falls_back_to_a_smaller_buffer_then_errors() {
            let d = dir();
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
            write_all_direct_with(&file, &raw, A, 64 * MIB, &limited).expect("smaller bounce");
            drop(file);
            assert_eq!(std::fs::read(&path).unwrap(), raw);

            // Every allocation fails: an error the caller turns into the buffered fallback.
            let path = d.path().join("b.bin");
            let file = open_direct_write(&path);
            let never = |len: usize, align: usize| Err(AlignedAllocError { len, align });
            assert!(write_all_direct_with(&file, &raw, A, 64 * MIB, &never).is_err());
        }

        /// Round 2, item 5 (#249): the stream is made durable once, at finalize, not per flush.
        #[tokio::test]
        async fn streaming_writer_syncs_once_not_per_flush() {
            let d = dir();
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
            // One for the O_DIRECT file and one for the buffered tail.
            assert!(
                syncs <= 2,
                "{syncs} fsyncs for one stream (was one per flush)"
            );
            assert_eq!(std::fs::read(&path).unwrap(), raw);
        }

        /// `sync_writes` is the explicit request for per-write durability and is honoured.
        #[tokio::test]
        async fn streaming_writer_honours_sync_writes() {
            let d = dir();
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
            assert!(
                syncs >= 3,
                "sync_writes must fsync after each write ({syncs})"
            );
            assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, 12 * MIB);
        }
    }
}
