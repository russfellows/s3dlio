// src/file_get_into.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Caller-buffer GET (`ObjectStore::get_into`) and typed HEAD (`ObjectStore::head`) for the
// local file stores, `file://` and `direct://` (#295, #267). One implementation serves
// `FileSystemObjectStore` and `ConfigurableFileSystemObjectStore`; the contract is the one #264
// gives the network stores:
//
// * `dst.len()` is the expected length. `Ok(())` means exactly that many bytes were written AND
//   the file has exactly that size. A different size is `LengthMismatch` (decided from the
//   file's size BEFORE anything is read, so nothing is written, and re-checked afterwards in
//   case the file changed under the read). Nothing is ever written past `dst`.
// * A missing file is `NotFound`, an unreadable one `Forbidden`, a directory (or any other
//   non-regular file) `Other`. A HEAD of a missing file is `Ok(None)`.
// * Progress reports the DISTINCT destination positions filled, recorded after each copy.
//
// ## Zero copy
//
// * `file://` (and any store with direct I/O off): `pread(2)` straight into the caller's
//   buffer. No other buffer is allocated.
// * `direct://` with a destination whose ADDRESS is a multiple of the store's alignment: the
//   alignment-multiple part is read with O_DIRECT straight into the caller's buffer. No copy.
// * `direct://` otherwise (unaligned address, or the unaligned tail of an aligned one, which is
//   less than one alignment unit): the read goes through the shared bounce pool
//   (`read_direct_into`): the kernel fills an aligned bounce buffer and the bytes are copied
//   once into the destination. That copy cannot be avoided (O_DIRECT forbids unaligned memory),
//   it is the documented exception to Rule #1, and it is bounded: the pool's own buffers,
//   never a buffer of the object's size.
//
// ## Cancellation
//
// `dst` is a borrowed `&mut [u8]`, so it is never handed to `spawn_blocking` (the future could
// be dropped while a thread still wrote into it). Every read runs INSIDE the poll of the
// returned future, through [`blocking_io`] (`block_in_place` on a multi-thread runtime, inline
// otherwise), one bounded chunk at a time ([`GET_INTO_CHUNK_BYTES`]), with a yield between
// chunks. A running read is therefore never separated from its future: when the future is
// dropped (or its task aborted) it is dropped at a chunk boundary, with no writer left.

use crate::data_loader::options::PageCacheMode;
use crate::object_error::{attach_object_error, ObjectError, ObjectErrorKind};
use crate::object_store::HeadMetadata;
use crate::receive_progress::{ReceiveProgress, SpanMark};
use anyhow::Result;
use std::fs::File;
use std::io;
use std::path::Path;
use std::time::SystemTime;

/// The most bytes one read call asks for. Bounds how long a read runs without yielding to the
/// runtime (the granularity of cancellation), and the size of one pass through the bounce pool.
pub(crate) const GET_INTO_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// Run blocking file I/O from async code without stalling a runtime worker: on a multi-thread
/// runtime the worker hands its queue to another thread first; otherwise the closure just runs
/// inline.
pub(crate) fn blocking_io<R>(f: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

/// O_DIRECT parameters of a store that reads with direct I/O.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectParams {
    /// The store's alignment for addresses, lengths and file offsets (never 0).
    pub(crate) alignment: usize,
    /// Files smaller than this are read buffered (as `get` does).
    pub(crate) min_io_size: usize,
}

/// What a store asks of [`get_into`].
pub(crate) struct LocalRead<'a> {
    pub(crate) path: &'a Path,
    /// The provider named in errors: `"file"` or `"direct"` (the URI's scheme).
    pub(crate) provider: &'static str,
    /// `Some` when the store reads with O_DIRECT.
    pub(crate) direct: Option<DirectParams>,
    /// The `posix_fadvise` hint for a buffered read, when the store applies one.
    pub(crate) page_cache: Option<PageCacheMode>,
}

/// The provider a URI names.
pub(crate) fn provider_of(uri: &str) -> &'static str {
    if uri.starts_with("direct://") {
        "direct"
    } else {
        "file"
    }
}

/// The identity of a file for `stat` and `head`: its size and modification time in whole
/// seconds. Not a content hash; it changes when the file is rewritten.
pub(crate) fn stat_etag(len: u64, modified: Option<SystemTime>) -> String {
    format!(
        "file-{}-{}",
        len,
        modified
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    )
}

// ---------------------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------------------

fn kind_of_io(e: &io::Error) -> ObjectErrorKind {
    match e.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory => ObjectErrorKind::NotFound,
        io::ErrorKind::PermissionDenied => ObjectErrorKind::Forbidden,
        _ => ObjectErrorKind::Other,
    }
}

/// A failed system call on `path`, classified from the OS error and keeping it as the source.
fn io_failure(provider: &'static str, path: &Path, what: &str, e: io::Error) -> anyhow::Error {
    let kind = kind_of_io(&e);
    attach_object_error(
        anyhow::Error::new(e),
        kind,
        provider,
        format!("cannot {what} {}", path.display()),
    )
}

fn other_error(provider: &'static str, message: String) -> anyhow::Error {
    anyhow::Error::new(ObjectError::new(ObjectErrorKind::Other, provider, message))
}

fn length_mismatch(
    provider: &'static str,
    path: &Path,
    expected: u64,
    actual: u64,
) -> anyhow::Error {
    anyhow::Error::new(ObjectError::new(
        ObjectErrorKind::LengthMismatch {
            expected,
            actual: Some(actual),
        },
        provider,
        format!(
            "{} is {actual} bytes but the destination holds {expected}",
            path.display()
        ),
    ))
}

fn not_a_regular_file(provider: &'static str, path: &Path) -> anyhow::Error {
    other_error(
        provider,
        format!("{} is not a regular file", path.display()),
    )
}

// ---------------------------------------------------------------------------------------
// HEAD
// ---------------------------------------------------------------------------------------

/// Typed HEAD of the local file at `path`: `Ok(None)` when it does not exist (or a parent is
/// not a directory), an error for a directory or any other non-regular file and for every
/// failure of `stat` (a refusal is `Forbidden`). The identity is [`stat_etag`]; a local file
/// has no version.
pub(crate) async fn head(path: &Path, provider: &'static str) -> Result<Option<HeadMetadata>> {
    match tokio::fs::metadata(path).await {
        Ok(md) if md.is_file() => Ok(Some(HeadMetadata {
            size: md.len(),
            etag: Some(stat_etag(md.len(), md.modified().ok())),
            version_id: None,
        })),
        Ok(_) => Err(not_a_regular_file(provider, path)),
        Err(e) if kind_of_io(&e) == ObjectErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_failure(provider, path, "stat", e)),
    }
}

// ---------------------------------------------------------------------------------------
// GET into the caller's buffer
// ---------------------------------------------------------------------------------------

/// Why one chunk could not be read.
enum ChunkError {
    /// The file ended before the chunk was full: it shrank after its size was checked.
    Short,
    /// A system call failed.
    Io(io::Error),
    /// Progress could not be recorded (an internal invariant).
    Progress(anyhow::Error),
}

#[derive(Clone, Copy)]
enum Mode {
    /// Positional reads into the caller's buffer.
    Buffered,
    /// O_DIRECT reads, aligned to `alignment`.
    #[cfg(unix)]
    Direct { alignment: usize },
}

/// An open file being read into a destination, chunk by chunk.
struct Reader<'a> {
    spec: &'a LocalRead<'a>,
    file: File,
    expected: usize,
    mode: Mode,
}

/// The OS errors that mean "this file or filesystem does not do O_DIRECT here" (as opposed to a
/// real failure): the read is retried buffered, as the allocating `get` does.
#[cfg(unix)]
fn direct_refused(e: &io::Error) -> bool {
    matches!(
        e.raw_os_error(),
        Some(libc::EINVAL) | Some(libc::EOPNOTSUPP)
    )
}

#[cfg(all(unix, test))]
thread_local! {
    /// Test hook: make the next O_DIRECT open fail with EINVAL, as a filesystem that does not
    /// support O_DIRECT does.
    pub(crate) static REFUSE_DIRECT_OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(unix)]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    #[cfg(test)]
    if REFUSE_DIRECT_OPEN.with(|r| r.get()) {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

/// One `read` at `offset` into `buf`; returns the bytes read (0 at the end of the file).
#[cfg(unix)]
fn read_some_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(not(unix))]
fn read_some_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::io::{Read, Seek};
    let mut file = file;
    file.seek(io::SeekFrom::Start(offset))?;
    file.read(buf)
}

impl<'a> Reader<'a> {
    /// Stat the file, check its size against `expected`, and open it for reading. `None` when
    /// there is nothing to read (an empty file into an empty destination).
    fn open(spec: &'a LocalRead<'a>, expected: usize) -> Result<Option<Self>> {
        let provider = spec.provider;
        let md =
            std::fs::metadata(spec.path).map_err(|e| io_failure(provider, spec.path, "stat", e))?;
        if !md.is_file() {
            return Err(not_a_regular_file(provider, spec.path));
        }
        let size = md.len();
        if size != expected as u64 {
            return Err(length_mismatch(provider, spec.path, expected as u64, size));
        }
        if expected == 0 {
            return Ok(None);
        }
        let (file, mode) = Self::open_file(spec, size)?;
        Ok(Some(Self {
            spec,
            file,
            expected,
            mode,
        }))
    }

    fn open_file(spec: &LocalRead<'_>, size: u64) -> Result<(File, Mode)> {
        #[cfg(unix)]
        if let Some(direct) = spec.direct {
            if size >= direct.min_io_size as u64 {
                match open_direct(spec.path) {
                    Ok(file) => {
                        return Ok((
                            file,
                            Mode::Direct {
                                alignment: direct.alignment.max(1),
                            },
                        ));
                    }
                    Err(e) if direct_refused(&e) => {
                        crate::file_store_direct::note_get_fallback();
                        tracing::warn!(
                            "O_DIRECT open failed for {}, reading with regular I/O: {}",
                            spec.path.display(),
                            e
                        );
                    }
                    Err(e) => return Err(io_failure(spec.provider, spec.path, "open", e)),
                }
            }
        }
        let file =
            File::open(spec.path).map_err(|e| io_failure(spec.provider, spec.path, "open", e))?;
        if let Some(mode) = spec.page_cache {
            // A hint only: failing to apply it changes nothing about the read.
            let _ = crate::page_cache::apply_page_cache_hint(&file, mode, size);
        }
        Ok((file, Mode::Buffered))
    }

    /// Read `dst` (the chunk at `offset` of the object) and record what was filled.
    fn read_chunk(&mut self, dst: &mut [u8], offset: usize, span: &mut SpanMark<'_>) -> Result<()> {
        let first = self.attempt(dst, offset, span);
        #[cfg(unix)]
        let first = match first {
            Err(ChunkError::Io(e))
                if matches!(self.mode, Mode::Direct { .. }) && direct_refused(&e) =>
            {
                // O_DIRECT is refused for this file after all (an alignment or filesystem rule
                // the open did not reveal): read the chunk, and the rest, with regular I/O.
                self.fall_back_to_buffered(&e)?;
                self.attempt(dst, offset, span)
            }
            other => other,
        };
        first.map_err(|e| self.classify(e))
    }

    #[cfg(unix)]
    fn fall_back_to_buffered(&mut self, cause: &io::Error) -> Result<()> {
        crate::file_store_direct::note_get_fallback();
        tracing::warn!(
            "O_DIRECT read failed for {}, reading with regular I/O: {}",
            self.spec.path.display(),
            cause
        );
        self.file = File::open(self.spec.path)
            .map_err(|e| io_failure(self.spec.provider, self.spec.path, "open", e))?;
        self.mode = Mode::Buffered;
        Ok(())
    }

    fn attempt(
        &self,
        dst: &mut [u8],
        offset: usize,
        span: &mut SpanMark<'_>,
    ) -> Result<(), ChunkError> {
        match self.mode {
            Mode::Buffered => self.read_buffered(dst, offset, span),
            #[cfg(unix)]
            Mode::Direct { alignment } => self.read_direct(dst, offset, alignment, span),
        }
    }

    /// `pread` until `dst` is full, straight into it.
    fn read_buffered(
        &self,
        dst: &mut [u8],
        offset: usize,
        span: &mut SpanMark<'_>,
    ) -> Result<(), ChunkError> {
        let mut done = 0;
        while done < dst.len() {
            let n = match read_some_at(&self.file, &mut dst[done..], (offset + done) as u64) {
                Ok(0) => return Err(ChunkError::Short),
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(ChunkError::Io(e)),
            };
            span.record_copy((offset + done) as u64, n as u64)
                .map_err(ChunkError::Progress)?;
            done += n;
        }
        Ok(())
    }

    /// O_DIRECT: the alignment-multiple part of an aligned destination is read straight into
    /// it; whatever remains (all of an unaligned destination, or the sub-alignment tail of an
    /// aligned one) goes through the bounce pool.
    #[cfg(unix)]
    fn read_direct(
        &self,
        dst: &mut [u8],
        offset: usize,
        alignment: usize,
        span: &mut SpanMark<'_>,
    ) -> Result<(), ChunkError> {
        use crate::direct_bounce::{SendPtr, NEVER_CANCELLED};
        use crate::file_store_direct::{pread_raw, read_direct_into};

        let len = dst.len();
        let head = if (dst.as_ptr() as usize).is_multiple_of(alignment) {
            len / alignment * alignment
        } else {
            0
        };

        if head > 0 {
            let max_io =
                (crate::constants::MAX_SINGLE_IO_BYTES / alignment * alignment).max(alignment);
            let base = dst.as_mut_ptr();
            let mut done = 0;
            while done < head {
                let want = (head - done).min(max_io);
                // SAFETY: `[done, done + want)` lies inside `dst` (done + want <= head <= len),
                // which is valid for writes and exclusively ours; its address and `want` are
                // multiples of the alignment, as O_DIRECT requires.
                let got = pread_raw(
                    &self.file,
                    SendPtr(unsafe { base.add(done) }),
                    want,
                    (offset + done) as u64,
                )
                .map_err(ChunkError::Io)?;
                if got == 0 {
                    return Err(ChunkError::Short);
                }
                span.record_copy((offset + done) as u64, got as u64)
                    .map_err(ChunkError::Progress)?;
                done += got;
            }
        }

        if head < len {
            // The documented bounce copy of an unaligned destination (see the module docs). The
            // future runs this inside its own poll, so nothing can cancel it midway: the flag
            // that would only abandon a wait for buffers is never set.
            let tail = len - head;
            let start = offset + head;
            let mut record = |at: usize, n: usize| {
                span.record_copy((start + at) as u64, n as u64)
                    .map(|_| ())
                    .map_err(io::Error::other)
            };
            // SAFETY: `dst[head..]` is `tail` bytes valid for writes and exclusively ours.
            unsafe {
                read_direct_into(
                    &self.file,
                    start as u64,
                    tail,
                    alignment,
                    dst.as_mut_ptr().add(head),
                    &NEVER_CANCELLED,
                    &mut record,
                )
            }
            .map_err(|e| {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    ChunkError::Short
                } else {
                    ChunkError::Io(e)
                }
            })?;
        }
        Ok(())
    }

    fn classify(&self, e: ChunkError) -> anyhow::Error {
        let spec = self.spec;
        match e {
            ChunkError::Short => {
                // The file ended early: report the size it has now.
                let actual = self.file.metadata().map(|m| m.len()).unwrap_or(0);
                length_mismatch(spec.provider, spec.path, self.expected as u64, actual)
            }
            ChunkError::Io(e) => io_failure(spec.provider, spec.path, "read", e),
            ChunkError::Progress(e) => {
                attach_object_error(e, ObjectErrorKind::Other, spec.provider, "progress")
            }
        }
    }

    /// After the last chunk: the file must still be exactly `expected` bytes.
    fn finish(&self) -> Result<()> {
        let spec = self.spec;
        let size = self
            .file
            .metadata()
            .map_err(|e| io_failure(spec.provider, spec.path, "stat", e))?
            .len();
        if size != self.expected as u64 {
            return Err(length_mismatch(
                spec.provider,
                spec.path,
                self.expected as u64,
                size,
            ));
        }
        #[cfg(unix)]
        if matches!(self.mode, Mode::Direct { .. }) {
            crate::file_store_direct::note_direct_get();
        }
        Ok(())
    }
}

/// Read the whole file at `spec.path` into `dst` (see the module docs for the contract).
///
/// Every write into `dst` happens inside this future, chunk by chunk, so dropping it stops
/// every writer.
pub(crate) async fn get_into(
    spec: &LocalRead<'_>,
    dst: &mut [u8],
    progress: &ReceiveProgress,
) -> Result<()> {
    let expected = dst.len();
    let Some(mut reader) = blocking_io(|| Reader::open(spec, expected))? else {
        return Ok(());
    };
    let mut span = SpanMark::new(progress);
    let mut offset = 0;
    while offset < expected {
        let n = GET_INTO_CHUNK_BYTES.min(expected - offset);
        blocking_io(|| reader.read_chunk(&mut dst[offset..offset + n], offset, &mut span))?;
        offset += n;
        if offset < expected {
            // The cancellation point: a dropped future or aborted task stops here, between
            // chunks, with no read in flight.
            tokio::task::yield_now().await;
        }
    }
    blocking_io(|| reader.finish())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| ((i % 160) + (i / 160) % 5) as u8)
            .collect()
    }

    fn spec<'a>(path: &'a Path, direct: Option<DirectParams>) -> LocalRead<'a> {
        LocalRead {
            path,
            provider: if direct.is_some() { "direct" } else { "file" },
            direct,
            page_cache: None,
        }
    }

    fn direct_params() -> Option<DirectParams> {
        Some(DirectParams {
            alignment: 4096,
            min_io_size: 4096,
        })
    }

    /// A filesystem (or file) that refuses O_DIRECT at open falls back to a buffered read into
    /// the same buffer: same bytes, the fallback counted, nothing written past the buffer.
    #[tokio::test(flavor = "current_thread")]
    async fn a_refused_o_direct_open_falls_back_to_a_buffered_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        let data = pattern(3 * MIB + 5);
        std::fs::write(&path, &data).unwrap();

        let before = crate::file_store_direct::direct_io_counters();
        REFUSE_DIRECT_OPEN.with(|r| r.set(true));
        let mut dst = vec![0xA5u8; data.len() + 16];
        let progress = ReceiveProgress::new();
        let r = get_into(
            &spec(&path, direct_params()),
            &mut dst[..data.len()],
            &progress,
        )
        .await;
        REFUSE_DIRECT_OPEN.with(|r| r.set(false));
        r.unwrap();
        assert_eq!(&dst[..data.len()], data.as_slice());
        assert!(dst[data.len()..].iter().all(|b| *b == 0xA5));
        assert_eq!(progress.filled(), data.len() as u64);
        let after = crate::file_store_direct::direct_io_counters();
        assert!(
            after.get_fallbacks > before.get_fallbacks,
            "the fallback is counted"
        );
        assert_eq!(
            after.direct_gets, before.direct_gets,
            "it was not a direct get"
        );
    }

    /// An error that is NOT a refusal of O_DIRECT is never retried buffered: a directory is a
    /// plain typed failure, and no fallback is counted.
    #[tokio::test(flavor = "current_thread")]
    async fn a_directory_is_not_retried_buffered() {
        let dir = tempfile::tempdir().unwrap();
        let before = crate::file_store_direct::direct_io_counters();
        let err = get_into(
            &spec(dir.path(), direct_params()),
            &mut [0u8; 8],
            &ReceiveProgress::new(),
        )
        .await
        .unwrap_err();
        assert_eq!(
            crate::object_error::object_error(&err).map(|e| e.kind),
            Some(ObjectErrorKind::Other)
        );
        let after = crate::file_store_direct::direct_io_counters();
        assert_eq!(after.get_fallbacks, before.get_fallbacks);
    }

    /// The reader records every copy as one contiguous run: the total is the destination
    /// length whatever the chunking.
    #[tokio::test(flavor = "current_thread")]
    async fn progress_totals_the_destination_across_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        let len = 2 * GET_INTO_CHUNK_BYTES + 12_345;
        std::fs::write(&path, pattern(len)).unwrap();
        for direct in [None, direct_params()] {
            let mut dst = vec![0u8; len + 8];
            let progress = ReceiveProgress::new();
            get_into(&spec(&path, direct), &mut dst[8..], &progress)
                .await
                .unwrap();
            assert_eq!(progress.filled(), len as u64);
        }
    }
}
