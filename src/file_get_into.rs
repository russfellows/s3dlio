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
// * A file that CHANGES while it is read is an error, never `Ok` with mixed data: the open
//   file's size, `mtime` and `ctime` are compared with what they were at open ([`FileId`]).
//   A different size is `LengthMismatch`, the same size with new timestamps (an in-place
//   overwrite, truncate-and-regrow, a rename-replace, an unlink, a chmod) is `VersionChanged`.
//   Limits: a write in the same kernel timestamp tick as the last change before the open, made
//   while the read runs, and a shared-`mmap` write that is not yet written back, are not seen.
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
//   never a buffer of the object's size. `get_into` ALWAYS uses the pool for these bytes and
//   ignores `S3DLIO_DIRECT_IO_GET_MODE` (whose `direct` mode allocates an object-sized buffer,
//   which a caller-buffer GET must not do); the pool is tuned by `S3DLIO_DIRECT_IO_BOUNCE_*`.
//
// ## Cancellation
//
// `dst` is a borrowed `&mut [u8]`, so it is never handed to `spawn_blocking` (the future could
// be dropped while a thread still wrote into it). Every read runs INSIDE the poll of the
// returned future, through [`blocking_io`] (`block_in_place` on a multi-thread runtime, inline in a `LocalSet`,
// inline
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

thread_local! {
    /// Set only while [`blocking_io`] is probing `block_in_place`: the panic hook installed by
    /// [`install_quiet_probe_hook`] stays silent for a panic raised on THIS thread in that window.
    static PROBING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Chain a panic hook in front of the current one that swallows the message of the panic the
/// `block_in_place` probe in [`blocking_io`] provokes on purpose (and nothing else: it is silent
/// only on a thread that is inside the probe window, before the closure has started). Installed
/// once per process, the first time a read runs on a multi-thread runtime.
fn install_quiet_probe_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !PROBING.with(|p| p.get()) {
                previous(info);
            }
        }));
    });
}

/// Run blocking file I/O from async code without stalling a runtime worker.
///
/// * On a multi-thread runtime the worker hands its queue to another thread first
///   (`block_in_place`), so the I/O does not stall the tasks queued behind it.
/// * Everywhere else (a current-thread runtime, no runtime) the closure runs inline.
/// * `block_in_place` itself PANICS when it is called from a tokio `LocalSet` that is driven by
///   a thread that is not a runtime worker (`rt.block_on(local.run_until(..))`,
///   `LocalSet::block_on`, `spawn_local` inside either): "can call blocking only when running
///   on the multi-threaded runtime". Tokio (1.52) exposes no way to ask whether that is the
///   case, so the call is made under `catch_unwind` with the closure held back in an `Option`:
///   the panic is raised before the closure is started, so it is still there, and it simply
///   runs inline. While probing, a one-time chained panic hook (see
///   [`install_quiet_probe_hook`]) keeps that expected panic from printing a message; a panic
///   raised BY the closure is not hidden (the flag is cleared before it starts) and is
///   propagated unchanged.
pub(crate) fn blocking_io<R>(f: impl FnOnce() -> R) -> R {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            install_quiet_probe_hook();
            let mut pending = Some(f);
            PROBING.with(|p| p.set(true));
            let probed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tokio::task::block_in_place(|| {
                    PROBING.with(|p| p.set(false));
                    (pending.take().expect("the closure runs once"))()
                })
            }));
            PROBING.with(|p| p.set(false));
            match probed {
                Ok(r) => r,
                // Refused before the closure started: run it here.
                Err(_) if pending.is_some() => (pending.take().expect("checked"))(),
                // The closure itself panicked: not ours to swallow.
                Err(payload) => std::panic::resume_unwind(payload),
            }
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

/// The content version of the OPEN file: its modification and status-change times, with
/// nanoseconds, from `fstat` of the descriptor, taken when the file is opened and compared again
/// after the last chunk. (Device and inode are deliberately not part of it: `fstat` of one open
/// descriptor always returns the same pair, so comparing them would test nothing. A file that is
/// replaced by `rename` over its path, or unlinked, has its LINK COUNT change, and the kernel
/// moves the `ctime` of that inode when it does, so those are seen through `ctime`.) Any
/// in-place write, truncate or extend moves `mtime` and `ctime`; a `chmod`, `chown`, hard link
/// or rename moves `ctime`. See the module docs for what this cannot see.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FileId {
    #[cfg(unix)]
    mtime: (i64, i64),
    #[cfg(unix)]
    ctime: (i64, i64),
}

impl FileId {
    #[cfg(unix)]
    fn of(md: &std::fs::Metadata) -> Self {
        use std::os::unix::fs::MetadataExt;
        Self {
            mtime: (md.mtime(), md.mtime_nsec()),
            ctime: (md.ctime(), md.ctime_nsec()),
        }
    }

    /// Without `fstat` timestamps there is nothing to compare (the size check still runs).
    #[cfg(not(unix))]
    fn of(_md: &std::fs::Metadata) -> Self {
        Self {}
    }

    /// Which attributes differ from `now`, for the error message.
    #[cfg(unix)]
    fn differences(&self, now: &Self) -> String {
        let mut what = Vec::new();
        if self.mtime != now.mtime {
            what.push("modification time");
        }
        if self.ctime != now.ctime {
            what.push("status-change time");
        }
        what.join(", ")
    }

    #[cfg(not(unix))]
    fn differences(&self, _now: &Self) -> String {
        String::new()
    }
}

/// An open file being read into a destination, chunk by chunk.
struct Reader<'a> {
    spec: &'a LocalRead<'a>,
    file: File,
    expected: usize,
    mode: Mode,
    /// The file as it was when it was opened.
    id: FileId,
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

#[cfg(all(unix, test))]
thread_local! {
    /// Test hook: `(chunk offset, after)`: make the O_DIRECT attempt at that chunk fail with
    /// EINVAL once, before reading it (`after == false`) or after it was read in full
    /// (`after == true`), as a mid-file alignment rule of a filesystem would.
    pub(crate) static FORCE_EINVAL: std::cell::Cell<Option<(usize, bool)>> = const { std::cell::Cell::new(None) };
    /// Test hook: how many times THIS thread fell back from O_DIRECT (the process-wide counter
    /// is shared with every parallel test).
    pub(crate) static TEST_FALLBACKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Test hook: every page-cache hint this thread applied.
    pub(crate) static HINTS_APPLIED: std::cell::RefCell<Vec<PageCacheMode>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Test hook: whether the injected EINVAL is due for the attempt at `offset`.
#[cfg(all(unix, test))]
fn inject_einval(offset: usize, after: bool) -> bool {
    FORCE_EINVAL.with(|f| {
        if f.get() == Some((offset, after)) {
            f.set(None);
            true
        } else {
            false
        }
    })
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

/// Count one `get_into` that fell back from O_DIRECT to buffered reads.
#[cfg(unix)]
fn note_fallback() {
    crate::file_store_direct::note_get_fallback();
    #[cfg(test)]
    TEST_FALLBACKS.with(|c| c.set(c.get() + 1));
}

/// Log that O_DIRECT was refused, so the read went through buffered I/O. A filesystem that
/// refuses O_DIRECT refuses it for EVERY read, so only the first occurrence in the process is a
/// WARN (the operator needs to know once); the rest are DEBUG lines. The counter
/// (`DirectIoCounters::get_fallbacks`) still ticks for every call.
#[cfg(unix)]
fn log_fallback(what: &str, path: &Path, cause: &io::Error) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    log_fallback_with(&WARNED, what, path, cause);
}

#[cfg(unix)]
fn log_fallback_with(
    warned: &std::sync::atomic::AtomicBool,
    what: &str,
    path: &Path,
    cause: &io::Error,
) {
    if !warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "O_DIRECT {what} failed for {}, reading with regular I/O: {cause} \
             (further fallbacks are logged at DEBUG; see DirectIoCounters::get_fallbacks)",
            path.display()
        );
    } else {
        tracing::debug!(
            "O_DIRECT {what} failed for {}, reading with regular I/O: {cause}",
            path.display()
        );
    }
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
        // The size was checked on the PATH; the file that was opened may be another one (the path
        // can change between `stat` and `open`). What is read is the OPEN file, so judge that.
        let opened = file
            .metadata()
            .map_err(|e| io_failure(provider, spec.path, "stat", e))?;
        if !opened.is_file() {
            return Err(not_a_regular_file(provider, spec.path));
        }
        if opened.len() != expected as u64 {
            return Err(length_mismatch(
                provider,
                spec.path,
                expected as u64,
                opened.len(),
            ));
        }
        Ok(Some(Self {
            spec,
            file,
            expected,
            mode,
            id: FileId::of(&opened),
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
                        note_fallback();
                        log_fallback("open", spec.path, &e);
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
            #[cfg(test)]
            HINTS_APPLIED.with(|h| h.borrow_mut().push(mode));
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
        first.map_err(|e| self.classify(e, span.mark()))
    }

    /// Continue with buffered reads on the SAME open file: O_DIRECT is cleared from the
    /// descriptor (`fcntl(F_SETFL)`; Linux lets a descriptor drop `O_DIRECT`). The path is never
    /// reopened, because the path may name a different file by now and the chunks already read
    /// came from this one.
    #[cfg(unix)]
    fn fall_back_to_buffered(&mut self, cause: &io::Error) -> Result<()> {
        use std::os::fd::AsRawFd;
        note_fallback();
        log_fallback("read", self.spec.path, cause);
        let fd = self.file.as_raw_fd();
        // SAFETY: `fd` is owned by `self.file`, open for the duration of both calls.
        let cleared = unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            flags >= 0
                && libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_DIRECT) >= 0
                && libc::fcntl(fd, libc::F_GETFL) & libc::O_DIRECT == 0
        };
        if !cleared {
            return Err(io_failure(
                self.spec.provider,
                self.spec.path,
                "switch to buffered reads on",
                io::Error::last_os_error(),
            ));
        }
        self.mode = Mode::Buffered;
        Ok(())
    }

    fn attempt(
        &self,
        dst: &mut [u8],
        offset: usize,
        span: &mut SpanMark<'_>,
    ) -> Result<(), ChunkError> {
        #[cfg(all(unix, test))]
        if matches!(self.mode, Mode::Direct { .. }) && inject_einval(offset, false) {
            return Err(ChunkError::Io(io::Error::from_raw_os_error(libc::EINVAL)));
        }
        let read = match self.mode {
            Mode::Buffered => self.read_buffered(dst, offset, span),
            #[cfg(unix)]
            Mode::Direct { alignment } => self.read_direct(dst, offset, alignment, span),
        };
        #[cfg(all(unix, test))]
        if read.is_ok() && matches!(self.mode, Mode::Direct { .. }) && inject_einval(offset, true) {
            return Err(ChunkError::Io(io::Error::from_raw_os_error(libc::EINVAL)));
        }
        read
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

    /// `filled`: the destination positions that were filled when the chunk failed.
    fn classify(&self, e: ChunkError, filled: u64) -> anyhow::Error {
        let spec = self.spec;
        match e {
            ChunkError::Short => {
                // The file ended early: say where, and what size it has now.
                let actual = self.file.metadata().map(|m| m.len()).unwrap_or(0);
                let expected = self.expected as u64;
                let (kind, how) = if actual == expected {
                    // It shrank during the read and was refilled to its old size since: the
                    // numbers agree, the file does not.
                    (
                        ObjectErrorKind::VersionChanged,
                        format!("and has been refilled to {actual} bytes since, so it changed during the read"),
                    )
                } else {
                    (
                        ObjectErrorKind::LengthMismatch {
                            expected,
                            actual: Some(actual),
                        },
                        format!(
                            "its size is now {actual} bytes but the destination holds {expected}"
                        ),
                    )
                };
                anyhow::Error::new(ObjectError::new(
                    kind,
                    spec.provider,
                    format!(
                        "{} ended during the read, after {filled} bytes had been read; {how}",
                        spec.path.display()
                    ),
                ))
            }
            ChunkError::Io(e) => io_failure(spec.provider, spec.path, "read", e),
            ChunkError::Progress(e) => {
                attach_object_error(e, ObjectErrorKind::Other, spec.provider, "progress")
            }
        }
    }

    /// After the last chunk the OPEN file must still be exactly `expected` bytes (else
    /// `LengthMismatch`) and still the same file version: same inode, same modification and
    /// status-change times (else `VersionChanged`: it was modified, replaced or deleted while
    /// being read, so the destination may mix versions).
    fn finish(&self) -> Result<()> {
        let spec = self.spec;
        let now = self
            .file
            .metadata()
            .map_err(|e| io_failure(spec.provider, spec.path, "stat", e))?;
        if now.len() != self.expected as u64 {
            return Err(length_mismatch(
                spec.provider,
                spec.path,
                self.expected as u64,
                now.len(),
            ));
        }
        let id = FileId::of(&now);
        if id != self.id {
            return Err(anyhow::Error::new(ObjectError::new(
                ObjectErrorKind::VersionChanged,
                spec.provider,
                format!(
                    "{} changed while it was being read ({} differ from when it was opened; the file was modified, replaced or deleted)",
                    spec.path.display(),
                    self.id.differences(&id)
                ),
            )));
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
#[path = "file_get_into_tests.rs"]
mod mid_read_tests;

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
