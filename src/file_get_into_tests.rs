// src/file_get_into_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// In-crate tests of `file_get_into` that need its test seams (see `FORCE_EINVAL`,
// `TEST_FALLBACKS`, `HINTS_APPLIED`): a mid-read EINVAL from O_DIRECT, the identity of the file
// across a fallback, the page-cache hint, the wording of a short read, and the fallback log.
//
// Mid-read interleavings are deterministic: the future is polled by hand once (which reads the
// first chunk and yields), the file is changed, and the future is polled to the end. Every
// test runs on the thread that owns the thread-local seams.

use super::*;
use crate::memory::AlignedBuf;
use std::pin::pin;
use std::task::{Context, Poll};

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const FILL: u8 = 0xA5;
const TICK: std::time::Duration = std::time::Duration::from_millis(30);

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i % 160) + (i / 160) % 5) as u8)
        .collect()
}

fn direct_spec(path: &Path) -> LocalRead<'_> {
    LocalRead {
        path,
        provider: "direct",
        direct: Some(DirectParams {
            alignment: PAGE,
            min_io_size: PAGE,
        }),
        page_cache: None,
    }
}

fn poll_once<F: std::future::Future>(f: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    f.poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
}

fn run_to_end<F: std::future::Future>(mut f: std::pin::Pin<&mut F>) -> F::Output {
    for _ in 0..100_000 {
        if let Poll::Ready(v) = poll_once(f.as_mut()) {
            return v;
        }
    }
    panic!("the future did not finish");
}

/// A scratch directory whose filesystem accepts O_DIRECT, or `None` (skip, with a message).
fn odirect_dir() -> Option<tempfile::TempDir> {
    let base = std::env::var_os("S3DLIO_TEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::Builder::new()
        .prefix("get-into-unit-")
        .tempdir_in(base)
        .unwrap();
    let probe = dir.path().join("probe");
    std::fs::write(&probe, [0u8; PAGE]).unwrap();
    if open_direct(&probe).is_err() {
        eprintln!("SKIPPED: the scratch filesystem refuses O_DIRECT; set S3DLIO_TEST_DIR");
        return None;
    }
    Some(dir)
}

fn backing(len: usize) -> AlignedBuf {
    let mut b = AlignedBuf::new_zeroed(len + 2 * PAGE, PAGE).unwrap();
    b.as_mut_slice().fill(FILL);
    b
}

fn kind(e: &anyhow::Error) -> ObjectErrorKind {
    crate::object_error::object_error(e)
        .expect("typed error")
        .kind
}

fn fallbacks() -> u64 {
    crate::file_store_direct::direct_io_counters().get_fallbacks
}

// ---------------------------------------------------------------------------------------
// M6: an EINVAL from O_DIRECT in the MIDDLE of the read
// ---------------------------------------------------------------------------------------

/// O_DIRECT is refused for a later chunk (before it was read, and after it was read in full):
/// the rest of the read continues buffered, the bytes are exact, the fallback is counted ONCE,
/// and progress is not counted twice for the chunk that was read twice.
#[tokio::test(flavor = "current_thread")]
async fn a_mid_read_einval_falls_back_once_and_counts_progress_once() {
    let Some(dir) = odirect_dir() else { return };
    for (after, off, len) in [
        (false, 0, 40 * MIB),
        (true, 0, 40 * MIB),
        (false, 1, 40 * MIB + 4097),
        (true, 1, 40 * MIB + 4097),
    ] {
        let data = pattern(len);
        let path = dir.path().join("mid.bin");
        std::fs::write(&path, &data).unwrap();
        FORCE_EINVAL.with(|f| f.set(Some((GET_INTO_CHUNK_BYTES, after))));
        TEST_FALLBACKS.with(|c| c.set(0));
        let global_before = fallbacks();

        let mut back = backing(len);
        let progress = ReceiveProgress::new();
        get_into(
            &direct_spec(&path),
            &mut back.as_mut_slice()[off..off + len],
            &progress,
        )
        .await
        .unwrap_or_else(|e| panic!("after={after} off={off}: {e:#}"));

        let all = back.as_slice();
        assert!(
            all[off..off + len] == *data,
            "after={after} off={off}: content"
        );
        assert!(all[..off].iter().all(|b| *b == FILL));
        assert!(all[off + len..].iter().all(|b| *b == FILL));
        assert_eq!(
            progress.filled(),
            len as u64,
            "after={after} off={off}: a chunk read twice is counted once"
        );
        assert_eq!(
            TEST_FALLBACKS.with(|c| c.get()),
            1,
            "after={after} off={off}: one fallback for the whole read (it is sticky)"
        );
        assert!(
            fallbacks() > global_before,
            "the process-wide counter ticked"
        );
        assert_eq!(FORCE_EINVAL.with(|f| f.get()), None, "the injection fired");
    }
}

/// After the fallback the read continues on the SAME open file: the path is replaced in the
/// middle of the read, and the result is an error, never `Ok` with bytes from two inodes.
#[tokio::test(flavor = "current_thread")]
async fn a_fallback_never_reopens_the_path() {
    let Some(dir) = odirect_dir() else { return };
    for off in [0usize, 1] {
        let len = 40 * MIB;
        let data = pattern(len);
        let path = dir.path().join("swap.bin");
        std::fs::write(&path, &data).unwrap();
        FORCE_EINVAL.with(|f| f.set(Some((GET_INTO_CHUNK_BYTES, false))));

        let mut back = backing(len);
        let progress = ReceiveProgress::new();
        let result = {
            let dst = &mut back.as_mut_slice()[off..off + len];
            let spec = direct_spec(&path);
            let mut fut = pin!(get_into(&spec, dst, &progress));
            assert!(poll_once(fut.as_mut()).is_pending());
            std::thread::sleep(TICK);
            let other = path.with_extension("new");
            std::fs::write(&other, vec![0xEEu8; len]).unwrap();
            std::fs::rename(&other, &path).unwrap();
            run_to_end(fut.as_mut())
        };
        let err = result.expect_err("replaced mid-read: must not be Ok");
        assert_eq!(kind(&err), ObjectErrorKind::VersionChanged, "{err:#}");
        assert!(
            !back.as_slice().contains(&0xEE),
            "off={off}: not one byte of the replacement file reached the destination"
        );
        assert_eq!(
            back.as_slice()[off..off + GET_INTO_CHUNK_BYTES],
            data[..GET_INTO_CHUNK_BYTES]
        );
    }
}

// ---------------------------------------------------------------------------------------
// M17: the page-cache hint
// ---------------------------------------------------------------------------------------

/// Pages of `[offset, offset + len)` of `file` that are in the page cache (`mincore`).
fn resident_pages(file: &File, offset: usize, len: usize) -> usize {
    use std::os::fd::AsRawFd;
    // SAFETY: a read-only shared mapping of a range inside the file, unmapped before returning;
    // `mincore` writes one byte per page into `vec`, which has exactly that many.
    unsafe {
        let p = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            offset as libc::off_t,
        );
        assert_ne!(p, libc::MAP_FAILED, "mmap");
        let mut vec = vec![0u8; len.div_ceil(PAGE)];
        let rc = libc::mincore(p, len, vec.as_mut_ptr());
        libc::munmap(p, len);
        assert_eq!(rc, 0, "mincore");
        vec.iter().filter(|b| **b & 1 == 1).count()
    }
}

/// `DontNeed` is applied to the file the read opens, and it has its kernel effect: pages of the
/// file that are cached and not yet read are dropped, while a `Normal` hint leaves them alone.
#[tokio::test(flavor = "current_thread")]
async fn the_page_cache_hint_is_applied_and_takes_effect() {
    let dir = tempfile::tempdir_in(std::env::temp_dir()).unwrap();
    let len = 48 * MIB;
    let path = dir.path().join("hint.bin");
    std::fs::write(&path, pattern(len)).unwrap();
    let tail = (2 * GET_INTO_CHUNK_BYTES, GET_INTO_CHUNK_BYTES);
    let tail_pages = tail.1 / PAGE;

    for (mode, dropped) in [
        (PageCacheMode::DontNeed, true),
        (PageCacheMode::Normal, false),
    ] {
        // Make every page cached and clean (DONTNEED only drops clean pages).
        std::fs::File::open(&path).unwrap().sync_all().unwrap();
        let _ = std::fs::read(&path).unwrap();
        let probe = File::open(&path).unwrap();
        assert_eq!(
            resident_pages(&probe, tail.0, tail.1),
            tail_pages,
            "precondition: the tail is cached"
        );

        HINTS_APPLIED.with(|h| h.borrow_mut().clear());
        let spec = LocalRead {
            path: &path,
            provider: "file",
            direct: None,
            page_cache: Some(mode),
        };
        let mut dst = vec![0u8; len];
        let progress = ReceiveProgress::new();
        let mut fut = pin!(get_into(&spec, &mut dst, &progress));
        // One poll reads the first chunk only: the tail has not been touched by the read.
        assert!(poll_once(fut.as_mut()).is_pending());
        let resident = resident_pages(&probe, tail.0, tail.1);
        run_to_end(fut.as_mut()).unwrap();

        assert_eq!(HINTS_APPLIED.with(|h| h.borrow().clone()), vec![mode]);
        if dropped {
            assert!(
                resident <= 256,
                "{mode:?}: the unread tail must be dropped from the cache, {resident} of {tail_pages} pages still cached"
            );
        } else {
            assert_eq!(resident, tail_pages, "{mode:?}: the tail stays cached");
        }
    }
}

/// Which hint each store applies: the `file://` store its configured mode (`Auto` by default),
/// the configurable store none, and `direct://` none (O_DIRECT has no page cache).
#[tokio::test(flavor = "current_thread")]
async fn the_stores_apply_the_hint_they_are_configured_with() {
    use crate::file_store::{FileSystemConfig as FileConfig, FileSystemObjectStore};
    use crate::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
    use crate::object_store::ObjectStore;

    let dir = odirect_dir();
    let dir = dir
        .as_ref()
        .map(|d| d.path().to_path_buf())
        .unwrap_or_else(std::env::temp_dir);
    let path = dir.join("hint-store.bin");
    std::fs::write(&path, pattern(MIB)).unwrap();
    let file_uri = format!("file://{}", path.display());
    let direct_uri = format!("direct://{}", path.display());

    async fn hints(store: &dyn ObjectStore, uri: &str) -> Vec<PageCacheMode> {
        HINTS_APPLIED.with(|h| h.borrow_mut().clear());
        let mut dst = vec![0u8; MIB];
        store
            .get_into(uri, &mut dst, &ReceiveProgress::new())
            .await
            .unwrap();
        HINTS_APPLIED.with(|h| h.borrow().clone())
    }

    let default = FileSystemObjectStore::new();
    assert_eq!(hints(&default, &file_uri).await, vec![PageCacheMode::Auto]);

    let dont = FileSystemObjectStore::with_config(FileConfig {
        page_cache_mode: Some(PageCacheMode::DontNeed),
        ..FileConfig::default()
    });
    assert_eq!(hints(&dont, &file_uri).await, vec![PageCacheMode::DontNeed]);

    let configurable = ConfigurableFileSystemObjectStore::new(FileSystemConfig::default());
    assert_eq!(hints(&configurable, &file_uri).await, vec![]);

    let direct = ConfigurableFileSystemObjectStore::with_direct_io();
    assert_eq!(hints(&direct, &direct_uri).await, vec![]);
}

// ---------------------------------------------------------------------------------------
// the fallback is counted per call
// ---------------------------------------------------------------------------------------

/// Every call that falls back ticks the counter, whatever is logged.
#[tokio::test(flavor = "current_thread")]
async fn every_refused_open_is_counted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.bin");
    let data = pattern(MIB + 5);
    std::fs::write(&path, &data).unwrap();
    TEST_FALLBACKS.with(|c| c.set(0));
    let global_before = fallbacks();
    REFUSE_DIRECT_OPEN.with(|r| r.set(true));
    for _ in 0..3 {
        let mut dst = vec![0u8; data.len()];
        get_into(&direct_spec(&path), &mut dst, &ReceiveProgress::new())
            .await
            .unwrap();
        assert!(dst == data);
    }
    REFUSE_DIRECT_OPEN.with(|r| r.set(false));
    assert_eq!(TEST_FALLBACKS.with(|c| c.get()), 3);
    assert!(fallbacks() >= global_before + 3);
}

// ---------------------------------------------------------------------------------------
// the wording of a short read
// ---------------------------------------------------------------------------------------

/// A read that hit the end of the file reports where it ended and what the size is now, also
/// when the file has been regrown to its old size (then it is a changed file, not "N bytes,
/// expected N").
#[test]
fn a_short_read_says_where_the_file_ended() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.bin");
    let len = 3 * MIB;
    std::fs::write(&path, pattern(len)).unwrap();
    let spec = LocalRead {
        path: &path,
        provider: "file",
        direct: None,
        page_cache: None,
    };
    let reader = Reader::open(&spec, len).unwrap().expect("a reader");

    // Regrown to the same size: not a length mismatch with equal numbers.
    let err = reader.classify(ChunkError::Short, MIB as u64);
    let text = format!("{err}");
    assert!(
        text.contains("ended"),
        "the message says the file ended during the read: {text}"
    );
    assert!(
        !text.contains(&format!("is {len} bytes but the destination holds {len}")),
        "no 'N but N' message: {text}"
    );
    assert_eq!(kind(&err), ObjectErrorKind::VersionChanged);

    // Really shorter now.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(MIB as u64)
        .unwrap();
    let err = reader.classify(ChunkError::Short, MIB as u64);
    let text = format!("{err}");
    assert!(text.contains("ended"), "{text}");
    assert!(
        text.contains(&MIB.to_string()),
        "the size it has now: {text}"
    );
    assert_eq!(
        kind(&err),
        ObjectErrorKind::LengthMismatch {
            expected: len as u64,
            actual: Some(MIB as u64)
        }
    );
}

// ---------------------------------------------------------------------------------------
// the fallback log: WARN once, DEBUG after
// ---------------------------------------------------------------------------------------

#[derive(Clone, Default)]
struct LogBuf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The first refusal in a process is a WARN, every later one a DEBUG line (and the message
/// still names the file and the cause).
#[test]
fn only_the_first_fallback_is_a_warning() {
    let log = LogBuf::default();
    let sink = log.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || sink.clone())
        .finish();
    let warned = std::sync::atomic::AtomicBool::new(false);
    let cause = io::Error::from_raw_os_error(libc::EINVAL);
    tracing::subscriber::with_default(subscriber, || {
        for i in 0..5 {
            log_fallback_with(&warned, "open", Path::new(&format!("/data/f{i}")), &cause);
        }
    });
    let text = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5, "{text}");
    assert!(
        lines[0].contains(" WARN ") && lines[0].contains("/data/f0"),
        "{text}"
    );
    for (i, line) in lines.iter().enumerate().skip(1) {
        assert!(
            line.contains(" DEBUG ") && line.contains(&format!("/data/f{i}")),
            "{text}"
        );
        assert!(!line.contains(" WARN "), "{text}");
    }
}

// ---------------------------------------------------------------------------------------
// the file identity
// ---------------------------------------------------------------------------------------

/// `FileId` reads the modification time and the status-change time separately, each with its
/// nanoseconds: a `chmod` moves only `ctime`, setting `mtime` moves `mtime` (and `ctime`, as the
/// kernel always does), a write moves both, and nothing at all (a read) moves neither.
#[test]
fn file_id_reads_each_timestamp_separately() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("id.bin");
    std::fs::write(&path, pattern(PAGE)).unwrap();
    let id = |p: &Path| FileId::of(&std::fs::metadata(p).unwrap());

    let base = id(&path);
    let _ = std::fs::read(&path).unwrap(); // a read: only atime may move
    assert_eq!(
        id(&path),
        base,
        "reading changes nothing the identity looks at"
    );

    std::thread::sleep(TICK);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let chmod = id(&path);
    assert_eq!(chmod.mtime, base.mtime, "chmod leaves mtime alone");
    assert_ne!(chmod.ctime, base.ctime, "chmod moves ctime");
    assert_eq!(base.differences(&chmod), "status-change time");

    std::thread::sleep(TICK);
    let stamp = std::time::UNIX_EPOCH + std::time::Duration::new(1_000_000_000, 123_456_789);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(stamp)
        .unwrap();
    let touched = id(&path);
    assert_eq!(
        touched.mtime,
        (1_000_000_000, 123_456_789),
        "mtime with its nanoseconds"
    );
    assert_ne!(touched.ctime, chmod.ctime, "setting mtime moves ctime");
    assert!(chmod.differences(&touched).contains("modification time"));

    std::thread::sleep(TICK);
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b"x").unwrap();
    }
    let written = id(&path);
    assert_ne!(written.mtime, touched.mtime);
    assert_ne!(written.ctime, touched.ctime);
    assert_eq!(
        touched.differences(&written),
        "modification time, status-change time"
    );
}
