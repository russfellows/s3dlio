// tests/get_into_local_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: the caller-buffer GET (`ObjectStore::get_into`) and the typed HEAD
// (`ObjectStore::head`) for `file://` and `direct://`, through every store type that serves
// those schemes, on a multi-thread AND a current-thread runtime.
//
// The contract is the one #264 gives the network stores: `dst.len()` is the expected length;
// `Ok(())` means exactly that many bytes were written AND the file has exactly that size; a
// longer or shorter file is `LengthMismatch` and nothing is written past `dst` (canaries in an
// outer buffer prove it); a missing file is `NotFound`; progress reports the distinct
// destination positions filled; a dropped future stops every writer.

#[path = "common/odirect.rs"]
mod odirect;

use bytes::Bytes;
use futures::FutureExt;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{
    direct_io_store_for_uri, store_for_uri, HeadMetadata, ObjectStore, ReceiveProgress,
};
use s3dlio::FileSystemObjectStore;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const MIB: usize = 1024 * 1024;
const FILL: u8 = 0xA5;
const CANARY: u8 = 0x5A;
const CANARY_LEN: usize = 64;

/// Content that never contains [`FILL`] or [`CANARY`] and has a long period, so a shifted or
/// repeated chunk is noticed.
fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

/// A destination of `len` bytes (all [`FILL`]) followed by canaries, in ONE allocation, so a
/// write past the destination is caught.
struct Dst {
    buf: Vec<u8>,
    len: usize,
}

impl Dst {
    fn new(len: usize) -> Self {
        let mut buf = vec![FILL; len];
        buf.extend(std::iter::repeat_n(CANARY, CANARY_LEN));
        Self { buf, len }
    }
    fn slice(&mut self) -> &mut [u8] {
        &mut self.buf[..self.len]
    }
    fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
    fn assert_canaries(&self) {
        assert!(
            self.buf[self.len..].iter().all(|b| *b == CANARY),
            "bytes past the destination were written"
        );
    }
    fn assert_untouched(&self) {
        assert!(
            self.bytes().iter().all(|b| *b == FILL),
            "the destination was written"
        );
        self.assert_canaries();
    }
}

fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
    object_error(err)
        .unwrap_or_else(|| panic!("an untyped error: {err:#}"))
        .kind
}

/// One store under test and the URI scheme its objects are addressed with.
struct Subject {
    label: &'static str,
    scheme: &'static str,
    store: Arc<dyn ObjectStore>,
}

fn subjects() -> Vec<Subject> {
    let boxed = |label, scheme, b: Box<dyn ObjectStore>| Subject {
        label,
        scheme,
        store: Arc::from(b),
    };
    vec![
        boxed(
            "FileSystemObjectStore",
            "file",
            Box::new(FileSystemObjectStore::new()),
        ),
        boxed(
            "store_for_uri(file)",
            "file",
            store_for_uri("file:///x").unwrap(),
        ),
        boxed(
            "Configurable(buffered) on file://",
            "file",
            Box::new(ConfigurableFileSystemObjectStore::new(
                FileSystemConfig::default(),
            )),
        ),
        boxed(
            "direct_io_store_for_uri(file)",
            "file",
            direct_io_store_for_uri("file:///x").unwrap(),
        ),
        boxed(
            "store_for_uri(direct)",
            "direct",
            store_for_uri("direct:///x").unwrap(),
        ),
        boxed(
            "Configurable(direct_io)",
            "direct",
            Box::new(ConfigurableFileSystemObjectStore::with_direct_io()),
        ),
        boxed(
            "Configurable(buffered) on direct://",
            "direct",
            Box::new(ConfigurableFileSystemObjectStore::new(
                FileSystemConfig::default(),
            )),
        ),
        boxed(
            "Configurable(high_performance)",
            "direct",
            Box::new(ConfigurableFileSystemObjectStore::high_performance()),
        ),
    ]
}

fn uri(scheme: &str, path: &Path) -> String {
    format!("{scheme}://{}", path.display())
}

fn write(dir: &Path, name: &str, data: &[u8]) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, data).unwrap();
    path
}

/// An exact-length file is read whole into the caller buffer, at many sizes around the
/// alignment and the internal chunk size.
async fn exact_length(s: &Subject, dir: &Path) {
    for (i, len) in [
        1,
        511,
        4095,
        4096,
        4097,
        64 * 1024,
        MIB,
        3 * MIB + 17,
        40 * MIB + 4097,
    ]
    .into_iter()
    .enumerate()
    {
        let data = pattern(len);
        let path = write(dir, &format!("exact-{i}"), &data);
        let mut dst = Dst::new(len);
        let progress = ReceiveProgress::new();
        s.store
            .get_into(&uri(s.scheme, &path), dst.slice(), &progress)
            .await
            .unwrap_or_else(|e| panic!("{}: len {len}: {e:#}", s.label));
        assert!(
            dst.bytes() == data.as_slice(),
            "{}: content, len {len}",
            s.label
        );
        dst.assert_canaries();
        assert_eq!(
            progress.filled(),
            len as u64,
            "{}: filled, len {len}",
            s.label
        );
        assert!(progress.first_byte().is_some(), "{}: first byte", s.label);
    }
}

/// A file shorter than the destination is a `LengthMismatch` that names both sizes; nothing is
/// written and progress says so.
async fn shorter_file(s: &Subject, dir: &Path) {
    let path = write(dir, "short", &pattern(100));
    let mut dst = Dst::new(200);
    let progress = ReceiveProgress::new();
    let err = s
        .store
        .get_into(&uri(s.scheme, &path), dst.slice(), &progress)
        .await
        .expect_err("a short file must fail");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 200,
            actual: Some(100)
        },
        "{}: {err:#}",
        s.label
    );
    dst.assert_untouched();
    assert_eq!(progress.filled(), 0, "{}: nothing was filled", s.label);
}

/// A file longer than the destination is a `LengthMismatch`; not one byte lands past `dst`.
async fn longer_file(s: &Subject, dir: &Path) {
    for len in [200 + 1, 4096 * 3 + 5, 20 * MIB] {
        let path = write(dir, &format!("long-{len}"), &pattern(len));
        let mut dst = Dst::new(200);
        let progress = ReceiveProgress::new();
        let err = s
            .store
            .get_into(&uri(s.scheme, &path), dst.slice(), &progress)
            .await
            .expect_err("a long file must fail");
        assert_eq!(
            kind_of(&err),
            ObjectErrorKind::LengthMismatch {
                expected: 200,
                actual: Some(len as u64)
            },
            "{}: {err:#}",
            s.label
        );
        dst.assert_untouched();
        assert_eq!(progress.filled(), 0, "{}", s.label);
    }
}

/// Zero-length objects: an empty destination reads an empty file (success, nothing filled) and
/// is a mismatch against a non-empty one; a non-empty destination is a mismatch against an
/// empty file.
async fn zero_length(s: &Subject, dir: &Path) {
    let empty = uri(s.scheme, &write(dir, "empty", b""));
    let progress = ReceiveProgress::new();
    s.store
        .get_into(&empty, &mut [], &progress)
        .await
        .unwrap_or_else(|e| panic!("{}: {e:#}", s.label));
    assert_eq!(progress.filled(), 0);
    assert!(progress.first_byte().is_none(), "no byte, no first byte");

    let mut dst = Dst::new(8);
    let err = s
        .store
        .get_into(&empty, dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("empty file into a non-empty destination");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 8,
            actual: Some(0)
        },
        "{}",
        s.label
    );
    dst.assert_untouched();

    let full = uri(s.scheme, &write(dir, "nonempty", &pattern(33)));
    let err = s
        .store
        .get_into(&full, &mut [], &ReceiveProgress::new())
        .await
        .expect_err("non-empty file into an empty destination");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: Some(33)
        },
        "{}",
        s.label
    );
}

/// Absence: `get_into` is `NotFound`, `head` is `Ok(None)`; also when a parent is a file.
async fn missing(s: &Subject, dir: &Path) {
    let gone = uri(s.scheme, &dir.join("no-such-file"));
    let mut dst = Dst::new(16);
    let err = s
        .store
        .get_into(&gone, dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("missing");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::NotFound,
        "{}: {err:#}",
        s.label
    );
    dst.assert_untouched();
    assert_eq!(s.store.head(&gone).await.unwrap(), None, "{}", s.label);

    let parent = write(dir, "a-file", b"x");
    let below = uri(s.scheme, &parent.join("child"));
    let err = s
        .store
        .get_into(&below, dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("a path below a file");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::NotFound,
        "{}: {err:#}",
        s.label
    );
    assert_eq!(s.store.head(&below).await.unwrap(), None, "{}", s.label);
}

/// A directory is an error (never `Ok(None)`, never `NotFound`), and nothing is written.
async fn directory(s: &Subject, dir: &Path) {
    let sub = dir.join("a-directory");
    std::fs::create_dir_all(&sub).unwrap();
    let target = uri(s.scheme, &sub);
    let mut dst = Dst::new(16);
    let err = s
        .store
        .get_into(&target, dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("a directory");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::Other,
        "{}: {err:#}",
        s.label
    );
    dst.assert_untouched();
    let err = s
        .store
        .head(&target)
        .await
        .expect_err("head of a directory");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::Other,
        "{}: {err:#}",
        s.label
    );
}

/// A file the process may not read is `Forbidden` for the read and for head.
async fn forbidden(s: &Subject, dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        eprintln!("SKIPPED: root ignores file permissions");
        return;
    }
    let path = write(dir, "secret", &pattern(64));
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
    let target = uri(s.scheme, &path);
    let mut dst = Dst::new(64);
    let err = s
        .store
        .get_into(&target, dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("unreadable");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::Forbidden,
        "{}: {err:#}",
        s.label
    );
    dst.assert_untouched();
    // HEAD only needs the directory entry, which stat can read: it succeeds.
    let head = s.store.head(&target).await.unwrap();
    assert_eq!(head.map(|h| h.size), Some(64), "{}", s.label);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
}

/// A directory the process may not search turns head into a `Forbidden` error.
async fn forbidden_directory_head(s: &Subject, dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let sealed = dir.join("sealed");
    std::fs::create_dir_all(&sealed).unwrap();
    let path = write(&sealed, "inner", b"x");
    std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o000)).unwrap();
    let target = uri(s.scheme, &path);
    let err = s.store.head(&target).await.expect_err("unsearchable");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::Forbidden,
        "{}: {err:#}",
        s.label
    );
    let err = s
        .store
        .get_into(&target, &mut [0u8; 1], &ReceiveProgress::new())
        .await
        .expect_err("unsearchable");
    assert_eq!(
        kind_of(&err),
        ObjectErrorKind::Forbidden,
        "{}: {err:#}",
        s.label
    );
    std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// HEAD: the size, an identity that changes with the content, no version.
async fn head_metadata(s: &Subject, dir: &Path) {
    let path = write(dir, "head-me", &pattern(12_345));
    let target = uri(s.scheme, &path);
    let found: HeadMetadata = s.store.head(&target).await.unwrap().expect("present");
    assert_eq!(found.size, 12_345, "{}", s.label);
    assert_eq!(found.version_id, None, "{}", s.label);
    let etag = found.etag.clone().expect("a file has a stat-derived etag");
    assert!(!etag.is_empty());
    assert_eq!(
        s.store.head(&target).await.unwrap(),
        Some(found.clone()),
        "stable"
    );
    // The same identity `stat` reports.
    let stat = s.store.stat(&target).await.unwrap();
    assert_eq!(stat.size, 12_345);
    assert_eq!(stat.e_tag.as_deref(), Some(etag.as_str()), "{}", s.label);

    let empty = write(dir, "head-empty", b"");
    let found = s.store.head(&uri(s.scheme, &empty)).await.unwrap().unwrap();
    assert_eq!(found.size, 0, "{}", s.label);
}

/// A URI of the wrong scheme is an error before anything is touched.
async fn wrong_scheme(s: &Subject) {
    let mut dst = Dst::new(4);
    s.store
        .get_into("s3://bucket/key", dst.slice(), &ReceiveProgress::new())
        .await
        .expect_err("wrong scheme");
    s.store
        .head("s3://bucket/key")
        .await
        .expect_err("wrong scheme");
    dst.assert_untouched();
}

/// Reading an object leaves `get` and `get_into` agreeing.
async fn agrees_with_get(s: &Subject, dir: &Path) {
    let data = pattern(5 * MIB + 3);
    let path = write(dir, "agree", &data);
    let target = uri(s.scheme, &path);
    let got: Bytes = s.store.get(&target).await.unwrap();
    let mut dst = Dst::new(data.len());
    s.store
        .get_into(&target, dst.slice(), &ReceiveProgress::new())
        .await
        .unwrap();
    assert_eq!(got.as_ref(), dst.bytes());
}

/// Dropping the future after its first poll stops every writer: what was written stays as it
/// was, progress stops moving, and the buffer can be reused for a complete read.
async fn dropped_future_writes_nothing_more(s: &Subject, dir: &Path) {
    let len = 64 * MIB;
    let data = pattern(len);
    let path = write(dir, "cancel-me", &data);
    let target = uri(s.scheme, &path);
    let mut dst = Dst::new(len);
    let progress = ReceiveProgress::new();
    {
        let fut = s.store.get_into(&target, dst.slice(), &progress);
        // One poll: the read starts, makes progress and yields, still unfinished. `now_or_never`
        // then drops the future, which is the cancellation.
        assert!(
            fut.now_or_never().is_none(),
            "{}: a {len}-byte read must still be in flight after one poll",
            s.label
        );
    }
    let filled = progress.filled();
    assert!(
        filled > 0 && filled < len as u64,
        "{}: partial progress expected, got {filled}",
        s.label
    );
    let snapshot = dst.buf.clone();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        dst.buf == snapshot,
        "{}: a write happened after the drop",
        s.label
    );
    assert_eq!(
        progress.filled(),
        filled,
        "{}: progress moved after the drop",
        s.label
    );
    dst.assert_canaries();
    // The filled prefix is real data.
    assert!(
        dst.bytes()[..filled as usize] == data[..filled as usize],
        "{}",
        s.label
    );

    // The buffer is free again: a full read into it succeeds.
    let again = ReceiveProgress::new();
    s.store
        .get_into(&target, dst.slice(), &again)
        .await
        .unwrap_or_else(|e| panic!("{}: {e:#}", s.label));
    assert!(dst.bytes() == data.as_slice(), "{}", s.label);
    assert_eq!(again.filled(), len as u64);
}

/// Aborting a spawned task that is mid-read: once the abort is joined the buffer never changes
/// again (the future, and with it every writer, is gone).
async fn aborted_task_writes_nothing_more(s: &Subject, dir: &Path) {
    let len = 256 * MIB;
    let path = write(dir, "abort-me", &pattern(len));
    let target = uri(s.scheme, &path);
    // The task borrows the buffer for 'static; it is reclaimed after the task is joined.
    let buf: &'static mut [u8] = Box::leak(vec![FILL; len].into_boxed_slice());
    let base = buf.as_mut_ptr() as usize;
    let progress = Arc::new(ReceiveProgress::new());
    let store = s.store.clone();
    let p2 = progress.clone();
    let handle = tokio::spawn(async move { store.get_into(&target, buf, &p2).await });
    while progress.filled() == 0 && !handle.is_finished() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    handle.abort();
    let _ = handle.await;
    // SAFETY: the task is joined, so its future (the only other user of the buffer) is gone.
    let view = unsafe { std::slice::from_raw_parts(base as *const u8, len) };
    let checksum = |v: &[u8]| v.iter().step_by(4093).map(|b| *b as u64).sum::<u64>();
    let filled = progress.filled();
    assert!(
        filled > 0,
        "{}: the read never made progress before the abort",
        s.label
    );
    let before = checksum(view);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        checksum(view),
        before,
        "{}: written after the abort",
        s.label
    );
    assert_eq!(progress.filled(), filled, "{}", s.label);
    // SAFETY: as above; this reclaims the leaked allocation.
    drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(base as *mut u8, len)) });
}

/// Run every scenario for every store.
async fn everything() {
    let Some(dir) = odirect::odirect_scratch_dir("get-into-295-") else {
        return;
    };
    for s in subjects() {
        let sub = dir
            .path()
            .join(s.label.replace(|c: char| !c.is_alphanumeric(), "_"));
        std::fs::create_dir_all(&sub).unwrap();
        exact_length(&s, &sub).await;
        shorter_file(&s, &sub).await;
        longer_file(&s, &sub).await;
        zero_length(&s, &sub).await;
        missing(&s, &sub).await;
        directory(&s, &sub).await;
        forbidden(&s, &sub).await;
        forbidden_directory_head(&s, &sub).await;
        head_metadata(&s, &sub).await;
        wrong_scheme(&s).await;
        agrees_with_get(&s, &sub).await;
        dropped_future_writes_nothing_more(&s, &sub).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_scenario_on_a_multi_thread_runtime() {
    everything().await;
}

/// `block_in_place` must not panic here: on a current-thread runtime the read runs inline.
#[tokio::test(flavor = "current_thread")]
async fn every_scenario_on_a_current_thread_runtime() {
    everything().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_aborted_task_stops_writing() {
    let Some(dir) = odirect::odirect_scratch_dir("get-into-295-abort-") else {
        return;
    };
    for s in subjects() {
        let sub = dir
            .path()
            .join(s.label.replace(|c: char| !c.is_alphanumeric(), "_"));
        std::fs::create_dir_all(&sub).unwrap();
        aborted_task_writes_nothing_more(&s, &sub).await;
    }
}

/// The capability report matches what the stores do.
#[tokio::test]
async fn capabilities_say_true_for_file_and_direct() {
    for (u, provider) in [("file:///tmp/x", "file"), ("direct:///tmp/x", "direct")] {
        let c = s3dlio::capabilities::object_capabilities(u).unwrap();
        assert_eq!(c.provider, provider);
        assert!(c.get_into, "{u}: get_into");
        assert!(c.head, "{u}: head");
    }
}
