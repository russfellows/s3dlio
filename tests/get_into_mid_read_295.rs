// tests/get_into_mid_read_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: what a file:// / direct:// `get_into` does when the FILE CHANGES WHILE IT IS
// BEING READ. Every case is deterministic: the future is polled by hand exactly once (that
// reads the first 16 MiB chunk and yields, see `common/step.rs`), the test changes the file,
// and the future is polled to the end.
//
// The contract (docs/Changelog.md, docs/PYTHON_API_GUIDE.md):
//
//   * the file GREW or SHRANK                 -> LengthMismatch, with the size it has now;
//   * the same size but it was modified in
//     place / truncated and regrown / replaced
//     / unlinked (mtime, ctime, inode or device
//     of the OPEN file changed)               -> VersionChanged;
//   * nothing changed (a concurrent reader,
//     an access-time update)                  -> Ok, byte-exact.
//
// `Ok(())` therefore never means "mixed data from two versions of the file". Nothing is ever
// written outside the destination (canaries around it prove that).

#[path = "common/odirect.rs"]
mod odirect;
#[path = "common/step.rs"]
mod step;

use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use s3dlio::FileSystemObjectStore;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::time::Duration;

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const FILL: u8 = 0xA5;
/// The read chunk of `get_into` (the first poll reads this much).
const CHUNK: usize = 16 * MIB;
/// Long enough for the kernel's coarse file timestamps to move on (they tick every 1-10 ms), so
/// a change made after the first chunk is a change the timestamps can show.
const TICK: Duration = Duration::from_millis(30);

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

/// Where the destination starts (bytes past a page boundary) and how long the file is.
#[derive(Clone, Copy, Debug)]
struct Shape {
    off: usize,
    len: usize,
}

/// A page-aligned destination over an aligned length, and an unaligned one over an unaligned
/// length (the bounce-pool path of `direct://`).
const SHAPES: [Shape; 2] = [
    Shape {
        off: 0,
        len: 40 * MIB,
    },
    Shape {
        off: 1,
        len: 40 * MIB + 4097,
    },
];

struct Case {
    name: &'static str,
    store: Box<dyn ObjectStore>,
    scheme: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "file (FileSystemObjectStore)",
            store: Box::new(FileSystemObjectStore::new()),
            scheme: "file",
        },
        Case {
            name: "file (ConfigurableFileSystemObjectStore)",
            store: Box::new(ConfigurableFileSystemObjectStore::new(
                FileSystemConfig::default(),
            )),
            scheme: "file",
        },
        Case {
            name: "direct",
            store: Box::new(ConfigurableFileSystemObjectStore::with_direct_io()),
            scheme: "direct",
        },
    ]
}

fn kind(e: &anyhow::Error) -> ObjectErrorKind {
    object_error(e)
        .unwrap_or_else(|| panic!("an untyped error: {e:#}"))
        .kind
}

struct Outcome {
    result: anyhow::Result<()>,
    filled: u64,
    backing: AlignedBuf,
    shape: Shape,
}

impl Outcome {
    fn dst(&self) -> &[u8] {
        &self.backing.as_slice()[self.shape.off..self.shape.off + self.shape.len]
    }
    /// Nothing outside the destination was written.
    fn assert_untouched_around(&self) {
        let all = self.backing.as_slice();
        assert!(
            all[..self.shape.off].iter().all(|b| *b == FILL),
            "wrote before the destination"
        );
        assert!(
            all[self.shape.off + self.shape.len..]
                .iter()
                .all(|b| *b == FILL),
            "wrote past the destination"
        );
    }
}

/// Create the file, read it with `store`, and run `mutate` between the first chunk and the rest.
fn read_with(
    case: &Case,
    dir: &Path,
    shape: Shape,
    tag: &str,
    mutate: impl FnOnce(&Path),
) -> (Outcome, PathBuf, Vec<u8>) {
    assert!(shape.len > CHUNK, "the read must need more than one chunk");
    let data = pattern(shape.len);
    let path = dir.join(format!("{tag}-{}-{}.bin", case.scheme, shape.off));
    std::fs::write(&path, &data).unwrap();
    let uri = format!("{}://{}", case.scheme, path.display());

    let mut backing = AlignedBuf::new_zeroed(shape.len + 2 * PAGE, PAGE).expect("aligned buffer");
    backing.as_mut_slice().fill(FILL);
    let progress = ReceiveProgress::new();
    let result = {
        let dst = &mut backing.as_mut_slice()[shape.off..shape.off + shape.len];
        let mut fut = pin!(case.store.get_into(&uri, dst, &progress));
        assert!(
            step::poll_once(fut.as_mut()).is_pending(),
            "{}: the first poll must read one chunk and yield",
            case.name
        );
        std::thread::sleep(TICK);
        mutate(&path);
        step::run_to_end(fut.as_mut())
    };
    let outcome = Outcome {
        result,
        filled: progress.filled(),
        backing,
        shape,
    };
    outcome.assert_untouched_around();
    (outcome, path, data)
}

fn each_case(scenario: &str, mut f: impl FnMut(&Case, &Path, Shape)) {
    let Some(scratch) = odirect::odirect_scratch_dir("get-into-mid-read-295-") else {
        return;
    };
    for case in cases() {
        for shape in SHAPES {
            eprintln!("{scenario}: {} {shape:?}", case.name);
            f(&case, scratch.path(), shape);
        }
    }
}

fn pwrite(path: &Path, at: u64, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(at)).unwrap();
    f.write_all(bytes).unwrap();
}

// ---------------------------------------------------------------------------------------
// the file changes SIZE
// ---------------------------------------------------------------------------------------

/// The file GREW after the first chunk: LengthMismatch, with the new size. (A read that
/// trusted its start-of-read size would return Ok with the first `len` bytes.)
#[test]
fn a_file_that_grows_during_the_read_is_a_length_mismatch() {
    each_case("grow", |case, dir, shape| {
        let (out, ..) = read_with(case, dir, shape, "grow", |p| {
            let mut f = std::fs::OpenOptions::new().append(true).open(p).unwrap();
            f.write_all(&[7u8; 4096]).unwrap();
        });
        let err = out.result.expect_err("grew: must not be Ok");
        assert_eq!(
            kind(&err),
            ObjectErrorKind::LengthMismatch {
                expected: shape.len as u64,
                actual: Some(shape.len as u64 + 4096)
            },
            "{}: {err:#}",
            case.name
        );
        assert_eq!(out.filled, shape.len as u64, "it read what it was asked to");
    });
}

/// The file SHRANK after the first chunk (to a chunk boundary, mid-chunk, and to nothing): the
/// next read hits the end of the file: LengthMismatch with the size it has now, in both schemes.
#[test]
fn a_file_that_shrinks_during_the_read_is_a_length_mismatch() {
    for new_len in [
        CHUNK as u64,
        CHUNK as u64 + 5 * 1024 + 7,
        20 * MIB as u64,
        0,
    ] {
        each_case(&format!("shrink to {new_len}"), |case, dir, shape| {
            let (out, ..) = read_with(case, dir, shape, "shrink", |p| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(p)
                    .unwrap()
                    .set_len(new_len)
                    .unwrap();
            });
            let err = out.result.expect_err("shrank: must not be Ok");
            assert_eq!(
                kind(&err),
                ObjectErrorKind::LengthMismatch {
                    expected: shape.len as u64,
                    actual: Some(new_len)
                },
                "{} (to {new_len}): {err:#}",
                case.name
            );
            assert!(
                out.filled >= CHUNK as u64 && out.filled <= new_len.max(CHUNK as u64),
                "progress counts only what was really read: {} of {new_len}",
                out.filled
            );
        });
    }
}

// ---------------------------------------------------------------------------------------
// the file keeps its SIZE but changes
// ---------------------------------------------------------------------------------------

/// Truncate to nothing and regrow to the same size: the file is the right size again (and
/// zero-filled), so only the timestamps can tell. Never `Ok` with garbage.
#[test]
fn truncate_then_regrow_to_the_same_size_is_version_changed() {
    each_case("truncate+regrow", |case, dir, shape| {
        let (out, ..) = read_with(case, dir, shape, "regrow", |p| {
            let f = std::fs::OpenOptions::new().write(true).open(p).unwrap();
            f.set_len(0).unwrap();
            f.set_len(shape.len as u64).unwrap();
        });
        let err = out.result.expect_err("regrown: must not be Ok");
        assert_eq!(
            kind(&err),
            ObjectErrorKind::VersionChanged,
            "{}: {err:#}",
            case.name
        );
    });
}

/// A same-size overwrite IN PLACE, in the part not yet read and in the part already read.
#[test]
fn a_same_size_overwrite_in_place_is_version_changed() {
    for at in [30 * MIB as u64, MIB as u64] {
        each_case(&format!("overwrite at {at}"), |case, dir, shape| {
            let (out, ..) = read_with(case, dir, shape, "overwrite", |p| {
                pwrite(p, at, &vec![0xEEu8; 64 * 1024]);
            });
            let err = out.result.expect_err("overwritten: must not be Ok");
            assert_eq!(
                kind(&err),
                ObjectErrorKind::VersionChanged,
                "{} (at {at}): {err:#}",
                case.name
            );
        });
    }
}

/// The path is replaced by a different file of the same size (rename over it): the read has the
/// OLD inode open, which is unlinked by the rename, so its status changed: VersionChanged. The
/// bytes in the destination are all from the old file or the call fails; never a mix.
#[test]
fn replacing_the_file_by_rename_is_version_changed() {
    each_case("rename-replace", |case, dir, shape| {
        let (out, ..) = read_with(case, dir, shape, "replace", |p| {
            let other = p.with_extension("new");
            std::fs::write(&other, vec![0xEEu8; shape.len]).unwrap();
            std::fs::rename(&other, p).unwrap();
        });
        let err = out.result.expect_err("replaced: must not be Ok");
        assert_eq!(
            kind(&err),
            ObjectErrorKind::VersionChanged,
            "{}: {err:#}",
            case.name
        );
    });
}

/// The file is deleted mid-read: the open descriptor keeps it readable, but it is gone as an
/// object: VersionChanged ("changed or disappeared").
#[test]
fn deleting_the_file_during_the_read_is_version_changed() {
    each_case("unlink", |case, dir, shape| {
        let (out, ..) = read_with(case, dir, shape, "unlink", |p| {
            std::fs::remove_file(p).unwrap();
        });
        let err = out.result.expect_err("deleted: must not be Ok");
        assert_eq!(
            kind(&err),
            ObjectErrorKind::VersionChanged,
            "{}: {err:#}",
            case.name
        );
    });
}

/// A metadata-only change (chmod) moves the file's ctime. That is reported too: the check is
/// deliberately conservative (it cannot tell a chmod from a write by timestamps alone), and the
/// caller may simply retry.
#[test]
fn a_metadata_change_during_the_read_is_reported_conservatively() {
    use std::os::unix::fs::PermissionsExt;
    each_case("chmod", |case, dir, shape| {
        let (out, ..) = read_with(case, dir, shape, "chmod", |p| {
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600)).unwrap();
        });
        let err = out.result.expect_err("ctime moved: reported");
        assert_eq!(
            kind(&err),
            ObjectErrorKind::VersionChanged,
            "{}: {err:#}",
            case.name
        );
    });
}

// ---------------------------------------------------------------------------------------
// controls: no change, no error
// ---------------------------------------------------------------------------------------

/// A concurrent READER of the same file (access time moves, nothing else) is not a change: the
/// read succeeds, byte-exact, with every position counted once.
#[test]
fn a_concurrent_reader_is_not_a_change() {
    each_case("concurrent reader", |case, dir, shape| {
        let (out, _, data) = read_with(case, dir, shape, "reader", |p| {
            let again = std::fs::read(p).unwrap();
            assert_eq!(again.len(), shape.len);
            let _ = std::fs::metadata(p).unwrap();
        });
        if let Err(e) = &out.result {
            panic!("{}: {e:#}", case.name);
        }
        assert!(out.dst() == data.as_slice(), "{}: content", case.name);
        assert_eq!(out.filled, shape.len as u64);
    });
}

/// With no change at all, the same harness succeeds (the harness itself is not what fails).
#[test]
fn an_undisturbed_read_succeeds() {
    each_case("undisturbed", |case, dir, shape| {
        let (out, _, data) = read_with(case, dir, shape, "calm", |_| {});
        if let Err(e) = &out.result {
            panic!("{}: {e:#}", case.name);
        }
        assert!(out.dst() == data.as_slice(), "{}: content", case.name);
        assert_eq!(out.filled, shape.len as u64);
    });
}
