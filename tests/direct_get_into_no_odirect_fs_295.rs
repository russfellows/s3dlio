// tests/direct_get_into_no_odirect_fs_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267, opt-in: a direct:// `get_into` on a filesystem that REALLY refuses O_DIRECT (no
// test seam). The unit tests in src/file_get_into.rs simulate the refusal; this one runs the
// real code against a real refusing filesystem.
//
// Skipped unless S3DLIO_TEST_NO_ODIRECT_DIR names a writable directory whose filesystem refuses
// O_DIRECT. tmpfs accepts O_DIRECT since Linux 6.6, so use ramfs, for example:
//
//   sudo mkdir -p /mnt/no-odirect
//   sudo mount -t ramfs -o size=1g ramfs /mnt/no-odirect
//   sudo chown "$USER" /mnt/no-odirect
//   S3DLIO_TEST_NO_ODIRECT_DIR=/mnt/no-odirect cargo test --test direct_get_into_no_odirect_fs_295
//   sudo umount /mnt/no-odirect
//
// When the variable is set but the directory ACCEPTS O_DIRECT the test FAILS: an opted-in run
// must not pass without exercising the fallback.
//
// Every read must fall back to ordinary positional reads straight into the caller's buffer:
// same bytes, nothing written outside the destination, progress equal to the length, the
// fallback counted in `get_fallbacks`, no O_DIRECT get counted and the bounce pool never used.
// The refusal is logged at WARN once per process and at DEBUG afterwards (a flood of identical
// warnings is not a diagnostic), while the counter still ticks for every call; both are checked
// from the real log output at the end.
// The counters are process-wide, so every scenario runs in ONE test, one after another.

use s3dlio::direct_bounce::{bounce_stats, BounceStats};
use s3dlio::file_store_direct::{
    direct_io_counters, ConfigurableFileSystemObjectStore, DirectIoCounters,
};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const FILL: u8 = 0xA5;

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

/// The opted-in directory, or `None` (skip) when the variable is unset. Panics when the
/// directory accepts O_DIRECT.
fn refusing_dir() -> Option<tempfile::TempDir> {
    let Some(base) = std::env::var_os("S3DLIO_TEST_NO_ODIRECT_DIR").map(PathBuf::from) else {
        eprintln!(
            "SKIPPED: set S3DLIO_TEST_NO_ODIRECT_DIR to a directory whose filesystem refuses \
             O_DIRECT (for example a ramfs mount) to run this test"
        );
        return None;
    };
    let dir = tempfile::Builder::new()
        .prefix("no-odirect-295-")
        .tempdir_in(&base)
        .unwrap_or_else(|e| panic!("create a directory in {}: {e}", base.display()));
    let probe = dir.path().join("o_direct_probe");
    std::fs::write(&probe, [0u8; PAGE]).unwrap();
    let opened = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(&probe);
    std::fs::remove_file(&probe).unwrap();
    assert!(
        opened.is_err(),
        "S3DLIO_TEST_NO_ODIRECT_DIR={} accepts O_DIRECT, so this run would not exercise the \
         fallback; use a filesystem that refuses it (ramfs)",
        base.display()
    );
    Some(dir)
}

/// Collects everything the process logs.
#[derive(Clone, Default)]
struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn write(dir: &Path, name: &str, data: &[u8]) -> String {
    let path = dir.join(name);
    std::fs::write(&path, data).unwrap();
    format!("direct://{}", path.display())
}

struct Snap {
    io: DirectIoCounters,
    pool: BounceStats,
}

fn snap() -> Snap {
    Snap {
        io: direct_io_counters(),
        pool: bounce_stats(),
    }
}

/// Bounce pool use for GETs between two snapshots.
fn bounce_uses(a: &Snap, b: &Snap) -> u64 {
    (b.pool.leases[1] - a.pool.leases[1])
        + (b.pool.implicit_fallbacks[1] - a.pool.implicit_fallbacks[1])
        + (b.pool.fallbacks[1] - a.pool.fallbacks[1])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_filesystem_that_refuses_o_direct_is_read_buffered_into_the_caller_buffer() {
    let Some(dir) = refusing_dir() else {
        return;
    };
    let dir = dir.path();
    let log = LogBuf::default();
    {
        let sink = log.clone();
        tracing::subscriber::set_global_default(
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::DEBUG)
                .with_ansi(false)
                .with_writer(move || sink.clone())
                .finish(),
        )
        .expect("this test binary installs the only log subscriber");
    }
    let store = ConfigurableFileSystemObjectStore::with_direct_io();

    // (name, length, offset of the destination in a page-aligned backing buffer, whether the
    // file is at or above the store's minimum direct size, so the store tries O_DIRECT).
    let cases = [
        ("aligned", 4 * MIB, 0, true),
        ("aligned-odd-length", 4 * MIB + 123, 0, true),
        ("unaligned-address", 4 * MIB, 1, true),
        ("multi-chunk", 40 * MIB + 7, 0, true),
        ("below-min-io", 1000, 0, false),
    ];
    for (name, len, start, tries_direct) in cases {
        let data = pattern(len);
        let uri = write(dir, name, &data);
        let mut backing = AlignedBuf::new_zeroed(start + len + 2 * PAGE, PAGE).unwrap();
        backing.as_mut_slice().fill(FILL);
        let progress = ReceiveProgress::new();

        let before = snap();
        store
            .get_into(
                &uri,
                &mut backing.as_mut_slice()[start..start + len],
                &progress,
            )
            .await
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let after = snap();

        let all = backing.as_slice();
        assert!(all[start..start + len] == *data, "{name}: content");
        assert!(
            all[..start].iter().all(|b| *b == FILL),
            "{name}: wrote before"
        );
        assert!(
            all[start + len..].iter().all(|b| *b == FILL),
            "{name}: wrote past the destination"
        );
        assert_eq!(progress.filled(), len as u64, "{name}: progress");
        assert_eq!(
            after.io.direct_gets, before.io.direct_gets,
            "{name}: no O_DIRECT get on a refusing filesystem"
        );
        assert_eq!(
            after.io.get_fallbacks - before.io.get_fallbacks,
            u64::from(tries_direct),
            "{name}: the refusal is counted once when O_DIRECT was tried"
        );
        assert_eq!(bounce_uses(&before, &after), 0, "{name}: no bounce copy");

        let head = store
            .head(&uri)
            .await
            .unwrap()
            .expect("head of an existing file");
        assert_eq!(head.size, len as u64, "{name}: head size");
    }

    // The contract still holds on the fallback path: a file of another size is a typed
    // LengthMismatch and nothing is written.
    let uri = write(dir, "short", &pattern(5000));
    let mut dst = vec![FILL; 6000];
    let err = store
        .get_into(&uri, &mut dst, &ReceiveProgress::new())
        .await
        .unwrap_err();
    assert_eq!(
        object_error(&err).map(|e| e.kind),
        Some(ObjectErrorKind::LengthMismatch {
            expected: 6000,
            actual: Some(5000)
        })
    );
    assert!(dst.iter().all(|b| *b == FILL), "mismatch wrote nothing");

    // The log: one WARN for the whole process, a DEBUG line for each later refusal (the
    // counter, checked above, ticked for every one of them).
    check_log(&log, cases.iter().filter(|c| c.3).count());
}

// (appended to the test above through `check_log`, called at its end)
fn check_log(log: &LogBuf, direct_attempts: usize) {
    let text = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
    let refusals: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("O_DIRECT open failed"))
        .collect();
    let warns = refusals.iter().filter(|l| l.contains(" WARN ")).count();
    let debugs = refusals.iter().filter(|l| l.contains(" DEBUG ")).count();
    assert_eq!(
        warns, 1,
        "the refusal is a WARN once per process, got {warns} in {refusals:#?}"
    );
    assert_eq!(
        debugs,
        direct_attempts - 1,
        "every later refusal is a DEBUG line: {refusals:#?}"
    );
}
