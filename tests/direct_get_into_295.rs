// tests/direct_get_into_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: which PATH a direct:// `get_into` takes, proved with the process-wide counters.
//
// * A destination whose address AND length are multiples of the store alignment is read with
//   O_DIRECT straight into the caller's buffer: `direct_gets` ticks, the bounce pool is never
//   leased, no private bounce buffer is allocated.
// * An unaligned address or an unaligned length goes through the shared bounce pool (the
//   documented, allowed copy): the pool is leased (or a private bounce buffer is counted).
// * A file below the O_DIRECT threshold and a store with direct I/O off read buffered.
//
// The counters are process-wide, so every scenario runs in ONE test, one after another.

#[path = "common/odirect.rs"]
mod odirect;

use s3dlio::direct_bounce::{bounce_stats, BounceStats};
use s3dlio::file_store_direct::{
    direct_io_counters, ConfigurableFileSystemObjectStore, DirectIoCounters, FileSystemConfig,
};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use std::path::Path;

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const FILL: u8 = 0xA5;

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

fn write(dir: &Path, name: &str, data: &[u8]) -> String {
    let path = dir.join(name);
    std::fs::write(&path, data).unwrap();
    format!("direct://{}", path.display())
}

/// A page-aligned backing buffer of `len` bytes, all [`FILL`].
fn backing(len: usize) -> AlignedBuf {
    let mut buf = AlignedBuf::new_zeroed(len, PAGE).expect("aligned buffer");
    buf.as_mut_slice().fill(FILL);
    buf
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

/// Bounce pool use for GETs between two snapshots: pooled leases plus private (implicit or
/// busy-fallback) bounce buffers.
fn bounce_uses(a: &Snap, b: &Snap) -> u64 {
    (b.pool.leases[1] - a.pool.leases[1])
        + (b.pool.implicit_fallbacks[1] - a.pool.implicit_fallbacks[1])
        + (b.pool.fallbacks[1] - a.pool.fallbacks[1])
}

/// Read `data.len()` bytes of `uri` into `backing[start..]` and check content, the canary
/// fill before and after, and progress.
async fn read_and_check(
    store: &ConfigurableFileSystemObjectStore,
    uri: &str,
    data: &[u8],
    backing: &mut AlignedBuf,
    start: usize,
) {
    let len = data.len();
    backing.as_mut_slice().fill(FILL);
    let progress = ReceiveProgress::new();
    store
        .get_into(
            uri,
            &mut backing.as_mut_slice()[start..start + len],
            &progress,
        )
        .await
        .unwrap_or_else(|e| panic!("{uri}: {e:#}"));
    let all = backing.as_slice();
    assert!(all[start..start + len] == *data, "content of {uri}");
    assert!(
        all[..start].iter().all(|b| *b == FILL),
        "wrote before the destination"
    );
    assert!(
        all[start + len..].iter().all(|b| *b == FILL),
        "wrote past the destination"
    );
    assert_eq!(progress.filled(), len as u64);
    assert!(progress.first_byte().is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_path_taken_is_the_one_the_alignment_says() {
    let Some(dir) = odirect::odirect_scratch_dir("direct-get-into-295-") else {
        return;
    };
    let dir = dir.path();
    let store = ConfigurableFileSystemObjectStore::with_direct_io();

    // ---- aligned address and length: O_DIRECT into the caller buffer, no bounce ----------
    for (name, len) in [
        ("a-small", PAGE),
        ("a-8m", 8 * MIB),
        ("a-multi-chunk", 40 * MIB),
    ] {
        let data = pattern(len);
        let uri = write(dir, name, &data);
        let mut b = backing(len + 2 * PAGE);
        let before = snap();
        read_and_check(&store, &uri, &data, &mut b, PAGE).await;
        let after = snap();
        assert_eq!(
            after.io.direct_gets - before.io.direct_gets,
            1,
            "{name}: O_DIRECT read"
        );
        assert_eq!(
            after.io.get_fallbacks, before.io.get_fallbacks,
            "{name}: no buffered fallback"
        );
        assert_eq!(
            bounce_uses(&before, &after),
            0,
            "{name}: aligned means no bounce copy"
        );
    }

    // ---- unaligned ADDRESS (aligned length): bounce ---------------------------------------
    for (name, len, offset) in [
        ("u-addr-8m", 8 * MIB, 8),
        ("u-addr-multi", 40 * MIB, PAGE / 2),
    ] {
        let data = pattern(len);
        let uri = write(dir, name, &data);
        let mut b = backing(len + 2 * PAGE);
        let before = snap();
        read_and_check(&store, &uri, &data, &mut b, offset).await;
        let after = snap();
        assert_eq!(after.io.direct_gets - before.io.direct_gets, 1, "{name}");
        assert!(
            bounce_uses(&before, &after) >= 1,
            "{name}: an unaligned address must bounce"
        );
    }

    // ---- unaligned LENGTH (aligned address): bounce ----------------------------------------
    for (name, len) in [
        ("u-len-4m", 4 * MIB + 100),
        ("u-len-multi", 40 * MIB + 1),
        ("u-len-5000", 5000),
    ] {
        let data = pattern(len);
        let uri = write(dir, name, &data);
        let mut b = backing(len + 2 * PAGE);
        let before = snap();
        read_and_check(&store, &uri, &data, &mut b, PAGE).await;
        let after = snap();
        assert_eq!(after.io.direct_gets - before.io.direct_gets, 1, "{name}");
        assert!(
            bounce_uses(&before, &after) >= 1,
            "{name}: an unaligned length must bounce"
        );
    }

    // ---- below the O_DIRECT threshold: buffered, never O_DIRECT --------------------------
    {
        let data = pattern(100);
        let uri = write(dir, "tiny", &data);
        let mut b = backing(PAGE);
        let before = snap();
        read_and_check(&store, &uri, &data, &mut b, 0).await;
        let after = snap();
        assert_eq!(
            after.io.direct_gets, before.io.direct_gets,
            "100 bytes is below min_io_size"
        );
        assert_eq!(bounce_uses(&before, &after), 0);
    }

    // ---- file:// served by the direct-I/O config is read O_DIRECT, like its `get` --------
    {
        let data = pattern(8 * MIB);
        let path = dir.join("file-scheme");
        std::fs::write(&path, &data).unwrap();
        let uri = format!("file://{}", path.display());
        let mut b = backing(8 * MIB + PAGE);
        let before = snap();
        read_and_check(&store, &uri, &data, &mut b, 0).await;
        let after = snap();
        assert_eq!(after.io.direct_gets - before.io.direct_gets, 1);
        assert_eq!(bounce_uses(&before, &after), 0);
    }

    // ---- a store with direct I/O off never touches the direct path ----------------------
    {
        let buffered = ConfigurableFileSystemObjectStore::new(FileSystemConfig::default());
        let data = pattern(8 * MIB);
        let uri = write(dir, "buffered", &data);
        let mut b = backing(8 * MIB + PAGE);
        let before = snap();
        read_and_check(&buffered, &uri, &data, &mut b, 0).await;
        let after = snap();
        assert_eq!(after.io.direct_gets, before.io.direct_gets);
        assert_eq!(after.io.get_fallbacks, before.io.get_fallbacks);
        assert_eq!(bounce_uses(&before, &after), 0);
    }

    // ---- length mismatches on the aligned path: nothing written -------------------------
    for (name, file_len, dst_len) in [
        ("m-short", 4 * MIB, 8 * MIB),
        ("m-long", 12 * MIB, 8 * MIB),
        ("m-long-by-a-page", 8 * MIB + PAGE, 8 * MIB),
    ] {
        let uri = write(dir, name, &pattern(file_len));
        let mut b = backing(dst_len + 2 * PAGE);
        let progress = ReceiveProgress::new();
        let err = store
            .get_into(&uri, &mut b.as_mut_slice()[PAGE..PAGE + dst_len], &progress)
            .await
            .expect_err(name);
        assert_eq!(
            object_error(&err).map(|e| e.kind),
            Some(ObjectErrorKind::LengthMismatch {
                expected: dst_len as u64,
                actual: Some(file_len as u64)
            }),
            "{name}: {err:#}"
        );
        assert!(
            b.as_slice().iter().all(|x| *x == FILL),
            "{name}: the buffer was written"
        );
        assert_eq!(progress.filled(), 0, "{name}");
    }
}
