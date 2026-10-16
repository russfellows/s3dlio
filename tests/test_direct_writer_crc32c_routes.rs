// tests/test_direct_writer_crc32c_routes.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Known-answer test of the direct:// streaming writer (`DirectIOWriter`): over every route
// the writer can take, `checksum()` must equal the CRC-32C of exactly the bytes the caller
// handed in (computed here, independently, with `checksum::crc32c_of`), and the stored file
// must hold those bytes (or, for a compressed stream, decode to them).
//
// Each route also proves it used O_DIRECT, not the buffered fallback: the writer's file
// descriptor is found through /proc/self/fd and its open flags must contain O_DIRECT
// (`DirectIOWriter` has no buffered fallback; a refused O_DIRECT open is an error), and the
// bytes already on disk before `finalize()` show which route the data took (nothing for a
// tail-only stream, the whole chunk for a page-aligned chunk that bypasses staging, whole
// staging buffers for a long stream). `direct_io_counters()` counts `put_object` transfers,
// not streaming writers, so it cannot serve as evidence here; this test asserts it does not
// move, so a future change that starts counting writers is noticed.
//
// The scratch directory is S3DLIO_TEST_DIR (a real filesystem, not tmpfs). When that
// filesystem refuses O_DIRECT the tests FAIL; they never skip.

use s3dlio::checksum::{crc32c_label, crc32c_of};
use s3dlio::file_store_direct::{direct_io_counters, DirectIOWriter, FileSystemConfig};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::{CompressionConfig, ObjectWriter};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const MIB: usize = 1024 * 1024;
const ALIGN: usize = 4096;
/// `DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE` (src/constants.rs).
const STAGING: usize = 64 * MIB;

/// A scratch directory on a filesystem that supports O_DIRECT. Panics if there is none.
fn scratch() -> tempfile::TempDir {
    let base = std::env::var_os("S3DLIO_TEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| env!("CARGO_TARGET_TMPDIR").into());
    std::fs::create_dir_all(&base).expect("create the test base directory");
    let dir = tempfile::Builder::new()
        .prefix("direct-crc32c-routes-")
        .tempdir_in(&base)
        .expect("temp dir");
    let probe = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(dir.path().join("o_direct_probe"));
    if let Err(e) = probe {
        panic!(
            "{} does not support O_DIRECT ({e}); set S3DLIO_TEST_DIR to a directory on a real \
             filesystem (not tmpfs)",
            dir.path().display()
        );
    }
    dir
}

fn direct_config() -> FileSystemConfig {
    FileSystemConfig {
        direct_io: true,
        ..FileSystemConfig::default()
    }
}

/// Deterministic, hard-to-compress bytes (xorshift64*), so a wrong offset or a lost byte
/// changes the CRC.
fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes();
        let take = (len - out.len()).min(8);
        out.extend_from_slice(&v[..take]);
    }
    out
}

/// True when some open file descriptor of this process points at `path` with O_DIRECT set.
fn path_open_with_o_direct(path: &Path) -> bool {
    let want = std::fs::canonicalize(path).expect("canonicalize stored path");
    for entry in std::fs::read_dir("/proc/self/fd").expect("/proc/self/fd") {
        let entry = entry.expect("fd entry");
        let Ok(target) = std::fs::read_link(entry.path()) else {
            continue;
        };
        if target != want {
            continue;
        }
        let fd = entry.file_name();
        let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{}", fd.to_string_lossy()))
            .expect("fdinfo");
        let flags = info
            .lines()
            .find_map(|l| l.strip_prefix("flags:"))
            .map(|v| i32::from_str_radix(v.trim(), 8).expect("octal flags"))
            .expect("flags line");
        if flags & libc::O_DIRECT != 0 {
            return true;
        }
    }
    false
}

/// Feed `chunks` to a fresh direct writer, then check checksum + stored bytes. `on_disk` is
/// told how many bytes are on disk before `finalize()` so a route can assert which path the
/// data took. Returns the stored file's path.
async fn run_route(
    dir: &Path,
    name: &str,
    compression: CompressionConfig,
    chunks: &[&[u8]],
    on_disk: impl FnOnce(u64),
) -> PathBuf {
    let counters_before = direct_io_counters();
    let path = dir.join(name);
    let mut w = Box::new(
        DirectIOWriter::new_with_compression(path.clone(), direct_config(), compression)
            .await
            .expect("open the O_DIRECT writer (no buffered fallback exists)"),
    );
    let stored = compression.stored_path(&path);
    let mut all = Vec::new();
    for c in chunks {
        w.write_chunk(c).await.expect("write_chunk");
        all.extend_from_slice(c);
    }

    // The route's evidence, taken while the O_DIRECT descriptor is still open.
    assert!(
        path_open_with_o_direct(&stored),
        "{name}: no open descriptor on {} carries O_DIRECT",
        stored.display()
    );
    on_disk(
        std::fs::metadata(&stored)
            .expect("stat before finalize")
            .len(),
    );

    let expected = crc32c_label(crc32c_of(&all));
    assert_eq!(
        w.checksum().as_deref(),
        Some(expected.as_str()),
        "{name}: checksum() before finalize"
    );
    assert_eq!(w.bytes_written(), all.len() as u64, "{name}: bytes_written");
    w.finalize().await.expect("finalize");

    let stored_bytes = std::fs::read(&stored).expect("read back the stored file");
    if compression.is_enabled() {
        assert!(
            stored.to_string_lossy().ends_with(compression.extension()),
            "{name}: compressed file lacks its extension"
        );
        let decoded = zstd::decode_all(stored_bytes.as_slice()).expect("zstd decode");
        assert_eq!(decoded.len(), all.len(), "{name}: decoded length");
        assert!(decoded == all, "{name}: decoded bytes differ from input");
        // The checksum is over the uncompressed bytes, not the stored ones.
        assert_ne!(
            crc32c_label(crc32c_of(&stored_bytes)),
            expected,
            "{name}: stored (compressed) bytes must not be what the checksum covers"
        );
    } else {
        assert_eq!(stored_bytes.len(), all.len(), "{name}: stored length");
        assert!(
            stored_bytes == all,
            "{name}: stored bytes differ from input"
        );
    }
    assert_eq!(
        crc32c_label(crc32c_of(&if compression.is_enabled() {
            zstd::decode_all(stored_bytes.as_slice()).unwrap()
        } else {
            stored_bytes
        })),
        expected,
        "{name}: CRC of the stored content"
    );
    assert_eq!(
        direct_io_counters(),
        counters_before,
        "{name}: direct_io_counters() counts put_object transfers, not streaming writers"
    );
    stored
}

#[tokio::test]
async fn crc32c_empty_stream() {
    let dir = scratch();
    let stored = run_route(dir.path(), "empty.bin", CompressionConfig::None, &[], |n| {
        assert_eq!(n, 0)
    })
    .await;
    assert_eq!(std::fs::metadata(stored).unwrap().len(), 0);
    // Known answer: the CRC-32C of nothing is 0.
    assert_eq!(crc32c_label(crc32c_of(&[])), "crc32c:00000000");
}

#[tokio::test]
async fn crc32c_one_small_unaligned_chunk_is_tail_only() {
    let dir = scratch();
    let data = pseudo_random(1000, 1);
    run_route(
        dir.path(),
        "tail.bin",
        CompressionConfig::None,
        &[&data],
        // Staged only: nothing reaches the disk until finalize() appends the tail.
        |n| assert_eq!(n, 0, "a sub-page chunk must stay staged"),
    )
    .await;
}

#[tokio::test]
async fn crc32c_many_small_unaligned_chunks() {
    let dir = scratch();
    let data = pseudo_random(3 * MIB + 777, 2);
    let sizes = [1usize, 7, 4095, 4097, 333, 8191, 12_345, 1];
    let mut chunks: Vec<&[u8]> = Vec::new();
    let (mut off, mut i) = (0usize, 0usize);
    while off < data.len() {
        let n = sizes[i % sizes.len()].min(data.len() - off);
        chunks.push(&data[off..off + n]);
        off += n;
        i += 1;
    }
    assert!(chunks.len() > 500);
    run_route(
        dir.path(),
        "many.bin",
        CompressionConfig::None,
        &chunks,
        // Far below one staging buffer: everything is still staged.
        |n| assert_eq!(n, 0, "small chunks below one staging buffer stay staged"),
    )
    .await;
}

#[tokio::test]
async fn crc32c_page_aligned_chunk_bypasses_staging() {
    let dir = scratch();
    let mut buf = AlignedBuf::new(2 * MIB, ALIGN).unwrap();
    buf.as_mut_slice()
        .copy_from_slice(&pseudo_random(2 * MIB, 3));
    run_route(
        dir.path(),
        "aligned.bin",
        CompressionConfig::None,
        &[buf.as_slice()],
        // Written straight from the caller's memory at write_chunk time.
        |n| {
            assert_eq!(
                n as usize,
                2 * MIB,
                "an aligned chunk must reach disk immediately"
            )
        },
    )
    .await;
}

#[tokio::test]
async fn crc32c_mixed_aligned_and_unaligned() {
    let dir = scratch();
    let mut a = AlignedBuf::new(MIB, ALIGN).unwrap();
    a.as_mut_slice().copy_from_slice(&pseudo_random(MIB, 4));
    let mut b = AlignedBuf::new(8 * ALIGN, ALIGN).unwrap();
    b.as_mut_slice()
        .copy_from_slice(&pseudo_random(8 * ALIGN, 5));
    let odd1 = pseudo_random(100_001, 6);
    let odd2 = pseudo_random(5, 7);
    // An aligned chunk after a staged remainder takes the top-up path; the aligned chunk
    // first takes the bypass; the trailing odd bytes are the tail.
    let odd3 = pseudo_random(3 * ALIGN + 17, 8);
    let chunks: [&[u8]; 6] = [
        a.as_slice(),
        &odd1,
        b.as_slice(),
        &odd2,
        &odd3,
        b.as_slice(),
    ];
    run_route(
        dir.path(),
        "mixed.bin",
        CompressionConfig::None,
        &chunks,
        |n| {
            assert!(
                n as usize >= MIB,
                "the leading aligned chunk must already be on disk"
            )
        },
    )
    .await;
}

#[tokio::test]
async fn crc32c_more_than_one_staging_buffer() {
    let dir = scratch();
    let total = 2 * STAGING + 2 * MIB + 12_345; // ~130 MiB
    let data = pseudo_random(total, 9);
    let sizes = [1_000_003usize, 4097, 65_537, 3, 2_097_153, 777_777];
    let mut chunks: Vec<&[u8]> = Vec::new();
    let (mut off, mut i) = (0usize, 0usize);
    while off < data.len() {
        let n = sizes[i % sizes.len()].min(data.len() - off);
        chunks.push(&data[off..off + n]);
        off += n;
        i += 1;
    }
    run_route(
        dir.path(),
        "long.bin",
        CompressionConfig::None,
        &chunks,
        // Two whole staging buffers have been flushed with O_DIRECT before finalize.
        |n| {
            assert!(
                n as usize >= 2 * STAGING,
                "{n} bytes on disk before finalize; expected at least two staging buffers"
            )
        },
    )
    .await;
}

#[tokio::test]
async fn crc32c_zstd_stream_checksums_the_uncompressed_bytes() {
    let dir = scratch();
    // Compressible text-like data, in odd-sized chunks.
    let mut data = Vec::new();
    for i in 0..60_000u32 {
        data.extend_from_slice(format!("line {i} of the direct writer crc32c test\n").as_bytes());
    }
    let chunks: Vec<&[u8]> = data.chunks(10_007).collect();
    let stored = run_route(
        dir.path(),
        "z.bin",
        CompressionConfig::Zstd { level: 3 },
        &chunks,
        |_| {},
    )
    .await;
    assert!(
        (std::fs::metadata(&stored).unwrap().len() as usize) < data.len() / 4,
        "the stream was not actually compressed"
    );
}

#[tokio::test]
async fn crc32c_zstd_stream_longer_than_one_staging_buffer() {
    let dir = scratch();
    // Incompressible data so the compressed stream itself exceeds one staging buffer.
    let data = pseudo_random(STAGING + 6 * MIB + 4321, 10);
    let chunks: Vec<&[u8]> = data.chunks(1_048_573).collect();
    run_route(
        dir.path(),
        "zlong.bin",
        CompressionConfig::Zstd { level: 1 },
        &chunks,
        |n| {
            assert!(
                n as usize >= STAGING,
                "{n} compressed bytes on disk before finalize; expected a full staging buffer"
            )
        },
    )
    .await;
}
