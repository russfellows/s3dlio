// tests/test_direct_io_chunking_244.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Issue #244: a `direct://` PUT wrote ONE 4 KiB block per write() (512 MiB took 45 s, about
// 11 MiB/s, 50-60x slower than file://). The GET side read 4 KiB per read() the same way.
//
// The structural tests count the process's write()/read() system calls through
// /proc/self/io (`syscw` / `syscr`): the old code needed len/4096 of them, the fixed code a
// handful. The count is process-wide, so every test runs under one lock. Round-trip tests
// use a REAL filesystem under the cargo target dir (O_DIRECT is unsupported on tmpfs).

use bytes::Bytes;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::ObjectStore;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::const_new(());

const MIB: usize = 1024 * 1024;
const ALIGN: usize = 4096;

fn scratch_dir() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("direct-chunking-244-")
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .expect("temp dir")
}

fn store() -> ConfigurableFileSystemObjectStore {
    ConfigurableFileSystemObjectStore::new(FileSystemConfig::direct_io())
}

/// Process-wide (write, read) system-call counts from /proc/self/io.
fn syscalls() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/self/io").expect("/proc/self/io");
    let field = |name: &str| -> u64 {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| panic!("{name} missing in /proc/self/io"))
    };
    (field("syscw:"), field("syscr:"))
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i.wrapping_mul(131) ^ (i >> 8).wrapping_mul(7)) as u8)
        .collect()
}

/// A page-aligned buffer usable as the owner of a `Bytes` (what pinned, page-aligned
/// caller buffers look like).
struct AlignedOwner(AlignedBuf, usize);
impl AsRef<[u8]> for AlignedOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0.as_slice()[..self.1]
    }
}

fn aligned_bytes(raw: &[u8]) -> Bytes {
    let mut buf = AlignedBuf::new(raw.len().max(1), ALIGN);
    buf.as_mut_slice()[..raw.len()].copy_from_slice(raw);
    let b = Bytes::from_owner(AlignedOwner(buf, raw.len()));
    if !raw.is_empty() {
        assert_eq!(
            b.as_ptr() as usize % ALIGN,
            0,
            "test buffer must be aligned"
        );
    }
    b
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn put_and_get_use_few_large_system_calls() {
    let _g = SERIAL.lock().await;
    let dir = scratch_dir();
    let s = store();
    assert!(
        ConfigurableFileSystemObjectStore::test_direct_io_support(dir.path()).await,
        "the test directory must support O_DIRECT, or this test proves nothing"
    );
    let len = 64 * MIB;
    let raw = pattern(len);
    let uri = format!("direct://{}/big.bin", dir.path().display());

    // Budget: the old code issued len/4096 = 16384 calls. A handful of calls for the file
    // create/truncate/close plus runtime noise fits easily under this.
    let budget = 256;

    for (label, payload) in [
        ("unaligned", Bytes::from(raw.clone())),
        ("aligned", aligned_bytes(&raw)),
    ] {
        let (w0, _) = syscalls();
        s.put(&uri, payload).await.expect("put");
        let (w1, _) = syscalls();
        assert!(
            w1 - w0 < budget,
            "{label}: direct:// put of {len} bytes issued {} write() calls (budget {budget}); \
             writing 4 KiB per call needs {}",
            w1 - w0,
            len / ALIGN
        );

        let (_, r0) = syscalls();
        let back = s.get(&uri).await.expect("get");
        let (_, r1) = syscalls();
        assert!(
            r1 - r0 < budget,
            "{label}: direct:// get of {len} bytes issued {} read() calls (budget {budget})",
            r1 - r0
        );
        assert_eq!(back.len(), len);
        assert!(back.as_ref() == raw.as_slice(), "{label}: content differs");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn roundtrip_size_matrix_on_a_real_filesystem() {
    let _g = SERIAL.lock().await;
    let dir = scratch_dir();
    let s = store();
    for len in [0usize, 1, 4095, 4096, 4097, MIB + 3, 33 * MIB] {
        let raw = pattern(len);
        for (label, payload) in [
            ("unaligned", Bytes::from(raw.clone())),
            ("aligned", aligned_bytes(&raw)),
        ] {
            let uri = format!("direct://{}/rt-{label}-{len}.bin", dir.path().display());
            s.put(&uri, payload).await.expect("put");
            let meta = s.stat(&uri).await.expect("stat");
            assert_eq!(meta.size as usize, len, "{label} size {len}");
            let back = s.get(&uri).await.expect("get");
            assert!(back.as_ref() == raw.as_slice(), "{label} size {len}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn put_multipart_with_an_unaligned_part_size_has_no_padding_inside() {
    let _g = SERIAL.lock().await;
    let dir = scratch_dir();
    let s = store();
    let raw = pattern(3 * MIB + 17);
    let uri = format!("direct://{}/mp.bin", dir.path().display());
    s.put_multipart(&uri, Bytes::from(raw.clone()), Some(5000))
        .await
        .expect("put_multipart");
    let back = s.get(&uri).await.expect("get");
    assert!(back.as_ref() == raw.as_slice());

    // And it is as cheap in system calls as a plain put.
    let (w0, _) = syscalls();
    s.put_multipart(&uri, Bytes::from(pattern(64 * MIB)), None)
        .await
        .expect("put_multipart");
    let (w1, _) = syscalls();
    assert!(
        w1 - w0 < 256,
        "put_multipart issued {} write() calls",
        w1 - w0
    );
}

/// Sanity check only (timing, not proof): 256 MiB must take seconds, not the ~25 s the
/// 4 KiB-per-write loop needed. The structural proof is the system-call count above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn throughput_sanity_256_mib_put() {
    let _g = SERIAL.lock().await;
    let dir = scratch_dir();
    let s = store();
    let data = Bytes::from(vec![0x5Au8; 256 * MIB]);
    let uri = format!("direct://{}/t.bin", dir.path().display());
    let t = Instant::now();
    s.put(&uri, data).await.expect("put");
    let took = t.elapsed();
    assert!(
        took < Duration::from_secs(10),
        "256 MiB direct:// put took {took:?} (sanity limit 10 s)"
    );
}
