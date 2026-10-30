// tests/direct_get_into_small_bounce_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: the bounce pool's buffer size is process-wide, read once (S3DLIO_DIRECT_IO_BOUNCE_
// SIZE_MB, default 16 MiB, the same as the 16 MiB read chunk of `get_into`). With the default,
// one chunk is one bounce buffer and the "copy position of buffer k inside the chunk" arithmetic
// is never exercised past k = 0. This binary sets 4 MiB BEFORE the pool exists, so every 16 MiB
// chunk is four buffers through the ping-pong pair, and sweeps sizes around every boundary
// (the bounce size, the chunk size, their multiples) at unaligned destination addresses.
//
// For each read: the bytes are exact, nothing outside the destination is written (canaries),
// progress is exactly the destination length, and the bounce pool was really leased.

#[path = "common/odirect.rs"]
mod odirect;

use s3dlio::direct_bounce::{bounce_stats, config};
use s3dlio::file_store_direct::ConfigurableFileSystemObjectStore;
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::{ObjectStore, ReceiveProgress};

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const FILL: u8 = 0xA5;

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chunk_of_several_bounce_buffers_is_copied_to_the_right_place() {
    // Before anything touches the pool: it is latched on first use.
    #[allow(deprecated)]
    std::env::set_var("S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB", "4");
    assert_eq!(
        config().buffer_size,
        4 * MIB,
        "the configuration must be 4 MiB for this test to mean anything"
    );

    let Some(scratch) = odirect::odirect_scratch_dir("get-into-small-bounce-295-") else {
        return;
    };
    let store = ConfigurableFileSystemObjectStore::with_direct_io();

    let sizes = [
        4 * MIB - 1,
        4 * MIB,
        4 * MIB + 1,
        8 * MIB + 4097,
        16 * MIB - 1,
        16 * MIB,
        16 * MIB + 1,
        16 * MIB + 4097,
        32 * MIB + 3,
        36 * MIB - 4095,
    ];
    // Destination addresses 1, 511 and 4095 bytes past a page boundary (always unaligned, so the
    // whole read is bounced), and an aligned address with an odd length (aligned head plus a
    // bounced tail).
    let offsets = [1usize, 511, 4095, 0];

    for &len in &sizes {
        let data = pattern(len);
        let path = scratch.path().join(format!("f-{len}.bin"));
        std::fs::write(&path, &data).unwrap();
        let uri = format!("direct://{}", path.display());
        for &off in &offsets {
            if off == 0 && len.is_multiple_of(PAGE) {
                continue; // fully aligned: not a bounce read
            }
            let mut backing = AlignedBuf::new_zeroed(len + 2 * PAGE, PAGE).unwrap();
            backing.as_mut_slice().fill(FILL);
            let progress = ReceiveProgress::new();
            let before = bounce_stats();
            store
                .get_into(&uri, &mut backing.as_mut_slice()[off..off + len], &progress)
                .await
                .unwrap_or_else(|e| panic!("len {len} off {off}: {e:#}"));
            let after = bounce_stats();

            let all = backing.as_slice();
            let got = &all[off..off + len];
            if got != data.as_slice() {
                let first = got.iter().zip(&data).position(|(a, b)| a != b).unwrap();
                panic!("len {len} off {off}: content differs, first at byte {first}");
            }
            assert!(
                all[..off].iter().all(|b| *b == FILL),
                "len {len} off {off}: wrote before the destination"
            );
            assert!(
                all[off + len..].iter().all(|b| *b == FILL),
                "len {len} off {off}: wrote past the destination"
            );
            assert_eq!(
                progress.filled(),
                len as u64,
                "len {len} off {off}: progress"
            );
            let used = (after.leases[1] - before.leases[1])
                + (after.implicit_fallbacks[1] - before.implicit_fallbacks[1])
                + (after.fallbacks[1] - before.fallbacks[1]);
            assert!(used >= 1, "len {len} off {off}: the bounce pool was used");
        }
    }
}
