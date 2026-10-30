// tests/get_into_local_alloc_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267, Rule #1: the local-file caller-buffer GET makes NO whole-object allocation, on
// any path (file://, direct:// aligned, direct:// bounce). A counting global allocator watches
// every thread while a read runs and counts allocations of at least half the object; the
// allocating `get` is shown to trip it, so the counter would catch a copy.
//
// The destination is allocated before the read and handed in as `&mut [u8]`; the bytes can
// only have landed there (the content check shows they arrived), because nothing else of the
// object's size was allocated. The bounce pool is warmed first, so its one-time buffer
// allocation is not part of the window.
//
// Own process: it installs a global allocator and measures process-wide.

#[path = "common/counting_alloc.rs"]
mod counting_alloc;
#[path = "common/odirect.rs"]
mod odirect;

use s3dlio::file_store_direct::ConfigurableFileSystemObjectStore;
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use s3dlio::FileSystemObjectStore;

fn no_thread_excluded() -> bool {
    false
}

#[global_allocator]
static ALLOC: counting_alloc::CountingAlloc =
    counting_alloc::CountingAlloc::new(no_thread_excluded);

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;
const OBJECT: usize = 64 * MIB;
/// Any single allocation this large is (a large part of) a copy of the object.
const BIG: usize = OBJECT / 2;

fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i % 160) as u8 + ((i / 160) % 5) as u8)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_path_allocates_the_object() {
    let Some(dir) = odirect::odirect_scratch_dir("get-into-alloc-295-") else {
        return;
    };
    let data = pattern(OBJECT);
    let path = dir.path().join("object");
    std::fs::write(&path, &data).unwrap();
    let file_uri = format!("file://{}", path.display());
    let direct_uri = format!("direct://{}", path.display());

    let file_store = FileSystemObjectStore::new();
    let direct_store = ConfigurableFileSystemObjectStore::with_direct_io();

    // Destinations, allocated before any window: one page aligned (the O_DIRECT zero-copy case)
    // and one whose usable window starts at an odd address (the bounce case).
    let mut aligned = AlignedBuf::new_zeroed(OBJECT + 2 * PAGE, PAGE).unwrap();
    let mut plain = vec![0u8; OBJECT + 16];

    // Warm the bounce pool, outside any window.
    direct_store
        .get_into(
            &direct_uri,
            &mut plain[8..8 + OBJECT],
            &ReceiveProgress::new(),
        )
        .await
        .expect("warm-up");

    for case in ["file://", "direct:// aligned", "direct:// bounce"] {
        plain.fill(0);
        aligned.as_mut_slice().fill(0);
        let progress = ReceiveProgress::new();
        ALLOC.arm(BIG);
        let result = match case {
            "file://" => {
                file_store
                    .get_into(&file_uri, &mut plain[..OBJECT], &progress)
                    .await
            }
            "direct:// aligned" => {
                let dst = &mut aligned.as_mut_slice()[PAGE..PAGE + OBJECT];
                direct_store.get_into(&direct_uri, dst, &progress).await
            }
            // An address that is not page aligned: the documented bounce copy.
            _ => {
                direct_store
                    .get_into(&direct_uri, &mut plain[8..8 + OBJECT], &progress)
                    .await
            }
        };
        let m = ALLOC.disarm();
        result.unwrap_or_else(|e| panic!("{case}: {e:#}"));
        let landed = match case {
            "file://" => &plain[..OBJECT],
            "direct:// aligned" => &aligned.as_slice()[PAGE..PAGE + OBJECT],
            _ => &plain[8..8 + OBJECT],
        };
        assert!(landed == data.as_slice(), "{case}: content");
        assert_eq!(progress.filled(), OBJECT as u64, "{case}");
        assert_eq!(
            m.big_allocs, 0,
            "{case}: a whole-object-sized allocation was made ({} bytes in total)",
            m.total
        );
        assert!(
            m.peak < BIG,
            "{case}: peak live allocation {} bytes",
            m.peak
        );
    }

    // Control: the allocating GET allocates the whole object, and the counter sees it.
    for (store, uri) in [
        (&file_store as &dyn ObjectStore, &file_uri),
        (&direct_store as &dyn ObjectStore, &direct_uri),
    ] {
        ALLOC.arm(BIG);
        let got = store.get(uri).await.unwrap();
        let m = ALLOC.disarm();
        assert_eq!(got.len(), OBJECT);
        assert!(
            m.big_allocs >= 1,
            "{uri}: the control `get` allocated nothing big; the counter cannot see a copy"
        );
    }
}
