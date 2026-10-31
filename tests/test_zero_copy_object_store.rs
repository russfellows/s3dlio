// tests/test_zero_copy_object_store.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Stage G (zero copy everywhere, RULE #1), object-store side: the proofs for
//
//   * W4  the buffered writers hand the buffer they accumulated to the store as `Bytes`
//         without another copy, and `write_owned_bytes` moves an owned `Vec` into an empty
//         writer instead of copying it;
//   * R3  the range engine receives every range straight into its disjoint segment of the
//         master buffer (no per-range `Bytes` that is copied into the master afterwards).
//
// How a read is measured: a counting global allocator keeps, for the flagged thread only, the
// live bytes, their peak and the large allocations.  R2 (an S3 `get` / `get_range` receives into
// ONE pre-sized buffer) is in `tests/test_zero_copy_s3_receive.rs`: since #303 an S3 read's
// transport runs on s3dlio's I/O-owner runtime, not on the caller's thread, so it counts every
// thread of its own process instead.
//
// Own process: it installs a global allocator.

use bytes::Bytes;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[cfg(feature = "backend-azure")]
#[path = "common/azure_fake.rs"]
mod azure_fake;

struct Counting;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
    static LARGE: Cell<usize> = const { Cell::new(0) };
}

fn grow(size: usize) {
    TRACKING
        .try_with(|t| {
            if t.get() {
                let size_signed = isize::try_from(size).unwrap_or(isize::MAX);
                LIVE.try_with(|l| {
                    let now = l.get().saturating_add(size_signed);
                    l.set(now);
                    PEAK.try_with(|p| p.set(p.get().max(now))).ok();
                })
                .ok();
                if size >= MIB {
                    LARGE.try_with(|c| c.set(c.get().saturating_add(size))).ok();
                }
            }
        })
        .ok();
}

fn shrink(size: usize) {
    TRACKING
        .try_with(|t| {
            if t.get() {
                let size_signed = isize::try_from(size).unwrap_or(isize::MAX);
                LIVE.try_with(|l| l.set(l.get().saturating_sub(size_signed)))
                    .ok();
            }
        })
        .ok();
}

// SAFETY: every method forwards to the system allocator unchanged, after noting the size; the
// bookkeeping uses const-initialised `Cell`s, which allocate nothing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrink(layout.size());
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        shrink(layout.size());
        grow(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const MIB: usize = 1024 * 1024;

/// What a measured future allocated on this thread.
#[derive(Debug, Clone, Copy)]
struct Measure {
    /// The most bytes live at once, above what was live when the window opened (read by the
    /// Azure tests only).
    #[cfg_attr(not(feature = "backend-azure"), allow(dead_code))]
    peak: usize,
    /// Total size of the allocations of 1 MiB or more made in the window.
    large: usize,
}

/// Run `fut`, measuring this thread's allocations while it runs.
async fn measured<T>(fut: impl std::future::Future<Output = T>) -> (T, Measure) {
    LIVE.with(|l| l.set(0));
    PEAK.with(|p| p.set(0));
    LARGE.with(|c| c.set(0));
    TRACKING.with(|t| t.set(true));
    let out = fut.await;
    TRACKING.with(|t| t.set(false));
    let peak = usize::try_from(PEAK.with(Cell::get)).unwrap_or(0);
    let large = LARGE.with(Cell::get);
    (out, Measure { peak, large })
}

// ───────────────────────────── W4: buffered writers ─────────────────────────────

mod w4 {
    use super::*;
    use async_trait::async_trait;
    use s3dlio::object_store::{BufferedObjectWriter, ObjectStore, ObjectWriter};
    use std::sync::{Arc, Mutex};

    /// A store that records the pointer and length of every `Bytes` handed to `put`.
    #[derive(Default, Clone)]
    struct Recording {
        puts: Arc<Mutex<Vec<(usize, usize)>>>,
    }

    macro_rules! unsupported {
        () => {
            Err(anyhow::anyhow!("the recording store supports only put"))
        };
    }

    #[async_trait]
    impl ObjectStore for Recording {
        async fn get(&self, _uri: &str) -> anyhow::Result<Bytes> {
            unsupported!()
        }
        async fn get_range(
            &self,
            _uri: &str,
            _offset: u64,
            _length: Option<u64>,
        ) -> anyhow::Result<Bytes> {
            unsupported!()
        }
        async fn put(&self, _uri: &str, data: Bytes) -> anyhow::Result<()> {
            if let Ok(mut puts) = self.puts.lock() {
                puts.push((data.as_ptr() as usize, data.len()));
            }
            Ok(())
        }
        async fn put_multipart(
            &self,
            _uri: &str,
            _data: Bytes,
            _part_size: Option<usize>,
        ) -> anyhow::Result<()> {
            unsupported!()
        }
        async fn list(&self, _uri_prefix: &str, _recursive: bool) -> anyhow::Result<Vec<String>> {
            unsupported!()
        }
        async fn stat(&self, _uri: &str) -> anyhow::Result<s3dlio::object_store::ObjectMetadata> {
            unsupported!()
        }
        async fn delete(&self, _uri: &str) -> anyhow::Result<()> {
            unsupported!()
        }
        async fn delete_batch(&self, _uris: &[String]) -> anyhow::Result<()> {
            unsupported!()
        }
        async fn delete_prefix(&self, _uri_prefix: &str) -> anyhow::Result<()> {
            unsupported!()
        }
        async fn create_container(&self, _name: &str) -> anyhow::Result<()> {
            unsupported!()
        }
        async fn delete_container(&self, _name: &str) -> anyhow::Result<()> {
            unsupported!()
        }
        fn list_stream<'a>(
            &'a self,
            _uri_prefix: &'a str,
            _recursive: bool,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = anyhow::Result<String>> + Send + 'a>>
        {
            Box::pin(futures::stream::empty())
        }
        async fn get_writer(&self, _uri: &str) -> anyhow::Result<Box<dyn ObjectWriter>> {
            unsupported!()
        }
        async fn create_writer(
            &self,
            _uri: &str,
            _options: s3dlio::object_store::WriterOptions,
        ) -> anyhow::Result<Box<dyn ObjectWriter>> {
            unsupported!()
        }
    }

    #[tokio::test]
    async fn write_owned_bytes_into_an_empty_buffered_writer_moves_the_vec() {
        let store = Recording::default();
        let payload: Vec<u8> = (0..(8 * MIB)).map(|i| (i % 251) as u8).collect();
        let expected = payload.clone();
        let origin = payload.as_ptr() as usize;

        let mut writer = BufferedObjectWriter::new("mem://x".to_string(), &store);
        writer.write_owned_bytes(payload).await.expect("write");
        Box::new(writer).finalize().await.expect("finalize");

        let puts = store.puts.lock().expect("lock").clone();
        assert_eq!(puts.len(), 1);
        let (ptr, len) = puts[0];
        assert_eq!(len, expected.len());
        assert_eq!(
            ptr, origin,
            "RED: write_owned_bytes copied the owned Vec into the writer's buffer"
        );
    }
}

// ───────────────── load_checkpoint_with_validation (trait default) ─────────────────

mod checkpoint_default {
    use super::*;
    use async_trait::async_trait;
    use s3dlio::object_store::{ObjectMetadata, ObjectStore, ObjectWriter, WriterOptions};
    use std::sync::Mutex;

    /// A store whose `get` hands out its one `Bytes` (the sole owner of a `Vec`).
    struct OneObject(Mutex<Option<Bytes>>);

    fn no<T>() -> anyhow::Result<T> {
        Err(anyhow::anyhow!("not supported by this test double"))
    }

    #[async_trait]
    impl ObjectStore for OneObject {
        async fn get(&self, _uri: &str) -> anyhow::Result<Bytes> {
            match self.0.lock() {
                Ok(mut slot) => slot.take().ok_or_else(|| anyhow::anyhow!("already taken")),
                Err(_) => no(),
            }
        }
        async fn get_range(&self, _: &str, _: u64, _: Option<u64>) -> anyhow::Result<Bytes> {
            no()
        }
        async fn put(&self, _: &str, _: Bytes) -> anyhow::Result<()> {
            no()
        }
        async fn put_multipart(&self, _: &str, _: Bytes, _: Option<usize>) -> anyhow::Result<()> {
            no()
        }
        async fn list(&self, _: &str, _: bool) -> anyhow::Result<Vec<String>> {
            no()
        }
        fn list_stream<'a>(
            &'a self,
            _: &'a str,
            _: bool,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = anyhow::Result<String>> + Send + 'a>>
        {
            Box::pin(futures::stream::empty())
        }
        async fn stat(&self, _: &str) -> anyhow::Result<ObjectMetadata> {
            no()
        }
        async fn delete(&self, _: &str) -> anyhow::Result<()> {
            no()
        }
        async fn delete_batch(&self, _: &[String]) -> anyhow::Result<()> {
            no()
        }
        async fn delete_prefix(&self, _: &str) -> anyhow::Result<()> {
            no()
        }
        async fn create_container(&self, _: &str) -> anyhow::Result<()> {
            no()
        }
        async fn delete_container(&self, _: &str) -> anyhow::Result<()> {
            no()
        }
        async fn get_writer(&self, _: &str) -> anyhow::Result<Box<dyn ObjectWriter>> {
            no()
        }
        async fn create_writer(
            &self,
            _: &str,
            _: WriterOptions,
        ) -> anyhow::Result<Box<dyn ObjectWriter>> {
            no()
        }
    }

    #[tokio::test]
    async fn the_default_returns_the_stores_allocation_not_a_copy() {
        let payload: Vec<u8> = (0..(4 * MIB)).map(|i| (i % 241) as u8).collect();
        let origin = payload.as_ptr() as usize;
        let store = OneObject(Mutex::new(Some(Bytes::from(payload))));
        let got = store
            .load_checkpoint_with_validation("mem://ckpt", None)
            .await
            .expect("load");
        assert_eq!(got.len(), 4 * MIB);
        assert_eq!(
            got.as_ptr() as usize,
            origin,
            "RED: load_checkpoint_with_validation copied the object into a new Vec"
        );
    }
}

// ───────────────────────── Azure: W4 writer and R5 large-blob GET ─────────────────────────
//
// Against the local fake Blob service in tests/common/azure_fake.rs (no cloud endpoint). The
// fake runs on its own thread and runtime, so its object-sized allocations are not counted; the
// client runs on a current-thread runtime, so the SDK's connection and partition tasks run on
// the flagged thread.

#[cfg(feature = "backend-azure")]
mod azure {
    use super::*;
    use s3dlio::object_store::{AzureConfig, AzureObjectStore, ObjectStore};

    use super::azure_fake::{clear_azure_env, pattern, set_blob, start, uri, State};

    /// One Azure test at a time: they share the process environment (the endpoint).
    static AZURE_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fake_on_its_own_thread() -> (State, u16) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("fake runtime");
            runtime.block_on(async move {
                let started = start().await;
                tx.send(started).expect("hand over the fake");
                std::future::pending::<()>().await;
            });
        });
        rx.recv().expect("fake started")
    }

    fn client_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("client runtime")
    }

    #[test]
    fn the_buffered_writer_hands_its_buffer_to_the_put_without_a_clone() {
        const N: usize = 32 * MIB;
        let _env = AZURE_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_azure_env();
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        let (_, port) = fake_on_its_own_thread();
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        let runtime = client_runtime();
        runtime.block_on(async {
            let store = AzureObjectStore::with_config(AzureConfig::default());
            // Warm-up: connection state is not what is measured.
            let mut warm = store.get_writer(&uri("warm")).await.expect("writer");
            warm.write_chunk(&[1u8; 1024]).await.expect("write");
            warm.finalize().await.expect("warm-up put");

            let payload = pattern(N, 5);
            let mut writer = store.get_writer(&uri("w4")).await.expect("writer");
            writer.write_owned_bytes(payload).await.expect("write");
            let (result, m) = measured(async { writer.finalize().await }).await;
            result.expect("finalize");
            assert!(
                m.peak < N / 2 + N / 4,
                "RED: finalize had {} bytes live at once for a {N}-byte buffer (a clone of the \
                 whole object)",
                m.peak
            );
        });
        // The fake keeps only the block list of the SDK's staged upload (it does not assemble
        // blocks), so the stored content cannot be compared; the upload having been accepted
        // (finalize returned Ok) and the allocation bound above are the assertions.
    }

    /// The Azure range engine (explicit s3dlio mode) receives every range into its window of the
    /// master buffer: the only large allocation is the master. The control reads the same blob
    /// through the engine's `Bytes` path over `get_range` (what this read did before) and must
    /// show a second, per-range allocation of the whole blob, or the counter would be blind.
    #[test]
    fn the_azure_range_engine_allocates_the_master_buffer_and_nothing_else_large() {
        const N: usize = 48 * MIB;
        let _env = AZURE_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_azure_env();
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        let (state, port) = fake_on_its_own_thread();
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        set_blob(&state, "engine", pattern(N, 4));
        set_blob(&state, "warm", pattern(1024, 1));
        let runtime = client_runtime();
        runtime.block_on(async {
            let mut config = AzureConfig {
                enable_range_engine: true,
                ..AzureConfig::default()
            };
            config.range_engine.min_split_size = MIB as u64;
            config.range_engine.chunk_size = 4 * MIB;
            config.range_engine.max_concurrent_ranges = 4;
            let store = AzureObjectStore::with_config(config.clone());
            store.get(&uri("warm")).await.expect("warm-up read");

            let (got, m) = measured(async { store.get(&uri("engine")).await.expect("get") }).await;
            assert_eq!(got.len(), N);
            assert!(got[..] == pattern(N, 4)[..], "wrong content");
            assert!(
                m.large < N + N / 4,
                "RED: {} bytes were allocated in blocks of 1 MiB or more for a {N}-byte blob read \
                 through the range engine (the master buffer plus a buffer per range)",
                m.large
            );

            // Control: the engine's Bytes path over get_range allocates every range again.
            let engine = s3dlio::range_engine_generic::RangeEngine::new(config.range_engine);
            let control_store = store.clone();
            let ((control, _), c) = measured(async {
                engine
                    .download(
                        N as u64,
                        move |offset: u64, length: u64| {
                            let store = control_store.clone();
                            async move {
                                store.get_range(&uri("engine"), offset, Some(length)).await
                            }
                        },
                        None,
                    )
                    .await
                    .expect("control download")
            })
            .await;
            assert_eq!(control.len(), N);
            assert!(
                c.large >= N + N / 2,
                "the control read allocated only {} large bytes: the counter cannot see the \
                 per-range allocation",
                c.large
            );
        });
    }

    /// Every size around the partition boundaries reads back byte for byte (the first response
    /// is received into the front of the buffer, the rest through pinned ranged reads, each into
    /// its own window of the same buffer).
    #[test]
    fn blobs_around_the_partition_boundaries_read_back_exactly() {
        const P: usize = MIB;
        let _env = AZURE_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_azure_env();
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        let (state, port) = fake_on_its_own_thread();
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        let runtime = client_runtime();
        runtime.block_on(async {
            for concurrency in [1usize, 4] {
                let config = AzureConfig {
                    sdk_download_concurrency: concurrency,
                    sdk_download_part_size: P,
                    ..AzureConfig::default()
                };
                let store = AzureObjectStore::with_config(config);
                for (i, size) in [0, 1, P - 1, P, P + 1, 2 * P, 3 * P + 17, 7 * P]
                    .into_iter()
                    .enumerate()
                {
                    let key = format!("edge-{concurrency}-{i}");
                    set_blob(&state, &key, pattern(size, 3));
                    let got = store.get(&uri(&key)).await.expect("get");
                    assert_eq!(got.len(), size, "length for size {size}");
                    assert!(got[..] == pattern(size, 3)[..], "content for size {size}");
                }
            }
        });
    }

    #[test]
    fn a_large_blob_get_copies_each_byte_once_into_one_buffer() {
        const N: usize = 96 * MIB;
        let _env = AZURE_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_azure_env();
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        let (state, port) = fake_on_its_own_thread();
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        set_blob(&state, "r5", pattern(N, 9));
        set_blob(&state, "warm", pattern(1024, 1));
        let runtime = client_runtime();
        runtime.block_on(async {
            let store = AzureObjectStore::with_config(AzureConfig::default());
            store.get(&uri("warm")).await.expect("warm-up read");

            let (got, m) = measured(async { store.get(&uri("r5")).await.expect("get") }).await;
            assert_eq!(got.len(), N);
            assert!(got[..] == pattern(N, 9)[..], "wrong content");
            assert!(
                m.peak < N + N / 2,
                "RED: a {N}-byte blob get had {} bytes live at once (SDK partition buffers AND \
                 the destination)",
                m.peak
            );
        });
    }

    // ───────────────────── fix round: cancellation, pinning, strict replies ─────────────────────

    use s3dlio::azure_client::AzureBlob;
    use s3dlio::receive_progress::ReceiveProgress;

    /// A client runtime and a fresh fake, with the environment pointed at it.
    fn setup() -> (
        std::sync::MutexGuard<'static, ()>,
        State,
        tokio::runtime::Runtime,
    ) {
        let env = AZURE_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_azure_env();
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "3");
        let (state, port) = fake_on_its_own_thread();
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        (env, state, client_runtime())
    }

    fn requests(state: &State) -> usize {
        state.lock().expect("state").requests.len()
    }

    /// A store reading blobs in 1 MiB partitions, two at a time.
    fn small_partition_store() -> AzureObjectStore {
        AzureObjectStore::with_config(AzureConfig {
            sdk_download_concurrency: 2,
            sdk_download_part_size: MIB,
            ..AzureConfig::default()
        })
    }

    /// Cancelling a large read (a caller timeout, a dropped future) stops EVERY request it
    /// started: nothing keeps fetching the rest of the blob in the background. (The SDK's
    /// `download_into` spawns a manager task nobody aborts; it kept issuing ranged GETs for the
    /// whole remaining blob after the read was gone.)
    #[test]
    fn cancelling_a_large_read_stops_every_request() {
        const N: usize = 48 * MIB;
        let (_env, state, runtime) = setup();
        set_blob(&state, "cancel", pattern(N, 6));
        runtime.block_on(async {
            let store = small_partition_store();
            let target = uri("cancel");
            let mut read = Box::pin(store.get(&target));
            let seen = async {
                while requests(&state) < 4 {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            };
            tokio::select! {
                result = &mut read => panic!("the read finished before it could be cancelled: {:?}", result.map(|b| b.len())),
                () = seen => {}
            }
            drop(read);
            let at_cancel = requests(&state);
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            let later = requests(&state);
            assert!(
                at_cancel < 40,
                "the read was nearly finished ({at_cancel} requests) when it was cancelled"
            );
            assert!(
                later <= at_cancel + 4,
                "RED: {later} requests were made after the read was cancelled at {at_cancel}: \
                 something kept fetching the blob in the background"
            );
        });
    }

    /// A partition that fails ends the read, and with it every other request.
    #[test]
    fn a_failed_partition_stops_every_request() {
        const N: usize = 48 * MIB;
        let (_env, state, runtime) = setup();
        set_blob(&state, "fail", pattern(N, 6));
        state.lock().expect("state").cap = 3;
        runtime.block_on(async {
            let store = small_partition_store();
            let result = store.get(&uri("fail")).await;
            assert!(
                result.is_err(),
                "a read whose partitions are refused succeeded"
            );
            let at_error = requests(&state);
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            let later = requests(&state);
            assert!(
                later <= at_error + 4,
                "RED: {later} requests were made after the read failed at {at_error}"
            );
        });
    }

    /// Without a strong ETag the parts cannot be pinned to one version, and unpinned responses
    /// are never combined: the blob is read by ONE request, so a blob replaced between requests
    /// cannot be stitched from two versions.
    #[test]
    fn a_blob_without_a_strong_etag_is_read_by_one_request() {
        const N: usize = 6 * MIB;
        for weak in [false, true] {
            let (_env, state, runtime) = setup();
            set_blob(&state, "unpinned", pattern(N, 8));
            if weak {
                super::azure_fake::set_etag(&state, "unpinned", "W/\"0x9\"");
            } else {
                state.lock().expect("state").omit_etag = true;
            }
            runtime.block_on(async {
                let store = small_partition_store();
                let before = requests(&state);
                let got = store.get(&uri("unpinned")).await.expect("get");
                assert!(got[..] == pattern(N, 8)[..], "wrong content (weak={weak})");
                let made = state.lock().expect("state").requests[before..].to_vec();
                let ranges: Vec<_> = made.iter().filter_map(|r| r.range.clone()).collect();
                assert!(
                    made.len() <= 2,
                    "RED: {} requests (weak={weak}) read an unpinnable blob in parts: {ranges:?}",
                    made.len()
                );
            });
        }
    }

    /// A first response without `Content-Range` is not the whole blob when it is exactly as long
    /// as the range asked for: taking it for the whole blob silently truncated the blob.
    #[test]
    fn a_first_response_without_content_range_is_not_taken_for_the_whole_blob() {
        let (_env, state, runtime) = setup();
        set_blob(&state, "stripped", pattern(3 * MIB, 2));
        set_blob(&state, "small", pattern(100 * 1024, 4));
        set_blob(&state, "ignored", pattern(3 * MIB, 5));
        runtime.block_on(async {
            let store = small_partition_store();
            state.lock().expect("state").strip_content_range = true;
            let truncated = store.get(&uri("stripped")).await;
            assert!(
                truncated.is_err(),
                "RED: a 3 MiB blob whose Content-Range was stripped came back as {} bytes",
                truncated.map(|b| b.len()).unwrap_or(0)
            );
            // Shorter than the range asked for: the whole blob, certainly.
            let small = store.get(&uri("small")).await.expect("small blob");
            assert!(small[..] == pattern(100 * 1024, 4)[..]);
            // A service that ignores the range and answers 200 with the whole blob.
            {
                let mut fake = state.lock().expect("state");
                fake.strip_content_range = false;
                fake.ignore_range = true;
            }
            let whole = store
                .get(&uri("ignored"))
                .await
                .expect("200 with the whole blob");
            assert!(whole[..] == pattern(3 * MIB, 5)[..]);
        });
    }

    // ------------------------------- AzureBlob::get_range_into -------------------------------

    fn blob_client() -> AzureBlob {
        AzureBlob::with_default_credential("acct", "ctr").expect("client")
    }

    #[test]
    fn get_range_into_reads_a_window_and_clamps_at_the_end() {
        let (_env, state, runtime) = setup();
        let data = pattern(1000, 1);
        set_blob(&state, "win", data.clone());
        runtime.block_on(async {
            let cli = blob_client();
            let progress = ReceiveProgress::new();
            let mut dst = vec![0xEEu8; 300];
            let n = cli
                .get_range_into("win", 100, &mut dst, &progress)
                .await
                .expect("window");
            assert_eq!(n, 300);
            assert_eq!(dst, data[100..400]);
            assert_eq!(progress.filled(), 300);
            // A window that runs past the end gets the bytes that exist.
            let mut tail = vec![0xEEu8; 300];
            let n = cli
                .get_range_into("win", 900, &mut tail, &ReceiveProgress::new())
                .await
                .expect("tail");
            assert_eq!(n, 100);
            assert_eq!(tail[..100], data[900..]);
            assert!(
                tail[100..].iter().all(|b| *b == 0xEE),
                "nothing past the data"
            );
        });
    }

    #[test]
    fn get_range_into_refuses_a_window_that_starts_at_or_past_the_end() {
        let (_env, state, runtime) = setup();
        set_blob(&state, "end", pattern(1000, 1));
        runtime.block_on(async {
            let cli = blob_client();
            for start in [1000u64, 1001, 5000] {
                let mut dst = vec![0xEEu8; 16];
                let result = cli
                    .get_range_into("end", start, &mut dst, &ReceiveProgress::new())
                    .await;
                assert!(result.is_err(), "start {start} past the end succeeded");
                assert!(dst.iter().all(|b| *b == 0xEE), "nothing may be written");
            }
        });
    }

    /// An empty blob read from offset 0 is the SDK's unranged retry after the service's 416: the
    /// reply is empty and the count is 0 (documented on `get_range_into`).
    #[test]
    fn get_range_into_on_an_empty_blob_at_offset_zero_returns_zero_bytes() {
        let (_env, state, runtime) = setup();
        set_blob(&state, "empty", Vec::new());
        runtime.block_on(async {
            let cli = blob_client();
            let mut dst = vec![0xEEu8; 16];
            let n = cli
                .get_range_into("empty", 0, &mut dst, &ReceiveProgress::new())
                .await
                .expect("empty blob");
            assert_eq!(n, 0);
            assert!(dst.iter().all(|b| *b == 0xEE));
            let past = cli
                .get_range_into("empty", 1, &mut dst, &ReceiveProgress::new())
                .await;
            assert!(past.is_err(), "offset 1 of an empty blob succeeded");
        });
    }

    /// A body longer than the window (a service that ignores the range) is refused before any
    /// byte is copied; so is a reply with no `Content-Range` to a range that does not start at 0.
    #[test]
    fn get_range_into_refuses_replies_that_are_not_the_window() {
        let (_env, state, runtime) = setup();
        set_blob(&state, "odd", pattern(1000, 1));
        runtime.block_on(async {
            let cli = blob_client();
            // A 200 with the whole blob for a range request.
            state.lock().expect("state").ignore_range = true;
            let mut dst = vec![0xEEu8; 16];
            let longer = cli
                .get_range_into("odd", 0, &mut dst, &ReceiveProgress::new())
                .await;
            assert!(
                longer.is_err(),
                "a 1000-byte body for a 16-byte window succeeded"
            );
            let no_range = cli
                .get_range_into("odd", 100, &mut dst, &ReceiveProgress::new())
                .await;
            assert!(
                no_range.is_err(),
                "a 200 for a window at offset 100 succeeded"
            );
            // A 206 whose Content-Range was stripped.
            {
                let mut fake = state.lock().expect("state");
                fake.ignore_range = false;
                fake.strip_content_range = true;
            }
            let stripped = cli
                .get_range_into("odd", 100, &mut dst, &ReceiveProgress::new())
                .await;
            assert!(
                stripped.is_err(),
                "a 206 without Content-Range at offset 100 succeeded"
            );
            assert!(dst.iter().all(|b| *b == 0xEE), "nothing may be written");
        });
    }

    // ----------------------- a raw server for headers the fake cannot send -----------------------

    /// A server that answers every request with `head` (status line and headers) and `body`.
    fn canned_server(head: String, body: Vec<u8>) -> u16 {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut seen = Vec::new();
                let mut buf = [0u8; 1024];
                while !seen.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => seen.extend_from_slice(&buf[..n]),
                    }
                }
                let reply = format!(
                    "{head}content-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        port
    }

    fn range_error(content_range: &str, body_len: usize, start: u64, window: usize) -> String {
        let _env = AZURE_ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_azure_env();
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "1");
        let head = format!(
            "HTTP/1.1 206 Partial Content\r\ncontent-range: {content_range}\r\netag: \"0x1\"\r\n"
        );
        let port = canned_server(head, vec![7u8; body_len]);
        std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
        let runtime = client_runtime();
        runtime.block_on(async {
            let cli = blob_client();
            let mut dst = vec![0xEEu8; window];
            let result = cli
                .get_range_into("canned", start, &mut dst, &ReceiveProgress::new())
                .await;
            assert!(dst.iter().all(|b| *b == 0xEE), "nothing may be written");
            match result {
                Ok(n) => {
                    panic!("a hostile Content-Range {content_range:?} was accepted ({n} bytes)")
                }
                Err(e) => format!("{e:#}"),
            }
        })
    }

    /// A `Content-Range` that does not start where the window does would put the wrong bytes at
    /// the window: refused.
    #[test]
    fn get_range_into_refuses_a_content_range_that_does_not_start_at_the_offset() {
        let text = range_error("bytes 4-7/100", 4, 0, 4);
        assert!(text.contains("starts at byte 4, not 0"), "{text}");
    }

    // A `Content-Range: bytes 0-18446744073709551615/*` cannot be sent end to end in a debug
    // build: the Azure SDK itself overflows parsing it (`http_ranges.rs`) before s3dlio sees the
    // reply. s3dlio's own length arithmetic is checked, and unit-tested in `azure_client.rs`
    // (`a_hostile_content_range_never_overflows_the_length`).
}

// ───────────────────────────── R3: range engine ─────────────────────────────

mod r3 {
    use super::*;
    use bytes::BytesMut;
    use s3dlio::range_engine_generic::{RangeEngine, RangeEngineConfig};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const N: usize = 64 * MIB;

    fn engine() -> RangeEngine {
        RangeEngine::new(RangeEngineConfig {
            chunk_size: 4 * MIB,
            max_concurrent_ranges: 4,
            min_split_size: 8 * MIB as u64,
            range_timeout: Duration::from_secs(20),
        })
    }

    fn byte_at(position: usize) -> u8 {
        (position % 251) as u8
    }

    /// What the receive-into path records for every range it was asked to fill.
    type Seen = Arc<Mutex<Vec<(u64, usize, usize)>>>;

    /// A provider that receives straight into the segment it is given, like a network
    /// receive loop, from a blob of `blob_len` bytes (a range past the end is clamped, as a
    /// service clamps it). It records (offset, segment address, segment length).
    fn receiver(
        blob_len: usize,
        seen: Seen,
    ) -> impl Fn(u64, BytesMut) -> std::future::Ready<anyhow::Result<(BytesMut, usize)>>
           + Send
           + Sync
           + Clone
           + 'static {
        move |offset: u64, mut segment: BytesMut| {
            if let Ok(mut seen) = seen.lock() {
                seen.push((offset, segment.as_ptr() as usize, segment.len()));
            }
            let start = usize::try_from(offset).unwrap_or(usize::MAX);
            let available = blob_len.saturating_sub(start).min(segment.len());
            for (i, slot) in segment[..available].iter_mut().enumerate() {
                *slot = byte_at(start + i);
            }
            std::future::ready(Ok((segment, available)))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn every_range_is_received_into_its_own_window_of_the_master_buffer() {
        let seen: Seen = Arc::default();
        let ((bytes, stats), m) = measured(async {
            engine()
                .download_into_master(N as u64, false, receiver(N, seen.clone()), None)
                .await
                .expect("download")
        })
        .await;
        assert_eq!(bytes.len(), N);
        assert_eq!(stats.ranges_processed, 16);
        assert!(
            bytes.iter().enumerate().all(|(i, b)| *b == byte_at(i)),
            "wrong content"
        );
        // Pointer identity: every window the provider filled lies at its offset inside the
        // allocation that was returned, so no range was ever copied into the master.
        let base = bytes.as_ptr() as usize;
        let seen = seen.lock().expect("lock").clone();
        assert_eq!(seen.len(), 16);
        for (offset, ptr, len) in seen {
            assert_eq!(len, 4 * MIB);
            assert_eq!(
                ptr,
                base + offset as usize,
                "range at {offset} was received somewhere other than its window of the master"
            );
        }
        // And the allocation bound: the master buffer is the only large allocation.
        assert!(
            m.large < N + N / 8,
            "{} bytes were allocated in blocks of 1 MiB or more for a {N}-byte object",
            m.large
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn the_bytes_returning_path_allocates_every_range_again() {
        // Control (and the documented remaining copy for providers that can only return
        // `Bytes`: the file:// and direct:// stores until they receive into the window): each
        // range is allocated by the provider and copied into the master, so the large
        // allocations add up to about twice the object.
        let ((bytes, _), m) = measured(async {
            engine()
                .download(
                    N as u64,
                    |offset: u64, length: u64| async move {
                        let start = offset as usize;
                        let v: Vec<u8> = (start..start + length as usize).map(byte_at).collect();
                        Ok(Bytes::from(v))
                    },
                    None,
                )
                .await
                .expect("download")
        })
        .await;
        assert_eq!(bytes.len(), N);
        assert!(
            m.large >= N + N / 2,
            "RED: the control path allocated only {} large bytes, so the counter cannot tell a \
             per-range allocation from none",
            m.large
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_small_object_is_one_window_of_one_buffer() {
        let seen: Seen = Arc::default();
        let size = 3 * MIB + 5;
        let (bytes, stats) = engine()
            .download_into_master(size as u64, false, receiver(size, seen.clone()), None)
            .await
            .expect("download");
        assert_eq!(stats.ranges_processed, 1);
        assert_eq!(bytes.len(), size);
        assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte_at(i)));
        let seen = seen.lock().expect("lock").clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].1, bytes.as_ptr() as usize);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn the_tail_probe_detects_a_blob_that_grew_or_shrank_and_accepts_an_unchanged_one() {
        let planned = 20 * MIB + 3;
        for (blob_len, ok) in [
            (planned, true),
            (planned + 1, false),
            (planned + 4 * MIB, false),
            (planned - 1, false),
            (planned - 5 * MIB, false),
        ] {
            let seen: Seen = Arc::default();
            let result = engine()
                .download_into_master(planned as u64, true, receiver(blob_len, seen), None)
                .await;
            match (ok, result) {
                (true, Ok((bytes, _))) => {
                    assert_eq!(
                        bytes.len(),
                        planned,
                        "the probe byte is not part of the result"
                    );
                    assert!(bytes.iter().enumerate().all(|(i, b)| *b == byte_at(i)));
                }
                (false, Err(e)) => {
                    let text = format!("{e:#}");
                    assert!(text.contains("short/over read"), "unexpected error: {text}");
                }
                (true, Err(e)) => panic!("an unchanged blob failed: {e:#}"),
                (false, Ok(_)) => {
                    panic!("a blob of {blob_len} bytes (planned {planned}) was accepted")
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_fill_that_returns_a_different_buffer_is_an_error() {
        let result = engine()
            .download_into_master(
                16 * MIB as u64,
                false,
                |_offset: u64, segment: BytesMut| async move {
                    let n = segment.len();
                    Ok((BytesMut::zeroed(n), n))
                },
                None,
            )
            .await;
        let text = format!(
            "{:#}",
            result.expect_err("a copied-away segment must be refused")
        );
        assert!(
            text.contains("different buffer"),
            "unexpected error: {text}"
        );
    }
}

mod chunk_size_zero {
    use bytes::{Bytes, BytesMut};
    use s3dlio::range_engine_generic::{RangeEngine, RangeEngineConfig};
    use std::time::Duration;

    fn engine() -> RangeEngine {
        RangeEngine::new(RangeEngineConfig {
            chunk_size: 0,
            max_concurrent_ranges: 2,
            min_split_size: 1024,
            range_timeout: Duration::from_secs(5),
        })
    }

    /// A zero chunk size used to make `calculate_ranges` loop for ever, growing a vector of
    /// empty ranges until the process ran out of memory. It is an error, before any range.
    #[tokio::test(flavor = "current_thread")]
    async fn a_zero_chunk_size_is_an_error_and_never_a_hang() {
        let fetch = engine()
            .download(
                1 << 20,
                |_offset: u64, _length: u64| async move { Ok(Bytes::new()) },
                None,
            )
            .await;
        let text = format!("{:#}", fetch.expect_err("download with chunk_size 0"));
        assert!(text.contains("chunk_size"), "{text}");

        let into = engine()
            .download_into_master(
                1 << 20,
                false,
                |_offset: u64, window: BytesMut| async move {
                    let n = window.len();
                    Ok((window, n))
                },
                None,
            )
            .await;
        let text = format!(
            "{:#}",
            into.expect_err("download_into_master with chunk_size 0")
        );
        assert!(text.contains("chunk_size"), "{text}");
    }
}
