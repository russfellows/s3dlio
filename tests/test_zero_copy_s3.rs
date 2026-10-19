// tests/test_zero_copy_s3.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Stage G (Rule #1, zero copy everywhere), S3 paths:
//   W2  `put_object_multipart_*` -- an owned `Bytes` was handed to the sink as `&[u8]` and every
//       part was copied.
//   R2  S3 single / ranged GET -- `body.collect().into_bytes()` re-copies a multi-chunk body
//       (2x peak memory).
//   R4  `S3BytesDataset` range mode -- each range was copied again into the output buffer.
//
// Proof is an allocation bound against the in-process mock S3 server: a counting global
// allocator tracks the bytes allocated (W2: total) and the peak live bytes (R2, R4) during the
// call, on every thread EXCEPT the mock server's own (it collects request bodies and builds
// response bodies; that is the server, not the code under test). The allocator is process-wide,
// so the tests run one after another under a lock.
//
// The unit-level pointer-identity proof of the part assembly lives in `src/multipart.rs`
// (`zero_copy_tests`).

#![cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{
    ensure_mock_s3_server, key_path, on_mock_server_thread, unique_test_key, MockObject,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

#[allow(dead_code)] // shared helper; each test binary uses a subset
#[path = "common/counting_alloc.rs"]
mod counting_alloc;
use counting_alloc::Measured;

#[global_allocator]
static ALLOC: counting_alloc::CountingAlloc =
    counting_alloc::CountingAlloc::new(on_mock_server_thread);

const BUCKET: &str = "zc-bucket";
const MIB: usize = 1024 * 1024;

fn arm() {
    ALLOC.arm(usize::MAX);
}

/// Like `arm`, also counting allocations of at least `big` bytes.
fn arm_counting_big(big: usize) {
    ALLOC.arm(big);
}

fn disarm() -> Measured {
    ALLOC.disarm()
}

/// One measured test at a time: the allocator counts the whole process.
static SERIAL: Mutex<()> = Mutex::const_new(());

fn setup() -> common::mock_s3_server::MockS3State {
    let (_, state) = ensure_mock_s3_server();
    std::env::set_var("S3DLIO_LOAD_DOTENV", "0");
    std::env::set_var("AWS_EC2_METADATA_DISABLED", "true");
    state
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| {
            (i as u8)
                .wrapping_mul(131)
                .wrapping_add((i >> 8) as u8)
                .wrapping_add(seed)
        })
        .collect()
}

// ---- W2 ------------------------------------------------------------------------------------

/// 48 MiB of owned `Bytes` uploaded in 8 MiB parts: the parts are slices of the one allocation,
/// so the upload allocates a small constant, not the object size (a per-part copy allocates
/// every byte of it).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multipart_put_of_owned_bytes_allocates_no_part_copies() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let key = unique_test_key("zc-mpu-put");
    let uri = format!("s3://{BUCKET}/{key}");
    let n = 48 * MIB;
    let data = Bytes::from(pattern(n, 1));

    // Warm up the client / runtime outside the measured window with a 1-part upload.
    let warm = unique_test_key("zc-mpu-warm");
    s3dlio::s3_utils::put_object_multipart_uri_async(
        &format!("s3://{BUCKET}/{warm}"),
        Bytes::from(vec![7u8; 5 * MIB]),
        Some(5 * MIB),
    )
    .await
    .expect("warm-up multipart put");

    arm();
    let result =
        s3dlio::s3_utils::put_object_multipart_uri_async(&uri, data.clone(), Some(8 * MIB)).await;
    let m = disarm();
    result.expect("multipart put");

    let counts = state.counts_for(&key_path(BUCKET, &key));
    assert_eq!(counts.upload_part_calls, 6, "48 MiB / 8 MiB parts");
    assert_eq!(counts.complete_calls, 1);
    assert!(
        m.total < 8 * MIB,
        "multipart put of {n} owned bytes allocated {} bytes: parts were copied \
         (zero-copy allows a small constant, far below one part)",
        m.total
    );
}

// ---- R2 ------------------------------------------------------------------------------------

const GET_LEN: usize = 32 * MIB;

/// Peak live allocation must stay near the object size (the one destination buffer plus a few
/// in-flight network chunks). `collect().into_bytes()` holds every chunk AND the re-copy: ~2x.
fn assert_one_buffer(what: &str, m: &Measured) {
    assert!(
        m.peak < GET_LEN + GET_LEN / 2,
        "{what}: peak live allocation {} bytes for a {GET_LEN}-byte body: the body was \
         received and then copied again (zero-copy peaks near 1x)",
        m.peak
    );
    assert_eq!(
        m.big_allocs, 1,
        "{what}: exactly one big allocation (the destination) was expected, got {}: the body \
         was staged in another large buffer",
        m.big_allocs
    );
}

fn register(state: &common::mock_s3_server::MockS3State, prefix: &str) -> (String, Vec<u8>) {
    let key = unique_test_key(prefix);
    let body = pattern(GET_LEN, 3);
    state.put_object(BUCKET, &key, MockObject::new(body.clone()));
    (format!("s3://{BUCKET}/{key}"), body)
}

async fn warm_get(state: &common::mock_s3_server::MockS3State) {
    let key = unique_test_key("zc-warm-obj");
    state.put_object(BUCKET, &key, MockObject::new(vec![1u8; 4096]));
    s3dlio::s3_utils::get_object_uri_async(&format!("s3://{BUCKET}/{key}"))
        .await
        .expect("warm-up get");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_get_object_receives_into_one_buffer() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let (uri, body) = register(&state, "zc-r2-whole");
    warm_get(&state).await;

    arm_counting_big(GET_LEN / 2);
    let got = s3dlio::s3_utils::get_object_uri_async(&uri).await;
    let m = disarm();
    let got = got.expect("get");

    assert_eq!(got.len(), GET_LEN);
    assert!(got[..] == body[..], "bytes differ");
    assert_one_buffer("get_object", &m);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_get_object_range_receives_into_one_buffer() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let (uri, body) = register(&state, "zc-r2-range");
    warm_get(&state).await;

    arm_counting_big(GET_LEN / 2);
    let got = s3dlio::s3_utils::get_object_range_uri_async(&uri, 0, Some(GET_LEN as u64)).await;
    let m = disarm();
    let got = got.expect("range get");

    assert_eq!(got.len(), GET_LEN);
    assert!(got[..] == body[..], "bytes differ");
    assert_one_buffer("get_object_range_uri_async", &m);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_get_object_range_open_ended_receives_into_one_buffer() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let (uri, body) = register(&state, "zc-r2-open");
    warm_get(&state).await;

    arm_counting_big(GET_LEN / 2);
    let got = s3dlio::s3_utils::get_object_range_uri_async(&uri, 1024, None).await;
    let m = disarm();
    let got = got.expect("open-ended range get");

    assert_eq!(got.len(), GET_LEN - 1024);
    assert!(got[..] == body[1024..], "bytes differ");
    assert_one_buffer("open-ended get_object_range_uri_async", &m);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_timed_range_get_receives_into_one_buffer() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let (uri, body) = register(&state, "zc-r2-timed");
    warm_get(&state).await;

    arm_counting_big(GET_LEN / 2);
    let got =
        s3dlio::s3_utils::get_object_range_uri_timed_async(&uri, 0, Some(GET_LEN as u64)).await;
    let m = disarm();
    let (got, _ttfb, _transfer) = got.expect("timed range get");

    assert_eq!(got.len(), GET_LEN);
    assert!(got[..] == body[..], "bytes differ");
    assert_one_buffer("get_object_range_uri_timed_async", &m);
}

// ---- R4 ------------------------------------------------------------------------------------

/// `S3BytesDataset` in range mode: every range is received straight into its segment of one
/// pre-sized buffer. The old code received each range into its own `Bytes` and copied it into
/// the output buffer, so the in-flight ranges and the output were live together (~2x) and every
/// byte was copied twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_dataset_range_mode_receives_into_the_output_buffer() {
    use s3dlio::data_loader::options::{LoaderOptions, ReaderMode};
    use s3dlio::data_loader::{Dataset, S3BytesDataset};

    let _serial = SERIAL.lock().await;
    let state = setup();
    let bucket = format!("zc-ds-{}", unique_test_key("b"));
    let key = "obj/sample.bin";
    let body = pattern(GET_LEN, 5);
    state.put_object(&bucket, key, MockObject::new(body.clone()));
    state.seed_bucket(&bucket, &[key]);
    warm_get(&state).await;

    let opts = LoaderOptions {
        reader_mode: ReaderMode::Range,
        part_size: GET_LEN / 4,
        max_inflight_parts: 4,
        ..LoaderOptions::default()
    };
    let uri = format!("s3://{bucket}/obj/");
    let ds =
        tokio::task::spawn_blocking(move || S3BytesDataset::from_prefix_with_opts(&uri, &opts))
            .await
            .expect("join")
            .expect("dataset");

    arm_counting_big(GET_LEN / 4);
    let got = ds.get(0).await;
    let m = disarm();
    let got = got.expect("range-mode get");

    assert_eq!(got.len(), GET_LEN);
    assert!(got[..] == body[..], "bytes differ");
    eprintln!("R4 range mode: peak {} total {}", m.peak, m.total);
    assert!(
        m.peak < GET_LEN + GET_LEN / 2,
        "range mode: peak {} for a {GET_LEN}-byte body",
        m.peak
    );
    // Exactly ONE allocation as large as a part: the master buffer. A copy of any segment out
    // and back (or the old per-part `Bytes`) would add one allocation per part.
    assert_eq!(
        m.big_allocs, 1,
        "range mode must receive every part straight into the master buffer"
    );
}

// ---- W3 on the wire (signed) and abort latency ---------------------------------------------

fn clean_signing_env() {
    // `S3DLIO_UNSIGNED_PAYLOAD` is latched per process: this binary tests the SIGNED path.
    std::env::remove_var("S3DLIO_UNSIGNED_PAYLOAD");
}

/// Owned and borrowed writes that straddle part boundaries, signed upload: the object the server
/// receives is byte-exact, every part is a contiguous `Content-Length` body of exactly one part
/// (the last one shorter), and the payload is signed (a SHA-256, not UNSIGNED-PAYLOAD).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn signed_multipart_with_straddling_writes_arrives_byte_exact() {
    use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
    let _serial = SERIAL.lock().await;
    clean_signing_env();
    let state = setup();
    let key = unique_test_key("zc-signed-straddle");
    let a = pattern(7 * MIB, 1);
    let b = pattern(7 * MIB + 1, 2);
    let c = pattern(3 * MIB, 3);
    let expected: Vec<u8> = [&a[..], &b[..], &c[..]].concat();
    let (a2, b2, c2, key2) = (Bytes::from(a), Bytes::from(b), c, key.clone());
    tokio::task::spawn_blocking(move || {
        let cfg = MultipartUploadConfig {
            part_size: 5 * MIB,
            max_in_flight: Some(4),
            ..Default::default()
        };
        let mut sink = MultipartUploadSink::new(BUCKET, &key2, cfg).expect("sink");
        sink.write_bytes_blocking(a2).expect("write a");
        sink.write_bytes_blocking(b2).expect("write b");
        sink.write_blocking(&c2).expect("write c");
        sink.finish_blocking().expect("finish");
    })
    .await
    .expect("join");

    let mut parts = state.part_records(BUCKET, &key);
    parts.sort_by_key(|p| p.part_number);
    let sizes: Vec<usize> = parts.iter().map(|p| p.body.len()).collect();
    assert_eq!(sizes, vec![5 * MIB, 5 * MIB, 5 * MIB, 2 * MIB + 1]);
    let joined: Vec<u8> = parts.iter().flat_map(|p| p.body.iter().copied()).collect();
    assert!(joined == expected, "object bytes differ");
    for p in &parts {
        assert_eq!(p.content_length, Some(p.body.len() as u64));
        assert_eq!(p.transfer_encoding, None);
        let sha = p.content_sha256.as_deref().unwrap_or("");
        assert!(
            sha.len() == 64 && sha != "UNSIGNED-PAYLOAD",
            "signed payload expected, got {sha:?}"
        );
    }
}

/// Probe whose drop shows that every clone of a part's bytes is gone.
struct DropProbe {
    data: Vec<u8>,
    gone: Arc<AtomicBool>,
}
impl AsRef<[u8]> for DropProbe {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.gone.store(true, Ordering::SeqCst);
    }
}

/// UploadPart requests that the server stalls for 30 s: `abort_blocking()` must return at once,
/// send AbortMultipartUpload, and the in-flight part tasks must be CANCELLED so they drop their
/// clones of the (caller's, in Python) buffer within seconds. The old coordinator detached the
/// tasks, which then held the buffer until the stalled requests finished.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn abort_cancels_stalled_part_uploads_and_releases_the_buffer() {
    use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
    let _serial = SERIAL.lock().await;
    clean_signing_env();
    let state = setup();
    let key = unique_test_key("zc-part-stalls");
    let path = key_path(BUCKET, &key);
    let gone = Arc::new(AtomicBool::new(false));
    let probe = DropProbe {
        data: vec![7u8; 10 * MIB],
        gone: gone.clone(),
    };

    let (key2, st) = (key.clone(), state.clone());
    let (path2, gone2) = (path.clone(), gone.clone());
    let elapsed = tokio::task::spawn_blocking(move || {
        let cfg = MultipartUploadConfig {
            part_size: 5 * MIB,
            max_in_flight: Some(4),
            ..Default::default()
        };
        let mut sink = MultipartUploadSink::new(BUCKET, &key2, cfg).expect("sink");
        sink.write_bytes_blocking(Bytes::from_owner(probe))
            .expect("write");
        // Both parts reach the server and stall there.
        let t = Instant::now();
        while st.counts_for(&path2).upload_part_calls < 2 {
            assert!(t.elapsed() < Duration::from_secs(10), "parts never arrived");
            std::thread::sleep(Duration::from_millis(20));
        }
        let t = Instant::now();
        sink.abort_blocking().expect("abort");
        let elapsed = t.elapsed();
        drop(sink);
        // The aborted part tasks drop their clones promptly (not after the 30 s stall).
        let t = Instant::now();
        while !gone2.load(Ordering::SeqCst) {
            assert!(
                t.elapsed() < Duration::from_secs(5),
                "the parts' bytes are still referenced 5 s after abort: tasks were detached, \
                 not cancelled"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        elapsed
    })
    .await
    .expect("join");

    assert!(elapsed < Duration::from_secs(5), "abort took {elapsed:?}");
    assert_eq!(state.counts_for(&path).abort_calls, 1);
    assert_eq!(state.counts_for(&path).complete_calls, 0);
}
