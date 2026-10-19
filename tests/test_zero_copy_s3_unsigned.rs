// tests/test_zero_copy_s3_unsigned.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Stage G (Rule #1), multipart upload with payload signing OFF (`S3DLIO_UNSIGNED_PAYLOAD=1`):
// a part that straddles two `write()` calls is NOT gathered into one buffer. It is sent as a
// multi-chunk request body built from the refcounted slices of the writes, with a known
// `Content-Length` and `x-amz-content-sha256: UNSIGNED-PAYLOAD`.
//
// The flag is latched once per process, so this lives in its own test binary (the signed path is
// in `test_zero_copy_s3.rs`). Proof:
//   * a counting allocator: the whole upload allocates a small constant, far less than one part
//     (a gather would allocate every straddling part);
//   * the server receives the exact bytes, one contiguous `Content-Length` body per part, no
//     chunked transfer encoding, no payload signature.
// (The pointer identity of each chunk of a straddling part is in the `src/multipart.rs` unit
// tests.)

#![cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod common;

use bytes::Bytes;
use common::mock_s3_server::{ensure_mock_s3_server, on_mock_server_thread, unique_test_key};
use tokio::sync::Mutex;

#[allow(dead_code)] // shared helper; each test binary uses a subset
#[path = "common/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOC: counting_alloc::CountingAlloc =
    counting_alloc::CountingAlloc::new(on_mock_server_thread);

const BUCKET: &str = "zc-unsigned-bucket";
const MIB: usize = 1024 * 1024;

static SERIAL: Mutex<()> = Mutex::const_new(());

fn setup() -> common::mock_s3_server::MockS3State {
    std::env::set_var("S3DLIO_UNSIGNED_PAYLOAD", "1");
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

fn assert_unsigned_exact(
    state: &common::mock_s3_server::MockS3State,
    key: &str,
    expected: &[u8],
    part_size: usize,
) {
    let mut parts = state.part_records(BUCKET, key);
    parts.sort_by_key(|p| p.part_number);
    let (last, whole) = parts.split_last().expect("parts");
    assert!(whole.iter().all(|p| p.body.len() == part_size));
    assert!(last.body.len() <= part_size && !last.body.is_empty());
    let joined: Vec<u8> = parts.iter().flat_map(|p| p.body.iter().copied()).collect();
    assert!(joined == expected, "object bytes differ");
    for p in &parts {
        assert_eq!(
            p.content_length,
            Some(p.body.len() as u64),
            "part {} must carry its Content-Length",
            p.part_number
        );
        assert_eq!(
            p.transfer_encoding, None,
            "part {} was chunked",
            p.part_number
        );
        assert_eq!(
            p.content_sha256.as_deref(),
            Some("UNSIGNED-PAYLOAD"),
            "part {} must be sent unsigned",
            p.part_number
        );
    }
}

/// Two owned writes whose parts straddle (7 MiB + 7 MiB + 1 byte in 5 MiB parts), signing off.
/// Returns the key, what the upload allocated, and the expected object.
async fn run_straddling_upload() -> (String, counting_alloc::Measured, Vec<u8>) {
    use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
    let key = unique_test_key("zc-unsigned-straddle");
    let a = pattern(7 * MIB, 1);
    let b = pattern(7 * MIB + 1, 2);
    let expected: Vec<u8> = [&a[..], &b[..]].concat();
    let (a2, b2, key2) = (Bytes::from(a), Bytes::from(b), key.clone());

    // Warm up the client / runtime outside the measured window.
    let warm = unique_test_key("zc-unsigned-warm");
    tokio::task::spawn_blocking(move || {
        let cfg = MultipartUploadConfig {
            part_size: 5 * MIB,
            ..Default::default()
        };
        let mut sink = MultipartUploadSink::new(BUCKET, &warm, cfg).expect("sink");
        sink.write_bytes_blocking(Bytes::from(vec![1u8; 5 * MIB]))
            .expect("write");
        sink.finish_blocking().expect("finish");
    })
    .await
    .expect("join");

    ALLOC.arm(usize::MAX);
    tokio::task::spawn_blocking(move || {
        let cfg = MultipartUploadConfig {
            part_size: 5 * MIB,
            max_in_flight: Some(4),
            ..Default::default()
        };
        let mut sink = MultipartUploadSink::new(BUCKET, &key2, cfg).expect("sink");
        sink.write_bytes_blocking(a2).expect("write a");
        sink.write_bytes_blocking(b2).expect("write b");
        sink.finish_blocking().expect("finish");
    })
    .await
    .expect("join");
    (key, ALLOC.disarm(), expected)
}

/// The straddling part reaches the server byte-exact as ONE `Content-Length` body, unsigned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn straddling_parts_are_sent_unsigned_with_a_content_length() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let (key, _, expected) = run_straddling_upload().await;
    assert_unsigned_exact(&state, &key, &expected, 5 * MIB);
}

/// STRICT zero-copy bound: the upload allocates a small constant, less than one part. The
/// straddling part reaches the wire as its original slices: the sink builds a multi-chunk body
/// and the connector (`src/reqwest_client.rs`) streams those segments instead of gathering them
/// (it used to collect the body into one buffer: one part-sized copy, ~5.6 MB here).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn straddling_parts_allocate_nothing_end_to_end() {
    let _serial = SERIAL.lock().await;
    setup();
    let (_, m, _) = run_straddling_upload().await;
    assert!(
        m.total < 2 * MIB,
        "unsigned multipart of straddling writes allocated {} bytes: a straddling part was \
         gathered into a new buffer",
        m.total
    );
}

/// The same through `put_object_multipart_uri_async` (the W2 entry point).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unsigned_multipart_put_of_owned_bytes_is_exact_and_unsigned() {
    let _serial = SERIAL.lock().await;
    let state = setup();
    let key = unique_test_key("zc-unsigned-put");
    let data = pattern(12 * MIB + 5, 4);
    s3dlio::s3_utils::put_object_multipart_uri_async(
        &format!("s3://{BUCKET}/{key}"),
        Bytes::from(data.clone()),
        Some(5 * MIB),
    )
    .await
    .expect("put");
    assert_unsigned_exact(&state, &key, &data, 5 * MIB);
}

/// `6 MiB` written in `chunk`-byte owned writes (5 MiB parts), unsigned; the object must arrive
/// byte-exact as `Content-Length` bodies whichever way the sink assembled the parts.
async fn run_small_writes(chunk: usize, tag: &str) {
    use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
    let _serial = SERIAL.lock().await;
    let state = setup();
    let key = unique_test_key(tag);
    let data = pattern(6 * MIB, 7);
    let (d2, key2) = (data.clone(), key.clone());
    tokio::task::spawn_blocking(move || {
        let cfg = MultipartUploadConfig {
            part_size: 5 * MIB,
            ..Default::default()
        };
        let mut sink = MultipartUploadSink::new(BUCKET, &key2, cfg).expect("sink");
        for piece in d2.chunks(chunk) {
            sink.write_bytes_blocking(Bytes::copy_from_slice(piece))
                .expect("write");
        }
        sink.finish_blocking().expect("finish");
    })
    .await
    .expect("join");
    assert_unsigned_exact(&state, &key, &data, 5 * MIB);
}

/// Writes below the unsigned threshold (2 KiB) are coalesced into contiguous part buffers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_below_the_unsigned_threshold_are_coalesced() {
    run_small_writes(1024, "zc-unsigned-small").await;
}

/// Writes at or above it are kept as slices: each part is a multi-chunk body of 1280 chunks,
/// still byte-exact with its Content-Length.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_above_the_unsigned_threshold_are_sent_as_chunk_lists() {
    run_small_writes(4096, "zc-unsigned-chunks").await;
}
