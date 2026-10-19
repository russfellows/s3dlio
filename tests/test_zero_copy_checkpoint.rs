// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Stage G (RULE #1, zero copy): the checkpoint writer hands the caller's `Bytes` to the
//! object store without copying it (W1).
//!
//! The proof is pointer identity: a recording `ObjectStore` notes the address range of the
//! `Bytes` it receives in `put` / `put_multipart`, and the test checks that range lies inside
//! the caller's allocation. A `Bytes::copy_from_slice` in the writer lands in a fresh
//! allocation and fails the check.
//!
//! The allocation-bound tests (R9 NPZ read, R10 zstd shard decode) use a counting global
//! allocator whose counters are per thread and only armed around the call under test, so the
//! other tests of this binary cannot disturb them; they run on the test thread (current-thread
//! runtime, in-memory store), so every allocation of the call is counted.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, Stream};
use s3dlio::checkpoint::{KeyLayout, Reader, Writer};
use s3dlio::object_store::{ObjectMetadata, ObjectStore, ObjectWriter};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Mutex;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
    static REALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Counts the bytes requested from the allocator by the ARMED thread only.
struct CountingAlloc;

fn count(bytes: usize, realloc: bool) {
    let _ = ARMED.try_with(|armed| {
        if armed.get() {
            let _ = ALLOCATED.try_with(|a| a.set(a.get() + bytes));
            if realloc {
                let _ = REALLOCS.try_with(|r| r.set(r.get() + 1));
            }
        }
    });
}

// SAFETY: every method forwards to `System` unchanged; the counters only observe sizes.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size(), false);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size(), false);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size, true);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn arm() {
    let _ = ALLOCATED.try_with(|a| a.set(0));
    let _ = REALLOCS.try_with(|r| r.set(0));
    let _ = ARMED.try_with(|a| a.set(true));
}

/// Stop counting; returns (bytes requested, number of reallocs) since `arm()`.
fn disarm() -> (usize, usize) {
    let _ = ARMED.try_with(|a| a.set(false));
    (
        ALLOCATED.try_with(Cell::get).unwrap_or(0),
        REALLOCS.try_with(Cell::get).unwrap_or(0),
    )
}

/// One payload the store received: its address range and which entry point delivered it.
#[derive(Clone, Debug)]
struct Received {
    uri: String,
    ptr: usize,
    len: usize,
    multipart: bool,
}

#[derive(Default)]
struct RecordingStore {
    received: Mutex<Vec<Received>>,
    /// What `get` answers (the stored, e.g. zstd-compressed, object); `None` = not found.
    canned_get: Option<Bytes>,
}

impl RecordingStore {
    fn note(&self, uri: &str, data: &Bytes, multipart: bool) {
        if let Ok(mut v) = self.received.lock() {
            v.push(Received {
                uri: uri.to_string(),
                ptr: data.as_ptr() as usize,
                len: data.len(),
                multipart,
            });
        }
    }

    fn received(&self) -> Vec<Received> {
        self.received.lock().map(|v| v.clone()).unwrap_or_default()
    }
}

fn metadata(size: u64) -> ObjectMetadata {
    ObjectMetadata {
        size,
        last_modified: None,
        e_tag: None,
        content_type: None,
        content_language: None,
        content_encoding: None,
        cache_control: None,
        content_disposition: None,
        expires: None,
        storage_class: None,
        server_side_encryption: None,
        ssekms_key_id: None,
        sse_customer_algorithm: None,
        version_id: None,
        replication_status: None,
        metadata: HashMap::new(),
    }
}

#[async_trait]
impl ObjectStore for RecordingStore {
    async fn get(&self, _uri: &str) -> Result<Bytes> {
        self.canned_get
            .clone()
            .ok_or_else(|| anyhow!("recording store: get unused"))
    }
    async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
        Err(anyhow!("recording store: get_range unused"))
    }
    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        self.note(uri, &data, false);
        Ok(())
    }
    async fn put_multipart(&self, uri: &str, data: Bytes, _part: Option<usize>) -> Result<()> {
        self.note(uri, &data, true);
        Ok(())
    }
    async fn list(&self, _prefix: &str, _recursive: bool) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn list_stream<'a>(
        &'a self,
        _prefix: &'a str,
        _recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(stream::empty())
    }
    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let size = self
            .received()
            .iter()
            .rev()
            .find(|r| r.uri == uri)
            .map(|r| r.len as u64)
            .ok_or_else(|| anyhow!("recording store: {uri} was never put"))?;
        Ok(metadata(size))
    }
    async fn delete(&self, _uri: &str) -> Result<()> {
        Ok(())
    }
    async fn delete_batch(&self, _uris: &[String]) -> Result<()> {
        Ok(())
    }
    async fn delete_prefix(&self, _prefix: &str) -> Result<()> {
        Ok(())
    }
    async fn create_container(&self, _name: &str) -> Result<()> {
        Ok(())
    }
    async fn delete_container(&self, _name: &str) -> Result<()> {
        Ok(())
    }
    async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
        Err(anyhow!("recording store: get_writer unused"))
    }
}

/// A distinctive 1 MiB payload (so a copy cannot alias the caller's allocation by accident).
fn payload() -> Bytes {
    let v: Vec<u8> = (0..(1usize << 20)).map(|i| (i % 251) as u8).collect();
    Bytes::from(v)
}

/// The one shard the store received (the manifest/latest writes are small JSON, never this size).
fn shard_put(store: &RecordingStore, len: usize) -> Received {
    let hits: Vec<Received> = store
        .received()
        .into_iter()
        .filter(|r| r.len == len)
        .collect();
    assert_eq!(hits.len(), 1, "expected exactly one shard put: {hits:?}");
    hits[0].clone()
}

fn assert_inside(r: &Received, buf: &Bytes, what: &str) {
    let start = buf.as_ptr() as usize;
    let end = start + buf.len();
    assert!(
        r.ptr >= start && r.ptr + r.len <= end,
        "{what}: the store received a COPY (store got {:#x}+{}, caller's buffer is {:#x}+{})",
        r.ptr,
        r.len,
        start,
        buf.len()
    );
}

#[tokio::test]
async fn put_shard_hands_the_callers_bytes_to_put() -> Result<()> {
    let store = RecordingStore::default();
    let writer = Writer::new(&store, "file:///zc".to_string(), 1, 0);
    let buf = payload();
    let layout = KeyLayout::new("file:///zc".to_string(), 1);
    let meta = writer.put_shard(&layout, buf.clone()).await?;
    assert_eq!(meta.size, buf.len() as u64);
    let got = shard_put(&store, buf.len());
    assert!(!got.multipart);
    assert_inside(&got, &buf, "put_shard/put");
    Ok(())
}

#[tokio::test]
async fn put_shard_hands_the_callers_bytes_to_put_multipart() -> Result<()> {
    let store = RecordingStore::default();
    let writer = Writer::new(&store, "file:///zc".to_string(), 1, 0).with_multipart_threshold(1024);
    let buf = payload();
    let layout = KeyLayout::new("file:///zc".to_string(), 2);
    writer.put_shard(&layout, buf.clone()).await?;
    let got = shard_put(&store, buf.len());
    assert!(got.multipart, "payload above the threshold goes multipart");
    assert_inside(&got, &buf, "put_shard/put_multipart");
    Ok(())
}

#[tokio::test]
async fn put_shard_with_key_hands_the_callers_bytes_to_the_store() -> Result<()> {
    for threshold in [usize::MAX, 1024] {
        let store = RecordingStore::default();
        let writer =
            Writer::new(&store, "file:///zc".to_string(), 1, 0).with_multipart_threshold(threshold);
        let buf = payload();
        writer
            .put_shard_with_key("custom/one.bin", buf.clone())
            .await?;
        let got = shard_put(&store, buf.len());
        assert_eq!(got.multipart, threshold == 1024);
        assert_inside(&got, &buf, "put_shard_with_key");
    }
    Ok(())
}

#[tokio::test]
async fn save_checkpoint_and_distributed_shard_do_not_copy_the_shard() -> Result<()> {
    let buf = payload();

    let store = RecordingStore::default();
    let writer = Writer::new(&store, "file:///zc".to_string(), 1, 0);
    writer
        .save_checkpoint(3, 1, "torch", buf.clone(), None)
        .await?;
    assert_inside(&shard_put(&store, buf.len()), &buf, "save_checkpoint");

    let store = RecordingStore::default();
    let writer = Writer::new(&store, "file:///zc".to_string(), 2, 1);
    writer
        .save_distributed_shard(4, 1, "torch", buf.clone())
        .await?;
    assert_inside(
        &shard_put(&store, buf.len()),
        &buf,
        "save_distributed_shard",
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// R10: zstd shard decode is pre-sized from the frame header (no doubling growth).
// ---------------------------------------------------------------------------

/// 16 MiB of compressible, non-trivial data.
fn compressible(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i / 7) % 61) as u8).collect()
}

#[tokio::test(flavor = "current_thread")]
async fn zstd_shard_decode_is_presized_from_the_frame_content_size() -> Result<()> {
    let plain = compressible(16 << 20);
    // `bulk::compress` records the content size in the frame header.
    let frame = zstd::bulk::compress(&plain, 1)?;
    let store = RecordingStore {
        canned_get: Some(Bytes::from(frame)),
        ..Default::default()
    };
    let reader = Reader::new(&store, "file:///zc".to_string());

    arm();
    let out = reader.read_shard("shards/rank0.bin.zst").await;
    let (allocated, reallocs) = disarm();
    let out = out?;

    assert_eq!(out.len(), plain.len());
    assert_eq!(&out[..], &plain[..]);
    // The decoded buffer is allocated once at its final size: ~1x the output, never the ~2x of
    // a Vec that grows by doubling, and no realloc of the output buffer.
    assert!(
        allocated < plain.len() + (plain.len() >> 3),
        "zstd decode allocated {allocated} bytes for a {} byte shard ({reallocs} reallocs): \
         the output Vec grew instead of being pre-sized",
        plain.len()
    );
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn zstd_shard_decode_ignores_an_implausible_frame_content_size() -> Result<()> {
    // A frame header may lie (corrupt or hostile): a claimed size beyond what zstd can expand
    // must not drive a huge reservation, and the decode must still be correct.
    // Hand-built frame: magic, descriptor 0xE0 (8-byte content size, single segment), a claimed
    // size of 2^50, then one last RAW block of 1000 bytes (the real content).
    let body: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0xE0];
    frame.extend_from_slice(&(1u64 << 50).to_le_bytes());
    let block_header = ((body.len() as u32) << 3) | 1; // last block, type raw
    frame.extend_from_slice(&block_header.to_le_bytes()[..3]);
    frame.extend_from_slice(&body);
    let store = RecordingStore {
        canned_get: Some(Bytes::from(frame)),
        ..Default::default()
    };
    let reader = Reader::new(&store, "file:///zc".to_string());
    arm();
    let out = reader.read_shard("shards/rank0.bin.zst").await;
    let (allocated, _) = disarm();
    // Either the lying header is ignored and decode succeeds, or decode reports an error; it
    // must never reserve the claimed 1 PiB.
    assert!(allocated < (1usize << 32), "allocated {allocated}");
    if let Ok(out) = out {
        assert_eq!(&out[..], &body[..]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// R9: a STORED NPZ entry is read in place; only the f32 decode copies.
// ---------------------------------------------------------------------------

#[test]
fn read_npz_array_of_a_stored_entry_copies_the_data_once() -> Result<()> {
    use ndarray::ArrayD;
    use s3dlio::data_formats::npz::{build_multi_npz, read_npz_array};

    let elements = 1usize << 20; // 4 MiB of f32
    let values: Vec<f32> = (0..elements).map(|i| i as f32 * 0.5).collect();
    let arr = ArrayD::from_shape_vec(vec![elements], values.clone())?;
    let npz = build_multi_npz(vec![("data", &arr)])?;
    let array_bytes = elements * 4;

    arm();
    let got = read_npz_array(&npz, "data");
    let (allocated, _) = disarm();
    let got = got?;

    assert_eq!(got.shape(), &[elements]);
    assert_eq!(got.as_slice().map(|s| s[12345]), Some(12345.0 * 0.5));
    // One allocation for the decoded numpy-owned f32 array (the format-decode copy). The extra
    // `read_to_end` copy of the entry would add >= 1x on top (more with its doubling growth).
    assert!(
        allocated < array_bytes + (array_bytes >> 3),
        "read_npz_array allocated {allocated} bytes for a {array_bytes} byte array: \
         the stored entry was copied before decoding"
    );
    Ok(())
}

#[test]
fn npz_entry_of_a_stored_member_points_into_the_archive() -> Result<()> {
    use ndarray::ArrayD;
    use s3dlio::data_formats::npz::{build_multi_npz, npz_entry};
    use std::borrow::Cow;

    let arr = ArrayD::from_shape_vec(vec![4], vec![1.0f32, 2.0, 3.0, 4.0])?;
    let npz = build_multi_npz(vec![("w", &arr)])?;
    let entry = npz_entry(&npz, "w")?;
    let range = npz.as_ptr() as usize..npz.as_ptr() as usize + npz.len();
    assert!(matches!(entry, Cow::Borrowed(_)), "stored entry was copied");
    assert!(range.contains(&(entry.as_ptr() as usize)));
    assert!(npz_entry(&npz, "missing").is_err());
    Ok(())
}

// ---------------------------------------------------------------------------
// Fix round: compressed saves from a slice, NPZ integrity, hostile headers.
// ---------------------------------------------------------------------------

/// A compressing `put_shard` / `put_shard_with_key` only needs a slice (the zstd encoder is a
/// transform), so a borrowed `&[u8]` caller must not pay the compat copy `into_shard_bytes`
/// makes for the uncompressed path.
#[tokio::test(flavor = "current_thread")]
async fn compressed_put_shard_from_a_slice_does_not_copy_the_payload() -> Result<()> {
    use s3dlio::object_store::{store_for_uri, CompressionConfig};

    let dir = tempfile::TempDir::new()?;
    let base = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base)?;
    let writer =
        Writer::new(&*store, base.clone(), 1, 0).with_compression(CompressionConfig::zstd_level(1));
    let plain = vec![0x5Au8; 16 << 20]; // compresses to a few KiB
    let layout = KeyLayout::new(base.clone(), 9);

    arm();
    let meta = writer.put_shard(&layout, &plain[..]).await;
    let (allocated, _) = disarm();
    let meta = meta?;
    assert!(meta.key.ends_with(".zst"), "{meta:?}");
    assert!(
        allocated < (4 << 20),
        "compressed put_shard from &[u8] allocated {allocated} bytes for a {} byte payload: \
         the slice was copied before compression",
        plain.len()
    );

    arm();
    let meta = writer.put_shard_with_key("custom/k.bin", &plain[..]).await;
    let (allocated, _) = disarm();
    meta?;
    assert!(
        allocated < (4 << 20),
        "compressed put_shard_with_key from &[u8] allocated {allocated} bytes"
    );
    Ok(())
}

fn flipped_npz() -> Result<(bytes::Bytes, bytes::Bytes)> {
    use ndarray::ArrayD;
    use s3dlio::data_formats::npz::build_multi_npz;

    let arr = ArrayD::from_shape_vec(vec![64], (0..64).map(|i| i as f32).collect())?;
    let clean = build_multi_npz(vec![("w", &arr)])?;
    // Flip one byte inside the f32 payload (the last bytes of the member data).
    let mut bad = clean.to_vec();
    let at = bad
        .windows(4)
        .position(|w| w == 62.0f32.to_le_bytes())
        .ok_or_else(|| anyhow!("payload not found"))?;
    bad[at] ^= 0x40;
    Ok((clean, bytes::Bytes::from(bad)))
}

#[test]
fn stored_npz_entry_with_a_flipped_payload_byte_fails_the_crc_check() -> Result<()> {
    use s3dlio::data_formats::npz::{npz_entry, read_npz_array};

    let (clean, bad) = flipped_npz()?;
    assert!(read_npz_array(&clean, "w").is_ok(), "a clean archive reads");
    assert!(npz_entry(&clean, "w").is_ok());

    let err = read_npz_array(&bad, "w").expect_err("corrupted payload must be refused");
    assert!(format!("{err:#}").contains("Invalid checksum"), "{err:#}");
    let err = npz_entry(&bad, "w").expect_err("corrupted payload must be refused");
    assert!(format!("{err:#}").contains("Invalid checksum"), "{err:#}");
    Ok(())
}

#[test]
fn hostile_npy_shape_header_is_an_error_not_a_panic() {
    use s3dlio::data_formats::npz::read_npy_bytes;

    let parse = |shape: &str| {
        let header = format!("{{'descr': '<f4', 'fortran_order': False, {shape} }}\n");
        let mut npy = b"\x93NUMPY\x01\x00".to_vec();
        npy.extend_from_slice(&(header.len() as u16).to_le_bytes());
        npy.extend_from_slice(header.as_bytes());
        npy.extend_from_slice(&[0u8; 16]);
        std::panic::catch_unwind(|| read_npy_bytes(&npy).is_ok())
    };

    // ')' before '(' used to slice [13..9] and panic; a well-formed tuple after the junk is
    // still readable, but it must never panic.
    assert!(parse("'shape': ), (3,)").is_ok(), "')' before '(' panicked");
    for shape in [
        "'shape': ) (",                                        // no ')' after the '('
        "'shape': (",                                          // unterminated
        "'shape': )",                                          // no '('
        "'shape': (99999999999999999999,)",                    // not a usize
        "'shape': (4611686018427387904, 4611686018427387904)", // overflowing product
    ] {
        let outcome = parse(shape);
        assert!(outcome.is_ok(), "header {shape:?} panicked");
        assert_eq!(
            outcome.ok(),
            Some(false),
            "header {shape:?} must be an error"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn zstd_shard_with_a_plausible_but_lying_size_fails_with_a_bounded_reservation() -> Result<()>
{
    // Claims 8 MiB (plausible against the stored size) but carries 1000 bytes: libzstd refuses
    // the frame, and the reservation made from the hint is bounded by the claim.
    let body: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0xE0];
    frame.extend_from_slice(&(8u64 << 20).to_le_bytes());
    let block_header = ((body.len() as u32) << 3) | 1;
    frame.extend_from_slice(&block_header.to_le_bytes()[..3]);
    frame.extend_from_slice(&body);
    let store = RecordingStore {
        canned_get: Some(Bytes::from(frame)),
        ..Default::default()
    };
    let reader = Reader::new(&store, "file:///zc".to_string());
    arm();
    let out = reader.read_shard("shards/rank0.bin.zst").await;
    let (allocated, _) = disarm();
    assert!(allocated < (9 << 20), "allocated {allocated}");
    match out {
        Err(_) => {}
        Ok(out) => assert_eq!(&out[..], &body[..]),
    }
    Ok(())
}

/// Source compatibility of the generic payload parameter: every shape a caller could pass as
/// the old `&[u8]` (through deref coercion) still compiles, and `&Bytes` is zero copy.
#[tokio::test]
async fn writer_accepts_every_shape_that_used_to_coerce_to_a_slice() -> Result<()> {
    use std::borrow::Cow;
    use std::sync::Arc;

    let plain = payload();
    let layout = KeyLayout::new("file:///zc".to_string(), 11);

    // &Bytes: a refcount clone, the store sees the caller's allocation.
    let store = RecordingStore::default();
    Writer::new(&store, "file:///zc".to_string(), 1, 0)
        .put_shard(&layout, &plain)
        .await?;
    assert_inside(&shard_put(&store, plain.len()), &plain, "put_shard(&Bytes)");

    // Compat shapes: content arrives intact (one documented copy).
    let mut vec = plain.to_vec();
    let boxed: Box<[u8]> = plain.to_vec().into_boxed_slice();
    let arc: Arc<[u8]> = Arc::from(plain.to_vec());
    let cow: Cow<[u8]> = Cow::Borrowed(&plain[..]);
    let bytes_mut = bytes::BytesMut::from(&plain[..]);
    let array = [7u8; 64];
    macro_rules! saved {
        ($arg:expr) => {{
            let store = RecordingStore::default();
            Writer::new(&store, "file:///zc".to_string(), 1, 0)
                .put_shard(&layout, $arg)
                .await?;
            store.received().len()
        }};
    }
    assert_eq!(saved!(&vec), 1);
    assert_eq!(saved!(&mut vec), 1);
    assert_eq!(saved!(&mut vec[..]), 1);
    assert_eq!(saved!(&vec[..]), 1);
    assert_eq!(saved!(&boxed), 1);
    assert_eq!(saved!(&arc), 1);
    assert_eq!(saved!(&cow), 1);
    assert_eq!(saved!(&bytes_mut), 1);
    assert_eq!(saved!(&array), 1);
    assert_eq!(saved!(vec), 1);
    Ok(())
}
