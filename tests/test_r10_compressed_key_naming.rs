// tests/test_r10_compressed_key_naming.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for review finding R10 (0.9.128): a compressing writer for a custom key that
// ALREADY ends in the compression extension (`custom.zst`) wrote `custom.zst.zst`, while the
// checkpoint helpers recorded and statted `custom.zst`. One suffix rule must name the stored
// object and the manifest entry: append the extension only if the name does not already end
// with it.

// The Arrow backend has no local-filesystem store in this build (its object_store `fs`
// feature is off) and no direct://, so its naming is covered by a unit test in
// src/object_store_arrow.rs; this file exercises the native file:// and direct:// stores.
#![cfg(feature = "native-backends")]

use anyhow::Result;
use s3dlio::checkpoint::{KeyLayout, Reader, Writer};
use s3dlio::object_store::{CompressionConfig, ObjectStore};
use std::collections::BTreeSet;
use std::path::Path;

use s3dlio::object_store::store_for_uri;

const DATA: &[u8] =
    b"R10: the manifest name must equal the stored object name, and it must decode.";

/// (key given to the writer, name the object must be stored and recorded under)
const CASES: &[(&str, &str)] = &[
    ("plain.bin", "plain.bin.zst"),
    ("noext", "noext.zst"),
    ("custom.zst", "custom.zst"),
    ("a.zst.bin", "a.zst.bin.zst"),
];

fn files_in(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// Scratch directory on the filesystem the test needs (O_DIRECT is refused by tmpfs).
fn scratch_dir(tag: &str) -> tempfile::TempDir {
    let parent = Path::new("/mnt/scratch/tmp");
    let prefix = format!("orange-r10-{tag}-");
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix);
    if parent.is_dir() {
        builder.tempdir_in(parent).unwrap()
    } else {
        let fallback = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        std::fs::create_dir_all(&fallback).unwrap();
        builder.tempdir_in(fallback).unwrap()
    }
}

/// Incompressible bytes (xorshift), so the compressed stream is about as long as the input.
fn incompressible(len: usize) -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

async fn check_stored_name_matches_manifest(
    store: &dyn ObjectStore,
    scheme: &str,
    tag: &str,
) -> Result<()> {
    check_with_payload(store, scheme, tag, DATA, 0).await
}

/// `min_stored`: the stored (compressed) object must be at least this long, so a test can
/// require the data to span aligned O_DIRECT blocks and not only a short final tail.
async fn check_with_payload(
    store: &dyn ObjectStore,
    scheme: &str,
    tag: &str,
    data: &[u8],
    min_stored: usize,
) -> Result<()> {
    for via_put_shard_with_key in [false, true] {
        for (key, stored) in CASES {
            let dir = scratch_dir(tag);
            let base_uri = format!("{scheme}://{}", dir.path().display());
            let writer = Writer::new(store, base_uri.clone(), 1, 0)
                .with_compression(CompressionConfig::zstd_default());
            let layout = KeyLayout::new("r10".to_string(), 1);
            let what = format!("{scheme} key={key} put_shard_with_key={via_put_shard_with_key}");

            let shard = if via_put_shard_with_key {
                writer.put_shard_with_key(key, data).await?
            } else {
                let mut w = writer.get_shard_writer_with_key(key).await?;
                w.write_chunk(data).await?;
                writer
                    .finalize_writer_to_shard_meta(&layout, key.to_string(), w)
                    .await?
            };

            assert_eq!(&shard.key, stored, "{what}: manifest name");
            assert_eq!(
                files_in(dir.path()),
                BTreeSet::from([stored.to_string()]),
                "{what}: the stored object is exactly the manifest name"
            );
            let on_disk = std::fs::read(dir.path().join(stored))?;
            assert!(
                on_disk.len() >= min_stored,
                "{what}: stored {}",
                on_disk.len()
            );
            assert_eq!(zstd::decode_all(&on_disk[..])?, data, "{what}: decodes");
            assert_eq!(shard.size, on_disk.len() as u64, "{what}: recorded size");

            // The recorded name reads back through the checkpoint reader.
            let reader = Reader::new(store, base_uri.clone());
            let back = reader
                .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
                .await?;
            if via_put_shard_with_key {
                // put_shard_with_key records no checksum; the plain read must still decode.
                assert_eq!(&reader.read_shard(&shard.key).await?[..], data, "{what}");
            } else {
                assert!(shard.checksum.is_some(), "{what}");
            }
            assert_eq!(&back[..], data, "{what}: round trip");
        }
    }
    Ok(())
}

#[test]
fn one_suffix_rule_helper() {
    let z = CompressionConfig::zstd_default();
    assert_eq!(z.stored_name("a.bin"), "a.bin.zst");
    assert_eq!(z.stored_name("a.zst"), "a.zst");
    assert_eq!(z.stored_name("a"), "a.zst");
    assert_eq!(z.stored_name("a.zst.zst"), "a.zst.zst");
    assert_eq!(
        z.stored_path(Path::new("/d/a.bin")),
        Path::new("/d/a.bin.zst")
    );
    assert_eq!(z.stored_path(Path::new("/d/a.zst")), Path::new("/d/a.zst"));
    assert_eq!(z.stored_path(Path::new("/d/a")), Path::new("/d/a.zst"));
    // No compression never renames.
    assert_eq!(CompressionConfig::None.stored_name("a.bin"), "a.bin");
    assert_eq!(
        CompressionConfig::None.stored_path(Path::new("/d/a")),
        Path::new("/d/a")
    );
}

#[tokio::test]
async fn file_scheme_stored_name_equals_manifest_name() -> Result<()> {
    let store = store_for_uri("file:///")?;
    check_stored_name_matches_manifest(store.as_ref(), "file", "file").await
}

/// `direct://` opens the object with O_DIRECT; this needs a filesystem that accepts it
/// (the scratch XFS does; tmpfs refuses, and the writer then fails rather than falling back).
#[tokio::test]
async fn direct_scheme_stored_name_equals_manifest_name() -> Result<()> {
    let store = store_for_uri("direct:///")?;
    check_stored_name_matches_manifest(store.as_ref(), "direct", "direct").await
}

/// The 76-byte payload above never reaches an aligned O_DIRECT block (its compressed stream is a
/// short tail). 256 KiB of incompressible data spans many aligned blocks plus an unaligned
/// tail, so the O_DIRECT data path is the one exercised.
#[tokio::test]
async fn direct_scheme_multi_block_payload_stored_name_equals_manifest_name() -> Result<()> {
    let store = store_for_uri("direct:///")?;
    let data = incompressible(256 * 1024 + 123);
    check_with_payload(store.as_ref(), "direct", "direct-big", &data, 256 * 1024).await
}

// ---- R10b: the Python CheckpointStream path (and the public finalize helpers) -------------

use s3dlio::checkpoint::{CheckpointConfig, CheckpointStore};

/// What `CheckpointStream.write_chunk` / `.finalize` do in `python_aiml_api.rs`: a writer taken
/// from a `CheckpointStore` configured with compression, `get_shard_writer`, `write_chunk`,
/// and the stream's finalize (`finalize_writer_to_shard_meta`).
async fn stream_path(compression: Option<CompressionConfig>, data: &[u8]) -> Result<()> {
    let dir = tempfile::TempDir::new()?;
    let uri = format!("file://{}", dir.path().display());
    let config = CheckpointConfig {
        compression,
        ..CheckpointConfig::default()
    };
    let ckpt = CheckpointStore::open_with_config(&uri, config)?;
    let writer = ckpt.writer(1, 0);
    let layout = KeyLayout::new("checkpoints".to_string(), 5);

    let (mut w, key) = writer.get_shard_writer(&layout).await?;
    w.write_chunk(data).await?;
    let shard = writer
        .finalize_writer_to_shard_meta(&layout, key, w)
        .await?;
    let expect_zst = compression.is_some();
    assert_eq!(shard.key.ends_with(".zst"), expect_zst, "{}", shard.key);
    assert!(dir.path().join(&shard.key).is_file(), "{}", shard.key);
    assert_eq!(
        shard.size,
        std::fs::metadata(dir.path().join(&shard.key))?.len()
    );

    // The public helpers a caller uses after finalizing the writer itself: with a compressing
    // Writer they record the stored name (the key `get_shard_writer` returned is un-suffixed).
    for with_checksum in [false, true] {
        let (mut w, key) = writer.get_shard_writer(&layout).await?;
        w.write_chunk(data).await?;
        let checksum = w.checksum();
        w.finalize().await?;
        let shard = if with_checksum {
            writer
                .finalize_shard_meta_with_checksum(&layout, key, checksum)
                .await?
        } else {
            writer.finalize_shard_meta(&layout, key).await?
        };
        assert_eq!(shard.key.ends_with(".zst"), expect_zst, "{}", shard.key);
        assert!(dir.path().join(&shard.key).is_file(), "{}", shard.key);
        assert_eq!(
            shard.size,
            std::fs::metadata(dir.path().join(&shard.key))?.len()
        );
    }
    Ok(())
}

#[tokio::test]
async fn checkpoint_stream_path_records_the_stored_name_with_compression() -> Result<()> {
    stream_path(Some(CompressionConfig::zstd_level(3)), DATA).await
}

#[tokio::test]
async fn checkpoint_stream_path_unchanged_without_compression() -> Result<()> {
    stream_path(None, DATA).await
}
