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

async fn check_stored_name_matches_manifest(
    store: &dyn ObjectStore,
    scheme: &str,
    tag: &str,
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
                writer.put_shard_with_key(key, DATA).await?
            } else {
                let mut w = writer.get_shard_writer_with_key(key).await?;
                w.write_chunk(DATA).await?;
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
            assert_eq!(zstd::decode_all(&on_disk[..])?, DATA, "{what}: decodes");
            assert_eq!(shard.size, on_disk.len() as u64, "{what}: recorded size");

            // The recorded name reads back through the checkpoint reader.
            let reader = Reader::new(store, base_uri.clone());
            let back = reader
                .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
                .await?;
            if via_put_shard_with_key {
                // put_shard_with_key records no checksum; the plain read must still decode.
                assert_eq!(&reader.read_shard(&shard.key).await?[..], DATA, "{what}");
            } else {
                assert!(shard.checksum.is_some(), "{what}");
            }
            assert_eq!(&back[..], DATA, "{what}: round trip");
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
