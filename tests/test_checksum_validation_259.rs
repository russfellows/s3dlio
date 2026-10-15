// tests/test_checksum_validation_259.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #259: writers and checkpoint manifests carry checksums
// as `<algorithm>:<hex>` (`crc32c:e3069283`), but `get_with_validation` compared the
// expected string with the BARE hex of the data, so validating the value a writer
// reported failed ("expected crc32c:xxxxxxxx, got xxxxxxxx") even for intact data.
//
// One canonical format: `<alg>:<hex>`, with the algorithm parsed (`crc32c` is supported,
// anything else is a clear error). A bare hex string is still accepted as CRC-32C, for
// callers that compute it themselves.

use anyhow::Result;
use s3dlio::checkpoint::{KeyLayout, Manifest, Reader, Writer};
use s3dlio::object_store::store_for_uri;
use tempfile::TempDir;

const DATA: &[u8] = b"Shard bytes that a checkpoint writer checksums while it writes them.";

/// Write one shard through the real checkpoint writer, record it in a manifest on disk,
/// and return the manifest as the reader loads it back.
async fn write_checkpoint(dir: &TempDir) -> Result<(String, Manifest)> {
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let writer = Writer::new(store.as_ref(), base_uri.clone(), 1, 0);
    let layout = KeyLayout::new("round_trip".to_string(), 7);

    let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;
    shard_writer.write_chunk(DATA).await?;
    let shard = writer
        .finalize_writer_to_shard_meta(&layout, key, shard_writer)
        .await?;
    assert!(
        shard.checksum.as_deref().unwrap().starts_with("crc32c:"),
        "the writer reports a prefixed checksum"
    );

    let mut manifest = Manifest::new("test".to_string(), 7, 0, 1);
    manifest.add_shard(shard);
    manifest.mark_complete();
    let manifest_key = writer.write_manifest(&layout, &manifest).await?;

    let reader = Reader::new(store.as_ref(), base_uri.clone());
    Ok((base_uri, reader.load_manifest(&manifest_key).await?))
}

#[tokio::test]
async fn a_shard_validates_against_the_checksum_in_its_manifest() -> Result<()> {
    let dir = TempDir::new()?;
    let (base_uri, manifest) = write_checkpoint(&dir).await?;
    let store = store_for_uri(&base_uri)?;
    let reader = Reader::new(store.as_ref(), base_uri);
    let shard = &manifest.shards[0];

    let data = reader
        .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
        .await?;
    assert_eq!(&data[..], DATA);
    let all = reader.read_all_shards_with_validation(&manifest).await?;
    assert_eq!(all.len(), 1);
    assert!(reader.validate_checkpoint_integrity(&manifest).await?);
    Ok(())
}

#[tokio::test]
async fn a_corrupted_shard_still_fails_validation() -> Result<()> {
    let dir = TempDir::new()?;
    let (base_uri, manifest) = write_checkpoint(&dir).await?;
    let shard = &manifest.shards[0];

    let path = dir.path().join(&shard.key);
    let mut bytes = std::fs::read(&path)?;
    bytes[3] ^= 0x01;
    std::fs::write(&path, bytes)?;

    let store = store_for_uri(&base_uri)?;
    let reader = Reader::new(store.as_ref(), base_uri);
    let err = reader
        .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
        .await
        .expect_err("a flipped bit must fail validation");
    assert!(
        format!("{err:#}").contains("Integrity validation failed"),
        "{err:#}"
    );
    assert!(!reader.validate_checkpoint_integrity(&manifest).await?);
    Ok(())
}

#[tokio::test]
async fn formats_accepted_by_get_with_validation() -> Result<()> {
    let dir = TempDir::new()?;
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let uri = format!("{base_uri}/obj.bin");
    store
        .put(&uri, bytes::Bytes::from_static(b"123456789"))
        .await?;

    // Canonical `<alg>:<hex>`, the algorithm name in any case, the hex in any case.
    for ok in [
        "crc32c:e3069283",
        "CRC32C:e3069283",
        "crc32c:E3069283",
        "e3069283",
    ] {
        let data = store.get_with_validation(&uri, Some(ok)).await;
        assert!(data.is_ok(), "{ok}: {data:?}");
        let range = store
            .get_range_with_validation(&uri, 0, None, Some(ok))
            .await;
        assert!(range.is_ok(), "{ok}: {range:?}");
    }
    // No expected checksum: no validation.
    assert!(store.get_with_validation(&uri, None).await.is_ok());

    // A different value of the right algorithm is a mismatch naming both values.
    let err = store
        .get_with_validation(&uri, Some("crc32c:cbf43926"))
        .await
        .expect_err("CRC-32 (IEEE) is not the CRC-32C of the object");
    let msg = format!("{err:#}");
    assert!(msg.contains("Integrity validation failed"), "{msg}");
    assert!(
        msg.contains("cbf43926") && msg.contains("e3069283"),
        "{msg}"
    );
    Ok(())
}

#[tokio::test]
async fn an_unknown_algorithm_is_a_clear_error_not_a_mismatch() -> Result<()> {
    let dir = TempDir::new()?;
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let uri = format!("{base_uri}/obj.bin");
    store
        .put(&uri, bytes::Bytes::from_static(b"123456789"))
        .await?;

    for expected in ["sha256:e3069283", "md5:abcd", "crc32:cbf43926", ":e3069283"] {
        let err = store
            .get_with_validation(&uri, Some(expected))
            .await
            .expect_err("unknown algorithm");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("unsupported checksum algorithm") && msg.contains("crc32c"),
            "{expected}: {msg}"
        );
        assert!(!msg.contains("Integrity validation failed"), "{msg}");
        let err = store
            .get_range_with_validation(&uri, 0, None, Some(expected))
            .await
            .expect_err("unknown algorithm");
        assert!(format!("{err:#}").contains("unsupported checksum algorithm"));
    }
    Ok(())
}
