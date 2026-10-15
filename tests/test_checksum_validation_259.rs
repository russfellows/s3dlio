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

// ---- round 4: message wording, malformed values, compressed shards -------------------

#[tokio::test]
async fn a_mismatch_names_the_algorithm_and_the_pre_256_change() -> Result<()> {
    let dir = TempDir::new()?;
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let uri = format!("{base_uri}/obj.bin");
    store
        .put(&uri, bytes::Bytes::from_static(b"123456789"))
        .await?;

    let msg = format!(
        "{:#}",
        store
            .get_with_validation(&uri, Some("crc32c:cbf43926"))
            .await
            .expect_err("mismatch")
    );
    assert!(msg.contains("CRC-32C"), "{msg}");
    assert!(
        msg.contains("before the #256 fix were CRC-32"),
        "a stored pre-#256 value must be recognisable as such: {msg}"
    );
    let msg = format!(
        "{:#}",
        store
            .get_range_with_validation(&uri, 0, None, Some("crc32c:cbf43926"))
            .await
            .expect_err("mismatch")
    );
    assert!(msg.contains("CRC-32C") && msg.contains("#256"), "{msg}");
    Ok(())
}

#[tokio::test]
async fn malformed_hex_is_a_clear_error_never_a_mismatch() -> Result<()> {
    let dir = TempDir::new()?;
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let uri = format!("{base_uri}/obj.bin");
    store
        .put(&uri, bytes::Bytes::from_static(b"123456789"))
        .await?;

    for bad in [
        "crc32c:",           // empty
        "",                  // empty, bare
        "crc32c:xyz",        // not hex
        "crc32c:123456789",  // nine digits
        "123456789",         // nine digits, bare
        "crc32c:0xe3069283", // 0x prefix is not accepted
        "0xe3069283",
        "crc32c:e3069283:extra", // extra colon
        "crc32c:e306 9283",      // inner space
        "crc32c:-1",
    ] {
        for result in [
            store.get_with_validation(&uri, Some(bad)).await,
            store
                .get_range_with_validation(&uri, 0, None, Some(bad))
                .await,
        ] {
            let msg = format!("{:#}", result.expect_err(bad));
            assert!(msg.contains("malformed checksum"), "{bad:?}: {msg}");
            assert!(
                !msg.contains("Integrity validation failed"),
                "{bad:?}: {msg}"
            );
        }
    }
    Ok(())
}

/// 1 to 8 hex digits are accepted and left-padded: the empty object's CRC-32C is 0, which a
/// writer reports as `crc32c:00000000` and a person may write as `crc32c:0`.
#[tokio::test]
async fn short_hex_is_left_padded() -> Result<()> {
    let dir = TempDir::new()?;
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let uri = format!("{base_uri}/empty.bin");
    store.put(&uri, bytes::Bytes::new()).await?;
    for ok in ["crc32c:00000000", "crc32c:0", "0", "crc32c:000", "00000000"] {
        let r = store.get_with_validation(&uri, Some(ok)).await;
        assert!(r.is_ok(), "{ok}: {r:?}");
    }
    let r = store.get_with_validation(&uri, Some("crc32c:1")).await;
    assert!(r.is_err());
    Ok(())
}

/// A zstd-compressed shard: the writer hashes the UNCOMPRESSED stream, the object in
/// the store is the compressed one, so validation must decompress first.
async fn write_compressed_checkpoint(dir: &TempDir) -> Result<(String, Manifest)> {
    use s3dlio::object_store::CompressionConfig;
    let base_uri = format!("file://{}", dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let writer = Writer::new(store.as_ref(), base_uri.clone(), 1, 0)
        .with_compression(CompressionConfig::zstd_default());
    let layout = KeyLayout::new("compressed".to_string(), 9);

    let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;
    shard_writer.write_chunk(DATA).await?;
    shard_writer.write_chunk(DATA).await?;
    let shard = writer
        .finalize_writer_to_shard_meta(&layout, key, shard_writer)
        .await?;
    assert!(
        shard.key.ends_with(".zst"),
        "the manifest names the stored object: {}",
        shard.key
    );
    assert!(shard.checksum.as_deref().unwrap().starts_with("crc32c:"));

    let mut manifest = Manifest::new("test".to_string(), 9, 0, 1);
    manifest.add_shard(shard);
    manifest.mark_complete();
    let manifest_key = writer.write_manifest(&layout, &manifest).await?;
    let reader = Reader::new(store.as_ref(), base_uri.clone());
    Ok((base_uri, reader.load_manifest(&manifest_key).await?))
}

#[tokio::test]
async fn a_compressed_shard_validates_against_its_uncompressed_checksum() -> Result<()> {
    let dir = TempDir::new()?;
    let (base_uri, manifest) = write_compressed_checkpoint(&dir).await?;
    let store = store_for_uri(&base_uri)?;
    let reader = Reader::new(store.as_ref(), base_uri);
    let shard = &manifest.shards[0];

    let mut want = DATA.to_vec();
    want.extend_from_slice(DATA);
    let data = reader
        .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
        .await?;
    assert_eq!(&data[..], &want[..], "validated and decompressed");
    let all = reader.read_all_shards_with_validation(&manifest).await?;
    assert_eq!(&all[0].1[..], &want[..]);
    let concurrent = reader
        .read_all_shards_concurrent_with_validation(&manifest)
        .await?;
    assert_eq!(&concurrent[0].1[..], &want[..]);
    assert!(reader.validate_checkpoint_integrity(&manifest).await?);
    Ok(())
}

#[tokio::test]
async fn a_corrupted_compressed_shard_still_fails_validation() -> Result<()> {
    let dir = TempDir::new()?;
    let (base_uri, manifest) = write_compressed_checkpoint(&dir).await?;
    let shard = &manifest.shards[0];

    // Replace the object with a well-formed zstd stream of DIFFERENT content: it
    // decompresses fine, so only the checksum can catch it.
    let other = zstd::encode_all(&b"not what the writer hashed"[..], 3)?;
    std::fs::write(dir.path().join(&shard.key), other)?;

    let store = store_for_uri(&base_uri)?;
    let reader = Reader::new(store.as_ref(), base_uri);
    let err = reader
        .read_shard_with_validation(&shard.key, shard.checksum.as_deref())
        .await
        .expect_err("different content must fail validation");
    assert!(
        format!("{err:#}").contains("Integrity validation failed"),
        "{err:#}"
    );
    assert!(!reader.validate_checkpoint_integrity(&manifest).await?);
    let err = reader
        .read_all_shards_concurrent_with_validation(&manifest)
        .await
        .expect_err("different content must fail validation");
    assert!(
        format!("{err:#}").contains("Integrity validation failed"),
        "{err:#}"
    );
    Ok(())
}

#[tokio::test]
async fn a_compressed_shard_with_an_unsupported_algorithm_is_a_clear_error() -> Result<()> {
    let dir = TempDir::new()?;
    let (base_uri, manifest) = write_compressed_checkpoint(&dir).await?;
    let store = store_for_uri(&base_uri)?;
    let reader = Reader::new(store.as_ref(), base_uri);
    let err = reader
        .read_shard_with_validation(&manifest.shards[0].key, Some("sha256:00"))
        .await
        .expect_err("unsupported");
    assert!(
        format!("{err:#}").contains("unsupported checksum algorithm"),
        "{err:#}"
    );
    Ok(())
}
