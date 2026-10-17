// tests/rfc3339_times_263.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #263: the file and direct-I/O stores report a file's modification time in
// `ObjectMetadata::last_modified` as RFC 3339 (like every other backend), not as a
// `SystemTime { tv_sec: .., tv_nsec: .. }` Debug dump.  Local and deterministic: the
// modification time is set explicitly, so the expected string is fixed.

use anyhow::Result;
use bytes::Bytes;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::ObjectStore;
use s3dlio::{store_for_uri, ObjectMetadata};
use std::path::Path;
use std::time::{Duration, UNIX_EPOCH};
use tempfile::TempDir;

/// 2025-05-16T09:46:12Z.
const BASE_SECS: u64 = 1_747_388_772;

/// Set a file's modification time to `BASE_SECS` plus `nanos`.
fn set_mtime(path: &Path, nanos: u32) -> Result<()> {
    let file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.set_modified(UNIX_EPOCH + Duration::new(BASE_SECS, nanos))?;
    Ok(())
}

/// Stat a file whose mtime was set to `BASE_SECS` + `nanos` through both stores.
async fn stat_both(nanos: u32) -> Result<(ObjectMetadata, ObjectMetadata)> {
    let dir = TempDir::new()?;
    let path = dir.path().join("t.dat");
    let uri = format!("file://{}", path.display());

    let plain = store_for_uri(&uri)?;
    plain.put(&uri, Bytes::from_static(b"0123456789")).await?;
    set_mtime(&path, nanos)?;
    let plain_meta = plain.stat(&uri).await?;

    let direct = ConfigurableFileSystemObjectStore::new(FileSystemConfig::direct_io());
    let direct_meta = direct.stat(&uri).await?;
    Ok((plain_meta, direct_meta))
}

/// A whole-second UTC time is an RFC 3339 string in both stores.
#[tokio::test]
async fn whole_second_mtime_is_rfc3339_in_file_and_direct_stores() -> Result<()> {
    let (plain, direct) = stat_both(0).await?;
    assert_eq!(
        plain.last_modified.as_deref(),
        Some("2025-05-16T09:46:12Z"),
        "file store"
    );
    assert_eq!(
        direct.last_modified.as_deref(),
        Some("2025-05-16T09:46:12Z"),
        "direct store"
    );
    Ok(())
}

/// The sub-second part the filesystem holds is kept, not truncated to whole seconds.
#[tokio::test]
async fn fractional_mtime_keeps_its_precision_in_file_and_direct_stores() -> Result<()> {
    let (plain, direct) = stat_both(123_456_789).await?;
    assert_eq!(
        plain.last_modified.as_deref(),
        Some("2025-05-16T09:46:12.123456789Z"),
        "file store"
    );
    assert_eq!(
        direct.last_modified.as_deref(),
        Some("2025-05-16T09:46:12.123456789Z"),
        "direct store"
    );
    Ok(())
}

/// Nothing but the time string changed: size, ETag identity and storage class are as before.
#[tokio::test]
async fn identity_fields_are_unchanged() -> Result<()> {
    let (plain, direct) = stat_both(250_000_000).await?;
    // The ETag is `file-<len>-<whole seconds of mtime>`, independent of the time string.
    let want_etag = format!("file-10-{BASE_SECS}");
    assert_eq!(plain.size, 10);
    assert_eq!(direct.size, 10);
    assert_eq!(plain.e_tag.as_deref(), Some(want_etag.as_str()));
    assert_eq!(direct.e_tag.as_deref(), Some(want_etag.as_str()));
    assert_eq!(plain.storage_class.as_deref(), Some("STANDARD"));
    assert_eq!(direct.storage_class.as_deref(), Some("DIRECT_IO"));
    Ok(())
}

/// The string parses back to the same instant.
#[tokio::test]
async fn stat_time_round_trips_to_the_same_instant() -> Result<()> {
    let (plain, direct) = stat_both(7_008_009).await?;
    for (name, meta) in [("file", plain), ("direct", direct)] {
        let text = meta.last_modified.expect("present");
        let parsed = chrono::DateTime::parse_from_rfc3339(&text).expect("RFC 3339");
        assert_eq!(parsed.timestamp(), BASE_SECS as i64, "{name}");
        assert_eq!(parsed.timestamp_subsec_nanos(), 7_008_009, "{name}");
    }
    Ok(())
}
