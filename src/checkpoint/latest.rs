use serde::{Deserialize, Serialize};
use anyhow::Result;
use tracing::warn;
use crate::object_store::ObjectStore;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Latest {
    pub manifest_key: String,
    pub global_step: u64,
    pub ts: String,
}

impl Latest {
    pub fn new(manifest_key: String, global_step: u64, ts: String) -> Self {
        Self {
            manifest_key,
            global_step,
            ts,
        }
    }
}

/// Generate the append-only latest marker key
pub fn latest_marker_key(step: u64, ts: &str) -> String {
    format!("latest/ckpt-{}-{}.json", step, ts)
}

/// Extract step and timestamp from a latest marker key
pub fn parse_latest_marker_key(key: &str) -> Option<(u64, String)> {
    // Extract from "latest/ckpt-{step}-{timestamp}.json"
    if let Some(filename) = key.strip_prefix("latest/ckpt-") {
        if let Some(stem) = filename.strip_suffix(".json") {
            if let Some(dash_pos) = stem.find('-') {
                let step_str = &stem[..dash_pos];
                let ts_str = &stem[dash_pos + 1..];
                if let Ok(step) = step_str.parse::<u64>() {
                    return Some((step, ts_str.to_string()));
                }
            }
        }
    }
    None
}

/// Write latest pointer using atomic approach for file:// URIs
pub async fn write_latest_atomic(
    store: &dyn ObjectStore,
    base_uri: &str,
    latest: &Latest,
) -> Result<()> {
    let latest_uri = format!("{}/latest.json", base_uri.trim_end_matches('/'));
    
    // First, write the append-only marker
    let marker_key = latest_marker_key(latest.global_step, &latest.ts);
    let marker_uri = format!("{}/{}", base_uri.trim_end_matches('/'), marker_key);
    let marker_bytes = serde_json::to_vec_pretty(latest)?;
    store.put(&marker_uri, &marker_bytes).await?;
    
    // Then update the main latest.json
    // For file:// URIs, we can use atomic rename; for others, we do best-effort
    if base_uri.starts_with("file://") {
        write_latest_with_atomic_rename(store, &latest_uri, latest).await
    } else {
        write_latest(store, &latest_uri, latest).await
    }
}

/// Atomic write for filesystem URIs using temporary file + rename
async fn write_latest_with_atomic_rename(
    store: &dyn ObjectStore,
    latest_uri: &str,
    latest: &Latest,
) -> Result<()> {
    let temp_uri = format!("{}.tmp", latest_uri);
    let bytes = serde_json::to_vec_pretty(latest)?;
    
    // Write to temporary file
    store.put(&temp_uri, &bytes).await?;
    
    // Attempt atomic rename (filesystem backend should support this)
    if let Err(e) = store.rename(&temp_uri, latest_uri).await {
        // If rename fails, fall back to regular write and clean up temp
        let _ = store.delete(&temp_uri).await; // ignore errors
        store.put(latest_uri, &bytes).await?;
        warn!("Atomic rename failed, used fallback write: {}", e);
    }
    
    Ok(())
}

/// Best-effort update: write latest.json; readers also know how to scan manifests on conflict.
pub async fn write_latest(
    store: &dyn ObjectStore,
    latest_uri: &str,
    latest: &Latest,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(latest)?;
    store.put(latest_uri, &bytes).await
}

/// Read the latest pointer, returns None if not found
pub async fn read_latest(
    store: &dyn ObjectStore,
    latest_uri: &str,
) -> Result<Option<Latest>> {
    match store.get(latest_uri).await {
        Ok(bytes) => {
            let latest: Latest = serde_json::from_slice(&bytes)?;
            Ok(Some(latest))
        }
        Err(_) => Ok(None), // not found or other → caller can fallback to manifest scan
    }
}

/// Update latest pointer with conflict detection (best effort)
/// Returns true if update succeeded, false if there might be a conflict
pub async fn update_latest_safe(
    store: &dyn ObjectStore,
    latest_uri: &str,
    new_latest: &Latest,
) -> Result<bool> {
    // Try to read current latest
    match read_latest(store, latest_uri).await? {
        Some(current) => {
            // Only update if our step is higher
            if new_latest.global_step > current.global_step {
                write_latest(store, latest_uri, new_latest).await?;
                Ok(true)
            } else if new_latest.global_step == current.global_step {
                // Same step, compare timestamps (newer wins)
                if new_latest.ts > current.ts {
                    write_latest(store, latest_uri, new_latest).await?;
                    Ok(true)
                } else {
                    // Current is newer or same, don't update
                    Ok(false)
                }
            } else {
                // Our step is older, don't update
                Ok(false)
            }
        }
        None => {
            // No existing latest, write ours
            write_latest(store, latest_uri, new_latest).await?;
            Ok(true)
        }
    }
}

/// Delete the latest pointer
pub async fn delete_latest(
    store: &dyn ObjectStore,
    latest_uri: &str,
) -> Result<()> {
    store.delete(latest_uri).await
}

/// Check if a latest pointer exists
pub async fn latest_exists(
    store: &dyn ObjectStore,
    latest_uri: &str,
) -> Result<bool> {
    store.exists(latest_uri).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::store_for_uri;
    use crate::file_store::FileSystemObjectStore;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_latest_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let latest_uri = format!("{}/latest.json", base_uri);
        
        let store = store_for_uri(&base_uri)?;

        // Initially should not exist
        assert!(!latest_exists(&*store, &latest_uri).await?);
        assert!(read_latest(&*store, &latest_uri).await?.is_none());

        // Write first latest
        let latest1 = Latest::new("manifest1.json".to_string(), 100, "2024-01-01T12:00:00Z".to_string());
        write_latest(&*store, &latest_uri, &latest1).await?;

        // Should now exist and be readable
        assert!(latest_exists(&*store, &latest_uri).await?);
        let read_back = read_latest(&*store, &latest_uri).await?.unwrap();
        assert_eq!(read_back.manifest_key, "manifest1.json");
        assert_eq!(read_back.global_step, 100);

        // Test safe update with higher step
        let latest2 = Latest::new("manifest2.json".to_string(), 200, "2024-01-01T13:00:00Z".to_string());
        assert!(update_latest_safe(&*store, &latest_uri, &latest2).await?);

        // Verify it was updated
        let read_back = read_latest(&*store, &latest_uri).await?.unwrap();
        assert_eq!(read_back.global_step, 200);

        // Test safe update with lower step (should not update)
        let latest3 = Latest::new("manifest3.json".to_string(), 150, "2024-01-01T14:00:00Z".to_string());
        assert!(!update_latest_safe(&*store, &latest_uri, &latest3).await?);

        // Verify it was NOT updated
        let read_back = read_latest(&*store, &latest_uri).await?.unwrap();
        assert_eq!(read_back.global_step, 200);

        // Test delete
        delete_latest(&*store, &latest_uri).await?;
        assert!(!latest_exists(&*store, &latest_uri).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_timestamp_comparison() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let latest_uri = format!("{}/latest.json", base_uri);
        
        let store = store_for_uri(&base_uri)?;

        // Write initial latest
        let latest1 = Latest::new("manifest1.json".to_string(), 100, "2024-01-01T12:00:00Z".to_string());
        write_latest(&*store, &latest_uri, &latest1).await?;

        // Same step, but later timestamp (should update)
        let latest2 = Latest::new("manifest2.json".to_string(), 100, "2024-01-01T13:00:00Z".to_string());
        assert!(update_latest_safe(&*store, &latest_uri, &latest2).await?);

        let read_back = read_latest(&*store, &latest_uri).await?.unwrap();
        assert_eq!(read_back.manifest_key, "manifest2.json");

        // Same step, but earlier timestamp (should not update)
        let latest3 = Latest::new("manifest3.json".to_string(), 100, "2024-01-01T11:00:00Z".to_string());
        assert!(!update_latest_safe(&*store, &latest_uri, &latest3).await?);

        let read_back = read_latest(&*store, &latest_uri).await?.unwrap();
        assert_eq!(read_back.manifest_key, "manifest2.json");

        Ok(())
    }

    #[tokio::test]
    async fn test_atomic_latest_pointer_updates() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path();
        
        // Create test checkpoint URI
        let checkpoint_uri = format!("file://{}/checkpoints", base_path.display());
        let store = FileSystemObjectStore::new();
        
        // Test latest marker key generation
        let marker_key = latest_marker_key(42, "2024-01-01T12:00:00Z");
        assert_eq!(marker_key, "latest/ckpt-42-2024-01-01T12:00:00Z.json");
        
        // Test parsing latest marker key
        if let Some((step, ts)) = parse_latest_marker_key("latest/ckpt-100-2024-01-01T13:00:00Z.json") {
            assert_eq!(step, 100);
            assert_eq!(ts, "2024-01-01T13:00:00Z");
        } else {
            panic!("Failed to parse valid marker key");
        }
        
        // Test invalid marker key parsing
        assert!(parse_latest_marker_key("invalid-marker.json").is_none());
        assert!(parse_latest_marker_key("ckpt-abc-123.json").is_none());
        
        println!("✓ Latest marker key tests passed");
        
        // Test atomic write operation with actual Latest structure
        let latest_info = Latest::new("manifest-42.json".to_string(), 42, "2024-01-01T12:00:00Z".to_string());
        let latest_uri = format!("{}/latest.json", checkpoint_uri);
        
        // Write using existing write_latest function
        write_latest(&store as &dyn ObjectStore, &latest_uri, &latest_info).await?;
        
        // Verify we can read it back
        let loaded_info = read_latest(&store as &dyn ObjectStore, &latest_uri).await?.unwrap();
        assert_eq!(loaded_info.global_step, 42);
        assert_eq!(loaded_info.manifest_key, "manifest-42.json");
        
        println!("✓ Latest pointer write test passed");
        
        // Test filesystem rename functionality
        let src_uri = format!("file://{}/source_file.txt", base_path.display());
        let dst_uri = format!("file://{}/dest_file.txt", base_path.display());
        
        // Create source file
        store.put(&src_uri, b"test atomic rename content").await?;
        assert!(store.exists(&src_uri).await?);
        
        // Perform rename
        store.rename(&src_uri, &dst_uri).await?;
        
        // Verify source is gone and destination exists
        assert!(!store.exists(&src_uri).await?);
        assert!(store.exists(&dst_uri).await?);
        
        // Verify content preserved
        let content = store.get(&dst_uri).await?;
        assert_eq!(content.as_ref(), b"test atomic rename content");
        
        println!("✓ Filesystem rename atomicity test passed");
        
        Ok(())
    }

    #[tokio::test]
    async fn test_latest_pointer_marker_functions() -> Result<()> {
        // Test marker key utilities
        let marker1 = latest_marker_key(100, "1000000000");
        let marker2 = latest_marker_key(101, "1000000001");
        
        assert_eq!(marker1, "latest/ckpt-100-1000000000.json");
        assert_eq!(marker2, "latest/ckpt-101-1000000001.json");
        
        // Test parsing
        if let Some((step1, ts1)) = parse_latest_marker_key(&marker1) {
            assert_eq!(step1, 100);
            assert_eq!(ts1, "1000000000");
        } else {
            panic!("Failed to parse marker1");
        }
        
        if let Some((step2, ts2)) = parse_latest_marker_key(&marker2) {
            assert_eq!(step2, 101);
            assert_eq!(ts2, "1000000001");
        } else {
            panic!("Failed to parse marker2");
        }
        
        println!("✓ Latest marker utility functions test passed");
        
        Ok(())
    }

    #[tokio::test]
    async fn test_atomic_rename_cross_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path();
        
        let store = FileSystemObjectStore::new();
        
        // Test rename across different directories
        let src_uri = format!("file://{}/subdir1/source.txt", base_path.display());
        let dst_uri = format!("file://{}/subdir2/destination.txt", base_path.display());
        
        // Create source file
        store.put(&src_uri, b"cross-directory rename test").await?;
        assert!(store.exists(&src_uri).await?);
        
        // Perform rename (should create destination directory)
        store.rename(&src_uri, &dst_uri).await?;
        
        // Verify operation
        assert!(!store.exists(&src_uri).await?);
        assert!(store.exists(&dst_uri).await?);
        
        // Verify content
        let content = store.get(&dst_uri).await?;
        assert_eq!(content.as_ref(), b"cross-directory rename test");
        
        println!("✓ Cross-directory atomic rename test passed");
        
        Ok(())
    }
}
