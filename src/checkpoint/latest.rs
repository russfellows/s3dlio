use serde::{Deserialize, Serialize};
use anyhow::Result;
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
}
