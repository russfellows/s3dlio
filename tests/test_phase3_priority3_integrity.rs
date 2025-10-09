// tests/test_phase3_priority3_integrity.rs
//
// Tests for Phase 3 Priority 3: Advanced Integrity Validation

use anyhow::Result;
use s3dlio::store_for_uri;
use s3dlio::checkpoint::{Reader, Manifest};
use s3dlio::checkpoint::manifest::ShardMeta;
use tempfile::TempDir;
use crc32fast::Hasher;

fn compute_checksum(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    format!("{:08x}", hasher.finalize())
}

#[tokio::test]
async fn test_object_store_integrity_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create test data
    let test_data = b"test data for integrity validation";
    let expected_checksum = compute_checksum(test_data);
    
    // Write test file
    let test_uri = format!("{}/test.bin", base_uri);
    store.put(&test_uri, test_data).await?;
    
    // Test successful validation
    let validated_data = store.get_with_validation(&test_uri, Some(&expected_checksum)).await?;
    assert_eq!(validated_data.as_ref(), test_data);
    
    // Test validation failure
    let wrong_checksum = "00000000";
    let result = store.get_with_validation(&test_uri, Some(wrong_checksum)).await;
    assert!(result.is_err());
    
    // Test validation with None checksum (should succeed)
    let data_no_validation = store.get_with_validation(&test_uri, None).await?;
    assert_eq!(data_no_validation.as_ref(), test_data);
    
    Ok(())
}

#[tokio::test]
async fn test_range_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create test data
    let test_data = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let test_uri = format!("{}/range_test.bin", base_uri);
    store.put(&test_uri, test_data).await?;
    
    // Test range with validation
    let range_data = store.get_range(&test_uri, 5, Some(10)).await?;
    let range_checksum = compute_checksum(&range_data);
    
    let validated_range = store.get_range_with_validation(&test_uri, 5, Some(10), Some(&range_checksum)).await?;
    assert_eq!(validated_range, range_data);
    
    // Test range validation failure
    let wrong_checksum = "deadbeef";
    let result = store.get_range_with_validation(&test_uri, 5, Some(10), Some(wrong_checksum)).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_checkpoint_integrity_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create test checkpoint data
    let shard_data = vec![
        (0, b"shard 0 data".to_vec()),
        (1, b"shard 1 data".to_vec()),
        (2, b"shard 2 data".to_vec()),
    ];
    
    // Create manifest with checksums
    let mut manifest = Manifest::new("torch".to_string(), 100, 5, 3);
    
    for (rank, data) in &shard_data {
        let key = format!("data/shard_{}.bin", rank);
        let shard_uri = format!("{}/{}", base_uri, key);
        let checksum = compute_checksum(data);
        
        // Write shard data
        store.put(&shard_uri, data).await?;
        
        // Add to manifest with checksum
        manifest.shards.push(ShardMeta {
            rank: *rank,
            key: key.clone(),
            size: data.len() as u64,
            etag: None,
            checksum: Some(checksum),
        });
    }
    
    manifest.mark_complete();
    
    // Write manifest
    let manifest_key = "manifests/manifest_step_100.json";
    let manifest_uri = format!("{}/{}", base_uri, manifest_key);
    let manifest_data = serde_json::to_vec(&manifest)?;
    store.put(&manifest_uri, &manifest_data).await?;
    
    // Test reading with validation
    let reader = Reader::new(store.as_ref(), base_uri.clone());
    
    // Test individual shard validation
    let shard_0_data = reader.read_shard_by_rank_with_validation(&manifest, 0).await?;
    assert_eq!(shard_0_data, shard_data[0].1);
    
    // Test all shards validation
    let all_shards = reader.read_all_shards_with_validation(&manifest).await?;
    assert_eq!(all_shards.len(), 3);
    for (rank, data) in all_shards {
        let expected_data = &shard_data[rank as usize].1;
        assert_eq!(data, *expected_data);
    }
    
    // Test checkpoint integrity validation
    let validation_result = reader.validate_checkpoint_integrity(&manifest).await?;
    assert!(validation_result, "Checkpoint should validate successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_corrupted_checkpoint_detection() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create test data
    let original_data = b"original shard data";
    let corrupted_data = b"corrupted shard data";
    let checksum = compute_checksum(original_data);
    
    // Write corrupted data but with checksum for original data
    let shard_uri = format!("{}/corrupted_shard.bin", base_uri);
    store.put(&shard_uri, corrupted_data).await?;
    
    // Create manifest with original checksum
    let mut manifest = Manifest::new("torch".to_string(), 200, 10, 1);
    manifest.shards.push(ShardMeta {
        rank: 0,
        key: "corrupted_shard.bin".to_string(),
        size: corrupted_data.len() as u64,
        etag: None,
        checksum: Some(checksum),
    });
    manifest.mark_complete();
    
    let reader = Reader::new(store.as_ref(), base_uri.clone());
    
    // Test that validation detects corruption
    let result = reader.read_shard_by_rank_with_validation(&manifest, 0).await;
    assert!(result.is_err(), "Should detect corruption");
    
    // Test that checkpoint validation fails
    let validation_result = reader.validate_checkpoint_integrity(&manifest).await?;
    assert!(!validation_result, "Checkpoint validation should fail for corrupted data");
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create multiple shards
    let num_shards = 10;
    let mut manifest = Manifest::new("torch".to_string(), 300, 15, num_shards);
    
    for rank in 0..num_shards {
        let data = format!("shard {} data", rank).into_bytes();
        let key = format!("data/shard_{}.bin", rank);
        let shard_uri = format!("{}/{}", base_uri, key);
        let checksum = compute_checksum(&data);
        
        store.put(&shard_uri, &data).await?;
        
        manifest.shards.push(ShardMeta {
            rank,
            key,
            size: data.len() as u64,
            etag: None,
            checksum: Some(checksum),
        });
    }
    
    manifest.mark_complete();
    
    let reader = Reader::new(store.as_ref(), base_uri.clone());
    
    // Test concurrent reading with validation
    let concurrent_results = reader.read_all_shards_concurrent_with_validation(&manifest).await?;
    assert_eq!(concurrent_results.len(), num_shards as usize);
    
    // Verify all data is correct
    for (rank, data) in concurrent_results {
        let expected = format!("shard {} data", rank);
        assert_eq!(data, expected.as_bytes());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_checkpoint_loading_with_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create checkpoint data
    let checkpoint_data = b"checkpoint state data";
    let checksum = compute_checksum(checkpoint_data);
    
    let checkpoint_uri = format!("{}/checkpoint.ckpt", base_uri);
    store.put(&checkpoint_uri, checkpoint_data).await?;
    
    // Test checkpoint loading with validation
    let loaded_data = store.load_checkpoint_with_validation(&checkpoint_uri, Some(&checksum)).await?;
    assert_eq!(loaded_data, checkpoint_data);
    
    // Test with wrong checksum
    let wrong_checksum = "deadbeef";
    let result = store.load_checkpoint_with_validation(&checkpoint_uri, Some(wrong_checksum)).await;
    assert!(result.is_err(), "Should fail with wrong checksum");
    
    // Test without checksum validation
    let loaded_no_validation = store.load_checkpoint_with_validation(&checkpoint_uri, None).await?;
    assert_eq!(loaded_no_validation, checkpoint_data);
    
    Ok(())
}
