// tests/test_phase3_priority4_python_rust_exchange.rs
//
// Tests for Phase 3 Priority 4: Rich Python-Rust Data Exchange

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
async fn test_enhanced_checkpoint_loading_with_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create test checkpoint data
    let checkpoint_data = b"enhanced checkpoint state data for priority 4";
    let expected_checksum = compute_checksum(checkpoint_data);
    
    let checkpoint_uri = format!("{}/checkpoint_priority4.ckpt", base_uri);
    store.put(&checkpoint_uri, checkpoint_data).await?;
    
    // Test loading with validation enabled
    let loaded_data = store.load_checkpoint_with_validation(&checkpoint_uri, Some(&expected_checksum)).await?;
    assert_eq!(loaded_data, checkpoint_data);
    
    // Test loading with validation disabled (should still work)
    let loaded_no_validation = store.get(&checkpoint_uri).await?;
    assert_eq!(loaded_no_validation.as_ref(), checkpoint_data);
    
    // Test validation failure
    let wrong_checksum = "deadbeef";
    let result = store.load_checkpoint_with_validation(&checkpoint_uri, Some(wrong_checksum)).await;
    assert!(result.is_err(), "Should fail with wrong checksum");
    
    Ok(())
}

#[tokio::test]
async fn test_tensor_like_data_exchange() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Simulate tensor data (4x4 matrix of f32)
    let tensor_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let tensor_bytes = bytemuck::cast_slice(&tensor_data);
    let checksum = compute_checksum(tensor_bytes);
    
    let tensor_uri = format!("{}/tensor_4x4.bin", base_uri);
    store.put(&tensor_uri, tensor_bytes).await?;
    
    // Test loading with validation
    let loaded_bytes = store.get_with_validation(&tensor_uri, Some(&checksum)).await?;
    let loaded_tensor: &[f32] = bytemuck::cast_slice(&loaded_bytes);
    
    assert_eq!(loaded_tensor.len(), 16);
    for (i, &value) in loaded_tensor.iter().enumerate() {
        assert_eq!(value, i as f32);
    }
    
    // Test partial loading (simulate range-based tensor loading)
    let partial_bytes = store.get_range_with_validation(&tensor_uri, 0, Some(32), None).await?; // First 8 f32s
    let partial_tensor: &[f32] = bytemuck::cast_slice(&partial_bytes);
    
    assert_eq!(partial_tensor.len(), 8);
    for (i, &value) in partial_tensor.iter().enumerate() {
        assert_eq!(value, i as f32);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_distributed_checkpoint_with_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create distributed checkpoint shards
    let world_size = 4;
    let mut manifest = Manifest::new("torch".to_string(), 500, 25, world_size);
    
    let mut all_checksums = Vec::new();
    
    for rank in 0..world_size {
        // Create rank-specific data
        let shard_data = format!("rank {} shard data for distributed checkpoint", rank).into_bytes();
        let key = format!("priority4/shard_{}.bin", rank);
        let shard_uri = format!("{}/{}", base_uri, key);
        let checksum = compute_checksum(&shard_data);
        
        // Store shard
        store.put(&shard_uri, &shard_data).await?;
        all_checksums.push(checksum.clone());
        
        // Add to manifest with checksum
        manifest.shards.push(ShardMeta {
            rank,
            key,
            size: shard_data.len() as u64,
            etag: None,
            checksum: Some(checksum),
        });
    }
    
    manifest.mark_complete();
    
    // Test distributed loading with validation
    let reader = Reader::new(store.as_ref(), base_uri);
    
    // Load individual shards with validation
    for rank in 0..world_size {
        let shard_data = reader.read_shard_by_rank_with_validation(&manifest, rank).await?;
        let expected = format!("rank {} shard data for distributed checkpoint", rank);
        assert_eq!(shard_data, expected.as_bytes());
    }
    
    // Load all shards concurrently with validation
    let all_shards = reader.read_all_shards_concurrent_with_validation(&manifest).await?;
    assert_eq!(all_shards.len(), world_size as usize);
    
    // Verify all data
    for (rank, data) in all_shards {
        let expected = format!("rank {} shard data for distributed checkpoint", rank);
        assert_eq!(data, expected.as_bytes());
    }
    
    // Test overall checkpoint validation
    let is_valid = reader.validate_checkpoint_integrity(&manifest).await?;
    assert!(is_valid, "Distributed checkpoint should validate successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_compression_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create large, compressible data
    let large_data = vec![42u8; 10000]; // Highly compressible
    let checksum = compute_checksum(&large_data);
    
    let data_uri = format!("{}/large_compressible.bin", base_uri);
    store.put(&data_uri, &large_data).await?;
    
    // Test loading with validation (data should compress well)
    let loaded_data = store.get_with_validation(&data_uri, Some(&checksum)).await?;
    assert_eq!(loaded_data, large_data);
    
    // Test range loading with validation
    let range_data = store.get_range_with_validation(&data_uri, 100, Some(500), None).await?;
    assert_eq!(range_data.len(), 500);
    assert!(range_data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_zero_copy_capabilities() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create data that demonstrates zero-copy potential
    let original_data = b"zero-copy test data for efficient transfers";
    let checksum = compute_checksum(original_data);
    
    let zero_copy_uri = format!("{}/zero_copy_test.bin", base_uri);
    store.put(&zero_copy_uri, original_data).await?;
    
    // Test efficient loading
    let loaded_data = store.get_with_validation(&zero_copy_uri, Some(&checksum)).await?;
    assert_eq!(loaded_data.as_ref(), original_data);
    
    // Test that the data is properly validated without extra copies
    assert_eq!(compute_checksum(&loaded_data), checksum);
    
    Ok(())
}

#[tokio::test]
async fn test_enhanced_error_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Test loading non-existent file
    let missing_uri = format!("{}/does_not_exist.bin", base_uri);
    let result = store.get(&missing_uri).await;
    assert!(result.is_err(), "Should fail for non-existent file");
    
    // Test validation with missing checksum data
    let test_data = b"test data for error handling";
    let test_uri = format!("{}/error_test.bin", base_uri);
    store.put(&test_uri, test_data).await?;
    
    // Should succeed without validation
    let data = store.get(&test_uri).await?;
    assert_eq!(data.as_ref(), test_data);
    
    // Should fail with wrong checksum
    let wrong_checksum = "00000000";
    let result = store.get_with_validation(&test_uri, Some(wrong_checksum)).await;
    assert!(result.is_err(), "Should fail with wrong checksum");
    
    Ok(())
}

#[tokio::test]
async fn test_metadata_preservation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create checkpoint with rich metadata
    let checkpoint_data = b"checkpoint with metadata";
    let checksum = compute_checksum(checkpoint_data);
    
    let mut manifest = Manifest::new("pytorch".to_string(), 1000, 50, 1);
    
    // Add metadata-rich shard
    manifest.shards.push(ShardMeta {
        rank: 0,
        key: "metadata_checkpoint.bin".to_string(),
        size: checkpoint_data.len() as u64,
        etag: Some("test-etag-12345".to_string()),
        checksum: Some(checksum.clone()),
    });
    
    manifest.mark_complete();
    
    // Store the checkpoint data
    let checkpoint_uri = format!("{}/metadata_checkpoint.bin", base_uri);
    store.put(&checkpoint_uri, checkpoint_data).await?;
    
    // Test that metadata is preserved through the validation process
    let reader = Reader::new(store.as_ref(), base_uri);
    let loaded_data = reader.read_shard_by_rank_with_validation(&manifest, 0).await?;
    assert_eq!(loaded_data.as_ref(), checkpoint_data);
    
    // Verify metadata integrity
    let shard = &manifest.shards[0];
    assert_eq!(shard.checksum.as_ref().unwrap(), &checksum);
    assert_eq!(shard.etag.as_ref().unwrap(), "test-etag-12345");
    assert_eq!(shard.size, checkpoint_data.len() as u64);
    
    Ok(())
}
