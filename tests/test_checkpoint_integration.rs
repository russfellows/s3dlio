use std::collections::HashMap;
use tempfile::TempDir;
use tokio;

use s3dlio::checkpoint::CheckpointStore;

/// Test basic checkpoint write and read operations
#[tokio::test]
async fn test_basic_checkpoint_operations() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    // Create checkpoint store with auto config
    let checkpoint_store = CheckpointStore::open_auto_config(&uri).unwrap();
    
    // Test data to checkpoint
    let test_data = b"test checkpoint data for rank 0";
    let mut user_metadata = HashMap::new();
    user_metadata.insert("model_type".to_string(), serde_json::Value::String("transformer".to_string()));
    user_metadata.insert("training_step".to_string(), serde_json::Value::Number(serde_json::Number::from(42)));
    
    // Create a writer for single-rank checkpoint
    let writer = checkpoint_store.writer(1, 0); // world_size=1, rank=0
    
    // Use the convenience method to save a complete checkpoint
    let user_meta_json = serde_json::to_value(user_metadata.clone()).unwrap();
    let manifest_key = writer.save_checkpoint(1, 1, "pytorch", test_data, Some(user_meta_json)).await.unwrap();
    
    println!("✓ Wrote checkpoint: manifest={}", manifest_key);
    
    // Read back the checkpoint
    let reader = checkpoint_store.reader();
    let latest_manifest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(latest_manifest_opt.is_some(), "Should have found a checkpoint");
    
    let latest_manifest = latest_manifest_opt.unwrap();
    
    // Verify metadata
    assert_eq!(latest_manifest.epoch, 1);
    assert_eq!(latest_manifest.framework, "pytorch");
    assert_eq!(latest_manifest.user_meta, Some(serde_json::to_value(user_metadata).unwrap()));
    assert_eq!(latest_manifest.shards.len(), 1);
    assert_eq!(latest_manifest.shards[0].rank, 0);
    
    // Read the shard data  
    let loaded_data = reader.read_shard_by_rank(&latest_manifest, 0).await.unwrap();
    assert_eq!(loaded_data.as_ref(), test_data);
    
    println!("✓ Successfully read back checkpoint data");
}

/// Test checkpoint validation and error handling
#[tokio::test]
async fn test_checkpoint_validation() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    let checkpoint_store = CheckpointStore::open(&uri).unwrap();
    
    // Test missing checkpoint
    let reader = checkpoint_store.reader();
    let result = reader.load_latest_manifest().await.unwrap();
    assert!(result.is_none(), "Should return None when no checkpoint exists");
    
    // Write a simple valid checkpoint first
    let writer = checkpoint_store.writer(1, 0);
    let _manifest_key = writer.save_checkpoint(1, 1, "pytorch", b"test data", None).await.unwrap();
    
    // Now we should be able to read it
    let manifest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(manifest_opt.is_some());
    let manifest = manifest_opt.unwrap();
    assert_eq!(manifest.epoch, 1);
    assert_eq!(manifest.framework, "pytorch");
    assert_eq!(manifest.shards.len(), 1);
    
    println!("✓ Validation test completed");
}

/// Test checkpoint versioning and epoch management
#[tokio::test]
async fn test_checkpoint_versioning() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    let checkpoint_store = CheckpointStore::open(&uri).unwrap();
    
    // Write multiple checkpoint epochs
    let epochs = vec![1, 3, 2, 5]; // Out of order to test sorting
    let framework = "pytorch";
    
    for (i, epoch) in epochs.iter().enumerate() {
        let test_data = format!("data for epoch {}", epoch).into_bytes();
        let mut user_meta = HashMap::new();
        user_meta.insert("epoch".to_string(), serde_json::Value::Number(serde_json::Number::from(*epoch)));
        
        let writer = checkpoint_store.writer(1, 0);
        let user_meta_json = serde_json::to_value(user_meta).unwrap();
        let _manifest_key = writer.save_checkpoint(i as u64 + 1, *epoch, framework, &test_data, Some(user_meta_json)).await.unwrap();
        
        println!("✓ Wrote checkpoint for epoch {}", epoch);
    }
    
    // Verify latest checkpoint is epoch 5 (highest)
    let reader = checkpoint_store.reader();
    let latest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(latest_opt.is_some());
    let latest = latest_opt.unwrap();
    assert_eq!(latest.epoch, 5);
    
    let data = reader.read_shard_by_rank(&latest, 0).await.unwrap();
    assert_eq!(data.as_ref(), b"data for epoch 5");
    
    println!("✓ Latest checkpoint correctly identified as epoch {}", latest.epoch);
}
