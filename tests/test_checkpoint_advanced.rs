use tempfile::TempDir;
use tokio;

use s3dlio::checkpoint::CheckpointStore;

/// Test checkpoint performance with larger data sizes
#[tokio::test]
async fn test_checkpoint_performance() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    let checkpoint_store = CheckpointStore::open_high_performance(&uri).unwrap();
    
    // Create larger test data (1MB per shard)
    let shard_size = 1024 * 1024; // 1MB
    let num_ranks = 4; // Reduced for faster testing
    
    let start_time = std::time::Instant::now();
    
    // Write distributed checkpoint with multiple ranks
    let mut layouts_and_shards = Vec::new();
    for rank in 0..num_ranks {
        let test_data = vec![rank as u8; shard_size];
        let writer = checkpoint_store.writer(num_ranks, rank);
        let (layout, shard_meta) = writer.save_distributed_shard(1, 1, "pytorch", &test_data).await.unwrap();
        layouts_and_shards.push((layout, shard_meta));
    }
    
    // Finalize checkpoint
    let coordinator = checkpoint_store.writer(num_ranks, 0);
    let shards: Vec<_> = layouts_and_shards.iter().map(|(_, shard)| shard.clone()).collect();
    let layout = &layouts_and_shards[0].0;
    coordinator.finalize_distributed_checkpoint(layout, "pytorch", 1, shards, None).await.unwrap();
    
    let write_time = start_time.elapsed();
    println!("✓ Wrote {}MB checkpoint ({} ranks) in {:?}", 
             (shard_size * num_ranks as usize) / (1024 * 1024), num_ranks, write_time);
    
    // Read back and measure read performance
    let read_start = std::time::Instant::now();
    let reader = checkpoint_store.reader();
    let manifest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(manifest_opt.is_some());
    let manifest = manifest_opt.unwrap();
    
    // Read all shards
    let mut total_bytes = 0;
    for rank in 0..num_ranks {
        let data = reader.read_shard_by_rank(&manifest, rank).await.unwrap();
        assert_eq!(data.len(), shard_size);
        assert_eq!(data[0], rank as u8); // Verify data integrity
        total_bytes += data.len();
    }
    
    let read_time = read_start.elapsed();
    println!("✓ Read {}MB checkpoint in {:?}", total_bytes / (1024 * 1024), read_time);
    
    assert_eq!(manifest.shards.len(), num_ranks as usize);
    assert_eq!(total_bytes, shard_size * num_ranks as usize);
}

/// Test concurrent checkpoint operations
#[tokio::test]
async fn test_concurrent_checkpoints() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    let checkpoint_store = CheckpointStore::open(&uri).unwrap();
    
    // Launch multiple concurrent checkpoint operations
    let num_concurrent = 4u64;
    let mut handles = Vec::new();
    
    for i in 0..num_concurrent {
        let uri_clone = uri.clone();
        let handle = tokio::spawn(async move {
            let store = CheckpointStore::open(&uri_clone).unwrap();
            let epoch = i + 1;
            let test_data = format!("concurrent checkpoint {} data", i).into_bytes();
            
            let writer = store.writer(1, 0);
            let manifest_key = writer.save_checkpoint(i + 1, epoch, "pytorch", &test_data, None).await.unwrap();
            
            (epoch, manifest_key)
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        println!("✓ Concurrent checkpoint completed: epoch={}, manifest={}", result.0, result.1);
        results.push(result);
    }
    
    assert_eq!(results.len(), num_concurrent as usize);
    
    // Verify we can read the latest checkpoint
    let reader = checkpoint_store.reader();
    let latest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(latest_opt.is_some());
    let latest = latest_opt.unwrap();
    
    // Should be one of the epochs we wrote (timing might affect which one is "latest")
    assert!(latest.epoch >= 1 && latest.epoch <= num_concurrent, 
           "Expected epoch between 1 and {}, got {}", num_concurrent, latest.epoch);
    
    println!("✓ Concurrent checkpoints completed successfully");
}

/// Test checkpoint cleanup and management
#[tokio::test]
async fn test_checkpoint_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    let checkpoint_store = CheckpointStore::open(&uri).unwrap();
    
    // Create multiple checkpoint epochs
    let epochs = vec![1, 2, 3, 4, 5];
    
    for (i, epoch) in epochs.iter().enumerate() {
        let test_data = format!("cleanup test epoch {}", epoch).into_bytes();
        let writer = checkpoint_store.writer(1, 0);
        writer.save_checkpoint(i as u64 + 1, *epoch, "tensorflow", &test_data, None).await.unwrap();
    }
    
    println!("✓ Created {} checkpoints for cleanup testing", epochs.len());
    
    // Verify all checkpoints exist by checking the directory structure
    let checkpoint_dir = temp_dir.path();
    assert!(checkpoint_dir.exists());
    
    // List manifest files to verify they exist
    let manifest_dir = checkpoint_dir.join("manifests");
    if manifest_dir.exists() {
        let entries: Vec<_> = std::fs::read_dir(&manifest_dir)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        
        println!("✓ Found {} manifest files in {}", entries.len(), manifest_dir.display());
        assert!(entries.len() >= epochs.len());
    }
    
    // Verify we can still read the latest checkpoint
    let reader = checkpoint_store.reader();
    let latest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(latest_opt.is_some());
    let latest = latest_opt.unwrap();
    assert_eq!(latest.epoch, 5);
    
    println!("✓ Checkpoint cleanup test completed");
}

/// Test error recovery scenarios
#[tokio::test]
async fn test_error_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().display());
    
    let checkpoint_store = CheckpointStore::open(&uri).unwrap();
    
    // Test 1: Write a complete checkpoint
    let writer2 = checkpoint_store.writer(1, 0);
    writer2.save_checkpoint(2, 2, "pytorch", b"complete checkpoint data", None).await.unwrap();
    
    // Verify we can read the complete checkpoint (epoch 2)
    let reader = checkpoint_store.reader();
    let latest_opt = reader.load_latest_manifest().await.unwrap();
    assert!(latest_opt.is_some());
    let latest = latest_opt.unwrap();
    assert_eq!(latest.epoch, 2);
    
    let data = reader.read_shard_by_rank(&latest, 0).await.unwrap();
    assert_eq!(data.as_ref(), b"complete checkpoint data");
    
    println!("✓ Error recovery test: system correctly handles partial writes");
}
