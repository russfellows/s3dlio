use s3dlio::*;
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_checkpoint_writer_with_checksums() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create a checkpoint writer
    let writer = s3dlio::checkpoint::Writer::new(store.as_ref(), base_uri.clone(), 1, 0)
        .with_multipart_threshold(1024 * 1024); // 1MB threshold
    
    // Create a layout for the checkpoint
    let layout = s3dlio::checkpoint::KeyLayout::new("test_checkpoint".to_string(), 42);
    
    // Get a streaming writer
    let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;
    
    // Write some test data
    let test_data1 = b"Checkpoint data chunk 1: ";
    let test_data2 = b"This is important state information that needs integrity validation.";
    
    shard_writer.write_chunk(test_data1).await?;
    shard_writer.write_chunk(test_data2).await?;
    
    // Get checksum before finalizing
    let computed_checksum = shard_writer.checksum();
    assert!(computed_checksum.is_some());
    let checksum_str = computed_checksum.as_ref().unwrap();
    assert!(checksum_str.starts_with("crc32c:"));
    
    // Use the new helper to finalize and create ShardMeta with checksum
    let shard_meta = writer.finalize_writer_to_shard_meta(&layout, key, shard_writer).await?;
    
    // Verify the shard metadata includes the checksum
    assert!(shard_meta.checksum.is_some());
    assert_eq!(shard_meta.checksum.as_ref().unwrap(), checksum_str);
    assert_eq!(shard_meta.rank, 0);
    assert!(shard_meta.size > 0);
    
    println!("✓ Checkpoint shard created with checksum: {}", checksum_str);
    println!("  Shard size: {} bytes", shard_meta.size);
    println!("  Shard key: {}", shard_meta.key);
    
    Ok(())
}

#[tokio::test]
async fn test_checkpoint_manifest_with_checksums() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create a checkpoint writer
    let writer = s3dlio::checkpoint::Writer::new(store.as_ref(), base_uri.clone(), 1, 0);
    
    // Create multiple shards with different data and collect their metadata
    let mut shard_metas = Vec::new();
    
    for i in 0..3 {
        let layout = s3dlio::checkpoint::KeyLayout::new(
            "test_multi_checkpoint".to_string(),
            100 + i, // Different step numbers
        );
        
        let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;
        
        // Write different data for each shard
        let test_data = format!("Shard {} data with unique content for checksum verification", i);
        shard_writer.write_chunk(test_data.as_bytes()).await?;
        
        // Finalize and collect metadata with checksum
        let shard_meta = writer.finalize_writer_to_shard_meta(&layout, key, shard_writer).await?;
        shard_metas.push(shard_meta);
    }
    
    // Verify all shards have unique checksums
    let checksums: Vec<_> = shard_metas.iter().map(|s| s.checksum.as_ref().unwrap()).collect();
    for i in 0..checksums.len() {
        for j in i+1..checksums.len() {
            assert_ne!(checksums[i], checksums[j], "Checksums should be unique for different data");
        }
    }
    
    // Create a manifest with all the shards
    let mut manifest = s3dlio::checkpoint::Manifest::new(
        "test_multi_checkpoint".to_string(),
        1, // format_version
        100, // step
        1  // world_size
    );
    for shard_meta in shard_metas {
        println!("Adding shard with checksum: {:?}", shard_meta.checksum);
        manifest.add_shard(shard_meta);
    }
    
    // Verify manifest contains all checksums
    assert_eq!(manifest.shards.len(), 3);
    for shard in &manifest.shards {
        assert!(shard.checksum.is_some());
        assert!(shard.checksum.as_ref().unwrap().starts_with("crc32c:"));
    }
    
    println!("✓ Created manifest with {} shards, all containing checksums", manifest.shards.len());
    
    Ok(())
}

#[tokio::test]
async fn test_checkpoint_integrity_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create checkpoint writer
    let writer = s3dlio::checkpoint::Writer::new(store.as_ref(), base_uri.clone(), 1, 0);
    
    let layout = s3dlio::checkpoint::KeyLayout::new(
        "integrity_test".to_string(),
        200,
    );
    
    // Write data and capture checksum
    let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;
    let original_data = b"Critical data that must maintain integrity across storage and retrieval operations.";
    shard_writer.write_chunk(original_data).await?;
    
    let expected_checksum = shard_writer.checksum().unwrap();
    let shard_meta = writer.finalize_writer_to_shard_meta(&layout, key.clone(), shard_writer).await?;
    
    // Verify file exists and read it back
    let file_path = temp_dir.path().join(&key);
    let read_data = tokio::fs::read(&file_path).await?;
    assert_eq!(read_data, original_data);
    
    // Simulate re-reading and computing checksum to verify integrity
    let store2 = store_for_uri(&base_uri)?;
    let full_uri = format!("{}/{}", base_uri, key);
    let mut writer2 = store2.get_writer(&full_uri).await?;
    writer2.write_chunk(&read_data).await?;
    let recomputed_checksum = writer2.checksum().unwrap();
    
    // Checksums should match, proving data integrity
    assert_eq!(expected_checksum, recomputed_checksum);
    assert_eq!(shard_meta.checksum.unwrap(), expected_checksum);
    
    println!("✓ Data integrity verified: original checksum {} matches recomputed checksum", 
             expected_checksum);
    
    Ok(())
}
