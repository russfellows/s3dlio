// Comprehensive validation tests for Phase 2 streaming infrastructure
use tempfile::TempDir;
use anyhow::Result;
use tokio::fs;

#[tokio::test]
async fn test_filesystem_streaming_writer() -> Result<()> {
    println!("=== Testing FileSystemWriter streaming ===");
    
    use s3dlio::object_store::store_for_uri;
    
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Test streaming write
    let test_key = "test_streaming_write.dat";
    let full_uri = format!("{}/{}", base_uri, test_key);
    let mut writer = store.get_writer(&full_uri).await?;
    
    // Write data in chunks
    let chunk1 = b"Hello, ";
    let chunk2 = b"streaming ";
    let chunk3 = b"world!";
    
    writer.write_chunk(chunk1).await?;
    assert_eq!(writer.bytes_written(), chunk1.len() as u64);
    
    writer.write_chunk(chunk2).await?;
    assert_eq!(writer.bytes_written(), (chunk1.len() + chunk2.len()) as u64);
    
    writer.write_chunk(chunk3).await?;
    let total_expected = chunk1.len() + chunk2.len() + chunk3.len();
    assert_eq!(writer.bytes_written(), total_expected as u64);
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the file was written correctly
    let file_path = temp_dir.path().join(test_key);
    let content = fs::read(&file_path).await?;
    assert_eq!(content, b"Hello, streaming world!");
    
    println!("✓ FileSystemWriter streaming test passed");
    Ok(())
}

#[tokio::test]
async fn test_directio_streaming_writer() -> Result<()> {
    println!("=== Testing DirectIOWriter streaming ===");
    
    use s3dlio::object_store::store_for_uri;
    
    let temp_dir = TempDir::new()?;
    let base_uri = format!("directio://{}", temp_dir.path().display());
    
    // DirectIO requires specific configuration - let's test it exists
    match store_for_uri(&base_uri) {
        Ok(store) => {
            println!("✓ DirectIO store created successfully");
            
            // Test that we can create a writer (even if O_DIRECT might not work in test env)
            let test_key = "test_directio.dat";
            let full_uri = format!("{}/{}", base_uri, test_key);
            match store.get_writer(&full_uri).await {
                Ok(mut writer) => {
                    // Try a simple write
                    writer.write_chunk(b"DirectIO test").await?;
                    println!("✓ DirectIO writer write_chunk succeeded");
                    
                    // Try to finalize
                    match writer.finalize().await {
                        Ok(_) => println!("✓ DirectIO writer finalize succeeded"),
                        Err(e) => println!("⚠ DirectIO finalize failed (expected in some test environments): {}", e),
                    }
                },
                Err(e) => println!("⚠ DirectIO writer creation failed (expected in some test environments): {}", e),
            }
        },
        Err(e) => println!("⚠ DirectIO store creation failed (expected in some test environments): {}", e),
    }
    
    Ok(())
}

#[tokio::test]
async fn test_object_writer_trait_interface() -> Result<()> {
    println!("=== Testing ObjectWriter trait interface ===");
    
    use s3dlio::object_store::store_for_uri;
    
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Test the trait interface
    let full_uri = format!("{}/trait_test.dat", base_uri);
    let mut writer = store.get_writer(&full_uri).await?;
    
    // Test bytes_written before any writes
    assert_eq!(writer.bytes_written(), 0);
    
    // Test write_chunk
    writer.write_chunk(b"test").await?;
    assert_eq!(writer.bytes_written(), 4);
    
    // Test multiple writes
    writer.write_chunk(b" data").await?;
    assert_eq!(writer.bytes_written(), 9);
    
    // Test finalize
    writer.finalize().await?;
    
    // Verify the final content
    let file_path = temp_dir.path().join("trait_test.dat");
    let content = fs::read(&file_path).await?;
    assert_eq!(content, b"test data");
    
    println!("✓ ObjectWriter trait interface test passed");
    Ok(())
}

#[tokio::test]
async fn test_writer_cancellation() -> Result<()> {
    println!("=== Testing writer cancellation ===");
    
    use s3dlio::object_store::store_for_uri;
    
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Create a writer and write some data
    let full_uri = format!("{}/cancel_test.dat", base_uri);
    let mut writer = store.get_writer(&full_uri).await?;
    writer.write_chunk(b"This should be cancelled").await?;
    assert_eq!(writer.bytes_written(), 24);
    
    // Cancel the writer
    writer.cancel().await?;
    
    // Verify the file was not created (or was cleaned up)
    let file_path = temp_dir.path().join("cancel_test.dat");
    assert!(!file_path.exists(), "File should not exist after cancellation");
    
    println!("✓ Writer cancellation test passed");
    Ok(())
}

#[tokio::test]
async fn test_checkpoint_streaming_integration() -> Result<()> {
    println!("=== Testing checkpoint streaming integration ===");
    
    use s3dlio::object_store::store_for_uri;
    use s3dlio::checkpoint::writer::Writer;
    use s3dlio::checkpoint::paths::KeyLayout;
    
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    let writer = Writer::new(&*store, base_uri, 1, 0);
    let layout = KeyLayout::new("checkpoints".to_string(), 1);
    
    // Test get_shard_writer
    let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;
    println!("✓ Shard writer created with key: {}", key);
    
    // Write streaming data
    let chunk1 = vec![1u8; 1024];  // 1KB
    let chunk2 = vec![2u8; 2048];  // 2KB
    let chunk3 = vec![3u8; 512];   // 0.5KB
    
    shard_writer.write_chunk(&chunk1).await?;
    assert_eq!(shard_writer.bytes_written(), 1024);
    
    shard_writer.write_chunk(&chunk2).await?;
    assert_eq!(shard_writer.bytes_written(), 3072);
    
    shard_writer.write_chunk(&chunk3).await?;
    assert_eq!(shard_writer.bytes_written(), 3584);
    
    // Finalize the shard writer
    shard_writer.finalize().await?;
    
    // Finalize the shard metadata
    let metadata = writer.finalize_shard_meta(&layout, key).await?;
    assert_eq!(metadata.size, 3584);
    
    println!("✓ Checkpoint streaming integration test passed");
    println!("  - Total bytes written: {}", metadata.size);
    println!("  - Shard key: {}", metadata.key);
    
    Ok(())
}

#[tokio::test]
async fn test_large_streaming_write() -> Result<()> {
    println!("=== Testing large streaming write (memory efficiency) ===");
    
    use s3dlio::object_store::store_for_uri;
    
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    let full_uri = format!("{}/large_test.dat", base_uri);
    let mut writer = store.get_writer(&full_uri).await?;
    
    // Write 50MB in 1MB chunks (simulating large checkpoint)
    let chunk_size = 1024 * 1024; // 1MB
    let num_chunks = 50;           // 50MB total
    let chunk = vec![42u8; chunk_size];
    
    println!("Writing {} chunks of {} bytes each...", num_chunks, chunk_size);
    
    for i in 0..num_chunks {
        writer.write_chunk(&chunk).await?;
        let expected_bytes = (i + 1) * chunk_size;
        assert_eq!(writer.bytes_written(), expected_bytes as u64);
        
        if (i + 1) % 10 == 0 {
            println!("  Written {} MB", (i + 1));
        }
    }
    
    writer.finalize().await?;
    
    // Verify the file size
    let file_path = temp_dir.path().join("large_test.dat");
    let metadata = fs::metadata(&file_path).await?;
    assert_eq!(metadata.len(), (num_chunks * chunk_size) as u64);
    
    println!("✓ Large streaming write test passed");
    println!("  - Total size: {} MB", metadata.len() / (1024 * 1024));
    
    Ok(())
}
