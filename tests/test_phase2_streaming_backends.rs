// tests/test_phase2_streaming_backends.rs

use anyhow::Result;
use bytes::Bytes;
use s3dlio::object_store::{ObjectStore, WriterOptions, CompressionConfig};
use s3dlio::file_store::FileSystemObjectStore;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::{AzureObjectStore, S3ObjectStore};
use tempfile::TempDir;


#[tokio::test]
async fn test_filesystem_streaming_writer() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = FileSystemObjectStore::new();
    let test_path = temp_dir.path().join("phase2-streaming-test.bin");
    let test_uri = format!("file://{}", test_path.display());
    
    // Test basic streaming write
    let options = WriterOptions::new();
    let mut writer = store.create_writer(&test_uri, options).await?;
    
    // Write some test chunks
    let chunk1 = Bytes::from(vec![10u8; 2048]);
    let chunk2 = Bytes::from(vec![20u8; 1024]);
    let chunk3 = Bytes::from(vec![30u8; 4096]);
    
    writer.write_chunk(&chunk1).await?;
    writer.write_chunk(&chunk2).await?;
    writer.write_chunk(&chunk3).await?;
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the data was written correctly
    let data = store.get(&test_uri).await?;
    assert_eq!(data.len(), 2048 + 1024 + 4096);
    assert_eq!(&data[0..2048], &vec![10u8; 2048]);
    assert_eq!(&data[2048..3072], &vec![20u8; 1024]);
    assert_eq!(&data[3072..7168], &vec![30u8; 4096]);
    
    println!("✅ FileSystem streaming writer test passed");
    Ok(())
}

#[tokio::test]
async fn test_filesystem_streaming_writer_with_compression() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = FileSystemObjectStore::new();
    let test_path = temp_dir.path().join("phase2-streaming-test-compressed.bin");
    let test_uri = format!("file://{}", test_path.display());
    
    // Test streaming write with compression
    let compression = CompressionConfig::zstd_default();
    let options = WriterOptions::new().with_compression(compression);
    let mut writer = store.create_writer(&test_uri, options).await?;
    
    // Write compressible test data
    let chunk1 = Bytes::from(vec![77u8; 8192]); // Highly compressible
    let chunk2 = Bytes::from(vec![88u8; 4096]);
    
    writer.write_chunk(&chunk1).await?;
    writer.write_chunk(&chunk2).await?;
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the file exists and is compressed (compression adds .zst extension)
    let actual_path = test_path.with_extension("bin.zst");
    assert!(actual_path.exists());
    let compressed_size = std::fs::metadata(&actual_path)?.len();
    assert!(compressed_size < 8192 + 4096); // Should be compressed
    
    println!("✅ FileSystem streaming writer with compression test passed");
    Ok(())
}

#[tokio::test]
async fn test_direct_io_streaming_writer() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Test if O_DIRECT is supported on this filesystem
    if !ConfigurableFileSystemObjectStore::test_direct_io_support(temp_dir.path()).await {
        println!("⚠️  Skipping DirectIO streaming test - O_DIRECT not supported on this filesystem");
        return Ok(());
    }
    
    let config = FileSystemConfig::direct_io();
    let store = ConfigurableFileSystemObjectStore::new(config);
    let test_path = temp_dir.path().join("phase2-streaming-test-direct.bin");
    let test_uri = format!("direct://{}", test_path.display());
    
    // Test basic streaming write with O_DIRECT
    let options = WriterOptions::new();
    let mut writer = store.create_writer(&test_uri, options).await?;
    
    // Write aligned test chunks (O_DIRECT requires alignment)
    let page_size = 4096; // Typical page size
    let chunk1 = Bytes::from(vec![100u8; page_size]);
    let chunk2 = Bytes::from(vec![200u8; page_size * 2]);
    
    writer.write_chunk(&chunk1).await?;
    writer.write_chunk(&chunk2).await?;
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the data was written correctly
    let data = store.get(&test_uri).await?;
    assert_eq!(data.len(), page_size + page_size * 2);
    assert_eq!(&data[0..page_size], &vec![100u8; page_size]);
    assert_eq!(&data[page_size..page_size * 3], &vec![200u8; page_size * 2]);
    
    println!("✅ DirectIO streaming writer test passed");
    Ok(())
}

#[tokio::test]
async fn test_write_owned_bytes() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = FileSystemObjectStore::new();
    let test_path = temp_dir.path().join("phase2-streaming-owned-test.bin");
    let test_uri = format!("file://{}", test_path.display());
    
    // Test write_owned_bytes method for zero-copy optimization
    let options = WriterOptions::new();
    let mut writer = store.create_writer(&test_uri, options).await?;
    
    // Create owned data that can be moved
    let owned_data1 = vec![111u8; 1024];
    let owned_data2 = vec![222u8; 2048];
    
    writer.write_owned_bytes(owned_data1).await?;
    writer.write_owned_bytes(owned_data2).await?;
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the data was written correctly
    let data = store.get(&test_uri).await?;
    assert_eq!(data.len(), 1024 + 2048);
    assert_eq!(&data[0..1024], &vec![111u8; 1024]);
    assert_eq!(&data[1024..3072], &vec![222u8; 2048]);
    
    println!("✅ write_owned_bytes test passed");
    Ok(())
}

#[tokio::test]
async fn test_s3_streaming_writer() -> Result<()> {
    // Skip if no S3 credentials
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        println!("⚠️  Skipping S3 streaming test - no AWS credentials");
        return Ok(());
    }

    let store = S3ObjectStore::new();
    let test_uri = "s3://test-bucket/phase2-streaming-test-s3.bin";
    
    // Test basic streaming write
    let options = WriterOptions::new();
    let mut writer = store.create_writer(test_uri, options).await?;
    
    // Write some test chunks
    let chunk1 = Bytes::from(vec![1u8; 1024]);
    let chunk2 = Bytes::from(vec![2u8; 2048]);
    let chunk3 = Bytes::from(vec![3u8; 512]);
    
    writer.write_chunk(&chunk1).await?;
    writer.write_chunk(&chunk2).await?;
    writer.write_chunk(&chunk3).await?;
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the data was written correctly
    let data = store.get(test_uri).await?;
    assert_eq!(data.len(), 1024 + 2048 + 512);
    assert_eq!(&data[0..1024], &vec![1u8; 1024]);
    assert_eq!(&data[1024..3072], &vec![2u8; 2048]);
    assert_eq!(&data[3072..3584], &vec![3u8; 512]);
    
    // Clean up
    store.delete(test_uri).await?;
    
    println!("✅ S3 streaming writer test passed");
    Ok(())
}

#[tokio::test]
async fn test_azure_streaming_writer() -> Result<()> {
    // Skip if no Azure credentials
    if std::env::var("AZURE_BLOB_ACCOUNT").is_err() {
        println!("⚠️  Skipping Azure streaming test - no Azure credentials");
        return Ok(());
    }

    let store = AzureObjectStore::new();
    let test_uri = "az://test-account/test-container/phase2-streaming-test-azure.bin";
    
    // Test basic streaming write
    let options = WriterOptions::new();
    let mut writer = store.create_writer(test_uri, options).await?;
    
    // Write some test chunks
    let chunk1 = Bytes::from(vec![50u8; 1024]);
    let chunk2 = Bytes::from(vec![60u8; 2048]);
    let chunk3 = Bytes::from(vec![70u8; 512]);
    
    writer.write_chunk(&chunk1).await?;
    writer.write_chunk(&chunk2).await?;
    writer.write_chunk(&chunk3).await?;
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the data was written correctly
    let data = store.get(test_uri).await?;
    assert_eq!(data.len(), 1024 + 2048 + 512);
    assert_eq!(&data[0..1024], &vec![50u8; 1024]);
    assert_eq!(&data[1024..3072], &vec![60u8; 2048]);
    assert_eq!(&data[3072..3584], &vec![70u8; 512]);
    
    // Clean up
    store.delete(test_uri).await?;
    
    println!("✅ Azure streaming writer test passed");
    Ok(())
}

#[tokio::test]
async fn test_large_streaming_write() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = FileSystemObjectStore::new();
    let test_path = temp_dir.path().join("phase2-large-streaming-test.bin");
    let test_uri = format!("file://{}", test_path.display());
    
    // Test large streaming write to verify multipart behavior
    let options = WriterOptions::new();
    let mut writer = store.create_writer(&test_uri, options).await?;
    
    let chunk_size = 1024 * 1024; // 1MB chunks
    let num_chunks = 10;
    let mut expected_data = Vec::new();
    
    for i in 0..num_chunks {
        let chunk_data = vec![(i % 256) as u8; chunk_size];
        expected_data.extend_from_slice(&chunk_data);
        writer.write_chunk(&Bytes::from(chunk_data)).await?;
    }
    
    // Finalize the write
    writer.finalize().await?;
    
    // Verify the large file was written correctly
    let data = store.get(&test_uri).await?;
    assert_eq!(data.len(), chunk_size * num_chunks);
    assert_eq!(data, expected_data);
    
    println!("✅ Large streaming write test passed ({} MB)", (chunk_size * num_chunks) / (1024 * 1024));
    Ok(())
}
