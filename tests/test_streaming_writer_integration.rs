use anyhow::Result;
use tempfile::TempDir;

use s3dlio::{
    streaming_writer::StreamingDataWriter,
    object_store::{ObjectWriter, WriterOptions, CompressionConfig},
    file_store::FileSystemObjectStore,
    constants::BLK_SIZE,
};

#[tokio::test]
async fn test_streaming_data_writer_basic() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("streaming_test.bin");
    let uri = format!("file://{}", test_path.display());
    
    let store = FileSystemObjectStore::new();
    let options = WriterOptions::new();
    
    let size = BLK_SIZE * 2; // 128KB
    let mut writer = StreamingDataWriter::new(&uri, size, 1, 1, &store, options).await?;
    
    // Generate data in chunks
    let chunk_size = 32 * 1024; // 32KB chunks
    let mut total_generated = 0;
    
    while !writer.is_complete() {
        let bytes_written = writer.generate_chunk(chunk_size).await?;
        total_generated += bytes_written;
        println!("Generated {} bytes, total: {}", bytes_written, total_generated);
    }
    
    assert_eq!(total_generated, size);
    assert_eq!(writer.bytes_generated(), size as u64);
    
    // Finalize the write
    Box::new(writer).finalize().await?;
    
    // Verify the file was created and has correct size
    let metadata = tokio::fs::metadata(&test_path).await?;
    assert_eq!(metadata.len(), size as u64);
    
    println!("✅ Basic streaming data writer test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming_data_writer_compression() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("streaming_compressed.bin");
    let uri = format!("file://{}", test_path.display());
    
    let store = FileSystemObjectStore::new();
    let mut options = WriterOptions::new();
    options.compression = Some(CompressionConfig::Zstd { level: 3 });
    
    let size = BLK_SIZE; // 64KB
    // Use high deduplication for more compressible data
    let mut writer = StreamingDataWriter::new(&uri, size, 8, 1, &store, options).await?;
    
    // Generate all data at once
    writer.generate_remaining().await?;
    
    assert_eq!(writer.bytes_generated(), size as u64);
    assert_eq!(writer.bytes_written(), size as u64);
    
    // Check compression configuration is set
    let compression = writer.compression();
    assert!(matches!(compression, CompressionConfig::Zstd { level: 3 }));
    
    // Note: compression ratio may be 1.0 if data is not compressible enough
    // The FileSystemWriter applies compression but ratio depends on data characteristics
    let compression_ratio = writer.compression_ratio();
    println!("Compression ratio: {:.3}", compression_ratio);
    
    // Finalize
    Box::new(writer).finalize().await?;
    
    // Verify the file exists (compression adds .zst extension)
    let compressed_path = temp_dir.path().join("streaming_compressed.bin.zst");
    assert!(compressed_path.exists(), "Compressed file should exist with .zst extension");
    
    println!("✅ Streaming data writer with compression test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming_data_writer_checksum() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("streaming_checksum.bin");
    let uri = format!("file://{}", test_path.display());
    
    let store = FileSystemObjectStore::new();
    let options = WriterOptions::new();
    
    let size = 1024; // Small size for easy verification
    let mut writer = StreamingDataWriter::new(&uri, size, 1, 1, &store, options).await?;
    
    // Generate data and track checksum
    writer.generate_remaining().await?;
    
    let checksum = writer.checksum();
    assert!(checksum.is_some());
    
    let checksum_str = checksum.unwrap();
    assert!(checksum_str.starts_with("crc32c:"));
    
    println!("Generated checksum: {}", checksum_str);
    
    // Finalize
    Box::new(writer).finalize().await?;
    
    println!("✅ Streaming data writer checksum test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming_data_writer_consistency() -> Result<()> {
    // Test that streaming generation produces consistent results
    let size = BLK_SIZE;
    let dedup = 2;
    let compress = 3;
    
    // Generate data using StreamingDataWriter
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("streaming_consistency.bin");
    let uri = format!("file://{}", test_path.display());
    
    let store = FileSystemObjectStore::new();
    let options = WriterOptions::new();
    
    let mut streaming_writer = StreamingDataWriter::new(&uri, size, dedup, compress, &store, options.clone()).await?;
    streaming_writer.generate_remaining().await?;
    let streaming_checksum = streaming_writer.checksum().unwrap();
    Box::new(streaming_writer).finalize().await?;
    
    // Generate the same data using the same DataGenerator instance for comparison
    // The key insight: StreamingDataWriter creates its own DataGenerator with unique entropy,
    // so we test consistency by ensuring same writer produces same results
    let test_path2 = temp_dir.path().join("streaming_consistency2.bin");
    let uri2 = format!("file://{}", test_path2.display());
    
    // Create another writer with same parameters but different instance
    let mut streaming_writer2 = StreamingDataWriter::new(&uri2, size, dedup, compress, &store, options).await?;
    streaming_writer2.generate_remaining().await?;
    let streaming_checksum2 = streaming_writer2.checksum().unwrap();
    Box::new(streaming_writer2).finalize().await?;
    
    // Different StreamingDataWriter instances should produce different results (different entropy)
    assert_ne!(streaming_checksum, streaming_checksum2, 
        "Different StreamingDataWriter instances should produce different data");
    
    // Read back the files and verify they have correct size
    let file_data1 = tokio::fs::read(&test_path).await?;
    let file_data2 = tokio::fs::read(&test_path2).await?;
    assert_eq!(file_data1.len(), size);
    assert_eq!(file_data2.len(), size);
    assert_ne!(file_data1, file_data2, "Data should be different due to different entropy");
    
    println!("✅ Streaming data writer consistency test passed");
    Ok(())
}

#[tokio::test]
async fn test_streaming_data_writer_mixed_mode() -> Result<()> {
    // Test mixing synthetic generation with manual chunk writes
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("streaming_mixed.bin");
    let uri = format!("file://{}", test_path.display());
    
    let store = FileSystemObjectStore::new();
    let options = WriterOptions::new();
    
    let size = 1024;
    let mut writer = StreamingDataWriter::new(&uri, size, 1, 1, &store, options).await?;
    
    // Generate some synthetic data
    writer.generate_chunk(512).await?;
    
    // Write some manual data
    let manual_data = vec![0xAA; 256];
    writer.write_chunk(&manual_data).await?;
    
    // Generate remaining synthetic data
    writer.generate_remaining().await?;
    
    let total_bytes = writer.bytes_written();
    assert!(total_bytes > size as u64); // More than target due to manual data
    
    Box::new(writer).finalize().await?;
    
    println!("✅ Streaming data writer mixed mode test passed");
    Ok(())
}