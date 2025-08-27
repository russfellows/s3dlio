// tests/test_phase3_priority2_compression.rs
//
// Test Phase 3 Priority 2: Compression Support in Streaming Pipeline

use anyhow::Result;
use tempfile::tempdir;

use s3dlio::file_store::FileSystemWriter;
use s3dlio::object_store::{ObjectWriter, CompressionConfig};

#[tokio::test]
async fn test_compression_basic_functionality() -> Result<()> {
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("test_compression.dat");
    
    // Test data (should compress well)
    let test_data = b"Hello, World! ".repeat(1000); // Repetitive data compresses well
    
    // Create writer with Zstd compression
    let mut writer = FileSystemWriter::new_with_compression(
        file_path.clone(), 
        CompressionConfig::zstd_default()
    ).await?;
    
    // Write test data
    writer.write_chunk(&test_data).await?;
    
    // Check compression stats before finalization
    assert_eq!(writer.bytes_written(), test_data.len() as u64);
    assert!(writer.compression().is_enabled());
    assert_eq!(writer.compression(), CompressionConfig::zstd_default());
    
    // Finalize
    let writer_boxed = Box::new(writer);
    writer_boxed.finalize().await?;
    
    // Verify compressed file exists with .zst extension
    let compressed_path = file_path.with_extension("dat.zst");
    assert!(compressed_path.exists());
    
    // Verify file is actually smaller (compression worked)
    let compressed_size = std::fs::metadata(&compressed_path)?.len();
    assert!(compressed_size < test_data.len() as u64, 
           "Compressed size {} should be less than original {}", 
           compressed_size, test_data.len());
    
    println!("✓ Compression test passed: {} bytes -> {} bytes (ratio: {:.2})", 
             test_data.len(), compressed_size, 
             compressed_size as f64 / test_data.len() as f64);
    
    Ok(())
}

#[tokio::test]
async fn test_compression_checksum_integrity() -> Result<()> {
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("test_checksum.dat");
    
    let test_data = b"Test data for checksum validation with compression enabled.";
    
    // Create writer with compression
    let mut writer = FileSystemWriter::new_with_compression(
        file_path.clone(), 
        CompressionConfig::zstd_level(5)
    ).await?;
    
    // Write data and get checksum
    writer.write_chunk(test_data).await?;
    let checksum = writer.checksum().expect("Checksum should be available");
    
    // Verify checksum format
    assert!(checksum.starts_with("crc32c:"), "Checksum should have crc32c format");
    assert_eq!(checksum.len(), 15, "Checksum should be 'crc32c:' + 8 hex chars");
    
    // Finalize
    let writer_boxed = Box::new(writer);
    writer_boxed.finalize().await?;
    
    // Create another writer with same data to verify checksum consistency
    let mut writer2 = FileSystemWriter::new_with_compression(
        temp_dir.path().join("test_checksum2.dat"), 
        CompressionConfig::zstd_level(5)
    ).await?;
    
    writer2.write_chunk(test_data).await?;
    let checksum2 = writer2.checksum().expect("Checksum should be available");
    
    assert_eq!(checksum, checksum2, "Checksums should be identical for same data");
    
    let writer2_boxed = Box::new(writer2);
    writer2_boxed.finalize().await?;
    
    println!("✓ Checksum integrity test passed: {}", checksum);
    
    Ok(())
}

#[tokio::test]
async fn test_compression_disabled() -> Result<()> {
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("test_no_compression.dat");
    
    let test_data = b"Test data without compression";
    
    // Create writer without compression
    let mut writer = FileSystemWriter::new_with_compression(
        file_path.clone(), 
        CompressionConfig::None
    ).await?;
    
    // Write data
    writer.write_chunk(test_data).await?;
    
    // Verify compression is disabled
    assert!(!writer.compression().is_enabled());
    assert_eq!(writer.compression(), CompressionConfig::None);
    // When compression is disabled, compressed_bytes remains 0 (no compression tracking)
    assert_eq!(writer.compressed_bytes(), 0);
    assert_eq!(writer.compression_ratio(), 1.0);
    
    // Check that we actually wrote the data
    assert_eq!(writer.bytes_written(), test_data.len() as u64);
    
    // Finalize
    let writer_boxed = Box::new(writer);
    writer_boxed.finalize().await?;
    
    // Verify file exists without compression extension
    assert!(file_path.exists());
    assert!(!file_path.with_extension("dat.zst").exists());
    
    // Verify file size matches original data
    let file_size = std::fs::metadata(&file_path)?.len();
    assert_eq!(file_size, test_data.len() as u64);
    
    println!("✓ No compression test passed: {} bytes uncompressed", test_data.len());
    
    Ok(())
}

#[tokio::test]
async fn test_compression_levels() -> Result<()> {
    let temp_dir = tempdir()?;
    let test_data = b"Compression level test data. ".repeat(100);
    
    for level in [1, 3, 9, 22] {
        let file_path = temp_dir.path().join(format!("test_level_{}.dat", level));
        
        let mut writer = FileSystemWriter::new_with_compression(
            file_path.clone(), 
            CompressionConfig::zstd_level(level)
        ).await?;
        
        writer.write_chunk(&test_data).await?;
        
        // Verify compression level is stored correctly
        if let CompressionConfig::Zstd { level: stored_level } = writer.compression() {
            assert_eq!(stored_level, level);
        } else {
            panic!("Expected Zstd compression config");
        }
        
        let writer_boxed = Box::new(writer);
        writer_boxed.finalize().await?;
        
        // Verify compressed file exists
        let compressed_path = file_path.with_extension("dat.zst");
        assert!(compressed_path.exists());
        
        println!("✓ Compression level {} test passed", level);
    }
    
    Ok(())
}
