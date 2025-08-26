use s3dlio::*;
use anyhow::Result;
use tokio::fs;
use tempfile::TempDir;

#[tokio::test]
async fn test_checksum_computation_across_backends() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_data = b"Hello, World! This is test data for checksum validation.";
    
    // Test FileSystemWriter checksum through ObjectStore
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    let test_key = "test_file.txt";
    let full_uri = format!("{}/{}", base_uri, test_key);
    let mut writer = store.get_writer(&full_uri).await?;
    
    writer.write_chunk(test_data).await?;
    let checksum = writer.checksum();
    assert!(checksum.is_some());
    let checksum_str = checksum.unwrap();
    assert!(checksum_str.starts_with("crc32c:"));
    
    // Finalize the writer
    writer.finalize().await?;
    
    // Verify file was written correctly
    let file_path = temp_dir.path().join(test_key);
    let written_data = fs::read(&file_path).await?;
    assert_eq!(written_data, test_data);
    
    println!("FileSystem checksum: {}", checksum_str);
    Ok(())
}

#[tokio::test]
async fn test_checksum_consistency() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_data = b"Consistency test data for checksum validation.";
    
    // Write same data with different files and compare checksums
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // First file
    let full_uri1 = format!("{}/file1.txt", base_uri);
    let mut writer1 = store.get_writer(&full_uri1).await?;
    writer1.write_chunk(test_data).await?;
    let checksum1 = writer1.checksum().unwrap();
    writer1.finalize().await?;
    
    // Second file
    let full_uri2 = format!("{}/file2.txt", base_uri);
    let mut writer2 = store.get_writer(&full_uri2).await?;
    writer2.write_chunk(test_data).await?;
    let checksum2 = writer2.checksum().unwrap();
    writer2.finalize().await?;
    
    // Checksums should be identical for the same data
    assert_eq!(checksum1, checksum2);
    println!("Consistent checksum: {}", checksum1);
    
    Ok(())
}

#[tokio::test]
async fn test_incremental_checksum_updates() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    let full_uri = format!("{}/incremental_test.txt", base_uri);
    let mut writer = store.get_writer(&full_uri).await?;
    
    // Write data in multiple chunks
    writer.write_chunk(b"First chunk ").await?;
    writer.write_chunk(b"Second chunk ").await?;
    writer.write_chunk(b"Third chunk").await?;
    
    let checksum = writer.checksum().unwrap();
    assert!(checksum.starts_with("crc32c:"));
    
    writer.finalize().await?;
    
    // Verify the final file contains all chunks
    let file_path = temp_dir.path().join("incremental_test.txt");
    let written_data = fs::read(&file_path).await?;
    assert_eq!(written_data, b"First chunk Second chunk Third chunk");
    
    println!("Incremental checksum: {}", checksum);
    Ok(())
}

#[tokio::test]
async fn test_checksum_format() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    let full_uri = format!("{}/format_test.txt", base_uri);
    let mut writer = store.get_writer(&full_uri).await?;
    writer.write_chunk(b"Test data").await?;
    
    let checksum = writer.checksum().unwrap();
    
    // Verify format is "crc32c:xxxxxxxx" where x is hex digit
    assert!(checksum.starts_with("crc32c:"));
    let hex_part = &checksum[7..]; // Skip "crc32c:"
    assert_eq!(hex_part.len(), 8); // CRC32 is 4 bytes = 8 hex chars
    
    // Verify all characters after prefix are valid hex
    for c in hex_part.chars() {
        assert!(c.is_ascii_hexdigit());
    }
    
    writer.finalize().await?;
    println!("Checksum format validated: {}", checksum);
    
    Ok(())
}

#[tokio::test]
async fn test_checksum_different_data() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_uri = format!("file://{}", temp_dir.path().display());
    let store = store_for_uri(&base_uri)?;
    
    // Write different data and verify checksums are different
    let data1 = b"First set of data";
    let data2 = b"Second set of data";
    
    let full_uri1 = format!("{}/data1.txt", base_uri);
    let mut writer1 = store.get_writer(&full_uri1).await?;
    writer1.write_chunk(data1).await?;
    let checksum1 = writer1.checksum().unwrap();
    writer1.finalize().await?;
    
    let full_uri2 = format!("{}/data2.txt", base_uri);
    let mut writer2 = store.get_writer(&full_uri2).await?;
    writer2.write_chunk(data2).await?;
    let checksum2 = writer2.checksum().unwrap();
    writer2.finalize().await?;
    
    // Checksums should be different for different data
    assert_ne!(checksum1, checksum2);
    println!("Checksum 1: {}", checksum1);
    println!("Checksum 2: {}", checksum2);
    
    Ok(())
}
