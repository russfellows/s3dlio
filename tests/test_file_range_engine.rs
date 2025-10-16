// tests/test_file_range_engine.rs
//
// Integration tests for File backend with RangeEngine

use s3dlio::object_store::{ObjectStore, store_for_uri};
use s3dlio::file_store::{FileSystemObjectStore, FileSystemConfig};
use s3dlio::range_engine_generic::RangeEngineConfig;
use anyhow::Result;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Create a test file with known content
fn create_test_file(path: &Path, size: usize) -> Result<()> {
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    fs::write(path, data)?;
    Ok(())
}

#[tokio::test]
async fn test_file_small_no_range_engine() -> Result<()> {
    // Create 1MB test file (< 4MB threshold)
    let path = "/tmp/test_small_file.bin";
    let size = 1024 * 1024; // 1MB
    create_test_file(Path::new(path), size)?;
    
    let store = FileSystemObjectStore::new();
    let uri = format!("file://{}", path);
    
    let bytes = store.get(&uri).await?;
    
    assert_eq!(bytes.len(), size);
    
    // Verify content
    for (i, &byte) in bytes.iter().enumerate() {
        assert_eq!(byte, (i % 256) as u8, "Mismatch at byte {}", i);
    }
    
    // Cleanup
    fs::remove_file(path)?;
    
    Ok(())
}

#[tokio::test]
async fn test_file_large_with_range_engine() -> Result<()> {
    // Create 10MB test file (> 4MB threshold, will use RangeEngine)
    let path = "/tmp/test_large_file.bin";
    let size = 10 * 1024 * 1024; // 10MB
    create_test_file(Path::new(path), size)?;
    
    let store = FileSystemObjectStore::new();
    let uri = format!("file://{}", path);
    
    let start = Instant::now();
    let bytes = store.get(&uri).await?;
    let elapsed = start.elapsed();
    
    println!(
        "Downloaded {} bytes in {:?} ({:.2} MB/s)",
        bytes.len(),
        elapsed,
        (bytes.len() as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64()
    );
    
    assert_eq!(bytes.len(), size);
    
    // Verify content (sample check)
    for i in (0..size).step_by(1024) {
        assert_eq!(bytes[i], (i % 256) as u8, "Mismatch at byte {}", i);
    }
    
    // Cleanup
    fs::remove_file(path)?;
    
    Ok(())
}

#[tokio::test]
async fn test_file_range_engine_custom_config() -> Result<()> {
    // Create 8MB test file
    let path = "/tmp/test_custom_config.bin";
    let size = 8 * 1024 * 1024; // 8MB
    create_test_file(Path::new(path), size)?;
    
    // Custom config: smaller chunks, lower concurrency, lower threshold
    let config = FileSystemConfig {
        enable_range_engine: true,
        range_engine: RangeEngineConfig {
            chunk_size: 1024 * 1024,    // 1MB chunks
            max_concurrent_ranges: 4,    // Only 4 concurrent
            min_split_size: 2 * 1024 * 1024, // 2MB threshold
            ..Default::default()
        },
        page_cache_mode: None,  // Use default Auto mode
    };
    
    let store = FileSystemObjectStore::with_config(config);
    let uri = format!("file://{}", path);
    
    let bytes = store.get(&uri).await?;
    
    assert_eq!(bytes.len(), size);
    
    // Verify content
    for i in (0..size).step_by(4096) {
        assert_eq!(bytes[i], (i % 256) as u8, "Mismatch at byte {}", i);
    }
    
    // Cleanup
    fs::remove_file(path)?;
    
    Ok(())
}

#[tokio::test]
async fn test_file_range_engine_disabled() -> Result<()> {
    // Create 10MB test file
    let path = "/tmp/test_range_disabled.bin";
    let size = 10 * 1024 * 1024; // 10MB
    create_test_file(Path::new(path), size)?;
    
    // Disable RangeEngine
    let config = FileSystemConfig {
        enable_range_engine: false,
        ..Default::default()
    };
    
    let store = FileSystemObjectStore::with_config(config);
    let uri = format!("file://{}", path);
    
    let bytes = store.get(&uri).await?;
    
    assert_eq!(bytes.len(), size);
    
    // Cleanup
    fs::remove_file(path)?;
    
    Ok(())
}

#[tokio::test]
async fn test_file_via_store_for_uri() -> Result<()> {
    // Test that store_for_uri returns correctly configured store
    let path = "/tmp/test_store_for_uri.bin";
    let size = 5 * 1024 * 1024; // 5MB
    create_test_file(Path::new(path), size)?;
    
    let uri = format!("file://{}", path);
    let store = store_for_uri(&uri)?;
    
    let bytes = store.get(&uri).await?;
    
    assert_eq!(bytes.len(), size);
    
    // Cleanup
    fs::remove_file(path)?;
    
    Ok(())
}
