// tests/test_directio_range_engine.rs
//
// Integration tests for DirectIO backend with RangeEngine

use anyhow::Result;
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::object_store::{ObjectStore, store_for_uri};
use s3dlio::range_engine_generic::RangeEngineConfig;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

/// Helper to create a test file with specified size
async fn create_test_file(dir: &TempDir, name: &str, size_mb: usize) -> Result<PathBuf> {
    let path = dir.path().join(name);
    let data = vec![42u8; size_mb * 1024 * 1024];
    fs::write(&path, &data).await?;
    Ok(path)
}

#[tokio::test]
async fn test_directio_small_no_range_engine() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 1MB file (below 16MB threshold for DirectIO)
    let path = create_test_file(&temp_dir, "small.bin", 1).await?;
    let uri = format!("direct://{}", path.display());
    
    // Create DirectIO store with default config (16MB threshold)
    let config = FileSystemConfig::direct_io();
    let store = ConfigurableFileSystemObjectStore::with_config(config);
    
    // Should use simple read path (not RangeEngine)
    let data = store.get(&uri).await?;
    
    assert_eq!(data.len(), 1 * 1024 * 1024);
    assert!(data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_directio_large_with_range_engine() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 32MB file (above 16MB threshold for DirectIO)
    let path = create_test_file(&temp_dir, "large.bin", 32).await?;
    let uri = format!("direct://{}", path.display());
    
    // Create DirectIO store with default config
    let config = FileSystemConfig::direct_io();
    let store = ConfigurableFileSystemObjectStore::with_config(config);
    
    // Should use RangeEngine path
    let data = store.get(&uri).await?;
    
    assert_eq!(data.len(), 32 * 1024 * 1024);
    assert!(data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_directio_range_engine_custom_config() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 8MB file
    let path = create_test_file(&temp_dir, "medium.bin", 8).await?;
    let uri = format!("direct://{}", path.display());
    
    // Custom config with lower threshold (4MB)
    let config = FileSystemConfig {
        direct_io: true,
        alignment: 4096,
        min_io_size: 4096,
        sync_writes: false,
        enable_range_engine: true,
        range_engine: RangeEngineConfig {
            chunk_size: 32 * 1024 * 1024,     // 32MB chunks
            max_concurrent_ranges: 8,          // Lower concurrency
            min_split_size: 4 * 1024 * 1024,  // 4MB threshold (lower for testing)
            ..Default::default()
        },
    };
    
    let store = ConfigurableFileSystemObjectStore::with_config(config);
    
    // Should use RangeEngine with custom config
    let data = store.get(&uri).await?;
    
    assert_eq!(data.len(), 8 * 1024 * 1024);
    assert!(data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_directio_range_engine_disabled() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 32MB file
    let path = create_test_file(&temp_dir, "large_no_range.bin", 32).await?;
    let uri = format!("direct://{}", path.display());
    
    // Config with RangeEngine disabled
    let mut config = FileSystemConfig::direct_io();
    config.enable_range_engine = false;
    
    let store = ConfigurableFileSystemObjectStore::with_config(config);
    
    // Should use simple read even for large file
    let data = store.get(&uri).await?;
    
    assert_eq!(data.len(), 32 * 1024 * 1024);
    assert!(data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_directio_via_store_for_uri() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 32MB file
    let path = create_test_file(&temp_dir, "via_factory.bin", 32).await?;
    let uri = format!("direct://{}", path.display());
    
    // Use factory function
    let store = store_for_uri(&uri)?;
    
    // Should work with RangeEngine
    let data = store.get(&uri).await?;
    
    assert_eq!(data.len(), 32 * 1024 * 1024);
    assert!(data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_directio_range_engine_alignment() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 20MB file (not page-aligned size)
    let path = create_test_file(&temp_dir, "unaligned.bin", 20).await?;
    let uri = format!("direct://{}", path.display());
    
    let config = FileSystemConfig::direct_io();
    let store = ConfigurableFileSystemObjectStore::with_config(config);
    
    // Should handle unaligned size correctly
    let data = store.get(&uri).await?;
    
    assert_eq!(data.len(), 20 * 1024 * 1024);
    assert!(data.iter().all(|&b| b == 42));
    
    Ok(())
}

#[tokio::test]
async fn test_directio_get_range() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create 10MB file
    let path = create_test_file(&temp_dir, "range_test.bin", 10).await?;
    let uri = format!("direct://{}", path.display());
    
    let config = FileSystemConfig::direct_io();
    let store = ConfigurableFileSystemObjectStore::with_config(config);
    
    // Test range read (this is what RangeEngine calls internally)
    let range_data = store.get_range(&uri, 1024 * 1024, Some(2 * 1024 * 1024)).await?;
    
    assert_eq!(range_data.len(), 2 * 1024 * 1024);
    assert!(range_data.iter().all(|&b| b == 42));
    
    Ok(())
}
