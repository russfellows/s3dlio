// tests/test_api_storage_config.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Tests for unified StorageConfig API (Issue #85)
//!
//! This test validates that the FileSystemConfig type mismatch is fixed and
//! that users can configure page cache mode via the public API.

use anyhow::Result;
use s3dlio::api::{
    FileSystemConfig, DirectFileSystemConfig, StorageConfig,
    PageCacheMode, store_for_uri_with_config,
};
use s3dlio::range_engine_generic::RangeEngineConfig;
use std::time::Duration;
use tempfile::TempDir;
use std::fs::File;
use std::io::Write;

#[tokio::test]
async fn test_file_config_public_api() -> Result<()> {
    // Create temp directory with test file
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Hello, world!")?;
    drop(file);
    
    // Create FileSystemConfig with page cache mode
    let config = FileSystemConfig {
        enable_range_engine: false,
        range_engine: Default::default(),
        page_cache_mode: Some(PageCacheMode::Sequential),
    };
    
    let uri = format!("file://{}", test_file.display());
    
    // This should compile and work now!
    let store = store_for_uri_with_config(&uri, Some(StorageConfig::File(config)))?;
    
    // Verify it works by reading the file
    let data = store.get(&uri).await?;
    assert_eq!(&data[..], b"Hello, world!");
    
    Ok(())
}

#[tokio::test]
async fn test_file_config_random_mode() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Random access test data")?;
    drop(file);
    
    let config = FileSystemConfig {
        enable_range_engine: false,
        range_engine: Default::default(),
        page_cache_mode: Some(PageCacheMode::Random),
    };
    
    let uri = format!("file://{}", test_file.display());
    let store = store_for_uri_with_config(&uri, Some(StorageConfig::File(config)))?;
    
    let data = store.get(&uri).await?;
    assert_eq!(&data[..], b"Random access test data");
    
    Ok(())
}

#[tokio::test]
async fn test_file_config_auto_mode() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Auto mode test")?;
    drop(file);
    
    let config = FileSystemConfig {
        enable_range_engine: false,
        range_engine: Default::default(),
        page_cache_mode: None,  // Auto mode (default)
    };
    
    let uri = format!("file://{}", test_file.display());
    let store = store_for_uri_with_config(&uri, Some(StorageConfig::File(config)))?;
    
    let data = store.get(&uri).await?;
    assert_eq!(&data[..], b"Auto mode test");
    
    Ok(())
}

#[tokio::test]
async fn test_direct_config_not_for_file_uri() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Test data")?;
    drop(file);
    
    // Create DirectFileSystemConfig
    let config = DirectFileSystemConfig {
        direct_io: true,
        alignment: 4096,
        min_io_size: 4096,
        sync_writes: false,
        enable_range_engine: false,
        range_engine: Default::default(),
        buffer_pool: None,
    };
    
    let uri = format!("file://{}", test_file.display());
    
    // This should fail with clear error message
    let result = store_for_uri_with_config(&uri, Some(StorageConfig::Direct(config)));
    
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("Cannot use DirectFileSystemConfig with file://"));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_file_config_not_for_direct_uri() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Test data")?;
    drop(file);
    
    // Create FileSystemConfig
    let config = FileSystemConfig {
        enable_range_engine: false,
        range_engine: Default::default(),
        page_cache_mode: Some(PageCacheMode::Sequential),
    };
    
    let uri = format!("direct://{}", test_file.display());
    
    // This should fail with clear error message
    let result = store_for_uri_with_config(&uri, Some(StorageConfig::File(config)));
    
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("Cannot use FileSystemConfig with direct://"));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_no_config_still_works() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let mut file = File::create(&test_file)?;
    file.write_all(b"Default config test")?;
    drop(file);
    
    let uri = format!("file://{}", test_file.display());
    
    // Should work with None config
    let store = store_for_uri_with_config(&uri, None)?;
    let data = store.get(&uri).await?;
    assert_eq!(&data[..], b"Default config test");
    
    Ok(())
}

#[tokio::test]
async fn test_range_engine_with_page_cache() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("large.txt");
    let mut file = File::create(&test_file)?;
    
    // Create a larger file (1MB) to potentially trigger range engine
    let data = vec![b'A'; 1024 * 1024];
    file.write_all(&data)?;
    drop(file);
    
    let config = FileSystemConfig {
        enable_range_engine: true,
        range_engine: RangeEngineConfig {
            chunk_size: 256 * 1024,           // 256KB chunks
            max_concurrent_ranges: 4,
            min_split_size: 512 * 1024,       // 512KB threshold
            range_timeout: Duration::from_secs(60),
        },
        page_cache_mode: Some(PageCacheMode::Sequential),
    };
    
    let uri = format!("file://{}", test_file.display());
    let store = store_for_uri_with_config(&uri, Some(StorageConfig::File(config)))?;
    
    let result = store.get(&uri).await?;
    assert_eq!(result.len(), 1024 * 1024);
    assert_eq!(result[0], b'A');
    
    Ok(())
}
