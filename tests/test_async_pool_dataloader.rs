// tests/test_async_pool_dataloader.rs
//
// Tests for enhanced async pooling dataloader with multi-backend support

use anyhow::Result;
use s3dlio::data_loader::{
    async_pool_dataloader::{AsyncPoolDataLoader, MultiBackendDataset, PoolConfig}, 
    LoaderOptions, Dataset
};
use s3dlio::object_store::store_for_uri;
use std::time::Duration;
use tempfile::TempDir;
use tokio_stream::StreamExt;

/// Helper to create test files with different sizes
async fn create_test_files(base_dir: &std::path::Path, count: usize) -> Result<Vec<String>> {
    let mut uris = Vec::new();
    
    for i in 0..count {
        let file_path = base_dir.join(format!("test_file_{:03}.dat", i));
        let uri = format!("file://{}", file_path.display());
        
        // Create files with varying sizes to simulate different read times
        let size = 1024 * (i % 10 + 1); // 1KB to 10KB files
        let data = vec![i as u8; size];
        
        let store = store_for_uri(&uri)?;
        store.put(&uri, &data).await?;
        
        uris.push(uri);
    }
    
    Ok(uris)
}

#[tokio::test]
async fn test_multi_backend_dataset_creation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let uris = create_test_files(temp_dir.path(), 5).await?;
    
    // Test creation from URI list
    let dataset = MultiBackendDataset::from_uris(uris.clone())?;
    assert_eq!(dataset.len(), 5);
    
    // Test data retrieval
    let data = dataset.get(0).await?;
    assert_eq!(data.len(), 1024); // First file should be 1KB
    
    let data = dataset.get(4).await?;
    assert_eq!(data.len(), 5120); // Fifth file should be 5KB
    
    Ok(())
}

#[tokio::test]
async fn test_async_pool_basic_batching() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let uris = create_test_files(temp_dir.path(), 20).await?;
    
    let dataset = MultiBackendDataset::from_uris(uris)?;
    let options = LoaderOptions {
        batch_size: 4,
        drop_last: false,
        ..Default::default()
    };
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream();
    
    let mut total_items = 0;
    let mut batch_count = 0;
    
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        batch_count += 1;
        total_items += batch.len();
        
        // Most batches should be size 4, except possibly the last
        if batch_count < 5 {  // First 4 batches
            assert_eq!(batch.len(), 4, "Batch {} should have 4 items", batch_count);
        }
        
        // Verify data integrity
        for data in batch {
            assert!(!data.is_empty(), "Data should not be empty");
        }
    }
    
    assert_eq!(total_items, 20, "Should process all 20 items");
    assert_eq!(batch_count, 5, "Should create 5 batches (4+4+4+4+4)");
    
    Ok(())
}

#[tokio::test]
async fn test_async_pool_out_of_order_completion() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create files with dramatically different sizes to force different completion times
    let mut uris = Vec::new();
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("test_file_{:03}.dat", i));
        let uri = format!("file://{}", file_path.display());
        
        // Alternate between small (1KB) and large (100KB) files
        let size = if i % 2 == 0 { 1024 } else { 100 * 1024 };
        let data = vec![i as u8; size];
        
        let store = store_for_uri(&uri)?;
        store.put(&uri, &data).await?;
        
        uris.push(uri);
    }
    
    let dataset = MultiBackendDataset::from_uris(uris)?;
    let options = LoaderOptions {
        batch_size: 3,
        drop_last: false,
        ..Default::default()
    };
    
    let pool_config = PoolConfig {
        pool_size: 8,  // High concurrency
        readahead_batches: 2,
        batch_timeout: Duration::from_millis(500),
        max_inflight: 16,
    };
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream_with_pool(pool_config);
    
    let mut total_items = 0;
    let mut batch_sizes = Vec::new();
    
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        batch_sizes.push(batch.len());
        total_items += batch.len();
        
        // Verify we get valid data
        for (idx, data) in batch.iter().enumerate() {
            assert!(!data.is_empty(), "Batch item {} should not be empty", idx);
        }
    }
    
    assert_eq!(total_items, 10, "Should process all 10 items");
    
    // With out-of-order completion, we might get different batch compositions
    // but still the right total
    println!("Batch sizes: {:?}", batch_sizes);
    
    Ok(())
}

#[tokio::test]
async fn test_multi_backend_support() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create files for file:// backend
    let file_uris = create_test_files(temp_dir.path(), 5).await?;
    
    // Test file:// backend
    let dataset = MultiBackendDataset::from_uris(file_uris)?;
    let options = LoaderOptions {
        batch_size: 2,
        drop_last: false,
        ..Default::default()
    };
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream();
    
    let mut batches = 0;
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        batches += 1;
        assert!(!batch.is_empty(), "Batch should not be empty");
    }
    
    assert_eq!(batches, 3, "Should get 3 batches (2+2+1)");
    
    // TODO: Add S3 and Azure tests when running with proper credentials
    // This would test:
    // - s3:// URIs with AsyncPoolDataLoader
    // - az:// URIs with AsyncPoolDataLoader  
    // - Mixed backend URIs (if supported)
    
    Ok(())
}

#[tokio::test]
async fn test_pool_configuration_effects() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let uris = create_test_files(temp_dir.path(), 12).await?;
    
    let dataset = MultiBackendDataset::from_uris(uris)?;
    let options = LoaderOptions {
        batch_size: 3,
        drop_last: true,  // Should drop incomplete last batch
        ..Default::default()
    };
    
    // Test with small pool vs large pool
    for &pool_size in &[2, 8] {
        let pool_config = PoolConfig {
            pool_size,
            readahead_batches: 2,
            batch_timeout: Duration::from_millis(1000),
            max_inflight: pool_size * 2,
        };
        
        let dataloader = AsyncPoolDataLoader::new(dataset.clone(), options.clone());
        let mut stream = dataloader.stream_with_pool(pool_config);
        
        let mut total_items = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            total_items += batch.len();
            assert_eq!(batch.len(), 3, "All batches should be size 3 with drop_last=true");
        }
        
        assert_eq!(total_items, 12, "Should process 12 items (4 complete batches)");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    // Create dataset with some valid and some invalid URIs
    let mut uris = Vec::new();
    
    // Add some invalid URIs that should cause errors
    uris.push("file:///nonexistent/path/file1.dat".to_string());
    uris.push("file:///nonexistent/path/file2.dat".to_string());
    
    let dataset = MultiBackendDataset::from_uris(uris)?;
    let options = LoaderOptions {
        batch_size: 2,
        drop_last: false,
        ..Default::default()
    };
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream();
    
    // Should get an error when trying to read nonexistent files
    if let Some(batch_result) = stream.next().await {
        assert!(batch_result.is_err(), "Should get error for nonexistent files");
    }
    
    Ok(())
}
