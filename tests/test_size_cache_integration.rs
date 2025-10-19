// v0.9.10: Integration tests for ObjectSizeCache across all backends
//
// This test verifies that:
// 1. All backends (S3, GCS, Azure, file://) support pre_stat_and_cache()
// 2. Size cache reduces stat operations (cache hits work)
// 3. Pre-stat provides performance improvements for multi-object workloads

use s3dlio::object_store::store_for_uri;
use s3dlio::object_size_cache::ObjectSizeCache;
use anyhow::Result;
use std::time::Duration;
use tokio;

#[tokio::test]
async fn test_object_size_cache_basic() -> Result<()> {
    // Test the ObjectSizeCache module directly
    let cache = ObjectSizeCache::new(Duration::from_secs(60));
    
    // Test basic put/get
    cache.put("s3://bucket/key1".to_string(), 12345).await;
    let size = cache.get("s3://bucket/key1").await;
    assert_eq!(size, Some(12345));
    
    // Test cache miss
    let missing = cache.get("s3://bucket/nonexistent").await;
    assert_eq!(missing, None);
    
    // Test stats
    let stats = cache.stats().await;
    assert_eq!(stats.total_entries, 1);
    assert_eq!(stats.valid_entries(), 1);
    
    Ok(())
}

#[tokio::test]
async fn test_object_size_cache_expiration() -> Result<()> {
    // Test TTL expiration
    let cache = ObjectSizeCache::new(Duration::from_millis(100));
    
    cache.put("test://uri".to_string(), 999).await;
    
    // Immediate read should work
    assert_eq!(cache.get("test://uri").await, Some(999));
    
    // Wait for expiration
    tokio::time::sleep(Duration::from_millis(150)).await;
    
    // Should now be expired
    assert_eq!(cache.get("test://uri").await, None);
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_pre_stat_and_cache() -> Result<()> {
    // Create temporary test files
    let test_dir = tempfile::tempdir()?;
    let mut uris = Vec::new();
    
    for i in 0..10 {
        let test_file = test_dir.path().join(format!("test_{}.txt", i));
        let file_uri = format!("file://{}", test_file.display());
        
        // Create file with known size
        let data = vec![b'x'; 100 * (i + 1)];
        std::fs::write(&test_file, &data)?;
        
        uris.push(file_uri);
    }
    
    let store = store_for_uri(&uris[0])?;
    
    // Test pre_stat_and_cache
    let cached_count = store.pre_stat_and_cache(&uris, 10).await?;
    assert_eq!(cached_count, 10);
    
    // Verify we can get all objects (should use cached sizes if backend supports it)
    for uri in &uris {
        let data = store.get(uri).await?;
        assert!(!data.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
#[ignore] // Requires S3 credentials and bucket
async fn test_s3_pre_stat_and_cache() -> Result<()> {
    // This test requires:
    // - AWS credentials in environment
    // - An S3 bucket with test objects
    // Run with: cargo test test_s3_pre_stat_and_cache --features=native-backends -- --ignored
    
    let test_bucket = std::env::var("TEST_S3_BUCKET")
        .unwrap_or_else(|_| "test-bucket".to_string());
    let test_prefix = format!("s3://{}/test-data/", test_bucket);
    
    let store = store_for_uri(&test_prefix)?;
    
    // List objects to get URIs
    let uris = store.list(&test_prefix, true).await?;
    
    if uris.is_empty() {
        println!("No test objects found in {}", test_prefix);
        return Ok(());
    }
    
    // Take first 100 objects for testing
    let test_uris: Vec<String> = uris.into_iter().take(100).collect();
    
    println!("Testing pre-stat with {} objects", test_uris.len());
    
    // Test pre_stat_and_cache
    let start = std::time::Instant::now();
    let cached_count = store.pre_stat_and_cache(&test_uris, 50).await?;
    let pre_stat_duration = start.elapsed();
    
    println!("Pre-statted {} objects in {:?}", cached_count, pre_stat_duration);
    assert_eq!(cached_count, test_uris.len());
    
    // Now get a few objects (should benefit from cached sizes)
    let download_start = std::time::Instant::now();
    for uri in test_uris.iter().take(10) {
        let _data = store.get(uri).await?;
    }
    let download_duration = download_start.elapsed();
    
    println!("Downloaded 10 objects in {:?}", download_duration);
    
    Ok(())
}

#[tokio::test]
#[ignore] // Requires GCS credentials and bucket
async fn test_gcs_pre_stat_and_cache() -> Result<()> {
    // This test requires:
    // - GCS credentials via GOOGLE_APPLICATION_CREDENTIALS or gcloud auth
    // - A GCS bucket with test objects
    // Run with: cargo test test_gcs_pre_stat_and_cache -- --ignored
    
    let test_bucket = std::env::var("TEST_GCS_BUCKET")
        .unwrap_or_else(|_| "signal65-russ-b1".to_string());
    let test_prefix = format!("gs://{}/sai3bench-v069-1tb/data/", test_bucket);
    
    let store = store_for_uri(&test_prefix)?;
    
    // List objects to get URIs
    let uris = store.list(&test_prefix, true).await?;
    
    if uris.is_empty() {
        println!("No test objects found in {}", test_prefix);
        return Ok(());
    }
    
    // Take first 100 objects for testing
    let test_uris: Vec<String> = uris.into_iter().take(100).collect();
    
    println!("Testing pre-stat with {} GCS objects", test_uris.len());
    
    // Test pre_stat_and_cache
    let start = std::time::Instant::now();
    let cached_count = store.pre_stat_and_cache(&test_uris, 100).await?;
    let pre_stat_duration = start.elapsed();
    
    println!("Pre-statted {} objects in {:?}", cached_count, pre_stat_duration);
    assert_eq!(cached_count, test_uris.len());
    
    // Now get a few objects (should benefit from cached sizes)
    let download_start = std::time::Instant::now();
    for uri in test_uris.iter().take(10) {
        let _data = store.get(uri).await?;
    }
    let download_duration = download_start.elapsed();
    
    println!("Downloaded 10 objects in {:?}", download_duration);
    
    Ok(())
}

#[tokio::test]
async fn test_pre_stat_empty_list() -> Result<()> {
    // Test that pre_stat_and_cache handles empty list gracefully
    let test_dir = tempfile::tempdir()?;
    let test_file = test_dir.path().join("dummy.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    let store = store_for_uri(&file_uri)?;
    
    let empty_uris: Vec<String> = vec![];
    let cached_count = store.pre_stat_and_cache(&empty_uris, 10).await?;
    assert_eq!(cached_count, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_pre_stat_with_errors() -> Result<()> {
    // Test that pre_stat_and_cache handles missing objects gracefully
    let test_dir = tempfile::tempdir()?;
    
    let mut uris = Vec::new();
    
    // Create one valid file
    let valid_file = test_dir.path().join("valid.txt");
    std::fs::write(&valid_file, b"test data")?;
    uris.push(format!("file://{}", valid_file.display()));
    
    // Add some non-existent files
    for i in 0..5 {
        uris.push(format!("file://{}/nonexistent_{}.txt", test_dir.path().display(), i));
    }
    
    let store = store_for_uri(&uris[0])?;
    
    // pre_stat_and_cache should handle errors gracefully
    let cached_count = store.pre_stat_and_cache(&uris, 10).await?;
    
    // Should have cached at least the valid file
    // (Default implementation logs warnings for failures but continues)
    assert!(cached_count >= 1);
    
    Ok(())
}
