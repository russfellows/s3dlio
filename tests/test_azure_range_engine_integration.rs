// tests/test_azure_range_engine_integration.rs
//
// Integration tests for Azure Backend RangeEngine
// Tests concurrent range downloads with ACTUAL Azure Blob Storage
//
// Prerequisites:
// 1. Azure credentials configured (az login)
// 2. Environment variables:
//    - AZURE_BLOB_ACCOUNT: Your storage account name
//    - AZURE_BLOB_CONTAINER: Container to use for testing
// 3. Test data uploaded to the container

use anyhow::Result;
use bytes::Bytes;
use s3dlio::api::store_for_uri;
use s3dlio::object_store::{AzureConfig, AzureObjectStore, ObjectStore};
use std::env;

/// Helper to check if Azure integration tests should run
fn should_run_azure_tests() -> bool {
    env::var("AZURE_BLOB_ACCOUNT").is_ok() && env::var("AZURE_BLOB_CONTAINER").is_ok()
}

/// Helper to get test URI
fn get_test_uri(key: &str) -> String {
    let account = env::var("AZURE_BLOB_ACCOUNT").expect("AZURE_BLOB_ACCOUNT not set");
    let container = env::var("AZURE_BLOB_CONTAINER").expect("AZURE_BLOB_CONTAINER not set");
    format!("az://{}/{}/{}", account, container, key)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_small_blob_simple_download() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping Azure integration test - no credentials");
        println!("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER to run");
        return Ok(());
    }

    println!("🔧 Testing Azure small blob (< 4MB) - should use simple download");
    
    let test_uri = get_test_uri("test-small.bin");
    let store = AzureObjectStore::new();
    
    // Create test data < 4MB threshold using Bytes (zero-copy)
    let test_data = Bytes::from(vec![42u8; 1024 * 1024]); // 1MB
    
    // Upload test data - pass as slice reference (zero-copy)
    println!("📤 Uploading 1MB test file...");
    store.put(&test_uri, &test_data).await?;
    
    // Download and verify (should use simple download, not RangeEngine)
    println!("📥 Downloading with simple download (size < threshold)...");
    let downloaded = store.get(&test_uri).await?;
    
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded, test_data, "Data mismatch");
    
    // Clean up
    store.delete(&test_uri).await?;
    
    println!("✅ Azure small blob test passed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_large_blob_range_engine() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping Azure integration test - no credentials");
        println!("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER to run");
        return Ok(());
    }

    println!("🔧 Testing Azure large blob (> 4MB) - should use RangeEngine");
    
    let test_uri = get_test_uri("test-large-range-engine.bin");
    let store = AzureObjectStore::new();
    
    // Create test data > 4MB threshold (8MB to ensure RangeEngine triggers) using Bytes
    let size = 8 * 1024 * 1024; // 8MB
    let test_data = Bytes::from(vec![99u8; size]);
    
    // Upload test data - pass as slice reference (zero-copy)
    println!("📤 Uploading 8MB test file...");
    store.put(&test_uri, &test_data).await?;
    
    // Download and verify (should use RangeEngine for concurrent download)
    println!("📥 Downloading with RangeEngine (size > 4MB threshold)...");
    let start = std::time::Instant::now();
    let downloaded = store.get(&test_uri).await?;
    let elapsed = start.elapsed();
    
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded, test_data, "Data mismatch");
    
    let throughput_mbps = (size as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
    println!("   Throughput: {:.2} MB/s", throughput_mbps);
    println!("   Duration: {:?}", elapsed);
    
    // Clean up
    store.delete(&test_uri).await?;
    
    println!("✅ Azure large blob RangeEngine test passed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_range_engine_with_custom_config() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping Azure integration test - no credentials");
        println!("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER to run");
        return Ok(());
    }

    println!("🔧 Testing Azure RangeEngine with custom configuration");
    
    let test_uri = get_test_uri("test-custom-config.bin");
    
    // Create custom config with aggressive settings
    let mut config = AzureConfig::default();
    config.enable_range_engine = true;
    config.range_engine.chunk_size = 32 * 1024 * 1024; // 32MB chunks
    config.range_engine.max_concurrent_ranges = 16; // 16 parallel downloads
    config.range_engine.min_split_size = 2 * 1024 * 1024; // 2MB threshold
    
    let store = AzureObjectStore::with_config(config);
    
    // Create test data > 2MB threshold (6MB) using Bytes
    let size = 6 * 1024 * 1024; // 6MB
    let test_data = Bytes::from(vec![77u8; size]);
    
    // Upload test data - pass as slice reference (zero-copy)
    println!("📤 Uploading 6MB test file...");
    store.put(&test_uri, &test_data).await?;
    
    // Download with custom config
    println!("📥 Downloading with custom RangeEngine config (2MB threshold, 32MB chunks, 16 concurrent)...");
    let start = std::time::Instant::now();
    let downloaded = store.get(&test_uri).await?;
    let elapsed = start.elapsed();
    
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded, test_data, "Data mismatch");
    
    let throughput_mbps = (size as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
    println!("   Throughput: {:.2} MB/s", throughput_mbps);
    println!("   Duration: {:?}", elapsed);
    
    // Clean up
    store.delete(&test_uri).await?;
    
    println!("✅ Azure custom config test passed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_range_engine_disabled() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping Azure integration test - no credentials");
        println!("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER to run");
        return Ok(());
    }

    println!("🔧 Testing Azure with RangeEngine disabled");
    
    let test_uri = get_test_uri("test-disabled-range-engine.bin");
    
    // Create config with RangeEngine disabled
    let mut config = AzureConfig::default();
    config.enable_range_engine = false;
    
    let store = AzureObjectStore::with_config(config);
    
    // Create large test data (should still use simple download even though > threshold) using Bytes
    let size = 8 * 1024 * 1024; // 8MB
    let test_data = Bytes::from(vec![88u8; size]);
    
    // Upload test data - pass as slice reference (zero-copy)
    println!("📤 Uploading 8MB test file...");
    store.put(&test_uri, &test_data).await?;
    
    // Download with RangeEngine disabled (should use simple download)
    println!("📥 Downloading with RangeEngine DISABLED (simple download for 8MB file)...");
    let start = std::time::Instant::now();
    let downloaded = store.get(&test_uri).await?;
    let elapsed = start.elapsed();
    
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded, test_data, "Data mismatch");
    
    let throughput_mbps = (size as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();
    println!("   Throughput: {:.2} MB/s", throughput_mbps);
    println!("   Duration: {:?}", elapsed);
    
    // Clean up
    store.delete(&test_uri).await?;
    
    println!("✅ Azure RangeEngine disabled test passed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_via_factory_function() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping Azure integration test - no credentials");
        println!("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER to run");
        return Ok(());
    }

    println!("🔧 Testing Azure via store_for_uri() factory");
    
    let test_uri = get_test_uri("test-factory.bin");
    
    // Create store via factory function
    let store = store_for_uri(&test_uri)?;
    
    // Create test data using Bytes
    let test_data = Bytes::from(vec![55u8; 5 * 1024 * 1024]); // 5MB
    
    // Upload - pass as slice reference (zero-copy)
    println!("📤 Uploading 5MB test file via factory-created store...");
    store.put(&test_uri, &test_data).await?;
    
    // Download (should use RangeEngine since > 4MB)
    println!("📥 Downloading via factory-created store...");
    let downloaded = store.get(&test_uri).await?;
    
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded, test_data, "Data mismatch");
    
    // Clean up
    store.delete(&test_uri).await?;
    
    println!("✅ Azure factory function test passed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_very_large_blob() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping Azure integration test - no credentials");
        println!("   Set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER to run");
        return Ok(());
    }

    // Skip this test if AZURE_RUN_LARGE_TESTS is not set (can be expensive/slow)
    if env::var("AZURE_RUN_LARGE_TESTS").is_err() {
        println!("⚠️  Skipping large blob test - set AZURE_RUN_LARGE_TESTS=1 to run");
        return Ok(());
    }

    println!("🔧 Testing Azure very large blob (100MB) - RangeEngine performance test");
    
    let test_uri = get_test_uri("test-very-large.bin");
    let store = AzureObjectStore::new();
    
    // Create 100MB test data using Bytes
    let size = 100 * 1024 * 1024; // 100MB
    println!("📤 Uploading 100MB test file...");
    let test_data = Bytes::from(vec![33u8; size]);
    
    let upload_start = std::time::Instant::now();
    store.put(&test_uri, &test_data).await?;
    let upload_elapsed = upload_start.elapsed();
    let upload_mbps = (size as f64 / 1024.0 / 1024.0) / upload_elapsed.as_secs_f64();
    
    println!("   Upload: {:.2} MB/s ({:?})", upload_mbps, upload_elapsed);
    
    // Download with RangeEngine
    println!("📥 Downloading 100MB with RangeEngine...");
    let download_start = std::time::Instant::now();
    let downloaded = store.get(&test_uri).await?;
    let download_elapsed = download_start.elapsed();
    let download_mbps = (size as f64 / 1024.0 / 1024.0) / download_elapsed.as_secs_f64();
    
    println!("   Download: {:.2} MB/s ({:?})", download_mbps, download_elapsed);
    
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    
    // Verify first and last chunks to avoid full comparison overhead
    assert_eq!(&downloaded[0..1024], &test_data[0..1024], "Start mismatch");
    assert_eq!(&downloaded[size-1024..size], &test_data[size-1024..size], "End mismatch");
    
    // Clean up
    store.delete(&test_uri).await?;
    
    println!("✅ Azure very large blob test passed");
    println!("   Expected improvement: 30-50% faster than simple download");
    Ok(())
}
