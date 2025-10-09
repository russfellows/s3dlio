// tests/test_gcs_smoke.rs
//
// Quick smoke tests for GCS Backend with RangeEngine
// Tests key ObjectStore trait methods with ACTUAL Google Cloud Storage
//
// Prerequisites:
// 1. GCS authentication: gcloud auth application-default login
// 2. Environment variable:
//    - GCS_TEST_BUCKET: Your GCS bucket name (e.g., signal65-russ-b1)
//
// Run: cargo test --release --test test_gcs_smoke -- --nocapture --test-threads=1

use anyhow::Result;
use bytes::Bytes;
use s3dlio::object_store::{GcsObjectStore, ObjectStore};
use std::env;

/// Helper to check if GCS integration tests should run
fn should_run_gcs_tests() -> bool {
    env::var("GCS_TEST_BUCKET").is_ok()
}

/// Helper to get test URI
fn get_test_uri(key: &str) -> String {
    let bucket = env::var("GCS_TEST_BUCKET").expect("GCS_TEST_BUCKET not set");
    format!("gs://{}/{}", bucket, key)
}

// ============================================================================
// SECTION 1: Basic Operations (put, get, delete)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_gcs_put_get_delete_small() -> Result<()> {
    if !should_run_gcs_tests() {
        println!("⚠️  Skipping - set GCS_TEST_BUCKET environment variable");
        return Ok(());
    }

    println!("\n=== TEST: put() → get() → delete() (1MB, below RangeEngine threshold) ===");
    
    let test_uri = get_test_uri("rust-test/small-file.bin");
    let store = GcsObjectStore::new();
    
    // Create test data
    let test_data = Bytes::from(vec![42u8; 1024 * 1024]); // 1MB
    
    // Upload
    println!("📤 PUT: Uploading 1MB...");
    store.put(&test_uri, &test_data).await?;
    
    // Download - should use simple download (below 4MB threshold)
    println!("📥 GET: Downloading 1MB (simple download)...");
    let downloaded: Bytes = store.get(&test_uri).await?;
    
    // Verify
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded, test_data, "Data mismatch");
    
    // Cleanup
    println!("🗑️  DELETE: Cleaning up...");
    store.delete(&test_uri).await?;
    
    println!("✅ Small file operations successful");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_gcs_put_get_delete_large_rangeengine() -> Result<()> {
    if !should_run_gcs_tests() {
        println!("⚠️  Skipping - set GCS_TEST_BUCKET environment variable");
        return Ok(());
    }

    println!("\n=== TEST: put() → get() → delete() (128MB, RangeEngine with 2 ranges) ===");
    
    let test_uri = get_test_uri("rust-test/large-rangeengine.bin");
    let store = GcsObjectStore::new();
    
    // Create large test data (128MB = 2 x 64MB chunks)
    let size = 128 * 1024 * 1024;
    println!("📦 Generating 128MB test data...");
    let test_data = Bytes::from(vec![77u8; size]);
    
    // Upload
    println!("📤 PUT: Uploading 128MB...");
    let start = std::time::Instant::now();
    store.put(&test_uri, &test_data).await?;
    let upload_time = start.elapsed();
    println!("   Upload: {:.2} MB/s", (size as f64 / 1024.0 / 1024.0) / upload_time.as_secs_f64());
    
    // Download - should use RangeEngine with 2 concurrent ranges
    println!("📥 GET: Downloading 128MB (RangeEngine with 2 concurrent ranges)...");
    let start = std::time::Instant::now();
    let downloaded: Bytes = store.get(&test_uri).await?;
    let download_time = start.elapsed();
    println!("   Download: {:.2} MB/s", (size as f64 / 1024.0 / 1024.0) / download_time.as_secs_f64());
    
    // Verify
    assert_eq!(downloaded.len(), test_data.len(), "Size mismatch");
    assert_eq!(downloaded[0], test_data[0], "Data mismatch at start");
    assert_eq!(downloaded[size-1], test_data[size-1], "Data mismatch at end");
    
    // Cleanup
    println!("🗑️  DELETE: Cleaning up...");
    store.delete(&test_uri).await?;
    
    println!("✅ Large file RangeEngine operations successful");
    Ok(())
}

// ============================================================================
// SECTION 2: get_range() - Partial reads
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_gcs_get_range() -> Result<()> {
    if !should_run_gcs_tests() {
        println!("⚠️  Skipping - set GCS_TEST_BUCKET environment variable");
        return Ok(());
    }

    println!("\n=== TEST: get_range() for partial reads ===");
    
    let test_uri = get_test_uri("rust-test/range-test.bin");
    let store = GcsObjectStore::new();
    
    // Create test data with pattern
    let mut test_data = vec![0u8; 10 * 1024]; // 10KB
    for (i, byte) in test_data.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }
    let test_data = Bytes::from(test_data);
    
    // Upload
    println!("📤 PUT: Uploading 10KB pattern data...");
    store.put(&test_uri, &test_data).await?;
    
    // Test range read: bytes 1024-2047 (1KB starting at offset 1024)
    println!("📥 GET_RANGE: Reading bytes 1024-2047...");
    let range_data: Bytes = store.get_range(&test_uri, 1024, Some(1024)).await?;
    
    // Verify
    assert_eq!(range_data.len(), 1024, "Range length mismatch");
    assert_eq!(range_data[0], test_data[1024], "Range data mismatch at start");
    assert_eq!(range_data[1023], test_data[2047], "Range data mismatch at end");
    
    // Cleanup
    println!("🗑️  DELETE: Cleaning up...");
    store.delete(&test_uri).await?;
    
    println!("✅ get_range() successful");
    Ok(())
}

// ============================================================================
// SECTION 3: stat() - Object metadata
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_gcs_stat() -> Result<()> {
    if !should_run_gcs_tests() {
        println!("⚠️  Skipping - set GCS_TEST_BUCKET environment variable");
        return Ok(());
    }

    println!("\n=== TEST: stat() for object metadata ===");
    
    let test_uri = get_test_uri("rust-test/stat-test.bin");
    let store = GcsObjectStore::new();
    
    // Create test data
    let size = 5 * 1024 * 1024; // 5MB
    let test_data = Bytes::from(vec![99u8; size]);
    
    // Upload
    println!("📤 PUT: Uploading 5MB...");
    store.put(&test_uri, &test_data).await?;
    
    // Stat
    println!("📊 STAT: Getting metadata...");
    let metadata = store.stat(&test_uri).await?;
    
    // Verify
    assert_eq!(metadata.size, size as u64, "Size mismatch in metadata");
    println!("   Size: {} bytes", metadata.size);
    if let Some(etag) = &metadata.e_tag {
        println!("   ETag: {}", etag);
    }
    
    // Cleanup
    println!("🗑️  DELETE: Cleaning up...");
    store.delete(&test_uri).await?;
    
    println!("✅ stat() successful");
    Ok(())
}

// ============================================================================
// SECTION 4: list() - Directory listing
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_gcs_list() -> Result<()> {
    if !should_run_gcs_tests() {
        println!("⚠️  Skipping - set GCS_TEST_BUCKET environment variable");
        return Ok(());
    }

    println!("\n=== TEST: list() for directory listing ===");
    
    let store = GcsObjectStore::new();
    let test_data = Bytes::from(vec![88u8; 1024]);
    
    // Create 3 test files
    let test_uris = vec![
        get_test_uri("rust-test/list-test/file-1.bin"),
        get_test_uri("rust-test/list-test/file-2.bin"),
        get_test_uri("rust-test/list-test/file-3.bin"),
    ];
    
    println!("📤 PUT: Creating 3 test files...");
    for uri in &test_uris {
        store.put(uri, &test_data).await?;
    }
    
    // List
    let bucket = env::var("GCS_TEST_BUCKET").expect("GCS_TEST_BUCKET not set");
    let list_uri = format!("gs://{}/rust-test/list-test/", bucket);
    println!("📋 LIST: Listing {}...", list_uri);
    let keys = store.list(&list_uri, false).await?;
    
    // Verify
    assert!(keys.len() >= 3, "Expected at least 3 files, got {}", keys.len());
    println!("   Found {} objects", keys.len());
    
    // Cleanup
    println!("🗑️  DELETE: Cleaning up 3 files...");
    for uri in &test_uris {
        store.delete(uri).await?;
    }
    
    println!("✅ list() successful");
    Ok(())
}
