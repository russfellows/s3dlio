// tests/test_azure_comprehensive.rs
//
// Comprehensive integration tests for Azure Backend
// Tests ALL ObjectStore trait methods with ACTUAL Azure Blob Storage
// Validates zero-copy Bytes API changes from v0.9.0
//
// Prerequisites:
// 1. Azure credentials configured (az login)
// 2. Environment variables:
//    - AZURE_BLOB_ACCOUNT: Your storage account name
//    - AZURE_BLOB_CONTAINER: Container to use for testing
//
// Run: cargo test --release --test test_azure_comprehensive -- --nocapture --test-threads=1

use anyhow::Result;
use bytes::Bytes;
use s3dlio::api::store_for_uri;
use s3dlio::object_store::{AzureObjectStore, ObjectStore};
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

// ============================================================================
// SECTION 1: Zero-Copy get() - Returns Bytes (v0.9.0 change)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_get_returns_bytes_small() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: get() returns Bytes (small blob) ===");
    
    let test_uri = get_test_uri("test-get-bytes-small.bin");
    let store = AzureObjectStore::new();
    
    // Create small test data
    let test_data = Bytes::from(vec![11u8; 1024]); // 1KB
    
    // Upload
    println!("📤 Uploading 1KB test file...");
    store.put(&test_uri, &test_data).await?;
    
    // Download - should return Bytes (zero-copy)
    println!("📥 Downloading with get()...");
    let downloaded: Bytes = store.get(&test_uri).await?;
    
    // Verify type and data
    assert_eq!(downloaded.len(), test_data.len());
    assert_eq!(downloaded, test_data);
    
    // Cleanup
    store.delete(&test_uri).await?;
    
    println!("✅ get() returns Bytes correctly for small blob");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_get_returns_bytes_large() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: get() returns Bytes (large blob) ===");
    
    let test_uri = get_test_uri("test-get-bytes-large.bin");
    let store = AzureObjectStore::new();
    
    // Create large test data
    let size = 10 * 1024 * 1024; // 10MB
    let test_data = Bytes::from(vec![22u8; size]);
    
    // Upload
    println!("📤 Uploading 10MB test file...");
    let start = std::time::Instant::now();
    store.put(&test_uri, &test_data).await?;
    let upload_time = start.elapsed();
    println!("   Upload time: {:?}", upload_time);
    
    // Download - should return Bytes (zero-copy)
    println!("📥 Downloading with get()...");
    let start = std::time::Instant::now();
    let downloaded: Bytes = store.get(&test_uri).await?;
    let download_time = start.elapsed();
    
    let throughput = (size as f64 / 1024.0 / 1024.0) / download_time.as_secs_f64();
    println!("   Download time: {:?}", download_time);
    println!("   Throughput: {:.2} MB/s", throughput);
    
    // Verify type and data
    assert_eq!(downloaded.len(), test_data.len());
    
    // Verify first and last 1KB to avoid full comparison
    assert_eq!(&downloaded[0..1024], &test_data[0..1024]);
    assert_eq!(&downloaded[size-1024..size], &test_data[size-1024..size]);
    
    // Cleanup
    store.delete(&test_uri).await?;
    
    println!("✅ get() returns Bytes correctly for large blob");
    Ok(())
}

// ============================================================================
// SECTION 2: Zero-Copy get_range() - Returns Bytes (v0.9.0 change)
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_get_range_returns_bytes() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: get_range() returns Bytes ===");
    
    let test_uri = get_test_uri("test-get-range-bytes.bin");
    let store = AzureObjectStore::new();
    
    // Create test data with distinct patterns
    let mut test_data = Vec::new();
    test_data.extend(vec![10u8; 1024]); // First 1KB
    test_data.extend(vec![20u8; 1024]); // Second 1KB
    test_data.extend(vec![30u8; 1024]); // Third 1KB
    let test_data = Bytes::from(test_data);
    
    // Upload
    println!("📤 Uploading 3KB test file with distinct patterns...");
    store.put(&test_uri, &test_data).await?;
    
    // Test get_range - first 1KB
    println!("📥 Testing get_range(0, 1024)...");
    let range1: Bytes = store.get_range(&test_uri, 0, Some(1024)).await?;
    assert_eq!(range1.len(), 1024);
    assert_eq!(range1, test_data.slice(0..1024));
    assert!(range1.iter().all(|&b| b == 10u8));
    
    // Test get_range - middle 1KB
    println!("📥 Testing get_range(1024, 1024)...");
    let range2: Bytes = store.get_range(&test_uri, 1024, Some(1024)).await?;
    assert_eq!(range2.len(), 1024);
    assert_eq!(range2, test_data.slice(1024..2048));
    assert!(range2.iter().all(|&b| b == 20u8));
    
    // Test get_range - last 1KB
    println!("📥 Testing get_range(2048, 1024)...");
    let range3: Bytes = store.get_range(&test_uri, 2048, Some(1024)).await?;
    assert_eq!(range3.len(), 1024);
    assert_eq!(range3, test_data.slice(2048..3072));
    assert!(range3.iter().all(|&b| b == 30u8));
    
    // Test get_range - rest of file (None length)
    println!("📥 Testing get_range(1024, None)...");
    let range4: Bytes = store.get_range(&test_uri, 1024, None).await?;
    assert_eq!(range4.len(), 2048); // Remaining 2KB
    assert_eq!(range4, test_data.slice(1024..3072));
    
    // Cleanup
    store.delete(&test_uri).await?;
    
    println!("✅ get_range() returns Bytes correctly with various ranges");
    Ok(())
}

// ============================================================================
// SECTION 3: put() with various sizes
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_put_various_sizes() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: put() with various blob sizes ===");
    
    let store = AzureObjectStore::new();
    
    // Test sizes: 0, 1B, 1KB, 1MB, 10MB
    let test_cases = vec![
        (0, "test-put-0bytes.bin"),
        (1, "test-put-1byte.bin"),
        (1024, "test-put-1kb.bin"),
        (1024 * 1024, "test-put-1mb.bin"),
        (10 * 1024 * 1024, "test-put-10mb.bin"),
    ];
    
    for (size, key) in test_cases {
        println!("  Testing put() with {} bytes...", size);
        let test_uri = get_test_uri(key);
        let test_data = Bytes::from(vec![77u8; size]);
        
        // Upload
        store.put(&test_uri, &test_data).await?;
        
        // Verify by downloading
        let downloaded = store.get(&test_uri).await?;
        assert_eq!(downloaded.len(), size);
        if size > 0 {
            assert_eq!(downloaded, test_data);
        }
        
        // Cleanup
        store.delete(&test_uri).await?;
    }
    
    println!("✅ put() works correctly for all tested sizes");
    Ok(())
}

// ============================================================================
// SECTION 4: put_multipart() for large blobs
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_put_multipart() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: put_multipart() for large blobs ===");
    
    let test_uri = get_test_uri("test-put-multipart.bin");
    let store = AzureObjectStore::new();
    
    // Create 20MB test data
    let size = 20 * 1024 * 1024;
    let test_data = Bytes::from(vec![88u8; size]);
    
    // Upload with multipart (16MB part size)
    println!("📤 Uploading 20MB with put_multipart (16MB parts)...");
    let start = std::time::Instant::now();
    store.put_multipart(&test_uri, &test_data, Some(16 * 1024 * 1024)).await?;
    let upload_time = start.elapsed();
    
    let throughput = (size as f64 / 1024.0 / 1024.0) / upload_time.as_secs_f64();
    println!("   Upload time: {:?}", upload_time);
    println!("   Throughput: {:.2} MB/s", throughput);
    
    // Download and verify
    println!("📥 Downloading to verify...");
    let downloaded = store.get(&test_uri).await?;
    assert_eq!(downloaded.len(), size);
    
    // Verify first and last 1KB
    assert_eq!(&downloaded[0..1024], &test_data[0..1024]);
    assert_eq!(&downloaded[size-1024..size], &test_data[size-1024..size]);
    
    // Cleanup
    store.delete(&test_uri).await?;
    
    println!("✅ put_multipart() works correctly");
    Ok(())
}

// ============================================================================
// SECTION 5: list() operations
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_list_operations() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: list() operations ===");
    
    let store = AzureObjectStore::new();
    let test_prefix = "test-list-ops";
    
    // Create test blobs
    let test_blobs = vec![
        format!("{}/file1.bin", test_prefix),
        format!("{}/file2.bin", test_prefix),
        format!("{}/subdir/file3.bin", test_prefix),
    ];
    
    println!("📤 Creating test blobs...");
    for key in &test_blobs {
        let uri = get_test_uri(key);
        let data = Bytes::from(vec![99u8; 100]);
        store.put(&uri, &data).await?;
    }
    
    // Test recursive list
    println!("📋 Testing recursive list...");
    let list_uri = get_test_uri(test_prefix);
    let listed = store.list(&list_uri, true).await?;
    assert_eq!(listed.len(), 3, "Should list all 3 blobs recursively");
    
    // Test non-recursive list (shallow)
    println!("📋 Testing non-recursive list...");
    let listed_shallow = store.list(&list_uri, false).await?;
    // Note: Azure behavior may vary, just verify it returns something
    assert!(!listed_shallow.is_empty(), "Should list at least some blobs");
    
    // Cleanup
    println!("🗑️  Cleaning up test blobs...");
    for key in &test_blobs {
        let uri = get_test_uri(key);
        store.delete(&uri).await?;
    }
    
    println!("✅ list() operations work correctly");
    Ok(())
}

// ============================================================================
// SECTION 6: stat() operations
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_stat_operations() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: stat() operations ===");
    
    let test_uri = get_test_uri("test-stat.bin");
    let store = AzureObjectStore::new();
    
    // Create test blob
    let size = 5 * 1024 * 1024; // 5MB
    let test_data = Bytes::from(vec![55u8; size]);
    
    println!("📤 Uploading 5MB test file...");
    store.put(&test_uri, &test_data).await?;
    
    // Stat the blob
    println!("📊 Getting blob metadata with stat()...");
    let metadata = store.stat(&test_uri).await?;
    
    println!("   Size: {} bytes", metadata.size);
    println!("   Last modified: {:?}", metadata.last_modified);
    println!("   E-Tag: {:?}", metadata.e_tag);
    
    // Verify size
    assert_eq!(metadata.size as usize, size, "stat() should return correct size");
    
    // Cleanup
    store.delete(&test_uri).await?;
    
    println!("✅ stat() returns correct metadata");
    Ok(())
}

// ============================================================================
// SECTION 7: delete() and delete_prefix() operations
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_delete_operations() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: delete() and delete_prefix() ===");
    
    let store = AzureObjectStore::new();
    
    // Test single delete
    println!("🗑️  Testing single delete()...");
    let test_uri = get_test_uri("test-delete-single.bin");
    let test_data = Bytes::from(vec![66u8; 1024]);
    store.put(&test_uri, &test_data).await?;
    
    // Verify exists
    let metadata = store.stat(&test_uri).await?;
    assert_eq!(metadata.size, 1024);
    
    // Delete
    store.delete(&test_uri).await?;
    
    // Verify deleted (stat should fail)
    let stat_result = store.stat(&test_uri).await;
    assert!(stat_result.is_err(), "stat() should fail after delete");
    
    // Test prefix delete
    println!("🗑️  Testing delete_prefix()...");
    let prefix = "test-delete-prefix";
    let test_blobs = vec![
        format!("{}/file1.bin", prefix),
        format!("{}/file2.bin", prefix),
        format!("{}/file3.bin", prefix),
    ];
    
    // Create multiple blobs
    for key in &test_blobs {
        let uri = get_test_uri(key);
        store.put(&uri, &test_data).await?;
    }
    
    // Delete prefix
    let prefix_uri = get_test_uri(prefix);
    store.delete_prefix(&prefix_uri).await?;
    
    // Verify all deleted
    for key in &test_blobs {
        let uri = get_test_uri(key);
        let stat_result = store.stat(&uri).await;
        assert!(stat_result.is_err(), "All blobs under prefix should be deleted");
    }
    
    println!("✅ delete() and delete_prefix() work correctly");
    Ok(())
}

// ============================================================================
// SECTION 8: Edge cases
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_edge_cases() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: Edge cases ===");
    
    let store = AzureObjectStore::new();
    
    // Empty blob
    println!("  Testing empty blob (0 bytes)...");
    let test_uri = get_test_uri("test-edge-empty.bin");
    let empty_data = Bytes::new();
    store.put(&test_uri, &empty_data).await?;
    let downloaded = store.get(&test_uri).await?;
    assert_eq!(downloaded.len(), 0);
    store.delete(&test_uri).await?;
    
    // Non-existent blob
    println!("  Testing get() on non-existent blob...");
    let nonexistent_uri = get_test_uri("nonexistent-blob-12345.bin");
    let result = store.get(&nonexistent_uri).await;
    assert!(result.is_err(), "get() should fail for non-existent blob");
    
    // Non-existent range
    println!("  Testing get_range() on non-existent blob...");
    let result = store.get_range(&nonexistent_uri, 0, Some(100)).await;
    assert!(result.is_err(), "get_range() should fail for non-existent blob");
    
    // Invalid range (offset beyond size)
    println!("  Testing get_range() with offset beyond size...");
    let test_uri = get_test_uri("test-edge-invalid-range.bin");
    let test_data = Bytes::from(vec![44u8; 1024]);
    store.put(&test_uri, &test_data).await?;
    
    // Request range beyond file size - should return empty or error
    let result = store.get_range(&test_uri, 2048, Some(1024)).await;
    // Azure may return empty Bytes or error depending on implementation
    if let Ok(data) = result {
        assert_eq!(data.len(), 0, "Should return empty for range beyond size");
    }
    
    store.delete(&test_uri).await?;
    
    println!("✅ Edge cases handled correctly");
    Ok(())
}

// ============================================================================
// SECTION 9: Concurrent operations
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_azure_concurrent_operations() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: Concurrent operations ===");
    
    let store = AzureObjectStore::new();
    let num_blobs = 10;
    
    // Concurrent uploads
    println!("📤 Testing {} concurrent uploads...", num_blobs);
    let upload_futures: Vec<_> = (0..num_blobs)
        .map(|i| {
            let store_clone = store.clone();
            let uri = get_test_uri(&format!("test-concurrent-{}.bin", i));
            let data = Bytes::from(vec![i as u8; 1024 * 100]); // 100KB each
            async move {
                store_clone.put(&uri, &data).await?;
                Ok::<_, anyhow::Error>((uri, data))
            }
        })
        .collect();
    
    let start = std::time::Instant::now();
    let results = futures::future::try_join_all(upload_futures).await?;
    let upload_time = start.elapsed();
    println!("   Upload time: {:?} ({} blobs)", upload_time, num_blobs);
    
    // Concurrent downloads
    println!("📥 Testing {} concurrent downloads...", num_blobs);
    let download_futures: Vec<_> = results
        .iter()
        .map(|(uri, original_data)| {
            let store_clone = store.clone();
            let uri_clone = uri.clone();
            let original_clone = original_data.clone();
            async move {
                let downloaded = store_clone.get(&uri_clone).await?;
                assert_eq!(downloaded, original_clone);
                Ok::<_, anyhow::Error>(uri_clone)
            }
        })
        .collect();
    
    let start = std::time::Instant::now();
    let download_uris = futures::future::try_join_all(download_futures).await?;
    let download_time = start.elapsed();
    println!("   Download time: {:?} ({} blobs)", download_time, num_blobs);
    
    // Cleanup
    println!("🗑️  Cleaning up...");
    for uri in download_uris {
        store.delete(&uri).await?;
    }
    
    println!("✅ Concurrent operations completed successfully");
    Ok(())
}

// ============================================================================
// SECTION 10: Factory function test
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_azure_via_factory() -> Result<()> {
    if !should_run_azure_tests() {
        println!("⚠️  Skipping - set AZURE_BLOB_ACCOUNT and AZURE_BLOB_CONTAINER");
        return Ok(());
    }

    println!("\n=== TEST: store_for_uri() factory ===");
    
    let test_uri = get_test_uri("test-factory.bin");
    
    // Create store via factory
    let store = store_for_uri(&test_uri)?;
    
    // Test upload/download
    let test_data = Bytes::from(vec![33u8; 2 * 1024 * 1024]); // 2MB
    
    println!("📤 Uploading via factory-created store...");
    store.put(&test_uri, &test_data).await?;
    
    println!("📥 Downloading via factory-created store...");
    let downloaded = store.get(&test_uri).await?;
    
    assert_eq!(downloaded, test_data);
    
    // Cleanup
    store.delete(&test_uri).await?;
    
    println!("✅ store_for_uri() works correctly for Azure");
    Ok(())
}
