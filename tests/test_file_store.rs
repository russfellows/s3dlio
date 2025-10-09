// tests/test_file_store.rs
//
// Integration tests for FileSystemObjectStore
// Tests the file:// URI scheme and POSIX filesystem operations
//
// This test suite demonstrates the new consistent URI scheme approach:
// - s3://bucket/path/file.txt    (S3)
// - az://container/path/file.txt (Azure - coming in Phase 3.3)
// - file:///absolute/path/file.txt (Local filesystem)
//
// Same ObjectStore API works for all backends:
//   let store = store_for_uri(uri)?;
//   let data = store.get(uri).await?;
//   store.put(uri, &data).await?;

use anyhow::Result;
use s3dlio::{store_for_uri, infer_scheme, Scheme};
use tempfile::TempDir;

#[tokio::test]
async fn test_file_scheme_inference() {
    // Test that scheme inference works correctly for file URIs
    assert_eq!(infer_scheme("file:///tmp/test.txt"), Scheme::File);
    assert_eq!(infer_scheme("file://./relative/test.txt"), Scheme::File);
    assert_eq!(infer_scheme("file://../relative/test.txt"), Scheme::File);
    
    // Test that other schemes are correctly identified
    assert_eq!(infer_scheme("s3://bucket/test.txt"), Scheme::S3);
    assert_eq!(infer_scheme("az://container/test.txt"), Scheme::Azure);
    assert_eq!(infer_scheme("https://container.blob.core.windows.net/test.txt"), Scheme::Azure);
    
    // Test that unknown schemes are handled
    assert_eq!(infer_scheme("/direct/path/test.txt"), Scheme::Unknown);
    assert_eq!(infer_scheme("https://example.com/test.txt"), Scheme::Unknown);
    
    println!("✅ URI scheme consistency verified - supports s3://, az://, file://");
}

#[tokio::test]
async fn test_file_store_factory() -> Result<()> {
    // Test that store_for_uri correctly creates FileSystemObjectStore
    let store = store_for_uri("file:///tmp/test.txt")?;
    
    // The store should be created successfully
    // We can't easily test the exact type, but we can test that it works
    assert!(store.exists("file:///this-file-does-not-exist-12345").await.is_ok());
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_basic_operations() -> Result<()> {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("test.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    let store = store_for_uri(&file_uri)?;
    
    // Test data
    let test_data = b"Hello, FileSystemObjectStore!";
    let updated_data = b"Updated content for file store test";
    
    // Test that file doesn't exist initially
    assert!(!store.exists(&file_uri).await?);
    
    // Test put operation
    store.put(&file_uri, test_data).await?;
    
    // Test that file now exists
    assert!(store.exists(&file_uri).await?);
    
    // Test get operation
    let retrieved_data = store.get(&file_uri).await?;
    assert_eq!(retrieved_data.as_ref(), test_data);
    
    // Test stat operation
    let metadata = store.stat(&file_uri).await?;
    assert_eq!(metadata.size, test_data.len() as u64);
    assert!(metadata.e_tag.is_some());
    assert_eq!(metadata.storage_class, Some("STANDARD".to_string()));
    
    // Test put with updated data
    store.put(&file_uri, updated_data).await?;
    let retrieved_updated = store.get(&file_uri).await?;
    assert_eq!(retrieved_updated.as_ref(), updated_data);
    
    // Test delete operation
    store.delete(&file_uri).await?;
    assert!(!store.exists(&file_uri).await?);
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_range_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("range_test.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    let store = store_for_uri(&file_uri)?;
    
    // Create test data with known pattern
    let test_data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    store.put(&file_uri, test_data).await?;
    
    // Test range read from beginning
    let range1 = store.get_range(&file_uri, 0, Some(10)).await?;
    assert_eq!(range1.as_ref(), b"0123456789");
    
    // Test range read from middle
    let range2 = store.get_range(&file_uri, 10, Some(6)).await?;
    assert_eq!(range2.as_ref(), b"ABCDEF");
    
    // Test range read to end (no length specified)
    let range3 = store.get_range(&file_uri, 30, None).await?;
    assert_eq!(range3.as_ref(), b"UVWXYZ");
    
    // Test range read beyond file size
    let range4 = store.get_range(&file_uri, 35, Some(10)).await?;
    assert_eq!(range4.as_ref(), b"Z"); // Should only return what's available
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_directory_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    let store = store_for_uri("file://dummy")?; // We'll override the URIs
    
    // Create test directory structure
    let dir1 = base_path.join("dir1");
    let file1 = base_path.join("dir1/file1.txt");
    let file2 = base_path.join("dir1/subdir/file2.txt");
    
    let dir1_uri = format!("file://{}", dir1.display());
    let file1_uri = format!("file://{}", file1.display());
    let file2_uri = format!("file://{}", file2.display());
    
    // Test create_container (directory creation)
    store.create_container(&dir1.to_string_lossy()).await?;
    assert!(dir1.exists());
    
    // Test putting files (should create subdirectories automatically)
    store.put(&file1_uri, b"File 1 content").await?;
    store.put(&file2_uri, b"File 2 content").await?;
    
    // Test listing files non-recursively
    let files_non_recursive = store.list(&dir1_uri, false).await?;
    assert_eq!(files_non_recursive.len(), 1); // Only file1.txt
    assert!(files_non_recursive.contains(&file1_uri));
    
    // Test listing files recursively
    let files_recursive = store.list(&dir1_uri, true).await?;
    assert_eq!(files_recursive.len(), 2); // Both file1.txt and file2.txt
    assert!(files_recursive.contains(&file1_uri));
    assert!(files_recursive.contains(&file2_uri));
    
    // Test delete_prefix (should delete all files under prefix)
    store.delete_prefix(&dir1_uri).await?;
    let files_after_delete = store.list(&dir1_uri, true).await?;
    assert_eq!(files_after_delete.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_multipart_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("multipart_test.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    let store = store_for_uri(&file_uri)?;
    
    // Test multipart put (should work the same as regular put for filesystem)
    let large_data = vec![b'X'; 1024 * 1024]; // 1MB of X's
    store.put_multipart(&file_uri, &large_data, Some(64 * 1024)).await?;
    
    // Verify the data was written correctly
    let retrieved_data = store.get(&file_uri).await?;
    assert_eq!(retrieved_data.len(), large_data.len());
    assert_eq!(retrieved_data, large_data);
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_copy_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let src_file = temp_dir.path().join("source.txt");
    let dst_file = temp_dir.path().join("destination.txt");
    
    let src_uri = format!("file://{}", src_file.display());
    let dst_uri = format!("file://{}", dst_file.display());
    
    let store = store_for_uri(&src_uri)?;
    
    // Create source file
    let test_data = b"Data to be copied";
    store.put(&src_uri, test_data).await?;
    
    // Test copy operation (uses default trait implementation)
    store.copy(&src_uri, &dst_uri).await?;
    
    // Verify both files exist and have same content
    assert!(store.exists(&src_uri).await?);
    assert!(store.exists(&dst_uri).await?);
    
    let src_data = store.get(&src_uri).await?;
    let dst_data = store.get(&dst_uri).await?;
    assert_eq!(src_data, dst_data);
    assert_eq!(src_data.as_ref(), test_data);
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_error_handling() -> Result<()> {
    let store = store_for_uri("file:///tmp/nonexistent")?;
    
    // Test operations on non-existent files
    assert!(store.get("file:///this/path/does/not/exist").await.is_err());
    assert!(store.stat("file:///this/path/does/not/exist").await.is_err());
    
    // Test invalid URI schemes
    assert!(store.get("s3://bucket/file").await.is_err());
    assert!(store.get("http://example.com/file").await.is_err());
    assert!(store.get("/direct/path").await.is_err());
    
    Ok(())
}

#[tokio::test] 
async fn test_file_store_relative_paths() -> Result<()> {
    // Test relative path URIs
    let temp_dir = TempDir::new()?;
    let current_dir = std::env::current_dir()?;
    
    // Change to temp directory for this test
    std::env::set_current_dir(&temp_dir)?;
    
    let store = store_for_uri("file://./test.txt")?;
    
    let test_data = b"Relative path test";
    store.put("file://./test.txt", test_data).await?;
    
    let retrieved_data = store.get("file://./test.txt").await?;
    assert_eq!(retrieved_data.as_ref(), test_data);
    
    // Restore original directory
    std::env::set_current_dir(current_dir)?;
    
    Ok(())
}
