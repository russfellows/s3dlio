// tests/metadata_ops_tests.rs
//
// Integration tests for metadata operations (mkdir, rmdir, update_metadata, update_properties)
// v0.10.0+ - Tests new ObjectStore trait methods

use anyhow::Result;
use s3dlio::object_store::{ObjectStore, ObjectProperties};
use s3dlio::file_store::FileSystemObjectStore;
use std::collections::HashMap;
use tempfile::TempDir;

/// Helper to create a test URI from tempdir
fn test_uri(temp_dir: &TempDir, path: &str) -> String {
    format!("file://{}/{}", temp_dir.path().display(), path)
}

/// Helper to check if a directory exists in tempdir
fn dir_exists(temp_dir: &TempDir, path: &str) -> bool {
    temp_dir.path().join(path).exists() && temp_dir.path().join(path).is_dir()
}

/// Helper to check if a file exists in tempdir
fn file_exists(temp_dir: &TempDir, path: &str) -> bool {
    temp_dir.path().join(path).exists() && temp_dir.path().join(path).is_file()
}

#[tokio::test]
async fn test_mkdir_creates_directory() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let test_uri = test_uri(&temp_dir, "test_dir");
    
    // Directory should not exist initially
    assert!(!dir_exists(&temp_dir, "test_dir"));
    
    // Create directory
    store.mkdir(&test_uri).await?;
    
    // Directory should now exist
    assert!(dir_exists(&temp_dir, "test_dir"));
    
    Ok(())
}

#[tokio::test]
async fn test_mkdir_creates_nested_directories() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let test_uri = test_uri(&temp_dir, "parent/child/grandchild");
    
    // Nested path should not exist
    assert!(!dir_exists(&temp_dir, "parent"));
    
    // Create nested directories (should use create_dir_all semantics)
    store.mkdir(&test_uri).await?;
    
    // All levels should exist
    assert!(dir_exists(&temp_dir, "parent"));
    assert!(dir_exists(&temp_dir, "parent/child"));
    assert!(dir_exists(&temp_dir, "parent/child/grandchild"));
    
    Ok(())
}

#[tokio::test]
async fn test_mkdir_idempotent() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let test_uri = test_uri(&temp_dir, "test_dir");
    
    // Create directory twice (should not fail)
    store.mkdir(&test_uri).await?;
    store.mkdir(&test_uri).await?;
    
    assert!(dir_exists(&temp_dir, "test_dir"));
    
    Ok(())
}

#[tokio::test]
async fn test_rmdir_removes_empty_directory() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let test_uri = test_uri(&temp_dir, "test_dir");
    
    // Create and verify directory
    store.mkdir(&test_uri).await?;
    assert!(dir_exists(&temp_dir, "test_dir"));
    
    // Remove directory (non-recursive)
    store.rmdir(&test_uri, false).await?;
    
    // Directory should be gone
    assert!(!dir_exists(&temp_dir, "test_dir"));
    
    Ok(())
}

#[tokio::test]
async fn test_rmdir_fails_on_non_empty_directory_without_recursive() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let dir_uri = test_uri(&temp_dir, "test_dir");
    let file_uri = test_uri(&temp_dir, "test_dir/file.txt");
    
    // Create directory with a file
    store.mkdir(&dir_uri).await?;
    store.put(&file_uri, b"test content").await?;
    
    assert!(dir_exists(&temp_dir, "test_dir"));
    assert!(file_exists(&temp_dir, "test_dir/file.txt"));
    
    // Attempt non-recursive removal (should fail)
    let result = store.rmdir(&dir_uri, false).await;
    assert!(result.is_err(), "rmdir should fail on non-empty directory without recursive=true");
    
    // Directory should still exist
    assert!(dir_exists(&temp_dir, "test_dir"));
    assert!(file_exists(&temp_dir, "test_dir/file.txt"));
    
    Ok(())
}

#[tokio::test]
async fn test_rmdir_recursive_removes_non_empty_directory() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let dir_uri = test_uri(&temp_dir, "test_dir");
    let file1_uri = test_uri(&temp_dir, "test_dir/file1.txt");
    let file2_uri = test_uri(&temp_dir, "test_dir/subdir/file2.txt");
    
    // Create directory structure with files
    store.mkdir(&test_uri(&temp_dir, "test_dir/subdir")).await?;
    store.put(&file1_uri, b"content 1").await?;
    store.put(&file2_uri, b"content 2").await?;
    
    assert!(dir_exists(&temp_dir, "test_dir"));
    assert!(file_exists(&temp_dir, "test_dir/file1.txt"));
    assert!(file_exists(&temp_dir, "test_dir/subdir/file2.txt"));
    
    // Recursive removal
    store.rmdir(&dir_uri, true).await?;
    
    // Everything should be gone
    assert!(!dir_exists(&temp_dir, "test_dir"));
    assert!(!file_exists(&temp_dir, "test_dir/file1.txt"));
    assert!(!file_exists(&temp_dir, "test_dir/subdir/file2.txt"));
    
    Ok(())
}

#[tokio::test]
async fn test_rmdir_on_nonexistent_directory_fails() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let test_uri = test_uri(&temp_dir, "nonexistent");
    
    // Should fail on non-existent directory
    let result = store.rmdir(&test_uri, false).await;
    assert!(result.is_err(), "rmdir should fail on non-existent directory");
    
    Ok(())
}

#[tokio::test]
async fn test_update_metadata_not_supported_for_file_backend() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let file_uri = test_uri(&temp_dir, "test.txt");
    
    // Create a file
    store.put(&file_uri, b"test content").await?;
    
    // Attempt to update metadata
    let mut metadata = HashMap::new();
    metadata.insert("x-custom-key".to_string(), "custom-value".to_string());
    
    let result = store.update_metadata(&file_uri, &metadata).await;
    
    // Should fail with clear error message
    assert!(result.is_err(), "update_metadata should not be supported for file:// backend");
    assert!(result.unwrap_err().to_string().contains("not supported"));
    
    Ok(())
}

#[tokio::test]
async fn test_update_properties_http_properties_not_supported() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let file_uri = test_uri(&temp_dir, "test.txt");
    
    // Create a file
    store.put(&file_uri, b"test content").await?;
    
    // Attempt to set HTTP properties (not supported for file backend)
    let properties = ObjectProperties {
        content_type: Some("application/json".to_string()),
        cache_control: None,
        content_encoding: None,
        content_language: None,
        content_disposition: None,
        expires: None,
        storage_class: None,
    };
    
    let result = store.update_properties(&file_uri, &properties).await;
    
    // Should fail
    assert!(result.is_err(), "HTTP properties should not be supported for file:// backend");
    assert!(result.unwrap_err().to_string().contains("not supported"));
    
    Ok(())
}

#[tokio::test]
async fn test_update_properties_storage_class_ignored() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let file_uri = test_uri(&temp_dir, "test.txt");
    
    // Create a file
    store.put(&file_uri, b"test content").await?;
    
    // Set only storage_class (should be accepted but ignored)
    let properties = ObjectProperties {
        content_type: None,
        cache_control: None,
        content_encoding: None,
        content_language: None,
        content_disposition: None,
        expires: None,
        storage_class: Some("COLD".to_string()),
    };
    
    let result = store.update_properties(&file_uri, &properties).await;
    
    // Should succeed (no-op)
    assert!(result.is_ok(), "storage_class should be accepted (as no-op) for file:// backend");
    
    Ok(())
}

#[tokio::test]
async fn test_update_properties_empty_properties_accepted() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    let file_uri = test_uri(&temp_dir, "test.txt");
    
    // Create a file
    store.put(&file_uri, b"test content").await?;
    
    // Empty properties (all None)
    let properties = ObjectProperties::default();
    
    let result = store.update_properties(&file_uri, &properties).await;
    
    // Should succeed (no-op)
    assert!(result.is_ok(), "Empty properties should be accepted");
    
    Ok(())
}

#[tokio::test]
async fn test_mkdir_rmdir_workflow() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    // Create multiple directories
    for i in 0..5 {
        let dir_uri = test_uri(&temp_dir, &format!("dir_{}", i));
        store.mkdir(&dir_uri).await?;
        assert!(dir_exists(&temp_dir, &format!("dir_{}", i)));
    }
    
    // Remove them
    for i in 0..5 {
        let dir_uri = test_uri(&temp_dir, &format!("dir_{}", i));
        store.rmdir(&dir_uri, false).await?;
        assert!(!dir_exists(&temp_dir, &format!("dir_{}", i)));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_mkdir_with_invalid_uri_scheme() -> Result<()> {
    let store = FileSystemObjectStore::new();
    
    // Wrong URI scheme
    let result = store.mkdir("s3://bucket/path").await;
    
    assert!(result.is_err(), "mkdir should reject non-file:// URIs");
    assert!(result.unwrap_err().to_string().contains("file://"));
    
    Ok(())
}

#[tokio::test]
async fn test_rmdir_with_invalid_uri_scheme() -> Result<()> {
    let store = FileSystemObjectStore::new();
    
    // Wrong URI scheme
    let result = store.rmdir("s3://bucket/path", false).await;
    
    assert!(result.is_err(), "rmdir should reject non-file:// URIs");
    assert!(result.unwrap_err().to_string().contains("file://"));
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_mkdir_operations() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    // Create 10 directories concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let store_clone = store.clone();
        let dir_uri = test_uri(&temp_dir, &format!("concurrent_dir_{}", i));
        
        let handle = tokio::spawn(async move {
            store_clone.mkdir(&dir_uri).await
        });
        
        handles.push(handle);
    }
    
    // Wait for all to complete
    for handle in handles {
        handle.await??;
    }
    
    // Verify all directories exist
    for i in 0..10 {
        assert!(dir_exists(&temp_dir, &format!("concurrent_dir_{}", i)));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_mkdir_rmdir_nested_structure() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let store = FileSystemObjectStore::new();
    
    // Create nested structure
    let deep_uri = test_uri(&temp_dir, "level1/level2/level3/level4/level5");
    store.mkdir(&deep_uri).await?;
    
    // Verify all levels exist
    assert!(dir_exists(&temp_dir, "level1"));
    assert!(dir_exists(&temp_dir, "level1/level2"));
    assert!(dir_exists(&temp_dir, "level1/level2/level3"));
    assert!(dir_exists(&temp_dir, "level1/level2/level3/level4"));
    assert!(dir_exists(&temp_dir, "level1/level2/level3/level4/level5"));
    
    // Remove recursively from root
    let root_uri = test_uri(&temp_dir, "level1");
    store.rmdir(&root_uri, true).await?;
    
    // Everything should be gone
    assert!(!dir_exists(&temp_dir, "level1"));
    
    Ok(())
}
