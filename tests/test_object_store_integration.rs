use s3dlio::object_store::store_for_uri;
use anyhow::Result;
use tokio;

#[tokio::test]
async fn test_object_store_factory() -> Result<()> {
    // Test file:// URI
    let _store = store_for_uri("file:///tmp/test")?;
    
    // Test s3:// URI  
    let _store = store_for_uri("s3://test-bucket/test")?;
    
    Ok(())
}

#[tokio::test]
async fn test_file_store_operations() -> Result<()> {
    let test_dir = tempfile::tempdir()?;
    let test_file = test_dir.path().join("test").join("data.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    let store = store_for_uri(&file_uri)?;
    
    // Test basic operations through unified interface
    let test_data = b"Hello, ObjectStore!";
    
    // Put operation
    store.put(&file_uri, test_data).await?;
    
    // Get operation
    let retrieved = store.get(&file_uri).await?;
    assert_eq!(retrieved.as_ref(), test_data);
    
    // List operation - list the directory containing our file
    let dir_uri = format!("file://{}", test_dir.path().join("test").display());
    let objects = store.list(&dir_uri, true).await?;
    assert!(!objects.is_empty());
    
    // Stat operation
    let metadata = store.stat(&file_uri).await?;
    assert_eq!(metadata.size, test_data.len() as u64);
    
    // Range operation
    let partial = store.get_range(&file_uri, 0, Some(5)).await?;
    assert_eq!(partial.as_ref(), b"Hello");
    
    // Delete operation
    store.delete(&file_uri).await?;
    
    // Verify deletion
    assert!(store.get(&file_uri).await.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_multipart_operations() -> Result<()> {
    let test_dir = tempfile::tempdir()?;
    let test_file = test_dir.path().join("test").join("multipart.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    let store = store_for_uri(&file_uri)?;
    
    // Test multipart upload through unified interface
    let part1 = b"Part 1 data - ";
    let part2 = b"Part 2 data - ";
    let part3 = b"Part 3 data";
    
    // Combine all parts into one buffer for multipart upload
    let combined_data: Vec<u8> = [part1.as_slice(), part2.as_slice(), part3.as_slice()].concat();
    
    // Use multipart upload with 5MB part size
    store.put_multipart(&file_uri, &combined_data, Some(5 * 1024 * 1024)).await?;
    
    // Verify multipart upload result
    let retrieved = store.get(&file_uri).await?;
    assert_eq!(retrieved, combined_data);
    
    Ok(())
}


#[tokio::test]
async fn test_azure_store_factory() -> Result<()> {
    // Test az:// URI (just factory, no actual operations without credentials)
    let _store = store_for_uri("az://test-container/test")?;
    Ok(())
}

#[tokio::test]
async fn test_cross_backend_consistency() -> Result<()> {
    let test_dir = tempfile::tempdir()?;
    let test_file = test_dir.path().join("consistency_test.txt");
    let file_uri = format!("file://{}", test_file.display());
    
    // Test that all backends expose the same interface
    let file_store = store_for_uri(&file_uri)?;
    let _s3_store = store_for_uri("s3://test-bucket/test")?;
    
    // Both should support the same operations (interface consistency)
    let test_data = b"test data";
    
    // File store operations
    file_store.put(&file_uri, test_data).await?;
    let file_result = file_store.get(&file_uri).await?;
    assert_eq!(file_result.as_ref(), test_data);
    
    // The interface is consistent even if S3 operations would fail
    // without proper credentials - the type system ensures compatibility
    Ok(())
}
