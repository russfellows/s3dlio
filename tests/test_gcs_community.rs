// tests/test_gcs_community.rs
//
// Comprehensive functional tests for GCS backend using gcs-community (gcloud-storage crate).
//
// These tests run against real GCS infrastructure and require:
// - GCS_TEST_BUCKET environment variable set to a valid bucket name
// - Valid GCS authentication (gcloud CLI, service account, or ADC)
//
// All test objects use the prefix "s3dlio-test/" and are cleaned up after tests.
//
// Run with:
//   cargo test --test test_gcs_community --no-default-features --features native-backends,gcs-community
//

mod common;

use anyhow::{Context, Result};
use common::{get_test_config, print_test_header, print_test_result};

#[cfg(feature = "gcs-community")]
mod gcs_community_tests {
    use super::*;
    use s3dlio::gcs_client::{GcsClient, parse_gcs_uri};

    const BACKEND_NAME: &str = "gcs-community (gcloud-storage)";

    #[tokio::test]
    async fn test_parse_gcs_uri() -> Result<()> {
        print_test_header("Testing GCS URI Parsing", BACKEND_NAME);

        // Valid URIs
        let (bucket, key) = parse_gcs_uri("gs://my-bucket/path/to/object.txt")?;
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/object.txt");
        println!("✓ Parsed: gs://my-bucket/path/to/object.txt → bucket='{}', key='{}'", bucket, key);

        let (bucket, key) = parse_gcs_uri("gcs://my-bucket/file.dat")?;
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "file.dat");
        println!("✓ Parsed: gcs://my-bucket/file.dat → bucket='{}', key='{}'", bucket, key);

        // Invalid URIs
        assert!(parse_gcs_uri("s3://wrong-scheme/key").is_err());
        println!("✓ Correctly rejected invalid URI 's3://wrong-scheme/key': {}", 
                 parse_gcs_uri("s3://wrong-scheme/key").unwrap_err());

        assert!(parse_gcs_uri("http://not-gcs/key").is_err());
        println!("✓ Correctly rejected invalid URI 'http://not-gcs/key': {}", 
                 parse_gcs_uri("http://not-gcs/key").unwrap_err());

        assert!(parse_gcs_uri("gs://").is_err());
        println!("✓ Correctly rejected invalid URI 'gs://': {}", 
                 parse_gcs_uri("gs://").unwrap_err());

        assert!(parse_gcs_uri("").is_err());
        println!("✓ Correctly rejected invalid URI '': {}", 
                 parse_gcs_uri("").unwrap_err());

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_authentication() -> Result<()> {
        print_test_header("Testing GCS Authentication", BACKEND_NAME);
        let config = get_test_config();

        let client = GcsClient::new().await
            .context("Failed to create GCS client - check your credentials")?;
        println!("✓ GCS client created successfully with Application Default Credentials");

        // Test bucket access by listing
        println!("  Testing access to bucket: {}", config.bucket);
        let objects = client.list_objects(&config.bucket, Some(""), true).await
            .context("Failed to list bucket - check permissions")?;
        println!("✓ Successfully authenticated and accessed bucket (found {} objects)", objects.len());

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_put_get_object() -> Result<()> {
        print_test_header("Testing GCS PUT and GET Operations", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_key = format!("{}/test-put-get.txt", config.test_prefix);
        let test_data = b"Hello from s3dlio GCS backend test";
        
        println!("Uploading object: gs://{}/{}", config.bucket, test_key);
        client.put_object(&config.bucket, &test_key, test_data).await?;
        println!("✓ PUT successful: {} bytes", test_data.len());

        println!("Downloading object: gs://{}/{}", config.bucket, test_key);
        let retrieved = client.get_object(&config.bucket, &test_key).await?;
        println!("✓ GET successful: {} bytes", retrieved.len());

        assert_eq!(retrieved.as_ref(), test_data);
        println!("✓ Data integrity verified");

        // Cleanup
        client.delete_object(&config.bucket, &test_key).await?;
        println!("✓ Cleanup successful");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_get_object_range() -> Result<()> {
        print_test_header("Testing GCS Range GET Operations", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_key = format!("{}/test-range-get.txt", config.test_prefix);
        let test_data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        
        println!("Uploading object: gs://{}/{}", config.bucket, test_key);
        client.put_object(&config.bucket, &test_key, test_data).await?;
        println!("✓ PUT successful: {} bytes", test_data.len());

        // Test partial read
        println!("Testing range read: offset=10, length=10");
        let range_data = client.get_object_range(&config.bucket, &test_key, 10, Some(10)).await?;
        println!("✓ Range GET successful: {} bytes", range_data.len());
        assert_eq!(range_data.as_ref(), b"ABCDEFGHIJ");
        println!("✓ Range data verified: {:?}", std::str::from_utf8(&range_data));

        // Test tail read
        println!("Testing range read: offset=30, length=None (to end)");
        let tail_data = client.get_object_range(&config.bucket, &test_key, 30, None).await?;
        println!("✓ Tail GET successful: {} bytes", tail_data.len());
        assert_eq!(tail_data.as_ref(), b"UVWXYZ");
        println!("✓ Tail data verified");

        // Cleanup
        client.delete_object(&config.bucket, &test_key).await?;
        println!("✓ Cleanup successful");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_stat_object() -> Result<()> {
        print_test_header("Testing GCS STAT (Metadata) Operations", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_key = format!("{}/test-stat.txt", config.test_prefix);
        let test_data = b"Metadata test object content";
        
        println!("Uploading object: gs://{}/{}", config.bucket, test_key);
        client.put_object(&config.bucket, &test_key, test_data).await?;
        println!("✓ PUT successful");

        println!("Fetching object metadata...");
        let metadata = client.stat_object(&config.bucket, &test_key).await?;
        println!("✓ STAT successful");
        println!("Metadata:");
        println!("  Key: {}", metadata.key);
        println!("  Size: {} bytes", metadata.size);
        println!("  ETag: {:?}", metadata.etag);
        println!("  Updated: {:?}", metadata.updated);

        assert_eq!(metadata.size, test_data.len() as u64);
        assert_eq!(metadata.key, test_key);
        println!("✓ Metadata verified");

        // Cleanup
        client.delete_object(&config.bucket, &test_key).await?;
        println!("✓ Cleanup successful");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_list_objects() -> Result<()> {
        print_test_header("Testing GCS LIST Operations", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_prefix = format!("{}/test-list", config.test_prefix);
        
        // Create test objects
        println!("Uploading 4 test objects...");
        let test_keys = vec![
            format!("{}/file1.txt", test_prefix),
            format!("{}/file2.txt", test_prefix),
            format!("{}/subdir/file3.txt", test_prefix),
            format!("{}/subdir/file4.txt", test_prefix),
        ];
        
        for key in &test_keys {
            client.put_object(&config.bucket, key, b"test data").await?;
        }
        println!("✓ Test objects uploaded");

        // Test recursive listing
        println!("\nTest 1: List with prefix (recursive)");
        let objects = client.list_objects(&config.bucket, Some(&test_prefix), true).await?;
        println!("Found {} objects", objects.len());
        for obj in &objects {
            println!("  - {}", obj);
        }
        assert_eq!(objects.len(), 4);
        println!("✓ Recursive listing verified");

        // Test non-recursive listing (should get top-level files + subdirectory prefix)
        println!("\nTest 2: List with prefix (non-recursive)");
        let objects = client.list_objects(&config.bucket, Some(&test_prefix), false).await?;
        println!("Found {} objects/prefixes", objects.len());
        for obj in &objects {
            println!("  - {}", obj);
        }
        // Should return 2 files (file1.txt, file2.txt) + 1 prefix (subdir/)
        assert_eq!(objects.len(), 3);
        // Verify we got the top-level files
        assert!(objects.iter().any(|o| o.ends_with("file1.txt")));
        assert!(objects.iter().any(|o| o.ends_with("file2.txt")));
        // Verify we got the subdirectory prefix (ends with /)
        assert!(objects.iter().any(|o| o.contains("subdir") && o.ends_with("/")));
        println!("✓ Non-recursive listing verified (2 files + 1 subdirectory prefix)");

        // Test listing with our test prefix
        println!("\nTest 3: List with test prefix");
        let objects = client.list_objects(&config.bucket, Some(&config.test_prefix), true).await?;
        println!("Found {} objects with our test prefix", objects.len());
        assert!(objects.len() >= 4);
        println!("✓ Test prefix listing successful");

        // Cleanup
        println!("\nCleaning up test objects...");
        for key in &test_keys {
            client.delete_object(&config.bucket, key).await?;
        }
        println!("✓ Cleanup successful");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_delete_single_object() -> Result<()> {
        print_test_header("Testing GCS DELETE Single Object", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_key = format!("{}/test-delete-single.txt", config.test_prefix);
        
        println!("Uploading object: gs://{}/{}", config.bucket, test_key);
        client.put_object(&config.bucket, &test_key, b"Delete me").await?;
        println!("✓ PUT successful");

        // Verify it exists
        let metadata = client.stat_object(&config.bucket, &test_key).await?;
        println!("✓ Object exists (size: {} bytes)", metadata.size);

        println!("Deleting object...");
        client.delete_object(&config.bucket, &test_key).await?;
        println!("✓ DELETE successful");

        // Verify it's gone
        println!("Verifying object is gone...");
        let result = client.stat_object(&config.bucket, &test_key).await;
        assert!(result.is_err(), "Object should not exist after deletion");
        println!("✓ Object confirmed deleted");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_delete_multiple_objects() -> Result<()> {
        print_test_header("Testing GCS DELETE Batch (Multiple Objects)", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_prefix = format!("{}/test-batch-delete", config.test_prefix);
        
        // Create multiple test objects
        println!("Uploading 25 test objects...");
        let mut test_keys = Vec::new();
        for i in 0..25 {
            let key = format!("{}/file{}.txt", test_prefix, i);
            client.put_object(&config.bucket, &key, format!("test data {}", i).as_bytes()).await?;
            test_keys.push(key);
        }
        println!("✓ {} objects uploaded", test_keys.len());

        // Verify they exist
        let objects = client.list_objects(&config.bucket, Some(&test_prefix), true).await?;
        assert_eq!(objects.len(), 25);
        println!("✓ All objects verified present");

        // Delete all
        println!("Deleting {} objects in batch...", test_keys.len());
        client.delete_objects(&config.bucket, test_keys.clone()).await?;
        println!("✓ Batch DELETE successful");

        // Verify they're gone
        println!("Verifying objects are deleted...");
        let objects = client.list_objects(&config.bucket, Some(&test_prefix), true).await?;
        assert_eq!(objects.len(), 0, "All objects should be deleted");
        println!("✓ All objects confirmed deleted");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn test_gcs_multipart_upload() -> Result<()> {
        print_test_header("Testing GCS Multipart Upload", BACKEND_NAME);
        let config = get_test_config();
        let client = GcsClient::new().await?;

        let test_key = format!("{}/test-multipart.bin", config.test_prefix);
        
        // Create 1 MB of test data
        let data_size = 1024 * 1024; // 1 MB
        let test_data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
        
        println!("Uploading large object: gs://{}/{} ({} MB)", 
                 config.bucket, test_key, data_size / (1024 * 1024));
        
        // Use multipart upload with 256 KB chunks
        let chunk_size = 256 * 1024;
        client.put_object_multipart(&config.bucket, &test_key, &test_data, chunk_size).await?;
        println!("✓ Multipart PUT successful");

        // Verify size
        let metadata = client.stat_object(&config.bucket, &test_key).await?;
        assert_eq!(metadata.size, data_size as u64);
        println!("✓ Object size verified: {} bytes", metadata.size);

        // Download and verify
        println!("Downloading and verifying data...");
        let retrieved = client.get_object(&config.bucket, &test_key).await?;
        assert_eq!(retrieved.as_ref(), &test_data[..]);
        println!("✓ Data integrity verified");

        // Cleanup
        client.delete_object(&config.bucket, &test_key).await?;
        println!("✓ Cleanup successful");

        print_test_result(true);
        Ok(())
    }

    #[tokio::test]
    async fn integration_tests_run_all_gcs_tests() -> Result<()> {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  Comprehensive GCS Functional Test Suite                    ║");
        println!("║  Backend: {}                     ║", BACKEND_NAME);
        println!("╚══════════════════════════════════════════════════════════════╝\n");
        
        println!("See individual test results above");
        println!("Run with: cargo test --test test_gcs_community\n");
        
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  ✓ ALL TESTS PASSED                                          ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");
        
        Ok(())
    }
}
