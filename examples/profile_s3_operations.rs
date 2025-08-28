// examples/profile_s3_operations.rs
//
// Example demonstrating how to profile s3dlio operations to identify performance bottlenecks.
// This example shows different profiling approaches for various workload patterns.

use anyhow::Result;
use s3dlio::profiling::*;
use s3dlio::profile_span;
use s3dlio::data_gen::generate_controlled_data;
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok(); // Load from default .env files
    
    // Try loading from local-env file specifically
    if let Err(_) = dotenvy::from_filename("local-env") {
        println!("⚠️  Could not load local-env file, using system environment");
    }
    
    // Initialize comprehensive profiling with environment-based configuration
    init_profiling()?;
    
    println!("🔬 s3dlio Performance Profiling Example");
    println!("=======================================");
    
    // Check if we have S3 configuration from .env
    let has_s3_config = env::var("AWS_ACCESS_KEY_ID").is_ok() && env::var("S3_BUCKET").is_ok();
    
    if has_s3_config {
        let bucket = env::var("S3_BUCKET").unwrap();
        let endpoint = env::var("AWS_ENDPOINT_URL").ok();
        println!("✅ S3 configuration loaded from .env - will run real S3 operations");
        println!("   Bucket: {}", bucket);
        if let Some(ref ep) = endpoint {
            println!("   Endpoint: {}", ep);
        }
        profile_real_s3_operations().await?;
    } else {
        println!("ℹ️  No S3 configuration in .env - running synthetic workload profiling");
        profile_synthetic_workload().await?;
    }
    
    // Always run local operations profiling
    profile_local_operations().await?;
    
    println!("✅ Profiling complete! Check logs and flamegraphs for insights.");
    Ok(())
}

/// Profile real S3 operations using configuration from .env file
async fn profile_real_s3_operations() -> Result<()> {
    use s3dlio::s3_utils::*;
    use s3dlio::data_gen::generate_controlled_data;
    
    println!("\n📊 Profiling Real S3 Operations");
    println!("==============================");
    
    // Get S3 configuration from environment
    let bucket = env::var("S3_BUCKET").expect("S3_BUCKET should be set in .env");
    let test_key_prefix = "s3dlio-profile-test";
    
    // Profile creating test data and uploading it
    {
        let _span = profile_span!("s3_upload_operations", bucket = %bucket);
        let start = Instant::now();
        
        // Create test objects of different sizes
        let test_objects = [
            (format!("{}/small-1MB.bin", test_key_prefix), 1024 * 1024),      // 1MB
            (format!("{}/medium-4MB.bin", test_key_prefix), 4 * 1024 * 1024), // 4MB
            (format!("{}/large-16MB.bin", test_key_prefix), 16 * 1024 * 1024), // 16MB
        ];
        
        for (key, size) in &test_objects {
            let _upload_span = profile_span!("upload_single", key = %key, size = %size);
            
            // Generate test data
            let data = generate_controlled_data(*size, 10, 1);
            let uri = format!("s3://{}/{}", bucket, key);
            
            // Upload the data
            match put_object_uri_async(&uri, &data).await {
                Ok(()) => println!("   ✅ Uploaded {} -> {} bytes", key, data.len()),
                Err(e) => println!("   ❌ Failed to upload {}: {}", key, e),
            }
        }
        
        println!("   ⏱️  Upload operations completed in {:?}", start.elapsed());
    }
    
    // Profile object stats (lightweight operation)
    {
        let _span = profile_span!("s3_stat_operations", bucket = %bucket);
        let start = Instant::now();
        
        for (key, expected_size) in &[
            (format!("{}/small-1MB.bin", test_key_prefix), 1024 * 1024),
            (format!("{}/medium-4MB.bin", test_key_prefix), 4 * 1024 * 1024),
            (format!("{}/large-16MB.bin", test_key_prefix), 16 * 1024 * 1024),
        ] {
            let uri = format!("s3://{}/{}", bucket, key);
            match stat_object_uri_async(&uri).await {
                Ok(stat) => {
                    println!("   📄 {} -> {} bytes (expected: {})", key, stat.size, expected_size);
                    if stat.size != *expected_size as u64 {
                        println!("   ⚠️  Size mismatch for {}", key);
                    }
                }
                Err(e) => println!("   ❌ Failed to stat {}: {}", key, e),
            }
        }
        
        println!("   ⏱️  Stat operations completed in {:?}", start.elapsed());
    }
    
    // Profile optimized GET operations (the main performance focus)
    {
        let _span = profile_span!("s3_get_operations", bucket = %bucket);
        let start = Instant::now();
        
        for (key, expected_size) in &[
            (format!("{}/small-1MB.bin", test_key_prefix), 1024 * 1024),
            (format!("{}/medium-4MB.bin", test_key_prefix), 4 * 1024 * 1024),
            (format!("{}/large-16MB.bin", test_key_prefix), 16 * 1024 * 1024),
        ] {
            let uri = format!("s3://{}/{}", bucket, key);
            let _download_span = profile_span!("download_single", key = %key, expected_size = %expected_size);
            
            match get_object_uri_optimized_async(&uri).await {
                Ok(data) => {
                    println!("   � Downloaded {} -> {} bytes", key, data.len());
                    if data.len() != *expected_size {
                        println!("   ⚠️  Downloaded size mismatch for {}: got {}, expected {}", 
                                key, data.len(), expected_size);
                    }
                }
                Err(e) => println!("   ❌ Failed to download {}: {}", key, e),
            }
        }
        
        println!("   ⏱️  Download operations completed in {:?}", start.elapsed());
    }
    
    // Profile range GET operations
    {
        let _span = profile_span!("s3_range_get_operations", bucket = %bucket);
        let start = Instant::now();
        
        // Test range downloads on the large object
        let large_key = format!("{}/large-16MB.bin", test_key_prefix);
        let uri = format!("s3://{}/{}", bucket, large_key);
        
        // Download first 1MB, middle 1MB, and last 1MB
        let range_tests = [
            ("first_1MB", 0, Some(1024 * 1024)),
            ("middle_1MB", 8 * 1024 * 1024, Some(1024 * 1024)),
            ("last_1MB", 15 * 1024 * 1024, Some(1024 * 1024)),
        ];
        
        for (test_name, offset, length) in &range_tests {
            let _range_span = profile_span!("range_download", 
                test = %test_name, 
                offset = %offset, 
                length = ?length
            );
            
            match get_object_range_uri_async(&uri, *offset, *length).await {
                Ok(data) => {
                    println!("   📎 Range {} -> {} bytes (offset: {}, length: {:?})", 
                            test_name, data.len(), offset, length);
                }
                Err(e) => println!("   ❌ Failed range download {}: {}", test_name, e),
            }
        }
        
        println!("   ⏱️  Range GET operations completed in {:?}", start.elapsed());
    }
    
    // Clean up test objects
    {
        let _span = profile_span!("s3_cleanup_operations", bucket = %bucket);
        let start = Instant::now();
        
        // Use delete_objects with just the keys
        let keys_to_delete: Vec<String> = [
            format!("{}/small-1MB.bin", test_key_prefix),
            format!("{}/medium-4MB.bin", test_key_prefix), 
            format!("{}/large-16MB.bin", test_key_prefix),
        ].into_iter().collect();
        
        match delete_objects(&bucket, &keys_to_delete) {
            Ok(()) => println!("   🗑️  Deleted {} test objects", keys_to_delete.len()),
            Err(e) => println!("   ❌ Failed to delete test objects: {}", e),
        }
        
        println!("   ⏱️  Cleanup operations completed in {:?}", start.elapsed());
    }
    
    Ok(())
}

/// Profile synthetic workload to demonstrate profiling capabilities
async fn profile_synthetic_workload() -> Result<()> {
    println!("\n🧪 Profiling Synthetic Workload");
    println!("==============================");
    
    // Profile data generation at different sizes
    {
        let section_profiler = profile_section("data_generation_suite")?;
        let _span = profile_span!("synthetic_data_generation");
        
        let sizes = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]; // 1MB, 4MB, 16MB
        let start = Instant::now();
        
        for &size in &sizes {
            let data_profiler = profile_section(&format!("generate_{}MB", size / (1024 * 1024)))?;
            
            let data = generate_controlled_data(size, 10, 1);
            println!("   📊 Generated {} MB of test data", data.len() / (1024 * 1024));
            
            // Optionally save individual flamegraph
            if env::var("S3DLIO_SAVE_INDIVIDUAL_PROFILES").is_ok() {
                data_profiler.save_flamegraph(&format!("profiles/profile_generate_{}MB.svg", size / (1024 * 1024)))?;
            }
        }
        
        println!("   ⏱️  Data generation completed in {:?}", start.elapsed());
        
        // Save overall flamegraph
        section_profiler.save_flamegraph("profiles/profile_data_generation_suite.svg")?;
        println!("   📈 Saved flamegraph: profiles/profile_data_generation_suite.svg");
    }
    
    Ok(())
}

/// Profile local operations (buffer management, parsing, etc.)
async fn profile_local_operations() -> Result<()> {
    use s3dlio::s3_utils::parse_s3_uri;
    
    println!("\n⚡ Profiling Local Operations");
    println!("============================");
    
    // Profile URI parsing (happens on every S3 operation)
    {
        let _span = profile_span!("uri_parsing_operations");
        let start = Instant::now();
        
        let test_uris = vec![
            "s3://bucket/key",
            "s3://my-bucket/path/to/file.dat",
            "s3://my-bucket/very/deeply/nested/path/with/many/segments/large-file.bin",
        ];
        
        for _ in 0..10000 {
            for uri in &test_uris {
                let _result = parse_s3_uri(uri);
            }
        }
        
        println!("   📊 Parsed 30K URIs in {:?}", start.elapsed());
    }
    
    // Profile buffer operations (common in multipart uploads)
    {
        let _span = profile_span!("buffer_operations");
        let start = Instant::now();
        
        let data = vec![42u8; 16 * 1024 * 1024]; // 16MB
        let mut total_copied = 0;
        
        // Simulate multipart buffer management
        for chunk_size in [4 * 1024 * 1024, 8 * 1024 * 1024] { // 4MB, 8MB chunks
            let mut offset = 0;
            while offset + chunk_size <= data.len() {
                let mut chunk = Vec::with_capacity(chunk_size);
                chunk.extend_from_slice(&data[offset..offset + chunk_size]);
                total_copied += chunk.len();
                offset += chunk_size;
            }
        }
        
        println!("   📊 Copied {} MB in buffer operations", total_copied / (1024 * 1024));
        println!("   ⏱️  Buffer operations completed in {:?}", start.elapsed());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use s3dlio::profile_async;
    
    #[tokio::test]
    async fn test_profiling_infrastructure() {
        // Test that profiling can be initialized without errors
        assert!(init_profiling().is_ok());
        
        // Test section profiler creation
        let profiler = profile_section("test_section");
        assert!(profiler.is_ok());
        
        // Test profiling macros
        let _span = profile_span!("test_span");
        let result = profile_async!("test_async", async { 42 }).await;
        assert_eq!(result, 42);
    }
}
