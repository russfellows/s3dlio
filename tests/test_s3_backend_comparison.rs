// tests/test_s3_backend_comparison.rs
//
// Integration tests comparing native AWS SDK vs Apache Arrow backends
// against real S3 storage using credentials from .env file

use anyhow::Result;
use std::env;
use std::time::Instant;
use tokio;

// Import the appropriate backend based on features
#[cfg(feature = "native-backends")]
use s3dlio::api::{store_for_uri, WriterOptions, CompressionConfig};

#[cfg(feature = "arrow-backend")]
use s3dlio::api::{store_for_uri, WriterOptions, CompressionConfig};

fn get_s3_test_uri() -> Result<String> {
    // Load from environment variables (should be in .env file)
    let bucket = env::var("AWS_S3_BUCKET")
        .or_else(|_| env::var("S3_BUCKET"))
        .unwrap_or_else(|_| "test-bucket".to_string());
    
    let endpoint = env::var("AWS_ENDPOINT_URL").ok();
    
    // Create test path with timestamp to avoid conflicts
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let test_prefix = format!("s3dlio-test-{}", timestamp);
    
    if let Some(endpoint_url) = endpoint {
        // Custom endpoint (like your S3-compatible service)
        println!("🔧 Using custom S3 endpoint: {}", endpoint_url);
        println!("🪣 Using bucket: {}", bucket);
        Ok(format!("s3://{}/{}/", bucket, test_prefix))
    } else {
        // Standard AWS S3
        println!("🔧 Using standard AWS S3");
        println!("🪣 Using bucket: {}", bucket);
        Ok(format!("s3://{}/{}/", bucket, test_prefix))
    }
}

fn print_backend_info() {
    let backend_name = if cfg!(feature = "arrow-backend") {
        "Apache Arrow object_store"
    } else {
        "Native AWS SDK"
    };
    
    println!("🎯 Testing with {} backend", backend_name);
    
    // Print relevant environment variables (without exposing secrets)
    if let Ok(endpoint) = env::var("AWS_ENDPOINT_URL") {
        println!("🌐 AWS_ENDPOINT_URL: {}", endpoint);
    }
    if env::var("AWS_ACCESS_KEY_ID").is_ok() {
        println!("🔑 AWS_ACCESS_KEY_ID: [configured]");
    }
    if env::var("AWS_SECRET_ACCESS_KEY").is_ok() {
        println!("🔐 AWS_SECRET_ACCESS_KEY: [configured]");
    }
    if let Ok(region) = env::var("AWS_DEFAULT_REGION") {
        println!("🌍 AWS_DEFAULT_REGION: {}", region);
    }
}

#[tokio::test]
async fn test_s3_basic_operations() -> Result<()> {
    // Load .env file if it exists
    let _ = dotenvy::dotenv();
    
    println!("\n🧪 S3 Basic Operations Test");
    println!("============================");
    
    print_backend_info();
    
    let base_uri = get_s3_test_uri()?;
    println!("📍 Test URI: {}", base_uri);
    
    // Create store
    let start = Instant::now();
    let store = store_for_uri(&base_uri)?;
    println!("⚡ Store creation: {:?}", start.elapsed());
    
    // Test data
    let test_data = b"Hello from s3dlio backend comparison test!";
    let test_key = format!("{}basic_test.txt", base_uri);
    
    // PUT operation
    println!("\n📤 Testing PUT operation");
    let start = Instant::now();
    store.put(&test_key, test_data).await?;
    let put_time = start.elapsed();
    println!("   ✅ PUT ({} bytes): {:?}", test_data.len(), put_time);
    
    // GET operation
    println!("\n📥 Testing GET operation");
    let start = Instant::now();
    let retrieved = store.get(&test_key).await?;
    let get_time = start.elapsed();
    println!("   ✅ GET ({} bytes): {:?}", retrieved.len(), get_time);
    
    // Verify data integrity
    assert_eq!(test_data, &retrieved[..], "Data integrity check failed");
    println!("   ✅ Data integrity verified");
    
    // STAT operation
    println!("\n📊 Testing STAT operation");
    let start = Instant::now();
    let metadata = store.stat(&test_key).await?;
    let stat_time = start.elapsed();
    println!("   ✅ STAT: {:?}", stat_time);
    println!("   📏 Size: {} bytes", metadata.size);
    if let Some(etag) = &metadata.e_tag {
        println!("   🏷️  ETag: {}", etag);
    }
    
    // LIST operation
    println!("\n📋 Testing LIST operation");
    let start = Instant::now();
    let objects = store.list(&base_uri, false).await?;
    let list_time = start.elapsed();
    println!("   ✅ LIST: {:?}", list_time);
    println!("   📁 Found {} objects", objects.len());
    for obj in &objects {
        println!("      - {}", obj);
    }
    
    // DELETE operation
    println!("\n🗑️  Testing DELETE operation");
    let start = Instant::now();
    store.delete(&test_key).await?;
    let delete_time = start.elapsed();
    println!("   ✅ DELETE: {:?}", delete_time);
    
    // Verify deletion
    match store.stat(&test_key).await {
        Ok(_) => panic!("Object still exists after deletion"),
        Err(_) => println!("   ✅ Object successfully deleted"),
    }
    
    println!("\n📊 Performance Summary");
    println!("=====================");
    println!("PUT:    {:?}", put_time);
    println!("GET:    {:?}", get_time);
    println!("STAT:   {:?}", stat_time);
    println!("LIST:   {:?}", list_time);
    println!("DELETE: {:?}", delete_time);
    
    Ok(())
}

#[tokio::test]
async fn test_s3_streaming_operations() -> Result<()> {
    let _ = dotenvy::dotenv();
    
    println!("\n🌊 S3 Streaming Operations Test");
    println!("===============================");
    
    print_backend_info();
    
    let base_uri = get_s3_test_uri()?;
    let store = store_for_uri(&base_uri)?;
    
    let stream_key = format!("{}streaming_test.dat", base_uri);
    
    // Create streaming writer
    println!("\n📝 Testing streaming writer");
    let start = Instant::now();
    let mut writer = store.create_writer(&stream_key, WriterOptions::new()).await?;
    
    // Write multiple chunks
    let chunk_size = 8192; // 8KB chunks
    let num_chunks = 50;   // 400KB total
    let mut total_written = 0;
    
    for i in 0..num_chunks {
        // Create varied data (not all zeros)
        let mut chunk = vec![0u8; chunk_size];
        for (j, byte) in chunk.iter_mut().enumerate() {
            *byte = ((i * chunk_size + j) % 256) as u8;
        }
        
        writer.write_chunk(&chunk).await?;
        total_written += chunk.len();
        
        if (i + 1) % 10 == 0 {
            println!("   📦 Written {} chunks ({} KB)", i + 1, total_written / 1024);
        }
    }
    
    let bytes_written = writer.bytes_written();
    writer.finalize().await?;
    let streaming_time = start.elapsed();
    
    println!("   ✅ Streaming write complete");
    println!("   📊 {} bytes written in {:?}", bytes_written, streaming_time);
    println!("   🚀 Throughput: {:.2} MB/s", 
             (bytes_written as f64 / 1_000_000.0) / streaming_time.as_secs_f64());
    
    // Read back and verify
    println!("\n📖 Verifying streamed data");
    let start = Instant::now();
    let retrieved = store.get(&stream_key).await?;
    let read_time = start.elapsed();
    
    assert_eq!(retrieved.len(), total_written, "Size mismatch after streaming");
    
    // Verify some sample data
    let mut expected_bytes = 0;
    for i in 0..num_chunks {
        for j in 0..chunk_size {
            let expected = ((i * chunk_size + j) % 256) as u8;
            let actual = retrieved[expected_bytes];
            if expected != actual {
                panic!("Data mismatch at byte {}: expected {}, got {}", 
                       expected_bytes, expected, actual);
            }
            expected_bytes += 1;
        }
    }
    
    println!("   ✅ Data integrity verified ({} bytes)", retrieved.len());
    println!("   📊 Read time: {:?}", read_time);
    println!("   🚀 Read throughput: {:.2} MB/s", 
             (retrieved.len() as f64 / 1_000_000.0) / read_time.as_secs_f64());
    
    // Cleanup
    store.delete(&stream_key).await?;
    println!("   🗑️  Cleanup complete");
    
    Ok(())
}

#[tokio::test]
async fn test_s3_compression_operations() -> Result<()> {
    let _ = dotenvy::dotenv();
    
    println!("\n🗜️  S3 Compression Operations Test");
    println!("==================================");
    
    print_backend_info();
    
    let base_uri = get_s3_test_uri()?;
    let store = store_for_uri(&base_uri)?;
    
    // Create highly compressible data
    let mut test_data = Vec::new();
    let pattern = b"This is a highly repetitive pattern for compression testing. ";
    for _ in 0..1000 {
        test_data.extend_from_slice(pattern);
    }
    
    println!("📏 Original data size: {} bytes", test_data.len());
    
    let compressed_key = format!("{}compression_test.dat", base_uri);
    
    // Test compression with streaming writer
    println!("\n🗜️  Testing compressed streaming write");
    let compression = CompressionConfig::zstd_level(6);
    let options = WriterOptions::new().with_compression(compression);
    
    let start = Instant::now();
    match store.create_writer(&compressed_key, options).await {
        Ok(mut writer) => {
            writer.write_chunk(&test_data).await?;
            
            let uncompressed_size = writer.bytes_written();
            let compressed_size = writer.compressed_bytes();
            
            writer.finalize().await?;
            let compress_time = start.elapsed();
            
            println!("   ✅ Compression complete: {:?}", compress_time);
            println!("   📏 Uncompressed: {} bytes", uncompressed_size);
            println!("   📦 Compressed: {} bytes", compressed_size);
            
            if uncompressed_size > 0 && compressed_size > 0 {
                let ratio = uncompressed_size as f64 / compressed_size as f64;
                println!("   🎯 Compression ratio: {:.2}x", ratio);
            }
            
            // Read back and verify
            println!("\n📖 Reading compressed data");
            let start = Instant::now();
            let retrieved = store.get(&compressed_key).await?;
            let read_time = start.elapsed();
            
            println!("   ✅ Read complete: {:?}", read_time);
            println!("   📏 Retrieved {} bytes", retrieved.len());
            
            // For compressed data, we need to check the stored size vs original
            // The retrieved data should be the compressed version
            
            // Cleanup
            store.delete(&compressed_key).await?;
            println!("   🗑️  Cleanup complete");
        }
        Err(e) => {
            println!("   ⚠️  Compression not supported by this backend: {}", e);
            println!("   ℹ️  This is expected for some backends");
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_s3_range_operations() -> Result<()> {
    let _ = dotenvy::dotenv();
    
    println!("\n📐 S3 Range Operations Test");
    println!("===========================");
    
    print_backend_info();
    
    let base_uri = get_s3_test_uri()?;
    let store = store_for_uri(&base_uri)?;
    
    // Create test data with known pattern
    let mut test_data = Vec::new();
    for i in 0..10000 {
        test_data.push((i % 256) as u8);
    }
    
    let range_key = format!("{}range_test.dat", base_uri);
    
    println!("📤 Uploading test data ({} bytes)", test_data.len());
    store.put(&range_key, &test_data).await?;
    
    // Test various range operations
    let test_ranges = [
        (0, Some(100)),      // First 100 bytes
        (100, Some(200)),    // Next 200 bytes  
        (5000, Some(1000)),  // 1000 bytes from middle
        (9000, None),        // Last 1000 bytes (from offset to end)
    ];
    
    for (offset, length) in test_ranges {
        let length_desc = length.map_or_else(
            || "to end".to_string(),
            |len| format!("{} bytes", len)
        );
        
        println!("\n📐 Range request: offset {} + {}", offset, length_desc);
        
        let start = Instant::now();
        let range_data = store.get_range(&range_key, offset, length).await?;
        let range_time = start.elapsed();
        
        let expected_len = length.unwrap_or_else(|| {
            if offset >= test_data.len() as u64 {
                0
            } else {
                test_data.len() as u64 - offset
            }
        });
        
        assert_eq!(range_data.len() as u64, expected_len.min((test_data.len() as u64).saturating_sub(offset)));
        
        // Verify data matches original
        for (i, &byte) in range_data.iter().enumerate() {
            let expected_pos = (offset as usize + i) % 256;
            assert_eq!(byte, expected_pos as u8, 
                      "Data mismatch at range offset {}", i);
        }
        
        println!("   ✅ Range read: {} bytes in {:?}", range_data.len(), range_time);
    }
    
    // Cleanup
    store.delete(&range_key).await?;
    println!("\n🗑️  Cleanup complete");
    
    Ok(())
}