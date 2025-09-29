// examples/backend_comparison_demo.rs
//
// Functional test comparing native vs Arrow backends

use anyhow::Result;
use std::time::Instant;
use tempfile::TempDir;

#[cfg(feature = "native-backends")]
use s3dlio::api::{store_for_uri, WriterOptions, CompressionConfig};

#[cfg(feature = "arrow-backend")]
use s3dlio::api::{store_for_uri, WriterOptions, CompressionConfig};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔧 S3DLIO Backend Functional Comparison");
    println!("=====================================");
    
    // Determine which backend we're using
    let backend_name = if cfg!(feature = "arrow-backend") {
        "Apache Arrow"
    } else {
        "Native AWS/Azure"
    };
    
    println!("🎯 Testing with {} backend", backend_name);
    
    // Create temporary directory for testing
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    let test_uri = format!("file://{}", base_path.display());
    
    println!("📁 Test directory: {}", base_path.display());
    
    // Create store
    let start = Instant::now();
    let store = store_for_uri(&test_uri)?;
    let create_time = start.elapsed();
    println!("⚡ Store creation: {:?}", create_time);
    
    // Test 1: Simple PUT/GET operations
    println!("\n🧪 Test 1: Basic PUT/GET operations");
    let test_data = b"Hello, s3dlio backend comparison!";
    let test_file = format!("{}/test1.txt", test_uri);
    
    let start = Instant::now();
    store.put(&test_file, test_data).await?;
    let put_time = start.elapsed();
    println!("   PUT ({} bytes): {:?}", test_data.len(), put_time);
    
    let start = Instant::now();
    let retrieved = store.get(&test_file).await?;
    let get_time = start.elapsed();
    println!("   GET ({} bytes): {:?}", retrieved.len(), get_time);
    
    assert_eq!(test_data, &retrieved[..]);
    println!("   ✅ Data integrity verified");
    
    // Test 2: Streaming writer operations
    println!("\n🧪 Test 2: Streaming writer operations");
    let stream_file = format!("{}/test2.txt", test_uri);
    let chunk_size = 1024;
    let num_chunks = 10;
    
    let start = Instant::now();
    let mut writer = store.create_writer(&stream_file, WriterOptions::new()).await?;
    
    for i in 0..num_chunks {
        let chunk_data = vec![i as u8; chunk_size];
        writer.write_chunk(&chunk_data).await?;
    }
    
    writer.finalize().await?;
    let streaming_time = start.elapsed();
    println!("   Streaming write ({} chunks × {} bytes): {:?}", 
             num_chunks, chunk_size, streaming_time);
    
    // Verify streamed data
    let streamed = store.get(&stream_file).await?;
    assert_eq!(streamed.len(), chunk_size * num_chunks);
    println!("   ✅ Streamed data verified ({} bytes)", streamed.len());
    
    // Test 3: Compression operations (if supported)
    println!("\n🧪 Test 3: Compression operations");
    let compressed_file = format!("{}/test3.txt.zst", test_uri);
    let large_data = vec![42u8; 10000]; // Repetitive data compresses well
    
    let compression = CompressionConfig::zstd_level(6);
    let options = WriterOptions::new().with_compression(compression);
    
    let start = Instant::now();
    match store.create_writer(&compressed_file, options).await {
        Ok(mut writer) => {
            writer.write_chunk(&large_data).await?;
            let stats_before = (writer.bytes_written(), writer.compressed_bytes());
            writer.finalize().await?;
            let compress_time = start.elapsed();
            
            println!("   Compression write: {:?}", compress_time);
            println!("   Uncompressed: {} bytes", stats_before.0);
            println!("   Compressed: {} bytes", stats_before.1);
            if stats_before.0 > 0 {
                let ratio = stats_before.1 as f64 / stats_before.0 as f64;
                println!("   Ratio: {:.2}x", ratio);
            }
            println!("   ✅ Compression supported");
        }
        Err(e) => {
            println!("   ⚠️  Compression not supported: {}", e);
        }
    }
    
    // Test 4: Metadata operations
    println!("\n🧪 Test 4: Metadata operations");
    let start = Instant::now();
    let metadata = store.stat(&test_file).await?;
    let stat_time = start.elapsed();
    println!("   STAT operation: {:?}", stat_time);
    println!("   File size: {} bytes", metadata.size);
    if let Some(modified) = &metadata.last_modified {
        println!("   Last modified: {}", modified);
    }
    println!("   ✅ Metadata retrieved");
    
    // Test 5: List operations
    println!("\n🧪 Test 5: List operations");
    let start = Instant::now();
    let files = store.list(&test_uri, false).await?;
    let list_time = start.elapsed();
    println!("   LIST operation: {:?}", list_time);
    println!("   Found {} files", files.len());
    for file in &files {
        println!("     - {}", file);
    }
    println!("   ✅ Listing successful");
    
    // Test 6: Delete operations  
    println!("\n🧪 Test 6: Delete operations");
    let start = Instant::now();
    store.delete(&test_file).await?;
    let delete_time = start.elapsed();
    println!("   DELETE operation: {:?}", delete_time);
    
    // Verify deletion
    match store.stat(&test_file).await {
        Ok(_) => println!("   ❌ File still exists after delete"),
        Err(_) => println!("   ✅ File successfully deleted"),
    }
    
    // Summary
    println!("\n📊 Performance Summary for {} Backend", backend_name);
    println!("==========================================");
    println!("Store creation:     {:?}", create_time);
    println!("PUT operation:      {:?}", put_time);
    println!("GET operation:      {:?}", get_time);
    println!("Streaming write:    {:?}", streaming_time);
    println!("STAT operation:     {:?}", stat_time);
    println!("LIST operation:     {:?}", list_time);
    println!("DELETE operation:   {:?}", delete_time);
    
    println!("\n✨ All tests completed successfully!");
    Ok(())
}