// examples/basic_usage.rs
//! Basic usage examples for the s3dlio Rust API
//!
//! This example demonstrates the core functionality:
//! - Creating stores for different backends
//! - Reading and writing objects
//! - Using streaming writes with compression
//! - Error handling patterns

use anyhow::Result;
use s3dlio::api::{
    store_for_uri, direct_io_store_for_uri,
    WriterOptions, CompressionConfig
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Example 1: Basic read/write operations
    basic_operations().await?;
    
    // Example 2: Streaming with compression
    streaming_example().await?;
    
    // Example 3: High-performance operations
    high_performance_example().await?;
    
    // Example 4: Cross-backend operations
    cross_backend_example().await?;

    println!("All examples completed successfully!");
    Ok(())
}

/// Demonstrates basic read/write operations
async fn basic_operations() -> Result<()> {
    println!("=== Basic Operations ===");
    
    // Create a store - automatically detects backend from URI
    let store = store_for_uri("file:///tmp/s3dlio_test/")?;
    
    // Write some data
    let test_data = b"Hello, s3dlio!";
    store.put("file:///tmp/s3dlio_test/test.txt", test_data).await?;
    println!("✓ Wrote {} bytes to test.txt", test_data.len());
    
    // Read it back
    let read_data = store.get("file:///tmp/s3dlio_test/test.txt").await?;
    println!("✓ Read {} bytes from test.txt", read_data.len());
    
    // Verify the data
    assert_eq!(test_data, &read_data[..]);
    println!("✓ Data integrity verified");
    
    // List objects
    let objects = store.list("file:///tmp/s3dlio_test/", false).await?;
    println!("✓ Found {} objects", objects.len());
    
    Ok(())
}

/// Demonstrates streaming writes with compression
async fn streaming_example() -> Result<()> {
    println!("\n=== Streaming with Compression ===");
    
    let store = store_for_uri("file:///tmp/s3dlio_test/")?;
    
    // Configure compression  
    let options = WriterOptions::new()
        .with_compression(CompressionConfig::zstd_default());
    
    // Create a streaming writer
    let mut writer = store.create_writer("file:///tmp/s3dlio_test/large_file.dat", options).await?;
    
    // Write data in chunks
    let chunks = vec![
        b"First chunk of data\n".to_vec(),
        b"Second chunk of data\n".to_vec(),
        b"Third chunk of data\n".to_vec(),
    ];
    
    for (i, chunk) in chunks.iter().enumerate() {
        writer.write_chunk(chunk).await?;
        println!("✓ Wrote chunk {} ({} bytes)", i + 1, chunk.len());
    }
    
    // Finalize and get statistics
    writer.finalize().await?;
    println!("✓ Streaming complete");
    
    Ok(())
}

/// Demonstrates high-performance operations
async fn high_performance_example() -> Result<()> {
    println!("\n=== High Performance Operations ===");
    
    // Try O_DIRECT on the current working directory (should support O_DIRECT)
    // Fall back to regular store if O_DIRECT fails
    let test_dir = "/home/eval/Documents/Rust-Devel/s3dlio/test_perf/";
    let (store, is_high_perf) = match direct_io_store_for_uri(&format!("file://{}", test_dir)) {
        Ok(store) => {
            println!("✓ Using O_DIRECT store");
            (store, true)
        }
        Err(e) => {
            println!("⚠ O_DIRECT not available ({}), using regular store", e);
            (store_for_uri(&format!("file://{}", test_dir))?, false)
        }
    };
    
    // Create test directory
    std::fs::create_dir_all(test_dir).ok();
    
    // Generate some test data - O_DIRECT requires proper alignment
    // Use 4KB aligned size for better O_DIRECT compatibility
    let size = 4096 * 4; // 16KB, aligned to page boundary
    let large_data: Vec<u8> = (0..size)
        .map(|i| (i % 256) as u8)
        .collect();

    // Write with O_DIRECT (use put instead of streaming to avoid finalize issue)
    let test_file = format!("file://{}large_file.dat", test_dir);
    let write_result = store.put(&test_file, &large_data).await;
    
    match write_result {
        Ok(()) => {
            if is_high_perf {
                println!("✓ O_DIRECT write completed successfully");
            } else {
                println!("✓ Regular write completed");
            }
        }
        Err(e) => {
            println!("✗ Write failed ({}), trying regular store", e);
            let regular_store = store_for_uri(&format!("file://{}", test_dir))?;
            regular_store.put(&test_file, &large_data).await?;
            println!("✓ Fallback write completed");
        }
    }
    
    // Read it back
    let read_result = store.get(&test_file).await;
    match read_result {
        Ok(read_data) => {
            assert_eq!(large_data.len(), read_data.len());
            if is_high_perf {
                println!("✓ High-performance read with O_DIRECT: {} bytes", read_data.len());
            } else {
                println!("✓ Regular read: {} bytes", read_data.len());
            }
        }
        Err(e) if is_high_perf => {
            println!("⚠ O_DIRECT read failed ({}), using regular store", e);
            let regular_store = store_for_uri(&format!("file://{}", test_dir))?;
            let read_data = regular_store.get(&test_file).await?;
            assert_eq!(large_data.len(), read_data.len());
            println!("✓ Fallback read: {} bytes", read_data.len());
        }
        Err(e) => return Err(e),
    }
    
    Ok(())
}

/// Demonstrates operations across different backends
async fn cross_backend_example() -> Result<()> {
    println!("\n=== Cross-Backend Operations ===");
    
    // Create stores for different backends
    let local_store = store_for_uri("file:///tmp/s3dlio_local/")?;
    
    // Note: These would work with real cloud credentials
    // let s3_store = store_for_uri("s3://my-bucket/data/")?;
    // let azure_store = store_for_uri("azure://my-container/data/")?;
    
    // Copy data between backends
    let test_data = b"Cross-backend test data";
    local_store.put("file:///tmp/s3dlio_local/cross_test.txt", test_data).await?;
    
    // Read from one backend
    let data = local_store.get("file:///tmp/s3dlio_local/cross_test.txt").await?;
    
    // Write to another backend (simulated)
    local_store.put("file:///tmp/s3dlio_local/cross_test_copy.txt", &data).await?;
    
    println!("✓ Cross-backend copy completed");
    
    Ok(())
}

/// Demonstrates error handling patterns
#[allow(dead_code)]
async fn error_handling_example() -> Result<()> {
    println!("\n=== Error Handling ===");
    
    let store = store_for_uri("file:///tmp/s3dlio_test/")?;
    
    // Handle missing object
    match store.get("file:///tmp/s3dlio_test/nonexistent.txt").await {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("✓ Handled expected error: {}", e),
    }
    
    // Use the ? operator for automatic error propagation
    let _existing_data = store.get("file:///tmp/s3dlio_test/test.txt").await?;
    println!("✓ Successfully read existing file");
    
    Ok(())
}
