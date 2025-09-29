// examples/test_direct_io.rs
//! Test O_DIRECT functionality specifically

use anyhow::Result;
use s3dlio::api::{direct_io_store_for_uri, store_for_uri};

#[tokio::main]
async fn main() -> Result<()> {
    let test_dir = "/home/eval/Documents/Rust-Devel/s3dlio/test_direct_io/";
    std::fs::create_dir_all(test_dir)?;
    
    println!("Testing O_DIRECT on: {}", test_dir);
    
    // Test if the path supports O_DIRECT
    println!("Filesystem: {:?}", std::fs::metadata(test_dir)?.file_type());
    
    // Create direct I/O store
    let direct_store = match direct_io_store_for_uri(&format!("file://{}", test_dir)) {
        Ok(store) => {
            println!("✓ Direct I/O store created successfully");
            store
        }
        Err(e) => {
            println!("✗ Failed to create direct I/O store: {}", e);
            return Ok(());
        }
    };
    
    // Test with various data sizes
    let sizes = vec![512, 1024, 2048, 4096, 8192];
    
    for size in sizes {
        println!("\n--- Testing size: {} bytes ---", size);
        
        // Create aligned data
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let test_file = format!("file://{}test_{}.dat", test_dir, size);
        
        // Try direct write
        match direct_store.put(&test_file, &data).await {
            Ok(()) => {
                println!("✓ Direct write successful for {} bytes", size);
                
                // Try direct read
                match direct_store.get(&test_file).await {
                    Ok(read_data) => {
                        if data == read_data {
                            println!("✓ Direct read successful and data matches");
                        } else {
                            println!("✗ Direct read data mismatch");
                        }
                    }
                    Err(e) => println!("✗ Direct read failed: {}", e),
                }
            }
            Err(e) => {
                println!("✗ Direct write failed for {} bytes: {}", size, e);
                
                // Try with regular store as comparison
                let regular_store = store_for_uri(&format!("file://{}", test_dir))?;
                match regular_store.put(&test_file, &data).await {
                    Ok(()) => println!("  ✓ Regular store write works for same data"),
                    Err(e2) => println!("  ✗ Regular store also fails: {}", e2),
                }
            }
        }
    }
    
    // Test streaming write
    println!("\n--- Testing streaming write ---");
    let stream_file = format!("file://{}stream_test.dat", test_dir);
    
    match direct_store.create_writer(&stream_file, Default::default()).await {
        Ok(mut writer) => {
            println!("✓ Direct I/O writer created");
            
            // Write aligned chunks
            let chunk = vec![0u8; 4096];
            match writer.write_chunk(&chunk).await {
                Ok(()) => {
                    println!("✓ Direct I/O chunk write successful");
                    match writer.finalize().await {
                        Ok(()) => println!("✓ Direct I/O write finalized"),
                        Err(e) => println!("✗ Direct I/O finalize failed: {}", e),
                    }
                }
                Err(e) => println!("✗ Direct I/O chunk write failed: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to create direct I/O writer: {}", e),
    }
    
    println!("\nO_DIRECT test completed");
    Ok(())
}
