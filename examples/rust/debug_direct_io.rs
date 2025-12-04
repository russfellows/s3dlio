#!/usr/bin/env rust
//! Simple test to debug O_DIRECT streaming issue

use anyhow::Result;
use s3dlio::api::direct_io_store_for_uri;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 Debugging O_DIRECT streaming write issue...\n");
    
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("debug_test.dat");
    let test_uri = format!("direct://{}", test_path.display());
    
    let store = direct_io_store_for_uri(&format!("direct://{}/", temp_dir.path().display()))?;
    
    println!("Creating streaming writer...");
    let mut writer = store.create_writer(&test_uri, Default::default()).await?;
    
    // Write three specific chunks with known patterns
    let chunk1 = vec![0x11u8; 2048];  // 2KB of 0x11
    let chunk2 = vec![0x22u8; 1536];  // 1.5KB of 0x22  
    let chunk3 = vec![0x33u8; 4096];  // 4KB of 0x33
    
    println!("Writing chunk 1: {} bytes of 0x11", chunk1.len());
    writer.write_chunk(&chunk1).await?;
    println!("Bytes written so far: {}", writer.bytes_written());
    
    println!("Writing chunk 2: {} bytes of 0x22", chunk2.len());
    writer.write_chunk(&chunk2).await?;
    println!("Bytes written so far: {}", writer.bytes_written());
    
    println!("Writing chunk 3: {} bytes of 0x33", chunk3.len());
    writer.write_chunk(&chunk3).await?;
    println!("Bytes written so far: {}", writer.bytes_written());
    
    println!("Finalizing writer...");
    writer.finalize().await?;
    
    // Check the actual file
    let metadata = std::fs::metadata(&test_path)?;
    let actual_size = metadata.len();
    let expected_size = chunk1.len() + chunk2.len() + chunk3.len();
    
    println!("\nResults:");
    println!("  Expected size: {} bytes", expected_size);
    println!("  Actual size: {} bytes", actual_size);
    
    if actual_size as usize == expected_size {
        println!("✅ File size correct!");
    } else {
        println!("❌ File size mismatch!");
        
        // Read the file and check its contents
        let contents = std::fs::read(&test_path)?;
        println!("  File contents length: {}", contents.len());
        
        if !contents.is_empty() {
            println!("  First 16 bytes: {:02x?}", &contents[..std::cmp::min(16, contents.len())]);
            if contents.len() >= 16 {
                println!("  Last 16 bytes: {:02x?}", &contents[contents.len()-16..]);
            }
        }
    }
    
    println!("\n🏁 Debug test completed!");
    Ok(())
}