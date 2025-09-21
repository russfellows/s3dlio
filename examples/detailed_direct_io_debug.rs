#!/usr/bin/env rust
//! Detailed debug test to understand O_DIRECT buffer handling

use anyhow::Result;
use s3dlio::api::direct_io_store_for_uri;
use tempfile::TempDir;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔬 Detailed O_DIRECT buffer debugging...\n");
    
    let temp_dir = TempDir::new()?;
    let test_path = temp_dir.path().join("detailed_debug.dat");
    let test_uri = format!("direct://{}", test_path.display());
    
    let store = direct_io_store_for_uri(&format!("direct://{}/", temp_dir.path().display()))?;
    
    // Test with very specific sizes to understand alignment behavior
    println!("Creating streaming writer...");
    let mut writer = store.create_writer(&test_uri, Default::default()).await?;
    
    // Write small chunk first (should stay in buffer)
    let chunk1 = vec![0xAAu8; 1000];  // 1KB - smaller than min_io_size (64KB)
    println!("Writing chunk 1: {} bytes of 0xAA (should stay in buffer)", chunk1.len());
    writer.write_chunk(&chunk1).await?;
    
    // Check file immediately after first chunk
    if test_path.exists() {
        let size = fs::metadata(&test_path)?.len();
        println!("  File size after chunk 1: {} bytes", size);
    } else {
        println!("  File doesn't exist yet after chunk 1");
    }
    
    // Write another small chunk
    let chunk2 = vec![0xBBu8; 2000];  // 2KB 
    println!("Writing chunk 2: {} bytes of 0xBB (should stay in buffer)", chunk2.len());
    writer.write_chunk(&chunk2).await?;
    
    if test_path.exists() {
        let size = fs::metadata(&test_path)?.len();
        println!("  File size after chunk 2: {} bytes", size);
    } else {
        println!("  File doesn't exist yet after chunk 2");
    }
    
    // Write a large chunk that should trigger O_DIRECT write
    let chunk3 = vec![0xCCu8; 70000];  // 70KB - larger than min_io_size (64KB)
    println!("Writing chunk 3: {} bytes of 0xCC (should trigger O_DIRECT)", chunk3.len());
    writer.write_chunk(&chunk3).await?;
    
    if test_path.exists() {
        let size = fs::metadata(&test_path)?.len();
        println!("  File size after chunk 3: {} bytes", size);
        
        // Read a sample to see what was written
        let contents = fs::read(&test_path)?;
        println!("  Contents length: {}", contents.len());
        if !contents.is_empty() {
            println!("  First 16 bytes: {:02x?}", &contents[..std::cmp::min(16, contents.len())]);
            if contents.len() > 3000 {
                println!("  Bytes 3000-3016: {:02x?}", &contents[3000..std::cmp::min(3016, contents.len())]);
            }
        }
    } else {
        println!("  File still doesn't exist after chunk 3!");
    }
    
    println!("Total bytes written by writer: {}", writer.bytes_written());
    
    println!("Finalizing writer...");
    writer.finalize().await?;
    
    // Final analysis
    let metadata = fs::metadata(&test_path)?;
    let actual_size = metadata.len();
    let expected_size = chunk1.len() + chunk2.len() + chunk3.len();
    
    println!("\nFinal Results:");
    println!("  Expected total: {} bytes", expected_size);
    println!("  Actual file size: {} bytes", actual_size);
    
    let contents = fs::read(&test_path)?;
    if !contents.is_empty() {
        println!("  Contents length: {}", contents.len());
        println!("  First 16 bytes: {:02x?}", &contents[..std::cmp::min(16, contents.len())]);
        println!("  Bytes 1000-1016: {:02x?}", &contents[1000..std::cmp::min(1016, contents.len())]);
        println!("  Bytes 3000-3016: {:02x?}", &contents[3000..std::cmp::min(3016, contents.len())]);
        if contents.len() >= 16 {
            println!("  Last 16 bytes: {:02x?}", &contents[contents.len()-16..]);
        }
    }
    
    println!("\n🏁 Detailed debug completed!");
    Ok(())
}