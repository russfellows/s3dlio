use anyhow::Result;
use s3dlio::api::direct_io_store_for_uri;

#[tokio::main]
async fn main() -> Result<()> {
    let test_dir = "/home/eval/Documents/Rust-Devel/s3dlio/test_direct_io/";
    std::fs::create_dir_all(test_dir)?;
    
    let direct_store = direct_io_store_for_uri(&format!("file://{}", test_dir))?;
    
    // Test JUST the hybrid I/O scenario to isolate the truncation issue
    println!("Testing hybrid I/O file truncation issue...");
    let test_file = format!("file://{}hybrid_test.dat", test_dir);
    
    let mut writer = direct_store.create_writer(&test_file, Default::default()).await?;
    
    // Write exactly one aligned chunk
    let aligned_chunk = vec![42u8; 4096];
    writer.write_chunk(&aligned_chunk).await?;
    println!("✓ Aligned chunk written (4096 bytes)");
    
    // Check file size immediately after the aligned write
    let file_path = format!("{}hybrid_test.dat", test_dir);
    if let Ok(metadata) = std::fs::metadata(&file_path) {
        println!("File size after aligned write: {} bytes", metadata.len());
    }
    
    // Write exactly one unaligned chunk
    let unaligned_chunk = vec![24u8; 1000];
    writer.write_chunk(&unaligned_chunk).await?;
    println!("✓ Unaligned chunk written (1000 bytes)");
    
    // Check file size before finalize
    if let Ok(metadata) = std::fs::metadata(&file_path) {
        println!("File size before finalize: {} bytes", metadata.len());
    }
    
    // Finalize (this is where the truncation might happen)
    writer.finalize().await?;
    println!("✓ Finalization completed");
    
    // Check final file size
    if let Ok(metadata) = std::fs::metadata(&file_path) {
        println!("File size after finalize: {} bytes", metadata.len());
        
        if metadata.len() == 5096 {
            println!("✅ SUCCESS: File has correct size (4096 + 1000 = 5096)");
        } else if metadata.len() == 1000 {
            println!("❌ TRUNCATION BUG: File only has unaligned data (1000 bytes)");
        } else {
            println!("❓ UNEXPECTED: File has unexpected size");
        }
    }
    
    // Verify file contents
    if let Ok(contents) = std::fs::read(&file_path) {
        println!("\nFile content analysis:");
        println!("Total bytes: {}", contents.len());
        
        if contents.len() >= 4096 {
            let aligned_correct = contents[..4096].iter().all(|&b| b == 42);
            println!("First 4096 bytes correct (42s): {}", aligned_correct);
            
            if contents.len() >= 5096 {
                let unaligned_correct = contents[4096..5096].iter().all(|&b| b == 24);
                println!("Next 1000 bytes correct (24s): {}", unaligned_correct);
            }
        } else {
            println!("File too small for aligned data verification");
            if contents.len() == 1000 {
                let unaligned_only = contents.iter().all(|&b| b == 24);
                println!("Contains only unaligned data (24s): {}", unaligned_only);
            }
        }
    }
    
    Ok(())
}
