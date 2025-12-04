use anyhow::Result;
use s3dlio::api::direct_io_store_for_uri;

#[tokio::main]
async fn main() -> Result<()> {
    let test_dir = "/home/eval/Documents/Rust-Devel/s3dlio/test_direct_io/";
    
    println!("Testing O_DIRECT comprehensive scenarios on: {}", test_dir);
    
    // Ensure directory exists
    std::fs::create_dir_all(test_dir)?;
    
    let direct_store = direct_io_store_for_uri(&format!("file://{}", test_dir))?;
    
    // Test 1: Streaming write with aligned data (like our previous test)
    println!("\n--- Test 1: Streaming write with perfectly aligned data ---");
    let stream_file1 = format!("file://{}stream_aligned.dat", test_dir);
    
    match direct_store.create_writer(&stream_file1, Default::default()).await {
        Ok(mut writer) => {
            println!("✓ Direct I/O writer created");
            
            // Write perfectly aligned 4096-byte chunk
            let chunk = vec![42u8; 4096];
            match writer.write_chunk(&chunk).await {
                Ok(()) => {
                    println!("✓ Direct I/O aligned chunk write successful");
                    match writer.finalize().await {
                        Ok(()) => println!("✓ Direct I/O aligned write finalized"),
                        Err(e) => println!("✗ Direct I/O aligned finalize failed: {}", e),
                    }
                }
                Err(e) => println!("✗ Direct I/O aligned chunk write failed: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to create direct I/O writer: {}", e),
    }
    
    // Test 2: Streaming write with unaligned data (hybrid I/O)
    println!("\n--- Test 2: Streaming write with unaligned data (hybrid I/O) ---");
    let stream_file2 = format!("file://{}stream_hybrid.dat", test_dir);
    
    match direct_store.create_writer(&stream_file2, Default::default()).await {
        Ok(mut writer) => {
            println!("✓ Direct I/O writer created for hybrid test");
            
            // Write aligned chunk first
            let aligned_chunk = vec![42u8; 4096];
            writer.write_chunk(&aligned_chunk).await?;
            println!("✓ Direct I/O aligned chunk written");
            
            // Write unaligned chunk that will need hybrid I/O
            let unaligned_chunk = vec![24u8; 1000]; // Not a multiple of 512 or 4096
            match writer.write_chunk(&unaligned_chunk).await {
                Ok(()) => {
                    println!("✓ Direct I/O unaligned chunk write successful");
                    match writer.finalize().await {
                        Ok(()) => println!("✓ Direct I/O hybrid write finalized"),
                        Err(e) => println!("✗ Direct I/O hybrid finalize failed: {}", e),
                    }
                }
                Err(e) => println!("✗ Direct I/O unaligned chunk write failed: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to create direct I/O writer for hybrid test: {}", e),
    }
    
    // Test 3: Multiple small unaligned writes
    println!("\n--- Test 3: Multiple small unaligned writes ---");
    let stream_file3 = format!("file://{}stream_small.dat", test_dir);
    
    match direct_store.create_writer(&stream_file3, Default::default()).await {
        Ok(mut writer) => {
            println!("✓ Direct I/O writer created for small writes test");
            
            // Write several small chunks that will accumulate in the buffer
            for i in 0..10 {
                let small_chunk = vec![(i + 65) as u8; 300]; // 300 bytes each, total 3000 bytes
                writer.write_chunk(&small_chunk).await?;
            }
            println!("✓ Direct I/O multiple small chunks written");
            
            match writer.finalize().await {
                Ok(()) => println!("✓ Direct I/O small writes finalized"),
                Err(e) => println!("✗ Direct I/O small writes finalize failed: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to create direct I/O writer for small writes test: {}", e),
    }
    
    // Test 4: Verify file contents are correct
    println!("\n--- Test 4: Verifying file contents ---");
    
    // Check the aligned file first
    let aligned_path = format!("{}stream_aligned.dat", test_dir);
    if let Ok(contents) = std::fs::read(&aligned_path) {
        println!("Aligned file size: {} bytes (expected 4096)", contents.len());
        if contents.len() == 4096 {
            let all_correct = contents.iter().all(|&b| b == 42);
            println!("Aligned file content correct: {}", all_correct);
        }
    }
    
    // Check the hybrid file
    let hybrid_path = format!("{}stream_hybrid.dat", test_dir);
    if let Ok(contents) = std::fs::read(&hybrid_path) {
        let expected_size = 4096 + 1000; // aligned chunk + unaligned chunk
        println!("Hybrid file size: {} bytes (expected ~{})", contents.len(), expected_size);
        
        if contents.len() >= 4096 {
            // Verify the first 4096 bytes are 42
            let aligned_correct = contents[..4096].iter().all(|&b| b == 42);
            println!("Aligned portion correct: {}", aligned_correct);
            
            // Verify the next portion contains our unaligned data (24s)
            if contents.len() >= 4096 + 1000 {
                let unaligned_correct = contents[4096..4096+1000].iter().all(|&b| b == 24);
                println!("Unaligned portion correct: {}", unaligned_correct);
            }
        } else {
            println!("⚠️  Hybrid file is missing O_DIRECT data (only {} bytes, needs at least 4096)", contents.len());
        }
    }
    
    // Check the small writes file
    let small_path = format!("{}stream_small.dat", test_dir);
    if let Ok(contents) = std::fs::read(&small_path) {
        println!("Small writes file size: {} bytes (expected 3000)", contents.len());
        if contents.len() == 3000 {
            // Check that the pattern is correct (should be A-J repeated)
            let mut expected_byte = 65u8; // 'A'
            let mut correct = true;
            for (i, &byte) in contents.iter().enumerate() {
                if i % 300 == 0 && i > 0 {
                    expected_byte += 1;
                }
                if byte != expected_byte {
                    correct = false;
                    break;
                }
            }
            println!("Small writes pattern correct: {}", correct);
        }
    }
    
    println!("\nO_DIRECT comprehensive test completed");
    Ok(())
}
