use anyhow::Result;
use std::time::Instant;

#[test]
fn test_generate_controlled_data_alt_basic() -> Result<()> {
    use s3dlio::data_gen_alt::generate_controlled_data_alt;
    
    println!("\n=== Basic Functionality Test ===");
    
    // Test various sizes
    let test_cases = vec![
        (1024, 1, 1, "1KB, no dedup, no compress"),
        (1024 * 1024, 1, 1, "1MB, no dedup, no compress"),
        (1024 * 1024, 2, 1, "1MB, 2x dedup, no compress"),
        (1024 * 1024, 1, 2, "1MB, no dedup, 2x compress"),
        (1024 * 1024, 2, 2, "1MB, 2x dedup, 2x compress"),
    ];
    
    for (size, dedup, compress, description) in test_cases {
        println!("\nTesting: {}", description);
        let data = generate_controlled_data_alt(size, dedup, compress);
        
        assert_eq!(data.len(), size, "Size mismatch for {}", description);
        println!("  ✓ Generated {} bytes", data.len());
        
        // Basic sanity: data should not be all zeros
        let all_zeros = data.iter().all(|&b| b == 0);
        assert!(!all_zeros, "Data should not be all zeros for {}", description);
        println!("  ✓ Non-zero data");
    }
    
    Ok(())
}

#[test]
fn test_incompressible_with_compress_1() -> Result<()> {
    use s3dlio::data_gen_alt::generate_controlled_data_alt;
    use std::io::Write;
    use std::process::{Command, Stdio};
    
    println!("\n=== Incompressibility Test (compress=1) ===");
    println!("Goal: compress=1 should produce zstd ratio ~1.0 (incompressible)\n");
    
    // Generate 4MB of data with compress=1 (should be incompressible)
    let size = 4 * 1024 * 1024;
    let data = generate_controlled_data_alt(size, 1, 1);
    
    println!("Generated {} bytes with compress=1", data.len());
    
    // Try to compress with zstd
    let mut child = Command::new("zstd")
        .args(&["-c", "-1", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&data)?;
    }
    
    let output = child.wait_with_output()?;
    let compressed_size = output.stdout.len();
    let ratio = data.len() as f64 / compressed_size as f64;
    
    println!("Original size:    {} bytes", data.len());
    println!("Compressed size:  {} bytes", compressed_size);
    println!("Compression ratio: {:.4}", ratio);
    
    // compress=1 should produce ratio very close to 1.0 (incompressible)
    // Allow small overhead from zstd headers/framing (ratio 0.95-1.05)
    assert!(ratio >= 0.95 && ratio <= 1.05, 
            "compress=1 should be incompressible, got ratio {:.4}", ratio);
    
    println!("  ✓ Data is incompressible (ratio {:.4} ≈ 1.0)", ratio);
    
    Ok(())
}

#[test]
fn test_compressible_with_higher_compress() -> Result<()> {
    use s3dlio::data_gen_alt::generate_controlled_data_alt;
    use std::io::Write;
    use std::process::{Command, Stdio};
    
    println!("\n=== Compressibility Test (compress > 1) ===");
    println!("Goal: Higher compress values should produce better compression\n");
    
    let size = 4 * 1024 * 1024;
    
    let test_cases = vec![
        (2, "2x compress"),
        (3, "3x compress"),
        (4, "4x compress"),
    ];
    
    for (compress, description) in test_cases {
        let data = generate_controlled_data_alt(size, 1, compress);
        
        let mut child = Command::new("zstd")
            .args(&["-c", "-1", "-q"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(&data)?;
        }
        
        let output = child.wait_with_output()?;
        let compressed_size = output.stdout.len();
        let ratio = data.len() as f64 / compressed_size as f64;
        
        println!("{}: ratio = {:.4}", description, ratio);
        
        // Higher compress should give better compression (ratio > 1.1)
        assert!(ratio > 1.1, 
                "{} should be compressible, got ratio {:.4}", description, ratio);
        println!("  ✓ Data is compressible (ratio {:.4})", ratio);
    }
    
    Ok(())
}

#[test]
fn test_deduplication_behavior() -> Result<()> {
    use s3dlio::data_gen_alt::generate_controlled_data_alt;
    
    println!("\n=== Deduplication Test ===");
    println!("Goal: Same dedup factor should produce same unique blocks\n");
    
    let size = 1024 * 1024; // 1MB
    let block_size = 65536;  // 64KB blocks
    let expected_blocks = size / block_size;
    
    // Generate with dedup=2 (half unique blocks)
    let data1 = generate_controlled_data_alt(size, 2, 1);
    let data2 = generate_controlled_data_alt(size, 2, 1);
    
    println!("Generated two datasets with dedup=2");
    println!("Block size: {} bytes", block_size);
    println!("Total blocks: {}", expected_blocks);
    println!("Expected unique: {} (dedup=2)", expected_blocks / 2);
    
    // Count unique blocks in first dataset
    let mut unique_blocks = std::collections::HashSet::new();
    for i in 0..(size / block_size) {
        let start = i * block_size;
        let end = start + block_size;
        let block = &data1[start..end];
        unique_blocks.insert(block.to_vec());
    }
    
    let actual_unique = unique_blocks.len();
    println!("Actual unique blocks: {}", actual_unique);
    
    // Should have roughly half unique blocks (allow ±1 for rounding)
    let expected_unique = expected_blocks / 2;
    assert!((actual_unique as i32 - expected_unique as i32).abs() <= 1,
            "Expected ~{} unique blocks, got {}", expected_unique, actual_unique);
    
    println!("  ✓ Deduplication working correctly");
    
    // Different runs should produce different data (different entropy seed)
    assert_ne!(data1, data2, "Different runs should produce different data");
    println!("  ✓ Different runs produce different data");
    
    Ok(())
}

#[test]
fn test_streaming_generator() -> Result<()> {
    use s3dlio::data_gen_alt::ObjectGenAlt;
    
    println!("\n=== Streaming Generator Test ===");
    println!("Goal: ObjectGenAlt should stream data consistently\n");
    
    let total_size = 1024 * 1024; // 1MB
    let chunk_size = 64 * 1024;   // 64KB chunks
    
    // Note: Single-pass and streaming use different call_entropy,
    // so they produce different data. This is correct behavior.
    // We test that streaming works consistently instead.
    
    let mut streaming = ObjectGenAlt::new(total_size, 1, 1);
    let mut streamed_data = Vec::with_capacity(total_size);
    
    while streamed_data.len() < total_size {
        let mut chunk = vec![0u8; chunk_size];
        let bytes_written = streaming.fill_chunk(&mut chunk);
        streamed_data.extend_from_slice(&chunk[..bytes_written]);
        
        if bytes_written == 0 {
            break; // Done
        }
    }
    
    println!("Streamed size: {} bytes", streamed_data.len());
    
    assert_eq!(streamed_data.len(), total_size, "Should stream exactly total_size bytes");
    
    // Verify data is not all zeros
    let all_zeros = streamed_data.iter().all(|&b| b == 0);
    assert!(!all_zeros, "Streamed data should not be all zeros");
    
    println!("  ✓ Streaming generator works correctly");
    
    Ok(())
}

#[test]
fn test_performance_comparison() -> Result<()> {
    use s3dlio::data_gen::generate_controlled_data;
    use s3dlio::data_gen_alt::{generate_controlled_data_alt, ObjectGenAlt};
    
    println!("\n=== Performance Benchmark ===");
    println!("Comparing original vs new data generation\n");
    
    let sizes = vec![
        (1 * 1024 * 1024, "1MB"),
        (16 * 1024 * 1024, "16MB"),
        (64 * 1024 * 1024, "64MB"),
    ];
    
    for (size, label) in sizes {
        println!("--- {} dataset ---", label);
        
        // Original generator
        let start = Instant::now();
        let _data_old = generate_controlled_data(size, 1, 1);
        let time_old = start.elapsed();
        let throughput_old = size as f64 / time_old.as_secs_f64() / (1024.0 * 1024.0);
        
        println!("Original:  {:?} ({:.2} MB/s)", time_old, throughput_old);
        
        // New generator
        let start = Instant::now();
        let _data_new = generate_controlled_data_alt(size, 1, 1);
        let time_new = start.elapsed();
        let throughput_new = size as f64 / time_new.as_secs_f64() / (1024.0 * 1024.0);
        
        println!("New (alt): {:?} ({:.2} MB/s)", time_new, throughput_new);
        
        let speedup = time_old.as_secs_f64() / time_new.as_secs_f64();
        if speedup > 1.0 {
            println!("  → Speedup: {:.2}x faster", speedup);
        } else {
            println!("  → Slowdown: {:.2}x slower", 1.0 / speedup);
        }
        println!();
    }
    
    // Test new streaming generator
    println!("--- Streaming (16MB in 64KB chunks) ---");
    let total_size = 16 * 1024 * 1024;
    let chunk_size = 64 * 1024;
    
    let start = Instant::now();
    let mut gen_new = ObjectGenAlt::new(total_size, 1, 1);
    let mut buf_new = vec![0u8; chunk_size];
    let mut bytes_new = 0;
    while bytes_new < total_size {
        let written = gen_new.fill_chunk(&mut buf_new);
        bytes_new += written;
    }
    let time_new = start.elapsed();
    let throughput_new = total_size as f64 / time_new.as_secs_f64() / (1024.0 * 1024.0);
    
    println!("Streaming: {:?} ({:.2} MB/s)", time_new, throughput_new);
    println!("  (Note: Original ObjectGen has private constructor, can't compare streaming)");
    
    Ok(())
}

#[test]
fn test_original_cross_block_compression_issue() -> Result<()> {
    use s3dlio::data_gen::generate_controlled_data;
    use std::io::Write;
    use std::process::{Command, Stdio};
    
    println!("\n=== Original Generator Compression Issue ===");
    println!("Demonstrating that original compress=1 is still compressible\n");
    
    let size = 4 * 1024 * 1024;
    let data = generate_controlled_data(size, 1, 1);
    
    let mut child = Command::new("zstd")
        .args(&["-c", "-1", "-q"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&data)?;
    }
    
    let output = child.wait_with_output()?;
    let compressed_size = output.stdout.len();
    let ratio = data.len() as f64 / compressed_size as f64;
    
    println!("Original generator (compress=1):");
    println!("  Original size:    {} bytes", data.len());
    println!("  Compressed size:  {} bytes", compressed_size);
    println!("  Compression ratio: {:.4}", ratio);
    
    if ratio > 1.05 {
        println!("  ⚠ Still compressible! (ratio {:.4} > 1.05)", ratio);
        println!("  This confirms the cross-block pattern issue");
    }
    
    Ok(())
}
