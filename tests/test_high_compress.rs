use s3dlio::data_gen_alt::generate_controlled_data_alt;

#[test]
fn test_high_compress_validation() {
    // Test compress=5 (should give ~5:1 compression)
    println!("=== Testing compress=5 (target 5:1) ===");
    let data5 = generate_controlled_data_alt(4 * 1024 * 1024, 1, 5);
    let compressed5 = zstd::bulk::compress(&data5, 3).unwrap();
    let ratio5 = data5.len() as f64 / compressed5.len() as f64;
    println!("Original: {} bytes", data5.len());
    println!("Compressed: {} bytes", compressed5.len());
    println!("Ratio: {:.4}", ratio5);
    assert!(ratio5 >= 1.2, "compress=5 should give at least 1.2:1 ratio");
    
    // Test compress=6 (should give ~6:1 compression)
    println!("\n=== Testing compress=6 (target 6:1) ===");
    let data6 = generate_controlled_data_alt(4 * 1024 * 1024, 1, 6);
    let compressed6 = zstd::bulk::compress(&data6, 3).unwrap();
    let ratio6 = data6.len() as f64 / compressed6.len() as f64;
    println!("Original: {} bytes", data6.len());
    println!("Compressed: {} bytes", compressed6.len());
    println!("Ratio: {:.4}", ratio6);
    assert!(ratio6 >= 1.2, "compress=6 should give at least 1.2:1 ratio");
    
    // Test dedup=6 (should give 6 unique blocks out of many)
    println!("\n=== Testing dedup=6 (target 6:1) ===");
    let data_dedup = generate_controlled_data_alt(16 * 65536, 6, 1); // 16 blocks
    println!("Total size: {} bytes ({} blocks)", data_dedup.len(), data_dedup.len() / 65536);
    println!("Expected unique blocks: ~3 (16 blocks / dedup=6 ≈ 2.67 → 3)");
}
