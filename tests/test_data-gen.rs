// tests/test_data-gen.rs
//
// Tests for data_gen_alt module - the ONLY supported data generation API
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

// Use the new data_gen_alt API directly - NO deprecated functions
use s3dlio::data_gen_alt::generate_data_simple;
use s3dlio::constants::DGEN_BLOCK_SIZE;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};

    // How close our dedupe and compression ratios must be, set to 0.15 = 15%
    const TOLERANCE: f64 = 0.15;

    /// Check that min buffer size of DGEN_BLOCK_SIZE (1 MiB) is enforced
    /// The data_gen_alt algorithm uses 1 MiB minimum block size
    #[test]
    fn test_min_size_enforcement() {
        let size = 1024;  // Request less than DGEN_BLOCK_SIZE
        let dedup = 1;
        let compress = 1;

        let data = generate_data_simple(size, dedup, compress);

        // Algorithm enforces minimum of DGEN_BLOCK_SIZE (1 MiB)
        assert!(data.len() >= DGEN_BLOCK_SIZE, 
            "Minimum size of {} should be enforced. Requested {} bytes but got {}", 
            DGEN_BLOCK_SIZE, size, data.len());
    }

    /// Check if one file gets created with random data 
    /// We need to check manually, so just create the file and examine offline
    #[test]
    fn test_write_random_data_to_file() {
        let size = 8 * DGEN_BLOCK_SIZE;  // 8 MiB
        let dedup = 3;
        let compress = 2;
        let output_dir = "/tmp";

        let data = generate_data_simple(size, dedup, compress);
    
        let filename = format!("{output_dir}/random_data_{}.bin", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        std::fs::create_dir_all(output_dir).expect("Failed to create test directory");
    
        let mut file = std::fs::File::create(&filename).expect("Failed to create file");
        file.write_all(data.as_slice()).expect("Failed to write data");
    
        println!("Generated file: {}", filename);
    }

    /// Check if multiple files seem to have different data (unique per call)
    #[test]
    fn test_create_multiple_files() {
        let size = 4 * DGEN_BLOCK_SIZE;  // 4 MiB
        let dedup = 3;
        let compress = 2;
        let output_dir = "/tmp";

        for i in 0..4 {
            let data = generate_data_simple(size, dedup, compress);
            let file_name = format!("{}/random_data_{}.bin", output_dir, i);
            std::fs::write(&file_name, data.as_slice()).expect("Failed to write file");
            println!("Created: {}", file_name);
        }
    }

    /// Verify deduplication: dedup=2 means 50% unique blocks
    /// With 64 blocks and dedup=2, we should have exactly 32 unique block patterns
    #[test]
    fn test_data_dedup_compress1() {
        let num_blocks_to_generate = 64;  // 64 MiB total
        let size = num_blocks_to_generate * DGEN_BLOCK_SIZE;
        let dedup = 2;
        let compress = 1;  // No compression

        let data = generate_data_simple(size, dedup, compress);
        let data_slice = data.as_slice();

        let num_blocks = data.len() / DGEN_BLOCK_SIZE;
        let mut unique_blocks = std::collections::HashSet::new();
        let mut total_zero_bytes = 0;

        for i in 0..num_blocks {
            let start = i * DGEN_BLOCK_SIZE;
            let end = start + DGEN_BLOCK_SIZE;
            let block = &data_slice[start..end];

            // Count zero bytes for compression estimation
            total_zero_bytes += block.iter().filter(|&&b| b == 0).count();

            // Insert block for dedup detection
            unique_blocks.insert(block.to_vec());
        }

        let dedup_ratio = unique_blocks.len() as f64 / num_blocks as f64;
        let total_bytes = num_blocks * DGEN_BLOCK_SIZE;
        let compression_ratio = total_zero_bytes as f64 / total_bytes as f64;

        // dedup=2 means 1/2 = 0.5 unique ratio (50% unique blocks)
        let expected_dedup_ratio = 1.0 / dedup as f64;
        // compress=1 means 0% zeros (incompressible)
        let expected_compress_ratio = (compress as f64 - 1.0) / compress as f64;

        eprintln!("Num blocks: {} ({} MiB)", num_blocks, num_blocks);
        eprintln!("Unique blocks: {} (expected {})", unique_blocks.len(), num_blocks / dedup);
        eprintln!("Calculated Dedup Ratio: {:.4}", dedup_ratio);
        eprintln!("Expected Dedup Ratio: {:.4}", expected_dedup_ratio);
        eprintln!("Calculated Compression Ratio: {:.4}", compression_ratio);
        eprintln!("Expected Compression Ratio: {:.4}", expected_compress_ratio);
        io::stdout().flush().unwrap();

        assert!(
            (dedup_ratio - expected_dedup_ratio).abs() < TOLERANCE,
            "Dedup ratio check failed: got {:.4}, expected {:.4} (diff {:.4}, tolerance {:.4})",
            dedup_ratio, expected_dedup_ratio, (dedup_ratio - expected_dedup_ratio).abs(), TOLERANCE
        );

        assert!(
            (compression_ratio - expected_compress_ratio).abs() < TOLERANCE,
            "Compression ratio check failed: got {:.4}, expected {:.4} (diff {:.4})",
            compression_ratio, expected_compress_ratio, (compression_ratio - expected_compress_ratio).abs()
        );

        println!("✅ Test Passed!");
        io::stdout().flush().unwrap();
    }

    /// Verify dedup=3 (33% unique) and compress=2 (50% zeros)
    #[test]
    fn test_data_dedup_compress2() {
        let num_blocks_to_generate = 66;  // 66 MiB (divisible by 3)
        let size = num_blocks_to_generate * DGEN_BLOCK_SIZE;
        let dedup = 3;
        let compress = 2;

        let data = generate_data_simple(size, dedup, compress);
        let data_slice = data.as_slice();

        let num_blocks = data.len() / DGEN_BLOCK_SIZE;
        let mut unique_blocks = std::collections::HashSet::new();
        let mut total_zero_bytes = 0;

        for i in 0..num_blocks {
            let start = i * DGEN_BLOCK_SIZE;
            let end = start + DGEN_BLOCK_SIZE;
            let block = &data_slice[start..end];

            total_zero_bytes += block.iter().filter(|&&b| b == 0).count();
            unique_blocks.insert(block.to_vec());
        }

        let dedup_ratio = unique_blocks.len() as f64 / num_blocks as f64;
        let total_bytes = num_blocks * DGEN_BLOCK_SIZE;
        let compression_ratio = total_zero_bytes as f64 / total_bytes as f64;

        // dedup=3 means 1/3 ≈ 0.333 unique ratio
        let expected_dedup_ratio = 1.0 / dedup as f64;
        // compress=2 means 50% zeros
        let expected_compress_ratio = (compress as f64 - 1.0) / compress as f64;

        eprintln!("Num blocks: {} ({} MiB)", num_blocks, num_blocks);
        eprintln!("Unique blocks: {} (expected {})", unique_blocks.len(), num_blocks / dedup);
        eprintln!("Calculated Dedup Ratio: {:.4}", dedup_ratio);
        eprintln!("Expected Dedup Ratio: {:.4}", expected_dedup_ratio);
        eprintln!("Calculated Compression Ratio: {:.4}", compression_ratio);
        eprintln!("Expected Compression Ratio: {:.4}", expected_compress_ratio);
        io::stdout().flush().unwrap();

        assert!(
            (dedup_ratio - expected_dedup_ratio).abs() < TOLERANCE,
            "Dedup ratio check failed: got {:.4}, expected {:.4} (diff {:.4}, tolerance {:.4})",
            dedup_ratio, expected_dedup_ratio, (dedup_ratio - expected_dedup_ratio).abs(), TOLERANCE
        );

        assert!(
            (compression_ratio - expected_compress_ratio).abs() < TOLERANCE,
            "Compression ratio check failed: got {:.4}, expected {:.4} (diff {:.4})",
            compression_ratio, expected_compress_ratio, (compression_ratio - expected_compress_ratio).abs()
        );

        println!("✅ Test Passed!");
        io::stdout().flush().unwrap();
    }
}
