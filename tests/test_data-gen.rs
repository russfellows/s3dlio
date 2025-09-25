// tests/test_data-gen.rs
//
// Test cases to test data generation
//

use s3dlio::data_gen::generate_controlled_data;


#[cfg(test)]
mod tests {
    use super::generate_controlled_data;
    use s3dlio::constants::BLK_SIZE;
    use std::io::{self, Write};

    // How close our dedupe and compression ratios must be, set to 0.10 = 10%
    const TOLERANCE: f64 = 0.10;

    /// Check if we can even use our function 
    #[test]
    fn test_generate_controlled_data() {
        let size = 1024;
        let dedup = 3;
        let compress = 2;

        let data = generate_controlled_data(size, dedup, compress);

        assert_eq!(data.len(), size, "Output size must match requested size. Requested {} bytes but got {}", size, data.len());
    }

    /// Check if our min buffer size of BLK_SIZE is enforced 
    #[test]
    fn test_min_size_enforcement() {
        let size = 256; // Smaller than BLK_SIZE
        let dedup = 1;
        let compress = 1;

        let data = generate_controlled_data(size, dedup, compress);
        assert!(data.len() >= BLK_SIZE, "Minimum size of {} should be enforced. Asked for {} bytes, returned {} bytes", BLK_SIZE, size, data.len());
    }

    /// Check if one file gets created with random data 
    /// We need to check manually, so just create the file and examine offline
    #[test]
    fn test_write_random_data_to_file() {
        // Step 1: Generate random data (using your existing function)
        //let size = 4096;
        let size = 8 * BLK_SIZE;
        let dedup = 3;
        let compress = 2;
        let output_dir = "/tmp";

        let data = generate_controlled_data(size, dedup, compress);
    
        // Step 2: Define the output file path
        let filename = format!("{output_dir}/random_data_{}.bin", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        std::fs::create_dir_all(output_dir).expect("Failed to create test directory");
    
        // Step 3: Write data to the file
        let mut file = std::fs::File::create(&filename).expect("Failed to create file");
        file.write_all(&data).expect("Failed to write data");
    
        // Step 4: Print the filename for offline inspection
        println!("Generated file: {}", filename);
    }

    /// Check if multiple files seem to have different data, with the correct bits changed 
    /// We need to check manually, so just create the files and examine offline
    #[test]
    fn test_create_multiple_files() {
        // Step 1: Generate random data (using your existing function)
        //let size = 4096;
        let size = 128 * BLK_SIZE;
        let dedup = 3;
        let compress = 2;
        let output_dir = "/tmp";

        // Step 2: Define the output file path
        // Create data in loop
        for i in 0..4 {
            // Generate random data
            let data = generate_controlled_data(size, dedup, compress);

            // Create unique filename with index
            let file_name = format!("{}/random_data_{}.bin", output_dir, i);

            // Write to file (ensure the directory exists)
            std::fs::write(&file_name, &data).expect("Failed to write file");

            // Print filename for verification
            println!("Created: {}", file_name);
        }
    }

    /// Check if dedupe and compression ratios seem to work
    #[test]
    fn test_data_dedup_compress1() {
        let size = 8192 * BLK_SIZE;
        let dedup = 2;
        let compress = 1;

        let data = generate_controlled_data(size, dedup, compress);

        let num_blocks = (data.len() + BLK_SIZE - 1) / BLK_SIZE;
        let mut unique_blocks = std::collections::HashSet::new();
        let mut total_constant_bytes = 0;

        for i in 0..num_blocks {
            let start = i * BLK_SIZE;
            let end = std::cmp::min(start + BLK_SIZE, data.len());
            let block = &data[start..end];

            // Count zero bytes
            total_constant_bytes += block.iter().filter(|&&b| b == 0).count();

            // Insert the entire block for true dedup detection
            unique_blocks.insert(block.to_vec());
        }

        let dedup_ratio = unique_blocks.len() as f64 / num_blocks as f64;
        let total_bytes = num_blocks * BLK_SIZE;
        let compression_ratio = total_constant_bytes as f64 / total_bytes as f64;

        let expected_dedup_ratio = 1.0 / dedup as f64;       // ~0.3333
        let expected_compress_ratio = (compress as f64 - 1.0) / compress as f64; // 0.5

        eprintln!("Calculated Dedup Ratio: {}", dedup_ratio);
        eprintln!("Expected Dedup Ratio: {}", expected_dedup_ratio);
        eprintln!("Calculated Compression Ratio: {}", compression_ratio);
        eprintln!("Expected Compression Ratio: {}", expected_compress_ratio);
        // FORCE a flush so the lines hit the terminal immediately
        io::stdout().flush().unwrap();

        // this will panic with full output if it’s out of tolerance
        assert!(
            (dedup_ratio - expected_dedup_ratio).abs() < TOLERANCE &&
            (compression_ratio - expected_compress_ratio).abs() < TOLERANCE,
            "Dedup or compression ratio check failed"
        );

        println!("✅ Test Passed!");
        // FORCE a flush so the lines hit the terminal immediately
        io::stdout().flush().unwrap();
    }

    /// Check if dedupe and compression ratios seem to work
    #[test]
    fn test_data_dedup_compress2() {
        let size = 8192 * BLK_SIZE;
        let dedup = 3;
        let compress = 2;

        let data = generate_controlled_data(size, dedup, compress);

        let num_blocks = (data.len() + BLK_SIZE - 1) / BLK_SIZE;
        let mut unique_blocks = std::collections::HashSet::new();
        let mut total_constant_bytes = 0;

        for i in 0..num_blocks {
            let start = i * BLK_SIZE;
            let end = std::cmp::min(start + BLK_SIZE, data.len());
            let block = &data[start..end];

            // Count zero bytes
            total_constant_bytes += block.iter().filter(|&&b| b == 0).count();

            // Insert the entire block for true dedup detection
            unique_blocks.insert(block.to_vec());
        }

        let dedup_ratio = unique_blocks.len() as f64 / num_blocks as f64;
        let total_bytes = num_blocks * BLK_SIZE;
        let compression_ratio = total_constant_bytes as f64 / total_bytes as f64;

        let expected_dedup_ratio = 1.0 / dedup as f64;       // ~0.3333
        let expected_compress_ratio = (compress as f64 - 1.0) / compress as f64; // 0.5

        eprintln!("Calculated Dedup Ratio: {}", dedup_ratio);
        eprintln!("Expected Dedup Ratio: {}", expected_dedup_ratio);
        eprintln!("Calculated Compression Ratio: {}", compression_ratio);
        eprintln!("Expected Compression Ratio: {}", expected_compress_ratio);
        // FORCE a flush so the lines hit the terminal immediately
        io::stdout().flush().unwrap();

        // this will panic with full output if it’s out of tolerance
        assert!(
            (dedup_ratio - expected_dedup_ratio).abs() < TOLERANCE &&
            (compression_ratio - expected_compress_ratio).abs() < TOLERANCE,
            "Dedup or compression ratio check failed"
        );

        println!("✅ Test Passed!");
        // FORCE a flush so the lines hit the terminal immediately
        io::stdout().flush().unwrap();
    }

// End of tests
}
