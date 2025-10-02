//! Example demonstrating OpLogStreamReader for memory-efficient op-log processing
//!
//! This example shows how to use OpLogStreamReader to process large operation logs
//! with constant memory usage (~1.5 MB) regardless of file size.

use anyhow::Result;
use s3dlio_oplog::{OpLogStreamReader, OpType};

fn main() -> Result<()> {
    // Example 1: Basic streaming
    println!("=== Example 1: Basic Streaming ===");
    basic_streaming()?;
    
    // Example 2: Environment variable tuning
    println!("\n=== Example 2: Environment Tuning ===");
    env_tuning_example();
    
    // Example 3: Iterator chaining
    println!("\n=== Example 3: Iterator Chaining ===");
    iterator_chaining()?;
    
    // Example 4: Error handling
    println!("\n=== Example 4: Error Handling ===");
    error_handling_example()?;
    
    Ok(())
}

fn basic_streaming() -> Result<()> {
    // Create example op-log file
    let (example_tsv, _temp) = create_example_oplog()?;
    
    println!("Reading op-log with streaming (constant memory)...");
    let stream = OpLogStreamReader::from_file(&example_tsv)?;
    
    let mut count = 0;
    for entry_result in stream {
        let entry = entry_result?;
        count += 1;
        if count <= 3 {
            println!("  Op {}: {:?} - {} ({} bytes)", 
                count, entry.op, entry.file, entry.bytes);
        }
    }
    println!("Total operations processed: {}", count);
    
    Ok(())
}

fn env_tuning_example() {
    println!("Set these environment variables to tune streaming performance:");
    println!("  S3DLIO_OPLOG_READ_BUF=2048      # Increase channel buffer (default: 1024)");
    println!("  S3DLIO_OPLOG_CHUNK_SIZE=2097152 # Use 2 MB chunks (default: 1 MB)");
    println!();
    println!("Example:");
    println!("  export S3DLIO_OPLOG_READ_BUF=2048");
    println!("  export S3DLIO_OPLOG_CHUNK_SIZE=2097152");
    println!("  cargo run --example oplog_streaming_demo");
}

fn iterator_chaining() -> Result<()> {
    let (example_tsv, _temp) = create_example_oplog()?;
    
    println!("Using iterator methods (take, filter, map)...");
    let stream = OpLogStreamReader::from_file(&example_tsv)?;
    
    // Take first 5 entries, filter GETs only
    let get_ops: Vec<_> = stream
        .take(5)
        .filter_map(|r| r.ok())
        .filter(|entry| entry.op == OpType::GET)
        .collect();
    
    println!("First 5 GET operations:");
    for (i, entry) in get_ops.iter().enumerate() {
        println!("  {}: {} ({} bytes)", i + 1, entry.file, entry.bytes);
    }
    
    Ok(())
}

fn error_handling_example() -> Result<()> {
    let (example_tsv, _temp) = create_example_oplog()?;
    
    println!("Streaming with error handling...");
    let stream = OpLogStreamReader::from_file(&example_tsv)?;
    
    let mut success = 0;
    let mut errors = 0;
    
    for entry_result in stream {
        match entry_result {
            Ok(entry) => {
                success += 1;
                // Process entry
                let _ = entry;
            }
            Err(e) => {
                errors += 1;
                eprintln!("Error parsing entry: {}", e);
            }
        }
    }
    
    println!("Processed {} successful entries, {} errors", success, errors);
    
    Ok(())
}

fn create_example_oplog() -> Result<(std::path::PathBuf, tempfile::NamedTempFile)> {
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(file, "operation\tfile\tbytes\tstart")?;
    writeln!(file, "GET\tfile1.dat\t1024\t2025-01-01T00:00:00Z")?;
    writeln!(file, "PUT\tfile2.dat\t2048\t2025-01-01T00:00:01Z")?;
    writeln!(file, "GET\tfile3.dat\t4096\t2025-01-01T00:00:02Z")?;
    writeln!(file, "GET\tfile4.dat\t8192\t2025-01-01T00:00:03Z")?;
    writeln!(file, "PUT\tfile5.dat\t16384\t2025-01-01T00:00:04Z")?;
    file.flush()?;
    
    let path = file.path().to_path_buf();
    Ok((path, file))
}
