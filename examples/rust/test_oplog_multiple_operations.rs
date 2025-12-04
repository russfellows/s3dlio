// Test program to generate multiple op-log entries for file:// and direct:// backends
use anyhow::Result;
use s3dlio::{init_op_logger, store_for_uri_with_logger, global_logger, finalize_op_logger};
use s3dlio::s3_logger::Logger;

async fn test_backend(backend: &str, base_path: &str, logger: Option<Logger>) -> Result<()> {
    println!("\n========================================");
    println!("Testing {} backend", backend);
    println!("========================================");
    
    let uri_prefix = format!("{}://{}", backend, base_path);
    
    println!("Creating test directory with files...");
    std::fs::create_dir_all(base_path)?;
    for i in 1..=20 {
        std::fs::write(
            format!("{}/file_{:03}.txt", base_path, i),
            format!("Test data for file {}", i)
        )?;
    }
    
    println!("Performing 30 LIST operations...");
    let store = store_for_uri_with_logger(&format!("{}/", uri_prefix), logger.clone())?;
    
    for i in 1..=30 {
        let _keys = store.list(&format!("{}/", uri_prefix), false).await?;
        if i % 10 == 0 {
            println!("  Completed {} LIST operations", i);
        }
    }
    
    println!("Performing 15 GET operations...");
    for i in 1..=15 {
        let uri = format!("{}/file_{:03}.txt", uri_prefix, i);
        let _data = store.get(&uri).await?;
        if i % 5 == 0 {
            println!("  Completed {} GET operations", i);
        }
    }
    
    println!("Performing 10 PUT operations...");
    for i in 21..=30 {
        let uri = format!("{}/new_file_{:03}.txt", uri_prefix, i);
        let data = format!("New test data for file {}", i);
        store.put(&uri, data.as_bytes()).await?;
        if (i - 20) % 5 == 0 {
            println!("  Completed {} PUT operations", i - 20);
        }
    }
    
    println!("✓ {} backend test completed", backend);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let log_file = "/tmp/test_op_log_multi_backend.tsv.zst";
    
    println!("========================================");
    println!("Op-Log Multi-Backend Test");
    println!("========================================");
    println!("Log file: {}", log_file);
    
    // Initialize logger once for all tests
    init_op_logger(log_file)?;
    let logger = global_logger();
    
    // Test file:// backend
    test_backend("file", "/tmp/s3dlio_test_rust_file", logger.clone()).await?;
    
    // Test direct:// backend
    test_backend("direct", "/tmp/s3dlio_test_rust_direct", logger.clone()).await?;
    
    println!("Finalizing logger...");
    finalize_op_logger();
    
    // Analyze the log
    let output = std::process::Command::new("zstd")
        .args(&["-dc", log_file])
        .output()?;
    
    let log_content = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = log_content.lines().collect();
    
    println!("\n========================================");
    println!("Op-log Analysis");
    println!("========================================");
    println!("Total operations logged: {}", lines.len() - 1); // Subtract header
    println!("\nOperation breakdown:");
    println!("  LIST: {}", log_content.matches("LIST").count());
    println!("  GET: {}", log_content.matches("\tGET\t").count());
    println!("  PUT: {}", log_content.matches("\tPUT\t").count());
    
    // Count by backend
    let file_ops = log_content.matches("file://").count();
    let direct_ops = log_content.matches("direct://").count();
    println!("\nOperations by backend:");
    println!("  file://   : {}", file_ops);
    println!("  direct:// : {}", direct_ops);
    
    println!("\nFirst 25 log entries:");
    for line in lines.iter().take(26) {
        println!("{}", line);
    }
    
    println!("\n... (middle entries truncated)");
    println!("\nLast 15 log entries:");
    for line in lines.iter().skip(lines.len().saturating_sub(15)) {
        println!("{}", line);
    }
    
    println!("\n========================================");
    println!("✓ Test completed successfully");
    println!("========================================");
    println!("\nLog file preserved at: {}", log_file);
    println!("View with: zstd -dc {} | less", log_file);
    println!("Or: zstd -dc {} | head -50", log_file);
    
    // Cleanup test directories (but keep log file)
    std::fs::remove_dir_all("/tmp/s3dlio_test_rust_file").ok();
    std::fs::remove_dir_all("/tmp/s3dlio_test_rust_direct").ok();
    
    Ok(())
}
