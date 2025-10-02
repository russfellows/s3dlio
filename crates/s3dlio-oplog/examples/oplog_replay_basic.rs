// Example: Basic usage of s3dlio-oplog replay functionality
//
// This demonstrates how to use the shared s3dlio-oplog crate to replay
// a captured workload with timing preservation and backend retargeting.

use anyhow::Result;
use s3dlio_oplog::{OpType, ReplayConfig, replay_with_s3dlio};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("s3dlio-oplog Replay Example");
    println!("===========================\n");

    // Example 1: Basic replay with default s3dlio executor
    println!("Example 1: Basic Replay");
    println!("-----------------------");
    
    let config = ReplayConfig {
        op_log_path: PathBuf::from("workload.tsv.zst"),
        target_uri: Some("s3://test-bucket/data/".to_string()),
        speed: 1.0,  // Real-time speed
        continue_on_error: false,
        filter_ops: None,
    };

    match replay_with_s3dlio(config).await {
        Ok(()) => println!("✓ Replay completed successfully\n"),
        Err(e) => println!("✗ Replay failed: {}\n", e),
    }

    // Example 2: Replay with speed multiplier
    println!("Example 2: Fast Replay (10x speed)");
    println!("-----------------------------------");
    
    let config = ReplayConfig {
        op_log_path: PathBuf::from("workload.tsv.zst"),
        target_uri: Some("s3://test-bucket/data/".to_string()),
        speed: 10.0,  // 10x faster
        continue_on_error: true,  // Don't stop on errors
        filter_ops: None,
    };

    match replay_with_s3dlio(config).await {
        Ok(()) => println!("✓ Fast replay completed\n"),
        Err(e) => println!("✗ Fast replay failed: {}\n", e),
    }

    // Example 3: Replay only GET operations
    println!("Example 3: GET-only Replay");
    println!("--------------------------");
    
    let config = ReplayConfig {
        op_log_path: PathBuf::from("workload.tsv.zst"),
        target_uri: Some("s3://test-bucket/data/".to_string()),
        speed: 1.0,
        continue_on_error: false,
        filter_ops: Some(vec![OpType::GET]),  // Only replay GET operations
    };

    match replay_with_s3dlio(config).await {
        Ok(()) => println!("✓ GET-only replay completed\n"),
        Err(e) => println!("✗ GET-only replay failed: {}\n", e),
    }

    // Example 4: Backend retargeting (S3 to Azure)
    println!("Example 4: Backend Retargeting (S3 → Azure)");
    println!("--------------------------------------------");
    
    let config = ReplayConfig {
        op_log_path: PathBuf::from("s3_workload.tsv.zst"),
        target_uri: Some("az://container/data/".to_string()),  // Retarget to Azure
        speed: 1.0,
        continue_on_error: true,
        filter_ops: None,
    };

    match replay_with_s3dlio(config).await {
        Ok(()) => println!("✓ Retargeted replay completed\n"),
        Err(e) => println!("✗ Retargeted replay failed: {}\n", e),
    }

    println!("\nAll examples completed!");
    println!("\nFor more advanced usage (custom executors, URI translation),");
    println!("see docs/S3DLIO_OPLOG_INTEGRATION.md");

    Ok(())
}
