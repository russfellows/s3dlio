// examples/s3_backend_comparison.rs
//
// Compare performance and functionality between native AWS SDK and Apache Arrow backends
// for real S3 operations using the configured S3 endpoint and credentials.

use anyhow::Result;
use std::env;
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    // Get S3 configuration from environment
    let bucket = env::var("S3_BUCKET").unwrap_or_else(|_| "my-bucket2".to_string());
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let endpoint = env::var("AWS_ENDPOINT_URL").ok();
    
    println!("🔧 S3 Configuration:");
    println!("   Bucket: {}", bucket);
    println!("   Region: {}", region);
    println!("   Endpoint: {:?}", endpoint);
    println!();

    // Test data sizes for comparison
    let test_sizes = vec![
        ("1KB", vec![0u8; 1024]),
        ("10KB", vec![1u8; 10 * 1024]), 
        ("100KB", vec![2u8; 100 * 1024]),
        ("1MB", vec![3u8; 1024 * 1024]),
    ];

    println!("🚀 Starting S3 Backend Comparison");
    println!("{}", "=".repeat(60));

    for (size_name, test_data) in &test_sizes {
        println!("\n📊 Testing with {} data ({} bytes)", size_name, test_data.len());
        println!("{}", "-".repeat(40));

        // Test native backend
        let native_result = test_native_backend(&bucket, size_name, test_data).await;
        
        // Small delay between tests
        sleep(Duration::from_millis(500)).await;
        
        // Test Arrow backend 
        let arrow_result = test_arrow_backend(&bucket, size_name, test_data).await;

        // Compare results
        match (native_result, arrow_result) {
            (Ok(native_stats), Ok(arrow_stats)) => {
                println!("\n📈 Performance Comparison for {}:", size_name);
                println!("   Native AWS SDK: {:.2}ms write, {:.2}ms read", 
                         native_stats.write_time_ms, native_stats.read_time_ms);
                println!("   Apache Arrow:   {:.2}ms write, {:.2}ms read",
                         arrow_stats.write_time_ms, arrow_stats.read_time_ms);
                
                let write_speedup = native_stats.write_time_ms / arrow_stats.write_time_ms;
                let read_speedup = native_stats.read_time_ms / arrow_stats.read_time_ms;
                
                println!("   Arrow Speedup:  {:.2}x write, {:.2}x read", write_speedup, read_speedup);
                
                if native_stats.data_matches && arrow_stats.data_matches {
                    println!("   ✅ Data integrity: Both backends returned correct data");
                } else {
                    println!("   ❌ Data integrity: Mismatch detected!");
                }
            }
            (Err(native_err), Ok(_)) => {
                println!("   ❌ Native backend failed: {}", native_err);
                println!("   ✅ Arrow backend succeeded");
            }
            (Ok(_), Err(arrow_err)) => {
                println!("   ✅ Native backend succeeded");
                println!("   ❌ Arrow backend failed: {}", arrow_err);
            }
            (Err(native_err), Err(arrow_err)) => {
                println!("   ❌ Both backends failed:");
                println!("      Native: {}", native_err);
                println!("      Arrow:  {}", arrow_err);
            }
        }
        
        println!();
    }

    println!("🏁 S3 Backend Comparison Complete!");
    Ok(())
}

#[derive(Debug)]
struct TestStats {
    write_time_ms: f64,
    read_time_ms: f64,
    data_matches: bool,
}

#[cfg(feature = "native-backends")]
async fn test_native_backend(bucket: &str, size_name: &str, test_data: &[u8]) -> Result<TestStats> {
    use s3dlio::api::store_for_uri;
    
    let s3_uri = format!("s3://{}/native_test_{}.dat", bucket, size_name.to_lowercase());
    let store = store_for_uri(&s3_uri)?;
    
    // Write test
    let write_start = Instant::now();
    store.put(&s3_uri, test_data).await?;
    let write_time_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    
    // Read test  
    let read_start = Instant::now();
    let read_data = store.get(&s3_uri).await?;
    let read_time_ms = read_start.elapsed().as_secs_f64() * 1000.0;
    
    // Verify data integrity
    let data_matches = read_data == test_data;
    
    // Cleanup
    let _ = store.delete(&s3_uri).await;
    
    Ok(TestStats {
        write_time_ms,
        read_time_ms, 
        data_matches,
    })
}

#[cfg(not(feature = "native-backends"))]
async fn test_native_backend(_bucket: &str, _size_name: &str, _test_data: &[u8]) -> Result<TestStats> {
    Err(anyhow::anyhow!("Native backend not compiled"))
}

#[cfg(feature = "arrow-backend")]
async fn test_arrow_backend(bucket: &str, size_name: &str, test_data: &[u8]) -> Result<TestStats> {
    use s3dlio::api::store_for_uri;
    
    let s3_uri = format!("s3://{}/arrow_test_{}.dat", bucket, size_name.to_lowercase());
    let store = store_for_uri(&s3_uri)?;
    
    // Write test
    let write_start = Instant::now();
    store.put(&s3_uri, test_data).await?;
    let write_time_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    
    // Read test
    let read_start = Instant::now();
    let read_data = store.get(&s3_uri).await?;
    let read_time_ms = read_start.elapsed().as_secs_f64() * 1000.0;
    
    // Verify data integrity
    let data_matches = read_data == test_data;
    
    // Cleanup
    let _ = store.delete(&s3_uri).await;
    
    Ok(TestStats {
        write_time_ms,
        read_time_ms,
        data_matches,
    })
}

#[cfg(not(feature = "arrow-backend"))]
async fn test_arrow_backend(_bucket: &str, _size_name: &str, _test_data: &[u8]) -> Result<TestStats> {
    Err(anyhow::anyhow!("Arrow backend not compiled"))
}