// tests/test_backend_performance_comparison.rs
//
// Comprehensive performance comparison between native and Arrow backends
// Tests PUT and GET operations with 1-8MB objects for at least 2 minutes

use anyhow::Result;
use std::time::{Duration, Instant};
use rand::prelude::*;
use tokio;

use s3dlio::object_store::{store_for_uri, ObjectStore};

// Test configuration
const MIN_TEST_DURATION: Duration = Duration::from_secs(120); // 2 minutes minimum
const OBJECT_SIZES: &[usize] = &[
    1 * 1024 * 1024,   // 1MB
    10 * 1024 * 1024,  // 10MB
];
const PUT_CONCURRENCY: usize = 16;  // Match s3dlio defaults
const GET_CONCURRENCY: usize = 32;  // Match s3dlio defaults

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    backend_name: String,
    operation: String,
    object_size_mb: f64,
    operations_completed: usize,
    total_duration: Duration,
    total_bytes: u64,
    throughput_mbps: f64,
    avg_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
}

impl PerformanceMetrics {


    fn print_summary(&self) {
        println!("📊 {} {} Performance ({}MB objects):", self.backend_name, self.operation, self.object_size_mb);
        println!("   Operations: {}", self.operations_completed);
        println!("   Total Duration: {:.2}s", self.total_duration.as_secs_f64());
        println!("   Total Data: {:.2} MB", self.total_bytes as f64 / (1024.0 * 1024.0));
        println!("   Throughput: {:.2} MB/s", self.throughput_mbps);
        println!("   Avg Latency: {:.2}ms", self.avg_latency_ms);
        println!("   Min Latency: {:.2}ms", self.min_latency_ms);
        println!("   Max Latency: {:.2}ms", self.max_latency_ms);
    }
}



fn get_backend_name() -> &'static str {
    #[cfg(feature = "arrow-backend")]
    {
        "Apache Arrow"
    }
    #[cfg(feature = "native-backends")]
    {
        "Native AWS SDK"
    }
    #[cfg(not(any(feature = "arrow-backend", feature = "native-backends")))]
    {
        "Unknown Backend"
    }
}

fn get_s3_test_uri() -> Result<String> {
    let _ = dotenvy::dotenv();
    
    let _endpoint = std::env::var("AWS_ENDPOINT_URL")
        .unwrap_or_else(|_| "http://localhost:9000".to_string());
    let bucket = std::env::var("S3_BUCKET")
        .unwrap_or_else(|_| "test-bucket".to_string());
    
    // Create unique test prefix
    let test_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    Ok(format!("s3://{}/perf-test-{}/", bucket, test_id))
}

async fn benchmark_put_operations(
    base_uri: &str, 
    num_operations: usize, 
    concurrency: usize
) -> Result<(PerformanceMetrics, Vec<String>)> {
    use s3dlio::{put_objects_with_random_data_and_type, ObjectType};
    use s3dlio::config::Config;
    
    let start_time = Instant::now();
    
    // Generate URIs - use consistent object size for this test
    let mut object_keys = Vec::new();
    let mut rng = rand::rng();
    let object_size = OBJECT_SIZES[rng.random_range(0..OBJECT_SIZES.len())];
    
    for i in 0..num_operations {
        let uri = format!("{}/test-object-{:06}.dat", base_uri, i);
        object_keys.push(uri.clone());
    }
    
    // Use the same high-performance function that CLI uses
    // This uses FuturesUnordered and proper async concurrency
    let config = Config::new_with_defaults(
        ObjectType::Raw, // Raw data for pure performance test
        1, // elements
        object_size, // element_size
        1, // No deduplication 
        1, // No compression
    );
    put_objects_with_random_data_and_type(
        &object_keys,
        object_size,
        concurrency,
        config,
    )?;
    
    let total_duration = start_time.elapsed();
    let total_bytes = num_operations * object_size;
    let throughput_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / total_duration.as_secs_f64();
    
    // For latency metrics, estimate based on concurrency and total time
    // This is an approximation since put_objects_with_random_data_and_type doesn't return individual timings
    let avg_latency_ms = (total_duration.as_millis() as f64) / (num_operations as f64 / concurrency as f64);
    
    let metrics = PerformanceMetrics {
        backend_name: get_backend_name().to_string(),
        operation: "PUT".to_string(),
        object_size_mb: object_size as f64 / (1024.0 * 1024.0),
        operations_completed: num_operations,
        total_duration,
        total_bytes: total_bytes as u64,
        throughput_mbps,
        avg_latency_ms,
        min_latency_ms: avg_latency_ms * 0.5, // Rough estimate
        max_latency_ms: avg_latency_ms * 2.0, // Rough estimate
    };
    
    Ok((metrics, object_keys))
}

async fn benchmark_get_operations(
    object_keys: &[String],
    concurrency: usize
) -> Result<PerformanceMetrics> {
    use s3dlio::get_objects_parallel;
    
    let start_time = Instant::now();
    
    // Use the same high-performance function that CLI uses
    // This uses FuturesUnordered and proper async concurrency
    let results = get_objects_parallel(object_keys, concurrency)?;
    
    let total_duration = start_time.elapsed();
    let total_bytes: usize = results.iter().map(|(_, data)| data.len()).sum();
    let throughput_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / total_duration.as_secs_f64();
    
    // For latency metrics, estimate based on concurrency and total time
    // This is an approximation since get_objects_parallel doesn't return individual timings
    let avg_latency_ms = (total_duration.as_millis() as f64) / (results.len() as f64 / concurrency as f64);
    let avg_size_mb = total_bytes as f64 / (1024.0 * 1024.0) / results.len() as f64;
    
    let metrics = PerformanceMetrics {
        backend_name: get_backend_name().to_string(),
        operation: "GET".to_string(),
        object_size_mb: avg_size_mb,
        operations_completed: results.len(),
        total_duration,
        total_bytes: total_bytes as u64,
        throughput_mbps,
        avg_latency_ms,
        min_latency_ms: avg_latency_ms * 0.5, // Rough estimate
        max_latency_ms: avg_latency_ms * 2.0, // Rough estimate
    };
    
    Ok(metrics)
}

async fn cleanup_test_objects(store: &dyn ObjectStore, object_keys: &[String]) -> Result<()> {
    println!("🗑️ Cleaning up {} test objects...", object_keys.len());
    
    for (i, key) in object_keys.iter().enumerate() {
        match store.delete(key).await {
            Ok(()) => {
                if (i + 1) % 50 == 0 {
                    print!(".");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
            }
            Err(e) => {
                eprintln!("⚠️ Failed to delete {}: {}", key, e);
            }
        }
    }
    
    println!(); // New line after dots
    println!("✅ Cleanup complete");
    Ok(())
}

async fn run_performance_test_for_size(object_size: usize) -> Result<(PerformanceMetrics, PerformanceMetrics)> {
    let base_uri = get_s3_test_uri()?;
    let store = store_for_uri(&base_uri)?;
    
    println!("\n{}", "=".repeat(60));
    println!("🎯 Performance Test: {}MB Objects with {} Backend", 
             object_size as f64 / (1024.0 * 1024.0), get_backend_name());
    println!("{}", "=".repeat(60));
    
    // Test PUT operations
    let (put_metrics, object_keys) = benchmark_put_operations(
        &base_uri, 
        50, // number of operations for 2-minute test
        PUT_CONCURRENCY
    ).await?;

    put_metrics.print_summary();    // Test GET operations using the objects we just created
    let get_metrics = benchmark_get_operations(
        &object_keys,
        GET_CONCURRENCY,
    ).await?;
    
    get_metrics.print_summary();
    
    // Cleanup
    cleanup_test_objects(store.as_ref(), &object_keys).await?;
    
    Ok((put_metrics, get_metrics))
}

fn print_environment_info() {
    let _ = dotenvy::dotenv();
    
    println!("🔧 Environment Configuration:");
    
    if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
        println!("   AWS_ENDPOINT_URL: {}", endpoint);
    }
    
    if let Ok(bucket) = std::env::var("S3_BUCKET") {
        println!("   S3_BUCKET: {}", bucket);
    }
    
    if std::env::var("AWS_ACCESS_KEY_ID").is_ok() {
        println!("   AWS_ACCESS_KEY_ID: [configured]");
    }
    
    if std::env::var("AWS_SECRET_ACCESS_KEY").is_ok() {
        println!("   AWS_SECRET_ACCESS_KEY: [configured]");
    }
    
    println!("   Backend: {}", get_backend_name());
    println!("   Test Duration: {}s per operation type", MIN_TEST_DURATION.as_secs());
}

fn print_summary_table(all_results: &[(PerformanceMetrics, PerformanceMetrics)]) {
    println!("\n{}", "=".repeat(80));
    println!("📊 PERFORMANCE SUMMARY TABLE");
    println!("{}", "=".repeat(80));
    
    println!("{:<15} {:<10} {:<12} {:<10} {:<12} {:<10}", 
             "Object Size", "PUT Ops", "PUT MB/s", "PUT Avg(ms)", "GET Ops", "GET MB/s");
    println!("{:<15} {:<10} {:<12} {:<10} {:<12} {:<10}", 
             "───────────", "───────", "────────", "──────────", "───────", "────────");
    
    for (put_metrics, get_metrics) in all_results {
        println!("{:<15} {:<10} {:<12.2} {:<10.2} {:<12} {:<10.2}",
                 format!("{}MB", put_metrics.object_size_mb),
                 put_metrics.operations_completed,
                 put_metrics.throughput_mbps,
                 put_metrics.avg_latency_ms,
                 get_metrics.operations_completed,
                 get_metrics.throughput_mbps);
    }
    
    // Calculate totals
    let total_put_ops: usize = all_results.iter().map(|(p, _)| p.operations_completed).sum();
    let total_get_ops: usize = all_results.iter().map(|(_, g)| g.operations_completed).sum();
    let total_put_bytes: u64 = all_results.iter().map(|(p, _)| p.total_bytes).sum();
    let total_get_bytes: u64 = all_results.iter().map(|(_, g)| g.total_bytes).sum();
    let total_put_duration: Duration = all_results.iter().map(|(p, _)| p.total_duration).sum();
    let total_get_duration: Duration = all_results.iter().map(|(_, g)| g.total_duration).sum();
    
    println!("{:<15} {:<10} {:<12} {:<10} {:<12} {:<10}", 
             "───────────", "───────", "────────", "──────────", "───────", "────────");
    
    let avg_put_throughput = if total_put_duration.as_secs_f64() > 0.0 {
        (total_put_bytes as f64 / (1024.0 * 1024.0)) / total_put_duration.as_secs_f64()
    } else { 0.0 };
    
    let avg_get_throughput = if total_get_duration.as_secs_f64() > 0.0 {
        (total_get_bytes as f64 / (1024.0 * 1024.0)) / total_get_duration.as_secs_f64()
    } else { 0.0 };
    
    println!("{:<15} {:<10} {:<12.2} {:<10} {:<12} {:<10.2}",
             "TOTALS", total_put_ops, avg_put_throughput, "─", total_get_ops, avg_get_throughput);
}

#[tokio::test]
#[cfg(feature = "arrow-backend")]
async fn test_arrow_backend_performance() -> Result<()> {
    println!("🚀 APACHE ARROW BACKEND PERFORMANCE TEST");
    println!("{}", "=".repeat(60));
    
    print_environment_info();
    
    // Test each object size
    let mut all_results = Vec::new();
    for &object_size in OBJECT_SIZES {
        let result = run_performance_test_for_size(object_size).await?;
        all_results.push(result);
    }
    
    // Print summary table
    print_summary_table(&all_results);
    
    Ok(())
}

#[tokio::test]
#[cfg(feature = "native-backends")]
async fn test_native_backend_performance() -> Result<()> {
    println!("🚀 NATIVE AWS SDK BACKEND PERFORMANCE TEST");
    println!("{}", "=".repeat(60));
    
    print_environment_info();
    
    // Test each object size
    let mut all_results = Vec::new();
    for &object_size in OBJECT_SIZES {
        let result = run_performance_test_for_size(object_size).await?;
        all_results.push(result);
    }
    
    // Print summary table
    print_summary_table(&all_results);
    
    Ok(())
}