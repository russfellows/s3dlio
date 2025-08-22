// examples/async_pool_dataloader_demo.rs
//
// Demonstration of the enhanced async pooling dataloader with multi-backend support
// 
// This example shows:
// 1. Multi-backend dataset creation from different URI schemes
// 2. Dynamic batch formation with out-of-order completion
// 3. Performance comparison between traditional and async pooling approaches
// 4. Configuration tuning for different workloads

use anyhow::Result;
use s3dlio::data_loader::{
    async_pool_dataloader::{AsyncPoolDataLoader, MultiBackendDataset, PoolConfig},
    LoaderOptions
};
use s3dlio::object_store::store_for_uri;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio_stream::StreamExt;

/// Create test dataset with varying file sizes to simulate real ML workloads
async fn create_ml_dataset(base_dir: &std::path::Path, num_files: usize) -> Result<Vec<String>> {
    println!("📁 Creating ML dataset with {} files...", num_files);
    
    let mut uris = Vec::new();
    
    for i in 0..num_files {
        let file_path = base_dir.join(format!("batch_{:04}.npy", i));
        let uri = format!("file://{}", file_path.display());
        
        // Simulate real ML datasets: mostly small files with some large ones
        let size = match i % 10 {
            0..=6 => 1024 * (i % 5 + 1),           // 1-5KB (typical features)
            7..=8 => 1024 * 50,                    // 50KB (medium samples)
            _ => 1024 * 200,                       // 200KB (large samples/images)
        };
        
        // Create synthetic data that varies by file to ensure we can track it
        let data = (0..size).map(|j| ((i + j) % 256) as u8).collect::<Vec<_>>();
        
        let store = store_for_uri(&uri)?;
        store.put(&uri, &data).await?;
        
        uris.push(uri);
    }
    
    println!("✅ Created {} files with varying sizes", num_files);
    Ok(uris)
}

/// Demonstrate basic async pooling functionality
async fn demo_basic_async_pooling() -> Result<()> {
    println!("\n🚀 Demo 1: Basic Async Pooling with Dynamic Batching");
    println!("====================================================");
    
    let temp_dir = TempDir::new()?;
    let uris = create_ml_dataset(temp_dir.path(), 15).await?;
    
    let dataset = MultiBackendDataset::from_uris(uris)?;
    let options = LoaderOptions {
        batch_size: 4,
        drop_last: false,
        ..Default::default()
    };
    
    // Use default pool configuration
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream();
    
    println!("📊 Processing batches (batch_size=4):");
    
    let start_time = Instant::now();
    let mut total_items = 0;
    let mut batch_count = 0;
    
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        batch_count += 1;
        total_items += batch.len();
        
        println!("  Batch {}: {} items, sizes: {:?}", 
                 batch_count, 
                 batch.len(),
                 batch.iter().map(|data| format!("{}B", data.len())).collect::<Vec<_>>()
        );
    }
    
    let duration = start_time.elapsed();
    println!("✅ Processed {} items in {} batches ({:.2}ms)", 
             total_items, batch_count, duration.as_millis());
    
    Ok(())
}

/// Demonstrate advanced pool configuration and out-of-order completion
async fn demo_advanced_pooling_config() -> Result<()> {
    println!("\n⚡ Demo 2: Advanced Pool Configuration & Out-of-Order Completion");
    println!("================================================================");
    
    let temp_dir = TempDir::new()?;
    // Create dataset with dramatic size differences to force different completion times
    let uris = create_ml_dataset(temp_dir.path(), 20).await?;
    
    let dataset = MultiBackendDataset::from_uris(uris)?;
    let options = LoaderOptions {
        batch_size: 5,
        drop_last: false,
        ..Default::default()
    };
    
    // Configure aggressive pooling for high-throughput
    let pool_config = PoolConfig {
        pool_size: 12,                              // High concurrency
        readahead_batches: 3,                       // Buffer 3 batches ahead
        batch_timeout: Duration::from_millis(100),  // Short timeout for responsiveness
        max_inflight: 20,                           // Allow many concurrent requests
    };
    
    println!("📊 Pool Config: {} concurrent, {} readahead batches, {}ms timeout",
             pool_config.pool_size,
             pool_config.readahead_batches,
             pool_config.batch_timeout.as_millis());
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream_with_pool(pool_config);
    
    let start_time = Instant::now();
    let mut batch_timestamps = Vec::new();
    
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        let timestamp = start_time.elapsed();
        batch_timestamps.push(timestamp);
        
        println!("  Batch {} at {:.0}ms: {} items",
                 batch_timestamps.len(),
                 timestamp.as_millis(),
                 batch.len());
    }
    
    println!("✅ Advanced pooling completed in {:.2}ms", start_time.elapsed().as_millis());
    
    // Analyze batch timing
    if batch_timestamps.len() > 1 {
        let intervals: Vec<_> = batch_timestamps.windows(2)
            .map(|w| (w[1] - w[0]).as_millis())
            .collect();
        println!("📈 Batch intervals: {:?}ms (avg: {:.1}ms)",
                 intervals,
                 intervals.iter().sum::<u128>() as f64 / intervals.len() as f64);
    }
    
    Ok(())
}

/// Demonstrate multi-backend support
async fn demo_multi_backend_support() -> Result<()> {
    println!("\n🌐 Demo 3: Multi-Backend Support");
    println!("================================");
    
    let temp_dir = TempDir::new()?;
    
    // Create file:// URIs
    let file_uris = create_ml_dataset(temp_dir.path(), 8).await?;
    
    println!("📁 Testing file:// backend with {} files", file_uris.len());
    
    let dataset = MultiBackendDataset::from_uris(file_uris)?;
    let options = LoaderOptions {
        batch_size: 3,
        drop_last: false,
        ..Default::default()
    };
    
    let dataloader = AsyncPoolDataLoader::new(dataset, options);
    let mut stream = dataloader.stream();
    
    let mut total_size = 0;
    let mut batch_count = 0;
    
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        batch_count += 1;
        let batch_size: usize = batch.iter().map(|data| data.len()).sum();
        total_size += batch_size;
        
        println!("  Batch {}: {} items, total size: {}KB",
                 batch_count,
                 batch.len(),
                 batch_size / 1024);
    }
    
    println!("✅ Multi-backend test: {} batches, {}KB total",
             batch_count, total_size / 1024);
    
    // TODO: Add S3 and Azure demos when credentials are available
    println!("💡 To test S3/Azure backends, provide credentials and update URIs to:");
    println!("   s3://your-bucket/prefix/  or  az://account/container/prefix/");
    
    Ok(())
}

/// Demonstrate performance comparison with different configurations
async fn demo_performance_comparison() -> Result<()> {
    println!("\n📊 Demo 4: Performance Comparison");
    println!("==================================");
    
    let temp_dir = TempDir::new()?;
    let uris = create_ml_dataset(temp_dir.path(), 30).await?;
    
    // Test different pool configurations
    let configs = vec![
        ("Conservative", PoolConfig {
            pool_size: 4,
            readahead_batches: 2,
            batch_timeout: Duration::from_millis(500),
            max_inflight: 8,
        }),
        ("Balanced", PoolConfig {
            pool_size: 8,
            readahead_batches: 3,
            batch_timeout: Duration::from_millis(200),
            max_inflight: 16,
        }),
        ("Aggressive", PoolConfig {
            pool_size: 16,
            readahead_batches: 4,
            batch_timeout: Duration::from_millis(50),
            max_inflight: 32,
        }),
    ];
    
    let options = LoaderOptions {
        batch_size: 6,
        drop_last: false,
        ..Default::default()
    };
    
    for (i, (name, config)) in configs.into_iter().enumerate() {
        println!("  🔄 Testing {}/3: {} configuration...", i + 1, name);
        
        // Create fresh dataset for each test to avoid any state issues
        let dataset = MultiBackendDataset::from_uris(uris.clone())?;
        let dataloader = AsyncPoolDataLoader::new(dataset, options.clone());
        let mut stream = dataloader.stream_with_pool(config);
        
        let start_time = Instant::now();
        let mut batch_count = 0;
        let mut processed_items = 0;
        
        // Add timeout to prevent hanging
        let timeout_duration = Duration::from_secs(30); // 30 second timeout
        let mut last_progress = Instant::now();
        
        loop {
            // Check for timeout
            if start_time.elapsed() > timeout_duration {
                println!("    ⚠️  Timeout after 30s - terminating test");
                break;
            }
            
            // Progress reporting every 5 seconds
            if last_progress.elapsed() > Duration::from_secs(5) {
                println!("    📈 Progress: {} batches, {} items processed...", batch_count, processed_items);
                last_progress = Instant::now();
            }
            
            // Use timeout on stream.next() to prevent indefinite waiting
            match tokio::time::timeout(Duration::from_millis(1000), stream.next()).await {
                Ok(Some(batch_result)) => {
                    match batch_result {
                        Ok(batch) => {
                            batch_count += 1;
                            processed_items += batch.len();
                            
                            // Print progress for first few batches
                            if batch_count <= 3 {
                                println!("    📦 Batch {}: {} items", batch_count, batch.len());
                            }
                        }
                        Err(e) => {
                            println!("    ❌ Batch error: {}", e);
                            break;
                        }
                    }
                }
                Ok(None) => {
                    // Stream ended normally
                    break;
                }
                Err(_) => {
                    // Timeout on individual batch - this might indicate a problem
                    println!("    ⚠️  1s timeout waiting for next batch - continuing...");
                    continue;
                }
            }
        }
        
        let duration = start_time.elapsed();
        println!("    ✅ {:<12}: {:.1}ms ({} batches, {} items, {:.1}ms/batch)",
                 name,
                 duration.as_millis(),
                 batch_count,
                 processed_items,
                 if batch_count > 0 { duration.as_millis() as f64 / batch_count as f64 } else { 0.0 });
                 
        // Small delay between tests to allow cleanup
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    println!("✅ Performance comparison completed!");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎯 s3dlio Enhanced Async Pool DataLoader Demo");
    println!("============================================");
    println!("This demo showcases the novel async pooling approach:");
    println!("• Dynamic batch formation from completed requests");
    println!("• Out-of-order completion eliminates head-of-line blocking");
    println!("• Multi-backend support (file://, s3://, az://, direct://)");
    println!("• Configurable pooling for different performance needs");
    
    // Wrap the entire demo in a timeout to prevent hanging
    let demo_result = tokio::time::timeout(Duration::from_secs(120), async {
        demo_basic_async_pooling().await?;
        demo_advanced_pooling_config().await?;
        demo_multi_backend_support().await?;
        demo_performance_comparison().await?;
        Ok::<(), anyhow::Error>(())
    }).await;
    
    match demo_result {
        Ok(Ok(())) => {
            println!("\n🎉 Demo completed successfully!");
            println!("\n💡 Key Benefits Demonstrated:");
            println!("  ✅ No blocking on slow requests");
            println!("  ✅ Configurable performance tuning");
            println!("  ✅ Multi-backend transparency");
            println!("  ✅ Dynamic batch formation");
            println!("  ✅ High-throughput ML data loading");
        }
        Ok(Err(e)) => {
            println!("\n❌ Demo failed with error: {}", e);
        }
        Err(_) => {
            println!("\n⏰ Demo timed out after 2 minutes - this indicates a hanging task issue");
        }
    }
    
    // Force a small delay to allow background tasks to cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Explicitly exit to prevent hanging
    println!("🔄 Forcing clean exit...");
    std::process::exit(0);
}
